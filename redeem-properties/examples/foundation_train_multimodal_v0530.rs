//! v0.53 frozen-v0.52 mobility distillation from the accepted v0.38 teacher.
//!
//! v0.52 recovered RT/MS2 but plateaued above the v0.38 mobility/CCS reference.  This trainer
//! freezes the complete v0.52-derived `student_v050.*` + `student_v051.*` parent representation,
//! ignores the failed `student_v052.*` mobility branch, and optimizes only `student_v053.*`.
//! A separately loaded frozen v0.38 model supplies TRAIN-only feature/prediction distillation
//! targets.  DEV remains the only selection partition; TRAIN-HOLDOUT and historical evaluation
//! partitions remain closed.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_ms2_loss, load_foundation_corpus, read_foundation_training_run_config,
    sample_foundation_validation_indices, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCollator, FoundationCollatorConfig,
    FoundationCorruptionConfig, FoundationFragmentContextBatchV0350,
    FoundationLearningRateSchedule, FoundationModificationSite, FoundationMs2LossConfig,
    FoundationOptimizerStep, FoundationPartition, FoundationRecordProvenance,
    FoundationRegressionNormalization, FoundationSamplePlan, FoundationSamplingConfig,
    FoundationScalarPhysicsBatchV0360, FoundationTargetNormalizationConfig,
    FoundationTrainingRecord, PeptideFoundationMultimodalV0350Config,
    PeptideFoundationMultimodalV0380Config, PeptideFoundationMultimodalV0380Model,
    PeptideFoundationV0510Config, PeptideFoundationV0530Config, PeptideFoundationV0530Model,
    RetentionTimeObjective, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0380,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0510, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0530,
    FOUNDATION_SCALAR_ROBUST_DELTA_V0360, FOUNDATION_V0500_STUDENT_NAMESPACE,
    FOUNDATION_V0510_STUDENT_NAMESPACE, FOUNDATION_V0510_TEACHER_SOURCE,
    FOUNDATION_V0530_STUDENT_NAMESPACE, FOUNDATION_V0530_TEACHER_DIM,
    FOUNDATION_V0530_TEACHER_SOURCE,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V053_VERSION: u32 = 530;
const V053_OBJECTIVE: &str = "v0530_v038_teacher_distilled_mobility";
const V052_PARENT_OBJECTIVE: &str = "v0520_mobility_aware_pair_representation";
const V052_PARENT_ARCHITECTURE: &str = "deep_pair_mobility_aware_representation_v0520";
const V052_PARENT_MOBILITY_NAMESPACE: &str = "student_v052";
const V038_PARENT_OBJECTIVE: &str = "v0380_mobility_native_trainable_ccs_representation";
const V053_MAX_STEPS_PER_EPOCH: usize = 1_536;
const V053_SMOKE_STEPS: usize = 16;
const V053_MAX_DEV_BATCHES: usize = 256;
const V053_SMOKE_DEV_BATCHES: usize = 4;
const V053_SMOKE_CCS_RECORDS: usize = 256;
const V053_LOSS_CALIBRATION_RECORDS: usize = 2_048;
const V053_SMOKE_LOSS_CALIBRATION_RECORDS: usize = 64;
const V053_MAX_GRADIENT_NORM: f64 = 1.0;
const V035_REFERENCE_DEV_BATCH_SIZE: usize = 64;
const V035_REFERENCE_SEED: u64 = 20_261_035;
const V053_EVAL_BATCH_SIZE: usize = 32;

// Frozen v0.38 TRAIN-only mobility supervision policy.
const V038_MOBILITY_MSE_WEIGHT: f64 = 0.25;
const V038_MOBILITY_ROBUST_WEIGHT: f64 = 1.0;
const V038_CCS_AUX_MSE_WEIGHT: f64 = 0.10;
const V038_CCS_AUX_ROBUST_WEIGHT: f64 = 0.35;
const V038_MIN_MOBILITY_LOSS_SCALE: f64 = 1.0e-5;
const V038_MIN_CCS_LOSS_SCALE: f64 = 1.0e-3;
const V038_SOURCE_SHRINKAGE: f64 = 256.0;
const V038_MIN_SOURCE_SHARED_IDENTITIES: usize = 20;
const V038_RELIABILITY_FLOOR: f64 = 0.35;
const V038_RELIABILITY_CEILING: f64 = 1.25;
const V038_SINGLETON_WEIGHT_SCALE: f64 = 0.65;
const V038_CONSENSUS_DISPERSION_SCALE: f64 = 0.010;
const V038_RAW_OBJECTIVE_WEIGHT: f64 = 0.65;
const V038_CONSENSUS_OBJECTIVE_WEIGHT: f64 = 0.35;

// Distillation is intentionally secondary to the actual TRAIN mobility/CCS target.
const V053_TEACHER_MOBILITY_WEIGHT: f64 = 0.50;
const V053_TEACHER_CCS_WEIGHT: f64 = 0.25;
const V053_TEACHER_RESIDUE_FEATURE_WEIGHT: f64 = 0.10;
const V053_TEACHER_POOLED_FEATURE_WEIGHT: f64 = 0.15;

// Frozen DEV references / promotion contract.
const V035_RT_DEV_MAE: f64 = 4.528_375;
const V038_RAW_CCS_DEV_MAE: f64 = 8.806_119_29;
const V038_CONSENSUS_CCS_DEV_MAE: f64 = 8.922_126_00;
const V035_MS2_COSINE_DEV: f64 = 0.904_061;
const V035_MS2_SPECTRAL_ANGLE_DEV: f64 = 0.747_700;
const V035_MS2_PEARSON_DEV: f64 = 0.684_548;
const V053_RT_TARGET_MAE: f64 = 4.3;
const V053_RAW_CCS_PREFERRED_MAE: f64 = 8.669;
const V053_MS2_REGRESSION_TOLERANCE: f64 = 0.003;

#[derive(Debug, Clone, Deserialize)]
struct V035ParentMetadata {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    completed_steps: usize,
    v0350_config: PeptideFoundationMultimodalV0350Config,
}

#[derive(Debug, Clone, Deserialize)]
struct V038ParentMetadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    supervision_fingerprint: String,
    parent_v0350_checkpoint: String,
    v0380_config: PeptideFoundationMultimodalV0380Config,
    completed_steps: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct V051ParentMetadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    mobility_supervision_fingerprint: String,
    parent_v0350_checkpoint: String,
    parent_v0500_checkpoint: String,
    teacher_source: String,
    backbone_namespace: String,
    specialist_namespace: String,
    v0510_config: PeptideFoundationV0510Config,
    completed_epochs: usize,
    completed_updates: usize,
    dev_objective: f64,
    smoke_mode: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct V052ParentMetadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    mobility_supervision_fingerprint: String,
    parent_v0350_checkpoint: String,
    parent_v0510_checkpoint: String,
    teacher_source: String,
    backbone_namespace: String,
    specialist_namespace: String,
    mobility_namespace: String,
    completed_epochs: usize,
    completed_updates: usize,
    dev_objective: f64,
    smoke_mode: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct PropertyMetrics {
    rt_mae_native: Option<f64>,
    rt_rmse_native: Option<f64>,
    ms2_loss: Option<f64>,
    ms2_pointwise_mse: Option<f64>,
    ms2_pointwise_mae: Option<f64>,
    ms2_mean_cosine: Option<f64>,
    ms2_mean_spectral_angle: Option<f64>,
    ms2_mean_pearson: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct DevMetrics {
    properties: PropertyMetrics,
    raw_ccs_mae: f64,
    consensus_ccs_mae: f64,
    objective: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V053Metadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    mobility_supervision_fingerprint: String,
    parent_v0350_checkpoint: String,
    parent_v0380_checkpoint: String,
    parent_v0510_checkpoint: String,
    parent_v0520_checkpoint: String,
    parent_v0520_completed_epochs: usize,
    parent_v0520_completed_updates: usize,
    parent_v0520_dev_objective: f64,
    teacher_source: String,
    backbone_namespace: String,
    specialist_namespace: String,
    mobility_namespace: String,
    v0530_config: PeptideFoundationV0530Config,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    max_epochs: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    dev_batches: usize,
    seed: u64,
    learning_rate: f64,
    mobility_loss_scale_native: f64,
    ccs_aux_loss_scale_native: f64,
    completed_epochs: usize,
    completed_updates: usize,
    dev_objective: f64,
    smoke_mode: bool,
}

#[derive(Debug, Clone)]
struct MobilityObservation {
    record_index: usize,
    source_id: String,
    target: f64,
}

#[derive(Debug, Clone)]
struct MobilityConsensusExample {
    representative_index: usize,
    target_mobility: f32,
    weight: f32,
    source_count: usize,
    identity_hash: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct AffineFit {
    intercept: f64,
    slope: f64,
}

#[derive(Debug, Clone, Default)]
struct SourceSupervision {
    source_id: String,
    raw_records: usize,
    shared_identities: usize,
    affine: AffineFit,
    residual_mae: f64,
    reliability: f64,
}

#[derive(Debug, Clone)]
struct MobilityConsensusSupervision {
    examples: Vec<MobilityConsensusExample>,
    source_supervision: BTreeMap<String, SourceSupervision>,
    raw_records: usize,
    multisource_examples: usize,
    singleton_examples: usize,
    mean_weight: f64,
    mean_abs_adjusted_delta_to_consensus: f64,
    fingerprint: u64,
}

#[derive(Debug, Clone, Copy)]
struct MobilityLossScales {
    mobility: f64,
    ccs: f64,
}

#[derive(Debug, Clone, Copy)]
struct UpdateDiagnostics {
    total: f64,
    primary: f64,
    auxiliary: f64,
    teacher_prediction: f64,
    teacher_features: f64,
}
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 7 || args.len() > 14 {
        anyhow::bail!(
            "usage: foundation_train_multimodal_v0530 RUN_V0260.yaml OUTPUT_DIR PARENT_V0350_CHECKPOINT PARENT_V0380_BEST PARENT_V0510_BEST PARENT_V0520_BEST [max_epochs=8] [batch_size=32] [patience=3] [min_delta=0.002] [seed=20261053] [learning_rate=2e-5] [mode=smoke|train|resume]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_v0350_checkpoint = PathBuf::from(&args[3]);
    let parent_v0380_checkpoint = PathBuf::from(&args[4]);
    let parent_v0510_checkpoint = PathBuf::from(&args[5]);
    let parent_v0520_checkpoint = PathBuf::from(&args[6]);
    let requested_max_epochs = parse_or(&args, 7, 8usize)?;
    let batch_size = parse_or(&args, 8, 32usize)?;
    let patience = parse_or(&args, 9, 3usize)?;
    let min_delta = parse_or(&args, 10, 0.002f64)?;
    let seed = parse_or(&args, 11, 20_261_053u64)?;
    let learning_rate = parse_or(&args, 12, 2.0e-5f64)?;
    let run_mode = args.get(13).map(String::as_str).unwrap_or("smoke");
    let (smoke_mode, resume_mode) = match run_mode {
        "smoke" => (true, false),
        "train" => (false, false),
        "resume" => (false, true),
        other => {
            anyhow::bail!("unsupported v0.53 run mode {other:?}; expected smoke, train, or resume")
        }
    };
    let max_epochs = if smoke_mode { 1 } else { requested_max_epochs };
    if max_epochs == 0 || batch_size < 2 || patience == 0 {
        anyhow::bail!("v0.53 requires max_epochs>0, batch_size>=2, and patience>0");
    }
    if !(min_delta >= 0.0 && min_delta.is_finite()) {
        anyhow::bail!("v0.53 min_delta must be finite and non-negative");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("v0.53 learning_rate must be positive and finite");
    }
    if resume_mode {
        for checkpoint in ["initial", "latest", "best"] {
            let directory = output_root.join(checkpoint);
            for name in [
                "model.safetensors",
                "optimizer.safetensors",
                "metadata.yaml",
            ] {
                if !directory.join(name).is_file() {
                    anyhow::bail!("v0.53 resume is missing {:?}", directory.join(name));
                }
            }
        }
    } else if output_root.exists() {
        anyhow::bail!("v0.53 output directory already exists: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.53 training requires a CUDA device")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let train_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .collect();
    let dev_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Validation)
        .map(|entry| entry.record_index)
        .collect();
    let holdout_record_count = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .count();

    let parent_v0350 = read_v035_metadata(&parent_v0350_checkpoint)?;
    if parent_v0350.version != 350
        || parent_v0350.objective
            != "v0350_trainable_forward_representation_context_conditioned_ms2"
        || parent_v0350.completed_steps == 0
    {
        anyhow::bail!("v0.53 requires the selected authoritative v0.35 checkpoint");
    }
    parent_v0350.v0350_config.validate()?;

    let parent_v0380 = read_v038_parent_metadata(&parent_v0380_checkpoint)?;
    if parent_v0380.version != 380
        || parent_v0380.objective != V038_PARENT_OBJECTIVE
        || parent_v0380.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0380
        || parent_v0380.completed_steps == 0
    {
        anyhow::bail!("v0.53 requires the selected completed v0.38 mobility checkpoint");
    }
    parent_v0380.v0380_config.validate()?;
    if parent_v0380.parent_v0350_checkpoint != parent_v0350_checkpoint.display().to_string() {
        anyhow::bail!("v0.53 v0.38 teacher does not descend from the supplied v0.35 checkpoint");
    }
    if parent_v0380.v0380_config.forward().model_dim != FOUNDATION_V0530_TEACHER_DIM {
        anyhow::bail!(
            "v0.53 requires a {}d v0.38 mobility teacher; observed {}",
            FOUNDATION_V0530_TEACHER_DIM,
            parent_v0380.v0380_config.forward().model_dim
        );
    }

    let parent_v0510 = read_v051_parent_metadata(&parent_v0510_checkpoint)?;
    if parent_v0510.version != 510
        || parent_v0510.objective != "v0510_teacher_bridged_deep_pair_specialists"
        || parent_v0510.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0510
        || parent_v0510.teacher_source != FOUNDATION_V0510_TEACHER_SOURCE
        || parent_v0510.backbone_namespace != FOUNDATION_V0500_STUDENT_NAMESPACE
        || parent_v0510.specialist_namespace != FOUNDATION_V0510_STUDENT_NAMESPACE
        || parent_v0510.completed_epochs == 0
        || parent_v0510.completed_updates == 0
        || parent_v0510.smoke_mode
    {
        anyhow::bail!("v0.53 requires the completed non-smoke selected v0.51 checkpoint");
    }
    parent_v0510.v0510_config.validate()?;
    if parent_v0510.parent_v0350_checkpoint != parent_v0350_checkpoint.display().to_string() {
        anyhow::bail!("v0.53 v0.51 parent does not descend from the supplied v0.35 checkpoint");
    }

    let parent_v0520 = read_v052_parent_metadata(&parent_v0520_checkpoint)?;
    if parent_v0520.version != 520
        || parent_v0520.objective != V052_PARENT_OBJECTIVE
        || parent_v0520.architecture != V052_PARENT_ARCHITECTURE
        || parent_v0520.teacher_source != FOUNDATION_V0510_TEACHER_SOURCE
        || parent_v0520.backbone_namespace != FOUNDATION_V0500_STUDENT_NAMESPACE
        || parent_v0520.specialist_namespace != FOUNDATION_V0510_STUDENT_NAMESPACE
        || parent_v0520.mobility_namespace != V052_PARENT_MOBILITY_NAMESPACE
        || parent_v0520.completed_epochs == 0
        || parent_v0520.completed_updates == 0
        || parent_v0520.smoke_mode
    {
        anyhow::bail!("v0.53 requires the completed non-smoke selected v0.52 checkpoint");
    }
    if parent_v0520.parent_v0350_checkpoint != parent_v0350_checkpoint.display().to_string()
        || parent_v0520.parent_v0510_checkpoint != parent_v0510_checkpoint.display().to_string()
    {
        anyhow::bail!("v0.53 v0.52 parent provenance differs from supplied ancestors");
    }

    let current_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let current_benchmark_fingerprint =
        format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    for (label, corpus_fp, benchmark_fp) in [
        (
            "v0.35",
            parent_v0350.corpus_fingerprint.as_str(),
            parent_v0350.benchmark_manifest_fingerprint.as_str(),
        ),
        (
            "v0.38",
            parent_v0380.corpus_fingerprint.as_str(),
            parent_v0380.benchmark_manifest_fingerprint.as_str(),
        ),
        (
            "v0.51",
            parent_v0510.corpus_fingerprint.as_str(),
            parent_v0510.benchmark_manifest_fingerprint.as_str(),
        ),
        (
            "v0.52",
            parent_v0520.corpus_fingerprint.as_str(),
            parent_v0520.benchmark_manifest_fingerprint.as_str(),
        ),
    ] {
        if corpus_fp != current_corpus_fingerprint || benchmark_fp != current_benchmark_fingerprint
        {
            anyhow::bail!(
                "v0.53 {label} parent fingerprint differs from the current prepared corpus"
            );
        }
    }

    let v0530_config = PeptideFoundationV0530Config::fixed(parent_v0510.v0510_config.clone())?;
    let forward_config = v0530_config.base_v0510.base_v0500.featurizer_config();
    let mut max_prepared_len = 0usize;
    for &index in train_indices.iter().chain(&dev_indices) {
        let length = corpus.records[index].peptidoform.sequence.chars().count();
        max_prepared_len = max_prepared_len.max(length);
        if length > v0530_config.base_v0510.base_v0500.max_sequence_len {
            anyhow::bail!(
                "v0.53 record {index} length {length} exceeds max_sequence_len={}",
                v0530_config.base_v0510.base_v0500.max_sequence_len
            );
        }
    }

    let mobility_train_indices = finite_mobility_ccs_indices(&corpus.records, &train_indices);
    let ccs_dev_indices_full = finite_mobility_ccs_indices(&corpus.records, &dev_indices);
    if mobility_train_indices.len() < batch_size || ccs_dev_indices_full.is_empty() {
        anyhow::bail!(
            "v0.53 requires mobility/CCS labels in TRAIN and DEV; observed train={} dev={}",
            mobility_train_indices.len(),
            ccs_dev_indices_full.len()
        );
    }
    let train_supervision = build_train_consensus_supervision(
        &corpus.records,
        &corpus.provenance,
        &mobility_train_indices,
    )?;
    if train_supervision.examples.len() < batch_size {
        anyhow::bail!("v0.53 consensus TRAIN has fewer examples than batch_size");
    }
    let current_mobility_supervision_fingerprint =
        format!("fnv1a64:{:016x}", train_supervision.fingerprint);
    if parent_v0380.supervision_fingerprint != current_mobility_supervision_fingerprint
        || parent_v0510.mobility_supervision_fingerprint != current_mobility_supervision_fingerprint
        || parent_v0520.mobility_supervision_fingerprint != current_mobility_supervision_fingerprint
    {
        anyhow::bail!("v0.53 mobility supervision fingerprint differs from a frozen parent");
    }
    let dev_consensus_full = build_partition_consensus_examples(
        &corpus.records,
        &corpus.provenance,
        &ccs_dev_indices_full,
        &train_supervision.source_supervision,
    )?;
    if dev_consensus_full.is_empty() {
        anyhow::bail!("v0.53 DEV mobility consensus set is empty");
    }

    let full_steps_per_epoch = (train_supervision.examples.len() / batch_size)
        .min(V053_MAX_STEPS_PER_EPOCH)
        .max(1);
    let steps_per_epoch = if smoke_mode {
        V053_SMOKE_STEPS.min(full_steps_per_epoch).max(1)
    } else {
        full_steps_per_epoch
    };
    let max_updates = max_epochs.saturating_mul(steps_per_epoch);

    let train_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_indices,
        &dev_indices,
    );
    let (dev_plan, dev_evaluation_batch_size, dev_cohort_policy) = if smoke_mode {
        let dev_batches = feasible_validation_batches(
            "dev_smoke",
            &train_sampling,
            &corpus.provenance,
            &dev_indices,
            batch_size,
            V053_SMOKE_DEV_BATCHES,
        )?;
        let mut dev_sampling = train_sampling.clone();
        dev_sampling.validation_steps = Some(dev_batches);
        (
            sample_foundation_validation_indices(
                &corpus.records,
                &corpus.provenance,
                &dev_indices,
                batch_size,
                seed ^ 0x5300_d3f0_1234_5678,
                &dev_sampling,
            )?,
            batch_size,
            "bounded_v0530_smoke_dev",
        )
    } else {
        let reference_batches = feasible_validation_batches(
            "v035_reference_dev",
            &train_sampling,
            &corpus.provenance,
            &dev_indices,
            V035_REFERENCE_DEV_BATCH_SIZE,
            V053_MAX_DEV_BATCHES,
        )?;
        let mut dev_sampling = train_sampling.clone();
        dev_sampling.validation_steps = Some(reference_batches);
        (
            sample_foundation_validation_indices(
                &corpus.records,
                &corpus.provenance,
                &dev_indices,
                V035_REFERENCE_DEV_BATCH_SIZE,
                V035_REFERENCE_SEED ^ 0x3d13_7f24_559c_81e7,
                &dev_sampling,
            )?,
            V053_EVAL_BATCH_SIZE,
            "v035_canonical_forward_dev_cohort",
        )
    };
    let ccs_dev_indices = if smoke_mode {
        deterministic_index_subset(
            &ccs_dev_indices_full,
            V053_SMOKE_CCS_RECORDS,
            seed ^ 0x5300_cc50_1111_2222,
        )
    } else {
        ccs_dev_indices_full.clone()
    };
    let dev_consensus = if smoke_mode {
        deterministic_consensus_subset(
            &dev_consensus_full,
            V053_SMOKE_CCS_RECORDS,
            seed ^ 0x5300_cc50_3333_4444,
        )
    } else {
        dev_consensus_full.clone()
    };

    let clean_collator = FoundationCollator::new(
        forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: parent_v0350.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let target_normalization = parent_v0350.target_normalization;
    let ms2_loss = parent_v0350.ms2_loss.validate()?;

    let mut student_varmap = VarMap::new();
    let student_vb = VarBuilder::from_varmap(&student_varmap, DType::F32, &device);
    let student = PeptideFoundationV0530Model::new(v0530_config.clone(), student_vb)?;
    let (loaded_parent_variables, fresh_v053_variables) = load_v0520_shared_parent_variables(
        &student_varmap,
        &parent_v0520_checkpoint.join("model.safetensors"),
        &device,
    )?;

    let mut teacher_varmap = VarMap::new();
    let teacher_vb = VarBuilder::from_varmap(&teacher_varmap, DType::F32, &device);
    let teacher =
        PeptideFoundationMultimodalV0380Model::new(parent_v0380.v0380_config.clone(), teacher_vb)?;
    teacher_varmap
        .load(parent_v0380_checkpoint.join("model.safetensors"))
        .with_context(|| {
            format!("failed to load frozen v0.38 teacher from {parent_v0380_checkpoint:?}")
        })?;

    let optimizer_prefix = format!("{FOUNDATION_V0530_STUDENT_NAMESPACE}.");
    let mut optimizer = FoundationAdamW::new_for_prefixes(
        &student_varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: run.trainer.weight_decay,
        },
        &[optimizer_prefix.as_str()],
    )?;
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 250u64.min(max_updates.saturating_sub(1) as u64),
        total_steps: max_updates.max(1) as u64,
        min_lr_ratio: 0.10,
    };

    let requested_calibration_records = if smoke_mode {
        V053_SMOKE_LOSS_CALIBRATION_RECORDS
    } else {
        V053_LOSS_CALIBRATION_RECORDS
    };
    let calibration_count = requested_calibration_records
        .min(train_supervision.examples.len())
        .max(batch_size);
    let calibration_order = deterministic_example_order(
        &train_supervision.examples,
        calibration_count,
        0,
        seed ^ 0x5300_4c4f_5353_4343,
    );
    let calibration_examples = calibration_order
        .iter()
        .map(|&index| train_supervision.examples[index].clone())
        .collect::<Vec<_>>();
    let loss_scales = calibrate_mobility_residual_scales_v0530(
        &student,
        &teacher,
        &clean_collator,
        &corpus.records,
        &calibration_examples,
        batch_size,
        &target_normalization,
        &device,
    )?;

    if !resume_mode {
        fs::create_dir_all(&output_root)?;
        write_mobility_supervision_summary(&output_root, &train_supervision)?;
    }

    print_sample_plan("dev", &dev_plan);

    let teacher_sentinel_records = dev_plan
        .indices
        .iter()
        .take(4)
        .map(|&index| corpus.records[index].clone())
        .collect::<Vec<_>>();
    let teacher_sentinel_initial = v038_teacher_sentinel_fingerprint(
        &teacher,
        &clean_collator,
        &teacher_sentinel_records,
        &device,
    )?;

    println!("v0530_version\tv0.53-v038-teacher-distilled-mobility");
    println!("objective\t{V053_OBJECTIVE}");
    println!("architecture\t{FOUNDATION_MULTIMODAL_ARCHITECTURE_V0530}");
    println!("device\t{device:?}");
    println!("run_mode\t{run_mode}");
    println!("backbone_namespace\t{FOUNDATION_V0500_STUDENT_NAMESPACE}");
    println!("specialist_namespace\t{FOUNDATION_V0510_STUDENT_NAMESPACE}");
    println!("mobility_namespace\t{FOUNDATION_V0530_STUDENT_NAMESPACE}");
    println!("teacher_source\t{FOUNDATION_V0530_TEACHER_SOURCE}");
    println!(
        "parent_v0350_checkpoint\t{}",
        parent_v0350_checkpoint.display()
    );
    println!(
        "parent_v0380_checkpoint\t{}",
        parent_v0380_checkpoint.display()
    );
    println!(
        "parent_v0510_checkpoint\t{}",
        parent_v0510_checkpoint.display()
    );
    println!(
        "parent_v0520_checkpoint\t{}",
        parent_v0520_checkpoint.display()
    );
    println!("parent_v0520_best_epoch\t{}", parent_v0520.completed_epochs);
    println!(
        "parent_v0520_best_update\t{}",
        parent_v0520.completed_updates
    );
    println!(
        "parent_v0520_dev_objective\t{:.8}",
        parent_v0520.dev_objective
    );
    println!("warm_start_shared_v0520_variables\t{loaded_parent_variables}");
    println!("warm_start_v0530_fresh_variables\t{fresh_v053_variables}");
    println!("teacher_update_policy\tfrozen_external_v0380_detached_outputs");
    println!("teacher_sentinel_initial\t{teacher_sentinel_initial:.8}");
    println!("teacher_dim\t{}", v0530_config.teacher_dim);
    println!(
        "mobility_teacher_bridge_layers\t{}",
        v0530_config.mobility_layers
    );
    println!(
        "mobility_teacher_bridge_heads\t{}",
        v0530_config.mobility_heads
    );
    println!(
        "mobility_teacher_bridge_ff_dim\t{}",
        v0530_config.mobility_ff_dim
    );
    println!("prepared_max_sequence_len\t{max_prepared_len}");
    println!("train_records\t{}", train_indices.len());
    println!("dev_records\t{}", dev_indices.len());
    println!("holdout_records_reserved_not_evaluated\t{holdout_record_count}");
    println!(
        "mobility_train_raw_records\t{}",
        mobility_train_indices.len()
    );
    println!(
        "mobility_train_consensus_examples\t{}",
        train_supervision.examples.len()
    );
    println!("dev_raw_ccs_records\t{}", ccs_dev_indices.len());
    println!("dev_consensus_examples\t{}", dev_consensus.len());
    println!("dev_property_cohort_policy\t{dev_cohort_policy}");
    println!("dev_property_records\t{}", dev_plan.indices.len());
    println!("dev_evaluation_batch_size\t{dev_evaluation_batch_size}");
    println!("steps_per_epoch\t{steps_per_epoch}");
    println!("max_epochs\t{max_epochs}");
    println!("max_optimizer_updates\t{max_updates}");
    println!("batch_size\t{batch_size}");
    println!("base_learning_rate\t{learning_rate}");
    println!("optimizer_scope\tstudent_v053.*_only");
    println!("frozen_parent_scope\tstudent_v050.*+student_v051.*");
    println!("discarded_parent_mobility_scope\tstudent_v052.*_not_loaded");
    println!("training_schedule\tmobility_adapter_only_v038_feature_and_prediction_distillation");
    println!("teacher_mobility_weight\t{V053_TEACHER_MOBILITY_WEIGHT}");
    println!("teacher_ccs_weight\t{V053_TEACHER_CCS_WEIGHT}");
    println!("teacher_residue_feature_weight\t{V053_TEACHER_RESIDUE_FEATURE_WEIGHT}");
    println!("teacher_pooled_feature_weight\t{V053_TEACHER_POOLED_FEATURE_WEIGHT}");
    println!("mobility_loss_scale_native\t{:.8}", loss_scales.mobility);
    println!("ccs_aux_loss_scale_native\t{:.8}", loss_scales.ccs);
    println!("dev_reference_rt_mae\t{V035_RT_DEV_MAE}");
    println!("dev_reference_raw_ccs_mae\t{V038_RAW_CCS_DEV_MAE}");
    println!("dev_reference_consensus_ccs_mae\t{V038_CONSENSUS_CCS_DEV_MAE}");
    println!("dev_reference_ms2_cosine\t{V035_MS2_COSINE_DEV}");
    println!("dev_reference_ms2_spectral_angle\t{V035_MS2_SPECTRAL_ANGLE_DEV}");
    println!("dev_reference_ms2_pearson\t{V035_MS2_PEARSON_DEV}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let initial_properties = evaluate_properties_v0530(
        &student,
        &clean_collator,
        &corpus.records,
        &dev_plan.indices,
        dev_evaluation_batch_size,
        &target_normalization,
        ms2_loss,
        &device,
    )?;
    let initial_raw_dev = evaluate_raw_ccs_indices_v0530(
        &student,
        &teacher,
        &clean_collator,
        &corpus.records,
        &ccs_dev_indices,
        batch_size,
        &target_normalization,
        &device,
    )?;
    let initial_consensus_dev = evaluate_consensus_targets_v0530(
        &student,
        &teacher,
        &clean_collator,
        &corpus.records,
        &dev_consensus,
        batch_size,
        &target_normalization,
        &device,
    )?;
    let initial_objective = combined_dev_objective(
        initial_raw_dev / V038_RAW_CCS_DEV_MAE,
        initial_consensus_dev / V038_CONSENSUS_CCS_DEV_MAE,
    );
    let initial_dev = DevMetrics {
        properties: initial_properties,
        raw_ccs_mae: initial_raw_dev,
        consensus_ccs_mae: initial_consensus_dev,
        objective: initial_objective,
    };
    print_dev_metrics_v0530("train_dev_initial", 0, initial_dev);

    let metadata_for =
        |completed_epochs: usize, completed_updates: usize, dev_objective: f64| V053Metadata {
            version: V053_VERSION,
            objective: V053_OBJECTIVE.into(),
            architecture: FOUNDATION_MULTIMODAL_ARCHITECTURE_V0530.into(),
            corpus_fingerprint: current_corpus_fingerprint.clone(),
            benchmark_manifest_fingerprint: current_benchmark_fingerprint.clone(),
            mobility_supervision_fingerprint: current_mobility_supervision_fingerprint.clone(),
            parent_v0350_checkpoint: parent_v0350_checkpoint.display().to_string(),
            parent_v0380_checkpoint: parent_v0380_checkpoint.display().to_string(),
            parent_v0510_checkpoint: parent_v0510_checkpoint.display().to_string(),
            parent_v0520_checkpoint: parent_v0520_checkpoint.display().to_string(),
            parent_v0520_completed_epochs: parent_v0520.completed_epochs,
            parent_v0520_completed_updates: parent_v0520.completed_updates,
            parent_v0520_dev_objective: parent_v0520.dev_objective,
            teacher_source: FOUNDATION_V0530_TEACHER_SOURCE.into(),
            backbone_namespace: FOUNDATION_V0500_STUDENT_NAMESPACE.into(),
            specialist_namespace: FOUNDATION_V0510_STUDENT_NAMESPACE.into(),
            mobility_namespace: FOUNDATION_V0530_STUDENT_NAMESPACE.into(),
            v0530_config: v0530_config.clone(),
            rt_objective: parent_v0350.rt_objective,
            target_normalization,
            ms2_loss,
            max_epochs,
            steps_per_epoch,
            batch_size,
            dev_batches: dev_plan.indices.len() / dev_evaluation_batch_size.max(1),
            seed,
            learning_rate,
            mobility_loss_scale_native: loss_scales.mobility,
            ccs_aux_loss_scale_native: loss_scales.ccs,
            completed_epochs,
            completed_updates,
            dev_objective,
            smoke_mode,
        };

    let mut start_epoch = 1usize;
    let mut global_step = 0usize;
    let mut best_epoch = 0usize;
    let mut best_step = 0usize;
    let mut best_objective = initial_objective;
    let mut stale_epochs = 0usize;
    if resume_mode {
        let latest_metadata = read_v053_metadata(&output_root.join("latest"))?;
        validate_v053_metadata(
            "latest",
            &latest_metadata,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &current_mobility_supervision_fingerprint,
            &parent_v0350_checkpoint,
            &parent_v0380_checkpoint,
            &parent_v0510_checkpoint,
            &parent_v0520_checkpoint,
            &v0530_config,
            batch_size,
            steps_per_epoch,
            max_epochs,
            seed,
            learning_rate,
        )?;
        let best_metadata = read_v053_metadata(&output_root.join("best"))?;
        validate_v053_metadata(
            "best",
            &best_metadata,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &current_mobility_supervision_fingerprint,
            &parent_v0350_checkpoint,
            &parent_v0380_checkpoint,
            &parent_v0510_checkpoint,
            &parent_v0520_checkpoint,
            &v0530_config,
            batch_size,
            steps_per_epoch,
            max_epochs,
            seed,
            learning_rate,
        )?;
        student_varmap.load(output_root.join("latest/model.safetensors"))?;
        optimizer.load_safetensors(output_root.join("latest/optimizer.safetensors"))?;
        start_epoch = latest_metadata.completed_epochs.saturating_add(1);
        global_step = latest_metadata.completed_updates;
        best_epoch = best_metadata.completed_epochs;
        best_step = best_metadata.completed_updates;
        best_objective = best_metadata.dev_objective;
        stale_epochs = latest_metadata
            .completed_epochs
            .saturating_sub(best_metadata.completed_epochs);
        println!(
            "v0530_resume\tlatest_epoch={}\tlatest_update={}\tbest_epoch={}\tbest_update={}\tbest_dev_objective={:.8}\tstale_epochs={}",
            latest_metadata.completed_epochs,
            latest_metadata.completed_updates,
            best_epoch,
            best_step,
            best_objective,
            stale_epochs
        );
    } else {
        save_checkpoint_v0530(
            &output_root.join("initial"),
            &student_varmap,
            &optimizer,
            &metadata_for(0, 0, initial_objective),
        )?;
        save_checkpoint_v0530(
            &output_root.join("best"),
            &student_varmap,
            &optimizer,
            &metadata_for(0, 0, initial_objective),
        )?;
    }

    let mut stopped_early = false;
    for epoch in start_epoch..=max_epochs {
        let needed = steps_per_epoch.saturating_mul(batch_size);
        let order = deterministic_example_order(
            &train_supervision.examples,
            needed,
            epoch as u64,
            seed ^ 0x5300_7a11_2233_4455,
        );
        println!(
            "v0530_epoch\tstage=start\tepoch={epoch}\tsteps={steps_per_epoch}\texamples={needed}"
        );
        for local_step in 0..steps_per_epoch {
            global_step += 1;
            let lr =
                lr_schedule.learning_rate(learning_rate, global_step.saturating_sub(1) as u64)?;
            optimizer.set_learning_rate(lr)?;
            let offset = local_step.saturating_mul(batch_size);
            let selected = order[offset..offset + batch_size]
                .iter()
                .map(|&example_index| train_supervision.examples[example_index].clone())
                .collect::<Vec<_>>();
            let (loss, diagnostics) = mobility_teacher_distillation_loss_v0530(
                &student,
                &teacher,
                &clean_collator,
                &corpus.records,
                &selected,
                &target_normalization,
                loss_scales,
                seed ^ (global_step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                &device,
            )?;
            let gradient_audit_step = if smoke_mode && global_step <= 2 {
                Some(global_step)
            } else {
                None
            };
            let update =
                backward_step_v0530(&loss, &mut optimizer, &student_varmap, gradient_audit_step)?;
            if global_step == 1 || global_step % 100 == 0 || local_step + 1 == steps_per_epoch {
                println!(
                    "v0530_train\tepoch={epoch}\tstep={global_step}\tepoch_step={}\tlr={:.8}\ttotal={:.6}\tprimary={:.6}\tauxiliary={:.6}\tteacher_prediction={:.6}\tteacher_features={:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                    local_step + 1,
                    update.learning_rate,
                    diagnostics.total,
                    diagnostics.primary,
                    diagnostics.auxiliary,
                    diagnostics.teacher_prediction,
                    diagnostics.teacher_features,
                    update.gradient_norm,
                    update.gradient_scale,
                );
            }
        }

        let teacher_sentinel = v038_teacher_sentinel_fingerprint(
            &teacher,
            &clean_collator,
            &teacher_sentinel_records,
            &device,
        )?;
        let teacher_delta = (teacher_sentinel - teacher_sentinel_initial).abs();
        if teacher_delta > 1.0e-5 * teacher_sentinel_initial.abs().max(1.0) {
            anyhow::bail!(
                "v0.53 frozen v0.38 teacher changed: initial={teacher_sentinel_initial:.8} current={teacher_sentinel:.8} delta={teacher_delta:.8}"
            );
        }
        println!(
            "teacher_freeze_audit\tepoch={epoch}\tstatus=PASS\tfingerprint={teacher_sentinel:.8}\tdelta={teacher_delta:.8}"
        );

        let properties = evaluate_properties_v0530(
            &student,
            &clean_collator,
            &corpus.records,
            &dev_plan.indices,
            dev_evaluation_batch_size,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        validate_property_invariance_v0530(properties, initial_properties)?;
        let raw_dev = evaluate_raw_ccs_indices_v0530(
            &student,
            &teacher,
            &clean_collator,
            &corpus.records,
            &ccs_dev_indices,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let consensus_dev = evaluate_consensus_targets_v0530(
            &student,
            &teacher,
            &clean_collator,
            &corpus.records,
            &dev_consensus,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let objective = combined_dev_objective(
            raw_dev / V038_RAW_CCS_DEV_MAE,
            consensus_dev / V038_CONSENSUS_CCS_DEV_MAE,
        );
        let dev = DevMetrics {
            properties,
            raw_ccs_mae: raw_dev,
            consensus_ccs_mae: consensus_dev,
            objective,
        };
        print_dev_metrics_v0530("train_dev", global_step, dev);
        let improved = best_objective - objective > min_delta;
        println!(
            "train_dev_objective\tepoch={epoch}\tupdate={global_step}\tvalue={objective:.8}\tprevious_best={best_objective:.8}\timproved={improved}"
        );
        save_checkpoint_v0530(
            &output_root.join("latest"),
            &student_varmap,
            &optimizer,
            &metadata_for(epoch, global_step, objective),
        )?;
        if improved {
            best_objective = objective;
            best_epoch = epoch;
            best_step = global_step;
            stale_epochs = 0;
            save_checkpoint_v0530(
                &output_root.join("best"),
                &student_varmap,
                &optimizer,
                &metadata_for(epoch, global_step, objective),
            )?;
            println!(
                "v0530_best_checkpoint\tepoch={best_epoch}\tupdate={best_step}\tdev_objective={best_objective:.8}"
            );
        } else {
            stale_epochs += 1;
        }
        println!(
            "v0530_epoch\tstage=complete\tepoch={epoch}\tupdate={global_step}\tstale_epochs={stale_epochs}"
        );
        if stale_epochs >= patience {
            stopped_early = true;
            println!(
                "v0530_early_stop\tepoch={epoch}\tupdate={global_step}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_update={best_step}\tbest_dev_objective={best_objective:.8}"
            );
            break;
        }
    }

    student_varmap.load(output_root.join("best/model.safetensors"))?;
    let best_properties = evaluate_properties_v0530(
        &student,
        &clean_collator,
        &corpus.records,
        &dev_plan.indices,
        dev_evaluation_batch_size,
        &target_normalization,
        ms2_loss,
        &device,
    )?;
    validate_property_invariance_v0530(best_properties, initial_properties)?;
    let best_raw_dev = evaluate_raw_ccs_indices_v0530(
        &student,
        &teacher,
        &clean_collator,
        &corpus.records,
        &ccs_dev_indices,
        batch_size,
        &target_normalization,
        &device,
    )?;
    let best_consensus_dev = evaluate_consensus_targets_v0530(
        &student,
        &teacher,
        &clean_collator,
        &corpus.records,
        &dev_consensus,
        batch_size,
        &target_normalization,
        &device,
    )?;
    let best_dev = DevMetrics {
        properties: best_properties,
        raw_ccs_mae: best_raw_dev,
        consensus_ccs_mae: best_consensus_dev,
        objective: combined_dev_objective(
            best_raw_dev / V038_RAW_CCS_DEV_MAE,
            best_consensus_dev / V038_CONSENSUS_CCS_DEV_MAE,
        ),
    };
    print_dev_metrics_v0530("best_train_dev", best_step, best_dev);
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!(
        "v0530_training_complete\tbest_epoch={best_epoch}\tbest_update={best_step}\tbest_dev_objective={:.8}\tstopped_early={stopped_early}\tsmoke_mode={smoke_mode}",
        best_dev.objective
    );
    print_material_gate_v0530(best_dev, smoke_mode);
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

fn finite_mobility_ccs_indices(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
) -> Vec<usize> {
    indices
        .iter()
        .copied()
        .filter(|&index| {
            let record = &records[index];
            record
                .ccs
                .is_some_and(|value| value.is_finite() && value > 0.0)
                && record
                    .context
                    .ion_mobility
                    .is_some_and(|value| value.is_finite() && value > 0.0)
                && record.context.charge.is_some_and(|value| value > 0)
                && record
                    .context
                    .precursor_mz
                    .is_some_and(|value| value.is_finite() && value > 0.0)
        })
        .collect()
}

fn peptidoform_charge_key(record: &FoundationTrainingRecord) -> String {
    let mut modifications = record
        .peptidoform
        .modifications
        .iter()
        .map(|modification| {
            let site = match modification.site {
                FoundationModificationSite::Residue(index) => format!("R{index}"),
                FoundationModificationSite::NTerm => "N".to_string(),
                FoundationModificationSite::CTerm => "C".to_string(),
            };
            format!(
                "{site}:{}:{:+.4}",
                modification.identity_label(),
                modification.mass_delta
            )
        })
        .collect::<Vec<_>>();
    modifications.sort();
    let charge = record
        .context
        .charge
        .map(|value| value.to_string())
        .unwrap_or_else(|| "missing".to_string());
    format!(
        "{}|z={}|{}",
        record.peptidoform.sequence,
        charge,
        modifications.join(";")
    )
}

fn collect_mobility_groups(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
) -> Result<BTreeMap<String, Vec<MobilityObservation>>> {
    let mut groups = BTreeMap::<String, Vec<MobilityObservation>>::new();
    for &index in indices {
        let record = records
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("v0.38 record index {index} outside corpus"))?;
        let Some(target) = record
            .context
            .ion_mobility
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("v0.38 missing provenance for record {index}"))?;
        groups
            .entry(peptidoform_charge_key(record))
            .or_default()
            .push(MobilityObservation {
                record_index: index,
                source_id: source.source_id.clone(),
                target: f64::from(target),
            });
    }
    Ok(groups)
}

fn source_means_with_representatives(
    observations: &[MobilityObservation],
) -> BTreeMap<String, (f64, usize, usize)> {
    let mut grouped = BTreeMap::<String, (f64, usize, usize)>::new();
    for observation in observations {
        let entry = grouped.entry(observation.source_id.clone()).or_insert((
            0.0,
            0,
            observation.record_index,
        ));
        entry.0 += observation.target;
        entry.1 += 1;
        entry.2 = entry.2.min(observation.record_index);
    }
    grouped
        .into_iter()
        .map(|(source, (sum, count, representative))| {
            (source, (sum / count as f64, count, representative))
        })
        .collect()
}

fn build_train_consensus_supervision(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
) -> Result<MobilityConsensusSupervision> {
    let groups = collect_mobility_groups(records, provenance, indices)?;
    let mut all_sources = BTreeSet::<String>::new();
    let mut raw_counts = BTreeMap::<String, usize>::new();
    let mut affine_examples = BTreeMap::<String, Vec<(f64, f64)>>::new();

    for observations in groups.values() {
        let source_means = source_means_with_representatives(observations);
        for (source, (_, count, _)) in &source_means {
            all_sources.insert(source.clone());
            *raw_counts.entry(source.clone()).or_insert(0) += *count;
        }
        if source_means.len() < 2 {
            continue;
        }
        let total = source_means
            .values()
            .map(|(value, _, _)| *value)
            .sum::<f64>();
        for (source, (source_mean, _, _)) in &source_means {
            let other_mean = (total - source_mean) / (source_means.len() - 1) as f64;
            affine_examples
                .entry(source.clone())
                .or_default()
                .push((*source_mean, other_mean));
        }
    }

    let mut source_supervision = BTreeMap::<String, SourceSupervision>::new();
    for source in all_sources {
        let examples = affine_examples
            .get(&source)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let (affine, shared_identities) = if examples.len() >= V038_MIN_SOURCE_SHARED_IDENTITIES {
            let raw = fit_xy_affine(examples)?;
            let blend = examples.len() as f64 / (examples.len() as f64 + V038_SOURCE_SHRINKAGE);
            (
                AffineFit {
                    intercept: (blend * raw.intercept).clamp(-0.08, 0.08),
                    slope: (1.0 + blend * (raw.slope - 1.0)).clamp(0.95, 1.05),
                },
                examples.len(),
            )
        } else {
            (
                AffineFit {
                    intercept: 0.0,
                    slope: 1.0,
                },
                examples.len(),
            )
        };
        source_supervision.insert(
            source.clone(),
            SourceSupervision {
                source_id: source.clone(),
                raw_records: raw_counts.get(&source).copied().unwrap_or(0),
                shared_identities,
                affine,
                residual_mae: 0.0,
                reliability: 1.0,
            },
        );
    }

    // Estimate source reliability after TRAIN-only affine alignment. Residuals
    // are measured against the median of the other aligned source views for
    // the same exact peptidoform+charge identity.
    let mut source_residual_sum = BTreeMap::<String, f64>::new();
    let mut source_residual_count = BTreeMap::<String, usize>::new();
    for observations in groups.values() {
        let source_means = source_means_with_representatives(observations);
        if source_means.len() < 2 {
            continue;
        }
        let adjusted = source_means
            .iter()
            .map(|(source, (value, _, _))| {
                let fit = source_supervision
                    .get(source)
                    .map(|entry| entry.affine)
                    .unwrap_or(AffineFit {
                        intercept: 0.0,
                        slope: 1.0,
                    });
                (source.clone(), fit.intercept + fit.slope * value)
            })
            .collect::<Vec<_>>();
        for (source, value) in &adjusted {
            let mut others = adjusted
                .iter()
                .filter(|(other, _)| other != source)
                .map(|(_, other_value)| *other_value)
                .collect::<Vec<_>>();
            others.sort_by(|a, b| a.total_cmp(b));
            let reference = quantile_sorted(&others, 0.5);
            *source_residual_sum.entry(source.clone()).or_insert(0.0) += (value - reference).abs();
            *source_residual_count.entry(source.clone()).or_insert(0) += 1;
        }
    }

    let mut residual_maes = Vec::<f64>::new();
    for supervision in source_supervision.values_mut() {
        let count = source_residual_count
            .get(&supervision.source_id)
            .copied()
            .unwrap_or(0);
        supervision.residual_mae = if count > 0 {
            source_residual_sum
                .get(&supervision.source_id)
                .copied()
                .unwrap_or(0.0)
                / count as f64
        } else {
            0.0
        };
        if count >= V038_MIN_SOURCE_SHARED_IDENTITIES && supervision.residual_mae.is_finite() {
            residual_maes.push(supervision.residual_mae);
        }
    }
    residual_maes.sort_by(|a, b| a.total_cmp(b));
    let global_residual_scale = if residual_maes.is_empty() {
        1.0
    } else {
        quantile_sorted(&residual_maes, 0.5).max(0.002)
    };
    for supervision in source_supervision.values_mut() {
        supervision.reliability =
            if supervision.shared_identities < V038_MIN_SOURCE_SHARED_IDENTITIES {
                0.60
            } else {
                (2.0 / (1.0 + supervision.residual_mae / global_residual_scale))
                    .clamp(V038_RELIABILITY_FLOOR, V038_RELIABILITY_CEILING)
            };
    }

    let (examples, multisource_examples, singleton_examples, mean_weight, mean_delta) =
        build_consensus_from_groups(&groups, &source_supervision)?;
    let fingerprint = consensus_fingerprint(&examples);
    Ok(MobilityConsensusSupervision {
        examples,
        source_supervision,
        raw_records: indices.len(),
        multisource_examples,
        singleton_examples,
        mean_weight,
        mean_abs_adjusted_delta_to_consensus: mean_delta,
        fingerprint,
    })
}

fn build_partition_consensus_examples(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
    source_supervision: &BTreeMap<String, SourceSupervision>,
) -> Result<Vec<MobilityConsensusExample>> {
    let groups = collect_mobility_groups(records, provenance, indices)?;
    let (examples, _, _, _, _) = build_consensus_from_groups(&groups, source_supervision)?;
    Ok(examples)
}

fn source_family(source_id: &str) -> String {
    if source_id.starts_with("pxd034128_") {
        "pxd034128".to_string()
    } else if source_id.starts_with("pxd058337_") {
        "pxd058337".to_string()
    } else {
        source_id.to_string()
    }
}

fn build_consensus_from_groups(
    groups: &BTreeMap<String, Vec<MobilityObservation>>,
    source_supervision: &BTreeMap<String, SourceSupervision>,
) -> Result<(Vec<MobilityConsensusExample>, usize, usize, f64, f64)> {
    let mut examples = Vec::<MobilityConsensusExample>::with_capacity(groups.len());
    let mut multisource_examples = 0usize;
    let mut singleton_examples = 0usize;
    let mut total_weight = 0.0f64;
    let mut total_abs_delta = 0.0f64;
    let mut total_abs_delta_count = 0usize;

    for (identity, observations) in groups {
        let source_means = source_means_with_representatives(observations);
        if source_means.is_empty() {
            continue;
        }

        // Align each concrete source view using TRAIN-only source fits, then
        // collapse highly overlapping exported views to one project-family
        // observation. PXD034128's four library views and PXD058337's nine
        // acquisition views therefore cannot out-vote independent sources.
        let mut family_views = BTreeMap::<String, Vec<(f64, f64, usize)>>::new();
        for (source, (value, _, representative)) in &source_means {
            let supervision = source_supervision.get(source);
            let affine = supervision.map(|entry| entry.affine).unwrap_or(AffineFit {
                intercept: 0.0,
                slope: 1.0,
            });
            let reliability = supervision.map(|entry| entry.reliability).unwrap_or(0.60);
            let adjusted = affine.intercept + affine.slope * value;
            family_views
                .entry(source_family(source))
                .or_default()
                .push((adjusted, reliability, *representative));
        }

        let mut adjusted = Vec::<(String, f64, f64, usize)>::new();
        for (family, mut views) in family_views {
            views.sort_by(|a, b| a.0.total_cmp(&b.0));
            let values = views.iter().map(|entry| entry.0).collect::<Vec<_>>();
            let family_center = quantile_sorted(&values, 0.5);
            let family_reliability =
                views.iter().map(|entry| entry.1).sum::<f64>() / views.len() as f64;
            let representative = views
                .iter()
                .min_by(|a, b| {
                    (a.0 - family_center)
                        .abs()
                        .total_cmp(&(b.0 - family_center).abs())
                        .then_with(|| a.2.cmp(&b.2))
                })
                .map(|entry| entry.2)
                .ok_or_else(|| anyhow::anyhow!("v0.38 family has no representative"))?;
            adjusted.push((family, family_center, family_reliability, representative));
        }

        adjusted.sort_by(|a, b| a.1.total_cmp(&b.1));
        let values = adjusted.iter().map(|entry| entry.1).collect::<Vec<_>>();
        let center = quantile_sorted(&values, 0.5);
        let mut absolute_center = values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>();
        absolute_center.sort_by(|a, b| a.total_cmp(b));
        let mad = quantile_sorted(&absolute_center, 0.5);
        let clip_radius = (3.0 * mad).max(0.005);
        let mut weighted_sum = 0.0f64;
        let mut reliability_sum = 0.0f64;
        for (_, value, reliability, _) in &adjusted {
            let clipped = value.clamp(center - clip_radius, center + clip_radius);
            weighted_sum += reliability * clipped;
            reliability_sum += reliability;
        }
        let consensus = if reliability_sum > 0.0 {
            weighted_sum / reliability_sum
        } else {
            center
        };
        let mut consensus_abs = adjusted
            .iter()
            .map(|(_, value, _, _)| (value - consensus).abs())
            .collect::<Vec<_>>();
        consensus_abs.sort_by(|a, b| a.total_cmp(b));
        let dispersion = quantile_sorted(&consensus_abs, 0.5);
        total_abs_delta += consensus_abs.iter().sum::<f64>();
        total_abs_delta_count += consensus_abs.len();

        let source_count = adjusted.len();
        let mean_reliability = adjusted
            .iter()
            .map(|(_, _, reliability, _)| *reliability)
            .sum::<f64>()
            / source_count as f64;
        let weight = if source_count == 1 {
            singleton_examples += 1;
            (V038_SINGLETON_WEIGHT_SCALE * mean_reliability).clamp(0.25, 0.80)
        } else {
            multisource_examples += 1;
            let dispersion_factor = 1.0 / (1.0 + dispersion / V038_CONSENSUS_DISPERSION_SCALE);
            let source_bonus = 1.0 + 0.12 * (source_count as f64).ln();
            (mean_reliability * dispersion_factor * source_bonus).clamp(0.35, 1.25)
        };
        let representative_index = adjusted
            .iter()
            .min_by(|a, b| {
                (a.1 - consensus)
                    .abs()
                    .total_cmp(&(b.1 - consensus).abs())
                    .then_with(|| a.3.cmp(&b.3))
            })
            .map(|entry| entry.3)
            .ok_or_else(|| anyhow::anyhow!("v0.38 consensus identity has no representative"))?;
        total_weight += weight;
        examples.push(MobilityConsensusExample {
            representative_index,
            target_mobility: consensus as f32,
            weight: weight as f32,
            source_count,
            identity_hash: stable_hash64(identity.as_bytes()),
        });
    }

    examples.sort_by_key(|example| (example.identity_hash, example.representative_index));
    let mean_weight = if examples.is_empty() {
        0.0
    } else {
        total_weight / examples.len() as f64
    };
    let mean_delta = if total_abs_delta_count == 0 {
        0.0
    } else {
        total_abs_delta / total_abs_delta_count as f64
    };
    Ok((
        examples,
        multisource_examples,
        singleton_examples,
        mean_weight,
        mean_delta,
    ))
}

fn fit_xy_affine(examples: &[(f64, f64)]) -> Result<AffineFit> {
    if examples.len() < 2 {
        return Ok(AffineFit {
            intercept: 0.0,
            slope: 1.0,
        });
    }
    let n = examples.len() as f64;
    let mean_x = examples.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = examples.iter().map(|(_, y)| y).sum::<f64>() / n;
    let mut covariance = 0.0f64;
    let mut variance = 0.0f64;
    for &(x, y) in examples {
        let dx = x - mean_x;
        covariance += dx * (y - mean_y);
        variance += dx * dx;
    }
    if variance <= 1.0e-12 {
        return Ok(AffineFit {
            intercept: mean_y - mean_x,
            slope: 1.0,
        });
    }
    let slope = covariance / variance;
    Ok(AffineFit {
        intercept: mean_y - slope * mean_x,
        slope,
    })
}

fn quantile_sorted(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let q = quantile.clamp(0.0, 1.0);
    let position = q * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - lower as f64;
        values[lower] * (1.0 - fraction) + values[upper] * fraction
    }
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn consensus_fingerprint(examples: &[MobilityConsensusExample]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for example in examples {
        for value in [
            example.representative_index as u64,
            u64::from(example.target_mobility.to_bits()),
            u64::from(example.weight.to_bits()),
            example.source_count as u64,
            example.identity_hash,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    hash
}

fn deterministic_example_order(
    examples: &[MobilityConsensusExample],
    needed: usize,
    epoch: u64,
    seed: u64,
) -> Vec<usize> {
    let mut output = Vec::<usize>::with_capacity(needed);
    let mut cycle = 0u64;
    while output.len() < needed {
        let mut indices = (0..examples.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&index| {
            mix64(
                examples[index].identity_hash
                    ^ seed
                    ^ epoch.wrapping_mul(0x9e3779b97f4a7c15)
                    ^ cycle.wrapping_mul(0xd1b54a32d192ed03),
            )
        });
        let take = (needed - output.len()).min(indices.len());
        output.extend_from_slice(&indices[..take]);
        cycle = cycle.wrapping_add(1);
    }
    output
}

fn write_mobility_supervision_summary(
    output_root: &Path,
    supervision: &MobilityConsensusSupervision,
) -> Result<()> {
    let mut source_lines = String::from(
        "source\traw_records\tshared_identities\taffine_intercept\taffine_slope\tresidual_mae\treliability\n",
    );
    for entry in supervision.source_supervision.values() {
        source_lines.push_str(&format!(
            "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\n",
            entry.source_id,
            entry.raw_records,
            entry.shared_identities,
            entry.affine.intercept,
            entry.affine.slope,
            entry.residual_mae,
            entry.reliability,
        ));
    }
    fs::write(
        output_root.join("mobility_source_reliability_v0530.tsv"),
        source_lines,
    )?;
    let summary = format!(
        "raw_train_mobility_records\t{}\nconsensus_train_examples\t{}\nmultisource_examples\t{}\nsingleton_examples\t{}\nmean_weight\t{:.8}\nmean_abs_adjusted_delta_to_consensus\t{:.8}\nsupervision_fingerprint\tfnv1a64:{:016x}\n",
        supervision.raw_records,
        supervision.examples.len(),
        supervision.multisource_examples,
        supervision.singleton_examples,
        supervision.mean_weight,
        supervision.mean_abs_adjusted_delta_to_consensus,
        supervision.fingerprint,
    );
    fs::write(
        output_root.join("mobility_consensus_supervision_v0530.tsv"),
        summary,
    )?;
    Ok(())
}

fn bruker_ccs_factor(
    context: &redeem_properties::foundation::PrecursorContextBatch,
) -> Result<Tensor> {
    let charge = context.charge.unsqueeze(1)?;
    let mz = context.precursor_mz.unsqueeze(1)?;
    let neutral_mass = charge.broadcast_mul(&mz)?;
    let reduced_mass = neutral_mass
        .affine(28.0, 0.0)?
        .broadcast_div(&neutral_mass.affine(1.0, 28.0)?)?;
    let denominator = reduced_mass.sqrt()?.clamp(1.0e-6, f64::INFINITY)?;
    Ok(charge
        .affine(1059.62245, 0.0)?
        .broadcast_div(&denominator)?)
}

fn predicted_native_mobility_and_ccs_v0530(
    student: &PeptideFoundationV0530Model,
    teacher: &PeptideFoundationMultimodalV0380Model,
    batch: &redeem_properties::foundation::FoundationTrainingBatch,
    records: &[FoundationTrainingRecord],
    normalization: &FoundationTargetNormalizationConfig,
    train: bool,
) -> Result<(Tensor, Tensor)> {
    let physics = FoundationScalarPhysicsBatchV0360::from_records(
        records,
        student.config().base_v0510.base_v0500.max_sequence_len,
        batch.input.residue_mask.device(),
    )?;
    let student_output =
        student.mobility_student_t(&batch.input, &batch.context, &physics, train)?;
    let teacher_output =
        teacher.mobility_teacher_features_v0380_t(&batch.input, &batch.context, &physics, false)?;
    let base_ccs_native = normalization
        .ccs
        .denormalize_tensor(&teacher_output.base_ccs_model.detach())?;
    let factor = bruker_ccs_factor(&batch.context)?;
    let base_mobility = base_ccs_native.broadcast_div(&factor)?;
    let predicted_mobility = (&base_mobility + &student_output.mobility_residual_native)?;
    let predicted_ccs = predicted_mobility.broadcast_mul(&factor)?;
    Ok((predicted_mobility, predicted_ccs))
}

fn mobility_teacher_distillation_loss_v0530(
    student: &PeptideFoundationV0530Model,
    teacher: &PeptideFoundationMultimodalV0380Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    normalization: &FoundationTargetNormalizationConfig,
    loss_scales: MobilityLossScales,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, UpdateDiagnostics)> {
    let owned = examples
        .iter()
        .map(|example| {
            records
                .get(example.representative_index)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("v0.53 representative index outside corpus"))
        })
        .collect::<Result<Vec<_>>>()?;
    let weights = examples
        .iter()
        .map(|example| example.weight)
        .collect::<Vec<_>>();
    let targets = examples
        .iter()
        .map(|example| example.target_mobility)
        .collect::<Vec<_>>();
    let batch = collator.collate(&owned, device, seed)?;
    let physics = FoundationScalarPhysicsBatchV0360::from_records(
        &owned,
        student.config().base_v0510.base_v0500.max_sequence_len,
        device,
    )?;
    let student_output =
        student.mobility_student_t(&batch.input, &batch.context, &physics, true)?;
    let teacher_output =
        teacher.mobility_teacher_features_v0380_t(&batch.input, &batch.context, &physics, false)?;
    let base_ccs_native = normalization
        .ccs
        .denormalize_tensor(&teacher_output.base_ccs_model.detach())?;
    let factor = bruker_ccs_factor(&batch.context)?;
    let base_mobility = base_ccs_native.broadcast_div(&factor)?;
    let predicted_mobility = (&base_mobility + &student_output.mobility_residual_native)?;
    let predicted_ccs = predicted_mobility.broadcast_mul(&factor)?;
    let teacher_mobility = (&base_mobility + &teacher_output.mobility_residual_native.detach())?;
    let teacher_ccs = teacher_mobility.broadcast_mul(&factor)?;

    let target_mobility = Tensor::from_vec(targets, (examples.len(), 1), device)?;
    let target_ccs = target_mobility.broadcast_mul(&factor)?;
    let weight = Tensor::from_vec(weights, (examples.len(), 1), device)?;
    let mask = Tensor::ones((examples.len(), 1), DType::F32, device)?;

    let mobility_mse = weighted_scaled_mse(
        &predicted_mobility,
        &target_mobility,
        &mask,
        &weight,
        loss_scales.mobility,
    )?;
    let mobility_robust = weighted_scaled_pseudo_huber(
        &predicted_mobility,
        &target_mobility,
        &mask,
        &weight,
        loss_scales.mobility,
        FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    )?;
    let ccs_mse =
        weighted_scaled_mse(&predicted_ccs, &target_ccs, &mask, &weight, loss_scales.ccs)?;
    let ccs_robust = weighted_scaled_pseudo_huber(
        &predicted_ccs,
        &target_ccs,
        &mask,
        &weight,
        loss_scales.ccs,
        FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    )?;
    let primary = (((mobility_mse.affine(V038_MOBILITY_MSE_WEIGHT, 0.0)?
        + mobility_robust.affine(V038_MOBILITY_ROBUST_WEIGHT, 0.0)?)?
        + ccs_mse.affine(V038_CCS_AUX_MSE_WEIGHT, 0.0)?)?
        + ccs_robust.affine(V038_CCS_AUX_ROBUST_WEIGHT, 0.0)?)?;

    let teacher_mobility_loss = weighted_scaled_mse(
        &predicted_mobility,
        &teacher_mobility.detach(),
        &mask,
        &weight,
        loss_scales.mobility,
    )?;
    let teacher_ccs_loss = weighted_scaled_mse(
        &predicted_ccs,
        &teacher_ccs.detach(),
        &mask,
        &weight,
        loss_scales.ccs,
    )?;
    let teacher_prediction = (teacher_mobility_loss.affine(V053_TEACHER_MOBILITY_WEIGHT, 0.0)?
        + teacher_ccs_loss.affine(V053_TEACHER_CCS_WEIGHT, 0.0)?)?;

    let residue_feature = normalized_masked_feature_mse_v0530(
        &student_output.residue_teacher_space,
        &teacher_output.context_residue_embeddings.detach(),
        &teacher_output.residue_mask,
    )?;
    let pooled_feature = normalized_feature_mse_v0530(
        &student_output.pooled_teacher_space,
        &teacher_output.context_pooled_embedding.detach(),
    )?;
    let teacher_features = (residue_feature.affine(V053_TEACHER_RESIDUE_FEATURE_WEIGHT, 0.0)?
        + pooled_feature.affine(V053_TEACHER_POOLED_FEATURE_WEIGHT, 0.0)?)?;
    let auxiliary = (&teacher_prediction + &teacher_features)?;
    let total = (&primary + &auxiliary)?;

    let total_value = f64::from(total.to_scalar::<f32>()?);
    let primary_value = f64::from(primary.to_scalar::<f32>()?);
    let teacher_prediction_value = f64::from(teacher_prediction.to_scalar::<f32>()?);
    let teacher_features_value = f64::from(teacher_features.to_scalar::<f32>()?);
    let auxiliary_value = f64::from(auxiliary.to_scalar::<f32>()?);
    if ![
        total_value,
        primary_value,
        auxiliary_value,
        teacher_prediction_value,
        teacher_features_value,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        anyhow::bail!("v0.53 mobility teacher-distillation loss is non-finite");
    }
    Ok((
        total,
        UpdateDiagnostics {
            total: total_value,
            primary: primary_value,
            auxiliary: auxiliary_value,
            teacher_prediction: teacher_prediction_value,
            teacher_features: teacher_features_value,
        },
    ))
}

fn calibrate_mobility_residual_scales_v0530(
    student: &PeptideFoundationV0530Model,
    teacher: &PeptideFoundationMultimodalV0380Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    batch_size: usize,
    normalization: &FoundationTargetNormalizationConfig,
    device: &Device,
) -> Result<MobilityLossScales> {
    let mut mobility_squared = 0.0f64;
    let mut ccs_squared = 0.0f64;
    let mut weight_sum = 0.0f64;
    for chunk in examples.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|example| records[example.representative_index].clone())
            .collect::<Vec<_>>();
        let weights = chunk
            .iter()
            .map(|example| example.weight)
            .collect::<Vec<_>>();
        let targets = chunk
            .iter()
            .map(|example| example.target_mobility)
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let (predicted_mobility, predicted_ccs) = predicted_native_mobility_and_ccs_v0530(
            student,
            teacher,
            &batch,
            &owned,
            normalization,
            false,
        )?;
        let target_mobility = Tensor::from_vec(targets, (chunk.len(), 1), device)?;
        let factor = bruker_ccs_factor(&batch.context)?;
        let target_ccs = target_mobility.broadcast_mul(&factor)?;
        let weight = Tensor::from_vec(weights, (chunk.len(), 1), device)?;
        let mobility_sq = (&predicted_mobility - &target_mobility)?
            .sqr()?
            .broadcast_mul(&weight)?
            .sum_all()?;
        let ccs_sq = (&predicted_ccs - &target_ccs)?
            .sqr()?
            .broadcast_mul(&weight)?
            .sum_all()?;
        let count = weight.sum_all()?;
        mobility_squared += f64::from(mobility_sq.to_scalar::<f32>()?);
        ccs_squared += f64::from(ccs_sq.to_scalar::<f32>()?);
        weight_sum += f64::from(count.to_scalar::<f32>()?);
    }
    if !(weight_sum > 0.0 && mobility_squared.is_finite() && ccs_squared.is_finite()) {
        anyhow::bail!("v0.53 TRAIN loss calibration has no finite weighted labels");
    }
    Ok(MobilityLossScales {
        mobility: (mobility_squared / weight_sum)
            .sqrt()
            .max(V038_MIN_MOBILITY_LOSS_SCALE),
        ccs: (ccs_squared / weight_sum)
            .sqrt()
            .max(V038_MIN_CCS_LOSS_SCALE),
    })
}

fn weighted_scaled_mse(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    weight: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!(
            "v0.38 weighted scalar loss shape mismatch: prediction {:?}, target {:?}",
            prediction.dims(),
            target.dims()
        );
    }
    if !(scale > 0.0 && scale.is_finite()) {
        anyhow::bail!("v0.38 invalid loss scale {scale}");
    }
    let combined = mask
        .broadcast_mul(weight)?
        .broadcast_as(prediction.dims())?;
    let scaled_error = (prediction - target)?.affine(1.0 / scale, 0.0)?;
    let numerator = scaled_error.sqr()?.broadcast_mul(&combined)?.sum_all()?;
    let denominator = combined.sum_all()?.clamp(1.0e-6, f64::INFINITY)?;
    Ok(numerator.broadcast_div(&denominator)?)
}

fn weighted_scaled_pseudo_huber(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    weight: &Tensor,
    scale: f64,
    delta: f64,
) -> Result<Tensor> {
    if !(scale > 0.0 && scale.is_finite() && delta > 0.0 && delta.is_finite()) {
        anyhow::bail!("v0.38 invalid robust loss scale={scale} delta={delta}");
    }
    let combined = mask
        .broadcast_mul(weight)?
        .broadcast_as(prediction.dims())?;
    let scaled = (prediction - target)?.affine(1.0 / (scale * delta), 0.0)?;
    let robust = (scaled.sqr()? + 1.0)?
        .sqrt()?
        .affine(delta * delta, -(delta * delta))?;
    let numerator = robust.broadcast_mul(&combined)?.sum_all()?;
    let denominator = combined.sum_all()?.clamp(1.0e-6, f64::INFINITY)?;
    Ok(numerator.broadcast_div(&denominator)?)
}

fn evaluate_raw_ccs_indices_v0530(
    student: &PeptideFoundationV0530Model,
    teacher: &PeptideFoundationMultimodalV0380Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    normalization: &FoundationTargetNormalizationConfig,
    device: &Device,
) -> Result<f64> {
    let mut absolute_error = 0.0f64;
    let mut count = 0usize;
    for chunk in indices.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|&index| records[index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let (_, predicted_ccs) = predicted_native_mobility_and_ccs_v0530(
            student,
            teacher,
            &batch,
            &owned,
            normalization,
            false,
        )?;
        let predicted = predicted_ccs.to_vec2::<f32>()?;
        for (row, record) in owned.iter().enumerate() {
            let Some(target) = record.ccs.filter(|value| value.is_finite()) else {
                continue;
            };
            absolute_error += f64::from((predicted[row][0] - target).abs());
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.53 raw CCS evaluation contains no labels");
    }
    Ok(absolute_error / count as f64)
}

fn evaluate_consensus_targets_v0530(
    student: &PeptideFoundationV0530Model,
    teacher: &PeptideFoundationMultimodalV0380Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    batch_size: usize,
    normalization: &FoundationTargetNormalizationConfig,
    device: &Device,
) -> Result<f64> {
    let mut absolute_error = 0.0f64;
    let mut count = 0usize;
    for chunk in examples.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|example| records[example.representative_index].clone())
            .collect::<Vec<_>>();
        let targets = chunk
            .iter()
            .map(|example| example.target_mobility)
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let (_, predicted_ccs) = predicted_native_mobility_and_ccs_v0530(
            student,
            teacher,
            &batch,
            &owned,
            normalization,
            false,
        )?;
        let target_mobility = Tensor::from_vec(targets, (chunk.len(), 1), device)?;
        let factor = bruker_ccs_factor(&batch.context)?;
        let target_ccs = target_mobility.broadcast_mul(&factor)?;
        let prediction = predicted_ccs.to_vec2::<f32>()?;
        let target = target_ccs.to_vec2::<f32>()?;
        for row in 0..prediction.len() {
            absolute_error += f64::from((prediction[row][0] - target[row][0]).abs());
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.53 consensus CCS evaluation contains no labels");
    }
    Ok(absolute_error / count as f64)
}

fn combined_dev_objective(raw_ratio: f64, consensus_ratio: f64) -> f64 {
    V038_RAW_OBJECTIVE_WEIGHT * raw_ratio + V038_CONSENSUS_OBJECTIVE_WEIGHT * consensus_ratio
}

fn accumulate_regression(
    prediction: &Tensor,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    normalization: &FoundationRegressionNormalization,
    abs_sum: &mut f64,
    sq_sum: &mut f64,
    count: &mut usize,
) -> Result<()> {
    let (Some(target), Some(mask)) = (target, mask) else {
        return Ok(());
    };
    let prediction = normalization
        .denormalize_tensor(prediction)?
        .to_vec2::<f32>()?;
    let target = normalization.denormalize_tensor(target)?.to_vec2::<f32>()?;
    let mask = mask.to_vec2::<f32>()?;
    for row in 0..prediction.len() {
        if mask[row][0] <= 0.0 {
            continue;
        }
        let error = f64::from(prediction[row][0] - target[row][0]);
        *abs_sum += error.abs();
        *sq_sum += error * error;
        *count += 1;
    }
    Ok(())
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "NA".into())
}

#[derive(Debug, Default)]
struct Ms2ShapeAccumulator {
    fragment_count: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    spectrum_count: usize,
    cosine_sum: f64,
    spectral_angle_sum: f64,
    pearson_count: usize,
    pearson_sum: f64,
}

impl Ms2ShapeAccumulator {
    fn accumulate(&mut self, prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<()> {
        let predicted = prediction.to_vec3::<f32>()?;
        let targets = target.to_vec3::<f32>()?;
        let masks = mask.to_vec3::<f32>()?;
        for batch_index in 0..predicted.len() {
            let mut pred_values = Vec::new();
            let mut target_values = Vec::new();
            for residue_index in 0..predicted[batch_index].len() {
                for channel_index in 0..predicted[batch_index][residue_index].len() {
                    if masks[batch_index][residue_index][channel_index] <= 0.0 {
                        continue;
                    }
                    let pred = f64::from(predicted[batch_index][residue_index][channel_index]);
                    let truth = f64::from(targets[batch_index][residue_index][channel_index]);
                    let error = pred - truth;
                    self.fragment_count += 1;
                    self.squared_error_sum += error * error;
                    self.absolute_error_sum += error.abs();
                    pred_values.push(pred);
                    target_values.push(truth);
                }
            }
            if pred_values.is_empty() {
                continue;
            }
            let dot = pred_values
                .iter()
                .zip(&target_values)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let pnorm = pred_values.iter().map(|v| v * v).sum::<f64>().sqrt();
            let tnorm = target_values.iter().map(|v| v * v).sum::<f64>().sqrt();
            let cosine = if pnorm > 0.0 && tnorm > 0.0 {
                (dot / (pnorm * tnorm)).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            self.spectrum_count += 1;
            self.cosine_sum += cosine;
            self.spectral_angle_sum += 1.0 - (2.0 / std::f64::consts::PI) * cosine.acos();
            if let Some(pearson) = pearson_correlation(&pred_values, &target_values) {
                self.pearson_count += 1;
                self.pearson_sum += pearson;
            }
        }
        Ok(())
    }

    fn pointwise_mse(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.squared_error_sum / self.fragment_count as f64)
    }
    fn pointwise_mae(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.absolute_error_sum / self.fragment_count as f64)
    }
    fn mean_cosine(&self) -> Option<f64> {
        (self.spectrum_count > 0).then(|| self.cosine_sum / self.spectrum_count as f64)
    }
    fn mean_spectral_angle(&self) -> Option<f64> {
        (self.spectrum_count > 0).then(|| self.spectral_angle_sum / self.spectrum_count as f64)
    }
    fn mean_pearson(&self) -> Option<f64> {
        (self.pearson_count > 0).then(|| self.pearson_sum / self.pearson_count as f64)
    }
}

fn pearson_correlation(first: &[f64], second: &[f64]) -> Option<f64> {
    if first.len() != second.len() || first.len() < 2 {
        return None;
    }
    let n = first.len() as f64;
    let first_mean = first.iter().sum::<f64>() / n;
    let second_mean = second.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut first_squared = 0.0;
    let mut second_squared = 0.0;
    for (&a, &b) in first.iter().zip(second) {
        let da = a - first_mean;
        let db = b - second_mean;
        numerator += da * db;
        first_squared += da * da;
        second_squared += db * db;
    }
    let denominator = (first_squared * second_squared).sqrt();
    (denominator > 1.0e-12).then(|| (numerator / denominator).clamp(-1.0, 1.0))
}

#[allow(clippy::too_many_arguments)]

fn normalize_rt_target(
    targets: &mut redeem_properties::foundation::FoundationTargets,
    normalization: &FoundationTargetNormalizationConfig,
) -> Result<()> {
    if let Some(rt) = targets.rt.take() {
        targets.rt = Some(normalization.rt.normalize_tensor(&rt)?);
    }
    Ok(())
}

fn normalized_masked_feature_mse_v0530(
    student: &Tensor,
    teacher: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    if student.dims() != teacher.dims() {
        anyhow::bail!(
            "v0.53 residue feature-distillation shape mismatch: student {:?}, teacher {:?}",
            student.dims(),
            teacher.dims()
        );
    }
    let (batch, length, dim) = student.dims3()?;
    let (mask_batch, mask_length) = mask.dims2()?;
    if mask_batch != batch || mask_length != length {
        anyhow::bail!(
            "v0.53 residue feature mask shape mismatch: mask {:?}, expected [{batch}, {length}]",
            mask.dims()
        );
    }
    let epsilon = 1.0e-6;
    let student_norm = (student.sqr()?.sum_keepdim(2)? + epsilon)?.sqrt()?;
    let teacher_norm = (teacher.sqr()?.sum_keepdim(2)? + epsilon)?.sqrt()?;
    let student_unit = student.broadcast_div(&student_norm)?;
    let teacher_unit = teacher.broadcast_div(&teacher_norm)?;
    let expanded_mask = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let numerator = (student_unit - teacher_unit)?
        .sqr()?
        .broadcast_mul(&expanded_mask)?
        .sum_all()?;
    let denominator = expanded_mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    Ok(numerator.broadcast_div(&denominator)?)
}

fn normalized_feature_mse_v0530(student: &Tensor, teacher: &Tensor) -> Result<Tensor> {
    if student.dims() != teacher.dims() {
        anyhow::bail!(
            "v0.53 pooled feature-distillation shape mismatch: student {:?}, teacher {:?}",
            student.dims(),
            teacher.dims()
        );
    }
    let (_, dim) = student.dims2()?;
    if dim == 0 {
        anyhow::bail!("v0.53 pooled feature width must be non-zero");
    }
    let epsilon = 1.0e-6;
    let student_norm = (student.sqr()?.sum_keepdim(1)? + epsilon)?.sqrt()?;
    let teacher_norm = (teacher.sqr()?.sum_keepdim(1)? + epsilon)?.sqrt()?;
    let student_unit = student.broadcast_div(&student_norm)?;
    let teacher_unit = teacher.broadcast_div(&teacher_norm)?;
    Ok((student_unit - teacher_unit)?.sqr()?.mean_all()?)
}

fn backward_step_v0530(
    loss: &Tensor,
    optimizer: &mut FoundationAdamW,
    varmap: &VarMap,
    gradient_audit_step: Option<usize>,
) -> Result<FoundationOptimizerStep> {
    let gradients = loss.backward()?;
    if let Some(audit_step) = gradient_audit_step {
        audit_gradients_v0530(varmap, &gradients, audit_step)?;
    }
    Ok(optimizer.step(&gradients, Some(V053_MAX_GRADIENT_NORM))?)
}

fn required_gradient_parameters_v0530(audit_step: usize) -> &'static [&'static str] {
    // The scalar mobility output is exactly zero-initialized to preserve the frozen-parent
    // prediction at step 0.  On the first backward pass that zero matrix necessarily blocks
    // scalar-loss gradients from reaching hidden/bottleneck/pair-only parameters.  Direct
    // feature-distillation losses still reach the residue/context/transformer path immediately.
    // After optimizer step 1, output.weight is non-zero; step 2 can then verify the scalar-only
    // branch all the way through pair_projection without weakening the zero-init contract.
    match audit_step {
        1 => &[
            "student_v053.mobility_teacher_bridge.output.weight",
            "student_v053.mobility_teacher_bridge.residue_projection.weight",
            "student_v053.mobility_teacher_bridge.context_projection.weight",
            "student_v053.mobility_teacher_bridge.transformer.0.attention.query.weight",
        ],
        2 => &[
            "student_v053.mobility_teacher_bridge.output.weight",
            "student_v053.mobility_teacher_bridge.hidden.weight",
            "student_v053.mobility_teacher_bridge.bottleneck.weight",
            "student_v053.mobility_teacher_bridge.pair_projection.weight",
        ],
        _ => &[],
    }
}

fn audit_gradients_v0530(
    varmap: &VarMap,
    gradients: &candle_core::backprop::GradStore,
    audit_step: usize,
) -> Result<()> {
    let required = required_gradient_parameters_v0530(audit_step);
    println!(
        "v0530_gradient_audit_stage\tstep={audit_step}\trequired_parameters={}",
        required.len()
    );
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.53 VarMap lock poisoned during gradient audit"))?;
    for &name in required {
        let variable = data
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("v0.53 gradient audit missing parameter {name}"))?;
        let gradient = gradients
            .get(variable)
            .ok_or_else(|| anyhow::anyhow!("v0.53 gradient missing for {name}"))?;
        let norm2 = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !norm2.is_finite() || norm2 <= 0.0 {
            anyhow::bail!("v0.53 gradient invalid for {name}: norm2={norm2}");
        }
        println!(
            "v0530_gradient_audit\tparameter={name}\tnorm={:.8}",
            norm2.sqrt()
        );
    }
    Ok(())
}

fn evaluate_properties_v0530(
    model: &PeptideFoundationV0530Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    normalization: &FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    device: &Device,
) -> Result<PropertyMetrics> {
    let mut rt_abs = 0.0f64;
    let mut rt_sq = 0.0f64;
    let mut rt_n = 0usize;
    let mut ms2_objective_sum = 0.0f64;
    let mut ms2_objective_batches = 0usize;
    let mut ms2_shape = Ms2ShapeAccumulator::default();

    for chunk in indices.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|&index| records[index].clone())
            .collect::<Vec<_>>();
        let mut batch = collator.collate(&owned, device, 0)?;
        normalize_rt_target(&mut batch.targets, normalization)?;
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &owned,
            model.config().base_v0510.base_v0500.max_sequence_len,
            device,
        )?;
        let fragment = FoundationFragmentContextBatchV0350::from_records(
            &owned,
            &model.config().base_v0510.base_v0500.featurizer_config(),
            device,
        )?;
        let output = model.property_forward_t(&batch.input, &batch.context, &physics, &fragment)?;
        accumulate_regression(
            &output.rt,
            batch.targets.rt.as_ref(),
            batch.targets.rt_mask.as_ref(),
            &normalization.rt,
            &mut rt_abs,
            &mut rt_sq,
            &mut rt_n,
        )?;
        if let (Some(target), Some(mask)) = (&batch.targets.ms2, &batch.targets.ms2_mask) {
            let theoretical = fragment.channel_mask()?;
            let effective_mask = mask.broadcast_mul(&theoretical)?;
            let components = foundation_ms2_loss(&output.ms2, target, &effective_mask, ms2_loss)?;
            ms2_objective_sum += f64::from(components.total.to_scalar::<f32>()?);
            ms2_objective_batches += 1;
            ms2_shape.accumulate(&output.ms2, target, &effective_mask)?;
        }
    }

    Ok(PropertyMetrics {
        rt_mae_native: (rt_n > 0).then(|| rt_abs / rt_n as f64),
        rt_rmse_native: (rt_n > 0).then(|| (rt_sq / rt_n as f64).sqrt()),
        ms2_loss: (ms2_objective_batches > 0)
            .then(|| ms2_objective_sum / ms2_objective_batches as f64),
        ms2_pointwise_mse: ms2_shape.pointwise_mse(),
        ms2_pointwise_mae: ms2_shape.pointwise_mae(),
        ms2_mean_cosine: ms2_shape.mean_cosine(),
        ms2_mean_spectral_angle: ms2_shape.mean_spectral_angle(),
        ms2_mean_pearson: ms2_shape.mean_pearson(),
    })
}

fn validate_property_invariance_v0530(
    observed: PropertyMetrics,
    reference: PropertyMetrics,
) -> Result<()> {
    fn check(label: &str, observed: Option<f64>, reference: Option<f64>) -> Result<()> {
        match (observed, reference) {
            (Some(a), Some(b)) => {
                let tolerance = 5.0e-5_f64.max(5.0e-6 * b.abs().max(1.0));
                if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance {
                    anyhow::bail!(
                        "v0.53 frozen property-path invariance failed for {label}: initial={b:.8} current={a:.8} tolerance={tolerance:.8}"
                    );
                }
            }
            (None, None) => {}
            _ => anyhow::bail!(
                "v0.53 frozen property-path invariance changed availability for {label}"
            ),
        }
        Ok(())
    }
    check(
        "rt_mae_native",
        observed.rt_mae_native,
        reference.rt_mae_native,
    )?;
    check(
        "rt_rmse_native",
        observed.rt_rmse_native,
        reference.rt_rmse_native,
    )?;
    check("ms2_loss", observed.ms2_loss, reference.ms2_loss)?;
    check(
        "ms2_pointwise_mse",
        observed.ms2_pointwise_mse,
        reference.ms2_pointwise_mse,
    )?;
    check(
        "ms2_pointwise_mae",
        observed.ms2_pointwise_mae,
        reference.ms2_pointwise_mae,
    )?;
    check(
        "ms2_cosine",
        observed.ms2_mean_cosine,
        reference.ms2_mean_cosine,
    )?;
    check(
        "ms2_spectral_angle",
        observed.ms2_mean_spectral_angle,
        reference.ms2_mean_spectral_angle,
    )?;
    check(
        "ms2_pearson",
        observed.ms2_mean_pearson,
        reference.ms2_mean_pearson,
    )?;
    Ok(())
}

fn print_dev_metrics_v0530(label: &str, update: usize, metrics: DevMetrics) {
    let p = metrics.properties;
    println!(
        "{label}\tupdate={update}\trt_mae_native={}\trt_rmse_native={}\traw_ccs_mae={:.8}\tconsensus_ccs_mae={:.8}\tms2_loss={}\tms2_pointwise_mse={}\tms2_pointwise_mae={}\tms2_cosine={}\tms2_spectral_angle={}\tms2_pearson={}\tdev_objective={:.8}",
        fmt_opt(p.rt_mae_native),
        fmt_opt(p.rt_rmse_native),
        metrics.raw_ccs_mae,
        metrics.consensus_ccs_mae,
        fmt_opt(p.ms2_loss),
        fmt_opt(p.ms2_pointwise_mse),
        fmt_opt(p.ms2_pointwise_mae),
        fmt_opt(p.ms2_mean_cosine),
        fmt_opt(p.ms2_mean_spectral_angle),
        fmt_opt(p.ms2_mean_pearson),
        metrics.objective,
    );
}

fn print_material_gate_v0530(metrics: DevMetrics, smoke_mode: bool) {
    let p = metrics.properties;
    let rt = p.rt_mae_native.unwrap_or(f64::INFINITY);
    let cosine = p.ms2_mean_cosine.unwrap_or(f64::NEG_INFINITY);
    let angle = p.ms2_mean_spectral_angle.unwrap_or(f64::NEG_INFINITY);
    let pearson = p.ms2_mean_pearson.unwrap_or(f64::NEG_INFINITY);
    let rt_target = rt <= V053_RT_TARGET_MAE;
    let raw_beats_v038 = metrics.raw_ccs_mae < V038_RAW_CCS_DEV_MAE;
    let raw_preferred = metrics.raw_ccs_mae <= V053_RAW_CCS_PREFERRED_MAE;
    let consensus_beats_v038 = metrics.consensus_ccs_mae < V038_CONSENSUS_CCS_DEV_MAE;
    let ms2_no_regression = cosine + V053_MS2_REGRESSION_TOLERANCE >= V035_MS2_COSINE_DEV
        && angle + V053_MS2_REGRESSION_TOLERANCE >= V035_MS2_SPECTRAL_ANGLE_DEV
        && pearson + V053_MS2_REGRESSION_TOLERANCE >= V035_MS2_PEARSON_DEV;
    // Require the preferred raw-CCS margin before opening HOLDOUT.  v0.53 is specifically a
    // mobility rescue lane, so merely edging the old v0.38 result is not enough evidence.
    let convincing_joint = rt_target && raw_preferred && consensus_beats_v038 && ms2_no_regression;
    println!("v0530_rt_target_met\t{}", yes_no(rt_target));
    println!("v0530_raw_ccs_beats_v038\t{}", yes_no(raw_beats_v038));
    println!(
        "v0530_raw_ccs_preferred_10pct_gate_met\t{}",
        yes_no(raw_preferred)
    );
    println!(
        "v0530_consensus_ccs_beats_v038\t{}",
        yes_no(consensus_beats_v038)
    );
    println!(
        "v0530_ms2_no_material_regression\t{}",
        yes_no(ms2_no_regression)
    );
    println!(
        "v0530_convincing_joint_dev_gain\t{}",
        yes_no(convincing_joint)
    );
    println!(
        "v0530_holdout_eligible\t{}",
        yes_no(convincing_joint && !smoke_mode)
    );
    println!("v0530_finalize_required\tNO_explicit_handoff_review_required");
}

fn v038_teacher_sentinel_fingerprint(
    teacher: &PeptideFoundationMultimodalV0380Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    device: &Device,
) -> Result<f64> {
    if records.is_empty() {
        anyhow::bail!("v0.53 teacher sentinel requires at least one record");
    }
    let batch = collator.collate(records, device, 0)?;
    let physics = FoundationScalarPhysicsBatchV0360::from_records(
        records,
        teacher.config().forward().max_sequence_len,
        device,
    )?;
    let output =
        teacher.mobility_teacher_features_v0380_t(&batch.input, &batch.context, &physics, false)?;
    let base = f64::from(output.base_ccs_model.sum_all()?.to_scalar::<f32>()?);
    let residual = f64::from(
        output
            .mobility_residual_native
            .sum_all()?
            .to_scalar::<f32>()?,
    );
    let residue = f64::from(
        output
            .context_residue_embeddings
            .sum_all()?
            .to_scalar::<f32>()?,
    );
    let pooled = f64::from(
        output
            .context_pooled_embedding
            .sum_all()?
            .to_scalar::<f32>()?,
    );
    let fingerprint = base + residual + residue + pooled;
    if !fingerprint.is_finite() {
        anyhow::bail!("v0.53 v0.38 teacher sentinel is non-finite");
    }
    Ok(fingerprint)
}

fn load_v0520_shared_parent_variables(
    varmap: &VarMap,
    checkpoint: &Path,
    device: &Device,
) -> Result<(usize, usize)> {
    let tensors = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("failed to load selected v0.52 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.53 VarMap lock poisoned during v0.52 warm start"))?;
    let mut loaded = 0usize;
    let mut fresh = 0usize;
    let mut missing = Vec::new();
    let mut unexpected = Vec::new();
    for (name, variable) in data.iter() {
        if name.starts_with(&format!("{FOUNDATION_V0530_STUDENT_NAMESPACE}.")) {
            fresh += 1;
            continue;
        }
        let required_parent = name.starts_with(&format!("{FOUNDATION_V0500_STUDENT_NAMESPACE}."))
            || name.starts_with(&format!("{FOUNDATION_V0510_STUDENT_NAMESPACE}."));
        if !required_parent {
            unexpected.push(name.clone());
            continue;
        }
        match tensors.get(name) {
            Some(tensor) => {
                if tensor.dims() != variable.as_tensor().dims() {
                    anyhow::bail!(
                        "v0.53 warm-start shape mismatch for {name}: parent {:?}, model {:?}",
                        tensor.dims(),
                        variable.as_tensor().dims()
                    );
                }
                variable.set(tensor)?;
                loaded += 1;
            }
            None => missing.push(name.clone()),
        }
    }
    drop(data);
    if !missing.is_empty() {
        anyhow::bail!(
            "selected v0.52 checkpoint is missing required shared variables: {}",
            missing.join(", ")
        );
    }
    if !unexpected.is_empty() {
        anyhow::bail!(
            "v0.53 student contains unexpected non-parent/non-v053 variables: {}",
            unexpected.join(", ")
        );
    }
    if loaded == 0 || fresh == 0 {
        anyhow::bail!(
            "v0.53 warm start is nonfunctional: loaded_shared_v0520={loaded} fresh_v0530={fresh}"
        );
    }
    Ok((loaded, fresh))
}

fn read_v035_metadata(checkpoint: &Path) -> Result<V035ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.35 metadata {path:?}"))
}

fn read_v038_parent_metadata(checkpoint: &Path) -> Result<V038ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.38 parent metadata {path:?}"))
}

fn read_v051_parent_metadata(checkpoint: &Path) -> Result<V051ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.51 parent metadata {path:?}"))
}

fn read_v052_parent_metadata(checkpoint: &Path) -> Result<V052ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.52 parent metadata {path:?}"))
}

fn read_v053_metadata(checkpoint: &Path) -> Result<V053Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.53 metadata {path:?}"))
}

#[allow(clippy::too_many_arguments)]
fn validate_v053_metadata(
    label: &str,
    metadata: &V053Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    supervision_fingerprint: &str,
    parent_v0350_checkpoint: &Path,
    parent_v0380_checkpoint: &Path,
    parent_v0510_checkpoint: &Path,
    parent_v0520_checkpoint: &Path,
    config: &PeptideFoundationV0530Config,
    batch_size: usize,
    steps_per_epoch: usize,
    max_epochs: usize,
    seed: u64,
    learning_rate: f64,
) -> Result<()> {
    if metadata.version != V053_VERSION
        || metadata.objective != V053_OBJECTIVE
        || metadata.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0530
    {
        anyhow::bail!("v0.53 {label} checkpoint identity mismatch");
    }
    if metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        || metadata.mobility_supervision_fingerprint != supervision_fingerprint
        || metadata.parent_v0350_checkpoint != parent_v0350_checkpoint.display().to_string()
        || metadata.parent_v0380_checkpoint != parent_v0380_checkpoint.display().to_string()
        || metadata.parent_v0510_checkpoint != parent_v0510_checkpoint.display().to_string()
        || metadata.parent_v0520_checkpoint != parent_v0520_checkpoint.display().to_string()
        || metadata.teacher_source != FOUNDATION_V0530_TEACHER_SOURCE
        || metadata.backbone_namespace != FOUNDATION_V0500_STUDENT_NAMESPACE
        || metadata.specialist_namespace != FOUNDATION_V0510_STUDENT_NAMESPACE
        || metadata.mobility_namespace != FOUNDATION_V0530_STUDENT_NAMESPACE
    {
        anyhow::bail!("v0.53 {label} checkpoint provenance mismatch");
    }
    if metadata.v0530_config != *config
        || metadata.batch_size != batch_size
        || metadata.steps_per_epoch != steps_per_epoch
        || metadata.max_epochs != max_epochs
        || metadata.seed != seed
    {
        anyhow::bail!("v0.53 {label} checkpoint run-contract mismatch");
    }
    if (metadata.learning_rate - learning_rate).abs()
        > f64::EPSILON * 64.0 * learning_rate.abs().max(1.0)
    {
        anyhow::bail!("v0.53 {label} checkpoint learning-rate mismatch");
    }
    Ok(())
}

fn save_checkpoint_v0530(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V053Metadata,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    optimizer.save_safetensors(directory.join("optimizer.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    Ok(())
}

fn deterministic_index_subset(indices: &[usize], max_records: usize, seed: u64) -> Vec<usize> {
    let mut ranked = indices
        .iter()
        .copied()
        .map(|index| (mix64(index as u64 ^ seed), index))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|item| item.0);
    ranked
        .into_iter()
        .take(max_records.min(indices.len()))
        .map(|item| item.1)
        .collect()
}

fn deterministic_consensus_subset(
    examples: &[MobilityConsensusExample],
    max_records: usize,
    seed: u64,
) -> Vec<MobilityConsensusExample> {
    let mut ranked = examples
        .iter()
        .cloned()
        .map(|example| (mix64(example.identity_hash ^ seed), example))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|item| item.0);
    ranked
        .into_iter()
        .take(max_records.min(examples.len()))
        .map(|item| item.1)
        .collect()
}

fn filtered_sampling_config(
    base: &FoundationSamplingConfig,
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    train_indices: &[usize],
    validation_indices: &[usize],
) -> FoundationSamplingConfig {
    let train_sources: BTreeSet<String> = train_indices
        .iter()
        .filter_map(|&index| provenance.get(index))
        .map(|item| item.source_id.clone())
        .collect();
    let validation_sources: BTreeSet<String> = validation_indices
        .iter()
        .filter_map(|&index| provenance.get(index))
        .map(|item| item.source_id.clone())
        .collect();
    let mut config = base.clone();
    if !config.source_weights.is_empty() {
        config
            .source_weights
            .retain(|source, _| train_sources.contains(source));
    }
    if !config.validation_source_weights.is_empty() {
        config
            .validation_source_weights
            .retain(|source, _| validation_sources.contains(source));
    }
    config
}

fn feasible_validation_batches(
    label: &str,
    config: &FoundationSamplingConfig,
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    validation_indices: &[usize],
    batch_size: usize,
    requested_batches: usize,
) -> Result<usize> {
    let max_batches = requested_batches.min(validation_indices.len() / batch_size);
    if max_batches == 0 {
        anyhow::bail!("v0.38 {label} cannot form one full batch");
    }
    if config.validation_source_weights.is_empty() {
        return Ok(max_batches);
    }
    let mut available = BTreeMap::<String, usize>::new();
    for &index in validation_indices {
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("v0.38 {label} provenance index out of bounds"))?;
        *available.entry(source.source_id.clone()).or_default() += 1;
    }
    let total_weight: f64 = config.validation_source_weights.values().copied().sum();
    if !(total_weight > 0.0 && total_weight.is_finite()) {
        anyhow::bail!("v0.38 {label} validation weights are invalid");
    }
    for batches in (1..=max_batches).rev() {
        let quotas = weighted_quotas(
            batches * batch_size,
            &config.validation_source_weights,
            total_weight,
        );
        if quotas
            .iter()
            .all(|(source, desired)| *desired <= available.get(source).copied().unwrap_or(0))
        {
            return Ok(batches);
        }
    }
    anyhow::bail!("v0.38 {label} cannot satisfy source quotas for one full batch")
}

fn weighted_quotas(
    target_records: usize,
    weights: &BTreeMap<String, f64>,
    total_weight: f64,
) -> BTreeMap<String, usize> {
    let mut quotas = BTreeMap::<String, usize>::new();
    let mut fractions = Vec::<(f64, String)>::new();
    let mut assigned = 0usize;
    for (source, weight) in weights {
        let exact = target_records as f64 * *weight / total_weight;
        let base = exact.floor() as usize;
        assigned += base;
        quotas.insert(source.clone(), base);
        fractions.push((exact - base as f64, source.clone()));
    }
    fractions.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    let mut remaining = target_records.saturating_sub(assigned);
    for (_, source) in fractions {
        if remaining == 0 {
            break;
        }
        *quotas.entry(source).or_default() += 1;
        remaining -= 1;
    }
    quotas
}

fn require_plan_records(label: &str, plan: &FoundationSamplePlan, expected: usize) -> Result<()> {
    if plan.indices.len() != expected {
        anyhow::bail!(
            "v0.38 {label} sample plan has {} records but {expected} are required",
            plan.indices.len()
        );
    }
    Ok(())
}

fn print_sample_plan(label: &str, plan: &FoundationSamplePlan) {
    println!(
        "sample_plan\tlane={label}\trecords={}\tunique_records={}",
        plan.indices.len(),
        plan.unique_records
    );
    for (source, count) in &plan.source_records {
        println!("sample_plan_source\tlane={label}\tsource={source}\trecords={count}");
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

fn parse_or<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse::<T>()
            .map_err(|error| anyhow::anyhow!("invalid argument {index} '{value}': {error}")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0530_gradient_audit_respects_zero_initialized_output_unlock() {
        let first = required_gradient_parameters_v0530(1);
        assert!(first.contains(&"student_v053.mobility_teacher_bridge.output.weight"));
        assert!(first.contains(&"student_v053.mobility_teacher_bridge.residue_projection.weight"));
        assert!(!first.contains(&"student_v053.mobility_teacher_bridge.pair_projection.weight"));
        assert!(!first.contains(&"student_v053.mobility_teacher_bridge.hidden.weight"));
        assert!(!first.contains(&"student_v053.mobility_teacher_bridge.bottleneck.weight"));

        let second = required_gradient_parameters_v0530(2);
        assert!(second.contains(&"student_v053.mobility_teacher_bridge.output.weight"));
        assert!(second.contains(&"student_v053.mobility_teacher_bridge.hidden.weight"));
        assert!(second.contains(&"student_v053.mobility_teacher_bridge.bottleneck.weight"));
        assert!(second.contains(&"student_v053.mobility_teacher_bridge.pair_projection.weight"));
        assert!(!second.contains(&"student_v053.mobility_teacher_bridge.residue_projection.weight"));

        assert!(required_gradient_parameters_v0530(3).is_empty());
    }

    #[test]
    fn v0530_affine_fit_recovers_linear_map() {
        let examples = vec![(1.0, 5.0), (2.0, 8.0), (3.0, 11.0), (4.0, 14.0)];
        let fit = fit_xy_affine(&examples).unwrap();
        assert!((fit.slope - 3.0).abs() < 1.0e-10);
        assert!((fit.intercept - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn v0530_consensus_prefers_reliable_source() {
        let mut groups = BTreeMap::new();
        groups.insert(
            "PEPTIDE|z=2|".to_string(),
            vec![
                MobilityObservation {
                    record_index: 10,
                    source_id: "reliable".into(),
                    target: 1.000,
                },
                MobilityObservation {
                    record_index: 20,
                    source_id: "noisy".into(),
                    target: 1.020,
                },
            ],
        );
        let mut supervision = BTreeMap::new();
        supervision.insert(
            "reliable".into(),
            SourceSupervision {
                source_id: "reliable".into(),
                reliability: 1.25,
                affine: AffineFit {
                    intercept: 0.0,
                    slope: 1.0,
                },
                ..Default::default()
            },
        );
        supervision.insert(
            "noisy".into(),
            SourceSupervision {
                source_id: "noisy".into(),
                reliability: 0.35,
                affine: AffineFit {
                    intercept: 0.0,
                    slope: 1.0,
                },
                ..Default::default()
            },
        );
        let (examples, multi, single, _, _) =
            build_consensus_from_groups(&groups, &supervision).unwrap();
        assert_eq!(multi, 1);
        assert_eq!(single, 0);
        assert_eq!(examples.len(), 1);
        assert!(f64::from(examples[0].target_mobility) < 1.010);
        assert!(f64::from(examples[0].target_mobility) > 1.000);
    }

    #[test]
    fn v0530_overlapping_views_collapse_to_project_families() {
        assert_eq!(source_family("pxd034128_11min"), "pxd034128");
        assert_eq!(
            source_family("pxd058337_60spd_fractionationgpf"),
            "pxd058337"
        );
        assert_eq!(source_family("ip2_bruker_human"), "ip2_bruker_human");
    }

    #[test]
    fn v0530_epoch_order_is_deterministic_and_unique_within_cycle() {
        let examples = (0..32)
            .map(|index| MobilityConsensusExample {
                representative_index: index,
                target_mobility: index as f32,
                weight: 1.0,
                source_count: 1,
                identity_hash: mix64(index as u64 + 1),
            })
            .collect::<Vec<_>>();
        let first = deterministic_example_order(&examples, 24, 3, 17);
        let second = deterministic_example_order(&examples, 24, 3, 17);
        assert_eq!(first, second);
        assert_eq!(first.iter().copied().collect::<BTreeSet<_>>().len(), 24);
    }
}
