//! v0.50 deep chemistry / residue-pair joint trainer.
//!
//! The v0.50 student is intentionally independent of the accepted v0.35 parameter
//! space.  v0.35 is loaded into a separate VarMap and used only as a detached
//! RT/MS2 teacher anchor.  The student is optimized end-to-end with three memory-
//! bounded update phases per training cycle:
//!
//! 1. clean RT/MS2 supervision + decaying v0.35 anchors + deterministic chemistry/pair auxiliaries;
//! 2. masked-residue / chemistry reconstruction + contrastive representation learning;
//! 3. TRAIN-only mobility consensus + exact Bruker mobility->CCS auxiliary supervision.
//!
//! TRAIN-HOLDOUT is never evaluated by this executable. Historical VALIDATION/APD
//! and historical TEST remain closed.  `smoke` uses small fixed TRAIN/DEV budgets
//! only to prove real-data trainer correctness; `train` performs DEV checkpoint
//! selection.  HOLDOUT finalization is deliberately a later explicit step.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    contrastive_info_nce_loss, foundation_ms2_loss, foundation_peptidoform_neutral_mass,
    load_foundation_corpus, multi_task_loss_with_ms2_config, read_foundation_training_run_config,
    sample_foundation_training_indices, sample_foundation_validation_indices, FoundationAdamW,
    FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCollator,
    FoundationCollatorConfig, FoundationCorruptionConfig, FoundationFragmentContextBatchV0350,
    FoundationLearningRateSchedule, FoundationLossWeights, FoundationModificationSite,
    FoundationMs2LossConfig, FoundationMultiTaskOutput, FoundationOptimizerStep, FoundationOutput,
    FoundationPartition, FoundationRecordProvenance, FoundationRegressionNormalization,
    FoundationSamplePlan, FoundationSamplingConfig, FoundationTargetNormalizationConfig,
    FoundationTrainingRecord, FoundationV0350TeacherV0500, PeptideFoundationMultimodalV0350Config,
    PeptideFoundationMultimodalV0350Model, PeptideFoundationV0500Config,
    PeptideFoundationV0500Model, RetentionTimeObjective, FOUNDATION_MS2_PEARSON_WEIGHT_V0350,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500, FOUNDATION_REPRESENTATION_CHEMISTRY_WEIGHT_V0350,
    FOUNDATION_REPRESENTATION_CONTRASTIVE_WEIGHT_V0350,
    FOUNDATION_REPRESENTATION_MASKED_WEIGHT_V0350, FOUNDATION_RT_ROBUST_DELTA_V0350,
    FOUNDATION_RT_ROBUST_WEIGHT_V0350, FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM, FOUNDATION_V0500_PAIR_CLASS_COUNT,
    FOUNDATION_V0500_STUDENT_NAMESPACE, FOUNDATION_V0500_TEACHER_SOURCE,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V050_VERSION: u32 = 500;
const V050_OBJECTIVE: &str = "v0500_deep_pair_joint_multimodal_foundation";
const V050_UPDATES_PER_CYCLE: usize = 3;
const V050_MAX_CYCLES_PER_EPOCH: usize = 3_072;
const V035_REFERENCE_DEV_BATCH_SIZE: usize = 64;
const V035_REFERENCE_DEV_BATCHES: usize = 256;
const V035_REFERENCE_SEED: u64 = 20_261_035;
const V050_EVAL_BATCH_SIZE: usize = 32;
const V050_SMOKE_CYCLES: usize = 4;
const V050_SMOKE_DEV_BATCHES: usize = 4;
const V050_SMOKE_CCS_RECORDS: usize = 256;
const V050_LOSS_CALIBRATION_RECORDS: usize = 2_048;
const V050_SMOKE_LOSS_CALIBRATION_RECORDS: usize = 64;
const V050_MAX_GRADIENT_NORM: f64 = 1.0;

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

// v0.50 auxiliary weights.  Teacher anchors are deliberately modest and decay.
const V050_PAIR_AUX_WEIGHT: f64 = 0.10;
const V050_CHEMISTRY_SUMMARY_WEIGHT: f64 = 0.10;
const V050_TEACHER_RT_START: f64 = 0.15;
const V050_TEACHER_RT_END: f64 = 0.02;
const V050_TEACHER_MS2_START: f64 = 0.10;
const V050_TEACHER_MS2_END: f64 = 0.01;
const V050_TEACHER_DECAY_FRACTION: f64 = 0.40;
const V050_MOBILITY_UPDATE_WEIGHT: f64 = 1.0;

// Frozen references used for DEV reporting/selection.  They are not re-fit from DEV.
const V035_RT_DEV_MAE: f64 = 4.528_375;
const V038_RAW_CCS_DEV_MAE: f64 = 8.806_119_29;
const V038_CONSENSUS_CCS_DEV_MAE: f64 = 8.922_126_00;
const V035_MS2_COSINE_DEV: f64 = 0.904_061;
const V035_MS2_SPECTRAL_ANGLE_DEV: f64 = 0.747_700;
const V035_MS2_PEARSON_DEV: f64 = 0.684_548;
const V050_RT_TARGET_MAE: f64 = 4.3;
const V050_RAW_CCS_PREFERRED_MAE: f64 = 8.669;
const V050_MS2_REGRESSION_TOLERANCE: f64 = 0.003;

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
struct V050Metadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    mobility_supervision_fingerprint: String,
    parent_v0350_checkpoint: String,
    parent_v0350_completed_steps: usize,
    teacher_source: String,
    student_namespace: String,
    v0500_config: PeptideFoundationV0500Config,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    batch_size: usize,
    cycles_per_epoch: usize,
    updates_per_cycle: usize,
    max_epochs: usize,
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
    teacher_rt_weight: f64,
    teacher_ms2_weight: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 11 {
        anyhow::bail!(
            "usage: foundation_train_multimodal_v0500 RUN_V0260.yaml OUTPUT_DIR PARENT_V0350_CHECKPOINT [max_epochs=8] [batch_size=8] [patience=3] [min_delta=0.002] [seed=20261050] [learning_rate=2e-5] [mode=smoke|train|resume]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let requested_max_epochs = parse_or(&args, 4, 8usize)?;
    let batch_size = parse_or(&args, 5, 8usize)?;
    let patience = parse_or(&args, 6, 3usize)?;
    let min_delta = parse_or(&args, 7, 0.002f64)?;
    let seed = parse_or(&args, 8, 20_261_050u64)?;
    let learning_rate = parse_or(&args, 9, 2.0e-5f64)?;
    let run_mode = args.get(10).map(String::as_str).unwrap_or("smoke");
    let (smoke_mode, resume_mode) = match run_mode {
        "smoke" => (true, false),
        "train" => (false, false),
        "resume" => (false, true),
        other => {
            anyhow::bail!("unsupported v0.50 run mode {other:?}; expected smoke, train, or resume")
        }
    };
    let max_epochs = if smoke_mode { 1 } else { requested_max_epochs };

    if max_epochs == 0 || batch_size < 2 || patience == 0 {
        anyhow::bail!("v0.50 requires max_epochs>0, batch_size>=2, and patience>0");
    }
    if !(min_delta >= 0.0 && min_delta.is_finite()) {
        anyhow::bail!("v0.50 min_delta must be finite and non-negative");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("v0.50 learning_rate must be positive and finite");
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
                    anyhow::bail!("v0.50 resume is missing {:?}", directory.join(name));
                }
            }
        }
    } else if output_root.exists() {
        anyhow::bail!("v0.50 output directory already exists: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.50 training requires a CUDA device")?;
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

    let parent_metadata = read_v035_metadata(&parent_checkpoint)?;
    if parent_metadata.version != 350 {
        anyhow::bail!(
            "v0.50 requires v0.35 metadata version 350, observed {}",
            parent_metadata.version
        );
    }
    if parent_metadata.objective != "v0350_trainable_forward_representation_context_conditioned_ms2"
    {
        anyhow::bail!(
            "v0.50 requires the accepted v0.35 objective, observed {}",
            parent_metadata.objective
        );
    }
    if parent_metadata.completed_steps == 0 {
        anyhow::bail!("v0.50 refuses an unselected v0.35 baseline checkpoint");
    }
    parent_metadata.v0350_config.validate()?;
    let parent_forward = parent_metadata.v0350_config.forward().clone();

    let current_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let current_benchmark_fingerprint =
        format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != current_corpus_fingerprint {
        anyhow::bail!("v0.50 corpus fingerprint differs from frozen v0.35 teacher");
    }
    if parent_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint {
        anyhow::bail!("v0.50 benchmark fingerprint differs from frozen v0.35 teacher");
    }

    let mut v0500_config = PeptideFoundationV0500Config::default();
    v0500_config.max_sequence_len = parent_forward.max_sequence_len;
    v0500_config.max_atoms_per_residue = parent_forward.max_atoms_per_residue;
    v0500_config.instrument_vocab_size = parent_forward.instrument_vocab_size;
    v0500_config.ms2_fragment_channels = parent_forward.ms2_fragment_channels;
    v0500_config.validate()?;

    let mut prepared_max_sequence_len = 0usize;
    for &index in train_indices.iter().chain(&dev_indices) {
        let length = corpus.records[index].peptidoform.sequence.chars().count();
        prepared_max_sequence_len = prepared_max_sequence_len.max(length);
        if length > v0500_config.max_sequence_len {
            anyhow::bail!(
                "v0.50 record {index} length {length} exceeds max_sequence_len={}",
                v0500_config.max_sequence_len
            );
        }
    }

    let mobility_train_indices = finite_mobility_ccs_indices(&corpus.records, &train_indices);
    let ccs_dev_indices_full = finite_mobility_ccs_indices(&corpus.records, &dev_indices);
    if mobility_train_indices.len() < batch_size || ccs_dev_indices_full.is_empty() {
        anyhow::bail!(
            "v0.50 requires mobility labels in TRAIN and DEV; observed train={} dev={} batch_size={batch_size}",
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
        anyhow::bail!("v0.50 mobility consensus TRAIN has fewer examples than batch_size");
    }
    let dev_consensus_full = build_partition_consensus_examples(
        &corpus.records,
        &corpus.provenance,
        &ccs_dev_indices_full,
        &train_supervision.source_supervision,
    )?;
    if dev_consensus_full.is_empty() {
        anyhow::bail!("v0.50 DEV mobility consensus set is empty");
    }

    let full_cycles_per_epoch = (train_indices.len() / batch_size)
        .min(train_supervision.examples.len() / batch_size)
        .min(V050_MAX_CYCLES_PER_EPOCH)
        .max(1);
    let cycles_per_epoch = if smoke_mode {
        V050_SMOKE_CYCLES.min(full_cycles_per_epoch).max(1)
    } else {
        full_cycles_per_epoch
    };
    let max_updates = max_epochs
        .saturating_mul(cycles_per_epoch)
        .saturating_mul(V050_UPDATES_PER_CYCLE);

    let train_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_indices,
        &dev_indices,
    );
    let (dev_plan, dev_evaluation_batch_size, dev_cohort_policy) = if smoke_mode {
        let requested_dev_batches = V050_SMOKE_DEV_BATCHES;
        let dev_batches = feasible_validation_batches(
            "dev_smoke",
            &train_sampling,
            &corpus.provenance,
            &dev_indices,
            batch_size,
            requested_dev_batches,
        )?;
        let mut dev_sampling = train_sampling.clone();
        dev_sampling.validation_steps = Some(dev_batches);
        (
            sample_foundation_validation_indices(
                &corpus.records,
                &corpus.provenance,
                &dev_indices,
                batch_size,
                seed ^ 0x5000_d3f0_1234_5678,
                &dev_sampling,
            )?,
            batch_size,
            "bounded_v0500_smoke_dev",
        )
    } else {
        // Reproduce the accepted v0.35 forward DEV selection cohort: 256 batches
        // at batch 64 under the frozen v0.35 seed/sampling policy.  Evaluation
        // itself can use a different physical batch size without changing cohort membership.
        let reference_batches = feasible_validation_batches(
            "v035_reference_dev",
            &train_sampling,
            &corpus.provenance,
            &dev_indices,
            V035_REFERENCE_DEV_BATCH_SIZE,
            V035_REFERENCE_DEV_BATCHES,
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
            V050_EVAL_BATCH_SIZE,
            "v035_canonical_forward_dev_cohort",
        )
    };
    let ccs_dev_indices = if smoke_mode {
        deterministic_index_subset(
            &ccs_dev_indices_full,
            V050_SMOKE_CCS_RECORDS,
            seed ^ 0x5000_cc50_1111_2222,
        )
    } else {
        ccs_dev_indices_full.clone()
    };
    let dev_consensus = if smoke_mode {
        deterministic_consensus_subset(
            &dev_consensus_full,
            V050_SMOKE_CCS_RECORDS,
            seed ^ 0x5000_cc50_3333_4444,
        )
    } else {
        dev_consensus_full.clone()
    };

    let featurizer_config = v0500_config.featurizer_config();
    let clean_collator = FoundationCollator::new(
        featurizer_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: parent_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let representation_collator = FoundationCollator::new(
        featurizer_config,
        FoundationCollatorConfig {
            retention_time_objective: parent_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.15,
                chemistry_mask_probability: 0.15,
            },
        },
    )?;

    let mut student_varmap = VarMap::new();
    let student_vb = VarBuilder::from_varmap(&student_varmap, DType::F32, &device);
    let student = PeptideFoundationV0500Model::new(v0500_config.clone(), student_vb)?;

    let mut teacher_varmap = VarMap::new();
    let teacher_vb = VarBuilder::from_varmap(&teacher_varmap, DType::F32, &device);
    let teacher_model = PeptideFoundationMultimodalV0350Model::new(
        parent_metadata.v0350_config.clone(),
        teacher_vb,
    )?;
    teacher_varmap
        .load(parent_checkpoint.join("model.safetensors"))
        .with_context(|| format!("failed to load v0.35 teacher from {parent_checkpoint:?}"))?;
    let teacher = FoundationV0350TeacherV0500::new(teacher_model);

    let optimizer_prefixes = [format!("{FOUNDATION_V0500_STUDENT_NAMESPACE}.")];
    let optimizer_prefix_refs = optimizer_prefixes
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut optimizer = FoundationAdamW::new_for_prefixes(
        &student_varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: run.trainer.weight_decay,
        },
        &optimizer_prefix_refs,
    )?;
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 500u64.min(max_updates.saturating_sub(1) as u64),
        total_steps: max_updates.max(1) as u64,
        min_lr_ratio: 0.10,
    };

    let target_normalization = parent_metadata.target_normalization;
    let ms2_loss = parent_metadata.ms2_loss.validate()?;
    let requested_calibration_records = if smoke_mode {
        V050_SMOKE_LOSS_CALIBRATION_RECORDS
    } else {
        V050_LOSS_CALIBRATION_RECORDS
    };
    let calibration_count = requested_calibration_records
        .min(train_supervision.examples.len())
        .max(batch_size);
    let calibration_order = deterministic_example_order(
        &train_supervision.examples,
        calibration_count,
        0,
        seed ^ 0x5000_4c4f_5353_4343,
    );
    let calibration_examples = calibration_order
        .iter()
        .map(|&index| train_supervision.examples[index].clone())
        .collect::<Vec<_>>();
    let loss_scales = calibrate_mobility_scales_v0500(
        &student,
        &clean_collator,
        &corpus.records,
        &calibration_examples,
        batch_size,
        &device,
    )?;

    if !resume_mode {
        fs::create_dir_all(&output_root)?;
        write_mobility_supervision_summary(&output_root, &train_supervision)?;
    }

    let teacher_sentinel_records = dev_plan
        .indices
        .iter()
        .take(batch_size)
        .map(|&index| corpus.records[index].clone())
        .collect::<Vec<_>>();
    let teacher_sentinel_initial = teacher_sentinel_fingerprint(
        &teacher,
        &clean_collator,
        &teacher_sentinel_records,
        &parent_forward,
        &device,
    )?;

    println!("v0500_version\tv0.50-deep-pair-joint-multimodal-foundation");
    println!("objective\t{V050_OBJECTIVE}");
    println!("architecture\t{FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500}");
    println!("device\t{device:?}");
    println!("run_mode\t{run_mode}");
    println!("student_namespace\t{FOUNDATION_V0500_STUDENT_NAMESPACE}");
    println!("teacher_source\t{FOUNDATION_V0500_TEACHER_SOURCE}");
    println!("parent_v0350_checkpoint\t{}", parent_checkpoint.display());
    println!(
        "parent_v0350_completed_steps\t{}",
        parent_metadata.completed_steps
    );
    println!("teacher_update_policy\tfrozen_separate_varmap_detached_outputs");
    println!("teacher_sentinel_initial\t{teacher_sentinel_initial:.8}");
    println!("graph_hidden_dim\t{}", v0500_config.graph_hidden_dim);
    println!("graph_layers\t{}", v0500_config.graph_layers);
    println!("residue_dim\t{}", v0500_config.residue_dim);
    println!("pair_dim\t{}", v0500_config.pair_dim);
    println!("interaction_blocks\t{}", v0500_config.interaction_blocks);
    println!("attention_heads\t{}", v0500_config.num_attention_heads);
    println!("feed_forward_dim\t{}", v0500_config.feed_forward_dim);
    println!("prepared_max_sequence_len\t{prepared_max_sequence_len}");
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
    println!(
        "mobility_train_multisource_examples\t{}",
        train_supervision.multisource_examples
    );
    println!(
        "mobility_train_singleton_examples\t{}",
        train_supervision.singleton_examples
    );
    println!("dev_raw_ccs_records\t{}", ccs_dev_indices.len());
    println!("dev_consensus_examples\t{}", dev_consensus.len());
    println!("dev_property_cohort_policy\t{dev_cohort_policy}");
    println!("dev_property_records\t{}", dev_plan.indices.len());
    println!("dev_evaluation_batch_size\t{dev_evaluation_batch_size}");
    println!("cycles_per_epoch\t{cycles_per_epoch}");
    println!("updates_per_cycle\t{V050_UPDATES_PER_CYCLE}");
    println!("max_epochs\t{max_epochs}");
    println!("max_optimizer_updates\t{max_updates}");
    println!("batch_size\t{batch_size}");
    println!("base_learning_rate\t{learning_rate}");
    println!("optimizer_scope\t{FOUNDATION_V0500_STUDENT_NAMESPACE}.*_only");
    println!("optimizer_variable_count\t{}", optimizer.variable_count());
    println!(
        "training_schedule\tproperty_teacher_pair_then_representation_then_mobility_consensus"
    );
    println!("mobility_loss_scale_native\t{:.8}", loss_scales.mobility);
    println!("ccs_aux_loss_scale_native\t{:.8}", loss_scales.ccs);
    println!("mobility_supervision_policy\tv038_train_only_source_affine_family_deduplicated_reliability_weighted_consensus");
    println!("teacher_rt_weight_start\t{V050_TEACHER_RT_START}");
    println!("teacher_rt_weight_end\t{V050_TEACHER_RT_END}");
    println!("teacher_ms2_weight_start\t{V050_TEACHER_MS2_START}");
    println!("teacher_ms2_weight_end\t{V050_TEACHER_MS2_END}");
    println!("teacher_decay_fraction\t{V050_TEACHER_DECAY_FRACTION}");
    println!("pair_aux_weight\t{V050_PAIR_AUX_WEIGHT}");
    println!("chemistry_summary_weight\t{V050_CHEMISTRY_SUMMARY_WEIGHT}");
    println!("representation_masked_weight\t{FOUNDATION_REPRESENTATION_MASKED_WEIGHT_V0350}");
    println!("representation_chemistry_weight\t{FOUNDATION_REPRESENTATION_CHEMISTRY_WEIGHT_V0350}");
    println!(
        "representation_contrastive_weight\t{FOUNDATION_REPRESENTATION_CONTRASTIVE_WEIGHT_V0350}"
    );
    println!("fragment_cleavage_supervision\tdirect_context_conditioned_ms2_per_cleavage");
    println!("dev_reference_rt_mae\t{V035_RT_DEV_MAE}");
    println!("dev_reference_raw_ccs_mae\t{V038_RAW_CCS_DEV_MAE}");
    println!("dev_reference_consensus_ccs_mae\t{V038_CONSENSUS_CCS_DEV_MAE}");
    println!("dev_reference_ms2_cosine\t{V035_MS2_COSINE_DEV}");
    println!("dev_reference_ms2_spectral_angle\t{V035_MS2_SPECTRAL_ANGLE_DEV}");
    println!("dev_reference_ms2_pearson\t{V035_MS2_PEARSON_DEV}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    print_sample_plan("dev", &dev_plan);

    let supervision_fingerprint = format!("fnv1a64:{:016x}", train_supervision.fingerprint);
    let metadata =
        |completed_epochs: usize, completed_updates: usize, dev_objective: f64| -> V050Metadata {
            V050Metadata {
                version: V050_VERSION,
                objective: V050_OBJECTIVE.into(),
                architecture: FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500.into(),
                corpus_fingerprint: current_corpus_fingerprint.clone(),
                benchmark_manifest_fingerprint: current_benchmark_fingerprint.clone(),
                mobility_supervision_fingerprint: supervision_fingerprint.clone(),
                parent_v0350_checkpoint: parent_checkpoint.display().to_string(),
                parent_v0350_completed_steps: parent_metadata.completed_steps,
                teacher_source: FOUNDATION_V0500_TEACHER_SOURCE.into(),
                student_namespace: FOUNDATION_V0500_STUDENT_NAMESPACE.into(),
                v0500_config: v0500_config.clone(),
                rt_objective: parent_metadata.rt_objective,
                target_normalization,
                ms2_loss,
                batch_size,
                cycles_per_epoch,
                updates_per_cycle: V050_UPDATES_PER_CYCLE,
                max_epochs,
                seed,
                learning_rate,
                mobility_loss_scale_native: loss_scales.mobility,
                ccs_aux_loss_scale_native: loss_scales.ccs,
                completed_epochs,
                completed_updates,
                dev_objective,
                smoke_mode,
            }
        };

    let initial_dev = if resume_mode {
        None
    } else {
        let value = evaluate_dev(
            &student,
            &clean_collator,
            &corpus.records,
            &dev_plan.indices,
            &ccs_dev_indices,
            &dev_consensus,
            dev_evaluation_batch_size,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        print_dev_metrics("train_dev_initial", 0, value);
        Some(value)
    };

    let (
        mut global_update,
        mut start_epoch,
        mut best_step,
        mut best_epoch,
        mut best_objective,
        mut stale_epochs,
    ) = if resume_mode {
        let initial_metadata = read_v050_metadata(&output_root.join("initial"))?;
        let latest_metadata = read_v050_metadata(&output_root.join("latest"))?;
        let best_metadata = read_v050_metadata(&output_root.join("best"))?;
        for (label, item) in [
            ("initial", &initial_metadata),
            ("latest", &latest_metadata),
            ("best", &best_metadata),
        ] {
            validate_v050_metadata(
                label,
                item,
                &current_corpus_fingerprint,
                &current_benchmark_fingerprint,
                &supervision_fingerprint,
                &v0500_config,
                batch_size,
                cycles_per_epoch,
                max_epochs,
                seed,
                learning_rate,
            )?;
        }
        student_varmap.load(output_root.join("latest/model.safetensors"))?;
        optimizer.load_safetensors(&output_root.join("latest/optimizer.safetensors"))?;
        optimizer.set_step_count(latest_metadata.completed_updates as u64);
        let best_step = best_metadata.completed_updates;
        let best_epoch = best_metadata.completed_epochs;
        let stale = latest_metadata.completed_epochs.saturating_sub(best_epoch);
        println!(
            "v0500_resume\tlatest_epoch={}\tlatest_update={}\tbest_epoch={}\tbest_update={}\tbest_dev_objective={:.8}\tstale_epochs={stale}",
            latest_metadata.completed_epochs,
            latest_metadata.completed_updates,
            best_epoch,
            best_step,
            best_metadata.dev_objective,
        );
        (
            latest_metadata.completed_updates,
            latest_metadata.completed_epochs + 1,
            best_step,
            best_epoch,
            best_metadata.dev_objective,
            stale,
        )
    } else {
        save_checkpoint(
            &output_root.join("initial"),
            &student_varmap,
            &optimizer,
            &metadata(
                0,
                0,
                initial_dev
                    .expect("fresh v0.50 run has initial DEV")
                    .objective,
            ),
        )?;
        let initial_objective = initial_dev
            .expect("fresh v0.50 run has initial DEV")
            .objective;
        save_checkpoint(
            &output_root.join("best"),
            &student_varmap,
            &optimizer,
            &metadata(0, 0, initial_objective),
        )?;
        (0usize, 1usize, 0usize, 0usize, initial_objective, 0usize)
    };

    let mut stopped_early = stale_epochs >= patience;
    if start_epoch > max_epochs {
        println!(
            "v0500_training_budget_complete\tstart_epoch={start_epoch}\tmax_epochs={max_epochs}"
        );
    } else if stopped_early {
        println!(
            "v0500_training_already_early_stopped\tstale_epochs={stale_epochs}\tpatience={patience}"
        );
    } else {
        for epoch in start_epoch..=max_epochs {
            let mut train_sampling_epoch = train_sampling.clone();
            train_sampling_epoch.train_steps_per_epoch = Some(cycles_per_epoch);
            let property_plan = sample_foundation_training_indices(
                &corpus.records,
                &corpus.provenance,
                &train_indices,
                batch_size,
                epoch as u64,
                seed ^ 0x5000_5052_4f50_4552,
                true,
                &train_sampling_epoch,
            )?;
            require_plan_records(
                "property_train",
                &property_plan,
                cycles_per_epoch.saturating_mul(batch_size),
            )?;
            let mobility_order = deterministic_example_order(
                &train_supervision.examples,
                cycles_per_epoch.saturating_mul(batch_size),
                epoch as u64,
                seed ^ 0x5000_4d4f_4249_4c49,
            );
            println!("v0500_epoch\tstage=start\tepoch={epoch}\tcycles={cycles_per_epoch}");

            for cycle in 0..cycles_per_epoch {
                let offset = cycle.saturating_mul(batch_size);
                let records = property_plan.indices[offset..offset + batch_size]
                    .iter()
                    .map(|&index| corpus.records[index].clone())
                    .collect::<Vec<_>>();
                let mobility_examples = mobility_order[offset..offset + batch_size]
                    .iter()
                    .map(|&example_index| train_supervision.examples[example_index].clone())
                    .collect::<Vec<_>>();

                global_update += 1;
                set_scheduled_lr(&mut optimizer, &lr_schedule, learning_rate, global_update)?;
                let progress = global_update as f64 / max_updates.max(1) as f64;
                let property = property_teacher_pair_loss(
                    &student,
                    &teacher,
                    &clean_collator,
                    &records,
                    &parent_forward,
                    &target_normalization,
                    ms2_loss,
                    progress,
                    seed ^ (global_update as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    &device,
                )?;
                let property_update = backward_step_v0500(
                    "property",
                    &property.0,
                    &mut optimizer,
                    &student_varmap,
                    smoke_mode && cycle == 0,
                    v0500_config.interaction_blocks,
                )?;
                log_update(
                    "property",
                    epoch,
                    cycle + 1,
                    global_update,
                    property.1,
                    property_update.learning_rate,
                    property_update.gradient_norm,
                    property_update.gradient_scale,
                );

                global_update += 1;
                set_scheduled_lr(&mut optimizer, &lr_schedule, learning_rate, global_update)?;
                let representation = representation_loss_v0500(
                    &student,
                    &representation_collator,
                    &records,
                    ms2_loss,
                    run.trainer.contrastive_temperature,
                    seed ^ (global_update as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
                    &device,
                )?;
                let representation_value = f64::from(representation.to_scalar::<f32>()?);
                let representation_update = backward_step_v0500(
                    "representation",
                    &representation,
                    &mut optimizer,
                    &student_varmap,
                    smoke_mode && cycle == 0,
                    v0500_config.interaction_blocks,
                )?;
                let representation_diag = UpdateDiagnostics {
                    total: representation_value,
                    primary: representation_value,
                    auxiliary: 0.0,
                    teacher_rt_weight: 0.0,
                    teacher_ms2_weight: 0.0,
                };
                log_update(
                    "representation",
                    epoch,
                    cycle + 1,
                    global_update,
                    representation_diag,
                    representation_update.learning_rate,
                    representation_update.gradient_norm,
                    representation_update.gradient_scale,
                );

                global_update += 1;
                set_scheduled_lr(&mut optimizer, &lr_schedule, learning_rate, global_update)?;
                let mobility = mobility_consensus_loss_v0500(
                    &student,
                    &clean_collator,
                    &corpus.records,
                    &mobility_examples,
                    loss_scales,
                    seed ^ (global_update as u64).wrapping_mul(0xa24b_1c62_4073_f5d9),
                    &device,
                )?;
                let mobility_value = f64::from(mobility.to_scalar::<f32>()?);
                let mobility_weighted = mobility.affine(V050_MOBILITY_UPDATE_WEIGHT, 0.0)?;
                let mobility_update = backward_step_v0500(
                    "mobility",
                    &mobility_weighted,
                    &mut optimizer,
                    &student_varmap,
                    smoke_mode && cycle == 0,
                    v0500_config.interaction_blocks,
                )?;
                let mobility_diag = UpdateDiagnostics {
                    total: mobility_value * V050_MOBILITY_UPDATE_WEIGHT,
                    primary: mobility_value,
                    auxiliary: 0.0,
                    teacher_rt_weight: 0.0,
                    teacher_ms2_weight: 0.0,
                };
                log_update(
                    "mobility",
                    epoch,
                    cycle + 1,
                    global_update,
                    mobility_diag,
                    mobility_update.learning_rate,
                    mobility_update.gradient_norm,
                    mobility_update.gradient_scale,
                );
            }

            let teacher_sentinel = teacher_sentinel_fingerprint(
                &teacher,
                &clean_collator,
                &teacher_sentinel_records,
                &parent_forward,
                &device,
            )?;
            let teacher_delta = (teacher_sentinel - teacher_sentinel_initial).abs();
            if teacher_delta > 1.0e-5 * teacher_sentinel_initial.abs().max(1.0) {
                anyhow::bail!(
                    "v0.50 frozen teacher changed: initial={teacher_sentinel_initial:.8} current={teacher_sentinel:.8} delta={teacher_delta:.8}"
                );
            }
            println!(
                "teacher_freeze_audit\tepoch={epoch}\tstatus=PASS\tfingerprint={teacher_sentinel:.8}\tdelta={teacher_delta:.8}"
            );

            let dev = evaluate_dev(
                &student,
                &clean_collator,
                &corpus.records,
                &dev_plan.indices,
                &ccs_dev_indices,
                &dev_consensus,
                dev_evaluation_batch_size,
                &target_normalization,
                ms2_loss,
                &device,
            )?;
            print_dev_metrics("train_dev", global_update, dev);
            let improved = best_objective - dev.objective > min_delta;
            println!(
                "train_dev_objective\tepoch={epoch}\tupdate={global_update}\tvalue={:.8}\tprevious_best={:.8}\timproved={improved}",
                dev.objective,
                best_objective,
            );
            save_checkpoint(
                &output_root.join("latest"),
                &student_varmap,
                &optimizer,
                &metadata(epoch, global_update, dev.objective),
            )?;
            if improved {
                best_objective = dev.objective;
                best_epoch = epoch;
                best_step = global_update;
                stale_epochs = 0;
                save_checkpoint(
                    &output_root.join("best"),
                    &student_varmap,
                    &optimizer,
                    &metadata(epoch, global_update, dev.objective),
                )?;
                println!(
                    "v0500_best_checkpoint\tepoch={best_epoch}\tupdate={best_step}\tdev_objective={best_objective:.8}"
                );
            } else {
                stale_epochs += 1;
            }
            println!(
                "v0500_epoch\tstage=complete\tepoch={epoch}\tupdate={global_update}\tstale_epochs={stale_epochs}"
            );
            if smoke_mode {
                println!("v0500_smoke_complete\tepoch={epoch}\tupdates={global_update}");
                break;
            }
            if stale_epochs >= patience {
                stopped_early = true;
                println!(
                    "v0500_early_stop\tepoch={epoch}\tupdate={global_update}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_update={best_step}\tbest_dev_objective={best_objective:.8}"
                );
                break;
            }
        }
    }

    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!(
        "v0500_training_complete\tbest_epoch={best_epoch}\tbest_update={best_step}\tbest_dev_objective={best_objective:.8}\tstopped_early={stopped_early}\tsmoke_mode={smoke_mode}"
    );

    student_varmap.load(output_root.join("best/model.safetensors"))?;
    let best_dev = evaluate_dev(
        &student,
        &clean_collator,
        &corpus.records,
        &dev_plan.indices,
        &ccs_dev_indices,
        &dev_consensus,
        dev_evaluation_batch_size,
        &target_normalization,
        ms2_loss,
        &device,
    )?;
    print_dev_metrics("best_train_dev", best_step, best_dev);
    print_material_gate(best_dev, smoke_mode);
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

fn backward_step_v0500(
    stage: &str,
    loss: &Tensor,
    optimizer: &mut FoundationAdamW,
    varmap: &VarMap,
    audit_gradients: bool,
    interaction_blocks: usize,
) -> Result<FoundationOptimizerStep> {
    let gradients = loss.backward()?;
    if audit_gradients {
        audit_stage_gradients_v0500(stage, varmap, &gradients, interaction_blocks)?;
    }
    Ok(optimizer.step(&gradients, Some(V050_MAX_GRADIENT_NORM))?)
}

fn audit_stage_gradients_v0500(
    stage: &str,
    varmap: &VarMap,
    gradients: &candle_core::backprop::GradStore,
    interaction_blocks: usize,
) -> Result<()> {
    let last_block = interaction_blocks.checked_sub(1).ok_or_else(|| {
        anyhow::anyhow!("v0.50 gradient audit requires at least one interaction block")
    })?;
    let mut required = vec![
        "student_v050.chemistry.atom_input.weight".to_string(),
        "student_v050.interaction.0.attention.query.weight".to_string(),
        format!("student_v050.interaction.{last_block}.attention.query.weight"),
        "student_v050.task.embedding.weight".to_string(),
    ];
    match stage {
        "property" => required.extend([
            format!("student_v050.interaction.{last_block}.pair_update_left.weight"),
            "student_v050.heads.rt.output.weight".to_string(),
            "student_v050.heads.ms2.output.weight".to_string(),
            "student_v050.heads.pair_interaction.weight".to_string(),
            "student_v050.heads.chemistry_summary.weight".to_string(),
        ]),
        "representation" => required.push("student_v050.heads.contrastive.weight".to_string()),
        "mobility" => required.push("student_v050.heads.mobility.output.weight".to_string()),
        other => anyhow::bail!("unknown v0.50 gradient-audit stage {other}"),
    }

    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.50 VarMap lock poisoned during gradient audit"))?;
    for name in required {
        let variable = data
            .get(name.as_str())
            .ok_or_else(|| anyhow::anyhow!("v0.50 gradient audit missing parameter {name}"))?;
        let gradient = gradients
            .get(variable)
            .ok_or_else(|| anyhow::anyhow!("v0.50 {stage} gradient missing for {name}"))?;
        let norm2 = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !norm2.is_finite() || norm2 <= 0.0 {
            anyhow::bail!("v0.50 {stage} gradient invalid for {name}: norm2={norm2}");
        }
        println!(
            "v0500_gradient_audit\tstage={stage}\tparameter={name}\tnorm={:.8}",
            norm2.sqrt()
        );
    }
    Ok(())
}

fn set_scheduled_lr(
    optimizer: &mut FoundationAdamW,
    schedule: &FoundationLearningRateSchedule,
    base_lr: f64,
    update: usize,
) -> Result<()> {
    let lr = schedule.learning_rate(base_lr, update.saturating_sub(1) as u64)?;
    optimizer.set_learning_rate(lr)?;
    Ok(())
}

fn log_update(
    stage: &str,
    epoch: usize,
    cycle: usize,
    update: usize,
    diag: UpdateDiagnostics,
    lr: f64,
    gradient_norm: f64,
    gradient_scale: f64,
) {
    if update <= V050_UPDATES_PER_CYCLE || cycle % 25 == 0 {
        println!(
            "v0500_train\tstage={stage}\tepoch={epoch}\tcycle={cycle}\tupdate={update}\tlr={lr:.8}\ttotal={:.6}\tprimary={:.6}\tauxiliary={:.6}\tteacher_rt_weight={:.6}\tteacher_ms2_weight={:.6}\tgradient_norm={gradient_norm:.6}\tgradient_scale={gradient_scale:.6}",
            diag.total,
            diag.primary,
            diag.auxiliary,
            diag.teacher_rt_weight,
            diag.teacher_ms2_weight,
        );
    }
}

fn teacher_anchor_weights(progress: f64) -> (f64, f64) {
    let fraction = (progress / V050_TEACHER_DECAY_FRACTION).clamp(0.0, 1.0);
    let rt = V050_TEACHER_RT_START + fraction * (V050_TEACHER_RT_END - V050_TEACHER_RT_START);
    let ms2 = V050_TEACHER_MS2_START + fraction * (V050_TEACHER_MS2_END - V050_TEACHER_MS2_START);
    (rt, ms2)
}

#[allow(clippy::too_many_arguments)]
fn property_teacher_pair_loss(
    student: &PeptideFoundationV0500Model,
    teacher: &FoundationV0350TeacherV0500,
    clean_collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    teacher_forward_config: &redeem_properties::foundation::FoundationConfig,
    normalization: &FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    progress: f64,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, UpdateDiagnostics)> {
    let mut batch = clean_collator.collate(records, device, seed)?;
    normalize_rt_target(&mut batch.targets, normalization)?;
    let student_output = student.forward_t(&batch.input, &batch.context, true)?;

    let dummy_ccs = Tensor::zeros((records.len(), 1), DType::F32, device)?;
    let foundation_output = FoundationOutput {
        residue_embeddings: student_output.representation.residue_embeddings.clone(),
        peptide_embedding: student_output.representation.global_embedding.clone(),
        residue_mask: student_output.representation.residue_mask.clone(),
        chemistry_targets: student_output.representation.chemistry_targets.clone(),
    };
    let multitask_output = FoundationMultiTaskOutput {
        foundation: foundation_output,
        rt: student_output.rt.clone(),
        ccs: dummy_ccs,
        ms2: student_output.ms2.clone(),
        residue_logits: student_output.residue_logits.clone(),
        chemistry_reconstruction: student_output.chemistry_reconstruction.clone(),
        contrastive_projection: student_output.contrastive_projection.clone(),
    };
    let mut supervised_targets = batch.targets.clone();
    supervised_targets.ccs = None;
    supervised_targets.ccs_mask = None;
    supervised_targets.masked_residue_classes = None;
    supervised_targets.masked_residue_indices = None;
    supervised_targets.chemistry = None;
    supervised_targets.chemistry_mask = None;
    let supervised = multi_task_loss_with_ms2_config(
        &multitask_output,
        &supervised_targets,
        FoundationLossWeights {
            rt: 1.0,
            ccs: 0.0,
            ms2: 1.0,
            masked_residue: 0.0,
            chemistry: 0.0,
            contrastive: 0.0,
        },
        ms2_loss,
    )?;
    let mut total = supervised.total.clone();
    let mut auxiliary_value = 0.0f64;

    if let Some(rt_robust) = paired_masked_pseudo_huber(
        &student_output.rt,
        supervised_targets.rt.as_ref(),
        supervised_targets.rt_mask.as_ref(),
        FOUNDATION_RT_ROBUST_DELTA_V0350,
    )? {
        let weighted = rt_robust.affine(FOUNDATION_RT_ROBUST_WEIGHT_V0350, 0.0)?;
        auxiliary_value += f64::from(weighted.to_scalar::<f32>()?);
        total = (total + weighted)?;
    }
    if let (Some(target), Some(mask)) = (
        supervised_targets.ms2.as_ref(),
        supervised_targets.ms2_mask.as_ref(),
    ) {
        let pearson = masked_pearson_loss(&student_output.ms2, target, mask, 1.0e-8)?;
        let weighted = pearson.affine(FOUNDATION_MS2_PEARSON_WEIGHT_V0350, 0.0)?;
        auxiliary_value += f64::from(weighted.to_scalar::<f32>()?);
        total = (total + weighted)?;
    }

    let fragment_context =
        FoundationFragmentContextBatchV0350::from_records(records, teacher_forward_config, device)?;
    let teacher_output =
        teacher.forward_detached_t(&batch.input, &batch.context, &fragment_context)?;
    let (teacher_rt_weight, teacher_ms2_weight) = teacher_anchor_weights(progress);
    let teacher_rt_mask = Tensor::ones((records.len(), 1), DType::F32, device)?;
    let teacher_rt = masked_mse(&student_output.rt, &teacher_output.rt, &teacher_rt_mask)?;
    let teacher_rt_weighted = teacher_rt.affine(teacher_rt_weight, 0.0)?;
    auxiliary_value += f64::from(teacher_rt_weighted.to_scalar::<f32>()?);
    total = (total + teacher_rt_weighted)?;

    let teacher_ms2_mask = cleavage_channel_mask(
        &batch.input.residue_mask,
        student.config().ms2_fragment_channels,
    )?;
    let teacher_ms2 = foundation_ms2_loss(
        &student_output.ms2,
        &teacher_output.ms2,
        &teacher_ms2_mask,
        ms2_loss,
    )?;
    let teacher_ms2_weighted = teacher_ms2.total.affine(teacher_ms2_weight, 0.0)?;
    auxiliary_value += f64::from(teacher_ms2_weighted.to_scalar::<f32>()?);
    total = (total + teacher_ms2_weighted)?;

    let (pair_target, pair_mask) =
        pair_interaction_targets(records, student.config().max_sequence_len, device)?;
    let pair_loss = masked_bce_with_logits(
        &student_output.pair_interaction_logits,
        &pair_target,
        &pair_mask,
    )?;
    let pair_weighted = pair_loss.affine(V050_PAIR_AUX_WEIGHT, 0.0)?;
    auxiliary_value += f64::from(pair_weighted.to_scalar::<f32>()?);
    total = (total + pair_weighted)?;

    let chemistry_target =
        chemistry_summary_targets(records, student.config().max_sequence_len, device)?;
    let chemistry_mask = Tensor::ones(
        (records.len(), FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM),
        DType::F32,
        device,
    )?;
    let chemistry_loss = masked_mse(
        &student_output.chemistry_summary,
        &chemistry_target,
        &chemistry_mask,
    )?;
    let chemistry_weighted = chemistry_loss.affine(V050_CHEMISTRY_SUMMARY_WEIGHT, 0.0)?;
    auxiliary_value += f64::from(chemistry_weighted.to_scalar::<f32>()?);
    total = (total + chemistry_weighted)?;

    let primary_value = f64::from(supervised.total.to_scalar::<f32>()?);
    let total_value = f64::from(total.to_scalar::<f32>()?);
    if !total_value.is_finite() {
        anyhow::bail!("v0.50 property loss is non-finite");
    }
    Ok((
        total,
        UpdateDiagnostics {
            total: total_value,
            primary: primary_value,
            auxiliary: auxiliary_value,
            teacher_rt_weight,
            teacher_ms2_weight,
        },
    ))
}

fn representation_loss_v0500(
    student: &PeptideFoundationV0500Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    ms2_loss: FoundationMs2LossConfig,
    contrastive_temperature: f64,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let views = collator.collate_views(records, device, seed)?;
    let first = student.forward_t(&views.first.input, &views.first.context, true)?;
    let second = student.forward_t(&views.second.input, &views.second.context, true)?;
    let batch_size = records.len();
    let dummy = Tensor::zeros((batch_size, 1), DType::F32, device)?;
    let dummy_ms2 = Tensor::zeros((batch_size, 1, 1), DType::F32, device)?;
    let first_output = FoundationMultiTaskOutput {
        foundation: FoundationOutput {
            residue_embeddings: first.representation.residue_embeddings.clone(),
            peptide_embedding: first.representation.global_embedding.clone(),
            residue_mask: first.representation.residue_mask.clone(),
            chemistry_targets: first.representation.chemistry_targets.clone(),
        },
        rt: dummy.clone(),
        ccs: dummy,
        ms2: dummy_ms2,
        residue_logits: first.residue_logits.clone(),
        chemistry_reconstruction: first.chemistry_reconstruction.clone(),
        contrastive_projection: first.contrastive_projection.clone(),
    };
    let mut targets = views.first.targets.clone();
    targets.rt = None;
    targets.rt_mask = None;
    targets.ccs = None;
    targets.ccs_mask = None;
    targets.ms2 = None;
    targets.ms2_mask = None;
    let reconstruction = multi_task_loss_with_ms2_config(
        &first_output,
        &targets,
        FoundationLossWeights {
            rt: 0.0,
            ccs: 0.0,
            ms2: 0.0,
            masked_residue: FOUNDATION_REPRESENTATION_MASKED_WEIGHT_V0350,
            chemistry: FOUNDATION_REPRESENTATION_CHEMISTRY_WEIGHT_V0350,
            contrastive: 0.0,
        },
        ms2_loss,
    )?;
    let contrastive = contrastive_info_nce_loss(
        &first.contrastive_projection,
        &second.contrastive_projection,
        contrastive_temperature,
    )?;
    Ok((reconstruction.total
        + contrastive.affine(FOUNDATION_REPRESENTATION_CONTRASTIVE_WEIGHT_V0350, 0.0)?)?)
}

fn normalize_rt_target(
    targets: &mut redeem_properties::foundation::FoundationTargets,
    normalization: &FoundationTargetNormalizationConfig,
) -> Result<()> {
    if let Some(rt) = targets.rt.take() {
        targets.rt = Some(normalization.rt.normalize_tensor(&rt)?);
    }
    Ok(())
}

fn cleavage_channel_mask(residue_mask: &Tensor, channels: usize) -> Result<Tensor> {
    let (batch, sequence) = residue_mask.dims2()?;
    if sequence < 2 {
        anyhow::bail!("v0.50 cleavage mask requires sequence length >=2");
    }
    let cleavages = sequence - 1;
    residue_mask
        .narrow(1, 0, cleavages)?
        .broadcast_mul(&residue_mask.narrow(1, 1, cleavages)?)?
        .unsqueeze(2)?
        .broadcast_as((batch, cleavages, channels))
        .map_err(Into::into)
}

fn chemistry_summary_targets(
    records: &[FoundationTrainingRecord],
    max_sequence_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(records.len() * FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM);
    for record in records {
        let sequence = &record.peptidoform.sequence;
        let length = sequence.chars().count().max(1);
        let mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        let acidic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'D' | 'E'))
            .count();
        let basic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'K' | 'R' | 'H'))
            .count();
        let aromatic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'F' | 'W' | 'Y' | 'H'))
            .count();
        let hydrophobic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y'))
            .count();
        let modifications = record.peptidoform.modifications.len();
        let terminal_modifications = record
            .peptidoform
            .modifications
            .iter()
            .filter(|modification| {
                matches!(
                    modification.site,
                    FoundationModificationSite::NTerm | FoundationModificationSite::CTerm
                )
            })
            .count();
        let denom = length as f32;
        values.extend_from_slice(&[
            (mass / 3000.0) as f32,
            length as f32 / max_sequence_len.max(1) as f32,
            acidic as f32 / denom,
            basic as f32 / denom,
            aromatic as f32 / denom,
            hydrophobic as f32 / denom,
            modifications as f32 / 8.0,
            terminal_modifications as f32 / 2.0,
        ]);
    }
    Tensor::from_vec(
        values,
        (records.len(), FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM),
        device,
    )
    .map_err(Into::into)
}

fn pair_interaction_targets(
    records: &[FoundationTrainingRecord],
    max_sequence_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let classes = FOUNDATION_V0500_PAIR_CLASS_COUNT;
    let mut target = vec![0.0f32; records.len() * max_sequence_len * max_sequence_len * classes];
    let mut mask = vec![0.0f32; target.len()];
    for (batch_index, record) in records.iter().enumerate() {
        let residues = record.peptidoform.sequence.chars().collect::<Vec<_>>();
        for i in 0..residues.len() {
            for j in 0..residues.len() {
                if i == j {
                    continue;
                }
                let left = residues[i];
                let right = residues[j];
                let left_acid = matches!(left, 'D' | 'E');
                let right_acid = matches!(right, 'D' | 'E');
                let left_basic = matches!(left, 'K' | 'R' | 'H');
                let right_basic = matches!(right, 'K' | 'R' | 'H');
                let left_donor =
                    matches!(left, 'K' | 'R' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let right_donor =
                    matches!(right, 'K' | 'R' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let left_acceptor =
                    matches!(left, 'D' | 'E' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let right_acceptor =
                    matches!(right, 'D' | 'E' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let left_aromatic = matches!(left, 'F' | 'W' | 'Y' | 'H');
                let right_aromatic = matches!(right, 'F' | 'W' | 'Y' | 'H');
                let left_hydrophobic =
                    matches!(left, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y');
                let right_hydrophobic =
                    matches!(right, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y');
                let terminal =
                    i == 0 || j == 0 || i + 1 == residues.len() || j + 1 == residues.len();
                let labels = [
                    (left_acid && right_basic) || (left_basic && right_acid),
                    (left_donor && right_acceptor) || (right_donor && left_acceptor),
                    left_aromatic && right_aromatic,
                    left_hydrophobic && right_hydrophobic,
                    left_basic && right_basic,
                    terminal,
                ];
                let base = (((batch_index * max_sequence_len + i) * max_sequence_len + j) * classes)
                    as usize;
                for class_index in 0..classes {
                    target[base + class_index] = if labels[class_index] { 1.0 } else { 0.0 };
                    mask[base + class_index] = 1.0;
                }
            }
        }
    }
    let shape = (records.len(), max_sequence_len, max_sequence_len, classes);
    Ok((
        Tensor::from_vec(target, shape, device)?,
        Tensor::from_vec(mask, shape, device)?,
    ))
}

fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!(
            "v0.50 masked MSE shape mismatch: prediction {:?}, target {:?}",
            prediction.dims(),
            target.dims()
        );
    }
    let mask = mask.broadcast_as(prediction.dims())?;
    let numerator = (prediction - target)?
        .sqr()?
        .broadcast_mul(&mask)?
        .sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

fn masked_bce_with_logits(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if logits.dims() != target.dims() {
        anyhow::bail!(
            "v0.50 BCE shape mismatch: logits {:?}, target {:?}",
            logits.dims(),
            target.dims()
        );
    }
    let positive = logits.relu()?;
    let linear = logits.broadcast_mul(target)?;
    let tail = logits.abs()?.affine(-1.0, 0.0)?.exp()?;
    let tail = (tail + 1.0)?.log()?;
    let loss = ((positive - linear)? + tail)?;
    let mask = mask.broadcast_as(logits.dims())?;
    let numerator = loss.broadcast_mul(&mask)?.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

fn paired_masked_pseudo_huber(
    prediction: &Tensor,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    delta: f64,
) -> Result<Option<Tensor>> {
    match (target, mask) {
        (Some(target), Some(mask)) => {
            if prediction.dims() != target.dims() {
                anyhow::bail!("v0.50 RT robust loss shape mismatch");
            }
            let mask = mask.broadcast_as(prediction.dims())?;
            let scaled = (prediction - target)?.affine(1.0 / delta, 0.0)?;
            let robust = (scaled.sqr()? + 1.0)?
                .sqrt()?
                .affine(delta * delta, -delta * delta)?;
            let numerator = robust.broadcast_mul(&mask)?.sum_all()?;
            let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
            Ok(Some(numerator.broadcast_div(&denominator)?))
        }
        (None, None) => Ok(None),
        _ => anyhow::bail!("v0.50 RT target/mask must be supplied together"),
    }
}

fn masked_pearson_loss(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!("v0.50 MS2 Pearson loss shape mismatch");
    }
    let (batch, cleavage, channels) = prediction.dims3()?;
    let mask = mask.broadcast_as(prediction.dims())?;
    let flat_prediction = prediction.reshape((batch, cleavage * channels))?;
    let flat_target = target.reshape((batch, cleavage * channels))?;
    let flat_mask = mask.reshape((batch, cleavage * channels))?;
    let count = flat_mask.sum(1)?.clamp(1.0, f64::INFINITY)?;
    let prediction_mean = flat_prediction
        .broadcast_mul(&flat_mask)?
        .sum(1)?
        .broadcast_div(&count)?;
    let target_mean = flat_target
        .broadcast_mul(&flat_mask)?
        .sum(1)?
        .broadcast_div(&count)?;
    let centered_prediction = flat_prediction
        .broadcast_sub(&prediction_mean.unsqueeze(1)?)?
        .broadcast_mul(&flat_mask)?;
    let centered_target = flat_target
        .broadcast_sub(&target_mean.unsqueeze(1)?)?
        .broadcast_mul(&flat_mask)?;
    let dot = centered_prediction
        .broadcast_mul(&centered_target)?
        .sum(1)?;
    let prediction_norm = (centered_prediction.sqr()?.sum(1)? + epsilon)?.sqrt()?;
    let target_norm = (centered_target.sqr()?.sum(1)? + epsilon)?.sqrt()?;
    let correlation = dot.broadcast_div(&prediction_norm.broadcast_mul(&target_norm)?)?;
    let per_spectrum = correlation.affine(-1.0, 1.0)?;
    let spectrum_present = flat_mask.sum(1)?.clamp(0.0, 1.0)?;
    let numerator = per_spectrum.broadcast_mul(&spectrum_present)?.sum_all()?;
    let denominator = spectrum_present.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

fn teacher_sentinel_fingerprint(
    teacher: &FoundationV0350TeacherV0500,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    teacher_forward_config: &redeem_properties::foundation::FoundationConfig,
    device: &Device,
) -> Result<f64> {
    let batch = collator.collate(records, device, 0)?;
    let fragment =
        FoundationFragmentContextBatchV0350::from_records(records, teacher_forward_config, device)?;
    let output = teacher.forward_detached_t(&batch.input, &batch.context, &fragment)?;
    let rt = f64::from(output.rt.sum_all()?.to_scalar::<f32>()?);
    let ccs = f64::from(output.ccs.sum_all()?.to_scalar::<f32>()?);
    let ms2 = f64::from(output.ms2.sum_all()?.to_scalar::<f32>()?);
    let embedding = f64::from(output.peptide_embedding.sum_all()?.to_scalar::<f32>()?);
    Ok(rt + ccs + ms2 + embedding)
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
        output_root.join("mobility_source_reliability_v0500.tsv"),
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
        output_root.join("mobility_consensus_supervision_v0500.tsv"),
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

fn mobility_consensus_loss_v0500(
    model: &PeptideFoundationV0500Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    loss_scales: MobilityLossScales,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let owned = examples
        .iter()
        .map(|example| {
            records
                .get(example.representative_index)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("v0.50 representative index outside corpus"))
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
    let output = model.forward_t(&batch.input, &batch.context, true)?;
    let predicted_mobility = output.mobility_native;
    let factor = bruker_ccs_factor(&batch.context)?;
    let predicted_ccs = predicted_mobility.broadcast_mul(&factor)?;
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

    Ok((((mobility_mse.affine(V038_MOBILITY_MSE_WEIGHT, 0.0)?
        + mobility_robust.affine(V038_MOBILITY_ROBUST_WEIGHT, 0.0)?)?
        + ccs_mse.affine(V038_CCS_AUX_MSE_WEIGHT, 0.0)?)?
        + ccs_robust.affine(V038_CCS_AUX_ROBUST_WEIGHT, 0.0)?)?)
}

fn calibrate_mobility_scales_v0500(
    model: &PeptideFoundationV0500Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    batch_size: usize,
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
        let output = model.forward_t(&batch.input, &batch.context, false)?;
        let predicted_mobility = output.mobility_native;
        let factor = bruker_ccs_factor(&batch.context)?;
        let predicted_ccs = predicted_mobility.broadcast_mul(&factor)?;
        let target_mobility = Tensor::from_vec(targets, (chunk.len(), 1), device)?;
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
        anyhow::bail!("v0.50 TRAIN loss calibration has no finite weighted labels");
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
        anyhow::bail!("v0.50 weighted scalar loss shape mismatch");
    }
    let scaled = (prediction - target)?.affine(1.0 / scale, 0.0)?;
    let effective = mask.broadcast_mul(weight)?;
    let numerator = scaled.sqr()?.broadcast_mul(&effective)?.sum_all()?;
    let denominator = effective.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

fn weighted_scaled_pseudo_huber(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    weight: &Tensor,
    scale: f64,
    delta: f64,
) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!("v0.50 weighted pseudo-Huber shape mismatch");
    }
    let scaled = (prediction - target)?.affine(1.0 / scale, 0.0)?;
    let robust = scaled
        .sqr()?
        .affine(1.0 / (delta * delta), 1.0)?
        .sqrt()?
        .affine(delta * delta, -delta * delta)?;
    let effective = mask.broadcast_mul(weight)?;
    let numerator = robust.broadcast_mul(&effective)?.sum_all()?;
    let denominator = effective.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

fn evaluate_raw_ccs_indices(
    model: &PeptideFoundationV0500Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    device: &Device,
) -> Result<f64> {
    let mut absolute = 0.0f64;
    let mut count = 0usize;
    for chunk in indices.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|&index| records[index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let output = model.forward_t(&batch.input, &batch.context, false)?;
        let factor = bruker_ccs_factor(&batch.context)?;
        let predicted_ccs = output
            .mobility_native
            .broadcast_mul(&factor)?
            .to_vec2::<f32>()?;
        for (row, record) in predicted_ccs.iter().zip(&owned) {
            let target = record
                .ccs
                .ok_or_else(|| anyhow::anyhow!("missing raw CCS target"))?;
            absolute += (f64::from(row[0]) - f64::from(target)).abs();
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.50 raw DEV CCS evaluation has zero records");
    }
    Ok(absolute / count as f64)
}

fn evaluate_consensus_targets(
    model: &PeptideFoundationV0500Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    batch_size: usize,
    device: &Device,
) -> Result<f64> {
    let mut absolute = 0.0f64;
    let mut count = 0usize;
    for chunk in examples.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|example| records[example.representative_index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let output = model.forward_t(&batch.input, &batch.context, false)?;
        let factor = bruker_ccs_factor(&batch.context)?.to_vec2::<f32>()?;
        let mobility = output.mobility_native.to_vec2::<f32>()?;
        for ((prediction, factor_row), example) in mobility.iter().zip(&factor).zip(chunk) {
            let predicted_ccs = f64::from(prediction[0] * factor_row[0]);
            let target_ccs = f64::from(example.target_mobility * factor_row[0]);
            absolute += (predicted_ccs - target_ccs).abs();
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.50 consensus DEV CCS evaluation has zero examples");
    }
    Ok(absolute / count as f64)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_dev(
    model: &PeptideFoundationV0500Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    property_indices: &[usize],
    ccs_indices: &[usize],
    consensus: &[MobilityConsensusExample],
    batch_size: usize,
    normalization: &FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    device: &Device,
) -> Result<DevMetrics> {
    let properties = evaluate_properties_v0500(
        model,
        collator,
        records,
        property_indices,
        batch_size,
        normalization,
        ms2_loss,
        device,
    )?;
    let raw_ccs_mae =
        evaluate_raw_ccs_indices(model, collator, records, ccs_indices, batch_size, device)?;
    let consensus_ccs_mae =
        evaluate_consensus_targets(model, collator, records, consensus, batch_size, device)?;
    let objective = dev_objective(properties, raw_ccs_mae, consensus_ccs_mae)?;
    Ok(DevMetrics {
        properties,
        raw_ccs_mae,
        consensus_ccs_mae,
        objective,
    })
}

fn evaluate_properties_v0500(
    model: &PeptideFoundationV0500Model,
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
        let output = model.forward_t(&batch.input, &batch.context, false)?;
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
            let components = foundation_ms2_loss(&output.ms2, target, mask, ms2_loss)?;
            ms2_objective_sum += f64::from(components.total.to_scalar::<f32>()?);
            ms2_objective_batches += 1;
            ms2_shape.accumulate(&output.ms2, target, mask)?;
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
    for ((predicted, truth), observed) in prediction.iter().zip(&target).zip(&mask) {
        if observed[0] <= 0.0 {
            continue;
        }
        let error = f64::from(predicted[0] - truth[0]);
        *abs_sum += error.abs();
        *sq_sum += error * error;
        *count += 1;
    }
    Ok(())
}

fn dev_objective(
    metrics: PropertyMetrics,
    raw_ccs_mae: f64,
    consensus_ccs_mae: f64,
) -> Result<f64> {
    let rt = metrics
        .rt_mae_native
        .ok_or_else(|| anyhow::anyhow!("DEV RT metric missing"))?;
    let cosine = metrics
        .ms2_mean_cosine
        .ok_or_else(|| anyhow::anyhow!("DEV MS2 cosine missing"))?;
    let angle = metrics
        .ms2_mean_spectral_angle
        .ok_or_else(|| anyhow::anyhow!("DEV MS2 spectral angle missing"))?;
    let pearson = metrics
        .ms2_mean_pearson
        .ok_or_else(|| anyhow::anyhow!("DEV MS2 Pearson missing"))?;
    let rt_ratio = rt / V035_RT_DEV_MAE;
    let raw_ratio = raw_ccs_mae / V038_RAW_CCS_DEV_MAE;
    let consensus_ratio = consensus_ccs_mae / V038_CONSENSUS_CCS_DEV_MAE;
    let cosine_error_ratio = (1.0 - cosine).max(0.0) / (1.0 - V035_MS2_COSINE_DEV);
    let angle_error_ratio = (1.0 - angle).max(0.0) / (1.0 - V035_MS2_SPECTRAL_ANGLE_DEV);
    let pearson_error_ratio = (1.0 - pearson).max(0.0) / (1.0 - V035_MS2_PEARSON_DEV);
    let objective = 0.25 * rt_ratio
        + 0.25 * raw_ratio
        + 0.15 * consensus_ratio
        + 0.12 * cosine_error_ratio
        + 0.12 * angle_error_ratio
        + 0.11 * pearson_error_ratio;
    if !objective.is_finite() {
        anyhow::bail!("v0.50 DEV objective is non-finite");
    }
    Ok(objective)
}

fn print_dev_metrics(label: &str, update: usize, metrics: DevMetrics) {
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

fn print_material_gate(metrics: DevMetrics, smoke_mode: bool) {
    let p = metrics.properties;
    let rt = p.rt_mae_native.unwrap_or(f64::INFINITY);
    let cosine = p.ms2_mean_cosine.unwrap_or(f64::NEG_INFINITY);
    let angle = p.ms2_mean_spectral_angle.unwrap_or(f64::NEG_INFINITY);
    let pearson = p.ms2_mean_pearson.unwrap_or(f64::NEG_INFINITY);
    let rt_target = rt <= V050_RT_TARGET_MAE;
    let raw_beats_v038 = metrics.raw_ccs_mae < V038_RAW_CCS_DEV_MAE;
    let raw_preferred = metrics.raw_ccs_mae <= V050_RAW_CCS_PREFERRED_MAE;
    let consensus_beats_v038 = metrics.consensus_ccs_mae < V038_CONSENSUS_CCS_DEV_MAE;
    let ms2_no_regression = cosine + V050_MS2_REGRESSION_TOLERANCE >= V035_MS2_COSINE_DEV
        && angle + V050_MS2_REGRESSION_TOLERANCE >= V035_MS2_SPECTRAL_ANGLE_DEV
        && pearson + V050_MS2_REGRESSION_TOLERANCE >= V035_MS2_PEARSON_DEV;
    let convincing_joint = rt_target && raw_beats_v038 && consensus_beats_v038 && ms2_no_regression;
    println!("v0500_rt_target_met\t{}", yes_no(rt_target));
    println!("v0500_raw_ccs_beats_v038\t{}", yes_no(raw_beats_v038));
    println!(
        "v0500_raw_ccs_preferred_10pct_gate_met\t{}",
        yes_no(raw_preferred)
    );
    println!(
        "v0500_consensus_ccs_beats_v038\t{}",
        yes_no(consensus_beats_v038)
    );
    println!(
        "v0500_ms2_no_material_regression\t{}",
        yes_no(ms2_no_regression)
    );
    println!(
        "v0500_convincing_joint_dev_gain\t{}",
        yes_no(convincing_joint)
    );
    println!(
        "v0500_holdout_eligible\t{}",
        yes_no(convincing_joint && !smoke_mode)
    );
    println!("v0500_finalize_required\tNO_explicit_handoff_review_required");
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
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
            let pnorm = pred_values
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let tnorm = target_values
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
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

fn read_v035_metadata(checkpoint: &Path) -> Result<V035ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.35 metadata {path:?}"))
}

fn read_v050_metadata(checkpoint: &Path) -> Result<V050Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.50 metadata {path:?}"))
}

#[allow(clippy::too_many_arguments)]
fn validate_v050_metadata(
    label: &str,
    metadata: &V050Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    supervision_fingerprint: &str,
    config: &PeptideFoundationV0500Config,
    batch_size: usize,
    cycles_per_epoch: usize,
    max_epochs: usize,
    seed: u64,
    learning_rate: f64,
) -> Result<()> {
    if metadata.version != V050_VERSION || metadata.objective != V050_OBJECTIVE {
        anyhow::bail!("v0.50 {label} checkpoint identity mismatch");
    }
    if metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        || metadata.mobility_supervision_fingerprint != supervision_fingerprint
    {
        anyhow::bail!("v0.50 {label} checkpoint provenance mismatch");
    }
    if metadata.v0500_config != *config
        || metadata.batch_size != batch_size
        || metadata.cycles_per_epoch != cycles_per_epoch
        || metadata.max_epochs != max_epochs
        || metadata.seed != seed
    {
        anyhow::bail!("v0.50 {label} checkpoint run-contract mismatch");
    }
    if (metadata.learning_rate - learning_rate).abs()
        > f64::EPSILON * 64.0 * learning_rate.abs().max(1.0)
    {
        anyhow::bail!("v0.50 {label} checkpoint learning-rate mismatch");
    }
    Ok(())
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V050Metadata,
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

fn filtered_sampling_config(
    base: &FoundationSamplingConfig,
    provenance: &[FoundationRecordProvenance],
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
    sampling: &FoundationSamplingConfig,
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
    batch_size: usize,
    requested_batches: usize,
) -> Result<usize> {
    if requested_batches == 0 {
        anyhow::bail!("v0.50 {label} requested zero validation batches");
    }
    let available_by_source =
        indices
            .iter()
            .fold(BTreeMap::<String, usize>::new(), |mut map, &index| {
                if let Some(item) = provenance.get(index) {
                    *map.entry(item.source_id.clone()).or_insert(0) += 1;
                }
                map
            });
    for batches in (1..=requested_batches).rev() {
        let target_records = batches.saturating_mul(batch_size);
        let quotas = weighted_quotas(
            &sampling.validation_source_weights,
            &available_by_source,
            target_records,
        );
        if quotas
            .iter()
            .all(|(source, quota)| available_by_source.get(source).copied().unwrap_or(0) >= *quota)
        {
            return Ok(batches);
        }
    }
    anyhow::bail!("v0.50 {label} cannot form one quota-feasible validation batch")
}

fn weighted_quotas(
    weights: &BTreeMap<String, f64>,
    available: &BTreeMap<String, usize>,
    target_records: usize,
) -> BTreeMap<String, usize> {
    if weights.is_empty() {
        return BTreeMap::new();
    }
    let total_weight = weights.values().sum::<f64>();
    if total_weight <= 0.0 {
        return BTreeMap::new();
    }
    let mut quotas = BTreeMap::<String, usize>::new();
    let mut fractional = Vec::new();
    let mut assigned = 0usize;
    for (source, weight) in weights {
        if !available.contains_key(source) || *weight <= 0.0 {
            continue;
        }
        let exact = target_records as f64 * *weight / total_weight;
        let floor = exact.floor() as usize;
        quotas.insert(source.clone(), floor);
        assigned += floor;
        fractional.push((exact - floor as f64, source.clone()));
    }
    fractional.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut remaining = target_records.saturating_sub(assigned);
    for (_, source) in fractional {
        if remaining == 0 {
            break;
        }
        *quotas.entry(source).or_insert(0) += 1;
        remaining -= 1;
    }
    quotas
}

fn require_plan_records(label: &str, plan: &FoundationSamplePlan, expected: usize) -> Result<()> {
    if plan.indices.len() < expected {
        anyhow::bail!(
            "v0.50 {label} sampling produced {} records, expected at least {expected}",
            plan.indices.len()
        );
    }
    Ok(())
}

fn print_sample_plan(label: &str, plan: &FoundationSamplePlan) {
    println!("sample_plan\tlabel={label}\trecords={}", plan.indices.len());
    for (source, count) in &plan.source_records {
        println!("sample_plan_source\tlabel={label}\tsource={source}\trecords={count}");
    }
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
    fn v0500_teacher_anchor_decays_but_never_disappears() {
        let start = teacher_anchor_weights(0.0);
        let middle = teacher_anchor_weights(V050_TEACHER_DECAY_FRACTION / 2.0);
        let end = teacher_anchor_weights(1.0);
        assert!((start.0 - V050_TEACHER_RT_START).abs() < 1.0e-12);
        assert!((start.1 - V050_TEACHER_MS2_START).abs() < 1.0e-12);
        assert!(middle.0 < start.0 && middle.0 > end.0);
        assert!(middle.1 < start.1 && middle.1 > end.1);
        assert!((end.0 - V050_TEACHER_RT_END).abs() < 1.0e-12);
        assert!((end.1 - V050_TEACHER_MS2_END).abs() < 1.0e-12);
    }

    #[test]
    fn v0500_affine_fit_recovers_linear_map() {
        let examples = vec![(1.0, 5.0), (2.0, 8.0), (3.0, 11.0), (4.0, 14.0)];
        let fit = fit_xy_affine(&examples).unwrap();
        assert!((fit.slope - 3.0).abs() < 1.0e-10);
        assert!((fit.intercept - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn v0500_consensus_prefers_reliable_source() {
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
    fn v0500_overlapping_views_collapse_to_project_families() {
        assert_eq!(source_family("pxd034128_11min"), "pxd034128");
        assert_eq!(
            source_family("pxd058337_60spd_fractionationgpf"),
            "pxd058337"
        );
        assert_eq!(source_family("ip2_bruker_human"), "ip2_bruker_human");
    }

    #[test]
    fn v0500_epoch_order_is_deterministic_and_unique_within_cycle() {
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

    #[test]
    fn v0500_dev_objective_rewards_joint_improvement() {
        let reference = PropertyMetrics {
            rt_mae_native: Some(V035_RT_DEV_MAE),
            ms2_mean_cosine: Some(V035_MS2_COSINE_DEV),
            ms2_mean_spectral_angle: Some(V035_MS2_SPECTRAL_ANGLE_DEV),
            ms2_mean_pearson: Some(V035_MS2_PEARSON_DEV),
            ..Default::default()
        };
        let baseline =
            dev_objective(reference, V038_RAW_CCS_DEV_MAE, V038_CONSENSUS_CCS_DEV_MAE).unwrap();
        let improved = PropertyMetrics {
            rt_mae_native: Some(4.2),
            ms2_mean_cosine: Some(0.91),
            ms2_mean_spectral_angle: Some(0.755),
            ms2_mean_pearson: Some(0.70),
            ..Default::default()
        };
        let improved = dev_objective(improved, 8.60, 8.70).unwrap();
        assert!(improved < baseline);
    }
}
