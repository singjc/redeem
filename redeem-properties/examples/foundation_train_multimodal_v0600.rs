//! ReDeeM v0.60 mobility-first conformational representation training.
//!
//! This executable intentionally closes the v0.37-v0.53 adapter/distillation lane.  It trains a
//! fresh deep chemistry/residue-pair representation end-to-end from TRAIN mobility evidence.
//! Stage 1 uses source-aligned raw mobility observations plus mass/charge-matched relative
//! mobility and exact-identity cross-source consistency. Stage 2 fine-tunes on the established
//! v0.38 TRAIN-only robust consensus targets. DEV selects checkpoints. TRAIN-HOLDOUT, historical
//! VALIDATION and historical TEST are never evaluated by this executable.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationAdamW,
    FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCollator,
    FoundationCollatorConfig, FoundationCorruptionConfig, FoundationLearningRateSchedule,
    FoundationModificationSite, FoundationOptimizerStep, FoundationPartition,
    FoundationRecordProvenance, FoundationScalarPhysicsBatchV0360,
    FoundationTargetNormalizationConfig, FoundationTrainingRecord,
    PeptideFoundationMultimodalV0350Config, PeptideFoundationV0500Config,
    PeptideFoundationV0600Config, PeptideFoundationV0600Model, RetentionTimeObjective,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0600, FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    FOUNDATION_V0600_STUDENT_NAMESPACE,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V060_VERSION: u32 = 600;
const V060_OBJECTIVE: &str = "v0600_mobility_first_charge_conformer_representation";
const V060_REPRESENTATION_STEPS_PER_EPOCH: usize = 3_072;
const V060_CONSENSUS_STEPS_PER_EPOCH: usize = 1_536;
const V060_SMOKE_STEPS: usize = 8;
const V060_SMOKE_DEV_RECORDS: usize = 256;
const V060_LOSS_CALIBRATION_RECORDS: usize = 2_048;
const V060_SMOKE_LOSS_CALIBRATION_RECORDS: usize = 64;
const V060_MAX_GRADIENT_NORM: f64 = 1.0;

// Established v0.38 TRAIN-only source alignment / consensus policy.
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

// v0.60 representation-pretraining terms.
const V060_RAW_VIEW_WEIGHT_SCALE: f64 = 0.75;
const V060_RELATIVE_MSE_WEIGHT: f64 = 0.50;
const V060_RELATIVE_ROBUST_WEIGHT: f64 = 0.50;
const V060_IDENTITY_CONSISTENCY_WEIGHT: f64 = 0.10;
const V060_HARD_PAIR_NEIGHBORHOOD: usize = 16;

// Frozen DEV references. v0.38 is benchmark only; it is never loaded as a teacher.
const V038_RAW_CCS_DEV_MAE: f64 = 8.806_119_29;
const V038_CONSENSUS_CCS_DEV_MAE: f64 = 8.922_126_00;
const V060_RAW_CCS_PREFERRED_MAE: f64 = 8.669;
const V060_CONSENSUS_MATERIAL_MAE: f64 = 8.80;

#[derive(Debug, Clone, Deserialize)]
struct V035ParentMetadata {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    completed_steps: usize,
    v0350_config: PeptideFoundationMultimodalV0350Config,
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

#[derive(Debug, Clone)]
struct RawMobilityExampleV0600 {
    record_index: usize,
    target_mobility: f32,
    weight: f32,
    identity_hash: u64,
    source_id: String,
    charge: u32,
    precursor_mz: f64,
}

#[derive(Debug, Clone)]
struct RepresentationExampleV0600 {
    anchor_raw: usize,
    hard_raw: usize,
    positive_raw: usize,
    positive_mask: f32,
    pair_hash: u64,
}

#[derive(Debug, Clone, Copy)]
struct MobilityLossScales {
    mobility: f64,
    ccs: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct DevMetricsV0600 {
    raw_ccs_mae: f64,
    consensus_ccs_mae: f64,
    objective: f64,
}

#[derive(Debug, Clone, Copy)]
struct UpdateDiagnosticsV0600 {
    total: f64,
    absolute: f64,
    relative: f64,
    identity_consistency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V060Metadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    mobility_supervision_fingerprint: String,
    representation_pair_fingerprint: String,
    parent_v0350_metadata_checkpoint: String,
    v0600_config: PeptideFoundationV0600Config,
    target_normalization: FoundationTargetNormalizationConfig,
    representation_epochs: usize,
    consensus_epochs: usize,
    representation_steps_per_epoch: usize,
    consensus_steps_per_epoch: usize,
    representation_batch_size: usize,
    consensus_batch_size: usize,
    seed: u64,
    learning_rate: f64,
    mobility_loss_scale_native: f64,
    ccs_aux_loss_scale_native: f64,
    completed_epochs: usize,
    completed_updates: usize,
    dev_objective: f64,
    smoke_mode: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 13 {
        anyhow::bail!(
            "usage: foundation_train_multimodal_v0600 RUN_V0260.yaml OUTPUT_DIR PARENT_V0350_FINAL [representation_epochs=6] [consensus_epochs=4] [representation_batch_size=8] [consensus_batch_size=32] [patience=3] [min_delta=0.002] [seed=20261060] [learning_rate=2e-5] [mode=smoke|train|resume|audit-export]"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_v0350_checkpoint = PathBuf::from(&args[3]);
    let mut representation_epochs = parse_or(&args, 4, 6usize)?;
    let mut consensus_epochs = parse_or(&args, 5, 4usize)?;
    let representation_batch_size = parse_or(&args, 6, 8usize)?;
    let consensus_batch_size = parse_or(&args, 7, 32usize)?;
    let patience = parse_or(&args, 8, 3usize)?;
    let min_delta = parse_or(&args, 9, 0.002f64)?;
    let seed = parse_or(&args, 10, 20_261_060u64)?;
    let learning_rate = parse_or(&args, 11, 2.0e-5f64)?;
    let mode = args.get(12).map(String::as_str).unwrap_or("train");
    let audit_export_only = mode == "audit-export";
    let smoke_mode = match mode {
        "smoke" => true,
        "train" | "resume" | "audit-export" => false,
        other => {
            anyhow::bail!(
                "unsupported v0.60 mode {other:?}; expected smoke, train, resume, or audit-export"
            )
        }
    };
    let resume_mode = mode == "resume";
    if smoke_mode {
        representation_epochs = 1;
        consensus_epochs = 0;
    }
    if representation_epochs == 0 || representation_batch_size < 2 || consensus_batch_size < 2 {
        anyhow::bail!("v0.60 requires representation_epochs>0 and batch sizes >=2");
    }
    if !smoke_mode && consensus_epochs == 0 {
        anyhow::bail!("v0.60 production training requires at least one consensus epoch");
    }
    if patience == 0 || !(min_delta > 0.0 && min_delta.is_finite()) {
        anyhow::bail!("v0.60 requires positive patience and min_delta");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("v0.60 learning_rate must be positive and finite");
    }

    if audit_export_only {
        let best = output_root.join("best");
        for file in ["model.safetensors", "metadata.yaml"] {
            if !best.join(file).is_file() {
                anyhow::bail!("v0.60 audit export is missing {:?}", best.join(file));
            }
        }
    } else if resume_mode {
        for checkpoint in ["latest", "best"] {
            let dir = output_root.join(checkpoint);
            for file in [
                "model.safetensors",
                "optimizer.safetensors",
                "metadata.yaml",
            ] {
                if !dir.join(file).is_file() {
                    anyhow::bail!("v0.60 resume is missing {:?}", dir.join(file));
                }
            }
        }
    } else if output_root.exists() {
        anyhow::bail!("v0.60 output directory must be fresh: {:?}", output_root);
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.60 requires a CUDA device")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;
    let train_indices = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .collect::<Vec<_>>();
    let dev_indices = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Validation)
        .map(|entry| entry.record_index)
        .collect::<Vec<_>>();
    let holdout_records_reserved = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .count();
    let mobility_train_indices = finite_mobility_ccs_indices(&corpus.records, &train_indices);
    let ccs_dev_indices_full = finite_mobility_ccs_indices(&corpus.records, &dev_indices);
    if mobility_train_indices.len() < consensus_batch_size
        || ccs_dev_indices_full.len() < consensus_batch_size
    {
        anyhow::bail!("v0.60 requires at least one TRAIN and DEV mobility batch");
    }

    let parent_metadata = read_v035_metadata_v0600(&parent_v0350_checkpoint)?;
    if parent_metadata.version != 350
        || parent_metadata.objective
            != "v0350_trainable_forward_representation_context_conditioned_ms2"
        || parent_metadata.completed_steps == 0
    {
        anyhow::bail!("v0.60 requires the selected v0.35 final metadata contract");
    }
    parent_metadata.v0350_config.validate()?;
    let current_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let current_benchmark_fingerprint =
        format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != current_corpus_fingerprint
        || parent_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint
    {
        anyhow::bail!("v0.60 corpus/benchmark differs from the frozen v0.35 contract");
    }

    let mut backbone = PeptideFoundationV0500Config::default();
    backbone.max_sequence_len = parent_metadata.v0350_config.forward().max_sequence_len;
    backbone.max_atoms_per_residue = parent_metadata.v0350_config.forward().max_atoms_per_residue;
    backbone.instrument_vocab_size = parent_metadata.v0350_config.forward().instrument_vocab_size;
    backbone.ms2_fragment_channels = parent_metadata.v0350_config.forward().ms2_fragment_channels;
    let v0600_config = PeptideFoundationV0600Config::fixed(backbone)?;
    let featurizer_config = v0600_config.backbone.featurizer_config();

    let mut max_prepared_len = 0usize;
    for &index in train_indices.iter().chain(&dev_indices) {
        let length = corpus.records[index].peptidoform.sequence.chars().count();
        max_prepared_len = max_prepared_len.max(length);
        if length > v0600_config.backbone.max_sequence_len {
            anyhow::bail!("v0.60 record {index} length {length} exceeds configured maximum");
        }
    }

    if audit_export_only {
        let export_path = PathBuf::from(
            env::var("REDEEM_CCS_AUDIT_EXPORT_TSV")
                .context("set REDEEM_CCS_AUDIT_EXPORT_TSV for v0.60 audit-export mode")?,
        );
        let best_dir = output_root.join("best");
        let best_metadata = read_v060_metadata(&best_dir)?;
        validate_v060_audit_metadata(
            &best_metadata,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &parent_v0350_checkpoint,
            &v0600_config,
        )?;
        if best_metadata.completed_updates == 0 {
            anyhow::bail!("v0.60 audit export refuses an unselected checkpoint");
        }
        let audit_collator = FoundationCollator::new(
            featurizer_config.clone(),
            FoundationCollatorConfig {
                retention_time_objective: parent_metadata.rt_objective,
                corruption: FoundationCorruptionConfig {
                    residue_mask_probability: 0.0,
                    chemistry_mask_probability: 0.0,
                },
            },
        )?;
        let mut audit_varmap = VarMap::new();
        let audit_vb = VarBuilder::from_varmap(&audit_varmap, DType::F32, &device);
        let audit_model = PeptideFoundationV0600Model::new(v0600_config.clone(), audit_vb)?;
        audit_varmap.load(best_dir.join("model.safetensors"))?;
        let (exported, mae) = export_raw_ccs_predictions_v0600(
            &audit_model,
            &audit_collator,
            &corpus.records,
            &corpus.provenance,
            &ccs_dev_indices_full,
            best_metadata.consensus_batch_size,
            &device,
            &export_path,
        )?;
        println!("v0600_audit_export_records\t{exported}");
        println!("v0600_audit_export_raw_ccs_mae\t{mae:.8}");
        println!("v0600_audit_export_path\t{}", export_path.display());
        println!("train_holdout_consumed\tNO");
        println!("historical_validation_consumed\tNO");
        println!("historical_test_consumed\tNO");
        return Ok(());
    }

    let train_supervision = build_train_consensus_supervision(
        &corpus.records,
        &corpus.provenance,
        &mobility_train_indices,
    )?;
    if train_supervision.examples.len() < consensus_batch_size {
        anyhow::bail!("v0.60 consensus TRAIN has fewer examples than consensus_batch_size");
    }
    let raw_examples = build_raw_mobility_examples_v0600(
        &corpus.records,
        &corpus.provenance,
        &mobility_train_indices,
        &train_supervision.source_supervision,
    )?;
    let representation_examples = build_representation_examples_v0600(&raw_examples)?;
    if representation_examples.len() < representation_batch_size {
        anyhow::bail!("v0.60 representation TRAIN has fewer hard-pair examples than batch size");
    }
    let representation_pair_fingerprint = format!(
        "fnv1a64:{:016x}",
        representation_fingerprint_v0600(&representation_examples)
    );
    let dev_consensus_full = build_partition_consensus_examples(
        &corpus.records,
        &corpus.provenance,
        &ccs_dev_indices_full,
        &train_supervision.source_supervision,
    )?;
    if dev_consensus_full.is_empty() {
        anyhow::bail!("v0.60 DEV consensus set is empty");
    }

    let ccs_dev_indices = if smoke_mode {
        deterministic_index_subset_v0600(
            &ccs_dev_indices_full,
            V060_SMOKE_DEV_RECORDS,
            seed ^ 0x6000_d3f0,
        )
    } else {
        ccs_dev_indices_full.clone()
    };
    let dev_consensus = if smoke_mode {
        deterministic_consensus_subset_v0600(
            &dev_consensus_full,
            V060_SMOKE_DEV_RECORDS,
            seed ^ 0x6000_c053,
        )
    } else {
        dev_consensus_full.clone()
    };

    let representation_steps_per_epoch = if smoke_mode {
        V060_SMOKE_STEPS
    } else {
        (representation_examples.len() / representation_batch_size)
            .min(V060_REPRESENTATION_STEPS_PER_EPOCH)
            .max(1)
    };
    let consensus_steps_per_epoch = if smoke_mode {
        0
    } else {
        (train_supervision.examples.len() / consensus_batch_size)
            .min(V060_CONSENSUS_STEPS_PER_EPOCH)
            .max(1)
    };
    let total_epochs = representation_epochs + consensus_epochs;
    let max_updates = representation_epochs.saturating_mul(representation_steps_per_epoch)
        + consensus_epochs.saturating_mul(consensus_steps_per_epoch);

    let clean_collator = FoundationCollator::new(
        featurizer_config,
        FoundationCollatorConfig {
            retention_time_objective: parent_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let target_normalization = parent_metadata.target_normalization;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationV0600Model::new(v0600_config.clone(), vb)?;
    let optimizer_prefix = format!("{FOUNDATION_V0600_STUDENT_NAMESPACE}.");
    let mut optimizer = FoundationAdamW::new_for_prefixes(
        &varmap,
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
        warmup_steps: 500u64.min(max_updates.saturating_sub(1) as u64),
        total_steps: max_updates.max(1) as u64,
        min_lr_ratio: 0.10,
    };

    let calibration_limit = if smoke_mode {
        V060_SMOKE_LOSS_CALIBRATION_RECORDS
    } else {
        V060_LOSS_CALIBRATION_RECORDS
    };
    let calibration_order = deterministic_example_order(
        &train_supervision.examples,
        calibration_limit
            .min(train_supervision.examples.len())
            .max(consensus_batch_size),
        0,
        seed ^ 0x6000_4c4f_5353,
    );
    let calibration_examples = calibration_order
        .iter()
        .map(|&index| train_supervision.examples[index].clone())
        .collect::<Vec<_>>();
    let loss_scales = calibrate_mobility_scales_v0600(
        &model,
        &clean_collator,
        &corpus.records,
        &calibration_examples,
        consensus_batch_size,
        &device,
    )?;

    if !resume_mode {
        fs::create_dir_all(&output_root)?;
        write_mobility_supervision_summary(&output_root, &train_supervision)?;
        write_representation_summary_v0600(
            &output_root,
            &raw_examples,
            &representation_examples,
            &representation_pair_fingerprint,
        )?;
    }

    println!("v0600_version\tv0.60-mobility-first-charge-conformer-representation");
    println!("objective\t{V060_OBJECTIVE}");
    println!("architecture\t{FOUNDATION_MULTIMODAL_ARCHITECTURE_V0600}");
    println!("device\t{:?}", device);
    println!("run_mode\t{mode}");
    println!(
        "parent_v0350_metadata_checkpoint\t{}",
        parent_v0350_checkpoint.display()
    );
    println!("historical_weight_initialization\tNONE_fresh_student_v060");
    println!("v038_role\tDEV_benchmark_only_not_loaded_not_distilled");
    println!("student_namespace\t{FOUNDATION_V0600_STUDENT_NAMESPACE}");
    println!("prepared_max_sequence_len\t{max_prepared_len}");
    println!("train_records\t{}", train_indices.len());
    println!("dev_records\t{}", dev_indices.len());
    println!("holdout_records_reserved_not_evaluated\t{holdout_records_reserved}");
    println!("raw_train_mobility_records\t{}", raw_examples.len());
    println!(
        "representation_hard_pair_examples\t{}",
        representation_examples.len()
    );
    println!(
        "consensus_train_examples\t{}",
        train_supervision.examples.len()
    );
    println!("dev_raw_ccs_records\t{}", ccs_dev_indices.len());
    println!("dev_consensus_examples\t{}", dev_consensus.len());
    println!("representation_epochs\t{representation_epochs}");
    println!("consensus_epochs\t{consensus_epochs}");
    println!("representation_steps_per_epoch\t{representation_steps_per_epoch}");
    println!("consensus_steps_per_epoch\t{consensus_steps_per_epoch}");
    println!("representation_batch_size\t{representation_batch_size}");
    println!("consensus_batch_size\t{consensus_batch_size}");
    println!("max_optimizer_updates\t{max_updates}");
    println!("base_learning_rate\t{learning_rate}");
    println!("optimizer_scope\tstudent_v060.*_fully_trainable");
    println!("optimizer_variable_count\t{}", optimizer.variable_count());
    println!("mobility_loss_scale_native\t{:.8}", loss_scales.mobility);
    println!("ccs_aux_loss_scale_native\t{:.8}", loss_scales.ccs);
    println!("representation_objective\tabsolute_source_aligned_plus_mass_charge_matched_delta_mobility_plus_exact_identity_consistency");
    println!("consensus_objective\tv038_train_only_affine_family_deduplicated_reliability_weighted_consensus");
    println!(
        "charge_representation\t{} latent charge-carrier slots",
        v0600_config.charge_slots
    );
    println!(
        "conformer_representation\t{} latent conformer slots",
        v0600_config.conformer_slots
    );
    println!("dev_reference_raw_ccs_mae\t{V038_RAW_CCS_DEV_MAE}");
    println!("dev_reference_consensus_ccs_mae\t{V038_CONSENSUS_CCS_DEV_MAE}");
    println!("preferred_raw_ccs_gate\t{V060_RAW_CCS_PREFERRED_MAE}");
    println!("material_consensus_ccs_gate\t{V060_CONSENSUS_MATERIAL_MAE}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let initial_raw = evaluate_raw_ccs_v0600(
        &model,
        &clean_collator,
        &corpus.records,
        &ccs_dev_indices,
        consensus_batch_size,
        &device,
    )?;
    let initial_consensus = evaluate_consensus_ccs_v0600(
        &model,
        &clean_collator,
        &corpus.records,
        &dev_consensus,
        consensus_batch_size,
        &device,
    )?;
    let initial = DevMetricsV0600 {
        raw_ccs_mae: initial_raw,
        consensus_ccs_mae: initial_consensus,
        objective: combined_dev_objective_v0600(initial_raw, initial_consensus),
    };
    print_dev_v0600("train_dev_initial", 0, initial);

    let supervision_fingerprint = format!("fnv1a64:{:016x}", train_supervision.fingerprint);
    let metadata_for =
        |completed_epochs: usize, completed_updates: usize, dev_objective: f64| V060Metadata {
            version: V060_VERSION,
            objective: V060_OBJECTIVE.to_string(),
            architecture: FOUNDATION_MULTIMODAL_ARCHITECTURE_V0600.to_string(),
            corpus_fingerprint: current_corpus_fingerprint.clone(),
            benchmark_manifest_fingerprint: current_benchmark_fingerprint.clone(),
            mobility_supervision_fingerprint: supervision_fingerprint.clone(),
            representation_pair_fingerprint: representation_pair_fingerprint.clone(),
            parent_v0350_metadata_checkpoint: parent_v0350_checkpoint.display().to_string(),
            v0600_config: v0600_config.clone(),
            target_normalization,
            representation_epochs,
            consensus_epochs,
            representation_steps_per_epoch,
            consensus_steps_per_epoch,
            representation_batch_size,
            consensus_batch_size,
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
    let mut best_objective = initial.objective;
    let mut consensus_stale_epochs = 0usize;
    if resume_mode {
        let latest = read_v060_metadata(&output_root.join("latest"))?;
        let best = read_v060_metadata(&output_root.join("best"))?;
        validate_v060_metadata(
            &latest,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &supervision_fingerprint,
            &representation_pair_fingerprint,
            &parent_v0350_checkpoint,
            &v0600_config,
            representation_epochs,
            consensus_epochs,
            representation_batch_size,
            consensus_batch_size,
            seed,
            learning_rate,
        )?;
        validate_v060_metadata(
            &best,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &supervision_fingerprint,
            &representation_pair_fingerprint,
            &parent_v0350_checkpoint,
            &v0600_config,
            representation_epochs,
            consensus_epochs,
            representation_batch_size,
            consensus_batch_size,
            seed,
            learning_rate,
        )?;
        varmap.load(output_root.join("latest/model.safetensors"))?;
        optimizer.load_safetensors(output_root.join("latest/optimizer.safetensors"))?;
        start_epoch = latest.completed_epochs.saturating_add(1);
        global_step = latest.completed_updates;
        best_epoch = best.completed_epochs;
        best_step = best.completed_updates;
        best_objective = best.dev_objective;
        if latest.completed_epochs > representation_epochs {
            consensus_stale_epochs = latest
                .completed_epochs
                .saturating_sub(best.completed_epochs.max(representation_epochs));
        }
        println!("v0600_resume\tlatest_epoch={}\tlatest_update={}\tbest_epoch={}\tbest_update={}\tbest_dev_objective={:.8}",
            latest.completed_epochs, latest.completed_updates, best_epoch, best_step, best_objective);
    } else {
        save_checkpoint_v0600(
            &output_root.join("initial"),
            &varmap,
            &optimizer,
            &metadata_for(0, 0, initial.objective),
        )?;
        save_checkpoint_v0600(
            &output_root.join("best"),
            &varmap,
            &optimizer,
            &metadata_for(0, 0, initial.objective),
        )?;
    }

    let mut stopped_early = false;
    for epoch in start_epoch..=total_epochs {
        let representation_stage = epoch <= representation_epochs;
        let stage = if representation_stage {
            "representation"
        } else {
            "consensus"
        };
        let steps = if representation_stage {
            representation_steps_per_epoch
        } else {
            consensus_steps_per_epoch
        };
        let batch_size = if representation_stage {
            representation_batch_size
        } else {
            consensus_batch_size
        };
        println!("v0600_epoch\tstage=start\tepoch={epoch}\tphase={stage}\tsteps={steps}\tbatch_size={batch_size}");

        if representation_stage {
            let needed = steps.saturating_mul(batch_size);
            let order = deterministic_representation_order_v0600(
                &representation_examples,
                needed,
                epoch as u64,
                seed ^ 0x6000_7265_7072_6573,
            );
            for local_step in 0..steps {
                global_step += 1;
                let lr = lr_schedule
                    .learning_rate(learning_rate, global_step.saturating_sub(1) as u64)?;
                optimizer.set_learning_rate(lr)?;
                let offset = local_step.saturating_mul(batch_size);
                let selected = order[offset..offset + batch_size]
                    .iter()
                    .map(|&idx| representation_examples[idx].clone())
                    .collect::<Vec<_>>();
                let (loss, diagnostics) = representation_loss_v0600(
                    &model,
                    &clean_collator,
                    &corpus.records,
                    &raw_examples,
                    &selected,
                    loss_scales,
                    seed ^ (global_step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    &device,
                )?;
                let update = backward_step_v0600(
                    &loss,
                    &mut optimizer,
                    &varmap,
                    smoke_mode && global_step == 1,
                )?;
                if global_step == 1 || global_step % 100 == 0 || local_step + 1 == steps {
                    println!("v0600_train\tepoch={epoch}\tphase=representation\tstep={global_step}\tepoch_step={}\tlr={:.8}\ttotal={:.6}\tabsolute={:.6}\trelative={:.6}\tidentity_consistency={:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                        local_step + 1, update.learning_rate, diagnostics.total,
                        diagnostics.absolute, diagnostics.relative, diagnostics.identity_consistency,
                        update.gradient_norm, update.gradient_scale);
                }
            }
        } else {
            let needed = steps.saturating_mul(batch_size);
            let order = deterministic_example_order(
                &train_supervision.examples,
                needed,
                epoch as u64,
                seed ^ 0x6000_636f_6e73_656e,
            );
            for local_step in 0..steps {
                global_step += 1;
                let lr = lr_schedule
                    .learning_rate(learning_rate, global_step.saturating_sub(1) as u64)?;
                optimizer.set_learning_rate(lr)?;
                let offset = local_step.saturating_mul(batch_size);
                let selected = order[offset..offset + batch_size]
                    .iter()
                    .map(|&idx| train_supervision.examples[idx].clone())
                    .collect::<Vec<_>>();
                let (loss, absolute_value) = consensus_loss_v0600(
                    &model,
                    &clean_collator,
                    &corpus.records,
                    &selected,
                    loss_scales,
                    seed ^ (global_step as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
                    &device,
                )?;
                let update = backward_step_v0600(&loss, &mut optimizer, &varmap, false)?;
                if global_step % 100 == 0 || local_step + 1 == steps {
                    println!("v0600_train\tepoch={epoch}\tphase=consensus\tstep={global_step}\tepoch_step={}\tlr={:.8}\ttotal={:.6}\tabsolute={:.6}\trelative=0.000000\tidentity_consistency=0.000000\tgradient_norm={:.6}\tgradient_scale={:.6}",
                        local_step + 1, update.learning_rate, absolute_value, absolute_value,
                        update.gradient_norm, update.gradient_scale);
                }
            }
        }

        let raw_dev = evaluate_raw_ccs_v0600(
            &model,
            &clean_collator,
            &corpus.records,
            &ccs_dev_indices,
            consensus_batch_size,
            &device,
        )?;
        let consensus_dev = evaluate_consensus_ccs_v0600(
            &model,
            &clean_collator,
            &corpus.records,
            &dev_consensus,
            consensus_batch_size,
            &device,
        )?;
        let dev = DevMetricsV0600 {
            raw_ccs_mae: raw_dev,
            consensus_ccs_mae: consensus_dev,
            objective: combined_dev_objective_v0600(raw_dev, consensus_dev),
        };
        print_dev_v0600("train_dev", global_step, dev);
        let improved = best_objective - dev.objective > min_delta;
        println!("train_dev_objective\tepoch={epoch}\tphase={stage}\tupdate={global_step}\tvalue={:.8}\tprevious_best={best_objective:.8}\timproved={improved}", dev.objective);
        save_checkpoint_v0600(
            &output_root.join("latest"),
            &varmap,
            &optimizer,
            &metadata_for(epoch, global_step, dev.objective),
        )?;
        if improved {
            best_objective = dev.objective;
            best_epoch = epoch;
            best_step = global_step;
            consensus_stale_epochs = 0;
            save_checkpoint_v0600(
                &output_root.join("best"),
                &varmap,
                &optimizer,
                &metadata_for(epoch, global_step, dev.objective),
            )?;
            println!("v0600_best_checkpoint\tepoch={best_epoch}\tphase={stage}\tupdate={best_step}\tdev_objective={best_objective:.8}");
        } else if !representation_stage {
            consensus_stale_epochs += 1;
        }
        println!("v0600_epoch\tstage=complete\tepoch={epoch}\tphase={stage}\tupdate={global_step}\tconsensus_stale_epochs={consensus_stale_epochs}");
        if !representation_stage && consensus_stale_epochs >= patience {
            stopped_early = true;
            println!("v0600_early_stop\tepoch={epoch}\tupdate={global_step}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_update={best_step}\tbest_dev_objective={best_objective:.8}");
            break;
        }
    }

    varmap.load(output_root.join("best/model.safetensors"))?;
    let best_raw = evaluate_raw_ccs_v0600(
        &model,
        &clean_collator,
        &corpus.records,
        &ccs_dev_indices,
        consensus_batch_size,
        &device,
    )?;
    let best_consensus = evaluate_consensus_ccs_v0600(
        &model,
        &clean_collator,
        &corpus.records,
        &dev_consensus,
        consensus_batch_size,
        &device,
    )?;
    let best_dev = DevMetricsV0600 {
        raw_ccs_mae: best_raw,
        consensus_ccs_mae: best_consensus,
        objective: combined_dev_objective_v0600(best_raw, best_consensus),
    };
    print_dev_v0600("best_train_dev", best_step, best_dev);
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!("v0600_training_complete\tbest_epoch={best_epoch}\tbest_update={best_step}\tbest_dev_objective={:.8}\tstopped_early={stopped_early}\tsmoke_mode={smoke_mode}", best_dev.objective);
    print_material_gate_v0600(best_dev, smoke_mode);
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
            .ok_or_else(|| anyhow::anyhow!("v0.60 record index {index} outside corpus"))?;
        let Some(target) = record
            .context
            .ion_mobility
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("v0.60 missing provenance for record {index}"))?;
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
                .ok_or_else(|| anyhow::anyhow!("v0.60 family has no representative"))?;
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
            .ok_or_else(|| anyhow::anyhow!("v0.60 consensus identity has no representative"))?;
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
        output_root.join("mobility_source_reliability_v0600.tsv"),
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
        output_root.join("mobility_consensus_supervision_v0600.tsv"),
        summary,
    )?;
    Ok(())
}

fn build_raw_mobility_examples_v0600(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
    source_supervision: &BTreeMap<String, SourceSupervision>,
) -> Result<Vec<RawMobilityExampleV0600>> {
    let mut examples = Vec::with_capacity(indices.len());
    for &record_index in indices {
        let record = records
            .get(record_index)
            .ok_or_else(|| anyhow::anyhow!("v0.60 raw record index outside corpus"))?;
        let Some(raw_target) = record
            .context
            .ion_mobility
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        let Some(charge) = record.context.charge.filter(|value| *value > 0) else {
            continue;
        };
        let Some(mz) = record
            .context
            .precursor_mz
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        let source_id = provenance
            .get(record_index)
            .ok_or_else(|| anyhow::anyhow!("v0.60 missing provenance for raw record"))?
            .source_id
            .clone();
        let source = source_supervision.get(&source_id);
        let affine = source.map(|entry| entry.affine).unwrap_or(AffineFit {
            intercept: 0.0,
            slope: 1.0,
        });
        let reliability = source.map(|entry| entry.reliability).unwrap_or(0.60);
        let adjusted = affine.intercept + affine.slope * f64::from(raw_target);
        if !(adjusted.is_finite() && adjusted > 0.0) {
            continue;
        }
        examples.push(RawMobilityExampleV0600 {
            record_index,
            target_mobility: adjusted as f32,
            weight: (V060_RAW_VIEW_WEIGHT_SCALE * reliability).clamp(0.25, 1.0) as f32,
            identity_hash: stable_hash64(peptidoform_charge_key(record).as_bytes()),
            source_id,
            charge: charge as u32,
            precursor_mz: f64::from(mz),
        });
    }
    examples.sort_by(|a, b| {
        a.charge
            .cmp(&b.charge)
            .then_with(|| a.precursor_mz.total_cmp(&b.precursor_mz))
            .then_with(|| a.identity_hash.cmp(&b.identity_hash))
            .then_with(|| a.record_index.cmp(&b.record_index))
    });
    Ok(examples)
}

fn build_representation_examples_v0600(
    raw: &[RawMobilityExampleV0600],
) -> Result<Vec<RepresentationExampleV0600>> {
    if raw.len() < 2 {
        anyhow::bail!("v0.60 raw mobility supervision too small");
    }
    let mut by_identity = BTreeMap::<u64, Vec<usize>>::new();
    for (index, example) in raw.iter().enumerate() {
        by_identity
            .entry(example.identity_hash)
            .or_default()
            .push(index);
    }
    let mut result = Vec::with_capacity(raw.len());
    for anchor in 0..raw.len() {
        let a = &raw[anchor];
        let mut best_hard: Option<(f64, usize)> = None;
        let lo = anchor.saturating_sub(V060_HARD_PAIR_NEIGHBORHOOD);
        let hi = (anchor + V060_HARD_PAIR_NEIGHBORHOOD + 1).min(raw.len());
        for candidate in lo..hi {
            if candidate == anchor {
                continue;
            }
            let b = &raw[candidate];
            if b.charge != a.charge || b.identity_hash == a.identity_hash {
                continue;
            }
            let delta = (b.precursor_mz - a.precursor_mz).abs();
            if best_hard.is_none_or(|(best, _)| delta < best) {
                best_hard = Some((delta, candidate));
            }
        }
        let Some((_, hard_raw)) = best_hard else {
            continue;
        };
        let positive_raw = by_identity.get(&a.identity_hash).and_then(|candidates| {
            candidates
                .iter()
                .copied()
                .find(|&idx| idx != anchor && raw[idx].source_id != a.source_id)
        });
        let (positive_raw, positive_mask) = positive_raw.map_or((anchor, 0.0), |idx| (idx, 1.0));
        let pair_hash =
            mix64(a.identity_hash ^ raw[hard_raw].identity_hash.rotate_left(17) ^ anchor as u64);
        result.push(RepresentationExampleV0600 {
            anchor_raw: anchor,
            hard_raw,
            positive_raw,
            positive_mask,
            pair_hash,
        });
    }
    if result.is_empty() {
        anyhow::bail!("v0.60 could not build mass/charge matched hard pairs");
    }
    result.sort_by_key(|entry| (entry.pair_hash, entry.anchor_raw));
    Ok(result)
}

fn representation_fingerprint_v0600(examples: &[RepresentationExampleV0600]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for example in examples {
        for value in [
            example.anchor_raw as u64,
            example.hard_raw as u64,
            example.positive_raw as u64,
            example.pair_hash,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash ^= if example.positive_mask > 0.0 { 1 } else { 0 };
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn deterministic_representation_order_v0600(
    examples: &[RepresentationExampleV0600],
    needed: usize,
    epoch: u64,
    seed: u64,
) -> Vec<usize> {
    let mut order = (0..examples.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| mix64(examples[index].pair_hash ^ seed ^ epoch.rotate_left(13)));
    if needed <= order.len() {
        order.truncate(needed);
        return order;
    }
    let base = order.clone();
    let mut cycle = 0u64;
    while order.len() < needed {
        cycle += 1;
        let mut extra = base.clone();
        extra.sort_by_key(|&index| {
            mix64(
                examples[index].pair_hash
                    ^ seed
                    ^ epoch
                    ^ cycle.wrapping_mul(0x9e37_79b9_7f4a_7c15),
            )
        });
        let take = (needed - order.len()).min(extra.len());
        order.extend_from_slice(&extra[..take]);
    }
    order
}

fn write_representation_summary_v0600(
    output_root: &Path,
    raw: &[RawMobilityExampleV0600],
    pairs: &[RepresentationExampleV0600],
    fingerprint: &str,
) -> Result<()> {
    let positives = pairs
        .iter()
        .filter(|entry| entry.positive_mask > 0.0)
        .count();
    let mean_mz_delta = if pairs.is_empty() {
        0.0
    } else {
        pairs
            .iter()
            .map(|entry| {
                (raw[entry.anchor_raw].precursor_mz - raw[entry.hard_raw].precursor_mz).abs()
            })
            .sum::<f64>()
            / pairs.len() as f64
    };
    fs::write(
        output_root.join("mobility_representation_supervision_v0600.tsv"),
        format!("raw_examples\t{}\nhard_pairs\t{}\ncross_source_identity_pairs\t{}\nmean_hard_pair_abs_mz_delta\t{:.8}\nrepresentation_pair_fingerprint\t{}\n",
            raw.len(), pairs.len(), positives, mean_mz_delta, fingerprint),
    )?;
    Ok(())
}

fn bruker_ccs_factor_v0600(
    context: &redeem_properties::foundation::PrecursorContextBatch,
) -> Result<Tensor> {
    // Exact v0.38 Bruker native-mobility -> CCS conversion. The v0.60 representation is new;
    // the physical unit conversion remains deliberately unchanged for apples-to-apples DEV.
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

fn predicted_mobility_and_ccs_v0600(
    model: &PeptideFoundationV0600Model,
    batch: &redeem_properties::foundation::FoundationTrainingBatch,
    records: &[FoundationTrainingRecord],
    train: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let physics = FoundationScalarPhysicsBatchV0360::from_records(
        records,
        model.config().backbone.max_sequence_len,
        batch.input.residue_mask.device(),
    )?;
    let output = model.forward_t(&batch.input, &batch.context, &physics, train)?;
    let factor = bruker_ccs_factor_v0600(&batch.context)?;
    let ccs = output.mobility_native.broadcast_mul(&factor)?;
    Ok((output.mobility_native, ccs, output.mobility_latent))
}

fn representation_loss_v0600(
    model: &PeptideFoundationV0600Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    raw: &[RawMobilityExampleV0600],
    examples: &[RepresentationExampleV0600],
    scales: MobilityLossScales,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, UpdateDiagnosticsV0600)> {
    let n = examples.len();
    let mut owned = Vec::with_capacity(3 * n);
    let mut absolute_targets = Vec::with_capacity(2 * n);
    let mut absolute_weights = Vec::with_capacity(2 * n);
    let mut relative_targets = Vec::with_capacity(n);
    let mut relative_weights = Vec::with_capacity(n);
    let mut positive_mask = Vec::with_capacity(n);
    for entry in examples {
        let anchor = &raw[entry.anchor_raw];
        let hard = &raw[entry.hard_raw];
        let positive = &raw[entry.positive_raw];
        owned.push(records[anchor.record_index].clone());
        absolute_targets.push(anchor.target_mobility);
        absolute_weights.push(anchor.weight);
        relative_targets.push(anchor.target_mobility - hard.target_mobility);
        relative_weights.push(0.5 * (anchor.weight + hard.weight));
        positive_mask.push(entry.positive_mask);
        // hard and positive records are appended after anchors below to preserve contiguous slices.
        let _ = positive;
    }
    for entry in examples {
        let hard = &raw[entry.hard_raw];
        owned.push(records[hard.record_index].clone());
        absolute_targets.push(hard.target_mobility);
        absolute_weights.push(hard.weight);
    }
    for entry in examples {
        let positive = &raw[entry.positive_raw];
        owned.push(records[positive.record_index].clone());
    }
    let batch = collator.collate(&owned, device, seed)?;
    let (mobility, ccs, latent) = predicted_mobility_and_ccs_v0600(model, &batch, &owned, true)?;
    let absolute_mobility = mobility.narrow(0, 0, 2 * n)?;
    let absolute_ccs = ccs.narrow(0, 0, 2 * n)?;
    let target_mobility = Tensor::from_vec(absolute_targets, (2 * n, 1), device)?;
    let factor = bruker_ccs_factor_v0600(&batch.context)?.narrow(0, 0, 2 * n)?;
    let target_ccs = target_mobility.broadcast_mul(&factor)?;
    let weight = Tensor::from_vec(absolute_weights, (2 * n, 1), device)?;
    let mask = Tensor::ones((2 * n, 1), DType::F32, device)?;
    let mobility_mse = weighted_scaled_mse_v0600(
        &absolute_mobility,
        &target_mobility,
        &mask,
        &weight,
        scales.mobility,
    )?;
    let mobility_robust = weighted_scaled_pseudo_huber_v0600(
        &absolute_mobility,
        &target_mobility,
        &mask,
        &weight,
        scales.mobility,
        FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    )?;
    let ccs_mse =
        weighted_scaled_mse_v0600(&absolute_ccs, &target_ccs, &mask, &weight, scales.ccs)?;
    let ccs_robust = weighted_scaled_pseudo_huber_v0600(
        &absolute_ccs,
        &target_ccs,
        &mask,
        &weight,
        scales.ccs,
        FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    )?;
    let absolute = (((mobility_mse.affine(V038_MOBILITY_MSE_WEIGHT, 0.0)?
        + mobility_robust.affine(V038_MOBILITY_ROBUST_WEIGHT, 0.0)?)?
        + ccs_mse.affine(V038_CCS_AUX_MSE_WEIGHT, 0.0)?)?
        + ccs_robust.affine(V038_CCS_AUX_ROBUST_WEIGHT, 0.0)?)?;

    let anchor_mobility = mobility.narrow(0, 0, n)?;
    let hard_mobility = mobility.narrow(0, n, n)?;
    let predicted_delta = (&anchor_mobility - &hard_mobility)?;
    let target_delta = Tensor::from_vec(relative_targets, (n, 1), device)?;
    let pair_weight = Tensor::from_vec(relative_weights, (n, 1), device)?;
    let pair_mask = Tensor::ones((n, 1), DType::F32, device)?;
    let relative_mse = weighted_scaled_mse_v0600(
        &predicted_delta,
        &target_delta,
        &pair_mask,
        &pair_weight,
        scales.mobility,
    )?;
    let relative_robust = weighted_scaled_pseudo_huber_v0600(
        &predicted_delta,
        &target_delta,
        &pair_mask,
        &pair_weight,
        scales.mobility,
        FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    )?;
    let relative = (relative_mse.affine(V060_RELATIVE_MSE_WEIGHT, 0.0)?
        + relative_robust.affine(V060_RELATIVE_ROBUST_WEIGHT, 0.0)?)?;

    let anchor_latent = latent.narrow(0, 0, n)?;
    let positive_latent = latent.narrow(0, 2 * n, n)?;
    let consistency = masked_normalized_embedding_mse_v0600(
        &anchor_latent,
        &positive_latent,
        &Tensor::from_vec(positive_mask, n, device)?,
    )?;
    let identity_consistency = consistency.affine(V060_IDENTITY_CONSISTENCY_WEIGHT, 0.0)?;
    let total = ((&absolute + &relative)? + &identity_consistency)?;
    let values = [
        f64::from(total.to_scalar::<f32>()?),
        f64::from(absolute.to_scalar::<f32>()?),
        f64::from(relative.to_scalar::<f32>()?),
        f64::from(identity_consistency.to_scalar::<f32>()?),
    ];
    if !values.iter().all(|value| value.is_finite()) {
        anyhow::bail!("v0.60 representation loss is non-finite");
    }
    Ok((
        total,
        UpdateDiagnosticsV0600 {
            total: values[0],
            absolute: values[1],
            relative: values[2],
            identity_consistency: values[3],
        },
    ))
}

fn consensus_loss_v0600(
    model: &PeptideFoundationV0600Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    scales: MobilityLossScales,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, f64)> {
    let owned = examples
        .iter()
        .map(|entry| records[entry.representative_index].clone())
        .collect::<Vec<_>>();
    let batch = collator.collate(&owned, device, seed)?;
    let (mobility, ccs, _) = predicted_mobility_and_ccs_v0600(model, &batch, &owned, true)?;
    let target_mobility = Tensor::from_vec(
        examples
            .iter()
            .map(|e| e.target_mobility)
            .collect::<Vec<_>>(),
        (examples.len(), 1),
        device,
    )?;
    let factor = bruker_ccs_factor_v0600(&batch.context)?;
    let target_ccs = target_mobility.broadcast_mul(&factor)?;
    let weight = Tensor::from_vec(
        examples.iter().map(|e| e.weight).collect::<Vec<_>>(),
        (examples.len(), 1),
        device,
    )?;
    let mask = Tensor::ones((examples.len(), 1), DType::F32, device)?;
    let mobility_mse =
        weighted_scaled_mse_v0600(&mobility, &target_mobility, &mask, &weight, scales.mobility)?;
    let mobility_robust = weighted_scaled_pseudo_huber_v0600(
        &mobility,
        &target_mobility,
        &mask,
        &weight,
        scales.mobility,
        FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    )?;
    let ccs_mse = weighted_scaled_mse_v0600(&ccs, &target_ccs, &mask, &weight, scales.ccs)?;
    let ccs_robust = weighted_scaled_pseudo_huber_v0600(
        &ccs,
        &target_ccs,
        &mask,
        &weight,
        scales.ccs,
        FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
    )?;
    let total = (((mobility_mse.affine(V038_MOBILITY_MSE_WEIGHT, 0.0)?
        + mobility_robust.affine(V038_MOBILITY_ROBUST_WEIGHT, 0.0)?)?
        + ccs_mse.affine(V038_CCS_AUX_MSE_WEIGHT, 0.0)?)?
        + ccs_robust.affine(V038_CCS_AUX_ROBUST_WEIGHT, 0.0)?)?;
    let value = f64::from(total.to_scalar::<f32>()?);
    Ok((total, value))
}

fn calibrate_mobility_scales_v0600(
    model: &PeptideFoundationV0600Model,
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
            .map(|e| records[e.representative_index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let (mobility, ccs, _) = predicted_mobility_and_ccs_v0600(model, &batch, &owned, false)?;
        let target_mobility = Tensor::from_vec(
            chunk.iter().map(|e| e.target_mobility).collect::<Vec<_>>(),
            (chunk.len(), 1),
            device,
        )?;
        let target_ccs =
            target_mobility.broadcast_mul(&bruker_ccs_factor_v0600(&batch.context)?)?;
        let weight = Tensor::from_vec(
            chunk.iter().map(|e| e.weight).collect::<Vec<_>>(),
            (chunk.len(), 1),
            device,
        )?;
        mobility_squared += f64::from(
            (mobility - &target_mobility)?
                .sqr()?
                .broadcast_mul(&weight)?
                .sum_all()?
                .to_scalar::<f32>()?,
        );
        ccs_squared += f64::from(
            (ccs - &target_ccs)?
                .sqr()?
                .broadcast_mul(&weight)?
                .sum_all()?
                .to_scalar::<f32>()?,
        );
        weight_sum += f64::from(weight.sum_all()?.to_scalar::<f32>()?);
    }
    if !(weight_sum > 0.0 && mobility_squared.is_finite() && ccs_squared.is_finite()) {
        anyhow::bail!("v0.60 loss calibration has no finite weighted labels");
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

fn weighted_scaled_mse_v0600(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    weight: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!("v0.60 scalar loss shape mismatch");
    }
    let combined = mask
        .broadcast_mul(weight)?
        .broadcast_as(prediction.dims())?;
    let scaled = (prediction - target)?.affine(1.0 / scale, 0.0)?;
    let numerator = scaled.sqr()?.broadcast_mul(&combined)?.sum_all()?;
    Ok(numerator.broadcast_div(&combined.sum_all()?.clamp(1.0e-6, f64::INFINITY)?)?)
}

fn weighted_scaled_pseudo_huber_v0600(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    weight: &Tensor,
    scale: f64,
    delta: f64,
) -> Result<Tensor> {
    let combined = mask
        .broadcast_mul(weight)?
        .broadcast_as(prediction.dims())?;
    let scaled = (prediction - target)?.affine(1.0 / (scale * delta), 0.0)?;
    let robust = (scaled.sqr()? + 1.0)?
        .sqrt()?
        .affine(delta * delta, -(delta * delta))?;
    let numerator = robust.broadcast_mul(&combined)?.sum_all()?;
    Ok(numerator.broadcast_div(&combined.sum_all()?.clamp(1.0e-6, f64::INFINITY)?)?)
}

fn masked_normalized_embedding_mse_v0600(
    first: &Tensor,
    second: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    if first.dims() != second.dims() {
        anyhow::bail!("v0.60 identity embedding shape mismatch");
    }
    let (batch, dim) = first.dims2()?;
    if mask.dims1()? != batch {
        anyhow::bail!("v0.60 identity mask shape mismatch");
    }
    let epsilon = 1.0e-6;
    let first_unit = first.broadcast_div(&(first.sqr()?.sum_keepdim(1)? + epsilon)?.sqrt()?)?;
    let second_unit = second.broadcast_div(&(second.sqr()?.sum_keepdim(1)? + epsilon)?.sqrt()?)?;
    let per_row = (first_unit - second_unit)?
        .sqr()?
        .sum(1)?
        .affine(1.0 / dim as f64, 0.0)?;
    let numerator = per_row.broadcast_mul(mask)?.sum_all()?;
    Ok(numerator.broadcast_div(&mask.sum_all()?.clamp(1.0, f64::INFINITY)?)?)
}

fn backward_step_v0600(
    loss: &Tensor,
    optimizer: &mut FoundationAdamW,
    varmap: &VarMap,
    audit: bool,
) -> Result<FoundationOptimizerStep> {
    let gradients = loss.backward()?;
    if audit {
        audit_gradients_v0600(varmap, &gradients)?;
    }
    Ok(optimizer.step(&gradients, Some(V060_MAX_GRADIENT_NORM))?)
}

fn audit_gradients_v0600(
    varmap: &VarMap,
    gradients: &candle_core::backprop::GradStore,
) -> Result<()> {
    let required = [
        "student_v060.mobility_backbone.student_v050.chemistry.atom_input.weight",
        "student_v060.mobility_backbone.student_v050.interaction.0.attention.query.weight",
        "student_v060.charge.slot_embedding.weight",
        "student_v060.conformer.slot_embedding.weight",
        "student_v060.mobility.output.weight",
    ];
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.60 VarMap lock poisoned"))?;
    for name in required {
        let variable = data
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("v0.60 gradient audit missing {name}"))?;
        let gradient = gradients
            .get(variable)
            .ok_or_else(|| anyhow::anyhow!("v0.60 gradient missing for {name}"))?;
        let norm2 = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !norm2.is_finite() || norm2 <= 0.0 {
            anyhow::bail!("v0.60 invalid gradient for {name}: norm2={norm2}");
        }
        println!(
            "v0600_gradient_audit\tparameter={name}\tnorm={:.8}",
            norm2.sqrt()
        );
    }
    Ok(())
}

fn audit_peptidoform_label_v0600(record: &FoundationTrainingRecord) -> String {
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
    if modifications.is_empty() {
        record.peptidoform.sequence.clone()
    } else {
        let annotated = modifications
            .iter()
            .map(|modification| format!("[{modification}]"))
            .collect::<String>();
        format!("{}{annotated}", record.peptidoform.sequence)
    }
}

fn audit_tsv_field_v0600(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\n', " ")
        .replace('\r', " ")
}

#[allow(clippy::too_many_arguments)]
fn export_raw_ccs_predictions_v0600(
    model: &PeptideFoundationV0600Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
    batch_size: usize,
    device: &Device,
    output: &Path,
) -> Result<(usize, f64)> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::from(
        "record_index\tsource_id\tsequence\tpeptidoform\tcharge\tprecursor_mz\ttarget_ccs\tpredicted_ccs\n",
    );
    let mut absolute_error = 0.0f64;
    let mut count = 0usize;
    for chunk in indices.chunks(batch_size.max(1)) {
        let owned = chunk
            .iter()
            .map(|&idx| records[idx].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let (_, ccs, _) = predicted_mobility_and_ccs_v0600(model, &batch, &owned, false)?;
        let predicted = ccs.to_vec2::<f32>()?;
        for (row, (&record_index, record)) in chunk.iter().zip(owned.iter()).enumerate() {
            let Some(target) = record.ccs.filter(|value| value.is_finite()) else {
                continue;
            };
            let source = provenance
                .get(record_index)
                .ok_or_else(|| anyhow::anyhow!("v0.60 audit export missing provenance"))?;
            let charge = record.context.charge.unwrap_or_default();
            let mz = record.context.precursor_mz.unwrap_or_default();
            let prediction = predicted[row][0];
            absolute_error += f64::from((prediction - target).abs());
            count += 1;
            text.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\n",
                record_index,
                audit_tsv_field_v0600(&source.source_id),
                audit_tsv_field_v0600(&record.peptidoform.sequence),
                audit_tsv_field_v0600(&audit_peptidoform_label_v0600(record)),
                charge,
                mz,
                target,
                prediction,
            ));
        }
    }
    if count == 0 {
        anyhow::bail!("v0.60 audit export contains no finite DEV CCS rows");
    }
    fs::write(output, text)?;
    Ok((count, absolute_error / count as f64))
}

fn evaluate_raw_ccs_v0600(
    model: &PeptideFoundationV0600Model,
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
            .map(|&idx| records[idx].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let (_, ccs, _) = predicted_mobility_and_ccs_v0600(model, &batch, &owned, false)?;
        let predicted = ccs.to_vec2::<f32>()?;
        for (row, record) in owned.iter().enumerate() {
            let Some(target) = record.ccs.filter(|value| value.is_finite()) else {
                continue;
            };
            absolute += f64::from((predicted[row][0] - target).abs());
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.60 raw DEV has no CCS labels");
    }
    Ok(absolute / count as f64)
}

fn evaluate_consensus_ccs_v0600(
    model: &PeptideFoundationV0600Model,
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
            .map(|entry| records[entry.representative_index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let (_, ccs, _) = predicted_mobility_and_ccs_v0600(model, &batch, &owned, false)?;
        let target_mobility = Tensor::from_vec(
            chunk.iter().map(|e| e.target_mobility).collect::<Vec<_>>(),
            (chunk.len(), 1),
            device,
        )?;
        let target_ccs =
            target_mobility.broadcast_mul(&bruker_ccs_factor_v0600(&batch.context)?)?;
        let p = ccs.to_vec2::<f32>()?;
        let t = target_ccs.to_vec2::<f32>()?;
        for row in 0..p.len() {
            absolute += f64::from((p[row][0] - t[row][0]).abs());
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.60 consensus DEV has no CCS labels");
    }
    Ok(absolute / count as f64)
}

fn combined_dev_objective_v0600(raw: f64, consensus: f64) -> f64 {
    V038_RAW_OBJECTIVE_WEIGHT * (raw / V038_RAW_CCS_DEV_MAE)
        + V038_CONSENSUS_OBJECTIVE_WEIGHT * (consensus / V038_CONSENSUS_CCS_DEV_MAE)
}

fn print_dev_v0600(label: &str, update: usize, metrics: DevMetricsV0600) {
    println!(
        "{label}\tupdate={update}\traw_ccs_mae={:.8}\tconsensus_ccs_mae={:.8}\tdev_objective={:.8}",
        metrics.raw_ccs_mae, metrics.consensus_ccs_mae, metrics.objective
    );
}

fn print_material_gate_v0600(metrics: DevMetricsV0600, smoke_mode: bool) {
    let raw_beats_v038 = metrics.raw_ccs_mae < V038_RAW_CCS_DEV_MAE;
    let raw_preferred = metrics.raw_ccs_mae <= V060_RAW_CCS_PREFERRED_MAE;
    let consensus_beats_v038 = metrics.consensus_ccs_mae < V038_CONSENSUS_CCS_DEV_MAE;
    let consensus_material = metrics.consensus_ccs_mae <= V060_CONSENSUS_MATERIAL_MAE;
    let material = raw_preferred && consensus_material;
    println!("v0600_raw_ccs_beats_v038\t{}", yes_no(raw_beats_v038));
    println!(
        "v0600_raw_ccs_preferred_gate_met\t{}",
        yes_no(raw_preferred)
    );
    println!(
        "v0600_consensus_ccs_beats_v038\t{}",
        yes_no(consensus_beats_v038)
    );
    println!(
        "v0600_consensus_ccs_material_gate_met\t{}",
        yes_no(consensus_material)
    );
    println!(
        "v0600_material_mobility_representation_gain\t{}",
        yes_no(material)
    );
    println!(
        "v0600_holdout_eligible\t{}",
        yes_no(material && !smoke_mode)
    );
    println!("v0600_finalize_required\tNO_explicit_handoff_review_required");
}

fn read_v035_metadata_v0600(checkpoint: &Path) -> Result<V035ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.35 metadata {path:?}"))
}

fn read_v060_metadata(checkpoint: &Path) -> Result<V060Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.60 metadata {path:?}"))
}

fn validate_v060_audit_metadata(
    metadata: &V060Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    parent_v0350: &Path,
    config: &PeptideFoundationV0600Config,
) -> Result<()> {
    if metadata.version != V060_VERSION
        || metadata.objective != V060_OBJECTIVE
        || metadata.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0600
        || metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        || metadata.parent_v0350_metadata_checkpoint != parent_v0350.display().to_string()
        || metadata.v0600_config != *config
    {
        anyhow::bail!("v0.60 audit-export checkpoint provenance mismatch");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_v060_metadata(
    metadata: &V060Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    supervision_fingerprint: &str,
    representation_fingerprint: &str,
    parent_v0350: &Path,
    config: &PeptideFoundationV0600Config,
    representation_epochs: usize,
    consensus_epochs: usize,
    representation_batch_size: usize,
    consensus_batch_size: usize,
    seed: u64,
    learning_rate: f64,
) -> Result<()> {
    if metadata.version != V060_VERSION
        || metadata.objective != V060_OBJECTIVE
        || metadata.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0600
        || metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        || metadata.mobility_supervision_fingerprint != supervision_fingerprint
        || metadata.representation_pair_fingerprint != representation_fingerprint
        || metadata.parent_v0350_metadata_checkpoint != parent_v0350.display().to_string()
        || metadata.v0600_config != *config
        || metadata.representation_epochs != representation_epochs
        || metadata.consensus_epochs != consensus_epochs
        || metadata.representation_batch_size != representation_batch_size
        || metadata.consensus_batch_size != consensus_batch_size
        || metadata.seed != seed
    {
        anyhow::bail!("v0.60 resume metadata contract mismatch");
    }
    if (metadata.learning_rate - learning_rate).abs()
        > f64::EPSILON * 64.0 * learning_rate.abs().max(1.0)
    {
        anyhow::bail!("v0.60 resume learning-rate mismatch");
    }
    Ok(())
}

fn save_checkpoint_v0600(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V060Metadata,
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

fn deterministic_index_subset_v0600(
    indices: &[usize],
    max_records: usize,
    seed: u64,
) -> Vec<usize> {
    let mut ranked = indices
        .iter()
        .copied()
        .map(|idx| (mix64(idx as u64 ^ seed), idx))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|entry| entry.0);
    ranked
        .into_iter()
        .take(max_records.min(indices.len()))
        .map(|entry| entry.1)
        .collect()
}

fn deterministic_consensus_subset_v0600(
    examples: &[MobilityConsensusExample],
    max_records: usize,
    seed: u64,
) -> Vec<MobilityConsensusExample> {
    let mut ranked = examples
        .iter()
        .cloned()
        .map(|entry| (mix64(entry.identity_hash ^ seed), entry))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|entry| entry.0);
    ranked
        .into_iter()
        .take(max_records.min(examples.len()))
        .map(|entry| entry.1)
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
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value.parse::<T>().map_err(|error| {
            anyhow::anyhow!("failed to parse argument {index}={value:?}: {error}")
        }),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0600_relative_pair_builder_uses_same_charge_different_identity() {
        let raw = vec![
            RawMobilityExampleV0600 {
                record_index: 0,
                target_mobility: 1.0,
                weight: 1.0,
                identity_hash: 1,
                source_id: "a".into(),
                charge: 2,
                precursor_mz: 500.0,
            },
            RawMobilityExampleV0600 {
                record_index: 1,
                target_mobility: 1.1,
                weight: 1.0,
                identity_hash: 2,
                source_id: "b".into(),
                charge: 2,
                precursor_mz: 500.2,
            },
            RawMobilityExampleV0600 {
                record_index: 2,
                target_mobility: 1.2,
                weight: 1.0,
                identity_hash: 3,
                source_id: "c".into(),
                charge: 3,
                precursor_mz: 500.1,
            },
        ];
        let pairs = build_representation_examples_v0600(&raw).unwrap();
        assert_eq!(pairs.len(), 2);
        for pair in pairs {
            assert_eq!(raw[pair.anchor_raw].charge, raw[pair.hard_raw].charge);
            assert_ne!(
                raw[pair.anchor_raw].identity_hash,
                raw[pair.hard_raw].identity_hash
            );
        }
    }

    #[test]
    fn v0600_consensus_reference_gate_is_stricter_than_v038() {
        assert!(V060_RAW_CCS_PREFERRED_MAE < V038_RAW_CCS_DEV_MAE);
        assert!(V060_CONSENSUS_MATERIAL_MAE < V038_CONSENSUS_CCS_DEV_MAE);
    }
}
