//! v0.39 precursor-conditioned conformer-token CCS representation from frozen v0.35.
//!
//! v0.38 showed that a trainable CCS-only representation improves DEV CCS, but
//! the precursor physics was still injected only inside a post-encoder head.
//! v0.39 makes precursor physics an explicit conformer token that participates in
//! self-attention with residue embeddings, allowing charge/mass context to alter
//! the sequence representation before pooling. RT, MS2, and inverse stay frozen.
//! TRAIN-only mobility consensus/reliability supervision and the v0.38 optimization
//! contract remain fixed so the experiment isolates the representation change.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_ms2_loss, load_foundation_corpus, read_foundation_training_run_config,
    sample_foundation_validation_indices, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCollator, FoundationCollatorConfig,
    FoundationCorruptionConfig, FoundationFragmentContextBatchV0350,
    FoundationLearningRateSchedule, FoundationModificationSite, FoundationMs2LossConfig,
    FoundationPartition, FoundationRecordProvenance, FoundationRegressionNormalization,
    FoundationSamplePlan, FoundationSamplingConfig, FoundationScalarPhysicsBatchV0360,
    FoundationTargetNormalizationConfig, FoundationTrainingRecord,
    PeptideFoundationMultimodalV0350Config, PeptideFoundationMultimodalV0390Config,
    PeptideFoundationMultimodalV0390Model, RetentionTimeObjective,
    FOUNDATION_CCS_STRETCH_TARGET_MAE_V0360, FOUNDATION_CCS_TARGET_MAE_V0360,
    FOUNDATION_RT_STRETCH_TARGET_MAE_V0360, FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V039_VERSION: u32 = 390;
const V039_OBJECTIVE: &str = "v0390_precursor_conditioned_conformer_token_ccs";
const V039_ARCHITECTURE: &str = "v0.39-frozen-v0350-precursor-conditioned-conformer-token-ccs";
const V039_MAX_STEPS_PER_EPOCH: usize = 1_536;
const V039_MAX_DEV_BATCHES: usize = 32;
const V039_MAX_HOLDOUT_BATCHES: usize = 32;
const V039_LOSS_CALIBRATION_RECORDS: usize = 4_096;
const V039_MOBILITY_MSE_WEIGHT: f64 = 0.25;
const V039_MOBILITY_ROBUST_WEIGHT: f64 = 1.0;
const V039_CCS_AUX_MSE_WEIGHT: f64 = 0.10;
const V039_CCS_AUX_ROBUST_WEIGHT: f64 = 0.35;
const V039_MIN_MOBILITY_LOSS_SCALE: f64 = 1.0e-5;
const V039_MIN_CCS_LOSS_SCALE: f64 = 1.0e-3;
const V039_MAX_GRADIENT_NORM: f64 = 1.0;
const V039_MATERIAL_RAW_CCS_RATIO: f64 = 0.90;
const V039_MATERIAL_CONSENSUS_CCS_RATIO: f64 = 0.88;
const V039_RAW_OBJECTIVE_WEIGHT: f64 = 0.65;
const V039_CONSENSUS_OBJECTIVE_WEIGHT: f64 = 0.35;
const V039_RT_INVARIANCE_TOLERANCE: f64 = 5.0e-5;
const V039_MS2_INVARIANCE_TOLERANCE: f64 = 5.0e-5;
const V039_SOURCE_SHRINKAGE: f64 = 256.0;
const V039_MIN_SOURCE_SHARED_IDENTITIES: usize = 20;
const V039_RELIABILITY_FLOOR: f64 = 0.35;
const V039_RELIABILITY_CEILING: f64 = 1.25;
const V039_SINGLETON_WEIGHT_SCALE: f64 = 0.65;
const V039_CONSENSUS_DISPERSION_SCALE: f64 = 0.010;

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
    ccs_mae_native: Option<f64>,
    ccs_rmse_native: Option<f64>,
    ms2_loss: Option<f64>,
    ms2_pointwise_mse: Option<f64>,
    ms2_pointwise_mae: Option<f64>,
    ms2_mean_cosine: Option<f64>,
    ms2_mean_spectral_angle: Option<f64>,
    ms2_mean_pearson: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V039Metadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    supervision_fingerprint: String,
    parent_v0350_checkpoint: String,
    parent_v0350_completed_steps: usize,
    v0390_config: PeptideFoundationMultimodalV0390Config,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    train_steps: usize,
    batch_size: usize,
    dev_batches: usize,
    holdout_batches: usize,
    seed: u64,
    learning_rate: f64,
    mobility_loss_scale_native: f64,
    ccs_aux_loss_scale_native: f64,
    completed_steps: usize,
    consensus_train_examples: usize,
    consensus_train_multisource_examples: usize,
    initial_dev_metrics: PropertyMetrics,
    initial_raw_dev_ccs_mae: f64,
    initial_consensus_dev_ccs_mae: f64,
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
struct WarmStartReport {
    loaded_variables: usize,
    fresh_variables: usize,
    ignored_parent_variables: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 11 {
        anyhow::bail!(
            "usage: foundation_train_multimodal_v0390 RUN_V0260.yaml OUTPUT_DIR PARENT_V0350_CHECKPOINT [max_epochs=10] [batch_size=96] [patience=3] [min_delta=0.002] [seed=20261038] [learning_rate=3e-5] [mode=train|finalize]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let max_epochs = parse_or(&args, 4, 10usize)?;
    let batch_size = parse_or(&args, 5, 96usize)?;
    let patience = parse_or(&args, 6, 3usize)?;
    let min_delta = parse_or(&args, 7, 0.002f64)?;
    let seed = parse_or(&args, 8, 20_261_038u64)?;
    let learning_rate = parse_or(&args, 9, 3.0e-5f64)?;
    let run_mode = args.get(10).map(String::as_str).unwrap_or("train");
    let finalize_only = match run_mode {
        "train" => false,
        "finalize" => true,
        other => anyhow::bail!("unsupported v0.39 run mode {other:?}; expected train or finalize"),
    };

    if max_epochs == 0 || batch_size < 2 || patience == 0 {
        anyhow::bail!("v0.39 requires max_epochs>0, batch_size>=2, and patience>0");
    }
    if !(min_delta > 0.0 && min_delta.is_finite()) {
        anyhow::bail!("v0.39 min_delta must be positive and finite");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("v0.39 learning_rate must be positive and finite");
    }

    if finalize_only {
        for name in ["initial", "best"] {
            let directory = output_root.join(name);
            for file in [
                "model.safetensors",
                "optimizer.safetensors",
                "metadata.yaml",
            ] {
                if !directory.join(file).is_file() {
                    anyhow::bail!("v0.39 finalize is missing {:?}", directory.join(file));
                }
            }
        }
        if output_root.join("final").exists() {
            anyhow::bail!(
                "v0.39 final checkpoint already exists; HOLDOUT must not be consumed twice"
            );
        }
    } else if output_root.exists() {
        anyhow::bail!(
            "v0.39 TRAIN output directory already exists: {:?}",
            output_root
        );
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.39 requires a CUDA device")?;
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
    let holdout_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .map(|entry| entry.record_index)
        .collect();
    let mobility_train_indices = finite_mobility_ccs_indices(&corpus.records, &train_indices);
    let ccs_dev_indices = finite_mobility_ccs_indices(&corpus.records, &dev_indices);
    if mobility_train_indices.len() < batch_size || ccs_dev_indices.len() < batch_size {
        anyhow::bail!(
            "v0.39 requires at least one batch of CCS labels in TRAIN and DEV; observed train={} dev={} batch_size={batch_size}",
            mobility_train_indices.len(),
            ccs_dev_indices.len()
        );
    }

    let parent_metadata = read_v035_metadata(&parent_checkpoint)?;
    if parent_metadata.version != 350 {
        anyhow::bail!(
            "v0.39 requires v0.35 metadata version 350, observed {}",
            parent_metadata.version
        );
    }
    if parent_metadata.objective != "v0350_trainable_forward_representation_context_conditioned_ms2"
    {
        anyhow::bail!(
            "v0.39 requires the accepted v0.35 objective, observed {}",
            parent_metadata.objective
        );
    }
    if parent_metadata.completed_steps == 0 {
        anyhow::bail!("v0.39 refuses an unselected v0.35 baseline checkpoint");
    }
    parent_metadata.v0350_config.validate()?;
    let v0390_config =
        PeptideFoundationMultimodalV0390Config::fixed(parent_metadata.v0350_config.clone())?;
    let forward_config = v0390_config.forward().clone();

    let current_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let current_benchmark_fingerprint =
        format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != current_corpus_fingerprint {
        anyhow::bail!("v0.39 corpus fingerprint differs from frozen v0.35 parent");
    }
    if parent_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint {
        anyhow::bail!("v0.39 benchmark fingerprint differs from frozen v0.35 parent");
    }

    let mut max_prepared_len = 0usize;
    for &index in train_indices
        .iter()
        .chain(&dev_indices)
        .chain(&holdout_indices)
    {
        let length = corpus.records[index].peptidoform.sequence.chars().count();
        max_prepared_len = max_prepared_len.max(length);
        if length > forward_config.max_sequence_len {
            anyhow::bail!(
                "v0.39 record {index} length {length} exceeds max_sequence_len={}",
                forward_config.max_sequence_len
            );
        }
    }

    // TRAIN-only target reconstruction. No DEV labels participate in source fitting
    // or source reliability estimation.
    let train_supervision = build_train_consensus_supervision(
        &corpus.records,
        &corpus.provenance,
        &mobility_train_indices,
    )?;
    if train_supervision.examples.len() < batch_size {
        anyhow::bail!("v0.39 consensus TRAIN has fewer examples than batch_size");
    }
    let dev_consensus = build_partition_consensus_examples(
        &corpus.records,
        &corpus.provenance,
        &ccs_dev_indices,
        &train_supervision.source_supervision,
    )?;
    if dev_consensus.is_empty() {
        anyhow::bail!("v0.39 DEV consensus set is empty");
    }

    let steps_per_epoch = (train_supervision.examples.len() / batch_size)
        .min(V039_MAX_STEPS_PER_EPOCH)
        .max(1);
    let max_total_steps = max_epochs.saturating_mul(steps_per_epoch);
    let requested_dev_batches = (dev_indices.len() / batch_size)
        .min(V039_MAX_DEV_BATCHES)
        .max(1);
    let requested_holdout_batches = (holdout_indices.len() / batch_size)
        .min(V039_MAX_HOLDOUT_BATCHES)
        .max(1);

    let train_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_indices,
        &dev_indices,
    );
    let dev_batches = feasible_validation_batches(
        "dev",
        &train_sampling,
        &corpus.provenance,
        &dev_indices,
        batch_size,
        requested_dev_batches,
    )?;
    let mut dev_sampling = train_sampling.clone();
    dev_sampling.validation_steps = Some(dev_batches);
    let dev_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &dev_indices,
        batch_size,
        seed ^ 0x3800_d3f0_1234_5678,
        &dev_sampling,
    )?;

    let mut holdout_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_indices,
        &holdout_indices,
    );
    let holdout_batches = feasible_validation_batches(
        "holdout",
        &holdout_sampling,
        &corpus.provenance,
        &holdout_indices,
        batch_size,
        requested_holdout_batches,
    )?;
    holdout_sampling.validation_steps = Some(holdout_batches);
    let holdout_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &holdout_indices,
        batch_size,
        seed ^ 0x3800_484f_4c44_4f55,
        &holdout_sampling,
    )?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultimodalV0390Model::new(v0390_config.clone(), vb)?;
    let warm_start = load_v0350_parent_variables(
        &varmap,
        &parent_checkpoint.join("model.safetensors"),
        &device,
    )?;

    let clean_collator = FoundationCollator::new(
        forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: parent_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let target_normalization = parent_metadata.target_normalization;
    let ms2_loss = parent_metadata.ms2_loss;
    let calibration_count = V039_LOSS_CALIBRATION_RECORDS
        .min(train_supervision.examples.len())
        .max(batch_size);
    let calibration_order = deterministic_example_order(
        &train_supervision.examples,
        calibration_count,
        0,
        seed ^ 0x3800_4c4f_5353_4343,
    );
    let calibration_examples = calibration_order
        .iter()
        .map(|&index| train_supervision.examples[index].clone())
        .collect::<Vec<_>>();
    let loss_scales = calibrate_mobility_residual_scales(
        &model,
        &clean_collator,
        &corpus.records,
        &calibration_examples,
        batch_size,
        &target_normalization,
        &device,
    )?;

    let optimizer_prefixes = [
        "ccs_forward_v0390.encoder.",
        "ccs_property_refinement_v0390.",
        "ccs_conformer_v0390.",
    ];
    let mut optimizer = FoundationAdamW::new_for_prefixes(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: run.trainer.weight_decay,
        },
        &optimizer_prefixes,
    )?;
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 500u64.min(max_total_steps.saturating_sub(1) as u64),
        total_steps: max_total_steps as u64,
        min_lr_ratio: 0.10,
    };

    if !finalize_only {
        fs::create_dir_all(&output_root)?;
        write_mobility_supervision_summary(&output_root, &train_supervision)?;
    }

    println!("v0390_version\tv0.39-precursor-conditioned-conformer-token-ccs");
    println!("objective\t{V039_OBJECTIVE}");
    println!("architecture\t{V039_ARCHITECTURE}");
    println!("parent_v0350_checkpoint\t{}", parent_checkpoint.display());
    println!(
        "parent_v0350_completed_steps\t{}",
        parent_metadata.completed_steps
    );
    println!(
        "warm_start_loaded_variables\t{}",
        warm_start.loaded_variables
    );
    println!("warm_start_fresh_variables\t{}", warm_start.fresh_variables);
    println!(
        "warm_start_ignored_parent_variables\t{}",
        warm_start.ignored_parent_variables
    );
    println!("device\t{:?}", device);
    println!("corpus_fingerprint\t{current_corpus_fingerprint}");
    println!("benchmark_manifest_fingerprint\t{current_benchmark_fingerprint}");
    println!(
        "supervision_fingerprint\tfnv1a64:{:016x}",
        train_supervision.fingerprint
    );
    println!("prepared_max_sequence_len\t{max_prepared_len}");
    println!(
        "raw_train_mobility_records\t{}",
        train_supervision.raw_records
    );
    println!(
        "consensus_train_examples\t{}",
        train_supervision.examples.len()
    );
    println!(
        "consensus_train_multisource_examples\t{}",
        train_supervision.multisource_examples
    );
    println!(
        "consensus_train_singleton_examples\t{}",
        train_supervision.singleton_examples
    );
    println!(
        "consensus_train_mean_weight\t{:.8}",
        train_supervision.mean_weight
    );
    println!(
        "consensus_train_mean_abs_adjusted_delta_to_target\t{:.8}",
        train_supervision.mean_abs_adjusted_delta_to_consensus
    );
    println!("dev_raw_ccs_records\t{}", ccs_dev_indices.len());
    println!("dev_consensus_examples\t{}", dev_consensus.len());
    println!("steps_per_epoch\t{steps_per_epoch}");
    println!("dev_protected_batches\t{dev_batches}");
    println!("holdout_protected_batches_reserved\t{holdout_batches}");
    println!("max_epochs\t{max_epochs}");
    println!("batch_size\t{batch_size}");
    println!("base_learning_rate\t{learning_rate}");
    println!("loss_calibration_records\t{}", calibration_examples.len());
    println!("mobility_loss_scale_native\t{:.8}", loss_scales.mobility);
    println!("ccs_aux_loss_scale_native\t{:.8}", loss_scales.ccs);
    println!("mobility_loss_objective\t0.25_scaled_mse_plus_1.0_scaled_pseudo_huber");
    println!("ccs_aux_loss_objective\t0.10_scaled_mse_plus_0.35_scaled_pseudo_huber");
    println!("lr_schedule\twarmup_cosine_500_to_0.10");
    println!("optimizer_scope\tccs_forward_v0390.encoder+ccs_property_refinement_v0390+ccs_conformer_v0390_only");
    println!("optimizer_variable_count\t{}", optimizer.variable_count());
    println!("v0350_update_policy\texact_frozen_anchor");
    println!("ccs_representation_update_policy\ttrainable_clone_with_precursor_conditioned_conformer_token_initialized_from_v0350");
    println!("ccs_context_injection_policy\tprecursor_physics_token_bidirectional_self_attention_before_pooling");
    println!("rt_update_policy\texact_frozen_v0350");
    println!("ms2_update_policy\texact_frozen_v0350");
    println!("inverse_update_policy\texact_frozen_v0350");
    println!("training_identity_policy\tone_example_per_exact_peptidoform_charge");
    println!("training_target_policy\ttrain_only_mobility_source_affine_then_family_deduplicated_reliability_weighted_consensus");
    println!("overlapping_view_policy\tpxd034128_and_pxd058337_collapsed_to_project_family_before_consensus");
    println!(
        "material_dev_gate\traw_ccs_ratio<={V039_MATERIAL_RAW_CCS_RATIO:.2}_and_consensus_ccs_ratio<={V039_MATERIAL_CONSENSUS_CCS_RATIO:.2}_and_rt_ms2_invariant"
    );
    println!("historical_validation_reused_for_v0390_selection\tNO");
    println!("historical_test_consumed\tNO");
    print_sample_plan("dev_protected", &dev_plan);
    print_sample_plan("holdout_protected_reserved", &holdout_plan);
    println!("run_mode\t{run_mode}");

    let supervision_fingerprint = format!("fnv1a64:{:016x}", train_supervision.fingerprint);

    if finalize_only {
        let best_dir = output_root.join("best");
        let best_metadata = read_v039_metadata(&best_dir)?;
        validate_metadata(
            &best_metadata,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &supervision_fingerprint,
            &v0390_config,
            batch_size,
            seed,
            learning_rate,
            max_total_steps,
        )?;
        if best_metadata.completed_steps == 0 {
            anyhow::bail!("v0.39 has no DEV improvement; refusing to consume HOLDOUT");
        }
        varmap.load(best_dir.join("model.safetensors"))?;
        let best_protected_dev = evaluate_properties(
            &model,
            &corpus.records,
            &dev_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        validate_rt_invariance(best_protected_dev, best_metadata.initial_dev_metrics)?;
        validate_ms2_invariance(best_protected_dev, best_metadata.initial_dev_metrics)?;
        let best_raw_dev = evaluate_raw_ccs_indices(
            &model,
            &clean_collator,
            &corpus.records,
            &ccs_dev_indices,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let best_consensus_dev = evaluate_consensus_targets(
            &model,
            &clean_collator,
            &corpus.records,
            &dev_consensus,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let raw_ratio = best_raw_dev / best_metadata.initial_raw_dev_ccs_mae;
        let consensus_ratio = best_consensus_dev / best_metadata.initial_consensus_dev_ccs_mae;
        let objective = combined_dev_objective(raw_ratio, consensus_ratio);
        println!("v0390_finalize_dev_objective\t{objective:.8}");
        println!("v0390_finalize_raw_ccs_ratio\t{raw_ratio:.8}");
        println!("v0390_finalize_consensus_ccs_ratio\t{consensus_ratio:.8}");
        if raw_ratio > V039_MATERIAL_RAW_CCS_RATIO
            || consensus_ratio > V039_MATERIAL_CONSENSUS_CCS_RATIO
        {
            anyhow::bail!(
                "v0.39 DEV result does not satisfy the fixed materiality gate; refusing HOLDOUT"
            );
        }

        println!("v0390_finalize_only\ttrue");
        println!(
            "v0390_finalize_best_step\t{}",
            best_metadata.completed_steps
        );
        let holdout_ccs_indices = finite_mobility_ccs_indices(&corpus.records, &holdout_indices);
        let holdout_consensus = build_partition_consensus_examples(
            &corpus.records,
            &corpus.provenance,
            &holdout_ccs_indices,
            &train_supervision.source_supervision,
        )?;
        let holdout_protected = evaluate_properties(
            &model,
            &corpus.records,
            &holdout_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        print_metrics(
            "train_holdout_once",
            best_metadata.completed_steps,
            holdout_protected,
        );
        let holdout_raw_ccs = evaluate_raw_ccs_indices(
            &model,
            &clean_collator,
            &corpus.records,
            &holdout_ccs_indices,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let holdout_consensus_ccs = evaluate_consensus_targets(
            &model,
            &clean_collator,
            &corpus.records,
            &holdout_consensus,
            batch_size,
            &target_normalization,
            &device,
        )?;
        println!(
            "v0390_holdout_raw_ccs\trecords={}\tmae={holdout_raw_ccs:.8}",
            holdout_ccs_indices.len()
        );
        println!(
            "v0390_holdout_consensus_ccs\tidentities={}\tmae={holdout_consensus_ccs:.8}",
            holdout_consensus.len()
        );
        println!("train_holdout_consumed_for_selection\tNO");
        println!("historical_validation_reused_for_v0390_selection\tNO");
        println!("historical_test_consumed\tNO");
        copy_checkpoint_dir(&best_dir, &output_root.join("final"))?;
        println!(
            "v0390_finalize_complete\tbest_step={}",
            best_metadata.completed_steps
        );
        println!("final_checkpoint\t{}", output_root.join("final").display());
        return Ok(());
    }

    let initial_protected_dev = evaluate_properties(
        &model,
        &corpus.records,
        &dev_plan.indices,
        batch_size,
        &clean_collator,
        &target_normalization,
        ms2_loss,
        &device,
    )?;
    print_metrics("train_dev_initial", 0, initial_protected_dev);
    let initial_raw_dev_ccs = evaluate_raw_ccs_indices(
        &model,
        &clean_collator,
        &corpus.records,
        &ccs_dev_indices,
        batch_size,
        &target_normalization,
        &device,
    )?;
    let initial_consensus_dev_ccs = evaluate_consensus_targets(
        &model,
        &clean_collator,
        &corpus.records,
        &dev_consensus,
        batch_size,
        &target_normalization,
        &device,
    )?;
    println!(
        "train_dev_ccs_raw_initial\tstep=0\trecords={}\tmae={initial_raw_dev_ccs:.8}",
        ccs_dev_indices.len()
    );
    println!("train_dev_ccs_consensus_initial\tstep=0\tidentities={}\tmae={initial_consensus_dev_ccs:.8}", dev_consensus.len());
    println!("train_dev_objective\tepoch=0\tstep=0\tvalue=1.00000000\tbest=true");

    let metadata = |completed_steps| V039Metadata {
        version: V039_VERSION,
        objective: V039_OBJECTIVE.into(),
        architecture: V039_ARCHITECTURE.into(),
        corpus_fingerprint: current_corpus_fingerprint.clone(),
        benchmark_manifest_fingerprint: current_benchmark_fingerprint.clone(),
        supervision_fingerprint: supervision_fingerprint.clone(),
        parent_v0350_checkpoint: parent_checkpoint.display().to_string(),
        parent_v0350_completed_steps: parent_metadata.completed_steps,
        v0390_config: v0390_config.clone(),
        rt_objective: parent_metadata.rt_objective,
        target_normalization,
        ms2_loss,
        train_steps: max_total_steps,
        batch_size,
        dev_batches,
        holdout_batches,
        seed,
        learning_rate,
        mobility_loss_scale_native: loss_scales.mobility,
        ccs_aux_loss_scale_native: loss_scales.ccs,
        completed_steps,
        consensus_train_examples: train_supervision.examples.len(),
        consensus_train_multisource_examples: train_supervision.multisource_examples,
        initial_dev_metrics: initial_protected_dev,
        initial_raw_dev_ccs_mae: initial_raw_dev_ccs,
        initial_consensus_dev_ccs_mae: initial_consensus_dev_ccs,
    };

    save_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &optimizer,
        &metadata(0),
    )?;
    save_checkpoint(&output_root.join("best"), &varmap, &optimizer, &metadata(0))?;

    let mut global_step = 0usize;
    let mut best_epoch = 0usize;
    let mut best_step = 0usize;
    let mut best_objective = 1.0f64;
    let mut stale_epochs = 0usize;
    let mut stopped_early = false;

    for epoch in 1..=max_epochs {
        let needed = steps_per_epoch.saturating_mul(batch_size);
        let order = deterministic_example_order(
            &train_supervision.examples,
            needed,
            epoch as u64,
            seed ^ 0x3800_7a11_2233_4455,
        );
        println!(
            "v0390_epoch\tstage=start\tepoch={epoch}\tsteps={steps_per_epoch}\texamples={needed}"
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
            let loss = mobility_consensus_loss(
                &model,
                &clean_collator,
                &corpus.records,
                &selected,
                &target_normalization,
                loss_scales,
                seed ^ (global_step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                &device,
            )?;
            let loss_value = f64::from(loss.to_scalar::<f32>()?);
            let update = optimizer.backward_step(&loss, Some(V039_MAX_GRADIENT_NORM))?;
            if global_step == 1 || global_step % 100 == 0 || local_step + 1 == steps_per_epoch {
                println!(
                    "v0390_train\tepoch={epoch}\tstep={global_step}\tepoch_step={}\tlr={:.8}\ttotal={loss_value:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                    local_step + 1,
                    update.learning_rate,
                    update.gradient_norm,
                    update.gradient_scale,
                );
            }
        }

        let protected_dev = evaluate_properties(
            &model,
            &corpus.records,
            &dev_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        print_metrics("train_dev", global_step, protected_dev);
        validate_rt_invariance(protected_dev, initial_protected_dev)?;
        validate_ms2_invariance(protected_dev, initial_protected_dev)?;
        let raw_dev_ccs = evaluate_raw_ccs_indices(
            &model,
            &clean_collator,
            &corpus.records,
            &ccs_dev_indices,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let consensus_dev_ccs = evaluate_consensus_targets(
            &model,
            &clean_collator,
            &corpus.records,
            &dev_consensus,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let raw_ratio = raw_dev_ccs / initial_raw_dev_ccs;
        let consensus_ratio = consensus_dev_ccs / initial_consensus_dev_ccs;
        let objective = combined_dev_objective(raw_ratio, consensus_ratio);
        let improved =
            raw_ratio < 1.0 && consensus_ratio < 1.0 && best_objective - objective > min_delta;
        println!("train_dev_ccs_raw\tepoch={epoch}\tstep={global_step}\trecords={}\tmae={raw_dev_ccs:.8}\tratio={raw_ratio:.8}", ccs_dev_indices.len());
        println!("train_dev_ccs_consensus\tepoch={epoch}\tstep={global_step}\tidentities={}\tmae={consensus_dev_ccs:.8}\tratio={consensus_ratio:.8}", dev_consensus.len());
        println!(
            "train_dev_objective\tepoch={epoch}\tstep={global_step}\tvalue={objective:.8}\traw_ccs_ratio={raw_ratio:.8}\tconsensus_ccs_ratio={consensus_ratio:.8}\tprevious_best={best_objective:.8}\timproved={improved}"
        );

        save_checkpoint(
            &output_root.join("latest"),
            &varmap,
            &optimizer,
            &metadata(global_step),
        )?;
        if improved {
            best_objective = objective;
            best_epoch = epoch;
            best_step = global_step;
            stale_epochs = 0;
            save_checkpoint(
                &output_root.join("best"),
                &varmap,
                &optimizer,
                &metadata(global_step),
            )?;
            println!("v0390_best_checkpoint\tepoch={best_epoch}\tstep={best_step}\tdev_objective={best_objective:.8}\traw_ccs_ratio={raw_ratio:.8}\tconsensus_ccs_ratio={consensus_ratio:.8}");
        } else {
            stale_epochs += 1;
        }
        println!("v0390_epoch\tstage=complete\tepoch={epoch}\tstep={global_step}\tstale_epochs={stale_epochs}");
        if stale_epochs >= patience {
            stopped_early = true;
            println!("v0390_early_stop\tepoch={epoch}\tstep={global_step}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}");
            break;
        }
    }

    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0390_selection\tNO");
    println!("historical_test_consumed\tNO");
    println!("v0390_training_complete\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}\tstopped_early={stopped_early}");

    if best_step == 0 {
        println!("v0390_material_dev_gain\tNO");
        println!("v0390_finalize_required\tNO");
        println!("v0390_rethink_required\tYES");
    } else {
        varmap.load(output_root.join("best/model.safetensors"))?;
        let best_protected_dev = evaluate_properties(
            &model,
            &corpus.records,
            &dev_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        validate_rt_invariance(best_protected_dev, initial_protected_dev)?;
        validate_ms2_invariance(best_protected_dev, initial_protected_dev)?;
        let best_raw_dev = evaluate_raw_ccs_indices(
            &model,
            &clean_collator,
            &corpus.records,
            &ccs_dev_indices,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let best_consensus_dev = evaluate_consensus_targets(
            &model,
            &clean_collator,
            &corpus.records,
            &dev_consensus,
            batch_size,
            &target_normalization,
            &device,
        )?;
        let raw_ratio = best_raw_dev / initial_raw_dev_ccs;
        let consensus_ratio = best_consensus_dev / initial_consensus_dev_ccs;
        let material = raw_ratio <= V039_MATERIAL_RAW_CCS_RATIO
            && consensus_ratio <= V039_MATERIAL_CONSENSUS_CCS_RATIO;
        println!("v0390_best_raw_dev_ccs_mae\t{best_raw_dev:.8}");
        println!("v0390_best_raw_ccs_ratio\t{raw_ratio:.8}");
        println!("v0390_best_consensus_dev_ccs_mae\t{best_consensus_dev:.8}");
        println!("v0390_best_consensus_ccs_ratio\t{consensus_ratio:.8}");
        println!(
            "v0390_rt_stretch_target_met\t{}",
            yes_no(
                best_protected_dev
                    .rt_mae_native
                    .is_some_and(|v| v <= FOUNDATION_RT_STRETCH_TARGET_MAE_V0360)
            )
        );
        println!(
            "v0390_ccs_target_met\t{}",
            yes_no(best_raw_dev <= FOUNDATION_CCS_TARGET_MAE_V0360)
        );
        println!(
            "v0390_ccs_stretch_target_met\t{}",
            yes_no(best_raw_dev <= FOUNDATION_CCS_STRETCH_TARGET_MAE_V0360)
        );
        if material {
            println!("v0390_material_dev_gain\tYES");
            println!("v0390_finalize_required\tYES");
            println!("v0390_rethink_required\tNO");
        } else {
            println!("v0390_material_dev_gain\tNO");
            println!("v0390_finalize_required\tNO");
            println!("v0390_rethink_required\tYES");
        }
    }
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
            .ok_or_else(|| anyhow::anyhow!("v0.39 record index {index} outside corpus"))?;
        let Some(target) = record
            .context
            .ion_mobility
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("v0.39 missing provenance for record {index}"))?;
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
        let (affine, shared_identities) = if examples.len() >= V039_MIN_SOURCE_SHARED_IDENTITIES {
            let raw = fit_xy_affine(examples)?;
            let blend = examples.len() as f64 / (examples.len() as f64 + V039_SOURCE_SHRINKAGE);
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
        if count >= V039_MIN_SOURCE_SHARED_IDENTITIES && supervision.residual_mae.is_finite() {
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
            if supervision.shared_identities < V039_MIN_SOURCE_SHARED_IDENTITIES {
                0.60
            } else {
                (2.0 / (1.0 + supervision.residual_mae / global_residual_scale))
                    .clamp(V039_RELIABILITY_FLOOR, V039_RELIABILITY_CEILING)
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
                .ok_or_else(|| anyhow::anyhow!("v0.39 family has no representative"))?;
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
            (V039_SINGLETON_WEIGHT_SCALE * mean_reliability).clamp(0.25, 0.80)
        } else {
            multisource_examples += 1;
            let dispersion_factor = 1.0 / (1.0 + dispersion / V039_CONSENSUS_DISPERSION_SCALE);
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
            .ok_or_else(|| anyhow::anyhow!("v0.39 consensus identity has no representative"))?;
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
        output_root.join("mobility_source_reliability_v0390.tsv"),
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
        output_root.join("mobility_consensus_supervision_v0390.tsv"),
        summary,
    )?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct MobilityLossScales {
    mobility: f64,
    ccs: f64,
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

fn predicted_native_mobility_and_ccs(
    model: &PeptideFoundationMultimodalV0390Model,
    batch: &redeem_properties::foundation::FoundationTrainingBatch,
    physics: &FoundationScalarPhysicsBatchV0360,
    normalization: &FoundationTargetNormalizationConfig,
    train: bool,
) -> Result<(Tensor, Tensor)> {
    let output = model.mobility_v0390_t(&batch.input, &batch.context, physics, train)?;
    let base_ccs_native = normalization
        .ccs
        .denormalize_tensor(&output.base_ccs_model)?;
    let factor = bruker_ccs_factor(&batch.context)?;
    let base_mobility = base_ccs_native.broadcast_div(&factor)?;
    let predicted_mobility = (&base_mobility + &output.mobility_residual_native)?;
    let predicted_ccs = predicted_mobility.broadcast_mul(&factor)?;
    Ok((predicted_mobility, predicted_ccs))
}

fn mobility_consensus_loss(
    model: &PeptideFoundationMultimodalV0390Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    normalization: &FoundationTargetNormalizationConfig,
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
                .ok_or_else(|| anyhow::anyhow!("v0.39 representative index outside corpus"))
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
        model.forward_config().max_sequence_len,
        device,
    )?;
    let (predicted_mobility, predicted_ccs) =
        predicted_native_mobility_and_ccs(model, &batch, &physics, normalization, true)?;
    let target_mobility = Tensor::from_vec(targets, (examples.len(), 1), device)?;
    let factor = bruker_ccs_factor(&batch.context)?;
    let target_ccs = target_mobility.broadcast_mul(&factor)?;
    let weight = Tensor::from_vec(weights, (examples.len(), 1), device)?;
    let mask = Tensor::from_vec(vec![1.0f32; examples.len()], (examples.len(), 1), device)?;

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

    Ok((((mobility_mse.affine(V039_MOBILITY_MSE_WEIGHT, 0.0)?
        + mobility_robust.affine(V039_MOBILITY_ROBUST_WEIGHT, 0.0)?)?
        + ccs_mse.affine(V039_CCS_AUX_MSE_WEIGHT, 0.0)?)?
        + ccs_robust.affine(V039_CCS_AUX_ROBUST_WEIGHT, 0.0)?)?)
}

fn calibrate_mobility_residual_scales(
    model: &PeptideFoundationMultimodalV0390Model,
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
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &owned,
            model.forward_config().max_sequence_len,
            device,
        )?;
        let (predicted_mobility, predicted_ccs) =
            predicted_native_mobility_and_ccs(model, &batch, &physics, normalization, false)?;
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
        anyhow::bail!("v0.39 TRAIN loss calibration has no finite weighted labels");
    }
    Ok(MobilityLossScales {
        mobility: (mobility_squared / weight_sum)
            .sqrt()
            .max(V039_MIN_MOBILITY_LOSS_SCALE),
        ccs: (ccs_squared / weight_sum)
            .sqrt()
            .max(V039_MIN_CCS_LOSS_SCALE),
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
            "v0.39 weighted scalar loss shape mismatch: prediction {:?}, target {:?}",
            prediction.dims(),
            target.dims()
        );
    }
    if !(scale > 0.0 && scale.is_finite()) {
        anyhow::bail!("v0.39 invalid loss scale {scale}");
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
        anyhow::bail!("v0.39 invalid robust loss scale={scale} delta={delta}");
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

fn evaluate_raw_ccs_indices(
    model: &PeptideFoundationMultimodalV0390Model,
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
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &owned,
            model.forward_config().max_sequence_len,
            device,
        )?;
        let (_, predicted_ccs) =
            predicted_native_mobility_and_ccs(model, &batch, &physics, normalization, false)?;
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
        anyhow::bail!("v0.39 raw CCS evaluation contains no labels");
    }
    Ok(absolute_error / count as f64)
}

fn evaluate_consensus_targets(
    model: &PeptideFoundationMultimodalV0390Model,
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
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &owned,
            model.forward_config().max_sequence_len,
            device,
        )?;
        let (_, predicted_ccs) =
            predicted_native_mobility_and_ccs(model, &batch, &physics, normalization, false)?;
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
        anyhow::bail!("v0.39 consensus CCS evaluation contains no labels");
    }
    Ok(absolute_error / count as f64)
}

fn combined_dev_objective(raw_ratio: f64, consensus_ratio: f64) -> f64 {
    V039_RAW_OBJECTIVE_WEIGHT * raw_ratio + V039_CONSENSUS_OBJECTIVE_WEIGHT * consensus_ratio
}

fn normalize_scalar_targets(
    target: &mut Option<Tensor>,
    normalization: &FoundationRegressionNormalization,
) -> Result<()> {
    if let Some(values) = target.take() {
        *target = Some(normalization.normalize_tensor(&values)?);
    }
    Ok(())
}

fn masked_scaled_mse(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!(
            "v0.39 scalar loss shape mismatch: prediction {:?}, target {:?}",
            prediction.dims(),
            target.dims()
        );
    }
    if !(scale > 0.0 && scale.is_finite()) {
        anyhow::bail!("v0.39 invalid CCS loss scale {scale}");
    }
    let mask = mask.broadcast_as(prediction.dims())?;
    let scaled_error = (prediction - target)?.affine(1.0 / scale, 0.0)?;
    let numerator = scaled_error.sqr()?.broadcast_mul(&mask)?.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    Ok(numerator.broadcast_div(&denominator)?)
}

fn masked_scaled_pseudo_huber(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    scale: f64,
    delta: f64,
) -> Result<Tensor> {
    if !(scale > 0.0 && scale.is_finite() && delta > 0.0 && delta.is_finite()) {
        anyhow::bail!("v0.39 invalid robust loss scale={scale} delta={delta}");
    }
    let mask = mask.broadcast_as(prediction.dims())?;
    let scaled = (prediction - target)?.affine(1.0 / (scale * delta), 0.0)?;
    let robust = (scaled.sqr()? + 1.0)?
        .sqrt()?
        .affine(delta * delta, -(delta * delta))?;
    let numerator = robust.broadcast_mul(&mask)?.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    Ok(numerator.broadcast_div(&denominator)?)
}

fn evaluate_properties(
    model: &PeptideFoundationMultimodalV0390Model,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    collator: &FoundationCollator,
    normalization: &FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    device: &Device,
) -> Result<PropertyMetrics> {
    let mut rt_abs = 0.0f64;
    let mut rt_sq = 0.0f64;
    let mut rt_n = 0usize;
    let mut ccs_abs = 0.0f64;
    let mut ccs_sq = 0.0f64;
    let mut ccs_n = 0usize;
    let mut ms2_objective = 0.0f64;
    let mut ms2_batches = 0usize;
    let mut ms2_shape = Ms2ShapeAccumulator::default();

    for chunk in indices.chunks(batch_size) {
        let owned: Vec<FoundationTrainingRecord> =
            chunk.iter().map(|&index| records[index].clone()).collect();
        let mut batch = collator.collate(&owned, device, 0)?;
        normalize_scalar_targets(&mut batch.targets.rt, &normalization.rt)?;
        let fragment = FoundationFragmentContextBatchV0350::from_records(
            &owned,
            model.forward_config(),
            device,
        )?;
        let fragment_mask = fragment.channel_mask()?;
        let protected = model.protected_forward_v0350_t(&batch.input, &batch.context, &fragment)?;
        accumulate_regression(
            &protected.base.rt,
            batch.targets.rt.as_ref(),
            batch.targets.rt_mask.as_ref(),
            &normalization.rt,
            &mut rt_abs,
            &mut rt_sq,
            &mut rt_n,
        )?;

        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &owned,
            model.forward_config().max_sequence_len,
            device,
        )?;
        let (_, predicted_ccs) =
            predicted_native_mobility_and_ccs(model, &batch, &physics, normalization, false)?;
        let predicted_ccs = predicted_ccs.to_vec2::<f32>()?;
        for (row, record) in owned.iter().enumerate() {
            let Some(target) = record.ccs.filter(|value| value.is_finite()) else {
                continue;
            };
            let error = f64::from(predicted_ccs[row][0] - target);
            ccs_abs += error.abs();
            ccs_sq += error * error;
            ccs_n += 1;
        }

        if let (Some(target), Some(mask)) = (&batch.targets.ms2, &batch.targets.ms2_mask) {
            let contextual_mask = mask.broadcast_mul(&fragment_mask)?;
            let components =
                foundation_ms2_loss(&protected.base.ms2, target, &contextual_mask, ms2_loss)?;
            ms2_objective += f64::from(components.total.to_scalar::<f32>()?);
            ms2_batches += 1;
            ms2_shape.accumulate(&protected.base.ms2, target, &contextual_mask)?;
        }
    }

    Ok(PropertyMetrics {
        rt_mae_native: (rt_n > 0).then(|| rt_abs / rt_n as f64),
        rt_rmse_native: (rt_n > 0).then(|| (rt_sq / rt_n as f64).sqrt()),
        ccs_mae_native: (ccs_n > 0).then(|| ccs_abs / ccs_n as f64),
        ccs_rmse_native: (ccs_n > 0).then(|| (ccs_sq / ccs_n as f64).sqrt()),
        ms2_loss: (ms2_batches > 0).then(|| ms2_objective / ms2_batches as f64),
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

fn normalized_dev_objective(metrics: PropertyMetrics, baseline: PropertyMetrics) -> Result<f64> {
    validate_rt_invariance(metrics, baseline)?;
    validate_ms2_invariance(metrics, baseline)?;
    finite_ratio(metrics.ccs_mae_native, baseline.ccs_mae_native, "CCS")
}

fn validate_rt_invariance(metrics: PropertyMetrics, baseline: PropertyMetrics) -> Result<()> {
    for (label, current, initial) in [
        ("RT MAE", metrics.rt_mae_native, baseline.rt_mae_native),
        ("RT RMSE", metrics.rt_rmse_native, baseline.rt_rmse_native),
    ] {
        let current = current.ok_or_else(|| anyhow::anyhow!("missing {label} metric"))?;
        let initial = initial.ok_or_else(|| anyhow::anyhow!("missing baseline {label} metric"))?;
        if (current - initial).abs() > V039_RT_INVARIANCE_TOLERANCE {
            anyhow::bail!(
                "v0.39 changed frozen {label}: baseline={initial:.8} current={current:.8}"
            );
        }
    }
    Ok(())
}

fn validate_ms2_invariance(metrics: PropertyMetrics, baseline: PropertyMetrics) -> Result<()> {
    for (label, current, initial) in [
        ("MS2 loss", metrics.ms2_loss, baseline.ms2_loss),
        (
            "MS2 cosine",
            metrics.ms2_mean_cosine,
            baseline.ms2_mean_cosine,
        ),
        (
            "MS2 spectral angle",
            metrics.ms2_mean_spectral_angle,
            baseline.ms2_mean_spectral_angle,
        ),
        (
            "MS2 Pearson",
            metrics.ms2_mean_pearson,
            baseline.ms2_mean_pearson,
        ),
    ] {
        let current = current.ok_or_else(|| anyhow::anyhow!("missing {label} metric"))?;
        let initial = initial.ok_or_else(|| anyhow::anyhow!("missing baseline {label} metric"))?;
        if (current - initial).abs() > V039_MS2_INVARIANCE_TOLERANCE {
            anyhow::bail!(
                "v0.39 changed frozen {label}: baseline={initial:.8} current={current:.8}"
            );
        }
    }
    Ok(())
}

fn finite_ratio(value: Option<f64>, baseline: Option<f64>, label: &str) -> Result<f64> {
    let value = value.ok_or_else(|| anyhow::anyhow!("missing {label} metric"))?;
    let baseline = baseline.ok_or_else(|| anyhow::anyhow!("missing baseline {label} metric"))?;
    if !(value.is_finite() && baseline.is_finite() && baseline > 0.0) {
        anyhow::bail!("invalid {label} ratio inputs: value={value}, baseline={baseline}");
    }
    Ok(value / baseline)
}

fn print_metrics(label: &str, step: usize, p: PropertyMetrics) {
    println!(
        "{label}_forward\tstep={step}\trt_mae_native={}\trt_rmse_native={}\tccs_mae_native={}\tccs_rmse_native={}\tms2_loss={}\tms2_pointwise_mse={}\tms2_pointwise_mae={}\tms2_cosine={}\tms2_spectral_angle={}\tms2_pearson={}",
        fmt_opt(p.rt_mae_native),
        fmt_opt(p.rt_rmse_native),
        fmt_opt(p.ccs_mae_native),
        fmt_opt(p.ccs_rmse_native),
        fmt_opt(p.ms2_loss),
        fmt_opt(p.ms2_pointwise_mse),
        fmt_opt(p.ms2_pointwise_mae),
        fmt_opt(p.ms2_mean_cosine),
        fmt_opt(p.ms2_mean_spectral_angle),
        fmt_opt(p.ms2_mean_pearson),
    );
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

fn load_v0350_parent_variables(
    varmap: &VarMap,
    checkpoint: &Path,
    device: &Device,
) -> Result<WarmStartReport> {
    let tensors = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("failed to load frozen v0.35 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.39 VarMap lock poisoned during warm start"))?;
    let mut loaded = 0usize;
    let mut fresh = 0usize;
    let mut used = BTreeSet::<String>::new();
    let mut missing = Vec::new();

    for (name, variable) in data.iter() {
        let parent_name = if name.starts_with("ccs_conformer_v0390.") {
            None
        } else if let Some(suffix) = name.strip_prefix("ccs_forward_v0390.") {
            Some(format!("forward_v0350.{suffix}"))
        } else if let Some(suffix) = name.strip_prefix("ccs_property_refinement_v0390.") {
            Some(format!("property_refinement_v0350.{suffix}"))
        } else {
            Some(name.clone())
        };

        let Some(parent_name) = parent_name else {
            fresh += 1;
            continue;
        };
        match tensors.get(&parent_name) {
            Some(tensor) => {
                if tensor.dims() != variable.as_tensor().dims() {
                    anyhow::bail!(
                        "v0.39 warm-start shape mismatch for {name} <- {parent_name}: parent {:?}, model {:?}",
                        tensor.dims(),
                        variable.as_tensor().dims()
                    );
                }
                variable.set(tensor)?;
                used.insert(parent_name);
                loaded += 1;
            }
            None => missing.push(format!("{name} <- {parent_name}")),
        }
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "v0.35 parent is missing required v0.39 reused variables: {}",
            missing.join(", ")
        );
    }
    let ignored = tensors.keys().filter(|name| !used.contains(*name)).count();
    drop(data);
    if loaded == 0 || fresh == 0 {
        anyhow::bail!("v0.39 warm start is nonfunctional: loaded={loaded} fresh={fresh}");
    }
    Ok(WarmStartReport {
        loaded_variables: loaded,
        fresh_variables: fresh,
        ignored_parent_variables: ignored,
    })
}

fn read_v035_metadata(checkpoint: &Path) -> Result<V035ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.35 metadata {path:?}"))
}

fn read_v039_metadata(checkpoint: &Path) -> Result<V039Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.39 metadata {path:?}"))
}

#[allow(clippy::too_many_arguments)]
fn validate_metadata(
    metadata: &V039Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    supervision_fingerprint: &str,
    config: &PeptideFoundationMultimodalV0390Config,
    batch_size: usize,
    seed: u64,
    learning_rate: f64,
    train_steps: usize,
) -> Result<()> {
    if metadata.version != V039_VERSION || metadata.objective != V039_OBJECTIVE {
        anyhow::bail!("v0.39 checkpoint identity mismatch");
    }
    if !(metadata.mobility_loss_scale_native > 0.0
        && metadata.mobility_loss_scale_native.is_finite()
        && metadata.ccs_aux_loss_scale_native > 0.0
        && metadata.ccs_aux_loss_scale_native.is_finite())
    {
        anyhow::bail!("v0.39 checkpoint has invalid mobility/CCS loss scale");
    }
    if metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        || metadata.supervision_fingerprint != supervision_fingerprint
    {
        anyhow::bail!("v0.39 checkpoint data/supervision provenance mismatch");
    }
    if metadata.v0390_config != *config {
        anyhow::bail!("v0.39 checkpoint architecture mismatch");
    }
    if metadata.batch_size != batch_size
        || metadata.seed != seed
        || metadata.train_steps != train_steps
    {
        anyhow::bail!("v0.39 checkpoint run-contract mismatch");
    }
    if (metadata.learning_rate - learning_rate).abs()
        > f64::EPSILON * 64.0 * learning_rate.abs().max(1.0)
    {
        anyhow::bail!("v0.39 checkpoint learning-rate mismatch");
    }
    Ok(())
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V039Metadata,
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

fn copy_checkpoint_dir(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        anyhow::bail!("refusing to overwrite checkpoint directory {destination:?}");
    }
    fs::create_dir_all(destination)?;
    for file in [
        "model.safetensors",
        "optimizer.safetensors",
        "metadata.yaml",
    ] {
        fs::copy(source.join(file), destination.join(file))?;
    }
    Ok(())
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
        anyhow::bail!("v0.39 {label} cannot form one full batch");
    }
    if config.validation_source_weights.is_empty() {
        return Ok(max_batches);
    }
    let mut available = BTreeMap::<String, usize>::new();
    for &index in validation_indices {
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("v0.39 {label} provenance index out of bounds"))?;
        *available.entry(source.source_id.clone()).or_default() += 1;
    }
    let total_weight: f64 = config.validation_source_weights.values().copied().sum();
    if !(total_weight > 0.0 && total_weight.is_finite()) {
        anyhow::bail!("v0.39 {label} validation weights are invalid");
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
    anyhow::bail!("v0.39 {label} cannot satisfy source quotas for one full batch")
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
            "v0.39 {label} sample plan has {} records but {expected} are required",
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
    fn v0390_affine_fit_recovers_linear_map() {
        let examples = vec![(1.0, 5.0), (2.0, 8.0), (3.0, 11.0), (4.0, 14.0)];
        let fit = fit_xy_affine(&examples).unwrap();
        assert!((fit.slope - 3.0).abs() < 1.0e-10);
        assert!((fit.intercept - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn v0390_consensus_prefers_reliable_source() {
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
    fn v0390_overlapping_views_collapse_to_project_families() {
        assert_eq!(source_family("pxd034128_11min"), "pxd034128");
        assert_eq!(
            source_family("pxd058337_60spd_fractionationgpf"),
            "pxd058337"
        );
        assert_eq!(source_family("ip2_bruker_human"), "ip2_bruker_human");
    }

    #[test]
    fn v0390_epoch_order_is_deterministic_and_unique_within_cycle() {
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
