//! v0.34 deep task-specialist forward continuation from frozen v0.31.
//!
//! This is one predeclared architecture experiment, not a hyperparameter sweep.
//! The paired historical v0.31 evaluation confirmed real RT/MS2 gains but closed
//! only a minority of the external AlphaPeptDeep gap. v0.34 therefore freezes the
//! complete accepted v0.31 model and trains two independent forward specialists:
//! a multi-scale local-motif + deep sequence RT tower and a four-layer 256d MS2
//! residue Transformer with fragment-conditioned residual decoding. CCS remains
//! the exact protected v0.27/v0.31 path. Supervised specialist training uses clean
//! peptide inputs rather than masked/corrupted pretraining views.
//!
//! Historical VALIDATION has already been consumed descriptively by v0.31 and is
//! not reused for v0.34 selection. Historical TEST remains closed.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    contrastive_info_nce_loss, foundation_causal_conditioning_margin_loss,
    foundation_causal_next_token_loss, foundation_diffusion_length_loss,
    foundation_diffusion_open_ptm_mass_loss, foundation_diffusion_x0_loss,
    foundation_fragment_relation_features, foundation_fragment_relation_validate_mass_geometry,
    foundation_ms2_loss, foundation_multimodal_ms2_loss_v0340,
    foundation_multimodal_relation_margin_loss_v0340, foundation_peptidoform_neutral_mass,
    foundation_spectrum_peptide_alignment_loss, load_foundation_corpus,
    load_unified_foundation_components, multi_task_loss_with_ms2_config,
    read_foundation_training_run_config, sample_foundation_training_indices,
    sample_foundation_validation_indices, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCausalBatch, FoundationCausalCollator,
    FoundationCausalOutput, FoundationCheckpointMetadata, FoundationCollator,
    FoundationCollatorConfig, FoundationCorruptionConfig, FoundationDiffusionCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationFragmentContextBatchV0340,
    FoundationLearningRateSchedule, FoundationLossWeights, FoundationMs2LossConfig,
    FoundationMs2OutputActivation, FoundationPartition, FoundationRegressionNormalization,
    FoundationRegressionNormalizationStrategy, FoundationSamplePlan, FoundationSamplingConfig,
    FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator,
    FoundationTargetNormalizationConfig, FoundationTrainingRecord, FoundationTrainingViews,
    PeptideFoundationMultimodalV0310Config, PeptideFoundationMultimodalV0340Config,
    PeptideFoundationMultimodalV0340Model, PeptidoformInput, PrecursorContextBatch,
    RetentionTimeObjective, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240,
    FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240,
    FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240, FOUNDATION_MS2_SOFTPLUS_BETA_V0138,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0340, FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0340,
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0340, FOUNDATION_OPEN_PTM_MASS_SCALE_DA,
    FOUNDATION_OPEN_PTM_VOCAB_SIZE,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct InverseCheckpointMetadata {
    diffusion: FoundationDiffusionConfig,
}

#[derive(Debug, Deserialize)]
struct UnifiedParentMetadata {
    version: u32,
    objective: String,
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
    v0310_config: PeptideFoundationMultimodalV0310Config,
    completed_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UnifiedPilotMetadata {
    version: u32,
    objective: String,
    schedule: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    forward_checkpoint: String,
    diffusion_checkpoint: String,
    causal_checkpoint: String,
    parent_unified_checkpoint: Option<String>,
    rt_objective: RetentionTimeObjective,
    rt_harmonization_calibration: Option<String>,
    target_normalization: FoundationTargetNormalizationConfig,
    train_steps: usize,
    batch_size: usize,
    validation_batches: usize,
    seed: u64,
    learning_rate: f64,
    max_gradient_norm: f64,
    diffusion_length_weight: f64,
    open_ptm_mass_loss_weight: f64,
    alignment_weight: f64,
    alignment_temperature: f64,
    alignment_initialization: String,
    alignment_initialization_seed: u64,
    alignment_initialization_fingerprint: String,
    forward_objective_weight: f64,
    diffusion_objective_weight: f64,
    causal_objective_weight: f64,
    relation_objective_weight: f64,
    relation_margin: f64,
    property_adapter_hidden: usize,
    causal_conditioning_margin_weight: f64,
    causal_conditioning_margin_nats: f64,
    causal_conditioning_negative: String,
    ms2_loss: FoundationMs2LossConfig,
    ms2_output_activation: FoundationMs2OutputActivation,
    ms2_head_reset: String,
    ms2_head_reset_channel: Option<usize>,
    ms2_head_fingerprint_before_reset: Option<String>,
    ms2_head_fingerprint_after_reset: Option<String>,
    completed_steps: usize,
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
    v0340_config: PeptideFoundationMultimodalV0340Config,
    warm_start_checkpoint: String,
    warm_start_loaded_variables: usize,
    warm_start_fresh_variables: usize,
    warm_start_ignored_parent_variables: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ms2HeadResetMode {
    None,
    B2ZeroV0139,
}

impl Ms2HeadResetMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::B2ZeroV0139 => "b2-zero-v0139",
        }
    }
}

impl std::str::FromStr for Ms2HeadResetMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "b2-zero-v0139" | "b2-zero" => Ok(Self::B2ZeroV0139),
            other => Err(format!(
                "unsupported MS2 head reset {other:?}; expected none or b2-zero-v0139"
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct Ms2HeadResetReport {
    mode: Ms2HeadResetMode,
    channel: Option<usize>,
    fingerprint_before: Option<String>,
    fingerprint_after: Option<String>,
}

impl Ms2HeadResetReport {
    fn none() -> Self {
        Self {
            mode: Ms2HeadResetMode::None,
            channel: None,
            fingerprint_before: None,
            fingerprint_after: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct V034WarmStartReport {
    loaded_variables: usize,
    fresh_variables: usize,
    ignored_parent_variables: usize,
}

#[derive(Debug, Clone, Copy, Default)]
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
    ms2_exact_zero_fraction: Option<f64>,
    ms2_near_zero_1e4_fraction: Option<f64>,
    ms2_near_zero_1e3_fraction: Option<f64>,
    ms2_near_zero_1e2_fraction: Option<f64>,
    ms2_mean_predicted_intensity: Option<f64>,
    ms2_mean_target_intensity: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct InverseMetrics {
    diffusion_loss: f64,
    diffusion_length_loss: f64,
    diffusion_token_accuracy: f64,
    causal_loss: f64,
    causal_perplexity: f64,
    causal_token_accuracy: f64,
    causal_shuffled_loss: f64,
    causal_conditioning_gap: f64,
    causal_conditioning_margin_loss: f64,
    causal_conditioning_preference_fraction: f64,
    causal_conditioning_margin_satisfied_fraction: f64,
    alignment_loss: f64,
    alignment_retrieval_top1: f64,
    shuffled_alignment_loss: f64,
    shuffled_alignment_retrieval_top1: f64,
    relation_margin_loss: f64,
    relation_preference_fraction: f64,
}

#[derive(Debug)]
struct CausalTrainObjective {
    total: Tensor,
    matched_ce: f64,
    shuffled_ce: Option<f64>,
    conditioning_gap: Option<f64>,
    conditioning_margin_loss: f64,
    alignment_loss: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 11 {
        anyhow::bail!(
            "usage: foundation_train_multimodal_v0340 RUN_V0260.yaml OUTPUT_DIR PARENT_V0310_CHECKPOINT [max_epochs=10] [batch_size=32] [patience=3] [min_delta=0.002] [seed=20261034] [learning_rate=5e-5] [mode=train|resume|finalize]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let max_epochs = parse_or(&args, 4, 10usize)?;
    let batch_size = parse_or(&args, 5, 32usize)?;
    let patience = parse_or(&args, 6, 3usize)?;
    let min_delta = parse_or(&args, 7, 0.002f64)?;
    let seed = parse_or(&args, 8, 20_261_034u64)?;
    let learning_rate = parse_or(&args, 9, 5.0e-5f64)?;
    let run_mode = args.get(10).map(String::as_str).unwrap_or("train");
    let (resume_training, finalize_only) = match run_mode {
        "train" => (false, false),
        "resume" => (true, false),
        "finalize" => (false, true),
        other => anyhow::bail!(
            "unsupported v0.34 run mode {other:?}; expected train, resume, or finalize"
        ),
    };

    if max_epochs == 0 || batch_size < 2 || patience == 0 {
        anyhow::bail!("max_epochs/patience must be positive and batch_size must be >=2");
    }
    if !(min_delta >= 0.0 && min_delta.is_finite()) {
        anyhow::bail!("min_delta must be finite and non-negative");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("learning_rate must be finite and positive");
    }
    if finalize_only || resume_training {
        if !output_root.is_dir() {
            anyhow::bail!(
                "v0.34 {run_mode} mode requires an existing model output directory: {:?}",
                output_root
            );
        }
        let required_checkpoints: &[&str] = if finalize_only {
            &["best"]
        } else {
            &["initial", "latest", "best"]
        };
        for checkpoint in required_checkpoints {
            for name in [
                "model.safetensors",
                "optimizer.safetensors",
                "metadata.yaml",
            ] {
                let path = output_root.join(checkpoint).join(name);
                if !path.is_file() {
                    anyhow::bail!("v0.34 {run_mode} mode is missing checkpoint file: {path:?}");
                }
            }
        }
    } else if output_root.exists() {
        anyhow::bail!("v0.34 output directory already exists: {:?}", output_root);
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.34 requires a CUDA device")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    // This lane is intentionally TRAIN-derived only. The derived benchmark's
    // Validation partition is TRAIN-dev, and Test is TRAIN-holdout.
    let train_forward_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .collect();
    let dev_forward_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Validation)
        .map(|entry| entry.record_index)
        .collect();
    let holdout_forward_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .map(|entry| entry.record_index)
        .collect();

    let parent_metadata = read_unified_parent_metadata(&parent_checkpoint)?;
    if parent_metadata.version != 310 {
        anyhow::bail!(
            "v0.34 requires the frozen v0.31 parent metadata version 310, observed {}",
            parent_metadata.version
        );
    }
    if parent_metadata.objective
        != "v0310_protected_ccs_fragment_aware_property_view_v027_heads_openptm32"
    {
        anyhow::bail!(
            "v0.34 requires the accepted v0.31 objective, observed {}",
            parent_metadata.objective
        );
    }
    if parent_metadata.completed_steps == 0 {
        anyhow::bail!("v0.34 refuses an unselected v0.31 baseline checkpoint");
    }
    let forward_config = parent_metadata.forward_config.clone();
    let inverse_config = parent_metadata.inverse_config.clone();
    forward_config.validate().map_err(anyhow::Error::msg)?;
    inverse_config.validate().map_err(anyhow::Error::msg)?;
    parent_metadata.v0310_config.validate()?;
    if parent_metadata.v0310_config.forward() != &forward_config
        || parent_metadata.v0310_config.inverse() != &inverse_config
    {
        anyhow::bail!("v0.34 parent v0.31 architecture config does not match its base configs");
    }
    let v0340_config =
        PeptideFoundationMultimodalV0340Config::fixed(parent_metadata.v0310_config.clone())?;

    // The v0.34 prepare step excludes peptides that cannot be represented by
    // the frozen architecture. Re-check that invariant here so a malformed or
    // stale prepared manifest fails before optimizer initialization rather than
    // part-way through an epoch. Peptides are never truncated.
    let mut prepared_max_sequence_len = 0usize;
    let mut prepared_overlength_records = 0usize;
    let mut first_overlength: Option<(usize, String, usize)> = None;
    for entry in &benchmark.entries {
        let sequence = &corpus.records[entry.record_index].peptidoform.sequence;
        let sequence_len = sequence.chars().count();
        prepared_max_sequence_len = prepared_max_sequence_len.max(sequence_len);
        if sequence_len > forward_config.max_sequence_len {
            prepared_overlength_records += 1;
            if first_overlength.is_none() {
                first_overlength = Some((entry.record_index, sequence.clone(), sequence_len));
            }
        }
    }
    if let Some((record_index, sequence, sequence_len)) = first_overlength {
        anyhow::bail!(
            "v0.34 prepared benchmark contains {prepared_overlength_records} peptide(s) longer than architecture max_sequence_len={}; first record_index={record_index} length={sequence_len} sequence={sequence}",
            forward_config.max_sequence_len
        );
    }

    let vocabulary = FoundationDiffusionVocabulary;
    let train_inverse_indices = usable_inverse_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &inverse_config,
        vocabulary,
    );
    let dev_inverse_indices = usable_inverse_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &inverse_config,
        vocabulary,
    );
    let holdout_inverse_indices = usable_inverse_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Test,
        &inverse_config,
        vocabulary,
    );

    for (label, n) in [
        ("train_forward", train_forward_indices.len()),
        ("train_inverse", train_inverse_indices.len()),
        ("dev_forward", dev_forward_indices.len()),
        ("dev_inverse", dev_inverse_indices.len()),
        ("holdout_forward", holdout_forward_indices.len()),
        ("holdout_inverse", holdout_inverse_indices.len()),
    ] {
        if n < batch_size {
            anyhow::bail!("insufficient {label} records for batch_size={batch_size}: {n}");
        }
    }

    let steps_per_epoch = (train_forward_indices.len().min(train_inverse_indices.len())
        / batch_size)
        .min(1_536)
        .max(1);
    let requested_dev_batches = (dev_forward_indices.len().min(dev_inverse_indices.len())
        / batch_size)
        .min(256)
        .max(1);
    let requested_holdout_batches = (holdout_forward_indices
        .len()
        .min(holdout_inverse_indices.len())
        / batch_size)
        .min(512)
        .max(1);
    let max_total_steps = max_epochs.saturating_mul(steps_per_epoch);

    let mut forward_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_forward_indices,
        &dev_forward_indices,
    );
    forward_sampling.train_steps_per_epoch = Some(steps_per_epoch);
    let mut inverse_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_inverse_indices,
        &dev_inverse_indices,
    );
    inverse_sampling.train_steps_per_epoch = Some(steps_per_epoch);

    // Historical validation-source weights describe the desired source mixture,
    // but the new TRAIN-derived DEV partition can contain fewer records from a
    // rare source than the old fixed validation budget requested. Validation is
    // strictly without replacement, so choose the largest common batch count
    // whose exact weighted quotas fit both forward and inverse DEV pools.
    let dev_batches = feasible_validation_batches(
        "dev_forward",
        &forward_sampling,
        &corpus.provenance,
        &dev_forward_indices,
        batch_size,
        requested_dev_batches,
    )?
    .min(feasible_validation_batches(
        "dev_inverse",
        &inverse_sampling,
        &corpus.provenance,
        &dev_inverse_indices,
        batch_size,
        requested_dev_batches,
    )?);
    forward_sampling.validation_steps = Some(dev_batches);
    inverse_sampling.validation_steps = Some(dev_batches);

    let dev_forward_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &dev_forward_indices,
        batch_size,
        seed ^ 0x3d13_7f24_559c_81e7,
        &forward_sampling,
    )?;
    let dev_inverse_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &dev_inverse_indices,
        batch_size,
        seed ^ 0x72a4_c11d_0b95_e683,
        &inverse_sampling,
    )?;
    let dev_forward = dev_forward_plan.indices.clone();
    let dev_inverse = dev_inverse_plan.indices.clone();

    // Filter validation weights against HOLDOUT independently of DEV, then
    // choose a quota-feasible common holdout size. This keeps HOLDOUT fixed and
    // without replacement while avoiding assumptions about per-source counts.
    let mut holdout_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_forward_indices,
        &holdout_forward_indices,
    );
    let mut holdout_inverse_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_inverse_indices,
        &holdout_inverse_indices,
    );
    let holdout_batches = feasible_validation_batches(
        "holdout_forward",
        &holdout_sampling,
        &corpus.provenance,
        &holdout_forward_indices,
        batch_size,
        requested_holdout_batches,
    )?
    .min(feasible_validation_batches(
        "holdout_inverse",
        &holdout_inverse_sampling,
        &corpus.provenance,
        &holdout_inverse_indices,
        batch_size,
        requested_holdout_batches,
    )?);
    holdout_sampling.validation_steps = Some(holdout_batches);
    holdout_inverse_sampling.validation_steps = Some(holdout_batches);
    let holdout_forward_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &holdout_forward_indices,
        batch_size,
        seed ^ 0x484f_4c44_4f55_5430,
        &holdout_sampling,
    )?;
    let holdout_inverse_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &holdout_inverse_indices,
        batch_size,
        seed ^ 0x484f_4c44_4f55_5431,
        &holdout_inverse_sampling,
    )?;

    // Full inverse-corpus PTM preflight happens before model/optimizer construction.
    // This prevents another one-off PTM failure after GPU optimization has started.
    let open_ptm_audit = audit_open_ptm_encoding(
        &corpus.records,
        [
            &train_inverse_indices[..],
            &dev_inverse_indices[..],
            &holdout_inverse_indices[..],
        ],
        inverse_config.max_tokens,
        FOUNDATION_OPEN_PTM_MASS_SCALE_DA,
    )?;
    let fragment_geometry_audit = audit_fragment_relation_mass_geometry(
        &corpus.records,
        [
            &train_forward_indices[..],
            &dev_forward_indices[..],
            &holdout_forward_indices[..],
        ],
    )?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultimodalV0340Model::new(v0340_config.clone(), vb)?;

    // Restore the complete accepted v0.31 model exactly. Only the two deep
    // v0.34 specialist namespaces are fresh. Their output projections are zero
    // initialized, so step 0 is an exact functional copy of v0.31.
    let warm_start_report = load_v0310_unchanged_variables(
        &varmap,
        &parent_checkpoint.join("model.safetensors"),
        &device,
    )?;
    let warm_start_description = format!(
        "v0310_all_namespaces_plus_fresh_rt_ms2_deep_specialists_from={}",
        parent_checkpoint.display()
    );
    let alignment_initialization = "v0310_frozen_checkpoint".to_string();
    let alignment_initialization_seed = 0u64;
    let alignment_initialization_fingerprint = alignment_projection_fingerprint(&varmap)?;
    let ms2_head_reset_report = Ms2HeadResetReport::none();

    let forward_trainer = &run.trainer;
    let rt_objective = run.trainer.collator.retention_time_objective;
    let rt_harmonization_calibration = rt_harmonization_calibration_id(&run)?;
    if rt_objective == RetentionTimeObjective::Harmonized {
        if rt_harmonization_calibration.is_none() {
            anyhow::bail!("harmonized RT objective requires source RT harmonization transforms");
        }
        require_harmonized_rt_coverage(
            &corpus.records,
            &benchmark,
            &[FoundationPartition::Train, FoundationPartition::Validation],
        )?;
    }

    let mut target_normalization = forward_trainer.target_normalization;
    if rt_objective == RetentionTimeObjective::Harmonized {
        target_normalization.rt = run.trainer.target_normalization.rt;
        target_normalization.rt.mean = None;
        target_normalization.rt.standard_deviation = None;
        target_normalization.rt.label_count = 0;
        if target_normalization.rt.strategy
            != FoundationRegressionNormalizationStrategy::TrainStandardize
        {
            anyhow::bail!("harmonized RT training requires train-standardize normalization");
        }
        let train_harmonized_rt = train_forward_indices
            .iter()
            .filter_map(|&index| corpus.records[index].retention_time.harmonized);
        target_normalization
            .rt
            .resolve_from_values(train_harmonized_rt)?;
    }
    // Re-fit CCS scaling on TRAIN-core as well; never reuse laptop-era stats.
    target_normalization.ccs.mean = None;
    target_normalization.ccs.standard_deviation = None;
    target_normalization.ccs.label_count = 0;
    if target_normalization.ccs.strategy
        == FoundationRegressionNormalizationStrategy::TrainStandardize
    {
        let train_ccs = train_forward_indices
            .iter()
            .filter_map(|&index| corpus.records[index].ccs);
        target_normalization.ccs.resolve_from_values(train_ccs)?;
    }

    let clean_collator = FoundationCollator::new(
        forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let diffusion_collator = FoundationDiffusionCollator::new_open_ptm(inverse_config.clone())?;
    let causal_collator = FoundationCausalCollator::new_open_ptm(inverse_config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(inverse_config.spectrum.clone())?;

    let ms2_loss = FoundationMs2LossConfig {
        pointwise_weight: 1.0,
        cosine_weight: 0.25,
        ..FoundationMs2LossConfig::default()
    };
    ms2_loss.validate().map_err(anyhow::Error::msg)?;
    // v0.34 is intentionally a forward-only specialist experiment. The
    // complete v0.31 parent, including inverse/alignment paths, is frozen.
    let forward_objective_weight = 1.00f64;
    let diffusion_objective_weight = 0.00f64;
    let causal_objective_weight = 0.00f64;
    let relation_objective_weight = 0.00f64;
    let alignment_weight = 0.05f64;
    let alignment_temperature = 0.07f64;
    let causal_conditioning_margin_weight = 0.0f64;
    let causal_conditioning_margin_nats = 0.25f64;
    let max_gradient_norm = 1.0f64;
    let diffusion_length_weight = 0.1f64;
    let open_ptm_mass_loss_weight = 0.10f64;
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 500u64.min(max_total_steps.saturating_sub(1) as u64),
        total_steps: max_total_steps as u64,
        min_lr_ratio: 0.05,
    };

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: forward_trainer.adam_beta1,
            beta2: forward_trainer.adam_beta2,
            epsilon: forward_trainer.adam_epsilon,
            weight_decay: forward_trainer.weight_decay,
        },
    )?;

    fs::create_dir_all(&output_root)?;
    println!("v0340_version\tv0.34-deep-task-specialist-forward");
    println!("objective\tv0340_deep_rt_ms2_specialists_from_frozen_v0310");
    println!("architecture\t{}", FOUNDATION_MULTIMODAL_ARCHITECTURE_V0340);
    println!("inverse_ptm_encoding\topen_site_token_plus_continuous_signed_mass_v1");
    println!(
        "legacy_diffusion_vocabulary_size\t{}",
        FOUNDATION_DIFFUSION_VOCAB_SIZE
    );
    println!(
        "open_ptm_vocabulary_size\t{}",
        FOUNDATION_OPEN_PTM_VOCAB_SIZE
    );
    println!(
        "open_ptm_mass_scale_da\t{}",
        FOUNDATION_OPEN_PTM_MASS_SCALE_DA
    );
    println!("parent_checkpoint\t{}", parent_checkpoint.display());
    println!("parent_checkpoint_weights_loaded\ttrue_all_v0310_namespaces");
    println!("random_initialization\trt_specialist_v0340_plus_ms2_specialist_v0340_only");
    println!(
        "warm_start_loaded_variables\t{}",
        warm_start_report.loaded_variables
    );
    println!(
        "warm_start_fresh_variables\t{}",
        warm_start_report.fresh_variables
    );
    println!(
        "warm_start_ignored_parent_variables\t{}",
        warm_start_report.ignored_parent_variables
    );
    println!("device\t{:?}", device);
    println!(
        "architecture_max_sequence_len\t{}",
        forward_config.max_sequence_len
    );
    println!("prepared_max_sequence_len\t{prepared_max_sequence_len}");
    println!("prepared_overlength_records\t{prepared_overlength_records}");
    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!(
        "benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        benchmark.manifest_fingerprint()
    );
    println!("historical_validation_reused_for_v0340_selection\tNO");
    println!("historical_test_consumed\tNO");
    println!("train_core_records\t{}", train_forward_indices.len());
    println!("train_dev_records\t{}", dev_forward_indices.len());
    println!("train_holdout_records\t{}", holdout_forward_indices.len());
    println!("train_inverse_records\t{}", train_inverse_indices.len());
    println!(
        "train_raw_spectrum_presence_records\t{}",
        train_forward_indices
            .iter()
            .filter(|&&index| !corpus.records[index].observed_spectrum_peaks.is_empty())
            .count()
    );
    println!("dev_inverse_records\t{}", dev_inverse_indices.len());
    println!("holdout_inverse_records\t{}", holdout_inverse_indices.len());
    println!("steps_per_epoch\t{steps_per_epoch}");
    println!("requested_dev_batches\t{requested_dev_batches}");
    println!("dev_batches\t{dev_batches}");
    println!("requested_holdout_batches\t{requested_holdout_batches}");
    println!("holdout_batches\t{holdout_batches}");
    println!(
        "validation_sampling_policy\thistorical_source_weights_quota_capped_without_replacement"
    );
    println!("max_epochs\t{max_epochs}");
    println!("early_stopping_patience\t{patience}");
    println!("early_stopping_min_delta\t{min_delta}");
    println!("base_learning_rate\t{learning_rate}");
    println!("lr_schedule\twarmup_cosine_500_to_0.05");
    println!("batch_size\t{batch_size}");
    println!("forward_objective_weight\t{forward_objective_weight}");
    println!("diffusion_objective_weight\t{diffusion_objective_weight}");
    println!("causal_objective_weight\t{causal_objective_weight}");
    println!("relation_objective_weight\t{relation_objective_weight}");
    println!(
        "relation_margin\t{}",
        FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0340
    );
    println!(
        "property_adapter_hidden\t{}",
        FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0340
    );
    println!("rt_specialist\tmultiscale_conv_3_5_7_plus_3x256_transformer_learned_query_pool");
    println!("ms2_specialist\t4x256_residue_transformer_plus_fragment_context_residual_decoder");
    println!("specialist_training_input\tclean_peptide_no_mask_corruption");
    println!("accepted_parent\tcomplete_v0310_frozen");
    println!("base_encoder_update_policy\tfrozen_v0310_stop_gradient");
    println!("ccs_update_policy\texact_frozen_v0310_v0270_path");
    println!("training_objectives\tforward_only_clean_rt_plus_factorized_ms2");
    println!("dev_selection\trt_and_all_three_ms2_quality_metrics_must_improve");
    println!("dev_materiality_gate\tall_rt_ms2_quality_metrics_improve_and_combined_error_objective_le_0.925");
    println!(
        "historical_validation_status\talready_consumed_by_v0310_not_available_for_v0340_selection"
    );
    println!("ccs_physics_path\tpreserved_exactly_from_v0310_v0270");
    println!("open_ptm_mass_loss_weight\t{open_ptm_mass_loss_weight}");
    println!("alignment_weight\t{alignment_weight}");
    println!("alignment_temperature\t{alignment_temperature}");
    println!("ms2_pointwise_weight\t{}", ms2_loss.pointwise_weight);
    println!("ms2_cosine_weight\t{}", ms2_loss.cosine_weight);
    println!("ms2_output_activation\tsoftplus-v0138");
    println!("warm_start\t{warm_start_description}");
    println!("alignment_initialization\t{alignment_initialization}");
    println!("alignment_initialization_fingerprint\t{alignment_initialization_fingerprint}");
    print_sample_plan("dev_forward", &dev_forward_plan);
    print_sample_plan("dev_inverse", &dev_inverse_plan);
    print_sample_plan("holdout_forward_reserved", &holdout_forward_plan);
    print_sample_plan("holdout_inverse_reserved", &holdout_inverse_plan);

    println!("open_ptm_preflight_status\tPASS");
    println!("open_ptm_preflight_records\t{}", open_ptm_audit.records);
    println!(
        "open_ptm_preflight_modifications\t{}",
        open_ptm_audit.modifications
    );
    println!(
        "open_ptm_preflight_known_unimod\t{}",
        open_ptm_audit.known_unimod
    );
    println!("open_ptm_preflight_open_mass\t{}", open_ptm_audit.open_mass);
    println!(
        "open_ptm_preflight_min_mass_delta_da\t{}",
        open_ptm_audit.min_mass.unwrap_or(0.0)
    );
    println!(
        "open_ptm_preflight_max_mass_delta_da\t{}",
        open_ptm_audit.max_mass.unwrap_or(0.0)
    );
    println!("fragment_mass_geometry_encoding\tcontinuous_peptidoform_mass_delta_v1");
    println!("fragment_mass_geometry_preflight_status\tPASS");
    println!(
        "fragment_mass_geometry_preflight_records\t{}",
        fragment_geometry_audit.records
    );
    println!(
        "fragment_mass_geometry_preflight_modifications\t{}",
        fragment_geometry_audit.modifications
    );
    println!(
        "fragment_mass_geometry_preflight_open_mass\t{}",
        fragment_geometry_audit.open_mass
    );
    println!(
        "fragment_mass_geometry_preflight_min_mass_delta_da\t{}",
        fragment_geometry_audit.min_mass.unwrap_or(0.0)
    );
    println!(
        "fragment_mass_geometry_preflight_max_mass_delta_da\t{}",
        fragment_geometry_audit.max_mass.unwrap_or(0.0)
    );

    println!(
        "run_mode\t{}",
        if finalize_only {
            "finalize"
        } else if resume_training {
            "resume"
        } else {
            "train"
        }
    );

    let metadata = |completed_steps| UnifiedPilotMetadata {
        version: 340,
        objective: "v0340_deep_rt_ms2_specialists_from_frozen_v0310".into(),
        schedule: "epochwise_no_replacement_train_core+warmup_cosine+train_dev_early_stop".into(),
        corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
        benchmark_manifest_fingerprint: format!(
            "fnv1a64:{:016x}",
            benchmark.manifest_fingerprint()
        ),
        forward_checkpoint: parent_checkpoint.display().to_string(),
        diffusion_checkpoint: parent_checkpoint.display().to_string(),
        causal_checkpoint: parent_checkpoint.display().to_string(),
        parent_unified_checkpoint: Some(parent_checkpoint.display().to_string()),
        rt_objective,
        rt_harmonization_calibration: rt_harmonization_calibration.clone(),
        target_normalization,
        train_steps: max_total_steps,
        batch_size,
        validation_batches: dev_batches,
        seed,
        learning_rate,
        max_gradient_norm,
        diffusion_length_weight,
        open_ptm_mass_loss_weight,
        alignment_weight,
        alignment_temperature,
        alignment_initialization: alignment_initialization.clone(),
        alignment_initialization_seed,
        alignment_initialization_fingerprint: alignment_initialization_fingerprint.clone(),
        forward_objective_weight,
        diffusion_objective_weight,
        causal_objective_weight,
        relation_objective_weight,
        relation_margin: FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0340,
        property_adapter_hidden: FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0340,
        causal_conditioning_margin_weight,
        causal_conditioning_margin_nats,
        causal_conditioning_negative: "disabled_v01310_objective".into(),
        ms2_loss,
        ms2_output_activation: FoundationMs2OutputActivation::SoftplusV0138,
        ms2_head_reset: ms2_head_reset_report.mode.as_str().into(),
        ms2_head_reset_channel: None,
        ms2_head_fingerprint_before_reset: None,
        ms2_head_fingerprint_after_reset: None,
        completed_steps,
        forward_config: forward_config.clone(),
        inverse_config: inverse_config.clone(),
        v0340_config: v0340_config.clone(),
        warm_start_checkpoint: parent_checkpoint.display().to_string(),
        warm_start_loaded_variables: warm_start_report.loaded_variables,
        warm_start_fresh_variables: warm_start_report.fresh_variables,
        warm_start_ignored_parent_variables: warm_start_report.ignored_parent_variables,
    };

    if finalize_only {
        let best_dir = output_root.join("best");
        let best_metadata = read_unified_pilot_metadata(&best_dir)?;
        let expected_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
        let expected_benchmark_fingerprint =
            format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
        if best_metadata.version != 340 {
            anyhow::bail!(
                "v0.34 finalize mode expected checkpoint metadata version 340, observed {}",
                best_metadata.version
            );
        }
        if best_metadata.completed_steps == 0 {
            anyhow::bail!(
                "v0.34 has no DEV improvement over frozen v0.31; refusing to consume HOLDOUT/finalize baseline"
            );
        }
        if best_metadata.corpus_fingerprint != expected_corpus_fingerprint {
            anyhow::bail!(
                "v0.34 finalize corpus fingerprint mismatch: checkpoint={} current={}",
                best_metadata.corpus_fingerprint,
                expected_corpus_fingerprint
            );
        }
        if best_metadata.benchmark_manifest_fingerprint != expected_benchmark_fingerprint {
            anyhow::bail!(
                "v0.34 finalize benchmark fingerprint mismatch: checkpoint={} current={}",
                best_metadata.benchmark_manifest_fingerprint,
                expected_benchmark_fingerprint
            );
        }
        if best_metadata.batch_size != batch_size {
            anyhow::bail!(
                "v0.34 finalize batch_size mismatch: checkpoint={} requested={batch_size}",
                best_metadata.batch_size
            );
        }
        if best_metadata.seed != seed {
            anyhow::bail!(
                "v0.34 finalize seed mismatch: checkpoint={} requested={seed}",
                best_metadata.seed
            );
        }

        let initial_model = output_root.join("initial/model.safetensors");
        if !initial_model.is_file() {
            anyhow::bail!("v0.34 finalize requires the saved step-0 initial checkpoint");
        }
        varmap
            .load(&initial_model)
            .with_context(|| format!("failed to restore v0.34 initial model {initial_model:?}"))?;
        let initial_dev_metrics = evaluate_all(
            &model,
            &corpus.records,
            &dev_forward_plan.indices,
            &dev_inverse_plan.indices,
            batch_size,
            &clean_collator,
            &diffusion_collator,
            &causal_collator,
            &spectrum_collator,
            &target_normalization,
            ms2_loss,
            alignment_temperature,
            causal_conditioning_margin_nats,
            &device,
        )?;

        let best_model = best_dir.join("model.safetensors");
        varmap
            .load(&best_model)
            .with_context(|| format!("failed to restore best v0.34 model {best_model:?}"))?;
        let best_dev_metrics = evaluate_all(
            &model,
            &corpus.records,
            &dev_forward_plan.indices,
            &dev_inverse_plan.indices,
            batch_size,
            &clean_collator,
            &diffusion_collator,
            &causal_collator,
            &spectrum_collator,
            &target_normalization,
            ms2_loss,
            alignment_temperature,
            causal_conditioning_margin_nats,
            &device,
        )?;
        let best_dev_objective = normalized_dev_objective(best_dev_metrics, initial_dev_metrics)?;
        println!("v0340_finalize_dev_objective\t{best_dev_objective:.8}");
        if best_dev_objective > 0.925 {
            anyhow::bail!(
                "v0.34 DEV gain is below the fixed materiality gate (objective={best_dev_objective:.8} > 0.925); refusing to consume HOLDOUT"
            );
        }
        let best_step = best_metadata.completed_steps;
        let best_epoch = if steps_per_epoch > 0 && best_step % steps_per_epoch == 0 {
            Some(best_step / steps_per_epoch)
        } else {
            None
        };
        println!("v0340_finalize_only\ttrue");
        println!("v0340_finalize_source_checkpoint\t{}", best_dir.display());
        println!("v0340_finalize_best_step\t{best_step}");
        println!(
            "v0340_finalize_best_epoch\t{}",
            best_epoch
                .map(|value| value.to_string())
                .unwrap_or_else(|| "NA".into())
        );

        let holdout_metrics = evaluate_all(
            &model,
            &corpus.records,
            &holdout_forward_plan.indices,
            &holdout_inverse_plan.indices,
            batch_size,
            &clean_collator,
            &diffusion_collator,
            &causal_collator,
            &spectrum_collator,
            &target_normalization,
            ms2_loss,
            alignment_temperature,
            causal_conditioning_margin_nats,
            &device,
        )?;
        print_evaluation("train_holdout_once", best_step, holdout_metrics);
        println!("train_holdout_consumed_for_selection\tNO");
        println!("historical_validation_reused_for_v0340_selection\tNO");
        println!("historical_test_consumed\tNO");

        copy_checkpoint_dir(&best_dir, &output_root.join("final"))?;
        println!("v0340_finalize_complete\tbest_step={best_step}");
        println!("final_checkpoint\t{}", output_root.join("final").display());
        return Ok(());
    }

    let (
        initial_metrics,
        mut global_step,
        mut best_epoch,
        mut best_step,
        mut best_objective,
        mut stale_epochs,
        start_epoch,
    ) = if resume_training {
        let initial_dir = output_root.join("initial");
        let latest_dir = output_root.join("latest");
        let best_dir = output_root.join("best");
        let initial_metadata = read_unified_pilot_metadata(&initial_dir)?;
        let latest_metadata = read_unified_pilot_metadata(&latest_dir)?;
        let best_metadata = read_unified_pilot_metadata(&best_dir)?;
        let expected_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
        let expected_benchmark_fingerprint =
            format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
        for (label, checkpoint_metadata) in [
            ("initial", &initial_metadata),
            ("latest", &latest_metadata),
            ("best", &best_metadata),
        ] {
            validate_v0340_training_checkpoint_metadata(
                label,
                checkpoint_metadata,
                &expected_corpus_fingerprint,
                &expected_benchmark_fingerprint,
                &v0340_config,
                batch_size,
                seed,
                learning_rate,
                max_total_steps,
            )?;
        }
        if initial_metadata.completed_steps != 0 {
            anyhow::bail!(
                "v0.34 resume expected initial checkpoint at step 0, observed {}",
                initial_metadata.completed_steps
            );
        }
        if latest_metadata.completed_steps == 0
            || latest_metadata.completed_steps % steps_per_epoch != 0
        {
            anyhow::bail!(
                "v0.34 resume requires latest checkpoint at a complete epoch boundary; completed_steps={} steps_per_epoch={steps_per_epoch}",
                latest_metadata.completed_steps
            );
        }
        if best_metadata.completed_steps % steps_per_epoch != 0 {
            anyhow::bail!(
                "v0.34 resume requires best checkpoint at baseline or a complete epoch boundary; completed_steps={} steps_per_epoch={steps_per_epoch}",
                best_metadata.completed_steps
            );
        }
        if best_metadata.completed_steps > latest_metadata.completed_steps {
            anyhow::bail!(
                "v0.34 resume best checkpoint step {} is newer than latest step {}",
                best_metadata.completed_steps,
                latest_metadata.completed_steps
            );
        }

        let initial_model = initial_dir.join("model.safetensors");
        varmap
            .load(&initial_model)
            .with_context(|| format!("failed to restore v0.34 initial model {initial_model:?}"))?;
        let initial_metrics = evaluate_all(
            &model,
            &corpus.records,
            &dev_forward,
            &dev_inverse,
            batch_size,
            &clean_collator,
            &diffusion_collator,
            &causal_collator,
            &spectrum_collator,
            &target_normalization,
            ms2_loss,
            alignment_temperature,
            causal_conditioning_margin_nats,
            &device,
        )?;
        print_evaluation("train_dev_initial_resume_reference", 0, initial_metrics);

        let best_model = best_dir.join("model.safetensors");
        varmap
            .load(&best_model)
            .with_context(|| format!("failed to restore v0.34 best model {best_model:?}"))?;
        let best_metrics = evaluate_all(
            &model,
            &corpus.records,
            &dev_forward,
            &dev_inverse,
            batch_size,
            &clean_collator,
            &diffusion_collator,
            &causal_collator,
            &spectrum_collator,
            &target_normalization,
            ms2_loss,
            alignment_temperature,
            causal_conditioning_margin_nats,
            &device,
        )?;
        let best_objective = normalized_dev_objective(best_metrics, initial_metrics)?;

        let latest_model = latest_dir.join("model.safetensors");
        varmap
            .load(&latest_model)
            .with_context(|| format!("failed to restore v0.34 latest model {latest_model:?}"))?;
        let latest_optimizer = latest_dir.join("optimizer.safetensors");
        optimizer
            .load_safetensors(&latest_optimizer)
            .with_context(|| {
                format!("failed to restore v0.34 optimizer state {latest_optimizer:?}")
            })?;
        optimizer.set_step_count(latest_metadata.completed_steps as u64);

        let global_step = latest_metadata.completed_steps;
        let completed_epoch = global_step / steps_per_epoch;
        let best_step = best_metadata.completed_steps;
        let best_epoch = best_step / steps_per_epoch;
        let stale_epochs = completed_epoch.saturating_sub(best_epoch);
        if stale_epochs >= patience {
            println!(
                "v0340_resume_already_early_stopped\tcompleted_epoch={completed_epoch}\tbest_epoch={best_epoch}\tstale_epochs={stale_epochs}\tpatience={patience}"
            );
        }
        println!(
            "v0340_resume\tlatest_epoch={completed_epoch}\tlatest_step={global_step}\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}\tstale_epochs={stale_epochs}"
        );
        (
            initial_metrics,
            global_step,
            best_epoch,
            best_step,
            best_objective,
            stale_epochs,
            completed_epoch + 1,
        )
    } else {
        save_checkpoint(
            &output_root.join("initial"),
            &varmap,
            &optimizer,
            &metadata(0),
        )?;
        let initial_metrics = evaluate_all(
            &model,
            &corpus.records,
            &dev_forward,
            &dev_inverse,
            batch_size,
            &clean_collator,
            &diffusion_collator,
            &causal_collator,
            &spectrum_collator,
            &target_normalization,
            ms2_loss,
            alignment_temperature,
            causal_conditioning_margin_nats,
            &device,
        )?;
        print_evaluation("train_dev_initial", 0, initial_metrics);
        let initial_objective = normalized_dev_objective(initial_metrics, initial_metrics)?;
        println!("train_dev_objective\tepoch=0\tstep=0\tvalue={initial_objective:.8}\tbest=true");
        // v0.34 is an identity-preserving continuation at the accepted output
        // surface. Seed model selection from the frozen v0.31 parent so a
        // worse epoch can never become `best` merely because it is epoch 1.
        save_checkpoint(&output_root.join("best"), &varmap, &optimizer, &metadata(0))?;
        (
            initial_metrics,
            0usize,
            0usize,
            0usize,
            initial_objective,
            0usize,
            1usize,
        )
    };

    let mut stopped_early = stale_epochs >= patience;
    if start_epoch > max_epochs {
        println!(
            "v0340_training_budget_complete\tstart_epoch={start_epoch}\tmax_epochs={max_epochs}\tlatest_step={global_step}"
        );
    } else if stopped_early {
        println!(
            "v0340_training_already_early_stopped\tstart_epoch={start_epoch}\tstale_epochs={stale_epochs}\tpatience={patience}"
        );
    } else {
        for epoch in start_epoch..=max_epochs {
            let forward_a_plan = sample_foundation_training_indices(
                &corpus.records,
                &corpus.provenance,
                &train_forward_indices,
                batch_size,
                epoch as u64,
                seed ^ 0x18e7_64ad_a82f_1121,
                true,
                &forward_sampling,
            )?;
            let forward_b_plan = sample_foundation_training_indices(
                &corpus.records,
                &corpus.provenance,
                &train_forward_indices,
                batch_size,
                epoch as u64,
                seed ^ 0x62c9_e03a_733d_a1b5,
                true,
                &forward_sampling,
            )?;
            for (label, plan) in [
                ("forward_a_train", &forward_a_plan),
                ("forward_b_train", &forward_b_plan),
            ] {
                require_plan_records(label, plan, steps_per_epoch.saturating_mul(batch_size))?;
            }
            println!("v0340_epoch\tstage=start\tepoch={epoch}\tsteps={steps_per_epoch}");

            for local_step in 0..steps_per_epoch {
                global_step += 1;
                let lr = lr_schedule
                    .learning_rate(learning_rate, global_step.saturating_sub(1) as u64)?;
                optimizer.set_learning_rate(lr)?;
                let offset = local_step.saturating_mul(batch_size);

                let forward_a_records: Vec<FoundationTrainingRecord> = forward_a_plan.indices
                    [offset..offset + batch_size]
                    .iter()
                    .map(|&index| corpus.records[index].clone())
                    .collect();
                let forward_b_records: Vec<FoundationTrainingRecord> = forward_b_plan.indices
                    [offset..offset + batch_size]
                    .iter()
                    .map(|&index| corpus.records[index].clone())
                    .collect();
                let forward_a = forward_loss(
                    &model,
                    &clean_collator,
                    &forward_a_records,
                    forward_trainer.loss_weights,
                    ms2_loss,
                    forward_trainer.contrastive_temperature,
                    forward_trainer.shared_gradient_scales.rt_encoder,
                    forward_trainer.shared_gradient_scales.ccs_encoder,
                    &target_normalization,
                    seed ^ (global_step as u64).wrapping_mul(0xa24b_1c62_4073_f5d9),
                    &device,
                )?;
                let forward_b = forward_loss(
                    &model,
                    &clean_collator,
                    &forward_b_records,
                    forward_trainer.loss_weights,
                    ms2_loss,
                    forward_trainer.contrastive_temperature,
                    forward_trainer.shared_gradient_scales.rt_encoder,
                    forward_trainer.shared_gradient_scales.ccs_encoder,
                    &target_normalization,
                    seed ^ (global_step as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
                    &device,
                )?;

                let forward_a_weighted = forward_a.affine(0.5 * forward_objective_weight, 0.0)?;
                let forward_b_weighted = forward_b.affine(0.5 * forward_objective_weight, 0.0)?;
                let total = (forward_a_weighted + forward_b_weighted)?;
                let total_value = f64::from(total.to_scalar::<f32>()?);
                let update = optimizer.backward_step(&total, Some(max_gradient_norm))?;

                if global_step == 1 || global_step % 100 == 0 || local_step + 1 == steps_per_epoch {
                    println!(
                        "v0340_train\tepoch={epoch}\tstep={global_step}\tepoch_step={}\tlr={:.8}\ttotal={total_value:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                        local_step + 1,
                        update.learning_rate,
                        update.gradient_norm,
                        update.gradient_scale,
                    );
                }
            }

            let metrics = evaluate_all(
                &model,
                &corpus.records,
                &dev_forward,
                &dev_inverse,
                batch_size,
                &clean_collator,
                &diffusion_collator,
                &causal_collator,
                &spectrum_collator,
                &target_normalization,
                ms2_loss,
                alignment_temperature,
                causal_conditioning_margin_nats,
                &device,
            )?;
            print_evaluation("train_dev", global_step, metrics);
            let dev_objective = normalized_dev_objective(metrics, initial_metrics)?;
            let improved =
                best_objective.is_infinite() || best_objective - dev_objective > min_delta;
            println!(
                "train_dev_objective\tepoch={epoch}\tstep={global_step}\tvalue={dev_objective:.8}\tprevious_best={}\timproved={improved}",
                if best_objective.is_finite() {
                    format!("{best_objective:.8}")
                } else {
                    "NA".into()
                }
            );

            save_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &optimizer,
                &metadata(global_step),
            )?;
            if improved {
                best_objective = dev_objective;
                best_epoch = epoch;
                best_step = global_step;
                stale_epochs = 0;
                save_checkpoint(
                    &output_root.join("best"),
                    &varmap,
                    &optimizer,
                    &metadata(global_step),
                )?;
                println!(
                    "v0340_best_checkpoint\tepoch={best_epoch}\tstep={best_step}\tdev_objective={best_objective:.8}"
                );
            } else {
                stale_epochs += 1;
            }

            println!(
                "v0340_epoch\tstage=complete\tepoch={epoch}\tstep={global_step}\tstale_epochs={stale_epochs}"
            );
            if stale_epochs >= patience {
                stopped_early = true;
                println!(
                    "v0340_early_stop\tepoch={epoch}\tstep={global_step}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}"
                );
                break;
            }
        }
    }

    // TRAIN mode stops after DEV-only checkpoint selection. HOLDOUT remains
    // untouched until an explicit finalize invocation after the architecture is frozen.
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0340_selection\tNO");
    println!("historical_test_consumed\tNO");
    println!("v0340_training_complete\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}\tstopped_early={stopped_early}");
    if best_step == 0 {
        println!("v0340_no_dev_improvement_over_parent\tYES");
        println!("v0340_material_dev_gain\tNO");
        println!("v0340_finalize_required\tNO");
        println!("v0340_rethink_required\tYES");
    } else if best_objective <= 0.925 {
        println!("v0340_no_dev_improvement_over_parent\tNO");
        println!("v0340_material_dev_gain\tYES");
        println!("v0340_finalize_required\tYES");
        println!("v0340_rethink_required\tNO");
    } else {
        println!("v0340_no_dev_improvement_over_parent\tNO");
        println!("v0340_material_dev_gain\tNO");
        println!("v0340_finalize_required\tNO");
        println!("v0340_rethink_required\tYES");
    }
    println!("best_checkpoint\t{}", output_root.join("best").display());

    Ok(())
}

fn normalized_dev_objective(
    metrics: (PropertyMetrics, InverseMetrics),
    baseline: (PropertyMetrics, InverseMetrics),
) -> Result<f64> {
    let (p, _i) = metrics;
    let (bp, _bi) = baseline;
    let rt = finite_ratio(p.rt_mae_native, bp.rt_mae_native, "RT")?;
    let ccs = finite_ratio(p.ccs_mae_native, bp.ccs_mae_native, "CCS")?;
    if (ccs - 1.0).abs() > 2.0e-4 {
        anyhow::bail!("v0.34 protected CCS drifted from the frozen step-0 path: ratio={ccs:.8}");
    }

    let cosine = quality_error_ratio(p.ms2_mean_cosine, bp.ms2_mean_cosine, "MS2 cosine")?;
    let spectral_angle = quality_error_ratio(
        p.ms2_mean_spectral_angle,
        bp.ms2_mean_spectral_angle,
        "MS2 spectral angle",
    )?;
    let pearson = quality_error_ratio(p.ms2_mean_pearson, bp.ms2_mean_pearson, "MS2 Pearson")?;
    let ms2_error = (cosine + spectral_angle + pearson) / 3.0;

    // A specialist checkpoint is eligible only when RT and every externally
    // relevant MS2 quality metric improve together. This prevents a pointwise
    // loss win from hiding a worse spectral shape/correlation model.
    let all_improve = rt < 1.0 && cosine < 1.0 && spectral_angle < 1.0 && pearson < 1.0;
    if !all_improve {
        let penalty = (rt - 1.0).max(0.0)
            + (cosine - 1.0).max(0.0)
            + (spectral_angle - 1.0).max(0.0)
            + (pearson - 1.0).max(0.0);
        return Ok(1.0 + 0.25 * penalty);
    }
    Ok(0.50 * rt + 0.50 * ms2_error)
}

fn quality_error_ratio(value: Option<f64>, baseline: Option<f64>, label: &str) -> Result<f64> {
    let (value, baseline) = match (value, baseline) {
        (Some(value), Some(baseline)) => (value, baseline),
        _ => anyhow::bail!("v0.34 dev objective is missing {label} metric"),
    };
    if !value.is_finite() || !baseline.is_finite() || baseline >= 1.0 {
        anyhow::bail!(
            "v0.34 dev objective has invalid {label}: value={value}, baseline={baseline}"
        );
    }
    let baseline_error = (1.0 - baseline).max(1.0e-8);
    let value_error = (1.0 - value).max(0.0);
    Ok(value_error / baseline_error)
}

fn finite_ratio(value: Option<f64>, baseline: Option<f64>, label: &str) -> Result<f64> {
    match (value, baseline) {
        (Some(value), Some(baseline)) => finite_ratio_value(value, baseline, label),
        _ => anyhow::bail!("v0.34 dev objective is missing {label} metric"),
    }
}

fn finite_ratio_value(value: f64, baseline: f64, label: &str) -> Result<f64> {
    if !value.is_finite() || !baseline.is_finite() || baseline.abs() <= 1.0e-12 {
        anyhow::bail!(
            "v0.34 dev objective has invalid {label} ratio: value={value}, baseline={baseline}"
        );
    }
    Ok(value / baseline)
}

fn push_ratio(values: &mut Vec<f64>, value: Option<f64>, baseline: Option<f64>) {
    if let (Some(value), Some(baseline)) = (value, baseline) {
        push_ratio_value(values, value, baseline);
    }
}

fn push_ratio_value(values: &mut Vec<f64>, value: f64, baseline: f64) {
    if value.is_finite() && baseline.is_finite() && baseline.abs() > 1.0e-12 {
        values.push(value / baseline);
    }
}

fn copy_checkpoint_dir(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        fs::remove_dir_all(destination)?;
    }
    fs::create_dir_all(destination)?;
    for name in [
        "model.safetensors",
        "optimizer.safetensors",
        "metadata.yaml",
    ] {
        fs::copy(source.join(name), destination.join(name)).with_context(|| {
            format!("failed to copy v0.34 checkpoint file {name} from {source:?}")
        })?;
    }
    Ok(())
}

fn forward_loss(
    model: &PeptideFoundationMultimodalV0340Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    weights: FoundationLossWeights,
    ms2_loss: FoundationMs2LossConfig,
    _contrastive_temperature: f64,
    _rt_scale: f64,
    _ccs_scale: f64,
    normalization: &FoundationTargetNormalizationConfig,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    // v0.34 deliberately trains supervised specialists on clean peptide inputs.
    // `collator` is the zero-corruption clean collator in the optimizer loop.
    let mut views = collator.collate_views(records, device, seed)?;
    normalize_targets(&mut views, normalization)?;
    let fragment_context =
        FoundationFragmentContextBatchV0340::from_records(records, model.forward_config(), device)?;
    let fragment_channel_mask = fragment_context.channel_mask()?;
    let first = model.forward_v0340_t(
        &views.first.input,
        &views.first.context,
        &fragment_context,
        true,
    )?;

    // Only RT and the factorized MS2 objective are trainable in v0.34. The
    // complete v0.31 parent (CCS, self-supervision, alignment/inverse) is frozen.
    let supervised_weights = FoundationLossWeights {
        rt: weights.rt,
        ccs: 0.0,
        ms2: 0.0,
        masked_residue: 0.0,
        chemistry: 0.0,
        contrastive: 0.0,
    };
    let losses = multi_task_loss_with_ms2_config(
        &first.base,
        &views.first.targets,
        supervised_weights,
        ms2_loss,
    )?;
    let mut total = losses.total;
    let (presence_target, presence_mask) = dense_core_presence_targets(
        records,
        model.forward_config().max_sequence_len,
        model.forward_config().ms2_fragment_channels,
        device,
    )?;
    let contextual_ms2_mask = match views.first.targets.ms2_mask.as_ref() {
        Some(mask) => Some(mask.broadcast_mul(&fragment_channel_mask)?),
        None => None,
    };
    let contextual_presence_mask = match presence_mask.as_ref() {
        Some(mask) => Some(mask.broadcast_mul(&fragment_channel_mask)?),
        None => None,
    };
    let factorized = foundation_multimodal_ms2_loss_v0340(
        &first,
        views.first.targets.ms2.as_ref(),
        contextual_ms2_mask.as_ref(),
        presence_target.as_ref(),
        contextual_presence_mask.as_ref(),
    )?;
    total = (total + factorized.total.affine(weights.ms2, 0.0)?)?;
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
fn diffusion_loss(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    _normalization: &FoundationTargetNormalizationConfig,
    length_weight: f64,
    open_ptm_mass_loss_weight: f64,
    alignment_weight: f64,
    alignment_temperature: f64,
    force_all_masked: bool,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, f64)> {
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let diffusion = if force_all_masked {
        diffusion_collator.collate_all_masked(
            &peptides,
            model.inverse_config().diffusion_steps,
            device,
        )?
    } else {
        diffusion_collator.collate_random_timesteps(&peptides, seed, device)?
    };
    let spectrum = collate_spectra(records, spectrum_collator, false, device)?;
    let precursor = precursor_context(records, device)?;
    let output = model
        .diffusion()
        .forward_t(&diffusion, &spectrum, &precursor, true)?;
    let x0 = foundation_diffusion_x0_loss(&output, &diffusion)?;
    let length = foundation_diffusion_length_loss(&output, &diffusion)?;
    let peptide_projection =
        clean_peptide_projection(model, records, clean_collator, true, device)?;
    let spectrum_projection = model.project_spectrum_embedding(&output.spectrum_embedding)?;
    let alignment = foundation_spectrum_peptide_alignment_loss(
        &spectrum_projection,
        &peptide_projection,
        alignment_temperature,
    )?;
    let alignment_value = f64::from(alignment.to_scalar::<f32>()?);
    let open_ptm_mass = foundation_diffusion_open_ptm_mass_loss(&output, &diffusion)?;
    let total = (((x0 + length.affine(length_weight, 0.0)?)?
        + open_ptm_mass.affine(open_ptm_mass_loss_weight, 0.0)?)?
        + alignment.affine(alignment_weight, 0.0)?)?;
    Ok((total, alignment_value))
}

#[allow(clippy::too_many_arguments)]
fn causal_loss(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    _normalization: &FoundationTargetNormalizationConfig,
    open_ptm_mass_loss_weight: f64,
    alignment_weight: f64,
    alignment_temperature: f64,
    conditioning_margin_weight: f64,
    conditioning_margin_nats: f64,
    device: &Device,
) -> Result<CausalTrainObjective> {
    if conditioning_margin_weight > 0.0 && records.len() < 2 {
        anyhow::bail!("causal conditioning margin requires at least two records per batch");
    }
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let causal = causal_collator.collate(&peptides, device)?;
    let spectrum = collate_spectra(records, spectrum_collator, false, device)?;
    let precursor = precursor_context(records, device)?;
    let output = model
        .causal()
        .forward_t(&causal.input, &spectrum, &precursor, true)?;
    let causal_ce = foundation_causal_next_token_loss(&output, &causal)?;
    let matched_ce = f64::from(causal_ce.to_scalar::<f32>()?);

    // v0.13.12 conditioning intervention: only the spectrum is rotated. The
    // teacher-forced peptide prefixes and precursor context remain matched to
    // the target peptide, so the hinge cannot be solved by peptide-language or
    // precursor-mass shortcuts alone.
    let (conditioning_penalty, shuffled_ce, conditioning_gap) = if conditioning_margin_weight > 0.0
    {
        let shuffled_spectrum = collate_spectra(records, spectrum_collator, true, device)?;
        let shuffled_output =
            model
                .causal()
                .forward_t(&causal.input, &shuffled_spectrum, &precursor, true)?;
        let shuffled_loss = foundation_causal_next_token_loss(&shuffled_output, &causal)?;
        let shuffled_value = f64::from(shuffled_loss.to_scalar::<f32>()?);
        let penalty = foundation_causal_conditioning_margin_loss(
            &causal_ce,
            &shuffled_loss,
            conditioning_margin_nats,
        )?;
        (
            penalty,
            Some(shuffled_value),
            Some(shuffled_value - matched_ce),
        )
    } else {
        (Tensor::new(0.0f32, device)?, None, None)
    };
    let conditioning_margin_loss = f64::from(conditioning_penalty.to_scalar::<f32>()?);

    let peptide_projection =
        clean_peptide_projection(model, records, clean_collator, true, device)?;
    let spectrum_projection = model.project_spectrum_embedding(&output.spectrum_embedding)?;
    let alignment = foundation_spectrum_peptide_alignment_loss(
        &spectrum_projection,
        &peptide_projection,
        alignment_temperature,
    )?;
    let alignment_loss = f64::from(alignment.to_scalar::<f32>()?);
    let open_ptm_mass = model.causal().open_ptm_mass_loss(&output, &causal)?;
    let total = (((causal_ce
        + conditioning_penalty.affine(conditioning_margin_weight, 0.0)?)?
        + open_ptm_mass.affine(open_ptm_mass_loss_weight, 0.0)?)?
        + alignment.affine(alignment_weight, 0.0)?)?;
    Ok(CausalTrainObjective {
        total,
        matched_ce,
        shuffled_ce,
        conditioning_gap,
        conditioning_margin_loss,
        alignment_loss,
    })
}

fn dense_core_presence_targets(
    records: &[FoundationTrainingRecord],
    max_sequence_len: usize,
    channels: usize,
    device: &Device,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let batch = records.len();
    if batch == 0 || max_sequence_len < 2 || channels == 0 {
        return Ok((None, None));
    }
    let cleavages = max_sequence_len - 1;
    let mut targets = vec![0.0f32; batch * cleavages * channels];
    let mut masks = vec![0.0f32; batch * cleavages * channels];
    let mut supervised_records = 0usize;

    for (batch_index, record) in records.iter().enumerate() {
        // Only raw observed peak lists are dense enough to support trustworthy
        // absence labels. Transition-list/product-mz fallback rows remain
        // positive-intensity supervision only.
        if record.observed_spectrum_peaks.is_empty() {
            continue;
        }
        let raw_pairs = record
            .observed_spectrum_peaks
            .iter()
            .filter(|peak| {
                peak.mz.is_finite()
                    && peak.mz > 0.0
                    && peak.intensity.is_finite()
                    && peak.intensity > 0.0
            })
            .map(|peak| (peak.mz, peak.intensity))
            .collect::<Vec<_>>();
        if raw_pairs.is_empty() {
            continue;
        }
        let spectrum = FoundationSpectrum::from_pairs(raw_pairs);
        let peptide_cleavages = record
            .peptidoform
            .sequence
            .chars()
            .count()
            .saturating_sub(1);
        if peptide_cleavages == 0 || peptide_cleavages > cleavages {
            continue;
        }
        let dummy_prediction =
            vec![vec![0.0f32; FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240]; peptide_cleavages];
        let relation = foundation_fragment_relation_features(
            &[record.peptidoform.clone()],
            &spectrum,
            &[dummy_prediction],
            &[0.0],
            cleavages,
        )
        .map_err(anyhow::Error::msg)?;
        supervised_records += 1;
        for cleavage in 0..peptide_cleavages {
            if relation.mask[cleavage] <= 0.0 {
                continue;
            }
            let feature_base = cleavage * FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240;
            for channel in 0..FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240.min(channels) {
                let index = (batch_index * cleavages + cleavage) * channels + channel;
                targets[index] = relation.features
                    [feature_base + FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240 + channel];
                masks[index] = 1.0;
            }
        }
    }

    if supervised_records == 0 {
        return Ok((None, None));
    }
    Ok((
        Some(Tensor::from_vec(
            targets,
            (batch, cleavages, channels),
            device,
        )?),
        Some(Tensor::from_vec(
            masks,
            (batch, cleavages, channels),
            device,
        )?),
    ))
}

fn cross_modal_relation_loss(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<Tensor> {
    let (loss, _) = cross_modal_relation_metrics(
        model,
        records,
        clean_collator,
        spectrum_collator,
        true,
        device,
    )?;
    Ok(loss)
}

fn cross_modal_relation_metrics(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    train: bool,
    device: &Device,
) -> Result<(Tensor, f64)> {
    if records.len() < 2 {
        anyhow::bail!("v0.34 relation objective requires at least two records");
    }
    let positive_owned: Vec<FoundationTrainingRecord> =
        records.iter().map(|record| (*record).clone()).collect();
    let negative_owned = mass_near_negative_records(records)?;
    let positive_batch = clean_collator.collate(&positive_owned, device, 0)?;
    let negative_batch = clean_collator.collate(&negative_owned, device, 0)?;
    let positive_foundation = model.property_foundation_t(&positive_batch.input, train)?;
    let negative_foundation = model.property_foundation_t(&negative_batch.input, train)?;
    let spectrum_batch = collate_spectra(records, spectrum_collator, false, device)?;
    let spectrum_encoding = model.encode_spectrum_t(&spectrum_batch, train)?;
    let positive_score = model.relation_score(&positive_foundation, &spectrum_encoding)?;
    let negative_score = model.relation_score(&negative_foundation, &spectrum_encoding)?;
    let loss = foundation_multimodal_relation_margin_loss_v0340(
        &positive_score,
        &negative_score,
        FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0340,
    )?;
    let positive = positive_score.to_vec2::<f32>()?;
    let negative = negative_score.to_vec2::<f32>()?;
    if positive.len() != negative.len() || positive.is_empty() {
        anyhow::bail!("v0.34 relation diagnostics require equal non-empty score batches");
    }
    let preferred = positive
        .iter()
        .zip(negative.iter())
        .filter(|(p, n)| p[0] > n[0])
        .count();
    Ok((loss, preferred as f64 / positive.len() as f64))
}

fn mass_near_negative_records(
    records: &[&FoundationTrainingRecord],
) -> Result<Vec<FoundationTrainingRecord>> {
    if records.len() < 2 {
        anyhow::bail!("v0.34 mass-near negatives require at least two records");
    }
    let masses = records
        .iter()
        .map(|record| {
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut negatives = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (j, candidate) in records.iter().enumerate() {
            if i == j || candidate.peptidoform == record.peptidoform {
                continue;
            }
            let error = (masses[i] - masses[j]).abs();
            match best {
                Some((_, best_error)) if error >= best_error => {}
                _ => best = Some((j, error)),
            }
        }
        let j = best
            .map(|(j, _)| j)
            // Extremely duplicate-heavy batches are allowed to continue. A
            // same-peptidoform fallback contributes a constant margin for that
            // row rather than aborting a long cluster run.
            .unwrap_or_else(|| (i + 1) % records.len());
        negatives.push((*records[j]).clone());
    }
    Ok(negatives)
}

fn normalize_targets(
    views: &mut FoundationTrainingViews,
    normalization: &FoundationTargetNormalizationConfig,
) -> Result<()> {
    normalize_one_targets(&mut views.first.targets, normalization)?;
    normalize_one_targets(&mut views.second.targets, normalization)
}

fn normalize_one_targets(
    targets: &mut redeem_properties::foundation::FoundationTargets,
    normalization: &FoundationTargetNormalizationConfig,
) -> Result<()> {
    if let Some(rt) = targets.rt.take() {
        targets.rt = Some(normalization.rt.normalize_tensor(&rt)?);
    }
    if let Some(ccs) = targets.ccs.take() {
        targets.ccs = Some(normalization.ccs.normalize_tensor(&ccs)?);
    }
    Ok(())
}

fn clean_peptide_projection(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    train: bool,
    device: &Device,
) -> Result<Tensor> {
    let owned: Vec<FoundationTrainingRecord> =
        records.iter().map(|record| (*record).clone()).collect();
    let batch = clean_collator.collate(&owned, device, 0)?;
    model
        .peptide_projection_t(&batch.input, train)
        .map_err(anyhow::Error::from)
}

fn alignment_gradient_probe(
    model: &PeptideFoundationMultimodalV0340Model,
    varmap: &VarMap,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    alignment_temperature: f64,
    device: &Device,
) -> Result<()> {
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let diffusion = diffusion_collator.collate_all_masked(
        &peptides,
        model.inverse_config().diffusion_steps,
        device,
    )?;
    let spectrum = collate_spectra(records, spectrum_collator, false, device)?;
    let precursor = precursor_context(records, device)?;
    let inverse = model
        .diffusion()
        .forward_t(&diffusion, &spectrum, &precursor, false)?;
    let peptide_projection =
        clean_peptide_projection(model, records, clean_collator, false, device)?;
    let spectrum_projection = model.project_spectrum_embedding(&inverse.spectrum_embedding)?;
    let alignment = foundation_spectrum_peptide_alignment_loss(
        &spectrum_projection,
        &peptide_projection,
        alignment_temperature,
    )?;
    let gradients = alignment.backward()?;
    println!(
        "alignment_gradient_probe\tloss={:.6}\tpeptide_encoder={:.8}\tpeptide_contrastive_head={:.8}\tspectrum_encoder={:.8}\tspectrum_projection={:.8}",
        f64::from(alignment.to_scalar::<f32>()?),
        gradient_norm_for_prefix(varmap, &gradients, "encoder.")?,
        gradient_norm_for_prefix(varmap, &gradients, "heads.contrastive.")?,
        gradient_norm_for_prefix(varmap, &gradients, "spectrum_encoder.")?,
        gradient_norm_for_prefix(varmap, &gradients, "alignment.spectrum_projection.")?,
    );
    Ok(())
}

const MS2_HEAD_WEIGHT_NAME: &str = "heads.ms2.weight";
const MS2_HEAD_BIAS_NAME: &str = "heads.ms2.bias";

/// Controlled v0.13.9 intervention: reset exactly one output row of the final MS2
/// linear head to zero while preserving every other row and every other variable.
///
/// With the accepted Softplus(beta=5) activation this yields a neutral initial
/// output ln(2)/5 for that channel and a local derivative of 0.5, restoring a
/// healthy optimization path without changing parameter names or shapes.
fn reset_ms2_output_channel_to_zero(
    varmap: &VarMap,
    channel: usize,
    device: &Device,
) -> Result<Ms2HeadResetReport> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("unified VarMap lock poisoned"))?;
    let weight = data
        .get(MS2_HEAD_WEIGHT_NAME)
        .ok_or_else(|| anyhow::anyhow!("missing {MS2_HEAD_WEIGHT_NAME}"))?;
    let bias = data
        .get(MS2_HEAD_BIAS_NAME)
        .ok_or_else(|| anyhow::anyhow!("missing {MS2_HEAD_BIAS_NAME}"))?;

    let (out_dim, in_dim) = weight.as_tensor().dims2()?;
    if bias.as_tensor().dims1()? != out_dim {
        anyhow::bail!(
            "MS2 head bias shape {:?} is incompatible with weight shape {:?}",
            bias.as_tensor().dims(),
            weight.as_tensor().dims()
        );
    }
    if channel >= out_dim {
        anyhow::bail!(
            "cannot reset MS2 channel {channel}; head has only {out_dim} output channels"
        );
    }

    let mut weights = weight.as_tensor().to_vec2::<f32>()?;
    let mut biases = bias.as_tensor().to_vec1::<f32>()?;
    let fingerprint_before = ms2_head_values_fingerprint(&weights, &biases);

    for value in &mut weights[channel] {
        *value = 0.0;
    }
    biases[channel] = 0.0;

    let fingerprint_after = ms2_head_values_fingerprint(&weights, &biases);
    let flat_weights = weights.into_iter().flatten().collect::<Vec<_>>();
    weight.set(&Tensor::from_vec(flat_weights, (out_dim, in_dim), device)?)?;
    bias.set(&Tensor::from_vec(biases, out_dim, device)?)?;
    drop(data);

    let observed_after = ms2_head_fingerprint(varmap)?;
    if observed_after != fingerprint_after {
        anyhow::bail!(
            "MS2 head reset verification failed: expected {fingerprint_after}, observed {observed_after}"
        );
    }

    Ok(Ms2HeadResetReport {
        mode: Ms2HeadResetMode::B2ZeroV0139,
        channel: Some(channel),
        fingerprint_before: Some(fingerprint_before),
        fingerprint_after: Some(fingerprint_after),
    })
}

fn ms2_head_fingerprint(varmap: &VarMap) -> Result<String> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("unified VarMap lock poisoned"))?;
    let weight = data
        .get(MS2_HEAD_WEIGHT_NAME)
        .ok_or_else(|| anyhow::anyhow!("missing {MS2_HEAD_WEIGHT_NAME}"))?;
    let bias = data
        .get(MS2_HEAD_BIAS_NAME)
        .ok_or_else(|| anyhow::anyhow!("missing {MS2_HEAD_BIAS_NAME}"))?;
    Ok(ms2_head_values_fingerprint(
        &weight.as_tensor().to_vec2::<f32>()?,
        &bias.as_tensor().to_vec1::<f32>()?,
    ))
}

fn ms2_head_values_fingerprint(weights: &[Vec<f32>], biases: &[f32]) -> String {
    let mut hash = FNV1A64_OFFSET;
    fnv1a64_bytes(&mut hash, MS2_HEAD_WEIGHT_NAME.as_bytes());
    fnv1a64_bytes(&mut hash, &(weights.len() as u64).to_le_bytes());
    fnv1a64_bytes(
        &mut hash,
        &(weights.first().map(Vec::len).unwrap_or(0) as u64).to_le_bytes(),
    );
    for row in weights {
        for value in row {
            fnv1a64_bytes(&mut hash, &value.to_bits().to_le_bytes());
        }
    }
    fnv1a64_bytes(&mut hash, MS2_HEAD_BIAS_NAME.as_bytes());
    fnv1a64_bytes(&mut hash, &(biases.len() as u64).to_le_bytes());
    for value in biases {
        fnv1a64_bytes(&mut hash, &value.to_bits().to_le_bytes());
    }
    format!("fnv1a64:{hash:016x}")
}

const ALIGNMENT_WEIGHT_NAME: &str = "alignment.spectrum_projection.weight";
const ALIGNMENT_BIAS_NAME: &str = "alignment.spectrum_projection.bias";
const FNV1A64_OFFSET: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x00000100000001b3;

/// Reinitialize only the fresh spectrum->peptide projection from an explicit seed.
///
/// This mirrors Candle's linear-layer statistical initialization contract: Kaiming
/// normal weights (fan-in/ReLU) and a bias uniform in +/-1/sqrt(fan_in). A tiny
/// local PRNG is used because Candle's CPU random backend cannot be seeded.
fn initialize_alignment_projection_deterministically(
    varmap: &VarMap,
    seed: u64,
    device: &Device,
) -> Result<String> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("unified VarMap lock poisoned"))?;
    let weight = data
        .get(ALIGNMENT_WEIGHT_NAME)
        .ok_or_else(|| anyhow::anyhow!("missing {ALIGNMENT_WEIGHT_NAME}"))?;
    let bias = data
        .get(ALIGNMENT_BIAS_NAME)
        .ok_or_else(|| anyhow::anyhow!("missing {ALIGNMENT_BIAS_NAME}"))?;

    let (out_dim, in_dim) = weight.as_tensor().dims2()?;
    if bias.as_tensor().dims1()? != out_dim {
        anyhow::bail!(
            "alignment projection bias shape {:?} is incompatible with weight shape {:?}",
            bias.as_tensor().dims(),
            weight.as_tensor().dims()
        );
    }

    let mut rng = DeterministicAlignmentRng::new(seed);
    let weight_stdev = (2.0f64 / in_dim as f64).sqrt();
    let mut weight_values = Vec::<f32>::with_capacity(out_dim * in_dim);
    for _ in 0..(out_dim * in_dim) {
        weight_values.push((weight_stdev * rng.standard_normal()) as f32);
    }
    let bias_bound = 1.0f64 / (in_dim as f64).sqrt();
    let mut bias_values = Vec::<f32>::with_capacity(out_dim);
    for _ in 0..out_dim {
        bias_values.push((bias_bound * (2.0 * rng.uniform_open01() - 1.0)) as f32);
    }

    weight.set(&Tensor::from_vec(weight_values, (out_dim, in_dim), device)?)?;
    bias.set(&Tensor::from_vec(bias_values, out_dim, device)?)?;
    drop(data);

    alignment_projection_fingerprint(varmap)
}

#[derive(Debug, Clone, Copy)]
struct DeterministicAlignmentRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl DeterministicAlignmentRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    fn uniform_open01(&mut self) -> f64 {
        const INV_2_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;
        (((self.next_u64() >> 11) as f64) + 0.5) * INV_2_POW_53
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let u1 = self.uniform_open01();
        let u2 = self.uniform_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        let first = radius * angle.cos();
        self.spare_normal = Some(radius * angle.sin());
        first
    }
}

fn alignment_projection_fingerprint(varmap: &VarMap) -> Result<String> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("unified VarMap lock poisoned"))?;
    let mut hash = FNV1A64_OFFSET;
    for name in [ALIGNMENT_WEIGHT_NAME, ALIGNMENT_BIAS_NAME] {
        let variable = data
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name} while fingerprinting"))?;
        fnv1a64_bytes(&mut hash, name.as_bytes());
        for &dim in variable.as_tensor().dims() {
            fnv1a64_bytes(&mut hash, &(dim as u64).to_le_bytes());
        }
        for value in variable.as_tensor().flatten_all()?.to_vec1::<f32>()? {
            fnv1a64_bytes(&mut hash, &value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn fnv1a64_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
}

fn gradient_norm_for_prefix(
    varmap: &VarMap,
    gradients: &candle_core::backprop::GradStore,
    prefix: &str,
) -> Result<f64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("unified VarMap lock poisoned"))?;
    let mut squared = 0.0f64;
    let mut matched = 0usize;
    for (name, variable) in data.iter() {
        if !name.starts_with(prefix) {
            continue;
        }
        matched += 1;
        if let Some(gradient) = gradients.get(variable) {
            squared += f64::from(gradient.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }
    if matched == 0 {
        anyhow::bail!("gradient probe matched no variables for prefix '{prefix}'");
    }
    Ok(squared.sqrt())
}

#[derive(Debug, Default)]
struct Ms2ShapeAccumulator {
    fragment_count: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    predicted_intensity_sum: f64,
    target_intensity_sum: f64,
    exact_zero_count: usize,
    near_zero_1e4_count: usize,
    near_zero_1e3_count: usize,
    near_zero_1e2_count: usize,
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
        if predicted.len() != targets.len() || predicted.len() != masks.len() {
            anyhow::bail!("MS2 validation tensors disagree on batch dimension");
        }
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
                    self.predicted_intensity_sum += pred;
                    self.target_intensity_sum += truth;
                    if pred == 0.0 {
                        self.exact_zero_count += 1;
                    }
                    self.near_zero_1e4_count += usize::from(pred <= 1.0e-4);
                    self.near_zero_1e3_count += usize::from(pred <= 1.0e-3);
                    self.near_zero_1e2_count += usize::from(pred <= 1.0e-2);
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
                .map(|(pred, truth)| pred * truth)
                .sum::<f64>();
            let pred_norm = pred_values
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let target_norm = target_values
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let cosine = if pred_norm > 0.0 && target_norm > 0.0 {
                (dot / (pred_norm * target_norm)).clamp(-1.0, 1.0)
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

    fn exact_zero_fraction(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.exact_zero_count as f64 / self.fragment_count as f64)
    }

    fn near_zero_fraction(&self, threshold: f64) -> Option<f64> {
        if self.fragment_count == 0 {
            return None;
        }
        let count = if (threshold - 1.0e-4).abs() < f64::EPSILON {
            self.near_zero_1e4_count
        } else if (threshold - 1.0e-3).abs() < f64::EPSILON {
            self.near_zero_1e3_count
        } else if (threshold - 1.0e-2).abs() < f64::EPSILON {
            self.near_zero_1e2_count
        } else {
            return None;
        };
        Some(count as f64 / self.fragment_count as f64)
    }

    fn mean_predicted_intensity(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.predicted_intensity_sum / self.fragment_count as f64)
    }

    fn mean_target_intensity(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.target_intensity_sum / self.fragment_count as f64)
    }
}

#[derive(Debug, Default)]
struct OpenPtmAudit {
    records: usize,
    modifications: usize,
    known_unimod: usize,
    open_mass: usize,
    min_mass: Option<f64>,
    max_mass: Option<f64>,
}

fn audit_open_ptm_encoding<'a>(
    records: &[FoundationTrainingRecord],
    index_sets: impl IntoIterator<Item = &'a [usize]>,
    max_tokens: usize,
    mass_scale_da: f64,
) -> Result<OpenPtmAudit> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut audit = OpenPtmAudit::default();
    // TRAIN-core, TRAIN-dev, and TRAIN-holdout are disjoint by construction, so
    // no large deduplication set is needed for this full-corpus preflight.
    for indices in index_sets {
        for &index in indices {
            let record = &records[index];
            vocabulary
                .encode_open_ptm(&record.peptidoform, max_tokens, mass_scale_da)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("open-PTM preflight failed for record index {index}"))?;
            audit.records += 1;
            for modification in &record.peptidoform.modifications {
                let mass = f64::from(modification.mass_delta);
                if !mass.is_finite() {
                    anyhow::bail!("non-finite PTM mass in record index {index}");
                }
                audit.modifications += 1;
                if modification.unimod_id.is_some() {
                    audit.known_unimod += 1;
                } else {
                    audit.open_mass += 1;
                }
                audit.min_mass = Some(audit.min_mass.map_or(mass, |v| v.min(mass)));
                audit.max_mass = Some(audit.max_mass.map_or(mass, |v| v.max(mass)));
            }
        }
    }
    Ok(audit)
}

#[derive(Debug, Default)]
struct FragmentMassGeometryAudit {
    records: usize,
    modifications: usize,
    open_mass: usize,
    min_mass: Option<f64>,
    max_mass: Option<f64>,
}

fn audit_fragment_relation_mass_geometry<'a>(
    records: &[FoundationTrainingRecord],
    index_sets: impl IntoIterator<Item = &'a [usize]>,
) -> Result<FragmentMassGeometryAudit> {
    let mut audit = FragmentMassGeometryAudit::default();
    for indices in index_sets {
        for &index in indices {
            let record = &records[index];
            if record.observed_spectrum_peaks.is_empty() {
                continue;
            }
            foundation_fragment_relation_validate_mass_geometry(&record.peptidoform)
                .map_err(anyhow::Error::msg)
                .with_context(|| {
                    format!("fragment mass-geometry preflight failed for record index {index}")
                })?;
            audit.records += 1;
            for modification in &record.peptidoform.modifications {
                let mass = f64::from(modification.mass_delta);
                audit.modifications += 1;
                if modification.unimod_id.is_none() {
                    audit.open_mass += 1;
                }
                audit.min_mass = Some(audit.min_mass.map_or(mass, |v| v.min(mass)));
                audit.max_mass = Some(audit.max_mass.map_or(mass, |v| v.max(mass)));
            }
        }
    }
    Ok(audit)
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
fn evaluate_all(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[FoundationTrainingRecord],
    forward_indices: &[usize],
    inverse_indices: &[usize],
    batch_size: usize,
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    normalization: &FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    alignment_temperature: f64,
    causal_conditioning_margin_nats: f64,
    device: &Device,
) -> Result<(PropertyMetrics, InverseMetrics)> {
    let properties = evaluate_properties(
        model,
        records,
        forward_indices,
        batch_size,
        clean_collator,
        normalization,
        ms2_loss,
        device,
    )?;
    let inverse = evaluate_inverse(
        model,
        records,
        inverse_indices,
        batch_size,
        clean_collator,
        diffusion_collator,
        causal_collator,
        spectrum_collator,
        alignment_temperature,
        causal_conditioning_margin_nats,
        device,
    )?;
    Ok((properties, inverse))
}

fn evaluate_properties(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    clean_collator: &FoundationCollator,
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
    let mut ms2_objective_sum = 0.0f64;
    let mut ms2_objective_batches = 0usize;
    let mut ms2_shape = Ms2ShapeAccumulator::default();

    for chunk in indices.chunks(batch_size) {
        let owned: Vec<FoundationTrainingRecord> =
            chunk.iter().map(|&i| records[i].clone()).collect();
        let mut batch = clean_collator.collate(&owned, device, 0)?;
        if let Some(rt) = batch.targets.rt.take() {
            batch.targets.rt = Some(normalization.rt.normalize_tensor(&rt)?);
        }
        if let Some(ccs) = batch.targets.ccs.take() {
            batch.targets.ccs = Some(normalization.ccs.normalize_tensor(&ccs)?);
        }
        let fragment_context = FoundationFragmentContextBatchV0340::from_records(
            &owned,
            model.forward_config(),
            device,
        )?;
        let fragment_channel_mask = fragment_context.channel_mask()?;
        let output =
            model.forward_v0340_t(&batch.input, &batch.context, &fragment_context, false)?;
        accumulate_regression(
            &output.base.rt,
            batch.targets.rt.as_ref(),
            batch.targets.rt_mask.as_ref(),
            &normalization.rt,
            &mut rt_abs,
            &mut rt_sq,
            &mut rt_n,
        )?;
        accumulate_regression(
            &output.base.ccs,
            batch.targets.ccs.as_ref(),
            batch.targets.ccs_mask.as_ref(),
            &normalization.ccs,
            &mut ccs_abs,
            &mut ccs_sq,
            &mut ccs_n,
        )?;
        if let (Some(target), Some(mask)) = (&batch.targets.ms2, &batch.targets.ms2_mask) {
            let contextual_mask = mask.broadcast_mul(&fragment_channel_mask)?;
            let components =
                foundation_ms2_loss(&output.base.ms2, target, &contextual_mask, ms2_loss)?;
            ms2_objective_sum += f64::from(components.total.to_scalar::<f32>()?);
            ms2_objective_batches += 1;
            ms2_shape.accumulate(&output.base.ms2, target, &contextual_mask)?;
        }
    }

    Ok(PropertyMetrics {
        rt_mae_native: (rt_n > 0).then(|| rt_abs / rt_n as f64),
        rt_rmse_native: (rt_n > 0).then(|| (rt_sq / rt_n as f64).sqrt()),
        ccs_mae_native: (ccs_n > 0).then(|| ccs_abs / ccs_n as f64),
        ccs_rmse_native: (ccs_n > 0).then(|| (ccs_sq / ccs_n as f64).sqrt()),
        ms2_loss: (ms2_objective_batches > 0)
            .then(|| ms2_objective_sum / ms2_objective_batches as f64),
        ms2_pointwise_mse: ms2_shape.pointwise_mse(),
        ms2_pointwise_mae: ms2_shape.pointwise_mae(),
        ms2_mean_cosine: ms2_shape.mean_cosine(),
        ms2_mean_spectral_angle: ms2_shape.mean_spectral_angle(),
        ms2_mean_pearson: ms2_shape.mean_pearson(),
        ms2_exact_zero_fraction: ms2_shape.exact_zero_fraction(),
        ms2_near_zero_1e4_fraction: ms2_shape.near_zero_fraction(1.0e-4),
        ms2_near_zero_1e3_fraction: ms2_shape.near_zero_fraction(1.0e-3),
        ms2_near_zero_1e2_fraction: ms2_shape.near_zero_fraction(1.0e-2),
        ms2_mean_predicted_intensity: ms2_shape.mean_predicted_intensity(),
        ms2_mean_target_intensity: ms2_shape.mean_target_intensity(),
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_inverse(
    model: &PeptideFoundationMultimodalV0340Model,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    alignment_temperature: f64,
    causal_conditioning_margin_nats: f64,
    device: &Device,
) -> Result<InverseMetrics> {
    let mut metrics = InverseMetrics::default();
    let mut batches = 0usize;
    let mut causal_sequence_records = 0usize;
    for chunk in indices.chunks(batch_size) {
        let selected: Vec<&FoundationTrainingRecord> = chunk.iter().map(|&i| &records[i]).collect();
        if selected.len() < 2 {
            anyhow::bail!("causal conditioning validation requires at least two records per batch");
        }
        let peptides: Vec<PeptidoformInput> = selected
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect();
        let diffusion = diffusion_collator.collate_all_masked(
            &peptides,
            model.inverse_config().diffusion_steps,
            device,
        )?;
        let causal = causal_collator.collate(&peptides, device)?;
        let precursor = precursor_context(&selected, device)?;
        let spectrum = collate_spectra(&selected, spectrum_collator, false, device)?;
        let shuffled_spectrum = collate_spectra(&selected, spectrum_collator, true, device)?;

        let diffusion_output = model
            .diffusion()
            .forward_t(&diffusion, &spectrum, &precursor, false)?;
        let diffusion_loss = foundation_diffusion_x0_loss(&diffusion_output, &diffusion)?;
        let length_loss = foundation_diffusion_length_loss(&diffusion_output, &diffusion)?;
        metrics.diffusion_loss += f64::from(diffusion_loss.to_scalar::<f32>()?);
        metrics.diffusion_length_loss += f64::from(length_loss.to_scalar::<f32>()?);
        metrics.diffusion_token_accuracy += token_accuracy(
            &diffusion_output.token_logits,
            &diffusion.active_indices,
            &diffusion.target_classes,
        )?;

        let causal_output =
            model
                .causal()
                .forward_t(&causal.input, &spectrum, &precursor, false)?;
        let causal_loss = foundation_causal_next_token_loss(&causal_output, &causal)?;
        let causal_loss_value = f64::from(causal_loss.to_scalar::<f32>()?);
        metrics.causal_loss += causal_loss_value;
        metrics.causal_perplexity += causal_loss_value.exp();
        metrics.causal_token_accuracy += token_accuracy(
            &causal_output.token_logits,
            &causal.active_indices,
            &causal.target_classes,
        )?;

        let shuffled_causal_output =
            model
                .causal()
                .forward_t(&causal.input, &shuffled_spectrum, &precursor, false)?;
        let shuffled_causal_loss =
            foundation_causal_next_token_loss(&shuffled_causal_output, &causal)?;
        let shuffled_causal_loss_value = f64::from(shuffled_causal_loss.to_scalar::<f32>()?);
        metrics.causal_shuffled_loss += shuffled_causal_loss_value;
        metrics.causal_conditioning_gap += shuffled_causal_loss_value - causal_loss_value;
        metrics.causal_conditioning_margin_loss += f64::from(
            foundation_causal_conditioning_margin_loss(
                &causal_loss,
                &shuffled_causal_loss,
                causal_conditioning_margin_nats,
            )?
            .to_scalar::<f32>()?,
        );

        let matched_sequence_nlls = causal_sequence_nlls(&causal_output, &causal)?;
        let shuffled_sequence_nlls = causal_sequence_nlls(&shuffled_causal_output, &causal)?;
        if matched_sequence_nlls.len() != shuffled_sequence_nlls.len() {
            anyhow::bail!("matched/shuffled causal sequence diagnostics disagree on batch size");
        }
        for (matched, shuffled) in matched_sequence_nlls
            .iter()
            .zip(shuffled_sequence_nlls.iter())
        {
            let gap = shuffled - matched;
            metrics.causal_conditioning_preference_fraction += if gap > 0.0 { 1.0 } else { 0.0 };
            metrics.causal_conditioning_margin_satisfied_fraction +=
                if gap >= causal_conditioning_margin_nats {
                    1.0
                } else {
                    0.0
                };
        }
        causal_sequence_records += matched_sequence_nlls.len();

        let peptide_projection =
            clean_peptide_projection(model, &selected, clean_collator, false, device)?;
        let matched_projection =
            model.project_spectrum_embedding(&diffusion_output.spectrum_embedding)?;
        let alignment_loss = foundation_spectrum_peptide_alignment_loss(
            &matched_projection,
            &peptide_projection,
            alignment_temperature,
        )?;
        metrics.alignment_loss += f64::from(alignment_loss.to_scalar::<f32>()?);
        metrics.alignment_retrieval_top1 +=
            retrieval_top1(&matched_projection, &peptide_projection)?;

        let shuffled_output =
            model
                .diffusion()
                .forward_t(&diffusion, &shuffled_spectrum, &precursor, false)?;
        let shuffled_projection =
            model.project_spectrum_embedding(&shuffled_output.spectrum_embedding)?;
        let shuffled_loss = foundation_spectrum_peptide_alignment_loss(
            &shuffled_projection,
            &peptide_projection,
            alignment_temperature,
        )?;
        metrics.shuffled_alignment_loss += f64::from(shuffled_loss.to_scalar::<f32>()?);
        metrics.shuffled_alignment_retrieval_top1 +=
            retrieval_top1(&shuffled_projection, &peptide_projection)?;

        let (relation_loss, relation_preference) = cross_modal_relation_metrics(
            model,
            &selected,
            clean_collator,
            spectrum_collator,
            false,
            device,
        )?;
        metrics.relation_margin_loss += f64::from(relation_loss.to_scalar::<f32>()?);
        metrics.relation_preference_fraction += relation_preference;
        batches += 1;
    }
    if batches == 0 {
        anyhow::bail!("unified inverse validation produced zero batches");
    }
    let n = batches as f64;
    metrics.diffusion_loss /= n;
    metrics.diffusion_length_loss /= n;
    metrics.diffusion_token_accuracy /= n;
    metrics.causal_loss /= n;
    metrics.causal_perplexity /= n;
    metrics.causal_token_accuracy /= n;
    metrics.causal_shuffled_loss /= n;
    metrics.causal_conditioning_gap /= n;
    metrics.causal_conditioning_margin_loss /= n;
    if causal_sequence_records == 0 {
        anyhow::bail!("causal conditioning validation produced zero sequence records");
    }
    let sequence_n = causal_sequence_records as f64;
    metrics.causal_conditioning_preference_fraction /= sequence_n;
    metrics.causal_conditioning_margin_satisfied_fraction /= sequence_n;
    metrics.alignment_loss /= n;
    metrics.alignment_retrieval_top1 /= n;
    metrics.shuffled_alignment_loss /= n;
    metrics.shuffled_alignment_retrieval_top1 /= n;
    metrics.relation_margin_loss /= n;
    metrics.relation_preference_fraction /= n;
    Ok(metrics)
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

fn causal_sequence_nlls(
    output: &FoundationCausalOutput,
    batch: &FoundationCausalBatch,
) -> Result<Vec<f64>> {
    let logits = output.token_logits.to_vec3::<f32>()?;
    let targets = batch.target_tokens.to_vec2::<u32>()?;
    let masks = batch.input.token_mask.to_vec2::<f32>()?;
    if logits.len() != targets.len() || logits.len() != masks.len() {
        anyhow::bail!("causal sequence diagnostics disagree on batch dimension");
    }
    let mut losses = Vec::with_capacity(logits.len());
    for batch_index in 0..logits.len() {
        if logits[batch_index].len() != targets[batch_index].len()
            || logits[batch_index].len() != masks[batch_index].len()
        {
            anyhow::bail!("causal sequence diagnostics disagree on token width");
        }
        let mut nll_sum = 0.0f64;
        let mut active = 0usize;
        for position in 0..logits[batch_index].len() {
            if masks[batch_index][position] <= 0.0 {
                continue;
            }
            let row = &logits[batch_index][position];
            let target = targets[batch_index][position] as usize;
            if target >= row.len() {
                anyhow::bail!(
                    "causal sequence target class {target} exceeds logits width {}",
                    row.len()
                );
            }
            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let exp_sum = row
                .iter()
                .map(|&value| (f64::from(value) - max_logit).exp())
                .sum::<f64>();
            let logsumexp = max_logit + exp_sum.ln();
            nll_sum += logsumexp - f64::from(row[target]);
            active += 1;
        }
        if active == 0 {
            anyhow::bail!("causal sequence diagnostics encountered an empty target row");
        }
        losses.push(nll_sum / active as f64);
    }
    Ok(losses)
}

fn token_accuracy(logits: &Tensor, active_indices: &Tensor, classes: &Tensor) -> Result<f64> {
    let (b, l, vocab) = logits.dims3()?;
    if vocab != FOUNDATION_DIFFUSION_VOCAB_SIZE && vocab != FOUNDATION_OPEN_PTM_VOCAB_SIZE {
        anyhow::bail!(
            "unexpected unified token vocabulary {vocab}; expected legacy {} or open-PTM {}",
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_OPEN_PTM_VOCAB_SIZE
        );
    }
    let flat = logits.reshape((b * l, vocab))?;
    let selected = flat.index_select(active_indices, 0)?.to_vec2::<f32>()?;
    let targets = classes.to_vec1::<u32>()?;
    let correct = selected
        .iter()
        .zip(targets.iter())
        .filter(|(row, target)| argmax(row) as u32 == **target)
        .count();
    Ok(correct as f64 / targets.len().max(1) as f64)
}

fn retrieval_top1(spectrum: &Tensor, peptide: &Tensor) -> Result<f64> {
    let spectrum = l2_rows(spectrum)?;
    let peptide = l2_rows(peptide)?;
    let s = spectrum.to_vec2::<f32>()?;
    let p = peptide.to_vec2::<f32>()?;
    if s.len() != p.len() || s.is_empty() {
        anyhow::bail!("alignment retrieval requires equal non-empty batches");
    }
    let mut correct = 0usize;
    for (i, row) in s.iter().enumerate() {
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (j, candidate) in p.iter().enumerate() {
            let score: f64 = row
                .iter()
                .zip(candidate.iter())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            if score > best_score {
                best_score = score;
                best = j;
            }
        }
        correct += usize::from(best == i);
    }
    Ok(correct as f64 / s.len() as f64)
}

fn l2_rows(values: &Tensor) -> Result<Tensor> {
    let denominator = values
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?
        .unsqueeze(1)?;
    Ok(values.broadcast_div(&denominator)?)
}

fn collate_spectra(
    records: &[&FoundationTrainingRecord],
    spectrum_collator: &FoundationSpectrumCollator,
    shuffled: bool,
    device: &Device,
) -> Result<FoundationSpectrumBatch> {
    let mut spectra: Vec<FoundationSpectrum> = records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record)
                .ok_or_else(|| anyhow::anyhow!("selected unified inverse record lacks spectrum"))
        })
        .collect::<Result<_>>()?;
    if shuffled && spectra.len() > 1 {
        spectra.rotate_left(1);
    }
    Ok(spectrum_collator.collate(&spectra, device)?)
}

fn precursor_context(
    records: &[&FoundationTrainingRecord],
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let charge: Vec<f32> = records
        .iter()
        .map(|r| r.context.charge.unwrap_or(0) as f32)
        .collect();
    let charge_present: Vec<f32> = records
        .iter()
        .map(|r| if r.context.charge.is_some() { 1.0 } else { 0.0 })
        .collect();
    let precursor_mz: Vec<f32> = records
        .iter()
        .map(|r| r.context.precursor_mz.unwrap_or(0.0))
        .collect();
    let precursor_mz_present: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.context.precursor_mz.is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let nce: Vec<f32> = records
        .iter()
        .map(|r| r.context.nce.unwrap_or(0.0))
        .collect();
    let nce_present: Vec<f32> = records
        .iter()
        .map(|r| if r.context.nce.is_some() { 1.0 } else { 0.0 })
        .collect();
    let b = records.len();
    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(charge, b, device)?,
        charge_present: Tensor::from_vec(charge_present, b, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, b, device)?,
        precursor_mz_present: Tensor::from_vec(precursor_mz_present, b, device)?,
        nce: Tensor::from_vec(nce, b, device)?,
        nce_present: Tensor::from_vec(nce_present, b, device)?,
        instrument_ids: Tensor::zeros(b, DType::U32, device)?,
        instrument_present: Tensor::zeros(b, DType::F32, device)?,
    })
}

fn usable_inverse_indices(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    config: &FoundationDiffusionConfig,
    vocabulary: FoundationDiffusionVocabulary,
) -> Vec<usize> {
    benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
        .filter_map(|entry| {
            let record = &records[entry.record_index];
            (FoundationSpectrum::from_training_record(record).is_some()
                && vocabulary
                    .encode_open_ptm(
                        &record.peptidoform,
                        config.max_tokens,
                        FOUNDATION_OPEN_PTM_MASS_SCALE_DA,
                    )
                    .is_ok())
            .then_some(entry.record_index)
        })
        .collect()
}

fn read_inverse_metadata(checkpoint: &Path) -> Result<InverseCheckpointMetadata> {
    let metadata_path = if checkpoint.is_dir() {
        checkpoint.join("metadata.yaml")
    } else {
        checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    };
    Ok(serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read inverse metadata {metadata_path:?}"))?,
    )?)
}

fn load_v0310_unchanged_variables(
    varmap: &VarMap,
    checkpoint: &Path,
    device: &Device,
) -> Result<V034WarmStartReport> {
    let tensors = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("failed to load frozen v0.31 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.34 VarMap lock poisoned during warm start"))?;
    let mut loaded_variables = 0usize;
    let mut fresh_variables = 0usize;
    let mut missing_reused = Vec::new();

    for (name, variable) in data.iter() {
        if name.starts_with("rt_specialist_v0340.") || name.starts_with("ms2_specialist_v0340.") {
            fresh_variables += 1;
            continue;
        }
        match tensors.get(name) {
            Some(tensor) => {
                if tensor.dims() != variable.as_tensor().dims() {
                    anyhow::bail!(
                        "v0.34 warm-start shape mismatch for '{name}': parent {:?}, model {:?}",
                        tensor.dims(),
                        variable.as_tensor().dims()
                    );
                }
                variable.set(tensor)?;
                loaded_variables += 1;
            }
            None => missing_reused.push(name.clone()),
        }
    }
    if !missing_reused.is_empty() {
        anyhow::bail!(
            "frozen v0.31 checkpoint is missing required v0.34 reused variables: {}",
            missing_reused.join(", ")
        );
    }
    let ignored_parent_variables = tensors
        .keys()
        .filter(|name| !data.contains_key(*name))
        .count();
    drop(data);

    if loaded_variables == 0 || fresh_variables == 0 {
        anyhow::bail!(
            "v0.34 selective warm start is nonfunctional: loaded={loaded_variables} fresh={fresh_variables}"
        );
    }
    Ok(V034WarmStartReport {
        loaded_variables,
        fresh_variables,
        ignored_parent_variables,
    })
}

fn validate_v0340_training_checkpoint_metadata(
    label: &str,
    metadata: &UnifiedPilotMetadata,
    expected_corpus_fingerprint: &str,
    expected_benchmark_fingerprint: &str,
    expected_config: &PeptideFoundationMultimodalV0340Config,
    batch_size: usize,
    seed: u64,
    learning_rate: f64,
    max_total_steps: usize,
) -> Result<()> {
    if metadata.version != 340 {
        anyhow::bail!(
            "v0.34 resume {label} checkpoint has metadata version {}, expected 340",
            metadata.version
        );
    }
    if metadata.objective != "v0340_deep_rt_ms2_specialists_from_frozen_v0310" {
        anyhow::bail!(
            "v0.34 resume {label} checkpoint objective mismatch: {}",
            metadata.objective
        );
    }
    if metadata.corpus_fingerprint != expected_corpus_fingerprint {
        anyhow::bail!(
            "v0.34 resume {label} corpus fingerprint mismatch: checkpoint={} current={expected_corpus_fingerprint}",
            metadata.corpus_fingerprint
        );
    }
    if metadata.benchmark_manifest_fingerprint != expected_benchmark_fingerprint {
        anyhow::bail!(
            "v0.34 resume {label} benchmark fingerprint mismatch: checkpoint={} current={expected_benchmark_fingerprint}",
            metadata.benchmark_manifest_fingerprint
        );
    }
    if metadata.v0340_config != *expected_config {
        anyhow::bail!("v0.34 resume {label} architecture config mismatch");
    }
    if metadata.batch_size != batch_size {
        anyhow::bail!(
            "v0.34 resume {label} batch_size mismatch: checkpoint={} requested={batch_size}",
            metadata.batch_size
        );
    }
    if metadata.seed != seed {
        anyhow::bail!(
            "v0.34 resume {label} seed mismatch: checkpoint={} requested={seed}",
            metadata.seed
        );
    }
    let lr_scale = learning_rate
        .abs()
        .max(metadata.learning_rate.abs())
        .max(1.0);
    if (metadata.learning_rate - learning_rate).abs() > f64::EPSILON * 32.0 * lr_scale {
        anyhow::bail!(
            "v0.34 resume {label} learning_rate mismatch: checkpoint={} requested={learning_rate}",
            metadata.learning_rate
        );
    }
    if metadata.train_steps != max_total_steps {
        anyhow::bail!(
            "v0.34 resume {label} train-step budget mismatch: checkpoint={} requested={max_total_steps}",
            metadata.train_steps
        );
    }
    if metadata.completed_steps > metadata.train_steps {
        anyhow::bail!(
            "v0.34 resume {label} checkpoint completed_steps={} exceeds train_steps={}",
            metadata.completed_steps,
            metadata.train_steps
        );
    }
    Ok(())
}

fn read_unified_pilot_metadata(checkpoint: &Path) -> Result<UnifiedPilotMetadata> {
    let metadata_path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read v0.34 metadata {metadata_path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.34 metadata {metadata_path:?}"))
}

fn read_unified_parent_metadata(checkpoint: &Path) -> Result<UnifiedParentMetadata> {
    let metadata_path = if checkpoint.is_dir() {
        checkpoint.join("metadata.yaml")
    } else {
        checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    };
    serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read unified parent metadata {metadata_path:?}"))?,
    )
    .with_context(|| format!("failed to parse unified parent metadata {metadata_path:?}"))
}

fn rt_harmonization_calibration_id(
    run: &redeem_properties::foundation::FoundationTrainingRunConfig,
) -> Result<Option<String>> {
    let ids: BTreeSet<String> = run
        .corpus
        .sources
        .iter()
        .filter_map(|source| source.rt_harmonization.as_ref())
        .map(|transform| transform.calibration_id.clone())
        .collect();
    if ids.len() > 1 {
        anyhow::bail!("unified run contains multiple RT harmonization calibration ids: {ids:?}");
    }
    Ok(ids.into_iter().next())
}

fn require_harmonized_rt_coverage(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partitions: &[FoundationPartition],
) -> Result<()> {
    let partitions: BTreeSet<FoundationPartition> = partitions.iter().copied().collect();
    let mut source_native = 0usize;
    let mut harmonized = 0usize;
    let mut missing = Vec::new();
    for entry in &benchmark.entries {
        if !partitions.contains(&entry.partition) {
            continue;
        }
        let record = &records[entry.record_index];
        if record.retention_time.normalized.is_some_and(f32::is_finite) {
            source_native += 1;
            if record.retention_time.harmonized.is_some_and(f32::is_finite) {
                harmonized += 1;
            } else if missing.len() < 8 {
                missing.push((entry.record_index, entry.peptidoform.clone()));
            }
        }
    }
    if source_native != harmonized {
        anyhow::bail!(
            "harmonized RT coverage is incomplete for TRAIN/VALIDATION: source-native labels={}, harmonized labels={}, examples={:?}",
            source_native,
            harmonized,
            missing
        );
    }
    Ok(())
}

fn resolve_forward_state(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.join("state.yaml")
    } else {
        checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("state.yaml")
    }
}

fn resolve_model_safetensors(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.join("model.safetensors")
    } else {
        checkpoint.to_path_buf()
    }
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &UnifiedPilotMetadata,
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

fn print_evaluation(label: &str, step: usize, metrics: (PropertyMetrics, InverseMetrics)) {
    let (p, i) = metrics;
    println!(
        "{label}_forward\tstep={step}\trt_mae_native={}\trt_rmse_native={}\tccs_mae_native={}\tccs_rmse_native={}\tms2_loss={}\tms2_pointwise_mse={}\tms2_pointwise_mae={}\tms2_cosine={}\tms2_spectral_angle={}\tms2_pearson={}\tms2_exact_zero_fraction={}\tms2_near_zero_1e4_fraction={}\tms2_near_zero_1e3_fraction={}\tms2_near_zero_1e2_fraction={}\tms2_mean_predicted_intensity={}\tms2_mean_target_intensity={}",
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
        fmt_opt(p.ms2_exact_zero_fraction),
        fmt_opt(p.ms2_near_zero_1e4_fraction),
        fmt_opt(p.ms2_near_zero_1e3_fraction),
        fmt_opt(p.ms2_near_zero_1e2_fraction),
        fmt_opt(p.ms2_mean_predicted_intensity),
        fmt_opt(p.ms2_mean_target_intensity),
    );
    println!(
        "{label}_inverse\tstep={step}\tdiffusion_loss={:.6}\tdiffusion_length_loss={:.6}\tdiffusion_token_accuracy={:.6}\tcausal_loss={:.6}\tcausal_perplexity={:.4}\tcausal_token_accuracy={:.6}\tcausal_shuffled_loss={:.6}\tcausal_conditioning_gap={:.6}\tcausal_conditioning_margin_loss={:.6}\tcausal_conditioning_preference_fraction={:.6}\tcausal_conditioning_margin_satisfied_fraction={:.6}\talignment_loss={:.6}\talignment_retrieval_top1={:.6}\tshuffled_alignment_loss={:.6}\tshuffled_alignment_retrieval_top1={:.6}",
        i.diffusion_loss,
        i.diffusion_length_loss,
        i.diffusion_token_accuracy,
        i.causal_loss,
        i.causal_perplexity,
        i.causal_token_accuracy,
        i.causal_shuffled_loss,
        i.causal_conditioning_gap,
        i.causal_conditioning_margin_loss,
        i.causal_conditioning_preference_fraction,
        i.causal_conditioning_margin_satisfied_fraction,
        i.alignment_loss,
        i.alignment_retrieval_top1,
        i.shuffled_alignment_loss,
        i.shuffled_alignment_retrieval_top1,
    );
    println!(
        "{label}_causal_conditioning_ablation\tstep={step}\tshuffled_minus_matched_nll={:.6}\tpreference_fraction={:.6}\tmargin_satisfied_fraction={:.6}\tmargin_hinge={:.6}",
        i.causal_conditioning_gap,
        i.causal_conditioning_preference_fraction,
        i.causal_conditioning_margin_satisfied_fraction,
        i.causal_conditioning_margin_loss,
    );
    println!(
        "{label}_alignment_ablation\tstep={step}\tloss_delta={:.6}\tretrieval_top1_delta={:.6}",
        i.shuffled_alignment_loss - i.alignment_loss,
        i.alignment_retrieval_top1 - i.shuffled_alignment_retrieval_top1,
    );
    println!(
        "{label}_relation\tstep={step}\tmargin_loss={:.6}\tpreference_fraction={:.6}",
        i.relation_margin_loss, i.relation_preference_fraction,
    );
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "NA".into())
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
    if batch_size == 0 || requested_batches == 0 {
        anyhow::bail!("v0.34 {label} validation requires positive batch_size/requested_batches");
    }
    let max_batches = requested_batches.min(validation_indices.len() / batch_size);
    if max_batches == 0 {
        anyhow::bail!(
            "v0.34 {label} has only {} records, fewer than batch_size={batch_size}",
            validation_indices.len()
        );
    }
    if config.validation_source_weights.is_empty() {
        return Ok(max_batches);
    }

    let mut available = BTreeMap::<String, usize>::new();
    for &index in validation_indices {
        let source = provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("v0.34 {label} provenance index {index} is out of bounds")
        })?;
        *available.entry(source.source_id.clone()).or_default() += 1;
    }
    for source in available.keys() {
        if !config.validation_source_weights.contains_key(source) {
            anyhow::bail!(
                "v0.34 {label} validation weights are missing represented source '{source}'"
            );
        }
    }
    for source in config.validation_source_weights.keys() {
        if !available.contains_key(source) {
            anyhow::bail!("v0.34 {label} validation weight refers to absent source '{source}'");
        }
    }

    let total_weight: f64 = config.validation_source_weights.values().copied().sum();
    if !(total_weight > 0.0 && total_weight.is_finite()) {
        anyhow::bail!("v0.34 {label} validation source weights have no positive finite mass");
    }

    for batches in (1..=max_batches).rev() {
        let target_records = batches.saturating_mul(batch_size);
        let quotas = weighted_quotas_v0340(
            target_records,
            &config.validation_source_weights,
            total_weight,
        );
        let fits = quotas
            .iter()
            .all(|(source, desired)| *desired <= available.get(source).copied().unwrap_or(0));
        if fits {
            return Ok(batches);
        }
    }

    anyhow::bail!(
        "v0.34 {label} cannot satisfy validation source weights for even one full batch without replacement"
    )
}

fn weighted_quotas_v0340(
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
        assigned = assigned.saturating_add(base);
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
            "unified {label} sample plan has {} records but {expected} are required",
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

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
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
