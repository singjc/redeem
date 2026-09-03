//! Bounded joint continuation of the peptide-property and spectrum-to-peptide foundation model.
//!
//! This pilot intentionally keeps the validated forward and inverse architectures
//! unchanged. One VarMap owns all parameters, diffusion/causal modes share their
//! historical inverse namespaces, and a trainable spectrum projection aligns the
//! inverse spectrum representation with the forward peptide contrastive space.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    contrastive_info_nce_loss, foundation_causal_conditioning_margin_loss,
    foundation_causal_next_token_loss, foundation_diffusion_length_loss,
    foundation_diffusion_x0_loss, foundation_ms2_loss, foundation_spectrum_peptide_alignment_loss,
    load_foundation_corpus, load_unified_foundation_components, multi_task_loss_with_ms2_config,
    read_foundation_training_run_config, sample_foundation_training_indices,
    sample_foundation_validation_indices, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCausalBatch, FoundationCausalCollator,
    FoundationCausalOutput, FoundationCheckpointMetadata, FoundationCollator,
    FoundationCollatorConfig, FoundationCorruptionConfig, FoundationDiffusionCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationLossWeights,
    FoundationMs2LossConfig, FoundationMs2OutputActivation, FoundationPartition,
    FoundationRegressionNormalization, FoundationRegressionNormalizationStrategy,
    FoundationSamplePlan, FoundationSamplingConfig, FoundationSpectrum, FoundationSpectrumBatch,
    FoundationSpectrumCollator, FoundationTargetNormalizationConfig, FoundationTrainingRecord,
    FoundationTrainingViews, PeptideFoundationUnifiedModel, PeptidoformInput,
    PrecursorContextBatch, RetentionTimeObjective, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_MS2_SOFTPLUS_BETA_V0138,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct InverseCheckpointMetadata {
    diffusion: FoundationDiffusionConfig,
}

#[derive(Debug, Deserialize)]
struct UnifiedParentMetadata {
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
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
    alignment_weight: f64,
    alignment_temperature: f64,
    alignment_initialization: String,
    alignment_initialization_seed: u64,
    alignment_initialization_fingerprint: String,
    forward_objective_weight: f64,
    diffusion_objective_weight: f64,
    causal_objective_weight: f64,
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
    if args.len() < 6 || args.len() > 23 {
        anyhow::bail!(
            "usage: foundation_train_unified FOUNDATION_TRAINING.yaml OUTPUT_DIR FORWARD_CHECKPOINT DIFFUSION_CHECKPOINT CAUSAL_CHECKPOINT [train_steps=30] [batch_size=8] [validation_batches=8] [seed=20260908] [learning_rate=2e-5] [alignment_weight=0.05] [alignment_temperature=0.07] [forward_objective_weight=0.5] [diffusion_objective_weight=0.25] [causal_objective_weight=0.25] [parent_unified_checkpoint] [ms2_pointwise_weight=1.0] [ms2_cosine_weight=0.0] [ms2_output_activation=relu] [ms2_head_reset=none] [causal_conditioning_margin_weight=0.0] [causal_conditioning_margin_nats=0.25]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let forward_checkpoint = PathBuf::from(&args[3]);
    let diffusion_checkpoint = PathBuf::from(&args[4]);
    let causal_checkpoint = PathBuf::from(&args[5]);
    let train_steps = parse_or(&args, 6, 30usize)?;
    let batch_size = parse_or(&args, 7, 8usize)?;
    let validation_batches = parse_or(&args, 8, 8usize)?;
    let seed = parse_or(&args, 9, 20_260_908u64)?;
    let learning_rate = parse_or(&args, 10, 2.0e-5f64)?;
    let alignment_weight = parse_or(&args, 11, 0.05f64)?;
    let alignment_temperature = parse_or(&args, 12, 0.07f64)?;
    let forward_objective_weight = parse_or(&args, 13, 0.5f64)?;
    let diffusion_objective_weight = parse_or(&args, 14, 0.25f64)?;
    let causal_objective_weight = parse_or(&args, 15, 0.25f64)?;
    let parent_unified_checkpoint = args.get(16).map(PathBuf::from);
    let ms2_pointwise_weight = parse_or(&args, 17, 1.0f64)?;
    let ms2_cosine_weight = parse_or(&args, 18, 0.0f64)?;
    let ms2_loss = FoundationMs2LossConfig {
        pointwise_weight: ms2_pointwise_weight,
        cosine_weight: ms2_cosine_weight,
        ..FoundationMs2LossConfig::default()
    };
    ms2_loss.validate().map_err(anyhow::Error::msg)?;
    let ms2_output_activation = args
        .get(19)
        .map(|value| value.parse::<FoundationMs2OutputActivation>())
        .transpose()
        .map_err(anyhow::Error::msg)?
        .unwrap_or_default();
    let ms2_head_reset = args
        .get(20)
        .map(|value| value.parse::<Ms2HeadResetMode>())
        .transpose()
        .map_err(anyhow::Error::msg)?
        .unwrap_or(Ms2HeadResetMode::None);
    let causal_conditioning_margin_weight = parse_or(&args, 21, 0.0f64)?;
    let causal_conditioning_margin_nats = parse_or(&args, 22, 0.25f64)?;
    if ms2_head_reset == Ms2HeadResetMode::B2ZeroV0139 {
        if parent_unified_checkpoint.is_none() {
            anyhow::bail!("b2-zero-v0139 requires a parent unified checkpoint");
        }
        if ms2_output_activation != FoundationMs2OutputActivation::SoftplusV0138 {
            anyhow::bail!("b2-zero-v0139 requires softplus-v0138 MS2 output activation");
        }
    }
    let max_gradient_norm = 1.0f64;
    let diffusion_length_weight = 0.1f64;

    if train_steps == 0 || batch_size < 2 || validation_batches == 0 {
        anyhow::bail!("train_steps/validation_batches must be positive and batch_size must be >=2");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("learning_rate must be finite and positive");
    }
    if !(alignment_weight >= 0.0 && alignment_weight.is_finite()) {
        anyhow::bail!("alignment_weight must be finite and non-negative");
    }
    if !(alignment_temperature > 0.0 && alignment_temperature.is_finite()) {
        anyhow::bail!("alignment_temperature must be finite and positive");
    }
    if !(causal_conditioning_margin_weight >= 0.0 && causal_conditioning_margin_weight.is_finite())
    {
        anyhow::bail!("causal_conditioning_margin_weight must be finite and non-negative");
    }
    if !(causal_conditioning_margin_nats >= 0.0 && causal_conditioning_margin_nats.is_finite()) {
        anyhow::bail!("causal_conditioning_margin_nats must be finite and non-negative");
    }
    for (name, value) in [
        ("forward_objective_weight", forward_objective_weight),
        ("diffusion_objective_weight", diffusion_objective_weight),
        ("causal_objective_weight", causal_objective_weight),
    ] {
        if !(value >= 0.0 && value.is_finite()) {
            anyhow::bail!("{name} must be finite and non-negative");
        }
    }
    let objective_weight_sum =
        forward_objective_weight + diffusion_objective_weight + causal_objective_weight;
    if (objective_weight_sum - 1.0).abs() > 1.0e-9 {
        anyhow::bail!(
            "forward/diffusion/causal objective weights must sum to 1.0; got {objective_weight_sum}"
        );
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let forward_state = resolve_forward_state(&forward_checkpoint);
    let forward_metadata = FoundationCheckpointMetadata::read_yaml(&forward_state)
        .with_context(|| format!("failed to read corrected forward state {forward_state:?}"))?;
    let forward_model_path = resolve_model_safetensors(&forward_checkpoint);
    let mut forward_config = forward_metadata.model_config.clone();
    forward_config.ms2_output_activation = ms2_output_activation;
    forward_config.validate().map_err(anyhow::Error::msg)?;

    let diffusion_metadata = read_inverse_metadata(&diffusion_checkpoint)?;
    let causal_metadata = read_inverse_metadata(&causal_checkpoint)?;
    if diffusion_metadata.diffusion != causal_metadata.diffusion {
        anyhow::bail!(
            "diffusion and causal checkpoints do not describe the same inverse architecture"
        );
    }
    let inverse_config = diffusion_metadata.diffusion;
    inverse_config.validate().map_err(anyhow::Error::msg)?;
    let diffusion_model_path = resolve_model_safetensors(&diffusion_checkpoint);
    let causal_model_path = resolve_model_safetensors(&causal_checkpoint);

    let train_forward_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .collect();
    let validation_forward_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Validation)
        .map(|entry| entry.record_index)
        .collect();
    let vocabulary = FoundationDiffusionVocabulary;
    let train_inverse_indices = usable_inverse_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &inverse_config,
        vocabulary,
    );
    let validation_inverse_indices = usable_inverse_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &inverse_config,
        vocabulary,
    );
    if train_forward_indices.len() < batch_size
        || validation_forward_indices.len() < batch_size
        || train_inverse_indices.len() < batch_size
        || validation_inverse_indices.len() < batch_size
    {
        anyhow::bail!(
            "insufficient records for unified pilot: forward train/val={}/{}, inverse train/val={}/{}",
            train_forward_indices.len(),
            validation_forward_indices.len(),
            train_inverse_indices.len(),
            validation_inverse_indices.len()
        );
    }

    let mut forward_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_forward_indices,
        &validation_forward_indices,
    );
    forward_sampling.train_steps_per_epoch = Some(train_steps);
    forward_sampling.validation_steps = Some(validation_batches);
    let mut inverse_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_inverse_indices,
        &validation_inverse_indices,
    );
    inverse_sampling.train_steps_per_epoch = Some(train_steps);
    inverse_sampling.validation_steps = Some(validation_batches);

    let forward_a_plan = sample_foundation_training_indices(
        &corpus.records,
        &corpus.provenance,
        &train_forward_indices,
        batch_size,
        0,
        seed ^ 0x18e7_64ad_a82f_1121,
        true,
        &forward_sampling,
    )?;
    let forward_b_plan = sample_foundation_training_indices(
        &corpus.records,
        &corpus.provenance,
        &train_forward_indices,
        batch_size,
        1,
        seed ^ 0x62c9_e03a_733d_a1b5,
        true,
        &forward_sampling,
    )?;
    let diffusion_plan = sample_foundation_training_indices(
        &corpus.records,
        &corpus.provenance,
        &train_inverse_indices,
        batch_size,
        2,
        seed ^ 0x9b43_7f21_c4a6_0d8b,
        true,
        &inverse_sampling,
    )?;
    let causal_plan = sample_foundation_training_indices(
        &corpus.records,
        &corpus.provenance,
        &train_inverse_indices,
        batch_size,
        3,
        seed ^ 0xd532_6a91_089b_f417,
        true,
        &inverse_sampling,
    )?;
    for (label, plan) in [
        ("forward_a_train", &forward_a_plan),
        ("forward_b_train", &forward_b_plan),
        ("diffusion_train", &diffusion_plan),
        ("causal_train", &causal_plan),
    ] {
        require_plan_records(label, plan, train_steps.saturating_mul(batch_size))?;
    }

    let validation_forward_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &validation_forward_indices,
        batch_size,
        seed ^ 0x3d13_7f24_559c_81e7,
        &forward_sampling,
    )?;
    let validation_inverse_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &validation_inverse_indices,
        batch_size,
        seed ^ 0x72a4_c11d_0b95_e683,
        &inverse_sampling,
    )?;
    require_plan_records(
        "forward_validation",
        &validation_forward_plan,
        validation_batches.saturating_mul(batch_size),
    )?;
    require_plan_records(
        "inverse_validation",
        &validation_inverse_plan,
        validation_batches.saturating_mul(batch_size),
    )?;
    let validation_forward = validation_forward_plan.indices.clone();
    let validation_inverse = validation_inverse_plan.indices.clone();

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model =
        PeptideFoundationUnifiedModel::new(forward_config.clone(), inverse_config.clone(), vb)?;

    let (
        warm_start_description,
        alignment_initialization,
        alignment_initialization_seed,
        alignment_initialization_fingerprint,
    ) = if let Some(parent) = &parent_unified_checkpoint {
        let parent_metadata = read_unified_parent_metadata(parent)?;
        if !parent_metadata
            .forward_config
            .parameter_compatible_with(&forward_config)
        {
            anyhow::bail!("parent unified checkpoint forward parameterization does not match the requested candidate");
        }
        if parent_metadata.inverse_config != inverse_config {
            anyhow::bail!("parent unified checkpoint inverse architecture does not match the supplied inverse checkpoints");
        }
        let parent_model = resolve_model_safetensors(parent);
        varmap
            .load(&parent_model)
            .with_context(|| format!("failed to load parent unified model {parent_model:?}"))?;
        let fingerprint = alignment_projection_fingerprint(&varmap)?;
        (
            format!(
                "unified_parent={}\tparent_completed_steps={}",
                parent_model.display(),
                parent_metadata.completed_steps
            ),
            "loaded_from_unified_parent".to_string(),
            0u64,
            fingerprint,
        )
    } else {
        let warm_start = load_unified_foundation_components(
            &varmap,
            &forward_model_path,
            &diffusion_model_path,
            &causal_model_path,
            &device,
        )?;
        let alignment_seed = mix64(seed ^ 0xa17e_11a9_5eed_0132);
        let fingerprint =
            initialize_alignment_projection_deterministically(&varmap, alignment_seed, &device)?;
        (
                format!(
                    "forward_loaded={}\tdiffusion_loaded={}\tcausal_overlay_loaded={}\tfresh_alignment={}",
                    warm_start.forward_loaded_variables,
                    warm_start.diffusion_loaded_variables,
                    warm_start.causal_overlay_loaded_variables,
                    warm_start.fresh_alignment_variables
                ),
                "seeded_kaiming_normal_weight_uniform_bias_v0132".to_string(),
                alignment_seed,
                fingerprint,
            )
    };

    let ms2_head_reset_report = match ms2_head_reset {
        Ms2HeadResetMode::None => Ms2HeadResetReport::none(),
        Ms2HeadResetMode::B2ZeroV0139 => reset_ms2_output_channel_to_zero(&varmap, 1, &device)?,
    };

    let forward_trainer = &forward_metadata.trainer_config;
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
    let mut forward_collator_config = forward_trainer.collator.clone();
    forward_collator_config.retention_time_objective = rt_objective;
    if rt_objective == RetentionTimeObjective::Harmonized {
        target_normalization.rt = run.trainer.target_normalization.rt;
        target_normalization.rt.mean = None;
        target_normalization.rt.standard_deviation = None;
        target_normalization.rt.label_count = 0;
        if target_normalization.rt.strategy
            != FoundationRegressionNormalizationStrategy::TrainStandardize
        {
            anyhow::bail!("harmonized RT training requires trainer.target_normalization.rt.strategy=TrainStandardize");
        }
        let train_harmonized_rt = train_forward_indices
            .iter()
            .filter_map(|&index| corpus.records[index].retention_time.harmonized);
        target_normalization
            .rt
            .resolve_from_values(train_harmonized_rt)?;
        if !target_normalization.rt.is_active() {
            anyhow::bail!("harmonized RT target normalization did not resolve from TRAIN labels");
        }
    }

    let forward_collator =
        FoundationCollator::new(forward_config.clone(), forward_collator_config)?;
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
    let diffusion_collator = FoundationDiffusionCollator::new(inverse_config.clone())?;
    let causal_collator = FoundationCausalCollator::new(inverse_config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(inverse_config.spectrum.clone())?;

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
    println!("objective\tunified_forward_diffusion_causal_alignment_spectrum_margin_v3");
    println!(
        "schedule\tone_optimizer_update=0.5*forward_weight*forward_a+0.5*forward_weight*forward_b+diffusion_weight*diffusion+causal_weight*(matched_causal_ce+conditioning_margin_weight*hinge+alignment_weight*alignment)"
    );
    println!("component_batches_per_optimizer_update\t4");
    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!(
        "benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        benchmark.manifest_fingerprint()
    );
    println!("forward_checkpoint\t{}", forward_model_path.display());
    println!("diffusion_checkpoint\t{}", diffusion_model_path.display());
    println!("causal_checkpoint\t{}", causal_model_path.display());
    println!("forward_model_dim\t{}", forward_config.model_dim);
    println!("inverse_model_dim\t{}", inverse_config.model_dim);
    println!("alignment_dim\t{}", forward_config.contrastive_dim);
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("validation_batches\t{validation_batches}");
    println!("learning_rate\t{learning_rate}");
    println!("alignment_weight\t{alignment_weight}");
    println!("alignment_temperature\t{alignment_temperature}");
    println!(
        "alignment_initialization\talgorithm={}\tseed={}\tfingerprint={}",
        alignment_initialization,
        alignment_initialization_seed,
        alignment_initialization_fingerprint,
    );
    println!("forward_objective_weight\t{forward_objective_weight}");
    println!("diffusion_objective_weight\t{diffusion_objective_weight}");
    println!("causal_objective_weight\t{causal_objective_weight}");
    println!("causal_conditioning_margin_weight\t{causal_conditioning_margin_weight}");
    println!("causal_conditioning_margin_nats\t{causal_conditioning_margin_nats}");
    println!(
        "causal_conditioning_negative\tdeterministic_rotate_left_1_spectrum_only_precursor_and_prefix_fixed"
    );
    println!("ms2_pointwise_weight\t{}", ms2_loss.pointwise_weight);
    println!("ms2_cosine_weight\t{}", ms2_loss.cosine_weight);
    println!("ms2_output_activation\t{}", ms2_output_activation.as_str());
    println!(
        "ms2_softplus_beta\t{}",
        if ms2_output_activation == FoundationMs2OutputActivation::SoftplusV0138 {
            FOUNDATION_MS2_SOFTPLUS_BETA_V0138.to_string()
        } else {
            "NA".to_string()
        }
    );
    println!(
        "ms2_head_reset\talgorithm={}\tchannel={}\tfingerprint_before={}\tfingerprint_after={}",
        ms2_head_reset_report.mode.as_str(),
        ms2_head_reset_report
            .channel
            .map(|value| value.to_string())
            .unwrap_or_else(|| "NA".into()),
        ms2_head_reset_report
            .fingerprint_before
            .as_deref()
            .unwrap_or("NA"),
        ms2_head_reset_report
            .fingerprint_after
            .as_deref()
            .unwrap_or("NA"),
    );
    println!(
        "forward_component_weight\t{}",
        0.5 * forward_objective_weight
    );
    println!("max_gradient_norm\t{max_gradient_norm}");
    println!("warm_start\t{warm_start_description}");
    println!("rt_objective\t{:?}", rt_objective);
    println!(
        "rt_harmonization_calibration\t{}",
        rt_harmonization_calibration.as_deref().unwrap_or("none")
    );
    println!(
        "rt_target_normalization\tstrategy={:?}\tmean={}\tstandard_deviation={}\tlabel_count={}",
        target_normalization.rt.strategy,
        target_normalization
            .rt
            .mean
            .map(|value| format!("{value:.8}"))
            .unwrap_or_else(|| "NA".into()),
        target_normalization
            .rt
            .standard_deviation
            .map(|value| format!("{value:.8}"))
            .unwrap_or_else(|| "NA".into()),
        target_normalization.rt.label_count,
    );
    println!("optimizer_variables\t{}", optimizer.variable_count());
    print_sample_plan("forward_a_train", &forward_a_plan);
    print_sample_plan("forward_b_train", &forward_b_plan);
    print_sample_plan("diffusion_train", &diffusion_plan);
    print_sample_plan("causal_train", &causal_plan);
    print_sample_plan("forward_validation", &validation_forward_plan);
    print_sample_plan("inverse_validation", &validation_inverse_plan);

    let metadata = |completed_steps| {
        UnifiedPilotMetadata {
        version: 8,
        objective: "unified_forward_diffusion_causal_alignment_spectrum_margin_v3".into(),
        schedule: "one_optimizer_update=0.5*forward_weight*forward_a+0.5*forward_weight*forward_b+diffusion_weight*diffusion+causal_weight*(matched_causal_ce+conditioning_margin_weight*hinge+alignment_weight*alignment)".into(),
        corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
        benchmark_manifest_fingerprint: format!(
            "fnv1a64:{:016x}",
            benchmark.manifest_fingerprint()
        ),
        forward_checkpoint: forward_model_path.display().to_string(),
        diffusion_checkpoint: diffusion_model_path.display().to_string(),
        causal_checkpoint: causal_model_path.display().to_string(),
        parent_unified_checkpoint: parent_unified_checkpoint
            .as_ref()
            .map(|path| resolve_model_safetensors(path).display().to_string()),
        rt_objective,
        rt_harmonization_calibration: rt_harmonization_calibration.clone(),
        target_normalization,
        train_steps,
        batch_size,
        validation_batches,
        seed,
        learning_rate,
        max_gradient_norm,
        diffusion_length_weight,
        alignment_weight,
        alignment_temperature,
        alignment_initialization: alignment_initialization.clone(),
        alignment_initialization_seed,
        alignment_initialization_fingerprint: alignment_initialization_fingerprint.clone(),
        forward_objective_weight,
        diffusion_objective_weight,
        causal_objective_weight,
        causal_conditioning_margin_weight,
        causal_conditioning_margin_nats,
        causal_conditioning_negative:
            "deterministic_rotate_left_1_spectrum_only_precursor_and_prefix_fixed".into(),
        ms2_loss,
        ms2_output_activation,
        ms2_head_reset: ms2_head_reset_report.mode.as_str().into(),
        ms2_head_reset_channel: ms2_head_reset_report.channel,
        ms2_head_fingerprint_before_reset: ms2_head_reset_report.fingerprint_before.clone(),
        ms2_head_fingerprint_after_reset: ms2_head_reset_report.fingerprint_after.clone(),
        completed_steps,
        forward_config: forward_config.clone(),
        inverse_config: inverse_config.clone(),
    }
    };

    save_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &optimizer,
        &metadata(0),
    )?;
    print_evaluation(
        "initial",
        0,
        evaluate_all(
            &model,
            &corpus.records,
            &validation_forward,
            &validation_inverse,
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
        )?,
    );

    let probe_indices = causal_plan.indices[..batch_size].to_vec();
    let probe_records: Vec<&FoundationTrainingRecord> = probe_indices
        .iter()
        .map(|&index| &corpus.records[index])
        .collect();
    alignment_gradient_probe(
        &model,
        &varmap,
        &probe_records,
        &clean_collator,
        &diffusion_collator,
        &spectrum_collator,
        alignment_temperature,
        &device,
    )?;

    let eval_every = train_steps.min(10).max(1);
    for step in 1..=train_steps {
        let offset = (step - 1).saturating_mul(batch_size);
        let forward_a_indices = &forward_a_plan.indices[offset..offset + batch_size];
        let forward_b_indices = &forward_b_plan.indices[offset..offset + batch_size];
        let forward_a_records: Vec<FoundationTrainingRecord> = forward_a_indices
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect();
        let forward_b_records: Vec<FoundationTrainingRecord> = forward_b_indices
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect();
        let forward_a = forward_loss(
            &model,
            &forward_collator,
            &forward_a_records,
            forward_trainer.loss_weights,
            ms2_loss,
            forward_trainer.contrastive_temperature,
            forward_trainer.shared_gradient_scales.rt_encoder,
            forward_trainer.shared_gradient_scales.ccs_encoder,
            &target_normalization,
            seed ^ (step as u64).wrapping_mul(0xa24b_1c62_4073_f5d9),
            &device,
        )?;
        let forward_b = forward_loss(
            &model,
            &forward_collator,
            &forward_b_records,
            forward_trainer.loss_weights,
            ms2_loss,
            forward_trainer.contrastive_temperature,
            forward_trainer.shared_gradient_scales.rt_encoder,
            forward_trainer.shared_gradient_scales.ccs_encoder,
            &target_normalization,
            seed ^ (step as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
            &device,
        )?;

        let diffusion_indices = &diffusion_plan.indices[offset..offset + batch_size];
        let diffusion_records: Vec<&FoundationTrainingRecord> = diffusion_indices
            .iter()
            .map(|&index| &corpus.records[index])
            .collect();
        let force_all_masked = mix64(seed ^ step as u64) & 1 == 0;
        let (diffusion, diffusion_alignment) = diffusion_loss(
            &model,
            &diffusion_records,
            &clean_collator,
            &diffusion_collator,
            &spectrum_collator,
            &target_normalization,
            diffusion_length_weight,
            alignment_weight,
            alignment_temperature,
            force_all_masked,
            seed ^ step as u64,
            &device,
        )?;

        let causal_indices = &causal_plan.indices[offset..offset + batch_size];
        let causal_records: Vec<&FoundationTrainingRecord> = causal_indices
            .iter()
            .map(|&index| &corpus.records[index])
            .collect();
        let causal = causal_loss(
            &model,
            &causal_records,
            &clean_collator,
            &causal_collator,
            &spectrum_collator,
            &target_normalization,
            alignment_weight,
            alignment_temperature,
            causal_conditioning_margin_weight,
            causal_conditioning_margin_nats,
            &device,
        )?;

        let forward_a_value = f64::from(forward_a.to_scalar::<f32>()?);
        let forward_b_value = f64::from(forward_b.to_scalar::<f32>()?);
        let diffusion_value = f64::from(diffusion.to_scalar::<f32>()?);
        let causal_value = f64::from(causal.total.to_scalar::<f32>()?);
        let forward_a_weighted = forward_a.affine(0.5 * forward_objective_weight, 0.0)?;
        let forward_b_weighted = forward_b.affine(0.5 * forward_objective_weight, 0.0)?;
        let diffusion_weighted = diffusion.affine(diffusion_objective_weight, 0.0)?;
        let causal_weighted = causal.total.affine(causal_objective_weight, 0.0)?;
        let total =
            (((forward_a_weighted + forward_b_weighted)? + diffusion_weighted)? + causal_weighted)?;
        let total_value = f64::from(total.to_scalar::<f32>()?);
        let update = optimizer.backward_step(&total, Some(max_gradient_norm))?;
        println!(
            "train\tstep={step}\tforward_a={forward_a_value:.6}\tforward_b={forward_b_value:.6}\tdiffusion={diffusion_value:.6}\tcausal={causal_value:.6}\tcausal_matched_ce={:.6}\tcausal_shuffled_ce={}\tcausal_conditioning_gap={}\tcausal_conditioning_margin_loss={:.6}\tdiffusion_alignment={diffusion_alignment:.6}\tcausal_alignment={:.6}\tforward_weight={forward_objective_weight:.6}\tdiffusion_weight={diffusion_objective_weight:.6}\tcausal_weight={causal_objective_weight:.6}\tconditioning_margin_weight={causal_conditioning_margin_weight:.6}\tconditioning_margin_nats={causal_conditioning_margin_nats:.6}\ttotal={total_value:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
            causal.matched_ce,
            fmt_opt(causal.shuffled_ce),
            fmt_opt(causal.conditioning_gap),
            causal.conditioning_margin_loss,
            causal.alignment_loss,
            update.gradient_norm,
            update.gradient_scale,
        );

        if step % eval_every == 0 || step == train_steps {
            let metrics = evaluate_all(
                &model,
                &corpus.records,
                &validation_forward,
                &validation_inverse,
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
            print_evaluation("validation", step, metrics);
            save_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &optimizer,
                &metadata(step),
            )?;
        }
    }

    save_checkpoint(
        &output_root.join("final"),
        &varmap,
        &optimizer,
        &metadata(train_steps),
    )?;
    println!("final_checkpoint\t{}", output_root.join("final").display());
    Ok(())
}

fn forward_loss(
    model: &PeptideFoundationUnifiedModel,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    weights: FoundationLossWeights,
    ms2_loss: FoundationMs2LossConfig,
    contrastive_temperature: f64,
    rt_scale: f64,
    ccs_scale: f64,
    normalization: &FoundationTargetNormalizationConfig,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let mut views = collator.collate_views(records, device, seed)?;
    normalize_targets(&mut views, normalization)?;
    let first = model.forward().forward_t_with_shared_gradient_scales(
        &views.first.input,
        &views.first.context,
        true,
        rt_scale,
        ccs_scale,
    )?;
    let second = model
        .forward()
        .forward_t(&views.second.input, &views.second.context, true)?;
    let losses = multi_task_loss_with_ms2_config(&first, &views.first.targets, weights, ms2_loss)?;
    let contrastive = contrastive_info_nce_loss(
        &first.contrastive_projection,
        &second.contrastive_projection,
        contrastive_temperature,
    )?;
    (losses.total + contrastive.affine(weights.contrastive, 0.0)?).map_err(anyhow::Error::from)
}

#[allow(clippy::too_many_arguments)]
fn diffusion_loss(
    model: &PeptideFoundationUnifiedModel,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    _normalization: &FoundationTargetNormalizationConfig,
    length_weight: f64,
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
    let total =
        ((x0 + length.affine(length_weight, 0.0)?)? + alignment.affine(alignment_weight, 0.0)?)?;
    Ok((total, alignment_value))
}

#[allow(clippy::too_many_arguments)]
fn causal_loss(
    model: &PeptideFoundationUnifiedModel,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    _normalization: &FoundationTargetNormalizationConfig,
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
    let total = ((causal_ce + conditioning_penalty.affine(conditioning_margin_weight, 0.0)?)?
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
    model: &PeptideFoundationUnifiedModel,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    train: bool,
    device: &Device,
) -> Result<Tensor> {
    let owned: Vec<FoundationTrainingRecord> =
        records.iter().map(|record| (*record).clone()).collect();
    let batch = clean_collator.collate(&owned, device, 0)?;
    Ok(model
        .forward()
        .forward_t(&batch.input, &batch.context, train)?
        .contrastive_projection)
}

fn alignment_gradient_probe(
    model: &PeptideFoundationUnifiedModel,
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
    model: &PeptideFoundationUnifiedModel,
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
    model: &PeptideFoundationUnifiedModel,
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
        let output = model
            .forward()
            .forward_t(&batch.input, &batch.context, false)?;
        accumulate_regression(
            &output.rt,
            batch.targets.rt.as_ref(),
            batch.targets.rt_mask.as_ref(),
            &normalization.rt,
            &mut rt_abs,
            &mut rt_sq,
            &mut rt_n,
        )?;
        accumulate_regression(
            &output.ccs,
            batch.targets.ccs.as_ref(),
            batch.targets.ccs_mask.as_ref(),
            &normalization.ccs,
            &mut ccs_abs,
            &mut ccs_sq,
            &mut ccs_n,
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
    model: &PeptideFoundationUnifiedModel,
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
    if vocab != FOUNDATION_DIFFUSION_VOCAB_SIZE {
        anyhow::bail!("unexpected unified token vocabulary {vocab}");
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
                    .encode(&record.peptidoform, config.max_tokens)
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
