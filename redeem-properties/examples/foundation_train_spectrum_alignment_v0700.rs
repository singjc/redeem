//! v0.70 spectrum<->peptide aligned representation.
//!
//! Scientific contract:
//! - warm-start the peptide representation from the already selected v0.52 checkpoint;
//! - keep every v0.52 parameter frozen in a separate VarMap;
//! - train only a new observed-spectrum encoder plus spectrum/peptide alignment projections;
//! - optimize paired symmetric InfoNCE on TRAIN only;
//! - select the alignment checkpoint on DEV retrieval only;
//! - never touch TRAIN-HOLDOUT, historical VALIDATION/APD, or historical TEST.
//!
//! This is deliberately a representation/retrieval experiment, not a sequence decoder.
//! If the spectrum embedding cannot retrieve its paired peptide, a downstream causal decoder
//! should not be expected to repair the missing inverse representation.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_peptidoform_neutral_mass, foundation_precursor_neutral_mass,
    foundation_spectrum_peptide_alignment_loss, load_foundation_corpus,
    read_foundation_training_run_config, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCollator, FoundationCollatorConfig,
    FoundationCorruptionConfig, FoundationDiffusionConfig, FoundationLearningRateSchedule,
    FoundationPartition, FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator,
    FoundationSpectrumEncoder, FoundationTrainingRecord, PeptideFoundationV0520Config,
    PeptideFoundationV0520Model, RetentionTimeObjective, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V070_VERSION: u32 = 700;
const V070_OBJECTIVE: &str = "v0700_frozen_v0520_spectrum_peptide_alignment";
const V070_ARCHITECTURE: &str =
    "frozen_v0520_peptide_plus_observed_spectrum_transformer_contrastive_v0700";
const V070_NAMESPACE: &str = "student_v070";
const V070_TEMPERATURE: f64 = 0.07;
const V070_ALIGNMENT_DIM: usize = 192;
const V070_SPECTRUM_MODEL_DIM: usize = 320;
const V070_SPECTRUM_LAYERS: usize = 6;
const V070_SPECTRUM_HEADS: usize = 8;
const V070_SPECTRUM_FF_DIM: usize = 1280;
const V070_PRECURSOR_FEATURES: usize = 6;
const V070_DEV_IDENTITIES: usize = 2048;
const V070_SMOKE_DEV_IDENTITIES: usize = 128;
const V070_MASS_CANDIDATES: usize = 64;
const V070_MAX_GRADIENT_NORM: f64 = 1.0;
const V070_MIN_SELECTION_GAIN: f64 = 0.025;
const V070_MIN_GLOBAL_IL_TOP10: f64 = 0.10;
const V070_MIN_MASS_IL_TOP10: f64 = 0.30;
const V070_MIN_GLOBAL_EXACT_TOP1: f64 = 0.01;

#[derive(Debug, Clone, Deserialize)]
struct V052ParentMetadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    v0520_config: PeptideFoundationV0520Config,
    rt_objective: RetentionTimeObjective,
    completed_epochs: usize,
    completed_updates: usize,
    dev_objective: f64,
    smoke_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct V070Config {
    spectrum: FoundationDiffusionConfig,
    peptide_input_dim: usize,
    spectrum_hidden_dim: usize,
    alignment_dim: usize,
    precursor_features: usize,
    temperature: f64,
}

impl V070Config {
    fn production(parent: &PeptideFoundationV0520Config) -> Result<Self> {
        let residue_dim = parent.base_v0510.base_v0500.residue_dim;
        let mut spectrum = FoundationDiffusionConfig::default();
        spectrum.model_dim = V070_SPECTRUM_MODEL_DIM;
        spectrum.num_attention_heads = V070_SPECTRUM_HEADS;
        spectrum.feed_forward_dim = V070_SPECTRUM_FF_DIM;
        spectrum.spectrum_layers = V070_SPECTRUM_LAYERS;
        // Decoder is not constructed in v0.70, but FoundationDiffusionConfig validates it.
        spectrum.decoder_layers = 1;
        spectrum.dropout = 0.05;
        spectrum.validate().map_err(anyhow::Error::msg)?;
        let config = Self {
            spectrum,
            peptide_input_dim: 2 * residue_dim,
            spectrum_hidden_dim: V070_SPECTRUM_MODEL_DIM,
            alignment_dim: V070_ALIGNMENT_DIM,
            precursor_features: V070_PRECURSOR_FEATURES,
            temperature: V070_TEMPERATURE,
        };
        config.validate()?;
        Ok(config)
    }

    #[cfg(test)]
    fn local_smoke() -> Self {
        let mut spectrum = FoundationDiffusionConfig::default();
        spectrum.model_dim = 32;
        spectrum.num_attention_heads = 4;
        spectrum.feed_forward_dim = 64;
        spectrum.spectrum_layers = 1;
        spectrum.decoder_layers = 1;
        spectrum.spectrum.max_peaks = 8;
        spectrum.spectrum.peak_feature_dim = 32;
        Self {
            spectrum,
            peptide_input_dim: 24,
            spectrum_hidden_dim: 32,
            alignment_dim: 16,
            precursor_features: V070_PRECURSOR_FEATURES,
            temperature: V070_TEMPERATURE,
        }
    }

    fn validate(&self) -> Result<()> {
        self.spectrum.validate().map_err(anyhow::Error::msg)?;
        if self.peptide_input_dim == 0
            || self.spectrum_hidden_dim != self.spectrum.model_dim
            || self.alignment_dim == 0
            || self.precursor_features != V070_PRECURSOR_FEATURES
        {
            anyhow::bail!("v0.70 alignment dimensions are inconsistent");
        }
        if !(self.temperature > 0.0 && self.temperature.is_finite()) {
            anyhow::bail!("v0.70 contrastive temperature must be finite and positive");
        }
        Ok(())
    }
}

#[derive(Clone)]
struct PeptideSpectrumAlignmentV0700 {
    config: V070Config,
    spectrum_encoder: FoundationSpectrumEncoder,
    precursor_projection: Linear,
    spectrum_hidden: Linear,
    spectrum_projection: Linear,
    peptide_hidden: Linear,
    peptide_projection: Linear,
}

impl PeptideSpectrumAlignmentV0700 {
    fn new(config: V070Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let ns = vb.pp(V070_NAMESPACE);
        Ok(Self {
            spectrum_encoder: FoundationSpectrumEncoder::new(
                &config.spectrum,
                ns.pp("spectrum_encoder"),
            )?,
            precursor_projection: nn::linear(
                config.precursor_features,
                config.spectrum_hidden_dim,
                ns.pp("precursor_projection"),
            )?,
            spectrum_hidden: nn::linear(
                config.spectrum_hidden_dim,
                config.spectrum_hidden_dim,
                ns.pp("spectrum_hidden"),
            )?,
            spectrum_projection: nn::linear(
                config.spectrum_hidden_dim,
                config.alignment_dim,
                ns.pp("spectrum_projection"),
            )?,
            peptide_hidden: nn::linear(
                config.peptide_input_dim,
                config.spectrum_hidden_dim,
                ns.pp("peptide_hidden"),
            )?,
            peptide_projection: nn::linear(
                config.spectrum_hidden_dim,
                config.alignment_dim,
                ns.pp("peptide_projection"),
            )?,
            config,
        })
    }

    fn encode_spectrum_t(
        &self,
        spectrum: &FoundationSpectrumBatch,
        precursor_features: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let encoded = self.spectrum_encoder.forward_t(spectrum, train)?;
        let precursor = self.precursor_projection.forward(precursor_features)?;
        let fused = (encoded.spectrum_embedding + precursor)?;
        let hidden = self.spectrum_hidden.forward(&fused.contiguous()?)?.relu()?;
        Ok(self.spectrum_projection.forward(&hidden.contiguous()?)?)
    }

    fn encode_peptide(&self, frozen_peptide_features: &Tensor) -> Result<Tensor> {
        let hidden = self
            .peptide_hidden
            .forward(&frozen_peptide_features.contiguous()?)?
            .relu()?;
        Ok(self.peptide_projection.forward(&hidden.contiguous()?)?)
    }

    fn alignment_loss(&self, spectrum: &Tensor, peptide: &Tensor) -> Result<Tensor> {
        Ok(foundation_spectrum_peptide_alignment_loss(
            spectrum,
            peptide,
            self.config.temperature,
        )?)
    }
}

#[derive(Debug, Clone)]
struct AlignmentGroup {
    key: String,
    peptidoform: String,
    sequence: String,
    charge: i32,
    record_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct DevIdentity {
    record_index: usize,
    exact_key: String,
    il_key: String,
    charge: i32,
    length: usize,
    modified: bool,
    observed_neutral_mass: f64,
    candidate_neutral_mass: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct RetrievalMetrics {
    identities: usize,
    exact_top1: f64,
    exact_top5: f64,
    exact_top10: f64,
    exact_mrr: f64,
    il_top1: f64,
    il_top5: f64,
    il_top10: f64,
    il_mrr: f64,
    mean_exact_rank: f64,
    median_exact_rank: f64,
    mass_true_coverage: f64,
    mass_exact_top1: f64,
    mass_exact_top10: f64,
    mass_exact_mrr: f64,
    mass_il_top1: f64,
    mass_il_top10: f64,
    mass_il_mrr: f64,
}

impl RetrievalMetrics {
    fn selection_score(self) -> f64 {
        0.5 * self.il_mrr + 0.5 * self.mass_il_mrr
    }
}

#[derive(Debug, Clone)]
struct QueryOutcome {
    charge: i32,
    length: usize,
    modified: bool,
    exact_top1: bool,
    il_top1: bool,
    exact_top10: bool,
    il_top10: bool,
    exact_rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V070Metadata {
    version: u32,
    objective: String,
    architecture: String,
    parent_v052_checkpoint: String,
    parent_v052_completed_epochs: usize,
    parent_v052_completed_updates: usize,
    parent_v052_dev_objective: f64,
    parent_update_policy: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    config: V070Config,
    max_epochs: usize,
    batch_size: usize,
    steps_per_epoch: usize,
    patience: usize,
    min_delta: f64,
    seed: u64,
    learning_rate: f64,
    dev_identity_count: usize,
    dev_identity_fingerprint: String,
    completed_epochs: usize,
    completed_updates: usize,
    dev_selection_score: f64,
    step0_selection_score: f64,
    smoke_mode: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 12 {
        anyhow::bail!(
            "usage: foundation_train_spectrum_alignment_v0700 RUN_V0260.yaml OUTPUT_DIR PARENT_V052_BEST [max_epochs=6] [batch_size=64] [steps_per_epoch=1000] [patience=2] [min_delta=0.002] [seed=20261070] [learning_rate=2e-4] [mode=smoke|train|resume]"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let requested_max_epochs = parse_or(&args, 4, 6usize)?;
    let batch_size = parse_or(&args, 5, 64usize)?;
    let requested_steps_per_epoch = parse_or(&args, 6, 1000usize)?;
    let patience = parse_or(&args, 7, 2usize)?;
    let min_delta = parse_or(&args, 8, 0.002f64)?;
    let seed = parse_or(&args, 9, 20_261_070u64)?;
    let learning_rate = parse_or(&args, 10, 2.0e-4f64)?;
    let mode = args.get(11).map(String::as_str).unwrap_or("train");
    if !matches!(mode, "smoke" | "train" | "resume") {
        anyhow::bail!("v0.70 mode must be smoke, train, or resume");
    }
    let smoke_mode = mode == "smoke";
    let resume_mode = mode == "resume";
    if batch_size < 2
        || patience == 0
        || requested_max_epochs == 0
        || requested_steps_per_epoch == 0
    {
        anyhow::bail!("v0.70 epochs/steps/patience must be positive and batch_size >= 2");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite())
        || !(min_delta >= 0.0 && min_delta.is_finite())
    {
        anyhow::bail!("v0.70 learning rate/min_delta are invalid");
    }
    if !resume_mode && output_root.exists() {
        anyhow::bail!("v0.70 output directory must be fresh: {output_root:?}");
    }
    if resume_mode && !output_root.is_dir() {
        anyhow::bail!("v0.70 resume requires an existing output directory: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.70 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let parent_metadata = read_parent_metadata(&parent_checkpoint)?;
    validate_parent_metadata(&parent_metadata)?;
    let current_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let current_benchmark_fingerprint =
        format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != current_corpus_fingerprint
        || parent_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint
    {
        anyhow::bail!("v0.70 parent provenance differs from current corpus/benchmark");
    }

    let config = V070Config::production(&parent_metadata.v0520_config)?;
    let train_groups = build_alignment_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        parent_metadata
            .v0520_config
            .base_v0510
            .base_v0500
            .max_sequence_len,
    )?;
    let dev_groups = build_alignment_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        parent_metadata
            .v0520_config
            .base_v0510
            .base_v0500
            .max_sequence_len,
    )?;
    if train_groups.len() < batch_size {
        anyhow::bail!(
            "v0.70 has only {} eligible TRAIN identities for batch_size={batch_size}",
            train_groups.len()
        );
    }
    let requested_dev = if smoke_mode {
        V070_SMOKE_DEV_IDENTITIES
    } else {
        V070_DEV_IDENTITIES
    };
    if dev_groups.len() < requested_dev.min(32) {
        anyhow::bail!(
            "v0.70 has too few eligible DEV identities: {}",
            dev_groups.len()
        );
    }
    let dev_identities = select_dev_identities(
        &corpus.records,
        &dev_groups,
        requested_dev.min(dev_groups.len()),
        seed ^ 0x7000_d3f0_a11e_0001,
    )?;
    let dev_identity_fingerprint =
        format!("fnv1a64:{:016x}", dev_identity_fingerprint(&dev_identities));

    let holdout_reserved = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .count();

    let collator = FoundationCollator::new(
        parent_metadata
            .v0520_config
            .base_v0510
            .base_v0500
            .featurizer_config(),
        FoundationCollatorConfig {
            retention_time_objective: parent_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.spectrum.clone())?;

    let mut parent_varmap = VarMap::new();
    let parent = PeptideFoundationV0520Model::new(
        parent_metadata.v0520_config.clone(),
        VarBuilder::from_varmap(&parent_varmap, DType::F32, &device),
    )?;
    parent_varmap
        .load(parent_checkpoint.join("model.safetensors"))
        .with_context(|| format!("load frozen v0.52 parent from {parent_checkpoint:?}"))?;

    let mut alignment_varmap = VarMap::new();
    let alignment = PeptideSpectrumAlignmentV0700::new(
        config.clone(),
        VarBuilder::from_varmap(&alignment_varmap, DType::F32, &device),
    )?;
    let mut optimizer = FoundationAdamW::new(
        &alignment_varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: run.trainer.weight_decay,
        },
    )?;

    let max_epochs = if smoke_mode { 1 } else { requested_max_epochs };
    let steps_per_epoch = if smoke_mode {
        8usize.min(requested_steps_per_epoch)
    } else {
        requested_steps_per_epoch
    };
    let max_updates = max_epochs.saturating_mul(steps_per_epoch).max(1);
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 250u64.min(max_updates.saturating_sub(1) as u64),
        total_steps: max_updates as u64,
        min_lr_ratio: 0.10,
    };

    if !resume_mode {
        fs::create_dir_all(&output_root)?;
    }
    let parent_sentinel_records = dev_identities
        .iter()
        .take(batch_size.min(dev_identities.len()))
        .map(|item| corpus.records[item.record_index].clone())
        .collect::<Vec<_>>();
    let parent_sentinel_initial =
        frozen_parent_sentinel(&parent, &collator, &parent_sentinel_records, &device)?;

    println!("v0700_version\tv0.70-spectrum-peptide-alignment");
    println!("objective\t{V070_OBJECTIVE}");
    println!("architecture\t{V070_ARCHITECTURE}");
    println!("device\t{device:?}");
    println!("mode\t{mode}");
    println!("parent_v052_checkpoint\t{}", parent_checkpoint.display());
    println!(
        "parent_v052_completed_epochs\t{}",
        parent_metadata.completed_epochs
    );
    println!(
        "parent_v052_completed_updates\t{}",
        parent_metadata.completed_updates
    );
    println!("parent_update_policy\tfrozen_separate_varmap_detached_embeddings");
    println!("peptide_embedding_source\tv052.base_v0500.global_plus_ms2");
    println!("spectrum_input\tobserved_mz_intensity_only_plus_precursor_context");
    println!("optimizer_scope\t{V070_NAMESPACE}.*");
    println!("optimizer_variable_count\t{}", optimizer.variable_count());
    println!("alignment_temperature\t{}", config.temperature);
    println!("alignment_dim\t{}", config.alignment_dim);
    println!("spectrum_model_dim\t{}", config.spectrum.model_dim);
    println!("spectrum_layers\t{}", config.spectrum.spectrum_layers);
    println!("spectrum_heads\t{}", config.spectrum.num_attention_heads);
    println!("train_eligible_unique_identities\t{}", train_groups.len());
    println!("dev_eligible_unique_identities\t{}", dev_groups.len());
    println!("dev_retrieval_identities\t{}", dev_identities.len());
    println!("dev_identity_fingerprint\t{dev_identity_fingerprint}");
    println!("holdout_records_reserved_not_read\t{holdout_reserved}");
    println!("max_epochs\t{max_epochs}");
    println!("steps_per_epoch\t{steps_per_epoch}");
    println!("batch_size\t{batch_size}");
    println!("learning_rate\t{learning_rate}");
    println!("selection_metric\t0.5_global_il_mrr_plus_0.5_mass64_il_mrr");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let metadata = |completed_epochs: usize,
                    completed_updates: usize,
                    dev_selection_score: f64,
                    step0_selection_score: f64|
     -> V070Metadata {
        V070Metadata {
            version: V070_VERSION,
            objective: V070_OBJECTIVE.into(),
            architecture: V070_ARCHITECTURE.into(),
            parent_v052_checkpoint: parent_checkpoint.display().to_string(),
            parent_v052_completed_epochs: parent_metadata.completed_epochs,
            parent_v052_completed_updates: parent_metadata.completed_updates,
            parent_v052_dev_objective: parent_metadata.dev_objective,
            parent_update_policy: "frozen_separate_varmap_detached_embeddings".into(),
            corpus_fingerprint: current_corpus_fingerprint.clone(),
            benchmark_manifest_fingerprint: current_benchmark_fingerprint.clone(),
            config: config.clone(),
            max_epochs,
            batch_size,
            steps_per_epoch,
            patience,
            min_delta,
            seed,
            learning_rate,
            dev_identity_count: dev_identities.len(),
            dev_identity_fingerprint: dev_identity_fingerprint.clone(),
            completed_epochs,
            completed_updates,
            dev_selection_score,
            step0_selection_score,
            smoke_mode,
        }
    };

    let (
        mut global_update,
        start_epoch,
        mut best_epoch,
        mut best_update,
        mut best_metrics,
        step0_metrics,
        mut stale_epochs,
    ) = if resume_mode {
        let _initial = read_v070_metadata(&output_root.join("initial"))?;
        let latest = read_v070_metadata(&output_root.join("latest"))?;
        let best = read_v070_metadata(&output_root.join("best"))?;
        validate_resume_metadata(
            &latest,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &parent_checkpoint,
            &config,
            batch_size,
            steps_per_epoch,
            seed,
            learning_rate,
            &dev_identity_fingerprint,
        )?;
        alignment_varmap.load(output_root.join("latest/model.safetensors"))?;
        optimizer.load_safetensors(&output_root.join("latest/optimizer.safetensors"))?;
        optimizer.set_step_count(latest.completed_updates as u64);
        let best_metrics = read_retrieval_metrics(&output_root.join("best/dev_metrics.tsv"))?;
        let step0_metrics = read_retrieval_metrics(&output_root.join("initial/dev_metrics.tsv"))?;
        (
            latest.completed_updates,
            latest.completed_epochs + 1,
            best.completed_epochs,
            best.completed_updates,
            best_metrics,
            step0_metrics,
            latest
                .completed_epochs
                .saturating_sub(best.completed_epochs),
        )
    } else {
        let (initial_metrics, outcomes) = evaluate_dev_retrieval(
            &parent,
            &alignment,
            &collator,
            &spectrum_collator,
            &corpus.records,
            &dev_identities,
            batch_size,
            &device,
        )?;
        print_retrieval("v0700_dev_initial", 0, initial_metrics);
        save_checkpoint(
            &output_root.join("initial"),
            &alignment_varmap,
            &optimizer,
            &metadata(
                0,
                0,
                initial_metrics.selection_score(),
                initial_metrics.selection_score(),
            ),
            initial_metrics,
            &outcomes,
        )?;
        save_checkpoint(
            &output_root.join("best"),
            &alignment_varmap,
            &optimizer,
            &metadata(
                0,
                0,
                initial_metrics.selection_score(),
                initial_metrics.selection_score(),
            ),
            initial_metrics,
            &outcomes,
        )?;
        (0, 1, 0, 0, initial_metrics, initial_metrics, 0)
    };

    for epoch in start_epoch..=max_epochs {
        let order = deterministic_group_order(
            train_groups.len(),
            seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let mut epoch_loss = 0.0f64;
        for step_in_epoch in 0..steps_per_epoch {
            let mut records = Vec::with_capacity(batch_size);
            for slot in 0..batch_size {
                let position = (step_in_epoch * batch_size + slot) % order.len();
                let group = &train_groups[order[position]];
                let record_slot = (mix64(
                    seed ^ (epoch as u64).rotate_left(13)
                        ^ (step_in_epoch as u64).rotate_left(29)
                        ^ hash64_str(&group.key),
                ) as usize)
                    % group.record_indices.len();
                records.push(corpus.records[group.record_indices[record_slot]].clone());
            }
            if unique_batch_identity_count(&records) != records.len() {
                anyhow::bail!("v0.70 TRAIN batch contains duplicate peptidoform+charge identities");
            }
            global_update += 1;
            let lr =
                lr_schedule.learning_rate(learning_rate, global_update.saturating_sub(1) as u64)?;
            optimizer.set_learning_rate(lr)?;
            let loss = alignment_batch_loss(
                &parent,
                &alignment,
                &collator,
                &spectrum_collator,
                &records,
                seed ^ global_update as u64,
                &device,
            )?;
            let loss_value = f64::from(loss.to_scalar::<f32>()?);
            let gradients = loss.backward()?;
            let update = optimizer.step(&gradients, Some(V070_MAX_GRADIENT_NORM))?;
            epoch_loss += loss_value;
            if global_update <= 4 || step_in_epoch % 50 == 0 {
                println!(
                    "v0700_train\tepoch={epoch}\tstep_in_epoch={step_in_epoch}\tupdate={global_update}\tlr={:.8}\tloss={loss_value:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                    update.learning_rate,
                    update.gradient_norm,
                    update.gradient_scale,
                );
            }
        }
        let mean_epoch_loss = epoch_loss / steps_per_epoch as f64;
        let parent_sentinel =
            frozen_parent_sentinel(&parent, &collator, &parent_sentinel_records, &device)?;
        let parent_delta = (parent_sentinel - parent_sentinel_initial).abs();
        if parent_delta > 1.0e-6 * parent_sentinel_initial.abs().max(1.0) {
            anyhow::bail!(
                "v0.70 frozen v0.52 parent changed: initial={parent_sentinel_initial:.8} current={parent_sentinel:.8} delta={parent_delta:.8}"
            );
        }
        println!(
            "v0700_parent_freeze_audit\tepoch={epoch}\tstatus=PASS\tfingerprint={parent_sentinel:.8}\tdelta={parent_delta:.8}"
        );
        let (dev_metrics, outcomes) = evaluate_dev_retrieval(
            &parent,
            &alignment,
            &collator,
            &spectrum_collator,
            &corpus.records,
            &dev_identities,
            batch_size,
            &device,
        )?;
        print_retrieval("v0700_dev", global_update, dev_metrics);
        println!(
            "v0700_epoch\tepoch={epoch}\tupdate={global_update}\tmean_train_loss={mean_epoch_loss:.6}\tdev_selection_score={:.8}",
            dev_metrics.selection_score()
        );
        save_checkpoint(
            &output_root.join("latest"),
            &alignment_varmap,
            &optimizer,
            &metadata(
                epoch,
                global_update,
                dev_metrics.selection_score(),
                step0_metrics.selection_score(),
            ),
            dev_metrics,
            &outcomes,
        )?;
        let improved = dev_metrics.selection_score() - best_metrics.selection_score() > min_delta;
        if improved {
            best_epoch = epoch;
            best_update = global_update;
            best_metrics = dev_metrics;
            stale_epochs = 0;
            save_checkpoint(
                &output_root.join("best"),
                &alignment_varmap,
                &optimizer,
                &metadata(
                    epoch,
                    global_update,
                    dev_metrics.selection_score(),
                    step0_metrics.selection_score(),
                ),
                dev_metrics,
                &outcomes,
            )?;
            println!(
                "v0700_best_checkpoint\tepoch={best_epoch}\tupdate={best_update}\tselection_score={:.8}",
                best_metrics.selection_score()
            );
        } else {
            stale_epochs += 1;
        }
        if smoke_mode {
            println!("v0700_smoke_complete\tepoch={epoch}\tupdates={global_update}");
            break;
        }
        if stale_epochs >= patience {
            println!(
                "v0700_early_stop\tepoch={epoch}\tupdate={global_update}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_update={best_update}"
            );
            break;
        }
    }

    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!(
        "v0700_training_complete\tbest_epoch={best_epoch}\tbest_update={best_update}\tstep0_selection_score={:.8}\tbest_selection_score={:.8}",
        step0_metrics.selection_score(),
        best_metrics.selection_score()
    );
    print_material_gate(step0_metrics, best_metrics, smoke_mode);
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

fn alignment_batch_loss(
    parent: &PeptideFoundationV0520Model,
    alignment: &PeptideSpectrumAlignmentV0700,
    collator: &FoundationCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    records: &[FoundationTrainingRecord],
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let batch = collator.collate(records, device, seed)?;
    let parent_output = parent.base_v0500_t(&batch.input, &batch.context, false)?;
    let peptide_features = Tensor::cat(
        &[
            &parent_output.representation.global_embedding.detach(),
            &parent_output.representation.ms2_embedding.detach(),
        ],
        1,
    )?;
    let spectra = records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record)
                .ok_or_else(|| anyhow::anyhow!("v0.70 eligible TRAIN record lacks spectrum"))
        })
        .collect::<Result<Vec<_>>>()?;
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let precursor = precursor_features(records, device)?;
    let spectrum_embedding = alignment.encode_spectrum_t(&spectrum_batch, &precursor, true)?;
    let peptide_embedding = alignment.encode_peptide(&peptide_features)?;
    alignment.alignment_loss(&spectrum_embedding, &peptide_embedding)
}

fn evaluate_dev_retrieval(
    parent: &PeptideFoundationV0520Model,
    alignment: &PeptideSpectrumAlignmentV0700,
    collator: &FoundationCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    records: &[FoundationTrainingRecord],
    identities: &[DevIdentity],
    batch_size: usize,
    device: &Device,
) -> Result<(RetrievalMetrics, Vec<QueryOutcome>)> {
    let mut spectrum_rows = Vec::<Vec<f32>>::new();
    let mut peptide_rows = Vec::<Vec<f32>>::new();
    for chunk in identities.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|item| records[item.record_index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, 0)?;
        let parent_output = parent.base_v0500_t(&batch.input, &batch.context, false)?;
        let peptide_features = Tensor::cat(
            &[
                &parent_output.representation.global_embedding.detach(),
                &parent_output.representation.ms2_embedding.detach(),
            ],
            1,
        )?;
        let spectra = owned
            .iter()
            .map(|record| {
                FoundationSpectrum::from_training_record(record)
                    .ok_or_else(|| anyhow::anyhow!("v0.70 DEV record lacks spectrum"))
            })
            .collect::<Result<Vec<_>>>()?;
        let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
        let precursor = precursor_features(&owned, device)?;
        let spectrum_embedding =
            normalize_rows(&alignment.encode_spectrum_t(&spectrum_batch, &precursor, false)?)?;
        let peptide_embedding = normalize_rows(&alignment.encode_peptide(&peptide_features)?)?;
        spectrum_rows.extend(spectrum_embedding.to_vec2::<f32>()?);
        peptide_rows.extend(peptide_embedding.to_vec2::<f32>()?);
    }
    let n = identities.len();
    let d = alignment.config.alignment_dim;
    let spectrum_flat = spectrum_rows.into_iter().flatten().collect::<Vec<_>>();
    let peptide_flat = peptide_rows.into_iter().flatten().collect::<Vec<_>>();
    let spectrum = Tensor::from_vec(spectrum_flat, (n, d), device)?;
    let peptide = Tensor::from_vec(peptide_flat, (n, d), device)?;
    let similarities = spectrum
        .matmul(&peptide.transpose(0, 1)?.contiguous()?)?
        .to_vec2::<f32>()?;
    retrieval_metrics(&similarities, identities, V070_MASS_CANDIDATES)
}

fn retrieval_metrics(
    similarities: &[Vec<f32>],
    identities: &[DevIdentity],
    mass_candidates: usize,
) -> Result<(RetrievalMetrics, Vec<QueryOutcome>)> {
    if similarities.len() != identities.len()
        || similarities.iter().any(|row| row.len() != identities.len())
    {
        anyhow::bail!("v0.70 retrieval similarity matrix shape mismatch");
    }
    let n = identities.len();
    if n == 0 {
        anyhow::bail!("v0.70 retrieval cohort is empty");
    }
    let mut exact_top1 = 0usize;
    let mut exact_top5 = 0usize;
    let mut exact_top10 = 0usize;
    let mut exact_rr = 0.0f64;
    let mut il_top1 = 0usize;
    let mut il_top5 = 0usize;
    let mut il_top10 = 0usize;
    let mut il_rr = 0.0f64;
    let mut exact_ranks = Vec::with_capacity(n);
    let mut mass_true_coverage = 0usize;
    let mut mass_exact_top1 = 0usize;
    let mut mass_exact_top10 = 0usize;
    let mut mass_exact_rr = 0.0f64;
    let mut mass_il_top1 = 0usize;
    let mut mass_il_top10 = 0usize;
    let mut mass_il_rr = 0.0f64;
    let mut outcomes = Vec::with_capacity(n);

    for query in 0..n {
        let mut ranked = (0..n).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            similarities[query][b]
                .partial_cmp(&similarities[query][a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let exact_rank = ranked
            .iter()
            .position(|&candidate| candidate == query)
            .map(|r| r + 1)
            .ok_or_else(|| anyhow::anyhow!("v0.70 exact candidate vanished"))?;
        let il_rank = ranked
            .iter()
            .position(|&candidate| identities[candidate].il_key == identities[query].il_key)
            .map(|r| r + 1)
            .ok_or_else(|| anyhow::anyhow!("v0.70 I/L candidate vanished"))?;
        exact_top1 += usize::from(exact_rank <= 1);
        exact_top5 += usize::from(exact_rank <= 5);
        exact_top10 += usize::from(exact_rank <= 10);
        exact_rr += 1.0 / exact_rank as f64;
        il_top1 += usize::from(il_rank <= 1);
        il_top5 += usize::from(il_rank <= 5);
        il_top10 += usize::from(il_rank <= 10);
        il_rr += 1.0 / il_rank as f64;
        exact_ranks.push(exact_rank);

        let mut mass_pool = (0..n).collect::<Vec<_>>();
        mass_pool.sort_by(|&a, &b| {
            (identities[a].candidate_neutral_mass - identities[query].observed_neutral_mass)
                .abs()
                .partial_cmp(
                    &(identities[b].candidate_neutral_mass
                        - identities[query].observed_neutral_mass)
                        .abs(),
                )
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        mass_pool.truncate(mass_candidates.min(n));
        mass_true_coverage += usize::from(mass_pool.contains(&query));
        mass_pool.sort_by(|&a, &b| {
            similarities[query][b]
                .partial_cmp(&similarities[query][a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        if let Some(position) = mass_pool.iter().position(|&candidate| candidate == query) {
            let rank = position + 1;
            mass_exact_top1 += usize::from(rank <= 1);
            mass_exact_top10 += usize::from(rank <= 10);
            mass_exact_rr += 1.0 / rank as f64;
        }
        if let Some(position) = mass_pool
            .iter()
            .position(|&candidate| identities[candidate].il_key == identities[query].il_key)
        {
            let rank = position + 1;
            mass_il_top1 += usize::from(rank <= 1);
            mass_il_top10 += usize::from(rank <= 10);
            mass_il_rr += 1.0 / rank as f64;
        }
        outcomes.push(QueryOutcome {
            charge: identities[query].charge,
            length: identities[query].length,
            modified: identities[query].modified,
            exact_top1: exact_rank <= 1,
            il_top1: il_rank <= 1,
            exact_top10: exact_rank <= 10,
            il_top10: il_rank <= 10,
            exact_rank,
        });
    }
    let mut sorted_ranks = exact_ranks.clone();
    sorted_ranks.sort_unstable();
    let median_rank = if n % 2 == 0 {
        (sorted_ranks[n / 2 - 1] as f64 + sorted_ranks[n / 2] as f64) / 2.0
    } else {
        sorted_ranks[n / 2] as f64
    };
    let denom = n as f64;
    Ok((
        RetrievalMetrics {
            identities: n,
            exact_top1: exact_top1 as f64 / denom,
            exact_top5: exact_top5 as f64 / denom,
            exact_top10: exact_top10 as f64 / denom,
            exact_mrr: exact_rr / denom,
            il_top1: il_top1 as f64 / denom,
            il_top5: il_top5 as f64 / denom,
            il_top10: il_top10 as f64 / denom,
            il_mrr: il_rr / denom,
            mean_exact_rank: exact_ranks.iter().sum::<usize>() as f64 / denom,
            median_exact_rank: median_rank,
            mass_true_coverage: mass_true_coverage as f64 / denom,
            mass_exact_top1: mass_exact_top1 as f64 / denom,
            mass_exact_top10: mass_exact_top10 as f64 / denom,
            mass_exact_mrr: mass_exact_rr / denom,
            mass_il_top1: mass_il_top1 as f64 / denom,
            mass_il_top10: mass_il_top10 as f64 / denom,
            mass_il_mrr: mass_il_rr / denom,
        },
        outcomes,
    ))
}

fn precursor_features(records: &[FoundationTrainingRecord], device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(records.len() * V070_PRECURSOR_FEATURES);
    for record in records {
        let charge = record.context.charge.map(|z| z as f32).unwrap_or(0.0);
        let mz = record.context.precursor_mz.unwrap_or(0.0);
        let nce = record.context.nce.unwrap_or(0.0);
        values.extend_from_slice(&[
            charge / 6.0,
            if record.context.charge.is_some() {
                1.0
            } else {
                0.0
            },
            mz / 2000.0,
            if record.context.precursor_mz.is_some() {
                1.0
            } else {
                0.0
            },
            nce / 100.0,
            if record.context.nce.is_some() {
                1.0
            } else {
                0.0
            },
        ]);
    }
    Ok(Tensor::from_vec(
        values,
        (records.len(), V070_PRECURSOR_FEATURES),
        device,
    )?)
}

fn normalize_rows(values: &Tensor) -> Result<Tensor> {
    let dims = values.dims2()?;
    let (_, width) = dims;
    let norm = values
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1.0e-8, f64::INFINITY)?
        .unsqueeze(1)?
        .broadcast_as(dims)?;
    let normalized = values.broadcast_div(&norm)?;
    if normalized.dims2()?.1 != width {
        anyhow::bail!("v0.70 normalized embedding width changed unexpectedly");
    }
    Ok(normalized)
}

fn frozen_parent_sentinel(
    parent: &PeptideFoundationV0520Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    device: &Device,
) -> Result<f64> {
    let batch = collator.collate(records, device, 0)?;
    let output = parent.base_v0500_t(&batch.input, &batch.context, false)?;
    let global = f64::from(
        output
            .representation
            .global_embedding
            .sum_all()?
            .to_scalar::<f32>()?,
    );
    let ms2 = f64::from(
        output
            .representation
            .ms2_embedding
            .sum_all()?
            .to_scalar::<f32>()?,
    );
    Ok(global + ms2)
}

fn build_alignment_groups(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    max_sequence_len: usize,
) -> Result<Vec<AlignmentGroup>> {
    let mut groups = BTreeMap::<String, AlignmentGroup>::new();
    for entry in benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
    {
        let record = records
            .get(entry.record_index)
            .ok_or_else(|| anyhow::anyhow!("v0.70 benchmark record index out of range"))?;
        if record.peptidoform.sequence.chars().count() > max_sequence_len {
            continue;
        }
        let Some(charge) = record.context.charge else {
            continue;
        };
        if charge <= 0 || record.context.precursor_mz.is_none() {
            continue;
        }
        if FoundationSpectrum::from_training_record(record).is_none() {
            continue;
        }
        let key = format!("{}|z{charge}", entry.peptidoform);
        let group = groups.entry(key.clone()).or_insert_with(|| AlignmentGroup {
            key,
            peptidoform: entry.peptidoform.clone(),
            sequence: entry.sequence.clone(),
            charge,
            record_indices: Vec::new(),
        });
        if group.peptidoform != entry.peptidoform
            || group.sequence != entry.sequence
            || group.charge != charge
        {
            anyhow::bail!("v0.70 identity grouping collision");
        }
        group.record_indices.push(entry.record_index);
    }
    Ok(groups.into_values().collect())
}

fn select_dev_identities(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    count: usize,
    seed: u64,
) -> Result<Vec<DevIdentity>> {
    let mut group_order = (0..groups.len()).collect::<Vec<_>>();
    group_order.sort_by_key(|&i| mix64(seed ^ hash64_str(&groups[i].key)));
    let mut selected = Vec::with_capacity(count);
    for group_index in group_order.into_iter().take(count) {
        let group = &groups[group_index];
        let mut records_in_group = group.record_indices.clone();
        records_in_group.sort_by_key(|&index| mix64(seed.rotate_left(17) ^ index as u64));
        let index = records_in_group[0];
        let record = &records[index];
        let mz = record
            .context
            .precursor_mz
            .ok_or_else(|| anyhow::anyhow!("v0.70 selected DEV record lacks precursor m/z"))?;
        let observed_neutral_mass = foundation_precursor_neutral_mass(f64::from(mz), group.charge)
            .map_err(anyhow::Error::msg)?;
        let candidate_neutral_mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        selected.push(DevIdentity {
            record_index: index,
            exact_key: group.key.clone(),
            il_key: format!("{}|z{}", il_label(&group.peptidoform), group.charge),
            charge: group.charge,
            length: group.sequence.chars().count(),
            modified: !record.peptidoform.modifications.is_empty(),
            observed_neutral_mass,
            candidate_neutral_mass,
        });
    }
    let unique = selected
        .iter()
        .map(|item| item.exact_key.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() != selected.len() {
        anyhow::bail!("v0.70 DEV retrieval cohort contains duplicate exact identities");
    }
    Ok(selected)
}

fn unique_batch_identity_count(records: &[FoundationTrainingRecord]) -> usize {
    records
        .iter()
        .filter_map(|record| {
            record
                .context
                .charge
                .map(|charge| format!("{}|z{charge}", peptidoform_key(record)))
        })
        .collect::<BTreeSet<_>>()
        .len()
}

fn peptidoform_key(record: &FoundationTrainingRecord) -> String {
    let mut key = record.peptidoform.sequence.clone();
    for modification in &record.peptidoform.modifications {
        key.push('|');
        key.push_str(&format!(
            "{:?}:{}:{:.4}",
            modification.site,
            modification
                .unimod_id
                .map(|v| v.to_string())
                .unwrap_or_else(|| "mass".into()),
            modification.mass_delta
        ));
    }
    key
}

fn deterministic_group_order(length: usize, seed: u64) -> Vec<usize> {
    let mut order = (0..length).collect::<Vec<_>>();
    order.sort_by_key(|&index| mix64(seed ^ index as u64));
    order
}

fn dev_identity_fingerprint(identities: &[DevIdentity]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for item in identities {
        for byte in item.exact_key.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= item.record_index as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn il_label(label: &str) -> String {
    label
        .chars()
        .map(|c| if c == 'I' { 'L' } else { c })
        .collect()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hash64_str(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn print_retrieval(label: &str, update: usize, metrics: RetrievalMetrics) {
    println!(
        "{label}\tupdate={update}\tidentities={}\texact_top1={:.6}\texact_top5={:.6}\texact_top10={:.6}\texact_mrr={:.6}\til_top1={:.6}\til_top5={:.6}\til_top10={:.6}\til_mrr={:.6}\tmean_exact_rank={:.3}\tmedian_exact_rank={:.3}\tmass64_true_coverage={:.6}\tmass64_exact_top1={:.6}\tmass64_exact_top10={:.6}\tmass64_exact_mrr={:.6}\tmass64_il_top1={:.6}\tmass64_il_top10={:.6}\tmass64_il_mrr={:.6}\tselection_score={:.6}",
        metrics.identities,
        metrics.exact_top1,
        metrics.exact_top5,
        metrics.exact_top10,
        metrics.exact_mrr,
        metrics.il_top1,
        metrics.il_top5,
        metrics.il_top10,
        metrics.il_mrr,
        metrics.mean_exact_rank,
        metrics.median_exact_rank,
        metrics.mass_true_coverage,
        metrics.mass_exact_top1,
        metrics.mass_exact_top10,
        metrics.mass_exact_mrr,
        metrics.mass_il_top1,
        metrics.mass_il_top10,
        metrics.mass_il_mrr,
        metrics.selection_score(),
    );
}

fn print_material_gate(step0: RetrievalMetrics, best: RetrievalMetrics, smoke: bool) {
    if smoke {
        println!("v0700_material_gate\tNOT_APPLICABLE_SMOKE");
        return;
    }
    let gain = best.selection_score() - step0.selection_score();
    let gain_pass = gain >= V070_MIN_SELECTION_GAIN;
    let global_top10_pass = best.il_top10 >= V070_MIN_GLOBAL_IL_TOP10;
    let mass_top10_pass = best.mass_il_top10 >= V070_MIN_MASS_IL_TOP10;
    let exact_top1_pass = best.exact_top1 >= V070_MIN_GLOBAL_EXACT_TOP1;
    println!(
        "v0700_gate_selection_gain_ge_0_025\t{}",
        pass_fail(gain_pass)
    );
    println!(
        "v0700_gate_global_il_top10_ge_0_10\t{}",
        pass_fail(global_top10_pass)
    );
    println!(
        "v0700_gate_mass64_il_top10_ge_0_30\t{}",
        pass_fail(mass_top10_pass)
    );
    println!(
        "v0700_gate_global_exact_top1_ge_0_01\t{}",
        pass_fail(exact_top1_pass)
    );
    println!(
        "v0700_material_gate\t{}",
        pass_fail(gain_pass && global_top10_pass && mass_top10_pass && exact_top1_pass)
    );
    println!("v0700_selection_gain\t{gain:.8}");
}

fn pass_fail(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V070Metadata,
    metrics: RetrievalMetrics,
    outcomes: &[QueryOutcome],
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    optimizer.save_safetensors(directory.join("optimizer.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    write_retrieval_metrics(&directory.join("dev_metrics.tsv"), metrics)?;
    write_stratified_metrics(&directory.join("dev_stratified.tsv"), outcomes)?;
    Ok(())
}

fn write_retrieval_metrics(path: &Path, m: RetrievalMetrics) -> Result<()> {
    let text = format!(
        "metric\tvalue\nidentities\t{}\nexact_top1\t{:.12}\nexact_top5\t{:.12}\nexact_top10\t{:.12}\nexact_mrr\t{:.12}\nil_top1\t{:.12}\nil_top5\t{:.12}\nil_top10\t{:.12}\nil_mrr\t{:.12}\nmean_exact_rank\t{:.12}\nmedian_exact_rank\t{:.12}\nmass_true_coverage\t{:.12}\nmass_exact_top1\t{:.12}\nmass_exact_top10\t{:.12}\nmass_exact_mrr\t{:.12}\nmass_il_top1\t{:.12}\nmass_il_top10\t{:.12}\nmass_il_mrr\t{:.12}\nselection_score\t{:.12}\n",
        m.identities,
        m.exact_top1,
        m.exact_top5,
        m.exact_top10,
        m.exact_mrr,
        m.il_top1,
        m.il_top5,
        m.il_top10,
        m.il_mrr,
        m.mean_exact_rank,
        m.median_exact_rank,
        m.mass_true_coverage,
        m.mass_exact_top1,
        m.mass_exact_top10,
        m.mass_exact_mrr,
        m.mass_il_top1,
        m.mass_il_top10,
        m.mass_il_mrr,
        m.selection_score(),
    );
    fs::write(path, text)?;
    Ok(())
}

fn read_retrieval_metrics(path: &Path) -> Result<RetrievalMetrics> {
    let text = fs::read_to_string(path)?;
    let values = text
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect::<BTreeMap<_, _>>();
    let get = |name: &str| -> Result<f64> {
        values
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing metric {name}"))?
            .parse()
            .map_err(anyhow::Error::from)
    };
    Ok(RetrievalMetrics {
        identities: get("identities")? as usize,
        exact_top1: get("exact_top1")?,
        exact_top5: get("exact_top5")?,
        exact_top10: get("exact_top10")?,
        exact_mrr: get("exact_mrr")?,
        il_top1: get("il_top1")?,
        il_top5: get("il_top5")?,
        il_top10: get("il_top10")?,
        il_mrr: get("il_mrr")?,
        mean_exact_rank: get("mean_exact_rank")?,
        median_exact_rank: get("median_exact_rank")?,
        mass_true_coverage: get("mass_true_coverage")?,
        mass_exact_top1: get("mass_exact_top1")?,
        mass_exact_top10: get("mass_exact_top10")?,
        mass_exact_mrr: get("mass_exact_mrr")?,
        mass_il_top1: get("mass_il_top1")?,
        mass_il_top10: get("mass_il_top10")?,
        mass_il_mrr: get("mass_il_mrr")?,
    })
}

fn write_stratified_metrics(path: &Path, outcomes: &[QueryOutcome]) -> Result<()> {
    let mut strata = BTreeMap::<String, Vec<&QueryOutcome>>::new();
    for outcome in outcomes {
        strata
            .entry(format!("charge_z{}", outcome.charge))
            .or_default()
            .push(outcome);
        strata
            .entry(format!("length_{}", length_bin(outcome.length)))
            .or_default()
            .push(outcome);
        strata
            .entry(format!(
                "ptm_{}",
                if outcome.modified {
                    "modified"
                } else {
                    "unmodified"
                }
            ))
            .or_default()
            .push(outcome);
    }
    let mut text =
        String::from("stratum\tn\texact_top1\til_top1\texact_top10\til_top10\tmean_exact_rank\n");
    for (name, rows) in strata {
        let n = rows.len() as f64;
        let exact_top1 = rows.iter().filter(|row| row.exact_top1).count() as f64 / n;
        let il_top1 = rows.iter().filter(|row| row.il_top1).count() as f64 / n;
        let exact_top10 = rows.iter().filter(|row| row.exact_top10).count() as f64 / n;
        let il_top10 = rows.iter().filter(|row| row.il_top10).count() as f64 / n;
        let mean_rank = rows.iter().map(|row| row.exact_rank).sum::<usize>() as f64 / n;
        text.push_str(&format!(
            "{name}\t{}\t{exact_top1:.8}\t{il_top1:.8}\t{exact_top10:.8}\t{il_top10:.8}\t{mean_rank:.4}\n",
            rows.len()
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn length_bin(length: usize) -> &'static str {
    match length {
        0..=7 => "01_07",
        8..=12 => "08_12",
        13..=18 => "13_18",
        19..=25 => "19_25",
        26..=35 => "26_35",
        _ => "36_plus",
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_resume_metadata(
    metadata: &V070Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    parent_checkpoint: &Path,
    config: &V070Config,
    batch_size: usize,
    steps_per_epoch: usize,
    seed: u64,
    learning_rate: f64,
    dev_identity_fingerprint: &str,
) -> Result<()> {
    if metadata.version != V070_VERSION
        || metadata.objective != V070_OBJECTIVE
        || metadata.architecture != V070_ARCHITECTURE
        || metadata.parent_v052_checkpoint != parent_checkpoint.display().to_string()
        || metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        || &metadata.config != config
        || metadata.batch_size != batch_size
        || metadata.steps_per_epoch != steps_per_epoch
        || metadata.seed != seed
        || metadata.learning_rate != learning_rate
        || metadata.dev_identity_fingerprint != dev_identity_fingerprint
        || metadata.smoke_mode
    {
        anyhow::bail!("v0.70 resume metadata does not match the fixed experiment contract");
    }
    Ok(())
}

fn read_parent_metadata(checkpoint: &Path) -> Result<V052ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.52 metadata {path:?}"))?,
    )
    .map_err(anyhow::Error::from)
}

fn read_v070_metadata(checkpoint: &Path) -> Result<V070Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.70 metadata {path:?}"))?,
    )
    .map_err(anyhow::Error::from)
}

fn validate_parent_metadata(metadata: &V052ParentMetadata) -> Result<()> {
    if metadata.version != 520
        || metadata.objective != "v0520_mobility_aware_pair_representation"
        || metadata.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520
        || metadata.completed_epochs == 0
        || metadata.completed_updates == 0
        || !metadata.dev_objective.is_finite()
        || metadata.smoke_mode
    {
        anyhow::bail!("v0.70 requires the selected completed non-smoke v0.52 best checkpoint");
    }
    metadata.v0520_config.validate()?;
    Ok(())
}

fn parse_or<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid argument {}: {error}", index)),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn v0700_config_is_encoder_only_alignment_contract() {
        let config = V070Config::local_smoke();
        config.validate().unwrap();
        assert_eq!(config.precursor_features, 6);
        assert_eq!(config.alignment_dim, 16);
        assert_eq!(config.spectrum.spectrum_layers, 1);
        assert_eq!(config.spectrum.decoder_layers, 1);
    }

    #[test]
    fn v0700_alignment_model_has_only_student_v070_namespace() {
        let config = V070Config::local_smoke();
        let varmap = VarMap::new();
        let model = PeptideSpectrumAlignmentV0700::new(
            config,
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        assert_eq!(model.config.alignment_dim, 16);
        let data = varmap.data().lock().unwrap();
        assert!(!data.is_empty());
        assert!(data.keys().all(|name| name.starts_with("student_v070.")));
    }

    #[test]
    fn v0700_retrieval_metrics_reward_diagonal_alignment() {
        let identities = (0..4)
            .map(|index| DevIdentity {
                record_index: index,
                exact_key: format!("PEPTIDE{index}|z2"),
                il_key: format!("PEPTLDE{index}|z2"),
                charge: 2,
                length: 8,
                modified: false,
                observed_neutral_mass: 1000.0 + index as f64,
                candidate_neutral_mass: 1000.0 + index as f64,
            })
            .collect::<Vec<_>>();
        let similarities = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let (metrics, _) = retrieval_metrics(&similarities, &identities, 4).unwrap();
        assert_eq!(metrics.exact_top1, 1.0);
        assert_eq!(metrics.il_top1, 1.0);
        assert_eq!(metrics.mass_exact_top1, 1.0);
        assert_eq!(metrics.mass_il_top10, 1.0);
    }

    #[test]
    fn v0700_il_label_collapses_isoleucine_only() {
        assert_eq!(il_label("PEPTIDE"), "PEPTLDE");
        assert_eq!(il_label("M[UniMod:35]ILK"), "M[UniMod:35]LLK");
    }
}
