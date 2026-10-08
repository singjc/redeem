//! Minimal end-to-end training loop for [`FoundationModel`](super::runtime::FoundationModel).
//!
//! This is intentionally one training regime rather than another experiment ladder:
//! random initialization (or optional same-architecture warm start), one optimizer,
//! TRAIN-only normalization, Validation-only early stopping, and one checkpoint format.

use super::causal::foundation_causal_sequence_mean_nlls;
use super::collate::{FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig};
use super::corpus::{load_foundation_corpus, FoundationCorpusConfig};
use super::data::{FoundationTrainingRecord, RetentionTimeObjective};
use super::diffusion::FOUNDATION_OPEN_PTM_MASS_SCALE_DA;
use super::experiment::{FoundationBenchmarkManifest, FoundationPartition};
use super::loss::{
    contrastive_info_nce_loss, multi_task_loss_with_ms2_config, FoundationLossWeights,
    FoundationMs2LossConfig, FoundationTargets,
};
use super::mobility_consensus::{
    bruker_ccs_factor_from_values, build_train_consensus_supervision, deterministic_example_order,
    finite_mobility_ccs_indices, MobilityConsensusExample, MobilityConsensusSupervision,
};
use super::normalization::{
    FoundationRegressionNormalization, FoundationRegressionNormalizationStrategy,
    FoundationTargetNormalizationConfig,
};
use super::optimizer::{FoundationAdamW, FoundationAdamWConfig};
use super::runtime::{FoundationCheckpointMetadata, FoundationModel, FoundationModelConfig};
use super::spectrum::FoundationSpectrum;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Production training controller. `Joint` preserves the cleanup-era behavior;
/// `ResearchCurriculum` trains one random-init model with deterministic staged
/// property, representation, mobility, and inverse updates inspired by the
/// successful v0.35-v0.52 research program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoundationTrainingStrategy {
    #[default]
    Joint,
    ResearchCurriculum,
}

fn legacy_training_strategy() -> FoundationTrainingStrategy {
    FoundationTrainingStrategy::Joint
}

fn legacy_warmup_steps() -> usize {
    0
}
fn legacy_min_lr_ratio() -> f64 {
    1.0
}
fn legacy_max_gradient_norm() -> Option<f64> {
    None
}

/// Hyperparameters for the one supported end-to-end foundation training regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTrainingConfig {
    pub model: FoundationModelConfig,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Update schedule. Historical configs deserialize as `joint`.
    #[serde(default = "legacy_training_strategy")]
    pub strategy: FoundationTrainingStrategy,
    /// Linear warmup updates before cosine decay. Historical configs use zero.
    #[serde(default = "legacy_warmup_steps")]
    pub warmup_steps: usize,
    /// Final/base LR ratio after cosine decay. Historical configs use one (constant LR).
    #[serde(default = "legacy_min_lr_ratio")]
    pub min_learning_rate_ratio: f64,
    /// Optional global gradient norm clip. Historical configs leave clipping disabled.
    #[serde(default = "legacy_max_gradient_norm")]
    pub max_gradient_norm: Option<f64>,
    /// Enable TRAIN-only source-aware Bruker mobility consensus in curriculum stage D.
    /// Missing field in historical JSON remains disabled (old direct-CCS stage).
    #[serde(default)]
    pub mobility_consensus_supervision: bool,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub seed: u64,
    /// Optional bounded smoke/debug mode; `None` trains the full partition.
    pub max_train_batches_per_epoch: Option<usize>,
    /// Optional bounded validation mode; `None` evaluates the full partition.
    pub max_validation_batches: Option<usize>,
    /// Maximum number of spectrum-bearing Validation records used for teacher-forced inverse and retrieval metrics per epoch.
    /// Set to zero to disable inverse evaluation metrics without changing the training objective.
    pub inverse_evaluation_records: usize,
    /// Maximum number of spectrum-bearing Validation records decoded greedily per epoch.
    /// Set to zero to disable free-generation metrics while retaining teacher-forced inverse metrics.
    pub generation_evaluation_records: usize,
    pub forward_loss_weights: FoundationLossWeights,
    pub ms2_loss: FoundationMs2LossConfig,
    pub peptide_contrastive_weight: f64,
    pub inverse_weight: f64,
    pub inverse_ptm_mass_weight: f64,
    pub cross_modal_alignment_weight: f64,
    pub contrastive_temperature: f64,
    pub corruption: FoundationCorruptionConfig,
}

impl Default for FoundationTrainingConfig {
    fn default() -> Self {
        Self {
            model: FoundationModelConfig::default(),
            batch_size: 32,
            learning_rate: 2.0e-5,
            strategy: FoundationTrainingStrategy::ResearchCurriculum,
            warmup_steps: 500,
            min_learning_rate_ratio: 0.10,
            max_gradient_norm: Some(1.0),
            mobility_consensus_supervision: true,
            epochs: 20,
            early_stopping_patience: 4,
            seed: 20_261_006,
            max_train_batches_per_epoch: None,
            max_validation_batches: None,
            inverse_evaluation_records: 256,
            generation_evaluation_records: 16,
            forward_loss_weights: FoundationLossWeights {
                rt: 1.0,
                ccs: 1.0,
                ms2: 1.0,
                masked_residue: 0.10,
                chemistry: 0.10,
                contrastive: 0.0,
            },
            ms2_loss: FoundationMs2LossConfig {
                pointwise_weight: 1.0,
                cosine_weight: 0.25,
                pearson_weight: 0.25,
                cosine_epsilon: 1.0e-8,
            },
            peptide_contrastive_weight: 0.10,
            inverse_weight: 1.0,
            inverse_ptm_mass_weight: 0.10,
            cross_modal_alignment_weight: 0.10,
            contrastive_temperature: 0.10,
            corruption: FoundationCorruptionConfig::default(),
        }
    }
}

impl FoundationTrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 || self.epochs == 0 || self.early_stopping_patience == 0 {
            anyhow::bail!(
                "foundation batch_size, epochs, and early_stopping_patience must be positive"
            );
        }
        if !(self.learning_rate > 0.0 && self.learning_rate.is_finite()) {
            anyhow::bail!("foundation learning_rate must be positive and finite");
        }
        if !(self.min_learning_rate_ratio > 0.0
            && self.min_learning_rate_ratio <= 1.0
            && self.min_learning_rate_ratio.is_finite())
        {
            anyhow::bail!("foundation min_learning_rate_ratio must be finite and in (0, 1]");
        }
        if let Some(max_norm) = self.max_gradient_norm {
            if !(max_norm > 0.0 && max_norm.is_finite()) {
                anyhow::bail!("foundation max_gradient_norm must be positive and finite");
            }
        }
        for (name, value) in [
            (
                "max_train_batches_per_epoch",
                self.max_train_batches_per_epoch,
            ),
            ("max_validation_batches", self.max_validation_batches),
        ] {
            if matches!(value, Some(0)) {
                anyhow::bail!("foundation {name} must be positive when set");
            }
        }
        self.model.validate()?;
        for (name, value) in [
            ("rt", self.forward_loss_weights.rt),
            ("ccs", self.forward_loss_weights.ccs),
            ("ms2", self.forward_loss_weights.ms2),
            ("masked_residue", self.forward_loss_weights.masked_residue),
            ("chemistry", self.forward_loss_weights.chemistry),
        ] {
            if !(value >= 0.0 && value.is_finite()) {
                anyhow::bail!(
                    "foundation forward loss weight {name} must be finite and non-negative"
                );
            }
        }
        for (name, value) in [
            (
                "residue_mask_probability",
                self.corruption.residue_mask_probability,
            ),
            (
                "chemistry_mask_probability",
                self.corruption.chemistry_mask_probability,
            ),
        ] {
            if !(0.0..=1.0).contains(&value) {
                anyhow::bail!("foundation corruption {name} must be in [0, 1]");
            }
        }
        for (name, value) in [
            (
                "peptide_contrastive_weight",
                self.peptide_contrastive_weight,
            ),
            ("inverse_weight", self.inverse_weight),
            ("inverse_ptm_mass_weight", self.inverse_ptm_mass_weight),
            (
                "cross_modal_alignment_weight",
                self.cross_modal_alignment_weight,
            ),
        ] {
            if !(value >= 0.0 && value.is_finite()) {
                anyhow::bail!("foundation {name} must be finite and non-negative");
            }
        }
        if !(self.contrastive_temperature > 0.0 && self.contrastive_temperature.is_finite()) {
            anyhow::bail!("foundation contrastive_temperature must be positive and finite");
        }
        self.ms2_loss.validate()?;
        Ok(())
    }
}

/// Minimal subset of the historical prepared-run YAML needed by production training.
#[derive(Debug, Clone, Deserialize)]
struct FoundationTrainingSource {
    corpus: FoundationCorpusConfig,
    benchmark_manifest: PathBuf,
}

/// Load materialized foundation records from the same prepared-run YAML used by training.
pub fn load_foundation_records_from_run(
    run_yaml: impl AsRef<Path>,
) -> Result<Vec<FoundationTrainingRecord>> {
    let (_, records) = load_foundation_run(run_yaml)?;
    Ok(records)
}

/// Load only one benchmark partition from a prepared run.
///
/// This keeps engineering inference/evaluation from silently consuming protected
/// partitions simply because they share the same materialized corpus.
pub fn load_foundation_records_from_run_partition(
    run_yaml: impl AsRef<Path>,
    partition: FoundationPartition,
) -> Result<Vec<FoundationTrainingRecord>> {
    let (benchmark, records) = load_foundation_run(run_yaml)?;
    let indices = benchmark.partition_indices(partition);
    if indices.is_empty() {
        anyhow::bail!("foundation requested partition contains no records");
    }
    indices
        .into_iter()
        .map(|index| {
            records.get(index).cloned().ok_or_else(|| {
                anyhow::anyhow!("foundation benchmark record index {index} is out of range")
            })
        })
        .collect()
}

fn load_foundation_run(
    run_yaml: impl AsRef<Path>,
) -> Result<(FoundationBenchmarkManifest, Vec<FoundationTrainingRecord>)> {
    let run_yaml = run_yaml.as_ref();
    let source: FoundationTrainingSource = serde_yaml::from_str(
        &fs::read_to_string(run_yaml)
            .with_context(|| format!("read foundation run YAML {}", run_yaml.display()))?,
    )
    .with_context(|| format!("parse foundation run YAML {}", run_yaml.display()))?;
    let corpus = load_foundation_corpus(&source.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&source.benchmark_manifest)
        .with_context(|| format!("read benchmark {}", source.benchmark_manifest.display()))?;
    benchmark.validate_against_records(&corpus.records)?;
    Ok((benchmark, corpus.records))
}

/// One epoch's aggregate optimization/evaluation losses.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationEpochLosses {
    pub total: f64,
    pub forward: f64,
    pub inverse: f64,
    pub inverse_ptm_mass: f64,
    pub peptide_contrastive: f64,
    pub cross_modal_alignment: f64,
    pub batches: usize,
}

impl FoundationEpochLosses {
    fn add(&mut self, value: BatchLossScalars) {
        self.total += f64::from(value.total);
        self.forward += f64::from(value.forward);
        self.inverse += f64::from(value.inverse.unwrap_or(0.0));
        self.inverse_ptm_mass += f64::from(value.inverse_ptm_mass.unwrap_or(0.0));
        self.peptide_contrastive += f64::from(value.peptide_contrastive.unwrap_or(0.0));
        self.cross_modal_alignment += f64::from(value.cross_modal_alignment.unwrap_or(0.0));
        self.batches += 1;
    }

    fn means(mut self) -> Self {
        if self.batches > 0 {
            let n = self.batches as f64;
            self.total /= n;
            self.forward /= n;
            self.inverse /= n;
            self.inverse_ptm_mass /= n;
            self.peptide_contrastive /= n;
            self.cross_modal_alignment /= n;
        }
        self
    }
}

#[derive(Debug, Clone)]
pub struct FoundationTrainingSummary {
    pub completed_epochs: usize,
    pub best_validation_loss: f64,
    pub checkpoint: PathBuf,
    pub validation_metrics: PathBuf,
}

/// Native-unit and inverse-task diagnostics for one Validation pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationEvaluationMetrics {
    pub records: usize,
    pub rt_count: usize,
    pub rt_mae_native: Option<f64>,
    pub rt_rmse_native: Option<f64>,
    pub ccs_count: usize,
    pub ccs_mae: Option<f64>,
    pub ccs_rmse: Option<f64>,
    pub ms2_point_count: usize,
    pub ms2_rmse: Option<f64>,
    pub ms2_spectrum_count: usize,
    pub ms2_cosine_similarity: Option<f64>,
    pub ms2_spectral_angle: Option<f64>,
    pub ms2_pearson: Option<f64>,
    pub inverse_sequence_count: usize,
    pub inverse_mean_nll: Option<f64>,
    pub inverse_ptm_site_count: usize,
    pub inverse_ptm_mass_rmse_da: Option<f64>,
    pub retrieval_queries: usize,
    pub retrieval_top1_accuracy: Option<f64>,
    pub retrieval_mrr: Option<f64>,
    pub generation_attempts: usize,
    pub generation_success_rate: Option<f64>,
    pub generation_sequence_exact_rate: Option<f64>,
    pub generation_sequence_il_equivalent_rate: Option<f64>,
    pub generation_mean_length: Option<f64>,
}

struct BatchLoss {
    total: Tensor,
    scalars: BatchLossScalars,
}

#[derive(Debug, Clone, Copy, Default)]
struct BatchLossScalars {
    total: f32,
    forward: f32,
    inverse: Option<f32>,
    inverse_ptm_mass: Option<f32>,
    peptide_contrastive: Option<f32>,
    cross_modal_alignment: Option<f32>,
}

/// Train a foundation model from a prepared run YAML.
///
/// `initial_checkpoint` is model-only warm start for the same stable architecture.
/// Omit it for true random initialization.
pub fn train_foundation_model(
    run_yaml: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    training: FoundationTrainingConfig,
    initial_checkpoint: Option<&Path>,
    device: Device,
) -> Result<FoundationTrainingSummary> {
    training.validate()?;
    let run_yaml = run_yaml.as_ref();
    let output_dir = output_dir.as_ref();
    if output_dir.exists()
        && fs::read_dir(output_dir)
            .with_context(|| format!("read foundation output {}", output_dir.display()))?
            .next()
            .is_some()
    {
        anyhow::bail!(
            "foundation training requires a fresh output directory: {}",
            output_dir.display()
        );
    }

    let source: FoundationTrainingSource = serde_yaml::from_str(
        &fs::read_to_string(run_yaml)
            .with_context(|| format!("read foundation run YAML {}", run_yaml.display()))?,
    )
    .with_context(|| format!("parse foundation run YAML {}", run_yaml.display()))?;
    let corpus = load_foundation_corpus(&source.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&source.benchmark_manifest)
        .with_context(|| format!("read benchmark {}", source.benchmark_manifest.display()))?;
    benchmark.validate_against_records(&corpus.records)?;

    let train_indices = partition_indices(&benchmark, FoundationPartition::Train);
    let validation_indices = partition_indices(&benchmark, FoundationPartition::Validation);
    if train_indices.is_empty() || validation_indices.is_empty() {
        anyhow::bail!(
            "foundation training requires non-empty TRAIN and Validation partitions; observed train={} validation={}",
            train_indices.len(),
            validation_indices.len()
        );
    }

    let mut normalization = FoundationTargetNormalizationConfig {
        rt: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
        ccs: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
    };
    normalization.resolve_from_training_partition(
        &corpus.records,
        &train_indices,
        RetentionTimeObjective::IntrinsicAndObserved,
    )?;

    let mut model_config = training.model.clone();
    model_config.peptide.instrument_vocab_size = source.corpus.instrument_vocab_size;
    let mut model = FoundationModel::new(
        model_config.clone(),
        normalization,
        corpus.instrument_names.clone(),
        device.clone(),
    )?;
    if let Some(checkpoint) = initial_checkpoint {
        model.load_weights(checkpoint)?;
    }

    // Fit the source affine/reliability model using TRAIN identities only.
    // Never refit on Validation; historical TEST and TRAIN-HOLDOUT stay closed.
    let mobility_consensus = if training.strategy == FoundationTrainingStrategy::ResearchCurriculum
        && training.mobility_consensus_supervision
    {
        let eligible = finite_mobility_ccs_indices(&corpus.records, &train_indices);
        if eligible.is_empty() {
            anyhow::bail!("TRAIN contains no valid mobility/charge/mz records for curriculum mobility consensus");
        }
        let fitted =
            build_train_consensus_supervision(&corpus.records, &corpus.provenance, &eligible)?;
        if fitted.examples.is_empty() {
            anyhow::bail!("TRAIN produced no source-aware mobility consensus examples");
        }
        Some(fitted)
    } else {
        None
    };

    let train_collator = FoundationCollator::new(
        model_config.peptide.clone(),
        FoundationCollatorConfig {
            retention_time_objective: RetentionTimeObjective::IntrinsicAndObserved,
            corruption: training.corruption,
        },
    )?;
    let optimizer_config = FoundationAdamWConfig {
        learning_rate: training.learning_rate,
        ..FoundationAdamWConfig::default()
    };
    let mut joint_optimizer = match training.strategy {
        FoundationTrainingStrategy::Joint => {
            Some(FoundationAdamW::new(model.varmap(), optimizer_config)?)
        }
        FoundationTrainingStrategy::ResearchCurriculum => None,
    };
    let mut peptide_optimizer =
        match training.strategy {
            FoundationTrainingStrategy::ResearchCurriculum => Some(
                FoundationAdamW::new_for_prefixes(model.varmap(), optimizer_config, &["peptide."])?,
            ),
            FoundationTrainingStrategy::Joint => None,
        };
    let mut inverse_optimizer = match training.strategy {
        FoundationTrainingStrategy::ResearchCurriculum => Some(FoundationAdamW::new_for_prefixes(
            model.varmap(),
            optimizer_config,
            &["inverse.", "alignment."],
        )?),
        FoundationTrainingStrategy::Joint => None,
    };
    let train_batches_per_epoch = training
        .max_train_batches_per_epoch
        .unwrap_or_else(|| (train_indices.len() + training.batch_size - 1) / training.batch_size)
        .min((train_indices.len() + training.batch_size - 1) / training.batch_size);
    let updates_per_cycle = match training.strategy {
        FoundationTrainingStrategy::Joint => 1usize,
        FoundationTrainingStrategy::ResearchCurriculum => 5usize,
    };
    let planned_updates = training
        .epochs
        .saturating_mul(train_batches_per_epoch)
        .saturating_mul(updates_per_cycle)
        .max(1);
    let mut global_update = 0usize;

    fs::create_dir_all(output_dir)
        .with_context(|| format!("create foundation output {}", output_dir.display()))?;
    if let Some(consensus) = mobility_consensus.as_ref() {
        write_mobility_consensus_provenance(output_dir, consensus)?;
    }
    let mut resolved_training = training.clone();
    resolved_training.model = model_config.clone();
    fs::write(
        output_dir.join("training.yaml"),
        serde_yaml::to_string(&resolved_training)?,
    )
    .with_context(|| {
        format!(
            "write foundation training config to {}",
            output_dir.display()
        )
    })?;
    let history = output_dir.join("training_history.tsv");
    let validation_metrics_path = output_dir.join("validation_metrics.tsv");
    fs::write(
        &history,
        "epoch\ttrain_total\ttrain_forward\ttrain_inverse\ttrain_inverse_ptm_mass\ttrain_peptide_contrastive\ttrain_cross_modal_alignment\tvalidation_total\tvalidation_forward\tvalidation_inverse\tvalidation_inverse_ptm_mass\tvalidation_peptide_contrastive\tvalidation_cross_modal_alignment\n",
    )?;
    fs::write(
        &validation_metrics_path,
        "epoch\trecords\trt_count\trt_mae_native\trt_rmse_native\tccs_count\tccs_mae\tccs_rmse\tms2_point_count\tms2_rmse\tms2_spectrum_count\tms2_cosine_similarity\tms2_spectral_angle\tms2_pearson\tinverse_sequence_count\tinverse_mean_nll\tinverse_ptm_site_count\tinverse_ptm_mass_rmse_da\tretrieval_queries\tretrieval_top1_accuracy\tretrieval_mrr\tgeneration_attempts\tgeneration_success_rate\tgeneration_sequence_exact_rate\tgeneration_sequence_il_equivalent_rate\tgeneration_mean_length\n",
    )?;

    let mut best_validation_loss = f64::INFINITY;
    let mut best_forward_score = f64::INFINITY;
    let mut best_inverse_nll = f64::INFINITY;
    let mut epochs_without_improvement = 0usize;
    let mut completed_epochs = 0usize;

    for epoch in 0..training.epochs {
        let mut order = train_indices.clone();
        shuffle_indices(&mut order, training.seed ^ epoch as u64);
        let train_losses = match training.strategy {
            FoundationTrainingStrategy::Joint => run_epoch_joint(
                &model,
                &corpus.records,
                &order,
                &train_collator,
                &training,
                joint_optimizer.as_mut(),
                training.max_train_batches_per_epoch,
                epoch as u64,
                &mut global_update,
                planned_updates,
                &device,
            )?,
            FoundationTrainingStrategy::ResearchCurriculum => run_epoch_research_curriculum(
                &model,
                &corpus.records,
                &order,
                &train_collator,
                &training,
                mobility_consensus.as_ref(),
                peptide_optimizer
                    .as_mut()
                    .expect("research peptide optimizer"),
                inverse_optimizer
                    .as_mut()
                    .expect("research inverse optimizer"),
                training.max_train_batches_per_epoch,
                epoch as u64,
                &mut global_update,
                planned_updates,
                &device,
            )?,
        };
        let validation_losses = run_epoch_joint(
            &model,
            &corpus.records,
            &validation_indices,
            &train_collator,
            &training,
            None,
            training.max_validation_batches,
            training.seed ^ 0xa5a5_5a5a_1234_5678,
            &mut global_update,
            planned_updates,
            &device,
        )?;
        if !train_losses.total.is_finite() || !validation_losses.total.is_finite() {
            anyhow::bail!(
                "foundation training produced non-finite loss at epoch {}: train={} validation={}",
                epoch + 1,
                train_losses.total,
                validation_losses.total
            );
        }
        completed_epochs = epoch + 1;

        append_history(&history, completed_epochs, train_losses, validation_losses)?;
        let validation_metrics = evaluate_partition(
            &model,
            &corpus.records,
            &validation_indices,
            training.batch_size,
            training.max_validation_batches,
            training.inverse_evaluation_records,
            training.generation_evaluation_records,
            training.ms2_loss.cosine_epsilon,
            &device,
        )?;
        append_validation_metrics(
            &validation_metrics_path,
            completed_epochs,
            validation_metrics,
        )?;
        if let Some(score) = forward_balanced_score(validation_metrics) {
            if score < best_forward_score {
                best_forward_score = score;
                let metadata = checkpoint_metadata(
                    model_config.clone(),
                    normalization,
                    &corpus,
                    &benchmark,
                    completed_epochs,
                    validation_losses.total,
                );
                model.save_checkpoint(output_dir.join("best_forward"), &metadata)?;
            }
        }
        if let Some(inverse_nll) = validation_metrics.inverse_mean_nll {
            if inverse_nll < best_inverse_nll {
                best_inverse_nll = inverse_nll;
                let metadata = checkpoint_metadata(
                    model_config.clone(),
                    normalization,
                    &corpus,
                    &benchmark,
                    completed_epochs,
                    validation_losses.total,
                );
                model.save_checkpoint(output_dir.join("best_inverse"), &metadata)?;
            }
        }
        eprintln!(
            "foundation epoch={} train_loss={:.6} validation_loss={:.6} rt_mae={} ccs_mae={} ms2_cosine={} inverse_nll={} retrieval_mrr={} generation_exact={}",
            completed_epochs,
            train_losses.total,
            validation_losses.total,
            optional_metric(validation_metrics.rt_mae_native),
            optional_metric(validation_metrics.ccs_mae),
            optional_metric(validation_metrics.ms2_cosine_similarity),
            optional_metric(validation_metrics.inverse_mean_nll),
            optional_metric(validation_metrics.retrieval_mrr),
            optional_metric(validation_metrics.generation_sequence_exact_rate),
        );

        if validation_losses.total < best_validation_loss {
            best_validation_loss = validation_losses.total;
            epochs_without_improvement = 0;
            let metadata = checkpoint_metadata(
                model_config.clone(),
                normalization,
                &corpus,
                &benchmark,
                completed_epochs,
                best_validation_loss,
            );
            model.save_checkpoint(output_dir, &metadata)?;
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= training.early_stopping_patience {
                break;
            }
        }
    }

    Ok(FoundationTrainingSummary {
        completed_epochs,
        best_validation_loss,
        checkpoint: output_dir.to_path_buf(),
        validation_metrics: validation_metrics_path,
    })
}

fn checkpoint_metadata(
    model_config: FoundationModelConfig,
    normalization: FoundationTargetNormalizationConfig,
    corpus: &super::corpus::FoundationCorpus,
    benchmark: &FoundationBenchmarkManifest,
    completed_epochs: usize,
    validation_loss: f64,
) -> FoundationCheckpointMetadata {
    let mut metadata = FoundationCheckpointMetadata::new(
        model_config,
        normalization,
        corpus.instrument_names.clone(),
    );
    metadata.completed_epochs = completed_epochs;
    metadata.best_validation_loss = Some(validation_loss);
    metadata.corpus_fingerprint = Some(format!("fnv1a64:{:016x}", corpus.corpus_fingerprint));
    metadata.benchmark_fingerprint =
        Some(format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint()));
    metadata
}

fn forward_balanced_score(metrics: FoundationEvaluationMetrics) -> Option<f64> {
    let rt = metrics.rt_mae_native? / 4.528375;
    let ccs = metrics.ccs_mae? / 8.80611929;
    let cosine = (1.0 - metrics.ms2_cosine_similarity?) / (1.0 - 0.904061);
    let spectral = (1.0 - metrics.ms2_spectral_angle?) / (1.0 - 0.747700);
    let pearson = (1.0 - metrics.ms2_pearson?) / (1.0 - 0.684548);
    Some(0.25 * rt + 0.40 * ccs + 0.12 * cosine + 0.12 * spectral + 0.11 * pearson)
}

fn run_epoch_joint(
    model: &FoundationModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    collator: &FoundationCollator,
    config: &FoundationTrainingConfig,
    mut optimizer: Option<&mut FoundationAdamW>,
    max_batches: Option<usize>,
    seed: u64,
    global_update: &mut usize,
    planned_updates: usize,
    device: &Device,
) -> Result<FoundationEpochLosses> {
    let mut aggregate = FoundationEpochLosses::default();
    for (batch_index, chunk) in indices
        .chunks(config.batch_size)
        .take(max_batches.unwrap_or(usize::MAX))
        .enumerate()
    {
        let owned = chunk
            .iter()
            .map(|&index| {
                records.get(index).cloned().ok_or_else(|| {
                    anyhow::anyhow!("foundation training record index {index} is out of bounds")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if owned.is_empty() {
            continue;
        }
        let batch_seed = seed ^ (batch_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let loss = batch_loss(
            model,
            &owned,
            collator,
            config,
            batch_seed,
            optimizer.is_some(),
            device,
        )?;
        if let Some(opt) = optimizer.as_mut() {
            let lr = scheduled_learning_rate(config, *global_update, planned_updates);
            (*opt).set_learning_rate(lr)?;
            (*opt).backward_step(&loss.total, config.max_gradient_norm)?;
            *global_update = (*global_update).saturating_add(1);
        }
        aggregate.add(loss.scalars);
    }
    if aggregate.batches == 0 {
        anyhow::bail!("foundation epoch produced no batches");
    }
    Ok(aggregate.means())
}

#[derive(Debug, Clone, Copy)]
enum FoundationCurriculumStage {
    Property,
    Representation,
    Mobility,
    Inverse,
}

fn run_epoch_research_curriculum(
    model: &FoundationModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    collator: &FoundationCollator,
    config: &FoundationTrainingConfig,
    mobility_consensus: Option<&MobilityConsensusSupervision>,
    peptide_optimizer: &mut FoundationAdamW,
    inverse_optimizer: &mut FoundationAdamW,
    max_batches: Option<usize>,
    seed: u64,
    global_update: &mut usize,
    planned_updates: usize,
    device: &Device,
) -> Result<FoundationEpochLosses> {
    let mut aggregate = FoundationEpochLosses::default();
    let cycles = indices
        .chunks(config.batch_size)
        .len()
        .min(max_batches.unwrap_or(usize::MAX));
    let consensus_order = mobility_consensus.map(|supervision| {
        deterministic_example_order(
            &supervision.examples,
            cycles.saturating_mul(config.batch_size),
            seed,
            config.seed,
        )
    });
    for (batch_index, chunk) in indices
        .chunks(config.batch_size)
        .take(max_batches.unwrap_or(usize::MAX))
        .enumerate()
    {
        let owned = chunk
            .iter()
            .map(|&index| {
                records.get(index).cloned().ok_or_else(|| {
                    anyhow::anyhow!("foundation training record index {index} is out of bounds")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if owned.is_empty() {
            continue;
        }
        let cycle_seed = seed ^ (batch_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for (stage_index, stage) in [
            FoundationCurriculumStage::Property,
            FoundationCurriculumStage::Property,
            FoundationCurriculumStage::Representation,
            FoundationCurriculumStage::Mobility,
            FoundationCurriculumStage::Inverse,
        ]
        .into_iter()
        .enumerate()
        {
            let stage_seed = cycle_seed ^ (stage_index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
            let staged = staged_training_config(config, stage);
            let stage_collator = match stage {
                FoundationCurriculumStage::Representation => collator,
                _ => model.peptide_collator(),
            };
            let loss = if matches!(stage, FoundationCurriculumStage::Mobility)
                && mobility_consensus.is_some()
            {
                let supervision = mobility_consensus.expect("checked presence");
                let order = consensus_order.as_ref().expect("consensus order exists");
                let start = batch_index.saturating_mul(config.batch_size);
                let end = (start + config.batch_size).min(order.len());
                let examples = order[start..end]
                    .iter()
                    .map(|&index| supervision.examples[index].clone())
                    .collect::<Vec<_>>();
                mobility_consensus_batch_loss(
                    model,
                    records,
                    &examples,
                    staged.forward_loss_weights.ccs,
                    stage_seed,
                    device,
                )?
            } else {
                batch_loss(
                    model,
                    &owned,
                    stage_collator,
                    &staged,
                    stage_seed,
                    true,
                    device,
                )?
            };
            let lr = scheduled_learning_rate(config, *global_update, planned_updates);
            match stage {
                FoundationCurriculumStage::Inverse => {
                    inverse_optimizer.set_learning_rate(lr)?;
                    inverse_optimizer.backward_step(&loss.total, config.max_gradient_norm)?;
                }
                _ => {
                    peptide_optimizer.set_learning_rate(lr)?;
                    peptide_optimizer.backward_step(&loss.total, config.max_gradient_norm)?;
                }
            }
            *global_update = (*global_update).saturating_add(1);
            aggregate.add(loss.scalars);
        }
    }
    if aggregate.batches == 0 {
        anyhow::bail!("foundation research-curriculum epoch produced no updates");
    }
    Ok(aggregate.means())
}

fn mobility_consensus_batch_loss(
    model: &FoundationModel,
    records: &[FoundationTrainingRecord],
    examples: &[MobilityConsensusExample],
    ccs_weight: f64,
    seed: u64,
    device: &Device,
) -> Result<BatchLoss> {
    if examples.is_empty() {
        anyhow::bail!("mobility consensus batch has no examples");
    }
    let mut owned = Vec::with_capacity(examples.len());
    let mut ccs_values = Vec::with_capacity(examples.len());
    let mut weights = Vec::with_capacity(examples.len());
    for example in examples {
        let record = records
            .get(example.representative_index)
            .ok_or_else(|| anyhow::anyhow!("consensus representative outside corpus"))?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("consensus representative missing charge"))?;
        let mz = record
            .context
            .precursor_mz
            .ok_or_else(|| anyhow::anyhow!("consensus representative missing precursor m/z"))?;
        let factor = bruker_ccs_factor_from_values(charge, mz)
            .ok_or_else(|| anyhow::anyhow!("invalid Bruker CCS conversion physics"))?;
        let ccs = f64::from(example.target_mobility) * factor;
        if !ccs.is_finite() || ccs <= 0.0 || !example.weight.is_finite() || example.weight <= 0.0 {
            anyhow::bail!("invalid mobility consensus target/weight");
        }
        ccs_values.push(ccs as f32);
        weights.push(example.weight);
        owned.push(record.clone());
    }
    let input = model.peptide_collator().collate(&owned, device, seed)?;
    let output = model.forward_properties_t(&input.input, &input.context, true)?;
    // The production specialist currently predicts standardized CCS, not native mobility.
    // Convert source-aware mobility consensus to native CCS *before* TRAIN-fitted
    // CCS standardization. This preserves a physically meaningful supervision target.
    let native_ccs = Tensor::from_vec(ccs_values, (examples.len(), 1), device)?;
    let target = model
        .target_normalization()
        .ccs
        .normalize_tensor(&native_ccs)?;
    let weight = Tensor::from_vec(weights, (examples.len(), 1), device)?;
    let delta = (&output.ccs - &target)?;
    let weight_sum = weight.sum_all()?.clamp(1.0e-6, f64::INFINITY)?;
    let mse = delta
        .sqr()?
        .broadcast_mul(&weight)?
        .sum_all()?
        .broadcast_div(&weight_sum)?;
    // Historical-style normalized pseudo-Huber alongside a lower-weight MSE.
    let robust = delta
        .affine(2.0, 0.0)?
        .sqr()?
        .affine(1.0, 1.0)?
        .sqrt()?
        .affine(0.25, -0.25)?;
    let pseudo_huber = robust
        .broadcast_mul(&weight)?
        .sum_all()?
        .broadcast_div(&weight_sum)?;
    let combined = (pseudo_huber + mse.affine(0.25, 0.0)?)?.affine(ccs_weight, 0.0)?;
    let scalar = combined.to_scalar::<f32>()?;
    Ok(BatchLoss {
        total: combined,
        scalars: BatchLossScalars {
            total: scalar,
            forward: scalar,
            ..BatchLossScalars::default()
        },
    })
}

fn write_mobility_consensus_provenance(
    output_dir: &Path,
    supervision: &MobilityConsensusSupervision,
) -> Result<()> {
    let mut detail = String::from("source\traw_records\tshared_identities\taffine_intercept\taffine_slope\tresidual_mae\treliability\n");
    for (source, entry) in &supervision.source_supervision {
        use std::fmt::Write as _;
        let (intercept, slope) = entry.affine_parameters();
        writeln!(
            &mut detail,
            "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            source,
            entry.raw_records,
            entry.shared_identities,
            intercept,
            slope,
            entry.residual_mae,
            entry.reliability
        )?;
    }
    fs::write(output_dir.join("mobility_source_reliability.tsv"), detail)?;
    fs::write(output_dir.join("mobility_consensus_summary.tsv"), format!(
        "metric\tvalue\ntrain_valid_mobility_records\t{}\ntrain_consensus_examples\t{}\nmultisource_examples\t{}\nsingleton_examples\t{}\nmean_weight\t{:.8}\nmean_adjusted_delta\t{:.8}\nfingerprint\tfnv1a64:{:016x}\n",
        supervision.raw_records, supervision.examples.len(),
        supervision.multisource_examples, supervision.singleton_examples,
        supervision.mean_weight, supervision.mean_abs_adjusted_delta_to_consensus,
        supervision.fingerprint,
    ))?;
    Ok(())
}

fn staged_training_config(
    base: &FoundationTrainingConfig,
    stage: FoundationCurriculumStage,
) -> FoundationTrainingConfig {
    let mut config = base.clone();
    config.forward_loss_weights = FoundationLossWeights {
        rt: 0.0,
        ccs: 0.0,
        ms2: 0.0,
        masked_residue: 0.0,
        chemistry: 0.0,
        contrastive: 0.0,
    };
    config.peptide_contrastive_weight = 0.0;
    config.inverse_weight = 0.0;
    config.inverse_ptm_mass_weight = 0.0;
    config.cross_modal_alignment_weight = 0.0;
    match stage {
        FoundationCurriculumStage::Property => {
            config.forward_loss_weights.rt = base.forward_loss_weights.rt;
            config.forward_loss_weights.ms2 = base.forward_loss_weights.ms2;
        }
        FoundationCurriculumStage::Representation => {
            config.forward_loss_weights.masked_residue =
                base.forward_loss_weights.masked_residue.max(0.15);
            config.forward_loss_weights.chemistry = base.forward_loss_weights.chemistry.max(0.10);
            config.peptide_contrastive_weight = base.peptide_contrastive_weight.max(0.05);
        }
        FoundationCurriculumStage::Mobility => {
            config.forward_loss_weights.ccs = base.forward_loss_weights.ccs;
        }
        FoundationCurriculumStage::Inverse => {
            config.inverse_weight = base.inverse_weight;
            config.inverse_ptm_mass_weight = base.inverse_ptm_mass_weight;
            config.cross_modal_alignment_weight = base.cross_modal_alignment_weight;
        }
    }
    config
}

fn scheduled_learning_rate(
    config: &FoundationTrainingConfig,
    update: usize,
    planned_updates: usize,
) -> f64 {
    if config.strategy == FoundationTrainingStrategy::Joint
        && config.warmup_steps == 0
        && (config.min_learning_rate_ratio - 1.0).abs() <= f64::EPSILON
    {
        return config.learning_rate;
    }
    if config.warmup_steps > 0 && update < config.warmup_steps {
        return config.learning_rate * ((update + 1) as f64 / config.warmup_steps as f64);
    }
    let decay_start = config.warmup_steps.min(planned_updates);
    let decay_steps = planned_updates.saturating_sub(decay_start).max(1);
    let progress = update.saturating_sub(decay_start).min(decay_steps) as f64 / decay_steps as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    let ratio = config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine;
    config.learning_rate * ratio
}

fn batch_loss(
    model: &FoundationModel,
    records: &[FoundationTrainingRecord],
    collator: &FoundationCollator,
    config: &FoundationTrainingConfig,
    seed: u64,
    train: bool,
    device: &Device,
) -> Result<BatchLoss> {
    let views = collator.collate_views(records, device, seed)?;
    let first = model.forward_properties_t(&views.first.input, &views.first.context, train)?;
    let targets = normalized_targets(&views.first.targets, model.target_normalization())?;
    let forward = multi_task_loss_with_ms2_config(
        &first,
        &targets,
        config.forward_loss_weights,
        config.ms2_loss,
    )?;
    let mut total = forward.total.clone();

    let peptide_contrastive = if records.len() > 1 && config.peptide_contrastive_weight > 0.0 {
        let second =
            model.forward_properties_t(&views.second.input, &views.second.context, train)?;
        let loss = contrastive_info_nce_loss(
            &first.contrastive_projection,
            &second.contrastive_projection,
            config.contrastive_temperature,
        )?;
        total = (total + loss.affine(config.peptide_contrastive_weight, 0.0)?)?;
        Some(loss)
    } else {
        None
    };

    let inverse_records = records
        .iter()
        .filter(|record| FoundationSpectrum::from_training_record(record).is_some())
        .cloned()
        .collect::<Vec<_>>();

    let mut inverse_loss = None;
    let mut inverse_ptm_mass_loss = None;
    let mut cross_modal_alignment = None;
    if !inverse_records.is_empty() && config.inverse_weight > 0.0 {
        let spectra = inverse_records
            .iter()
            .map(|record| FoundationSpectrum::from_training_record(record).unwrap())
            .collect::<Vec<_>>();
        let spectrum_batch = model.spectrum_collator().collate(&spectra, device)?;
        let peptides = inverse_records
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect::<Vec<_>>();
        let causal_batch = model.causal_collator().collate(&peptides, device)?;
        let clean = model
            .peptide_collator()
            .collate(&inverse_records, device, 0)?;
        let inverse_output =
            model.inverse_forward_t(&causal_batch, &spectrum_batch, &clean.context, train)?;
        let token_loss =
            super::causal::foundation_causal_next_token_loss(&inverse_output, &causal_batch)?;
        total = (total + token_loss.affine(config.inverse_weight, 0.0)?)?;
        inverse_loss = Some(token_loss);

        if config.inverse_ptm_mass_weight > 0.0 {
            let mass_loss = model.inverse_mass_loss(&inverse_output, &causal_batch)?;
            total = (total + mass_loss.affine(config.inverse_ptm_mass_weight, 0.0)?)?;
            inverse_ptm_mass_loss = Some(mass_loss);
        }

        if inverse_records.len() > 1 && config.cross_modal_alignment_weight > 0.0 {
            let peptide_output = model.forward_properties_t(&clean.input, &clean.context, train)?;
            let spectrum_projection = model.project_spectrum(&inverse_output.spectrum_embedding)?;
            let alignment = contrastive_info_nce_loss(
                &peptide_output.contrastive_projection,
                &spectrum_projection,
                config.contrastive_temperature,
            )?;
            total = (total + alignment.affine(config.cross_modal_alignment_weight, 0.0)?)?;
            cross_modal_alignment = Some(alignment);
        }
    }

    let scalars = BatchLossScalars {
        total: total.to_scalar::<f32>()?,
        forward: forward.total.to_scalar::<f32>()?,
        inverse: optional_scalar(inverse_loss.as_ref())?,
        inverse_ptm_mass: optional_scalar(inverse_ptm_mass_loss.as_ref())?,
        peptide_contrastive: optional_scalar(peptide_contrastive.as_ref())?,
        cross_modal_alignment: optional_scalar(cross_modal_alignment.as_ref())?,
    };
    Ok(BatchLoss { total, scalars })
}

fn normalized_targets(
    targets: &FoundationTargets,
    normalization: FoundationTargetNormalizationConfig,
) -> Result<FoundationTargets> {
    let mut normalized = targets.clone();
    normalized.rt = targets
        .rt
        .as_ref()
        .map(|value| normalization.rt.normalize_tensor(value))
        .transpose()?;
    normalized.ccs = targets
        .ccs
        .as_ref()
        .map(|value| normalization.ccs.normalize_tensor(value))
        .transpose()?;
    Ok(normalized)
}

fn optional_scalar(value: Option<&Tensor>) -> Result<Option<f32>> {
    Ok(value.map(|tensor| tensor.to_scalar::<f32>()).transpose()?)
}

fn partition_indices(
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
) -> Vec<usize> {
    benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
        .map(|entry| entry.record_index)
        .collect()
}

fn shuffle_indices(values: &mut [usize], seed: u64) {
    if values.len() < 2 {
        return;
    }
    let mut state = seed.max(1);
    for index in (1..values.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let swap = (state as usize) % (index + 1);
        values.swap(index, swap);
    }
}

#[derive(Debug, Default)]
struct EvaluationAccumulator {
    records: usize,
    rt_abs_error_sum: f64,
    rt_squared_error_sum: f64,
    rt_count: usize,
    ccs_abs_error_sum: f64,
    ccs_squared_error_sum: f64,
    ccs_count: usize,
    ms2_squared_error_sum: f64,
    ms2_point_count: usize,
    ms2_cosine_sum: f64,
    ms2_spectral_angle_sum: f64,
    ms2_pearson_sum: f64,
    ms2_pearson_count: usize,
    ms2_spectrum_count: usize,
    inverse_nll_sum: f64,
    inverse_sequence_count: usize,
    inverse_ptm_scaled_squared_error_sum: f64,
    inverse_ptm_site_count: usize,
    retrieval_top1_count: usize,
    retrieval_reciprocal_rank_sum: f64,
    retrieval_queries: usize,
    generation_attempts: usize,
    generation_successes: usize,
    generation_exact: usize,
    generation_il_equivalent: usize,
    generation_length_sum: usize,
}

impl EvaluationAccumulator {
    fn finish(self) -> FoundationEvaluationMetrics {
        FoundationEvaluationMetrics {
            records: self.records,
            rt_count: self.rt_count,
            rt_mae_native: ratio(self.rt_abs_error_sum, self.rt_count),
            rt_rmse_native: ratio(self.rt_squared_error_sum, self.rt_count).map(f64::sqrt),
            ccs_count: self.ccs_count,
            ccs_mae: ratio(self.ccs_abs_error_sum, self.ccs_count),
            ccs_rmse: ratio(self.ccs_squared_error_sum, self.ccs_count).map(f64::sqrt),
            ms2_point_count: self.ms2_point_count,
            ms2_rmse: ratio(self.ms2_squared_error_sum, self.ms2_point_count).map(f64::sqrt),
            ms2_spectrum_count: self.ms2_spectrum_count,
            ms2_cosine_similarity: ratio(self.ms2_cosine_sum, self.ms2_spectrum_count),
            ms2_spectral_angle: ratio(self.ms2_spectral_angle_sum, self.ms2_spectrum_count),
            ms2_pearson: ratio(self.ms2_pearson_sum, self.ms2_pearson_count),
            inverse_sequence_count: self.inverse_sequence_count,
            inverse_mean_nll: ratio(self.inverse_nll_sum, self.inverse_sequence_count),
            inverse_ptm_site_count: self.inverse_ptm_site_count,
            inverse_ptm_mass_rmse_da: ratio(
                self.inverse_ptm_scaled_squared_error_sum,
                self.inverse_ptm_site_count,
            )
            .map(|mse| mse.sqrt() * f64::from(FOUNDATION_OPEN_PTM_MASS_SCALE_DA)),
            retrieval_queries: self.retrieval_queries,
            retrieval_top1_accuracy: ratio(
                self.retrieval_top1_count as f64,
                self.retrieval_queries,
            ),
            retrieval_mrr: ratio(self.retrieval_reciprocal_rank_sum, self.retrieval_queries),
            generation_attempts: self.generation_attempts,
            generation_success_rate: ratio(
                self.generation_successes as f64,
                self.generation_attempts,
            ),
            generation_sequence_exact_rate: ratio(
                self.generation_exact as f64,
                self.generation_attempts,
            ),
            generation_sequence_il_equivalent_rate: ratio(
                self.generation_il_equivalent as f64,
                self.generation_attempts,
            ),
            generation_mean_length: ratio(
                self.generation_length_sum as f64,
                self.generation_successes,
            ),
        }
    }
}

fn evaluate_partition(
    model: &FoundationModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    max_batches: Option<usize>,
    inverse_limit: usize,
    generation_limit: usize,
    ms2_cosine_epsilon: f64,
    device: &Device,
) -> Result<FoundationEvaluationMetrics> {
    let mut aggregate = EvaluationAccumulator::default();

    // Forward metrics use the same bounded Validation slice as early-stopping loss,
    // but with clean (uncorrupted) peptide inputs and native-unit predictions.
    for chunk in indices
        .chunks(batch_size)
        .take(max_batches.unwrap_or(usize::MAX))
    {
        let owned = collect_records(records, chunk, "evaluation")?;
        if owned.is_empty() {
            continue;
        }
        aggregate.records += owned.len();
        let predictions = model.predict_records(&owned)?;
        for (record, prediction) in owned.iter().zip(&predictions) {
            accumulate_forward_metrics(&mut aggregate, record, prediction, ms2_cosine_epsilon);
        }
    }

    if aggregate.records == 0 {
        anyhow::bail!("foundation evaluation produced no records");
    }

    // Inverse diagnostics deliberately scan the whole Validation assignment for
    // spectrum-bearing rows before applying their own cap. This prevents a
    // property-heavy prefix of Validation from silently producing zero inverse
    // metrics, while still keeping the engineering/scientific evaluation bounded.
    let inverse_indices = indices
        .iter()
        .copied()
        .filter(|&index| {
            records
                .get(index)
                .and_then(FoundationSpectrum::from_training_record)
                .is_some()
        })
        .take(inverse_limit)
        .collect::<Vec<_>>();

    for chunk in inverse_indices.chunks(batch_size) {
        let inverse_records = collect_records(records, chunk, "inverse evaluation")?;
        if inverse_records.is_empty() {
            continue;
        }

        let spectra = inverse_records
            .iter()
            .map(|record| FoundationSpectrum::from_training_record(record).unwrap())
            .collect::<Vec<_>>();
        let spectrum_batch = model.spectrum_collator().collate(&spectra, device)?;
        let peptides = inverse_records
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect::<Vec<_>>();
        let causal_batch = model.causal_collator().collate(&peptides, device)?;
        let clean = model
            .peptide_collator()
            .collate(&inverse_records, device, 0)?;
        let inverse_output =
            model.inverse_forward_t(&causal_batch, &spectrum_batch, &clean.context, false)?;

        let sequence_nlls = foundation_causal_sequence_mean_nlls(&inverse_output, &causal_batch)?
            .to_vec1::<f32>()?;
        for value in sequence_nlls {
            if value.is_finite() {
                aggregate.inverse_nll_sum += f64::from(value);
                aggregate.inverse_sequence_count += 1;
            }
        }

        let ptm_sites = causal_batch
            .target_modification_mask
            .sum_all()?
            .to_scalar::<f32>()?;
        if ptm_sites > 0.0 && ptm_sites.is_finite() {
            let scaled_mse = model
                .inverse_mass_loss(&inverse_output, &causal_batch)?
                .to_scalar::<f32>()?;
            if scaled_mse.is_finite() {
                aggregate.inverse_ptm_scaled_squared_error_sum +=
                    f64::from(scaled_mse) * f64::from(ptm_sites);
                aggregate.inverse_ptm_site_count += ptm_sites.round() as usize;
            }
        }

        if inverse_records.len() > 1 {
            let peptide_output = model.forward_properties_t(&clean.input, &clean.context, false)?;
            let spectrum_projection = model.project_spectrum(&inverse_output.spectrum_embedding)?;
            accumulate_retrieval_metrics(
                &mut aggregate,
                &peptide_output.contrastive_projection,
                &spectrum_projection,
            )?;
        }

        for record in &inverse_records {
            if aggregate.generation_attempts >= generation_limit {
                break;
            }
            aggregate.generation_attempts += 1;
            if let Ok(generated) = model.generate_peptide(record) {
                aggregate.generation_successes += 1;
                aggregate.generation_length_sum += generated.sequence.chars().count();
                if generated.sequence == record.peptidoform.sequence {
                    aggregate.generation_exact += 1;
                }
                if il_equivalent(&generated.sequence, &record.peptidoform.sequence) {
                    aggregate.generation_il_equivalent += 1;
                }
            }
        }
    }

    Ok(aggregate.finish())
}

fn collect_records(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    context: &str,
) -> Result<Vec<FoundationTrainingRecord>> {
    indices
        .iter()
        .map(|&index| {
            records.get(index).cloned().ok_or_else(|| {
                anyhow::anyhow!("foundation {context} record index {index} is out of bounds")
            })
        })
        .collect()
}

fn accumulate_forward_metrics(
    aggregate: &mut EvaluationAccumulator,
    record: &FoundationTrainingRecord,
    prediction: &crate::models::model_interface::PredictionOutput,
    cosine_epsilon: f64,
) {
    if let (Some(target), Some(predicted)) = (
        training_rt_target(record).filter(|value| value.is_finite()),
        prediction.rt.filter(|value| value.is_finite()),
    ) {
        let error = f64::from(predicted - target);
        aggregate.rt_abs_error_sum += error.abs();
        aggregate.rt_squared_error_sum += error * error;
        aggregate.rt_count += 1;
    }

    if let (Some(target), Some(predicted)) = (
        record.ccs.filter(|value| value.is_finite()),
        prediction.ccs.filter(|value| value.is_finite()),
    ) {
        let error = f64::from(predicted - target);
        aggregate.ccs_abs_error_sum += error.abs();
        aggregate.ccs_squared_error_sum += error * error;
        aggregate.ccs_count += 1;
    }

    let Some(predicted) = prediction.ms2.as_ref() else {
        return;
    };
    let Some(channels) = predicted.first().map(Vec::len) else {
        return;
    };
    if channels == 0 {
        return;
    }
    let cleavage = predicted.len();
    let mut target = vec![vec![None::<f32>; channels]; cleavage];
    for fragment in &record.fragments {
        if fragment.cleavage_index < cleavage
            && fragment.channel < channels
            && fragment.intensity.is_finite()
        {
            target[fragment.cleavage_index][fragment.channel] = Some(fragment.intensity);
        }
    }

    let mut dot = 0.0f64;
    let mut prediction_norm = 0.0f64;
    let mut target_norm = 0.0f64;
    let mut points = 0usize;
    let mut pred_values = Vec::new();
    let mut target_values = Vec::new();
    for cleavage_index in 0..cleavage {
        for channel in 0..channels {
            let Some(target_value) = target[cleavage_index][channel] else {
                continue;
            };
            let predicted_value = predicted[cleavage_index][channel];
            if !predicted_value.is_finite() {
                continue;
            }
            let p = f64::from(predicted_value);
            let t = f64::from(target_value);
            let error = p - t;
            aggregate.ms2_squared_error_sum += error * error;
            aggregate.ms2_point_count += 1;
            dot += p * t;
            prediction_norm += p * p;
            target_norm += t * t;
            pred_values.push(p);
            target_values.push(t);
            points += 1;
        }
    }
    if points > 0 {
        let denominator =
            (prediction_norm + cosine_epsilon).sqrt() * (target_norm + cosine_epsilon).sqrt();
        if denominator > 0.0 && denominator.is_finite() {
            let cosine = (dot / denominator).clamp(-1.0, 1.0);
            if cosine.is_finite() {
                aggregate.ms2_cosine_sum += cosine;
                aggregate.ms2_spectral_angle_sum +=
                    1.0 - (2.0 / std::f64::consts::PI) * cosine.acos();
                aggregate.ms2_spectrum_count += 1;
                if let Some(pearson) = pearson_correlation(&pred_values, &target_values) {
                    aggregate.ms2_pearson_sum += pearson;
                    aggregate.ms2_pearson_count += 1;
                }
            }
        }
    }
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let n = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / n;
    let right_mean = right.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&a, &b) in left.iter().zip(right) {
        let da = a - left_mean;
        let db = b - right_mean;
        covariance += da * db;
        left_variance += da * da;
        right_variance += db * db;
    }
    let denominator = (left_variance * right_variance).sqrt();
    (denominator > 0.0 && denominator.is_finite())
        .then(|| (covariance / denominator).clamp(-1.0, 1.0))
}

fn accumulate_retrieval_metrics(
    aggregate: &mut EvaluationAccumulator,
    peptide_projection: &Tensor,
    spectrum_projection: &Tensor,
) -> Result<()> {
    let peptides = peptide_projection.to_vec2::<f32>()?;
    let spectra = spectrum_projection.to_vec2::<f32>()?;
    if peptides.len() != spectra.len() || peptides.len() < 2 {
        return Ok(());
    }
    if peptides.first().map(Vec::len) != spectra.first().map(Vec::len) {
        anyhow::bail!("foundation cross-modal retrieval projection dimensions differ");
    }
    let peptide_norm = l2_normalize_rows(&peptides);
    let spectrum_norm = l2_normalize_rows(&spectra);
    if peptide_norm.iter().any(Option::is_none) || spectrum_norm.iter().any(Option::is_none) {
        return Ok(());
    }
    let peptide_norm = peptide_norm.into_iter().flatten().collect::<Vec<_>>();
    let spectrum_norm = spectrum_norm.into_iter().flatten().collect::<Vec<_>>();
    accumulate_retrieval_direction(aggregate, &peptide_norm, &spectrum_norm);
    accumulate_retrieval_direction(aggregate, &spectrum_norm, &peptide_norm);
    Ok(())
}

fn accumulate_retrieval_direction(
    aggregate: &mut EvaluationAccumulator,
    queries: &[Vec<f64>],
    candidates: &[Vec<f64>],
) {
    for (index, query) in queries.iter().enumerate() {
        let positive = dot_product(query, &candidates[index]);
        let mut better = 0usize;
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            if candidate_index != index && dot_product(query, candidate) > positive {
                better += 1;
            }
        }
        let rank = better + 1;
        aggregate.retrieval_queries += 1;
        if rank == 1 {
            aggregate.retrieval_top1_count += 1;
        }
        aggregate.retrieval_reciprocal_rank_sum += 1.0 / rank as f64;
    }
}

fn l2_normalize_rows(rows: &[Vec<f32>]) -> Vec<Option<Vec<f64>>> {
    rows.iter()
        .map(|row| {
            let norm = row
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            if !(norm > 0.0 && norm.is_finite()) {
                return None;
            }
            Some(row.iter().map(|value| f64::from(*value) / norm).collect())
        })
        .collect()
}

fn dot_product(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn training_rt_target(record: &FoundationTrainingRecord) -> Option<f32> {
    record.retention_time.normalized
}

fn il_equivalent(left: &str, right: &str) -> bool {
    left.chars().count() == right.chars().count()
        && left
            .chars()
            .zip(right.chars())
            .all(|(a, b)| canonical_il(a) == canonical_il(b))
}

fn canonical_il(residue: char) -> char {
    if matches!(residue, 'I' | 'L') {
        'J'
    } else {
        residue
    }
}

fn ratio(sum: f64, count: usize) -> Option<f64> {
    if count > 0 {
        Some(sum / count as f64)
    } else {
        None
    }
}

fn optional_metric(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn append_validation_metrics(
    path: &Path,
    epoch: usize,
    metrics: FoundationEvaluationMetrics,
) -> Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(
        file,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        epoch,
        metrics.records,
        metrics.rt_count,
        tsv_metric(metrics.rt_mae_native),
        tsv_metric(metrics.rt_rmse_native),
        metrics.ccs_count,
        tsv_metric(metrics.ccs_mae),
        tsv_metric(metrics.ccs_rmse),
        metrics.ms2_point_count,
        tsv_metric(metrics.ms2_rmse),
        metrics.ms2_spectrum_count,
        tsv_metric(metrics.ms2_cosine_similarity),
        tsv_metric(metrics.ms2_spectral_angle),
        tsv_metric(metrics.ms2_pearson),
        metrics.inverse_sequence_count,
        tsv_metric(metrics.inverse_mean_nll),
        metrics.inverse_ptm_site_count,
        tsv_metric(metrics.inverse_ptm_mass_rmse_da),
        metrics.retrieval_queries,
        tsv_metric(metrics.retrieval_top1_accuracy),
        tsv_metric(metrics.retrieval_mrr),
        metrics.generation_attempts,
        tsv_metric(metrics.generation_success_rate),
        tsv_metric(metrics.generation_sequence_exact_rate),
        tsv_metric(metrics.generation_sequence_il_equivalent_rate),
        tsv_metric(metrics.generation_mean_length),
    )?;
    Ok(())
}

fn tsv_metric(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.8}")).unwrap_or_default()
}

fn append_history(
    path: &Path,
    epoch: usize,
    train: FoundationEpochLosses,
    validation: FoundationEpochLosses,
) -> Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(
        file,
        "{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
        epoch,
        train.total,
        train.forward,
        train.inverse,
        train.inverse_ptm_mass,
        train.peptide_contrastive,
        train.cross_modal_alignment,
        validation.total,
        validation.forward,
        validation.inverse,
        validation.inverse_ptm_mass,
        validation.peptide_contrastive,
        validation.cross_modal_alignment,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_training_config_enables_forward_inverse_and_alignment() {
        let config = FoundationTrainingConfig::default();
        config.validate().unwrap();
        assert!(config.inverse_weight > 0.0);
        assert!(config.cross_modal_alignment_weight > 0.0);
        assert!(config.peptide_contrastive_weight > 0.0);
        assert_eq!(config.inverse_evaluation_records, 256);
        assert_eq!(config.generation_evaluation_records, 16);
    }

    #[test]
    fn unified_defaults_use_research_curriculum_controls() {
        let config = FoundationTrainingConfig::default();
        assert_eq!(
            config.strategy,
            FoundationTrainingStrategy::ResearchCurriculum
        );
        assert_eq!(config.warmup_steps, 500);
        assert!((config.min_learning_rate_ratio - 0.10).abs() < 1e-12);
        assert_eq!(config.max_gradient_norm, Some(1.0));
        assert!(config.mobility_consensus_supervision);
        assert!((scheduled_learning_rate(&config, 0, 10_000) - 4.0e-8).abs() < 1e-12);
        let final_lr = scheduled_learning_rate(&config, 9_999, 10_000);
        assert!(final_lr >= config.learning_rate * config.min_learning_rate_ratio);
        assert!(final_lr < config.learning_rate * 0.101);
    }

    #[test]
    fn historical_training_config_without_curriculum_fields_stays_joint_constant_lr() {
        let mut value = serde_yaml::to_value(FoundationTrainingConfig::default()).unwrap();
        for key in [
            "strategy",
            "warmup_steps",
            "min_learning_rate_ratio",
            "max_gradient_norm",
            "mobility_consensus_supervision",
        ] {
            value
                .as_mapping_mut()
                .unwrap()
                .remove(&serde_yaml::Value::String(key.into()));
        }
        let restored: FoundationTrainingConfig = serde_yaml::from_value(value).unwrap();
        assert_eq!(restored.strategy, FoundationTrainingStrategy::Joint);
        assert_eq!(restored.warmup_steps, 0);
        assert_eq!(restored.min_learning_rate_ratio, 1.0);
        assert_eq!(restored.max_gradient_norm, None);
        assert!(!restored.mobility_consensus_supervision);
    }

    #[test]
    fn il_equivalence_collapses_only_isoleucine_and_leucine() {
        assert!(il_equivalent("PEPTIDE", "PEPTLDE"));
        assert!(il_equivalent("LLLL", "IIII"));
        assert!(!il_equivalent("PEPTIDE", "PEPTVDE"));
        assert!(!il_equivalent("PEPTIDE", "PEPTIDES"));
    }

    #[test]
    fn retrieval_metrics_reward_diagonal_pairs() {
        let peptide = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let spectrum = Tensor::from_vec(vec![0.9f32, 0.1, 0.1, 0.9], (2, 2), &Device::Cpu).unwrap();
        let mut metrics = EvaluationAccumulator::default();
        accumulate_retrieval_metrics(&mut metrics, &peptide, &spectrum).unwrap();
        assert_eq!(metrics.retrieval_queries, 4);
        assert_eq!(metrics.retrieval_top1_count, 4);
        assert!((metrics.retrieval_reciprocal_rank_sum - 4.0).abs() < 1e-9);
    }

    #[test]
    fn training_shuffle_is_seeded_and_permutation_preserving() {
        let mut first = (0..16).collect::<Vec<_>>();
        let mut second = first.clone();
        shuffle_indices(&mut first, 42);
        shuffle_indices(&mut second, 42);
        assert_eq!(first, second);
        first.sort_unstable();
        assert_eq!(first, (0..16).collect::<Vec<_>>());
    }
}
