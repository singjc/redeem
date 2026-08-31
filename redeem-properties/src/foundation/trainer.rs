//! Candle training loop for multi-task peptide foundation pretraining.
//!
//! The trainer operates on heterogeneous [`FoundationTrainingRecord`](crate::foundation::FoundationTrainingRecord)
//! batches rather than the legacy property-specific tensor interface. It owns
//! a serializable AdamW optimizer, optional gradient clipping, deterministic
//! learning-rate scheduling, validation, exact checkpoint/resume state, and a
//! bounded early-stopping fit loop.

use super::checkpoint::{
    foundation_checkpoint_paths, FoundationCheckpointMetadata, FoundationCheckpointProvenance,
    FoundationTrainingProgress,
};
use super::collate::{
    FoundationCollator, FoundationCollatorConfig, FoundationTrainingViews,
};
use super::config::FoundationConfig;
use super::control::{
    FoundationFitConfig, FoundationLearningRateSchedule, FoundationSplitMix64,
};
use super::corpus::FoundationRecordProvenance;
use super::data::{FoundationTrainingRecord, TrainingContext};
use super::loss::{
    contrastive_info_nce_loss, multi_task_loss, FoundationLossWeights, FoundationLosses,
};
use super::featurize::PeptidoformInput;
use super::model::FoundationMultiTaskOutput;
use super::optimizer::{FoundationAdamW, FoundationAdamWConfig};
use super::normalization::{
    FoundationRegressionNormalization, FoundationTargetNormalizationConfig,
};
use super::sampling::{
    sample_foundation_training_indices, sample_foundation_validation_indices,
    FoundationSamplingConfig,
};
use super::wrapper::FoundationModelWrapper;
use anyhow::{Context, Result as AnyResult};
use candle_core::{backprop::GradStore, Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Optional per-task gradient diagnostics.
///
/// Computing a task gradient norm requires an additional backward pass for that
/// task, so diagnostics are intentionally sampled rather than run every step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationGradientDiagnosticsConfig {
    /// Enable weighted per-task gradient-norm measurements.
    pub enabled: bool,
    /// Measure on optimizer steps where `global_step % every_n_steps == 0`.
    pub every_n_steps: u64,
}

impl Default for FoundationGradientDiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_n_steps: 1,
        }
    }
}

impl FoundationGradientDiagnosticsConfig {
    fn validate(self) -> Result<Self> {
        if self.enabled && self.every_n_steps == 0 {
            candle_core::bail!(
                "foundation gradient diagnostics every_n_steps must be at least 1 when enabled"
            );
        }
        Ok(self)
    }

    fn should_measure(self, global_step: u64) -> bool {
        self.enabled && self.every_n_steps > 0 && global_step % self.every_n_steps == 0
    }
}

/// Training hyperparameters independent from the model architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTrainerConfig {
    /// Mini-batch size used by [`FoundationTrainer::train_epoch`].
    pub batch_size: usize,
    /// Base AdamW learning rate.
    pub learning_rate: f64,
    /// AdamW decoupled weight decay.
    pub weight_decay: f64,
    /// AdamW first-moment decay.
    pub adam_beta1: f64,
    /// AdamW second-moment decay.
    pub adam_beta2: f64,
    /// AdamW numerical stabilizer.
    pub adam_epsilon: f64,
    /// Optional global L2 gradient clipping threshold.
    pub max_gradient_norm: Option<f64>,
    /// Sampled weighted per-task gradient diagnostics.
    pub gradient_diagnostics: FoundationGradientDiagnosticsConfig,
    /// Per-step learning-rate schedule.
    pub learning_rate_schedule: FoundationLearningRateSchedule,
    /// Temperature used by the symmetric InfoNCE objective.
    pub contrastive_temperature: f64,
    /// Relative task-loss contributions.
    pub loss_weights: FoundationLossWeights,
    /// Label/corruption collation settings.
    pub collator: FoundationCollatorConfig,
    /// Train-partition-only scaling for continuous property targets. Resolved
    /// statistics are persisted in checkpoint trainer configuration.
    pub target_normalization: FoundationTargetNormalizationConfig,
    /// Large-corpus/source-aware sampling controls.
    pub sampling: FoundationSamplingConfig,
    /// Base seed for deterministic view corruption and epoch shuffling.
    pub seed: u64,
}

impl Default for FoundationTrainerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            max_gradient_norm: Some(1.0),
            gradient_diagnostics: FoundationGradientDiagnosticsConfig::default(),
            learning_rate_schedule: FoundationLearningRateSchedule::Constant,
            contrastive_temperature: 0.10,
            loss_weights: FoundationLossWeights::default(),
            collator: FoundationCollatorConfig::default(),
            target_normalization: FoundationTargetNormalizationConfig::default(),
            sampling: FoundationSamplingConfig::default(),
            seed: 20260831,
        }
    }
}

impl FoundationTrainerConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            candle_core::bail!("foundation batch_size must be greater than zero");
        }
        let _ = self.optimizer_config().validate()?;
        if let Some(max_norm) = self.max_gradient_norm {
            if !(max_norm > 0.0 && max_norm.is_finite()) {
                candle_core::bail!("foundation max_gradient_norm must be positive and finite");
            }
        }
        let _ = self.gradient_diagnostics.validate()?;
        self.learning_rate_schedule
            .validate()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if !(self.contrastive_temperature > 0.0 && self.contrastive_temperature.is_finite()) {
            candle_core::bail!("foundation contrastive_temperature must be positive and finite");
        }
        self.target_normalization.validate()?;
        self.sampling
            .validate()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Ok(())
    }

    fn optimizer_config(&self) -> FoundationAdamWConfig {
        FoundationAdamWConfig {
            learning_rate: self.learning_rate,
            beta1: self.adam_beta1,
            beta2: self.adam_beta2,
            epsilon: self.adam_epsilon,
            weight_decay: self.weight_decay,
        }
    }
}

/// Weighted per-task global gradient norms measured on one diagnostic step.
///
/// Each norm includes the task's configured loss weight and is measured before
/// global gradient clipping. Missing/inactive objectives remain `None`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationTaskGradientNorms {
    /// Weighted RT gradient norm.
    pub rt: Option<f64>,
    /// Weighted CCS gradient norm.
    pub ccs: Option<f64>,
    /// Weighted MS2 gradient norm.
    pub ms2: Option<f64>,
    /// Weighted masked-residue gradient norm.
    pub masked_residue: Option<f64>,
    /// Weighted chemistry-reconstruction gradient norm.
    pub chemistry: Option<f64>,
    /// Weighted contrastive gradient norm.
    pub contrastive: Option<f64>,
    /// Cosine alignment of the weighted RT gradient with the total update gradient.
    pub rt_cosine_to_total: Option<f64>,
    /// Cosine alignment of the weighted CCS gradient with the total update gradient.
    pub ccs_cosine_to_total: Option<f64>,
    /// Cosine alignment of the weighted MS2 gradient with the total update gradient.
    pub ms2_cosine_to_total: Option<f64>,
    /// Cosine alignment of the weighted masked-residue gradient with the total update gradient.
    pub masked_residue_cosine_to_total: Option<f64>,
    /// Cosine alignment of the weighted chemistry gradient with the total update gradient.
    pub chemistry_cosine_to_total: Option<f64>,
    /// Cosine alignment of the weighted contrastive gradient with the total update gradient.
    pub contrastive_cosine_to_total: Option<f64>,
}

/// Scalar diagnostics returned after one optimizer update.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationStepMetrics {
    /// Weighted total loss including contrastive alignment.
    pub total_loss: f32,
    /// Single-view RT loss in the regression space used for optimization.
    pub rt_loss: Option<f32>,
    /// RT mean absolute error in native target units.
    pub rt_mae_native: Option<f32>,
    /// RT root mean squared error in native target units.
    pub rt_rmse_native: Option<f32>,
    /// Single-view CCS loss in the regression space used for optimization.
    pub ccs_loss: Option<f32>,
    /// CCS mean absolute error in native target units.
    pub ccs_mae_native: Option<f32>,
    /// CCS root mean squared error in native target units.
    pub ccs_rmse_native: Option<f32>,
    /// Single-view MS2 loss.
    pub ms2_loss: Option<f32>,
    /// Masked residue-token reconstruction loss.
    pub masked_residue_loss: Option<f32>,
    /// Masked chemistry reconstruction loss.
    pub chemistry_loss: Option<f32>,
    /// Symmetric contrastive loss; absent for batch size one.
    pub contrastive_loss: Option<f32>,
    /// Learning rate used for this update.
    pub learning_rate: f64,
    /// Global L2 gradient norm before clipping.
    pub gradient_norm: f64,
    /// Multiplicative gradient scale applied by clipping.
    pub gradient_scale: f64,
    /// Optional weighted per-task gradient diagnostics measured on this step.
    pub task_gradient_norms: Option<FoundationTaskGradientNorms>,
}

/// Aggregate diagnostics for one pass through a record slice.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationEpochMetrics {
    /// Number of batches evaluated/optimized.
    pub steps: usize,
    /// Mean total loss across batches.
    pub mean_total_loss: f32,
    /// Mean RT loss across batches where RT labels were present, in normalized regression space.
    pub mean_rt_loss: Option<f32>,
    /// Mean per-batch RT MAE in native target units.
    pub mean_rt_mae_native: Option<f32>,
    /// Mean per-batch RT RMSE in native target units.
    pub mean_rt_rmse_native: Option<f32>,
    /// Mean CCS loss across batches where CCS labels were present, in normalized regression space.
    pub mean_ccs_loss: Option<f32>,
    /// Mean per-batch CCS MAE in native target units.
    pub mean_ccs_mae_native: Option<f32>,
    /// Mean per-batch CCS RMSE in native target units.
    pub mean_ccs_rmse_native: Option<f32>,
    /// Mean MS2 loss across batches where fragment labels were present.
    pub mean_ms2_loss: Option<f32>,
    /// Mean masked-residue reconstruction loss when active.
    pub mean_masked_residue_loss: Option<f32>,
    /// Mean masked-chemistry reconstruction loss when active.
    pub mean_chemistry_loss: Option<f32>,
    /// Mean symmetric contrastive loss when batch size permitted it.
    pub mean_contrastive_loss: Option<f32>,
    /// Mean pre-clipping gradient norm for training epochs.
    pub mean_gradient_norm: Option<f64>,
    /// Mean multiplicative gradient scale after global-norm clipping.
    pub mean_gradient_scale: Option<f64>,
    /// Number of optimizer steps where global-norm clipping was active.
    pub clipped_steps: usize,
    /// Fraction of optimizer steps where global-norm clipping was active.
    pub clipped_fraction: Option<f64>,
    /// Number of steps where per-task gradient diagnostics were measured.
    pub gradient_diagnostic_steps: usize,
    /// Mean weighted RT gradient norm across diagnostic steps.
    pub mean_rt_gradient_norm: Option<f64>,
    /// Mean weighted CCS gradient norm across diagnostic steps.
    pub mean_ccs_gradient_norm: Option<f64>,
    /// Mean weighted MS2 gradient norm across diagnostic steps.
    pub mean_ms2_gradient_norm: Option<f64>,
    /// Mean weighted masked-residue gradient norm across diagnostic steps.
    pub mean_masked_residue_gradient_norm: Option<f64>,
    /// Mean weighted chemistry gradient norm across diagnostic steps.
    pub mean_chemistry_gradient_norm: Option<f64>,
    /// Mean weighted contrastive gradient norm across diagnostic steps.
    pub mean_contrastive_gradient_norm: Option<f64>,
    /// Mean RT-gradient cosine with the combined multi-task update direction.
    pub mean_rt_gradient_cosine_to_total: Option<f64>,
    /// Mean CCS-gradient cosine with the combined multi-task update direction.
    pub mean_ccs_gradient_cosine_to_total: Option<f64>,
    /// Mean MS2-gradient cosine with the combined multi-task update direction.
    pub mean_ms2_gradient_cosine_to_total: Option<f64>,
    /// Mean masked-residue-gradient cosine with the combined update direction.
    pub mean_masked_residue_gradient_cosine_to_total: Option<f64>,
    /// Mean chemistry-gradient cosine with the combined update direction.
    pub mean_chemistry_gradient_cosine_to_total: Option<f64>,
    /// Mean contrastive-gradient cosine with the combined update direction.
    pub mean_contrastive_gradient_cosine_to_total: Option<f64>,
    /// Final learning rate used during the epoch.
    pub final_learning_rate: Option<f64>,
}

/// One completed fit epoch.
#[derive(Debug, Clone, Copy)]
pub struct FoundationFitEpochMetrics {
    /// Zero-based epoch index.
    pub epoch: u64,
    /// Training metrics.
    pub train: FoundationEpochMetrics,
    /// Validation metrics.
    pub validation: FoundationEpochMetrics,
    /// Whether this epoch established a new best validation loss.
    pub improved: bool,
}

/// Summary returned by [`FoundationTrainer::fit`].
#[derive(Debug, Clone)]
pub struct FoundationFitSummary {
    /// Epoch metrics in completion order.
    pub epochs: Vec<FoundationFitEpochMetrics>,
    /// Final progress state.
    pub progress: FoundationTrainingProgress,
    /// True when early stopping ended the run before `max_epochs`.
    pub stopped_early: bool,
}

/// Stateful foundation trainer with model, optimizer, collator, and checkpoint state.
pub struct FoundationTrainer {
    wrapper: FoundationModelWrapper,
    optimizer: FoundationAdamW,
    collator: FoundationCollator,
    config: FoundationTrainerConfig,
    global_step: u64,
}

impl FoundationTrainer {
    /// Create a randomly initialized trainer.
    pub fn new(
        model_config: FoundationConfig,
        config: FoundationTrainerConfig,
        device: Device,
    ) -> Result<Self> {
        config.validate()?;
        let wrapper = FoundationModelWrapper::new(model_config.clone(), device)?;
        let optimizer = FoundationAdamW::new(wrapper.varmap(), config.optimizer_config())?;
        let collator = FoundationCollator::new(model_config, config.collator.clone())?;
        Ok(Self {
            wrapper,
            optimizer,
            collator,
            config,
            global_step: 0,
        })
    }

    /// Load only model weights from a SafeTensors file.
    ///
    /// This intentionally does not claim to be an exact training resume. Use
    /// [`Self::from_checkpoint`] when optimizer moments/counters matter.
    pub fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.wrapper.varmap_mut().load(path)
    }

    /// Save model weights in SafeTensors format.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.wrapper.save_safetensors(path)
    }

    /// Save model weights, named AdamW moments, and YAML trainer state.
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        directory: P,
        progress: FoundationTrainingProgress,
        provenance: FoundationCheckpointProvenance,
    ) -> AnyResult<FoundationCheckpointMetadata> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)
            .with_context(|| format!("failed to create foundation checkpoint {directory:?}"))?;
        let (model_path, optimizer_path, state_path) = foundation_checkpoint_paths(directory);
        self.wrapper
            .save_safetensors(&model_path)
            .with_context(|| format!("failed to save model checkpoint {model_path:?}"))?;
        self.optimizer
            .save_safetensors(&optimizer_path)
            .with_context(|| format!("failed to save optimizer checkpoint {optimizer_path:?}"))?;
        let metadata = FoundationCheckpointMetadata::new(
            self.wrapper.config().clone(),
            self.config.clone(),
            self.global_step,
            self.optimizer.step_count(),
            progress,
            provenance,
        );
        metadata.validate()?;
        metadata.write_yaml(&state_path)?;
        Ok(metadata)
    }

    /// Reconstruct a trainer and restore model weights plus exact AdamW state.
    pub fn from_checkpoint<P: AsRef<Path>>(
        directory: P,
        device: Device,
    ) -> AnyResult<(Self, FoundationCheckpointMetadata)> {
        let directory = directory.as_ref();
        let (model_path, optimizer_path, state_path) = foundation_checkpoint_paths(directory);
        let metadata = FoundationCheckpointMetadata::read_yaml(&state_path)?;
        let mut trainer = Self::new(
            metadata.model_config.clone(),
            metadata.trainer_config.clone(),
            device,
        )?;
        trainer
            .wrapper
            .varmap_mut()
            .load(&model_path)
            .with_context(|| format!("failed to restore model checkpoint {model_path:?}"))?;
        trainer
            .optimizer
            .load_safetensors(&optimizer_path)
            .with_context(|| format!("failed to restore optimizer checkpoint {optimizer_path:?}"))?;
        trainer.optimizer.set_step_count(metadata.optimizer_step);
        trainer.global_step = metadata.global_step;
        Ok((trainer, metadata))
    }

    /// Execute one two-view multi-task optimizer update.
    pub fn train_step(
        &mut self,
        records: &[FoundationTrainingRecord],
    ) -> Result<FoundationStepMetrics> {
        if records.is_empty() {
            candle_core::bail!("cannot train a foundation step with zero records");
        }
        let scheduled_lr = self
            .config
            .learning_rate_schedule
            .learning_rate(self.config.learning_rate, self.global_step)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.optimizer.set_learning_rate(scheduled_lr)?;

        let seed = self.config.seed.wrapping_add(self.global_step);
        let mut views = self
            .collator
            .collate_views(records, self.wrapper.device(), seed)?;
        self.normalize_regression_targets(&mut views)?;
        let (total, losses, contrastive, regression) = self.loss_for_views(&views, true)?;
        let diagnostic_step = self
            .config
            .gradient_diagnostics
            .should_measure(self.global_step);
        let (task_gradient_norms, optimizer_metrics) = if diagnostic_step {
            // Reuse the combined gradient store for the optimizer update so
            // alignment diagnostics do not add a redundant total-loss backward pass.
            let total_gradients = total.backward()?;
            let diagnostics = self.measure_task_gradient_norms(
                &losses,
                contrastive.as_ref(),
                &total_gradients,
            )?;
            let optimizer_metrics = self
                .optimizer
                .step(&total_gradients, self.config.max_gradient_norm)?;
            (Some(diagnostics), optimizer_metrics)
        } else {
            (
                None,
                self.optimizer
                    .backward_step(&total, self.config.max_gradient_norm)?,
            )
        };
        self.global_step = optimizer_metrics.step;
        metrics_from_tensors(
            &total,
            &losses,
            contrastive.as_ref(),
            regression,
            Some((
                optimizer_metrics.learning_rate,
                optimizer_metrics.gradient_norm,
                optimizer_metrics.gradient_scale,
            )),
            task_gradient_norms,
        )
    }

    /// Train sequential mini-batches for one epoch.
    pub fn train_epoch(
        &mut self,
        records: &[FoundationTrainingRecord],
    ) -> Result<FoundationEpochMetrics> {
        if records.is_empty() {
            candle_core::bail!("cannot train a foundation epoch with zero records");
        }
        let mut accumulator = EpochAccumulator::default();
        for batch in records.chunks(self.config.batch_size) {
            accumulator.push(self.train_step(batch)?);
        }
        Ok(accumulator.finish())
    }

    /// Train one deterministic shuffled epoch.
    ///
    /// Only one mini-batch is cloned at a time, so shuffled training does not
    /// duplicate the full corpus in memory.
    pub fn train_epoch_shuffled(
        &mut self,
        records: &[FoundationTrainingRecord],
        epoch: u64,
    ) -> Result<FoundationEpochMetrics> {
        if records.is_empty() {
            candle_core::bail!("cannot train a foundation epoch with zero records");
        }
        let mut indices: Vec<usize> = (0..records.len()).collect();
        let mut rng = FoundationSplitMix64::new(
            self.config
                .seed
                .wrapping_add(0x5348_5546_464c_4500)
                .wrapping_add(epoch),
        );
        rng.shuffle(&mut indices);
        let mut accumulator = EpochAccumulator::default();
        for batch_indices in indices.chunks(self.config.batch_size) {
            let batch: Vec<FoundationTrainingRecord> = batch_indices
                .iter()
                .map(|index| records[*index].clone())
                .collect();
            accumulator.push(self.train_step(&batch)?);
        }
        Ok(accumulator.finish())
    }

    /// Train one epoch from a subset of record indices without materializing a
    /// second full corpus. Only the current mini-batch is cloned.
    pub fn train_epoch_indices(
        &mut self,
        records: &[FoundationTrainingRecord],
        indices: &[usize],
        epoch: u64,
        shuffle: bool,
    ) -> Result<FoundationEpochMetrics> {
        if indices.is_empty() {
            candle_core::bail!("cannot train a foundation epoch with zero selected records");
        }
        let mut order = indices.to_vec();
        if shuffle {
            let mut rng = FoundationSplitMix64::new(
                self.config
                    .seed
                    .wrapping_add(0x494e_4445_5845_5300)
                    .wrapping_add(epoch),
            );
            rng.shuffle(&mut order);
        }
        let mut accumulator = EpochAccumulator::default();
        for batch_indices in order.chunks(self.config.batch_size) {
            let mut batch = Vec::with_capacity(batch_indices.len());
            for &index in batch_indices {
                let record = records.get(index).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "foundation training index {index} is out of bounds for {} records",
                        records.len()
                    ))
                })?;
                batch.push(record.clone());
            }
            accumulator.push(self.train_step(&batch)?);
        }
        Ok(accumulator.finish())
    }

    /// Evaluate a materialized validation/test index subset without cloning the
    /// complete partition. Validation never changes optimizer/global-step state.
    pub fn evaluate_epoch_indices(
        &self,
        records: &[FoundationTrainingRecord],
        indices: &[usize],
    ) -> Result<FoundationEpochMetrics> {
        if indices.is_empty() {
            candle_core::bail!("cannot evaluate a foundation epoch with zero selected records");
        }
        let mut accumulator = EpochAccumulator::default();
        for (batch_index, batch_indices) in indices.chunks(self.config.batch_size).enumerate() {
            let mut batch = Vec::with_capacity(batch_indices.len());
            for &index in batch_indices {
                let record = records.get(index).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "foundation evaluation index {index} is out of bounds for {} records",
                        records.len()
                    ))
                })?;
                batch.push(record.clone());
            }
            let seed = self
                .config
                .seed
                .wrapping_add(0x4556_494e_4445_5800)
                .wrapping_add(batch_index as u64);
            let mut views = self
                .collator
                .collate_views(&batch, self.wrapper.device(), seed)?;
            self.normalize_regression_targets(&mut views)?;
            let (total, losses, contrastive, regression) = self.loss_for_views(&views, false)?;
            accumulator.push(metrics_from_tensors(
                &total,
                &losses,
                contrastive.as_ref(),
                regression,
                None,
                None,
            )?);
        }
        Ok(accumulator.finish())
    }

    /// Evaluate deterministic corrupted views without updating model weights,
    /// optimizer moments, or global step.
    pub fn evaluate_epoch(
        &self,
        records: &[FoundationTrainingRecord],
    ) -> Result<FoundationEpochMetrics> {
        if records.is_empty() {
            candle_core::bail!("cannot evaluate a foundation epoch with zero records");
        }
        let mut accumulator = EpochAccumulator::default();
        for (batch_index, batch) in records.chunks(self.config.batch_size).enumerate() {
            let seed = self
                .config
                .seed
                .wrapping_add(0x4556_414c_0000_0000)
                .wrapping_add(batch_index as u64);
            let mut views = self
                .collator
                .collate_views(batch, self.wrapper.device(), seed)?;
            self.normalize_regression_targets(&mut views)?;
            let (total, losses, contrastive, regression) = self.loss_for_views(&views, false)?;
            accumulator.push(metrics_from_tensors(
                &total,
                &losses,
                contrastive.as_ref(),
                regression,
                None,
                None,
            )?);
        }
        Ok(accumulator.finish())
    }

    /// Train/evaluate until `max_epochs` or early stopping, writing exact
    /// `latest` and best-validation checkpoints under `checkpoint_root`.
    pub fn fit<P: AsRef<Path>>(
        &mut self,
        train_records: &[FoundationTrainingRecord],
        validation_records: &[FoundationTrainingRecord],
        fit_config: FoundationFitConfig,
        checkpoint_root: P,
        provenance: FoundationCheckpointProvenance,
        mut progress: FoundationTrainingProgress,
    ) -> AnyResult<FoundationFitSummary> {
        let fit_config = fit_config.validate()?;
        if train_records.is_empty() || validation_records.is_empty() {
            anyhow::bail!("foundation fit requires non-empty train and validation records");
        }
        let checkpoint_root = checkpoint_root.as_ref();
        fs::create_dir_all(checkpoint_root)?;
        let mut epochs = Vec::new();
        let mut stopped_early = false;

        while progress.completed_epochs < fit_config.max_epochs {
            let epoch = progress.completed_epochs;
            let train = if fit_config.shuffle_each_epoch {
                self.train_epoch_shuffled(train_records, epoch)?
            } else {
                self.train_epoch(train_records)?
            };
            let validation = self.evaluate_epoch(validation_records)?;
            let improved = progress
                .best_validation_loss
                .map(|best| validation.mean_total_loss < best - fit_config.early_stopping_min_delta)
                .unwrap_or(true);

            progress.completed_epochs = progress.completed_epochs.saturating_add(1);
            if improved {
                progress.best_validation_loss = Some(validation.mean_total_loss);
                progress.best_epoch = Some(epoch);
                progress.epochs_without_improvement = 0;
            } else {
                progress.epochs_without_improvement =
                    progress.epochs_without_improvement.saturating_add(1);
            }

            self.save_checkpoint(
                checkpoint_root.join("latest"),
                progress.clone(),
                provenance.clone(),
            )?;
            if improved {
                self.save_checkpoint(
                    checkpoint_root.join("best"),
                    progress.clone(),
                    provenance.clone(),
                )?;
            }

            epochs.push(FoundationFitEpochMetrics {
                epoch,
                train,
                validation,
                improved,
            });

            if fit_config
                .early_stopping_patience
                .is_some_and(|patience| progress.epochs_without_improvement >= patience)
            {
                stopped_early = true;
                break;
            }
        }

        Ok(FoundationFitSummary {
            epochs,
            progress,
            stopped_early,
        })
    }

    /// Fit directly from materialized train/validation record indices. This is
    /// the preferred large-corpus API because it keeps one copy of the corpus.
    pub fn fit_indices<P: AsRef<Path>>(
        &mut self,
        records: &[FoundationTrainingRecord],
        train_indices: &[usize],
        validation_indices: &[usize],
        fit_config: FoundationFitConfig,
        checkpoint_root: P,
        provenance: FoundationCheckpointProvenance,
        mut progress: FoundationTrainingProgress,
    ) -> AnyResult<FoundationFitSummary> {
        let fit_config = fit_config.validate()?;
        if train_indices.is_empty() || validation_indices.is_empty() {
            anyhow::bail!("foundation fit requires non-empty train and validation indices");
        }
        let checkpoint_root = checkpoint_root.as_ref();
        fs::create_dir_all(checkpoint_root)?;
        let mut epochs = Vec::new();
        let mut stopped_early = false;

        while progress.completed_epochs < fit_config.max_epochs {
            let epoch = progress.completed_epochs;
            let train = self.train_epoch_indices(
                records,
                train_indices,
                epoch,
                fit_config.shuffle_each_epoch,
            )?;
            let validation = self.evaluate_epoch_indices(records, validation_indices)?;
            let improved = progress
                .best_validation_loss
                .map(|best| validation.mean_total_loss < best - fit_config.early_stopping_min_delta)
                .unwrap_or(true);

            progress.completed_epochs = progress.completed_epochs.saturating_add(1);
            if improved {
                progress.best_validation_loss = Some(validation.mean_total_loss);
                progress.best_epoch = Some(epoch);
                progress.epochs_without_improvement = 0;
            } else {
                progress.epochs_without_improvement =
                    progress.epochs_without_improvement.saturating_add(1);
            }

            self.save_checkpoint(
                checkpoint_root.join("latest"),
                progress.clone(),
                provenance.clone(),
            )?;
            if improved {
                self.save_checkpoint(
                    checkpoint_root.join("best"),
                    progress.clone(),
                    provenance.clone(),
                )?;
            }

            epochs.push(FoundationFitEpochMetrics {
                epoch,
                train,
                validation,
                improved,
            });
            if fit_config
                .early_stopping_patience
                .is_some_and(|patience| progress.epochs_without_improvement >= patience)
            {
                stopped_early = true;
                break;
            }
        }

        Ok(FoundationFitSummary {
            epochs,
            progress,
            stopped_early,
        })
    }

    /// Fit a combined corpus using deterministic bounded/source-aware sampling.
    ///
    /// This is the preferred production multi-source API. The materialized
    /// benchmark remains the authority for which records belong to train and
    /// validation; the sampling configuration only selects/reweights records
    /// *within* those partitions. Validation subsampling, when enabled, is
    /// fixed across epochs.
    pub fn fit_corpus_indices<P: AsRef<Path>>(
        &mut self,
        records: &[FoundationTrainingRecord],
        record_provenance: &[FoundationRecordProvenance],
        train_indices: &[usize],
        validation_indices: &[usize],
        fit_config: FoundationFitConfig,
        checkpoint_root: P,
        provenance: FoundationCheckpointProvenance,
        mut progress: FoundationTrainingProgress,
    ) -> AnyResult<FoundationFitSummary> {
        let fit_config = fit_config.validate()?;
        if train_indices.is_empty() || validation_indices.is_empty() {
            anyhow::bail!("foundation fit requires non-empty train and validation indices");
        }
        if records.len() != record_provenance.len() {
            anyhow::bail!(
                "foundation corpus record/provenance lengths differ: {} records, {} provenance entries",
                records.len(),
                record_provenance.len()
            );
        }
        let checkpoint_root = checkpoint_root.as_ref();
        fs::create_dir_all(checkpoint_root)?;
        let validation_plan = sample_foundation_validation_indices(
            records,
            record_provenance,
            validation_indices,
            self.config.batch_size,
            self.config.seed,
            &self.config.sampling,
        )?;
        let mut epochs = Vec::new();
        let mut stopped_early = false;

        while progress.completed_epochs < fit_config.max_epochs {
            let epoch = progress.completed_epochs;
            let train_plan = sample_foundation_training_indices(
                records,
                record_provenance,
                train_indices,
                self.config.batch_size,
                epoch,
                self.config.seed,
                fit_config.shuffle_each_epoch,
                &self.config.sampling,
            )?;
            let train = self.train_epoch_indices(
                records,
                &train_plan.indices,
                epoch,
                false,
            )?;
            let validation =
                self.evaluate_epoch_indices(records, &validation_plan.indices)?;
            let improved = progress
                .best_validation_loss
                .map(|best| validation.mean_total_loss < best - fit_config.early_stopping_min_delta)
                .unwrap_or(true);

            progress.completed_epochs = progress.completed_epochs.saturating_add(1);
            if improved {
                progress.best_validation_loss = Some(validation.mean_total_loss);
                progress.best_epoch = Some(epoch);
                progress.epochs_without_improvement = 0;
            } else {
                progress.epochs_without_improvement =
                    progress.epochs_without_improvement.saturating_add(1);
            }

            self.save_checkpoint(
                checkpoint_root.join("latest"),
                progress.clone(),
                provenance.clone(),
            )?;
            if improved {
                self.save_checkpoint(
                    checkpoint_root.join("best"),
                    progress.clone(),
                    provenance.clone(),
                )?;
            }

            epochs.push(FoundationFitEpochMetrics {
                epoch,
                train,
                validation,
                improved,
            });
            if fit_config
                .early_stopping_patience
                .is_some_and(|patience| progress.epochs_without_improvement >= patience)
            {
                stopped_early = true;
                break;
            }
        }

        Ok(FoundationFitSummary {
            epochs,
            progress,
            stopped_early,
        })
    }

    /// Predict property heads and return continuous regression outputs in their
    /// native target units. The underlying model head itself operates in the
    /// standardized training space when normalization is active.
    pub fn predict_native(
        &self,
        peptides: &[PeptidoformInput],
        context: &[TrainingContext],
    ) -> Result<FoundationMultiTaskOutput> {
        let mut output = self.wrapper.predict(peptides, context)?;
        output.rt = self.config.target_normalization.rt.denormalize_tensor(&output.rt)?;
        output.ccs = self.config.target_normalization.ccs.denormalize_tensor(&output.ccs)?;
        Ok(output)
    }

    /// Predict native-unit properties when acquisition context is completely
    /// unavailable.
    pub fn predict_unknown_context_native(
        &self,
        peptides: &[PeptidoformInput],
    ) -> Result<FoundationMultiTaskOutput> {
        let mut output = self.wrapper.predict_unknown_context(peptides)?;
        output.rt = self.config.target_normalization.rt.denormalize_tensor(&output.rt)?;
        output.ccs = self.config.target_normalization.ccs.denormalize_tensor(&output.ccs)?;
        Ok(output)
    }

    /// Access the high-level model wrapper for embedding/evaluation/checkpointing.
    pub fn model(&self) -> &FoundationModelWrapper {
        &self.wrapper
    }

    /// Trainer configuration used for this run.
    pub fn config(&self) -> &FoundationTrainerConfig {
        &self.config
    }

    /// Number of completed optimizer steps.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Current AdamW moment update count.
    pub fn optimizer_step(&self) -> u64 {
        self.optimizer.step_count()
    }

    fn normalize_regression_targets(&self, views: &mut FoundationTrainingViews) -> Result<()> {
        for batch in [&mut views.first, &mut views.second] {
            if let Some(rt) = batch.targets.rt.take() {
                batch.targets.rt = Some(self.config.target_normalization.rt.normalize_tensor(&rt)?);
            }
            if let Some(ccs) = batch.targets.ccs.take() {
                batch.targets.ccs = Some(self.config.target_normalization.ccs.normalize_tensor(&ccs)?);
            }
        }
        Ok(())
    }

    fn measure_task_gradient_norms(
        &self,
        losses: &FoundationLosses,
        contrastive: Option<&Tensor>,
        total_gradients: &GradStore,
    ) -> Result<FoundationTaskGradientNorms> {
        let weights = self.config.loss_weights;
        let (rt, rt_cosine_to_total) = self.weighted_gradient_diagnostics(
            losses.rt.as_ref(),
            weights.rt,
            total_gradients,
        )?;
        let (ccs, ccs_cosine_to_total) = self.weighted_gradient_diagnostics(
            losses.ccs.as_ref(),
            weights.ccs,
            total_gradients,
        )?;
        let (ms2, ms2_cosine_to_total) = self.weighted_gradient_diagnostics(
            losses.ms2.as_ref(),
            weights.ms2,
            total_gradients,
        )?;
        let (masked_residue, masked_residue_cosine_to_total) = self
            .weighted_gradient_diagnostics(
                losses.masked_residue.as_ref(),
                weights.masked_residue,
                total_gradients,
            )?;
        let (chemistry, chemistry_cosine_to_total) = self.weighted_gradient_diagnostics(
            losses.chemistry.as_ref(),
            weights.chemistry,
            total_gradients,
        )?;
        let (contrastive, contrastive_cosine_to_total) = self.weighted_gradient_diagnostics(
            contrastive,
            weights.contrastive,
            total_gradients,
        )?;
        Ok(FoundationTaskGradientNorms {
            rt,
            ccs,
            ms2,
            masked_residue,
            chemistry,
            contrastive,
            rt_cosine_to_total,
            ccs_cosine_to_total,
            ms2_cosine_to_total,
            masked_residue_cosine_to_total,
            chemistry_cosine_to_total,
            contrastive_cosine_to_total,
        })
    }

    fn weighted_gradient_diagnostics(
        &self,
        loss: Option<&Tensor>,
        weight: f64,
        total_gradients: &GradStore,
    ) -> Result<(Option<f64>, Option<f64>)> {
        let Some(loss) = loss else {
            return Ok((None, None));
        };
        if weight == 0.0 {
            return Ok((None, None));
        }
        let weighted = loss.affine(weight, 0.0)?;
        let gradients = weighted.backward()?;
        let norm = self.optimizer.gradient_norm(&gradients)?;
        let cosine = self
            .optimizer
            .gradient_cosine(&gradients, total_gradients)?;
        Ok((Some(norm), cosine))
    }

    fn loss_for_views(
        &self,
        views: &FoundationTrainingViews,
        train: bool,
    ) -> Result<(Tensor, FoundationLosses, Option<Tensor>, RegressionDiagnostics)> {
        let first = self
            .wrapper
            .model()
            .forward_t(&views.first.input, &views.first.context, train)?;
        let second = self
            .wrapper
            .model()
            .forward_t(&views.second.input, &views.second.context, train)?;
        let regression = RegressionDiagnostics {
            rt: regression_native_metrics(
                &first.rt,
                views.first.targets.rt.as_ref(),
                views.first.targets.rt_mask.as_ref(),
                &self.config.target_normalization.rt,
            )?,
            ccs: regression_native_metrics(
                &first.ccs,
                views.first.targets.ccs.as_ref(),
                views.first.targets.ccs_mask.as_ref(),
                &self.config.target_normalization.ccs,
            )?,
        };
        let losses = multi_task_loss(&first, &views.first.targets, self.config.loss_weights)?;
        let contrastive = if views.first.input.residue_ids.dims2()?.0 > 1
            && self.config.loss_weights.contrastive > 0.0
        {
            Some(contrastive_info_nce_loss(
                &first.contrastive_projection,
                &second.contrastive_projection,
                self.config.contrastive_temperature,
            )?)
        } else {
            None
        };
        let total = if let Some(contrastive) = &contrastive {
            (losses.total.clone()
                + contrastive.affine(self.config.loss_weights.contrastive, 0.0)?)?
        } else {
            losses.total.clone()
        };
        Ok((total, losses, contrastive, regression))
    }
}

#[derive(Default)]
struct RegressionDiagnostics {
    rt: Option<(Tensor, Tensor)>,
    ccs: Option<(Tensor, Tensor)>,
}

fn regression_native_metrics(
    prediction: &Tensor,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    normalization: &FoundationRegressionNormalization,
) -> Result<Option<(Tensor, Tensor)>> {
    let (Some(target), Some(mask)) = (target, mask) else {
        if target.is_some() || mask.is_some() {
            candle_core::bail!("foundation regression target and mask must be supplied together");
        }
        return Ok(None);
    };
    let prediction = normalization.denormalize_tensor(prediction)?;
    let target = normalization.denormalize_tensor(target)?;
    let mask = mask.broadcast_as(prediction.dims())?;
    let difference = (&prediction - &target)?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    let mae = difference
        .sqr()?
        .sqrt()?
        .broadcast_mul(&mask)?
        .sum_all()?
        .broadcast_div(&denominator)?;
    let rmse = difference
        .sqr()?
        .broadcast_mul(&mask)?
        .sum_all()?
        .broadcast_div(&denominator)?
        .sqrt()?;
    Ok(Some((mae, rmse)))
}

#[derive(Default)]
struct EpochAccumulator {
    steps: usize,
    total: f64,
    rt: OptionalMean,
    rt_mae_native: OptionalMean,
    rt_rmse_native: OptionalMean,
    ccs: OptionalMean,
    ccs_mae_native: OptionalMean,
    ccs_rmse_native: OptionalMean,
    ms2: OptionalMean,
    masked_residue: OptionalMean,
    chemistry: OptionalMean,
    contrastive: OptionalMean,
    gradient_norm: OptionalMean64,
    gradient_scale: OptionalMean64,
    clipped_steps: usize,
    gradient_diagnostic_steps: usize,
    rt_gradient_norm: OptionalMean64,
    ccs_gradient_norm: OptionalMean64,
    ms2_gradient_norm: OptionalMean64,
    masked_residue_gradient_norm: OptionalMean64,
    chemistry_gradient_norm: OptionalMean64,
    contrastive_gradient_norm: OptionalMean64,
    rt_gradient_cosine_to_total: OptionalMean64,
    ccs_gradient_cosine_to_total: OptionalMean64,
    ms2_gradient_cosine_to_total: OptionalMean64,
    masked_residue_gradient_cosine_to_total: OptionalMean64,
    chemistry_gradient_cosine_to_total: OptionalMean64,
    contrastive_gradient_cosine_to_total: OptionalMean64,
    final_learning_rate: Option<f64>,
}

impl EpochAccumulator {
    fn push(&mut self, metrics: FoundationStepMetrics) {
        self.steps += 1;
        self.total += f64::from(metrics.total_loss);
        self.rt.push(metrics.rt_loss);
        self.rt_mae_native.push(metrics.rt_mae_native);
        self.rt_rmse_native.push(metrics.rt_rmse_native);
        self.ccs.push(metrics.ccs_loss);
        self.ccs_mae_native.push(metrics.ccs_mae_native);
        self.ccs_rmse_native.push(metrics.ccs_rmse_native);
        self.ms2.push(metrics.ms2_loss);
        self.masked_residue.push(metrics.masked_residue_loss);
        self.chemistry.push(metrics.chemistry_loss);
        self.contrastive.push(metrics.contrastive_loss);
        if metrics.learning_rate > 0.0 {
            self.gradient_norm.push(Some(metrics.gradient_norm));
            self.gradient_scale.push(Some(metrics.gradient_scale));
            if metrics.gradient_scale < 1.0 - 1e-12 {
                self.clipped_steps = self.clipped_steps.saturating_add(1);
            }
            self.final_learning_rate = Some(metrics.learning_rate);
        }
        if let Some(task) = metrics.task_gradient_norms {
            self.gradient_diagnostic_steps = self.gradient_diagnostic_steps.saturating_add(1);
            self.rt_gradient_norm.push(task.rt);
            self.ccs_gradient_norm.push(task.ccs);
            self.ms2_gradient_norm.push(task.ms2);
            self.masked_residue_gradient_norm.push(task.masked_residue);
            self.chemistry_gradient_norm.push(task.chemistry);
            self.contrastive_gradient_norm.push(task.contrastive);
            self.rt_gradient_cosine_to_total.push(task.rt_cosine_to_total);
            self.ccs_gradient_cosine_to_total
                .push(task.ccs_cosine_to_total);
            self.ms2_gradient_cosine_to_total
                .push(task.ms2_cosine_to_total);
            self.masked_residue_gradient_cosine_to_total
                .push(task.masked_residue_cosine_to_total);
            self.chemistry_gradient_cosine_to_total
                .push(task.chemistry_cosine_to_total);
            self.contrastive_gradient_cosine_to_total
                .push(task.contrastive_cosine_to_total);
        }
    }

    fn finish(self) -> FoundationEpochMetrics {
        FoundationEpochMetrics {
            steps: self.steps,
            mean_total_loss: if self.steps == 0 {
                0.0
            } else {
                (self.total / self.steps as f64) as f32
            },
            mean_rt_loss: self.rt.mean(),
            mean_rt_mae_native: self.rt_mae_native.mean(),
            mean_rt_rmse_native: self.rt_rmse_native.mean(),
            mean_ccs_loss: self.ccs.mean(),
            mean_ccs_mae_native: self.ccs_mae_native.mean(),
            mean_ccs_rmse_native: self.ccs_rmse_native.mean(),
            mean_ms2_loss: self.ms2.mean(),
            mean_masked_residue_loss: self.masked_residue.mean(),
            mean_chemistry_loss: self.chemistry.mean(),
            mean_contrastive_loss: self.contrastive.mean(),
            mean_gradient_norm: self.gradient_norm.mean(),
            mean_gradient_scale: self.gradient_scale.mean(),
            clipped_steps: self.clipped_steps,
            clipped_fraction: (self.gradient_scale.count > 0)
                .then(|| self.clipped_steps as f64 / self.gradient_scale.count as f64),
            gradient_diagnostic_steps: self.gradient_diagnostic_steps,
            mean_rt_gradient_norm: self.rt_gradient_norm.mean(),
            mean_ccs_gradient_norm: self.ccs_gradient_norm.mean(),
            mean_ms2_gradient_norm: self.ms2_gradient_norm.mean(),
            mean_masked_residue_gradient_norm: self.masked_residue_gradient_norm.mean(),
            mean_chemistry_gradient_norm: self.chemistry_gradient_norm.mean(),
            mean_contrastive_gradient_norm: self.contrastive_gradient_norm.mean(),
            mean_rt_gradient_cosine_to_total: self.rt_gradient_cosine_to_total.mean(),
            mean_ccs_gradient_cosine_to_total: self.ccs_gradient_cosine_to_total.mean(),
            mean_ms2_gradient_cosine_to_total: self.ms2_gradient_cosine_to_total.mean(),
            mean_masked_residue_gradient_cosine_to_total: self
                .masked_residue_gradient_cosine_to_total
                .mean(),
            mean_chemistry_gradient_cosine_to_total: self
                .chemistry_gradient_cosine_to_total
                .mean(),
            mean_contrastive_gradient_cosine_to_total: self
                .contrastive_gradient_cosine_to_total
                .mean(),
            final_learning_rate: self.final_learning_rate,
        }
    }
}

#[derive(Default)]
struct OptionalMean {
    sum: f64,
    count: usize,
}

impl OptionalMean {
    fn push(&mut self, value: Option<f32>) {
        if let Some(value) = value {
            self.sum += f64::from(value);
            self.count += 1;
        }
    }

    fn mean(&self) -> Option<f32> {
        (self.count > 0).then(|| (self.sum / self.count as f64) as f32)
    }
}

#[derive(Default)]
struct OptionalMean64 {
    sum: f64,
    count: usize,
}

impl OptionalMean64 {
    fn push(&mut self, value: Option<f64>) {
        if let Some(value) = value {
            self.sum += value;
            self.count += 1;
        }
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }
}

fn metrics_from_tensors(
    total: &Tensor,
    losses: &FoundationLosses,
    contrastive: Option<&Tensor>,
    regression: RegressionDiagnostics,
    optimizer: Option<(f64, f64, f64)>,
    task_gradient_norms: Option<FoundationTaskGradientNorms>,
) -> Result<FoundationStepMetrics> {
    let (learning_rate, gradient_norm, gradient_scale) = optimizer.unwrap_or((0.0, 0.0, 1.0));
    Ok(FoundationStepMetrics {
        total_loss: total.to_scalar::<f32>()?,
        rt_loss: scalar_option(losses.rt.as_ref())?,
        rt_mae_native: scalar_pair_option(regression.rt.as_ref(), 0)?,
        rt_rmse_native: scalar_pair_option(regression.rt.as_ref(), 1)?,
        ccs_loss: scalar_option(losses.ccs.as_ref())?,
        ccs_mae_native: scalar_pair_option(regression.ccs.as_ref(), 0)?,
        ccs_rmse_native: scalar_pair_option(regression.ccs.as_ref(), 1)?,
        ms2_loss: scalar_option(losses.ms2.as_ref())?,
        masked_residue_loss: scalar_option(losses.masked_residue.as_ref())?,
        chemistry_loss: scalar_option(losses.chemistry.as_ref())?,
        contrastive_loss: scalar_option(contrastive)?,
        learning_rate,
        gradient_norm,
        gradient_scale,
        task_gradient_norms,
    })
}

fn scalar_option(value: Option<&Tensor>) -> Result<Option<f32>> {
    value.map(|value| value.to_scalar::<f32>()).transpose()
}

fn scalar_pair_option(value: Option<&(Tensor, Tensor)>, index: usize) -> Result<Option<f32>> {
    value
        .map(|pair| if index == 0 { &pair.0 } else { &pair.1 })
        .map(|value| value.to_scalar::<f32>())
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationCorruptionConfig, FragmentTarget, PeptidoformInput, RetentionTimeLabels,
        TrainingContext,
    };

    #[test]
    fn one_training_step_backpropagates_all_available_objectives() {
        let model_config = FoundationConfig {
            max_sequence_len: 12,
            transformer_layers: 1,
            graph_layers: 1,
            ..FoundationConfig::default()
        };
        let trainer_config = FoundationTrainerConfig {
            batch_size: 2,
            collator: FoundationCollatorConfig {
                corruption: FoundationCorruptionConfig {
                    residue_mask_probability: 0.5,
                    chemistry_mask_probability: 0.5,
                },
                ..FoundationCollatorConfig::default()
            },
            ..FoundationTrainerConfig::default()
        };
        let mut trainer =
            FoundationTrainer::new(model_config, trainer_config, Device::Cpu).unwrap();
        let records = vec![
            training_record("PEPTIDEK", 2, 25.0, 410.0),
            training_record("AGHCEWQMK", 3, 42.0, 455.0),
        ];
        let metrics = trainer.train_step(&records).unwrap();
        assert!(metrics.total_loss.is_finite());
        assert!(metrics.rt_loss.unwrap().is_finite());
        assert!(metrics.ccs_loss.unwrap().is_finite());
        assert!(metrics.ms2_loss.unwrap().is_finite());
        assert!(metrics.contrastive_loss.unwrap().is_finite());
        assert!(metrics.gradient_norm.is_finite());
        assert_eq!(trainer.global_step(), 1);
        assert_eq!(trainer.optimizer_step(), 1);
    }

    fn training_record(
        sequence: &str,
        charge: i32,
        rt: f32,
        ccs: f32,
    ) -> FoundationTrainingRecord {
        FoundationTrainingRecord {
            peptidoform: PeptidoformInput::unmodified(sequence),
            retention_time: RetentionTimeLabels {
                normalized: Some(rt),
                observed_seconds: None,
            },
            ccs: Some(ccs),
            fragments: vec![FragmentTarget {
                cleavage_index: 1,
                channel: 0,
                intensity: 1.0,
            }],
            context: TrainingContext {
                charge: Some(charge),
                nce: Some(27.0),
                instrument_id: Some(1),
                ..TrainingContext::default()
            },
            run_id: None,
        }
    }
}
