//! Candle training loop for multi-task peptide foundation pretraining.
//!
//! The trainer intentionally operates on [`FoundationTrainingRecord`](crate::foundation::FoundationTrainingRecord)
//! batches rather than the legacy property-specific `PeptideData` tensor
//! interface.  It performs two independently corrupted forward passes,
//! combines sparse supervised labels with masked reconstruction objectives,
//! adds symmetric InfoNCE, and updates all trainable variables with AdamW.

use super::collate::{FoundationCollator, FoundationCollatorConfig, FoundationTrainingViews};
use super::config::FoundationConfig;
use super::data::FoundationTrainingRecord;
use super::loss::{
    contrastive_info_nce_loss, multi_task_loss, FoundationLossWeights, FoundationLosses,
};
use super::wrapper::FoundationModelWrapper;
use candle_core::{Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Training hyperparameters independent from the model architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTrainerConfig {
    /// Mini-batch size used by [`FoundationTrainer::train_epoch`].
    pub batch_size: usize,
    /// AdamW learning rate.
    pub learning_rate: f64,
    /// AdamW decoupled weight decay.
    pub weight_decay: f64,
    /// Temperature used by the symmetric InfoNCE objective.
    pub contrastive_temperature: f64,
    /// Relative task-loss contributions.
    pub loss_weights: FoundationLossWeights,
    /// Label/corruption collation settings.
    pub collator: FoundationCollatorConfig,
    /// Base seed for deterministic view corruption.
    pub seed: u64,
}

impl Default for FoundationTrainerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            contrastive_temperature: 0.10,
            loss_weights: FoundationLossWeights::default(),
            collator: FoundationCollatorConfig::default(),
            seed: 20260831,
        }
    }
}

impl FoundationTrainerConfig {
    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            candle_core::bail!("foundation batch_size must be greater than zero");
        }
        if !(self.learning_rate > 0.0 && self.learning_rate.is_finite()) {
            candle_core::bail!("foundation learning_rate must be positive and finite");
        }
        if !(self.weight_decay >= 0.0 && self.weight_decay.is_finite()) {
            candle_core::bail!("foundation weight_decay must be non-negative and finite");
        }
        if !(self.contrastive_temperature > 0.0 && self.contrastive_temperature.is_finite()) {
            candle_core::bail!("foundation contrastive_temperature must be positive and finite");
        }
        Ok(())
    }
}

/// Scalar diagnostics returned after one optimizer update.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationStepMetrics {
    /// Weighted total loss including contrastive alignment.
    pub total_loss: f32,
    /// Single-view RT loss.
    pub rt_loss: Option<f32>,
    /// Single-view CCS loss.
    pub ccs_loss: Option<f32>,
    /// Single-view MS2 loss.
    pub ms2_loss: Option<f32>,
    /// Masked residue-token reconstruction loss.
    pub masked_residue_loss: Option<f32>,
    /// Masked chemistry reconstruction loss.
    pub chemistry_loss: Option<f32>,
    /// Symmetric contrastive loss; absent for batch size one.
    pub contrastive_loss: Option<f32>,
}

/// Aggregate diagnostics for one sequential pass through a record slice.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationEpochMetrics {
    /// Number of batches evaluated/optimized.
    pub steps: usize,
    /// Mean total loss across batches.
    pub mean_total_loss: f32,
    /// Mean RT loss across batches where RT labels were present.
    pub mean_rt_loss: Option<f32>,
    /// Mean CCS loss across batches where CCS labels were present.
    pub mean_ccs_loss: Option<f32>,
    /// Mean MS2 loss across batches where fragment labels were present.
    pub mean_ms2_loss: Option<f32>,
    /// Mean masked-residue reconstruction loss when active.
    pub mean_masked_residue_loss: Option<f32>,
    /// Mean masked-chemistry reconstruction loss when active.
    pub mean_chemistry_loss: Option<f32>,
    /// Mean symmetric contrastive loss when batch size permitted it.
    pub mean_contrastive_loss: Option<f32>,
}

/// Stateful foundation trainer with model, optimizer, collator, and checkpoint state.
pub struct FoundationTrainer {
    wrapper: FoundationModelWrapper,
    optimizer: AdamW,
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
        let optimizer = AdamW::new(
            wrapper.varmap().all_vars(),
            ParamsAdamW {
                lr: config.learning_rate,
                weight_decay: config.weight_decay,
                ..ParamsAdamW::default()
            },
        )?;
        let collator = FoundationCollator::new(model_config, config.collator.clone())?;
        Ok(Self {
            wrapper,
            optimizer,
            collator,
            config,
            global_step: 0,
        })
    }

    /// Resume model weights from a SafeTensors checkpoint.
    ///
    /// AdamW moment state is intentionally not serialized in the first
    /// implementation; resuming restores model weights and starts fresh
    /// optimizer moments.  Optimizer-state persistence is a later milestone.
    pub fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.wrapper.varmap_mut().load(path)
    }

    /// Save model weights in SafeTensors format.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.wrapper.save_safetensors(path)
    }

    /// Execute one two-view multi-task optimizer update.
    pub fn train_step(
        &mut self,
        records: &[FoundationTrainingRecord],
    ) -> Result<FoundationStepMetrics> {
        let seed = self.config.seed.wrapping_add(self.global_step);
        let views = self
            .collator
            .collate_views(records, self.wrapper.device(), seed)?;
        let (total, losses, contrastive) = self.loss_for_views(&views, true)?;
        self.optimizer.backward_step(&total)?;
        self.global_step = self.global_step.wrapping_add(1);
        metrics_from_tensors(&total, &losses, contrastive.as_ref())
    }

    /// Train sequential mini-batches for one epoch.
    ///
    /// Dataset shuffling/sampling policy is intentionally kept outside this
    /// method so callers can enforce peptide-/run-disjoint sampling and other
    /// leakage controls before records reach the trainer.
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

    /// Evaluate deterministic corrupted views without updating model weights or
    /// optimizer state. Validation never increments `global_step`.
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
            let views = self
                .collator
                .collate_views(batch, self.wrapper.device(), seed)?;
            let (total, losses, contrastive) = self.loss_for_views(&views, false)?;
            accumulator.push(metrics_from_tensors(&total, &losses, contrastive.as_ref())?);
        }
        Ok(accumulator.finish())
    }

    /// Access the high-level model wrapper for embedding/evaluation/checkpointing.
    pub fn model(&self) -> &FoundationModelWrapper {
        &self.wrapper
    }

    /// Number of completed optimizer steps.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    fn loss_for_views(
        &self,
        views: &FoundationTrainingViews,
        train: bool,
    ) -> Result<(Tensor, FoundationLosses, Option<Tensor>)> {
        let first =
            self.wrapper
                .model()
                .forward_t(&views.first.input, &views.first.context, train)?;
        let second =
            self.wrapper
                .model()
                .forward_t(&views.second.input, &views.second.context, train)?;
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
        Ok((total, losses, contrastive))
    }
}

#[derive(Default)]
struct EpochAccumulator {
    steps: usize,
    total: f64,
    rt: OptionalMean,
    ccs: OptionalMean,
    ms2: OptionalMean,
    masked_residue: OptionalMean,
    chemistry: OptionalMean,
    contrastive: OptionalMean,
}

impl EpochAccumulator {
    fn push(&mut self, metrics: FoundationStepMetrics) {
        self.steps += 1;
        self.total += f64::from(metrics.total_loss);
        self.rt.push(metrics.rt_loss);
        self.ccs.push(metrics.ccs_loss);
        self.ms2.push(metrics.ms2_loss);
        self.masked_residue.push(metrics.masked_residue_loss);
        self.chemistry.push(metrics.chemistry_loss);
        self.contrastive.push(metrics.contrastive_loss);
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
            mean_ccs_loss: self.ccs.mean(),
            mean_ms2_loss: self.ms2.mean(),
            mean_masked_residue_loss: self.masked_residue.mean(),
            mean_chemistry_loss: self.chemistry.mean(),
            mean_contrastive_loss: self.contrastive.mean(),
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

    fn mean(self) -> Option<f32> {
        (self.count > 0).then(|| (self.sum / self.count as f64) as f32)
    }
}

fn metrics_from_tensors(
    total: &Tensor,
    losses: &FoundationLosses,
    contrastive: Option<&Tensor>,
) -> Result<FoundationStepMetrics> {
    Ok(FoundationStepMetrics {
        total_loss: total.to_scalar::<f32>()?,
        rt_loss: scalar_option(losses.rt.as_ref())?,
        ccs_loss: scalar_option(losses.ccs.as_ref())?,
        ms2_loss: scalar_option(losses.ms2.as_ref())?,
        masked_residue_loss: scalar_option(losses.masked_residue.as_ref())?,
        chemistry_loss: scalar_option(losses.chemistry.as_ref())?,
        contrastive_loss: scalar_option(contrastive)?,
    })
}

fn scalar_option(value: Option<&Tensor>) -> Result<Option<f32>> {
    value.map(|value| value.to_scalar::<f32>()).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationCorruptionConfig, FoundationTrainingRecord, FragmentTarget, PeptidoformInput,
        RetentionTimeLabels, TrainingContext,
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
        assert_eq!(trainer.global_step(), 1);
    }

    fn training_record(sequence: &str, charge: i32, rt: f32, ccs: f32) -> FoundationTrainingRecord {
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
