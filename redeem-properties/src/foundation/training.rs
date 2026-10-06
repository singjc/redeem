//! Minimal end-to-end training loop for [`FoundationModel`](super::runtime::FoundationModel).
//!
//! This is intentionally one training regime rather than another experiment ladder:
//! random initialization (or optional same-architecture warm start), one optimizer,
//! TRAIN-only normalization, Validation-only early stopping, and one checkpoint format.

use super::collate::{FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig};
use super::corpus::{load_foundation_corpus, FoundationCorpusConfig};
use super::data::{FoundationTrainingRecord, RetentionTimeObjective};
use super::experiment::{FoundationBenchmarkManifest, FoundationPartition};
use super::loss::{
    contrastive_info_nce_loss, multi_task_loss_with_ms2_config, FoundationLossWeights,
    FoundationMs2LossConfig, FoundationTargets,
};
use super::normalization::{
    FoundationRegressionNormalization, FoundationRegressionNormalizationStrategy,
    FoundationTargetNormalizationConfig,
};
use super::runtime::{FoundationCheckpointMetadata, FoundationModel, FoundationModelConfig};
use super::spectrum::FoundationSpectrum;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Hyperparameters for the one supported end-to-end foundation training regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTrainingConfig {
    pub model: FoundationModelConfig,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub seed: u64,
    /// Optional bounded smoke/debug mode; `None` trains the full partition.
    pub max_train_batches_per_epoch: Option<usize>,
    /// Optional bounded validation mode; `None` evaluates the full partition.
    pub max_validation_batches: Option<usize>,
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
            learning_rate: 3.0e-5,
            epochs: 20,
            early_stopping_patience: 4,
            seed: 20_261_006,
            max_train_batches_per_epoch: None,
            max_validation_batches: None,
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

    let train_collator = FoundationCollator::new(
        model_config.peptide.clone(),
        FoundationCollatorConfig {
            retention_time_objective: RetentionTimeObjective::IntrinsicAndObserved,
            corruption: training.corruption,
        },
    )?;
    let params = ParamsAdamW {
        lr: training.learning_rate,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(model.trainable_variables(), params)?;

    fs::create_dir_all(output_dir)
        .with_context(|| format!("create foundation output {}", output_dir.display()))?;
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
    fs::write(
        &history,
        "epoch\ttrain_total\ttrain_forward\ttrain_inverse\ttrain_inverse_ptm_mass\ttrain_peptide_contrastive\ttrain_cross_modal_alignment\tvalidation_total\tvalidation_forward\tvalidation_inverse\tvalidation_inverse_ptm_mass\tvalidation_peptide_contrastive\tvalidation_cross_modal_alignment\n",
    )?;

    let mut best_validation_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0usize;
    let mut completed_epochs = 0usize;

    for epoch in 0..training.epochs {
        let mut order = train_indices.clone();
        shuffle_indices(&mut order, training.seed ^ epoch as u64);
        let train_losses = run_epoch(
            &model,
            &corpus.records,
            &order,
            &train_collator,
            &training,
            Some(&mut optimizer),
            training.max_train_batches_per_epoch,
            epoch as u64,
            &device,
        )?;
        let validation_losses = run_epoch(
            &model,
            &corpus.records,
            &validation_indices,
            &train_collator,
            &training,
            None,
            training.max_validation_batches,
            training.seed ^ 0xa5a5_5a5a_1234_5678,
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
        eprintln!(
            "foundation epoch={} train_loss={:.6} validation_loss={:.6} inverse={:.6} alignment={:.6}",
            completed_epochs,
            train_losses.total,
            validation_losses.total,
            validation_losses.inverse,
            validation_losses.cross_modal_alignment,
        );

        if validation_losses.total < best_validation_loss {
            best_validation_loss = validation_losses.total;
            epochs_without_improvement = 0;
            let mut metadata = FoundationCheckpointMetadata::new(
                model_config.clone(),
                normalization,
                corpus.instrument_names.clone(),
            );
            metadata.completed_epochs = completed_epochs;
            metadata.best_validation_loss = Some(best_validation_loss);
            metadata.corpus_fingerprint =
                Some(format!("fnv1a64:{:016x}", corpus.corpus_fingerprint));
            metadata.benchmark_fingerprint =
                Some(format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint()));
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
    })
}

fn run_epoch(
    model: &FoundationModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    collator: &FoundationCollator,
    config: &FoundationTrainingConfig,
    mut optimizer: Option<&mut AdamW>,
    max_batches: Option<usize>,
    seed: u64,
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
            (*opt).backward_step(&loss.total)?;
        }
        aggregate.add(loss.scalars);
    }
    if aggregate.batches == 0 {
        anyhow::bail!("foundation epoch produced no batches");
    }
    Ok(aggregate.means())
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
