//! End-to-end multi-source foundation pretraining orchestration.
//!
//! This is intentionally a library API first. The example CLI simply loads a
//! YAML configuration and calls [`run_foundation_pretraining`], keeping the
//! actual experiment semantics testable and reusable by the future ReDeeM CLI.

use super::checkpoint::{FoundationCheckpointMetadata, FoundationCheckpointProvenance};
use super::control::FoundationFitConfig;
use super::corpus::{load_foundation_corpus, FoundationCorpusConfig};
use super::experiment::{FoundationBenchmarkManifest, FoundationPartition};
use super::normalization::FoundationTargetNormalizationConfig;
use super::sampling::{
    sample_foundation_training_indices, sample_foundation_validation_indices, FoundationSamplePlan,
};
use super::trainer::{
    FoundationEpochMetrics, FoundationFitSummary, FoundationTrainer, FoundationTrainerConfig,
};
use super::FoundationConfig;
use anyhow::{Context, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// YAML configuration for one production foundation pretraining run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTrainingRunConfig {
    /// Model architecture.
    pub model: FoundationConfig,
    /// Optimizer/loss/collation/scheduler configuration.
    pub trainer: FoundationTrainerConfig,
    /// Epoch/early-stopping controls.
    pub fit: FoundationFitConfig,
    /// Multi-source data configuration.
    pub corpus: FoundationCorpusConfig,
    /// Materialized corpus-wide benchmark manifest.
    pub benchmark_manifest: PathBuf,
    /// Root where `latest/` and `best/` checkpoint directories are written.
    pub checkpoint_root: PathBuf,
    /// Resume from `checkpoint_root/latest` when it exists.
    pub resume: bool,
    /// Optional experiment identifier stored in checkpoint provenance.
    pub experiment_id: Option<String>,
}

impl Default for FoundationTrainingRunConfig {
    fn default() -> Self {
        Self {
            model: FoundationConfig::default(),
            trainer: FoundationTrainerConfig::default(),
            fit: FoundationFitConfig::default(),
            corpus: FoundationCorpusConfig::default(),
            benchmark_manifest: PathBuf::new(),
            checkpoint_root: PathBuf::new(),
            resume: false,
            experiment_id: None,
        }
    }
}

impl FoundationTrainingRunConfig {
    /// Validate cross-component configuration invariants.
    pub fn validate(&self) -> Result<()> {
        self.model.validate().map_err(anyhow::Error::msg)?;
        self.trainer
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        self.fit.validate()?;
        if self.corpus.instrument_vocab_size != self.model.instrument_vocab_size {
            anyhow::bail!(
                "corpus instrument_vocab_size {} must match model instrument_vocab_size {}",
                self.corpus.instrument_vocab_size,
                self.model.instrument_vocab_size
            );
        }
        if self.benchmark_manifest.as_os_str().is_empty() {
            anyhow::bail!("foundation benchmark_manifest path cannot be empty");
        }
        if self.checkpoint_root.as_os_str().is_empty() {
            anyhow::bail!("foundation checkpoint_root path cannot be empty");
        }
        Ok(())
    }
}

/// Result of one end-to-end pretraining invocation.
#[derive(Debug, Clone)]
pub struct FoundationTrainingRunSummary {
    /// Combined source/corpus fingerprint.
    pub corpus_fingerprint: u64,
    /// Number of combined grouped precursor records.
    pub corpus_records: usize,
    /// Number of training records selected by the benchmark manifest.
    pub train_records: usize,
    /// Number of validation records selected by the benchmark manifest.
    pub validation_records: usize,
    /// Number of test records reserved and never consumed by `fit`.
    pub test_records: usize,
    /// Whether a checkpoint was loaded before fitting.
    pub resumed: bool,
    /// Metadata loaded when resuming.
    pub resume_metadata: Option<FoundationCheckpointMetadata>,
    /// Preview of the first/current training epoch sampling mixture.
    pub train_sampling_preview: FoundationSamplePlan,
    /// Fixed validation selection used by the fit loop.
    pub validation_sampling: FoundationSamplePlan,
    /// Optional final corrupted multi-task validation diagnostics evaluated separately by source.
    /// These do not affect checkpoint selection or early stopping.
    pub validation_by_source: BTreeMap<String, FoundationEpochMetrics>,
    /// Optional final clean RT/CCS/MS2 validation diagnostics by source.
    /// These use uncorrupted peptide inputs and are disabled unless
    /// `trainer.evaluation.clean_property_validation` is enabled.
    pub property_validation_by_source: BTreeMap<String, FoundationEpochMetrics>,
    /// Train-partition-only regression normalization actually used by the run.
    pub target_normalization: FoundationTargetNormalizationConfig,
    /// Completed fit summary.
    pub fit: FoundationFitSummary,
}

/// Result of evaluating a frozen foundation checkpoint on one benchmark partition.
#[derive(Debug, Clone)]
pub struct FoundationCheckpointEvaluationSummary {
    /// Combined source/corpus fingerprint.
    pub corpus_fingerprint: u64,
    /// Number of combined grouped precursor records.
    pub corpus_records: usize,
    /// Benchmark partition evaluated.
    pub partition: FoundationPartition,
    /// Number of records in the complete requested partition.
    pub partition_records: usize,
    /// Deterministic sampled records consumed by this evaluation.
    pub sampling: FoundationSamplePlan,
    /// Checkpoint metadata used to reconstruct the model and target normalization.
    pub checkpoint_metadata: FoundationCheckpointMetadata,
    /// Clean RT/CCS/MS2 metrics on the sampled partition.
    pub property_metrics: FoundationEpochMetrics,
    /// Clean property metrics separately by source represented in the sample.
    pub property_metrics_by_source: BTreeMap<String, FoundationEpochMetrics>,
}

/// Evaluate a frozen checkpoint without optimizer updates.
///
/// The checkpoint is the authority for model architecture, target normalization,
/// gradient-gate/trainer settings, and model weights. The supplied training-run
/// configuration is used only to reconstruct the exact corpus and benchmark.
/// Corpus and benchmark fingerprints are checked against checkpoint provenance
/// before any metrics are computed.
pub fn evaluate_foundation_checkpoint<P: AsRef<Path>>(
    training_config: &FoundationTrainingRunConfig,
    checkpoint_dir: P,
    partition: FoundationPartition,
    evaluation_steps: Option<usize>,
    stratify_with_validation_source_weights: bool,
    device: Device,
) -> Result<FoundationCheckpointEvaluationSummary> {
    training_config.validate()?;
    if partition == FoundationPartition::Train {
        anyhow::bail!(
            "foundation frozen-checkpoint evaluation is intended for validation or test partitions, not train"
        );
    }
    if evaluation_steps == Some(0) {
        anyhow::bail!("foundation evaluation_steps must be at least 1 when set");
    }

    let corpus = load_foundation_corpus(&training_config.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&training_config.benchmark_manifest)?;
    benchmark
        .validate_against_records(&corpus.records)
        .context("foundation benchmark does not match the assembled corpus")?;

    let partition_indices = benchmark.partition_indices(partition);
    if partition_indices.is_empty() {
        anyhow::bail!("foundation requested evaluation partition is empty");
    }

    let (trainer, checkpoint_metadata) =
        FoundationTrainer::from_checkpoint(checkpoint_dir.as_ref(), device)?;
    validate_evaluation_checkpoint_provenance(
        &checkpoint_metadata,
        corpus.corpus_fingerprint,
        benchmark.dataset_fingerprint,
        benchmark.manifest_fingerprint(),
    )?;

    if training_config.corpus.instrument_vocab_size
        != checkpoint_metadata.model_config.instrument_vocab_size
    {
        anyhow::bail!(
            "foundation corpus instrument_vocab_size {} does not match checkpoint model instrument_vocab_size {}",
            training_config.corpus.instrument_vocab_size,
            checkpoint_metadata.model_config.instrument_vocab_size
        );
    }

    let mut sampling_config = checkpoint_metadata.trainer_config.sampling.clone();
    sampling_config.validation_steps = evaluation_steps;
    sampling_config.report_validation_by_source = true;
    if !stratify_with_validation_source_weights {
        sampling_config.validation_source_weights.clear();
    }

    let sampling = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &partition_indices,
        checkpoint_metadata.trainer_config.batch_size,
        checkpoint_metadata.trainer_config.seed,
        &sampling_config,
    )?;
    let property_metrics =
        trainer.evaluate_property_epoch_indices(&corpus.records, &sampling.indices)?;

    let mut by_source = BTreeMap::<String, Vec<usize>>::new();
    for &index in &sampling.indices {
        let source = corpus.provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation evaluation provenance index {index} is out of bounds")
        })?;
        by_source
            .entry(source.source_id.clone())
            .or_default()
            .push(index);
    }
    let mut property_metrics_by_source = BTreeMap::new();
    for (source, indices) in by_source {
        let metrics = trainer.evaluate_property_epoch_indices(&corpus.records, &indices)?;
        property_metrics_by_source.insert(source, metrics);
    }

    Ok(FoundationCheckpointEvaluationSummary {
        corpus_fingerprint: corpus.corpus_fingerprint,
        corpus_records: corpus.records.len(),
        partition,
        partition_records: partition_indices.len(),
        sampling,
        checkpoint_metadata,
        property_metrics,
        property_metrics_by_source,
    })
}

fn validate_evaluation_checkpoint_provenance(
    checkpoint: &FoundationCheckpointMetadata,
    corpus_fingerprint: u64,
    benchmark_dataset_fingerprint: u64,
    benchmark_manifest_fingerprint: u64,
) -> Result<()> {
    if checkpoint.provenance.corpus_fingerprint != Some(corpus_fingerprint) {
        anyhow::bail!(
            "foundation checkpoint corpus fingerprint does not match the assembled evaluation corpus"
        );
    }
    if checkpoint.provenance.benchmark_dataset_fingerprint != Some(benchmark_dataset_fingerprint) {
        anyhow::bail!(
            "foundation checkpoint benchmark dataset fingerprint does not match evaluation benchmark"
        );
    }
    if checkpoint.provenance.benchmark_manifest_fingerprint != Some(benchmark_manifest_fingerprint)
    {
        anyhow::bail!(
            "foundation checkpoint benchmark manifest fingerprint does not match evaluation benchmark"
        );
    }
    Ok(())
}

/// Execute one multi-source pretraining run from a materialized benchmark.
pub fn run_foundation_pretraining(
    config: &FoundationTrainingRunConfig,
    device: Device,
) -> Result<FoundationTrainingRunSummary> {
    config.validate()?;
    let corpus = load_foundation_corpus(&config.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&config.benchmark_manifest)?;
    benchmark
        .validate_against_records(&corpus.records)
        .context("foundation benchmark does not match the assembled corpus")?;

    let train_indices = benchmark.partition_indices(FoundationPartition::Train);
    let validation_indices = benchmark.partition_indices(FoundationPartition::Validation);
    let test_indices = benchmark.partition_indices(FoundationPartition::Test);
    if train_indices.is_empty() || validation_indices.is_empty() {
        anyhow::bail!(
            "foundation benchmark must contain non-empty train and validation partitions"
        );
    }

    let mut resolved_trainer_config = config.trainer.clone();
    resolved_trainer_config
        .target_normalization
        .resolve_from_training_partition(
            &corpus.records,
            &train_indices,
            resolved_trainer_config.collator.retention_time_objective,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    resolved_trainer_config
        .validate()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let latest = config.checkpoint_root.join("latest");
    let (mut trainer, progress, resume_metadata, resumed) = if config.resume && latest.exists() {
        let (trainer, metadata) = FoundationTrainer::from_checkpoint(&latest, device)?;
        validate_resume_config(
            &config.model,
            &resolved_trainer_config,
            &metadata,
            corpus.corpus_fingerprint,
            benchmark.dataset_fingerprint,
            benchmark.manifest_fingerprint(),
        )?;
        let progress = metadata.progress.clone();
        (trainer, progress, Some(metadata), true)
    } else {
        (
            FoundationTrainer::new(
                config.model.clone(),
                resolved_trainer_config.clone(),
                device,
            )?,
            Default::default(),
            None,
            false,
        )
    };

    if resumed && config.model.dropout > 0.0 {
        log::warn!(
            "foundation resume restored model and AdamW state, but Candle dropout RNG state is not serialized; stochastic trajectories need not be bitwise identical across processes"
        );
    }

    let train_sampling_preview = sample_foundation_training_indices(
        &corpus.records,
        &corpus.provenance,
        &train_indices,
        resolved_trainer_config.batch_size,
        progress.completed_epochs,
        resolved_trainer_config.seed,
        config.fit.shuffle_each_epoch,
        &resolved_trainer_config.sampling,
    )?;
    let validation_sampling = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &validation_indices,
        resolved_trainer_config.batch_size,
        resolved_trainer_config.seed,
        &resolved_trainer_config.sampling,
    )?;

    let provenance = FoundationCheckpointProvenance {
        corpus_fingerprint: Some(corpus.corpus_fingerprint),
        benchmark_dataset_fingerprint: Some(benchmark.dataset_fingerprint),
        benchmark_manifest_fingerprint: Some(benchmark.manifest_fingerprint()),
        benchmark_manifest: Some(config.benchmark_manifest.to_string_lossy().into_owned()),
        experiment_id: config.experiment_id.clone(),
    };
    let fit = trainer.fit_corpus_indices(
        &corpus.records,
        &corpus.provenance,
        &train_indices,
        &validation_indices,
        config.fit,
        &config.checkpoint_root,
        provenance,
        progress,
    )?;

    let mut validation_by_source = BTreeMap::new();
    let mut property_validation_by_source = BTreeMap::new();
    if resolved_trainer_config.sampling.report_validation_by_source {
        let mut by_source = BTreeMap::<String, Vec<usize>>::new();
        for &index in &validation_sampling.indices {
            let source = corpus.provenance.get(index).ok_or_else(|| {
                anyhow::anyhow!("foundation validation provenance index {index} is out of bounds")
            })?;
            by_source
                .entry(source.source_id.clone())
                .or_default()
                .push(index);
        }
        for (source, indices) in by_source {
            let metrics = trainer.evaluate_epoch_indices(&corpus.records, &indices)?;
            validation_by_source.insert(source.clone(), metrics);
            if resolved_trainer_config.evaluation.clean_property_validation {
                let property_metrics =
                    trainer.evaluate_property_epoch_indices(&corpus.records, &indices)?;
                property_validation_by_source.insert(source, property_metrics);
            }
        }
    }

    Ok(FoundationTrainingRunSummary {
        corpus_fingerprint: corpus.corpus_fingerprint,
        corpus_records: corpus.records.len(),
        train_records: train_indices.len(),
        validation_records: validation_indices.len(),
        test_records: test_indices.len(),
        resumed,
        resume_metadata,
        train_sampling_preview,
        validation_sampling,
        validation_by_source,
        property_validation_by_source,
        target_normalization: trainer.config().target_normalization,
        fit,
    })
}

fn validate_resume_config(
    requested_model: &FoundationConfig,
    requested_trainer: &FoundationTrainerConfig,
    checkpoint: &FoundationCheckpointMetadata,
    corpus_fingerprint: u64,
    benchmark_dataset_fingerprint: u64,
    benchmark_manifest_fingerprint: u64,
) -> Result<()> {
    if checkpoint.model_config != *requested_model {
        anyhow::bail!("foundation resume model configuration differs from checkpoint");
    }
    let requested_trainer = serde_yaml::to_string(requested_trainer)?;
    let checkpoint_trainer = serde_yaml::to_string(&checkpoint.trainer_config)?;
    if requested_trainer != checkpoint_trainer {
        anyhow::bail!("foundation resume trainer configuration differs from checkpoint");
    }
    if let Some(expected) = checkpoint.provenance.corpus_fingerprint {
        if expected != corpus_fingerprint {
            anyhow::bail!(
                "foundation resume corpus fingerprint differs from checkpoint: current fnv1a64:{corpus_fingerprint:016x}, checkpoint fnv1a64:{expected:016x}"
            );
        }
    }
    if let Some(expected) = checkpoint.provenance.benchmark_dataset_fingerprint {
        if expected != benchmark_dataset_fingerprint {
            anyhow::bail!(
                "foundation resume benchmark dataset fingerprint differs from checkpoint: current fnv1a64:{benchmark_dataset_fingerprint:016x}, checkpoint fnv1a64:{expected:016x}"
            );
        }
    }
    if let Some(expected) = checkpoint.provenance.benchmark_manifest_fingerprint {
        if expected != benchmark_manifest_fingerprint {
            anyhow::bail!(
                "foundation resume benchmark assignment fingerprint differs from checkpoint: current fnv1a64:{benchmark_manifest_fingerprint:016x}, checkpoint fnv1a64:{expected:016x}"
            );
        }
    }
    Ok(())
}

/// Read a YAML run configuration from disk.
pub fn read_foundation_training_run_config<P: AsRef<Path>>(
    path: P,
) -> Result<FoundationTrainingRunConfig> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open foundation training config {path:?}"))?;
    let config: FoundationTrainingRunConfig = serde_yaml::from_reader(file)
        .with_context(|| format!("failed to parse foundation training config {path:?}"))?;
    config.validate()?;
    Ok(config)
}
