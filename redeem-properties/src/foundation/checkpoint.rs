//! Foundation checkpoint metadata and exact model/optimizer resume support.
//!
//! A checkpoint directory contains model weights, AdamW moments, and a YAML
//! metadata file. Keeping optimizer state separate from model SafeTensors makes
//! the contents inspectable while still allowing an exact continuation of the
//! optimizer trajectory.

use super::config::FoundationConfig;
use super::trainer::FoundationTrainerConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Current foundation training-checkpoint format.
pub const FOUNDATION_CHECKPOINT_VERSION: u32 = 1;
/// File containing named model parameters.
pub const FOUNDATION_MODEL_FILE: &str = "model.safetensors";
/// File containing named AdamW first/second moments.
pub const FOUNDATION_OPTIMIZER_FILE: &str = "optimizer.safetensors";
/// Human-readable checkpoint state.
pub const FOUNDATION_STATE_FILE: &str = "state.yaml";

/// Optional provenance used to prevent accidental resume on a different corpus.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationCheckpointProvenance {
    /// Combined source/corpus fingerprint when known.
    pub corpus_fingerprint: Option<u64>,
    /// Benchmark dataset fingerprint when a materialized split is in use.
    pub benchmark_dataset_fingerprint: Option<u64>,
    /// Fingerprint over the exact benchmark partition assignments.
    pub benchmark_manifest_fingerprint: Option<u64>,
    /// Optional path or logical name of the materialized benchmark manifest.
    pub benchmark_manifest: Option<String>,
    /// Free-form experiment identifier supplied by the caller.
    pub experiment_id: Option<String>,
}

/// Mutable training progress that is independent from model architecture.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTrainingProgress {
    /// Number of completely finished epochs.
    pub completed_epochs: u64,
    /// Best validation total loss observed so far.
    pub best_validation_loss: Option<f32>,
    /// Epoch index at which the best validation loss was observed.
    pub best_epoch: Option<u64>,
    /// Consecutive completed epochs without sufficient validation improvement.
    pub epochs_without_improvement: u64,
}

/// Human-readable metadata stored next to SafeTensors checkpoint files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundationCheckpointMetadata {
    /// Checkpoint format version.
    pub format_version: u32,
    /// Model architecture needed to reconstruct the exact parameter shapes.
    pub model_config: FoundationConfig,
    /// Trainer/optimizer/scheduler configuration.
    pub trainer_config: FoundationTrainerConfig,
    /// Number of completed optimizer updates.
    pub global_step: u64,
    /// AdamW bias-correction update count. Must match the saved moment state.
    pub optimizer_step: u64,
    /// Epoch/best-validation state.
    pub progress: FoundationTrainingProgress,
    /// Corpus/benchmark provenance, when available.
    pub provenance: FoundationCheckpointProvenance,
}

impl FoundationCheckpointMetadata {
    /// Construct metadata for the current format.
    pub fn new(
        model_config: FoundationConfig,
        trainer_config: FoundationTrainerConfig,
        global_step: u64,
        optimizer_step: u64,
        progress: FoundationTrainingProgress,
        provenance: FoundationCheckpointProvenance,
    ) -> Self {
        Self {
            format_version: FOUNDATION_CHECKPOINT_VERSION,
            model_config,
            trainer_config,
            global_step,
            optimizer_step,
            progress,
            provenance,
        }
    }

    /// Validate format and internal counters before loading tensors.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != FOUNDATION_CHECKPOINT_VERSION {
            anyhow::bail!(
                "unsupported foundation checkpoint version {} (expected {})",
                self.format_version,
                FOUNDATION_CHECKPOINT_VERSION
            );
        }
        self.model_config.validate().map_err(anyhow::Error::msg)?;
        self.trainer_config
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if self.global_step != self.optimizer_step {
            anyhow::bail!(
                "foundation checkpoint global_step {} does not match optimizer_step {}",
                self.global_step,
                self.optimizer_step
            );
        }
        Ok(())
    }

    /// Write metadata as YAML.
    pub fn write_yaml<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("failed to create foundation checkpoint metadata {path:?}"))?;
        serde_yaml::to_writer(file, self)
            .with_context(|| format!("failed to write foundation checkpoint metadata {path:?}"))
    }

    /// Read metadata from YAML and validate it.
    pub fn read_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open foundation checkpoint metadata {path:?}"))?;
        let metadata: Self = serde_yaml::from_reader(file)
            .with_context(|| format!("failed to parse foundation checkpoint metadata {path:?}"))?;
        metadata.validate()?;
        Ok(metadata)
    }
}

/// Resolve the three standard files under one checkpoint directory.
pub fn foundation_checkpoint_paths<P: AsRef<Path>>(directory: P) -> (PathBuf, PathBuf, PathBuf) {
    let directory = directory.as_ref();
    (
        directory.join(FOUNDATION_MODEL_FILE),
        directory.join(FOUNDATION_OPTIMIZER_FILE),
        directory.join(FOUNDATION_STATE_FILE),
    )
}
