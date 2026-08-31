//! Training schedules, early stopping, and deterministic epoch ordering.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Learning-rate schedule applied before each optimizer update.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FoundationLearningRateSchedule {
    /// Keep the configured base learning rate fixed.
    #[default]
    Constant,
    /// Linear warm-up followed by cosine decay toward a base-LR fraction.
    WarmupCosine {
        /// Number of initial optimizer steps used for linear warm-up.
        warmup_steps: u64,
        /// Total planned optimizer steps including warm-up.
        total_steps: u64,
        /// Final learning-rate fraction relative to the configured base LR.
        min_lr_ratio: f64,
    },
}

impl FoundationLearningRateSchedule {
    /// Validate schedule parameters.
    pub fn validate(self) -> Result<Self> {
        if let Self::WarmupCosine {
            warmup_steps,
            total_steps,
            min_lr_ratio,
        } = self
        {
            if total_steps == 0 {
                anyhow::bail!("foundation warmup-cosine total_steps must be greater than zero");
            }
            if warmup_steps >= total_steps {
                anyhow::bail!(
                    "foundation warmup_steps ({warmup_steps}) must be smaller than total_steps ({total_steps})"
                );
            }
            if !(0.0..=1.0).contains(&min_lr_ratio) || !min_lr_ratio.is_finite() {
                anyhow::bail!("foundation min_lr_ratio must be finite and in [0, 1]");
            }
        }
        Ok(self)
    }

    /// Learning rate for the next 0-based optimizer step.
    pub fn learning_rate(self, base_learning_rate: f64, step: u64) -> Result<f64> {
        self.validate()?;
        if !(base_learning_rate > 0.0 && base_learning_rate.is_finite()) {
            anyhow::bail!("foundation base learning rate must be positive and finite");
        }
        match self {
            Self::Constant => Ok(base_learning_rate),
            Self::WarmupCosine {
                warmup_steps,
                total_steps,
                min_lr_ratio,
            } => {
                if warmup_steps > 0 && step < warmup_steps {
                    return Ok(base_learning_rate * (step + 1) as f64 / warmup_steps as f64);
                }
                let decay_steps = total_steps - warmup_steps;
                let elapsed = step.saturating_sub(warmup_steps).min(decay_steps);
                let progress = elapsed as f64 / decay_steps as f64;
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                let ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine;
                Ok(base_learning_rate * ratio)
            }
        }
    }
}

/// High-level epoch-loop controls.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationFitConfig {
    /// Maximum number of completed epochs.
    pub max_epochs: u64,
    /// Stop after this many epochs without sufficient validation improvement.
    /// `None` disables early stopping.
    pub early_stopping_patience: Option<u64>,
    /// Required reduction in validation total loss to reset patience.
    pub early_stopping_min_delta: f32,
    /// Deterministically shuffle training records each epoch.
    pub shuffle_each_epoch: bool,
}

impl Default for FoundationFitConfig {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            early_stopping_patience: Some(10),
            early_stopping_min_delta: 0.0,
            shuffle_each_epoch: true,
        }
    }
}

impl FoundationFitConfig {
    /// Validate high-level fit controls.
    pub fn validate(self) -> Result<Self> {
        if self.max_epochs == 0 {
            anyhow::bail!("foundation max_epochs must be greater than zero");
        }
        if !(self.early_stopping_min_delta >= 0.0 && self.early_stopping_min_delta.is_finite()) {
            anyhow::bail!("foundation early_stopping_min_delta must be non-negative and finite");
        }
        if self.early_stopping_patience == Some(0) {
            anyhow::bail!("foundation early_stopping_patience must be at least 1 when enabled");
        }
        Ok(self)
    }
}

/// Tiny deterministic PRNG sufficient for reproducible Fisher-Yates shuffling.
pub(crate) struct FoundationSplitMix64 {
    state: u64,
}

impl FoundationSplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(crate) fn shuffle<T>(&mut self, values: &mut [T]) {
        for upper in (1..values.len()).rev() {
            let index = (self.next_u64() % (upper as u64 + 1)) as usize;
            values.swap(upper, index);
        }
    }
}
