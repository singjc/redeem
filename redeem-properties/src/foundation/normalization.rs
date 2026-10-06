//! Train-partition target normalization for heterogeneous foundation regression.
//!
//! Regression targets can live on very different numerical scales. The
//! foundation trainer therefore standardizes supported regression targets using
//! statistics fitted strictly from the materialized training partition. The
//! resolved statistics are serialized as part of [`FoundationTrainerConfig`],
//! so checkpoint resume uses exactly the same transform and never re-fits on
//! validation/test data.

use super::data::{FoundationTrainingRecord, RetentionTimeObjective};
use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

/// Policy used to scale one continuous regression target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FoundationRegressionNormalizationStrategy {
    /// Preserve the target in its native units.
    #[default]
    None,
    /// Subtract the train-partition mean and divide by its population standard
    /// deviation before evaluating the regression loss.
    TrainStandardize,
}

/// Resolved normalization state for one continuous target.
///
/// `mean`, `standard_deviation`, and `label_count` are runtime-resolved fields.
/// They may be omitted from a user-authored YAML configuration and are filled
/// before a production trainer is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationRegressionNormalization {
    /// Scaling policy.
    pub strategy: FoundationRegressionNormalizationStrategy,
    /// Mean estimated only from finite training-partition labels.
    pub mean: Option<f64>,
    /// Population standard deviation estimated only from training labels.
    pub standard_deviation: Option<f64>,
    /// Number of finite labels contributing to the fitted statistics.
    pub label_count: usize,
    /// Lower bound below which the scale falls back to one rather than divide
    /// by a near-zero value.
    pub min_standard_deviation: f64,
}

impl Default for FoundationRegressionNormalization {
    fn default() -> Self {
        Self {
            strategy: FoundationRegressionNormalizationStrategy::None,
            mean: None,
            standard_deviation: None,
            label_count: 0,
            min_standard_deviation: 1e-6,
        }
    }
}

impl FoundationRegressionNormalization {
    /// Validate both user-facing policy and any already-resolved statistics.
    pub fn validate(&self) -> Result<()> {
        if !(self.min_standard_deviation > 0.0 && self.min_standard_deviation.is_finite()) {
            candle_core::bail!(
                "foundation minimum regression standard deviation must be positive and finite"
            );
        }
        match (self.mean, self.standard_deviation) {
            (Some(mean), Some(scale)) => {
                if !mean.is_finite() || !(scale > 0.0 && scale.is_finite()) {
                    candle_core::bail!(
                        "foundation resolved regression normalization statistics must be finite with positive scale"
                    );
                }
            }
            (None, None) => {}
            _ => candle_core::bail!(
                "foundation regression normalization mean and standard deviation must be resolved together"
            ),
        }
        Ok(())
    }

    /// Whether a non-identity transform is currently resolved.
    pub fn is_active(&self) -> bool {
        self.strategy == FoundationRegressionNormalizationStrategy::TrainStandardize
            && self.label_count > 0
            && self.mean.is_some()
            && self.standard_deviation.is_some()
    }

    /// Fit/overwrite the runtime statistics from finite scalar labels.
    pub fn resolve_from_values<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = f32>,
    {
        self.mean = None;
        self.standard_deviation = None;
        self.label_count = 0;
        if self.strategy == FoundationRegressionNormalizationStrategy::None {
            return Ok(());
        }

        // Numerically stable single-pass Welford estimator. We use the
        // population variance because the fitted transform describes the
        // complete materialized training partition rather than estimating an
        // external population parameter.
        let mut count = 0usize;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        for value in values {
            if !value.is_finite() {
                continue;
            }
            count += 1;
            let x = f64::from(value);
            let delta = x - mean;
            mean += delta / count as f64;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }

        self.label_count = count;
        if count == 0 {
            // Missing labels remain valid in heterogeneous corpora. An
            // unresolved transform simply acts as identity.
            return Ok(());
        }
        let variance = (m2 / count as f64).max(0.0);
        let observed_sd = variance.sqrt();
        let scale = if observed_sd >= self.min_standard_deviation {
            observed_sd
        } else {
            1.0
        };
        self.mean = Some(mean);
        self.standard_deviation = Some(scale);
        self.validate()
    }

    /// Transform native-unit targets into the regression space used by loss.
    pub fn normalize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        if !self.is_active() {
            return Ok(tensor.clone());
        }
        let mean = self.mean.unwrap_or(0.0);
        let scale = self.standard_deviation.unwrap_or(1.0);
        tensor.affine(1.0 / scale, -mean / scale)
    }

    /// Convert standardized predictions/targets back to their native units.
    pub fn denormalize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        if !self.is_active() {
            return Ok(tensor.clone());
        }
        tensor.affine(
            self.standard_deviation.unwrap_or(1.0),
            self.mean.unwrap_or(0.0),
        )
    }
}

/// Regression normalization policies/stats used by foundation heads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTargetNormalizationConfig {
    /// Intrinsic RT/iRT regression scaling.
    pub rt: FoundationRegressionNormalization,
    /// CCS regression scaling. This remains inactive when a corpus has no CCS.
    pub ccs: FoundationRegressionNormalization,
}

impl FoundationTargetNormalizationConfig {
    /// Validate both target transforms.
    pub fn validate(&self) -> Result<()> {
        self.rt.validate()?;
        self.ccs.validate()?;
        Ok(())
    }

    /// Fit all configured target transforms from the materialized training
    /// partition only. Validation and test records are never inspected.
    pub fn resolve_from_training_partition(
        &mut self,
        records: &[FoundationTrainingRecord],
        train_indices: &[usize],
        rt_objective: RetentionTimeObjective,
    ) -> Result<()> {
        let mut rt_values = Vec::new();
        let mut ccs_values = Vec::new();
        for &index in train_indices {
            let record = records.get(index).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "foundation normalization index {index} is out of bounds for {} records",
                    records.len()
                ))
            })?;
            let rt = match rt_objective {
                RetentionTimeObjective::Normalized
                | RetentionTimeObjective::IntrinsicAndObserved => record.retention_time.normalized,
                RetentionTimeObjective::Harmonized => record.retention_time.harmonized,
                RetentionTimeObjective::Observed => record.retention_time.observed_seconds,
            };
            if let Some(value) = rt.filter(|value| value.is_finite()) {
                rt_values.push(value);
            }
            if let Some(value) = record.ccs.filter(|value| value.is_finite()) {
                ccs_values.push(value);
            }
        }
        self.rt.resolve_from_values(rt_values)?;
        self.ccs.resolve_from_values(ccs_values)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_standardization_round_trips_tensor_values() {
        let mut normalization = FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        };
        normalization
            .resolve_from_values([10.0f32, 20.0, 30.0])
            .unwrap();
        assert_eq!(normalization.label_count, 3);
        assert!((normalization.mean.unwrap() - 20.0).abs() < 1e-8);

        let tensor = Tensor::new(&[10.0f32, 20.0, 30.0], &candle_core::Device::Cpu).unwrap();
        let standardized = normalization.normalize_tensor(&tensor).unwrap();
        let restored = normalization.denormalize_tensor(&standardized).unwrap();
        let values = restored.to_vec1::<f32>().unwrap();
        for (left, right) in values.iter().zip([10.0f32, 20.0, 30.0]) {
            assert!((*left - right).abs() < 1e-4);
        }
    }

    #[test]
    fn missing_labels_leave_train_standardization_inactive() {
        let mut normalization = FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        };
        normalization
            .resolve_from_values(Vec::<f32>::new())
            .unwrap();
        assert_eq!(normalization.label_count, 0);
        assert!(!normalization.is_active());
    }
}
