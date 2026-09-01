//! Train-only physical CCS priors for the foundation residual head.
//!
//! The learned CCS branch is intentionally responsible for peptide-specific
//! residual structure, while gross precursor physics is represented by a frozen
//! linear prior fitted only on the materialized training partition.  This module
//! owns the scalar feature contract used by offline fitting/auditing so production
//! coefficients cannot silently drift away from the model-side feature definition.

use super::config::FoundationCcsPhysicsBaselineConfig;
use super::data::{FoundationTrainingRecord, TrainingContext};
use super::featurize::PeptidoformInput;
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

/// Number of scalar features in the frozen CCS physical prior.
pub const FOUNDATION_CCS_PHYSICS_FEATURE_COUNT: usize = 8;

/// Stable feature names/order for [`FoundationCcsPhysicsBaselineConfig::coefficients_native`].
pub const FOUNDATION_CCS_PHYSICS_FEATURE_NAMES: [&str; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT] = [
    "intercept",
    "charge_over_4",
    "charge_squared_over_16",
    "precursor_mz_over_1000",
    "neutral_mass_proxy_over_3000",
    "sequence_len_over_30",
    "charge_present",
    "precursor_mz_present",
];

/// Configuration for fitting the frozen train-derived physical CCS prior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationCcsPhysicsFitConfig {
    /// L2 penalty applied to non-intercept coefficients.
    ///
    /// The current corpus has charge/mz present for essentially every CCS label,
    /// making the presence-mask columns collinear with the intercept.  A small
    /// ridge penalty keeps that documented feature contract numerically well posed
    /// without changing the physical interpretation of the fit.
    pub ridge_lambda: f64,
}

impl Default for FoundationCcsPhysicsFitConfig {
    fn default() -> Self {
        Self { ridge_lambda: 1e-4 }
    }
}

impl FoundationCcsPhysicsFitConfig {
    fn validate(self) -> Result<Self> {
        if !self.ridge_lambda.is_finite() || self.ridge_lambda < 0.0 {
            bail!("CCS physics ridge_lambda must be finite and non-negative");
        }
        Ok(self)
    }
}

/// Distribution summary for one physical input feature on the fitted train labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationCcsPhysicsFeatureSummary {
    /// Stable feature name.
    pub name: String,
    /// Minimum observed value.
    pub min: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population standard deviation.
    pub standard_deviation: f64,
    /// Maximum observed value.
    pub max: f64,
}

/// Native-unit regression diagnostics for a frozen physical CCS prior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct FoundationCcsPhysicsMetrics {
    /// Number of finite CCS labels evaluated.
    pub label_count: usize,
    /// Native CCS mean absolute error.
    pub mae_native: Option<f64>,
    /// Native CCS root mean squared error.
    pub rmse_native: Option<f64>,
    /// Coefficient of determination relative to the evaluated target mean.
    pub r_squared: Option<f64>,
    /// Pearson correlation between target and prediction.
    pub pearson_r: Option<f64>,
    /// Evaluated target mean in native CCS units.
    pub target_mean_native: Option<f64>,
    /// Evaluated prediction mean in native CCS units.
    pub prediction_mean_native: Option<f64>,
    /// Evaluated target population standard deviation.
    pub target_std_native: Option<f64>,
    /// Evaluated prediction population standard deviation.
    pub prediction_std_native: Option<f64>,
}

/// Result of fitting a frozen CCS physical prior on one training partition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationCcsPhysicsFitResult {
    /// Fitted production baseline ready for `model.ccs_physics_baseline`.
    pub baseline: FoundationCcsPhysicsBaselineConfig,
    /// Number of finite training CCS labels used by the fit.
    pub train_label_count: usize,
    /// Ridge regularization used by the fit.
    pub ridge_lambda: f64,
    /// Native-unit fit diagnostics on the training partition.
    pub train_metrics: FoundationCcsPhysicsMetrics,
    /// Feature distributions on the training labels.
    pub feature_summaries: Vec<FoundationCcsPhysicsFeatureSummary>,
}

/// Build the documented eight-feature physical CCS vector.
///
/// Missing scalar context is represented by zero plus an explicit presence mask,
/// matching the model-side context semantics.  Sequence length is the intrinsic
/// peptide residue count and is independent of acquisition metadata.
pub fn foundation_ccs_physics_features(
    peptide: &PeptidoformInput,
    context: &TrainingContext,
) -> [f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT] {
    foundation_ccs_physics_features_from_values(
        peptide.sequence.chars().count(),
        context.charge,
        context.precursor_mz,
    )
}

/// Build the physical CCS feature vector from scalar values.
///
/// This is shared by production fitting and the historical scalar-head audit so
/// all offline diagnostics use exactly the same feature order/scaling.
pub fn foundation_ccs_physics_features_from_values(
    sequence_len: usize,
    charge: Option<i32>,
    precursor_mz: Option<f32>,
) -> [f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT] {
    let charge_present = if charge.is_some() { 1.0 } else { 0.0 };
    let mz_present = if precursor_mz.is_some() { 1.0 } else { 0.0 };
    let charge = charge.unwrap_or(0) as f64 * charge_present;
    let mz = f64::from(precursor_mz.unwrap_or(0.0)) * mz_present;
    let neutral_mass_proxy = charge * mz * charge_present * mz_present;
    [
        1.0,
        charge / 4.0,
        charge * charge / 16.0,
        mz / 1000.0,
        neutral_mass_proxy / 3000.0,
        sequence_len as f64 / 30.0,
        charge_present,
        mz_present,
    ]
}

/// Predict native CCS from a frozen physical prior for one record.
pub fn predict_foundation_ccs_physics_native(
    baseline: &FoundationCcsPhysicsBaselineConfig,
    record: &FoundationTrainingRecord,
) -> f64 {
    let features = foundation_ccs_physics_features(&record.peptidoform, &record.context);
    dot(&baseline.coefficients_native, &features)
}

/// Fit the physical CCS prior on the supplied original record indices.
///
/// Callers are responsible for supplying only the benchmark training partition.
/// The function consumes every finite CCS label in `indices`; it performs no
/// sampling and never accesses validation/test records implicitly.
pub fn fit_foundation_ccs_physics_baseline(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    config: FoundationCcsPhysicsFitConfig,
) -> Result<FoundationCcsPhysicsFitResult> {
    let config = config.validate()?;
    if indices.is_empty() {
        bail!("CCS physics fit requires at least one training record index");
    }

    let mut normal =
        [[0.0f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT]; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT];
    let mut rhs = [0.0f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT];
    let mut target_stats = RunningStats::default();
    let mut feature_stats = [RunningStats::default(); FOUNDATION_CCS_PHYSICS_FEATURE_COUNT];
    let mut labelled_indices = Vec::new();

    for &index in indices {
        let record = records
            .get(index)
            .ok_or_else(|| anyhow!("CCS physics fit record index {index} is out of bounds"))?;
        let Some(target) = record.ccs.filter(|value| value.is_finite()) else {
            continue;
        };
        let features = foundation_ccs_physics_features(&record.peptidoform, &record.context);
        if features.iter().any(|value| !value.is_finite()) {
            bail!("CCS physics fit encountered non-finite features at record {index}");
        }
        let target = f64::from(target);
        target_stats.push(target);
        for (feature_index, value) in features.iter().copied().enumerate() {
            feature_stats[feature_index].push(value);
            rhs[feature_index] += value * target;
            for (other_index, other) in features.iter().copied().enumerate() {
                normal[feature_index][other_index] += value * other;
            }
        }
        labelled_indices.push(index);
    }

    if labelled_indices.len() < FOUNDATION_CCS_PHYSICS_FEATURE_COUNT {
        bail!(
            "CCS physics fit found only {} finite labels; at least {} are required",
            labelled_indices.len(),
            FOUNDATION_CCS_PHYSICS_FEATURE_COUNT
        );
    }

    for index in 1..FOUNDATION_CCS_PHYSICS_FEATURE_COUNT {
        normal[index][index] += config.ridge_lambda;
    }
    let coefficients_native = solve_linear_system(normal, rhs)?;
    if coefficients_native.iter().any(|value| !value.is_finite()) {
        bail!("CCS physics fit produced a non-finite coefficient");
    }

    let target_mean_native = target_stats
        .mean()
        .ok_or_else(|| anyhow!("CCS physics fit could not resolve target mean"))?;
    let target_std_native = target_stats
        .population_std()
        .ok_or_else(|| anyhow!("CCS physics fit could not resolve target standard deviation"))?;
    if !(target_std_native > 0.0 && target_std_native.is_finite()) {
        bail!("CCS physics training labels require a finite positive standard deviation");
    }

    let baseline = FoundationCcsPhysicsBaselineConfig {
        coefficients_native,
        target_mean_native,
        target_std_native,
    };
    let train_metrics =
        evaluate_foundation_ccs_physics_baseline(records, &labelled_indices, &baseline)?;
    let feature_summaries = FOUNDATION_CCS_PHYSICS_FEATURE_NAMES
        .iter()
        .zip(feature_stats)
        .map(|(&name, stats)| FoundationCcsPhysicsFeatureSummary {
            name: name.to_string(),
            min: stats.min.unwrap_or(f64::NAN),
            mean: stats.mean().unwrap_or(f64::NAN),
            standard_deviation: stats.population_std().unwrap_or(f64::NAN),
            max: stats.max.unwrap_or(f64::NAN),
        })
        .collect();

    Ok(FoundationCcsPhysicsFitResult {
        baseline,
        train_label_count: labelled_indices.len(),
        ridge_lambda: config.ridge_lambda,
        train_metrics,
        feature_summaries,
    })
}

/// Evaluate a frozen physical prior on the supplied original record indices.
pub fn evaluate_foundation_ccs_physics_baseline(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    baseline: &FoundationCcsPhysicsBaselineConfig,
) -> Result<FoundationCcsPhysicsMetrics> {
    let mut target_stats = RunningStats::default();
    let mut prediction_stats = RunningStats::default();
    let mut absolute_error_sum = 0.0f64;
    let mut squared_error_sum = 0.0f64;
    let mut target_prediction_product_sum = 0.0f64;

    for &index in indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow!("CCS physics evaluation record index {index} is out of bounds")
        })?;
        let Some(target) = record.ccs.filter(|value| value.is_finite()) else {
            continue;
        };
        let target = f64::from(target);
        let features = foundation_ccs_physics_features(&record.peptidoform, &record.context);
        if features.iter().any(|value| !value.is_finite()) {
            bail!("CCS physics evaluation encountered non-finite features at record {index}");
        }
        let prediction = dot(&baseline.coefficients_native, &features);
        if !prediction.is_finite() {
            bail!("CCS physics baseline produced a non-finite prediction at record {index}");
        }
        let error = prediction - target;
        absolute_error_sum += error.abs();
        squared_error_sum += error * error;
        target_prediction_product_sum += target * prediction;
        target_stats.push(target);
        prediction_stats.push(prediction);
    }

    let count = target_stats.count;
    if count == 0 {
        return Ok(FoundationCcsPhysicsMetrics::default());
    }
    let n = count as f64;
    let target_mean = target_stats.mean().unwrap_or(0.0);
    let prediction_mean = prediction_stats.mean().unwrap_or(0.0);
    let target_ss = target_stats.m2;
    let prediction_ss = prediction_stats.m2;
    let covariance = target_prediction_product_sum - n * target_mean * prediction_mean;
    let pearson_r = if target_ss > 1e-12 && prediction_ss > 1e-12 {
        Some(covariance / (target_ss * prediction_ss).sqrt())
    } else {
        None
    };
    let r_squared = if target_ss > 1e-12 {
        Some(1.0 - squared_error_sum / target_ss)
    } else {
        None
    };

    Ok(FoundationCcsPhysicsMetrics {
        label_count: count,
        mae_native: Some(absolute_error_sum / n),
        rmse_native: Some((squared_error_sum / n).sqrt()),
        r_squared,
        pearson_r,
        target_mean_native: Some(target_mean),
        prediction_mean_native: Some(prediction_mean),
        target_std_native: target_stats.population_std(),
        prediction_std_native: prediction_stats.population_std(),
    })
}

fn dot(
    coefficients: &[f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT],
    features: &[f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT],
) -> f64 {
    coefficients
        .iter()
        .zip(features)
        .map(|(coefficient, feature)| *coefficient * *feature)
        .sum()
}

fn solve_linear_system(
    mut matrix: [[f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT]; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT],
    mut rhs: [f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT],
) -> Result<[f64; FOUNDATION_CCS_PHYSICS_FEATURE_COUNT]> {
    for pivot in 0..FOUNDATION_CCS_PHYSICS_FEATURE_COUNT {
        let best = (pivot..FOUNDATION_CCS_PHYSICS_FEATURE_COUNT)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .total_cmp(&matrix[right][pivot].abs())
            })
            .unwrap_or(pivot);
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);

        let diagonal = matrix[pivot][pivot];
        if !diagonal.is_finite() || diagonal.abs() < 1e-12 {
            bail!("CCS physics ridge system is numerically singular at column {pivot}");
        }
        for col in pivot..FOUNDATION_CCS_PHYSICS_FEATURE_COUNT {
            matrix[pivot][col] /= diagonal;
        }
        rhs[pivot] /= diagonal;

        for row in 0..FOUNDATION_CCS_PHYSICS_FEATURE_COUNT {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..FOUNDATION_CCS_PHYSICS_FEATURE_COUNT {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl RunningStats {
    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = Some(self.max.map_or(value, |current| current.max(value)));
    }

    fn mean(self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    fn population_std(self) -> Option<f64> {
        (self.count > 0).then_some((self.m2 / self.count as f64).max(0.0).sqrt())
    }
}
