//! Train-only cross-source retention-time harmonization.
//!
//! Public spectral libraries frequently expose fields named normalized RT/iRT that are only
//! normalized within the producing workflow. Their numerical coordinates are therefore not
//! guaranteed to be interchangeable across sources. This module fits one robust affine map per
//! source from shared TRAIN-partition peptidoforms while preserving the raw source-native label.
//!
//! The portable peptide RT head should predict the harmonized latent coordinate. Source-specific
//! transforms are target semantics/provenance only and are never injected into the peptide encoder.

use super::{
    FoundationBenchmarkManifest, FoundationPartition, FoundationRecordProvenance,
    FoundationTrainingRecord,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Source-native RT -> common latent RT affine transform.
///
/// `harmonized = scale * source_native + offset`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationRtHarmonizationTransform {
    /// Positive multiplicative scale from source-native RT to the common latent coordinate.
    pub scale: f64,
    /// Additive offset from source-native RT to the common latent coordinate.
    pub offset: f64,
    /// Stable fit identifier shared by all transforms produced in one calibration.
    pub calibration_id: String,
}

impl FoundationRtHarmonizationTransform {
    /// Validate transform parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.scale > 0.0 && self.scale.is_finite()) {
            anyhow::bail!("RT harmonization scale must be positive and finite");
        }
        if !self.offset.is_finite() {
            anyhow::bail!("RT harmonization offset must be finite");
        }
        if self.calibration_id.trim().is_empty() {
            anyhow::bail!("RT harmonization calibration_id cannot be empty");
        }
        Ok(())
    }

    /// Map source-native RT to the common latent coordinate.
    pub fn harmonize(&self, source_native: f32) -> Option<f32> {
        if !source_native.is_finite() {
            return None;
        }
        let value = self.scale * f64::from(source_native) + self.offset;
        let value = value as f32;
        value.is_finite().then_some(value)
    }

    /// Map a common-latent RT prediction back to the source-native coordinate.
    pub fn source_native(&self, harmonized: f32) -> Option<f32> {
        if !harmonized.is_finite() {
            return None;
        }
        let value = (f64::from(harmonized) - self.offset) / self.scale;
        let value = value as f32;
        value.is_finite().then_some(value)
    }
}

/// Robust train-only calibration controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationRtHarmonizationFitConfig {
    /// Minimum shared TRAIN peptidoforms required to connect two sources in the overlap graph.
    pub min_pair_overlap: usize,
    /// Minimum shared TRAIN peptidoforms required for each fitted source.
    pub min_source_shared_peptides: usize,
    /// Maximum alternating robust-regression iterations.
    pub max_iterations: usize,
    /// Relative latent-coordinate convergence threshold.
    pub convergence_tolerance: f64,
    /// Huber transition in robust residual-scale units.
    pub huber_delta: f64,
    /// Number of IRLS updates for each per-source affine regression.
    pub irls_iterations: usize,
    /// Human-readable center of the final common latent RT coordinate.
    pub canonical_center: f64,
    /// Robust standard-deviation unit of the final common latent RT coordinate.
    pub canonical_scale: f64,
}

impl Default for FoundationRtHarmonizationFitConfig {
    fn default() -> Self {
        Self {
            min_pair_overlap: 10,
            min_source_shared_peptides: 25,
            max_iterations: 50,
            convergence_tolerance: 1.0e-6,
            huber_delta: 1.5,
            irls_iterations: 8,
            canonical_center: 50.0,
            canonical_scale: 25.0,
        }
    }
}

impl FoundationRtHarmonizationFitConfig {
    /// Validate numerical controls.
    pub fn validate(&self) -> Result<()> {
        if self.min_pair_overlap == 0 || self.min_source_shared_peptides < 2 {
            anyhow::bail!("RT harmonization overlap thresholds must be positive");
        }
        if self.max_iterations == 0 || self.irls_iterations == 0 {
            anyhow::bail!("RT harmonization iteration counts must be positive");
        }
        if !(self.convergence_tolerance > 0.0 && self.convergence_tolerance.is_finite()) {
            anyhow::bail!("RT harmonization convergence_tolerance must be positive and finite");
        }
        if !(self.huber_delta > 0.0 && self.huber_delta.is_finite()) {
            anyhow::bail!("RT harmonization huber_delta must be positive and finite");
        }
        if !self.canonical_center.is_finite()
            || !(self.canonical_scale > 0.0 && self.canonical_scale.is_finite())
        {
            anyhow::bail!(
                "RT harmonization canonical center/scale must be finite with positive scale"
            );
        }
        Ok(())
    }
}

/// Cross-source consistency on one benchmark partition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct FoundationRtCrossSourceConsistency {
    /// Peptidoforms observed with finite RT in at least two fitted sources.
    pub shared_peptidoforms: usize,
    /// Source-peptidoform observations contributing after within-source aggregation.
    pub observations: usize,
    /// Mean absolute deviation from the per-peptidoform cross-source median.
    pub mean_absolute_deviation: Option<f64>,
    /// Root mean squared deviation from the per-peptidoform cross-source median.
    pub root_mean_squared_deviation: Option<f64>,
}

/// Per-source fit/audit information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationRtSourceCalibrationSummary {
    /// Source identifier.
    pub source_id: String,
    /// TRAIN RT-labelled records before within-source peptidoform aggregation.
    pub train_rt_records: usize,
    /// Distinct TRAIN peptidoforms carrying RT in this source.
    pub train_rt_peptidoforms: usize,
    /// Distinct TRAIN peptidoforms shared with at least one fitted source.
    pub shared_train_peptidoforms: usize,
    /// Source-native median on the shared TRAIN overlap used for initialization/audit.
    pub source_native_median: f64,
    /// Robust source-native scale on the shared TRAIN overlap.
    pub source_native_robust_scale: f64,
    /// Learned source-native -> latent transform.
    pub transform: FoundationRtHarmonizationTransform,
}

/// Complete train-only RT harmonization fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationRtHarmonizationFitResult {
    /// Stable fit identifier.
    pub calibration_id: String,
    /// Fit configuration.
    pub config: FoundationRtHarmonizationFitConfig,
    /// Number of alternating updates performed.
    pub iterations: usize,
    /// Whether the latent coordinate met the convergence threshold.
    pub converged: bool,
    /// Sources carrying RT labels in the materialized TRAIN partition.
    pub rt_labelled_sources: Vec<String>,
    /// Connected RT sources included in the fit.
    pub fitted_sources: Vec<String>,
    /// Per-source transforms and audit statistics.
    pub sources: BTreeMap<String, FoundationRtSourceCalibrationSummary>,
    /// Distribution-only source standardization consistency on TRAIN.
    pub train_distribution_baseline: FoundationRtCrossSourceConsistency,
    /// Learned overlap-based consistency on TRAIN.
    pub train_harmonized: FoundationRtCrossSourceConsistency,
    /// Distribution-only source standardization consistency on VALIDATION.
    pub validation_distribution_baseline: FoundationRtCrossSourceConsistency,
    /// Learned overlap-based consistency on VALIDATION.
    pub validation_harmonized: FoundationRtCrossSourceConsistency,
}

#[derive(Debug, Clone, Copy)]
struct AffineRawFromLatent {
    slope: f64,
    intercept: f64,
}

#[derive(Debug, Clone, Copy)]
struct InitialSourceScale {
    median: f64,
    robust_scale: f64,
}

/// Fit robust source-specific affine RT transforms strictly from TRAIN records.
pub fn fit_foundation_rt_harmonization(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    benchmark: &FoundationBenchmarkManifest,
    config: FoundationRtHarmonizationFitConfig,
) -> Result<FoundationRtHarmonizationFitResult> {
    config.validate()?;
    if records.len() != provenance.len() {
        anyhow::bail!("RT harmonization record/provenance lengths differ");
    }
    benchmark.validate_against_records(records)?;

    let train = aggregate_partition(records, provenance, benchmark, FoundationPartition::Train)?;
    let validation = aggregate_partition(
        records,
        provenance,
        benchmark,
        FoundationPartition::Validation,
    )?;
    let rt_labelled_sources: Vec<String> = train
        .iter()
        .filter(|(_, peptides)| !peptides.is_empty())
        .map(|(source, _)| source.clone())
        .collect();
    if rt_labelled_sources.len() < 2 {
        anyhow::bail!("RT harmonization requires at least two TRAIN sources with RT labels");
    }

    let overlap_counts = source_pair_overlap_counts(&train);
    let fitted_sources = connected_rt_sources(
        &rt_labelled_sources,
        &overlap_counts,
        config.min_pair_overlap,
    )?;
    if fitted_sources.len() != rt_labelled_sources.len() {
        let fitted: BTreeSet<&String> = fitted_sources.iter().collect();
        let excluded: Vec<&String> = rt_labelled_sources
            .iter()
            .filter(|source| !fitted.contains(source))
            .collect();
        anyhow::bail!(
            "RT harmonization overlap graph is not connected at min_pair_overlap={}; disconnected RT sources: {:?}",
            config.min_pair_overlap,
            excluded
        );
    }

    let fitted_set: BTreeSet<String> = fitted_sources.iter().cloned().collect();
    let validation_rt_sources: Vec<String> = validation
        .iter()
        .filter(|(_, peptides)| !peptides.is_empty())
        .map(|(source, _)| source.clone())
        .collect();
    let missing_validation_sources: Vec<String> = validation_rt_sources
        .into_iter()
        .filter(|source| !fitted_set.contains(source))
        .collect();
    if !missing_validation_sources.is_empty() {
        anyhow::bail!(
            "RT harmonization validation contains sources without TRAIN-fit transforms: {:?}",
            missing_validation_sources
        );
    }
    let shared_peptides = shared_peptides_by_source(&train, &fitted_set);
    for source in &fitted_sources {
        let shared = shared_peptides.get(source).map_or(0, BTreeSet::len);
        if shared < config.min_source_shared_peptides {
            anyhow::bail!(
                "RT source '{}' has only {} shared TRAIN peptidoforms; {} required",
                source,
                shared,
                config.min_source_shared_peptides
            );
        }
    }

    let initial_scales = initial_source_scales(&train, &shared_peptides, &config)?;
    let mut latent = initialize_latent(&train, &fitted_set, &initial_scales)?;
    gauge_fix_latent_only(&mut latent)?;

    let mut source_models = BTreeMap::<String, AffineRawFromLatent>::new();
    let mut converged = false;
    let mut completed_iterations = 0usize;
    for iteration in 0..config.max_iterations {
        let mut next_models = BTreeMap::new();
        for source in &fitted_sources {
            let peptides = train
                .get(source)
                .ok_or_else(|| anyhow!("missing source {source}"))?;
            let pairs: Vec<(f64, f64)> = peptides
                .iter()
                .filter_map(|(peptide, &raw)| latent.get(peptide).map(|&z| (z, raw)))
                .collect();
            if pairs.len() < config.min_source_shared_peptides {
                anyhow::bail!(
                    "RT source '{source}' lost required shared-peptide support during fit"
                );
            }
            let model = robust_linear_fit(&pairs, config.huber_delta, config.irls_iterations)?;
            if !(model.slope > 1.0e-10 && model.slope.is_finite() && model.intercept.is_finite()) {
                anyhow::bail!("RT source '{source}' produced invalid/non-positive affine slope");
            }
            next_models.insert(source.clone(), model);
        }

        let mut next_latent = BTreeMap::<String, f64>::new();
        let peptide_sources = invert_source_peptides(&train, &fitted_set);
        for (peptide, observations) in peptide_sources {
            if observations.len() < 2 {
                continue;
            }
            let mut estimates = Vec::with_capacity(observations.len());
            for (source, raw) in observations {
                if let Some(model) = next_models.get(&source) {
                    estimates.push((raw - model.intercept) / model.slope);
                }
            }
            if estimates.len() >= 2 {
                next_latent.insert(peptide, median(&mut estimates)?);
            }
        }
        if next_latent.is_empty() {
            anyhow::bail!("RT harmonization latent update produced no shared peptides");
        }
        gauge_fix_latent_and_models(&mut next_latent, &mut next_models)?;

        let delta = relative_latent_change(&latent, &next_latent);
        latent = next_latent;
        source_models = next_models;
        completed_iterations = iteration + 1;
        if delta <= config.convergence_tolerance {
            converged = true;
            break;
        }
    }
    if source_models.is_empty() {
        anyhow::bail!("RT harmonization did not produce source models");
    }

    let transform_bits: Vec<(String, u64, u64)> = fitted_sources
        .iter()
        .map(|source| {
            let model = source_models[source];
            let scale = config.canonical_scale / model.slope;
            let offset =
                config.canonical_center - config.canonical_scale * model.intercept / model.slope;
            (source.clone(), scale.to_bits(), offset.to_bits())
        })
        .collect();
    let calibration_id = calibration_fingerprint(benchmark, &config, &transform_bits);

    let mut source_summaries = BTreeMap::new();
    let mut transforms = BTreeMap::new();
    for source in &fitted_sources {
        let model = source_models[source];
        let transform = FoundationRtHarmonizationTransform {
            scale: config.canonical_scale / model.slope,
            offset: config.canonical_center
                - config.canonical_scale * model.intercept / model.slope,
            calibration_id: calibration_id.clone(),
        };
        transform.validate()?;
        transforms.insert(source.clone(), transform.clone());
        let train_rt_records = train_record_count(records, provenance, benchmark, source);
        let train_rt_peptidoforms = train.get(source).map_or(0, BTreeMap::len);
        let shared_train_peptidoforms = shared_peptides.get(source).map_or(0, BTreeSet::len);
        let initial = initial_scales[source];
        source_summaries.insert(
            source.clone(),
            FoundationRtSourceCalibrationSummary {
                source_id: source.clone(),
                train_rt_records,
                train_rt_peptidoforms,
                shared_train_peptidoforms,
                source_native_median: initial.median,
                source_native_robust_scale: initial.robust_scale,
                transform,
            },
        );
    }

    let baseline_transforms: BTreeMap<String, FoundationRtHarmonizationTransform> = initial_scales
        .iter()
        .map(|(source, initial)| {
            (
                source.clone(),
                FoundationRtHarmonizationTransform {
                    scale: config.canonical_scale / initial.robust_scale,
                    offset: config.canonical_center
                        - config.canonical_scale * initial.median / initial.robust_scale,
                    calibration_id: "distribution-only-train-baseline".into(),
                },
            )
        })
        .collect();

    Ok(FoundationRtHarmonizationFitResult {
        calibration_id,
        config,
        iterations: completed_iterations,
        converged,
        rt_labelled_sources,
        fitted_sources,
        sources: source_summaries,
        train_distribution_baseline: cross_source_consistency(&train, &baseline_transforms),
        train_harmonized: cross_source_consistency(&train, &transforms),
        validation_distribution_baseline: cross_source_consistency(
            &validation,
            &baseline_transforms,
        ),
        validation_harmonized: cross_source_consistency(&validation, &transforms),
    })
}

/// Apply one already-fit transform to a training record while preserving source-native RT.
pub fn apply_foundation_rt_harmonization(
    record: &mut FoundationTrainingRecord,
    transform: &FoundationRtHarmonizationTransform,
) -> Result<()> {
    transform.validate()?;
    record.retention_time.harmonized = record
        .retention_time
        .normalized
        .and_then(|value| transform.harmonize(value));
    Ok(())
}

type AggregatedRt = BTreeMap<String, BTreeMap<String, f64>>;

fn aggregate_partition(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
) -> Result<AggregatedRt> {
    let mut values = BTreeMap::<String, BTreeMap<String, Vec<f64>>>::new();
    for entry in benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
    {
        let record = records
            .get(entry.record_index)
            .ok_or_else(|| anyhow!("RT harmonization benchmark record out of bounds"))?;
        let Some(raw) = record
            .retention_time
            .normalized
            .filter(|value| value.is_finite())
        else {
            continue;
        };
        let source = provenance
            .get(entry.record_index)
            .ok_or_else(|| anyhow!("RT harmonization provenance record out of bounds"))?
            .source_id
            .clone();
        values
            .entry(source)
            .or_default()
            .entry(entry.peptidoform.clone())
            .or_default()
            .push(f64::from(raw));
    }

    let mut aggregated = BTreeMap::new();
    for (source, peptides) in values {
        let mut out = BTreeMap::new();
        for (peptide, mut observations) in peptides {
            out.insert(peptide, median(&mut observations)?);
        }
        aggregated.insert(source, out);
    }
    Ok(aggregated)
}

fn source_pair_overlap_counts(data: &AggregatedRt) -> BTreeMap<(String, String), usize> {
    let mut peptide_sources = BTreeMap::<String, Vec<String>>::new();
    for (source, peptides) in data {
        for peptide in peptides.keys() {
            peptide_sources
                .entry(peptide.clone())
                .or_default()
                .push(source.clone());
        }
    }
    let mut counts = BTreeMap::new();
    for mut sources in peptide_sources.into_values() {
        sources.sort();
        sources.dedup();
        for left in 0..sources.len() {
            for right in (left + 1)..sources.len() {
                *counts
                    .entry((sources[left].clone(), sources[right].clone()))
                    .or_insert(0) += 1;
            }
        }
    }
    counts
}

fn connected_rt_sources(
    sources: &[String],
    overlaps: &BTreeMap<(String, String), usize>,
    min_overlap: usize,
) -> Result<Vec<String>> {
    let source_set: BTreeSet<String> = sources.iter().cloned().collect();
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    for source in sources {
        adjacency.entry(source.clone()).or_default();
    }
    for ((left, right), &count) in overlaps {
        if count >= min_overlap && source_set.contains(left) && source_set.contains(right) {
            adjacency
                .entry(left.clone())
                .or_default()
                .insert(right.clone());
            adjacency
                .entry(right.clone())
                .or_default()
                .insert(left.clone());
        }
    }

    let mut seen = BTreeSet::new();
    let mut components = Vec::<Vec<String>>::new();
    for source in sources {
        if seen.contains(source) {
            continue;
        }
        let mut queue = VecDeque::from([source.clone()]);
        let mut component = Vec::new();
        seen.insert(source.clone());
        while let Some(current) = queue.pop_front() {
            component.push(current.clone());
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if seen.insert(neighbor.clone()) {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        component.sort();
        components.push(component);
    }
    components.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    components
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("RT harmonization found no source-overlap component"))
}

fn shared_peptides_by_source(
    data: &AggregatedRt,
    fitted_sources: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut peptide_count = BTreeMap::<String, usize>::new();
    for (source, peptides) in data {
        if !fitted_sources.contains(source) {
            continue;
        }
        for peptide in peptides.keys() {
            *peptide_count.entry(peptide.clone()).or_insert(0) += 1;
        }
    }
    let mut output = BTreeMap::<String, BTreeSet<String>>::new();
    for (source, peptides) in data {
        if !fitted_sources.contains(source) {
            continue;
        }
        for peptide in peptides.keys() {
            if peptide_count.get(peptide).copied().unwrap_or_default() >= 2 {
                output
                    .entry(source.clone())
                    .or_default()
                    .insert(peptide.clone());
            }
        }
    }
    output
}

fn initial_source_scales(
    data: &AggregatedRt,
    shared: &BTreeMap<String, BTreeSet<String>>,
    config: &FoundationRtHarmonizationFitConfig,
) -> Result<BTreeMap<String, InitialSourceScale>> {
    let mut output = BTreeMap::new();
    for (source, peptides) in shared {
        let source_values = data
            .get(source)
            .ok_or_else(|| anyhow!("missing RT source {source}"))?;
        let mut values: Vec<f64> = peptides
            .iter()
            .filter_map(|peptide| source_values.get(peptide).copied())
            .collect();
        if values.len() < config.min_source_shared_peptides {
            anyhow::bail!("RT source '{source}' has insufficient shared values for initialization");
        }
        let med = median(&mut values)?;
        let scale = robust_scale(&values, med)?;
        output.insert(
            source.clone(),
            InitialSourceScale {
                median: med,
                robust_scale: scale,
            },
        );
    }
    Ok(output)
}

fn initialize_latent(
    data: &AggregatedRt,
    fitted_sources: &BTreeSet<String>,
    initial: &BTreeMap<String, InitialSourceScale>,
) -> Result<BTreeMap<String, f64>> {
    let peptide_sources = invert_source_peptides(data, fitted_sources);
    let mut latent = BTreeMap::new();
    for (peptide, observations) in peptide_sources {
        if observations.len() < 2 {
            continue;
        }
        let mut standardized = Vec::with_capacity(observations.len());
        for (source, raw) in observations {
            let init = initial
                .get(&source)
                .ok_or_else(|| anyhow!("missing RT source initialization for {source}"))?;
            standardized.push((raw - init.median) / init.robust_scale);
        }
        latent.insert(peptide, median(&mut standardized)?);
    }
    Ok(latent)
}

fn invert_source_peptides(
    data: &AggregatedRt,
    fitted_sources: &BTreeSet<String>,
) -> BTreeMap<String, Vec<(String, f64)>> {
    let mut output = BTreeMap::<String, Vec<(String, f64)>>::new();
    for (source, peptides) in data {
        if !fitted_sources.contains(source) {
            continue;
        }
        for (peptide, &raw) in peptides {
            output
                .entry(peptide.clone())
                .or_default()
                .push((source.clone(), raw));
        }
    }
    output
}

fn robust_linear_fit(
    pairs: &[(f64, f64)],
    huber_delta: f64,
    iterations: usize,
) -> Result<AffineRawFromLatent> {
    if pairs.len() < 2 {
        anyhow::bail!("robust affine RT fit requires at least two observations");
    }
    let mut weights = vec![1.0f64; pairs.len()];
    let mut model = weighted_linear_fit(pairs, &weights)?;
    for _ in 0..iterations {
        let mut abs_residuals: Vec<f64> = pairs
            .iter()
            .map(|&(x, y)| (y - (model.slope * x + model.intercept)).abs())
            .collect();
        let med_abs = median(&mut abs_residuals)?;
        let sigma = (1.4826 * med_abs).max(1.0e-8);
        let threshold = huber_delta * sigma;
        for (weight, &(x, y)) in weights.iter_mut().zip(pairs) {
            let residual = (y - (model.slope * x + model.intercept)).abs();
            *weight = if residual <= threshold || residual == 0.0 {
                1.0
            } else {
                threshold / residual
            };
        }
        model = weighted_linear_fit(pairs, &weights)?;
    }
    Ok(model)
}

fn weighted_linear_fit(pairs: &[(f64, f64)], weights: &[f64]) -> Result<AffineRawFromLatent> {
    if pairs.len() != weights.len() || pairs.is_empty() {
        anyhow::bail!("weighted affine RT fit received inconsistent arrays");
    }
    let weight_sum: f64 = weights.iter().sum();
    if !(weight_sum > 0.0 && weight_sum.is_finite()) {
        anyhow::bail!("weighted affine RT fit has invalid total weight");
    }
    let x_mean = pairs
        .iter()
        .zip(weights)
        .map(|(&(x, _), &w)| w * x)
        .sum::<f64>()
        / weight_sum;
    let y_mean = pairs
        .iter()
        .zip(weights)
        .map(|(&(_, y), &w)| w * y)
        .sum::<f64>()
        / weight_sum;
    let covariance = pairs
        .iter()
        .zip(weights)
        .map(|(&(x, y), &w)| w * (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    let variance = pairs
        .iter()
        .zip(weights)
        .map(|(&(x, _), &w)| w * (x - x_mean).powi(2))
        .sum::<f64>();
    if !(variance > 1.0e-12 && variance.is_finite()) {
        anyhow::bail!("weighted affine RT fit has degenerate latent variance");
    }
    let slope = covariance / variance;
    let intercept = y_mean - slope * x_mean;
    Ok(AffineRawFromLatent { slope, intercept })
}

fn gauge_fix_latent_only(latent: &mut BTreeMap<String, f64>) -> Result<()> {
    let mut values: Vec<f64> = latent.values().copied().collect();
    let med = median(&mut values)?;
    let scale = robust_scale(&values, med)?;
    for value in latent.values_mut() {
        *value = (*value - med) / scale;
    }
    Ok(())
}

fn gauge_fix_latent_and_models(
    latent: &mut BTreeMap<String, f64>,
    models: &mut BTreeMap<String, AffineRawFromLatent>,
) -> Result<()> {
    let mut values: Vec<f64> = latent.values().copied().collect();
    let med = median(&mut values)?;
    let scale = robust_scale(&values, med)?;
    for value in latent.values_mut() {
        *value = (*value - med) / scale;
    }
    for model in models.values_mut() {
        let old_slope = model.slope;
        model.slope = old_slope * scale;
        model.intercept += old_slope * med;
    }
    Ok(())
}

fn relative_latent_change(old: &BTreeMap<String, f64>, new: &BTreeMap<String, f64>) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    let mut count = 0usize;
    for (peptide, &value) in new {
        if let Some(&previous) = old.get(peptide) {
            numerator += (value - previous).powi(2);
            denominator += previous.powi(2);
            count += 1;
        }
    }
    if count == 0 {
        return f64::INFINITY;
    }
    (numerator / count as f64).sqrt() / ((denominator / count as f64).sqrt().max(1.0))
}

fn cross_source_consistency(
    data: &AggregatedRt,
    transforms: &BTreeMap<String, FoundationRtHarmonizationTransform>,
) -> FoundationRtCrossSourceConsistency {
    let fitted_set: BTreeSet<String> = transforms.keys().cloned().collect();
    let peptide_sources = invert_source_peptides(data, &fitted_set);
    let mut squared = 0.0f64;
    let mut absolute = 0.0f64;
    let mut observations = 0usize;
    let mut shared = 0usize;
    for (_peptide, raw_observations) in peptide_sources {
        let mut values: Vec<f64> = raw_observations
            .iter()
            .filter_map(|(source, raw)| {
                transforms
                    .get(source)
                    .map(|transform| transform.scale * raw + transform.offset)
            })
            .filter(|value| value.is_finite())
            .collect();
        if values.len() < 2 {
            continue;
        }
        let consensus = match median(&mut values) {
            Ok(value) => value,
            Err(_) => continue,
        };
        shared += 1;
        for value in values {
            let error = value - consensus;
            absolute += error.abs();
            squared += error * error;
            observations += 1;
        }
    }
    FoundationRtCrossSourceConsistency {
        shared_peptidoforms: shared,
        observations,
        mean_absolute_deviation: (observations > 0).then_some(absolute / observations as f64),
        root_mean_squared_deviation: (observations > 0)
            .then_some((squared / observations as f64).sqrt()),
    }
}

fn train_record_count(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    benchmark: &FoundationBenchmarkManifest,
    source_id: &str,
) -> usize {
    benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .filter(|entry| {
            provenance
                .get(entry.record_index)
                .is_some_and(|item| item.source_id == source_id)
        })
        .filter(|entry| {
            records
                .get(entry.record_index)
                .and_then(|record| record.retention_time.normalized)
                .is_some_and(f32::is_finite)
        })
        .count()
}

fn robust_scale(values: &[f64], center: f64) -> Result<f64> {
    if values.len() < 2 {
        anyhow::bail!("robust RT scale requires at least two values");
    }
    let mut deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    let mad = median(&mut deviations)?;
    let mad_scale = 1.4826 * mad;
    if mad_scale > 1.0e-8 && mad_scale.is_finite() {
        return Ok(mad_scale);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let sd = variance.sqrt();
    if sd > 1.0e-8 && sd.is_finite() {
        Ok(sd)
    } else {
        anyhow::bail!("RT values have degenerate robust scale")
    }
}

fn median(values: &mut [f64]) -> Result<f64> {
    if values.is_empty() {
        anyhow::bail!("median requires at least one value");
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

fn calibration_fingerprint(
    benchmark: &FoundationBenchmarkManifest,
    config: &FoundationRtHarmonizationFitConfig,
    transforms: &[(String, u64, u64)],
) -> String {
    let mut hash = StableFnv64::new();
    hash.str("foundation-rt-harmonization-v0136");
    hash.u64(benchmark.dataset_fingerprint);
    hash.u64(benchmark.manifest_fingerprint());
    hash.usize(config.min_pair_overlap);
    hash.usize(config.min_source_shared_peptides);
    hash.usize(config.max_iterations);
    hash.u64(config.convergence_tolerance.to_bits());
    hash.u64(config.huber_delta.to_bits());
    hash.usize(config.irls_iterations);
    hash.u64(config.canonical_center.to_bits());
    hash.u64(config.canonical_scale.to_bits());
    for (source, scale, offset) in transforms {
        hash.str(source);
        hash.u64(*scale);
        hash.u64(*offset);
    }
    format!("fnv1a64:{:016x}", hash.finish())
}

struct StableFnv64(u64);

impl StableFnv64 {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.bytes(value.as_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_transform_round_trips() {
        let transform = FoundationRtHarmonizationTransform {
            scale: 0.5,
            offset: 10.0,
            calibration_id: "test".into(),
        };
        let harmonized = transform.harmonize(80.0).unwrap();
        assert!((harmonized - 50.0).abs() < 1.0e-6);
        let source = transform.source_native(harmonized).unwrap();
        assert!((source - 80.0).abs() < 1.0e-6);
    }

    #[test]
    fn robust_affine_fit_recovers_positive_scale_with_outlier() {
        let mut pairs: Vec<(f64, f64)> = (0..100)
            .map(|index| {
                let x = (index as f64 - 50.0) / 10.0;
                (x, 2.5 * x + 17.0)
            })
            .collect();
        pairs.push((0.0, 500.0));
        let fit = robust_linear_fit(&pairs, 1.5, 12).unwrap();
        assert!((fit.slope - 2.5).abs() < 0.05, "slope={}", fit.slope);
        assert!(
            (fit.intercept - 17.0).abs() < 0.1,
            "intercept={}",
            fit.intercept
        );
    }
}
