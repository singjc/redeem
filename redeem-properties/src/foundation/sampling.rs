//! Deterministic bounded/source-aware sampling for large multi-source corpora.
//!
//! Full-corpus epochs are desirable for final training, but they are expensive
//! while validating a new model/training stack on CPU. This module adds a
//! reproducible bounded-epoch mode and an optional source-weighted mixture.
//! Validation subsampling is always without replacement and fixed across epochs
//! so smoke-run metrics remain comparable.

use super::control::FoundationSplitMix64;
use super::corpus::FoundationRecordProvenance;
use super::data::FoundationTrainingRecord;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// How training records are selected for one epoch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FoundationSamplingStrategy {
    /// Shuffle/cap the materialized train partition without replacement.
    #[default]
    UniformRecords,
    /// Draw a deterministic weighted mixture of corpus sources. Individual
    /// sources are cycled with reshuffling when their requested quota exceeds
    /// their available training records.
    SourceWeighted,
}

/// Large-corpus sampling controls stored with trainer configuration/checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationSamplingConfig {
    /// Record-selection strategy for the training partition.
    pub strategy: FoundationSamplingStrategy,
    /// Optional hard cap on optimizer steps per training epoch.
    pub train_steps_per_epoch: Option<usize>,
    /// Optional hard cap on validation batches. The same deterministic subset
    /// is evaluated every epoch.
    pub validation_steps: Option<usize>,
    /// Per-source mixture weights for `source-weighted` training. When this map
    /// is non-empty it must contain every source represented in the selected
    /// train partition. Values may be zero to intentionally exclude a source.
    pub source_weights: BTreeMap<String, f64>,
    /// Optional source weights for a bounded validation subset. Used only when
    /// `validation_steps` is set; validation remains without replacement.
    pub validation_source_weights: BTreeMap<String, f64>,
    /// Report final validation metrics separately for each represented source.
    /// These diagnostics do not affect checkpoint selection or early stopping.
    pub report_validation_by_source: bool,
}

impl Default for FoundationSamplingConfig {
    fn default() -> Self {
        Self {
            strategy: FoundationSamplingStrategy::UniformRecords,
            train_steps_per_epoch: None,
            validation_steps: None,
            source_weights: BTreeMap::new(),
            validation_source_weights: BTreeMap::new(),
            report_validation_by_source: false,
        }
    }
}

impl FoundationSamplingConfig {
    /// Validate source-independent settings.
    pub fn validate(&self) -> Result<()> {
        if self.train_steps_per_epoch == Some(0) {
            anyhow::bail!("foundation train_steps_per_epoch must be at least 1 when set");
        }
        if self.validation_steps == Some(0) {
            anyhow::bail!("foundation validation_steps must be at least 1 when set");
        }
        validate_weight_map(&self.source_weights, "source")?;
        validate_weight_map(&self.validation_source_weights, "validation source")?;
        Ok(())
    }
}

/// Label/context coverage of the records selected for one pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoundationSampleCoverage {
    /// Selected records carrying normalized RT/iRT.
    pub normalized_rt_records: usize,
    /// Selected records carrying observed/raw RT.
    pub observed_rt_records: usize,
    /// Selected records carrying CCS.
    pub ccs_records: usize,
    /// Selected records carrying at least one supported MS2 fragment.
    pub ms2_records: usize,
    /// Selected records carrying precursor charge.
    pub charge_records: usize,
    /// Selected records carrying NCE.
    pub nce_records: usize,
    /// Selected records carrying instrument identity.
    pub instrument_records: usize,
}

/// Auditable index-selection result for one epoch/validation pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoundationSamplePlan {
    /// Ordered record indices consumed by the trainer.
    pub indices: Vec<usize>,
    /// Number of distinct corpus records represented in `indices`.
    pub unique_records: usize,
    /// Selected records per logical corpus source.
    pub source_records: BTreeMap<String, usize>,
    /// Supervision/context coverage in the selected records.
    pub coverage: FoundationSampleCoverage,
}

/// Build one deterministic training order from a materialized benchmark.
pub fn sample_foundation_training_indices(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    train_indices: &[usize],
    batch_size: usize,
    epoch: u64,
    seed: u64,
    shuffle: bool,
    config: &FoundationSamplingConfig,
) -> Result<FoundationSamplePlan> {
    config.validate()?;
    validate_selection_inputs(records, provenance, train_indices, batch_size)?;
    let max_records = config
        .train_steps_per_epoch
        .map(|steps| steps.saturating_mul(batch_size));

    let indices = match config.strategy {
        FoundationSamplingStrategy::UniformRecords => {
            let mut order = train_indices.to_vec();
            if shuffle {
                let mut rng = FoundationSplitMix64::new(
                    seed.wrapping_add(0x5341_4d50_4c45_0000).wrapping_add(epoch),
                );
                rng.shuffle(&mut order);
            }
            if let Some(limit) = max_records {
                order.truncate(limit.min(order.len()));
            }
            order
        }
        FoundationSamplingStrategy::SourceWeighted => source_weighted_indices(
            provenance,
            train_indices,
            max_records.unwrap_or(train_indices.len()),
            epoch,
            seed,
            &config.source_weights,
        )?,
    };

    summarize_plan(records, provenance, indices)
}

/// Select a fixed deterministic validation subset. This never samples with
/// replacement and deliberately ignores training source weights.
pub fn sample_foundation_validation_indices(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    validation_indices: &[usize],
    batch_size: usize,
    seed: u64,
    config: &FoundationSamplingConfig,
) -> Result<FoundationSamplePlan> {
    config.validate()?;
    validate_selection_inputs(records, provenance, validation_indices, batch_size)?;
    let mut order = validation_indices.to_vec();
    if let Some(steps) = config.validation_steps {
        let limit = steps.saturating_mul(batch_size).min(order.len());
        if config.validation_source_weights.is_empty() {
            // Corpus records are stored source-by-source, so taking the first N
            // validation records would be source-biased. Shuffle once with a fixed
            // seed and then truncate; the same subset is reused every epoch.
            let mut rng = FoundationSplitMix64::new(seed.wrapping_add(0x5641_4c49_4441_5445));
            rng.shuffle(&mut order);
            order.truncate(limit);
        } else {
            order = source_weighted_validation_indices(
                provenance,
                validation_indices,
                limit,
                seed,
                &config.validation_source_weights,
            )?;
        }
    }
    summarize_plan(records, provenance, order)
}

fn source_weighted_validation_indices(
    provenance: &[FoundationRecordProvenance],
    validation_indices: &[usize],
    target_records: usize,
    seed: u64,
    source_weights: &BTreeMap<String, f64>,
) -> Result<Vec<usize>> {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for &index in validation_indices {
        let source = provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation provenance index {index} is out of bounds")
        })?;
        groups
            .entry(source.source_id.clone())
            .or_default()
            .push(index);
    }
    validate_weights_against_groups(&groups, source_weights, "validation")?;
    let total_weight: f64 = source_weights.values().copied().sum();
    if !(total_weight > 0.0 && total_weight.is_finite()) {
        anyhow::bail!("foundation validation source weights require at least one positive weight");
    }
    let quotas = weighted_quotas(target_records, source_weights, total_weight);
    let mut sampled = Vec::with_capacity(target_records);
    for (source_ordinal, (source, pool)) in groups.iter().enumerate() {
        let desired = *quotas.get(source).unwrap_or(&0);
        if desired > pool.len() {
            anyhow::bail!(
                "foundation validation source '{source}' quota {desired} exceeds its available {} records; reduce validation_steps or its validation_source_weight",
                pool.len()
            );
        }
        if desired == 0 {
            continue;
        }
        let mut shuffled = pool.clone();
        let mut rng = FoundationSplitMix64::new(
            seed.wrapping_add(0x5641_4c53_4f55_5243)
                .wrapping_add((source_ordinal as u64).rotate_left(23)),
        );
        rng.shuffle(&mut shuffled);
        sampled.extend_from_slice(&shuffled[..desired]);
    }
    let mut rng = FoundationSplitMix64::new(seed.wrapping_add(0x5641_4c4d_4958_0000));
    rng.shuffle(&mut sampled);
    Ok(sampled)
}

fn validate_weight_map(weights: &BTreeMap<String, f64>, label: &str) -> Result<()> {
    for (source, weight) in weights {
        if source.trim().is_empty() {
            anyhow::bail!("foundation {label}-weight key cannot be empty");
        }
        if !(*weight >= 0.0 && weight.is_finite()) {
            anyhow::bail!(
                "foundation {label} weight for '{source}' must be finite and non-negative"
            );
        }
    }
    Ok(())
}

fn validate_weights_against_groups(
    groups: &BTreeMap<String, Vec<usize>>,
    source_weights: &BTreeMap<String, f64>,
    label: &str,
) -> Result<()> {
    for source in groups.keys() {
        if !source_weights.contains_key(source) {
            anyhow::bail!("foundation {label} source weights are missing source '{source}'");
        }
    }
    for source in source_weights.keys() {
        if !groups.contains_key(source) {
            anyhow::bail!(
                "foundation {label} source weight refers to '{source}', which has no selected records"
            );
        }
    }
    Ok(())
}

fn source_weighted_indices(
    provenance: &[FoundationRecordProvenance],
    train_indices: &[usize],
    target_records: usize,
    epoch: u64,
    seed: u64,
    source_weights: &BTreeMap<String, f64>,
) -> Result<Vec<usize>> {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for &index in train_indices {
        let source = provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation provenance index {index} is out of bounds")
        })?;
        groups
            .entry(source.source_id.clone())
            .or_default()
            .push(index);
    }
    if groups.is_empty() {
        anyhow::bail!("foundation source-weighted sampler received no source groups");
    }

    let weights = if source_weights.is_empty() {
        groups
            .keys()
            .map(|source| (source.clone(), 1.0f64))
            .collect::<BTreeMap<_, _>>()
    } else {
        validate_weights_against_groups(&groups, source_weights, "training")?;
        source_weights.clone()
    };
    let total_weight: f64 = weights.values().copied().sum();
    if !(total_weight > 0.0 && total_weight.is_finite()) {
        anyhow::bail!("foundation source-weighted sampler requires at least one positive weight");
    }

    let quotas = weighted_quotas(target_records, &weights, total_weight);
    let mut sampled = Vec::<usize>::with_capacity(target_records);
    for (source_ordinal, (source, pool)) in groups.iter().enumerate() {
        let desired = *quotas.get(source).unwrap_or(&0);
        if desired == 0 {
            continue;
        }
        if pool.is_empty() {
            anyhow::bail!("foundation source '{source}' has an empty training pool");
        }
        let mut remaining = desired;
        let mut cycle = 0u64;
        while remaining > 0 {
            let mut cycle_pool = pool.clone();
            let mut rng = FoundationSplitMix64::new(
                seed.wrapping_add(0x534f_5552_4345_0000)
                    .wrapping_add(epoch.rotate_left(17))
                    .wrapping_add((source_ordinal as u64).rotate_left(29))
                    .wrapping_add(cycle),
            );
            rng.shuffle(&mut cycle_pool);
            let take = remaining.min(cycle_pool.len());
            sampled.extend_from_slice(&cycle_pool[..take]);
            remaining -= take;
            cycle = cycle.saturating_add(1);
        }
    }
    let mut rng =
        FoundationSplitMix64::new(seed.wrapping_add(0x4d49_5854_5552_4500).wrapping_add(epoch));
    rng.shuffle(&mut sampled);
    Ok(sampled)
}

fn weighted_quotas(
    target_records: usize,
    weights: &BTreeMap<String, f64>,
    total_weight: f64,
) -> BTreeMap<String, usize> {
    let mut quotas = BTreeMap::<String, usize>::new();
    let mut fractions = Vec::<(f64, String)>::new();
    let mut assigned = 0usize;
    for (source, weight) in weights {
        let exact = target_records as f64 * *weight / total_weight;
        let base = exact.floor() as usize;
        assigned = assigned.saturating_add(base);
        quotas.insert(source.clone(), base);
        fractions.push((exact - base as f64, source.clone()));
    }
    fractions.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    let mut remaining = target_records.saturating_sub(assigned);
    for (_, source) in fractions {
        if remaining == 0 {
            break;
        }
        *quotas.entry(source).or_default() += 1;
        remaining -= 1;
    }
    quotas
}

fn validate_selection_inputs(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 {
        anyhow::bail!("foundation sampling batch_size must be greater than zero");
    }
    if records.len() != provenance.len() {
        anyhow::bail!(
            "foundation record/provenance length mismatch: {} records, {} provenance entries",
            records.len(),
            provenance.len()
        );
    }
    if indices.is_empty() {
        anyhow::bail!("foundation sampler cannot select from an empty partition");
    }
    if let Some(index) = indices
        .iter()
        .copied()
        .find(|index| *index >= records.len())
    {
        anyhow::bail!(
            "foundation sampling index {index} is out of bounds for {} records",
            records.len()
        );
    }
    Ok(())
}

fn summarize_plan(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: Vec<usize>,
) -> Result<FoundationSamplePlan> {
    let mut source_records = BTreeMap::<String, usize>::new();
    let mut unique = BTreeSet::<usize>::new();
    let mut coverage = FoundationSampleCoverage::default();
    for &index in &indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation sampled record index {index} is out of bounds")
        })?;
        let source = provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation sampled provenance index {index} is out of bounds")
        })?;
        *source_records.entry(source.source_id.clone()).or_default() += 1;
        unique.insert(index);
        coverage.normalized_rt_records += usize::from(record.retention_time.normalized.is_some());
        coverage.observed_rt_records +=
            usize::from(record.retention_time.observed_seconds.is_some());
        coverage.ccs_records += usize::from(record.ccs.is_some());
        coverage.ms2_records += usize::from(!record.fragments.is_empty());
        coverage.charge_records += usize::from(record.context.charge.is_some());
        coverage.nce_records += usize::from(record.context.nce.is_some());
        coverage.instrument_records += usize::from(record.context.instrument_id.is_some());
    }
    Ok(FoundationSamplePlan {
        indices,
        unique_records: unique.len(),
        source_records,
        coverage,
    })
}
