//! Source-aware mobility consensus prepared exclusively from TRAIN identities.
//!
//! This is the v0.50/v0.52 source-affine, reliability-weighted consensus
//! mechanism used without the historical checkpoint dependency.

use super::corpus::FoundationRecordProvenance;
use super::data::FoundationTrainingRecord;
use super::featurize::FoundationModificationSite;
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};

const SOURCE_SHRINKAGE: f64 = 256.0;
const MIN_SOURCE_SHARED_IDENTITIES: usize = 20;
const RELIABILITY_FLOOR: f64 = 0.35;
const RELIABILITY_CEILING: f64 = 1.25;
const SINGLETON_WEIGHT_SCALE: f64 = 0.65;
const CONSENSUS_DISPERSION_SCALE: f64 = 0.010;

#[derive(Debug, Clone)]
struct MobilityObservation {
    record_index: usize,
    source_id: String,
    target: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct MobilityConsensusExample {
    pub(crate) representative_index: usize,
    pub(crate) target_mobility: f32,
    pub(crate) weight: f32,
    pub(crate) source_count: usize,
    pub(crate) identity_hash: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct AffineFit {
    intercept: f64,
    slope: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SourceSupervision {
    pub(crate) source_id: String,
    pub(crate) raw_records: usize,
    pub(crate) shared_identities: usize,
    affine: AffineFit,
    pub(crate) residual_mae: f64,
    pub(crate) reliability: f64,
}

impl SourceSupervision {
    pub(crate) fn affine_parameters(&self) -> (f64, f64) {
        (self.affine.intercept, self.affine.slope)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MobilityConsensusSupervision {
    pub(crate) examples: Vec<MobilityConsensusExample>,
    pub(crate) source_supervision: BTreeMap<String, SourceSupervision>,
    pub(crate) raw_records: usize,
    pub(crate) multisource_examples: usize,
    pub(crate) singleton_examples: usize,
    pub(crate) mean_weight: f64,
    pub(crate) mean_abs_adjusted_delta_to_consensus: f64,
    pub(crate) fingerprint: u64,
}

pub(crate) fn finite_mobility_ccs_indices(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
) -> Vec<usize> {
    indices
        .iter()
        .copied()
        .filter(|&index| {
            let record = &records[index];
            record
                .ccs
                .is_some_and(|value| value.is_finite() && value > 0.0)
                && record
                    .context
                    .ion_mobility
                    .is_some_and(|value| value.is_finite() && value > 0.0)
                && record.context.charge.is_some_and(|value| value > 0)
                && record
                    .context
                    .precursor_mz
                    .is_some_and(|value| value.is_finite() && value > 0.0)
        })
        .collect()
}

pub(crate) fn build_train_consensus_supervision(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
) -> Result<MobilityConsensusSupervision> {
    let groups = collect_mobility_groups(records, provenance, indices)?;
    let mut all_sources = BTreeSet::<String>::new();
    let mut raw_counts = BTreeMap::<String, usize>::new();
    let mut affine_examples = BTreeMap::<String, Vec<(f64, f64)>>::new();

    for observations in groups.values() {
        let source_means = source_means_with_representatives(observations);
        for (source, (_, count, _)) in &source_means {
            all_sources.insert(source.clone());
            *raw_counts.entry(source.clone()).or_insert(0) += *count;
        }
        if source_means.len() < 2 {
            continue;
        }
        let total = source_means
            .values()
            .map(|(value, _, _)| *value)
            .sum::<f64>();
        for (source, (source_mean, _, _)) in &source_means {
            let other_mean = (total - source_mean) / (source_means.len() - 1) as f64;
            affine_examples
                .entry(source.clone())
                .or_default()
                .push((*source_mean, other_mean));
        }
    }

    let mut source_supervision = BTreeMap::<String, SourceSupervision>::new();
    for source in all_sources {
        let examples = affine_examples
            .get(&source)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let (affine, shared_identities) = if examples.len() >= MIN_SOURCE_SHARED_IDENTITIES {
            let raw = fit_xy_affine(examples)?;
            let blend = examples.len() as f64 / (examples.len() as f64 + SOURCE_SHRINKAGE);
            (
                AffineFit {
                    intercept: (blend * raw.intercept).clamp(-0.08, 0.08),
                    slope: (1.0 + blend * (raw.slope - 1.0)).clamp(0.95, 1.05),
                },
                examples.len(),
            )
        } else {
            (
                AffineFit {
                    intercept: 0.0,
                    slope: 1.0,
                },
                examples.len(),
            )
        };
        source_supervision.insert(
            source.clone(),
            SourceSupervision {
                source_id: source.clone(),
                raw_records: raw_counts.get(&source).copied().unwrap_or(0),
                shared_identities,
                affine,
                residual_mae: 0.0,
                reliability: 1.0,
            },
        );
    }

    let mut source_residual_sum = BTreeMap::<String, f64>::new();
    let mut source_residual_count = BTreeMap::<String, usize>::new();
    for observations in groups.values() {
        let source_means = source_means_with_representatives(observations);
        if source_means.len() < 2 {
            continue;
        }
        let adjusted = source_means
            .iter()
            .map(|(source, (value, _, _))| {
                let fit = source_supervision
                    .get(source)
                    .map(|entry| entry.affine)
                    .unwrap_or(AffineFit {
                        intercept: 0.0,
                        slope: 1.0,
                    });
                (source.clone(), fit.intercept + fit.slope * value)
            })
            .collect::<Vec<_>>();
        for (source, value) in &adjusted {
            let mut others = adjusted
                .iter()
                .filter(|(other, _)| other != source)
                .map(|(_, other_value)| *other_value)
                .collect::<Vec<_>>();
            others.sort_by(|a, b| a.total_cmp(b));
            let reference = quantile_sorted(&others, 0.5);
            *source_residual_sum.entry(source.clone()).or_insert(0.0) += (value - reference).abs();
            *source_residual_count.entry(source.clone()).or_insert(0) += 1;
        }
    }

    let mut residual_maes = Vec::<f64>::new();
    for supervision in source_supervision.values_mut() {
        let count = source_residual_count
            .get(&supervision.source_id)
            .copied()
            .unwrap_or(0);
        supervision.residual_mae = if count > 0 {
            source_residual_sum
                .get(&supervision.source_id)
                .copied()
                .unwrap_or(0.0)
                / count as f64
        } else {
            0.0
        };
        if count >= MIN_SOURCE_SHARED_IDENTITIES && supervision.residual_mae.is_finite() {
            residual_maes.push(supervision.residual_mae);
        }
    }
    residual_maes.sort_by(|a, b| a.total_cmp(b));
    let global_residual_scale = if residual_maes.is_empty() {
        1.0
    } else {
        quantile_sorted(&residual_maes, 0.5).max(0.002)
    };
    for supervision in source_supervision.values_mut() {
        supervision.reliability = if supervision.shared_identities < MIN_SOURCE_SHARED_IDENTITIES {
            0.60
        } else {
            (2.0 / (1.0 + supervision.residual_mae / global_residual_scale))
                .clamp(RELIABILITY_FLOOR, RELIABILITY_CEILING)
        };
    }

    let (examples, multisource_examples, singleton_examples, mean_weight, mean_delta) =
        build_consensus_from_groups(&groups, &source_supervision)?;
    let fingerprint = consensus_fingerprint(&examples);
    Ok(MobilityConsensusSupervision {
        examples,
        source_supervision,
        raw_records: indices.len(),
        multisource_examples,
        singleton_examples,
        mean_weight,
        mean_abs_adjusted_delta_to_consensus: mean_delta,
        fingerprint,
    })
}

pub(crate) fn build_partition_consensus_examples(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
    source_supervision: &BTreeMap<String, SourceSupervision>,
) -> Result<Vec<MobilityConsensusExample>> {
    let groups = collect_mobility_groups(records, provenance, indices)?;
    let (examples, _, _, _, _) = build_consensus_from_groups(&groups, source_supervision)?;
    Ok(examples)
}

pub(crate) fn deterministic_example_order(
    examples: &[MobilityConsensusExample],
    needed: usize,
    epoch: u64,
    seed: u64,
) -> Vec<usize> {
    if examples.is_empty() || needed == 0 {
        return Vec::new();
    }
    let mut output = Vec::<usize>::with_capacity(needed);
    let mut cycle = 0u64;
    while output.len() < needed {
        let mut indices = (0..examples.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&index| {
            mix64(
                examples[index].identity_hash
                    ^ seed
                    ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ cycle.wrapping_mul(0xd1b5_4a32_d192_ed03),
            )
        });
        let take = (needed - output.len()).min(indices.len());
        output.extend_from_slice(&indices[..take]);
        cycle = cycle.wrapping_add(1);
    }
    output
}
pub(crate) fn bruker_ccs_factor_from_values(charge: i32, precursor_mz: f32) -> Option<f64> {
    if charge <= 0 || !precursor_mz.is_finite() || precursor_mz <= 0.0 {
        return None;
    }
    let charge_f = f64::from(charge);
    let mz = f64::from(precursor_mz);
    let neutral_mass = charge_f * mz;
    let reduced_mass = (28.0 * neutral_mass) / (28.0 + neutral_mass);
    if !(reduced_mass > 0.0 && reduced_mass.is_finite()) {
        return None;
    }
    Some(charge_f * 1059.62245 / reduced_mass.sqrt())
}
fn collect_mobility_groups(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    indices: &[usize],
) -> Result<BTreeMap<String, Vec<MobilityObservation>>> {
    let mut groups = BTreeMap::<String, Vec<MobilityObservation>>::new();
    for &index in indices {
        let record = records
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("mobility record index {index} outside corpus"))?;
        let Some(target) = record
            .context
            .ion_mobility
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("missing provenance for mobility record {index}"))?;
        groups
            .entry(peptidoform_charge_key(record))
            .or_default()
            .push(MobilityObservation {
                record_index: index,
                source_id: source.source_id.clone(),
                target: f64::from(target),
            });
    }
    Ok(groups)
}

fn source_means_with_representatives(
    observations: &[MobilityObservation],
) -> BTreeMap<String, (f64, usize, usize)> {
    let mut grouped = BTreeMap::<String, (f64, usize, usize)>::new();
    for observation in observations {
        let entry = grouped.entry(observation.source_id.clone()).or_insert((
            0.0,
            0,
            observation.record_index,
        ));
        entry.0 += observation.target;
        entry.1 += 1;
        entry.2 = entry.2.min(observation.record_index);
    }
    grouped
        .into_iter()
        .map(|(source, (sum, count, representative))| {
            (source, (sum / count as f64, count, representative))
        })
        .collect()
}

fn source_family(source_id: &str) -> String {
    if source_id.starts_with("pxd034128_") {
        "pxd034128".to_string()
    } else if source_id.starts_with("pxd058337_") {
        "pxd058337".to_string()
    } else {
        source_id.to_string()
    }
}

fn build_consensus_from_groups(
    groups: &BTreeMap<String, Vec<MobilityObservation>>,
    source_supervision: &BTreeMap<String, SourceSupervision>,
) -> Result<(Vec<MobilityConsensusExample>, usize, usize, f64, f64)> {
    let mut examples = Vec::<MobilityConsensusExample>::with_capacity(groups.len());
    let mut multisource_examples = 0usize;
    let mut singleton_examples = 0usize;
    let mut total_weight = 0.0f64;
    let mut total_abs_delta = 0.0f64;
    let mut total_abs_delta_count = 0usize;

    for (identity, observations) in groups {
        let source_means = source_means_with_representatives(observations);
        if source_means.is_empty() {
            continue;
        }
        let mut family_views = BTreeMap::<String, Vec<(f64, f64, usize)>>::new();
        for (source, (value, _, representative)) in &source_means {
            let supervision = source_supervision.get(source);
            let affine = supervision.map(|entry| entry.affine).unwrap_or(AffineFit {
                intercept: 0.0,
                slope: 1.0,
            });
            let reliability = supervision.map(|entry| entry.reliability).unwrap_or(0.60);
            let adjusted = affine.intercept + affine.slope * value;
            family_views
                .entry(source_family(source))
                .or_default()
                .push((adjusted, reliability, *representative));
        }

        let mut adjusted = Vec::<(String, f64, f64, usize)>::new();
        for (family, mut views) in family_views {
            views.sort_by(|a, b| a.0.total_cmp(&b.0));
            let values = views.iter().map(|entry| entry.0).collect::<Vec<_>>();
            let family_center = quantile_sorted(&values, 0.5);
            let family_reliability =
                views.iter().map(|entry| entry.1).sum::<f64>() / views.len() as f64;
            let representative = views
                .iter()
                .min_by(|a, b| {
                    (a.0 - family_center)
                        .abs()
                        .total_cmp(&(b.0 - family_center).abs())
                        .then_with(|| a.2.cmp(&b.2))
                })
                .map(|entry| entry.2)
                .ok_or_else(|| anyhow::anyhow!("mobility family has no representative"))?;
            adjusted.push((family, family_center, family_reliability, representative));
        }

        adjusted.sort_by(|a, b| a.1.total_cmp(&b.1));
        let values = adjusted.iter().map(|entry| entry.1).collect::<Vec<_>>();
        let center = quantile_sorted(&values, 0.5);
        let mut absolute_center = values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>();
        absolute_center.sort_by(|a, b| a.total_cmp(b));
        let mad = quantile_sorted(&absolute_center, 0.5);
        let clip_radius = (3.0 * mad).max(0.005);
        let mut weighted_sum = 0.0f64;
        let mut reliability_sum = 0.0f64;
        for (_, value, reliability, _) in &adjusted {
            let clipped = value.clamp(center - clip_radius, center + clip_radius);
            weighted_sum += reliability * clipped;
            reliability_sum += reliability;
        }
        let consensus = if reliability_sum > 0.0 {
            weighted_sum / reliability_sum
        } else {
            center
        };
        let mut consensus_abs = adjusted
            .iter()
            .map(|(_, value, _, _)| (value - consensus).abs())
            .collect::<Vec<_>>();
        consensus_abs.sort_by(|a, b| a.total_cmp(b));
        let dispersion = quantile_sorted(&consensus_abs, 0.5);
        total_abs_delta += consensus_abs.iter().sum::<f64>();
        total_abs_delta_count += consensus_abs.len();

        let source_count = adjusted.len();
        let mean_reliability = adjusted
            .iter()
            .map(|(_, _, reliability, _)| *reliability)
            .sum::<f64>()
            / source_count as f64;
        let weight = if source_count == 1 {
            singleton_examples += 1;
            (SINGLETON_WEIGHT_SCALE * mean_reliability).clamp(0.25, 0.80)
        } else {
            multisource_examples += 1;
            let dispersion_factor = 1.0 / (1.0 + dispersion / CONSENSUS_DISPERSION_SCALE);
            let source_bonus = 1.0 + 0.12 * (source_count as f64).ln();
            (mean_reliability * dispersion_factor * source_bonus).clamp(0.35, 1.25)
        };
        let representative_index = adjusted
            .iter()
            .min_by(|a, b| {
                (a.1 - consensus)
                    .abs()
                    .total_cmp(&(b.1 - consensus).abs())
                    .then_with(|| a.3.cmp(&b.3))
            })
            .map(|entry| entry.3)
            .ok_or_else(|| anyhow::anyhow!("mobility consensus identity has no representative"))?;
        total_weight += weight;
        examples.push(MobilityConsensusExample {
            representative_index,
            target_mobility: consensus as f32,
            weight: weight as f32,
            source_count,
            identity_hash: stable_hash64(identity.as_bytes()),
        });
    }

    examples.sort_by_key(|example| (example.identity_hash, example.representative_index));
    let mean_weight = if examples.is_empty() {
        0.0
    } else {
        total_weight / examples.len() as f64
    };
    let mean_delta = if total_abs_delta_count == 0 {
        0.0
    } else {
        total_abs_delta / total_abs_delta_count as f64
    };
    Ok((
        examples,
        multisource_examples,
        singleton_examples,
        mean_weight,
        mean_delta,
    ))
}

fn peptidoform_charge_key(record: &FoundationTrainingRecord) -> String {
    let mut modifications = record
        .peptidoform
        .modifications
        .iter()
        .map(|modification| {
            let site = match modification.site {
                FoundationModificationSite::Residue(index) => format!("R{index}"),
                FoundationModificationSite::NTerm => "N".to_string(),
                FoundationModificationSite::CTerm => "C".to_string(),
            };
            format!(
                "{site}:{}:{:+.4}",
                modification.identity_label(),
                modification.mass_delta
            )
        })
        .collect::<Vec<_>>();
    modifications.sort();
    let charge = record
        .context
        .charge
        .map(|value| value.to_string())
        .unwrap_or_else(|| "missing".to_string());
    format!(
        "{}|z={}|{}",
        record.peptidoform.sequence,
        charge,
        modifications.join(";")
    )
}

fn fit_xy_affine(examples: &[(f64, f64)]) -> Result<AffineFit> {
    if examples.len() < 2 {
        return Ok(AffineFit {
            intercept: 0.0,
            slope: 1.0,
        });
    }
    let n = examples.len() as f64;
    let mean_x = examples.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = examples.iter().map(|(_, y)| y).sum::<f64>() / n;
    let mut covariance = 0.0f64;
    let mut variance = 0.0f64;
    for &(x, y) in examples {
        let dx = x - mean_x;
        covariance += dx * (y - mean_y);
        variance += dx * dx;
    }
    if variance <= 1.0e-12 {
        return Ok(AffineFit {
            intercept: mean_y - mean_x,
            slope: 1.0,
        });
    }
    let slope = covariance / variance;
    Ok(AffineFit {
        intercept: mean_y - slope * mean_x,
        slope,
    })
}

fn quantile_sorted(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let q = quantile.clamp(0.0, 1.0);
    let position = q * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - lower as f64;
        values[lower] * (1.0 - fraction) + values[upper] * fraction
    }
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn consensus_fingerprint(examples: &[MobilityConsensusExample]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for example in examples {
        for value in [
            example.representative_index as u64,
            u64::from(example.target_mobility.to_bits()),
            u64::from(example.weight.to_bits()),
            example.source_count as u64,
            example.identity_hash,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::super::data::{RetentionTimeLabels, TrainingContext};
    use super::super::featurize::PeptidoformInput;
    use super::*;

    fn example(sequence: &str, source_mobility: f32) -> FoundationTrainingRecord {
        FoundationTrainingRecord {
            peptidoform: PeptidoformInput::unmodified(sequence),
            retention_time: RetentionTimeLabels::default(),
            ccs: Some(350.0 * source_mobility),
            fragments: Vec::new(),
            observed_spectrum_peaks: Vec::new(),
            context: TrainingContext {
                charge: Some(2),
                precursor_mz: Some(500.0),
                ion_mobility: Some(source_mobility),
                ..Default::default()
            },
            run_id: None,
        }
    }
    fn provenance(source_id: &str, index: usize) -> FoundationRecordProvenance {
        FoundationRecordProvenance {
            source_index: index,
            source_id: source_id.to_string(),
            source_record_index: index,
        }
    }

    #[test]
    fn train_only_consensus_excludes_validation_and_requires_valid_physics() -> Result<()> {
        let records = vec![
            example("PEPTIDE", 1.0),
            example("PEPTIDE", 1.2),
            example("PEPTIDE", 9.0),
            example("DIAK", 0.0),
        ];
        let sources = vec![
            provenance("source_a", 0),
            provenance("source_b", 1),
            provenance("validation_source", 2),
            provenance("bad_source", 3),
        ];
        let train_indices = finite_mobility_ccs_indices(&records, &[0, 1, 3]);
        assert_eq!(train_indices, vec![0, 1]);
        let training = build_train_consensus_supervision(&records, &sources, &train_indices)?;
        assert_eq!(training.examples.len(), 1);
        assert_eq!(training.multisource_examples, 1);
        assert!(training.examples[0].target_mobility > 0.99);
        assert!(training.examples[0].target_mobility < 1.21);
        assert!(!training
            .source_supervision
            .contains_key("validation_source"));
        assert_eq!(
            training.fingerprint,
            build_train_consensus_supervision(&records, &sources, &[1, 0])?.fingerprint
        );
        Ok(())
    }

    #[test]
    fn validation_consensus_uses_frozen_train_source_alignment() -> Result<()> {
        let records = vec![
            example("PEPTIDE", 1.0),
            example("PEPTIDE", 1.1),
            example("DIAK", 1.4),
            example("DIAK", 1.6),
        ];
        let sources = vec![
            provenance("a", 0),
            provenance("b", 1),
            provenance("a", 2),
            provenance("b", 3),
        ];
        let fitted = build_train_consensus_supervision(&records, &sources, &[0, 1])?;
        let original_source_count = fitted.source_supervision.len();
        let held_out = build_partition_consensus_examples(
            &records,
            &sources,
            &[2, 3],
            &fitted.source_supervision,
        )?;
        assert_eq!(held_out.len(), 1);
        assert_eq!(fitted.source_supervision.len(), original_source_count);
        assert_eq!(held_out[0].source_count, 2);
        Ok(())
    }

    #[test]
    fn exact_bruker_ccs_factor_fails_closed_on_invalid_charge_mz() {
        assert!(bruker_ccs_factor_from_values(2, 500.0).is_some());
        assert!(bruker_ccs_factor_from_values(0, 500.0).is_none());
        assert!(bruker_ccs_factor_from_values(2, f32::NAN).is_none());
    }
}
