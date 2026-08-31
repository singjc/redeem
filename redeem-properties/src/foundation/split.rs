//! Leakage-resistant dataset splitting for foundation-model evaluation.
//!
//! Random row-level splitting is inappropriate for peptide property models:
//! the same naked peptide, peptidoform, LC run, or instrument can otherwise
//! appear in both training and evaluation data.  This module assigns complete
//! groups to train/validation/test partitions with deterministic stable hashing
//! and a size-aware greedy balancer.
//!
//! The default split is sequence-disjoint.  Peptidoform-, run-, instrument-,
//! modification-signature-, and canonical modification-family-disjoint
//! variants are also available for focused generalization experiments.

use super::data::FoundationTrainingRecord;
use super::dataset::canonical_peptidoform_label;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Identity boundary that must remain disjoint across train/validation/test.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoundationSplitMode {
    /// Keep every occurrence/charge/run of the same naked peptide sequence in
    /// one partition.  This is the recommended default benchmark.
    #[default]
    Sequence,
    /// Keep the exact sequence plus site-specific mass-shift pattern together.
    Peptidoform,
    /// Keep complete acquisition runs together.  Every record must carry a
    /// non-empty run identifier.
    Run,
    /// Keep instrument labels together.  Every record must carry a non-empty
    /// instrument name.
    Instrument,
    /// Keep residue-specific modification-mass signatures together across
    /// peptide sequences. This remains useful for open-modification transfer
    /// experiments where no canonical identity is available.
    ModificationSignature,
    /// Keep canonical UniMod families disjoint across partitions, independent
    /// of peptide sequence and modification site. Every modified record must
    /// carry canonical UniMod identity; open/numeric-only mass shifts return an
    /// error rather than being silently treated as a known PTM family.
    ///
    /// Unmodified records share one `unmodified` family. For a dedicated
    /// unseen-PTM benchmark, callers will typically filter to modified records
    /// before applying this split.
    ModificationFamily,
}

/// Deterministic split configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationSplitConfig {
    /// Grouping boundary that must not cross partitions.
    pub mode: FoundationSplitMode,
    /// Fraction of records targeted for validation.
    pub validation_fraction: f64,
    /// Fraction of records targeted for final test evaluation.
    pub test_fraction: f64,
    /// Stable seed mixed into group hashing.
    pub seed: u64,
}

impl Default for FoundationSplitConfig {
    fn default() -> Self {
        Self {
            mode: FoundationSplitMode::Sequence,
            validation_fraction: 0.1,
            test_fraction: 0.1,
            seed: 2026_08_31,
        }
    }
}

impl FoundationSplitConfig {
    /// Validate split fractions before assigning any records.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("validation_fraction", self.validation_fraction),
            ("test_fraction", self.test_fraction),
        ] {
            if !value.is_finite() || !(0.0..1.0).contains(&value) {
                return Err(anyhow!("{name} must be finite and in [0, 1)"));
            }
        }
        if self.validation_fraction + self.test_fraction >= 1.0 {
            return Err(anyhow!("validation_fraction + test_fraction must be < 1"));
        }
        Ok(())
    }
}

/// Counts describing one deterministic split.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationSplitSummary {
    /// Total input records.
    pub total_records: usize,
    /// Total identity groups under the requested split mode.
    pub total_groups: usize,
    /// Training records.
    pub train_records: usize,
    /// Validation records.
    pub validation_records: usize,
    /// Test records.
    pub test_records: usize,
    /// Training identity groups.
    pub train_groups: usize,
    /// Validation identity groups.
    pub validation_groups: usize,
    /// Test identity groups.
    pub test_groups: usize,
}

/// Index-only split preserving the original record collection without cloning
/// large MS2 target vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundationSplitIndices {
    /// Original indices assigned to training.
    pub train: Vec<usize>,
    /// Original indices assigned to validation.
    pub validation: Vec<usize>,
    /// Original indices assigned to final test evaluation.
    pub test: Vec<usize>,
    /// Partition/group counts.
    pub summary: FoundationSplitSummary,
}

#[derive(Debug)]
struct SplitGroup {
    key: String,
    indices: Vec<usize>,
    hash: u64,
}

/// Split foundation records while keeping the configured identity groups
/// strictly disjoint.
///
/// Groups are sorted largest-first and assigned with a deterministic greedy
/// objective that minimizes normalized deviation from the requested record
/// fractions.  This behaves better than assigning each group independently by
/// a hash threshold when some peptides/runs contribute many more observations
/// than others.
pub fn split_foundation_records(
    records: &[FoundationTrainingRecord],
    config: &FoundationSplitConfig,
) -> Result<FoundationSplitIndices> {
    config.validate()?;
    if records.is_empty() {
        return Ok(FoundationSplitIndices {
            train: Vec::new(),
            validation: Vec::new(),
            test: Vec::new(),
            summary: FoundationSplitSummary::default(),
        });
    }

    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for (index, record) in records.iter().enumerate() {
        let key = split_key(record, config.mode, index)?;
        grouped.entry(key).or_default().push(index);
    }

    let mut groups: Vec<SplitGroup> = grouped
        .into_iter()
        .map(|(key, indices)| SplitGroup {
            hash: stable_group_hash(&key, config.seed),
            key,
            indices,
        })
        .collect();
    groups.sort_by(|left, right| {
        right
            .indices
            .len()
            .cmp(&left.indices.len())
            .then_with(|| left.hash.cmp(&right.hash))
            .then_with(|| left.key.cmp(&right.key))
    });

    let total = records.len() as f64;
    let targets = [
        total * (1.0 - config.validation_fraction - config.test_fraction),
        total * config.validation_fraction,
        total * config.test_fraction,
    ];
    let enabled = [
        true,
        config.validation_fraction > 0.0,
        config.test_fraction > 0.0,
    ];
    let mut counts = [0usize; 3];
    let mut group_counts = [0usize; 3];
    let mut partitions = [
        Vec::<usize>::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
    ];

    for group in groups {
        let split = best_partition(group.indices.len(), counts, targets, enabled);
        counts[split] += group.indices.len();
        group_counts[split] += 1;
        partitions[split].extend(group.indices);
    }

    for partition in &mut partitions {
        partition.sort_unstable();
    }

    let [train, validation, test] = partitions;
    let summary = FoundationSplitSummary {
        total_records: records.len(),
        total_groups: group_counts.iter().sum(),
        train_records: train.len(),
        validation_records: validation.len(),
        test_records: test.len(),
        train_groups: group_counts[0],
        validation_groups: group_counts[1],
        test_groups: group_counts[2],
    };

    Ok(FoundationSplitIndices {
        train,
        validation,
        test,
        summary,
    })
}

fn best_partition(
    group_size: usize,
    counts: [usize; 3],
    targets: [f64; 3],
    enabled: [bool; 3],
) -> usize {
    let mut best = 0usize;
    let mut best_cost = f64::INFINITY;
    for candidate in 0..3 {
        if !enabled[candidate] {
            continue;
        }
        let mut trial = counts;
        trial[candidate] += group_size;
        let cost = trial
            .iter()
            .zip(targets)
            .map(|(observed, target)| {
                let scale = target.max(1.0);
                let delta = *observed as f64 - target;
                delta * delta / scale
            })
            .sum::<f64>();
        if cost < best_cost {
            best = candidate;
            best_cost = cost;
        }
    }
    best
}

fn split_key(
    record: &FoundationTrainingRecord,
    mode: FoundationSplitMode,
    record_index: usize,
) -> Result<String> {
    match mode {
        FoundationSplitMode::Sequence => Ok(record.peptidoform.sequence.clone()),
        FoundationSplitMode::Peptidoform => Ok(canonical_peptidoform_label(&record.peptidoform)),
        FoundationSplitMode::Run => record
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .ok_or_else(|| anyhow!("record {record_index} has no run id for run-disjoint split")),
        FoundationSplitMode::Instrument => record
            .context
            .instrument_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_ascii_lowercase())
            .ok_or_else(|| {
                anyhow!(
                    "record {record_index} has no instrument name for instrument-disjoint split"
                )
            }),
        FoundationSplitMode::ModificationSignature => Ok(modification_signature(record)),
        FoundationSplitMode::ModificationFamily => modification_family(record, record_index),
    }
}

fn modification_signature(record: &FoundationTrainingRecord) -> String {
    if record.peptidoform.modifications.is_empty() {
        return "unmodified".to_string();
    }
    let sequence = record.peptidoform.sequence.as_bytes();
    let mut modifications: Vec<String> = record
        .peptidoform
        .modifications
        .iter()
        .map(|modification| {
            let residue = sequence
                .get(modification.residue_index)
                .copied()
                .map(char::from)
                .unwrap_or('?');
            format!("{residue}:{:+.4}", modification.mass_delta)
        })
        .collect();
    modifications.sort();
    modifications.join("|")
}

fn modification_family(record: &FoundationTrainingRecord, record_index: usize) -> Result<String> {
    if record.peptidoform.modifications.is_empty() {
        return Ok("unmodified".to_string());
    }
    let mut families = Vec::<u32>::new();
    for modification in &record.peptidoform.modifications {
        let Some(unimod_id) = modification.unimod_id else {
            return Err(anyhow!(
                "record {record_index} contains a modification without canonical UniMod identity;                  modification-family split requires canonical PTM ids"
            ));
        };
        families.push(unimod_id);
    }
    families.sort_unstable();
    families.dedup();
    Ok(families
        .into_iter()
        .map(|id| format!("UniMod:{id}"))
        .collect::<Vec<_>>()
        .join("|"))
}

fn stable_group_hash(value: &str, seed: u64) -> u64 {
    // FNV-1a is deliberately implemented locally rather than using
    // DefaultHasher, whose algorithm is not a stable serialization contract.
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed.rotate_left(17);
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationModification, FoundationModificationSite, PeptidoformInput, RetentionTimeLabels,
        TrainingContext,
    };

    fn record(sequence: &str, run: &str, instrument: &str) -> FoundationTrainingRecord {
        FoundationTrainingRecord {
            peptidoform: PeptidoformInput::unmodified(sequence),
            retention_time: RetentionTimeLabels::default(),
            ccs: None,
            fragments: Vec::new(),
            context: TrainingContext {
                instrument_name: Some(instrument.to_string()),
                ..TrainingContext::default()
            },
            run_id: Some(run.to_string()),
        }
    }

    #[test]
    fn repeated_sequences_never_cross_partitions() {
        let records = vec![
            record("PEPTIDEK", "run-a", "A"),
            record("PEPTIDEK", "run-b", "B"),
            record("AAAAAAK", "run-a", "A"),
            record("CCCCCCK", "run-b", "B"),
            record("DDDDDDK", "run-c", "C"),
            record("EEEEEEK", "run-d", "D"),
        ];
        let split = split_foundation_records(
            &records,
            &FoundationSplitConfig {
                validation_fraction: 0.2,
                test_fraction: 0.2,
                ..FoundationSplitConfig::default()
            },
        )
        .unwrap();
        let partition = |index: usize| {
            if split.train.contains(&index) {
                0
            } else if split.validation.contains(&index) {
                1
            } else {
                2
            }
        };
        assert_eq!(partition(0), partition(1));
        assert_eq!(split.summary.total_records, records.len());
    }

    #[test]
    fn run_disjoint_split_requires_run_metadata() {
        let mut records = vec![record("PEPTIDEK", "run-a", "A")];
        records[0].run_id = None;
        let error = split_foundation_records(
            &records,
            &FoundationSplitConfig {
                mode: FoundationSplitMode::Run,
                ..FoundationSplitConfig::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("no run id"));
    }

    #[test]
    fn modification_signature_is_sequence_independent() {
        let mut first = record("PEPMIDEK", "run-a", "A");
        first.peptidoform.modifications = vec![FoundationModification::mass_delta(3, 15.994_915)];
        let mut second = record("AAAAAMK", "run-b", "B");
        second.peptidoform.modifications = vec![FoundationModification::mass_delta(5, 15.994_915)];
        assert_eq!(
            modification_signature(&first),
            modification_signature(&second)
        );
    }

    #[test]
    fn modification_family_uses_unimod_identity_not_sequence_or_site() {
        let mut first = record("PEPMIDEK", "run-a", "A");
        first.peptidoform.modifications = vec![FoundationModification::unimod(
            FoundationModificationSite::Residue(3),
            3,
            35,
            15.994_915,
        )];
        let mut second = record("AAAAAMK", "run-b", "B");
        second.peptidoform.modifications = vec![FoundationModification::unimod(
            FoundationModificationSite::Residue(5),
            5,
            35,
            15.994_915,
        )];
        assert_eq!(modification_family(&first, 0).unwrap(), "UniMod:35");
        assert_eq!(
            modification_family(&first, 0).unwrap(),
            modification_family(&second, 1).unwrap()
        );
    }

    #[test]
    fn modification_family_rejects_open_mass_shifts() {
        let mut record = record("PEPMIDEK", "run-a", "A");
        record.peptidoform.modifications = vec![FoundationModification::mass_delta(3, 15.994_915)];
        let error = modification_family(&record, 0).unwrap_err();
        assert!(error.to_string().contains("canonical UniMod identity"));
    }
}
