//! Reproducible benchmark manifests for peptide-foundation experiments.
//!
//! A split is only scientifically useful if it can be materialized, audited,
//! and reused without silently changing when source tables or loader behavior
//! change. This module stores stable per-record fingerprints alongside the
//! deterministic split assignment and validates them before reuse.

use super::data::FoundationTrainingRecord;
use super::dataset::canonical_peptidoform_label;
use super::split::{
    foundation_split_group_key, split_foundation_record_indices, FoundationSplitConfig,
    FoundationSplitIndices, FoundationSplitMode, FoundationSplitSummary,
};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Current on-disk benchmark-manifest format.
pub const FOUNDATION_BENCHMARK_MANIFEST_VERSION: u32 = 1;

/// Train/validation/test assignment stored in a benchmark manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FoundationPartition {
    /// Optimization partition.
    Train,
    /// Hyperparameter/model-selection partition.
    Validation,
    /// Final held-out evaluation partition.
    Test,
}

impl FoundationPartition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "train" => Ok(Self::Train),
            "validation" => Ok(Self::Validation),
            "test" => Ok(Self::Test),
            other => Err(anyhow!("unknown foundation partition '{other}'")),
        }
    }
}

/// One materialized record assignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationBenchmarkEntry {
    /// Original index in the loaded record collection.
    pub record_index: usize,
    /// Stable fingerprint over peptidoform, labels, context, and fragments.
    pub record_fingerprint: u64,
    /// Assigned partition.
    pub partition: FoundationPartition,
    /// Human-readable split identity for audit/debugging.
    pub identity_key: String,
    /// Naked peptide sequence retained for convenient leakage audits.
    pub sequence: String,
    /// Canonical peptidoform label retained for convenient leakage audits.
    pub peptidoform: String,
}

/// Reusable deterministic benchmark definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationBenchmarkManifest {
    /// On-disk format version.
    pub format_version: u32,
    /// Stable content fingerprint over the selected record collection.
    pub dataset_fingerprint: u64,
    /// Number of records in the complete source collection.
    pub source_records: usize,
    /// Number of records selected into this benchmark.
    pub selected_records: usize,
    /// Whether the benchmark intentionally excludes unmodified records.
    pub modified_only: bool,
    /// Whether mixed PTM-family records were excluded so each selected record
    /// belongs to one canonical modification family.
    pub single_modification_family_only: bool,
    /// Modified source records excluded because they contain more than one
    /// distinct canonical PTM family.
    pub excluded_mixed_family_records: usize,
    /// Split parameters used to create the assignment.
    pub split_config: FoundationSplitConfig,
    /// Realized partition/group counts.
    pub summary: FoundationSplitSummary,
    /// Materialized assignments.
    pub entries: Vec<FoundationBenchmarkEntry>,
}

impl FoundationBenchmarkManifest {
    /// Stable fingerprint over the exact split configuration and every
    /// materialized record-to-partition assignment. Unlike
    /// `dataset_fingerprint`, this changes when partition assignments change.
    pub fn manifest_fingerprint(&self) -> u64 {
        let mut hash = StableFnv64::new();
        hash.u32(self.format_version);
        hash.u64(self.dataset_fingerprint);
        hash.usize(self.source_records);
        hash.usize(self.selected_records);
        hash.bytes(&[u8::from(self.modified_only)]);
        hash.bytes(&[u8::from(self.single_modification_family_only)]);
        hash.usize(self.excluded_mixed_family_records);
        hash.str(split_mode_name(self.split_config.mode));
        hash.u64(self.split_config.validation_fraction.to_bits());
        hash.u64(self.split_config.test_fraction.to_bits());
        hash.u64(self.split_config.seed);
        let mut entries: Vec<&FoundationBenchmarkEntry> = self.entries.iter().collect();
        entries.sort_by_key(|entry| entry.record_index);
        for entry in entries {
            hash.usize(entry.record_index);
            hash.u64(entry.record_fingerprint);
            hash.bytes(&[match entry.partition {
                FoundationPartition::Train => 0,
                FoundationPartition::Validation => 1,
                FoundationPartition::Test => 2,
            }]);
            hash.str(&entry.identity_key);
        }
        hash.finish()
    }

    /// Return original record indices assigned to one partition.
    pub fn partition_indices(&self, partition: FoundationPartition) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|entry| entry.partition == partition)
            .map(|entry| entry.record_index)
            .collect()
    }

    /// Write the manifest as an auditable line-oriented TSV.
    pub fn write_tsv<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("failed to create foundation benchmark {:?}", path))?;
        let mut writer = BufWriter::new(file);
        writeln!(
            writer,
            "# foundation_benchmark_manifest_version={}",
            self.format_version
        )?;
        writeln!(
            writer,
            "# dataset_fingerprint=fnv1a64:{:016x}",
            self.dataset_fingerprint
        )?;
        writeln!(writer, "# source_records={}", self.source_records)?;
        writeln!(writer, "# selected_records={}", self.selected_records)?;
        writeln!(writer, "# modified_only={}", self.modified_only)?;
        writeln!(
            writer,
            "# single_modification_family_only={}",
            self.single_modification_family_only
        )?;
        writeln!(
            writer,
            "# excluded_mixed_family_records={}",
            self.excluded_mixed_family_records
        )?;
        writeln!(
            writer,
            "# split_mode={}",
            split_mode_name(self.split_config.mode)
        )?;
        writeln!(
            writer,
            "# validation_fraction={:.12}",
            self.split_config.validation_fraction
        )?;
        writeln!(
            writer,
            "# test_fraction={:.12}",
            self.split_config.test_fraction
        )?;
        writeln!(writer, "# seed={}", self.split_config.seed)?;
        writeln!(writer, "# total_groups={}", self.summary.total_groups)?;
        writeln!(writer, "# train_records={}", self.summary.train_records)?;
        writeln!(
            writer,
            "# validation_records={}",
            self.summary.validation_records
        )?;
        writeln!(writer, "# test_records={}", self.summary.test_records)?;
        writeln!(writer, "# train_groups={}", self.summary.train_groups)?;
        writeln!(
            writer,
            "# validation_groups={}",
            self.summary.validation_groups
        )?;
        writeln!(writer, "# test_groups={}", self.summary.test_groups)?;
        writeln!(
            writer,
            "record_index\trecord_fingerprint\tpartition\tidentity_key\tsequence\tpeptidoform"
        )?;
        for entry in &self.entries {
            writeln!(
                writer,
                "{}\t{:016x}\t{}\t{}\t{}\t{}",
                entry.record_index,
                entry.record_fingerprint,
                entry.partition.as_str(),
                escape_tsv(&entry.identity_key),
                escape_tsv(&entry.sequence),
                escape_tsv(&entry.peptidoform),
            )?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Read a manifest previously written by [`Self::write_tsv`].
    pub fn read_tsv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open foundation benchmark {:?}", path))?;
        let reader = BufReader::new(file);
        let mut metadata = BTreeMap::<String, String>::new();
        let mut entries = Vec::<FoundationBenchmarkEntry>::new();
        let mut saw_header = false;

        for line in reader.lines() {
            let line = line?;
            if let Some(comment) = line.strip_prefix("# ") {
                if let Some((key, value)) = comment.split_once('=') {
                    metadata.insert(key.to_string(), value.to_string());
                }
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
            if !saw_header {
                let expected = "record_index\trecord_fingerprint\tpartition\tidentity_key\tsequence\tpeptidoform";
                if line != expected {
                    return Err(anyhow!(
                        "unexpected foundation benchmark TSV header: '{line}'"
                    ));
                }
                saw_header = true;
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 6 {
                return Err(anyhow!(
                    "foundation benchmark row has {} columns, expected 6",
                    fields.len()
                ));
            }
            entries.push(FoundationBenchmarkEntry {
                record_index: fields[0].parse()?,
                record_fingerprint: u64::from_str_radix(fields[1], 16)?,
                partition: FoundationPartition::parse(fields[2])?,
                identity_key: unescape_tsv(fields[3])?,
                sequence: unescape_tsv(fields[4])?,
                peptidoform: unescape_tsv(fields[5])?,
            });
        }

        let format_version: u32 =
            required_meta(&metadata, "foundation_benchmark_manifest_version")?.parse()?;
        let fingerprint = required_meta(&metadata, "dataset_fingerprint")?
            .strip_prefix("fnv1a64:")
            .ok_or_else(|| anyhow!("unsupported dataset fingerprint encoding"))?;
        let split_config = FoundationSplitConfig {
            mode: parse_split_mode(required_meta(&metadata, "split_mode")?)?,
            validation_fraction: required_meta(&metadata, "validation_fraction")?.parse()?,
            test_fraction: required_meta(&metadata, "test_fraction")?.parse()?,
            seed: required_meta(&metadata, "seed")?.parse()?,
        };
        split_config.validate()?;
        let source_records = required_meta(&metadata, "source_records")?.parse()?;
        let selected_records = required_meta(&metadata, "selected_records")?.parse()?;
        let modified_only = required_meta(&metadata, "modified_only")?.parse()?;
        let single_modification_family_only =
            required_meta(&metadata, "single_modification_family_only")?.parse()?;
        let excluded_mixed_family_records =
            required_meta(&metadata, "excluded_mixed_family_records")?.parse()?;
        if entries.len() != selected_records {
            return Err(anyhow!(
                "foundation benchmark contains {} entries but metadata declares {selected_records}",
                entries.len()
            ));
        }
        let summary = FoundationSplitSummary {
            total_records: selected_records,
            total_groups: required_meta(&metadata, "total_groups")?.parse()?,
            train_records: required_meta(&metadata, "train_records")?.parse()?,
            validation_records: required_meta(&metadata, "validation_records")?.parse()?,
            test_records: required_meta(&metadata, "test_records")?.parse()?,
            train_groups: required_meta(&metadata, "train_groups")?.parse()?,
            validation_groups: required_meta(&metadata, "validation_groups")?.parse()?,
            test_groups: required_meta(&metadata, "test_groups")?.parse()?,
        };
        if summary.train_records + summary.validation_records + summary.test_records
            != selected_records
        {
            return Err(anyhow!(
                "foundation benchmark partition counts do not sum to selected_records"
            ));
        }
        Ok(Self {
            format_version,
            dataset_fingerprint: u64::from_str_radix(fingerprint, 16)?,
            source_records,
            selected_records,
            modified_only,
            single_modification_family_only,
            excluded_mixed_family_records,
            split_config,
            summary,
            entries,
        })
    }

    /// Validate fingerprints and leakage boundaries against a loaded dataset,
    /// returning the original indices for each partition.
    pub fn validate_against_records(
        &self,
        records: &[FoundationTrainingRecord],
    ) -> Result<FoundationSplitIndices> {
        if self.format_version != FOUNDATION_BENCHMARK_MANIFEST_VERSION {
            return Err(anyhow!(
                "unsupported foundation benchmark manifest version {}",
                self.format_version
            ));
        }
        if self.source_records != records.len() {
            return Err(anyhow!(
                "foundation benchmark expects {} source records but loader produced {}",
                self.source_records,
                records.len()
            ));
        }
        let mut selected = Vec::<usize>::with_capacity(self.entries.len());
        let mut seen = BTreeSet::<usize>::new();
        for entry in &self.entries {
            if !seen.insert(entry.record_index) {
                return Err(anyhow!(
                    "foundation benchmark repeats record index {}",
                    entry.record_index
                ));
            }
            let record = records.get(entry.record_index).ok_or_else(|| {
                anyhow!(
                    "foundation benchmark record index {} is out of bounds",
                    entry.record_index
                )
            })?;
            let fingerprint = foundation_record_fingerprint(record);
            if fingerprint != entry.record_fingerprint {
                return Err(anyhow!(
                    "foundation benchmark record {} fingerprint changed: expected {:016x}, observed {:016x}",
                    entry.record_index,
                    entry.record_fingerprint,
                    fingerprint
                ));
            }
            selected.push(entry.record_index);
        }
        let all_indices: Vec<usize> = (0..records.len()).collect();
        let fingerprint = foundation_dataset_fingerprint(records, &all_indices)?;
        if fingerprint != self.dataset_fingerprint {
            return Err(anyhow!(
                "foundation benchmark dataset fingerprint changed: expected {:016x}, observed {:016x}",
                self.dataset_fingerprint,
                fingerprint
            ));
        }
        validate_manifest_leakage(self)?;

        let mut train = Vec::new();
        let mut validation = Vec::new();
        let mut test = Vec::new();
        for entry in &self.entries {
            match entry.partition {
                FoundationPartition::Train => train.push(entry.record_index),
                FoundationPartition::Validation => validation.push(entry.record_index),
                FoundationPartition::Test => test.push(entry.record_index),
            }
        }
        train.sort_unstable();
        validation.sort_unstable();
        test.sort_unstable();
        Ok(FoundationSplitIndices {
            train,
            validation,
            test,
            summary: self.summary.clone(),
        })
    }
}

/// Build a reusable benchmark manifest from loaded records.
pub fn build_foundation_benchmark_manifest(
    records: &[FoundationTrainingRecord],
    split_config: FoundationSplitConfig,
    modified_only: bool,
) -> Result<FoundationBenchmarkManifest> {
    if split_config.mode == FoundationSplitMode::ModificationFamily && !modified_only {
        return Err(anyhow!(
            "modification-family benchmarks must set modified_only=true so the unmodified corpus does not become one dominant identity group"
        ));
    }
    let single_modification_family_only =
        split_config.mode == FoundationSplitMode::ModificationFamily;
    let mut excluded_mixed_family_records = 0usize;
    let mut selected = Vec::<usize>::new();
    for (index, record) in records.iter().enumerate() {
        if modified_only && record.peptidoform.modifications.is_empty() {
            continue;
        }
        if single_modification_family_only {
            let families = canonical_modification_families(record, index)?;
            if families.len() != 1 {
                excluded_mixed_family_records += 1;
                continue;
            }
        }
        selected.push(index);
    }
    if selected.is_empty() {
        return Err(anyhow!(
            "foundation benchmark selection contains zero records"
        ));
    }
    let split = split_foundation_record_indices(records, &selected, &split_config)?;
    let mut partition = BTreeMap::<usize, FoundationPartition>::new();
    for &index in &split.train {
        partition.insert(index, FoundationPartition::Train);
    }
    for &index in &split.validation {
        partition.insert(index, FoundationPartition::Validation);
    }
    for &index in &split.test {
        partition.insert(index, FoundationPartition::Test);
    }

    let mut entries = Vec::with_capacity(selected.len());
    for &index in &selected {
        let record = &records[index];
        entries.push(FoundationBenchmarkEntry {
            record_index: index,
            record_fingerprint: foundation_record_fingerprint(record),
            partition: partition[&index],
            identity_key: foundation_split_group_key(record, split_config.mode, index)?,
            sequence: record.peptidoform.sequence.clone(),
            peptidoform: canonical_peptidoform_label(&record.peptidoform),
        });
    }
    entries.sort_by_key(|entry| entry.record_index);
    let manifest = FoundationBenchmarkManifest {
        format_version: FOUNDATION_BENCHMARK_MANIFEST_VERSION,
        dataset_fingerprint: {
            let all_indices: Vec<usize> = (0..records.len()).collect();
            foundation_dataset_fingerprint(records, &all_indices)?
        },
        source_records: records.len(),
        selected_records: selected.len(),
        modified_only,
        single_modification_family_only,
        excluded_mixed_family_records,
        split_config,
        summary: split.summary,
        entries,
    };
    validate_manifest_leakage(&manifest)?;
    Ok(manifest)
}

/// Stable content fingerprint for one foundation training record.
pub fn foundation_record_fingerprint(record: &FoundationTrainingRecord) -> u64 {
    let mut hash = StableFnv64::new();
    hash.str(&record.peptidoform.sequence);
    let mut modifications = record.peptidoform.modifications.clone();
    modifications.sort_by(|left, right| {
        left.residue_index
            .cmp(&right.residue_index)
            .then_with(|| left.unimod_id.cmp(&right.unimod_id))
            .then_with(|| left.mass_delta.to_bits().cmp(&right.mass_delta.to_bits()))
    });
    hash.usize(modifications.len());
    for modification in modifications {
        hash.usize(modification.residue_index);
        hash.u32(modification.unimod_id.unwrap_or(u32::MAX));
        hash.u32(modification.mass_delta.to_bits());
        hash.str(&format!("{:?}", modification.site));
    }
    hash.option_f32(record.retention_time.normalized);
    hash.option_f32(record.retention_time.observed_seconds);
    hash.option_f32(record.ccs);
    hash.option_i32(record.context.charge);
    hash.option_f32(record.context.precursor_mz);
    hash.option_f32(record.context.nce);
    hash.option_u32(record.context.instrument_id);
    hash.option_str(record.context.instrument_name.as_deref());
    hash.option_f32(record.context.ion_mobility);
    hash.option_f32(record.context.gradient_seconds);
    hash.option_str(record.run_id.as_deref());

    let mut fragments: Vec<(usize, usize, u32)> = record
        .fragments
        .iter()
        .map(|fragment| {
            (
                fragment.cleavage_index,
                fragment.channel,
                fragment.intensity.to_bits(),
            )
        })
        .collect();
    fragments.sort_unstable();
    hash.usize(fragments.len());
    for (cleavage, channel, intensity) in fragments {
        hash.usize(cleavage);
        hash.usize(channel);
        hash.u32(intensity);
    }
    hash.finish()
}

/// Stable order-independent fingerprint over selected records.
pub fn foundation_dataset_fingerprint(
    records: &[FoundationTrainingRecord],
    selected_indices: &[usize],
) -> Result<u64> {
    let mut record_hashes = Vec::<u64>::with_capacity(selected_indices.len());
    for &index in selected_indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow!("foundation dataset fingerprint index {index} is out of bounds")
        })?;
        record_hashes.push(foundation_record_fingerprint(record));
    }
    record_hashes.sort_unstable();
    let mut hash = StableFnv64::new();
    hash.usize(record_hashes.len());
    for value in record_hashes {
        hash.u64(value);
    }
    Ok(hash.finish())
}

fn canonical_modification_families(
    record: &FoundationTrainingRecord,
    record_index: usize,
) -> Result<BTreeSet<u32>> {
    let mut families = BTreeSet::<u32>::new();
    for modification in &record.peptidoform.modifications {
        let Some(unimod_id) = modification.unimod_id else {
            return Err(anyhow!(
                "record {record_index} contains a non-canonical modification; strict PTM-family benchmarks require UniMod identity"
            ));
        };
        families.insert(unimod_id);
    }
    Ok(families)
}

fn validate_manifest_leakage(manifest: &FoundationBenchmarkManifest) -> Result<()> {
    match manifest.split_config.mode {
        FoundationSplitMode::ModificationFamily => {
            let mut family_partition = BTreeMap::<String, FoundationPartition>::new();
            for entry in &manifest.entries {
                for family in entry
                    .identity_key
                    .split('|')
                    .filter(|value| !value.is_empty())
                {
                    if family == "unmodified" {
                        continue;
                    }
                    if let Some(existing) =
                        family_partition.insert(family.to_string(), entry.partition)
                    {
                        if existing != entry.partition {
                            return Err(anyhow!(
                                "PTM family {family} crosses benchmark partitions ({:?} vs {:?})",
                                existing,
                                entry.partition
                            ));
                        }
                    }
                }
            }
        }
        _ => {
            let mut group_partition = BTreeMap::<String, FoundationPartition>::new();
            for entry in &manifest.entries {
                if let Some(existing) =
                    group_partition.insert(entry.identity_key.clone(), entry.partition)
                {
                    if existing != entry.partition {
                        return Err(anyhow!(
                            "foundation split identity '{}' crosses benchmark partitions",
                            entry.identity_key
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn split_mode_name(mode: FoundationSplitMode) -> &'static str {
    match mode {
        FoundationSplitMode::Sequence => "sequence",
        FoundationSplitMode::Peptidoform => "peptidoform",
        FoundationSplitMode::Run => "run",
        FoundationSplitMode::Instrument => "instrument",
        FoundationSplitMode::ModificationSignature => "modification-signature",
        FoundationSplitMode::ModificationFamily => "modification-family",
    }
}

fn parse_split_mode(value: &str) -> Result<FoundationSplitMode> {
    match value {
        "sequence" => Ok(FoundationSplitMode::Sequence),
        "peptidoform" => Ok(FoundationSplitMode::Peptidoform),
        "run" => Ok(FoundationSplitMode::Run),
        "instrument" => Ok(FoundationSplitMode::Instrument),
        "modification-signature" => Ok(FoundationSplitMode::ModificationSignature),
        "modification-family" => Ok(FoundationSplitMode::ModificationFamily),
        other => Err(anyhow!("unknown foundation split mode '{other}'")),
    }
}

fn required_meta<'a>(metadata: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| anyhow!("foundation benchmark is missing metadata key '{key}'"))
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn unescape_tsv(value: &str) -> Result<String> {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            output.push(character);
            continue;
        }
        let Some(escaped) = chars.next() else {
            return Err(anyhow!("unterminated escape in foundation benchmark field"));
        };
        output.push(match escaped {
            '\\' => '\\',
            't' => '\t',
            'n' => '\n',
            'r' => '\r',
            other => return Err(anyhow!("unsupported foundation benchmark escape \\{other}")),
        });
    }
    Ok(output)
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

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
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

    fn option_f32(&mut self, value: Option<f32>) {
        match value {
            Some(value) => {
                self.bytes(&[1]);
                self.u32(value.to_bits());
            }
            None => self.bytes(&[0]),
        }
    }

    fn option_i32(&mut self, value: Option<i32>) {
        match value {
            Some(value) => {
                self.bytes(&[1]);
                self.bytes(&value.to_le_bytes());
            }
            None => self.bytes(&[0]),
        }
    }

    fn option_u32(&mut self, value: Option<u32>) {
        match value {
            Some(value) => {
                self.bytes(&[1]);
                self.u32(value);
            }
            None => self.bytes(&[0]),
        }
    }

    fn option_str(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.bytes(&[1]);
                self.str(value);
            }
            None => self.bytes(&[0]),
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}
