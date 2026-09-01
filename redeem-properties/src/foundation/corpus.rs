//! Multi-source corpus assembly with source provenance and corpus-wide splits.
//!
//! Foundation pretraining commonly combines transition libraries produced by
//! different pipelines. This module keeps source identity beside each grouped
//! precursor record while using one shared loader/instrument vocabulary. Split
//! manifests are then generated over the *combined* record collection so a
//! sequence cannot leak across sources.

use super::dataset::{
    FoundationCcsDerivationMode, FoundationDatasetLoader, FoundationTableLoadReport,
    FoundationTableLoadStats, FoundationTableLoaderConfig,
};
use super::experiment::{
    build_foundation_benchmark_manifest, foundation_dataset_fingerprint,
    FoundationBenchmarkManifest,
};
use super::metadata::FoundationSourceMetadata;
use super::split::FoundationSplitConfig;
use super::FoundationTrainingRecord;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Human-readable delimiter selector for YAML corpus configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FoundationCorpusDelimiter {
    /// Infer from extension; `.tsv` and `.tsv.zst` use tab, otherwise comma.
    #[default]
    Auto,
    /// Tab-separated values.
    Tab,
    /// Comma-separated values.
    Comma,
    /// Semicolon-separated values.
    Semicolon,
}

impl FoundationCorpusDelimiter {
    fn byte(self, path: &Path) -> u8 {
        match self {
            Self::Auto => {
                let name = path.to_string_lossy().to_ascii_lowercase();
                if name.ends_with(".tsv") || name.ends_with(".tsv.zst") {
                    b'\t'
                } else {
                    b','
                }
            }
            Self::Tab => b'\t',
            Self::Comma => b',',
            Self::Semicolon => b';',
        }
    }
}

/// One source table in a multi-source foundation corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationCorpusSourceSpec {
    /// Stable logical source identifier used in provenance/fingerprints.
    ///
    /// YAML accepts either the canonical `id:` spelling or the friendlier
    /// `name:` alias for compatibility with early foundation examples.
    #[serde(alias = "name")]
    pub id: String,
    /// Source path. `.zst` files are streamed through the system `zstd -dc`.
    pub path: PathBuf,
    /// Source delimiter.
    pub delimiter: FoundationCorpusDelimiter,
    /// Optional source-level acquisition metadata. Every field may be absent.
    pub metadata: FoundationSourceMetadata,
    /// Optional per-source strictness override.
    pub strict: Option<bool>,
    /// Optional per-source override for CCS derivation from ion mobility.
    ///
    /// This is useful for mixed corpora where only some sources contain
    /// Bruker/timsTOF inverse reduced mobility (`1/K0`).
    pub ccs_derivation: Option<FoundationCcsDerivationMode>,
}

impl Default for FoundationCorpusSourceSpec {
    fn default() -> Self {
        Self {
            id: String::new(),
            path: PathBuf::new(),
            delimiter: FoundationCorpusDelimiter::Auto,
            metadata: FoundationSourceMetadata::default(),
            strict: None,
            ccs_derivation: None,
        }
    }
}

/// YAML-friendly corpus assembly configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationCorpusConfig {
    /// Maximum instrument vocabulary size used by the model.
    pub instrument_vocab_size: usize,
    /// Shared loader defaults. Source metadata is applied as missing-value defaults.
    pub loader: FoundationTableLoaderConfig,
    /// Source tables in deterministic configuration order.
    pub sources: Vec<FoundationCorpusSourceSpec>,
}

impl Default for FoundationCorpusConfig {
    fn default() -> Self {
        Self {
            instrument_vocab_size: 16,
            loader: FoundationTableLoaderConfig::default(),
            sources: Vec::new(),
        }
    }
}

/// Provenance for one record in the combined corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationRecordProvenance {
    /// Index into [`FoundationCorpus::sources`].
    pub source_index: usize,
    /// Logical source id.
    pub source_id: String,
    /// Record index within that source after transition-row grouping.
    pub source_record_index: usize,
}

/// Audit summary for one loaded source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundationCorpusSourceSummary {
    /// Logical source id.
    pub id: String,
    /// Input path as configured.
    pub path: PathBuf,
    /// Schema profile selected by the table loader.
    pub profile: String,
    /// Combined-corpus start offset.
    pub record_start: usize,
    /// Number of grouped precursor records contributed by this source.
    pub record_count: usize,
    /// Content fingerprint over source records.
    pub dataset_fingerprint: u64,
    /// Source table parse/coverage statistics.
    pub stats: FoundationTableLoadStats,
}

/// Combined records plus source provenance and one shared instrument vocabulary.
#[derive(Debug, Clone)]
pub struct FoundationCorpus {
    /// All grouped precursor records in source configuration order.
    pub records: Vec<FoundationTrainingRecord>,
    /// One provenance entry per record.
    pub provenance: Vec<FoundationRecordProvenance>,
    /// Per-source audit summaries.
    pub sources: Vec<FoundationCorpusSourceSummary>,
    /// Shared instrument labels in model-id order.
    pub instrument_names: Vec<String>,
    /// Stable fingerprint over content plus source assignment.
    pub corpus_fingerprint: u64,
}

impl FoundationCorpus {
    /// Build a benchmark over the combined corpus. Sequence mode is therefore
    /// automatically sequence-disjoint across all sources.
    pub fn build_benchmark_manifest(
        &self,
        split_config: FoundationSplitConfig,
        modified_only: bool,
    ) -> Result<FoundationBenchmarkManifest> {
        build_foundation_benchmark_manifest(&self.records, split_config, modified_only)
    }

    /// Write record/source provenance for external audits.
    pub fn write_provenance_tsv<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let mut writer = BufWriter::new(
            File::create(path)
                .with_context(|| format!("failed to create corpus provenance {path:?}"))?,
        );
        writeln!(
            writer,
            "# corpus_fingerprint=fnv1a64:{:016x}",
            self.corpus_fingerprint
        )?;
        writeln!(
            writer,
            "record_index\tsource_index\tsource_id\tsource_record_index"
        )?;
        for (record_index, provenance) in self.provenance.iter().enumerate() {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}",
                record_index,
                provenance.source_index,
                escape_tsv(&provenance.source_id),
                provenance.source_record_index
            )?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Load all configured sources using one shared instrument vocabulary.
pub fn load_foundation_corpus(config: &FoundationCorpusConfig) -> Result<FoundationCorpus> {
    if config.instrument_vocab_size == 0 {
        anyhow::bail!("foundation corpus instrument_vocab_size must be greater than zero");
    }
    if config.sources.is_empty() {
        anyhow::bail!("foundation corpus requires at least one source");
    }
    let mut ids = std::collections::BTreeSet::<String>::new();
    for source in &config.sources {
        if source.id.trim().is_empty() {
            anyhow::bail!("foundation corpus source id cannot be empty");
        }
        if !ids.insert(source.id.clone()) {
            anyhow::bail!("duplicate foundation corpus source id '{}'", source.id);
        }
        if source.path.as_os_str().is_empty() {
            anyhow::bail!("foundation corpus source '{}' has an empty path", source.id);
        }
    }

    let mut loader = FoundationDatasetLoader::new(config.instrument_vocab_size);
    let mut records = Vec::<FoundationTrainingRecord>::new();
    let mut provenance = Vec::<FoundationRecordProvenance>::new();
    let mut summaries = Vec::<FoundationCorpusSourceSummary>::new();

    for (source_index, source) in config.sources.iter().enumerate() {
        let mut loader_config = config.loader.clone();
        loader_config.delimiter = Some(source.delimiter.byte(&source.path));
        if let Some(strict) = source.strict {
            loader_config.strict = strict;
        }
        if let Some(ccs_derivation) = source.ccs_derivation {
            loader_config.ccs_derivation = ccs_derivation;
        }
        if let Some(nce) = source.metadata.nce {
            loader_config.default_nce = Some(nce);
        }
        if let Some(instrument) = source.metadata.instrument.clone() {
            loader_config.default_instrument = Some(instrument);
        }
        if let Some(run_id) = source.metadata.run_id.clone() {
            loader_config.default_run_id = Some(run_id);
        }
        if let Some(gradient_seconds) = source.metadata.gradient_seconds {
            loader_config.default_gradient_seconds = Some(gradient_seconds);
        }

        let report = load_source(&mut loader, source, &loader_config)?;
        let record_start = records.len();
        let record_count = report.records.len();
        let local_indices: Vec<usize> = (0..record_count).collect();
        let source_fingerprint = foundation_dataset_fingerprint(&report.records, &local_indices)?;
        for source_record_index in 0..record_count {
            provenance.push(FoundationRecordProvenance {
                source_index,
                source_id: source.id.clone(),
                source_record_index,
            });
        }
        records.extend(report.records);
        summaries.push(FoundationCorpusSourceSummary {
            id: source.id.clone(),
            path: source.path.clone(),
            profile: report.schema.profile,
            record_start,
            record_count,
            dataset_fingerprint: source_fingerprint,
            stats: report.stats,
        });
    }

    let instrument_names = loader.instruments().names().to_vec();
    let corpus_fingerprint = corpus_fingerprint(&records, &provenance, &summaries)?;
    Ok(FoundationCorpus {
        records,
        provenance,
        sources: summaries,
        instrument_names,
        corpus_fingerprint,
    })
}

fn load_source(
    loader: &mut FoundationDatasetLoader,
    source: &FoundationCorpusSourceSpec,
    config: &FoundationTableLoaderConfig,
) -> Result<FoundationTableLoadReport> {
    let path_text = source.path.to_string_lossy().to_ascii_lowercase();
    if !path_text.ends_with(".zst") {
        return loader
            .load_path_with_report(&source.path, config)
            .with_context(|| format!("failed to load corpus source '{}'", source.id));
    }

    let mut child = Command::new("zstd")
        .arg("-dc")
        .arg(&source.path)
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to start 'zstd -dc' for corpus source '{}' ({:?})",
                source.id, source.path
            )
        })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        anyhow!(
            "failed to capture zstd stdout for corpus source '{}'",
            source.id
        )
    })?;
    let report = loader.load_reader_with_report(
        BufReader::new(stdout),
        source.delimiter.byte(&source.path),
        config,
    );
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!(
            "zstd decompression failed for corpus source '{}' with status {}",
            source.id,
            status
        );
    }
    report.with_context(|| format!("failed to parse decompressed corpus source '{}'", source.id))
}

fn corpus_fingerprint(
    records: &[FoundationTrainingRecord],
    provenance: &[FoundationRecordProvenance],
    sources: &[FoundationCorpusSourceSummary],
) -> Result<u64> {
    if records.len() != provenance.len() {
        anyhow::bail!("foundation corpus record/provenance lengths differ");
    }
    let all_indices: Vec<usize> = (0..records.len()).collect();
    let dataset_fingerprint = foundation_dataset_fingerprint(records, &all_indices)?;
    let mut hash = StableFnv64::new();
    hash.u64(dataset_fingerprint);
    hash.usize(records.len());
    for source in sources {
        hash.str(&source.id);
        hash.u64(source.dataset_fingerprint);
        hash.usize(source.record_count);
    }
    for item in provenance {
        hash.str(&item.source_id);
        hash.usize(item.source_record_index);
    }
    Ok(hash.finish())
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
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
