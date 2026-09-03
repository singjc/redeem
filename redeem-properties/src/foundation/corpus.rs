//! Multi-source corpus assembly with source provenance and corpus-wide splits.
//!
//! Foundation pretraining commonly combines transition libraries produced by
//! different pipelines. This module keeps source identity beside each grouped
//! precursor record while using one shared loader/instrument vocabulary. Split
//! manifests are then generated over the *combined* record collection so a
//! sequence cannot leak across sources.

use super::dataset::{
    FoundationCcsDerivationMode, FoundationDatasetLoader, FoundationTableLoadStats,
    FoundationTableLoaderConfig,
};
use super::experiment::{
    build_foundation_benchmark_manifest, foundation_dataset_fingerprint,
    FoundationBenchmarkManifest,
};
use super::metadata::FoundationSourceMetadata;
use super::msp::load_foundation_msp_reader;
use super::rt_harmonization::{
    apply_foundation_rt_harmonization, FoundationRtHarmonizationTransform,
};
use super::spectrum::foundation_diffusion_dataset_fingerprint;
use super::split::FoundationSplitConfig;
use super::FoundationTrainingRecord;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Human-readable delimiter selector for YAML corpus configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FoundationCorpusDelimiter {
    /// Infer from extension; `.tsv`, `.tsv.zst`, and `.tsv.gz` use tab, otherwise comma.
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
                if name.ends_with(".tsv") || name.ends_with(".tsv.zst") || name.ends_with(".tsv.gz")
                {
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

/// Physical source format for one foundation-corpus entry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FoundationCorpusSourceFormat {
    /// Infer MSP from `.msp`, `.msp.gz`, or `.msp.zst`; otherwise use the
    /// long-form transition/spectral-library table loader.
    #[default]
    Auto,
    /// Long-form CSV/TSV transition or spectral-library table.
    Table,
    /// Entry-oriented MSP spectral library with raw observed peak lists.
    Msp,
}

impl FoundationCorpusSourceFormat {
    fn resolve(self, path: &Path) -> Self {
        if self != Self::Auto {
            return self;
        }
        let name = path.to_string_lossy().to_ascii_lowercase();
        if name.ends_with(".msp") || name.ends_with(".msp.gz") || name.ends_with(".msp.zst") {
            Self::Msp
        } else {
            Self::Table
        }
    }
}

/// One source table or spectral library in a multi-source foundation corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationCorpusSourceSpec {
    /// Stable logical source identifier used in provenance/fingerprints.
    ///
    /// YAML accepts either the canonical `id:` spelling or the friendlier
    /// `name:` alias for compatibility with early foundation examples.
    #[serde(alias = "name")]
    pub id: String,
    /// Source path. `.zst` and `.gz` inputs are streamed through the system
    /// `zstd -dc` / `gzip -dc` commands respectively.
    pub path: PathBuf,
    /// Physical source format. `auto` recognizes MSP by extension.
    pub format: FoundationCorpusSourceFormat,
    /// Source delimiter for tabular inputs. Ignored for MSP.
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
    /// Optional TRAIN-fit affine transform from this source's native normalized RT
    /// coordinate into the common harmonized intrinsic RT coordinate.
    pub rt_harmonization: Option<FoundationRtHarmonizationTransform>,
}

impl Default for FoundationCorpusSourceSpec {
    fn default() -> Self {
        Self {
            id: String::new(),
            path: PathBuf::new(),
            format: FoundationCorpusSourceFormat::Auto,
            delimiter: FoundationCorpusDelimiter::Auto,
            metadata: FoundationSourceMetadata::default(),
            strict: None,
            ccs_derivation: None,
            rt_harmonization: None,
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
    /// Source table parse/coverage statistics in source-native coordinates.
    pub stats: FoundationTableLoadStats,
    /// Records carrying the train-calibrated harmonized RT target.
    pub harmonized_rt_records: usize,
    /// Minimum harmonized RT among finite records.
    pub min_harmonized_rt: Option<f64>,
    /// Mean harmonized RT among finite records.
    pub mean_harmonized_rt: Option<f64>,
    /// Maximum harmonized RT among finite records.
    pub max_harmonized_rt: Option<f64>,
    /// Calibration id applied to this source, when present.
    pub rt_harmonization_calibration_id: Option<String>,
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
        if let Some(transform) = &source.rt_harmonization {
            transform.validate().with_context(|| {
                format!(
                    "invalid RT harmonization transform for source '{}'",
                    source.id
                )
            })?;
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

        let mut report = load_source(&mut loader, source, &loader_config)?;
        if let Some(transform) = &source.rt_harmonization {
            for record in &mut report.records {
                apply_foundation_rt_harmonization(record, transform)?;
            }
        }
        let harmonized_values: Vec<f64> = report
            .records
            .iter()
            .filter_map(|record| record.retention_time.harmonized)
            .filter(|value| value.is_finite())
            .map(f64::from)
            .collect();
        let harmonized_rt_records = harmonized_values.len();
        let min_harmonized_rt = harmonized_values.iter().copied().reduce(f64::min);
        let max_harmonized_rt = harmonized_values.iter().copied().reduce(f64::max);
        let mean_harmonized_rt = (!harmonized_values.is_empty())
            .then(|| harmonized_values.iter().sum::<f64>() / harmonized_values.len() as f64);
        let record_start = records.len();
        let record_count = report.records.len();
        let local_indices: Vec<usize> = (0..record_count).collect();
        let source_fingerprint = if report
            .records
            .iter()
            .any(|record| !record.observed_spectrum_peaks.is_empty())
        {
            foundation_diffusion_dataset_fingerprint(&report.records, &local_indices)?
        } else {
            foundation_dataset_fingerprint(&report.records, &local_indices)?
        };
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
            profile: report.profile,
            record_start,
            record_count,
            dataset_fingerprint: source_fingerprint,
            stats: report.stats,
            harmonized_rt_records,
            min_harmonized_rt,
            mean_harmonized_rt,
            max_harmonized_rt,
            rt_harmonization_calibration_id: source
                .rt_harmonization
                .as_ref()
                .map(|transform| transform.calibration_id.clone()),
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

struct FoundationSourceLoadReport {
    records: Vec<FoundationTrainingRecord>,
    profile: String,
    stats: FoundationTableLoadStats,
}

fn load_source(
    loader: &mut FoundationDatasetLoader,
    source: &FoundationCorpusSourceSpec,
    config: &FoundationTableLoaderConfig,
) -> Result<FoundationSourceLoadReport> {
    match source.format.resolve(&source.path) {
        FoundationCorpusSourceFormat::Auto => {
            unreachable!("source format must resolve before load")
        }
        FoundationCorpusSourceFormat::Table => with_source_reader(source, |reader| {
            let report = loader.load_reader_with_report(
                reader,
                source.delimiter.byte(&source.path),
                config,
            )?;
            Ok(FoundationSourceLoadReport {
                records: report.records,
                profile: report.schema.profile,
                stats: report.stats,
            })
        }),
        FoundationCorpusSourceFormat::Msp => with_source_reader(source, |reader| {
            let report = load_foundation_msp_reader(reader, loader, config)?;
            Ok(FoundationSourceLoadReport {
                records: report.records,
                profile: "msp_spectral_library".to_string(),
                stats: report.stats,
            })
        }),
    }
    .with_context(|| format!("failed to load corpus source '{}'", source.id))
}

fn with_source_reader<T>(
    source: &FoundationCorpusSourceSpec,
    parse: impl FnOnce(Box<dyn BufRead>) -> Result<T>,
) -> Result<T> {
    let path_text = source.path.to_string_lossy().to_ascii_lowercase();
    if path_text.ends_with(".zst") {
        return with_decompressor(source, "zstd", &["-dc"], parse);
    }
    if path_text.ends_with(".gz") {
        return with_decompressor(source, "gzip", &["-dc"], parse);
    }
    let file = File::open(&source.path).with_context(|| {
        format!(
            "failed to open corpus source '{}' ({:?})",
            source.id, source.path
        )
    })?;
    parse(Box::new(BufReader::new(file)))
}

fn with_decompressor<T>(
    source: &FoundationCorpusSourceSpec,
    command: &str,
    args: &[&str],
    parse: impl FnOnce(Box<dyn BufRead>) -> Result<T>,
) -> Result<T> {
    let mut child = Command::new(command)
        .args(args)
        .arg(&source.path)
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to start '{command} {}' for corpus source '{}' ({:?})",
                args.join(" "),
                source.id,
                source.path
            )
        })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        anyhow!(
            "failed to capture {command} stdout for corpus source '{}'",
            source.id
        )
    })?;
    let result = parse(Box::new(BufReader::new(stdout)));
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!(
            "{command} decompression failed for corpus source '{}' with status {}",
            source.id,
            status
        );
    }
    result
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
