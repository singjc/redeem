//! Tabular dataset loading for foundation-model pretraining and fine-tuning.
//!
//! ReDeeM already consumes several transition-list and spectral-library
//! dialects.  This module provides the foundation model with a deliberately
//! small, schema-tolerant loader that groups long-form fragment rows into one
//! [`FoundationTrainingRecord`](crate::foundation::FoundationTrainingRecord)
//! per precursor observation.  Header matching ignores case, spaces,
//! underscores, and hyphens so OpenSWATH-style and generic TSV/CSV exports can
//! share the same adapter.
//!
//! Normalized RT/iRT and observed chromatographic RT are kept as distinct
//! labels.  This is important for cross-run pretraining: normalized RT is an
//! intrinsic/portable target whereas observed RT depends on LC context.

use super::data::{FoundationTrainingRecord, FragmentTarget, RetentionTimeLabels, TrainingContext};
use super::featurize::{FoundationModification, PeptidoformInput};
use anyhow::{anyhow, Context, Result};
use csv::{ReaderBuilder, StringRecord};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Fragment-intensity normalization applied independently to each precursor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentIntensityNormalization {
    /// Keep input intensities unchanged.
    None,
    /// Divide all fragment intensities by the precursor's maximum intensity.
    #[default]
    Max,
    /// Divide all fragment intensities by the precursor's summed intensity.
    Sum,
}

/// Options controlling schema inference and default acquisition context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationTableLoaderConfig {
    /// Optional explicit delimiter.  When absent, `.tsv` uses tab and all other
    /// paths use comma.
    pub delimiter: Option<u8>,
    /// NCE used when a table has no collision-energy column.
    pub default_nce: Option<f32>,
    /// Instrument label used when a table has no instrument column.
    pub default_instrument: Option<String>,
    /// Optional run id assigned to every row when no run column exists.
    pub default_run_id: Option<String>,
    /// Per-precursor fragment-intensity normalization.
    pub fragment_normalization: FragmentIntensityNormalization,
    /// When true, malformed peptide/modification annotations return an error;
    /// otherwise those rows are skipped.
    pub strict: bool,
}

impl Default for FoundationTableLoaderConfig {
    fn default() -> Self {
        Self {
            delimiter: None,
            default_nce: None,
            default_instrument: None,
            default_run_id: None,
            fragment_normalization: FragmentIntensityNormalization::Max,
            strict: true,
        }
    }
}

/// Stable mapping from instrument names to compact ids used by the context
/// embedding.  Id zero is reserved for unknown/missing instruments.
#[derive(Debug, Clone)]
pub struct InstrumentVocabulary {
    max_size: usize,
    by_name: HashMap<String, u32>,
    names: Vec<String>,
}

impl InstrumentVocabulary {
    /// Create a vocabulary with a fixed maximum number of model categories.
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size: max_size.max(1),
            by_name: HashMap::new(),
            names: vec!["unknown".to_string()],
        }
    }

    /// Return the stable id for `name`, allocating one when capacity permits.
    pub fn id_for(&mut self, name: Option<&str>) -> u32 {
        let Some(name) = name.map(str::trim).filter(|name| !name.is_empty()) else {
            return 0;
        };
        let canonical = name.to_ascii_lowercase();
        if let Some(id) = self.by_name.get(&canonical) {
            return *id;
        }
        if self.names.len() >= self.max_size {
            return 0;
        }
        let id = self.names.len() as u32;
        self.by_name.insert(canonical, id);
        self.names.push(name.to_string());
        id
    }

    /// Instrument labels in id order; element zero is always `unknown`.
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

/// One inferred semantic field in a transition/spectral-library table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationSchemaField {
    /// Zero-based source-column index.
    pub index: usize,
    /// Original source-column header.
    pub header: String,
}

/// One source column claimed by more than one inferred semantic field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationSchemaCollision {
    /// Zero-based source-column index.
    pub index: usize,
    /// Original source-column header.
    pub header: String,
    /// Semantic fields that all resolved to this source column.
    pub semantics: Vec<String>,
}

/// Human-readable schema inferred from a transition or spectral-library table.
///
/// The report intentionally preserves original header names so schema decisions
/// made by the permissive alias matcher can be audited before a large training
/// job starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationTableSchemaReport {
    /// Adapter/profile selected for this header set.
    ///
    /// Known real-data dialects use explicit source-specific mappings; all
    /// other tables fall back to the generic normalized-header matcher.
    pub profile: String,
    /// All source headers in input order.
    pub headers: Vec<String>,
    /// Peptide or modified-peptide column.
    pub sequence: FoundationSchemaField,
    /// Precursor charge column.
    pub charge: Option<FoundationSchemaField>,
    /// Normalized RT/iRT column.
    pub normalized_rt: Option<FoundationSchemaField>,
    /// Observed chromatographic RT column.
    pub observed_rt: Option<FoundationSchemaField>,
    /// Collision cross section column.
    pub ccs: Option<FoundationSchemaField>,
    /// Precursor m/z column.
    pub precursor_mz: Option<FoundationSchemaField>,
    /// Ion-mobility column.
    pub ion_mobility: Option<FoundationSchemaField>,
    /// NCE/collision-energy column.
    pub nce: Option<FoundationSchemaField>,
    /// Instrument column.
    pub instrument: Option<FoundationSchemaField>,
    /// Run/file identifier column.
    pub run_id: Option<FoundationSchemaField>,
    /// LC-gradient-duration column.
    pub gradient_seconds: Option<FoundationSchemaField>,
    /// Fragment ion-series/type column.
    pub fragment_type: Option<FoundationSchemaField>,
    /// Fragment ordinal/series-number column.
    pub fragment_series: Option<FoundationSchemaField>,
    /// Product/fragment charge column.
    pub fragment_charge: Option<FoundationSchemaField>,
    /// Fragment intensity column.
    pub fragment_intensity: Option<FoundationSchemaField>,
    /// Fragment neutral-loss column.
    pub fragment_loss: Option<FoundationSchemaField>,
    /// Source columns claimed by multiple semantic fields.  Non-empty output
    /// should be reviewed before using the table for training.
    pub collisions: Vec<FoundationSchemaCollision>,
}

/// Parse/load statistics emitted while inspecting a real training table.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FoundationTableLoadStats {
    /// Number of physical data rows read after the header.
    pub input_rows: usize,
    /// Number of rows that produced a valid precursor observation.
    pub parsed_rows: usize,
    /// Rows skipped because the peptide field was empty or too short.
    pub skipped_empty_or_short_peptide_rows: usize,
    /// Rows skipped after a parse error in non-strict mode.
    pub skipped_error_rows: usize,
    /// Parsed rows that mapped to a supported MS2 output channel.
    pub supported_fragment_rows: usize,
    /// Number of grouped precursor-level records after aggregation.
    pub precursor_records: usize,
    /// Number of unique unmodified peptide sequences.
    pub unique_sequences: usize,
    /// Number of unique peptidoforms including site-specific mass shifts.
    pub unique_peptidoforms: usize,
    /// Grouped records containing at least one modification.
    pub modified_records: usize,
    /// Records carrying normalized RT/iRT.
    pub normalized_rt_records: usize,
    /// Records carrying observed chromatographic RT.
    pub observed_rt_records: usize,
    /// Records carrying CCS.
    pub ccs_records: usize,
    /// Records carrying at least one supported fragment target.
    pub ms2_records: usize,
    /// Distinct non-empty run identifiers.
    pub unique_runs: usize,
    /// Distinct non-empty instrument labels.
    pub unique_instruments: usize,
    /// Minimum peptide length among grouped records.
    pub min_sequence_len: Option<usize>,
    /// Maximum peptide length among grouped records.
    pub max_sequence_len: Option<usize>,
    /// Mean peptide length among grouped records.
    pub mean_sequence_len: Option<f64>,
    /// Up to a small bounded set of parse-error examples collected in
    /// non-strict mode.
    pub error_examples: Vec<String>,
}

/// Auditable result of loading/inspecting one table.
#[derive(Debug, Clone)]
pub struct FoundationTableLoadReport {
    /// Parsed and grouped precursor records.
    pub records: Vec<FoundationTrainingRecord>,
    /// Schema inferred from the original source headers.
    pub schema: FoundationTableSchemaReport,
    /// Input delimiter used by the CSV reader.
    pub delimiter: u8,
    /// Parse and label-coverage statistics.
    pub stats: FoundationTableLoadStats,
}

/// Loaded foundation records plus the instrument vocabulary inferred from the
/// same table(s).
#[derive(Debug, Clone)]
pub struct FoundationDataset {
    /// One grouped training record per precursor observation.
    pub records: Vec<FoundationTrainingRecord>,
    /// Instrument id mapping used by the records.
    pub instruments: InstrumentVocabulary,
}

/// Stateful loader that can accumulate a consistent instrument vocabulary
/// across multiple training tables.
#[derive(Debug, Clone)]
pub struct FoundationDatasetLoader {
    instruments: InstrumentVocabulary,
}

impl FoundationDatasetLoader {
    /// Create a loader using the model's configured instrument vocabulary size.
    pub fn new(instrument_vocab_size: usize) -> Self {
        Self {
            instruments: InstrumentVocabulary::new(instrument_vocab_size),
        }
    }

    /// Load a CSV/TSV transition table and group long fragment rows by
    /// peptidoform, precursor charge, run, NCE, and instrument.
    pub fn load_path<P: AsRef<Path>>(
        &mut self,
        path: P,
        config: &FoundationTableLoaderConfig,
    ) -> Result<Vec<FoundationTrainingRecord>> {
        Ok(self.load_path_with_report(path, config)?.records)
    }

    /// Load a path and retain the inferred schema plus parse/coverage
    /// statistics.  This is intended for validating unfamiliar real-data
    /// exports before using them for pretraining.
    pub fn load_path_with_report<P: AsRef<Path>>(
        &mut self,
        path: P,
        config: &FoundationTableLoaderConfig,
    ) -> Result<FoundationTableLoadReport> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("failed to open {:?}", path))?;
        let delimiter = infer_delimiter(path, config);
        self.load_reader_with_report(BufReader::new(file), delimiter, config)
            .with_context(|| format!("failed to load foundation table {:?}", path))
    }

    /// Load records from any reader.  This is useful for tests and callers that
    /// already manage decompression/streaming themselves.
    pub fn load_reader<R: std::io::Read>(
        &mut self,
        reader: R,
        delimiter: u8,
        config: &FoundationTableLoaderConfig,
    ) -> Result<Vec<FoundationTrainingRecord>> {
        Ok(self
            .load_reader_with_report(reader, delimiter, config)?
            .records)
    }

    /// Load records and return an auditable schema/coverage report.
    pub fn load_reader_with_report<R: std::io::Read>(
        &mut self,
        reader: R,
        delimiter: u8,
        config: &FoundationTableLoaderConfig,
    ) -> Result<FoundationTableLoadReport> {
        const MAX_ERROR_EXAMPLES: usize = 8;

        let mut csv = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .flexible(true)
            .from_reader(reader);
        let headers = csv.headers()?.clone();
        let schema = TableSchema::infer(&headers)?;
        let schema_report = schema.report(&headers);
        if !schema_report.collisions.is_empty() {
            log::warn!(
                "foundation table schema has {} semantic column collision(s); inspect the schema report before training",
                schema_report.collisions.len()
            );
        }
        let mut records = Vec::<FoundationTrainingRecord>::new();
        let mut group_to_index = HashMap::<String, usize>::new();
        let mut stats = FoundationTableLoadStats::default();

        for row_result in csv.records() {
            stats.input_rows += 1;
            let row = match row_result {
                Ok(row) => row,
                Err(error) if !config.strict => {
                    stats.skipped_error_rows += 1;
                    if stats.error_examples.len() < MAX_ERROR_EXAMPLES {
                        stats.error_examples.push(error.to_string());
                    }
                    log::debug!("skipping malformed foundation CSV row: {error}");
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            let parsed = match self.parse_row(&row, &schema, config) {
                Ok(Some(parsed)) => parsed,
                Ok(None) => {
                    stats.skipped_empty_or_short_peptide_rows += 1;
                    continue;
                }
                Err(error) if !config.strict => {
                    stats.skipped_error_rows += 1;
                    if stats.error_examples.len() < MAX_ERROR_EXAMPLES {
                        stats.error_examples.push(format!("{error:#}"));
                    }
                    log::debug!("skipping foundation row: {error:#}");
                    continue;
                }
                Err(error) => return Err(error),
            };
            stats.parsed_rows += 1;
            if parsed.fragment.is_some() {
                stats.supported_fragment_rows += 1;
            }

            let ParsedRow {
                group_key,
                peptidoform,
                retention_time,
                ccs,
                fragment,
                context,
                run_id,
            } = parsed;
            let index = if let Some(index) = group_to_index.get(&group_key).copied() {
                index
            } else {
                let index = records.len();
                group_to_index.insert(group_key, index);
                records.push(FoundationTrainingRecord {
                    peptidoform,
                    retention_time,
                    ccs,
                    fragments: Vec::new(),
                    context,
                    run_id,
                });
                index
            };
            if let Some(fragment) = fragment {
                records[index].fragments.push(fragment);
            }
        }

        for record in &mut records {
            aggregate_duplicate_fragments(record);
            normalize_fragment_intensities(record, config.fragment_normalization);
        }
        finalize_load_stats(&mut stats, &records);

        Ok(FoundationTableLoadReport {
            records,
            schema: schema_report,
            delimiter,
            stats,
        })
    }

    /// Return the currently accumulated instrument vocabulary.
    pub fn instruments(&self) -> &InstrumentVocabulary {
        &self.instruments
    }

    /// Consume the loader and package records with the inferred vocabulary.
    pub fn finish(self, records: Vec<FoundationTrainingRecord>) -> FoundationDataset {
        FoundationDataset {
            records,
            instruments: self.instruments,
        }
    }

    fn parse_row(
        &mut self,
        row: &StringRecord,
        schema: &TableSchema,
        config: &FoundationTableLoaderConfig,
    ) -> Result<Option<ParsedRow>> {
        let raw_sequence = field(row, Some(schema.sequence)).unwrap_or("").trim();
        if raw_sequence.is_empty() {
            return Ok(None);
        }
        let peptidoform = parse_modified_peptide(raw_sequence)?;
        if peptidoform.sequence.len() < 2 {
            return Ok(None);
        }

        let charge = parse_i32(field(row, schema.charge));
        let nce = parse_f32(field(row, schema.nce)).or(config.default_nce);
        let instrument_name = field(row, schema.instrument)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| config.default_instrument.clone());
        let instrument_id = self.instruments.id_for(instrument_name.as_deref());
        let run_id = field(row, schema.run_id)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| config.default_run_id.clone());
        let normalized_rt = parse_f32(field(row, schema.normalized_rt));
        let observed_rt = parse_f32(field(row, schema.observed_rt));
        let ccs = parse_f32(field(row, schema.ccs));
        let precursor_mz = parse_f32(field(row, schema.precursor_mz));
        let ion_mobility = parse_f32(field(row, schema.ion_mobility));
        let gradient_seconds = parse_f32(field(row, schema.gradient_seconds));

        let fragment = parse_fragment(row, schema, peptidoform.sequence.len())?;
        let group_key = format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
            raw_sequence,
            charge.unwrap_or_default(),
            run_id.as_deref().unwrap_or(""),
            nce.map(|value| format!("{value:.4}")).unwrap_or_default(),
            instrument_name.as_deref().unwrap_or("")
        );

        Ok(Some(ParsedRow {
            group_key,
            peptidoform,
            retention_time: RetentionTimeLabels {
                normalized: normalized_rt,
                observed_seconds: observed_rt,
            },
            ccs,
            fragment,
            context: TrainingContext {
                charge,
                precursor_mz,
                nce,
                instrument_id: Some(instrument_id),
                instrument_name,
                ion_mobility,
                gradient_seconds,
            },
            run_id,
        }))
    }
}

#[derive(Debug)]
struct ParsedRow {
    group_key: String,
    peptidoform: PeptidoformInput,
    retention_time: RetentionTimeLabels,
    ccs: Option<f32>,
    fragment: Option<FragmentTarget>,
    context: TrainingContext,
    run_id: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct TableSchema {
    profile: &'static str,
    sequence: usize,
    charge: Option<usize>,
    normalized_rt: Option<usize>,
    observed_rt: Option<usize>,
    ccs: Option<usize>,
    precursor_mz: Option<usize>,
    ion_mobility: Option<usize>,
    nce: Option<usize>,
    instrument: Option<usize>,
    run_id: Option<usize>,
    gradient_seconds: Option<usize>,
    fragment_type: Option<usize>,
    fragment_series: Option<usize>,
    fragment_charge: Option<usize>,
    fragment_intensity: Option<usize>,
    fragment_loss: Option<usize>,
}

impl TableSchema {
    fn infer(headers: &StringRecord) -> Result<Self> {
        if is_ip2_bruker_spectral_library(headers) {
            return Self::ip2_bruker(headers);
        }
        if is_openswath_finetuning_table(headers) {
            return Self::openswath_finetuning(headers);
        }
        Self::generic(headers)
    }

    fn openswath_finetuning(headers: &StringRecord) -> Result<Self> {
        Ok(Self {
            profile: "openswath_finetuning",
            sequence: require_header(headers, "sequence")?,
            charge: exact_header(headers, "precursor_charge"),
            normalized_rt: None,
            observed_rt: exact_header(headers, "retention_time"),
            ccs: None,
            precursor_mz: exact_header(headers, "precursor_mz"),
            ion_mobility: exact_header(headers, "ion_mobility"),
            nce: None,
            instrument: None,
            run_id: None,
            gradient_seconds: None,
            fragment_type: exact_header(headers, "fragment_type"),
            fragment_series: exact_header(headers, "fragment_series_number"),
            fragment_charge: exact_header(headers, "product_charge"),
            fragment_intensity: exact_header(headers, "intensity"),
            fragment_loss: None,
        })
    }

    fn ip2_bruker(headers: &StringRecord) -> Result<Self> {
        Ok(Self {
            profile: "ip2_bruker_spectral_library",
            // This table contains both naked and modified peptide columns.
            // Foundation chemistry must consume the modified form or PTM
            // supervision silently disappears.
            sequence: require_header(headers, "ModifiedPeptideSequence")?,
            charge: exact_header(headers, "PrecursorCharge"),
            normalized_rt: exact_header(headers, "NormalizedRetentionTime"),
            observed_rt: None,
            ccs: None,
            precursor_mz: exact_header(headers, "PrecursorMz"),
            ion_mobility: exact_header(headers, "PrecursorIonMobility"),
            nce: None,
            instrument: None,
            run_id: None,
            gradient_seconds: None,
            fragment_type: exact_header(headers, "FragmentType"),
            fragment_series: exact_header(headers, "FragmentSeriesNumber"),
            fragment_charge: exact_header(headers, "FragmentCharge"),
            fragment_intensity: exact_header(headers, "LibraryIntensity"),
            fragment_loss: exact_header(headers, "FragmentLossType"),
        })
    }

    fn generic(headers: &StringRecord) -> Result<Self> {
        let sequence = find_header(
            headers,
            &[
                "modifiedpeptidesequence",
                "modifiedpeptide",
                "fullpeptidename",
                "modified_sequence",
                "modifiedsequence",
                "sequence",
                "naked_sequence",
                "peptide_sequence",
                "peptide",
            ],
        )
        .ok_or_else(|| anyhow!("no peptide-sequence column found in table headers"))?;
        let normalized_rt = find_header(
            headers,
            &[
                "normalizedretentiontime",
                "normalized_retention_time",
                "irt",
                "indexedretentiontime",
                "normalizedrt",
            ],
        );
        let observed_rt = find_header_excluding(
            headers,
            &[
                "observedretentiontime",
                "observed_retention_time",
                "retention_time",
                "retentiontime",
                "retention time",
                "rtseconds",
                "rt",
            ],
            normalized_rt,
        );
        Ok(Self {
            profile: "generic",
            sequence,
            charge: find_header(headers, &["precursorcharge", "precursor_charge", "charge"]),
            normalized_rt,
            observed_rt,
            ccs: find_header(headers, &["ccs", "collisioncrosssection"]),
            precursor_mz: find_header(
                headers,
                &[
                    "precursormz",
                    "precursor_mz",
                    "precursor mass",
                    "precursor_mass",
                ],
            ),
            ion_mobility: find_header(
                headers,
                &["precursorionmobility", "ion_mobility", "ionmobility", "im"],
            ),
            nce: find_header(
                headers,
                &["collisionenergy", "normalizedcollisionenergy", "nce"],
            ),
            instrument: find_header(headers, &["instrument", "instrumenttype", "massanalyzer"]),
            run_id: find_header(
                headers,
                &["run", "runid", "run_id", "filename", "file", "rawfile"],
            ),
            gradient_seconds: find_header(
                headers,
                &[
                    "gradientseconds",
                    "gradient_seconds",
                    "gradienttime",
                    "gradient_time",
                ],
            ),
            fragment_type: find_header(
                headers,
                &["fragmenttype", "fragment_type", "iontype", "ion_type"],
            ),
            fragment_series: find_header(
                headers,
                &[
                    "fragmentseriesnumber",
                    "fragment_series_number",
                    "seriesnumber",
                    "series_number",
                    "ordinal",
                ],
            ),
            fragment_charge: find_header(
                headers,
                &[
                    "productcharge",
                    "product_charge",
                    "fragmentcharge",
                    "fragment_charge",
                ],
            ),
            fragment_intensity: find_header(
                headers,
                &[
                    "libraryintensity",
                    "library_intensity",
                    "intensity",
                    "relativeintensity",
                ],
            ),
            fragment_loss: find_header(
                headers,
                &[
                    "fragmentlosstype",
                    "fragment_loss_type",
                    "neutralloss",
                    "neutral_loss",
                    "loss",
                ],
            ),
        })
    }

    fn report(&self, headers: &StringRecord) -> FoundationTableSchemaReport {
        FoundationTableSchemaReport {
            profile: self.profile.to_string(),
            headers: headers.iter().map(ToOwned::to_owned).collect(),
            sequence: FoundationSchemaField {
                index: self.sequence,
                header: headers.get(self.sequence).unwrap_or_default().to_string(),
            },
            charge: schema_field(headers, self.charge),
            normalized_rt: schema_field(headers, self.normalized_rt),
            observed_rt: schema_field(headers, self.observed_rt),
            ccs: schema_field(headers, self.ccs),
            precursor_mz: schema_field(headers, self.precursor_mz),
            ion_mobility: schema_field(headers, self.ion_mobility),
            nce: schema_field(headers, self.nce),
            instrument: schema_field(headers, self.instrument),
            run_id: schema_field(headers, self.run_id),
            gradient_seconds: schema_field(headers, self.gradient_seconds),
            fragment_type: schema_field(headers, self.fragment_type),
            fragment_series: schema_field(headers, self.fragment_series),
            fragment_charge: schema_field(headers, self.fragment_charge),
            fragment_intensity: schema_field(headers, self.fragment_intensity),
            fragment_loss: schema_field(headers, self.fragment_loss),
            collisions: schema_collisions(self, headers),
        }
    }
}

fn normalized_headers(headers: &StringRecord) -> BTreeSet<String> {
    headers.iter().map(normalize_header).collect()
}

fn is_openswath_finetuning_table(headers: &StringRecord) -> bool {
    let headers = normalized_headers(headers);
    [
        "sequence",
        "precursormz",
        "precursorcharge",
        "fragmenttype",
        "fragmentseriesnumber",
        "productcharge",
        "retentiontime",
        "ionmobility",
        "intensity",
    ]
    .iter()
    .all(|header| headers.contains(*header))
}

fn is_ip2_bruker_spectral_library(headers: &StringRecord) -> bool {
    let headers = normalized_headers(headers);
    [
        "peptidesequence",
        "modifiedpeptidesequence",
        "precursorcharge",
        "libraryintensity",
        "normalizedretentiontime",
        "precursorionmobility",
        "fragmenttype",
        "fragmentcharge",
        "fragmentseriesnumber",
        "fragmentlosstype",
    ]
    .iter()
    .all(|header| headers.contains(*header))
}

fn exact_header(headers: &StringRecord, name: &str) -> Option<usize> {
    let name = normalize_header(name);
    headers
        .iter()
        .position(|header| normalize_header(header) == name)
}

fn require_header(headers: &StringRecord, name: &str) -> Result<usize> {
    exact_header(headers, name)
        .ok_or_else(|| anyhow!("required foundation table column '{name}' was not found"))
}

fn infer_delimiter(path: &Path, config: &FoundationTableLoaderConfig) -> u8 {
    config.delimiter.unwrap_or_else(|| {
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if extension == "tsv" {
            b'\t'
        } else {
            b','
        }
    })
}

fn schema_field(headers: &StringRecord, index: Option<usize>) -> Option<FoundationSchemaField> {
    index.and_then(|index| {
        headers.get(index).map(|header| FoundationSchemaField {
            index,
            header: header.to_string(),
        })
    })
}

fn schema_collisions(
    schema: &TableSchema,
    headers: &StringRecord,
) -> Vec<FoundationSchemaCollision> {
    let fields = [
        ("sequence", Some(schema.sequence)),
        ("charge", schema.charge),
        ("normalized_rt", schema.normalized_rt),
        ("observed_rt", schema.observed_rt),
        ("ccs", schema.ccs),
        ("precursor_mz", schema.precursor_mz),
        ("ion_mobility", schema.ion_mobility),
        ("nce", schema.nce),
        ("instrument", schema.instrument),
        ("run_id", schema.run_id),
        ("gradient_seconds", schema.gradient_seconds),
        ("fragment_type", schema.fragment_type),
        ("fragment_series", schema.fragment_series),
        ("fragment_charge", schema.fragment_charge),
        ("fragment_intensity", schema.fragment_intensity),
        ("fragment_loss", schema.fragment_loss),
    ];
    let mut by_index = BTreeMap::<usize, Vec<String>>::new();
    for (semantic, index) in fields {
        if let Some(index) = index {
            by_index
                .entry(index)
                .or_default()
                .push(semantic.to_string());
        }
    }
    by_index
        .into_iter()
        .filter_map(|(index, semantics)| {
            (semantics.len() > 1).then(|| FoundationSchemaCollision {
                index,
                header: headers.get(index).unwrap_or_default().to_string(),
                semantics,
            })
        })
        .collect()
}

fn finalize_load_stats(stats: &mut FoundationTableLoadStats, records: &[FoundationTrainingRecord]) {
    stats.precursor_records = records.len();
    let mut sequences = BTreeSet::<String>::new();
    let mut peptidoforms = BTreeSet::<String>::new();
    let mut runs = BTreeSet::<String>::new();
    let mut instruments = BTreeSet::<String>::new();
    let mut total_sequence_len = 0usize;

    for record in records {
        let sequence_len = record.peptidoform.sequence.len();
        total_sequence_len += sequence_len;
        stats.min_sequence_len = Some(
            stats
                .min_sequence_len
                .map_or(sequence_len, |current| current.min(sequence_len)),
        );
        stats.max_sequence_len = Some(
            stats
                .max_sequence_len
                .map_or(sequence_len, |current| current.max(sequence_len)),
        );
        sequences.insert(record.peptidoform.sequence.clone());
        peptidoforms.insert(canonical_peptidoform_label(&record.peptidoform));
        if !record.peptidoform.modifications.is_empty() {
            stats.modified_records += 1;
        }
        if record.retention_time.normalized.is_some() {
            stats.normalized_rt_records += 1;
        }
        if record.retention_time.observed_seconds.is_some() {
            stats.observed_rt_records += 1;
        }
        if record.ccs.is_some() {
            stats.ccs_records += 1;
        }
        if !record.fragments.is_empty() {
            stats.ms2_records += 1;
        }
        if let Some(run_id) = record
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            runs.insert(run_id.to_string());
        }
        if let Some(instrument) = record
            .context
            .instrument_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            instruments.insert(instrument.to_ascii_lowercase());
        }
    }

    stats.unique_sequences = sequences.len();
    stats.unique_peptidoforms = peptidoforms.len();
    stats.unique_runs = runs.len();
    stats.unique_instruments = instruments.len();
    stats.mean_sequence_len =
        (!records.is_empty()).then_some(total_sequence_len as f64 / records.len() as f64);
}

pub(crate) fn canonical_peptidoform_label(peptidoform: &PeptidoformInput) -> String {
    let mut modifications = peptidoform.modifications.clone();
    modifications.sort_by(|left, right| {
        left.residue_index.cmp(&right.residue_index).then_with(|| {
            left.mass_delta
                .partial_cmp(&right.mass_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    let mut label = peptidoform.sequence.clone();
    for modification in modifications {
        label.push('|');
        label.push_str(&format!(
            "{}:{:+.4}",
            modification.residue_index, modification.mass_delta
        ));
    }
    label
}

fn normalize_header(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn find_header(headers: &StringRecord, aliases: &[&str]) -> Option<usize> {
    let aliases: Vec<String> = aliases
        .iter()
        .map(|alias| normalize_header(alias))
        .collect();
    for alias in &aliases {
        if let Some(index) = headers
            .iter()
            .position(|header| normalize_header(header) == *alias)
        {
            return Some(index);
        }
    }

    // Fuzzy substring matching is intentionally disabled for very short aliases
    // such as "nce", "rt", and "im".  For example, "sequence" ends in "nce",
    // which previously caused the peptide column to be misidentified as NCE.
    for alias in aliases.iter().filter(|alias| alias.len() >= 5) {
        if let Some(index) = headers.iter().position(|header| {
            let header = normalize_header(header);
            header.starts_with(alias) || header.ends_with(alias)
        }) {
            return Some(index);
        }
    }
    None
}

fn find_header_excluding(
    headers: &StringRecord,
    aliases: &[&str],
    excluded_index: Option<usize>,
) -> Option<usize> {
    let aliases: Vec<String> = aliases
        .iter()
        .map(|alias| normalize_header(alias))
        .collect();
    for alias in &aliases {
        if let Some(index) = headers.iter().enumerate().find_map(|(index, header)| {
            (Some(index) != excluded_index && normalize_header(header) == *alias).then_some(index)
        }) {
            return Some(index);
        }
    }
    for alias in aliases.iter().filter(|alias| alias.len() >= 5) {
        if let Some(index) = headers.iter().enumerate().find_map(|(index, header)| {
            if Some(index) == excluded_index {
                return None;
            }
            let header = normalize_header(header);
            (header.starts_with(alias) || header.ends_with(alias)).then_some(index)
        }) {
            return Some(index);
        }
    }
    None
}

fn field<'a>(row: &'a StringRecord, index: Option<usize>) -> Option<&'a str> {
    index.and_then(|index| row.get(index))
}

fn parse_f32(value: Option<&str>) -> Option<f32> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
}

fn parse_i32(value: Option<&str>) -> Option<i32> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<i32>().ok())
}

fn parse_fragment(
    row: &StringRecord,
    schema: &TableSchema,
    peptide_len: usize,
) -> Result<Option<FragmentTarget>> {
    let (Some(type_index), Some(series_index), Some(intensity_index)) = (
        schema.fragment_type,
        schema.fragment_series,
        schema.fragment_intensity,
    ) else {
        return Ok(None);
    };
    let fragment_type = row
        .get(type_index)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let ordinal = row
        .get(series_index)
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let intensity = row
        .get(intensity_index)
        .and_then(|value| value.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    if ordinal == 0 || !intensity.is_finite() || intensity < 0.0 {
        return Ok(None);
    }
    let charge = field(row, schema.fragment_charge)
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1);
    let loss = field(row, schema.fragment_loss)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let channel = fragment_channel(&fragment_type, charge, &loss)?;
    let Some(channel) = channel else {
        return Ok(None);
    };
    let cleavage_index = match fragment_type.as_str() {
        "b" => ordinal.checked_sub(1),
        "y" => peptide_len.checked_sub(ordinal + 1),
        _ => None,
    };
    let Some(cleavage_index) =
        cleavage_index.filter(|index| *index < peptide_len.saturating_sub(1))
    else {
        return Ok(None);
    };
    Ok(Some(FragmentTarget {
        cleavage_index,
        channel,
        intensity,
    }))
}

fn fragment_channel(fragment_type: &str, charge: usize, loss: &str) -> Result<Option<usize>> {
    let loss = loss.replace(' ', "").replace('_', "").replace('-', "");
    let no_loss = loss.is_empty() || loss == "noloss" || loss == "none";
    if no_loss {
        return Ok(match (fragment_type, charge) {
            ("b", 1) => Some(0),
            ("b", 2) => Some(1),
            ("y", 1) => Some(2),
            ("y", 2) => Some(3),
            _ => None,
        });
    }
    let is_water = loss.contains("h2o") || loss.contains("water");
    let is_ammonia = loss.contains("nh3") || loss.contains("ammonia");
    Ok(match (fragment_type, is_water, is_ammonia) {
        ("b", true, false) => Some(4),
        ("y", true, false) => Some(5),
        ("b", false, true) => Some(6),
        ("y", false, true) => Some(7),
        (_, false, false) => None,
        _ => return Err(anyhow!("ambiguous fragment neutral loss '{loss}'")),
    })
}

fn aggregate_duplicate_fragments(record: &mut FoundationTrainingRecord) {
    let mut merged = HashMap::<(usize, usize), f32>::new();
    for fragment in record.fragments.drain(..) {
        *merged
            .entry((fragment.cleavage_index, fragment.channel))
            .or_insert(0.0) += fragment.intensity;
    }
    let mut fragments: Vec<_> = merged
        .into_iter()
        .map(|((cleavage_index, channel), intensity)| FragmentTarget {
            cleavage_index,
            channel,
            intensity,
        })
        .collect();
    fragments.sort_by_key(|fragment| (fragment.cleavage_index, fragment.channel));
    record.fragments = fragments;
}

fn normalize_fragment_intensities(
    record: &mut FoundationTrainingRecord,
    normalization: FragmentIntensityNormalization,
) {
    let denominator = match normalization {
        FragmentIntensityNormalization::None => return,
        FragmentIntensityNormalization::Max => record
            .fragments
            .iter()
            .map(|fragment| fragment.intensity)
            .fold(0.0f32, f32::max),
        FragmentIntensityNormalization::Sum => record
            .fragments
            .iter()
            .map(|fragment| fragment.intensity)
            .sum::<f32>(),
    };
    if denominator > 0.0 {
        for fragment in &mut record.fragments {
            fragment.intensity /= denominator;
        }
    }
}

/// Parse a peptide carrying numeric mass-shift annotations into the foundation
/// peptidoform representation.
///
/// Supported forms include `M[+15.9949]`, `C(+57.0215)`, and common
/// `UniMod:<id>` annotations.  Flanking-residue notation such as
/// `K.PEPTIDE.R` is also accepted.  Unknown named modifications return an
/// explicit error rather than silently becoming zero-mass modifications.
pub fn parse_modified_peptide(raw: &str) -> Result<PeptidoformInput> {
    let raw = strip_flanking_residues(raw.trim());
    // Some OpenSWATH exports use a terminal sentinel dot before an N-terminal
    // modification, e.g. `.(UniMod:1)PEPTIDE`.  It is not a residue/flanking
    // amino acid and should not make an otherwise valid peptidoform fail.
    let raw = raw.trim_matches('.');
    let chars: Vec<char> = raw.chars().collect();
    let mut sequence = String::new();
    let mut modifications = Vec::new();
    let mut index = 0usize;
    let mut last_residue = None::<usize>;

    while index < chars.len() {
        let ch = chars[index];
        if ch.is_ascii_alphabetic() {
            let residue = ch.to_ascii_uppercase();
            if !"ACDEFGHIKLMNPQRSTVWY".contains(residue) {
                return Err(anyhow!("unsupported residue '{ch}' in '{raw}'"));
            }
            sequence.push(residue);
            last_residue = Some(sequence.len() - 1);
            index += 1;
            continue;
        }
        if ch == '[' || ch == '(' {
            let closing = if ch == '[' { ']' } else { ')' };
            let start = index + 1;
            let mut end = start;
            while end < chars.len() && chars[end] != closing {
                end += 1;
            }
            if end >= chars.len() {
                return Err(anyhow!("unterminated modification in '{raw}'"));
            }
            let token: String = chars[start..end].iter().collect();
            let mass_delta = modification_mass_delta(&token)?;
            let residue_index = last_residue.unwrap_or(0);
            modifications.push(FoundationModification {
                residue_index,
                mass_delta,
            });
            index = end + 1;
            continue;
        }
        if ch == '_' || ch == '-' || ch.is_whitespace() {
            index += 1;
            continue;
        }
        return Err(anyhow!("unexpected character '{ch}' in peptide '{raw}'"));
    }

    if sequence.is_empty() {
        return Err(anyhow!("peptide '{raw}' contains no amino acids"));
    }
    Ok(PeptidoformInput {
        sequence,
        modifications,
    })
}

fn strip_flanking_residues(raw: &str) -> &str {
    let mut dots = raw.match_indices('.');
    let first = dots.next().map(|(index, _)| index);
    let second = dots.next().map(|(index, _)| index);
    if let (Some(first), Some(second)) = (first, second) {
        if first <= 2 && second > first + 1 && second + 1 < raw.len() {
            return &raw[first + 1..second];
        }
    }
    raw
}

fn modification_mass_delta(token: &str) -> Result<f32> {
    let normalized = token.trim().trim_start_matches('+').replace(' ', "");
    if let Ok(value) = normalized.parse::<f32>() {
        return Ok(value);
    }
    let lowercase = normalized.to_ascii_lowercase();
    if let Some(id) = lowercase
        .strip_prefix("unimod:")
        .and_then(|value| value.parse::<u32>().ok())
    {
        return common_unimod_mass(id)
            .ok_or_else(|| anyhow!("UniMod:{id} is not in the built-in foundation mass resolver"));
    }
    Err(anyhow!(
        "unsupported modification annotation '{token}'; provide a numeric mass delta or supported UniMod id"
    ))
}

fn common_unimod_mass(id: u32) -> Option<f32> {
    match id {
        1 => Some(42.010_567),    // Acetyl
        4 => Some(57.021_465),    // Carbamidomethyl
        7 => Some(0.984_016),     // Deamidated
        21 => Some(79.966_33),    // Phospho
        35 => Some(15.994_915),   // Oxidation
        737 => Some(229.162_93),  // TMT6plex
        2016 => Some(304.207_15), // TMTpro
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numeric_and_unimod_modifications() {
        let peptide = parse_modified_peptide("SKEEET[+79.9663]SIDM(UniMod:35)K").unwrap();
        assert_eq!(peptide.sequence, "SKEEETSIDMK");
        assert_eq!(peptide.modifications.len(), 2);
        assert_eq!(peptide.modifications[0].residue_index, 5);
        assert!((peptide.modifications[0].mass_delta - 79.9663).abs() < 1e-4);
        assert!((peptide.modifications[1].mass_delta - 15.994915).abs() < 1e-5);
    }

    #[test]
    fn parses_openswath_terminal_sentinel_modification() {
        let peptide = parse_modified_peptide(".(UniMod:1)AAAAAAGAASGLPGPVAQGLK").unwrap();
        assert_eq!(peptide.sequence, "AAAAAAGAASGLPGPVAQGLK");
        assert_eq!(peptide.modifications.len(), 1);
        assert_eq!(peptide.modifications[0].residue_index, 0);
        assert!((peptide.modifications[0].mass_delta - 42.010567).abs() < 1e-5);
    }

    #[test]
    fn loads_and_groups_long_transition_rows() {
        let table = concat!(
            "ModifiedPeptide\tPrecursorCharge\tFragmentType\tFragmentSeriesNumber\tProductCharge\tLibraryIntensity\tNormalizedRetentionTime\tCCS\tCollisionEnergy\tInstrument\n",
            "PEPTIDEK\t2\tb\t2\t1\t50\t31.5\t410.0\t27\ttimsTOF\n",
            "PEPTIDEK\t2\ty\t3\t1\t100\t31.5\t410.0\t27\ttimsTOF\n",
        );
        let mut loader = FoundationDatasetLoader::new(8);
        let records = loader
            .load_reader(
                table.as_bytes(),
                b'\t',
                &FoundationTableLoaderConfig::default(),
            )
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].fragments.len(), 2);
        assert_eq!(records[0].retention_time.normalized, Some(31.5));
        assert_eq!(records[0].retention_time.observed_seconds, None);
        assert_eq!(records[0].ccs, Some(410.0));
        assert_eq!(records[0].context.charge, Some(2));
        assert_eq!(records[0].context.instrument_id, Some(1));
        assert_eq!(records[0].fragments[1].intensity, 1.0);
    }

    #[test]
    fn keeps_normalized_and_observed_retention_time_columns_distinct() {
        let table = concat!(
            "ModifiedPeptide\tPrecursorCharge\tNormalizedRetentionTime\tRetentionTime\n",
            "PEPTIDEK\t2\t31.5\t1800.0\n",
        );
        let mut loader = FoundationDatasetLoader::new(8);
        let records = loader
            .load_reader(
                table.as_bytes(),
                b'\t',
                &FoundationTableLoaderConfig::default(),
            )
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].retention_time.normalized, Some(31.5));
        assert_eq!(records[0].retention_time.observed_seconds, Some(1800.0));
    }
}
