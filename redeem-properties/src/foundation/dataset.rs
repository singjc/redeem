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
use std::collections::HashMap;
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
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("failed to open {:?}", path))?;
        let delimiter = config.delimiter.unwrap_or_else(|| {
            if path.extension().and_then(|value| value.to_str()) == Some("tsv") {
                b'\t'
            } else {
                b','
            }
        });
        self.load_reader(BufReader::new(file), delimiter, config)
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
        let mut csv = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .flexible(true)
            .from_reader(reader);
        let headers = csv.headers()?.clone();
        let schema = TableSchema::infer(&headers)?;
        let mut records = Vec::<FoundationTrainingRecord>::new();
        let mut group_to_index = HashMap::<String, usize>::new();

        for row_result in csv.records() {
            let row = row_result?;
            let parsed = match self.parse_row(&row, &schema, config) {
                Ok(Some(parsed)) => parsed,
                Ok(None) => continue,
                Err(error) if !config.strict => {
                    log::debug!("skipping foundation row: {error:#}");
                    continue;
                }
                Err(error) => return Err(error),
            };

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
        Ok(records)
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
        let raw_sequence = field(row, schema.sequence).unwrap_or("").trim();
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
        let sequence = find_header(
            headers,
            &[
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
        Ok(Self {
            sequence,
            charge: find_header(headers, &["precursorcharge", "precursor_charge", "charge"]),
            normalized_rt: find_header(
                headers,
                &[
                    "normalizedretentiontime",
                    "normalized_retention_time",
                    "irt",
                    "indexedretentiontime",
                    "normalizedrt",
                ],
            ),
            observed_rt: find_header(
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
            ),
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
    for alias in &aliases {
        if let Some(index) = headers
            .iter()
            .position(|header| normalize_header(header).contains(alias))
        {
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
}
