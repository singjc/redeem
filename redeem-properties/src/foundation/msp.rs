//! Streaming MSP spectral-library ingestion for inverse foundation training.
//!
//! MSP entries contain an identified peptide/precursor plus an observed MS/MS
//! peak list. Peak annotations are optional and highly dialect-dependent, so
//! this adapter intentionally stores measured `(m/z, intensity)` pairs as raw
//! observed-spectrum evidence rather than fabricating cleavage/channel labels.
//! Consequently MSP entries can supervise spectrum-to-peptide objectives and
//! peptide self-supervision, while the forward cleavage-channel MS2 head remains
//! unsupervised unless a different source provides explicit fragment identity.

use super::chemistry::common_unimod_definition;
use super::data::{
    FoundationTrainingRecord, ObservedSpectrumPeak, RetentionTimeLabels, TrainingContext,
};
use super::dataset::{
    finalize_load_stats, FoundationDatasetLoader, FoundationTableLoadStats,
    FoundationTableLoaderConfig,
};
use super::featurize::{FoundationModification, FoundationModificationSite, PeptidoformInput};
use anyhow::{anyhow, Context, Result};
use std::io::BufRead;

const MAX_ERROR_EXAMPLES: usize = 8;

/// Auditable result of loading one MSP spectral library.
#[derive(Debug, Clone)]
pub struct FoundationMspLoadReport {
    /// One training record per successfully parsed MSP entry.
    pub records: Vec<FoundationTrainingRecord>,
    /// Coverage/error statistics using the common corpus audit structure.
    pub stats: FoundationTableLoadStats,
}

#[derive(Debug, Clone, Default)]
struct MspEntry {
    name: Option<String>,
    comment: Option<String>,
    expected_peaks: Option<usize>,
    peaks: Vec<ObservedSpectrumPeak>,
}

impl MspEntry {
    fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.comment.is_none()
            && self.expected_peaks.is_none()
            && self.peaks.is_empty()
    }
}

/// Load an MSP library from a buffered reader.
///
/// The parser supports the common NIST-style `Name: PEPTIDE/charge`, `Parent=`,
/// `Mods=`, and `Num peaks:` fields. Both legacy slash-separated modifications
/// (`Mods=2/2,C,Carbamidomethyl/...`) and parenthesized modifications
/// (`Mods=2(2,C,CAM)(7,M,Oxidation)`) are accepted.
pub fn load_foundation_msp_reader<R: BufRead>(
    reader: R,
    loader: &mut FoundationDatasetLoader,
    config: &FoundationTableLoaderConfig,
) -> Result<FoundationMspLoadReport> {
    let mut records = Vec::<FoundationTrainingRecord>::new();
    let mut stats = FoundationTableLoadStats::default();
    let mut entry = MspEntry::default();
    let mut remaining_peak_lines = 0usize;

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read MSP line {}", line_number + 1))?;
        let trimmed = line.trim();

        if remaining_peak_lines > 0 {
            if trimmed.is_empty() {
                if config.strict {
                    anyhow::bail!(
                        "MSP entry ended before its declared peak count at line {}",
                        line_number + 1
                    );
                }
                remaining_peak_lines = 0;
                finish_entry(&mut entry, loader, config, &mut records, &mut stats)?;
                continue;
            }
            match parse_peak_line(trimmed) {
                Ok(peaks) => {
                    if peaks.is_empty() {
                        if config.strict {
                            return Err(anyhow!(
                                "MSP peak line '{}' contains no numeric m/z/intensity pair",
                                trimmed
                            )
                            .context(format!("MSP peak line {}", line_number + 1)));
                        }
                    } else {
                        if peaks.len() > remaining_peak_lines {
                            if config.strict {
                                return Err(anyhow!(
                                    "MSP peak line {} contains {} peaks but only {} remain from Num peaks",
                                    line_number + 1,
                                    peaks.len(),
                                    remaining_peak_lines
                                ));
                            }
                        }
                        let accepted = peaks.len().min(remaining_peak_lines);
                        entry.peaks.extend(peaks.into_iter().take(accepted));
                        stats.raw_observed_peak_rows += accepted;
                        remaining_peak_lines -= accepted;
                    }
                }
                Err(error) if !config.strict => {
                    if stats.error_examples.len() < MAX_ERROR_EXAMPLES {
                        stats
                            .error_examples
                            .push(format!("line {}: {error:#}", line_number + 1));
                    }
                    // Count a malformed physical line as one declared peak in
                    // non-strict mode so one bad peak does not desynchronize
                    // all following MSP entries.
                    remaining_peak_lines = remaining_peak_lines.saturating_sub(1);
                }
                Err(error) => {
                    return Err(error.context(format!("MSP peak line {}", line_number + 1)))
                }
            }
            if remaining_peak_lines == 0 {
                finish_entry(&mut entry, loader, config, &mut records, &mut stats)?;
            }
            continue;
        }

        if trimmed.is_empty() {
            if !entry.is_empty() {
                finish_entry(&mut entry, loader, config, &mut records, &mut stats)?;
            }
            continue;
        }

        if let Some(value) = strip_ascii_prefix(trimmed, "Name:") {
            if !entry.is_empty() {
                finish_entry(&mut entry, loader, config, &mut records, &mut stats)?;
            }
            stats.input_rows += 1;
            entry.name = Some(value.trim().to_string());
            continue;
        }
        if let Some(value) = strip_ascii_prefix(trimmed, "Comment:")
            .or_else(|| strip_ascii_prefix(trimmed, "Comments:"))
        {
            entry.comment = Some(value.trim().to_string());
            continue;
        }
        if let Some(value) = strip_ascii_prefix(trimmed, "Num peaks:")
            .or_else(|| strip_ascii_prefix(trimmed, "Num Peaks:"))
        {
            let count = value
                .trim()
                .parse::<usize>()
                .with_context(|| format!("invalid MSP peak count '{value}'"))?;
            entry.expected_peaks = Some(count);
            remaining_peak_lines = count;
            if count == 0 {
                finish_entry(&mut entry, loader, config, &mut records, &mut stats)?;
            }
            continue;
        }
        // Other MSP metadata lines (MW, Synon, Instrument_type, etc.) are
        // intentionally ignored unless their semantics are explicitly mapped.
    }

    if remaining_peak_lines > 0 && config.strict {
        anyhow::bail!(
            "MSP file ended with {remaining_peak_lines} declared peak lines still missing"
        );
    }
    if !entry.is_empty() {
        finish_entry(&mut entry, loader, config, &mut records, &mut stats)?;
    }

    finalize_load_stats(&mut stats, &records);
    Ok(FoundationMspLoadReport { records, stats })
}

fn finish_entry(
    entry: &mut MspEntry,
    loader: &mut FoundationDatasetLoader,
    config: &FoundationTableLoaderConfig,
    records: &mut Vec<FoundationTrainingRecord>,
    stats: &mut FoundationTableLoadStats,
) -> Result<()> {
    if entry.is_empty() {
        return Ok(());
    }
    let candidate = std::mem::take(entry);
    match build_record(candidate, loader, config) {
        Ok(record) => {
            stats.parsed_rows += 1;
            records.push(record);
        }
        Err(error) if !config.strict => {
            stats.skipped_error_rows += 1;
            if stats.error_examples.len() < MAX_ERROR_EXAMPLES {
                stats.error_examples.push(format!("{error:#}"));
            }
        }
        Err(error) => return Err(error),
    }
    Ok(())
}

fn build_record(
    entry: MspEntry,
    loader: &mut FoundationDatasetLoader,
    config: &FoundationTableLoaderConfig,
) -> Result<FoundationTrainingRecord> {
    let name = entry
        .name
        .as_deref()
        .ok_or_else(|| anyhow!("MSP entry is missing Name"))?;
    let (sequence, charge, name_mods) = parse_name(name)?;
    if sequence.len() < 2 {
        return Err(anyhow!("MSP peptide '{sequence}' is too short"));
    }
    if entry.peaks.is_empty() {
        return Err(anyhow!(
            "MSP entry '{name}' contains no valid positive peaks"
        ));
    }
    if let Some(expected) = entry.expected_peaks {
        if config.strict && entry.peaks.len() != expected {
            return Err(anyhow!(
                "MSP entry '{name}' declared {expected} peaks but parsed {} valid peaks",
                entry.peaks.len()
            ));
        }
    }

    let comment = entry.comment.unwrap_or_default();
    let precursor_mz = comment_value(&comment, "Parent")
        .or_else(|| comment_value(&comment, "PrecursorMZ"))
        .and_then(parse_positive_f32);
    let mods_text = comment_value(&comment, "Mods")
        .map(str::to_string)
        .or(name_mods);
    let modifications = mods_text
        .as_deref()
        .map(|value| parse_modifications(value, &sequence))
        .transpose()?
        .unwrap_or_default();

    let nce = comment_value(&comment, "NCE")
        .and_then(|value| value.split(',').next())
        .and_then(parse_positive_f32)
        .or(config.default_nce);
    let instrument_name = config
        .default_instrument
        .clone()
        .or_else(|| comment_value(&comment, "Instrument_type").map(str::to_string));
    let instrument_id = loader.instrument_id_for(instrument_name.as_deref());

    Ok(FoundationTrainingRecord {
        peptidoform: PeptidoformInput {
            sequence,
            modifications,
        },
        retention_time: RetentionTimeLabels::default(),
        ccs: None,
        fragments: Vec::new(),
        observed_spectrum_peaks: entry.peaks,
        context: TrainingContext {
            charge: Some(charge),
            precursor_mz,
            nce,
            instrument_id: (instrument_id != 0).then_some(instrument_id),
            instrument_name,
            ion_mobility: None,
            gradient_seconds: config.default_gradient_seconds,
        },
        run_id: config.default_run_id.clone(),
    })
}

fn parse_name(name: &str) -> Result<(String, i32, Option<String>)> {
    let name = name.trim();
    let slash = name
        .rfind('/')
        .ok_or_else(|| anyhow!("MSP Name '{name}' does not contain /charge"))?;
    let sequence = name[..slash].trim().to_ascii_uppercase();
    if sequence.is_empty()
        || !sequence
            .chars()
            .all(|aa| "ACDEFGHIKLMNPQRSTVWY".contains(aa))
    {
        return Err(anyhow!("unsupported MSP peptide sequence in Name '{name}'"));
    }
    let suffix = name[slash + 1..].trim();
    let charge_digits: String = suffix
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    let charge = charge_digits
        .parse::<i32>()
        .with_context(|| format!("invalid MSP precursor charge in Name '{name}'"))?;
    if charge <= 0 {
        return Err(anyhow!(
            "MSP precursor charge must be positive in Name '{name}'"
        ));
    }
    let name_mods = suffix
        .strip_prefix(&charge_digits)
        .and_then(|tail| tail.strip_prefix('_'))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    Ok((sequence, charge, name_mods))
}

fn parse_modifications(text: &str, sequence: &str) -> Result<Vec<FoundationModification>> {
    let text = text.trim();
    if text.is_empty() || text == "0" {
        return Ok(Vec::new());
    }
    if text.contains('/') {
        parse_legacy_modifications(text, sequence)
    } else if text.contains('(') {
        parse_parenthesized_modifications(text, sequence)
    } else {
        let count = text
            .parse::<usize>()
            .with_context(|| format!("unsupported MSP Mods value '{text}'"))?;
        if count == 0 {
            Ok(Vec::new())
        } else {
            Err(anyhow!(
                "MSP Mods='{text}' declares modifications without site annotations"
            ))
        }
    }
}

fn parse_legacy_modifications(text: &str, sequence: &str) -> Result<Vec<FoundationModification>> {
    let mut fields = text.split('/');
    let count = fields
        .next()
        .ok_or_else(|| anyhow!("empty MSP Mods field"))?
        .parse::<usize>()
        .with_context(|| format!("invalid MSP Mods count in '{text}'"))?;
    let mut modifications = Vec::with_capacity(count);
    for field in fields {
        if field.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = field.splitn(3, ',').map(str::trim).collect();
        if parts.len() != 3 {
            return Err(anyhow!("invalid MSP modification '{field}' in '{text}'"));
        }
        modifications.push(build_modification(parts[0], parts[1], parts[2], sequence)?);
    }
    if modifications.len() != count {
        return Err(anyhow!(
            "MSP Mods='{text}' declared {count} modifications but parsed {}",
            modifications.len()
        ));
    }
    Ok(modifications)
}

fn parse_parenthesized_modifications(
    text: &str,
    sequence: &str,
) -> Result<Vec<FoundationModification>> {
    let count_end = text.find('(').unwrap_or(text.len());
    let count = text[..count_end]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("invalid MSP Mods count in '{text}'"))?;
    let mut modifications = Vec::with_capacity(count);
    let bytes = text.as_bytes();
    let mut offset = count_end;
    while offset < bytes.len() {
        while offset < bytes.len() && bytes[offset].is_ascii_whitespace() {
            offset += 1;
        }
        if offset >= bytes.len() {
            break;
        }
        if bytes[offset] != b'(' {
            return Err(anyhow!(
                "unexpected MSP Mods syntax near '{}'",
                &text[offset..]
            ));
        }
        let start = offset + 1;
        let end = text[start..]
            .find(')')
            .map(|relative| start + relative)
            .ok_or_else(|| anyhow!("unterminated MSP modification in '{text}'"))?;
        let parts: Vec<&str> = text[start..end].splitn(3, ',').map(str::trim).collect();
        if parts.len() != 3 {
            return Err(anyhow!("invalid MSP modification '{}'", &text[start..end]));
        }
        modifications.push(build_modification(parts[0], parts[1], parts[2], sequence)?);
        offset = end + 1;
    }
    if modifications.len() != count {
        return Err(anyhow!(
            "MSP Mods='{text}' declared {count} modifications but parsed {}",
            modifications.len()
        ));
    }
    Ok(modifications)
}

fn build_modification(
    position_text: &str,
    amino_acid_text: &str,
    tag: &str,
    sequence: &str,
) -> Result<FoundationModification> {
    let position = position_text
        .parse::<usize>()
        .with_context(|| format!("invalid MSP modification position '{position_text}'"))?;
    let aa_text = amino_acid_text.trim();
    let aa_lower = aa_text.to_ascii_lowercase();
    let residues: Vec<char> = sequence.chars().collect();

    let (site, residue_index) =
        if matches!(aa_lower.as_str(), "n-term" | "nterm" | "n_term" | "_" | "-") {
            (FoundationModificationSite::NTerm, 0)
        } else if matches!(aa_lower.as_str(), "c-term" | "cterm" | "c_term") {
            (
                FoundationModificationSite::CTerm,
                residues.len().saturating_sub(1),
            )
        } else {
            let expected = aa_text.chars().next().map(|aa| aa.to_ascii_uppercase());
            let zero_based = residues.get(position).copied();
            let one_based = position
                .checked_sub(1)
                .and_then(|index| residues.get(index).copied());
            let residue_index = match expected {
                Some(expected) if zero_based == Some(expected) => position,
                Some(expected) if one_based == Some(expected) => position - 1,
                Some(expected) => {
                    return Err(anyhow!(
                "MSP modification site {position},{expected} does not match peptide '{sequence}'"
            ))
                }
                None if position < residues.len() => position,
                _ => {
                    return Err(anyhow!(
                        "MSP modification position {position} is outside peptide '{sequence}'"
                    ))
                }
            };
            (
                FoundationModificationSite::Residue(residue_index),
                residue_index,
            )
        };

    let tag_normalized = tag.trim().trim_matches('"').to_ascii_lowercase();
    if let Ok(mass_delta) = tag_normalized.trim_start_matches('+').parse::<f32>() {
        return Ok(FoundationModification::mass_delta_at_site(
            site,
            residue_index,
            mass_delta,
        ));
    }
    let unimod_id = match tag_normalized.as_str() {
        "oxidation" | "ox" => 35,
        "carbamidomethyl" | "carbamidomethylation" | "cam" => 4,
        "acetyl" | "acetylation" => 1,
        "deamidation" | "deamidated" | "deamid" => 7,
        "phospho" | "phosphorylation" => 21,
        "tmt6plex" | "tmt6" => 737,
        "tmtpro" => 2016,
        _ => return Err(anyhow!("unsupported MSP modification tag '{tag}'")),
    };
    let definition = common_unimod_definition(unimod_id)
        .ok_or_else(|| anyhow!("UniMod:{unimod_id} is missing from the foundation registry"))?;
    Ok(FoundationModification::unimod(
        site,
        residue_index,
        unimod_id,
        definition.mass_delta,
    ))
}

fn parse_peak_line(line: &str) -> Result<Vec<ObservedSpectrumPeak>> {
    // Peptide MSP dialects commonly append a quoted annotation after one
    // m/z-intensity pair. Generic NIST MSP also permits several pairs on one
    // physical line separated by commas/semicolons/brackets. Parse only the
    // numeric prefix before a quoted annotation, then consume numeric pairs.
    let numeric_prefix = line.split('"').next().unwrap_or(line);
    let normalized: String = numeric_prefix
        .chars()
        .map(|ch| match ch {
            ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' => ' ',
            other => other,
        })
        .collect();
    let mut numeric = Vec::<f32>::new();
    for token in normalized.split_whitespace() {
        match token.parse::<f32>() {
            Ok(value) => numeric.push(value),
            Err(_) if numeric.len() >= 2 => break,
            Err(_) => {
                return Err(anyhow!(
                    "MSP peak line '{line}' does not begin with numeric m/z and intensity"
                ));
            }
        }
    }
    if numeric.len() < 2 {
        return Err(anyhow!("MSP peak line '{line}' has no m/z-intensity pair"));
    }
    if numeric.len() % 2 != 0 {
        numeric.pop();
    }

    let mut peaks = Vec::with_capacity(numeric.len() / 2);
    for pair in numeric.chunks_exact(2) {
        let mz = pair[0];
        let intensity = pair[1];
        if mz.is_finite() && mz > 0.0 && intensity.is_finite() && intensity > 0.0 {
            peaks.push(ObservedSpectrumPeak { mz, intensity });
        }
    }
    Ok(peaks)
}

fn comment_value<'a>(comment: &'a str, key: &str) -> Option<&'a str> {
    comment.split_whitespace().find_map(|token| {
        let (candidate, value) = token.split_once('=')?;
        candidate.eq_ignore_ascii_case(key).then_some(value)
    })
}

fn parse_positive_f32(value: &str) -> Option<f32> {
    value
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn strip_ascii_prefix<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    (value.len() >= prefix.len() && value[..prefix.len()].eq_ignore_ascii_case(prefix))
        .then(|| &value[prefix.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_unannotated_msp_peaks_without_forward_fragments() {
        let input = concat!(
            "Name: LGLHSLR/3\n",
            "MW: 794.4750234375\n",
            "Comment: Spec=Consensus Mods=0 Parent=265.833 Nreps=17\n",
            "Num peaks: 3\n",
            "69.071 109.29\n",
            "86.097 12806.1\n",
            "175.119 1981.28\n\n",
        );
        let mut loader = FoundationDatasetLoader::new(16);
        let report = load_foundation_msp_reader(
            std::io::BufReader::new(input.as_bytes()),
            &mut loader,
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
        assert_eq!(report.records.len(), 1);
        let record = &report.records[0];
        assert_eq!(record.peptidoform.sequence, "LGLHSLR");
        assert_eq!(record.context.charge, Some(3));
        assert_eq!(record.context.precursor_mz, Some(265.833));
        assert!(record.fragments.is_empty());
        assert_eq!(record.observed_spectrum_peaks.len(), 3);
        assert_eq!(report.stats.ms2_records, 0);
        assert_eq!(report.stats.observed_spectrum_records, 1);
        assert_eq!(report.stats.raw_observed_peak_rows, 3);
    }

    #[test]
    fn parses_multiple_msp_peak_pairs_on_one_physical_line() {
        let input = concat!(
            "Name: PEPTIDEK/2\n",
            "Comment: Mods=0 Parent=464.2\n",
            "Num peaks: 4\n",
            "100.0 10.0; 200.0 20.0\n",
            "300.0 30.0, 400.0 40.0\n\n",
        );
        let mut loader = FoundationDatasetLoader::new(16);
        let report = load_foundation_msp_reader(
            std::io::BufReader::new(input.as_bytes()),
            &mut loader,
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
        assert_eq!(report.records.len(), 1);
        assert_eq!(report.records[0].observed_spectrum_peaks.len(), 4);
        assert_eq!(report.stats.raw_observed_peak_rows, 4);
    }

    #[test]
    fn parses_legacy_and_parenthesized_msp_modifications() {
        let legacy = parse_modifications("2/1,C,Carbamidomethyl/4,M,Oxidation", "ACDMK").unwrap();
        assert_eq!(legacy.len(), 2);
        assert_eq!(legacy[0].unimod_id, Some(4));
        assert_eq!(legacy[0].residue_index, 1);
        assert_eq!(legacy[1].unimod_id, Some(35));
        assert_eq!(legacy[1].residue_index, 3);

        let modern = parse_modifications("2(2,C,CAM)(5,M,Oxidation)", "ACDEM").unwrap();
        assert_eq!(modern.len(), 2);
        assert_eq!(modern[0].residue_index, 1); // one-based fallback
        assert_eq!(modern[1].residue_index, 4);
    }
}
