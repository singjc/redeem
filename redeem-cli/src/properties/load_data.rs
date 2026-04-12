use anyhow::{Context, Result};
use csv::ReaderBuilder;
use redeem_properties::utils::peptdeep_utils::{
    ModificationMap, get_modification_indices, get_modification_string,
};
use redeem_properties::utils::{
    data_handling::{PeptideData, TargetNormalization},
    peptdeep_utils::remove_mass_shift,
};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::{collections::HashMap, sync::Arc};

fn find_header_idx(headers: &csv::StringRecord, aliases: &[&str]) -> Option<usize> {
    let lower_aliases: Vec<String> = aliases.iter().map(|s| s.to_lowercase()).collect();
    headers
        .iter()
        .position(|h| lower_aliases.contains(&h.to_lowercase()))
}

fn normalize_header_name(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && *c != '_' && *c != '-')
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn find_header_idx_flexible(headers: &csv::StringRecord, aliases: &[&str]) -> Option<usize> {
    let normalized_aliases: Vec<String> =
        aliases.iter().map(|s| normalize_header_name(s)).collect();

    for alias in &normalized_aliases {
        if let Some(idx) = headers
            .iter()
            .position(|h| normalize_header_name(h) == *alias)
        {
            return Some(idx);
        }
    }

    for alias in &normalized_aliases {
        if let Some(idx) = headers
            .iter()
            .position(|h| normalize_header_name(h).contains(alias))
        {
            return Some(idx);
        }
    }

    None
}

fn record_field<'a>(
    record: &'a csv::StringRecord,
    headers: &csv::StringRecord,
    aliases: &[&str],
) -> Option<&'a str> {
    find_header_idx_flexible(headers, aliases).and_then(|idx| record.get(idx))
}

fn arc_bytes(s: &str) -> Arc<[u8]> {
    Arc::from(s.as_bytes().to_vec().into_boxed_slice())
}

fn normalize_field_for_model(model_arch: &str) -> &'static str {
    if model_arch == "ms2_bert" {
        "ms2_intensities"
    } else if model_arch.contains("ccs") {
        "ccs"
    } else {
        "retention time"
    }
}

fn parse_ms2_matrix(raw: &str) -> Option<Vec<Vec<f32>>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with('[') {
        if let Ok(matrix) = serde_json::from_str::<Vec<Vec<f32>>>(trimmed) {
            return Some(matrix);
        }
    }

    let row_sep = if trimmed.contains('|') {
        '|'
    } else if trimmed.contains(';') {
        ';'
    } else {
        '|'
    };

    let matrix: Vec<Vec<f32>> = trimmed
        .split(row_sep)
        .filter_map(|row| {
            let values: Vec<f32> = row
                .split(|c| c == ',' || c == ' ' || c == '\t')
                .filter(|s| !s.trim().is_empty())
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();
            if values.is_empty() {
                None
            } else {
                Some(values)
            }
        })
        .collect();

    if matrix.is_empty() {
        None
    } else {
        Some(matrix)
    }
}

fn collect_target_values(peptides: &[PeptideData], normalize_field: &str) -> Vec<f32> {
    match normalize_field {
        "ccs" => peptides.iter().filter_map(|p| p.ccs).collect(),
        "ms2_intensities" => peptides
            .iter()
            .filter_map(|p| p.ms2_intensities.as_ref())
            .flat_map(|m| m.iter().flat_map(|row| row.iter().copied()))
            .filter(|v| v.is_finite())
            .collect(),
        _ => peptides.iter().filter_map(|p| p.retention_time).collect(),
    }
}

fn normalize_value(value: f32, norm: TargetNormalization) -> f32 {
    match norm {
        TargetNormalization::ZScore(mean, std) if std != 0.0 => (value - mean) / std,
        TargetNormalization::MinMax(min, max) if max != min => (value - min) / (max - min),
        TargetNormalization::ZScore(_, _) | TargetNormalization::MinMax(_, _) => 0.0,
        TargetNormalization::None => value,
    }
}

fn apply_normalization(
    peptides: &mut [PeptideData],
    normalize_field: &str,
    norm: TargetNormalization,
) {
    match normalize_field {
        "ccs" => {
            for peptide in peptides {
                if let Some(value) = peptide.ccs.as_mut() {
                    *value = normalize_value(*value, norm);
                }
            }
        }
        "ms2_intensities" => {
            for peptide in peptides {
                if let Some(intensities) = peptide.ms2_intensities.as_mut() {
                    for row in intensities {
                        for value in row {
                            *value = normalize_value(*value, norm);
                        }
                    }
                }
            }
        }
        _ => {
            for peptide in peptides {
                if let Some(value) = peptide.retention_time.as_mut() {
                    *value = normalize_value(*value, norm);
                }
            }
        }
    }
}

fn finalize_peptide_data(
    mut peptides: Vec<PeptideData>,
    model_arch: &str,
    normalize_target: Option<String>,
    norm_override: Option<TargetNormalization>,
    should_apply_normalization: bool,
) -> Result<(Vec<PeptideData>, TargetNormalization)> {
    let normalize_field = normalize_field_for_model(model_arch);

    if let Some(norm) = norm_override {
        if should_apply_normalization {
            apply_normalization(&mut peptides, normalize_field, norm);
        }
        return Ok((peptides, norm));
    }

    let target_values = collect_target_values(&peptides, normalize_field);
    let norm = match TargetNormalization::from_str(normalize_target) {
        TargetNormalization::ZScore(_, _) if !target_values.is_empty() => {
            let mean = target_values.iter().copied().sum::<f32>() / target_values.len() as f32;
            let std = (target_values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f32>()
                / target_values.len() as f32)
                .sqrt();
            TargetNormalization::ZScore(mean, if std == 0.0 { 1.0 } else { std })
        }
        TargetNormalization::MinMax(_, _) if !target_values.is_empty() => {
            let min = target_values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = target_values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            TargetNormalization::MinMax(min, max)
        }
        _ => TargetNormalization::None,
    };

    if should_apply_normalization {
        apply_normalization(&mut peptides, normalize_field, norm);
    }

    Ok((peptides, norm))
}

fn has_ms2_transition_columns(headers: &csv::StringRecord) -> bool {
    find_header_idx_flexible(
        headers,
        &[
            "modifiedpeptide",
            "fullpeptidename",
            "modified_sequence",
            "sequence",
            "peptide_sequence",
            "peptide",
        ],
    )
    .is_some()
        && find_header_idx_flexible(headers, &["precursorcharge", "precursor_charge", "charge"])
            .is_some()
        && find_header_idx_flexible(headers, &["fragmenttype", "fragment_type"]).is_some()
        && find_header_idx_flexible(
            headers,
            &[
                "fragmentseriesnumber",
                "fragment_series_number",
                "series_number",
            ],
        )
        .is_some()
        && find_header_idx_flexible(headers, &["libraryintensity", "intensity"]).is_some()
}

fn read_ms2_transition_records<R: std::io::Read>(
    rdr: &mut csv::Reader<R>,
    headers: &csv::StringRecord,
    nce: Option<i32>,
    instrument: Option<String>,
    modifications: &HashMap<(String, Option<char>), ModificationMap>,
) -> Result<Vec<PeptideData>> {
    let mut peptide_map: HashMap<(String, i32), PeptideData> = HashMap::new();

    for result in rdr.records() {
        let record = result?;
        let sequence = record_field(
            &record,
            headers,
            &[
                "modifiedpeptide",
                "fullpeptidename",
                "modified_sequence",
                "sequence",
                "peptide_sequence",
                "peptide",
            ],
        )
        .unwrap_or("")
        .to_string();
        if sequence.is_empty() {
            continue;
        }

        let charge = record_field(
            &record,
            headers,
            &[
                "precursorcharge",
                "precursor_charge",
                "precursor charge",
                "charge",
            ],
        )
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
        if charge == 0 {
            continue;
        }

        let fragment_type = record_field(&record, headers, &["fragmenttype", "fragment_type"])
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let series_number = record_field(
            &record,
            headers,
            &[
                "fragmentseriesnumber",
                "fragment_series_number",
                "series_number",
            ],
        )
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
        if series_number <= 0 {
            continue;
        }

        let product_charge = record_field(
            &record,
            headers,
            &["productcharge", "product_charge", "fragment_charge"],
        )
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);
        let intensity = record_field(&record, headers, &["libraryintensity", "intensity"])
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);

        let precursor_mass = record_field(
            &record,
            headers,
            &[
                "precursor_mass",
                "precursor mass",
                "precursormz",
                "precursor_mz",
                "precursor mz",
            ],
        )
        .and_then(|s| s.parse::<f32>().ok());
        let retention_time = record_field(
            &record,
            headers,
            &[
                "normalizedretentiontime",
                "retention_time",
                "retention time",
                "irt",
                "rt",
            ],
        )
        .and_then(|s| s.parse::<f32>().ok());
        let ion_mobility = record_field(
            &record,
            headers,
            &["precursorionmobility", "ion_mobility", "ion mobility", "im"],
        )
        .and_then(|s| s.parse::<f32>().ok());
        let ccs = record_field(&record, headers, &["ccs"]).and_then(|s| s.parse::<f32>().ok());
        let in_nce = nce.or_else(|| {
            record_field(&record, headers, &["collisionenergy", "nce"])
                .and_then(|s| s.parse::<i32>().ok())
        });
        let in_instrument = instrument
            .as_ref()
            .map(|s| arc_bytes(s))
            .or_else(|| record_field(&record, headers, &["instrument"]).map(arc_bytes));

        let entry = peptide_map
            .entry((sequence.clone(), charge))
            .or_insert_with(|| {
                let naked = remove_mass_shift(&sequence);
                let peptide_len = naked.len();
                PeptideData {
                    modified_sequence: arc_bytes(&sequence),
                    naked_sequence: arc_bytes(&naked),
                    mods: arc_bytes(&get_modification_string(&sequence, modifications)),
                    mod_sites: arc_bytes(&get_modification_indices(&sequence)),
                    charge: Some(charge),
                    precursor_mass,
                    nce: in_nce,
                    instrument: in_instrument,
                    retention_time,
                    ion_mobility,
                    ccs,
                    ms2_intensities: Some(vec![vec![0.0; 8]; peptide_len.saturating_sub(1)]),
                }
            });

        if let Some(intensities) = entry.ms2_intensities.as_mut() {
            let col = match (fragment_type.as_str(), product_charge) {
                ("b", 1) => 0,
                ("b", 2) => 1,
                ("y", 1) => 2,
                ("y", 2) => 3,
                _ => continue,
            };
            let row = (series_number - 1) as usize;
            if row < intensities.len() && col < intensities[row].len() {
                intensities[row][col] = intensity;
            }
        }
    }

    Ok(peptide_map.into_values().collect())
}

/// Load peptide training data from a CSV or TSV file and optionally normalize RT.
///
/// Returns both the peptide vector and optionally (mean, std) of retention times.
pub fn load_peptide_data<P: AsRef<Path>>(
    path: P,
    model_arch: &str,
    nce: Option<i32>,
    instrument: Option<String>,
    normalize_target: Option<String>,
    norm_override: Option<TargetNormalization>,
    apply_normalization: bool,
    modifications: &HashMap<(String, Option<char>), ModificationMap>,
) -> Result<(Vec<PeptideData>, TargetNormalization)> {
    let file =
        File::open(&path).with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);

    let is_tsv = path
        .as_ref()
        .extension()
        .map(|e| e == "tsv")
        .unwrap_or(false);
    let delimiter = if is_tsv { b'\t' } else { b',' };

    let mut rdr = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .from_reader(reader);

    let headers = rdr.headers()?.clone();
    let mut peptides = Vec::new();

    if model_arch == "ms2_bert" && has_ms2_transition_columns(&headers) {
        let peptides =
            read_ms2_transition_records(&mut rdr, &headers, nce, instrument, modifications)?;
        return finalize_peptide_data(
            peptides,
            model_arch,
            normalize_target,
            norm_override,
            apply_normalization,
        );
    }

    for result in rdr.records() {
        let record = result?;

        let sequence_bytes: Arc<[u8]> = Arc::from(
            record
                .get(
                    find_header_idx(
                        &headers,
                        &[
                            "sequence",
                            "naked_sequence",
                            "modified_sequence",
                            "peptide",
                            "peptide_sequence",
                        ],
                    )
                    .unwrap_or(2),
                )
                .unwrap_or("")
                .as_bytes()
                .to_vec()
                .into_boxed_slice(),
        );

        let sequence_str = String::from_utf8_lossy(&sequence_bytes);

        let naked_sequence = Arc::from(
            remove_mass_shift(&sequence_str)
                .as_bytes()
                .to_vec()
                .into_boxed_slice(),
        );
        let mods: Arc<[u8]> = Arc::from(
            get_modification_string(&sequence_str, modifications)
                .into_bytes()
                .into_boxed_slice(),
        );
        let mod_sites: Arc<[u8]> = Arc::from(
            get_modification_indices(&sequence_str)
                .into_bytes()
                .into_boxed_slice(),
        );

        let retention_time = record
            .get(
                find_header_idx_flexible(&headers, &["retention time", "retention_time", "rt"])
                    .unwrap_or(3),
            )
            .and_then(|s| s.parse::<f32>().ok());

        let charge = match model_arch {
            "rt_cnn_lstm" | "rt_cnn_tf" => None,
            _ => record_field(
                &record,
                &headers,
                &[
                    "precursorcharge",
                    "precursor_charge",
                    "precursor charge",
                    "charge",
                ],
            )
            .and_then(|s| s.parse::<i32>().ok()),
        };

        let precursor_mass = record_field(
            &record,
            &headers,
            &[
                "precursor_mass",
                "precursor mass",
                "precursormz",
                "precursor_mz",
                "precursor mz",
            ],
        )
        .and_then(|s| s.parse::<f32>().ok());

        let ion_mobility = record_field(&record, &headers, &["ion_mobility", "ion mobility", "im"])
            .and_then(|s| s.parse::<f32>().ok());

        let ccs = record_field(&record, &headers, &["ccs"]).and_then(|s| s.parse::<f32>().ok());

        let ms2_intensities = if model_arch == "ms2_bert" {
            record_field(
                &record,
                &headers,
                &[
                    "ms2_intensities",
                    "ms2 intensities",
                    "fragment_intensities",
                    "fragment intensities",
                    "intensities",
                ],
            )
            .and_then(parse_ms2_matrix)
        } else {
            None
        };

        let in_nce = match model_arch {
            "ms2_bert" => nce.or_else(|| {
                record_field(&record, &headers, &["collisionenergy", "nce"])
                    .and_then(|s| s.parse::<i32>().ok())
            }),
            _ => None,
        };

        let in_instrument = match model_arch {
            "ms2_bert" => instrument
                .as_ref()
                .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
                .or_else(|| {
                    record_field(&record, &headers, &["instrument"])
                        .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
                }),
            _ => None,
        };

        peptides.push(PeptideData {
            modified_sequence: sequence_bytes,
            naked_sequence,
            mods,
            mod_sites,
            charge,
            precursor_mass,
            nce: in_nce,
            instrument: in_instrument,
            retention_time,
            ion_mobility,
            ccs,
            ms2_intensities,
        });
    }

    finalize_peptide_data(
        peptides,
        model_arch,
        normalize_target,
        norm_override,
        apply_normalization,
    )
}
