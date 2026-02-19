//! Percolator .pin TSV reader.
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use csv::StringRecord;

use crate::data_handling::{Experiment, PsmMetadata};
use crate::math::{Array1, Array2};

/// Parsed PIN data ready for model training or evaluation.
#[derive(Debug)]
pub struct PinData {
    pub x: Array2<f32>,
    pub y: Array1<i32>,
    pub metadata: PsmMetadata,
}

/// Configuration for reading Percolator .pin TSV files.
#[derive(Debug, Clone)]
pub struct PinReaderConfig {
    /// Column name holding target/decoy labels (1 / -1).
    pub label_column: String,
    /// Column name for spectrum identifiers.
    pub spec_id_column: String,
    /// Optional column name for scan number.
    pub scan_number_column: Option<String>,
    /// Optional column name for file id/name.
    pub file_id_column: Option<String>,
    /// Optional list of feature columns to load (in order).
    /// When `None`, all non-metadata columns are treated as features.
    pub feature_columns: Option<Vec<String>>,
    /// Columns to ignore when auto-selecting features.
    pub ignore_columns: Vec<String>,
}

impl Default for PinReaderConfig {
    fn default() -> Self {
        Self {
            label_column: "Label".to_string(),
            spec_id_column: "SpecId".to_string(),
            scan_number_column: None,
            file_id_column: None,
            feature_columns: None,
            ignore_columns: vec![
                "SpecId".to_string(),
                "Label".to_string(),
                "ScanNr".to_string(),
                "Scan".to_string(),
                "ExpMass".to_string(),
                "Peptide".to_string(),
                "Proteins".to_string(),
                "ProteinId".to_string(),
                "SpecFile".to_string(),
                "File".to_string(),
                "FileName".to_string(),
                "FileId".to_string(),
                "FileIdx".to_string(),
            ],
        }
    }
}

/// Read a Percolator .pin TSV file into arrays and metadata.
pub fn read_pin_tsv<P: AsRef<Path>>(path: P) -> Result<PinData> {
    read_pin_tsv_with_config(path, &PinReaderConfig::default())
}

/// Read a Percolator .pin TSV file using a custom configuration.
pub fn read_pin_tsv_with_config<P: AsRef<Path>>(path: P, config: &PinReaderConfig) -> Result<PinData> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .flexible(true)
        .from_path(&path)
        .with_context(|| format!("Failed to open PIN file: {}", path.as_ref().display()))?;

    let headers = reader
        .headers()
        .context("Failed to read PIN header row")?
        .clone();

    let label_idx = find_column(&headers, &config.label_column)
        .ok_or_else(|| anyhow!("Missing label column '{}'", config.label_column))?;

    let spec_id_idx = find_column(&headers, &config.spec_id_column);

    let scan_idx = match &config.scan_number_column {
        Some(name) => find_column(&headers, name),
        None => find_any_column(&headers, &["ScanNr", "ScanNum", "Scan"]),
    };

    let file_idx = match &config.file_id_column {
        Some(name) => find_column(&headers, name),
        None => find_any_column(
            &headers,
            &["FileId", "FileIdx", "FileName", "File", "SpecFile", "RawFile"],
        ),
    };
    let exp_mass_idx = find_any_column(&headers, &["ExpMass"]);

    let feature_indices =
        resolve_feature_indices(&headers, config, label_idx, spec_id_idx, scan_idx, file_idx)?;
    if feature_indices.is_empty() {
        return Err(anyhow!("No feature columns detected in PIN header"));
    }

    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut spec_ids = Vec::new();
    let mut file_ids = Vec::new();
    let mut scan_numbers = Vec::new();
    let mut exp_masses = Vec::new();

    let mut file_id_map: HashMap<String, usize> = HashMap::new();

    for (row_idx, result) in reader.records().enumerate() {
        let record = result.with_context(|| format!("Failed to read row {}", row_idx + 1))?;

        if record
            .get(0)
            .map(|value| value.eq_ignore_ascii_case("DefaultDirection"))
            .unwrap_or(false)
        {
            continue;
        }

        let label_value = record.get(label_idx).unwrap_or("").trim();
        let label = match label_value.parse::<i32>() {
            Ok(value) => value,
            Err(_) => {
                continue;
            }
        };
        labels.push(label);

        let (spec_id, file_key) = extract_spec_and_file(
            record.get(spec_id_idx.unwrap_or(usize::MAX)),
            record.get(scan_idx.unwrap_or(usize::MAX)),
            row_idx,
        );
        spec_ids.push(spec_id);

        let file_id = if let Some(idx) = file_idx {
            let value = record.get(idx).unwrap_or_default().trim();
            map_file_id(value, &mut file_id_map)
        } else if let Some(key) = file_key {
            map_file_id(&key, &mut file_id_map)
        } else {
            0
        };
        file_ids.push(file_id);

        let scan_value = scan_idx
            .and_then(|idx| record.get(idx))
            .and_then(|value| value.trim().parse::<i32>().ok());
        scan_numbers.push(scan_value);

        let exp_mass_value = exp_mass_idx
            .and_then(|idx| record.get(idx))
            .and_then(|value| value.trim().parse::<f32>().ok());
        exp_masses.push(exp_mass_value);

        for &idx in &feature_indices {
            let value = record.get(idx).unwrap_or("").trim();
            let parsed = if value.is_empty() {
                0.0
            } else {
                value.parse::<f32>().with_context(|| {
                    format!(
                        "Invalid feature '{}' at row {}",
                        headers.get(idx).unwrap_or(""),
                        row_idx + 1
                    )
                })?
            };
            features.push(parsed);
        }
    }

    let n_samples = labels.len();
    let n_features = feature_indices.len();
    let x = Array2::from_shape_vec((n_samples, n_features), features)
        .context("Failed to build feature matrix")?;
    let y = Array1::from_vec(labels);

    let feature_names = feature_indices
        .iter()
        .map(|&idx| headers.get(idx).unwrap_or("").to_string())
        .collect();

    let metadata = PsmMetadata {
        spec_id: spec_ids,
        file_id: file_ids,
        feature_names,
        scan_nr: scan_idx.map(|_| scan_numbers),
        exp_mass: exp_mass_idx.map(|_| exp_masses),
    };

    Ok(PinData { x, y, metadata })
}

/// Convenience helper to directly build an Experiment from a PIN file.
pub fn read_pin_experiment<P: AsRef<Path>>(path: P) -> Result<Experiment> {
    let data = read_pin_tsv(path)?;
    Ok(Experiment::new(data.x, data.y, data.metadata))
}

fn find_column(headers: &StringRecord, name: &str) -> Option<usize> {
    headers
        .iter()
        .position(|header| header.eq_ignore_ascii_case(name))
}

fn find_any_column(headers: &StringRecord, names: &[&str]) -> Option<usize> {
    names.iter().find_map(|name| find_column(headers, name))
}

fn resolve_feature_indices(
    headers: &StringRecord,
    config: &PinReaderConfig,
    label_idx: usize,
    spec_id_idx: Option<usize>,
    scan_idx: Option<usize>,
    file_idx: Option<usize>,
) -> Result<Vec<usize>> {
    if let Some(names) = &config.feature_columns {
        let mut indices = Vec::with_capacity(names.len());
        for name in names {
            let idx = find_column(headers, name)
                .ok_or_else(|| anyhow!("Missing feature column '{}'", name))?;
            indices.push(idx);
        }
        return Ok(indices);
    }

    let mut ignore = HashSet::new();
    for name in &config.ignore_columns {
        ignore.insert(name.to_ascii_lowercase());
    }

    let mut indices = Vec::new();
    for (idx, header) in headers.iter().enumerate() {
        if idx == label_idx {
            continue;
        }
        if Some(idx) == spec_id_idx || Some(idx) == scan_idx || Some(idx) == file_idx {
            continue;
        }
        if ignore.contains(&header.to_ascii_lowercase()) {
            continue;
        }
        indices.push(idx);
    }

    Ok(indices)
}

fn extract_spec_and_file(
    spec_id: Option<&str>,
    scan_nr: Option<&str>,
    row_idx: usize,
) -> (String, Option<String>) {
    if let Some(spec) = spec_id {
        let trimmed = spec.trim();
        if let Some((file, rest)) = split_spec_id(trimmed) {
            return (rest.to_string(), Some(file.to_string()));
        }
        return (trimmed.to_string(), None);
    }
    if let Some(scan) = scan_nr {
        return (scan.trim().to_string(), None);
    }
    (format!("row_{}", row_idx + 1), None)
}

fn split_spec_id(spec_id: &str) -> Option<(&str, &str)> {
    for delimiter in [':', '|'] {
        if let Some(idx) = spec_id.find(delimiter) {
            let (left, right) = spec_id.split_at(idx);
            let right = right.trim_start_matches(delimiter);
            if !left.is_empty() && !right.is_empty() {
                return Some((left, right));
            }
        }
    }
    None
}

fn map_file_id(value: &str, map: &mut HashMap<String, usize>) -> usize {
    let key = value.trim();
    if key.is_empty() {
        return 0;
    }
    if let Some(&id) = map.get(key) {
        return id;
    }
    let next_id = map.len();
    map.insert(key.to_string(), next_id);
    next_id
}
