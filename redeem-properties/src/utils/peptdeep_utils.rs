use anyhow::{Error, Result};
use csv::ReaderBuilder;
use log::info;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io;
use std::ops::Index;
use std::path::PathBuf;
use std::sync::Arc;
use zip::ZipArchive;

const MODIFICATIONS_TSV_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/modification.tsv"
));

const PRETRAINED_MODELS_URL: &str =
    "https://github.com/singjc/redeem/releases/download/v0.1.0-alpha/pretrained_models.zip";
const PRETRAINED_MODELS_ZIP: &str = "data/pretrained_models.zip";
const PRETRAINED_MODELS_PATH: &str = "data/pretrained_models";

// Constants and Utility Structs

const INSTRUMENT_DICT: &[(&str, usize)] = &[
    ("QE", 0),
    ("LUMOS", 1),
    ("TIMSTOF", 2),
    ("SCIEXTOF", 3),
    ("THERMOTOF", 4),
];

const MAX_INSTRUMENT_NUM: usize = 8;

const UNKNOWN_INSTRUMENT_NUM: usize = MAX_INSTRUMENT_NUM - 1;

#[derive(Debug, Clone)]
pub struct ModificationMap {
    pub name: String,
    pub amino_acid: Option<char>, // Optional if not applicable
    pub unimod_id: Option<u32>,
}

/// Loads a unified modification map where the key is either:
/// - ("57.0215", Some('C')) for mass-based lookup
/// - ("UniMod:4", Some('C')) for UniMod ID–based lookup
/// Loads the modification map, parsing the embedded modifications.tsv.
pub fn load_modifications() -> Result<HashMap<(String, Option<char>), ModificationMap>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(MODIFICATIONS_TSV_BYTES);

    let mut modifications = HashMap::new();

    for result in rdr.records() {
        let record = result?;
        let mod_name = record.get(0).unwrap_or("").to_string();
        let unimod_mass: f64 = record.get(1).unwrap_or("0").parse().unwrap_or(0.0);
        let unimod_id: Option<u32> = record.get(7).and_then(|s| s.parse().ok());

        let mass_key = format!("{:.4}", unimod_mass);
        let unimod_key = unimod_id.map(|id| format!("UniMod:{}", id));

        let amino_acid = mod_name.split('@').nth(1).and_then(|aa| aa.chars().next());

        let modification = ModificationMap {
            name: mod_name,
            amino_acid,
            unimod_id,
        };

        // Insert mass-based key
        modifications.insert((mass_key.clone(), amino_acid), modification.clone());

        // Insert unimod-id based key if available
        if let Some(key) = unimod_key {
            modifications.insert((key, amino_acid), modification.clone());
        }
    }

    Ok(modifications)
}

// Lazy static variable to hold the loaded modification map
pub static MODIFICATION_MAP: Lazy<HashMap<(String, Option<char>), ModificationMap>> =
    Lazy::new(|| load_modifications().expect("Failed to load modifications"));

#[derive(Clone, Debug, Deserialize)]
/// Represents the constants used in a model.
pub struct ModelConstants {
    /// The size of the amino acid embedding.
    pub aa_embedding_size: Option<usize>,
    /// The charge factor used in the model.
    pub charge_factor: Option<f32>,
    /// The list of instruments used in the model.
    pub instruments: Vec<String>,
    /// The maximum number of instruments allowed in the model.
    pub max_instrument_num: usize,
    /// The list of modification elements used in the model.
    pub mod_elements: Vec<String>,
    /// The NCE (Normalized Collision Energy) factor used in the model.
    pub nce_factor: Option<f32>,
}

#[derive(Debug, serde::Deserialize)]
struct ModFeature {
    mod_name: String,
    composition: String,
    // Add other fields if needed
}

impl Default for ModelConstants {
    fn default() -> Self {
        Self {
            aa_embedding_size: Some(27),
            charge_factor: Some(0.1),
            instruments: vec![
                "QE".into(),
                "Lumos".into(),
                "timsTOF".into(),
                "SciexTOF".into(),
                "ThermoTOF".into(),
            ],
            max_instrument_num: 8,
            mod_elements: vec![
                "C", "H", "N", "O", "P", "S", "B", "F", "I", "K", "U", "V", "W", "X", "Y", "Ac",
                "Ag", "Al", "Am", "Ar", "As", "At", "Au", "Ba", "Be", "Bi", "Bk", "Br", "Ca", "Cd",
                "Ce", "Cf", "Cl", "Cm", "Co", "Cr", "Cs", "Cu", "Dy", "Er", "Es", "Eu", "Fe", "Fm",
                "Fr", "Ga", "Gd", "Ge", "He", "Hf", "Hg", "Ho", "In", "Ir", "Kr", "La", "Li", "Lr",
                "Lu", "Md", "Mg", "Mn", "Mo", "Na", "Nb", "Nd", "Ne", "Ni", "No", "Np", "Os", "Pa",
                "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rh", "Rn", "Ru", "Sb",
                "Sc", "Se", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm",
                "Xe", "Yb", "Zn", "Zr", "2H", "13C", "15N", "18O", "?",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            nce_factor: Some(0.01),
        }
    }
}

/// Parse the model constants from a YAML file.
pub fn parse_model_constants(path: &str) -> Result<ModelConstants> {
    let f = std::fs::File::open(path).map_err(|e| Error::msg(e.to_string()))?;
    // Holds the model constants.
    let constants: ModelConstants =
        serde_yaml::from_reader(f).map_err(|e| Error::msg(e.to_string()))?;
    Ok(constants)
}

fn parse_mod_formula(
    formula: &str,
    mod_elem_to_idx: &HashMap<String, usize>,
    mod_feature_size: usize,
) -> Vec<f32> {
    let mut feature = vec![0.0; mod_feature_size];
    let elems: Vec<&str> = formula.trim_end_matches(')').split(')').collect();
    for elem in elems {
        let parts: Vec<&str> = elem.split('(').collect();
        if parts.len() == 2 {
            let chem = parts[0];
            let num: i32 = parts[1].parse().unwrap_or(0);
            if let Some(&idx) = mod_elem_to_idx.get(chem) {
                feature[idx] = num as f32;
            } else {
                feature[mod_feature_size - 1] += num as f32;
            }
        }
    }
    feature
}

pub fn load_mod_to_feature(constants: &ModelConstants) -> Result<HashMap<String, Vec<f32>>, Error> {
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(MODIFICATIONS_TSV_BYTES); // Read from the byte slice

    // Create mod_elem_to_idx mapping
    let mod_elem_to_idx: HashMap<String, usize> = constants
        .mod_elements
        .iter()
        .enumerate()
        .map(|(i, elem)| (elem.clone(), i))
        .collect();

    let mod_feature_size = constants.mod_elements.len();

    let mut mod_to_feature = HashMap::new();

    for result in rdr.deserialize() {
        let record: ModFeature = result?;
        let feature_vector =
            parse_mod_formula(&record.composition, &mod_elem_to_idx, mod_feature_size);
        mod_to_feature.insert(record.mod_name, feature_vector);
    }

    Ok(mod_to_feature)
}

pub fn load_mod_to_feature_arc(
    constants: &ModelConstants,
) -> Result<HashMap<Arc<[u8]>, Vec<f32>>, Error> {
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(MODIFICATIONS_TSV_BYTES);

    let mod_elem_to_idx: HashMap<String, usize> = constants
        .mod_elements
        .iter()
        .enumerate()
        .map(|(i, elem)| (elem.clone(), i))
        .collect();

    let mod_feature_size = constants.mod_elements.len();
    let mut mod_to_feature = HashMap::new();

    for result in rdr.deserialize() {
        let record: ModFeature = result?;
        let feature_vector =
            parse_mod_formula(&record.composition, &mod_elem_to_idx, mod_feature_size);
        mod_to_feature.insert(Arc::from(record.mod_name.as_bytes()), feature_vector);
    }

    Ok(mod_to_feature)
}

/// Removes mass shifts and UniMod annotations from a modified peptide sequence.
///
/// Supports both bracketed mass shifts (e.g., `[+57.0215]`) and UniMod-style
/// annotations (e.g., `(UniMod:4)`).
///
/// # Example
/// ```
/// use redeem_properties::utils::peptdeep_utils::remove_mass_shift;
///
/// let peptide = "MGC[+57.0215]AAR";
/// assert_eq!(remove_mass_shift(peptide), "MGCAAR");
/// let peptide = "MGC(UniMod:4)AAR";
/// assert_eq!(remove_mass_shift(peptide), "MGCAAR");
/// ```
pub fn remove_mass_shift(peptide: &str) -> String {
    // Regex to remove either [mass shift] or (UniMod:x) patterns
    let re = Regex::new(r"(\[.*?\]|\(UniMod:\d+\))").unwrap();
    re.replace_all(peptide, "").to_string()
}

pub fn extract_masses(peptide: &str) -> Vec<f64> {
    let mut masses = Vec::new();
    let mut start = 0;

    while let Some(open_bracket) = peptide[start..].find('[') {
        let open_index = start + open_bracket;
        if let Some(close_index) = peptide[open_index..].find(']') {
            let mass_str = &peptide[open_index + 1..open_index + close_index];
            if let Ok(mass) = mass_str.trim_start_matches('+').parse::<f64>() {
                masses.push(mass);
            }
            start = open_index + close_index + 1; // Move past the current bracket
        } else {
            break; // No closing bracket found
        }
    }

    masses
}

pub fn extract_masses_and_indices(peptide: &str) -> Vec<(f64, usize)> {
    let re = Regex::new(r"\[([+-]?\d*\.?\d+)\]").unwrap();
    let mut results = Vec::new();
    let mut offset = 0;

    for cap in re.captures_iter(peptide) {
        if let (Some(whole_match), Some(mass_str)) = (cap.get(0), cap.get(1)) {
            if let Ok(mass) = mass_str.as_str().parse::<f64>() {
                let index = whole_match.start() - offset;
                results.push((mass, index));
                offset += whole_match.len();
            }
        }
    }

    results
}

/// Extracts modification indices from a peptide string.
/// The indices are 0-based and represent the positions of the modifications.
///
/// # Example
/// ```
/// use redeem_properties::utils::peptdeep_utils::get_modification_indices;
/// let result = get_modification_indices("AC[+57.0215]DE");
/// assert_eq!(result, "2");
///
/// let result = get_modification_indices("AC(UniMod:4)DE");
/// assert_eq!(result, "2");
/// ```
pub fn get_modification_indices(peptide: &str) -> String {
    let re = Regex::new(r"(\[.*?\]|\(UniMod:\d+\)|\([a-zA-Z]+\))").unwrap();
    let mut indices = Vec::new();
    let mut offset = 0;
    let mut aa_index = 0;
    let mut i = 0;

    while i < peptide.len() {
        let c = peptide[i..].chars().next().unwrap();

        if c == '[' || c == '(' {
            if let Some(mat) = re.find_at(peptide, i) {
                if mat.start() == i {
                    // If the modification is at the beginning (i == 0), it's on the N-term
                    indices.push(aa_index.to_string());
                    i = mat.end();
                    continue;
                }
            }
        }

        aa_index += 1;
        i += c.len_utf8();
    }

    indices.join(";")
}

/// Extracts mass shift annotations (e.g., [+57.0215]) from a peptide string and returns them
/// as a vector of (mass_string, position) where position is the index of the annotated amino acid.
///
/// # Example
/// ```
/// use redeem_properties::utils::peptdeep_utils::extract_mass_annotations;
/// let result = extract_mass_annotations("AC[+57.0215]DE");
/// assert_eq!(result, vec![("57.0215".to_string(), 2)]);
/// ```
pub fn extract_mass_annotations(peptide: &str) -> Vec<(String, usize)> {
    let re_mass = Regex::new(r"\[([+-]?\d*\.?\d+)\]").unwrap();
    let mut results = Vec::new();
    let mut offset = 0;
    let mut idx = 0;

    while idx < peptide.len() {
        if let Some(mat) = re_mass.find_at(peptide, idx) {
            if mat.start() == idx {
                let cap = re_mass.captures(&peptide[idx..mat.end()]).unwrap();
                let mass_str = format!("{:.4}", cap[1].parse::<f64>().unwrap_or(0.0));
                let pos = idx - offset;
                results.push((mass_str, pos));
                offset += mat.end() - mat.start();
                idx = mat.end();
                continue;
            }
        }
        idx += peptide[idx..].chars().next().unwrap().len_utf8();
    }

    results
}

/// Extracts UniMod annotations (e.g., (UniMod:4)) from a peptide string and returns them
/// as a vector of (unimod_id_string, position) where position is the index of the annotated amino acid.
///
/// # Example
/// ```
/// use redeem_properties::utils::peptdeep_utils::extract_unimod_annotations;
/// let result = extract_unimod_annotations("AC(UniMod:4)DE");
/// assert_eq!(result, vec![("UniMod:4".to_string(), 2)]);
/// ```
pub fn extract_unimod_annotations(peptide: &str) -> Vec<(String, usize)> {
    let re_unimod = Regex::new(r"\(UniMod:(\d+)\)").unwrap();
    let mut results = Vec::new();
    let mut offset = 0;
    let mut aa_index = 0;
    let mut idx = 0;

    while idx < peptide.len() {
        if let Some(mat) = re_unimod.find_at(peptide, idx) {
            if mat.start() == idx {
                // UniMod annotation
                let cap = re_unimod.captures(&peptide[idx..mat.end()]).unwrap();
                let unimod_str = format!("UniMod:{}", &cap[1]);
                results.push((unimod_str, aa_index));
                offset += mat.end() - mat.start();
                idx = mat.end();
                continue;
            }
        }

        // Only increment aa_index on actual amino acid
        let ch = peptide[idx..].chars().next().unwrap();
        if ch.is_alphabetic() {
            aa_index += 1;
        }
        idx += ch.len_utf8();
    }

    results
}

/// Extracts either mass shift or UniMod annotations from a peptide string,
/// returning a vector of (mod_str, position).
///
/// Dispatches to `extract_mass_annotations` if it finds `[+mass]`,
/// or to `extract_unimod_annotations` if it finds `(UniMod:id)`.
///
/// # Example
/// ```
/// use redeem_properties::utils::peptdeep_utils::extract_mod_annotations;
/// let mass = extract_mod_annotations("AC[+57.0215]DE");
/// assert_eq!(mass, vec![("57.0215".to_string(), 2)]);
///
/// let unimod = extract_mod_annotations("AC(UniMod:4)DE");
/// assert_eq!(unimod, vec![("UniMod:4".to_string(), 2)]);
/// ```
pub fn extract_mod_annotations(peptide: &str) -> Vec<(String, usize)> {
    if peptide.contains("[+") || peptide.contains("[-") {
        extract_mass_annotations(peptide)
    } else if peptide.contains("(UniMod:") {
        extract_unimod_annotations(peptide)
    } else {
        Vec::new()
    }
}

/// Attempts to look up a modification name from a map using the provided key and amino acid.
/// Falls back to a key with `None` if the exact amino acid is not matched.
///
/// # Example
/// ```
/// use redeem_properties::utils::peptdeep_utils::{ModificationMap, lookup_modification};
/// let mut map = std::collections::HashMap::new();
/// map.insert(("57.0215".to_string(), Some('C')), ModificationMap { name: "Carbamidomethyl@C".to_string(), amino_acid: Some('C'), unimod_id: Some(4) });
///
/// let result = lookup_modification("57.0215".to_string(), Some('C'), &map);
/// assert_eq!(result, Some("Carbamidomethyl@C".to_string()));
/// ```
pub fn lookup_modification(
    key: String,
    aa: Option<char>,
    map: &HashMap<(String, Option<char>), ModificationMap>,
) -> Option<String> {
    map.get(&(key.clone(), aa))
        .or_else(|| map.get(&(key, None)))
        .map(|m| m.name.clone())
}

/// Generates a standardized modification string (e.g., "Carbamidomethyl@C")
/// for a peptide sequence based on mass shifts (e.g., `[+57.0215]`) or
/// UniMod annotations (e.g., `(UniMod:4)`), using a preloaded modification map.
///
/// The function supports both mass-shift format and UniMod notation,
/// matching entries from the `modification_map` using mass or UniMod ID along
/// with the local amino acid context.
///
/// # Arguments
/// * `peptide` - A modified peptide sequence string (e.g., `"MGC[+57.0215]AAR"` or `"MGC(UniMod:4)AAR"`).
/// * `modification_map` - A HashMap mapping (key, amino_acid) to `ModificationMap`.
///   - For `[+mass]`, key is formatted as a mass string (e.g., `"57.0215"`).
///   - For `(UniMod:ID)`, key is the UniMod ID as string (e.g., `"4"`).
///
/// # Returns
/// A `String` containing semicolon-separated modification names (e.g., `"Carbamidomethyl@C"`).
///
/// # Example
/// ```
/// use std::collections::HashMap;
/// use redeem_properties::utils::peptdeep_utils::{load_modifications, get_modification_string};
///
/// let mod_map = load_modifications().unwrap();
/// let peptide1 = "MGC[+57.0215]AAR";
/// let result1 = get_modification_string(peptide1, &mod_map);
/// assert_eq!(result1, "Carbamidomethyl@C");
///
/// let peptide2 = "MGC(UniMod:4)AAR";
/// let result2 = get_modification_string(peptide2, &mod_map);
/// assert_eq!(result2, "Carbamidomethyl@C");
/// ```
pub fn get_modification_string(
    peptide: &str,
    modification_map: &HashMap<(String, Option<char>), ModificationMap>,
) -> String {
    let naked_peptide = remove_mass_shift(peptide);

    extract_mod_annotations(peptide)
        .into_iter()
        .filter_map(|(key, pos)| {
            let aa_opt = if pos == 0 {
                naked_peptide.chars().next()
            } else {
                naked_peptide.chars().nth(pos - 1)
            };

            // Try normal lookup first
            let mod_str = lookup_modification(key.clone(), aa_opt, modification_map);

            // If not found and it's a terminal mod, look for Protein_N-term
            if mod_str.is_none() && pos == 0 {
                // Try all entries with same key and look for *_N-term, preferring Protein_N-term
                let mut candidates: Vec<String> = modification_map
                    .iter()
                    .filter_map(|((k, _), v)| {
                        if k == &key
                            && (v.name.contains("Protein_N-term") || v.name.contains("Any_N-term"))
                        {
                            Some(v.name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                // Prefer Protein_N-term over Any_N-term for deterministic results
                candidates.sort_by(|a, b| {
                    let a_protein = a.contains("Protein_N-term");
                    let b_protein = b.contains("Protein_N-term");
                    b_protein.cmp(&a_protein)
                });
                candidates.into_iter().next()
            } else {
                mod_str
            }
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// Thread-safe, one-shot download and extraction of pretrained models.
///
/// Uses a global `OnceLock` so that even when many tests run in parallel
/// the download + unzip happens exactly once.  Subsequent callers just
/// get the cached path.
pub fn download_pretrained_models_exist() -> Result<PathBuf, io::Error> {
    use std::sync::OnceLock;
    static MODELS: OnceLock<Result<PathBuf, String>> = OnceLock::new();

    let result =
        MODELS.get_or_init(|| download_pretrained_models_inner().map_err(|e| e.to_string()));

    match result {
        Ok(path) => Ok(path.clone()),
        Err(msg) => Err(io::Error::new(io::ErrorKind::Other, msg.clone())),
    }
}

fn download_pretrained_models_inner() -> Result<PathBuf, io::Error> {
    let extract_dir = if let Ok(dir) = std::env::var("REDEEM_PRETRAINED_MODELS_DIR") {
        PathBuf::from(dir)
    } else {
        crate::pretrained::default_pretrained_models_dir()
    };
    let zip_path = extract_dir.join("pretrained_models.zip");

    // If already fully extracted, return immediately
    if extract_dir.exists() {
        return Ok(extract_dir);
    }

    // Ensure the parent directory exists
    if let Some(parent) = zip_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download the zip file if it doesn't exist
    if !zip_path.exists() {
        info!(
            "Downloading pretrained models from {} ...",
            PRETRAINED_MODELS_URL
        );

        // Use a temporary file so concurrent readers never see a partial zip
        let tmp_path = zip_path.with_extension("zip.tmp");

        let client = reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        let response = client
            .get(PRETRAINED_MODELS_URL)
            .header("Accept", "application/octet-stream")
            .send()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Failed to download pretrained models: HTTP {}",
                    response.status()
                ),
            ));
        }

        let bytes = response
            .bytes()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        // Verify we got a real zip (starts with PK magic bytes)
        if bytes.len() < 4 || &bytes[..2] != b"PK" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Downloaded file is not a valid zip archive ({} bytes, starts with {:?})",
                    bytes.len(),
                    &bytes[..bytes.len().min(16)]
                ),
            ));
        }

        fs::write(&tmp_path, &bytes)?;
        fs::rename(&tmp_path, &zip_path)?;
    }

    // Unzip directly into `extract_dir`. If the archive has a top-level
    // "pretrained_models/" folder, strip that prefix first.
    info!("Unzipping pretrained models...");
    let file = File::open(&zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    fs::create_dir_all(&extract_dir)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let archive_path = file.mangled_name();
        let relative_path = archive_path
            .strip_prefix("pretrained_models")
            .unwrap_or(archive_path.as_path());

        // Skip entries that correspond to the stripped root itself.
        if relative_path.as_os_str().is_empty() {
            continue;
        }

        let outpath = extract_dir.join(relative_path);

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(extract_dir)
}

pub fn parse_instrument_index(instrument: &str) -> usize {
    let upper_instrument = instrument.to_uppercase();

    INSTRUMENT_DICT
        .iter()
        .find(|&&(name, _)| name == upper_instrument)
        .map_or(UNKNOWN_INSTRUMENT_NUM, |&(_, index)| index)
}

// TODO: Derive from PeptDep constants yaml
const IM_GAS_MASS: f64 = 28.0;
const CCS_IM_COEF: f64 = 1059.62245;

/// Calculates the reduced mass for CCS and mobility calculation.
///
/// This function computes the reduced mass using the formula:
/// reduced_mass = (precursor_mz * charge * IM_GAS_MASS) / (precursor_mz * charge + IM_GAS_MASS)
///
/// # Arguments
///
/// * `precursor_mz` - The precursor mass-to-charge ratio (m/z)
/// * `charge` - The charge of the ion
///
/// # Returns
///
/// The calculated reduced mass as a f64 value
pub fn get_reduced_mass(precursor_mz: f64, charge: f64) -> f64 {
    let reduced_mass = precursor_mz * charge;
    reduced_mass * IM_GAS_MASS / (reduced_mass + IM_GAS_MASS)
}

/// Converts CCS (Collision Cross Section) to mobility for Bruker (timsTOF) instruments.
///
/// This function calculates the mobility using the formula:
/// mobility = (ccs_value * sqrt(reduced_mass)) / (charge * CCS_IM_COEF)
///
/// # Arguments
///
/// * `ccs_value` - The Collision Cross Section value
/// * `charge` - The charge of the ion
/// * `precursor_mz` - The precursor mass-to-charge ratio (m/z)
///
/// # Returns
///
/// The calculated mobility as a f64 value
pub fn ccs_to_mobility_bruker(ccs_value: f64, charge: f64, precursor_mz: f64) -> f64 {
    let reduced_mass = get_reduced_mass(precursor_mz, charge);
    ccs_value * f64::sqrt(reduced_mass) / (charge * CCS_IM_COEF)
}

/// Converts mobility to CCS (Collision Cross Section) for Bruker (timsTOF) instruments.
///
/// This function calculates the CCS using the formula:
/// ccs_value = (mobility * charge * CCS_IM_COEF) / sqrt(reduced_mass)
///
/// # Arguments
///
/// * `mobility` - The mobility value
/// * `charge` - The charge of the ion
/// * `precursor_mz` - The precursor mass-to-charge ratio (m/z)
///
/// # Returns
///
/// The calculated CCS value as a f64
pub fn ion_mobility_to_ccs_bruker(mobility: f64, charge: i32, precursor_mz: f64) -> f32 {
    let reduced_mass = get_reduced_mass(precursor_mz, charge as f64);
    ((mobility * (charge as f64) * CCS_IM_COEF) / f64::sqrt(reduced_mass)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn test_ensure_pretrained_models_exist() {
        let result = download_pretrained_models_exist();
        assert!(result.is_ok());
    }

    #[test]
    fn test_download_pretrained_models_exist_returns_valid_path() {
        let result = download_pretrained_models_exist();

        // Should return Ok with a valid path
        assert!(
            result.is_ok(),
            "Failed to download/extract pretrained models"
        );

        let path = result.unwrap();

        // Path should exist
        assert!(
            path.exists(),
            "Pretrained models path does not exist: {:?}",
            path
        );

        // Path should be a directory
        assert!(
            path.is_dir(),
            "Pretrained models path is not a directory: {:?}",
            path
        );

        // Check that the path matches expected location
        let expected = if let Ok(dir) = std::env::var("REDEEM_PRETRAINED_MODELS_DIR") {
            PathBuf::from(dir)
        } else {
            crate::pretrained::default_pretrained_models_dir()
        };
        assert_eq!(path, expected);
        
        println!("Pretrained models successfully available at: {:?}", path);
    }

    #[test]
    fn test_extract_unimod_annotations() {
        let peptide = "AC(UniMod:4)DE(UniMod:7)FG";
        let result = extract_unimod_annotations(peptide);
        println!("Peptide: {}, Result: {:?}", peptide, result);
        assert_eq!(
            result,
            vec![("UniMod:4".to_string(), 2), ("UniMod:7".to_string(), 4)]
        );

        let peptide = "AC(UniMod:4)DE(UniMod:7)FG(UniMod:10)";
        let result = extract_unimod_annotations(peptide);
        println!("Peptide: {}, Result: {:?}", peptide, result);
        assert_eq!(
            result,
            vec![
                ("UniMod:4".to_string(), 2),
                ("UniMod:7".to_string(), 4),
                ("UniMod:10".to_string(), 6)
            ]
        );

        let peptide = "(UniMod:1)M(UniMod:35)AAAATMAAAAR";
        let result = extract_unimod_annotations(peptide);
        println!("Peptide: {}, Result: {:?}", peptide, result);
        assert_eq!(
            result,
            vec![("UniMod:1".to_string(), 0), ("UniMod:35".to_string(), 1)]
        );
    }

    #[test]
    fn test_extract_mod_annotations() {
        let peptide = "[+42.0105]M[+15.9949]AAAATMAAAAR";
        let result = extract_mod_annotations(peptide);
        println!("Peptide: {}, Result: {:?}", peptide, result);
        assert_eq!(
            result,
            vec![("42.0105".to_string(), 0), ("15.9949".to_string(), 1)]
        );

        let peptide = "(UniMod:1)M(UniMod:35)AAAATMAAAAR";
        let result = extract_mod_annotations(peptide);
        println!("Peptide: {}, Result: {:?}", peptide, result);
        assert_eq!(
            result,
            vec![("UniMod:1".to_string(), 0), ("UniMod:35".to_string(), 1)]
        );
    }

    #[test]
    fn test_get_modification_indices() {
        // Compile the regex once for all tests
        // let re = Regex::new(r"\[.*?\]").unwrap();

        // Test cases
        let test_cases = vec![
            ("PEPTIDE", ""),
            ("PEPT[+15.9949]IDE", "4"),
            ("P[+15.9949]EPT[+79.99]IDE", "1;4"),
            ("TVQSLEIDLDSM[+15.9949]R", "12"),
            ("TVQS[+79.99]LEIDLDSM[+15.9949]R", "4;12"),
            ("[+42.0106]PEPTIDE", "0"),
            ("PEPTIDE[+42.0106]", "7"),
            ("P[+15.9949]EP[+79.99]T[+15.9949]IDE", "1;3;4"),
            ("(UniMod:1)M(UniMod:35)AAAATMAAAAR", "0;1"),
        ];

        for (peptide, expected) in test_cases {
            let result = get_modification_indices(peptide);
            println!(
                "Peptide: {}, Expected: {}, Result: {}",
                peptide, expected, result
            );
            assert_eq!(result, expected, "Failed for peptide: {}", peptide);
        }
    }

    #[test]
    fn test_extract_masses_and_indices() {
        let test_cases = vec![
            ("PEPTIDE", vec![]),
            ("PEPT[+15.9949]IDE", vec![(15.9949, 4)]),
            ("P[+15.9949]EPT[+79.99]IDE", vec![(15.9949, 1), (79.99, 4)]),
            ("TVQSLEIDLDSM[+15.9949]R", vec![(15.9949, 12)]),
            (
                "TVQS[+79.99]LEIDLDSM[+15.9949]R",
                vec![(79.99, 4), (15.9949, 12)],
            ),
            ("[+42.0106]PEPTIDE", vec![(42.0106, 0)]),
            ("PEPTIDE[+42.0106]", vec![(42.0106, 7)]),
            (
                "P[+15.9949]EP[+79.99]T[+15.9949]IDE",
                vec![(15.9949, 1), (79.99, 3), (15.9949, 4)],
            ),
            ("PEPTIDE[]", vec![]), // Empty modification
            ("P[+15.9949]EP[-79.99]TIDE", vec![(15.9949, 1), (-79.99, 3)]), // Negative mass
            ("P[+15.9949]EP[79.99]TIDE", vec![(15.9949, 1), (79.99, 3)]), // No plus sign
        ];

        for (peptide, expected) in test_cases {
            let result = extract_masses_and_indices(peptide);
            println!(
                "Peptide: {}, Expected: {:?}, Result: {:?}",
                peptide, expected, result
            );
            assert_eq!(result, expected, "Failed for peptide: {}", peptide);
        }
    }

    #[test]
    fn test_get_modification_string() {
        let modification_map = MODIFICATION_MAP.clone();

        let test_cases = vec![
            ("PEPTIDE", ""),
            ("PEPT[+15.9949]IDE", "Oxidation@T"),
            ("P[+15.9949]EPT[+79.9663]IDE", "Oxidation@P;Phospho@T"),
            ("TVQSLEIDLDSM[+15.9949]R", "Oxidation@M"),
            ("TVQS[+79.9663]LEIDLDSM[+15.9949]R", "Phospho@S;Oxidation@M"),
            (
                "(UniMod:1)M(UniMod:35)AAAATMAAAAR",
                "Acetyl@Protein_N-term;Oxidation@M",
            ),
            ("[+42.0106]PEPTIDE", "Acetyl@Protein_N-term"),
            ("PEPTIDE[+42.0106]", ""),
            (
                "P[+15.9949]EP[+79.9663]T[+15.9949]IDE",
                "Oxidation@P;Oxidation@T",
            ),
            (
                "(UniMod:1)M(UniMod:35)AAAATMAAAAR",
                "Acetyl@Protein_N-term;Oxidation@M",
            ),
        ];

        for (peptide, expected) in test_cases {
            let result = get_modification_string(peptide, &modification_map);
            println!(
                "Peptide: {}, Expected: {}, Result: {}",
                peptide, expected, result
            );
            assert_eq!(result, expected, "Failed for peptide: {}", peptide);
        }
    }

    #[test]
    fn test_ccs_to_mobility_bruker() {
        let ccs_value = 0.032652;
        let charge = 2.0;
        let precursor_mz = 762.329553;

        let result = ccs_to_mobility_bruker(ccs_value, charge, precursor_mz);

        let expected = 8.078969627454307e-05;
        println!("Rust result: {}", result);
        println!("Python result: {}", expected);
        println!("Difference: {}", (result - expected).abs());

        assert_eq!(result, expected, "Failed for ccs_to_mobility_bruker");
    }

    #[test]
    fn test_ion_mobility_to_ccs_bruker() {
        let mobility = 8.078969627454307e-05;
        let charge = 2;
        let precursor_mz = 762.329553;

        let result = ion_mobility_to_ccs_bruker(mobility, charge, precursor_mz);

        let expected = 0.032652;
        println!("Rust result: {}", result);
        println!("Python result: {}", expected);
        println!("Difference: {}", (result - expected).abs());

        assert_eq!(result, expected, "Failed for ion_mobility_to_ccs_bruker");
    }
}
