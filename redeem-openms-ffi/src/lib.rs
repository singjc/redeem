use std::ffi::{c_char, c_int, CStr, CString};
use std::fs;
use std::hash::{Hash, Hasher};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use csv::{ReaderBuilder, StringRecord};
use once_cell::sync::Lazy;
use redeem_properties::models::{
    ccs_model::CCSModelWrapper, model_interface::PredictionResult, ms2_model::MS2ModelWrapper,
    rt_model::RTModelWrapper,
};
use redeem_properties::utils::data_handling::{PeptideData, TargetNormalization};
use redeem_properties::utils::peptdeep_utils::{
    extract_mod_annotations, get_modification_indices, get_modification_string,
    ion_mobility_to_ccs_bruker, load_modifications, remove_mass_shift, ModificationMap,
    MODIFICATION_MAP,
};
use std::collections::HashMap;

const DEFAULT_NCE: i32 = 30;
const DEFAULT_INSTRUMENT: &str = "Lumos";
const RT_ARCH: &str = "rt_cnn_tf";
const CCS_ARCH: &str = "ccs_cnn_tf";
const MS2_ARCH: &str = "ms2_bert";

static LAST_ERROR: Lazy<Mutex<Option<CString>>> = Lazy::new(|| Mutex::new(None));

#[repr(C)]
pub struct OpenMsRedeemPredictorConfig {
    pub rt_model_path: *const c_char,
    pub ccs_model_path: *const c_char,
    pub ms2_model_path: *const c_char,
    pub device_preference: *const c_char,
}

#[repr(C)]
pub struct OpenMsRedeemPredictionInput {
    pub modified_peptide: *const c_char,
    pub precursor_charge: i32,
    pub nce: i32,
    pub instrument: *const c_char,
}

#[repr(C)]
pub struct OpenMsRedeemFineTuneConfig {
    pub training_tsv_path: *const c_char,
    pub validation_tsv_path: *const c_char,
    pub validation_fraction: f32,
    pub batch_size: usize,
    pub validation_batch_size: usize,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub learning_rate: f64,
    pub warmup_fraction: f64,
    pub default_nce: i32,
    pub default_instrument: *const c_char,
    pub enable_rt: c_int,
    pub enable_ccs: c_int,
    pub enable_ms2: c_int,
    pub rt_model_output_path: *const c_char,
    pub ccs_model_output_path: *const c_char,
    pub ms2_model_output_path: *const c_char,
}

#[repr(C)]
pub struct OpenMsRedeemBatchOutput {
    pub count: usize,
    pub has_ccs: c_int,
    pub rt_values: *mut f32,
    pub ccs_values: *mut f32,
    pub ms2_row_counts: *mut usize,
    pub ms2_values: *mut f32,
    pub ms2_value_count: usize,
}

impl Default for OpenMsRedeemBatchOutput {
    fn default() -> Self {
        Self {
            count: 0,
            has_ccs: 0,
            rt_values: ptr::null_mut(),
            ccs_values: ptr::null_mut(),
            ms2_row_counts: ptr::null_mut(),
            ms2_values: ptr::null_mut(),
            ms2_value_count: 0,
        }
    }
}

enum DevicePreference {
    Auto,
    Cpu,
    Cuda(usize),
}

struct ParsedInput {
    sequence: String,
    mods: String,
    mod_sites: String,
    charge: i32,
    nce: i32,
    instrument: Option<Arc<[u8]>>,
}

struct ParsedFineTuneConfig {
    training_tsv_path: PathBuf,
    validation_tsv_path: Option<PathBuf>,
    validation_fraction: f32,
    batch_size: usize,
    validation_batch_size: usize,
    epochs: usize,
    early_stopping_patience: usize,
    learning_rate: f64,
    warmup_fraction: Option<f64>,
    default_nce: i32,
    default_instrument: String,
    enable_rt: bool,
    enable_ccs: bool,
    enable_ms2: bool,
    rt_model_output_path: Option<PathBuf>,
    ccs_model_output_path: Option<PathBuf>,
    ms2_model_output_path: Option<PathBuf>,
}

pub struct Predictor {
    rt_model: RTModelWrapper,
    ccs_model: Option<CCSModelWrapper>,
    ms2_model: MS2ModelWrapper,
    rt_model_path: PathBuf,
    ccs_model_path: Option<PathBuf>,
    ms2_model_path: PathBuf,
}

impl Predictor {
    fn from_config(config: &OpenMsRedeemPredictorConfig) -> Result<Self> {
        let device = select_device(parse_device_preference(config.device_preference)?)?;
        let rt_model_path = required_model_path(config.rt_model_path, "RT")?;
        let ms2_model_path = required_model_path(config.ms2_model_path, "MS2")?;
        let ccs_model_path = optional_model_path(config.ccs_model_path)?;

        let rt_model = RTModelWrapper::new(
            &rt_model_path,
            neighboring_constants_path(&rt_model_path).as_ref(),
            RT_ARCH,
            device.clone(),
        )
        .with_context(|| format!("Failed to load RT model from {}", rt_model_path.display()))?;

        let ms2_model = MS2ModelWrapper::new(
            &ms2_model_path,
            neighboring_constants_path(&ms2_model_path).as_ref(),
            MS2_ARCH,
            device.clone(),
        )
        .with_context(|| format!("Failed to load MS2 model from {}", ms2_model_path.display()))?;

        let ccs_model = match ccs_model_path.as_ref() {
            Some(path) => Some(
                CCSModelWrapper::new(
                    path,
                    neighboring_constants_path(path).as_ref(),
                    CCS_ARCH,
                    device,
                )
                .with_context(|| format!("Failed to load CCS model from {}", path.display()))?,
            ),
            None => None,
        };

        Ok(Self {
            rt_model,
            ccs_model,
            ms2_model,
            rt_model_path,
            ccs_model_path,
            ms2_model_path,
        })
    }

    fn predict_batch(
        &self,
        inputs: &[OpenMsRedeemPredictionInput],
    ) -> Result<OpenMsRedeemBatchOutput> {
        let parsed_inputs: Vec<ParsedInput> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                parse_input(input)
                    .with_context(|| format!("Invalid input at batch index {}", index))
            })
            .collect::<Result<_>>()?;

        let sequences: Vec<Arc<[u8]>> = parsed_inputs
            .iter()
            .map(|input| Arc::from(input.sequence.clone().into_bytes()))
            .collect();
        let mods: Vec<Arc<[u8]>> = parsed_inputs
            .iter()
            .map(|input| Arc::from(input.mods.clone().into_bytes()))
            .collect();
        let mod_sites: Vec<Arc<[u8]>> = parsed_inputs
            .iter()
            .map(|input| Arc::from(input.mod_sites.clone().into_bytes()))
            .collect();
        let charges: Vec<i32> = parsed_inputs.iter().map(|input| input.charge).collect();
        let nces: Vec<i32> = parsed_inputs.iter().map(|input| input.nce).collect();
        let instruments: Vec<Option<Arc<[u8]>>> = parsed_inputs
            .iter()
            .map(|input| input.instrument.clone())
            .collect();

        let rt_predictions = match self.rt_model.predict(&sequences, &mods, &mod_sites)? {
            PredictionResult::RTResult(values) => values,
            other => {
                return Err(anyhow!(
                    "RT prediction returned unexpected result type with {} entries",
                    other.len()
                ))
            }
        };

        let ccs_predictions = match &self.ccs_model {
            Some(model) => match model.predict(&sequences, &mods, &mod_sites, charges.clone())? {
                PredictionResult::CCSResult(values) => Some(values),
                other => {
                    return Err(anyhow!(
                        "CCS prediction returned unexpected result type with {} entries",
                        other.len()
                    ))
                }
            },
            None => None,
        };

        let ms2_predictions = match self.ms2_model.predict(
            &sequences,
            &mods,
            &mod_sites,
            charges,
            nces,
            instruments,
        )? {
            PredictionResult::MS2Result(values) => values,
            other => {
                return Err(anyhow!(
                    "MS2 prediction returned unexpected result type with {} entries",
                    other.len()
                ))
            }
        };

        if rt_predictions.len() != parsed_inputs.len() {
            return Err(anyhow!(
                "RT prediction count mismatch: expected {}, got {}",
                parsed_inputs.len(),
                rt_predictions.len()
            ));
        }
        if let Some(values) = &ccs_predictions {
            if values.len() != parsed_inputs.len() {
                return Err(anyhow!(
                    "CCS prediction count mismatch: expected {}, got {}",
                    parsed_inputs.len(),
                    values.len()
                ));
            }
        }
        if ms2_predictions.len() != parsed_inputs.len() {
            return Err(anyhow!(
                "MS2 prediction count mismatch: expected {}, got {}",
                parsed_inputs.len(),
                ms2_predictions.len()
            ));
        }

        let mut row_counts = Vec::with_capacity(ms2_predictions.len());
        let total_row_count: usize = ms2_predictions.iter().map(|matrix| matrix.len()).sum();
        let mut flat_ms2 = Vec::with_capacity(total_row_count * 8);

        for matrix in ms2_predictions {
            row_counts.push(matrix.len());
            for row in matrix {
                if row.len() != 8 {
                    return Err(anyhow!(
                        "MS2 prediction row has {} columns; expected 8 channels",
                        row.len()
                    ));
                }
                flat_ms2.extend(row);
            }
        }

        Ok(OpenMsRedeemBatchOutput {
            count: rt_predictions.len(),
            has_ccs: if ccs_predictions.is_some() { 1 } else { 0 },
            rt_values: boxed_slice_into_raw(rt_predictions.into_boxed_slice()),
            ccs_values: ccs_predictions
                .map(|values| boxed_slice_into_raw(values.into_boxed_slice()))
                .unwrap_or(ptr::null_mut()),
            ms2_row_counts: boxed_slice_into_raw(row_counts.into_boxed_slice()),
            ms2_value_count: flat_ms2.len(),
            ms2_values: boxed_slice_into_raw(flat_ms2.into_boxed_slice()),
        })
    }

    fn fine_tune_from_transition_tsv(&mut self, config: &OpenMsRedeemFineTuneConfig) -> Result<()> {
        let config = parse_fine_tune_config(config)?;
        let _validation_batch_size = config.validation_batch_size;
        let modifications = load_modifications().context("Failed to load modification map")?;

        if config.enable_rt {
            let (train_data, validation_data, norm) =
                load_fine_tune_data_for_model(&config, RT_ARCH, &modifications)?;
            self.rt_model
                .fine_tune(
                    &train_data,
                    validation_data.as_ref(),
                    modifications.clone(),
                    config.batch_size,
                    config.learning_rate,
                    config.epochs,
                    Some(config.early_stopping_patience),
                    norm,
                    config.warmup_fraction,
                )
                .context("Failed to fine-tune RT model")?;

            if let Some(path) = &config.rt_model_output_path {
                save_model_with_constants(
                    &mut self.rt_model,
                    path,
                    Some(&self.rt_model_path),
                    "RT",
                )?;
            }
        }

        if config.enable_ccs {
            let ccs_model = self
                .ccs_model
                .as_mut()
                .ok_or_else(|| anyhow!("CCS fine-tuning requested, but no CCS model is loaded"))?;
            let (train_data, validation_data, norm) =
                load_fine_tune_data_for_model(&config, CCS_ARCH, &modifications)?;
            ccs_model
                .fine_tune(
                    &train_data,
                    validation_data.as_ref(),
                    modifications.clone(),
                    config.batch_size,
                    config.learning_rate,
                    config.epochs,
                    Some(config.early_stopping_patience),
                    norm,
                    config.warmup_fraction,
                )
                .context("Failed to fine-tune CCS model")?;

            if let Some(path) = &config.ccs_model_output_path {
                save_model_with_constants(ccs_model, path, self.ccs_model_path.as_ref(), "CCS")?;
            }
        }

        if config.enable_ms2 {
            let (train_data, validation_data, norm) =
                load_fine_tune_data_for_model(&config, MS2_ARCH, &modifications)?;
            self.ms2_model
                .fine_tune(
                    &train_data,
                    validation_data.as_ref(),
                    modifications,
                    config.batch_size,
                    config.learning_rate,
                    config.epochs,
                    Some(config.early_stopping_patience),
                    norm,
                    config.warmup_fraction,
                )
                .context("Failed to fine-tune MS2 model")?;

            if let Some(path) = &config.ms2_model_output_path {
                save_model_with_constants(
                    &mut self.ms2_model,
                    path,
                    Some(&self.ms2_model_path),
                    "MS2",
                )?;
            }
        }

        Ok(())
    }
}

trait SaveableModel {
    fn save_model(&mut self, path: &str) -> Result<()>;
}

impl SaveableModel for RTModelWrapper {
    fn save_model(&mut self, path: &str) -> Result<()> {
        self.save(path)
    }
}

impl SaveableModel for CCSModelWrapper {
    fn save_model(&mut self, path: &str) -> Result<()> {
        self.save(path)
    }
}

impl SaveableModel for MS2ModelWrapper {
    fn save_model(&mut self, path: &str) -> Result<()> {
        self.save(path)
    }
}

fn parse_fine_tune_config(config: &OpenMsRedeemFineTuneConfig) -> Result<ParsedFineTuneConfig> {
    let training_tsv_path = required_existing_path(config.training_tsv_path, "training_tsv_path")?;
    let validation_tsv_path =
        optional_existing_path(config.validation_tsv_path, "validation_tsv_path")?;
    let batch_size = config.batch_size.max(1);
    let validation_batch_size = config.validation_batch_size.max(1);
    let epochs = config.epochs.max(1);
    let early_stopping_patience = config.early_stopping_patience.max(1);
    let learning_rate = if config.learning_rate > 0.0 {
        config.learning_rate
    } else {
        1e-4
    };
    let warmup_fraction = (config.warmup_fraction > 0.0).then_some(config.warmup_fraction);
    let default_nce = if config.default_nce > 0 {
        config.default_nce
    } else {
        DEFAULT_NCE
    };
    let default_instrument = optional_string_from_c_str(config.default_instrument)?
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_INSTRUMENT.to_string());

    let enable_rt = config.enable_rt != 0;
    let enable_ccs = config.enable_ccs != 0;
    let enable_ms2 = config.enable_ms2 != 0;
    if !enable_rt && !enable_ccs && !enable_ms2 {
        return Err(anyhow!(
            "At least one of enable_rt, enable_ccs, or enable_ms2 must be enabled for fine-tuning"
        ));
    }

    Ok(ParsedFineTuneConfig {
        training_tsv_path,
        validation_tsv_path,
        validation_fraction: config.validation_fraction.clamp(0.0, 0.95),
        batch_size,
        validation_batch_size,
        epochs,
        early_stopping_patience,
        learning_rate,
        warmup_fraction,
        default_nce,
        default_instrument,
        enable_rt,
        enable_ccs,
        enable_ms2,
        rt_model_output_path: optional_output_model_path(
            config.rt_model_output_path,
            "rt_model_output_path",
        )?,
        ccs_model_output_path: optional_output_model_path(
            config.ccs_model_output_path,
            "ccs_model_output_path",
        )?,
        ms2_model_output_path: optional_output_model_path(
            config.ms2_model_output_path,
            "ms2_model_output_path",
        )?,
    })
}

fn load_fine_tune_data_for_model(
    config: &ParsedFineTuneConfig,
    model_arch: &str,
    modifications: &HashMap<(String, Option<char>), ModificationMap>,
) -> Result<(
    Vec<PeptideData>,
    Option<Vec<PeptideData>>,
    TargetNormalization,
)> {
    let train_raw = load_transition_training_records(
        &config.training_tsv_path,
        model_arch,
        config.default_nce,
        &config.default_instrument,
        modifications,
    )?;

    if train_raw.is_empty() {
        return Err(anyhow!(
            "No training precursors with usable {} targets were loaded from {}",
            model_arch,
            config.training_tsv_path.display()
        ));
    }

    let (train_raw, validation_raw) = if let Some(path) = &config.validation_tsv_path {
        let validation_raw = load_transition_training_records(
            path,
            model_arch,
            config.default_nce,
            &config.default_instrument,
            modifications,
        )?;
        if validation_raw.is_empty() {
            return Err(anyhow!(
                "No validation precursors with usable {} targets were loaded from {}",
                model_arch,
                path.display()
            ));
        }
        (train_raw, Some(validation_raw))
    } else {
        split_training_validation(train_raw, config.validation_fraction)
    };

    let norm = compute_target_normalization(&train_raw, model_arch)?;

    let mut train_data = train_raw;
    apply_target_normalization(&mut train_data, model_arch, norm);

    let validation_data = validation_raw.map(|mut values| {
        apply_target_normalization(&mut values, model_arch, norm);
        values
    });

    Ok((train_data, validation_data, norm))
}

fn load_transition_training_records(
    path: &Path,
    model_arch: &str,
    default_nce: i32,
    default_instrument: &str,
    modifications: &HashMap<(String, Option<char>), ModificationMap>,
) -> Result<Vec<PeptideData>> {
    let delimiter = match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => b',',
        _ => b'\t',
    };
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("Failed to open fine-tuning TSV {}", path.display()))?;
    let headers = reader.headers()?.clone();

    let has_fragment_columns = has_transition_fragment_columns(&headers);
    if model_arch == MS2_ARCH && !has_fragment_columns {
        return Err(anyhow!(
            "MS2 fine-tuning requires transition-level columns such as fragment_type, fragment_series_number, product_charge, and intensity"
        ));
    }

    let mut peptide_map: HashMap<(String, i32), PeptideData> = HashMap::new();
    for record in reader.records() {
        let record = record?;
        let modified_sequence = required_record_field(
            &record,
            &headers,
            &[
                "modified_peptide_sequence",
                "modifiedpeptidesequence",
                "modifiedpeptide",
                "fullpeptidename",
                "modified_sequence",
                "sequence",
                "peptide_sequence",
                "peptide",
            ],
            "modified peptide sequence",
        )?
        .to_string();
        let charge = required_record_field(
            &record,
            &headers,
            &[
                "precursorcharge",
                "precursor_charge",
                "precursor charge",
                "charge",
            ],
            "precursor charge",
        )?
        .parse::<i32>()
        .with_context(|| {
            format!(
                "Invalid precursor charge '{}' in {}",
                required_record_field(
                    &record,
                    &headers,
                    &[
                        "precursorcharge",
                        "precursor_charge",
                        "precursor charge",
                        "charge",
                    ],
                    "precursor charge",
                )
                .unwrap_or(""),
                path.display()
            )
        })?;
        if charge <= 0 {
            continue;
        }

        let parsed = parse_modified_sequence_for_training(
            &modified_sequence,
            charge,
            default_nce,
            record_field(
                &record,
                &headers,
                &["collisionenergy", "collision_energy", "nce"],
            )
            .and_then(|value| value.parse::<i32>().ok()),
            record_field(&record, &headers, &["instrument"]).map(|value| value.to_string()),
            default_instrument,
            modifications,
        )?;

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
        .and_then(|value| value.parse::<f32>().ok());
        let retention_time = record_field(
            &record,
            &headers,
            &[
                "normalizedretentiontime",
                "retention_time",
                "retention time",
                "retentiontime",
                "rt",
            ],
        )
        .and_then(|value| value.parse::<f32>().ok());
        let ion_mobility = record_field(
            &record,
            &headers,
            &["precursorionmobility", "ion_mobility", "ion mobility", "im"],
        )
        .and_then(|value| value.parse::<f32>().ok());
        let ccs = record_field(&record, &headers, &["ccs"])
            .and_then(|value| value.parse::<f32>().ok())
            .or_else(|| {
                ion_mobility.zip(precursor_mass).map(|(mobility, mz)| {
                    ion_mobility_to_ccs_bruker(mobility as f64, charge, mz as f64)
                })
            });

        let has_ms2_matrix = has_fragment_columns;
        let key = (modified_sequence.clone(), charge);
        let peptide_len = parsed.sequence.len();
        let entry = peptide_map.entry(key).or_insert_with(|| PeptideData {
            modified_sequence: Arc::from(modified_sequence.as_bytes().to_vec().into_boxed_slice()),
            naked_sequence: Arc::from(parsed.sequence.clone().into_bytes()),
            mods: Arc::from(parsed.mods.clone().into_bytes()),
            mod_sites: Arc::from(parsed.mod_sites.clone().into_bytes()),
            charge: Some(charge),
            precursor_mass,
            nce: Some(parsed.nce),
            instrument: parsed.instrument.clone(),
            retention_time,
            ion_mobility,
            ccs,
            ms2_intensities: has_ms2_matrix
                .then(|| vec![vec![0.0; 8]; peptide_len.saturating_sub(1)]),
        });

        if entry.precursor_mass.is_none() {
            entry.precursor_mass = precursor_mass;
        }
        if entry.retention_time.is_none() {
            entry.retention_time = retention_time;
        }
        if entry.ion_mobility.is_none() {
            entry.ion_mobility = ion_mobility;
        }
        if entry.ccs.is_none() {
            entry.ccs = ccs;
        }
        if entry.nce.is_none() {
            entry.nce = Some(parsed.nce);
        }
        if entry.instrument.is_none() {
            entry.instrument = parsed.instrument.clone();
        }

        if has_fragment_columns {
            fill_ms2_entry(entry, &record, &headers);
        }
    }

    let peptides = peptide_map
        .into_values()
        .filter(|peptide| match model_arch {
            RT_ARCH => peptide.retention_time.is_some(),
            CCS_ARCH => peptide.ccs.is_some(),
            MS2_ARCH => peptide.ms2_intensities.is_some(),
            _ => false,
        })
        .collect();

    Ok(peptides)
}

fn fill_ms2_entry(peptide: &mut PeptideData, record: &StringRecord, headers: &StringRecord) {
    let fragment_type = match record_field(record, headers, &["fragmenttype", "fragment_type"]) {
        Some(value) => value.trim().to_ascii_lowercase(),
        None => return,
    };
    let series_number = match record_field(
        record,
        headers,
        &[
            "fragmentseriesnumber",
            "fragment_series_number",
            "series_number",
        ],
    )
    .and_then(|value| value.parse::<usize>().ok())
    {
        Some(value) if value > 0 => value,
        _ => return,
    };
    let product_charge = record_field(
        record,
        headers,
        &["productcharge", "product_charge", "fragment_charge"],
    )
    .and_then(|value| value.parse::<i32>().ok())
    .unwrap_or(1);
    let intensity = record_field(record, headers, &["libraryintensity", "intensity"])
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);

    let column = match (fragment_type.as_str(), product_charge) {
        ("b", 1) => 0,
        ("b", 2) => 1,
        ("y", 1) => 2,
        ("y", 2) => 3,
        _ => return,
    };
    let row = series_number.saturating_sub(1);

    if let Some(matrix) = peptide.ms2_intensities.as_mut() {
        if row < matrix.len() && column < matrix[row].len() {
            matrix[row][column] = intensity;
        }
    }
}

fn has_transition_fragment_columns(headers: &StringRecord) -> bool {
    record_header_idx(headers, &["fragmenttype", "fragment_type"]).is_some()
        && record_header_idx(
            headers,
            &[
                "fragmentseriesnumber",
                "fragment_series_number",
                "series_number",
            ],
        )
        .is_some()
        && record_header_idx(headers, &["libraryintensity", "intensity"]).is_some()
}

fn parse_modified_sequence_for_training(
    peptide: &str,
    charge: i32,
    default_nce: i32,
    explicit_nce: Option<i32>,
    explicit_instrument: Option<String>,
    default_instrument: &str,
    modifications: &HashMap<(String, Option<char>), ModificationMap>,
) -> Result<ParsedInput> {
    let peptide = canonicalize_modified_peptide_for_redeem(peptide)?;
    let annotation_count = extract_mod_annotations(&peptide).len();
    let sequence = remove_mass_shift(&peptide);
    if sequence.is_empty() {
        return Err(anyhow!(
            "Modified peptide '{}' resolved to an empty sequence during fine-tuning",
            peptide
        ));
    }

    let mods = get_modification_string(&peptide, modifications);
    let mod_sites = get_modification_indices(&peptide);
    let resolved_mod_count = split_count(&mods);
    let site_count = split_count(&mod_sites);
    if annotation_count != resolved_mod_count || annotation_count != site_count {
        return Err(anyhow!(
            "Unsupported modification syntax or unmapped modification in fine-tuning peptide '{}'",
            peptide
        ));
    }

    let instrument = explicit_instrument
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_instrument.to_string());

    Ok(ParsedInput {
        sequence,
        mods,
        mod_sites,
        charge,
        nce: explicit_nce
            .filter(|value| *value > 0)
            .unwrap_or(default_nce),
        instrument: Some(Arc::from(instrument.into_bytes())),
    })
}

fn compute_target_normalization(
    peptides: &[PeptideData],
    model_arch: &str,
) -> Result<TargetNormalization> {
    let values: Vec<f32> = match model_arch {
        RT_ARCH => peptides
            .iter()
            .filter_map(|peptide| peptide.retention_time)
            .collect(),
        CCS_ARCH => peptides.iter().filter_map(|peptide| peptide.ccs).collect(),
        MS2_ARCH => peptides
            .iter()
            .filter_map(|peptide| peptide.ms2_intensities.as_ref())
            .flat_map(|matrix| matrix.iter().flat_map(|row| row.iter().copied()))
            .filter(|value| value.is_finite())
            .collect(),
        _ => {
            return Err(anyhow!(
                "Unsupported model architecture '{}' for target normalization",
                model_arch
            ))
        }
    };

    if values.is_empty() {
        return Err(anyhow!(
            "No target values were found while preparing {} fine-tuning data",
            model_arch
        ));
    }

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(TargetNormalization::MinMax(min, max))
}

fn apply_target_normalization(
    peptides: &mut [PeptideData],
    model_arch: &str,
    norm: TargetNormalization,
) {
    let normalize = |value: f32| match norm {
        TargetNormalization::MinMax(min, max) if max != min => (value - min) / (max - min),
        TargetNormalization::ZScore(mean, std) if std != 0.0 => (value - mean) / std,
        TargetNormalization::MinMax(_, _) | TargetNormalization::ZScore(_, _) => 0.0,
        TargetNormalization::None => value,
    };

    match model_arch {
        RT_ARCH => {
            for peptide in peptides {
                if let Some(value) = peptide.retention_time.as_mut() {
                    *value = normalize(*value);
                }
            }
        }
        CCS_ARCH => {
            for peptide in peptides {
                if let Some(value) = peptide.ccs.as_mut() {
                    *value = normalize(*value);
                }
            }
        }
        MS2_ARCH => {
            for peptide in peptides {
                if let Some(matrix) = peptide.ms2_intensities.as_mut() {
                    for row in matrix {
                        for value in row {
                            *value = normalize(*value);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn split_training_validation(
    peptides: Vec<PeptideData>,
    validation_fraction: f32,
) -> (Vec<PeptideData>, Option<Vec<PeptideData>>) {
    if peptides.len() < 2 || validation_fraction <= 0.0 {
        return (peptides, None);
    }

    let threshold = (validation_fraction.clamp(0.0, 0.95) * 10_000.0) as u64;
    let mut train = Vec::new();
    let mut validation = Vec::new();

    for peptide in peptides {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        peptide.modified_sequence.hash(&mut hasher);
        peptide.charge.hash(&mut hasher);
        if hasher.finish() % 10_000 < threshold {
            validation.push(peptide);
        } else {
            train.push(peptide);
        }
    }

    if train.is_empty() && !validation.is_empty() {
        if let Some(peptide) = validation.pop() {
            train.push(peptide);
        }
    }
    if validation.is_empty() && train.len() > 1 {
        if let Some(peptide) = train.pop() {
            validation.push(peptide);
        }
    }

    if validation.is_empty() {
        (train, None)
    } else {
        (train, Some(validation))
    }
}

fn record_field<'a>(
    record: &'a StringRecord,
    headers: &StringRecord,
    aliases: &[&str],
) -> Option<&'a str> {
    record_header_idx(headers, aliases).and_then(|index| record.get(index))
}

fn required_record_field<'a>(
    record: &'a StringRecord,
    headers: &StringRecord,
    aliases: &[&str],
    label: &str,
) -> Result<&'a str> {
    record_field(record, headers, aliases)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("Missing required {label} column in fine-tuning TSV"))
}

fn record_header_idx(headers: &StringRecord, aliases: &[&str]) -> Option<usize> {
    let normalized_aliases: Vec<String> = aliases
        .iter()
        .map(|value| normalize_header_name(value))
        .collect();

    headers.iter().position(|header| {
        let normalized_header = normalize_header_name(header);
        normalized_aliases
            .iter()
            .any(|alias| normalized_header == *alias || normalized_header.contains(alias))
    })
}

fn normalize_header_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && *character != '_' && *character != '-')
        .flat_map(|character| character.to_lowercase())
        .collect()
}

fn save_model_with_constants<M: SaveableModel>(
    model: &mut M,
    output_path: &Path,
    source_model_path: Option<&PathBuf>,
    label: &str,
) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    let output = output_path
        .to_str()
        .ok_or_else(|| anyhow!("{label} output path is not valid UTF-8"))?;
    model.save_model(output).with_context(|| {
        format!(
            "Failed to save {label} fine-tuned model to {}",
            output_path.display()
        )
    })?;
    copy_neighboring_constants(source_model_path, output_path)?;
    Ok(())
}

fn copy_neighboring_constants(
    source_model_path: Option<&PathBuf>,
    output_path: &Path,
) -> Result<()> {
    let Some(source_model_path) = source_model_path else {
        return Ok(());
    };
    let Some(source_constants_path) = neighboring_constants_path(source_model_path) else {
        return Ok(());
    };
    let extension = output_path
        .extension()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            anyhow!(
                "Output model path '{}' must include a file extension",
                output_path.display()
            )
        })?;
    let destination = output_path.with_extension(format!("{extension}.model_const.yaml"));
    fs::copy(&source_constants_path, &destination).with_context(|| {
        format!(
            "Failed to copy model constants from {} to {}",
            source_constants_path.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn parse_input(input: &OpenMsRedeemPredictionInput) -> Result<ParsedInput> {
    let peptide = string_from_c_str(input.modified_peptide, "modified_peptide")?;
    if peptide.is_empty() {
        return Err(anyhow!("modified_peptide must not be empty"));
    }
    if input.precursor_charge <= 0 {
        return Err(anyhow!(
            "precursor_charge must be positive, got {}",
            input.precursor_charge
        ));
    }

    let peptide = canonicalize_modified_peptide_for_redeem(&peptide)?;
    let annotation_count = extract_mod_annotations(&peptide).len();
    let sequence = remove_mass_shift(&peptide);
    if sequence.is_empty() {
        return Err(anyhow!("modified_peptide resolved to an empty sequence"));
    }

    let mods = get_modification_string(&peptide, &MODIFICATION_MAP);
    let mod_sites = get_modification_indices(&peptide);
    let resolved_mod_count = split_count(&mods);
    let site_count = split_count(&mod_sites);

    if annotation_count != resolved_mod_count || annotation_count != site_count {
        return Err(anyhow!(
            "Unsupported modification syntax or unmapped modification in peptide '{}'",
            peptide
        ));
    }

    let nce = if input.nce > 0 {
        input.nce
    } else {
        DEFAULT_NCE
    };
    let instrument_string = optional_string_from_c_str(input.instrument)?
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_INSTRUMENT.to_string());

    Ok(ParsedInput {
        sequence,
        mods,
        mod_sites,
        charge: input.precursor_charge,
        nce,
        instrument: Some(Arc::from(instrument_string.into_bytes())),
    })
}

fn canonicalize_modified_peptide_for_redeem(peptide: &str) -> Result<String> {
    let canonical = peptide.strip_prefix('.').unwrap_or(peptide).to_string();

    if canonical.contains(".(") || canonical.contains(".[") {
        return Err(anyhow!(
            "C-terminal peptide modifications are not supported in peptide '{}'",
            peptide
        ));
    }

    Ok(canonical)
}

fn split_count(value: &str) -> usize {
    if value.is_empty() {
        0
    } else {
        value.split(';').count()
    }
}

fn parse_device_preference(raw: *const c_char) -> Result<DevicePreference> {
    let preference = optional_string_from_c_str(raw)?
        .unwrap_or_else(|| "auto".to_string())
        .to_ascii_lowercase();

    match preference.as_str() {
        "" | "auto" => Ok(DevicePreference::Auto),
        "cpu" => Ok(DevicePreference::Cpu),
        "cuda" => Ok(DevicePreference::Cuda(0)),
        _ if preference.starts_with("cuda:") => {
            let index = preference
                .split(':')
                .nth(1)
                .ok_or_else(|| anyhow!("Invalid CUDA device string '{}'", preference))?
                .parse::<usize>()
                .with_context(|| format!("Invalid CUDA device string '{}'", preference))?;
            Ok(DevicePreference::Cuda(index))
        }
        _ => Err(anyhow!(
            "Unsupported device preference '{}'; expected auto, cpu, cuda, or cuda:N",
            preference
        )),
    }
}

fn select_device(preference: DevicePreference) -> Result<Device> {
    match preference {
        DevicePreference::Cpu => Ok(Device::Cpu),
        DevicePreference::Auto => select_auto_device(),
        DevicePreference::Cuda(index) => select_cuda_device(index),
    }
}

fn select_auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            return redeem_properties::utils::utils::get_device("cuda");
        }
    }
    Ok(Device::Cpu)
}

fn select_cuda_device(_index: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let device_name = if _index == 0 {
            "cuda".to_string()
        } else {
            format!("cuda:{_index}")
        };
        return redeem_properties::utils::utils::get_device(&device_name);
    }

    #[allow(unreachable_code)]
    Err(anyhow!(
        "CUDA was requested, but redeem-openms-ffi was built without the 'cuda' feature"
    ))
}

fn neighboring_constants_path(model_path: &Path) -> Option<PathBuf> {
    let extension = model_path.extension()?.to_str()?;
    let candidate = model_path.with_extension(format!("{extension}.model_const.yaml"));
    candidate.exists().then_some(candidate)
}

fn required_existing_path(path: *const c_char, label: &str) -> Result<PathBuf> {
    let value = string_from_c_str(path, label)?;
    if value.is_empty() {
        return Err(anyhow!("{label} must not be empty"));
    }
    let path = PathBuf::from(value);
    if !path.exists() {
        return Err(anyhow!("{label} does not exist: {}", path.display()));
    }
    Ok(path)
}

fn optional_existing_path(path: *const c_char, label: &str) -> Result<Option<PathBuf>> {
    match optional_string_from_c_str(path)? {
        Some(value) if !value.is_empty() => {
            let path = PathBuf::from(value);
            if !path.exists() {
                return Err(anyhow!("{label} does not exist: {}", path.display()));
            }
            Ok(Some(path))
        }
        _ => Ok(None),
    }
}

fn optional_output_model_path(path: *const c_char, label: &str) -> Result<Option<PathBuf>> {
    match optional_string_from_c_str(path)? {
        Some(value) if !value.is_empty() => {
            let path = PathBuf::from(value);
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            if extension.to_ascii_lowercase() != "safetensors" {
                return Err(anyhow!(
                    "{label} must use the .safetensors extension so the fine-tuned model can be reloaded"
                ));
            }
            Ok(Some(path))
        }
        _ => Ok(None),
    }
}

fn required_model_path(path: *const c_char, label: &str) -> Result<PathBuf> {
    let value = string_from_c_str(path, label)?;
    if value.is_empty() {
        return Err(anyhow!("{label} model path must not be empty"));
    }
    let path = PathBuf::from(value);
    if !path.exists() {
        return Err(anyhow!(
            "{label} model path does not exist: {}",
            path.display()
        ));
    }
    Ok(path)
}

fn optional_model_path(path: *const c_char) -> Result<Option<PathBuf>> {
    match optional_string_from_c_str(path)? {
        Some(value) if !value.is_empty() => {
            let path = PathBuf::from(value);
            if !path.exists() {
                return Err(anyhow!("CCS model path does not exist: {}", path.display()));
            }
            Ok(Some(path))
        }
        _ => Ok(None),
    }
}

fn string_from_c_str(ptr: *const c_char, field_name: &str) -> Result<String> {
    optional_string_from_c_str(ptr)?.ok_or_else(|| anyhow!("{field_name} pointer must not be null"))
}

fn optional_string_from_c_str(ptr: *const c_char) -> Result<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }

    let c_str = unsafe { CStr::from_ptr(ptr) };
    Ok(Some(
        c_str
            .to_str()
            .context("Input string is not valid UTF-8")?
            .to_string(),
    ))
}

fn boxed_slice_into_raw<T>(values: Box<[T]>) -> *mut T {
    Box::into_raw(values) as *mut T
}

fn set_last_error(error: impl AsRef<str>) {
    let sanitized = error.as_ref().replace('\0', " ");
    let message =
        CString::new(sanitized).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
    if let Ok(mut slot) = LAST_ERROR.lock() {
        *slot = Some(message);
    }
}

fn clear_last_error() {
    if let Ok(mut slot) = LAST_ERROR.lock() {
        *slot = None;
    }
}

fn set_panic_error() {
    set_last_error("Panic crossed the FFI boundary");
}

#[no_mangle]
pub extern "C" fn openms_redeem_predictor_create(
    config: *const OpenMsRedeemPredictorConfig,
) -> *mut Predictor {
    match std::panic::catch_unwind(|| {
        clear_last_error();

        if config.is_null() {
            set_last_error("Predictor config pointer must not be null");
            return ptr::null_mut();
        }

        match Predictor::from_config(unsafe { &*config }) {
            Ok(predictor) => Box::into_raw(Box::new(predictor)),
            Err(error) => {
                set_last_error(format!("{error:#}"));
                ptr::null_mut()
            }
        }
    }) {
        Ok(ptr) => ptr,
        Err(_) => {
            set_panic_error();
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn openms_redeem_predictor_destroy(predictor: *mut Predictor) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if !predictor.is_null() {
            unsafe {
                drop(Box::from_raw(predictor));
            }
        }
    }));
}

#[no_mangle]
pub extern "C" fn openms_redeem_predict_batch(
    predictor: *const Predictor,
    inputs: *const OpenMsRedeemPredictionInput,
    input_count: usize,
    output: *mut OpenMsRedeemBatchOutput,
) -> c_int {
    match std::panic::catch_unwind(AssertUnwindSafe(|| {
        clear_last_error();

        if predictor.is_null() {
            set_last_error("Predictor pointer must not be null");
            return 0;
        }
        if output.is_null() {
            set_last_error("Output pointer must not be null");
            return 0;
        }

        unsafe {
            ptr::write(output, OpenMsRedeemBatchOutput::default());
        }

        let input_slice = if input_count == 0 {
            &[]
        } else {
            if inputs.is_null() {
                set_last_error("Input pointer must not be null when input_count is non-zero");
                return 0;
            }
            unsafe { slice::from_raw_parts(inputs, input_count) }
        };

        match unsafe { &*predictor }.predict_batch(input_slice) {
            Ok(batch_output) => {
                unsafe {
                    ptr::write(output, batch_output);
                }
                1
            }
            Err(error) => {
                set_last_error(format!("{error:#}"));
                0
            }
        }
    })) {
        Ok(status) => status,
        Err(_) => {
            set_panic_error();
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn openms_redeem_predictor_fine_tune_from_tsv(
    predictor: *mut Predictor,
    config: *const OpenMsRedeemFineTuneConfig,
) -> c_int {
    match std::panic::catch_unwind(AssertUnwindSafe(|| {
        clear_last_error();

        if predictor.is_null() {
            set_last_error("Predictor pointer must not be null");
            return 0;
        }
        if config.is_null() {
            set_last_error("Fine-tuning config pointer must not be null");
            return 0;
        }

        match unsafe { &mut *predictor }.fine_tune_from_transition_tsv(unsafe { &*config }) {
            Ok(()) => 1,
            Err(error) => {
                set_last_error(format!("{error:#}"));
                0
            }
        }
    })) {
        Ok(status) => status,
        Err(_) => {
            set_panic_error();
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn openms_redeem_batch_output_free(output: *mut OpenMsRedeemBatchOutput) {
    let _ = std::panic::catch_unwind(|| {
        if output.is_null() {
            return;
        }

        let output = unsafe { &mut *output };

        if !output.rt_values.is_null() {
            unsafe {
                drop(Box::from_raw(slice::from_raw_parts_mut(
                    output.rt_values,
                    output.count,
                )));
            }
        }
        if !output.ccs_values.is_null() {
            unsafe {
                drop(Box::from_raw(slice::from_raw_parts_mut(
                    output.ccs_values,
                    output.count,
                )));
            }
        }
        if !output.ms2_row_counts.is_null() {
            unsafe {
                drop(Box::from_raw(slice::from_raw_parts_mut(
                    output.ms2_row_counts,
                    output.count,
                )));
            }
        }
        if !output.ms2_values.is_null() {
            unsafe {
                drop(Box::from_raw(slice::from_raw_parts_mut(
                    output.ms2_values,
                    output.ms2_value_count,
                )));
            }
        }

        *output = OpenMsRedeemBatchOutput::default();
    });
}

#[no_mangle]
pub extern "C" fn openms_redeem_last_error() -> *const c_char {
    match LAST_ERROR.lock() {
        Ok(guard) => guard
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null()),
        Err(_) => ptr::null(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redeem_properties::utils::peptdeep_utils::download_pretrained_models_exist;
    use std::ffi::CString;
    use std::fs;
    use std::io::{self, Write};
    use std::sync::OnceLock;
    use zip::ZipArchive;

    static TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
    static EXTRACTED_MODELS_DIR: OnceLock<PathBuf> = OnceLock::new();

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn required_model_relpaths() -> [&'static str; 3] {
        [
            "redeem/20251205_100_epochs_min_max_rt_cnn_tf.safetensors",
            "redeem/20251205_500_epochs_early_stopped_100_min_max_ccs_cnn_tf.safetensors",
            "alphapeptdeep/generic/ms2.pth",
        ]
    }

    fn all_models_exist(root: &Path, required: &[&str]) -> bool {
        required.iter().all(|relative| root.join(relative).exists())
    }

    fn extract_models_from_archive(archive_path: &Path, output_dir: &Path, required: &[&str]) {
        fs::create_dir_all(output_dir).expect("failed to create extracted model directory");
        let archive_file =
            fs::File::open(archive_path).expect("failed to open pretrained_models.zip");
        let mut archive =
            ZipArchive::new(archive_file).expect("failed to read pretrained_models.zip");

        for relative in required {
            let archive_name = format!("pretrained_models/{relative}");
            let mut entry = archive
                .by_name(&archive_name)
                .unwrap_or_else(|_| panic!("missing {archive_name} in pretrained_models.zip"));
            let destination = output_dir.join(relative);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)
                    .unwrap_or_else(|e| panic!("failed to create {}: {e}", parent.display()));
            }
            let mut out = fs::File::create(&destination).unwrap_or_else(|e| {
                panic!(
                    "failed to create extracted model {}: {e}",
                    destination.display()
                )
            });
            io::copy(&mut entry, &mut out).unwrap_or_else(|e| {
                panic!(
                    "failed to extract {} to {}: {e}",
                    archive_name,
                    destination.display()
                )
            });
        }
    }

    fn extract_required_models() -> &'static PathBuf {
        EXTRACTED_MODELS_DIR.get_or_init(|| {
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let archive_path = manifest_dir.join("../redeem-properties/data/pretrained_models.zip");
            let output_dir = std::env::temp_dir().join("redeem-openms-ffi-test-models");
            let required = required_model_relpaths();

            let rt_path = output_dir.join(required[0]);
            let ccs_path = output_dir.join(required[1]);
            let ms2_path = output_dir.join(required[2]);
            if rt_path.exists() && ccs_path.exists() && ms2_path.exists() {
                return output_dir;
            }

            if archive_path.exists() {
                extract_models_from_archive(&archive_path, &output_dir, &required);
                return output_dir;
            }

            let downloaded_dir = download_pretrained_models_exist().unwrap_or_else(|error| {
                panic!(
                    "failed to provision pretrained models for FFI tests; \
                     repo archive missing at {} and download failed: {error}",
                    archive_path.display()
                )
            });

            if all_models_exist(&downloaded_dir, &required) {
                return downloaded_dir;
            }

            panic!(
                "missing required pretrained models for FFI tests under {}",
                downloaded_dir.display()
            );
        })
    }

    fn model_path(relative: &str) -> CString {
        CString::new(
            extract_required_models()
                .join(relative)
                .to_string_lossy()
                .into_owned(),
        )
        .unwrap()
    }

    fn predictor_config(
        with_ccs: bool,
    ) -> (
        OpenMsRedeemPredictorConfig,
        CString,
        Option<CString>,
        CString,
        CString,
    ) {
        let rt = model_path("redeem/20251205_100_epochs_min_max_rt_cnn_tf.safetensors");
        let ccs = with_ccs.then(|| {
            model_path(
                "redeem/20251205_500_epochs_early_stopped_100_min_max_ccs_cnn_tf.safetensors",
            )
        });
        let ms2 = model_path("alphapeptdeep/generic/ms2.pth");
        let device = CString::new("cpu").unwrap();

        let config = OpenMsRedeemPredictorConfig {
            rt_model_path: rt.as_ptr(),
            ccs_model_path: ccs
                .as_ref()
                .map(|value| value.as_ptr())
                .unwrap_or(ptr::null()),
            ms2_model_path: ms2.as_ptr(),
            device_preference: device.as_ptr(),
        };

        (config, rt, ccs, ms2, device)
    }

    fn peptide_input(
        peptide: &str,
        charge: i32,
        nce: i32,
        instrument: Option<&str>,
    ) -> (OpenMsRedeemPredictionInput, CString, Option<CString>) {
        let peptide = CString::new(peptide).unwrap();
        let instrument = instrument.map(|value| CString::new(value).unwrap());
        let input = OpenMsRedeemPredictionInput {
            modified_peptide: peptide.as_ptr(),
            precursor_charge: charge,
            nce,
            instrument: instrument
                .as_ref()
                .map(|value| value.as_ptr())
                .unwrap_or(ptr::null()),
        };
        (input, peptide, instrument)
    }

    fn write_finetune_fixture() -> PathBuf {
        let destination = std::env::temp_dir().join("redeem-openms-ffi-finetune-fixture.tsv");
        let mut writer =
            fs::File::create(&destination).expect("failed to create fine-tuning fixture TSV");

        writeln!(
            writer,
            "sequence\tprecursor_mz\tprecursor_charge\tfragment_type\tfragment_series_number\tproduct_charge\tretention_time\tintensity"
        )
        .expect("failed to write fine-tuning fixture header");

        let rows = [
            (".(UniMod:1)PEPTIDE", 400.6873_f32, 2_i32, 32.5_f32),
            ("MGC(UniMod:4)AAR", 289.4681_f32, 3_i32, 18.2_f32),
            ("PEPC(UniMod:4)PEPR", 478.2192_f32, 2_i32, 41.7_f32),
            ("ACDEFGHIK", 510.2424_f32, 2_i32, 55.3_f32),
            ("VVTADK", 316.6762_f32, 2_i32, 24.1_f32),
            ("LGGNEQVTR", 487.2588_f32, 2_i32, 47.9_f32),
        ];

        for (sequence, precursor_mz, precursor_charge, retention_time) in rows {
            for (fragment_type, series_number, product_charge, intensity) in [
                ("b", 1_usize, 1_i32, 1200.0_f32),
                ("y", 1_usize, 1_i32, 950.0_f32),
                ("b", 2_usize, 1_i32, 700.0_f32),
            ] {
                writeln!(
                    writer,
                    "{sequence}\t{precursor_mz}\t{precursor_charge}\t{fragment_type}\t{series_number}\t{product_charge}\t{retention_time}\t{intensity}"
                )
                .expect("failed to write fine-tuning fixture row");
            }
        }

        destination
    }

    unsafe fn last_error_string() -> String {
        let ptr = openms_redeem_last_error();
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }

    #[test]
    fn create_and_destroy_predictor() {
        let _guard = test_lock();
        let (config, _rt, _ccs, _ms2, _device) = predictor_config(true);

        let predictor = openms_redeem_predictor_create(&config);
        assert!(!predictor.is_null(), "{}", unsafe { last_error_string() });

        openms_redeem_predictor_destroy(predictor);
    }

    #[test]
    fn invalid_charge_sets_last_error() {
        let _guard = test_lock();
        let (config, _rt, _ccs, _ms2, _device) = predictor_config(false);
        let predictor = openms_redeem_predictor_create(&config);
        assert!(!predictor.is_null(), "{}", unsafe { last_error_string() });

        let (input, _peptide, _instrument) = peptide_input("PEPTIDE", 0, 0, None);
        let mut output = OpenMsRedeemBatchOutput::default();

        let status = openms_redeem_predict_batch(predictor, &input, 1, &mut output);
        assert_eq!(status, 0);
        assert!(
            unsafe { last_error_string() }.contains("precursor_charge must be positive"),
            "{}",
            unsafe { last_error_string() }
        );

        openms_redeem_predictor_destroy(predictor);
    }

    #[test]
    fn batch_prediction_returns_expected_shapes() {
        let _guard = test_lock();
        let (config, _rt, _ccs, _ms2, _device) = predictor_config(true);
        let predictor = openms_redeem_predictor_create(&config);
        assert!(!predictor.is_null(), "{}", unsafe { last_error_string() });

        let (input_a, _peptide_a, _instrument_a) = peptide_input("PEPTIDE", 2, 0, None);
        let (input_b, _peptide_b, _instrument_b) =
            peptide_input("MGC[+57.0215]AAR", 3, 27, Some("QE"));
        let inputs = [input_a, input_b];
        let mut output = OpenMsRedeemBatchOutput::default();

        let status =
            openms_redeem_predict_batch(predictor, inputs.as_ptr(), inputs.len(), &mut output);
        assert_eq!(status, 1, "{}", unsafe { last_error_string() });
        assert_eq!(output.count, 2);
        assert_eq!(output.has_ccs, 1);
        assert!(!output.rt_values.is_null());
        assert!(!output.ccs_values.is_null());
        assert!(!output.ms2_row_counts.is_null());
        assert!(!output.ms2_values.is_null());

        let row_counts = unsafe { slice::from_raw_parts(output.ms2_row_counts, output.count) };
        assert_eq!(row_counts.len(), 2);
        assert!(row_counts.iter().all(|count| *count > 0));
        assert_eq!(output.ms2_value_count, row_counts.iter().sum::<usize>() * 8);

        openms_redeem_batch_output_free(&mut output);
        openms_redeem_predictor_destroy(predictor);
    }

    #[test]
    fn ccs_can_be_disabled() {
        let _guard = test_lock();
        let (config, _rt, _ccs, _ms2, _device) = predictor_config(false);
        let predictor = openms_redeem_predictor_create(&config);
        assert!(!predictor.is_null(), "{}", unsafe { last_error_string() });

        let (input, _peptide, _instrument) = peptide_input("PEPTIDE", 2, 0, None);
        let mut output = OpenMsRedeemBatchOutput::default();

        let status = openms_redeem_predict_batch(predictor, &input, 1, &mut output);
        assert_eq!(status, 1, "{}", unsafe { last_error_string() });
        assert_eq!(output.has_ccs, 0);
        assert!(output.ccs_values.is_null());

        openms_redeem_batch_output_free(&mut output);
        openms_redeem_predictor_destroy(predictor);
    }

    #[test]
    fn fine_tune_rt_from_transition_tsv_and_save_model() {
        let _guard = test_lock();
        let (config, _rt, _ccs, _ms2, _device) = predictor_config(true);
        let predictor = openms_redeem_predictor_create(&config);
        assert!(!predictor.is_null(), "{}", unsafe { last_error_string() });

        let training_tsv = write_finetune_fixture();
        let output_model = std::env::temp_dir().join("redeem-openms-ffi-finetuned-rt.safetensors");
        let output_constants = PathBuf::from(format!(
            "{}.model_const.yaml",
            output_model.to_string_lossy()
        ));
        let source_constants = neighboring_constants_path(Path::new(
            _rt.to_str().expect("RT model path must be valid UTF-8"),
        ));
        let training_tsv_c = CString::new(training_tsv.to_string_lossy().into_owned()).unwrap();
        let output_model_c = CString::new(output_model.to_string_lossy().into_owned()).unwrap();
        let instrument_c = CString::new("Lumos").unwrap();

        let fine_tune_config = OpenMsRedeemFineTuneConfig {
            training_tsv_path: training_tsv_c.as_ptr(),
            validation_tsv_path: ptr::null(),
            validation_fraction: 0.2,
            batch_size: 16,
            validation_batch_size: 16,
            epochs: 1,
            early_stopping_patience: 1,
            learning_rate: 1e-4,
            warmup_fraction: 0.0,
            default_nce: 30,
            default_instrument: instrument_c.as_ptr(),
            enable_rt: 1,
            enable_ccs: 0,
            enable_ms2: 0,
            rt_model_output_path: output_model_c.as_ptr(),
            ccs_model_output_path: ptr::null(),
            ms2_model_output_path: ptr::null(),
        };

        let status = openms_redeem_predictor_fine_tune_from_tsv(predictor, &fine_tune_config);
        assert_eq!(status, 1, "{}", unsafe { last_error_string() });
        assert!(output_model.exists(), "fine-tuned RT model was not saved");
        if source_constants.is_some() {
            assert!(
                output_constants.exists(),
                "fine-tuned RT model constants sidecar was not saved"
            );
        }

        let (input, _peptide, _instrument) = peptide_input("PEPTIDE", 2, 0, None);
        let mut output = OpenMsRedeemBatchOutput::default();
        let predict_status = openms_redeem_predict_batch(predictor, &input, 1, &mut output);
        assert_eq!(predict_status, 1, "{}", unsafe { last_error_string() });

        openms_redeem_batch_output_free(&mut output);
        openms_redeem_predictor_destroy(predictor);
    }
}
