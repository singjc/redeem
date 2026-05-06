use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use once_cell::sync::Lazy;
use redeem_properties::models::{
    ccs_model::CCSModelWrapper, model_interface::PredictionResult, ms2_model::MS2ModelWrapper,
    rt_model::RTModelWrapper,
};
use redeem_properties::utils::peptdeep_utils::{
    extract_mod_annotations, get_modification_indices, get_modification_string, remove_mass_shift,
    MODIFICATION_MAP,
};

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

pub struct Predictor {
    rt_model: RTModelWrapper,
    ccs_model: Option<CCSModelWrapper>,
    ms2_model: MS2ModelWrapper,
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

        let ccs_model = match ccs_model_path {
            Some(path) => Some(
                CCSModelWrapper::new(
                    &path,
                    neighboring_constants_path(&path).as_ref(),
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
    use std::ffi::CString;
    use std::fs;
    use std::io;
    use std::sync::OnceLock;
    use zip::ZipArchive;

    static TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
    static EXTRACTED_MODELS_DIR: OnceLock<PathBuf> = OnceLock::new();

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn extract_required_models() -> &'static PathBuf {
        EXTRACTED_MODELS_DIR.get_or_init(|| {
            let archive_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../redeem-properties/data/pretrained_models.zip");
            let output_dir = std::env::temp_dir().join("redeem-openms-ffi-test-models");
            let required = [
                "redeem/20251205_100_epochs_min_max_rt_cnn_tf.safetensors",
                "redeem/20251205_500_epochs_early_stopped_100_min_max_ccs_cnn_tf.safetensors",
                "alphapeptdeep/generic/ms2.pth",
            ];

            let rt_path = output_dir.join(required[0]);
            let ccs_path = output_dir.join(required[1]);
            let ms2_path = output_dir.join(required[2]);
            if rt_path.exists() && ccs_path.exists() && ms2_path.exists() {
                return output_dir;
            }

            fs::create_dir_all(&output_dir).expect("failed to create extracted model directory");
            let archive_file =
                fs::File::open(&archive_path).expect("failed to open pretrained_models.zip");
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
                    panic!("failed to create extracted model {}: {e}", destination.display())
                });
                io::copy(&mut entry, &mut out).unwrap_or_else(|e| {
                    panic!(
                        "failed to extract {} to {}: {e}",
                        archive_name,
                        destination.display()
                    )
                });
            }

            output_dir
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
}
