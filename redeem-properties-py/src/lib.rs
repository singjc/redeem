use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyRuntimeError;
use redeem_properties::models::{
    ccs_model::CCSModelWrapper,
    model_interface::{ModelInterface, PredictionResult},
    ms2_model::MS2ModelWrapper,
    rt_model::RTModelWrapper,
};
use redeem_properties::pretrained::{locate_pretrained_model, PretrainedModel};
use redeem_properties::utils::mz_utils;
use redeem_properties::utils::peptdeep_utils::{
    ccs_to_mobility_bruker, get_modification_indices, get_modification_string,
    remove_mass_shift, MODIFICATION_MAP,
};
use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use numpy::{PyArray1, PyArray2};

fn strings_to_arcs(strings: Vec<String>) -> Vec<Arc<[u8]>> {
    strings.into_iter().map(|s| Arc::from(s.into_bytes())).collect()
}

fn opt_strings_to_arcs(strings: Vec<Option<String>>) -> Vec<Option<Arc<[u8]>>> {
    strings
        .into_iter()
        .map(|opt| opt.map(|s| Arc::from(s.into_bytes())))
        .collect()
}

/// Decompose a list of (possibly modified) peptides into naked sequences, mod strings,
/// and mod site strings that the Rust model API expects.
///
/// Each peptide may contain inline modification annotations, e.g.:
///   `"SEQU[+42.0106]ENCE"` — mass-shift notation
///   `"SEQUEN(UniMod:4)CE"` — UniMod notation
///
/// Unmodified peptides are handled transparently.
fn parse_peptides(peptides: &[String]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mod_map = &*MODIFICATION_MAP;
    let mut naked_seqs = Vec::with_capacity(peptides.len());
    let mut mods_out = Vec::with_capacity(peptides.len());
    let mut sites_out = Vec::with_capacity(peptides.len());

    for pep in peptides {
        naked_seqs.push(remove_mass_shift(pep));
        mods_out.push(get_modification_string(pep, mod_map));
        sites_out.push(get_modification_indices(pep));
    }

    (naked_seqs, mods_out, sites_out)
}

/// Parse a pretrained model name and validate it belongs to the expected family.
///
/// Returns `(arch_str, model_path)` on success.  The `family` argument is one of
/// `"rt"`, `"ccs"`, or `"ms2"` and is used only for the error message.
fn resolve_pretrained(name: &str, family: &str) -> PyResult<(String, PathBuf)> {
    let pm = PretrainedModel::from_str(name)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let arch = pm.arch();
    if !arch.starts_with(family) && !arch.contains(family) {
        return Err(PyRuntimeError::new_err(format!(
            "Pretrained model '{}' (arch '{}') is not a {} model. \
             Pass the correct name to the matching class.",
            name, arch, family
        )));
    }

    let model_path = locate_pretrained_model(pm)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((arch.to_string(), model_path))
}

/// Locate the absolute path to a pretrained model file on disk.
///
/// This function searches the pretrained model registry for the given model name.
/// It checks the following locations in order:
///   1. The directory specified by the `REDEEM_PRETRAINED_MODELS_DIR` environment variable.
///   2. The user's local data directory (e.g., `~/.local/share/redeem/pretrained_models/` on Linux).
///   3. The `data/pretrained_models/` directory in development locations.
///
/// Args:
///     name (str): The identifier of the pretrained model (e.g., "rt", "ccs", "ms2").
///
/// Returns:
///     str: The absolute path to the located model file.
///
/// Raises:
///     RuntimeError: If the model name is invalid or the file cannot be found.
#[pyfunction]
fn locate_pretrained(name: &str) -> PyResult<String> {
    let pm = PretrainedModel::from_str(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let p = locate_pretrained_model(pm).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(p.to_string_lossy().into_owned())
}

/// Validate and inspect a pretrained model lookup.
///
/// Returns a dict with keys:
/// - `located_path`: concrete path used by the pretrained registry
/// - `extension`: file extension (if any)
/// - `file_type`: one of `safetensors`, `pth`, or `unknown` (best-effort)
/// - `sidecar_exists`: whether a `<model>.<ext>.model_const.yaml` sidecar exists
/// - `sidecar_path`: path to the sidecar or None
/// - `sidecar_preview`: first 2048 characters of the sidecar (if present)
#[pyfunction]
fn validate_pretrained(py: Python, name: &str) -> PyResult<PyObject> {
    let pm = PretrainedModel::from_str(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let model_path = locate_pretrained_model(pm).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let ext = model_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    // Heuristic file type detection: prefer extension, fall back to magic byte for pickle
    let mut file_type = "unknown".to_string();
    if ext.eq_ignore_ascii_case("safetensors") {
        file_type = "safetensors".to_string();
    } else if ext.eq_ignore_ascii_case("pth") || ext.eq_ignore_ascii_case("pt") {
        file_type = "pth".to_string();
    } else {
        // Try to read first byte to detect pickle (.pth/.pt often start with 0x80)
        if let Ok(mut f) = std::fs::File::open(&model_path) {
            let mut buf = [0u8; 1];
            if let Ok(n) = f.read(&mut buf) {
                if n > 0 && buf[0] == 0x80 {
                    file_type = "pth".to_string();
                }
            }
        }
    }

    // Sidecar YAML candidate (e.g., model.safetensors.model_const.yaml)
    let sidecar_candidate: Option<std::path::PathBuf> = model_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| model_path.with_extension(format!("{}.model_const.yaml", ext)))
        .and_then(|cand| if cand.exists() { Some(cand) } else { None });

    let (sidecar_exists, sidecar_path, sidecar_preview) = match sidecar_candidate.as_ref() {
        Some(p) => {
            let preview = std::fs::read_to_string(p)
                .map(|s| s.chars().take(2048).collect::<String>())
                .unwrap_or_else(|_| "<failed to read sidecar>".to_string());
            (true, Some(p.to_string_lossy().into_owned()), Some(preview))
        }
        None => (false, None, None),
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("located_path", model_path.to_string_lossy().into_owned())?;
    dict.set_item("extension", ext)?;
    dict.set_item("file_type", file_type)?;
    dict.set_item("sidecar_exists", sidecar_exists)?;
    dict.set_item("sidecar_path", sidecar_path)?;
    dict.set_item("sidecar_preview", sidecar_preview)?;

    Ok(dict.into())
}

/// Download pretrained models from the GitHub release and extract them locally.
///
/// This fetches the pretrained model archive from the redeem GitHub releases page,
/// validates the download, and extracts the model files so they can be used by
/// ``RTModel.from_pretrained()``, ``CCSModel.from_pretrained()``, and
/// ``MS2Model.from_pretrained()``.
///
/// The models are downloaded to a stable user-local directory (or
/// ``REDEEM_PRETRAINED_MODELS_DIR`` when set), so they are reusable across working
/// directories. If the models already exist, this function returns immediately without
/// re-downloading.
///
/// Returns:
///     str: The absolute path to the extracted pretrained models directory.
///
/// Raises:
///     RuntimeError: If the download fails, the archive is corrupted, or extraction fails.
///
/// Example:
///     >>> import redeem_properties as rp
///     >>> models_dir = rp.download_pretrained_models()
///     >>> print(f"Models available at: {models_dir}")
///     >>> rt_model = rp.RTModel.from_pretrained("rt")
#[pyfunction]
fn download_pretrained_models() -> PyResult<String> {
    use redeem_properties::utils::peptdeep_utils::download_pretrained_models_exist;

    let path = download_pretrained_models_exist()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to download pretrained models: {}", e)))?;

    // Return the absolute path as a string
    let abs_path = std::fs::canonicalize(&path)
        .unwrap_or(path);

    Ok(abs_path.to_string_lossy().into_owned())
}

/// Python wrapper for the RT (retention time) prediction model.
#[pyclass]
struct RTModel {
    inner: RTModelWrapper,
}

#[pymethods]
impl RTModel {
    /// Create a new RTModel.
    ///
    /// Args:
    ///     model_path (str): Path to the model file.
    ///     arch (str): Model architecture (e.g. "rt_cnn_lstm" or "rt_cnn_tf").
    ///     constants_path (str, optional): Path to the constants/config file.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path=None, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: Option<String>,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = get_device(use_cuda)?;
        let wrapper = RTModelWrapper::new(
            std::path::Path::new(&model_path),
            constants_path.as_deref().map(std::path::Path::new),
            &arch,
            device,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Load an RTModel from the shipped pretrained weights.
    ///
    /// Models are located using the ``redeem_properties`` pretrained-model registry.
    /// On first use the search order is:
    ///   1. ``$REDEEM_PRETRAINED_MODELS_DIR/<name>``
    ///   2. user-local pretrained directory (e.g. ``~/.local/share/redeem/pretrained_models``)
    ///   3. ``data/pretrained_models/`` development locations
    ///
    /// Accepted ``name`` values (case-insensitive):
    ///   - ``"rt"`` / ``"alphapeptdeep-rt"`` / ``"alphapeptdeep-rt-cnn-lstm"``
    ///   - ``"redeem-rt"`` / ``"redeem-rt-cnn-tf"``
    ///
    /// Args:
    ///     name (str): Pretrained model identifier.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    ///
    /// Returns:
    ///     RTModel: A ready-to-use model loaded with pretrained weights.
    #[staticmethod]
    #[pyo3(signature = (name, use_cuda=false))]
    fn from_pretrained(name: String, use_cuda: bool) -> PyResult<Self> {
        let (arch, model_path) = resolve_pretrained(&name, "rt")?;
        let device = get_device(use_cuda)?;
        let wrapper = {
            let call = || RTModelWrapper::new(&model_path, None::<std::path::PathBuf>.as_ref(), &arch, device);
            match std::panic::catch_unwind(call) {
                Ok(Ok(w)) => w,
                Ok(Err(e)) => return Err(PyRuntimeError::new_err(e.to_string())),
                Err(payload) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "panic while loading pretrained RT model: {:?}", payload
                    )))
                }
            }
        };
        Ok(Self { inner: wrapper })
    }

    /// Predict retention times for a list of peptides.
    ///
    /// Peptides may contain inline modification annotations, which are parsed
    /// automatically — no need to supply separate mod or mod_site strings:
    ///
    /// .. code-block:: python
    ///
    ///     rt_values = model.predict([
    ///         "PEPTIDE",
    ///         "SEQU[+42.0106]ENCE",
    ///         "SEQUEN(UniMod:4)CE",
    ///     ])
    ///
    /// Args:
    ///     peptides (list[str]): Peptide sequences, optionally containing inline
    ///         modification annotations (mass-shift ``[+X.X]`` or UniMod
    ///         ``(UniMod:N)`` notation).
    ///
    /// Returns:
    ///     numpy.ndarray: 1-D array of predicted RT values (f32).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        peptides: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let (naked, mods, sites) = parse_peptides(&peptides);
        let seqs = strings_to_arcs(naked);
        let mods_arc = strings_to_arcs(mods);
        let sites_arc = strings_to_arcs(sites);

        let result = self
            .inner
            .predict(&seqs, &mods_arc, &sites_arc)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match result {
            PredictionResult::RTResult(values) => {
                Ok(PyArray1::from_slice_bound(py, &values))
            }
            _ => Err(PyRuntimeError::new_err("Unexpected prediction result type")),
        }
    }

    /// Return the total number of parameters in the loaded model.
    #[pyo3(name = "param_count")]
    fn param_count_py(&mut self) -> PyResult<usize> {
        Ok(self.inner.param_count())
    }

    /// Return a pretty hierarchical model summary (preferred) or fall back to
    /// the detailed tabular summary.
    fn summary(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_pretty())
    }

    /// Return the full detailed summary (groups and top tensors).
    #[pyo3(name = "summary_detailed")]
    fn summary_detailed_py(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_detailed())
    }

    /// Expose pretty hierarchical summary to Python.
    #[pyo3(name = "summary_pretty")]
    fn summary_pretty_py(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_pretty())
    }
}

/// Python wrapper for the CCS (collision cross section) prediction model.
#[pyclass]
struct CCSModel {
    inner: CCSModelWrapper,
}

#[pymethods]
impl CCSModel {
    /// Create a new CCSModel.
    ///
    /// Args:
    ///     model_path (str): Path to the model file.
    ///     arch (str): Model architecture (e.g. "ccs_cnn_lstm" or "ccs_cnn_tf").
    ///     constants_path (str): Path to the constants/config file.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: Option<String>,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = get_device(use_cuda)?;
        let model_path_arg = std::path::Path::new(&model_path);
        let cpath_buf_opt: Option<std::path::PathBuf> = constants_path.map(|s| std::path::PathBuf::from(s));
        let cpath_arg_opt: Option<&std::path::Path> = cpath_buf_opt.as_ref().map(|p| p.as_path());

        let wrapper = CCSModelWrapper::new(
            model_path_arg,
            cpath_arg_opt,
            &arch,
            device,
        )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Load a CCSModel from the shipped pretrained weights.
    ///
    /// Models are located using the ``redeem_properties`` pretrained-model registry.
    ///
    /// Accepted ``name`` values (case-insensitive):
    ///   - ``"ccs"`` / ``"alphapeptdeep-ccs"`` / ``"alphapeptdeep-ccs-cnn-lstm"``
    ///   - ``"redeem-ccs"`` / ``"redeem-ccs-cnn-tf"``
    ///
    /// Args:
    ///     name (str): Pretrained model identifier.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    ///
    /// Returns:
    ///     CCSModel: A ready-to-use model loaded with pretrained weights.
    #[staticmethod]
    #[pyo3(signature = (name, use_cuda=false))]
    fn from_pretrained(name: String, use_cuda: bool) -> PyResult<Self> {
        let (arch, model_path) = resolve_pretrained(&name, "ccs")?;
        let device = get_device(use_cuda)?;
        // Try to find a sidecar constants YAML next to the model file instead of
        // blindly parsing the model file itself (which is binary for safetensors).
        let constants_candidate: Option<std::path::PathBuf> = model_path
            .extension()
            .and_then(|s| s.to_str())
            .map(|ext| model_path.with_extension(format!("{}.model_const.yaml", ext)))
            .and_then(|cand| if cand.exists() { Some(cand) } else { None });

        // Build the wrapper while catching panics so we return a Python error
        let wrapper = {
            let call = || {
                match constants_candidate.as_ref() {
                    Some(cpath) => CCSModelWrapper::new(&model_path, Some(cpath), &arch, device),
                    None => CCSModelWrapper::new(&model_path, None::<&std::path::PathBuf>, &arch, device),
                }
            };

            match std::panic::catch_unwind(call) {
                Ok(Ok(w)) => w,
                Ok(Err(e)) => return Err(PyRuntimeError::new_err(e.to_string())),
                Err(payload) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "panic while loading pretrained CCS model: {:?}", payload
                    )))
                }
            }
        };
        Ok(Self { inner: wrapper })
    }

    /// Predict CCS values for a list of peptides.
    ///
    /// Peptides may contain inline modification annotations, which are parsed
    /// automatically — no need to supply separate mod or mod_site strings.
    ///
    /// .. code-block:: python
    ///
    ///     results = model.predict(
    ///         ["PEPTIDE", "SEQU[+42.0106]ENCE"],
    ///         charges=[2, 3],
    ///     )
    ///     for res in results:
    ///         print(res["ccs"])     # predicted CCS value (Å²)
    ///         print(res["charge"])  # charge state used for this prediction
    ///
    /// Args:
    ///     peptides (list[str]): Peptide sequences, optionally containing inline
    ///         modification annotations (mass-shift ``[+X.X]`` or UniMod
    ///         ``(UniMod:N)`` notation).
    ///     charges (list[int]): Charge states per peptide.
    ///
    /// Returns:
    ///     list[dict]: One dict per peptide, each containing:
    ///
    ///         * ``"ccs"`` (*float*): predicted CCS value in Å².
    ///         * ``"charge"`` (*int*): charge state used for this prediction.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        peptides: Vec<String>,
        charges: Vec<i32>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let (naked, mods, sites) = parse_peptides(&peptides);
        let seqs = strings_to_arcs(naked);
        let mods_arc = strings_to_arcs(mods);
        let sites_arc = strings_to_arcs(sites);

        let result = self
            .inner
            .predict(&seqs, &mods_arc, &sites_arc, charges.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match result {
            PredictionResult::CCSResult(values) => {
                if values.len() != charges.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "CCS prediction returned {} values but {} charges were provided",
                        values.len(),
                        charges.len()
                    )));
                }
                let mut out = Vec::with_capacity(values.len());
                for (ccs, charge) in values.into_iter().zip(charges.into_iter()) {
                    let d = PyDict::new_bound(py);
                    d.set_item("ccs", ccs)?;
                    d.set_item("charge", charge)?;
                    out.push(d);
                }
                Ok(out)
            }
            _ => Err(PyRuntimeError::new_err("Unexpected prediction result type")),
        }
    }

    /// Return the total number of parameters in the loaded model.
    #[pyo3(name = "param_count")]
    fn param_count_py(&mut self) -> PyResult<usize> {
        Ok(self.inner.param_count())
    }

    /// Return a pretty hierarchical model summary (preferred) or fall back to
    /// the detailed tabular summary.
    fn summary(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_pretty())
    }

    /// Expose detailed summary to Python.
    #[pyo3(name = "summary_detailed")]
    fn summary_detailed_py(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_detailed())
    }

    #[pyo3(name = "summary_pretty")]
    fn summary_pretty_py(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_pretty())
    }
}

/// Python wrapper for the MS2 fragment intensity prediction model.
#[pyclass]
struct MS2Model {
    inner: MS2ModelWrapper,
}

#[pymethods]
impl MS2Model {
    /// Create a new MS2Model.
    ///
    /// Args:
    ///     model_path (str): Path to the model file.
    ///     arch (str): Model architecture (e.g. "ms2_bert").
    ///     constants_path (str): Path to the constants/config file.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: Option<String>,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = get_device(use_cuda)?;
        let model_path_arg = std::path::Path::new(&model_path);
        let cpath_buf_opt: Option<std::path::PathBuf> = constants_path.map(|s| std::path::PathBuf::from(s));
        let cpath_arg_opt: Option<&std::path::Path> = cpath_buf_opt.as_ref().map(|p| p.as_path());

        let wrapper = MS2ModelWrapper::new(
            model_path_arg,
            cpath_arg_opt,
            &arch,
            device,
        )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Load an MS2Model from the shipped pretrained weights.
    ///
    /// Models are located using the ``redeem_properties`` pretrained-model registry.
    ///
    /// Accepted ``name`` values (case-insensitive):
    ///   - ``"ms2"`` / ``"alphapeptdeep-ms2"`` / ``"alphapeptdeep-ms2-bert"``
    ///
    /// Args:
    ///     name (str): Pretrained model identifier.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    ///
    /// Returns:
    ///     MS2Model: A ready-to-use model loaded with pretrained weights.
    #[staticmethod]
    #[pyo3(signature = (name, use_cuda=false))]
    fn from_pretrained(name: String, use_cuda: bool) -> PyResult<Self> {
        let (arch, model_path) = resolve_pretrained(&name, "ms2")?;
        let device = get_device(use_cuda)?;
        // Try to find a sidecar constants YAML next to the model file instead of
        // parsing the model file itself (binary for safetensors).
        let constants_candidate: Option<std::path::PathBuf> = model_path
            .extension()
            .and_then(|s| s.to_str())
            .map(|ext| model_path.with_extension(format!("{}.model_const.yaml", ext)))
            .and_then(|cand| if cand.exists() { Some(cand) } else { None });

        let wrapper = {
            let call = || {
                match constants_candidate.as_ref() {
                    Some(cpath) => MS2ModelWrapper::new(&model_path, Some(cpath), &arch, device),
                    None => MS2ModelWrapper::new(&model_path, None::<&std::path::PathBuf>, &arch, device),
                }
            };

            match std::panic::catch_unwind(call) {
                Ok(Ok(w)) => w,
                Ok(Err(e)) => return Err(PyRuntimeError::new_err(e.to_string())),
                Err(payload) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "panic while loading pretrained MS2 model: {:?}", payload
                    )))
                }
            }
        };
        Ok(Self { inner: wrapper })
    }

    /// Predict MS2 fragment intensities for a list of peptides.
    ///
    /// Peptides may contain inline modification annotations, which are parsed
    /// automatically — no need to supply separate mod or mod_site strings.
    ///
    /// .. code-block:: python
    ///
    ///     results = model.predict(
    ///         ["PEPTIDE", "SEQU[+42.0106]ENCE"],
    ///         charges=[2, 2],
    ///         nces=[20, 20],
    ///         instruments=["QE", "QE"],
    ///     )
    ///     for res in results:
    ///         print(res["intensities"].shape)    # (n_positions, 8)
    ///         print(res["ion_types"])            # ["b", "b", "y", "y", ...]
    ///         print(res["ion_charges"])          # [1, 2, 1, 2, ...]
    ///         print(res["b_ordinals"])           # [1, 2, ..., n_positions]
    ///         print(res["y_ordinals"])           # [n_positions, ..., 1]
    ///
    ///     # Easy DataFrame creation:
    ///     import pandas as pd, numpy as np
    ///     res = results[0]
    ///     n_pos, n_types = res["intensities"].shape
    ///     # Build per-row ordinals respecting ion type (b and y have reversed numbering)
    ///     b_types = {"b", "b_nl"}
    ///     ordinals = np.array([
    ///         res["b_ordinals"][r] if t in b_types else res["y_ordinals"][r]
    ///         for r in range(n_pos) for t in res["ion_types"]
    ///     ])
    ///     df = pd.DataFrame({
    ///         "ion_type":  np.tile(res["ion_types"], n_pos),
    ///         "charge":    np.tile(res["ion_charges"], n_pos),
    ///         "ordinal":   ordinals,
    ///         "intensity": res["intensities"].ravel(),
    ///     })
    ///
    /// The 8 columns per position correspond to, in order:
    ///   ``b+1``, ``b+2``, ``y+1``, ``y+2``,
    ///   ``b_nl+1``, ``b_nl+2``, ``y_nl+1``, ``y_nl+2``
    ///   where ``nl`` denotes neutral-loss variants (zeros when not applicable).
    ///
    /// Args:
    ///     peptides (list[str]): Peptide sequences, optionally containing inline
    ///         modification annotations (mass-shift ``[+X.X]`` or UniMod
    ///         ``(UniMod:N)`` notation).
    ///     charges (list[int]): Charge states per peptide.
    ///     nces (list[int]): Normalized collision energies per peptide.
    ///     instruments (list[str | None], optional): Instrument names per peptide.
    ///
    /// Returns:
    ///     list[dict]: One dict per peptide, each containing:
    ///
    ///         * ``"intensities"`` (*numpy.ndarray* shape ``(n_positions, 8)``): predicted
    ///           fragment intensities.
    ///         * ``"ion_types"`` (*list[str]* len 8): ion type per column —
    ///           ``"b"``, ``"b"``, ``"y"``, ``"y"``, ``"b_nl"``, ``"b_nl"``, ``"y_nl"``, ``"y_nl"``.
    ///         * ``"ion_charges"`` (*list[int]* len 8): fragment ion charge per column —
    ///           ``1``, ``2``, ``1``, ``2``, ``1``, ``2``, ``1``, ``2``.
    ///         * ``"b_ordinals"`` (*numpy.ndarray* len ``n_positions``): 1-indexed b-ion
    ///           ordinals for each row — ``[1, 2, …, n_positions]``.
    ///         * ``"y_ordinals"`` (*numpy.ndarray* len ``n_positions``): 1-indexed y-ion
    ///           ordinals for each row — ``[n_positions, …, 1]``.
    #[pyo3(signature = (peptides, charges, nces, instruments=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        peptides: Vec<String>,
        charges: Vec<i32>,
        nces: Vec<i32>,
        instruments: Option<Vec<Option<String>>>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let (naked, mods, sites) = parse_peptides(&peptides);
        let n = naked.len();
        let seqs = strings_to_arcs(naked);
        let mods_arc = strings_to_arcs(mods);
        let sites_arc = strings_to_arcs(sites);
        let instr_vec = match instruments {
            Some(v) => opt_strings_to_arcs(v),
            None => vec![None; n],
        };

        let result = self
            .inner
            .predict(&seqs, &mods_arc, &sites_arc, charges, nces, instr_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Fixed column metadata for all 8 fragment type columns
        // Order: b+1, b+2, y+1, y+2, b_nl+1, b_nl+2, y_nl+1, y_nl+2
        let ion_types: Vec<&str> = vec!["b", "b", "y", "y", "b_nl", "b_nl", "y_nl", "y_nl"];
        let ion_charges: Vec<i32> = vec![1, 2, 1, 2, 1, 2, 1, 2];

        match result {
            PredictionResult::MS2Result(matrices) => {
                let mut out = Vec::with_capacity(matrices.len());
                for matrix in matrices {
                    let n_pos = matrix.len();

                    // b ordinals: 1, 2, ..., n_pos
                    let b_ords: Vec<i32> = (1..=(n_pos as i32)).collect();
                    // y ordinals: n_pos, n_pos-1, ..., 1
                    let y_ords: Vec<i32> = (1..=(n_pos as i32)).rev().collect();

                    let d = PyDict::new_bound(py);
                    let intensities = PyArray2::from_vec2_bound(py, &matrix)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create intensities array: {}", e)))?;
                    d.set_item("intensities", intensities)?;
                    d.set_item("ion_types", ion_types.to_vec())?;
                    d.set_item("ion_charges", ion_charges.to_vec())?;
                    d.set_item("b_ordinals", PyArray1::from_slice_bound(py, &b_ords))?;
                    d.set_item("y_ordinals", PyArray1::from_slice_bound(py, &y_ords))?;
                    out.push(d);
                }
                Ok(out)
            }
            _ => Err(PyRuntimeError::new_err("Unexpected prediction result type")),
        }
    }

    /// Return the total number of parameters in the loaded model.
    #[pyo3(name = "param_count")]
    fn param_count_py(&mut self) -> PyResult<usize> {
        Ok(self.inner.param_count())
    }

    /// Return a pretty hierarchical model summary (preferred) or fall back to
    /// the detailed tabular summary.
    fn summary(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_pretty())
    }

    /// Expose detailed summary to Python.
    #[pyo3(name = "summary_detailed")]
    fn summary_detailed_py(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_detailed())
    }

    #[pyo3(name = "summary_pretty")]
    fn summary_pretty_py(&mut self) -> PyResult<String> {
        Ok(self.inner.summary_pretty())
    }
}

#[cfg(feature = "cuda")]
fn get_device(use_cuda: bool) -> PyResult<candle_core::Device> {
    if use_cuda {
        candle_core::Device::new_cuda(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

#[cfg(not(feature = "cuda"))]
fn get_device(use_cuda: bool) -> PyResult<candle_core::Device> {
    if use_cuda {
        Err(PyRuntimeError::new_err(
            "CUDA support is not compiled in. Rebuild with --features cuda.",
        ))
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

/// Compute the precursor m/z for a peptide in ProForma notation.
///
/// Parameters
/// ----------
/// proforma_sequence : str
///     Peptide sequence in ProForma notation (e.g., "PEPTM[+15.9949]IDE").
/// charge : int
///     Precursor charge state (must be > 0).
///
/// Returns
/// -------
/// float
///     The monoisotopic precursor m/z value.
#[pyfunction]
fn compute_precursor_mz(proforma_sequence: &str, charge: i32) -> PyResult<f64> {
    mz_utils::compute_precursor_mz(proforma_sequence, charge)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Compute theoretical product (fragment) ion m/z values for a peptide.
///
/// Generates b and y ion m/z values (without neutral losses) up to the
/// specified maximum fragment charge.
///
/// Parameters
/// ----------
/// proforma_sequence : str
///     Peptide sequence in ProForma notation.
/// max_fragment_charge : int
///     Maximum fragment ion charge state to generate.
///
/// Returns
/// -------
/// dict
///     A dictionary with keys:
///     - "ion_types": list of str ("b" or "y")
///     - "charges": list of int
///     - "ordinals": list of int (1-based series number)
///     - "mzs": list of float (monoisotopic m/z values)
#[pyfunction]
fn compute_fragment_mzs<'py>(
    py: Python<'py>,
    proforma_sequence: &str,
    max_fragment_charge: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let product_ions = mz_utils::compute_product_mzs(proforma_sequence, max_fragment_charge)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let ion_types: Vec<String> = product_ions.iter().map(|i| i.ion_type.clone()).collect();
    let charges: Vec<i32> = product_ions.iter().map(|i| i.charge).collect();
    let ordinals: Vec<usize> = product_ions.iter().map(|i| i.ordinal).collect();
    let mzs: Vec<f64> = product_ions.iter().map(|i| i.mz).collect();

    let dict = PyDict::new_bound(py);
    dict.set_item("ion_types", ion_types)?;
    dict.set_item("charges", charges)?;
    dict.set_item("ordinals", ordinals)?;
    dict.set_item("mzs", mzs)?;
    Ok(dict)
}

/// Compute both precursor and fragment m/z values for a peptide.
///
/// Parameters
/// ----------
/// proforma_sequence : str
///     Peptide sequence in ProForma notation.
/// charge : int
///     Precursor charge state.
/// max_fragment_charge : int
///     Maximum fragment ion charge state to generate.
///
/// Returns
/// -------
/// dict
///     A dictionary with keys:
///     - "precursor_mz": float
///     - "ion_types": list of str
///     - "charges": list of int
///     - "ordinals": list of int
///     - "mzs": list of float
#[pyfunction]
fn compute_peptide_mz_info<'py>(
    py: Python<'py>,
    proforma_sequence: &str,
    charge: i32,
    max_fragment_charge: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let info = mz_utils::compute_peptide_mz_info(proforma_sequence, charge, max_fragment_charge)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let ion_types: Vec<String> = info.product_ions.iter().map(|i| i.ion_type.clone()).collect();
    let charges: Vec<i32> = info.product_ions.iter().map(|i| i.charge).collect();
    let ordinals: Vec<usize> = info.product_ions.iter().map(|i| i.ordinal).collect();
    let mzs: Vec<f64> = info.product_ions.iter().map(|i| i.mz).collect();

    let dict = PyDict::new_bound(py);
    dict.set_item("precursor_mz", info.precursor_mz)?;
    dict.set_item("ion_types", ion_types)?;
    dict.set_item("charges", charges)?;
    dict.set_item("ordinals", ordinals)?;
    dict.set_item("mzs", mzs)?;
    Ok(dict)
}

/// Match theoretical product ion m/z values to predicted fragment annotations.
///
/// Given the predicted ion types, charges, and ordinals from the MS2 model,
/// look up the corresponding theoretical m/z for each.
///
/// Parameters
/// ----------
/// proforma_sequence : str
///     Peptide sequence in ProForma notation.
/// max_fragment_charge : int
///     Maximum fragment charge used to generate theoretical fragments.
/// predicted_ion_types : list of str
///     Ion types from MS2 prediction (e.g., ["b", "y", "b", "y"]).
/// predicted_charges : list of int
///     Charges from MS2 prediction.
/// predicted_ordinals : list of int
///     Ordinals (series numbers) from MS2 prediction.
///
/// Returns
/// -------
/// list of float
///     m/z values aligned with the predicted arrays. NaN if no match found.
#[pyfunction]
fn match_fragment_mzs(
    proforma_sequence: &str,
    max_fragment_charge: i32,
    predicted_ion_types: Vec<String>,
    predicted_charges: Vec<i32>,
    predicted_ordinals: Vec<usize>,
) -> PyResult<Vec<f64>> {
    let product_ions = mz_utils::compute_product_mzs(proforma_sequence, max_fragment_charge)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(mz_utils::match_product_mzs(
        &product_ions,
        &predicted_ion_types,
        &predicted_charges,
        &predicted_ordinals,
    ))
}

/// Convert CCS to ion mobility for Bruker (timsTOF) instruments.
#[pyfunction]
fn ccs_to_mobility(ccs_value: f64, charge: f64, precursor_mz: f64) -> f64 {
    ccs_to_mobility_bruker(ccs_value, charge, precursor_mz)
}

/// Python bindings for the redeem-properties peptide property prediction models.
#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RTModel>()?;
    m.add_class::<CCSModel>()?;
    m.add_class::<MS2Model>()?;
    m.add_function(wrap_pyfunction!(locate_pretrained, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pretrained, m)?)?;
    m.add_function(wrap_pyfunction!(download_pretrained_models, m)?)?;
    m.add_function(wrap_pyfunction!(compute_precursor_mz, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fragment_mzs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_peptide_mz_info, m)?)?;
    m.add_function(wrap_pyfunction!(match_fragment_mzs, m)?)?;
    m.add_function(wrap_pyfunction!(ccs_to_mobility, m)?)?;
    Ok(())
}
