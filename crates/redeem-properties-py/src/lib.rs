use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, PyArray2};
use redeem_properties::models::{
    rt_model::RTModelWrapper,
    ccs_model::CCSModelWrapper,
    ms2_model::MS2ModelWrapper,
    model_interface::PredictionResult,
};
use candle_core::Device;
use std::sync::Arc;

/// Convert Python runtime errors from anyhow::Error
fn to_py_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{}", err))
}

/// Python wrapper for the RT (Retention Time) prediction model.
/// 
/// This class provides an interface to predict peptide retention times using
/// pre-trained deep learning models.
/// 
/// # Examples
/// 
/// ```python
/// from redeem_properties_py import RTModel
/// 
/// # Create a model instance
/// model = RTModel("path/to/model.safetensors", "rt_cnn_lstm")
/// 
/// # Predict retention times
/// sequences = ["PEPTIDE", "SEQUENCE"]
/// mods = ["", ""]
/// mod_sites = ["", ""]
/// rt_predictions = model.predict(sequences, mods, mod_sites)
/// ```
#[pyclass]
struct RTModel {
    inner: RTModelWrapper,
}

#[pymethods]
impl RTModel {
    /// Create a new RT prediction model.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file (.safetensors or .pth)
    /// * `arch` - Model architecture ("rt_cnn_lstm" or "rt_cnn_tf")
    /// * `constants_path` - Optional path to the constants file (.yaml)
    /// * `use_cuda` - Whether to use CUDA for inference (default: False)
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path=None, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: Option<String>,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = if use_cuda {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(to_py_err)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(PyRuntimeError::new_err("CUDA support not enabled. Rebuild with --features cuda"));
            }
        } else {
            Device::Cpu
        };

        let inner = RTModelWrapper::new(
            model_path,
            constants_path,
            &arch,
            device,
        ).map_err(to_py_err)?;

        Ok(RTModel { inner })
    }

    /// Predict retention times for a batch of peptides.
    ///
    /// # Arguments
    /// * `sequences` - List of peptide sequences (strings)
    /// * `mods` - List of modification strings (e.g., "Oxidation@M", "Acetyl@^")
    /// * `mod_sites` - List of modification site strings
    ///
    /// # Returns
    /// A numpy array of predicted retention times
    fn predict<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<String>,
        mods: Vec<String>,
        mod_sites: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Convert strings to Arc<[u8]>
        let sequences_arc: Vec<Arc<[u8]>> = sequences
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        
        let mods_arc: Vec<Arc<[u8]>> = mods
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        
        let mod_sites_arc: Vec<Arc<[u8]>> = mod_sites
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();

        // Perform prediction
        let result = self.inner
            .predict(&sequences_arc, &mods_arc, &mod_sites_arc)
            .map_err(to_py_err)?;

        // Extract RT values
        let rt_values = match result {
            PredictionResult::RTResult(values) => values,
            _ => return Err(PyRuntimeError::new_err("Expected RT prediction result")),
        };

        // Convert to numpy array
        Ok(PyArray1::from_slice_bound(py, &rt_values))
    }
}

/// Python wrapper for the CCS (Collision Cross-Section) prediction model.
/// 
/// This class provides an interface to predict peptide collision cross-sections using
/// pre-trained deep learning models.
/// 
/// # Examples
/// 
/// ```python
/// from redeem_properties_py import CCSModel
/// 
/// # Create a model instance
/// model = CCSModel("path/to/model.safetensors", "ccs_cnn_lstm")
/// 
/// # Predict collision cross-sections
/// sequences = ["PEPTIDE", "SEQUENCE"]
/// mods = ["", ""]
/// mod_sites = ["", ""]
/// charges = [2, 3]
/// ccs_predictions = model.predict(sequences, mods, mod_sites, charges)
/// ```
#[pyclass]
struct CCSModel {
    inner: CCSModelWrapper,
}

#[pymethods]
impl CCSModel {
    /// Create a new CCS prediction model.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file (.safetensors or .pth)
    /// * `arch` - Model architecture ("ccs_cnn_lstm" or "ccs_cnn_tf")
    /// * `constants_path` - Path to the constants file (.yaml)
    /// * `use_cuda` - Whether to use CUDA for inference (default: False)
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: String,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = if use_cuda {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(to_py_err)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(PyRuntimeError::new_err("CUDA support not enabled. Rebuild with --features cuda"));
            }
        } else {
            Device::Cpu
        };

        let inner = CCSModelWrapper::new(
            model_path,
            constants_path,
            &arch,
            device,
        ).map_err(to_py_err)?;

        Ok(CCSModel { inner })
    }

    /// Predict collision cross-sections for a batch of peptides.
    ///
    /// # Arguments
    /// * `sequences` - List of peptide sequences (strings)
    /// * `mods` - List of modification strings
    /// * `mod_sites` - List of modification site strings
    /// * `charges` - List of charge states (integers)
    ///
    /// # Returns
    /// A numpy array of predicted collision cross-sections
    fn predict<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<String>,
        mods: Vec<String>,
        mod_sites: Vec<String>,
        charges: Vec<i32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Convert strings to Arc<[u8]>
        let sequences_arc: Vec<Arc<[u8]>> = sequences
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        
        let mods_arc: Vec<Arc<[u8]>> = mods
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        
        let mod_sites_arc: Vec<Arc<[u8]>> = mod_sites
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();

        // Perform prediction
        let result = self.inner
            .predict(&sequences_arc, &mods_arc, &mod_sites_arc, charges)
            .map_err(to_py_err)?;

        // Extract CCS values
        let ccs_values = match result {
            PredictionResult::CCSResult(values) => values,
            _ => return Err(PyRuntimeError::new_err("Expected CCS prediction result")),
        };

        // Convert to numpy array
        Ok(PyArray1::from_slice_bound(py, &ccs_values))
    }
}

/// Python wrapper for the MS2 (Fragment Intensity) prediction model.
/// 
/// This class provides an interface to predict peptide MS2 fragment intensities using
/// pre-trained deep learning models.
/// 
/// # Examples
/// 
/// ```python
/// from redeem_properties_py import MS2Model
/// 
/// # Create a model instance
/// model = MS2Model("path/to/model.safetensors", "ms2_bert")
/// 
/// # Predict MS2 intensities
/// sequences = ["PEPTIDE"]
/// mods = [""]
/// mod_sites = [""]
/// charges = [2]
/// nces = [30.0]
/// intensities = model.predict(sequences, mods, mod_sites, charges, nces)
/// ```
#[pyclass]
struct MS2Model {
    inner: MS2ModelWrapper,
}

#[pymethods]
impl MS2Model {
    /// Create a new MS2 prediction model.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file (.safetensors or .pth)
    /// * `arch` - Model architecture (currently only "ms2_bert" is supported)
    /// * `constants_path` - Path to the constants file (.yaml)
    /// * `use_cuda` - Whether to use CUDA for inference (default: False)
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: String,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = if use_cuda {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(to_py_err)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(PyRuntimeError::new_err("CUDA support not enabled. Rebuild with --features cuda"));
            }
        } else {
            Device::Cpu
        };

        let inner = MS2ModelWrapper::new(
            model_path,
            constants_path,
            &arch,
            device,
        ).map_err(to_py_err)?;

        Ok(MS2Model { inner })
    }

    /// Predict MS2 fragment intensities for a batch of peptides.
    ///
    /// # Arguments
    /// * `sequences` - List of peptide sequences (strings)
    /// * `mods` - List of modification strings
    /// * `mod_sites` - List of modification site strings
    /// * `charges` - List of charge states (integers)
    /// * `nces` - List of normalized collision energies (integers)
    /// * `instruments` - Optional list of instrument names (strings)
    ///
    /// # Returns
    /// A list of numpy arrays, where each array contains the predicted intensities
    /// for one peptide (shape: [num_fragments, num_ion_types])
    #[pyo3(signature = (sequences, mods, mod_sites, charges, nces, instruments=None))]
    fn predict(
        &self,
        py: Python<'_>,
        sequences: Vec<String>,
        mods: Vec<String>,
        mod_sites: Vec<String>,
        charges: Vec<i32>,
        nces: Vec<i32>,
        instruments: Option<Vec<String>>,
    ) -> PyResult<Vec<PyObject>> {
        // Convert strings to Arc<[u8]>
        let sequences_arc: Vec<Arc<[u8]>> = sequences
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        
        let mods_arc: Vec<Arc<[u8]>> = mods
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        
        let mod_sites_arc: Vec<Arc<[u8]>> = mod_sites
            .iter()
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();

        let instruments_vec = instruments.map(|inst| {
            inst.iter()
                .map(|s| Some(Arc::from(s.as_bytes().to_vec().into_boxed_slice()) as Arc<[u8]>))
                .collect::<Vec<Option<Arc<[u8]>>>>()
        }).unwrap_or_else(|| vec![None; sequences.len()]);

        // Perform prediction
        let result = self.inner
            .predict(
                &sequences_arc,
                &mods_arc,
                &mod_sites_arc,
                charges,
                nces,
                instruments_vec,
            )
            .map_err(to_py_err)?;

        // Extract MS2 intensities
        let ms2_values = match result {
            PredictionResult::MS2Result(values) => values,
            _ => return Err(PyRuntimeError::new_err("Expected MS2 prediction result")),
        };

        // Convert each peptide's intensities to a numpy array
        let result: Vec<PyObject> = ms2_values
            .into_iter()
            .map(|peptide_intensities| {
                // peptide_intensities is Vec<Vec<f32>> - convert to 2D array
                let rows = peptide_intensities.len();
                let cols = if rows > 0 { peptide_intensities[0].len() } else { 0 };
                
                // Convert to ndarray first, then to PyArray
                let array2d = ndarray::Array2::from_shape_fn((rows, cols), |(i, j)| {
                    peptide_intensities[i][j]
                });
                
                // Create PyArray2 from the ndarray
                PyArray2::from_array_bound(py, &array2d).into_py(py)
            })
            .collect();

        Ok(result)
    }
}

/// Python module for redeem-properties peptide property prediction.
///
/// This module provides Python bindings for the ReDeeM peptide property prediction models.
/// It includes models for predicting:
/// - Retention Time (RT)
/// - Collision Cross-Section (CCS)
/// - MS2 Fragment Intensities
#[pymodule]
fn redeem_properties_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RTModel>()?;
    m.add_class::<CCSModel>()?;
    m.add_class::<MS2Model>()?;
    Ok(())
}
