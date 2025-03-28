use crate::{
    building_blocks::featurize::{self, get_aa_indices, get_mod_features},
    models::{ccs_model::CCSModelWrapper, ms2_model::MS2ModelWrapper, rt_model::RTModelWrapper},
    utils::{
        data_handling::PeptideData,
        logging::Progress,
        peptdeep_utils::{
            get_modification_indices, get_modification_string, parse_instrument_index,
            remove_mass_shift, ModificationMap,
        }
    },
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Module, Optimizer, VarMap};
use log::info;
use rayon::prelude::*;
use std::ops::{Index, IndexMut};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{collections::HashMap, path::PathBuf};

// Constants
const CHARGE_FACTOR: f64 = 0.1;
const NCE_FACTOR: f64 = 0.01;


/// Load tensors from a model file.
/// 
/// Supported model formats include:
/// - PyTorch (.pt, .pth, .pkl)
/// - SafeTensors (.safetensors)
/// 
/// # Arguments
/// * `model_path` - Path to the model file.
/// * `device` - Device to load the tensors on.
/// 
/// # Returns
/// A vector of tuples containing the tensor names and their corresponding tensors.
pub fn load_tensors_from_model<P: AsRef<Path>>(model_path: P, device: &Device) -> Result<Vec<(String, Tensor)>> {
    let path: &Path = model_path.as_ref();
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    match extension.as_str() {
        "pt" | "pth" | "pkl" => {
            log::trace!("Loading tensors from PyTorch model: {:?}", path);
            let tensor_data = candle_core::pickle::read_all(path)
                .with_context(|| format!("Failed to read PyTorch model from: {:?}", path))?;
            Ok(tensor_data)
        }
        "safetensors" => {
            log::trace!("Loading tensors from SafeTensors model: {:?}", path);
            let tensor_data = candle_core::safetensors::load(path, device)
                .with_context(|| format!("Failed to load SafeTensors from: {:?}", path))?;

            // Convert HashMap<String, Tensor> to Vec<(String, Tensor)>
            let tensors = tensor_data.into_iter().collect();

            Ok(tensors)
        }
        _ => Err(anyhow::anyhow!("Unsupported model format: {:?}", path)),
    }
}


/// Represents the type of property to predict.
#[derive(Clone)]
pub enum PropertyType {
    RT,
    CCS,
    MS2,
}

impl PropertyType {
    pub fn as_str(&self) -> &str {
        match self {
            PropertyType::RT => "RT",
            PropertyType::CCS => "CCS",
            PropertyType::MS2 => "MS2",
        }
    }
}

/// Represents a single prediction value or a matrix of prediction values.
/// 
/// This enum is used to store the output of a model prediction, which can be a single value or a matrix of values. For example, retention time (RT) and collision cross-section (CCS) predictions are single values, while MS2 intensity predictions are matrices.
#[derive(Clone)]
pub enum PredictionValue {
    Single(f32),
    Matrix(Vec<Vec<f32>>),
}

impl PredictionValue {
    // Returns a reference to the element at position (i, j) if it exists
    pub fn get(&self, i: usize, j: usize) -> Option<&f32> {
        match self {
            PredictionValue::Single(_) => None,  
            PredictionValue::Matrix(vec) => vec.get(i).and_then(|row| row.get(j)),
        }
    }

}

impl Index<usize> for PredictionValue {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            PredictionValue::Single(val) => val,
            PredictionValue::Matrix(matrix) => &matrix[index][0], // Single index for Matrix (first element of each row)
        }
    }
}

impl Index<(usize, usize)> for PredictionValue {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        match self {
            PredictionValue::Single(val) => val, // Single variant does not support multi-indexing
            PredictionValue::Matrix(matrix) => &matrix[index.0][index.1], // Multi-index for Matrix
        }
    }
}


/// Represents the output of a model prediction.
/// 
/// This enum is used to store the output of a model prediction, which can be a vector of retention times (RT), collision cross-sections (CCS), or a vector matrices of MS2 intensities.
#[derive(Debug, Clone)]
pub enum PredictionResult {
    RTResult(Vec<f32>),
    IMResult(Vec<f32>),
    MS2Result(Vec<Vec<Vec<f32>>>),
}

impl PredictionResult {
    pub fn len(&self) -> usize {
        match self {
            PredictionResult::RTResult(vec) => vec.len(),
            PredictionResult::IMResult(vec) => vec.len(),
            PredictionResult::MS2Result(vec) => vec.len(),
        }
    }

    pub fn get_prediction_entry(&self, index: usize) -> PredictionValue {
        match self {
            PredictionResult::RTResult(vec) => PredictionValue::Single(vec[index].clone()),
            PredictionResult::IMResult(vec) => PredictionValue::Single(vec[index].clone()),
            PredictionResult::MS2Result(vec) => PredictionValue::Matrix(vec[index].clone()), 
        }
    }
}

/// Populates a mutable `VarMap` instance with tensors.
/// 
/// # Arguments
/// * `var_map` - A mutable reference to a `VarMap` instance.
/// * `tensor_data` - A vector of tuples containing the tensor names and their corresponding tensors.
/// * `device` - The device to load the tensors on.
pub fn create_var_map(
    var_map: &mut VarMap,
    tensor_data: Vec<(String, Tensor)>,
    device: &Device,
) -> Result<()> {
    let mut ws = var_map.data().lock().unwrap();

    for (name, tensor) in tensor_data {
        ws.insert(name, Var::from_tensor(&tensor.to_device(device)?)?);
    }

    Ok(())
}


/// Represents an abstract deep learning model interface.
/// 
/// This trait defines the methods and properties that a deep learning model must implement to be used for property prediction tasks.
pub trait ModelInterface: Send + Sync {

    /// Get the property type of the model.
    fn property_type(&self) -> PropertyType;

    /// Get the model architecture name.
    fn model_arch(&self) -> &'static str;

    /// Create a new instance of the model.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: P,
        fixed_sequence_len: usize,
        num_frag_types: usize,
        num_modloss_types: usize,
        mask_modloss: bool,
        device: Device,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Forward pass through the model.
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error>;

    /// Predict the retention times for a peptide sequence.
    ///
    /// # Arguments
    ///   * `peptide_sequences` - A vector of peptide sequences.
    ///   * `mods` - A vector of strings representing the modifications for each peptide.
    ///   * `mod_sites` - A vector of strings representing the modification sites for each peptide.
    ///  * `charge` - An optional vector of charge states for each peptide.
    ///  * `nce` - An optional vector of nominal collision energies for each peptide.
    ///  * `instrument` - An optional vector of instrument names for each peptide.
    ///
    /// # Returns
    ///    A vector of predicted retention times.
    fn predict(
        &self,
        peptide_sequences: &[String],
        mods: &[String],
        mod_sites: &[String],
        charge: Option<Vec<i32>>,
        nce: Option<Vec<i32>>,
        instrument: Option<Vec<String>>,
    ) -> Result<PredictionResult> {
        // Encode the batch of peptides
        let input_tensor =
            self.encode_peptides(peptide_sequences, mods, mod_sites, charge, nce, instrument)?;

        // Forward pass through the model
        let output = self.forward(&input_tensor)?;

        match self.property_type() {
            PropertyType::RT => {
                let predictions: Vec<f32> = output.to_vec1()?;
                Ok(PredictionResult::RTResult(predictions))
            }
            PropertyType::CCS => {
                let predictions: Vec<f32> = output.to_vec1()?;
                Ok(PredictionResult::IMResult(predictions))
            }
            PropertyType::MS2 => {
                let out = self.process_predictions(&output, self.get_min_pred_intensity())?;
                // Each prediction per peptide is a vector of vectors of f32, i.e. Number of fragment ions by number of ion types ordered as b_z1, b_z2, y_z1, y_z2, b_modloss_z1, b_modloss_z2, y_modloss_z1, y_modloss_z2
                let predictions: Vec<Vec<Vec<f32>>> = out.to_vec3()?;
                Ok(PredictionResult::MS2Result(predictions))
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported property type: {:?}",
                self.get_property_type()
            )),
        }
    }

    /// Encode a batch of peptide sequences (plus modifications) into a tensor.
    /// 
    /// # Arguments
    /// * `peptide_sequences` - A vector of peptide sequences.
    /// * `mods` - A vector of strings representing the modifications for each peptide.
    /// * `mod_sites` - A vector of strings representing the modification site indices for each peptide.
    /// * `charge` - An optional vector of charge states for each peptide.
    /// * `nce` - An optional vector of nominal collision energies for each peptide.
    /// * `instruments` - An optional vector of instrument names for each peptide.
    /// 
    /// # Returns
    /// A tensor containing the encoded peptide sequences.
    fn encode_peptides(
        &self,
        peptide_sequences: &[String],
        mods: &[String],
        mod_sites: &[String],
        charges: Option<Vec<i32>>,
        nces: Option<Vec<i32>>,
        instruments: Option<Vec<String>>,
    ) -> Result<Tensor> {
        if peptide_sequences.len() != mods.len() || peptide_sequences.len() != mod_sites.len() {
            return Err(anyhow::anyhow!(
                "Mismatch in input lengths: peptide_sequences, mods, and mod_sites must have the same length."
            ));
        }

        // Encode peptides in parallel using Rayon
        let encoded_tensors: Vec<Tensor> = peptide_sequences
            .par_iter() // Use Rayon's parallel iterator
            .enumerate()
            .map(|(i, peptide)| {
                self.encode_peptide(
                    peptide,
                    &mods[i],
                    &mod_sites[i],
                    charges.as_ref().map(|c| c[i]),
                    nces.as_ref().map(|n| n[i]),
                    instruments.as_ref().map(|ins| ins[i].as_str()),
                )
            })
            .collect::<Result<Vec<Tensor>>>()?; // Collect results and propagate errors if any

        // Determine the maximum sequence length
        let max_seq_len = encoded_tensors
            .par_iter()
            .map(|t| t.shape().dims3().unwrap().1) // Get sequence length (dimension 1)
            .max()
            .unwrap_or(0);

        // Pad tensors to the max_seq_len
        let padded_tensors: Result<Vec<Tensor>> = encoded_tensors
            .into_par_iter() // Use Rayon's parallel iterator
            .map(|t| {
                let (_, seq_len, feature_size) = t.shape().dims3()?; // Extract feature dimension
                if seq_len < max_seq_len {
                    let pad_size = max_seq_len - seq_len;
                    // Create a padding tensor with the correct shape and type
                    let pad = Tensor::zeros(
                        &[1, pad_size, feature_size], // Use the correct feature dimension
                        t.dtype(),
                        t.device(),
                    )?;
                    // Concatenate padding along sequence length
                    Tensor::cat(&[&t, &pad], 1)
                } else {
                    Ok(t)
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into);

        let padded_tensors = padded_tensors?;

        // Concatenate all padded tensors along the batch dimension
        let batch_tensor = Tensor::cat(&padded_tensors, 0)?;

        Ok(batch_tensor)
    }

    /// Encode peptide sequence (plus modifications) into a tensor.
    /// 
    /// # Arguments
    /// * `peptide_sequence` - The peptide sequence.
    /// * `mods` - A string representing the modifications for the peptide.
    /// * `mod_sites` - A string representing the modification site indices for the peptide.
    /// * `charge` - An optional charge state for the peptide.
    /// * `nce` - An optional nominal collision energy for the peptide.
    /// * `instrument` - An optional instrument name for the peptide.
    /// 
    /// # Returns
    /// A tensor containing the encoded peptide sequence.
    fn encode_peptide(
        &self,
        peptide_sequence: &str,
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        nce: Option<i32>,
        instrument: Option<&str>,
    ) -> Result<Tensor> {
        log::trace!(
            "[ModelInterface::encode_peptide] Encoding peptide: {:?}, mods: {:?}, mod_sites: {:?}, charge: {:?}, nce: {:?}, instrument: {:?}",
            peptide_sequence,
            mods,
            mod_sites,
            charge,
            nce,
            instrument
        );
        let aa_indices = get_aa_indices(peptide_sequence)?;
        log::trace!(
            "[ModelInterface::encode_peptide] aa_indices_tensor shape: {:?}, min: {:?}, max: {:?}",
            aa_indices.shape(),
            aa_indices.iter().min(),
            aa_indices.iter().max()
        );

        // Convert ndarray to Tensor (F32)
        let aa_indices_tensor = Tensor::from_slice(
            &aa_indices.as_slice().unwrap(),
            (aa_indices.shape()[0], aa_indices.shape()[1]),
            &self.get_device(),
        )?
        .to_dtype(DType::F32)?;

        let (batch_size, seq_len) = aa_indices_tensor.shape().dims2()?;
        let aa_indices_tensor = aa_indices_tensor.unsqueeze(2)?; // Shape: batch_size x seq_len x 1

        log::trace!(
            "[ModelInterface::encode_peptide] aa_indices_tensor shape: {:?}, min: {:?}, max: {:?}",
            aa_indices_tensor.shape(),
            aa_indices_tensor.min_all(),
            aa_indices_tensor.max_all() 
        );

        // Get modification features
        let mod_x = get_mod_features(
            mods,
            mod_sites,
            seq_len,
            self.get_mod_element_count(),
            self.get_mod_to_feature().clone(),
            self.get_device().clone(),
        )?;

        let mut features = vec![aa_indices_tensor, mod_x];

        // Conditionally add charge
        if let Some(c) = charge {
            let charge_tensor = Tensor::from_slice(
                &vec![c as f64 * CHARGE_FACTOR; seq_len],
                &[batch_size, seq_len, 1],
                &self.get_device(),
            )?
            .to_dtype(DType::F32)?;
            features.push(charge_tensor);
        }

        // Conditionally add NCE
        if let Some(n) = nce {
            let nce_tensor = Tensor::from_slice(
                &vec![n as f64 * NCE_FACTOR; seq_len],
                &[batch_size, seq_len, 1],
                &self.get_device(),
            )?
            .to_dtype(DType::F32)?;
            features.push(nce_tensor);
        }

        // Conditionally add instrument
        if let Some(instr) = instrument {
            let instrument_tensor = Tensor::from_slice(
                &vec![parse_instrument_index(instr) as u32; seq_len],
                &[batch_size, seq_len, 1],
                &self.get_device(),
            )?
            .to_dtype(DType::F32)?;
            features.push(instrument_tensor);
        }

        // Concatenate features
        Ok(Tensor::cat(&features, 2)?)
    }

    /// Fine-tune the model on a batch of training data.
    /// 
    /// # Arguments
    /// * `training_data` - A vector of `PeptideData` instances representing the training data.
    /// * `modifications` - A map of modifications and their corresponding feature vectors.
    /// * `batch_size` - The batch size to use for training.
    /// * `learning_rate` - The learning rate to use for training.
    /// * `epochs` - The number of epochs to train for.
    fn fine_tune(
        &mut self,
        training_data: &Vec<PeptideData>,
        modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        batch_size: usize,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        let num_batches = if training_data.len() < batch_size {
            1
        } else {
            let full_batches = training_data.len() / batch_size;
            if training_data.len() % batch_size > 0 {
                full_batches + 1
            } else {
                full_batches
            }
        };
    
        info!(
            "Fine-tuning {} model on {} peptide features ({} batches) for {} epochs",
            self.get_model_arch(), training_data.len(), num_batches, epochs
        );
    
        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(self.get_mut_varmap().all_vars(), params)?;
    
        for epoch in 0..epochs {
            let progress = Progress::new(num_batches, &format!("[fine-tuning] Epoch {}: ", epoch));
            let mut total_loss = 0.0;
    
            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(training_data.len());
                let batch_data = &training_data[start..end];
    
                let peptides: Vec<String> = batch_data.iter().map(|p| remove_mass_shift(&p.sequence)).collect();
                let mods: Vec<String> = batch_data.iter().map(|p| get_modification_string(&p.sequence, &modifications)).collect();
                let mod_sites: Vec<String> = batch_data.iter().map(|p| get_modification_indices(&p.sequence)).collect();
    
                let charges = batch_data.iter().filter_map(|p| p.charge).collect::<Vec<_>>();
                let charges = if charges.len() == batch_data.len() { Some(charges) } else { None };
    
                let nces = batch_data.iter().filter_map(|p| p.nce).collect::<Vec<_>>();
                let nces = if nces.len() == batch_data.len() { Some(nces) } else { None };
    
                let instruments = batch_data.iter().filter_map(|p| p.instrument.clone()).collect::<Vec<_>>();
                let instruments = if instruments.len() == batch_data.len() { Some(instruments) } else { None };
    
                let input_batch = self.encode_peptides(&peptides, &mods, &mod_sites, charges, nces, instruments)?;

                log::trace!("[ModelInterface::fine_tune] input_batch shape: {:?}, device: {:?}", input_batch.shape(), input_batch.device());
    
                let batch_targets = match self.property_type() {
                    PropertyType::RT => PredictionResult::RTResult(
                        batch_data.iter().map(|p| p.retention_time.unwrap_or_default()).collect(),
                    ),
                    PropertyType::CCS => PredictionResult::IMResult(
                        batch_data.iter().map(|p| p.ion_mobility.unwrap_or_default()).collect(),
                    ),
                    PropertyType::MS2 => PredictionResult::MS2Result(
                        batch_data.iter().map(|p| p.ms2_intensities.clone().unwrap_or_default()).collect(),
                    ),
                };
    
                let target_batch = match batch_targets {
                    PredictionResult::RTResult(ref values) | PredictionResult::IMResult(ref values) => {
                        Tensor::new(values.clone(), &self.get_device())?
                    }
                    PredictionResult::MS2Result(ref spectra) => {
                        let max_len = spectra.iter().map(|s| s.len()).max().unwrap_or(1);
                        let feature_dim = spectra.get(0).and_then(|s| s.get(0)).map(|v| v.len()).unwrap_or(1);
                        let mut padded_spectra = spectra.clone();
                        for s in &mut padded_spectra {
                            s.resize(max_len, vec![0.0; feature_dim]);
                        }
                        Tensor::new(padded_spectra.concat(), &self.get_device())?.reshape((batch_data.len(), max_len, feature_dim))?
                    }
                };
    
                let predicted = self.forward(&input_batch)?;
                let loss = candle_nn::loss::mse(&predicted, &target_batch)?;
                opt.backward_step(&loss)?;
    
                total_loss += loss.to_vec0::<f32>().unwrap_or(990.0);

                progress.update_description(&format!("[fine-tuning] Epoch {}: Loss: {}", epoch, loss.to_vec0::<f32>()?));
                progress.inc();
            }
    
            let avg_loss = total_loss / num_batches as f32;
            progress.update_description(&format!("[fine-tuning] Epoch {}: Avg. Batch Loss: {}", epoch, avg_loss));
            progress.finish();
        }
    
        Ok(())
    }

    /// Set model to evaluation mode for inference
    /// This disables dropout and other training-specific layers.
    fn set_evaluation_mode(&mut self);

    /// Set model to training mode for training
    /// This enables dropout and other training-specific layers.
    fn set_training_mode(&mut self);

    fn get_property_type(&self) -> String;

    fn get_model_arch(&self) -> String;

    fn get_device(&self) -> &Device;

    fn get_mod_element_count(&self) -> usize;

    fn get_mod_to_feature(&self) -> &HashMap<String, Vec<f32>>;

    fn get_min_pred_intensity(&self) -> f32;

    fn get_mut_varmap(&mut self) -> &mut VarMap;

    fn print_summary(&self);
    fn print_weights(&self);

    /// Save model weights to a file in safetensors format.
    fn save(&mut self, path: &str) -> Result<()> {
        info!(
            "Saving {} model weights to: {:?}",
            self.get_model_arch(),
            path
        );
        self.get_mut_varmap().clone().save(&PathBuf::from(path))?;
        Ok(())
    }

    fn apply_min_pred_value(&self, tensor: &Tensor, min_pred_value: f32) -> Result<Tensor> {
        // Create a tensor with the same shape as the input, filled with min_pred_value
        let min_tensor = Tensor::full(min_pred_value, tensor.shape(), tensor.device())?;

        // Use the maximum operation to replace values less than min_pred_value
        Ok(tensor.maximum(&min_tensor)?)
    }

    // TODO: Maybe move to ms2_bert_model, since it's specific to that model
    fn process_predictions(&self, predicts: &Tensor, min_inten: f32) -> Result<Tensor> {
        // Reshape and get max
        let (batch_size, seq_len, feature_size) = predicts.shape().dims3()?;
        let reshaped = predicts.reshape((batch_size, ()))?;
        let apex_intens = reshaped.max(1)?;

        // Replace values <= 0 with 1
        // let ones = Tensor::ones_like(&apex_intens)?;
        let apex_intens = apex_intens.maximum(&apex_intens)?;

        // Reshape apex_intens for broadcasting
        let apex_intens_reshaped = apex_intens.reshape(((), 1, 1))?;

        // Explicitly broadcast apex_intens_reshaped to match predicts shape
        let broadcasted_apex_intens = apex_intens_reshaped.broadcast_as(predicts.shape())?;

        // Divide predicts by broadcasted apex_intens
        let normalized = predicts.div(&broadcasted_apex_intens)?;

        // Replace values < min_inten with 0.0
        let zeros = Tensor::zeros_like(&normalized)?;
        let min_inten_tensor = Tensor::full(min_inten, normalized.shape(), normalized.device())?;
        let mask = normalized.ge(&min_inten_tensor)?;
        let final_predicts = mask.where_cond(&normalized, &zeros)?;

        Ok(final_predicts)
    }
}

/// Parameters for the `predict` method of a `ModelInterface` implementation.
pub struct Parameters {
    /// The instrument data was acquired on. Refer to list of supported instruments in const yaml file.
    pub instrument: String,
    /// The nominal collision energy (NCE) used for data acquisition.
    pub nce: f32,
}

impl Parameters {
    /// Creates a new instance of `Parameters` with the given instrument and NCE.
    ///
    /// # Arguments
    ///
    /// * `instrument` - The instrument data was acquired on.
    /// * `nce` - The nominal collision energy (NCE) used for data acquisition.
    ///
    /// # Returns
    ///
    /// A new `Parameters` instance with the given instrument and NCE.
    ///
    /// # Example
    ///
    /// ```
    /// let params = Parameters::new("QE", 20);
    /// ```
    pub fn new(instrument: &str, nce: f32) -> Self {
        Parameters {
            instrument: instrument.to_string(),
            nce,
        }
    }
}

/// Represents a collection of deep learning models for various property prediction tasks.
///
/// This struct holds optional references to models for retention time (RT),
/// collision cross-section (CCS), and MS2 intensity predictions. Each model
/// is wrapped in an Arc<Mutex<>> for thread-safe shared ownership.
pub struct DLModels {
    /// Parameters for prediction models.
    pub params: Option<Parameters>,

    /// Optional retention time prediction model.
    pub rt_model: Option<Arc<Mutex<RTModelWrapper>>>,

    /// Optional collision cross-section prediction model.
    pub ccs_model: Option<Arc<Mutex<CCSModelWrapper>>>,

    /// Optional MS2 intensity prediction model.
    pub ms2_model: Option<Arc<Mutex<MS2ModelWrapper>>>,
}

impl DLModels {
    /// Creates a new instance of `DLModels` with all models set to `None`.
    ///
    /// # Returns
    ///
    /// A new `DLModels` instance with no models initialized.
    ///
    /// # Example
    ///
    /// ```
    /// let mut models = DLModels::new();
    ///
    /// models.rt_model = Some(Arc::new(Mutex::new(RTModelWrapper::new())));
    ///
    /// ```
    pub fn new() -> Self {
        DLModels {
            params: None,
            rt_model: None,
            ccs_model: None,
            ms2_model: None,
        }
    }

    /// Checks if any of the models are present (not None).
    ///
    /// # Returns
    ///
    /// `true` if at least one model is present, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// let mut models = DLModels::new();
    /// assert!(!models.is_not_empty());
    ///
    /// models.rt_model = Some(Arc::new(Mutex::new(RTModelWrapper::new())));
    /// assert!(models.is_not_empty());
    /// ```
    pub fn is_not_empty(&self) -> bool {
        self.rt_model.is_some() || self.ccs_model.is_some() || self.ms2_model.is_some()
    }
}
