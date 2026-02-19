use crate::{
    building_blocks::featurize::{
        self, aa_indices_tensor, aa_indices_tensor_from_arc, get_mod_features_from_parsed,
        get_mod_features_from_parsed_arc,
    },
    models::{ccs_model::CCSModelWrapper, ms2_model::MS2ModelWrapper, rt_model::RTModelWrapper},
    utils::{
        data_handling::{PeptideBatchData, PeptideData, TargetNormalization},
        logging::Progress,
        peptdeep_utils::{
            get_modification_indices, get_modification_string, parse_instrument_index,
            remove_mass_shift,
        },
        stats::{compute_loss_stats, Metrics, TrainingPhase, TrainingStepMetrics},
        utils::{get_tensor_stats, CosineWithWarmup, LRScheduler},
    },
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Optimizer, VarMap};
use log::info;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::HashMap, path::PathBuf};
use std::{
    ops::{Deref, Index},
    process::Output,
    sync::Arc,
};

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
pub fn load_tensors_from_model<P: AsRef<Path>>(
    model_path: P,
    device: &Device,
) -> Result<Vec<(String, Tensor)>> {
    let path: &Path = model_path.as_ref();
    let extension = path
        .extension()
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
    CCSResult(Vec<f32>),
    MS2Result(Vec<Vec<Vec<f32>>>),
}

impl PredictionResult {
    pub fn len(&self) -> usize {
        match self {
            PredictionResult::RTResult(vec) => vec.len(),
            PredictionResult::CCSResult(vec) => vec.len(),
            PredictionResult::MS2Result(vec) => vec.len(),
        }
    }

    pub fn get_prediction_entry(&self, index: usize) -> PredictionValue {
        match self {
            PredictionResult::RTResult(vec) => PredictionValue::Single(vec[index].clone()),
            PredictionResult::CCSResult(vec) => PredictionValue::Single(vec[index].clone()),
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

pub trait ModelClone {
    fn clone_box(&self) -> Box<dyn ModelInterface + Send + Sync>;
}

impl<T> ModelClone for T
where
    T: 'static + ModelInterface + Clone + Send + Sync,
{
    fn clone_box(&self) -> Box<dyn ModelInterface + Send + Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ModelInterface + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Represents an abstract deep learning model interface.
///
/// This trait defines the methods and properties that a deep learning model must implement to be used for property prediction tasks.
pub trait ModelInterface: Send + Sync + ModelClone {
    /// Get the property type of the model.
    fn property_type(&self) -> PropertyType;

    /// Get the model architecture name.
    fn model_arch(&self) -> &'static str;

    /// Create a new model instance from scratch (no pretrained weights).
    /// This is typically used when training a new model from scratch.
    fn new_untrained(device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Create a new instance of the model (given a pretrained model (.pth or .safetensors and constants file).
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file (.pth or .safetensors).
    /// * `constants_path` - Optional path to the model constants file (.yaml). If none, will use the default constants.
    ///
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: Option<P>,
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

    /// Predict the property for a batch of peptide sequences.
    ///
    /// # Arguments
    /// * `peptide_sequences` - A slice of `Arc<[u8]>` containing each peptide sequence.
    /// * `mods` - A slice of `Arc<[u8]>` with modifications for each peptide.
    /// * `mod_sites` - A slice of `Arc<[u8]>` representing modification sites per peptide.
    /// * `charges` - Optional vector of charge states.
    /// * `nces` - Optional vector of normalized collision energies.
    /// * `instruments` - Optional vector of instrument names as `Arc<[u8]>`.
    ///
    /// # Returns
    /// A `PredictionResult` containing either RT, CCS, or MS2 predictions.
    fn predict(
        &self,
        peptide_sequences: &[Arc<[u8]>],
        mods: &[Arc<[u8]>],
        mod_sites: &[Arc<[u8]>],
        charges: Option<Vec<i32>>,
        nces: Option<Vec<i32>>,
        instruments: Option<Vec<Option<Arc<[u8]>>>>,
    ) -> Result<PredictionResult> {
        let input_tensor = self
            .encode_peptides(
                peptide_sequences,
                mods,
                mod_sites,
                charges,
                nces,
                instruments,
            )?
            .to_device(self.get_device())?;

        let output = self.forward(&input_tensor)?;

        match self.property_type() {
            PropertyType::RT => {
                let predictions: Vec<f32> = output.to_vec1()?;
                Ok(PredictionResult::RTResult(predictions))
            }
            PropertyType::CCS => {
                let predictions: Vec<f32> = output.to_vec1()?;
                Ok(PredictionResult::CCSResult(predictions))
            }
            PropertyType::MS2 => {
                let out = self.process_predictions(&output, self.get_min_pred_intensity())?;
                let predictions: Vec<Vec<Vec<f32>>> = out.to_vec3()?;
                Ok(PredictionResult::MS2Result(predictions))
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported property type: {:?}",
                self.get_property_type()
            )),
        }
    }

    /// Encode peptide sequence (plus modifications) into a tensor.
    fn encode_peptide(
        &self,
        peptide_sequence: &Arc<[u8]>,
        mods: &Arc<[u8]>,
        mod_sites: &Arc<[u8]>,
        charge: Option<i32>,
        nce: Option<i32>,
        instrument: Option<&Arc<[u8]>>,
    ) -> Result<Tensor> {
        let device = self.get_device();
        let mod_feature_size = self.get_mod_element_count();
        let mod_to_feature = self.get_mod_to_feature();

        // log::trace!(
        //     "[ModelInterface::encode_peptide] peptide_sequence: {:?} | mods: {:?} | mod_sites: {:?} | charge: {:?} | nce: {:?} | instrument: {:?}",
        //     peptide_sequence, mods, mod_sites, charge, nce, instrument
        // );

        let aa_tensor = aa_indices_tensor_from_arc(peptide_sequence, device)?;
        let (batch_size, seq_len, _) = aa_tensor.shape().dims3()?;

        let mod_names: Vec<Arc<[u8]>> = std::str::from_utf8(mods)
            .unwrap_or("")
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();

        let mod_indices: Vec<usize> = std::str::from_utf8(mod_sites)
            .unwrap_or("")
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().unwrap())
            .collect();

        let mod_tensor = get_mod_features_from_parsed_arc(
            &mod_names,
            &mod_indices,
            seq_len,
            mod_feature_size,
            mod_to_feature,
            device,
        )?;

        let mut features = vec![aa_tensor, mod_tensor];

        if let Some(c) = charge {
            let charge_tensor = Tensor::from_slice(
                &vec![c as f64 * CHARGE_FACTOR; seq_len],
                &[batch_size, seq_len, 1],
                device,
            )?
            .to_dtype(DType::F32)?;
            features.push(charge_tensor);
        }

        if let Some(n) = nce {
            let nce_tensor = Tensor::from_slice(
                &vec![n as f64 * NCE_FACTOR; seq_len],
                &[batch_size, seq_len, 1],
                device,
            )?
            .to_dtype(DType::F32)?;
            features.push(nce_tensor);
        }

        if let Some(instr) = instrument {
            let instr_str = std::str::from_utf8(instr).unwrap_or("");
            let instr_idx = parse_instrument_index(instr_str) as u32;
            let instr_tensor =
                Tensor::from_slice(&vec![instr_idx; seq_len], &[batch_size, seq_len, 1], device)?
                    .to_dtype(DType::F32)?;
            features.push(instr_tensor);
        }

        if features.len() == 1 {
            let output = features.remove(0);
            let (mean, min, max) = get_tensor_stats(&output)?;
            if !mean.is_finite() || !min.is_finite() || !max.is_finite() {
                log::error!("For Peptide = {:?} encode_peptides produced non-finite tensor stats: mean={mean}, min={min}, max={max}", peptide_sequence);
                anyhow::bail!("Non-finite values found in peptide encoding output.");
            }
            Ok(output)
        } else {
            let output = Tensor::cat(&features, 2)?;
            let (mean, min, max) = get_tensor_stats(&output)?;
            if !mean.is_finite() || !min.is_finite() || !max.is_finite() {
                log::error!("For Peptide = {:?} encode_peptides produced non-finite tensor stats: mean={mean}, min={min}, max={max}", peptide_sequence);
                anyhow::bail!("Non-finite values found in peptide encoding output.");
            }
            Ok(output)
        }
    }

    /// Encode a batch of peptide sequences into a tensor
    fn encode_peptides(
        &self,
        peptide_sequences: &[Arc<[u8]>],
        mods: &[Arc<[u8]>],
        mod_sites: &[Arc<[u8]>],
        charges: Option<Vec<i32>>,
        nces: Option<Vec<i32>>,
        instruments: Option<Vec<Option<Arc<[u8]>>>>,
    ) -> Result<Tensor> {
        let len = peptide_sequences.len();

        let tensors: Vec<_> = (0..len)
            .into_par_iter()
            .map(|i| {
                self.encode_peptide(
                    &peptide_sequences[i],
                    &mods[i],
                    &mod_sites[i],
                    charges.as_ref().map(|v| v[i]),
                    nces.as_ref().map(|v| v[i]),
                    instruments.as_ref().and_then(|v| v[i].as_ref()),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        if tensors.is_empty() {
            return Err(anyhow::anyhow!(
                "Encoding batch of peptides failed, the resulting tensor batch is empty."
            ));
        }

        let max_len = tensors
            .iter()
            .map(|t| t.shape().dims3().unwrap().1)
            .max()
            .unwrap_or(0);

        let expected_feat_dim = tensors
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("Empty input batch"))?
            .shape()
            .dims3()?
            .2;

        let padded = tensors
            .into_par_iter()
            .map(|t| {
                let (_, seq_len, feat_dim) = t.shape().dims3()?;

                if feat_dim != expected_feat_dim {
                    return Err(anyhow::anyhow!(
                        "Inconsistent feature dim: expected {}, got {}",
                        expected_feat_dim,
                        feat_dim
                    ));
                }

                if seq_len < max_len {
                    let pad =
                        Tensor::zeros(&[1, max_len - seq_len, feat_dim], t.dtype(), t.device())?;
                    Ok(Tensor::cat(&[&t, &pad], 1)?)
                } else {
                    Ok(t)
                }
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        Ok(Tensor::cat(&padded, 0)?)
    }

    /// Train the model from scratch using a batch of training data.
    ///
    /// This method initializes model weights from scratch and trains over the given peptide feature data for a specified
    /// number of epochs. Optionally performs validation and tracks both training and validation loss statistics.
    /// Early stopping is applied if the validation loss does not improve for a consecutive number of epochs.
    ///
    /// A Cosine Annealing with Warmup learning rate scheduler is used to adjust the learning rate during training. The initial warmup period is set to 10% of the total training steps.
    ///
    /// # Arguments
    /// * `training_data` - Vector of peptide records used for training.
    /// * `validation_data` - Optional vector of peptide records used for validation at the end of each epoch.
    /// * `modifications` - A map of known modifications to encode modified peptides.
    /// * `batch_size` - Batch size used for training.
    /// * `validation_batch_size` - Batch size used during validation.
    /// * `learning_rate` - Learning rate for the AdamW optimizer.
    /// * `epochs` - Maximum number of training epochs.
    /// * `early_stopping_patience` - Number of epochs to wait before stopping if validation loss does not improve.
    /// * `context` - A string representing the context for logging, e.g., "training" or "fine-tuning".
    /// * `save_checkpoints` - Flag to save model checkpoints during training.
    /// * `track_metrics` - Flag to track training and validation metrics.
    /// * `target_norm` - Normalization parameters to de-normalize metrics for reporting.
    /// * `warmup_fraction` - Optional fraction of total steps to use for LR warmup (0.0–1.0). Defaults to 0.12.
    ///
    /// # Returns
    /// [`TrainingStepMetrics`] - A struct containing training and validation loss statistics, learning rates, and other metrics.
    fn train(
        &mut self,
        training_data: &Vec<PeptideData>,
        validation_data: Option<&Vec<PeptideData>>,
        modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        batch_size: usize,
        validation_batch_size: usize,
        learning_rate: f64,
        epochs: usize,
        early_stopping_patience: usize,
        context: &str,
        save_checkpoints: bool,
        track_metrics: bool,
        target_norm: TargetNormalization,
        // Optional list of variable name prefixes to train. When `None`, all vars are trained.
        train_var_prefixes: Option<Vec<String>>,
        warmup_fraction: Option<f64>,
    ) -> Result<TrainingStepMetrics> {
        let num_batches = (training_data.len() + batch_size - 1) / batch_size;
        let total_steps = num_batches * epochs;
        // Default warmup is 12% of total steps unless overridden.
        let warmup_frac = warmup_fraction.unwrap_or(0.12).clamp(0.0, 1.0);
        let warmup_steps = ((total_steps as f64) * warmup_frac).ceil() as usize;

        info!(
            "{} {} model on {} peptide features ({} batches) for {} epochs",
            context,
            self.get_model_arch(),
            training_data.len(),
            num_batches,
            epochs
        );

        let mut step_metrics = TrainingStepMetrics {
            epochs: vec![],
            steps: vec![],
            learning_rates: vec![],
            losses: vec![],
            phases: vec![],
            precisions: vec![],
            recalls: vec![],
            accuracies: vec![],
            maes: vec![],
            rmses: vec![],
            r2s: vec![],
        };

        let mut step_idx = 0;
        let mut val_step_idx = 0;

        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        // If the caller specified prefixes to train, select only those vars from the VarMap.
        // If the list is empty or omitted, train all variables.
        let mut opt = {
            if let Some(prefixes) = &train_var_prefixes {
                if prefixes.is_empty() {
                    candle_nn::AdamW::new(self.get_mut_varmap().all_vars(), params)?
                } else {
                    let selected: Vec<Var> = {
                        let ws = self.get_mut_varmap().data().lock().unwrap();
                        ws.iter()
                            .filter(|(name, _)| prefixes.iter().any(|p| name.starts_with(p)))
                            .map(|(_, v)| v.clone())
                            .collect()
                    };
                    if selected.is_empty() {
                        candle_nn::AdamW::new(self.get_mut_varmap().all_vars(), params)?
                    } else {
                        candle_nn::AdamW::new(selected, params)?
                    }
                }
            } else {
                candle_nn::AdamW::new(self.get_mut_varmap().all_vars(), params)?
            }
        };
        let mut lr_scheduler = CosineWithWarmup::new(
            learning_rate,
            warmup_steps,
            total_steps,
            0.5, // one full cosine cycle
        );

        let mut best_val_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut epoch_losses = vec![];

        let compute_loss = |predicted: &Tensor, target: &Tensor| -> Result<Tensor> {
            // Ensure target matches the predicted shape before any math.
            let adjusted_target = {
                let mut t = target.clone();
                let t_elems = t.elem_count();
                let p_elems = predicted.elem_count();

                // Fast path: already same count and shape.
                if t_elems == p_elems && t.shape() == predicted.shape() {
                    t
                } else if t_elems == 0 {
                    Tensor::zeros_like(predicted)?
                } else if t_elems == p_elems {
                    // Same number of elements, just reshape.
                    t.reshape(predicted.shape())?
                } else if t_elems == 1 && p_elems > 1 {
                    t.reshape(predicted.shape())?
                } else {
                    // Best effort: broadcast, then fall back to zeros_like if impossible.
                    match t.broadcast_as(predicted.shape()) {
                        Ok(b) => b,
                        Err(_) => Tensor::zeros_like(predicted)?,
                    }
                }
            };

            let p_shape = predicted.shape();
            let t_shape = adjusted_target.shape();
            let p_dims = predicted.dims();
            let t_dims = adjusted_target.dims();
            if p_shape != t_shape || p_dims != t_dims {
                log::error!(
                    "compute_loss shape mismatch before op: pred shape {:?} dims {:?}, tgt shape {:?} dims {:?}, pred elems {}, tgt elems {}",
                    p_shape,
                    p_dims,
                    t_shape,
                    t_dims,
                    predicted.elem_count(),
                    adjusted_target.elem_count()
                );
                return Err(anyhow::anyhow!(
                    "compute_loss shape mismatch: pred {:?} dims {:?}, tgt {:?} dims {:?}",
                    p_shape,
                    p_dims,
                    t_shape,
                    t_dims
                ));
            }

            match target_norm {
                TargetNormalization::ZScore(_, _) => {
                    // Huber with delta=1.0
                    let delta = 1.0f32;
                    let delta_t = Tensor::new(delta, predicted.device())?;
                    let half = Tensor::new(0.5f32, predicted.device())?;
                    let delta_sq_half = Tensor::new(0.5f32 * delta * delta, predicted.device())?;
                    let delta_full = delta_t.broadcast_as(predicted.shape())?;
                    let half_full = half.broadcast_as(predicted.shape())?;
                    let delta_sq_half_full = delta_sq_half.broadcast_as(predicted.shape())?;
                    let diff = predicted.sub(&adjusted_target)?;
                    let abs_diff = diff.abs()?;
                    let quadratic = abs_diff.le(delta)?.to_dtype(DType::F32)?;
                    let linear = abs_diff.gt(delta)?.to_dtype(DType::F32)?;
                    let quad_loss = diff
                        .sqr()?
                        .broadcast_mul(&quadratic)?
                        .broadcast_mul(&half_full)?;
                    let lin_loss = abs_diff
                        .sub(&delta_full)?
                        .broadcast_mul(&delta_full)?
                        .add(&delta_sq_half_full)?
                        .mul(&linear)?;
                    let total = quad_loss.add(&lin_loss)?;
                    Ok(total.mean_all()?)
                }
                _ => {
                    let diff = predicted.sub(&adjusted_target)?;
                    Ok(diff.abs()?.mean_all()?)
                }
            }
        };

        for epoch in 0..epochs {
            let progress = Progress::new(num_batches, &format!("[{}] Epoch {}: ", context, epoch));
            let mut batch_losses = vec![];

            training_data.chunks(batch_size).enumerate().try_for_each(
                |(_batch_idx, batch_data)| -> anyhow::Result<()> {
                    let (input_batch, target_batch) =
                        self.prepare_batch_inputs(batch_data, &modifications)?;

                    let predicted = self.forward(&input_batch)?;
                    let tgt = target_batch.clone();
                    let pred_flat = predicted.flatten_all()?;
                    let mut tgt_flat = tgt.flatten_all()?;
                    // Align target to prediction shape early to avoid shape errors.
                    if tgt_flat.elem_count() == 0 {
                        log::error!(
                            "Empty target tensor in train loop: pred shape {:?}, tgt shape {:?}",
                            pred_flat.shape(),
                            tgt_flat.shape()
                        );
                        tgt_flat = Tensor::zeros(
                            pred_flat.shape(),
                            pred_flat.dtype(),
                            pred_flat.device(),
                        )?;
                        return Err(anyhow::anyhow!(
                            "Empty target tensor after flatten in train loop"
                        ));
                    } else if tgt_flat.elem_count() == 1 && pred_flat.elem_count() > 1 {
                        tgt_flat = tgt_flat.reshape(pred_flat.shape())?;
                    } else if tgt_flat.shape() != pred_flat.shape() {
                        match tgt_flat.broadcast_as(pred_flat.shape()) {
                            Ok(b) => tgt_flat = b,
                            Err(_) => tgt_flat = tgt_flat.reshape(pred_flat.shape())?,
                        }
                    }
                    let loss = compute_loss(&pred_flat, &tgt_flat)?;
                    opt.backward_step(&loss)?;

                    // Update learning rate after optimizer step
                    opt.set_learning_rate(lr_scheduler.get_last_lr());
                    lr_scheduler.step();

                    let loss_val = loss.to_vec0::<f32>().unwrap_or(999.0);
                    batch_losses.push(loss_val);

                    let (mae, rmse, r2, acc) = match self.property_type() {
                        PropertyType::RT => {
                            let predictions = predicted.flatten_all()?.to_vec1::<f32>()?;
                            let targets = target_batch.flatten_all()?.to_vec1::<f32>()?;
                            let denorm = |v: &[f32]| -> Vec<f32> {
                                match target_norm {
                                    TargetNormalization::ZScore(mean, std) => {
                                        v.iter().map(|x| *x * std + mean).collect()
                                    }
                                    TargetNormalization::MinMax(min, max) => {
                                        v.iter().map(|x| *x * (max - min) + min).collect()
                                    }
                                    TargetNormalization::None => v.to_vec(),
                                }
                            };
                            let predictions_dn = denorm(&predictions);
                            let targets_dn = denorm(&targets);
                            (
                                Some(Metrics::mae(&predictions_dn, &targets_dn)),
                                Some(Metrics::rmse(&predictions_dn, &targets_dn)),
                                Some(Metrics::r2(&predictions_dn, &targets_dn)),
                                None,
                            )
                        }
                        PropertyType::CCS => {
                            let predictions = predicted.flatten_all()?.to_vec1::<f32>()?;
                            let targets = target_batch.flatten_all()?.to_vec1::<f32>()?;
                            let tol: Vec<f32> = targets.iter().map(|t| t * 0.02).collect();
                            (
                                Some(Metrics::mae(&predictions, &targets)),
                                Some(Metrics::rmse(&predictions, &targets)),
                                Some(Metrics::r2(&predictions, &targets)),
                                Some(Metrics::accuracy_dynamic(&predictions, &targets, &tol)),
                            )
                        }
                        _ => (None, None, None, None),
                    };

                    if track_metrics {
                        step_metrics.epochs.push(epoch);
                        step_metrics.steps.push(step_idx);
                        step_metrics
                            .learning_rates
                            .push(lr_scheduler.get_last_lr() as f64);
                        step_metrics.losses.push(loss_val);
                        step_metrics.phases.push(TrainingPhase::Train);
                        step_metrics.accuracies.push(acc);
                        step_metrics.maes.push(mae);
                        step_metrics.rmses.push(rmse);
                        step_metrics.r2s.push(r2);
                        step_metrics.precisions.push(None);
                        step_metrics.recalls.push(None);
                        step_idx += 1;
                    }

                    progress.update_description(&format!(
                        "[{}] Epoch {}: Loss: {:.4}",
                        context, epoch, loss_val
                    ));
                    progress.inc();

                    // If the loss is NaN, stop training and throw an error
                    if loss_val.is_nan() {
                        log::error!("Loss is NaN, stopping training.");
                        return Err(anyhow::anyhow!("Loss is NaN, stopping training."));
                    }

                    Ok(())
                },
            )?;

            let (avg_loss, std_loss) = compute_loss_stats(&batch_losses);

            if let Some(val_data) = validation_data {
                let val_batches =
                    (val_data.len() + validation_batch_size - 1) / validation_batch_size;

                // Ensure the model is in evaluation mode during validation so
                // layers like Dropout are disabled
                self.set_evaluation_mode();

                let val_results: Vec<(
                    f32,
                    usize,
                    f64,
                    Option<f32>,
                    Option<f32>,
                    Option<f32>,
                    Option<f32>,
                )> = val_data
                    .par_chunks(validation_batch_size)
                    .enumerate()
                    .map(|(idx, batch_data)| {
                        

                        let (input_val, target_val) =
                            self.prepare_batch_inputs(batch_data, &modifications)?;
                        let predicted = self.forward(&input_val)?;
                        let tgt = target_val.clone();
                        let pred_flat = predicted.flatten_all()?;
                        let mut tgt_flat = tgt.flatten_all()?;
                        if tgt_flat.elem_count() == 0 {
                            log::error!(
                                "Empty target tensor in val loop: pred shape {:?}, tgt shape {:?}",
                                pred_flat.shape(),
                                tgt_flat.shape()
                            );
                            tgt_flat = Tensor::zeros(
                                pred_flat.shape(),
                                pred_flat.dtype(),
                                pred_flat.device(),
                            )?;
                            return Err(anyhow::anyhow!(
                                "Empty target tensor after flatten in val loop"
                            ));
                        } else if tgt_flat.elem_count() == 1 && pred_flat.elem_count() > 1 {
                            tgt_flat = tgt_flat.reshape(pred_flat.shape())?;
                        } else if tgt_flat.shape() != pred_flat.shape() {
                            match tgt_flat.broadcast_as(pred_flat.shape()) {
                                Ok(b) => tgt_flat = b,
                                Err(_) => tgt_flat = tgt_flat.reshape(pred_flat.shape())?,
                            }
                        }
                        let val_loss = compute_loss(&pred_flat, &tgt_flat)?;
                        let loss_val = val_loss.to_vec0::<f32>()?;

                        let (mae, rmse, r2, acc) = match self.property_type() {
                            PropertyType::RT => {
                                let predictions = predicted.flatten_all()?.to_vec1::<f32>()?;
                                let targets = target_val.flatten_all()?.to_vec1::<f32>()?;
                                let denorm = |v: &[f32]| -> Vec<f32> {
                                    match target_norm {
                                        TargetNormalization::ZScore(mean, std) => {
                                            v.iter().map(|x| *x * std + mean).collect()
                                        }
                                        TargetNormalization::MinMax(min, max) => {
                                            v.iter().map(|x| *x * (max - min) + min).collect()
                                        }
                                        TargetNormalization::None => v.to_vec(),
                                    }
                                };
                                let predictions_dn = denorm(&predictions);
                                let targets_dn = denorm(&targets);
                                (
                                    Some(Metrics::mae(&predictions_dn, &targets_dn)),
                                    Some(Metrics::rmse(&predictions_dn, &targets_dn)),
                                    Some(Metrics::r2(&predictions_dn, &targets_dn)),
                                    None,
                                )
                            }
                            PropertyType::CCS => {
                                let predictions = predicted.flatten_all()?.to_vec1::<f32>()?;
                                let targets = target_val.flatten_all()?.to_vec1::<f32>()?;
                                let tol: Vec<f32> = targets.iter().map(|t| t * 0.02).collect();
                                (
                                    Some(Metrics::mae(&predictions, &targets)),
                                    Some(Metrics::rmse(&predictions, &targets)),
                                    Some(Metrics::r2(&predictions, &targets)),
                                    Some(Metrics::accuracy_dynamic(&predictions, &targets, &tol)),
                                )
                            }
                            _ => (None, None, None, None),
                        };

                        Ok((
                            loss_val,
                            idx,
                            lr_scheduler.get_last_lr(),
                            acc,
                            mae,
                            rmse,
                            r2,
                        ))
                    })
                    .collect::<Result<_>>()?;

                // Restore training mode after validation so subsequent training
                // steps keep Dropout and other training-only behavior enabled.
                self.set_training_mode();

                if track_metrics {
                    for (val_loss, idx, lr, acc, mae, rmse, r2) in &val_results {
                        step_metrics.epochs.push(epoch);
                        step_metrics.steps.push(val_step_idx + idx);
                        step_metrics.learning_rates.push(*lr);
                        step_metrics.losses.push(*val_loss);
                        step_metrics.phases.push(TrainingPhase::Validation);
                        step_metrics.accuracies.push(*acc);
                        step_metrics.maes.push(*mae);
                        step_metrics.rmses.push(*rmse);
                        step_metrics.r2s.push(*r2);
                        step_metrics.precisions.push(None);
                        step_metrics.recalls.push(None);
                    }
                    val_step_idx += val_results.len();
                }

                let val_losses: Vec<f32> = val_results
                    .iter()
                    .map(|(loss, _, _, _, _, _, _)| *loss)
                    .collect();
                let (avg_val_loss, std_val_loss): (f32, f32) = compute_loss_stats(&val_losses);

                epoch_losses.push((
                    epoch,
                    avg_loss,
                    Some(avg_val_loss),
                    std_loss,
                    Some(std_val_loss),
                ));

                progress.update_description(&format!(
                    "Epoch {}: Avg. Train Loss: {:.4} (±{:.4}) | Avg. Val. Loss: {:.4} (±{:.4})",
                    epoch, avg_loss, std_loss, avg_val_loss, std_val_loss
                ));
                progress.finish();

                if avg_val_loss < best_val_loss {
                    best_val_loss = avg_val_loss;
                    epochs_without_improvement = 0;
                    if save_checkpoints {
                        self.save_epoch_checkpoint(epoch, "val")?;
                    }
                } else {
                    epochs_without_improvement += 1;
                    if epochs_without_improvement >= early_stopping_patience {
                        info!("Early stopping triggered after {} epochs without validation loss improvement.", early_stopping_patience);
                        return Ok(step_metrics);
                    }
                    if save_checkpoints {
                        self.save_epoch_checkpoint(epoch, "train")?;
                    }
                }
            } else {
                epoch_losses.push((epoch, avg_loss, None, std_loss, None));
                progress.update_description(&format!(
                    "Epoch {}: Avg. Train Loss: {:.4} (±{:.4})",
                    epoch, avg_loss, std_loss
                ));
                progress.finish();
                if save_checkpoints {
                    self.save_epoch_checkpoint(epoch, "train")?;
                }
            }
        }

        Ok(step_metrics)
    }

    /// Fine-tune the model on new data using the main [`ModelInterface::train`] method.
    /// This is a wrapper that disables validation and early stopping.
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
        target_norm: TargetNormalization,
        warmup_fraction: Option<f64>,
    ) -> Result<()> {
        let _metrics = self.train(
            training_data,
            None, // No validation data
            modifications,
            batch_size,
            batch_size, // Validation batch size is same but unused
            learning_rate,
            epochs,
            usize::MAX, // Disable early stopping
            "fine-tuning",
            false, // No checkpoints
            false, // No metrics
            target_norm,
            None,
            warmup_fraction,
        )?;

        Ok(())
    }

    /// Perform inference over a batch of peptides.
    fn inference(
        &self,
        inference_data: &Vec<PeptideData>,
        batch_size: usize,
        modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        target_norm: TargetNormalization,
    ) -> Result<Vec<PeptideData>> {
        let num_batches = (inference_data.len() + batch_size - 1) / batch_size;
        info!(
            "Performing inference on {} peptide features ({} batches)",
            inference_data.len(),
            num_batches
        );

        let progress = Progress::new(inference_data.len(), "[inference] Batch:");
        let mut result: Vec<Option<PeptideData>> = vec![None; inference_data.len()];



        inference_data
            .par_chunks(batch_size)
            .enumerate()
            .map(|(batch_idx, batch_data)| {
                let start_idx = batch_idx * batch_size;

                // Extract input features only (ignore targets)
                let (input_tensor, _) = self.prepare_batch_inputs(batch_data, &modifications)?;

                // Now run model forward.
                let predicted = self.forward(&input_tensor)?;


                let predictions = predicted.to_vec1::<f32>()?;


                let updated = predictions
                    .into_iter()
                    .enumerate()
                    .map(|(i, pred)| {
                        let mut peptide = batch_data[i].clone();
                        let value = match target_norm {
                            TargetNormalization::ZScore(mean, std) => pred * std + mean,
                            TargetNormalization::MinMax(min, max) => pred * (max - min) + min,
                            TargetNormalization::None => pred,
                        };
                        match self.property_type() {
                            PropertyType::RT => peptide.retention_time = Some(value),
                            PropertyType::CCS => peptide.ccs = Some(value),
                            _ => {}
                        }
                        (start_idx + i, peptide)
                    })
                    .collect::<Vec<_>>();

                Ok(updated)
            })
            .collect::<Result<Vec<Vec<(usize, PeptideData)>>>>()?
            .into_iter()
            .flatten()
            .for_each(|(idx, peptide)| {
                result[idx] = Some(peptide);
                progress.inc();
            });

        progress.finish();
        Ok(result.into_iter().flatten().collect())
    }

    /// Extract encoded input and target tensor for a batch of peptides.
    fn prepare_batch_inputs(
        &self,
        batch_data: &[PeptideData],
        _modifications: &HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
    ) -> Result<(Tensor, Tensor)> {
        let batch: PeptideBatchData = batch_data.into();

        let naked_sequences = &batch.naked_sequence;
        let mods = &batch.mods;
        let mod_sites = &batch.mod_sites;

        let charges = if batch.charges.iter().all(|c| c.is_some()) {
            Some(batch.charges.iter().map(|c| c.unwrap()).collect::<Vec<_>>())
        } else {
            None
        };

        let nces = if batch.nces.iter().all(|n| n.is_some()) {
            Some(batch.nces.iter().map(|n| n.unwrap()).collect::<Vec<_>>())
        } else {
            None
        };

        let instruments = if batch.instruments.iter().all(|i| i.is_some()) {
            Some(batch.instruments.clone())
        } else {
            None
        };

        let input_batch = self
            .encode_peptides(naked_sequences, mods, mod_sites, charges, nces, instruments)?
            .to_device(self.get_device())?;

        let target_tensor = match self.property_type() {
            PropertyType::RT => {
                let mut target_values: Vec<f32> = batch
                    .retention_times
                    .iter()
                    .map(|v| v.unwrap_or(0.0))
                    .collect();
                if target_values.is_empty() {
                    let batch_size = batch_data.len();
                    log::warn!(
                        "prepare_batch_inputs: empty RT targets for batch size {}; filling zeros",
                        batch_size
                    );
                    target_values = vec![0.0; batch_size];
                }
                Tensor::new(target_values, &self.get_device())?
            }
            PropertyType::CCS => {
                let target_values: Vec<f32> = batch.ccs.iter().map(|v| v.unwrap_or(0.0)).collect();
                Tensor::new(target_values, &self.get_device())?
            }
            PropertyType::MS2 => {
                let batch_size = batch.ms2_intensities.len();
                if batch_size == 0 {
                    anyhow::bail!("MS2 batch is empty; cannot build target tensor");
                }

                // MS2 intensities are per-fragment (len - 1). Pad to the max length in the batch
                // so the target shape matches the padded model output.
                let max_frag_len = batch
                    .naked_sequence
                    .iter()
                    .map(|seq| seq.len().saturating_sub(1))
                    .max()
                    .unwrap_or(0);
                if max_frag_len == 0 {
                    anyhow::bail!("Could not determine fragment length for MS2 targets");
                }

                let mut targets = vec![0f32; batch_size * max_frag_len * 8];
                for (i, opt_peptide) in batch.ms2_intensities.iter().enumerate() {
                    let peptide = opt_peptide.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Missing MS2 intensities for peptide at index {i}")
                    })?;
                    let expected_len = batch.naked_sequence[i].len().saturating_sub(1);
                    if peptide.len() != expected_len {
                        log::warn!(
                            "MS2 intensities length {} does not match expected {} for peptide {}",
                            peptide.len(),
                            expected_len,
                            i
                        );
                    }

                    for (row_idx, row) in peptide.iter().enumerate() {
                        if row_idx >= max_frag_len {
                            break;
                        }
                        if row.len() != 8 {
                            log::warn!(
                                "MS2 intensity row {} for peptide {} has len {}, expected 8",
                                row_idx,
                                i,
                                row.len()
                            );
                        }
                        let cols = row.len().min(8);
                        let base = (i * max_frag_len + row_idx) * 8;
                        targets[base..base + cols].copy_from_slice(&row[..cols]);
                    }
                }

                Tensor::from_vec(
                    targets,
                    (batch_size, max_frag_len, 8),
                    &self.get_device(),
                )?
            }
        };

        Ok((input_batch, target_tensor))
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

    fn get_mod_to_feature(&self) -> &HashMap<Arc<[u8]>, Vec<f32>>;

    fn get_min_pred_intensity(&self) -> f32;

    fn get_mut_varmap(&mut self) -> &mut VarMap;

    fn print_summary(&self);
    fn print_weights(&self);

    /// Save model weights to a file in safetensors format.
    fn save(&mut self, path: &str) -> Result<()> {
        let p = PathBuf::from(path);
        info!("Saving {} model weights to: {:?}", self.get_model_arch(), p);
    self.get_mut_varmap().save(&p)?;
        Ok(())
    }

    /// Save epoch checkpoint and delete prior checkpoint
    fn save_epoch_checkpoint(&mut self, epoch: usize, ctx: &str) -> Result<()> {
        let insert_ctx = match ctx {
            "train" => "_",
            "val" => "_best_val_",
            _ => panic!("Invalid context for saving checkpoint. Must be 'train' or 'val'."),
        };

        // Check if the prior checkpoint exists, if it does delete it
        let checkpoint_path = format!(
            "redeem_{}{}ckpt_model_epoch_{}.safetensors",
            self.get_model_arch(),
            insert_ctx,
            epoch - 1
        );
        if PathBuf::from(&checkpoint_path).exists() {
            std::fs::remove_file(&checkpoint_path)?;
        }
        // Save the current checkpoint
        let checkpoint_path = format!(
            "redeem_{}{}ckpt_model_epoch_{}.safetensors",
            self.get_model_arch(),
            insert_ctx,
            epoch
        );
        self.get_mut_varmap().save(&checkpoint_path)?;
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
#[derive(Clone)]
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
    pub rt_model: Option<RTModelWrapper>,

    /// Optional collision cross-section prediction model.
    pub ccs_model: Option<CCSModelWrapper>,

    /// Optional MS2 intensity prediction model.
    pub ms2_model: Option<MS2ModelWrapper>,
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
    /// models.rt_model = Some(RTModelWrapper::new());
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
    /// models.rt_model = Some(RTModelWrapper::new());
    /// assert!(models.is_not_empty());
    /// ```
    pub fn is_not_empty(&self) -> bool {
        self.rt_model.is_some() || self.ccs_model.is_some() || self.ms2_model.is_some()
    }
}
