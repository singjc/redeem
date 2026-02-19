use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Dropout, Module, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::building_blocks::building_blocks::DecoderSmall;
use crate::building_blocks::building_blocks::{
    DecoderHead, DecoderLinear, DecoderMLP, Encoder26aaModCnnTransformerAttnSum, MOD_FEATURE_SIZE,
};
use crate::models::model_interface::{
    create_var_map, load_tensors_from_model, ModelInterface, PropertyType,
};
use crate::utils::peptdeep_utils::{
    load_mod_to_feature_arc, parse_model_constants, ModelConstants,
};
use crate::utils::utils::get_tensor_stats;

// Main Model Struct

#[derive(Clone)]
/// Represents an CNN-TF Retention Time model.
pub struct RTCNNTFModel {
    var_store: VarBuilder<'static>,
    varmap: VarMap,
    constants: ModelConstants,
    device: Device,
    mod_to_feature: HashMap<Arc<[u8]>, Vec<f32>>,
    dropout: Dropout,
    rt_encoder: Encoder26aaModCnnTransformerAttnSum,
    rt_decoder: DecoderHead,
    is_training: bool,
}

impl RTCNNTFModel {
    /// Inherent constructor that supports selecting the head type and optional
    /// learnable scalar. Kept separate from the trait impl so it's an inherent
    /// method (callable as `RTCNNTFModel::new_untrained_with_options`).
    pub fn new_untrained_with_options(
        device: Device,
        head_type: &str,
        head_learnable_scaler: bool,
    ) -> Result<Self> {
        let mut varmap = VarMap::new();
        let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        log::trace!("[RTCNNTFModel] Initializing rt_encoder");
        let rt_encoder = Encoder26aaModCnnTransformerAttnSum::new(
            &varbuilder.pp("rt_encoder"),
            8,    // mod_hidden_dim
            192,  // hidden_dim
            768,  // ff_dim
            4,    // num_heads
            2,    // num_layers
            100,  // max_len
            0.05, // dropout_prob
            &device,
        )?;

        log::trace!("[RTCNNTFModel] Initializing rt_decoder (selected head for training)");
        let rt_decoder = match head_type {
            "small" => DecoderHead::Small(DecoderSmall::new(
                192,
                1,
                &varbuilder.pp("rt_decoder"),
                head_learnable_scaler,
            )?),
            "linear" => {
                DecoderHead::Linear(DecoderLinear::new(192, 1, &varbuilder.pp("rt_decoder"))?)
            }
            _ => DecoderHead::MLP(DecoderMLP::new(192, 1, &varbuilder.pp("rt_decoder"))?),
        };

        let constants = ModelConstants::default();
        let mod_to_feature = load_mod_to_feature_arc(&constants)?;

        Ok(Self {
            var_store: varbuilder,
            varmap,
            constants,
            device,
            mod_to_feature,
            dropout: Dropout::new(0.1),
            rt_encoder,
            rt_decoder,
            is_training: true,
        })
    }
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl Send for RTCNNTFModel {}
unsafe impl Sync for RTCNNTFModel {}

// Core Model Implementation

impl ModelInterface for RTCNNTFModel {
    fn property_type(&self) -> PropertyType {
        PropertyType::RT
    }

    fn model_arch(&self) -> &'static str {
        "rt_cnn_tf"
    }

    /// Create a new RTCNNTFModel to train
    fn new_untrained(device: Device) -> Result<Self> {
        // Default behavior preserves previous behavior: use MLP head without learnable scaler.
        Self::new_untrained_with_options(device, "mlp", false)
    }

    /// Create a new RTCNNTFModel from the given pretrained model and constants files.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: Option<P>,
        _fixed_sequence_len: usize,
        _num_frag_types: usize,
        _num_modloss_types: usize,
        _mask_modloss: bool,
        device: Device,
    ) -> Result<Self> {
        let tensor_data = load_tensors_from_model(model_path.as_ref(), &device)?;

        let mut varmap = candle_nn::VarMap::new();
        create_var_map(&mut varmap, tensor_data, &device)?;
        // varmap keys are available in `varmap` if needed;
        log::debug!(
            "VarMap populated with {} entries",
            varmap.data().lock().map(|m| m.len()).unwrap_or(0)
        );
        let var_store = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let constants = match constants_path {
            Some(path) => parse_model_constants(path.as_ref().to_str().unwrap())?,
            None => ModelConstants::default(),
        };

        let mod_to_feature = load_mod_to_feature_arc(&constants)?;
        let dropout = Dropout::new(0.1);

        // Pass a scoped VarBuilder for the encoder root so internal
        // varstore.get("proj_cnn_to_transformer.weight") resolves locally
        // without relying on fallback-qualified names. Previous runs showed
        // the encoder observing a zero tensor when given the root VarBuilder;
        // using the scoped pp("rt_encoder") matches how the new() path
        // constructs encoders and should avoid scope mismatches.
        // Use a scoped varbuilder view (`pp("rt_encoder")`) and pass
        // relative names / local prefixes to `from_varstore` so that the
        // encoder resolves keys consistently within the same scope.
        let rt_encoder = Encoder26aaModCnnTransformerAttnSum::from_varstore(
            &var_store.pp("rt_encoder"),
            8,    // mod_hidden_dim
            192,  // hidden_dim
            768,  // ff_dim
            4,    // num_heads
            2,    // num_layers
            100,  // max_len (sequence length)
            0.05, // dropout_prob
            // names are relative to the rt_encoder scope now
            vec!["mod_nn.nn.weight"],
            vec![
                "input_cnn.cnn_short.weight",
                "input_cnn.cnn_medium.weight",
                "input_cnn.cnn_long.weight",
            ],
            vec![
                "input_cnn.cnn_short.bias",
                "input_cnn.cnn_medium.bias",
                "input_cnn.cnn_long.bias",
            ],
            // transformer prefix is local within the rt_encoder scope
            "input_transformer",
            vec!["attn_sum.attn.0.weight"],
            &device,
        )?;

        // Optional manual head override via env var; otherwise auto-detect by shapes.
        let forced_head = std::env::var("REDEEM_RT_DECODER_HEAD").ok();

        // Detect decoder head layout in varstore and load the appropriate head to preserve
        // compatibility with checkpoints saved from different training runs.
        let rt_decoder = if let Some(force) = forced_head.as_deref() {
            match force {
                "small" => {
                    log::info!("[RTCNNTFModel] Forcing decoder head: small (4->out)");
                    DecoderHead::Small(DecoderSmall::from_varstore(
                        &var_store.pp("rt_decoder"),
                        192,
                        1,
                        vec![
                            "rt_decoder.nn.0.weight",
                            "rt_decoder.nn.1.weight",
                            "rt_decoder.nn.2.weight",
                        ],
                        vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"],
                    )?)
                }
                "mlp" => {
                    log::info!("[RTCNNTFModel] Forcing decoder head: MLP (64->16->out)");
                    DecoderHead::MLP(DecoderMLP::from_varstore(
                        &var_store.pp("rt_decoder"),
                        192,
                        1,
                        vec![
                            "rt_decoder.nn.0.weight",
                            "rt_decoder.nn.1.weight",
                            "rt_decoder.nn.2.weight",
                        ],
                        vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"],
                    )?)
                }
                "linear" => {
                    log::info!("[RTCNNTFModel] Forcing decoder head: linear (64->out)");
                    DecoderHead::Linear(DecoderLinear::from_varstore(
                        &var_store.pp("rt_decoder"),
                        192,
                        1,
                        vec![
                            "rt_decoder.nn.0.weight",
                            "rt_decoder.nn.1.weight",
                            "rt_decoder.nn.2.weight",
                        ],
                        vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"],
                    )?)
                }
                other => {
                    return Err(anyhow::anyhow!(
                        "Unsupported forced decoder head '{}'. Use one of: small, mlp, linear",
                        other
                    ))
                }
            }
        } else if var_store.get((4, 192), "rt_decoder.nn.0.weight").is_ok() {
            // Small head layout
            log::info!("[RTCNNTFModel] Detected decoder head: small (4->out)");
            DecoderHead::Small(DecoderSmall::from_varstore(
                &var_store.pp("rt_decoder"),
                192,
                1,
                vec![
                    "rt_decoder.nn.0.weight",
                    "rt_decoder.nn.1.weight",
                    "rt_decoder.nn.2.weight",
                ],
                vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"],
            )?)
        } else if var_store.get((64, 192), "rt_decoder.nn.0.weight").is_ok() {
            // Could be MLP (has nn.2 of shape (16,64)) or legacy linear (nn.2 of shape (1,64) / out_features)
            if var_store.get((16, 64), "rt_decoder.nn.2.weight").is_ok() {
                log::info!("[RTCNNTFModel] Detected decoder head: MLP (64->16->out)");
                DecoderHead::MLP(DecoderMLP::from_varstore(
                    &var_store.pp("rt_decoder"),
                    192,
                    1,
                    vec![
                        "rt_decoder.nn.0.weight",
                        "rt_decoder.nn.1.weight",
                        "rt_decoder.nn.2.weight",
                    ],
                    vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"],
                )?)
            } else {
                log::info!("[RTCNNTFModel] Detected decoder head: linear (64->out)");
                DecoderHead::Linear(DecoderLinear::from_varstore(
                    &var_store.pp("rt_decoder"),
                    192,
                    1,
                    vec![
                        "rt_decoder.nn.0.weight",
                        "rt_decoder.nn.1.weight",
                        "rt_decoder.nn.2.weight",
                    ],
                    vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"],
                )?)
            }
        } else {
            return Err(anyhow::anyhow!(
                "Unrecognized decoder layout in model varstore"
            ));
        };

        // Ensure the encoder's internal transformer is in evaluation mode
        // when constructing from a pretrained varstore so that dropout and
        // other training-specific layers are disabled by default. This
        // prevents a subtle mismatch where `is_training` on the outer model
        // was false but the nested `SeqTransformer` retained its default
        // `training=true` state (which would keep dropout active).
        let mut model = Self {
            var_store,
            varmap,
            constants,
            device,
            mod_to_feature,
            dropout,
            rt_encoder,
            rt_decoder,
            is_training: false,
        };
        model.rt_encoder.set_training(false);
        Ok(model)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let aa_indices_out = xs.i((.., .., 0))?;
        let (mean, min, max) = get_tensor_stats(&aa_indices_out)?;
        log::debug!("[RTCNNTFModel] aa_indices_out stats - min: {min}, max: {max}, mean: {mean}");
        let mod_x_out = xs.i((.., .., 1..1 + MOD_FEATURE_SIZE))?;

        let x = self.rt_encoder.forward(&aa_indices_out, &mod_x_out)?;

        log::debug!("Encoder output shape: {:?}", x.clone().shape());

        let x = self.dropout.forward(&x, self.is_training)?;

        log::debug!(
            "Decoder input shape (post-dropout): {:?}",
            x.clone().shape()
        );

        let x = self.rt_decoder.forward(&x)?;

        Ok(x.squeeze(1)?)
    }

    /// Set model to evaluation mode for inference
    /// This disables dropout and other training-specific layers.
    fn set_evaluation_mode(&mut self) {
        // println!("Setting evaluation mode");
        self.is_training = false;
        self.rt_encoder.set_training(false);
    }

    /// Set model to training mode for training
    /// This enables dropout and other training-specific layers.
    fn set_training_mode(&mut self) {
        self.is_training = true;
        self.rt_encoder.set_training(true);
    }

    fn get_property_type(&self) -> String {
        self.property_type().clone().as_str().to_string()
    }

    fn get_model_arch(&self) -> String {
        self.model_arch().to_string()
    }

    fn get_device(&self) -> &Device {
        &self.device
    }

    fn get_mod_element_count(&self) -> usize {
        self.constants.mod_elements.len()
    }

    fn get_mod_to_feature(&self) -> &HashMap<Arc<[u8]>, Vec<f32>> {
        &self.mod_to_feature
    }

    fn get_min_pred_intensity(&self) -> f32 {
        unimplemented!(
            "Method not implemented for architecture: {}",
            self.model_arch()
        )
    }

    fn get_mut_varmap(&mut self) -> &mut VarMap {
        &mut self.varmap
    }

    /// Print a summary of the model's constants.
    fn print_summary(&self) {
        println!("RTModel Summary:");
        println!(
            "AA Embedding Size: {}",
            self.constants.aa_embedding_size.unwrap()
        );
        println!("Charge Factor: {:?}", self.constants.charge_factor);
        println!("Instruments: {:?}", self.constants.instruments);
        println!("Max Instrument Num: {}", self.constants.max_instrument_num);
        println!("Mod Elements: {:?}", self.constants.mod_elements);
        println!("NCE Factor: {:?}", self.constants.nce_factor);
    }

    /// Print the model's weights.
    fn print_weights(&self) {
        todo!("Implement print_weights for RTCNNTFModel");
    }
}
