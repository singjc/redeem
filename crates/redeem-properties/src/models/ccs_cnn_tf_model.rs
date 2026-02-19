use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Dropout, Module, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::building_blocks::building_blocks::DecoderSmall;
use crate::building_blocks::building_blocks::{
    DecoderHead, DecoderLinear, DecoderMLP, Encoder26aaModChargeCnnTransformerAttnSum,
    MOD_FEATURE_SIZE,
};
use crate::models::model_interface::{
    create_var_map, load_tensors_from_model, ModelInterface, PropertyType,
};
use crate::utils::peptdeep_utils::{
    load_mod_to_feature_arc, parse_model_constants, ModelConstants,
};
use crate::utils::utils::get_tensor_stats;

// Constants
const CHARGE_FACTOR: f64 = 0.1;
const NCE_FACTOR: f64 = 0.01;

// Main Model Struct

#[derive(Clone)]
/// Represents an CNN-TF Collision Cross Section model.
pub struct CCSCNNTFModel {
    var_store: VarBuilder<'static>,
    varmap: VarMap,
    constants: ModelConstants,
    device: Device,
    mod_to_feature: HashMap<Arc<[u8]>, Vec<f32>>,
    dropout: Dropout,
    ccs_encoder: Encoder26aaModChargeCnnTransformerAttnSum,
    ccs_decoder: DecoderHead,
    is_training: bool,
}

impl CCSCNNTFModel {
    /// Inherent constructor supporting head selection options.
    pub fn new_untrained_with_options(
        device: Device,
        head_type: &str,
        head_learnable_scaler: bool,
    ) -> Result<Self> {
        let mut varmap = VarMap::new();
        let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        log::trace!("[CCSCNNTFModel] Initializing ccs_encoder");
        let ccs_encoder = Encoder26aaModChargeCnnTransformerAttnSum::new(
            &varbuilder.pp("ccs_encoder"),
            8,   // mod_hidden_dim
            128, // hidden_dim
            256, // ff_dim
            4,   // num_heads
            2,   // num_layers
            100, // max_len
            0.1, // dropout_prob
            &device,
        )?;

        log::trace!("[CCSCNNTFModel] Initializing ccs_decoder (selected head for training)");
        let ccs_decoder = match head_type {
            "small" => DecoderHead::Small(DecoderSmall::new(
                129,
                1,
                &varbuilder.pp("ccs_decoder"),
                head_learnable_scaler,
            )?),
            "linear" => {
                DecoderHead::Linear(DecoderLinear::new(129, 1, &varbuilder.pp("ccs_decoder"))?)
            }
            _ => DecoderHead::MLP(DecoderMLP::new(129, 1, &varbuilder.pp("ccs_decoder"))?),
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
            ccs_encoder,
            ccs_decoder,
            is_training: true,
        })
    }
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl Send for CCSCNNTFModel {}
unsafe impl Sync for CCSCNNTFModel {}

// Core Model Implementation

impl ModelInterface for CCSCNNTFModel {
    fn property_type(&self) -> PropertyType {
        PropertyType::CCS
    }

    fn model_arch(&self) -> &'static str {
        "ccs_cnn_tf"
    }

    fn new_untrained(device: Device) -> Result<Self> {
        // Default delegates to the options-based constructor using the previous default (MLP head).
        Self::new_untrained_with_options(device, "mlp", false)
    }

    /// Create a new CCSCNNTFModel from the given model and constants files.
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
        let var_store = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let constants = match constants_path {
            Some(path) => parse_model_constants(path.as_ref().to_str().unwrap())?,
            None => ModelConstants::default(),
        };

        let mod_to_feature = load_mod_to_feature_arc(&constants)?;
        let dropout = Dropout::new(0.1);

        // Use a scoped view of the varstore for the encoder so that
        // internal keys resolve relative to the encoder scope. This
        // avoids accidental key-scope mismatches (root vs scoped names)
        // that can lead to tensors being read as zeros when the
        // top-level VarBuilder is used directly.
        let ccs_encoder = Encoder26aaModChargeCnnTransformerAttnSum::from_varstore(
            &var_store.pp("ccs_encoder"),
            8,   // mod_hidden_dim
            128, // hidden_dim
            256, // ff_dim
            4,   // num_heads
            2,   // num_layers
            100, // max_len (set appropriately for your sequence length)
            0.1, // dropout_prob
            // names are now relative to the `ccs_encoder` scope
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
            // transformer prefix is local within the ccs_encoder scope
            "input_transformer",
            vec!["attn_sum.attn.0.weight"],
            &device,
        )?;

        // Detect decoder head layout in varstore and load the appropriate head.
        let ccs_decoder = if var_store.get((4, 129), "ccs_decoder.nn.0.weight").is_ok() {
            // Use a scoped view of the var_store for the decoder so relative
            // names like "nn.0.weight" resolve correctly within the
            // `ccs_decoder` prefix. Using the root var_store here previously
            // caused the decoder submodules to fall back to zeros when keys
            // were looked up as local names.
            DecoderHead::Small(DecoderSmall::from_varstore(
                &var_store.pp("ccs_decoder"),
                129,
                1,
                vec![
                    "ccs_decoder.nn.0.weight",
                    "ccs_decoder.nn.1.weight",
                    "ccs_decoder.nn.2.weight",
                ],
                vec!["ccs_decoder.nn.0.bias", "ccs_decoder.nn.2.bias"],
            )?)
        } else if var_store.get((64, 129), "ccs_decoder.nn.0.weight").is_ok() {
            if var_store.get((16, 64), "ccs_decoder.nn.2.weight").is_ok() {
                DecoderHead::MLP(DecoderMLP::from_varstore(
                    &var_store.pp("ccs_decoder"),
                    129,
                    1,
                    vec![
                        "ccs_decoder.nn.0.weight",
                        "ccs_decoder.nn.1.weight",
                        "ccs_decoder.nn.2.weight",
                    ],
                    vec!["ccs_decoder.nn.0.bias", "ccs_decoder.nn.2.bias"],
                )?)
            } else {
                DecoderHead::Linear(DecoderLinear::from_varstore(
                    &var_store.pp("ccs_decoder"),
                    129,
                    1,
                    vec![
                        "ccs_decoder.nn.0.weight",
                        "ccs_decoder.nn.1.weight",
                        "ccs_decoder.nn.2.weight",
                    ],
                    vec!["ccs_decoder.nn.0.bias", "ccs_decoder.nn.2.bias"],
                )?)
            }
        } else {
            return Err(anyhow::anyhow!(
                "Unrecognized decoder layout in model varstore"
            ));
        };

        // Ensure nested encoder transformer is in eval mode when constructed
        // from a pretrained varstore so dropout and training-only layers
        // are disabled by default (prevents subtle mismatches).
        let mut model = Self {
            var_store,
            varmap,
            constants,
            device,
            mod_to_feature,
            dropout,
            ccs_encoder,
            ccs_decoder,
            is_training: false,
        };
        model.ccs_encoder.set_training(false);
        Ok(model)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (_batch_size, _seq_len, _) = xs.shape().dims3()?;

        // Separate input into aa_indices, mod_x, charge
        let start_mod_x = 1;
        let start_charge = start_mod_x + MOD_FEATURE_SIZE;

        let aa_indices_out = xs.i((.., .., 0))?;
        let (mean, min, max) = get_tensor_stats(&aa_indices_out)?;
        log::debug!("[CCSCNNTFModel] aa_indices_out stats - min: {min}, max: {max}, mean: {mean}");

        let mod_x_out = xs.i((.., .., start_mod_x..start_mod_x + MOD_FEATURE_SIZE))?;
        let charge_out = xs.i((.., 0..1, start_charge..start_charge + 1))?;
        let charge_out = charge_out.squeeze(2)?;

        let x = self
            .ccs_encoder
            .forward(&aa_indices_out, &mod_x_out, &charge_out)?;

        let x = self.dropout.forward(&x, self.is_training)?;

        let x = Tensor::cat(&[x, charge_out], 1)?;

        let x = self.ccs_decoder.forward(&x)?;

        Ok(x.squeeze(1)?)
    }

    /// Set model to evaluation mode for inference
    /// This disables dropout and other training-specific layers.
    fn set_evaluation_mode(&mut self) {
        // println!("Setting evaluation mode");
        self.is_training = false;
    }

    /// Set model to training mode for training
    /// This enables dropout and other training-specific layers.
    fn set_training_mode(&mut self) {
        self.is_training = true;
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
        println!("CCSModel Summary:");
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
        todo!("Implement print_weights for CCSCNNTFModel");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ccs_cnn_tf_model::CCSCNNTFModel;
    use crate::models::model_interface::ModelInterface;
    use candle_core::Device;
    use std::path::PathBuf;

    #[test]
    fn test_encode_peptides() {
        let device = Device::Cpu;
        let model = Box::new(CCSCNNTFModel::new_untrained(device.clone()).unwrap());

        let seq = Arc::from(b"AGHCEWQMKYR".to_vec().into_boxed_slice());
        let mods = Arc::from(
            b"Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M"
                .to_vec()
                .into_boxed_slice(),
        );
        let mod_sites = Arc::from(b"0;4;8".to_vec().into_boxed_slice());
        let charge = Some(2);
        let nce = Some(20);
        let instrument = Some(Arc::from(b"QE".to_vec().into_boxed_slice()));

        let result =
            model.encode_peptide(&seq, &mods, &mod_sites, charge, nce, instrument.as_ref());

        println!("{:?}", result);
        assert!(result.is_ok());
    }
}
