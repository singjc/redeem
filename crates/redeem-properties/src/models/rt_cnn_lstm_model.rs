use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Dropout, Module, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::building_blocks::building_blocks::{
    DecoderLinear, Encoder26aaModCnnLstmAttnSum, MOD_FEATURE_SIZE,
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
/// Represents an AlphaPeptDeep CNN-LSTM Retention Time model.
pub struct RTCNNLSTMModel {
    var_store: VarBuilder<'static>,
    varmap: VarMap,
    constants: ModelConstants,
    device: Device,
    mod_to_feature: HashMap<Arc<[u8]>, Vec<f32>>,
    dropout: Dropout,
    rt_encoder: Encoder26aaModCnnLstmAttnSum,
    rt_decoder: DecoderLinear,
    is_training: bool,
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl Send for RTCNNLSTMModel {}
unsafe impl Sync for RTCNNLSTMModel {}

// Core Model Implementation

impl ModelInterface for RTCNNLSTMModel {
    fn property_type(&self) -> PropertyType {
        PropertyType::RT
    }

    fn model_arch(&self) -> &'static str {
        "rt_cnn_lstm"
    }

    fn new_untrained(_device: Device) -> Result<Self> {
        unimplemented!("Untrained model creation is not implemented for this architecture.");
    }

    /// Create a new RTCNNLSTMModel from the given model and constants files.
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

        // Load the mod_to_feature mapping
        let mod_to_feature = load_mod_to_feature_arc(&constants)?;

        // Encoder
        let dropout = Dropout::new(0.1);

        let rt_encoder = Encoder26aaModCnnLstmAttnSum::from_varstore(
            &var_store,
            8,
            128,
            2,
            vec!["rt_encoder.mod_nn.nn.weight"],
            vec![
                "rt_encoder.input_cnn.cnn_short.weight",
                "rt_encoder.input_cnn.cnn_medium.weight",
                "rt_encoder.input_cnn.cnn_long.weight",
            ],
            vec![
                "rt_encoder.input_cnn.cnn_short.bias",
                "rt_encoder.input_cnn.cnn_medium.bias",
                "rt_encoder.input_cnn.cnn_long.bias",
            ],
            "rt_encoder.hidden_nn",
            vec!["rt_encoder.attn_sum.attn.0.weight"],
        )
        .unwrap();

        let rt_decoder = DecoderLinear::from_varstore(
            &var_store,
            256,
            1,
            vec![
                "rt_decoder.nn.0.weight",
                "rt_decoder.nn.1.weight",
                "rt_decoder.nn.2.weight",
            ],
            vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"],
        )
        .unwrap();

        Ok(Self {
            var_store,
            varmap,
            constants,
            device,
            mod_to_feature,
            dropout,
            rt_encoder,
            rt_decoder,
            // Default to evaluation mode for inference use cases.
            is_training: false,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (_batch_size, _seq_len, _) = xs.shape().dims3()?;

        let aa_indices_out = xs.i((.., .., 0))?;
        let (mean, min, max) = get_tensor_stats(&aa_indices_out)?;
        log::debug!("[RTCNNLSTMModel] aa_indices_out stats - min: {min}, max: {max}, mean: {mean}");
        let mod_x_out = xs.i((.., .., 1..1 + MOD_FEATURE_SIZE))?;

        let x = self.rt_encoder.forward(&aa_indices_out, &mod_x_out)?;

        let x = self.dropout.forward(&x, self.is_training)?;

        let x = self.rt_decoder.forward(&x)?;

        let result = x.squeeze(1)?;

        Ok(result)
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
        println!("RTModel Weights:");

        // Helper function to print the first 5 values of a tensor
        fn print_first_5_values(tensor: &Tensor, name: &str) {
            let shape = tensor.shape();
            if shape.dims().len() == 2 {
                // Extract the first row
                if let Ok(row) = tensor.i((0, ..)) {
                    match row.to_vec1::<f32>() {
                        Ok(values) => println!(
                            "{} (first 5 values of first row): {:?}",
                            name,
                            &values[..5.min(values.len())]
                        ),
                        Err(e) => eprintln!("Error printing {}: {:?}", name, e),
                    }
                } else {
                    eprintln!("Error extracting first row for {}", name);
                }
            } else {
                match tensor.to_vec1::<f32>() {
                    Ok(values) => println!(
                        "{} (first 5 values): {:?}",
                        name,
                        &values[..5.min(values.len())]
                    ),
                    Err(e) => eprintln!("Error printing {}: {:?}", name, e),
                }
            }
        }

        // Print the first 5 values of each weight tensor
        if let Ok(tensor) = self.var_store.get((2, 103), "rt_encoder.mod_nn.nn.weight") {
            print_first_5_values(&tensor, "rt_encoder.mod_nn.nn.weight");
        }
        // if let Ok(tensor) = self.var_store.get((35, 35, 3), "rt_encoder.input_cnn.cnn_short.weight") {
        //     print_first_5_values(&tensor, "rt_encoder.input_cnn.cnn_short.weight");
        // }
        // if let Ok(tensor) = self.var_store.get((35, 35, 5), "rt_encoder.input_cnn.cnn_medium.weight") {
        //     print_first_5_values(&tensor, "rt_encoder.input_cnn.cnn_medium.weight");
        // }
        // if let Ok(tensor) = self.var_store.get((35, 35, 7), "rt_encoder.input_cnn.cnn_long.weight") {
        //     print_first_5_values(&tensor, "rt_encoder.input_cnn.cnn_long.weight");
        // }
        // if let Ok(tensor) = self.var_store.get((4, 1, 128), "rt_encoder.hidden_nn.rnn_h0") {
        //     print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn_h0");
        // }
        // if let Ok(tensor) = self.var_store.get((4, 1, 128), "rt_encoder.hidden_nn.rnn_c0") {
        //     print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn_c0");
        // }
        if let Ok(tensor) = self
            .var_store
            .get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l0");
        }
        if let Ok(tensor) = self
            .var_store
            .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l0");
        }
        if let Ok(tensor) = self
            .var_store
            .get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse");
        }
        if let Ok(tensor) = self
            .var_store
            .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse");
        }
        if let Ok(tensor) = self
            .var_store
            .get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l1");
        }
        if let Ok(tensor) = self
            .var_store
            .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l1");
        }
        if let Ok(tensor) = self
            .var_store
            .get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse");
        }
        if let Ok(tensor) = self
            .var_store
            .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse")
        {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse");
        }
        if let Ok(tensor) = self
            .var_store
            .get((1, 256), "rt_encoder.attn_sum.attn.0.weight")
        {
            print_first_5_values(&tensor, "rt_encoder.attn_sum.attn.0.weight");
        }
        if let Ok(tensor) = self.var_store.get((256, 256), "rt_decoder.nn.0.weight") {
            print_first_5_values(&tensor, "rt_decoder.nn.0.weight");
        }
        if let Ok(tensor) = self.var_store.get((256, 256), "rt_decoder.nn.1.weight") {
            print_first_5_values(&tensor, "rt_decoder.nn.1.weight");
        }
        if let Ok(tensor) = self.var_store.get((1, 256), "rt_decoder.nn.2.weight") {
            print_first_5_values(&tensor, "rt_decoder.nn.2.weight");
        }
    }
}

// Module Trait Implementation

// impl Module for RTCNNLSTMModel {
//     fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
//         ModelInterface::forward(self, input)
//     }
// }

#[cfg(test)]
mod tests {
    use crate::models::model_interface::{ModelInterface, PredictionResult};
    use crate::models::rt_cnn_lstm_model::RTCNNLSTMModel;
    use candle_core::Device;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_tensor_from_pth() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let tensor_data = candle_core::pickle::read_all(model_path).unwrap();
        println!("{:?}", tensor_data);
    }

    #[test]
    fn test_parse_model_constants() {
        let path = "data/models/alphapeptdeep/generic/rt.pth.model_const.yaml";
        let result = parse_model_constants(path);
        assert!(result.is_ok());
        let constants = result.unwrap();
        assert_eq!(constants.aa_embedding_size.unwrap(), 27);
        assert_eq!(constants.charge_factor, Some(0.1));
        assert_eq!(constants.instruments.len(), 4);
        assert_eq!(constants.max_instrument_num, 8);
        assert_eq!(constants.mod_elements.len(), 109);
        assert_eq!(constants.nce_factor, Some(0.01));
    }

    #[test]
    fn test_encode_peptides() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        let device = Device::Cpu;
        let model =
            RTCNNLSTMModel::new(&model_path, Some(&constants_path), 0, 8, 4, true, device).unwrap();

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

    #[test]
    fn test_encode_peptides_batch() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        let device = Device::Cpu;

        let model = RTCNNLSTMModel::new(
            &model_path,
            Some(&constants_path),
            0,
            8,
            4,
            true,
            device.clone(),
        )
        .unwrap();

        let naked_sequence = vec![
            Arc::from(b"ACDEFGHIK".to_vec().into_boxed_slice()),
            Arc::from(b"AGHCEWQMKYR".to_vec().into_boxed_slice()),
        ];
        let mods = vec![
            Arc::from(b"Carbamidomethyl@C".to_vec().into_boxed_slice()),
            Arc::from(
                b"Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M"
                    .to_vec()
                    .into_boxed_slice(),
            ),
        ];
        let mod_sites = vec![
            Arc::from(b"1".to_vec().into_boxed_slice()),
            Arc::from(b"0;4;8".to_vec().into_boxed_slice()),
        ];

        let result = model.encode_peptides(&naked_sequence, &mods, &mod_sites, None, None, None);

        assert!(result.is_ok());
        let tensor = result.unwrap();
        println!("Batched encoded tensor shape: {:?}", tensor.shape());

        let (batch, seq_len, feat_dim) = tensor.shape().dims3().unwrap();
        assert_eq!(batch, 2);
        assert!(seq_len >= 11);
        assert!(feat_dim > 1);
    }

    #[test]
    fn test_prediction() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let model_path = PathBuf::from(
            "/home/singjc/Documents/github/easypqp-rs/pretrained_models_v3/generic/rt.pth",
        );
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let mut model =
            RTCNNLSTMModel::new(&model_path, Some(&constants_path), 0, 8, 4, true, device).unwrap();

        let test_peptides = vec![
            (
                "AGHCEWQMKYR",
                "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M",
                "0;4;8",
                0.2945,
            ),
            ("QPYAVSELAGHQTSAESWGTGR", "", "", 0.4328955),
            ("GMSVSDLADKLSTDDLNSLIAHAHR", "Oxidation@M", "1", 0.6536107),
            (
                "TVQHHVLFTDNMVLICR",
                "Oxidation@M;Carbamidomethyl@C",
                "11;15",
                0.7811949,
            ),
            ("EAELDVNEELDKK", "", "", 0.2934583),
            ("YTPVQQGPVGVNVTYGGDPIPK", "", "", 0.5863009),
            ("YYAIDFTLDEIK", "", "", 0.8048359),
            ("VSSLQAEPLPR", "", "", 0.3201348),
            (
                "NHAVVCQGCHNAIDPEVQR",
                "Carbamidomethyl@C;Carbamidomethyl@C",
                "5;8",
                0.1730425,
            ),
            ("IPNIYAIGDVVAGPMLAHK", "", "", 0.8220097),
        ];

        let peptides: Vec<Arc<[u8]>> = test_peptides
            .iter()
            .map(|(pep, _, _, _)| Arc::from(pep.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        let mods: Vec<Arc<[u8]>> = test_peptides
            .iter()
            .map(|(_, mod_, _, _)| Arc::from(mod_.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        let mod_sites: Vec<Arc<[u8]>> = test_peptides
            .iter()
            .map(|(_, _, sites, _)| Arc::from(sites.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        let observed_rts: Vec<f32> = test_peptides.iter().map(|(_, _, _, rt)| *rt).collect();

        match model.predict(&peptides, &mods, &mod_sites, None, None, None) {
            Ok(PredictionResult::RTResult(rt_preds)) => {
                let total_error: f32 = rt_preds
                    .iter()
                    .zip(observed_rts.iter())
                    .map(|(pred, obs)| (pred - obs).abs())
                    .sum();

                for ((pep_bytes, pred), obs) in peptides
                    .iter()
                    .zip(rt_preds.iter())
                    .zip(observed_rts.iter())
                {
                    let pep = std::str::from_utf8(pep_bytes).unwrap_or("");
                    println!(
                        "Peptide: {}, Predicted RT: {}, Observed RT: {}",
                        pep, pred, obs
                    );
                }

                let mean_absolute_error = total_error / rt_preds.len() as f32;
                println!("Mean Absolute Error: {:.6}", mean_absolute_error);
            }
            Ok(_) => println!("Unexpected prediction result type."),
            Err(e) => println!("Error during batch prediction: {:?}", e),
        }
    }
}
