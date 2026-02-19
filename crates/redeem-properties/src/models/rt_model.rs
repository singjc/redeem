// rt_model.rs

use crate::models::model_interface::{ModelInterface, PredictionResult};
use crate::models::rt_cnn_lstm_model::RTCNNLSTMModel;
use crate::models::rt_cnn_transformer_model::RTCNNTFModel;
use crate::utils::data_handling::{PeptideData, TargetNormalization};
use crate::utils::peptdeep_utils::ModificationMap;
use crate::utils::stats::TrainingStepMetrics;
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

// Enum for different types of retention time models
pub enum RTModelArch {
    RTCNNLSTM,
    RTCNNTF,
}

// Constants for different types of retention time models
pub const RTMODEL_ARCHS: &[&str] = &["rt_cnn_lstm", "rt_cnn_tf"];

// A wrapper struct for RT models
pub struct RTModelWrapper {
    model: Box<dyn ModelInterface + Send + Sync>,
}

impl Clone for RTModelWrapper {
    fn clone(&self) -> Self {
        RTModelWrapper {
            model: self.model.clone(), // uses clone_box() behind the scenes
        }
    }
}

impl RTModelWrapper {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: Option<P>,
        arch: &str,
        device: Device,
    ) -> Result<Self> {
        let model: Box<dyn ModelInterface> = match arch {
            "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new(
                model_path,
                constants_path,
                0,
                8,
                4,
                true,
                device,
            )?),
            "rt_cnn_tf" => Box::new(RTCNNTFModel::new(
                model_path,
                constants_path,
                0,
                8,
                4,
                true,
                device,
            )?),
            _ => return Err(anyhow!("Unsupported RT model architecture: {}", arch)),
        };

        Ok(Self { model })
    }

    pub fn predict(
        &self,
        peptide_sequence: &[Arc<[u8]>],
        mods: &[Arc<[u8]>],
        mod_sites: &[Arc<[u8]>],
    ) -> Result<PredictionResult> {
        self.model
            .predict(peptide_sequence, mods, mod_sites, None, None, None)
    }

    pub fn train(
        &mut self,
        training_data: &Vec<PeptideData>,
        val_data: Option<&Vec<PeptideData>>,
        modifications: HashMap<(String, Option<char>), ModificationMap>,
        batch_size: usize,
        val_batch_size: usize,
        learning_rate: f64,
        epochs: usize,
        early_stopping_patience: usize,
        target_norm: TargetNormalization,
        train_var_prefixes: Option<Vec<String>>,
        warmup_fraction: Option<f64>,
    ) -> Result<TrainingStepMetrics> {
        self.model.train(
            training_data,
            val_data,
            modifications,
            batch_size,
            val_batch_size,
            learning_rate,
            epochs,
            early_stopping_patience,
            "training",
            true,
            true,
            target_norm,
            train_var_prefixes,
            warmup_fraction,
        )
    }

    pub fn fine_tune(
        &mut self,
        training_data: &Vec<PeptideData>,
        modifications: HashMap<(String, Option<char>), ModificationMap>,
        batch_size: usize,
        learning_rate: f64,
        epochs: usize,
        target_norm: TargetNormalization,
    ) -> Result<()> {
        self.model.fine_tune(
            training_data,
            modifications,
            batch_size,
            learning_rate,
            epochs,
            target_norm,
            None,
        )
    }

    pub fn inference(
        &mut self,
        inference_data: &Vec<PeptideData>,
        batch_size: usize,
        modifications: HashMap<(String, Option<char>), ModificationMap>,
        rt_norm_params: TargetNormalization,
    ) -> Result<Vec<PeptideData>> {
        self.model
            .inference(inference_data, batch_size, modifications, rt_norm_params)
    }

    pub fn set_evaluation_mode(&mut self) {
        self.model.set_evaluation_mode()
    }

    pub fn set_training_mode(&mut self) {
        self.model.set_training_mode()
    }

    pub fn print_summary(&self) {
        self.model.print_summary()
    }

    pub fn print_weights(&self) {
        self.model.print_weights()
    }

    pub fn save(&mut self, path: &str) -> Result<()> {
        self.model.save(path)
    }
}

// Public API Function to load a new RT model
pub fn load_retention_time_model<P: AsRef<Path>>(
    model_path: P,
    constants_path: Option<P>,
    arch: &str,
    device: Device,
) -> Result<RTModelWrapper> {
    RTModelWrapper::new(model_path, constants_path, arch, device)
}

// #[cfg(test)]
// mod tests {
//     use crate::models::rt_model::load_retention_time_model;
//     use candle_core::Device;
//     use std::path::PathBuf;
//     use std::time::Instant;

//     #[test]
//     fn peptide_retention_time_prediction() {
//         let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
//         // let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt_resaved_model.pth");
//         let constants_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");

//         assert!(
//             model_path.exists(),
//             "\n╔══════════════════════════════════════════════════════════════════╗\n\
//              ║                     *** ERROR: FILE NOT FOUND ***                ║\n\
//              ╠══════════════════════════════════════════════════════════════════╣\n\
//              ║ Test model file does not exist:                                  ║\n\
//              ║ {:?}\n\
//              ║ \n\
//              ║ Visit AlphaPeptDeeps github repo on how to download their \n\
//              ║ pretrained model files: https://github.com/MannLabs/\n\
//              ║ alphapeptdeep?tab=readme-ov-file#install-models\n\
//              ╚══════════════════════════════════════════════════════════════════╝\n",
//             model_path
//         );

//         assert!(
//             constants_path.exists(),
//             "\n╔══════════════════════════════════════════════════════════════════╗\n\
//              ║                     *** ERROR: FILE NOT FOUND ***                  ║\n\
//              ╠══════════════════════════════════════════════════════════════════╣\n\
//              ║ Test constants file does not exist:                                ║\n\
//              ║ {:?}\n\
//              ║ \n\
//              ║ Visit AlphaPeptDeeps github repo on how to download their \n\
//              ║ pretrained model files: https://github.com/MannLabs/\n\
//              ║ alphapeptdeep?tab=readme-ov-file#install-models\n\
//              ╚══════════════════════════════════════════════════════════════════╝\n",
//             constants_path
//         );

//         let result = load_retention_time_model(&model_path, &constants_path, "rt_cnn_lstm", Device::Cpu);

//         assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

//         let mut model = result.unwrap();
//         model.print_summary();

//         // Print the model's weights
//         model.print_weights();

//         // Test prediction with real peptides
//         let peptide = "AGHCEWQMKYR".to_string();
//         let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M".to_string();
//         let mod_sites = "0;4;8".to_string();

//         println!("Predicting retention time for peptide: {:?}", peptide);
//         println!("Modifications: {:?}", mods);
//         println!("Modification sites: {:?}", mod_sites);

//         model.set_evaluation_mode();

//         let start = Instant::now();
//         match model.predict(&[peptide.clone()], &mods, &mod_sites) {
//             Ok(predictions) => {
//                 let io_time = Instant::now() - start;
//                 assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
//                 println!("Prediction for real peptide:");
//                 println!("Peptide: {} ({} @ {}), Predicted RT: {}:  {:8} ms", peptide, mods, mod_sites, predictions[0], io_time.as_millis());
//             },
//             Err(e) => {
//                 println!("Error during prediction: {:?}", e);
//                 println!("Attempting to encode peptide...");
//                 match model.encode_peptides(&[peptide.clone()], &mods, &mod_sites) {
//                     Ok(encoded) => println!("Encoded peptide shape: {:?}", encoded.shape()),
//                     Err(e) => println!("Error encoding peptide: {:?}", e),
//                 }
//             },
//         }
//     }
// }
