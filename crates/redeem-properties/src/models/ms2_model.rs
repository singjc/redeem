use crate::models::model_interface::{ModelInterface, PredictionResult};
use crate::models::ms2_bert_model::MS2BertModel;
use crate::utils::data_handling::{PeptideData, TargetNormalization};
use crate::utils::peptdeep_utils::ModificationMap;
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// Enum for different types of MS2 models
pub enum MS2ModelArch {
    MS2Bert,
    // Add other architectures here as needed
}

// Constants for different types of MS2 models
pub const MS2MODEL_ARCHS: &[&str] = &["ms2_bert"];

// A wrapper struct for MS2 models
pub struct MS2ModelWrapper {
    model: Box<dyn ModelInterface + Send + Sync>,
}

impl Clone for MS2ModelWrapper {
    fn clone(&self) -> Self {
        MS2ModelWrapper {
            model: self.model.clone(),
        }
    }
}

impl MS2ModelWrapper {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: P,
        arch: &str,
        device: Device,
    ) -> Result<Self> {
        let model: Box<dyn ModelInterface> = match arch {
            "ms2_bert" => Box::new(MS2BertModel::new(
                model_path,
                Some(constants_path),
                0,
                8,
                4,
                true,
                device,
            )?),
            // Add other cases here as you implement more models
            _ => return Err(anyhow!("Unsupported MS2 model architecture: {}", arch)),
        };

        Ok(Self { model })
    }

    pub fn predict(
        &self,
        peptide_sequence: &[Arc<[u8]>],
        mods: &[Arc<[u8]>],
        mod_sites: &[Arc<[u8]>],
        charge: Vec<i32>,
        nce: Vec<i32>,
        intsrument: Vec<Option<Arc<[u8]>>>,
    ) -> Result<PredictionResult> {
        self.model.predict(
            peptide_sequence,
            mods,
            mod_sites,
            Some(charge),
            Some(nce),
            Some(intsrument),
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

// Public API Function to load a new MS2 model
pub fn load_ms2_model<P: AsRef<Path>>(
    model_path: P,
    constants_path: P,
    arch: &str,
    device: Device,
) -> Result<MS2ModelWrapper> {
    MS2ModelWrapper::new(model_path, constants_path, arch, device)
}

// #[cfg(test)]
// mod tests {
//     use crate::models::ms2_model::load_ms2_model;
//     use candle_core::Device;
//     use std::char;
//     use std::path::PathBuf;
//     use std::time::Instant;

//     #[test]
//     fn peptide_ms2_prediction() {
//         let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
//         let constants_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

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

//         let result = load_ms2_model(&model_path, &constants_path, "ms2_bert", Device::Cpu);

//         assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

//         let mut model = result.unwrap();
//         // model.print_summary();

//         // Print the model's weights
//         // model.print_weights();

//         // Test prediction with real peptides
//         let peptide = "AGHCEWQMKYR".to_string();
//         let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M".to_string();
//         let mod_sites = "0;4;8".to_string();
//         let charge = 2;
//         let nce = 20;
//         let instrument = "QE";

//         println!("Predicting MS2 for peptide: {:?}", peptide);
//         println!("Modifications: {:?}", mods);
//         println!("Modification sites: {:?}", mod_sites);
//         println!("Charge: {:?}", charge);
//         println!("NCE: {:?}", nce);
//         println!("Instrument: {:?}", instrument);

//         // model.set_evaluation_mode();

//         let start = Instant::now();
//         match model.predict(&[peptide.clone()], &mods, &mod_sites, charge, nce, instrument) {
//             Ok(predictions) => {
//                 let io_time = Instant::now() - start;
//                 assert_eq!(predictions.len(), 10, "Unexpected number of predictions");
//                 println!("Prediction for real peptide:");
//                 println!("Peptide: {} ({} @ {}), Predicted MS2: {}:  {:8} ms", peptide, mods, mod_sites, predictions[0], io_time.as_millis());
//             },
//             Err(e) => {
//                 println!("Error during prediction: {:?}", e);
//                 println!("Attempting to encode peptide...");
//                 match model.encode_peptides(&[peptide.clone()], &mods, &mod_sites, charge, nce, instrument) {
//                     Ok(encoded) => println!("Encoded peptide shape: {:?}", encoded.shape()),
//                     Err(e) => println!("Error encoding peptide: {:?}", e),
//                 }
//             },
//         }
//     }
// }
