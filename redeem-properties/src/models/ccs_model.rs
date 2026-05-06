use crate::models::ccs_cnn_tf_model::CCSCNNTFModel;
use crate::models::model_interface::{ModelInterface, PredictionResult};
use crate::utils::data_handling::{PeptideData, TargetNormalization};
use crate::utils::peptdeep_utils::ModificationMap;
use crate::utils::stats::TrainingStepMetrics;
use anyhow::{anyhow, Result};
use candle_core::Device;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "legacy-peptdeep-models")]
use crate::models::ccs_cnn_lstm_model::CCSCNNLSTMModel;

// Enum for different types of CCS models
pub enum CCSModelArch {
    #[cfg(feature = "legacy-peptdeep-models")]
    CCSCNNLSTM,
    CCSCNNTF,
}

// Constants for different types of CCS models
#[cfg(feature = "legacy-peptdeep-models")]
pub const CCSMODEL_ARCHS: &[&str] = &["ccs_cnn_lstm", "ccs_cnn_tf"];

#[cfg(not(feature = "legacy-peptdeep-models"))]
pub const CCSMODEL_ARCHS: &[&str] = &["ccs_cnn_tf"];

// A wrapper struct for CCS models
pub struct CCSModelWrapper {
    model: Box<dyn ModelInterface + Send + Sync>,
}

impl Clone for CCSModelWrapper {
    fn clone(&self) -> Self {
        CCSModelWrapper {
            model: self.model.clone(),
        }
    }
}

impl CCSModelWrapper {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: Option<P>,
        arch: &str,
        device: Device,
    ) -> Result<Self> {
        let model: Box<dyn ModelInterface> = match arch {
            #[cfg(feature = "legacy-peptdeep-models")]
            "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new(
                model_path,
                constants_path,
                0,
                8,
                4,
                true,
                device,
            )?),
            "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new(
                model_path,
                constants_path,
                0,
                8,
                4,
                true,
                device,
            )?),
            _ => return Err(anyhow!("Unsupported CCS model architecture: {}", arch)),
        };

        Ok(Self { model })
    }

    pub fn predict(
        &self,
        peptide_sequence: &[Arc<[u8]>],
        mods: &[Arc<[u8]>],
        mod_sites: &[Arc<[u8]>],
        charge: Vec<i32>,
    ) -> Result<PredictionResult> {
        self.model
            .predict(peptide_sequence, mods, mod_sites, Some(charge), None, None)
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
        validation_data: Option<&Vec<PeptideData>>,
        modifications: HashMap<(String, Option<char>), ModificationMap>,
        batch_size: usize,
        learning_rate: f64,
        epochs: usize,
        early_stopping_patience: Option<usize>,
        target_norm: TargetNormalization,
        warmup_fraction: Option<f64>,
    ) -> Result<()> {
        self.model.fine_tune(
            training_data,
            validation_data,
            modifications,
            batch_size,
            learning_rate,
            epochs,
            early_stopping_patience,
            target_norm,
            warmup_fraction,
        )
    }

    pub fn output_normalization(&self) -> Option<TargetNormalization> {
        self.model.get_output_denormalization()
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

    /// Return the model architecture string (delegates to the inner model).
    pub fn model_arch(&self) -> String {
        self.model.get_model_arch()
    }

    /// Compute the total number of parameters stored in the model's VarMap.
    pub fn param_count(&mut self) -> usize {
        let vm = self.model.get_mut_varmap();
        let data = vm.data().lock().unwrap();
        let mut total: usize = 0;
        for (_name, tensor) in data.iter() {
            let shape = tensor.shape().dims();
            let numel: usize = shape.iter().product();
            total = total.saturating_add(numel);
        }
        total
    }

    /// Detailed summary for developer repr: groups and top tensors.
    pub fn summary_detailed(&mut self) -> String {
        let arch = self.model.get_model_arch();
        let vm = self.model.get_mut_varmap();
        let data = vm.data().lock().unwrap();

        let mut entries: Vec<(String, Vec<usize>, usize)> = Vec::with_capacity(data.len());
        for (name, tensor) in data.iter() {
            let shape = tensor.shape().dims().to_vec();
            let numel: usize = shape.iter().product();
            entries.push((name.clone(), shape, numel));
        }

        let total: usize = entries.iter().map(|e| e.2).sum();

        use std::collections::HashMap;
        let mut groups: HashMap<String, (usize, usize)> = HashMap::new();
        for (name, _shape, numel) in &entries {
            let prefix = name.split('.').next().unwrap_or(name.as_str()).to_string();
            let ent = groups.entry(prefix).or_insert((0usize, 0usize));
            ent.0 += 1;
            ent.1 += *numel;
        }

        let mut groups_vec: Vec<(String, (usize, usize))> = groups.into_iter().collect();
        groups_vec.sort_by_key(|(_, (_count, params))| std::cmp::Reverse(*params));

        entries.sort_by_key(|e| std::cmp::Reverse(e.2));

        let mut s = String::new();
        s.push_str(&format!(
            "{} total_params={} groups={}\n",
            arch,
            total,
            groups_vec.len()
        ));
        s.push_str(&format!(
            "{:<24} {:>8} {:>12}\n",
            "component", "tensors", "params"
        ));
        for (name, (count, params)) in groups_vec.iter().take(20) {
            s.push_str(&format!("{:<24} {:>8} {:>12}\n", name, count, params));
        }

        s.push_str("\nTop tensors:\n");
        for (name, shape, numel) in entries.iter().take(20) {
            s.push_str(&format!(
                "{:<40} {:<20} {:>12}\n",
                name,
                format!("{:?}", shape),
                numel
            ));
        }

        s
    }

    /// Pretty hierarchical summary similar to PyTorch's module repr.
    pub fn summary_pretty(&mut self) -> String {
        let arch = self.model.get_model_arch();
        // Reuse the same logic as RTModelWrapper but simpler: build a tree and print
        let vm = self.model.get_mut_varmap();
        let data = vm.data().lock().unwrap();
        use std::collections::BTreeMap;

        #[derive(Default)]
        struct Node {
            children: BTreeMap<String, Node>,
            tensors: Vec<(String, Vec<usize>, usize)>,
        }

        let mut root = Node::default();
        for (name, tensor) in data.iter() {
            let parts: Vec<&str> = name.split('.').collect();
            let shape = tensor.shape().dims().to_vec();
            let numel: usize = shape.iter().product();
            let mut node = &mut root;
            for p in &parts[..parts.len().saturating_sub(1)] {
                node = node.children.entry(p.to_string()).or_default();
            }
            node.tensors
                .push((parts.last().unwrap().to_string(), shape, numel));
        }

        fn fmt_tensor(name: &str, shape: &[usize]) -> String {
            if name.ends_with("weight") && shape.len() == 2 {
                return format!(
                    "Linear(in_features={}, out_features={})",
                    shape[1], shape[0]
                );
            }
            if name.ends_with("bias") && shape.len() == 1 {
                return format!("bias[{}]", shape[0]);
            }
            format!("{} {:?}", name, shape)
        }

        fn write_node(s: &mut String, node: &Node, indent: usize, name: Option<&str>) {
            let pad = "  ".repeat(indent);
            if let Some(n) = name {
                s.push_str(&format!("{}{}(\n", pad, n));
            }
            for (tname, shape, _numel) in &node.tensors {
                s.push_str(&format!(
                    "{}  ({}): {}\n",
                    pad,
                    tname,
                    fmt_tensor(tname, shape)
                ));
            }
            let mut numeric_keys: Vec<usize> = vec![];
            let mut numeric_map: std::collections::BTreeMap<usize, &Node> =
                std::collections::BTreeMap::new();
            let mut non_numeric: Vec<(&String, &Node)> = vec![];
            for (k, v) in &node.children {
                if let Ok(idx) = k.parse::<usize>() {
                    numeric_keys.push(idx);
                    numeric_map.insert(idx, v);
                } else {
                    non_numeric.push((k, v));
                }
            }
            for (k, v) in non_numeric {
                s.push_str(&format!("{}  ({}):\n", pad, k));
                write_node(s, v, indent + 2, None);
            }
            if !numeric_keys.is_empty() {
                numeric_keys.sort();
                let mut ranges: Vec<(usize, usize)> = Vec::new();
                let mut start = numeric_keys[0];
                let mut last = start;
                for &k in &numeric_keys[1..] {
                    if k == last + 1 {
                        last = k;
                    } else {
                        ranges.push((start, last));
                        start = k;
                        last = k;
                    }
                }
                ranges.push((start, last));
                for (a, b) in ranges {
                    let count = b - a + 1;
                    if let Some(rep) = numeric_map.get(&a) {
                        s.push_str(&format!("{}  ({}-{}): {} x <module>\n", pad, a, b, count));
                        for (child_name, child_node) in &rep.children {
                            s.push_str(&format!("{}    ({}):\n", pad, child_name));
                            write_node(s, child_node, indent + 3, None);
                        }
                    }
                }
            }
            if name.is_some() {
                s.push_str(&format!("{} )\n", pad));
            }
        }

        let mut out = String::new();
        out.push_str(&format!("{}(\n", arch));
        write_node(&mut out, &root, 1, None);
        out.push_str(")\n");
        out
    }
}

// Public API Function to load a new CCS model
pub fn load_collision_cross_section_model<P: AsRef<Path>>(
    model_path: P,
    constants_path: Option<P>,
    arch: &str,
    device: Device,
) -> Result<CCSModelWrapper> {
    CCSModelWrapper::new(model_path, constants_path, arch, device)
}

// #[cfg(test)]
// mod tests {
//     use crate::models::ccs_model::load_collision_cross_section_model;
//     use candle_core::Device;
//     use std::char;
//     use std::path::PathBuf;
//     use std::time::Instant;

//     #[test]
//     fn peptide_ccs_prediction() {
//         let model_path = PathBuf::from("data/pretrained_models/alphapeptdeep/generic/ccs.pth");
//         let constants_path = PathBuf::from("data/pretrained_models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

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

//         let result = load_collision_cross_section_model(&model_path, &constants_path, "ccs_cnn_lstm", Device::Cpu);

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

//         println!("Predicting ccs for peptide: {:?}", peptide);
//         println!("Modifications: {:?}", mods);
//         println!("Modification sites: {:?}", mod_sites);
//         println!("Charge: {:?}", charge);

//         // model.set_evaluation_mode();

//         let start = Instant::now();
//         match model.predict(&[peptide.clone()], &mods, &mod_sites, charge) {
//             Ok(predictions) => {
//                 let io_time = Instant::now() - start;
//                 assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
//                 println!("Prediction for real peptide:");
//                 println!("Peptide: {} ({} @ {}), Predicted CCS: {}:  {:8} ms", peptide, mods, mod_sites, predictions[0], io_time.as_millis());
//             },
//             Err(e) => {
//                 println!("Error during prediction: {:?}", e);
//                 println!("Attempting to encode peptide...");
//                 match model.encode_peptides(&[peptide.clone()], &mods, &mod_sites, charge) {
//                     Ok(encoded) => println!("Encoded peptide shape: {:?}", encoded.shape()),
//                     Err(e) => println!("Error encoding peptide: {:?}", e),
//                 }
//             },
//         }
//     }
// }
