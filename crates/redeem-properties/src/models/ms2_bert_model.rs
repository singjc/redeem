use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Dropout, Module, VarBuilder, VarMap};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::HashMap, sync::Arc};

use crate::utils::utils::get_tensor_stats;

use crate::{
    building_blocks::building_blocks::{
        DecoderLinear, HiddenHfaceTransformer, Input26aaModPositionalEncoding, MetaEmbedding,
        ModLossNN, MOD_FEATURE_SIZE,
    },
    models::model_interface::{
        create_var_map, load_tensors_from_model, ModelInterface, PropertyType,
    },
    utils::peptdeep_utils::{load_mod_to_feature_arc, parse_model_constants, ModelConstants},
};

// Constants
const CHARGE_FACTOR: f64 = 0.1;
const NCE_FACTOR: f64 = 0.01;

// Main Model Struct
#[derive(Clone)]
/// Represents an AlphaPeptDeep MS2BERT model.
pub struct MS2BertModel {
    var_store: VarBuilder<'static>,
    varmap: VarMap,
    constants: ModelConstants,
    mod_to_feature: HashMap<Arc<[u8]>, Vec<f32>>,
    fixed_sequence_len: usize,
    // Total number of fragment types of a fragmentation position to predict
    num_frag_types: usize,
    // Number of fragment types of a fragmentation position to predict, by default 0
    num_modloss_types: usize,
    // If True, the modloss layer will be disabled, by default True
    mask_modloss: bool,
    min_inten: f32,
    device: Device,
    is_training: bool,
    dropout: Dropout,
    input_nn: Input26aaModPositionalEncoding,
    meta_nn: MetaEmbedding,
    hidden_nn: HiddenHfaceTransformer,
    output_nn: DecoderLinear,
    modloss_nn: ModLossNN,
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl Send for MS2BertModel {}
unsafe impl Sync for MS2BertModel {}

// Code Model Implementation
impl ModelInterface for MS2BertModel {
    fn property_type(&self) -> PropertyType {
        PropertyType::MS2
    }

    fn model_arch(&self) -> &'static str {
        "ms2_bert"
    }

    fn new_untrained(_device: Device) -> Result<Self> {
        unimplemented!("Untrained model creation is not implemented for this architecture.");
    }

    /// Create a new MS2BERT model from the given model and constants files.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: Option<P>,
        fixed_sequence_len: usize,
        num_frag_types: usize,
        num_modloss_types: usize,
        mask_modloss: bool,
        device: Device,
    ) -> Result<Self> {
        let tensor_data = load_tensors_from_model(model_path.as_ref(), &device)?;

        let mut varmap = VarMap::new();
        create_var_map(&mut varmap, tensor_data, &device)?;

        let var_store = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let constants = match constants_path {
            Some(path) => parse_model_constants(path.as_ref().to_str().unwrap())?,
            None => ModelConstants::default(),
        };

        // Load the mod_to_feature mapping
        let mod_to_feature = load_mod_to_feature_arc(&constants)?;

        let dropout = Dropout::new(0.1);

        let meta_dim = 8;
        let input_nn = Input26aaModPositionalEncoding::from_varstore(
            &var_store,
            256 - 8,
            200,
            vec![
                "input_nn.mod_nn.nn.weight",
                "input_nn.aa_emb.weight",
                "input_nn.pos_encoder.pe",
            ],
        )
        .unwrap();

        let meta_nn = MetaEmbedding::from_varstore(
            &var_store,
            8,
            vec!["meta_nn.nn.weight", "meta_nn.nn.bias"],
        )
        .unwrap();

        let hidden_nn = HiddenHfaceTransformer::from_varstore(
            var_store.pp("hidden_nn.bert"),
            256,
            4,
            8,
            4,
            0.1,
            false,
        )
        .unwrap();

        let output_nn = DecoderLinear::from_varstore(
            &var_store,
            256,
            4,
            vec![
                "output_nn.nn.0.weight",
                "output_nn.nn.1.weight",
                "output_nn.nn.2.weight",
            ],
            vec!["output_nn.nn.0.bias", "output_nn.nn.2.bias"],
        )
        .unwrap();

        let modloss_nn = ModLossNN::from_varstore(
            var_store.clone(),
            256,
            4,
            8,
            1,
            0.1,
            false,
            4,
            "modloss_nn.0.bert",
            vec![
                "modloss_nn.1.nn.0.weight",
                "modloss_nn.1.nn.1.weight",
                "modloss_nn.1.nn.2.weight",
            ],
            vec!["modloss_nn.1.nn.0.bias", "modloss_nn.1.nn.2.bias"],
        )
        .unwrap();

        Ok(Self {
            var_store: var_store,
            varmap: varmap,
            constants: constants,
            mod_to_feature: mod_to_feature,
            fixed_sequence_len: fixed_sequence_len,
            num_frag_types: num_frag_types,
            num_modloss_types: num_modloss_types,
            mask_modloss: mask_modloss,
            min_inten: 1e-4,
            device,
            is_training: false,
            dropout: dropout,
            input_nn: input_nn,
            meta_nn: meta_nn,
            hidden_nn: hidden_nn,
            output_nn: output_nn,
            modloss_nn: modloss_nn,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (_batch_size, seq_len, _) = xs.shape().dims3()?;

        // Separate the input tensor into the different parts

        // Calculate starting indices
        let start_mod_x = 1;
        let start_charge = start_mod_x + MOD_FEATURE_SIZE;
        let start_nce = start_charge + 1;
        let start_instrument = start_nce + 1;

        // Extract tensors using indexing
        let aa_indices_out = xs.i((.., .., 0))?;
        let mod_x_out = xs.i((.., .., start_mod_x..start_mod_x + MOD_FEATURE_SIZE))?;
        let charge_out = xs.i((.., 0..1, start_charge..start_charge + 1))?;
        let nce_out = xs.i((.., 0..1, start_nce..start_nce + 1))?;
        let instrument_out = xs.i((.., 0..1, start_instrument..start_instrument + 1))?;

        // Adjust shapes after extraction if necessary
        // let aa_indices_out = aa_indices_out.squeeze(2)?; // Squeeze to remove dimensions of size 1 if needed
        let charge_out = charge_out.squeeze(2)?; // Squeeze to remove dimensions of size 1 if needed
        let nce_out = nce_out.squeeze(2)?; // Squeeze to remove dimensions of size 1 if needed
        let instrument_out = instrument_out.squeeze(2)?.squeeze(1)?; // Squeeze to remove dimensions of size 1 if needed

        log::trace!(
            "[MS2BertModel::forward] aa_indices_out shape: {:?}, device: {:?}",
            aa_indices_out.shape(),
            aa_indices_out.device()
        );
        log::trace!(
            "[MS2BertModel::forward] mod_x_out shape: {:?}, device: {:?}",
            mod_x_out.shape(),
            mod_x_out.device()
        );

        log::trace!(
            "[MS2BertModel::forward] charge_out shape: {:?}, device: {:?}",
            charge_out.shape(),
            charge_out.device()
        );
        log::trace!(
            "[MS2BertModel::forward] nce_out shape: {:?}, device: {:?}",
            nce_out.shape(),
            nce_out.device()
        );
        log::trace!(
            "[MS2BertModel::forward] instrument_out shape: {:?}, device: {:?}",
            instrument_out.shape(),
            instrument_out.device()
        );

        // Forward pass through input_nn with dropout
        let in_x = self.dropout.forward(
            &self.input_nn.forward(&aa_indices_out, &mod_x_out)?,
            self.is_training,
        )?;

        log::trace!(
            "[MS2BertModel::forward] in_x shape (post dropout-input_nn): {:?}, device: {:?}",
            in_x.shape(),
            in_x.device()
        );

        // Prepare metadata for meta_nn
        let meta_x = self
            .meta_nn
            .forward(&charge_out, &nce_out, &instrument_out)?
            .unsqueeze(1)?
            .repeat(vec![1, seq_len as usize, 1])?;
        log::trace!(
            "[MS2BertModel::forward] meta_x (post meta_nn) shape: {:?}, device: {:?}",
            meta_x.shape(),
            meta_x.device()
        );

        // Concatenate in_x and meta_x along dimension 2
        let combined_input = Tensor::cat(&[in_x.clone(), meta_x], 2)?;
        log::trace!(
            "[MS2BertModel::forward] combined_input shape: {:?}, device: {:?}",
            combined_input.shape(),
            combined_input.device()
        );

        // Forward pass through hidden_nn
        let hidden_x = self.hidden_nn.forward(&combined_input.clone(), None)?;
        log::trace!(
            "[MS2BertModel::forward] hidden_x shape: {:?}, device: {:?}",
            hidden_x.shape(),
            hidden_x.device()
        );

        // // Handle attentions if needed (similar to PyTorch)
        // if self.output_attentions {
        //     self.attentions = hidden_x[1];
        // } else {
        //     self.attentions = None;
        // }

        // Apply dropout and combine with input
        let x_tmp = (hidden_x + combined_input * 0.2)?;
        let hidden_output = self.dropout.forward(&x_tmp, true)?;
        log::trace!(
            "[MS2BertModel::forward] hidden_output shape: {:?}, device: {:?}",
            hidden_output.shape(),
            hidden_output.device()
        );

        // Forward pass through output_nn
        let mut out_x = self.output_nn.forward(&hidden_output)?;
        log::trace!(
            "[MS2BertModel::forward] out_x shape: {:?}, device: {:?}",
            out_x.shape(),
            out_x.device()
        );

        // Handle modloss if applicable (similar logic as PyTorch)
        if self.num_modloss_types > 0 {
            if self.mask_modloss {
                // Create a tensor of zeros with the appropriate shape
                let zeros_shape = (
                    out_x.shape().dims()[0],
                    out_x.shape().dims()[1],
                    self.num_modloss_types,
                );
                let zeros_tensor = Tensor::zeros(zeros_shape, DType::F32, &self.device)?; // Adjust device as necessary

                // Concatenate along the last dimension
                out_x = Tensor::cat(&[out_x, zeros_tensor], 2)?;
            } else {
                // // Forward pass through the first modloss neural network
                // let modloss_output = self.modloss_nn[0].forward(in_x)?;

                // // // If output attentions is enabled, save them
                // // if self.output_attentions {
                // //     self.modloss_attentions = Some(modloss_output.clone()); // Assuming you want to store a clone
                // // }

                // // Add hidden_x to the first output
                // let modloss_combined = &modloss_output + hidden_x;

                // // Forward pass through the last modloss neural network
                // let modloss_final = self.modloss_nn.last().unwrap().forward(&modloss_combined)?;

                // // Concatenate the outputs along the last dimension
                // out_x = Tensor::cat(&[out_x, modloss_final], 2)?;
                todo!();
            }
        }

        Ok(out_x.i((.., 3.., ..))?)
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
        self.min_inten
    }

    fn get_mut_varmap(&mut self) -> &mut VarMap {
        &mut self.varmap
    }

    fn print_summary(&self) {
        todo!()
    }

    fn print_weights(&self) {
        todo!()
    }
}

// // Module Trait Implementation
// impl Module for MS2BertModel {
//     fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
//         ModelInterface::forward(self, input)
//     }
// }

impl fmt::Debug for MS2BertModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MS2BertModel(")?;
        writeln!(f, "  (dropout): Dropout(p={})", 0.1)?;
        writeln!(f, "  (input_nn): Input26aaModPositionalEncoding(")?;
        writeln!(f, "    (mod_nn): Mod_Embedding_FixFirstK(")?;
        writeln!(
            f,
            "      (nn): Linear(in_features=103, out_features=2, bias=False)"
        )?;
        writeln!(f, "    )")?;
        writeln!(f, "    (aa_emb): Embedding(27, 240, padding_idx=0)")?;
        writeln!(f, "    (pos_encoder): PositionalEncoding()")?;
        writeln!(f, "  )")?;
        writeln!(f, "  (meta_nn): MetaEmbedding(")?;
        writeln!(
            f,
            "    (nn): Linear(in_features=9, out_features=7, bias=True)"
        )?;
        writeln!(f, "  )")?;
        writeln!(f, "  (hidden_nn): HiddenHfaceTransformer(")?;
        writeln!(f, "    (bert): BertEncoder(")?;
        writeln!(f, "      (layer): ModuleList(")?;
        for i in 0..self.hidden_nn.bert.layers.len() {
            writeln!(f, "        ({}): BertLayer(...)", i)?;
        }
        writeln!(f, "      )")?;
        writeln!(f, "    )")?;
        writeln!(f, "  )")?;
        writeln!(f, "  (output_nn): DecoderLinear(")?;
        writeln!(f, "    (nn): Sequential(")?;
        writeln!(
            f,
            "      (0): Linear(in_features=256, out_features=64, bias=True)"
        )?;
        writeln!(f, "      (1): PReLU(num_parameters=1)")?;
        writeln!(
            f,
            "      (2): Linear(in_features=64, out_features=4, bias=True)"
        )?;
        writeln!(f, "    )")?;
        writeln!(f, "  )")?;
        writeln!(f, "  (modloss_nn): ModLossNN(")?;
        writeln!(f, "    (0): HiddenHfaceTransformer(...)")?;
        writeln!(f, "    (1): DecoderLinear(...)")?;
        writeln!(f, "  )")?;
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::model_interface::ModelInterface;
    use crate::models::ms2_bert_model::MS2BertModel;
    use candle_core::Device;
    use std::path::PathBuf;

    #[test]
    fn test_parse_model_constants() {
        let path = "data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml";
        let result = parse_model_constants(path);
        assert!(result.is_ok());
        let constants = result.unwrap();
        assert_eq!(constants.aa_embedding_size.unwrap(), 27);
        // assert_eq!(constants.charge_factor, 0.1);
        assert_eq!(constants.instruments.len(), 4);
        assert_eq!(constants.max_instrument_num, 8);
        assert_eq!(constants.mod_elements.len(), 109);
        // assert_eq!(constants.nce_factor, 0.01);
    }

    #[test]
    fn test_load_pretrained_ms2_bert_model() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model =
            MS2BertModel::new(model_path, Some(constants_path), 0, 8, 4, true, device).unwrap();

        println!("{:?}", model);
    }

    #[test]
    fn test_encode_peptides() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model =
            MS2BertModel::new(model_path, Some(constants_path), 0, 8, 4, true, device).unwrap();

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
    fn test_forward() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model =
            MS2BertModel::new(model_path, Some(constants_path), 0, 8, 4, true, device).unwrap();

        let seq: Arc<[u8]> = Arc::from(b"AGHCEWQMKYR".to_vec().into_boxed_slice());
        let mods: Arc<[u8]> = Arc::from(
            b"Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M"
                .to_vec()
                .into_boxed_slice(),
        );
        let mod_sites: Arc<[u8]> = Arc::from(b"0;4;8".to_vec().into_boxed_slice());
        let charge = Some(vec![2, 2]);
        let nce = Some(vec![20, 20]);
        let instrument = vec![
            Arc::from(b"QE".to_vec().into_boxed_slice()),
            Arc::from(b"QE".to_vec().into_boxed_slice()),
        ];

        let peptide_sequences = vec![seq.clone(), seq];
        let mods = vec![mods.clone(), mods];
        let mod_sites = vec![mod_sites.clone(), mod_sites];

        let input_tensor = model
            .encode_peptides(
                &peptide_sequences,
                &mods,
                &mod_sites,
                charge,
                nce,
                Some(instrument.into_iter().map(Some).collect()),
            )
            .unwrap();

        let output = model.forward(&input_tensor).unwrap();
        println!("{:?}", output);

        let prediction: Vec<Vec<Vec<f32>>> = output.to_vec3().unwrap();
        println!("{:?}", prediction);
        assert_eq!(prediction.len(), 2);
    }
}
