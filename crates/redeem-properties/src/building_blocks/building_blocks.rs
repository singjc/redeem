use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn as nn;
use candle_transformers as transformers;
use std::fmt;

use crate::building_blocks::bilstm::BidirectionalLSTM;
use crate::building_blocks::featurize::aa_one_hot;
use crate::building_blocks::nn::{BertEncoderModule, ModuleList};
use crate::building_blocks::sequential::{seq, Sequential};
use crate::utils::utils::get_tensor_stats;

use super::nn::TransformerEncoder;

/// constants used by PeptDeep Models
pub const AA_EMBEDDING_SIZE: usize = 27; // TODO: derive from constants yaml
const MAX_INSTRUMENT_NUM: usize = 8; // TODO: derive from constants yaml
                                     // Number of features in the modification vector (kept as a single global
                                     // constant to match the legacy code paths that expect a fixed layout).
pub const MOD_FEATURE_SIZE: usize = 109;

/// Decode w linear NN
#[derive(Clone)]
pub struct DecoderLinear {
    nn: Sequential,
}

impl DecoderLinear {
    pub fn new(in_features: usize, out_features: usize, vb: &nn::VarBuilder) -> Result<Self> {
        let linear1 = nn::linear(in_features, 64, vb.pp("nn.0"))?;
        // PReLU with a shared, learnable slope (scalar) registered in the var store.
        let prelu = nn::PReLU::new(vb.get(1, "nn.1.weight")?, true);
        let linear2 = nn::linear(64, out_features, vb.pp("nn.2"))?;

        let mut nn = seq();
        nn = nn.add(linear1);
        nn = nn.add(prelu);
        nn = nn.add(linear2);

        Ok(Self { nn })
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        in_features: usize,
        out_features: usize,
        names: Vec<&str>,
        names_bias: Vec<&str>,
    ) -> Result<Self> {
        // Load tensors defensively to avoid panics when a checkpoint uses a different
        // decoder head layout (e.g., small/MLP vs legacy linear). Return a clear
        // error instead of unwrapping so higher-level detection can react.
        let w0 = varstore.get((64, in_features), names[0]);
        let b0 = varstore.get(64, names_bias[0]);
        let p = varstore.get(1, names[1]);
        let w_out = varstore.get((out_features, 64), names[2]);
        let b_out = varstore.get(out_features, names_bias[1]);

        let linear1 = match (w0, b0) {
            (Ok(w), Ok(b)) => nn::Linear::new(w, Some(b)),
            (Ok(w), Err(_)) => nn::Linear::new(w, None),
            _ => {
                return Err(candle_core::Error::Msg(format!(
                "DecoderLinear missing required varstore tensors for {} (expected shape ({}, {}))",
                names[0], 64, in_features
            )))
            }
        };

        let prelu = if let Ok(pv) = p {
            let channel_shared = pv.elem_count() == 1;
            nn::PReLU::new(pv, channel_shared)
        } else {
            // Shared, learnable slope fallback (registered so it can be trained if needed).
            nn::PReLU::new(varstore.get(1, "nn.1.weight")?, true)
        };

        let linear2 = match (w_out, b_out) {
            (Ok(w), Ok(b)) => nn::Linear::new(w, Some(b)),
            (Ok(w), Err(_)) => nn::Linear::new(w, None),
            _ => {
                return Err(candle_core::Error::Msg(format!(
                "DecoderLinear missing required varstore tensors for {} (expected shape ({}, {}))",
                names[2], out_features, 64
            )))
            }
        };

        let mut nn = seq();
        nn = nn.add(linear1);
        nn = nn.add(prelu);
        nn = nn.add(linear2);

        Ok(Self { nn })
    }
}

// A richer MLP decoder head used for transformer-based models during training.
// This preserves the original `DecoderLinear` so older/lstm models remain compatible with
// existing varstore layouts. `DecoderMLP` provides additional capacity.
#[derive(Clone)]
pub struct DecoderMLP {
    nn: Sequential,
}

impl DecoderMLP {
    pub fn new(in_features: usize, out_features: usize, vb: &nn::VarBuilder) -> Result<Self> {
        // Architecture: Linear(in -> 64) -> PReLU -> Linear(64 -> 16) -> PReLU -> Linear(16 -> out)
        let linear1 = nn::linear(in_features, 64, vb.pp("nn.0"))?;
        let prelu1 = nn::PReLU::new(vb.get(1, "nn.1.weight")?, true);
        let linear_hidden = nn::linear(64, 16, vb.pp("nn.2"))?;
        let prelu2 = nn::PReLU::new(vb.get(1, "nn.3.weight")?, true);
        let linear2 = nn::linear(16, out_features, vb.pp("nn.4"))?;

        let mut nn_mod = seq();
        nn_mod = nn_mod.add(linear1);
        nn_mod = nn_mod.add(prelu1);
        nn_mod = nn_mod.add(linear_hidden);
        nn_mod = nn_mod.add(prelu2);
        nn_mod = nn_mod.add(linear2);

        Ok(Self { nn: nn_mod })
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        in_features: usize,
        out_features: usize,
        _names: Vec<&str>,
        _names_bias: Vec<&str>,
    ) -> Result<Self> {
        // For now require an MLP varstore layout when loading from a checkpoint; when
        // migrating older checkpoints we can add compatibility mapping. The training path
        // (new_untrained) will construct an MLP directly.
        // Attempt to read weights for nn.0, nn.2, nn.4 and their biases if present.
        let w0 = varstore.get((64, in_features), "nn.0.weight");
        let b0 = varstore.get(64, "nn.0.bias");
        let w_hidden = varstore.get((16, 64), "nn.2.weight");
        let b_hidden = varstore.get(16, "nn.2.bias");
        let w_out = varstore.get((out_features, 16), "nn.4.weight");
        let b_out = varstore.get(out_features, "nn.4.bias");

        let linear1 = match (w0, b0) {
            (Ok(w), Ok(b)) => nn::Linear::new(w, Some(b)),
            (Ok(w), Err(_)) => nn::Linear::new(w, None),
            _ => {
                return Err(candle_core::Error::Msg(
                    "DecoderMLP missing required varstore tensors for nn.0".to_string(),
                ))
            }
        };

        let prelu1 = if let Ok(p) = varstore.get(1, "nn.1.weight") {
            let channel_shared = p.elem_count() == 1;
            nn::PReLU::new(p, channel_shared)
        } else {
            nn::PReLU::new(varstore.get(1, "nn.1.weight")?, true)
        };

        let linear_hidden = match (w_hidden, b_hidden) {
            (Ok(w), Ok(b)) => nn::Linear::new(w, Some(b)),
            (Ok(w), Err(_)) => nn::Linear::new(w, None),
            _ => {
                return Err(candle_core::Error::Msg(
                    "DecoderMLP missing required varstore tensors for nn.2".to_string(),
                ))
            }
        };

        let prelu2 = if let Ok(p) = varstore.get(1, "nn.3.weight") {
            let channel_shared = p.elem_count() == 1;
            nn::PReLU::new(p, channel_shared)
        } else {
            nn::PReLU::new(varstore.get(1, "nn.3.weight")?, true)
        };

        let linear2 = match (w_out, b_out) {
            (Ok(w), Ok(b)) => nn::Linear::new(w, Some(b)),
            (Ok(w), Err(_)) => nn::Linear::new(w, None),
            _ => {
                return Err(candle_core::Error::Msg(
                    "DecoderMLP missing required varstore tensors for nn.4".to_string(),
                ))
            }
        };

        let mut nn_mod = seq();
        nn_mod = nn_mod.add(linear1);
        nn_mod = nn_mod.add(prelu1);
        nn_mod = nn_mod.add(linear_hidden);
        nn_mod = nn_mod.add(prelu2);
        nn_mod = nn_mod.add(linear2);

        Ok(Self { nn: nn_mod })
    }
}

impl Module for DecoderMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // reuse the instrumentation approach used previously for Debugging
        match self.nn.forward_all(x) {
            Ok(outputs) => {
                if let Some(last) = outputs.last() {
                    Ok(last.clone())
                } else {
                    Ok(x.clone())
                }
            }
            Err(e) => {
                log::error!("[DecoderMLP] forward pass failed: {:?}", e);
                Err(e)
            }
        }
    }
}

impl fmt::Debug for DecoderMLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecoderMLP")
            .field("nn", &"<Sequential>")
            .finish()
    }
}

/// A very small projection head that first projects to a low-dimensional
/// bottleneck (e.g. 4-d) then back to the scalar output. Optionally it can
/// contain a learnable scalar multiplier (a 1->1 linear) that is applied to
/// the final output. This is useful to experiment with temperature-like
/// scaling during head-only fine-tuning.
#[derive(Clone)]
pub struct DecoderSmall {
    nn: Sequential,
    scale: Option<nn::Linear>,
}

impl DecoderSmall {
    pub fn new(
        in_features: usize,
        out_features: usize,
        vb: &nn::VarBuilder,
        learnable_scaler: bool,
    ) -> Result<Self> {
        let linear1 = nn::linear(in_features, 4, vb.pp("nn.0"))?;
        // Shared, learnable slope (scalar) registered in the var store.
        let prelu = nn::PReLU::new(vb.get(1, "nn.1.weight")?, true);
        let linear2 = nn::linear(4, out_features, vb.pp("nn.2"))?;

        let mut nn_mod = seq();
        nn_mod = nn_mod.add(linear1);
        nn_mod = nn_mod.add(prelu);
        nn_mod = nn_mod.add(linear2);

        let scale = if learnable_scaler {
            Some(nn::linear(1, 1, vb.pp("scale"))?)
        } else {
            None
        };

        Ok(Self { nn: nn_mod, scale })
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        in_features: usize,
        out_features: usize,
        _names: Vec<&str>,
        _names_bias: Vec<&str>,
    ) -> Result<Self> {
        // Load a small head assuming the varstore uses the small layout.
        let w0 = varstore.get((4, in_features), "nn.0.weight");
        let b0 = varstore.get(4, "nn.0.bias");
        let w_out = varstore.get((out_features, 4), "nn.2.weight");
        let b_out = varstore.get(out_features, "nn.2.bias");

        let linear1 = match (w0, b0) {
            (Ok(w), Ok(b)) => nn::Linear::new(w, Some(b)),
            (Ok(w), Err(_)) => nn::Linear::new(w, None),
            _ => {
                return Err(candle_core::Error::Msg(
                    "DecoderSmall missing required varstore tensors for nn.0".to_string(),
                ))
            }
        };

        let prelu = if let Ok(p) = varstore.get(1, "nn.1.weight") {
            let channel_shared = p.elem_count() == 1;
            nn::PReLU::new(p, channel_shared)
        } else {
            // Shared, learnable slope fallback.
            nn::PReLU::new(varstore.get(1, "nn.1.weight")?, true)
        };

        let linear2 = match (w_out, b_out) {
            (Ok(w), Ok(b)) => nn::Linear::new(w, Some(b)),
            (Ok(w), Err(_)) => nn::Linear::new(w, None),
            _ => {
                return Err(candle_core::Error::Msg(
                    "DecoderSmall missing required varstore tensors for nn.2".to_string(),
                ))
            }
        };

        let mut nn_mod = seq();
        nn_mod = nn_mod.add(linear1);
        nn_mod = nn_mod.add(prelu);
        nn_mod = nn_mod.add(linear2);

        // Try to load optional scalar
        let scale = if let Ok(_) = varstore.get((1, 1), "scale.weight") {
            match nn::linear(1, 1, varstore.pp("scale")) {
                Ok(s) => Some(s),
                Err(_) => None,
            }
        } else {
            None
        };

        Ok(Self { nn: nn_mod, scale })
    }
}

impl Module for DecoderSmall {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self.nn.forward_all(x) {
            Ok(outputs) => {
                if let Some(last) = outputs.last() {
                    let mut out = last.clone();
                    if let Some(scale_layer) = &self.scale {
                        // Determine batch size from output
                        if let Ok((r, _c)) = out.shape().dims2() {
                            let ones = vec![1.0f32; r * 1];
                            if let Ok(ones_t) = Tensor::from_slice(&ones, (r, 1), out.device()) {
                                if let Ok(scalar) = scale_layer.forward(&ones_t) {
                                    if let Ok(scaled) = out.mul(&scalar) {
                                        out = scaled;
                                    }
                                }
                            }
                        } else if let Ok(n) = out.shape().dims1() {
                            let ones = vec![1.0f32; n];
                            if let Ok(ones_t) = Tensor::from_slice(&ones, (n, 1), out.device()) {
                                if let Ok(scalar) = scale_layer.forward(&ones_t) {
                                    if let Ok(scaled) = out.mul(&scalar) {
                                        out = scaled;
                                    }
                                }
                            }
                        }
                    }

                    Ok(out)
                } else {
                    Ok(x.clone())
                }
            }
            Err(e) => {
                log::error!("[DecoderSmall] forward pass failed: {:?}", e);
                Err(e)
            }
        }
    }
}

impl fmt::Debug for DecoderSmall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecoderSmall")
            .field("nn", &"<Sequential>")
            .finish()
    }
}

/// Wrapper enum so models can hold either the legacy linear decoder or the new MLP
/// head without changing the rest of the model API. It dispatches forward calls
/// to the contained implementation.
#[derive(Clone)]
pub enum DecoderHead {
    Linear(DecoderLinear),
    MLP(DecoderMLP),
    Small(DecoderSmall),
}

impl Module for DecoderHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            DecoderHead::Linear(l) => l.forward(x),
            DecoderHead::MLP(m) => m.forward(x),
            DecoderHead::Small(s) => s.forward(x),
        }
    }
}

impl fmt::Debug for DecoderHead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecoderHead::Linear(_) => f.debug_struct("DecoderHead::Linear").finish(),
            DecoderHead::MLP(_) => f.debug_struct("DecoderHead::MLP").finish(),
            DecoderHead::Small(_) => f.debug_struct("DecoderHead::Small").finish(),
        }
    }
}

impl Module for DecoderLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use forward_all to capture outputs after each sub-layer
        match self.nn.forward_all(x) {
            Ok(outputs) => {
                // outputs contains intermediate results after each layer
                // outputs[0] = after linear1 (pre-activation)
                // outputs[1] = after prelu (post-activation)
                // outputs[2] = after linear2 (final)

                // Return the final output
                if let Some(last) = outputs.last() {
                    Ok(last.clone())
                } else {
                    Ok(x.clone())
                }
            }
            Err(e) => {
                log::error!("[DecoderLinear] forward pass failed: {:?}", e);
                Err(e)
            }
        }
    }
}

impl fmt::Debug for DecoderLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecoderLinear")
            .field("nn", &"<Sequential>")
            .finish()
    }
}

#[derive(Debug, Clone)]
struct AAEmbedding {
    embeddings: nn::Embedding,
}

impl AAEmbedding {
    fn new(hidden_size: usize, vb: &nn::VarBuilder) -> Result<Self> {
        // Create the embedding layer
        let embeddings = nn::embedding(AA_EMBEDDING_SIZE, hidden_size, vb.pp("embedding"))?;

        Ok(Self { embeddings })
    }

    fn from_varstore(varstore: &nn::VarBuilder, hidden_size: usize, name: &str) -> Result<Self> {
        let weight = varstore.get((AA_EMBEDDING_SIZE, hidden_size), name)?;
        let embeddings = nn::Embedding::new(weight, hidden_size);
        Ok(Self { embeddings })
    }
}

impl Module for AAEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::I64)?;
        self.embeddings.forward(&x)
    }
}

/// transform sequence input into a positional representation
#[derive(Debug, Clone)]
struct PositionalEncoding {
    pe: Tensor,
}

impl PositionalEncoding {
    fn new(out_features: usize, max_len: usize, device: &Device) -> Result<Self> {
        let position = Tensor::arange(0, max_len as u32, device)?.unsqueeze(1)?;

        let scalar = Tensor::new(-2.0 * std::f32::consts::PI / out_features as f32, device)?;
        let div_term = Tensor::arange(0f32, (out_features / 2) as f32, device)?
            .mul(&scalar)?
            .exp()?;

        let pe = Tensor::zeros((1, max_len, out_features), DType::F32, device)?;

        let sin_term = position
            .to_dtype(DType::F32)?
            .matmul(&div_term.unsqueeze(0)?)?
            .sin()?;
        let cos_term = position
            .to_dtype(DType::F32)?
            .matmul(&div_term.unsqueeze(0)?)?
            .cos()?;

        // Assign sine terms
        for i in 0..(out_features / 2) {
            pe.slice_assign(
                &[0..1, 0..max_len, (2 * i)..(2 * i + 1)],
                &sin_term.narrow(1, i, 1)?,
            )?;
        }

        // Assign cosine terms
        for i in 0..(out_features / 2) {
            pe.slice_assign(
                &[0..1, 0..max_len, (2 * i + 1)..(2 * i + 2)],
                &cos_term.narrow(1, i, 1)?,
            )?;
        }

        Ok(Self { pe })
    }

    fn from_varstore(
        varstore: &nn::VarBuilder,
        out_features: usize,
        max_len: usize,
        name: &str,
    ) -> Result<Self> {
        let pe = varstore.get((1, max_len, out_features), name);
        Ok(Self { pe: pe.unwrap() })
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let batch_size = x.dim(0)?;

        let pe_slice = self.pe.narrow(1, 0, seq_len)?;
        let pe_expanded = pe_slice.expand(&[batch_size, seq_len, 248])?; // Match batch size

        x.add(&pe_expanded)
    }
}

/// Encodes the modification vector in a single layer feed forward network, but not transforming the first k features
#[derive(Debug, Clone)]
struct ModEmbeddingFixFirstK {
    k: usize,
    nn: nn::Linear,
}

impl ModEmbeddingFixFirstK {
    fn new(
        mod_feature_size: usize,
        out_features: usize,
        varbuilder: &nn::VarBuilder,
    ) -> Result<Self> {
        let k = 6;
        let nn = nn::linear_no_bias(mod_feature_size - k, out_features - k, varbuilder.pp("nn"))?;
        Ok(Self { k, nn })
    }

    fn from_varstore(
        varstore: &nn::VarBuilder,
        mod_feature_size: usize,
        out_features: usize,
        name: &str,
    ) -> Result<Self> {
        let k = 6;
        let weight = varstore.get((out_features - k, MOD_FEATURE_SIZE - k), name);
        let nn = nn::Linear::new(weight.unwrap(), None);
        Ok(Self { k, nn })
    }
}

impl Module for ModEmbeddingFixFirstK {
    fn forward(&self, mod_x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = mod_x.dims3()?;

        // Split the input tensor
        let first_k = mod_x.narrow(D::Minus1, 0, self.k)?;
        let rest = mod_x.narrow(D::Minus1, self.k, mod_x.dim(D::Minus1)? - self.k)?;

        // Apply the linear transformation to the rest
        let transformed = self.nn.forward(&rest.contiguous()?)?;

        // Concatenate the results
        let out = Tensor::cat(&[&first_k, &transformed], D::Minus1)?;

        Ok(out)
    }
}

/// Encodes AA (26 AA letters) and modification vector
#[derive(Debug, Clone)]
pub struct Input26aaModPositionalEncoding {
    mod_nn: ModEmbeddingFixFirstK,
    aa_emb: AAEmbedding,
    pos_encoder: PositionalEncoding,
}

impl Input26aaModPositionalEncoding {
    fn new(_out_features: usize, _max_len: usize, _device: &Device) -> Result<Self> {
        todo!("new untrained instance of Input26aaModPositionalEncoding not implemented");
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        out_features: usize,
        max_len: usize,
        names: Vec<&str>,
    ) -> Result<Self> {
        let mod_hidden = 8;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::from_varstore(
                varstore,
                MOD_FEATURE_SIZE,
                mod_hidden,
                names[0],
            )?,
            aa_emb: AAEmbedding::from_varstore(varstore, out_features - mod_hidden, names[1])?,
            pos_encoder: PositionalEncoding::from_varstore(
                varstore,
                out_features,
                max_len,
                names[2],
            )?,
        })
    }

    pub fn forward(&self, aa_indices: &Tensor, mod_x: &Tensor) -> Result<Tensor> {
        // Compute embeddings
        let mod_out = self.mod_nn.forward(mod_x)?;
        let aa_out = self.aa_emb.forward(aa_indices)?;

        // Concatenate aa and mod embeddings and apply positional encoding
        let concatenated = Tensor::cat(&[&aa_out, &mod_out], 2)?;

        self.pos_encoder.forward(&concatenated)
    }
}

/// Encodes Charge state, Normalized Collision Energy (NCE) and Instrument for a given spectrum into a 'meta' single layer network
#[derive(Debug, Clone)]
pub struct MetaEmbedding {
    nn: nn::Linear,
}

impl MetaEmbedding {
    fn _new(out_features: usize, device: &Device) -> Result<Self> {
        let nn = nn::linear(
            MAX_INSTRUMENT_NUM + 1,
            out_features - 1,
            nn::VarBuilder::zeros(DType::F32, device).pp("linear"),
        )?;
        Ok(Self { nn })
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        out_features: usize,
        name: Vec<&str>,
    ) -> Result<Self> {
        let weight = varstore.get((out_features - 1, MAX_INSTRUMENT_NUM + 1), name[0]);
        let bias = varstore.get(out_features - 1, name[1]);
        let nn = nn::Linear::new(weight.unwrap(), Some(bias.unwrap()));
        Ok(Self { nn })
    }

    fn one_hot(&self, indices: &Tensor, num_classes: usize) -> Result<Tensor> {
        let batch_size = indices.dim(0)?;
        let device = indices.device();

        // Create a zeros vector for one-hot encoding
        let mut one_hot_data = vec![0.0f32; batch_size * num_classes];

        // Convert indices to a 1D vector of i64
        let indices = indices.to_dtype(DType::I64)?.to_vec1::<i64>()?;

        // Iterate over the batch and set the appropriate indices to 1.0
        for (i, &index) in indices.iter().enumerate() {
            if index >= 0 && (index as usize) < num_classes {
                one_hot_data[i * num_classes + index as usize] = 1.0;
            } else {
                return Err(candle_core::Error::Msg(format!(
                    "Index {} out of bounds for one-hot encoding",
                    index
                )));
            }
        }

        // Create a tensor from the one-hot data
        let one_hot = Tensor::from_slice(&one_hot_data, (batch_size, num_classes), device)?;

        Ok(one_hot)
    }

    pub fn forward(
        &self,
        charges: &Tensor,
        nces: &Tensor,
        instrument_indices: &Tensor,
    ) -> Result<Tensor> {
        // One-hot encode the instrument indices
        let inst_x = self.one_hot(
            &instrument_indices.to_dtype(DType::I64)?,
            MAX_INSTRUMENT_NUM,
        )?;

        // Ensure all tensors are on the same device
        let charges = &charges.to_device(inst_x.device())?;
        let nces = &nces.to_device(inst_x.device())?;

        // Concatenate the one-hot encoded instrument indices with NCEs
        let combined_input = Tensor::cat(&[&inst_x, nces], 1)?;

        // Pass through the linear layer
        let meta_x = self.nn.forward(&combined_input)?;

        // Concatenate the output with charges
        let meta_x = Tensor::cat(&[&meta_x, charges], 1)?;

        Ok(meta_x)
    }
}

/// Transformer NN based on HuggingFace's BertEncoder class
#[derive(Clone)]
pub struct HiddenHfaceTransformer {
    pub bert: transformers::models::bert::BertEncoder,
    pub config: transformers::models::bert::Config,
}

impl HiddenHfaceTransformer {
    fn _new(
        _hidden_dim: usize,
        _hidden_expand: usize,
        _nheads: usize,
        _nlayers: usize,
        _dropout: f64,
        _output_attentions: bool,
    ) -> Result<Self> {
        unimplemented!()
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        hidden_dim: usize,
        hidden_expand: usize,
        nheads: usize,
        nlayers: usize,
        dropout: f64,
        _output_attentions: bool,
    ) -> Result<Self> {
        let config = transformers::models::bert::Config {
            hidden_size: hidden_dim,
            intermediate_size: hidden_dim * hidden_expand,
            num_attention_heads: nheads,
            num_hidden_layers: nlayers,
            hidden_dropout_prob: dropout,
            ..Default::default()
        };

        let bert = transformers::models::bert::BertEncoder::load(varstore, &config).unwrap();

        Ok(Self { bert, config })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Determine batch size and sequence length from input tensor
        let (batch_size, seq_len, _) = x.shape().dims3()?; // Assuming x has shape [batch_size, seq_len, hidden_dim]

        // Create or adjust the mask
        let mask = match mask {
            Some(m) => {
                // Unsqueeze and repeat to match attention scores shape
                m.unsqueeze(1)?
                    .repeat(vec![1, self.config.num_attention_heads as usize, 1])? // Shape: [batch_size, n_heads, seq_len]
                    .unsqueeze(2)? // Add an extra dimension for broadcasting
            }
            None => {
                // Create a new mask filled with zeros
                let zeros_mask = Tensor::zeros((batch_size, 1, seq_len), DType::F32, x.device())?; // Shape: [batch_size, 1, seq_len]
                zeros_mask
                    .repeat(vec![1, self.config.num_attention_heads as usize, 1])? // Shape: [batch_size, n_heads, seq_len]
                    .unsqueeze(2)? // Add an extra dimension for broadcasting
            }
        };

        // Forward pass through BERT encoder
        self.bert.forward(&x.contiguous()?, &mask)
    }
}

impl fmt::Debug for HiddenHfaceTransformer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HiddenHfaceTransformer")
            .field("bert_encoder", &"<BertEncoder>")
            .finish()
    }
}

/// For transfer learning of modloss frags
#[derive(Clone)]
pub struct ModLossNN {
    modules: ModuleList,
}

impl ModLossNN {
    fn new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        hidden_dim: usize,
        hidden_expand: usize,
        nheads: usize,
        nlayers: usize,
        dropout: f64,
        output_attentions: bool,
        decoder_linear_output_dim: usize,
        bert_name: &str,
        decoder_linear_names: Vec<&str>,
        decoder_linear_bias_names: Vec<&str>,
    ) -> Result<Self> {
        let config = transformers::models::bert::Config {
            hidden_size: hidden_dim,
            intermediate_size: hidden_dim * hidden_expand,
            num_attention_heads: nheads,
            num_hidden_layers: nlayers,
            hidden_dropout_prob: dropout,
            ..Default::default()
        };

        let bert =
            transformers::models::bert::BertEncoder::load(varstore.clone().pp(bert_name), &config)
                .unwrap();
        let bert_module = BertEncoderModule::new(bert);

        let mut modules = ModuleList::new();
        modules.push(bert_module);

        let decoder_linear = DecoderLinear::from_varstore(
            &varstore,
            hidden_dim,
            decoder_linear_output_dim,
            decoder_linear_names,
            decoder_linear_bias_names,
        )
        .unwrap();
        modules.push(decoder_linear);

        Ok(Self { modules })
    }
}

impl Module for ModLossNN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut output = xs.clone();
        for module in self.modules.iter() {
            output = module.forward(&output)?;
        }
        Ok(output)
    }
}

impl fmt::Debug for ModLossNN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModLossNN")
            .field("modules[0]", &"<Hidden_HFace_Transformer>")
            .field("modules[1]", &"<DecoderLinear>")
            .finish()
    }
}

/// Extract sequence features using `Conv1D` with different kernel sizes (1(residue connection),3,5,7), and then concatenate the outputs of these Conv1Ds. The Output dim is 4*embedding_hidden.
#[derive(Debug, Clone)]
struct SeqCNN {
    cnn_short: nn::Conv1d,
    cnn_medium: nn::Conv1d,
    cnn_long: nn::Conv1d,
}

impl SeqCNN {
    pub fn new(embedding_hidden: usize, varbuilder: &nn::VarBuilder) -> Result<Self> {
        let cnn_short = nn::conv1d(
            embedding_hidden,
            embedding_hidden,
            3,
            nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            varbuilder.pp("cnn_short"),
        )?;

        let cnn_medium = nn::conv1d(
            embedding_hidden,
            embedding_hidden,
            5,
            nn::Conv1dConfig {
                padding: 2,
                ..Default::default()
            },
            varbuilder.pp("cnn_medium"),
        )?;

        let cnn_long = nn::conv1d(
            embedding_hidden,
            embedding_hidden,
            7,
            nn::Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            varbuilder.pp("cnn_long"),
        )?;

        Ok(Self {
            cnn_short,
            cnn_medium,
            cnn_long,
        })
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        embedding_hidden: usize,
        names_weight: Vec<&str>,
        names_bias: Vec<&str>,
    ) -> Result<Self> {
        let cnn_short = nn::Conv1d::new(
            varstore
                .get((embedding_hidden, embedding_hidden, 3), names_weight[0])
                .unwrap(),
            Some(varstore.get(embedding_hidden, names_bias[0]).unwrap()),
            nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
        );

        let cnn_medium = nn::Conv1d::new(
            varstore
                .get((embedding_hidden, embedding_hidden, 5), names_weight[1])
                .unwrap(),
            Some(varstore.get(embedding_hidden, names_bias[1]).unwrap()),
            nn::Conv1dConfig {
                padding: 2,
                ..Default::default()
            },
        );

        let cnn_long = nn::Conv1d::new(
            varstore
                .get((embedding_hidden, embedding_hidden, 7), names_weight[2])
                .unwrap(),
            Some(varstore.get(embedding_hidden, names_bias[2]).unwrap()),
            nn::Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );

        Ok(Self {
            cnn_short,
            cnn_medium,
            cnn_long,
        })
    }
}

impl Module for SeqCNN {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?;

        let short = match self.cnn_short.forward(&x) {
            Ok(output) => output,
            Err(e) => {
                log::error!("[SeqCNN::forward] cnn_short.forward failed: {:?}", e);
                return Err(e);
            }
        };

        let medium = self.cnn_medium.forward(&x)?;

        let long = self.cnn_long.forward(&x)?;

        let output = Tensor::cat(&[x, short, medium, long], 1)?;

        Ok(output.transpose(1, 2)?)
    }
}

/// LSTM applied on sequence input
#[derive(Debug, Clone)]
struct SeqLSTM {
    lstm: BidirectionalLSTM,
}

impl SeqLSTM {
    fn _new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        input_size: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let lstm = BidirectionalLSTM::new(input_size, hidden_dim, num_layers, &varstore)?;
        Ok(Self { lstm })
    }
}

impl Module for SeqLSTM {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.lstm.forward(&x)?)
    }
}

/// Transformer block applied on sequence input using a custom Transformer encoder implementation.
/// This replaces the LSTM with a Transformer encoder for sequence modeling.
#[derive(Debug, Clone)]
pub struct SeqTransformer {
    encoder: TransformerEncoder,
    training: bool,
}

impl SeqTransformer {
    /// Construct a new transformer encoder block for sequence modeling.
    ///
    /// # Arguments
    /// * `varbuilder` - The variable builder for creating the model parameters.
    /// * `input_dim` - The input embedding dimension (e.g., CNN output).
    /// * `model_dim` - The internal model dimension of the transformer.
    /// * `ff_dim` - The feedforward hidden layer dimension.
    /// * `num_heads` - Number of attention heads.
    /// * `num_layers` - Number of transformer encoder layers.
    /// * `max_len` - Maximum input sequence length.
    /// * `dropout_prob` - Dropout probability.
    /// * `device` - The device to place the tensors on.
    pub fn new(
        varbuilder: &nn::VarBuilder,
        input_dim: usize,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout: f32,
        device: &Device,
    ) -> Result<Self> {
        let encoder = TransformerEncoder::new(
            varbuilder, input_dim, model_dim, ff_dim, num_heads, num_layers, max_len, dropout,
            device,
        )?;
        Ok(Self {
            encoder,
            training: true,
        })
    }

    /// Load a transformer encoder from a varstore (used when loading from pre-trained weights).
    pub fn from_varstore(
        varstore: nn::VarBuilder,
        input_dim: usize,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout_prob: f32,
        device: &Device,
    ) -> Result<Self> {
        let encoder = TransformerEncoder::new(
            &varstore,
            input_dim,
            model_dim,
            ff_dim,
            num_heads,
            num_layers,
            max_len,
            dropout_prob,
            device,
        )?;
        Ok(Self {
            encoder,
            training: true,
        })
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward_with_mask(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        Ok(self.encoder.forward_with_mask(x, mask, self.training)?)
    }
}

impl Module for SeqTransformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.encoder.forward_with_mask(x, None, self.training)?)
    }
}

/// apply linear transformation and tensor rescaling with softmax
#[derive(Debug, Clone)]
struct SeqAttentionSum {
    attention: nn::Linear,
}

impl SeqAttentionSum {
    pub fn new(hidden_dim: usize, varbuilder: &nn::VarBuilder) -> Result<Self> {
        let attention = nn::linear_no_bias(hidden_dim, 1, varbuilder.pp("attn.0"))?;
        Ok(Self { attention })
    }

    pub fn from_varstore(varstore: nn::VarBuilder, hidden_dim: usize, name: &str) -> Result<Self> {
        let attention = nn::Linear::new(varstore.get((1, hidden_dim), name).unwrap(), None);
        Ok(Self { attention })
    }
}

impl Module for SeqAttentionSum {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attention_weights = match self.attention.forward(x) {
            Ok(weights) => weights,
            Err(e) => {
                log::error!("Attention forward pass failed: {}", e);
                return Err(e);
            }
        };

        // Apply softmax to normalize weights
        // TODO: This is done in the model itself in the PyTorch implementation
        let attention_weights = nn::ops::softmax(&attention_weights, 1)?;

        // Explicitly broadcast attention_weights to match x's shape
        let broadcasted_weights = attention_weights.broadcast_as(x.shape())?;

        // Element-wise multiplication
        let weighted = x.mul(&broadcasted_weights)?;

        // Sum over the sequence length dimension (dim 1)
        let output = weighted.sum(1)?;

        Ok(output)
    }
}

/// Encode AAs (26 AA letters) and modifications by CNN and LSTM layers, then by 'SeqAttentionSum'.
#[derive(Debug, Clone)]
pub struct Encoder26aaModCnnLstmAttnSum {
    mod_nn: ModEmbeddingFixFirstK,
    input_cnn: SeqCNN,
    input_lstm: SeqLSTM,
    attn_sum: SeqAttentionSum,
}

impl Encoder26aaModCnnLstmAttnSum {
    fn _new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        mod_hidden_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        names_mod_nn: Vec<&str>,
        names_input_cnn_weight: Vec<&str>,
        names_input_cnn_bias: Vec<&str>,
        lstm_pp: &str,
        names_attn_sum: Vec<&str>,
    ) -> Result<Self> {
        let input_dim = AA_EMBEDDING_SIZE + mod_hidden_dim;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::from_varstore(
                &varstore,
                MOD_FEATURE_SIZE,
                mod_hidden_dim,
                names_mod_nn[0],
            )?,
            input_cnn: SeqCNN::from_varstore(
                varstore.clone(),
                input_dim,
                names_input_cnn_weight,
                names_input_cnn_bias,
            )?,
            input_lstm: SeqLSTM::from_varstore(
                varstore.pp(lstm_pp).clone(),
                input_dim * 4,
                hidden_dim,
                num_layers,
            )?,
            attn_sum: SeqAttentionSum::from_varstore(
                varstore.clone(),
                hidden_dim * 2,
                names_attn_sum[0],
            )?,
        })
    }

    pub fn forward(&self, aa_indices: &Tensor, mod_x: &Tensor) -> Result<Tensor> {
        let mod_out = self.mod_nn.forward(mod_x)?;

        let additional_tensors: Vec<&Tensor> = vec![&mod_out];

        let x = aa_one_hot(&aa_indices, &additional_tensors)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // let (mean, min, max) = get_tensor_stats(&x)?;
        // log::trace!("[Encoder26aaModCnnLstmAttnSum] one-hot output stats - min: {min}, max: {max}, mean: {mean}");

        let x = self.input_cnn.forward(&x)?;
        let (b, s, d) = x.dims3()?;
        // println!("x (post input_cnn): batch size: {b}, seq len: {s}, embedding dim: {d}");

        let x = self.input_lstm.forward(&x)?;
        let (b, s, d) = x.dims3()?;
        // println!("x (post input_lstm): batch size: {b}, seq len: {s}, embedding dim: {d}");

        let x = self.attn_sum.forward(&x)?;

        Ok(x)
    }
}

/// Encode AAs (26 AA letters), modifications and charge by CNN and LSTM layers, then by 'SeqAttentionSum'.
#[derive(Debug, Clone)]
pub struct Encoder26aaModChargeCnnLstmAttnSum {
    mod_nn: ModEmbeddingFixFirstK,
    input_cnn: SeqCNN,
    input_lstm: SeqLSTM,
    attn_sum: SeqAttentionSum,
}

impl Encoder26aaModChargeCnnLstmAttnSum {
    fn _new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        mod_hidden_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        names_mod_nn: Vec<&str>,
        names_input_cnn_weight: Vec<&str>,
        names_input_cnn_bias: Vec<&str>,
        lstm_pp: &str,
        names_attn_sum: Vec<&str>,
    ) -> Result<Self> {
        let input_dim = AA_EMBEDDING_SIZE + mod_hidden_dim + 1;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::from_varstore(
                &varstore,
                MOD_FEATURE_SIZE,
                mod_hidden_dim,
                names_mod_nn[0],
            )?,
            input_cnn: SeqCNN::from_varstore(
                varstore.clone(),
                input_dim,
                names_input_cnn_weight,
                names_input_cnn_bias,
            )?,
            input_lstm: SeqLSTM::from_varstore(
                varstore.pp(lstm_pp).clone(),
                input_dim * 4,
                hidden_dim,
                num_layers,
            )?,
            attn_sum: SeqAttentionSum::from_varstore(
                varstore.clone(),
                hidden_dim * 2,
                names_attn_sum[0],
            )?,
        })
    }

    pub fn forward(&self, aa_indices: &Tensor, mod_x: &Tensor, charges: &Tensor) -> Result<Tensor> {
        let mod_x = self.mod_nn.forward(mod_x)?;

        let charges_repeated = charges.unsqueeze(1)?.repeat(&[1, mod_x.dim(1)?, 1])?;
        let additional_tensors: Vec<&Tensor> = vec![&mod_x, &charges_repeated];

        let x = aa_one_hot(&aa_indices, &additional_tensors)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let (mean, min, max) = get_tensor_stats(&x)?;
        log::trace!(
            "[Encoder26aaModChargeCnnLstmAttnSum] one-hot output stats - min: {min}, max: {max}, mean: {mean}"
        );

        let x = self.input_cnn.forward(&x)?;

        let x = self.input_lstm.forward(&x)?;

        let x = self.attn_sum.forward(&x)?;

        Ok(x)
    }
}

/// Encode AAs (26 AA letters) and modifications using CNN + Transformer + AttentionSum.
#[derive(Debug, Clone)]
pub struct Encoder26aaModCnnTransformerAttnSum {
    mod_nn: ModEmbeddingFixFirstK,
    input_cnn: SeqCNN,
    proj_cnn_to_transformer: candle_nn::Linear,
    input_transformer: SeqTransformer,
    attn_sum: SeqAttentionSum,
}

impl Encoder26aaModCnnTransformerAttnSum {
    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        mod_hidden_dim: usize,
        hidden_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout_prob: f32,
        names_mod_nn: Vec<&str>,
        names_input_cnn_weight: Vec<&str>,
        names_input_cnn_bias: Vec<&str>,
        transformer_pp: &str,
        names_attn_sum: Vec<&str>,
        device: &Device,
    ) -> Result<Self> {
        let input_dim = AA_EMBEDDING_SIZE + mod_hidden_dim;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::from_varstore(
                &varstore,
                MOD_FEATURE_SIZE,
                mod_hidden_dim,
                names_mod_nn[0],
            )?,
            input_cnn: SeqCNN::from_varstore(
                varstore.clone(),
                input_dim,
                names_input_cnn_weight,
                names_input_cnn_bias,
            )?,
            // Retrieve the raw weight tensor from the varstore so we can both
            // construct the Linear module and optionally retain a clone for
            // in-process diagnostic dot-product checks.
            // Diagnostic probe: attempt multiple get() variants at this exact
            // call-site (local name, qualified name, and scoped pp(transformer_pp))
            // and write their results to separate files so we can see what the
            // encoder actually resolves at runtime.
            proj_cnn_to_transformer: {
                // Try local name first; if not present in this varstore scope,
                // fall back to a qualified name derived from the transformer
                // prefix (e.g. "rt_encoder.proj_cnn_to_transformer.weight").

                let w = match varstore.get(
                    (hidden_dim, input_dim * 4),
                    "proj_cnn_to_transformer.weight",
                ) {
                    Ok(t) => t,
                    Err(_) => {
                        let prefix = transformer_pp.split('.').next().unwrap_or(transformer_pp);
                        let qualified = format!("{}.proj_cnn_to_transformer.weight", prefix);
                        varstore.get((hidden_dim, input_dim * 4), &qualified)?
                    }
                };

                let lin = candle_nn::Linear::new(w.clone(), None);

                lin
            },
            input_transformer: SeqTransformer::from_varstore(
                varstore.pp(transformer_pp).clone(),
                input_dim * 4,
                hidden_dim,
                ff_dim,
                num_heads,
                num_layers,
                max_len,
                dropout_prob,
                device,
            )?,
            attn_sum: SeqAttentionSum::from_varstore(
                varstore.clone(),
                hidden_dim,
                names_attn_sum[0],
            )?,
        })
    }

    /// Construct a CNN+Transformer+Attention encoder from scratch (no pretrained weights).
    pub fn new(
        varbuilder: &nn::VarBuilder,
        mod_hidden_dim: usize,
        hidden_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout_prob: f32,
        device: &Device,
    ) -> Result<Self> {
        let input_dim = AA_EMBEDDING_SIZE + mod_hidden_dim;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::new(
                MOD_FEATURE_SIZE,
                mod_hidden_dim,
                &varbuilder.pp("mod_nn"),
            )?,
            input_cnn: SeqCNN::new(input_dim, &varbuilder.pp("input_cnn"))?,
            proj_cnn_to_transformer: candle_nn::linear_no_bias(
                input_dim * 4,
                hidden_dim,
                varbuilder.pp("proj_cnn_to_transformer"),
            )?,
            input_transformer: SeqTransformer::new(
                &varbuilder.pp("input_transformer"),
                input_dim * 4,
                hidden_dim,
                ff_dim,
                num_heads,
                num_layers,
                max_len,
                dropout_prob,
                device,
            )?,
            attn_sum: SeqAttentionSum::new(hidden_dim, &varbuilder.pp("attn_sum"))?,
        })
    }

    pub fn forward(&self, aa_indices: &Tensor, mod_x: &Tensor) -> Result<Tensor> {
        let mod_x = self.mod_nn.forward(mod_x)?;

        let additional_tensors: Vec<&Tensor> = vec![&mod_x];

        let x = aa_one_hot(aa_indices, &additional_tensors)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let (mean, min, max) = get_tensor_stats(&x)?;
        // log::trace!("[Encoder26aaModCnnTransformerAttnSum] one-hot output stats - min: {min}, max: {max}, mean: {mean}");

        if !mean.is_finite() || !min.is_finite() || !max.is_finite() {
            log::error!(
                "ERROR [Encoder26aaModCnnTransformerAttnSum] aa_one_hot produced non-finite tensor stats: mean={mean}, min={min}, max={max}"
            );
            candle_core::bail!("ERROR: Non-finite values found in peptide encoding output.");
        }

        let zero = Tensor::zeros(aa_indices.shape(), aa_indices.dtype(), aa_indices.device())?;
        let padding_mask = aa_indices.eq(&zero)?.to_dtype(candle_core::DType::F32)?;

        let x = self.input_cnn.forward(&x)?;
        let x = x.contiguous()?;

        let x = self.proj_cnn_to_transformer.forward(&x)?;
        let x = x.contiguous()?;

        let x = self
            .input_transformer
            .forward_with_mask(&x, Some(&padding_mask))?;
        let x = x.contiguous()?;
        let x = self.attn_sum.forward(&x)?;

        Ok(x)
    }

    pub fn set_training(&mut self, training: bool) {
        self.input_transformer.set_training(training);
    }
}

/// Encode AAs (26 AA letters), modifications and Charge state using CNN + Transformer + AttentionSum.
#[derive(Debug, Clone)]
pub struct Encoder26aaModChargeCnnTransformerAttnSum {
    mod_nn: ModEmbeddingFixFirstK,
    input_cnn: SeqCNN,
    proj_cnn_to_transformer: candle_nn::Linear,
    input_transformer: SeqTransformer,
    attn_sum: SeqAttentionSum,
}

impl Encoder26aaModChargeCnnTransformerAttnSum {
    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        mod_hidden_dim: usize,
        hidden_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout_prob: f32,
        names_mod_nn: Vec<&str>,
        names_input_cnn_weight: Vec<&str>,
        names_input_cnn_bias: Vec<&str>,
        transformer_pp: &str,
        names_attn_sum: Vec<&str>,
        device: &Device,
    ) -> Result<Self> {
        let input_dim = AA_EMBEDDING_SIZE + mod_hidden_dim + 1;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::from_varstore(
                &varstore,
                MOD_FEATURE_SIZE,
                mod_hidden_dim,
                names_mod_nn[0],
            )?,
            input_cnn: SeqCNN::from_varstore(
                varstore.clone(),
                input_dim,
                names_input_cnn_weight,
                names_input_cnn_bias,
            )?,
            proj_cnn_to_transformer: {
                // Try local name first; if not present in this varstore scope,
                // fall back to a qualified name derived from the transformer
                // prefix (e.g. "rt_encoder.proj_cnn_to_transformer.weight").
                let w = match varstore.get(
                    (hidden_dim, input_dim * 4),
                    "proj_cnn_to_transformer.weight",
                ) {
                    Ok(t) => t,
                    Err(_) => {
                        let prefix = transformer_pp.split('.').next().unwrap_or(transformer_pp);
                        let qualified = format!("{}.proj_cnn_to_transformer.weight", prefix);
                        varstore.get((hidden_dim, input_dim * 4), &qualified)?
                    }
                };
                candle_nn::Linear::new(w.clone(), None)
            },
            input_transformer: SeqTransformer::from_varstore(
                varstore.pp(transformer_pp).clone(),
                input_dim * 4,
                hidden_dim,
                ff_dim,
                num_heads,
                num_layers,
                max_len,
                dropout_prob,
                device,
            )?,
            attn_sum: SeqAttentionSum::from_varstore(
                varstore.clone(),
                hidden_dim,
                names_attn_sum[0],
            )?,
        })
    }

    /// Construct a CNN+Transformer+Attention encoder from scratch (no pretrained weights).
    pub fn new(
        varbuilder: &nn::VarBuilder,
        mod_hidden_dim: usize,
        hidden_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout_prob: f32,
        device: &Device,
    ) -> Result<Self> {
        let input_dim = AA_EMBEDDING_SIZE + mod_hidden_dim + 1;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::new(
                MOD_FEATURE_SIZE,
                mod_hidden_dim,
                &varbuilder.pp("mod_nn"),
            )?,
            input_cnn: SeqCNN::new(input_dim, &varbuilder.pp("input_cnn"))?,
            proj_cnn_to_transformer: candle_nn::linear_no_bias(
                input_dim * 4,
                hidden_dim,
                varbuilder.pp("proj_cnn_to_transformer"),
            )?,
            input_transformer: SeqTransformer::new(
                &varbuilder.pp("input_transformer"),
                input_dim * 4,
                hidden_dim,
                ff_dim,
                num_heads,
                num_layers,
                max_len,
                dropout_prob,
                device,
            )?,
            attn_sum: SeqAttentionSum::new(hidden_dim, &varbuilder.pp("attn_sum"))?,
        })
    }

    pub fn forward(&self, aa_indices: &Tensor, mod_x: &Tensor, charges: &Tensor) -> Result<Tensor> {
        let mod_x = self.mod_nn.forward(mod_x)?;
        let charges_repeated = charges.unsqueeze(1)?.repeat(&[1, mod_x.dim(1)?, 1])?;

        let additional_tensors: Vec<&Tensor> = vec![&mod_x, &charges_repeated];

        let x = aa_one_hot(aa_indices, &additional_tensors)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let (mean, min, max) = get_tensor_stats(&x)?;
        log::trace!(
            "[Encoder26aaModChargeCnnTransformerAttnSum] one-hot output stats - min: {min}, max: {max}, mean: {mean}"
        );

        let x = self.input_cnn.forward(&x)?;
        let x = x.contiguous()?;

        let x = self.proj_cnn_to_transformer.forward(&x)?;
        let x = x.contiguous()?;

        let x = self.input_transformer.forward(&x)?;
        let x = x.contiguous()?;
        let x = self.attn_sum.forward(&x)?;

        Ok(x)
    }

    /// Set training/eval mode for nested components (e.g., SeqTransformer)
    /// so that dropout and other training-only behavior can be toggled.
    pub fn set_training(&mut self, training: bool) {
        self.input_transformer.set_training(training);
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_decoder_linear_new() -> Result<()> {
        // Set up the device and random seed
        let device = Device::Cpu;

        // Create an instance of DecoderLinear
        let in_features = 10;
        let out_features = 5;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let decoder_linear = DecoderLinear::new(in_features, out_features, &vb.pp("output_nn"))?;

        // Create the input tensor
        let x = Tensor::new(
            &[
                [
                    0.0698, -1.6635, -0.7651, 0.7928, 0.2882, -1.1787, -2.1401, -0.4903, 0.6598,
                    -0.0703,
                ],
                [
                    0.2745, 0.5569, -0.2322, -0.0597, 0.1211, 0.2984, 1.3448, 1.4614, 1.0566,
                    -0.5461,
                ],
                [
                    -0.2178, -0.2809, -0.3869, -0.3572, -1.1568, -1.7660, -2.5380, -0.0334,
                    -1.7017, 0.5863,
                ],
            ],
            &device,
        )?
        .to_dtype(DType::F32)?;

        // Perform forward pass
        let output = decoder_linear.forward(&x)?;

        println!("Output:\n{}", output);

        // Expected output from PyTorch
        let expected_output = Tensor::new(
            &[
                [-0.3355, 0.0190, 0.3605, -0.6334, -0.4229],
                [0.0960, 0.1447, 0.1348, -0.1506, 0.0620],
                [-0.4250, 0.2105, -0.0994, -0.5364, -0.6821],
            ],
            &device,
        )?
        .to_dtype(DType::F32)?;

        // Check output shape
        assert_eq!(
            output.shape(),
            expected_output.shape(),
            "Output shape mismatch"
        );

        // Check output values
        let output_vec = output.to_vec2::<f32>()?;
        let expected_vec = expected_output.to_vec2::<f32>()?;

        for (row_out, row_exp) in output_vec.iter().zip(expected_vec.iter()) {
            for (val_out, val_exp) in row_out.iter().zip(row_exp.iter()) {
                assert!(
                    (val_out - val_exp).abs() < 1e-4,
                    "Output value {:.4} doesn't match expected {:.4}",
                    val_out,
                    val_exp
                );
            }
        }

        // Print shapes for verification
        println!("Input shape: {:?}", x.shape());
        println!("Output shape: {:?}", output.shape());
        println!("Output:\n{}", output);

        Ok(())
    }

    #[test]
    fn test_decoder_linear() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let decoder_linear = DecoderLinear::from_varstore(
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

        println!("decoder_linear : {:?}", decoder_linear);
    }

    #[test]
    fn test_mod_embedding_fix_first_k() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let mod_nn = ModEmbeddingFixFirstK::from_varstore(
            &var_store,
            MOD_FEATURE_SIZE,
            8,
            "input_nn.mod_nn.nn.weight",
        )
        .unwrap();

        println!("mod_nn : {:?}", mod_nn);
    }

    #[test]
    fn test_aa_embedding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let aa_emb = AAEmbedding::from_varstore(&var_store, 256, "input_nn.aa_emb.weight").unwrap();

        println!("aa_emb : {:?}", aa_emb);
    }

    #[test]
    fn test_positional_encoding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let pos_enc =
            PositionalEncoding::from_varstore(&var_store, 248, 200, "input_nn.pos_encoder.pe")
                .unwrap();

        println!("pos_enc : {:?}", pos_enc);
    }

    #[test]
    fn test_input_26aa_mod_positional_encoding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let input_26aa_mod_pos_enc = Input26aaModPositionalEncoding::from_varstore(
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

        println!("input_26aa_mod_pos_enc : {:?}", input_26aa_mod_pos_enc);
    }

    #[test]
    fn test_meta_embedding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let meta_emb = MetaEmbedding::from_varstore(
            &var_store,
            8,
            vec!["meta_nn.nn.weight", "meta_nn.nn.bias"],
        )
        .unwrap();

        println!("meta_emb : {:?}", meta_emb);
    }

    #[test]
    fn test_hidden_hface_transformer() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let hidden_hface_transformer = HiddenHfaceTransformer::from_varstore(
            var_store.pp("hidden_nn.bert"),
            256,
            4,
            8,
            4,
            0.1,
            false,
        )
        .unwrap();

        println!("hidden_hface_transformer : {:?}", hidden_hface_transformer);
    }

    #[test]
    fn test_mod_loss_nn() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let mod_loss_nn = ModLossNN::from_varstore(
            var_store,
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

        println!("mod_loss_nn : {:?}", mod_loss_nn);
    }

    #[test]
    fn test_seq_cnn() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let seq_cnn = SeqCNN::from_varstore(
            var_store,
            36,
            vec![
                "ccs_encoder.input_cnn.cnn_short.weight",
                "ccs_encoder.input_cnn.cnn_medium.weight",
                "ccs_encoder.input_cnn.cnn_long.weight",
            ],
            vec![
                "ccs_encoder.input_cnn.cnn_short.bias",
                "ccs_encoder.input_cnn.cnn_medium.bias",
                "ccs_encoder.input_cnn.cnn_long.bias",
            ],
        )
        .unwrap();

        println!("seq_cnn : {:?}", seq_cnn);
    }

    #[test]
    fn test_seq_lstm() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let seq_lstm =
            SeqLSTM::from_varstore(var_store.pp("ccs_encoder.hidden_nn"), 144, 128, 2).unwrap();

        println!("seq_lstm : {:?}", seq_lstm);
    }

    #[test]
    fn test_seq_attention_sum() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let seq_attention_sum =
            SeqAttentionSum::from_varstore(var_store, 256, "ccs_encoder.attn_sum.attn.0.weight")
                .unwrap();

        println!("seq_attention_sum : {:?}", seq_attention_sum);
    }
}
