use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::bert::BertEncoder;

#[derive(Clone)]
pub struct ModuleList {
    modules: Vec<Arc<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Arc::new(module));
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for ModuleList {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut output = xs.clone();
        for module in &self.modules {
            output = module.forward(&output)?;
        }
        Ok(output)
    }
}

impl Deref for ModuleList {
    type Target = Vec<Arc<dyn Module>>;

    fn deref(&self) -> &Self::Target {
        &self.modules
    }
}

impl DerefMut for ModuleList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.modules
    }
}

// BertEncoderModule remains the same
pub struct BertEncoderModule {
    encoder: BertEncoder,
}

impl BertEncoderModule {
    pub fn new(encoder: BertEncoder) -> Self {
        Self { encoder }
    }
}

impl Module for BertEncoderModule {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_mask = hidden_states.ones_like()?;
        self.encoder.forward(hidden_states, &attention_mask)
    }
}

/// A minimal Transformer encoder layer with multi-head self-attention, feedforward block,
/// dropout, and optional sinusoidal positional encoding and padding mask support.
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    pos_encoding: Tensor,
    dropout: Dropout,
    pub model_dim: usize,
}

impl TransformerEncoder {
    pub fn new(
        varbuilder: &VarBuilder,
        input_dim: usize,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout_prob: f32,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer = TransformerEncoderLayer::new(
                &varbuilder.pp(&format!("layer_{}", i)),
                model_dim,
                ff_dim,
                num_heads,
                dropout_prob,
            )?;
            layers.push(layer);
        }
        let pos_encoding = create_sinusoidal_encoding(max_len, model_dim, device)?;
        let dropout = Dropout::new(dropout_prob);
        Ok(Self {
            layers,
            pos_encoding,
            dropout,
            model_dim,
        })
    }

    pub fn forward_with_mask(
        &self,
        x: &Tensor,
        padding_mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let pe = self
            .pos_encoding
            .i((..t, ..))?
            .unsqueeze(0)?
            .broadcast_as((b, t, self.pos_encoding.dim(1)?))?;

        let mut out = x.broadcast_add(&pe)?;

        out = self.dropout.forward(&out, training)?;

        for (_i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out, padding_mask, training)?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout1: Dropout,
    dropout2: Dropout,
}

impl TransformerEncoderLayer {
    pub fn new(
        varbuilder: &VarBuilder,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(varbuilder, model_dim, model_dim, num_heads)?,
            ff: FeedForward::new(varbuilder, model_dim, ff_dim)?,
            norm1: candle_nn::layer_norm(
                model_dim,
                candle_nn::LayerNormConfig::default(),
                varbuilder.pp("norm1"),
            )?,
            norm2: candle_nn::layer_norm(
                model_dim,
                candle_nn::LayerNormConfig::default(),
                varbuilder.pp("norm2"),
            )?,
            dropout1: Dropout::new(dropout_prob),
            dropout2: Dropout::new(dropout_prob),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        let attn = self.self_attn.forward(x, mask)?;
        let tmp = self.dropout1.forward(&attn, training)?;
        let tmp2 = x.broadcast_add(&tmp)?;
        let x = self.norm1.forward(&tmp2)?;
        let ff = self.ff.forward(&x)?;
        let result = self
            .norm2
            .forward(&x.broadcast_add(&self.dropout2.forward(&ff, training)?)?)?;
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    proj_q: Linear,
    proj_k: Linear,
    proj_v: Linear,
    proj_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(
        varbuilder: &VarBuilder,
        input_dim: usize,
        model_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = model_dim / num_heads;
        Ok(Self {
            proj_q: candle_nn::linear(input_dim, model_dim, varbuilder.pp("proj_q"))?,
            proj_k: candle_nn::linear(input_dim, model_dim, varbuilder.pp("proj_k"))?,
            proj_v: candle_nn::linear(input_dim, model_dim, varbuilder.pp("proj_v"))?,
            proj_out: candle_nn::linear(model_dim, model_dim, varbuilder.pp("proj_out"))?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        // Project once, then reshape to (batch, heads, seq, head_dim)
        let q = self.proj_q.forward(x)?;
        let k = self.proj_k.forward(x)?;
        let v = self.proj_v.forward(x)?;

        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut scores = match q.matmul(&k_t) {
            Ok(s) => (s / (self.head_dim as f64).sqrt())?,
            Err(e) => {
                log::error!(
                    "[MultiHeadAttention] Failed during matmul for scores: {}",
                    e
                );
                return Err(e.into());
            }
        };

        if let Some(mask) = mask {
            // Expect mask shape [batch, seq]; expand to [batch, heads, seq, seq]
            let mask = mask
                .unsqueeze(1)? // (b, 1, seq)
                .unsqueeze(2)?; // (b, 1, 1, seq)
            let mask = match mask.broadcast_as(scores.shape()) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("[MultiHeadAttention] Failed to broadcast mask: {}", e);
                    return Err(e.into());
                }
            };
            let neg_inf = Tensor::new(-1e9f32, x.device())?;
            let scaled_mask = match mask.broadcast_mul(&neg_inf) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("[MultiHeadAttention] Failed to scale mask: {}", e);
                    return Err(e.into());
                }
            };
            scores = match scores.broadcast_add(&scaled_mask) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("[MultiHeadAttention] Failed during masking: {}", e);
                    return Err(e.into());
                }
            };
        }

        let attn = match candle_nn::ops::softmax(&scores, scores.dims().len() - 1) {
            Ok(a) => a,
            Err(e) => {
                log::error!("[MultiHeadAttention] Failed during softmax: {}", e);
                return Err(e.into());
            }
        };

        let context = match attn.matmul(&v) {
            Ok(ctx) => ctx
                .transpose(1, 2)?
                .reshape((b, t, self.num_heads * self.head_dim))?,
            Err(e) => {
                log::error!(
                    "[MultiHeadAttention] Failed during attention context computation: {}",
                    e
                );
                return Err(e.into());
            }
        };

        self.proj_out.forward(&context)
    }
}

#[derive(Debug, Clone)]
pub struct FeedForward {
    lin1: Linear,
    lin2: Linear,
}

impl FeedForward {
    pub fn new(varbuilder: &VarBuilder, model_dim: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            lin1: candle_nn::linear(model_dim, ff_dim, varbuilder.pp("lin1"))?,
            lin2: candle_nn::linear(ff_dim, model_dim, varbuilder.pp("lin2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(x)?.relu()?;
        self.lin2.forward(&x)
    }
}

/// Generate sinusoidal positional encoding like in "Attention is All You Need".
pub fn create_sinusoidal_encoding(
    seq_len: usize,
    model_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut pe = vec![0f32; seq_len * model_dim];
    for pos in 0..seq_len {
        for i in 0..model_dim {
            let angle = pos as f32 / (10000f32).powf(2. * (i / 2) as f32 / model_dim as f32);
            pe[pos * model_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::from_vec(pe, (seq_len, model_dim), device)
}
