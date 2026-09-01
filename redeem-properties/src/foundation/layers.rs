//! Candle layers used by the hierarchical graph/Transformer encoder.

use candle_core::{Module, ModuleT, Result, Tensor, D};
use candle_nn::{self as nn, ops, Dropout, LayerNorm, Linear, VarBuilder};

/// One residual graph-message-passing layer operating on dense residue graphs.
#[derive(Clone)]
pub struct GraphMessageLayer {
    self_projection: Linear,
    neighbor_projection: Linear,
    output_projection: Linear,
    norm: LayerNorm,
}

impl GraphMessageLayer {
    /// Build one graph layer.
    pub fn new(hidden_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            self_projection: nn::linear(hidden_dim, hidden_dim, vb.pp("self"))?,
            neighbor_projection: nn::linear(hidden_dim, hidden_dim, vb.pp("neighbor"))?,
            output_projection: nn::linear(hidden_dim, hidden_dim, vb.pp("output"))?,
            norm: nn::layer_norm(hidden_dim, 1e-5, vb.pp("norm"))?,
        })
    }

    /// Apply adjacency-normalized message passing.
    pub fn forward(
        &self,
        hidden: &Tensor,
        adjacency: &Tensor,
        atom_mask: &Tensor,
    ) -> Result<Tensor> {
        let (n_graphs, n_atoms, hidden_dim) = hidden.dims3()?;
        let degrees = adjacency.sum_keepdim(2)?.clamp(1e-6, f64::INFINITY)?;
        let normalized = adjacency.broadcast_div(&degrees)?;
        let neighbor_state = normalized.matmul(hidden)?;
        let self_state = self.self_projection.forward(hidden)?;
        let neighbor_state = self.neighbor_projection.forward(&neighbor_state)?;
        let update = (self_state + neighbor_state)?.relu()?;
        let update = self.output_projection.forward(&update)?;
        let hidden = self.norm.forward(&(hidden + update)?)?;
        let mask = atom_mask
            .reshape((n_graphs, n_atoms, 1))?
            .broadcast_as((n_graphs, n_atoms, hidden_dim))?;
        hidden.broadcast_mul(&mask)
    }
}

/// Multi-head self-attention used by the peptide Transformer.
#[derive(Clone)]
pub struct MultiHeadSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    /// Build a self-attention layer.
    pub fn new(model_dim: usize, num_heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let head_dim = model_dim / num_heads;
        Ok(Self {
            query: nn::linear_no_bias(model_dim, model_dim, vb.pp("query"))?,
            key: nn::linear_no_bias(model_dim, model_dim, vb.pp("key"))?,
            value: nn::linear_no_bias(model_dim, model_dim, vb.pp("value"))?,
            output: nn::linear(model_dim, model_dim, vb.pp("output"))?,
            num_heads,
            head_dim,
        })
    }

    /// Apply masked self-attention to `[batch, sequence, model_dim]` states.
    pub fn forward(&self, hidden: &Tensor, residue_mask: &Tensor) -> Result<Tensor> {
        let (batch, sequence, model_dim) = hidden.dims3()?;
        let q = self
            .query
            .forward(hidden)?
            .reshape((batch, sequence, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .key
            .forward(hidden)?
            .reshape((batch, sequence, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .value
            .forward(hidden)?
            .reshape((batch, sequence, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // `transpose` creates a strided view. Candle's CPU batched matmul
        // requires contiguous Q/K storage for this layout, so materialize the
        // transposed key view before computing attention scores. This matters
        // in particular for batch sizes greater than one.
        let key_transposed = k.transpose(2, 3)?.contiguous()?;
        let scores = q
            .matmul(&key_transposed)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        // Candle's ordinary tensor addition requires equal shapes and does not
        // implicitly broadcast. Expand the key mask explicitly from
        // `[batch, 1, 1, sequence]` to the attention-score shape before
        // applying it.
        let key_mask = residue_mask
            .affine(-1.0, 1.0)?
            .affine(-10_000.0, 0.0)?
            .unsqueeze(1)?
            .unsqueeze(1)?
            .broadcast_as((batch, self.num_heads, sequence, sequence))?;
        let probabilities = ops::softmax(&(scores + key_mask)?, D::Minus1)?;
        let context = probabilities
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, sequence, model_dim))?;
        self.output.forward(&context)
    }
}

/// Multi-head cross-attention from a query sequence into an encoded memory.
#[derive(Clone)]
pub struct MultiHeadCrossAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadCrossAttention {
    /// Build a cross-attention layer.
    pub fn new(model_dim: usize, num_heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let head_dim = model_dim / num_heads;
        Ok(Self {
            query: nn::linear_no_bias(model_dim, model_dim, vb.pp("query"))?,
            key: nn::linear_no_bias(model_dim, model_dim, vb.pp("key"))?,
            value: nn::linear_no_bias(model_dim, model_dim, vb.pp("value"))?,
            output: nn::linear(model_dim, model_dim, vb.pp("output"))?,
            num_heads,
            head_dim,
        })
    }

    /// Attend from `[batch, query_len, model_dim]` into
    /// `[batch, memory_len, model_dim]` using `memory_mask` as the key mask.
    pub fn forward(
        &self,
        query_hidden: &Tensor,
        memory: &Tensor,
        memory_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch, query_len, model_dim) = query_hidden.dims3()?;
        let (memory_batch, memory_len, memory_dim) = memory.dims3()?;
        if batch != memory_batch || model_dim != memory_dim {
            candle_core::bail!(
                "cross-attention shape mismatch: query [{batch}, {query_len}, {model_dim}], memory [{memory_batch}, {memory_len}, {memory_dim}]"
            );
        }
        let q = self
            .query
            .forward(query_hidden)?
            .reshape((batch, query_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .key
            .forward(memory)?
            .reshape((batch, memory_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .value
            .forward(memory)?
            .reshape((batch, memory_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let key_transposed = k.transpose(2, 3)?.contiguous()?;
        let scores = q
            .matmul(&key_transposed)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let key_mask = memory_mask
            .affine(-1.0, 1.0)?
            .affine(-10_000.0, 0.0)?
            .unsqueeze(1)?
            .unsqueeze(1)?
            .broadcast_as((batch, self.num_heads, query_len, memory_len))?;
        let probabilities = ops::softmax(&(scores + key_mask)?, D::Minus1)?;
        let context = probabilities
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, query_len, model_dim))?;
        self.output.forward(&context)
    }
}

/// Pre-norm Transformer encoder block for peptide-level sequence modelling.
#[derive(Clone)]
pub struct PeptideTransformerBlock {
    attention_norm: LayerNorm,
    attention: MultiHeadSelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward_in: Linear,
    feed_forward_out: Linear,
    dropout: Dropout,
}

impl PeptideTransformerBlock {
    /// Build one peptide Transformer block.
    pub fn new(
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout: f32,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            attention_norm: nn::layer_norm(model_dim, 1e-5, vb.pp("attention_norm"))?,
            attention: MultiHeadSelfAttention::new(model_dim, num_heads, vb.pp("attention"))?,
            feed_forward_norm: nn::layer_norm(model_dim, 1e-5, vb.pp("ff_norm"))?,
            feed_forward_in: nn::linear(model_dim, ff_dim, vb.pp("ff_in"))?,
            feed_forward_out: nn::linear(ff_dim, model_dim, vb.pp("ff_out"))?,
            dropout: Dropout::new(dropout),
        })
    }

    /// Forward pass with explicit training/evaluation dropout behavior.
    pub fn forward_t(&self, hidden: &Tensor, residue_mask: &Tensor, train: bool) -> Result<Tensor> {
        let normalized = self.attention_norm.forward(hidden)?;
        let attention = self.attention.forward(&normalized, residue_mask)?;
        let hidden = (hidden + self.dropout.forward_t(&attention, train)?)?;

        let normalized = self.feed_forward_norm.forward(&hidden)?;
        let ff = self.feed_forward_in.forward(&normalized)?.relu()?;
        let ff = self.feed_forward_out.forward(&ff)?;
        let hidden = (hidden + self.dropout.forward_t(&ff, train)?)?;

        let (batch, sequence, model_dim) = hidden.dims3()?;
        let mask = residue_mask
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, model_dim))?;
        hidden.broadcast_mul(&mask)
    }
}
