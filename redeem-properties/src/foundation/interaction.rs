//! Trainable spectrum-candidate interaction adapters for downstream peptide ranking.
//!
//! The adapter deliberately consumes frozen causal decoder states and frozen spectrum
//! memory.  This lets downstream ranking learn a task-specific spectrum<->candidate
//! compatibility transform without fine-tuning the large foundation backbone.

use super::layers::{FoundationLayerNorm, MultiHeadCrossAttention};
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};

/// Small trainable cross-attention adapter used by the v0.16.0 reranking lane.
///
/// Input candidate states and spectrum memory are expected to come from a frozen
/// [`super::causal::PeptideSpectrumCausalModel`].  Only the parameters owned by
/// this adapter are optimized.
#[derive(Clone)]
pub struct FoundationSpectrumCandidateInteractionAdapter {
    query_norm: FoundationLayerNorm,
    cross_attention: MultiHeadCrossAttention,
    feed_forward_norm: FoundationLayerNorm,
    feed_forward_in: Linear,
    feed_forward_out: Linear,
    output_norm: FoundationLayerNorm,
    residual_head: Linear,
    model_dim: usize,
}

impl FoundationSpectrumCandidateInteractionAdapter {
    /// Build one trainable interaction adapter.
    ///
    /// `num_heads` must divide `model_dim`. `bottleneck_dim` is the width of the
    /// adapter feed-forward bottleneck.
    pub fn new(
        model_dim: usize,
        num_heads: usize,
        bottleneck_dim: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        if model_dim == 0 || num_heads == 0 || model_dim % num_heads != 0 {
            candle_core::bail!(
                "interaction adapter requires positive model_dim divisible by num_heads"
            );
        }
        if bottleneck_dim == 0 {
            candle_core::bail!("interaction adapter bottleneck_dim must be positive");
        }
        let residual_weight = vb.pp("residual_head").get_with_hints(
            (1, model_dim),
            "weight",
            nn::Init::Const(0.0),
        )?;
        let residual_bias =
            vb.pp("residual_head")
                .get_with_hints(1, "bias", nn::Init::Const(0.0))?;
        Ok(Self {
            query_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("query_norm"))?,
            cross_attention: MultiHeadCrossAttention::new(
                model_dim,
                num_heads,
                vb.pp("cross_attention"),
            )?,
            feed_forward_norm: FoundationLayerNorm::new(
                model_dim,
                1e-5,
                vb.pp("feed_forward_norm"),
            )?,
            feed_forward_in: nn::linear(model_dim, bottleneck_dim, vb.pp("feed_forward_in"))?,
            feed_forward_out: nn::linear(bottleneck_dim, model_dim, vb.pp("feed_forward_out"))?,
            output_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("output_norm"))?,
            residual_head: Linear::new(residual_weight, Some(residual_bias)),
            model_dim,
        })
    }

    /// Compute one scalar residual score per candidate.
    ///
    /// Shapes:
    /// - `candidate_hidden`: `[batch, token_len, model_dim]`
    /// - `candidate_mask`: `[batch, token_len]`
    /// - `spectrum_memory`: `[batch, memory_len, model_dim]`
    /// - `spectrum_memory_mask`: `[batch, memory_len]`
    /// - output: `[batch]`
    pub fn forward(
        &self,
        candidate_hidden: &Tensor,
        candidate_mask: &Tensor,
        spectrum_memory: &Tensor,
        spectrum_memory_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch, token_len, model_dim) = candidate_hidden.dims3()?;
        if model_dim != self.model_dim {
            candle_core::bail!(
                "interaction adapter candidate dim {model_dim} != configured {}",
                self.model_dim
            );
        }
        let (mask_batch, mask_len) = candidate_mask.dims2()?;
        if mask_batch != batch || mask_len != token_len {
            candle_core::bail!(
                "interaction adapter candidate-mask mismatch: hidden [{batch},{token_len},{model_dim}] mask [{mask_batch},{mask_len}]"
            );
        }

        let normalized = self.query_norm.forward(candidate_hidden)?;
        let attended =
            self.cross_attention
                .forward(&normalized, spectrum_memory, spectrum_memory_mask)?;
        let mut hidden = (candidate_hidden + attended)?;

        let normalized = self.feed_forward_norm.forward(&hidden)?;
        let ff = self.feed_forward_in.forward(&normalized)?.relu()?;
        let ff = self.feed_forward_out.forward(&ff)?;
        hidden = (hidden + ff)?;
        hidden = self.output_norm.forward(&hidden)?;

        let token_mask = candidate_mask
            .unsqueeze(2)?
            .broadcast_as((batch, token_len, model_dim))?;
        hidden = hidden.broadcast_mul(&token_mask)?;
        let pooled = hidden.sum(1)?.broadcast_div(
            &candidate_mask
                .sum(1)?
                .clamp(1.0, f64::INFINITY)?
                .unsqueeze(1)?,
        )?;
        self.residual_head.forward(&pooled)?.squeeze(1)
    }

    /// Model width expected by this adapter.
    pub fn model_dim(&self) -> usize {
        self.model_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn zero_initialized_residual_head_preserves_base_score() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let adapter = FoundationSpectrumCandidateInteractionAdapter::new(8, 2, 4, vb)?;
        let candidate = Tensor::ones((3, 5, 8), DType::F32, &device)?;
        let candidate_mask = Tensor::ones((3, 5), DType::F32, &device)?;
        let memory = Tensor::ones((3, 7, 8), DType::F32, &device)?;
        let memory_mask = Tensor::ones((3, 7), DType::F32, &device)?;
        let residual = adapter.forward(&candidate, &candidate_mask, &memory, &memory_mask)?;
        let values = residual.to_vec1::<f32>()?;
        assert!(values.iter().all(|value| value.abs() <= 1e-7));
        Ok(())
    }
}
