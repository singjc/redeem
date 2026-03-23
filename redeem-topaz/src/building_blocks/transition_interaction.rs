//! Lightweight transition-interaction encoder for small DIA trace channel sets.
//!
//! This block builds one descriptor per transition channel, applies a compact
//! self-attention over the channel axis, and pools the result into a fixed-size
//! embedding that can be concatenated with the main convolutional branch.

use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::building_blocks::trace_input::TraceInputMode;

/// Learned interaction embedding over the transition/channel axis.
pub struct TransitionInteractionBlock {
    time_proj: nn::Linear,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    trace_input_mode: TraceInputMode,
    hidden_dim: usize,
    out_dim: usize,
}

impl TransitionInteractionBlock {
    /// Build an interaction block for traces of fixed length `l`.
    pub fn new(
        vb: VarBuilder,
        l: usize,
        trace_input_mode: TraceInputMode,
        hidden_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let time_dim = if trace_input_mode == TraceInputMode::Dual {
            2 * l
        } else {
            l
        };
        let hidden_dim = hidden_dim.max(4);
        let time_proj = nn::linear(time_dim, hidden_dim, vb.pp("time_proj"))?;
        let q_proj = nn::linear(hidden_dim, hidden_dim, vb.pp("q_proj"))?;
        let k_proj = nn::linear(hidden_dim, hidden_dim, vb.pp("k_proj"))?;
        let v_proj = nn::linear(hidden_dim, hidden_dim, vb.pp("v_proj"))?;
        let out_proj = nn::linear(2 * hidden_dim, out_dim, vb.pp("out_proj"))?;
        Ok(Self {
            time_proj,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            trace_input_mode,
            hidden_dim,
            out_dim,
        })
    }

    /// Output dimensionality of the pooled interaction embedding.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    fn prepare_channel_input(&self, x: &Tensor) -> Result<Tensor> {
        match self.trace_input_mode {
            TraceInputMode::Single => Ok(x.clone()),
            TraceInputMode::Dual => {
                let eps = 1e-6f32;
                let m = x.max_keepdim(2)?;
                let eps_t = Tensor::full(eps, m.dims(), m.device())?.to_dtype(m.dtype())?;
                let x0 = x.broadcast_div(&m.broadcast_add(&eps_t)?)?;

                let x_cl = x.maximum(0f32)?;
                let ones = Tensor::ones(x_cl.dims(), x_cl.dtype(), x_cl.device())?;
                let x1 = x_cl.broadcast_add(&ones)?.log()?;

                // Keep the original channel count and expose both views by
                // concatenating them along the temporal axis.
                Tensor::cat(&[x0, x1], 2)
            }
        }
    }

    /// Encode cross-transition interactions for one modality-specific trace tensor.
    ///
    /// Input: `(N, C, L)` and output: `(N, out_dim)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.prepare_channel_input(x)?;
        let (n, c, t) = x.dims3()?;
        if c < 2 || self.out_dim == 0 {
            return Tensor::zeros((n, self.out_dim), DType::F32, x.device());
        }

        let desc = x
            .reshape((n * c, t))?
            .apply(&self.time_proj)?
            .apply(&nn::Activation::Relu)?
            .reshape((n, c, self.hidden_dim))?;
        let desc_flat = desc.reshape((n * c, self.hidden_dim))?;
        let q = desc_flat
            .apply(&self.q_proj)?
            .reshape((n, c, self.hidden_dim))?;
        let k = desc_flat
            .apply(&self.k_proj)?
            .reshape((n, c, self.hidden_dim))?;
        let v = desc_flat
            .apply(&self.v_proj)?
            .reshape((n, c, self.hidden_dim))?;

        let logits = q.matmul(&k.transpose(1, 2)?)?;
        let attn = candle_nn::ops::softmax(&(&logits / (self.hidden_dim as f64).sqrt())?, 2)?;
        let ctx = attn.matmul(&v)?;
        let fused = (&ctx + &desc)?;

        let pooled_mean = fused.mean(1)?;
        let pooled_max = fused.max(1)?;
        Tensor::cat(&[pooled_mean, pooled_max], 1)?.apply(&self.out_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_transition_interaction_shape() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = TransitionInteractionBlock::new(vb.pp("ti"), 8, TraceInputMode::Dual, 16, 12)?;
        let x = Tensor::zeros((3usize, 4usize, 8usize), DType::F32, &device)?;
        let y = block.forward(&x)?;
        assert_eq!(y.dims2()?, (3, 12));
        Ok(())
    }
}
