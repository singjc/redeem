//! Lightweight bag-level interaction encoder over candidate representations.
//!
//! This block operates over the padded candidate axis `K` inside one precursor
//! bag. It lets the model compare candidates directly before the final scorer
//! consumes them, which is useful for rank-1 peak-group selection.

use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Learned interaction embedding over the within-bag candidate axis.
pub struct BagInteractionBlock {
    in_proj: nn::Linear,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    hidden_dim: usize,
    out_dim: usize,
}

impl BagInteractionBlock {
    /// Build a bag interaction block for per-candidate input width `in_dim`.
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let hidden_dim = hidden_dim.max(4);
        let in_proj = nn::linear(in_dim, hidden_dim, vb.pp("in_proj"))?;
        let q_proj = nn::linear(hidden_dim, hidden_dim, vb.pp("q_proj"))?;
        let k_proj = nn::linear(hidden_dim, hidden_dim, vb.pp("k_proj"))?;
        let v_proj = nn::linear(hidden_dim, hidden_dim, vb.pp("v_proj"))?;
        let out_proj = nn::linear(2 * hidden_dim, out_dim, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            hidden_dim,
            out_dim,
        })
    }

    /// Output dimensionality of the per-candidate context vector.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Encode within-bag candidate interactions.
    ///
    /// Input:
    /// - `x`: `(B, K, D)` candidate descriptors
    /// - `mask`: `(B, K)` validity mask
    ///
    /// Output:
    /// - `(B, K, out_dim)` per-candidate contextual embeddings
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, k, d) = x.dims3()?;
        if self.out_dim == 0 || k < 2 {
            return Tensor::zeros((b, k, self.out_dim), DType::F32, x.device());
        }

        let desc = x
            .reshape((b * k, d))?
            .apply(&self.in_proj)?
            .apply(&nn::Activation::Relu)?
            .reshape((b, k, self.hidden_dim))?;
        let desc_flat = desc.reshape((b * k, self.hidden_dim))?;
        let q = desc_flat
            .apply(&self.q_proj)?
            .reshape((b, k, self.hidden_dim))?;
        let kk = desc_flat
            .apply(&self.k_proj)?
            .reshape((b, k, self.hidden_dim))?;
        let v = desc_flat
            .apply(&self.v_proj)?
            .reshape((b, k, self.hidden_dim))?;

        let logits = q.matmul(&kk.transpose(1, 2)?)?;
        let scale = (self.hidden_dim as f64).sqrt();
        let logits = (&logits / scale)?;

        let key_mask = mask
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .broadcast_as((b, k, k))?;
        let neg_big = Tensor::full(-1e9f32, (b, k, k), x.device())?;
        let ones = key_mask.ones_like()?;
        let invalid = ones.broadcast_sub(&key_mask)?;
        let logits = (logits.broadcast_mul(&key_mask)? + neg_big.broadcast_mul(&invalid)?)?;

        let attn = candle_nn::ops::softmax(&logits, 2)?;
        let ctx = attn.matmul(&v)?;
        let fused = Tensor::cat(&[desc, ctx], 2)?
            .reshape((b * k, 2 * self.hidden_dim))?
            .apply(&self.out_proj)?
            .reshape((b, k, self.out_dim))?;

        let q_mask = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
        fused.broadcast_mul(&q_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_bag_interaction_shape() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = BagInteractionBlock::new(vb.pp("bag"), 12, 16, 10)?;
        let x = Tensor::zeros((3usize, 5usize, 12usize), DType::F32, &device)?;
        let mask = Tensor::new(
            vec![
                1u8, 1, 1, 0, 0, //
                1, 1, 1, 1, 0, //
                1, 0, 0, 0, 0,
            ],
            &device,
        )?
        .reshape((3usize, 5usize))?;
        let y = block.forward(&x, &mask)?;
        assert_eq!(y.dims3()?, (3, 5, 10));
        Ok(())
    }
}
