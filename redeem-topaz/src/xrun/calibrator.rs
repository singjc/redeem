//! Attention-based cross-run calibrator used to predict additive per-run score
//! deltas.
//!
//! Shape notation used here:
//!
//! - `P`: number of precursor sequences in a batch.
//! - `R`: number of run positions per sequence.
//! - `D_in`: XRUN input width.
//! - `D_model`: internal projected width.

use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
struct DropoutAlways(nn::Dropout);

impl Module for DropoutAlways {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs, true)
    }
}

/// Network configuration for the XRUN attention calibrator.
///
/// The calibrator reads per-run items of the form `(bag_score, winner_hidden)`
/// and predicts an additive score correction for each run position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrunConfig {
    pub in_dim: usize,
    pub d_model: usize,
    pub attn_hidden: usize,
    pub head_hidden: Vec<usize>,
    pub dropout: f64,
    pub center_delta: bool,
    pub delta_clip: Option<f64>,
}

/// Lightweight attention model that reads `(bag_score, winner_hidden)` sequences
/// and predicts one delta per run.
///
/// For each precursor sequence, the model:
/// - projects each run item into an internal representation,
/// - computes attention weights over runs,
/// - forms a context vector,
/// - predicts one additive delta per run conditioned on both the local run
///   state and the precursor-wide context.
pub struct XrunAttentionCalibrator {
    proj: nn::Sequential,
    attn: nn::Sequential,
    delta: nn::Sequential,
    cfg: XrunConfig,
}

impl XrunAttentionCalibrator {
    /// Construct a calibrator from [`XrunConfig`].
    pub fn new(vb: VarBuilder, cfg: XrunConfig) -> Result<Self> {
        let mut proj = nn::seq();
        proj = proj
            .add(nn::linear(cfg.in_dim, cfg.d_model, vb.pp("proj0"))?)
            .add(nn::Activation::Relu);
        if cfg.dropout > 0.0 {
            proj = proj.add(DropoutAlways(nn::Dropout::new(cfg.dropout as f32)));
        }

        let attn = nn::seq()
            .add(nn::linear(cfg.d_model, cfg.attn_hidden, vb.pp("attn0"))?)
            .add(nn::func(|xs| xs.tanh()))
            .add(nn::linear(cfg.attn_hidden, 1, vb.pp("attn1"))?);

        let mut delta = nn::seq();
        let mut d = 2 * cfg.d_model;
        for (i, &h) in cfg.head_hidden.iter().enumerate() {
            delta = delta
                .add(nn::linear(d, h, vb.pp(format!("delta{i}")))?)
                .add(nn::Activation::Relu);
            if cfg.dropout > 0.0 {
                delta = delta.add(DropoutAlways(nn::Dropout::new(cfg.dropout as f32)));
            }
            d = h;
        }
        delta = delta.add(nn::linear(d, 1, vb.pp("delta_out"))?);

        Ok(Self {
            proj,
            attn,
            delta,
            cfg,
        })
    }

    /// Predict per-run deltas and attention weights for a masked sequence batch.
    ///
    /// # Inputs
    /// - `x`: `(P, R, D_in)` where each run item typically contains one base
    ///   bag score followed by the winner-hidden embedding.
    /// - `mask`: `(P, R)` validity mask; `0` marks padded run positions.
    ///
    /// # Output
    /// Returns `(delta, attn)` where both tensors have shape `(P, R)`:
    /// - `delta`: additive correction to apply to each run's bag score
    /// - `attn`: precursor-specific attention distribution over runs
    pub fn forward_masked(&self, x: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        // XRUN batches are typically built with `narrow(0, ...)` over a large
        // `(P, R, D_in)` tensor. Candle's linear/matmul path requires
        // contiguous input layouts, so normalize the batch layout at the
        // boundary here instead of relying on every call site to do it.
        let x = x.contiguous()?;
        let mask = mask.contiguous()?;
        let (b, r, _d) = x.dims3()?;

        let r_proj = x.apply(&self.proj)?.contiguous()?; // (B,R,dm)

        // attn logits
        let a_logits = r_proj.apply(&self.attn)?.squeeze(2)?; // (B,R)
        let m = mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, r), x.device())?;
        let ones = m.ones_like()?;
        let a_logits = ((&a_logits * &m)? + (&neg_big * (&ones - &m)?)?)?;

        // softmax over runs
        let a = candle_nn::ops::softmax(&a_logits, 1)?; // (B,R)
        let a = (&a * &m)?; // zero invalid

        // context = sum(r_proj * a)
        let a3 = a.unsqueeze(2)?; // (B,R,1)
        let context = r_proj.broadcast_mul(&a3)?.sum_keepdim(1)?; // (B,1,dm)
        let ctx = context.broadcast_as(r_proj.dims())?.contiguous()?;

        // delta head on concat([r_proj, ctx])
        let cat = Tensor::cat(&[r_proj, ctx], 2)?.contiguous()?; // (B,R,2dm)
        let mut dlt = cat.apply(&self.delta)?.squeeze(2)?; // (B,R)
        dlt = (&dlt * &m)?;

        // center per precursor
        if self.cfg.center_delta {
            let denom = m.sum_keepdim(1)?.maximum(1.0f32)?;
            let mu = dlt
                .broadcast_mul(&m)?
                .sum_keepdim(1)?
                .broadcast_div(&denom)?; // (B,1)
            let mu = mu.broadcast_as((b, r))?;
            dlt = (&dlt - &mu)?;
            dlt = (&dlt * &m)?;
        }

        // smooth clip
        if let Some(dc) = self.cfg.delta_clip {
            if dc > 0.0 {
                let dc_t = Tensor::full(dc as f32, dlt.dims(), dlt.device())?;
                let clipped = (&dlt / &dc_t)?.tanh()?;
                dlt = dc_t.broadcast_mul(&clipped)?;
            }
        }

        Ok((dlt, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_forward_masked_accepts_narrowed_batch() -> Result<()> {
        let device = Device::Cpu;
        let cfg = XrunConfig {
            in_dim: 128,
            d_model: 64,
            attn_hidden: 32,
            head_hidden: vec![32],
            dropout: 0.0,
            center_delta: true,
            delta_clip: Some(5.0),
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = XrunAttentionCalibrator::new(vb.pp("xrun"), cfg)?;

        let p = 256usize;
        let r = 64usize;
        let din = 128usize;
        let xseq: Vec<f32> = (0..p * r * din).map(|i| ((i % 37) as f32) * 0.01).collect();
        let mask_u8 = vec![1u8; p * r];
        let x = Tensor::from_vec(xseq, (p, r, din), &device)?;
        let m = Tensor::from_vec(mask_u8, (p, r), &device)?;

        let xb = x.narrow(0, 64, 128)?;
        let mb = m.narrow(0, 64, 128)?;
        let (delta, attn) = model.forward_masked(&xb, &mb)?;
        assert_eq!(delta.dims2()?, (128, 64));
        assert_eq!(attn.dims2()?, (128, 64));
        Ok(())
    }
}
