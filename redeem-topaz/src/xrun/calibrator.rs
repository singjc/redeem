// redeem-topaz/src/xrun/calibrator.rs

use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

#[derive(Clone, Debug)]
struct DropoutAlways(nn::Dropout);

impl Module for DropoutAlways {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs, true)
    }
}

#[derive(Debug, Clone)]
pub struct XrunConfig {
    pub in_dim: usize,
    pub d_model: usize,
    pub attn_hidden: usize,
    pub head_hidden: Vec<usize>,
    pub dropout: f64,
    pub center_delta: bool,
    pub delta_clip: Option<f64>,
}

pub struct XrunAttentionCalibrator {
    proj: nn::Sequential,
    attn: nn::Sequential,
    delta: nn::Sequential,
    cfg: XrunConfig,
}

impl XrunAttentionCalibrator {
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

        Ok(Self { proj, attn, delta, cfg })
    }

    /// x: (B,R,D)  mask: (B,R) bool
    /// returns (delta: (B,R), attn: (B,R))
    pub fn forward_masked(&self, x: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, r, d) = x.dims3()?;

        let r_proj = x.apply(&self.proj)?; // (B,R,dm)

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
        let ctx = context.broadcast_as(r_proj.dims())?;

        // delta head on concat([r_proj, ctx])
        let cat = Tensor::cat(&[r_proj, ctx], 2)?; // (B,R,2dm)
        let mut dlt = cat.apply(&self.delta)?.squeeze(2)?; // (B,R)
        dlt = (&dlt * &m)?;

        // center per precursor
        if self.cfg.center_delta {
            let denom = m.sum_keepdim(1)?.maximum(1.0f32)?;
            let mu = dlt.broadcast_mul(&m)?.sum_keepdim(1)?.broadcast_div(&denom)?; // (B,1)
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
