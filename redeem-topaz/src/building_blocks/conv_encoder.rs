// redeem-topaz/src/building_blocks/conv_encoder.rs

use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::building_blocks::trace_input::{make_trace_input, TraceInputMode};
use crate::building_blocks::coelution::{linspace_0_1, zscore_time};
use crate::model::topaz::TopazConfig;

pub struct ConvBranch {
    conv0: nn::Conv1d,
    conv1: nn::Conv1d,
    conv2: nn::Conv1d,
    proj: nn::Linear,
    trace_input_mode: TraceInputMode,
    emb_dim: usize,
}

impl ConvBranch {
    pub fn new(vb: VarBuilder, cin: usize, emb_dim: usize, trace_input_mode: TraceInputMode) -> Result<Self> {
        let conv_cfg_k1 = nn::Conv1dConfig { padding: 0, ..Default::default() };
        let conv_cfg_k3 = nn::Conv1dConfig { padding: 1, ..Default::default() };

        let conv0 = nn::conv1d(cin, 32, 1, conv_cfg_k1, vb.pp("conv0"))?;
        let conv1 = nn::conv1d(32, 64, 3, conv_cfg_k3, vb.pp("conv1"))?;
        let conv2 = nn::conv1d(64, 64, 3, conv_cfg_k3, vb.pp("conv2"))?;
        let proj = nn::linear(128, emb_dim, vb.pp("proj"))?;

        Ok(Self { conv0, conv1, conv2, proj, trace_input_mode, emb_dim })
    }

    pub fn emb_dim(&self) -> usize {
        self.emb_dim
    }

    /// x: (N,C,L)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_in = make_trace_input(x, self.trace_input_mode, 1e-6)?;
        let h = x_in.apply(&self.conv0)?.relu()?;
        let h = h.apply(&self.conv1)?.relu()?;
        let h = h.apply(&self.conv2)?.relu()?;

        // pool along time
        let h_max = h.max(2)?;  // (N,64)
        let h_mean = h.mean(2)?; // (N,64)
        let h_pool = Tensor::cat(&[h_max, h_mean], 1)?; // (N,128)
        h_pool.apply(&self.proj)
    }
}

pub struct TraceEncoder {
    ms2_c: usize,
    ms1_c: usize,

    ms2: ConvBranch,
    ms1: Option<ConvBranch>,

    use_coelution: bool,
    ms2_coe: Option<crate::building_blocks::coelution::CoelutionHead>,
    ms1_coe: Option<crate::building_blocks::coelution::CoelutionHead>,

    beta: f64,
}

impl TraceEncoder {
    pub fn new(vb: VarBuilder, cfg: &TopazConfig) -> Result<Self> {
        let dual_mul = if cfg.trace_input_mode == TraceInputMode::Dual { 2 } else { 1 };

        let ms2 = ConvBranch::new(
            vb.pp("ms2"),
            cfg.ms2_cmax * dual_mul,
            cfg.trace_emb_dim,
            cfg.trace_input_mode,
        )?;

        let ms1 = if cfg.ms1_cmax > 0 {
            Some(ConvBranch::new(
                vb.pp("ms1"),
                cfg.ms1_cmax * dual_mul,
                cfg.trace_emb_dim,
                cfg.trace_input_mode,
            )?)
        } else {
            None
        };

        let use_coelution = cfg.use_coelution_head;
        let ms2_coe = if use_coelution {
            Some(crate::building_blocks::coelution::CoelutionHead::new(
                vb.pp("ms2_coe"),
                cfg.ms2_cmax,
                cfg.coelution_beta,
                cfg.coelution_max_lag,
                cfg.coelution_sim_emb_dim,
            )?)
        } else { None };

        let ms1_coe = if use_coelution && cfg.ms1_cmax > 0 {
            Some(crate::building_blocks::coelution::CoelutionHead::new(
                vb.pp("ms1_coe"),
                cfg.ms1_cmax,
                cfg.coelution_beta,
                cfg.coelution_max_lag,
                cfg.coelution_sim_emb_dim,
            )?)
        } else { None };

        Ok(Self {
            ms2_c: cfg.ms2_cmax,
            ms1_c: cfg.ms1_cmax,
            ms2,
            ms1,
            use_coelution,
            ms2_coe,
            ms1_coe,
            beta: cfg.coelution_beta,
        })
    }

    pub fn emb_out_dim(&self) -> usize {
        let d = self.ms2.emb_dim();
        if self.ms1_c > 0 { 2 * d } else { d }
    }

    pub fn coelution_dim(&self) -> usize {
        let mut d = 0usize;
        if self.use_coelution {
            if let Some(c) = &self.ms2_coe { d += c.out_dim(); }
            if let Some(c) = &self.ms1_coe { d += c.out_dim(); d += 4; }
        }
        d
    }

    /// Forward on flattened candidates.
    /// x: (N, C_total, L). If ms1 enabled, layout is [MS1 channels | MS2 channels].
    /// Returns (emb: (N,E), coe: (N,Coe)).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (emb, coe, _coe_ms12) = self.forward_components(x)?;
        Ok((emb, coe))
    }

    /// Forward returning coelution components (including optional ms12 features).
    /// Returns (emb, coe_all, coe_ms12).
    pub fn forward_components(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (n, c_total, l) = x.dims3()?;
        let expected = if self.ms1_c > 0 { self.ms1_c + self.ms2_c } else { self.ms2_c };

        let x = if c_total != expected {
            // defensive pad/crop (Python prints warning once)
            if c_total < expected {
                let pad = Tensor::zeros((n, expected - c_total, l), x.dtype(), x.device())?;
                Tensor::cat(&[x.clone(), pad], 1)?
            } else {
                x.narrow(1, 0, expected)?
            }
        } else {
            x.clone()
        };

        let (ms1, ms2) = if self.ms1_c > 0 {
            let a = x.narrow(1, 0, self.ms1_c)?;
            let b = x.narrow(1, self.ms1_c, self.ms2_c)?;
            (Some(a), b)
        } else {
            (None, x)
        };
        let device = ms2.device();

        // coelution
        let mut coe_parts: Vec<Tensor> = Vec::new();
        let mut coe_ms12 = Tensor::zeros((n, 0), DType::F32, device)?;
        if self.use_coelution {
            if let Some(head) = &self.ms2_coe {
                coe_parts.push(head.forward(&ms2)?);
            }
            if let (Some(ms1x), Some(head1)) = (ms1.as_ref(), &self.ms1_coe) {
                coe_parts.push(head1.forward(ms1x)?);
                coe_ms12 = self.ms12_features(ms1x, &ms2)?; // (N,4)
                coe_parts.push(coe_ms12.clone());
            }
        }
        let coe = if coe_parts.is_empty() {
            Tensor::zeros((n, 0), DType::F32, device)?
        } else {
            Tensor::cat(&coe_parts, 1)?
        };

        // embeddings
        let emb2 = self.ms2.forward(&ms2)?;
        let emb = if let (Some(ms1x), Some(b1)) = (ms1, self.ms1.as_ref()) {
            let emb1 = b1.forward(&ms1x)?;
            Tensor::cat(&[emb2, emb1], 1)?
        } else {
            emb2
        };

        Ok((emb, coe, coe_ms12))
    }

    /// Port of Python `_ms12_features`.
    fn ms12_features(&self, ms1: &Tensor, ms2: &Tensor) -> Result<Tensor> {
        let eps = 1e-6;

        let ms1m = ms1.max_keepdim(2)?;
        let ms2m = ms2.max_keepdim(2)?;
        let ms1n = ms1.broadcast_div(&(&ms1m + eps)?)?;
        let ms2n = ms2.broadcast_div(&(&ms2m + eps)?)?;

        let z1 = zscore_time(&ms1n, eps)?;
        let z2 = zscore_time(&ms2n, eps)?;

        let n1 = z1.broadcast_mul(&z1)?.sum_keepdim(2)?;
        let n1 = (n1 + eps)?.sqrt()?;
        let n2 = z2.broadcast_mul(&z2)?.sum_keepdim(2)?;
        let n2 = (n2 + eps)?.sqrt()?;
        let v1 = z1.broadcast_div(&n1)?;
        let v2 = z2.broadcast_div(&n2)?;

        let g12 = v1.matmul(&v2.transpose(1, 2)?)?;
        let mean_cos = g12.mean((1, 2))?;
        let max_cos = g12.max(2)?.max(1)?;

        let l = ms1n.dim(2)?;
        let idx = linspace_0_1(l, ms1.device())?;
        let w1 = candle_nn::ops::softmax(&(&ms1n * self.beta)?, 2)?;
        let w2 = candle_nn::ops::softmax(&(&ms2n * self.beta)?, 2)?;
        let apex1 = w1.broadcast_mul(&idx)?.sum(2)?;
        let apex2 = w2.broadcast_mul(&idx)?.sum(2)?;

        let mean1 = apex1.mean(1)?;
        let mean2 = apex2.mean(1)?;
        let d_mean = (&mean1 - &mean2)?.abs()?;

        let apex_all = Tensor::cat(&[apex1, apex2], 1)?;
        let apex_std = apex_all.var(1)?.sqrt()?;

        Tensor::stack(&[mean_cos, max_cos, d_mean, apex_std], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_ms12_features_identical_traces() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TopazConfig {
            feat_dim: 0,
            ms2_cmax: 2,
            ms1_cmax: 2,
            l: 5,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: false,
            use_coelution_head: true,
            ..Default::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let enc = TraceEncoder::new(vb.pp("enc"), &cfg)?;

        let base = vec![0.0f32, 1.0, 2.0, 1.0, 0.0];
        let mut data = Vec::new();
        for _ in 0..2 {
            data.extend_from_slice(&base);
        }
        let ms1 = Tensor::new(data.clone(), &device)?.reshape((1, 2, 5))?;
        let ms2 = Tensor::new(data, &device)?.reshape((1, 2, 5))?;

        let f = enc.ms12_features(&ms1, &ms2)?;
        let v = f.to_vec2::<f32>()?;
        assert_eq!(v[0].len(), 4);
        assert!(v[0][0] > 0.95); // mean_cos
        assert!(v[0][1] > 0.95); // max_cos
        assert!(v[0][2].abs() < 1e-3); // d_mean apex
        assert!(v[0][3].abs() < 1e-3); // apex_std
        Ok(())
    }
}
