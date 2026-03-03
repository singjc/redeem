// redeem-topaz/src/building_blocks/coelution.rs

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

pub(crate) fn zscore_time(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mu = x.mean_keepdim(2)?;
    let var = x.var_keepdim(2)?;
    let sd = var.sqrt()?;
    let sd = (&sd + eps)?;
    x.broadcast_sub(&mu)?.broadcast_div(&sd)
}

pub(crate) fn linspace_0_1(l: usize, device: &Device) -> Result<Tensor> {
    if l <= 1 {
        return Tensor::zeros((1, 1, l), DType::F32, device);
    }
    let idx = Tensor::arange(0f32, l as f32, device)?;
    let idx = (&idx / ((l - 1) as f64))?;
    idx.reshape((1, 1, l))
}

fn upper_triangle_indices(c: usize) -> Vec<i64> {
    let mut idx = Vec::new();
    for i in 0..c {
        for j in (i + 1)..c {
            idx.push((i * c + j) as i64);
        }
    }
    idx
}

/// Compute differentiable co-elution features + learned embedding from traces.
pub struct CoelutionHead {
    cmax: usize,
    p: usize,
    beta: f64,
    max_lag: usize,
    sim_emb_dim: usize,
    peakiness_thr: f64,
    peakiness_scale: f64,
    eps: f64,
    sim_mlp: Option<nn::Sequential>,
    upper_idx: Tensor,
}

impl CoelutionHead {
    pub fn new(
        vb: VarBuilder,
        cmax: usize,
        beta: f64,
        max_lag: usize,
        sim_emb_dim: usize,
    ) -> Result<Self> {
        let p = cmax * (cmax.saturating_sub(1)) / 2;
        let upper_idx = if p > 0 {
            Tensor::new(upper_triangle_indices(cmax), vb.device())?
        } else {
            Tensor::zeros((0usize,), DType::I64, vb.device())?
        };

        let sim_mlp = if p > 0 && sim_emb_dim > 0 {
            let mut seq = nn::seq();
            seq = seq
                .add(nn::linear(2 * p, 32, vb.pp("sim_mlp.0"))?)
                .add(nn::Activation::Relu)
                .add(nn::linear(32, sim_emb_dim, vb.pp("sim_mlp.2"))?);
            Some(seq)
        } else {
            None
        };

        Ok(Self {
            cmax,
            p,
            beta,
            max_lag,
            sim_emb_dim,
            peakiness_thr: 2.0,
            peakiness_scale: 2.0,
            eps: 1e-6,
            sim_mlp,
            upper_idx,
        })
    }

    pub fn out_dim(&self) -> usize {
        10 + self.sim_emb_dim
    }

    fn upper_triangle(&self, g: &Tensor) -> Result<Tensor> {
        let (n, c1, c2) = g.dims3()?;
        if c1 != c2 || c1 != self.cmax {
            candle_core::bail!(
                "CoelutionHead upper_triangle: expected C={}, got ({}, {})",
                self.cmax,
                c1,
                c2
            );
        }
        if self.p == 0 {
            return Tensor::zeros((n, 0), DType::F32, g.device());
        }
        let g_flat = g.reshape((n, c1 * c2))?;
        g_flat.index_select(&self.upper_idx, 1)
    }

    fn lagged_pairwise_cos_max(&self, z: &Tensor) -> Result<Tensor> {
        let (_n, _c, l) = z.dims3()?;
        let max_lag = self.max_lag as isize;
        let mut sims_per_lag: Vec<Tensor> = Vec::new();

        for lag in -max_lag..=max_lag {
            let (a, b) = if lag == 0 {
                (z.clone(), z.clone())
            } else if lag > 0 {
                let lag = lag as usize;
                let a = z.narrow(2, lag, l - lag)?;
                let b = z.narrow(2, 0, l - lag)?;
                (a, b)
            } else {
                let k = (-lag) as usize;
                let a = z.narrow(2, 0, l - k)?;
                let b = z.narrow(2, k, l - k)?;
                (a, b)
            };

            let na = a.broadcast_mul(&a)?.sum_keepdim(2)?;
            let na = (na + self.eps)?.sqrt()?;
            let nb = b.broadcast_mul(&b)?.sum_keepdim(2)?;
            let nb = (nb + self.eps)?.sqrt()?;

            let va = a.broadcast_div(&na)?;
            let vb = b.broadcast_div(&nb)?;
            let g = va.matmul(&vb.transpose(1, 2)?)?;

            let sims = self.upper_triangle(&g)?;
            sims_per_lag.push(sims);
        }

        let stacked = Tensor::stack(&sims_per_lag, 2)?;
        stacked.max(2)
    }

    /// x: (N,C,L) raw traces.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, _c, l) = x.dims3()?;
        let device = x.device();

        // per-transition max normalize
        let m = x.max_keepdim(2)?;
        let m = (&m + self.eps)?;
        let x_n = x.broadcast_div(&m)?;

        let z = zscore_time(&x_n, self.eps)?;
        let nrm = z.broadcast_mul(&z)?.sum_keepdim(2)?;
        let nrm = (nrm + self.eps)?.sqrt()?;
        let v = z.broadcast_div(&nrm)?;

        let g0 = v.matmul(&v.transpose(1, 2)?)?;
        let sims0 = self.upper_triangle(&g0)?;
        let sims_lagmax = self.lagged_pairwise_cos_max(&z)?;

        let (mean_cos, min_cos, mean_lag, min_lag) = if self.p > 0 {
            (
                sims0.mean(1)?,
                sims0.min(1)?,
                sims_lagmax.mean(1)?,
                sims_lagmax.min(1)?,
            )
        } else {
            (
                Tensor::zeros((n,), DType::F32, device)?,
                Tensor::zeros((n,), DType::F32, device)?,
                Tensor::zeros((n,), DType::F32, device)?,
                Tensor::zeros((n,), DType::F32, device)?,
            )
        };

        // soft apex positions (0..1)
        let idx = linspace_0_1(l, device)?;
        let w = candle_nn::ops::softmax(&(&x_n * self.beta)?, 2)?;
        let apex = w.broadcast_mul(&idx)?.sum(2)?;

        let apex_mean = apex.mean_keepdim(1)?;
        let apex_std = apex.var(1)?.sqrt()?;
        let apex_max = apex.max(1)?;
        let apex_min = apex.min(1)?;
        let apex_range = (&apex_max - &apex_min)?;
        let apex_mad = apex.broadcast_sub(&apex_mean)?.abs()?.mean(1)?;

        let eps_t = Tensor::full(self.eps as f32, w.dims(), device)?;
        let logw = (&w + &eps_t)?.log()?;
        let ent = (&w * &logw)?;
        let ent = (ent * -1.0)?;
        let ent = ent.sum(2)?;
        let ent_mean = ent.mean(1)?;
        let ent_min = ent.min(1)?;

        let peak_max = x_n.max(2)?;
        let peak_mean = x_n.mean(2)?;
        let peak_ratio = peak_max.broadcast_div(&(&peak_mean + self.eps)?)?;
        let peak_score = (&peak_ratio - self.peakiness_thr)?;
        let peak_score = (peak_score * self.peakiness_scale)?;
        let peak_frac = candle_nn::ops::sigmoid(&peak_score)?.mean(1)?;

        let sim_in = if self.p > 0 {
            Tensor::cat(&[sims0, sims_lagmax], 1)?
        } else {
            Tensor::zeros((n, 0), DType::F32, device)?
        };

        let sim_emb = if let Some(mlp) = &self.sim_mlp {
            sim_in.apply(mlp)?
        } else {
            Tensor::zeros((n, self.sim_emb_dim), DType::F32, device)?
        };

        let mut parts: Vec<Tensor> = Vec::with_capacity(11);
        parts.push(mean_cos.unsqueeze(1)?);
        parts.push(min_cos.unsqueeze(1)?);
        parts.push(mean_lag.unsqueeze(1)?);
        parts.push(min_lag.unsqueeze(1)?);
        parts.push(apex_std.unsqueeze(1)?);
        parts.push(apex_range.unsqueeze(1)?);
        parts.push(apex_mad.unsqueeze(1)?);
        parts.push(ent_mean.unsqueeze(1)?);
        parts.push(ent_min.unsqueeze(1)?);
        parts.push(peak_frac.unsqueeze(1)?);
        parts.push(sim_emb);

        Tensor::cat(&parts, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_coelution_identical_traces_high_cos() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let head = CoelutionHead::new(vb.pp("coe"), 3, 10.0, 2, 4)?;

        let base = vec![0.0f32, 1.0, 2.0, 1.0, 0.0];
        let mut data = Vec::new();
        for _ in 0..3 {
            data.extend_from_slice(&base);
        }
        let x = Tensor::new(data, &device)?.reshape((1, 3, 5))?;
        let out = head.forward(&x)?;
        let (n, d) = out.dims2()?;
        assert_eq!(n, 1);
        assert_eq!(d, head.out_dim());

        let v = out.to_vec2::<f32>()?;
        let mean_cos = v[0][0];
        let min_cos = v[0][1];
        let mean_lag = v[0][2];
        let min_lag = v[0][3];

        assert!(mean_cos > 0.95);
        assert!(min_cos > 0.90);
        assert!(mean_lag > 0.90);
        assert!(min_lag > 0.85);
        Ok(())
    }
}
