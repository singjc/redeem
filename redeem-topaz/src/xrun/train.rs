//! Training loop for the XRUN attention calibrator.
//!
//! Shape notation used in this module:
//!
//! - `P`: number of precursor sequences.
//! - `R`: maximum number of run positions kept per precursor.
//! - `D_in`: XRUN input width, usually `1 + H` for `(bag_score, winner_hidden)`.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap, optim::AdamW};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::xrun::calibrator::{XrunAttentionCalibrator, XrunConfig};

/// Pooling strategies used to collapse per-run calibrated scores into a
/// precursor-level supervision target during XRUN training.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XrunPoolMode {
    Max,
    Lse,
    SoftmaxMean,
    AttnMean,
}

/// Weighting scheme for the variance regularizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XrunVarWeight {
    Uniform,
    Attn,
    Pool,
}

/// Hyper-parameters for XRUN calibrator training.
///
/// These settings control only the XRUN sidecar, not the base TOPAZ model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrunTrainConfig {
    pub d_model: usize,
    pub attn_hidden: usize,
    pub head_hidden: Vec<usize>,
    pub dropout: f64,
    pub lr: f64,
    pub weight_decay: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub delta_l2: f64,
    pub pool: XrunPoolMode,
    pub tau: f64,
    pub lambda_var: f64,
    pub var_weight: XrunVarWeight,
    pub lambda_mean_delta: f64,
    pub delta_clip: Option<f64>,
    pub center_delta: bool,
}

impl Default for XrunTrainConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            attn_hidden: 64,
            head_hidden: vec![64, 32],
            dropout: 0.1,
            lr: 5e-4,
            weight_decay: 1e-4,
            batch_size: 512,
            max_epochs: 20,
            patience: 5,
            delta_l2: 1e-4,
            pool: XrunPoolMode::Max,
            tau: 1.0,
            lambda_var: 0.0,
            var_weight: XrunVarWeight::Uniform,
            lambda_mean_delta: 0.0,
            delta_clip: None,
            center_delta: false,
        }
    }
}

/// Summary returned after training an XRUN calibrator.
#[derive(Debug, Clone)]
pub struct XrunTrainMeta {
    pub best_val: f32,
    pub epochs: usize,
}

/// Tensorizable XRUN dataset with precursor-major layout.
///
/// The flattened buffers represent:
/// - `xseq`: `(P, R, D_in)` row-major
/// - `mask`: `(P, R)` validity mask
/// - `y`: `(P,)` precursor-level binary labels
#[derive(Debug, Clone)]
pub struct XrunDataset {
    pub xseq: Vec<f32>,
    pub mask: Vec<bool>,
    pub y: Vec<f32>,
    pub p: usize,
    pub r: usize,
    pub din: usize,
}

/// Periodically emits batch-level progress for XRUN calibrator training.
///
/// XRUN epochs can still take minutes on large multi-run datasets. This logger
/// provides a coarse ETA and running loss so long jobs show visible progress
/// between the per-epoch summaries.
struct XrunProgressLogger {
    total_epochs: usize,
    batches_per_epoch: usize,
    total_batches: usize,
    started: Instant,
    last_log: Instant,
    log_every: Duration,
}

impl XrunProgressLogger {
    /// Create a progress logger for XRUN training with fixed epoch and batch counts.
    fn new(total_epochs: usize, batches_per_epoch: usize) -> Self {
        let now = Instant::now();
        let total_epochs = total_epochs.max(1);
        let batches_per_epoch = batches_per_epoch.max(1);
        Self {
            total_epochs,
            batches_per_epoch,
            total_batches: total_epochs * batches_per_epoch,
            started: now,
            last_log: now,
            log_every: Duration::from_secs(15),
        }
    }

    /// Emit an `info!` progress line when enough time has elapsed.
    fn maybe_log(&mut self, epoch: usize, batch_in_epoch: usize, mean_loss: f32) {
        let now = Instant::now();
        let processed_batches = ((epoch.saturating_sub(1)) * self.batches_per_epoch
            + batch_in_epoch)
            .min(self.total_batches);
        let should_log = processed_batches >= self.total_batches
            || now.duration_since(self.last_log) >= self.log_every
            || processed_batches <= 1;
        if !should_log {
            return;
        }
        self.last_log = now;

        let elapsed = now.duration_since(self.started);
        let pct = 100.0 * processed_batches as f64 / self.total_batches as f64;
        let batches_per_sec = if elapsed.as_secs_f64() > 0.0 {
            processed_batches as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let eta = if processed_batches > 0 && processed_batches < self.total_batches {
            let remaining = (self.total_batches - processed_batches) as f64;
            Duration::from_secs_f64(remaining / batches_per_sec.max(1e-9))
        } else {
            Duration::ZERO
        };

        log::info!(
            "[xrun] Progress | epoch={}/{} batch={}/{} overall={}/{} ({:.1}%) elapsed={} eta={} rate={:.1} batches/s loss={:.4}",
            epoch,
            self.total_epochs,
            batch_in_epoch.min(self.batches_per_epoch),
            self.batches_per_epoch,
            processed_batches,
            self.total_batches,
            pct,
            format_duration(elapsed),
            format_duration(eta),
            batches_per_sec,
            mean_loss
        );
    }
}

impl XrunDataset {
    /// Materialize the dataset as Candle tensors on `device`.
    ///
    /// Returns `(x, mask, y)` with shapes `(P, R, D_in)`, `(P, R)`, and `(P,)`.
    pub fn to_tensors(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let x = Tensor::from_vec(self.xseq.clone(), (self.p, self.r, self.din), device)?;
        let mask_u8: Vec<u8> = self.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
        let m = Tensor::from_vec(mask_u8, (self.p, self.r), device)?;
        let y = Tensor::from_vec(self.y.clone(), (self.p,), device)?;
        Ok((x, m, y))
    }
}

fn subset_dataset(ds: &XrunDataset, idxs: &[usize]) -> XrunDataset {
    let mut xseq = Vec::with_capacity(idxs.len() * ds.r * ds.din);
    let mut mask = Vec::with_capacity(idxs.len() * ds.r);
    let mut y = Vec::with_capacity(idxs.len());
    for &pi in idxs {
        let x0 = pi * ds.r * ds.din;
        let x1 = x0 + ds.r * ds.din;
        xseq.extend_from_slice(&ds.xseq[x0..x1]);

        let m0 = pi * ds.r;
        let m1 = m0 + ds.r;
        mask.extend_from_slice(&ds.mask[m0..m1]);

        y.push(ds.y[pi]);
    }
    XrunDataset {
        xseq,
        mask,
        y,
        p: idxs.len(),
        r: ds.r,
        din: ds.din,
    }
}

fn shuffle_indices(idxs: &mut [usize], seed: u64) {
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    for i in (1..idxs.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        idxs.swap(i, j);
    }
}

/// Split XRUN sequences by precursor.
///
/// This mirrors the Python behavior: `n_val = ceil(val_frac * P)` and is then
/// clamped to `[1, P-1]` when `P > 1`.
pub fn split_train_val(ds: &XrunDataset, val_frac: f32, seed: u64) -> (XrunDataset, XrunDataset) {
    let p = ds.p;
    if p == 0 {
        return (
            XrunDataset {
                xseq: Vec::new(),
                mask: Vec::new(),
                y: Vec::new(),
                p: 0,
                r: ds.r,
                din: ds.din,
            },
            XrunDataset {
                xseq: Vec::new(),
                mask: Vec::new(),
                y: Vec::new(),
                p: 0,
                r: ds.r,
                din: ds.din,
            },
        );
    }
    let mut idxs: Vec<usize> = (0..p).collect();
    shuffle_indices(&mut idxs, seed);

    let mut n_val = (val_frac.max(0.0).min(1.0) * p as f32).ceil() as usize;
    if p <= 1 {
        n_val = 0;
    } else {
        n_val = n_val.max(1).min(p - 1);
    }
    let (val_idx, train_idx) = idxs.split_at(n_val);
    let val = subset_dataset(ds, val_idx);
    let train = subset_dataset(ds, train_idx);
    (train, val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_split_train_val_sizes() {
        let p = 5usize;
        let r = 2usize;
        let din = 1usize;
        let xseq: Vec<f32> = (0..p * r * din).map(|v| v as f32).collect();
        let mask = vec![true; p * r];
        let y: Vec<f32> = (0..p).map(|v| v as f32).collect();
        let ds = XrunDataset {
            xseq,
            mask,
            y,
            p,
            r,
            din,
        };

        let (tr, va) = split_train_val(&ds, 0.2, 123);
        assert_eq!(tr.p + va.p, p);
        assert_eq!(va.p, 1); // ceil(0.2*5)=1

        let tr_set: HashSet<i32> = tr.y.iter().map(|v| *v as i32).collect();
        let va_set: HashSet<i32> = va.y.iter().map(|v| *v as i32).collect();
        assert!(tr_set.is_disjoint(&va_set));
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let abs = x.abs()?;
    let neg_abs = (abs * -1.0)?;
    let exp = neg_abs.exp()?;
    let ones = exp.ones_like()?;
    let log1p = exp.broadcast_add(&ones)?.log()?;
    let max0 = x.maximum(0f32)?;
    Ok(log1p.broadcast_add(&max0)?)
}

fn bce_with_logits_pos_weight(
    logits: &Tensor,
    targets: &Tensor,
    pos_weight: f32,
) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;
    let targets = targets.to_dtype(DType::F32)?;
    let sp = softplus(&logits)?;
    let yt = targets.broadcast_mul(&logits)?;
    let loss = sp.broadcast_sub(&yt)?;
    let scale = Tensor::full(pos_weight - 1.0f32, targets.dims(), targets.device())?;
    let w = targets.broadcast_mul(&scale)?;
    let ones = w.ones_like()?;
    let w = w.broadcast_add(&ones)?;
    let loss = loss.broadcast_mul(&w)?;
    Ok(loss.mean_all()?)
}

fn pool_scores(
    s_adj: &Tensor,
    mask: &Tensor,
    attn: &Tensor,
    mode: &XrunPoolMode,
    tau: f64,
) -> Result<(Tensor, Option<Tensor>)> {
    let (b, r) = s_adj.dims2()?;
    let m = mask.to_dtype(DType::F32)?;
    let neg_big = Tensor::full(-1e9f32, (b, r), s_adj.device())?;
    let ones = m.ones_like()?;
    let s_masked = (s_adj.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;

    match mode {
        XrunPoolMode::Max => Ok((s_masked.max(1)?, None)),
        XrunPoolMode::AttnMean => {
            let w = attn.broadcast_mul(&m)?;
            let denom = w.sum_keepdim(1)?.maximum(1e-12f32)?;
            let w = w.broadcast_div(&denom)?;
            let pooled = w.broadcast_mul(s_adj)?.sum(1)?;
            Ok((pooled, Some(w)))
        }
        XrunPoolMode::SoftmaxMean => {
            let t = tau.max(1e-8) as f64;
            let w = candle_nn::ops::softmax(&(s_masked / t)?, 1)?;
            let w = w.broadcast_mul(&m)?;
            let denom = w.sum_keepdim(1)?.maximum(1e-12f32)?;
            let w = w.broadcast_div(&denom)?;
            let pooled = w.broadcast_mul(s_adj)?.sum(1)?;
            Ok((pooled, Some(w)))
        }
        XrunPoolMode::Lse => {
            let t = tau.max(1e-8) as f64;
            let s_tau = (s_masked / t)?;
            let s_max = s_tau.max(1)?; // (B,)
            let s_max_b = s_max.unsqueeze(1)?.broadcast_as((b, r))?;
            let exp = s_tau.broadcast_sub(&s_max_b)?.exp()?;
            let sum = exp.sum(1)?;
            let lse = (sum.log()? + s_max)?;
            let pooled = (lse * t)?;

            let w = candle_nn::ops::softmax(&s_tau, 1)?;
            let w = w.broadcast_mul(&m)?;
            let denom = w.sum_keepdim(1)?.maximum(1e-12f32)?;
            let w = w.broadcast_div(&denom)?;
            Ok((pooled, Some(w)))
        }
    }
}

fn uniform_weights(mask: &Tensor) -> Result<Tensor> {
    let m = mask.to_dtype(DType::F32)?;
    let denom = m.sum_keepdim(1)?.maximum(1e-12f32)?;
    Ok(m.broadcast_div(&denom)?)
}

/// Trainer for the XRUN calibrator network.
pub struct XrunTrainer {
    pub cfg: XrunTrainConfig,
    pub varmap: VarMap,
    pub model: XrunAttentionCalibrator,
    pub opt: AdamW,
}

impl XrunTrainer {
    /// Construct a new XRUN trainer.
    pub fn new(cfg: XrunTrainConfig, in_dim: usize, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = XrunAttentionCalibrator::new(
            vb.pp("xrun"),
            XrunConfig {
                in_dim,
                d_model: cfg.d_model,
                attn_hidden: cfg.attn_hidden,
                head_hidden: cfg.head_hidden.clone(),
                dropout: cfg.dropout,
                center_delta: cfg.center_delta,
                delta_clip: cfg.delta_clip,
            },
        )?;
        let params = nn::optim::ParamsAdamW {
            lr: cfg.lr as f64,
            weight_decay: cfg.weight_decay as f64,
            ..Default::default()
        };
        let opt = AdamW::new(varmap.all_vars(), params)?;
        Ok(Self {
            cfg,
            varmap,
            model,
            opt,
        })
    }

    /// Train the calibrator and restore the best validation epoch.
    pub fn train(
        &mut self,
        train: &XrunDataset,
        val: &XrunDataset,
        device: &Device,
    ) -> Result<XrunTrainMeta> {
        let (x_tr, m_tr, y_tr) = train.to_tensors(device)?;
        let (x_va, m_va, y_va) = val.to_tensors(device)?;

        let n_pos = train.y.iter().filter(|&&v| v > 0.5).count() as f32;
        let n_neg = (train.y.len() as f32) - n_pos;
        let pos_weight = if n_pos > 0.0 { n_neg / n_pos } else { 1.0 };

        let mut best_val = f32::INFINITY;
        let mut bad = 0usize;
        let mut best_path: Option<std::path::PathBuf> = None;
        let batches_per_epoch = train.p.div_ceil(self.cfg.batch_size.max(1));
        let mut progress = XrunProgressLogger::new(self.cfg.max_epochs.max(1), batches_per_epoch);

        for epoch in 1..=self.cfg.max_epochs.max(1) {
            let mut tr_loss = 0f32;
            let mut tr_batches = 0usize;

            let mut s = 0usize;
            while s < train.p {
                let take = (train.p - s).min(self.cfg.batch_size.max(1));
                let xb = x_tr.narrow(0, s, take)?;
                let mb = m_tr.narrow(0, s, take)?;
                let yb = y_tr.narrow(0, s, take)?;

                let (delta, attn) = self.model.forward_masked(&xb, &mb)?;
                let base = xb.narrow(2, 0, 1)?.squeeze(2)?;
                let s_adj = base.broadcast_add(&delta)?;
                let (s_pool, w_pool) =
                    pool_scores(&s_adj, &mb, &attn, &self.cfg.pool, self.cfg.tau)?;
                let mut loss = bce_with_logits_pos_weight(&s_pool, &yb, pos_weight)?;

                if self.cfg.delta_l2 > 0.0 {
                    let l2 = delta.broadcast_mul(&delta)?.mean_all()?;
                    loss = (loss + (l2 * self.cfg.delta_l2)?)?;
                }

                if self.cfg.lambda_mean_delta > 0.0 {
                    let mf = mb.to_dtype(DType::F32)?;
                    let denom = mf.sum(1)?.maximum(1.0f32)?;
                    let mean_d = delta.broadcast_mul(&mf)?.sum(1)?.broadcast_div(&denom)?;
                    let tgt = yb.gt(0.5f32)?.to_dtype(DType::F32)?;
                    let mean_sq = mean_d.broadcast_mul(&mean_d)?;
                    let loss_mean = mean_sq.broadcast_mul(&tgt)?.sum_all()?;
                    let denom = tgt.sum_all()?.maximum(1.0f32)?;
                    let loss_mean = loss_mean.broadcast_div(&denom)?;
                    loss = (loss + (loss_mean * self.cfg.lambda_mean_delta as f64)?)?;
                }

                if self.cfg.lambda_var > 0.0 {
                    let w = match self.cfg.var_weight {
                        XrunVarWeight::Uniform => uniform_weights(&mb)?,
                        XrunVarWeight::Attn => {
                            let w = attn.broadcast_mul(&mb.to_dtype(DType::F32)?)?;
                            let denom = w.sum_keepdim(1)?.maximum(1e-12f32)?;
                            w.broadcast_div(&denom)?
                        }
                        XrunVarWeight::Pool => {
                            if let Some(wp) = w_pool {
                                wp
                            } else {
                                uniform_weights(&mb)?
                            }
                        }
                    };
                    let mu = w.broadcast_mul(&s_adj)?.sum_keepdim(1)?;
                    let diff = s_adj.broadcast_sub(&mu)?;
                    let var = w.broadcast_mul(&diff.broadcast_mul(&diff)?)?.sum(1)?;
                    let tgt = yb.gt(0.5f32)?.to_dtype(DType::F32)?;
                    let loss_var = var.broadcast_mul(&tgt)?.sum_all()?;
                    let denom = tgt.sum_all()?.maximum(1.0f32)?;
                    let loss_var = loss_var.broadcast_div(&denom)?;
                    loss = (loss + (loss_var * self.cfg.lambda_var as f64)?)?;
                }

                self.opt.backward_step(&loss)?;

                tr_loss += loss.to_scalar::<f32>()?;
                tr_batches += 1;
                progress.maybe_log(epoch, tr_batches, tr_loss / tr_batches as f32);
                s += take;
            }

            let mut va_loss = 0f32;
            let mut va_batches = 0usize;
            let mut s = 0usize;
            while s < val.p {
                let take = (val.p - s).min(self.cfg.batch_size.max(1));
                let xb = x_va.narrow(0, s, take)?;
                let mb = m_va.narrow(0, s, take)?;
                let yb = y_va.narrow(0, s, take)?;

                let (delta, attn) = self.model.forward_masked(&xb, &mb)?;
                let base = xb.narrow(2, 0, 1)?.squeeze(2)?;
                let s_adj = base.broadcast_add(&delta)?;
                let (s_pool, _) = pool_scores(&s_adj, &mb, &attn, &self.cfg.pool, self.cfg.tau)?;
                let loss = bce_with_logits_pos_weight(&s_pool, &yb, pos_weight)?;

                va_loss += loss.to_scalar::<f32>()?;
                va_batches += 1;
                s += take;
            }

            let tr_loss = if tr_batches > 0 {
                tr_loss / tr_batches as f32
            } else {
                f32::INFINITY
            };
            let va_loss = if va_batches > 0 {
                va_loss / va_batches as f32
            } else {
                f32::INFINITY
            };
            log::info!(
                "[xrun] Epoch {:02} train={:.4} val={:.4}",
                epoch,
                tr_loss,
                va_loss
            );

            if va_loss < best_val - 1e-4 {
                best_val = va_loss;
                bad = 0;
                let mut p = std::env::temp_dir();
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                p.push(format!("redeem_xrun_best_{stamp}.safetensors"));
                self.varmap.save(&p)?;
                best_path = Some(p);
            } else {
                bad += 1;
                if bad >= self.cfg.patience.max(1) {
                    log::info!("[xrun] Early stopping (best val={:.4})", best_val);
                    break;
                }
            }
        }

        if let Some(p) = best_path {
            let _ = self.varmap.load(&p);
            let _ = std::fs::remove_file(&p);
        }

        Ok(XrunTrainMeta {
            best_val,
            epochs: self.cfg.max_epochs.max(1),
        })
    }
}

/// Format a wall-clock duration for compact XRUN progress logs.
fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let rem_secs = secs % 60;
    if hours > 0 {
        format!("{hours:02}:{mins:02}:{rem_secs:02}")
    } else {
        format!("{mins:02}:{rem_secs:02}")
    }
}
