use crate::config::Config;
use crate::model::topaz::{TopazBagRanker, TopazConfig};
use crate::train::scheduler::CosineWarmupScheduler;
use crate::train::losses;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, optim::AdamW, Optimizer, VarBuilder, VarMap};

#[derive(Debug)]
pub struct TrainBatch {
    /// (B,K,D)
    pub xb: Tensor,
    /// (B,K,C,L)
    pub tb: Tensor,
    /// (B,K) bool
    pub mask: Tensor,
    /// (B,)
    pub yb: Tensor,
}

#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub loss: f32,
    pub loss_bag: f32,
    pub loss_pair: f32,
    pub loss_inbag: f32,
    pub loss_winner_margin: f32,
    pub loss_ms12: f32,
}

#[derive(Debug, Clone)]
pub struct TrainHistory {
    pub epochs_ran: usize,
    pub best_epoch: usize,
    pub best_val: f32,
}

pub struct Trainer {
    pub config: Config,
    pub varmap: VarMap,
    pub model: TopazBagRanker,
    pub opt: AdamW,
    pub ms12_head: Option<nn::Linear>,
}

impl Trainer {
    pub fn new(cfg: Config, model_cfg: &TopazConfig, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = TopazBagRanker::new(vb.pp("topaz"), model_cfg)?;
        let ms12_head = if cfg.lambda_ms12 > 0.0 && model_cfg.ms1_cmax > 0 && model_cfg.use_coelution_head
        {
            Some(nn::linear(4, 1, vb.pp("ms12_head"))?)
        } else {
            None
        };

        let params = nn::optim::ParamsAdamW {
            lr: cfg.learning_rate as f64,
            weight_decay: cfg.weight_decay as f64,
            ..Default::default()
        };
        let opt = AdamW::new(varmap.all_vars(), params)?;

        Ok(Self { config: cfg, varmap, model, opt, ms12_head })
    }

    pub fn train_step(&mut self, batch: &TrainBatch) -> Result<TrainMetrics> {
        let (b, k, d) = batch.xb.dims3()?;
        let (_, _, c, l) = batch.tb.dims4()?;

        let xf = batch.xb.reshape((b * k, d))?;
        let tf = batch.tb.reshape((b * k, c, l))?;

        let (emb, coe, coe_ms12) = self.model.trace_enc.forward_components(&tf)?;
        let logits = self.model.scorer.forward(&xf, &emb, &coe)?;
        let cand = logits.reshape((b, k))?;

        let m = batch.mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = (cand.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;
        let bag = cand_masked.max(1)?;

        let loss_bag = losses::bce_with_logits(&bag, &batch.yb)?;

        let mut loss = loss_bag.clone();
        let mut loss_pair = Tensor::zeros((), DType::F32, bag.device())?;
        if self.config.lambda_pair > 0.0 {
            loss_pair = losses::pairwise_pos_neg_softplus(&bag, &batch.yb)?;
            loss = (loss + (loss_pair.clone() * self.config.lambda_pair as f64)?)?;
        }

        let mut loss_inbag = Tensor::zeros((), DType::F32, bag.device())?;
        if self.config.lambda_inbag > 0.0 {
            loss_inbag = losses::inbag_ranking_loss(
                &cand,
                &batch.mask,
                &batch.yb,
                self.config.inbag_margin,
            )?;
            loss = (loss + (loss_inbag.clone() * self.config.lambda_inbag as f64)?)?;
        }

        let mut loss_wm = Tensor::zeros((), DType::F32, bag.device())?;
        if self.config.lambda_winner_margin > 0.0 {
            loss_wm = losses::winner_margin_loss(
                &cand,
                &batch.mask,
                &batch.yb,
                self.config.winner_margin,
            )?;
            loss = (loss + (loss_wm.clone() * self.config.lambda_winner_margin as f64)?)?;
        }

        let mut loss_ms12 = Tensor::zeros((), DType::F32, bag.device())?;
        if let Some(head) = &self.ms12_head {
            if coe_ms12.elem_count() > 0 {
                let ms12 = coe_ms12.reshape((b, k, 4))?;
                let t = if self.config.ms12_soft_temp > 0.0 {
                    self.config.ms12_soft_temp as f64
                } else {
                    1.0
                };
                let s_for_soft = (&cand_masked / t)?;
                let w = candle_nn::ops::softmax(&s_for_soft, 1)?;
                let w = w.broadcast_mul(&m)?;
                let denom = w.sum_keepdim(1)?.maximum(1e-12f32)?;
                let w = w.broadcast_div(&denom)?;
                let ms12_soft = ms12.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
                let ms12_logit = ms12_soft.apply(head)?.squeeze(1)?;
                loss_ms12 = losses::bce_with_logits(&ms12_logit, &batch.yb)?;
                loss = (loss + (loss_ms12.clone() * self.config.lambda_ms12 as f64)?)?;
            }
        }

        self.opt.backward_step(&loss)?;

        Ok(TrainMetrics {
            loss: loss.to_scalar::<f32>()?,
            loss_bag: loss_bag.to_scalar::<f32>()?,
            loss_pair: loss_pair.to_scalar::<f32>()?,
            loss_inbag: loss_inbag.to_scalar::<f32>()?,
            loss_winner_margin: loss_wm.to_scalar::<f32>()?,
            loss_ms12: loss_ms12.to_scalar::<f32>()?,
        })
    }

    fn bag_logits(&self, batch: &TrainBatch) -> Result<Tensor> {
        let (b, k, d) = batch.xb.dims3()?;
        let (_, _, c, l) = batch.tb.dims4()?;

        let xf = batch.xb.reshape((b * k, d))?;
        let tf = batch.tb.reshape((b * k, c, l))?;

        let (emb, coe, _coe_ms12) = self.model.trace_enc.forward_components(&tf)?;
        let logits = self.model.scorer.forward(&xf, &emb, &coe)?;
        let cand = logits.reshape((b, k))?;

        let m = batch.mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = (cand.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;
        cand_masked.max(1)
    }

    pub fn eval_bag_loss(&self, batches: &[TrainBatch]) -> Result<f32> {
        if batches.is_empty() {
            return Ok(f32::INFINITY);
        }
        let mut sum = 0f32;
        for batch in batches {
            let bag = self.bag_logits(batch)?;
            let loss = losses::bce_with_logits(&bag, &batch.yb)?;
            sum += loss.to_scalar::<f32>()?;
        }
        Ok(sum / batches.len() as f32)
    }

    pub fn train_one_epoch(&mut self, batches: &[TrainBatch]) -> Result<Vec<TrainMetrics>> {
        let mut out = Vec::with_capacity(batches.len());
        for batch in batches {
            out.push(self.train_step(batch)?);
        }
        Ok(out)
    }

    /// Train for multiple epochs with optional cosine warmup scheduler.
    pub fn train_epochs(
        &mut self,
        batches: &[TrainBatch],
        max_epochs: usize,
        scheduler: Option<&CosineWarmupScheduler>,
    ) -> Result<Vec<TrainMetrics>> {
        let mut out = Vec::new();
        let mut step = 0usize;
        let total_steps = max_epochs.max(1) * batches.len().max(1);
        let sched = scheduler.cloned().unwrap_or_else(|| {
            CosineWarmupScheduler::new(
                self.config.learning_rate as f64,
                total_steps,
                self.config.warmup_frac as f64,
                self.config.warmup_steps,
                self.config.min_lr_ratio as f64,
            )
        });

        for _epoch in 0..max_epochs.max(1) {
            for batch in batches {
                if self.config.use_lr_scheduler {
                    let lr = sched.lr_at_step(step);
                    self.opt.set_learning_rate(lr);
                }
                out.push(self.train_step(batch)?);
                step += 1;
            }
        }
        Ok(out)
    }

    /// Train with optional early stopping on validation bag loss.
    pub fn train_epochs_early_stop(
        &mut self,
        train_batches: &[TrainBatch],
        val_batches: &[TrainBatch],
        max_epochs: usize,
        scheduler: Option<&CosineWarmupScheduler>,
    ) -> Result<TrainHistory> {
        if train_batches.is_empty() {
            return Ok(TrainHistory { epochs_ran: 0, best_epoch: 0, best_val: f32::INFINITY });
        }
        let mut step = 0usize;
        let total_steps = max_epochs.max(1) * train_batches.len().max(1);
        let sched = scheduler.cloned().unwrap_or_else(|| {
            CosineWarmupScheduler::new(
                self.config.learning_rate as f64,
                total_steps,
                self.config.warmup_frac as f64,
                self.config.warmup_steps,
                self.config.min_lr_ratio as f64,
            )
        });

        let mut best_val = f32::INFINITY;
        let mut best_epoch = 0usize;
        let mut bad = 0usize;
        let min_delta = 1e-4f32;
        let patience = self.config.patience.max(1);

        let mut best_path = None;
        if !val_batches.is_empty() {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "redeem_topaz_best_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            best_path = Some(p);
        }

        let mut epochs_ran = 0usize;
        for epoch in 1..=max_epochs.max(1) {
            let mut train_sum = 0f32;
            for batch in train_batches {
                if self.config.use_lr_scheduler {
                    let lr = sched.lr_at_step(step);
                    self.opt.set_learning_rate(lr);
                }
                let metrics = self.train_step(batch)?;
                train_sum += metrics.loss;
                step += 1;
            }
            epochs_ran = epoch;
            let train_loss = train_sum / train_batches.len() as f32;
            if val_batches.is_empty() {
                if self.config.use_lr_scheduler {
                    log::info!(
                        "Epoch {:02} train={:.4} lr={:.3e}",
                        epoch,
                        train_loss,
                        self.opt.learning_rate()
                    );
                } else {
                    log::info!("Epoch {:02} train={:.4}", epoch, train_loss);
                }
                continue;
            }

            let val_loss = self.eval_bag_loss(val_batches)?;
            if self.config.use_lr_scheduler {
                log::info!(
                    "Epoch {:02} train={:.4} val={:.4} lr={:.3e}",
                    epoch,
                    train_loss,
                    val_loss,
                    self.opt.learning_rate()
                );
            } else {
                log::info!("Epoch {:02} train={:.4} val={:.4}", epoch, train_loss, val_loss);
            }

            if val_loss + min_delta < best_val {
                best_val = val_loss;
                best_epoch = epoch;
                bad = 0;
                if let Some(path) = &best_path {
                    if let Err(err) = self.varmap.save(path) {
                        log::warn!("Failed to save best checkpoint snapshot: {err}");
                    }
                }
            } else {
                bad += 1;
                if bad >= patience {
                    log::info!("Early stopping (best val={:.4})", best_val);
                    break;
                }
            }
        }

        if let Some(path) = &best_path {
            if best_epoch > 0 {
                if let Err(err) = self.varmap.load(path) {
                    log::warn!("Failed to restore best checkpoint snapshot: {err}");
                }
            }
            let _ = std::fs::remove_file(path);
        }

        Ok(TrainHistory { epochs_ran, best_epoch, best_val })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::topaz::TopazConfig;
    use candle_core::Tensor;

    #[test]
    fn test_forward_backward_smoke() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = Config::default();
        cfg.learning_rate = 1e-2;
        cfg.lambda_pair = 0.0;
        cfg.lambda_inbag = 0.0;
        cfg.lambda_winner_margin = 0.0;

        let model_cfg = TopazConfig {
            feat_dim: 3,
            ms2_cmax: 4,
            ms1_cmax: 0,
            l: 8,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: crate::building_blocks::trace_input::TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            ..Default::default()
        };

        let mut trainer = Trainer::new(cfg, &model_cfg, &device)?;

        let (b, k, d) = (4usize, 3usize, model_cfg.feat_dim);
        let (c, l) = (model_cfg.ms2_cmax, model_cfg.l);

        let xb = Tensor::rand(0f32, 1f32, (b, k, d), &device)?;
        let tb = Tensor::rand(0f32, 1f32, (b, k, c, l), &device)?;
        let mask = Tensor::ones((b, k), DType::U8, &device)?;
        let yb = Tensor::new(vec![1f32, 0.0, 1.0, 0.0], &device)?;

        let batch = TrainBatch { xb, tb, mask, yb };

        let m1 = trainer.train_step(&batch)?;
        let m2 = trainer.train_step(&batch)?;

        assert!(m1.loss.is_finite());
        assert!(m2.loss.is_finite());
        Ok(())
    }
}
