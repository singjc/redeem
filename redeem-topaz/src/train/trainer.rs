//! Base TOPAZ trainer, early stopping, and optimizer setup.

use crate::config::Config;
use crate::model::topaz::{TopazBagRanker, TopazConfig};
use crate::train::losses;
use crate::train::scheduler::CosineWarmupScheduler;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap, optim::AdamW};

/// One tensor mini-batch used by the base TOPAZ trainer.
#[derive(Debug)]
pub struct TrainBatch {
    /// (B,K,D)
    pub xb: Tensor,
    /// (B,K,C,L)
    pub tb: Tensor,
    /// Optional auxiliary signal tensor, e.g. mobilograms, with shape
    /// `(B, K, C_aux, L_aux)`.
    pub tb_aux: Option<Tensor>,
    /// (B,K) bool
    pub mask: Tensor,
    /// (B,)
    pub yb: Tensor,
}

/// Scalar losses returned after one optimization step.
#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub loss: f32,
    pub loss_bag: f32,
    pub loss_pair: f32,
    pub loss_inbag: f32,
    pub loss_winner_margin: f32,
    pub loss_ms12: f32,
}

/// Summary of an early-stopped training run.
#[derive(Debug, Clone)]
pub struct TrainHistory {
    pub epochs_ran: usize,
    pub best_epoch: usize,
    pub best_val: f32,
}

/// Stateful TOPAZ trainer holding the model, optimizer, and optional auxiliary
/// MS1/MS2 head.
pub struct Trainer {
    pub config: Config,
    pub varmap: VarMap,
    pub model: TopazBagRanker,
    pub opt: AdamW,
    pub ms12_head: Option<nn::Linear>,
    pub pos_weight: Option<f32>,
    pub shuffle_seed: Option<u64>,
}

fn starts_with_any(name: &str, prefixes: &[String]) -> bool {
    prefixes
        .iter()
        .any(|p| !p.is_empty() && name.starts_with(p))
}

fn select_optimizer_vars(cfg: &Config, varmap: &VarMap) -> Result<Vec<candle_core::Var>> {
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort();
    let mut vars = Vec::new();
    let mut selected_names = Vec::new();
    for name in names {
        let trainable = if cfg.trainable_prefixes.is_empty() {
            true
        } else {
            starts_with_any(&name, &cfg.trainable_prefixes)
        };
        let frozen = starts_with_any(&name, &cfg.frozen_prefixes);
        if trainable && !frozen {
            if let Some(var) = data.get(&name) {
                vars.push(var.clone());
                selected_names.push(name);
            }
        }
    }
    drop(data);
    if vars.is_empty() {
        candle_core::bail!(
            "no trainable variables matched trainable_prefixes={:?} frozen_prefixes={:?}",
            cfg.trainable_prefixes,
            cfg.frozen_prefixes
        );
    }
    if !cfg.trainable_prefixes.is_empty() || !cfg.frozen_prefixes.is_empty() {
        log::info!(
            "Optimizer will update {} tensors (trainable_prefixes={:?}, frozen_prefixes={:?})",
            vars.len(),
            cfg.trainable_prefixes,
            cfg.frozen_prefixes
        );
        log::debug!("Trainable tensor names: {:?}", selected_names);
    }
    Ok(vars)
}

impl Trainer {
    /// Construct a trainer and initialize optimizer parameters.
    pub fn new(cfg: Config, model_cfg: &TopazConfig, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = TopazBagRanker::new(vb.pp("topaz"), model_cfg)?;
        let ms12_head =
            if cfg.lambda_ms12 > 0.0 && model_cfg.ms1_cmax > 0 && model_cfg.use_coelution_head {
                Some(nn::linear(4, 1, vb.pp("ms12_head"))?)
            } else {
                None
            };

        let params = nn::optim::ParamsAdamW {
            lr: cfg.learning_rate as f64,
            weight_decay: cfg.weight_decay as f64,
            ..Default::default()
        };
        let opt_vars = select_optimizer_vars(&cfg, &varmap)?;
        let opt = AdamW::new(opt_vars, params)?;

        Ok(Self {
            config: cfg,
            varmap,
            model,
            opt,
            ms12_head,
            pos_weight: None,
            shuffle_seed: None,
        })
    }

    /// Set the positive-class weight used by bag BCE.
    pub fn set_pos_weight(&mut self, pos_weight: f32) {
        if pos_weight.is_finite() && pos_weight > 0.0 {
            self.pos_weight = Some(pos_weight);
        } else {
            self.pos_weight = None;
        }
    }

    /// Set the deterministic seed used for per-epoch batch shuffling.
    pub fn set_shuffle_seed(&mut self, seed: u64) {
        self.shuffle_seed = Some(seed);
    }

    fn clip_grad_norm(&self, grads: &mut candle_core::backprop::GradStore) -> Result<()> {
        let max_norm = self.config.max_grad_norm;
        if !(max_norm > 0.0) {
            return Ok(());
        }
        let mut sum = 0f64;
        for var in self.varmap.all_vars() {
            if let Some(g) = grads.get(&var) {
                let g = g.to_dtype(DType::F32)?;
                let v = g.sqr()?.sum_all()?.to_scalar::<f32>()?;
                sum += v as f64;
            }
        }
        let norm = sum.sqrt() as f32;
        if norm <= max_norm || norm == 0.0 {
            return Ok(());
        }
        let scale = max_norm / norm;
        for var in self.varmap.all_vars() {
            if let Some(g) = grads.remove(&var) {
                let scale_t = Tensor::full(scale, g.dims(), g.device())?;
                let scaled = g.broadcast_mul(&scale_t)?;
                grads.insert(&var, scaled);
            }
        }
        Ok(())
    }

    /// Execute one optimization step on a single batch.
    pub fn train_step(&mut self, batch: &TrainBatch) -> Result<TrainMetrics> {
        let (b, k, d) = batch.xb.dims3()?;
        let (_, _, c, l) = batch.tb.dims4()?;

        let xf = batch.xb.reshape((b * k, d))?;
        let tf = batch.tb.reshape((b * k, c, l))?;
        let tf_aux = if let Some(tb_aux) = batch.tb_aux.as_ref() {
            let (_, _, c_aux, l_aux) = tb_aux.dims4()?;
            Some(tb_aux.reshape((b * k, c_aux, l_aux))?)
        } else {
            None
        };

        let (emb, coe, coe_ms12) = self.model.encode_inputs(&tf, tf_aux.as_ref())?;
        let logits = self.model.scorer.forward(&xf, &emb, &coe)?;
        let cand = logits.reshape((b, k))?;

        let m = batch.mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = (cand.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;
        let bag = cand_masked.max(1)?;

        let loss_bag = losses::bce_with_logits_weighted(&bag, &batch.yb, self.pos_weight)?;

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
                loss_ms12 =
                    losses::bce_with_logits_weighted(&ms12_logit, &batch.yb, self.pos_weight)?;
                loss = (loss + (loss_ms12.clone() * self.config.lambda_ms12 as f64)?)?;
            }
        }

        let mut grads = loss.backward()?;
        self.clip_grad_norm(&mut grads)?;
        self.opt.step(&grads)?;

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
        let tf_aux = if let Some(tb_aux) = batch.tb_aux.as_ref() {
            let (_, _, c_aux, l_aux) = tb_aux.dims4()?;
            Some(tb_aux.reshape((b * k, c_aux, l_aux))?)
        } else {
            None
        };

        let (emb, coe, _coe_ms12) = self.model.encode_inputs(&tf, tf_aux.as_ref())?;
        let logits = self.model.scorer.forward_eval(&xf, &emb, &coe)?;
        let cand = logits.reshape((b, k))?;

        let m = batch.mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = (cand.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;
        cand_masked.max(1)
    }

    /// Evaluate mean bag BCE loss over a validation set.
    pub fn eval_bag_loss(&self, batches: &[TrainBatch]) -> Result<f32> {
        if batches.is_empty() {
            return Ok(f32::INFINITY);
        }
        let mut sum = 0f32;
        for batch in batches {
            let bag = self.bag_logits(batch)?;
            let loss = losses::bce_with_logits_weighted(&bag, &batch.yb, self.pos_weight)?;
            sum += loss.to_scalar::<f32>()?;
        }
        Ok(sum / batches.len() as f32)
    }

    /// Train for one epoch with deterministic shuffling.
    pub fn train_one_epoch(&mut self, batches: &[TrainBatch]) -> Result<Vec<TrainMetrics>> {
        let mut out = Vec::with_capacity(batches.len());
        let mut order: Vec<usize> = (0..batches.len()).collect();
        if batches.len() > 1 {
            let seed = self.shuffle_seed.unwrap_or(0).wrapping_add(1);
            shuffle_indices(&mut order, seed);
        }
        for &bi in &order {
            out.push(self.train_step(&batches[bi])?);
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

        for epoch in 0..max_epochs.max(1) {
            let mut order: Vec<usize> = (0..batches.len()).collect();
            if batches.len() > 1 {
                let seed = self
                    .shuffle_seed
                    .unwrap_or(0)
                    .wrapping_add(epoch as u64 + 1);
                shuffle_indices(&mut order, seed);
            }
            for &bi in &order {
                let batch = &batches[bi];
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
            return Ok(TrainHistory {
                epochs_ran: 0,
                best_epoch: 0,
                best_val: f32::INFINITY,
            });
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
            let mut order: Vec<usize> = (0..train_batches.len()).collect();
            if train_batches.len() > 1 {
                let seed = self.shuffle_seed.unwrap_or(0).wrapping_add(epoch as u64);
                shuffle_indices(&mut order, seed);
            }
            for &bi in &order {
                let batch = &train_batches[bi];
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
                log::info!(
                    "Epoch {:02} train={:.4} val={:.4}",
                    epoch,
                    train_loss,
                    val_loss
                );
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

        Ok(TrainHistory {
            epochs_ran,
            best_epoch,
            best_val,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::topaz::TopazConfig;
    use candle_core::Tensor;
    use candle_nn::{self as nn, VarBuilder};

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

        let batch = TrainBatch {
            xb,
            tb,
            tb_aux: None,
            mask,
            yb,
        };

        let m1 = trainer.train_step(&batch)?;
        let m2 = trainer.train_step(&batch)?;

        assert!(m1.loss.is_finite());
        assert!(m2.loss.is_finite());
        Ok(())
    }

    #[test]
    fn test_optimizer_prefix_filter_selects_subset() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _keep = nn::linear(4, 3, vb.pp("keep"))?;
        let _freeze = nn::linear(4, 3, vb.pp("freeze"))?;

        let total = varmap.data().lock().unwrap().len();
        assert!(total >= 4);

        let mut cfg = Config::default();
        cfg.trainable_prefixes = vec!["keep".to_string()];
        let keep_only = select_optimizer_vars(&cfg, &varmap)?;
        let keep_expected = varmap
            .data()
            .lock()
            .unwrap()
            .keys()
            .filter(|name| name.starts_with("keep"))
            .count();
        assert_eq!(keep_only.len(), keep_expected);

        let mut cfg = Config::default();
        cfg.frozen_prefixes = vec!["freeze".to_string()];
        let no_freeze = select_optimizer_vars(&cfg, &varmap)?;
        let freeze_expected = varmap
            .data()
            .lock()
            .unwrap()
            .keys()
            .filter(|name| name.starts_with("freeze"))
            .count();
        assert_eq!(no_freeze.len(), total - freeze_expected);

        Ok(())
    }
}
