//! Base TOPAZ trainer, early stopping, and optimizer setup.

use crate::config::Config;
use crate::model::topaz::{TopazBagRanker, TopazConfig};
use crate::train::losses;
use crate::train::scheduler::CosineWarmupScheduler;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap, optim::AdamW};
use std::time::{Duration, Instant};

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
    /// Optional auxiliary regression targets with shape `(B, K, T_distill)`.
    pub distill_targets: Option<Tensor>,
    /// Optional mask aligned to `distill_targets`, with `1` for valid targets.
    pub distill_mask: Option<Tensor>,
}

/// Scalar losses returned after one optimization step.
#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub loss: f32,
    pub loss_bag: f32,
    pub loss_pair: f32,
    pub loss_inbag: f32,
    pub loss_winner_margin: f32,
    pub loss_topk_runner: f32,
    pub loss_ms12: f32,
    pub loss_xic_bag: f32,
    pub loss_xim_bag: f32,
    pub loss_distill: f32,
}

#[derive(Clone, Debug)]
struct AuxLayerBlock {
    lin: nn::Linear,
    dropout: Option<nn::Dropout>,
}

impl AuxLayerBlock {
    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let h = xs.apply(&self.lin)?.apply(&nn::Activation::Relu)?;
        if let Some(d) = &self.dropout {
            d.forward(&h, train)
        } else {
            Ok(h)
        }
    }
}

struct MlpHead {
    layers: Vec<AuxLayerBlock>,
    head: nn::Linear,
}

impl MlpHead {
    fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        hidden: &[usize],
        dropout: f64,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(hidden.len());
        let mut d = in_dim;
        for (i, &h) in hidden.iter().enumerate() {
            let lin = nn::linear(d, h, vb.pp(format!("lin{i}")))?;
            let dropout = if dropout > 0.0 {
                Some(nn::Dropout::new(dropout as f32))
            } else {
                None
            };
            layers.push(AuxLayerBlock { lin, dropout });
            d = h;
        }
        let head = nn::linear(d, out_dim, vb.pp("head"))?;
        Ok(Self { layers, head })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.layers {
            h = layer.forward(&h, train)?;
        }
        h.apply(&self.head)
    }
}

/// Summary of an early-stopped training run.
#[derive(Debug, Clone)]
pub struct TrainHistory {
    pub epochs_ran: usize,
    pub best_epoch: usize,
    pub best_val: f32,
}

/// Periodically emits batch-level progress for long base-TOPAZ training runs.
///
/// The logger reports global progress across all epochs so multi-hour jobs show
/// visible forward motion in the logs instead of only emitting one line per
/// finished epoch.
struct TrainingProgressLogger {
    total_epochs: usize,
    batches_per_epoch: usize,
    total_batches: usize,
    started: Instant,
    last_log: Instant,
    log_every: Duration,
}

impl TrainingProgressLogger {
    /// Create a progress logger for a fixed `(epochs, batches)` training plan.
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
            log_every: Duration::from_secs(30),
        }
    }

    /// Emit an `info!` log when enough time has elapsed or training completed.
    fn maybe_log(&mut self, epoch: usize, batch_in_epoch: usize, mean_loss: f32, lr: f64) {
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
            "Training progress | epoch={}/{} batch={}/{} overall={}/{} ({:.1}%) elapsed={} eta={} rate={:.1} batches/s loss={:.4} lr={:.3e}",
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
            mean_loss,
            lr
        );
    }
}

/// Stateful TOPAZ trainer holding the model, optimizer, and optional auxiliary
/// MS1/MS2 head.
pub struct Trainer {
    pub config: Config,
    pub varmap: VarMap,
    pub model: TopazBagRanker,
    pub opt: AdamW,
    pub ms12_head: Option<nn::Linear>,
    xic_bag_head: Option<MlpHead>,
    xim_bag_head: Option<MlpHead>,
    distill_head: Option<MlpHead>,
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
    fn masked_max_bag_logits(cand: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, k) = cand.dims2()?;
        let m = mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = (cand.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;
        cand_masked.max(1)
    }

    fn zero_xim_input(&self, n: usize, device: &Device, dtype: DType) -> Result<Option<Tensor>> {
        let Some(cfg) = &self.model.xim_cfg else {
            return Ok(None);
        };
        let c_total = cfg.ms1_cmax + cfg.ms2_cmax;
        Ok(Some(Tensor::zeros((n, c_total, cfg.l), dtype, device)?))
    }

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
        let trace_repr_dim = model.trace_enc.emb_out_dim()
            + model
                .xim_enc
                .as_ref()
                .map(|enc| enc.emb_out_dim())
                .unwrap_or(0)
            + model.trace_enc.coelution_dim()
            + model
                .xim_enc
                .as_ref()
                .map(|enc| enc.coelution_dim())
                .unwrap_or(0);
        let xic_repr_dim = model.trace_enc.emb_out_dim() + model.trace_enc.coelution_dim();
        let xic_bag_head = if cfg.lambda_xic_bag > 0.0 {
            Some(MlpHead::new(
                vb.pp("xic_bag_head"),
                xic_repr_dim,
                1,
                &cfg.branch_aux_hidden,
                cfg.branch_aux_dropout,
            )?)
        } else {
            None
        };
        let xim_repr_dim = model
            .xim_enc
            .as_ref()
            .map(|enc| enc.emb_out_dim() + enc.coelution_dim());
        let xim_bag_head = if cfg.lambda_xim_bag > 0.0 {
            if let Some(in_dim) = xim_repr_dim {
                Some(MlpHead::new(
                    vb.pp("xim_bag_head"),
                    in_dim,
                    1,
                    &cfg.branch_aux_hidden,
                    cfg.branch_aux_dropout,
                )?)
            } else {
                log::warn!(
                    "lambda_xim_bag > 0 but model has no XIM branch; disabling XIM auxiliary bag head"
                );
                None
            }
        } else {
            None
        };
        let distill_head = if cfg.distill.is_enabled() {
            Some(MlpHead::new(
                vb.pp("distill_head"),
                trace_repr_dim,
                cfg.distill.cols.len(),
                &cfg.distill.hidden,
                cfg.distill.dropout,
            )?)
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
            xic_bag_head,
            xim_bag_head,
            distill_head,
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

        let (emb_xic, coe_xic, coe_ms12) = self.model.trace_enc.forward_components(&tf)?;
        let (emb_xim, coe_xim) = if let Some(xim_enc) = &self.model.xim_enc {
            let xim_tensor = if let Some(xim) = tf_aux.as_ref() {
                xim.clone()
            } else {
                self.zero_xim_input(b * k, tf.device(), tf.dtype())?
                    .expect("zero XIM requested without XIM config")
            };
            let (emb, coe, _coe_ms12_xim) = xim_enc.forward_components(&xim_tensor)?;
            (Some(emb), Some(coe))
        } else {
            (None, None)
        };
        let emb = if let Some(emb_xim) = &emb_xim {
            Tensor::cat(&[emb_xic.clone(), emb_xim.clone()], 1)?
        } else {
            emb_xic.clone()
        };
        let coe = if let Some(coe_xim) = &coe_xim {
            Tensor::cat(&[coe_xic.clone(), coe_xim.clone()], 1)?
        } else {
            coe_xic.clone()
        };
        let logits = self.model.scorer.forward(&xf, &emb, &coe)?;
        let cand = logits.reshape((b, k))?;
        let bag = Self::masked_max_bag_logits(&cand, &batch.mask)?;

        let loss_bag = losses::bce_with_logits_weighted(&bag, &batch.yb, self.pos_weight)?;

        let mut loss = loss_bag.clone();
        let m = batch.mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = (cand.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;
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

        let mut loss_topk_runner = Tensor::zeros((), DType::F32, bag.device())?;
        if self.config.lambda_topk_runner > 0.0 {
            loss_topk_runner = losses::topk_runner_margin_loss(
                &cand,
                &batch.mask,
                &batch.yb,
                self.config.topk_runner_margin,
                self.config.topk_runner_k,
            )?;
            loss = (loss + (loss_topk_runner.clone() * self.config.lambda_topk_runner as f64)?)?;
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

        let mut loss_xic_bag = Tensor::zeros((), DType::F32, bag.device())?;
        if let Some(head) = &self.xic_bag_head {
            let xic_repr = Tensor::cat(&[emb_xic.clone(), coe_xic.clone()], 1)?;
            let xic_logits = head.forward(&xic_repr, true)?.squeeze(1)?;
            let xic_cand = xic_logits.reshape((b, k))?;
            let xic_bag = Self::masked_max_bag_logits(&xic_cand, &batch.mask)?;
            loss_xic_bag = losses::bce_with_logits_weighted(&xic_bag, &batch.yb, self.pos_weight)?;
            loss = (loss + (loss_xic_bag.clone() * self.config.lambda_xic_bag as f64)?)?;
        }

        let mut loss_xim_bag = Tensor::zeros((), DType::F32, bag.device())?;
        if let (Some(head), Some(emb_xim), Some(coe_xim)) =
            (&self.xim_bag_head, emb_xim.as_ref(), coe_xim.as_ref())
        {
            let xim_repr = Tensor::cat(&[emb_xim.clone(), coe_xim.clone()], 1)?;
            let xim_logits = head.forward(&xim_repr, true)?.squeeze(1)?;
            let xim_cand = xim_logits.reshape((b, k))?;
            let xim_bag = Self::masked_max_bag_logits(&xim_cand, &batch.mask)?;
            loss_xim_bag = losses::bce_with_logits_weighted(&xim_bag, &batch.yb, self.pos_weight)?;
            loss = (loss + (loss_xim_bag.clone() * self.config.lambda_xim_bag as f64)?)?;
        }

        let mut loss_distill = Tensor::zeros((), DType::F32, bag.device())?;
        if let (Some(head), Some(targets), Some(mask)) = (
            self.distill_head.as_ref(),
            batch.distill_targets.as_ref(),
            batch.distill_mask.as_ref(),
        ) {
            let (_, _, n_targets) = targets.dims3()?;
            if n_targets > 0 {
                let trace_repr = Tensor::cat(&[emb.clone(), coe.clone()], 1)?;
                let pred = head.forward(&trace_repr, true)?;
                let target_flat = targets.reshape((b * k, n_targets))?;
                let mask_flat = mask.reshape((b * k, n_targets))?;
                loss_distill = losses::masked_huber_loss(
                    &pred,
                    &target_flat,
                    &mask_flat,
                    self.config.distill.huber_delta,
                )?;
                loss = (loss + (loss_distill.clone() * self.config.distill.lambda as f64)?)?;
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
            loss_topk_runner: loss_topk_runner.to_scalar::<f32>()?,
            loss_ms12: loss_ms12.to_scalar::<f32>()?,
            loss_xic_bag: loss_xic_bag.to_scalar::<f32>()?,
            loss_xim_bag: loss_xim_bag.to_scalar::<f32>()?,
            loss_distill: loss_distill.to_scalar::<f32>()?,
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
        Self::masked_max_bag_logits(&cand, &batch.mask)
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
        let epochs = max_epochs.max(1);
        let total_steps = epochs * batches.len().max(1);
        let sched = scheduler.cloned().unwrap_or_else(|| {
            CosineWarmupScheduler::new(
                self.config.learning_rate as f64,
                total_steps,
                self.config.warmup_frac as f64,
                self.config.warmup_steps,
                self.config.min_lr_ratio as f64,
            )
        });
        let mut progress = TrainingProgressLogger::new(epochs, batches.len());

        for epoch in 0..epochs {
            let mut order: Vec<usize> = (0..batches.len()).collect();
            if batches.len() > 1 {
                let seed = self
                    .shuffle_seed
                    .unwrap_or(0)
                    .wrapping_add(epoch as u64 + 1);
                shuffle_indices(&mut order, seed);
            }
            let mut epoch_sum = 0.0f32;
            let mut epoch_batches = 0usize;
            for &bi in &order {
                let batch = &batches[bi];
                if self.config.use_lr_scheduler {
                    let lr = sched.lr_at_step(step);
                    self.opt.set_learning_rate(lr);
                }
                let metrics = self.train_step(batch)?;
                epoch_sum += metrics.loss;
                epoch_batches += 1;
                progress.maybe_log(
                    epoch + 1,
                    epoch_batches,
                    epoch_sum / epoch_batches as f32,
                    self.opt.learning_rate(),
                );
                out.push(metrics);
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
        let eval_every = self.config.eval_every.max(1);

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

        let epochs = max_epochs.max(1);
        let mut progress = TrainingProgressLogger::new(epochs, train_batches.len());
        let mut epochs_ran = 0usize;
        for epoch in 1..=epochs {
            let mut train_sum = 0f32;
            let mut train_batches_done = 0usize;
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
                train_batches_done += 1;
                progress.maybe_log(
                    epoch,
                    train_batches_done,
                    train_sum / train_batches_done as f32,
                    self.opt.learning_rate(),
                );
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

            if epoch % eval_every != 0 && epoch != epochs {
                if self.config.use_lr_scheduler {
                    log::info!(
                        "Epoch {:02} train={:.4} lr={:.3e} val=skipped(eval_every={})",
                        epoch,
                        train_loss,
                        self.opt.learning_rate(),
                        eval_every
                    );
                } else {
                    log::info!(
                        "Epoch {:02} train={:.4} val=skipped(eval_every={})",
                        epoch,
                        train_loss,
                        eval_every
                    );
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

/// Format a wall-clock duration for compact progress logs.
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
            distill_targets: None,
            distill_mask: None,
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
