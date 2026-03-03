use crate::config::Config;
use crate::model::topaz::{TopazBagRanker, TopazConfig};
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

    pub fn train_one_epoch(&mut self, batches: &[TrainBatch]) -> Result<Vec<TrainMetrics>> {
        let mut out = Vec::with_capacity(batches.len());
        for batch in batches {
            out.push(self.train_step(batch)?);
        }
        Ok(out)
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
