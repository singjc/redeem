// redeem-topaz/src/model/topaz.rs

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use crate::building_blocks::trace_input::TraceInputMode;
use crate::model_interface::{
    BagRankerInterface, BagRankerWithHiddenInterface, CandidateScorerInterface, ModelInterface,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopazConfig {
    pub feat_dim: usize,
    pub ms2_cmax: usize,
    pub ms1_cmax: usize,
    pub l: usize,

    pub trace_emb_dim: usize,
    pub mlp_hidden: Vec<usize>,
    pub dropout: f64,

    pub trace_input_mode: TraceInputMode,
    pub use_heuristic_features: bool,
    pub use_coelution_head: bool,

    // coelution
    pub coelution_beta: f64,
    pub coelution_max_lag: usize,
    pub coelution_sim_emb_dim: usize,
}

impl Default for TopazConfig {
    fn default() -> Self {
        Self {
            feat_dim: 0,
            ms2_cmax: 6,
            ms1_cmax: 0,
            l: 64,
            trace_emb_dim: 64,
            mlp_hidden: vec![128, 64],
            dropout: 0.1,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: true,
            coelution_beta: 10.0,
            coelution_max_lag: 2,
            coelution_sim_emb_dim: 8,
        }
    }
}

pub struct TopazBagRanker {
    pub trace_enc: crate::building_blocks::conv_encoder::TraceEncoder,
    pub scorer: crate::building_blocks::mlp::CandidateScorer,
}

#[derive(Debug, Clone)]
pub struct BagHeadComponents {
    pub emb_ms2: Tensor,
    pub emb_ms1: Tensor,
    pub emb_all: Tensor,
    pub coe_ms2: Tensor,
    pub coe_ms1: Tensor,
    pub coe_ms12: Tensor,
    pub coe_all: Tensor,
}

impl TopazBagRanker {
    pub fn new(vb: VarBuilder, cfg: &TopazConfig) -> Result<Self> {
        let trace_enc = crate::building_blocks::conv_encoder::TraceEncoder::new(
            vb.pp("trace_enc"),
            cfg,
        )?;
        let scorer = crate::building_blocks::mlp::CandidateScorer::new(
            vb.pp("candidate_scorer"),
            cfg,
            trace_enc.emb_out_dim(),
            trace_enc.coelution_dim(),
        )?;
        Ok(Self { trace_enc, scorer })
    }

    /// Xb: (B,K,D), Tb: (B,K,C,L), mask: (B,K) bool
    /// Returns (cand_logits: (B,K), bag_logits: (B,))
    pub fn forward_bags(&self, xb: &Tensor, tb: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, k, d) = xb.dims3()?;
        let (_, _, c, l) = tb.dims4()?;

        let xf = xb.reshape((b * k, d))?;
        let tf = tb.reshape((b * k, c, l))?;

        let (emb, coe) = self.trace_enc.forward(&tf)?; // (B*K, E), (B*K, Coe)
        let logits = self.scorer.forward_eval(&xf, &emb, &coe)?; // (B*K,)
        let cand = logits.reshape((b, k))?;

        // masked max over K (avoid -inf with float mask)
        let m = mask.to_dtype(DType::F32)?; // 1 valid, 0 invalid
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = ((&cand * &m)? + (&neg_big * (&ones - &m)?)?)?;
        let bag = cand_masked.max(1)?; // (B,)

        Ok((cand, bag))
    }

    /// Forward for bags and also return winner hidden (penultimate) per bag.
    /// Returns (cand_logits: (B,K), bag_logits: (B,), winner_hidden: (B,H))
    pub fn forward_bags_with_hidden(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, k, d) = xb.dims3()?;
        let (_, _, c, l) = tb.dims4()?;

        let xf = xb.reshape((b * k, d))?;
        let tf = tb.reshape((b * k, c, l))?;

        let (emb, coe) = self.trace_enc.forward(&tf)?;
        let (logits, hidden) = self.scorer.forward_with_hidden_eval(&xf, &emb, &coe)?;
        let cand = logits.reshape((b, k))?;
        let hidden = hidden.reshape((b, k, self.scorer.hidden_dim()))?;

        let m = mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = ((&cand * &m)? + (&neg_big * (&ones - &m)?)?)?;
        let bag = cand_masked.max(1)?;

        let k_best = cand_masked.argmax(1)?.to_dtype(DType::I64)?; // (B,)
        let idx = Tensor::arange(0i64, k as i64, cand.device())?
            .reshape((1, k))?
            .broadcast_as((b, k))?;
        let k_best = k_best.reshape((b, 1))?.broadcast_as((b, k))?;
        let onehot = idx.eq(&k_best)?;
        let onehot_f = onehot.to_dtype(DType::F32)?;
        let win = hidden.broadcast_mul(&onehot_f.unsqueeze(2)?)?.sum(1)?; // (B,H)

        let has = m.sum(1)?.gt(0.0f32)?.to_dtype(DType::F32)?;
        let win = win.broadcast_mul(&has.unsqueeze(1)?)?;

        Ok((cand, bag, win))
    }

    /// Forward for bags, returning winner hidden and per-head components.
    /// Returns (cand_logits: (B,K), bag_logits: (B,), winner_hidden: (B,H), components).
    pub fn forward_bags_with_heads(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, BagHeadComponents)> {
        let (b, k, d) = xb.dims3()?;
        let (_, _, c, l) = tb.dims4()?;

        let xf = xb.reshape((b * k, d))?;
        let tf = tb.reshape((b * k, c, l))?;

        let (emb_all, coe_all, comps) = self.trace_enc.forward_with_heads(&tf)?;
        let (logits, hidden) = self.scorer.forward_with_hidden_eval(&xf, &emb_all, &coe_all)?;
        let cand = logits.reshape((b, k))?;
        let hidden = hidden.reshape((b, k, self.scorer.hidden_dim()))?;

        let m = mask.to_dtype(DType::F32)?;
        let neg_big = Tensor::full(-1e9f32, (b, k), cand.device())?;
        let ones = m.ones_like()?;
        let cand_masked = ((&cand * &m)? + (&neg_big * (&ones - &m)?)?)?;
        let bag = cand_masked.max(1)?;

        let k_best = cand_masked.argmax(1)?.to_dtype(DType::I64)?;
        let idx = Tensor::arange(0i64, k as i64, cand.device())?
            .reshape((1, k))?
            .broadcast_as((b, k))?;
        let k_best = k_best.reshape((b, 1))?.broadcast_as((b, k))?;
        let onehot = idx.eq(&k_best)?;
        let onehot_f = onehot.to_dtype(DType::F32)?;

        let win = hidden.broadcast_mul(&onehot_f.unsqueeze(2)?)?.sum(1)?;
        let has = m.sum(1)?.gt(0.0f32)?.to_dtype(DType::F32)?;
        let win = win.broadcast_mul(&has.unsqueeze(1)?)?;

        fn select_win(
            comp: &Tensor,
            b: usize,
            k: usize,
            onehot_f: &Tensor,
        ) -> Result<Tensor> {
            let (n, d) = comp.dims2()?;
            if d == 0 || n == 0 {
                return Tensor::zeros((b, 0), DType::F32, comp.device());
            }
            let comp_bk = comp.reshape((b, k, d))?;
            comp_bk.broadcast_mul(&onehot_f.unsqueeze(2)?)?.sum(1)
        }

        let emb_ms2 = select_win(&comps.emb_ms2, b, k, &onehot_f)?;
        let emb_ms1 = select_win(&comps.emb_ms1, b, k, &onehot_f)?;
        let emb_all = select_win(&comps.emb_all, b, k, &onehot_f)?;
        let coe_ms2 = select_win(&comps.coe_ms2, b, k, &onehot_f)?;
        let coe_ms1 = select_win(&comps.coe_ms1, b, k, &onehot_f)?;
        let coe_ms12 = select_win(&comps.coe_ms12, b, k, &onehot_f)?;
        let coe_all = select_win(&comps.coe_all, b, k, &onehot_f)?;

        Ok((
            cand,
            bag,
            win,
            BagHeadComponents {
                emb_ms2,
                emb_ms1,
                emb_all,
                coe_ms2,
                coe_ms1,
                coe_ms12,
                coe_all,
            },
        ))
    }
}

impl ModelInterface for TopazBagRanker {
    type Config = TopazConfig;
    type Input = (Tensor, Tensor, Tensor); // (xb, tb, mask)
    type Output = (Tensor, Tensor); // (cand, bag)

    fn new(vb: VarBuilder, cfg: &Self::Config) -> Result<Self> {
        TopazBagRanker::new(vb, cfg)
    }

    fn forward(&self, input: &Self::Input) -> Result<Self::Output> {
        self.forward_bags(&input.0, &input.1, &input.2)
    }
}

impl CandidateScorerInterface for TopazBagRanker {
    fn forward_candidates(&self, x_feat: &Tensor, x_trace: &Tensor) -> Result<Tensor> {
        let (emb, coe) = self.trace_enc.forward(x_trace)?;
        self.scorer.forward_eval(x_feat, &emb, &coe)
    }
}

impl BagRankerInterface for TopazBagRanker {
    fn forward_bags(&self, xb: &Tensor, tb: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        TopazBagRanker::forward_bags(self, xb, tb, mask)
    }
}

impl BagRankerWithHiddenInterface for TopazBagRanker {
    fn forward_bags_with_hidden(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        TopazBagRanker::forward_bags_with_hidden(self, xb, tb, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_forward_bags_ms2_only_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TopazConfig {
            feat_dim: 5,
            ms2_cmax: 4,
            ms1_cmax: 0,
            l: 16,
            trace_emb_dim: 8,
            mlp_hidden: vec![16],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        let (b, k, d) = (2usize, 3usize, cfg.feat_dim);
        let (c, l) = (cfg.ms2_cmax, cfg.l);

        let xb = Tensor::zeros((b, k, d), DType::F32, &device)?;
        let tb = Tensor::zeros((b, k, c, l), DType::F32, &device)?;
        let mask = Tensor::new(vec![1u8, 1, 0, 1, 0, 0], &device)?.reshape((b, k))?;

        let (cand, bag) = model.forward_bags(&xb, &tb, &mask)?;

        let (cb, ck) = cand.dims2()?;
        assert_eq!((cb, ck), (b, k));
        let bb = bag.dims1()?;
        assert_eq!(bb, b);

        Ok(())
    }
}
