//! The concrete TOPAZ trace-first DIA peak-group scoring model.
//!
//! TOPAZ combines three main components:
//!
//! 1. [`crate::building_blocks::conv_encoder::TraceEncoder`], which embeds the
//!    fixed RT-window traces and optionally computes explicit coelution
//!    statistics.
//! 2. [`crate::building_blocks::mlp::CandidateScorer`], which fuses heuristic
//!    features and learned trace representations into candidate logits.
//! 3. Masked-max multiple-instance learning over the candidate dimension `K`
//!    to produce one bag score per precursor/run group.
//!
//! Shape notation used in this module:
//!
//! - `N`: number of flattened candidate rows.
//! - `B`: number of bags.
//! - `K`: padded candidate count per bag.
//! - `D`: heuristic feature dimension.
//! - `C_total`: total number of trace channels presented to the encoder.
//! - `L`: fixed trace-window length.
//! - `E`: learned trace-embedding dimension after branch fusion.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use crate::building_blocks::trace_input::TraceInputMode;
use crate::model_interface::{
    BagRankerInterface, BagRankerWithHiddenInterface, CandidateScorerInterface, ModelInterface,
};

/// Configuration for the base TOPAZ model.
///
/// The model sees:
/// - heuristic features of size `feat_dim`
/// - `ms1_cmax + ms2_cmax` trace channels
/// - a fixed trace length `l`
///
/// MS1 channels, when present, are always ordered before MS2 channels.
/// That means a trace tensor `(N, C_total, L)` is interpreted as
/// `[MS1 channels | MS2 channels]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopazConfig {
    /// Number of scalar heuristic/library features per candidate row.
    pub feat_dim: usize,
    /// Maximum number of MS2 fragment traces kept per candidate.
    pub ms2_cmax: usize,
    /// Maximum number of MS1 precursor/isotope traces kept per candidate.
    pub ms1_cmax: usize,
    /// Fixed trace-window length `L`.
    pub l: usize,

    /// Embedding width produced by each trace branch before fusion.
    pub trace_emb_dim: usize,
    /// Hidden layer sizes for the candidate-scoring MLP.
    pub mlp_hidden: Vec<usize>,
    /// Dropout probability used in the scorer (and related heads when enabled).
    pub dropout: f64,

    /// Raw-vs-dual trace representation fed to the convolutional encoder.
    pub trace_input_mode: TraceInputMode,
    /// Whether scalar heuristic features are concatenated with learned trace features.
    pub use_heuristic_features: bool,
    /// Whether to compute explicit coelution features alongside learned embeddings.
    pub use_coelution_head: bool,

    /// Temperature-like scaling used when computing soft apex positions.
    pub coelution_beta: f64,
    /// Maximum lag (in samples) searched by lag-tolerant cosine features.
    pub coelution_max_lag: usize,
    /// Width of the learned similarity embedding inside the coelution head.
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

/// Concrete TOPAZ model used for both candidate-level and bag-level scoring.
///
/// This type wires together the reusable building blocks into the exact
/// architecture used by the Rust TOPAZ port: trace encoder, candidate scorer,
/// and masked-max MIL bag pooling.
pub struct TopazBagRanker {
    /// Convolutional trace encoder producing learned embeddings and optional
    /// coelution features.
    pub trace_enc: crate::building_blocks::conv_encoder::TraceEncoder,
    /// Candidate-level MLP scorer operating on heuristic features plus trace
    /// embeddings.
    pub scorer: crate::building_blocks::mlp::CandidateScorer,
}

/// Per-bag winner-side diagnostic tensors exported for reports.
///
/// Each row corresponds to the candidate that won one bag after masked-max
/// pooling.
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
    /// Construct a TOPAZ scorer under the provided variable scope.
    ///
    /// All parameters are created beneath the supplied `vb` subtree so they can
    /// be saved and loaded with stable names via Candle's `VarMap`.
    pub fn new(vb: VarBuilder, cfg: &TopazConfig) -> Result<Self> {
        let trace_enc =
            crate::building_blocks::conv_encoder::TraceEncoder::new(vb.pp("trace_enc"), cfg)?;
        let scorer = crate::building_blocks::mlp::CandidateScorer::new(
            vb.pp("candidate_scorer"),
            cfg,
            trace_enc.emb_out_dim(),
            trace_enc.coelution_dim(),
        )?;
        Ok(Self { trace_enc, scorer })
    }

    /// Score a batch of bags with masked-max MIL pooling.
    ///
    /// # Inputs
    /// - `xb`: `(B, K, D)` heuristic features.
    /// - `tb`: `(B, K, C_total, L)` trace windows.
    /// - `mask`: `(B, K)` validity mask, where `1` marks a real candidate and
    ///   `0` marks padding.
    ///
    /// # Output
    /// - candidate logits `(B, K)`
    /// - bag logits `(B,)`
    pub fn forward_bags(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
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

    /// Score bags and return the hidden state of the winning candidate in each
    /// bag.
    ///
    /// The returned hidden tensor has shape `(B, H)` and is the input consumed
    /// by XRUN together with the bag score.
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

    /// Score bags and return winner-side intermediate tensors that are useful
    /// for diagnostics, reporting, and XRUN calibration.
    ///
    /// This is the most verbose forward path and is mainly intended for report
    /// generation rather than training.
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
        let (logits, hidden) = self
            .scorer
            .forward_with_hidden_eval(&xf, &emb_all, &coe_all)?;
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

        fn select_win(comp: &Tensor, b: usize, k: usize, onehot_f: &Tensor) -> Result<Tensor> {
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
