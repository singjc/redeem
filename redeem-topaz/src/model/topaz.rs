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
use candle_nn::{self as nn, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::building_blocks::trace_input::TraceInputMode;
use crate::model_interface::{
    BagRankerInterface, BagRankerWithHiddenInterface, CandidateScorerInterface, ModelInterface,
};

/// Configuration for the optional ion-mobilogram (`XIM`) branch.
///
/// This branch mirrors the main chromatogram encoder but operates on
/// candidate-specific mobilograms keyed by `FEATURE_ID` rather than on
/// precursor chromatograms keyed by `PRECURSOR_ID`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TopazXimConfig {
    /// Maximum number of MS2 mobilogram channels kept per candidate.
    pub ms2_cmax: usize,
    /// Maximum number of MS1 mobilogram channels kept per candidate.
    pub ms1_cmax: usize,
    /// Fixed mobilogram length after cropping/padding along the mobility axis.
    pub l: usize,
    /// Embedding width produced by the mobilogram encoder before fusion.
    pub trace_emb_dim: usize,
    /// Input transform applied before the mobilogram convolutions.
    pub trace_input_mode: TraceInputMode,
    /// Whether explicit mobilogram coelution/shape features are enabled.
    #[serde(alias = "use_coelution")]
    pub use_coelution_head: bool,
    /// Temperature-like scaling used by soft apex calculations.
    pub coelution_beta: f64,
    /// Maximum lag searched by lag-tolerant cosine features.
    pub coelution_max_lag: usize,
    /// Width of the learned similarity embedding inside the coelution head.
    #[serde(alias = "coelution_emb_dim")]
    pub coelution_sim_emb_dim: usize,
    /// Whether to add an explicit transition-interaction embedding over XIM channels.
    pub use_transition_interaction: bool,
    /// Width of the learned transition-interaction embedding appended per branch.
    pub transition_interaction_dim: usize,
}

impl Default for TopazXimConfig {
    fn default() -> Self {
        Self {
            ms2_cmax: 6,
            ms1_cmax: 0,
            l: 258,
            trace_emb_dim: 64,
            trace_input_mode: TraceInputMode::Single,
            use_coelution_head: true,
            coelution_beta: 10.0,
            coelution_max_lag: 2,
            coelution_sim_emb_dim: 8,
            use_transition_interaction: false,
            transition_interaction_dim: 16,
        }
    }
}

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
#[serde(default)]
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
    /// Whether to add an explicit transition-interaction embedding over XIC channels.
    pub use_transition_interaction: bool,
    /// Width of the learned transition-interaction embedding appended per branch.
    pub transition_interaction_dim: usize,
    /// Whether to learn per-candidate XIC/XIM fusion weights before final scoring.
    pub use_modality_gated_fusion: bool,
    /// Hidden layer widths for the modality gate MLP.
    pub modality_gate_hidden: Vec<usize>,
    /// Dropout applied inside the modality gate.
    pub modality_gate_dropout: f64,
    /// How the learned modality gate combines XIC and XIM evidence.
    pub modality_gate_mode: ModalityGateMode,
    /// Optional ion-mobilogram encoder branch used for diaPASEF-style inputs.
    pub xim: Option<TopazXimConfig>,
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
            use_transition_interaction: false,
            transition_interaction_dim: 16,
            use_modality_gated_fusion: false,
            modality_gate_hidden: Vec::new(),
            modality_gate_dropout: 0.0,
            modality_gate_mode: ModalityGateMode::default(),
            xim: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModalityGateMode {
    #[default]
    Competitive,
    ResidualXim,
}

#[derive(Clone, Debug)]
struct FusionLayerBlock {
    lin: nn::Linear,
    dropout: Option<nn::Dropout>,
}

impl FusionLayerBlock {
    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let h = xs.apply(&self.lin)?.apply(&nn::Activation::Relu)?;
        if let Some(d) = &self.dropout {
            d.forward(&h, train)
        } else {
            Ok(h)
        }
    }
}

#[derive(Clone, Debug)]
struct ModalityGate {
    layers: Vec<FusionLayerBlock>,
    head: nn::Linear,
    mode: ModalityGateMode,
}

impl ModalityGate {
    fn new(
        vb: VarBuilder,
        in_dim: usize,
        hidden: &[usize],
        dropout: f64,
        mode: ModalityGateMode,
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
            layers.push(FusionLayerBlock { lin, dropout });
            d = h;
        }
        let out_dim = match mode {
            ModalityGateMode::Competitive => 2,
            ModalityGateMode::ResidualXim => 1,
        };
        let head = nn::linear(d, out_dim, vb.pp("head"))?;
        Ok(Self { layers, head, mode })
    }

    fn reweight(
        &self,
        xic_emb: &Tensor,
        xic_coe: &Tensor,
        xim_emb: &Tensor,
        xim_coe: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut x = Tensor::cat(
            &[
                xic_emb.clone(),
                xic_coe.clone(),
                xim_emb.clone(),
                xim_coe.clone(),
            ],
            1,
        )?;
        for layer in &self.layers {
            x = layer.forward(&x, train)?;
        }
        let logits = x.apply(&self.head)?;
        match self.mode {
            ModalityGateMode::Competitive => {
                let w = candle_nn::ops::softmax(&logits, 1)?;
                let w_xic = w.narrow(1, 0, 1)?;
                let w_xim = w.narrow(1, 1, 1)?;
                Ok((
                    xic_emb.broadcast_mul(&w_xic)?,
                    xic_coe.broadcast_mul(&w_xic)?,
                    xim_emb.broadcast_mul(&w_xim)?,
                    xim_coe.broadcast_mul(&w_xim)?,
                ))
            }
            ModalityGateMode::ResidualXim => {
                let w_xim = candle_nn::ops::sigmoid(&logits)?;
                Ok((
                    xic_emb.clone(),
                    xic_coe.clone(),
                    xim_emb.broadcast_mul(&w_xim)?,
                    xim_coe.broadcast_mul(&w_xim)?,
                ))
            }
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
    /// Optional mobilogram encoder branch used for diaPASEF-style XIM inputs.
    pub xim_enc: Option<crate::building_blocks::conv_encoder::TraceEncoder>,
    /// Configuration for the optional mobilogram branch, retained so the model
    /// can create aligned zero tensors when XIM input is absent.
    pub xim_cfg: Option<TopazXimConfig>,
    /// Optional learned gate that reweights XIC vs XIM evidence per candidate.
    modality_gate: Option<ModalityGate>,
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
    fn encoder_cfg_from_xim(cfg: &TopazXimConfig) -> TopazConfig {
        TopazConfig {
            feat_dim: 0,
            ms2_cmax: cfg.ms2_cmax,
            ms1_cmax: cfg.ms1_cmax,
            l: cfg.l,
            trace_emb_dim: cfg.trace_emb_dim,
            mlp_hidden: Vec::new(),
            dropout: 0.0,
            trace_input_mode: cfg.trace_input_mode,
            use_heuristic_features: false,
            use_coelution_head: cfg.use_coelution_head,
            coelution_beta: cfg.coelution_beta,
            coelution_max_lag: cfg.coelution_max_lag,
            coelution_sim_emb_dim: cfg.coelution_sim_emb_dim,
            use_transition_interaction: cfg.use_transition_interaction,
            transition_interaction_dim: cfg.transition_interaction_dim,
            use_modality_gated_fusion: false,
            modality_gate_hidden: Vec::new(),
            modality_gate_dropout: 0.0,
            modality_gate_mode: ModalityGateMode::default(),
            xim: None,
        }
    }

    fn zero_xim_input(
        &self,
        n: usize,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let Some(cfg) = &self.xim_cfg else {
            candle_core::bail!("missing XIM config for XIM-enabled model");
        };
        let c_total = cfg.ms1_cmax + cfg.ms2_cmax;
        Tensor::zeros((n, c_total, cfg.l), dtype, device)
    }

    pub(crate) fn fuse_modalities(
        &self,
        xic_emb: &Tensor,
        xic_coe: &Tensor,
        xim_emb: Option<&Tensor>,
        xim_coe: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        if let (Some(xim_emb), Some(xim_coe)) = (xim_emb, xim_coe) {
            let (xic_emb_f, xic_coe_f, xim_emb_f, xim_coe_f) =
                if let Some(gate) = &self.modality_gate {
                    gate.reweight(xic_emb, xic_coe, xim_emb, xim_coe, train)?
                } else {
                    (
                        xic_emb.clone(),
                        xic_coe.clone(),
                        xim_emb.clone(),
                        xim_coe.clone(),
                    )
                };
            Ok((
                Tensor::cat(&[xic_emb_f, xim_emb_f], 1)?,
                Tensor::cat(&[xic_coe_f, xim_coe_f], 1)?,
            ))
        } else {
            Ok((xic_emb.clone(), xic_coe.clone()))
        }
    }

    pub(crate) fn encode_inputs(
        &self,
        x_trace: &Tensor,
        x_xim: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (emb_xic, coe_xic, coe_ms12_xic) = self.trace_enc.forward_components(x_trace)?;
        if let Some(xim_enc) = &self.xim_enc {
            let (n, _, _) = x_trace.dims3()?;
            let xim_tensor = if let Some(xim) = x_xim {
                xim.clone()
            } else {
                self.zero_xim_input(n, x_trace.device(), x_trace.dtype())?
            };
            let (emb_xim, coe_xim, _coe_ms12_xim) = xim_enc.forward_components(&xim_tensor)?;
            let (emb, coe) =
                self.fuse_modalities(&emb_xic, &coe_xic, Some(&emb_xim), Some(&coe_xim), false)?;
            Ok((emb, coe, coe_ms12_xic))
        } else {
            Ok((emb_xic, coe_xic, coe_ms12_xic))
        }
    }

    /// Construct a TOPAZ scorer under the provided variable scope.
    ///
    /// All parameters are created beneath the supplied `vb` subtree so they can
    /// be saved and loaded with stable names via Candle's `VarMap`.
    pub fn new(vb: VarBuilder, cfg: &TopazConfig) -> Result<Self> {
        let trace_enc =
            crate::building_blocks::conv_encoder::TraceEncoder::new(vb.pp("trace_enc"), cfg)?;
        let xim_enc = if let Some(xim_cfg) = &cfg.xim {
            let enc_cfg = Self::encoder_cfg_from_xim(xim_cfg);
            Some(crate::building_blocks::conv_encoder::TraceEncoder::new(
                vb.pp("xim_enc"),
                &enc_cfg,
            )?)
        } else {
            None
        };
        let modality_gate = if cfg.use_modality_gated_fusion && xim_enc.is_some() {
            let xic_in_dim = trace_enc.emb_out_dim() + trace_enc.coelution_dim();
            let xim_in_dim = xim_enc
                .as_ref()
                .map(|enc| enc.emb_out_dim() + enc.coelution_dim())
                .unwrap_or(0);
            Some(ModalityGate::new(
                vb.pp("modality_gate"),
                xic_in_dim + xim_in_dim,
                &cfg.modality_gate_hidden,
                cfg.modality_gate_dropout,
                cfg.modality_gate_mode,
            )?)
        } else {
            None
        };
        let total_emb_dim =
            trace_enc.emb_out_dim() + xim_enc.as_ref().map(|enc| enc.emb_out_dim()).unwrap_or(0);
        let total_coe_dim = trace_enc.coelution_dim()
            + xim_enc.as_ref().map(|enc| enc.coelution_dim()).unwrap_or(0);
        let scorer = crate::building_blocks::mlp::CandidateScorer::new(
            vb.pp("candidate_scorer"),
            cfg,
            total_emb_dim,
            total_coe_dim,
        )?;
        Ok(Self {
            trace_enc,
            xim_enc,
            xim_cfg: cfg.xim.clone(),
            modality_gate,
            scorer,
        })
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
        self.forward_bags_aux(xb, tb, mask, None)
    }

    /// Score a batch of bags with an optional aligned XIM tensor.
    pub fn forward_bags_aux(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
        tb_aux: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, k, d) = xb.dims3()?;
        let (_, _, c, l) = tb.dims4()?;

        let xf = xb.reshape((b * k, d))?;
        let tf = tb.reshape((b * k, c, l))?;
        let tf_aux = if let Some(tb_aux) = tb_aux {
            let (_, _, c_aux, l_aux) = tb_aux.dims4()?;
            Some(tb_aux.reshape((b * k, c_aux, l_aux))?)
        } else {
            None
        };

        let (emb, coe, _coe_ms12) = self.encode_inputs(&tf, tf_aux.as_ref())?;
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
        self.forward_bags_with_hidden_aux(xb, tb, mask, None)
    }

    /// Score bags and return winner hidden vectors with an optional aligned
    /// XIM tensor.
    pub fn forward_bags_with_hidden_aux(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
        tb_aux: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, k, d) = xb.dims3()?;
        let (_, _, c, l) = tb.dims4()?;

        let xf = xb.reshape((b * k, d))?;
        let tf = tb.reshape((b * k, c, l))?;
        let tf_aux = if let Some(tb_aux) = tb_aux {
            let (_, _, c_aux, l_aux) = tb_aux.dims4()?;
            Some(tb_aux.reshape((b * k, c_aux, l_aux))?)
        } else {
            None
        };

        let (emb, coe, _coe_ms12) = self.encode_inputs(&tf, tf_aux.as_ref())?;
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
        self.forward_bags_with_heads_aux(xb, tb, mask, None)
    }

    /// Score bags, export winner-side components, and optionally fuse an
    /// aligned XIM tensor into the candidate representation.
    pub fn forward_bags_with_heads_aux(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
        tb_aux: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, BagHeadComponents)> {
        let (b, k, d) = xb.dims3()?;
        let (_, _, c, l) = tb.dims4()?;

        let xf = xb.reshape((b * k, d))?;
        let tf = tb.reshape((b * k, c, l))?;

        let (mut emb_all, mut coe_all, mut comps) = self.trace_enc.forward_with_heads(&tf)?;
        if let Some(xim_enc) = &self.xim_enc {
            let tf_aux = if let Some(tb_aux) = tb_aux {
                let (_, _, c_aux, l_aux) = tb_aux.dims4()?;
                tb_aux.reshape((b * k, c_aux, l_aux))?
            } else {
                self.zero_xim_input(b * k, tf.device(), tf.dtype())?
            };
            let (xim_emb_all, xim_coe_all, _xim_comps) = xim_enc.forward_with_heads(&tf_aux)?;
            let (fused_emb, fused_coe) = self.fuse_modalities(
                &emb_all,
                &coe_all,
                Some(&xim_emb_all),
                Some(&xim_coe_all),
                false,
            )?;
            emb_all = fused_emb;
            coe_all = fused_coe;
            comps.emb_all = emb_all.clone();
            comps.coe_all = coe_all.clone();
        }
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
    fn forward_candidates_aux(
        &self,
        x_feat: &Tensor,
        x_trace: &Tensor,
        x_aux: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (emb, coe, _coe_ms12) = self.encode_inputs(x_trace, x_aux)?;
        self.scorer.forward_eval(x_feat, &emb, &coe)
    }
}

impl BagRankerInterface for TopazBagRanker {
    fn forward_bags_aux(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
        tb_aux: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        TopazBagRanker::forward_bags_aux(self, xb, tb, mask, tb_aux)
    }
}

impl BagRankerWithHiddenInterface for TopazBagRanker {
    fn forward_bags_with_hidden_aux(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
        tb_aux: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        TopazBagRanker::forward_bags_with_hidden_aux(self, xb, tb, mask, tb_aux)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use serde_json::json;

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

    #[test]
    fn test_forward_bags_with_xim_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TopazConfig {
            feat_dim: 3,
            ms2_cmax: 2,
            ms1_cmax: 0,
            l: 8,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            xim: Some(TopazXimConfig {
                ms2_cmax: 3,
                ms1_cmax: 1,
                l: 12,
                trace_emb_dim: 6,
                trace_input_mode: TraceInputMode::Single,
                use_coelution_head: true,
                ..Default::default()
            }),
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        let (b, k, d) = (2usize, 2usize, cfg.feat_dim);
        let xb = Tensor::zeros((b, k, d), DType::F32, &device)?;
        let tb = Tensor::zeros((b, k, cfg.ms2_cmax, cfg.l), DType::F32, &device)?;
        let xim = Tensor::zeros((b, k, 4usize, 12usize), DType::F32, &device)?;
        let mask = Tensor::ones((b, k), DType::U8, &device)?;

        let (cand, bag, win) = model.forward_bags_with_hidden_aux(&xb, &tb, &mask, Some(&xim))?;
        assert_eq!(cand.dims2()?, (b, k));
        assert_eq!(bag.dims1()?, b);
        assert_eq!(win.dims2()?.0, b);
        Ok(())
    }

    #[test]
    fn test_forward_bags_with_transition_interaction_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TopazConfig {
            feat_dim: 4,
            ms2_cmax: 3,
            ms1_cmax: 2,
            l: 10,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Dual,
            use_heuristic_features: true,
            use_coelution_head: true,
            use_transition_interaction: true,
            transition_interaction_dim: 6,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        let (b, k, d) = (2usize, 3usize, cfg.feat_dim);
        let xb = Tensor::zeros((b, k, d), DType::F32, &device)?;
        let tb = Tensor::zeros(
            (b, k, cfg.ms1_cmax + cfg.ms2_cmax, cfg.l),
            DType::F32,
            &device,
        )?;
        let mask = Tensor::ones((b, k), DType::U8, &device)?;

        let (cand, bag, win) = model.forward_bags_with_hidden(&xb, &tb, &mask)?;
        assert_eq!(cand.dims2()?, (b, k));
        assert_eq!(bag.dims1()?, b);
        assert_eq!(win.dims2()?.0, b);
        Ok(())
    }

    #[test]
    fn test_forward_bags_with_modality_gated_fusion_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TopazConfig {
            feat_dim: 4,
            ms2_cmax: 3,
            ms1_cmax: 2,
            l: 10,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Dual,
            use_heuristic_features: true,
            use_coelution_head: true,
            use_transition_interaction: true,
            transition_interaction_dim: 6,
            use_modality_gated_fusion: true,
            modality_gate_hidden: vec![8],
            modality_gate_mode: ModalityGateMode::Competitive,
            xim: Some(TopazXimConfig {
                ms2_cmax: 4,
                ms1_cmax: 2,
                l: 12,
                trace_emb_dim: 6,
                trace_input_mode: TraceInputMode::Dual,
                use_coelution_head: true,
                use_transition_interaction: true,
                transition_interaction_dim: 4,
                ..Default::default()
            }),
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        let (b, k, d) = (2usize, 3usize, cfg.feat_dim);
        let xb = Tensor::zeros((b, k, d), DType::F32, &device)?;
        let tb = Tensor::zeros(
            (b, k, cfg.ms1_cmax + cfg.ms2_cmax, cfg.l),
            DType::F32,
            &device,
        )?;
        let xim_cfg = cfg.xim.as_ref().expect("xim branch should be present");
        let xim = Tensor::zeros(
            (b, k, xim_cfg.ms1_cmax + xim_cfg.ms2_cmax, xim_cfg.l),
            DType::F32,
            &device,
        )?;
        let mask = Tensor::ones((b, k), DType::U8, &device)?;

        let (cand, bag, win) = model.forward_bags_with_hidden_aux(&xb, &tb, &mask, Some(&xim))?;
        assert_eq!(cand.dims2()?, (b, k));
        assert_eq!(bag.dims1()?, b);
        assert_eq!(win.dims2()?.0, b);
        Ok(())
    }

    #[test]
    fn test_forward_bags_with_residual_xim_gate_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TopazConfig {
            feat_dim: 4,
            ms2_cmax: 3,
            ms1_cmax: 2,
            l: 10,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Dual,
            use_heuristic_features: true,
            use_coelution_head: true,
            use_transition_interaction: true,
            transition_interaction_dim: 6,
            use_modality_gated_fusion: true,
            modality_gate_hidden: vec![8],
            modality_gate_mode: ModalityGateMode::ResidualXim,
            xim: Some(TopazXimConfig {
                ms2_cmax: 4,
                ms1_cmax: 2,
                l: 12,
                trace_emb_dim: 6,
                trace_input_mode: TraceInputMode::Dual,
                use_coelution_head: true,
                use_transition_interaction: true,
                transition_interaction_dim: 4,
                ..Default::default()
            }),
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        let (b, k, d) = (2usize, 3usize, cfg.feat_dim);
        let xb = Tensor::zeros((b, k, d), DType::F32, &device)?;
        let tb = Tensor::zeros(
            (b, k, cfg.ms1_cmax + cfg.ms2_cmax, cfg.l),
            DType::F32,
            &device,
        )?;
        let xim_cfg = cfg.xim.as_ref().expect("xim branch should be present");
        let xim = Tensor::zeros(
            (b, k, xim_cfg.ms1_cmax + xim_cfg.ms2_cmax, xim_cfg.l),
            DType::F32,
            &device,
        )?;
        let mask = Tensor::ones((b, k), DType::U8, &device)?;

        let (cand, bag, win) = model.forward_bags_with_hidden_aux(&xb, &tb, &mask, Some(&xim))?;
        assert_eq!(cand.dims2()?, (b, k));
        assert_eq!(bag.dims1()?, b);
        assert_eq!(win.dims2()?.0, b);
        Ok(())
    }

    #[test]
    fn test_xim_config_deserializes_legacy_field_names() {
        let cfg: TopazConfig = serde_json::from_value(json!({
            "feat_dim": 7,
            "use_modality_gated_fusion": true,
            "modality_gate_hidden": [32],
            "modality_gate_dropout": 0.1,
            "modality_gate_mode": "residual_xim",
            "xim": {
                "ms2_cmax": 6,
                "ms1_cmax": 4,
                "l": 100,
                "trace_emb_dim": 32,
                "trace_input_mode": "Dual",
                "use_coelution": true,
                "coelution_emb_dim": 16,
                "coelution_max_lag": 3
            }
        }))
        .expect("legacy xim config should deserialize");

        let xim = cfg.xim.expect("xim branch should be present");
        assert!(xim.use_coelution_head);
        assert_eq!(xim.coelution_sim_emb_dim, 16);
        assert_eq!(xim.trace_input_mode, TraceInputMode::Dual);
        assert_eq!(xim.coelution_beta, TopazXimConfig::default().coelution_beta);
        assert!(cfg.use_modality_gated_fusion);
        assert_eq!(cfg.modality_gate_hidden, vec![32]);
        assert_eq!(cfg.modality_gate_dropout, 0.1);
        assert_eq!(cfg.modality_gate_mode, ModalityGateMode::ResidualXim);
    }
}
