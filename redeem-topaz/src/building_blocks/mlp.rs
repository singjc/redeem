//! Candidate-level MLP scorer used on top of trace embeddings.
//!
//! Shape notation used in this module:
//!
//! - `N`: number of candidate rows.
//! - `D`: scalar heuristic feature dimension.
//! - `E`: learned trace-embedding dimension.
//! - `Coe`: explicit coelution feature dimension.

use candle_core::{Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::model::topaz::TopazConfig;

#[derive(Clone, Debug)]
struct LayerBlock {
    lin: nn::Linear,
    dropout: Option<nn::Dropout>,
}

impl LayerBlock {
    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let h = xs.apply(&self.lin)?.apply(&nn::Activation::Relu)?;
        if let Some(d) = &self.dropout {
            d.forward(&h, train)
        } else {
            Ok(h)
        }
    }
}

/// MLP that fuses heuristic features, learned trace embeddings, and optional
/// coelution features into a single candidate logit.
///
/// The input to this block is a row-wise concatenation of:
/// - heuristic features `(N, D)` when enabled,
/// - learned trace embeddings `(N, E)`,
/// - explicit coelution features `(N, Coe)`.
pub struct CandidateScorer {
    use_features: bool,
    feat_dim_used: usize,
    layers: Vec<LayerBlock>,
    head: nn::Linear,
    hidden_dim: usize,
}

impl CandidateScorer {
    /// Build the candidate scorer for a configured TOPAZ model.
    ///
    /// `trace_emb_dim` and `coelution_dim` are the dimensions returned by the
    /// trace encoder. Together with the optional scalar feature block they
    /// determine the MLP input width.
    pub fn new(
        vb: VarBuilder,
        cfg: &TopazConfig,
        trace_emb_dim: usize,
        coelution_dim: usize,
    ) -> Result<Self> {
        let use_features = cfg.use_heuristic_features;
        let feat_dim_used = if use_features { cfg.feat_dim } else { 0 };
        let in_dim = feat_dim_used + trace_emb_dim + coelution_dim;

        let mut layers = Vec::with_capacity(cfg.mlp_hidden.len());
        let mut d = in_dim;
        for (i, &h) in cfg.mlp_hidden.iter().enumerate() {
            let lin = nn::linear(d, h, vb.pp(format!("lin{i}")))?;
            let dropout = if cfg.dropout > 0.0 {
                Some(nn::Dropout::new(cfg.dropout as f32))
            } else {
                None
            };
            layers.push(LayerBlock { lin, dropout });
            d = h;
        }
        let head = nn::linear(d, 1, vb.pp("head"))?;

        Ok(Self {
            use_features,
            feat_dim_used,
            layers,
            head,
            hidden_dim: d,
        })
    }

    /// Return the penultimate hidden representation in training mode.
    ///
    /// This is the hidden vector immediately before the final scalar logit
    /// layer. It is useful for diagnostics and for XRUN winner-hidden export.
    pub fn penultimate(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        self.penultimate_internal(feat, emb, coe, true)
    }

    /// Return the penultimate hidden representation with dropout disabled.
    pub fn penultimate_eval(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        self.penultimate_internal(feat, emb, coe, false)
    }

    fn penultimate_internal(
        &self,
        feat: &Tensor,
        emb: &Tensor,
        coe: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let mut x = self.concat_input(feat, emb, coe)?;
        for layer in &self.layers {
            x = layer.forward(&x, train)?;
        }
        Ok(x)
    }

    /// Forward pass in training mode.
    ///
    /// # Inputs
    /// - `feat`: `(N, D)` heuristic features.
    /// - `emb`: `(N, E)` learned trace embeddings.
    /// - `coe`: `(N, Coe)` explicit coelution features.
    ///
    /// # Output
    /// Returns `(N,)` logits.
    pub fn forward(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        let h = self.penultimate(feat, emb, coe)?;
        h.apply(&self.head)?.squeeze(1)
    }

    /// Forward pass with dropout disabled.
    pub fn forward_eval(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        let h = self.penultimate_eval(feat, emb, coe)?;
        h.apply(&self.head)?.squeeze(1)
    }

    /// Forward pass returning both the logit and the penultimate hidden vector.
    ///
    /// Returns `(logits, hidden)` where `hidden` has shape `(N, H)`.
    pub fn forward_with_hidden(
        &self,
        feat: &Tensor,
        emb: &Tensor,
        coe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let h = self.penultimate(feat, emb, coe)?;
        let logits = h.apply(&self.head)?.squeeze(1)?;
        Ok((logits, h))
    }

    /// Return (logits, hidden) with dropout disabled.
    /// Evaluation-mode version of [`Self::forward_with_hidden`].
    pub fn forward_with_hidden_eval(
        &self,
        feat: &Tensor,
        emb: &Tensor,
        coe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let h = self.penultimate_eval(feat, emb, coe)?;
        let logits = h.apply(&self.head)?.squeeze(1)?;
        Ok((logits, h))
    }

    /// Hidden dimensionality used by the penultimate representation.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn concat_input(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        if !self.use_features {
            return Tensor::cat(&[emb.clone(), coe.clone()], 1);
        }
        Tensor::cat(&[feat.clone(), emb.clone(), coe.clone()], 1)
    }
}
