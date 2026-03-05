// redeem-topaz/src/building_blocks/mlp.rs

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

pub struct CandidateScorer {
    use_features: bool,
    feat_dim_used: usize,
    layers: Vec<LayerBlock>,
    head: nn::Linear,
    hidden_dim: usize,
}

impl CandidateScorer {
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

        Ok(Self { use_features, feat_dim_used, layers, head, hidden_dim: d })
    }

    pub fn penultimate(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        self.penultimate_internal(feat, emb, coe, true)
    }

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

    pub fn forward(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        let h = self.penultimate(feat, emb, coe)?;
        h.apply(&self.head)?.squeeze(1)
    }

    pub fn forward_eval(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        let h = self.penultimate_eval(feat, emb, coe)?;
        h.apply(&self.head)?.squeeze(1)
    }

    /// Return (logits, hidden) where hidden is the penultimate layer.
    pub fn forward_with_hidden(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.penultimate(feat, emb, coe)?;
        let logits = h.apply(&self.head)?.squeeze(1)?;
        Ok((logits, h))
    }

    /// Return (logits, hidden) with dropout disabled.
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
