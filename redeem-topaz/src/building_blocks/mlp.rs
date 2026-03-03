// redeem-topaz/src/building_blocks/mlp.rs

use candle_core::{Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

use crate::model::topaz::TopazConfig;

#[derive(Clone, Debug)]
struct DropoutAlways(nn::Dropout);

impl Module for DropoutAlways {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs, true)
    }
}

pub struct CandidateScorer {
    use_features: bool,
    feat_dim_used: usize,
    body: nn::Sequential,
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

        let mut seq = nn::seq();
        let mut d = in_dim;
        for (i, &h) in cfg.mlp_hidden.iter().enumerate() {
            let lin = nn::linear(d, h, vb.pp(format!("lin{i}")))?;
            seq = seq.add(lin).add(nn::Activation::Relu);
            if cfg.dropout > 0.0 {
                seq = seq.add(DropoutAlways(nn::Dropout::new(cfg.dropout as f32)));
            }
            d = h;
        }
        let head = nn::linear(d, 1, vb.pp("head"))?;

        Ok(Self { use_features, feat_dim_used, body: seq, head, hidden_dim: d })
    }

    pub fn penultimate(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        let x = self.concat_input(feat, emb, coe)?;
        x.apply(&self.body)
    }

    pub fn forward(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<Tensor> {
        let h = self.penultimate(feat, emb, coe)?;
        h.apply(&self.head)?.squeeze(1)
    }

    /// Return (logits, hidden) where hidden is the penultimate layer.
    pub fn forward_with_hidden(&self, feat: &Tensor, emb: &Tensor, coe: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.penultimate(feat, emb, coe)?;
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
