// redeem-topaz/src/model_interface.rs

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

pub trait ModelInterface: Sized {
    type Config: Clone + Send + Sync;
    type Input;
    type Output;

    fn new(vb: VarBuilder, cfg: &Self::Config) -> Result<Self>;
    fn forward(&self, input: &Self::Input) -> Result<Self::Output>;
}

/// A convenience trait for models that score candidates (N,) from (features + traces).
pub trait CandidateScorerInterface {
    fn forward_candidates(&self, x_feat: &Tensor, x_trace: &Tensor) -> Result<Tensor>; // (N,)

    /// Default chunked candidate scoring (shared by all models that implement
    /// `forward_candidates`).
    fn score_candidates_chunked(
        &self,
        x_feat: &Tensor,  // (N,D)
        x_trace: &Tensor, // (N,C,L)
        batch_size: usize,
    ) -> Result<Tensor> {
        let (n, _d) = x_feat.dims2()?;
        let mut out: Vec<Tensor> = Vec::new();
        let bs = batch_size.max(1);

        let mut i = 0usize;
        while i < n {
            let take = (n - i).min(bs);
            let xf = x_feat.narrow(0, i, take)?;
            let tf = x_trace.narrow(0, i, take)?;
            let logits = self.forward_candidates(&xf, &tf)?;
            out.push(logits);
            i += take;
        }

        if out.is_empty() {
            Tensor::zeros((0usize,), DType::F32, x_feat.device())
        } else {
            Tensor::cat(&out, 0)
        }
    }
}

/// Interface for bag-level models (B,K,...) -> (candidate logits, bag logits).
pub trait BagRankerInterface {
    fn forward_bags(&self, xb: &Tensor, tb: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)>;

    /// Default chunked bag scoring.
    fn score_bags_chunked(
        &self,
        xb: &Tensor,   // (B,K,D)
        tb: &Tensor,   // (B,K,C,L)
        mask: &Tensor, // (B,K)
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b, k, _d) = xb.dims3()?;
        let mut cand_chunks: Vec<Tensor> = Vec::new();
        let mut bag_chunks: Vec<Tensor> = Vec::new();
        let bs = batch_size.max(1);

        let mut i = 0usize;
        while i < b {
            let take = (b - i).min(bs);
            let xb_i = xb.narrow(0, i, take)?;
            let tb_i = tb.narrow(0, i, take)?;
            let m_i = mask.narrow(0, i, take)?;
            let (cand, bag) = self.forward_bags(&xb_i, &tb_i, &m_i)?;
            cand_chunks.push(cand);
            bag_chunks.push(bag);
            i += take;
        }

        let cand = if cand_chunks.is_empty() {
            Tensor::zeros((0usize, k), DType::F32, xb.device())?
        } else {
            Tensor::cat(&cand_chunks, 0)?
        };
        let bag = if bag_chunks.is_empty() {
            Tensor::zeros((0usize,), DType::F32, xb.device())?
        } else {
            Tensor::cat(&bag_chunks, 0)?
        };

        Ok((cand, bag))
    }
}

/// Bag-level models that also expose per-bag winner hidden.
pub trait BagRankerWithHiddenInterface: BagRankerInterface {
    fn forward_bags_with_hidden(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)>;

    /// Default chunked bag scoring with winner hidden.
    fn score_bags_with_hidden_chunked(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, k, _d) = xb.dims3()?;
        let mut cand_chunks: Vec<Tensor> = Vec::new();
        let mut bag_chunks: Vec<Tensor> = Vec::new();
        let mut hid_chunks: Vec<Tensor> = Vec::new();
        let bs = batch_size.max(1);

        let mut i = 0usize;
        while i < b {
            let take = (b - i).min(bs);
            let xb_i = xb.narrow(0, i, take)?;
            let tb_i = tb.narrow(0, i, take)?;
            let m_i = mask.narrow(0, i, take)?;
            let (cand, bag, hid) = self.forward_bags_with_hidden(&xb_i, &tb_i, &m_i)?;
            cand_chunks.push(cand);
            bag_chunks.push(bag);
            hid_chunks.push(hid);
            i += take;
        }

        let cand = if cand_chunks.is_empty() {
            Tensor::zeros((0usize, k), DType::F32, xb.device())?
        } else {
            Tensor::cat(&cand_chunks, 0)?
        };
        let bag = if bag_chunks.is_empty() {
            Tensor::zeros((0usize,), DType::F32, xb.device())?
        } else {
            Tensor::cat(&bag_chunks, 0)?
        };
        let hid = if hid_chunks.is_empty() {
            Tensor::zeros((0usize, 0usize), DType::F32, xb.device())?
        } else {
            Tensor::cat(&hid_chunks, 0)?
        };

        Ok((cand, bag, hid))
    }
}
