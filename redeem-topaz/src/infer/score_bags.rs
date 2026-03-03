use candle_core::{DType, Result, Tensor};

use crate::model::topaz::TopazBagRanker;

/// Score bags in chunks along B.
pub fn score_bags(
    model: &TopazBagRanker,
    xb: &Tensor,   // (B,K,D)
    tb: &Tensor,   // (B,K,C,L)
    mask: &Tensor, // (B,K)
    batch_size: usize,
) -> Result<(Tensor, Tensor)> {
    let (b, _k, _d) = xb.dims3()?;
    let mut cand_chunks: Vec<Tensor> = Vec::new();
    let mut bag_chunks: Vec<Tensor> = Vec::new();
    let bs = batch_size.max(1);

    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = xb.narrow(0, i, take)?;
        let tb_i = tb.narrow(0, i, take)?;
        let m_i = mask.narrow(0, i, take)?;
        let (cand, bag) = model.forward_bags(&xb_i, &tb_i, &m_i)?;
        cand_chunks.push(cand);
        bag_chunks.push(bag);
        i += take;
    }

    let cand = if cand_chunks.is_empty() {
        Tensor::zeros((0usize, 0usize), DType::F32, xb.device())?
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

pub struct ScoreBags;
