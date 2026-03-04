use candle_core::{Result, Tensor};

use crate::model_interface::BagRankerInterface;

/// Score bags in chunks along B.
pub fn score_bags(
    model: &impl BagRankerInterface,
    xb: &Tensor,   // (B,K,D)
    tb: &Tensor,   // (B,K,C,L)
    mask: &Tensor, // (B,K)
    batch_size: usize,
) -> Result<(Tensor, Tensor)> {
    model.score_bags_chunked(xb, tb, mask, batch_size)
}

pub struct ScoreBags;
