//! Chunked bag-level scoring helpers.
//!
//! These helpers operate on already-bagged tensors and delegate the actual
//! forward pass to the active model implementation via
//! [`BagRankerInterface`]. The returned tensors preserve the Python MIL
//! semantics used by TOPAZ: candidate logits per bag plus a bag logit obtained
//! from masked max pooling over candidates.

use candle_core::{Result, Tensor};

use crate::model_interface::BagRankerInterface;

/// Score a batch of bags in chunks and return candidate- and bag-level logits.
///
/// # Inputs
/// - `model`: model implementing [`BagRankerInterface`].
/// - `xb`: heuristic feature tensor with shape `(B, K, D)`.
/// - `tb`: trace tensor with shape `(B, K, C, L)`.
/// - `mask`: candidate-validity mask with shape `(B, K)`.
/// - `batch_size`: maximum number of bags to score per chunk.
///
/// # Output
/// Returns `(candidate_logits, bag_logits)` where:
/// - `candidate_logits` has shape `(B, K)`
/// - `bag_logits` has shape `(B,)`
pub fn score_bags(
    model: &impl BagRankerInterface,
    xb: &Tensor,   // (B,K,D)
    tb: &Tensor,   // (B,K,C,L)
    mask: &Tensor, // (B,K)
    batch_size: usize,
) -> Result<(Tensor, Tensor)> {
    model.score_bags_chunked(xb, tb, mask, batch_size)
}

/// Score bags in chunks with an optional auxiliary signal tensor aligned to
/// `tb`.
pub fn score_bags_with_aux(
    model: &impl BagRankerInterface,
    xb: &Tensor,
    tb: &Tensor,
    mask: &Tensor,
    tb_aux: Option<&Tensor>,
    batch_size: usize,
) -> Result<(Tensor, Tensor)> {
    model.score_bags_chunked_aux(xb, tb, mask, tb_aux, batch_size)
}

/// Namespace marker for bag-scoring utilities.
pub struct ScoreBags;
