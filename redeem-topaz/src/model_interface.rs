//! Shared model traits used by TOPAZ and future ReDeeM models.
//!
//! The goal of these traits is to separate reusable orchestration code
//! (chunked scoring, training loops, calibration pipelines) from any one model
//! implementation. A future model can implement the same interfaces and reuse
//! most of the surrounding infrastructure.
//!
//! Shape notation used here:
//!
//! - `N`: flat candidate count.
//! - `B`: bag count.
//! - `K`: candidate slots per bag.
//! - `D`: heuristic feature dimension.
//! - `C`: trace-channel count for the tensor at hand.
//! - `L`: fixed trace-window length.
//! - `H`: hidden or embedding dimension returned by a model.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

/// Minimal constructor/forward interface for a Candle model.
///
/// This trait is intentionally generic and does not assume anything about TOPAZ
/// specifically. It is the lowest common denominator needed by orchestration
/// code that wants to build a model under a Candle variable scope and execute a
/// forward pass over a model-specific input type.
pub trait ModelInterface: Sized {
    type Config: Clone + Send + Sync;
    type Input;
    type Output;

    /// Build the model under the provided variable-builder subtree.
    fn new(vb: VarBuilder, cfg: &Self::Config) -> Result<Self>;
    /// Execute the model-specific forward pass.
    fn forward(&self, input: &Self::Input) -> Result<Self::Output>;
}

/// Interface for models that score flat candidate rows.
///
/// A candidate row corresponds to one OSW feature row. The caller provides the
/// heuristic feature matrix and the aligned extracted trace windows; the model
/// returns one raw logit per row. No bagging or cross-run calibration happens
/// at this stage.
pub trait CandidateScorerInterface {
    /// Score a flat batch of candidate rows.
    ///
    /// # Inputs
    /// - `x_feat`: `(N, D)` heuristic/library feature matrix, where each row is
    ///   one candidate peak group and `D` is the number of selected scalar
    ///   feature columns.
    /// - `x_trace`: `(N, C, L)` chromatogram tensor, where `C` is the number of
    ///   extracted trace channels for that model configuration and `L` is the
    ///   fixed window length.
    ///
    /// # Output
    /// Returns `(N,)` logits, one per candidate row.
    fn forward_candidates(&self, x_feat: &Tensor, x_trace: &Tensor) -> Result<Tensor>; // (N,)

    /// Default chunked candidate scoring (shared by all models that implement
    /// [`Self::forward_candidates`].
    ///
    /// This is the standard path used by inference code to keep peak memory
    /// bounded while preserving row order exactly.
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

/// Interface for bag-level multiple-instance-learning models.
///
/// A bag usually groups all candidate rows that belong to one precursor in one
/// run. The model returns both candidate logits and a bag logit. In TOPAZ the
/// bag logit is produced by masked max pooling over candidate logits, matching
/// the Python PSTC implementation.
pub trait BagRankerInterface {
    /// Score a batch of bags.
    ///
    /// # Inputs
    /// - `xb`: `(B, K, D)` feature tensor. `B` is the number of bags in the
    ///   mini-batch, `K` is the padded candidate capacity per bag, and `D` is
    ///   the heuristic feature dimension.
    /// - `tb`: `(B, K, C, L)` trace tensor aligned with `xb`.
    /// - `mask`: `(B, K)` validity mask indicating which candidate slots are
    ///   real rows and which are padding.
    ///
    /// # Output
    /// Returns:
    /// - candidate logits `(B, K)`
    /// - bag logits `(B,)`, usually computed by masked max MIL pooling.
    fn forward_bags(&self, xb: &Tensor, tb: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)>;

    /// Default chunked bag scoring.
    ///
    /// This preserves bag order and concatenates the per-chunk results back
    /// into the full `(B, K)` / `(B,)` outputs.
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

/// Extension of [`BagRankerInterface`] for models that expose winner embeddings.
///
/// The "winner hidden" is the hidden representation of the candidate that won
/// the bag-level masked-max pooling step. TOPAZ uses this for diagnostics and
/// for XRUN calibration, where the calibrator consumes `(bag_score,
/// winner_hidden)` sequences across runs.
pub trait BagRankerWithHiddenInterface: BagRankerInterface {
    /// Score bags and return the hidden representation of the winning candidate
    /// in every bag.
    ///
    /// # Output
    /// Returns `(candidate_logits, bag_logits, winner_hidden)` where
    /// `winner_hidden` has shape `(B, H)`.
    fn forward_bags_with_hidden(
        &self,
        xb: &Tensor,
        tb: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)>;

    /// Default chunked bag scoring with winner hidden.
    ///
    /// As with the other chunked helpers, this exists so callers can reuse the
    /// same memory-safe scoring path across different model implementations.
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
