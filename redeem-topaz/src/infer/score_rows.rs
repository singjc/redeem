//! Chunked candidate-level scoring helpers.
//!
//! This module provides a thin API boundary around the model-interface trait
//! used for candidate scoring. Keeping this wrapper separate makes it easier to
//! share the same chunked scoring path across training diagnostics, offline
//! inference, and future CLI/report tooling without coupling those call sites
//! to a specific model implementation.

use candle_core::{Result, Tensor};

use crate::model_interface::CandidateScorerInterface;

/// Score a candidate matrix in chunks and return one logit per input row.
///
/// # Inputs
/// - `model`: model implementing [`CandidateScorerInterface`].
/// - `x_feat`: heuristic feature tensor with shape `(N, D)`.
/// - `x_trace`: trace tensor with shape `(N, C, L)`.
/// - `batch_size`: maximum number of rows to score per chunk.
///
/// # Output
/// Returns a tensor of shape `(N,)` containing the raw candidate logits before
/// any bag-level aggregation or calibration.
pub fn score_candidates(
    model: &impl CandidateScorerInterface,
    x_feat: &Tensor,  // (N,D)
    x_trace: &Tensor, // (N,C,L)
    batch_size: usize,
) -> Result<Tensor> {
    model.score_candidates_chunked(x_feat, x_trace, batch_size)
}

/// Namespace marker for row-scoring utilities.
///
/// The current implementation exposes only free functions, but this marker type
/// keeps the public surface aligned with the rest of the inference modules and
/// leaves room for future method-based extensions.
pub struct ScoreRows;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{score_bags, tdc_qvalues};
    use crate::model::topaz::{TopazBagRanker, TopazConfig};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_infer_smoke_candidates_bags_qvalues() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TopazConfig {
            feat_dim: 3,
            ms2_cmax: 4,
            ms1_cmax: 0,
            l: 8,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: crate::building_blocks::trace_input::TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            ..Default::default()
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        // Candidate scoring
        let n = 6usize;
        let x_feat = Tensor::zeros((n, cfg.feat_dim), DType::F32, &device)?;
        let x_trace = Tensor::zeros((n, cfg.ms2_cmax, cfg.l), DType::F32, &device)?;
        let cand = score_candidates(&model, &x_feat, &x_trace, 4)?;
        assert_eq!(cand.dims1()?, n);

        // Bag scoring
        let (b, k) = (2usize, 3usize);
        let xb = Tensor::zeros((b, k, cfg.feat_dim), DType::F32, &device)?;
        let tb = Tensor::zeros((b, k, cfg.ms2_cmax, cfg.l), DType::F32, &device)?;
        let mask = Tensor::ones((b, k), DType::U8, &device)?;
        let (_cand_b, bag) = score_bags(&model, &xb, &tb, &mask, 1)?;
        assert_eq!(bag.dims1()?, b);

        // Q-values
        let bag_scores = bag.to_vec1::<f32>()?;
        let q = tdc_qvalues(&bag_scores, &[false, true]);
        assert_eq!(q.len(), b);
        for v in q {
            assert!((0.0..=1.0).contains(&v));
        }
        Ok(())
    }
}
