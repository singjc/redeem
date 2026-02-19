//! Statistical helpers used across the crate.
//!
//! These utilities include target-decoy competition (TDC) based q-value estimation and
//! other small helpers used by `data_handling` and feature-selection routines.
use std::cmp::Ordering;

use crate::math::Array1;

/// Estimate q-values using target-decoy competition.
///
/// This function implements the simple target-decoy competition method to estimate q-values.
/// For a set of target and decoy PSMs meeting a specified score threshold, the false discovery
/// rate (FDR) is estimated as:
///
/// FDR = (Decoys + 1) / Targets
///
/// Adpated from: https://github.com/wfondrie/mokapot/blob/main/mokapot/qvalues.py#L28
///
/// # Arguments
///
/// * `scores` - A 1D array containing the scores to rank by.
/// * `target` - A 1D boolean array indicating if the entry is from a target (true) or decoy (false) hit.
/// * `desc` - A boolean indicating if higher scores are better (true) or if lower scores are better (false).
///
/// # Returns
///
/// A 1D array with the estimated q-value for each entry. The array is the same length as the `scores` and `target` arrays.
pub fn tdc(scores: &Array1<f32>, target: &Array1<bool>, desc: bool) -> Array1<f32> {
    assert_eq!(
        scores.len(),
        target.len(),
        "scores and target must have equal lengths"
    );

    let mut sorted_indices: Vec<usize> = (0..scores.len()).collect();
    sorted_indices.sort_unstable_by(|&a, &b| {
        let ordering = scores[a].partial_cmp(&scores[b]).unwrap_or(Ordering::Equal);
        if desc {
            ordering.reverse()
        } else {
            ordering
        }
    });

    let mut cum_targets = Vec::with_capacity(scores.len());
    let mut cum_decoys = Vec::with_capacity(scores.len());
    let mut t_count = 0usize;
    let mut d_count = 0usize;
    for &idx in &sorted_indices {
        if target[idx] {
            t_count += 1;
        } else {
            d_count += 1;
        }
        cum_targets.push(t_count);
        cum_decoys.push(d_count);
    }

    let mut fdr = Vec::with_capacity(scores.len());
    for i in 0..scores.len() {
        let decoys = (cum_decoys[i] + 1) as f32;
        let targets = cum_targets[i].max(1) as f32;
        fdr.push(decoys / targets);
    }

    let mut qvals_sorted = vec![0f32; scores.len()];
    let mut min_q = f32::INFINITY;
    for i in (0..fdr.len()).rev() {
        if fdr[i] < min_q {
            min_q = fdr[i];
        }
        qvals_sorted[i] = min_q;
    }

    let mut final_qvals = vec![0f32; scores.len()];
    for (sorted_idx, &original_idx) in sorted_indices.iter().enumerate() {
        final_qvals[original_idx] = qvals_sorted[sorted_idx];
    }

    Array1::from_vec(final_qvals)
}
