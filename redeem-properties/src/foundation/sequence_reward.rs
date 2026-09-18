//! Sequence-level reward utilities for spectrum-conditioned peptide post-training.
//!
//! v0.29 intentionally avoids a learned reward model. Rewards are deterministic
//! combinations of target-sequence agreement and physical spectrum/mass evidence,
//! while optimization uses group-relative advantages over candidates generated
//! for the same observed spectrum.

use candle_core::{DType, Result, Tensor};

/// Stable v0.29 reward identifier.
pub const FOUNDATION_SEQUENCE_REWARD_OBJECTIVE_V0290: &str =
    "group_relative_sequence_reward_mass_fragment_reference_anchored_v1";
/// Weight of sequence agreement in the scalar reward.
pub const FOUNDATION_SEQUENCE_REWARD_SEQUENCE_WEIGHT_V0290: f64 = 0.55;
/// Weight of fragment/spectrum evidence in the scalar reward.
pub const FOUNDATION_SEQUENCE_REWARD_FRAGMENT_WEIGHT_V0290: f64 = 0.30;
/// Weight of exact precursor-mass closure in the scalar reward.
pub const FOUNDATION_SEQUENCE_REWARD_MASS_WEIGHT_V0290: f64 = 0.15;
/// Small stabilizer for group-relative standardization.
pub const FOUNDATION_SEQUENCE_REWARD_EPS_V0290: f64 = 1.0e-6;

/// One deterministic candidate reward audit row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoundationSequenceRewardV0290 {
    pub total: f64,
    pub sequence: f64,
    pub fragment: f64,
    pub mass: f64,
    pub literal_match: bool,
    pub il_match: bool,
    pub sequence_similarity: f64,
}

/// Score one peptide candidate against the known training target plus label-free
/// physical evidence extracted from the same observed spectrum.
///
/// `fragment_evidence` is expected in `[0,1]` and should be computed without
/// consulting the target peptide identity. `mass_error_da` is the signed neutral
/// precursor-mass error returned by the hard-mass beam search.
pub fn foundation_sequence_reward_v0290(
    target_sequence: &str,
    candidate_sequence: &str,
    fragment_evidence: f64,
    mass_error_da: f64,
    mass_tolerance_da: f64,
) -> FoundationSequenceRewardV0290 {
    let literal_match = candidate_sequence == target_sequence;
    let il_match = il_normalize(candidate_sequence) == il_normalize(target_sequence);
    let sequence_similarity = normalized_edit_similarity(target_sequence, candidate_sequence);
    let sequence = if literal_match {
        1.0
    } else if il_match {
        0.90
    } else {
        0.70 * sequence_similarity
    };
    let fragment = fragment_evidence.clamp(0.0, 1.0);
    let mass = if mass_tolerance_da > 0.0 && mass_tolerance_da.is_finite() {
        (-mass_error_da.abs() / mass_tolerance_da)
            .exp()
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let total = FOUNDATION_SEQUENCE_REWARD_SEQUENCE_WEIGHT_V0290 * sequence
        + FOUNDATION_SEQUENCE_REWARD_FRAGMENT_WEIGHT_V0290 * fragment
        + FOUNDATION_SEQUENCE_REWARD_MASS_WEIGHT_V0290 * mass;
    FoundationSequenceRewardV0290 {
        total,
        sequence,
        fragment,
        mass,
        literal_match,
        il_match,
        sequence_similarity,
    }
}

/// Standardize rewards within one same-spectrum candidate group.
pub fn foundation_group_relative_advantages_v0290(rewards: &[f64]) -> Vec<f32> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let variance = rewards
        .iter()
        .map(|reward| {
            let delta = reward - mean;
            delta * delta
        })
        .sum::<f64>()
        / rewards.len() as f64;
    let scale = variance.sqrt().max(FOUNDATION_SEQUENCE_REWARD_EPS_V0290);
    rewards
        .iter()
        .map(|reward| ((reward - mean) / scale) as f32)
        .collect()
}

/// GRPO-like policy loss over complete candidate mean-NLL values.
///
/// Minimizing `advantage * NLL` lowers NLL for above-group candidates and raises
/// it for below-group candidates. The reward/advantage tensor is detached CPU
/// metadata by construction; gradients flow only through `candidate_mean_nlls`.
pub fn foundation_group_relative_policy_loss_v0290(
    candidate_mean_nlls: &Tensor,
    advantages: &[f32],
) -> Result<Tensor> {
    let rows = candidate_mean_nlls.dims1()?;
    if rows != advantages.len() || rows < 2 {
        candle_core::bail!(
            "v0.29 policy loss requires >=2 candidates and aligned advantages: nlls={rows} rewards={}",
            advantages.len()
        );
    }
    let advantage = Tensor::from_vec(advantages.to_vec(), rows, candidate_mean_nlls.device())?
        .to_dtype(DType::F32)?;
    candidate_mean_nlls.broadcast_mul(&advantage)?.mean_all()
}

/// Stable reference-policy anchor over candidate mean-NLL values.
///
/// This is deliberately a simple squared log-likelihood drift penalty rather
/// than a separately trained reward/value model. The frozen reference tensor is
/// detached so only the current policy receives gradients.
pub fn foundation_reference_nll_anchor_v0290(
    current_mean_nlls: &Tensor,
    reference_mean_nlls: &Tensor,
) -> Result<Tensor> {
    if current_mean_nlls.dims() != reference_mean_nlls.dims() {
        candle_core::bail!(
            "v0.29 reference anchor shape mismatch: current {:?}, reference {:?}",
            current_mean_nlls.dims(),
            reference_mean_nlls.dims()
        );
    }
    (current_mean_nlls - &reference_mean_nlls.detach())?
        .sqr()?
        .mean_all()
}

fn il_normalize(sequence: &str) -> String {
    sequence
        .chars()
        .map(|residue| if residue == 'I' { 'L' } else { residue })
        .collect()
}

fn normalized_edit_similarity(left: &str, right: &str) -> f64 {
    let a = left.chars().collect::<Vec<_>>();
    let b = right.chars().collect::<Vec<_>>();
    let denom = a.len().max(b.len()).max(1) as f64;
    1.0 - levenshtein(&a, &b) as f64 / denom
}

fn levenshtein(left: &[char], right: &[char]) -> usize {
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0usize; right.len() + 1];
    for (i, &l) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, &r) in right.iter().enumerate() {
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + usize::from(l != r));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn exact_sequence_gets_higher_reward_than_distant_mass_matched_sequence() {
        let exact = foundation_sequence_reward_v0290("PEPTIDE", "PEPTIDE", 0.5, 0.0, 0.02);
        let wrong = foundation_sequence_reward_v0290("PEPTIDE", "AAAAAAA", 0.5, 0.0, 0.02);
        assert!(exact.total > wrong.total);
        assert!(exact.literal_match);
        assert!(!wrong.literal_match);
    }

    #[test]
    fn il_equivalence_is_rewarded_without_being_literal() {
        let reward = foundation_sequence_reward_v0290("AILK", "ALLK", 0.5, 0.0, 0.02);
        assert!(!reward.literal_match);
        assert!(reward.il_match);
        assert_eq!(reward.sequence, 0.90);
    }

    #[test]
    fn group_advantages_are_centered_and_policy_loss_is_finite() {
        let advantages = foundation_group_relative_advantages_v0290(&[0.1, 0.5, 0.9]);
        let mean = advantages.iter().map(|&v| f64::from(v)).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1.0e-6);
        let nll = Tensor::from_vec(vec![2.0f32, 1.5, 1.0], 3, &Device::Cpu).unwrap();
        let loss = foundation_group_relative_policy_loss_v0290(&nll, &advantages).unwrap();
        assert!(loss.to_scalar::<f32>().unwrap().is_finite());
    }
}
