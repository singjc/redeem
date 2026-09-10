//! v0.19 direct spectrum-conditioned autoregressive peptide decoding.
//!
//! The accepted causal backbone already implements the required decoder block:
//! masked peptide self-attention, cross-attention to every contextualized peak
//! (plus one precursor token), and a feed-forward residual block.  This module
//! gives that architecture a direct-generation contract, an explicit
//! matched-versus-shuffled spectrum objective, and precursor-mass-constrained
//! beam search.  It deliberately contains no proposal or scalar compatibility
//! head.

use super::causal::{
    foundation_causal_conditioning_margin_loss, FoundationCausalBatch, FoundationCausalOutput,
    PeptideSpectrumCausalModel,
};
use super::diffusion::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarMap;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Frozen v0.19 primary objective identifier.
pub const FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0190: &str =
    "teacher_forced_token_ce_plus_matched_shuffled_margin_v0190";
/// Weight of the dependence guard in the one coherent v0.19 experiment.
pub const FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190: f64 = 0.25;
/// Required per-token matched-spectrum advantage, in nats.
pub const FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190: f64 = 0.10;

/// Frozen v0.19.1 objective identifier.
///
/// The architecture is unchanged from v0.19.0. The added term mines the
/// current model's strongest search-legal wrong token at each gold prefix and
/// requires the true next token to outrank it by a fixed margin. This directly
/// targets gold-prefix loss from cumulative beam competition without reviving
/// candidate-energy/reranking objectives.
pub const FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0191: &str =
    "teacher_forced_ce_plus_on_policy_prefix_margin_plus_matched_shuffled_guard_v0191";
/// Fixed logit margin between the true next token and the strongest current
/// search-legal wrong token in the one v0.19.1 experiment.
pub const FOUNDATION_DIRECT_PREFIX_MARGIN_V0191: f64 = 0.25;

/// The v0.19 model is the existing, semantically matching causal architecture.
///
/// This is not the rejected compatibility model.  Its output is a token
/// distribution at every autoregressive position and every layer cross-attends
/// to the full contextual spectrum memory.
pub type PeptideSpectrumDirectDecoder = PeptideSpectrumCausalModel;

/// Combine teacher-forced CE with the matched-versus-shuffled dependence guard.
pub fn foundation_direct_conditioning_loss(
    matched_nll: &Tensor,
    shuffled_nll: &Tensor,
    margin: f64,
    weight: f64,
) -> Result<Tensor> {
    if !(weight >= 0.0 && weight.is_finite()) {
        candle_core::bail!("direct-decoder conditioning weight must be finite and non-negative");
    }
    let guard = foundation_causal_conditioning_margin_loss(matched_nll, shuffled_nll, margin)?;
    matched_nll + (guard * weight)?
}

/// Diagnostics accompanying the v0.19.1 search-aligned prefix loss.
#[derive(Debug)]
pub struct DirectPrefixCompetition {
    /// Differentiable hard-competitor hinge loss.
    pub loss: Tensor,
    /// Active next-token positions compared.
    pub positions: usize,
    /// Mean rank of the true token among search-legal local alternatives.
    pub mean_legal_rank: f64,
    /// Fraction of active positions where the true token is locally rank 1.
    pub top1_fraction: f64,
    /// Fraction of active positions where the true token is within rank 5.
    pub top5_fraction: f64,
    /// Fraction of active positions where the true token is within rank 10.
    pub top10_fraction: f64,
}

/// Mine the model's strongest legal wrong next token at every gold prefix and
/// require the true token to beat it by `margin` logits.
///
/// Hard-negative token identities are selected from detached current logits,
/// then the selected true/wrong logits are gathered again from the live tensor
/// so gradients flow through both sides of the hinge. The prefix itself remains
/// the clean autoregressive target prefix; the *competitor* is on-policy because
/// it is re-mined from the current model at every optimization step.
pub fn foundation_direct_prefix_competitive_loss(
    output: &FoundationCausalOutput,
    batch: &FoundationCausalBatch,
    margin: f64,
) -> Result<DirectPrefixCompetition> {
    if !(margin >= 0.0 && margin.is_finite()) {
        candle_core::bail!("direct prefix margin must be finite and non-negative");
    }
    let (b, l, classes) = output.token_logits.dims3()?;
    if classes != FOUNDATION_DIFFUSION_VOCAB_SIZE {
        candle_core::bail!(
            "direct prefix competition saw {classes} classes, expected {}",
            FOUNDATION_DIFFUSION_VOCAB_SIZE
        );
    }
    let flat_logits = output.token_logits.reshape((b * l, classes))?;
    let selected_logits = flat_logits.index_select(&batch.active_indices, 0)?;
    let selected_rows = selected_logits.to_vec2::<f32>()?;
    let target_classes = batch.target_classes.to_vec1::<u32>()?;
    let target_rows = batch.target_tokens.to_vec2::<u32>()?;

    let mut prefixes = Vec::<Vec<u32>>::with_capacity(target_classes.len());
    let mut reconstructed_targets = Vec::<u32>::with_capacity(target_classes.len());
    for row in &target_rows {
        let active = row
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
            .unwrap_or(row.len());
        for position in 0..active {
            prefixes.push(row[..position].to_vec());
            reconstructed_targets.push(row[position]);
        }
    }
    if reconstructed_targets != target_classes
        || selected_rows.len() != target_classes.len()
        || prefixes.len() != target_classes.len()
    {
        candle_core::bail!(
            "direct prefix competition active-position contract mismatch: logits={} targets={} prefixes={}",
            selected_rows.len(),
            target_classes.len(),
            prefixes.len()
        );
    }

    let (hard_tokens, mean_rank, top1, top5, top10) =
        select_prefix_hard_competitors(&selected_rows, &prefixes, &target_classes)
            .map_err(candle_core::Error::Msg)?;
    let positions = target_classes.len();
    if positions == 0 {
        candle_core::bail!("direct prefix competition requires active target positions");
    }

    let true_flat_indices: Vec<u32> = target_classes
        .iter()
        .enumerate()
        .map(|(row, &token)| (row * classes + token as usize) as u32)
        .collect();
    let hard_flat_indices: Vec<u32> = hard_tokens
        .iter()
        .enumerate()
        .map(|(row, &token)| (row * classes + token as usize) as u32)
        .collect();
    let device = selected_logits.device();
    let true_indices =
        Tensor::from_vec(true_flat_indices, positions, device)?.to_dtype(DType::U32)?;
    let hard_indices =
        Tensor::from_vec(hard_flat_indices, positions, device)?.to_dtype(DType::U32)?;
    let flat_selected = selected_logits.flatten_all()?;
    let true_logits = flat_selected.index_select(&true_indices, 0)?;
    let hard_logits = flat_selected.index_select(&hard_indices, 0)?;
    let loss = (hard_logits - true_logits)?
        .affine(1.0, margin)?
        .relu()?
        .mean_all()?;

    Ok(DirectPrefixCompetition {
        loss,
        positions,
        mean_legal_rank: mean_rank,
        top1_fraction: top1,
        top5_fraction: top5,
        top10_fraction: top10,
    })
}

fn select_prefix_hard_competitors(
    logits: &[Vec<f32>],
    prefixes: &[Vec<u32>],
    targets: &[u32],
) -> std::result::Result<(Vec<u32>, f64, f64, f64, f64), String> {
    if logits.len() != prefixes.len() || logits.len() != targets.len() || logits.is_empty() {
        return Err("direct prefix hard-competitor inputs have inconsistent lengths".into());
    }
    let mut hard_tokens = Vec::with_capacity(logits.len());
    let mut rank_sum = 0usize;
    let mut top1 = 0usize;
    let mut top5 = 0usize;
    let mut top10 = 0usize;

    for ((row, prefix), &target) in logits.iter().zip(prefixes).zip(targets) {
        if row.len() != FOUNDATION_DIFFUSION_VOCAB_SIZE || target as usize >= row.len() {
            return Err("direct prefix hard-competitor row/target is invalid".into());
        }
        let target_score = row[target as usize];
        if !target_score.is_finite() {
            return Err("direct prefix target logit is not finite".into());
        }
        let mut best: Option<(u32, f32)> = None;
        let mut better = 0usize;
        for token in FOUNDATION_DIFFUSION_EOS + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
            if token == target || !token_allowed(prefix, token) {
                continue;
            }
            let score = row[token as usize];
            if !score.is_finite() {
                continue;
            }
            if score > target_score || (score == target_score && token < target) {
                better += 1;
            }
            match best {
                None => best = Some((token, score)),
                Some((best_token, best_score))
                    if score > best_score || (score == best_score && token < best_token) =>
                {
                    best = Some((token, score));
                }
                _ => {}
            }
        }
        let Some((hard, _)) = best else {
            return Err("direct prefix position has no legal wrong-token competitor".into());
        };
        hard_tokens.push(hard);
        let rank = better + 1;
        rank_sum += rank;
        top1 += usize::from(rank <= 1);
        top5 += usize::from(rank <= 5);
        top10 += usize::from(rank <= 10);
    }
    let n = logits.len() as f64;
    Ok((
        hard_tokens,
        rank_sum as f64 / n,
        top1 as f64 / n,
        top5 as f64 / n,
        top10 as f64 / n,
    ))
}

/// Deterministic derangement used to pair each peptide with another spectrum.
///
/// A cyclic non-zero offset guarantees that no row keeps its matched spectrum.
pub fn foundation_direct_shuffled_order(batch: usize, seed: u64) -> Result<Vec<usize>> {
    if batch < 2 {
        candle_core::bail!("matched-vs-shuffled conditioning requires batch size >= 2");
    }
    let offset = 1 + (mix64(seed) as usize % (batch - 1));
    Ok((0..batch).map(|index| (index + offset) % batch).collect())
}

/// Warm-start accounting for direct decoder initialization from the accepted
/// unified checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectDecoderWarmStartReport {
    /// Spectrum-encoder tensors loaded from the accepted unified parent.
    pub spectrum_encoder_variables: usize,
    /// Shape- and semantics-matching causal decoder tensors loaded.
    pub decoder_variables: usize,
    /// Unrelated forward, diffusion-only, alignment, and compatibility tensors ignored.
    pub ignored_parent_variables: usize,
}

/// Warm-start only the spectrum encoder and genuinely matching AR decoder.
pub fn load_direct_decoder_from_unified_checkpoint(
    varmap: &VarMap,
    parent_checkpoint: &Path,
    device: &Device,
) -> Result<DirectDecoderWarmStartReport> {
    let parent = candle_core::safetensors::load(parent_checkpoint, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("direct-decoder VarMap lock poisoned".into()))?;
    let mut spectrum_encoder_variables = 0usize;
    let mut decoder_variables = 0usize;
    let mut consumed = HashSet::<String>::new();
    let mut missing = Vec::<String>::new();
    for (name, variable) in data.iter() {
        if !(name.starts_with("spectrum_encoder.") || name.starts_with("decoder.")) {
            candle_core::bail!("v0.19 instantiated unexpected variable namespace '{name}'");
        }
        let Some(tensor) = parent.get(name) else {
            missing.push(name.clone());
            continue;
        };
        if tensor.dims() != variable.as_tensor().dims() {
            candle_core::bail!(
                "v0.19 warm-start shape mismatch for '{name}': model {:?}, parent {:?}",
                variable.as_tensor().dims(),
                tensor.dims()
            );
        }
        variable.set(tensor)?;
        consumed.insert(name.clone());
        if name.starts_with("spectrum_encoder.") {
            spectrum_encoder_variables += 1;
        } else {
            decoder_variables += 1;
        }
    }
    drop(data);
    if !missing.is_empty() {
        candle_core::bail!(
            "accepted unified parent lacks required v0.19 tensors: {}",
            missing.join(", ")
        );
    }
    Ok(DirectDecoderWarmStartReport {
        spectrum_encoder_variables,
        decoder_variables,
        ignored_parent_variables: parent
            .keys()
            .filter(|name| !consumed.contains(*name))
            .count(),
    })
}

/// Fixed physical beam-search policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectDecoderBeamConfig {
    /// Number of live partial hypotheses.
    pub beam_width: usize,
    /// Maximum number of closed hypotheses returned.
    pub top_k: usize,
    /// Hard final neutral-mass tolerance.
    pub mass_tolerance_da: f64,
    /// Maximum token width including EOS.
    pub max_tokens: usize,
}

impl DirectDecoderBeamConfig {
    /// Validate the frozen decoding policy.
    pub fn validate(self) -> std::result::Result<(), String> {
        if self.beam_width == 0 || self.top_k == 0 {
            return Err("direct decoder beam_width and top_k must be positive".into());
        }
        if self.max_tokens < 3 {
            return Err("direct decoder max_tokens must be at least three".into());
        }
        if !(self.mass_tolerance_da > 0.0 && self.mass_tolerance_da.is_finite()) {
            return Err("direct decoder mass tolerance must be finite and positive".into());
        }
        Ok(())
    }
}

/// One completed, hard-mass-closed peptide beam.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectDecoderBeamCandidate {
    /// Active token row ending in EOS (no padding).
    pub tokens: Vec<u32>,
    /// Sum of autoregressive token log probabilities, including EOS.
    pub log_probability: f64,
    /// Signed final neutral-mass error.
    pub mass_error_da: f64,
}

#[derive(Debug, Clone)]
struct BeamState {
    tokens: Vec<u32>,
    neutral_mass: f64,
    log_probability: f64,
    residues: usize,
}

/// Run mass-constrained direct beam search from caller-provided next-token logits.
///
/// The callback is invoked once per depth with equal-length prefixes and must
/// return one vocabulary-sized logit row per prefix.  This keeps physical search
/// independently testable while the executable supplies logits from the neural
/// direct decoder and its cached contextual peak memory.
pub fn foundation_direct_beam_search<F>(
    precursor_neutral_mass: f64,
    config: DirectDecoderBeamConfig,
    mut next_logits: F,
) -> std::result::Result<Vec<DirectDecoderBeamCandidate>, String>
where
    F: FnMut(&[Vec<u32>]) -> std::result::Result<Vec<Vec<f32>>, String>,
{
    config.validate()?;
    if !(precursor_neutral_mass > FOUNDATION_PEPTIDE_WATER_MASS_DA
        && precursor_neutral_mass.is_finite())
    {
        return Err("direct decoder requires a finite positive precursor neutral mass".into());
    }
    let residue_masses: Vec<f64> = (0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
        .filter(|&token| foundation_diffusion_token_residue(token).is_some())
        .filter_map(foundation_diffusion_token_mass_da)
        .collect();
    let min_residue_mass = residue_masses.iter().copied().fold(f64::INFINITY, f64::min);
    let max_token_mass = (0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
        .filter_map(foundation_diffusion_token_mass_da)
        .fold(0.0f64, f64::max);
    let mut live = vec![BeamState {
        tokens: Vec::new(),
        neutral_mass: FOUNDATION_PEPTIDE_WATER_MASS_DA,
        log_probability: 0.0,
        residues: 0,
    }];
    let mut completed = HashMap::<Vec<u32>, DirectDecoderBeamCandidate>::new();

    for depth in 0..config.max_tokens {
        if live.is_empty() {
            break;
        }
        let prefixes: Vec<Vec<u32>> = live.iter().map(|state| state.tokens.clone()).collect();
        let logits = next_logits(&prefixes)?;
        if logits.len() != live.len()
            || logits
                .iter()
                .any(|row| row.len() != FOUNDATION_DIFFUSION_VOCAB_SIZE)
        {
            return Err("direct decoder logit callback returned an invalid shape".into());
        }
        // Every autoregressive prefix is a distinct neural state. Prefixes
        // with the same approximate neutral mass and final token are *not*
        // interchangeable because the complete prefix changes all subsequent
        // decoder logits. Earlier v0.19 search collapsed such states by
        // `(mass_bin, last_token)`, which is invalid for a full-prefix causal
        // decoder and can discard a lower-scoring prefix whose continuation is
        // ultimately much better. Keep all physically valid expansions until
        // the ordinary global beam cap is applied below.
        let mut expanded = Vec::<BeamState>::new();
        for (state, row) in live.iter().zip(logits.iter()) {
            if state.residues > 0 {
                let error = state.neutral_mass - precursor_neutral_mass;
                if error.abs() <= config.mass_tolerance_da {
                    let mut tokens = state.tokens.clone();
                    tokens.push(FOUNDATION_DIFFUSION_EOS);
                    let candidate = DirectDecoderBeamCandidate {
                        tokens: tokens.clone(),
                        log_probability: state.log_probability
                            + selected_log_softmax(row, FOUNDATION_DIFFUSION_EOS as usize)?,
                        mass_error_da: error,
                    };
                    completed
                        .entry(tokens)
                        .and_modify(|old| {
                            if candidate.log_probability > old.log_probability {
                                *old = candidate.clone();
                            }
                        })
                        .or_insert(candidate);
                }
            }
            if depth + 1 >= config.max_tokens {
                continue;
            }
            for token in FOUNDATION_DIFFUSION_EOS + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
                if !token_allowed(&state.tokens, token) || !row[token as usize].is_finite() {
                    continue;
                }
                let Some(token_mass) = foundation_diffusion_token_mass_da(token) else {
                    continue;
                };
                let neutral_mass = state.neutral_mass + token_mass;
                if neutral_mass > precursor_neutral_mass + config.mass_tolerance_da {
                    continue;
                }
                let remaining = config.max_tokens - depth - 2;
                if neutral_mass + remaining as f64 * max_token_mass + config.mass_tolerance_da
                    < precursor_neutral_mass
                {
                    continue;
                }
                // A PTM cannot finish a peptide by itself; reserve enough mass
                // for at least one residue when none has yet been emitted.
                if state.residues == 0
                    && foundation_diffusion_token_residue(token).is_none()
                    && neutral_mass + min_residue_mass
                        > precursor_neutral_mass + config.mass_tolerance_da
                {
                    continue;
                }
                let mut tokens = state.tokens.clone();
                tokens.push(token);
                let candidate = BeamState {
                    tokens,
                    neutral_mass,
                    log_probability: state.log_probability
                        + selected_log_softmax(row, token as usize)?,
                    residues: state.residues
                        + usize::from(foundation_diffusion_token_residue(token).is_some()),
                };
                expanded.push(candidate);
            }
        }
        expanded.sort_by(|a, b| b.log_probability.total_cmp(&a.log_probability));
        expanded.truncate(config.beam_width);
        live = expanded;
    }
    let mut completed: Vec<_> = completed.into_values().collect();
    completed.sort_by(|a, b| {
        b.log_probability
            .total_cmp(&a.log_probability)
            .then_with(|| a.mass_error_da.abs().total_cmp(&b.mass_error_da.abs()))
    });
    completed.truncate(config.top_k);
    Ok(completed)
}

fn token_allowed(prefix: &[u32], token: u32) -> bool {
    if token == FOUNDATION_DIFFUSION_PAD
        || token == FOUNDATION_DIFFUSION_MASK
        || token == FOUNDATION_DIFFUSION_EOS
    {
        return false;
    }
    if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
        return prefix.is_empty();
    }
    if let Some(_) = foundation_diffusion_token_residue(token) {
        return true;
    }
    let Some(previous) = prefix.last().copied() else {
        return false;
    };
    let Some(residue) = foundation_diffusion_token_residue(previous) else {
        return false;
    };
    foundation_diffusion_residue_ptm_valid(token, residue)
}

fn selected_log_softmax(logits: &[f32], selected: usize) -> std::result::Result<f64, String> {
    if selected >= logits.len() || !logits[selected].is_finite() {
        return Err("selected direct-decoder logit is invalid".into());
    }
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let denominator: f64 = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| f64::from(v - max).exp())
        .sum();
    if !(denominator > 0.0 && denominator.is_finite()) {
        return Err("direct-decoder softmax denominator is invalid".into());
    }
    Ok(f64::from(logits[selected] - max) - denominator.ln())
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffled_order_is_a_deterministic_derangement() {
        let first = foundation_direct_shuffled_order(8, 20260919).unwrap();
        let second = foundation_direct_shuffled_order(8, 20260919).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().enumerate().all(|(i, &j)| i != j));
    }

    #[test]
    fn prefix_hard_competitor_tracks_current_legal_top_wrong_token() {
        let alanine = 3u32;
        let cysteine = alanine + 1;
        let aspartate = alanine + 2;
        let mut row = vec![-10.0f32; FOUNDATION_DIFFUSION_VOCAB_SIZE];
        row[alanine as usize] = 1.0;
        row[cysteine as usize] = 3.0;
        row[aspartate as usize] = 2.0;
        // PAD/MASK/EOS are deliberately larger but are not live expansion
        // competitors for a nonterminal true token.
        row[FOUNDATION_DIFFUSION_PAD as usize] = 20.0;
        row[FOUNDATION_DIFFUSION_MASK as usize] = 19.0;
        row[FOUNDATION_DIFFUSION_EOS as usize] = 18.0;
        let (hard, mean_rank, top1, top5, top10) =
            select_prefix_hard_competitors(&[row], &[Vec::new()], &[alanine]).unwrap();
        assert_eq!(hard, vec![cysteine]);
        assert_eq!(mean_rank, 3.0);
        assert_eq!(top1, 0.0);
        assert_eq!(top5, 1.0);
        assert_eq!(top10, 1.0);
    }

    #[test]
    fn beam_does_not_merge_distinct_autoregressive_prefixes_with_same_mass_and_last_token() {
        let alanine = 3u32;
        let glycine = 8u32;
        let isoleucine = 10u32;
        let leucine = 12u32;
        assert_eq!(foundation_diffusion_token_residue(isoleucine), Some('I'));
        assert_eq!(foundation_diffusion_token_residue(leucine), Some('L'));
        assert_eq!(
            foundation_diffusion_token_mass_da(isoleucine),
            foundation_diffusion_token_mass_da(leucine)
        );

        let target = FOUNDATION_PEPTIDE_WATER_MASS_DA
            + foundation_diffusion_token_mass_da(leucine).unwrap()
            + foundation_diffusion_token_mass_da(alanine).unwrap()
            + foundation_diffusion_token_mass_da(glycine).unwrap();
        let candidates = foundation_direct_beam_search(
            target,
            DirectDecoderBeamConfig {
                beam_width: 4,
                top_k: 4,
                mass_tolerance_da: 1.0e-5,
                max_tokens: 5,
            },
            |prefixes| {
                Ok(prefixes
                    .iter()
                    .map(|prefix| {
                        let mut row = vec![-50.0; FOUNDATION_DIFFUSION_VOCAB_SIZE];
                        match prefix.as_slice() {
                            [] => {
                                // The I-prefix is initially slightly better than L.
                                row[isoleucine as usize] = 10.0;
                                row[leucine as usize] = 9.8;
                            }
                            [token] if *token == isoleucine => {
                                row[alanine as usize] = 10.0;
                            }
                            [token] if *token == leucine => {
                                row[alanine as usize] = 10.0;
                            }
                            [first, second] if *first == isoleucine && *second == alanine => {
                                // After the mass-isobaric [I,A] / [L,A] collision,
                                // the lower-scoring L-prefix has the much stronger
                                // continuation. A `(mass_bin,last_token)` merge would
                                // have destroyed this path before these logits exist.
                                row[glycine as usize] = -5.0;
                            }
                            [first, second] if *first == leucine && *second == alanine => {
                                row[glycine as usize] = 10.0;
                            }
                            [first, second, third]
                                if (*first == isoleucine || *first == leucine)
                                    && *second == alanine
                                    && *third == glycine =>
                            {
                                row[FOUNDATION_DIFFUSION_EOS as usize] = 10.0;
                            }
                            _ => {
                                row[FOUNDATION_DIFFUSION_EOS as usize] = 0.0;
                            }
                        }
                        row
                    })
                    .collect())
            },
        )
        .unwrap();

        assert!(candidates.iter().any(|candidate| {
            candidate.tokens == vec![leucine, alanine, glycine, FOUNDATION_DIFFUSION_EOS]
        }));
        assert_eq!(
            candidates[0].tokens,
            vec![leucine, alanine, glycine, FOUNDATION_DIFFUSION_EOS,]
        );
    }

    #[test]
    fn beam_enforces_exact_mass_closure() {
        let alanine = 3u32;
        let target =
            FOUNDATION_PEPTIDE_WATER_MASS_DA + foundation_diffusion_token_mass_da(alanine).unwrap();
        let candidates = foundation_direct_beam_search(
            target,
            DirectDecoderBeamConfig {
                beam_width: 8,
                top_k: 5,
                mass_tolerance_da: 1.0e-5,
                max_tokens: 4,
            },
            |prefixes| {
                Ok(prefixes
                    .iter()
                    .map(|prefix| {
                        let mut row = vec![-20.0; FOUNDATION_DIFFUSION_VOCAB_SIZE];
                        if prefix.is_empty() {
                            row[alanine as usize] = 10.0;
                        } else {
                            row[FOUNDATION_DIFFUSION_EOS as usize] = 10.0;
                        }
                        row
                    })
                    .collect())
            },
        )
        .unwrap();
        assert_eq!(
            candidates[0].tokens,
            vec![alanine, FOUNDATION_DIFFUSION_EOS]
        );
        assert!(candidates
            .iter()
            .all(|candidate| candidate.mass_error_da.abs() <= 1.0e-5));
    }
}
