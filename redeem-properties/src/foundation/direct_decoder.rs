//! v0.19 direct spectrum-conditioned autoregressive peptide decoding.
//!
//! The accepted causal backbone already implements the required decoder block:
//! masked peptide self-attention, cross-attention to every contextualized peak
//! (plus one precursor token), and a feed-forward residual block.  This module
//! gives that architecture a direct-generation contract, an explicit
//! matched-versus-shuffled spectrum objective, and precursor-mass-constrained
//! beam search.  It deliberately contains no proposal or scalar compatibility
//! head.

use super::causal::{foundation_causal_conditioning_margin_loss, PeptideSpectrumCausalModel};
use super::diffusion::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use candle_core::{Device, Result, Tensor};
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
