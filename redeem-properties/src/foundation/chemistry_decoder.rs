//! v0.20 chemistry-structured spectrum-conditioned peptide decoder.
//!
//! This module keeps the accepted spectrum-conditioned causal decoder but adds
//! an explicit biochemical transition score at every autoregressive position.
//! Candidate transitions remain chemistry-generated even when observed fragment
//! evidence is missing: measured peaks are soft evidence, never graph/node
//! existence requirements. A conservative suffix-mass lattice is used as a
//! feature during teacher forcing and as a hard feasibility mask only during
//! generation.

use super::causal::{
    FoundationCausalContext, FoundationCausalInputBatch, FoundationCausalOutput,
    PeptideSpectrumCausalModel,
};
use super::chemistry::{common_unimod_definition, residue_graph, ATOM_FEATURE_DIM};
use super::diffusion::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FOUNDATION_DIFFUSION_CARBAMIDOMETHYL, FOUNDATION_DIFFUSION_DEAMIDATED,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_FIRST_RESIDUE, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_OXIDATION, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use super::featurize::PeptidoformInput;
use super::model::PrecursorContextBatch;
use super::spectrum::{FoundationSpectrum, FoundationSpectrumBatch};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder, VarMap};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;

/// Stable v0.20 scientific objective identifier.
pub const FOUNDATION_CHEMISTRY_DECODER_OBJECTIVE_V0200: &str =
    "teacher_forced_ce_plus_matched_shuffled_guard_with_biochemical_transition_logits_v0200";
/// Stable v0.20 architecture identifier.
pub const FOUNDATION_CHEMISTRY_DECODER_ARCHITECTURE_V0200: &str =
    "causal_full_peak_cross_attention_plus_residual_mass_residue_ptm_fragment_transition_head";
/// Width of the explicit candidate-transition feature vector.
pub const FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200: usize = 44;
/// Fragment matching tolerance in parts per million.
pub const FOUNDATION_CHEMISTRY_FRAGMENT_PPM_V0200: f64 = 20.0;
/// Absolute floor on fragment matching tolerance in Da.
pub const FOUNDATION_CHEMISTRY_FRAGMENT_ABS_TOLERANCE_DA_V0200: f64 = 0.02;
/// Discretization of the conservative suffix mass lattice.
pub const FOUNDATION_CHEMISTRY_SUFFIX_BIN_DA_V0200: f64 = 0.01;

const PROTON_MASS_DA: f64 = 1.007_276_466_77;
const CARBON_MONOXIDE_MASS_DA: f64 = 27.994_914_62;
const AMMONIA_MASS_DA: f64 = 17.026_549_10;

/// v0.20 warm-start accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChemistryDecoderWarmStartReport {
    /// Spectrum-encoder tensors copied from the accepted unified parent.
    pub spectrum_encoder_variables: usize,
    /// Shape-compatible causal decoder tensors copied from the parent.
    pub decoder_variables: usize,
    /// New zero-initialized chemistry-transition tensors.
    pub chemistry_transition_variables: usize,
    /// Parent-only tensors ignored by the v0.20 inverse decoder.
    pub ignored_parent_variables: usize,
}

/// Conservative chemistry-grammar-aware suffix-mass reachability table.
///
/// Reachability is generated only from supported residue and residue-local PTM
/// transitions. Observed peaks never participate. The mass discretization uses
/// an explicit rounding allowance so a physically feasible true path is not
/// rejected merely because monoisotopic masses do not land exactly on lattice bins.
#[derive(Debug, Clone)]
pub struct ChemistrySuffixMassLattice {
    resolution_da: f64,
    max_mass_da: f64,
    max_tokens: usize,
    words: usize,
    /// Flattened `[state][remaining_token_budget][word]` bitsets. State 0 is
    /// N-terminal/no-residue-yet, state 1 is after a residue-local PTM, and
    /// states 2..22 correspond to the twenty residue tokens.
    reachable: Vec<u64>,
}

const SUFFIX_STATE_NO_RESIDUE: usize = 0;
const SUFFIX_STATE_AFTER_PTM: usize = 1;
const SUFFIX_STATE_RESIDUE_BASE: usize = 2;
const SUFFIX_STATE_COUNT: usize = 22;

impl ChemistrySuffixMassLattice {
    /// Build a reusable backwards suffix reachability table.
    pub fn new(max_tokens: usize, max_mass_da: f64) -> std::result::Result<Self, String> {
        if max_tokens < 2 {
            return Err("chemistry suffix lattice requires max_tokens >= 2".into());
        }
        if !(max_mass_da > 0.0 && max_mass_da.is_finite()) {
            return Err("chemistry suffix lattice max mass must be finite and positive".into());
        }
        let resolution_da = FOUNDATION_CHEMISTRY_SUFFIX_BIN_DA_V0200;
        let max_bin = (max_mass_da / resolution_da).ceil() as usize + 2;
        let words = (max_bin + 64) / 64;
        let table_len = SUFFIX_STATE_COUNT
            .checked_mul(max_tokens + 1)
            .and_then(|value| value.checked_mul(words))
            .ok_or_else(|| "chemistry suffix lattice allocation overflow".to_string())?;
        let mut lattice = Self {
            resolution_da,
            max_mass_da,
            max_tokens,
            words,
            reachable: vec![0u64; table_len],
        };

        // With zero remaining non-EOS tokens, EOS may close only after at least
        // one residue has already been emitted. The no-residue state therefore
        // intentionally lacks the zero-mass bit.
        for state in SUFFIX_STATE_AFTER_PTM..SUFFIX_STATE_COUNT {
            set_bit(lattice.bits_mut(state, 0), 0);
        }

        let residue_tokens = (0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
            .filter(|&token| foundation_diffusion_token_residue(token).is_some())
            .collect::<Vec<_>>();
        let ptm_tokens = [
            FOUNDATION_DIFFUSION_RESIDUE_ACETYL,
            FOUNDATION_DIFFUSION_CARBAMIDOMETHYL,
            FOUNDATION_DIFFUSION_DEAMIDATED,
            FOUNDATION_DIFFUSION_OXIDATION,
        ];

        for budget in 1..=max_tokens {
            // All states may next emit any standard residue. Compute this common
            // union once for the budget instead of repeating it for 22 states.
            let mut residue_union = vec![0u64; words];
            for &token in &residue_tokens {
                let shift = lattice.mass_shift(token)?;
                let next_state = suffix_state_for_residue_token(token)
                    .ok_or_else(|| format!("missing suffix state for residue token {token}"))?;
                let source = lattice.bits(next_state, budget - 1);
                or_shifted_bits(&mut residue_union, source, shift, max_bin);
            }

            for state in 0..SUFFIX_STATE_COUNT {
                let mut current = lattice.bits(state, budget - 1).to_vec();
                or_bits(&mut current, &residue_union);
                if let Some(residue) = suffix_state_residue(state) {
                    for &token in &ptm_tokens {
                        if foundation_diffusion_residue_ptm_valid(token, residue) {
                            let shift = lattice.mass_shift(token)?;
                            let source = lattice.bits(SUFFIX_STATE_AFTER_PTM, budget - 1);
                            or_shifted_bits(&mut current, source, shift, max_bin);
                        }
                    }
                }
                lattice.bits_mut(state, budget).copy_from_slice(&current);
            }
        }
        Ok(lattice)
    }

    /// Whether the residual mass after one proposed transition can be completed
    /// under supported residue/PTM token grammar within the remaining budget.
    pub fn reachable_after_transition(
        &self,
        prefix: &[u32],
        token: u32,
        mass_da: f64,
        remaining_tokens: usize,
        tolerance_da: f64,
    ) -> bool {
        let Some(state) = suffix_state_after_transition(prefix, token) else {
            return false;
        };
        self.reachable_from_state(state, mass_da, remaining_tokens, tolerance_da)
    }

    fn reachable_from_state(
        &self,
        state: usize,
        mass_da: f64,
        remaining_tokens: usize,
        tolerance_da: f64,
    ) -> bool {
        if state >= SUFFIX_STATE_COUNT
            || !mass_da.is_finite()
            || !tolerance_da.is_finite()
            || tolerance_da < 0.0
        {
            return false;
        }
        if mass_da < -tolerance_da {
            return false;
        }
        if mass_da.abs() <= tolerance_da {
            return state != SUFFIX_STATE_NO_RESIDUE;
        }
        // Outside the precomputed range, do not hard-reject. The runner sizes
        // the lattice above every frozen validation precursor mass, so this is
        // only a conservative training-feature fallback.
        if mass_da > self.max_mass_da {
            return true;
        }
        let budget = remaining_tokens.min(self.max_tokens);
        let rounding_pad = (budget as f64 * self.resolution_da * 0.5) + self.resolution_da;
        let window = tolerance_da + rounding_pad;
        let low = ((mass_da - window).max(0.0) / self.resolution_da).floor() as usize;
        let high = ((mass_da + window) / self.resolution_da).ceil() as usize;
        let bits = self.bits(state, budget);
        (low..=high).any(|bin| bit_is_set(bits, bin))
    }

    fn mass_shift(&self, token: u32) -> std::result::Result<usize, String> {
        let mass = foundation_diffusion_token_mass_da(token)
            .ok_or_else(|| format!("suffix lattice token {token} has no clean mass"))?;
        let shift = (mass / self.resolution_da).round() as usize;
        if shift == 0 {
            return Err(format!("suffix lattice token {token} rounded to zero mass"));
        }
        Ok(shift)
    }

    fn offset(&self, state: usize, budget: usize) -> usize {
        (state * (self.max_tokens + 1) + budget) * self.words
    }

    fn bits(&self, state: usize, budget: usize) -> &[u64] {
        let offset = self.offset(state, budget);
        &self.reachable[offset..offset + self.words]
    }

    fn bits_mut(&mut self, state: usize, budget: usize) -> &mut [u64] {
        let offset = self.offset(state, budget);
        &mut self.reachable[offset..offset + self.words]
    }
}

fn suffix_state_for_residue_token(token: u32) -> Option<usize> {
    foundation_diffusion_token_residue(token)?;
    let residue_offset = token.checked_sub(FOUNDATION_DIFFUSION_FIRST_RESIDUE)? as usize;
    (residue_offset < 20).then_some(SUFFIX_STATE_RESIDUE_BASE + residue_offset)
}

fn suffix_state_residue(state: usize) -> Option<char> {
    if !(SUFFIX_STATE_RESIDUE_BASE..SUFFIX_STATE_COUNT).contains(&state) {
        return None;
    }
    foundation_diffusion_token_residue(
        FOUNDATION_DIFFUSION_FIRST_RESIDUE + (state - SUFFIX_STATE_RESIDUE_BASE) as u32,
    )
}

fn suffix_state_after_transition(prefix: &[u32], token: u32) -> Option<usize> {
    if let Some(state) = suffix_state_for_residue_token(token) {
        return Some(state);
    }
    if token == FOUNDATION_DIFFUSION_NTERM_ACETYL && prefix.is_empty() {
        return Some(SUFFIX_STATE_NO_RESIDUE);
    }
    if matches!(
        token,
        FOUNDATION_DIFFUSION_RESIDUE_ACETYL
            | FOUNDATION_DIFFUSION_CARBAMIDOMETHYL
            | FOUNDATION_DIFFUSION_DEAMIDATED
            | FOUNDATION_DIFFUSION_OXIDATION
    ) && prefix
        .iter()
        .any(|&value| foundation_diffusion_token_residue(value).is_some())
    {
        return Some(SUFFIX_STATE_AFTER_PTM);
    }
    None
}

fn set_bit(bits: &mut [u64], bit: usize) {
    let word = bit / 64;
    let offset = bit % 64;
    if let Some(value) = bits.get_mut(word) {
        *value |= 1u64 << offset;
    }
}

fn bit_is_set(bits: &[u64], bit: usize) -> bool {
    bits.get(bit / 64)
        .map(|word| (word & (1u64 << (bit % 64))) != 0)
        .unwrap_or(false)
}

fn or_bits(destination: &mut [u64], source: &[u64]) {
    for (dst, src) in destination.iter_mut().zip(source) {
        *dst |= *src;
    }
}

fn or_shifted_bits(destination: &mut [u64], source: &[u64], shift: usize, max_bit: usize) {
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    for (source_word, &value) in source.iter().enumerate() {
        if value == 0 {
            continue;
        }
        let destination_word = source_word + word_shift;
        if destination_word >= destination.len() {
            break;
        }
        destination[destination_word] |= value << bit_shift;
        if bit_shift != 0 && destination_word + 1 < destination.len() {
            destination[destination_word + 1] |= value >> (64 - bit_shift);
        }
    }
    let excess = destination.len() * 64 - (max_bit + 1);
    if excess > 0 && excess < 64 {
        if let Some(last) = destination.last_mut() {
            *last &= u64::MAX >> excess;
        }
    }
}

#[derive(Debug, Clone)]
struct TokenChemistrySummary {
    atom_mean: [f32; ATOM_FEATURE_DIM],
    atom_count: f32,
    bond_count: f32,
    composition: [f32; 8],
}

impl Default for TokenChemistrySummary {
    fn default() -> Self {
        Self {
            atom_mean: [0.0; ATOM_FEATURE_DIM],
            atom_count: 0.0,
            bond_count: 0.0,
            composition: [0.0; 8],
        }
    }
}

/// CPU-side deterministic featurizer shared by teacher forcing and generation.
#[derive(Debug, Clone)]
pub struct ChemistryTransitionFeaturizer {
    max_tokens: usize,
    max_peaks: usize,
    precursor_mass_tolerance_da: f64,
    token_summaries: Vec<TokenChemistrySummary>,
    suffix_lattice: ChemistrySuffixMassLattice,
}

/// Tensorized candidate-transition features.
#[derive(Debug, Clone)]
pub struct ChemistryTransitionBatch {
    /// `[batch, positions, vocabulary, 44]` for teacher forcing.
    pub features: Tensor,
}

impl ChemistryTransitionFeaturizer {
    /// Construct the frozen v0.20 featurizer.
    pub fn new(
        config: &FoundationDiffusionConfig,
        suffix_lattice: ChemistrySuffixMassLattice,
    ) -> std::result::Result<Self, String> {
        config.validate()?;
        let token_summaries = (0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
            .map(token_chemistry_summary)
            .collect();
        Ok(Self {
            max_tokens: config.max_tokens,
            max_peaks: config.spectrum.max_peaks,
            precursor_mass_tolerance_da: config.precursor_mass_tolerance_da,
            token_summaries,
            suffix_lattice,
        })
    }

    /// Build all candidate features at every clean teacher-forced target prefix.
    pub fn teacher_forced(
        &self,
        peptides: &[PeptidoformInput],
        spectra: &[FoundationSpectrum],
        precursor_masses: &[f64],
        charges: &[i32],
        device: &Device,
    ) -> Result<ChemistryTransitionBatch> {
        let batch = peptides.len();
        if batch == 0
            || spectra.len() != batch
            || precursor_masses.len() != batch
            || charges.len() != batch
        {
            candle_core::bail!("v0.20 teacher feature inputs have inconsistent batch lengths");
        }
        let vocabulary = FoundationDiffusionVocabulary;
        let rows = (0..batch)
            .into_par_iter()
            .map(|index| {
                let target = vocabulary
                    .encode(&peptides[index], self.max_tokens)
                    .map_err(|error| format!("v0.20 encode teacher row {index}: {error}"))?;
                let evidence = PreparedSpectrum::new(&spectra[index], self.max_peaks)?;
                self.teacher_row_features(
                    &target,
                    &evidence,
                    precursor_masses[index],
                    charges[index],
                )
            })
            .collect::<std::result::Result<Vec<_>, String>>()
            .map_err(candle_core::Error::Msg)?;
        let flat = rows.into_iter().flatten().collect::<Vec<_>>();
        Ok(ChemistryTransitionBatch {
            features: Tensor::from_vec(
                flat,
                (
                    batch,
                    self.max_tokens,
                    FOUNDATION_DIFFUSION_VOCAB_SIZE,
                    FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
                ),
                device,
            )?,
        })
    }

    /// Build candidate features for the next-token position of arbitrary beam prefixes.
    pub fn next_prefixes(
        &self,
        prefixes: &[Vec<u32>],
        spectrum: &FoundationSpectrum,
        precursor_mass: f64,
        charge: i32,
        device: &Device,
    ) -> Result<Tensor> {
        if prefixes.is_empty() {
            candle_core::bail!("v0.20 next-prefix features require non-empty prefix batch");
        }
        let evidence =
            PreparedSpectrum::new(spectrum, self.max_peaks).map_err(candle_core::Error::Msg)?;
        let rows = prefixes
            .par_iter()
            .map(|prefix| self.next_row_features(prefix, &evidence, precursor_mass, charge))
            .collect::<std::result::Result<Vec<_>, String>>()
            .map_err(candle_core::Error::Msg)?;
        let flat = rows.into_iter().flatten().collect::<Vec<_>>();
        Tensor::from_vec(
            flat,
            (
                prefixes.len(),
                FOUNDATION_DIFFUSION_VOCAB_SIZE,
                FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            ),
            device,
        )
    }

    /// Apply the hard conservative suffix-feasibility mask to next-token logits.
    pub fn mask_infeasible_next_logits(
        &self,
        prefixes: &[Vec<u32>],
        rows: &mut [Vec<f32>],
        precursor_mass: f64,
    ) -> std::result::Result<(), String> {
        if prefixes.len() != rows.len() {
            return Err("v0.20 prefix/logit batch mismatch".into());
        }
        for (prefix, row) in prefixes.iter().zip(rows.iter_mut()) {
            if row.len() != FOUNDATION_DIFFUSION_VOCAB_SIZE {
                return Err("v0.20 next-token logit row has wrong vocabulary width".into());
            }
            for token in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
                if token == FOUNDATION_DIFFUSION_EOS {
                    // EOS is evaluated by the existing exact mass-closure logic.
                    continue;
                }
                if !self.transition_suffix_feasible(prefix, token, precursor_mass) {
                    row[token as usize] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(())
    }

    /// Verify that an encoded true path survives the v0.20 chemical grammar and
    /// hard conservative suffix mass mask.
    pub fn true_path_feasible(
        &self,
        peptide: &PeptidoformInput,
        precursor_mass: f64,
    ) -> std::result::Result<bool, String> {
        let row = FoundationDiffusionVocabulary.encode(peptide, self.max_tokens)?;
        let active = row
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
            .unwrap_or(row.len());
        if active == 0 || row[active - 1] != FOUNDATION_DIFFUSION_EOS {
            return Ok(false);
        }
        let mut prefix = Vec::new();
        for &token in &row[..active - 1] {
            if !self.transition_suffix_feasible(&prefix, token, precursor_mass) {
                return Ok(false);
            }
            prefix.push(token);
        }
        let prefix_mass = neutral_mass_for_prefix(&prefix)?;
        Ok((prefix_mass - precursor_mass).abs() <= self.precursor_mass_tolerance_da)
    }

    fn teacher_row_features(
        &self,
        target: &[u32],
        evidence: &PreparedSpectrum,
        precursor_mass: f64,
        charge: i32,
    ) -> std::result::Result<Vec<f32>, String> {
        if target.len() != self.max_tokens {
            return Err("v0.20 teacher row width mismatch".into());
        }
        let mut output = vec![
            0.0f32;
            self.max_tokens
                * FOUNDATION_DIFFUSION_VOCAB_SIZE
                * FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200
        ];
        let mut prefix = Vec::<u32>::new();
        for position in 0..self.max_tokens {
            let target_token = target[position];
            if target_token == FOUNDATION_DIFFUSION_PAD {
                break;
            }
            let row_features =
                self.candidate_features(&prefix, evidence, precursor_mass, charge, position)?;
            let start = position
                * FOUNDATION_DIFFUSION_VOCAB_SIZE
                * FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200;
            output[start..start + row_features.len()].copy_from_slice(&row_features);
            if target_token != FOUNDATION_DIFFUSION_EOS {
                prefix.push(target_token);
            }
        }
        Ok(output)
    }

    fn next_row_features(
        &self,
        prefix: &[u32],
        evidence: &PreparedSpectrum,
        precursor_mass: f64,
        charge: i32,
    ) -> std::result::Result<Vec<f32>, String> {
        self.candidate_features(prefix, evidence, precursor_mass, charge, prefix.len())
    }

    fn candidate_features(
        &self,
        prefix: &[u32],
        evidence: &PreparedSpectrum,
        precursor_mass: f64,
        charge: i32,
        position: usize,
    ) -> std::result::Result<Vec<f32>, String> {
        let prefix_mass = neutral_mass_for_prefix(prefix)?;
        let mut output = vec![
            0.0f32;
            FOUNDATION_DIFFUSION_VOCAB_SIZE
                * FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200
        ];
        for token in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
            let features = self.transition_features(
                prefix,
                prefix_mass,
                token,
                evidence,
                precursor_mass,
                charge,
                position,
            );
            let start = token as usize * FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200;
            output[start..start + FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200]
                .copy_from_slice(&features);
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn transition_features(
        &self,
        prefix: &[u32],
        prefix_mass: f64,
        token: u32,
        evidence: &PreparedSpectrum,
        precursor_mass: f64,
        charge: i32,
        position: usize,
    ) -> [f32; FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200] {
        let mut feature = [0.0f32; FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200];
        if !precursor_mass.is_finite() || precursor_mass <= FOUNDATION_PEPTIDE_WATER_MASS_DA {
            return feature;
        }

        let is_eos = token == FOUNDATION_DIFFUSION_EOS;
        let token_mass = foundation_diffusion_token_mass_da(token).unwrap_or(0.0);
        let legal = if is_eos {
            prefix
                .iter()
                .any(|&value| foundation_diffusion_token_residue(value).is_some())
        } else {
            transition_token_allowed(prefix, token)
        };
        let new_mass = if is_eos {
            prefix_mass
        } else {
            prefix_mass + token_mass
        };
        let residual_before = precursor_mass - prefix_mass;
        let residual_after = precursor_mass - new_mass;
        let remaining_slots = self
            .max_tokens
            .saturating_sub(prefix.len().saturating_add(2));
        let suffix_feasible = if is_eos {
            residual_before.abs() <= self.precursor_mass_tolerance_da
        } else if legal && residual_after >= -self.precursor_mass_tolerance_da {
            self.suffix_lattice.reachable_after_transition(
                prefix,
                token,
                residual_after,
                remaining_slots,
                self.precursor_mass_tolerance_da,
            )
        } else {
            false
        };

        feature[0] = (token_mass / 250.0).clamp(-2.0, 2.0) as f32;
        feature[1] = (prefix_mass / precursor_mass).clamp(0.0, 1.5) as f32;
        feature[2] = (new_mass / precursor_mass).clamp(0.0, 1.5) as f32;
        feature[3] = (residual_before / precursor_mass).clamp(-0.5, 1.5) as f32;
        feature[4] = (residual_after / precursor_mass).clamp(-0.5, 1.5) as f32;
        feature[5] = (position as f64 / self.max_tokens.max(1) as f64).clamp(0.0, 1.0) as f32;
        feature[6] = if foundation_diffusion_token_residue(token).is_some() {
            1.0
        } else {
            0.0
        };
        feature[7] = if matches!(
            token,
            FOUNDATION_DIFFUSION_NTERM_ACETYL
                | FOUNDATION_DIFFUSION_RESIDUE_ACETYL
                | FOUNDATION_DIFFUSION_CARBAMIDOMETHYL
                | FOUNDATION_DIFFUSION_DEAMIDATED
                | FOUNDATION_DIFFUSION_OXIDATION
        ) {
            1.0
        } else {
            0.0
        };
        feature[8] = if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
            1.0
        } else {
            0.0
        };
        feature[9] = if is_eos { 1.0 } else { 0.0 };
        feature[10] = if suffix_feasible { 1.0 } else { 0.0 };
        feature[11] = (charge as f32 / 6.0).clamp(0.0, 1.5);

        if let Some(summary) = self.token_summaries.get(token as usize) {
            feature[12..24].copy_from_slice(&summary.atom_mean);
            feature[24] = summary.atom_count;
            feature[25] = summary.bond_count;
            feature[26..34].copy_from_slice(&summary.composition);
        }

        if legal && !is_eos && prefix_or_token_has_residue(prefix, token) {
            // Neutral mass currently stores water + emitted residue/PTM masses.
            // b-ion neutral residue contribution excludes the peptide water.
            let prefix_residue_mass = (new_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA).max(0.0);
            let complementary_y_neutral = precursor_mass - prefix_residue_mass;
            let b1_mz = prefix_residue_mass + PROTON_MASS_DA;
            let b2_mz = (prefix_residue_mass + 2.0 * PROTON_MASS_DA) / 2.0;
            let y1_mz = complementary_y_neutral + PROTON_MASS_DA;
            let y2_mz = (complementary_y_neutral + 2.0 * PROTON_MASS_DA) / 2.0;
            let b1 = evidence.support(b1_mz);
            let b2 = if charge >= 2 {
                evidence.support(b2_mz)
            } else {
                0.0
            };
            let y1 = evidence.support(y1_mz);
            let y2 = if charge >= 2 {
                evidence.support(y2_mz)
            } else {
                0.0
            };
            feature[34] = b1;
            feature[35] = b2;
            feature[36] = y1;
            feature[37] = y2;
            feature[38] = (b1 * y1).sqrt();
            feature[39] = (b2 * y2).sqrt();
            feature[40] = evidence.support(b1_mz - CARBON_MONOXIDE_MASS_DA);
            feature[41] = evidence
                .support(b1_mz - FOUNDATION_PEPTIDE_WATER_MASS_DA)
                .max(evidence.support(y1_mz - FOUNDATION_PEPTIDE_WATER_MASS_DA));
            feature[42] = evidence
                .support(b1_mz - AMMONIA_MASS_DA)
                .max(evidence.support(y1_mz - AMMONIA_MASS_DA));
            feature[43] = b1.max(b2).max(y1).max(y2);
        }
        feature
    }

    fn transition_suffix_feasible(&self, prefix: &[u32], token: u32, precursor_mass: f64) -> bool {
        if !transition_token_allowed(prefix, token) {
            return false;
        }
        let Ok(prefix_mass) = neutral_mass_for_prefix(prefix) else {
            return false;
        };
        let Some(token_mass) = foundation_diffusion_token_mass_da(token) else {
            return false;
        };
        let new_mass = prefix_mass + token_mass;
        let residual = precursor_mass - new_mass;
        if residual < -self.precursor_mass_tolerance_da {
            return false;
        }
        if !prefix_or_token_has_residue(prefix, token) {
            let min_residue = minimum_residue_mass();
            if residual + self.precursor_mass_tolerance_da < min_residue {
                return false;
            }
        }
        let remaining_slots = self
            .max_tokens
            .saturating_sub(prefix.len().saturating_add(2));
        self.suffix_lattice.reachable_after_transition(
            prefix,
            token,
            residual,
            remaining_slots,
            self.precursor_mass_tolerance_da,
        )
    }
}

#[derive(Debug, Clone)]
struct PreparedSpectrum {
    peaks: Vec<(f64, f32)>,
}

impl PreparedSpectrum {
    fn new(spectrum: &FoundationSpectrum, max_peaks: usize) -> std::result::Result<Self, String> {
        let mut peaks = spectrum
            .peaks
            .iter()
            .filter(|peak| {
                peak.mz.is_finite()
                    && peak.mz > 0.0
                    && peak.intensity.is_finite()
                    && peak.intensity > 0.0
            })
            .map(|peak| (f64::from(peak.mz), peak.intensity))
            .collect::<Vec<_>>();
        peaks.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.total_cmp(&right.0))
        });
        peaks.truncate(max_peaks);
        if peaks.is_empty() {
            return Err("v0.20 spectrum contains no finite positive observed peaks".into());
        }
        let max_intensity = peaks
            .iter()
            .map(|(_, intensity)| *intensity)
            .fold(0.0f32, f32::max)
            .max(f32::EPSILON);
        for (_, intensity) in &mut peaks {
            *intensity = (*intensity / max_intensity).clamp(0.0, 1.0);
        }
        peaks.sort_by(|left, right| left.0.total_cmp(&right.0));
        Ok(Self { peaks })
    }

    fn support(&self, target_mz: f64) -> f32 {
        if !(target_mz > 0.0 && target_mz.is_finite()) {
            return 0.0;
        }
        let tolerance = (target_mz * FOUNDATION_CHEMISTRY_FRAGMENT_PPM_V0200 * 1.0e-6)
            .max(FOUNDATION_CHEMISTRY_FRAGMENT_ABS_TOLERANCE_DA_V0200);
        let lower = target_mz - tolerance;
        let upper = target_mz + tolerance;
        let start = self.peaks.partition_point(|(mz, _)| *mz < lower);
        self.peaks[start..]
            .iter()
            .take_while(|(mz, _)| *mz <= upper)
            .map(|(_, intensity)| *intensity)
            .fold(0.0f32, f32::max)
    }
}

fn token_chemistry_summary(token: u32) -> TokenChemistrySummary {
    let mut summary = TokenChemistrySummary::default();
    if let Some(residue) = foundation_diffusion_token_residue(token) {
        if let Some(graph) = residue_graph(residue) {
            let atom_count = graph.atoms.len().max(1) as f32;
            for atom in &graph.atoms {
                let features = atom.features(false, false);
                for (dst, value) in summary.atom_mean.iter_mut().zip(features) {
                    *dst += value / atom_count;
                }
            }
            summary.atom_count = (graph.atoms.len() as f32 / 20.0).min(1.5);
            summary.bond_count = (graph.bonds.len() as f32 / 20.0).min(1.5);
        }
    }
    let unimod: Option<u32> = match token {
        FOUNDATION_DIFFUSION_NTERM_ACETYL | FOUNDATION_DIFFUSION_RESIDUE_ACETYL => Some(1),
        FOUNDATION_DIFFUSION_CARBAMIDOMETHYL => Some(4),
        FOUNDATION_DIFFUSION_DEAMIDATED => Some(7),
        FOUNDATION_DIFFUSION_OXIDATION => Some(35),
        _ => None,
    };
    if let Some(definition) = unimod.and_then(common_unimod_definition) {
        let c = definition.composition;
        summary.composition = [
            c.carbon as f32 / 10.0,
            c.carbon_13 as f32 / 10.0,
            c.hydrogen as f32 / 25.0,
            c.nitrogen as f32 / 10.0,
            c.nitrogen_15 as f32 / 10.0,
            c.oxygen as f32 / 10.0,
            c.sulfur as f32 / 4.0,
            c.phosphorus as f32 / 4.0,
        ];
    }
    summary
}

fn transition_token_allowed(prefix: &[u32], token: u32) -> bool {
    if token == FOUNDATION_DIFFUSION_PAD
        || token == FOUNDATION_DIFFUSION_MASK
        || token == FOUNDATION_DIFFUSION_EOS
    {
        return false;
    }
    if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
        return prefix.is_empty();
    }
    if foundation_diffusion_token_residue(token).is_some() {
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

fn prefix_or_token_has_residue(prefix: &[u32], token: u32) -> bool {
    foundation_diffusion_token_residue(token).is_some()
        || prefix
            .iter()
            .any(|&value| foundation_diffusion_token_residue(value).is_some())
}

fn neutral_mass_for_prefix(prefix: &[u32]) -> std::result::Result<f64, String> {
    let mut mass = FOUNDATION_PEPTIDE_WATER_MASS_DA;
    for &token in prefix {
        let token_mass = foundation_diffusion_token_mass_da(token)
            .ok_or_else(|| format!("v0.20 prefix contains non-clean token {token}"))?;
        mass += token_mass;
    }
    Ok(mass)
}

fn minimum_residue_mass() -> f64 {
    (0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
        .filter(|&token| foundation_diffusion_token_residue(token).is_some())
        .filter_map(foundation_diffusion_token_mass_da)
        .fold(f64::INFINITY, f64::min)
}

/// v0.20 model: accepted causal decoder plus an explicit biochemical transition score.
#[derive(Clone)]
pub struct PeptideSpectrumChemistryDecoder {
    base: PeptideSpectrumCausalModel,
    hidden_to_features: Linear,
    feature_bias: Linear,
}

impl PeptideSpectrumChemistryDecoder {
    /// Construct the chemistry-structured decoder. Both new output paths are
    /// initialized to exactly zero so step-0 logits equal the accepted causal warm start.
    pub fn new(config: FoundationDiffusionConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let base = PeptideSpectrumCausalModel::new(config.clone(), vb.clone())?;
        let chemistry = vb.pp("chemistry_transition");
        let hidden_vb = chemistry.pp("hidden_to_features");
        let hidden_weight = hidden_vb.get_with_hints(
            (
                FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
                config.model_dim,
            ),
            "weight",
            nn::Init::Const(0.0),
        )?;
        let hidden_bias = hidden_vb.get_with_hints(
            FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            "bias",
            nn::Init::Const(0.0),
        )?;
        let feature_vb = chemistry.pp("feature_bias");
        let feature_weight = feature_vb.get_with_hints(
            (1, FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200),
            "weight",
            nn::Init::Const(0.0),
        )?;
        let feature_bias_value = feature_vb.get_with_hints(1, "bias", nn::Init::Const(0.0))?;
        Ok(Self {
            base,
            hidden_to_features: Linear::new(hidden_weight, Some(hidden_bias)),
            feature_bias: Linear::new(feature_weight, Some(feature_bias_value)),
        })
    }

    /// Encode the observed spectrum/precursor once for generation.
    pub fn prepare_context(
        &self,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationCausalContext> {
        self.base.prepare_context(spectrum, precursor, train)
    }

    /// Teacher-forced forward pass with explicit candidate-transition features.
    pub fn forward_t(
        &self,
        input: &FoundationCausalInputBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        chemistry: &ChemistryTransitionBatch,
        train: bool,
    ) -> Result<FoundationCausalOutput> {
        let output = self.base.forward_t(input, spectrum, precursor, train)?;
        self.apply_full_transition_logits(output, &chemistry.features)
    }

    /// Next-token generation pass using a cached spectrum context.
    pub fn forward_next_t_with_context(
        &self,
        input: &FoundationCausalInputBatch,
        context: &FoundationCausalContext,
        chemistry_features: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let output = self.base.forward_t_with_context(input, context, train)?;
        let (_, token_len, _) = output.decoder_hidden.dims3()?;
        if token_len == 0 {
            candle_core::bail!("v0.20 next-token decoder requires an active START position");
        }
        let hidden = output
            .decoder_hidden
            .narrow(1, token_len - 1, 1)?
            .squeeze(1)?;
        let base_logits = output
            .token_logits
            .narrow(1, token_len - 1, 1)?
            .squeeze(1)?;
        self.apply_next_transition_logits(&hidden, &base_logits, chemistry_features)
    }

    /// Shared inverse configuration.
    pub fn config(&self) -> &FoundationDiffusionConfig {
        self.base.config()
    }

    fn apply_full_transition_logits(
        &self,
        output: FoundationCausalOutput,
        features: &Tensor,
    ) -> Result<FoundationCausalOutput> {
        let (batch, positions, classes) = output.token_logits.dims3()?;
        let feature_dims = features.dims4()?;
        if feature_dims
            != (
                batch,
                positions,
                classes,
                FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            )
        {
            candle_core::bail!(
                "v0.20 chemistry feature shape {:?} does not match logits ({batch},{positions},{classes})",
                feature_dims
            );
        }
        let coefficients = self.hidden_to_features.forward(&output.decoder_hidden)?;
        let dynamic = features
            .broadcast_mul(&coefficients.unsqueeze(2)?)?
            .sum(3)?
            .affine(
                1.0 / (FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200 as f64).sqrt(),
                0.0,
            )?;
        let flat_features = features.reshape((
            batch * positions * classes,
            FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
        ))?;
        let static_bias = self
            .feature_bias
            .forward(&flat_features)?
            .reshape((batch, positions, classes))?;
        let transition = (&dynamic + &static_bias)?;
        let token_logits = (&output.token_logits + &transition)?;
        Ok(FoundationCausalOutput {
            token_logits,
            decoder_hidden: output.decoder_hidden,
            spectrum_memory: output.spectrum_memory,
            spectrum_memory_mask: output.spectrum_memory_mask,
            spectrum_embedding: output.spectrum_embedding,
        })
    }

    fn apply_next_transition_logits(
        &self,
        hidden: &Tensor,
        base_logits: &Tensor,
        features: &Tensor,
    ) -> Result<Tensor> {
        let (batch, classes) = base_logits.dims2()?;
        let feature_dims = features.dims3()?;
        if feature_dims
            != (
                batch,
                classes,
                FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            )
        {
            candle_core::bail!(
                "v0.20 next chemistry feature shape {:?} does not match logits ({batch},{classes})",
                feature_dims
            );
        }
        let coefficients = self.hidden_to_features.forward(hidden)?;
        let dynamic = features
            .broadcast_mul(&coefficients.unsqueeze(1)?)?
            .sum(2)?
            .affine(
                1.0 / (FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200 as f64).sqrt(),
                0.0,
            )?;
        let flat_features = features.reshape((
            batch * classes,
            FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
        ))?;
        let static_bias = self
            .feature_bias
            .forward(&flat_features)?
            .reshape((batch, classes))?;
        base_logits + (&dynamic + &static_bias)?
    }
}

/// Warm-start the accepted spectrum/causal tensors and retain the new chemistry
/// transition parameters at their explicit zero initialization.
pub fn load_chemistry_decoder_from_unified_checkpoint(
    varmap: &VarMap,
    parent_checkpoint: &Path,
    device: &Device,
) -> Result<ChemistryDecoderWarmStartReport> {
    let parent = candle_core::safetensors::load(parent_checkpoint, device)?;
    let data = varmap.data().lock().map_err(|_| {
        candle_core::Error::Msg("v0.20 chemistry decoder VarMap lock poisoned".into())
    })?;
    let mut spectrum_encoder_variables = 0usize;
    let mut decoder_variables = 0usize;
    let mut chemistry_transition_variables = 0usize;
    let mut consumed = HashSet::<String>::new();
    let mut missing = Vec::<String>::new();

    for (name, variable) in data.iter() {
        if name.starts_with("chemistry_transition.") {
            chemistry_transition_variables += 1;
            continue;
        }
        if !(name.starts_with("spectrum_encoder.") || name.starts_with("decoder.")) {
            candle_core::bail!("v0.20 instantiated unexpected variable namespace '{name}'");
        }
        let Some(tensor) = parent.get(name) else {
            missing.push(name.clone());
            continue;
        };
        if tensor.dims() != variable.as_tensor().dims() {
            candle_core::bail!(
                "v0.20 warm-start shape mismatch for '{name}': model {:?}, parent {:?}",
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
            "accepted unified parent lacks required v0.20 tensors: {}",
            missing.join(", ")
        );
    }
    Ok(ChemistryDecoderWarmStartReport {
        spectrum_encoder_variables,
        decoder_variables,
        chemistry_transition_variables,
        ignored_parent_variables: parent
            .keys()
            .filter(|name| !consumed.contains(*name))
            .count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_lattice_accepts_known_residue_compositions() {
        let lattice = ChemistrySuffixMassLattice::new(12, 2_000.0).unwrap();
        let a = foundation_diffusion_token_mass_da(3).unwrap();
        let g = foundation_diffusion_token_mass_da(8).unwrap();
        assert!(lattice.reachable_from_state(SUFFIX_STATE_NO_RESIDUE, a + g, 2, 0.05));
        assert!(!lattice.reachable_from_state(SUFFIX_STATE_NO_RESIDUE, 1.234, 2, 0.01));
    }

    #[test]
    fn fragment_support_is_soft_and_missing_peaks_do_not_delete_transition() {
        let spectrum = FoundationSpectrum::from_pairs([(100.0, 10.0), (200.0, 5.0)]);
        let evidence = PreparedSpectrum::new(&spectrum, 256).unwrap();
        assert_eq!(evidence.support(150.0), 0.0);
        assert!(evidence.support(100.0) > 0.99);
    }

    #[test]
    fn chemistry_summary_uses_residue_graph_and_ptm_composition() {
        let alanine = token_chemistry_summary(3);
        assert!(alanine.atom_count > 0.0);
        let oxidation = token_chemistry_summary(FOUNDATION_DIFFUSION_OXIDATION);
        assert!(oxidation.composition[5] > 0.0);
    }
}
