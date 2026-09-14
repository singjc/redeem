//! v0.21 chemistry-conditioned whole-sequence categorical diffusion/refinement.
//!
//! This module deliberately combines the two successful ingredients established
//! before v0.21: the bidirectional spectrum-conditioned discrete diffusion
//! scaffold from v0.12.x and the explicit biochemical transition evidence from
//! v0.20. Unlike the closed autoregressive family, every active peptide position
//! is visible to bidirectional self-attention and may be revised at every
//! refinement step.
//!
//! Observed fragment peaks are always soft evidence. They never determine
//! whether a chemical state exists. Precursor-mass consistency is also soft in
//! noisy/intermediate states; exact mass is required only for accepted final
//! hypotheses.

use super::chemistry::{common_unimod_definition, residue_graph, ATOM_FEATURE_DIM};
use super::chemistry_decoder::{
    FOUNDATION_CHEMISTRY_FRAGMENT_ABS_TOLERANCE_DA_V0200, FOUNDATION_CHEMISTRY_FRAGMENT_PPM_V0200,
    FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
};
use super::diffusion::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, FoundationDiffusionBatch, FoundationDiffusionConfig,
    FoundationDiffusionOutput, PeptideSpectrumDiffusionModel, FOUNDATION_DIFFUSION_CARBAMIDOMETHYL,
    FOUNDATION_DIFFUSION_DEAMIDATED, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_OXIDATION, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIFFUSION_PHOSPHO, FOUNDATION_DIFFUSION_RESIDUE_ACETYL,
    FOUNDATION_DIFFUSION_VOCAB_SIZE, FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use super::experiment::FoundationPartition;
use super::model::PrecursorContextBatch;
use super::spectrum::{FoundationSpectrum, FoundationSpectrumBatch};
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder, VarMap};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;

/// Frozen v0.21 architecture identifier.
pub const FOUNDATION_CHEMISTRY_DIFFUSION_ARCHITECTURE_V0210: &str =
    "bidirectional_x0_diffusion_plus_v0200_chemistry_from_current_full_hypothesis";
/// Frozen v0.21 objective identifier.
pub const FOUNDATION_CHEMISTRY_DIFFUSION_OBJECTIVE_V0210: &str =
    "categorical_x0_ce_plus_matched_shuffled_spectrum_guard_v0210";
/// Fixed number of whole-sequence refinement passes in the first experiment.
pub const FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_STEPS_V0210: usize = 8;
/// Frozen starting noise level for repair of an intact v0.20 hypothesis.
/// At the v0.21 20-step beta schedule, t=8 corresponds to about 49% cumulative
/// replacement probability: enough revision pressure without pretending the
/// already-formed v0.20 peptide is a near-pure-noise t=20 sample.
pub const FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210: usize = 8;
/// Fixed v0.20 beam width used only to obtain the frozen initialization.
pub const FOUNDATION_CHEMISTRY_DIFFUSION_INITIAL_BEAM_WIDTH_V0210: usize = 128;
/// New whole-hypothesis mass state appended beside the preserved v0.20 44 features.
pub const FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210: usize = 4;

const PROTON_MASS_DA: f64 = 1.007_276_466_77;
const CARBON_MONOXIDE_MASS_DA: f64 = 27.994_914_62;
const AMMONIA_MASS_DA: f64 = 17.026_549_10;

#[derive(Debug, Clone, Default)]
struct TokenChemistrySummaryV0210 {
    atom_mean: [f32; ATOM_FEATURE_DIM],
    atom_count: f32,
    bond_count: f32,
    composition: [f32; 8],
}

/// Tensorized v0.21 candidate-token chemistry features.
#[derive(Debug, Clone)]
pub struct ChemistryDiffusionFeatureBatch {
    /// Preserved v0.20-compatible candidate-transition features
    /// `[batch, max_tokens, vocabulary, 44]`.
    pub features: Tensor,
    /// New full-hypothesis mass/suffix context
    /// `[batch, max_tokens, vocabulary, 4]`.
    pub full_state_features: Tensor,
}

/// CPU-side full-hypothesis chemistry featurizer.
#[derive(Debug, Clone)]
pub struct ChemistryDiffusionFeaturizer {
    max_tokens: usize,
    max_peaks: usize,
    token_summaries: Vec<TokenChemistrySummaryV0210>,
}

impl ChemistryDiffusionFeaturizer {
    /// Construct the fixed v0.21 featurizer. The feature width intentionally
    /// remains 44 so the learned v0.20 chemistry scorer can be warm-started.
    pub fn new(config: &FoundationDiffusionConfig) -> std::result::Result<Self, String> {
        config.validate()?;
        Ok(Self {
            max_tokens: config.max_tokens,
            max_peaks: config.spectrum.max_peaks,
            token_summaries: (0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
                .map(token_chemistry_summary_v0210)
                .collect(),
        })
    }

    /// Recompute candidate chemistry/mass/fragment features from complete noisy
    /// hypotheses. Unknown/MASK states contribute no residue mass; therefore
    /// mass inconsistency is informative but never a hard early-state barrier.
    pub fn featurize(
        &self,
        noisy_rows: &[Vec<u32>],
        active_lengths: &[usize],
        spectra: &[FoundationSpectrum],
        precursor_masses: &[f64],
        charges: &[i32],
        device: &Device,
    ) -> Result<ChemistryDiffusionFeatureBatch> {
        let batch = noisy_rows.len();
        if batch == 0
            || active_lengths.len() != batch
            || spectra.len() != batch
            || precursor_masses.len() != batch
            || charges.len() != batch
        {
            candle_core::bail!("v0.21 chemistry diffusion inputs have inconsistent batch lengths");
        }
        let rows = (0..batch)
            .into_par_iter()
            .map(|index| {
                if noisy_rows[index].len() != self.max_tokens {
                    return Err(format!(
                        "v0.21 noisy row {} width {} != {}",
                        index,
                        noisy_rows[index].len(),
                        self.max_tokens
                    ));
                }
                let active = active_lengths[index];
                if active == 0 || active > self.max_tokens {
                    return Err(format!(
                        "v0.21 active length {active} outside configured range"
                    ));
                }
                let evidence = PreparedSpectrumV0210::new(&spectra[index], self.max_peaks)?;
                self.row_features(
                    &noisy_rows[index],
                    active,
                    &evidence,
                    precursor_masses[index],
                    charges[index],
                )
            })
            .collect::<std::result::Result<Vec<_>, String>>()
            .map_err(candle_core::Error::Msg)?;
        let mut transition_flat = Vec::new();
        let mut full_state_flat = Vec::new();
        for (transition, full_state) in rows {
            transition_flat.extend(transition);
            full_state_flat.extend(full_state);
        }
        Ok(ChemistryDiffusionFeatureBatch {
            features: Tensor::from_vec(
                transition_flat,
                (
                    batch,
                    self.max_tokens,
                    FOUNDATION_DIFFUSION_VOCAB_SIZE,
                    FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
                ),
                device,
            )?,
            full_state_features: Tensor::from_vec(
                full_state_flat,
                (
                    batch,
                    self.max_tokens,
                    FOUNDATION_DIFFUSION_VOCAB_SIZE,
                    FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210,
                ),
                device,
            )?,
        })
    }

    fn row_features(
        &self,
        row: &[u32],
        active: usize,
        evidence: &PreparedSpectrumV0210,
        precursor_mass: f64,
        charge: i32,
    ) -> std::result::Result<(Vec<f32>, Vec<f32>), String> {
        if !(precursor_mass.is_finite() && precursor_mass > FOUNDATION_PEPTIDE_WATER_MASS_DA) {
            return Err("v0.21 requires finite measured precursor neutral mass".into());
        }
        let mut prefix_known = vec![0.0f64; active + 1];
        for position in 0..active {
            prefix_known[position + 1] =
                prefix_known[position] + clean_token_mass_or_zero(row[position]);
        }
        let mut suffix_known = vec![0.0f64; active + 1];
        for position in (0..active).rev() {
            suffix_known[position] =
                suffix_known[position + 1] + clean_token_mass_or_zero(row[position]);
        }
        let current_known_total = FOUNDATION_PEPTIDE_WATER_MASS_DA + prefix_known[active];
        let mut transition_output = vec![
            0.0f32;
            self.max_tokens
                * FOUNDATION_DIFFUSION_VOCAB_SIZE
                * FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200
        ];
        let mut full_state_output = vec![
            0.0f32;
            self.max_tokens
                * FOUNDATION_DIFFUSION_VOCAB_SIZE
                * FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210
        ];
        for position in 0..active {
            for token in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
                let mut feature = [0.0f32; FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200];
                let mut full_state =
                    [0.0f32; FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210];
                let candidate_mass = foundation_diffusion_token_mass_da(token).unwrap_or(0.0);
                let prefix_mass = FOUNDATION_PEPTIDE_WATER_MASS_DA + prefix_known[position];
                let new_prefix_mass = if token == FOUNDATION_DIFFUSION_EOS {
                    prefix_mass
                } else {
                    prefix_mass + candidate_mass
                };
                let suffix_mass = suffix_known[position + 1];
                let candidate_total = new_prefix_mass + suffix_mass;
                let residual_before = precursor_mass - prefix_mass;
                let residual_after = precursor_mass - new_prefix_mass;
                let current_error = precursor_mass - current_known_total;
                let candidate_error = precursor_mass - candidate_total;
                let final_position = position + 1 == active;
                let grammar = candidate_grammar_status(row, active, position, token);

                // Preserve the numerical semantics of v0.20 features 0..9 and
                // 11..43 wherever whole-sequence denoising permits it. Feature
                // 10 was the hard future-suffix-feasible flag in v0.20; here it
                // becomes a soft grammar/plausibility input and never gates a
                // state. Full suffix/current-state mass information is carried
                // separately by the new zero-initialized four-feature head.
                feature[0] = (candidate_mass / 250.0).clamp(-2.0, 2.0) as f32;
                feature[1] = (prefix_mass / precursor_mass).clamp(0.0, 1.5) as f32;
                feature[2] = (new_prefix_mass / precursor_mass).clamp(0.0, 1.5) as f32;
                feature[3] = (residual_before / precursor_mass).clamp(-0.5, 1.5) as f32;
                feature[4] = (residual_after / precursor_mass).clamp(-0.5, 1.5) as f32;
                feature[5] =
                    (position as f64 / self.max_tokens.max(1) as f64).clamp(0.0, 1.0) as f32;
                feature[6] = if foundation_diffusion_token_residue(token).is_some() {
                    1.0
                } else {
                    0.0
                };
                feature[7] = if is_ptm_token(token) { 1.0 } else { 0.0 };
                feature[8] = if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
                    1.0
                } else {
                    0.0
                };
                feature[9] = if token == FOUNDATION_DIFFUSION_EOS {
                    1.0
                } else {
                    0.0
                };
                feature[10] = grammar;
                feature[11] = (charge as f32 / 6.0).clamp(0.0, 1.5);
                if let Some(summary) = self.token_summaries.get(token as usize) {
                    feature[12..24].copy_from_slice(&summary.atom_mean);
                    feature[24] = summary.atom_count;
                    feature[25] = summary.bond_count;
                    feature[26..34].copy_from_slice(&summary.composition);
                }

                // Explicit new full-hypothesis state. These values are valid
                // even when MASK/random categories make the current mass wrong;
                // they remain conditioning only until the final projection.
                full_state[0] = (suffix_mass / precursor_mass).clamp(0.0, 1.5) as f32;
                full_state[1] = (current_error / precursor_mass).clamp(-1.5, 1.5) as f32;
                full_state[2] = (candidate_error / precursor_mass).clamp(-1.5, 1.5) as f32;
                full_state[3] = (current_known_total / precursor_mass).clamp(0.0, 1.5) as f32;

                // Fragment evidence is evaluated at the cleavage after this
                // candidate position. It is deliberately zero when the state is
                // chemically undefined, not used as a feasibility gate.
                if grammar > 0.0 && !final_position && token != FOUNDATION_DIFFUSION_EOS {
                    let prefix_residue_mass =
                        (new_prefix_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA).max(0.0);
                    if prefix_residue_mass > 0.0 {
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
                }

                let transition_start = (position * FOUNDATION_DIFFUSION_VOCAB_SIZE
                    + token as usize)
                    * FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200;
                transition_output[transition_start
                    ..transition_start + FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200]
                    .copy_from_slice(&feature);
                let full_state_start = (position * FOUNDATION_DIFFUSION_VOCAB_SIZE
                    + token as usize)
                    * FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210;
                full_state_output[full_state_start
                    ..full_state_start
                        + FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210]
                    .copy_from_slice(&full_state);
            }
        }
        Ok((transition_output, full_state_output))
    }
}

/// v0.21 model: bidirectional spectrum-conditioned x0 denoiser plus the learned
/// v0.20 44-dimensional biochemical token scorer.
#[derive(Clone)]
pub struct PeptideSpectrumChemistryDiffusionModel {
    base: PeptideSpectrumDiffusionModel,
    hidden_to_features: Linear,
    feature_bias: Linear,
    full_state_hidden_to_features: Linear,
    full_state_bias: Linear,
}

impl PeptideSpectrumChemistryDiffusionModel {
    /// Construct the model under a namespace deliberately compatible with the
    /// v0.20 chemistry scorer for warm-starting.
    pub fn new(config: FoundationDiffusionConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let base = PeptideSpectrumDiffusionModel::new(config.clone(), vb.clone())?;
        let chemistry = vb.pp("chemistry_transition");
        let hidden_to_features = nn::linear(
            config.model_dim,
            FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            chemistry.pp("hidden_to_features"),
        )?;
        let feature_bias = nn::linear(
            FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            1,
            chemistry.pp("feature_bias"),
        )?;
        // Whole-hypothesis mass state is genuinely new in v0.21. Zero
        // initialization guarantees the warm-start prediction is unchanged
        // until training learns to use these features.
        let full_state = vb.pp("chemistry_diffusion");
        let full_hidden_vb = full_state.pp("full_state_hidden_to_features");
        let full_hidden_weight = full_hidden_vb.get_with_hints(
            (
                FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210,
                config.model_dim,
            ),
            "weight",
            nn::Init::Const(0.0),
        )?;
        let full_hidden_bias = full_hidden_vb.get_with_hints(
            FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210,
            "bias",
            nn::Init::Const(0.0),
        )?;
        let full_bias_vb = full_state.pp("full_state_bias");
        let full_bias_weight = full_bias_vb.get_with_hints(
            (
                1,
                FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210,
            ),
            "weight",
            nn::Init::Const(0.0),
        )?;
        let full_bias_value = full_bias_vb.get_with_hints(1, "bias", nn::Init::Const(0.0))?;
        Ok(Self {
            base,
            hidden_to_features,
            feature_bias,
            full_state_hidden_to_features: Linear::new(full_hidden_weight, Some(full_hidden_bias)),
            full_state_bias: Linear::new(full_bias_weight, Some(full_bias_value)),
        })
    }

    /// Predict x0 token logits at every peptide position from the complete noisy
    /// hypothesis, timestep, spectrum, precursor, and recomputed chemistry.
    pub fn forward_t(
        &self,
        diffusion: &FoundationDiffusionBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        chemistry: &ChemistryDiffusionFeatureBatch,
        train: bool,
    ) -> Result<FoundationDiffusionOutput> {
        let mut output = self.base.forward_t(diffusion, spectrum, precursor, train)?;
        let (batch, positions, classes) = output.token_logits.dims3()?;
        if chemistry.features.dims4()?
            != (
                batch,
                positions,
                classes,
                FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            )
        {
            candle_core::bail!("v0.21 chemistry feature shape does not match denoiser logits");
        }
        let coefficients = self.hidden_to_features.forward(&output.decoder_hidden)?;
        let dynamic = chemistry
            .features
            .broadcast_mul(&coefficients.unsqueeze(2)?)?
            .sum(3)?
            .affine(
                1.0 / (FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200 as f64).sqrt(),
                0.0,
            )?;
        let static_bias = self
            .feature_bias
            .forward(&chemistry.features.reshape((
                batch * positions * classes,
                FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
            ))?)?
            .reshape((batch, positions, classes))?;
        if chemistry.full_state_features.dims4()?
            != (
                batch,
                positions,
                classes,
                FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210,
            )
        {
            candle_core::bail!("v0.21 full-state feature shape does not match denoiser logits");
        }
        let full_coefficients = self
            .full_state_hidden_to_features
            .forward(&output.decoder_hidden)?;
        let full_dynamic = chemistry
            .full_state_features
            .broadcast_mul(&full_coefficients.unsqueeze(2)?)?
            .sum(3)?
            .affine(
                1.0 / (FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210 as f64).sqrt(),
                0.0,
            )?;
        let full_static = self
            .full_state_bias
            .forward(&chemistry.full_state_features.reshape((
                batch * positions * classes,
                FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210,
            ))?)?
            .reshape((batch, positions, classes))?;
        let transition = (&dynamic + &static_bias)?;
        let full_transition = (&full_dynamic + &full_static)?;
        output.token_logits = ((&output.token_logits + &transition)? + &full_transition)?;
        Ok(output)
    }

    pub fn config(&self) -> &FoundationDiffusionConfig {
        self.base.config()
    }
}

/// v0.20 -> v0.21 warm-start accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChemistryDiffusionWarmStartReport {
    pub shared_variables: usize,
    pub chemistry_variables: usize,
    pub new_diffusion_variables: usize,
    pub new_full_state_variables: usize,
    pub ignored_v0200_variables: usize,
}

/// Warm-start every shape-compatible v0.21 tensor from the trained v0.20
/// checkpoint. Timestep/length-head and whole-hypothesis-state parameters are
/// new; the causal START embedding is intentionally ignored.
pub fn load_chemistry_diffusion_from_v0200_checkpoint(
    varmap: &VarMap,
    path: &Path,
    device: &Device,
) -> Result<ChemistryDiffusionWarmStartReport> {
    let checkpoint = candle_core::safetensors::load(path, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("v0.21 VarMap lock poisoned".into()))?;
    let current_names: HashSet<String> = data.keys().cloned().collect();
    let mut shared_variables = 0usize;
    let mut chemistry_variables = 0usize;
    let mut new_diffusion_variables = 0usize;
    let mut new_full_state_variables = 0usize;
    let mut missing = Vec::<String>::new();
    for (name, variable) in data.iter() {
        if let Some(source) = checkpoint.get(name) {
            if variable.as_tensor().dims() != source.dims() {
                candle_core::bail!(
                    "v0.21 warm-start shape mismatch for '{name}': current {:?}, v0.20 {:?}",
                    variable.as_tensor().dims(),
                    source.dims()
                );
            }
            variable.set(source)?;
            if name.starts_with("chemistry_transition.") {
                chemistry_variables += 1;
            } else {
                shared_variables += 1;
            }
        } else if name.starts_with("decoder.timestep.") || name.starts_with("decoder.length_head.")
        {
            new_diffusion_variables += 1;
        } else if name.starts_with("chemistry_diffusion.") {
            new_full_state_variables += 1;
        } else {
            missing.push(name.clone());
        }
    }
    drop(data);
    if !missing.is_empty() {
        candle_core::bail!(
            "v0.20 checkpoint missing required v0.21 warm-start variables: {}",
            missing.join(", ")
        );
    }
    let ignored_v0200_variables = checkpoint
        .keys()
        .filter(|name| !current_names.contains(*name))
        .count();
    Ok(ChemistryDiffusionWarmStartReport {
        shared_variables,
        chemistry_variables,
        new_diffusion_variables,
        new_full_state_variables,
        ignored_v0200_variables,
    })
}

/// Validate that the optimizer and frozen evaluation selections are isolated
/// to TRAIN and VALIDATION respectively. TEST is never an allowed label.
pub fn foundation_chemistry_diffusion_partition_isolated(
    train_labels: &[FoundationPartition],
    validation_labels: &[FoundationPartition],
) -> bool {
    !train_labels.is_empty()
        && !validation_labels.is_empty()
        && train_labels
            .iter()
            .all(|&partition| partition == FoundationPartition::Train)
        && validation_labels
            .iter()
            .all(|&partition| partition == FoundationPartition::Validation)
}

/// Fixed decreasing noise-level schedule for repair of a v0.20 hypothesis.
///
/// The first experiment does not re-noise the v0.20 prediction. Starting at
/// t=20 would therefore present a nearly clean categorical row under a timestep
/// whose training distribution is ~98.6% corrupted. We instead freeze t=8 as
/// the first repair level and descend one level per whole-sequence pass.
pub fn foundation_chemistry_diffusion_refinement_timesteps(
    total_steps: usize,
) -> std::result::Result<Vec<usize>, String> {
    if total_steps == 0 {
        return Err("v0.21 refinement requires positive diffusion_steps".into());
    }
    let rounds = FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_STEPS_V0210;
    let start = FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210.min(total_steps);
    if start < rounds {
        return Err(format!(
            "v0.21 requires at least {rounds} diffusion timesteps for the frozen repair schedule"
        ));
    }
    Ok((1..=rounds).map(|offset| start + 1 - offset).collect())
}

/// Convert x0 logits into the next complete hypothesis while preserving the
/// initialization length. All non-final positions may change simultaneously;
/// EOS is structurally fixed at the final active position.
pub fn foundation_chemistry_diffusion_argmax_refine(
    current_rows: &[Vec<u32>],
    active_lengths: &[usize],
    logits: &[Vec<Vec<f32>>],
) -> std::result::Result<(Vec<Vec<u32>>, usize), String> {
    if current_rows.len() != active_lengths.len() || current_rows.len() != logits.len() {
        return Err("v0.21 refinement batch lengths differ".into());
    }
    let mut changed = 0usize;
    let mut output = current_rows.to_vec();
    for row_index in 0..current_rows.len() {
        let active = active_lengths[row_index];
        if active == 0 || active > current_rows[row_index].len() || logits[row_index].len() < active
        {
            return Err("v0.21 refinement active length is invalid".into());
        }
        for position in 0..active {
            let predicted = if position + 1 == active {
                FOUNDATION_DIFFUSION_EOS
            } else {
                best_clean_token_for_position(
                    &current_rows[row_index],
                    active,
                    position,
                    &logits[row_index][position],
                )?
            };
            changed += usize::from(predicted != current_rows[row_index][position]);
            output[row_index][position] = predicted;
        }
        for position in active..output[row_index].len() {
            output[row_index][position] = FOUNDATION_DIFFUSION_PAD;
        }
    }
    Ok((output, changed))
}

/// Hard final projection within a fixed Hamming radius of two positions.
///
/// This is not an autoregressive beam: every position is scored from the same
/// final bidirectional x0 prediction. The bounded projection only enforces the
/// physical precursor-mass/grammar contract after soft mass-conditioned
/// denoising. Radius two is frozen for v0.21 and is not a search hyperparameter.
pub fn foundation_chemistry_diffusion_project_mass_valid(
    current: &[u32],
    active_length: usize,
    logits: &[Vec<f32>],
    precursor_mass: f64,
    tolerance_da: f64,
) -> std::result::Result<Option<Vec<u32>>, String> {
    if active_length < 2 || active_length > current.len() || logits.len() < active_length {
        return Err("v0.21 mass projection received invalid row dimensions".into());
    }
    if logits[..active_length]
        .iter()
        .any(|row| row.len() != FOUNDATION_DIFFUSION_VOCAB_SIZE)
    {
        return Err("v0.21 mass projection logit width mismatch".into());
    }
    let base_mass = relaxed_row_mass(current, active_length)?;
    let mut best: Option<(f64, Vec<u32>)> = None;
    let mut consider = |candidate: Vec<u32>| {
        if !foundation_chemistry_diffusion_final_mass_valid(
            &candidate,
            active_length,
            precursor_mass,
            tolerance_da,
        ) {
            return;
        }
        let score = (0..active_length)
            .map(|position| logits[position][candidate[position] as usize] as f64)
            .sum::<f64>();
        if score.is_finite() && best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
            best = Some((score, candidate));
        }
    };

    if (base_mass - precursor_mass).abs() <= tolerance_da {
        consider(current.to_vec());
    }
    let last_editable = active_length - 1;
    for i in 0..last_editable {
        let old_i = foundation_diffusion_token_mass_da(current[i]).unwrap_or(0.0);
        for token_i in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
            if !is_projectable_clean_token(token_i) || token_i == current[i] {
                continue;
            }
            let Some(mass_i) = foundation_diffusion_token_mass_da(token_i) else {
                continue;
            };
            let mass_one = base_mass - old_i + mass_i;
            if (mass_one - precursor_mass).abs() <= tolerance_da {
                let mut candidate = current.to_vec();
                candidate[i] = token_i;
                consider(candidate);
            }
            for j in i + 1..last_editable {
                let old_j = foundation_diffusion_token_mass_da(current[j]).unwrap_or(0.0);
                for token_j in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
                    if !is_projectable_clean_token(token_j) || token_j == current[j] {
                        continue;
                    }
                    let Some(mass_j) = foundation_diffusion_token_mass_da(token_j) else {
                        continue;
                    };
                    let mass_two = mass_one - old_j + mass_j;
                    if (mass_two - precursor_mass).abs() > tolerance_da {
                        continue;
                    }
                    let mut candidate = current.to_vec();
                    candidate[i] = token_i;
                    candidate[j] = token_j;
                    consider(candidate);
                }
            }
        }
    }
    Ok(best.map(|(_, row)| row))
}

fn relaxed_row_mass(row: &[u32], active_length: usize) -> std::result::Result<f64, String> {
    let mut mass = FOUNDATION_PEPTIDE_WATER_MASS_DA;
    for &token in &row[..active_length - 1] {
        let Some(value) = foundation_diffusion_token_mass_da(token) else {
            return Err(format!(
                "v0.21 projection row contains non-mass token {token}"
            ));
        };
        mass += value;
    }
    Ok(mass)
}

fn is_projectable_clean_token(token: u32) -> bool {
    foundation_diffusion_token_mass_da(token).is_some()
        && token != FOUNDATION_DIFFUSION_PAD
        && token != FOUNDATION_DIFFUSION_MASK
        && token != FOUNDATION_DIFFUSION_EOS
}

/// Exact neutral mass of a clean active token row. EOS contributes zero mass.
pub fn foundation_chemistry_diffusion_row_neutral_mass(
    row: &[u32],
    active_length: usize,
) -> std::result::Result<f64, String> {
    if active_length == 0 || active_length > row.len() {
        return Err("v0.21 row active length is invalid".into());
    }
    let mut mass = FOUNDATION_PEPTIDE_WATER_MASS_DA;
    for (position, &token) in row[..active_length].iter().enumerate() {
        if position + 1 == active_length {
            if token != FOUNDATION_DIFFUSION_EOS {
                return Err("v0.21 clean row must terminate with EOS".into());
            }
            continue;
        }
        let token_mass = foundation_diffusion_token_mass_da(token)
            .ok_or_else(|| format!("v0.21 clean row contains non-mass token {token}"))?;
        mass += token_mass;
    }
    Ok(mass)
}

/// Validate clean token grammar and final precursor mass.
pub fn foundation_chemistry_diffusion_final_mass_valid(
    row: &[u32],
    active_length: usize,
    precursor_mass: f64,
    tolerance_da: f64,
) -> bool {
    if !(precursor_mass.is_finite() && tolerance_da.is_finite() && tolerance_da > 0.0)
        || active_length == 0
        || active_length > row.len()
        || row[active_length - 1] != FOUNDATION_DIFFUSION_EOS
    {
        return false;
    }
    let mut saw_residue = false;
    for position in 0..active_length - 1 {
        let token = row[position];
        if token == FOUNDATION_DIFFUSION_PAD
            || token == FOUNDATION_DIFFUSION_MASK
            || token == FOUNDATION_DIFFUSION_EOS
        {
            return false;
        }
        if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
            if position != 0 {
                return false;
            }
            continue;
        }
        if foundation_diffusion_token_residue(token).is_some() {
            saw_residue = true;
            continue;
        }
        if is_residue_local_ptm(token) {
            if position == 0 {
                return false;
            }
            let Some(previous_residue) = foundation_diffusion_token_residue(row[position - 1])
            else {
                return false;
            };
            if !foundation_diffusion_residue_ptm_valid(token, previous_residue) {
                return false;
            }
            continue;
        }
        return false;
    }
    if !saw_residue {
        return false;
    }
    foundation_chemistry_diffusion_row_neutral_mass(row, active_length)
        .map(|mass| (mass - precursor_mass).abs() <= tolerance_da)
        .unwrap_or(false)
}

fn best_clean_token_for_position(
    _current: &[u32],
    _active: usize,
    position: usize,
    logits: &[f32],
) -> std::result::Result<u32, String> {
    if logits.len() != FOUNDATION_DIFFUSION_VOCAB_SIZE {
        return Err("v0.21 logit row has wrong vocabulary width".into());
    }
    let mut best = None::<(f32, u32)>;
    for token in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
        if token == FOUNDATION_DIFFUSION_PAD
            || token == FOUNDATION_DIFFUSION_MASK
            || token == FOUNDATION_DIFFUSION_EOS
        {
            continue;
        }
        // Intermediate refinement must remain jointly revisable. Do not hard
        // gate a residue-local PTM on the *current* previous residue, because
        // that previous position may be repaired in the same pass. Only the
        // sequence-position semantics that cannot be repaired elsewhere are
        // structural here; complete grammar is hard-checked at the end.
        if token == FOUNDATION_DIFFUSION_NTERM_ACETYL && position != 0 {
            continue;
        }
        if is_residue_local_ptm(token) && position == 0 {
            continue;
        }
        let value = logits[token as usize];
        if !value.is_finite() {
            continue;
        }
        if best.map(|(score, _)| value > score).unwrap_or(true) {
            best = Some((value, token));
        }
    }
    best.map(|(_, token)| token)
        .ok_or_else(|| "v0.21 no clean token available at active position".into())
}

fn candidate_grammar_status(row: &[u32], active: usize, position: usize, token: u32) -> f32 {
    if token == FOUNDATION_DIFFUSION_PAD || token == FOUNDATION_DIFFUSION_MASK {
        return 0.0;
    }
    if position + 1 == active {
        return if token == FOUNDATION_DIFFUSION_EOS {
            1.0
        } else {
            0.0
        };
    }
    if token == FOUNDATION_DIFFUSION_EOS {
        return 0.0;
    }
    if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
        return if position == 0 { 1.0 } else { 0.0 };
    }
    if foundation_diffusion_token_residue(token).is_some() {
        return 1.0;
    }
    if is_residue_local_ptm(token) {
        if position == 0 {
            return 0.0;
        }
        let previous = row[position - 1];
        let Some(residue) = foundation_diffusion_token_residue(previous) else {
            // The previous category can itself be corrupted and jointly
            // repaired. Treat local PTM compatibility as uncertain rather
            // than impossible in intermediate states.
            return 0.5;
        };
        return if foundation_diffusion_residue_ptm_valid(token, residue) {
            1.0
        } else {
            0.25
        };
    }
    0.0
}

fn clean_token_mass_or_zero(token: u32) -> f64 {
    foundation_diffusion_token_mass_da(token).unwrap_or(0.0)
}

fn is_residue_local_ptm(token: u32) -> bool {
    matches!(
        token,
        FOUNDATION_DIFFUSION_RESIDUE_ACETYL
            | FOUNDATION_DIFFUSION_CARBAMIDOMETHYL
            | FOUNDATION_DIFFUSION_DEAMIDATED
            | FOUNDATION_DIFFUSION_OXIDATION
            | FOUNDATION_DIFFUSION_PHOSPHO
    )
}

fn is_ptm_token(token: u32) -> bool {
    token == FOUNDATION_DIFFUSION_NTERM_ACETYL || is_residue_local_ptm(token)
}

fn token_chemistry_summary_v0210(token: u32) -> TokenChemistrySummaryV0210 {
    let mut summary = TokenChemistrySummaryV0210::default();
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
    let unimod = match token {
        FOUNDATION_DIFFUSION_NTERM_ACETYL | FOUNDATION_DIFFUSION_RESIDUE_ACETYL => Some(1),
        FOUNDATION_DIFFUSION_CARBAMIDOMETHYL => Some(4),
        FOUNDATION_DIFFUSION_DEAMIDATED => Some(7),
        FOUNDATION_DIFFUSION_OXIDATION => Some(35),
        FOUNDATION_DIFFUSION_PHOSPHO => Some(21),
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

#[derive(Debug, Clone)]
struct PreparedSpectrumV0210 {
    peaks: Vec<(f64, f32)>,
}

impl PreparedSpectrumV0210 {
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
            return Err("v0.21 spectrum contains no finite positive observed peaks".into());
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
