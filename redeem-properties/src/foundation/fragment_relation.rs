//! v0.24 hard-negative fragment-to-peak relational energy.
//!
//! v0.23 established that the frozen peptide->MS2 model contains strong
//! spectrum-specific biochemical information, but an absolute predicted-vs-
//! observed cosine cannot order difficult same-spectrum, mass-compatible
//! alternatives.  v0.24 therefore exposes the evidence at the level where the
//! ambiguity actually lives: each peptide bond / core b/y ion is related to the
//! nearby measured peak, its mass error, intensity, forward-predicted intensity,
//! complementary-ion support, local residue context, modification state, and
//! the number of competing peptide hypotheses claiming the same observed peak.
//!
//! The trainable model is intentionally small.  It does not receive target
//! labels, legacy candidate scores, or proposal ranks as input features.  TRAIN
//! labels are used only by the same-spectrum listwise objective in the caller.
//! Validation labels are consulted only after scores have been frozen.

use super::diffusion::{
    foundation_diffusion_token_mass_da, foundation_diffusion_token_residue,
    FoundationDiffusionVocabulary, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use super::featurize::{FoundationModificationSite, PeptidoformInput};
use super::spectrum::{FoundationSpectrum, FoundationSpectrumPeak};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};
use std::collections::HashSet;

/// Stable architecture identifier for the bounded v0.24 experiment.
pub const FOUNDATION_FRAGMENT_RELATION_ARCHITECTURE_V0240: &str =
    "same_spectrum_hard_negative_cleavage_peak_relational_residual_energy";
/// Frozen v0.24 training objective.
pub const FOUNDATION_FRAGMENT_RELATION_OBJECTIVE_V0240: &str =
    "positive_first_32way_listwise_fixed_negative_log_legacy_rank_prior_plus_fragment_peak_relation_residual";
/// Four unambiguous core forward-MS2 channels: b1, b2, y1, y2.
pub const FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240: usize = 4;
/// Fixed ppm tolerance inherited from v0.23.
pub const FOUNDATION_FRAGMENT_RELATION_PPM_V0240: f64 = 20.0;
/// Fixed absolute tolerance floor inherited from v0.23.
pub const FOUNDATION_FRAGMENT_RELATION_ABS_TOLERANCE_DA_V0240: f64 = 0.02;
/// Peak cap retained from the inverse/forward spectrum representation.
pub const FOUNDATION_FRAGMENT_RELATION_MAX_PEAKS_V0240: usize = 256;
/// Number of local residue classes (20 canonical + unknown).
pub const FOUNDATION_FRAGMENT_RELATION_RESIDUE_CLASSES_V0240: usize = 21;
/// Explicit CPU-side relation feature width.
pub const FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240: usize = 77;
/// First relation MLP width.
pub const FOUNDATION_FRAGMENT_RELATION_HIDDEN_V0240: usize = 64;
/// Pooled relation width.
pub const FOUNDATION_FRAGMENT_RELATION_POOLED_V0240: usize = 32;
/// Candidate head hidden width.
pub const FOUNDATION_FRAGMENT_RELATION_CANDIDATE_HIDDEN_V0240: usize = 16;

/// Feature offsets are public so focused regression tests can assert that the
/// candidate-relative competition terms are actually present.
pub const FOUNDATION_FRAGMENT_RELATION_LEFT_AA_OFFSET_V0240: usize = 0;
pub const FOUNDATION_FRAGMENT_RELATION_RIGHT_AA_OFFSET_V0240: usize = 21;
pub const FOUNDATION_FRAGMENT_RELATION_LEFT_MOD_MASS_V0240: usize = 42;
pub const FOUNDATION_FRAGMENT_RELATION_RIGHT_MOD_MASS_V0240: usize = 43;
pub const FOUNDATION_FRAGMENT_RELATION_LEFT_MOD_FLAG_V0240: usize = 44;
pub const FOUNDATION_FRAGMENT_RELATION_RIGHT_MOD_FLAG_V0240: usize = 45;
pub const FOUNDATION_FRAGMENT_RELATION_PREDICTED_OFFSET_V0240: usize = 46;
pub const FOUNDATION_FRAGMENT_RELATION_OBSERVED_OFFSET_V0240: usize = 50;
pub const FOUNDATION_FRAGMENT_RELATION_MASS_ERROR_OFFSET_V0240: usize = 54;
pub const FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240: usize = 58;
pub const FOUNDATION_FRAGMENT_RELATION_INVERSE_CLAIM_OFFSET_V0240: usize = 62;
pub const FOUNDATION_FRAGMENT_RELATION_COMPLEMENT_OBS_OFFSET_V0240: usize = 66;
pub const FOUNDATION_FRAGMENT_RELATION_COMPLEMENT_PRED_OFFSET_V0240: usize = 68;
pub const FOUNDATION_FRAGMENT_RELATION_POSITION_V0240: usize = 70;
pub const FOUNDATION_FRAGMENT_RELATION_REVERSE_POSITION_V0240: usize = 71;
pub const FOUNDATION_FRAGMENT_RELATION_LENGTH_V0240: usize = 72;
pub const FOUNDATION_FRAGMENT_RELATION_PEAK_COVERAGE_V0240: usize = 73;
pub const FOUNDATION_FRAGMENT_RELATION_EXPLAINED_INTENSITY_V0240: usize = 74;
pub const FOUNDATION_FRAGMENT_RELATION_MEAN_COMPETITION_V0240: usize = 75;
pub const FOUNDATION_FRAGMENT_RELATION_PRECURSOR_ERROR_V0240: usize = 76;

const PROTON_MASS_DA: f64 = 1.007_276_466_77;

/// Fixed, label-free proposal prior used by the v0.24 residual energy.
///
/// The accepted v0.13.23 ordering is converted to a Plackett-Luce-like
/// log-prior without fitting any validation-set coefficient.  Rank one is zero;
/// lower-ranked candidates receive increasingly negative prior energy.
pub fn foundation_fragment_relation_legacy_log_prior(rank: usize) -> f32 {
    if rank == 0 || rank == usize::MAX {
        return -20.0;
    }
    -(rank as f32).ln()
}

/// CPU feature rows for one same-spectrum candidate group.
#[derive(Debug, Clone)]
pub struct FoundationFragmentRelationFeatureRows {
    /// Flat `[candidates, max_cleavages, feature_dim]` payload.
    pub features: Vec<f32>,
    /// Flat `[candidates, max_cleavages]` cleavage mask.
    pub mask: Vec<f32>,
    /// Number of peptide hypotheses in this same-spectrum group.
    pub candidates: usize,
    /// Padded cleavage width (normally 63 for a 64-residue canvas).
    pub max_cleavages: usize,
    /// Number of core fragment relations with observed support.
    pub matched_relations: usize,
    /// Number of distinct retained peaks claimed by at least two candidates.
    pub contested_peaks: usize,
}

impl FoundationFragmentRelationFeatureRows {
    /// Materialize the CPU rows as Candle tensors.
    pub fn to_batch(&self, device: &Device) -> Result<FoundationFragmentRelationBatch> {
        if self.candidates == 0 || self.max_cleavages == 0 {
            candle_core::bail!("v0.24 relation batch requires non-empty candidates/cleavages");
        }
        let expected =
            self.candidates * self.max_cleavages * FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240;
        if self.features.len() != expected {
            candle_core::bail!(
                "v0.24 relation feature payload length {} != expected {expected}",
                self.features.len()
            );
        }
        if self.mask.len() != self.candidates * self.max_cleavages {
            candle_core::bail!("v0.24 relation mask payload length mismatch");
        }
        Ok(FoundationFragmentRelationBatch {
            features: Tensor::from_vec(
                self.features.clone(),
                (
                    self.candidates,
                    self.max_cleavages,
                    FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240,
                ),
                device,
            )?,
            mask: Tensor::from_vec(
                self.mask.clone(),
                (self.candidates, self.max_cleavages),
                device,
            )?,
        })
    }
}

/// Tensorized v0.24 fragment/peak relations.
#[derive(Debug, Clone)]
pub struct FoundationFragmentRelationBatch {
    /// `[candidates, cleavages, 77]` explicit biochemical relations.
    pub features: Tensor,
    /// `[candidates, cleavages]` valid-cleavage mask.
    pub mask: Tensor,
}

impl FoundationFragmentRelationBatch {
    /// Concatenate independent same-spectrum groups along the candidate axis.
    pub fn cat(groups: &[FoundationFragmentRelationFeatureRows], device: &Device) -> Result<Self> {
        if groups.is_empty() {
            candle_core::bail!("v0.24 relation batch concatenation requires at least one group");
        }
        let max_cleavages = groups[0].max_cleavages;
        let mut features = Vec::<f32>::new();
        let mut mask = Vec::<f32>::new();
        let mut candidates = 0usize;
        for group in groups {
            if group.max_cleavages != max_cleavages {
                candle_core::bail!("v0.24 relation groups use inconsistent cleavage widths");
            }
            features.extend_from_slice(&group.features);
            mask.extend_from_slice(&group.mask);
            candidates += group.candidates;
        }
        FoundationFragmentRelationFeatureRows {
            features,
            mask,
            candidates,
            max_cleavages,
            matched_relations: 0,
            contested_peaks: 0,
        }
        .to_batch(device)
    }
}

/// Small trainable relation model.  Every cleavage is first represented from
/// explicit fragment↔peak evidence, then pooled to one complete-candidate
/// energy.  There is no black-box spectrum encoder or candidate rank input.
#[derive(Clone)]
pub struct PeptideSpectrumFragmentRelationEnergy {
    relation_in: Linear,
    relation_hidden: Linear,
    candidate_hidden: Linear,
    candidate_out: Linear,
}

impl PeptideSpectrumFragmentRelationEnergy {
    /// Construct the isolated v0.24 relation head.
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let root = vb.pp("fragment_relation_v0240");
        Ok(Self {
            relation_in: nn::linear(
                FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240,
                FOUNDATION_FRAGMENT_RELATION_HIDDEN_V0240,
                root.pp("relation_in"),
            )?,
            relation_hidden: nn::linear(
                FOUNDATION_FRAGMENT_RELATION_HIDDEN_V0240,
                FOUNDATION_FRAGMENT_RELATION_POOLED_V0240,
                root.pp("relation_hidden"),
            )?,
            candidate_hidden: nn::linear(
                FOUNDATION_FRAGMENT_RELATION_POOLED_V0240,
                FOUNDATION_FRAGMENT_RELATION_CANDIDATE_HIDDEN_V0240,
                root.pp("candidate_hidden"),
            )?,
            candidate_out: nn::linear(
                FOUNDATION_FRAGMENT_RELATION_CANDIDATE_HIDDEN_V0240,
                1,
                root.pp("candidate_out"),
            )?,
        })
    }

    /// Score candidates and reshape them into same-spectrum groups.
    ///
    /// `candidates_per_group` is fixed by the caller during TRAIN (32) and may
    /// equal the full proposal-group width at validation.  The score is learned
    /// only from explicit relation features; legacy ranks remain diagnostics.
    pub fn forward_grouped(
        &self,
        batch: &FoundationFragmentRelationBatch,
        candidates_per_group: usize,
    ) -> Result<Tensor> {
        if candidates_per_group == 0 {
            candle_core::bail!("v0.24 candidates_per_group must be positive");
        }
        let (candidate_count, cleavage_count, feature_dim) = batch.features.dims3()?;
        if feature_dim != FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240 {
            candle_core::bail!(
                "v0.24 relation feature width {feature_dim} != {}",
                FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240
            );
        }
        if candidate_count % candidates_per_group != 0 {
            candle_core::bail!(
                "v0.24 candidate count {candidate_count} not divisible by group width {candidates_per_group}"
            );
        }
        let flat = batch
            .features
            .reshape((candidate_count * cleavage_count, feature_dim))?
            .contiguous()?;
        let hidden = self.relation_in.forward(&flat)?.relu()?;
        let hidden = self.relation_hidden.forward(&hidden)?.relu()?.reshape((
            candidate_count,
            cleavage_count,
            FOUNDATION_FRAGMENT_RELATION_POOLED_V0240,
        ))?;
        let mask = batch.mask.unsqueeze(2)?.broadcast_as((
            candidate_count,
            cleavage_count,
            FOUNDATION_FRAGMENT_RELATION_POOLED_V0240,
        ))?;
        let pooled_sum = hidden.broadcast_mul(&mask)?.sum(1)?;
        let denominator = batch.mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
        let pooled = pooled_sum.broadcast_div(&denominator)?;
        let hidden = self.candidate_hidden.forward(&pooled)?.relu()?;
        let score = self.candidate_out.forward(&hidden)?.squeeze(1)?;
        score.reshape((candidate_count / candidates_per_group, candidates_per_group))
    }
}

#[derive(Debug, Clone, Copy)]
struct NormalizedPeak {
    mz: f64,
    intensity: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PeakMatch {
    peak_index: Option<usize>,
    observed: f32,
    normalized_error: f32,
}

/// Build candidate-relative cleavage/peak relations for one observed spectrum.
///
/// This function is target-label-free.  Candidate competition is computed only
/// from theoretical ion claims made by the supplied hypothesis set.
pub fn foundation_fragment_relation_features(
    peptides: &[PeptidoformInput],
    spectrum: &FoundationSpectrum,
    predicted_ms2: &[Vec<Vec<f32>>],
    precursor_mass_errors_da: &[f64],
    max_cleavages: usize,
) -> std::result::Result<FoundationFragmentRelationFeatureRows, String> {
    if peptides.is_empty() {
        return Err("v0.24 relation features require at least one candidate".into());
    }
    if predicted_ms2.len() != peptides.len() || precursor_mass_errors_da.len() != peptides.len() {
        return Err("v0.24 relation candidate/prediction/mass-error length mismatch".into());
    }
    if max_cleavages == 0 {
        return Err("v0.24 max_cleavages must be positive".into());
    }

    let peaks = normalized_retained_peaks(spectrum);
    if peaks.is_empty() {
        return Err("v0.24 relation spectrum has no finite positive peaks".into());
    }

    let mut all_matches =
        Vec::<Vec<[PeakMatch; FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240]>>::with_capacity(
            peptides.len(),
        );
    let mut prefix_masses = Vec::<Vec<f64>>::with_capacity(peptides.len());

    for (candidate_index, peptide) in peptides.iter().enumerate() {
        let prefixes = cleavage_prefix_masses(peptide)?;
        if prefixes.len() > max_cleavages {
            return Err(format!(
                "v0.24 candidate {candidate_index} has {} cleavages > padded width {max_cleavages}",
                prefixes.len()
            ));
        }
        if predicted_ms2[candidate_index].len() < prefixes.len() {
            return Err(format!(
                "v0.24 candidate {candidate_index} forward rows {} < cleavage count {}",
                predicted_ms2[candidate_index].len(),
                prefixes.len()
            ));
        }
        let total = peptide_residue_mass(peptide)?;
        let mut candidate_matches = Vec::with_capacity(prefixes.len());
        for &prefix_mass in &prefixes {
            let suffix_with_water = total - prefix_mass + FOUNDATION_PEPTIDE_WATER_MASS_DA;
            let mz = core_fragment_mz(prefix_mass, suffix_with_water);
            let matches = [
                best_peak_match(mz[0], &peaks),
                best_peak_match(mz[1], &peaks),
                best_peak_match(mz[2], &peaks),
                best_peak_match(mz[3], &peaks),
            ];
            candidate_matches.push(matches);
        }
        prefix_masses.push(prefixes);
        all_matches.push(candidate_matches);
    }

    // Count distinct candidate hypotheses claiming each retained peak.  Multiple
    // ions from the same candidate do not inflate competition.
    let mut claim_counts = vec![0usize; peaks.len()];
    for candidate_matches in &all_matches {
        let mut claimed = HashSet::<usize>::new();
        for cleavage in candidate_matches {
            for relation in cleavage {
                if let Some(index) = relation.peak_index {
                    claimed.insert(index);
                }
            }
        }
        for index in claimed {
            claim_counts[index] += 1;
        }
    }
    let contested_peaks = claim_counts.iter().filter(|&&count| count >= 2).count();

    let total_observed_intensity = peaks
        .iter()
        .map(|peak| peak.intensity)
        .sum::<f64>()
        .max(f64::EPSILON);
    let mut features =
        vec![
            0.0f32;
            peptides.len() * max_cleavages * FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240
        ];
    let mut mask = vec![0.0f32; peptides.len() * max_cleavages];
    let mut matched_relations = 0usize;

    for candidate_index in 0..peptides.len() {
        let peptide = &peptides[candidate_index];
        let chars = peptide.sequence.chars().collect::<Vec<_>>();
        if chars.len().saturating_sub(1) != prefix_masses[candidate_index].len() {
            return Err(format!(
                "v0.24 candidate {candidate_index} sequence/cleavage count mismatch"
            ));
        }
        let (mod_mass, mod_flag) = residue_modification_state(peptide, chars.len());
        let predicted = &predicted_ms2[candidate_index];
        let candidate_matches = &all_matches[candidate_index];

        let mut pred_max = 0.0f64;
        for row in predicted.iter().take(candidate_matches.len()) {
            for channel in 0..FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240 {
                if let Some(value) = row.get(channel) {
                    pred_max = pred_max.max(f64::from(*value).max(0.0));
                }
            }
        }
        pred_max = pred_max.max(f64::EPSILON);

        let mut unique_peaks = HashSet::<usize>::new();
        let mut explained_intensity = 0.0f64;
        let mut competition_sum = 0.0f64;
        let mut competition_count = 0usize;
        for cleavage in candidate_matches {
            for relation in cleavage {
                if let Some(index) = relation.peak_index {
                    if unique_peaks.insert(index) {
                        explained_intensity += peaks[index].intensity;
                    }
                    competition_sum += claim_counts[index] as f64;
                    competition_count += 1;
                }
            }
        }
        let peak_coverage = unique_peaks.len() as f32 / peaks.len().max(1) as f32;
        let explained_fraction =
            (explained_intensity / total_observed_intensity).clamp(0.0, 1.0) as f32;
        let mean_competition = if competition_count > 0 {
            (competition_sum / competition_count as f64 / peptides.len().max(1) as f64)
                .clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        let length_norm = (chars.len() as f32 / (max_cleavages + 1) as f32).clamp(0.0, 1.0);
        let precursor_error =
            (precursor_mass_errors_da[candidate_index] / 0.05).clamp(-2.0, 2.0) as f32;

        for cleavage_index in 0..candidate_matches.len() {
            let base = (candidate_index * max_cleavages + cleavage_index)
                * FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240;
            let row = &mut features[base..base + FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240];
            let left_index = residue_class(chars[cleavage_index]);
            let right_index = residue_class(chars[cleavage_index + 1]);
            row[FOUNDATION_FRAGMENT_RELATION_LEFT_AA_OFFSET_V0240 + left_index] = 1.0;
            row[FOUNDATION_FRAGMENT_RELATION_RIGHT_AA_OFFSET_V0240 + right_index] = 1.0;
            row[FOUNDATION_FRAGMENT_RELATION_LEFT_MOD_MASS_V0240] =
                (mod_mass[cleavage_index] / 100.0).clamp(-2.0, 2.0);
            row[FOUNDATION_FRAGMENT_RELATION_RIGHT_MOD_MASS_V0240] =
                (mod_mass[cleavage_index + 1] / 100.0).clamp(-2.0, 2.0);
            row[FOUNDATION_FRAGMENT_RELATION_LEFT_MOD_FLAG_V0240] = mod_flag[cleavage_index];
            row[FOUNDATION_FRAGMENT_RELATION_RIGHT_MOD_FLAG_V0240] = mod_flag[cleavage_index + 1];

            for channel in 0..FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240 {
                let pred = predicted[cleavage_index]
                    .get(channel)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0) as f64;
                row[FOUNDATION_FRAGMENT_RELATION_PREDICTED_OFFSET_V0240 + channel] =
                    (pred / pred_max).sqrt() as f32;
                let relation = candidate_matches[cleavage_index][channel];
                row[FOUNDATION_FRAGMENT_RELATION_OBSERVED_OFFSET_V0240 + channel] =
                    relation.observed.max(0.0).sqrt();
                row[FOUNDATION_FRAGMENT_RELATION_MASS_ERROR_OFFSET_V0240 + channel] =
                    relation.normalized_error;
                if let Some(index) = relation.peak_index {
                    row[FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240 + channel] = 1.0;
                    row[FOUNDATION_FRAGMENT_RELATION_INVERSE_CLAIM_OFFSET_V0240 + channel] =
                        1.0 / claim_counts[index].max(1) as f32;
                    matched_relations += 1;
                }
            }

            // Complementary evidence at the same cleavage (b1<->y1, b2<->y2).
            for pair in 0..2 {
                let b = candidate_matches[cleavage_index][pair];
                let y = candidate_matches[cleavage_index][pair + 2];
                row[FOUNDATION_FRAGMENT_RELATION_COMPLEMENT_OBS_OFFSET_V0240 + pair] =
                    b.observed.min(y.observed).max(0.0).sqrt();
                let bp = predicted[cleavage_index]
                    .get(pair)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0) as f64
                    / pred_max;
                let yp = predicted[cleavage_index]
                    .get(pair + 2)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0) as f64
                    / pred_max;
                row[FOUNDATION_FRAGMENT_RELATION_COMPLEMENT_PRED_OFFSET_V0240 + pair] =
                    (bp * yp).sqrt() as f32;
            }

            let denom = candidate_matches.len().max(1) as f32;
            row[FOUNDATION_FRAGMENT_RELATION_POSITION_V0240] =
                (cleavage_index as f32 + 1.0) / denom;
            row[FOUNDATION_FRAGMENT_RELATION_REVERSE_POSITION_V0240] =
                (candidate_matches.len() - cleavage_index) as f32 / denom;
            row[FOUNDATION_FRAGMENT_RELATION_LENGTH_V0240] = length_norm;
            row[FOUNDATION_FRAGMENT_RELATION_PEAK_COVERAGE_V0240] = peak_coverage;
            row[FOUNDATION_FRAGMENT_RELATION_EXPLAINED_INTENSITY_V0240] = explained_fraction;
            row[FOUNDATION_FRAGMENT_RELATION_MEAN_COMPETITION_V0240] = mean_competition;
            row[FOUNDATION_FRAGMENT_RELATION_PRECURSOR_ERROR_V0240] = precursor_error;
            mask[candidate_index * max_cleavages + cleavage_index] = 1.0;
        }
    }

    Ok(FoundationFragmentRelationFeatureRows {
        features,
        mask,
        candidates: peptides.len(),
        max_cleavages,
        matched_relations,
        contested_peaks,
    })
}

fn residue_modification_state(peptide: &PeptidoformInput, residues: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mass = vec![0.0f32; residues];
    let mut flag = vec![0.0f32; residues];
    for modification in &peptide.modifications {
        let index = match modification.site {
            FoundationModificationSite::Residue(index) => index,
            FoundationModificationSite::NTerm => 0,
            FoundationModificationSite::CTerm => residues.saturating_sub(1),
        };
        if index < residues {
            mass[index] += modification.mass_delta;
            flag[index] = 1.0;
        }
    }
    (mass, flag)
}

fn residue_class(residue: char) -> usize {
    match residue.to_ascii_uppercase() {
        'A' => 0,
        'C' => 1,
        'D' => 2,
        'E' => 3,
        'F' => 4,
        'G' => 5,
        'H' => 6,
        'I' => 7,
        'K' => 8,
        'L' => 9,
        'M' => 10,
        'N' => 11,
        'P' => 12,
        'Q' => 13,
        'R' => 14,
        'S' => 15,
        'T' => 16,
        'V' => 17,
        'W' => 18,
        'Y' => 19,
        _ => 20,
    }
}

fn normalized_retained_peaks(spectrum: &FoundationSpectrum) -> Vec<NormalizedPeak> {
    let mut peaks = spectrum
        .peaks
        .iter()
        .copied()
        .filter(|peak| {
            peak.mz.is_finite()
                && peak.mz > 0.0
                && peak.intensity.is_finite()
                && peak.intensity > 0.0
        })
        .collect::<Vec<FoundationSpectrumPeak>>();
    peaks.sort_by(|left, right| {
        right
            .intensity
            .total_cmp(&left.intensity)
            .then_with(|| left.mz.total_cmp(&right.mz))
    });
    peaks.truncate(FOUNDATION_FRAGMENT_RELATION_MAX_PEAKS_V0240);
    let max_intensity = peaks
        .iter()
        .map(|peak| f64::from(peak.intensity))
        .fold(0.0f64, f64::max)
        .max(f64::EPSILON);
    let mut normalized = peaks
        .into_iter()
        .map(|peak| NormalizedPeak {
            mz: f64::from(peak.mz),
            intensity: (f64::from(peak.intensity) / max_intensity).clamp(0.0, 1.0),
        })
        .collect::<Vec<_>>();
    normalized.sort_by(|left, right| left.mz.total_cmp(&right.mz));
    normalized
}

fn best_peak_match(theoretical_mz: f64, peaks: &[NormalizedPeak]) -> PeakMatch {
    if !(theoretical_mz > 0.0 && theoretical_mz.is_finite()) {
        return PeakMatch::default();
    }
    let tolerance = FOUNDATION_FRAGMENT_RELATION_ABS_TOLERANCE_DA_V0240
        .max(theoretical_mz * FOUNDATION_FRAGMENT_RELATION_PPM_V0240 * 1.0e-6);
    let lower = theoretical_mz - tolerance;
    let upper = theoretical_mz + tolerance;
    let start = peaks.partition_point(|peak| peak.mz < lower);
    let mut best = None::<(usize, f64, f64)>;
    for (offset, peak) in peaks[start..]
        .iter()
        .take_while(|peak| peak.mz <= upper)
        .enumerate()
    {
        let index = start + offset;
        let delta = peak.mz - theoretical_mz;
        // Prefer intense peaks but retain mass proximity.  This mirrors the
        // fixed v0.23 soft evidence policy while exposing the error separately.
        let normalized_error = delta / tolerance;
        let support = peak.intensity.sqrt() * (-0.5 * normalized_error * normalized_error).exp();
        match best {
            Some((_, best_support, best_abs_error))
                if support < best_support
                    || (support == best_support && delta.abs() >= best_abs_error) => {}
            _ => best = Some((index, support, delta.abs())),
        }
    }
    match best {
        Some((index, _, _)) => PeakMatch {
            peak_index: Some(index),
            observed: peaks[index].intensity as f32,
            normalized_error: ((peaks[index].mz - theoretical_mz) / tolerance).clamp(-1.0, 1.0)
                as f32,
        },
        None => PeakMatch::default(),
    }
}

fn core_fragment_mz(prefix_mass: f64, suffix_with_water: f64) -> [f64; 4] {
    [
        prefix_mass + PROTON_MASS_DA,
        (prefix_mass + 2.0 * PROTON_MASS_DA) / 2.0,
        suffix_with_water + PROTON_MASS_DA,
        (suffix_with_water + 2.0 * PROTON_MASS_DA) / 2.0,
    ]
}

fn peptide_residue_mass(peptide: &PeptidoformInput) -> std::result::Result<f64, String> {
    let vocabulary = FoundationDiffusionVocabulary;
    let tokens = vocabulary.encode(peptide, 256)?;
    Ok(tokens
        .into_iter()
        .take_while(|&token| token != FOUNDATION_DIFFUSION_PAD && token != FOUNDATION_DIFFUSION_EOS)
        .filter_map(foundation_diffusion_token_mass_da)
        .sum())
}

fn cleavage_prefix_masses(peptide: &PeptidoformInput) -> std::result::Result<Vec<f64>, String> {
    let vocabulary = FoundationDiffusionVocabulary;
    let tokens = vocabulary.encode(peptide, 256)?;
    let mut prefix_mass = 0.0f64;
    let mut residue_count = 0usize;
    let mut prefixes = Vec::<f64>::new();
    for token in tokens {
        if token == FOUNDATION_DIFFUSION_PAD || token == FOUNDATION_DIFFUSION_EOS {
            break;
        }
        if foundation_diffusion_token_residue(token).is_some() {
            if residue_count > 0 {
                prefixes.push(prefix_mass);
            }
            residue_count += 1;
        }
        if let Some(mass) = foundation_diffusion_token_mass_da(token) {
            prefix_mass += mass;
        }
    }
    Ok(prefixes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn spectrum_at_b1_a() -> FoundationSpectrum {
        FoundationSpectrum {
            peaks: vec![FoundationSpectrumPeak {
                mz: 71.037_113_805 + PROTON_MASS_DA,
                intensity: 100.0,
            }],
        }
    }

    #[test]
    fn relation_feature_width_and_mask_are_stable() {
        let peptides = vec![PeptidoformInput::unmodified("AG")];
        let predicted = vec![vec![vec![1.0, 0.5, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0]]];
        let rows = foundation_fragment_relation_features(
            &peptides,
            &spectrum_at_b1_a(),
            &predicted,
            &[0.0],
            63,
        )
        .unwrap();
        assert_eq!(
            rows.features.len(),
            63 * FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240
        );
        assert_eq!(rows.mask.iter().filter(|&&value| value > 0.5).count(), 1);
        assert_eq!(rows.candidates, 1);
    }

    #[test]
    fn same_spectrum_candidate_competition_changes_peak_claim_feature() {
        let peptide = PeptidoformInput::unmodified("AG");
        let predicted_one = vec![vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]];
        let single = foundation_fragment_relation_features(
            &[peptide.clone()],
            &spectrum_at_b1_a(),
            &predicted_one,
            &[0.0],
            1,
        )
        .unwrap();
        let shared = foundation_fragment_relation_features(
            &[peptide.clone(), peptide],
            &spectrum_at_b1_a(),
            &[predicted_one[0].clone(), predicted_one[0].clone()],
            &[0.0, 0.0],
            1,
        )
        .unwrap();
        assert!(
            (single.features[FOUNDATION_FRAGMENT_RELATION_INVERSE_CLAIM_OFFSET_V0240] - 1.0).abs()
                < 1e-6
        );
        assert!(
            (shared.features[FOUNDATION_FRAGMENT_RELATION_INVERSE_CLAIM_OFFSET_V0240] - 0.5).abs()
                < 1e-6
        );
        assert_eq!(shared.contested_peaks, 1);
    }

    #[test]
    fn relation_model_scores_grouped_candidates() {
        let device = Device::Cpu;
        let peptide = PeptidoformInput::unmodified("AG");
        let predicted = vec![vec![vec![1.0, 0.2, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0]]; 2];
        let rows = foundation_fragment_relation_features(
            &[peptide.clone(), peptide],
            &spectrum_at_b1_a(),
            &predicted,
            &[0.0, 0.0],
            63,
        )
        .unwrap();
        let batch = rows.to_batch(&device).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideSpectrumFragmentRelationEnergy::new(vb).unwrap();
        let scores = model.forward_grouped(&batch, 2).unwrap();
        assert_eq!(scores.dims(), &[1, 2]);
    }
}
