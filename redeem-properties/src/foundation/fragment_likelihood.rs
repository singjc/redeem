//! v0.23 forward-fragment-intensity likelihood for global peptide proposals.
//!
//! The inverse lanes through v0.22 showed that a single decoded peptide is often
//! outside the basin of the true answer even when the mass-complete v0.13.23
//! proposal pool still contains the target.  This module therefore inverts the
//! already-trained forward MS2 model instead of learning another proposal-label
//! reranker: for one complete peptide candidate, predicted cleavage-channel
//! intensities are aligned to the measured spectrum at the candidate's
//! theoretical fragment m/z values and compared by a fixed spectral-shape score.
//!
//! Missing observed peaks are soft zero evidence.  They never invalidate a
//! chemically valid peptide.  The primary v0.23 score uses unambiguous b/y z=1/2
//! channels; neutral-loss channels are retained as diagnostic telemetry because
//! the historical forward target aggregates their fragment charge.

use super::diffusion::{
    foundation_diffusion_token_mass_da, foundation_diffusion_token_residue,
    FoundationDiffusionVocabulary, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use super::featurize::PeptidoformInput;
use super::spectrum::{FoundationSpectrum, FoundationSpectrumPeak};

/// Stable architecture identifier for the bounded v0.23 experiment.
pub const FOUNDATION_FRAGMENT_LIKELIHOOD_ARCHITECTURE_V0230: &str =
    "v01323_mass_aware_global_proposal_prior_plus_frozen_forward_ms2_cleavage_intensity_likelihood";
/// Primary score definition.
pub const FOUNDATION_FRAGMENT_LIKELIHOOD_PRIMARY_SCORE_V0230: &str =
    "fixed_equal_rank_fusion_of_v01323_legacy_rank_and_sqrt_intensity_cosine_b1_b2_y1_y2";
/// Fixed fragment matching tolerance in ppm.
pub const FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230: f64 = 20.0;
/// Fixed absolute fragment matching floor in Da.
pub const FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230: f64 = 0.02;
/// Historical inverse spectrum encoder retained at most this many peaks.
pub const FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230: usize = 256;

const PROTON_MASS_DA: f64 = 1.007_276_466_77;
const WATER_MASS_DA: f64 = 18.010_564_684;
const AMMONIA_MASS_DA: f64 = 17.026_549_101;

/// One fixed candidate-to-spectrum biochemical likelihood summary.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FoundationFragmentLikelihoodScore {
    /// Primary sqrt-intensity cosine over b1, b2, y1, y2 channels.
    pub core_cosine: f64,
    /// Same diagnostic over all eight historical forward-MS2 channels.
    pub all_channel_cosine: f64,
    /// Cleavage-level cosine after summing the four core channels per bond.
    pub cleavage_cosine: f64,
    /// Core theoretical ion channels with non-zero observed support.
    pub matched_core_ions: usize,
    /// Total core theoretical ion channels evaluated.
    pub core_ions: usize,
    /// Fraction of predicted core intensity assigned to observed-supported ions.
    pub predicted_supported_fraction: f64,
}

/// Score one complete peptide candidate against one measured spectrum using a
/// precomputed forward-MS2 intensity prediction.
///
/// `predicted_ms2[cleavage][channel]` must follow the foundation head channel
/// order: b1, b2, y1, y2, b-H2O, y-H2O, b-NH3, y-NH3.  The first four channels
/// define the frozen primary v0.23 score.
pub fn foundation_fragment_likelihood_score(
    peptide: &PeptidoformInput,
    spectrum: &FoundationSpectrum,
    predicted_ms2: &[Vec<f32>],
) -> std::result::Result<FoundationFragmentLikelihoodScore, String> {
    let cleavage_prefixes = cleavage_prefix_masses(peptide)?;
    if cleavage_prefixes.is_empty() {
        return Ok(FoundationFragmentLikelihoodScore::default());
    }
    if predicted_ms2.len() < cleavage_prefixes.len() {
        return Err(format!(
            "forward MS2 rows {} shorter than candidate cleavage count {}",
            predicted_ms2.len(),
            cleavage_prefixes.len()
        ));
    }
    let total_residue_mass = peptide_residue_mass(peptide)?;
    let peaks = normalized_retained_peaks(spectrum);

    let mut predicted_core = Vec::<f64>::with_capacity(cleavage_prefixes.len() * 4);
    let mut observed_core = Vec::<f64>::with_capacity(cleavage_prefixes.len() * 4);
    let mut predicted_all = Vec::<f64>::with_capacity(cleavage_prefixes.len() * 8);
    let mut observed_all = Vec::<f64>::with_capacity(cleavage_prefixes.len() * 8);
    let mut predicted_cleavage = Vec::<f64>::with_capacity(cleavage_prefixes.len());
    let mut observed_cleavage = Vec::<f64>::with_capacity(cleavage_prefixes.len());
    let mut matched_core_ions = 0usize;
    let mut predicted_supported = 0.0f64;
    let mut predicted_sum = 0.0f64;

    for (cleavage_index, &prefix_mass) in cleavage_prefixes.iter().enumerate() {
        let row = &predicted_ms2[cleavage_index];
        if row.len() < 4 {
            return Err(format!(
                "forward MS2 row {cleavage_index} has {} channels; need >=4",
                row.len()
            ));
        }
        let suffix_with_water = total_residue_mass - prefix_mass + FOUNDATION_PEPTIDE_WATER_MASS_DA;
        if !(prefix_mass > 0.0 && suffix_with_water > FOUNDATION_PEPTIDE_WATER_MASS_DA) {
            continue;
        }

        let core_mz = [
            prefix_mass + PROTON_MASS_DA,
            (prefix_mass + 2.0 * PROTON_MASS_DA) / 2.0,
            suffix_with_water + PROTON_MASS_DA,
            (suffix_with_water + 2.0 * PROTON_MASS_DA) / 2.0,
        ];
        let mut cleavage_pred = 0.0f64;
        let mut cleavage_obs = 0.0f64;
        for channel in 0..4 {
            let pred = f64::from(row[channel]).max(0.0);
            let obs = best_peak_support(core_mz[channel], &peaks);
            predicted_core.push(pred);
            observed_core.push(obs);
            predicted_all.push(pred);
            observed_all.push(obs);
            cleavage_pred += pred;
            cleavage_obs += obs;
            predicted_sum += pred;
            if obs > 0.0 {
                matched_core_ions += 1;
                predicted_supported += pred;
            }
        }
        predicted_cleavage.push(cleavage_pred);
        observed_cleavage.push(cleavage_obs);

        if row.len() >= 8 {
            let loss_specs = [
                (prefix_mass - WATER_MASS_DA, false),
                (suffix_with_water - WATER_MASS_DA, true),
                (prefix_mass - AMMONIA_MASS_DA, false),
                (suffix_with_water - AMMONIA_MASS_DA, true),
            ];
            for (offset, &(neutral_mass, _is_y)) in loss_specs.iter().enumerate() {
                let channel = 4 + offset;
                let pred = f64::from(row[channel]).max(0.0);
                let obs = if neutral_mass > 0.0 {
                    let z1 = neutral_mass + PROTON_MASS_DA;
                    let z2 = (neutral_mass + 2.0 * PROTON_MASS_DA) / 2.0;
                    best_peak_support(z1, &peaks).max(best_peak_support(z2, &peaks))
                } else {
                    0.0
                };
                predicted_all.push(pred);
                observed_all.push(obs);
            }
        }
    }

    Ok(FoundationFragmentLikelihoodScore {
        core_cosine: sqrt_intensity_cosine(&predicted_core, &observed_core),
        all_channel_cosine: sqrt_intensity_cosine(&predicted_all, &observed_all),
        cleavage_cosine: sqrt_intensity_cosine(&predicted_cleavage, &observed_cleavage),
        matched_core_ions,
        core_ions: predicted_core.len(),
        predicted_supported_fraction: if predicted_sum > 0.0 {
            (predicted_supported / predicted_sum).clamp(0.0, 1.0)
        } else {
            0.0
        },
    })
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
    // There must be exactly residues-1 cleavage masses. PTM tokens after the
    // final residue do not create an extra cleavage.
    Ok(prefixes)
}

fn normalized_retained_peaks(spectrum: &FoundationSpectrum) -> Vec<(f64, f64)> {
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
    peaks.truncate(FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230);
    let max_intensity = peaks
        .iter()
        .map(|peak| f64::from(peak.intensity))
        .fold(0.0f64, f64::max)
        .max(f64::EPSILON);
    let mut normalized = peaks
        .into_iter()
        .map(|peak| {
            (
                f64::from(peak.mz),
                (f64::from(peak.intensity) / max_intensity).clamp(0.0, 1.0),
            )
        })
        .collect::<Vec<_>>();
    normalized.sort_by(|left, right| left.0.total_cmp(&right.0));
    normalized
}

fn best_peak_support(theoretical_mz: f64, peaks: &[(f64, f64)]) -> f64 {
    if !(theoretical_mz > 0.0 && theoretical_mz.is_finite()) {
        return 0.0;
    }
    let sigma = (theoretical_mz * FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230 * 1e-6)
        .max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230 / 3.0);
    let cutoff = (3.0 * sigma).max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230);
    let mut best = 0.0f64;
    for &(observed_mz, normalized_intensity) in peaks {
        let error = (observed_mz - theoretical_mz).abs();
        if error > cutoff {
            continue;
        }
        let mass_weight = (-0.5 * (error / sigma).powi(2)).exp();
        best = best.max(normalized_intensity * mass_weight);
    }
    best
}

fn sqrt_intensity_cosine(predicted: &[f64], observed: &[f64]) -> f64 {
    if predicted.len() != observed.len() || predicted.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut pred_norm = 0.0f64;
    let mut obs_norm = 0.0f64;
    for (&predicted, &observed) in predicted.iter().zip(observed) {
        let p = predicted.max(0.0).sqrt();
        let o = observed.max(0.0).sqrt();
        dot += p * o;
        pred_norm += p * p;
        obs_norm += o * o;
    }
    if pred_norm > 0.0 && obs_norm > 0.0 {
        (dot / (pred_norm.sqrt() * obs_norm.sqrt())).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_peaks_are_soft_zero_evidence_not_invalidity() {
        let peptide = PeptidoformInput::unmodified("PEPTIDE");
        let spectrum = FoundationSpectrum::default();
        let predicted = vec![vec![1.0; 8]; 6];
        let score = foundation_fragment_likelihood_score(&peptide, &spectrum, &predicted).unwrap();
        assert_eq!(score.core_cosine, 0.0);
        assert_eq!(score.matched_core_ions, 0);
        assert_eq!(score.core_ions, 24);
    }

    #[test]
    fn aligned_fragment_pattern_scores_above_empty_pattern() {
        let peptide = PeptidoformInput::unmodified("AG");
        // A b1 peak for alanine at ~72.0444 and complementary y1 for glycine at ~76.0393.
        let spectrum = FoundationSpectrum::from_pairs([(72.0444, 100.0), (76.0393, 80.0)]);
        let predicted = vec![vec![1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let score = foundation_fragment_likelihood_score(&peptide, &spectrum, &predicted).unwrap();
        assert!(score.core_cosine > 0.8, "score={score:?}");
        assert!(score.matched_core_ions >= 2);
    }

    #[test]
    fn candidate_ptm_mass_changes_fragment_alignment() {
        let unmodified = PeptidoformInput::unmodified("MC");
        let oxidized = super::super::dataset::parse_modified_peptide("M[UniMod:35]C").unwrap();
        let predicted = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        // Unmodified M b1 is ~132.0478; oxidation shifts it by ~15.9949 Da.
        let spectrum = FoundationSpectrum::from_pairs([(132.0478, 100.0)]);
        let base =
            foundation_fragment_likelihood_score(&unmodified, &spectrum, &predicted).unwrap();
        let ox = foundation_fragment_likelihood_score(&oxidized, &spectrum, &predicted).unwrap();
        assert!(base.core_cosine > ox.core_cosine);
    }
}
