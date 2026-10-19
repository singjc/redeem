//! Frozen practical inverse identifier promoted from the validated v0.75 lane.
//!
//! v0.75.1 is an integration/refactor release, not a new scientific model. It
//! exposes the protected-confirmed v0.75 search path as a reusable library API:
//! observed precursor charge + neutral mass -> nearest 256 same-charge
//! peptidoforms -> deterministic b1/b2/y1/y2 fragment-geometry ranking.
//!
//! The implementation intentionally contains no learned candidate embedding,
//! no forward-model checkpoint, no optimizer, no target forcing, and no tuning
//! surface for the candidate-pool size or fragment score.

use super::diffusion::{foundation_peptidoform_neutral_mass, foundation_precursor_neutral_mass};
use super::featurize::PeptidoformInput;
use super::fragment_likelihood::{
    FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230, FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230,
};
use super::fragment_relation::foundation_fragment_cleavage_geometry;
use super::spectrum::FoundationSpectrum;
use anyhow::{bail, Result};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// Integration version for the practical identifier API.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_VERSION_V0751: u32 = 751;
/// Frozen architecture selected and protected-confirmed by v0.75.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751: &str =
    "observed_charge_neutral_mass256_plus_deterministic_open_ptm_fragment_geometry";
/// Frozen deterministic score selected and protected-confirmed by v0.75.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751: &str =
    "open_ptm_core_b1_b2_y1_y2_uniform_sqrt_intensity_cosine_20ppm_abs0p02Da";
/// Frozen candidate-generation policy. This value is not a tunable parameter.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POLICY_V0751: &str =
    "observed_charge_compatible_nearest_neutral_mass256_no_target_forcing";
/// Number of same-charge neutral-mass-nearest candidates scored per query when available.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751: usize = 256;

/// One peptidoform/charge candidate supplied to the practical identifier.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationPracticalIdentifierCandidateV0751 {
    /// Stable caller-provided key returned in ranked hits.
    pub key: String,
    /// Exact peptidoform chemistry used for neutral-mass and fragment calculations.
    pub peptidoform: PeptidoformInput,
    /// Positive precursor charge associated with this candidate identity.
    pub charge: i32,
}

impl FoundationPracticalIdentifierCandidateV0751 {
    /// Construct one candidate. Validation occurs when the identifier is built.
    pub fn new(key: impl Into<String>, peptidoform: PeptidoformInput, charge: i32) -> Self {
        Self {
            key: key.into(),
            peptidoform,
            charge,
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedCandidateV0751 {
    key: String,
    charge: i32,
    neutral_mass: f64,
    fragment_mz: Vec<f64>,
}

/// Build-time diagnostics only; they do not participate in ranking.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FoundationPracticalIdentifierBuildTimingsV0751 {
    pub index_build_seconds: f64,
    pub geometry_precompute_seconds: f64,
}

/// One ranked practical-identifier candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationPracticalIdentifierHitV0751 {
    /// One-based rank after deterministic geometry scoring.
    pub rank: usize,
    /// Zero-based position in the original candidate catalog.
    pub candidate_index: usize,
    /// Caller-provided stable candidate key.
    pub key: String,
    /// Candidate precursor charge.
    pub charge: i32,
    /// Theoretical peptidoform neutral mass.
    pub theoretical_neutral_mass: f64,
    /// Absolute error from the observed neutral precursor mass.
    pub absolute_neutral_mass_error: f64,
    /// Frozen deterministic fragment-geometry score in [0, 1].
    pub geometry_score: f64,
}

/// Reusable frozen v0.75 practical identifier.
#[derive(Debug, Clone)]
pub struct FoundationPracticalIdentifierV0751 {
    candidates: Vec<PreparedCandidateV0751>,
    mass_sorted_by_charge: BTreeMap<i32, Vec<(f64, usize)>>,
    build_timings: FoundationPracticalIdentifierBuildTimingsV0751,
}

impl FoundationPracticalIdentifierV0751 {
    /// Build and validate the frozen practical identifier candidate index.
    pub fn new(candidates: Vec<FoundationPracticalIdentifierCandidateV0751>) -> Result<Self> {
        if candidates.is_empty() {
            bail!("v0.75.1 practical identifier requires a non-empty candidate catalog");
        }

        let unique = candidates
            .iter()
            .map(|candidate| candidate.key.as_str())
            .collect::<BTreeSet<_>>();
        if unique.len() != candidates.len() {
            bail!("v0.75.1 practical identifier candidate keys must be unique");
        }
        if candidates.iter().any(|candidate| candidate.key.is_empty()) {
            bail!("v0.75.1 practical identifier candidate keys must be non-empty");
        }
        if candidates.iter().any(|candidate| candidate.charge <= 0) {
            bail!("v0.75.1 practical identifier candidate charges must be positive");
        }

        let index_started = Instant::now();
        let mut masses = Vec::with_capacity(candidates.len());
        let mut mass_sorted_by_charge = BTreeMap::<i32, Vec<(f64, usize)>>::new();
        for (index, candidate) in candidates.iter().enumerate() {
            let neutral_mass = foundation_peptidoform_neutral_mass(&candidate.peptidoform)
                .map_err(anyhow::Error::msg)?;
            if !(neutral_mass.is_finite() && neutral_mass > 0.0) {
                bail!("v0.75.1 candidate neutral mass must be positive and finite");
            }
            masses.push(neutral_mass);
            mass_sorted_by_charge
                .entry(candidate.charge)
                .or_default()
                .push((neutral_mass, index));
        }
        for rows in mass_sorted_by_charge.values_mut() {
            rows.sort_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.1.cmp(&right.1))
            });
        }
        let index_build_seconds = index_started.elapsed().as_secs_f64();

        let geometry_started = Instant::now();
        let mut prepared = Vec::with_capacity(candidates.len());
        for (candidate, neutral_mass) in candidates.into_iter().zip(masses) {
            let geometry = foundation_fragment_cleavage_geometry(&candidate.peptidoform)
                .map_err(anyhow::Error::msg)?;
            let mut fragment_mz = Vec::with_capacity(geometry.len() * 4);
            for cleavage in geometry {
                fragment_mz.extend_from_slice(&cleavage.core_mz);
            }
            prepared.push(PreparedCandidateV0751 {
                key: candidate.key,
                charge: candidate.charge,
                neutral_mass,
                fragment_mz,
            });
        }
        let geometry_precompute_seconds = geometry_started.elapsed().as_secs_f64();

        Ok(Self {
            candidates: prepared,
            mass_sorted_by_charge,
            build_timings: FoundationPracticalIdentifierBuildTimingsV0751 {
                index_build_seconds,
                geometry_precompute_seconds,
            },
        })
    }

    /// Number of unique peptidoform/charge identities in the catalog.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Build-time diagnostics. They do not affect ranking.
    pub fn build_timings(&self) -> FoundationPracticalIdentifierBuildTimingsV0751 {
        self.build_timings
    }

    /// Identify from the directly observed precursor m/z and charge.
    pub fn identify_precursor_mz(
        &self,
        observed_charge: i32,
        observed_precursor_mz: f64,
        spectrum: &FoundationSpectrum,
    ) -> Result<Vec<FoundationPracticalIdentifierHitV0751>> {
        let observed_neutral_mass =
            foundation_precursor_neutral_mass(observed_precursor_mz, observed_charge)
                .map_err(anyhow::Error::msg)?;
        self.identify_neutral_mass(observed_charge, observed_neutral_mass, spectrum)
    }

    /// Identify from an already calculated neutral precursor mass and observed charge.
    pub fn identify_neutral_mass(
        &self,
        observed_charge: i32,
        observed_neutral_mass: f64,
        spectrum: &FoundationSpectrum,
    ) -> Result<Vec<FoundationPracticalIdentifierHitV0751>> {
        if observed_charge <= 0 {
            bail!("v0.75.1 observed precursor charge must be positive");
        }
        if !(observed_neutral_mass.is_finite() && observed_neutral_mass > 0.0) {
            bail!("v0.75.1 observed neutral mass must be positive and finite");
        }

        let peaks = normalized_retained_peaks_v0751(spectrum);
        if peaks.is_empty() {
            bail!("v0.75.1 observed spectrum contains no finite positive peaks");
        }

        let charge_mass_sorted = self
            .mass_sorted_by_charge
            .get(&observed_charge)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "v0.75.1 candidate catalog has no entries for observed precursor charge {}",
                    observed_charge
                )
            })?;
        let pool = nearest_mass_pool_v0751(
            charge_mass_sorted,
            observed_neutral_mass,
            FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
        );
        if pool.is_empty() {
            bail!("v0.75.1 same-charge candidate pool is empty");
        }

        let mut scored = Vec::<(usize, f64)>::with_capacity(pool.len());
        for candidate_index in pool {
            let candidate = &self.candidates[candidate_index];
            if candidate.charge != observed_charge {
                bail!("v0.75.1 charge-incompatible candidate escaped per-charge mass index");
            }
            let score = geometry_uniform_score_v0751(&candidate.fragment_mz, &peaks);
            if !score.is_finite() {
                bail!("v0.75.1 geometry score is not finite");
            }
            scored.push((candidate_index, score));
        }
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });

        Ok(scored
            .into_iter()
            .enumerate()
            .map(|(rank0, (candidate_index, geometry_score))| {
                let candidate = &self.candidates[candidate_index];
                FoundationPracticalIdentifierHitV0751 {
                    rank: rank0 + 1,
                    candidate_index,
                    key: candidate.key.clone(),
                    charge: candidate.charge,
                    theoretical_neutral_mass: candidate.neutral_mass,
                    absolute_neutral_mass_error: (candidate.neutral_mass - observed_neutral_mass)
                        .abs(),
                    geometry_score,
                }
            })
            .collect())
    }
}

fn nearest_mass_pool_v0751(
    mass_sorted: &[(f64, usize)],
    observed_mass: f64,
    count: usize,
) -> Vec<usize> {
    if mass_sorted.is_empty() || count == 0 {
        return Vec::new();
    }
    let split = mass_sorted.partition_point(|row| row.0 < observed_mass);
    let mut left = split;
    let mut right = split;
    let mut out = Vec::with_capacity(count.min(mass_sorted.len()));
    while out.len() < count.min(mass_sorted.len()) {
        let left_item = if left > 0 {
            Some(mass_sorted[left - 1])
        } else {
            None
        };
        let right_item = mass_sorted.get(right).copied();
        let take_left = match (left_item, right_item) {
            (Some(l), Some(r)) => {
                let ld = (l.0 - observed_mass).abs();
                let rd = (r.0 - observed_mass).abs();
                ld < rd || (ld == rd && l.1 < r.1)
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_left {
            left -= 1;
            out.push(mass_sorted[left].1);
        } else {
            out.push(mass_sorted[right].1);
            right += 1;
        }
    }
    out
}

fn normalized_retained_peaks_v0751(spectrum: &FoundationSpectrum) -> Vec<(f64, f64)> {
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
        .collect::<Vec<_>>();
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

fn best_peak_support_fast_v0751(theoretical_mz: f64, peaks: &[(f64, f64)]) -> f64 {
    if !(theoretical_mz > 0.0 && theoretical_mz.is_finite()) {
        return 0.0;
    }
    let sigma = (theoretical_mz * FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230 * 1e-6)
        .max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230 / 3.0);
    let cutoff = (3.0 * sigma).max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230);
    let low = theoretical_mz - cutoff;
    let high = theoretical_mz + cutoff;
    let start = peaks.partition_point(|row| row.0 < low);
    let mut best = 0.0f64;
    for &(observed_mz, normalized_intensity) in &peaks[start..] {
        if observed_mz > high {
            break;
        }
        let error = (observed_mz - theoretical_mz).abs();
        let mass_weight = (-0.5 * (error / sigma).powi(2)).exp();
        best = best.max(normalized_intensity * mass_weight);
    }
    best
}

fn geometry_uniform_score_v0751(theoretical_mz: &[f64], peaks: &[(f64, f64)]) -> f64 {
    if theoretical_mz.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut obs_norm = 0.0f64;
    for &mz in theoretical_mz {
        let observed = best_peak_support_fast_v0751(mz, peaks).max(0.0);
        dot += observed.sqrt();
        obs_norm += observed;
    }
    if obs_norm > 0.0 {
        (dot / ((theoretical_mz.len() as f64).sqrt() * obs_norm.sqrt())).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_nearest_mass_pool(
        mass_sorted: &[(f64, usize)],
        observed_mass: f64,
        count: usize,
    ) -> Vec<usize> {
        if mass_sorted.is_empty() || count == 0 {
            return Vec::new();
        }
        let split = mass_sorted.partition_point(|row| row.0 < observed_mass);
        let mut left = split;
        let mut right = split;
        let mut out = Vec::with_capacity(count.min(mass_sorted.len()));
        while out.len() < count.min(mass_sorted.len()) {
            let left_item = if left > 0 {
                Some(mass_sorted[left - 1])
            } else {
                None
            };
            let right_item = mass_sorted.get(right).copied();
            let take_left = match (left_item, right_item) {
                (Some(l), Some(r)) => {
                    let ld = (l.0 - observed_mass).abs();
                    let rd = (r.0 - observed_mass).abs();
                    ld < rd || (ld == rd && l.1 < r.1)
                }
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if take_left {
                left -= 1;
                out.push(mass_sorted[left].1);
            } else {
                out.push(mass_sorted[right].1);
                right += 1;
            }
        }
        out
    }

    fn legacy_geometry_uniform_score(theoretical_mz: &[f64], peaks: &[(f64, f64)]) -> f64 {
        if theoretical_mz.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0f64;
        let mut obs_norm = 0.0f64;
        for &mz in theoretical_mz {
            let sigma = (mz * FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230 * 1e-6)
                .max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230 / 3.0);
            let cutoff = (3.0 * sigma).max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230);
            let low = mz - cutoff;
            let high = mz + cutoff;
            let start = peaks.partition_point(|row| row.0 < low);
            let mut observed = 0.0f64;
            for &(observed_mz, normalized_intensity) in &peaks[start..] {
                if observed_mz > high {
                    break;
                }
                let error = (observed_mz - mz).abs();
                let mass_weight = (-0.5 * (error / sigma).powi(2)).exp();
                observed = observed.max(normalized_intensity * mass_weight);
            }
            dot += observed.sqrt();
            obs_norm += observed;
        }
        if obs_norm > 0.0 {
            (dot / ((theoretical_mz.len() as f64).sqrt() * obs_norm.sqrt())).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    #[test]
    fn frozen_constants_match_v075_contract() {
        assert_eq!(FOUNDATION_PRACTICAL_IDENTIFIER_VERSION_V0751, 751);
        assert_eq!(FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751, 256);
        assert_eq!(
            FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751,
            "observed_charge_neutral_mass256_plus_deterministic_open_ptm_fragment_geometry"
        );
        assert_eq!(
            FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751,
            "open_ptm_core_b1_b2_y1_y2_uniform_sqrt_intensity_cosine_20ppm_abs0p02Da"
        );
    }

    #[test]
    fn mass_pool_is_exactly_legacy_equivalent() {
        let masses = vec![(100.0, 0), (101.0, 1), (102.0, 2), (103.0, 3), (104.0, 4)];
        for observed in [99.5, 101.5, 102.0, 104.5] {
            for count in 0..=6 {
                assert_eq!(
                    nearest_mass_pool_v0751(&masses, observed, count),
                    legacy_nearest_mass_pool(&masses, observed, count)
                );
            }
        }
    }

    #[test]
    fn geometry_score_is_exactly_legacy_equivalent() {
        let theoretical = vec![100.0, 200.0, 300.0, 400.0];
        let peaks = vec![(99.999, 1.0), (200.001, 0.8), (250.0, 0.7), (399.999, 0.4)];
        assert_eq!(
            geometry_uniform_score_v0751(&theoretical, &peaks),
            legacy_geometry_uniform_score(&theoretical, &peaks)
        );
    }

    #[test]
    fn practical_identifier_is_charge_isolated_and_deterministic() {
        let candidates = vec![
            FoundationPracticalIdentifierCandidateV0751::new(
                "PEPTIDE|z2",
                PeptidoformInput::unmodified("PEPTIDE"),
                2,
            ),
            FoundationPracticalIdentifierCandidateV0751::new(
                "PEPTIDK|z2",
                PeptidoformInput::unmodified("PEPTIDK"),
                2,
            ),
            FoundationPracticalIdentifierCandidateV0751::new(
                "PEPTIDE|z3",
                PeptidoformInput::unmodified("PEPTIDE"),
                3,
            ),
        ];
        let identifier = FoundationPracticalIdentifierV0751::new(candidates).unwrap();
        let query_geometry =
            foundation_fragment_cleavage_geometry(&PeptidoformInput::unmodified("PEPTIDE"))
                .unwrap();
        let spectrum = FoundationSpectrum::from_pairs(
            query_geometry
                .iter()
                .flat_map(|row| row.core_mz)
                .map(|mz| (mz as f32, 1.0f32)),
        );
        let neutral_mass =
            foundation_peptidoform_neutral_mass(&PeptidoformInput::unmodified("PEPTIDE")).unwrap();
        let first = identifier
            .identify_neutral_mass(2, neutral_mass, &spectrum)
            .unwrap();
        let second = identifier
            .identify_neutral_mass(2, neutral_mass, &spectrum)
            .unwrap();
        assert_eq!(first, second);
        assert!(!first.is_empty());
        assert!(first.iter().all(|hit| hit.charge == 2));
        assert_eq!(first[0].key, "PEPTIDE|z2");
    }

    #[test]
    fn duplicate_keys_and_nonpositive_charges_fail_closed() {
        let duplicate = vec![
            FoundationPracticalIdentifierCandidateV0751::new(
                "same",
                PeptidoformInput::unmodified("PEPTIDE"),
                2,
            ),
            FoundationPracticalIdentifierCandidateV0751::new(
                "same",
                PeptidoformInput::unmodified("PEPTIDK"),
                2,
            ),
        ];
        assert!(FoundationPracticalIdentifierV0751::new(duplicate).is_err());

        let bad_charge = vec![FoundationPracticalIdentifierCandidateV0751::new(
            "bad",
            PeptidoformInput::unmodified("PEPTIDE"),
            0,
        )];
        assert!(FoundationPracticalIdentifierV0751::new(bad_charge).is_err());
    }
}
