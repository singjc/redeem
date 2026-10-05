//! Production-facing request/result API for the frozen practical inverse identifier.
//!
//! v0.75.2 does not change identification science. It wraps the byte-equivalent
//! [`FoundationPracticalIdentifierV0751`] core with a serde-friendly catalog,
//! request, response, and batch interface suitable for downstream callers.
//! The scientific candidate pool remains fixed at 256; `top_k` only truncates
//! already-ranked output returned to the caller.

use super::diffusion::foundation_precursor_neutral_mass;
use super::featurize::{FoundationModification, FoundationModificationSite, PeptidoformInput};
use super::inverse_identifier_v0751::{
    FoundationPracticalIdentifierBuildTimingsV0751, FoundationPracticalIdentifierCandidateV0751,
    FoundationPracticalIdentifierHitV0751, FoundationPracticalIdentifierV0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POLICY_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751,
};
use super::spectrum::FoundationSpectrum;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Public serialization/API contract version.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_API_VERSION_V0752: u32 = 752;
/// Stable schema identifier embedded in serialized catalogs, requests, and responses.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752: &str =
    "redeem.foundation.practical_identifier.v0752";
/// Default number of already-ranked hits returned to callers.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_DEFAULT_TOP_K_V0752: usize = 10;

fn default_schema_v0752() -> String {
    FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752.to_string()
}

fn default_top_k_v0752() -> usize {
    FOUNDATION_PRACTICAL_IDENTIFIER_DEFAULT_TOP_K_V0752
}

/// Stable serialized modification-site representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "site", rename_all = "snake_case", deny_unknown_fields)]
pub enum FoundationPracticalIdentifierModificationSiteV0752 {
    /// Modification on one zero-based residue index.
    Residue { residue_index: usize },
    /// Modification on the peptide N terminus.
    NTerm,
    /// Modification on the peptide C terminus.
    CTerm,
}

/// Stable serialized modification representation for production catalogs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierModificationV0752 {
    pub location: FoundationPracticalIdentifierModificationSiteV0752,
    pub mass_delta: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unimod_id: Option<u32>,
}

/// One serializable peptidoform/charge candidate in a production catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierCatalogCandidateV0752 {
    pub key: String,
    pub sequence: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modifications: Vec<FoundationPracticalIdentifierModificationV0752>,
    pub charge: i32,
}

/// Serializable candidate catalog for constructing the frozen practical identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierCatalogV0752 {
    pub schema: String,
    pub candidates: Vec<FoundationPracticalIdentifierCatalogCandidateV0752>,
}

impl FoundationPracticalIdentifierCatalogV0752 {
    /// Construct a catalog using the v0.75.2 schema identifier.
    pub fn new(candidates: Vec<FoundationPracticalIdentifierCatalogCandidateV0752>) -> Self {
        Self {
            schema: default_schema_v0752(),
            candidates,
        }
    }

    /// Deserialize one catalog from YAML and validate its schema during service construction.
    pub fn from_yaml_str(input: &str) -> Result<Self> {
        serde_yaml::from_str(input).context("failed to deserialize v0.75.2 identifier catalog YAML")
    }

    /// Serialize one catalog using the stable serde field contract.
    pub fn to_yaml_string(&self) -> Result<String> {
        serde_yaml::to_string(self).context("failed to serialize v0.75.2 identifier catalog YAML")
    }
}

/// One serializable observed centroided product-ion peak.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierPeakV0752 {
    pub mz: f32,
    pub intensity: f32,
}

/// One production identification request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierRequestV0752 {
    pub schema: String,
    pub query_id: String,
    pub observed_precursor_mz: f64,
    pub observed_charge: i32,
    pub peaks: Vec<FoundationPracticalIdentifierPeakV0752>,
    /// Number of already-ranked hits to return. This never changes the frozen 256-candidate pool.
    #[serde(default = "default_top_k_v0752")]
    pub top_k: usize,
}

impl FoundationPracticalIdentifierRequestV0752 {
    /// Deserialize one request from YAML.
    pub fn from_yaml_str(input: &str) -> Result<Self> {
        serde_yaml::from_str(input).context("failed to deserialize v0.75.2 identifier request YAML")
    }

    /// Serialize one request using the stable serde field contract.
    pub fn to_yaml_string(&self) -> Result<String> {
        serde_yaml::to_string(self).context("failed to serialize v0.75.2 identifier request YAML")
    }
}

/// One serializable ranked hit returned by the production API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierResultHitV0752 {
    pub rank: usize,
    pub candidate_index: usize,
    pub key: String,
    pub charge: i32,
    pub theoretical_neutral_mass: f64,
    pub absolute_neutral_mass_error: f64,
    pub geometry_score: f64,
}

impl From<FoundationPracticalIdentifierHitV0751> for FoundationPracticalIdentifierResultHitV0752 {
    fn from(hit: FoundationPracticalIdentifierHitV0751) -> Self {
        Self {
            rank: hit.rank,
            candidate_index: hit.candidate_index,
            key: hit.key,
            charge: hit.charge,
            theoretical_neutral_mass: hit.theoretical_neutral_mass,
            absolute_neutral_mass_error: hit.absolute_neutral_mass_error,
            geometry_score: hit.geometry_score,
        }
    }
}

/// Stable production response for one spectrum identification request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierResponseV0752 {
    pub schema: String,
    pub query_id: String,
    pub architecture: String,
    pub score: String,
    pub candidate_policy: String,
    pub candidate_pool: usize,
    pub catalog_fingerprint: String,
    pub catalog_size: usize,
    pub candidates_scored: usize,
    pub observed_precursor_mz: f64,
    pub observed_neutral_mass: f64,
    pub observed_charge: i32,
    pub returned_hits: usize,
    pub hits: Vec<FoundationPracticalIdentifierResultHitV0752>,
}

impl FoundationPracticalIdentifierResponseV0752 {
    /// Serialize one response using the stable serde field contract.
    pub fn to_yaml_string(&self) -> Result<String> {
        serde_yaml::to_string(self).context("failed to serialize v0.75.2 identifier response YAML")
    }
}

/// Production wrapper around the frozen, byte-equivalent v0.75.1 core.
#[derive(Debug, Clone)]
pub struct FoundationPracticalIdentifierServiceV0752 {
    identifier: FoundationPracticalIdentifierV0751,
    catalog_fingerprint: String,
    catalog_size: usize,
}

impl FoundationPracticalIdentifierServiceV0752 {
    /// Build a production identifier service directly from catalog YAML.
    pub fn from_catalog_yaml_str(input: &str) -> Result<Self> {
        Self::new(FoundationPracticalIdentifierCatalogV0752::from_yaml_str(
            input,
        )?)
    }

    /// Build a production identifier service from one serialized-domain catalog.
    pub fn new(catalog: FoundationPracticalIdentifierCatalogV0752) -> Result<Self> {
        if catalog.schema != FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752 {
            bail!(
                "v0.75.2 catalog schema mismatch: expected {}, observed {}",
                FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
                catalog.schema
            );
        }
        if catalog.candidates.is_empty() {
            bail!("v0.75.2 practical identifier catalog must be non-empty");
        }

        let catalog_fingerprint = catalog_fingerprint_v0752(&catalog.candidates);
        let catalog_size = catalog.candidates.len();
        let candidates = catalog
            .candidates
            .into_iter()
            .map(catalog_candidate_into_v0751)
            .collect::<Result<Vec<_>>>()?;
        let identifier = FoundationPracticalIdentifierV0751::new(candidates)?;

        Ok(Self {
            identifier,
            catalog_fingerprint,
            catalog_size,
        })
    }

    /// Stable fingerprint of the exact ordered production catalog supplied by the caller.
    pub fn catalog_fingerprint(&self) -> &str {
        &self.catalog_fingerprint
    }

    /// Number of peptidoform/charge candidates in the service.
    pub fn catalog_size(&self) -> usize {
        self.catalog_size
    }

    /// Build-time index/geometry timings from the frozen v0.75.1 core.
    pub fn build_timings(&self) -> FoundationPracticalIdentifierBuildTimingsV0751 {
        self.identifier.build_timings()
    }

    /// Identify one spectrum using the frozen v0.75.1 ranking semantics.
    pub fn identify(
        &self,
        request: &FoundationPracticalIdentifierRequestV0752,
    ) -> Result<FoundationPracticalIdentifierResponseV0752> {
        validate_request_v0752(request)?;

        let spectrum = FoundationSpectrum::from_pairs(
            request.peaks.iter().map(|peak| (peak.mz, peak.intensity)),
        );
        let observed_neutral_mass = foundation_precursor_neutral_mass(
            request.observed_precursor_mz,
            request.observed_charge,
        )
        .map_err(anyhow::Error::msg)?;

        let mut core_hits = self.identifier.identify_precursor_mz(
            request.observed_charge,
            request.observed_precursor_mz,
            &spectrum,
        )?;
        let candidates_scored = core_hits.len();
        core_hits.truncate(request.top_k);
        let hits = core_hits
            .into_iter()
            .map(FoundationPracticalIdentifierResultHitV0752::from)
            .collect::<Vec<_>>();

        Ok(FoundationPracticalIdentifierResponseV0752 {
            schema: default_schema_v0752(),
            query_id: request.query_id.clone(),
            architecture: FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751.to_string(),
            score: FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751.to_string(),
            candidate_policy: FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POLICY_V0751.to_string(),
            candidate_pool: FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
            catalog_fingerprint: self.catalog_fingerprint.clone(),
            catalog_size: self.catalog_size,
            candidates_scored,
            observed_precursor_mz: request.observed_precursor_mz,
            observed_neutral_mass,
            observed_charge: request.observed_charge,
            returned_hits: hits.len(),
            hits,
        })
    }

    /// Identify one YAML request and return one YAML response.
    pub fn identify_yaml_str(&self, input: &str) -> Result<String> {
        let request = FoundationPracticalIdentifierRequestV0752::from_yaml_str(input)?;
        self.identify(&request)?.to_yaml_string()
    }

    /// Identify several requests in caller-provided order, failing closed on the first invalid item.
    pub fn identify_many(
        &self,
        requests: &[FoundationPracticalIdentifierRequestV0752],
    ) -> Result<Vec<FoundationPracticalIdentifierResponseV0752>> {
        requests
            .iter()
            .map(|request| self.identify(request))
            .collect()
    }
}

fn catalog_candidate_into_v0751(
    candidate: FoundationPracticalIdentifierCatalogCandidateV0752,
) -> Result<FoundationPracticalIdentifierCandidateV0751> {
    if candidate.key.trim().is_empty() {
        bail!("v0.75.2 candidate key must be non-empty");
    }
    if candidate.sequence.is_empty() {
        bail!("v0.75.2 candidate sequence must be non-empty");
    }
    if candidate.charge <= 0 {
        bail!("v0.75.2 candidate charge must be positive");
    }

    let residue_count = candidate.sequence.chars().count();
    let mut modifications = Vec::with_capacity(candidate.modifications.len());
    for modification in candidate.modifications {
        if !modification.mass_delta.is_finite() {
            bail!("v0.75.2 modification mass_delta must be finite");
        }
        let (site, residue_index) = match modification.location {
            FoundationPracticalIdentifierModificationSiteV0752::Residue { residue_index } => {
                if residue_index >= residue_count {
                    bail!(
                        "v0.75.2 residue modification index {} is outside sequence length {}",
                        residue_index,
                        residue_count
                    );
                }
                (
                    FoundationModificationSite::Residue(residue_index),
                    residue_index,
                )
            }
            FoundationPracticalIdentifierModificationSiteV0752::NTerm => {
                (FoundationModificationSite::NTerm, 0)
            }
            FoundationPracticalIdentifierModificationSiteV0752::CTerm => {
                (FoundationModificationSite::CTerm, residue_count - 1)
            }
        };
        let internal = match modification.unimod_id {
            Some(unimod_id) => FoundationModification::unimod(
                site,
                residue_index,
                unimod_id,
                modification.mass_delta,
            ),
            None => FoundationModification::mass_delta_at_site(
                site,
                residue_index,
                modification.mass_delta,
            ),
        };
        modifications.push(internal);
    }

    Ok(FoundationPracticalIdentifierCandidateV0751::new(
        candidate.key,
        PeptidoformInput {
            sequence: candidate.sequence,
            modifications,
        },
        candidate.charge,
    ))
}

fn validate_request_v0752(request: &FoundationPracticalIdentifierRequestV0752) -> Result<()> {
    if request.schema != FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752 {
        bail!(
            "v0.75.2 request schema mismatch: expected {}, observed {}",
            FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
            request.schema
        );
    }
    if request.query_id.trim().is_empty() {
        bail!("v0.75.2 query_id must be non-empty");
    }
    if request.observed_charge <= 0 {
        bail!("v0.75.2 observed precursor charge must be positive");
    }
    if !(request.observed_precursor_mz.is_finite() && request.observed_precursor_mz > 0.0) {
        bail!("v0.75.2 observed precursor m/z must be positive and finite");
    }
    if !(1..=FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751).contains(&request.top_k) {
        bail!(
            "v0.75.2 top_k must be in 1..={} and only truncates ranked output",
            FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751
        );
    }
    if !request.peaks.iter().any(|peak| {
        peak.mz.is_finite() && peak.mz > 0.0 && peak.intensity.is_finite() && peak.intensity > 0.0
    }) {
        bail!("v0.75.2 request must contain at least one finite positive spectrum peak");
    }
    Ok(())
}

fn catalog_fingerprint_v0752(
    candidates: &[FoundationPracticalIdentifierCatalogCandidateV0752],
) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    fn feed(hash: &mut u64, bytes: &[u8]) {
        for &byte in bytes {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x00000100000001b3);
        }
    }

    feed(
        &mut hash,
        FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752.as_bytes(),
    );
    for candidate in candidates {
        feed(&mut hash, &[0xff]);
        feed(&mut hash, candidate.key.as_bytes());
        feed(&mut hash, &[0xfe]);
        feed(&mut hash, candidate.sequence.as_bytes());
        feed(&mut hash, &candidate.charge.to_le_bytes());
        feed(
            &mut hash,
            &(candidate.modifications.len() as u64).to_le_bytes(),
        );
        for modification in &candidate.modifications {
            match modification.location {
                FoundationPracticalIdentifierModificationSiteV0752::Residue { residue_index } => {
                    feed(&mut hash, &[0]);
                    feed(&mut hash, &(residue_index as u64).to_le_bytes());
                }
                FoundationPracticalIdentifierModificationSiteV0752::NTerm => {
                    feed(&mut hash, &[1]);
                }
                FoundationPracticalIdentifierModificationSiteV0752::CTerm => {
                    feed(&mut hash, &[2]);
                }
            }
            feed(&mut hash, &modification.mass_delta.to_bits().to_le_bytes());
            match modification.unimod_id {
                Some(unimod_id) => {
                    feed(&mut hash, &[1]);
                    feed(&mut hash, &unimod_id.to_le_bytes());
                }
                None => feed(&mut hash, &[0]),
            }
        }
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        foundation_fragment_cleavage_geometry, foundation_peptidoform_neutral_mass,
    };

    fn synthetic_catalog() -> FoundationPracticalIdentifierCatalogV0752 {
        FoundationPracticalIdentifierCatalogV0752::new(vec![
            FoundationPracticalIdentifierCatalogCandidateV0752 {
                key: "PEPTIDE|z2".into(),
                sequence: "PEPTIDE".into(),
                modifications: Vec::new(),
                charge: 2,
            },
            FoundationPracticalIdentifierCatalogCandidateV0752 {
                key: "PEPTIDK|z2".into(),
                sequence: "PEPTIDK".into(),
                modifications: Vec::new(),
                charge: 2,
            },
            FoundationPracticalIdentifierCatalogCandidateV0752 {
                key: "PEPTIDE|z3".into(),
                sequence: "PEPTIDE".into(),
                modifications: Vec::new(),
                charge: 3,
            },
        ])
    }

    fn synthetic_request(top_k: usize) -> FoundationPracticalIdentifierRequestV0752 {
        let peptidoform = PeptidoformInput::unmodified("PEPTIDE");
        let geometry = foundation_fragment_cleavage_geometry(&peptidoform).unwrap();
        let neutral_mass = foundation_peptidoform_neutral_mass(&peptidoform).unwrap();
        let proton = 1.007_276_466_77f64;
        let precursor_mz = (neutral_mass + 2.0 * proton) / 2.0;
        FoundationPracticalIdentifierRequestV0752 {
            schema: default_schema_v0752(),
            query_id: "query-1".into(),
            observed_precursor_mz: precursor_mz,
            observed_charge: 2,
            peaks: geometry
                .iter()
                .flat_map(|row| row.core_mz)
                .map(|mz| FoundationPracticalIdentifierPeakV0752 {
                    mz: mz as f32,
                    intensity: 1.0,
                })
                .collect(),
            top_k,
        }
    }

    #[test]
    fn production_wrapper_preserves_frozen_core_ranking() {
        let catalog = synthetic_catalog();
        let internal_candidates = catalog
            .candidates
            .clone()
            .into_iter()
            .map(catalog_candidate_into_v0751)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let core = FoundationPracticalIdentifierV0751::new(internal_candidates).unwrap();
        let service = FoundationPracticalIdentifierServiceV0752::new(catalog).unwrap();
        let request = synthetic_request(2);
        let spectrum = FoundationSpectrum::from_pairs(
            request.peaks.iter().map(|peak| (peak.mz, peak.intensity)),
        );
        let core_hits = core
            .identify_precursor_mz(
                request.observed_charge,
                request.observed_precursor_mz,
                &spectrum,
            )
            .unwrap();
        let response = service.identify(&request).unwrap();

        assert_eq!(response.candidate_pool, 256);
        assert_eq!(response.candidates_scored, core_hits.len());
        assert_eq!(response.hits.len(), 2);
        for (public_hit, core_hit) in response.hits.iter().zip(core_hits.iter()) {
            assert_eq!(public_hit.rank, core_hit.rank);
            assert_eq!(public_hit.candidate_index, core_hit.candidate_index);
            assert_eq!(public_hit.key, core_hit.key);
            assert_eq!(public_hit.charge, core_hit.charge);
            assert_eq!(
                public_hit.theoretical_neutral_mass,
                core_hit.theoretical_neutral_mass
            );
            assert_eq!(
                public_hit.absolute_neutral_mass_error,
                core_hit.absolute_neutral_mass_error
            );
            assert_eq!(public_hit.geometry_score, core_hit.geometry_score);
        }
    }

    #[test]
    fn top_k_only_truncates_already_ranked_output() {
        let service = FoundationPracticalIdentifierServiceV0752::new(synthetic_catalog()).unwrap();
        let top1 = service.identify(&synthetic_request(1)).unwrap();
        let top2 = service.identify(&synthetic_request(2)).unwrap();
        assert_eq!(top1.candidate_pool, 256);
        assert_eq!(top2.candidate_pool, 256);
        assert_eq!(top1.candidates_scored, top2.candidates_scored);
        assert_eq!(top1.hits.as_slice(), &top2.hits[..1]);
    }

    #[test]
    fn yaml_contract_round_trips_catalog_and_request() {
        let catalog = synthetic_catalog();
        let catalog_yaml = catalog.to_yaml_string().unwrap();
        let round_trip_catalog =
            FoundationPracticalIdentifierCatalogV0752::from_yaml_str(&catalog_yaml).unwrap();
        assert_eq!(round_trip_catalog, catalog);

        let request = synthetic_request(10);
        let request_yaml = request.to_yaml_string().unwrap();
        let round_trip_request =
            FoundationPracticalIdentifierRequestV0752::from_yaml_str(&request_yaml).unwrap();
        assert_eq!(round_trip_request, request);

        let service =
            FoundationPracticalIdentifierServiceV0752::from_catalog_yaml_str(&catalog_yaml)
                .unwrap();
        let response_yaml = service.identify_yaml_str(&request_yaml).unwrap();
        let response: FoundationPracticalIdentifierResponseV0752 =
            serde_yaml::from_str(&response_yaml).unwrap();
        assert_eq!(
            response.schema,
            FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752
        );
        assert_eq!(response.query_id, request.query_id);
        assert!(!response.hits.is_empty());
    }

    #[test]
    fn catalog_fingerprint_is_deterministic_and_order_sensitive() {
        let catalog = synthetic_catalog();
        let service_a = FoundationPracticalIdentifierServiceV0752::new(catalog.clone()).unwrap();
        let service_b = FoundationPracticalIdentifierServiceV0752::new(catalog.clone()).unwrap();
        assert_eq!(
            service_a.catalog_fingerprint(),
            service_b.catalog_fingerprint()
        );

        let mut reordered = catalog;
        reordered.candidates.swap(0, 1);
        let service_c = FoundationPracticalIdentifierServiceV0752::new(reordered).unwrap();
        assert_ne!(
            service_a.catalog_fingerprint(),
            service_c.catalog_fingerprint()
        );
    }

    #[test]
    fn request_and_catalog_validation_fail_closed() {
        let missing_schema = "candidates: []\n";
        assert!(FoundationPracticalIdentifierCatalogV0752::from_yaml_str(missing_schema).is_err());

        let mut wrong_schema = synthetic_catalog();
        wrong_schema.schema = "wrong.schema".into();
        assert!(FoundationPracticalIdentifierServiceV0752::new(wrong_schema).is_err());

        let service = FoundationPracticalIdentifierServiceV0752::new(synthetic_catalog()).unwrap();
        let mut zero_top_k = synthetic_request(1);
        zero_top_k.top_k = 0;
        assert!(service.identify(&zero_top_k).is_err());

        let mut oversized_top_k = synthetic_request(1);
        oversized_top_k.top_k = 257;
        assert!(service.identify(&oversized_top_k).is_err());
    }

    #[test]
    fn identify_many_preserves_request_order() {
        let service = FoundationPracticalIdentifierServiceV0752::new(synthetic_catalog()).unwrap();
        let mut first = synthetic_request(1);
        first.query_id = "first".into();
        let mut second = synthetic_request(1);
        second.query_id = "second".into();
        let responses = service.identify_many(&[first, second]).unwrap();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].query_id, "first");
        assert_eq!(responses[1].query_id, "second");
    }
}
