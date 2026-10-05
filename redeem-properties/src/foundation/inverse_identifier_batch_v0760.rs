//! Production search-space materialization and batch-identification layer for the frozen practical identifier.
//!
//! v0.76.0 is an engineering integration only. It does not change the scientific behavior of
//! [`FoundationPracticalIdentifierV0751`] or the stable v0.75.2 request/result contract. It adds
//! deterministic expansion of peptidoforms across explicit charge states plus an ordered batch
//! request/response contract around one prebuilt v0.75.2 service.

use super::inverse_identifier_api_v0752::{
    FoundationPracticalIdentifierCatalogCandidateV0752, FoundationPracticalIdentifierCatalogV0752,
    FoundationPracticalIdentifierModificationV0752, FoundationPracticalIdentifierRequestV0752,
    FoundationPracticalIdentifierResponseV0752, FoundationPracticalIdentifierServiceV0752,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Public production integration version.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_VERSION_V0760: u32 = 760;
/// Stable search-space schema used before materialization into the v0.75.2 candidate catalog.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760: &str =
    "redeem.foundation.practical_identifier.search_space.v0760";
/// Stable ordered batch-request schema.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760: &str =
    "redeem.foundation.practical_identifier.batch.v0760";
/// Stable ordered batch-response schema.
pub const FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_RESPONSE_SCHEMA_V0760: &str =
    "redeem.foundation.practical_identifier.batch_response.v0760";

fn default_search_space_schema_v0760() -> String {
    FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760.to_string()
}

fn default_batch_schema_v0760() -> String {
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760.to_string()
}

fn default_batch_response_schema_v0760() -> String {
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_RESPONSE_SCHEMA_V0760.to_string()
}

/// One peptidoform search-space entry expanded deterministically over explicit charge states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierSearchSpaceEntryV0760 {
    /// Stable caller-provided peptidoform identifier. Materialized candidate keys are `{id}|z{charge}`.
    pub id: String,
    pub sequence: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modifications: Vec<FoundationPracticalIdentifierModificationV0752>,
    /// Positive, strictly increasing charge states. Input order is preserved during materialization.
    pub charges: Vec<i32>,
}

/// Serializable production search space that can be materialized into the frozen v0.75.2 catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierSearchSpaceV0760 {
    pub schema: String,
    pub peptidoforms: Vec<FoundationPracticalIdentifierSearchSpaceEntryV0760>,
}

impl FoundationPracticalIdentifierSearchSpaceV0760 {
    pub fn new(peptidoforms: Vec<FoundationPracticalIdentifierSearchSpaceEntryV0760>) -> Self {
        Self {
            schema: default_search_space_schema_v0760(),
            peptidoforms,
        }
    }

    pub fn from_yaml_str(input: &str) -> Result<Self> {
        serde_yaml::from_str(input)
            .context("failed to deserialize v0.76.0 practical-identifier search-space YAML")
    }

    pub fn to_yaml_string(&self) -> Result<String> {
        serde_yaml::to_string(self)
            .context("failed to serialize v0.76.0 practical-identifier search-space YAML")
    }

    /// Expand every peptidoform over its explicit charge states in stable input order.
    pub fn materialize_catalog(&self) -> Result<FoundationPracticalIdentifierCatalogV0752> {
        materialize_catalog_v0760(self)
    }
}

/// One ordered batch of existing v0.75.2 spectrum requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierBatchRequestV0760 {
    pub schema: String,
    pub requests: Vec<FoundationPracticalIdentifierRequestV0752>,
}

impl FoundationPracticalIdentifierBatchRequestV0760 {
    pub fn new(requests: Vec<FoundationPracticalIdentifierRequestV0752>) -> Self {
        Self {
            schema: default_batch_schema_v0760(),
            requests,
        }
    }

    pub fn from_yaml_str(input: &str) -> Result<Self> {
        serde_yaml::from_str(input)
            .context("failed to deserialize v0.76.0 practical-identifier batch YAML")
    }

    pub fn to_yaml_string(&self) -> Result<String> {
        serde_yaml::to_string(self)
            .context("failed to serialize v0.76.0 practical-identifier batch YAML")
    }
}

/// Ordered production response for one v0.76.0 batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoundationPracticalIdentifierBatchResponseV0760 {
    pub schema: String,
    pub catalog_fingerprint: String,
    pub catalog_size: usize,
    pub request_count: usize,
    pub total_candidates_scored: usize,
    pub total_returned_hits: usize,
    pub responses: Vec<FoundationPracticalIdentifierResponseV0752>,
}

impl FoundationPracticalIdentifierBatchResponseV0760 {
    pub fn to_yaml_string(&self) -> Result<String> {
        serde_yaml::to_string(self)
            .context("failed to serialize v0.76.0 practical-identifier batch response YAML")
    }
}

/// One prebuilt catalog/service reused across an ordered batch of spectra.
#[derive(Debug, Clone)]
pub struct FoundationPracticalIdentifierBatchServiceV0760 {
    service: FoundationPracticalIdentifierServiceV0752,
}

impl FoundationPracticalIdentifierBatchServiceV0760 {
    pub fn from_catalog(catalog: FoundationPracticalIdentifierCatalogV0752) -> Result<Self> {
        Ok(Self {
            service: FoundationPracticalIdentifierServiceV0752::new(catalog)?,
        })
    }

    pub fn from_catalog_yaml_str(input: &str) -> Result<Self> {
        Ok(Self {
            service: FoundationPracticalIdentifierServiceV0752::from_catalog_yaml_str(input)?,
        })
    }

    pub fn from_search_space(
        search_space: &FoundationPracticalIdentifierSearchSpaceV0760,
    ) -> Result<Self> {
        Self::from_catalog(search_space.materialize_catalog()?)
    }

    pub fn catalog_fingerprint(&self) -> &str {
        self.service.catalog_fingerprint()
    }

    pub fn catalog_size(&self) -> usize {
        self.service.catalog_size()
    }

    /// Identify requests in caller-provided order using one frozen prebuilt identifier service.
    pub fn identify_batch(
        &self,
        batch: &FoundationPracticalIdentifierBatchRequestV0760,
    ) -> Result<FoundationPracticalIdentifierBatchResponseV0760> {
        validate_batch_v0760(batch)?;
        let responses = self.service.identify_many(&batch.requests)?;
        let total_candidates_scored = responses
            .iter()
            .map(|response| response.candidates_scored)
            .sum();
        let total_returned_hits = responses
            .iter()
            .map(|response| response.returned_hits)
            .sum();

        Ok(FoundationPracticalIdentifierBatchResponseV0760 {
            schema: default_batch_response_schema_v0760(),
            catalog_fingerprint: self.service.catalog_fingerprint().to_string(),
            catalog_size: self.service.catalog_size(),
            request_count: responses.len(),
            total_candidates_scored,
            total_returned_hits,
            responses,
        })
    }
}

/// Deterministically expand a versioned peptidoform search space into the frozen v0.75.2 catalog.
pub fn materialize_catalog_v0760(
    search_space: &FoundationPracticalIdentifierSearchSpaceV0760,
) -> Result<FoundationPracticalIdentifierCatalogV0752> {
    if search_space.schema != FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760 {
        bail!(
            "v0.76.0 search-space schema mismatch: expected {}, observed {}",
            FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760,
            search_space.schema
        );
    }
    if search_space.peptidoforms.is_empty() {
        bail!("v0.76.0 search space must contain at least one peptidoform");
    }

    let mut seen_ids = BTreeSet::<String>::new();
    let mut seen_candidate_keys = BTreeSet::<String>::new();
    let mut candidates = Vec::new();

    for entry in &search_space.peptidoforms {
        if entry.id.trim().is_empty() {
            bail!("v0.76.0 peptidoform id must be non-empty");
        }
        if !seen_ids.insert(entry.id.clone()) {
            bail!("v0.76.0 duplicate peptidoform id {:?}", entry.id);
        }
        if entry.sequence.is_empty() {
            bail!("v0.76.0 peptidoform sequence must be non-empty");
        }
        if entry.charges.is_empty() {
            bail!("v0.76.0 peptidoform {:?} has no charge states", entry.id);
        }

        let mut previous_charge = None;
        for &charge in &entry.charges {
            if charge <= 0 {
                bail!("v0.76.0 charge states must be positive");
            }
            if previous_charge.is_some_and(|previous| charge <= previous) {
                bail!(
                    "v0.76.0 charge states for {:?} must be strictly increasing and unique",
                    entry.id
                );
            }
            previous_charge = Some(charge);

            let key = format!("{}|z{}", entry.id, charge);
            if !seen_candidate_keys.insert(key.clone()) {
                bail!("v0.76.0 duplicate materialized candidate key {key:?}");
            }
            candidates.push(FoundationPracticalIdentifierCatalogCandidateV0752 {
                key,
                sequence: entry.sequence.clone(),
                modifications: entry.modifications.clone(),
                charge,
            });
        }
    }

    Ok(FoundationPracticalIdentifierCatalogV0752::new(candidates))
}

fn validate_batch_v0760(batch: &FoundationPracticalIdentifierBatchRequestV0760) -> Result<()> {
    if batch.schema != FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760 {
        bail!(
            "v0.76.0 batch schema mismatch: expected {}, observed {}",
            FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760,
            batch.schema
        );
    }
    if batch.requests.is_empty() {
        bail!("v0.76.0 batch must contain at least one request");
    }

    let mut query_ids = BTreeSet::<String>::new();
    for request in &batch.requests {
        if !query_ids.insert(request.query_id.clone()) {
            bail!(
                "v0.76.0 duplicate query_id {:?} in one batch",
                request.query_id
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationPracticalIdentifierPeakV0752, FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
        FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
    };

    fn search_space() -> FoundationPracticalIdentifierSearchSpaceV0760 {
        FoundationPracticalIdentifierSearchSpaceV0760::new(vec![
            FoundationPracticalIdentifierSearchSpaceEntryV0760 {
                id: "pep-a".into(),
                sequence: "PEPTIDE".into(),
                modifications: Vec::new(),
                charges: vec![2, 3],
            },
            FoundationPracticalIdentifierSearchSpaceEntryV0760 {
                id: "pep-b".into(),
                sequence: "PEPTIDK".into(),
                modifications: Vec::new(),
                charges: vec![2],
            },
        ])
    }

    fn request(
        query_id: &str,
        charge: i32,
        top_k: usize,
    ) -> FoundationPracticalIdentifierRequestV0752 {
        FoundationPracticalIdentifierRequestV0752 {
            schema: FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752.to_string(),
            query_id: query_id.into(),
            observed_precursor_mz: 400.0,
            observed_charge: charge,
            peaks: vec![FoundationPracticalIdentifierPeakV0752 {
                mz: 100.0,
                intensity: 1.0,
            }],
            top_k,
        }
    }

    #[test]
    fn search_space_materializes_deterministically_in_input_order() {
        let catalog = search_space().materialize_catalog().unwrap();
        assert_eq!(catalog.candidates.len(), 3);
        assert_eq!(catalog.candidates[0].key, "pep-a|z2");
        assert_eq!(catalog.candidates[1].key, "pep-a|z3");
        assert_eq!(catalog.candidates[2].key, "pep-b|z2");
        assert_eq!(catalog.candidates[0].charge, 2);
        assert_eq!(catalog.candidates[1].charge, 3);
        assert_eq!(catalog.candidates[2].charge, 2);
    }

    #[test]
    fn search_space_rejects_duplicate_ids_and_noncanonical_charges() {
        let mut duplicate = search_space();
        duplicate.peptidoforms[1].id = "pep-a".into();
        assert!(duplicate.materialize_catalog().is_err());

        let mut unsorted = search_space();
        unsorted.peptidoforms[0].charges = vec![3, 2];
        assert!(unsorted.materialize_catalog().is_err());

        let mut repeated = search_space();
        repeated.peptidoforms[0].charges = vec![2, 2];
        assert!(repeated.materialize_catalog().is_err());
    }

    #[test]
    fn batch_service_matches_direct_v0752_order_and_results() {
        let catalog = search_space().materialize_catalog().unwrap();
        let direct = FoundationPracticalIdentifierServiceV0752::new(catalog.clone()).unwrap();
        let batch_service =
            FoundationPracticalIdentifierBatchServiceV0760::from_catalog(catalog).unwrap();
        let requests = vec![request("q2", 2, 2), request("q3", 3, 1)];
        let batch = FoundationPracticalIdentifierBatchRequestV0760::new(requests.clone());
        let response = batch_service.identify_batch(&batch).unwrap();
        let direct_responses = direct.identify_many(&requests).unwrap();

        assert_eq!(response.responses, direct_responses);
        assert_eq!(response.request_count, 2);
        assert_eq!(response.catalog_size, 3);
        assert_eq!(response.responses[0].query_id, "q2");
        assert_eq!(response.responses[1].query_id, "q3");
        assert_eq!(
            response.responses[0].candidate_pool,
            FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751
        );
        assert_eq!(
            response.responses[1].candidate_pool,
            FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751
        );
        assert_eq!(response.total_candidates_scored, 3);
        assert_eq!(response.total_returned_hits, 3);
    }

    #[test]
    fn yaml_contract_round_trips_search_space_and_batch() {
        let space = search_space();
        let round_trip_space = FoundationPracticalIdentifierSearchSpaceV0760::from_yaml_str(
            &space.to_yaml_string().unwrap(),
        )
        .unwrap();
        assert_eq!(space, round_trip_space);

        let batch = FoundationPracticalIdentifierBatchRequestV0760::new(vec![request("q", 2, 1)]);
        let round_trip_batch = FoundationPracticalIdentifierBatchRequestV0760::from_yaml_str(
            &batch.to_yaml_string().unwrap(),
        )
        .unwrap();
        assert_eq!(batch, round_trip_batch);
    }

    #[test]
    fn batch_rejects_duplicate_query_ids() {
        let service =
            FoundationPracticalIdentifierBatchServiceV0760::from_search_space(&search_space())
                .unwrap();
        let batch = FoundationPracticalIdentifierBatchRequestV0760::new(vec![
            request("duplicate", 2, 1),
            request("duplicate", 3, 1),
        ]);
        assert!(service.identify_batch(&batch).is_err());
    }
}
