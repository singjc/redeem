//! Optional source-level acquisition metadata.
//!
//! Transition libraries frequently omit instrument, NCE, run, or LC-method
//! fields even when a curator knows them from a publication or acquisition
//! manifest. This module allows such metadata to be attached without making it
//! mandatory. Missing metadata is a first-class state and never causes a load
//! or inference failure.

use super::dataset::FoundationDataset;
use serde::{Deserialize, Serialize};

/// Optional metadata that applies to a whole imported source/table.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationSourceMetadata {
    /// Normalized collision energy when it is genuinely known for the source.
    pub nce: Option<f32>,
    /// Instrument family/model label when known.
    pub instrument: Option<String>,
    /// Run/file identifier used for provenance, grouping, and leakage-safe splitting.
    pub run_id: Option<String>,
    /// LC gradient duration in seconds for future context-conditioned observed-RT heads.
    pub gradient_seconds: Option<f32>,
}

/// How source-level metadata interacts with values already present in rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoundationMetadataMergePolicy {
    /// Preserve row-level values and only fill fields that are missing.
    #[default]
    FillMissing,
    /// Replace row-level values when the source manifest provides a value.
    Override,
}

/// Counts describing an optional metadata application operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationMetadataApplicationStats {
    /// Number of records visited.
    pub records: usize,
    /// NCE fields assigned by the source metadata.
    pub nce_assignments: usize,
    /// Instrument fields assigned by the source metadata.
    pub instrument_assignments: usize,
    /// Run-id fields assigned by the source metadata.
    pub run_id_assignments: usize,
    /// LC-gradient fields assigned by the source metadata.
    pub gradient_seconds_assignments: usize,
}

/// Apply optional source metadata to an already loaded foundation dataset.
///
/// An empty/default metadata value is a valid no-op. This makes source
/// manifests additive rather than a prerequisite for training or inference.
pub fn apply_source_metadata(
    dataset: &mut FoundationDataset,
    metadata: &FoundationSourceMetadata,
    policy: FoundationMetadataMergePolicy,
) -> FoundationMetadataApplicationStats {
    let mut stats = FoundationMetadataApplicationStats {
        records: dataset.records.len(),
        ..FoundationMetadataApplicationStats::default()
    };

    for record in &mut dataset.records {
        if let Some(nce) = metadata.nce {
            if policy == FoundationMetadataMergePolicy::Override || record.context.nce.is_none() {
                record.context.nce = Some(nce);
                stats.nce_assignments += 1;
            }
        }

        if let Some(instrument) = metadata
            .instrument
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let existing = record
                .context
                .instrument_name
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty());
            if policy == FoundationMetadataMergePolicy::Override || !existing {
                let id = dataset.instruments.id_for(Some(instrument));
                record.context.instrument_id = Some(id);
                record.context.instrument_name = Some(instrument.to_string());
                stats.instrument_assignments += 1;
            }
        }

        if let Some(run_id) = metadata
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let existing = record
                .run_id
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty());
            if policy == FoundationMetadataMergePolicy::Override || !existing {
                record.run_id = Some(run_id.to_string());
                stats.run_id_assignments += 1;
            }
        }

        if let Some(gradient_seconds) = metadata.gradient_seconds {
            if policy == FoundationMetadataMergePolicy::Override
                || record.context.gradient_seconds.is_none()
            {
                record.context.gradient_seconds = Some(gradient_seconds);
                stats.gradient_seconds_assignments += 1;
            }
        }
    }

    stats
}
