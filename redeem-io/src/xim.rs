//! Shared in-memory ion-mobilogram types.
//!
//! Unlike XICs, XIMs are candidate-specific: OpenSWATH extracts them at the
//! apex RT of a particular feature/peak-group. The primary key is therefore
//! `(run_id, feature_id)` rather than `(run_id, precursor_id)`.

use anyhow::Result;

/// One sampled point on a mobilogram.
#[derive(Debug, Clone)]
pub struct XimPoint {
    /// Mobility coordinate for this sample.
    pub mobility: f32,
    /// Signal intensity at the mobility coordinate.
    pub intensity: f32,
}

/// One mobilogram trace associated with a candidate feature.
#[derive(Debug, Clone)]
pub struct MobilogramTrace {
    /// Human-readable annotation, e.g. fragment label or precursor label.
    pub annotation: String,
    /// Ordinal used to sort channels deterministically.
    pub ordinal: i32,
    /// MS level for this trace, usually `1` or `2`.
    pub ms_level: Option<u8>,
    /// OpenMS `MOBILOGRAM_TYPE`, e.g. `ms1` or `ms2`.
    pub mobilogram_type: Option<String>,
    /// Sampled mobility/intensity points.
    pub points: Vec<XimPoint>,
}

/// All mobilograms associated with one candidate feature in one run.
#[derive(Debug, Clone)]
pub struct FeatureXim {
    /// OSW/OpenSWATH `FEATURE_ID`.
    pub feature_id: u64,
    /// Underlying precursor ID, when present in the parquet.
    ///
    /// Some large OpenMS XIM exports leave this column null even though the
    /// mobilogram is still uniquely identified by `(run_id, feature_id)`. In
    /// that case the reader falls back to `0` so callers can continue loading
    /// feature-keyed mobilograms without failing the entire run.
    pub precursor_id: u64,
    /// Candidate apex RT at which the mobilogram was extracted.
    pub feature_rt: f32,
    /// All mobilogram traces for the candidate.
    pub traces: Vec<MobilogramTrace>,
}

/// Abstract source of mobilograms keyed by `(run_id, feature_id)`.
pub trait XimSource {
    /// Fetch mobilograms for the requested feature IDs.
    ///
    /// The returned vector may be shorter than `feature_ids` if some features
    /// are missing from the source.
    fn fetch_features(&mut self, run_id: u64, feature_ids: &[u64]) -> Result<Vec<FeatureXim>>;
}
