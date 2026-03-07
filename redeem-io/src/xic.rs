//! Shared in-memory chromatogram types.
//!
//! These types represent already-decoded chromatogram data in a model-friendly
//! form. They intentionally do not expose parquet/SQLite details; callers see
//! only precursor-centric traces that can be windowed and turned into tensors.

use anyhow::Result;

/// One chromatogram sample.
///
/// A point consists of a retention-time coordinate and the corresponding
/// intensity value for one trace.
#[derive(Debug, Clone)]
pub struct XicPoint {
    pub rt: f32,
    pub intensity: f32,
}

/// A single transition or precursor trace.
///
/// For MS2 data this usually corresponds to one fragment-ion chromatogram. For
/// MS1 data it corresponds to one precursor/isotope trace.
#[derive(Debug, Clone)]
pub struct TransitionTrace {
    pub annotation: String,
    pub ordinal: i32,
    pub ms_level: Option<u8>,
    pub points: Vec<XicPoint>,
}

/// All traces associated with one precursor in one run.
///
/// TOPAZ expects the caller to later split these traces into MS1 and MS2
/// subsets, order them deterministically, and crop/pad them into fixed windows.
#[derive(Debug, Clone)]
pub struct PrecursorXic {
    pub precursor_id: u64,
    pub transitions: Vec<TransitionTrace>,
}

/// Abstract source of chromatograms keyed by `(run_id, precursor_id)`.
///
/// The main purpose of this trait is to decouple trace extraction logic from
/// the underlying storage format. An implementation may read from parquet,
/// cache files, an in-memory map, or any other backing store.
pub trait XicSource {
    /// Fetch chromatograms for the requested precursor IDs.
    ///
    /// The returned vector may be shorter than `precursor_ids` if some
    /// precursors are missing from the source.
    fn fetch_precursors(&mut self, run_id: u64, precursor_ids: &[u64])
    -> Result<Vec<PrecursorXic>>;
}
