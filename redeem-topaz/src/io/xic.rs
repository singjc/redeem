//! Lightweight chromatogram domain types used by TOPAZ internals.
//!
//! These mirror the richer types in `redeem-io` but keep the TOPAZ crate
//! independent from any particular storage backend.

use anyhow::Result;

/// One chromatogram sample.
///
/// `rt` is the retention-time coordinate and `intensity` is the signal value at
/// that point for one trace.
#[derive(Debug, Clone)]
pub struct XicPoint {
    pub rt: f32,
    pub intensity: f32,
}

/// One chromatographic trace associated with a precursor.
///
/// A trace can represent either an MS1 precursor/isotope channel or an MS2
/// fragment-ion channel depending on `ms_level`.
#[derive(Debug, Clone)]
pub struct TransitionTrace {
    pub annotation: String,
    pub ordinal: i32,
    pub ms_level: Option<u8>,
    pub points: Vec<XicPoint>,
}

/// Collection of traces associated with a precursor in a run.
///
/// TOPAZ later sorts these traces deterministically, selects up to `cmax`
/// channels per modality, and converts them into fixed-size tensors.
#[derive(Debug, Clone)]
pub struct PrecursorXic {
    pub precursor_id: u64,
    pub transitions: Vec<TransitionTrace>,
}

/// Abstract source of precursor chromatograms.
///
/// The inference and training pipelines use this trait so that trace extraction
/// can work with parquet readers, caches, or synthetic test data.
pub trait XicSource {
    /// Fetch chromatograms for these precursor IDs in a run.
    ///
    /// The implementation may return fewer precursors than requested when some
    /// entries are missing from the data source.
    fn fetch_precursors(&mut self, run_id: u64, precursor_ids: &[u64]) -> Result<Vec<PrecursorXic>>;
}
