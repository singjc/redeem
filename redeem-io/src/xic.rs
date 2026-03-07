//! Shared in-memory chromatogram types.

use anyhow::Result;

/// One chromatogram sample.
#[derive(Debug, Clone)]
pub struct XicPoint {
    pub rt: f32,
    pub intensity: f32,
}

/// A single transition or precursor trace.
#[derive(Debug, Clone)]
pub struct TransitionTrace {
    pub annotation: String,
    pub ordinal: i32,
    pub ms_level: Option<u8>,
    pub points: Vec<XicPoint>,
}

/// All traces associated with one precursor in one run.
#[derive(Debug, Clone)]
pub struct PrecursorXic {
    pub precursor_id: u64,
    pub transitions: Vec<TransitionTrace>,
}

/// Abstract source of chromatograms keyed by `(run_id, precursor_id)`.
pub trait XicSource {
    /// Fetch chromatograms for the requested precursor IDs.
    fn fetch_precursors(&mut self, run_id: u64, precursor_ids: &[u64]) -> Result<Vec<PrecursorXic>>;
}
