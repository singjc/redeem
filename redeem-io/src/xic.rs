// redeem-io/src/xic.rs

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct XicPoint {
    pub rt: f32,
    pub intensity: f32,
}

#[derive(Debug, Clone)]
pub struct TransitionTrace {
    pub annotation: String,
    pub ordinal: i32,
    pub ms_level: Option<u8>,
    pub points: Vec<XicPoint>,
}

#[derive(Debug, Clone)]
pub struct PrecursorXic {
    pub precursor_id: u64,
    pub transitions: Vec<TransitionTrace>,
}

pub trait XicSource {
    fn fetch_precursors(&mut self, run_id: u64, precursor_ids: &[u64]) -> Result<Vec<PrecursorXic>>;
}
