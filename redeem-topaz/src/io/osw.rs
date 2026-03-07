//! Lightweight feature-row type used throughout TOPAZ.
//!
//! This type is the minimal row representation consumed by bagging, trace
//! extraction, training, and inference inside the TOPAZ crate.

/// Minimal per-candidate record extracted from an OSW file.
///
/// One `FeatureRow` corresponds to one candidate peak group produced by
/// OpenSWATH/PyProphet-style processing. Rows are later grouped by `group_id`
/// into bags for multiple-instance learning.
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub feature_id: u64,
    pub precursor_id: u64,
    pub run_id: u64,
    /// Bagging key. In the Python implementation this was often
    /// `RUN_ID_PRECURSOR_ID`.
    pub group_id: String,
    pub exp_rt: f32,
    pub is_decoy: bool,

    /// Selected scalar heuristic/library features for this candidate row.
    pub features: Vec<f32>,
}
