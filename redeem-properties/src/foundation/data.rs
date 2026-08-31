//! Training-record types shared by transition-list and spectral-library loaders.
//!
//! This module deliberately separates intrinsic peptide labels from run-level
//! experimental context.  In particular, normalized RT is treated as the
//! portable intrinsic retention target, while observed RT can be retained as
//! an auxiliary context-conditioned target when LC metadata are available.

use super::featurize::PeptidoformInput;
use serde::{Deserialize, Serialize};

/// Retention-time labels that may coexist for one peptide observation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RetentionTimeLabels {
    /// Experiment-normalized RT/iRT-like value used by the portable RT head.
    pub normalized: Option<f32>,
    /// Actual chromatographic retention time in seconds.
    pub observed_seconds: Option<f32>,
}

/// One fragment observation from a transition list or spectral library.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentTarget {
    /// Zero-based cleavage index (between residues `i` and `i + 1`).
    pub cleavage_index: usize,
    /// Output channel index (for example b1+, b2+, y1+, y2+, losses).
    pub channel: usize,
    /// Relative or normalized fragment intensity.
    pub intensity: f32,
}

/// Experiment context that should condition a property head rather than the
/// intrinsic peptide embedding.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingContext {
    /// Precursor charge.
    pub charge: Option<i32>,
    /// Precursor m/z when available.
    pub precursor_mz: Option<f32>,
    /// Normalized collision energy.
    pub nce: Option<f32>,
    /// Stable instrument-category id assigned by the dataset adapter.
    pub instrument_id: Option<u32>,
    /// Original instrument label retained for checkpoint metadata/debugging.
    pub instrument_name: Option<String>,
    /// Ion mobility value when present in the source table.
    pub ion_mobility: Option<f32>,
    /// Optional LC gradient duration in seconds for future observed-RT heads.
    pub gradient_seconds: Option<f32>,
}

/// Heterogeneous training example consumed by a future data collator.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationTrainingRecord {
    /// Peptidoform identity and site-specific chemistry.
    pub peptidoform: PeptidoformInput,
    /// Retention labels, if present.
    pub retention_time: RetentionTimeLabels,
    /// Collision cross section, if present.
    pub ccs: Option<f32>,
    /// Fragment targets, if present.
    pub fragments: Vec<FragmentTarget>,
    /// Acquisition and precursor context.
    pub context: TrainingContext,
    /// Optional run identifier used for grouped data splitting/calibration.
    pub run_id: Option<String>,
}

/// Which RT target a downstream training adapter should optimize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionTimeObjective {
    /// Train only against normalized RT/iRT.
    Normalized,
    /// Train only against actual observed chromatographic RT.
    Observed,
    /// Use normalized RT as the intrinsic task and observed RT as an auxiliary
    /// context-conditioned task when both are available.
    #[default]
    IntrinsicAndObserved,
}
