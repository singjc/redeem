//! Training-record types shared by transition-list and spectral-library loaders.
//!
//! This module deliberately separates intrinsic peptide labels from run-level
//! experimental context. Source-native normalized RT is retained for provenance;
//! an optional TRAIN-fit harmonized RT coordinate is the portable intrinsic target
//! across heterogeneous libraries. Observed RT can coexist as an auxiliary
//! context-conditioned target when LC metadata are available.

use super::featurize::PeptidoformInput;
use serde::{Deserialize, Serialize};

/// Retention-time labels that may coexist for one peptide observation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RetentionTimeLabels {
    /// Source-native normalized RT/iRT-like value exactly as provided by the library.
    ///
    /// Different public libraries may use incompatible numerical coordinate systems, so this
    /// field is retained for provenance/audit and is not necessarily a globally portable target.
    pub normalized: Option<f32>,
    /// Train-only cross-source harmonized intrinsic RT coordinate, when configured.
    ///
    /// This value is derived from `normalized` using source-specific affine transforms fit only
    /// on the materialized TRAIN partition. The raw source-native value above is never overwritten.
    pub harmonized: Option<f32>,
    /// Actual chromatographic retention time in seconds.
    pub observed_seconds: Option<f32>,
}

/// One raw observed centroided spectrum peak.
///
/// This is deliberately separate from [`FragmentTarget`]. Raw spectral-library
/// formats such as MSP may provide measured `(m/z, intensity)` pairs without
/// assigning those peaks to a b/y cleavage channel. Keeping the raw spectrum in
/// a separate field lets the inverse spectrum-to-peptide lane use the measured
/// evidence without turning unannotated peaks into fabricated forward-MS2 labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservedSpectrumPeak {
    /// Observed product-ion mass-to-charge ratio.
    pub mz: f32,
    /// Observed peak intensity in arbitrary units.
    pub intensity: f32,
}

/// One annotated fragment observation from a transition list or spectral library.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentTarget {
    /// Zero-based cleavage index (between residues `i` and `i + 1`).
    pub cleavage_index: usize,
    /// Output channel index (for example b1+, b2+, y1+, y2+, losses).
    pub channel: usize,
    /// Relative or normalized fragment intensity.
    pub intensity: f32,
    /// Observed/library product-ion m/z when the source explicitly provides it.
    ///
    /// This is retained for the inverse spectrum-to-peptide lane. It is never
    /// reconstructed from the known peptide sequence by the loader, because
    /// doing so would leak the training target into the inverse input.
    pub product_mz: Option<f32>,
}

/// Optional experiment context that may condition property-specific heads.
///
/// Every field is optional by design. Foundation encoding and property
/// inference must remain valid when acquisition metadata were never curated,
/// were lost during export, or are genuinely unknown at test time. Run/LC
/// provenance is retained here but is not injected into the intrinsic peptide
/// representation.
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
    /// Annotated fragment targets, if present. These may supervise the forward
    /// cleavage/channel MS2 head when their ion identity is known.
    pub fragments: Vec<FragmentTarget>,
    /// Raw observed spectrum peaks when the source provides an MS/MS peak list
    /// without trustworthy cleavage/channel annotations (for example MSP).
    ///
    /// The inverse spectrum-to-peptide lane consumes these peaks directly. The
    /// forward MS2 head intentionally ignores them.
    pub observed_spectrum_peaks: Vec<ObservedSpectrumPeak>,
    /// Acquisition and precursor context.
    pub context: TrainingContext,
    /// Optional run identifier used for grouped data splitting/calibration.
    pub run_id: Option<String>,
}

/// Which RT target a downstream training adapter should optimize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionTimeObjective {
    /// Train only against the source-native normalized RT/iRT coordinate.
    Normalized,
    /// Train against the train-only cross-source harmonized intrinsic RT coordinate.
    Harmonized,
    /// Train only against actual observed chromatographic RT.
    Observed,
    /// Use source-native normalized RT as the intrinsic task and observed RT as an auxiliary
    /// context-conditioned task when both are available.
    #[default]
    IntrinsicAndObserved,
}
