//! TOPAZ: a trace-first DIA peak-group scoring model implemented with Candle.
//!
//! The crate is structured around a few layers:
//!
//! - [`building_blocks`] contains reusable neural-network and preprocessing
//!   components such as trace encoders, coelution features, bagging utilities,
//!   and input transforms.
//! - [`model`] contains the concrete [`TopazBagRanker`] model that combines
//!   trace encoding, heuristic features, and MIL bag scoring.
//! - [`train`] contains loss functions, optimizers, schedulers, and dataset
//!   preparation helpers for fitting the model.
//! - [`infer`] contains scoring, statistics, diagnostics, and trace extraction
//!   helpers used during validation and full-data inference.
//! - [`xrun`] contains the cross-run calibration stage that learns per-run
//!   score deltas from sequences of `(bag_score, winner_hidden)` pairs.
//! - [`run`] exposes the high-level training, inference, and XRUN sweep entry
//!   points used by `redeem-cli`.
//!
//! Tensor-shape notation used throughout the crate:
//!
//! - `N`: number of candidate rows after flattening the OSW table.
//! - `B`: number of bags, where each bag is typically one `(run_id,
//!   precursor_id)` group.
//! - `K`: number of candidate slots per bag after bagging/padding.
//! - `D`: number of heuristic/library feature columns.
//! - `L`: fixed trace-window length in retention-time samples, this corresponds to the length of the peak-group i.e. 30-60 seconds depending on the dataset.
//! - `C`: number of channels in a specific trace tensor.
//! - `C_total`: total trace channels presented to the encoder, typically
//!   `ms1_cmax + ms2_cmax`.
//! - `E`: learned trace-embedding dimensionality.
//!
//! Model architecture summary:
//!
//! 1. Input traces are extracted as fixed windows with optional MS1 channels
//!    preceding MS2 channels.
//! 2. The [`building_blocks::conv_encoder::TraceEncoder`] embeds the traces and
//!    optionally augments them with coelution features.
//! 3. The [`building_blocks::mlp::CandidateScorer`] scores each candidate row.
//! 4. Bag-level multiple-instance learning uses a masked max over candidates to
//!    produce a precursor/run score.
//! 5. Optionally, XRUN learns precursor-aligned cross-run deltas that calibrate
//!    those bag scores after the base model has been trained.

pub mod building_blocks;
pub mod infer;
pub mod io;
pub mod model;
pub mod preprocess;
pub mod preprocessed;
pub mod run;
pub mod train;
pub mod xrun;

pub mod checkpoint;
pub mod config;
pub mod inspect;
pub mod model_interface;

pub use model::topaz::{TopazBagRanker, TopazConfig, TopazXimConfig};
pub use model_interface::{
    BagRankerInterface, BagRankerWithHiddenInterface, CandidateScorerInterface, ModelInterface,
};
pub use preprocess::Preprocessor;
pub use preprocessed::{
    PreprocessedBundleReader, PreprocessedBundleWriter, PreprocessedChunk, PreprocessedChunkMeta,
    PreprocessedDataset, PreprocessedManifest, PreprocessedProvenance,
};
pub use run::{
    DiagnosticsConfig, FeatureMode, FeatureSelectConfig, InferRunConfig, InferRunOutput,
    PreprocessRunConfig, PreprocessRunOutput, TrainRunConfig, TrainRunOutput, XrunRunConfig,
    XrunSweepConfig, XrunSweepRow, XrunTrainOnlyOutput, run_inference, run_preprocess,
    run_training, run_xrun_sweep, run_xrun_training,
};
pub use xrun::calibrator::{XrunAttentionCalibrator, XrunConfig};
