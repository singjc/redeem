//! High-level TOPAZ training, inference, and XRUN-sweep entry points.
//!
//! This module is the orchestration layer used by `redeem-cli`. It combines:
//!
//! - OSW feature loading
//! - XIC extraction and caching
//! - feature-column selection and preprocessing
//! - base TOPAZ training/inference
//! - optional XRUN training and calibrated inference
//! - OSW/TSV writeback and diagnostics

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use std::sync::mpsc;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use std::thread;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use std::time::{Duration, Instant};

use candle_core::Device;

use crate::checkpoint::{
    CheckpointMeta, XrunCheckpointMeta, load_checkpoint_partial, load_xrun_checkpoint,
    read_xrun_checkpoint_meta, save_xrun_checkpoint, xrun_checkpoint_exists,
};
use crate::config::Config as TrainConfig;
use crate::infer::{TraceBuildConfig, XicFetchConfig, XimFetchConfig};
use crate::io::osw::OswReadConfig;
use crate::model::topaz::TopazConfig;
use crate::preprocessed::{
    PreprocessedBundleReader, PreprocessedManifest, PreprocessedProvenance,
    ResumablePreprocessedBundleWriter,
};
use crate::train::TrainFilter;
use crate::xrun::XrunTrainConfig;
use crate::xrun::calibrator::{XrunAttentionCalibrator, XrunConfig};

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::building_blocks::bagging::make_bags_with_traces;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::checkpoint::save_checkpoint;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::diagnostics::{print_trace_summary, trace_summary, warn_if_missing_ms1};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::{
    SharedXicCache, SharedXimCache, XicDiskCache, XimDiskCache, build_score_table_from_rows,
    build_trace_tensors_from_parquet, build_trace_tensors_from_parquet_cached,
    build_trace_tensors_from_parquet_map, build_trace_tensors_from_parquet_map_cached,
    build_xim_tensors_from_parquet, build_xim_tensors_from_parquet_cached,
    build_xim_tensors_from_parquet_map, build_xim_tensors_from_parquet_map_cached,
    rows_to_feature_matrix_with_cols, score_candidates_with_aux, tdc_summary,
};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::io::osw::FeatureRow;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::io::osw::read_feature_rows;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::model::topaz::TopazBagRanker;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::train::{
    Trainer, bags_to_train_batches, bags_to_train_batches_with_aux, filter_training_rows,
    fit_preprocessor_from_rows_with_cols, split_rows_by_precursor, subsample_train_rows_by_bag,
};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::xrun::pipeline::{XrunPredictConfig, apply_xrun_deltas, apply_xrun_deltas_to_rows};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::xrun::sequence::build_xrun_sequences_from_bags;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::xrun::train::{XrunDataset, XrunPoolMode, XrunTrainer, split_train_val};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use candle_nn::{VarBuilder, VarMap};

#[cfg(feature = "io-sqlite")]
use crate::infer::write_rank1_disagreement_tsvs;
#[cfg(feature = "io-sqlite")]
use crate::io::osw::ScoreRow as OswScoreRow;

/// Optional numeric and file-based diagnostics emitted during training and
/// inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DiagnosticsConfig {
    pub trace_summary: bool,
    pub probe_ms1: bool,
    pub rank1_disagreements: bool,
    pub rank1_outdir: Option<PathBuf>,
    pub save_head_embeddings: bool,
    pub head_embeddings_outdir: Option<PathBuf>,
}

/// XRUN runtime/training configuration embedded inside the main TOPAZ config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct XrunRunConfig {
    pub enabled: bool,
    pub max_runs: usize,
    pub sort_by: String,
    pub batch_size: usize,
    pub train: XrunTrainConfig,
}

impl Default for XrunRunConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_runs: 64,
            sort_by: "run".to_string(),
            batch_size: 256,
            train: XrunTrainConfig::default(),
        }
    }
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            trace_summary: false,
            probe_ms1: false,
            rank1_disagreements: false,
            rank1_outdir: None,
            save_head_embeddings: false,
            head_embeddings_outdir: None,
        }
    }
}

pub const DEFAULT_LIB_COLS: &[&str] = &[
    "var_norm_rt_score",
    "var_library_corr",
    "var_library_dotprod",
    "var_library_manhattan",
    "var_library_rmsd",
    "var_library_rootmeansquare",
    "var_library_sangle",
];

/// Selection policy for heuristic/library feature columns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeatureMode {
    All,
    #[serde(alias = "lib", alias = "library", alias = "default")]
    DefaultLib,
    Custom,
    None,
}

impl Default for FeatureMode {
    fn default() -> Self {
        FeatureMode::All
    }
}

/// Configuration describing which heuristic feature columns are exposed to the
/// candidate scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureSelectConfig {
    pub mode: FeatureMode,
    pub cols: Option<Vec<String>>,
}

impl Default for FeatureSelectConfig {
    fn default() -> Self {
        Self {
            mode: FeatureMode::All,
            cols: None,
        }
    }
}

/// Top-level training configuration consumed by [`run_training`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainRunConfig {
    /// Input OSW SQLite file containing candidate feature rows.
    pub osw_path: PathBuf,
    /// Single XIC parquet path used when all requested runs live in one file.
    ///
    /// This is retained for backward compatibility with the original single-file
    /// interface. When `xic_paths` or `xic_map_path` is provided they take
    /// precedence.
    pub xic_path: PathBuf,
    /// Optional list of XIC parquet files.
    ///
    /// This is the common case for OpenSWATH exports where each run is written
    /// to its own parquet file. The pipeline will infer a run-to-path mapping by
    /// inspecting the parquet metadata.
    pub xic_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIC mapping file.
    ///
    /// Each row is expected to provide `run_id` and the parquet path to use for
    /// that run. This is the most reliable option when OSW `RUN_ID` values do
    /// not match the parquet-internal run IDs.
    pub xic_map_path: Option<PathBuf>,
    /// Single XIM parquet path used when all requested runs live in one file.
    pub xim_path: Option<PathBuf>,
    /// Optional list of XIM parquet files, typically one per run.
    pub xim_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIM mapping file.
    pub xim_map_path: Option<PathBuf>,
    /// Output prefix for the saved model checkpoint bundle.
    ///
    /// The base weights, metadata, and optional XRUN sidecar are written into
    /// `topaz.model` beneath this prefix.
    pub output_prefix: PathBuf,
    /// Optional checkpoint used to initialize the model before training.
    pub init_checkpoint: Option<PathBuf>,
    /// Optional preprocessed bundle generated by `topaz preprocess`.
    ///
    /// When provided, TOPAZ loads candidate rows plus fixed-width XIC/XIM
    /// tensors from this archive instead of re-reading the OSW/XIC/XIM raw
    /// inputs. The bundle must match the requested trace configuration and must
    /// contain every heuristic feature column required by the active model.
    pub preprocessed_path: Option<PathBuf>,
    /// Compute device string, e.g. `cpu`, `cuda`, or `cuda:0`.
    pub device: String,
    /// Maximum padded candidate count per bag.
    pub bag_k: usize,
    /// Number of bags per optimization step.
    pub batch_size: usize,
    /// Upper bound on training epochs before early stopping.
    pub max_epochs: usize,
    /// Fraction of bags held out for validation when no explicit split exists.
    pub val_frac: f32,
    /// Optional sub-sampling fraction applied to the training bags.
    pub train_frac: f32,
    /// Whether the train/validation split should preserve run balance.
    pub train_stratify_run: bool,
    /// Global RNG seed used for splitting, subsampling, and shuffling.
    pub seed: u64,
    /// Neural-network architecture and feature-fusion settings.
    pub model: TopazConfig,
    /// Optimizer, loss, and scheduler settings for base model training.
    pub train: TrainConfig,
    /// XIC extraction settings controlling window length and channel counts.
    pub trace: TraceBuildConfig,
    /// Optional XIM extraction settings. Required when `model.xim` is enabled.
    pub xim_trace: Option<TraceBuildConfig>,
    /// XIC parquet fetch filters such as `MS_LEVEL` and decoy handling.
    pub fetch: XicFetchConfig,
    /// XIM parquet fetch filters such as mobilogram type and decoy handling.
    pub xim_fetch: XimFetchConfig,
    /// OSW reader settings such as feature-table level and selected columns.
    pub osw: OswReadConfig,
    /// Optional row-level restriction applied before bagging/training.
    pub filter: TrainFilter,
    /// Policy for selecting scalar heuristic/library features from the OSW table.
    pub feature_select: FeatureSelectConfig,
    /// Controls extra diagnostics such as embedding export and disagreement tables.
    pub diagnostics: DiagnosticsConfig,
    /// If `true`, drop OSW rows whose `RUN_ID` is not represented in the XIC map.
    pub restrict_osw_to_xic_map: bool,
    /// Maximum number of decoded precursor chromatograms stored in memory.
    pub xic_cache_max_precursors: usize,
    /// Optional on-disk cache root for decoded XIC payloads.
    pub xic_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIC disk cache.
    pub xic_cache_max_bytes: Option<u64>,
    /// Maximum number of decoded feature mobilograms stored in memory.
    pub xim_cache_max_features: usize,
    /// Optional on-disk cache root for decoded XIM payloads.
    pub xim_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIM disk cache.
    pub xim_cache_max_bytes: Option<u64>,
    /// Number of candidate rows to process per inference-style trace chunk.
    ///
    /// Training primarily builds train/validation tensors once, but this value is
    /// also reused in shared utility paths that may need to chunk large row sets.
    pub trace_chunk_size: usize,
    /// Optional cross-run calibration stage trained after the base model.
    pub xrun: XrunRunConfig,
}

impl Default for TrainRunConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_paths: None,
            xic_map_path: None,
            xim_path: None,
            xim_paths: None,
            xim_map_path: None,
            output_prefix: PathBuf::from("topaz_checkpoint"),
            init_checkpoint: None,
            preprocessed_path: None,
            device: "cpu".to_string(),
            bag_k: 5,
            batch_size: 128,
            max_epochs: 5,
            val_frac: 0.1,
            train_frac: 1.0,
            train_stratify_run: false,
            seed: 0,
            model: TopazConfig::default(),
            train: TrainConfig::default(),
            trace: TraceBuildConfig {
                l: 64,
                ms1_cmax: 0,
                ms2_cmax: 6,
                normalize_max: true,
            },
            xim_trace: None,
            fetch: XicFetchConfig::default(),
            xim_fetch: XimFetchConfig::default(),
            osw: OswReadConfig::default(),
            filter: TrainFilter::default(),
            feature_select: FeatureSelectConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            restrict_osw_to_xic_map: false,
            xic_cache_max_precursors: 50_000,
            xic_cache_dir: None,
            xic_cache_max_bytes: None,
            xim_cache_max_features: 50_000,
            xim_cache_dir: None,
            xim_cache_max_bytes: None,
            trace_chunk_size: 5000,
            xrun: XrunRunConfig::default(),
        }
    }
}

/// Top-level inference configuration consumed by [`run_inference`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferRunConfig {
    /// Input OSW SQLite file containing the rows to score.
    pub osw_path: PathBuf,
    /// Single XIC parquet path used when all requested runs live in one file.
    pub xic_path: PathBuf,
    /// Optional list of XIC parquet files, typically one per run.
    pub xic_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIC mapping file.
    pub xic_map_path: Option<PathBuf>,
    /// Single XIM parquet path used when all requested runs live in one file.
    pub xim_path: Option<PathBuf>,
    /// Optional list of XIM parquet files, typically one per run.
    pub xim_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIM mapping file.
    pub xim_map_path: Option<PathBuf>,
    /// Trained checkpoint prefix or `topaz.model` archive to load.
    pub checkpoint: PathBuf,
    /// Optional preprocessed bundle generated by `topaz preprocess`.
    ///
    /// When set, inference loads row-aligned XIC/XIM tensors directly from the
    /// bundle and skips raw parquet decoding. The manifest is validated against
    /// the checkpoint before scoring starts.
    pub preprocessed_path: Option<PathBuf>,
    /// TSV file written with per-row TOPAZ scores and derived statistics.
    pub output_tsv: PathBuf,
    /// Optional OSW SQLite file that should receive score-table writeback.
    pub output_osw: Option<PathBuf>,
    /// Default output score-table name used when writing a single table.
    pub output_table: String,
    /// Optional score-table name for uncalibrated base TOPAZ scores.
    pub output_table_base: Option<String>,
    /// Optional score-table name for XRUN-calibrated TOPAZ scores.
    pub output_table_xrun: Option<String>,
    /// Compute device string, e.g. `cpu`, `cuda`, or `cuda:0`.
    pub device: String,
    /// Number of bags scored per forward batch.
    pub batch_size: usize,
    /// Number of bins used by the simple PEP estimator.
    pub pep_bins: usize,
    /// Maximum padded candidate count per bag.
    pub bag_k: usize,
    /// XIC extraction settings controlling window length and channel counts.
    pub trace: TraceBuildConfig,
    /// Optional XIM extraction settings used when the checkpoint enables XIM.
    pub xim_trace: Option<TraceBuildConfig>,
    /// XIC parquet fetch filters.
    pub fetch: XicFetchConfig,
    /// XIM parquet fetch filters.
    pub xim_fetch: XimFetchConfig,
    /// OSW reader settings such as feature-table level and selected columns.
    pub osw: OswReadConfig,
    /// Controls optional report generation and diagnostic outputs.
    pub diagnostics: DiagnosticsConfig,
    /// If `true`, drop OSW rows whose `RUN_ID` is not represented in the XIC map.
    pub restrict_osw_to_xic_map: bool,
    /// Maximum number of decoded precursor chromatograms stored in memory.
    pub xic_cache_max_precursors: usize,
    /// Optional on-disk cache root for decoded XIC payloads.
    pub xic_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIC disk cache.
    pub xic_cache_max_bytes: Option<u64>,
    /// Maximum number of decoded feature mobilograms stored in memory.
    pub xim_cache_max_features: usize,
    /// Optional on-disk cache root for decoded XIM payloads.
    pub xim_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIM disk cache.
    pub xim_cache_max_bytes: Option<u64>,
    /// Candidate-row chunk size used to keep full-dataset inference bounded in memory.
    pub trace_chunk_size: usize,
    /// If `true`, pre-build the full inference XIC/XIM tensors once and only
    /// chunk the model forward passes.
    ///
    /// This usually speeds up large inference jobs because parquet decoding and
    /// cache lookup happen once per run instead of once per scoring chunk, but
    /// it increases peak host RAM usage.
    pub prefetch_traces_once: bool,
    /// If `true`, overlap chunk-wise XIC/XIM loading with GPU scoring.
    ///
    /// This keeps host RAM bounded like the legacy chunked path while reducing
    /// the long "GPU idle while CPU decodes parquet" phase. The flag is ignored
    /// when `prefetch_traces_once` is enabled, because the full-dataset prefetch
    /// path already loads everything eagerly.
    pub stream_inference: bool,
    /// If `true`, run a score-only inference pass.
    ///
    /// Fast inference skips expensive post-processing stages that are useful
    /// for analysis but not required to write the base score TSV/OSW tables:
    ///
    /// - XRUN calibration application
    /// - head-embedding export
    /// - automatic HTML report generation in `redeem-cli`
    ///
    /// This is the recommended mode for large production-scoring jobs where
    /// the primary goal is to materialize score tables quickly and defer
    /// diagnostics to a later `topaz report` command.
    pub fast_inference: bool,
    /// XRUN loading/application settings.
    pub xrun: XrunRunConfig,
}

impl Default for InferRunConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_paths: None,
            xic_map_path: None,
            xim_path: None,
            xim_paths: None,
            xim_map_path: None,
            checkpoint: PathBuf::from("topaz_checkpoint"),
            preprocessed_path: None,
            output_tsv: PathBuf::from("score_topaz.tsv"),
            output_osw: None,
            output_table: "SCORE_TOPAZ".to_string(),
            output_table_base: None,
            output_table_xrun: None,
            device: "cpu".to_string(),
            batch_size: 256,
            pep_bins: 20,
            bag_k: 5,
            trace: TraceBuildConfig {
                l: 64,
                ms1_cmax: 0,
                ms2_cmax: 6,
                normalize_max: true,
            },
            xim_trace: None,
            fetch: XicFetchConfig::default(),
            xim_fetch: XimFetchConfig::default(),
            osw: OswReadConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            restrict_osw_to_xic_map: false,
            xic_cache_max_precursors: 50_000,
            xic_cache_dir: None,
            xic_cache_max_bytes: None,
            xim_cache_max_features: 50_000,
            xim_cache_dir: None,
            xim_cache_max_bytes: None,
            trace_chunk_size: 5000,
            prefetch_traces_once: false,
            stream_inference: false,
            fast_inference: false,
            xrun: XrunRunConfig::default(),
        }
    }
}

/// Top-level preprocessing configuration consumed by [`run_preprocess`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PreprocessRunConfig {
    /// Input OSW SQLite file containing the candidate rows to materialize.
    pub osw_path: PathBuf,
    /// Single XIC parquet path used when all requested runs live in one file.
    pub xic_path: PathBuf,
    /// Optional list of XIC parquet files, typically one per run.
    pub xic_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIC mapping file.
    pub xic_map_path: Option<PathBuf>,
    /// Single XIM parquet path used when all requested runs live in one file.
    pub xim_path: Option<PathBuf>,
    /// Optional list of XIM parquet files, typically one per run.
    pub xim_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIM mapping file.
    pub xim_map_path: Option<PathBuf>,
    /// Output archive path, typically ending in `.topazdata`.
    pub output_path: PathBuf,
    /// XIC extraction settings controlling the stored `(N, C, L)` tensor shape.
    pub trace: TraceBuildConfig,
    /// Optional XIM extraction settings. When omitted the bundle stores only
    /// XIC tensors.
    pub xim_trace: Option<TraceBuildConfig>,
    /// XIC parquet fetch filters.
    pub fetch: XicFetchConfig,
    /// XIM parquet fetch filters.
    pub xim_fetch: XimFetchConfig,
    /// OSW reader settings such as feature-table level and selected columns.
    pub osw: OswReadConfig,
    /// If `true`, drop OSW rows whose `RUN_ID` is not represented in the XIC map.
    pub restrict_osw_to_xic_map: bool,
    /// Maximum number of decoded precursor chromatograms stored in memory.
    pub xic_cache_max_precursors: usize,
    /// Optional on-disk cache root for decoded XIC payloads.
    pub xic_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIC disk cache.
    pub xic_cache_max_bytes: Option<u64>,
    /// Maximum number of decoded feature mobilograms stored in memory.
    pub xim_cache_max_features: usize,
    /// Optional on-disk cache root for decoded XIM payloads.
    pub xim_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIM disk cache.
    pub xim_cache_max_bytes: Option<u64>,
    /// Number of rows written to each archive chunk.
    pub chunk_row_count: usize,
}

impl Default for PreprocessRunConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_paths: None,
            xic_map_path: None,
            xim_path: None,
            xim_paths: None,
            xim_map_path: None,
            output_path: PathBuf::from("topaz_inputs.topazdata"),
            trace: TraceBuildConfig {
                l: 64,
                ms1_cmax: 0,
                ms2_cmax: 6,
                normalize_max: true,
            },
            xim_trace: None,
            fetch: XicFetchConfig::default(),
            xim_fetch: XimFetchConfig::default(),
            osw: OswReadConfig::default(),
            restrict_osw_to_xic_map: false,
            xic_cache_max_precursors: 50_000,
            xic_cache_dir: None,
            xic_cache_max_bytes: None,
            xim_cache_max_features: 50_000,
            xim_cache_dir: None,
            xim_cache_max_bytes: None,
            chunk_row_count: 50_000,
        }
    }
}

/// Result returned after a successful training run.
#[derive(Debug, Clone)]
pub struct TrainRunOutput {
    pub checkpoint_prefix: PathBuf,
}

/// Result returned after preprocessing raw OSW/XIC/XIM inputs.
#[derive(Debug, Clone)]
pub struct PreprocessRunOutput {
    pub output_path: PathBuf,
    pub n_rows: usize,
}

/// Result returned after a successful inference run.
#[derive(Debug, Clone)]
pub struct InferRunOutput {
    pub n_rows: usize,
}

/// Configuration for evaluating multiple XRUN hyper-parameter combinations from
/// a fixed base TOPAZ checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct XrunSweepConfig {
    /// Input OSW SQLite file containing rows aligned across runs.
    pub osw_path: PathBuf,
    /// Single XIC parquet path used when all requested runs live in one file.
    pub xic_path: PathBuf,
    /// Optional list of XIC parquet files, typically one per run.
    pub xic_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIC mapping file.
    pub xic_map_path: Option<PathBuf>,
    /// Single XIM parquet path used when all requested runs live in one file.
    pub xim_path: Option<PathBuf>,
    /// Optional list of XIM parquet files, typically one per run.
    pub xim_paths: Option<Vec<PathBuf>>,
    /// Optional explicit run-to-XIM mapping file.
    pub xim_map_path: Option<PathBuf>,
    /// Base TOPAZ checkpoint used to generate winner embeddings and bag scores.
    pub checkpoint: PathBuf,
    /// Optional preprocessed bundle generated by `topaz preprocess`.
    ///
    /// XRUN-only workflows can use the same reusable archive as inference and
    /// training, avoiding repeated raw XIC/XIM decode before winner-hidden
    /// extraction.
    pub preprocessed_path: Option<PathBuf>,
    /// TSV file summarizing each XRUN hyper-parameter combination.
    pub output_tsv: PathBuf,
    /// Compute device string, e.g. `cpu`, `cuda`, or `cuda:0`.
    pub device: String,
    /// XIC extraction settings controlling window length and channel counts.
    pub trace: TraceBuildConfig,
    /// Optional XIM extraction settings used when the checkpoint enables XIM.
    pub xim_trace: Option<TraceBuildConfig>,
    /// XIC parquet fetch filters.
    pub fetch: XicFetchConfig,
    /// XIM parquet fetch filters.
    pub xim_fetch: XimFetchConfig,
    /// OSW reader settings such as feature-table level and selected columns.
    pub osw: OswReadConfig,
    /// Maximum padded candidate count per bag.
    pub bag_k: usize,
    /// Number of bags scored per forward batch while building XRUN sequences.
    pub batch_size: usize,
    /// Fraction of aligned precursors held out for XRUN validation.
    pub val_frac: f32,
    /// Global RNG seed used for XRUN splitting and training.
    pub seed: u64,
    /// Hard cap on the number of runs per precursor sequence.
    pub max_runs: usize,
    /// Sort mode used when constructing precursor-aligned run sequences.
    pub sort_by: String,
    /// XRUN trainer hyper-parameters shared by every sweep point.
    pub train: XrunTrainConfig,
    /// Optional set of pooling modes to evaluate.
    pub sweep_pools: Option<Vec<String>>,
    /// Optional set of temperature values to evaluate.
    pub sweep_taus: Option<Vec<f64>>,
    /// If `true`, drop OSW rows whose `RUN_ID` is not represented in the XIC map.
    pub restrict_osw_to_xic_map: bool,
    /// Maximum number of decoded precursor chromatograms stored in memory.
    pub xic_cache_max_precursors: usize,
    /// Optional on-disk cache root for decoded XIC payloads.
    pub xic_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIC disk cache.
    pub xic_cache_max_bytes: Option<u64>,
    /// Maximum number of decoded feature mobilograms stored in memory.
    pub xim_cache_max_features: usize,
    /// Optional on-disk cache root for decoded XIM payloads.
    pub xim_cache_dir: Option<PathBuf>,
    /// Optional byte budget for the XIM disk cache.
    pub xim_cache_max_bytes: Option<u64>,
}

impl Default for XrunSweepConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_paths: None,
            xic_map_path: None,
            xim_path: None,
            xim_paths: None,
            xim_map_path: None,
            checkpoint: PathBuf::from("topaz_checkpoint"),
            preprocessed_path: None,
            output_tsv: PathBuf::from("xrun_sweep.tsv"),
            device: "cpu".to_string(),
            trace: TraceBuildConfig {
                l: 64,
                ms1_cmax: 0,
                ms2_cmax: 6,
                normalize_max: true,
            },
            xim_trace: None,
            fetch: XicFetchConfig::default(),
            xim_fetch: XimFetchConfig::default(),
            osw: OswReadConfig::default(),
            bag_k: 5,
            batch_size: 256,
            val_frac: 0.2,
            seed: 0,
            max_runs: 64,
            sort_by: "run".to_string(),
            train: XrunTrainConfig::default(),
            sweep_pools: None,
            sweep_taus: None,
            restrict_osw_to_xic_map: false,
            xic_cache_max_precursors: 50_000,
            xic_cache_dir: None,
            xic_cache_max_bytes: None,
            xim_cache_max_features: 50_000,
            xim_cache_dir: None,
            xim_cache_max_bytes: None,
        }
    }
}

/// One row of XRUN sweep output.
#[derive(Debug, Clone)]
pub struct XrunSweepRow {
    pub pool: String,
    pub tau: f64,
    pub best_val: f32,
}

/// Result returned after training an XRUN calibrator from a saved base
/// checkpoint without re-running base TOPAZ training.
#[derive(Debug, Clone)]
pub struct XrunTrainOnlyOutput {
    pub checkpoint_prefix: PathBuf,
    pub best_val: f32,
}

fn resolve_feature_cols(osw_cols: &[String], cfg: &FeatureSelectConfig) -> Vec<String> {
    match cfg.mode {
        FeatureMode::All => return osw_cols.iter().map(|c| c.to_lowercase()).collect(),
        FeatureMode::None => return Vec::new(),
        _ => {}
    }

    let osw_set: HashSet<String> = osw_cols.iter().map(|c| c.to_lowercase()).collect();
    let mut out: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    let mut push_col = |name: &str| {
        let lc = name.to_lowercase();
        if out.iter().any(|c| c == &lc) {
            return;
        }
        if !osw_set.contains(&lc) {
            missing.push(lc.clone());
        }
        out.push(lc);
    };

    match cfg.mode {
        FeatureMode::DefaultLib => {
            for &c in DEFAULT_LIB_COLS {
                push_col(c);
            }
        }
        FeatureMode::Custom => {
            let Some(cols) = &cfg.cols else {
                log::warn!("feature_select.mode=custom but no cols provided");
                return Vec::new();
            };
            if cols.is_empty() {
                log::warn!("feature_select.cols is empty; disabling heuristic features");
                return Vec::new();
            }
            for c in cols {
                push_col(c);
            }
        }
        FeatureMode::All | FeatureMode::None => {}
    }

    if !missing.is_empty() {
        log::warn!(
            "requested feature columns not found in OSW (will be imputed): {}",
            missing.join(", ")
        );
    }
    out
}

fn align_rows_to_cols(
    rows: &[crate::io::osw::FeatureRow],
    osw_cols: &[String],
    target_cols: &[String],
) -> Vec<crate::io::osw::FeatureRow> {
    let mut map = std::collections::HashMap::new();
    for (i, name) in osw_cols.iter().enumerate() {
        map.insert(name.as_str(), i);
    }
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let mut feat = vec![f32::NAN; target_cols.len()];
        for (j, name) in target_cols.iter().enumerate() {
            if let Some(&k) = map.get(name.as_str()) {
                if k < r.features.len() {
                    feat[j] = r.features[k];
                }
            }
        }
        let mut rr = r.clone();
        rr.features = feat;
        out.push(rr);
    }
    out
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn filter_rows_by_trace_with_aux(
    rows: Vec<crate::io::osw::FeatureRow>,
    x_trace: Vec<f32>,
    x_aux: Option<Vec<f32>>,
    c_total: usize,
    l: usize,
    aux_c_total: Option<usize>,
    aux_l: Option<usize>,
) -> (Vec<crate::io::osw::FeatureRow>, Vec<f32>, Option<Vec<f32>>) {
    if rows.is_empty() {
        return (rows, x_trace, x_aux);
    }
    let mut out_rows = Vec::new();
    let mut out_trace = Vec::new();
    let mut out_aux = x_aux.as_ref().map(|_| Vec::new());
    let span = c_total * l;
    let aux_span = aux_c_total.zip(aux_l).map(|(c, l)| c * l);

    for (i, row) in rows.into_iter().enumerate() {
        let start = i * span;
        let end = start + span;
        let slice = &x_trace[start..end];
        let mut max_v = 0f32;
        for &v in slice {
            let a = v.abs();
            if a > max_v {
                max_v = a;
            }
        }
        if max_v > 0.0 {
            out_rows.push(row);
            out_trace.extend_from_slice(slice);
            if let (Some(aux_src), Some(aux_dst), Some(aux_span)) =
                (x_aux.as_ref(), out_aux.as_mut(), aux_span)
            {
                let aux_start = i * aux_span;
                let aux_end = aux_start + aux_span;
                aux_dst.extend_from_slice(&aux_src[aux_start..aux_end]);
            }
        }
    }
    (out_rows, out_trace, out_aux)
}

fn effective_xim_trace_cfg(
    explicit: &Option<TraceBuildConfig>,
    model_cfg: &TopazConfig,
) -> Option<TraceBuildConfig> {
    explicit.clone().or_else(|| {
        model_cfg.xim.as_ref().map(|xim| TraceBuildConfig {
            l: xim.l,
            ms1_cmax: xim.ms1_cmax,
            ms2_cmax: xim.ms2_cmax,
            normalize_max: true,
        })
    })
}

fn same_trace_cfg(a: &TraceBuildConfig, b: &TraceBuildConfig) -> bool {
    a.l == b.l
        && a.ms1_cmax == b.ms1_cmax
        && a.ms2_cmax == b.ms2_cmax
        && a.normalize_max == b.normalize_max
}

fn same_osw_cfg(a: &OswReadConfig, b: &OswReadConfig) -> bool {
    a.level == b.level
        && a.ipf_max_rank == b.ipf_max_rank
        && (a.ipf_max_pep - b.ipf_max_pep).abs() < f32::EPSILON
        && (a.ipf_max_transition_isotope_overlap - b.ipf_max_transition_isotope_overlap).abs()
            < f32::EPSILON
        && (a.ipf_min_transition_sn - b.ipf_min_transition_sn).abs() < f32::EPSILON
}

fn missing_feature_cols(available: &[String], required: &[String]) -> Vec<String> {
    let available: HashSet<&str> = available.iter().map(|s| s.as_str()).collect();
    let mut missing: Vec<String> = required
        .iter()
        .filter(|name| !available.contains(name.as_str()))
        .cloned()
        .collect();
    missing.sort();
    missing.dedup();
    missing
}

fn validate_preprocessed_manifest_for_training(
    manifest: &PreprocessedManifest,
    cfg: &TrainRunConfig,
    model_cfg: &TopazConfig,
    selected_cols: &[String],
) -> Result<()> {
    if !same_trace_cfg(&manifest.trace, &cfg.trace) {
        bail!(
            "preprocessed trace configuration mismatch: bundle has l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}, config requests l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}",
            manifest.trace.l,
            manifest.trace.ms1_cmax,
            manifest.trace.ms2_cmax,
            manifest.trace.normalize_max,
            cfg.trace.l,
            cfg.trace.ms1_cmax,
            cfg.trace.ms2_cmax,
            cfg.trace.normalize_max
        );
    }
    let expected_xim = effective_xim_trace_cfg(&cfg.xim_trace, model_cfg);
    match (manifest.xim_trace.as_ref(), expected_xim.as_ref()) {
        (Some(bundle), Some(expected)) if !same_trace_cfg(bundle, expected) => {
            bail!(
                "preprocessed XIM configuration mismatch: bundle has l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}, config expects l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}",
                bundle.l,
                bundle.ms1_cmax,
                bundle.ms2_cmax,
                bundle.normalize_max,
                expected.l,
                expected.ms1_cmax,
                expected.ms2_cmax,
                expected.normalize_max
            );
        }
        (None, Some(_)) => {
            bail!("preprocessed bundle does not contain XIM data required by the current model")
        }
        (Some(_), None) if model_cfg.xim.is_none() => {
            log::debug!(
                "preprocessed bundle contains XIM data but the active training model does not use it"
            );
        }
        _ => {}
    }
    if !same_osw_cfg(&manifest.osw, &cfg.osw) {
        bail!("preprocessed bundle was created with a different OSW reader configuration");
    }
    let missing = missing_feature_cols(&manifest.feature_cols, selected_cols);
    if !missing.is_empty() {
        bail!(
            "preprocessed bundle is missing required heuristic feature columns: {}",
            missing.join(", ")
        );
    }
    Ok(())
}

fn validate_preprocessed_manifest_for_inference(
    manifest: &PreprocessedManifest,
    cfg: &InferRunConfig,
    meta: &CheckpointMeta,
) -> Result<()> {
    if !same_trace_cfg(&manifest.trace, &cfg.trace) {
        bail!(
            "preprocessed trace configuration mismatch: bundle has l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}, config requests l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}",
            manifest.trace.l,
            manifest.trace.ms1_cmax,
            manifest.trace.ms2_cmax,
            manifest.trace.normalize_max,
            cfg.trace.l,
            cfg.trace.ms1_cmax,
            cfg.trace.ms2_cmax,
            cfg.trace.normalize_max
        );
    }
    let expected_xim = effective_xim_trace_cfg(&cfg.xim_trace, &meta.model);
    match (manifest.xim_trace.as_ref(), expected_xim.as_ref()) {
        (Some(bundle), Some(expected)) if !same_trace_cfg(bundle, expected) => {
            bail!(
                "preprocessed XIM configuration mismatch: bundle has l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}, checkpoint expects l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}",
                bundle.l,
                bundle.ms1_cmax,
                bundle.ms2_cmax,
                bundle.normalize_max,
                expected.l,
                expected.ms1_cmax,
                expected.ms2_cmax,
                expected.normalize_max
            );
        }
        (None, Some(_)) => {
            bail!("preprocessed bundle does not contain XIM data required by the checkpoint")
        }
        _ => {}
    }
    if !same_osw_cfg(&manifest.osw, &cfg.osw) {
        bail!("preprocessed bundle was created with a different OSW reader configuration");
    }
    let missing = missing_feature_cols(&manifest.feature_cols, &meta.feature_cols);
    if !missing.is_empty() {
        bail!(
            "preprocessed bundle is missing checkpoint feature columns: {}",
            missing.join(", ")
        );
    }
    Ok(())
}

fn validate_preprocessed_manifest_for_xrun(
    manifest: &PreprocessedManifest,
    cfg: &XrunSweepConfig,
    meta: &CheckpointMeta,
) -> Result<()> {
    if !same_trace_cfg(&manifest.trace, &cfg.trace) {
        bail!(
            "preprocessed trace configuration mismatch: bundle has l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}, config requests l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}",
            manifest.trace.l,
            manifest.trace.ms1_cmax,
            manifest.trace.ms2_cmax,
            manifest.trace.normalize_max,
            cfg.trace.l,
            cfg.trace.ms1_cmax,
            cfg.trace.ms2_cmax,
            cfg.trace.normalize_max
        );
    }
    let expected_xim = effective_xim_trace_cfg(&cfg.xim_trace, &meta.model);
    match (manifest.xim_trace.as_ref(), expected_xim.as_ref()) {
        (Some(bundle), Some(expected)) if !same_trace_cfg(bundle, expected) => {
            bail!(
                "preprocessed XIM configuration mismatch: bundle has l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}, checkpoint expects l={}, ms1_cmax={}, ms2_cmax={}, normalize_max={}",
                bundle.l,
                bundle.ms1_cmax,
                bundle.ms2_cmax,
                bundle.normalize_max,
                expected.l,
                expected.ms1_cmax,
                expected.ms2_cmax,
                expected.normalize_max
            );
        }
        (None, Some(_)) => {
            bail!("preprocessed bundle does not contain XIM data required by the checkpoint")
        }
        _ => {}
    }
    if !same_osw_cfg(&manifest.osw, &cfg.osw) {
        bail!("preprocessed bundle was created with a different OSW reader configuration");
    }
    let missing = missing_feature_cols(&manifest.feature_cols, &meta.feature_cols);
    if !missing.is_empty() {
        bail!(
            "preprocessed bundle is missing checkpoint feature columns: {}",
            missing.join(", ")
        );
    }
    Ok(())
}

fn build_feature_row_index(rows: &[FeatureRow]) -> HashMap<u64, usize> {
    rows.iter()
        .enumerate()
        .map(|(idx, row)| (row.feature_id, idx))
        .collect()
}

fn gather_row_aligned_tensor_by_feature_id(
    source_rows: &[FeatureRow],
    row_index: &HashMap<u64, usize>,
    tensor: &[f32],
    row_span: usize,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(source_rows.len() * row_span);
    for row in source_rows {
        let Some(&idx) = row_index.get(&row.feature_id) else {
            bail!(
                "preprocessed bundle is missing tensor data for feature_id {}",
                row.feature_id
            );
        };
        let start = idx * row_span;
        let end = start + row_span;
        out.extend_from_slice(&tensor[start..end]);
    }
    Ok(out)
}

fn collect_preprocess_provenance(
    osw_path: &Path,
    xic_path: &Path,
    xic_paths: &Option<Vec<PathBuf>>,
    xic_map_path: &Option<PathBuf>,
    xim_path: &Option<PathBuf>,
    xim_paths: &Option<Vec<PathBuf>>,
    xim_map_path: &Option<PathBuf>,
) -> PreprocessedProvenance {
    let xic_paths = xic_paths.clone().unwrap_or_else(|| {
        if xic_path.as_os_str().is_empty() {
            Vec::new()
        } else {
            vec![xic_path.to_path_buf()]
        }
    });
    let xim_paths = xim_paths
        .clone()
        .or_else(|| xim_path.clone().map(|path| vec![path]))
        .unwrap_or_default();
    PreprocessedProvenance {
        osw_path: Some(osw_path.to_path_buf()),
        xic_paths,
        xic_map_path: xic_map_path.clone(),
        xim_paths,
        xim_map_path: xim_map_path.clone(),
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Log the distinct OSW, XIC, and XIM run identifiers seen in the current run.
///
/// The XIC/XIM summaries read parquet metadata directly from the configured
/// inputs so mismatches between OSW run ids and raw trace files are visible
/// immediately near startup.
fn log_run_id_summary(
    rows: &[FeatureRow],
    xic_path: &Path,
    xic_paths: &Option<Vec<PathBuf>>,
    xim_path: &Option<PathBuf>,
    xim_paths: &Option<Vec<PathBuf>>,
) {
    if !log::log_enabled!(log::Level::Info) {
        return;
    }
    let mut osw_runs: Vec<u64> = rows.iter().map(|r| r.run_id).collect();
    osw_runs.sort_unstable();
    osw_runs.dedup();
    log::info!("OSW run_ids (n={}): {:?}", osw_runs.len(), osw_runs);

    let run_ids = if let Some(paths) = xic_paths {
        let mut set = HashSet::new();
        for path in paths {
            match crate::io::xic_parquet::list_run_ids(path) {
                Ok(runs) => {
                    for run_id in runs {
                        set.insert(run_id);
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read XIC run_ids from {:?}: {e:#}", path);
                }
            }
        }
        let mut runs: Vec<u64> = set.into_iter().collect();
        runs.sort_unstable();
        runs
    } else {
        match crate::io::xic_parquet::list_run_ids(xic_path) {
            Ok(mut runs) => {
                runs.sort_unstable();
                runs.dedup();
                runs
            }
            Err(e) => {
                log::warn!("Failed to read XIC run_ids: {e:#}");
                Vec::new()
            }
        }
    };
    if !run_ids.is_empty() {
        log::info!("XIC run_ids (n={}): {:?}", run_ids.len(), run_ids);
    }

    let xim_run_ids = if let Some(paths) = xim_paths {
        let mut set = HashSet::new();
        for path in paths {
            match crate::io::xim_parquet::list_run_ids(path) {
                Ok(runs) => {
                    for run_id in runs {
                        set.insert(run_id);
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read XIM run_ids from {:?}: {e:#}", path);
                }
            }
        }
        let mut runs: Vec<u64> = set.into_iter().collect();
        runs.sort_unstable();
        runs
    } else if let Some(path) = xim_path {
        match crate::io::xim_parquet::list_run_ids(path) {
            Ok(mut runs) => {
                runs.sort_unstable();
                runs.dedup();
                runs
            }
            Err(e) => {
                log::warn!("Failed to read XIM run_ids: {e:#}");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    if !xim_run_ids.is_empty() {
        log::info!("XIM run_ids (n={}): {:?}", xim_run_ids.len(), xim_run_ids);
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn read_run_path_map(path: &Path, label: &str) -> Result<HashMap<u64, PathBuf>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {label} map: {path:?}"))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut map: HashMap<u64, PathBuf> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let first = parts.next().unwrap_or("");
        if first.eq_ignore_ascii_case("run_id") {
            continue;
        }
        let run_id: u64 = match first.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(path_str) = parts.next() else {
            continue;
        };
        let mut p = PathBuf::from(path_str);
        if p.is_relative() {
            let base_candidate = base.join(&p);
            let cwd_candidate = cwd.join(&p);
            let base_exists = base_candidate.exists();
            let cwd_exists = cwd_candidate.exists();
            p = match (base_exists, cwd_exists) {
                (true, true) => {
                    log::warn!(
                        "XIC map path {} exists relative to both {:?} and {:?}; using {:?}",
                        path_str,
                        base,
                        cwd,
                        base_candidate
                    );
                    base_candidate
                }
                (true, false) => base_candidate,
                (false, true) => cwd_candidate,
                (false, false) => {
                    log::warn!(
                        "XIC map path {} not found relative to {:?} or {:?}; using {:?}",
                        path_str,
                        base,
                        cwd,
                        base_candidate
                    );
                    base_candidate
                }
            };
        }
        map.insert(run_id, p);
    }
    if map.is_empty() {
        bail!("{label} map has no usable entries: {path:?}");
    }
    Ok(map)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn read_xic_map(path: &Path) -> Result<HashMap<u64, PathBuf>> {
    read_run_path_map(path, "XIC")
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn read_xim_map(path: &Path) -> Result<HashMap<u64, PathBuf>> {
    read_run_path_map(path, "XIM")
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn infer_run_path_map_from_paths<F>(
    paths: &[PathBuf],
    label: &str,
    list_run_ids: F,
) -> Result<HashMap<u64, PathBuf>>
where
    F: Fn(&Path) -> Result<Vec<u64>>,
{
    let mut map = HashMap::new();
    for path in paths {
        let run_ids = list_run_ids(path)?;
        if run_ids.is_empty() {
            log::warn!("{label} path {:?} reported no run_ids", path);
            continue;
        }
        for run_id in run_ids {
            if let Some(prev) = map.insert(run_id, path.clone()) {
                bail!(
                    "{label} paths map the same run_id {run_id} to both {:?} and {:?}",
                    prev,
                    path
                );
            }
        }
    }
    if map.is_empty() {
        bail!("no usable run_id -> path mapping could be inferred from {label} paths");
    }
    Ok(map)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn resolve_xic_map(
    xic_paths: &Option<Vec<PathBuf>>,
    xic_map_path: &Option<PathBuf>,
) -> Result<Option<HashMap<u64, PathBuf>>> {
    if let Some(map_path) = xic_map_path {
        return read_xic_map(map_path).map(Some);
    }
    if let Some(paths) = xic_paths {
        return infer_run_path_map_from_paths(paths, "XIC", crate::io::xic_parquet::list_run_ids)
            .map(Some);
    }
    Ok(None)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn resolve_xim_map(
    xim_paths: &Option<Vec<PathBuf>>,
    xim_map_path: &Option<PathBuf>,
) -> Result<Option<HashMap<u64, PathBuf>>> {
    if let Some(map_path) = xim_map_path {
        return read_xim_map(map_path).map(Some);
    }
    if let Some(paths) = xim_paths {
        return infer_run_path_map_from_paths(paths, "XIM", crate::io::xim_parquet::list_run_ids)
            .map(Some);
    }
    Ok(None)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn filter_rows_by_xic_map(
    rows: Vec<FeatureRow>,
    xic_paths: &Option<Vec<PathBuf>>,
    xic_map_path: &Option<PathBuf>,
) -> Result<Vec<FeatureRow>> {
    let Some(map) = resolve_xic_map(xic_paths, xic_map_path)? else {
        return Ok(rows);
    };
    let allowed: HashSet<u64> = map.keys().copied().collect();
    let before = rows.len();
    let out: Vec<FeatureRow> = rows
        .into_iter()
        .filter(|r| allowed.contains(&r.run_id))
        .collect();
    let dropped = before.saturating_sub(out.len());
    if dropped > 0 {
        log::info!(
            "Filtered OSW rows by XIC map run_ids: dropped {} of {}",
            dropped,
            before
        );
    }
    if out.is_empty() {
        log::warn!("No OSW rows remain after XIC map filtering");
    }
    Ok(out)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn build_traces_for_rows(
    rows: &[FeatureRow],
    xic_path: &Path,
    xic_paths: &Option<Vec<PathBuf>>,
    xic_map_path: &Option<PathBuf>,
    trace: &TraceBuildConfig,
    fetch: &XicFetchConfig,
    cache: Option<&SharedXicCache>,
    disk: Option<&XicDiskCache>,
) -> Result<Vec<f32>> {
    if let Some(map) = resolve_xic_map(xic_paths, xic_map_path)? {
        log::debug!(
            "Using XIC map with {} entries from {:?}",
            map.len(),
            xic_map_path
                .as_ref()
                .map(|p| p.as_path())
                .unwrap_or_else(|| Path::new("<inferred from xic_paths>"))
        );
        if let Some(cache) = cache {
            build_trace_tensors_from_parquet_map_cached(rows, &map, trace, fetch, cache, disk)
        } else {
            build_trace_tensors_from_parquet_map(rows, &map, trace, fetch)
        }
    } else {
        if let Some(cache) = cache {
            build_trace_tensors_from_parquet_cached(rows, xic_path, trace, fetch, cache, disk)
        } else {
            build_trace_tensors_from_parquet(rows, xic_path, trace, fetch)
        }
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn build_xim_for_rows(
    rows: &[FeatureRow],
    xim_path: &Option<PathBuf>,
    xim_paths: &Option<Vec<PathBuf>>,
    xim_map_path: &Option<PathBuf>,
    xim_trace: &Option<TraceBuildConfig>,
    xim_fetch: &XimFetchConfig,
    cache: Option<&SharedXimCache>,
    disk: Option<&XimDiskCache>,
) -> Result<Option<Vec<f32>>> {
    let Some(trace) = xim_trace.as_ref() else {
        return Ok(None);
    };
    if let Some(map) = resolve_xim_map(xim_paths, xim_map_path)? {
        log::debug!(
            "Using XIM map with {} entries from {:?}",
            map.len(),
            xim_map_path
                .as_ref()
                .map(|p| p.as_path())
                .unwrap_or_else(|| Path::new("<inferred from xim_paths>"))
        );
        return if let Some(cache) = cache {
            build_xim_tensors_from_parquet_map_cached(rows, &map, trace, xim_fetch, cache, disk)
                .map(Some)
        } else {
            build_xim_tensors_from_parquet_map(rows, &map, trace, xim_fetch).map(Some)
        };
    }
    let Some(path) = xim_path.as_ref() else {
        bail!("XIM branch enabled but neither xim_path/xim_paths nor xim_map_path was provided");
    };
    if let Some(cache) = cache {
        build_xim_tensors_from_parquet_cached(rows, path, trace, xim_fetch, cache, disk).map(Some)
    } else {
        build_xim_tensors_from_parquet(rows, path, trace, xim_fetch).map(Some)
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn build_xim_for_train_val(
    rows_tr: &[FeatureRow],
    rows_va: &[FeatureRow],
    xim_path: &Option<PathBuf>,
    xim_paths: &Option<Vec<PathBuf>>,
    xim_map_path: &Option<PathBuf>,
    xim_trace: &Option<TraceBuildConfig>,
    xim_fetch: &XimFetchConfig,
    cache: Option<&SharedXimCache>,
    disk: Option<&XimDiskCache>,
) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>)> {
    let Some(trace) = xim_trace.as_ref() else {
        return Ok((None, None));
    };
    if rows_va.is_empty() {
        let trace_opt = Some(trace.clone());
        let x_tr = build_xim_for_rows(
            rows_tr,
            xim_path,
            xim_paths,
            xim_map_path,
            &trace_opt,
            xim_fetch,
            cache,
            disk,
        )?;
        return Ok((x_tr, None));
    }

    let mut rows_all = Vec::with_capacity(rows_tr.len() + rows_va.len());
    rows_all.extend(rows_tr.iter().cloned());
    rows_all.extend(rows_va.iter().cloned());

    log::info!(
        "Building XIM mobilograms once for train+val (N={} + {} rows)",
        rows_tr.len(),
        rows_va.len()
    );
    let trace_opt = Some(trace.clone());
    let x_all = build_xim_for_rows(
        &rows_all,
        xim_path,
        xim_paths,
        xim_map_path,
        &trace_opt,
        xim_fetch,
        cache,
        disk,
    )?
    .unwrap_or_default();
    let span = trace.total_c() * trace.l;
    let split = rows_tr.len() * span;
    let x_tr = x_all[..split].to_vec();
    let x_va = x_all[split..].to_vec();
    Ok((Some(x_tr), Some(x_va)))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn build_traces_for_train_val(
    rows_tr: &[FeatureRow],
    rows_va: &[FeatureRow],
    xic_path: &Path,
    xic_paths: &Option<Vec<PathBuf>>,
    xic_map_path: &Option<PathBuf>,
    trace: &TraceBuildConfig,
    fetch: &XicFetchConfig,
    cache: Option<&SharedXicCache>,
    disk: Option<&XicDiskCache>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if rows_va.is_empty() {
        let x_tr = build_traces_for_rows(
            rows_tr,
            xic_path,
            xic_paths,
            xic_map_path,
            trace,
            fetch,
            cache,
            disk,
        )?;
        return Ok((x_tr, Vec::new()));
    }

    let mut rows_all = Vec::with_capacity(rows_tr.len() + rows_va.len());
    rows_all.extend(rows_tr.iter().cloned());
    rows_all.extend(rows_va.iter().cloned());

    log::info!(
        "Building XIC traces once for train+val (N={} + {} rows)",
        rows_tr.len(),
        rows_va.len()
    );
    let x_all = build_traces_for_rows(
        &rows_all,
        xic_path,
        xic_paths,
        xic_map_path,
        trace,
        fetch,
        cache,
        disk,
    )?;
    let span = trace.total_c() * trace.l;
    let split = rows_tr.len() * span;
    let x_tr = x_all[..split].to_vec();
    let x_va = x_all[split..].to_vec();
    Ok((x_tr, x_va))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Build the full inference XIC/XIM tensors once before chunked scoring.
///
/// When `restrict_osw_to_xic_map` is active without an explicit XIC map, TOPAZ
/// filters away rows whose extracted traces are entirely zero. This helper
/// keeps the row list and the returned tensor buffers aligned by applying that
/// filter immediately after prefetch.
fn prefetch_modalities_for_inference(
    rows: &mut Vec<FeatureRow>,
    cfg: &InferRunConfig,
    xim_trace_cfg: &Option<TraceBuildConfig>,
    cache_opt: Option<&SharedXicCache>,
    disk_cache: Option<&XicDiskCache>,
    xim_cache_opt: Option<&SharedXimCache>,
    xim_disk_cache: Option<&XimDiskCache>,
    apply_trace_filter: bool,
    xim_decode_issues: &mut Vec<crate::io::xim_parquet::XimDecodeIssue>,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let started = Instant::now();
    log::info!(
        "Prefetching XIC/XIM tensors once for inference (N={} rows)",
        rows.len()
    );
    let x_trace = build_traces_for_rows(
        rows,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
        cache_opt,
        disk_cache,
    )?;
    log::info!(
        "Inference prefetch stage | XIC complete for {} rows in {}",
        rows.len(),
        format_duration(started.elapsed())
    );
    let x_xim = build_xim_for_rows(
        rows,
        &cfg.xim_path,
        &cfg.xim_paths,
        &cfg.xim_map_path,
        xim_trace_cfg,
        &cfg.xim_fetch,
        xim_cache_opt,
        xim_disk_cache,
    )?;
    drain_xim_decode_issues(xim_decode_issues);
    log::info!(
        "Inference prefetch stage | XIM complete for {} rows in {}",
        rows.len(),
        format_duration(started.elapsed())
    );

    if !apply_trace_filter {
        return Ok((x_trace, x_xim));
    }

    let original_n = rows.len();
    let original_rows = std::mem::take(rows);
    let (rows_f, x_tr_f, x_xim_f) = filter_rows_by_trace_with_aux(
        original_rows,
        x_trace,
        x_xim,
        cfg.trace.total_c(),
        cfg.trace.l,
        xim_trace_cfg.as_ref().map(|c| c.total_c()),
        xim_trace_cfg.as_ref().map(|c| c.l),
    );
    *rows = rows_f;
    log::info!(
        "Inference prefetch stage | trace-based filter retained {}/{} rows",
        rows.len(),
        original_n
    );
    Ok((x_tr_f, x_xim_f))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Score one contiguous row-aligned chunk from already-built XIC/XIM tensors.
///
/// `x_trace` must contain `rows.len() * trace_cfg.total_c() * trace_cfg.l`
/// floats in row-major `(N, C, L)` order. When present, `x_xim` must contain
/// `rows.len() * xim_cfg.total_c() * xim_cfg.l` floats with the same row order.
fn score_inference_chunk(
    model: &TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    x_xim: Option<&[f32]>,
    table_feature_cols: &[String],
    target_cols: &[String],
    trace_cfg: &TraceBuildConfig,
    xim_trace_cfg: Option<&TraceBuildConfig>,
    use_heuristic_features: bool,
    feat_dim: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&crate::Preprocessor>,
) -> Result<Vec<f32>> {
    let n = rows.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let x_feat = if use_heuristic_features {
        rows_to_feature_matrix_with_cols(rows, table_feature_cols, target_cols, pre)
    } else {
        Vec::new()
    };
    let x_feat_t = candle_core::Tensor::from_vec(x_feat, (n, feat_dim), device)?;
    let x_trace_t =
        candle_core::Tensor::from_slice(x_trace, (n, trace_cfg.total_c(), trace_cfg.l), device)?;
    let x_xim_t = if let (Some(x_xim), Some(xim_cfg)) = (x_xim, xim_trace_cfg) {
        Some(candle_core::Tensor::from_slice(
            x_xim,
            (n, xim_cfg.total_c(), xim_cfg.l),
            device,
        )?)
    } else {
        None
    };
    let scores_t =
        score_candidates_with_aux(model, &x_feat_t, &x_trace_t, x_xim_t.as_ref(), batch_size)?;
    Ok(scores_t.to_vec1::<f32>()?)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Choose a smaller streaming work unit than the user-facing `trace_chunk_size`.
///
/// Large chunk sizes are useful for coarse memory budgeting, but waiting for a
/// full chunk of decoded XIC/XIM tensors before GPU scoring starts can leave
/// the device idle for a long time. Streaming therefore subdivides each chunk
/// into smaller work units that are large enough to keep batching efficient
/// while small enough to reduce time-to-first-score.
fn derive_stream_work_unit_size(cfg: &InferRunConfig) -> usize {
    let chunk_size = cfg.trace_chunk_size.max(1);
    let batch_floor = cfg.batch_size.max(1);
    let target = batch_floor.max(10_000).min(50_000);
    chunk_size.min(target).max(1)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Choose a small producer-pool size for streaming inference.
///
/// Producer workers spend most of their time in XIC/XIM parquet decode/build
/// functions, which already use Rayon internally. A small worker pool gives the
/// host side enough parallel slack to overlap multiple subchunks without
/// creating excessive oversubscription on CPU-limited jobs.
fn derive_stream_producer_workers() -> usize {
    let available = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    available.saturating_sub(1).clamp(1, 3)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Derive the bounded queue depth for subchunk streaming.
///
/// The queue depth scales with the producer pool so workers can stay busy, but
/// stays small enough that each queued payload does not blow up host RAM.
fn derive_stream_queue_depth(producer_workers: usize) -> usize {
    producer_workers.saturating_mul(2).clamp(2, 8)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Choose how many preprocessing workers should build chunks concurrently.
///
/// Each worker spends most of its time in XIC/XIM fetch and tensor-assembly
/// helpers, which already use Rayon internally. A moderate number of outer
/// workers is enough to keep the global Rayon pool fed while avoiding
/// excessive memory growth from too many in-flight chunks. XIM-heavy
/// preprocessing is capped more aggressively because each in-flight work unit
/// carries substantially larger tensors than XIC-only preprocessing.
fn derive_preprocess_producer_workers(
    total_chunks: usize,
    has_xim: bool,
    chunk_row_count: usize,
) -> usize {
    let available = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let max_workers = if has_xim {
        if chunk_row_count >= 50_000 {
            3
        } else if chunk_row_count >= 25_000 {
            4
        } else {
            6
        }
    } else {
        8
    };
    available.min(total_chunks.max(1)).clamp(1, max_workers)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Bound the number of preprocessing payloads waiting to be written.
///
/// Preprocessed chunks can be large because they carry full XIC/XIM tensors, so
/// the queue should stay shallow even on CPU-rich nodes. XIM-heavy runs keep an
/// even tighter queue to avoid holding multiple large mobilogram groups in
/// memory at once.
fn derive_preprocess_queue_depth(producer_workers: usize, has_xim: bool) -> usize {
    if has_xim {
        producer_workers.clamp(1, 2)
    } else {
        producer_workers.clamp(2, 6)
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Choose how many contiguous archive chunks a preprocessing worker should
/// merge into one fetch/build task.
///
/// Grouping adjacent chunks reduces the number of full parquet scans needed for
/// XIM-heavy datasets because one worker can materialize a larger union of
/// feature ids and then split the finished tensor back into regular archive
/// chunks. Very large XIM chunk sizes disable grouping to keep peak memory
/// bounded. Trace-filtered preprocessing also keeps a group size of `1` so row
/// filtering remains localized to the original chunk boundaries.
fn derive_preprocess_chunk_group_size(
    has_xim: bool,
    apply_trace_filter: bool,
    producer_workers: usize,
    chunk_row_count: usize,
) -> usize {
    if apply_trace_filter {
        return 1;
    }
    if has_xim {
        if chunk_row_count >= 50_000 {
            1
        } else if producer_workers >= 6 {
            2
        } else {
            3
        }
    } else {
        4
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Partition the inference row slice into contiguous streaming work units.
fn build_stream_work_ranges(total_rows: usize, work_unit_size: usize) -> Vec<(usize, usize)> {
    if total_rows == 0 {
        return Vec::new();
    }
    let work_unit_size = work_unit_size.max(1);
    (0..total_rows)
        .step_by(work_unit_size)
        .map(|start| (start, (start + work_unit_size).min(total_rows)))
        .collect()
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Stream inference chunks through a bounded host queue.
///
/// A small pool of producer workers decodes XIC/XIM parquet data for upcoming
/// row subchunks while the main thread performs GPU scoring for the current
/// subchunk. Compared with the legacy whole-chunk streaming implementation,
/// this lowers time-to-first-score and gives the decode side more opportunity
/// to use available CPU cores before the GPU becomes the bottleneck.
fn score_rows_streaming_inference(
    model: &TopazBagRanker,
    rows: &[FeatureRow],
    cfg: &InferRunConfig,
    xim_trace_cfg: &Option<TraceBuildConfig>,
    table_feature_cols: &[String],
    target_cols: &[String],
    use_heuristic_features: bool,
    feat_dim: usize,
    device: &Device,
    pre: Option<&crate::Preprocessor>,
    cache_opt: Option<&SharedXicCache>,
    disk_cache: Option<&XicDiskCache>,
    xim_cache_opt: Option<&SharedXimCache>,
    xim_disk_cache: Option<&XimDiskCache>,
    apply_trace_filter: bool,
) -> Result<StreamingInferenceOutput> {
    let chunk_size = cfg.trace_chunk_size.max(1);
    let work_unit_size = derive_stream_work_unit_size(cfg);
    let work_ranges = build_stream_work_ranges(rows.len(), work_unit_size);
    let producer_workers = derive_stream_producer_workers();
    let queue_depth = derive_stream_queue_depth(producer_workers);
    let (tx, rx) = mpsc::sync_channel::<Result<StreamingProducerMessage>>(queue_depth);
    let mut progress = InferenceProgressLogger::new("streaming", rows.len(), work_unit_size);
    let next_work_idx = AtomicUsize::new(0);

    log::info!(
        "Streaming inference configured | chunk_size={} work_unit_size={} producer_workers={} queue_depth={}",
        chunk_size,
        work_unit_size,
        producer_workers,
        queue_depth
    );

    let mut scores = if apply_trace_filter {
        Vec::new()
    } else {
        vec![0f32; rows.len()]
    };
    let mut rows_filtered = if apply_trace_filter {
        Some(Vec::new())
    } else {
        None
    };
    let mut summary = crate::infer::diagnostics::TraceSummary {
        n: 0,
        l: cfg.trace.l,
        ms1_cmax: cfg.trace.ms1_cmax,
        ms2_cmax: cfg.trace.ms2_cmax,
        ms1_nonzero_rows: 0,
        ms2_nonzero_rows: 0,
    };
    let mut xim_decode_issues = Vec::new();
    let mut decode_time = Duration::ZERO;
    let mut score_time = Duration::ZERO;
    let mut processed_units = 0usize;

    thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::new();
        for _worker_idx in 0..producer_workers {
            let tx_producer = tx.clone();
            let next_work_idx_ref = &next_work_idx;
            let work_ranges_ref = &work_ranges;
            let rows_ref = rows;
            let cfg_ref = cfg;
            let xim_trace_cfg_ref = xim_trace_cfg;
            let cache_opt_ref = cache_opt;
            let disk_cache_ref = disk_cache;
            let xim_cache_opt_ref = xim_cache_opt;
            let xim_disk_cache_ref = xim_disk_cache;
            let worker = scope.spawn(move || -> Result<()> {
                loop {
                    let seq_idx = next_work_idx_ref.fetch_add(1, Ordering::Relaxed);
                    let Some(&(start, end)) = work_ranges_ref.get(seq_idx) else {
                        break;
                    };
                    let chunk = &rows_ref[start..end];
                    let decode_started = Instant::now();
                    let x_trace = build_traces_for_rows(
                        chunk,
                        &cfg_ref.xic_path,
                        &cfg_ref.xic_paths,
                        &cfg_ref.xic_map_path,
                        &cfg_ref.trace,
                        &cfg_ref.fetch,
                        cache_opt_ref,
                        disk_cache_ref,
                    )?;
                    let x_xim = build_xim_for_rows(
                        chunk,
                        &cfg_ref.xim_path,
                        &cfg_ref.xim_paths,
                        &cfg_ref.xim_map_path,
                        xim_trace_cfg_ref,
                        &cfg_ref.xim_fetch,
                        xim_cache_opt_ref,
                        xim_disk_cache_ref,
                    )?;

                    let (rows_override, x_trace, x_xim) = if apply_trace_filter {
                        let (rows_f, x_tr_f, x_xim_f) = filter_rows_by_trace_with_aux(
                            chunk.to_vec(),
                            x_trace,
                            x_xim,
                            cfg_ref.trace.total_c(),
                            cfg_ref.trace.l,
                            xim_trace_cfg_ref.as_ref().map(|c| c.total_c()),
                            xim_trace_cfg_ref.as_ref().map(|c| c.l),
                        );
                        (Some(rows_f), x_tr_f, x_xim_f)
                    } else {
                        (None, x_trace, x_xim)
                    };

                    let payload = InferenceChunkPayload {
                        seq_idx,
                        start,
                        end,
                        rows_override,
                        x_trace,
                        x_xim,
                        xim_decode_issues: crate::io::xim_parquet::take_decode_issues(),
                        decode_time: decode_started.elapsed(),
                    };
                    tx_producer
                        .send(Ok(StreamingProducerMessage::Payload(payload)))
                        .context("streaming inference consumer dropped before decode finished")?;
                }
                tx_producer
                    .send(Ok(StreamingProducerMessage::Done))
                    .context("streaming inference consumer dropped before completion")?;
                Ok(())
            });
            workers.push(worker);
        }
        drop(tx);

        let mut completed_workers = 0usize;
        let mut next_expected = 0usize;
        let mut pending: BTreeMap<usize, InferenceChunkPayload> = BTreeMap::new();

        while completed_workers < producer_workers {
            let message = rx
                .recv()
                .context("streaming inference producer disconnected")?;
            match message? {
                StreamingProducerMessage::Done => {
                    completed_workers += 1;
                }
                StreamingProducerMessage::Payload(payload) => {
                    pending.insert(payload.seq_idx, payload);
                }
            }

            while let Some(payload) = pending.remove(&next_expected) {
                xim_decode_issues.extend(payload.xim_decode_issues);
                decode_time += payload.decode_time;
                processed_units += 1;
                let row_slice: &[FeatureRow] =
                    if let Some(rows_override) = payload.rows_override.as_ref() {
                        rows_override.as_slice()
                    } else {
                        &rows[payload.start..payload.end]
                    };
                if row_slice.is_empty() {
                    progress.maybe_log(payload.end.min(rows.len()), processed_units);
                    next_expected += 1;
                    continue;
                }

                if cfg.diagnostics.trace_summary {
                    let sum = trace_summary(
                        &payload.x_trace,
                        row_slice.len(),
                        cfg.trace.total_c(),
                        cfg.trace.l,
                        cfg.trace.ms1_cmax,
                        cfg.trace.ms2_cmax,
                    );
                    summary.n += sum.n;
                    summary.ms1_nonzero_rows += sum.ms1_nonzero_rows;
                    summary.ms2_nonzero_rows += sum.ms2_nonzero_rows;
                }

                let score_started = Instant::now();
                let scores_chunk = score_inference_chunk(
                    model,
                    row_slice,
                    &payload.x_trace,
                    payload.x_xim.as_deref(),
                    table_feature_cols,
                    target_cols,
                    &cfg.trace,
                    xim_trace_cfg.as_ref(),
                    use_heuristic_features,
                    feat_dim,
                    device,
                    cfg.batch_size.max(1),
                    pre,
                )?;
                score_time += score_started.elapsed();

                if let Some(mut filtered) = payload.rows_override {
                    if let Some(all_rows) = rows_filtered.as_mut() {
                        all_rows.append(&mut filtered);
                    }
                    scores.extend(scores_chunk);
                } else {
                    scores[payload.start..payload.start + row_slice.len()]
                        .copy_from_slice(&scores_chunk);
                }
                progress.maybe_log(payload.end.min(rows.len()), processed_units);
                next_expected += 1;
            }
        }

        while let Some(payload) = pending.remove(&next_expected) {
            xim_decode_issues.extend(payload.xim_decode_issues);
            decode_time += payload.decode_time;
            processed_units += 1;
            let row_slice: &[FeatureRow] =
                if let Some(rows_override) = payload.rows_override.as_ref() {
                    rows_override.as_slice()
                } else {
                    &rows[payload.start..payload.end]
                };
            if row_slice.is_empty() {
                progress.maybe_log(payload.end.min(rows.len()), processed_units);
                next_expected += 1;
                continue;
            }

            if cfg.diagnostics.trace_summary {
                let sum = trace_summary(
                    &payload.x_trace,
                    row_slice.len(),
                    cfg.trace.total_c(),
                    cfg.trace.l,
                    cfg.trace.ms1_cmax,
                    cfg.trace.ms2_cmax,
                );
                summary.n += sum.n;
                summary.ms1_nonzero_rows += sum.ms1_nonzero_rows;
                summary.ms2_nonzero_rows += sum.ms2_nonzero_rows;
            }

            let score_started = Instant::now();
            let scores_chunk = score_inference_chunk(
                model,
                row_slice,
                &payload.x_trace,
                payload.x_xim.as_deref(),
                table_feature_cols,
                target_cols,
                &cfg.trace,
                xim_trace_cfg.as_ref(),
                use_heuristic_features,
                feat_dim,
                device,
                cfg.batch_size.max(1),
                pre,
            )?;
            score_time += score_started.elapsed();

            if let Some(mut filtered) = payload.rows_override {
                if let Some(all_rows) = rows_filtered.as_mut() {
                    all_rows.append(&mut filtered);
                }
                scores.extend(scores_chunk);
            } else {
                scores[payload.start..payload.start + row_slice.len()]
                    .copy_from_slice(&scores_chunk);
            }
            progress.maybe_log(payload.end.min(rows.len()), processed_units);
            next_expected += 1;
        }

        if next_expected != work_ranges.len() {
            bail!(
                "streaming inference terminated early: processed {} of {} work units",
                next_expected,
                work_ranges.len()
            );
        }

        for worker in workers {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("streaming inference producer thread panicked"))??;
        }
        Ok(())
    })?;

    Ok(StreamingInferenceOutput {
        scores,
        rows_filtered,
        summary,
        xim_decode_issues,
        decode_time,
        score_time,
        work_unit_size,
        producer_workers,
        queue_depth,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn select_bag_winners(
    rows: &[FeatureRow],
    scores: &[f32],
) -> (Vec<FeatureRow>, Vec<String>, Vec<f32>, Vec<bool>, Vec<f32>) {
    let mut bag_idx: HashMap<String, usize> = HashMap::new();
    let mut bag_pid: Vec<String> = Vec::new();
    let mut bag_score: Vec<f32> = Vec::new();
    let mut bag_is_decoy: Vec<bool> = Vec::new();
    let mut bag_row_idx: Vec<usize> = Vec::new();

    for (i, row) in rows.iter().enumerate() {
        let pid = row.group_id.clone();
        if let Some(&bi) = bag_idx.get(&pid) {
            if scores[i] > bag_score[bi] {
                bag_score[bi] = scores[i];
                bag_row_idx[bi] = i;
            }
        } else {
            let bi = bag_pid.len();
            bag_idx.insert(pid.clone(), bi);
            bag_pid.push(pid);
            bag_score.push(scores[i]);
            bag_is_decoy.push(row.is_decoy);
            bag_row_idx.push(i);
        }
    }

    let mut winner_rows = Vec::with_capacity(bag_row_idx.len());
    for &idx in &bag_row_idx {
        winner_rows.push(rows[idx].clone());
    }
    let bag_y: Vec<f32> = bag_is_decoy
        .iter()
        .map(|&d| if d { 0.0 } else { 1.0 })
        .collect();

    (winner_rows, bag_pid, bag_score, bag_is_decoy, bag_y)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
struct BagTensorPack {
    xb: candle_core::Tensor,
    tb: candle_core::Tensor,
    tb_aux: Option<candle_core::Tensor>,
    mask: candle_core::Tensor,
    y_bag: Vec<f32>,
    bag_pid: Vec<String>,
    b: usize,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Host-side payload produced by the streaming inference loader thread.
///
/// Each payload contains one row chunk plus its decoded XIC/XIM tensors in the
/// same row order. When trace-based filtering is active, `rows_override`
/// contains the filtered chunk rows; otherwise the main thread reuses the
/// original `rows[start..end]` slice.
struct InferenceChunkPayload {
    seq_idx: usize,
    start: usize,
    end: usize,
    rows_override: Option<Vec<FeatureRow>>,
    x_trace: Vec<f32>,
    x_xim: Option<Vec<f32>>,
    xim_decode_issues: Vec<crate::io::xim_parquet::XimDecodeIssue>,
    decode_time: Duration,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Result of streaming inference before XRUN calibration and writeback.
struct StreamingInferenceOutput {
    scores: Vec<f32>,
    rows_filtered: Option<Vec<FeatureRow>>,
    summary: crate::infer::diagnostics::TraceSummary,
    xim_decode_issues: Vec<crate::io::xim_parquet::XimDecodeIssue>,
    decode_time: Duration,
    score_time: Duration,
    work_unit_size: usize,
    producer_workers: usize,
    queue_depth: usize,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Messages sent from producer workers to the streaming inference consumer.
///
/// Payloads can arrive out of order when multiple workers are active, so the
/// consumer buffers them by `seq_idx` and only scores them once all earlier
/// subchunks have been processed.
enum StreamingProducerMessage {
    Payload(InferenceChunkPayload),
    Done,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// One completed preprocessing chunk waiting to be written into the bundle.
///
/// Preprocessing is CPU-heavy but archive output must remain ordered and use a
/// single zip writer. Producer workers therefore build row-aligned tensors in
/// parallel and hand them back to the main thread as payloads like this.
struct PreprocessChunkPayload {
    seq_idx: usize,
    source_rows: usize,
    rows: Vec<FeatureRow>,
    x_trace: Vec<f32>,
    x_xim: Option<Vec<f32>>,
    xim_decode_issues: Vec<crate::io::xim_parquet::XimDecodeIssue>,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// One completed preprocessing work group.
///
/// Each producer may combine multiple adjacent archive chunks into one larger
/// XIC/XIM fetch so parquet files are scanned fewer times on XIM-heavy
/// datasets. The ordered writer still commits the constituent chunks one by
/// one, preserving the on-disk archive layout.
struct PreprocessChunkGroupPayload {
    payloads: Vec<PreprocessChunkPayload>,
    stage_stats: PreprocessStageRuntimeStats,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Cumulative wall-clock counters for XIC/XIM preprocessing.
///
/// The preprocessing pipeline spends most of its time in two host-side stages:
/// building fixed-width chromatogram tensors and building fixed-width
/// mobilogram tensors. Tracking them separately makes it obvious whether XIM
/// preprocessing dominates runtime on a given dataset.
#[derive(Debug, Clone, Copy, Default)]
struct PreprocessStageRuntimeStats {
    xic_rows: usize,
    xim_rows: usize,
    xic_build_time: Duration,
    xim_build_time: Duration,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
impl std::ops::AddAssign for PreprocessStageRuntimeStats {
    fn add_assign(&mut self, rhs: Self) {
        self.xic_rows += rhs.xic_rows;
        self.xim_rows += rhs.xim_rows;
        self.xic_build_time += rhs.xic_build_time;
        self.xim_build_time += rhs.xim_build_time;
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Messages exchanged between preprocessing workers and the ordered writer.
enum PreprocessProducerMessage {
    PayloadGroup(PreprocessChunkGroupPayload),
    Done,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Lightweight runtime counters for the main row-scoring phase of inference.
///
/// The timers distinguish host-side decode/build work from model scoring so it
/// is easier to see whether a run is I/O bound or GPU bound.
#[derive(Debug, Clone)]
struct InferenceRuntimeStats {
    mode: &'static str,
    chunk_size: usize,
    work_unit_size: Option<usize>,
    producer_workers: Option<usize>,
    queue_depth: Option<usize>,
    decode_time: Duration,
    score_time: Duration,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Emit a compact log summary describing how inference spent its time.
fn log_inference_runtime_stats(stats: &InferenceRuntimeStats, n_rows: usize) {
    let decode_s = stats.decode_time.as_secs_f64();
    let score_s = stats.score_time.as_secs_f64();
    let total_s = decode_s + score_s;
    let decode_pct = if total_s > 0.0 {
        100.0 * decode_s / total_s
    } else {
        0.0
    };
    let score_pct = if total_s > 0.0 {
        100.0 * score_s / total_s
    } else {
        0.0
    };

    let mut extra = Vec::new();
    if let Some(work_unit_size) = stats.work_unit_size {
        extra.push(format!("work_unit_size={work_unit_size}"));
    }
    if let Some(producer_workers) = stats.producer_workers {
        extra.push(format!("producer_workers={producer_workers}"));
    }
    if let Some(queue_depth) = stats.queue_depth {
        extra.push(format!("queue_depth={queue_depth}"));
    }
    let extra = if extra.is_empty() {
        String::new()
    } else {
        format!(" {}", extra.join(" "))
    };

    log::info!(
        "Inference runtime ({}) | rows={} chunk_size={}{} decode={:.1}s ({:.1}%) score={:.1}s ({:.1}%)",
        stats.mode,
        n_rows,
        stats.chunk_size,
        extra,
        decode_s,
        decode_pct,
        score_s,
        score_pct
    );
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Emit cumulative XIC/XIM preprocessing throughput counters.
///
/// The reported rows-per-second values are computed from the number of source
/// rows whose XIC/XIM tensors have been materialized divided by the measured
/// wall-clock time spent inside each builder. This intentionally excludes zip
/// writing and bookkeeping so the log isolates the expensive preprocessing
/// stages themselves.
fn log_preprocess_stage_throughput(stats: &PreprocessStageRuntimeStats) {
    let xic_secs = stats.xic_build_time.as_secs_f64();
    let xic_rows_per_sec = if xic_secs > 0.0 {
        stats.xic_rows as f64 / xic_secs
    } else {
        0.0
    };

    if stats.xim_rows == 0 {
        log::info!(
            "Preprocess stage throughput | xic={:.0} rows/s (rows={} time={:.1}s) xim=disabled",
            xic_rows_per_sec,
            stats.xic_rows,
            xic_secs
        );
        return;
    }

    let xim_secs = stats.xim_build_time.as_secs_f64();
    let xim_rows_per_sec = if xim_secs > 0.0 {
        stats.xim_rows as f64 / xim_secs
    } else {
        0.0
    };

    log::info!(
        "Preprocess stage throughput | xic={:.0} rows/s (rows={} time={:.1}s) xim={:.0} rows/s (rows={} time={:.1}s)",
        xic_rows_per_sec,
        stats.xic_rows,
        xic_secs,
        xim_rows_per_sec,
        stats.xim_rows,
        xim_secs
    );
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Periodically emits chunk-level inference progress with a coarse ETA.
///
/// Progress is reported in terms of input rows consumed rather than retained
/// rows because trace-based filtering can drop zero-trace rows after loading.
/// This keeps the percentage monotonic and lets long-running jobs show clear
/// forward motion in the logs.
struct InferenceProgressLogger {
    mode: &'static str,
    total_rows: usize,
    total_chunks: usize,
    started: Instant,
    last_log: Instant,
    log_every: Duration,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
impl InferenceProgressLogger {
    /// Create a progress logger for one inference pass.
    fn new(mode: &'static str, total_rows: usize, chunk_size: usize) -> Self {
        let now = Instant::now();
        let total_chunks = if total_rows == 0 {
            0
        } else {
            total_rows.div_ceil(chunk_size.max(1))
        };
        Self {
            mode,
            total_rows,
            total_chunks,
            started: now,
            last_log: now,
            log_every: Duration::from_secs(30),
        }
    }

    /// Log progress after a completed chunk when enough time has elapsed.
    fn maybe_log(&mut self, processed_rows: usize, processed_chunks: usize) -> bool {
        let now = Instant::now();
        let should_log = processed_rows >= self.total_rows
            || processed_chunks >= self.total_chunks
            || now.duration_since(self.last_log) >= self.log_every
            || processed_chunks <= 1;
        if !should_log {
            return false;
        }
        self.last_log = now;

        let elapsed = now.duration_since(self.started);
        let pct = if self.total_rows > 0 {
            100.0 * processed_rows as f64 / self.total_rows as f64
        } else {
            100.0
        };
        let rows_per_sec = if elapsed.as_secs_f64() > 0.0 {
            processed_rows as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let eta = if processed_rows > 0 && processed_rows < self.total_rows {
            let remaining_rows = (self.total_rows - processed_rows) as f64;
            Duration::from_secs_f64(remaining_rows / rows_per_sec.max(1e-9))
        } else {
            Duration::ZERO
        };

        log::info!(
            "Inference progress ({}) | chunks={}/{} rows={}/{} ({:.1}%) elapsed={} eta={} rate={:.0} rows/s",
            self.mode,
            processed_chunks,
            self.total_chunks,
            processed_rows,
            self.total_rows,
            pct,
            format_duration(elapsed),
            format_duration(eta),
            rows_per_sec
        );
        true
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Format a wall-clock duration for human-readable progress logs.
fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let rem_secs = secs % 60;
    if hours > 0 {
        format!("{hours:02}:{mins:02}:{rem_secs:02}")
    } else {
        format!("{mins:02}:{rem_secs:02}")
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn build_bag_tensor_pack_with_cols(
    rows: &[FeatureRow],
    x_trace: &[f32],
    x_aux: Option<(&[f32], &TraceBuildConfig)>,
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    pre: Option<&crate::Preprocessor>,
) -> Result<BagTensorPack> {
    let n = rows.len();
    let d = target_cols.len();
    let x_feat = rows_to_feature_matrix_with_cols(rows, osw_cols, target_cols, pre);
    let y_rows: Vec<u8> = rows
        .iter()
        .map(|r| if r.is_decoy { 1 } else { 0 })
        .collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat, n, d, x_trace, c_total, l, &y_rows, &pid_rows, bag_k,
    );
    let xb = candle_core::Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = candle_core::Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = candle_core::Tensor::from_vec(mask_u8.clone(), (bags.b, bags.k), device)?;

    let tb_aux = if let Some((x_aux, aux_cfg)) = x_aux {
        let aux_bags = make_bags_with_traces(
            &x_feat,
            n,
            d,
            x_aux,
            aux_cfg.total_c(),
            aux_cfg.l,
            &y_rows,
            &pid_rows,
            bag_k,
        );
        if aux_bags.b != bags.b || aux_bags.k != bags.k || aux_bags.bag_pid != bags.bag_pid {
            bail!("auxiliary bagging order mismatch between XIC and XIM inputs");
        }
        Some(candle_core::Tensor::from_vec(
            aux_bags.t_bag,
            (aux_bags.b, aux_bags.k, aux_bags.c, aux_bags.l),
            device,
        )?)
    } else {
        None
    };

    Ok(BagTensorPack {
        xb,
        tb,
        tb_aux,
        mask,
        y_bag: bags.y_bag,
        bag_pid: bags.bag_pid,
        b: bags.b,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn score_bags_from_rows_with_cols_with_aux(
    model: &TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    x_aux: Option<(&[f32], &TraceBuildConfig)>,
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&crate::Preprocessor>,
) -> Result<crate::infer::BagScoreOutput> {
    if rows.is_empty() {
        return Ok(crate::infer::BagScoreOutput {
            bag_score: Vec::new(),
            bag_y: Vec::new(),
            is_decoy: Vec::new(),
            bag_pid: Vec::new(),
            winner_hidden: Vec::new(),
            hidden_dim: 0,
        });
    }

    let pack = build_bag_tensor_pack_with_cols(
        rows,
        x_trace,
        x_aux,
        osw_cols,
        target_cols,
        c_total,
        l,
        bag_k,
        device,
        pre,
    )?;

    let b = pack.b;
    let mut bag_scores = Vec::with_capacity(b);
    let mut hidden: Vec<f32> = Vec::new();
    let mut hidden_dim = 0usize;
    let bs = batch_size.max(1);

    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = pack.xb.narrow(0, i, take)?;
        let tb_i = pack.tb.narrow(0, i, take)?;
        let m_i = pack.mask.narrow(0, i, take)?;
        let tb_aux_i = if let Some(tb_aux) = pack.tb_aux.as_ref() {
            Some(tb_aux.narrow(0, i, take)?)
        } else {
            None
        };

        let (_cand, bag, win) =
            model.forward_bags_with_hidden_aux(&xb_i, &tb_i, &m_i, tb_aux_i.as_ref())?;
        bag_scores.extend(bag.to_vec1::<f32>()?);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.first().map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }
        i += take;
    }

    let is_decoy: Vec<bool> = pack.y_bag.iter().map(|&y| y < 0.5).collect();
    Ok(crate::infer::BagScoreOutput {
        bag_score: bag_scores,
        bag_y: pack.y_bag,
        is_decoy,
        bag_pid: pack.bag_pid,
        winner_hidden: hidden,
        hidden_dim,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn score_bags_with_heads_from_rows_with_cols_with_aux(
    model: &TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    x_aux: Option<(&[f32], &TraceBuildConfig)>,
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&crate::Preprocessor>,
) -> Result<crate::infer::BagHeadOutput> {
    if rows.is_empty() {
        return Ok(crate::infer::BagHeadOutput {
            bag_score: Vec::new(),
            bag_y: Vec::new(),
            is_decoy: Vec::new(),
            bag_pid: Vec::new(),
            winner_hidden: Vec::new(),
            hidden_dim: 0,
            emb_ms2: Vec::new(),
            emb_ms2_dim: 0,
            emb_ms1: Vec::new(),
            emb_ms1_dim: 0,
            emb_all: Vec::new(),
            emb_all_dim: 0,
            coe_ms2: Vec::new(),
            coe_ms2_dim: 0,
            coe_ms1: Vec::new(),
            coe_ms1_dim: 0,
            coe_ms12: Vec::new(),
            coe_ms12_dim: 0,
            coe_all: Vec::new(),
            coe_all_dim: 0,
        });
    }

    let pack = build_bag_tensor_pack_with_cols(
        rows,
        x_trace,
        x_aux,
        osw_cols,
        target_cols,
        c_total,
        l,
        bag_k,
        device,
        pre,
    )?;

    let b = pack.b;
    let mut bag_scores = Vec::with_capacity(b);
    let mut hidden: Vec<f32> = Vec::new();
    let mut hidden_dim = 0usize;
    let mut emb_ms2 = Vec::new();
    let mut emb_ms1 = Vec::new();
    let mut emb_all = Vec::new();
    let mut coe_ms2 = Vec::new();
    let mut coe_ms1 = Vec::new();
    let mut coe_ms12 = Vec::new();
    let mut coe_all = Vec::new();
    let mut emb_ms2_dim = 0usize;
    let mut emb_ms1_dim = 0usize;
    let mut emb_all_dim = 0usize;
    let mut coe_ms2_dim = 0usize;
    let mut coe_ms1_dim = 0usize;
    let mut coe_ms12_dim = 0usize;
    let mut coe_all_dim = 0usize;
    let bs = batch_size.max(1);

    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = pack.xb.narrow(0, i, take)?;
        let tb_i = pack.tb.narrow(0, i, take)?;
        let m_i = pack.mask.narrow(0, i, take)?;
        let tb_aux_i = if let Some(tb_aux) = pack.tb_aux.as_ref() {
            Some(tb_aux.narrow(0, i, take)?)
        } else {
            None
        };

        let (_cand, bag, win, comps) =
            model.forward_bags_with_heads_aux(&xb_i, &tb_i, &m_i, tb_aux_i.as_ref())?;
        bag_scores.extend(bag.to_vec1::<f32>()?);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.first().map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }

        let emb2 = comps.emb_ms2.to_vec2::<f32>()?;
        let emb1 = comps.emb_ms1.to_vec2::<f32>()?;
        let emba = comps.emb_all.to_vec2::<f32>()?;
        let coe2 = comps.coe_ms2.to_vec2::<f32>()?;
        let coe1 = comps.coe_ms1.to_vec2::<f32>()?;
        let coe12b = comps.coe_ms12.to_vec2::<f32>()?;
        let coeab = comps.coe_all.to_vec2::<f32>()?;

        if emb_ms2_dim == 0 {
            emb_ms2_dim = emb2.first().map(|v| v.len()).unwrap_or(0);
        }
        if emb_ms1_dim == 0 {
            emb_ms1_dim = emb1.first().map(|v| v.len()).unwrap_or(0);
        }
        if emb_all_dim == 0 {
            emb_all_dim = emba.first().map(|v| v.len()).unwrap_or(0);
        }
        if coe_ms2_dim == 0 {
            coe_ms2_dim = coe2.first().map(|v| v.len()).unwrap_or(0);
        }
        if coe_ms1_dim == 0 {
            coe_ms1_dim = coe1.first().map(|v| v.len()).unwrap_or(0);
        }
        if coe_ms12_dim == 0 {
            coe_ms12_dim = coe12b.first().map(|v| v.len()).unwrap_or(0);
        }
        if coe_all_dim == 0 {
            coe_all_dim = coeab.first().map(|v| v.len()).unwrap_or(0);
        }

        for row in emb2 {
            emb_ms2.extend(row);
        }
        for row in emb1 {
            emb_ms1.extend(row);
        }
        for row in emba {
            emb_all.extend(row);
        }
        for row in coe2 {
            coe_ms2.extend(row);
        }
        for row in coe1 {
            coe_ms1.extend(row);
        }
        for row in coe12b {
            coe_ms12.extend(row);
        }
        for row in coeab {
            coe_all.extend(row);
        }
        i += take;
    }

    let is_decoy: Vec<bool> = pack.y_bag.iter().map(|&y| y < 0.5).collect();
    Ok(crate::infer::BagHeadOutput {
        bag_score: bag_scores,
        bag_y: pack.y_bag,
        is_decoy,
        bag_pid: pack.bag_pid,
        winner_hidden: hidden,
        hidden_dim,
        emb_ms2,
        emb_ms2_dim,
        emb_ms1,
        emb_ms1_dim,
        emb_all,
        emb_all_dim,
        coe_ms2,
        coe_ms2_dim,
        coe_ms1,
        coe_ms1_dim,
        coe_ms12,
        coe_ms12_dim,
        coe_all,
        coe_all_dim,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn build_xrun_bag_data_from_rows_with_cols_with_aux(
    model: &TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    x_aux: Option<(&[f32], &TraceBuildConfig)>,
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&crate::Preprocessor>,
) -> Result<crate::xrun::XrunBagData> {
    let out = score_bags_from_rows_with_cols_with_aux(
        model,
        rows,
        x_trace,
        x_aux,
        osw_cols,
        target_cols,
        c_total,
        l,
        bag_k,
        device,
        batch_size,
        pre,
    )?;
    Ok(crate::xrun::XrunBagData {
        bag_score: out.bag_score,
        bag_hidden: out.winner_hidden,
        hidden_dim: out.hidden_dim,
        bag_y: out.bag_y,
        bag_pid: out.bag_pid,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn drain_xim_decode_issues(accum: &mut Vec<crate::io::xim_parquet::XimDecodeIssue>) {
    accum.extend(crate::io::xim_parquet::take_decode_issues());
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn diagnostic_tsv_path(path: &Path, label: &str) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .or_else(|| path.file_name())
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("topaz");
    parent.join(format!("{stem}.{label}.tsv"))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn write_xim_decode_issue_tsv(
    path: &Path,
    issues: &[crate::io::xim_parquet::XimDecodeIssue],
) -> Result<()> {
    let mut uniq = issues.to_vec();
    uniq.sort();
    uniq.dedup();

    let mut text = String::new();
    text.push_str("XIM_PATH\tRUN_ID\tFEATURE_ID\tANNOTATION\tFIELD\tCOMPRESSION\tERROR\n");
    for issue in uniq {
        let xim_path = issue.xim_path.display().to_string().replace('\t', " ");
        let annotation = issue.annotation.replace('\t', " ");
        let field = issue.field.replace('\t', " ");
        let error = issue.error.replace(['\t', '\n', '\r'], " ");
        text.push_str(&format!(
            "{xim_path}\t{}\t{}\t{annotation}\t{field}\t{}\t{error}\n",
            issue.run_id, issue.feature_id, issue.compression
        ));
    }
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn report_xim_decode_issues(
    stage: &str,
    issues: &[crate::io::xim_parquet::XimDecodeIssue],
    out_path: &Path,
) -> Result<()> {
    if issues.is_empty() {
        return Ok(());
    }

    let mut uniq = issues.to_vec();
    uniq.sort();
    uniq.dedup();

    let mut files = HashSet::new();
    let mut features = HashSet::new();
    for issue in &uniq {
        files.insert(issue.xim_path.clone());
        features.insert(issue.feature_id);
    }

    write_xim_decode_issue_tsv(out_path, &uniq)?;
    log::warn!(
        "Skipped {} malformed XIM traces during {} loading ({} unique traces across {} features from {} files). Wrote diagnostics to {:?}",
        issues.len(),
        stage,
        uniq.len(),
        features.len(),
        files.len(),
        out_path
    );
    Ok(())
}

fn write_head_embeddings_tsv(path: &Path, out: &crate::infer::BagHeadOutput) -> Result<()> {
    let mut header: Vec<String> = Vec::new();
    header.push("bag_pid".to_string());
    header.push("is_decoy".to_string());
    header.push("bag_score".to_string());
    for i in 0..out.hidden_dim {
        header.push(format!("win_hidden_{i}"));
    }
    for i in 0..out.emb_ms2_dim {
        header.push(format!("emb_ms2_{i}"));
    }
    for i in 0..out.emb_ms1_dim {
        header.push(format!("emb_ms1_{i}"));
    }
    for i in 0..out.emb_all_dim {
        header.push(format!("emb_all_{i}"));
    }
    for i in 0..out.coe_ms2_dim {
        header.push(format!("coe_ms2_{i}"));
    }
    for i in 0..out.coe_ms1_dim {
        header.push(format!("coe_ms1_{i}"));
    }
    for i in 0..out.coe_ms12_dim {
        header.push(format!("coe_ms12_{i}"));
    }
    for i in 0..out.coe_all_dim {
        header.push(format!("coe_all_{i}"));
    }

    let mut text = String::new();
    text.push_str(&header.join("\t"));
    text.push('\n');

    let b = out.bag_pid.len();
    for i in 0..b {
        let mut row = Vec::with_capacity(header.len());
        row.push(out.bag_pid[i].clone());
        row.push(if out.is_decoy.get(i).copied().unwrap_or(false) {
            "1".to_string()
        } else {
            "0".to_string()
        });
        row.push(format!("{}", out.bag_score.get(i).copied().unwrap_or(0.0)));

        let off = i * out.hidden_dim;
        for j in 0..out.hidden_dim {
            row.push(format!(
                "{}",
                out.winner_hidden.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        let off = i * out.emb_ms2_dim;
        for j in 0..out.emb_ms2_dim {
            row.push(format!(
                "{}",
                out.emb_ms2.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        let off = i * out.emb_ms1_dim;
        for j in 0..out.emb_ms1_dim {
            row.push(format!(
                "{}",
                out.emb_ms1.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        let off = i * out.emb_all_dim;
        for j in 0..out.emb_all_dim {
            row.push(format!(
                "{}",
                out.emb_all.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        let off = i * out.coe_ms2_dim;
        for j in 0..out.coe_ms2_dim {
            row.push(format!(
                "{}",
                out.coe_ms2.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        let off = i * out.coe_ms1_dim;
        for j in 0..out.coe_ms1_dim {
            row.push(format!(
                "{}",
                out.coe_ms1.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        let off = i * out.coe_ms12_dim;
        for j in 0..out.coe_ms12_dim {
            row.push(format!(
                "{}",
                out.coe_ms12.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        let off = i * out.coe_all_dim;
        for j in 0..out.coe_all_dim {
            row.push(format!(
                "{}",
                out.coe_all.get(off + j).copied().unwrap_or(0.0)
            ));
        }
        text.push_str(&row.join("\t"));
        text.push('\n');
    }

    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn log_xic_cache_stats(label: &str, stats: &crate::infer::XicCacheStats) {
    let (mem_hits, disk_hits, misses, stores, evictions) = stats.snapshot();
    let total = mem_hits + disk_hits + misses;
    let hit_rate = if total > 0 {
        (mem_hits + disk_hits) as f64 / (total as f64)
    } else {
        0.0
    };
    log::info!(
        "XIC cache stats ({label}): mem_hits={} disk_hits={} misses={} stores={} evictions={} hit_rate={:.3}",
        mem_hits,
        disk_hits,
        misses,
        stores,
        evictions,
        hit_rate
    );
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn log_xim_cache_stats(label: &str, stats: &crate::infer::XimCacheStats) {
    let (mem_hits, disk_hits, misses, stores, evictions) = stats.snapshot();
    let total = mem_hits + disk_hits + misses;
    let hit_rate = if total > 0 {
        (mem_hits + disk_hits) as f64 / (total as f64)
    } else {
        0.0
    };
    log::info!(
        "XIM cache stats ({label}): mem_hits={} disk_hits={} misses={} stores={} evictions={} hit_rate={:.3}",
        mem_hits,
        disk_hits,
        misses,
        stores,
        evictions,
        hit_rate
    );
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn summarize_names(names: &[String], max_show: usize) -> String {
    if names.is_empty() {
        return "none".to_string();
    }
    let shown = names
        .iter()
        .take(max_show)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > max_show {
        format!("{shown}, ... (+{} more)", names.len() - max_show)
    } else {
        shown
    }
}

fn checkpoint_base(path: &Path) -> PathBuf {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        if ext == "safetensors" || ext == "json" || ext == "model" {
            return path.with_extension("");
        }
    }
    path.to_path_buf()
}

fn read_checkpoint_meta(base: &Path) -> Result<CheckpointMeta> {
    crate::checkpoint::read_checkpoint_meta(base)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn load_checkpoint_weights(base: &Path, varmap: &mut VarMap) -> Result<()> {
    crate::checkpoint::load_checkpoint(base, varmap).map(|_| ())
}

#[cfg(feature = "io-sqlite")]
fn score_rows_to_osw(rows: &[crate::infer::ScoreTableRow]) -> Vec<OswScoreRow> {
    rows.iter()
        .map(|r| OswScoreRow {
            feature_id: r.feature_id,
            score: r.score,
            rank: r.rank,
            pvalue: r.pvalue,
            qvalue: r.qvalue,
            pep: r.pep,
        })
        .collect()
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn load_xrun_calibrator(
    base: &Path,
    device: &Device,
) -> Result<Option<(VarMap, XrunAttentionCalibrator, XrunCheckpointMeta)>> {
    if !xrun_checkpoint_exists(base) {
        return Ok(None);
    }
    let meta = read_xrun_checkpoint_meta(base)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let calibrator = XrunAttentionCalibrator::new(
        vb.pp("xrun"),
        XrunConfig {
            in_dim: meta.in_dim,
            d_model: meta.train.d_model,
            attn_hidden: meta.train.attn_hidden,
            head_hidden: meta.train.head_hidden.clone(),
            dropout: meta.train.dropout,
            center_delta: meta.train.center_delta,
            delta_clip: meta.train.delta_clip,
        },
    )?;
    load_xrun_checkpoint(base, &mut varmap)?;
    Ok(Some((varmap, calibrator, meta)))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
#[derive(Debug, Clone)]
struct XrunAppliedScores {
    row_scores: Vec<f32>,
    bag_pid: Vec<String>,
    bag_score: Vec<f32>,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn apply_xrun_to_row_scores(
    rows: &[FeatureRow],
    row_scores: &[f32],
    model: &TopazBagRanker,
    table_feature_cols: &[String],
    target_cols: &[String],
    trace_cfg: &TraceBuildConfig,
    xim_trace_cfg: Option<&TraceBuildConfig>,
    fetch_cfg: &XicFetchConfig,
    xim_fetch_cfg: &XimFetchConfig,
    xic_path: &Path,
    xic_paths: &Option<Vec<PathBuf>>,
    xic_map_path: &Option<PathBuf>,
    xim_path: &Option<PathBuf>,
    xim_paths: &Option<Vec<PathBuf>>,
    xim_map_path: &Option<PathBuf>,
    cache_opt: Option<&SharedXicCache>,
    disk_cache: Option<&XicDiskCache>,
    xim_cache_opt: Option<&SharedXimCache>,
    xim_disk_cache: Option<&XimDiskCache>,
    xrun_cfg: &XrunRunConfig,
    xrun_model: &XrunAttentionCalibrator,
    xrun_meta: &XrunCheckpointMeta,
    device: &Device,
    base_batch_size: usize,
    pre: Option<&crate::Preprocessor>,
) -> Result<XrunAppliedScores> {
    let (winner_rows, bag_pid, bag_score, _bag_is_decoy, bag_y) =
        select_bag_winners(rows, row_scores);
    if winner_rows.is_empty() {
        return Ok(XrunAppliedScores {
            row_scores: row_scores.to_vec(),
            bag_pid,
            bag_score,
        });
    }

    let x_trace = build_traces_for_rows(
        &winner_rows,
        xic_path,
        xic_paths,
        xic_map_path,
        trace_cfg,
        fetch_cfg,
        cache_opt,
        disk_cache,
    )?;
    let xim_trace_owned = xim_trace_cfg.cloned();
    let x_xim = build_xim_for_rows(
        &winner_rows,
        xim_path,
        xim_paths,
        xim_map_path,
        &xim_trace_owned,
        xim_fetch_cfg,
        xim_cache_opt,
        xim_disk_cache,
    )?;
    let head_out = score_bags_with_heads_from_rows_with_cols_with_aux(
        model,
        &winner_rows,
        &x_trace,
        x_xim
            .as_ref()
            .zip(xim_trace_cfg)
            .map(|(x, cfg)| (x.as_slice(), cfg)),
        table_feature_cols,
        target_cols,
        trace_cfg.total_c(),
        trace_cfg.l,
        1,
        device,
        base_batch_size,
        pre,
    )?;

    let predict_cfg = XrunPredictConfig {
        max_runs: if xrun_cfg.max_runs > 0 {
            xrun_cfg.max_runs
        } else {
            xrun_meta.predict.max_runs
        },
        sort_by: if !xrun_cfg.sort_by.is_empty() {
            xrun_cfg.sort_by.clone()
        } else {
            xrun_meta.predict.sort_by.clone()
        },
        batch_size: if xrun_cfg.batch_size > 0 {
            xrun_cfg.batch_size
        } else {
            xrun_meta.predict.batch_size
        },
    };
    let bag_data = crate::xrun::XrunBagData {
        bag_score: bag_score.clone(),
        bag_hidden: head_out.winner_hidden.clone(),
        hidden_dim: head_out.hidden_dim,
        bag_y: bag_y.clone(),
        bag_pid: bag_pid.clone(),
    };
    let (delta_bag, _attn_entropy) =
        crate::xrun::xrun_predict_deltas_for_bags(xrun_model, &bag_data, &predict_cfg, device)?;
    let row_scores = apply_xrun_deltas_to_rows(row_scores, rows, &bag_pid, &delta_bag);
    let bag_score = apply_xrun_deltas(&bag_score, &delta_bag);

    Ok(XrunAppliedScores {
        row_scores,
        bag_pid,
        bag_score,
    })
}

fn get_device(device_str: &str) -> Result<Device> {
    if device_str.starts_with("cuda") {
        let cuda_index = if device_str == "cuda" {
            0
        } else {
            device_str
                .split(':')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        };
        let device = Device::cuda_if_available(cuda_index)?;
        if !device.is_cuda() {
            bail!("CUDA device {} is not available", cuda_index);
        }
        Ok(device)
    } else {
        match device_str {
            "cpu" => Ok(Device::Cpu),
            _ => bail!("unsupported device type: {}", device_str),
        }
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Build one preprocessing payload for the ordered bundle writer.
///
/// The returned tensors stay aligned to `rows[start..end]` unless
/// `apply_trace_filter` compacts all-zero trace rows away, in which case the
/// payload contains only the retained rows and matching tensor buffers.
fn preprocess_chunk_payload(
    rows: &[FeatureRow],
    seq_idx: usize,
    start: usize,
    end: usize,
    cfg: &PreprocessRunConfig,
    cache_opt: Option<&SharedXicCache>,
    disk_cache: Option<&XicDiskCache>,
    xim_cache_opt: Option<&SharedXimCache>,
    xim_disk_cache: Option<&XimDiskCache>,
    apply_trace_filter: bool,
) -> Result<(PreprocessChunkPayload, PreprocessStageRuntimeStats)> {
    let chunk = &rows[start..end];
    let xic_started = Instant::now();
    let x_trace = build_traces_for_rows(
        chunk,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
        cache_opt,
        disk_cache,
    )?;
    let xic_elapsed = xic_started.elapsed();
    let xim_started = Instant::now();
    let x_xim = build_xim_for_rows(
        chunk,
        &cfg.xim_path,
        &cfg.xim_paths,
        &cfg.xim_map_path,
        &cfg.xim_trace,
        &cfg.xim_fetch,
        xim_cache_opt,
        xim_disk_cache,
    )?;
    let xim_elapsed = xim_started.elapsed();
    let (rows_chunk, x_trace, x_xim) = if apply_trace_filter {
        filter_rows_by_trace_with_aux(
            chunk.to_vec(),
            x_trace,
            x_xim,
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.xim_trace.as_ref().map(|c| c.total_c()),
            cfg.xim_trace.as_ref().map(|c| c.l),
        )
    } else {
        (chunk.to_vec(), x_trace, x_xim)
    };

    let source_rows = end.saturating_sub(start);
    let stats = PreprocessStageRuntimeStats {
        xic_rows: source_rows,
        xim_rows: if cfg.xim_trace.is_some() {
            source_rows
        } else {
            0
        },
        xic_build_time: xic_elapsed,
        xim_build_time: if cfg.xim_trace.is_some() {
            xim_elapsed
        } else {
            Duration::ZERO
        },
    };

    Ok((
        PreprocessChunkPayload {
            seq_idx,
            source_rows,
            rows: rows_chunk,
            x_trace,
            x_xim,
            xim_decode_issues: crate::io::xim_parquet::take_decode_issues(),
        },
        stats,
    ))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Build one or more preprocessing payloads from a contiguous group of archive
/// chunks.
///
/// When `group_ranges` contains multiple adjacent chunks, this helper builds
/// XIC/XIM tensors once for the full row union and then slices the result back
/// into per-chunk payloads. That substantially reduces repeated parquet scans
/// for XIM-heavy datasets while preserving the final archive layout.
fn preprocess_chunk_group_payloads(
    rows: &[FeatureRow],
    group_ranges: &[(usize, usize, usize)],
    cfg: &PreprocessRunConfig,
    cache_opt: Option<&SharedXicCache>,
    disk_cache: Option<&XicDiskCache>,
    xim_cache_opt: Option<&SharedXimCache>,
    xim_disk_cache: Option<&XimDiskCache>,
    apply_trace_filter: bool,
) -> Result<PreprocessChunkGroupPayload> {
    if group_ranges.is_empty() {
        return Ok(PreprocessChunkGroupPayload {
            payloads: Vec::new(),
            stage_stats: PreprocessStageRuntimeStats::default(),
        });
    }
    if apply_trace_filter || group_ranges.len() == 1 {
        let mut payloads = Vec::with_capacity(group_ranges.len());
        let mut stage_stats = PreprocessStageRuntimeStats::default();
        for &(seq_idx, start, end) in group_ranges {
            let (payload, payload_stats) = preprocess_chunk_payload(
                rows,
                seq_idx,
                start,
                end,
                cfg,
                cache_opt,
                disk_cache,
                xim_cache_opt,
                xim_disk_cache,
                apply_trace_filter,
            )?;
            payloads.push(payload);
            stage_stats += payload_stats;
        }
        return Ok(PreprocessChunkGroupPayload {
            payloads,
            stage_stats,
        });
    }

    let group_start = group_ranges
        .first()
        .map(|(_, start, _)| *start)
        .unwrap_or(0);
    let group_end = group_ranges
        .last()
        .map(|(_, _, end)| *end)
        .unwrap_or(group_start);
    let group_rows = &rows[group_start..group_end];
    let xic_started = Instant::now();
    let x_trace_all = build_traces_for_rows(
        group_rows,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
        cache_opt,
        disk_cache,
    )?;
    let xic_elapsed = xic_started.elapsed();
    let xim_started = Instant::now();
    let x_xim_all = build_xim_for_rows(
        group_rows,
        &cfg.xim_path,
        &cfg.xim_paths,
        &cfg.xim_map_path,
        &cfg.xim_trace,
        &cfg.xim_fetch,
        xim_cache_opt,
        xim_disk_cache,
    )?;
    let xim_elapsed = xim_started.elapsed();
    let xim_decode_issues = crate::io::xim_parquet::take_decode_issues();

    let trace_row_span = cfg.trace.total_c() * cfg.trace.l;
    let xim_row_span = cfg
        .xim_trace
        .as_ref()
        .map(|trace| trace.total_c() * trace.l);
    let mut payloads = Vec::with_capacity(group_ranges.len());
    for (payload_idx, &(seq_idx, start, end)) in group_ranges.iter().enumerate() {
        let local_start = start.saturating_sub(group_start);
        let local_end = end.saturating_sub(group_start);
        let trace_slice =
            x_trace_all[local_start * trace_row_span..local_end * trace_row_span].to_vec();
        let xim_slice = x_xim_all.as_ref().and_then(|all| {
            xim_row_span.map(|span| all[local_start * span..local_end * span].to_vec())
        });
        payloads.push(PreprocessChunkPayload {
            seq_idx,
            source_rows: end.saturating_sub(start),
            rows: rows[start..end].to_vec(),
            x_trace: trace_slice,
            x_xim: xim_slice,
            xim_decode_issues: if payload_idx == 0 {
                xim_decode_issues.clone()
            } else {
                Vec::new()
            },
        });
    }

    Ok(PreprocessChunkGroupPayload {
        payloads,
        stage_stats: PreprocessStageRuntimeStats {
            xic_rows: group_rows.len(),
            xim_rows: if cfg.xim_trace.is_some() {
                group_rows.len()
            } else {
                0
            },
            xic_build_time: xic_elapsed,
            xim_build_time: if cfg.xim_trace.is_some() {
                xim_elapsed
            } else {
                Duration::ZERO
            },
        },
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Materialize OSW rows plus fixed-width XIC/XIM tensors into a reusable bundle.
pub fn run_preprocess(cfg: &PreprocessRunConfig) -> Result<PreprocessRunOutput> {
    crate::io::xim_parquet::clear_decode_issues();
    let mut xim_decode_issues = Vec::new();

    let table = read_feature_rows(&cfg.osw_path, &cfg.osw)?;
    let mut rows = table.rows;
    if rows.is_empty() {
        bail!("no rows in OSW");
    }
    log_run_id_summary(
        &rows,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xim_path,
        &cfg.xim_paths,
    );
    if cfg.restrict_osw_to_xic_map {
        rows = filter_rows_by_xic_map(rows, &cfg.xic_paths, &cfg.xic_map_path)?;
        if rows.is_empty() {
            bail!("no rows after XIC map restriction; check run_id mapping");
        }
    }

    let xic_cache = SharedXicCache::new(cfg.xic_cache_max_precursors);
    let cache_stats = xic_cache.stats();
    if xic_cache.is_enabled() {
        log::info!(
            "Enabled XIC cache (max_precursors={})",
            cfg.xic_cache_max_precursors
        );
    }
    let disk_cache = match &cfg.xic_cache_dir {
        Some(dir) => Some(XicDiskCache::new(
            dir.clone(),
            cfg.xic_cache_max_bytes,
            cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xic_cache_dir {
        log::info!("Enabled XIC disk cache at {:?}", dir);
    }
    let cache_opt = if xic_cache.is_enabled() || disk_cache.is_some() {
        Some(&xic_cache)
    } else {
        None
    };

    let xim_cache = SharedXimCache::new(cfg.xim_cache_max_features);
    let xim_cache_stats = xim_cache.stats();
    if xim_cache.is_enabled() {
        log::info!(
            "Enabled XIM cache (max_features={})",
            cfg.xim_cache_max_features
        );
    }
    let xim_disk_cache = match &cfg.xim_cache_dir {
        Some(dir) => Some(XimDiskCache::new(
            dir.clone(),
            cfg.xim_cache_max_bytes,
            xim_cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xim_cache_dir {
        log::info!("Enabled XIM disk cache at {:?}", dir);
    }
    let xim_cache_opt = if xim_cache.is_enabled() || xim_disk_cache.is_some() {
        Some(&xim_cache)
    } else {
        None
    };

    let apply_trace_filter = cfg.restrict_osw_to_xic_map
        && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_none();
    if cfg.restrict_osw_to_xic_map && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_some()
    {
        log::info!(
            "XIC map/path list provided; skipping trace-based restriction during preprocessing (run_id filter only)"
        );
    }

    let provenance = collect_preprocess_provenance(
        &cfg.osw_path,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xic_map_path,
        &cfg.xim_path,
        &cfg.xim_paths,
        &cfg.xim_map_path,
    );
    let manifest = PreprocessedManifest::new(
        cfg.chunk_row_count.max(1),
        table.feature_cols.clone(),
        cfg.trace.clone(),
        cfg.xim_trace.clone(),
        cfg.osw.clone(),
        provenance,
    );
    let mut writer =
        ResumablePreprocessedBundleWriter::resume_or_create(&cfg.output_path, manifest)?;
    let chunk_row_count = cfg.chunk_row_count.max(1);
    let mut progress = InferenceProgressLogger::new("preprocess", rows.len(), chunk_row_count);
    let work_ranges = build_stream_work_ranges(rows.len(), chunk_row_count);
    let completed_chunks = writer.completed_chunks();
    let mut written_rows = writer.completed_rows();
    let mut written_chunks = completed_chunks;
    let mut processed_source_rows: usize = writer
        .manifest()
        .chunks
        .iter()
        .map(|chunk| {
            if chunk.source_rows == 0 {
                chunk.rows
            } else {
                chunk.source_rows
            }
        })
        .sum();
    let remaining_work: Vec<(usize, usize, usize)> = work_ranges
        .iter()
        .enumerate()
        .skip(completed_chunks)
        .map(|(seq_idx, &(start, end))| (seq_idx, start, end))
        .collect();
    let remaining_source_rows: usize = remaining_work
        .iter()
        .map(|(_, start, end)| end.saturating_sub(*start))
        .sum();
    log::info!(
        "Preprocess resume state | output={:?} resumed_chunks={} resumed_rows={} remaining_chunks={} remaining_rows={}",
        cfg.output_path,
        completed_chunks,
        written_rows,
        remaining_work.len(),
        remaining_source_rows
    );
    let producer_workers = derive_preprocess_producer_workers(
        remaining_work.len(),
        cfg.xim_trace.is_some(),
        chunk_row_count,
    );
    let queue_depth = derive_preprocess_queue_depth(producer_workers, cfg.xim_trace.is_some());
    let chunk_group_size = derive_preprocess_chunk_group_size(
        cfg.xim_trace.is_some(),
        apply_trace_filter,
        producer_workers,
        chunk_row_count,
    );
    let grouped_work: Vec<Vec<(usize, usize, usize)>> = if remaining_work.is_empty() {
        Vec::new()
    } else {
        remaining_work
            .chunks(chunk_group_size.max(1))
            .map(|group| group.to_vec())
            .collect()
    };
    #[cfg(feature = "rayon")]
    let rayon_threads = rayon::current_num_threads();
    #[cfg(not(feature = "rayon"))]
    let rayon_threads = 1usize;
    let grouped_work_unit_rows = chunk_row_count.saturating_mul(chunk_group_size.max(1));
    log::info!(
        "Parallel preprocessing configured | chunk_rows={} grouped_work_unit_size={} rows ({} chunk(s)) grouped_work_units={} producer_workers={} queue_depth={} rayon_threads={}",
        chunk_row_count,
        grouped_work_unit_rows,
        chunk_group_size,
        grouped_work.len(),
        producer_workers,
        queue_depth,
        rayon_threads
    );

    let next_chunk_idx = AtomicUsize::new(0);
    let (tx, rx) = mpsc::sync_channel::<Result<PreprocessProducerMessage>>(queue_depth);

    thread::scope(|scope| -> Result<()> {
        for _worker_idx in 0..producer_workers {
            let tx_producer = tx.clone();
            let next_chunk_idx_ref = &next_chunk_idx;
            let grouped_work_ref = &grouped_work;
            let rows_ref = rows.as_slice();
            let cfg_ref = cfg;
            let cache_opt_ref = cache_opt;
            let disk_cache_ref = disk_cache.as_ref();
            let xim_cache_opt_ref = xim_cache_opt;
            let xim_disk_cache_ref = xim_disk_cache.as_ref();
            scope.spawn(move || -> Result<()> {
                loop {
                    let seq_idx = next_chunk_idx_ref.fetch_add(1, Ordering::Relaxed);
                    let Some(group_ranges) = grouped_work_ref.get(seq_idx) else {
                        break;
                    };
                    let payload_group = preprocess_chunk_group_payloads(
                        rows_ref,
                        group_ranges,
                        cfg_ref,
                        cache_opt_ref,
                        disk_cache_ref,
                        xim_cache_opt_ref,
                        xim_disk_cache_ref,
                        apply_trace_filter,
                    )?;
                    tx_producer
                        .send(Ok(PreprocessProducerMessage::PayloadGroup(payload_group)))
                        .context(
                            "preprocess writer dropped before chunk preprocessing completed",
                        )?;
                }
                tx_producer
                    .send(Ok(PreprocessProducerMessage::Done))
                    .context("preprocess writer dropped before completion")?;
                Ok(())
            });
        }
        drop(tx);

        let mut completed_workers = 0usize;
        let mut next_expected = completed_chunks;
        let mut pending: BTreeMap<usize, PreprocessChunkGroupPayload> = BTreeMap::new();
        let mut stage_totals = PreprocessStageRuntimeStats::default();

        while completed_workers < producer_workers {
            let message = rx.recv().context("preprocess worker disconnected")?;
            match message? {
                PreprocessProducerMessage::Done => {
                    completed_workers += 1;
                }
                PreprocessProducerMessage::PayloadGroup(group) => {
                    if let Some(first_seq_idx) =
                        group.payloads.first().map(|payload| payload.seq_idx)
                    {
                        pending.insert(first_seq_idx, group);
                    }
                }
            }

            while let Some(group) = pending.remove(&next_expected) {
                let PreprocessChunkGroupPayload {
                    payloads,
                    stage_stats,
                } = group;
                for payload in payloads {
                    xim_decode_issues.extend(payload.xim_decode_issues);
                    processed_source_rows += payload.source_rows;
                    if !payload.rows.is_empty() {
                        writer.write_chunk(
                            &payload.rows,
                            &payload.x_trace,
                            payload.x_xim.as_deref(),
                            payload.source_rows,
                        )?;
                        written_rows += payload.rows.len();
                        written_chunks += 1;
                    }
                    next_expected += 1;
                }
                stage_totals += stage_stats;
                if progress.maybe_log(processed_source_rows.min(rows.len()), next_expected) {
                    log_preprocess_stage_throughput(&stage_totals);
                }
            }
        }

        while let Some(group) = pending.remove(&next_expected) {
            let PreprocessChunkGroupPayload {
                payloads,
                stage_stats,
            } = group;
            for payload in payloads {
                xim_decode_issues.extend(payload.xim_decode_issues);
                processed_source_rows += payload.source_rows;
                if !payload.rows.is_empty() {
                    writer.write_chunk(
                        &payload.rows,
                        &payload.x_trace,
                        payload.x_xim.as_deref(),
                        payload.source_rows,
                    )?;
                    written_rows += payload.rows.len();
                    written_chunks += 1;
                }
                next_expected += 1;
            }
            stage_totals += stage_stats;
            if progress.maybe_log(processed_source_rows.min(rows.len()), next_expected) {
                log_preprocess_stage_throughput(&stage_totals);
            }
        }
        Ok(())
    })?;

    log::info!(
        "Finalizing preprocessed bundle archive {:?} from {} staged chunks...",
        cfg.output_path,
        writer.completed_chunks()
    );
    let manifest = writer.finish()?;
    log_xic_cache_stats("preprocess", &cache_stats);
    log_xim_cache_stats("preprocess", &xim_cache_stats);
    report_xim_decode_issues(
        "preprocess",
        &xim_decode_issues,
        &diagnostic_tsv_path(&cfg.output_path, "xim_skipped"),
    )?;
    log::info!(
        "Wrote preprocessed TOPAZ bundle to {:?} (rows={} chunks={})",
        cfg.output_path,
        manifest.row_count,
        manifest.chunks.len()
    );
    Ok(PreprocessRunOutput {
        output_path: cfg.output_path.clone(),
        n_rows: manifest.row_count,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn apply_xrun_to_row_scores_from_preprocessed(
    rows: &[FeatureRow],
    row_scores: &[f32],
    x_trace: &[f32],
    x_xim: Option<&[f32]>,
    model: &TopazBagRanker,
    table_feature_cols: &[String],
    target_cols: &[String],
    trace_cfg: &TraceBuildConfig,
    xim_trace_cfg: Option<&TraceBuildConfig>,
    xrun_cfg: &XrunRunConfig,
    xrun_model: &XrunAttentionCalibrator,
    xrun_meta: &XrunCheckpointMeta,
    device: &Device,
    base_batch_size: usize,
    pre: Option<&crate::Preprocessor>,
) -> Result<XrunAppliedScores> {
    let (winner_rows, bag_pid, bag_score, _bag_is_decoy, bag_y) =
        select_bag_winners(rows, row_scores);
    if winner_rows.is_empty() {
        return Ok(XrunAppliedScores {
            row_scores: row_scores.to_vec(),
            bag_pid,
            bag_score,
        });
    }
    let row_index = build_feature_row_index(rows);
    let x_trace_winners = gather_row_aligned_tensor_by_feature_id(
        &winner_rows,
        &row_index,
        x_trace,
        trace_cfg.total_c() * trace_cfg.l,
    )?;
    let x_xim_winners = if let (Some(x_xim), Some(xim_cfg)) = (x_xim, xim_trace_cfg) {
        Some(gather_row_aligned_tensor_by_feature_id(
            &winner_rows,
            &row_index,
            x_xim,
            xim_cfg.total_c() * xim_cfg.l,
        )?)
    } else {
        None
    };
    let head_out = score_bags_with_heads_from_rows_with_cols_with_aux(
        model,
        &winner_rows,
        &x_trace_winners,
        x_xim_winners
            .as_ref()
            .zip(xim_trace_cfg)
            .map(|(x, cfg)| (x.as_slice(), cfg)),
        table_feature_cols,
        target_cols,
        trace_cfg.total_c(),
        trace_cfg.l,
        1,
        device,
        base_batch_size,
        pre,
    )?;

    let predict_cfg = XrunPredictConfig {
        max_runs: if xrun_cfg.max_runs > 0 {
            xrun_cfg.max_runs
        } else {
            xrun_meta.predict.max_runs
        },
        sort_by: if !xrun_cfg.sort_by.is_empty() {
            xrun_cfg.sort_by.clone()
        } else {
            xrun_meta.predict.sort_by.clone()
        },
        batch_size: if xrun_cfg.batch_size > 0 {
            xrun_cfg.batch_size
        } else {
            xrun_meta.predict.batch_size
        },
    };
    let bag_data = crate::xrun::XrunBagData {
        bag_score: bag_score.clone(),
        bag_hidden: head_out.winner_hidden.clone(),
        hidden_dim: head_out.hidden_dim,
        bag_y: bag_y.clone(),
        bag_pid: bag_pid.clone(),
    };
    let (delta_bag, _attn_entropy) =
        crate::xrun::xrun_predict_deltas_for_bags(xrun_model, &bag_data, &predict_cfg, device)?;
    let row_scores = apply_xrun_deltas_to_rows(row_scores, rows, &bag_pid, &delta_bag);
    let bag_score = apply_xrun_deltas(&bag_score, &delta_bag);

    Ok(XrunAppliedScores {
        row_scores,
        bag_pid,
        bag_score,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn run_training_with_preprocessed(cfg: &TrainRunConfig, device: &Device) -> Result<TrainRunOutput> {
    let bundle_path = cfg
        .preprocessed_path
        .as_ref()
        .expect("preprocessed training branch requires a bundle path");
    let init_base: Option<PathBuf> = cfg
        .init_checkpoint
        .as_ref()
        .map(|path| checkpoint_base(path.as_path()));
    let init_meta = if let Some(base) = init_base.as_ref() {
        Some(read_checkpoint_meta(base)?)
    } else {
        None
    };

    let reader = PreprocessedBundleReader::open(bundle_path)?;
    let manifest = reader.manifest().clone();
    let bundle_feature_cols = manifest.feature_cols.clone();
    let mut selected_cols = if let Some(meta) = init_meta.as_ref() {
        if cfg.feature_select.mode != FeatureMode::All || cfg.feature_select.cols.is_some() {
            log::warn!("init_checkpoint provided; using feature columns stored in checkpoint");
        }
        meta.feature_cols.clone()
    } else {
        resolve_feature_cols(&bundle_feature_cols, &cfg.feature_select)
    };
    let mut model_cfg = if let Some(meta) = init_meta.as_ref() {
        meta.model.clone()
    } else {
        cfg.model.clone()
    };
    if !model_cfg.use_heuristic_features && !selected_cols.is_empty() {
        log::warn!("model.use_heuristic_features=false; ignoring selected feature columns");
        selected_cols.clear();
    }
    if selected_cols.is_empty() {
        model_cfg.use_heuristic_features = false;
        model_cfg.feat_dim = 0;
    } else {
        model_cfg.use_heuristic_features = true;
        if model_cfg.feat_dim != selected_cols.len() {
            if init_meta.is_some() {
                bail!(
                    "init_checkpoint feature dim mismatch: checkpoint expects {}, recovered {} columns",
                    model_cfg.feat_dim,
                    selected_cols.len()
                );
            }
            if model_cfg.feat_dim != 0 {
                log::warn!(
                    "overriding model.feat_dim={} to match selected columns ({})",
                    model_cfg.feat_dim,
                    selected_cols.len()
                );
            }
            model_cfg.feat_dim = selected_cols.len();
        }
    }
    validate_preprocessed_manifest_for_training(&manifest, cfg, &model_cfg, &selected_cols)?;
    let xim_trace_cfg = effective_xim_trace_cfg(&cfg.xim_trace, &model_cfg);

    log::info!(
        "Loading preprocessed TOPAZ bundle {:?} (rows={} chunks={})",
        bundle_path,
        manifest.row_count,
        manifest.chunks.len()
    );
    let dataset = reader.load_all()?;
    let mut rows = filter_training_rows(dataset.rows.clone(), &cfg.filter);
    if rows.is_empty() {
        bail!("no rows after filtering");
    }
    log_run_id_summary(
        &rows,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xim_path,
        &cfg.xim_paths,
    );
    if cfg.restrict_osw_to_xic_map {
        rows = filter_rows_by_xic_map(rows, &cfg.xic_paths, &cfg.xic_map_path)?;
        if rows.is_empty() {
            bail!("no rows after XIC map restriction; check run_id mapping");
        }
    }

    let full_index = build_feature_row_index(&dataset.rows);
    let trace_span = cfg.trace.total_c() * cfg.trace.l;
    let x_all =
        gather_row_aligned_tensor_by_feature_id(&rows, &full_index, &dataset.x_trace, trace_span)?;
    let x_all_xim =
        if let (Some(all), Some(xim_cfg)) = (dataset.x_xim.as_ref(), xim_trace_cfg.as_ref()) {
            Some(gather_row_aligned_tensor_by_feature_id(
                &rows,
                &full_index,
                all,
                xim_cfg.total_c() * xim_cfg.l,
            )?)
        } else {
            None
        };

    let (mut rows_tr, rows_va) = split_rows_by_precursor(&rows, cfg.val_frac, cfg.seed);
    if cfg.train_frac < 1.0 {
        rows_tr =
            subsample_train_rows_by_bag(rows_tr, cfg.train_frac, cfg.train_stratify_run, cfg.seed);
        if rows_tr.is_empty() {
            bail!("no rows after train subsample");
        }
    }

    let filtered_index = build_feature_row_index(&rows);
    let x_tr =
        gather_row_aligned_tensor_by_feature_id(&rows_tr, &filtered_index, &x_all, trace_span)?;
    let x_va =
        gather_row_aligned_tensor_by_feature_id(&rows_va, &filtered_index, &x_all, trace_span)?;
    let (x_tr_xim, x_va_xim) =
        if let (Some(all), Some(xim_cfg)) = (x_all_xim.as_ref(), xim_trace_cfg.as_ref()) {
            let span = xim_cfg.total_c() * xim_cfg.l;
            (
                Some(gather_row_aligned_tensor_by_feature_id(
                    &rows_tr,
                    &filtered_index,
                    all,
                    span,
                )?),
                Some(gather_row_aligned_tensor_by_feature_id(
                    &rows_va,
                    &filtered_index,
                    all,
                    span,
                )?),
            )
        } else {
            (None, None)
        };

    let pre = if let Some(meta) = init_meta.as_ref() {
        if model_cfg.use_heuristic_features {
            meta.preprocess.clone()
        } else {
            None
        }
    } else if model_cfg.use_heuristic_features {
        Some(fit_preprocessor_from_rows_with_cols(
            &rows_tr,
            &bundle_feature_cols,
            &selected_cols,
        ))
    } else {
        None
    };

    if cfg.diagnostics.trace_summary {
        let sum_tr = trace_summary(
            &x_tr,
            rows_tr.len(),
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.trace.ms1_cmax,
            cfg.trace.ms2_cmax,
        );
        print_trace_summary(&sum_tr, "train");
        warn_if_missing_ms1(&sum_tr, "train");
        if !rows_va.is_empty() {
            let sum_va = trace_summary(
                &x_va,
                rows_va.len(),
                cfg.trace.total_c(),
                cfg.trace.l,
                cfg.trace.ms1_cmax,
                cfg.trace.ms2_cmax,
            );
            print_trace_summary(&sum_va, "val");
            warn_if_missing_ms1(&sum_va, "val");
        }
    }

    let x_feat = if model_cfg.use_heuristic_features {
        rows_to_feature_matrix_with_cols(
            &rows_tr,
            &bundle_feature_cols,
            &selected_cols,
            pre.as_ref(),
        )
    } else {
        Vec::new()
    };
    let x_feat_va = if model_cfg.use_heuristic_features && !rows_va.is_empty() {
        rows_to_feature_matrix_with_cols(
            &rows_va,
            &bundle_feature_cols,
            &selected_cols,
            pre.as_ref(),
        )
    } else {
        Vec::new()
    };

    let y_rows: Vec<u8> = rows_tr
        .iter()
        .map(|r| if r.is_decoy { 1 } else { 0 })
        .collect();
    let pid_rows: Vec<String> = rows_tr.iter().map(|r| r.group_id.clone()).collect();
    let bags = crate::building_blocks::bagging::make_bags_with_traces(
        &x_feat,
        rows_tr.len(),
        model_cfg.feat_dim,
        &x_tr,
        cfg.trace.total_c(),
        cfg.trace.l,
        &y_rows,
        &pid_rows,
        cfg.bag_k,
    );
    let (mut n_pos, mut n_neg) = (0usize, 0usize);
    for &y in &bags.y_bag {
        if y > 0.5 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    let pos_weight = if n_pos > 0 {
        n_neg as f32 / n_pos as f32
    } else {
        1.0
    };
    let batches =
        if let (Some(x_tr_xim), Some(xim_cfg)) = (x_tr_xim.as_ref(), xim_trace_cfg.as_ref()) {
            let aux_bags = crate::building_blocks::bagging::make_bags_with_traces(
                &x_feat,
                rows_tr.len(),
                model_cfg.feat_dim,
                x_tr_xim,
                xim_cfg.total_c(),
                xim_cfg.l,
                &y_rows,
                &pid_rows,
                cfg.bag_k,
            );
            bags_to_train_batches_with_aux(
                bags,
                Some((aux_bags.t_bag, xim_cfg.total_c(), xim_cfg.l)),
                device,
                cfg.batch_size,
            )?
        } else {
            bags_to_train_batches(bags, device, cfg.batch_size)?
        };

    let mut trainer = Trainer::new(cfg.train.clone(), &model_cfg, device)?;
    if let Some(base) = init_base.as_ref() {
        let (_meta, report) = load_checkpoint_partial(base, &mut trainer.varmap)?;
        if report.loaded == 0 {
            bail!(
                "init_checkpoint {:?} did not provide any compatible tensors for the current model",
                base
            );
        }
        log::info!(
            "Loaded initialization checkpoint from {:?} (loaded={}, missing={}, shape_mismatch={}, set_errors={}, extra_in_checkpoint={})",
            base,
            report.loaded,
            report.missing_in_checkpoint.len(),
            report.shape_mismatch.len(),
            report.set_errors.len(),
            report.extra_in_checkpoint.len()
        );
    }
    trainer.set_pos_weight(pos_weight);
    trainer.set_shuffle_seed(cfg.seed);
    log::info!(
        "Using pos_weight={:.4} (n_pos={}, n_neg={})",
        pos_weight,
        n_pos,
        n_neg
    );
    let val_batches = if !rows_va.is_empty() {
        let y_rows_va: Vec<u8> = rows_va
            .iter()
            .map(|r| if r.is_decoy { 1 } else { 0 })
            .collect();
        let pid_rows_va: Vec<String> = rows_va.iter().map(|r| r.group_id.clone()).collect();
        let bags_va = crate::building_blocks::bagging::make_bags_with_traces(
            &x_feat_va,
            rows_va.len(),
            model_cfg.feat_dim,
            &x_va,
            cfg.trace.total_c(),
            cfg.trace.l,
            &y_rows_va,
            &pid_rows_va,
            cfg.bag_k,
        );
        if let (Some(x_va_xim), Some(xim_cfg)) = (x_va_xim.as_ref(), xim_trace_cfg.as_ref()) {
            let aux_bags_va = crate::building_blocks::bagging::make_bags_with_traces(
                &x_feat_va,
                rows_va.len(),
                model_cfg.feat_dim,
                x_va_xim,
                xim_cfg.total_c(),
                xim_cfg.l,
                &y_rows_va,
                &pid_rows_va,
                cfg.bag_k,
            );
            bags_to_train_batches_with_aux(
                bags_va,
                Some((aux_bags_va.t_bag, xim_cfg.total_c(), xim_cfg.l)),
                device,
                cfg.batch_size,
            )?
        } else {
            bags_to_train_batches(bags_va, device, cfg.batch_size)?
        }
    } else {
        Vec::new()
    };
    let _history = trainer.train_epochs_early_stop(&batches, &val_batches, cfg.max_epochs, None)?;

    if !rows_va.is_empty() {
        let out = score_bags_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows_va,
            &x_va,
            x_va_xim
                .as_ref()
                .zip(xim_trace_cfg.as_ref())
                .map(|(x, cfg)| (x.as_slice(), cfg)),
            &bundle_feature_cols,
            &selected_cols,
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.bag_k,
            device,
            cfg.batch_size,
            pre.as_ref(),
        )?;
        let summ = tdc_summary(&out.bag_score, &out.is_decoy, 0.01);
        log::info!(
            "VAL TDC summary @q=0.01: cutoff={:.4} targets={} decoys={}",
            summ.cutoff,
            summ.n_targets,
            summ.n_decoys
        );
    }

    if cfg.diagnostics.save_head_embeddings {
        let outdir = cfg
            .diagnostics
            .head_embeddings_outdir
            .clone()
            .unwrap_or_else(|| PathBuf::from("head_embeddings"));
        std::fs::create_dir_all(&outdir)?;
        let mut rows_all = rows_tr.clone();
        rows_all.extend(rows_va.iter().cloned());
        let mut x_all = x_tr.clone();
        x_all.extend_from_slice(&x_va);
        let mut x_all_xim = None;
        if let Some(x_tr_xim) = x_tr_xim.as_ref() {
            let mut buf = x_tr_xim.clone();
            if let Some(x_va_xim) = x_va_xim.as_ref() {
                buf.extend_from_slice(x_va_xim);
            }
            x_all_xim = Some(buf);
        }
        let out = score_bags_with_heads_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows_all,
            &x_all,
            x_all_xim
                .as_ref()
                .zip(xim_trace_cfg.as_ref())
                .map(|(x, cfg)| (x.as_slice(), cfg)),
            &bundle_feature_cols,
            &selected_cols,
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.bag_k,
            device,
            cfg.batch_size,
            pre.as_ref(),
        )?;
        let out_path = outdir.join("head_embeddings.tsv");
        write_head_embeddings_tsv(&out_path, &out)?;
    }

    let meta = CheckpointMeta {
        model: model_cfg,
        train: Some(cfg.train.clone()),
        trace: Some(cfg.trace.clone()),
        feature_cols: selected_cols,
        preprocess: pre,
        version: 1,
    };
    save_checkpoint(&cfg.output_prefix, &trainer.varmap, &meta)?;

    if cfg.xrun.enabled {
        let mut rows_all = rows_tr.clone();
        rows_all.extend(rows_va.iter().cloned());
        let mut x_all = x_tr.clone();
        x_all.extend_from_slice(&x_va);
        let mut x_all_xim = None;
        if let Some(x_tr_xim) = x_tr_xim.as_ref() {
            let mut buf = x_tr_xim.clone();
            if let Some(x_va_xim) = x_va_xim.as_ref() {
                buf.extend_from_slice(x_va_xim);
            }
            x_all_xim = Some(buf);
        }
        let bag_data = build_xrun_bag_data_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows_all,
            &x_all,
            x_all_xim
                .as_ref()
                .zip(xim_trace_cfg.as_ref())
                .map(|(x, cfg)| (x.as_slice(), cfg)),
            &bundle_feature_cols,
            &meta.feature_cols,
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.bag_k,
            device,
            cfg.xrun.batch_size.max(1),
            meta.preprocess.as_ref(),
        )?;
        let seq = build_xrun_sequences_from_bags(
            &bag_data.bag_pid,
            &bag_data.bag_score,
            &bag_data.bag_hidden,
            bag_data.hidden_dim,
            &bag_data.bag_y,
            cfg.xrun.max_runs,
            &cfg.xrun.sort_by,
        );
        let ds = XrunDataset {
            xseq: seq.xseq,
            mask: seq.mask,
            y: seq.y_prec,
            p: seq.p,
            r: seq.r,
            din: seq.din,
        };
        let (tr_ds, va_ds) = split_train_val(&ds, cfg.val_frac, cfg.seed);
        if tr_ds.p == 0 || va_ds.p == 0 {
            log::warn!(
                "Skipping XRUN training because train/val sequence split is empty (train_p={}, val_p={})",
                tr_ds.p,
                va_ds.p
            );
        } else {
            let mut xrun_trainer = XrunTrainer::new(cfg.xrun.train.clone(), ds.din, device)?;
            let xrun_meta = xrun_trainer.train(&tr_ds, &va_ds, device)?;
            let xrun_ckpt_meta = XrunCheckpointMeta {
                train: cfg.xrun.train.clone(),
                predict: XrunPredictConfig {
                    max_runs: cfg.xrun.max_runs,
                    sort_by: cfg.xrun.sort_by.clone(),
                    batch_size: cfg.xrun.batch_size.max(1),
                },
                in_dim: ds.din,
                best_val: Some(xrun_meta.best_val),
                version: 1,
            };
            save_xrun_checkpoint(&cfg.output_prefix, &xrun_trainer.varmap, &xrun_ckpt_meta)?;
            log::info!(
                "Saved XRUN calibrator alongside checkpoint (best_val={:.4})",
                xrun_meta.best_val
            );
        }
    }

    Ok(TrainRunOutput {
        checkpoint_prefix: cfg.output_prefix.clone(),
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn run_inference_with_preprocessed(
    cfg: &InferRunConfig,
    device: &Device,
) -> Result<InferRunOutput> {
    let bundle_path = cfg
        .preprocessed_path
        .as_ref()
        .expect("preprocessed inference branch requires a bundle path");
    let base = checkpoint_base(&cfg.checkpoint);
    let meta = read_checkpoint_meta(&base)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = TopazBagRanker::new(vb.pp("topaz"), &meta.model)?;
    load_checkpoint_weights(&base, &mut varmap)?;

    let reader = PreprocessedBundleReader::open(bundle_path)?;
    let manifest = reader.manifest().clone();
    validate_preprocessed_manifest_for_inference(&manifest, cfg, &meta)?;
    log::info!(
        "Loading preprocessed TOPAZ bundle {:?} (rows={} chunks={})",
        bundle_path,
        manifest.row_count,
        manifest.chunks.len()
    );
    let load_started = Instant::now();
    let dataset = reader.load_all()?;
    let mut runtime_stats = InferenceRuntimeStats {
        mode: "preprocessed",
        chunk_size: if cfg.trace_chunk_size == 0 {
            dataset.rows.len().max(1)
        } else {
            cfg.trace_chunk_size.max(1)
        },
        work_unit_size: None,
        producer_workers: None,
        queue_depth: None,
        decode_time: load_started.elapsed(),
        score_time: Duration::ZERO,
    };

    let all_rows = dataset.rows;
    let mut rows = all_rows.clone();
    if rows.is_empty() {
        bail!("no rows in preprocessed bundle");
    }
    if cfg.restrict_osw_to_xic_map {
        rows = filter_rows_by_xic_map(rows, &cfg.xic_paths, &cfg.xic_map_path)?;
        if rows.is_empty() {
            bail!("no rows after XIC map restriction; check run_id mapping");
        }
    }
    let full_index = build_feature_row_index(&all_rows);
    let trace_span = cfg.trace.total_c() * cfg.trace.l;
    let x_trace_all = if rows.len() == all_rows.len() {
        dataset.x_trace
    } else {
        gather_row_aligned_tensor_by_feature_id(&rows, &full_index, &dataset.x_trace, trace_span)?
    };
    let xim_trace_cfg = effective_xim_trace_cfg(&cfg.xim_trace, &meta.model);
    let x_xim_all =
        if let (Some(all), Some(xim_cfg)) = (dataset.x_xim.as_ref(), xim_trace_cfg.as_ref()) {
            Some(gather_row_aligned_tensor_by_feature_id(
                &rows,
                &full_index,
                all,
                xim_cfg.total_c() * xim_cfg.l,
            )?)
        } else {
            None
        };

    let mut scores = vec![0f32; rows.len()];
    let mut sum_n = 0usize;
    let mut sum_ms1 = 0usize;
    let mut sum_ms2 = 0usize;
    let mut progress =
        InferenceProgressLogger::new("preprocessed", rows.len(), runtime_stats.chunk_size);
    for start in (0..rows.len()).step_by(runtime_stats.chunk_size) {
        let end = (start + runtime_stats.chunk_size).min(rows.len());
        let row_slice = &rows[start..end];
        let x_trace = &x_trace_all[start * trace_span..end * trace_span];
        let x_xim = x_xim_all.as_ref().and_then(|all| {
            xim_trace_cfg
                .as_ref()
                .map(|cfg| &all[start * (cfg.total_c() * cfg.l)..end * (cfg.total_c() * cfg.l)])
        });
        if cfg.diagnostics.trace_summary {
            let sum = trace_summary(
                x_trace,
                row_slice.len(),
                cfg.trace.total_c(),
                cfg.trace.l,
                cfg.trace.ms1_cmax,
                cfg.trace.ms2_cmax,
            );
            sum_n += sum.n;
            sum_ms1 += sum.ms1_nonzero_rows;
            sum_ms2 += sum.ms2_nonzero_rows;
        }
        let score_started = Instant::now();
        let scores_chunk = score_inference_chunk(
            &model,
            row_slice,
            x_trace,
            x_xim,
            &manifest.feature_cols,
            &meta.feature_cols,
            &cfg.trace,
            xim_trace_cfg.as_ref(),
            meta.model.use_heuristic_features,
            meta.model.feat_dim,
            device,
            cfg.batch_size.max(1),
            meta.preprocess.as_ref(),
        )?;
        runtime_stats.score_time += score_started.elapsed();
        scores[start..end].copy_from_slice(&scores_chunk);
        progress.maybe_log(end, start / runtime_stats.chunk_size + 1);
    }

    log_inference_runtime_stats(&runtime_stats, rows.len());
    if cfg.diagnostics.trace_summary {
        let sum = crate::infer::diagnostics::TraceSummary {
            n: sum_n,
            l: cfg.trace.l,
            ms1_cmax: cfg.trace.ms1_cmax,
            ms2_cmax: cfg.trace.ms2_cmax,
            ms1_nonzero_rows: sum_ms1,
            ms2_nonzero_rows: sum_ms2,
        };
        print_trace_summary(&sum, "infer");
        warn_if_missing_ms1(&sum, "infer");
    }

    let skip_xrun = cfg.fast_inference && cfg.xrun.enabled;
    let skip_head_embeddings = cfg.fast_inference && cfg.diagnostics.save_head_embeddings;
    if skip_xrun {
        log::info!("fast_inference=true: skipping XRUN application during main inference run");
    }
    if skip_head_embeddings {
        log::info!("fast_inference=true: skipping head-embedding export during main inference run");
    }

    let base_scores = scores.clone();
    let xrun_applied = if cfg.xrun.enabled && !cfg.fast_inference {
        let (_xrun_varmap, xrun_model, xrun_meta) = load_xrun_calibrator(&base, device)?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "XRUN enabled but no XRUN checkpoint sidecar found for {:?}",
                    base
                )
            })?;
        let applied = apply_xrun_to_row_scores_from_preprocessed(
            &rows,
            &base_scores,
            &x_trace_all,
            x_xim_all.as_deref(),
            &model,
            &manifest.feature_cols,
            &meta.feature_cols,
            &cfg.trace,
            xim_trace_cfg.as_ref(),
            &cfg.xrun,
            &xrun_model,
            &xrun_meta,
            device,
            cfg.batch_size.max(1),
            meta.preprocess.as_ref(),
        )?;
        log::info!(
            "Applied XRUN calibration to {} bags using {:?} pooling",
            applied.bag_pid.len(),
            xrun_meta.train.pool
        );
        scores = applied.row_scores.clone();
        Some(applied)
    } else {
        None
    };

    let table_rows_base = build_score_table_from_rows(&rows, &base_scores, cfg.pep_bins);
    let table_rows_final = build_score_table_from_rows(&rows, &scores, cfg.pep_bins);
    crate::infer::write_score_tsv(&cfg.output_tsv, &table_rows_final)?;

    if cfg.diagnostics.save_head_embeddings && !cfg.fast_inference {
        let outdir = cfg
            .diagnostics
            .head_embeddings_outdir
            .clone()
            .unwrap_or_else(|| PathBuf::from("head_embeddings"));
        std::fs::create_dir_all(&outdir)?;
        if cfg.trace_chunk_size > 0 {
            log::info!("Computing head embeddings from preprocessed winner rows...");
            let (winner_rows, bag_pid, bag_score, bag_is_decoy, bag_y) =
                select_bag_winners(&rows, &scores);
            if !winner_rows.is_empty() {
                let row_index = build_feature_row_index(&rows);
                let x_trace = gather_row_aligned_tensor_by_feature_id(
                    &winner_rows,
                    &row_index,
                    &x_trace_all,
                    trace_span,
                )?;
                let x_xim = if let (Some(all), Some(xim_cfg)) =
                    (x_xim_all.as_ref(), xim_trace_cfg.as_ref())
                {
                    Some(gather_row_aligned_tensor_by_feature_id(
                        &winner_rows,
                        &row_index,
                        all,
                        xim_cfg.total_c() * xim_cfg.l,
                    )?)
                } else {
                    None
                };
                let mut out = score_bags_with_heads_from_rows_with_cols_with_aux(
                    &model,
                    &winner_rows,
                    &x_trace,
                    x_xim
                        .as_ref()
                        .zip(xim_trace_cfg.as_ref())
                        .map(|(x, cfg)| (x.as_slice(), cfg)),
                    &manifest.feature_cols,
                    &meta.feature_cols,
                    cfg.trace.total_c(),
                    cfg.trace.l,
                    1,
                    device,
                    cfg.batch_size,
                    meta.preprocess.as_ref(),
                )?;
                out.bag_pid = bag_pid;
                out.bag_score = bag_score;
                out.is_decoy = bag_is_decoy;
                out.bag_y = bag_y;
                let out_path = outdir.join("head_embeddings.tsv");
                write_head_embeddings_tsv(&out_path, &out)?;
            }
        } else {
            let mut out = score_bags_with_heads_from_rows_with_cols_with_aux(
                &model,
                &rows,
                &x_trace_all,
                x_xim_all
                    .as_ref()
                    .zip(xim_trace_cfg.as_ref())
                    .map(|(x, cfg)| (x.as_slice(), cfg)),
                &manifest.feature_cols,
                &meta.feature_cols,
                cfg.trace.total_c(),
                cfg.trace.l,
                cfg.bag_k,
                device,
                cfg.batch_size,
                meta.preprocess.as_ref(),
            )?;
            if let Some(applied) = xrun_applied.as_ref() {
                let mut bag_score_map = HashMap::new();
                for (pid, score) in applied.bag_pid.iter().zip(applied.bag_score.iter()) {
                    bag_score_map.insert(pid.clone(), *score);
                }
                for (i, pid) in out.bag_pid.iter().enumerate() {
                    if let Some(score) = bag_score_map.get(pid) {
                        out.bag_score[i] = *score;
                    }
                }
            }
            let out_path = outdir.join("head_embeddings.tsv");
            write_head_embeddings_tsv(&out_path, &out)?;
        }
    }

    if let Some(osw_path) = &cfg.output_osw {
        crate::io::osw::prepare_output_osw(&cfg.osw_path, osw_path)?;
        if let Some(base_name) = cfg
            .output_table_base
            .as_ref()
            .filter(|name| !name.is_empty())
        {
            let osw_rows = score_rows_to_osw(&table_rows_base);
            crate::io::osw::write_score_table(osw_path, base_name, &osw_rows)?;
            log::info!("Wrote base TOPAZ scores to OSW table {:?}", base_name);
        }
        let final_table_name = if cfg.xrun.enabled && !cfg.fast_inference {
            cfg.output_table_xrun
                .as_deref()
                .filter(|name| !name.is_empty())
                .unwrap_or(cfg.output_table.as_str())
        } else {
            cfg.output_table.as_str()
        };
        let osw_rows = score_rows_to_osw(&table_rows_final);
        crate::io::osw::write_score_table(osw_path, final_table_name, &osw_rows)?;
        log::info!(
            "Wrote final TOPAZ scores to OSW table {:?}",
            final_table_name
        );
        if cfg.diagnostics.rank1_disagreements {
            let outdir = cfg
                .diagnostics
                .rank1_outdir
                .clone()
                .unwrap_or_else(|| PathBuf::from("rank1_disagreements"));
            let summ = write_rank1_disagreement_tsvs(osw_path, final_table_name, 0.01, &outdir)?;
            log::info!(
                "Rank1 disagreement summary: rows={} cutoff_pstc={:?} cutoff_ms2={:?}",
                summ.rows,
                summ.pstc_cutoff,
                summ.ms2_cutoff
            );
        }
    }

    Ok(InferRunOutput { n_rows: rows.len() })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn prepare_xrun_dataset_from_preprocessed(
    cfg: &XrunSweepConfig,
    device: &Device,
) -> Result<(XrunDataset, Vec<crate::io::xim_parquet::XimDecodeIssue>)> {
    let bundle_path = cfg
        .preprocessed_path
        .as_ref()
        .expect("preprocessed XRUN branch requires a bundle path");
    let base = checkpoint_base(&cfg.checkpoint);
    let meta = read_checkpoint_meta(&base)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = TopazBagRanker::new(vb.pp("topaz"), &meta.model)?;
    load_checkpoint_weights(&base, &mut varmap)?;

    let reader = PreprocessedBundleReader::open(bundle_path)?;
    let manifest = reader.manifest().clone();
    validate_preprocessed_manifest_for_xrun(&manifest, cfg, &meta)?;
    log::info!(
        "Loading preprocessed TOPAZ bundle {:?} for XRUN (rows={} chunks={})",
        bundle_path,
        manifest.row_count,
        manifest.chunks.len()
    );
    let dataset = reader.load_all()?;
    let all_rows = if meta.model.use_heuristic_features {
        align_rows_to_cols(&dataset.rows, &manifest.feature_cols, &meta.feature_cols)
    } else {
        dataset.rows
    };
    let mut rows_aligned = all_rows.clone();
    log_run_id_summary(
        &rows_aligned,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xim_path,
        &cfg.xim_paths,
    );
    if cfg.restrict_osw_to_xic_map {
        rows_aligned = filter_rows_by_xic_map(rows_aligned, &cfg.xic_paths, &cfg.xic_map_path)?;
    }
    let row_index = build_feature_row_index(&all_rows);
    let trace_span = cfg.trace.total_c() * cfg.trace.l;
    let x_trace = gather_row_aligned_tensor_by_feature_id(
        &rows_aligned,
        &row_index,
        &dataset.x_trace,
        trace_span,
    )?;
    let xim_trace_cfg = effective_xim_trace_cfg(&cfg.xim_trace, &meta.model);
    let x_xim = if let (Some(all), Some(xim_cfg)) = (dataset.x_xim.as_ref(), xim_trace_cfg.as_ref())
    {
        Some(gather_row_aligned_tensor_by_feature_id(
            &rows_aligned,
            &row_index,
            all,
            xim_cfg.total_c() * xim_cfg.l,
        )?)
    } else {
        None
    };
    let bag_data = build_xrun_bag_data_from_rows_with_cols_with_aux(
        &model,
        &rows_aligned,
        &x_trace,
        x_xim
            .as_ref()
            .zip(xim_trace_cfg.as_ref())
            .map(|(x, cfg)| (x.as_slice(), cfg)),
        &manifest.feature_cols,
        &meta.feature_cols,
        cfg.trace.total_c(),
        cfg.trace.l,
        cfg.bag_k,
        device,
        cfg.batch_size.max(1),
        meta.preprocess.as_ref(),
    )?;
    let seq = build_xrun_sequences_from_bags(
        &bag_data.bag_pid,
        &bag_data.bag_score,
        &bag_data.bag_hidden,
        bag_data.hidden_dim,
        &bag_data.bag_y,
        cfg.max_runs,
        &cfg.sort_by,
    );
    Ok((
        XrunDataset {
            xseq: seq.xseq,
            mask: seq.mask,
            y: seq.y_prec,
            p: seq.p,
            r: seq.r,
            din: seq.din,
        },
        Vec::new(),
    ))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Train the base TOPAZ model and, optionally, an XRUN calibrator sidecar.
pub fn run_training(cfg: &TrainRunConfig) -> Result<TrainRunOutput> {
    let device = get_device(&cfg.device)?;
    if cfg.preprocessed_path.is_some() {
        return run_training_with_preprocessed(cfg, &device);
    }
    crate::io::xim_parquet::clear_decode_issues();
    let mut xim_decode_issues = Vec::new();
    let init_base: Option<PathBuf> = cfg
        .init_checkpoint
        .as_ref()
        .map(|path| checkpoint_base(path.as_path()));
    let init_meta = if let Some(base) = init_base.as_ref() {
        Some(read_checkpoint_meta(base)?)
    } else {
        None
    };

    let table = read_feature_rows(&cfg.osw_path, &cfg.osw)?;
    let mut rows = filter_training_rows(table.rows, &cfg.filter);
    if rows.is_empty() {
        bail!("no rows after filtering");
    }
    log_run_id_summary(
        &rows,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xim_path,
        &cfg.xim_paths,
    );
    if cfg.restrict_osw_to_xic_map {
        rows = filter_rows_by_xic_map(rows, &cfg.xic_paths, &cfg.xic_map_path)?;
        if rows.is_empty() {
            bail!("no rows after XIC map restriction; check run_id mapping");
        }
    }

    let mut selected_cols = if let Some(meta) = init_meta.as_ref() {
        if cfg.feature_select.mode != FeatureMode::All || cfg.feature_select.cols.is_some() {
            log::warn!("init_checkpoint provided; using feature columns stored in checkpoint");
        }
        meta.feature_cols.clone()
    } else {
        resolve_feature_cols(&table.feature_cols, &cfg.feature_select)
    };
    let mut model_cfg = if let Some(meta) = init_meta.as_ref() {
        if meta.model.l != cfg.trace.l
            || meta.model.ms1_cmax != cfg.trace.ms1_cmax
            || meta.model.ms2_cmax != cfg.trace.ms2_cmax
        {
            bail!(
                "init_checkpoint trace/model mismatch: checkpoint expects l={}, ms1_cmax={}, ms2_cmax={}, current config has l={}, ms1_cmax={}, ms2_cmax={}",
                meta.model.l,
                meta.model.ms1_cmax,
                meta.model.ms2_cmax,
                cfg.trace.l,
                cfg.trace.ms1_cmax,
                cfg.trace.ms2_cmax
            );
        }
        meta.model.clone()
    } else {
        cfg.model.clone()
    };
    if !model_cfg.use_heuristic_features && !selected_cols.is_empty() {
        log::warn!("model.use_heuristic_features=false; ignoring selected feature columns");
        selected_cols.clear();
    }
    if selected_cols.is_empty() {
        model_cfg.use_heuristic_features = false;
        model_cfg.feat_dim = 0;
    } else {
        model_cfg.use_heuristic_features = true;
        if model_cfg.feat_dim != selected_cols.len() {
            if init_meta.is_some() {
                bail!(
                    "init_checkpoint feature dim mismatch: checkpoint expects {}, recovered {} columns",
                    model_cfg.feat_dim,
                    selected_cols.len()
                );
            }
            if model_cfg.feat_dim != 0 {
                log::warn!(
                    "overriding model.feat_dim={} to match selected columns ({})",
                    model_cfg.feat_dim,
                    selected_cols.len()
                );
            }
            model_cfg.feat_dim = selected_cols.len();
        }
    }
    let xim_trace_cfg = effective_xim_trace_cfg(&cfg.xim_trace, &model_cfg);
    if model_cfg.xim.is_some() && xim_trace_cfg.is_none() {
        bail!("model.xim is enabled but no xim_trace configuration could be inferred");
    }
    if model_cfg.xim.is_some()
        && cfg.xim_path.is_none()
        && cfg.xim_paths.is_none()
        && cfg.xim_map_path.is_none()
    {
        bail!("model.xim is enabled but neither xim_path/xim_paths nor xim_map_path was provided");
    }

    let (mut rows_tr, rows_va) = split_rows_by_precursor(&rows, cfg.val_frac, cfg.seed);
    if cfg.train_frac < 1.0 {
        rows_tr =
            subsample_train_rows_by_bag(rows_tr, cfg.train_frac, cfg.train_stratify_run, cfg.seed);
        if rows_tr.is_empty() {
            bail!("no rows after train subsample");
        }
    }
    let xic_cache = SharedXicCache::new(cfg.xic_cache_max_precursors);
    let cache_stats = xic_cache.stats();
    if xic_cache.is_enabled() {
        log::info!(
            "Enabled XIC cache (max_precursors={})",
            cfg.xic_cache_max_precursors
        );
    }
    let disk_cache = match &cfg.xic_cache_dir {
        Some(dir) => Some(XicDiskCache::new(
            dir.clone(),
            cfg.xic_cache_max_bytes,
            cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xic_cache_dir {
        log::info!("Enabled XIC disk cache at {:?}", dir);
    }
    let cache_opt = if xic_cache.is_enabled() || disk_cache.is_some() {
        Some(&xic_cache)
    } else {
        None
    };
    let xim_cache = SharedXimCache::new(cfg.xim_cache_max_features);
    let xim_cache_stats = xim_cache.stats();
    if xim_cache.is_enabled() {
        log::info!(
            "Enabled XIM cache (max_features={})",
            cfg.xim_cache_max_features
        );
    }
    let xim_disk_cache = match &cfg.xim_cache_dir {
        Some(dir) => Some(XimDiskCache::new(
            dir.clone(),
            cfg.xim_cache_max_bytes,
            xim_cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xim_cache_dir {
        log::info!("Enabled XIM disk cache at {:?}", dir);
    }
    let xim_cache_opt = if xim_cache.is_enabled() || xim_disk_cache.is_some() {
        Some(&xim_cache)
    } else {
        None
    };
    let (x_tr, x_va) = build_traces_for_train_val(
        &rows_tr,
        &rows_va,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
        cache_opt,
        disk_cache.as_ref(),
    )?;
    let (x_tr_xim, x_va_xim) = build_xim_for_train_val(
        &rows_tr,
        &rows_va,
        &cfg.xim_path,
        &cfg.xim_paths,
        &cfg.xim_map_path,
        &xim_trace_cfg,
        &cfg.xim_fetch,
        xim_cache_opt,
        xim_disk_cache.as_ref(),
    )?;
    drain_xim_decode_issues(&mut xim_decode_issues);
    log_xic_cache_stats("train", &cache_stats);
    log_xim_cache_stats("train", &xim_cache_stats);

    let apply_trace_filter = cfg.restrict_osw_to_xic_map
        && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_none();
    if cfg.restrict_osw_to_xic_map && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_some()
    {
        log::info!(
            "XIC map/path list provided; skipping trace-based restriction (run_id filter only)"
        );
    }
    let (rows_tr, x_tr, x_tr_xim) = if apply_trace_filter {
        filter_rows_by_trace_with_aux(
            rows_tr,
            x_tr,
            x_tr_xim,
            cfg.trace.total_c(),
            cfg.trace.l,
            xim_trace_cfg.as_ref().map(|c| c.total_c()),
            xim_trace_cfg.as_ref().map(|c| c.l),
        )
    } else {
        (rows_tr, x_tr, x_tr_xim)
    };
    if apply_trace_filter && rows_tr.is_empty() {
        bail!(
            "all training rows were dropped after XIC restriction; check run_id match and xic_path"
        );
    }
    let (rows_va, x_va, x_va_xim) = if apply_trace_filter {
        filter_rows_by_trace_with_aux(
            rows_va,
            x_va,
            x_va_xim,
            cfg.trace.total_c(),
            cfg.trace.l,
            xim_trace_cfg.as_ref().map(|c| c.total_c()),
            xim_trace_cfg.as_ref().map(|c| c.l),
        )
    } else {
        (rows_va, x_va, x_va_xim)
    };

    let pre = if let Some(meta) = init_meta.as_ref() {
        if model_cfg.use_heuristic_features {
            meta.preprocess.clone()
        } else {
            None
        }
    } else if model_cfg.use_heuristic_features {
        Some(fit_preprocessor_from_rows_with_cols(
            &rows_tr,
            &table.feature_cols,
            &selected_cols,
        ))
    } else {
        None
    };

    if cfg.diagnostics.trace_summary {
        let sum_tr = trace_summary(
            &x_tr,
            rows_tr.len(),
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.trace.ms1_cmax,
            cfg.trace.ms2_cmax,
        );
        print_trace_summary(&sum_tr, "train");
        warn_if_missing_ms1(&sum_tr, "train");
        if !rows_va.is_empty() {
            let sum_va = trace_summary(
                &x_va,
                rows_va.len(),
                cfg.trace.total_c(),
                cfg.trace.l,
                cfg.trace.ms1_cmax,
                cfg.trace.ms2_cmax,
            );
            print_trace_summary(&sum_va, "val");
            warn_if_missing_ms1(&sum_va, "val");
        }
    }

    let x_feat = if model_cfg.use_heuristic_features {
        rows_to_feature_matrix_with_cols(
            &rows_tr,
            &table.feature_cols,
            &selected_cols,
            pre.as_ref(),
        )
    } else {
        Vec::new()
    };
    let x_feat_va = if model_cfg.use_heuristic_features && !rows_va.is_empty() {
        rows_to_feature_matrix_with_cols(
            &rows_va,
            &table.feature_cols,
            &selected_cols,
            pre.as_ref(),
        )
    } else {
        Vec::new()
    };

    let y_rows: Vec<u8> = rows_tr
        .iter()
        .map(|r| if r.is_decoy { 1 } else { 0 })
        .collect();
    let pid_rows: Vec<String> = rows_tr.iter().map(|r| r.group_id.clone()).collect();

    let bags = crate::building_blocks::bagging::make_bags_with_traces(
        &x_feat,
        rows_tr.len(),
        model_cfg.feat_dim,
        &x_tr,
        cfg.trace.total_c(),
        cfg.trace.l,
        &y_rows,
        &pid_rows,
        cfg.bag_k,
    );
    let (mut n_pos, mut n_neg) = (0usize, 0usize);
    for &y in &bags.y_bag {
        if y > 0.5 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    let pos_weight = if n_pos > 0 {
        n_neg as f32 / n_pos as f32
    } else {
        1.0
    };
    let batches =
        if let (Some(x_tr_xim), Some(xim_cfg)) = (x_tr_xim.as_ref(), xim_trace_cfg.as_ref()) {
            let aux_bags = crate::building_blocks::bagging::make_bags_with_traces(
                &x_feat,
                rows_tr.len(),
                model_cfg.feat_dim,
                x_tr_xim,
                xim_cfg.total_c(),
                xim_cfg.l,
                &y_rows,
                &pid_rows,
                cfg.bag_k,
            );
            bags_to_train_batches_with_aux(
                bags,
                Some((aux_bags.t_bag, xim_cfg.total_c(), xim_cfg.l)),
                &device,
                cfg.batch_size,
            )?
        } else {
            bags_to_train_batches(bags, &device, cfg.batch_size)?
        };

    let mut trainer = Trainer::new(cfg.train.clone(), &model_cfg, &device)?;
    if let Some(base) = init_base.as_ref() {
        let (_meta, report) = load_checkpoint_partial(base, &mut trainer.varmap)?;
        if report.loaded == 0 {
            bail!(
                "init_checkpoint {:?} did not provide any compatible tensors for the current model",
                base
            );
        }
        log::info!(
            "Loaded initialization checkpoint from {:?} (loaded={}, missing={}, shape_mismatch={}, set_errors={}, extra_in_checkpoint={})",
            base,
            report.loaded,
            report.missing_in_checkpoint.len(),
            report.shape_mismatch.len(),
            report.set_errors.len(),
            report.extra_in_checkpoint.len()
        );
        if !report.missing_in_checkpoint.is_empty() {
            log::warn!(
                "init_checkpoint tensors missing in checkpoint: {}",
                summarize_names(&report.missing_in_checkpoint, 8)
            );
        }
        if !report.shape_mismatch.is_empty() {
            log::warn!(
                "init_checkpoint tensors skipped due to shape mismatch: {}",
                summarize_names(&report.shape_mismatch, 8)
            );
        }
        if !report.set_errors.is_empty() {
            log::warn!(
                "init_checkpoint tensors skipped due to assignment errors: {}",
                summarize_names(&report.set_errors, 8)
            );
        }
    }
    trainer.set_pos_weight(pos_weight);
    trainer.set_shuffle_seed(cfg.seed);
    log::info!(
        "Using pos_weight={:.4} (n_pos={}, n_neg={})",
        pos_weight,
        n_pos,
        n_neg
    );
    let val_batches = if !rows_va.is_empty() {
        let y_rows_va: Vec<u8> = rows_va
            .iter()
            .map(|r| if r.is_decoy { 1 } else { 0 })
            .collect();
        let pid_rows_va: Vec<String> = rows_va.iter().map(|r| r.group_id.clone()).collect();
        let bags_va = crate::building_blocks::bagging::make_bags_with_traces(
            &x_feat_va,
            rows_va.len(),
            model_cfg.feat_dim,
            &x_va,
            cfg.trace.total_c(),
            cfg.trace.l,
            &y_rows_va,
            &pid_rows_va,
            cfg.bag_k,
        );
        if let (Some(x_va_xim), Some(xim_cfg)) = (x_va_xim.as_ref(), xim_trace_cfg.as_ref()) {
            let aux_bags_va = crate::building_blocks::bagging::make_bags_with_traces(
                &x_feat_va,
                rows_va.len(),
                model_cfg.feat_dim,
                x_va_xim,
                xim_cfg.total_c(),
                xim_cfg.l,
                &y_rows_va,
                &pid_rows_va,
                cfg.bag_k,
            );
            bags_to_train_batches_with_aux(
                bags_va,
                Some((aux_bags_va.t_bag, xim_cfg.total_c(), xim_cfg.l)),
                &device,
                cfg.batch_size,
            )?
        } else {
            bags_to_train_batches(bags_va, &device, cfg.batch_size)?
        }
    } else {
        Vec::new()
    };
    let _history = trainer.train_epochs_early_stop(&batches, &val_batches, cfg.max_epochs, None)?;

    if !rows_va.is_empty() {
        let out = score_bags_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows_va,
            &x_va,
            x_va_xim
                .as_ref()
                .zip(xim_trace_cfg.as_ref())
                .map(|(x, cfg)| (x.as_slice(), cfg)),
            &table.feature_cols,
            &selected_cols,
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.bag_k,
            &device,
            cfg.batch_size,
            pre.as_ref(),
        )?;
        let summ = tdc_summary(&out.bag_score, &out.is_decoy, 0.01);
        log::info!(
            "VAL TDC summary @q=0.01: cutoff={:.4} targets={} decoys={}",
            summ.cutoff,
            summ.n_targets,
            summ.n_decoys
        );
    }

    if cfg.diagnostics.save_head_embeddings {
        let outdir = cfg
            .diagnostics
            .head_embeddings_outdir
            .clone()
            .unwrap_or_else(|| PathBuf::from("head_embeddings"));
        std::fs::create_dir_all(&outdir)?;
        let mut rows_all = rows_tr.clone();
        if !rows_va.is_empty() {
            rows_all.extend(rows_va.iter().cloned());
        }
        let mut x_all = x_tr.clone();
        if !x_va.is_empty() {
            x_all.extend_from_slice(&x_va);
        }
        let mut x_all_xim = None;
        if let Some(x_tr_xim) = x_tr_xim.as_ref() {
            let mut buf = x_tr_xim.clone();
            if let Some(x_va_xim) = x_va_xim.as_ref() {
                buf.extend_from_slice(x_va_xim);
            }
            x_all_xim = Some(buf);
        }
        let out = score_bags_with_heads_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows_all,
            &x_all,
            x_all_xim
                .as_ref()
                .zip(xim_trace_cfg.as_ref())
                .map(|(x, cfg)| (x.as_slice(), cfg)),
            &table.feature_cols,
            &selected_cols,
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.bag_k,
            &device,
            cfg.batch_size,
            pre.as_ref(),
        )?;
        let out_path = outdir.join("head_embeddings.tsv");
        write_head_embeddings_tsv(&out_path, &out)?;
    }

    let meta = CheckpointMeta {
        model: model_cfg,
        train: Some(cfg.train.clone()),
        trace: Some(cfg.trace.clone()),
        feature_cols: selected_cols,
        preprocess: pre,
        version: 1,
    };
    save_checkpoint(&cfg.output_prefix, &trainer.varmap, &meta)?;

    if cfg.xrun.enabled {
        let mut rows_all = rows_tr.clone();
        rows_all.extend(rows_va.iter().cloned());
        let mut x_all = x_tr.clone();
        x_all.extend_from_slice(&x_va);
        let mut x_all_xim = None;
        if let Some(x_tr_xim) = x_tr_xim.as_ref() {
            let mut buf = x_tr_xim.clone();
            if let Some(x_va_xim) = x_va_xim.as_ref() {
                buf.extend_from_slice(x_va_xim);
            }
            x_all_xim = Some(buf);
        }
        let bag_data = build_xrun_bag_data_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows_all,
            &x_all,
            x_all_xim
                .as_ref()
                .zip(xim_trace_cfg.as_ref())
                .map(|(x, cfg)| (x.as_slice(), cfg)),
            &table.feature_cols,
            &meta.feature_cols,
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.bag_k,
            &device,
            cfg.xrun.batch_size.max(1),
            meta.preprocess.as_ref(),
        )?;
        let seq = build_xrun_sequences_from_bags(
            &bag_data.bag_pid,
            &bag_data.bag_score,
            &bag_data.bag_hidden,
            bag_data.hidden_dim,
            &bag_data.bag_y,
            cfg.xrun.max_runs,
            &cfg.xrun.sort_by,
        );
        let ds = XrunDataset {
            xseq: seq.xseq,
            mask: seq.mask,
            y: seq.y_prec,
            p: seq.p,
            r: seq.r,
            din: seq.din,
        };
        let (tr_ds, va_ds) = split_train_val(&ds, cfg.val_frac, cfg.seed);
        if tr_ds.p == 0 || va_ds.p == 0 {
            log::warn!(
                "Skipping XRUN training because train/val sequence split is empty (train_p={}, val_p={})",
                tr_ds.p,
                va_ds.p
            );
        } else {
            let mut xrun_trainer = XrunTrainer::new(cfg.xrun.train.clone(), ds.din, &device)?;
            let xrun_meta = xrun_trainer.train(&tr_ds, &va_ds, &device)?;
            let xrun_ckpt_meta = XrunCheckpointMeta {
                train: cfg.xrun.train.clone(),
                predict: XrunPredictConfig {
                    max_runs: cfg.xrun.max_runs,
                    sort_by: cfg.xrun.sort_by.clone(),
                    batch_size: cfg.xrun.batch_size.max(1),
                },
                in_dim: ds.din,
                best_val: Some(xrun_meta.best_val),
                version: 1,
            };
            save_xrun_checkpoint(&cfg.output_prefix, &xrun_trainer.varmap, &xrun_ckpt_meta)?;
            log::info!(
                "Saved XRUN calibrator alongside checkpoint (best_val={:.4})",
                xrun_meta.best_val
            );
        }
    }

    report_xim_decode_issues(
        "train",
        &xim_decode_issues,
        &diagnostic_tsv_path(&cfg.output_prefix, "xim_skipped"),
    )?;

    Ok(TrainRunOutput {
        checkpoint_prefix: cfg.output_prefix.clone(),
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Run TOPAZ inference and optionally apply XRUN calibration before writeback.
///
/// TSV output always contains the final score stream:
/// - base TOPAZ scores when `xrun.enabled == false`
/// - XRUN-calibrated scores when `xrun.enabled == true`
///
/// When OSW writeback is enabled, `output_table_base` can be used to persist
/// the uncalibrated base table in addition to the final table.
pub fn run_inference(cfg: &InferRunConfig) -> Result<InferRunOutput> {
    let device = get_device(&cfg.device)?;
    if cfg.preprocessed_path.is_some() {
        return run_inference_with_preprocessed(cfg, &device);
    }
    crate::io::xim_parquet::clear_decode_issues();
    let mut xim_decode_issues = Vec::new();

    let base = checkpoint_base(&cfg.checkpoint);
    let meta = read_checkpoint_meta(&base)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = TopazBagRanker::new(vb.pp("topaz"), &meta.model)?;
    load_checkpoint_weights(&base, &mut varmap)?;

    let table = read_feature_rows(&cfg.osw_path, &cfg.osw)?;
    let mut rows = table.rows;
    if rows.is_empty() {
        bail!("no rows in OSW");
    }
    log_run_id_summary(
        &rows,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xim_path,
        &cfg.xim_paths,
    );
    if cfg.restrict_osw_to_xic_map {
        rows = filter_rows_by_xic_map(rows, &cfg.xic_paths, &cfg.xic_map_path)?;
        if rows.is_empty() {
            bail!("no rows after XIC map restriction; check run_id mapping");
        }
    }

    let xic_cache = SharedXicCache::new(cfg.xic_cache_max_precursors);
    let cache_stats = xic_cache.stats();
    if xic_cache.is_enabled() {
        log::info!(
            "Enabled XIC cache (max_precursors={})",
            cfg.xic_cache_max_precursors
        );
    }
    let disk_cache = match &cfg.xic_cache_dir {
        Some(dir) => Some(XicDiskCache::new(
            dir.clone(),
            cfg.xic_cache_max_bytes,
            cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xic_cache_dir {
        log::info!("Enabled XIC disk cache at {:?}", dir);
    }
    let cache_opt = if xic_cache.is_enabled() || disk_cache.is_some() {
        Some(&xic_cache)
    } else {
        None
    };
    let target_cols = meta.feature_cols.clone();
    let feat_dim = if meta.model.use_heuristic_features {
        meta.model.feat_dim
    } else {
        0
    };
    let xim_trace_cfg = effective_xim_trace_cfg(&cfg.xim_trace, &meta.model);
    if meta.model.xim.is_some() && xim_trace_cfg.is_none() {
        bail!("checkpoint enables model.xim but no xim_trace configuration could be inferred");
    }
    if meta.model.xim.is_some()
        && cfg.xim_path.is_none()
        && cfg.xim_paths.is_none()
        && cfg.xim_map_path.is_none()
    {
        bail!(
            "checkpoint enables model.xim but neither xim_path/xim_paths nor xim_map_path was provided"
        );
    }
    let xim_cache = SharedXimCache::new(cfg.xim_cache_max_features);
    let xim_cache_stats = xim_cache.stats();
    if xim_cache.is_enabled() {
        log::info!(
            "Enabled XIM cache (max_features={})",
            cfg.xim_cache_max_features
        );
    }
    let xim_disk_cache = match &cfg.xim_cache_dir {
        Some(dir) => Some(XimDiskCache::new(
            dir.clone(),
            cfg.xim_cache_max_bytes,
            xim_cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xim_cache_dir {
        log::info!("Enabled XIM disk cache at {:?}", dir);
    }
    let xim_cache_opt = if xim_cache.is_enabled() || xim_disk_cache.is_some() {
        Some(&xim_cache)
    } else {
        None
    };
    let apply_trace_filter = cfg.restrict_osw_to_xic_map
        && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_none();
    if cfg.restrict_osw_to_xic_map && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_some()
    {
        log::info!(
            "XIC map/path list provided; skipping trace-based restriction (run_id filter only)"
        );
    }
    let chunk_size = if cfg.trace_chunk_size == 0 {
        rows.len().max(1)
    } else {
        cfg.trace_chunk_size.max(1)
    };
    if cfg.prefetch_traces_once && cfg.stream_inference {
        log::info!(
            "prefetch_traces_once=true overrides stream_inference=true; using full-dataset prefetch"
        );
    }
    let mut runtime_stats = InferenceRuntimeStats {
        mode: if cfg.prefetch_traces_once {
            "prefetch"
        } else if cfg.stream_inference {
            "streaming"
        } else {
            "chunked"
        },
        chunk_size,
        work_unit_size: None,
        producer_workers: None,
        queue_depth: None,
        decode_time: Duration::ZERO,
        score_time: Duration::ZERO,
    };

    let prefetched = if cfg.prefetch_traces_once {
        let prefetch_started = Instant::now();
        let prefetched = prefetch_modalities_for_inference(
            &mut rows,
            cfg,
            &xim_trace_cfg,
            cache_opt,
            disk_cache.as_ref(),
            xim_cache_opt,
            xim_disk_cache.as_ref(),
            apply_trace_filter,
            &mut xim_decode_issues,
        )?;
        runtime_stats.decode_time += prefetch_started.elapsed();
        Some(prefetched)
    } else {
        None
    };

    let mut scores: Vec<f32> =
        if apply_trace_filter && prefetched.is_none() && !cfg.stream_inference {
            Vec::new()
        } else {
            vec![0f32; rows.len()]
        };
    let mut rows_scored: Vec<FeatureRow> = Vec::new();

    let mut sum_n = 0usize;
    let mut sum_ms1 = 0usize;
    let mut sum_ms2 = 0usize;

    let mut offset = 0usize;
    if let Some((x_trace_all, x_xim_all)) = prefetched.as_ref() {
        let mut progress = InferenceProgressLogger::new("prefetch", rows.len(), chunk_size);
        let trace_row_span = cfg.trace.total_c() * cfg.trace.l;
        let xim_row_span = xim_trace_cfg
            .as_ref()
            .map(|xim_cfg| xim_cfg.total_c() * xim_cfg.l);
        for start in (0..rows.len()).step_by(chunk_size) {
            let end = (start + chunk_size).min(rows.len());
            let row_slice = &rows[start..end];
            let n_chunk = row_slice.len();
            if n_chunk == 0 {
                continue;
            }
            let x_trace = &x_trace_all[start * trace_row_span..end * trace_row_span];
            let x_xim = x_xim_all
                .as_ref()
                .and_then(|all| xim_row_span.map(|span| &all[start * span..end * span]));

            if cfg.diagnostics.trace_summary {
                let sum = trace_summary(
                    x_trace,
                    n_chunk,
                    cfg.trace.total_c(),
                    cfg.trace.l,
                    cfg.trace.ms1_cmax,
                    cfg.trace.ms2_cmax,
                );
                sum_n += sum.n;
                sum_ms1 += sum.ms1_nonzero_rows;
                sum_ms2 += sum.ms2_nonzero_rows;
            }

            let score_started = Instant::now();
            let scores_chunk = score_inference_chunk(
                &model,
                row_slice,
                x_trace,
                x_xim,
                &table.feature_cols,
                &target_cols,
                &cfg.trace,
                xim_trace_cfg.as_ref(),
                meta.model.use_heuristic_features,
                feat_dim,
                &device,
                cfg.batch_size.max(1),
                meta.preprocess.as_ref(),
            )?;
            runtime_stats.score_time += score_started.elapsed();

            scores[offset..offset + n_chunk].copy_from_slice(&scores_chunk);
            offset += n_chunk;
            progress.maybe_log(end, start / chunk_size + 1);
        }
    } else if cfg.stream_inference {
        let streamed = score_rows_streaming_inference(
            &model,
            &rows,
            cfg,
            &xim_trace_cfg,
            &table.feature_cols,
            &target_cols,
            meta.model.use_heuristic_features,
            feat_dim,
            &device,
            meta.preprocess.as_ref(),
            cache_opt,
            disk_cache.as_ref(),
            xim_cache_opt,
            xim_disk_cache.as_ref(),
            apply_trace_filter,
        )?;
        runtime_stats.decode_time += streamed.decode_time;
        runtime_stats.score_time += streamed.score_time;
        runtime_stats.work_unit_size = Some(streamed.work_unit_size);
        runtime_stats.producer_workers = Some(streamed.producer_workers);
        runtime_stats.queue_depth = Some(streamed.queue_depth);
        scores = streamed.scores;
        if let Some(filtered) = streamed.rows_filtered {
            rows_scored = filtered;
        }
        if cfg.diagnostics.trace_summary {
            sum_n = streamed.summary.n;
            sum_ms1 = streamed.summary.ms1_nonzero_rows;
            sum_ms2 = streamed.summary.ms2_nonzero_rows;
        }
        xim_decode_issues.extend(streamed.xim_decode_issues);
    } else {
        let mut progress = InferenceProgressLogger::new("chunked", rows.len(), chunk_size);
        let mut processed_rows = 0usize;
        let mut processed_chunks = 0usize;
        for chunk in rows.chunks(chunk_size) {
            let decode_started = Instant::now();
            let x_trace = build_traces_for_rows(
                chunk,
                &cfg.xic_path,
                &cfg.xic_paths,
                &cfg.xic_map_path,
                &cfg.trace,
                &cfg.fetch,
                cache_opt,
                disk_cache.as_ref(),
            )?;
            let x_xim = build_xim_for_rows(
                chunk,
                &cfg.xim_path,
                &cfg.xim_paths,
                &cfg.xim_map_path,
                &xim_trace_cfg,
                &cfg.xim_fetch,
                xim_cache_opt,
                xim_disk_cache.as_ref(),
            )?;
            drain_xim_decode_issues(&mut xim_decode_issues);
            runtime_stats.decode_time += decode_started.elapsed();

            let (chunk_rows, x_trace, x_xim) = if apply_trace_filter {
                let (rows_f, x_tr_f, x_xim_f) = filter_rows_by_trace_with_aux(
                    chunk.to_vec(),
                    x_trace,
                    x_xim,
                    cfg.trace.total_c(),
                    cfg.trace.l,
                    xim_trace_cfg.as_ref().map(|c| c.total_c()),
                    xim_trace_cfg.as_ref().map(|c| c.l),
                );
                if rows_f.is_empty() {
                    continue;
                }
                (rows_f, x_tr_f, x_xim_f)
            } else {
                (Vec::new(), x_trace, x_xim)
            };

            let row_slice: &[FeatureRow] = if apply_trace_filter {
                &chunk_rows
            } else {
                chunk
            };
            let n_chunk = row_slice.len();
            processed_rows += chunk.len();
            processed_chunks += 1;
            if n_chunk == 0 {
                progress.maybe_log(processed_rows, processed_chunks);
                continue;
            }

            if cfg.diagnostics.trace_summary {
                let sum = trace_summary(
                    &x_trace,
                    n_chunk,
                    cfg.trace.total_c(),
                    cfg.trace.l,
                    cfg.trace.ms1_cmax,
                    cfg.trace.ms2_cmax,
                );
                sum_n += sum.n;
                sum_ms1 += sum.ms1_nonzero_rows;
                sum_ms2 += sum.ms2_nonzero_rows;
            }

            let score_started = Instant::now();
            let scores_chunk = score_inference_chunk(
                &model,
                row_slice,
                &x_trace,
                x_xim.as_deref(),
                &table.feature_cols,
                &target_cols,
                &cfg.trace,
                xim_trace_cfg.as_ref(),
                meta.model.use_heuristic_features,
                feat_dim,
                &device,
                cfg.batch_size.max(1),
                meta.preprocess.as_ref(),
            )?;
            runtime_stats.score_time += score_started.elapsed();

            if apply_trace_filter {
                rows_scored.extend(chunk_rows);
                scores.extend(scores_chunk);
            } else {
                scores[offset..offset + n_chunk].copy_from_slice(&scores_chunk);
                offset += n_chunk;
            }
            progress.maybe_log(processed_rows, processed_chunks);
        }
    }

    log_xic_cache_stats("infer", &cache_stats);
    log_xim_cache_stats("infer", &xim_cache_stats);
    if apply_trace_filter && prefetched.is_none() {
        rows = rows_scored;
    }
    log_inference_runtime_stats(&runtime_stats, rows.len());
    if cfg.diagnostics.trace_summary {
        let sum = crate::infer::diagnostics::TraceSummary {
            n: sum_n,
            l: cfg.trace.l,
            ms1_cmax: cfg.trace.ms1_cmax,
            ms2_cmax: cfg.trace.ms2_cmax,
            ms1_nonzero_rows: sum_ms1,
            ms2_nonzero_rows: sum_ms2,
        };
        print_trace_summary(&sum, "infer");
        warn_if_missing_ms1(&sum, "infer");
    }
    // Release any full-dataset prefetched trace buffers before XRUN, writeback,
    // and report generation. Those later stages rebuild only the smaller
    // winner-row subsets they actually need.
    drop(prefetched);

    let skip_xrun = cfg.fast_inference && cfg.xrun.enabled;
    let skip_head_embeddings = cfg.fast_inference && cfg.diagnostics.save_head_embeddings;
    if skip_xrun {
        log::info!("fast_inference=true: skipping XRUN application during main inference run");
    }
    if skip_head_embeddings {
        log::info!("fast_inference=true: skipping head-embedding export during main inference run");
    }

    let base_scores = scores.clone();
    let xrun_applied = if cfg.xrun.enabled && !cfg.fast_inference {
        let (_xrun_varmap, xrun_model, xrun_meta) = load_xrun_calibrator(&base, &device)?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "XRUN enabled but no XRUN checkpoint sidecar found for {:?}",
                    base
                )
            })?;
        let applied = apply_xrun_to_row_scores(
            &rows,
            &base_scores,
            &model,
            &table.feature_cols,
            &target_cols,
            &cfg.trace,
            xim_trace_cfg.as_ref(),
            &cfg.fetch,
            &cfg.xim_fetch,
            &cfg.xic_path,
            &cfg.xic_paths,
            &cfg.xic_map_path,
            &cfg.xim_path,
            &cfg.xim_paths,
            &cfg.xim_map_path,
            cache_opt,
            disk_cache.as_ref(),
            xim_cache_opt,
            xim_disk_cache.as_ref(),
            &cfg.xrun,
            &xrun_model,
            &xrun_meta,
            &device,
            cfg.batch_size.max(1),
            meta.preprocess.as_ref(),
        )?;
        drain_xim_decode_issues(&mut xim_decode_issues);
        log::info!(
            "Applied XRUN calibration to {} bags using {:?} pooling",
            applied.bag_pid.len(),
            xrun_meta.train.pool
        );
        scores = applied.row_scores.clone();
        Some(applied)
    } else {
        None
    };

    let table_rows_base = build_score_table_from_rows(&rows, &base_scores, cfg.pep_bins);
    let table_rows_final = build_score_table_from_rows(&rows, &scores, cfg.pep_bins);
    crate::infer::write_score_tsv(&cfg.output_tsv, &table_rows_final)?;

    if cfg.diagnostics.save_head_embeddings && !cfg.fast_inference {
        let outdir = cfg
            .diagnostics
            .head_embeddings_outdir
            .clone()
            .unwrap_or_else(|| PathBuf::from("head_embeddings"));
        if cfg.trace_chunk_size > 0 {
            std::fs::create_dir_all(&outdir)?;
            log::info!("Computing head embeddings in a chunk-safe second pass...");
            let (winner_rows, bag_pid, bag_score, bag_is_decoy, bag_y) =
                select_bag_winners(&rows, &scores);
            if winner_rows.is_empty() {
                log::warn!("No winner rows found for head embeddings.");
            } else {
                let x_trace = build_traces_for_rows(
                    &winner_rows,
                    &cfg.xic_path,
                    &cfg.xic_paths,
                    &cfg.xic_map_path,
                    &cfg.trace,
                    &cfg.fetch,
                    cache_opt,
                    disk_cache.as_ref(),
                )?;
                let x_xim = build_xim_for_rows(
                    &winner_rows,
                    &cfg.xim_path,
                    &cfg.xim_paths,
                    &cfg.xim_map_path,
                    &xim_trace_cfg,
                    &cfg.xim_fetch,
                    xim_cache_opt,
                    xim_disk_cache.as_ref(),
                )?;
                drain_xim_decode_issues(&mut xim_decode_issues);
                let mut out = score_bags_with_heads_from_rows_with_cols_with_aux(
                    &model,
                    &winner_rows,
                    &x_trace,
                    x_xim
                        .as_ref()
                        .zip(xim_trace_cfg.as_ref())
                        .map(|(x, cfg)| (x.as_slice(), cfg)),
                    &table.feature_cols,
                    &meta.feature_cols,
                    cfg.trace.total_c(),
                    cfg.trace.l,
                    1,
                    &device,
                    cfg.batch_size,
                    meta.preprocess.as_ref(),
                )?;
                out.bag_pid = bag_pid;
                out.bag_score = bag_score;
                out.is_decoy = bag_is_decoy;
                out.bag_y = bag_y;
                let out_path = outdir.join("head_embeddings.tsv");
                write_head_embeddings_tsv(&out_path, &out)?;
            }
        } else {
            std::fs::create_dir_all(&outdir)?;
            let x_trace = build_traces_for_rows(
                &rows,
                &cfg.xic_path,
                &cfg.xic_paths,
                &cfg.xic_map_path,
                &cfg.trace,
                &cfg.fetch,
                cache_opt,
                disk_cache.as_ref(),
            )?;
            let x_xim = build_xim_for_rows(
                &rows,
                &cfg.xim_path,
                &cfg.xim_paths,
                &cfg.xim_map_path,
                &xim_trace_cfg,
                &cfg.xim_fetch,
                xim_cache_opt,
                xim_disk_cache.as_ref(),
            )?;
            drain_xim_decode_issues(&mut xim_decode_issues);
            let mut out = score_bags_with_heads_from_rows_with_cols_with_aux(
                &model,
                &rows,
                &x_trace,
                x_xim
                    .as_ref()
                    .zip(xim_trace_cfg.as_ref())
                    .map(|(x, cfg)| (x.as_slice(), cfg)),
                &table.feature_cols,
                &meta.feature_cols,
                cfg.trace.total_c(),
                cfg.trace.l,
                cfg.bag_k,
                &device,
                cfg.batch_size,
                meta.preprocess.as_ref(),
            )?;
            if let Some(applied) = xrun_applied.as_ref() {
                let mut bag_score_map = HashMap::new();
                for (pid, score) in applied.bag_pid.iter().zip(applied.bag_score.iter()) {
                    bag_score_map.insert(pid.clone(), *score);
                }
                for (i, pid) in out.bag_pid.iter().enumerate() {
                    if let Some(score) = bag_score_map.get(pid) {
                        out.bag_score[i] = *score;
                    }
                }
            }
            let out_path = outdir.join("head_embeddings.tsv");
            write_head_embeddings_tsv(&out_path, &out)?;
        }
    }

    if let Some(osw_path) = &cfg.output_osw {
        #[cfg(feature = "io-sqlite")]
        {
            crate::io::osw::prepare_output_osw(&cfg.osw_path, osw_path)?;
            if let Some(base_name) = cfg
                .output_table_base
                .as_ref()
                .filter(|name| !name.is_empty())
            {
                let osw_rows = score_rows_to_osw(&table_rows_base);
                crate::io::osw::write_score_table(osw_path, base_name, &osw_rows)?;
                log::info!("Wrote base TOPAZ scores to OSW table {:?}", base_name);
            }

            let final_table_name = if cfg.xrun.enabled && !cfg.fast_inference {
                cfg.output_table_xrun
                    .as_deref()
                    .filter(|name| !name.is_empty())
                    .unwrap_or(cfg.output_table.as_str())
            } else {
                cfg.output_table.as_str()
            };
            let osw_rows = score_rows_to_osw(&table_rows_final);
            crate::io::osw::write_score_table(osw_path, final_table_name, &osw_rows)?;
            log::info!(
                "Wrote final TOPAZ scores to OSW table {:?}",
                final_table_name
            );

            if cfg.diagnostics.rank1_disagreements {
                let outdir = cfg
                    .diagnostics
                    .rank1_outdir
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("rank1_disagreements"));
                let summ =
                    write_rank1_disagreement_tsvs(osw_path, final_table_name, 0.01, &outdir)?;
                log::info!(
                    "Rank1 disagreement summary: rows={} cutoff_pstc={:?} cutoff_ms2={:?}",
                    summ.rows,
                    summ.pstc_cutoff,
                    summ.ms2_cutoff
                );
            }
        }
    }

    report_xim_decode_issues(
        "infer",
        &xim_decode_issues,
        &diagnostic_tsv_path(&cfg.output_tsv, "xim_skipped"),
    )?;

    Ok(InferRunOutput { n_rows: rows.len() })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn write_xrun_summary_tsv(path: &Path, rows: &[XrunSweepRow]) -> Result<()> {
    let mut text = String::new();
    text.push_str("pool\ttau\tbest_val\n");
    for r in rows {
        text.push_str(&format!("{}\t{}\t{}\n", r.pool, r.tau, r.best_val));
    }
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn prepare_xrun_dataset_from_checkpoint(
    cfg: &XrunSweepConfig,
    device: &Device,
) -> Result<(XrunDataset, Vec<crate::io::xim_parquet::XimDecodeIssue>)> {
    if cfg.preprocessed_path.is_some() {
        return prepare_xrun_dataset_from_preprocessed(cfg, device);
    }
    crate::io::xim_parquet::clear_decode_issues();
    let mut xim_decode_issues = Vec::new();

    let base = checkpoint_base(&cfg.checkpoint);
    let meta = read_checkpoint_meta(&base)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = TopazBagRanker::new(vb.pp("topaz"), &meta.model)?;
    load_checkpoint_weights(&base, &mut varmap)?;

    let table = read_feature_rows(&cfg.osw_path, &cfg.osw)?;
    let mut rows_aligned = if meta.model.use_heuristic_features {
        align_rows_to_cols(&table.rows, &table.feature_cols, &meta.feature_cols)
    } else {
        table.rows
    };
    log_run_id_summary(
        &rows_aligned,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xim_path,
        &cfg.xim_paths,
    );
    if cfg.restrict_osw_to_xic_map {
        rows_aligned = filter_rows_by_xic_map(rows_aligned, &cfg.xic_paths, &cfg.xic_map_path)?;
        if rows_aligned.is_empty() {
            bail!("no rows after XIC map restriction; check run_id mapping");
        }
    }

    let xic_cache = SharedXicCache::new(cfg.xic_cache_max_precursors);
    let cache_stats = xic_cache.stats();
    if xic_cache.is_enabled() {
        log::info!(
            "Enabled XIC cache (max_precursors={})",
            cfg.xic_cache_max_precursors
        );
    }
    let disk_cache = match &cfg.xic_cache_dir {
        Some(dir) => Some(XicDiskCache::new(
            dir.clone(),
            cfg.xic_cache_max_bytes,
            cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xic_cache_dir {
        log::info!("Enabled XIC disk cache at {:?}", dir);
    }
    let cache_opt = if xic_cache.is_enabled() || disk_cache.is_some() {
        Some(&xic_cache)
    } else {
        None
    };
    let xim_trace_cfg = effective_xim_trace_cfg(&cfg.xim_trace, &meta.model);
    if meta.model.xim.is_some() && xim_trace_cfg.is_none() {
        bail!("checkpoint enables model.xim but no xim_trace configuration could be inferred");
    }
    if meta.model.xim.is_some()
        && cfg.xim_path.is_none()
        && cfg.xim_paths.is_none()
        && cfg.xim_map_path.is_none()
    {
        bail!(
            "checkpoint enables model.xim but neither xim_path/xim_paths nor xim_map_path was provided"
        );
    }
    let xim_cache = SharedXimCache::new(cfg.xim_cache_max_features);
    let xim_cache_stats = xim_cache.stats();
    if xim_cache.is_enabled() {
        log::info!(
            "Enabled XIM cache (max_features={})",
            cfg.xim_cache_max_features
        );
    }
    let xim_disk_cache = match &cfg.xim_cache_dir {
        Some(dir) => Some(XimDiskCache::new(
            dir.clone(),
            cfg.xim_cache_max_bytes,
            xim_cache_stats.clone(),
        )?),
        None => None,
    };
    if let Some(dir) = &cfg.xim_cache_dir {
        log::info!("Enabled XIM disk cache at {:?}", dir);
    }
    let xim_cache_opt = if xim_cache.is_enabled() || xim_disk_cache.is_some() {
        Some(&xim_cache)
    } else {
        None
    };
    let x_trace = build_traces_for_rows(
        &rows_aligned,
        &cfg.xic_path,
        &cfg.xic_paths,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
        cache_opt,
        disk_cache.as_ref(),
    )?;
    let x_xim = build_xim_for_rows(
        &rows_aligned,
        &cfg.xim_path,
        &cfg.xim_paths,
        &cfg.xim_map_path,
        &xim_trace_cfg,
        &cfg.xim_fetch,
        xim_cache_opt,
        xim_disk_cache.as_ref(),
    )?;
    drain_xim_decode_issues(&mut xim_decode_issues);
    log_xic_cache_stats("xrun", &cache_stats);
    log_xim_cache_stats("xrun", &xim_cache_stats);
    let apply_trace_filter = cfg.restrict_osw_to_xic_map
        && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_none();
    if cfg.restrict_osw_to_xic_map && resolve_xic_map(&cfg.xic_paths, &cfg.xic_map_path)?.is_some()
    {
        log::info!(
            "XIC map/path list provided; skipping trace-based restriction (run_id filter only)"
        );
    }
    let (rows_aligned, x_trace, x_xim) = if apply_trace_filter {
        filter_rows_by_trace_with_aux(
            rows_aligned,
            x_trace,
            x_xim,
            cfg.trace.total_c(),
            cfg.trace.l,
            xim_trace_cfg.as_ref().map(|c| c.total_c()),
            xim_trace_cfg.as_ref().map(|c| c.l),
        )
    } else {
        (rows_aligned, x_trace, x_xim)
    };
    let bag_data = build_xrun_bag_data_from_rows_with_cols_with_aux(
        &model,
        &rows_aligned,
        &x_trace,
        x_xim
            .as_ref()
            .zip(xim_trace_cfg.as_ref())
            .map(|(x, cfg)| (x.as_slice(), cfg)),
        &table.feature_cols,
        &meta.feature_cols,
        cfg.trace.total_c(),
        cfg.trace.l,
        cfg.bag_k,
        &device,
        cfg.batch_size,
        meta.preprocess.as_ref(),
    )?;

    let seq = build_xrun_sequences_from_bags(
        &bag_data.bag_pid,
        &bag_data.bag_score,
        &bag_data.bag_hidden,
        bag_data.hidden_dim,
        &bag_data.bag_y,
        cfg.max_runs,
        &cfg.sort_by,
    );
    let ds = XrunDataset {
        xseq: seq.xseq,
        mask: seq.mask,
        y: seq.y_prec,
        p: seq.p,
        r: seq.r,
        din: seq.din,
    };
    Ok((ds, xim_decode_issues))
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn fit_xrun_sidecar_from_dataset(
    checkpoint: &Path,
    summary_tsv: &Path,
    ds: &XrunDataset,
    cfg: &XrunSweepConfig,
    device: &Device,
) -> Result<XrunTrainOnlyOutput> {
    let (tr_ds, va_ds) = split_train_val(ds, cfg.val_frac, cfg.seed);
    if tr_ds.p == 0 || va_ds.p == 0 {
        bail!(
            "XRUN split is empty (train_p={}, val_p={}); increase data size or adjust val_frac",
            tr_ds.p,
            va_ds.p
        );
    }

    let mut trainer = XrunTrainer::new(cfg.train.clone(), ds.din, device)?;
    let meta = trainer.train(&tr_ds, &va_ds, device)?;
    let ckpt_meta = XrunCheckpointMeta {
        train: cfg.train.clone(),
        predict: XrunPredictConfig {
            max_runs: cfg.max_runs,
            sort_by: cfg.sort_by.clone(),
            batch_size: cfg.batch_size.max(1),
        },
        in_dim: ds.din,
        best_val: Some(meta.best_val),
        version: 1,
    };
    save_xrun_checkpoint(checkpoint, &trainer.varmap, &ckpt_meta)?;
    let summary_rows = [XrunSweepRow {
        pool: format!("{:?}", cfg.train.pool),
        tau: cfg.train.tau,
        best_val: meta.best_val,
    }];
    write_xrun_summary_tsv(summary_tsv, &summary_rows)?;
    Ok(XrunTrainOnlyOutput {
        checkpoint_prefix: checkpoint_base(checkpoint),
        best_val: meta.best_val,
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Train only the XRUN sidecar from an existing base TOPAZ checkpoint.
///
/// This reuses the saved base model to rebuild bag scores and winner-hidden
/// embeddings, trains the XRUN calibrator, and writes the sidecar back into the
/// same `topaz.model` archive.
pub fn run_xrun_training(cfg: &XrunSweepConfig) -> Result<XrunTrainOnlyOutput> {
    let device = get_device(&cfg.device)?;
    let (ds, xim_decode_issues) = prepare_xrun_dataset_from_checkpoint(cfg, &device)?;
    let out = fit_xrun_sidecar_from_dataset(&cfg.checkpoint, &cfg.output_tsv, &ds, cfg, &device)?;
    report_xim_decode_issues(
        "xrun",
        &xim_decode_issues,
        &diagnostic_tsv_path(&cfg.output_tsv, "xim_skipped"),
    )?;
    Ok(out)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Evaluate a grid of XRUN calibrator settings without retraining the base
/// TOPAZ model.
pub fn run_xrun_sweep(cfg: &XrunSweepConfig) -> Result<Vec<XrunSweepRow>> {
    let device = get_device(&cfg.device)?;
    let (ds, xim_decode_issues) = prepare_xrun_dataset_from_checkpoint(cfg, &device)?;
    let (tr_ds, va_ds) = split_train_val(&ds, cfg.val_frac, cfg.seed);

    let pools: Vec<String> = cfg
        .sweep_pools
        .clone()
        .unwrap_or_else(|| vec![format!("{:?}", cfg.train.pool).to_lowercase()]);
    let taus: Vec<f64> = cfg
        .sweep_taus
        .clone()
        .unwrap_or_else(|| vec![cfg.train.tau]);

    let mut rows = Vec::new();
    for p in pools {
        let pool = match p.as_str() {
            "max" => XrunPoolMode::Max,
            "lse" => XrunPoolMode::Lse,
            "softmax_mean" => XrunPoolMode::SoftmaxMean,
            "attn_mean" => XrunPoolMode::AttnMean,
            _ => XrunPoolMode::Max,
        };
        for &tau in &taus {
            let mut cfg_t = cfg.train.clone();
            cfg_t.pool = pool.clone();
            cfg_t.tau = tau;

            let mut trainer = XrunTrainer::new(cfg_t, ds.din, &device)?;
            let meta = trainer.train(&tr_ds, &va_ds, &device)?;
            rows.push(XrunSweepRow {
                pool: format!("{:?}", pool),
                tau,
                best_val: meta.best_val,
            });
        }
    }

    write_xrun_summary_tsv(&cfg.output_tsv, &rows)?;
    report_xim_decode_issues(
        "xrun",
        &xim_decode_issues,
        &diagnostic_tsv_path(&cfg.output_tsv, "xim_skipped"),
    )?;
    Ok(rows)
}

#[cfg(all(test, feature = "io-sqlite", feature = "io-parquet"))]
mod tests {
    use super::*;
    use crate::building_blocks::trace_input::TraceInputMode;
    use crate::checkpoint::save_checkpoint;
    use crate::io::xim_parquet::XimDecodeIssue;
    use candle_core::{DType, Tensor};
    use candle_nn::VarBuilder;

    fn tmp_base(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("redeem_topaz_{name}_{stamp}"));
        p
    }

    #[test]
    fn test_diagnostic_tsv_path_uses_input_stem() {
        let p = diagnostic_tsv_path(Path::new("scores/output.tsv"), "xim_skipped");
        assert_eq!(p, PathBuf::from("scores/output.xim_skipped.tsv"));

        let p = diagnostic_tsv_path(Path::new("checkpoints/topaz_v1"), "xim_skipped");
        assert_eq!(p, PathBuf::from("checkpoints/topaz_v1.xim_skipped.tsv"));
    }

    #[test]
    fn test_write_xim_decode_issue_tsv_dedups_rows() -> Result<()> {
        let path = tmp_base("xim_decode_issue").with_extension("tsv");
        let issue = XimDecodeIssue {
            xim_path: PathBuf::from("run1.xim"),
            run_id: 7,
            feature_id: 42,
            annotation: "y7".to_string(),
            field: "MOBILITY_DATA".to_string(),
            compression: 5,
            error: "data too small".to_string(),
        };
        write_xim_decode_issue_tsv(&path, &[issue.clone(), issue])?;
        let text = std::fs::read_to_string(&path)?;
        let _ = std::fs::remove_file(&path);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            "XIM_PATH\tRUN_ID\tFEATURE_ID\tANNOTATION\tFIELD\tCOMPRESSION\tERROR"
        );
        assert!(lines[1].contains("run1.xim\t7\t42\ty7\tMOBILITY_DATA\t5\tdata too small"));
        Ok(())
    }

    fn synthetic_rows() -> Vec<FeatureRow> {
        let mut rows = Vec::new();
        let runs = [101u64, 102, 103];
        let precs = [1001u64, 1002, 1003, 1004];
        let mut feature_id = 1u64;
        for &prec in &precs {
            for (ri, &run_id) in runs.iter().enumerate() {
                rows.push(FeatureRow {
                    feature_id,
                    precursor_id: prec,
                    run_id,
                    group_id: format!("{run_id}_{prec}"),
                    exp_rt: 1000.0 + prec as f32 * 0.1 + ri as f32,
                    rt_left_width: None,
                    rt_right_width: None,
                    exp_im: Some(1.0 + ri as f32 * 0.01),
                    exp_im_left_width: Some(0.95),
                    exp_im_right_width: Some(1.05),
                    is_decoy: prec % 2 == 0,
                    features: vec![
                        prec as f32 * 0.001,
                        run_id as f32 * 0.0001 + ri as f32 * 0.01,
                    ],
                });
                feature_id += 1;
            }
        }
        rows
    }

    fn synthetic_traces(rows: &[FeatureRow], c: usize, l: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows.len() * c * l);
        for (i, row) in rows.iter().enumerate() {
            for ch in 0..c {
                for t in 0..l {
                    let v = ((i + 1) as f32 * 0.1)
                        + (ch as f32 * 0.2)
                        + (t as f32 / l as f32)
                        + (row.precursor_id % 5) as f32 * 0.05;
                    out.push(v.max(0.0));
                }
            }
        }
        out
    }

    #[test]
    fn test_xrun_sidecar_smoke_end_to_end() -> Result<()> {
        let device = Device::Cpu;
        let base = tmp_base("xrun_sidecar");

        let model_cfg = TopazConfig {
            feat_dim: 2,
            ms2_cmax: 2,
            ms1_cmax: 0,
            l: 8,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            ..Default::default()
        };
        let trace_cfg = TraceBuildConfig {
            l: model_cfg.l,
            ms1_cmax: model_cfg.ms1_cmax,
            ms2_cmax: model_cfg.ms2_cmax,
            normalize_max: false,
        };
        let feature_cols = vec!["f0".to_string(), "f1".to_string()];
        let rows = synthetic_rows();
        let x_trace = synthetic_traces(&rows, trace_cfg.total_c(), trace_cfg.l);

        let train_cfg = TrainConfig {
            learning_rate: 1e-3,
            lambda_pair: 0.0,
            lambda_inbag: 0.0,
            lambda_winner_margin: 0.0,
            ..Default::default()
        };
        let trainer = Trainer::new(train_cfg.clone(), &model_cfg, &device)?;
        let meta = CheckpointMeta {
            model: model_cfg.clone(),
            train: Some(train_cfg),
            trace: Some(trace_cfg.clone()),
            feature_cols: feature_cols.clone(),
            preprocess: None,
            version: 1,
        };
        save_checkpoint(&base, &trainer.varmap, &meta)?;

        let bag_data = build_xrun_bag_data_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows,
            &x_trace,
            None,
            &feature_cols,
            &feature_cols,
            trace_cfg.total_c(),
            trace_cfg.l,
            1,
            &device,
            8,
            None,
        )?;
        let seq = build_xrun_sequences_from_bags(
            &bag_data.bag_pid,
            &bag_data.bag_score,
            &bag_data.bag_hidden,
            bag_data.hidden_dim,
            &bag_data.bag_y,
            8,
            "run",
        );
        let ds = XrunDataset {
            xseq: seq.xseq,
            mask: seq.mask,
            y: seq.y_prec,
            p: seq.p,
            r: seq.r,
            din: seq.din,
        };
        let (tr_ds, va_ds) = split_train_val(&ds, 0.25, 7);
        assert!(tr_ds.p > 0);
        assert!(va_ds.p > 0);

        let xrun_train_cfg = XrunTrainConfig {
            d_model: 16,
            attn_hidden: 16,
            head_hidden: vec![8],
            batch_size: 8,
            max_epochs: 2,
            patience: 1,
            dropout: 0.0,
            ..Default::default()
        };
        let mut xrun_trainer = XrunTrainer::new(xrun_train_cfg.clone(), ds.din, &device)?;
        let xrun_meta = xrun_trainer.train(&tr_ds, &va_ds, &device)?;
        save_xrun_checkpoint(
            &base,
            &xrun_trainer.varmap,
            &XrunCheckpointMeta {
                train: xrun_train_cfg.clone(),
                predict: XrunPredictConfig {
                    max_runs: 8,
                    sort_by: "run".to_string(),
                    batch_size: 8,
                },
                in_dim: ds.din,
                best_val: Some(xrun_meta.best_val),
                version: 1,
            },
        )?;

        let mut loaded_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&loaded_varmap, DType::F32, &device);
        let loaded_model = TopazBagRanker::new(vb.pp("topaz"), &model_cfg)?;
        load_checkpoint_weights(&base, &mut loaded_varmap)?;
        let (_xrun_varmap, loaded_xrun, loaded_xrun_meta) =
            load_xrun_calibrator(&base, &device)?.expect("xrun sidecar should exist");

        let x_feat = rows_to_feature_matrix_with_cols(&rows, &feature_cols, &feature_cols, None);
        let x_feat_t = Tensor::from_vec(x_feat, (rows.len(), model_cfg.feat_dim), &device)?;
        let x_trace_t = Tensor::from_slice(
            &x_trace,
            (rows.len(), trace_cfg.total_c(), trace_cfg.l),
            &device,
        )?;
        let base_scores = crate::infer::score_candidates(&loaded_model, &x_feat_t, &x_trace_t, 8)?
            .to_vec1::<f32>()?;
        let (winner_rows, bag_pid, bag_score, _bag_is_decoy, bag_y) =
            select_bag_winners(&rows, &base_scores);
        let winner_trace = synthetic_traces(&winner_rows, trace_cfg.total_c(), trace_cfg.l);
        let head_out = score_bags_with_heads_from_rows_with_cols_with_aux(
            &loaded_model,
            &winner_rows,
            &winner_trace,
            None,
            &feature_cols,
            &feature_cols,
            trace_cfg.total_c(),
            trace_cfg.l,
            1,
            &device,
            8,
            None,
        )?;
        let loaded_bag_data = crate::xrun::XrunBagData {
            bag_score: bag_score.clone(),
            bag_hidden: head_out.winner_hidden.clone(),
            hidden_dim: head_out.hidden_dim,
            bag_y: bag_y.clone(),
            bag_pid: bag_pid.clone(),
        };
        let (delta_bag, _attn_entropy) = crate::xrun::xrun_predict_deltas_for_bags(
            &loaded_xrun,
            &loaded_bag_data,
            &loaded_xrun_meta.predict,
            &device,
        )?;
        let calibrated_scores =
            apply_xrun_deltas_to_rows(&base_scores, &rows, &bag_pid, &delta_bag);

        assert_eq!(calibrated_scores.len(), rows.len());
        assert_eq!(bag_pid.len(), bag_data.bag_pid.len());
        assert_eq!(delta_bag.len(), bag_data.bag_pid.len());
        assert!(xrun_checkpoint_exists(&base));

        let _ = std::fs::remove_file(base.with_extension("model"));
        Ok(())
    }

    #[test]
    fn test_fit_xrun_sidecar_from_dataset_writes_summary_and_sidecar() -> Result<()> {
        let device = Device::Cpu;
        let base = tmp_base("xrun_only");
        let summary = base.with_extension("xrun.tsv");

        let ds = XrunDataset {
            xseq: vec![
                2.0, 0.2, 0.1, 1.8, 0.3, 0.2, 0.0, 0.0, 0.0, // precursor 0
                -1.5, 0.1, -0.2, -1.2, 0.0, -0.1, 0.0, 0.0, 0.0, // precursor 1
            ],
            mask: vec![true, true, false, true, true, false],
            y: vec![1.0, 0.0],
            p: 2,
            r: 3,
            din: 3,
        };
        let cfg = XrunSweepConfig {
            checkpoint: base.clone(),
            output_tsv: summary.clone(),
            val_frac: 0.5,
            seed: 7,
            max_runs: 3,
            sort_by: "run".to_string(),
            train: XrunTrainConfig {
                d_model: 8,
                attn_hidden: 8,
                head_hidden: vec![4],
                batch_size: 2,
                max_epochs: 2,
                patience: 1,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let out = fit_xrun_sidecar_from_dataset(&base, &summary, &ds, &cfg, &device)?;
        assert_eq!(out.checkpoint_prefix, base);
        assert!(xrun_checkpoint_exists(&base));
        let summary_text = std::fs::read_to_string(&summary)?;
        assert!(summary_text.contains("pool\ttau\tbest_val"));

        let _ = std::fs::remove_file(base.with_extension("model"));
        let _ = std::fs::remove_file(summary);
        Ok(())
    }

    #[test]
    fn test_xim_branch_smoke_end_to_end() -> Result<()> {
        let device = Device::Cpu;
        let rows = synthetic_rows();
        let feature_cols = vec!["f0".to_string(), "f1".to_string()];

        let model_cfg = TopazConfig {
            feat_dim: 2,
            ms2_cmax: 2,
            ms1_cmax: 0,
            l: 8,
            trace_emb_dim: 8,
            mlp_hidden: vec![8],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            xim: Some(crate::model::topaz::TopazXimConfig {
                ms2_cmax: 3,
                ms1_cmax: 1,
                l: 12,
                trace_emb_dim: 6,
                trace_input_mode: TraceInputMode::Single,
                use_coelution_head: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let trace_cfg = TraceBuildConfig {
            l: model_cfg.l,
            ms1_cmax: model_cfg.ms1_cmax,
            ms2_cmax: model_cfg.ms2_cmax,
            normalize_max: false,
        };
        let xim_cfg = effective_xim_trace_cfg(&None, &model_cfg).expect("xim config should exist");

        let train_cfg = TrainConfig {
            learning_rate: 1e-3,
            lambda_pair: 0.0,
            lambda_inbag: 0.0,
            lambda_winner_margin: 0.0,
            ..Default::default()
        };
        let trainer = Trainer::new(train_cfg, &model_cfg, &device)?;

        let x_trace = synthetic_traces(&rows, trace_cfg.total_c(), trace_cfg.l);
        let x_xim = synthetic_traces(&rows, xim_cfg.total_c(), xim_cfg.l);
        let x_feat = rows_to_feature_matrix_with_cols(&rows, &feature_cols, &feature_cols, None);
        let x_feat_t = Tensor::from_vec(x_feat, (rows.len(), model_cfg.feat_dim), &device)?;
        let x_trace_t = Tensor::from_slice(
            &x_trace,
            (rows.len(), trace_cfg.total_c(), trace_cfg.l),
            &device,
        )?;
        let x_xim_t =
            Tensor::from_slice(&x_xim, (rows.len(), xim_cfg.total_c(), xim_cfg.l), &device)?;

        let scores = crate::infer::score_candidates_with_aux(
            &trainer.model,
            &x_feat_t,
            &x_trace_t,
            Some(&x_xim_t),
            8,
        )?
        .to_vec1::<f32>()?;
        assert_eq!(scores.len(), rows.len());

        let head_out = score_bags_with_heads_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows,
            &x_trace,
            Some((x_xim.as_slice(), &xim_cfg)),
            &feature_cols,
            &feature_cols,
            trace_cfg.total_c(),
            trace_cfg.l,
            1,
            &device,
            8,
            None,
        )?;
        assert_eq!(head_out.bag_pid.len(), rows.len());
        assert!(head_out.hidden_dim > 0);
        assert!(head_out.emb_all_dim >= head_out.emb_ms2_dim);
        assert!(head_out.coe_all_dim >= head_out.coe_ms2_dim);

        let bag_data = build_xrun_bag_data_from_rows_with_cols_with_aux(
            &trainer.model,
            &rows,
            &x_trace,
            Some((x_xim.as_slice(), &xim_cfg)),
            &feature_cols,
            &feature_cols,
            trace_cfg.total_c(),
            trace_cfg.l,
            1,
            &device,
            8,
            None,
        )?;
        assert_eq!(bag_data.bag_pid.len(), rows.len());
        assert!(bag_data.hidden_dim > 0);
        Ok(())
    }
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
/// Stub entry point used when the crate is built without the IO features
/// required by preprocessing.
pub fn run_preprocess(_cfg: &PreprocessRunConfig) -> Result<PreprocessRunOutput> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
/// Stub entry point used when the crate is built without the IO features
/// required by the XRUN sweep.
pub fn run_xrun_sweep(_cfg: &XrunSweepConfig) -> Result<Vec<XrunSweepRow>> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
/// Stub entry point used when the crate is built without the IO features
/// required by standalone XRUN training.
pub fn run_xrun_training(_cfg: &XrunSweepConfig) -> Result<XrunTrainOnlyOutput> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
/// Stub entry point used when the crate is built without the IO features
/// required by training.
pub fn run_training(_cfg: &TrainRunConfig) -> Result<TrainRunOutput> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
/// Stub entry point used when the crate is built without the IO features
/// required by inference.
pub fn run_inference(_cfg: &InferRunConfig) -> Result<InferRunOutput> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}
