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
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use candle_core::Device;

use crate::checkpoint::{
    CheckpointMeta, XrunCheckpointMeta, load_checkpoint_partial, load_xrun_checkpoint,
    read_xrun_checkpoint_meta, save_xrun_checkpoint, xrun_checkpoint_exists,
};
use crate::config::Config as TrainConfig;
use crate::infer::{TraceBuildConfig, XicFetchConfig};
use crate::io::osw::OswReadConfig;
use crate::model::topaz::TopazConfig;
use crate::train::TrainFilter;
use crate::xrun::XrunTrainConfig;
use crate::xrun::calibrator::{XrunAttentionCalibrator, XrunConfig};

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::checkpoint::save_checkpoint;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::diagnostics::{print_trace_summary, trace_summary, warn_if_missing_ms1};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::{
    SharedXicCache, XicDiskCache, build_score_table_from_rows, build_trace_tensors_from_parquet,
    build_trace_tensors_from_parquet_cached, build_trace_tensors_from_parquet_map,
    build_trace_tensors_from_parquet_map_cached, rows_to_feature_matrix_with_cols,
    score_bags_from_rows_with_cols, score_bags_with_heads_from_rows_with_cols, score_candidates,
    tdc_summary,
};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::io::osw::FeatureRow;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::io::osw::read_feature_rows;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::model::topaz::TopazBagRanker;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::train::{
    Trainer, bags_to_train_batches, filter_training_rows, fit_preprocessor_from_rows_with_cols,
    split_rows_by_precursor, subsample_train_rows_by_bag,
};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::xrun::pipeline::{
    XrunPredictConfig, apply_xrun_deltas, apply_xrun_deltas_to_rows, build_xrun_bag_data_from_rows,
    build_xrun_bag_data_from_rows_with_cols,
};
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
    pub osw_path: PathBuf,
    pub xic_path: PathBuf,
    pub xic_map_path: Option<PathBuf>,
    pub output_prefix: PathBuf,
    pub init_checkpoint: Option<PathBuf>,
    pub device: String,
    pub bag_k: usize,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub val_frac: f32,
    pub train_frac: f32,
    pub train_stratify_run: bool,
    pub seed: u64,
    pub model: TopazConfig,
    pub train: TrainConfig,
    pub trace: TraceBuildConfig,
    pub fetch: XicFetchConfig,
    pub osw: OswReadConfig,
    pub filter: TrainFilter,
    pub feature_select: FeatureSelectConfig,
    pub diagnostics: DiagnosticsConfig,
    pub restrict_osw_to_xic_map: bool,
    pub xic_cache_max_precursors: usize,
    pub xic_cache_dir: Option<PathBuf>,
    pub xic_cache_max_bytes: Option<u64>,
    pub trace_chunk_size: usize,
    pub xrun: XrunRunConfig,
}

impl Default for TrainRunConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_map_path: None,
            output_prefix: PathBuf::from("topaz_checkpoint"),
            init_checkpoint: None,
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
            fetch: XicFetchConfig::default(),
            osw: OswReadConfig::default(),
            filter: TrainFilter::default(),
            feature_select: FeatureSelectConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            restrict_osw_to_xic_map: false,
            xic_cache_max_precursors: 50_000,
            xic_cache_dir: None,
            xic_cache_max_bytes: None,
            trace_chunk_size: 5000,
            xrun: XrunRunConfig::default(),
        }
    }
}

/// Top-level inference configuration consumed by [`run_inference`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferRunConfig {
    pub osw_path: PathBuf,
    pub xic_path: PathBuf,
    pub xic_map_path: Option<PathBuf>,
    pub checkpoint: PathBuf,
    pub output_tsv: PathBuf,
    pub output_osw: Option<PathBuf>,
    pub output_table: String,
    pub output_table_base: Option<String>,
    pub output_table_xrun: Option<String>,
    pub device: String,
    pub batch_size: usize,
    pub pep_bins: usize,
    pub bag_k: usize,
    pub trace: TraceBuildConfig,
    pub fetch: XicFetchConfig,
    pub osw: OswReadConfig,
    pub diagnostics: DiagnosticsConfig,
    pub restrict_osw_to_xic_map: bool,
    pub xic_cache_max_precursors: usize,
    pub xic_cache_dir: Option<PathBuf>,
    pub xic_cache_max_bytes: Option<u64>,
    pub trace_chunk_size: usize,
    pub xrun: XrunRunConfig,
}

impl Default for InferRunConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_map_path: None,
            checkpoint: PathBuf::from("topaz_checkpoint"),
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
            fetch: XicFetchConfig::default(),
            osw: OswReadConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            restrict_osw_to_xic_map: false,
            xic_cache_max_precursors: 50_000,
            xic_cache_dir: None,
            xic_cache_max_bytes: None,
            trace_chunk_size: 5000,
            xrun: XrunRunConfig::default(),
        }
    }
}

/// Result returned after a successful training run.
#[derive(Debug, Clone)]
pub struct TrainRunOutput {
    pub checkpoint_prefix: PathBuf,
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
    pub osw_path: PathBuf,
    pub xic_path: PathBuf,
    pub xic_map_path: Option<PathBuf>,
    pub checkpoint: PathBuf,
    pub output_tsv: PathBuf,
    pub device: String,
    pub trace: TraceBuildConfig,
    pub fetch: XicFetchConfig,
    pub osw: OswReadConfig,
    pub bag_k: usize,
    pub batch_size: usize,
    pub val_frac: f32,
    pub seed: u64,
    pub max_runs: usize,
    pub sort_by: String,
    pub train: XrunTrainConfig,
    pub sweep_pools: Option<Vec<String>>,
    pub sweep_taus: Option<Vec<f64>>,
    pub restrict_osw_to_xic_map: bool,
    pub xic_cache_max_precursors: usize,
    pub xic_cache_dir: Option<PathBuf>,
    pub xic_cache_max_bytes: Option<u64>,
}

impl Default for XrunSweepConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_map_path: None,
            checkpoint: PathBuf::from("topaz_checkpoint"),
            output_tsv: PathBuf::from("xrun_sweep.tsv"),
            device: "cpu".to_string(),
            trace: TraceBuildConfig {
                l: 64,
                ms1_cmax: 0,
                ms2_cmax: 6,
                normalize_max: true,
            },
            fetch: XicFetchConfig::default(),
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

fn filter_rows_by_trace(
    rows: Vec<crate::io::osw::FeatureRow>,
    x_trace: Vec<f32>,
    c_total: usize,
    l: usize,
) -> (Vec<crate::io::osw::FeatureRow>, Vec<f32>) {
    if rows.is_empty() {
        return (rows, x_trace);
    }
    let mut out_rows = Vec::new();
    let mut out_trace = Vec::new();
    let span = c_total * l;
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
        }
    }
    (out_rows, out_trace)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn log_run_id_summary(rows: &[FeatureRow], xic_path: &Path) {
    if !log::log_enabled!(log::Level::Info) {
        return;
    }
    let mut osw_runs: Vec<u64> = rows.iter().map(|r| r.run_id).collect();
    osw_runs.sort_unstable();
    osw_runs.dedup();
    log::info!("OSW run_ids (n={}): {:?}", osw_runs.len(), osw_runs);

    match crate::io::xic_parquet::list_run_ids(xic_path) {
        Ok(mut runs) => {
            runs.sort_unstable();
            runs.dedup();
            log::info!("XIC run_ids (n={}): {:?}", runs.len(), runs);
        }
        Err(e) => {
            log::warn!("Failed to read XIC run_ids: {e:#}");
        }
    }
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn read_xic_map(path: &Path) -> Result<HashMap<u64, PathBuf>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read xic_map: {path:?}"))?;
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
        bail!("xic_map has no usable entries: {path:?}");
    }
    Ok(map)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn filter_rows_by_xic_map(
    rows: Vec<FeatureRow>,
    xic_map_path: &Option<PathBuf>,
) -> Result<Vec<FeatureRow>> {
    let Some(map_path) = xic_map_path else {
        return Ok(rows);
    };
    let map = read_xic_map(map_path)?;
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
    xic_map_path: &Option<PathBuf>,
    trace: &TraceBuildConfig,
    fetch: &XicFetchConfig,
    cache: Option<&SharedXicCache>,
    disk: Option<&XicDiskCache>,
) -> Result<Vec<f32>> {
    if let Some(map_path) = xic_map_path {
        let map = read_xic_map(map_path)?;
        log::info!(
            "Using XIC map with {} entries from {:?}",
            map.len(),
            map_path
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
fn build_traces_for_train_val(
    rows_tr: &[FeatureRow],
    rows_va: &[FeatureRow],
    xic_path: &Path,
    xic_map_path: &Option<PathBuf>,
    trace: &TraceBuildConfig,
    fetch: &XicFetchConfig,
    cache: Option<&SharedXicCache>,
    disk: Option<&XicDiskCache>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if rows_va.is_empty() {
        let x_tr =
            build_traces_for_rows(rows_tr, xic_path, xic_map_path, trace, fetch, cache, disk)?;
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
    let x_all =
        build_traces_for_rows(&rows_all, xic_path, xic_map_path, trace, fetch, cache, disk)?;
    let span = trace.total_c() * trace.l;
    let split = rows_tr.len() * span;
    let x_tr = x_all[..split].to_vec();
    let x_va = x_all[split..].to_vec();
    Ok((x_tr, x_va))
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
        if ext == "safetensors" || ext == "json" {
            return path.with_extension("");
        }
    }
    path.to_path_buf()
}

fn read_checkpoint_meta(base: &Path) -> Result<CheckpointMeta> {
    let meta_path = base.with_extension("json");
    let text = std::fs::read_to_string(&meta_path)
        .with_context(|| format!("failed to read checkpoint meta: {meta_path:?}"))?;
    let meta: CheckpointMeta = serde_json::from_str(&text)?;
    Ok(meta)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn load_checkpoint_weights(base: &Path, varmap: &mut VarMap) -> Result<()> {
    let weights = base.with_extension("safetensors");
    varmap.load(&weights)?;
    Ok(())
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
    bag_is_decoy: Vec<bool>,
    bag_y: Vec<f32>,
    delta_bag: Vec<f32>,
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn apply_xrun_to_row_scores(
    rows: &[FeatureRow],
    row_scores: &[f32],
    model: &TopazBagRanker,
    table_feature_cols: &[String],
    target_cols: &[String],
    trace_cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
    xic_path: &Path,
    xic_map_path: &Option<PathBuf>,
    cache_opt: Option<&SharedXicCache>,
    disk_cache: Option<&XicDiskCache>,
    xrun_cfg: &XrunRunConfig,
    xrun_model: &XrunAttentionCalibrator,
    xrun_meta: &XrunCheckpointMeta,
    device: &Device,
    base_batch_size: usize,
    pre: Option<&crate::Preprocessor>,
) -> Result<XrunAppliedScores> {
    let (winner_rows, bag_pid, bag_score, bag_is_decoy, bag_y) =
        select_bag_winners(rows, row_scores);
    if winner_rows.is_empty() {
        return Ok(XrunAppliedScores {
            row_scores: row_scores.to_vec(),
            bag_pid,
            bag_score,
            bag_is_decoy,
            bag_y,
            delta_bag: Vec::new(),
        });
    }

    let x_trace = build_traces_for_rows(
        &winner_rows,
        xic_path,
        xic_map_path,
        trace_cfg,
        fetch_cfg,
        cache_opt,
        disk_cache,
    )?;
    let head_out = score_bags_with_heads_from_rows_with_cols(
        model,
        &winner_rows,
        &x_trace,
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
        bag_is_decoy,
        bag_y,
        delta_bag,
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
/// Train the base TOPAZ model and, optionally, an XRUN calibrator sidecar.
pub fn run_training(cfg: &TrainRunConfig) -> Result<TrainRunOutput> {
    let device = get_device(&cfg.device)?;
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
    log_run_id_summary(&rows, &cfg.xic_path);
    if cfg.restrict_osw_to_xic_map {
        rows = filter_rows_by_xic_map(rows, &cfg.xic_map_path)?;
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
    let (x_tr, x_va) = build_traces_for_train_val(
        &rows_tr,
        &rows_va,
        &cfg.xic_path,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
        cache_opt,
        disk_cache.as_ref(),
    )?;
    log_xic_cache_stats("train", &cache_stats);

    let apply_trace_filter = cfg.restrict_osw_to_xic_map && cfg.xic_map_path.is_none();
    if cfg.restrict_osw_to_xic_map && cfg.xic_map_path.is_some() {
        log::info!("XIC map provided; skipping trace-based restriction (run_id filter only)");
    }
    let (rows_tr, x_tr) = if apply_trace_filter {
        filter_rows_by_trace(rows_tr, x_tr, cfg.trace.total_c(), cfg.trace.l)
    } else {
        (rows_tr, x_tr)
    };
    if apply_trace_filter && rows_tr.is_empty() {
        bail!(
            "all training rows were dropped after XIC restriction; check run_id match and xic_path"
        );
    }
    let (rows_va, x_va) = if apply_trace_filter {
        filter_rows_by_trace(rows_va, x_va, cfg.trace.total_c(), cfg.trace.l)
    } else {
        (rows_va, x_va)
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
    let batches = bags_to_train_batches(bags, &device, cfg.batch_size)?;

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
        bags_to_train_batches(bags_va, &device, cfg.batch_size)?
    } else {
        Vec::new()
    };
    let _history = trainer.train_epochs_early_stop(&batches, &val_batches, cfg.max_epochs, None)?;

    if !rows_va.is_empty() {
        let out = score_bags_from_rows_with_cols(
            &trainer.model,
            &rows_va,
            &x_va,
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
        let out = score_bags_with_heads_from_rows_with_cols(
            &trainer.model,
            &rows_all,
            &x_all,
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
        let bag_data = build_xrun_bag_data_from_rows_with_cols(
            &trainer.model,
            &rows_all,
            &x_all,
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
    log_run_id_summary(&rows, &cfg.xic_path);
    if cfg.restrict_osw_to_xic_map {
        rows = filter_rows_by_xic_map(rows, &cfg.xic_map_path)?;
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
    let apply_trace_filter = cfg.restrict_osw_to_xic_map && cfg.xic_map_path.is_none();
    if cfg.restrict_osw_to_xic_map && cfg.xic_map_path.is_some() {
        log::info!("XIC map provided; skipping trace-based restriction (run_id filter only)");
    }

    let target_cols = meta.feature_cols.clone();
    let feat_dim = if meta.model.use_heuristic_features {
        meta.model.feat_dim
    } else {
        0
    };
    let chunk_size = if cfg.trace_chunk_size == 0 {
        rows.len().max(1)
    } else {
        cfg.trace_chunk_size.max(1)
    };

    let mut scores: Vec<f32> = if apply_trace_filter {
        Vec::new()
    } else {
        vec![0f32; rows.len()]
    };
    let mut rows_scored: Vec<FeatureRow> = Vec::new();

    let mut sum_n = 0usize;
    let mut sum_ms1 = 0usize;
    let mut sum_ms2 = 0usize;

    let mut offset = 0usize;
    for chunk in rows.chunks(chunk_size) {
        let x_trace = build_traces_for_rows(
            chunk,
            &cfg.xic_path,
            &cfg.xic_map_path,
            &cfg.trace,
            &cfg.fetch,
            cache_opt,
            disk_cache.as_ref(),
        )?;

        let (chunk_rows, x_trace) = if apply_trace_filter {
            let (rows_f, x_tr_f) =
                filter_rows_by_trace(chunk.to_vec(), x_trace, cfg.trace.total_c(), cfg.trace.l);
            if rows_f.is_empty() {
                continue;
            }
            (rows_f, x_tr_f)
        } else {
            (Vec::new(), x_trace)
        };

        let row_slice: &[FeatureRow] = if apply_trace_filter {
            &chunk_rows
        } else {
            chunk
        };
        let n_chunk = row_slice.len();
        if n_chunk == 0 {
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

        let x_feat = if meta.model.use_heuristic_features {
            rows_to_feature_matrix_with_cols(
                row_slice,
                &table.feature_cols,
                &target_cols,
                meta.preprocess.as_ref(),
            )
        } else {
            Vec::new()
        };
        let x_feat_t = candle_core::Tensor::from_vec(x_feat, (n_chunk, feat_dim), &device)?;
        let x_trace_t = candle_core::Tensor::from_slice(
            &x_trace,
            (n_chunk, cfg.trace.total_c(), cfg.trace.l),
            &device,
        )?;
        let scores_t = score_candidates(&model, &x_feat_t, &x_trace_t, cfg.batch_size.max(1))?;
        let scores_chunk = scores_t.to_vec1::<f32>()?;

        if apply_trace_filter {
            rows_scored.extend(chunk_rows);
            scores.extend(scores_chunk);
        } else {
            scores[offset..offset + n_chunk].copy_from_slice(&scores_chunk);
            offset += n_chunk;
        }
    }

    log_xic_cache_stats("infer", &cache_stats);
    if apply_trace_filter {
        rows = rows_scored;
    }
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

    let base_scores = scores.clone();
    let xrun_applied = if cfg.xrun.enabled {
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
            &cfg.fetch,
            &cfg.xic_path,
            &cfg.xic_map_path,
            cache_opt,
            disk_cache.as_ref(),
            &cfg.xrun,
            &xrun_model,
            &xrun_meta,
            &device,
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

    if cfg.diagnostics.save_head_embeddings {
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
                    &cfg.xic_map_path,
                    &cfg.trace,
                    &cfg.fetch,
                    cache_opt,
                    disk_cache.as_ref(),
                )?;
                let mut out = score_bags_with_heads_from_rows_with_cols(
                    &model,
                    &winner_rows,
                    &x_trace,
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
                &cfg.xic_map_path,
                &cfg.trace,
                &cfg.fetch,
                cache_opt,
                disk_cache.as_ref(),
            )?;
            let mut out = score_bags_with_heads_from_rows_with_cols(
                &model,
                &rows,
                &x_trace,
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
            if let Some(base_name) = cfg
                .output_table_base
                .as_ref()
                .filter(|name| !name.is_empty())
            {
                let osw_rows = score_rows_to_osw(&table_rows_base);
                crate::io::osw::write_score_table(osw_path, base_name, &osw_rows)?;
                log::info!("Wrote base TOPAZ scores to OSW table {:?}", base_name);
            }

            let final_table_name = if cfg.xrun.enabled {
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

    Ok(InferRunOutput { n_rows: rows.len() })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Evaluate a grid of XRUN calibrator settings without retraining the base
/// TOPAZ model.
pub fn run_xrun_sweep(cfg: &XrunSweepConfig) -> Result<Vec<XrunSweepRow>> {
    let device = get_device(&cfg.device)?;

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
    log_run_id_summary(&rows_aligned, &cfg.xic_path);
    if cfg.restrict_osw_to_xic_map {
        rows_aligned = filter_rows_by_xic_map(rows_aligned, &cfg.xic_map_path)?;
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
    let x_trace = build_traces_for_rows(
        &rows_aligned,
        &cfg.xic_path,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
        cache_opt,
        disk_cache.as_ref(),
    )?;
    log_xic_cache_stats("xrun", &cache_stats);
    let apply_trace_filter = cfg.restrict_osw_to_xic_map && cfg.xic_map_path.is_none();
    if cfg.restrict_osw_to_xic_map && cfg.xic_map_path.is_some() {
        log::info!("XIC map provided; skipping trace-based restriction (run_id filter only)");
    }
    let (rows_aligned, x_trace) = if apply_trace_filter {
        filter_rows_by_trace(rows_aligned, x_trace, cfg.trace.total_c(), cfg.trace.l)
    } else {
        (rows_aligned, x_trace)
    };
    let bag_data = build_xrun_bag_data_from_rows(
        &model,
        &rows_aligned,
        &x_trace,
        meta.model.feat_dim,
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

    let mut text = String::new();
    text.push_str("pool\ttau\tbest_val\n");
    for r in &rows {
        text.push_str(&format!("{}\t{}\t{}\n", r.pool, r.tau, r.best_val));
    }
    std::fs::write(&cfg.output_tsv, text)?;
    Ok(rows)
}

#[cfg(all(test, feature = "io-sqlite", feature = "io-parquet"))]
mod tests {
    use super::*;
    use crate::building_blocks::trace_input::TraceInputMode;
    use crate::checkpoint::save_checkpoint;
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

        let bag_data = build_xrun_bag_data_from_rows_with_cols(
            &trainer.model,
            &rows,
            &x_trace,
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
        let base_scores =
            score_candidates(&loaded_model, &x_feat_t, &x_trace_t, 8)?.to_vec1::<f32>()?;
        let (winner_rows, bag_pid, bag_score, _bag_is_decoy, bag_y) =
            select_bag_winners(&rows, &base_scores);
        let winner_trace = synthetic_traces(&winner_rows, trace_cfg.total_c(), trace_cfg.l);
        let head_out = score_bags_with_heads_from_rows_with_cols(
            &loaded_model,
            &winner_rows,
            &winner_trace,
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

        let _ = std::fs::remove_file(base.with_extension("safetensors"));
        let _ = std::fs::remove_file(base.with_extension("json"));
        let _ = std::fs::remove_file(base.with_extension("xrun.safetensors"));
        let _ = std::fs::remove_file(base.with_extension("xrun.json"));
        Ok(())
    }
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
/// Stub entry point used when the crate is built without the IO features
/// required by the XRUN sweep.
pub fn run_xrun_sweep(_cfg: &XrunSweepConfig) -> Result<Vec<XrunSweepRow>> {
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
