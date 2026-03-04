use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};

use candle_core::Device;

use crate::checkpoint::CheckpointMeta;
use crate::config::Config as TrainConfig;
use crate::infer::{TraceBuildConfig, XicFetchConfig};
use crate::io::osw::OswReadConfig;
use crate::model::topaz::TopazConfig;
use crate::train::TrainFilter;
use crate::xrun::XrunTrainConfig;

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use candle_nn::{VarBuilder, VarMap};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::checkpoint::save_checkpoint;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::{
    build_trace_tensors_from_parquet,
    build_trace_tensors_from_parquet_map,
    build_score_table_from_rows,
    score_candidates,
    score_bags_from_rows_with_cols,
    score_bags_with_heads_from_rows_with_cols,
    tdc_summary,
    rows_to_feature_matrix_with_cols,
};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::io::osw::read_feature_rows;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::model::topaz::TopazBagRanker;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::train::{
    bags_to_train_batches,
    filter_training_rows,
    fit_preprocessor_from_rows_with_cols,
    split_rows_by_precursor,
    subsample_train_rows_by_bag,
    Trainer,
};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::diagnostics::{print_trace_summary, trace_summary, warn_if_missing_ms1};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::io::osw::FeatureRow;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::xrun::pipeline::build_xrun_bag_data_from_rows;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::xrun::sequence::build_xrun_sequences_from_bags;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::xrun::train::{split_train_val, XrunDataset, XrunTrainer, XrunPoolMode};

#[cfg(feature = "io-sqlite")]
use crate::io::osw::ScoreRow as OswScoreRow;
#[cfg(feature = "io-sqlite")]
use crate::infer::write_rank1_disagreement_tsvs;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainRunConfig {
    pub osw_path: PathBuf,
    pub xic_path: PathBuf,
    pub xic_map_path: Option<PathBuf>,
    pub output_prefix: PathBuf,
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
}

impl Default for TrainRunConfig {
    fn default() -> Self {
        Self {
            osw_path: PathBuf::new(),
            xic_path: PathBuf::new(),
            xic_map_path: None,
            output_prefix: PathBuf::from("topaz_checkpoint"),
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
        }
    }
}

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
    pub device: String,
    pub batch_size: usize,
    pub pep_bins: usize,
    pub bag_k: usize,
    pub trace: TraceBuildConfig,
    pub fetch: XicFetchConfig,
    pub osw: OswReadConfig,
    pub diagnostics: DiagnosticsConfig,
    pub restrict_osw_to_xic_map: bool,
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainRunOutput {
    pub checkpoint_prefix: PathBuf,
}

#[derive(Debug, Clone)]
pub struct InferRunOutput {
    pub n_rows: usize,
}

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
        }
    }
}

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
        let Some(path_str) = parts.next() else { continue; };
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
fn build_traces_for_rows(
    rows: &[FeatureRow],
    xic_path: &Path,
    xic_map_path: &Option<PathBuf>,
    trace: &TraceBuildConfig,
    fetch: &XicFetchConfig,
) -> Result<Vec<f32>> {
    if let Some(map_path) = xic_map_path {
        let map = read_xic_map(map_path)?;
        log::info!("Using XIC map with {} entries from {:?}", map.len(), map_path);
        build_trace_tensors_from_parquet_map(rows, &map, trace, fetch)
    } else {
        build_trace_tensors_from_parquet(rows, xic_path, trace, fetch)
    }
}

fn write_head_embeddings_tsv(
    path: &Path,
    out: &crate::infer::BagHeadOutput,
) -> Result<()> {
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
        row.push(if out.is_decoy.get(i).copied().unwrap_or(false) { "1".to_string() } else { "0".to_string() });
        row.push(format!("{}", out.bag_score.get(i).copied().unwrap_or(0.0)));

        let off = i * out.hidden_dim;
        for j in 0..out.hidden_dim {
            row.push(format!("{}", out.winner_hidden.get(off + j).copied().unwrap_or(0.0)));
        }
        let off = i * out.emb_ms2_dim;
        for j in 0..out.emb_ms2_dim {
            row.push(format!("{}", out.emb_ms2.get(off + j).copied().unwrap_or(0.0)));
        }
        let off = i * out.emb_ms1_dim;
        for j in 0..out.emb_ms1_dim {
            row.push(format!("{}", out.emb_ms1.get(off + j).copied().unwrap_or(0.0)));
        }
        let off = i * out.emb_all_dim;
        for j in 0..out.emb_all_dim {
            row.push(format!("{}", out.emb_all.get(off + j).copied().unwrap_or(0.0)));
        }
        let off = i * out.coe_ms2_dim;
        for j in 0..out.coe_ms2_dim {
            row.push(format!("{}", out.coe_ms2.get(off + j).copied().unwrap_or(0.0)));
        }
        let off = i * out.coe_ms1_dim;
        for j in 0..out.coe_ms1_dim {
            row.push(format!("{}", out.coe_ms1.get(off + j).copied().unwrap_or(0.0)));
        }
        let off = i * out.coe_ms12_dim;
        for j in 0..out.coe_ms12_dim {
            row.push(format!("{}", out.coe_ms12.get(off + j).copied().unwrap_or(0.0)));
        }
        let off = i * out.coe_all_dim;
        for j in 0..out.coe_all_dim {
            row.push(format!("{}", out.coe_all.get(off + j).copied().unwrap_or(0.0)));
        }
        text.push_str(&row.join("\t"));
        text.push('\n');
    }

    std::fs::write(path, text)?;
    Ok(())
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
pub fn run_training(cfg: &TrainRunConfig) -> Result<TrainRunOutput> {
    let device = get_device(&cfg.device)?;

    let table = read_feature_rows(&cfg.osw_path, &cfg.osw)?;
    let rows = filter_training_rows(table.rows, &cfg.filter);
    if rows.is_empty() {
        bail!("no rows after filtering");
    }
    log_run_id_summary(&rows, &cfg.xic_path);

    let mut selected_cols = resolve_feature_cols(&table.feature_cols, &cfg.feature_select);
    let mut model_cfg = cfg.model.clone();
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
        rows_tr = subsample_train_rows_by_bag(
            rows_tr,
            cfg.train_frac,
            cfg.train_stratify_run,
            cfg.seed,
        );
        if rows_tr.is_empty() {
            bail!("no rows after train subsample");
        }
    }
    let x_tr = build_traces_for_rows(
        &rows_tr,
        &cfg.xic_path,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
    )?;
    let x_va = if rows_va.is_empty() {
        Vec::new()
    } else {
        build_traces_for_rows(
            &rows_va,
            &cfg.xic_path,
            &cfg.xic_map_path,
            &cfg.trace,
            &cfg.fetch,
        )?
    };

    let (rows_tr, x_tr) = if cfg.restrict_osw_to_xic_map {
        filter_rows_by_trace(rows_tr, x_tr, cfg.trace.total_c(), cfg.trace.l)
    } else {
        (rows_tr, x_tr)
    };
    if cfg.restrict_osw_to_xic_map && rows_tr.is_empty() {
        bail!("all training rows were dropped after XIC restriction; check run_id match and xic_path");
    }
    let (rows_va, x_va) = if cfg.restrict_osw_to_xic_map {
        filter_rows_by_trace(rows_va, x_va, cfg.trace.total_c(), cfg.trace.l)
    } else {
        (rows_va, x_va)
    };

    let pre = if model_cfg.use_heuristic_features {
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

    let y_rows: Vec<u8> = rows_tr.iter().map(|r| if r.is_decoy { 1 } else { 0 }).collect();
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
    let batches = bags_to_train_batches(bags, &device, cfg.batch_size)?;

    let mut trainer = Trainer::new(cfg.train.clone(), &model_cfg, &device)?;
    let _metrics = trainer.train_epochs(&batches, cfg.max_epochs, None)?;

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
            summ.cutoff, summ.n_targets, summ.n_decoys
        );
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

    Ok(TrainRunOutput {
        checkpoint_prefix: cfg.output_prefix.clone(),
    })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
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

    let mut x_trace = build_traces_for_rows(
        &rows,
        &cfg.xic_path,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
    )?;
    if cfg.restrict_osw_to_xic_map {
        let filtered = filter_rows_by_trace(rows, x_trace, cfg.trace.total_c(), cfg.trace.l);
        rows = filtered.0;
        x_trace = filtered.1;
    }

    let target_cols = meta.feature_cols.clone();
    let x_feat = if meta.model.use_heuristic_features {
        rows_to_feature_matrix_with_cols(
            &rows,
            &table.feature_cols,
            &target_cols,
            meta.preprocess.as_ref(),
        )
    } else {
        Vec::new()
    };
    if cfg.diagnostics.trace_summary {
        let sum = trace_summary(
            &x_trace,
            rows.len(),
            cfg.trace.total_c(),
            cfg.trace.l,
            cfg.trace.ms1_cmax,
            cfg.trace.ms2_cmax,
        );
        print_trace_summary(&sum, "infer");
        warn_if_missing_ms1(&sum, "infer");
    }

    let n = rows.len();
    let x_feat_t = candle_core::Tensor::from_vec(x_feat, (n, meta.model.feat_dim), &device)?;
    let x_trace_t = candle_core::Tensor::from_slice(
        &x_trace,
        (n, cfg.trace.total_c(), cfg.trace.l),
        &device,
    )?;
    let scores_t = score_candidates(&model, &x_feat_t, &x_trace_t, cfg.batch_size.max(1))?;
    let scores = scores_t.to_vec1::<f32>()?;

    let table_rows = build_score_table_from_rows(&rows, &scores, cfg.pep_bins);
    crate::infer::write_score_tsv(&cfg.output_tsv, &table_rows)?;

    if cfg.diagnostics.save_head_embeddings {
        let outdir = cfg
            .diagnostics
            .head_embeddings_outdir
            .clone()
            .unwrap_or_else(|| PathBuf::from("head_embeddings"));
        std::fs::create_dir_all(&outdir)?;
        let out = score_bags_with_heads_from_rows_with_cols(
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
        let out_path = outdir.join("head_embeddings.tsv");
        write_head_embeddings_tsv(&out_path, &out)?;
    }

    if let Some(osw_path) = &cfg.output_osw {
        #[cfg(feature = "io-sqlite")]
        {
            let osw_rows: Vec<OswScoreRow> = table_rows
                .iter()
                .map(|r| OswScoreRow {
                    feature_id: r.feature_id,
                    score: r.score,
                    rank: r.rank,
                    pvalue: r.pvalue,
                    qvalue: r.qvalue,
                    pep: r.pep,
                })
                .collect();
            crate::io::osw::write_score_table(osw_path, &cfg.output_table, &osw_rows)?;

            if cfg.diagnostics.rank1_disagreements {
                let outdir = cfg
                    .diagnostics
                    .rank1_outdir
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("rank1_disagreements"));
                let summ = write_rank1_disagreement_tsvs(
                    osw_path,
                    &cfg.output_table,
                    0.01,
                    &outdir,
                )?;
                log::info!(
                    "Rank1 disagreement summary: rows={} cutoff_pstc={:?} cutoff_ms2={:?}",
                    summ.rows, summ.pstc_cutoff, summ.ms2_cutoff
                );
            }
        }
    }

    Ok(InferRunOutput { n_rows: rows.len() })
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub fn run_xrun_sweep(cfg: &XrunSweepConfig) -> Result<Vec<XrunSweepRow>> {
    let device = get_device(&cfg.device)?;

    let base = checkpoint_base(&cfg.checkpoint);
    let meta = read_checkpoint_meta(&base)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = TopazBagRanker::new(vb.pp("topaz"), &meta.model)?;
    load_checkpoint_weights(&base, &mut varmap)?;

    let table = read_feature_rows(&cfg.osw_path, &cfg.osw)?;
    let rows_aligned = if meta.model.use_heuristic_features {
        align_rows_to_cols(&table.rows, &table.feature_cols, &meta.feature_cols)
    } else {
        table.rows
    };
    log_run_id_summary(&rows_aligned, &cfg.xic_path);

    let x_trace = build_traces_for_rows(
        &rows_aligned,
        &cfg.xic_path,
        &cfg.xic_map_path,
        &cfg.trace,
        &cfg.fetch,
    )?;
    let (rows_aligned, x_trace) = if cfg.restrict_osw_to_xic_map {
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
    let taus: Vec<f64> = cfg.sweep_taus.clone().unwrap_or_else(|| vec![cfg.train.tau]);

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

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
pub fn run_xrun_sweep(_cfg: &XrunSweepConfig) -> Result<Vec<XrunSweepRow>> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
pub fn run_training(_cfg: &TrainRunConfig) -> Result<TrainRunOutput> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}

#[cfg(not(all(feature = "io-sqlite", feature = "io-parquet")))]
pub fn run_inference(_cfg: &InferRunConfig) -> Result<InferRunOutput> {
    bail!("redeem-topaz built without io-sqlite/io-parquet features");
}
