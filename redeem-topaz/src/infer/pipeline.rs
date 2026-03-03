use anyhow::Result;
use std::collections::{HashMap, HashSet};
#[cfg(any(feature = "io-sqlite", feature = "io-parquet"))]
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{Device, Tensor};

use crate::building_blocks::bagging::make_bags_with_traces;
use crate::building_blocks::trace_window::extract_trace_tensor_centered;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::{build_score_table_from_rows, ScoreTableRow};
use crate::infer::score_candidates;
use crate::io::osw::FeatureRow;
#[cfg(feature = "io-sqlite")]
use crate::io::osw::OswFeatureTable;
use crate::io::xic::{PrecursorXic, TransitionTrace, XicSource};
use crate::model::topaz::TopazBagRanker;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::model::topaz::TopazConfig;

static WARNED_MISSING_MS1: AtomicBool = AtomicBool::new(false);

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceBuildConfig {
    pub l: usize,
    pub ms1_cmax: usize,
    pub ms2_cmax: usize,
    pub normalize_max: bool,
}

#[derive(Debug, Clone)]
pub struct BagScoreOutput {
    pub bag_score: Vec<f32>,
    pub bag_y: Vec<f32>,
    pub is_decoy: Vec<bool>,
    pub bag_pid: Vec<String>,
    pub winner_hidden: Vec<f32>,
    pub hidden_dim: usize,
}

impl TraceBuildConfig {
    pub fn total_c(&self) -> usize {
        self.ms1_cmax + self.ms2_cmax
    }
}

#[derive(Debug, Clone, Default)]
pub struct XicFetchConfig {
    pub ms_levels: Option<Vec<i64>>,
    pub detecting_transition: Option<i64>,
    pub decoy: Option<i64>,
}

/// Build a dense (N, D) feature matrix from OSW feature rows.
pub fn rows_to_feature_matrix(rows: &[FeatureRow], feat_dim: usize) -> Vec<f32> {
    let n = rows.len();
    let mut out = vec![0f32; n * feat_dim];
    if feat_dim == 0 {
        return out;
    }
    for (i, row) in rows.iter().enumerate() {
        let take = row.features.len().min(feat_dim);
        let dst = i * feat_dim;
        if take > 0 {
            out[dst..dst + take].copy_from_slice(&row.features[..take]);
        }
    }
    out
}

fn sort_series(series: &mut [TransitionTrace]) {
    series.sort_by(|a, b| {
        a.ordinal
            .cmp(&b.ordinal)
            .then_with(|| a.annotation.cmp(&b.annotation))
    });
}

fn split_ms1_ms2(xic: &PrecursorXic) -> (Vec<TransitionTrace>, Vec<TransitionTrace>) {
    let mut ms1 = Vec::new();
    let mut ms2 = Vec::new();
    for t in &xic.transitions {
        match t.ms_level.unwrap_or(2) {
            1 => ms1.push(t.clone()),
            _ => ms2.push(t.clone()),
        }
    }
    sort_series(&mut ms1);
    sort_series(&mut ms2);
    (ms1, ms2)
}

/// Build trace tensors for rows using an arbitrary XIC source.
///
/// Output layout: (N, C_total, L) flattened row-major.
pub fn build_trace_tensors_from_source(
    rows: &[FeatureRow],
    xic_source: &mut impl XicSource,
    cfg: &TraceBuildConfig,
) -> Result<Vec<f32>> {
    let n = rows.len();
    let c_total = cfg.total_c();
    let mut out = vec![0f32; n * c_total * cfg.l];
    if n == 0 || c_total == 0 || cfg.l == 0 {
        return Ok(out);
    }

    let mut by_run: HashMap<u64, HashSet<u64>> = HashMap::new();
    for row in rows {
        by_run
            .entry(row.run_id)
            .or_default()
            .insert(row.precursor_id);
    }

    let mut xic_by_run: HashMap<u64, HashMap<u64, PrecursorXic>> = HashMap::new();
    for (run_id, prec_set) in by_run {
        let precs: Vec<u64> = prec_set.into_iter().collect();
        if precs.is_empty() {
            continue;
        }
        let fetched = xic_source.fetch_precursors(run_id, &precs)?;
        let mut map: HashMap<u64, PrecursorXic> = HashMap::new();
        for xic in fetched {
            map.insert(xic.precursor_id, xic);
        }
        xic_by_run.insert(run_id, map);
    }

    for (i, row) in rows.iter().enumerate() {
        let mut trace_row = vec![0f32; c_total * cfg.l];
        if let Some(run_map) = xic_by_run.get(&row.run_id) {
            if let Some(xic) = run_map.get(&row.precursor_id) {
                let (ms1_series, ms2_series) = split_ms1_ms2(xic);

                let mut offset = 0usize;
                if cfg.ms1_cmax > 0 {
                    if ms1_series.is_empty()
                        && !WARNED_MISSING_MS1.swap(true, Ordering::Relaxed)
                    {
                        eprintln!(
                            "warning: missing MS1 traces for at least one precursor; padding zeros"
                        );
                    }
                    let t_ms1 = extract_trace_tensor_centered(
                        &ms1_series,
                        row.exp_rt,
                        cfg.l,
                        cfg.ms1_cmax,
                        cfg.normalize_max,
                    );
                    trace_row[offset..offset + cfg.ms1_cmax * cfg.l].copy_from_slice(&t_ms1);
                    offset += cfg.ms1_cmax * cfg.l;
                }
                let t_ms2 = extract_trace_tensor_centered(
                    &ms2_series,
                    row.exp_rt,
                    cfg.l,
                    cfg.ms2_cmax,
                    cfg.normalize_max,
                );
                trace_row[offset..offset + cfg.ms2_cmax * cfg.l].copy_from_slice(&t_ms2);
            }
        }

        let dst = i * c_total * cfg.l;
        out[dst..dst + c_total * cfg.l].copy_from_slice(&trace_row);
    }

    Ok(out)
}

/// Score candidates directly from rows + traces.
pub fn score_rows_from_rows(
    model: &TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    feat_dim: usize,
    c_total: usize,
    l: usize,
    device: &Device,
    batch_size: usize,
) -> Result<Vec<f32>> {
    let n = rows.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let x_feat = rows_to_feature_matrix(rows, feat_dim);
    let x_feat_t = Tensor::from_vec(x_feat, (n, feat_dim), device)?;
    let x_trace_t = Tensor::from_vec(x_trace.to_vec(), (n, c_total, l), device)?;
    let scores_t = score_candidates(model, &x_feat_t, &x_trace_t, batch_size.max(1))?;
    Ok(scores_t.to_vec1::<f32>()?)
}

/// Score bags from rows + traces, returning bag-level diagnostics.
pub fn score_bags_from_rows(
    model: &TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    feat_dim: usize,
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
) -> Result<BagScoreOutput> {
    let n = rows.len();
    if n == 0 {
        return Ok(BagScoreOutput {
            bag_score: Vec::new(),
            bag_y: Vec::new(),
            is_decoy: Vec::new(),
            bag_pid: Vec::new(),
            winner_hidden: Vec::new(),
            hidden_dim: 0,
        });
    }

    let x_feat = rows_to_feature_matrix(rows, feat_dim);
    let y_rows: Vec<u8> = rows.iter().map(|r| if r.is_decoy { 1 } else { 0 }).collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat,
        n,
        feat_dim,
        x_trace,
        c_total,
        l,
        &y_rows,
        &pid_rows,
        bag_k,
    );

    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;

    let b = bags.b;
    let mut bag_scores = Vec::with_capacity(b);
    let mut hidden: Vec<f32> = Vec::new();
    let mut hidden_dim = 0usize;
    let bs = batch_size.max(1);

    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = xb.narrow(0, i, take)?;
        let tb_i = tb.narrow(0, i, take)?;
        let m_i = mask.narrow(0, i, take)?;

        let (_cand, bag, win) = model.forward_bags_with_hidden(&xb_i, &tb_i, &m_i)?;
        let bag_vec = bag.to_vec1::<f32>()?;
        bag_scores.extend(bag_vec);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.get(0).map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }
        i += take;
    }

    let is_decoy: Vec<bool> = bags.y_bag.iter().map(|&y| y < 0.5).collect();
    Ok(BagScoreOutput {
        bag_score: bag_scores,
        bag_y: bags.y_bag,
        is_decoy,
        bag_pid: bags.bag_pid,
        winner_hidden: hidden,
        hidden_dim,
    })
}

#[cfg(feature = "io-parquet")]
pub fn build_trace_tensors_from_parquet(
    rows: &[FeatureRow],
    xic_path: &Path,
    cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
) -> Result<Vec<f32>> {
    let mut reader = crate::io::xic_parquet::XicParquetReader::new(xic_path);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    build_trace_tensors_from_source(rows, &mut reader, cfg)
}

#[cfg(feature = "io-sqlite")]
pub fn read_osw_features(
    path: &Path,
    cfg: &crate::io::osw::OswReadConfig,
) -> Result<OswFeatureTable> {
    crate::io::osw::read_feature_rows(path, cfg)
}

/// End-to-end inference: OSW + XIC -> candidate scores -> SCORE table.
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub fn infer_score_table_from_osw_xic(
    model: &TopazBagRanker,
    device: &Device,
    model_cfg: &TopazConfig,
    osw_path: &Path,
    xic_path: &Path,
    osw_cfg: &crate::io::osw::OswReadConfig,
    trace_cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
    batch_size: usize,
    pep_bins: usize,
) -> Result<Vec<ScoreTableRow>> {
    let table = read_osw_features(osw_path, osw_cfg)?;
    let rows = table.rows;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let feat_dim = table.feature_cols.len();
    if model_cfg.use_heuristic_features && model_cfg.feat_dim != feat_dim {
        anyhow::bail!(
            "feature dim mismatch: model expects {}, OSW has {}",
            model_cfg.feat_dim,
            feat_dim
        );
    }

    let x_feat = rows_to_feature_matrix(&rows, model_cfg.feat_dim);
    let x_trace = build_trace_tensors_from_parquet(&rows, xic_path, trace_cfg, fetch_cfg)?;

    let n = rows.len();
    let c_total = trace_cfg.total_c();
    let x_feat_t = Tensor::from_vec(x_feat, (n, model_cfg.feat_dim), device)?;
    let x_trace_t = Tensor::from_vec(x_trace, (n, c_total, trace_cfg.l), device)?;

    let scores_t = score_candidates(model, &x_feat_t, &x_trace_t, batch_size.max(1))?;
    let scores = scores_t.to_vec1::<f32>()?;

    Ok(build_score_table_from_rows(&rows, &scores, pep_bins))
}

#[cfg(all(test, feature = "io-sqlite", feature = "io-parquet"))]
mod tests {
    use super::*;
    use crate::infer::{build_score_table_from_rows, score_candidates, write_score_tsv};
    use crate::io::osw::{OswLevel, OswReadConfig};
    use crate::model::topaz::{TopazBagRanker, TopazConfig};
    use crate::building_blocks::trace_input::TraceInputMode;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::fs;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("redeem_topaz_{name}_{stamp}.tsv"));
        p
    }

    #[test]
    fn test_osw_xic_end_to_end_tsv() -> Result<()> {
        let osw_path = Path::new(
            "/home/singjc/Documents/github/PASS01508_DIAlignR_Spyo/2026026_for_ptsc_model/gold_standard_spyo.osw",
        );
        let xic_path = Path::new(
            "/home/singjc/Documents/github/PASS01508_DIAlignR_Spyo/2026026_for_ptsc_model/hroest_K120808_Strep0%PlasmaBiolRepl1_R01_SW.xic",
        );
        if !osw_path.exists() || !xic_path.exists() {
            eprintln!("skipping test_osw_xic_end_to_end_tsv: sample files not found");
            return Ok(());
        }

        let osw_cfg = OswReadConfig {
            level: OswLevel::Ms2,
            ..Default::default()
        };
        let table = read_osw_features(osw_path, &osw_cfg)?;
        if table.rows.is_empty() {
            eprintln!("skipping test_osw_xic_end_to_end_tsv: OSW has no rows");
            return Ok(());
        }

        let run_id = table.rows[0].run_id;
        let rows: Vec<FeatureRow> = table
            .rows
            .into_iter()
            .filter(|r| r.run_id == run_id)
            .take(64)
            .collect();
        let feat_dim = table.feature_cols.len();

        let cfg = TopazConfig {
            feat_dim,
            ms2_cmax: 6,
            ms1_cmax: 0,
            l: 64,
            trace_emb_dim: 8,
            mlp_hidden: vec![16],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            ..Default::default()
        };

        let trace_cfg = TraceBuildConfig {
            l: cfg.l,
            ms1_cmax: cfg.ms1_cmax,
            ms2_cmax: cfg.ms2_cmax,
            normalize_max: false,
        };
        let fetch_cfg = XicFetchConfig {
            ms_levels: Some(vec![2]),
            detecting_transition: Some(1),
            decoy: None,
        };

        let x_feat = rows_to_feature_matrix(&rows, cfg.feat_dim);
        let x_trace = build_trace_tensors_from_parquet(&rows, xic_path, &trace_cfg, &fetch_cfg)?;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        let n = rows.len();
        let x_feat_t = Tensor::from_vec(x_feat, (n, cfg.feat_dim), &device)?;
        let x_trace_t = Tensor::from_vec(x_trace, (n, trace_cfg.total_c(), trace_cfg.l), &device)?;
        let scores_t = score_candidates(&model, &x_feat_t, &x_trace_t, 128)?;
        let scores = scores_t.to_vec1::<f32>()?;

        let score_rows = build_score_table_from_rows(&rows, &scores, 10);
        let path = tmp_path("osw_xic_score");
        write_score_tsv(&path, &score_rows)?;

        let text = fs::read_to_string(&path)?;
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.len() >= 2);
        assert_eq!(lines[0], "FEATURE_ID\tSCORE\tRANK\tPVALUE\tQVALUE\tPEP");

        let _ = fs::remove_file(&path);
        Ok(())
    }
}
