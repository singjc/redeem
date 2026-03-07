//! XRUN data preparation and score-application helpers.

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::building_blocks::bagging::make_bags_with_traces;
#[cfg(feature = "io-sqlite")]
use crate::infer::build_score_table_from_rows;
use crate::infer::{rows_to_feature_matrix_preprocessed, rows_to_feature_matrix_with_cols};
use crate::io::osw::FeatureRow;
use crate::model_interface::BagRankerWithHiddenInterface;
use crate::preprocess::Preprocessor;
use crate::xrun::calibrator::XrunAttentionCalibrator;
use crate::xrun::sequence::build_xrun_sequences_from_bags;

#[cfg(feature = "io-sqlite")]
use crate::io::osw::ScoreRow as OswScoreRow;
#[cfg(feature = "io-sqlite")]
use std::path::Path;

/// Bag-level inputs consumed by XRUN.
#[derive(Debug, Clone)]
pub struct XrunBagData {
    pub bag_score: Vec<f32>,
    pub bag_hidden: Vec<f32>,
    pub hidden_dim: usize,
    pub bag_y: Vec<f32>,
    pub bag_pid: Vec<String>,
}

/// Prediction-time settings for applying a trained XRUN calibrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrunPredictConfig {
    pub max_runs: usize,
    pub sort_by: String, // "run" | "score"
    pub batch_size: usize,
}

impl Default for XrunPredictConfig {
    fn default() -> Self {
        Self {
            max_runs: 64,
            sort_by: "run".to_string(),
            batch_size: 256,
        }
    }
}

/// Chunked helper that scores bags and also returns winner hidden vectors.
pub fn score_bags_with_hidden_chunked(
    model: &impl BagRankerWithHiddenInterface,
    xb: &Tensor,
    tb: &Tensor,
    mask: &Tensor,
    batch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let (b, _k, _d) = xb.dims3()?;
    let mut scores = Vec::with_capacity(b);
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
        scores.extend(bag_vec);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.get(0).map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }
        i += take;
    }

    Ok((scores, hidden, hidden_dim))
}

/// Build bag-level inputs and compute `(bag_score, winner_hidden)` for XRUN.
pub fn build_xrun_bag_data_from_rows(
    model: &impl BagRankerWithHiddenInterface,
    rows: &[FeatureRow],
    x_trace: &[f32],
    feat_dim: usize,
    c_total: usize,
    l: usize,
    k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<XrunBagData> {
    let n = rows.len();
    let x_feat = rows_to_feature_matrix_preprocessed(rows, feat_dim, pre);
    let y_rows: Vec<u8> = rows
        .iter()
        .map(|r| if r.is_decoy { 1 } else { 0 })
        .collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat, n, feat_dim, x_trace, c_total, l, &y_rows, &pid_rows, k,
    );

    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;

    let (bag_score, bag_hidden, hidden_dim) =
        score_bags_with_hidden_chunked(model, &xb, &tb, &mask, batch_size)?;

    Ok(XrunBagData {
        bag_score,
        bag_hidden,
        hidden_dim,
        bag_y: bags.y_bag,
        bag_pid: bags.bag_pid,
    })
}

/// Same as [`build_xrun_bag_data_from_rows`] but with explicit feature-column
/// projection.
pub fn build_xrun_bag_data_from_rows_with_cols(
    model: &impl BagRankerWithHiddenInterface,
    rows: &[FeatureRow],
    x_trace: &[f32],
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<XrunBagData> {
    let n = rows.len();
    let d = target_cols.len();
    let x_feat = rows_to_feature_matrix_with_cols(rows, osw_cols, target_cols, pre);
    let y_rows: Vec<u8> = rows
        .iter()
        .map(|r| if r.is_decoy { 1 } else { 0 })
        .collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(&x_feat, n, d, x_trace, c_total, l, &y_rows, &pid_rows, k);

    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;

    let (bag_score, bag_hidden, hidden_dim) =
        score_bags_with_hidden_chunked(model, &xb, &tb, &mask, batch_size)?;

    Ok(XrunBagData {
        bag_score,
        bag_hidden,
        hidden_dim,
        bag_y: bags.y_bag,
        bag_pid: bags.bag_pid,
    })
}

/// Predict per-bag deltas using a trained calibrator.
pub fn xrun_predict_deltas_for_bags(
    calibrator: &XrunAttentionCalibrator,
    bag_data: &XrunBagData,
    cfg: &XrunPredictConfig,
    device: &Device,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let seq = build_xrun_sequences_from_bags(
        &bag_data.bag_pid,
        &bag_data.bag_score,
        &bag_data.bag_hidden,
        bag_data.hidden_dim,
        &bag_data.bag_y,
        cfg.max_runs,
        &cfg.sort_by,
    );

    let p = seq.p;
    let r = seq.r;
    let din = seq.din;
    let x = Tensor::from_vec(seq.xseq.clone(), (p, r, din), device)?;
    let mask_u8: Vec<u8> = seq.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let m = Tensor::from_vec(mask_u8, (p, r), device)?;

    let mut delta_all = vec![0f32; p * r];
    let mut ent = vec![0f32; p];
    let bs = cfg.batch_size.max(1);

    let mut s = 0usize;
    while s < p {
        let take = (p - s).min(bs);
        let xb = x.narrow(0, s, take)?;
        let mb = m.narrow(0, s, take)?;
        let (dlt, attn) = calibrator.forward_masked(&xb, &mb)?;
        let d_vec = dlt.to_vec2::<f32>()?;
        let a_vec = attn.to_vec2::<f32>()?;
        for i in 0..take {
            for j in 0..r {
                delta_all[(s + i) * r + j] = d_vec[i][j];
            }
            let mut e = 0f32;
            for &a in &a_vec[i] {
                if a > 0.0 {
                    e -= a * a.ln();
                }
            }
            ent[s + i] = e;
        }
        s += take;
    }

    let b = bag_data.bag_pid.len();
    let mut delta_bag = vec![0f32; b];
    for pi in 0..p {
        for ri in 0..r {
            let bi = seq.idx_mat[pi * r + ri];
            if bi >= 0 {
                delta_bag[bi as usize] = delta_all[pi * r + ri];
            }
        }
    }

    Ok((delta_bag, ent))
}

/// Apply per-bag deltas to bag scores.
pub fn apply_xrun_deltas(bag_score: &[f32], delta_bag: &[f32]) -> Vec<f32> {
    let n = bag_score.len().min(delta_bag.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(bag_score[i] + delta_bag[i]);
    }
    out
}

/// Apply per-bag deltas to per-row (candidate) scores using `group_id`
/// mapping.
pub fn apply_xrun_deltas_to_rows(
    row_scores: &[f32],
    rows: &[FeatureRow],
    bag_pid: &[String],
    delta_bag: &[f32],
) -> Vec<f32> {
    let mut map = std::collections::HashMap::new();
    for (pid, dlt) in bag_pid.iter().zip(delta_bag.iter()) {
        map.insert(pid.as_str(), *dlt);
    }
    let mut out = Vec::with_capacity(row_scores.len());
    for (i, row) in rows.iter().enumerate() {
        let base = row_scores.get(i).copied().unwrap_or(0.0);
        let dlt = map.get(row.group_id.as_str()).copied().unwrap_or(0.0);
        out.push(base + dlt);
    }
    out
}

/// Write XRUN-calibrated scores to an OSW score table.
#[cfg(feature = "io-sqlite")]
pub fn write_xrun_scores_to_osw(
    osw_path: &Path,
    table_name: &str,
    rows: &[FeatureRow],
    row_scores: &[f32],
    bag_pid: &[String],
    delta_bag: &[f32],
    pep_bins: usize,
) -> Result<Vec<f32>> {
    let scores_x = apply_xrun_deltas_to_rows(row_scores, rows, bag_pid, delta_bag);
    let table = build_score_table_from_rows(rows, &scores_x, pep_bins);
    let osw_rows: Vec<OswScoreRow> = table
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
    crate::io::osw::write_score_table(osw_path, table_name, &osw_rows)?;
    Ok(scores_x)
}

#[cfg(all(test, feature = "io-sqlite"))]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::fs;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("redeem_xrun_{name}_{stamp}.osw"));
        p
    }

    #[test]
    fn test_write_xrun_scores_to_osw_smoke() -> Result<()> {
        let path = tmp_path("writeback");
        let rows = vec![
            FeatureRow {
                feature_id: 1,
                precursor_id: 10,
                run_id: 1,
                group_id: "1_10".to_string(),
                exp_rt: 0.0,
                is_decoy: false,
                features: vec![],
            },
            FeatureRow {
                feature_id: 2,
                precursor_id: 10,
                run_id: 1,
                group_id: "1_10".to_string(),
                exp_rt: 0.0,
                is_decoy: true,
                features: vec![],
            },
        ];
        let row_scores = vec![0.1f32, 0.2];
        let bag_pid = vec!["1_10".to_string()];
        let delta_bag = vec![0.5f32];

        let out = write_xrun_scores_to_osw(
            &path,
            "SCORE_XRUN",
            &rows,
            &row_scores,
            &bag_pid,
            &delta_bag,
            5,
        )?;
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.6).abs() < 1e-6);

        let conn = Connection::open(&path)?;
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM SCORE_XRUN")?;
        let count: i64 = stmt.query_row([], |r| r.get(0))?;
        assert_eq!(count, 2);

        let mut stmt = conn.prepare("SELECT SCORE FROM SCORE_XRUN WHERE FEATURE_ID = 1")?;
        let score: f64 = stmt.query_row([], |r| r.get(0))?;
        assert!((score as f32 - 0.6).abs() < 1e-6);

        let _ = fs::remove_file(&path);
        Ok(())
    }
}
