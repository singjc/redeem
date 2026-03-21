//! Dataset filtering, splitting, preprocessing, and batch assembly for TOPAZ.

use std::collections::{HashMap, HashSet};

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::building_blocks::bagging::{Bags, make_bags_with_traces};
use crate::infer::{rows_to_feature_matrix, rows_to_feature_matrix_with_cols};
use crate::io::osw::FeatureRow;
use crate::preprocess::Preprocessor;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::train::trainer::TrainBatch;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use anyhow::Result;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

/// Row-level filters applied before train/validation splitting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainFilter {
    /// Keep only these run IDs (if provided).
    pub run_ids: Option<Vec<u64>>,
    /// Limit total unique precursors across all runs.
    pub max_precursors: Option<usize>,
    /// Limit unique precursors per run.
    pub max_precursors_per_run: Option<usize>,
    /// Random seed for sampling.
    pub seed: u64,
}

fn shuffle_indices(idxs: &mut [usize], seed: u64) {
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    for i in (1..idxs.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        idxs.swap(i, j);
    }
}

fn filter_rows_by_runs(rows: Vec<FeatureRow>, run_ids: &Option<Vec<u64>>) -> Vec<FeatureRow> {
    if let Some(runs) = run_ids {
        let set: HashSet<u64> = runs.iter().copied().collect();
        rows.into_iter()
            .filter(|r| set.contains(&r.run_id))
            .collect()
    } else {
        rows
    }
}

fn limit_precursors(
    rows: Vec<FeatureRow>,
    max_precursors: Option<usize>,
    seed: u64,
) -> Vec<FeatureRow> {
    let Some(max_p) = max_precursors else {
        return rows;
    };
    if max_p == 0 {
        return Vec::new();
    }
    let mut precs: Vec<u64> = rows.iter().map(|r| r.precursor_id).collect();
    precs.sort_unstable();
    precs.dedup();
    if precs.len() <= max_p {
        return rows;
    }
    let mut idxs: Vec<usize> = (0..precs.len()).collect();
    shuffle_indices(&mut idxs, seed);
    let keep: HashSet<u64> = idxs.into_iter().take(max_p).map(|i| precs[i]).collect();
    rows.into_iter()
        .filter(|r| keep.contains(&r.precursor_id))
        .collect()
}

fn limit_precursors_per_run(
    rows: Vec<FeatureRow>,
    max_precursors_per_run: Option<usize>,
    seed: u64,
) -> Vec<FeatureRow> {
    let Some(max_p) = max_precursors_per_run else {
        return rows;
    };
    if max_p == 0 {
        return Vec::new();
    }
    let mut by_run: HashMap<u64, Vec<FeatureRow>> = HashMap::new();
    for r in rows {
        by_run.entry(r.run_id).or_default().push(r);
    }
    let mut out = Vec::new();
    for (run_id, mut rs) in by_run {
        let mut precs: Vec<u64> = rs.iter().map(|r| r.precursor_id).collect();
        precs.sort_unstable();
        precs.dedup();
        if precs.len() > max_p {
            let mut idxs: Vec<usize> = (0..precs.len()).collect();
            shuffle_indices(&mut idxs, seed ^ run_id);
            let keep: HashSet<u64> = idxs.into_iter().take(max_p).map(|i| precs[i]).collect();
            rs.retain(|r| keep.contains(&r.precursor_id));
        }
        out.extend(rs);
    }
    out
}

/// Apply run/precursor filters for training.
pub fn filter_training_rows(rows: Vec<FeatureRow>, filt: &TrainFilter) -> Vec<FeatureRow> {
    let rows = filter_rows_by_runs(rows, &filt.run_ids);
    let rows = limit_precursors_per_run(rows, filt.max_precursors_per_run, filt.seed);
    limit_precursors(rows, filt.max_precursors, filt.seed)
}

/// Subsample training rows by bag (`group_id`), optionally stratified by run.
pub fn subsample_train_rows_by_bag(
    rows: Vec<FeatureRow>,
    frac: f32,
    stratify_run: bool,
    seed: u64,
) -> Vec<FeatureRow> {
    if rows.is_empty() {
        return rows;
    }
    if frac >= 1.0 {
        return rows;
    }
    if frac <= 0.0 {
        return Vec::new();
    }

    let mut bag_ids: Vec<String> = Vec::new();
    let mut bag_run: Vec<u64> = Vec::new();
    let mut bag_is_decoy: Vec<bool> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for r in &rows {
        if !seen.contains_key(&r.group_id) {
            let idx = bag_ids.len();
            seen.insert(r.group_id.clone(), idx);
            bag_ids.push(r.group_id.clone());
            bag_run.push(r.run_id);
            bag_is_decoy.push(r.is_decoy);
        }
    }

    let n_bags = bag_ids.len();
    if n_bags == 0 {
        return Vec::new();
    }

    let mut keep_idx: Vec<usize> = Vec::new();
    if stratify_run {
        let mut by_run: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, &run_id) in bag_run.iter().enumerate() {
            by_run.entry(run_id).or_default().push(i);
        }
        for (run_id, mut idxs) in by_run {
            if idxs.is_empty() {
                continue;
            }
            shuffle_indices(&mut idxs, seed ^ run_id);
            let n_keep = ((idxs.len() as f32) * frac).ceil() as usize;
            let n_keep = n_keep.max(1).min(idxs.len());
            keep_idx.extend_from_slice(&idxs[..n_keep]);
        }
    } else {
        let mut idxs: Vec<usize> = (0..n_bags).collect();
        shuffle_indices(&mut idxs, seed);
        let n_keep = ((n_bags as f32) * frac).ceil() as usize;
        let n_keep = n_keep.max(1).min(n_bags);
        keep_idx.extend_from_slice(&idxs[..n_keep]);
    }

    keep_idx.sort_unstable();
    keep_idx.dedup();

    let mut kept_targets = 0usize;
    let mut kept_decoys = 0usize;
    for &i in &keep_idx {
        if bag_is_decoy[i] {
            kept_decoys += 1;
        } else {
            kept_targets += 1;
        }
    }
    log::info!(
        "Train subsample: kept_bags={}/{} (targets={} decoys={}) frac={}",
        keep_idx.len(),
        n_bags,
        kept_targets,
        kept_decoys,
        frac
    );

    let keep_set: HashSet<&str> = keep_idx.iter().map(|&i| bag_ids[i].as_str()).collect();
    rows.into_iter()
        .filter(|r| keep_set.contains(r.group_id.as_str()))
        .collect()
}

/// Split rows by unique `precursor_id` to avoid train/validation leakage across
/// candidates from the same precursor.
pub fn split_rows_by_precursor(
    rows: &[FeatureRow],
    val_frac: f32,
    seed: u64,
) -> (Vec<FeatureRow>, Vec<FeatureRow>) {
    let mut precs: Vec<u64> = rows.iter().map(|r| r.precursor_id).collect();
    precs.sort_unstable();
    precs.dedup();
    if precs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut idxs: Vec<usize> = (0..precs.len()).collect();
    shuffle_indices(&mut idxs, seed);
    let frac = val_frac.max(0.0).min(1.0);
    let mut n_val = ((precs.len() as f32) * frac).floor() as usize;
    if precs.len() > 1 {
        n_val = n_val.clamp(1, precs.len() - 1);
    } else {
        n_val = 0;
    }
    let val_set: std::collections::HashSet<u64> =
        idxs.into_iter().take(n_val).map(|i| precs[i]).collect();
    let mut tr = Vec::new();
    let mut va = Vec::new();
    for r in rows {
        if val_set.contains(&r.precursor_id) {
            va.push(r.clone());
        } else {
            tr.push(r.clone());
        }
    }
    (tr, va)
}

/// Fit preprocessing statistics on the provided rows.
pub fn fit_preprocessor_from_rows(rows: &[FeatureRow], feat_dim: usize) -> Preprocessor {
    let x = rows_to_feature_matrix(rows, feat_dim);
    Preprocessor::fit(&x, rows.len(), feat_dim)
}

/// Fit preprocessing stats aligned to a target column order.
pub fn fit_preprocessor_from_rows_with_cols(
    rows: &[FeatureRow],
    osw_cols: &[String],
    target_cols: &[String],
) -> Preprocessor {
    let d = target_cols.len();
    let x = rows_to_feature_matrix_with_cols(rows, osw_cols, target_cols, None);
    Preprocessor::fit(&x, rows.len(), d)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Convert flattened bag buffers into a vector of tensor mini-batches.
pub fn bags_to_train_batches(
    bags: Bags,
    device: &Device,
    batch_size: usize,
) -> Result<Vec<TrainBatch>> {
    bags_to_train_batches_with_aux_and_distill(bags, None, None, device, batch_size)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Convert flattened bag buffers plus an optional auxiliary signal tensor into
/// a vector of tensor mini-batches.
pub fn bags_to_train_batches_with_aux(
    bags: Bags,
    aux: Option<(Vec<f32>, usize, usize)>,
    device: &Device,
    batch_size: usize,
) -> Result<Vec<TrainBatch>> {
    bags_to_train_batches_with_aux_and_distill(bags, aux, None, device, batch_size)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Convert flattened bag buffers, an optional auxiliary tensor, and optional
/// distillation targets into a vector of tensor mini-batches.
pub fn bags_to_train_batches_with_aux_and_distill(
    bags: Bags,
    aux: Option<(Vec<f32>, usize, usize)>,
    distill: Option<(Vec<f32>, Vec<f32>, usize)>,
    device: &Device,
    batch_size: usize,
) -> Result<Vec<TrainBatch>> {
    let b = bags.b;
    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let tb_aux = if let Some((data, c_aux, l_aux)) = aux {
        Some(Tensor::from_vec(
            data,
            (bags.b, bags.k, c_aux, l_aux),
            device,
        )?)
    } else {
        None
    };
    let (distill_targets, distill_mask) = if let Some((targets, mask, d_distill)) = distill {
        (
            Some(Tensor::from_vec(
                targets,
                (bags.b, bags.k, d_distill),
                device,
            )?),
            Some(Tensor::from_vec(mask, (bags.b, bags.k, d_distill), device)?),
        )
    } else {
        (None, None)
    };
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;
    let yb = Tensor::from_vec(bags.y_bag, (bags.b,), device)?;

    let bs = batch_size.max(1);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = xb.narrow(0, i, take)?;
        let tb_i = tb.narrow(0, i, take)?;
        let tb_aux_i = if let Some(tb_aux) = tb_aux.as_ref() {
            Some(tb_aux.narrow(0, i, take)?)
        } else {
            None
        };
        let distill_targets_i = if let Some(targets) = distill_targets.as_ref() {
            Some(targets.narrow(0, i, take)?)
        } else {
            None
        };
        let distill_mask_i = if let Some(mask) = distill_mask.as_ref() {
            Some(mask.narrow(0, i, take)?)
        } else {
            None
        };
        let m_i = mask.narrow(0, i, take)?;
        let yb_i = yb.narrow(0, i, take)?;
        out.push(TrainBatch {
            xb: xb_i,
            tb: tb_i,
            tb_aux: tb_aux_i,
            mask: m_i,
            yb: yb_i,
            distill_targets: distill_targets_i,
            distill_mask: distill_mask_i,
        });
        i += take;
    }
    Ok(out)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
/// Build training batches directly from OSW and XIC files.
pub fn build_train_batches_from_osw_xic(
    osw_path: &std::path::Path,
    xic_path: &std::path::Path,
    osw_cfg: &crate::io::osw::OswReadConfig,
    trace_cfg: &crate::infer::TraceBuildConfig,
    fetch_cfg: &crate::infer::XicFetchConfig,
    model_cfg: &crate::model::topaz::TopazConfig,
    pre: Option<&Preprocessor>,
    filt: &TrainFilter,
    bag_k: usize,
    batch_size: usize,
    device: &Device,
) -> Result<Vec<TrainBatch>> {
    let table = crate::io::osw::read_feature_rows(osw_path, osw_cfg)?;
    let rows = filter_training_rows(table.rows, filt);
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let x_trace =
        crate::infer::build_trace_tensors_from_parquet(&rows, xic_path, trace_cfg, fetch_cfg)?;
    let n = rows.len();
    let c_total = trace_cfg.total_c();
    let x_feat = crate::infer::rows_to_feature_matrix_preprocessed(&rows, model_cfg.feat_dim, pre);

    let y_rows: Vec<u8> = rows
        .iter()
        .map(|r| if r.is_decoy { 1 } else { 0 })
        .collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat,
        n,
        model_cfg.feat_dim,
        &x_trace,
        c_total,
        trace_cfg.l,
        &y_rows,
        &pid_rows,
        bag_k,
    );

    bags_to_train_batches(bags, device, batch_size)
}
