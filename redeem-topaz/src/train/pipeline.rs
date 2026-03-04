use std::collections::{HashMap, HashSet};

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::building_blocks::bagging::{make_bags_with_traces, Bags};
use crate::infer::rows_to_feature_matrix;
use crate::io::osw::FeatureRow;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::train::trainer::TrainBatch;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use anyhow::Result;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use candle_core::{Device, Tensor};
use crate::preprocess::Preprocessor;

#[derive(Debug, Clone, Default)]
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
        rows.into_iter().filter(|r| set.contains(&r.run_id)).collect()
    } else {
        rows
    }
}

fn limit_precursors(rows: Vec<FeatureRow>, max_precursors: Option<usize>, seed: u64) -> Vec<FeatureRow> {
    let Some(max_p) = max_precursors else { return rows };
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
    let keep: HashSet<u64> = idxs
        .into_iter()
        .take(max_p)
        .map(|i| precs[i])
        .collect();
    rows.into_iter().filter(|r| keep.contains(&r.precursor_id)).collect()
}

fn limit_precursors_per_run(
    rows: Vec<FeatureRow>,
    max_precursors_per_run: Option<usize>,
    seed: u64,
) -> Vec<FeatureRow> {
    let Some(max_p) = max_precursors_per_run else { return rows };
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
            let keep: HashSet<u64> = idxs
                .into_iter()
                .take(max_p)
                .map(|i| precs[i])
                .collect();
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

/// Fit preprocessing statistics on the provided rows.
pub fn fit_preprocessor_from_rows(rows: &[FeatureRow], feat_dim: usize) -> Preprocessor {
    let x = rows_to_feature_matrix(rows, feat_dim);
    Preprocessor::fit(&x, rows.len(), feat_dim)
}

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
fn bags_to_train_batches(
    bags: Bags,
    device: &Device,
    batch_size: usize,
) -> Result<Vec<TrainBatch>> {
    let b = bags.b;
    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
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
        let m_i = mask.narrow(0, i, take)?;
        let yb_i = yb.narrow(0, i, take)?;
        out.push(TrainBatch { xb: xb_i, tb: tb_i, mask: m_i, yb: yb_i });
        i += take;
    }
    Ok(out)
}

/// Build training batches from OSW + XIC with filtering support.
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
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

    let x_trace = crate::infer::build_trace_tensors_from_parquet(&rows, xic_path, trace_cfg, fetch_cfg)?;
    let n = rows.len();
    let c_total = trace_cfg.total_c();
    let x_feat = crate::infer::rows_to_feature_matrix_preprocessed(
        &rows,
        model_cfg.feat_dim,
        pre,
    );

    let y_rows: Vec<u8> = rows.iter().map(|r| if r.is_decoy { 1 } else { 0 }).collect();
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
