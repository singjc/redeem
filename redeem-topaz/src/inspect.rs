//! Helpers for inspecting the raw XIC/XIM signals associated with OSW feature
//! rows.
//!
//! These utilities are shared by the standalone `inspect_inputs` example and
//! the HTML report generation code in `redeem-cli`.

use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::building_blocks::trace_window::nearest_index_sorted;
use crate::infer::{XicFetchConfig, XimFetchConfig};
use crate::io::osw::FeatureRow;
use crate::io::xic::PrecursorXic;
use crate::io::xic_parquet::XicParquetReader;
use crate::io::xim::FeatureXim;
use crate::io::xim_parquet::XimParquetReader;

/// Read a simple two-column `run_id -> parquet path` TSV/whitespace map.
///
/// Relative parquet paths are resolved against the directory containing the
/// mapping file.
pub fn read_run_path_map(path: &Path) -> Result<HashMap<u64, PathBuf>> {
    let text = std::fs::read_to_string(path)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(first) = parts.next() else { continue };
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
            p = base.join(p);
        }
        out.insert(run_id, p);
    }
    if out.is_empty() {
        bail!(
            "map file {:?} did not contain any usable run_id/path rows",
            path
        );
    }
    Ok(out)
}

/// Return valid chromatographic peak boundaries from an OSW feature row.
pub fn valid_rt_bounds(row: &FeatureRow) -> Option<(f32, f32)> {
    let left = row.rt_left_width?;
    let right = row.rt_right_width?;
    if !left.is_finite() || !right.is_finite() || left < 0.0 || right < 0.0 || left >= right {
        return None;
    }
    Some((left, right))
}

/// Return valid ion-mobility peak boundaries from an OSW feature row.
pub fn valid_im_bounds(row: &FeatureRow) -> Option<(f32, f32)> {
    let left = row.exp_im_left_width?;
    let right = row.exp_im_right_width?;
    if !left.is_finite() || !right.is_finite() || left < 0.0 || right < 0.0 || left >= right {
        return None;
    }
    Some((left, right))
}

/// Project raw coordinate-space boundaries into sample indices of a centered
/// fixed-width tensor window.
pub fn boundary_indices(
    coords: &[f32],
    center: f32,
    left: f32,
    right: f32,
    l: usize,
) -> Option<(usize, usize)> {
    if coords.is_empty() || l == 0 {
        return None;
    }
    let center_idx = nearest_index_sorted(coords, center) as isize;
    let start = center_idx - (l as isize) / 2;
    let left_idx = nearest_index_sorted(coords, left) as isize - start;
    let right_idx = nearest_index_sorted(coords, right) as isize - start;
    if left_idx < 0 || right_idx < 0 || left_idx >= l as isize || right_idx >= l as isize {
        return None;
    }
    Some((left_idx as usize, right_idx as usize))
}

/// Resolve the parquet path to use for one OSW run.
pub fn resolve_run_path(
    run_id: u64,
    single_path: Option<&Path>,
    run_map: Option<&HashMap<u64, PathBuf>>,
) -> Option<PathBuf> {
    run_map
        .and_then(|m| m.get(&run_id).cloned())
        .or_else(|| single_path.map(|p| p.to_path_buf()))
}

/// Fetch the raw precursor chromatogram for one feature row.
///
/// The helper mirrors the runtime fallback used by TOPAZ: it first filters by
/// the row's `RUN_ID` and, if nothing is found, retries without the run filter.
pub fn fetch_xic_for_row(
    row: &FeatureRow,
    xic_path: &Path,
    fetch_cfg: &XicFetchConfig,
) -> Result<Option<PrecursorXic>> {
    let mut reader = XicParquetReader::new(xic_path);
    reader.filter_run_id(row.run_id);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    reader.filter_precursor_id([row.precursor_id]);
    let mut out = reader.fetch()?;
    if let Some(hit) = out.pop() {
        return Ok(Some(hit));
    }

    let mut reader = XicParquetReader::new(xic_path);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    reader.filter_precursor_id([row.precursor_id]);
    let mut out = reader.fetch()?;
    Ok(out.pop())
}

/// Fetch the raw feature mobilogram for one feature row.
pub fn fetch_xim_for_row(
    row: &FeatureRow,
    xim_path: &Path,
    fetch_cfg: &XimFetchConfig,
) -> Result<Option<FeatureXim>> {
    let mut reader = XimParquetReader::new(xim_path);
    reader.filter_run_id(row.run_id);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(types) = &fetch_cfg.mobilogram_types {
        reader.filter_mobilogram_type(types.iter().map(|s| s.as_str()));
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    reader.filter_feature_id([row.feature_id]);
    let mut out = reader.fetch()?;
    if let Some(hit) = out.pop() {
        return Ok(Some(hit));
    }

    let mut reader = XimParquetReader::new(xim_path);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(types) = &fetch_cfg.mobilogram_types {
        reader.filter_mobilogram_type(types.iter().map(|s| s.as_str()));
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    reader.filter_feature_id([row.feature_id]);
    let mut out = reader.fetch()?;
    Ok(out.pop())
}

/// Fetch raw feature mobilograms for multiple OSW rows in one parquet scan.
///
/// All rows are expected to belong to the same run. The helper uses the same
/// fallback semantics as [`fetch_xim_for_row`]: it first filters by `RUN_ID`
/// and retries without the run filter if nothing is found.
pub fn fetch_xims_for_rows(
    rows: &[FeatureRow],
    xim_path: &Path,
    fetch_cfg: &XimFetchConfig,
) -> Result<HashMap<u64, FeatureXim>> {
    if rows.is_empty() {
        return Ok(HashMap::new());
    }

    let run_id = rows[0].run_id;
    let feature_ids: Vec<u64> = rows.iter().map(|row| row.feature_id).collect();

    let mut reader = XimParquetReader::new(xim_path);
    reader.filter_run_id(run_id);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(types) = &fetch_cfg.mobilogram_types {
        reader.filter_mobilogram_type(types.iter().map(|s| s.as_str()));
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    reader.filter_feature_id(feature_ids.iter().copied());
    let first = reader.fetch()?;
    if !first.is_empty() {
        return Ok(first.into_iter().map(|xim| (xim.feature_id, xim)).collect());
    }

    let mut reader = XimParquetReader::new(xim_path);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(types) = &fetch_cfg.mobilogram_types {
        reader.filter_mobilogram_type(types.iter().map(|s| s.as_str()));
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    reader.filter_feature_id(feature_ids);
    Ok(reader
        .fetch()?
        .into_iter()
        .map(|xim| (xim.feature_id, xim))
        .collect())
}

/// Return the coordinate axis from one representative XIC trace.
pub fn representative_xic_coords(
    xic: &PrecursorXic,
    ms1_cmax: usize,
    want_ms1: bool,
) -> Option<Vec<f32>> {
    let mut traces = xic.transitions.iter().collect::<Vec<_>>();
    traces.sort_by(|a, b| {
        a.ms_level
            .cmp(&b.ms_level)
            .then_with(|| a.ordinal.cmp(&b.ordinal))
            .then_with(|| a.annotation.cmp(&b.annotation))
    });
    let filtered = traces
        .into_iter()
        .filter(|t| matches!(t.ms_level, Some(1)) == want_ms1)
        .take(if want_ms1 {
            ms1_cmax.max(1)
        } else {
            usize::MAX
        });
    for trace in filtered {
        if !trace.points.is_empty() {
            return Some(trace.points.iter().map(|p| p.rt).collect());
        }
    }
    None
}

/// Return the coordinate axis from one representative XIM trace.
pub fn representative_xim_coords(
    xim: &FeatureXim,
    ms1_cmax: usize,
    want_ms1: bool,
) -> Option<Vec<f32>> {
    let mut traces = xim.traces.iter().collect::<Vec<_>>();
    traces.sort_by(|a, b| {
        a.mobilogram_type
            .cmp(&b.mobilogram_type)
            .then_with(|| a.ms_level.cmp(&b.ms_level))
            .then_with(|| a.ordinal.cmp(&b.ordinal))
            .then_with(|| a.annotation.cmp(&b.annotation))
    });
    let filtered = traces
        .into_iter()
        .filter(|t| matches!(t.ms_level, Some(1)) == want_ms1)
        .take(if want_ms1 {
            ms1_cmax.max(1)
        } else {
            usize::MAX
        });
    for trace in filtered {
        if !trace.points.is_empty() {
            return Some(trace.points.iter().map(|p| p.mobility).collect());
        }
    }
    None
}
