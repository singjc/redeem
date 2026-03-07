//! Checkpoint persistence for TOPAZ base weights and XRUN sidecars.
//!
//! Base model state and cross-run calibration state are intentionally stored as
//! separate artifacts so that a trained TOPAZ network can be reused with or
//! without XRUN calibration.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::safetensors::Load;
use serde::{Deserialize, Serialize};

use candle_nn::VarMap;

use crate::config::Config;
use crate::infer::TraceBuildConfig;
use crate::model::topaz::TopazConfig;
use crate::preprocess::Preprocessor;
use crate::xrun::{XrunPredictConfig, XrunTrainConfig};

/// Metadata stored alongside the base TOPAZ checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub model: TopazConfig,
    pub train: Option<Config>,
    pub trace: Option<TraceBuildConfig>,
    pub feature_cols: Vec<String>,
    pub preprocess: Option<Preprocessor>,
    pub version: u32,
}

/// Metadata stored alongside the XRUN calibrator checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrunCheckpointMeta {
    pub train: XrunTrainConfig,
    pub predict: XrunPredictConfig,
    pub in_dim: usize,
    pub best_val: Option<f32>,
    pub version: u32,
}

/// Summary of a partial checkpoint restore.
///
/// This is primarily used during fine-tuning when the current model may add or
/// remove auxiliary heads relative to the initialization checkpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartialLoadReport {
    pub loaded: usize,
    pub missing_in_checkpoint: Vec<String>,
    pub shape_mismatch: Vec<String>,
    pub set_errors: Vec<String>,
    pub extra_in_checkpoint: Vec<String>,
}

fn paths_for<P: AsRef<Path>>(path: P) -> (PathBuf, PathBuf) {
    let base = path.as_ref();
    let weights = base.with_extension("safetensors");
    let meta = base.with_extension("json");
    (weights, meta)
}

fn xrun_paths_for<P: AsRef<Path>>(path: P) -> (PathBuf, PathBuf) {
    let base = path.as_ref();
    let weights = base.with_extension("xrun.safetensors");
    let meta = base.with_extension("xrun.json");
    (weights, meta)
}

/// Save a base TOPAZ checkpoint.
///
/// Weights are written to `*.safetensors` and metadata to `*.json`.
pub fn save_checkpoint<P: AsRef<Path>>(
    path: P,
    varmap: &VarMap,
    meta: &CheckpointMeta,
) -> Result<()> {
    let (weights, meta_path) = paths_for(path);
    if let Some(parent) = weights.parent() {
        std::fs::create_dir_all(parent)?;
    }
    varmap.save(&weights)?;
    let json = serde_json::to_string_pretty(meta)?;
    std::fs::write(meta_path, json)?;
    Ok(())
}

/// Strictly load a base TOPAZ checkpoint into an existing variable map.
pub fn load_checkpoint<P: AsRef<Path>>(path: P, varmap: &mut VarMap) -> Result<CheckpointMeta> {
    let (weights, meta_path) = paths_for(path);
    let text = std::fs::read_to_string(meta_path)?;
    let meta: CheckpointMeta = serde_json::from_str(&text)?;
    varmap.load(&weights)?;
    Ok(meta)
}

/// Load only the compatible tensors from a checkpoint.
///
/// Tensors that are missing from the checkpoint, have incompatible shapes, or
/// fail to assign are skipped and recorded in the returned report instead of
/// causing the whole load to fail.
pub fn load_checkpoint_partial<P: AsRef<Path>>(
    path: P,
    varmap: &mut VarMap,
) -> Result<(CheckpointMeta, PartialLoadReport)> {
    let (weights, meta_path) = paths_for(path);
    let text = std::fs::read_to_string(meta_path)?;
    let meta: CheckpointMeta = serde_json::from_str(&text)?;
    let data = unsafe { candle_core::safetensors::MmapedSafetensors::new(&weights)? };

    let available: HashSet<String> = data.tensors().into_iter().map(|(name, _)| name).collect();
    let mut matched = HashSet::new();
    let mut report = PartialLoadReport::default();

    let mut tensor_data = varmap.data().lock().unwrap();
    for (name, var) in tensor_data.iter_mut() {
        let Ok(view) = data.get(name) else {
            report.missing_in_checkpoint.push(name.clone());
            continue;
        };
        matched.insert(name.clone());

        let saved_shape = view.shape();
        let current_shape = var.shape().dims();
        if saved_shape != current_shape {
            report.shape_mismatch.push(format!(
                "{name} (ckpt={saved_shape:?}, current={current_shape:?})"
            ));
            continue;
        }

        let tensor = view.load(var.device())?;
        if let Err(err) = var.set(&tensor) {
            report.set_errors.push(format!("{name}: {err}"));
            continue;
        }
        report.loaded += 1;
    }
    drop(tensor_data);

    report.extra_in_checkpoint = available
        .into_iter()
        .filter(|name| !matched.contains(name))
        .collect();
    report.extra_in_checkpoint.sort();

    Ok((meta, report))
}

/// Save an XRUN calibrator sidecar.
pub fn save_xrun_checkpoint<P: AsRef<Path>>(
    path: P,
    varmap: &VarMap,
    meta: &XrunCheckpointMeta,
) -> Result<()> {
    let (weights, meta_path) = xrun_paths_for(path);
    if let Some(parent) = weights.parent() {
        std::fs::create_dir_all(parent)?;
    }
    varmap.save(&weights)?;
    let json = serde_json::to_string_pretty(meta)?;
    std::fs::write(meta_path, json)?;
    Ok(())
}

/// Load an XRUN calibrator sidecar into an existing variable map.
///
/// This is the strict variant used for inference: the current calibrator layout
/// must match the saved XRUN checkpoint exactly.
pub fn load_xrun_checkpoint<P: AsRef<Path>>(
    path: P,
    varmap: &mut VarMap,
) -> Result<XrunCheckpointMeta> {
    let (weights, meta_path) = xrun_paths_for(path);
    let text = std::fs::read_to_string(meta_path)?;
    let meta: XrunCheckpointMeta = serde_json::from_str(&text)?;
    varmap.load(&weights)?;
    Ok(meta)
}

/// Read only the XRUN sidecar metadata.
pub fn read_xrun_checkpoint_meta<P: AsRef<Path>>(path: P) -> Result<XrunCheckpointMeta> {
    let (_weights, meta_path) = xrun_paths_for(path);
    let text = std::fs::read_to_string(meta_path)?;
    let meta: XrunCheckpointMeta = serde_json::from_str(&text)?;
    Ok(meta)
}

/// Return `true` when both XRUN sidecar files exist.
pub fn xrun_checkpoint_exists<P: AsRef<Path>>(path: P) -> bool {
    let (weights, meta) = xrun_paths_for(path);
    weights.exists() && meta.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{self as nn, VarBuilder};

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
    fn test_xrun_checkpoint_roundtrip() -> Result<()> {
        let base = tmp_base("xrun_ckpt");
        let device = Device::Cpu;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _layer = nn::linear(3, 2, vb.pp("xrun.head"))?;

        let meta = XrunCheckpointMeta {
            train: XrunTrainConfig::default(),
            predict: XrunPredictConfig::default(),
            in_dim: 17,
            best_val: Some(0.123),
            version: 1,
        };
        save_xrun_checkpoint(&base, &varmap, &meta)?;
        assert!(xrun_checkpoint_exists(&base));

        let mut loaded_varmap = VarMap::new();
        let loaded_vb = VarBuilder::from_varmap(&loaded_varmap, DType::F32, &device);
        let _loaded_layer = nn::linear(3, 2, loaded_vb.pp("xrun.head"))?;
        let loaded = load_xrun_checkpoint(&base, &mut loaded_varmap)?;

        assert_eq!(loaded.in_dim, meta.in_dim);
        assert_eq!(loaded.version, meta.version);
        assert_eq!(loaded.best_val, meta.best_val);
        assert_eq!(loaded.predict.batch_size, meta.predict.batch_size);

        let _ = std::fs::remove_file(base.with_extension("xrun.safetensors"));
        let _ = std::fs::remove_file(base.with_extension("xrun.json"));
        Ok(())
    }
}
