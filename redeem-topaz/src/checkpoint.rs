//! Checkpoint persistence for TOPAZ base weights and XRUN sidecars.
//!
//! Base model state and cross-run calibration state are intentionally stored as
//! separate artifacts so that a trained TOPAZ network can be reused with or
//! without XRUN calibration.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::safetensors::Load;
use serde::{Deserialize, Serialize};
use zip::CompressionMethod;
use zip::ZipArchive;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

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

const BASE_WEIGHTS_ENTRY: &str = "base.safetensors";
const BASE_META_ENTRY: &str = "base.json";
const XRUN_WEIGHTS_ENTRY: &str = "xrun.safetensors";
const XRUN_META_ENTRY: &str = "xrun.json";

fn paths_for<P: AsRef<Path>>(path: P) -> (PathBuf, PathBuf) {
    let base = path.as_ref();
    (
        base.with_extension("safetensors"),
        base.with_extension("json"),
    )
}

fn xrun_paths_for<P: AsRef<Path>>(path: P) -> (PathBuf, PathBuf) {
    let base = path.as_ref();
    (
        base.with_extension("xrun.safetensors"),
        base.with_extension("xrun.json"),
    )
}

fn archive_path_for<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if path.extension().and_then(|s| s.to_str()) == Some("model") {
        path.to_path_buf()
    } else {
        path.with_extension("model")
    }
}

fn temp_path(stem: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    p.push(format!(
        "redeem_topaz_{stem}_{}_{}.{}",
        std::process::id(),
        stamp,
        ext
    ));
    p
}

fn save_varmap_to_bytes(varmap: &VarMap, stem: &str) -> Result<Vec<u8>> {
    let path = temp_path(stem, "safetensors");
    varmap.save(&path)?;
    let bytes = std::fs::read(&path)?;
    let _ = std::fs::remove_file(&path);
    Ok(bytes)
}

fn write_temp_bytes(bytes: &[u8], stem: &str, ext: &str) -> Result<PathBuf> {
    let path = temp_path(stem, ext);
    std::fs::write(&path, bytes)?;
    Ok(path)
}

fn read_archive_entries(path: &Path) -> Result<std::collections::HashMap<String, Vec<u8>>> {
    let mut out = std::collections::HashMap::new();
    if !path.exists() {
        return Ok(out);
    }
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        out.insert(entry.name().to_string(), buf);
    }
    Ok(out)
}

fn write_archive_entries(
    path: &Path,
    entries: &std::collections::HashMap<String, Vec<u8>>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = ZipWriter::new(file);
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let mut names: Vec<&String> = entries.keys().collect();
    names.sort();
    for name in names {
        writer.start_file(name, opts)?;
        writer.write_all(entries.get(name).expect("entry name from map keys"))?;
    }
    writer.finish()?;
    Ok(())
}

fn upsert_archive_entries(path: &Path, updates: &[(&str, Vec<u8>)]) -> Result<()> {
    let mut entries = read_archive_entries(path)?;
    for (name, bytes) in updates {
        entries.insert((*name).to_string(), bytes.clone());
    }
    write_archive_entries(path, &entries)
}

fn read_archive_entry(path: &Path, name: &str) -> Result<Option<Vec<u8>>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    match archive.by_name(name) {
        Ok(mut entry) => {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            Ok(Some(buf))
        }
        Err(zip::result::ZipError::FileNotFound) => Ok(None),
        Err(err) => Err(err.into()),
    }
}

fn read_bytes_from_archive_or_legacy(
    archive: &Path,
    entry_name: &str,
    legacy: &Path,
) -> Result<Vec<u8>> {
    if let Some(bytes) = read_archive_entry(archive, entry_name)? {
        return Ok(bytes);
    }
    Ok(std::fs::read(legacy)?)
}

/// Read only the base TOPAZ checkpoint metadata.
pub fn read_checkpoint_meta<P: AsRef<Path>>(path: P) -> Result<CheckpointMeta> {
    let archive = archive_path_for(path.as_ref());
    if let Some(bytes) = read_archive_entry(&archive, BASE_META_ENTRY)? {
        return serde_json::from_slice(&bytes).map_err(Into::into);
    }
    let (_weights, meta_path) = paths_for(path);
    let text = std::fs::read_to_string(&meta_path)
        .with_context(|| format!("failed to read checkpoint meta: {meta_path:?}"))?;
    let meta: CheckpointMeta = serde_json::from_str(&text)?;
    Ok(meta)
}

/// Save a base TOPAZ checkpoint into a `topaz.model` archive.
///
/// The archive is an uncompressed zip bundle containing:
/// - `base.safetensors`
/// - `base.json`
///
/// Existing XRUN entries in the same archive are preserved.
pub fn save_checkpoint<P: AsRef<Path>>(
    path: P,
    varmap: &VarMap,
    meta: &CheckpointMeta,
) -> Result<()> {
    let archive = archive_path_for(path);
    let weights = save_varmap_to_bytes(varmap, "base_ckpt")?;
    let meta_bytes = serde_json::to_vec_pretty(meta)?;
    upsert_archive_entries(
        &archive,
        &[(BASE_WEIGHTS_ENTRY, weights), (BASE_META_ENTRY, meta_bytes)],
    )
}

/// Strictly load a base TOPAZ checkpoint into an existing variable map.
pub fn load_checkpoint<P: AsRef<Path>>(path: P, varmap: &mut VarMap) -> Result<CheckpointMeta> {
    let meta = read_checkpoint_meta(&path)?;
    let archive = archive_path_for(path.as_ref());
    let (legacy_weights, _legacy_meta) = paths_for(path);
    let bytes = read_bytes_from_archive_or_legacy(&archive, BASE_WEIGHTS_ENTRY, &legacy_weights)?;
    let tmp = write_temp_bytes(&bytes, "base_load", "safetensors")?;
    let result = varmap.load(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result?;
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
    let meta = read_checkpoint_meta(&path)?;
    let archive = archive_path_for(path.as_ref());
    let (legacy_weights, _legacy_meta) = paths_for(path);
    let bytes = read_bytes_from_archive_or_legacy(&archive, BASE_WEIGHTS_ENTRY, &legacy_weights)?;
    let tmp = write_temp_bytes(&bytes, "base_partial", "safetensors")?;
    let data = unsafe { candle_core::safetensors::MmapedSafetensors::new(&tmp)? };

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
    let _ = std::fs::remove_file(&tmp);

    report.extra_in_checkpoint = available
        .into_iter()
        .filter(|name| !matched.contains(name))
        .collect();
    report.extra_in_checkpoint.sort();

    Ok((meta, report))
}

/// Save XRUN calibrator weights into the same `topaz.model` archive.
pub fn save_xrun_checkpoint<P: AsRef<Path>>(
    path: P,
    varmap: &VarMap,
    meta: &XrunCheckpointMeta,
) -> Result<()> {
    let archive = archive_path_for(path);
    let weights = save_varmap_to_bytes(varmap, "xrun_ckpt")?;
    let meta_bytes = serde_json::to_vec_pretty(meta)?;
    upsert_archive_entries(
        &archive,
        &[(XRUN_WEIGHTS_ENTRY, weights), (XRUN_META_ENTRY, meta_bytes)],
    )
}

/// Load an XRUN calibrator from `topaz.model` or the legacy split files.
pub fn load_xrun_checkpoint<P: AsRef<Path>>(
    path: P,
    varmap: &mut VarMap,
) -> Result<XrunCheckpointMeta> {
    let meta = read_xrun_checkpoint_meta(&path)?;
    let archive = archive_path_for(path.as_ref());
    let (legacy_weights, _legacy_meta) = xrun_paths_for(path);
    let bytes = read_bytes_from_archive_or_legacy(&archive, XRUN_WEIGHTS_ENTRY, &legacy_weights)?;
    let tmp = write_temp_bytes(&bytes, "xrun_load", "safetensors")?;
    let result = varmap.load(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result?;
    Ok(meta)
}

/// Read only the XRUN checkpoint metadata.
pub fn read_xrun_checkpoint_meta<P: AsRef<Path>>(path: P) -> Result<XrunCheckpointMeta> {
    let archive = archive_path_for(path.as_ref());
    if let Some(bytes) = read_archive_entry(&archive, XRUN_META_ENTRY)? {
        return serde_json::from_slice(&bytes).map_err(Into::into);
    }
    let (_weights, meta_path) = xrun_paths_for(path);
    let text = std::fs::read_to_string(meta_path)?;
    let meta: XrunCheckpointMeta = serde_json::from_str(&text)?;
    Ok(meta)
}

/// Return `true` when XRUN checkpoint content is available.
pub fn xrun_checkpoint_exists<P: AsRef<Path>>(path: P) -> bool {
    let archive = archive_path_for(path.as_ref());
    if archive.exists() {
        if let Ok(Some(_)) = read_archive_entry(&archive, XRUN_WEIGHTS_ENTRY) {
            if let Ok(Some(_)) = read_archive_entry(&archive, XRUN_META_ENTRY) {
                return true;
            }
        }
    }
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

        let _ = std::fs::remove_file(archive_path_for(&base));
        Ok(())
    }
}
