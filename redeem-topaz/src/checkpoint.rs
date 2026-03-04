use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use candle_nn::VarMap;

use crate::config::Config;
use crate::infer::TraceBuildConfig;
use crate::model::topaz::TopazConfig;
use crate::preprocess::Preprocessor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub model: TopazConfig,
    pub train: Option<Config>,
    pub trace: Option<TraceBuildConfig>,
    pub feature_cols: Vec<String>,
    pub preprocess: Option<Preprocessor>,
    pub version: u32,
}

fn paths_for<P: AsRef<Path>>(path: P) -> (PathBuf, PathBuf) {
    let base = path.as_ref();
    let weights = base.with_extension("safetensors");
    let meta = base.with_extension("json");
    (weights, meta)
}

/// Save weights to `.safetensors` + metadata to `.json`.
pub fn save_checkpoint<P: AsRef<Path>>(path: P, varmap: &VarMap, meta: &CheckpointMeta) -> Result<()> {
    let (weights, meta_path) = paths_for(path);
    varmap.save(&weights)?;
    let json = serde_json::to_string_pretty(meta)?;
    std::fs::write(meta_path, json)?;
    Ok(())
}

/// Load metadata and populate an existing VarMap from `.safetensors`.
pub fn load_checkpoint<P: AsRef<Path>>(path: P, varmap: &mut VarMap) -> Result<CheckpointMeta> {
    let (weights, meta_path) = paths_for(path);
    let text = std::fs::read_to_string(meta_path)?;
    let meta: CheckpointMeta = serde_json::from_str(&text)?;
    varmap.load(&weights)?;
    Ok(meta)
}
