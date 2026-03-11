use anyhow::Result;
use clap::ArgMatches;
use std::fs;
use std::path::PathBuf;

use redeem_topaz::PreprocessRunConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TopazPreprocessConfig {
    pub inner: PreprocessRunConfig,
}

impl Default for TopazPreprocessConfig {
    fn default() -> Self {
        Self {
            inner: PreprocessRunConfig::default(),
        }
    }
}

impl TopazPreprocessConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        let text = fs::read_to_string(config_path)?;
        let mut cfg: TopazPreprocessConfig = serde_json::from_str(&text)?;
        let config_dir = config_path
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        if let Some(p) = matches.get_one::<PathBuf>("osw_path") {
            cfg.inner.osw_path = p.clone();
        }
        if let Some(p) = matches.get_one::<PathBuf>("xic_path") {
            cfg.inner.xic_path = p.clone();
        }
        if let Some(values) = matches.get_many::<PathBuf>("xic_paths") {
            cfg.inner.xic_paths = Some(values.cloned().collect());
        }
        if let Some(p) = matches.get_one::<PathBuf>("xic_map_path") {
            cfg.inner.xic_map_path = Some(p.clone());
        }
        if let Some(p) = matches.get_one::<PathBuf>("xim_path") {
            cfg.inner.xim_path = Some(p.clone());
        }
        if let Some(values) = matches.get_many::<PathBuf>("xim_paths") {
            cfg.inner.xim_paths = Some(values.cloned().collect());
        }
        if let Some(p) = matches.get_one::<PathBuf>("xim_map_path") {
            cfg.inner.xim_map_path = Some(p.clone());
        }
        if let Some(p) = matches.get_one::<PathBuf>("output_path") {
            cfg.inner.output_path = p.clone();
        }
        if let Some(v) = matches.get_one::<usize>("chunk_row_count") {
            cfg.inner.chunk_row_count = *v;
        }
        if let Some(p) = matches.get_one::<PathBuf>("xic_cache_dir") {
            cfg.inner.xic_cache_dir = Some(p.clone());
        }
        if let Some(v) = matches.get_one::<u64>("xic_cache_max_bytes") {
            cfg.inner.xic_cache_max_bytes = Some(*v);
        }
        if let Some(p) = matches.get_one::<PathBuf>("xim_cache_dir") {
            cfg.inner.xim_cache_dir = Some(p.clone());
        }
        if let Some(v) = matches.get_one::<u64>("xim_cache_max_bytes") {
            cfg.inner.xim_cache_max_bytes = Some(*v);
        }
        if matches.get_flag("restrict_xic") {
            cfg.inner.restrict_osw_to_xic_map = true;
        }

        resolve_paths_relative_to(&mut cfg.inner, &config_dir);
        Ok(cfg)
    }
}

fn resolve_paths_relative_to(cfg: &mut PreprocessRunConfig, base: &PathBuf) {
    cfg.osw_path = resolve_relative(base, &cfg.osw_path);
    cfg.xic_path = resolve_relative(base, &cfg.xic_path);
    cfg.xic_paths = cfg.xic_paths.take().map(|paths| {
        paths
            .into_iter()
            .map(|p| resolve_relative(base, &p))
            .collect()
    });
    cfg.xic_map_path = cfg.xic_map_path.take().map(|p| resolve_relative(base, &p));
    cfg.xim_path = cfg.xim_path.take().map(|p| resolve_relative(base, &p));
    cfg.xim_paths = cfg.xim_paths.take().map(|paths| {
        paths
            .into_iter()
            .map(|p| resolve_relative(base, &p))
            .collect()
    });
    cfg.xim_map_path = cfg.xim_map_path.take().map(|p| resolve_relative(base, &p));
    cfg.output_path = resolve_relative(base, &cfg.output_path);
    cfg.xic_cache_dir = cfg.xic_cache_dir.take().map(|p| resolve_relative(base, &p));
    cfg.xim_cache_dir = cfg.xim_cache_dir.take().map(|p| resolve_relative(base, &p));
}

fn resolve_relative(base: &PathBuf, path: &PathBuf) -> PathBuf {
    if path.is_absolute() {
        path.clone()
    } else {
        base.join(path)
    }
}
