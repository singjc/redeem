use anyhow::Result;
use clap::ArgMatches;
use std::fs;
use std::path::PathBuf;

use redeem_topaz::XrunSweepConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TopazXrunSweepConfig {
    pub inner: XrunSweepConfig,
}

impl Default for TopazXrunSweepConfig {
    fn default() -> Self {
        Self {
            inner: XrunSweepConfig::default(),
        }
    }
}

impl TopazXrunSweepConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        Ok(Self {
            inner: load_xrun_config_from_arguments(config_path, matches)?,
        })
    }
}

pub(crate) fn load_xrun_config_from_arguments(
    config_path: &PathBuf,
    matches: &ArgMatches,
) -> Result<XrunSweepConfig> {
    let text = fs::read_to_string(config_path)?;
    let mut cfg: XrunSweepConfig = serde_json::from_str(&text)?;
    let config_dir = config_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    if let Some(p) = matches.get_one::<PathBuf>("osw_path") {
        cfg.osw_path = p.clone();
    }
    if let Some(p) = matches.get_one::<PathBuf>("xic_path") {
        cfg.xic_path = p.clone();
    }
    if let Some(values) = matches.get_many::<PathBuf>("xic_paths") {
        cfg.xic_paths = Some(values.cloned().collect());
    }
    if let Some(p) = matches.get_one::<PathBuf>("xic_map_path") {
        cfg.xic_map_path = Some(p.clone());
    }
    if let Some(p) = matches.get_one::<PathBuf>("xim_path") {
        cfg.xim_path = Some(p.clone());
    }
    if let Some(values) = matches.get_many::<PathBuf>("xim_paths") {
        cfg.xim_paths = Some(values.cloned().collect());
    }
    if let Some(p) = matches.get_one::<PathBuf>("xim_map_path") {
        cfg.xim_map_path = Some(p.clone());
    }
    if let Some(p) = matches.get_one::<PathBuf>("xic_cache_dir") {
        cfg.xic_cache_dir = Some(p.clone());
    }
    if let Some(v) = matches.get_one::<u64>("xic_cache_max_bytes") {
        cfg.xic_cache_max_bytes = Some(*v);
    }
    if let Some(p) = matches.get_one::<PathBuf>("xim_cache_dir") {
        cfg.xim_cache_dir = Some(p.clone());
    }
    if let Some(v) = matches.get_one::<u64>("xim_cache_max_bytes") {
        cfg.xim_cache_max_bytes = Some(*v);
    }
    if let Some(p) = matches.get_one::<PathBuf>("checkpoint") {
        cfg.checkpoint = p.clone();
    }
    if let Some(p) = matches.get_one::<PathBuf>("output_tsv") {
        cfg.output_tsv = p.clone();
    }
    if let Some(d) = matches.get_one::<String>("device") {
        cfg.device = d.clone();
    }
    if matches.get_flag("restrict_xic") {
        cfg.restrict_osw_to_xic_map = true;
    }

    resolve_paths_relative_to(&mut cfg, &config_dir);
    Ok(cfg)
}

fn resolve_paths_relative_to(cfg: &mut XrunSweepConfig, base: &PathBuf) {
    cfg.osw_path = resolve_relative(base, &cfg.osw_path);
    cfg.xic_path = resolve_relative(base, &cfg.xic_path);
    cfg.xic_paths = cfg
        .xic_paths
        .take()
        .map(|paths| paths.into_iter().map(|p| resolve_relative(base, &p)).collect());
    cfg.xic_map_path = cfg
        .xic_map_path
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.xim_path = cfg
        .xim_path
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.xim_paths = cfg
        .xim_paths
        .take()
        .map(|paths| paths.into_iter().map(|p| resolve_relative(base, &p)).collect());
    cfg.xim_map_path = cfg
        .xim_map_path
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.xic_cache_dir = cfg
        .xic_cache_dir
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.xim_cache_dir = cfg
        .xim_cache_dir
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.checkpoint = resolve_relative(base, &cfg.checkpoint);
    cfg.output_tsv = resolve_relative(base, &cfg.output_tsv);
}

fn resolve_relative(base: &PathBuf, path: &PathBuf) -> PathBuf {
    if path.is_absolute() {
        path.clone()
    } else {
        base.join(path)
    }
}
