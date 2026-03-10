use anyhow::Result;
use clap::ArgMatches;
use std::fs;
use std::path::PathBuf;

use redeem_topaz::InferRunConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TopazInferConfig {
    pub inner: InferRunConfig,
}

impl Default for TopazInferConfig {
    fn default() -> Self {
        Self {
            inner: InferRunConfig::default(),
        }
    }
}

impl TopazInferConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        let text = fs::read_to_string(config_path)?;
        let mut cfg: TopazInferConfig = serde_json::from_str(&text)?;
        let config_dir = config_path
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        if let Some(p) = try_get_one::<PathBuf>(matches, "osw_path") {
            cfg.inner.osw_path = p.clone();
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "xic_path") {
            cfg.inner.xic_path = p.clone();
        }
        if let Some(values) = try_get_many::<PathBuf>(matches, "xic_paths") {
            cfg.inner.xic_paths = Some(values.cloned().collect());
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "xic_map_path") {
            cfg.inner.xic_map_path = Some(p.clone());
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "xim_path") {
            cfg.inner.xim_path = Some(p.clone());
        }
        if let Some(values) = try_get_many::<PathBuf>(matches, "xim_paths") {
            cfg.inner.xim_paths = Some(values.cloned().collect());
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "xim_map_path") {
            cfg.inner.xim_map_path = Some(p.clone());
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "xic_cache_dir") {
            cfg.inner.xic_cache_dir = Some(p.clone());
        }
        if let Some(v) = try_get_one::<u64>(matches, "xic_cache_max_bytes") {
            cfg.inner.xic_cache_max_bytes = Some(*v);
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "xim_cache_dir") {
            cfg.inner.xim_cache_dir = Some(p.clone());
        }
        if let Some(v) = try_get_one::<u64>(matches, "xim_cache_max_bytes") {
            cfg.inner.xim_cache_max_bytes = Some(*v);
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "checkpoint") {
            cfg.inner.checkpoint = p.clone();
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "output_tsv") {
            cfg.inner.output_tsv = p.clone();
        }
        if let Some(p) = try_get_one::<PathBuf>(matches, "output_osw") {
            cfg.inner.output_osw = Some(p.clone());
        }
        if let Some(name) = try_get_one::<String>(matches, "output_table") {
            cfg.inner.output_table = name.clone();
        }
        if let Some(name) = try_get_one::<String>(matches, "output_table_base") {
            cfg.inner.output_table_base = Some(name.clone());
        }
        if let Some(name) = try_get_one::<String>(matches, "output_table_xrun") {
            cfg.inner.output_table_xrun = Some(name.clone());
        }
        if let Some(d) = try_get_one::<String>(matches, "device") {
            cfg.inner.device = d.clone();
        }
        if let Some(v) = try_get_one::<usize>(matches, "batch_size") {
            cfg.inner.batch_size = *v;
        }
        if let Some(v) = try_get_one::<usize>(matches, "pep_bins") {
            cfg.inner.pep_bins = *v;
        }
        if try_get_flag(matches, "prefetch_traces_once") {
            cfg.inner.prefetch_traces_once = true;
        }
        if try_get_flag(matches, "stream_inference") {
            cfg.inner.stream_inference = true;
        }
        if try_get_flag(matches, "xrun") {
            cfg.inner.xrun.enabled = true;
        }
        if try_get_flag(matches, "restrict_xic") {
            cfg.inner.restrict_osw_to_xic_map = true;
        }

        resolve_paths_relative_to(&mut cfg, &config_dir);

        Ok(cfg)
    }
}

fn try_get_one<'a, T: Clone + Send + Sync + 'static>(
    matches: &'a ArgMatches,
    id: &str,
) -> Option<&'a T> {
    matches.try_get_one::<T>(id).ok().flatten()
}

fn try_get_many<'a, T: Clone + Send + Sync + 'static>(
    matches: &'a ArgMatches,
    id: &str,
) -> Option<clap::parser::ValuesRef<'a, T>> {
    matches.try_get_many::<T>(id).ok().flatten()
}

fn try_get_flag(matches: &ArgMatches, id: &str) -> bool {
    matches
        .try_get_one::<bool>(id)
        .ok()
        .flatten()
        .copied()
        .unwrap_or(false)
}

fn resolve_paths_relative_to(cfg: &mut TopazInferConfig, base: &PathBuf) {
    cfg.inner.osw_path = resolve_relative(base, &cfg.inner.osw_path);
    cfg.inner.xic_path = resolve_relative(base, &cfg.inner.xic_path);
    cfg.inner.xic_paths = cfg.inner.xic_paths.take().map(|paths| {
        paths
            .into_iter()
            .map(|p| resolve_relative(base, &p))
            .collect()
    });
    cfg.inner.xic_map_path = cfg
        .inner
        .xic_map_path
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.inner.xim_path = cfg
        .inner
        .xim_path
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.inner.xim_paths = cfg.inner.xim_paths.take().map(|paths| {
        paths
            .into_iter()
            .map(|p| resolve_relative(base, &p))
            .collect()
    });
    cfg.inner.xim_map_path = cfg
        .inner
        .xim_map_path
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.inner.xic_cache_dir = cfg
        .inner
        .xic_cache_dir
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.inner.xim_cache_dir = cfg
        .inner
        .xim_cache_dir
        .take()
        .map(|p| resolve_relative(base, &p));
    cfg.inner.checkpoint = resolve_relative(base, &cfg.inner.checkpoint);
    cfg.inner.output_tsv = resolve_relative(base, &cfg.inner.output_tsv);
    cfg.inner.output_osw = cfg
        .inner
        .output_osw
        .take()
        .map(|p| resolve_relative(base, &p));
    if let Some(path) = cfg.inner.diagnostics.rank1_outdir.take() {
        cfg.inner.diagnostics.rank1_outdir = Some(resolve_relative(base, &path));
    }
    if let Some(path) = cfg.inner.diagnostics.head_embeddings_outdir.take() {
        cfg.inner.diagnostics.head_embeddings_outdir = Some(resolve_relative(base, &path));
    }
}

fn resolve_relative(base: &PathBuf, path: &PathBuf) -> PathBuf {
    if path.is_absolute() {
        path.clone()
    } else {
        base.join(path)
    }
}
