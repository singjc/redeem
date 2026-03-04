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
        Self { inner: InferRunConfig::default() }
    }
}

impl TopazInferConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        let text = fs::read_to_string(config_path)?;
        let mut cfg: TopazInferConfig = serde_json::from_str(&text)?;

        if let Some(p) = matches.get_one::<PathBuf>("osw_path") {
            cfg.inner.osw_path = p.clone();
        }
        if let Some(p) = matches.get_one::<PathBuf>("xic_path") {
            cfg.inner.xic_path = p.clone();
        }
        if let Some(p) = matches.get_one::<PathBuf>("xic_map_path") {
            cfg.inner.xic_map_path = Some(p.clone());
        }
        if let Some(p) = matches.get_one::<PathBuf>("checkpoint") {
            cfg.inner.checkpoint = p.clone();
        }
        if let Some(p) = matches.get_one::<PathBuf>("output_tsv") {
            cfg.inner.output_tsv = p.clone();
        }
        if let Some(p) = matches.get_one::<PathBuf>("output_osw") {
            cfg.inner.output_osw = Some(p.clone());
        }
        if let Some(d) = matches.get_one::<String>("device") {
            cfg.inner.device = d.clone();
        }
        if let Some(v) = matches.get_one::<usize>("batch_size") {
            cfg.inner.batch_size = *v;
        }
        if let Some(v) = matches.get_one::<usize>("pep_bins") {
            cfg.inner.pep_bins = *v;
        }
        if matches.get_flag("restrict_xic") {
            cfg.inner.restrict_osw_to_xic_map = true;
        }

        Ok(cfg)
    }
}
