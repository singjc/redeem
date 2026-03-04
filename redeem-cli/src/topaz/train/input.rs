use anyhow::Result;
use clap::ArgMatches;
use std::fs;
use std::path::PathBuf;

use redeem_topaz::{TrainRunConfig, FeatureMode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TopazTrainConfig {
    pub inner: TrainRunConfig,
}

impl Default for TopazTrainConfig {
    fn default() -> Self {
        Self { inner: TrainRunConfig::default() }
    }
}

impl TopazTrainConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        let text = fs::read_to_string(config_path)?;
        let mut cfg: TopazTrainConfig = serde_json::from_str(&text)?;

        if let Some(p) = matches.get_one::<PathBuf>("osw_path") {
            cfg.inner.osw_path = p.clone();
        }
        if let Some(p) = matches.get_one::<PathBuf>("xic_path") {
            cfg.inner.xic_path = p.clone();
        }
        if let Some(p) = matches.get_one::<PathBuf>("output_prefix") {
            cfg.inner.output_prefix = p.clone();
        }
        if let Some(d) = matches.get_one::<String>("device") {
            cfg.inner.device = d.clone();
        }
        if let Some(v) = matches.get_one::<u64>("seed") {
            cfg.inner.seed = *v;
        }
        if let Some(v) = matches.get_one::<usize>("batch_size") {
            cfg.inner.batch_size = *v;
        }
        if let Some(v) = matches.get_one::<usize>("epochs") {
            cfg.inner.max_epochs = *v;
        }
        if let Some(v) = matches.get_one::<usize>("bag_k") {
            cfg.inner.bag_k = *v;
        }
        if let Some(v) = matches.get_one::<f32>("val_frac") {
            cfg.inner.val_frac = *v;
        }
        if let Some(v) = matches.get_one::<f32>("train_frac") {
            cfg.inner.train_frac = *v;
        }
        if matches.get_flag("train_stratify_run") {
            cfg.inner.train_stratify_run = true;
        }
        if matches.get_flag("restrict_xic") {
            cfg.inner.restrict_osw_to_xic_map = true;
        }
        if let Some(mode) = matches.get_one::<String>("feature_mode") {
            cfg.inner.feature_select.mode = match mode.as_str() {
                "lib" | "default-lib" => FeatureMode::DefaultLib,
                "custom" => FeatureMode::Custom,
                "none" => FeatureMode::None,
                _ => FeatureMode::All,
            };
        }
        if let Some(values) = matches.get_many::<String>("feature_cols") {
            cfg.inner.feature_select.cols = Some(values.cloned().collect());
            cfg.inner.feature_select.mode = FeatureMode::Custom;
        }

        Ok(cfg)
    }
}
