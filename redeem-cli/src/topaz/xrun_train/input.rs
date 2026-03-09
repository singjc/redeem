use anyhow::Result;
use clap::ArgMatches;
use std::path::PathBuf;

use redeem_topaz::XrunSweepConfig;
use serde::{Deserialize, Serialize};

use crate::topaz::xrun::input::load_xrun_config_from_arguments;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TopazXrunTrainConfig {
    pub inner: XrunSweepConfig,
}

impl Default for TopazXrunTrainConfig {
    fn default() -> Self {
        Self {
            inner: XrunSweepConfig::default(),
        }
    }
}

impl TopazXrunTrainConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        Ok(Self {
            inner: load_xrun_config_from_arguments(config_path, matches)?,
        })
    }
}
