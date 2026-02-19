use anyhow::{Context, Result};
use clap::ArgMatches;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::properties::util::validate_tsv_or_csv_file;
use redeem_properties::pretrained::PretrainedModel;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PropertyInferenceConfig {
    pub version: String,
    pub model_path: String,
    pub inference_data: String,
    pub output_file: String,
    pub normalization: Option<String>,
    pub model_arch: String,
    pub device: String,
    pub batch_size: usize,
    pub instrument: String,
    pub nce: i32,
    /// Optional path to a dataset whose normalization stats should be reused (e.g., training data).
    /// If set, we will compute normalization from this file and apply it to inference data,
    /// ensuring parity with training-time validation.
    pub normalization_reference_data: Option<String>,
    /// Optional explicit normalization override (min/max or mean/std). When provided,
    /// this takes priority over the reference-data path.
    pub normalization_override_type: Option<String>, // "min_max" or "z_score"
    pub normalization_override_min: Option<f32>,
    pub normalization_override_max: Option<f32>,
    pub normalization_override_mean: Option<f32>,
    pub normalization_override_std: Option<f32>,
    /// Optional explicit decoder head selection (e.g., "small", "mlp", "linear") to force
    /// a given head layout when loading checkpoints.
    pub head_type: Option<String>,
}

impl Default for PropertyInferenceConfig {
    fn default() -> Self {
        PropertyInferenceConfig {
            version: clap::crate_version!().to_string(),
            model_path: String::new(),
            inference_data: String::new(),
            output_file: String::from("redeem_inference.csv"),
            normalization: Some(String::from("min_max")),
            model_arch: String::from("rt_cnn_tf"),
            device: String::from("cpu"),
            batch_size: 64,
            instrument: String::from("QE"),
            nce: 20,
            normalization_reference_data: None,
            normalization_override_type: None,
            normalization_override_min: None,
            normalization_override_max: None,
            normalization_override_mean: None,
            normalization_override_std: None,
            head_type: None,
        }
    }
}

impl PropertyInferenceConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        let config_json = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let partial: serde_json::Value = serde_json::from_str(&config_json)?;
        let mut config = PropertyInferenceConfig::default();

        macro_rules! load_or_default {
            ($field:ident) => {
                if let Some(val) = partial.get(stringify!($field)) {
                    if let Ok(parsed) = serde_json::from_value(val.clone()) {
                        config.$field = parsed;
                    } else {
                        log::warn!(
                            "Config Invalid value for '{}', using default: {:?}",
                            stringify!($field),
                            config.$field
                        );
                    }
                } else {
                    log::warn!(
                        "Config Missing field '{}', using default: {:?}",
                        stringify!($field),
                        config.$field
                    );
                }
            };
        }

        load_or_default!(model_path);
        load_or_default!(inference_data);
        load_or_default!(output_file);
        load_or_default!(normalization);
        load_or_default!(model_arch);
        load_or_default!(device);
        load_or_default!(batch_size);
        load_or_default!(instrument);
        load_or_default!(nce);
        load_or_default!(normalization_reference_data);
        load_or_default!(normalization_override_type);
        load_or_default!(normalization_override_min);
        load_or_default!(normalization_override_max);
        load_or_default!(normalization_override_mean);
        load_or_default!(normalization_override_std);
        load_or_default!(head_type);

        // Apply CLI overrides
        if let Some(model_path) = matches.get_one::<String>("model_path") {
            config.model_path = model_path.clone();
        }
        // If a pretrained shorthand is provided, resolve it to a cached model path and override model_path
        if let Some(pre) = matches.get_one::<String>("pretrained") {
            match pre.parse::<PretrainedModel>() {
                Ok(pm) => match redeem_properties::pretrained::cache_pretrained_model(pm) {
                    Ok(p) => config.model_path = p.to_string_lossy().into_owned(),
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to resolve pretrained model '{}': {}",
                            pre,
                            e
                        ));
                    }
                },
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Invalid pretrained model name '{}': {}",
                        pre,
                        e
                    ));
                }
            }
        }
        if let Some(inference_data) = matches.get_one::<String>("inference_data") {
            validate_tsv_or_csv_file(inference_data)?;
            config.inference_data = inference_data.clone();
        } else {
            validate_tsv_or_csv_file(&config.inference_data)?;
        }
        if let Some(output_file) = matches.get_one::<String>("output_file") {
            config.output_file = output_file.clone();
        }
        // Note: the inference subcommand does not define a `model_arch` CLI arg (it's defined
        // for `train`). We intentionally skip attempting to read a CLI override here to
        // avoid mismatched-arg panics from clap. The config file value (or default) will be
        // used for `model_arch`.

        Ok(config)
    }
}
