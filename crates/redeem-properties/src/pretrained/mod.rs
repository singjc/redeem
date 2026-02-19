use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::{env, fs};

// Optional compile-time embedded models directory. Enabled with feature `embed-pretrained`.
#[cfg(feature = "embed-pretrained")]
use include_dir::{include_dir, Dir};

// When embedding, prefer the workspace-level `data/pretrained_models` if present.
// Use a compile-time path anchored at the crate manifest to reach the workspace data dir.
#[cfg(feature = "embed-pretrained")]
static EMBEDDED_PRETRAINED_DIR: Dir = include_dir!("data/pretrained_models");

/// Enum of known pretrained model identifiers supported by the library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PretrainedModel {
    AlphapeptdeepRtCnnLstm,
    AlphapeptdeepCcsCnnLstm,
    AlphapeptdeepMs2Bert,
    RedeemRtCnnTf,
    RedeemCcsCnnTf,
}

impl std::fmt::Display for PretrainedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PretrainedModel::AlphapeptdeepRtCnnLstm => "alphapeptdeep-rt-cnn-lstm",
            PretrainedModel::AlphapeptdeepCcsCnnLstm => "alphapeptdeep-ccs-cnn-lstm",
            PretrainedModel::AlphapeptdeepMs2Bert => "alphapeptdeep-ms2-bert",
            PretrainedModel::RedeemRtCnnTf => "redeem-rt-cnn-tf",
            PretrainedModel::RedeemCcsCnnTf => "redeem-ccs-cnn-tf",
        };
        write!(f, "{}", s)
    }
}

impl std::str::FromStr for PretrainedModel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "alphapeptdeep-rt" | "alphapeptdeep-rt-cnn-lstm" | "rt" => {
                Ok(PretrainedModel::AlphapeptdeepRtCnnLstm)
            }
            "alphapeptdeep-ccs" | "alphapeptdeep-ccs-cnn-lstm" | "ccs" => {
                Ok(PretrainedModel::AlphapeptdeepCcsCnnLstm)
            }
            "alphapeptdeep-ms2" | "alphapeptdeep-ms2-bert" | "ms2" => {
                Ok(PretrainedModel::AlphapeptdeepMs2Bert)
            }
            "redeem-rt" | "redeem-rt-cnn-tf" | "redeem-rt-cnn" => {
                Ok(PretrainedModel::RedeemRtCnnTf)
            }
            "redeem-ccs" | "redeem-ccs-cnn-tf" | "redeem-ccs-cnn" => {
                Ok(PretrainedModel::RedeemCcsCnnTf)
            }
            other => Err(anyhow::anyhow!("Unknown pretrained model name: {}", other)),
        }
    }
}

/// Load a pretrained model as a `Box<dyn ModelInterface + Send + Sync>` using the crate's
/// high-level loaders. This requires the `redeem-properties::models` types to be available.
pub fn load_pretrained_model(
    model: PretrainedModel,
    device: candle_core::Device,
) -> Result<Box<dyn crate::models::model_interface::ModelInterface + Send + Sync>> {
    // Copy to cache and get stable path
    let cached = cache_pretrained_model(model.clone())?;

    use crate::models::model_interface::ModelInterface as _ModelInterface;
    match model {
        PretrainedModel::AlphapeptdeepRtCnnLstm => {
            let m =
                <crate::models::rt_cnn_transformer_model::RTCNNTFModel as _ModelInterface>::new(
                    cached.clone(),
                    None::<PathBuf>,
                    0,
                    8,
                    4,
                    true,
                    device,
                )?;
            Ok(Box::new(m))
        }
        PretrainedModel::AlphapeptdeepCcsCnnLstm => {
            let m = <crate::models::ccs_cnn_tf_model::CCSCNNTFModel as _ModelInterface>::new(
                cached.clone(),
                None::<PathBuf>,
                0,
                8,
                4,
                true,
                device,
            )?;
            Ok(Box::new(m))
        }
        PretrainedModel::AlphapeptdeepMs2Bert => {
            let m = <crate::models::ms2_bert_model::MS2BertModel as _ModelInterface>::new(
                cached.clone(),
                None::<PathBuf>,
                0,
                0,
                0,
                true,
                device,
            )?;
            Ok(Box::new(m))
        }
        PretrainedModel::RedeemRtCnnTf => {
            let m =
                <crate::models::rt_cnn_transformer_model::RTCNNTFModel as _ModelInterface>::new(
                    cached.clone(),
                    None::<PathBuf>,
                    0,
                    8,
                    4,
                    true,
                    device,
                )?;
            Ok(Box::new(m))
        }
        PretrainedModel::RedeemCcsCnnTf => {
            let m = <crate::models::ccs_cnn_tf_model::CCSCNNTFModel as _ModelInterface>::new(
                cached.clone(),
                None::<PathBuf>,
                0,
                8,
                4,
                true,
                device,
            )?;
            Ok(Box::new(m))
        }
    }
}

impl PretrainedModel {
    /// Return the canonical filename (or subpath) for the given pretrained model.
    pub fn filename(&self) -> &'static str {
        match self {
            PretrainedModel::AlphapeptdeepRtCnnLstm => "alphapeptdeep/generic/rt.pth",
            PretrainedModel::AlphapeptdeepCcsCnnLstm => "alphapeptdeep/generic/ccs.pth",
            PretrainedModel::AlphapeptdeepMs2Bert => "alphapeptdeep/generic/ms2.pth",
            PretrainedModel::RedeemRtCnnTf => {
                "redeem/20251205_100_epochs_min_max_rt_cnn_tf.safetensors"
            }
            PretrainedModel::RedeemCcsCnnTf => {
                "redeem/20251205_500_epochs_early_stopped_100_min_max_ccs_cnn_tf.safetensors"
            }
        }
    }
}

/// Try to locate a pretrained model file on disk.
///
/// Search order:
/// 1. Directory pointed to by `REDEEM_PRETRAINED_MODELS_DIR` environment variable.
/// 2. `data/pretrained_models/` relative to the crate (useful during development).
/// 3. `data/pretrained_models/` in the current working directory.
/// 4. User cache directory: `$HOME/.local/share/redeem/models/`.
pub fn locate_pretrained_model(model: PretrainedModel) -> Result<PathBuf> {
    // If crate was built with `embed-pretrained`, try extracting embedded asset to cache first.
    #[cfg(feature = "embed-pretrained")]
    {
        if let Ok(p) = extract_embedded_model_to_cache(&model) {
            return Ok(p);
        }
    }
    // 1) Env override
    if let Ok(dir) = env::var("REDEEM_PRETRAINED_MODELS_DIR") {
        let candidate = Path::new(&dir).join(model.filename());
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    // 2) Try relative to the crate manifest dir (useful during development/running from repo)
    if let Ok(manifest) = env::var("CARGO_MANIFEST_DIR") {
        let candidate = Path::new(&manifest)
            .join("data/pretrained_models")
            .join(model.filename());
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    // 3) Try ./data/pretrained_models in current working dir
    let cwd_candidate = Path::new("data/pretrained_models").join(model.filename());
    if cwd_candidate.exists() {
        return Ok(cwd_candidate);
    }

    // 4) User cache: $HOME/.local/share/redeem/models/<filename>
    if let Ok(home) = env::var("HOME") {
        let user_candidate = Path::new(&home)
            .join(".local/share/redeem/models")
            .join(model.filename());
        if user_candidate.exists() {
            return Ok(user_candidate);
        }
    }

    Err(anyhow::anyhow!(
        "Pretrained model not found for {:?}. Try setting REDEEM_PRETRAINED_MODELS_DIR or placing the models under data/pretrained_models/",
        model
    ))
}

/// When built with `embed-pretrained`, extract the embedded file for `model` into the
/// user cache and return the path. Returns an error if the embedded resource isn't present.
#[cfg(feature = "embed-pretrained")]
fn extract_embedded_model_to_cache(model: &PretrainedModel) -> Result<PathBuf> {
    if let Some(file) = EMBEDDED_PRETRAINED_DIR.get_file(model.filename()) {
        // target cache dir: $XDG_DATA_HOME/redeem/models or fallback to $HOME/.local/share/redeem/models
        let target_base = if let Ok(xdg) = env::var("XDG_DATA_HOME") {
            PathBuf::from(xdg).join("redeem/models")
        } else if let Ok(home) = env::var("HOME") {
            PathBuf::from(home).join(".local/share/redeem/models")
        } else {
            PathBuf::from("./.redeem_models_cache")
        };

        fs::create_dir_all(&target_base).with_context(|| {
            format!("Failed to create cache directory {}", target_base.display())
        })?;

        let filename = Path::new(model.filename())
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("pretrained_model.bin");

        let dst = target_base.join(filename);
        if !dst.exists() {
            fs::write(&dst, file.contents())
                .with_context(|| format!("Failed to write embedded model to {}", dst.display()))?;
        }

        return Ok(dst);
    }

    Err(anyhow::anyhow!(
        "Embedded pretrained model not available: {}",
        model.filename()
    ))
}

/// Convenience: copy the located pretrained model into a writable cache directory and return the path.
/// This is helpful when downstream code expects a stable file path (for example loader functions).
pub fn cache_pretrained_model(model: PretrainedModel) -> Result<PathBuf> {
    let src = locate_pretrained_model(model.clone())?;
    // target cache dir: $XDG_DATA_HOME/redeem/models or fallback to $HOME/.local/share/redeem/models
    let target_base = if let Ok(xdg) = env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg).join("redeem/models")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".local/share/redeem/models")
    } else {
        PathBuf::from("./.redeem_models_cache")
    };

    fs::create_dir_all(&target_base)
        .with_context(|| format!("Failed to create cache directory {}", target_base.display()))?;

    let filename = Path::new(model.filename())
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("pretrained_model.bin");

    let dst = target_base.join(filename);
    if !dst.exists() {
        fs::copy(&src, &dst)
            .with_context(|| format!("Failed to copy {} to {}", src.display(), dst.display()))?;
    }

    Ok(dst)
}
