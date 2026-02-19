//! Configuration types for models exposed by this crate.
//!
//! `ModelConfig` and `ModelType` are the canonical configuration
//! representations used by the model factory and examples. Variants for
//! XGBoost and SVM are feature-gated so they are only available when the
//! corresponding Cargo feature is enabled.
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Central configuration for models in the crate.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelConfig {
    pub learning_rate: f32,

    #[serde(flatten)]
    pub model_type: ModelType,
}

/// Supported model types and their hyper-parameters.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum ModelType {
    #[cfg(feature = "xgboost")]
    XGBoost {
        max_depth: u32,
        num_boost_round: u32,
        early_stopping_rounds: u32,
        verbose_eval: bool,
    },
    #[cfg(feature = "svm")]
    SVM {
        eps: f64,
        c: (f64, f64),
        kernel: String,
        gaussian_kernel_eps: f64,
        polynomial_kernel_constant: f64,
        polynomial_kernel_degree: f64,
    },
    GBDT {
        max_depth: u32,
        num_boost_round: u32,
        debug: bool,
        training_optimization_level: u8,
        loss_type: String,
    },
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::GBDT {
            max_depth: 6,
            num_boost_round: 3,
            debug: false,
            training_optimization_level: 2,
            loss_type: "LogLikelyhood".to_string(),
        }
    }
}

impl ModelType {}

impl FromStr for ModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gbdt" => Ok(ModelType::GBDT {
                max_depth: 6,
                num_boost_round: 3,
                debug: false,
                training_optimization_level: 2,
                loss_type: "LogLikelyhood".to_string(),
            }),
            #[cfg(feature = "xgboost")]
            "xgboost" => Ok(ModelType::XGBoost {
                max_depth: 6,
                num_boost_round: 200,
                early_stopping_rounds: 10,
                verbose_eval: false,
            }),
            #[cfg(feature = "svm")]
            "svm" => Ok(ModelType::SVM {
                eps: 0.1,
                c: (1.0, 1.0),
                kernel: "linear".to_string(),
                gaussian_kernel_eps: 0.1,
                polynomial_kernel_constant: 1.0,
                polynomial_kernel_degree: 3.0,
            }),
            _ => Err(format!(
                "Unknown model type: {}. To use xgboost or svm, please compile with `--features xgboost` or `--features svm`",
                s
            )),
        }
    }
}

impl ModelConfig {
    pub fn new(learning_rate: f32, model_type: ModelType) -> Self {
        Self {
            learning_rate,
            model_type,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            model_type: ModelType::GBDT {
                max_depth: 6,
                num_boost_round: 50,
                debug: false,
                training_optimization_level: 2,
                loss_type: "LogLikelyhood".to_string(),
            },
        }
    }
}
