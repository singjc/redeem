//! Training hyper-parameters for the base TOPAZ model.

use serde::{Deserialize, Serialize};

/// Validation metric used for checkpoint selection and early stopping.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EarlyStopMetric {
    ValLoss,
    ValTdcTargets,
}

impl Default for EarlyStopMetric {
    fn default() -> Self {
        Self::ValLoss
    }
}

/// Training-time bag pooling used by the main bag-level objectives.
///
/// Inference remains hard masked-max. This only affects how the trainer
/// aggregates candidate logits into a bag logit for optimization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainBagPoolMode {
    Max,
    SoftmaxMean,
}

impl Default for TrainBagPoolMode {
    fn default() -> Self {
        Self::Max
    }
}

/// Base-model optimization and auxiliary-loss configuration.
///
/// This struct intentionally excludes data-loading and trace extraction
/// settings; those belong to [`crate::run::TrainRunConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub use_lr_scheduler: bool,
    pub warmup_frac: f32,
    pub warmup_steps: Option<usize>,
    pub min_lr_ratio: f32,
    pub lambda_pair: f32,
    pub lambda_inbag: f32,
    pub inbag_margin: f32,
    pub lambda_winner_margin: f32,
    pub winner_margin: f32,
    pub lambda_ms12: f32,
    pub ms12_soft_temp: f32,
    pub bag_pool: TrainBagPoolMode,
    pub bag_pool_temp: f32,
    pub max_grad_norm: f32,
    pub patience: usize,
    pub eval_every: usize,
    pub early_stop_metric: EarlyStopMetric,
    pub early_stop_qvalue: f32,
    pub trainable_prefixes: Vec<String>,
    pub frozen_prefixes: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 1e-2,
            use_lr_scheduler: true,
            warmup_frac: 0.05,
            warmup_steps: None,
            min_lr_ratio: 0.05,
            lambda_pair: 0.1,
            lambda_inbag: 0.1,
            inbag_margin: 1.0,
            lambda_winner_margin: 0.0,
            winner_margin: 1.0,
            lambda_ms12: 0.0,
            ms12_soft_temp: 1.0,
            bag_pool: TrainBagPoolMode::Max,
            bag_pool_temp: 1.0,
            max_grad_norm: 5.0,
            patience: 3,
            eval_every: 1,
            early_stop_metric: EarlyStopMetric::ValLoss,
            early_stop_qvalue: 0.01,
            trainable_prefixes: Vec::new(),
            frozen_prefixes: Vec::new(),
        }
    }
}
