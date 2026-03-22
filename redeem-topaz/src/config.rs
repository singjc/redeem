//! Training hyper-parameters for the base TOPAZ model.

use serde::{Deserialize, Serialize};

/// Auxiliary trace-only distillation settings.
///
/// These targets are used only as supervision during training. They are not
/// concatenated into the scorer inputs at inference time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillConfig {
    /// Heuristic columns to predict from the learned trace representation.
    pub cols: Vec<String>,
    /// Global loss weight applied to the masked regression objective.
    pub lambda: f32,
    /// Hidden layer widths for the auxiliary regressor.
    pub hidden: Vec<usize>,
    /// Dropout applied inside the auxiliary regressor.
    pub dropout: f64,
    /// Huber transition point used by the masked regression loss.
    pub huber_delta: f32,
    /// Minimum per-column standard deviation before a target is kept.
    pub min_std: f32,
}

impl DistillConfig {
    pub fn is_enabled(&self) -> bool {
        self.lambda > 0.0 && !self.cols.is_empty()
    }
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            cols: Vec::new(),
            lambda: 0.0,
            hidden: vec![128, 64],
            dropout: 0.1,
            huber_delta: 1.0,
            min_std: 1e-3,
        }
    }
}

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
    pub lambda_xic_bag: f32,
    pub lambda_xim_bag: f32,
    pub branch_aux_hidden: Vec<usize>,
    pub branch_aux_dropout: f64,
    pub lambda_topk_runner: f32,
    pub topk_runner_margin: f32,
    pub topk_runner_k: usize,
    pub max_grad_norm: f32,
    pub patience: usize,
    pub eval_every: usize,
    pub early_stop_metric: EarlyStopMetric,
    pub early_stop_qvalue: f32,
    pub distill: DistillConfig,
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
            lambda_xic_bag: 0.0,
            lambda_xim_bag: 0.0,
            branch_aux_hidden: vec![64],
            branch_aux_dropout: 0.1,
            lambda_topk_runner: 0.0,
            topk_runner_margin: 1.0,
            topk_runner_k: 3,
            max_grad_norm: 5.0,
            patience: 3,
            eval_every: 1,
            early_stop_metric: EarlyStopMetric::ValLoss,
            early_stop_qvalue: 0.01,
            distill: DistillConfig::default(),
            trainable_prefixes: Vec::new(),
            frozen_prefixes: Vec::new(),
        }
    }
}
