use serde::{Deserialize, Serialize};

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
    pub patience: usize,
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
            patience: 3,
        }
    }
}
