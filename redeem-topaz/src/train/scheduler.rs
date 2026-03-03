#[derive(Debug, Clone)]
pub struct CosineWarmupScheduler {
    pub base_lr: f64,
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub min_lr_ratio: f64,
}

impl CosineWarmupScheduler {
    pub fn new(
        base_lr: f64,
        total_steps: usize,
        warmup_frac: f64,
        warmup_steps: Option<usize>,
        min_lr_ratio: f64,
    ) -> Self {
        let total_steps = total_steps.max(1);
        let wu = if let Some(ws) = warmup_steps {
            ws.max(1)
        } else {
            ((warmup_frac * total_steps as f64).round() as usize).max(1)
        };
        let wu = wu.min(total_steps.max(1));
        let min_lr_ratio = min_lr_ratio.max(0.0).min(1.0);
        Self { base_lr, total_steps, warmup_steps: wu, min_lr_ratio }
    }

    /// Step is 0-based. Mirrors Python lambda schedule.
    pub fn lr_at_step(&self, step: usize) -> f64 {
        let step = step.min(self.total_steps.saturating_sub(1));
        if step < self.warmup_steps {
            return (step + 1) as f64 / self.warmup_steps as f64 * self.base_lr;
        }
        if self.total_steps <= self.warmup_steps {
            return self.base_lr;
        }
        let prog = (step - self.warmup_steps) as f64
            / (self.total_steps - self.warmup_steps) as f64;
        let prog = prog.max(0.0).min(1.0);
        let cos = 0.5 * (1.0 + (std::f64::consts::PI * prog).cos());
        let scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos;
        self.base_lr * scale
    }
}
