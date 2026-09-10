//! v0.21 chemistry-conditioned whole-sequence discrete diffusion/refinement.
//!
//! Reuses the frozen v0.20 spectrum/chemistry representation, trains a
//! bidirectional categorical x0 denoiser, and refines the frozen v0.20 top-1
//! hypothesis without reopening autoregressive tuning.

#[path = "foundation_train_causal.rs"]
mod causal_training;

fn main() -> anyhow::Result<()> {
    causal_training::v0210_main()
}
