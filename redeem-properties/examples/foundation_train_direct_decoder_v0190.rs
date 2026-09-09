//! v0.19 direct spectrum-conditioned autoregressive peptide decoder executable.

#[path = "foundation_train_causal.rs"]
mod causal_training;

fn main() -> anyhow::Result<()> {
    causal_training::v0190_main()
}
