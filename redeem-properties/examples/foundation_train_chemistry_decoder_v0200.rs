//! v0.20 chemistry-structured spectrum-conditioned peptide decoder.
//!
//! Reuses the established causal corpus/partition/checkpoint machinery while
//! replacing the failed v0.19.1 local prefix-margin intervention with explicit
//! biochemical candidate-transition evidence and conservative suffix-mass
//! feasibility during generation.

#[path = "foundation_train_causal.rs"]
mod causal_training;

fn main() -> anyhow::Result<()> {
    causal_training::v0200_main()
}
