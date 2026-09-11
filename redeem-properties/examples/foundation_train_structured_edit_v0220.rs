//! v0.22 on-policy spectrum-conditioned structured peptide editor.
//!
//! Trains directly on frozen v0.20 TRAIN top-1 hypotheses rather than synthetic
//! categorical corruption. The complete peptide may change jointly, including
//! EOS/length, before one final physical precursor-mass projection.

#[path = "foundation_train_causal.rs"]
mod causal_training;

fn main() -> anyhow::Result<()> {
    causal_training::v0220_main()
}
