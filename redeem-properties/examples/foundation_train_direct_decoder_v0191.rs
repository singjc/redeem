//! v0.19.1 prefix-competitive direct spectrum-conditioned peptide decoder.
//!
//! The implementation reuses the established causal training/corpus helpers so
//! v0.19.0 and v0.19.1 share identical partition, collation, checkpoint and
//! direct-generation contracts. v0.19.1 changes only the sequence-training
//! objective by adding the current model's strongest search-legal wrong token
//! as a hard local competitor at every clean target prefix.

#[path = "foundation_train_causal.rs"]
mod causal_training;

fn main() -> anyhow::Result<()> {
    causal_training::v0191_main()
}
