//! Minimal end-to-end foundation pretraining step on long-form transition rows.

use anyhow::Result;
use candle_core::Device;
use redeem_properties::foundation::{
    FoundationConfig, FoundationDatasetLoader, FoundationTableLoaderConfig, FoundationTrainer,
    FoundationTrainerConfig,
};

fn main() -> Result<()> {
    let table = concat!(
        "ModifiedPeptide\tPrecursorCharge\tFragmentType\tFragmentSeriesNumber\tProductCharge\tLibraryIntensity\tNormalizedRetentionTime\tCCS\tCollisionEnergy\tInstrument\n",
        "PEPTIDEK\t2\tb\t2\t1\t50\t31.5\t410\t27\ttimsTOF\n",
        "PEPTIDEK\t2\ty\t3\t1\t100\t31.5\t410\t27\ttimsTOF\n",
        "AGHCEWQMK\t3\tb\t3\t1\t80\t47.0\t455\t30\tQE\n",
        "AGHCEWQMK\t3\ty\t4\t2\t40\t47.0\t455\t30\tQE\n",
    );

    let model_config = FoundationConfig {
        max_sequence_len: 16,
        graph_hidden_dim: 16,
        graph_layers: 1,
        model_dim: 32,
        num_attention_heads: 4,
        transformer_ff_dim: 64,
        transformer_layers: 1,
        contrastive_dim: 16,
        ..FoundationConfig::default()
    };
    let mut loader = FoundationDatasetLoader::new(model_config.instrument_vocab_size);
    let records = loader.load_reader(
        table.as_bytes(),
        b'\t',
        &FoundationTableLoaderConfig::default(),
    )?;

    let mut trainer = FoundationTrainer::new(
        model_config,
        FoundationTrainerConfig {
            batch_size: 2,
            ..FoundationTrainerConfig::default()
        },
        Device::Cpu,
    )?;
    let metrics = trainer.train_step(&records)?;
    println!("records: {}", records.len());
    println!("total loss: {:.6}", metrics.total_loss);
    println!("contrastive loss: {:?}", metrics.contrastive_loss);
    println!("global step: {}", trainer.global_step());
    Ok(())
}
