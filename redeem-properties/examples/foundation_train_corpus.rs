//! Train/resume a multi-source foundation run from one YAML configuration.
//!
//! This example is the staging surface for future `redeem-cli` integration.

use anyhow::{bail, Result};
use candle_core::Device;
use redeem_properties::foundation::{
    read_foundation_training_run_config, run_foundation_pretraining,
};
use std::env;

fn main() -> Result<()> {
    let Some(config_path) = env::args().nth(1) else {
        bail!("usage: foundation_train_corpus <training.yaml>");
    };
    let config = read_foundation_training_run_config(&config_path)?;
    let summary = run_foundation_pretraining(&config, Device::Cpu)?;

    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        summary.corpus_fingerprint
    );
    println!("corpus_records\t{}", summary.corpus_records);
    println!("train_records\t{}", summary.train_records);
    println!("validation_records\t{}", summary.validation_records);
    println!("test_records\t{}", summary.test_records);
    println!("resumed\t{}", summary.resumed);
    println!(
        "train_sampling_records\t{}",
        summary.train_sampling_preview.indices.len()
    );
    println!(
        "train_sampling_unique_records\t{}",
        summary.train_sampling_preview.unique_records
    );
    for (source, count) in &summary.train_sampling_preview.source_records {
        println!("train_sampling_source\t{source}\t{count}");
    }
    println!(
        "train_sampling_normalized_rt_records\t{}",
        summary
            .train_sampling_preview
            .coverage
            .normalized_rt_records
    );
    println!(
        "train_sampling_observed_rt_records\t{}",
        summary.train_sampling_preview.coverage.observed_rt_records
    );
    println!(
        "train_sampling_ms2_records\t{}",
        summary.train_sampling_preview.coverage.ms2_records
    );
    println!(
        "validation_sampling_records\t{}",
        summary.validation_sampling.indices.len()
    );
    println!(
        "validation_sampling_unique_records\t{}",
        summary.validation_sampling.unique_records
    );
    for (source, count) in &summary.validation_sampling.source_records {
        println!("validation_sampling_source\t{source}\t{count}");
    }
    println!("epochs_this_invocation\t{}", summary.fit.epochs.len());
    println!(
        "completed_epochs\t{}",
        summary.fit.progress.completed_epochs
    );
    println!(
        "global_best_validation_loss\t{:?}",
        summary.fit.progress.best_validation_loss
    );
    println!("best_epoch\t{:?}", summary.fit.progress.best_epoch);
    println!("stopped_early\t{}", summary.fit.stopped_early);
    if let Some(last) = summary.fit.epochs.last() {
        println!("last_train_loss\t{}", last.train.mean_total_loss);
        println!("last_validation_loss\t{}", last.validation.mean_total_loss);
        println!("last_learning_rate\t{:?}", last.train.final_learning_rate);
        println!("last_gradient_norm\t{:?}", last.train.mean_gradient_norm);
    }
    Ok(())
}
