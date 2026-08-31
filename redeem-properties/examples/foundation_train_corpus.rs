//! Train/resume a multi-source foundation run from one YAML configuration.
//!
//! This example is the staging surface for future `redeem-cli` integration.

use anyhow::{bail, Result};
use candle_core::Device;
use redeem_properties::foundation::{
    read_foundation_training_run_config, run_foundation_pretraining, FoundationEpochMetrics,
    FoundationRegressionNormalization,
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
    print_normalization("rt", &summary.target_normalization.rt);
    print_normalization("ccs", &summary.target_normalization.ccs);
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
    println!(
        "validation_sampling_normalized_rt_records\t{}",
        summary.validation_sampling.coverage.normalized_rt_records
    );
    println!(
        "validation_sampling_observed_rt_records\t{}",
        summary.validation_sampling.coverage.observed_rt_records
    );
    println!(
        "validation_sampling_ms2_records\t{}",
        summary.validation_sampling.coverage.ms2_records
    );
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
        print_epoch_metrics("last_train", &last.train);
        print_epoch_metrics("last_validation", &last.validation);
    }
    for (source, metrics) in &summary.validation_by_source {
        println!(
            "validation_source_records\t{source}\t{}",
            summary
                .validation_sampling
                .source_records
                .get(source)
                .copied()
                .unwrap_or(0)
        );
        print_source_metrics(source, metrics);
    }
    Ok(())
}

fn print_epoch_metrics(prefix: &str, metrics: &FoundationEpochMetrics) {
    println!("{prefix}_loss\t{}", metrics.mean_total_loss);
    print_optional_f32(&format!("{prefix}_rt_loss"), metrics.mean_rt_loss);
    print_optional_f32(
        &format!("{prefix}_rt_mae_native"),
        metrics.mean_rt_mae_native,
    );
    print_optional_f32(
        &format!("{prefix}_rt_rmse_native"),
        metrics.mean_rt_rmse_native,
    );
    print_optional_f32(&format!("{prefix}_ccs_loss"), metrics.mean_ccs_loss);
    print_optional_f32(
        &format!("{prefix}_ccs_mae_native"),
        metrics.mean_ccs_mae_native,
    );
    print_optional_f32(
        &format!("{prefix}_ccs_rmse_native"),
        metrics.mean_ccs_rmse_native,
    );
    print_optional_f32(&format!("{prefix}_ms2_loss"), metrics.mean_ms2_loss);
    print_optional_f32(
        &format!("{prefix}_masked_residue_loss"),
        metrics.mean_masked_residue_loss,
    );
    print_optional_f32(
        &format!("{prefix}_chemistry_loss"),
        metrics.mean_chemistry_loss,
    );
    print_optional_f32(
        &format!("{prefix}_contrastive_loss"),
        metrics.mean_contrastive_loss,
    );
    if let Some(value) = metrics.final_learning_rate {
        println!("{prefix}_learning_rate\t{value}");
    }
    if let Some(value) = metrics.mean_gradient_norm {
        println!("{prefix}_gradient_norm\t{value}");
    }
    if let Some(value) = metrics.mean_gradient_scale {
        println!("{prefix}_gradient_scale\t{value}");
    }
    if let Some(value) = metrics.clipped_fraction {
        println!("{prefix}_clipped_steps\t{}", metrics.clipped_steps);
        println!("{prefix}_clipped_fraction\t{value}");
    }
}

fn print_source_metrics(source: &str, metrics: &FoundationEpochMetrics) {
    println!(
        "validation_source_total_loss\t{source}\t{}",
        metrics.mean_total_loss
    );
    print_source_optional(source, "rt_loss", metrics.mean_rt_loss);
    print_source_optional(source, "rt_mae_native", metrics.mean_rt_mae_native);
    print_source_optional(source, "rt_rmse_native", metrics.mean_rt_rmse_native);
    print_source_optional(source, "ccs_loss", metrics.mean_ccs_loss);
    print_source_optional(source, "ccs_mae_native", metrics.mean_ccs_mae_native);
    print_source_optional(source, "ccs_rmse_native", metrics.mean_ccs_rmse_native);
    print_source_optional(source, "ms2_loss", metrics.mean_ms2_loss);
    print_source_optional(
        source,
        "masked_residue_loss",
        metrics.mean_masked_residue_loss,
    );
    print_source_optional(source, "chemistry_loss", metrics.mean_chemistry_loss);
    print_source_optional(source, "contrastive_loss", metrics.mean_contrastive_loss);
}

fn print_normalization(label: &str, normalization: &FoundationRegressionNormalization) {
    println!(
        "{label}_normalization_strategy\t{:?}",
        normalization.strategy
    );
    println!(
        "{label}_normalization_labels\t{}",
        normalization.label_count
    );
    match normalization.mean {
        Some(value) => println!("{label}_normalization_mean\t{value}"),
        None => println!("{label}_normalization_mean\tNA"),
    }
    match normalization.standard_deviation {
        Some(value) => println!("{label}_normalization_std\t{value}"),
        None => println!("{label}_normalization_std\tNA"),
    }
    println!(
        "{label}_normalization_active\t{}",
        normalization.is_active()
    );
}

fn print_optional_f32(label: &str, value: Option<f32>) {
    match value {
        Some(value) => println!("{label}\t{value}"),
        None => println!("{label}\tNA"),
    }
}

fn print_source_optional(source: &str, label: &str, value: Option<f32>) {
    match value {
        Some(value) => println!("validation_source_{label}\t{source}\t{value}"),
        None => println!("validation_source_{label}\t{source}\tNA"),
    }
}
