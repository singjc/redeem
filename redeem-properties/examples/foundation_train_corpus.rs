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
    println!(
        "rt_encoder_gradient_scale\t{}",
        config.trainer.shared_gradient_scales.rt_encoder
    );
    println!(
        "ccs_encoder_gradient_scale\t{}",
        config.trainer.shared_gradient_scales.ccs_encoder
    );
    println!(
        "clean_property_validation\t{}",
        config.trainer.evaluation.clean_property_validation
    );
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
        "train_sampling_ccs_records\t{}",
        summary.train_sampling_preview.coverage.ccs_records
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
        "validation_sampling_ccs_records\t{}",
        summary.validation_sampling.coverage.ccs_records
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
    for epoch in &summary.fit.epochs {
        println!("epoch\t{}\timproved={}", epoch.epoch, epoch.improved);
        print_epoch_metrics(&format!("epoch{}_train", epoch.epoch), &epoch.train);
        print_epoch_metrics(
            &format!("epoch{}_validation", epoch.epoch),
            &epoch.validation,
        );
        if let Some(property_validation) = &epoch.property_validation {
            print_epoch_metrics(
                &format!("epoch{}_property_validation", epoch.epoch),
                property_validation,
            );
        }
    }
    if let Some(last) = summary.fit.epochs.last() {
        print_epoch_metrics("last_train", &last.train);
        print_epoch_metrics("last_validation", &last.validation);
        if let Some(property_validation) = &last.property_validation {
            print_epoch_metrics("last_property_validation", property_validation);
        }
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
        print_source_metrics("validation_source", source, metrics);
    }
    for (source, metrics) in &summary.property_validation_by_source {
        println!(
            "property_validation_source_records\t{source}\t{}",
            summary
                .validation_sampling
                .source_records
                .get(source)
                .copied()
                .unwrap_or(0)
        );
        print_source_metrics("property_validation_source", source, metrics);
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
    println!(
        "{prefix}_rt_native_labels\t{}",
        metrics.rt_native_label_count
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
    println!(
        "{prefix}_ccs_native_labels\t{}",
        metrics.ccs_native_label_count
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
    println!(
        "{prefix}_gradient_diagnostic_steps\t{}",
        metrics.gradient_diagnostic_steps
    );
    print_optional_f64(
        &format!("{prefix}_rt_gradient_norm"),
        metrics.mean_rt_gradient_norm,
    );
    print_optional_f64(
        &format!("{prefix}_ccs_gradient_norm"),
        metrics.mean_ccs_gradient_norm,
    );
    print_optional_f64(
        &format!("{prefix}_ms2_gradient_norm"),
        metrics.mean_ms2_gradient_norm,
    );
    print_optional_f64(
        &format!("{prefix}_masked_residue_gradient_norm"),
        metrics.mean_masked_residue_gradient_norm,
    );
    print_optional_f64(
        &format!("{prefix}_chemistry_gradient_norm"),
        metrics.mean_chemistry_gradient_norm,
    );
    print_optional_f64(
        &format!("{prefix}_contrastive_gradient_norm"),
        metrics.mean_contrastive_gradient_norm,
    );
    print_optional_f64(
        &format!("{prefix}_rt_gradient_cosine_to_total"),
        metrics.mean_rt_gradient_cosine_to_total,
    );
    print_optional_f64(
        &format!("{prefix}_ccs_gradient_cosine_to_total"),
        metrics.mean_ccs_gradient_cosine_to_total,
    );
    print_optional_f64(
        &format!("{prefix}_ms2_gradient_cosine_to_total"),
        metrics.mean_ms2_gradient_cosine_to_total,
    );
    print_optional_f64(
        &format!("{prefix}_masked_residue_gradient_cosine_to_total"),
        metrics.mean_masked_residue_gradient_cosine_to_total,
    );
    print_optional_f64(
        &format!("{prefix}_chemistry_gradient_cosine_to_total"),
        metrics.mean_chemistry_gradient_cosine_to_total,
    );
    print_optional_f64(
        &format!("{prefix}_contrastive_gradient_cosine_to_total"),
        metrics.mean_contrastive_gradient_cosine_to_total,
    );
}

fn print_source_metrics(prefix: &str, source: &str, metrics: &FoundationEpochMetrics) {
    println!("{prefix}_total_loss\t{source}\t{}", metrics.mean_total_loss);
    print_source_optional(prefix, source, "rt_loss", metrics.mean_rt_loss);
    print_source_optional(prefix, source, "rt_mae_native", metrics.mean_rt_mae_native);
    print_source_optional(
        prefix,
        source,
        "rt_rmse_native",
        metrics.mean_rt_rmse_native,
    );
    println!(
        "{prefix}_rt_native_labels\t{source}\t{}",
        metrics.rt_native_label_count
    );
    print_source_optional(prefix, source, "ccs_loss", metrics.mean_ccs_loss);
    print_source_optional(
        prefix,
        source,
        "ccs_mae_native",
        metrics.mean_ccs_mae_native,
    );
    print_source_optional(
        prefix,
        source,
        "ccs_rmse_native",
        metrics.mean_ccs_rmse_native,
    );
    println!(
        "{prefix}_ccs_native_labels\t{source}\t{}",
        metrics.ccs_native_label_count
    );
    print_source_optional(prefix, source, "ms2_loss", metrics.mean_ms2_loss);
    print_source_optional(
        prefix,
        source,
        "masked_residue_loss",
        metrics.mean_masked_residue_loss,
    );
    print_source_optional(
        prefix,
        source,
        "chemistry_loss",
        metrics.mean_chemistry_loss,
    );
    print_source_optional(
        prefix,
        source,
        "contrastive_loss",
        metrics.mean_contrastive_loss,
    );
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

fn print_optional_f64(label: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("{label}\t{value}"),
        None => println!("{label}\tNA"),
    }
}

fn print_source_optional(prefix: &str, source: &str, label: &str, value: Option<f32>) {
    match value {
        Some(value) => println!("{prefix}_{label}\t{source}\t{value}"),
        None => println!("{prefix}_{label}\t{source}\tNA"),
    }
}
