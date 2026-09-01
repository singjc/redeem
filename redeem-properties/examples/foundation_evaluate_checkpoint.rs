//! Evaluate a frozen foundation checkpoint on a held-out benchmark partition.
//!
//! This example performs no optimizer updates. The checkpoint provides the
//! model architecture, trained weights, trainer settings, and persisted target
//! normalization. The training YAML is used only to reconstruct the exact
//! corpus and benchmark identified by checkpoint provenance.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use redeem_properties::foundation::{
    evaluate_foundation_checkpoint, read_foundation_training_run_config, FoundationEpochMetrics,
    FoundationPartition,
};
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(4..=5).contains(&args.len()) {
        bail!(
            "usage: foundation_evaluate_checkpoint <training.yaml> <checkpoint_dir> <validation|test> <steps|all> [output.tsv]"
        );
    }
    let training_config = read_foundation_training_run_config(&args[0])?;
    let checkpoint_dir = PathBuf::from(&args[1]);
    let partition = match args[2].as_str() {
        "validation" => FoundationPartition::Validation,
        "test" => FoundationPartition::Test,
        other => bail!("unsupported evaluation partition '{other}'; use validation or test"),
    };
    let steps = if args[3] == "all" {
        None
    } else {
        Some(
            args[3]
                .parse::<usize>()
                .with_context(|| format!("invalid evaluation step count '{}'", args[3]))?,
        )
    };
    if steps == Some(0) {
        bail!("evaluation steps must be at least 1");
    }

    let summary = evaluate_foundation_checkpoint(
        &training_config,
        &checkpoint_dir,
        partition,
        steps,
        true,
        Device::Cpu,
    )?;

    let mut lines = Vec::new();
    push(
        &mut lines,
        "corpus_fingerprint",
        format!("fnv1a64:{:016x}", summary.corpus_fingerprint),
    );
    push(&mut lines, "corpus_records", summary.corpus_records);
    push(
        &mut lines,
        "partition",
        format!("{:?}", summary.partition).to_lowercase(),
    );
    push(&mut lines, "partition_records", summary.partition_records);
    push(
        &mut lines,
        "sampled_records",
        summary.sampling.indices.len(),
    );
    push(
        &mut lines,
        "sampled_unique_records",
        summary.sampling.unique_records,
    );
    push(
        &mut lines,
        "checkpoint_global_step",
        summary.checkpoint_metadata.global_step,
    );
    push(
        &mut lines,
        "checkpoint_optimizer_step",
        summary.checkpoint_metadata.optimizer_step,
    );
    push(
        &mut lines,
        "checkpoint_completed_epochs",
        summary.checkpoint_metadata.progress.completed_epochs,
    );
    if let Some(epoch) = summary.checkpoint_metadata.progress.best_epoch {
        push(&mut lines, "checkpoint_best_epoch", epoch);
    }
    if let Some(loss) = summary.checkpoint_metadata.progress.best_validation_loss {
        push(&mut lines, "checkpoint_best_validation_loss", loss);
    }
    for (source, count) in &summary.sampling.source_records {
        lines.push(format!("sampling_source\t{source}\t{count}"));
    }
    push_metrics(&mut lines, "property", &summary.property_metrics);
    for (source, metrics) in &summary.property_metrics_by_source {
        push_source_metrics(&mut lines, "property_source", source, metrics);
    }

    for line in &lines {
        println!("{line}");
    }
    if let Some(output) = args.get(4) {
        fs::write(output, format!("{}\n", lines.join("\n")))
            .with_context(|| format!("failed to write evaluation report {output}"))?;
        println!("report\t{output}");
    }
    Ok(())
}

fn push(lines: &mut Vec<String>, key: &str, value: impl std::fmt::Display) {
    lines.push(format!("{key}\t{value}"));
}

fn option<T: std::fmt::Display>(value: Option<T>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "NA".to_owned())
}

fn push_metrics(lines: &mut Vec<String>, prefix: &str, metrics: &FoundationEpochMetrics) {
    push(lines, &format!("{prefix}_steps"), metrics.steps);
    push(lines, &format!("{prefix}_loss"), metrics.mean_total_loss);
    push(
        lines,
        &format!("{prefix}_rt_loss"),
        option(metrics.mean_rt_loss),
    );
    push(
        lines,
        &format!("{prefix}_rt_mae_native"),
        option(metrics.mean_rt_mae_native),
    );
    push(
        lines,
        &format!("{prefix}_rt_rmse_native"),
        option(metrics.mean_rt_rmse_native),
    );
    push(
        lines,
        &format!("{prefix}_rt_native_labels"),
        metrics.rt_native_label_count,
    );
    push(
        lines,
        &format!("{prefix}_ccs_loss"),
        option(metrics.mean_ccs_loss),
    );
    push(
        lines,
        &format!("{prefix}_ccs_mae_native"),
        option(metrics.mean_ccs_mae_native),
    );
    push(
        lines,
        &format!("{prefix}_ccs_rmse_native"),
        option(metrics.mean_ccs_rmse_native),
    );
    push(
        lines,
        &format!("{prefix}_ccs_native_labels"),
        metrics.ccs_native_label_count,
    );
    push(
        lines,
        &format!("{prefix}_ms2_loss"),
        option(metrics.mean_ms2_loss),
    );
}

fn push_source_metrics(
    lines: &mut Vec<String>,
    prefix: &str,
    source: &str,
    metrics: &FoundationEpochMetrics,
) {
    lines.push(format!("{prefix}_steps\t{source}\t{}", metrics.steps));
    lines.push(format!(
        "{prefix}_loss\t{source}\t{}",
        metrics.mean_total_loss
    ));
    lines.push(format!(
        "{prefix}_rt_loss\t{source}\t{}",
        option(metrics.mean_rt_loss)
    ));
    lines.push(format!(
        "{prefix}_rt_mae_native\t{source}\t{}",
        option(metrics.mean_rt_mae_native)
    ));
    lines.push(format!(
        "{prefix}_rt_rmse_native\t{source}\t{}",
        option(metrics.mean_rt_rmse_native)
    ));
    lines.push(format!(
        "{prefix}_rt_native_labels\t{source}\t{}",
        metrics.rt_native_label_count
    ));
    lines.push(format!(
        "{prefix}_ms2_loss\t{source}\t{}",
        option(metrics.mean_ms2_loss)
    ));
}
