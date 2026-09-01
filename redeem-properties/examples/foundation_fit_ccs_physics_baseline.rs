//! Fit the production CCS physical prior from the complete benchmark TRAIN partition.
//!
//! No neural-network inference is performed. Every finite CCS label in the
//! materialized training partition contributes to the ridge fit, while validation
//! is used only for reporting. The held-out test partition is never accessed.

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    evaluate_foundation_ccs_physics_baseline, fit_foundation_ccs_physics_baseline,
    load_foundation_corpus, read_foundation_training_run_config,
    sample_foundation_validation_indices, FoundationBenchmarkManifest,
    FoundationCcsPhysicsBaselineConfig, FoundationCcsPhysicsFitConfig, FoundationCcsPhysicsMetrics,
    FoundationPartition, FOUNDATION_CCS_PHYSICS_FEATURE_NAMES,
};
use std::{collections::BTreeMap, env, fs, path::Path};

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(2..=4).contains(&args.len()) {
        bail!(
            "usage: foundation_fit_ccs_physics_baseline <training.yaml> <validation_steps|all> [baseline.yaml] [report.tsv]"
        );
    }

    let training_config = read_foundation_training_run_config(&args[0])?;
    let validation_steps = parse_steps_or_all(&args[1])?;
    let baseline_output = args.get(2).map(String::as_str);
    let report_output = args.get(3).map(String::as_str);

    let corpus = load_foundation_corpus(&training_config.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&training_config.benchmark_manifest)?;
    benchmark
        .validate_against_records(&corpus.records)
        .context("foundation benchmark does not match the assembled corpus")?;

    let train_indices = benchmark.partition_indices(FoundationPartition::Train);
    let validation_partition = benchmark.partition_indices(FoundationPartition::Validation);
    if train_indices.is_empty() || validation_partition.is_empty() {
        bail!("foundation CCS physics fit requires non-empty train and validation partitions");
    }

    let fit = fit_foundation_ccs_physics_baseline(
        &corpus.records,
        &train_indices,
        FoundationCcsPhysicsFitConfig::default(),
    )?;

    let validation_indices = match validation_steps {
        None => validation_partition.clone(),
        Some(steps) => {
            let mut sampling = training_config.trainer.sampling.clone();
            sampling.validation_steps = Some(steps);
            sampling.report_validation_by_source = true;
            sample_foundation_validation_indices(
                &corpus.records,
                &corpus.provenance,
                &validation_partition,
                training_config.trainer.batch_size,
                training_config.trainer.seed,
                &sampling,
            )?
            .indices
        }
    };
    let validation_metrics = evaluate_foundation_ccs_physics_baseline(
        &corpus.records,
        &validation_indices,
        &fit.baseline,
    )?;

    let configured_baseline_metrics = training_config
        .model
        .ccs_physics_baseline
        .as_ref()
        .map(|baseline| {
            evaluate_foundation_ccs_physics_baseline(&corpus.records, &validation_indices, baseline)
        })
        .transpose()?;

    let mut by_source = BTreeMap::<String, Vec<usize>>::new();
    for &index in &validation_indices {
        let provenance = corpus.provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation validation provenance index {index} is out of bounds")
        })?;
        by_source
            .entry(provenance.source_id.clone())
            .or_default()
            .push(index);
    }

    let mut lines = Vec::new();
    push(
        &mut lines,
        "corpus_fingerprint",
        format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
    );
    push(&mut lines, "corpus_records", corpus.records.len());
    push(&mut lines, "train_partition_records", train_indices.len());
    push(&mut lines, "train_ccs_labels", fit.train_label_count);
    push(&mut lines, "ridge_lambda", fit.ridge_lambda);
    push(
        &mut lines,
        "target_mean_native",
        fit.baseline.target_mean_native,
    );
    push(
        &mut lines,
        "target_std_native",
        fit.baseline.target_std_native,
    );
    push(
        &mut lines,
        "validation_partition_records",
        validation_partition.len(),
    );
    push(
        &mut lines,
        "validation_sampled_records",
        validation_indices.len(),
    );

    for (index, name) in FOUNDATION_CCS_PHYSICS_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("physics_feature\t{index}\t{name}"));
        lines.push(format!(
            "physics_coefficient\t{index}\t{}",
            fit.baseline.coefficients_native[index]
        ));
    }
    for (index, feature) in fit.feature_summaries.iter().enumerate() {
        lines.push(format!("train_feature_min\t{index}\t{}", feature.min));
        lines.push(format!("train_feature_mean\t{index}\t{}", feature.mean));
        lines.push(format!(
            "train_feature_std\t{index}\t{}",
            feature.standard_deviation
        ));
        lines.push(format!("train_feature_max\t{index}\t{}", feature.max));
    }

    push_metrics(&mut lines, "full_train_physics", "train", fit.train_metrics);
    push_metrics(
        &mut lines,
        "full_train_physics",
        "validation",
        validation_metrics,
    );
    if let Some(metrics) = configured_baseline_metrics {
        push_metrics(&mut lines, "configured_physics", "validation", metrics);
    }

    for (source, indices) in &by_source {
        let metrics =
            evaluate_foundation_ccs_physics_baseline(&corpus.records, indices, &fit.baseline)?;
        push_metrics(&mut lines, "full_train_physics", source, metrics);
        if let Some(configured) = &training_config.model.ccs_physics_baseline {
            let configured_metrics =
                evaluate_foundation_ccs_physics_baseline(&corpus.records, indices, configured)?;
            push_metrics(&mut lines, "configured_physics", source, configured_metrics);
        }
    }

    for line in &lines {
        println!("{line}");
    }

    println!("baseline_yaml_begin");
    for line in baseline_yaml(&fit.baseline)?.lines() {
        println!("{line}");
    }
    println!("baseline_yaml_end");

    if let Some(path) = baseline_output {
        write_baseline_yaml(path, &fit.baseline)?;
        println!("baseline_yaml\t{path}");
    }
    if let Some(path) = report_output {
        fs::write(path, format!("{}\n", lines.join("\n")))
            .with_context(|| format!("failed to write CCS physics fit report {path}"))?;
        println!("report\t{path}");
    }
    Ok(())
}

fn parse_steps_or_all(value: &str) -> Result<Option<usize>> {
    if value.eq_ignore_ascii_case("all") {
        return Ok(None);
    }
    let steps = value
        .parse::<usize>()
        .with_context(|| format!("invalid validation step count '{value}'"))?;
    if steps == 0 {
        bail!("validation steps must be at least 1");
    }
    Ok(Some(steps))
}

fn baseline_yaml(baseline: &FoundationCcsPhysicsBaselineConfig) -> Result<String> {
    serde_yaml::to_string(baseline).context("failed to serialize CCS physics baseline")
}

fn write_baseline_yaml(path: &str, baseline: &FoundationCcsPhysicsBaselineConfig) -> Result<()> {
    if Path::new(path).as_os_str().is_empty() {
        bail!("CCS physics baseline output path cannot be empty");
    }
    fs::write(path, baseline_yaml(baseline)?)
        .with_context(|| format!("failed to write CCS physics baseline {path}"))
}

fn push_metrics(
    lines: &mut Vec<String>,
    method: &str,
    scope: &str,
    metrics: FoundationCcsPhysicsMetrics,
) {
    let prefix = format!("metric\t{method}\t{scope}");
    lines.push(format!("{prefix}\tn\t{}", metrics.label_count));
    lines.push(format!("{prefix}\tmae\t{}", option(metrics.mae_native)));
    lines.push(format!("{prefix}\trmse\t{}", option(metrics.rmse_native)));
    lines.push(format!(
        "{prefix}\tr_squared\t{}",
        option(metrics.r_squared)
    ));
    lines.push(format!(
        "{prefix}\tpearson_r\t{}",
        option(metrics.pearson_r)
    ));
    lines.push(format!(
        "{prefix}\ttarget_mean\t{}",
        option(metrics.target_mean_native)
    ));
    lines.push(format!(
        "{prefix}\tprediction_mean\t{}",
        option(metrics.prediction_mean_native)
    ));
    lines.push(format!(
        "{prefix}\ttarget_std\t{}",
        option(metrics.target_std_native)
    ));
    lines.push(format!(
        "{prefix}\tprediction_std\t{}",
        option(metrics.prediction_std_native)
    ));
}

fn option(value: Option<f64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "NA".to_owned())
}

fn push(lines: &mut Vec<String>, key: &str, value: impl std::fmt::Display) {
    lines.push(format!("{key}\t{value}"));
}
