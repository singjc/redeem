//! Fit the production CCS physical prior from the complete benchmark TRAIN partition.
//!
//! No neural-network inference is performed. Every finite CCS label in the
//! materialized training partition contributes to the ridge fit, while validation
//! is used only for reporting. The held-out test partition is never accessed.
//!
//! When the trainer uses source-weighted sampling, the production ridge uses all
//! train labels but assigns each source the same total objective weight used by
//! training. This prevents a naturally dominant corpus source from silently
//! defining a different physical-prior objective than the downstream model.

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    evaluate_foundation_ccs_physics_baseline, fit_foundation_ccs_physics_baseline,
    fit_foundation_ccs_physics_baseline_source_weighted, load_foundation_corpus,
    read_foundation_training_run_config, sample_foundation_validation_indices,
    FoundationBenchmarkManifest, FoundationCcsPhysicsBaselineConfig, FoundationCcsPhysicsFitConfig,
    FoundationCcsPhysicsMetrics, FoundationPartition, FoundationSamplingStrategy,
    FOUNDATION_CCS_PHYSICS_FEATURE_NAMES,
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

    // Always retain the natural-frequency full-train fit as a diagnostic. The
    // selected production fit follows the trainer's source objective when the
    // trainer itself uses source-weighted sampling.
    let uniform_fit = fit_foundation_ccs_physics_baseline(
        &corpus.records,
        &train_indices,
        FoundationCcsPhysicsFitConfig::default(),
    )?;
    let fit_source_weights = resolve_fit_source_weights(
        &corpus.records,
        &corpus.provenance,
        &train_indices,
        training_config.trainer.sampling.strategy,
        &training_config.trainer.sampling.source_weights,
    )?;
    let weighted_fit = fit_source_weights
        .as_ref()
        .map(|weights| {
            fit_foundation_ccs_physics_baseline_source_weighted(
                &corpus.records,
                &corpus.provenance,
                &train_indices,
                weights,
                FoundationCcsPhysicsFitConfig::default(),
            )
        })
        .transpose()?;
    let fit = weighted_fit.as_ref().unwrap_or(&uniform_fit);
    let fit_weighting = if weighted_fit.is_some() {
        "source-weighted"
    } else {
        "uniform-records"
    };

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
    let uniform_validation_metrics = weighted_fit
        .as_ref()
        .map(|_| {
            evaluate_foundation_ccs_physics_baseline(
                &corpus.records,
                &validation_indices,
                &uniform_fit.baseline,
            )
        })
        .transpose()?;

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
    push(&mut lines, "fit_weighting", fit_weighting);
    push(&mut lines, "ridge_lambda", fit.ridge_lambda);
    push(
        &mut lines,
        "effective_fit_weight_sum",
        fit.effective_weight_sum,
    );
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

    for source in &fit.source_weight_summaries {
        lines.push(format!(
            "fit_source\t{}\tlabels={}\trequested_weight={}\tnormalized_weight={}\tper_record_weight={}",
            source.source_id,
            source.label_count,
            source.requested_weight,
            source.normalized_weight,
            source.per_record_weight
        ));
    }

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
    if let Some(metrics) = uniform_validation_metrics {
        push_metrics(
            &mut lines,
            "uniform_full_train_physics",
            "validation",
            metrics,
        );
    }
    if let Some(metrics) = configured_baseline_metrics {
        push_metrics(&mut lines, "configured_physics", "validation", metrics);
    }

    for (source, indices) in &by_source {
        let metrics =
            evaluate_foundation_ccs_physics_baseline(&corpus.records, indices, &fit.baseline)?;
        push_metrics(&mut lines, "full_train_physics", source, metrics);
        if weighted_fit.is_some() {
            let uniform_metrics = evaluate_foundation_ccs_physics_baseline(
                &corpus.records,
                indices,
                &uniform_fit.baseline,
            )?;
            push_metrics(
                &mut lines,
                "uniform_full_train_physics",
                source,
                uniform_metrics,
            );
        }
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

fn resolve_fit_source_weights(
    records: &[redeem_properties::foundation::FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    train_indices: &[usize],
    strategy: FoundationSamplingStrategy,
    configured: &BTreeMap<String, f64>,
) -> Result<Option<BTreeMap<String, f64>>> {
    if strategy != FoundationSamplingStrategy::SourceWeighted {
        return Ok(None);
    }

    let mut ccs_sources = BTreeMap::<String, ()>::new();
    for &index in train_indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation train record index {index} is out of bounds")
        })?;
        if !record.ccs.is_some_and(|value| value.is_finite()) {
            continue;
        }
        let source = provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation train provenance index {index} is out of bounds")
        })?;
        ccs_sources.insert(source.source_id.clone(), ());
    }
    if ccs_sources.is_empty() {
        bail!("foundation CCS physics fit found no CCS-bearing training sources");
    }

    let weights = if configured.is_empty() {
        ccs_sources
            .keys()
            .map(|source| (source.clone(), 1.0))
            .collect::<BTreeMap<_, _>>()
    } else {
        let mut weights = BTreeMap::new();
        for source in ccs_sources.keys() {
            let weight = configured.get(source).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "foundation source-weighted training config has no weight for CCS source '{source}'"
                )
            })?;
            weights.insert(source.clone(), weight);
        }
        weights
    };
    Ok(Some(weights))
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
