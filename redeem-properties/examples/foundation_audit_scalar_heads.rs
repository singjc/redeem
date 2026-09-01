//! Diagnose scalar-property calibration and simple physics baselines without touching test data.
//!
//! The utility fits calibration/baseline models only on a deterministic sample from the benchmark
//! training partition and evaluates them on a disjoint deterministic validation sample.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config,
    sample_foundation_validation_indices, FoundationBenchmarkManifest, FoundationPartition,
    FoundationTrainer,
};
use std::{collections::BTreeMap, env, fs, path::PathBuf};

#[derive(Debug, Clone)]
struct CcsObservation {
    target: f64,
    model_prediction: f64,
    charge: Option<i32>,
    precursor_mz: Option<f32>,
    sequence_len: usize,
    source: String,
}

#[derive(Debug, Clone, Copy)]
struct RegressionMetrics {
    n: usize,
    mae: f64,
    rmse: f64,
    r_squared: f64,
    pearson_r: f64,
    target_std: f64,
    prediction_std: f64,
    prediction_mean: f64,
    target_mean: f64,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(4..=5).contains(&args.len()) {
        bail!(
            "usage: foundation_audit_scalar_heads <training.yaml> <checkpoint_dir> <calibration_steps> <validation_steps> [output.tsv]"
        );
    }
    let training_config = read_foundation_training_run_config(&args[0])?;
    let checkpoint_dir = PathBuf::from(&args[1]);
    let calibration_steps = parse_steps(&args[2], "calibration")?;
    let validation_steps = parse_steps(&args[3], "validation")?;

    let corpus = load_foundation_corpus(&training_config.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&training_config.benchmark_manifest)?;
    benchmark
        .validate_against_records(&corpus.records)
        .context("foundation benchmark does not match the assembled corpus")?;

    let (trainer, metadata) = FoundationTrainer::from_checkpoint(&checkpoint_dir, Device::Cpu)?;
    if metadata.provenance.corpus_fingerprint != Some(corpus.corpus_fingerprint) {
        bail!("checkpoint corpus fingerprint does not match assembled corpus");
    }
    if metadata.provenance.benchmark_dataset_fingerprint != Some(benchmark.dataset_fingerprint)
        || metadata.provenance.benchmark_manifest_fingerprint
            != Some(benchmark.manifest_fingerprint())
    {
        bail!("checkpoint benchmark provenance does not match supplied benchmark");
    }

    let train_indices = benchmark.partition_indices(FoundationPartition::Train);
    let validation_indices = benchmark.partition_indices(FoundationPartition::Validation);
    if train_indices.is_empty() || validation_indices.is_empty() {
        bail!("benchmark requires non-empty train and validation partitions");
    }

    let mut calibration_sampling = metadata.trainer_config.sampling.clone();
    calibration_sampling.validation_steps = Some(calibration_steps);
    calibration_sampling.report_validation_by_source = true;
    let calibration_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &train_indices,
        metadata.trainer_config.batch_size,
        metadata.trainer_config.seed ^ 0x4341_4c49_4252_4154,
        &calibration_sampling,
    )?;

    let mut validation_sampling = metadata.trainer_config.sampling.clone();
    validation_sampling.validation_steps = Some(validation_steps);
    validation_sampling.report_validation_by_source = true;
    let validation_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &validation_indices,
        metadata.trainer_config.batch_size,
        metadata.trainer_config.seed,
        &validation_sampling,
    )?;

    let calibration = collect_ccs_observations(
        &trainer,
        &corpus.records,
        &corpus.provenance,
        &calibration_plan.indices,
        metadata.trainer_config.batch_size,
    )?;
    let validation = collect_ccs_observations(
        &trainer,
        &corpus.records,
        &corpus.provenance,
        &validation_plan.indices,
        metadata.trainer_config.batch_size,
    )?;
    if calibration.len() < 16 || validation.len() < 16 {
        bail!("not enough CCS-labelled observations for calibration audit");
    }

    let (affine_intercept, affine_slope) = fit_affine(&calibration);
    let physics_coefficients = fit_ridge(&calibration, false, 1e-4)?;
    let combined_coefficients = fit_ridge(&calibration, true, 1e-4)?;

    let mut lines = Vec::new();
    push(
        &mut lines,
        "corpus_fingerprint",
        format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
    );
    push(&mut lines, "checkpoint_global_step", metadata.global_step);
    push(&mut lines, "calibration_partition", "train");
    push(
        &mut lines,
        "calibration_sampled_records",
        calibration_plan.indices.len(),
    );
    push(&mut lines, "calibration_ccs_labels", calibration.len());
    push(&mut lines, "validation_partition", "validation");
    push(
        &mut lines,
        "validation_sampled_records",
        validation_plan.indices.len(),
    );
    push(&mut lines, "validation_ccs_labels", validation.len());
    for (source, count) in &calibration_plan.source_records {
        lines.push(format!("calibration_source\t{source}\t{count}"));
    }
    for (source, count) in &validation_plan.source_records {
        lines.push(format!("validation_source\t{source}\t{count}"));
    }

    push(&mut lines, "affine_intercept", affine_intercept);
    push(&mut lines, "affine_slope", affine_slope);
    for (index, name) in [
        "intercept",
        "charge_over_4",
        "charge_squared_over_16",
        "precursor_mz_over_1000",
        "neutral_mass_proxy_over_3000",
        "sequence_len_over_30",
        "charge_present",
        "precursor_mz_present",
    ]
    .iter()
    .enumerate()
    {
        lines.push(format!("physics_feature\t{index}\t{name}"));
    }
    lines.push("combined_feature\t8\tmodel_prediction_over_500".to_string());
    for (index, coefficient) in physics_coefficients.iter().enumerate() {
        lines.push(format!("physics_coefficient\t{index}\t{coefficient}"));
    }
    for (index, coefficient) in combined_coefficients.iter().enumerate() {
        lines.push(format!("combined_coefficient\t{index}\t{coefficient}"));
    }

    report_method(&mut lines, "raw_model", &validation, |obs| {
        obs.model_prediction
    });
    report_method(&mut lines, "affine_model", &validation, |obs| {
        affine_intercept + affine_slope * obs.model_prediction
    });
    report_method(&mut lines, "physics_ridge", &validation, |obs| {
        dot(&physics_coefficients, &physics_features(obs, false))
    });
    report_method(&mut lines, "combined_ridge", &validation, |obs| {
        dot(&combined_coefficients, &physics_features(obs, true))
    });

    let mut by_source = BTreeMap::<String, Vec<CcsObservation>>::new();
    for observation in &validation {
        by_source
            .entry(observation.source.clone())
            .or_default()
            .push(observation.clone());
    }
    for (source, observations) in by_source {
        report_source_method(&mut lines, "raw_model", &source, &observations, |obs| {
            obs.model_prediction
        });
        report_source_method(&mut lines, "affine_model", &source, &observations, |obs| {
            affine_intercept + affine_slope * obs.model_prediction
        });
        report_source_method(&mut lines, "physics_ridge", &source, &observations, |obs| {
            dot(&physics_coefficients, &physics_features(obs, false))
        });
        report_source_method(
            &mut lines,
            "combined_ridge",
            &source,
            &observations,
            |obs| dot(&combined_coefficients, &physics_features(obs, true)),
        );
    }

    for line in &lines {
        println!("{line}");
    }
    if let Some(output) = args.get(4) {
        fs::write(output, format!("{}\n", lines.join("\n")))
            .with_context(|| format!("failed to write scalar-head audit report {output}"))?;
        println!("report\t{output}");
    }
    Ok(())
}

fn parse_steps(value: &str, label: &str) -> Result<usize> {
    let steps = value
        .parse::<usize>()
        .with_context(|| format!("invalid {label} step count '{value}'"))?;
    if steps == 0 {
        bail!("{label} steps must be at least 1");
    }
    Ok(steps)
}

fn collect_ccs_observations(
    trainer: &FoundationTrainer,
    records: &[redeem_properties::foundation::FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    indices: &[usize],
    batch_size: usize,
) -> Result<Vec<CcsObservation>> {
    let mut observations = Vec::new();
    for chunk in indices.chunks(batch_size.max(1)) {
        let peptides = chunk
            .iter()
            .map(|&index| records[index].peptidoform.clone())
            .collect::<Vec<_>>();
        let contexts = chunk
            .iter()
            .map(|&index| records[index].context.clone())
            .collect::<Vec<_>>();
        let prediction = trainer.predict_native(&peptides, &contexts)?;
        let predicted_ccs = prediction.ccs.squeeze(1)?.to_vec1::<f32>()?;
        for (&index, predicted) in chunk.iter().zip(predicted_ccs) {
            let record = &records[index];
            let Some(target) = record.ccs else {
                continue;
            };
            observations.push(CcsObservation {
                target: f64::from(target),
                model_prediction: f64::from(predicted),
                charge: record.context.charge,
                precursor_mz: record.context.precursor_mz,
                sequence_len: record.peptidoform.sequence.len(),
                source: provenance[index].source_id.clone(),
            });
        }
    }
    Ok(observations)
}

fn fit_affine(observations: &[CcsObservation]) -> (f64, f64) {
    let n = observations.len() as f64;
    let mean_x = observations
        .iter()
        .map(|obs| obs.model_prediction)
        .sum::<f64>()
        / n;
    let mean_y = observations.iter().map(|obs| obs.target).sum::<f64>() / n;
    let covariance = observations
        .iter()
        .map(|obs| (obs.model_prediction - mean_x) * (obs.target - mean_y))
        .sum::<f64>();
    let variance = observations
        .iter()
        .map(|obs| (obs.model_prediction - mean_x).powi(2))
        .sum::<f64>();
    let slope = if variance > 1e-12 {
        covariance / variance
    } else {
        0.0
    };
    (mean_y - slope * mean_x, slope)
}

fn physics_features(observation: &CcsObservation, include_model_prediction: bool) -> Vec<f64> {
    let charge = observation.charge.unwrap_or(0) as f64;
    let mz = f64::from(observation.precursor_mz.unwrap_or(0.0));
    let charge_present = if observation.charge.is_some() {
        1.0
    } else {
        0.0
    };
    let mz_present = if observation.precursor_mz.is_some() {
        1.0
    } else {
        0.0
    };
    let neutral_mass_proxy = charge * mz;
    let mut features = vec![
        1.0,
        charge / 4.0,
        charge * charge / 16.0,
        mz / 1000.0,
        neutral_mass_proxy / 3000.0,
        observation.sequence_len as f64 / 30.0,
        charge_present,
        mz_present,
    ];
    if include_model_prediction {
        features.push(observation.model_prediction / 500.0);
    }
    features
}

fn fit_ridge(
    observations: &[CcsObservation],
    include_model_prediction: bool,
    lambda: f64,
) -> Result<Vec<f64>> {
    let feature_count = physics_features(&observations[0], include_model_prediction).len();
    let mut normal = vec![vec![0.0; feature_count]; feature_count];
    let mut rhs = vec![0.0; feature_count];
    for observation in observations {
        let features = physics_features(observation, include_model_prediction);
        for row in 0..feature_count {
            rhs[row] += features[row] * observation.target;
            for col in 0..feature_count {
                normal[row][col] += features[row] * features[col];
            }
        }
    }
    for index in 1..feature_count {
        normal[index][index] += lambda;
    }
    solve_linear_system(normal, rhs)
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .total_cmp(&matrix[right][pivot].abs())
            })
            .unwrap_or(pivot);
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let diagonal = matrix[pivot][pivot];
        if diagonal.abs() < 1e-12 {
            bail!("scalar-head ridge system is numerically singular at column {pivot}");
        }
        for col in pivot..n {
            matrix[pivot][col] /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..n {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}

fn dot(coefficients: &[f64], features: &[f64]) -> f64 {
    coefficients
        .iter()
        .zip(features)
        .map(|(coefficient, feature)| coefficient * feature)
        .sum()
}

fn report_method<F>(
    lines: &mut Vec<String>,
    method: &str,
    observations: &[CcsObservation],
    predictor: F,
) where
    F: Fn(&CcsObservation) -> f64,
{
    let metrics = regression_metrics(observations, predictor);
    push_metrics(lines, method, None, metrics);
}

fn report_source_method<F>(
    lines: &mut Vec<String>,
    method: &str,
    source: &str,
    observations: &[CcsObservation],
    predictor: F,
) where
    F: Fn(&CcsObservation) -> f64,
{
    let metrics = regression_metrics(observations, predictor);
    push_metrics(lines, method, Some(source), metrics);
}

fn regression_metrics<F>(observations: &[CcsObservation], predictor: F) -> RegressionMetrics
where
    F: Fn(&CcsObservation) -> f64,
{
    let n = observations.len();
    let n_f = n as f64;
    let targets = observations
        .iter()
        .map(|obs| obs.target)
        .collect::<Vec<_>>();
    let predictions = observations.iter().map(predictor).collect::<Vec<_>>();
    let target_mean = targets.iter().sum::<f64>() / n_f;
    let prediction_mean = predictions.iter().sum::<f64>() / n_f;
    let target_ss = targets
        .iter()
        .map(|value| (value - target_mean).powi(2))
        .sum::<f64>();
    let prediction_ss = predictions
        .iter()
        .map(|value| (value - prediction_mean).powi(2))
        .sum::<f64>();
    let covariance = targets
        .iter()
        .zip(&predictions)
        .map(|(target, prediction)| (target - target_mean) * (prediction - prediction_mean))
        .sum::<f64>();
    let squared_error = targets
        .iter()
        .zip(&predictions)
        .map(|(target, prediction)| (prediction - target).powi(2))
        .sum::<f64>();
    let absolute_error = targets
        .iter()
        .zip(&predictions)
        .map(|(target, prediction)| (prediction - target).abs())
        .sum::<f64>();
    RegressionMetrics {
        n,
        mae: absolute_error / n_f,
        rmse: (squared_error / n_f).sqrt(),
        r_squared: if target_ss > 1e-12 {
            1.0 - squared_error / target_ss
        } else {
            f64::NAN
        },
        pearson_r: if target_ss > 1e-12 && prediction_ss > 1e-12 {
            covariance / (target_ss * prediction_ss).sqrt()
        } else {
            f64::NAN
        },
        target_std: (target_ss / n_f).sqrt(),
        prediction_std: (prediction_ss / n_f).sqrt(),
        prediction_mean,
        target_mean,
    }
}

fn push_metrics(
    lines: &mut Vec<String>,
    method: &str,
    source: Option<&str>,
    metrics: RegressionMetrics,
) {
    let prefix = source
        .map(|source| format!("metric\t{method}\t{source}"))
        .unwrap_or_else(|| format!("metric\t{method}\toverall"));
    lines.push(format!("{prefix}\tn\t{}", metrics.n));
    lines.push(format!("{prefix}\tmae\t{}", metrics.mae));
    lines.push(format!("{prefix}\trmse\t{}", metrics.rmse));
    lines.push(format!("{prefix}\tr_squared\t{}", metrics.r_squared));
    lines.push(format!("{prefix}\tpearson_r\t{}", metrics.pearson_r));
    lines.push(format!("{prefix}\ttarget_std\t{}", metrics.target_std));
    lines.push(format!(
        "{prefix}\tprediction_std\t{}",
        metrics.prediction_std
    ));
    lines.push(format!("{prefix}\ttarget_mean\t{}", metrics.target_mean));
    lines.push(format!(
        "{prefix}\tprediction_mean\t{}",
        metrics.prediction_mean
    ));
}

fn push(lines: &mut Vec<String>, key: &str, value: impl std::fmt::Display) {
    lines.push(format!("{key}\t{value}"));
}
