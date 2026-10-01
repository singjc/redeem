//! One-time protected TRAIN-HOLDOUT evaluation for the frozen ReDeeM v0.52 checkpoint.
//!
//! This evaluator is intentionally read-only. It does not train, select checkpoints, access
//! historical VALIDATION/APD, or access historical TEST. The prepared v0.26 benchmark's `Test`
//! partition is the reserved TRAIN-HOLDOUT and is consumed exactly once by this program.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_ms2_loss, load_foundation_corpus, read_foundation_training_run_config,
    FoundationBenchmarkManifest, FoundationCollator, FoundationCollatorConfig,
    FoundationCorruptionConfig, FoundationFragmentContextBatchV0350, FoundationMs2LossConfig,
    FoundationPartition, FoundationRegressionNormalization, FoundationScalarPhysicsBatchV0360,
    FoundationTargetNormalizationConfig, FoundationTrainingRecord, PeptideFoundationV0520Config,
    PeptideFoundationV0520Model, RetentionTimeObjective, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520,
};
use serde::Deserialize;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = "foundation_v0520_train_holdout_once_v1";
const V052_VERSION: u32 = 520;
const V052_OBJECTIVE: &str = "v0520_mobility_aware_pair_representation";
const DEFAULT_BATCH_SIZE: usize = 32;

#[derive(Debug, Clone, Deserialize)]
struct V052Metadata {
    version: u32,
    objective: String,
    architecture: String,
    v0520_config: PeptideFoundationV0520Config,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    completed_epochs: usize,
    completed_updates: usize,
    smoke_mode: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct PropertyMetrics {
    rt_mae_native: Option<f64>,
    rt_rmse_native: Option<f64>,
    ms2_loss: Option<f64>,
    ms2_pointwise_mse: Option<f64>,
    ms2_pointwise_mae: Option<f64>,
    ms2_mean_cosine: Option<f64>,
    ms2_mean_spectral_angle: Option<f64>,
    ms2_mean_pearson: Option<f64>,
    rt_records: usize,
    ms2_spectra: usize,
    ms2_fragments: usize,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(3..=4).contains(&args.len()) {
        anyhow::bail!(
            "usage: foundation_evaluate_v0520_holdout RUN_V0260.yaml V0520_BEST OUTPUT_DIR [batch_size=32]"
        );
    }

    let training_yaml = PathBuf::from(&args[0]);
    let checkpoint = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    let batch_size = parse_or(&args, 3, DEFAULT_BATCH_SIZE)?;
    if batch_size == 0 {
        anyhow::bail!("holdout batch_size must be positive");
    }
    if output_dir.exists() {
        anyhow::bail!("holdout output directory must be fresh: {output_dir:?}");
    }

    let metadata = read_metadata(&checkpoint)?;
    validate_metadata(&metadata)?;

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.52 HOLDOUT evaluation requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let train_records = benchmark
        .partition_indices(FoundationPartition::Train)
        .len();
    let dev_records = benchmark
        .partition_indices(FoundationPartition::Validation)
        .len();
    let holdout_indices = benchmark.partition_indices(FoundationPartition::Test);
    if holdout_indices.is_empty() {
        anyhow::bail!("prepared benchmark has zero TRAIN-HOLDOUT records");
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = PeptideFoundationV0520Model::new(metadata.v0520_config.clone(), vb)?;
    let model_path = checkpoint.join("model.safetensors");
    varmap
        .load(&model_path)
        .with_context(|| format!("failed to load frozen v0.52 checkpoint {model_path:?}"))?;

    let collator = FoundationCollator::new(
        metadata
            .v0520_config
            .base_v0510
            .base_v0500
            .featurizer_config(),
        FoundationCollatorConfig {
            retention_time_objective: metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;

    let metrics = evaluate_properties_v0520(
        &model,
        &collator,
        &corpus.records,
        &holdout_indices,
        batch_size,
        &metadata.target_normalization,
        metadata.ms2_loss,
        &device,
    )?;

    fs::create_dir_all(&output_dir)?;
    write_summary(
        &output_dir.join("holdout_summary.tsv"),
        &training_yaml,
        &checkpoint,
        train_records,
        dev_records,
        holdout_indices.len(),
        batch_size,
        metadata.completed_epochs,
        metadata.completed_updates,
        metrics,
    )?;
    write_report(&output_dir.join("holdout_report.md"), metrics)?;

    println!("audit_version\t{VERSION}");
    println!("evaluation_partition\tTRAIN_HOLDOUT_ONCE");
    println!("checkpoint\t{}", checkpoint.display());
    println!("checkpoint_completed_epochs\t{}", metadata.completed_epochs);
    println!(
        "checkpoint_completed_updates\t{}",
        metadata.completed_updates
    );
    println!("holdout_records\t{}", holdout_indices.len());
    println!("rt_records\t{}", metrics.rt_records);
    println!("ms2_spectra\t{}", metrics.ms2_spectra);
    println!("ms2_fragments\t{}", metrics.ms2_fragments);
    println!("rt_mae_native\t{}", fmt_opt(metrics.rt_mae_native));
    println!("rt_rmse_native\t{}", fmt_opt(metrics.rt_rmse_native));
    println!("ms2_loss\t{}", fmt_opt(metrics.ms2_loss));
    println!("ms2_pointwise_mse\t{}", fmt_opt(metrics.ms2_pointwise_mse));
    println!("ms2_pointwise_mae\t{}", fmt_opt(metrics.ms2_pointwise_mae));
    println!("ms2_cosine\t{}", fmt_opt(metrics.ms2_mean_cosine));
    println!(
        "ms2_spectral_angle\t{}",
        fmt_opt(metrics.ms2_mean_spectral_angle)
    );
    println!("ms2_pearson\t{}", fmt_opt(metrics.ms2_mean_pearson));
    println!("train_holdout_consumed\tYES");
    println!("train_holdout_consumed_for_selection\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!("holdout_evaluation_complete\tYES");
    println!("holdout_out\t{}", output_dir.display());
    Ok(())
}

fn read_metadata(checkpoint: &Path) -> Result<V052Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.52 metadata {path:?}"))?,
    )
    .with_context(|| format!("parse v0.52 metadata {path:?}"))
}

fn validate_metadata(metadata: &V052Metadata) -> Result<()> {
    if metadata.version != V052_VERSION {
        anyhow::bail!(
            "expected v0.52 metadata version {V052_VERSION}, observed {}",
            metadata.version
        );
    }
    if metadata.objective != V052_OBJECTIVE {
        anyhow::bail!(
            "expected v0.52 objective {V052_OBJECTIVE:?}, observed {:?}",
            metadata.objective
        );
    }
    if metadata.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520 {
        anyhow::bail!(
            "expected v0.52 architecture {:?}, observed {:?}",
            FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520,
            metadata.architecture
        );
    }
    if metadata.smoke_mode || metadata.completed_updates == 0 || metadata.completed_epochs == 0 {
        anyhow::bail!("HOLDOUT requires a completed non-smoke v0.52 selected checkpoint");
    }
    metadata.v0520_config.validate()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_properties_v0520(
    model: &PeptideFoundationV0520Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    normalization: &FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    device: &Device,
) -> Result<PropertyMetrics> {
    let mut rt_abs = 0.0f64;
    let mut rt_sq = 0.0f64;
    let mut rt_n = 0usize;
    let mut ms2_objective_sum = 0.0f64;
    let mut ms2_objective_batches = 0usize;
    let mut ms2_shape = Ms2ShapeAccumulator::default();

    for chunk in indices.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|&index| records[index].clone())
            .collect::<Vec<_>>();
        let mut batch = collator.collate(&owned, device, 0)?;
        normalize_rt_target(&mut batch.targets, normalization)?;
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &owned,
            model.config().base_v0510.base_v0500.max_sequence_len,
            device,
        )?;
        let fragment = FoundationFragmentContextBatchV0350::from_records(
            &owned,
            &model.config().base_v0510.base_v0500.featurizer_config(),
            device,
        )?;
        let output =
            model.property_forward_t(&batch.input, &batch.context, &physics, &fragment, false)?;

        accumulate_regression(
            &output.rt,
            batch.targets.rt.as_ref(),
            batch.targets.rt_mask.as_ref(),
            &normalization.rt,
            &mut rt_abs,
            &mut rt_sq,
            &mut rt_n,
        )?;

        if let (Some(target), Some(mask)) = (&batch.targets.ms2, &batch.targets.ms2_mask) {
            let theoretical = fragment.channel_mask()?;
            let effective_mask = mask.broadcast_mul(&theoretical)?;
            let components = foundation_ms2_loss(&output.ms2, target, &effective_mask, ms2_loss)?;
            ms2_objective_sum += f64::from(components.total.to_scalar::<f32>()?);
            ms2_objective_batches += 1;
            ms2_shape.accumulate(&output.ms2, target, &effective_mask)?;
        }
    }

    if rt_n == 0 {
        anyhow::bail!("TRAIN-HOLDOUT has zero RT-labelled records");
    }
    if ms2_shape.spectrum_count == 0 {
        anyhow::bail!("TRAIN-HOLDOUT has zero MS2-labelled spectra");
    }

    Ok(PropertyMetrics {
        rt_mae_native: Some(rt_abs / rt_n as f64),
        rt_rmse_native: Some((rt_sq / rt_n as f64).sqrt()),
        ms2_loss: (ms2_objective_batches > 0)
            .then(|| ms2_objective_sum / ms2_objective_batches as f64),
        ms2_pointwise_mse: ms2_shape.pointwise_mse(),
        ms2_pointwise_mae: ms2_shape.pointwise_mae(),
        ms2_mean_cosine: ms2_shape.mean_cosine(),
        ms2_mean_spectral_angle: ms2_shape.mean_spectral_angle(),
        ms2_mean_pearson: ms2_shape.mean_pearson(),
        rt_records: rt_n,
        ms2_spectra: ms2_shape.spectrum_count,
        ms2_fragments: ms2_shape.fragment_count,
    })
}

fn normalize_rt_target(
    targets: &mut redeem_properties::foundation::FoundationTargets,
    normalization: &FoundationTargetNormalizationConfig,
) -> Result<()> {
    if let Some(rt) = targets.rt.take() {
        targets.rt = Some(normalization.rt.normalize_tensor(&rt)?);
    }
    Ok(())
}

fn accumulate_regression(
    prediction: &Tensor,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    normalization: &FoundationRegressionNormalization,
    abs_sum: &mut f64,
    sq_sum: &mut f64,
    count: &mut usize,
) -> Result<()> {
    let (Some(target), Some(mask)) = (target, mask) else {
        return Ok(());
    };
    let prediction = normalization
        .denormalize_tensor(prediction)?
        .to_vec2::<f32>()?;
    let target = normalization.denormalize_tensor(target)?.to_vec2::<f32>()?;
    let mask = mask.to_vec2::<f32>()?;
    for ((predicted, truth), observed) in prediction.iter().zip(&target).zip(&mask) {
        if observed[0] <= 0.0 {
            continue;
        }
        let error = f64::from(predicted[0] - truth[0]);
        *abs_sum += error.abs();
        *sq_sum += error * error;
        *count += 1;
    }
    Ok(())
}

#[derive(Debug, Default)]
struct Ms2ShapeAccumulator {
    fragment_count: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    spectrum_count: usize,
    cosine_sum: f64,
    spectral_angle_sum: f64,
    pearson_count: usize,
    pearson_sum: f64,
}

impl Ms2ShapeAccumulator {
    fn accumulate(&mut self, prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<()> {
        let predicted = prediction.to_vec3::<f32>()?;
        let targets = target.to_vec3::<f32>()?;
        let masks = mask.to_vec3::<f32>()?;
        for batch_index in 0..predicted.len() {
            let mut pred_values = Vec::new();
            let mut target_values = Vec::new();
            for residue_index in 0..predicted[batch_index].len() {
                for channel_index in 0..predicted[batch_index][residue_index].len() {
                    if masks[batch_index][residue_index][channel_index] <= 0.0 {
                        continue;
                    }
                    let pred = f64::from(predicted[batch_index][residue_index][channel_index]);
                    let truth = f64::from(targets[batch_index][residue_index][channel_index]);
                    let error = pred - truth;
                    self.fragment_count += 1;
                    self.squared_error_sum += error * error;
                    self.absolute_error_sum += error.abs();
                    pred_values.push(pred);
                    target_values.push(truth);
                }
            }
            if pred_values.is_empty() {
                continue;
            }
            let dot = pred_values
                .iter()
                .zip(&target_values)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let pnorm = pred_values.iter().map(|v| v * v).sum::<f64>().sqrt();
            let tnorm = target_values.iter().map(|v| v * v).sum::<f64>().sqrt();
            let cosine = if pnorm > 0.0 && tnorm > 0.0 {
                (dot / (pnorm * tnorm)).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            self.spectrum_count += 1;
            self.cosine_sum += cosine;
            self.spectral_angle_sum += 1.0 - (2.0 / std::f64::consts::PI) * cosine.acos();
            if let Some(pearson) = pearson_correlation(&pred_values, &target_values) {
                self.pearson_count += 1;
                self.pearson_sum += pearson;
            }
        }
        Ok(())
    }

    fn pointwise_mse(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.squared_error_sum / self.fragment_count as f64)
    }

    fn pointwise_mae(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.absolute_error_sum / self.fragment_count as f64)
    }

    fn mean_cosine(&self) -> Option<f64> {
        (self.spectrum_count > 0).then(|| self.cosine_sum / self.spectrum_count as f64)
    }

    fn mean_spectral_angle(&self) -> Option<f64> {
        (self.spectrum_count > 0).then(|| self.spectral_angle_sum / self.spectrum_count as f64)
    }

    fn mean_pearson(&self) -> Option<f64> {
        (self.pearson_count > 0).then(|| self.pearson_sum / self.pearson_count as f64)
    }
}

fn pearson_correlation(first: &[f64], second: &[f64]) -> Option<f64> {
    if first.len() != second.len() || first.len() < 2 {
        return None;
    }
    let n = first.len() as f64;
    let first_mean = first.iter().sum::<f64>() / n;
    let second_mean = second.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut first_variance = 0.0;
    let mut second_variance = 0.0;
    for (&a, &b) in first.iter().zip(second) {
        let da = a - first_mean;
        let db = b - second_mean;
        covariance += da * db;
        first_variance += da * da;
        second_variance += db * db;
    }
    let denominator = (first_variance * second_variance).sqrt();
    (denominator > 0.0).then(|| (covariance / denominator).clamp(-1.0, 1.0))
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    path: &Path,
    run_yaml: &Path,
    checkpoint: &Path,
    train_records: usize,
    dev_records: usize,
    holdout_records: usize,
    batch_size: usize,
    completed_epochs: usize,
    completed_updates: usize,
    metrics: PropertyMetrics,
) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "metric\tvalue")?;
    writeln!(out, "audit_version\t{VERSION}")?;
    writeln!(out, "evaluation_partition\tTRAIN_HOLDOUT_ONCE")?;
    writeln!(out, "run_yaml\t{}", run_yaml.display())?;
    writeln!(out, "checkpoint\t{}", checkpoint.display())?;
    writeln!(out, "checkpoint_completed_epochs\t{completed_epochs}")?;
    writeln!(out, "checkpoint_completed_updates\t{completed_updates}")?;
    writeln!(out, "train_records_reserved\t{train_records}")?;
    writeln!(out, "dev_records_reserved\t{dev_records}")?;
    writeln!(out, "holdout_records\t{holdout_records}")?;
    writeln!(out, "batch_size\t{batch_size}")?;
    writeln!(out, "rt_records\t{}", metrics.rt_records)?;
    writeln!(out, "ms2_spectra\t{}", metrics.ms2_spectra)?;
    writeln!(out, "ms2_fragments\t{}", metrics.ms2_fragments)?;
    writeln!(out, "rt_mae_native\t{}", fmt_opt(metrics.rt_mae_native))?;
    writeln!(out, "rt_rmse_native\t{}", fmt_opt(metrics.rt_rmse_native))?;
    writeln!(out, "ms2_loss\t{}", fmt_opt(metrics.ms2_loss))?;
    writeln!(
        out,
        "ms2_pointwise_mse\t{}",
        fmt_opt(metrics.ms2_pointwise_mse)
    )?;
    writeln!(
        out,
        "ms2_pointwise_mae\t{}",
        fmt_opt(metrics.ms2_pointwise_mae)
    )?;
    writeln!(out, "ms2_cosine\t{}", fmt_opt(metrics.ms2_mean_cosine))?;
    writeln!(
        out,
        "ms2_spectral_angle\t{}",
        fmt_opt(metrics.ms2_mean_spectral_angle)
    )?;
    writeln!(out, "ms2_pearson\t{}", fmt_opt(metrics.ms2_mean_pearson))?;
    writeln!(out, "train_holdout_consumed\tYES")?;
    writeln!(out, "train_holdout_consumed_for_selection\tNO")?;
    writeln!(out, "historical_validation_consumed\tNO")?;
    writeln!(out, "historical_test_consumed\tNO")?;
    out.flush()?;
    Ok(())
}

fn write_report(path: &Path, metrics: PropertyMetrics) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "# ReDeeM v0.52 one-time TRAIN-HOLDOUT evaluation")?;
    writeln!(out)?;
    writeln!(out, "- RT records: `{}`", metrics.rt_records)?;
    writeln!(out, "- MS2 spectra: `{}`", metrics.ms2_spectra)?;
    writeln!(out, "- RT MAE: `{}`", fmt_opt(metrics.rt_mae_native))?;
    writeln!(out, "- RT RMSE: `{}`", fmt_opt(metrics.rt_rmse_native))?;
    writeln!(out, "- MS2 cosine: `{}`", fmt_opt(metrics.ms2_mean_cosine))?;
    writeln!(
        out,
        "- MS2 spectral angle: `{}`",
        fmt_opt(metrics.ms2_mean_spectral_angle)
    )?;
    writeln!(
        out,
        "- MS2 Pearson: `{}`",
        fmt_opt(metrics.ms2_mean_pearson)
    )?;
    writeln!(out)?;
    writeln!(out, "This is the one-time reserved TRAIN-HOLDOUT evaluation of the already selected v0.52 checkpoint. It is descriptive only and must not be used for checkpoint selection or post-hoc tuning.")?;
    writeln!(out)?;
    writeln!(
        out,
        "Historical VALIDATION/APD and historical TEST remain unopened by this evaluator."
    )?;
    out.flush()?;
    Ok(())
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.8}"))
        .unwrap_or_else(|| "NA".into())
}

fn parse_or<T>(args: &[String], index: usize, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse::<T>()
            .map_err(|err| anyhow::anyhow!("failed to parse argument {index}={value:?}: {err}")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_contract_rejects_smoke_or_wrong_version() {
        let mut metadata = V052Metadata {
            version: V052_VERSION,
            objective: V052_OBJECTIVE.into(),
            architecture: FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520.into(),
            v0520_config: PeptideFoundationV0520Config::default(),
            rt_objective: RetentionTimeObjective::default(),
            target_normalization: FoundationTargetNormalizationConfig::default(),
            ms2_loss: FoundationMs2LossConfig::default(),
            completed_epochs: 1,
            completed_updates: 1,
            smoke_mode: false,
        };
        assert!(validate_metadata(&metadata).is_ok());
        metadata.smoke_mode = true;
        assert!(validate_metadata(&metadata).is_err());
        metadata.smoke_mode = false;
        metadata.version = 519;
        assert!(validate_metadata(&metadata).is_err());
    }
}
