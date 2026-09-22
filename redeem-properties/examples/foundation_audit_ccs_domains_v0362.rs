//! v0.36.2 CCS residual/domain diagnostic from the frozen v0.35 forward model.
//!
//! This executable is intentionally non-training. It exports native-unit CCS
//! residuals for every CCS-labelled TRAIN and DEV record, summarizes systematic
//! error by biological/acquisition strata, fits a fixed set of calibration
//! hypotheses on TRAIN only, and evaluates them on DEV. TRAIN-HOLDOUT and
//! historical VALIDATION/TEST are never evaluated.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_peptidoform_neutral_mass, load_foundation_corpus,
    read_foundation_training_run_config, FoundationBenchmarkManifest, FoundationCollator,
    FoundationCollatorConfig, FoundationCorruptionConfig, FoundationPartition,
    FoundationRegressionNormalization, FoundationTargetNormalizationConfig,
    FoundationTrainingRecord, PeptideFoundationMultimodalV0350Config,
    PeptideFoundationMultimodalV0350Model, RetentionTimeObjective,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const V0362_VERSION: u32 = 362;
const V0362_OBJECTIVE: &str = "v0362_ccs_train_dev_domain_residual_calibration_audit";
const DEFAULT_BATCH_SIZE: usize = 512;
const GROUP_AFFINE_SHRINKAGE: f64 = 1024.0;
const SOURCE_CHARGE_AFFINE_SHRINKAGE: f64 = 512.0;
const RIDGE_LAMBDA: f64 = 100.0;
const MATERIAL_DEV_RATIO: f64 = 0.90;
const SOURCE_SIGNAL_MARGIN: f64 = 0.02;
const INSTRUMENT_SLOTS: usize = 16;

#[derive(Debug, Clone, Deserialize)]
struct V035ParentMetadata {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    completed_steps: usize,
    v0350_config: PeptideFoundationMultimodalV0350Config,
}

#[derive(Debug, Clone)]
struct ResidualRow {
    record_index: usize,
    source_id: String,
    source_record_index: usize,
    target: f64,
    prediction: f64,
    charge: Option<i32>,
    precursor_mz: Option<f64>,
    neutral_mass: f64,
    sequence_len: usize,
    modification_count: usize,
    total_mod_mass: f64,
    absolute_mod_mass: f64,
    instrument_id: Option<u32>,
    instrument_name: Option<String>,
    run_id: Option<String>,
}

impl ResidualRow {
    fn residual(&self) -> f64 {
        self.target - self.prediction
    }

    fn charge_key(&self) -> String {
        match self.charge {
            Some(charge) if charge >= 5 => "5+".to_string(),
            Some(charge) => charge.to_string(),
            None => "missing".to_string(),
        }
    }

    fn mz_bin(&self) -> &'static str {
        match self.precursor_mz {
            None => "missing",
            Some(value) if value < 400.0 => "<400",
            Some(value) if value < 600.0 => "400-600",
            Some(value) if value < 800.0 => "600-800",
            Some(value) if value < 1000.0 => "800-1000",
            Some(_) => ">=1000",
        }
    }

    fn mass_bin(&self) -> &'static str {
        match self.neutral_mass {
            value if value < 1000.0 => "<1000",
            value if value < 1500.0 => "1000-1500",
            value if value < 2000.0 => "1500-2000",
            value if value < 2500.0 => "2000-2500",
            value if value < 3000.0 => "2500-3000",
            _ => ">=3000",
        }
    }

    fn length_bin(&self) -> &'static str {
        match self.sequence_len {
            0..=7 => "<=7",
            8..=10 => "8-10",
            11..=15 => "11-15",
            16..=20 => "16-20",
            21..=30 => "21-30",
            _ => "31+",
        }
    }

    fn ptm_presence(&self) -> &'static str {
        if self.modification_count == 0 {
            "unmodified"
        } else {
            "modified"
        }
    }

    fn ptm_mass_bin(&self) -> &'static str {
        match self.absolute_mod_mass {
            value if value <= 1.0e-9 => "0",
            value if value <= 50.0 => "(0,50]",
            value if value <= 100.0 => "(50,100]",
            value if value <= 200.0 => "(100,200]",
            _ => ">200",
        }
    }

    fn instrument_key(&self) -> String {
        if let Some(name) = self
            .instrument_name
            .as_ref()
            .filter(|name| !name.trim().is_empty())
        {
            name.clone()
        } else if let Some(id) = self.instrument_id {
            format!("instrument_id_{id}")
        } else {
            "missing".to_string()
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct Metrics {
    records: usize,
    mae: f64,
    rmse: f64,
    pearson: f64,
    mean_error: f64,
    median_error: f64,
    residual_std: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Affine {
    intercept: f64,
    slope: f64,
}

#[derive(Debug, Clone)]
struct GroupAffine {
    records: usize,
    blend: f64,
    affine: Affine,
}

#[derive(Debug, Clone)]
struct RidgeCalibrator {
    feature_names: Vec<String>,
    coefficients: Vec<f64>,
    source_offsets: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticSummary {
    version: u32,
    objective: String,
    parent_checkpoint: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    train_ccs_records: usize,
    dev_ccs_records: usize,
    holdout_consumed: bool,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
    baseline_dev_mae: f64,
    best_calibrator: String,
    best_dev_mae: f64,
    best_dev_ratio: f64,
    best_source_aware_calibrator: String,
    best_source_aware_dev_mae: f64,
    best_source_agnostic_calibrator: String,
    best_source_agnostic_dev_mae: f64,
    source_domain_signal: bool,
    material_calibration_gain: bool,
    v0370_domain_calibration_recommended: bool,
    ccs_corpus_harmonization_recommended: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        anyhow::bail!(
            "usage: foundation_audit_ccs_domains_v0362 RUN_V0260.yaml OUTPUT_DIR PARENT_V0350_CHECKPOINT [batch_size=512]"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let batch_size = args
        .get(4)
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("invalid v0.36.2 batch size")?
        .unwrap_or(DEFAULT_BATCH_SIZE);
    if batch_size == 0 {
        anyhow::bail!("v0.36.2 batch_size must be positive");
    }
    if output_root.exists() {
        anyhow::bail!("v0.36.2 output directory already exists: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.36.2 requires a CUDA device")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let train_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .filter(|&index| {
            corpus.records[index]
                .ccs
                .is_some_and(|value| value.is_finite())
        })
        .collect();
    let dev_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Validation)
        .map(|entry| entry.record_index)
        .filter(|&index| {
            corpus.records[index]
                .ccs
                .is_some_and(|value| value.is_finite())
        })
        .collect();
    if train_indices.is_empty() || dev_indices.is_empty() {
        anyhow::bail!("v0.36.2 requires finite CCS labels in TRAIN and DEV");
    }

    let parent_metadata = read_parent_metadata(&parent_checkpoint)?;
    if parent_metadata.version != 350
        || parent_metadata.objective
            != "v0350_trainable_forward_representation_context_conditioned_ms2"
    {
        anyhow::bail!(
            "v0.36.2 requires accepted v0.35 final metadata; observed version={} objective={}",
            parent_metadata.version,
            parent_metadata.objective
        );
    }
    if parent_metadata.completed_steps == 0 {
        anyhow::bail!("v0.36.2 refuses an unselected v0.35 checkpoint");
    }
    parent_metadata.v0350_config.validate()?;
    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != corpus_fingerprint
        || parent_metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
    {
        anyhow::bail!("v0.36.2 data provenance differs from frozen v0.35 parent");
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model =
        PeptideFoundationMultimodalV0350Model::new(parent_metadata.v0350_config.clone(), vb)?;
    let loaded_variables = load_exact_parent(
        &varmap,
        &parent_checkpoint.join("model.safetensors"),
        &device,
    )?;

    let collator = FoundationCollator::new(
        parent_metadata.v0350_config.forward().clone(),
        FoundationCollatorConfig {
            retention_time_objective: parent_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;

    fs::create_dir_all(&output_root)?;
    println!("v0362_version\tv0.36.2-ccs-domain-residual-diagnostic");
    println!("objective\t{V0362_OBJECTIVE}");
    println!("parent_v0350_checkpoint\t{}", parent_checkpoint.display());
    println!(
        "parent_v0350_completed_steps\t{}",
        parent_metadata.completed_steps
    );
    println!("loaded_parent_variables\t{loaded_variables}");
    println!("device\t{device:?}");
    println!("batch_size\t{batch_size}");
    println!("train_ccs_records\t{}", train_indices.len());
    println!("dev_ccs_records\t{}", dev_indices.len());
    println!("holdout_evaluated\tNO");
    println!("historical_validation_reused_for_v0362_selection\tNO");
    println!("historical_test_consumed\tNO");

    let train_rows = predict_rows(
        &model,
        &collator,
        &corpus.records,
        &corpus.provenance,
        &train_indices,
        batch_size,
        &parent_metadata.target_normalization.ccs,
        &device,
    )?;
    let dev_rows = predict_rows(
        &model,
        &collator,
        &corpus.records,
        &corpus.provenance,
        &dev_indices,
        batch_size,
        &parent_metadata.target_normalization.ccs,
        &device,
    )?;

    let baseline_train = metrics_for(&train_rows, |row| row.prediction)?;
    let baseline_dev = metrics_for(&dev_rows, |row| row.prediction)?;
    print_metric("baseline_train", baseline_train);
    print_metric("baseline_dev", baseline_dev);

    write_dev_residuals(&output_root.join("ccs_dev_residuals.tsv"), &dev_rows)?;
    write_strata(
        &output_root.join("ccs_residual_strata.tsv"),
        &train_rows,
        &dev_rows,
    )?;

    let global_affine = fit_affine(&train_rows);
    let charge_affines = fit_group_affines(
        &train_rows,
        |row| row.charge_key(),
        global_affine,
        GROUP_AFFINE_SHRINKAGE,
    );
    let source_affines = fit_group_affines(
        &train_rows,
        |row| row.source_id.clone(),
        global_affine,
        GROUP_AFFINE_SHRINKAGE,
    );
    let source_charge_affines = fit_group_affines(
        &train_rows,
        |row| format!("{}|{}", row.source_id, row.charge_key()),
        global_affine,
        SOURCE_CHARGE_AFFINE_SHRINKAGE,
    );

    let physics_ridge = fit_ridge(&train_rows, false)?;
    let source_physics_ridge = fit_ridge(&train_rows, true)?;

    let mut calibration_metrics = BTreeMap::<String, (Metrics, Metrics, bool)>::new();
    insert_calibration(
        &mut calibration_metrics,
        "baseline",
        metrics_for(&train_rows, |row| row.prediction)?,
        metrics_for(&dev_rows, |row| row.prediction)?,
        false,
    );
    insert_calibration(
        &mut calibration_metrics,
        "global_affine",
        metrics_for(&train_rows, |row| global_affine.predict(row.prediction))?,
        metrics_for(&dev_rows, |row| global_affine.predict(row.prediction))?,
        false,
    );
    insert_calibration(
        &mut calibration_metrics,
        "charge_affine",
        metrics_for(&train_rows, |row| {
            group_predict(row, &charge_affines, row.charge_key(), global_affine)
        })?,
        metrics_for(&dev_rows, |row| {
            group_predict(row, &charge_affines, row.charge_key(), global_affine)
        })?,
        false,
    );
    insert_calibration(
        &mut calibration_metrics,
        "source_affine",
        metrics_for(&train_rows, |row| {
            group_predict(row, &source_affines, row.source_id.clone(), global_affine)
        })?,
        metrics_for(&dev_rows, |row| {
            group_predict(row, &source_affines, row.source_id.clone(), global_affine)
        })?,
        true,
    );
    insert_calibration(
        &mut calibration_metrics,
        "source_charge_affine",
        metrics_for(&train_rows, |row| {
            let key = format!("{}|{}", row.source_id, row.charge_key());
            group_predict(row, &source_charge_affines, key, global_affine)
        })?,
        metrics_for(&dev_rows, |row| {
            let key = format!("{}|{}", row.source_id, row.charge_key());
            group_predict(row, &source_charge_affines, key, global_affine)
        })?,
        true,
    );
    insert_calibration(
        &mut calibration_metrics,
        "physics_ridge",
        metrics_for(&train_rows, |row| physics_ridge.predict(row))?,
        metrics_for(&dev_rows, |row| physics_ridge.predict(row))?,
        false,
    );
    insert_calibration(
        &mut calibration_metrics,
        "source_physics_ridge",
        metrics_for(&train_rows, |row| source_physics_ridge.predict(row))?,
        metrics_for(&dev_rows, |row| source_physics_ridge.predict(row))?,
        true,
    );

    write_calibration_summary(
        &output_root.join("ccs_calibration_summary.tsv"),
        baseline_dev.mae,
        &calibration_metrics,
    )?;
    write_affine_parameters(
        &output_root.join("ccs_affine_parameters.tsv"),
        global_affine,
        &charge_affines,
        &source_affines,
        &source_charge_affines,
    )?;
    write_ridge_parameters(
        &output_root.join("ccs_ridge_parameters.tsv"),
        &physics_ridge,
        &source_physics_ridge,
    )?;

    let (best_name, best_dev, _) = best_calibration(&calibration_metrics, |_| true)?;
    let (best_source_name, best_source_dev, _) =
        best_calibration(&calibration_metrics, |source_aware| source_aware)?;
    let (best_agnostic_name, best_agnostic_dev, _) =
        best_calibration(&calibration_metrics, |source_aware| !source_aware)?;
    let best_ratio = best_dev.mae / baseline_dev.mae;
    let source_margin = (best_agnostic_dev.mae - best_source_dev.mae) / baseline_dev.mae;
    let source_domain_signal = source_margin >= SOURCE_SIGNAL_MARGIN;
    let material_gain = best_ratio <= MATERIAL_DEV_RATIO;
    let v0370_recommended = material_gain && source_domain_signal;
    let corpus_harmonization_recommended = !material_gain;

    println!("v0362_best_calibrator\t{best_name}");
    println!("v0362_best_dev_ccs_mae\t{:.8}", best_dev.mae);
    println!("v0362_best_dev_ccs_ratio\t{best_ratio:.8}");
    println!("v0362_best_source_aware_calibrator\t{best_source_name}");
    println!(
        "v0362_best_source_aware_dev_ccs_mae\t{:.8}",
        best_source_dev.mae
    );
    println!("v0362_best_source_agnostic_calibrator\t{best_agnostic_name}");
    println!(
        "v0362_best_source_agnostic_dev_ccs_mae\t{:.8}",
        best_agnostic_dev.mae
    );
    println!("v0362_source_domain_margin_ratio\t{source_margin:.8}");
    println!(
        "v0362_source_domain_signal\t{}",
        yes_no(source_domain_signal)
    );
    println!("v0362_material_calibration_gain\t{}", yes_no(material_gain));
    println!(
        "v0362_v0370_domain_calibration_recommended\t{}",
        yes_no(v0370_recommended)
    );
    println!(
        "v0362_ccs_corpus_harmonization_recommended\t{}",
        yes_no(corpus_harmonization_recommended)
    );
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0362_selection\tNO");
    println!("historical_test_consumed\tNO");

    let summary = DiagnosticSummary {
        version: V0362_VERSION,
        objective: V0362_OBJECTIVE.to_string(),
        parent_checkpoint: parent_checkpoint.display().to_string(),
        corpus_fingerprint,
        benchmark_manifest_fingerprint: benchmark_fingerprint,
        train_ccs_records: train_rows.len(),
        dev_ccs_records: dev_rows.len(),
        holdout_consumed: false,
        historical_validation_consumed: false,
        historical_test_consumed: false,
        baseline_dev_mae: baseline_dev.mae,
        best_calibrator: best_name.clone(),
        best_dev_mae: best_dev.mae,
        best_dev_ratio: best_ratio,
        best_source_aware_calibrator: best_source_name.clone(),
        best_source_aware_dev_mae: best_source_dev.mae,
        best_source_agnostic_calibrator: best_agnostic_name.clone(),
        best_source_agnostic_dev_mae: best_agnostic_dev.mae,
        source_domain_signal,
        material_calibration_gain: material_gain,
        v0370_domain_calibration_recommended: v0370_recommended,
        ccs_corpus_harmonization_recommended: corpus_harmonization_recommended,
    };
    fs::write(
        output_root.join("ccs_domain_diagnostic_summary.yaml"),
        serde_yaml::to_string(&summary)?,
    )?;

    Ok(())
}

impl Affine {
    fn predict(self, value: f64) -> f64 {
        self.intercept + self.slope * value
    }
}

impl RidgeCalibrator {
    fn predict(&self, row: &ResidualRow) -> f64 {
        let features = ridge_features(row, &self.source_offsets);
        row.prediction + dot(&self.coefficients, &features)
    }
}

fn read_parent_metadata(checkpoint: &Path) -> Result<V035ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.35 metadata {path:?}"))
}

fn load_exact_parent(varmap: &VarMap, checkpoint: &Path, device: &Device) -> Result<usize> {
    let tensors = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("failed to load v0.35 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.36.2 VarMap lock poisoned"))?;
    let mut loaded = 0usize;
    let mut used = BTreeSet::<String>::new();
    let mut missing = Vec::new();
    for (name, variable) in data.iter() {
        match tensors.get(name) {
            Some(tensor) => {
                if tensor.dims() != variable.as_tensor().dims() {
                    anyhow::bail!(
                        "v0.36.2 parent shape mismatch for {name}: parent {:?}, model {:?}",
                        tensor.dims(),
                        variable.as_tensor().dims()
                    );
                }
                variable.set(tensor)?;
                loaded += 1;
                used.insert(name.clone());
            }
            None => missing.push(name.clone()),
        }
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "v0.36.2 parent is missing variables: {}",
            missing.join(", ")
        );
    }
    let ignored = tensors.keys().filter(|name| !used.contains(*name)).count();
    if ignored != 0 {
        anyhow::bail!("v0.36.2 parent contains {ignored} unexpected tensors");
    }
    Ok(loaded)
}

#[allow(clippy::too_many_arguments)]
fn predict_rows(
    model: &PeptideFoundationMultimodalV0350Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    indices: &[usize],
    batch_size: usize,
    ccs_normalization: &FoundationRegressionNormalization,
    device: &Device,
) -> Result<Vec<ResidualRow>> {
    let mut rows = Vec::with_capacity(indices.len());
    for chunk in indices.chunks(batch_size) {
        let owned: Vec<FoundationTrainingRecord> =
            chunk.iter().map(|&index| records[index].clone()).collect();
        let batch = collator.collate(&owned, device, 0)?;
        let ccs = model.protected_ccs_v0350_t(&batch.input, &batch.context)?;
        let ccs = ccs_normalization
            .denormalize_tensor(&ccs)?
            .to_vec2::<f32>()?;
        for (position, &index) in chunk.iter().enumerate() {
            let record = &records[index];
            let provenance = provenance
                .get(index)
                .ok_or_else(|| anyhow::anyhow!("missing provenance for record {index}"))?;
            let target = f64::from(
                record
                    .ccs
                    .ok_or_else(|| anyhow::anyhow!("CCS-labelled index {index} lost target"))?,
            );
            let prediction = f64::from(ccs[position][0]);
            let neutral_mass =
                foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(|error| {
                    anyhow::anyhow!("failed neutral-mass calculation for record {index}: {error}")
                })?;
            let modification_count = record.peptidoform.modifications.len();
            let total_mod_mass = record
                .peptidoform
                .modifications
                .iter()
                .map(|item| f64::from(item.mass_delta))
                .sum::<f64>();
            let absolute_mod_mass = record
                .peptidoform
                .modifications
                .iter()
                .map(|item| f64::from(item.mass_delta).abs())
                .sum::<f64>();
            rows.push(ResidualRow {
                record_index: index,
                source_id: provenance.source_id.clone(),
                source_record_index: provenance.source_record_index,
                target,
                prediction,
                charge: record.context.charge,
                precursor_mz: record.context.precursor_mz.map(f64::from),
                neutral_mass,
                sequence_len: record.peptidoform.sequence.chars().count(),
                modification_count,
                total_mod_mass,
                absolute_mod_mass,
                instrument_id: record.context.instrument_id,
                instrument_name: record.context.instrument_name.clone(),
                run_id: record.run_id.clone(),
            });
        }
    }
    Ok(rows)
}

fn metrics_for<F>(rows: &[ResidualRow], mut predictor: F) -> Result<Metrics>
where
    F: FnMut(&ResidualRow) -> f64,
{
    if rows.is_empty() {
        anyhow::bail!("cannot calculate CCS metrics on zero rows");
    }
    let mut abs = 0.0;
    let mut sq = 0.0;
    let mut error_sum = 0.0;
    let mut errors = Vec::with_capacity(rows.len());
    let mut targets = Vec::with_capacity(rows.len());
    let mut predictions = Vec::with_capacity(rows.len());
    for row in rows {
        let prediction = predictor(row);
        if !prediction.is_finite() {
            anyhow::bail!(
                "non-finite calibrated CCS prediction for record {}",
                row.record_index
            );
        }
        let error = prediction - row.target;
        abs += error.abs();
        sq += error * error;
        error_sum += error;
        errors.push(error);
        targets.push(row.target);
        predictions.push(prediction);
    }
    errors.sort_by(|a, b| a.total_cmp(b));
    let mean_error = error_sum / rows.len() as f64;
    let residual_var = errors
        .iter()
        .map(|value| {
            let delta = value - mean_error;
            delta * delta
        })
        .sum::<f64>()
        / rows.len() as f64;
    Ok(Metrics {
        records: rows.len(),
        mae: abs / rows.len() as f64,
        rmse: (sq / rows.len() as f64).sqrt(),
        pearson: pearson(&targets, &predictions).unwrap_or(0.0),
        mean_error,
        median_error: median_sorted(&errors),
        residual_std: residual_var.sqrt(),
    })
}

fn fit_affine(rows: &[ResidualRow]) -> Affine {
    let n = rows.len() as f64;
    let mean_x = rows.iter().map(|row| row.prediction).sum::<f64>() / n;
    let mean_y = rows.iter().map(|row| row.target).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance = 0.0;
    for row in rows {
        let dx = row.prediction - mean_x;
        covariance += dx * (row.target - mean_y);
        variance += dx * dx;
    }
    let slope = if variance > 1.0e-12 {
        covariance / variance
    } else {
        1.0
    };
    Affine {
        intercept: mean_y - slope * mean_x,
        slope,
    }
}

fn fit_group_affines<F>(
    rows: &[ResidualRow],
    mut key_fn: F,
    global: Affine,
    shrinkage: f64,
) -> BTreeMap<String, GroupAffine>
where
    F: FnMut(&ResidualRow) -> String,
{
    let mut grouped = BTreeMap::<String, Vec<ResidualRow>>::new();
    for row in rows {
        grouped.entry(key_fn(row)).or_default().push(row.clone());
    }
    grouped
        .into_iter()
        .map(|(key, group)| {
            let raw = fit_affine(&group);
            let blend = group.len() as f64 / (group.len() as f64 + shrinkage);
            let affine = Affine {
                intercept: global.intercept + blend * (raw.intercept - global.intercept),
                slope: global.slope + blend * (raw.slope - global.slope),
            };
            (
                key,
                GroupAffine {
                    records: group.len(),
                    blend,
                    affine,
                },
            )
        })
        .collect()
}

fn group_predict(
    row: &ResidualRow,
    groups: &BTreeMap<String, GroupAffine>,
    key: String,
    fallback: Affine,
) -> f64 {
    groups
        .get(&key)
        .map(|group| group.affine)
        .unwrap_or(fallback)
        .predict(row.prediction)
}

fn fit_ridge(rows: &[ResidualRow], include_source: bool) -> Result<RidgeCalibrator> {
    let source_offsets = if include_source {
        let mut sources: Vec<String> = rows.iter().map(|row| row.source_id.clone()).collect();
        sources.sort();
        sources.dedup();
        sources
            .into_iter()
            .skip(1)
            .enumerate()
            .map(|(offset, source)| (source, offset))
            .collect::<BTreeMap<_, _>>()
    } else {
        BTreeMap::new()
    };
    let feature_names = ridge_feature_names(&source_offsets);
    let dimension = feature_names.len();
    let mut normal = vec![vec![0.0f64; dimension]; dimension];
    let mut rhs = vec![0.0f64; dimension];
    for row in rows {
        let features = ridge_features(row, &source_offsets);
        let target_residual = row.target - row.prediction;
        for i in 0..dimension {
            rhs[i] += features[i] * target_residual;
            for j in 0..=i {
                normal[i][j] += features[i] * features[j];
            }
        }
    }
    for i in 0..dimension {
        for j in 0..i {
            normal[j][i] = normal[i][j];
        }
    }
    for i in 1..dimension {
        normal[i][i] += RIDGE_LAMBDA;
    }
    let coefficients = solve_linear_system(normal, rhs)?;
    Ok(RidgeCalibrator {
        feature_names,
        coefficients,
        source_offsets,
    })
}

fn ridge_feature_names(source_offsets: &BTreeMap<String, usize>) -> Vec<String> {
    let mut names = vec![
        "intercept".to_string(),
        "parent_prediction_over_200".to_string(),
        "charge_over_4".to_string(),
        "charge_squared_over_16".to_string(),
        "precursor_mz_over_1000".to_string(),
        "neutral_mass_over_3000".to_string(),
        "sequence_len_over_30".to_string(),
        "modification_count_over_4".to_string(),
        "total_mod_mass_over_500".to_string(),
        "absolute_mod_mass_over_500".to_string(),
    ];
    for slot in 0..INSTRUMENT_SLOTS {
        names.push(format!("instrument_id_{slot}"));
    }
    let mut sources: Vec<(&String, &usize)> = source_offsets.iter().collect();
    sources.sort_by_key(|(_, offset)| **offset);
    for (source, _) in sources {
        names.push(format!("source::{source}"));
    }
    names
}

fn ridge_features(row: &ResidualRow, source_offsets: &BTreeMap<String, usize>) -> Vec<f64> {
    let charge = f64::from(row.charge.unwrap_or(0));
    let mz = row.precursor_mz.unwrap_or(0.0);
    let mut values = vec![
        1.0,
        row.prediction / 200.0,
        charge / 4.0,
        charge * charge / 16.0,
        mz / 1000.0,
        row.neutral_mass / 3000.0,
        row.sequence_len as f64 / 30.0,
        row.modification_count as f64 / 4.0,
        row.total_mod_mass / 500.0,
        row.absolute_mod_mass / 500.0,
    ];
    for slot in 0..INSTRUMENT_SLOTS {
        values.push(if row.instrument_id == Some(slot as u32) {
            1.0
        } else {
            0.0
        });
    }
    let source_base = values.len();
    values.resize(source_base + source_offsets.len(), 0.0);
    if let Some(&offset) = source_offsets.get(&row.source_id) {
        values[source_base + offset] = 1.0;
    }
    values
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        anyhow::bail!("invalid ridge normal-equation dimensions");
    }
    for pivot in 0..n {
        let mut best = pivot;
        for row in (pivot + 1)..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if matrix[best][pivot].abs() < 1.0e-12 {
            anyhow::bail!("singular ridge normal equations at pivot {pivot}");
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for col in pivot..n {
            matrix[pivot][col] /= divisor;
        }
        rhs[pivot] /= divisor;
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor.abs() <= 1.0e-18 {
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

fn write_dev_residuals(path: &Path, rows: &[ResidualRow]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "record_index\tsource_id\tsource_record_index\ttarget_ccs\tpredicted_ccs\tresidual_target_minus_prediction\tcharge\tprecursor_mz\tneutral_mass\tsequence_len\tmodification_count\ttotal_mod_mass\tabsolute_mod_mass\tinstrument_id\tinstrument_name\trun_id"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{:.8}\t{}\t{}\t{:.8}\t{:.8}\t{}\t{}\t{}",
            row.record_index,
            escape_tsv(&row.source_id),
            row.source_record_index,
            row.target,
            row.prediction,
            row.residual(),
            row.charge
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.precursor_mz
                .map(|value| format!("{value:.8}"))
                .unwrap_or_default(),
            row.neutral_mass,
            row.sequence_len,
            row.modification_count,
            row.total_mod_mass,
            row.absolute_mod_mass,
            row.instrument_id
                .map(|value| value.to_string())
                .unwrap_or_default(),
            escape_tsv(row.instrument_name.as_deref().unwrap_or("")),
            escape_tsv(row.run_id.as_deref().unwrap_or("")),
        )?;
    }
    Ok(())
}

fn write_strata(path: &Path, train: &[ResidualRow], dev: &[ResidualRow]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "partition\tdimension\tstratum\trecords\tmae\trmse\tpearson\tmean_error_prediction_minus_target\tmedian_error_prediction_minus_target\tresidual_std"
    )?;
    for (partition, rows) in [("TRAIN", train), ("DEV", dev)] {
        for (dimension, key_fn) in stratum_functions() {
            let mut groups = BTreeMap::<String, Vec<ResidualRow>>::new();
            for row in rows {
                groups.entry(key_fn(row)).or_default().push(row.clone());
            }
            for (key, group) in groups {
                let metrics = metrics_for(&group, |row| row.prediction)?;
                write_metric_row(&mut writer, partition, dimension, &key, metrics)?;
            }
        }
    }
    Ok(())
}

type StratumFn = fn(&ResidualRow) -> String;

fn stratum_functions() -> Vec<(&'static str, StratumFn)> {
    vec![
        ("source", |row| row.source_id.clone()),
        ("charge", |row| row.charge_key()),
        ("source_charge", |row| {
            format!("{}|{}", row.source_id, row.charge_key())
        }),
        ("mz_bin", |row| row.mz_bin().to_string()),
        ("neutral_mass_bin", |row| row.mass_bin().to_string()),
        ("sequence_length_bin", |row| row.length_bin().to_string()),
        ("ptm_presence", |row| row.ptm_presence().to_string()),
        ("ptm_mass_bin", |row| row.ptm_mass_bin().to_string()),
        ("instrument", |row| row.instrument_key()),
        ("run", |row| {
            row.run_id.clone().unwrap_or_else(|| "missing".to_string())
        }),
    ]
}

fn write_metric_row<W: Write>(
    writer: &mut W,
    partition: &str,
    dimension: &str,
    key: &str,
    metrics: Metrics,
) -> Result<()> {
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
        partition,
        dimension,
        escape_tsv(key),
        metrics.records,
        metrics.mae,
        metrics.rmse,
        metrics.pearson,
        metrics.mean_error,
        metrics.median_error,
        metrics.residual_std,
    )?;
    Ok(())
}

fn write_calibration_summary(
    path: &Path,
    baseline_dev_mae: f64,
    metrics: &BTreeMap<String, (Metrics, Metrics, bool)>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "calibrator\tsource_aware\ttrain_records\ttrain_mae\ttrain_rmse\ttrain_pearson\ttrain_bias\tdev_records\tdev_mae\tdev_rmse\tdev_pearson\tdev_bias\tdev_mae_ratio_to_baseline"
    )?;
    for (name, (train, dev, source_aware)) in metrics {
        writeln!(
            writer,
            "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            name,
            yes_no(*source_aware),
            train.records,
            train.mae,
            train.rmse,
            train.pearson,
            train.mean_error,
            dev.records,
            dev.mae,
            dev.rmse,
            dev.pearson,
            dev.mean_error,
            dev.mae / baseline_dev_mae,
        )?;
    }
    Ok(())
}

fn write_affine_parameters(
    path: &Path,
    global: Affine,
    charge: &BTreeMap<String, GroupAffine>,
    source: &BTreeMap<String, GroupAffine>,
    source_charge: &BTreeMap<String, GroupAffine>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "calibrator\tgroup\trecords\tblend\tintercept\tslope"
    )?;
    writeln!(
        writer,
        "global_affine\tALL\t0\t1.00000000\t{:.10}\t{:.10}",
        global.intercept, global.slope
    )?;
    for (label, groups) in [
        ("charge_affine", charge),
        ("source_affine", source),
        ("source_charge_affine", source_charge),
    ] {
        for (key, group) in groups {
            writeln!(
                writer,
                "{}\t{}\t{}\t{:.8}\t{:.10}\t{:.10}",
                label,
                escape_tsv(key),
                group.records,
                group.blend,
                group.affine.intercept,
                group.affine.slope,
            )?;
        }
    }
    Ok(())
}

fn write_ridge_parameters(
    path: &Path,
    physics: &RidgeCalibrator,
    source_physics: &RidgeCalibrator,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "calibrator\tfeature\tcoefficient")?;
    for (label, model) in [
        ("physics_ridge", physics),
        ("source_physics_ridge", source_physics),
    ] {
        for (name, coefficient) in model.feature_names.iter().zip(&model.coefficients) {
            writeln!(
                writer,
                "{}\t{}\t{:.12}",
                label,
                escape_tsv(name),
                coefficient
            )?;
        }
    }
    Ok(())
}

fn insert_calibration(
    metrics: &mut BTreeMap<String, (Metrics, Metrics, bool)>,
    name: &str,
    train: Metrics,
    dev: Metrics,
    source_aware: bool,
) {
    print_metric(&format!("calibration_{name}_train"), train);
    print_metric(&format!("calibration_{name}_dev"), dev);
    metrics.insert(name.to_string(), (train, dev, source_aware));
}

fn best_calibration<F>(
    metrics: &BTreeMap<String, (Metrics, Metrics, bool)>,
    mut predicate: F,
) -> Result<(String, Metrics, bool)>
where
    F: FnMut(bool) -> bool,
{
    let mut best: Option<(String, Metrics, bool)> = None;
    for (name, (_, dev, source_aware)) in metrics {
        if !predicate(*source_aware) {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, best_dev, _)| dev.mae < best_dev.mae)
        {
            best = Some((name.clone(), *dev, *source_aware));
        }
    }
    best.ok_or_else(|| anyhow::anyhow!("no calibration candidate matched selection predicate"))
}

fn print_metric(label: &str, metrics: Metrics) {
    println!(
        "{label}\trecords={}\tmae={:.8}\trmse={:.8}\tpearson={:.8}\tbias={:.8}\tmedian_error={:.8}\tresidual_std={:.8}",
        metrics.records,
        metrics.mae,
        metrics.rmse,
        metrics.pearson,
        metrics.mean_error,
        metrics.median_error,
        metrics.residual_std,
    );
}

fn pearson(first: &[f64], second: &[f64]) -> Option<f64> {
    if first.len() != second.len() || first.len() < 2 {
        return None;
    }
    let n = first.len() as f64;
    let mean_first = first.iter().sum::<f64>() / n;
    let mean_second = second.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut first_sq = 0.0;
    let mut second_sq = 0.0;
    for (&a, &b) in first.iter().zip(second) {
        let da = a - mean_first;
        let db = b - mean_second;
        numerator += da * db;
        first_sq += da * da;
        second_sq += db * db;
    }
    let denominator = (first_sq * second_sq).sqrt();
    (denominator > 1.0e-12).then(|| (numerator / denominator).clamp(-1.0, 1.0))
}

fn median_sorted(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn dot(first: &[f64], second: &[f64]) -> f64 {
    first.iter().zip(second).map(|(a, b)| a * b).sum()
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(prediction: f64, target: f64) -> ResidualRow {
        ResidualRow {
            record_index: 0,
            source_id: "source".to_string(),
            source_record_index: 0,
            target,
            prediction,
            charge: Some(2),
            precursor_mz: Some(500.0),
            neutral_mass: 1000.0,
            sequence_len: 10,
            modification_count: 0,
            total_mod_mass: 0.0,
            absolute_mod_mass: 0.0,
            instrument_id: Some(0),
            instrument_name: None,
            run_id: None,
        }
    }

    #[test]
    fn v0362_affine_recovers_exact_linear_calibration() {
        let rows = vec![row(1.0, 3.5), row(2.0, 5.0), row(3.0, 6.5), row(4.0, 8.0)];
        let affine = fit_affine(&rows);
        assert!((affine.intercept - 2.0).abs() < 1.0e-10);
        assert!((affine.slope - 1.5).abs() < 1.0e-10);
    }

    #[test]
    fn v0362_linear_solver_recovers_known_solution() {
        let matrix = vec![vec![3.0, 2.0], vec![1.0, 2.0]];
        let rhs = vec![5.0, 5.0];
        let solution = solve_linear_system(matrix, rhs).unwrap();
        assert!((solution[0] - 0.0).abs() < 1.0e-10);
        assert!((solution[1] - 2.5).abs() < 1.0e-10);
    }
}
