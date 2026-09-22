//! v0.36.4 final CCS audit: raw ion-mobility / derived-CCS provenance.
//!
//! This is deliberately the final diagnostic lane before returning to model
//! improvement. It uses TRAIN + DEV only and never evaluates HOLDOUT. The audit
//! separates disagreement already present in raw Bruker/timsTOF 1/K0 mobility
//! from disagreement introduced by mobility->CCS conversion or by mixing
//! explicit and derived CCS targets.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationCcsDerivationMode, FoundationModificationSite, FoundationPartition,
    FoundationTrainingRecord,
};
use redeem_properties::utils::peptdeep_utils::ion_mobility_to_ccs_bruker;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const V0364_VERSION: u32 = 364;
const V0364_OBJECTIVE: &str = "v0364_raw_mobility_ccs_provenance_final_audit";
const CONVERSION_MISMATCH_TOLERANCE_CCS: f64 = 0.05;
const RAW_MOBILITY_EXPLAINS_RATIO: f64 = 0.80;
const MATERIAL_PAIRWISE_CCS_REFERENCE: f64 = 4.916_453_78;

#[derive(Debug, Clone)]
struct MobilityRow {
    source_id: String,
    identity: String,
    stored_ccs: f64,
    mobility: Option<f64>,
    precursor_mz: Option<f64>,
    charge: Option<i32>,
    recomputed_ccs: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct PairMetrics {
    identities: usize,
    comparisons: usize,
    mean_abs_delta: f64,
    rmse_delta: f64,
    median_abs_delta: f64,
    q90_abs_delta: f64,
    q95_abs_delta: f64,
    max_abs_delta: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct MobilityPairMetrics {
    identities: usize,
    comparisons: usize,
    mean_abs_delta: f64,
    median_abs_delta: f64,
    mean_symmetric_relative_delta: f64,
    median_symmetric_relative_delta: f64,
}

#[derive(Debug, Clone, Default)]
struct SourceAggregate {
    ccs_records: usize,
    mobility_records: usize,
    conversion_comparable_records: usize,
    conversion_abs_errors: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
struct SourcePairAggregate {
    shared_ccs: usize,
    ccs_abs: Vec<f64>,
    ccs_sq: Vec<f64>,
    shared_mobility: usize,
    mobility_abs: Vec<f64>,
    mobility_rel: Vec<f64>,
    recomputed_ccs_abs: Vec<f64>,
    recomputed_ccs_sq: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticSummary {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    train_ccs_records: usize,
    dev_ccs_records: usize,
    train_mobility_records: usize,
    dev_mobility_records: usize,
    dev_cross_source_ccs_pairwise_mae: f64,
    dev_cross_source_mobility_pairwise_mae: f64,
    dev_cross_source_mobility_symmetric_relative_mae: f64,
    dev_cross_source_recomputed_ccs_pairwise_mae: f64,
    dev_recomputed_to_stored_pairwise_ratio: f64,
    dev_consensus_label_mae: f64,
    derived_source_conversion_mae: f64,
    conversion_provenance_mismatch: bool,
    raw_mobility_disagreement_dominant: bool,
    target_rebuild_and_retrain_recommended: bool,
    conversion_fix_and_retrain_recommended: bool,
    no_more_audits: bool,
    train_holdout_consumed: bool,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        anyhow::bail!(
            "usage: foundation_audit_ccs_mobility_provenance_v0364 RUN_V0260.yaml OUTPUT_DIR"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    if output_root.exists() {
        anyhow::bail!("v0.36.4 output directory already exists: {output_root:?}");
    }

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());

    let train_rows = collect_rows(
        &corpus.records,
        &corpus.provenance,
        &benchmark,
        FoundationPartition::Train,
    )?;
    let dev_rows = collect_rows(
        &corpus.records,
        &corpus.provenance,
        &benchmark,
        FoundationPartition::Validation,
    )?;
    if train_rows.is_empty() || dev_rows.is_empty() {
        anyhow::bail!("v0.36.4 requires finite CCS labels in TRAIN and DEV");
    }

    fs::create_dir_all(&output_root)?;

    println!("v0364_version\tv0.36.4-raw-mobility-ccs-provenance-final-audit");
    println!("objective\t{V0364_OBJECTIVE}");
    println!("corpus_fingerprint\t{corpus_fingerprint}");
    println!("benchmark_manifest_fingerprint\t{benchmark_fingerprint}");
    println!("model_evaluated\tNO");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0364_selection\tNO");
    println!("historical_test_consumed\tNO");
    println!("v0364_no_more_audits\tYES");

    let train_mobility = train_rows
        .iter()
        .filter(|row| row.mobility.is_some())
        .count();
    let dev_mobility = dev_rows.iter().filter(|row| row.mobility.is_some()).count();
    println!("v0364_train_ccs_records\t{}", train_rows.len());
    println!("v0364_dev_ccs_records\t{}", dev_rows.len());
    println!("v0364_train_mobility_records\t{train_mobility}");
    println!("v0364_dev_mobility_records\t{dev_mobility}");

    let source_modes = write_source_provenance(
        &output_root.join("ccs_source_provenance.tsv"),
        &run,
        &corpus.sources,
        &train_rows,
        &dev_rows,
    )?;

    let train_groups = group_rows(&train_rows);
    let dev_groups = group_rows(&dev_rows);

    let train_ccs = pair_metrics_for(&train_rows, &train_groups, |row| Some(row.stored_ccs));
    let dev_ccs = pair_metrics_for(&dev_rows, &dev_groups, |row| Some(row.stored_ccs));
    let train_recomputed = pair_metrics_for(&train_rows, &train_groups, |row| row.recomputed_ccs);
    let dev_recomputed = pair_metrics_for(&dev_rows, &dev_groups, |row| row.recomputed_ccs);
    let train_stored_mobility_subset = pair_metrics_for(&train_rows, &train_groups, |row| {
        row.recomputed_ccs.map(|_| row.stored_ccs)
    });
    let dev_stored_mobility_subset = pair_metrics_for(&dev_rows, &dev_groups, |row| {
        row.recomputed_ccs.map(|_| row.stored_ccs)
    });
    let train_mobility_metrics = mobility_pair_metrics(&train_rows, &train_groups);
    let dev_mobility_metrics = mobility_pair_metrics(&dev_rows, &dev_groups);

    let train_consensus_mae = consensus_label_mae(&train_rows, &train_groups);
    let dev_consensus_mae = consensus_label_mae(&dev_rows, &dev_groups);

    write_source_pair_provenance(
        &output_root.join("ccs_mobility_source_pairs_train.tsv"),
        "TRAIN",
        &train_rows,
        &train_groups,
        &source_modes,
    )?;
    write_source_pair_provenance(
        &output_root.join("ccs_mobility_source_pairs_dev.tsv"),
        "DEV",
        &dev_rows,
        &dev_groups,
        &source_modes,
    )?;

    write_identity_detail(
        &output_root.join("ccs_mobility_identity_detail_dev.tsv"),
        &dev_rows,
        &dev_groups,
    )?;

    let source_conversion = source_conversion_metrics(&train_rows, &dev_rows);
    write_conversion_consistency(
        &output_root.join("ccs_conversion_consistency.tsv"),
        &source_conversion,
        &source_modes,
    )?;

    let derived_source_conversion_mae =
        weighted_derived_conversion_mae(&source_conversion, &source_modes);
    let conversion_mismatch = derived_source_conversion_mae > CONVERSION_MISMATCH_TOLERANCE_CCS;

    let recomputed_to_stored_ratio =
        if dev_stored_mobility_subset.mean_abs_delta > 1.0e-12 && dev_recomputed.comparisons > 0 {
            dev_recomputed.mean_abs_delta / dev_stored_mobility_subset.mean_abs_delta
        } else {
            0.0
        };
    let raw_mobility_dominant = !conversion_mismatch
        && dev_recomputed.comparisons > 0
        && recomputed_to_stored_ratio >= RAW_MOBILITY_EXPLAINS_RATIO;

    let target_rebuild = !conversion_mismatch;
    let conversion_fix = conversion_mismatch;

    write_global_summary(
        &output_root.join("ccs_mobility_provenance_metrics.tsv"),
        train_ccs,
        dev_ccs,
        train_recomputed,
        dev_recomputed,
        train_stored_mobility_subset,
        dev_stored_mobility_subset,
        train_mobility_metrics,
        dev_mobility_metrics,
        train_consensus_mae,
        dev_consensus_mae,
        derived_source_conversion_mae,
        recomputed_to_stored_ratio,
    )?;

    println!(
        "v0364_train_cross_source_ccs_pairwise_mae\t{:.8}",
        train_ccs.mean_abs_delta
    );
    println!(
        "v0364_dev_cross_source_ccs_pairwise_mae\t{:.8}",
        dev_ccs.mean_abs_delta
    );
    println!(
        "v0364_dev_cross_source_mobility_pairwise_mae\t{:.8}",
        dev_mobility_metrics.mean_abs_delta
    );
    println!(
        "v0364_dev_cross_source_mobility_symmetric_relative_mae\t{:.8}",
        dev_mobility_metrics.mean_symmetric_relative_delta
    );
    println!(
        "v0364_dev_cross_source_recomputed_ccs_pairwise_mae\t{:.8}",
        dev_recomputed.mean_abs_delta
    );
    println!("v0364_dev_recomputed_to_stored_pairwise_ratio\t{recomputed_to_stored_ratio:.8}");
    println!("v0364_train_consensus_label_mae\t{train_consensus_mae:.8}");
    println!("v0364_dev_consensus_label_mae\t{dev_consensus_mae:.8}");
    println!("v0364_derived_source_conversion_mae\t{derived_source_conversion_mae:.8}");
    println!(
        "v0364_conversion_provenance_mismatch\t{}",
        yes_no(conversion_mismatch)
    );
    println!(
        "v0364_raw_mobility_disagreement_dominant\t{}",
        yes_no(raw_mobility_dominant)
    );
    println!(
        "v0364_target_rebuild_and_retrain_recommended\t{}",
        yes_no(target_rebuild)
    );
    println!(
        "v0364_conversion_fix_and_retrain_recommended\t{}",
        yes_no(conversion_fix)
    );
    println!("v0364_next_iteration\tv0.37_CCS_IMPROVEMENT_TRAINING_NOT_AUDIT");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0364_selection\tNO");
    println!("historical_test_consumed\tNO");

    let summary = DiagnosticSummary {
        version: V0364_VERSION,
        objective: V0364_OBJECTIVE.to_string(),
        corpus_fingerprint,
        benchmark_manifest_fingerprint: benchmark_fingerprint,
        train_ccs_records: train_rows.len(),
        dev_ccs_records: dev_rows.len(),
        train_mobility_records: train_mobility,
        dev_mobility_records: dev_mobility,
        dev_cross_source_ccs_pairwise_mae: dev_ccs.mean_abs_delta,
        dev_cross_source_mobility_pairwise_mae: dev_mobility_metrics.mean_abs_delta,
        dev_cross_source_mobility_symmetric_relative_mae: dev_mobility_metrics
            .mean_symmetric_relative_delta,
        dev_cross_source_recomputed_ccs_pairwise_mae: dev_recomputed.mean_abs_delta,
        dev_recomputed_to_stored_pairwise_ratio: recomputed_to_stored_ratio,
        dev_consensus_label_mae: dev_consensus_mae,
        derived_source_conversion_mae,
        conversion_provenance_mismatch: conversion_mismatch,
        raw_mobility_disagreement_dominant: raw_mobility_dominant,
        target_rebuild_and_retrain_recommended: target_rebuild,
        conversion_fix_and_retrain_recommended: conversion_fix,
        no_more_audits: true,
        train_holdout_consumed: false,
        historical_validation_consumed: false,
        historical_test_consumed: false,
    };
    fs::write(
        output_root.join("ccs_mobility_provenance_summary.yaml"),
        serde_yaml::to_string(&summary)?,
    )?;

    Ok(())
}

fn collect_rows(
    records: &[FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
) -> Result<Vec<MobilityRow>> {
    let mut rows = Vec::new();
    for entry in benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
    {
        let index = entry.record_index;
        let record = records
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("benchmark index {index} outside corpus"))?;
        let Some(stored_ccs) = record.ccs.filter(|value| value.is_finite() && *value > 0.0) else {
            continue;
        };
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("missing provenance for record {index}"))?;
        let mobility = record
            .context
            .ion_mobility
            .map(f64::from)
            .filter(|v| v.is_finite() && *v > 0.0);
        let precursor_mz = record
            .context
            .precursor_mz
            .map(f64::from)
            .filter(|v| v.is_finite() && *v > 0.0);
        let charge = record.context.charge.filter(|value| *value > 0);
        let recomputed_ccs = match (mobility, precursor_mz, charge) {
            (Some(mobility), Some(mz), Some(charge)) => {
                let value = f64::from(ion_mobility_to_ccs_bruker(mobility, charge, mz));
                (value.is_finite() && value > 0.0).then_some(value)
            }
            _ => None,
        };
        rows.push(MobilityRow {
            source_id: source.source_id.clone(),
            identity: peptidoform_charge_key(record),
            stored_ccs: f64::from(stored_ccs),
            mobility,
            precursor_mz,
            charge,
            recomputed_ccs,
        });
    }
    Ok(rows)
}

fn peptidoform_charge_key(record: &FoundationTrainingRecord) -> String {
    let mut modifications = record
        .peptidoform
        .modifications
        .iter()
        .map(|modification| {
            let site = match modification.site {
                FoundationModificationSite::Residue(index) => format!("R{index}"),
                FoundationModificationSite::NTerm => "N".to_string(),
                FoundationModificationSite::CTerm => "C".to_string(),
            };
            format!(
                "{site}:{}:{:+.4}",
                modification.identity_label(),
                modification.mass_delta
            )
        })
        .collect::<Vec<_>>();
    modifications.sort();
    let charge = record
        .context
        .charge
        .map(|value| value.to_string())
        .unwrap_or_else(|| "missing".to_string());
    format!(
        "{}|z={}|{}",
        record.peptidoform.sequence,
        charge,
        modifications.join(";")
    )
}

fn group_rows(rows: &[MobilityRow]) -> BTreeMap<String, Vec<usize>> {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (index, row) in rows.iter().enumerate() {
        groups.entry(row.identity.clone()).or_default().push(index);
    }
    groups
}

fn source_means<F>(rows: &[MobilityRow], indices: &[usize], value_fn: F) -> Vec<(String, f64)>
where
    F: Fn(&MobilityRow) -> Option<f64>,
{
    let mut by_source = BTreeMap::<String, Vec<f64>>::new();
    for &index in indices {
        if let Some(value) = value_fn(&rows[index]).filter(|value| value.is_finite()) {
            by_source
                .entry(rows[index].source_id.clone())
                .or_default()
                .push(value);
        }
    }
    by_source
        .into_iter()
        .filter_map(|(source, values)| {
            (!values.is_empty()).then(|| {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                (source, mean)
            })
        })
        .collect()
}

fn pair_metrics_for<F>(
    rows: &[MobilityRow],
    groups: &BTreeMap<String, Vec<usize>>,
    value_fn: F,
) -> PairMetrics
where
    F: Fn(&MobilityRow) -> Option<f64> + Copy,
{
    let mut abs_values = Vec::<f64>::new();
    let mut squared = Vec::<f64>::new();
    let mut identities = 0usize;
    for indices in groups.values() {
        let values = source_means(rows, indices, value_fn);
        let mut used = false;
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                let delta = values[i].1 - values[j].1;
                abs_values.push(delta.abs());
                squared.push(delta * delta);
                used = true;
            }
        }
        if used {
            identities += 1;
        }
    }
    pair_metrics_from_values(identities, abs_values, squared)
}

fn pair_metrics_from_values(
    identities: usize,
    mut abs_values: Vec<f64>,
    squared: Vec<f64>,
) -> PairMetrics {
    if abs_values.is_empty() {
        return PairMetrics::default();
    }
    abs_values.sort_by(|a, b| a.total_cmp(b));
    PairMetrics {
        identities,
        comparisons: abs_values.len(),
        mean_abs_delta: abs_values.iter().sum::<f64>() / abs_values.len() as f64,
        rmse_delta: (squared.iter().sum::<f64>() / squared.len() as f64).sqrt(),
        median_abs_delta: quantile_sorted(&abs_values, 0.50),
        q90_abs_delta: quantile_sorted(&abs_values, 0.90),
        q95_abs_delta: quantile_sorted(&abs_values, 0.95),
        max_abs_delta: *abs_values.last().unwrap_or(&0.0),
    }
}

fn mobility_pair_metrics(
    rows: &[MobilityRow],
    groups: &BTreeMap<String, Vec<usize>>,
) -> MobilityPairMetrics {
    let mut abs_values = Vec::<f64>::new();
    let mut relative_values = Vec::<f64>::new();
    let mut identities = 0usize;
    for indices in groups.values() {
        let values = source_means(rows, indices, |row| row.mobility);
        let mut used = false;
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                let left = values[i].1;
                let right = values[j].1;
                let abs = (left - right).abs();
                let denom = left.abs() + right.abs();
                let rel = if denom > 1.0e-12 {
                    2.0 * abs / denom
                } else {
                    0.0
                };
                abs_values.push(abs);
                relative_values.push(rel);
                used = true;
            }
        }
        if used {
            identities += 1;
        }
    }
    if abs_values.is_empty() {
        return MobilityPairMetrics::default();
    }
    abs_values.sort_by(|a, b| a.total_cmp(b));
    relative_values.sort_by(|a, b| a.total_cmp(b));
    MobilityPairMetrics {
        identities,
        comparisons: abs_values.len(),
        mean_abs_delta: abs_values.iter().sum::<f64>() / abs_values.len() as f64,
        median_abs_delta: quantile_sorted(&abs_values, 0.50),
        mean_symmetric_relative_delta: relative_values.iter().sum::<f64>()
            / relative_values.len() as f64,
        median_symmetric_relative_delta: quantile_sorted(&relative_values, 0.50),
    }
}

fn consensus_label_mae(rows: &[MobilityRow], groups: &BTreeMap<String, Vec<usize>>) -> f64 {
    let mut deviations = Vec::<f64>::new();
    for indices in groups.values() {
        let values = source_means(rows, indices, |row| Some(row.stored_ccs));
        if values.len() < 2 {
            continue;
        }
        let mut targets = values.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        targets.sort_by(|a, b| a.total_cmp(b));
        let median = quantile_sorted(&targets, 0.50);
        for target in targets {
            deviations.push((target - median).abs());
        }
    }
    mean_or_zero(&deviations)
}

fn write_source_provenance(
    path: &Path,
    run: &redeem_properties::foundation::FoundationTrainingRunConfig,
    summaries: &[redeem_properties::foundation::FoundationCorpusSourceSummary],
    train_rows: &[MobilityRow],
    dev_rows: &[MobilityRow],
) -> Result<BTreeMap<String, String>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "source\tprofile\tconfigured_ccs_derivation\tccs_origin_class\tall_ccs_records\texplicit_ccs_records\tderived_ccs_records\tion_mobility_records\ttrain_ccs_records\ttrain_mobility_records\tdev_ccs_records\tdev_mobility_records"
    )?;
    let mut modes = BTreeMap::<String, String>::new();
    for (index, summary) in summaries.iter().enumerate() {
        let spec = run.corpus.sources.get(index);
        let configured = spec
            .and_then(|source| source.ccs_derivation)
            .unwrap_or(run.corpus.loader.ccs_derivation);
        let configured_label = derivation_label(configured);
        let explicit = summary.stats.explicit_ccs_records;
        let derived = summary.stats.derived_ccs_records;
        let origin = match (explicit > 0, derived > 0) {
            (true, true) => "mixed",
            (true, false) => "explicit",
            (false, true) => "derived",
            (false, false) => "none",
        };
        modes.insert(summary.id.clone(), origin.to_string());
        let train_source = train_rows
            .iter()
            .filter(|row| row.source_id == summary.id)
            .collect::<Vec<_>>();
        let dev_source = dev_rows
            .iter()
            .filter(|row| row.source_id == summary.id)
            .collect::<Vec<_>>();
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            escape_tsv(&summary.id),
            escape_tsv(&summary.profile),
            configured_label,
            origin,
            summary.stats.ccs_records,
            explicit,
            derived,
            summary.stats.ion_mobility_records,
            train_source.len(),
            train_source
                .iter()
                .filter(|row| row.mobility.is_some())
                .count(),
            dev_source.len(),
            dev_source
                .iter()
                .filter(|row| row.mobility.is_some())
                .count(),
        )?;
    }
    Ok(modes)
}

fn derivation_label(mode: FoundationCcsDerivationMode) -> &'static str {
    match mode {
        FoundationCcsDerivationMode::Disabled => "disabled",
        FoundationCcsDerivationMode::AutoBruker => "auto-bruker",
    }
}

fn source_conversion_metrics(
    train_rows: &[MobilityRow],
    dev_rows: &[MobilityRow],
) -> BTreeMap<String, SourceAggregate> {
    let mut map = BTreeMap::<String, SourceAggregate>::new();
    for row in train_rows.iter().chain(dev_rows.iter()) {
        let entry = map.entry(row.source_id.clone()).or_default();
        entry.ccs_records += 1;
        if row.mobility.is_some() {
            entry.mobility_records += 1;
        }
        if let Some(recomputed) = row.recomputed_ccs {
            entry.conversion_comparable_records += 1;
            entry
                .conversion_abs_errors
                .push((row.stored_ccs - recomputed).abs());
        }
    }
    map
}

fn write_conversion_consistency(
    path: &Path,
    metrics: &BTreeMap<String, SourceAggregate>,
    modes: &BTreeMap<String, String>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "source\tccs_origin_class\tccs_records\tmobility_records\tconversion_comparable_records\tstored_vs_bruker_recomputed_mae\tstored_vs_bruker_recomputed_median_abs\tstored_vs_bruker_recomputed_q95_abs\tstored_vs_bruker_recomputed_max_abs"
    )?;
    for (source, metric) in metrics {
        let mut values = metric.conversion_abs_errors.clone();
        values.sort_by(|a, b| a.total_cmp(b));
        let mean = mean_or_zero(&values);
        let median = quantile_or_zero(&values, 0.50);
        let q95 = quantile_or_zero(&values, 0.95);
        let max = values.last().copied().unwrap_or(0.0);
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            escape_tsv(source),
            modes.get(source).map(String::as_str).unwrap_or("unknown"),
            metric.ccs_records,
            metric.mobility_records,
            metric.conversion_comparable_records,
            mean,
            median,
            q95,
            max,
        )?;
    }
    Ok(())
}

fn weighted_derived_conversion_mae(
    metrics: &BTreeMap<String, SourceAggregate>,
    modes: &BTreeMap<String, String>,
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (source, metric) in metrics {
        if modes.get(source).map(String::as_str) != Some("derived") {
            continue;
        }
        sum += metric.conversion_abs_errors.iter().sum::<f64>();
        n += metric.conversion_abs_errors.len();
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn write_source_pair_provenance(
    path: &Path,
    partition: &str,
    rows: &[MobilityRow],
    groups: &BTreeMap<String, Vec<usize>>,
    source_modes: &BTreeMap<String, String>,
) -> Result<()> {
    let mut pairs = BTreeMap::<(String, String), SourcePairAggregate>::new();
    for indices in groups.values() {
        let by_source = source_row_means(rows, indices);
        let sources = by_source.keys().cloned().collect::<Vec<_>>();
        for i in 0..sources.len() {
            for j in (i + 1)..sources.len() {
                let left = &by_source[&sources[i]];
                let right = &by_source[&sources[j]];
                let pair = pairs
                    .entry((sources[i].clone(), sources[j].clone()))
                    .or_default();
                let ccs_delta = left.stored_ccs - right.stored_ccs;
                pair.shared_ccs += 1;
                pair.ccs_abs.push(ccs_delta.abs());
                pair.ccs_sq.push(ccs_delta * ccs_delta);
                if let (Some(lm), Some(rm), Some(lc), Some(rc)) = (
                    left.mobility,
                    right.mobility,
                    left.recomputed_ccs,
                    right.recomputed_ccs,
                ) {
                    pair.shared_mobility += 1;
                    let mobility_abs = (lm - rm).abs();
                    pair.mobility_abs.push(mobility_abs);
                    let denom = lm.abs() + rm.abs();
                    pair.mobility_rel.push(if denom > 1.0e-12 {
                        2.0 * mobility_abs / denom
                    } else {
                        0.0
                    });
                    let delta = lc - rc;
                    pair.recomputed_ccs_abs.push(delta.abs());
                    pair.recomputed_ccs_sq.push(delta * delta);
                }
            }
        }
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "partition\tsource_a\tsource_b\torigin_a\torigin_b\tshared_ccs_identities\tstored_ccs_pairwise_mae\tstored_ccs_pairwise_rmse\tshared_mobility_identities\tmobility_pairwise_mae\tmobility_symmetric_relative_mae\trecomputed_ccs_pairwise_mae\trecomputed_ccs_pairwise_rmse"
    )?;
    for ((source_a, source_b), pair) in pairs {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            partition,
            escape_tsv(&source_a),
            escape_tsv(&source_b),
            source_modes
                .get(&source_a)
                .map(String::as_str)
                .unwrap_or("unknown"),
            source_modes
                .get(&source_b)
                .map(String::as_str)
                .unwrap_or("unknown"),
            pair.shared_ccs,
            mean_or_zero(&pair.ccs_abs),
            rms_or_zero(&pair.ccs_sq),
            pair.shared_mobility,
            mean_or_zero(&pair.mobility_abs),
            mean_or_zero(&pair.mobility_rel),
            mean_or_zero(&pair.recomputed_ccs_abs),
            rms_or_zero(&pair.recomputed_ccs_sq),
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct SourceRowMean {
    stored_ccs: f64,
    mobility: Option<f64>,
    recomputed_ccs: Option<f64>,
}

fn source_row_means(rows: &[MobilityRow], indices: &[usize]) -> BTreeMap<String, SourceRowMean> {
    let mut grouped = BTreeMap::<String, Vec<&MobilityRow>>::new();
    for &index in indices {
        grouped
            .entry(rows[index].source_id.clone())
            .or_default()
            .push(&rows[index]);
    }
    grouped
        .into_iter()
        .map(|(source, values)| {
            let stored_ccs =
                values.iter().map(|row| row.stored_ccs).sum::<f64>() / values.len() as f64;
            let mobility_values = values
                .iter()
                .filter_map(|row| row.mobility)
                .collect::<Vec<_>>();
            let recomputed_values = values
                .iter()
                .filter_map(|row| row.recomputed_ccs)
                .collect::<Vec<_>>();
            let mobility = (!mobility_values.is_empty()).then(|| mean_or_zero(&mobility_values));
            let recomputed_ccs =
                (!recomputed_values.is_empty()).then(|| mean_or_zero(&recomputed_values));
            (
                source,
                SourceRowMean {
                    stored_ccs,
                    mobility,
                    recomputed_ccs,
                },
            )
        })
        .collect()
}

fn write_identity_detail(
    path: &Path,
    rows: &[MobilityRow],
    groups: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "identity\tsource\tstored_ccs\tion_mobility\tprecursor_mz\tcharge\tbruker_recomputed_ccs\tstored_minus_recomputed_ccs"
    )?;
    for (identity, indices) in groups {
        let distinct_sources = indices
            .iter()
            .map(|&index| rows[index].source_id.as_str())
            .collect::<BTreeSet<_>>();
        if distinct_sources.len() < 2 {
            continue;
        }
        for &index in indices {
            let row = &rows[index];
            writeln!(
                writer,
                "{}\t{}\t{:.8}\t{}\t{}\t{}\t{}\t{}",
                escape_tsv(identity),
                escape_tsv(&row.source_id),
                row.stored_ccs,
                fmt_opt(row.mobility),
                fmt_opt(row.precursor_mz),
                row.charge.map(|v| v.to_string()).unwrap_or_default(),
                fmt_opt(row.recomputed_ccs),
                row.recomputed_ccs
                    .map(|value| format!("{:.8}", row.stored_ccs - value))
                    .unwrap_or_default(),
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_global_summary(
    path: &Path,
    train_ccs: PairMetrics,
    dev_ccs: PairMetrics,
    train_recomputed: PairMetrics,
    dev_recomputed: PairMetrics,
    train_stored_mobility_subset: PairMetrics,
    dev_stored_mobility_subset: PairMetrics,
    train_mobility: MobilityPairMetrics,
    dev_mobility: MobilityPairMetrics,
    train_consensus_mae: f64,
    dev_consensus_mae: f64,
    derived_conversion_mae: f64,
    recomputed_to_stored_ratio: f64,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "metric\tTRAIN\tDEV")?;
    writeln!(
        writer,
        "stored_ccs_pairwise_mae\t{:.8}\t{:.8}",
        train_ccs.mean_abs_delta, dev_ccs.mean_abs_delta
    )?;
    writeln!(
        writer,
        "stored_ccs_pairwise_rmse\t{:.8}\t{:.8}",
        train_ccs.rmse_delta, dev_ccs.rmse_delta
    )?;
    writeln!(
        writer,
        "mobility_pairwise_mae\t{:.8}\t{:.8}",
        train_mobility.mean_abs_delta, dev_mobility.mean_abs_delta
    )?;
    writeln!(
        writer,
        "mobility_symmetric_relative_mae\t{:.8}\t{:.8}",
        train_mobility.mean_symmetric_relative_delta, dev_mobility.mean_symmetric_relative_delta
    )?;
    writeln!(
        writer,
        "stored_ccs_pairwise_mae_mobility_subset\t{:.8}\t{:.8}",
        train_stored_mobility_subset.mean_abs_delta, dev_stored_mobility_subset.mean_abs_delta
    )?;
    writeln!(
        writer,
        "recomputed_ccs_pairwise_mae\t{:.8}\t{:.8}",
        train_recomputed.mean_abs_delta, dev_recomputed.mean_abs_delta
    )?;
    writeln!(
        writer,
        "consensus_label_mae\t{:.8}\t{:.8}",
        train_consensus_mae, dev_consensus_mae
    )?;
    writeln!(
        writer,
        "derived_source_stored_vs_recomputed_mae\t{:.8}\t{:.8}",
        derived_conversion_mae, derived_conversion_mae
    )?;
    writeln!(
        writer,
        "dev_recomputed_to_stored_pairwise_ratio\t\t{recomputed_to_stored_ratio:.8}"
    )?;
    writeln!(
        writer,
        "v0363_dev_stored_ccs_pairwise_mae_reference\t\t{MATERIAL_PAIRWISE_CCS_REFERENCE:.8}"
    )?;
    Ok(())
}

fn quantile_sorted(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let position = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let fraction = position - lo as f64;
        values[lo] * (1.0 - fraction) + values[hi] * fraction
    }
}

fn quantile_or_zero(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    quantile_sorted(values, q)
}

fn mean_or_zero(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn rms_or_zero(squared_values: &[f64]) -> f64 {
    if squared_values.is_empty() {
        0.0
    } else {
        (squared_values.iter().sum::<f64>() / squared_values.len() as f64).sqrt()
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.8}")).unwrap_or_default()
}

fn escape_tsv(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0364_symmetric_relative_delta_is_scale_free() {
        let rows = vec![
            MobilityRow {
                source_id: "a".into(),
                identity: "x".into(),
                stored_ccs: 100.0,
                mobility: Some(1.0),
                precursor_mz: Some(500.0),
                charge: Some(2),
                recomputed_ccs: Some(100.0),
            },
            MobilityRow {
                source_id: "b".into(),
                identity: "x".into(),
                stored_ccs: 110.0,
                mobility: Some(1.1),
                precursor_mz: Some(500.0),
                charge: Some(2),
                recomputed_ccs: Some(110.0),
            },
        ];
        let groups = group_rows(&rows);
        let metrics = mobility_pair_metrics(&rows, &groups);
        assert_eq!(metrics.comparisons, 1);
        assert!((metrics.mean_abs_delta - 0.1).abs() < 1.0e-12);
        assert!((metrics.mean_symmetric_relative_delta - (0.2 / 2.1)).abs() < 1.0e-12);
    }

    #[test]
    fn v0364_consensus_mae_uses_identity_median() {
        let rows = vec![
            MobilityRow {
                source_id: "a".into(),
                identity: "x".into(),
                stored_ccs: 100.0,
                mobility: None,
                precursor_mz: None,
                charge: Some(2),
                recomputed_ccs: None,
            },
            MobilityRow {
                source_id: "b".into(),
                identity: "x".into(),
                stored_ccs: 104.0,
                mobility: None,
                precursor_mz: None,
                charge: Some(2),
                recomputed_ccs: None,
            },
            MobilityRow {
                source_id: "c".into(),
                identity: "x".into(),
                stored_ccs: 110.0,
                mobility: None,
                precursor_mz: None,
                charge: Some(2),
                recomputed_ccs: None,
            },
        ];
        let groups = group_rows(&rows);
        let mae = consensus_label_mae(&rows, &groups);
        assert!((mae - (10.0 / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn v0364_pair_metrics_match_expected_mae() {
        let rows = vec![
            MobilityRow {
                source_id: "a".into(),
                identity: "x".into(),
                stored_ccs: 10.0,
                mobility: None,
                precursor_mz: None,
                charge: Some(2),
                recomputed_ccs: None,
            },
            MobilityRow {
                source_id: "b".into(),
                identity: "x".into(),
                stored_ccs: 14.0,
                mobility: None,
                precursor_mz: None,
                charge: Some(2),
                recomputed_ccs: None,
            },
        ];
        let groups = group_rows(&rows);
        let metrics = pair_metrics_for(&rows, &groups, |row| Some(row.stored_ccs));
        assert_eq!(metrics.identities, 1);
        assert_eq!(metrics.comparisons, 1);
        assert!((metrics.mean_abs_delta - 4.0).abs() < 1.0e-12);
    }
}
