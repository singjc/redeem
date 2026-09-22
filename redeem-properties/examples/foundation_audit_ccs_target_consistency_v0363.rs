//! v0.36.3 CCS target-consistency / corpus-harmonization audit.
//!
//! This executable is intentionally model-free and non-training. It audits CCS
//! labels on TRAIN and DEV only, measures repeated-measure disagreement within
//! sources/runs and across sources for identical peptidoform+charge identities,
//! fits source harmonization maps on TRAIN only, and evaluates their ability to
//! reduce cross-source disagreement on DEV. TRAIN-HOLDOUT and historical
//! VALIDATION/TEST are never read.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationModificationSite, FoundationPartition, FoundationTrainingRecord,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const V0363_VERSION: u32 = 363;
const V0363_OBJECTIVE: &str = "v0363_ccs_target_consistency_corpus_harmonization_audit";
const SOURCE_SHRINKAGE: f64 = 256.0;
const MIN_SOURCE_SHARED_IDENTITIES: usize = 20;
const V0362_FULL_DEV_MODEL_MAE_REFERENCE: f64 = 9.631_886_42;
const MATERIAL_DISAGREEMENT_FRACTION_OF_MODEL_MAE: f64 = 0.25;
const MATERIAL_HARMONIZATION_RATIO: f64 = 0.80;

#[derive(Debug, Clone)]
struct LabelRow {
    record_index: usize,
    source_id: String,
    source_record_index: usize,
    run_id: String,
    identity: String,
    charge: String,
    target: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct DistributionMetrics {
    records: usize,
    unique_identities: usize,
    mean: f64,
    median: f64,
    std: f64,
    min: f64,
    q05: f64,
    q95: f64,
    max: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct DisagreementMetrics {
    identities: usize,
    comparisons: usize,
    mean_abs_delta: f64,
    rmse_delta: f64,
    median_abs_delta: f64,
    q90_abs_delta: f64,
    q95_abs_delta: f64,
    max_abs_delta: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Affine {
    intercept: f64,
    slope: f64,
}

#[derive(Debug, Clone, Default)]
struct SourceFit {
    shared_identities: usize,
    blend: f64,
    offset: f64,
    affine: Affine,
}

#[derive(Debug, Clone, Default)]
struct PairAccumulator {
    shared_identities: usize,
    sum_delta: f64,
    sum_abs: f64,
    sum_sq: f64,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    sum_xy: f64,
    max_abs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticSummary {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    train_ccs_records: usize,
    dev_ccs_records: usize,
    train_unique_identities: usize,
    dev_unique_identities: usize,
    train_cross_source_identities: usize,
    dev_cross_source_identities: usize,
    train_dev_shared_identities: usize,
    train_cross_source_pairwise_mae: f64,
    dev_cross_source_pairwise_mae: f64,
    dev_cross_source_pairwise_mae_after_offset: f64,
    dev_cross_source_pairwise_mae_after_affine: f64,
    best_harmonization: String,
    best_harmonized_dev_pairwise_mae: f64,
    best_harmonized_dev_ratio: f64,
    dev_within_source_repeat_mae_to_group_mean: f64,
    dev_cross_run_pairwise_mae: f64,
    material_target_disagreement: bool,
    train_only_harmonization_effective: bool,
    ccs_corpus_rebuild_recommended: bool,
    ccs_harmonize_then_retrain_recommended: bool,
    ccs_physics_modeling_recommended: bool,
    train_holdout_consumed: bool,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        anyhow::bail!(
            "usage: foundation_audit_ccs_target_consistency_v0363 RUN_V0260.yaml OUTPUT_DIR"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    if output_root.exists() {
        anyhow::bail!("v0.36.3 output directory already exists: {output_root:?}");
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
        anyhow::bail!("v0.36.3 requires finite CCS labels in TRAIN and DEV");
    }

    fs::create_dir_all(&output_root)?;

    println!("v0363_version\tv0.36.3-ccs-target-consistency-corpus-harmonization-audit");
    println!("objective\t{V0363_OBJECTIVE}");
    println!("corpus_fingerprint\t{corpus_fingerprint}");
    println!("benchmark_manifest_fingerprint\t{benchmark_fingerprint}");
    println!("train_ccs_records\t{}", train_rows.len());
    println!("dev_ccs_records\t{}", dev_rows.len());
    println!("model_evaluated\tNO");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0363_selection\tNO");
    println!("historical_test_consumed\tNO");

    let train_groups = identity_groups(&train_rows);
    let dev_groups = identity_groups(&dev_rows);
    let train_unique = train_groups.len();
    let dev_unique = dev_groups.len();
    let train_cross_source = count_cross_source_identities(&train_groups, &train_rows);
    let dev_cross_source = count_cross_source_identities(&dev_groups, &dev_rows);

    println!("v0363_train_unique_identities\t{train_unique}");
    println!("v0363_dev_unique_identities\t{dev_unique}");
    println!("v0363_train_cross_source_identities\t{train_cross_source}");
    println!("v0363_dev_cross_source_identities\t{dev_cross_source}");

    write_target_distributions(
        &output_root.join("ccs_target_distributions.tsv"),
        &train_rows,
        &dev_rows,
    )?;

    let train_consistency = consistency_audit(
        "TRAIN",
        &train_rows,
        &train_groups,
        &output_root.join("ccs_identity_consistency_train.tsv"),
        &output_root.join("ccs_source_pair_consistency_train.tsv"),
        &output_root.join("ccs_source_outlier_scores_train.tsv"),
    )?;
    let dev_consistency = consistency_audit(
        "DEV",
        &dev_rows,
        &dev_groups,
        &output_root.join("ccs_identity_consistency_dev.tsv"),
        &output_root.join("ccs_source_pair_consistency_dev.tsv"),
        &output_root.join("ccs_source_outlier_scores_dev.tsv"),
    )?;

    let train_dev_shared = write_partition_overlap(
        &output_root.join("ccs_train_dev_identity_overlap.tsv"),
        &train_rows,
        &dev_rows,
        &train_groups,
        &dev_groups,
    )?;

    let source_fits = fit_source_harmonization(&train_rows, &train_groups)?;
    write_source_harmonization_parameters(
        &output_root.join("ccs_source_harmonization_parameters.tsv"),
        &source_fits,
    )?;

    let raw_dev =
        cross_source_disagreement_with_transform(&dev_rows, &dev_groups, |row| row.target);
    let offset_dev = cross_source_disagreement_with_transform(&dev_rows, &dev_groups, |row| {
        apply_source_offset(row, &source_fits)
    });
    let affine_dev = cross_source_disagreement_with_transform(&dev_rows, &dev_groups, |row| {
        apply_source_affine(row, &source_fits)
    });

    let (best_harmonization, best_harmonized) =
        if affine_dev.mean_abs_delta <= offset_dev.mean_abs_delta {
            ("source_affine", affine_dev)
        } else {
            ("source_offset", offset_dev)
        };
    let best_ratio = if raw_dev.mean_abs_delta > 1.0e-12 {
        best_harmonized.mean_abs_delta / raw_dev.mean_abs_delta
    } else {
        1.0
    };

    write_harmonization_summary(
        &output_root.join("ccs_harmonization_summary.tsv"),
        train_consistency.cross_source,
        raw_dev,
        offset_dev,
        affine_dev,
    )?;

    let material_threshold =
        MATERIAL_DISAGREEMENT_FRACTION_OF_MODEL_MAE * V0362_FULL_DEV_MODEL_MAE_REFERENCE;
    let material_target_disagreement = raw_dev.mean_abs_delta >= material_threshold
        || dev_consistency.within_source.mean_abs_delta >= material_threshold;
    let harmonization_effective = best_ratio <= MATERIAL_HARMONIZATION_RATIO;
    let corpus_rebuild_recommended = material_target_disagreement;
    let harmonize_then_retrain = material_target_disagreement && harmonization_effective;
    let physics_modeling_recommended = !material_target_disagreement;

    println!(
        "v0363_train_cross_source_pairwise_mae\t{:.8}",
        train_consistency.cross_source.mean_abs_delta
    );
    println!(
        "v0363_dev_cross_source_pairwise_mae\t{:.8}",
        raw_dev.mean_abs_delta
    );
    println!(
        "v0363_dev_cross_source_pairwise_mae_after_offset\t{:.8}",
        offset_dev.mean_abs_delta
    );
    println!(
        "v0363_dev_cross_source_pairwise_mae_after_affine\t{:.8}",
        affine_dev.mean_abs_delta
    );
    println!("v0363_best_harmonization\t{best_harmonization}");
    println!(
        "v0363_best_harmonized_dev_pairwise_mae\t{:.8}",
        best_harmonized.mean_abs_delta
    );
    println!("v0363_best_harmonized_dev_ratio\t{best_ratio:.8}");
    println!(
        "v0363_dev_within_source_repeat_mae_to_group_mean\t{:.8}",
        dev_consistency.within_source.mean_abs_delta
    );
    println!(
        "v0363_dev_cross_run_pairwise_mae\t{:.8}",
        dev_consistency.cross_run.mean_abs_delta
    );
    println!("v0363_train_dev_shared_identities\t{train_dev_shared}");
    println!("v0363_material_disagreement_threshold\t{material_threshold:.8}");
    println!(
        "v0363_material_target_disagreement\t{}",
        yes_no(material_target_disagreement)
    );
    println!(
        "v0363_train_only_harmonization_effective\t{}",
        yes_no(harmonization_effective)
    );
    println!(
        "v0363_ccs_corpus_rebuild_recommended\t{}",
        yes_no(corpus_rebuild_recommended)
    );
    println!(
        "v0363_ccs_harmonize_then_retrain_recommended\t{}",
        yes_no(harmonize_then_retrain)
    );
    println!(
        "v0363_ccs_physics_modeling_recommended\t{}",
        yes_no(physics_modeling_recommended)
    );
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0363_selection\tNO");
    println!("historical_test_consumed\tNO");

    let summary = DiagnosticSummary {
        version: V0363_VERSION,
        objective: V0363_OBJECTIVE.to_string(),
        corpus_fingerprint,
        benchmark_manifest_fingerprint: benchmark_fingerprint,
        train_ccs_records: train_rows.len(),
        dev_ccs_records: dev_rows.len(),
        train_unique_identities: train_unique,
        dev_unique_identities: dev_unique,
        train_cross_source_identities: train_cross_source,
        dev_cross_source_identities: dev_cross_source,
        train_dev_shared_identities: train_dev_shared,
        train_cross_source_pairwise_mae: train_consistency.cross_source.mean_abs_delta,
        dev_cross_source_pairwise_mae: raw_dev.mean_abs_delta,
        dev_cross_source_pairwise_mae_after_offset: offset_dev.mean_abs_delta,
        dev_cross_source_pairwise_mae_after_affine: affine_dev.mean_abs_delta,
        best_harmonization: best_harmonization.to_string(),
        best_harmonized_dev_pairwise_mae: best_harmonized.mean_abs_delta,
        best_harmonized_dev_ratio: best_ratio,
        dev_within_source_repeat_mae_to_group_mean: dev_consistency.within_source.mean_abs_delta,
        dev_cross_run_pairwise_mae: dev_consistency.cross_run.mean_abs_delta,
        material_target_disagreement,
        train_only_harmonization_effective: harmonization_effective,
        ccs_corpus_rebuild_recommended: corpus_rebuild_recommended,
        ccs_harmonize_then_retrain_recommended: harmonize_then_retrain,
        ccs_physics_modeling_recommended: physics_modeling_recommended,
        train_holdout_consumed: false,
        historical_validation_consumed: false,
        historical_test_consumed: false,
    };
    fs::write(
        output_root.join("ccs_target_consistency_summary.yaml"),
        serde_yaml::to_string(&summary)?,
    )?;

    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct ConsistencySummary {
    within_source: DisagreementMetrics,
    cross_source: DisagreementMetrics,
    cross_run: DisagreementMetrics,
}

fn collect_rows(
    records: &[FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
) -> Result<Vec<LabelRow>> {
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
        let Some(target) = record.ccs.filter(|value| value.is_finite()) else {
            continue;
        };
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("missing provenance for record {index}"))?;
        rows.push(LabelRow {
            record_index: index,
            source_id: source.source_id.clone(),
            source_record_index: source.source_record_index,
            run_id: record
                .run_id
                .clone()
                .unwrap_or_else(|| "missing".to_string()),
            identity: peptidoform_charge_key(record),
            charge: record
                .context
                .charge
                .map(|value| value.to_string())
                .unwrap_or_else(|| "missing".to_string()),
            target: f64::from(target),
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
            let identity = modification.identity_label();
            format!("{site}:{identity}:{:+.4}", modification.mass_delta)
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

fn identity_groups(rows: &[LabelRow]) -> BTreeMap<String, Vec<usize>> {
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (index, row) in rows.iter().enumerate() {
        groups.entry(row.identity.clone()).or_default().push(index);
    }
    groups
}

fn count_cross_source_identities(
    groups: &BTreeMap<String, Vec<usize>>,
    rows: &[LabelRow],
) -> usize {
    groups
        .values()
        .filter(|indices| {
            indices
                .iter()
                .map(|&index| rows[index].source_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                >= 2
        })
        .count()
}

fn write_target_distributions(path: &Path, train: &[LabelRow], dev: &[LabelRow]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "partition\tdimension\tstratum\trecords\tunique_identities\tmean\tmedian\tstd\tmin\tq05\tq95\tmax"
    )?;
    for (partition, rows) in [("TRAIN", train), ("DEV", dev)] {
        for (dimension, key_fn) in [
            ("source", source_key as fn(&LabelRow) -> String),
            ("charge", charge_key as fn(&LabelRow) -> String),
            (
                "source_charge",
                source_charge_key as fn(&LabelRow) -> String,
            ),
        ] {
            let mut groups = BTreeMap::<String, Vec<&LabelRow>>::new();
            for row in rows {
                groups.entry(key_fn(row)).or_default().push(row);
            }
            for (stratum, group) in groups {
                let metrics = distribution_metrics(&group);
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
                    partition,
                    dimension,
                    escape_tsv(&stratum),
                    metrics.records,
                    metrics.unique_identities,
                    metrics.mean,
                    metrics.median,
                    metrics.std,
                    metrics.min,
                    metrics.q05,
                    metrics.q95,
                    metrics.max,
                )?;
            }
        }
    }
    Ok(())
}

fn source_key(row: &LabelRow) -> String {
    row.source_id.clone()
}

fn charge_key(row: &LabelRow) -> String {
    row.charge.clone()
}

fn source_charge_key(row: &LabelRow) -> String {
    format!("{}|{}", row.source_id, row.charge)
}

fn distribution_metrics(rows: &[&LabelRow]) -> DistributionMetrics {
    let mut values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
    values.sort_by(|a, b| a.total_cmp(b));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    DistributionMetrics {
        records: rows.len(),
        unique_identities: rows
            .iter()
            .map(|row| row.identity.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        mean,
        median: quantile_sorted(&values, 0.5),
        std: variance.sqrt(),
        min: values[0],
        q05: quantile_sorted(&values, 0.05),
        q95: quantile_sorted(&values, 0.95),
        max: values[values.len() - 1],
    }
}

fn consistency_audit(
    partition: &str,
    rows: &[LabelRow],
    groups: &BTreeMap<String, Vec<usize>>,
    identity_path: &Path,
    source_pair_path: &Path,
    source_outlier_path: &Path,
) -> Result<ConsistencySummary> {
    let mut identity_writer = BufWriter::new(File::create(identity_path)?);
    writeln!(
        identity_writer,
        "partition\tidentity\trecords\tsources\truns\ttarget_mean\ttarget_std\ttarget_range\twithin_source_mae_to_group_mean\tcross_source_pairwise_mae\tcross_run_pairwise_mae"
    )?;

    let mut within_abs = Vec::<f64>::new();
    let mut cross_source_abs = Vec::<f64>::new();
    let mut cross_run_abs = Vec::<f64>::new();
    let mut source_pairs = BTreeMap::<(String, String), PairAccumulator>::new();
    let mut within_identities = 0usize;
    let mut cross_source_identities = 0usize;
    let mut cross_run_identities = 0usize;

    for (identity, indices) in groups {
        if indices.len() < 2 {
            continue;
        }
        let mut source_targets = BTreeMap::<String, Vec<f64>>::new();
        let mut run_targets = BTreeMap::<String, Vec<f64>>::new();
        let mut source_run_targets = BTreeMap::<String, BTreeMap<String, Vec<f64>>>::new();
        let mut all_targets = Vec::with_capacity(indices.len());
        for &index in indices {
            let row = &rows[index];
            source_targets
                .entry(row.source_id.clone())
                .or_default()
                .push(row.target);
            run_targets
                .entry(format!("{}|{}", row.source_id, row.run_id))
                .or_default()
                .push(row.target);
            source_run_targets
                .entry(row.source_id.clone())
                .or_default()
                .entry(row.run_id.clone())
                .or_default()
                .push(row.target);
            all_targets.push(row.target);
        }

        let mut within_for_identity = Vec::new();
        for targets in source_targets.values() {
            if targets.len() < 2 {
                continue;
            }
            let mean = targets.iter().sum::<f64>() / targets.len() as f64;
            for &target in targets {
                let delta = (target - mean).abs();
                within_abs.push(delta);
                within_for_identity.push(delta);
            }
        }
        if !within_for_identity.is_empty() {
            within_identities += 1;
        }

        let source_means = source_targets
            .iter()
            .map(|(source, targets)| {
                (
                    source.clone(),
                    targets.iter().sum::<f64>() / targets.len() as f64,
                )
            })
            .collect::<Vec<_>>();
        let mut cross_source_for_identity = Vec::new();
        for i in 0..source_means.len() {
            for j in (i + 1)..source_means.len() {
                let (source_a, value_a) = &source_means[i];
                let (source_b, value_b) = &source_means[j];
                let delta = value_a - value_b;
                let abs = delta.abs();
                cross_source_abs.push(abs);
                cross_source_for_identity.push(abs);
                source_pairs
                    .entry((source_a.clone(), source_b.clone()))
                    .or_default()
                    .update(*value_a, *value_b);
            }
        }
        if !cross_source_for_identity.is_empty() {
            cross_source_identities += 1;
        }

        let mut cross_run_for_identity = Vec::new();
        for runs in source_run_targets.values() {
            if runs.len() < 2 {
                continue;
            }
            let run_means = runs
                .values()
                .map(|targets| targets.iter().sum::<f64>() / targets.len() as f64)
                .collect::<Vec<_>>();
            for i in 0..run_means.len() {
                for j in (i + 1)..run_means.len() {
                    let abs = (run_means[i] - run_means[j]).abs();
                    cross_run_abs.push(abs);
                    cross_run_for_identity.push(abs);
                }
            }
        }
        if !cross_run_for_identity.is_empty() {
            cross_run_identities += 1;
        }

        all_targets.sort_by(|a, b| a.total_cmp(b));
        let mean = all_targets.iter().sum::<f64>() / all_targets.len() as f64;
        let variance = all_targets
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / all_targets.len() as f64;
        writeln!(
            identity_writer,
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            partition,
            escape_tsv(identity),
            indices.len(),
            source_targets.len(),
            run_targets.len(),
            mean,
            variance.sqrt(),
            all_targets[all_targets.len() - 1] - all_targets[0],
            mean_or_zero(&within_for_identity),
            mean_or_zero(&cross_source_for_identity),
            mean_or_zero(&cross_run_for_identity),
        )?;
    }

    write_source_pair_table(source_pair_path, partition, &source_pairs)?;
    write_source_outlier_table(source_outlier_path, partition, &source_pairs)?;

    Ok(ConsistencySummary {
        within_source: disagreement_metrics(within_identities, within_abs),
        cross_source: disagreement_metrics(cross_source_identities, cross_source_abs),
        cross_run: disagreement_metrics(cross_run_identities, cross_run_abs),
    })
}

impl PairAccumulator {
    fn update(&mut self, x: f64, y: f64) {
        let delta = x - y;
        let abs = delta.abs();
        self.shared_identities += 1;
        self.sum_delta += delta;
        self.sum_abs += abs;
        self.sum_sq += delta * delta;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.sum_xy += x * y;
        self.max_abs = self.max_abs.max(abs);
    }

    fn mean_delta(&self) -> f64 {
        self.sum_delta / self.shared_identities.max(1) as f64
    }

    fn mae(&self) -> f64 {
        self.sum_abs / self.shared_identities.max(1) as f64
    }

    fn rmse(&self) -> f64 {
        (self.sum_sq / self.shared_identities.max(1) as f64).sqrt()
    }

    fn pearson(&self) -> f64 {
        let n = self.shared_identities as f64;
        if n < 2.0 {
            return 0.0;
        }
        let numerator = n * self.sum_xy - self.sum_x * self.sum_y;
        let left = n * self.sum_x2 - self.sum_x * self.sum_x;
        let right = n * self.sum_y2 - self.sum_y * self.sum_y;
        let denominator = (left.max(0.0) * right.max(0.0)).sqrt();
        if denominator > 1.0e-12 {
            (numerator / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

fn write_source_pair_table(
    path: &Path,
    partition: &str,
    pairs: &BTreeMap<(String, String), PairAccumulator>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "partition\tsource_a\tsource_b\tshared_identities\tmean_delta_a_minus_b\tmae_delta\trmse_delta\tpearson\tmax_abs_delta"
    )?;
    for ((source_a, source_b), pair) in pairs {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            partition,
            escape_tsv(source_a),
            escape_tsv(source_b),
            pair.shared_identities,
            pair.mean_delta(),
            pair.mae(),
            pair.rmse(),
            pair.pearson(),
            pair.max_abs,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct SourceOutlierAccumulator {
    partners: BTreeSet<String>,
    overlap_comparisons: usize,
    weighted_abs: f64,
    weighted_signed: f64,
    max_pair_mae: f64,
}

fn write_source_outlier_table(
    path: &Path,
    partition: &str,
    pairs: &BTreeMap<(String, String), PairAccumulator>,
) -> Result<()> {
    let mut sources = BTreeMap::<String, SourceOutlierAccumulator>::new();
    for ((source_a, source_b), pair) in pairs {
        let n = pair.shared_identities;
        let pair_mae = pair.mae();
        let signed = pair.mean_delta();
        let a = sources.entry(source_a.clone()).or_default();
        a.partners.insert(source_b.clone());
        a.overlap_comparisons += n;
        a.weighted_abs += pair_mae * n as f64;
        a.weighted_signed += signed * n as f64;
        a.max_pair_mae = a.max_pair_mae.max(pair_mae);

        let b = sources.entry(source_b.clone()).or_default();
        b.partners.insert(source_a.clone());
        b.overlap_comparisons += n;
        b.weighted_abs += pair_mae * n as f64;
        b.weighted_signed -= signed * n as f64;
        b.max_pair_mae = b.max_pair_mae.max(pair_mae);
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "partition\tsource\tpartners\tshared_identity_comparisons\tweighted_pairwise_mae\tweighted_signed_delta_source_minus_partners\tmax_pair_mae"
    )?;
    for (source, value) in sources {
        let denom = value.overlap_comparisons.max(1) as f64;
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}",
            partition,
            escape_tsv(&source),
            value.partners.len(),
            value.overlap_comparisons,
            value.weighted_abs / denom,
            value.weighted_signed / denom,
            value.max_pair_mae,
        )?;
    }
    Ok(())
}

fn disagreement_metrics(identities: usize, mut abs_deltas: Vec<f64>) -> DisagreementMetrics {
    if abs_deltas.is_empty() {
        return DisagreementMetrics {
            identities,
            ..DisagreementMetrics::default()
        };
    }
    abs_deltas.sort_by(|a, b| a.total_cmp(b));
    let sum_sq = abs_deltas.iter().map(|value| value * value).sum::<f64>();
    DisagreementMetrics {
        identities,
        comparisons: abs_deltas.len(),
        mean_abs_delta: abs_deltas.iter().sum::<f64>() / abs_deltas.len() as f64,
        rmse_delta: (sum_sq / abs_deltas.len() as f64).sqrt(),
        median_abs_delta: quantile_sorted(&abs_deltas, 0.5),
        q90_abs_delta: quantile_sorted(&abs_deltas, 0.90),
        q95_abs_delta: quantile_sorted(&abs_deltas, 0.95),
        max_abs_delta: abs_deltas[abs_deltas.len() - 1],
    }
}

fn write_partition_overlap(
    path: &Path,
    train_rows: &[LabelRow],
    dev_rows: &[LabelRow],
    train_groups: &BTreeMap<String, Vec<usize>>,
    dev_groups: &BTreeMap<String, Vec<usize>>,
) -> Result<usize> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "identity\ttrain_records\tdev_records\ttrain_sources\tdev_sources\ttrain_mean_ccs\tdev_mean_ccs\tabs_mean_delta"
    )?;
    let mut shared = 0usize;
    for (identity, train_indices) in train_groups {
        let Some(dev_indices) = dev_groups.get(identity) else {
            continue;
        };
        shared += 1;
        let train_mean = mean_indices(train_rows, train_indices);
        let dev_mean = mean_indices(dev_rows, dev_indices);
        let train_sources = train_indices
            .iter()
            .map(|&index| train_rows[index].source_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let dev_sources = dev_indices
            .iter()
            .map(|&index| dev_rows[index].source_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}",
            escape_tsv(identity),
            train_indices.len(),
            dev_indices.len(),
            train_sources,
            dev_sources,
            train_mean,
            dev_mean,
            (train_mean - dev_mean).abs(),
        )?;
    }
    Ok(shared)
}

fn mean_indices(rows: &[LabelRow], indices: &[usize]) -> f64 {
    indices.iter().map(|&index| rows[index].target).sum::<f64>() / indices.len() as f64
}

fn fit_source_harmonization(
    rows: &[LabelRow],
    groups: &BTreeMap<String, Vec<usize>>,
) -> Result<BTreeMap<String, SourceFit>> {
    let mut examples = BTreeMap::<String, Vec<(f64, f64)>>::new();
    let mut all_sources = BTreeSet::<String>::new();
    for row in rows {
        all_sources.insert(row.source_id.clone());
    }

    for indices in groups.values() {
        let source_means = source_means_for_group(rows, indices);
        if source_means.len() < 2 {
            continue;
        }
        let total = source_means.values().sum::<f64>();
        for (source, &source_mean) in &source_means {
            let other_mean = (total - source_mean) / (source_means.len() - 1) as f64;
            examples
                .entry(source.clone())
                .or_default()
                .push((source_mean, other_mean));
        }
    }

    let mut fits = BTreeMap::<String, SourceFit>::new();
    for source in all_sources {
        let source_examples = examples.get(&source).map(Vec::as_slice).unwrap_or(&[]);
        if source_examples.len() < MIN_SOURCE_SHARED_IDENTITIES {
            fits.insert(
                source,
                SourceFit {
                    shared_identities: source_examples.len(),
                    blend: 0.0,
                    offset: 0.0,
                    affine: Affine {
                        intercept: 0.0,
                        slope: 1.0,
                    },
                },
            );
            continue;
        }
        let blend =
            source_examples.len() as f64 / (source_examples.len() as f64 + SOURCE_SHRINKAGE);
        let mut deltas = source_examples
            .iter()
            .map(|(x, y)| y - x)
            .collect::<Vec<_>>();
        deltas.sort_by(|a, b| a.total_cmp(b));
        let raw_offset = quantile_sorted(&deltas, 0.5);
        let raw_affine = fit_xy_affine(source_examples)?;
        fits.insert(
            source,
            SourceFit {
                shared_identities: source_examples.len(),
                blend,
                offset: blend * raw_offset,
                affine: Affine {
                    intercept: blend * raw_affine.intercept,
                    slope: 1.0 + blend * (raw_affine.slope - 1.0),
                },
            },
        );
    }
    Ok(fits)
}

fn source_means_for_group(rows: &[LabelRow], indices: &[usize]) -> BTreeMap<String, f64> {
    let mut grouped = BTreeMap::<String, (f64, usize)>::new();
    for &index in indices {
        let row = &rows[index];
        let entry = grouped.entry(row.source_id.clone()).or_insert((0.0, 0));
        entry.0 += row.target;
        entry.1 += 1;
    }
    grouped
        .into_iter()
        .map(|(source, (sum, count))| (source, sum / count as f64))
        .collect()
}

fn fit_xy_affine(examples: &[(f64, f64)]) -> Result<Affine> {
    if examples.len() < 2 {
        return Ok(Affine {
            intercept: 0.0,
            slope: 1.0,
        });
    }
    let n = examples.len() as f64;
    let mean_x = examples.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = examples.iter().map(|(_, y)| y).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance = 0.0;
    for &(x, y) in examples {
        let dx = x - mean_x;
        covariance += dx * (y - mean_y);
        variance += dx * dx;
    }
    if variance <= 1.0e-12 {
        return Ok(Affine {
            intercept: mean_y - mean_x,
            slope: 1.0,
        });
    }
    let slope = covariance / variance;
    Ok(Affine {
        intercept: mean_y - slope * mean_x,
        slope,
    })
}

fn apply_source_offset(row: &LabelRow, fits: &BTreeMap<String, SourceFit>) -> f64 {
    fits.get(&row.source_id)
        .map(|fit| row.target + fit.offset)
        .unwrap_or(row.target)
}

fn apply_source_affine(row: &LabelRow, fits: &BTreeMap<String, SourceFit>) -> f64 {
    fits.get(&row.source_id)
        .map(|fit| fit.affine.intercept + fit.affine.slope * row.target)
        .unwrap_or(row.target)
}

fn cross_source_disagreement_with_transform<F>(
    rows: &[LabelRow],
    groups: &BTreeMap<String, Vec<usize>>,
    mut transform: F,
) -> DisagreementMetrics
where
    F: FnMut(&LabelRow) -> f64,
{
    let mut abs_deltas = Vec::<f64>::new();
    let mut identities = 0usize;
    for indices in groups.values() {
        let mut grouped = BTreeMap::<String, (f64, usize)>::new();
        for &index in indices {
            let row = &rows[index];
            let entry = grouped.entry(row.source_id.clone()).or_insert((0.0, 0));
            entry.0 += transform(row);
            entry.1 += 1;
        }
        if grouped.len() < 2 {
            continue;
        }
        identities += 1;
        let means = grouped
            .values()
            .map(|(sum, count)| sum / *count as f64)
            .collect::<Vec<_>>();
        for i in 0..means.len() {
            for j in (i + 1)..means.len() {
                abs_deltas.push((means[i] - means[j]).abs());
            }
        }
    }
    disagreement_metrics(identities, abs_deltas)
}

fn write_source_harmonization_parameters(
    path: &Path,
    fits: &BTreeMap<String, SourceFit>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "source\ttrain_shared_identities\tblend\toffset\taffine_intercept\taffine_slope"
    )?;
    for (source, fit) in fits {
        writeln!(
            writer,
            "{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            escape_tsv(source),
            fit.shared_identities,
            fit.blend,
            fit.offset,
            fit.affine.intercept,
            fit.affine.slope,
        )?;
    }
    Ok(())
}

fn write_harmonization_summary(
    path: &Path,
    train_raw: DisagreementMetrics,
    dev_raw: DisagreementMetrics,
    dev_offset: DisagreementMetrics,
    dev_affine: DisagreementMetrics,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "partition\tmethod\tidentities\tcomparisons\tmean_abs_delta\trmse_delta\tmedian_abs_delta\tq90_abs_delta\tq95_abs_delta\tmax_abs_delta\tratio_to_dev_raw"
    )?;
    write_disagreement_row(&mut writer, "TRAIN", "raw", train_raw, None)?;
    write_disagreement_row(
        &mut writer,
        "DEV",
        "raw",
        dev_raw,
        Some(dev_raw.mean_abs_delta),
    )?;
    write_disagreement_row(
        &mut writer,
        "DEV",
        "source_offset_train_fit",
        dev_offset,
        Some(dev_raw.mean_abs_delta),
    )?;
    write_disagreement_row(
        &mut writer,
        "DEV",
        "source_affine_train_fit",
        dev_affine,
        Some(dev_raw.mean_abs_delta),
    )?;
    Ok(())
}

fn write_disagreement_row<W: Write>(
    writer: &mut W,
    partition: &str,
    method: &str,
    metrics: DisagreementMetrics,
    baseline: Option<f64>,
) -> Result<()> {
    let ratio = baseline
        .filter(|value| *value > 1.0e-12)
        .map(|value| metrics.mean_abs_delta / value)
        .unwrap_or(1.0);
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
        partition,
        method,
        metrics.identities,
        metrics.comparisons,
        metrics.mean_abs_delta,
        metrics.rmse_delta,
        metrics.median_abs_delta,
        metrics.q90_abs_delta,
        metrics.q95_abs_delta,
        metrics.max_abs_delta,
        ratio,
    )?;
    Ok(())
}

fn quantile_sorted(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let weight = position - lower as f64;
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

fn mean_or_zero(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
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

    #[test]
    fn v0363_affine_recovers_linear_map() {
        let examples = vec![(1.0, 4.0), (2.0, 6.0), (3.0, 8.0), (4.0, 10.0)];
        let affine = fit_xy_affine(&examples).unwrap();
        assert!((affine.intercept - 2.0).abs() < 1.0e-10);
        assert!((affine.slope - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn v0363_quantile_interpolates() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert!((quantile_sorted(&values, 0.5) - 2.5).abs() < 1.0e-10);
        assert!((quantile_sorted(&values, 0.0) - 1.0).abs() < 1.0e-10);
        assert!((quantile_sorted(&values, 1.0) - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn v0363_pair_accumulator_reports_expected_metrics() {
        let mut pair = PairAccumulator::default();
        pair.update(10.0, 8.0);
        pair.update(20.0, 17.0);
        assert!((pair.mae() - 2.5).abs() < 1.0e-10);
        assert!((pair.mean_delta() - 2.5).abs() < 1.0e-10);
        assert_eq!(pair.shared_identities, 2);
    }
}
