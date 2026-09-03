//! Fit train-only cross-source RT harmonization and materialize a new corpus benchmark.
//!
//! This tool deliberately consumes only TRAIN records while fitting source affine transforms.
//! Validation is used only for a post-fit consistency audit. TEST is never inspected beyond
//! preserving its deterministic sequence partition assignment in the regenerated manifest.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    fit_foundation_rt_harmonization, load_foundation_corpus, read_foundation_training_run_config,
    FoundationBenchmarkManifest, FoundationRegressionNormalization,
    FoundationRegressionNormalizationStrategy, FoundationRtHarmonizationFitConfig,
    FoundationRtHarmonizationFitResult, RetentionTimeObjective,
};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 || args.len() > 8 {
        anyhow::bail!(
            "usage: foundation_fit_rt_harmonization INPUT_RUN.yaml OUTPUT_RUN.yaml CALIBRATION.tsv HARMONIZED_BENCHMARK.tsv HARMONIZED_PROVENANCE.tsv [min_pair_overlap=10] [min_source_shared_peptides=25]"
        );
    }

    let input_run_path = PathBuf::from(&args[1]);
    let output_run_path = PathBuf::from(&args[2]);
    let calibration_path = PathBuf::from(&args[3]);
    let benchmark_path = PathBuf::from(&args[4]);
    let provenance_path = PathBuf::from(&args[5]);
    let min_pair_overlap = parse_or(&args, 6, 10usize)?;
    let min_source_shared_peptides = parse_or(&args, 7, 25usize)?;

    let raw_run = read_foundation_training_run_config(&input_run_path)?;
    let raw_corpus = load_foundation_corpus(&raw_run.corpus)?;
    let raw_benchmark = FoundationBenchmarkManifest::read_tsv(&raw_run.benchmark_manifest)
        .with_context(|| {
            format!(
                "failed to read raw benchmark {:?}",
                raw_run.benchmark_manifest
            )
        })?;
    raw_benchmark.validate_against_records(&raw_corpus.records)?;

    let fit_config = FoundationRtHarmonizationFitConfig {
        min_pair_overlap,
        min_source_shared_peptides,
        ..FoundationRtHarmonizationFitConfig::default()
    };
    let fit = fit_foundation_rt_harmonization(
        &raw_corpus.records,
        &raw_corpus.provenance,
        &raw_benchmark,
        fit_config,
    )?;

    let mut harmonized_run = raw_run.clone();
    for source in &mut harmonized_run.corpus.sources {
        source.rt_harmonization = fit
            .sources
            .get(&source.id)
            .map(|summary| summary.transform.clone());
    }
    harmonized_run.trainer.collator.retention_time_objective = RetentionTimeObjective::Harmonized;
    harmonized_run.trainer.target_normalization.rt = FoundationRegressionNormalization {
        strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
        ..FoundationRegressionNormalization::default()
    };
    harmonized_run.benchmark_manifest = benchmark_path.clone();
    harmonized_run.resume = false;
    harmonized_run.initial_model_safetensors = None;
    harmonized_run.experiment_id = Some(format!(
        "v0136_train_only_rt_harmonization_{}",
        fit.calibration_id.replace(':', "_")
    ));

    let harmonized_corpus = load_foundation_corpus(&harmonized_run.corpus)?;
    let harmonized_benchmark = harmonized_corpus.build_benchmark_manifest(
        raw_benchmark.split_config.clone(),
        raw_benchmark.modified_only,
    )?;
    assert_partition_identity_parity(&raw_benchmark, &harmonized_benchmark)?;

    if let Some(parent) = output_run_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = calibration_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = benchmark_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = provenance_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&output_run_path, serde_yaml::to_string(&harmonized_run)?)?;
    write_calibration_tsv(&calibration_path, &fit, &raw_corpus, &raw_benchmark)?;
    let fit_yaml_path = calibration_path.with_extension("yaml");
    fs::write(&fit_yaml_path, serde_yaml::to_string(&fit)?)?;
    harmonized_benchmark.write_tsv(&benchmark_path)?;
    harmonized_corpus.write_provenance_tsv(&provenance_path)?;

    println!("rt_harmonization\t{}", fit.calibration_id);
    println!("fit_converged\t{}", fit.converged);
    println!("fit_iterations\t{}", fit.iterations);
    println!("fitted_sources\t{}", fit.fitted_sources.len());
    println!(
        "raw_corpus_fingerprint\tfnv1a64:{:016x}",
        raw_corpus.corpus_fingerprint
    );
    println!(
        "raw_benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        raw_benchmark.manifest_fingerprint()
    );
    println!(
        "harmonized_corpus_fingerprint\tfnv1a64:{:016x}",
        harmonized_corpus.corpus_fingerprint
    );
    println!(
        "harmonized_benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        harmonized_benchmark.manifest_fingerprint()
    );
    print_consistency(
        "train_distribution_baseline",
        fit.train_distribution_baseline,
    );
    print_consistency("train_harmonized", fit.train_harmonized);
    print_consistency(
        "validation_distribution_baseline",
        fit.validation_distribution_baseline,
    );
    print_consistency("validation_harmonized", fit.validation_harmonized);
    for source in fit.sources.values() {
        println!(
            "rt_source\tsource={}\ttrain_records={}\ttrain_peptidoforms={}\tshared_train_peptidoforms={}\traw_median={:.6}\traw_robust_scale={:.6}\tharmonized_scale={:.12}\tharmonized_offset={:.6}",
            source.source_id,
            source.train_rt_records,
            source.train_rt_peptidoforms,
            source.shared_train_peptidoforms,
            source.source_native_median,
            source.source_native_robust_scale,
            source.transform.scale,
            source.transform.offset,
        );
    }
    println!("output_run\t{}", output_run_path.display());
    println!("calibration_tsv\t{}", calibration_path.display());
    println!("calibration_yaml\t{}", fit_yaml_path.display());
    println!("benchmark\t{}", benchmark_path.display());
    println!("provenance\t{}", provenance_path.display());
    println!("partition_identity_parity\tPASS");
    println!("test_labels_consumed_by_fit\tNO");
    Ok(())
}

fn assert_partition_identity_parity(
    raw: &FoundationBenchmarkManifest,
    harmonized: &FoundationBenchmarkManifest,
) -> Result<()> {
    if raw.entries.len() != harmonized.entries.len() {
        anyhow::bail!(
            "harmonized benchmark selected {} entries but raw benchmark selected {}",
            harmonized.entries.len(),
            raw.entries.len()
        );
    }
    for (left, right) in raw.entries.iter().zip(&harmonized.entries) {
        if left.record_index != right.record_index
            || left.partition != right.partition
            || left.identity_key != right.identity_key
            || left.sequence != right.sequence
            || left.peptidoform != right.peptidoform
        {
            anyhow::bail!(
                "harmonized benchmark changed partition/identity assignment at record {}",
                left.record_index
            );
        }
    }
    Ok(())
}

fn write_calibration_tsv(
    path: &Path,
    fit: &FoundationRtHarmonizationFitResult,
    raw_corpus: &redeem_properties::foundation::FoundationCorpus,
    raw_benchmark: &FoundationBenchmarkManifest,
) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# calibration_id={}", fit.calibration_id)?;
    writeln!(
        writer,
        "# raw_corpus_fingerprint=fnv1a64:{:016x}",
        raw_corpus.corpus_fingerprint
    )?;
    writeln!(
        writer,
        "# raw_benchmark_manifest_fingerprint=fnv1a64:{:016x}",
        raw_benchmark.manifest_fingerprint()
    )?;
    writeln!(writer, "# fit_partition=train")?;
    writeln!(writer, "# validation_used_for_fit=false")?;
    writeln!(writer, "# test_used_for_fit_or_audit=false")?;
    writeln!(writer, "# canonical_center={}", fit.config.canonical_center)?;
    writeln!(writer, "# canonical_scale={}", fit.config.canonical_scale)?;
    writeln!(writer, "# converged={}", fit.converged)?;
    writeln!(writer, "# iterations={}", fit.iterations)?;
    writeln!(
        writer,
        "source_id\ttrain_rt_records\ttrain_rt_peptidoforms\tshared_train_peptidoforms\tsource_native_median\tsource_native_robust_scale\tharmonized_scale\tharmonized_offset\tcalibration_id"
    )?;
    for source in fit.sources.values() {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.16}\t{:.12}\t{}",
            source.source_id,
            source.train_rt_records,
            source.train_rt_peptidoforms,
            source.shared_train_peptidoforms,
            source.source_native_median,
            source.source_native_robust_scale,
            source.transform.scale,
            source.transform.offset,
            source.transform.calibration_id,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn print_consistency(
    label: &str,
    metrics: redeem_properties::foundation::FoundationRtCrossSourceConsistency,
) {
    println!(
        "rt_consistency\tlabel={}\tshared_peptidoforms={}\tobservations={}\tmae={}\trmse={}",
        label,
        metrics.shared_peptidoforms,
        metrics.observations,
        fmt_opt(metrics.mean_absolute_deviation),
        fmt_opt(metrics.root_mean_squared_deviation),
    );
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "NA".into())
}

fn parse_or<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse()
            .map_err(|error| anyhow::anyhow!("failed to parse argument {index}: {error}")),
        None => Ok(default),
    }
}
