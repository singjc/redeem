//! Prepare a leakage-safe TRAIN-derived split for v0.25 GPU re-foundation.
//!
//! Only records that belonged to the historical benchmark TRAIN partition are
//! eligible. They are re-split sequence-disjoint into TRAIN-core, TRAIN-dev,
//! and TRAIN-holdout. Historical VALIDATION and TEST records are excluded.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    foundation_split_group_key, load_foundation_corpus, read_foundation_training_run_config,
    split_foundation_record_indices, FoundationBenchmarkEntry, FoundationBenchmarkManifest,
    FoundationPartition, FoundationSplitConfig, FoundationSplitMode,
    FOUNDATION_BENCHMARK_MANIFEST_VERSION,
};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_prepare_refoundation_v0250 SOURCE_RUN.yaml OUTPUT_DIR [dev_fraction=0.05] [holdout_fraction=0.05] [seed=20260925]"
        );
    }
    let source_yaml = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    let dev_fraction = parse_or(&args, 3, 0.05f64)?;
    let holdout_fraction = parse_or(&args, 4, 0.05f64)?;
    let seed = parse_or(&args, 5, 20_260_925u64)?;

    let mut run = read_foundation_training_run_config(&source_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let source_manifest_path = run.benchmark_manifest.clone();
    let source_manifest = FoundationBenchmarkManifest::read_tsv(&source_manifest_path)
        .with_context(|| format!("failed to read {:?}", source_manifest_path))?;
    source_manifest.validate_against_records(&corpus.records)?;

    let source_train: Vec<usize> = source_manifest
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .collect();
    if source_train.is_empty() {
        anyhow::bail!("historical benchmark TRAIN partition is empty");
    }

    let split_config = FoundationSplitConfig {
        mode: FoundationSplitMode::Sequence,
        validation_fraction: dev_fraction,
        test_fraction: holdout_fraction,
        seed,
    };
    let split = split_foundation_record_indices(&corpus.records, &source_train, &split_config)?;

    let train: BTreeSet<usize> = split.train.iter().copied().collect();
    let dev: BTreeSet<usize> = split.validation.iter().copied().collect();
    let holdout: BTreeSet<usize> = split.test.iter().copied().collect();

    let mut entries = Vec::with_capacity(source_train.len());
    for source_entry in source_manifest
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
    {
        let index = source_entry.record_index;
        let partition = if train.contains(&index) {
            FoundationPartition::Train
        } else if dev.contains(&index) {
            FoundationPartition::Validation
        } else if holdout.contains(&index) {
            FoundationPartition::Test
        } else {
            anyhow::bail!("TRAIN-derived split dropped record index {index}");
        };
        entries.push(FoundationBenchmarkEntry {
            record_index: index,
            record_fingerprint: source_entry.record_fingerprint,
            partition,
            identity_key: foundation_split_group_key(
                &corpus.records[index],
                FoundationSplitMode::Sequence,
                index,
            )?,
            sequence: source_entry.sequence.clone(),
            peptidoform: source_entry.peptidoform.clone(),
        });
    }
    entries.sort_by_key(|entry| entry.record_index);

    let manifest = FoundationBenchmarkManifest {
        format_version: FOUNDATION_BENCHMARK_MANIFEST_VERSION,
        dataset_fingerprint: source_manifest.dataset_fingerprint,
        source_records: source_manifest.source_records,
        selected_records: entries.len(),
        modified_only: source_manifest.modified_only,
        single_modification_family_only: false,
        excluded_mixed_family_records: source_manifest.excluded_mixed_family_records,
        split_config,
        summary: split.summary.clone(),
        entries,
    };
    manifest.validate_against_records(&corpus.records)?;

    fs::create_dir_all(&output_dir)?;
    let manifest_path = output_dir.join("benchmark_train_derived_v0250.tsv");
    manifest.write_tsv(&manifest_path)?;

    run.benchmark_manifest = manifest_path.clone();
    run.checkpoint_root = output_dir.join("unused_forward_only_checkpoint_root");
    run.resume = false;
    run.initial_model_safetensors = None;
    run.experiment_id = Some("redeem-refoundation-v0250-random-init".into());
    let run_path = output_dir.join("run_v0250.yaml");
    fs::write(&run_path, serde_yaml::to_string(&run)?)?;

    println!("v0250_prepare_version\tv0.25.0-train-derived-sequence-split");
    println!("source_run_yaml\t{}", source_yaml.display());
    println!("source_benchmark\t{}", source_manifest_path.display());
    println!("source_train_records\t{}", source_train.len());
    println!("train_core_records\t{}", split.summary.train_records);
    println!("train_dev_records\t{}", split.summary.validation_records);
    println!("train_holdout_records\t{}", split.summary.test_records);
    println!("train_core_groups\t{}", split.summary.train_groups);
    println!("train_dev_groups\t{}", split.summary.validation_groups);
    println!("train_holdout_groups\t{}", split.summary.test_groups);
    println!("split_mode\tsequence");
    println!("split_seed\t{seed}");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!("manifest\t{}", manifest_path.display());
    println!("run_yaml\t{}", run_path.display());
    Ok(())
}

fn parse_or<T>(args: &[String], index: usize, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse::<T>()
            .map_err(|error| anyhow::anyhow!("cannot parse argument {index}: {error}")),
        None => Ok(default),
    }
}
