//! F1 read-only RT target-contract audit.
//!
//! Question: does the unified trainer's source-native RT target (`retention_time.normalized`)
//! disagree across sources for the same peptidoform, while the TRAIN-fit harmonized target
//! used by v0.35-v0.52 agrees? If so, one context-free RT head receives contradictory labels.
//!
//! Reads TRAIN and Validation only. Never reads TEST labels. No model, no GPU, no writes
//! except stdout.
//!
//! usage: foundation_f1_rt_contract_audit RUN.yaml RUN_SHA256 CORPUS_FNV BENCHMARK_FNV
//!        foundation_f1_rt_contract_audit --self-test
//! Fingerprints use the checkpoint form `fnv1a64:<16 hex>`. Any mismatch stops the audit.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    foundation_rt_partition_consistency, load_foundation_corpus, FoundationBenchmarkManifest,
    FoundationCorpusConfig, FoundationPartition, FoundationRtCrossSourceConsistency,
    FoundationRtHarmonizationTransform,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Run {
    corpus: FoundationCorpusConfig,
    benchmark_manifest: PathBuf,
}

#[derive(Default)]
struct Moments {
    n: usize,
    sum: f64,
    sum_sq: f64,
}

impl Moments {
    fn add(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }
    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }
    fn sd(&self) -> Option<f64> {
        let mean = self.mean()?;
        (self.n > 1).then(|| (self.sum_sq / self.n as f64 - mean * mean).max(0.0).sqrt())
    }
}

fn fmt(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".into(), |v| format!("{v:.6}"))
}

fn line(label: &str, partition: &str, c: FoundationRtCrossSourceConsistency) {
    println!(
        "consistency\t{label}\t{partition}\t{}\t{}\t{}\t{}",
        c.shared_peptidoforms,
        c.observations,
        fmt(c.mean_absolute_deviation),
        fmt(c.root_mean_squared_deviation)
    );
}

fn sha256_file(path: &str) -> Result<String> {
    let output = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .context("run sha256sum")?;
    anyhow::ensure!(output.status.success(), "sha256sum failed for {path}");
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string())
}

fn self_test() {
    let mut m = Moments::default();
    for v in [1.0, 2.0, 3.0] {
        m.add(v);
    }
    assert!((m.mean().unwrap() - 2.0).abs() < 1e-12);
    assert!((m.sd().unwrap() - (2.0f64 / 3.0).sqrt()).abs() < 1e-12);
    assert_eq!(fmt(None), "NA");
    println!("F1_RT_CONTRACT_SELF_TEST=PASS");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--self-test") {
        self_test();
        return Ok(());
    }
    anyhow::ensure!(
        args.len() == 5,
        "usage: foundation_f1_rt_contract_audit RUN.yaml RUN_SHA256 CORPUS_FNV BENCHMARK_FNV"
    );
    let observed = sha256_file(&args[1])?;
    anyhow::ensure!(
        args[2] == observed,
        "run YAML SHA256 mismatch: expected {} observed {observed}",
        args[2]
    );
    println!("run_yaml\t{}\nrun_yaml_sha256\t{observed}", args[1]);

    let run: Run = serde_yaml::from_str(&std::fs::read_to_string(&args[1])?)?;
    let transforms: BTreeMap<String, FoundationRtHarmonizationTransform> = run
        .corpus
        .sources
        .iter()
        .filter_map(|s| s.rt_harmonization.clone().map(|t| (s.id.clone(), t)))
        .collect();
    let identity: BTreeMap<_, _> = transforms
        .keys()
        .map(|id| {
            let transform = FoundationRtHarmonizationTransform {
                scale: 1.0,
                offset: 0.0,
                calibration_id: "identity".into(),
            };
            (id.clone(), transform)
        })
        .collect();
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let corpus_fnv = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fnv = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    anyhow::ensure!(
        corpus_fnv == args[3],
        "corpus fingerprint {corpus_fnv} != {}",
        args[3]
    );
    anyhow::ensure!(
        benchmark_fnv == args[4],
        "benchmark fingerprint {benchmark_fnv} != {}",
        args[4]
    );
    anyhow::ensure!(
        corpus.provenance.len() == corpus.records.len(),
        "provenance/record count mismatch"
    );
    println!(
        "corpus_fingerprint\t{corpus_fnv}\nbenchmark_fingerprint\t{benchmark_fnv}\nrecords\t{}\nharmonized_sources\t{}\ntest_labels_read\tNO",
        corpus.records.len(),
        transforms.len()
    );

    // Same-peptidoform cross-source agreement, raw versus harmonized coordinate.
    println!("#consistency\tcoordinate\tpartition\tshared_peptidoforms\tobservations\tmad\trmsd");
    for (name, partition) in [
        ("train", FoundationPartition::Train),
        ("validation", FoundationPartition::Validation),
    ] {
        for (label, map) in [("source_native", &identity), ("harmonized", &transforms)] {
            let c = foundation_rt_partition_consistency(
                &corpus.records,
                &corpus.provenance,
                &benchmark,
                partition,
                map,
            )?;
            line(label, name, c);
        }
    }

    // Per-source label coverage and moments in both coordinates (TRAIN, Validation).
    let mut stats = BTreeMap::<(String, &str, &str), Moments>::new();
    for entry in &benchmark.entries {
        let partition = match entry.partition {
            FoundationPartition::Train => "train",
            FoundationPartition::Validation => "validation",
            FoundationPartition::Test => continue,
        };
        let record = &corpus.records[entry.record_index];
        let source = corpus.provenance[entry.record_index].source_id.clone();
        let rt = record.retention_time;
        if let Some(v) = rt.normalized.filter(|v| v.is_finite()) {
            stats
                .entry((source.clone(), partition, "source_native"))
                .or_default()
                .add(f64::from(v));
        }
        if let Some(v) = rt.harmonized.filter(|v| v.is_finite()) {
            stats
                .entry((source.clone(), partition, "harmonized"))
                .or_default()
                .add(f64::from(v));
        }
        if let Some(v) = rt.observed_seconds.filter(|v| v.is_finite()) {
            stats
                .entry((source, partition, "observed_seconds"))
                .or_default()
                .add(f64::from(v));
        }
    }
    println!("#source_rt\tsource\tpartition\tcoordinate\tcount\tmean\tsd");
    for ((source, partition, coordinate), m) in &stats {
        println!(
            "source_rt\t{source}\t{partition}\t{coordinate}\t{}\t{}\t{}",
            m.n,
            fmt(m.mean()),
            fmt(m.sd())
        );
    }
    println!("F1_RT_CONTRACT_AUDIT=COMPLETE");
    Ok(())
}
