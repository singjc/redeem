//! F1-D scorer: RT of unified checkpoints on one common (harmonized) coordinate.
//!
//! Uses the first MAX_RECORDS Validation records, which is the trainer's fixed DEV slice.
//! Each checkpoint's prediction is mapped to the harmonized coordinate with the TRAIN-fit
//! per-source transform when the checkpoint predicts source-native RT. Records need a
//! harmonized label, a source-native label, and a source transform. Never reads TEST.
//!
//! usage: foundation_f1_rt_arm_score RUN.yaml RUN_SHA256 CORPUS_FNV BENCHMARK_FNV MAX_RECORDS CKPT...
//!        foundation_f1_rt_arm_score --self-test

use anyhow::{Context, Result};
use candle_core::Device;
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationBenchmarkManifest, FoundationCheckpointMetadata,
    FoundationCorpusConfig, FoundationModel, FoundationPartition,
    FoundationRtHarmonizationTransform, RetentionTimeObjective,
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
struct Errors {
    target: Vec<f64>,
    predicted: Vec<f64>,
}

impl Errors {
    fn push(&mut self, target: f64, predicted: f64) {
        self.target.push(target);
        self.predicted.push(predicted);
    }
    fn mae(&self) -> f64 {
        let n = self.target.len().max(1) as f64;
        self.target
            .iter()
            .zip(&self.predicted)
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n
    }
    fn pearson(&self) -> f64 {
        let n = self.target.len() as f64;
        let mt = self.target.iter().sum::<f64>() / n;
        let mp = self.predicted.iter().sum::<f64>() / n;
        let (mut c, mut vt, mut vp) = (0.0, 0.0, 0.0);
        for (t, p) in self.target.iter().zip(&self.predicted) {
            c += (t - mt) * (p - mp);
            vt += (t - mt).powi(2);
            vp += (p - mp).powi(2);
        }
        c / (vt * vp).sqrt()
    }
}

/// Map a prediction made in `objective` space into the harmonized coordinate.
fn to_harmonized(
    objective: RetentionTimeObjective,
    predicted: f32,
    transform: &FoundationRtHarmonizationTransform,
) -> Option<f64> {
    match objective {
        RetentionTimeObjective::Harmonized => Some(f64::from(predicted)),
        RetentionTimeObjective::Normalized | RetentionTimeObjective::IntrinsicAndObserved => {
            transform.harmonize(predicted).map(f64::from)
        }
        RetentionTimeObjective::Observed => None,
    }
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
    let t = FoundationRtHarmonizationTransform {
        scale: 0.5,
        offset: 30.0,
        calibration_id: "t".into(),
    };
    assert_eq!(
        to_harmonized(RetentionTimeObjective::IntrinsicAndObserved, 40.0, &t),
        Some(50.0)
    );
    assert_eq!(
        to_harmonized(RetentionTimeObjective::Harmonized, 40.0, &t),
        Some(40.0)
    );
    assert_eq!(
        to_harmonized(RetentionTimeObjective::Observed, 40.0, &t),
        None
    );
    let mut e = Errors::default();
    for (a, b) in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)] {
        e.push(a, b);
    }
    assert!((e.mae() - 2.0).abs() < 1e-12);
    assert!((e.pearson() - 1.0).abs() < 1e-12);
    println!("F1_RT_ARM_SCORE_SELF_TEST=PASS");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--self-test") {
        self_test();
        return Ok(());
    }
    anyhow::ensure!(
        args.len() >= 7,
        "usage: foundation_f1_rt_arm_score RUN.yaml RUN_SHA256 CORPUS_FNV BENCHMARK_FNV MAX_RECORDS CKPT..."
    );
    let observed = sha256_file(&args[1])?;
    anyhow::ensure!(args[2] == observed, "run YAML SHA256 mismatch: {observed}");
    let max_records: usize = args[5].parse().context("MAX_RECORDS")?;
    let run: Run = serde_yaml::from_str(&std::fs::read_to_string(&args[1])?)?;
    let transforms: BTreeMap<String, FoundationRtHarmonizationTransform> = run
        .corpus
        .sources
        .iter()
        .filter_map(|s| s.rt_harmonization.clone().map(|t| (s.id.clone(), t)))
        .collect();
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let corpus_fnv = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fnv = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    anyhow::ensure!(corpus_fnv == args[3], "corpus fingerprint {corpus_fnv}");
    anyhow::ensure!(
        benchmark_fnv == args[4],
        "benchmark fingerprint {benchmark_fnv}"
    );
    let indices: Vec<usize> = benchmark
        .partition_indices(FoundationPartition::Validation)
        .into_iter()
        .take(max_records)
        .collect();
    anyhow::ensure!(
        indices.len() == max_records,
        "Validation has too few records"
    );
    // Common eligible set: identical for every checkpoint.
    let eligible: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&i| {
            let rt = corpus.records[i].retention_time;
            rt.normalized.is_some_and(f32::is_finite)
                && rt.harmonized.is_some_and(f32::is_finite)
                && transforms.contains_key(&corpus.provenance[i].source_id)
        })
        .collect();
    println!(
        "run_yaml_sha256\t{observed}\ncorpus_fingerprint\t{corpus_fnv}\nbenchmark_fingerprint\t{benchmark_fnv}\nvalidation_records\t{}\neligible_rt_records\t{}\ntest_labels_read\tNO",
        indices.len(),
        eligible.len()
    );
    println!(
        "#score\tcheckpoint\trt_objective\tcompleted_epochs\tn\tmae_harmonized\tpearson_harmonized"
    );
    println!("#source\tcheckpoint\tsource\tn\tmae_harmonized");
    for checkpoint in &args[6..] {
        let metadata: FoundationCheckpointMetadata = serde_yaml::from_str(
            &std::fs::read_to_string(PathBuf::from(checkpoint).join("metadata.yaml"))?,
        )?;
        anyhow::ensure!(
            metadata.corpus_fingerprint.as_deref() == Some(corpus_fnv.as_str())
                && metadata.benchmark_fingerprint.as_deref() == Some(benchmark_fnv.as_str()),
            "checkpoint fingerprints differ: {checkpoint}"
        );
        let model = FoundationModel::load(checkpoint, Device::Cpu)?;
        let objective = model.config().rt_objective;
        let mut all = Errors::default();
        let mut by_source = BTreeMap::<String, Errors>::new();
        for chunk in eligible.chunks(8) {
            let records: Vec<_> = chunk.iter().map(|&i| corpus.records[i].clone()).collect();
            for (&i, prediction) in chunk.iter().zip(model.predict_records(&records)?) {
                let source = &corpus.provenance[i].source_id;
                let target = f64::from(corpus.records[i].retention_time.harmonized.unwrap());
                let predicted = prediction
                    .rt
                    .and_then(|p| to_harmonized(objective, p, &transforms[source]))
                    .with_context(|| format!("no RT prediction for record {i}"))?;
                all.push(target, predicted);
                by_source
                    .entry(source.clone())
                    .or_default()
                    .push(target, predicted);
            }
        }
        println!(
            "score\t{checkpoint}\t{objective:?}\t{}\t{}\t{:.6}\t{:.6}",
            metadata.completed_epochs,
            all.target.len(),
            all.mae(),
            all.pearson()
        );
        for (source, errors) in &by_source {
            println!(
                "source\t{checkpoint}\t{source}\t{}\t{:.6}",
                errors.target.len(),
                errors.mae()
            );
        }
    }
    println!("F1_RT_ARM_SCORE=COMPLETE");
    Ok(())
}
