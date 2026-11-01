//! F2-A unified scorecard: fixed, source-stratified Validation cohort with per-record
//! RT / CCS / MS2 predictions for unified and frozen v0.52 checkpoints.
//!
//! The cohort excludes the first 2,048 Validation ordinals (used by F1 and unified selection).
//! Per source and per task, up to PER_TASK labelled records are drawn with a fixed seed.
//! Never reads TEST or any other partition.
//!
//! usage:
//!   foundation_f2_scorecard select RUN.yaml RUN_SHA256 CORPUS_FNV BENCHMARK_FNV SEED PER_TASK > cohort.tsv
//!   foundation_f2_scorecard score  RUN.yaml RUN_SHA256 CORPUS_FNV BENCHMARK_FNV COHORT.tsv V035_TEACHER LABEL=CKPT...
//!   foundation_f2_scorecard --self-test
//!
//! `score` writes one row per (checkpoint, record, task). A checkpoint whose metadata has
//! `version: 520` is read with the frozen v0.52 readers (CCS needs the v0.35 teacher).
//! RT is reported in the harmonized coordinate (source-native models are mapped with the
//! TRAIN-fit per-source transform). MS2 cosine uses annotated channels 0..4 only.

use anyhow::{Context, Result};
use candle_core::Device;
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationBenchmarkManifest, FoundationCorpusConfig, FoundationModel,
    FoundationPartition, FoundationRtHarmonizationTransform, FoundationTrainingRecord,
    ResearchV0520Forward, ResearchV0520Mobility, RetentionTimeObjective,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const EXCLUDED_PREFIX: usize = 2048;
const MS2_CHANNELS: usize = 4;
const MIN_MS2_FRAGMENTS: usize = 6;
const BATCH: usize = 16;

#[derive(Deserialize)]
struct Run {
    corpus: FoundationCorpusConfig,
    benchmark_manifest: PathBuf,
}

#[derive(Deserialize)]
struct V0520Header {
    version: u32,
    rt_objective: RetentionTimeObjective,
}

/// SplitMix64: small, dependency-free, deterministic.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Deterministic sample of up to `k` items (partial Fisher-Yates), returned sorted.
fn sample(mut items: Vec<usize>, k: usize, rng: &mut Rng) -> Vec<usize> {
    let k = k.min(items.len());
    for i in 0..k {
        let j = i + (rng.next() % (items.len() - i) as u64) as usize;
        items.swap(i, j);
    }
    items.truncate(k);
    items.sort_unstable();
    items
}

fn has_rt(r: &FoundationTrainingRecord) -> bool {
    r.retention_time.harmonized.is_some_and(f32::is_finite)
        && r.retention_time.normalized.is_some_and(f32::is_finite)
}
fn has_ccs(r: &FoundationTrainingRecord) -> bool {
    r.ccs.is_some_and(|v| v.is_finite() && v > 0.0) && r.context.charge.is_some()
}
fn ms2_fragments(r: &FoundationTrainingRecord) -> usize {
    r.fragments
        .iter()
        .filter(|f| f.channel < MS2_CHANNELS && f.intensity.is_finite())
        .count()
}
fn has_ms2(r: &FoundationTrainingRecord) -> bool {
    r.context.charge.is_some() && ms2_fragments(r) >= MIN_MS2_FRAGMENTS
}

/// Cosine on annotated fragments (channels 0..4) of one record; None if not computable.
fn ms2_cosine(r: &FoundationTrainingRecord, predicted: &[Vec<f32>]) -> Option<(f64, usize)> {
    let (mut dot, mut pp, mut tt, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for f in &r.fragments {
        if f.channel >= MS2_CHANNELS || !f.intensity.is_finite() {
            continue;
        }
        let p = f64::from(*predicted.get(f.cleavage_index)?.get(f.channel)?);
        if !p.is_finite() {
            return None;
        }
        let t = f64::from(f.intensity);
        dot += p * t;
        pp += p * p;
        tt += t * t;
        n += 1;
    }
    let d = (pp + 1e-8).sqrt() * (tt + 1e-8).sqrt();
    (n > 0 && d > 0.0).then(|| ((dot / d).clamp(-1.0, 1.0), n))
}

fn to_harmonized(
    objective: RetentionTimeObjective,
    p: f32,
    t: &FoundationRtHarmonizationTransform,
) -> Option<f64> {
    match objective {
        RetentionTimeObjective::Harmonized => Some(f64::from(p)),
        RetentionTimeObjective::Normalized | RetentionTimeObjective::IntrinsicAndObserved => {
            t.harmonize(p).map(f64::from)
        }
        RetentionTimeObjective::Observed => None,
    }
}

fn sha256_file(path: &str) -> Result<String> {
    let out = std::process::Command::new("sha256sum").arg(path).output()?;
    anyhow::ensure!(out.status.success(), "sha256sum failed for {path}");
    Ok(String::from_utf8(out.stdout)?
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string())
}

struct Loaded {
    corpus: redeem_properties::foundation::FoundationCorpus,
    benchmark: FoundationBenchmarkManifest,
    transforms: BTreeMap<String, FoundationRtHarmonizationTransform>,
}

fn load(args: &[String]) -> Result<Loaded> {
    let observed = sha256_file(&args[0])?;
    anyhow::ensure!(args[1] == observed, "run YAML SHA256 mismatch: {observed}");
    let run: Run = serde_yaml::from_str(&std::fs::read_to_string(&args[0])?)?;
    let transforms = run
        .corpus
        .sources
        .iter()
        .filter_map(|s| s.rt_harmonization.clone().map(|t| (s.id.clone(), t)))
        .collect();
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let c = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let b = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    anyhow::ensure!(c == args[2], "corpus fingerprint {c}");
    anyhow::ensure!(b == args[3], "benchmark fingerprint {b}");
    println!("#run_yaml_sha256\t{observed}\n#corpus_fingerprint\t{c}\n#benchmark_fingerprint\t{b}\n#test_labels_read\tNO");
    Ok(Loaded {
        corpus,
        benchmark,
        transforms,
    })
}

fn select(args: &[String]) -> Result<()> {
    anyhow::ensure!(
        args.len() == 6,
        "select needs RUN SHA CORPUS BENCH SEED PER_TASK"
    );
    let l = load(args)?;
    let seed: u64 = args[4].parse().context("SEED")?;
    let per_task: usize = args[5].parse().context("PER_TASK")?;
    let validation = l
        .benchmark
        .partition_indices(FoundationPartition::Validation);
    anyhow::ensure!(validation.len() > EXCLUDED_PREFIX, "Validation too small");
    // pool[source][task] -> record indices
    let mut pool: BTreeMap<String, [Vec<usize>; 3]> = BTreeMap::new();
    for &i in &validation[EXCLUDED_PREFIX..] {
        let r = &l.corpus.records[i];
        let source = l.corpus.provenance[i].source_id.clone();
        let entry = pool.entry(source.clone()).or_default();
        if has_rt(r) && l.transforms.contains_key(&source) {
            entry[0].push(i);
        }
        if has_ccs(r) {
            entry[1].push(i);
        }
        if has_ms2(r) {
            entry[2].push(i);
        }
    }
    println!(
        "#seed\t{seed}\n#per_task\t{per_task}\n#excluded_validation_prefix\t{EXCLUDED_PREFIX}"
    );
    println!("#pool\tsource\trt\tccs\tms2");
    for (s, p) in &pool {
        println!("#pool\t{s}\t{}\t{}\t{}", p[0].len(), p[1].len(), p[2].len());
    }
    println!("task\tsource\tcorpus_record_index");
    for (s, p) in &pool {
        // Independent stream per (source, task) so adding a source never changes others.
        for (t, name) in ["rt", "ccs", "ms2"].iter().enumerate() {
            let mut h = seed ^ (t as u64 + 1).wrapping_mul(0x9e37_79b9);
            for b in s.bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
            for i in sample(p[t].clone(), per_task, &mut Rng(h)) {
                println!("{name}\t{s}\t{i}");
            }
        }
    }
    println!("#F2_SCORECARD_SELECT=COMPLETE");
    Ok(())
}

enum Scored {
    Unified(FoundationModel),
    V0520(ResearchV0520Forward, Option<ResearchV0520Mobility>),
}

fn score(args: &[String]) -> Result<()> {
    anyhow::ensure!(
        args.len() >= 7,
        "score needs RUN SHA CORPUS BENCH COHORT V035 LABEL=CKPT..."
    );
    let l = load(args)?;
    let cohort_sha = sha256_file(&args[4])?;
    println!("#cohort_sha256\t{cohort_sha}");
    let validation: BTreeSet<usize> = l
        .benchmark
        .partition_indices(FoundationPartition::Validation)
        .into_iter()
        .collect();
    let mut tasks: BTreeMap<&'static str, Vec<usize>> = BTreeMap::new();
    let text = std::fs::read_to_string(&args[4])?;
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("task\t"))
    {
        let f: Vec<&str> = line.split('\t').collect();
        anyhow::ensure!(f.len() == 3, "bad cohort row {line}");
        let i: usize = f[2].parse()?;
        anyhow::ensure!(
            validation.contains(&i),
            "cohort record {i} is not Validation"
        );
        anyhow::ensure!(
            l.corpus.provenance[i].source_id == f[1],
            "source mismatch for {i}"
        );
        let task = match f[0] {
            "rt" => "rt",
            "ccs" => "ccs",
            "ms2" => "ms2",
            other => anyhow::bail!("unknown task {other}"),
        };
        tasks.entry(task).or_default().push(i);
    }
    let teacher = Path::new(&args[5]);
    println!("label\ttask\tsource\tcorpus_record_index\ttarget\tprediction\tn_fragments");
    for spec in &args[6..] {
        let (label, ckpt) = spec.split_once('=').context("LABEL=CKPT expected")?;
        let meta = std::fs::read_to_string(Path::new(ckpt).join("metadata.yaml"))?;
        let header = serde_yaml::from_str::<V0520Header>(&meta)
            .ok()
            .filter(|h| h.version == 520);
        let (model, objective) = if let Some(h) = header {
            let fwd = ResearchV0520Forward::load(Path::new(ckpt), Device::Cpu)?;
            fwd.verify_fingerprints(
                l.corpus.corpus_fingerprint,
                l.benchmark.manifest_fingerprint(),
            )?;
            let mob = ResearchV0520Mobility::load(Path::new(ckpt), teacher, Device::Cpu)?;
            (Scored::V0520(fwd, Some(mob)), h.rt_objective)
        } else {
            let m = FoundationModel::load(ckpt, Device::Cpu)?;
            let o = m.config().rt_objective;
            (Scored::Unified(m), o)
        };
        println!("#checkpoint\t{label}\t{ckpt}\t{objective:?}");
        for (task, indices) in &tasks {
            for chunk in indices.chunks(BATCH) {
                let records: Vec<_> = chunk.iter().map(|&i| l.corpus.records[i].clone()).collect();
                // (rt, ccs, ms2) per record
                let preds: Vec<(Option<f32>, Option<f32>, Option<Vec<Vec<f32>>>)> = match &model {
                    Scored::Unified(m) => m
                        .predict_records(&records)?
                        .into_iter()
                        .map(|p| (p.rt, p.ccs, p.ms2))
                        .collect(),
                    Scored::V0520(fwd, mob) => {
                        let f = fwd.infer(&records)?;
                        let c = if *task == "ccs" {
                            mob.as_ref()
                                .context("no v0.35 teacher")?
                                .infer(&records)?
                                .into_iter()
                                .map(|p| Some(p.ccs))
                                .collect()
                        } else {
                            vec![None; records.len()]
                        };
                        f.into_iter()
                            .zip(c)
                            .map(|(p, c)| (Some(p.rt), c, Some(p.ms2)))
                            .collect()
                    }
                };
                for ((&i, r), (rt, ccs, ms2)) in chunk.iter().zip(&records).zip(preds) {
                    let source = &l.corpus.provenance[i].source_id;
                    let (target, prediction, n) = match *task {
                        "rt" => (
                            f64::from(r.retention_time.harmonized.unwrap()),
                            rt.and_then(|p| to_harmonized(objective, p, &l.transforms[source])),
                            1,
                        ),
                        "ccs" => (f64::from(r.ccs.unwrap()), ccs.map(f64::from), 1),
                        _ => match ms2.as_deref().and_then(|p| ms2_cosine(r, p)) {
                            Some((c, n)) => (1.0, Some(c), n),
                            None => (1.0, None, 0),
                        },
                    };
                    let p = prediction.map_or("NA".to_string(), |v| format!("{v:.6}"));
                    println!("{label}\t{task}\t{source}\t{i}\t{target:.6}\t{p}\t{n}");
                }
            }
        }
    }
    println!("#F2_SCORECARD_SCORE=COMPLETE");
    Ok(())
}

fn self_test() {
    let a = sample((0..100).collect(), 10, &mut Rng(7));
    let b = sample((0..100).collect(), 10, &mut Rng(7));
    assert_eq!(a, b);
    assert_eq!(a.len(), 10);
    assert!(a.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(sample((0..3).collect(), 10, &mut Rng(1)), vec![0, 1, 2]);
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
    println!("F2_SCORECARD_SELF_TEST=PASS");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--self-test") => {
            self_test();
            Ok(())
        }
        Some("select") => select(&args[1..]),
        Some("score") => score(&args[1..]),
        _ => anyhow::bail!("usage: foundation_f2_scorecard select|score|--self-test ..."),
    }
}
