use anyhow::{Context, Result};
use candle_core::Device;
use csv::StringRecord;
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationBenchmarkManifest, FoundationCorpusConfig,
    FoundationModificationSite, FoundationPartition, FoundationPredictor,
    FoundationPredictorConfig, FoundationTrainingRecord,
};
use redeem_properties::models::model_interface::{
    PredictionInput, PredictionModification, PredictionModificationSite,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_LIMIT: usize = 32;
const RT_TOLERANCE: f64 = 1.0e-3;
const CCS_TOLERANCE: f64 = 2.0e-3;
const MS2_TOLERANCE: f64 = 1.0e-4;

#[derive(Debug, Deserialize)]
struct ParityRunConfig {
    corpus: FoundationCorpusConfig,
    benchmark_manifest: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceScalar {
    record_index: usize,
    expected: f32,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceFragment {
    record_index: usize,
    cleavage_index: usize,
    channel: usize,
    expected: f32,
}

#[derive(Debug, Clone, Copy)]
struct ComparisonSummary {
    comparisons: usize,
    max_abs_delta: f64,
    tolerance: f64,
}

impl ComparisonSummary {
    fn new(tolerance: f64) -> Self {
        Self {
            comparisons: 0,
            max_abs_delta: 0.0,
            tolerance,
        }
    }

    fn observe(&mut self, actual: f32, expected: f32) -> Result<()> {
        let delta = f64::from((actual - expected).abs());
        if !delta.is_finite() {
            anyhow::bail!("non-finite prediction delta");
        }
        self.comparisons += 1;
        self.max_abs_delta = self.max_abs_delta.max(delta);
        Ok(())
    }

    fn validate(&self, label: &str) -> Result<()> {
        if self.comparisons == 0 {
            anyhow::bail!("{label} parity made no comparisons");
        }
        if self.max_abs_delta > self.tolerance {
            anyhow::bail!(
                "{label} checkpoint parity failed: max_abs_delta={} tolerance={}",
                self.max_abs_delta,
                self.tolerance
            );
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(7..=8).contains(&args.len()) {
        anyhow::bail!(
            "usage: foundation_predict_checkpoint_smoke RUN.yaml RT_MS2_CHECKPOINT CCS_CHECKPOINT FROZEN_RT_MS2_PRECURSORS.tsv FROZEN_RT_MS2_FRAGMENTS.tsv FROZEN_CCS.tsv OUTPUT.tsv [limit=32]"
        );
    }

    let run_yaml = PathBuf::from(&args[0]);
    let rt_ms2_checkpoint = PathBuf::from(&args[1]);
    let ccs_checkpoint = PathBuf::from(&args[2]);
    let rt_reference_path = PathBuf::from(&args[3]);
    let ms2_reference_path = PathBuf::from(&args[4]);
    let ccs_reference_path = PathBuf::from(&args[5]);
    let output_path = PathBuf::from(&args[6]);
    let limit = args
        .get(7)
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("parse parity limit")?
        .unwrap_or(DEFAULT_LIMIT);
    if limit == 0 {
        anyhow::bail!("parity limit must be positive");
    }
    if output_path.exists() {
        anyhow::bail!("parity output must be fresh: {}", output_path.display());
    }

    let all_ms2_references = read_ms2_references(&ms2_reference_path)?;
    let ms2_indices = all_ms2_references
        .iter()
        .map(|reference| reference.record_index)
        .collect::<BTreeSet<_>>();
    let rt_references = read_rt_references(&rt_reference_path, &ms2_indices, limit)?;
    let rt_indices = rt_references
        .iter()
        .map(|reference| reference.record_index)
        .collect::<BTreeSet<_>>();
    let ms2_references = all_ms2_references
        .into_iter()
        .filter(|reference| rt_indices.contains(&reference.record_index))
        .collect::<Vec<_>>();
    let ccs_references = read_ccs_references(&ccs_reference_path, limit)?;

    let run = read_run_config(&run_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark =
        FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest).with_context(|| {
            format!(
                "read benchmark manifest {}",
                run.benchmark_manifest.display()
            )
        })?;
    benchmark.validate_against_records(&corpus.records)?;

    let mut selected = BTreeSet::new();
    selected.extend(rt_indices.iter().copied());
    selected.extend(
        ccs_references
            .iter()
            .map(|reference| reference.record_index),
    );
    validate_reference_partitions(&benchmark, &selected)?;

    let selected_indices = selected.into_iter().collect::<Vec<_>>();
    let inputs = selected_indices
        .iter()
        .map(|&index| {
            let record = corpus.records.get(index).with_context(|| {
                format!("frozen parity record index {index} is outside loaded corpus")
            })?;
            prediction_input(record)
        })
        .collect::<Result<Vec<_>>>()?;

    let predictor = FoundationPredictor::load(
        FoundationPredictorConfig::new(rt_ms2_checkpoint, ccs_checkpoint),
        Device::Cpu,
    )?;
    let outputs = predictor.predict_batch(&inputs)?;
    if outputs.len() != selected_indices.len() {
        anyhow::bail!(
            "foundation predictor returned {} rows for {} parity inputs",
            outputs.len(),
            selected_indices.len()
        );
    }
    let predictions = selected_indices
        .iter()
        .copied()
        .zip(outputs)
        .collect::<BTreeMap<_, _>>();

    let mut rt_summary = ComparisonSummary::new(RT_TOLERANCE);
    for reference in &rt_references {
        let actual = predictions
            .get(&reference.record_index)
            .and_then(|prediction| prediction.rt)
            .context("missing RT prediction")?;
        rt_summary.observe(actual, reference.expected)?;
    }

    let mut ms2_summary = ComparisonSummary::new(MS2_TOLERANCE);
    for reference in &ms2_references {
        let actual = predictions
            .get(&reference.record_index)
            .and_then(|prediction| prediction.ms2.as_ref())
            .and_then(|matrix| matrix.get(reference.cleavage_index))
            .and_then(|row| row.get(reference.channel))
            .copied()
            .with_context(|| {
                format!(
                    "missing MS2 prediction record={} cleavage={} channel={}",
                    reference.record_index, reference.cleavage_index, reference.channel
                )
            })?;
        ms2_summary.observe(actual, reference.expected)?;
    }

    let mut ccs_summary = ComparisonSummary::new(CCS_TOLERANCE);
    for reference in &ccs_references {
        let actual = predictions
            .get(&reference.record_index)
            .and_then(|prediction| prediction.ccs)
            .context("missing CCS prediction")?;
        ccs_summary.observe(actual, reference.expected)?;
    }

    rt_summary.validate("RT")?;
    ms2_summary.validate("MS2")?;
    ccs_summary.validate("CCS")?;

    write_report(&output_path, rt_summary, ms2_summary, ccs_summary)?;
    println!("foundation_predict_checkpoint_parity=PASS");
    println!("rt_comparisons={}", rt_summary.comparisons);
    println!("rt_max_abs_delta={:.9}", rt_summary.max_abs_delta);
    println!("ms2_comparisons={}", ms2_summary.comparisons);
    println!("ms2_max_abs_delta={:.9}", ms2_summary.max_abs_delta);
    println!("ccs_comparisons={}", ccs_summary.comparisons);
    println!("ccs_max_abs_delta={:.9}", ccs_summary.max_abs_delta);
    println!("historical_validation_reference_reused=YES");
    println!("historical_test_compared=NO");
    println!("train_holdout_compared=NO");
    println!("output={}", output_path.display());
    Ok(())
}

fn read_run_config(path: &Path) -> Result<ParityRunConfig> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read foundation run config {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("parse foundation run config {}", path.display()))
}

fn read_rt_references(
    path: &Path,
    ms2_indices: &BTreeSet<usize>,
    limit: usize,
) -> Result<Vec<ReferenceScalar>> {
    let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let headers = reader.headers()?.clone();
    let record_index = column(&headers, "record_index")?;
    let foundation_rt = column(&headers, "foundation_rt")?;
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for row in reader.records() {
        let row = row?;
        let index = parse_usize(&row, record_index, "record_index")?;
        let Some(expected) = parse_optional_f32(&row, foundation_rt)? else {
            continue;
        };
        if !ms2_indices.contains(&index) {
            continue;
        }
        if seen.insert(index) {
            out.push(ReferenceScalar {
                record_index: index,
                expected,
            });
            if out.len() == limit {
                break;
            }
        }
    }
    if out.is_empty() {
        anyhow::bail!(
            "frozen RT reference contains no usable rows: {}",
            path.display()
        );
    }
    Ok(out)
}

fn read_ms2_references(path: &Path) -> Result<Vec<ReferenceFragment>> {
    let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let headers = reader.headers()?.clone();
    let record_index = column(&headers, "record_index")?;
    let cleavage_index = column(&headers, "cleavage_index")?;
    let channel = column(&headers, "channel")?;
    let foundation_intensity = column(&headers, "foundation_intensity")?;
    let mut out = Vec::new();
    for row in reader.records() {
        let row = row?;
        let Some(expected) = parse_optional_f32(&row, foundation_intensity)? else {
            continue;
        };
        out.push(ReferenceFragment {
            record_index: parse_usize(&row, record_index, "record_index")?,
            cleavage_index: parse_usize(&row, cleavage_index, "cleavage_index")?,
            channel: parse_usize(&row, channel, "channel")?,
            expected,
        });
    }
    if out.is_empty() {
        anyhow::bail!(
            "frozen MS2 reference contains no usable rows: {}",
            path.display()
        );
    }
    Ok(out)
}

fn read_ccs_references(path: &Path, limit: usize) -> Result<Vec<ReferenceScalar>> {
    let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let headers = reader.headers()?.clone();
    let record_index = column(&headers, "record_index")?;
    let predicted_ccs = column(&headers, "predicted_ccs")?;
    let mut out = Vec::new();
    for row in reader.records() {
        let row = row?;
        let Some(expected) = parse_optional_f32(&row, predicted_ccs)? else {
            continue;
        };
        out.push(ReferenceScalar {
            record_index: parse_usize(&row, record_index, "record_index")?,
            expected,
        });
        if out.len() == limit {
            break;
        }
    }
    if out.is_empty() {
        anyhow::bail!(
            "frozen CCS reference contains no usable rows: {}",
            path.display()
        );
    }
    Ok(out)
}

fn validate_reference_partitions(
    benchmark: &FoundationBenchmarkManifest,
    selected: &BTreeSet<usize>,
) -> Result<()> {
    let partitions = benchmark
        .entries
        .iter()
        .map(|entry| (entry.record_index, entry.partition))
        .collect::<BTreeMap<_, _>>();
    for index in selected {
        match partitions.get(index) {
            Some(FoundationPartition::Validation) => {}
            Some(partition) => {
                anyhow::bail!("frozen parity record {index} is {partition:?}, expected Validation")
            }
            None => anyhow::bail!("frozen parity record {index} is absent from benchmark manifest"),
        }
    }
    Ok(())
}

fn prediction_input(record: &FoundationTrainingRecord) -> Result<PredictionInput> {
    let modifications = record
        .peptidoform
        .modifications
        .iter()
        .map(|modification| PredictionModification {
            site: match modification.site {
                FoundationModificationSite::Residue(index) => {
                    PredictionModificationSite::Residue(index)
                }
                FoundationModificationSite::NTerm => PredictionModificationSite::NTerm,
                FoundationModificationSite::CTerm => PredictionModificationSite::CTerm,
            },
            mass_delta: modification.mass_delta,
            unimod_id: modification.unimod_id,
        })
        .collect();
    Ok(PredictionInput {
        sequence: record.peptidoform.sequence.clone(),
        modifications,
        charge: record.context.charge,
        precursor_mz: record.context.precursor_mz,
        nce: record.context.nce,
        instrument_id: record.context.instrument_id,
        instrument_name: record.context.instrument_name.clone(),
    })
}

fn write_report(
    path: &Path,
    rt: ComparisonSummary,
    ms2: ComparisonSummary,
    ccs: ComparisonSummary,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = format!(
        "property\tcomparisons\tmax_abs_delta\ttolerance\tstatus\nRT\t{}\t{:.9}\t{:.9}\tPASS\nMS2\t{}\t{:.9}\t{:.9}\tPASS\nCCS\t{}\t{:.9}\t{:.9}\tPASS\n",
        rt.comparisons,
        rt.max_abs_delta,
        rt.tolerance,
        ms2.comparisons,
        ms2.max_abs_delta,
        ms2.tolerance,
        ccs.comparisons,
        ccs.max_abs_delta,
        ccs.tolerance,
    );
    fs::write(path, text)?;
    Ok(())
}

fn column(headers: &StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|header| header == name)
        .with_context(|| format!("missing TSV column {name:?}"))
}

fn parse_usize(row: &StringRecord, index: usize, name: &str) -> Result<usize> {
    row.get(index)
        .with_context(|| format!("missing TSV field {name:?}"))?
        .parse::<usize>()
        .with_context(|| format!("parse TSV field {name:?}"))
}

fn parse_optional_f32(row: &StringRecord, index: usize) -> Result<Option<f32>> {
    let value = row.get(index).context("missing TSV numeric field")?.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("NA") {
        return Ok(None);
    }
    let parsed = value.parse::<f32>().context("parse TSV floating value")?;
    if !parsed.is_finite() {
        anyhow::bail!("TSV reference contains non-finite floating value {value:?}");
    }
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_summary_rejects_delta_above_tolerance() {
        let mut summary = ComparisonSummary::new(0.01);
        summary.observe(1.02, 1.0).unwrap();
        assert!(summary.validate("test").is_err());
    }

    #[test]
    fn prediction_input_preserves_acquisition_context() {
        let record = FoundationTrainingRecord {
            peptidoform: redeem_properties::foundation::PeptidoformInput {
                sequence: "PEPTIDE".into(),
                modifications: Vec::new(),
            },
            retention_time: Default::default(),
            ccs: None,
            fragments: Vec::new(),
            observed_spectrum_peaks: Vec::new(),
            context: redeem_properties::foundation::TrainingContext {
                charge: Some(2),
                precursor_mz: Some(400.2),
                nce: Some(30.0),
                instrument_id: Some(3),
                instrument_name: Some("instrument".into()),
                ion_mobility: None,
                gradient_seconds: None,
            },
            run_id: None,
        };
        let input = prediction_input(&record).unwrap();
        assert_eq!(input.charge, Some(2));
        assert_eq!(input.instrument_id, Some(3));
        assert_eq!(input.instrument_name.as_deref(), Some("instrument"));
    }
}
