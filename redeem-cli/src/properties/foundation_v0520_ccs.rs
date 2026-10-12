//! Original v0.52 native CCS parity: frozen Validation prefix and frozen v0.35 teacher.
use crate::properties::inference::input::PropertyInferenceConfig;
use anyhow::{Context, Result, ensure};
use redeem_properties::foundation::{
    FoundationBenchmarkManifest, FoundationCorpusConfig, FoundationPartition,
    FoundationTrainingRecord, ResearchV0520Mobility, load_foundation_corpus,
};
use redeem_properties::utils::utils::get_device;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const PREFIX: usize = 2048;
const BATCH: usize = 32;
#[derive(Deserialize)]
struct PreparedRun {
    corpus: FoundationCorpusConfig,
    benchmark_manifest: PathBuf,
}

fn eligible(r: &FoundationTrainingRecord) -> bool {
    r.peptidoform.modifications.is_empty()
        && (7..=35).contains(&r.peptidoform.sequence.len())
        && r.ccs.is_some_and(|v| v.is_finite() && v > 0.0)
        && r.context
            .ion_mobility
            .is_some_and(|v| v.is_finite() && v > 0.0)
        && r.context.charge.is_some_and(|v| v > 0)
        && r.context
            .precursor_mz
            .is_some_and(|v| v.is_finite() && v > 0.0)
}

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    ensure!(
        config.foundation_partition == "validation",
        "historical CCS is Validation-only"
    );
    ensure!(
        config.foundation_max_records == Some(PREFIX),
        "requires frozen 2048-record Validation prefix"
    );
    ensure!(
        config.batch_size == BATCH,
        "historical mobility parity requires batch_size=32"
    );
    ensure!(
        !config.foundation_spectra_only,
        "spectra_only unsupported for CCS parity"
    );
    let teacher = config
        .foundation_v0350_teacher_checkpoint
        .as_ref()
        .context("v0520_native_ccs requires foundation_v0350_teacher_checkpoint")?;
    let output = Path::new(&config.output_file);
    ensure!(!output.exists(), "CCS parity output already exists");
    let run: PreparedRun = serde_yaml::from_str(
        &fs::read_to_string(&config.inference_data)
            .with_context(|| format!("read {}", config.inference_data))?,
    )?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let validation = benchmark.partition_indices(FoundationPartition::Validation);
    ensure!(
        validation.len() >= PREFIX,
        "insufficient Validation records"
    );
    let indices: Vec<usize> = validation[..PREFIX]
        .iter()
        .copied()
        .filter(|&index| eligible(&corpus.records[index]))
        .collect();
    ensure!(
        !indices.is_empty(),
        "no finite mobility/CCS targets on Validation prefix"
    );
    let model = ResearchV0520Mobility::load(
        Path::new(&config.model_path),
        Path::new(teacher),
        get_device(&config.device)?,
    )?;
    model.verify_fingerprints(corpus.corpus_fingerprint, benchmark.manifest_fingerprint())?;
    let mut file = BufWriter::new(File::create(output)?);
    writeln!(
        file,
        "corpus_record_index\tsequence\tcharge\tprecursor_mz\ttarget_native_mobility\ttarget_ccs\tpredicted_native_mobility\tpredicted_ccs\tccs_factor"
    )?;
    for chunk in indices.chunks(BATCH) {
        let records: Vec<_> = chunk
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect();
        let predictions = model.infer(&records)?;
        for ((&index, record), pred) in chunk.iter().zip(&records).zip(predictions) {
            writeln!(
                file,
                "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
                index,
                record.peptidoform.sequence,
                record.context.charge.unwrap(),
                record.context.precursor_mz.unwrap(),
                record.context.ion_mobility.unwrap(),
                record.ccs.unwrap(),
                pred.native_mobility,
                pred.ccs,
                pred.factor
            )?;
        }
    }
    file.flush()?;
    println!("foundation_inference=PASS");
    println!("foundation_backend=v0520_native_ccs");
    println!("partition=validation");
    println!("ccs_records={}", indices.len());
    println!("output={}", output.display());
    println!("historical_test_consumed=NO");
    println!("train_holdout_read_by_this_inference=NO");
    Ok(())
}
