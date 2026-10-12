//! Opt-in direct inference from the original trained v0.52 checkpoint.
//! Only the frozen 2048-record Validation prefix is exposed until numeric parity
//! with the separately built historical exporter is confirmed.
use crate::properties::inference::input::PropertyInferenceConfig;
use anyhow::{Context, Result, ensure};
use redeem_properties::foundation::{
    FoundationBenchmarkManifest, FoundationCorpusConfig, FoundationPartition,
    FoundationTrainingRecord, ResearchV0520Forward, RetentionTimeObjective, load_foundation_corpus,
};
use redeem_properties::utils::utils::get_device;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
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

#[derive(Deserialize)]
struct MetadataRt {
    rt_objective: RetentionTimeObjective,
}

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    ensure!(
        config.foundation_partition == "validation",
        "historical native backend is Validation-only"
    );
    ensure!(
        config.foundation_max_records == Some(PREFIX),
        "historical native backend requires foundation_max_records=2048"
    );
    ensure!(
        !config.foundation_spectra_only,
        "historical native backend does not support spectra_only"
    );
    ensure!(
        config.batch_size == BATCH,
        "historical native Validation parity requires batch_size=32"
    );
    let checkpoint = Path::new(&config.model_path);
    let output_path = Path::new(&config.output_file);
    let fragment_path = PathBuf::from(format!("{}.annotated_fragments.tsv", config.output_file));
    ensure!(
        !output_path.exists() && !fragment_path.exists(),
        "native reference output already exists"
    );
    let run: PreparedRun = serde_yaml::from_str(
        &fs::read_to_string(&config.inference_data)
            .with_context(|| format!("read prepared run {}", config.inference_data))?,
    )?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let metadata: MetadataRt =
        serde_yaml::from_str(&fs::read_to_string(checkpoint.join("metadata.yaml"))?)?;
    let device = get_device(&config.device)?;
    let model = ResearchV0520Forward::load(checkpoint, device)?;
    model.verify_fingerprints(corpus.corpus_fingerprint, benchmark.manifest_fingerprint())?;
    let allowed = benchmark.partition_indices(FoundationPartition::Validation);
    ensure!(
        allowed.len() >= PREFIX,
        "fewer than 2048 Validation records"
    );
    let ordinal: HashMap<usize, usize> = allowed
        .iter()
        .take(PREFIX)
        .enumerate()
        .map(|(ord, &idx)| (idx, ord))
        .collect();
    let mut rt_indices = BTreeSet::new();
    let mut ms2_indices = BTreeSet::new();
    for &index in allowed.iter().take(PREFIX) {
        let record = &corpus.records[index];
        if !eligible(record) {
            continue;
        }
        if rt_target(record, metadata.rt_objective).is_some() {
            rt_indices.insert(index);
        }
        if record.context.charge.is_some()
            && record
                .fragments
                .iter()
                .filter(|fragment| fragment.channel < 4 && fragment.intensity.is_finite())
                .count()
                >= 8
        {
            ms2_indices.insert(index);
        }
    }
    ensure!(
        !rt_indices.is_empty() && !ms2_indices.is_empty(),
        "no eligible Validation RT or MS2 records"
    );
    let selected: BTreeSet<usize> = rt_indices.union(&ms2_indices).copied().collect();

    let mut results = Vec::with_capacity(selected.len());
    let indices: Vec<usize> = selected.iter().copied().collect();
    for chunk in indices.chunks(BATCH) {
        let batch: Vec<FoundationTrainingRecord> = chunk
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect();
        let predicted = model.infer(&batch)?;
        for (&index, prediction) in chunk.iter().zip(predicted) {
            results.push((index, prediction));
        }
    }

    let mut rt_file = BufWriter::new(File::create(output_path)?);
    writeln!(
        rt_file,
        "record_index\tsequence\tcharge\tpredicted_rt\tpredicted_ccs\tpredicted_ms2\tinverse_mean_nll\tgeneration_succeeded\tgenerated_sequence\tgenerated_modifications\tcorpus_record_index\tms2_annotation_scope"
    )?;
    let mut fragments_file = BufWriter::new(File::create(&fragment_path)?);
    writeln!(
        fragments_file,
        "record_index\tsequence\tcleavage_index\tchannel\tcharged_frag_type\tfragment_number\tproduct_mz\ttarget_intensity\tfoundation_intensity"
    )?;
    for (index, prediction) in results {
        let record = &corpus.records[index];
        let is_ms2 = ms2_indices.contains(&index);
        let rt = if rt_indices.contains(&index) {
            format!("{:.8}", prediction.rt)
        } else {
            String::new()
        };
        writeln!(
            rt_file,
            "{}\t{}\t{}\t{}\t\t\t\tfalse\t\t\t{}\t{}",
            ordinal[&index],
            record.peptidoform.sequence,
            record
                .context
                .charge
                .map(|c| c.to_string())
                .unwrap_or_default(),
            rt,
            index,
            if is_ms2 {
                "annotated_fragments_only"
            } else {
                "none"
            }
        )?;
        if is_ms2 {
            for f in record
                .fragments
                .iter()
                .filter(|f| f.channel < 4 && f.intensity.is_finite())
            {
                let predicted = prediction
                    .ms2
                    .get(f.cleavage_index)
                    .and_then(|v| v.get(f.channel))
                    .copied();
                if let Some(value) = predicted {
                    let (frag_type, number) = fragment_identity(
                        f.channel,
                        f.cleavage_index,
                        record.peptidoform.sequence.len(),
                    );
                    writeln!(
                        fragments_file,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}",
                        index,
                        record.peptidoform.sequence,
                        f.cleavage_index,
                        f.channel,
                        frag_type,
                        number,
                        f.product_mz
                            .filter(|mz| mz.is_finite())
                            .map(|mz| format!("{mz:.8}"))
                            .unwrap_or_else(|| "NA".into()),
                        f.intensity,
                        value
                    )?;
                }
            }
        }
    }
    rt_file.flush()?;
    fragments_file.flush()?;
    println!("foundation_inference=PASS");
    println!("foundation_backend=v0520_native");
    println!("partition=validation");
    println!("records={}", selected.len());
    println!("rt_records={}", rt_indices.len());
    println!("ms2_precursors={}", ms2_indices.len());
    println!("output={}", output_path.display());
    println!("annotated_ms2={}", fragment_path.display());
    println!("historical_test_consumed=NO");
    println!("train_holdout_read_by_this_inference=NO");
    Ok(())
}

fn eligible(record: &FoundationTrainingRecord) -> bool {
    record.peptidoform.modifications.is_empty()
        && (7..=35).contains(&record.peptidoform.sequence.len())
}

fn rt_target(record: &FoundationTrainingRecord, objective: RetentionTimeObjective) -> Option<f32> {
    match objective {
        RetentionTimeObjective::Normalized | RetentionTimeObjective::IntrinsicAndObserved => {
            record.retention_time.normalized
        }
        RetentionTimeObjective::Harmonized => record.retention_time.harmonized,
        RetentionTimeObjective::Observed => record.retention_time.observed_seconds,
    }
    .filter(|v| v.is_finite())
}

fn fragment_identity(channel: usize, cleavage: usize, len: usize) -> (&'static str, usize) {
    match channel {
        0 => ("b_z1", cleavage + 1),
        1 => ("b_z2", cleavage + 1),
        2 => ("y_z1", len.saturating_sub(cleavage + 1)),
        3 => ("y_z2", len.saturating_sub(cleavage + 1)),
        _ => unreachable!("channels are filtered before this function"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_fragment_channel_mapping_matches_original_v0520() {
        assert_eq!(fragment_identity(0, 2, 11), ("b_z1", 3));
        assert_eq!(fragment_identity(2, 2, 11), ("y_z1", 8));
        assert_eq!(fragment_identity(3, 9, 11), ("y_z2", 1));
    }
}
