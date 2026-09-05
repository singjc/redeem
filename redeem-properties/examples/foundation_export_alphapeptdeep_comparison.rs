//! Export a small, deterministic held-out comparison set for ReDeeM vs AlphaPeptDeep.
//!
//! Only TRAIN and VALIDATION benchmark partitions are accessed.  VALIDATION supplies the
//! descriptive held-out RT/CCS/MS2 comparison peptides; a separate TRAIN-only RT calibration
//! sample is exported so AlphaPeptDeep's model-native RT coordinate can be mapped to the
//! ReDeeM target coordinate without fitting on validation.  TEST is never opened.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig, FoundationPartition,
    FoundationTargetNormalizationConfig, FoundationTrainingRecord, PeptideFoundationUnifiedModel,
    RetentionTimeObjective,
};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const VERSION: &str = "v0.16.0-alphapeptdeep-comparison-export";
const DEFAULT_PER_TASK: usize = 12;
const DEFAULT_RT_CALIBRATION: usize = 256;
const DEFAULT_SEED: u64 = 20_260_916;
const BATCH_SIZE: usize = 32;

#[derive(Debug, Deserialize)]
struct UnifiedMetadata {
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: redeem_properties::foundation::FoundationDiffusionConfig,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
}

#[derive(Debug, Clone)]
struct Predictions {
    rt: f32,
    ccs: f32,
    ms2: Vec<Vec<f32>>,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(3..=6).contains(&args.len()) {
        anyhow::bail!(
            "usage: foundation_export_alphapeptdeep_comparison RUN.yaml UNIFIED_CHECKPOINT OUTPUT_DIR [validation_per_task=12] [rt_train_calibration=256] [seed=20260916]"
        );
    }
    let training_yaml = PathBuf::from(&args[0]);
    let checkpoint = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    let per_task = parse_or(&args, 3, DEFAULT_PER_TASK)?;
    let rt_calibration_n = parse_or(&args, 4, DEFAULT_RT_CALIBRATION)?;
    let seed = parse_or(&args, 5, DEFAULT_SEED)?;
    if per_task == 0 || rt_calibration_n == 0 {
        anyhow::bail!("comparison sample sizes must be positive");
    }

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;

    let metadata_path = checkpoint.join("metadata.yaml");
    let metadata: UnifiedMetadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("read unified metadata {metadata_path:?}"))?,
    )?;

    let validation_allowed = benchmark.partition_indices(FoundationPartition::Validation);
    let train_allowed = benchmark.partition_indices(FoundationPartition::Train);

    let rt_validation = select_records(
        &validation_allowed,
        &corpus.records,
        per_task,
        seed ^ 0x5254,
        |record| rt_target(record, metadata.rt_objective).is_some(),
    );
    let ccs_validation = select_records(
        &validation_allowed,
        &corpus.records,
        per_task,
        seed ^ 0x4343_53,
        |record| record.ccs.is_some() && record.context.charge.is_some(),
    );
    let ms2_validation = select_records(
        &validation_allowed,
        &corpus.records,
        per_task,
        seed ^ 0x4d53_32,
        |record| {
            record.context.charge.is_some()
                && record
                    .fragments
                    .iter()
                    .filter(|fragment| fragment.channel < 4 && fragment.intensity.is_finite())
                    .count()
                    >= 8
        },
    );
    let rt_train = select_records(
        &train_allowed,
        &corpus.records,
        rt_calibration_n,
        seed ^ 0x4341_4c52_54,
        |record| rt_target(record, metadata.rt_objective).is_some(),
    );

    if rt_validation.len() < per_task
        || ccs_validation.len() < per_task
        || ms2_validation.len() < per_task
    {
        anyhow::bail!(
            "insufficient eligible unmodified VALIDATION records: RT {} CCS {} MS2 {} requested {} each",
            rt_validation.len(), ccs_validation.len(), ms2_validation.len(), per_task
        );
    }
    if rt_train.len() < rt_calibration_n {
        anyhow::bail!(
            "insufficient eligible unmodified TRAIN RT calibration records: {} requested {}",
            rt_train.len(),
            rt_calibration_n
        );
    }

    let mut validation_union = BTreeSet::new();
    validation_union.extend(rt_validation.iter().copied());
    validation_union.extend(ccs_validation.iter().copied());
    validation_union.extend(ms2_validation.iter().copied());
    let validation_union = validation_union.into_iter().collect::<Vec<_>>();

    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationUnifiedModel::new(
        metadata.forward_config.clone(),
        metadata.inverse_config.clone(),
        vb,
    )?;
    varmap
        .load(checkpoint.join("model.safetensors"))
        .with_context(|| format!("load unified checkpoint {checkpoint:?}"))?;
    let collator = FoundationCollator::new(
        metadata.forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;

    let predictions = predict_records(
        &validation_union,
        &corpus.records,
        &model,
        &collator,
        &metadata.target_normalization,
        seed,
        &device,
    )?;

    fs::create_dir_all(&output_dir)?;
    let precursor_path = output_dir.join("foundation_comparison_precursors.tsv");
    let fragment_path = output_dir.join("foundation_comparison_fragments.tsv");
    let calibration_path = output_dir.join("foundation_comparison_rt_train_calibration.tsv");

    let rt_set: BTreeSet<usize> = rt_validation.iter().copied().collect();
    let ccs_set: BTreeSet<usize> = ccs_validation.iter().copied().collect();
    let ms2_set: BTreeSet<usize> = ms2_validation.iter().copied().collect();

    let mut precursor = BufWriter::new(File::create(&precursor_path)?);
    writeln!(
        precursor,
        "record_index\tpartition\tsource_id\tsequence\tmods\tmod_sites\tcharge\tnce\tinstrument\tselected_rt\tselected_ccs\tselected_ms2\ttarget_rt\tfoundation_rt\ttarget_ccs\tfoundation_ccs"
    )?;
    for &index in &validation_union {
        let record = &corpus.records[index];
        let pred = predictions
            .get(&index)
            .context("missing foundation prediction")?;
        writeln!(
            precursor,
            "{}\tVALIDATION\t{}\t{}\t\t\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{:.8}",
            index,
            escape_tsv(&corpus.provenance[index].source_id),
            record.peptidoform.sequence,
            option_i32(record.context.charge),
            option_f32(record.context.nce),
            escape_tsv(record.context.instrument_name.as_deref().unwrap_or("")),
            yes_no(rt_set.contains(&index)),
            yes_no(ccs_set.contains(&index)),
            yes_no(ms2_set.contains(&index)),
            option_f32(rt_target(record, metadata.rt_objective)),
            pred.rt,
            option_f32(record.ccs),
            pred.ccs,
        )?;
    }
    precursor.flush()?;

    let mut fragments = BufWriter::new(File::create(&fragment_path)?);
    writeln!(
        fragments,
        "record_index\tsequence\tcleavage_index\tchannel\tcharged_frag_type\tfragment_number\ttarget_intensity\tfoundation_intensity"
    )?;
    for &index in &ms2_validation {
        let record = &corpus.records[index];
        let pred = predictions
            .get(&index)
            .context("missing foundation MS2 prediction")?;
        for fragment in &record.fragments {
            if fragment.channel >= 4 || !fragment.intensity.is_finite() {
                continue;
            }
            let Some(predicted) = pred
                .ms2
                .get(fragment.cleavage_index)
                .and_then(|row| row.get(fragment.channel))
                .copied()
            else {
                continue;
            };
            let (frag_type, number) = fragment_identity(
                fragment.channel,
                fragment.cleavage_index,
                record.peptidoform.sequence.len(),
            )?;
            writeln!(
                fragments,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}",
                index,
                record.peptidoform.sequence,
                fragment.cleavage_index,
                fragment.channel,
                frag_type,
                number,
                fragment.intensity,
                predicted,
            )?;
        }
    }
    fragments.flush()?;

    let mut calibration = BufWriter::new(File::create(&calibration_path)?);
    writeln!(
        calibration,
        "record_index\tpartition\tsource_id\tsequence\tmods\tmod_sites\tcharge\tnce\tinstrument\ttarget_rt"
    )?;
    for &index in &rt_train {
        let record = &corpus.records[index];
        writeln!(
            calibration,
            "{}\tTRAIN\t{}\t{}\t\t\t{}\t{}\t{}\t{}",
            index,
            escape_tsv(&corpus.provenance[index].source_id),
            record.peptidoform.sequence,
            option_i32(record.context.charge),
            option_f32(record.context.nce),
            escape_tsv(record.context.instrument_name.as_deref().unwrap_or("")),
            option_f32(rt_target(record, metadata.rt_objective)),
        )?;
    }
    calibration.flush()?;

    println!("comparison_export_version\t{VERSION}");
    println!("partition\tVALIDATION");
    println!("test_partition_consumed\tNO");
    println!("rt_objective\t{:?}", metadata.rt_objective);
    println!("validation_rt_records\t{}", rt_validation.len());
    println!("validation_ccs_records\t{}", ccs_validation.len());
    println!("validation_ms2_records\t{}", ms2_validation.len());
    println!("validation_union_records\t{}", validation_union.len());
    println!("rt_train_calibration_records\t{}", rt_train.len());
    println!("seed\t{seed}");
    println!(
        "foundation_comparison_precursors\t{}",
        precursor_path.display()
    );
    println!(
        "foundation_comparison_fragments\t{}",
        fragment_path.display()
    );
    println!(
        "foundation_comparison_rt_train_calibration\t{}",
        calibration_path.display()
    );
    Ok(())
}

fn select_records<F>(
    partition_indices: &[usize],
    records: &[FoundationTrainingRecord],
    n: usize,
    seed: u64,
    eligible: F,
) -> Vec<usize>
where
    F: Fn(&FoundationTrainingRecord) -> bool,
{
    let mut candidates = partition_indices
        .iter()
        .copied()
        .filter(|&index| {
            records.get(index).is_some_and(|record| {
                record.peptidoform.modifications.is_empty()
                    && record.peptidoform.sequence.len() >= 7
                    && record.peptidoform.sequence.len() <= 35
                    && eligible(record)
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|&index| mix64(seed ^ index as u64));
    candidates.truncate(n.min(candidates.len()));
    candidates
}

#[allow(clippy::too_many_arguments)]
fn predict_records(
    indices: &[usize],
    records: &[FoundationTrainingRecord],
    model: &PeptideFoundationUnifiedModel,
    collator: &FoundationCollator,
    normalization: &FoundationTargetNormalizationConfig,
    seed: u64,
    device: &Device,
) -> Result<HashMap<usize, Predictions>> {
    let mut out = HashMap::new();
    for chunk in indices.chunks(BATCH_SIZE) {
        let batch_records = chunk
            .iter()
            .map(|&index| records[index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&batch_records, device, seed ^ chunk[0] as u64)?;
        let mut prediction = model
            .forward()
            .forward_t(&batch.input, &batch.context, false)?;
        prediction.rt = normalization.rt.denormalize_tensor(&prediction.rt)?;
        prediction.ccs = normalization.ccs.denormalize_tensor(&prediction.ccs)?;
        let rt = prediction.rt.squeeze(1)?.to_vec1::<f32>()?;
        let ccs = prediction.ccs.squeeze(1)?.to_vec1::<f32>()?;
        let ms2 = prediction.ms2.to_vec3::<f32>()?;
        for local in 0..chunk.len() {
            out.insert(
                chunk[local],
                Predictions {
                    rt: rt[local],
                    ccs: ccs[local],
                    ms2: ms2[local].clone(),
                },
            );
        }
    }
    Ok(out)
}

fn rt_target(record: &FoundationTrainingRecord, objective: RetentionTimeObjective) -> Option<f32> {
    match objective {
        RetentionTimeObjective::Normalized => record.retention_time.normalized,
        RetentionTimeObjective::Harmonized => record.retention_time.harmonized,
        RetentionTimeObjective::Observed => record.retention_time.observed_seconds,
        RetentionTimeObjective::IntrinsicAndObserved => record.retention_time.normalized,
    }
    .filter(|value| value.is_finite())
}

fn fragment_identity(
    channel: usize,
    cleavage_index: usize,
    peptide_len: usize,
) -> Result<(&'static str, usize)> {
    let b_number = cleavage_index + 1;
    let y_number = peptide_len.saturating_sub(cleavage_index + 1);
    match channel {
        0 => Ok(("b_z1", b_number)),
        1 => Ok(("b_z2", b_number)),
        2 => Ok(("y_z1", y_number)),
        3 => Ok(("y_z2", y_number)),
        _ => anyhow::bail!("unsupported AlphaPeptDeep comparison channel {channel}"),
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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

fn option_i32(value: Option<i32>) -> String {
    value.map(|x| x.to_string()).unwrap_or_else(|| "NA".into())
}

fn option_f32(value: Option<f32>) -> String {
    value
        .filter(|x| x.is_finite())
        .map(|x| format!("{x:.8}"))
        .unwrap_or_else(|| "NA".into())
}

fn parse_or<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse::<T>()
            .map_err(|error| anyhow::anyhow!("invalid argument {index} '{value}': {error}")),
        None => Ok(default),
    }
}
