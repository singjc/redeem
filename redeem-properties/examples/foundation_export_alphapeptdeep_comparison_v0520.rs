//! Export the frozen historical VALIDATION RT/MS2 comparison set for ReDeeM v0.52 vs
//! AlphaPeptDeep 1.5.1.
//!
//! TRAIN is accessed only for the fixed 256-record RT calibration sample. Historical VALIDATION
//! supplies the descriptive RT/MS2 comparison cohort. The reserved TRAIN-HOLDOUT was consumed
//! previously and is not opened here; historical TEST is never opened.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationFragmentContextBatchV0350, FoundationPartition, FoundationScalarPhysicsBatchV0360,
    FoundationTargetNormalizationConfig, FoundationTrainingRecord, PeptideFoundationV0520Config,
    PeptideFoundationV0520Model, RetentionTimeObjective, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520,
};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = "foundation_v0520_historical_validation_apd_export_v1";
const V052_VERSION: u32 = 520;
const V052_OBJECTIVE: &str = "v0520_mobility_aware_pair_representation";
const DEFAULT_PER_TASK: usize = usize::MAX;
const DEFAULT_RT_CALIBRATION: usize = 256;
const DEFAULT_SEED: u64 = 20_260_916;
const CPU_BATCH_SIZE: usize = 16;
const CUDA_BATCH_SIZE: usize = 32;

#[derive(Debug, Clone, Deserialize)]
struct V052Metadata {
    version: u32,
    objective: String,
    architecture: String,
    v0520_config: PeptideFoundationV0520Config,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    completed_epochs: usize,
    completed_updates: usize,
    smoke_mode: bool,
}

#[derive(Debug, Clone)]
struct Predictions {
    rt: f32,
    ms2: Vec<Vec<f32>>,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(3..=6).contains(&args.len()) {
        anyhow::bail!(
            "usage: foundation_export_alphapeptdeep_comparison_v0520 RUN_V0260.yaml V0520_BEST OUTPUT_DIR [validation_per_task=all_if_omitted] [rt_train_calibration=256] [seed=20260916]"
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
    if output_dir.exists() {
        anyhow::bail!("comparison export output must be fresh: {output_dir:?}");
    }

    let metadata = read_metadata(&checkpoint)?;
    validate_metadata(&metadata)?;

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let validation_allowed = benchmark.partition_indices(FoundationPartition::Validation);
    let train_allowed = benchmark.partition_indices(FoundationPartition::Train);

    let rt_validation = select_records(
        &validation_allowed,
        &corpus.records,
        per_task,
        seed ^ 0x5254,
        |record| rt_target(record, metadata.rt_objective).is_some(),
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

    if rt_validation.is_empty() || ms2_validation.is_empty() {
        anyhow::bail!(
            "no eligible unmodified historical VALIDATION records: RT {} MS2 {}",
            rt_validation.len(),
            ms2_validation.len()
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
    validation_union.extend(ms2_validation.iter().copied());
    let validation_union = validation_union.into_iter().collect::<Vec<_>>();

    let device = comparison_device()?;
    let batch_size = if matches!(device, Device::Cpu) {
        CPU_BATCH_SIZE
    } else {
        CUDA_BATCH_SIZE
    };

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationV0520Model::new(metadata.v0520_config.clone(), vb)?;
    let model_path = checkpoint.join("model.safetensors");
    varmap
        .load(&model_path)
        .with_context(|| format!("load frozen v0.52 checkpoint {model_path:?}"))?;

    let featurizer = metadata
        .v0520_config
        .base_v0510
        .base_v0500
        .featurizer_config();
    let collator = FoundationCollator::new(
        featurizer.clone(),
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
        batch_size,
        &device,
    )?;

    fs::create_dir_all(&output_dir)?;
    let precursor_path = output_dir.join("foundation_comparison_precursors.tsv");
    let fragment_path = output_dir.join("foundation_comparison_fragments.tsv");
    let calibration_path = output_dir.join("foundation_comparison_rt_train_calibration.tsv");

    let rt_set: BTreeSet<usize> = rt_validation.iter().copied().collect();
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
            .context("missing frozen v0.52 prediction")?;
        writeln!(
            precursor,
            "{}\tVALIDATION\t{}\t{}\t\t\t{}\t{}\t{}\t{}\tNO\t{}\t{}\t{:.8}\tNA\tNA",
            index,
            escape_tsv(&corpus.provenance[index].source_id),
            record.peptidoform.sequence,
            option_i32(record.context.charge),
            option_f32(record.context.nce),
            escape_tsv(record.context.instrument_name.as_deref().unwrap_or("")),
            yes_no(rt_set.contains(&index)),
            yes_no(ms2_set.contains(&index)),
            option_f32(rt_target(record, metadata.rt_objective)),
            pred.rt,
        )?;
    }
    precursor.flush()?;

    let mut fragments = BufWriter::new(File::create(&fragment_path)?);
    writeln!(
        fragments,
        "record_index\tsequence\tcleavage_index\tchannel\tcharged_frag_type\tfragment_number\tproduct_mz\ttarget_intensity\tfoundation_intensity"
    )?;
    for &index in &ms2_validation {
        let record = &corpus.records[index];
        let pred = predictions
            .get(&index)
            .context("missing frozen v0.52 MS2 prediction")?;
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
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}",
                index,
                record.peptidoform.sequence,
                fragment.cleavage_index,
                fragment.channel,
                frag_type,
                number,
                option_f32(fragment.product_mz),
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
    println!(
        "model_architecture\t{}",
        FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520
    );
    println!("checkpoint_metadata_version\t{}", metadata.version);
    println!("checkpoint_completed_epochs\t{}", metadata.completed_epochs);
    println!(
        "checkpoint_completed_updates\t{}",
        metadata.completed_updates
    );
    println!("comparison_device\t{device:?}");
    println!("comparison_batch_size\t{batch_size}");
    println!("partition\tHISTORICAL_VALIDATION");
    println!("validation_rt_records\t{}", rt_validation.len());
    println!("validation_ms2_records\t{}", ms2_validation.len());
    println!("validation_union_records\t{}", validation_union.len());
    println!("rt_train_calibration_records\t{}", rt_train.len());
    println!("train_holdout_consumed\tYES");
    println!("historical_validation_consumed\tYES");
    println!("historical_test_consumed\tNO");
    println!("rt_objective\t{:?}", metadata.rt_objective);
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

fn read_metadata(checkpoint: &Path) -> Result<V052Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.52 metadata {path:?}"))?,
    )
    .with_context(|| format!("parse v0.52 metadata {path:?}"))
}

fn validate_metadata(metadata: &V052Metadata) -> Result<()> {
    if metadata.version != V052_VERSION
        || metadata.objective != V052_OBJECTIVE
        || metadata.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520
        || metadata.completed_epochs == 0
        || metadata.completed_updates == 0
        || metadata.smoke_mode
    {
        anyhow::bail!("historical VALIDATION comparison requires the selected completed non-smoke v0.52 best checkpoint");
    }
    metadata.v0520_config.validate()?;
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
    model: &PeptideFoundationV0520Model,
    collator: &FoundationCollator,
    normalization: &FoundationTargetNormalizationConfig,
    batch_size: usize,
    device: &Device,
) -> Result<HashMap<usize, Predictions>> {
    let mut out = HashMap::new();
    for chunk in indices.chunks(batch_size) {
        let batch_records = chunk
            .iter()
            .map(|&index| records[index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&batch_records, device, 0)?;
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &batch_records,
            model.config().base_v0510.base_v0500.max_sequence_len,
            device,
        )?;
        let fragment = FoundationFragmentContextBatchV0350::from_records(
            &batch_records,
            &model.config().base_v0510.base_v0500.featurizer_config(),
            device,
        )?;
        let prediction =
            model.property_forward_t(&batch.input, &batch.context, &physics, &fragment, false)?;
        let rt = normalization
            .rt
            .denormalize_tensor(&prediction.rt)?
            .squeeze(1)?
            .to_vec1::<f32>()?;
        let ms2 = prediction.ms2.to_vec3::<f32>()?;
        for local in 0..chunk.len() {
            out.insert(
                chunk[local],
                Predictions {
                    rt: rt[local],
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

fn comparison_device() -> Result<Device> {
    match env::var("REDEEM_COMPARISON_DEVICE")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => Ok(Device::Cpu),
        "auto" | "cuda" => {
            #[cfg(feature = "cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(device) => return Ok(device),
                    Err(error)
                        if env::var("REDEEM_COMPARISON_DEVICE")
                            .unwrap_or_else(|_| "auto".to_string())
                            .eq_ignore_ascii_case("cuda") =>
                    {
                        anyhow::bail!("CUDA comparison device requested but unavailable: {error}")
                    }
                    Err(_) => {}
                }
            }
            #[cfg(not(feature = "cuda"))]
            if env::var("REDEEM_COMPARISON_DEVICE")
                .unwrap_or_else(|_| "auto".to_string())
                .eq_ignore_ascii_case("cuda")
            {
                anyhow::bail!("CUDA comparison device requested but binary was built without cuda");
            }
            Ok(Device::Cpu)
        }
        other => {
            anyhow::bail!("invalid REDEEM_COMPARISON_DEVICE={other:?}; expected auto, cuda, or cpu")
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0520_apd_metadata_contract_rejects_smoke_or_wrong_version() {
        let mut metadata = V052Metadata {
            version: V052_VERSION,
            objective: V052_OBJECTIVE.into(),
            architecture: FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520.into(),
            v0520_config: PeptideFoundationV0520Config::default(),
            rt_objective: RetentionTimeObjective::default(),
            target_normalization: FoundationTargetNormalizationConfig::default(),
            completed_epochs: 1,
            completed_updates: 1,
            smoke_mode: false,
        };
        assert!(validate_metadata(&metadata).is_ok());
        metadata.smoke_mode = true;
        assert!(validate_metadata(&metadata).is_err());
        metadata.smoke_mode = false;
        metadata.version = 519;
        assert!(validate_metadata(&metadata).is_err());
    }
}
