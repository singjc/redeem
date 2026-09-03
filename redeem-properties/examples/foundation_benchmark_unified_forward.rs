//! Source-resolved forward-property benchmark for a unified peptide foundation checkpoint.
//!
//! The evaluator compares the corrected forward warm-start checkpoint and a unified
//! checkpoint on exactly the same sequence-disjoint benchmark records. It emits
//! per-record RT/CCS predictions, annotated-fragment MS2 similarity metrics, and
//! per-fragment values suitable for downstream mirror plots.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig, FoundationPartition,
    FoundationRegressionNormalization, FoundationTrainer, FoundationTrainingRecord,
    PeptideFoundationUnifiedModel,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct UnifiedMetadata {
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: redeem_properties::foundation::FoundationDiffusionConfig,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    completed_steps: usize,
}

#[derive(Debug, Default, Clone)]
struct RegressionAccumulator {
    n: usize,
    abs_error: f64,
    squared_error: f64,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    sum_xy: f64,
}

impl RegressionAccumulator {
    fn push(&mut self, target: f64, prediction: f64) {
        if !(target.is_finite() && prediction.is_finite()) {
            return;
        }
        self.n += 1;
        let error = prediction - target;
        self.abs_error += error.abs();
        self.squared_error += error * error;
        self.sum_x += target;
        self.sum_y += prediction;
        self.sum_x2 += target * target;
        self.sum_y2 += prediction * prediction;
        self.sum_xy += target * prediction;
    }

    fn mae(&self) -> Option<f64> {
        (self.n > 0).then_some(self.abs_error / self.n as f64)
    }

    fn rmse(&self) -> Option<f64> {
        (self.n > 0).then_some((self.squared_error / self.n as f64).sqrt())
    }

    fn pearson(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let numerator = n * self.sum_xy - self.sum_x * self.sum_y;
        let left = n * self.sum_x2 - self.sum_x * self.sum_x;
        let right = n * self.sum_y2 - self.sum_y * self.sum_y;
        let denom = (left.max(0.0) * right.max(0.0)).sqrt();
        (denom > 0.0).then_some(numerator / denom)
    }

    fn r2(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let mean = self.sum_x / n;
        let sst = self.sum_x2 - 2.0 * mean * self.sum_x + n * mean * mean;
        (sst > 0.0).then_some(1.0 - self.squared_error / sst)
    }
}

#[derive(Debug, Default, Clone)]
struct Ms2Accumulator {
    records: usize,
    fragments: usize,
    cosine_sum: f64,
    spectral_angle_sum: f64,
    pearson_sum: f64,
    pearson_records: usize,
}

impl Ms2Accumulator {
    fn push(&mut self, metrics: Ms2RecordMetrics) {
        if metrics.fragments == 0 {
            return;
        }
        self.records += 1;
        self.fragments += metrics.fragments;
        self.cosine_sum += metrics.cosine;
        self.spectral_angle_sum += metrics.spectral_angle;
        if let Some(value) = metrics.pearson {
            self.pearson_sum += value;
            self.pearson_records += 1;
        }
    }
}

#[derive(Debug, Default, Clone)]
struct SourceMetrics {
    records: usize,
    rt: RegressionAccumulator,
    ccs: RegressionAccumulator,
    ms2: Ms2Accumulator,
}

#[derive(Debug, Clone, Copy)]
struct Ms2RecordMetrics {
    fragments: usize,
    cosine: f64,
    spectral_angle: f64,
    pearson: Option<f64>,
}

struct UnifiedPredictor {
    _varmap: VarMap,
    model: PeptideFoundationUnifiedModel,
    collator: FoundationCollator,
    rt_normalization: FoundationRegressionNormalization,
    ccs_normalization: FoundationRegressionNormalization,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(4..=8).contains(&args.len()) {
        bail!(
            "usage: foundation_benchmark_unified_forward <training.yaml> <baseline_forward_checkpoint> <unified_checkpoint> <output_dir> [validation|test=validation] [max_records_per_source=512] [batch_size=32] [seed=20260912]"
        );
    }

    let training_yaml = PathBuf::from(&args[0]);
    let baseline_checkpoint = PathBuf::from(&args[1]);
    let unified_checkpoint = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);
    let partition = match args.get(4).map(String::as_str).unwrap_or("validation") {
        "validation" => FoundationPartition::Validation,
        "test" => FoundationPartition::Test,
        other => bail!("unsupported partition '{other}'"),
    };
    let max_per_source = parse_or(&args, 5, 512usize)?;
    let batch_size = parse_or(&args, 6, 32usize)?;
    let seed = parse_or(&args, 7, 20_260_912u64)?;
    if max_per_source == 0 || batch_size == 0 {
        bail!("max_records_per_source and batch_size must be positive");
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let selected = select_source_balanced_indices(
        &benchmark,
        &corpus.provenance,
        partition,
        max_per_source,
        seed,
    );
    if selected.is_empty() {
        bail!("no records selected for forward benchmark");
    }

    let (baseline, baseline_metadata) =
        FoundationTrainer::from_checkpoint(&baseline_checkpoint, Device::Cpu).with_context(
            || format!("failed to load baseline forward checkpoint {baseline_checkpoint:?}"),
        )?;

    let unified_metadata_path = unified_checkpoint.join("metadata.yaml");
    let unified_metadata: UnifiedMetadata = serde_yaml::from_str(
        &fs::read_to_string(&unified_metadata_path)
            .with_context(|| format!("failed to read {unified_metadata_path:?}"))?,
    )?;
    if unified_metadata.forward_config != baseline_metadata.model_config {
        bail!("unified forward architecture differs from corrected forward checkpoint");
    }

    let mut unified_varmap = VarMap::new();
    let unified_vb = VarBuilder::from_varmap(&unified_varmap, DType::F32, &device);
    let unified_model = PeptideFoundationUnifiedModel::new(
        unified_metadata.forward_config.clone(),
        unified_metadata.inverse_config.clone(),
        unified_vb,
    )?;
    unified_varmap
        .load(unified_checkpoint.join("model.safetensors"))
        .with_context(|| format!("failed to load unified checkpoint {unified_checkpoint:?}"))?;
    let clean_collator = FoundationCollator::new(
        unified_metadata.forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: baseline_metadata
                .trainer_config
                .collator
                .retention_time_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let unified = UnifiedPredictor {
        _varmap: unified_varmap,
        model: unified_model,
        collator: clean_collator,
        rt_normalization: baseline_metadata
            .trainer_config
            .target_normalization
            .rt
            .clone(),
        ccs_normalization: baseline_metadata
            .trainer_config
            .target_normalization
            .ccs
            .clone(),
    };

    fs::create_dir_all(&output_dir)?;
    let predictions_path = output_dir.join("forward_predictions.tsv");
    let fragments_path = output_dir.join("ms2_fragments.tsv");
    let summary_path = output_dir.join("forward_source_summary.tsv");
    let mut predictions = BufWriter::new(File::create(&predictions_path)?);
    let mut fragments = BufWriter::new(File::create(&fragments_path)?);

    writeln!(
        predictions,
        "checkpoint\trecord_index\tsource_id\tsource_record_index\tsequence\tpeptidoform\tcharge\tprecursor_mz\tinstrument\ttarget_rt\tpredicted_rt\trt_error\ttarget_ccs\tpredicted_ccs\tccs_error\tms2_fragments\tms2_cosine\tms2_spectral_angle\tms2_pearson"
    )?;
    writeln!(
        fragments,
        "checkpoint\trecord_index\tsource_id\tsequence\tpeptidoform\tcleavage_index\tchannel\tion_label\tproduct_mz\ttarget_intensity\tpredicted_intensity"
    )?;

    let mut summaries = BTreeMap::<(String, String), SourceMetrics>::new();
    for chunk in selected.chunks(batch_size) {
        let records = chunk
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect::<Vec<_>>();
        let peptides = records
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect::<Vec<_>>();
        let contexts = records
            .iter()
            .map(|record| record.context.clone())
            .collect::<Vec<_>>();

        let baseline_output = baseline.predict_native(&peptides, &contexts)?;
        write_batch(
            "baseline_v0129",
            chunk,
            &records,
            &corpus.provenance,
            &baseline_output.rt,
            &baseline_output.ccs,
            &baseline_output.ms2,
            &mut predictions,
            &mut fragments,
            &mut summaries,
        )?;

        let batch = unified
            .collator
            .collate(&records, &device, seed ^ chunk[0] as u64)?;
        let mut unified_output =
            unified
                .model
                .forward()
                .forward_t(&batch.input, &batch.context, false)?;
        unified_output.rt = unified
            .rt_normalization
            .denormalize_tensor(&unified_output.rt)?;
        unified_output.ccs = unified
            .ccs_normalization
            .denormalize_tensor(&unified_output.ccs)?;
        write_batch(
            "unified_v0134",
            chunk,
            &records,
            &corpus.provenance,
            &unified_output.rt,
            &unified_output.ccs,
            &unified_output.ms2,
            &mut predictions,
            &mut fragments,
            &mut summaries,
        )?;
    }
    predictions.flush()?;
    fragments.flush()?;

    let mut summary = BufWriter::new(File::create(&summary_path)?);
    writeln!(
        summary,
        "checkpoint\tsource_id\trecords\trt_n\trt_mae\trt_rmse\trt_pearson\trt_r2\tccs_n\tccs_mae\tccs_rmse\tccs_pearson\tccs_r2\tms2_records\tms2_fragments\tms2_mean_cosine\tms2_mean_spectral_angle\tms2_mean_pearson"
    )?;
    for ((checkpoint, source), metrics) in &summaries {
        writeln!(
            summary,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            checkpoint,
            source,
            metrics.records,
            metrics.rt.n,
            fmt_opt(metrics.rt.mae()),
            fmt_opt(metrics.rt.rmse()),
            fmt_opt(metrics.rt.pearson()),
            fmt_opt(metrics.rt.r2()),
            metrics.ccs.n,
            fmt_opt(metrics.ccs.mae()),
            fmt_opt(metrics.ccs.rmse()),
            fmt_opt(metrics.ccs.pearson()),
            fmt_opt(metrics.ccs.r2()),
            metrics.ms2.records,
            metrics.ms2.fragments,
            fmt_ratio(metrics.ms2.cosine_sum, metrics.ms2.records),
            fmt_ratio(metrics.ms2.spectral_angle_sum, metrics.ms2.records),
            fmt_ratio(metrics.ms2.pearson_sum, metrics.ms2.pearson_records),
        )?;
    }
    summary.flush()?;

    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!(
        "benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        benchmark.manifest_fingerprint()
    );
    println!("partition\t{:?}", partition);
    println!("selected_records\t{}", selected.len());
    println!("max_records_per_source\t{max_per_source}");
    println!(
        "unified_completed_steps\t{}",
        unified_metadata.completed_steps
    );
    println!(
        "unified_recorded_corpus_fingerprint\t{}",
        unified_metadata.corpus_fingerprint
    );
    println!(
        "unified_recorded_benchmark_fingerprint\t{}",
        unified_metadata.benchmark_manifest_fingerprint
    );
    println!("forward_predictions\t{}", predictions_path.display());
    println!("ms2_fragments\t{}", fragments_path.display());
    println!("forward_source_summary\t{}", summary_path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_batch(
    checkpoint: &str,
    indices: &[usize],
    records: &[FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    rt: &Tensor,
    ccs: &Tensor,
    ms2: &Tensor,
    predictions: &mut BufWriter<File>,
    fragments_writer: &mut BufWriter<File>,
    summaries: &mut BTreeMap<(String, String), SourceMetrics>,
) -> Result<()> {
    let predicted_rt = rt.squeeze(1)?.to_vec1::<f32>()?;
    let predicted_ccs = ccs.squeeze(1)?.to_vec1::<f32>()?;
    let ms2_values = ms2.to_vec3::<f32>()?;

    for (local, (&record_index, record)) in indices.iter().zip(records).enumerate() {
        let prov = &provenance[record_index];
        let source_id = &prov.source_id;
        let summary = summaries
            .entry((checkpoint.to_owned(), source_id.clone()))
            .or_default();
        summary.records += 1;

        let target_rt = record.retention_time.normalized.map(f64::from);
        let pred_rt = f64::from(predicted_rt[local]);
        if let Some(target) = target_rt {
            summary.rt.push(target, pred_rt);
        }
        let target_ccs = record.ccs.map(f64::from);
        let pred_ccs = f64::from(predicted_ccs[local]);
        if let Some(target) = target_ccs {
            summary.ccs.push(target, pred_ccs);
        }

        let metrics = ms2_record_metrics(record, &ms2_values[local]);
        summary.ms2.push(metrics);

        let peptidoform = format_peptidoform(record);
        writeln!(
            predictions,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{}\t{:.8}\t{}\t{}\t{}\t{}\t{}",
            checkpoint,
            record_index,
            escape_tsv(source_id),
            prov.source_record_index,
            record.peptidoform.sequence,
            escape_tsv(&peptidoform),
            option_i32(record.context.charge),
            option_f32(record.context.precursor_mz),
            escape_tsv(record.context.instrument_name.as_deref().unwrap_or("")),
            option_f64(target_rt),
            pred_rt,
            option_f64(target_rt.map(|value| pred_rt - value)),
            option_f64(target_ccs),
            pred_ccs,
            option_f64(target_ccs.map(|value| pred_ccs - value)),
            metrics.fragments,
            if metrics.fragments > 0 {
                format!("{:.8}", metrics.cosine)
            } else {
                "NA".into()
            },
            if metrics.fragments > 0 {
                format!("{:.8}", metrics.spectral_angle)
            } else {
                "NA".into()
            },
            fmt_opt(metrics.pearson),
        )?;

        for fragment in &record.fragments {
            let predicted = ms2_values[local]
                .get(fragment.cleavage_index)
                .and_then(|row| row.get(fragment.channel))
                .copied()
                .unwrap_or(0.0);
            writeln!(
                fragments_writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}",
                checkpoint,
                record_index,
                escape_tsv(source_id),
                record.peptidoform.sequence,
                escape_tsv(&peptidoform),
                fragment.cleavage_index,
                fragment.channel,
                ion_label(
                    fragment.channel,
                    fragment.cleavage_index,
                    record.peptidoform.sequence.len()
                ),
                option_f32(fragment.product_mz),
                fragment.intensity,
                predicted,
            )?;
        }
    }
    Ok(())
}

fn ms2_record_metrics(
    record: &FoundationTrainingRecord,
    predicted: &[Vec<f32>],
) -> Ms2RecordMetrics {
    let mut targets = Vec::<f64>::new();
    let mut predictions = Vec::<f64>::new();
    for fragment in &record.fragments {
        let Some(prediction) = predicted
            .get(fragment.cleavage_index)
            .and_then(|row| row.get(fragment.channel))
        else {
            continue;
        };
        if fragment.intensity.is_finite() && prediction.is_finite() {
            targets.push(f64::from(fragment.intensity));
            predictions.push(f64::from(*prediction));
        }
    }
    let fragments = targets.len();
    if fragments == 0 {
        return Ms2RecordMetrics {
            fragments: 0,
            cosine: f64::NAN,
            spectral_angle: f64::NAN,
            pearson: None,
        };
    }
    let dot = targets
        .iter()
        .zip(&predictions)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let left = targets.iter().map(|x| x * x).sum::<f64>().sqrt();
    let right = predictions.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine = if left > 0.0 && right > 0.0 {
        (dot / (left * right)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let spectral_angle = 1.0 - 2.0 * cosine.acos() / std::f64::consts::PI;
    let mut correlation = RegressionAccumulator::default();
    for (&target, &prediction) in targets.iter().zip(&predictions) {
        correlation.push(target, prediction);
    }
    Ms2RecordMetrics {
        fragments,
        cosine,
        spectral_angle,
        pearson: correlation.pearson(),
    }
}

fn select_source_balanced_indices(
    benchmark: &FoundationBenchmarkManifest,
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    partition: FoundationPartition,
    max_per_source: usize,
    seed: u64,
) -> Vec<usize> {
    let mut by_source = BTreeMap::<String, Vec<usize>>::new();
    for entry in &benchmark.entries {
        if entry.partition != partition {
            continue;
        }
        let index = entry.record_index;
        if let Some(prov) = provenance.get(index) {
            by_source
                .entry(prov.source_id.clone())
                .or_default()
                .push(index);
        }
    }
    let mut selected = Vec::new();
    for (source, mut indices) in by_source {
        let source_seed = seed ^ stable_text_hash(&source);
        indices.sort_by_key(|&index| mix64(source_seed ^ index as u64));
        indices.truncate(max_per_source.min(indices.len()));
        selected.extend(indices);
    }
    selected.sort_unstable();
    selected
}

fn stable_text_hash(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn ion_label(channel: usize, cleavage_index: usize, peptide_len: usize) -> String {
    let b_ordinal = cleavage_index + 1;
    let y_ordinal = peptide_len.saturating_sub(cleavage_index + 1);
    match channel {
        0 => format!("b{b_ordinal}^1"),
        1 => format!("b{b_ordinal}^2"),
        2 => format!("y{y_ordinal}^1"),
        3 => format!("y{y_ordinal}^2"),
        4 => format!("b{b_ordinal}-H2O"),
        5 => format!("y{y_ordinal}-H2O"),
        6 => format!("b{b_ordinal}-NH3"),
        7 => format!("y{y_ordinal}-NH3"),
        other => format!("channel{other}@{}", cleavage_index),
    }
}

fn format_peptidoform(record: &FoundationTrainingRecord) -> String {
    if record.peptidoform.modifications.is_empty() {
        return record.peptidoform.sequence.clone();
    }
    let mods = record
        .peptidoform
        .modifications
        .iter()
        .map(|m| format!("{}@{:?}", m.identity_label(), m.site))
        .collect::<Vec<_>>()
        .join(";");
    format!("{}|{}", record.peptidoform.sequence, mods)
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

fn option_i32(value: Option<i32>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "NA".into())
}

fn option_f32(value: Option<f32>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.8}"))
        .unwrap_or_else(|| "NA".into())
}

fn option_f64(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.8}"))
        .unwrap_or_else(|| "NA".into())
}

fn fmt_opt(value: Option<f64>) -> String {
    option_f64(value)
}

fn fmt_ratio(sum: f64, n: usize) -> String {
    if n == 0 {
        "NA".into()
    } else {
        format!("{:.8}", sum / n as f64)
    }
}

fn escape_tsv(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
