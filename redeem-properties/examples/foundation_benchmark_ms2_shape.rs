//! Validation-only MS2 spectral-shape audit for a controlled unified continuation.
//!
//! The evaluator compares the zero-step `initial/` and trained `final/` unified
//! checkpoints on exactly the same source-balanced VALIDATION records. TEST is
//! intentionally not exposed by this tool so v0.13.7 cannot accidentally turn
//! the held-out partition into a tuning target.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig, FoundationPartition,
    FoundationTrainingRecord, PeptideFoundationUnifiedModel,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct UnifiedMetadata {
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: redeem_properties::foundation::FoundationDiffusionConfig,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    completed_steps: usize,
}

struct UnifiedPredictor {
    _varmap: VarMap,
    model: PeptideFoundationUnifiedModel,
}

impl UnifiedPredictor {
    fn load(checkpoint: &Path, device: &Device) -> Result<(Self, UnifiedMetadata)> {
        let metadata_path = checkpoint.join("metadata.yaml");
        let metadata: UnifiedMetadata = serde_yaml::from_str(
            &fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {metadata_path:?}"))?,
        )?;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = PeptideFoundationUnifiedModel::new(
            metadata.forward_config.clone(),
            metadata.inverse_config.clone(),
            vb,
        )?;
        let model_path = checkpoint.join("model.safetensors");
        varmap
            .load(&model_path)
            .with_context(|| format!("failed to load {model_path:?}"))?;
        Ok((
            Self {
                _varmap: varmap,
                model,
            },
            metadata,
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct RecordMetrics {
    fragments: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    cosine: f64,
    spectral_angle: f64,
    pearson: Option<f64>,
    exact_zero_count: usize,
    predicted_sum: f64,
    target_sum: f64,
}

impl RecordMetrics {
    fn mse(self) -> Option<f64> {
        (self.fragments > 0).then(|| self.squared_error_sum / self.fragments as f64)
    }

    fn mae(self) -> Option<f64> {
        (self.fragments > 0).then(|| self.absolute_error_sum / self.fragments as f64)
    }

    fn exact_zero_fraction(self) -> Option<f64> {
        (self.fragments > 0).then(|| self.exact_zero_count as f64 / self.fragments as f64)
    }

    fn mean_prediction(self) -> Option<f64> {
        (self.fragments > 0).then(|| self.predicted_sum / self.fragments as f64)
    }

    fn mean_target(self) -> Option<f64> {
        (self.fragments > 0).then(|| self.target_sum / self.fragments as f64)
    }
}

#[derive(Debug, Default, Clone)]
struct SummaryAccumulator {
    records: usize,
    fragments: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    cosine_sum: f64,
    spectral_angle_sum: f64,
    pearson_sum: f64,
    pearson_records: usize,
    exact_zero_count: usize,
    predicted_sum: f64,
    target_sum: f64,
}

impl SummaryAccumulator {
    fn push(&mut self, metrics: RecordMetrics) {
        if metrics.fragments == 0 {
            return;
        }
        self.records += 1;
        self.fragments += metrics.fragments;
        self.squared_error_sum += metrics.squared_error_sum;
        self.absolute_error_sum += metrics.absolute_error_sum;
        self.cosine_sum += metrics.cosine;
        self.spectral_angle_sum += metrics.spectral_angle;
        if let Some(pearson) = metrics.pearson {
            self.pearson_sum += pearson;
            self.pearson_records += 1;
        }
        self.exact_zero_count += metrics.exact_zero_count;
        self.predicted_sum += metrics.predicted_sum;
        self.target_sum += metrics.target_sum;
    }

    fn mse(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.squared_error_sum / self.fragments as f64)
    }

    fn mae(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.absolute_error_sum / self.fragments as f64)
    }

    fn mean_cosine(&self) -> Option<f64> {
        (self.records > 0).then(|| self.cosine_sum / self.records as f64)
    }

    fn mean_spectral_angle(&self) -> Option<f64> {
        (self.records > 0).then(|| self.spectral_angle_sum / self.records as f64)
    }

    fn mean_pearson(&self) -> Option<f64> {
        (self.pearson_records > 0).then(|| self.pearson_sum / self.pearson_records as f64)
    }

    fn exact_zero_fraction(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.exact_zero_count as f64 / self.fragments as f64)
    }

    fn mean_prediction(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.predicted_sum / self.fragments as f64)
    }

    fn mean_target(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.target_sum / self.fragments as f64)
    }
}

#[derive(Debug, Default)]
struct PairedAccumulator {
    records: usize,
    cosine_improved: usize,
    angle_improved: usize,
    pearson_pairs: usize,
    pearson_improved: usize,
    mse_improved: usize,
    zero_fraction_reduced: usize,
    cosine_delta_sum: f64,
    angle_delta_sum: f64,
    mse_delta_sum: f64,
    zero_fraction_delta_sum: f64,
}

impl PairedAccumulator {
    fn push(&mut self, initial: RecordMetrics, final_metrics: RecordMetrics) {
        if initial.fragments == 0 || final_metrics.fragments == 0 {
            return;
        }
        self.records += 1;
        let cosine_delta = final_metrics.cosine - initial.cosine;
        let angle_delta = final_metrics.spectral_angle - initial.spectral_angle;
        self.cosine_delta_sum += cosine_delta;
        self.angle_delta_sum += angle_delta;
        self.cosine_improved += usize::from(cosine_delta > 0.0);
        self.angle_improved += usize::from(angle_delta > 0.0);
        if let (Some(initial_pearson), Some(final_pearson)) =
            (initial.pearson, final_metrics.pearson)
        {
            self.pearson_pairs += 1;
            self.pearson_improved += usize::from(final_pearson > initial_pearson);
        }
        if let (Some(initial_mse), Some(final_mse)) = (initial.mse(), final_metrics.mse()) {
            self.mse_delta_sum += final_mse - initial_mse;
            self.mse_improved += usize::from(final_mse < initial_mse);
        }
        if let (Some(initial_zero), Some(final_zero)) = (
            initial.exact_zero_fraction(),
            final_metrics.exact_zero_fraction(),
        ) {
            self.zero_fraction_delta_sum += final_zero - initial_zero;
            self.zero_fraction_reduced += usize::from(final_zero < initial_zero);
        }
    }
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(4..=7).contains(&args.len()) {
        bail!(
            "usage: foundation_benchmark_ms2_shape <training.yaml> <initial_unified_checkpoint> <final_unified_checkpoint> <output_dir> [max_records_per_source=512] [batch_size=32] [seed=20260912]"
        );
    }

    let training_yaml = PathBuf::from(&args[0]);
    let initial_checkpoint = PathBuf::from(&args[1]);
    let final_checkpoint = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);
    let max_per_source = parse_or(&args, 4, 512usize)?;
    let batch_size = parse_or(&args, 5, 32usize)?;
    let seed = parse_or(&args, 6, 20_260_912u64)?;
    if max_per_source == 0 || batch_size == 0 {
        bail!("max_records_per_source and batch_size must be positive");
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let selected = select_source_balanced_validation_indices(
        &benchmark,
        &corpus.provenance,
        max_per_source,
        seed,
    );
    if selected.is_empty() {
        bail!("no VALIDATION records selected for MS2 shape audit");
    }

    let (initial, initial_metadata) = UnifiedPredictor::load(&initial_checkpoint, &device)?;
    let (final_model, final_metadata) = UnifiedPredictor::load(&final_checkpoint, &device)?;
    if initial_metadata.forward_config != final_metadata.forward_config
        || initial_metadata.inverse_config != final_metadata.inverse_config
    {
        bail!("initial/final unified checkpoints do not share one architecture");
    }
    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    for (label, metadata) in [("initial", &initial_metadata), ("final", &final_metadata)] {
        if metadata.corpus_fingerprint != corpus_fingerprint {
            bail!(
                "{label} checkpoint corpus fingerprint {} does not match loaded corpus {}",
                metadata.corpus_fingerprint,
                corpus_fingerprint
            );
        }
        if metadata.benchmark_manifest_fingerprint != benchmark_fingerprint {
            bail!(
                "{label} checkpoint benchmark fingerprint {} does not match loaded benchmark {}",
                metadata.benchmark_manifest_fingerprint,
                benchmark_fingerprint
            );
        }
    }

    let collator = FoundationCollator::new(
        initial_metadata.forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: run.trainer.collator.retention_time_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;

    fs::create_dir_all(&output_dir)?;
    let records_path = output_dir.join("ms2_shape_records.tsv");
    let fragments_path = output_dir.join("ms2_shape_fragments.tsv");
    let source_summary_path = output_dir.join("ms2_shape_source_summary.tsv");
    let family_summary_path = output_dir.join("ms2_shape_family_summary.tsv");
    let paired_summary_path = output_dir.join("ms2_shape_paired_summary.tsv");
    let mut records_writer = BufWriter::new(File::create(&records_path)?);
    let mut fragments_writer = BufWriter::new(File::create(&fragments_path)?);
    writeln!(
        records_writer,
        "checkpoint\trecord_index\tsource_id\tsource_family\tsequence\tpeptidoform\tfragments\tpointwise_mse\tpointwise_mae\tcosine\tspectral_angle\tpearson\texact_zero_fraction\tmean_target_intensity\tmean_predicted_intensity"
    )?;
    writeln!(
        fragments_writer,
        "checkpoint\trecord_index\tsource_id\tsource_family\tsequence\tpeptidoform\tcleavage_index\tchannel\tion_label\tproduct_mz\ttarget_intensity\tpredicted_intensity"
    )?;

    let mut source_summaries = BTreeMap::<(String, String), SummaryAccumulator>::new();
    let mut family_summaries = BTreeMap::<(String, String), SummaryAccumulator>::new();
    let mut overall = BTreeMap::<String, SummaryAccumulator>::new();
    let mut paired = PairedAccumulator::default();

    for chunk in selected.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, &device, seed ^ chunk[0] as u64)?;
        let initial_output =
            initial
                .model
                .forward()
                .forward_t(&batch.input, &batch.context, false)?;
        let final_output =
            final_model
                .model
                .forward()
                .forward_t(&batch.input, &batch.context, false)?;
        let initial_ms2 = initial_output.ms2.to_vec3::<f32>()?;
        let final_ms2 = final_output.ms2.to_vec3::<f32>()?;

        for (local, (&record_index, record)) in chunk.iter().zip(&owned).enumerate() {
            let provenance = &corpus.provenance[record_index];
            let source = provenance.source_id.as_str();
            let family = source_family(source);
            let initial_metrics = record_metrics(record, &initial_ms2[local]);
            let final_metrics = record_metrics(record, &final_ms2[local]);
            paired.push(initial_metrics, final_metrics);

            for (checkpoint, metrics, predictions) in [
                ("initial_v0136", initial_metrics, &initial_ms2[local]),
                ("final_v0137", final_metrics, &final_ms2[local]),
            ] {
                source_summaries
                    .entry((checkpoint.to_string(), source.to_string()))
                    .or_default()
                    .push(metrics);
                family_summaries
                    .entry((checkpoint.to_string(), family.to_string()))
                    .or_default()
                    .push(metrics);
                overall
                    .entry(checkpoint.to_string())
                    .or_default()
                    .push(metrics);
                write_record(
                    &mut records_writer,
                    checkpoint,
                    record_index,
                    source,
                    family,
                    record,
                    metrics,
                )?;
                write_fragments(
                    &mut fragments_writer,
                    checkpoint,
                    record_index,
                    source,
                    family,
                    record,
                    predictions,
                )?;
            }
        }
    }
    records_writer.flush()?;
    fragments_writer.flush()?;

    write_summary(&source_summary_path, "source_id", &source_summaries)?;
    write_summary(&family_summary_path, "source_family", &family_summaries)?;
    write_paired_summary(&paired_summary_path, &paired)?;

    println!("partition\tValidation");
    println!("test_partition_consumed\tNO");
    println!("corpus_fingerprint\t{corpus_fingerprint}");
    println!("benchmark_manifest_fingerprint\t{benchmark_fingerprint}");
    println!("selected_records\t{}", selected.len());
    println!("max_records_per_source\t{max_per_source}");
    println!(
        "initial_completed_steps\t{}",
        initial_metadata.completed_steps
    );
    println!("final_completed_steps\t{}", final_metadata.completed_steps);
    for checkpoint in ["initial_v0136", "final_v0137"] {
        if let Some(summary) = overall.get(checkpoint) {
            println!(
                "ms2_summary\tcheckpoint={checkpoint}\trecords={}\tfragments={}\tpointwise_mse={}\tpointwise_mae={}\tcosine={}\tspectral_angle={}\tpearson={}\texact_zero_fraction={}\tmean_target_intensity={}\tmean_predicted_intensity={}",
                summary.records,
                summary.fragments,
                fmt_opt(summary.mse()),
                fmt_opt(summary.mae()),
                fmt_opt(summary.mean_cosine()),
                fmt_opt(summary.mean_spectral_angle()),
                fmt_opt(summary.mean_pearson()),
                fmt_opt(summary.exact_zero_fraction()),
                fmt_opt(summary.mean_target()),
                fmt_opt(summary.mean_prediction()),
            );
        }
    }
    println!(
        "paired_change\trecords={}\tcosine_improved_fraction={}\tmean_cosine_delta={}\tspectral_angle_improved_fraction={}\tmean_spectral_angle_delta={}\tpearson_improved_fraction={}\tmse_improved_fraction={}\tmean_mse_delta={}\tzero_fraction_reduced_fraction={}\tmean_zero_fraction_delta={}",
        paired.records,
        fmt_ratio(paired.cosine_improved as f64, paired.records),
        fmt_ratio(paired.cosine_delta_sum, paired.records),
        fmt_ratio(paired.angle_improved as f64, paired.records),
        fmt_ratio(paired.angle_delta_sum, paired.records),
        fmt_ratio(paired.pearson_improved as f64, paired.pearson_pairs),
        fmt_ratio(paired.mse_improved as f64, paired.records),
        fmt_ratio(paired.mse_delta_sum, paired.records),
        fmt_ratio(paired.zero_fraction_reduced as f64, paired.records),
        fmt_ratio(paired.zero_fraction_delta_sum, paired.records),
    );
    println!("records_tsv\t{}", records_path.display());
    println!("fragments_tsv\t{}", fragments_path.display());
    println!("source_summary_tsv\t{}", source_summary_path.display());
    println!("family_summary_tsv\t{}", family_summary_path.display());
    println!("paired_summary_tsv\t{}", paired_summary_path.display());
    Ok(())
}

fn record_metrics(record: &FoundationTrainingRecord, predicted: &[Vec<f32>]) -> RecordMetrics {
    let mut targets = Vec::<f64>::new();
    let mut predictions = Vec::<f64>::new();
    for fragment in &record.fragments {
        let Some(&prediction) = predicted
            .get(fragment.cleavage_index)
            .and_then(|row| row.get(fragment.channel))
        else {
            continue;
        };
        if fragment.intensity.is_finite() && prediction.is_finite() {
            targets.push(f64::from(fragment.intensity));
            predictions.push(f64::from(prediction));
        }
    }
    if targets.is_empty() {
        return RecordMetrics {
            fragments: 0,
            squared_error_sum: 0.0,
            absolute_error_sum: 0.0,
            cosine: f64::NAN,
            spectral_angle: f64::NAN,
            pearson: None,
            exact_zero_count: 0,
            predicted_sum: 0.0,
            target_sum: 0.0,
        };
    }

    let mut squared_error_sum = 0.0;
    let mut absolute_error_sum = 0.0;
    let mut exact_zero_count = 0usize;
    let mut predicted_sum = 0.0;
    let mut target_sum = 0.0;
    for (&target, &prediction) in targets.iter().zip(&predictions) {
        let error = prediction - target;
        squared_error_sum += error * error;
        absolute_error_sum += error.abs();
        exact_zero_count += usize::from(prediction == 0.0);
        predicted_sum += prediction;
        target_sum += target;
    }
    let dot = targets
        .iter()
        .zip(&predictions)
        .map(|(target, prediction)| target * prediction)
        .sum::<f64>();
    let target_norm = targets
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let prediction_norm = predictions
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let cosine = if target_norm > 0.0 && prediction_norm > 0.0 {
        (dot / (target_norm * prediction_norm)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    RecordMetrics {
        fragments: targets.len(),
        squared_error_sum,
        absolute_error_sum,
        cosine,
        spectral_angle: 1.0 - 2.0 * cosine.acos() / std::f64::consts::PI,
        pearson: pearson(&targets, &predictions),
        exact_zero_count,
        predicted_sum,
        target_sum,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_record(
    writer: &mut BufWriter<File>,
    checkpoint: &str,
    record_index: usize,
    source: &str,
    family: &str,
    record: &FoundationTrainingRecord,
    metrics: RecordMetrics,
) -> Result<()> {
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        checkpoint,
        record_index,
        escape_tsv(source),
        escape_tsv(family),
        record.peptidoform.sequence,
        escape_tsv(&format_peptidoform(record)),
        metrics.fragments,
        fmt_opt(metrics.mse()),
        fmt_opt(metrics.mae()),
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
        fmt_opt(metrics.exact_zero_fraction()),
        fmt_opt(metrics.mean_target()),
        fmt_opt(metrics.mean_prediction()),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_fragments(
    writer: &mut BufWriter<File>,
    checkpoint: &str,
    record_index: usize,
    source: &str,
    family: &str,
    record: &FoundationTrainingRecord,
    predicted: &[Vec<f32>],
) -> Result<()> {
    for fragment in &record.fragments {
        let prediction = predicted
            .get(fragment.cleavage_index)
            .and_then(|row| row.get(fragment.channel))
            .copied()
            .unwrap_or(0.0);
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}",
            checkpoint,
            record_index,
            escape_tsv(source),
            escape_tsv(family),
            record.peptidoform.sequence,
            escape_tsv(&format_peptidoform(record)),
            fragment.cleavage_index,
            fragment.channel,
            ion_label(
                fragment.channel,
                fragment.cleavage_index,
                record.peptidoform.sequence.len()
            ),
            option_f32(fragment.product_mz),
            fragment.intensity,
            prediction,
        )?;
    }
    Ok(())
}

fn write_summary(
    path: &Path,
    grouping_header: &str,
    summaries: &BTreeMap<(String, String), SummaryAccumulator>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "checkpoint\t{}\trecords\tfragments\tpointwise_mse\tpointwise_mae\tmean_cosine\tmean_spectral_angle\tmean_pearson\texact_zero_fraction\tmean_target_intensity\tmean_predicted_intensity",
        grouping_header
    )?;
    for ((checkpoint, group), summary) in summaries {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            checkpoint,
            escape_tsv(group),
            summary.records,
            summary.fragments,
            fmt_opt(summary.mse()),
            fmt_opt(summary.mae()),
            fmt_opt(summary.mean_cosine()),
            fmt_opt(summary.mean_spectral_angle()),
            fmt_opt(summary.mean_pearson()),
            fmt_opt(summary.exact_zero_fraction()),
            fmt_opt(summary.mean_target()),
            fmt_opt(summary.mean_prediction()),
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_paired_summary(path: &Path, paired: &PairedAccumulator) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "records\tcosine_improved_fraction\tmean_cosine_delta\tspectral_angle_improved_fraction\tmean_spectral_angle_delta\tpearson_pairs\tpearson_improved_fraction\tmse_improved_fraction\tmean_mse_delta\tzero_fraction_reduced_fraction\tmean_zero_fraction_delta"
    )?;
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        paired.records,
        fmt_ratio(paired.cosine_improved as f64, paired.records),
        fmt_ratio(paired.cosine_delta_sum, paired.records),
        fmt_ratio(paired.angle_improved as f64, paired.records),
        fmt_ratio(paired.angle_delta_sum, paired.records),
        paired.pearson_pairs,
        fmt_ratio(paired.pearson_improved as f64, paired.pearson_pairs),
        fmt_ratio(paired.mse_improved as f64, paired.records),
        fmt_ratio(paired.mse_delta_sum, paired.records),
        fmt_ratio(paired.zero_fraction_reduced as f64, paired.records),
        fmt_ratio(paired.zero_fraction_delta_sum, paired.records),
    )?;
    writer.flush()?;
    Ok(())
}

fn select_source_balanced_validation_indices(
    benchmark: &FoundationBenchmarkManifest,
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    max_per_source: usize,
    seed: u64,
) -> Vec<usize> {
    let mut by_source = BTreeMap::<String, Vec<usize>>::new();
    for entry in &benchmark.entries {
        if entry.partition != FoundationPartition::Validation {
            continue;
        }
        if let Some(item) = provenance.get(entry.record_index) {
            by_source
                .entry(item.source_id.clone())
                .or_default()
                .push(entry.record_index);
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

fn source_family(source: &str) -> &str {
    let lower = source.to_ascii_lowercase();
    if lower.starts_with("dphlv2_") || lower == "dphlv2" {
        "DPHLv2"
    } else if lower.starts_with("pxd058337_") || lower == "pxd058337" {
        "PXD058337"
    } else if lower.starts_with("pxd034128_") || lower == "pxd034128" {
        "PXD034128"
    } else if lower.starts_with("pxd035249_") || lower == "pxd035249_csf" {
        "PXD035249"
    } else if lower == "ip2_bruker_human" || lower.starts_with("ip2_") {
        "IP2/Bruker"
    } else if lower == "openswath_finetuning" || lower.starts_with("openswath_") {
        "OpenSWATH"
    } else if lower == "pan_human_library" || lower.starts_with("pan_human") {
        "Pan-Human"
    } else if lower == "pride_human_msp" || lower.starts_with("pride_human") {
        "PRIDE Human MSP"
    } else {
        source
    }
}

fn pearson(first: &[f64], second: &[f64]) -> Option<f64> {
    if first.len() != second.len() || first.len() < 2 {
        return None;
    }
    let n = first.len() as f64;
    let mean_first = first.iter().sum::<f64>() / n;
    let mean_second = second.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut first_squared = 0.0;
    let mut second_squared = 0.0;
    for (&a, &b) in first.iter().zip(second) {
        let da = a - mean_first;
        let db = b - mean_second;
        numerator += da * db;
        first_squared += da * da;
        second_squared += db * db;
    }
    let denominator = (first_squared * second_squared).sqrt();
    (denominator > 1.0e-12).then(|| (numerator / denominator).clamp(-1.0, 1.0))
}

fn stable_text_hash(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
        other => format!("channel{other}@{cleavage_index}"),
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
        .map(|modification| format!("{}@{:?}", modification.identity_label(), modification.site))
        .collect::<Vec<_>>()
        .join(";");
    format!("{}|{}", record.peptidoform.sequence, mods)
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.8}"))
        .unwrap_or_else(|| "NA".into())
}

fn fmt_ratio(numerator: f64, denominator: usize) -> String {
    (denominator > 0)
        .then(|| format!("{:.8}", numerator / denominator as f64))
        .unwrap_or_else(|| "NA".into())
}

fn option_f32(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.8}"))
        .unwrap_or_else(|| "NA".into())
}

fn escape_tsv(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
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
