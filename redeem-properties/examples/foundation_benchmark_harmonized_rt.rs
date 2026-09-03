//! Validation-only benchmark for a unified checkpoint trained on harmonized intrinsic RT.
//!
//! Predictions are reported both in the common latent RT coordinate and after inversion back to
//! each source's original normalized-RT coordinate using the TRAIN-fit source transform.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig, FoundationPartition,
    FoundationTargetNormalizationConfig, PeptideFoundationUnifiedModel, RetentionTimeObjective,
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
    target_normalization: FoundationTargetNormalizationConfig,
    rt_objective: RetentionTimeObjective,
    rt_harmonization_calibration: Option<String>,
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
        let denominator = (left.max(0.0) * right.max(0.0)).sqrt();
        (denominator > 0.0).then_some(numerator / denominator)
    }
}

#[derive(Debug, Default)]
struct SourceMetrics {
    latent: RegressionAccumulator,
    source_native: RegressionAccumulator,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 7 {
        bail!(
            "usage: foundation_benchmark_harmonized_rt HARMONIZED_RUN.yaml UNIFIED_CHECKPOINT OUTPUT_PREFIX [max_records_per_source=512] [batch_size=32] [seed=20260913]"
        );
    }
    let run_path = PathBuf::from(&args[1]);
    let checkpoint = PathBuf::from(&args[2]);
    let output_prefix = PathBuf::from(&args[3]);
    let max_per_source = parse_or(&args, 4, 512usize)?;
    let batch_size = parse_or(&args, 5, 32usize)?;
    let seed = parse_or(&args, 6, 20_260_913u64)?;
    if max_per_source == 0 || batch_size == 0 {
        bail!("max_records_per_source and batch_size must be positive");
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(&run_path)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;

    let metadata_path = checkpoint.join("metadata.yaml");
    let metadata: UnifiedMetadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read {metadata_path:?}"))?,
    )?;
    if metadata.rt_objective != RetentionTimeObjective::Harmonized {
        bail!("checkpoint does not declare the Harmonized RT objective");
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationUnifiedModel::new(
        metadata.forward_config.clone(),
        metadata.inverse_config.clone(),
        vb,
    )?;
    varmap
        .load(checkpoint.join("model.safetensors"))
        .with_context(|| format!("failed to load unified checkpoint {checkpoint:?}"))?;
    let collator = FoundationCollator::new(
        metadata.forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: RetentionTimeObjective::Harmonized,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;

    let transforms: BTreeMap<
        String,
        redeem_properties::foundation::FoundationRtHarmonizationTransform,
    > = run
        .corpus
        .sources
        .iter()
        .filter_map(|source| {
            source
                .rt_harmonization
                .as_ref()
                .map(|transform| (source.id.clone(), transform.clone()))
        })
        .collect();
    if transforms.is_empty() {
        bail!("harmonized run contains no source RT transforms");
    }

    let selected =
        select_source_balanced_indices(&benchmark, &corpus.provenance, max_per_source, seed);
    if selected.is_empty() {
        bail!("no harmonized RT validation records selected");
    }

    let predictions_path = output_prefix.with_extension("predictions.tsv");
    let summary_path = output_prefix.with_extension("summary.tsv");
    if let Some(parent) = predictions_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut predictions = BufWriter::new(File::create(&predictions_path)?);
    writeln!(
        predictions,
        "record_index\tsource_id\tsource_record_index\tsequence\tpeptidoform\ttarget_source_rt\ttarget_harmonized_rt\tpredicted_harmonized_rt\tharmonized_error\tpredicted_source_rt\tsource_error"
    )?;

    let mut metrics = BTreeMap::<String, SourceMetrics>::new();
    let mut global = SourceMetrics::default();
    for chunk in selected.chunks(batch_size) {
        let records = chunk
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&records, &device, seed ^ chunk[0] as u64)?;
        let output = model
            .forward()
            .forward_t(&batch.input, &batch.context, false)?;
        let latent = metadata
            .target_normalization
            .rt
            .denormalize_tensor(&output.rt)?
            .squeeze(1)?
            .to_vec1::<f32>()?;

        for (local, (&index, record)) in chunk.iter().zip(&records).enumerate() {
            let provenance = &corpus.provenance[index];
            let Some(target_raw) = record
                .retention_time
                .normalized
                .filter(|value| value.is_finite())
            else {
                continue;
            };
            let Some(target_latent) = record
                .retention_time
                .harmonized
                .filter(|value| value.is_finite())
            else {
                continue;
            };
            let transform = transforms.get(&provenance.source_id).ok_or_else(|| {
                anyhow::anyhow!("missing RT transform for source '{}'", provenance.source_id)
            })?;
            let predicted_latent = latent[local];
            let predicted_raw = transform.source_native(predicted_latent).ok_or_else(|| {
                anyhow::anyhow!(
                    "non-finite inverse RT transform for source '{}'",
                    provenance.source_id
                )
            })?;

            let source_metrics = metrics.entry(provenance.source_id.clone()).or_default();
            source_metrics
                .latent
                .push(f64::from(target_latent), f64::from(predicted_latent));
            source_metrics
                .source_native
                .push(f64::from(target_raw), f64::from(predicted_raw));
            global
                .latent
                .push(f64::from(target_latent), f64::from(predicted_latent));
            global
                .source_native
                .push(f64::from(target_raw), f64::from(predicted_raw));

            writeln!(
                predictions,
                "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
                index,
                escape_tsv(&provenance.source_id),
                provenance.source_record_index,
                record.peptidoform.sequence,
                escape_tsv(&format_peptidoform(record)),
                target_raw,
                target_latent,
                predicted_latent,
                predicted_latent - target_latent,
                predicted_raw,
                predicted_raw - target_raw,
            )?;
        }
    }
    predictions.flush()?;

    let mut summary = BufWriter::new(File::create(&summary_path)?);
    writeln!(
        summary,
        "source_id\tn\tlatent_mae\tlatent_rmse\tlatent_pearson\tsource_native_mae\tsource_native_rmse\tsource_native_pearson"
    )?;
    write_summary(&mut summary, "__GLOBAL__", &global)?;
    for (source, values) in &metrics {
        write_summary(&mut summary, source, values)?;
    }
    summary.flush()?;

    println!("partition\tValidation");
    println!("selected_records\t{}", selected.len());
    println!("checkpoint_completed_steps\t{}", metadata.completed_steps);
    println!(
        "rt_harmonization_calibration\t{}",
        metadata
            .rt_harmonization_calibration
            .as_deref()
            .unwrap_or("none")
    );
    println!("latent_rt_mae\t{}", fmt_opt(global.latent.mae()));
    println!("latent_rt_rmse\t{}", fmt_opt(global.latent.rmse()));
    println!("latent_rt_pearson\t{}", fmt_opt(global.latent.pearson()));
    println!(
        "source_native_rt_mae\t{}",
        fmt_opt(global.source_native.mae())
    );
    println!(
        "source_native_rt_rmse\t{}",
        fmt_opt(global.source_native.rmse())
    );
    println!(
        "source_native_rt_pearson\t{}",
        fmt_opt(global.source_native.pearson())
    );
    println!("predictions\t{}", predictions_path.display());
    println!("summary\t{}", summary_path.display());
    println!("test_partition_consumed\tNO");
    Ok(())
}

fn write_summary(
    writer: &mut BufWriter<File>,
    source: &str,
    metrics: &SourceMetrics,
) -> Result<()> {
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        source,
        metrics.latent.n,
        fmt_opt(metrics.latent.mae()),
        fmt_opt(metrics.latent.rmse()),
        fmt_opt(metrics.latent.pearson()),
        fmt_opt(metrics.source_native.mae()),
        fmt_opt(metrics.source_native.rmse()),
        fmt_opt(metrics.source_native.pearson()),
    )?;
    Ok(())
}

fn select_source_balanced_indices(
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
        if let Some(provenance) = provenance.get(entry.record_index) {
            by_source
                .entry(provenance.source_id.clone())
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

fn format_peptidoform(record: &redeem_properties::foundation::FoundationTrainingRecord) -> String {
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
