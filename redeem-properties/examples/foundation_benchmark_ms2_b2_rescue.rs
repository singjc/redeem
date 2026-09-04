//! Validation-only three-state audit for the v0.13.9 b²/channel-1 MS2 rescue.
//!
//! This compares, on exactly the same source-balanced VALIDATION records:
//! 1. the accepted v0.13.8 Softplus checkpoint;
//! 2. the v0.13.9 zero-step checkpoint after resetting only b²/channel 1;
//! 3. the trained v0.13.9 final checkpoint.
//!
//! TEST is intentionally not exposed by this utility.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationMs2OutputActivation, FoundationPartition, FoundationTrainingRecord,
    PeptideFoundationUnifiedModel,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const NEAR_ZERO_THRESHOLDS: [f64; 3] = [1.0e-4, 1.0e-3, 1.0e-2];

#[derive(Debug, Deserialize)]
struct UnifiedMetadata {
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: redeem_properties::foundation::FoundationDiffusionConfig,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    #[serde(default)]
    ms2_head_reset: Option<String>,
    #[serde(default)]
    ms2_head_reset_channel: Option<usize>,
    #[serde(default)]
    ms2_head_fingerprint_before_reset: Option<String>,
    #[serde(default)]
    ms2_head_fingerprint_after_reset: Option<String>,
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

#[derive(Debug, Clone, Copy, Default)]
struct RecordMetrics {
    fragments: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    cosine: f64,
    spectral_angle: f64,
    pearson: Option<f64>,
    exact_zero_count: usize,
    near_zero_counts: [usize; 3],
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

    fn near_zero_fraction(self, threshold_index: usize) -> Option<f64> {
        (self.fragments > 0)
            .then(|| self.near_zero_counts[threshold_index] as f64 / self.fragments as f64)
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
    near_zero_counts: [usize; 3],
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
        for index in 0..self.near_zero_counts.len() {
            self.near_zero_counts[index] += metrics.near_zero_counts[index];
        }
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

    fn near_zero_fraction(&self, threshold_index: usize) -> Option<f64> {
        (self.fragments > 0)
            .then(|| self.near_zero_counts[threshold_index] as f64 / self.fragments as f64)
    }

    fn mean_prediction(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.predicted_sum / self.fragments as f64)
    }

    fn mean_target(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.target_sum / self.fragments as f64)
    }
}

#[derive(Debug, Default, Clone)]
struct ChannelAccumulator {
    fragments: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    exact_zero_count: usize,
    near_zero_counts: [usize; 3],
    predicted_sum: f64,
    target_sum: f64,
}

impl ChannelAccumulator {
    fn push(&mut self, target: f64, prediction: f64) {
        let error = prediction - target;
        self.fragments += 1;
        self.squared_error_sum += error * error;
        self.absolute_error_sum += error.abs();
        self.exact_zero_count += usize::from(prediction == 0.0);
        for (index, threshold) in NEAR_ZERO_THRESHOLDS.iter().enumerate() {
            self.near_zero_counts[index] += usize::from(prediction <= *threshold);
        }
        self.predicted_sum += prediction;
        self.target_sum += target;
    }

    fn mse(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.squared_error_sum / self.fragments as f64)
    }

    fn mae(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.absolute_error_sum / self.fragments as f64)
    }

    fn exact_zero_fraction(&self) -> Option<f64> {
        (self.fragments > 0).then(|| self.exact_zero_count as f64 / self.fragments as f64)
    }

    fn near_zero_fraction(&self, threshold_index: usize) -> Option<f64> {
        (self.fragments > 0)
            .then(|| self.near_zero_counts[threshold_index] as f64 / self.fragments as f64)
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
    exact_zero_reduced: usize,
    near_zero_reduced: [usize; 3],
    cosine_delta_sum: f64,
    angle_delta_sum: f64,
    mse_delta_sum: f64,
    exact_zero_delta_sum: f64,
    near_zero_delta_sum: [f64; 3],
}

impl PairedAccumulator {
    fn push(&mut self, before: RecordMetrics, after: RecordMetrics) {
        if before.fragments == 0 || after.fragments == 0 {
            return;
        }
        self.records += 1;
        let cosine_delta = after.cosine - before.cosine;
        let angle_delta = after.spectral_angle - before.spectral_angle;
        self.cosine_delta_sum += cosine_delta;
        self.angle_delta_sum += angle_delta;
        self.cosine_improved += usize::from(cosine_delta > 0.0);
        self.angle_improved += usize::from(angle_delta > 0.0);
        if let (Some(before_pearson), Some(after_pearson)) = (before.pearson, after.pearson) {
            self.pearson_pairs += 1;
            self.pearson_improved += usize::from(after_pearson > before_pearson);
        }
        if let (Some(before_mse), Some(after_mse)) = (before.mse(), after.mse()) {
            self.mse_delta_sum += after_mse - before_mse;
            self.mse_improved += usize::from(after_mse < before_mse);
        }
        if let (Some(before_zero), Some(after_zero)) =
            (before.exact_zero_fraction(), after.exact_zero_fraction())
        {
            self.exact_zero_delta_sum += after_zero - before_zero;
            self.exact_zero_reduced += usize::from(after_zero < before_zero);
        }
        for threshold_index in 0..NEAR_ZERO_THRESHOLDS.len() {
            if let (Some(before_zero), Some(after_zero)) = (
                before.near_zero_fraction(threshold_index),
                after.near_zero_fraction(threshold_index),
            ) {
                self.near_zero_delta_sum[threshold_index] += after_zero - before_zero;
                self.near_zero_reduced[threshold_index] += usize::from(after_zero < before_zero);
            }
        }
    }
}

struct EvaluationWriters<'a> {
    records: &'a mut BufWriter<File>,
    fragments: &'a mut BufWriter<File>,
}

#[derive(Default)]
struct EvaluationResult {
    record_metrics: BTreeMap<usize, RecordMetrics>,
    overall: SummaryAccumulator,
    source_summaries: BTreeMap<String, SummaryAccumulator>,
    family_summaries: BTreeMap<String, SummaryAccumulator>,
    channel_summaries: BTreeMap<usize, ChannelAccumulator>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_checkpoint(
    label: &str,
    checkpoint: &Path,
    expected_corpus_fingerprint: &str,
    expected_benchmark_fingerprint: &str,
    selected: &[usize],
    records: &[FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    collator: &FoundationCollator,
    batch_size: usize,
    seed: u64,
    writers: &mut EvaluationWriters<'_>,
    device: &Device,
) -> Result<(EvaluationResult, UnifiedMetadata)> {
    let (predictor, metadata) = UnifiedPredictor::load(checkpoint, device)?;
    if metadata.corpus_fingerprint != expected_corpus_fingerprint {
        bail!(
            "{label} checkpoint corpus fingerprint {} does not match loaded corpus {}",
            metadata.corpus_fingerprint,
            expected_corpus_fingerprint
        );
    }
    if metadata.benchmark_manifest_fingerprint != expected_benchmark_fingerprint {
        bail!(
            "{label} checkpoint benchmark fingerprint {} does not match loaded benchmark {}",
            metadata.benchmark_manifest_fingerprint,
            expected_benchmark_fingerprint
        );
    }

    let mut result = EvaluationResult::default();
    for chunk in selected.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|&index| records[index].clone())
            .collect::<Vec<_>>();
        let batch = collator.collate(&owned, device, seed ^ chunk[0] as u64)?;
        let output = predictor
            .model
            .forward()
            .forward_t(&batch.input, &batch.context, false)?;
        let ms2 = output.ms2.to_vec3::<f32>()?;

        for (local, (&record_index, record)) in chunk.iter().zip(&owned).enumerate() {
            let source = provenance[record_index].source_id.as_str();
            let family = source_family(source);
            let metrics = record_metrics(record, &ms2[local]);
            result.record_metrics.insert(record_index, metrics);
            result.overall.push(metrics);
            result
                .source_summaries
                .entry(source.to_string())
                .or_default()
                .push(metrics);
            result
                .family_summaries
                .entry(family.to_string())
                .or_default()
                .push(metrics);
            accumulate_channels(&mut result.channel_summaries, record, &ms2[local]);
            write_record(
                writers.records,
                label,
                record_index,
                source,
                family,
                record,
                metrics,
            )?;
            write_fragments(
                writers.fragments,
                label,
                record_index,
                source,
                family,
                record,
                &ms2[local],
            )?;
        }
    }
    Ok((result, metadata))
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(5..=8).contains(&args.len()) {
        bail!(
            "usage: foundation_benchmark_ms2_b2_rescue <training.yaml> <reference_softplus_checkpoint> <candidate_initial_checkpoint> <candidate_final_checkpoint> <output_dir> [max_records_per_source=512] [batch_size=32] [seed=20260912]"
        );
    }

    let training_yaml = PathBuf::from(&args[0]);
    let reference_checkpoint = PathBuf::from(&args[1]);
    let initial_checkpoint = PathBuf::from(&args[2]);
    let final_checkpoint = PathBuf::from(&args[3]);
    let output_dir = PathBuf::from(&args[4]);
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
    let selected = select_source_balanced_validation_indices(
        &benchmark,
        &corpus.provenance,
        max_per_source,
        seed,
    );
    if selected.is_empty() {
        bail!("no VALIDATION records selected for MS2 b2-rescue audit");
    }

    let reference_metadata = read_unified_metadata(&reference_checkpoint)?;
    let initial_metadata = read_unified_metadata(&initial_checkpoint)?;
    let final_metadata = read_unified_metadata(&final_checkpoint)?;
    if reference_metadata.forward_config.ms2_output_activation
        != FoundationMs2OutputActivation::SoftplusV0138
        || initial_metadata.forward_config.ms2_output_activation
            != FoundationMs2OutputActivation::SoftplusV0138
        || final_metadata.forward_config.ms2_output_activation
            != FoundationMs2OutputActivation::SoftplusV0138
    {
        bail!("reference and candidate checkpoints must all use softplus-v0138 MS2 activation");
    }
    for (label, metadata) in [
        ("candidate initial", &initial_metadata),
        ("candidate final", &final_metadata),
    ] {
        if metadata.ms2_head_reset.as_deref() != Some("b2-zero-v0139")
            || metadata.ms2_head_reset_channel != Some(1)
        {
            bail!(
                "{label} checkpoint is not marked as the controlled b2-zero-v0139 channel-1 reset"
            );
        }
        if metadata.ms2_head_fingerprint_before_reset.is_none()
            || metadata.ms2_head_fingerprint_after_reset.is_none()
        {
            bail!("{label} checkpoint is missing v0.13.9 MS2-head reset fingerprints");
        }
    }
    if initial_metadata.ms2_head_fingerprint_before_reset
        != final_metadata.ms2_head_fingerprint_before_reset
        || initial_metadata.ms2_head_fingerprint_after_reset
            != final_metadata.ms2_head_fingerprint_after_reset
    {
        bail!("candidate initial/final checkpoints do not describe the same b2 reset event");
    }
    if !reference_metadata
        .forward_config
        .parameter_compatible_with(&initial_metadata.forward_config)
        || !reference_metadata
            .forward_config
            .parameter_compatible_with(&final_metadata.forward_config)
    {
        bail!("reference and candidate checkpoints do not share one forward parameterization");
    }
    if initial_metadata.forward_config != final_metadata.forward_config
        || reference_metadata.inverse_config != initial_metadata.inverse_config
        || initial_metadata.inverse_config != final_metadata.inverse_config
    {
        bail!("candidate initial/final config or shared inverse architecture does not match");
    }
    if initial_metadata.completed_steps != 0 {
        bail!(
            "candidate initial checkpoint must be zero-step, got {}",
            initial_metadata.completed_steps
        );
    }

    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
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
    let records_path = output_dir.join("ms2_b2_rescue_records.tsv");
    let fragments_path = output_dir.join("ms2_b2_rescue_fragments.tsv");
    let source_summary_path = output_dir.join("ms2_b2_rescue_source_summary.tsv");
    let family_summary_path = output_dir.join("ms2_b2_rescue_family_summary.tsv");
    let channel_summary_path = output_dir.join("ms2_b2_rescue_channel_summary.tsv");
    let head_summary_path = output_dir.join("ms2_b2_rescue_head_summary.tsv");
    let paired_summary_path = output_dir.join("ms2_b2_rescue_paired_summary.tsv");
    let mut records_writer = BufWriter::new(File::create(&records_path)?);
    let mut fragments_writer = BufWriter::new(File::create(&fragments_path)?);
    writeln!(
        records_writer,
        "checkpoint\trecord_index\tsource_id\tsource_family\tsequence\tpeptidoform\tfragments\tpointwise_mse\tpointwise_mae\tcosine\tspectral_angle\tpearson\texact_zero_fraction\tnear_zero_1e4_fraction\tnear_zero_1e3_fraction\tnear_zero_1e2_fraction\tmean_target_intensity\tmean_predicted_intensity"
    )?;
    writeln!(
        fragments_writer,
        "checkpoint\trecord_index\tsource_id\tsource_family\tsequence\tpeptidoform\tcleavage_index\tchannel\tion_label\tproduct_mz\ttarget_intensity\tpredicted_intensity"
    )?;
    let mut writers = EvaluationWriters {
        records: &mut records_writer,
        fragments: &mut fragments_writer,
    };

    let states = [
        ("reference_softplus_v0138", &reference_checkpoint),
        ("initial_b2_reset_v0139", &initial_checkpoint),
        ("final_b2_rescue_v0139", &final_checkpoint),
    ];
    let mut evaluations = BTreeMap::<String, EvaluationResult>::new();
    let mut metadata_by_label = BTreeMap::<String, UnifiedMetadata>::new();
    for (label, checkpoint) in states {
        let (evaluation, metadata) = evaluate_checkpoint(
            label,
            checkpoint,
            &corpus_fingerprint,
            &benchmark_fingerprint,
            &selected,
            &corpus.records,
            &corpus.provenance,
            &collator,
            batch_size,
            seed,
            &mut writers,
            &device,
        )?;
        evaluations.insert(label.to_string(), evaluation);
        metadata_by_label.insert(label.to_string(), metadata);
    }
    writers.records.flush()?;
    writers.fragments.flush()?;

    write_group_summaries(
        &source_summary_path,
        "source_id",
        &evaluations,
        Grouping::Source,
    )?;
    write_group_summaries(
        &family_summary_path,
        "source_family",
        &evaluations,
        Grouping::Family,
    )?;
    write_channel_summaries(&channel_summary_path, &evaluations)?;
    write_head_parameter_summaries(
        &head_summary_path,
        &[
            ("reference_softplus_v0138", &reference_checkpoint),
            ("initial_b2_reset_v0139", &initial_checkpoint),
            ("final_b2_rescue_v0139", &final_checkpoint),
        ],
        &device,
    )?;

    let comparisons = [
        (
            "reset_effect",
            "reference_softplus_v0138",
            "initial_b2_reset_v0139",
        ),
        (
            "training_effect",
            "initial_b2_reset_v0139",
            "final_b2_rescue_v0139",
        ),
        (
            "net_effect",
            "reference_softplus_v0138",
            "final_b2_rescue_v0139",
        ),
    ];
    let paired = comparisons
        .iter()
        .map(|(label, before, after)| {
            Ok((
                (*label).to_string(),
                paired_metrics(
                    &evaluations
                        .get(*before)
                        .expect("paired before checkpoint must exist")
                        .record_metrics,
                    &evaluations
                        .get(*after)
                        .expect("paired after checkpoint must exist")
                        .record_metrics,
                )?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    write_paired_summary(&paired_summary_path, &paired)?;

    println!("partition\tValidation");
    println!("test_partition_consumed\tNO");
    println!("corpus_fingerprint\t{corpus_fingerprint}");
    println!("benchmark_manifest_fingerprint\t{benchmark_fingerprint}");
    println!("selected_records\t{}", selected.len());
    println!("max_records_per_source\t{max_per_source}");
    println!(
        "b2_reset\talgorithm=b2-zero-v0139\tchannel=1\tfingerprint_before={}\tfingerprint_after={}",
        initial_metadata
            .ms2_head_fingerprint_before_reset
            .as_deref()
            .unwrap_or("NA"),
        initial_metadata
            .ms2_head_fingerprint_after_reset
            .as_deref()
            .unwrap_or("NA"),
    );
    for label in [
        "reference_softplus_v0138",
        "initial_b2_reset_v0139",
        "final_b2_rescue_v0139",
    ] {
        let metadata = metadata_by_label
            .get(label)
            .expect("checkpoint metadata must exist");
        let evaluation = evaluations
            .get(label)
            .expect("checkpoint evaluation must exist");
        let summary = &evaluation.overall;
        println!(
            "ms2_b2_rescue_summary\tcheckpoint={label}\tactivation={}\tcompleted_steps={}\trecords={}\tfragments={}\tpointwise_mse={}\tpointwise_mae={}\tcosine={}\tspectral_angle={}\tpearson={}\texact_zero_fraction={}\tnear_zero_1e4_fraction={}\tnear_zero_1e3_fraction={}\tnear_zero_1e2_fraction={}\tmean_target_intensity={}\tmean_predicted_intensity={}",
            metadata.forward_config.ms2_output_activation.as_str(),
            metadata.completed_steps,
            summary.records,
            summary.fragments,
            fmt_opt(summary.mse()),
            fmt_opt(summary.mae()),
            fmt_opt(summary.mean_cosine()),
            fmt_opt(summary.mean_spectral_angle()),
            fmt_opt(summary.mean_pearson()),
            fmt_opt(summary.exact_zero_fraction()),
            fmt_opt(summary.near_zero_fraction(0)),
            fmt_opt(summary.near_zero_fraction(1)),
            fmt_opt(summary.near_zero_fraction(2)),
            fmt_opt(summary.mean_target()),
            fmt_opt(summary.mean_prediction()),
        );
        for channel in 0..4 {
            if let Some(summary) = evaluation.channel_summaries.get(&channel) {
                print_channel_summary(label, channel, summary);
            }
        }
    }
    for comparison in ["reset_effect", "training_effect", "net_effect"] {
        print_paired_summary(
            comparison,
            paired
                .get(comparison)
                .expect("paired comparison must exist"),
        );
    }
    println!("records_tsv\t{}", records_path.display());
    println!("fragments_tsv\t{}", fragments_path.display());
    println!("source_summary_tsv\t{}", source_summary_path.display());
    println!("family_summary_tsv\t{}", family_summary_path.display());
    println!("channel_summary_tsv\t{}", channel_summary_path.display());
    println!("head_summary_tsv\t{}", head_summary_path.display());
    println!("paired_summary_tsv\t{}", paired_summary_path.display());
    Ok(())
}

fn write_head_parameter_summaries(
    path: &Path,
    checkpoints: &[(&str, &PathBuf)],
    device: &Device,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "checkpoint\tchannel\tion_family\tweight_l2\tweight_mean_abs\tbias"
    )?;
    for (label, checkpoint) in checkpoints {
        let tensors = candle_core::safetensors::load(&checkpoint.join("model.safetensors"), device)
            .with_context(|| format!("failed to load MS2 head parameters from {checkpoint:?}"))?;
        let weights = tensors
            .get("heads.ms2.weight")
            .ok_or_else(|| anyhow::anyhow!("missing heads.ms2.weight in {checkpoint:?}"))?
            .to_vec2::<f32>()?;
        let biases = tensors
            .get("heads.ms2.bias")
            .ok_or_else(|| anyhow::anyhow!("missing heads.ms2.bias in {checkpoint:?}"))?
            .to_vec1::<f32>()?;
        if weights.len() != biases.len() {
            bail!(
                "MS2 head weight/bias output dimensions disagree in {checkpoint:?}: {} vs {}",
                weights.len(),
                biases.len()
            );
        }
        for channel in 0..weights.len().min(4) {
            let row = &weights[channel];
            let weight_l2 = row
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            let weight_mean_abs = if row.is_empty() {
                0.0
            } else {
                row.iter().map(|value| f64::from(*value).abs()).sum::<f64>() / row.len() as f64
            };
            writeln!(
                writer,
                "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}",
                label,
                channel,
                channel_family(channel),
                weight_l2,
                weight_mean_abs,
                biases[channel],
            )?;
            println!(
                "ms2_head_row_summary\tcheckpoint={}\tchannel={}\tion_family={}\tweight_l2={:.8}\tweight_mean_abs={:.8}\tbias={:.8}",
                label,
                channel,
                channel_family(channel),
                weight_l2,
                weight_mean_abs,
                biases[channel],
            );
        }
    }
    writer.flush()?;
    Ok(())
}

fn read_unified_metadata(checkpoint: &Path) -> Result<UnifiedMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse {path:?}"))
}

fn record_metrics(record: &FoundationTrainingRecord, predicted: &[Vec<f32>]) -> RecordMetrics {
    let mut targets = Vec::new();
    let mut predictions = Vec::new();
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
        return RecordMetrics::default();
    }

    let mut result = RecordMetrics {
        fragments: targets.len(),
        ..RecordMetrics::default()
    };
    for (&target, &prediction) in targets.iter().zip(&predictions) {
        let error = prediction - target;
        result.squared_error_sum += error * error;
        result.absolute_error_sum += error.abs();
        result.exact_zero_count += usize::from(prediction == 0.0);
        for (index, threshold) in NEAR_ZERO_THRESHOLDS.iter().enumerate() {
            result.near_zero_counts[index] += usize::from(prediction <= *threshold);
        }
        result.predicted_sum += prediction;
        result.target_sum += target;
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
    result.cosine = if target_norm > 0.0 && prediction_norm > 0.0 {
        (dot / (target_norm * prediction_norm)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    result.spectral_angle = 1.0 - 2.0 * result.cosine.acos() / std::f64::consts::PI;
    result.pearson = pearson(&targets, &predictions);
    result
}

fn accumulate_channels(
    summaries: &mut BTreeMap<usize, ChannelAccumulator>,
    record: &FoundationTrainingRecord,
    predicted: &[Vec<f32>],
) {
    for fragment in &record.fragments {
        let Some(&prediction) = predicted
            .get(fragment.cleavage_index)
            .and_then(|row| row.get(fragment.channel))
        else {
            continue;
        };
        if fragment.intensity.is_finite() && prediction.is_finite() {
            summaries
                .entry(fragment.channel)
                .or_default()
                .push(f64::from(fragment.intensity), f64::from(prediction));
        }
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
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
        fmt_opt(metrics.near_zero_fraction(0)),
        fmt_opt(metrics.near_zero_fraction(1)),
        fmt_opt(metrics.near_zero_fraction(2)),
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

#[derive(Clone, Copy)]
enum Grouping {
    Source,
    Family,
}

fn write_group_summaries(
    path: &Path,
    grouping_header: &str,
    evaluations: &BTreeMap<String, EvaluationResult>,
    grouping: Grouping,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "checkpoint\t{}\trecords\tfragments\tpointwise_mse\tpointwise_mae\tmean_cosine\tmean_spectral_angle\tmean_pearson\texact_zero_fraction\tnear_zero_1e4_fraction\tnear_zero_1e3_fraction\tnear_zero_1e2_fraction\tmean_target_intensity\tmean_predicted_intensity",
        grouping_header
    )?;
    for (checkpoint, evaluation) in evaluations {
        let groups = match grouping {
            Grouping::Source => &evaluation.source_summaries,
            Grouping::Family => &evaluation.family_summaries,
        };
        for (group, summary) in groups {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
                fmt_opt(summary.near_zero_fraction(0)),
                fmt_opt(summary.near_zero_fraction(1)),
                fmt_opt(summary.near_zero_fraction(2)),
                fmt_opt(summary.mean_target()),
                fmt_opt(summary.mean_prediction()),
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_channel_summaries(
    path: &Path,
    evaluations: &BTreeMap<String, EvaluationResult>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "checkpoint\tchannel\tion_family\tfragments\tpointwise_mse\tpointwise_mae\texact_zero_fraction\tnear_zero_1e4_fraction\tnear_zero_1e3_fraction\tnear_zero_1e2_fraction\tmean_target_intensity\tmean_predicted_intensity"
    )?;
    for (checkpoint, evaluation) in evaluations {
        for (&channel, summary) in &evaluation.channel_summaries {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                checkpoint,
                channel,
                channel_family(channel),
                summary.fragments,
                fmt_opt(summary.mse()),
                fmt_opt(summary.mae()),
                fmt_opt(summary.exact_zero_fraction()),
                fmt_opt(summary.near_zero_fraction(0)),
                fmt_opt(summary.near_zero_fraction(1)),
                fmt_opt(summary.near_zero_fraction(2)),
                fmt_opt(summary.mean_target()),
                fmt_opt(summary.mean_prediction()),
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn paired_metrics(
    before: &BTreeMap<usize, RecordMetrics>,
    after: &BTreeMap<usize, RecordMetrics>,
) -> Result<PairedAccumulator> {
    if before.len() != after.len() || before.keys().ne(after.keys()) {
        bail!("paired MS2 activation evaluations do not contain identical validation records");
    }
    let mut paired = PairedAccumulator::default();
    for (record_index, before_metrics) in before {
        paired.push(
            *before_metrics,
            *after
                .get(record_index)
                .expect("paired record must exist in after evaluation"),
        );
    }
    Ok(paired)
}

fn write_paired_summary(
    path: &Path,
    comparisons: &BTreeMap<String, PairedAccumulator>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "comparison\trecords\tcosine_improved_fraction\tmean_cosine_delta\tspectral_angle_improved_fraction\tmean_spectral_angle_delta\tpearson_pairs\tpearson_improved_fraction\tmse_improved_fraction\tmean_mse_delta\texact_zero_reduced_fraction\tmean_exact_zero_delta\tnear_zero_1e4_reduced_fraction\tmean_near_zero_1e4_delta\tnear_zero_1e3_reduced_fraction\tmean_near_zero_1e3_delta\tnear_zero_1e2_reduced_fraction\tmean_near_zero_1e2_delta"
    )?;
    for (comparison, paired) in comparisons {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            comparison,
            paired.records,
            fmt_ratio(paired.cosine_improved as f64, paired.records),
            fmt_ratio(paired.cosine_delta_sum, paired.records),
            fmt_ratio(paired.angle_improved as f64, paired.records),
            fmt_ratio(paired.angle_delta_sum, paired.records),
            paired.pearson_pairs,
            fmt_ratio(paired.pearson_improved as f64, paired.pearson_pairs),
            fmt_ratio(paired.mse_improved as f64, paired.records),
            fmt_ratio(paired.mse_delta_sum, paired.records),
            fmt_ratio(paired.exact_zero_reduced as f64, paired.records),
            fmt_ratio(paired.exact_zero_delta_sum, paired.records),
            fmt_ratio(paired.near_zero_reduced[0] as f64, paired.records),
            fmt_ratio(paired.near_zero_delta_sum[0], paired.records),
            fmt_ratio(paired.near_zero_reduced[1] as f64, paired.records),
            fmt_ratio(paired.near_zero_delta_sum[1], paired.records),
            fmt_ratio(paired.near_zero_reduced[2] as f64, paired.records),
            fmt_ratio(paired.near_zero_delta_sum[2], paired.records),
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn print_channel_summary(checkpoint: &str, channel: usize, summary: &ChannelAccumulator) {
    println!(
        "ms2_channel_summary\tcheckpoint={checkpoint}\tchannel={channel}\tion_family={}\tfragments={}\tpointwise_mse={}\tpointwise_mae={}\texact_zero_fraction={}\tnear_zero_1e4_fraction={}\tnear_zero_1e3_fraction={}\tnear_zero_1e2_fraction={}\tmean_target_intensity={}\tmean_predicted_intensity={}",
        channel_family(channel),
        summary.fragments,
        fmt_opt(summary.mse()),
        fmt_opt(summary.mae()),
        fmt_opt(summary.exact_zero_fraction()),
        fmt_opt(summary.near_zero_fraction(0)),
        fmt_opt(summary.near_zero_fraction(1)),
        fmt_opt(summary.near_zero_fraction(2)),
        fmt_opt(summary.mean_target()),
        fmt_opt(summary.mean_prediction()),
    );
}

fn print_paired_summary(comparison: &str, paired: &PairedAccumulator) {
    println!(
        "paired_change\tcomparison={comparison}\trecords={}\tcosine_improved_fraction={}\tmean_cosine_delta={}\tspectral_angle_improved_fraction={}\tmean_spectral_angle_delta={}\tpearson_improved_fraction={}\tmse_improved_fraction={}\tmean_mse_delta={}\texact_zero_reduced_fraction={}\tmean_exact_zero_delta={}\tnear_zero_1e4_reduced_fraction={}\tmean_near_zero_1e4_delta={}\tnear_zero_1e3_reduced_fraction={}\tmean_near_zero_1e3_delta={}\tnear_zero_1e2_reduced_fraction={}\tmean_near_zero_1e2_delta={}",
        paired.records,
        fmt_ratio(paired.cosine_improved as f64, paired.records),
        fmt_ratio(paired.cosine_delta_sum, paired.records),
        fmt_ratio(paired.angle_improved as f64, paired.records),
        fmt_ratio(paired.angle_delta_sum, paired.records),
        fmt_ratio(paired.pearson_improved as f64, paired.pearson_pairs),
        fmt_ratio(paired.mse_improved as f64, paired.records),
        fmt_ratio(paired.mse_delta_sum, paired.records),
        fmt_ratio(paired.exact_zero_reduced as f64, paired.records),
        fmt_ratio(paired.exact_zero_delta_sum, paired.records),
        fmt_ratio(paired.near_zero_reduced[0] as f64, paired.records),
        fmt_ratio(paired.near_zero_delta_sum[0], paired.records),
        fmt_ratio(paired.near_zero_reduced[1] as f64, paired.records),
        fmt_ratio(paired.near_zero_delta_sum[1], paired.records),
        fmt_ratio(paired.near_zero_reduced[2] as f64, paired.records),
        fmt_ratio(paired.near_zero_delta_sum[2], paired.records),
    );
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

fn channel_family(channel: usize) -> &'static str {
    match channel {
        0 => "b^1",
        1 => "b^2",
        2 => "y^1",
        3 => "y^2",
        4 => "b-H2O",
        5 => "y-H2O",
        6 => "b-NH3",
        7 => "y-NH3",
        _ => "other",
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
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.8}"))
        .unwrap_or_else(|| "NA".into())
}

fn escape_tsv(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn parse_or<T>(args: &[String], index: usize, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.get(index)
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|error| anyhow::anyhow!("invalid argument {value:?}: {error}"))
        })
        .unwrap_or(Ok(default))
}
