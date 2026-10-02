//! v0.71 spectrum-conditioned causal peptide decoder.
//!
//! Scientific contract:
//! - use the selected v0.70 observed-spectrum Transformer as a frozen peak-token encoder;
//! - train only the causal decoder namespace against TRAIN;
//! - every causal layer cross-attends to the frozen contextualized v0.70 peak states;
//! - retain the established matched-vs-shuffled spectrum guard and search-legal prefix margin;
//! - select checkpoints on direct mass-constrained DEV generation;
//! - keep RT/CCS out of the decoder input in v0.71 so their value can be tested later as
//!   optional observed evidence on matched cohorts without confounding this representation test;
//! - never access TRAIN-HOLDOUT, historical VALIDATION/APD, or historical TEST.
//!
//! v0.71 deliberately uses the established finite common-PTM causal vocabulary. Open-PTM
//! decoding is a later, separate extension after direct sequence recovery is proven.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_causal_next_token_loss, foundation_direct_beam_search,
    foundation_direct_conditioning_loss, foundation_direct_prefix_competitive_loss,
    foundation_direct_shuffled_order, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    DirectDecoderBeamConfig, FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest,
    FoundationCausalCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationLearningRateSchedule, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumCollator, FoundationTrainingRecord, PeptideSpectrumCausalModel,
    PrecursorContextBatch, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
    FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190, FOUNDATION_DIRECT_PREFIX_MARGIN_V0191,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V071_VERSION: u32 = 710;
const V071_OBJECTIVE: &str =
    "v0710_frozen_v070_peak_tokens_causal_ce_prefix_margin_conditioning_guard";
const V071_ARCHITECTURE: &str =
    "frozen_v070_spectrum_encoder_plus_6layer_causal_cross_attention_decoder";
const V071_CONTEXT_POLICY: &str =
    "spectrum_plus_precursor_only_rt_ccs_reserved_for_matched_ablation";
const V071_DECODER_LAYERS: usize = 6;
const V071_TRAIN_BATCH: usize = 32;
const V071_STEPS_PER_EPOCH: usize = 1000;
const V071_MAX_EPOCHS: usize = 6;
const V071_PATIENCE: usize = 2;
const V071_MIN_DELTA: f64 = 0.002;
const V071_SEED: u64 = 20_261_071;
const V071_LEARNING_RATE: f64 = 1.0e-4;
const V071_WEIGHT_DECAY: f64 = 1.0e-4;
const V071_MAX_GRADIENT_NORM: f64 = 5.0;
const V071_DEV_IDENTITIES: usize = 256;
const V071_SMOKE_DEV_IDENTITIES: usize = 16;
const V071_BEAM_WIDTH: usize = 64;
const V071_SMOKE_BEAM_WIDTH: usize = 8;
const V071_TOP_K: usize = 10;
const V071_MASS_TOLERANCE_DA: f64 = 0.05;

const V071_GATE_MIN_SELECTION_GAIN: f64 = 0.05;
const V071_GATE_MIN_IL_TOP1: f64 = 0.08;
const V071_GATE_MIN_IL_TOP10: f64 = 0.25;
const V071_GATE_MIN_RETURNED_FRACTION: f64 = 0.80;
const V071_GATE_MIN_CONDITIONING_GAP: f64 = 0.02;

#[derive(Debug, Clone, Deserialize)]
struct V070Metadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    config: V070Config,
    completed_epochs: usize,
    completed_updates: usize,
    dev_selection_score: f64,
    smoke_mode: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct V070Config {
    spectrum: FoundationDiffusionConfig,
    alignment_dim: usize,
    temperature: f64,
}

#[derive(Debug, Clone)]
struct IdentityGroup {
    key: String,
    record_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct TeacherMetrics {
    token_nll: f64,
    token_accuracy: f64,
    sequence_exact: f64,
    conditioning_gap: f64,
    legal_rank: f64,
    legal_top1: f64,
    active_tokens: usize,
    sequences: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct GenerationMetrics {
    identities: usize,
    returned_fraction: f64,
    peptidoform_top1: f64,
    peptidoform_top5: f64,
    peptidoform_top10: f64,
    sequence_top1: f64,
    sequence_top5: f64,
    sequence_top10: f64,
    il_top1: f64,
    il_top5: f64,
    il_top10: f64,
    mean_top1_edit_distance: f64,
    mean_top1_normalized_edit_distance: f64,
    mass_valid_fraction: f64,
}

impl GenerationMetrics {
    fn selection_score(self) -> f64 {
        0.7 * self.il_top1 + 0.3 * self.il_top10
    }
}

#[derive(Debug, Clone)]
struct GenerationOutcome {
    charge: i32,
    length: usize,
    modified: bool,
    returned: bool,
    peptidoform_top1: bool,
    peptidoform_top10: bool,
    sequence_top1: bool,
    sequence_top10: bool,
    il_top1: bool,
    il_top10: bool,
    top1_edit_distance: usize,
    top1_normalized_edit_distance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V071Metadata {
    version: u32,
    objective: String,
    architecture: String,
    context_policy: String,
    parent_v070_checkpoint: String,
    parent_v070_completed_epochs: usize,
    parent_v070_completed_updates: usize,
    parent_v070_dev_selection_score: f64,
    parent_v070_spectrum_variables_loaded: usize,
    parent_v070_ignored_variables: usize,
    parent_update_policy: String,
    decoder_update_policy: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    decoder_config: FoundationDiffusionConfig,
    max_epochs: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    patience: usize,
    min_delta: f64,
    seed: u64,
    learning_rate: f64,
    weight_decay: f64,
    dev_identity_count: usize,
    dev_identity_fingerprint: String,
    beam_width: usize,
    top_k: usize,
    mass_tolerance_da: f64,
    completed_epochs: usize,
    completed_updates: usize,
    dev_selection_score: f64,
    step0_selection_score: f64,
    smoke_mode: bool,
}

#[derive(Debug, Clone, Copy)]
struct V070LoadReport {
    spectrum_variables_loaded: usize,
    ignored_checkpoint_variables: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct PropertyAvailability {
    records: usize,
    harmonized_rt: usize,
    ccs: usize,
    both: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct MassFeasibilityAudit {
    basic_eligible_records: usize,
    mass_feasible_records: usize,
    mass_infeasible_records: usize,
    mass_feasible_unique_identities: usize,
    mean_abs_mass_error_da: f64,
    max_abs_mass_error_da: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 12 {
        anyhow::bail!(
            "usage: foundation_train_spectrum_causal_v0710 RUN.yaml OUTPUT_DIR V070_BEST [mode=smoke|train|resume] [max_epochs=6] [batch_size=32] [steps_per_epoch=1000] [patience=2] [min_delta=0.002] [seed=20261071] [learning_rate=1e-4]"
        );
    }
    let run_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let parent_v070 = PathBuf::from(&args[3]);
    let mode = args.get(4).map(String::as_str).unwrap_or("train");
    if !matches!(mode, "smoke" | "train" | "resume") {
        anyhow::bail!("v0.71 mode must be smoke, train, or resume");
    }
    let smoke_mode = mode == "smoke";
    let resume_mode = mode == "resume";
    let requested_max_epochs = parse_or(&args, 5, V071_MAX_EPOCHS)?;
    let batch_size = parse_or(&args, 6, V071_TRAIN_BATCH)?;
    let requested_steps_per_epoch = parse_or(&args, 7, V071_STEPS_PER_EPOCH)?;
    let patience = parse_or(&args, 8, V071_PATIENCE)?;
    let min_delta = parse_or(&args, 9, V071_MIN_DELTA)?;
    let seed = parse_or(&args, 10, V071_SEED)?;
    let learning_rate = parse_or(&args, 11, V071_LEARNING_RATE)?;
    if requested_max_epochs == 0
        || requested_steps_per_epoch == 0
        || batch_size < 2
        || patience == 0
    {
        anyhow::bail!("v0.71 epochs/steps/patience must be positive and batch_size >= 2");
    }
    if !(min_delta >= 0.0 && min_delta.is_finite())
        || !(learning_rate > 0.0 && learning_rate.is_finite())
    {
        anyhow::bail!("v0.71 min_delta/learning_rate are invalid");
    }
    if !resume_mode && output_root.exists() {
        anyhow::bail!("v0.71 output directory must be fresh: {output_root:?}");
    }
    if resume_mode && !output_root.is_dir() {
        anyhow::bail!("v0.71 resume requires an existing output directory: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.71 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(&run_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());

    let parent_metadata = read_v070_metadata(&parent_v070)?;
    validate_v070_parent(
        &parent_metadata,
        &corpus_fingerprint,
        &benchmark_fingerprint,
    )?;
    let parent_model_path = parent_v070.join("model.safetensors");
    if !parent_model_path.is_file() {
        anyhow::bail!("v0.71 parent is missing model.safetensors: {parent_v070:?}");
    }

    let mut decoder_config = parent_metadata.config.spectrum.clone();
    decoder_config.decoder_layers = V071_DECODER_LAYERS;
    decoder_config.validate().map_err(anyhow::Error::msg)?;

    let vocabulary = FoundationDiffusionVocabulary;
    let (train_groups, train_mass_audit) = build_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &decoder_config,
        vocabulary,
    )?;
    let (dev_groups, dev_mass_audit) = build_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &decoder_config,
        vocabulary,
    )?;
    if train_groups.len() < batch_size || dev_groups.is_empty() {
        anyhow::bail!(
            "insufficient v0.71 identities: train={} dev={} batch={batch_size}",
            train_groups.len(),
            dev_groups.len()
        );
    }

    let dev_count = if smoke_mode {
        V071_SMOKE_DEV_IDENTITIES
    } else {
        V071_DEV_IDENTITIES
    }
    .min(dev_groups.len());
    let dev_indices = select_identity_records(
        &dev_groups,
        &corpus.records,
        dev_count,
        seed ^ 0x0710_d3f0_0000_0001,
    );
    let dev_fingerprint = format!("fnv1a64:{:016x}", index_fingerprint(&dev_indices));
    let beam_width = if smoke_mode {
        V071_SMOKE_BEAM_WIDTH
    } else {
        V071_BEAM_WIDTH
    };
    let max_epochs = if smoke_mode { 1 } else { requested_max_epochs };
    let steps_per_epoch = if smoke_mode {
        8usize.min(requested_steps_per_epoch)
    } else {
        requested_steps_per_epoch
    };

    let mut varmap = VarMap::new();
    let model = PeptideSpectrumCausalModel::new(
        decoder_config.clone(),
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
    )?;
    let load_report = load_v070_spectrum_encoder(&varmap, &parent_model_path, &device)?;
    let frozen_spectrum_checksum_initial = spectrum_encoder_checksum(&varmap)?;

    let causal_collator = FoundationCausalCollator::new(decoder_config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(decoder_config.spectrum.clone())?;
    let mut optimizer = FoundationAdamW::new_for_prefixes(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            weight_decay: V071_WEIGHT_DECAY,
            ..FoundationAdamWConfig::default()
        },
        &["decoder."],
    )?;

    let max_updates = max_epochs.saturating_mul(steps_per_epoch).max(1);
    let schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 250u64.min(max_updates.saturating_sub(1) as u64),
        total_steps: max_updates as u64,
        min_lr_ratio: 0.10,
    };

    if !resume_mode {
        fs::create_dir_all(&output_root)?;
    }

    let train_record_indices = train_groups
        .iter()
        .flat_map(|group| group.record_indices.iter().copied())
        .collect::<Vec<_>>();
    let train_availability = property_availability(&corpus.records, &train_record_indices);
    let dev_availability = property_availability(&corpus.records, &dev_indices);
    let holdout_reserved = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .count();

    println!("v0710_version\tv0.71-spectrum-conditioned-causal-decoder");
    println!("objective\t{V071_OBJECTIVE}");
    println!("architecture\t{V071_ARCHITECTURE}");
    println!("context_policy\t{V071_CONTEXT_POLICY}");
    println!("device\t{device:?}");
    println!("mode\t{mode}");
    println!("parent_v070_checkpoint\t{}", parent_v070.display());
    println!(
        "parent_v070_completed_epochs\t{}",
        parent_metadata.completed_epochs
    );
    println!(
        "parent_v070_completed_updates\t{}",
        parent_metadata.completed_updates
    );
    println!(
        "parent_v070_dev_selection_score\t{:.8}",
        parent_metadata.dev_selection_score
    );
    println!(
        "parent_v070_spectrum_variables_loaded\t{}",
        load_report.spectrum_variables_loaded
    );
    println!(
        "parent_v070_ignored_variables\t{}",
        load_report.ignored_checkpoint_variables
    );
    println!("parent_update_policy\tfrozen_spectrum_encoder_optimizer_excluded");
    println!("decoder_update_policy\tdecoder_namespace_only");
    println!("optimizer_variable_count\t{}", optimizer.variable_count());
    println!("decoder_model_dim\t{}", decoder_config.model_dim);
    println!("decoder_layers\t{}", decoder_config.decoder_layers);
    println!("decoder_heads\t{}", decoder_config.num_attention_heads);
    println!("decoder_ff_dim\t{}", decoder_config.feed_forward_dim);
    println!("spectrum_layers\t{}", decoder_config.spectrum_layers);
    println!("vocabulary\tlegacy_common_ptm_causal");
    println!("optional_rt_conditioning\tNO");
    println!("optional_ccs_conditioning\tNO");
    println!("optional_property_reason\treserved_for_matched_post_v071_ablation");
    println!("train_eligible_unique_identities\t{}", train_groups.len());
    println!("dev_eligible_unique_identities\t{}", dev_groups.len());
    print_mass_feasibility("train_mass_feasibility", train_mass_audit);
    print_mass_feasibility("dev_mass_feasibility", dev_mass_audit);
    println!("dev_generation_identities\t{}", dev_indices.len());
    println!("dev_identity_fingerprint\t{dev_fingerprint}");
    print_property_availability("train_property_availability", train_availability);
    print_property_availability("dev_property_availability", dev_availability);
    println!("holdout_records_reserved_not_read\t{holdout_reserved}");
    println!("max_epochs\t{max_epochs}");
    println!("steps_per_epoch\t{steps_per_epoch}");
    println!("batch_size\t{batch_size}");
    println!("beam_width\t{beam_width}");
    println!("beam_top_k\t{V071_TOP_K}");
    println!("mass_tolerance_da\t{V071_MASS_TOLERANCE_DA}");
    println!("learning_rate\t{learning_rate}");
    println!("weight_decay\t{V071_WEIGHT_DECAY}");
    println!("max_gradient_norm\t{V071_MAX_GRADIENT_NORM}");
    println!("selection_metric\t0.7_il_top1_plus_0.3_il_top10");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let metadata = |completed_epochs: usize,
                    completed_updates: usize,
                    score: f64,
                    step0_score: f64|
     -> V071Metadata {
        V071Metadata {
            version: V071_VERSION,
            objective: V071_OBJECTIVE.into(),
            architecture: V071_ARCHITECTURE.into(),
            context_policy: V071_CONTEXT_POLICY.into(),
            parent_v070_checkpoint: parent_v070.display().to_string(),
            parent_v070_completed_epochs: parent_metadata.completed_epochs,
            parent_v070_completed_updates: parent_metadata.completed_updates,
            parent_v070_dev_selection_score: parent_metadata.dev_selection_score,
            parent_v070_spectrum_variables_loaded: load_report.spectrum_variables_loaded,
            parent_v070_ignored_variables: load_report.ignored_checkpoint_variables,
            parent_update_policy: "frozen_spectrum_encoder_optimizer_excluded".into(),
            decoder_update_policy: "decoder_namespace_only".into(),
            corpus_fingerprint: corpus_fingerprint.clone(),
            benchmark_manifest_fingerprint: benchmark_fingerprint.clone(),
            decoder_config: decoder_config.clone(),
            max_epochs,
            steps_per_epoch,
            batch_size,
            patience,
            min_delta,
            seed,
            learning_rate,
            weight_decay: V071_WEIGHT_DECAY,
            dev_identity_count: dev_indices.len(),
            dev_identity_fingerprint: dev_fingerprint.clone(),
            beam_width,
            top_k: V071_TOP_K,
            mass_tolerance_da: V071_MASS_TOLERANCE_DA,
            completed_epochs,
            completed_updates,
            dev_selection_score: score,
            step0_selection_score: step0_score,
            smoke_mode,
        }
    };

    let (
        mut global_update,
        start_epoch,
        mut best_epoch,
        mut best_update,
        mut best_generation,
        step0_generation,
        mut stale_epochs,
    ) = if resume_mode {
        let initial_meta = read_v071_metadata(&output_root.join("model/initial"))?;
        let latest_meta = read_v071_metadata(&output_root.join("model/latest"))?;
        let best_meta = read_v071_metadata(&output_root.join("model/best"))?;
        validate_resume(
            &latest_meta,
            &parent_v070,
            &corpus_fingerprint,
            &benchmark_fingerprint,
            &decoder_config,
            max_epochs,
            batch_size,
            steps_per_epoch,
            patience,
            min_delta,
            seed,
            learning_rate,
            beam_width,
            &dev_fingerprint,
        )?;
        varmap.load(output_root.join("model/latest/model.safetensors"))?;
        optimizer.load_safetensors(output_root.join("model/latest/optimizer.safetensors"))?;
        optimizer.set_step_count(latest_meta.completed_updates as u64);
        let best_generation =
            read_generation_metrics(&output_root.join("model/best/dev_generation.tsv"))?;
        let step0_generation =
            read_generation_metrics(&output_root.join("model/initial/dev_generation.tsv"))?;
        let _ = initial_meta;
        (
            latest_meta.completed_updates,
            latest_meta.completed_epochs + 1,
            best_meta.completed_epochs,
            best_meta.completed_updates,
            best_generation,
            step0_generation,
            latest_meta
                .completed_epochs
                .saturating_sub(best_meta.completed_epochs),
        )
    } else {
        let teacher = evaluate_teacher(
            &model,
            &corpus.records,
            &dev_indices,
            batch_size,
            &causal_collator,
            &spectrum_collator,
            &device,
            seed,
        )?;
        let (generation, generation_outcomes) = evaluate_generation(
            &model,
            &corpus.records,
            &dev_indices,
            &causal_collator,
            &spectrum_collator,
            &decoder_config,
            beam_width,
            &device,
        )?;
        print_teacher("v0710_dev_teacher_initial", 0, teacher);
        print_generation("v0710_dev_generation_initial", 0, generation);
        save_checkpoint(
            &output_root.join("model/initial"),
            &varmap,
            &optimizer,
            &metadata(
                0,
                0,
                generation.selection_score(),
                generation.selection_score(),
            ),
            teacher,
            generation,
            &generation_outcomes,
        )?;
        save_checkpoint(
            &output_root.join("model/best"),
            &varmap,
            &optimizer,
            &metadata(
                0,
                0,
                generation.selection_score(),
                generation.selection_score(),
            ),
            teacher,
            generation,
            &generation_outcomes,
        )?;
        (0, 1, 0, 0, generation, generation, 0)
    };

    for epoch in start_epoch..=max_epochs {
        let order = deterministic_order(
            train_groups.len(),
            seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let mut loss_sum = 0.0f64;
        let mut matched_sum = 0.0f64;
        let mut conditioning_gap_sum = 0.0f64;
        for step_in_epoch in 0..steps_per_epoch {
            let records = select_train_records(
                &corpus.records,
                &train_groups,
                &order,
                batch_size,
                step_in_epoch,
                seed ^ (epoch as u64).rotate_left(17),
            );
            global_update += 1;
            let lr =
                schedule.learning_rate(learning_rate, global_update.saturating_sub(1) as u64)?;
            optimizer.set_learning_rate(lr)?;

            let peptides = records
                .iter()
                .map(|record| record.peptidoform.clone())
                .collect::<Vec<_>>();
            let causal = causal_collator.collate(&peptides, &device)?;
            let spectra = record_spectra(&records)?;
            let spectrum_batch = spectrum_collator.collate(&spectra, &device)?;
            let precursor = precursor_context(&records, &device)?;
            let matched_context = model.prepare_context(&spectrum_batch, &precursor, false)?;
            let matched_output =
                model.forward_t_with_context(&causal.input, &matched_context, true)?;
            let matched = foundation_causal_next_token_loss(&matched_output, &causal)?;
            let prefix = foundation_direct_prefix_competitive_loss(
                &matched_output,
                &causal,
                FOUNDATION_DIRECT_PREFIX_MARGIN_V0191,
            )?;

            let shuffled_order =
                foundation_direct_shuffled_order(records.len(), seed ^ global_update as u64)?;
            let shuffled_spectra = shuffled_order
                .iter()
                .map(|&index| {
                    FoundationSpectrum::from_training_record(records[index]).ok_or_else(|| {
                        anyhow::anyhow!("v0.71 shuffled TRAIN record lacks spectrum")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let shuffled_batch = spectrum_collator.collate(&shuffled_spectra, &device)?;
            // Keep precursor metadata fixed; only the observed spectrum is shuffled.
            let shuffled_context = model.prepare_context(&shuffled_batch, &precursor, false)?;
            let shuffled_output =
                model.forward_t_with_context(&causal.input, &shuffled_context, true)?;
            let shuffled = foundation_causal_next_token_loss(&shuffled_output, &causal)?;
            let conditioned = foundation_direct_conditioning_loss(
                &matched,
                &shuffled,
                FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
                FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
            )?;
            let loss = (&conditioned + &prefix.loss)?;
            let loss_value = f64::from(loss.to_scalar::<f32>()?);
            let matched_value = f64::from(matched.to_scalar::<f32>()?);
            let shuffled_value = f64::from(shuffled.to_scalar::<f32>()?);
            let update = optimizer.backward_step(&loss, Some(V071_MAX_GRADIENT_NORM))?;
            loss_sum += loss_value;
            matched_sum += matched_value;
            conditioning_gap_sum += shuffled_value - matched_value;
            if global_update <= 4 || step_in_epoch % 50 == 0 {
                println!(
                    "v0710_train\tepoch={epoch}\tstep_in_epoch={step_in_epoch}\tupdate={global_update}\tlr={:.8}\tobjective={loss_value:.6}\tmatched_nll={matched_value:.6}\tshuffled_nll={shuffled_value:.6}\tconditioning_gap={:.6}\tprefix_margin_loss={:.6}\tlegal_rank={:.4}\tlegal_top1={:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                    update.learning_rate,
                    shuffled_value - matched_value,
                    f64::from(prefix.loss.to_scalar::<f32>()?),
                    prefix.mean_legal_rank,
                    prefix.top1_fraction,
                    update.gradient_norm,
                    update.gradient_scale,
                );
            }
        }

        let checksum = spectrum_encoder_checksum(&varmap)?;
        let checksum_delta = (checksum - frozen_spectrum_checksum_initial).abs();
        if checksum_delta > 1.0e-6 * frozen_spectrum_checksum_initial.abs().max(1.0) {
            anyhow::bail!(
                "v0.71 frozen v0.70 spectrum encoder changed: initial={frozen_spectrum_checksum_initial:.8} current={checksum:.8} delta={checksum_delta:.8}"
            );
        }
        println!(
            "v0710_parent_freeze_audit\tepoch={epoch}\tstatus=PASS\tchecksum={checksum:.8}\tdelta={checksum_delta:.8}"
        );

        let teacher = evaluate_teacher(
            &model,
            &corpus.records,
            &dev_indices,
            batch_size,
            &causal_collator,
            &spectrum_collator,
            &device,
            seed ^ global_update as u64,
        )?;
        let (generation, generation_outcomes) = evaluate_generation(
            &model,
            &corpus.records,
            &dev_indices,
            &causal_collator,
            &spectrum_collator,
            &decoder_config,
            beam_width,
            &device,
        )?;
        print_teacher("v0710_dev_teacher", global_update, teacher);
        print_generation("v0710_dev_generation", global_update, generation);
        println!(
            "v0710_epoch\tepoch={epoch}\tupdate={global_update}\tmean_objective={:.6}\tmean_matched_nll={:.6}\tmean_conditioning_gap={:.6}\tdev_selection_score={:.8}",
            loss_sum / steps_per_epoch as f64,
            matched_sum / steps_per_epoch as f64,
            conditioning_gap_sum / steps_per_epoch as f64,
            generation.selection_score(),
        );
        save_checkpoint(
            &output_root.join("model/latest"),
            &varmap,
            &optimizer,
            &metadata(
                epoch,
                global_update,
                generation.selection_score(),
                step0_generation.selection_score(),
            ),
            teacher,
            generation,
            &generation_outcomes,
        )?;

        let improved = generation.selection_score() - best_generation.selection_score() > min_delta;
        if improved {
            best_epoch = epoch;
            best_update = global_update;
            best_generation = generation;
            stale_epochs = 0;
            save_checkpoint(
                &output_root.join("model/best"),
                &varmap,
                &optimizer,
                &metadata(
                    epoch,
                    global_update,
                    generation.selection_score(),
                    step0_generation.selection_score(),
                ),
                teacher,
                generation,
                &generation_outcomes,
            )?;
            println!(
                "v0710_best_checkpoint\tepoch={best_epoch}\tupdate={best_update}\tselection_score={:.8}",
                best_generation.selection_score()
            );
        } else {
            stale_epochs += 1;
        }

        if smoke_mode {
            println!("v0710_smoke_complete\tepoch={epoch}\tupdates={global_update}");
            break;
        }
        if stale_epochs >= patience {
            println!(
                "v0710_early_stop\tepoch={epoch}\tupdate={global_update}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_update={best_update}"
            );
            break;
        }
    }

    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!(
        "v0710_training_complete\tbest_epoch={best_epoch}\tbest_update={best_update}\tstep0_selection_score={:.8}\tbest_selection_score={:.8}",
        step0_generation.selection_score(),
        best_generation.selection_score(),
    );
    let best_teacher = read_teacher_metrics(&output_root.join("model/best/dev_teacher.tsv"))?;
    print_material_gate(step0_generation, best_generation, best_teacher, smoke_mode);
    println!(
        "best_checkpoint\t{}",
        output_root.join("model/best").display()
    );
    Ok(())
}

fn read_v070_metadata(checkpoint: &Path) -> Result<V070Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.70 metadata {path:?}"))?,
    )
    .map_err(anyhow::Error::from)
}

fn validate_v070_parent(
    metadata: &V070Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
) -> Result<()> {
    if metadata.version != 700
        || metadata.objective != "v0700_frozen_v0520_spectrum_peptide_alignment"
        || metadata.architecture
            != "frozen_v0520_peptide_plus_observed_spectrum_transformer_contrastive_v0700"
        || metadata.completed_epochs == 0
        || metadata.completed_updates == 0
        || metadata.smoke_mode
        || !(metadata.dev_selection_score.is_finite() && metadata.dev_selection_score > 0.0)
    {
        anyhow::bail!("v0.71 requires the selected completed non-smoke v0.70 checkpoint");
    }
    if metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
    {
        anyhow::bail!("v0.71 v0.70 parent provenance differs from current corpus/benchmark");
    }
    if metadata.config.spectrum.model_dim != 320
        || metadata.config.spectrum.spectrum_layers != 6
        || metadata.config.alignment_dim != 192
        || (metadata.config.temperature - 0.07).abs() > 1.0e-12
    {
        anyhow::bail!("v0.71 parent does not match the frozen v0.70 architecture");
    }
    Ok(())
}

fn load_v070_spectrum_encoder(
    varmap: &VarMap,
    checkpoint_path: &Path,
    device: &Device,
) -> Result<V070LoadReport> {
    let checkpoint = candle_core::safetensors::load(checkpoint_path, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.71 VarMap lock poisoned"))?;
    let mut loaded = 0usize;
    for (name, variable) in data.iter() {
        if !name.starts_with("spectrum_encoder.") {
            continue;
        }
        let parent_name = format!("student_v070.{name}");
        let tensor = checkpoint.get(&parent_name).ok_or_else(|| {
            anyhow::anyhow!("v0.70 checkpoint missing required spectrum tensor {parent_name}")
        })?;
        if tensor.dims() != variable.as_tensor().dims() {
            anyhow::bail!(
                "v0.71 spectrum warm-start shape mismatch for {name}: model {:?}, parent {:?}",
                variable.as_tensor().dims(),
                tensor.dims()
            );
        }
        variable.set(tensor)?;
        loaded += 1;
    }
    drop(data);
    if loaded == 0 {
        anyhow::bail!("v0.71 loaded zero spectrum-encoder tensors from v0.70");
    }
    Ok(V070LoadReport {
        spectrum_variables_loaded: loaded,
        ignored_checkpoint_variables: checkpoint
            .keys()
            .filter(|name| !name.starts_with("student_v070.spectrum_encoder."))
            .count(),
    })
}

fn spectrum_encoder_checksum(varmap: &VarMap) -> Result<f64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.71 VarMap lock poisoned"))?;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (name, variable) in data.iter() {
        if name.starts_with("spectrum_encoder.") {
            sum += f64::from(variable.as_tensor().sum_all()?.to_scalar::<f32>()?);
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.71 spectrum checksum saw zero variables");
    }
    Ok(sum)
}

fn build_groups(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    config: &FoundationDiffusionConfig,
    vocabulary: FoundationDiffusionVocabulary,
) -> Result<(Vec<IdentityGroup>, MassFeasibilityAudit)> {
    let mut groups = BTreeMap::<String, IdentityGroup>::new();
    let mut audit = MassFeasibilityAudit::default();
    let mut abs_mass_error_sum = 0.0f64;

    for entry in benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
    {
        let record = records
            .get(entry.record_index)
            .ok_or_else(|| anyhow::anyhow!("benchmark record index out of range"))?;
        let Some(charge) = record.context.charge else {
            continue;
        };
        let Some(precursor_mz) = record
            .context
            .precursor_mz
            .filter(|v| v.is_finite() && *v > 0.0)
        else {
            continue;
        };
        if charge <= 0
            || FoundationSpectrum::from_training_record(record).is_none()
            || vocabulary
                .encode(&record.peptidoform, config.max_tokens)
                .is_err()
        {
            continue;
        }

        audit.basic_eligible_records += 1;
        let target_mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        let precursor_mass = foundation_precursor_neutral_mass(f64::from(precursor_mz), charge)
            .map_err(anyhow::Error::msg)?;
        let abs_error = (target_mass - precursor_mass).abs();
        abs_mass_error_sum += abs_error;
        audit.max_abs_mass_error_da = audit.max_abs_mass_error_da.max(abs_error);
        if abs_error > V071_MASS_TOLERANCE_DA {
            audit.mass_infeasible_records += 1;
            continue;
        }
        audit.mass_feasible_records += 1;

        let key = format!("{}|z{charge}", entry.peptidoform);
        groups
            .entry(key.clone())
            .or_insert_with(|| IdentityGroup {
                key,
                record_indices: Vec::new(),
            })
            .record_indices
            .push(entry.record_index);
    }

    audit.mass_feasible_unique_identities = groups.len();
    audit.mean_abs_mass_error_da = if audit.basic_eligible_records > 0 {
        abs_mass_error_sum / audit.basic_eligible_records as f64
    } else {
        0.0
    };
    Ok((groups.into_values().collect(), audit))
}

fn print_mass_feasibility(label: &str, audit: MassFeasibilityAudit) {
    let feasible_fraction = if audit.basic_eligible_records > 0 {
        audit.mass_feasible_records as f64 / audit.basic_eligible_records as f64
    } else {
        0.0
    };
    println!(
        "{label}\tbasic_records={}\tmass_feasible_records={}\tmass_infeasible_records={}\tmass_feasible_fraction={feasible_fraction:.6}\tmass_feasible_unique_identities={}\tmean_abs_mass_error_da={:.6}\tmax_abs_mass_error_da={:.6}",
        audit.basic_eligible_records,
        audit.mass_feasible_records,
        audit.mass_infeasible_records,
        audit.mass_feasible_unique_identities,
        audit.mean_abs_mass_error_da,
        audit.max_abs_mass_error_da,
    );
}

fn select_identity_records(
    groups: &[IdentityGroup],
    records: &[FoundationTrainingRecord],
    count: usize,
    seed: u64,
) -> Vec<usize> {
    let mut order = (0..groups.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| mix64(seed ^ hash64_str(&groups[index].key)));
    order
        .into_iter()
        .take(count)
        .map(|group_index| {
            let group = &groups[group_index];
            let slot = (mix64(seed.rotate_left(19) ^ hash64_str(&group.key)) as usize)
                % group.record_indices.len();
            let index = group.record_indices[slot];
            debug_assert!(records[index].context.charge.is_some());
            index
        })
        .collect()
}

fn select_train_records<'a>(
    records: &'a [FoundationTrainingRecord],
    groups: &[IdentityGroup],
    order: &[usize],
    batch_size: usize,
    step_in_epoch: usize,
    seed: u64,
) -> Vec<&'a FoundationTrainingRecord> {
    (0..batch_size)
        .map(|slot| {
            let position = (step_in_epoch * batch_size + slot) % order.len();
            let group = &groups[order[position]];
            let record_slot = (mix64(seed ^ step_in_epoch as u64 ^ hash64_str(&group.key))
                as usize)
                % group.record_indices.len();
            &records[group.record_indices[record_slot]]
        })
        .collect()
}

fn deterministic_order(length: usize, seed: u64) -> Vec<usize> {
    let mut order = (0..length).collect::<Vec<_>>();
    order.sort_by_key(|&index| mix64(seed ^ index as u64));
    order
}

fn record_spectra(records: &[&FoundationTrainingRecord]) -> Result<Vec<FoundationSpectrum>> {
    records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record)
                .ok_or_else(|| anyhow::anyhow!("v0.71 selected record lacks observed spectrum"))
        })
        .collect()
}

fn precursor_context(
    records: &[&FoundationTrainingRecord],
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let b = records.len();
    let charge = records
        .iter()
        .map(|record| record.context.charge.unwrap_or(0) as f32)
        .collect::<Vec<_>>();
    let charge_present: Vec<f32> = records
        .iter()
        .map(|record| {
            if record.context.charge.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect();
    let precursor_mz: Vec<f32> = records
        .iter()
        .map(|record| record.context.precursor_mz.unwrap_or(0.0))
        .collect();
    let precursor_mz_present: Vec<f32> = records
        .iter()
        .map(|record| {
            if record.context.precursor_mz.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect();
    let nce: Vec<f32> = records
        .iter()
        .map(|record| record.context.nce.unwrap_or(0.0))
        .collect();
    let nce_present: Vec<f32> = records
        .iter()
        .map(|record| {
            if record.context.nce.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect();
    let context = PrecursorContextBatch {
        charge: Tensor::from_vec(charge, b, device)?,
        charge_present: Tensor::from_vec(charge_present, b, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, b, device)?,
        precursor_mz_present: Tensor::from_vec(precursor_mz_present, b, device)?,
        nce: Tensor::from_vec(nce, b, device)?,
        nce_present: Tensor::from_vec(nce_present, b, device)?,
        instrument_ids: Tensor::zeros(b, DType::U32, device)?,
        instrument_present: Tensor::zeros(b, DType::F32, device)?,
    };
    for (name, tensor) in [
        ("charge", &context.charge),
        ("charge_present", &context.charge_present),
        ("precursor_mz", &context.precursor_mz),
        ("precursor_mz_present", &context.precursor_mz_present),
        ("nce", &context.nce),
        ("nce_present", &context.nce_present),
        ("instrument_present", &context.instrument_present),
    ] {
        if tensor.dtype() != DType::F32 {
            anyhow::bail!(
                "v0.71 precursor context tensor {name} must be F32, observed {:?}",
                tensor.dtype()
            );
        }
    }
    Ok(context)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_teacher(
    model: &PeptideSpectrumCausalModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
    seed: u64,
) -> Result<TeacherMetrics> {
    let mut nll_weighted = 0.0f64;
    let mut active_tokens = 0usize;
    let mut correct_tokens = 0usize;
    let mut sequences = 0usize;
    let mut exact_sequences = 0usize;
    let mut gap_weighted = 0.0f64;
    let mut conditioning_tokens = 0usize;
    let mut legal_rank_weighted = 0.0f64;
    let mut legal_top1_weighted = 0.0f64;

    for (chunk_index, chunk) in indices.chunks(batch_size).enumerate() {
        let selected = chunk.iter().map(|&i| &records[i]).collect::<Vec<_>>();
        let peptides = selected
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect::<Vec<_>>();
        let causal = causal_collator.collate(&peptides, device)?;
        let spectra = record_spectra(&selected)?;
        let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
        let precursor = precursor_context(&selected, device)?;
        let context = model.prepare_context(&spectrum_batch, &precursor, false)?;
        let output = model.forward_t_with_context(&causal.input, &context, false)?;
        let nll = foundation_causal_next_token_loss(&output, &causal)?;
        let prefix = foundation_direct_prefix_competitive_loss(&output, &causal, 0.0)?;

        let shuffled_nll = if selected.len() >= 2 {
            let order =
                foundation_direct_shuffled_order(selected.len(), seed ^ chunk_index as u64)?;
            let shuffled_spectra = order
                .iter()
                .map(|&i| {
                    FoundationSpectrum::from_training_record(selected[i])
                        .ok_or_else(|| anyhow::anyhow!("v0.71 DEV shuffled record lacks spectrum"))
                })
                .collect::<Result<Vec<_>>>()?;
            let shuffled_batch = spectrum_collator.collate(&shuffled_spectra, device)?;
            let shuffled_context = model.prepare_context(&shuffled_batch, &precursor, false)?;
            let shuffled_output =
                model.forward_t_with_context(&causal.input, &shuffled_context, false)?;
            Some(foundation_causal_next_token_loss(
                &shuffled_output,
                &causal,
            )?)
        } else {
            None
        };

        let logits = output.token_logits.to_vec3::<f32>()?;
        let targets = causal.target_tokens.to_vec2::<u32>()?;
        let masks = causal.input.token_mask.to_vec2::<f32>()?;
        let chunk_active = causal.active_indices.dims1()?;
        nll_weighted += f64::from(nll.to_scalar::<f32>()?) * chunk_active as f64;
        if let Some(shuffled_nll) = shuffled_nll {
            gap_weighted += (f64::from(shuffled_nll.to_scalar::<f32>()?)
                - f64::from(nll.to_scalar::<f32>()?))
                * chunk_active as f64;
            conditioning_tokens += chunk_active;
        }
        legal_rank_weighted += prefix.mean_legal_rank * chunk_active as f64;
        legal_top1_weighted += prefix.top1_fraction * chunk_active as f64;
        active_tokens += chunk_active;

        for row in 0..selected.len() {
            let mut exact = true;
            for position in 0..targets[row].len() {
                if masks[row][position] <= 0.0 {
                    break;
                }
                let predicted = argmax(&logits[row][position]) as u32;
                if predicted == targets[row][position] {
                    correct_tokens += 1;
                } else {
                    exact = false;
                }
            }
            exact_sequences += usize::from(exact);
            sequences += 1;
        }
    }

    let token_denom = active_tokens.max(1) as f64;
    let sequence_denom = sequences.max(1) as f64;
    Ok(TeacherMetrics {
        token_nll: nll_weighted / token_denom,
        token_accuracy: correct_tokens as f64 / token_denom,
        sequence_exact: exact_sequences as f64 / sequence_denom,
        conditioning_gap: if conditioning_tokens > 0 {
            gap_weighted / conditioning_tokens as f64
        } else {
            0.0
        },
        legal_rank: legal_rank_weighted / token_denom,
        legal_top1: legal_top1_weighted / token_denom,
        active_tokens,
        sequences,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_generation(
    model: &PeptideSpectrumCausalModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    beam_width: usize,
    device: &Device,
) -> Result<(GenerationMetrics, Vec<GenerationOutcome>)> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut returned = 0usize;
    let mut peptidoform_hits = [0usize; 3];
    let mut sequence_hits = [0usize; 3];
    let mut il_hits = [0usize; 3];
    let mut edit_sum = 0usize;
    let mut normalized_edit_sum = 0.0f64;
    let mut mass_valid = 0usize;
    let mut returned_beams = 0usize;
    let mut outcomes = Vec::with_capacity(indices.len());
    let ks = [1usize, 5usize, 10usize];

    for &index in indices {
        let record = &records[index];
        let target_tokens = vocabulary
            .encode(&record.peptidoform, config.max_tokens)
            .map_err(anyhow::Error::msg)?
            .into_iter()
            .take_while(|&token| token != FOUNDATION_DIFFUSION_PAD)
            .collect::<Vec<_>>();
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("v0.71 generation record lacks spectrum"))?;
        let one = vec![record];
        let spectrum_batch = spectrum_collator.collate(&[spectrum], device)?;
        let precursor = precursor_context(&one, device)?;
        let context = model.prepare_context(&spectrum_batch, &precursor, false)?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("v0.71 generation record lacks charge"))?;
        let mz = record
            .context
            .precursor_mz
            .ok_or_else(|| anyhow::anyhow!("v0.71 generation record lacks precursor m/z"))?;
        let neutral_mass =
            foundation_precursor_neutral_mass(f64::from(mz), charge).map_err(anyhow::Error::msg)?;
        let beam = foundation_direct_beam_search(
            neutral_mass,
            DirectDecoderBeamConfig {
                beam_width,
                top_k: V071_TOP_K,
                mass_tolerance_da: V071_MASS_TOLERANCE_DA,
                max_tokens: config.max_tokens,
            },
            |prefixes| {
                let input = causal_collator
                    .collate_compact_prefix_rows(prefixes, device)
                    .map_err(|error| error.to_string())?;
                model
                    .forward_next_t_with_context(&input, &context, false)
                    .and_then(|logits| logits.to_vec2::<f32>())
                    .map_err(|error| error.to_string())
            },
        )
        .map_err(anyhow::Error::msg)?;

        if !beam.is_empty() {
            returned += 1;
        }
        returned_beams += beam.len();
        mass_valid += beam
            .iter()
            .filter(|candidate| candidate.mass_error_da.abs() <= V071_MASS_TOLERANCE_DA)
            .count();

        let decoded = beam
            .iter()
            .filter_map(|candidate| vocabulary.decode(&candidate.tokens).ok())
            .collect::<Vec<_>>();
        let mut local_peptidoform_hits = [false; 3];
        let mut local_sequence_hits = [false; 3];
        let mut local_il_hits = [false; 3];
        for (slot, &k) in ks.iter().enumerate() {
            let beam_upto = beam.len().min(k);
            let decoded_upto = decoded.len().min(k);
            local_peptidoform_hits[slot] = beam[..beam_upto]
                .iter()
                .any(|candidate| candidate.tokens == target_tokens);
            local_sequence_hits[slot] = decoded[..decoded_upto]
                .iter()
                .any(|candidate| candidate.sequence == record.peptidoform.sequence);
            local_il_hits[slot] = decoded[..decoded_upto].iter().any(|candidate| {
                il_sequence(&candidate.sequence) == il_sequence(&record.peptidoform.sequence)
            });
            peptidoform_hits[slot] += usize::from(local_peptidoform_hits[slot]);
            sequence_hits[slot] += usize::from(local_sequence_hits[slot]);
            il_hits[slot] += usize::from(local_il_hits[slot]);
        }

        let (edit, normalized_edit) = if let Some(top) = decoded.first() {
            let edit = levenshtein(&top.sequence, &record.peptidoform.sequence);
            let denominator = top
                .sequence
                .chars()
                .count()
                .max(record.peptidoform.sequence.chars().count())
                .max(1);
            (edit, edit as f64 / denominator as f64)
        } else {
            (record.peptidoform.sequence.chars().count(), 1.0)
        };
        edit_sum += edit;
        normalized_edit_sum += normalized_edit;
        outcomes.push(GenerationOutcome {
            charge,
            length: record.peptidoform.sequence.chars().count(),
            modified: !record.peptidoform.modifications.is_empty(),
            returned: !beam.is_empty(),
            peptidoform_top1: local_peptidoform_hits[0],
            peptidoform_top10: local_peptidoform_hits[2],
            sequence_top1: local_sequence_hits[0],
            sequence_top10: local_sequence_hits[2],
            il_top1: local_il_hits[0],
            il_top10: local_il_hits[2],
            top1_edit_distance: edit,
            top1_normalized_edit_distance: normalized_edit,
        });
    }

    let n = indices.len().max(1) as f64;
    let returned_beams = returned_beams.max(1);
    Ok((
        GenerationMetrics {
            identities: indices.len(),
            returned_fraction: returned as f64 / n,
            peptidoform_top1: peptidoform_hits[0] as f64 / n,
            peptidoform_top5: peptidoform_hits[1] as f64 / n,
            peptidoform_top10: peptidoform_hits[2] as f64 / n,
            sequence_top1: sequence_hits[0] as f64 / n,
            sequence_top5: sequence_hits[1] as f64 / n,
            sequence_top10: sequence_hits[2] as f64 / n,
            il_top1: il_hits[0] as f64 / n,
            il_top5: il_hits[1] as f64 / n,
            il_top10: il_hits[2] as f64 / n,
            mean_top1_edit_distance: edit_sum as f64 / n,
            mean_top1_normalized_edit_distance: normalized_edit_sum / n,
            mass_valid_fraction: mass_valid as f64 / returned_beams as f64,
        },
        outcomes,
    ))
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V071Metadata,
    teacher: TeacherMetrics,
    generation: GenerationMetrics,
    outcomes: &[GenerationOutcome],
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    optimizer.save_safetensors(directory.join("optimizer.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    write_teacher_metrics(&directory.join("dev_teacher.tsv"), teacher)?;
    write_generation_metrics(&directory.join("dev_generation.tsv"), generation)?;
    write_generation_stratified(&directory.join("dev_generation_stratified.tsv"), outcomes)?;
    Ok(())
}

fn write_teacher_metrics(path: &Path, m: TeacherMetrics) -> Result<()> {
    fs::write(
        path,
        format!(
            "metric\tvalue\ntoken_nll\t{:.12}\ntoken_accuracy\t{:.12}\nsequence_exact\t{:.12}\nconditioning_gap\t{:.12}\nlegal_rank\t{:.12}\nlegal_top1\t{:.12}\nactive_tokens\t{}\nsequences\t{}\n",
            m.token_nll,
            m.token_accuracy,
            m.sequence_exact,
            m.conditioning_gap,
            m.legal_rank,
            m.legal_top1,
            m.active_tokens,
            m.sequences,
        ),
    )?;
    Ok(())
}

fn write_generation_metrics(path: &Path, m: GenerationMetrics) -> Result<()> {
    fs::write(
        path,
        format!(
            "metric\tvalue\nidentities\t{}\nreturned_fraction\t{:.12}\npeptidoform_top1\t{:.12}\npeptidoform_top5\t{:.12}\npeptidoform_top10\t{:.12}\nsequence_top1\t{:.12}\nsequence_top5\t{:.12}\nsequence_top10\t{:.12}\nil_top1\t{:.12}\nil_top5\t{:.12}\nil_top10\t{:.12}\nmean_top1_edit_distance\t{:.12}\nmean_top1_normalized_edit_distance\t{:.12}\nmass_valid_fraction\t{:.12}\nselection_score\t{:.12}\n",
            m.identities,
            m.returned_fraction,
            m.peptidoform_top1,
            m.peptidoform_top5,
            m.peptidoform_top10,
            m.sequence_top1,
            m.sequence_top5,
            m.sequence_top10,
            m.il_top1,
            m.il_top5,
            m.il_top10,
            m.mean_top1_edit_distance,
            m.mean_top1_normalized_edit_distance,
            m.mass_valid_fraction,
            m.selection_score(),
        ),
    )?;
    Ok(())
}

fn read_teacher_metrics(path: &Path) -> Result<TeacherMetrics> {
    let text = fs::read_to_string(path)?;
    let values = text
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect::<BTreeMap<_, _>>();
    let get = |name: &str| -> Result<f64> {
        values
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing v0.71 teacher metric {name}"))?
            .parse()
            .map_err(anyhow::Error::from)
    };
    Ok(TeacherMetrics {
        token_nll: get("token_nll")?,
        token_accuracy: get("token_accuracy")?,
        sequence_exact: get("sequence_exact")?,
        conditioning_gap: get("conditioning_gap")?,
        legal_rank: get("legal_rank")?,
        legal_top1: get("legal_top1")?,
        active_tokens: get("active_tokens")? as usize,
        sequences: get("sequences")? as usize,
    })
}

fn write_generation_stratified(path: &Path, outcomes: &[GenerationOutcome]) -> Result<()> {
    let mut strata = BTreeMap::<String, Vec<&GenerationOutcome>>::new();
    for outcome in outcomes {
        strata
            .entry(format!("charge_z{}", outcome.charge))
            .or_default()
            .push(outcome);
        strata
            .entry(format!("length_{}", length_bin(outcome.length)))
            .or_default()
            .push(outcome);
        strata
            .entry(format!(
                "ptm_{}",
                if outcome.modified {
                    "modified"
                } else {
                    "unmodified"
                }
            ))
            .or_default()
            .push(outcome);
    }
    let mut text = String::from(
        "stratum\tn\treturned_fraction\tpeptidoform_top1\tpeptidoform_top10\tsequence_top1\tsequence_top10\til_top1\til_top10\tmean_edit\tmean_normalized_edit\n",
    );
    for (name, rows) in strata {
        let n = rows.len() as f64;
        let frac =
            |f: fn(&GenerationOutcome) -> bool| rows.iter().filter(|row| f(row)).count() as f64 / n;
        let mean_edit = rows.iter().map(|row| row.top1_edit_distance).sum::<usize>() as f64 / n;
        let mean_normalized_edit = rows
            .iter()
            .map(|row| row.top1_normalized_edit_distance)
            .sum::<f64>()
            / n;
        text.push_str(&format!(
            "{name}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{mean_edit:.4}\t{mean_normalized_edit:.8}\n",
            rows.len(),
            frac(|row| row.returned),
            frac(|row| row.peptidoform_top1),
            frac(|row| row.peptidoform_top10),
            frac(|row| row.sequence_top1),
            frac(|row| row.sequence_top10),
            frac(|row| row.il_top1),
            frac(|row| row.il_top10),
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn length_bin(length: usize) -> &'static str {
    match length {
        0..=7 => "01_07",
        8..=12 => "08_12",
        13..=18 => "13_18",
        19..=25 => "19_25",
        26..=35 => "26_35",
        _ => "36_plus",
    }
}

fn read_generation_metrics(path: &Path) -> Result<GenerationMetrics> {
    let text = fs::read_to_string(path)?;
    let values = text
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect::<BTreeMap<_, _>>();
    let get = |name: &str| -> Result<f64> {
        values
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing v0.71 generation metric {name}"))?
            .parse()
            .map_err(anyhow::Error::from)
    };
    Ok(GenerationMetrics {
        identities: get("identities")? as usize,
        returned_fraction: get("returned_fraction")?,
        peptidoform_top1: get("peptidoform_top1")?,
        peptidoform_top5: get("peptidoform_top5")?,
        peptidoform_top10: get("peptidoform_top10")?,
        sequence_top1: get("sequence_top1")?,
        sequence_top5: get("sequence_top5")?,
        sequence_top10: get("sequence_top10")?,
        il_top1: get("il_top1")?,
        il_top5: get("il_top5")?,
        il_top10: get("il_top10")?,
        mean_top1_edit_distance: get("mean_top1_edit_distance")?,
        mean_top1_normalized_edit_distance: get("mean_top1_normalized_edit_distance")?,
        mass_valid_fraction: get("mass_valid_fraction")?,
    })
}

fn read_v071_metadata(checkpoint: &Path) -> Result<V071Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(&fs::read_to_string(&path)?).map_err(anyhow::Error::from)
}

#[allow(clippy::too_many_arguments)]
fn validate_resume(
    metadata: &V071Metadata,
    parent: &Path,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    config: &FoundationDiffusionConfig,
    max_epochs: usize,
    batch_size: usize,
    steps_per_epoch: usize,
    patience: usize,
    min_delta: f64,
    seed: u64,
    learning_rate: f64,
    beam_width: usize,
    dev_fingerprint: &str,
) -> Result<()> {
    if metadata.version != V071_VERSION
        || metadata.objective != V071_OBJECTIVE
        || metadata.architecture != V071_ARCHITECTURE
        || metadata.parent_v070_checkpoint != parent.display().to_string()
        || metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        || metadata.context_policy != V071_CONTEXT_POLICY
        || &metadata.decoder_config != config
        || metadata.max_epochs != max_epochs
        || metadata.batch_size != batch_size
        || metadata.steps_per_epoch != steps_per_epoch
        || metadata.patience != patience
        || metadata.min_delta != min_delta
        || metadata.seed != seed
        || metadata.learning_rate != learning_rate
        || metadata.weight_decay != V071_WEIGHT_DECAY
        || metadata.beam_width != beam_width
        || metadata.top_k != V071_TOP_K
        || metadata.mass_tolerance_da != V071_MASS_TOLERANCE_DA
        || metadata.dev_identity_fingerprint != dev_fingerprint
        || metadata.smoke_mode
    {
        anyhow::bail!("v0.71 resume metadata does not match the fixed experiment contract");
    }
    Ok(())
}

fn property_availability(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
) -> PropertyAvailability {
    let mut out = PropertyAvailability::default();
    for &index in indices {
        let record = &records[index];
        let rt = record
            .retention_time
            .harmonized
            .is_some_and(|v| v.is_finite());
        let ccs = record.ccs.is_some_and(|v| v.is_finite());
        out.records += 1;
        out.harmonized_rt += usize::from(rt);
        out.ccs += usize::from(ccs);
        out.both += usize::from(rt && ccs);
    }
    out
}

fn print_property_availability(label: &str, value: PropertyAvailability) {
    println!(
        "{label}\trecords={}\tharmonized_rt={}\tccs={}\tboth={}",
        value.records, value.harmonized_rt, value.ccs, value.both
    );
}

fn print_teacher(label: &str, update: usize, m: TeacherMetrics) {
    println!(
        "{label}\tupdate={update}\ttoken_nll={:.6}\ttoken_accuracy={:.6}\tsequence_exact={:.6}\tconditioning_gap={:.6}\tlegal_rank={:.4}\tlegal_top1={:.6}\tactive_tokens={}\tsequences={}",
        m.token_nll,
        m.token_accuracy,
        m.sequence_exact,
        m.conditioning_gap,
        m.legal_rank,
        m.legal_top1,
        m.active_tokens,
        m.sequences,
    );
}

fn print_generation(label: &str, update: usize, m: GenerationMetrics) {
    println!(
        "{label}\tupdate={update}\tidentities={}\treturned_fraction={:.6}\tpeptidoform_top1={:.6}\tpeptidoform_top5={:.6}\tpeptidoform_top10={:.6}\tsequence_top1={:.6}\tsequence_top5={:.6}\tsequence_top10={:.6}\til_top1={:.6}\til_top5={:.6}\til_top10={:.6}\tmean_edit={:.4}\tnormalized_edit={:.6}\tmass_valid_fraction={:.6}\tselection_score={:.6}",
        m.identities,
        m.returned_fraction,
        m.peptidoform_top1,
        m.peptidoform_top5,
        m.peptidoform_top10,
        m.sequence_top1,
        m.sequence_top5,
        m.sequence_top10,
        m.il_top1,
        m.il_top5,
        m.il_top10,
        m.mean_top1_edit_distance,
        m.mean_top1_normalized_edit_distance,
        m.mass_valid_fraction,
        m.selection_score(),
    );
}

fn print_material_gate(
    step0: GenerationMetrics,
    best: GenerationMetrics,
    best_teacher: TeacherMetrics,
    smoke: bool,
) {
    if smoke {
        println!("v0710_material_gate\tNOT_APPLICABLE_SMOKE");
        return;
    }
    let gain = best.selection_score() - step0.selection_score();
    let gain_pass = gain >= V071_GATE_MIN_SELECTION_GAIN;
    let top1_pass = best.il_top1 >= V071_GATE_MIN_IL_TOP1;
    let top10_pass = best.il_top10 >= V071_GATE_MIN_IL_TOP10;
    let returned_pass = best.returned_fraction >= V071_GATE_MIN_RETURNED_FRACTION;
    let conditioning_pass = best_teacher.conditioning_gap >= V071_GATE_MIN_CONDITIONING_GAP;
    println!(
        "v0710_gate_selection_gain_ge_0_05\t{}",
        pass_fail(gain_pass)
    );
    println!("v0710_gate_il_top1_ge_0_08\t{}", pass_fail(top1_pass));
    println!("v0710_gate_il_top10_ge_0_25\t{}", pass_fail(top10_pass));
    println!(
        "v0710_gate_returned_fraction_ge_0_80\t{}",
        pass_fail(returned_pass)
    );
    println!(
        "v0710_gate_conditioning_gap_ge_0_02\t{}",
        pass_fail(conditioning_pass)
    );
    println!(
        "v0710_material_gate\t{}",
        pass_fail(gain_pass && top1_pass && top10_pass && returned_pass && conditioning_pass)
    );
    println!("v0710_selection_gain\t{gain:.8}");
    println!(
        "v0710_best_conditioning_gap\t{:.8}",
        best_teacher.conditioning_gap
    );
}

fn pass_fail(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

fn index_fingerprint(indices: &[usize]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &index in indices {
        hash ^= index as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn hash64_str(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
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

fn il_sequence(sequence: &str) -> String {
    sequence
        .chars()
        .map(|residue| if residue == 'I' { 'L' } else { residue })
        .collect()
}

fn levenshtein(left: &str, right: &str) -> usize {
    let a = left.chars().collect::<Vec<_>>();
    let b = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=b.len()).collect::<Vec<_>>();
    let mut current = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + usize::from(ca != cb));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn parse_or<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid argument {index}: {error}")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0710_selection_score_prioritizes_top1() {
        let metrics = GenerationMetrics {
            il_top1: 0.2,
            il_top10: 0.8,
            ..GenerationMetrics::default()
        };
        assert!((metrics.selection_score() - 0.38).abs() < 1.0e-12);
    }

    #[test]
    fn v0710_il_equivalence_collapses_isoleucine() {
        assert_eq!(il_sequence("PEPTIDE"), "PEPTLDE");
        assert_eq!(il_sequence("LLIL"), "LLLL");
    }

    #[test]
    fn v0710_edit_distance_is_sequence_level() {
        assert_eq!(levenshtein("PEPTIDE", "PEPTIDE"), 0);
        assert_eq!(levenshtein("PEPTIDE", "PEPTLDE"), 1);
        assert_eq!(levenshtein("ABC", "ABCD"), 1);
    }

    #[test]
    fn v0710_parent_tensor_mapping_is_spectrum_only() {
        let current = "spectrum_encoder.layers.0.self_attention.query.weight";
        let parent = format!("student_v070.{current}");
        assert_eq!(
            parent,
            "student_v070.spectrum_encoder.layers.0.self_attention.query.weight"
        );
        assert!(!"decoder.precursor.weight".starts_with("spectrum_encoder."));
    }
}
