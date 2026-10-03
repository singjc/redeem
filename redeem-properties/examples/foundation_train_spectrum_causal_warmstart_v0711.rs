//! v0.71.1 warm-started spectrum-conditioned chemistry-aware causal decoder.
//!
//! Scientific contract:
//! - preserve the accepted trained v0.20 chemistry decoder exactly at step 0;
//! - preserve the selected v0.70 spectrum encoder frozen;
//! - project v0.70 320-d contextual peak states into the v0.20 native raw peak-feature width;
//! - fuse the projected states as a residual before the frozen v0.20 spectrum encoder;
//! - initialize the residual gate to exactly zero so step 0 reproduces v0.20 behavior;
//! - optimize only the adapter projection and scalar residual gate;
//! - use mass-feasible TRAIN/DEV identities and the v0.20 chemistry suffix mask;
//! - keep RT/CCS out of the decoder input;
//! - use TRAIN/DEV only; never access protected HOLDOUT/historical TEST.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_causal_next_token_loss, foundation_direct_beam_search,
    foundation_direct_conditioning_loss, foundation_direct_prefix_competitive_loss,
    foundation_direct_shuffled_order, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    ChemistrySuffixMassLattice, ChemistryTransitionFeaturizer, DirectDecoderBeamConfig,
    FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCausalCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationLearningRateSchedule,
    FoundationPartition, FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator,
    FoundationSpectrumEncoder, FoundationTrainingRecord, PeptideSpectrumChemistryDecoder,
    PeptidoformInput, PrecursorContextBatch, FOUNDATION_CHEMISTRY_DECODER_ARCHITECTURE_V0200,
    FOUNDATION_CHEMISTRY_DECODER_OBJECTIVE_V0200, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190, FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V0711_VERSION: u32 = 711;
const V0711_OBJECTIVE: &str =
    "v0711_frozen_v0200_decoder_zero_gate_v070_residual_adapter_conditioning";
const V0711_ARCHITECTURE: &str =
    "frozen_v0200_chemistry_decoder_plus_zero_gate_v070_peak_residual_adapter";
const V0711_CONTEXT_POLICY: &str =
    "v0200_native_spectrum_context_plus_trainable_v070_residual_rt_ccs_excluded";
const V0711_BATCH: usize = 32;
const V0711_PROBE_STEPS: usize = 256;
const V0711_SMOKE_STEPS: usize = 8;
const V0711_DEV_IDENTITIES: usize = 256;
const V0711_SMOKE_DEV_IDENTITIES: usize = 16;
const V0711_BEAM_WIDTH: usize = 64;
const V0711_SMOKE_BEAM_WIDTH: usize = 8;
const V0711_TOP_K: usize = 10;
const V0711_MASS_TOLERANCE_DA: f64 = 0.05;
const V0711_SEED: u64 = 20_261_071_1;
const V0711_LEARNING_RATE: f64 = 1.0e-4;
const V0711_WEIGHT_DECAY: f64 = 1.0e-4;
const V0711_MAX_GRADIENT_NORM: f64 = 5.0;

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

#[derive(Debug, Clone, Deserialize)]
struct V0200Metadata {
    version: String,
    objective: String,
    architecture: String,
    global_step: usize,
    inverse_config: FoundationDiffusionConfig,
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

#[derive(Debug, Clone, Serialize)]
struct V0711Metadata {
    version: u32,
    objective: String,
    architecture: String,
    context_policy: String,
    parent_v070_checkpoint: String,
    parent_v0200_checkpoint: String,
    parent_v0200_global_step: usize,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    v070_config: FoundationDiffusionConfig,
    decoder_config: FoundationDiffusionConfig,
    updates: usize,
    batch_size: usize,
    seed: u64,
    learning_rate: f64,
    beam_width: usize,
    top_k: usize,
    mass_tolerance_da: f64,
    dev_identity_count: usize,
    dev_identity_fingerprint: String,
    residual_gate: f32,
    teacher: TeacherMetrics,
    generation: GenerationMetrics,
}

struct SpectrumResidualAdapter {
    projection: Linear,
    gate: Tensor,
}

impl SpectrumResidualAdapter {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let projection = nn::linear(input_dim, output_dim, vb.pp("projection"))?;
        let gate = vb.get_with_hints(1, "residual_gate", nn::Init::Const(0.0))?;
        Ok(Self { projection, gate })
    }

    fn peak_residual(
        &self,
        encoding: &redeem_properties::foundation::FoundationSpectrumEncoding,
    ) -> candle_core::Result<Tensor> {
        let peaks = self.projection.forward(&encoding.peak_embeddings)?;
        let peak_gate = self.gate.reshape((1, 1, 1))?;
        peaks.broadcast_mul(&peak_gate)
    }

    fn gate_value(&self) -> candle_core::Result<f32> {
        self.gate.to_vec1::<f32>().map(|v| v[0])
    }
}

#[derive(Debug)]
struct ChemistryComponents {
    peptides: Vec<PeptidoformInput>,
    spectra: Vec<FoundationSpectrum>,
    precursor_masses: Vec<f64>,
    charges: Vec<i32>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_train_spectrum_causal_warmstart_v0711 RUN.yaml OUTPUT_DIR V070_BEST V0200_BEST [smoke|probe]"
        );
    }
    let run_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let parent_v070 = PathBuf::from(&args[3]);
    let parent_v0200 = PathBuf::from(&args[4]);
    let mode = args.get(5).map(String::as_str).unwrap_or("probe");
    if !matches!(mode, "smoke" | "probe") {
        anyhow::bail!("v0.71.1 mode must be smoke or probe");
    }
    if output_root.exists() {
        anyhow::bail!("v0.71.1 output directory must be fresh: {output_root:?}");
    }
    reject_protected_path(&run_yaml)?;
    reject_protected_path(&output_root)?;

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.71.1 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(&run_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());

    let v070_meta = read_v070_metadata(&parent_v070)?;
    validate_v070_parent(&v070_meta, &corpus_fingerprint, &benchmark_fingerprint)?;
    let v070_model_path = parent_v070.join("model.safetensors");
    if !v070_model_path.is_file() {
        anyhow::bail!("v0.71.1 v0.70 parent missing model.safetensors: {parent_v070:?}");
    }

    let v0200_meta = read_v0200_metadata(&parent_v0200)?;
    validate_v0200_parent(&v0200_meta)?;
    let v0200_model_path = parent_v0200.join("model.safetensors");
    if !v0200_model_path.is_file() {
        anyhow::bail!("v0.71.1 v0.20 parent missing model.safetensors: {parent_v0200:?}");
    }

    let decoder_config = v0200_meta.inverse_config.clone();
    let v070_config = v070_meta.config.spectrum.clone();

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
    if train_groups.len() < V0711_BATCH || dev_groups.is_empty() {
        anyhow::bail!(
            "insufficient v0.71.1 identities: train={} dev={}",
            train_groups.len(),
            dev_groups.len()
        );
    }

    let smoke = mode == "smoke";
    let updates = if smoke {
        V0711_SMOKE_STEPS
    } else {
        V0711_PROBE_STEPS
    };
    let dev_count = if smoke {
        V0711_SMOKE_DEV_IDENTITIES
    } else {
        V0711_DEV_IDENTITIES
    }
    .min(dev_groups.len());
    let beam_width = if smoke {
        V0711_SMOKE_BEAM_WIDTH
    } else {
        V0711_BEAM_WIDTH
    };
    let dev_indices = select_identity_records(
        &dev_groups,
        &corpus.records,
        dev_count,
        V0711_SEED ^ 0x0711_d3f0_0000_0001,
    );
    let dev_fingerprint = format!("fnv1a64:{:016x}", index_fingerprint(&dev_indices));

    let max_precursor_mass =
        max_precursor_mass_for_groups(&train_groups, &dev_groups, &corpus.records)?;
    let suffix_lattice =
        ChemistrySuffixMassLattice::new(decoder_config.max_tokens, max_precursor_mass + 100.0)
            .map_err(anyhow::Error::msg)?;
    let chemistry_featurizer = ChemistryTransitionFeaturizer::new(&decoder_config, suffix_lattice)
        .map_err(anyhow::Error::msg)?;
    let suffix_feasible = dev_indices.iter().try_fold(0usize, |count, &index| {
        let record = &corpus.records[index];
        let mass = record_precursor_mass(record)?;
        let feasible = chemistry_featurizer
            .true_path_feasible(&record.peptidoform, mass)
            .map_err(anyhow::Error::msg)?;
        Ok::<usize, anyhow::Error>(count + usize::from(feasible))
    })?;
    if suffix_feasible != dev_indices.len() {
        anyhow::bail!(
            "v0.71.1 DEV chemistry suffix contract failed: feasible={suffix_feasible}/{}",
            dev_indices.len()
        );
    }

    let mut decoder_varmap = VarMap::new();
    let decoder = PeptideSpectrumChemistryDecoder::new(
        decoder_config.clone(),
        VarBuilder::from_varmap(&decoder_varmap, DType::F32, &device),
    )?;
    decoder_varmap.load(&v0200_model_path).with_context(|| {
        format!(
            "load exact v0.20 decoder checkpoint {}",
            v0200_model_path.display()
        )
    })?;

    let mut v070_varmap = VarMap::new();
    let v070_encoder = FoundationSpectrumEncoder::new(
        &v070_config,
        VarBuilder::from_varmap(&v070_varmap, DType::F32, &device).pp("spectrum_encoder"),
    )?;
    let v070_loaded = load_v070_spectrum_encoder(&v070_varmap, &v070_model_path, &device)?;

    let mut adapter_varmap = VarMap::new();
    let adapter = SpectrumResidualAdapter::new(
        v070_config.model_dim,
        decoder_config.spectrum.peak_feature_dim,
        VarBuilder::from_varmap(&adapter_varmap, DType::F32, &device),
    )?;
    let mut optimizer = FoundationAdamW::new(
        &adapter_varmap,
        FoundationAdamWConfig {
            learning_rate: V0711_LEARNING_RATE,
            weight_decay: V0711_WEIGHT_DECAY,
            ..FoundationAdamWConfig::default()
        },
    )?;
    let schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 250u64.min(updates.saturating_sub(1) as u64),
        total_steps: updates as u64,
        min_lr_ratio: 0.10,
    };

    let decoder_checksum_initial = varmap_checksum(&decoder_varmap)?;
    let v070_checksum_initial = varmap_checksum(&v070_varmap)?;
    let gate_initial = adapter.gate_value()?;
    if gate_initial != 0.0 {
        anyhow::bail!(
            "v0.71.1 residual gate must initialize exactly at zero, found {gate_initial}"
        );
    }

    let causal_collator = FoundationCausalCollator::new(decoder_config.clone())?;
    let decoder_spectrum_collator =
        FoundationSpectrumCollator::new(decoder_config.spectrum.clone())?;
    let mut v070_bridge_spectrum_config = v070_config.spectrum.clone();
    v070_bridge_spectrum_config.max_peaks = decoder_config.spectrum.max_peaks;
    let v070_spectrum_collator = FoundationSpectrumCollator::new(v070_bridge_spectrum_config)?;

    fs::create_dir_all(&output_root)?;
    let train_indices = train_groups
        .iter()
        .flat_map(|g| g.record_indices.iter().copied())
        .collect::<Vec<_>>();
    let train_availability = property_availability(&corpus.records, &train_indices);
    let dev_availability = property_availability(&corpus.records, &dev_indices);
    let holdout_reserved = benchmark
        .entries
        .iter()
        .filter(|e| e.partition == FoundationPartition::Test)
        .count();

    println!("v0711_version\tv0.71.1-warmstarted-spectrum-causal-decoder");
    println!("objective\t{V0711_OBJECTIVE}");
    println!("architecture\t{V0711_ARCHITECTURE}");
    println!("context_policy\t{V0711_CONTEXT_POLICY}");
    println!("device\t{device:?}");
    println!("mode\t{mode}");
    println!("parent_v070_checkpoint\t{}", parent_v070.display());
    println!(
        "parent_v070_completed_updates\t{}",
        v070_meta.completed_updates
    );
    println!(
        "parent_v070_dev_selection_score\t{:.8}",
        v070_meta.dev_selection_score
    );
    println!("parent_v070_spectrum_variables_loaded\t{v070_loaded}");
    println!("parent_v0200_checkpoint\t{}", parent_v0200.display());
    println!("parent_v0200_global_step\t{}", v0200_meta.global_step);
    println!("parent_v0200_objective\t{}", v0200_meta.objective);
    println!("parent_v0200_architecture\t{}", v0200_meta.architecture);
    println!("decoder_update_policy\tfrozen_exact_v0200_checkpoint");
    println!("v070_update_policy\tfrozen_selected_v070_spectrum_encoder");
    println!("adapter_update_policy\tprojection_plus_scalar_gate_only");
    println!(
        "adapter_optimizer_variables\t{}",
        optimizer.variable_count()
    );
    println!("adapter_input_dim\t{}", v070_config.model_dim);
    println!(
        "adapter_output_dim\t{}",
        decoder_config.spectrum.peak_feature_dim
    );
    println!(
        "adapter_injection_space\tv0200_raw_peak_feature_residual_before_native_spectrum_encoder"
    );
    println!("adapter_gate_initial\t{gate_initial:.8}");
    println!("v0200_decoder_model_dim\t{}", decoder_config.model_dim);
    println!("v070_spectrum_model_dim\t{}", v070_config.model_dim);
    println!("spectrum_collation_policy\tdual_collation_same_peak_grid_v0200_native_features_plus_v070_context_features");
    println!(
        "v0200_peak_feature_dim\t{}",
        decoder_config.spectrum.peak_feature_dim
    );
    println!(
        "v070_peak_feature_dim\t{}",
        v070_config.spectrum.peak_feature_dim
    );
    println!("v0200_max_peaks\t{}", decoder_config.spectrum.max_peaks);
    println!(
        "v070_original_max_peaks\t{}",
        v070_config.spectrum.max_peaks
    );
    println!("optional_rt_conditioning\tNO");
    println!("optional_ccs_conditioning\tNO");
    println!("train_eligible_unique_identities\t{}", train_groups.len());
    println!("dev_eligible_unique_identities\t{}", dev_groups.len());
    print_mass_feasibility("train_mass_feasibility", train_mass_audit);
    print_mass_feasibility("dev_mass_feasibility", dev_mass_audit);
    println!("dev_generation_identities\t{}", dev_indices.len());
    println!("dev_identity_fingerprint\t{dev_fingerprint}");
    println!(
        "dev_suffix_true_path_feasible\t{suffix_feasible}/{}",
        dev_indices.len()
    );
    print_property_availability("train_property_availability", train_availability);
    print_property_availability("dev_property_availability", dev_availability);
    println!("holdout_records_reserved_not_read\t{holdout_reserved}");
    println!("updates\t{updates}");
    println!("batch_size\t{V0711_BATCH}");
    println!("beam_width\t{beam_width}");
    println!("beam_top_k\t{V0711_TOP_K}");
    println!("mass_tolerance_da\t{V0711_MASS_TOLERANCE_DA}");
    println!("learning_rate\t{V0711_LEARNING_RATE}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let initial_teacher = evaluate_teacher(
        &decoder,
        &v070_encoder,
        &adapter,
        &chemistry_featurizer,
        &corpus.records,
        &dev_indices,
        V0711_BATCH,
        &causal_collator,
        &decoder_spectrum_collator,
        &v070_spectrum_collator,
        &device,
        V0711_SEED,
    )?;
    let (initial_generation, initial_outcomes) = evaluate_generation(
        &decoder,
        &v070_encoder,
        &adapter,
        &chemistry_featurizer,
        &corpus.records,
        &dev_indices,
        &causal_collator,
        &decoder_spectrum_collator,
        &v070_spectrum_collator,
        &decoder_config,
        beam_width,
        &device,
    )?;
    print_teacher("v0711_dev_teacher_initial", 0, initial_teacher);
    print_generation("v0711_dev_generation_initial", 0, initial_generation);
    save_probe_checkpoint(
        &output_root.join("model/initial"),
        &adapter_varmap,
        &V0711Metadata {
            version: V0711_VERSION,
            objective: V0711_OBJECTIVE.into(),
            architecture: V0711_ARCHITECTURE.into(),
            context_policy: V0711_CONTEXT_POLICY.into(),
            parent_v070_checkpoint: parent_v070.display().to_string(),
            parent_v0200_checkpoint: parent_v0200.display().to_string(),
            parent_v0200_global_step: v0200_meta.global_step,
            corpus_fingerprint: corpus_fingerprint.clone(),
            benchmark_manifest_fingerprint: benchmark_fingerprint.clone(),
            v070_config: v070_config.clone(),
            decoder_config: decoder_config.clone(),
            updates: 0,
            batch_size: V0711_BATCH,
            seed: V0711_SEED,
            learning_rate: V0711_LEARNING_RATE,
            beam_width,
            top_k: V0711_TOP_K,
            mass_tolerance_da: V0711_MASS_TOLERANCE_DA,
            dev_identity_count: dev_indices.len(),
            dev_identity_fingerprint: dev_fingerprint.clone(),
            residual_gate: adapter.gate_value()?,
            teacher: initial_teacher,
            generation: initial_generation,
        },
        &initial_outcomes,
    )?;

    let order = deterministic_order(train_groups.len(), V0711_SEED ^ 0x9e37_79b9_7f4a_7c15);
    let mut objective_sum = 0.0f64;
    let mut gap_sum = 0.0f64;
    for step in 1..=updates {
        let records = select_train_records(
            &corpus.records,
            &train_groups,
            &order,
            V0711_BATCH,
            step - 1,
            V0711_SEED ^ (step as u64).rotate_left(17),
        );
        let lr = schedule.learning_rate(V0711_LEARNING_RATE, (step - 1) as u64)?;
        optimizer.set_learning_rate(lr)?;
        let peptides = records
            .iter()
            .map(|r| r.peptidoform.clone())
            .collect::<Vec<_>>();
        let causal = causal_collator.collate(&peptides, &device)?;
        let components = chemistry_components(&records)?;
        let decoder_spectrum_batch =
            decoder_spectrum_collator.collate(&components.spectra, &device)?;
        let v070_spectrum_batch = v070_spectrum_collator.collate(&components.spectra, &device)?;
        let precursor = precursor_context(&records, &device)?;
        let matched_chemistry = chemistry_featurizer.teacher_forced(
            &components.peptides,
            &components.spectra,
            &components.precursor_masses,
            &components.charges,
            &device,
        )?;
        let matched_spectrum = fused_spectrum(
            &v070_encoder,
            &adapter,
            &decoder_spectrum_batch,
            &v070_spectrum_batch,
        )?;
        let matched_output = decoder.forward_t(
            &causal.input,
            &matched_spectrum,
            &precursor,
            &matched_chemistry,
            false,
        )?;
        let matched = foundation_causal_next_token_loss(&matched_output, &causal)?;

        let shuffled_order =
            foundation_direct_shuffled_order(records.len(), V0711_SEED ^ step as u64)?;
        let shuffled_spectra = shuffled_order
            .iter()
            .map(|&i| components.spectra[i].clone())
            .collect::<Vec<_>>();
        let shuffled_decoder_batch =
            decoder_spectrum_collator.collate(&shuffled_spectra, &device)?;
        let shuffled_v070_batch = v070_spectrum_collator.collate(&shuffled_spectra, &device)?;
        let shuffled_chemistry = chemistry_featurizer.teacher_forced(
            &components.peptides,
            &shuffled_spectra,
            &components.precursor_masses,
            &components.charges,
            &device,
        )?;
        let shuffled_spectrum = fused_spectrum(
            &v070_encoder,
            &adapter,
            &shuffled_decoder_batch,
            &shuffled_v070_batch,
        )?;
        let shuffled_output = decoder.forward_t(
            &causal.input,
            &shuffled_spectrum,
            &precursor,
            &shuffled_chemistry,
            false,
        )?;
        let shuffled = foundation_causal_next_token_loss(&shuffled_output, &causal)?;
        let loss = foundation_direct_conditioning_loss(
            &matched,
            &shuffled,
            FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
            FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
        )?;
        let loss_value = f64::from(loss.to_scalar::<f32>()?);
        let matched_value = f64::from(matched.to_scalar::<f32>()?);
        let shuffled_value = f64::from(shuffled.to_scalar::<f32>()?);
        let update = optimizer.backward_step(&loss, Some(V0711_MAX_GRADIENT_NORM))?;
        objective_sum += loss_value;
        gap_sum += shuffled_value - matched_value;
        if step <= 4 || step % 50 == 0 || step == updates {
            let rank = foundation_direct_prefix_competitive_loss(&matched_output, &causal, 0.0)?;
            println!(
                "v0711_train\tupdate={step}\tlr={:.8}\tobjective={loss_value:.6}\tmatched_nll={matched_value:.6}\tshuffled_nll={shuffled_value:.6}\tconditioning_gap={:.6}\tlegal_rank={:.4}\tlegal_top1={:.6}\tadapter_gate={:.8}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                update.learning_rate,
                shuffled_value - matched_value,
                rank.mean_legal_rank,
                rank.top1_fraction,
                adapter.gate_value()?,
                update.gradient_norm,
                update.gradient_scale,
            );
        }
    }

    let decoder_checksum = varmap_checksum(&decoder_varmap)?;
    let v070_checksum = varmap_checksum(&v070_varmap)?;
    assert_frozen_checksum("v0200_decoder", decoder_checksum_initial, decoder_checksum)?;
    assert_frozen_checksum(
        "v070_spectrum_encoder",
        v070_checksum_initial,
        v070_checksum,
    )?;

    let final_teacher = evaluate_teacher(
        &decoder,
        &v070_encoder,
        &adapter,
        &chemistry_featurizer,
        &corpus.records,
        &dev_indices,
        V0711_BATCH,
        &causal_collator,
        &decoder_spectrum_collator,
        &v070_spectrum_collator,
        &device,
        V0711_SEED ^ updates as u64,
    )?;
    let (final_generation, final_outcomes) = evaluate_generation(
        &decoder,
        &v070_encoder,
        &adapter,
        &chemistry_featurizer,
        &corpus.records,
        &dev_indices,
        &causal_collator,
        &decoder_spectrum_collator,
        &v070_spectrum_collator,
        &decoder_config,
        beam_width,
        &device,
    )?;
    print_teacher("v0711_dev_teacher_final", updates, final_teacher);
    print_generation("v0711_dev_generation_final", updates, final_generation);
    println!(
        "v0711_mean_train_objective\t{:.8}",
        objective_sum / updates as f64
    );
    println!(
        "v0711_mean_train_conditioning_gap\t{:.8}",
        gap_sum / updates as f64
    );
    println!("v0711_adapter_gate_final\t{:.8}", adapter.gate_value()?);

    save_probe_checkpoint(
        &output_root.join("model/final"),
        &adapter_varmap,
        &V0711Metadata {
            version: V0711_VERSION,
            objective: V0711_OBJECTIVE.into(),
            architecture: V0711_ARCHITECTURE.into(),
            context_policy: V0711_CONTEXT_POLICY.into(),
            parent_v070_checkpoint: parent_v070.display().to_string(),
            parent_v0200_checkpoint: parent_v0200.display().to_string(),
            parent_v0200_global_step: v0200_meta.global_step,
            corpus_fingerprint,
            benchmark_manifest_fingerprint: benchmark_fingerprint,
            v070_config,
            decoder_config,
            updates,
            batch_size: V0711_BATCH,
            seed: V0711_SEED,
            learning_rate: V0711_LEARNING_RATE,
            beam_width,
            top_k: V0711_TOP_K,
            mass_tolerance_da: V0711_MASS_TOLERANCE_DA,
            dev_identity_count: dev_indices.len(),
            dev_identity_fingerprint: dev_fingerprint,
            residual_gate: adapter.gate_value()?,
            teacher: final_teacher,
            generation: final_generation,
        },
        &final_outcomes,
    )?;

    let selection_gain = final_generation.selection_score() - initial_generation.selection_score();
    let returned_delta = final_generation.returned_fraction - initial_generation.returned_fraction;
    println!("v0711_selection_gain\t{selection_gain:.8}");
    println!("v0711_returned_fraction_delta\t{returned_delta:.8}");
    println!(
        "v0711_conditioning_gap_final\t{:.8}",
        final_teacher.conditioning_gap
    );
    println!(
        "v0711_gate_nonzero\t{}",
        pass_fail(adapter.gate_value()?.abs() > 1.0e-6)
    );
    println!(
        "v0711_gate_conditioning_ge_0_02\t{}",
        pass_fail(final_teacher.conditioning_gap >= 0.02)
    );
    println!(
        "v0711_gate_returned_fraction_ge_0_50\t{}",
        pass_fail(final_generation.returned_fraction >= 0.50)
    );
    println!(
        "v0711_gate_direct_il_top10_gt_0\t{}",
        pass_fail(final_generation.il_top10 > 0.0)
    );
    println!(
        "v0711_probe_decision\t{}",
        if final_teacher.conditioning_gap >= 0.02
            && final_generation.returned_fraction >= 0.50
            && final_generation.il_top10 > 0.0
        {
            "PASS_WARMSTART_BRIDGE"
        } else {
            "FAIL_WARMSTART_BRIDGE"
        }
    );
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    Ok(())
}

fn read_v070_metadata(checkpoint: &Path) -> Result<V070Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.70 metadata {path:?}"))?,
    )
    .map_err(anyhow::Error::from)
}

fn validate_v070_parent(metadata: &V070Metadata, corpus: &str, benchmark: &str) -> Result<()> {
    if metadata.version != 700
        || metadata.smoke_mode
        || metadata.completed_epochs != 6
        || metadata.completed_updates != 6000
    {
        anyhow::bail!("v0.71.1 requires selected full v0.70 epoch6/update6000 checkpoint");
    }
    if metadata.corpus_fingerprint != corpus || metadata.benchmark_manifest_fingerprint != benchmark
    {
        anyhow::bail!("v0.71.1 v0.70 parent provenance differs from current corpus/benchmark");
    }
    if metadata.config.spectrum.model_dim != 320
        || metadata.config.alignment_dim != 192
        || (metadata.config.temperature - 0.07).abs() > 1e-12
    {
        anyhow::bail!("v0.71.1 v0.70 parent architecture contract drift");
    }
    Ok(())
}

fn read_v0200_metadata(checkpoint: &Path) -> Result<V0200Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.20 metadata {path:?}"))?,
    )
    .map_err(anyhow::Error::from)
}

fn validate_v0200_parent(metadata: &V0200Metadata) -> Result<()> {
    if metadata.version != "v0.20.0"
        || metadata.objective != FOUNDATION_CHEMISTRY_DECODER_OBJECTIVE_V0200
        || metadata.architecture != FOUNDATION_CHEMISTRY_DECODER_ARCHITECTURE_V0200
        || metadata.global_step != 4000
        || metadata.inverse_config.model_dim != 96
    {
        anyhow::bail!(
            "v0.71.1 requires the accepted trained v0.20 chemistry checkpoint at step4000/model_dim96"
        );
    }
    metadata
        .inverse_config
        .validate()
        .map_err(anyhow::Error::msg)
}

fn load_v070_spectrum_encoder(
    varmap: &VarMap,
    checkpoint_path: &Path,
    device: &Device,
) -> Result<usize> {
    let checkpoint = candle_core::safetensors::load(checkpoint_path, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.71.1 v0.70 VarMap lock poisoned"))?;
    let mut loaded = 0usize;
    for (name, variable) in data.iter() {
        let parent_name = format!("student_v070.{name}");
        let tensor = checkpoint
            .get(&parent_name)
            .ok_or_else(|| anyhow::anyhow!("v0.70 checkpoint missing {parent_name}"))?;
        if tensor.dims() != variable.as_tensor().dims() {
            anyhow::bail!(
                "v0.71.1 v0.70 shape mismatch for {name}: model {:?}, parent {:?}",
                variable.as_tensor().dims(),
                tensor.dims()
            );
        }
        variable.set(tensor)?;
        loaded += 1;
    }
    if loaded == 0 {
        anyhow::bail!("v0.71.1 loaded zero v0.70 spectrum tensors");
    }
    Ok(loaded)
}

fn fused_spectrum(
    v070_encoder: &FoundationSpectrumEncoder,
    adapter: &SpectrumResidualAdapter,
    decoder_spectrum: &FoundationSpectrumBatch,
    v070_spectrum: &FoundationSpectrumBatch,
) -> Result<FoundationSpectrumBatch> {
    let encoding = v070_encoder.forward_t(v070_spectrum, false)?;
    let detached_encoding = redeem_properties::foundation::FoundationSpectrumEncoding {
        peak_embeddings: encoding.peak_embeddings.detach(),
        spectrum_embedding: encoding.spectrum_embedding.detach(),
    };
    let residual = adapter.peak_residual(&detached_encoding)?;
    let (batch, peaks, features) = decoder_spectrum.peak_features.dims3()?;
    let (v070_batch, v070_peaks, _) = v070_spectrum.peak_features.dims3()?;
    if (v070_batch, v070_peaks) != (batch, peaks) {
        anyhow::bail!(
            "v0.71.1 decoder/v0.70 retained peak-grid mismatch: decoder=({batch},{peaks}) v070=({v070_batch},{v070_peaks})"
        );
    }
    if residual.dims3()? != (batch, peaks, features) {
        anyhow::bail!(
            "v0.71.1 projected peak residual shape {:?} does not match raw peak features ({batch},{peaks},{features})",
            residual.dims3()?
        );
    }
    let mask = decoder_spectrum
        .peak_mask
        .unsqueeze(2)?
        .broadcast_as((batch, peaks, features))?;
    let residual = residual.broadcast_mul(&mask)?;
    Ok(FoundationSpectrumBatch {
        peak_features: (&decoder_spectrum.peak_features + &residual)?,
        peak_mask: decoder_spectrum.peak_mask.clone(),
        mz: decoder_spectrum.mz.clone(),
        intensity: decoder_spectrum.intensity.clone(),
    })
}

fn chemistry_components(records: &[&FoundationTrainingRecord]) -> Result<ChemistryComponents> {
    let peptides = records
        .iter()
        .map(|r| r.peptidoform.clone())
        .collect::<Vec<_>>();
    let spectra = record_spectra(records)?;
    let precursor_masses = records
        .iter()
        .map(|r| record_precursor_mass(r))
        .collect::<Result<Vec<_>>>()?;
    let charges = records
        .iter()
        .map(|r| {
            r.context
                .charge
                .ok_or_else(|| anyhow::anyhow!("v0.71.1 chemistry record lacks charge"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ChemistryComponents {
        peptides,
        spectra,
        precursor_masses,
        charges,
    })
}

fn record_precursor_mass(record: &FoundationTrainingRecord) -> Result<f64> {
    let charge = record
        .context
        .charge
        .ok_or_else(|| anyhow::anyhow!("record lacks charge"))?;
    let mz = record
        .context
        .precursor_mz
        .ok_or_else(|| anyhow::anyhow!("record lacks precursor m/z"))?;
    foundation_precursor_neutral_mass(f64::from(mz), charge).map_err(anyhow::Error::msg)
}

fn max_precursor_mass_for_groups(
    train: &[IdentityGroup],
    dev: &[IdentityGroup],
    records: &[FoundationTrainingRecord],
) -> Result<f64> {
    let mut max_mass = 0.0f64;
    for index in train
        .iter()
        .chain(dev.iter())
        .flat_map(|g| g.record_indices.iter().copied())
    {
        max_mass = max_mass.max(record_precursor_mass(&records[index])?);
    }
    if max_mass <= 0.0 {
        anyhow::bail!("v0.71.1 max precursor mass audit found no records");
    }
    Ok(max_mass)
}

fn varmap_checksum(varmap: &VarMap) -> Result<f64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.71.1 VarMap lock poisoned"))?;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for variable in data.values() {
        if variable.dtype().is_float() {
            sum += f64::from(variable.as_tensor().sum_all()?.to_scalar::<f32>()?);
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.71.1 checksum saw zero floating variables");
    }
    Ok(sum)
}

fn assert_frozen_checksum(label: &str, initial: f64, current: f64) -> Result<()> {
    let delta = (current - initial).abs();
    let tolerance = 1.0e-6 * initial.abs().max(1.0);
    if delta > tolerance {
        anyhow::bail!("v0.71.1 frozen {label} changed: initial={initial:.8} current={current:.8} delta={delta:.8}");
    }
    println!("v0711_freeze_audit\tcomponent={label}\tstatus=PASS\tchecksum={current:.8}\tdelta={delta:.8}");
    Ok(())
}

fn reject_protected_path(path: &Path) -> Result<()> {
    let text = path.to_string_lossy().to_ascii_lowercase();
    if text.contains("historical_test") || text.contains("/test/") || text.contains("train_holdout")
    {
        anyhow::bail!("v0.71.1 refuses protected path {path:?}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_teacher(
    decoder: &PeptideSpectrumChemistryDecoder,
    v070_encoder: &FoundationSpectrumEncoder,
    adapter: &SpectrumResidualAdapter,
    chemistry_featurizer: &ChemistryTransitionFeaturizer,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    causal_collator: &FoundationCausalCollator,
    decoder_spectrum_collator: &FoundationSpectrumCollator,
    v070_spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
    seed: u64,
) -> Result<TeacherMetrics> {
    let mut nll_weighted = 0.0;
    let mut gap_weighted = 0.0;
    let mut rank_weighted = 0.0;
    let mut top1_weighted = 0.0;
    let mut active_tokens = 0usize;
    let mut conditioning_tokens = 0usize;
    let mut correct_tokens = 0usize;
    let mut exact_sequences = 0usize;
    let mut sequences = 0usize;

    for (chunk_index, chunk) in indices.chunks(batch_size).enumerate() {
        let selected = chunk.iter().map(|&i| &records[i]).collect::<Vec<_>>();
        let peptides = selected
            .iter()
            .map(|r| r.peptidoform.clone())
            .collect::<Vec<_>>();
        let causal = causal_collator.collate(&peptides, device)?;
        let components = chemistry_components(&selected)?;
        let decoder_spectrum_batch =
            decoder_spectrum_collator.collate(&components.spectra, device)?;
        let v070_spectrum_batch = v070_spectrum_collator.collate(&components.spectra, device)?;
        let precursor = precursor_context(&selected, device)?;
        let chemistry = chemistry_featurizer.teacher_forced(
            &components.peptides,
            &components.spectra,
            &components.precursor_masses,
            &components.charges,
            device,
        )?;
        let fused = fused_spectrum(
            v070_encoder,
            adapter,
            &decoder_spectrum_batch,
            &v070_spectrum_batch,
        )?;
        let output = decoder.forward_t(&causal.input, &fused, &precursor, &chemistry, false)?;
        let nll = foundation_causal_next_token_loss(&output, &causal)?;
        let rank = foundation_direct_prefix_competitive_loss(&output, &causal, 0.0)?;

        let shuffled_nll = if selected.len() >= 2 {
            let order =
                foundation_direct_shuffled_order(selected.len(), seed ^ chunk_index as u64)?;
            let shuffled_spectra = order
                .iter()
                .map(|&i| components.spectra[i].clone())
                .collect::<Vec<_>>();
            let shuffled_decoder_batch =
                decoder_spectrum_collator.collate(&shuffled_spectra, device)?;
            let shuffled_v070_batch = v070_spectrum_collator.collate(&shuffled_spectra, device)?;
            let shuffled_chemistry = chemistry_featurizer.teacher_forced(
                &components.peptides,
                &shuffled_spectra,
                &components.precursor_masses,
                &components.charges,
                device,
            )?;
            let shuffled_fused = fused_spectrum(
                v070_encoder,
                adapter,
                &shuffled_decoder_batch,
                &shuffled_v070_batch,
            )?;
            let shuffled_output = decoder.forward_t(
                &causal.input,
                &shuffled_fused,
                &precursor,
                &shuffled_chemistry,
                false,
            )?;
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
        let nll_value = f64::from(nll.to_scalar::<f32>()?);
        nll_weighted += nll_value * chunk_active as f64;
        if let Some(shuffled_nll) = shuffled_nll {
            gap_weighted +=
                (f64::from(shuffled_nll.to_scalar::<f32>()?) - nll_value) * chunk_active as f64;
            conditioning_tokens += chunk_active;
        }
        rank_weighted += rank.mean_legal_rank * chunk_active as f64;
        top1_weighted += rank.top1_fraction * chunk_active as f64;
        active_tokens += chunk_active;
        for row in 0..selected.len() {
            let mut exact = true;
            for pos in 0..targets[row].len() {
                if masks[row][pos] <= 0.0 {
                    break;
                }
                let pred = argmax(&logits[row][pos]) as u32;
                if pred == targets[row][pos] {
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
    Ok(TeacherMetrics {
        token_nll: nll_weighted / token_denom,
        token_accuracy: correct_tokens as f64 / token_denom,
        sequence_exact: exact_sequences as f64 / sequences.max(1) as f64,
        conditioning_gap: if conditioning_tokens > 0 {
            gap_weighted / conditioning_tokens as f64
        } else {
            0.0
        },
        legal_rank: rank_weighted / token_denom,
        legal_top1: top1_weighted / token_denom,
        active_tokens,
        sequences,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_generation(
    decoder: &PeptideSpectrumChemistryDecoder,
    v070_encoder: &FoundationSpectrumEncoder,
    adapter: &SpectrumResidualAdapter,
    chemistry_featurizer: &ChemistryTransitionFeaturizer,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    decoder_spectrum_collator: &FoundationSpectrumCollator,
    v070_spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    beam_width: usize,
    device: &Device,
) -> Result<(GenerationMetrics, Vec<GenerationOutcome>)> {
    let vocabulary = FoundationDiffusionVocabulary;
    let ks = [1usize, 5usize, 10usize];
    let mut returned_records = 0usize;
    let mut returned_beams = 0usize;
    let mut mass_valid = 0usize;
    let mut peptidoform_hits = [0usize; 3];
    let mut sequence_hits = [0usize; 3];
    let mut il_hits = [0usize; 3];
    let mut edit_sum = 0usize;
    let mut normalized_sum = 0.0f64;
    let mut outcomes = Vec::with_capacity(indices.len());

    for &index in indices {
        let record = &records[index];
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("v0.71.1 generation record lacks spectrum"))?;
        let decoder_spectrum_batch =
            decoder_spectrum_collator.collate(&[spectrum.clone()], device)?;
        let v070_spectrum_batch = v070_spectrum_collator.collate(&[spectrum.clone()], device)?;
        let one = vec![record];
        let precursor = precursor_context(&one, device)?;
        let fused = fused_spectrum(
            v070_encoder,
            adapter,
            &decoder_spectrum_batch,
            &v070_spectrum_batch,
        )?;
        let context = decoder.prepare_context(&fused, &precursor, false)?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("generation record lacks charge"))?;
        let neutral_mass = record_precursor_mass(record)?;
        let beam = foundation_direct_beam_search(
            neutral_mass,
            DirectDecoderBeamConfig {
                beam_width,
                top_k: V0711_TOP_K,
                mass_tolerance_da: V0711_MASS_TOLERANCE_DA,
                max_tokens: config.max_tokens,
            },
            |prefixes| {
                let input = causal_collator
                    .collate_compact_prefix_rows(prefixes, device)
                    .map_err(|e| e.to_string())?;
                let chemistry = chemistry_featurizer
                    .next_prefixes(prefixes, &spectrum, neutral_mass, charge, device)
                    .map_err(|e| e.to_string())?;
                let mut rows = decoder
                    .forward_next_t_with_context(&input, &context, &chemistry, false)
                    .and_then(|t| t.to_vec2::<f32>())
                    .map_err(|e| e.to_string())?;
                chemistry_featurizer.mask_infeasible_next_logits(
                    prefixes,
                    &mut rows,
                    neutral_mass,
                )?;
                Ok(rows)
            },
        )
        .map_err(anyhow::Error::msg)?;

        returned_records += usize::from(!beam.is_empty());
        returned_beams += beam.len();
        mass_valid += beam
            .iter()
            .filter(|c| c.mass_error_da.abs() <= V0711_MASS_TOLERANCE_DA)
            .count();
        let decoded = beam
            .iter()
            .filter_map(|c| vocabulary.decode(&c.tokens).ok())
            .collect::<Vec<_>>();
        for (slot, &k) in ks.iter().enumerate() {
            let limit = k.min(decoded.len());
            peptidoform_hits[slot] +=
                usize::from(decoded[..limit].iter().any(|p| p == &record.peptidoform));
            sequence_hits[slot] += usize::from(
                decoded[..limit]
                    .iter()
                    .any(|p| p.sequence == record.peptidoform.sequence),
            );
            il_hits[slot] +=
                usize::from(decoded[..limit].iter().any(|p| {
                    il_sequence(&p.sequence) == il_sequence(&record.peptidoform.sequence)
                }));
        }
        let (edit, normalized) = if let Some(first) = decoded.first() {
            let e = levenshtein(&first.sequence, &record.peptidoform.sequence);
            (
                e,
                e as f64
                    / first
                        .sequence
                        .len()
                        .max(record.peptidoform.sequence.len())
                        .max(1) as f64,
            )
        } else {
            let e = record.peptidoform.sequence.len();
            (e, 1.0)
        };
        edit_sum += edit;
        normalized_sum += normalized;
        outcomes.push(GenerationOutcome {
            charge,
            length: record.peptidoform.sequence.len(),
            modified: !record.peptidoform.modifications.is_empty(),
            returned: !beam.is_empty(),
            peptidoform_top1: decoded.first().is_some_and(|p| p == &record.peptidoform),
            peptidoform_top10: decoded[..10.min(decoded.len())]
                .iter()
                .any(|p| p == &record.peptidoform),
            sequence_top1: decoded
                .first()
                .is_some_and(|p| p.sequence == record.peptidoform.sequence),
            sequence_top10: decoded[..10.min(decoded.len())]
                .iter()
                .any(|p| p.sequence == record.peptidoform.sequence),
            il_top1: decoded.first().is_some_and(|p| {
                il_sequence(&p.sequence) == il_sequence(&record.peptidoform.sequence)
            }),
            il_top10: decoded[..10.min(decoded.len())]
                .iter()
                .any(|p| il_sequence(&p.sequence) == il_sequence(&record.peptidoform.sequence)),
            top1_edit_distance: edit,
            top1_normalized_edit_distance: normalized,
        });
    }
    let n = indices.len().max(1) as f64;
    Ok((
        GenerationMetrics {
            identities: indices.len(),
            returned_fraction: returned_records as f64 / n,
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
            mean_top1_normalized_edit_distance: normalized_sum / n,
            mass_valid_fraction: if returned_beams > 0 {
                mass_valid as f64 / returned_beams as f64
            } else {
                0.0
            },
        },
        outcomes,
    ))
}

fn save_probe_checkpoint(
    directory: &Path,
    adapter_varmap: &VarMap,
    metadata: &V0711Metadata,
    outcomes: &[GenerationOutcome],
) -> Result<()> {
    fs::create_dir_all(directory)?;
    adapter_varmap.save(directory.join("adapter.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    write_teacher_metrics(
        directory.join("dev_teacher.tsv").as_path(),
        metadata.teacher,
    )?;
    write_generation_metrics(
        directory.join("dev_generation.tsv").as_path(),
        metadata.generation,
    )?;
    write_generation_stratified(
        directory.join("dev_generation_stratified.tsv").as_path(),
        outcomes,
    )?;
    Ok(())
}

fn write_teacher_metrics(path: &Path, m: TeacherMetrics) -> Result<()> {
    fs::write(path, format!(
        "metric\tvalue\ntoken_nll\t{:.12}\ntoken_accuracy\t{:.12}\nsequence_exact\t{:.12}\nconditioning_gap\t{:.12}\nlegal_rank\t{:.12}\nlegal_top1\t{:.12}\nactive_tokens\t{}\nsequences\t{}\n",
        m.token_nll, m.token_accuracy, m.sequence_exact, m.conditioning_gap,
        m.legal_rank, m.legal_top1, m.active_tokens, m.sequences,
    ))?;
    Ok(())
}

fn write_generation_metrics(path: &Path, m: GenerationMetrics) -> Result<()> {
    fs::write(path, format!(
        "metric\tvalue\nidentities\t{}\nreturned_fraction\t{:.12}\npeptidoform_top1\t{:.12}\npeptidoform_top5\t{:.12}\npeptidoform_top10\t{:.12}\nsequence_top1\t{:.12}\nsequence_top5\t{:.12}\nsequence_top10\t{:.12}\nil_top1\t{:.12}\nil_top5\t{:.12}\nil_top10\t{:.12}\nmean_top1_edit_distance\t{:.12}\nmean_top1_normalized_edit_distance\t{:.12}\nmass_valid_fraction\t{:.12}\nselection_score\t{:.12}\n",
        m.identities, m.returned_fraction, m.peptidoform_top1, m.peptidoform_top5,
        m.peptidoform_top10, m.sequence_top1, m.sequence_top5, m.sequence_top10,
        m.il_top1, m.il_top5, m.il_top10, m.mean_top1_edit_distance,
        m.mean_top1_normalized_edit_distance, m.mass_valid_fraction, m.selection_score(),
    ))?;
    Ok(())
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
        if abs_error > V0711_MASS_TOLERANCE_DA {
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
                .ok_or_else(|| anyhow::anyhow!("v0.71.1 selected record lacks observed spectrum"))
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
                "v0.71.1 precursor context tensor {name} must be F32, observed {:?}",
                tensor.dtype()
            );
        }
    }
    Ok(context)
}

#[allow(clippy::too_many_arguments)]
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
    fn v0711_gate_is_predeclared_zero() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let adapter = SpectrumResidualAdapter::new(
            320,
            32,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )
        .unwrap();
        assert_eq!(adapter.gate_value().unwrap(), 0.0);
    }

    #[test]
    fn v0711_selection_prioritizes_top1() {
        let a = GenerationMetrics {
            il_top1: 0.1,
            il_top10: 0.1,
            ..GenerationMetrics::default()
        };
        let b = GenerationMetrics {
            il_top1: 0.0,
            il_top10: 0.2,
            ..GenerationMetrics::default()
        };
        assert!(a.selection_score() > b.selection_score());
    }

    #[test]
    fn v0711_il_equivalence_collapses_isoleucine() {
        assert_eq!(il_sequence("PEPTIDE"), il_sequence("PEPTLDE"));
    }
}
