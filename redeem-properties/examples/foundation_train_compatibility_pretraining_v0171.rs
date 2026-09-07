//! Train the v0.17.1 proposal-scale end-to-end spectrum-peptide compatibility representation.
//!
//! This experiment keeps the v0.17.0 end-to-end compatibility architecture but
//! changes the pretraining discrimination task from 8-way to proposal-scale
//! 128-way lists. It does not regenerate
//! proposal candidates and it never consumes TEST. Instead it warm-starts an
//! isolated, trainable copy of the accepted chemistry-aware peptide encoder and
//! observed-spectrum encoder, then pretrains them end to end with local
//! residue/cleavage cross-modal interaction using TRAIN-only precursor-compatible
//! hard peptide negatives.
//!
//! VALIDATION candidate generation remains frozen at v0.13.23. The final model
//! is evaluated as a standalone compatibility scorer over the existing top128
//! validation candidate windows so representation quality is measured directly,
//! without tuning a legacy-score blend.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_compatibility_listwise_loss, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, load_compatibility_from_unified_checkpoint,
    load_foundation_corpus, parse_modified_peptide, read_foundation_training_run_config,
    FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationConfig,
    FoundationDiffusionConfig, FoundationPartition, FoundationSpectrum, FoundationSpectrumCollator,
    FoundationSpectrumPeptideCompatibilityModel, FoundationTrainingRecord, PeptideGraphFeaturizer,
    PeptidoformInput, PrecursorContextBatch,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = "v0.17.1";
const RESIDUE_INTERACTION_LAYERS: usize = 2;
const CLEAVAGE_INTERACTION_LAYERS: usize = 1;
const HARD_NEGATIVES_PER_ANCHOR: usize = 127;
const HARD_NEGATIVE_POOL: usize = 127;
const HARD_NEGATIVE_MASS_WINDOW_DA: f64 = 0.10;
const MIN_FULL_HARD_NEGATIVE_GROUPS: usize = 2_048;
const FULL_TRAIN_STEPS: usize = 4_000;
const FULL_ANCHOR_BATCH: usize = 8;
const SMOKE_TRAIN_STEPS: usize = 2;
const SMOKE_ANCHOR_BATCH: usize = 2;
const SMOKE_VALIDATION_GROUPS: usize = 4;
const VALIDATION_INTERACTION_WINDOW: usize = 128;
const VALIDATION_ENCODE_BATCH: usize = 32;
const LEARNING_RATE: f64 = 1.0e-4;
const WEIGHT_DECAY: f64 = 1.0e-4;
const MAX_GRADIENT_NORM: f64 = 5.0;
const LOG_EVERY_STEPS: usize = 100;
const SEED: u64 = 20_260_917;

const REQUIRED_LITERAL_TOP1: usize = 28;
const REQUIRED_IL_TOP1: usize = 42;
const REQUIRED_LITERAL_ORACLE: usize = 44;
const REQUIRED_IL_ORACLE: usize = 54;
const REQUIRED_LEGACY_LITERAL_TOP1: usize = 24;
const REQUIRED_LEGACY_IL_TOP1: usize = 38;

const FNV1A64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Full,
    Smoke,
}

impl RunMode {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("full").trim().to_ascii_lowercase().as_str() {
            "full" => Ok(Self::Full),
            "smoke" => Ok(Self::Smoke),
            other => anyhow::bail!("unsupported v0.17.1 mode {other:?}; expected full or smoke"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Smoke => "smoke",
        }
    }

    fn train_steps(self) -> usize {
        match self {
            Self::Full => FULL_TRAIN_STEPS,
            Self::Smoke => SMOKE_TRAIN_STEPS,
        }
    }

    fn anchor_batch(self) -> usize {
        match self {
            Self::Full => FULL_ANCHOR_BATCH,
            Self::Smoke => SMOKE_ANCHOR_BATCH,
        }
    }

    fn validation_limit(self) -> Option<usize> {
        match self {
            Self::Full => None,
            Self::Smoke => Some(SMOKE_VALIDATION_GROUPS),
        }
    }
}

#[derive(Debug, Deserialize)]
struct UnifiedCheckpointMetadata {
    forward_config: FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone)]
struct HardNegativeCandidate {
    record_index: usize,
    absolute_mass_error_da: f64,
}

#[derive(Debug, Clone)]
struct HardNegativeGroup {
    anchor_index: usize,
    negative_pool: Vec<HardNegativeCandidate>,
}

#[derive(Debug, Clone)]
struct CandidateRow {
    sequence: String,
    modifications: String,
    exact: bool,
    il_exact: bool,
    legacy_score: f64,
    legacy_rank: usize,
}

#[derive(Debug, Clone)]
struct CandidateGroup {
    record_index: usize,
    rows: Vec<CandidateRow>,
}

#[derive(Debug, Clone, Copy)]
struct ValidationContract {
    records: usize,
    oracle_exact: usize,
    oracle_il: usize,
    legacy_top1_exact: usize,
    legacy_top1_il: usize,
    exact_in_window: usize,
    il_in_window: usize,
}

#[derive(Debug, Clone, Copy)]
struct ValidationMetrics {
    records: usize,
    oracle_exact: usize,
    oracle_il: usize,
    legacy_top1_exact: usize,
    legacy_top1_il: usize,
    compatibility_top1_exact: usize,
    compatibility_top1_il: usize,
    exact_in_window: usize,
    il_in_window: usize,
}

#[derive(Debug, Serialize)]
struct CompatibilityMetadata {
    version: String,
    run_mode: String,
    objective: String,
    architecture: String,
    representation_level_change: String,
    proposal_policy: String,
    candidate_generation: String,
    validation_scoring: String,
    validation_selection_policy: String,
    test_partition_consumed: bool,
    unified_checkpoint: String,
    training_yaml: String,
    validation_candidate_tsv: String,
    seed: u64,
    model_dim: usize,
    residue_interaction_layers: usize,
    cleavage_interaction_layers: usize,
    hard_negatives_per_anchor: usize,
    hard_negative_pool: usize,
    training_list_width: usize,
    candidate_pairs_per_optimizer_step: usize,
    negative_sampling_policy: String,
    hard_negative_mass_window_da: f64,
    train_steps: usize,
    anchor_batch: usize,
    learning_rate: f64,
    weight_decay: f64,
    max_gradient_norm: f64,
    validation_interaction_window: usize,
    train_partition_records: usize,
    train_spectrum_anchors: usize,
    train_hard_negative_groups: usize,
    unique_anchor_groups_seen: usize,
    peptide_encoder_warm_started_variables: usize,
    spectrum_encoder_warm_started_variables: usize,
    fresh_compatibility_variables: usize,
    initialization_fingerprint: String,
    final_validation_records: usize,
    final_literal_top1: usize,
    final_il_top1: usize,
    validation_oracle_literal: usize,
    validation_oracle_il: usize,
    gate: String,
}

#[derive(Debug)]
struct TrainingBatch {
    peptide_batch: redeem_properties::foundation::FoundationBatch,
    spectrum_batch: redeem_properties::foundation::FoundationSpectrumBatch,
    precursor_batch: PrecursorContextBatch,
    anchor_record_indices: Vec<usize>,
    mean_selected_negative_mass_error_da: f64,
    max_selected_negative_mass_error_da: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_train_compatibility_pretraining RUN.yaml UNIFIED_CHECKPOINT VALIDATION_CANDIDATES.tsv OUTPUT_DIR [full|smoke]"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let unified_checkpoint = PathBuf::from(&args[2]);
    let validation_path = PathBuf::from(&args[3]);
    let output_dir = PathBuf::from(&args[4]);
    let mode = RunMode::parse(args.get(5).map(String::as_str))?;

    reject_test_path(&validation_path)?;

    let run = read_foundation_training_run_config(&training_yaml)
        .with_context(|| format!("read foundation run config {training_yaml:?}"))?;
    let mut corpus = load_foundation_corpus(&run.corpus).context("load foundation corpus")?;
    log_host_memory("after_corpus_load");
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("read benchmark manifest {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;
    log_host_memory("after_benchmark_validation");

    let validation_groups = read_candidate_groups(&validation_path)
        .with_context(|| format!("read validation candidate TSV {validation_path:?}"))?;
    if validation_groups.is_empty() {
        anyhow::bail!("v0.17.1 validation candidate TSV contains no mass-valid groups");
    }
    verify_partition(
        &validation_groups,
        &benchmark,
        FoundationPartition::Validation,
        "VALIDATION",
    )?;
    let validation_contract = validate_candidate_contract(&validation_groups)?;

    let metadata_path = unified_checkpoint.join("metadata.yaml");
    let metadata: UnifiedCheckpointMetadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("read unified metadata {metadata_path:?}"))?,
    )?;
    metadata
        .forward_config
        .validate()
        .map_err(anyhow::Error::msg)?;
    metadata
        .inverse_config
        .validate()
        .map_err(anyhow::Error::msg)?;
    if metadata.forward_config.model_dim != 96 || metadata.inverse_config.model_dim != 96 {
        anyhow::bail!(
            "v0.17.1 is anchored to the accepted 96-d unified checkpoint; forward={} inverse={}",
            metadata.forward_config.model_dim,
            metadata.inverse_config.model_dim
        );
    }
    if metadata.forward_config.model_dim != metadata.inverse_config.model_dim {
        anyhow::bail!("v0.17.1 requires matching forward/inverse model widths");
    }

    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = FoundationSpectrumPeptideCompatibilityModel::new(
        metadata.forward_config.clone(),
        metadata.inverse_config.clone(),
        RESIDUE_INTERACTION_LAYERS,
        CLEAVAGE_INTERACTION_LAYERS,
        vb,
    )?;
    let warm_start = load_compatibility_from_unified_checkpoint(
        &varmap,
        &unified_checkpoint.join("model.safetensors"),
        &device,
    )?;
    initialize_fresh_compatibility_variables(&varmap, SEED, &device)?;
    let initialization_fingerprint = compatibility_model_fingerprint(&varmap)?;
    log_host_memory("after_model_initialization");

    let featurizer = PeptideGraphFeaturizer::new(metadata.forward_config.clone())?;
    let spectrum_collator =
        FoundationSpectrumCollator::new(metadata.inverse_config.spectrum.clone())?;

    let train_indices = benchmark.partition_indices(FoundationPartition::Train);
    drop(benchmark);
    let (hard_groups, mining_stats) = build_hard_negative_groups(
        &corpus.records,
        &train_indices,
        metadata.forward_config.max_sequence_len,
    )?;
    if hard_groups.len() < mode.anchor_batch() {
        anyhow::bail!(
            "v0.17.1 hard-negative mining produced only {} groups, fewer than anchor batch {}",
            hard_groups.len(),
            mode.anchor_batch()
        );
    }
    if mode == RunMode::Full && hard_groups.len() < MIN_FULL_HARD_NEGATIVE_GROUPS {
        anyhow::bail!(
            "v0.17.1 full pretraining requires at least {MIN_FULL_HARD_NEGATIVE_GROUPS} TRAIN hard-negative groups; observed {}. Inspect TRAIN-only mining statistics and revise the representation-pretraining data construction before any scientific validation run",
            hard_groups.len()
        );
    }

    if hard_groups
        .iter()
        .any(|group| group.negative_pool.len() != HARD_NEGATIVES_PER_ANCHOR)
    {
        anyhow::bail!(
            "v0.17.1 requires every retained TRAIN group to contain exactly {HARD_NEGATIVES_PER_ANCHOR} hard negatives"
        );
    }

    log_host_memory("after_hard_negative_mining");
    let retention = prune_unused_spectrum_payloads(
        &mut corpus.records,
        &hard_groups,
        &validation_groups,
        mode,
    )?;
    log_host_memory("after_spectrum_payload_prune");

    println!("compatibility_pretraining_version\t{VERSION}");
    println!("run_mode\t{}", mode.as_str());
    println!(
        "compute_device\t{}",
        if device.is_cuda() { "cuda:0" } else { "cpu" }
    );
    println!(
        "objective\tpositive_first_128way_listwise_precursor_compatible_hard_negative_pretraining"
    );
    println!("architecture\ttrainable_chemistry_peptide_encoder_plus_trainable_spectrum_encoder_plus_residue_and_cleavage_cross_modal_interaction");
    println!("representation_level_change\twhole_peptidoform_bidirectional_chemistry_representation_trained_end_to_end_for_spectrum_compatibility");
    println!("proposal_policy\tv01323_final_two_view_fixed_budget_frozen");
    println!("candidate_generation\tFROZEN_NO_REGENERATION");
    println!("validation_scoring\tstandalone_compatibility_score_no_legacy_blend");
    println!("validation_selection_policy\tnone_fixed_final_evaluation_only");
    println!("test_partition_consumed\tNO");
    println!("unified_checkpoint\t{}", unified_checkpoint.display());
    println!("model_dim\t{}", model.model_dim());
    println!("residue_interaction_layers\t{RESIDUE_INTERACTION_LAYERS}");
    println!("cleavage_interaction_layers\t{CLEAVAGE_INTERACTION_LAYERS}");
    println!("hard_negatives_per_anchor\t{HARD_NEGATIVES_PER_ANCHOR}");
    println!("hard_negative_pool\t{HARD_NEGATIVE_POOL}");
    println!("training_list_width\t{}", HARD_NEGATIVES_PER_ANCHOR + 1);
    println!("negative_sampling_policy\tall_127_nearest_mass_candidates_no_subsampling");
    println!("minimum_full_hard_negative_groups\t{MIN_FULL_HARD_NEGATIVE_GROUPS}");
    println!("hard_negative_mass_window_da\t{HARD_NEGATIVE_MASS_WINDOW_DA:.4}");
    println!("train_steps\t{}", mode.train_steps());
    println!("anchor_batch\t{}", mode.anchor_batch());
    println!(
        "candidate_pairs_per_optimizer_step\t{}",
        mode.anchor_batch() * (HARD_NEGATIVES_PER_ANCHOR + 1)
    );
    println!("learning_rate\t{LEARNING_RATE}");
    println!("weight_decay\t{WEIGHT_DECAY}");
    println!("max_gradient_norm\t{MAX_GRADIENT_NORM}");
    println!("seed\t{SEED}");
    println!("train_partition_records\t{}", train_indices.len());
    println!("train_spectrum_anchors\t{}", mining_stats.spectrum_anchors);
    println!(
        "train_mass_charge_candidates\t{}",
        mining_stats.mass_charge_candidates
    );
    println!("train_hard_negative_groups\t{}", hard_groups.len());
    println!("train_group_width_contract\tEXACT_128_WAY");
    println!(
        "hard_negative_pool_mean\t{:.3}",
        mining_stats.mean_pool_size
    );
    println!("hard_negative_pool_min\t{}", mining_stats.min_pool_size);
    println!("hard_negative_pool_max\t{}", mining_stats.max_pool_size);
    println!(
        "hard_negative_abs_mass_error_mean_da\t{:.6}",
        mining_stats.mean_absolute_mass_error_da
    );
    println!(
        "hard_negative_abs_mass_error_max_da\t{:.6}",
        mining_stats.max_absolute_mass_error_da
    );
    println!(
        "retained_training_spectrum_records\t{}",
        retention.training_records
    );
    println!(
        "retained_validation_spectrum_records\t{}",
        retention.validation_records
    );
    println!(
        "pruned_spectrum_payload_records\t{}",
        retention.pruned_records
    );
    println!(
        "warm_start_peptide_encoder_variables\t{}",
        warm_start.peptide_encoder_loaded_variables
    );
    println!(
        "warm_start_spectrum_encoder_variables\t{}",
        warm_start.spectrum_encoder_loaded_variables
    );
    println!(
        "fresh_compatibility_variables\t{}",
        warm_start.fresh_compatibility_variables
    );
    println!("initialization_fingerprint\t{initialization_fingerprint}");
    println!(
        "validation_candidate_contract\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}\tliteral_in_window={}\til_in_window={}",
        validation_contract.records,
        validation_contract.oracle_exact,
        validation_contract.oracle_il,
        validation_contract.legacy_top1_exact,
        validation_contract.legacy_top1_il,
        validation_contract.exact_in_window,
        validation_contract.il_in_window,
    );

    let mut sampler = AnchorSampler::new(hard_groups.len(), SEED);
    let mut probe_sampler = sampler.clone();
    let probe_group_indices = probe_sampler.next_batch(mode.anchor_batch());
    let probe_batch = build_training_batch(
        &probe_group_indices,
        &hard_groups,
        &corpus.records,
        0,
        &featurizer,
        &spectrum_collator,
        &device,
    )?;
    let probe_output = model.forward_grouped_t(
        &probe_batch.peptide_batch,
        &probe_batch.spectrum_batch,
        &probe_batch.precursor_batch,
        HARD_NEGATIVES_PER_ANCHOR + 1,
        true,
    )?;
    let probe_loss = foundation_compatibility_listwise_loss(&probe_output.scores)?;
    let probe_gradients = probe_loss.backward()?;
    println!(
        "end_to_end_gradient_probe\tloss={:.6}\tpeptide_encoder={:.8}\tspectrum_encoder={:.8}\tcontext={:.8}\tspectrum_pool={:.8}\tresidue_interaction={:.8}\tcleavage_interaction={:.8}\toutput={:.8}",
        f64::from(probe_loss.to_scalar::<f32>()?),
        gradient_norm_for_prefix(&varmap, &probe_gradients, "compatibility.peptide_encoder.")?,
        gradient_norm_for_prefix(&varmap, &probe_gradients, "compatibility.spectrum_encoder.")?,
        gradient_norm_for_prefix(&varmap, &probe_gradients, "compatibility.context.")?,
        gradient_norm_for_prefix(&varmap, &probe_gradients, "compatibility.spectrum_pool.")?,
        gradient_norm_for_prefix(&varmap, &probe_gradients, "compatibility.residue_interaction.")?,
        gradient_norm_for_prefix(&varmap, &probe_gradients, "compatibility.cleavage.")?,
        gradient_norm_for_prefix(&varmap, &probe_gradients, "compatibility.output.")?,
    );
    require_nonzero_end_to_end_gradients(&varmap, &probe_gradients)?;
    drop(probe_gradients);
    drop(probe_loss);
    drop(probe_output);
    drop(probe_batch);
    log_host_memory("after_gradient_probe");

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            weight_decay: WEIGHT_DECAY,
        },
    )?;

    let mut unique_anchors = HashSet::<usize>::new();
    let mut window_loss = 0.0f64;
    let mut window_accuracy = 0.0f64;
    let mut window_gradient_norm = 0.0f64;
    let mut window_gradient_scale = 0.0f64;
    let mut window_negative_mass_error = 0.0f64;
    let mut window_negative_mass_error_max = 0.0f64;
    let mut window_steps = 0usize;

    for step0 in 0..mode.train_steps() {
        let step = step0 + 1;
        let group_indices = sampler.next_batch(mode.anchor_batch());
        let batch = build_training_batch(
            &group_indices,
            &hard_groups,
            &corpus.records,
            step0,
            &featurizer,
            &spectrum_collator,
            &device,
        )?;
        for &record_index in &batch.anchor_record_indices {
            unique_anchors.insert(record_index);
        }

        let output = model.forward_grouped_t(
            &batch.peptide_batch,
            &batch.spectrum_batch,
            &batch.precursor_batch,
            HARD_NEGATIVES_PER_ANCHOR + 1,
            true,
        )?;
        let loss = foundation_compatibility_listwise_loss(&output.scores)?;
        let loss_value = f64::from(loss.to_scalar::<f32>()?);
        let accuracy = positive_first_accuracy(&output.scores)?;
        let optimizer_step = optimizer.backward_step(&loss, Some(MAX_GRADIENT_NORM))?;

        window_loss += loss_value;
        window_accuracy += accuracy;
        window_gradient_norm += optimizer_step.gradient_norm;
        window_gradient_scale += optimizer_step.gradient_scale;
        window_negative_mass_error += batch.mean_selected_negative_mass_error_da;
        window_negative_mass_error_max =
            window_negative_mass_error_max.max(batch.max_selected_negative_mass_error_da);
        window_steps += 1;

        if step % LOG_EVERY_STEPS == 0 || step == mode.train_steps() {
            println!(
                "compatibility_training\tstep={}\tsteps_total={}\tmean_listwise_loss={:.8}\tmean_train_top1_fraction={:.6}\tmean_gradient_norm={:.6}\tmean_gradient_scale={:.6}\tmean_selected_negative_abs_mass_error_da={:.6}\tmax_selected_negative_abs_mass_error_da={:.6}\tunique_anchor_groups_seen={}",
                step,
                mode.train_steps(),
                window_loss / window_steps as f64,
                window_accuracy / window_steps as f64,
                window_gradient_norm / window_steps as f64,
                window_gradient_scale / window_steps as f64,
                window_negative_mass_error / window_steps as f64,
                window_negative_mass_error_max,
                unique_anchors.len(),
            );
            window_loss = 0.0;
            window_accuracy = 0.0;
            window_gradient_norm = 0.0;
            window_gradient_scale = 0.0;
            window_negative_mass_error = 0.0;
            window_negative_mass_error_max = 0.0;
            window_steps = 0;
        }
    }

    fs::create_dir_all(&output_dir)?;
    let model_path = output_dir.join("compatibility_model.safetensors");
    let optimizer_path = output_dir.join("optimizer.safetensors");
    varmap.save(&model_path)?;
    optimizer.save_safetensors(&optimizer_path)?;

    let diagnostics_path = output_dir.join("validation_ranking_diagnostics.tsv");
    let metrics = evaluate_validation(
        &validation_groups,
        &corpus.records,
        &model,
        &featurizer,
        &spectrum_collator,
        &device,
        mode.validation_limit(),
        &diagnostics_path,
    )?;

    let gate = mode == RunMode::Full
        && metrics.records == validation_contract.records
        && metrics.oracle_exact >= REQUIRED_LITERAL_ORACLE
        && metrics.oracle_il >= REQUIRED_IL_ORACLE
        && metrics.compatibility_top1_exact >= REQUIRED_LITERAL_TOP1
        && metrics.compatibility_top1_il >= REQUIRED_IL_TOP1;

    println!(
        "validation_summary\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}\tcompatibility_top1_literal={}\tcompatibility_top1_il={}\tliteral_in_window={}\til_in_window={}",
        metrics.records,
        metrics.oracle_exact,
        metrics.oracle_il,
        metrics.legacy_top1_exact,
        metrics.legacy_top1_il,
        metrics.compatibility_top1_exact,
        metrics.compatibility_top1_il,
        metrics.exact_in_window,
        metrics.il_in_window,
    );
    if mode == RunMode::Full {
        println!(
            "v0171_representation_gate\trequired_literal={}\trequired_il={}\trequired_oracle_literal={}\trequired_oracle_il={}\tobserved_literal={}\tobserved_il={}\tobserved_oracle_literal={}\tobserved_oracle_il={}\tgate={}",
            REQUIRED_LITERAL_TOP1,
            REQUIRED_IL_TOP1,
            REQUIRED_LITERAL_ORACLE,
            REQUIRED_IL_ORACLE,
            metrics.compatibility_top1_exact,
            metrics.compatibility_top1_il,
            metrics.oracle_exact,
            metrics.oracle_il,
            if gate { "PASS" } else { "FAIL" },
        );
        println!(
            "v0171_stop_rule\t{}",
            if gate {
                "ACCEPT_PROPOSAL_SCALE_COMPATIBILITY_REPRESENTATION_AND_FREEZE_V0171"
            } else {
                "CLOSE_MASS_ONLY_COMPATIBILITY_PRETRAINING"
            }
        );
    } else {
        println!("v0171_representation_gate\tNOT_EVALUATED_SMOKE_RUNTIME_ONLY");
        println!("v0171_stop_rule\tSMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION");
    }

    let summary_path = output_dir.join("validation_summary.tsv");
    let mut summary = BufWriter::new(File::create(&summary_path)?);
    writeln!(
        summary,
        "run_mode\trecords\toracle_literal\toracle_il\tlegacy_top1_literal\tlegacy_top1_il\tcompatibility_top1_literal\tcompatibility_top1_il\tliteral_in_window\til_in_window\tgate"
    )?;
    writeln!(
        summary,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        mode.as_str(),
        metrics.records,
        metrics.oracle_exact,
        metrics.oracle_il,
        metrics.legacy_top1_exact,
        metrics.legacy_top1_il,
        metrics.compatibility_top1_exact,
        metrics.compatibility_top1_il,
        metrics.exact_in_window,
        metrics.il_in_window,
        if mode == RunMode::Smoke {
            "NOT_EVALUATED_SMOKE"
        } else if gate {
            "PASS"
        } else {
            "FAIL"
        },
    )?;
    summary.flush()?;

    let metadata_out = CompatibilityMetadata {
        version: VERSION.into(),
        run_mode: mode.as_str().into(),
        objective: "positive_first_128way_listwise_precursor_compatible_hard_negative_pretraining".into(),
        architecture: "trainable_chemistry_peptide_encoder+trainable_spectrum_encoder+residue_cross_modal_2+cleavage_cross_modal_1".into(),
        representation_level_change: "whole_peptidoform_bidirectional_chemistry_representation_trained_end_to_end_for_spectrum_compatibility".into(),
        proposal_policy: "v01323_final_two_view_fixed_budget_frozen".into(),
        candidate_generation: "FROZEN_NO_REGENERATION".into(),
        validation_scoring: "standalone_compatibility_score_no_legacy_blend".into(),
        validation_selection_policy: "none_fixed_final_evaluation_only".into(),
        test_partition_consumed: false,
        unified_checkpoint: unified_checkpoint.display().to_string(),
        training_yaml: training_yaml.display().to_string(),
        validation_candidate_tsv: validation_path.display().to_string(),
        seed: SEED,
        model_dim: model.model_dim(),
        residue_interaction_layers: RESIDUE_INTERACTION_LAYERS,
        cleavage_interaction_layers: CLEAVAGE_INTERACTION_LAYERS,
        hard_negatives_per_anchor: HARD_NEGATIVES_PER_ANCHOR,
        hard_negative_pool: HARD_NEGATIVE_POOL,
        training_list_width: HARD_NEGATIVES_PER_ANCHOR + 1,
        candidate_pairs_per_optimizer_step: mode.anchor_batch() * (HARD_NEGATIVES_PER_ANCHOR + 1),
        negative_sampling_policy: "all_127_nearest_mass_candidates_no_subsampling".into(),
        hard_negative_mass_window_da: HARD_NEGATIVE_MASS_WINDOW_DA,
        train_steps: mode.train_steps(),
        anchor_batch: mode.anchor_batch(),
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        max_gradient_norm: MAX_GRADIENT_NORM,
        validation_interaction_window: VALIDATION_INTERACTION_WINDOW,
        train_partition_records: train_indices.len(),
        train_spectrum_anchors: mining_stats.spectrum_anchors,
        train_hard_negative_groups: hard_groups.len(),
        unique_anchor_groups_seen: unique_anchors.len(),
        peptide_encoder_warm_started_variables: warm_start.peptide_encoder_loaded_variables,
        spectrum_encoder_warm_started_variables: warm_start.spectrum_encoder_loaded_variables,
        fresh_compatibility_variables: warm_start.fresh_compatibility_variables,
        initialization_fingerprint,
        final_validation_records: metrics.records,
        final_literal_top1: metrics.compatibility_top1_exact,
        final_il_top1: metrics.compatibility_top1_il,
        validation_oracle_literal: metrics.oracle_exact,
        validation_oracle_il: metrics.oracle_il,
        gate: if mode == RunMode::Smoke {
            "NOT_EVALUATED_SMOKE".into()
        } else if gate {
            "PASS".into()
        } else {
            "FAIL".into()
        },
    };
    serde_yaml::to_writer(
        BufWriter::new(File::create(output_dir.join("metadata.yaml"))?),
        &metadata_out,
    )?;

    println!("checkpoint\t{}", model_path.display());
    println!("optimizer_checkpoint\t{}", optimizer_path.display());
    println!("validation_summary_tsv\t{}", summary_path.display());
    println!(
        "validation_ranking_diagnostics_tsv\t{}",
        diagnostics_path.display()
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct HardNegativeMiningStats {
    spectrum_anchors: usize,
    mass_charge_candidates: usize,
    mean_pool_size: f64,
    min_pool_size: usize,
    max_pool_size: usize,
    mean_absolute_mass_error_da: f64,
    max_absolute_mass_error_da: f64,
}

#[derive(Debug, Clone, Copy)]
struct MassCandidate {
    record_index: usize,
    mass: f64,
}

fn hard_negative_candidate_order(
    left: &HardNegativeCandidate,
    right: &HardNegativeCandidate,
) -> std::cmp::Ordering {
    left.absolute_mass_error_da
        .total_cmp(&right.absolute_mass_error_da)
        .then_with(|| left.record_index.cmp(&right.record_index))
}

fn retain_bounded_hard_negative_candidate(
    candidates: &mut Vec<HardNegativeCandidate>,
    candidate: HardNegativeCandidate,
) {
    let insert_at = candidates.partition_point(|existing| {
        hard_negative_candidate_order(existing, &candidate) != std::cmp::Ordering::Greater
    });

    if candidates.len() < HARD_NEGATIVE_POOL {
        candidates.insert(insert_at, candidate);
    } else if insert_at < HARD_NEGATIVE_POOL {
        // Drop the current worst candidate before inserting so the Vec never
        // grows beyond its fixed HARD_NEGATIVE_POOL allocation.
        candidates.pop();
        candidates.insert(insert_at, candidate);
    }
}

fn build_hard_negative_groups(
    records: &[FoundationTrainingRecord],
    train_indices: &[usize],
    max_sequence_len: usize,
) -> Result<(Vec<HardNegativeGroup>, HardNegativeMiningStats)> {
    let mut buckets = BTreeMap::<i32, Vec<MassCandidate>>::new();
    let mut il_keys = vec![None::<String>; records.len()];
    let mut spectrum_anchors = Vec::<usize>::new();

    for (position, &record_index) in train_indices.iter().enumerate() {
        if position > 0 && position % 50_000 == 0 {
            println!(
                "hard_negative_index_progress\trecords_done={}\trecords_total={}\tspectrum_anchors={}",
                position,
                train_indices.len(),
                spectrum_anchors.len(),
            );
            log_host_memory("hard_negative_index_progress");
        }
        let record = records
            .get(record_index)
            .with_context(|| format!("TRAIN record index {record_index} exceeds corpus"))?;
        if !peptide_supported(&record.peptidoform, max_sequence_len) {
            continue;
        }
        let Some(charge) = record.context.charge.filter(|value| *value > 0) else {
            continue;
        };
        let mass = match foundation_peptidoform_neutral_mass(&record.peptidoform) {
            Ok(value) if value.is_finite() && value > 0.0 => value,
            _ => continue,
        };
        il_keys[record_index] = Some(il_sequence_key(&record.peptidoform.sequence));
        buckets
            .entry(charge)
            .or_default()
            .push(MassCandidate { record_index, mass });

        let Some(precursor_mz) = record
            .context
            .precursor_mz
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        if !training_record_has_usable_spectrum(record) {
            continue;
        }
        let observed_mass = match foundation_precursor_neutral_mass(f64::from(precursor_mz), charge)
        {
            Ok(value) => value,
            Err(_) => continue,
        };
        if (mass - observed_mass).abs() > HARD_NEGATIVE_MASS_WINDOW_DA {
            continue;
        }
        spectrum_anchors.push(record_index);
    }
    println!(
        "hard_negative_index_progress\trecords_done={}\trecords_total={}\tspectrum_anchors={}",
        train_indices.len(),
        train_indices.len(),
        spectrum_anchors.len(),
    );
    log_host_memory("hard_negative_index_complete");

    for bucket in buckets.values_mut() {
        bucket.sort_by(|left, right| {
            left.mass
                .total_cmp(&right.mass)
                .then_with(|| left.record_index.cmp(&right.record_index))
        });
    }

    let mass_charge_candidates = buckets.values().map(Vec::len).sum::<usize>();
    let mut groups = Vec::<HardNegativeGroup>::new();
    let mut pool_size_sum = 0usize;
    let mut pool_size_min = usize::MAX;
    let mut pool_size_max = 0usize;
    let mut mass_error_sum = 0.0f64;
    let mut mass_error_count = 0usize;
    let mut mass_error_max = 0.0f64;

    for (anchor_position, anchor_index) in spectrum_anchors.iter().copied().enumerate() {
        let done = anchor_position + 1;
        if done % 50_000 == 0 || done == spectrum_anchors.len() {
            println!(
                "hard_negative_group_progress\tanchors_done={}\tanchors_total={}\tgroups_built={}",
                done,
                spectrum_anchors.len(),
                groups.len(),
            );
            log_host_memory("hard_negative_group_progress");
        }
        let record = &records[anchor_index];
        let charge = record.context.charge.context("anchor charge disappeared")?;
        let precursor_mz = record
            .context
            .precursor_mz
            .context("anchor precursor m/z disappeared")?;
        let observed_mass = foundation_precursor_neutral_mass(f64::from(precursor_mz), charge)
            .map_err(anyhow::Error::msg)?;
        let anchor_il = il_keys
            .get(anchor_index)
            .and_then(Option::as_ref)
            .context("missing anchor I/L key")?;
        let bucket = buckets
            .get(&charge)
            .context("missing charge bucket for anchor")?;
        let lower = observed_mass - HARD_NEGATIVE_MASS_WINDOW_DA;
        let upper = observed_mass + HARD_NEGATIVE_MASS_WINDOW_DA;
        let start = bucket.partition_point(|candidate| candidate.mass < lower);
        let end = bucket.partition_point(|candidate| candidate.mass <= upper);

        // Keep only the scientifically used top-K candidates while scanning the
        // mass window.  The previous implementation collected the entire window
        // and then called Vec::truncate(HARD_NEGATIVE_POOL).  truncate() preserves
        // the Vec capacity, so millions of groups retained allocations sized for
        // hundreds or thousands of candidates even though only the fixed top-K
        // entries were ever used. Maintaining this sorted bounded buffer is exactly
        // equivalent to sorting the full eligible window by the same key and
        // taking the first HARD_NEGATIVE_POOL entries.
        let mut candidates = Vec::<HardNegativeCandidate>::with_capacity(HARD_NEGATIVE_POOL);
        for candidate in &bucket[start..end] {
            if candidate.record_index == anchor_index {
                continue;
            }
            let Some(candidate_il) = il_keys.get(candidate.record_index).and_then(Option::as_ref)
            else {
                continue;
            };
            if candidate_il == anchor_il {
                continue;
            }

            retain_bounded_hard_negative_candidate(
                &mut candidates,
                HardNegativeCandidate {
                    record_index: candidate.record_index,
                    absolute_mass_error_da: (candidate.mass - observed_mass).abs(),
                },
            );
        }
        if candidates.len() < HARD_NEGATIVES_PER_ANCHOR {
            continue;
        }

        pool_size_sum += candidates.len();
        pool_size_min = pool_size_min.min(candidates.len());
        pool_size_max = pool_size_max.max(candidates.len());
        for candidate in &candidates {
            mass_error_sum += candidate.absolute_mass_error_da;
            mass_error_count += 1;
            mass_error_max = mass_error_max.max(candidate.absolute_mass_error_da);
        }
        groups.push(HardNegativeGroup {
            anchor_index,
            negative_pool: candidates,
        });
    }

    if groups.is_empty() {
        anyhow::bail!(
            "v0.17.1 found no TRAIN spectrum anchors with at least {HARD_NEGATIVES_PER_ANCHOR} non-I/L hard negatives within +/-{HARD_NEGATIVE_MASS_WINDOW_DA:.3} Da"
        );
    }

    let group_count = groups.len();
    Ok((
        groups,
        HardNegativeMiningStats {
            spectrum_anchors: spectrum_anchors.len(),
            mass_charge_candidates,
            mean_pool_size: pool_size_sum as f64 / group_count as f64,
            min_pool_size: pool_size_min,
            max_pool_size: pool_size_max,
            mean_absolute_mass_error_da: mass_error_sum / mass_error_count.max(1) as f64,
            max_absolute_mass_error_da: mass_error_max,
        },
    ))
}

#[derive(Debug, Clone, Copy)]
struct SpectrumPayloadRetentionStats {
    training_records: usize,
    validation_records: usize,
    pruned_records: usize,
}

fn training_record_has_usable_spectrum(record: &FoundationTrainingRecord) -> bool {
    record.observed_spectrum_peaks.iter().any(|peak| {
        peak.mz.is_finite() && peak.mz > 0.0 && peak.intensity.is_finite() && peak.intensity > 0.0
    }) || record.fragments.iter().any(|fragment| {
        fragment
            .product_mz
            .is_some_and(|mz| mz.is_finite() && mz > 0.0)
            && fragment.intensity.is_finite()
            && fragment.intensity > 0.0
    })
}

fn prune_unused_spectrum_payloads(
    records: &mut [FoundationTrainingRecord],
    groups: &[HardNegativeGroup],
    validation_groups: &[CandidateGroup],
    mode: RunMode,
) -> Result<SpectrumPayloadRetentionStats> {
    let mut planned_sampler = AnchorSampler::new(groups.len(), SEED);
    let mut training_records = HashSet::<usize>::new();
    for _ in 0..mode.train_steps() {
        for group_index in planned_sampler.next_batch(mode.anchor_batch()) {
            let group = groups.get(group_index).with_context(|| {
                format!("planned hard-negative group {group_index} exceeds groups")
            })?;
            training_records.insert(group.anchor_index);
        }
    }

    let validation_limit = mode
        .validation_limit()
        .unwrap_or(validation_groups.len())
        .min(validation_groups.len());
    let validation_records = validation_groups
        .iter()
        .take(validation_limit)
        .map(|group| group.record_index)
        .collect::<HashSet<_>>();

    let mut keep = training_records.clone();
    keep.extend(validation_records.iter().copied());
    let mut pruned_records = 0usize;
    for (record_index, record) in records.iter_mut().enumerate() {
        if keep.contains(&record_index) {
            continue;
        }
        if !record.observed_spectrum_peaks.is_empty() || !record.fragments.is_empty() {
            record.observed_spectrum_peaks = Vec::new();
            record.fragments = Vec::new();
            pruned_records += 1;
        }
    }

    Ok(SpectrumPayloadRetentionStats {
        training_records: training_records.len(),
        validation_records: validation_records.len(),
        pruned_records,
    })
}

fn proc_status_value<'a>(status: &'a str, key: &str) -> Option<&'a str> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .map(str::trim)
}

fn log_host_memory(stage: &str) {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return;
    };
    println!(
        "host_memory\tstage={}\trss={}\tpeak={}\tdata={}",
        stage,
        proc_status_value(&status, "VmRSS:").unwrap_or("NA"),
        proc_status_value(&status, "VmHWM:").unwrap_or("NA"),
        proc_status_value(&status, "VmData:").unwrap_or("NA"),
    );
}

fn peptide_supported(peptide: &PeptidoformInput, max_sequence_len: usize) -> bool {
    let length = peptide.sequence.chars().count();
    length > 0
        && length <= max_sequence_len
        && peptide
            .sequence
            .chars()
            .all(|residue| "ACDEFGHIKLMNPQRSTVWY".contains(residue))
}

fn il_sequence_key(sequence: &str) -> String {
    sequence
        .chars()
        .map(|residue| match residue {
            'I' | 'L' => 'J',
            other => other,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct AnchorSampler {
    order: Vec<usize>,
    cursor: usize,
    cycle: u64,
    seed: u64,
}

impl AnchorSampler {
    fn new(groups: usize, seed: u64) -> Self {
        let mut order = (0..groups).collect::<Vec<_>>();
        deterministic_shuffle(&mut order, seed);
        Self {
            order,
            cursor: 0,
            cycle: 0,
            seed,
        }
    }

    fn next_batch(&mut self, batch: usize) -> Vec<usize> {
        let mut selected = Vec::with_capacity(batch);
        while selected.len() < batch {
            if self.cursor >= self.order.len() {
                self.cycle = self.cycle.wrapping_add(1);
                deterministic_shuffle(
                    &mut self.order,
                    self.seed ^ self.cycle.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                );
                self.cursor = 0;
            }
            selected.push(self.order[self.cursor]);
            self.cursor += 1;
        }
        selected
    }
}

#[allow(clippy::too_many_arguments)]
fn build_training_batch(
    group_indices: &[usize],
    groups: &[HardNegativeGroup],
    records: &[FoundationTrainingRecord],
    step: usize,
    featurizer: &PeptideGraphFeaturizer,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<TrainingBatch> {
    let group_width = HARD_NEGATIVES_PER_ANCHOR + 1;
    let mut peptides = Vec::<PeptidoformInput>::with_capacity(group_indices.len() * group_width);
    let mut spectra = Vec::<FoundationSpectrum>::with_capacity(group_indices.len());
    let mut anchor_refs = Vec::<&FoundationTrainingRecord>::with_capacity(group_indices.len());
    let mut anchor_record_indices = Vec::<usize>::with_capacity(group_indices.len());
    let mut selected_mass_error_sum = 0.0f64;
    let mut selected_mass_error_count = 0usize;
    let mut selected_mass_error_max = 0.0f64;

    for &group_index in group_indices {
        let group = groups
            .get(group_index)
            .with_context(|| format!("hard-negative group index {group_index} exceeds groups"))?;
        let anchor = records
            .get(group.anchor_index)
            .with_context(|| format!("anchor record {} exceeds corpus", group.anchor_index))?;
        peptides.push(anchor.peptidoform.clone());

        let mut negative_pool = group.negative_pool.clone();
        deterministic_shuffle(
            &mut negative_pool,
            SEED ^ (group.anchor_index as u64).rotate_left(17)
                ^ (step as u64 + 1).wrapping_mul(0xd6e8_feb8_6659_fd93),
        );
        for candidate in negative_pool.iter().take(HARD_NEGATIVES_PER_ANCHOR) {
            peptides.push(records[candidate.record_index].peptidoform.clone());
            selected_mass_error_sum += candidate.absolute_mass_error_da;
            selected_mass_error_count += 1;
            selected_mass_error_max = selected_mass_error_max.max(candidate.absolute_mass_error_da);
        }

        spectra.push(
            FoundationSpectrum::from_training_record(anchor).with_context(|| {
                format!("TRAIN anchor {} lost observed spectrum", group.anchor_index)
            })?,
        );
        anchor_refs.push(anchor);
        anchor_record_indices.push(group.anchor_index);
    }

    if peptides.len() != group_indices.len() * group_width {
        anyhow::bail!("v0.17.1 internal hard-negative group width mismatch");
    }
    let peptide_batch = featurizer.featurize(&peptides, device)?;
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let precursor_batch = precursor_context(&anchor_refs, device)?;

    Ok(TrainingBatch {
        peptide_batch,
        spectrum_batch,
        precursor_batch,
        anchor_record_indices,
        mean_selected_negative_mass_error_da: selected_mass_error_sum
            / selected_mass_error_count.max(1) as f64,
        max_selected_negative_mass_error_da: selected_mass_error_max,
    })
}

fn positive_first_accuracy(scores: &Tensor) -> Result<f64> {
    let rows = scores.to_vec2::<f32>()?;
    if rows.is_empty() {
        return Ok(0.0);
    }
    let mut correct = 0usize;
    for row in &rows {
        let top = row
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(index, _)| index)
            .unwrap_or(usize::MAX);
        correct += usize::from(top == 0);
    }
    Ok(correct as f64 / rows.len() as f64)
}

fn precursor_context(
    records: &[&FoundationTrainingRecord],
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let charge: Vec<f32> = records
        .iter()
        .map(|record| record.context.charge.unwrap_or(0) as f32)
        .collect();
    let charge_present: Vec<f32> = records
        .iter()
        .map(|record| {
            if record.context.charge.is_some() {
                1.0
            } else {
                0.0
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
                1.0
            } else {
                0.0
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
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let instrument_ids: Vec<u32> = records
        .iter()
        .map(|record| record.context.instrument_id.unwrap_or(0))
        .collect();
    let instrument_present: Vec<f32> = records
        .iter()
        .map(|record| {
            if record.context.instrument_id.is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let batch = records.len();
    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(charge, batch, device)?,
        charge_present: Tensor::from_vec(charge_present, batch, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, batch, device)?,
        precursor_mz_present: Tensor::from_vec(precursor_mz_present, batch, device)?,
        nce: Tensor::from_vec(nce, batch, device)?,
        nce_present: Tensor::from_vec(nce_present, batch, device)?,
        instrument_ids: Tensor::from_vec(instrument_ids, batch, device)?.to_dtype(DType::U32)?,
        instrument_present: Tensor::from_vec(instrument_present, batch, device)?,
    })
}

fn validate_candidate_contract(groups: &[CandidateGroup]) -> Result<ValidationContract> {
    let mut contract = ValidationContract {
        records: groups.len(),
        oracle_exact: 0,
        oracle_il: 0,
        legacy_top1_exact: 0,
        legacy_top1_il: 0,
        exact_in_window: 0,
        il_in_window: 0,
    };
    for group in groups {
        contract.oracle_exact += usize::from(group.rows.iter().any(|row| row.exact));
        contract.oracle_il += usize::from(group.rows.iter().any(|row| row.il_exact));
        if let Some(row) = group.rows.iter().min_by_key(|row| row.legacy_rank) {
            contract.legacy_top1_exact += usize::from(row.exact);
            contract.legacy_top1_il += usize::from(row.il_exact);
        }
        let window = fixed_validation_indices(group);
        contract.exact_in_window +=
            usize::from(window.iter().any(|&index| group.rows[index].exact));
        contract.il_in_window +=
            usize::from(window.iter().any(|&index| group.rows[index].il_exact));
    }

    if contract.oracle_exact != REQUIRED_LITERAL_ORACLE
        || contract.oracle_il != REQUIRED_IL_ORACLE
        || contract.exact_in_window != REQUIRED_LITERAL_ORACLE
        || contract.il_in_window != REQUIRED_IL_ORACLE
    {
        anyhow::bail!(
            "v0.17.1 frozen candidate oracle/window mismatch: oracle={}/{} window={}/{} expected={}/{}",
            contract.oracle_exact,
            contract.oracle_il,
            contract.exact_in_window,
            contract.il_in_window,
            REQUIRED_LITERAL_ORACLE,
            REQUIRED_IL_ORACLE
        );
    }
    if contract.legacy_top1_exact != REQUIRED_LEGACY_LITERAL_TOP1
        || contract.legacy_top1_il != REQUIRED_LEGACY_IL_TOP1
    {
        anyhow::bail!(
            "v0.17.1 frozen legacy baseline mismatch: observed={}/{} expected={}/{}",
            contract.legacy_top1_exact,
            contract.legacy_top1_il,
            REQUIRED_LEGACY_LITERAL_TOP1,
            REQUIRED_LEGACY_IL_TOP1
        );
    }
    Ok(contract)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_validation(
    groups: &[CandidateGroup],
    records: &[FoundationTrainingRecord],
    model: &FoundationSpectrumPeptideCompatibilityModel,
    featurizer: &PeptideGraphFeaturizer,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
    limit: Option<usize>,
    diagnostics_path: &Path,
) -> Result<ValidationMetrics> {
    let total = limit.unwrap_or(groups.len()).min(groups.len());
    let mut metrics = ValidationMetrics {
        records: total,
        oracle_exact: 0,
        oracle_il: 0,
        legacy_top1_exact: 0,
        legacy_top1_il: 0,
        compatibility_top1_exact: 0,
        compatibility_top1_il: 0,
        exact_in_window: 0,
        il_in_window: 0,
    };
    let mut diagnostics = BufWriter::new(File::create(diagnostics_path)?);
    writeln!(
        diagnostics,
        "record_index\toracle_literal\toracle_il\tlegacy_top1_literal\tlegacy_top1_il\tcompatibility_top1_literal\tcompatibility_top1_il\tbest_literal_rank\tbest_il_rank\tlegacy_top1_sequence\tlegacy_top1_score\tcompatibility_top1_sequence\tcompatibility_top1_score"
    )?;

    for (group_number, group) in groups.iter().take(total).enumerate() {
        let record = records.get(group.record_index).with_context(|| {
            format!(
                "VALIDATION record index {} exceeds corpus",
                group.record_index
            )
        })?;
        let oracle_exact = group.rows.iter().any(|row| row.exact);
        let oracle_il = group.rows.iter().any(|row| row.il_exact);
        metrics.oracle_exact += usize::from(oracle_exact);
        metrics.oracle_il += usize::from(oracle_il);

        let legacy_index = group
            .rows
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.legacy_rank)
            .map(|(index, _)| index)
            .context("validation candidate group is empty")?;
        metrics.legacy_top1_exact += usize::from(group.rows[legacy_index].exact);
        metrics.legacy_top1_il += usize::from(group.rows[legacy_index].il_exact);

        let window = fixed_validation_indices(group);
        metrics.exact_in_window += usize::from(window.iter().any(|&index| group.rows[index].exact));
        metrics.il_in_window += usize::from(window.iter().any(|&index| group.rows[index].il_exact));

        let spectrum = FoundationSpectrum::from_training_record(record).with_context(|| {
            format!(
                "VALIDATION record {} lacks observed spectrum",
                group.record_index
            )
        })?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum], device)?;
        let precursor_batch = precursor_context(&[record], device)?;
        let spectrum_context = model.encode_spectrum_t(&spectrum_batch, &precursor_batch, false)?;

        let mut scored = Vec::<(usize, f64)>::with_capacity(window.len());
        for chunk in window.chunks(VALIDATION_ENCODE_BATCH) {
            let peptides = chunk
                .iter()
                .map(|&index| exported_peptidoform(&group.rows[index]))
                .collect::<Result<Vec<_>>>()?;
            let peptide_batch = featurizer.featurize(&peptides, device)?;
            let output =
                model.score_candidates_t(&peptide_batch, &spectrum_context, chunk.len(), false)?;
            let values = output.scores.flatten_all()?.to_vec1::<f32>()?;
            for (local, &index) in chunk.iter().enumerate() {
                scored.push((index, f64::from(values[local])));
            }
        }
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| {
                    group.rows[left.0]
                        .legacy_rank
                        .cmp(&group.rows[right.0].legacy_rank)
                })
                .then_with(|| left.0.cmp(&right.0))
        });
        let (compatibility_index, compatibility_score) = scored
            .first()
            .copied()
            .context("validation interaction window is empty")?;
        metrics.compatibility_top1_exact += usize::from(group.rows[compatibility_index].exact);
        metrics.compatibility_top1_il += usize::from(group.rows[compatibility_index].il_exact);

        let best_exact_rank = scored
            .iter()
            .position(|(index, _)| group.rows[*index].exact)
            .map(|rank| rank + 1);
        let best_il_rank = scored
            .iter()
            .position(|(index, _)| group.rows[*index].il_exact)
            .map(|rank| rank + 1);
        writeln!(
            diagnostics,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{:.8}",
            group.record_index,
            yes_no(oracle_exact),
            yes_no(oracle_il),
            yes_no(group.rows[legacy_index].exact),
            yes_no(group.rows[legacy_index].il_exact),
            yes_no(group.rows[compatibility_index].exact),
            yes_no(group.rows[compatibility_index].il_exact),
            format_rank(best_exact_rank),
            format_rank(best_il_rank),
            group.rows[legacy_index].sequence,
            group.rows[legacy_index].legacy_score,
            group.rows[compatibility_index].sequence,
            compatibility_score,
        )?;

        let done = group_number + 1;
        if done == 1 || done % 25 == 0 || done == total {
            println!(
                "compatibility_validation\tgroups_done={}\tgroups_total={}\tliteral_top1={}\til_top1={}",
                done,
                total,
                metrics.compatibility_top1_exact,
                metrics.compatibility_top1_il,
            );
        }
    }
    diagnostics.flush()?;
    Ok(metrics)
}

fn fixed_validation_indices(group: &CandidateGroup) -> Vec<usize> {
    let mut ranked = (0..group.rows.len()).collect::<Vec<_>>();
    ranked.sort_by_key(|&index| (group.rows[index].legacy_rank, index));
    ranked.truncate(VALIDATION_INTERACTION_WINDOW.min(ranked.len()));
    ranked.sort_unstable();
    ranked
}

fn read_candidate_groups(path: &Path) -> Result<Vec<CandidateGroup>> {
    let file = BufReader::new(File::open(path)?);
    let mut lines = file.lines();
    let header = lines.next().context("candidate TSV is empty")??;
    let columns = header.split('\t').collect::<Vec<_>>();
    let mut index = HashMap::<&str, usize>::new();
    for (column_index, name) in columns.iter().enumerate() {
        index.insert(*name, column_index);
    }
    for required in [
        "record_index",
        "candidate_sequence",
        "candidate_modifications",
        "fragment_causal_score",
        "fragment_causal_mass_rank",
        "mass_valid",
        "peptidoform_exact",
        "il_sequence_exact",
    ] {
        if !index.contains_key(required) {
            anyhow::bail!("candidate TSV missing required column '{required}'");
        }
    }

    let mut groups = Vec::<CandidateGroup>::new();
    let mut current_id = None::<usize>;
    let mut current_rows = Vec::<CandidateRow>::new();
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let get = |name: &str| -> Result<&str> {
            let column_index = *index.get(name).context("internal missing TSV index")?;
            fields
                .get(column_index)
                .copied()
                .with_context(|| format!("candidate row missing column {name}"))
        };
        if !parse_bool(get("mass_valid")?) {
            continue;
        }
        let record_index: usize = get("record_index")?.parse()?;
        if current_id != Some(record_index) {
            if let Some(previous) = current_id.take() {
                groups.push(CandidateGroup {
                    record_index: previous,
                    rows: std::mem::take(&mut current_rows),
                });
            }
            current_id = Some(record_index);
        }
        current_rows.push(CandidateRow {
            sequence: get("candidate_sequence")?.to_string(),
            modifications: get("candidate_modifications")?.to_string(),
            exact: parse_bool(get("peptidoform_exact")?),
            il_exact: parse_bool(get("il_sequence_exact")?),
            legacy_score: parse_finite(get("fragment_causal_score")?, f64::NEG_INFINITY),
            legacy_rank: parse_usize(get("fragment_causal_mass_rank")?, usize::MAX),
        });
    }
    if let Some(record_index) = current_id {
        groups.push(CandidateGroup {
            record_index,
            rows: current_rows,
        });
    }
    Ok(groups
        .into_iter()
        .filter(|group| !group.rows.is_empty())
        .collect())
}

fn verify_partition(
    groups: &[CandidateGroup],
    benchmark: &FoundationBenchmarkManifest,
    expected: FoundationPartition,
    label: &str,
) -> Result<()> {
    let allowed = benchmark
        .partition_indices(expected)
        .into_iter()
        .collect::<HashSet<_>>();
    for group in groups {
        if !allowed.contains(&group.record_index) {
            anyhow::bail!(
                "v0.17.1 {label} candidate group {} is not assigned to expected benchmark partition",
                group.record_index
            );
        }
    }
    Ok(())
}

fn reject_test_path(path: &Path) -> Result<()> {
    let name = path.to_string_lossy().to_ascii_lowercase();
    let suspicious = name.contains("/test/")
        || name.contains("\\test\\")
        || name.contains("test_partition")
        || name.ends_with("/test.tsv")
        || name.ends_with("\\test.tsv");
    if suspicious {
        anyhow::bail!("v0.17.1 forbids TEST-partition inputs; suspicious path {path:?}");
    }
    Ok(())
}

fn exported_peptidoform(row: &CandidateRow) -> Result<PeptidoformInput> {
    if row.modifications.trim().is_empty() {
        return Ok(PeptidoformInput::unmodified(row.sequence.clone()));
    }
    let residues = row.sequence.chars().collect::<Vec<_>>();
    let mut nterm = Vec::<u32>::new();
    let mut residue_mods = HashMap::<usize, Vec<u32>>::new();
    for part in row
        .modifications
        .split(';')
        .filter(|part| !part.trim().is_empty())
    {
        let (identity, site) = part
            .split_once('@')
            .with_context(|| format!("invalid exported modification '{part}'"))?;
        let id: u32 = identity
            .strip_prefix("UniMod:")
            .with_context(|| format!("unsupported exported modification identity '{identity}'"))?
            .parse()?;
        if site == "NTerm" {
            nterm.push(id);
        } else if let Some(inner) = site
            .strip_prefix("Residue(")
            .and_then(|value| value.strip_suffix(')'))
        {
            let residue_index: usize = inner.parse()?;
            if residue_index >= residues.len() {
                anyhow::bail!(
                    "exported modification residue index {} exceeds sequence '{}'",
                    residue_index,
                    row.sequence
                );
            }
            residue_mods.entry(residue_index).or_default().push(id);
        } else {
            anyhow::bail!(
                "v0.17.1 cannot reconstruct unsupported exported modification site '{site}'"
            );
        }
    }
    nterm.sort_unstable();
    for values in residue_mods.values_mut() {
        values.sort_unstable();
    }
    let mut encoded = String::new();
    for id in nterm {
        encoded.push_str(&format!("[UniMod:{id}]"));
    }
    for (index, residue) in residues.into_iter().enumerate() {
        encoded.push(residue);
        if let Some(ids) = residue_mods.get(&index) {
            for id in ids {
                encoded.push_str(&format!("[UniMod:{id}]"));
            }
        }
    }
    parse_modified_peptide(&encoded)
        .with_context(|| format!("reconstruct exported candidate '{}'", row.sequence))
}

fn initialize_fresh_compatibility_variables(
    varmap: &VarMap,
    seed: u64,
    device: &Device,
) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("compatibility VarMap lock poisoned"))?;
    let mut names = data.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut rng = DeterministicRng::new(seed);

    for name in names {
        if name.starts_with("compatibility.peptide_encoder.")
            || name.starts_with("compatibility.spectrum_encoder.")
        {
            continue;
        }
        let variable = data
            .get(&name)
            .with_context(|| format!("missing compatibility variable {name}"))?;
        if !variable.dtype().is_float() {
            continue;
        }
        let dims = variable.as_tensor().dims().to_vec();
        let tensor = if name.ends_with(".bias") {
            Tensor::zeros(variable.shape(), DType::F32, device)?
        } else if dims.len() == 1 {
            Tensor::ones(dims[0], DType::F32, device)?
        } else if dims.len() == 2 {
            let out_dim = dims[0];
            let in_dim = dims[1];
            let stdev = (2.0f64 / in_dim.max(1) as f64).sqrt();
            let values = (0..out_dim * in_dim)
                .map(|_| (stdev * rng.standard_normal()) as f32)
                .collect::<Vec<_>>();
            Tensor::from_vec(values, (out_dim, in_dim), device)?
        } else {
            anyhow::bail!(
                "fresh compatibility variable '{name}' has unsupported deterministic-init shape {:?}",
                dims
            );
        };
        variable.set(&tensor)?;
    }
    Ok(())
}

fn compatibility_model_fingerprint(varmap: &VarMap) -> Result<String> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("compatibility VarMap lock poisoned"))?;
    let mut names = data.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut hash = FNV1A64_OFFSET;
    for name in names {
        let variable = data
            .get(&name)
            .context("fingerprint variable disappeared")?;
        fnv1a64_bytes(&mut hash, name.as_bytes());
        for &dim in variable.as_tensor().dims() {
            fnv1a64_bytes(&mut hash, &(dim as u64).to_le_bytes());
        }
        for value in variable.as_tensor().flatten_all()?.to_vec1::<f32>()? {
            fnv1a64_bytes(&mut hash, &value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn gradient_norm_for_prefix(
    varmap: &VarMap,
    gradients: &candle_core::backprop::GradStore,
    prefix: &str,
) -> Result<f64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("compatibility VarMap lock poisoned"))?;
    let mut squared = 0.0f64;
    for (name, variable) in data.iter() {
        if !name.starts_with(prefix) {
            continue;
        }
        if let Some(gradient) = gradients.get(variable) {
            squared += f64::from(gradient.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }
    Ok(squared.sqrt())
}

fn require_nonzero_end_to_end_gradients(
    varmap: &VarMap,
    gradients: &candle_core::backprop::GradStore,
) -> Result<()> {
    for (label, prefix) in [
        ("peptide_encoder", "compatibility.peptide_encoder."),
        ("spectrum_encoder", "compatibility.spectrum_encoder."),
        ("context", "compatibility.context."),
        ("spectrum_pool", "compatibility.spectrum_pool."),
        ("residue_interaction", "compatibility.residue_interaction."),
        ("cleavage_interaction", "compatibility.cleavage."),
        ("output", "compatibility.output."),
    ] {
        let norm = gradient_norm_for_prefix(varmap, gradients, prefix)?;
        if !(norm > 0.0 && norm.is_finite()) {
            anyhow::bail!(
                "v0.17.1 end-to-end gradient probe failed for {label}: gradient_norm={norm}"
            );
        }
    }
    Ok(())
}

fn fnv1a64_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    fn uniform_open01(&mut self) -> f64 {
        const INV_2_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;
        (((self.next_u64() >> 11) as f64) + 0.5) * INV_2_POW_53
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let u1 = self.uniform_open01();
        let u2 = self.uniform_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        let first = radius * angle.cos();
        self.spare_normal = Some(radius * angle.sin());
        first
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn deterministic_shuffle<T>(values: &mut [T], seed: u64) {
    let mut rng = DeterministicRng::new(seed);
    for index in (1..values.len()).rev() {
        let selected = (rng.next_u64() as usize) % (index + 1);
        values.swap(index, selected);
    }
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "y"
    )
}

fn parse_finite(value: &str, fallback: f64) -> f64 {
    value
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

fn parse_usize(value: &str, fallback: usize) -> usize {
    value.parse::<usize>().unwrap_or(fallback)
}

fn format_rank(rank: Option<usize>) -> String {
    rank.map(|value| value.to_string())
        .unwrap_or_else(|| "NA".to_string())
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn il_key_collapses_isoleucine_and_leucine() {
        assert_eq!(il_sequence_key("PEPTIDEIL"), il_sequence_key("PEPTIDELI"));
    }

    #[test]
    fn full_and_smoke_protocols_are_distinct() {
        assert_eq!(RunMode::Full.train_steps(), FULL_TRAIN_STEPS);
        assert_eq!(RunMode::Smoke.train_steps(), SMOKE_TRAIN_STEPS);
        assert!(RunMode::Smoke.validation_limit().is_some());
        assert!(RunMode::Full.validation_limit().is_none());
    }

    #[test]
    fn proposal_scale_protocol_is_exactly_128_way() {
        assert_eq!(HARD_NEGATIVES_PER_ANCHOR, 127);
        assert_eq!(HARD_NEGATIVE_POOL, 127);
        assert_eq!(HARD_NEGATIVES_PER_ANCHOR + 1, VALIDATION_INTERACTION_WINDOW);
        assert_eq!(
            RunMode::Full.anchor_batch() * (HARD_NEGATIVES_PER_ANCHOR + 1),
            1024
        );
    }

    #[test]
    fn deterministic_sampler_repeats() {
        let mut first = AnchorSampler::new(20, 123);
        let mut second = AnchorSampler::new(20, 123);
        assert_eq!(first.next_batch(12), second.next_batch(12));
        assert_eq!(first.next_batch(12), second.next_batch(12));
    }

    #[test]
    fn bounded_hard_negative_pool_matches_full_sort_and_never_grows_capacity() {
        let source = (0..257usize)
            .map(|record_index| HardNegativeCandidate {
                record_index,
                absolute_mass_error_da: (((record_index * 73) % 251) as f64) / 10_000.0,
            })
            .collect::<Vec<_>>();

        let mut expected = source.clone();
        expected.sort_by(hard_negative_candidate_order);
        expected.truncate(HARD_NEGATIVE_POOL);

        let mut bounded = Vec::with_capacity(HARD_NEGATIVE_POOL);
        for candidate in source {
            retain_bounded_hard_negative_candidate(&mut bounded, candidate);
            assert!(bounded.len() <= HARD_NEGATIVE_POOL);
            assert_eq!(bounded.capacity(), HARD_NEGATIVE_POOL);
        }

        assert_eq!(bounded.len(), expected.len());
        for (observed, expected) in bounded.iter().zip(expected.iter()) {
            assert_eq!(observed.record_index, expected.record_index);
            assert_eq!(
                observed.absolute_mass_error_da.to_bits(),
                expected.absolute_mass_error_da.to_bits()
            );
        }
    }

    #[test]
    fn modification_reconstruction_accepts_supported_export_syntax() -> Result<()> {
        let row = CandidateRow {
            sequence: "ACDMK".into(),
            modifications: "UniMod:1@NTerm;UniMod:4@Residue(1);UniMod:35@Residue(3)".into(),
            exact: false,
            il_exact: false,
            legacy_score: 0.0,
            legacy_rank: 1,
        };
        let peptide = exported_peptidoform(&row)?;
        assert_eq!(peptide.sequence, "ACDMK");
        Ok(())
    }
}
