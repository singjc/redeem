//! Train the spectrum-conditioned causal next-token reranker.
//!
//! This lane is warm-started from a trained diffusion checkpoint but optimizes
//! only the teacher-forced causal chain-rule objective. Reverse diffusion is not
//! changed by this executable.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_causal_next_token_loss, foundation_chemistry_diffusion_argmax_refine,
    foundation_chemistry_diffusion_final_mass_valid,
    foundation_chemistry_diffusion_partition_isolated,
    foundation_chemistry_diffusion_project_mass_valid,
    foundation_chemistry_diffusion_refinement_timesteps,
    foundation_chemistry_diffusion_row_neutral_mass, foundation_diffusion_dataset_fingerprint,
    foundation_diffusion_length_loss, foundation_diffusion_residue_ptm_valid,
    foundation_diffusion_token_mass_da, foundation_diffusion_token_residue,
    foundation_diffusion_x0_loss, foundation_direct_beam_search,
    foundation_direct_conditioning_loss, foundation_direct_prefix_competitive_loss,
    foundation_direct_shuffled_order, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, foundation_structured_edit_argmax,
    foundation_structured_edit_finalize, foundation_structured_edit_gate_loss,
    foundation_structured_edit_open_row, foundation_structured_edit_partition_isolated,
    foundation_structured_edit_set_targets, load_causal_from_diffusion_checkpoint,
    load_chemistry_decoder_from_unified_checkpoint, load_chemistry_diffusion_from_v0200_checkpoint,
    load_direct_decoder_from_unified_checkpoint, load_foundation_corpus,
    load_structured_editor_from_v0210_checkpoint, read_foundation_training_run_config,
    ChemistryDiffusionFeaturizer, ChemistrySuffixMassLattice, ChemistryTransitionFeaturizer,
    DirectDecoderBeamConfig, FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest,
    FoundationCausalCollator, FoundationDiffusionCollator, FoundationDiffusionConfig,
    FoundationDiffusionVocabulary, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumCollator, FoundationStructuredEditOutput, FoundationTrainingRecord,
    PeptideSpectrumCausalModel, PeptideSpectrumChemistryDecoder,
    PeptideSpectrumChemistryDiffusionModel, PeptideSpectrumStructuredEditor, PeptidoformInput,
    PrecursorContextBatch, FOUNDATION_CHEMISTRY_DECODER_ARCHITECTURE_V0200,
    FOUNDATION_CHEMISTRY_DECODER_OBJECTIVE_V0200,
    FOUNDATION_CHEMISTRY_DIFFUSION_ARCHITECTURE_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_INITIAL_BEAM_WIDTH_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_OBJECTIVE_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_STEPS_V0210,
    FOUNDATION_CHEMISTRY_FRAGMENT_ABS_TOLERANCE_DA_V0200, FOUNDATION_CHEMISTRY_FRAGMENT_PPM_V0200,
    FOUNDATION_CHEMISTRY_SUFFIX_BIN_DA_V0200, FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
    FOUNDATION_DIFFUSION_CARBAMIDOMETHYL, FOUNDATION_DIFFUSION_DEAMIDATED,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_NTERM_ACETYL,
    FOUNDATION_DIFFUSION_OXIDATION, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_RESIDUE_ACETYL,
    FOUNDATION_DIFFUSION_VOCAB_SIZE, FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
    FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190, FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0190,
    FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0191, FOUNDATION_DIRECT_PREFIX_MARGIN_V0191,
    FOUNDATION_PEPTIDE_WATER_MASS_DA, FOUNDATION_STRUCTURED_EDIT_ARCHITECTURE_V0220,
    FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220,
    FOUNDATION_STRUCTURED_EDIT_GATE_WEIGHT_V0220,
    FOUNDATION_STRUCTURED_EDIT_INITIAL_BEAM_WIDTH_V0220,
    FOUNDATION_STRUCTURED_EDIT_LENGTH_WEIGHT_V0220, FOUNDATION_STRUCTURED_EDIT_OBJECTIVE_V0220,
    FOUNDATION_STRUCTURED_EDIT_PASSES_V0220, FOUNDATION_STRUCTURED_EDIT_TRAIN_INITIALIZERS_V0220,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct DiffusionCheckpointMetadata {
    diffusion: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Serialize)]
struct CausalCheckpointMetadata {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    train_diffusion_fingerprint: String,
    validation_diffusion_fingerprint: String,
    usable_train_pairs: usize,
    usable_validation_pairs: usize,
    train_steps: usize,
    batch_size: usize,
    validation_batches: usize,
    seed: u64,
    learning_rate: f64,
    max_gradient_norm: f64,
    warm_start_diffusion_checkpoint: String,
    global_step: usize,
    best_validation_loss: f64,
    diffusion: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Copy, Default)]
struct CausalMetrics {
    loss: f64,
    perplexity: f64,
    token_accuracy: f64,
    eos_accuracy: f64,
    exact_sequence_rate: f64,
    active_tokens: usize,
    sequences: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 10 {
        anyhow::bail!(
            "usage: foundation_train_causal FOUNDATION_TRAINING.yaml OUTPUT_DIR DIFFUSION_CHECKPOINT [train_steps=2000] [batch_size=32] [validation_batches=32] [seed=20260902] [learning_rate=1e-4] [max_gradient_norm=1.0]"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let diffusion_checkpoint = PathBuf::from(&args[3]);
    let train_steps = parse_or(&args, 4, 2_000usize)?;
    let batch_size = parse_or(&args, 5, 32usize)?;
    let validation_batches = parse_or(&args, 6, 32usize)?;
    let seed = parse_or(&args, 7, 20_260_902u64)?;
    let learning_rate = parse_or(&args, 8, 1.0e-4f64)?;
    let max_gradient_norm = parse_or(&args, 9, 1.0f64)?;
    if train_steps == 0 || batch_size == 0 || validation_batches == 0 {
        anyhow::bail!("train_steps, batch_size and validation_batches must be positive");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("learning_rate must be finite and positive");
    }
    if !(max_gradient_norm > 0.0 && max_gradient_norm.is_finite()) {
        anyhow::bail!("max_gradient_norm must be finite and positive");
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let diffusion_metadata_path = if diffusion_checkpoint.is_dir() {
        diffusion_checkpoint.join("metadata.yaml")
    } else {
        diffusion_checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    };
    let diffusion_metadata: DiffusionCheckpointMetadata = serde_yaml::from_str(
        &fs::read_to_string(&diffusion_metadata_path)
            .with_context(|| format!("failed to read {diffusion_metadata_path:?}"))?,
    )?;
    let config = diffusion_metadata.diffusion;
    config.validate().map_err(anyhow::Error::msg)?;
    let diffusion_model_path = resolve_model_safetensors(&diffusion_checkpoint);

    let vocabulary = FoundationDiffusionVocabulary;
    let train_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &config,
        vocabulary,
    );
    let validation_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &config,
        vocabulary,
    );
    if train_indices.len() < batch_size || validation_indices.len() < batch_size {
        anyhow::bail!(
            "insufficient causal pairs: train={} validation={} batch={batch_size}",
            train_indices.len(),
            validation_indices.len()
        );
    }
    let train_fingerprint =
        foundation_diffusion_dataset_fingerprint(&corpus.records, &train_indices)?;
    let validation_fingerprint =
        foundation_diffusion_dataset_fingerprint(&corpus.records, &validation_indices)?;
    let validation_selection = deterministic_subset(
        &validation_indices,
        validation_batches.saturating_mul(batch_size),
        seed ^ 0xc451_a61b_5927_eb31,
    );

    fs::create_dir_all(&output_root)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumCausalModel::new(config.clone(), vb)?;
    let warm_start = load_causal_from_diffusion_checkpoint(&varmap, &diffusion_model_path, &device)
        .with_context(|| format!("failed to warm-start from {diffusion_model_path:?}"))?;
    println!(
        "causal_warm_start\tloaded_variables={}\tcausal_only_variables={}\tignored_diffusion_variables={}",
        warm_start.loaded_variables,
        warm_start.causal_only_variables,
        warm_start.ignored_checkpoint_variables
    );

    let causal_collator = FoundationCausalCollator::new(config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;
    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            ..FoundationAdamWConfig::default()
        },
    )?;

    println!("objective\tcausal_next_token_ce_v1");
    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!(
        "benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        benchmark.manifest_fingerprint()
    );
    println!("train_diffusion_fingerprint\tfnv1a64:{train_fingerprint:016x}");
    println!("validation_diffusion_fingerprint\tfnv1a64:{validation_fingerprint:016x}");
    println!("usable_train_pairs\t{}", train_indices.len());
    println!("usable_validation_pairs\t{}", validation_indices.len());
    println!("model_dim\t{}", config.model_dim);
    println!("decoder_layers\t{}", config.decoder_layers);
    println!("max_tokens\t{}", config.max_tokens);
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("validation_batches\t{validation_batches}");
    println!("seed\t{seed}");
    println!("learning_rate\t{learning_rate}");
    println!("max_gradient_norm\t{max_gradient_norm}");
    println!(
        "warm_start_diffusion_checkpoint\t{}",
        diffusion_model_path.display()
    );

    let initial_metrics = evaluate(
        &model,
        &corpus.records,
        &validation_selection,
        batch_size,
        &causal_collator,
        &spectrum_collator,
        &device,
    )?;
    println_metrics("initial_validation", 0, initial_metrics);

    let initial_metadata = checkpoint_metadata(
        &config,
        &corpus,
        &benchmark,
        train_fingerprint,
        validation_fingerprint,
        train_indices.len(),
        validation_indices.len(),
        train_steps,
        batch_size,
        validation_batches,
        seed,
        learning_rate,
        max_gradient_norm,
        &diffusion_model_path,
        0,
        initial_metrics.loss,
    );
    save_checkpoint(&output_root.join("initial"), &varmap, &initial_metadata)?;
    // The warm-start itself is a legitimate model candidate. Materialize it as
    // `best/` so the output contract remains valid even if short causal
    // fine-tuning temporarily worsens validation loss.
    save_checkpoint(&output_root.join("best"), &varmap, &initial_metadata)?;

    let mut best_validation_loss = initial_metrics.loss;
    let mut best_step = 0usize;
    let eval_every = train_steps.min(100).max(1);

    for step in 1..=train_steps {
        let selected = deterministic_batch(
            &train_indices,
            batch_size,
            seed ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let records: Vec<&FoundationTrainingRecord> = selected
            .iter()
            .map(|&index| &corpus.records[index])
            .collect();
        let packed = collate_records(&records, &causal_collator, &spectrum_collator, &device)?;
        let output = model.forward_t(
            &packed.causal.input,
            &packed.spectrum,
            &packed.precursor,
            true,
        )?;
        let loss = foundation_causal_next_token_loss(&output, &packed.causal)?;
        let loss_value = f64::from(loss.to_scalar::<f32>()?);
        let optimizer_step = optimizer.backward_step(&loss, Some(max_gradient_norm))?;
        if step == 1 || step % 10 == 0 || step == train_steps {
            println!(
                "train\tstep={step}\tloss={loss_value:.6}\tperplexity={:.4}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                loss_value.exp(),
                optimizer_step.gradient_norm,
                optimizer_step.gradient_scale
            );
        }

        if step % eval_every == 0 || step == train_steps {
            let metrics = evaluate(
                &model,
                &corpus.records,
                &validation_selection,
                batch_size,
                &causal_collator,
                &spectrum_collator,
                &device,
            )?;
            println_metrics("validation", step, metrics);
            let metadata = checkpoint_metadata(
                &config,
                &corpus,
                &benchmark,
                train_fingerprint,
                validation_fingerprint,
                train_indices.len(),
                validation_indices.len(),
                train_steps,
                batch_size,
                validation_batches,
                seed,
                learning_rate,
                max_gradient_norm,
                &diffusion_model_path,
                step,
                best_validation_loss.min(metrics.loss),
            );
            save_checkpoint(&output_root.join("latest"), &varmap, &metadata)?;
            if metrics.loss < best_validation_loss {
                best_validation_loss = metrics.loss;
                best_step = step;
                let best_metadata = checkpoint_metadata(
                    &config,
                    &corpus,
                    &benchmark,
                    train_fingerprint,
                    validation_fingerprint,
                    train_indices.len(),
                    validation_indices.len(),
                    train_steps,
                    batch_size,
                    validation_batches,
                    seed,
                    learning_rate,
                    max_gradient_norm,
                    &diffusion_model_path,
                    step,
                    best_validation_loss,
                );
                save_checkpoint(&output_root.join("best"), &varmap, &best_metadata)?;
            }
        }
    }

    println!("best_step\t{best_step}");
    println!("best_validation_causal_loss\t{best_validation_loss:.8}");
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

/// v0.19 entry point reused by `foundation_train_direct_decoder_v0190`.
///
/// It lives here so the established, audited corpus/partition/collation helpers
/// remain the single implementation used by both causal generations. Unlike
/// the historical executable, v0.19 trains on CUDA, adds the spectrum-use
/// objective, evaluates the frozen 125-record v0.13.23 validation cohort, and
/// finishes with direct mass-constrained beam decoding.
pub(crate) fn v0190_main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_train_direct_decoder_v0190 RUN.yaml MODEL_CHECKPOINT VALIDATION_CANDIDATES.tsv OUTPUT_DIR [full|smoke|audit|rescue]"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let parent = PathBuf::from(&args[2]);
    let validation_candidates = PathBuf::from(&args[3]);
    let output_root = PathBuf::from(&args[4]);
    let mode = args.get(5).map(String::as_str).unwrap_or("full");
    let (train_steps, batch_size, validation_limit, beam_width) = match mode {
        "full" => (4_000usize, 32usize, None, 128usize),
        "smoke" => (2usize, 2usize, Some(4usize), 8usize),
        "audit" => (0usize, 32usize, None, 128usize),
        // No-retraining search-recovery experiment. 4096 is the one fixed
        // breadth escalation after the v0.19 audit showed 123/125 valid target
        // paths but 61/125 records with no returned beam at width 128. The
        // A100-80GB run used only ~1.2 GiB at width 128, so this 32x breadth is
        // intentionally bounded and is not a width sweep.
        "rescue" => (0usize, 32usize, None, 4_096usize),
        other => anyhow::bail!(
            "unsupported v0.19 mode {other:?}; expected full, smoke, audit, or rescue"
        ),
    };
    reject_test_path_v0190(&training_yaml)?;
    reject_test_path_v0190(&parent)?;
    reject_test_path_v0190(&validation_candidates)?;
    reject_test_path_v0190(&output_root)?;

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let metadata_path = parent.join("metadata.yaml");
    let metadata: UnifiedV0190Metadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("read accepted unified metadata {metadata_path:?}"))?,
    )?;
    metadata
        .inverse_config
        .validate()
        .map_err(anyhow::Error::msg)?;
    if metadata.inverse_config.model_dim != 96 {
        anyhow::bail!(
            "v0.19 is anchored to the accepted 96-d unified checkpoint, found {}",
            metadata.inverse_config.model_dim
        );
    }
    let vocabulary = FoundationDiffusionVocabulary;
    let train_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &metadata.inverse_config,
        vocabulary,
    );
    let validation_cohort = read_frozen_validation_cohort_v0190(
        &validation_candidates,
        &corpus.records,
        &benchmark,
        &metadata.inverse_config,
        vocabulary,
    )?;
    let mut validation_indices = validation_cohort.indices.clone();
    if let Some(limit) = validation_limit {
        validation_indices.truncate(limit);
    }
    if (mode != "audit" && train_indices.len() < batch_size) || validation_indices.is_empty() {
        anyhow::bail!(
            "insufficient v0.19 pairs: train={} validation={} batch={batch_size}",
            train_indices.len(),
            validation_indices.len()
        );
    }

    fs::create_dir_all(&output_root)?;
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumCausalModel::new(metadata.inverse_config.clone(), vb)?;
    let parent_model = resolve_model_safetensors(&parent);
    let warm = load_direct_decoder_from_unified_checkpoint(&varmap, &parent_model, &device)?;
    let causal_collator = FoundationCausalCollator::new(metadata.inverse_config.clone())?;
    let spectrum_collator =
        FoundationSpectrumCollator::new(metadata.inverse_config.spectrum.clone())?;
    println!("version\tv0.19.0");
    println!("architecture\tmasked_self_attention+full_peak_cross_attention+ff");
    println!("objective\t{FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0190}");
    println!("proposal_generation_required\tfalse");
    println!("proposal_candidates_used_for_training\tfalse");
    println!("proposal_candidates_used_for_decoding\tfalse");
    println!(
        "validation_cohort_source\t{}",
        validation_candidates.display()
    );
    println!("validation_cohort_policy\tfrozen_v01323_mass_valid_record_ids_only");
    println!(
        "validation_cohort_contract\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}",
        validation_cohort.indices.len(),
        validation_cohort.oracle_literal,
        validation_cohort.oracle_il,
        validation_cohort.legacy_literal_top1,
        validation_cohort.legacy_il_top1
    );
    println!("test_partition_consumed\tfalse");
    println!("mode\t{mode}");
    println!("device\t{device:?}");
    println!("train_records\t{}", train_indices.len());
    println!("validation_records\t{}", validation_indices.len());
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("beam_width\t{beam_width}");
    println!("decoder_search_policy\tfull_prefix_distinct_standard_beam_no_mass_state_merge");
    if mode == "rescue" {
        println!(
            "decoder_search_rescue_policy\tdiagnostic_width128_then_fixed_width4096_no_retraining"
        );
    }
    println!(
        "spectrum_encoder_warm_started_variables\t{}",
        warm.spectrum_encoder_variables
    );
    println!("decoder_warm_started_variables\t{}", warm.decoder_variables);
    println!(
        "ignored_parent_variables\t{}",
        warm.ignored_parent_variables
    );
    println!("conditioning_margin_nats\t{FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190}");
    println!("conditioning_weight\t{FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190}");

    if mode == "audit" {
        let metrics = evaluate_conditioning_v0190(
            &model,
            &corpus.records,
            &validation_indices,
            batch_size,
            &causal_collator,
            &spectrum_collator,
            &device,
            20_260_919,
        )?;
        print_conditioning_v0190(
            "audit_validation",
            metadata.global_step.unwrap_or(0),
            metrics,
        );
        let audit = evaluate_direct_decoder_audit_v0190(
            &model,
            &corpus.records,
            &validation_indices,
            &causal_collator,
            &spectrum_collator,
            &metadata.inverse_config,
            beam_width,
            &device,
            &output_root,
        )?;
        print_direct_decoder_audit_v0190(&audit);
        println!("v0190_audit_stop_rule\tV0190_DECODER_AUDIT_COMPLETE_NO_RETRAINING");
        return Ok(());
    }

    if mode == "rescue" {
        let metrics = evaluate_conditioning_v0190(
            &model,
            &corpus.records,
            &validation_indices,
            batch_size,
            &causal_collator,
            &spectrum_collator,
            &device,
            20_260_919,
        )?;
        print_conditioning_v0190(
            "rescue_validation",
            metadata.global_step.unwrap_or(0),
            metrics,
        );

        // Width 128 is a diagnostic rerun under corrected prefix-distinct beam
        // semantics. It separates the invalid mass-bin state merge from the
        // one predeclared breadth rescue below; it is not used to select a
        // checkpoint or tune a width.
        let diagnostic = evaluate_direct_generation_v0190(
            &model,
            &corpus.records,
            &validation_indices,
            &causal_collator,
            &spectrum_collator,
            &metadata.inverse_config,
            128,
            &device,
        )?;
        print_generation_v0190("decoder_rescue_width128", &diagnostic);

        // One fixed breadth escalation only. If this remains below the frozen
        // baseline, do not continue widening the beam in circles; escalate the
        // sequence-training objective instead.
        let rescue = evaluate_direct_generation_v0190(
            &model,
            &corpus.records,
            &validation_indices,
            &causal_collator,
            &spectrum_collator,
            &metadata.inverse_config,
            beam_width,
            &device,
        )?;
        print_generation_v0190("decoder_rescue_width4096", &rescue);
        let conditioning_guard = metrics.conditioning_gap > 0.0;
        println!(
            "conditioning_guard\t{}",
            if conditioning_guard { "PASS" } else { "FAIL" }
        );
        let gate = if rescue.literal_top1 >= 28 && rescue.il_top1 >= 42 && conditioning_guard {
            "PROGRESS_MILESTONE"
        } else if rescue.literal_top1 >= 24 && rescue.il_top1 >= 38 && conditioning_guard {
            "BASELINE_RECOVERY"
        } else if !conditioning_guard {
            "CONDITIONING_GUARD_NOT_MET"
        } else {
            "BELOW_BASELINE_RECOVERY"
        };
        println!("v0190_search_rescue_gate\t{gate}");
        let stop = match gate {
            "PROGRESS_MILESTONE" => "V0190_SEARCH_RESCUE_PROGRESS_MILESTONE",
            "BASELINE_RECOVERY" => "V0190_SEARCH_RESCUE_BASELINE_RECOVERED",
            "CONDITIONING_GUARD_NOT_MET" => "V0190_SEARCH_RESCUE_CONDITIONING_GUARD_FAILED",
            _ => "V0190_SEARCH_RESCUE_BELOW_BASELINE_CLOSE_BREADTH_LANE",
        };
        println!("v0190_search_rescue_stop_rule\t{stop}");
        return Ok(());
    }

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: 1.0e-4,
            weight_decay: 1.0e-4,
            ..FoundationAdamWConfig::default()
        },
    )?;

    let initial = evaluate_conditioning_v0190(
        &model,
        &corpus.records,
        &validation_indices,
        batch_size,
        &causal_collator,
        &spectrum_collator,
        &device,
        20_260_919,
    )?;
    print_conditioning_v0190("initial_validation", 0, initial);
    save_v0190_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &metadata.inverse_config,
        mode,
        0,
        initial,
        &parent_model,
    )?;
    save_v0190_checkpoint(
        &output_root.join("best"),
        &varmap,
        &metadata.inverse_config,
        mode,
        0,
        initial,
        &parent_model,
    )?;
    let mut best_objective = initial.objective;
    let mut best_conditioning = initial;
    let mut best_step = 0usize;

    for step in 1..=train_steps {
        let selected = deterministic_batch(
            &train_indices,
            batch_size,
            20_260_919 ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let records: Vec<&FoundationTrainingRecord> =
            selected.iter().map(|&i| &corpus.records[i]).collect();
        let packed = collate_records(&records, &causal_collator, &spectrum_collator, &device)?;
        let matched_output = model.forward_t(
            &packed.causal.input,
            &packed.spectrum,
            &packed.precursor,
            true,
        )?;
        let matched = foundation_causal_next_token_loss(&matched_output, &packed.causal)?;
        let order = foundation_direct_shuffled_order(records.len(), 20_260_919 ^ step as u64)?;
        let shuffled_spectra: Vec<FoundationSpectrum> = order
            .iter()
            .map(|&i| FoundationSpectrum::from_training_record(records[i]).unwrap())
            .collect();
        let shuffled_batch = spectrum_collator.collate(&shuffled_spectra, &device)?;
        let shuffled_output = model.forward_t(
            &packed.causal.input,
            &shuffled_batch,
            &packed.precursor,
            true,
        )?;
        let shuffled = foundation_causal_next_token_loss(&shuffled_output, &packed.causal)?;
        let loss = foundation_direct_conditioning_loss(
            &matched,
            &shuffled,
            FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
            FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
        )?;
        let matched_value = f64::from(matched.to_scalar::<f32>()?);
        let shuffled_value = f64::from(shuffled.to_scalar::<f32>()?);
        let objective_value = f64::from(loss.to_scalar::<f32>()?);
        let update = optimizer.backward_step(&loss, Some(5.0))?;
        if step == 1 || step % 25 == 0 || step == train_steps {
            println!(
                "train\tstep={step}\tobjective={objective_value:.6}\tmatched_nll={matched_value:.6}\tshuffled_nll={shuffled_value:.6}\tconditioning_gap={:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                shuffled_value - matched_value, update.gradient_norm, update.gradient_scale
            );
        }
        if step % 100 == 0 || step == train_steps {
            let metrics = evaluate_conditioning_v0190(
                &model,
                &corpus.records,
                &validation_indices,
                batch_size,
                &causal_collator,
                &spectrum_collator,
                &device,
                20_260_919 ^ step as u64,
            )?;
            print_conditioning_v0190("validation", step, metrics);
            save_v0190_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &metadata.inverse_config,
                mode,
                step,
                metrics,
                &parent_model,
            )?;
            if metrics.objective < best_objective && metrics.conditioning_gap > 0.0 {
                best_objective = metrics.objective;
                best_conditioning = metrics;
                best_step = step;
                save_v0190_checkpoint(
                    &output_root.join("best"),
                    &varmap,
                    &metadata.inverse_config,
                    mode,
                    step,
                    metrics,
                    &parent_model,
                )?;
            }
        }
    }

    let best_path = output_root.join("best/model.safetensors");
    let best_tensors = candle_core::safetensors::load(&best_path, &device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.19 VarMap lock poisoned"))?;
    for (name, variable) in data.iter() {
        variable.set(
            best_tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("best checkpoint missing {name}"))?,
        )?;
    }
    drop(data);
    let generation = evaluate_direct_generation_v0190(
        &model,
        &corpus.records,
        &validation_indices,
        &causal_collator,
        &spectrum_collator,
        &metadata.inverse_config,
        beam_width,
        &device,
    )?;
    println!("final_literal_top1\t{}", generation.literal_top1);
    println!("final_sequence_top1\t{}", generation.sequence_top1);
    println!("final_il_top1\t{}", generation.il_top1);
    for &(k, literal, il) in &generation.topk {
        println!("final_top{k}_literal\t{literal}");
        println!("final_top{k}_il\t{il}");
    }
    println!("mass_valid_beams\t{}", generation.mass_valid_beams);
    println!("returned_beams\t{}", generation.returned_beams);
    println!(
        "mass_valid_beam_fraction\t{:.8}",
        generation.mass_valid_fraction()
    );
    print_conditioning_v0190("final_best_validation", best_step, best_conditioning);
    let conditioning_guard = best_conditioning.conditioning_gap > 0.0;
    println!(
        "conditioning_guard\t{}",
        if conditioning_guard { "PASS" } else { "FAIL" }
    );
    if mode == "smoke" {
        println!("v0190_representation_gate\tNOT_EVALUATED_SMOKE_RUNTIME_ONLY");
        println!("v0190_stop_rule\tSMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION");
        return Ok(());
    }
    let gate = if generation.literal_top1 >= 28 && generation.il_top1 >= 42 && conditioning_guard {
        "PROGRESS_MILESTONE"
    } else if generation.literal_top1 >= 24 && generation.il_top1 >= 38 && conditioning_guard {
        "BASELINE_RECOVERY"
    } else if !conditioning_guard {
        "CONDITIONING_GUARD_NOT_MET"
    } else {
        "BELOW_BASELINE_RECOVERY"
    };
    println!("scientific_gate\t{gate}");
    Ok(())
}

/// v0.19.1 prefix-competitive direct-decoder experiment.
///
/// The decoder architecture, accepted unified initialization, spectrum guard,
/// TRAIN population, frozen 125-record VALIDATION cohort, and width-128 final
/// search are unchanged from v0.19.0. The only scientific intervention is a
/// search-aligned hard next-token margin at each clean target prefix. The
/// strongest currently predicted legal wrong token is re-mined every step.
pub(crate) fn v0191_main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_train_direct_decoder_v0191 RUN.yaml MODEL_CHECKPOINT VALIDATION_CANDIDATES.tsv OUTPUT_DIR [full|smoke]"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let parent = PathBuf::from(&args[2]);
    let validation_candidates = PathBuf::from(&args[3]);
    let output_root = PathBuf::from(&args[4]);
    let mode = args.get(5).map(String::as_str).unwrap_or("full");
    let (train_steps, batch_size, validation_limit, beam_width) = match mode {
        "full" => (4_000usize, 32usize, None, 128usize),
        "smoke" => (2usize, 2usize, Some(4usize), 8usize),
        other => anyhow::bail!("unsupported v0.19.1 mode {other:?}; expected full or smoke"),
    };
    reject_test_path_v0190(&training_yaml)?;
    reject_test_path_v0190(&parent)?;
    reject_test_path_v0190(&validation_candidates)?;
    reject_test_path_v0190(&output_root)?;

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let metadata_path = parent.join("metadata.yaml");
    let metadata: UnifiedV0190Metadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("read accepted unified metadata {metadata_path:?}"))?,
    )?;
    metadata
        .inverse_config
        .validate()
        .map_err(anyhow::Error::msg)?;
    if metadata.inverse_config.model_dim != 96 {
        anyhow::bail!(
            "v0.19.1 is anchored to the accepted 96-d unified checkpoint, found {}",
            metadata.inverse_config.model_dim
        );
    }
    let vocabulary = FoundationDiffusionVocabulary;
    let train_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &metadata.inverse_config,
        vocabulary,
    );
    let validation_cohort = read_frozen_validation_cohort_v0190(
        &validation_candidates,
        &corpus.records,
        &benchmark,
        &metadata.inverse_config,
        vocabulary,
    )?;
    let mut validation_indices = validation_cohort.indices.clone();
    if let Some(limit) = validation_limit {
        validation_indices.truncate(limit);
    }
    if train_indices.len() < batch_size || validation_indices.is_empty() {
        anyhow::bail!(
            "insufficient v0.19.1 pairs: train={} validation={} batch={batch_size}",
            train_indices.len(),
            validation_indices.len()
        );
    }

    fs::create_dir_all(&output_root)?;
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumCausalModel::new(metadata.inverse_config.clone(), vb)?;
    let parent_model = resolve_model_safetensors(&parent);
    let warm = load_direct_decoder_from_unified_checkpoint(&varmap, &parent_model, &device)?;
    let causal_collator = FoundationCausalCollator::new(metadata.inverse_config.clone())?;
    let spectrum_collator =
        FoundationSpectrumCollator::new(metadata.inverse_config.spectrum.clone())?;

    println!("version\tv0.19.1");
    println!("architecture\tmasked_self_attention+full_peak_cross_attention+ff");
    println!("objective\t{FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0191}");
    println!("scientific_change\ton_policy_search_legal_hard_next_token_margin_only");
    println!("initialization\taccepted_unified_parent_only");
    println!("rejected_v0190_checkpoint_reused\tfalse");
    println!("proposal_candidates_used_for_training\tfalse");
    println!("proposal_candidates_used_for_decoding\tfalse");
    println!("search_breadth_lane\tCLOSED");
    println!("decoder_search_policy\tfull_prefix_distinct_standard_width128");
    println!(
        "validation_cohort_source\t{}",
        validation_candidates.display()
    );
    println!("validation_cohort_policy\tfrozen_v01323_mass_valid_record_ids_only");
    println!(
        "validation_cohort_contract\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}",
        validation_cohort.indices.len(),
        validation_cohort.oracle_literal,
        validation_cohort.oracle_il,
        validation_cohort.legacy_literal_top1,
        validation_cohort.legacy_il_top1
    );
    println!("test_partition_consumed\tfalse");
    println!("mode\t{mode}");
    println!("device\t{device:?}");
    println!("train_records\t{}", train_indices.len());
    println!("validation_records\t{}", validation_indices.len());
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("beam_width\t{beam_width}");
    println!("learning_rate\t0.0001");
    println!("weight_decay\t0.0001");
    println!("max_gradient_norm\t5");
    println!("training_seed\t20260919");
    println!("prefix_competition_margin_nats\t{FOUNDATION_DIRECT_PREFIX_MARGIN_V0191}");
    println!("prefix_competition_weight\t1.0");
    println!("conditioning_margin_nats\t{FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190}");
    println!("conditioning_weight\t{FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190}");
    println!(
        "spectrum_encoder_warm_started_variables\t{}",
        warm.spectrum_encoder_variables
    );
    println!("decoder_warm_started_variables\t{}", warm.decoder_variables);
    println!(
        "ignored_parent_variables\t{}",
        warm.ignored_parent_variables
    );

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: 1.0e-4,
            weight_decay: 1.0e-4,
            ..FoundationAdamWConfig::default()
        },
    )?;

    let initial = evaluate_prefix_competitive_v0191(
        &model,
        &corpus.records,
        &validation_indices,
        batch_size,
        &causal_collator,
        &spectrum_collator,
        &device,
        20_260_919,
    )?;
    print_prefix_competitive_v0191("initial_validation", 0, initial);
    save_v0191_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &metadata.inverse_config,
        mode,
        0,
        initial,
        &parent_model,
    )?;
    save_v0191_checkpoint(
        &output_root.join("best"),
        &varmap,
        &metadata.inverse_config,
        mode,
        0,
        initial,
        &parent_model,
    )?;
    let mut best_objective = initial.objective;
    let mut best_metrics = initial;
    let mut best_step = 0usize;

    for step in 1..=train_steps {
        let selected = deterministic_batch(
            &train_indices,
            batch_size,
            20_260_919 ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let records: Vec<&FoundationTrainingRecord> =
            selected.iter().map(|&i| &corpus.records[i]).collect();
        let packed = collate_records(&records, &causal_collator, &spectrum_collator, &device)?;
        let matched_output = model.forward_t(
            &packed.causal.input,
            &packed.spectrum,
            &packed.precursor,
            true,
        )?;
        let matched = foundation_causal_next_token_loss(&matched_output, &packed.causal)?;
        let prefix = foundation_direct_prefix_competitive_loss(
            &matched_output,
            &packed.causal,
            FOUNDATION_DIRECT_PREFIX_MARGIN_V0191,
        )?;
        let order = foundation_direct_shuffled_order(records.len(), 20_260_919 ^ step as u64)?;
        let shuffled_spectra: Vec<FoundationSpectrum> = order
            .iter()
            .map(|&i| FoundationSpectrum::from_training_record(records[i]).unwrap())
            .collect();
        let shuffled_batch = spectrum_collator.collate(&shuffled_spectra, &device)?;
        let shuffled_output = model.forward_t(
            &packed.causal.input,
            &shuffled_batch,
            &packed.precursor,
            true,
        )?;
        let shuffled = foundation_causal_next_token_loss(&shuffled_output, &packed.causal)?;
        let conditioning = foundation_direct_conditioning_loss(
            &matched,
            &shuffled,
            FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
            FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
        )?;
        let loss = (&conditioning + &prefix.loss)?;
        let matched_value = f64::from(matched.to_scalar::<f32>()?);
        let shuffled_value = f64::from(shuffled.to_scalar::<f32>()?);
        let prefix_value = f64::from(prefix.loss.to_scalar::<f32>()?);
        let objective_value = f64::from(loss.to_scalar::<f32>()?);
        let update = optimizer.backward_step(&loss, Some(5.0))?;
        if step == 1 || step % 25 == 0 || step == train_steps {
            println!(
                "train\tstep={step}\tobjective={objective_value:.6}\tmatched_nll={matched_value:.6}\tshuffled_nll={shuffled_value:.6}\tconditioning_gap={:.6}\tprefix_margin_loss={prefix_value:.6}\ttarget_token_mean_legal_rank={:.6}\ttarget_token_top1_fraction={:.8}\ttarget_token_top5_fraction={:.8}\ttarget_token_top10_fraction={:.8}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                shuffled_value - matched_value,
                prefix.mean_legal_rank,
                prefix.top1_fraction,
                prefix.top5_fraction,
                prefix.top10_fraction,
                update.gradient_norm,
                update.gradient_scale
            );
        }
        if step % 100 == 0 || step == train_steps {
            let metrics = evaluate_prefix_competitive_v0191(
                &model,
                &corpus.records,
                &validation_indices,
                batch_size,
                &causal_collator,
                &spectrum_collator,
                &device,
                20_260_919 ^ step as u64,
            )?;
            print_prefix_competitive_v0191("validation", step, metrics);
            save_v0191_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &metadata.inverse_config,
                mode,
                step,
                metrics,
                &parent_model,
            )?;
            if metrics.objective < best_objective
                && metrics.conditioning_gap >= FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190
            {
                best_objective = metrics.objective;
                best_metrics = metrics;
                best_step = step;
                save_v0191_checkpoint(
                    &output_root.join("best"),
                    &varmap,
                    &metadata.inverse_config,
                    mode,
                    step,
                    metrics,
                    &parent_model,
                )?;
            }
        }
    }

    let best_path = output_root.join("best/model.safetensors");
    let best_tensors = candle_core::safetensors::load(&best_path, &device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.19.1 VarMap lock poisoned"))?;
    for (name, variable) in data.iter() {
        variable.set(
            best_tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("v0.19.1 best checkpoint missing {name}"))?,
        )?;
    }
    drop(data);

    let generation = evaluate_direct_generation_v0190(
        &model,
        &corpus.records,
        &validation_indices,
        &causal_collator,
        &spectrum_collator,
        &metadata.inverse_config,
        beam_width,
        &device,
    )?;
    print_generation_v0190("final", &generation);
    print_prefix_competitive_v0191("final_best_validation", best_step, best_metrics);
    let conditioning_guard =
        best_metrics.conditioning_gap >= FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190;
    println!(
        "conditioning_guard\t{}",
        if conditioning_guard { "PASS" } else { "FAIL" }
    );
    if mode == "smoke" {
        println!("v0191_representation_gate\tNOT_EVALUATED_SMOKE_RUNTIME_ONLY");
        println!("v0191_stop_rule\tSMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION");
        return Ok(());
    }

    let gate = if generation.literal_top1 >= 28 && generation.il_top1 >= 42 && conditioning_guard {
        "PROGRESS_MILESTONE"
    } else if generation.literal_top1 >= 24 && generation.il_top1 >= 38 && conditioning_guard {
        "BASELINE_RECOVERY"
    } else if !conditioning_guard {
        "CONDITIONING_GUARD_NOT_MET"
    } else {
        "BELOW_BASELINE_RECOVERY"
    };
    println!("scientific_gate\t{gate}");
    let stop = match gate {
        "PROGRESS_MILESTONE" => "ACCEPT_PREFIX_COMPETITIVE_DIRECT_DECODER_PROGRESS_V0191",
        "BASELINE_RECOVERY" => "V0191_PREFIX_COMPETITIVE_BASELINE_RECOVERED",
        "CONDITIONING_GUARD_NOT_MET" => "V0191_PREFIX_COMPETITIVE_CONDITIONING_GUARD_FAILED",
        _ => "V0191_PREFIX_COMPETITIVE_GATE_NOT_MET_CLOSE_LOCAL_MARGIN_LANE",
    };
    println!("v0191_stop_rule\t{stop}");
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize)]
struct PrefixCompetitiveV0191 {
    matched_nll: f64,
    shuffled_nll: f64,
    conditioning_gap: f64,
    prefix_margin_loss: f64,
    target_token_mean_legal_rank: f64,
    target_token_top1_fraction: f64,
    target_token_top5_fraction: f64,
    target_token_top10_fraction: f64,
    objective: f64,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_prefix_competitive_v0191(
    model: &PeptideSpectrumCausalModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
    seed: u64,
) -> Result<PrefixCompetitiveV0191> {
    let mut matched_sum = 0.0;
    let mut shuffled_sum = 0.0;
    let mut prefix_loss_sum = 0.0;
    let mut rank_sum = 0.0;
    let mut top1_sum = 0.0;
    let mut top5_sum = 0.0;
    let mut top10_sum = 0.0;
    let mut batches = 0usize;
    for (chunk_index, chunk) in indices.chunks(batch_size).enumerate() {
        if chunk.len() < 2 {
            continue;
        }
        let selected: Vec<&FoundationTrainingRecord> = chunk.iter().map(|&i| &records[i]).collect();
        let packed = collate_records(&selected, causal_collator, spectrum_collator, device)?;
        let matched_output = model.forward_t(
            &packed.causal.input,
            &packed.spectrum,
            &packed.precursor,
            false,
        )?;
        let matched = foundation_causal_next_token_loss(&matched_output, &packed.causal)?;
        let prefix = foundation_direct_prefix_competitive_loss(
            &matched_output,
            &packed.causal,
            FOUNDATION_DIRECT_PREFIX_MARGIN_V0191,
        )?;
        let order = foundation_direct_shuffled_order(selected.len(), seed ^ chunk_index as u64)?;
        let spectra: Vec<FoundationSpectrum> = order
            .iter()
            .map(|&i| FoundationSpectrum::from_training_record(selected[i]).unwrap())
            .collect();
        let shuffled_spectrum = spectrum_collator.collate(&spectra, device)?;
        let shuffled = foundation_causal_next_token_loss(
            &model.forward_t(
                &packed.causal.input,
                &shuffled_spectrum,
                &packed.precursor,
                false,
            )?,
            &packed.causal,
        )?;
        matched_sum += f64::from(matched.to_scalar::<f32>()?);
        shuffled_sum += f64::from(shuffled.to_scalar::<f32>()?);
        prefix_loss_sum += f64::from(prefix.loss.to_scalar::<f32>()?);
        rank_sum += prefix.mean_legal_rank;
        top1_sum += prefix.top1_fraction;
        top5_sum += prefix.top5_fraction;
        top10_sum += prefix.top10_fraction;
        batches += 1;
    }
    if batches == 0 {
        anyhow::bail!("v0.19.1 validation produced no batches");
    }
    let n = batches as f64;
    let matched_nll = matched_sum / n;
    let shuffled_nll = shuffled_sum / n;
    let conditioning_gap = shuffled_nll - matched_nll;
    let prefix_margin_loss = prefix_loss_sum / n;
    let objective = matched_nll
        + prefix_margin_loss
        + FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190
            * (FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190 - conditioning_gap).max(0.0);
    Ok(PrefixCompetitiveV0191 {
        matched_nll,
        shuffled_nll,
        conditioning_gap,
        prefix_margin_loss,
        target_token_mean_legal_rank: rank_sum / n,
        target_token_top1_fraction: top1_sum / n,
        target_token_top5_fraction: top5_sum / n,
        target_token_top10_fraction: top10_sum / n,
        objective,
    })
}

fn print_prefix_competitive_v0191(label: &str, step: usize, metrics: PrefixCompetitiveV0191) {
    println!(
        "{label}\tstep={step}\tobjective={:.6}\tmatched_spectrum_token_nll={:.6}\tshuffled_spectrum_token_nll={:.6}\tconditioning_gap={:.6}\tprefix_margin_loss={:.6}\ttarget_token_mean_legal_rank={:.6}\ttarget_token_top1_fraction={:.8}\ttarget_token_top5_fraction={:.8}\ttarget_token_top10_fraction={:.8}",
        metrics.objective,
        metrics.matched_nll,
        metrics.shuffled_nll,
        metrics.conditioning_gap,
        metrics.prefix_margin_loss,
        metrics.target_token_mean_legal_rank,
        metrics.target_token_top1_fraction,
        metrics.target_token_top5_fraction,
        metrics.target_token_top10_fraction,
    );
}

#[derive(Debug, Serialize)]
struct DirectCheckpointV0191<'a> {
    version: &'a str,
    objective: &'a str,
    architecture: &'a str,
    run_mode: &'a str,
    test_partition_consumed: bool,
    global_step: usize,
    prefix_competition_margin_nats: f64,
    prefix_competition_weight: f64,
    validation: PrefixCompetitiveV0191,
    unified_parent: String,
    inverse_config: FoundationDiffusionConfig,
}

fn save_v0191_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    config: &FoundationDiffusionConfig,
    mode: &str,
    step: usize,
    validation: PrefixCompetitiveV0191,
    parent: &Path,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    let metadata = DirectCheckpointV0191 {
        version: "v0.19.1",
        objective: FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0191,
        architecture: "masked_self_attention+full_peak_cross_attention+ff",
        run_mode: mode,
        test_partition_consumed: false,
        global_step: step,
        prefix_competition_margin_nats: FOUNDATION_DIRECT_PREFIX_MARGIN_V0191,
        prefix_competition_weight: 1.0,
        validation,
        unified_parent: parent.display().to_string(),
        inverse_config: config.clone(),
    };
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(&metadata)?,
    )?;
    Ok(())
}

/// v0.20 chemistry-structured direct decoder.
///
/// This is the single predeclared autoregressive architecture escalation after
/// v0.19.0/0.19.1. It restarts from the accepted unified parent, retains the
/// full-spectrum causal decoder, removes the rejected local prefix-margin loss,
/// and adds explicit candidate-transition chemistry plus a conservative
/// chemistry-only suffix feasibility mask at generation time.
pub(crate) fn v0200_main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_train_chemistry_decoder_v0200 RUN.yaml MODEL_CHECKPOINT VALIDATION_CANDIDATES.tsv OUTPUT_DIR [full|smoke|decode]"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let parent = PathBuf::from(&args[2]);
    let validation_candidates = PathBuf::from(&args[3]);
    let output_root = PathBuf::from(&args[4]);
    let mode = args.get(5).map(String::as_str).unwrap_or("full");
    let decode_only = mode == "decode";
    let (train_steps, batch_size, validation_limit, beam_width) = match mode {
        "full" => (4_000usize, 32usize, None, 128usize),
        "smoke" => (2usize, 2usize, Some(4usize), 8usize),
        // Decode-only exists solely to finish the frozen v0.20 evaluation from
        // an already-trained checkpoint after an inference-only runtime fix.
        // It performs zero optimizer steps and evaluates all 125 frozen rows.
        "decode" => (0usize, 32usize, None, 128usize),
        other => anyhow::bail!("unsupported v0.20 mode {other:?}; expected full, smoke, or decode"),
    };
    reject_test_path_v0190(&training_yaml)?;
    reject_test_path_v0190(&parent)?;
    reject_test_path_v0190(&validation_candidates)?;
    reject_test_path_v0190(&output_root)?;

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    let metadata_path = parent.join("metadata.yaml");
    let metadata: UnifiedV0190Metadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("read v0.20 model metadata {metadata_path:?}"))?,
    )?;
    metadata
        .inverse_config
        .validate()
        .map_err(anyhow::Error::msg)?;
    if metadata.inverse_config.model_dim != 96 {
        anyhow::bail!(
            "v0.20 is anchored to the accepted 96-d unified checkpoint, found {}",
            metadata.inverse_config.model_dim
        );
    }

    let vocabulary = FoundationDiffusionVocabulary;
    let mut train_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &metadata.inverse_config,
        vocabulary,
    );
    // Explicit transition mass features require measured precursor m/z/charge.
    // Missing precursor context is not replaced by theoretical target mass.
    train_indices.retain(|&index| v0200_precursor_mass(&corpus.records[index]).is_ok());

    let validation_cohort = read_frozen_validation_cohort_v0190(
        &validation_candidates,
        &corpus.records,
        &benchmark,
        &metadata.inverse_config,
        vocabulary,
    )?;
    let mut validation_indices = validation_cohort.indices.clone();
    if let Some(limit) = validation_limit {
        validation_indices.truncate(limit);
    }
    if train_indices.len() < batch_size || validation_indices.is_empty() {
        anyhow::bail!(
            "insufficient v0.20 pairs with measured precursor context: train={} validation={} batch={batch_size}",
            train_indices.len(),
            validation_indices.len()
        );
    }

    let max_validation_mass = validation_cohort
        .indices
        .iter()
        .map(|&index| v0200_precursor_mass(&corpus.records[index]))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(0.0f64, f64::max);
    let suffix_lattice = ChemistrySuffixMassLattice::new(
        metadata.inverse_config.max_tokens,
        max_validation_mass + 100.0,
    )
    .map_err(anyhow::Error::msg)?;
    let chemistry_featurizer =
        ChemistryTransitionFeaturizer::new(&metadata.inverse_config, suffix_lattice)
            .map_err(anyhow::Error::msg)?;

    // The old v0.19 audit established 123/125 true targets as physically mass
    // feasible. The new hard suffix constraint may not destroy any of those
    // paths. This gate always audits the complete frozen cohort, including smoke.
    let (mass_feasible_targets, suffix_feasible_targets) = v0200_true_path_contract(
        &corpus.records,
        &validation_cohort.indices,
        &metadata.inverse_config,
        &chemistry_featurizer,
    )?;
    if mass_feasible_targets != 123 {
        anyhow::bail!(
            "v0.20 frozen mass-feasibility contract changed: expected 123/125, found {mass_feasible_targets}/{}",
            validation_cohort.indices.len()
        );
    }
    if suffix_feasible_targets < mass_feasible_targets {
        anyhow::bail!(
            "v0.20 suffix lattice deletes true paths: mass_feasible={mass_feasible_targets} suffix_feasible={suffix_feasible_targets}"
        );
    }

    fs::create_dir_all(&output_root)?;
    let device = Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumChemistryDecoder::new(metadata.inverse_config.clone(), vb)?;
    let parent_model = resolve_model_safetensors(&parent);
    let warm = if decode_only {
        varmap.load(&parent_model).with_context(|| {
            format!(
                "failed to load trained v0.20 decode checkpoint {}",
                parent_model.display()
            )
        })?;
        None
    } else {
        Some(load_chemistry_decoder_from_unified_checkpoint(
            &varmap,
            &parent_model,
            &device,
        )?)
    };
    let causal_collator = FoundationCausalCollator::new(metadata.inverse_config.clone())?;
    let spectrum_collator =
        FoundationSpectrumCollator::new(metadata.inverse_config.spectrum.clone())?;

    println!("version\tv0.20.0");
    println!("architecture\t{FOUNDATION_CHEMISTRY_DECODER_ARCHITECTURE_V0200}");
    println!("objective\t{FOUNDATION_CHEMISTRY_DECODER_OBJECTIVE_V0200}");
    println!("scientific_change\texplicit_residual_mass_residue_ptm_and_complementary_fragment_transition_scoring");
    println!(
        "initialization\t{}",
        if decode_only {
            "trained_v0200_checkpoint_decode_only"
        } else {
            "accepted_unified_parent_only"
        }
    );
    println!("rejected_v0190_checkpoint_reused\tfalse");
    println!("rejected_v0191_checkpoint_reused\tfalse");
    println!("decode_only_no_optimizer_steps\t{decode_only}");
    println!("prefix_margin_objective\tREMOVED");
    println!("compatibility_reranking_lane\tCLOSED");
    println!("search_breadth_lane\tCLOSED");
    println!("fragment_peaks_define_transition_existence\tfalse");
    println!("fragment_peaks_are_soft_transition_evidence\ttrue");
    println!("suffix_lattice_policy\tconservative_supported_residue_ptm_mass_reachability_inference_hard_mask");
    println!("decoder_search_policy\tfull_prefix_distinct_standard_width128_plus_suffix_mass_feasibility");
    println!("transition_feature_dim\t{FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200}");
    println!("fragment_match_ppm\t{FOUNDATION_CHEMISTRY_FRAGMENT_PPM_V0200}");
    println!("fragment_match_abs_floor_da\t{FOUNDATION_CHEMISTRY_FRAGMENT_ABS_TOLERANCE_DA_V0200}");
    println!("suffix_mass_bin_da\t{FOUNDATION_CHEMISTRY_SUFFIX_BIN_DA_V0200}");
    println!(
        "chemistry_true_path_contract\trecords={}\tmass_feasible={}\tsuffix_feasible={}\trequired_suffix_feasible=123",
        validation_cohort.indices.len(), mass_feasible_targets, suffix_feasible_targets
    );
    println!(
        "validation_cohort_source\t{}",
        validation_candidates.display()
    );
    println!("validation_cohort_policy\tfrozen_v01323_mass_valid_record_ids_only");
    println!(
        "validation_cohort_contract\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}",
        validation_cohort.indices.len(),
        validation_cohort.oracle_literal,
        validation_cohort.oracle_il,
        validation_cohort.legacy_literal_top1,
        validation_cohort.legacy_il_top1
    );
    println!("test_partition_consumed\tfalse");
    println!("mode\t{mode}");
    println!("device\t{device:?}");
    println!("train_records_with_precursor\t{}", train_indices.len());
    println!("validation_records\t{}", validation_indices.len());
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("beam_width\t{beam_width}");
    println!("learning_rate\t0.0001");
    println!("weight_decay\t0.0001");
    println!("max_gradient_norm\t5");
    println!("training_seed\t20260919");
    println!("conditioning_margin_nats\t{FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190}");
    println!("conditioning_weight\t{FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190}");
    println!(
        "chemistry_transition_head_zero_initialized\t{}",
        !decode_only
    );
    if let Some(warm) = warm.as_ref() {
        println!(
            "spectrum_encoder_warm_started_variables\t{}",
            warm.spectrum_encoder_variables
        );
        println!("decoder_warm_started_variables\t{}", warm.decoder_variables);
        println!(
            "chemistry_transition_new_variables\t{}",
            warm.chemistry_transition_variables
        );
        println!(
            "ignored_parent_variables\t{}",
            warm.ignored_parent_variables
        );
    } else {
        println!("decode_checkpoint_loaded\t{}", parent_model.display());
        println!(
            "decode_checkpoint_global_step\t{}",
            metadata.global_step.unwrap_or(0)
        );
    }

    if decode_only {
        let checkpoint_step = metadata.global_step.unwrap_or(0);
        let metrics = evaluate_chemistry_v0200(
            &model,
            &corpus.records,
            &validation_indices,
            batch_size,
            &causal_collator,
            &spectrum_collator,
            &chemistry_featurizer,
            &device,
            20_260_919 ^ checkpoint_step as u64,
        )?;
        print_chemistry_v0200("decode_validation", checkpoint_step, metrics);
        let generation = evaluate_chemistry_generation_v0200(
            &model,
            &corpus.records,
            &validation_indices,
            &causal_collator,
            &spectrum_collator,
            &chemistry_featurizer,
            &metadata.inverse_config,
            beam_width,
            &device,
        )?;
        print_generation_v0190("final", &generation);
        print_chemistry_v0200("final_best_validation", checkpoint_step, metrics);
        let conditioning_guard =
            metrics.conditioning_gap >= FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190;
        println!(
            "conditioning_guard\t{}",
            if conditioning_guard { "PASS" } else { "FAIL" }
        );
        print_v0200_scientific_gate(&generation, conditioning_guard);
        println!("v0200_decode_stop_rule\tV0200_FROZEN_DECODE_COMPLETE_NO_RETRAINING");
        return Ok(());
    }

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: 1.0e-4,
            weight_decay: 1.0e-4,
            ..FoundationAdamWConfig::default()
        },
    )?;

    let initial = evaluate_chemistry_v0200(
        &model,
        &corpus.records,
        &validation_indices,
        batch_size,
        &causal_collator,
        &spectrum_collator,
        &chemistry_featurizer,
        &device,
        20_260_919,
    )?;
    print_chemistry_v0200("initial_validation", 0, initial);
    save_v0200_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &metadata.inverse_config,
        mode,
        0,
        initial,
        &parent_model,
    )?;
    save_v0200_checkpoint(
        &output_root.join("best"),
        &varmap,
        &metadata.inverse_config,
        mode,
        0,
        initial,
        &parent_model,
    )?;
    let mut best_objective = initial.objective;
    let mut best_metrics = initial;
    let mut best_step = 0usize;

    for step in 1..=train_steps {
        let selected = deterministic_batch(
            &train_indices,
            batch_size,
            20_260_919 ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let selected_records: Vec<&FoundationTrainingRecord> =
            selected.iter().map(|&i| &corpus.records[i]).collect();
        let packed = collate_records(
            &selected_records,
            &causal_collator,
            &spectrum_collator,
            &device,
        )?;
        let components = v0200_batch_components(&selected_records)?;
        let matched_chemistry = chemistry_featurizer.teacher_forced(
            &components.peptides,
            &components.spectra,
            &components.precursor_masses,
            &components.charges,
            &device,
        )?;
        let matched_output = model.forward_t(
            &packed.causal.input,
            &packed.spectrum,
            &packed.precursor,
            &matched_chemistry,
            true,
        )?;
        let matched = foundation_causal_next_token_loss(&matched_output, &packed.causal)?;

        let order =
            foundation_direct_shuffled_order(selected_records.len(), 20_260_919 ^ step as u64)?;
        let shuffled_spectra = order
            .iter()
            .map(|&index| components.spectra[index].clone())
            .collect::<Vec<_>>();
        let shuffled_batch = spectrum_collator.collate(&shuffled_spectra, &device)?;
        let shuffled_chemistry = chemistry_featurizer.teacher_forced(
            &components.peptides,
            &shuffled_spectra,
            &components.precursor_masses,
            &components.charges,
            &device,
        )?;
        let shuffled_output = model.forward_t(
            &packed.causal.input,
            &shuffled_batch,
            &packed.precursor,
            &shuffled_chemistry,
            true,
        )?;
        let shuffled = foundation_causal_next_token_loss(&shuffled_output, &packed.causal)?;
        let loss = foundation_direct_conditioning_loss(
            &matched,
            &shuffled,
            FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
            FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
        )?;
        let matched_value = f64::from(matched.to_scalar::<f32>()?);
        let shuffled_value = f64::from(shuffled.to_scalar::<f32>()?);
        let objective_value = f64::from(loss.to_scalar::<f32>()?);
        let should_print = step == 1 || step % 25 == 0 || step == train_steps;
        let rank_diagnostic = if should_print {
            let rank =
                foundation_direct_prefix_competitive_loss(&matched_output, &packed.causal, 0.0)?;
            Some((
                rank.mean_legal_rank,
                rank.top1_fraction,
                rank.top5_fraction,
                rank.top10_fraction,
            ))
        } else {
            None
        };
        let update = optimizer.backward_step(&loss, Some(5.0))?;
        if let Some((mean_legal_rank, top1_fraction, top5_fraction, top10_fraction)) =
            rank_diagnostic
        {
            println!(
                "train\tstep={step}\tobjective={objective_value:.6}\tmatched_nll={matched_value:.6}\tshuffled_nll={shuffled_value:.6}\tconditioning_gap={:.6}\ttarget_token_mean_legal_rank={:.6}\ttarget_token_top1_fraction={:.8}\ttarget_token_top5_fraction={:.8}\ttarget_token_top10_fraction={:.8}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                shuffled_value - matched_value,
                mean_legal_rank,
                top1_fraction,
                top5_fraction,
                top10_fraction,
                update.gradient_norm,
                update.gradient_scale
            );
        }
        if step % 100 == 0 || step == train_steps {
            let metrics = evaluate_chemistry_v0200(
                &model,
                &corpus.records,
                &validation_indices,
                batch_size,
                &causal_collator,
                &spectrum_collator,
                &chemistry_featurizer,
                &device,
                20_260_919 ^ step as u64,
            )?;
            print_chemistry_v0200("validation", step, metrics);
            save_v0200_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &metadata.inverse_config,
                mode,
                step,
                metrics,
                &parent_model,
            )?;
            if metrics.objective < best_objective
                && metrics.conditioning_gap >= FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190
            {
                best_objective = metrics.objective;
                best_metrics = metrics;
                best_step = step;
                save_v0200_checkpoint(
                    &output_root.join("best"),
                    &varmap,
                    &metadata.inverse_config,
                    mode,
                    step,
                    metrics,
                    &parent_model,
                )?;
            }
        }
    }

    let best_path = output_root.join("best/model.safetensors");
    let best_tensors = candle_core::safetensors::load(&best_path, &device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.20 VarMap lock poisoned"))?;
    for (name, variable) in data.iter() {
        variable.set(
            best_tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("v0.20 best checkpoint missing {name}"))?,
        )?;
    }
    drop(data);

    let generation = evaluate_chemistry_generation_v0200(
        &model,
        &corpus.records,
        &validation_indices,
        &causal_collator,
        &spectrum_collator,
        &chemistry_featurizer,
        &metadata.inverse_config,
        beam_width,
        &device,
    )?;
    print_generation_v0190("final", &generation);
    print_chemistry_v0200("final_best_validation", best_step, best_metrics);
    let conditioning_guard =
        best_metrics.conditioning_gap >= FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190;
    println!(
        "conditioning_guard\t{}",
        if conditioning_guard { "PASS" } else { "FAIL" }
    );
    if mode == "smoke" {
        println!("v0200_representation_gate\tNOT_EVALUATED_SMOKE_RUNTIME_ONLY");
        println!("v0200_stop_rule\tSMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION");
        return Ok(());
    }

    print_v0200_scientific_gate(&generation, conditioning_guard);
    Ok(())
}

fn print_v0200_scientific_gate(generation: &GenerationV0190, conditioning_guard: bool) {
    let gate = if generation.literal_top1 >= 28 && generation.il_top1 >= 42 && conditioning_guard {
        "PROGRESS_MILESTONE"
    } else if generation.literal_top1 >= 24 && generation.il_top1 >= 38 && conditioning_guard {
        "BASELINE_RECOVERY"
    } else if !conditioning_guard {
        "CONDITIONING_GUARD_NOT_MET"
    } else {
        "BELOW_BASELINE_RECOVERY"
    };
    println!("scientific_gate\t{gate}");
    let stop = match gate {
        "PROGRESS_MILESTONE" => "ACCEPT_V0200_CHEMISTRY_STRUCTURED_DECODER_PROGRESS",
        "BASELINE_RECOVERY" => "V0200_CHEMISTRY_STRUCTURED_BASELINE_RECOVERED",
        "CONDITIONING_GUARD_NOT_MET" => {
            "V0200_CHEMISTRY_CONDITIONING_GUARD_FAILED_CLOSE_AUTOREGRESSIVE_FAMILY_PIVOT_DIFFUSION"
        }
        _ => "V0200_CHEMISTRY_GATE_NOT_MET_CLOSE_AUTOREGRESSIVE_FAMILY_PIVOT_DIFFUSION",
    };
    println!("v0200_stop_rule\t{stop}");
}

#[derive(Debug, Clone)]
struct ChemistryBatchComponentsV0200 {
    peptides: Vec<PeptidoformInput>,
    spectra: Vec<FoundationSpectrum>,
    precursor_masses: Vec<f64>,
    charges: Vec<i32>,
}

fn v0200_batch_components(
    records: &[&FoundationTrainingRecord],
) -> Result<ChemistryBatchComponentsV0200> {
    let mut peptides = Vec::with_capacity(records.len());
    let mut spectra = Vec::with_capacity(records.len());
    let mut precursor_masses = Vec::with_capacity(records.len());
    let mut charges = Vec::with_capacity(records.len());
    for record in records {
        peptides.push(record.peptidoform.clone());
        spectra.push(
            FoundationSpectrum::from_training_record(record)
                .ok_or_else(|| anyhow::anyhow!("v0.20 selected record lacks observed spectrum"))?,
        );
        precursor_masses.push(v0200_precursor_mass(record)?);
        charges.push(
            record
                .context
                .charge
                .ok_or_else(|| anyhow::anyhow!("v0.20 selected record lacks precursor charge"))?,
        );
    }
    Ok(ChemistryBatchComponentsV0200 {
        peptides,
        spectra,
        precursor_masses,
        charges,
    })
}

fn v0200_precursor_mass(record: &FoundationTrainingRecord) -> Result<f64> {
    let mz = record
        .context
        .precursor_mz
        .ok_or_else(|| anyhow::anyhow!("v0.20 record lacks precursor m/z"))?;
    let charge = record
        .context
        .charge
        .ok_or_else(|| anyhow::anyhow!("v0.20 record lacks precursor charge"))?;
    foundation_precursor_neutral_mass(f64::from(mz), charge).map_err(anyhow::Error::msg)
}

fn v0200_true_path_contract(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    config: &FoundationDiffusionConfig,
    chemistry_featurizer: &ChemistryTransitionFeaturizer,
) -> Result<(usize, usize)> {
    let mut mass_feasible = 0usize;
    let mut suffix_feasible = 0usize;
    for &index in indices {
        let record = &records[index];
        let precursor_mass = v0200_precursor_mass(record)?;
        let target_mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        let physical = (target_mass - precursor_mass).abs() <= config.precursor_mass_tolerance_da;
        mass_feasible += usize::from(physical);
        if physical
            && chemistry_featurizer
                .true_path_feasible(&record.peptidoform, precursor_mass)
                .map_err(anyhow::Error::msg)?
        {
            suffix_feasible += 1;
        }
    }
    Ok((mass_feasible, suffix_feasible))
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ChemistryMetricsV0200 {
    matched_nll: f64,
    shuffled_nll: f64,
    conditioning_gap: f64,
    target_token_mean_legal_rank: f64,
    target_token_top1_fraction: f64,
    target_token_top5_fraction: f64,
    target_token_top10_fraction: f64,
    objective: f64,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_chemistry_v0200(
    model: &PeptideSpectrumChemistryDecoder,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    chemistry_featurizer: &ChemistryTransitionFeaturizer,
    device: &Device,
    seed: u64,
) -> Result<ChemistryMetricsV0200> {
    let mut matched_sum = 0.0;
    let mut shuffled_sum = 0.0;
    let mut rank_sum = 0.0;
    let mut top1_sum = 0.0;
    let mut top5_sum = 0.0;
    let mut top10_sum = 0.0;
    let mut batches = 0usize;
    for (chunk_index, chunk) in indices.chunks(batch_size).enumerate() {
        if chunk.len() < 2 {
            continue;
        }
        let selected = chunk.iter().map(|&i| &records[i]).collect::<Vec<_>>();
        let packed = collate_records(&selected, causal_collator, spectrum_collator, device)?;
        let components = v0200_batch_components(&selected)?;
        let matched_chemistry = chemistry_featurizer.teacher_forced(
            &components.peptides,
            &components.spectra,
            &components.precursor_masses,
            &components.charges,
            device,
        )?;
        let matched_output = model.forward_t(
            &packed.causal.input,
            &packed.spectrum,
            &packed.precursor,
            &matched_chemistry,
            false,
        )?;
        let matched = foundation_causal_next_token_loss(&matched_output, &packed.causal)?;
        let rank = foundation_direct_prefix_competitive_loss(&matched_output, &packed.causal, 0.0)?;

        let order = foundation_direct_shuffled_order(selected.len(), seed ^ chunk_index as u64)?;
        let shuffled_spectra = order
            .iter()
            .map(|&i| components.spectra[i].clone())
            .collect::<Vec<_>>();
        let shuffled_batch = spectrum_collator.collate(&shuffled_spectra, device)?;
        let shuffled_chemistry = chemistry_featurizer.teacher_forced(
            &components.peptides,
            &shuffled_spectra,
            &components.precursor_masses,
            &components.charges,
            device,
        )?;
        let shuffled_output = model.forward_t(
            &packed.causal.input,
            &shuffled_batch,
            &packed.precursor,
            &shuffled_chemistry,
            false,
        )?;
        let shuffled = foundation_causal_next_token_loss(&shuffled_output, &packed.causal)?;

        matched_sum += f64::from(matched.to_scalar::<f32>()?);
        shuffled_sum += f64::from(shuffled.to_scalar::<f32>()?);
        rank_sum += rank.mean_legal_rank;
        top1_sum += rank.top1_fraction;
        top5_sum += rank.top5_fraction;
        top10_sum += rank.top10_fraction;
        batches += 1;
    }
    if batches == 0 {
        anyhow::bail!("v0.20 validation produced no batches");
    }
    let n = batches as f64;
    let matched_nll = matched_sum / n;
    let shuffled_nll = shuffled_sum / n;
    let conditioning_gap = shuffled_nll - matched_nll;
    let objective = matched_nll
        + FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190
            * (FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190 - conditioning_gap).max(0.0);
    Ok(ChemistryMetricsV0200 {
        matched_nll,
        shuffled_nll,
        conditioning_gap,
        target_token_mean_legal_rank: rank_sum / n,
        target_token_top1_fraction: top1_sum / n,
        target_token_top5_fraction: top5_sum / n,
        target_token_top10_fraction: top10_sum / n,
        objective,
    })
}

fn print_chemistry_v0200(label: &str, step: usize, metrics: ChemistryMetricsV0200) {
    println!(
        "{label}\tstep={step}\tobjective={:.6}\tmatched_spectrum_token_nll={:.6}\tshuffled_spectrum_token_nll={:.6}\tconditioning_gap={:.6}\ttarget_token_mean_legal_rank={:.6}\ttarget_token_top1_fraction={:.8}\ttarget_token_top5_fraction={:.8}\ttarget_token_top10_fraction={:.8}",
        metrics.objective,
        metrics.matched_nll,
        metrics.shuffled_nll,
        metrics.conditioning_gap,
        metrics.target_token_mean_legal_rank,
        metrics.target_token_top1_fraction,
        metrics.target_token_top5_fraction,
        metrics.target_token_top10_fraction,
    );
}

#[derive(Debug, Serialize)]
struct ChemistryCheckpointV0200<'a> {
    version: &'a str,
    objective: &'a str,
    architecture: &'a str,
    run_mode: &'a str,
    test_partition_consumed: bool,
    global_step: usize,
    transition_feature_dim: usize,
    fragment_match_ppm: f64,
    fragment_match_abs_floor_da: f64,
    suffix_mass_bin_da: f64,
    validation: ChemistryMetricsV0200,
    unified_parent: String,
    inverse_config: FoundationDiffusionConfig,
}

fn save_v0200_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    config: &FoundationDiffusionConfig,
    mode: &str,
    step: usize,
    validation: ChemistryMetricsV0200,
    parent: &Path,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    let metadata = ChemistryCheckpointV0200 {
        version: "v0.20.0",
        objective: FOUNDATION_CHEMISTRY_DECODER_OBJECTIVE_V0200,
        architecture: FOUNDATION_CHEMISTRY_DECODER_ARCHITECTURE_V0200,
        run_mode: mode,
        test_partition_consumed: false,
        global_step: step,
        transition_feature_dim: FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
        fragment_match_ppm: FOUNDATION_CHEMISTRY_FRAGMENT_PPM_V0200,
        fragment_match_abs_floor_da: FOUNDATION_CHEMISTRY_FRAGMENT_ABS_TOLERANCE_DA_V0200,
        suffix_mass_bin_da: FOUNDATION_CHEMISTRY_SUFFIX_BIN_DA_V0200,
        validation,
        unified_parent: parent.display().to_string(),
        inverse_config: config.clone(),
    };
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(&metadata)?,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_chemistry_generation_v0200(
    model: &PeptideSpectrumChemistryDecoder,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    chemistry_featurizer: &ChemistryTransitionFeaturizer,
    config: &FoundationDiffusionConfig,
    beam_width: usize,
    device: &Device,
) -> Result<GenerationV0190> {
    let vocabulary = FoundationDiffusionVocabulary;
    let ks = [5usize, 10, 32, 128];
    let mut out = GenerationV0190 {
        records: 0,
        literal_top1: 0,
        sequence_top1: 0,
        il_top1: 0,
        topk: ks.iter().copied().map(|k| (k, 0, 0)).collect(),
        mass_valid_beams: 0,
        returned_beams: 0,
        zero_returned_beam_records: 0,
    };
    for &index in indices {
        let record = &records[index];
        let neutral_mass = v0200_precursor_mass(record)?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("v0.20 validation record {index} lacks charge"))?;
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("v0.20 validation record {index} lacks spectrum"))?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
        let precursor = precursor_context(&[record], device)?;
        let context = model.prepare_context(&spectrum_batch, &precursor, false)?;
        let candidates = foundation_direct_beam_search(
            neutral_mass,
            DirectDecoderBeamConfig {
                beam_width,
                top_k: 128,
                mass_tolerance_da: config.precursor_mass_tolerance_da,
                max_tokens: config.max_tokens,
            },
            |prefixes| {
                let input = causal_collator
                    .collate_compact_prefix_rows(prefixes, device)
                    .map_err(|error| error.to_string())?;
                let chemistry = chemistry_featurizer
                    .next_prefixes(prefixes, &spectrum, neutral_mass, charge, device)
                    .map_err(|error| error.to_string())?;
                let mut rows = model
                    .forward_next_t_with_context(&input, &context, &chemistry, false)
                    .and_then(|tensor| tensor.to_vec2::<f32>())
                    .map_err(|error| error.to_string())?;
                chemistry_featurizer.mask_infeasible_next_logits(
                    prefixes,
                    &mut rows,
                    neutral_mass,
                )?;
                Ok(rows)
            },
        )
        .map_err(anyhow::Error::msg)?;
        out.records += 1;
        out.zero_returned_beam_records += usize::from(candidates.is_empty());
        out.returned_beams += candidates.len();
        out.mass_valid_beams += candidates
            .iter()
            .filter(|candidate| candidate.mass_error_da.abs() <= config.precursor_mass_tolerance_da)
            .count();
        let decoded = candidates
            .iter()
            .filter_map(|candidate| vocabulary.decode(&candidate.tokens).ok())
            .collect::<Vec<_>>();
        if let Some(first) = decoded.first() {
            out.literal_top1 += usize::from(first == &record.peptidoform);
            out.sequence_top1 += usize::from(first.sequence == record.peptidoform.sequence);
            out.il_top1 += usize::from(
                il_sequence(&first.sequence) == il_sequence(&record.peptidoform.sequence),
            );
        }
        for metric in &mut out.topk {
            let limit = metric.0.min(decoded.len());
            metric.1 += usize::from(decoded[..limit].iter().any(|p| p == &record.peptidoform));
            metric.2 +=
                usize::from(decoded[..limit].iter().any(|p| {
                    il_sequence(&p.sequence) == il_sequence(&record.peptidoform.sequence)
                }));
        }
    }
    Ok(out)
}

const FROZEN_V0190_VALIDATION_RECORDS: usize = 125;
const FROZEN_V0190_ORACLE_LITERAL: usize = 44;
const FROZEN_V0190_ORACLE_IL: usize = 54;
const FROZEN_V0190_LEGACY_LITERAL_TOP1: usize = 24;
const FROZEN_V0190_LEGACY_IL_TOP1: usize = 38;

#[derive(Debug, Clone)]
struct FrozenValidationCohortV0190 {
    indices: Vec<usize>,
    oracle_literal: usize,
    oracle_il: usize,
    legacy_literal_top1: usize,
    legacy_il_top1: usize,
}

#[derive(Debug, Clone)]
struct FrozenValidationGroupV0190 {
    oracle_literal: bool,
    oracle_il: bool,
    legacy_rank: usize,
    legacy_literal_top1: bool,
    legacy_il_top1: bool,
}

impl Default for FrozenValidationGroupV0190 {
    fn default() -> Self {
        Self {
            oracle_literal: false,
            oracle_il: false,
            legacy_rank: usize::MAX,
            legacy_literal_top1: false,
            legacy_il_top1: false,
        }
    }
}

fn read_frozen_validation_cohort_v0190(
    path: &Path,
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    config: &FoundationDiffusionConfig,
    vocabulary: FoundationDiffusionVocabulary,
) -> Result<FrozenValidationCohortV0190> {
    let file = fs::File::open(path)
        .with_context(|| format!("read frozen v0.19 validation cohort {path:?}"))?;
    let cohort = parse_frozen_validation_cohort_v0190(BufReader::new(file))?;

    let validation_partition = benchmark
        .partition_indices(FoundationPartition::Validation)
        .into_iter()
        .collect::<HashSet<_>>();
    for &index in &cohort.indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow::anyhow!(
                "frozen v0.19 validation cohort contains out-of-range record index {index}"
            )
        })?;
        if !validation_partition.contains(&index) {
            anyhow::bail!(
                "frozen v0.19 validation cohort record {index} is not assigned to VALIDATION"
            );
        }
        if FoundationSpectrum::from_training_record(record).is_none()
            || vocabulary
                .encode(&record.peptidoform, config.max_tokens)
                .is_err()
        {
            anyhow::bail!(
                "frozen v0.19 validation cohort record {index} is not usable by the direct decoder"
            );
        }
    }
    Ok(cohort)
}

fn parse_frozen_validation_cohort_v0190<R: BufRead>(
    mut reader: R,
) -> Result<FrozenValidationCohortV0190> {
    let mut header = String::new();
    if reader.read_line(&mut header)? == 0 {
        anyhow::bail!("frozen v0.19 validation candidate TSV is empty");
    }
    let columns = header
        .trim_end_matches(&['\r', '\n'][..])
        .split('\t')
        .collect::<Vec<_>>();
    let mut column_index = BTreeMap::<&str, usize>::new();
    for (index, name) in columns.iter().enumerate() {
        column_index.insert(*name, index);
    }
    for required in [
        "record_index",
        "fragment_causal_mass_rank",
        "mass_valid",
        "peptidoform_exact",
        "il_sequence_exact",
    ] {
        if !column_index.contains_key(required) {
            anyhow::bail!(
                "frozen v0.19 validation candidate TSV missing required column '{required}'"
            );
        }
    }

    let mut groups = BTreeMap::<usize, FrozenValidationGroupV0190>::new();
    let mut group_order = Vec::<usize>::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let get = |name: &str| -> Result<&str> {
            let index = *column_index
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("internal missing candidate TSV column {name}"))?;
            fields
                .get(index)
                .copied()
                .with_context(|| format!("candidate row missing column {name}"))
        };
        if !parse_bool_v0190(get("mass_valid")?) {
            continue;
        }
        let record_index: usize = get("record_index")?.parse()?;
        let rank: usize = get("fragment_causal_mass_rank")?.parse()?;
        let exact = parse_bool_v0190(get("peptidoform_exact")?);
        let il_exact = parse_bool_v0190(get("il_sequence_exact")?);
        if !groups.contains_key(&record_index) {
            group_order.push(record_index);
        }
        let group = groups.entry(record_index).or_default();
        group.oracle_literal |= exact;
        group.oracle_il |= il_exact;
        if rank < group.legacy_rank {
            group.legacy_rank = rank;
            group.legacy_literal_top1 = exact;
            group.legacy_il_top1 = il_exact;
        }
    }

    let cohort = FrozenValidationCohortV0190 {
        indices: group_order,
        oracle_literal: groups.values().filter(|group| group.oracle_literal).count(),
        oracle_il: groups.values().filter(|group| group.oracle_il).count(),
        legacy_literal_top1: groups
            .values()
            .filter(|group| group.legacy_literal_top1)
            .count(),
        legacy_il_top1: groups.values().filter(|group| group.legacy_il_top1).count(),
    };
    if cohort.indices.len() != FROZEN_V0190_VALIDATION_RECORDS
        || cohort.oracle_literal != FROZEN_V0190_ORACLE_LITERAL
        || cohort.oracle_il != FROZEN_V0190_ORACLE_IL
        || cohort.legacy_literal_top1 != FROZEN_V0190_LEGACY_LITERAL_TOP1
        || cohort.legacy_il_top1 != FROZEN_V0190_LEGACY_IL_TOP1
    {
        anyhow::bail!(
            "frozen v0.19 validation cohort mismatch: records={} oracle={}/{} legacy_top1={}/{} expected records={} oracle={}/{} legacy_top1={}/{}",
            cohort.indices.len(),
            cohort.oracle_literal,
            cohort.oracle_il,
            cohort.legacy_literal_top1,
            cohort.legacy_il_top1,
            FROZEN_V0190_VALIDATION_RECORDS,
            FROZEN_V0190_ORACLE_LITERAL,
            FROZEN_V0190_ORACLE_IL,
            FROZEN_V0190_LEGACY_LITERAL_TOP1,
            FROZEN_V0190_LEGACY_IL_TOP1
        );
    }
    Ok(cohort)
}

fn parse_bool_v0190(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "y"
    )
}

#[cfg(test)]
mod v0190_validation_cohort_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn frozen_validation_candidate_contract_reconstructs_accepted_125_record_cohort() {
        let mut tsv = String::from(
            "record_index\tfragment_causal_mass_rank\tmass_valid\tpeptidoform_exact\til_sequence_exact\n",
        );
        for record_index in 0..FROZEN_V0190_VALIDATION_RECORDS {
            let top1_exact = record_index < FROZEN_V0190_LEGACY_LITERAL_TOP1;
            let top1_il = record_index < FROZEN_V0190_LEGACY_IL_TOP1;
            tsv.push_str(&format!(
                "{record_index}\t0\ttrue\t{top1_exact}\t{top1_il}\n"
            ));
            if (24..44).contains(&record_index) {
                tsv.push_str(&format!("{record_index}\t1\ttrue\ttrue\ttrue\n"));
            } else if (44..54).contains(&record_index) {
                tsv.push_str(&format!("{record_index}\t1\ttrue\tfalse\ttrue\n"));
            }
        }

        let cohort = parse_frozen_validation_cohort_v0190(Cursor::new(tsv)).unwrap();
        assert_eq!(cohort.indices.len(), FROZEN_V0190_VALIDATION_RECORDS);
        assert_eq!(
            cohort.indices,
            (0..FROZEN_V0190_VALIDATION_RECORDS).collect::<Vec<_>>()
        );
        assert_eq!(cohort.oracle_literal, FROZEN_V0190_ORACLE_LITERAL);
        assert_eq!(cohort.oracle_il, FROZEN_V0190_ORACLE_IL);
        assert_eq!(cohort.legacy_literal_top1, FROZEN_V0190_LEGACY_LITERAL_TOP1);
        assert_eq!(cohort.legacy_il_top1, FROZEN_V0190_LEGACY_IL_TOP1);
    }
}

#[derive(Debug, Deserialize)]
struct UnifiedV0190Metadata {
    inverse_config: FoundationDiffusionConfig,
    #[serde(default)]
    global_step: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ConditioningV0190 {
    matched_nll: f64,
    shuffled_nll: f64,
    conditioning_gap: f64,
    objective: f64,
}

fn evaluate_conditioning_v0190(
    model: &PeptideSpectrumCausalModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
    seed: u64,
) -> Result<ConditioningV0190> {
    let mut matched_sum = 0.0;
    let mut shuffled_sum = 0.0;
    let mut batches = 0usize;
    for (chunk_index, chunk) in indices.chunks(batch_size).enumerate() {
        if chunk.len() < 2 {
            continue;
        }
        let selected: Vec<&FoundationTrainingRecord> = chunk.iter().map(|&i| &records[i]).collect();
        let packed = collate_records(&selected, causal_collator, spectrum_collator, device)?;
        let matched = foundation_causal_next_token_loss(
            &model.forward_t(
                &packed.causal.input,
                &packed.spectrum,
                &packed.precursor,
                false,
            )?,
            &packed.causal,
        )?;
        let order = foundation_direct_shuffled_order(selected.len(), seed ^ chunk_index as u64)?;
        let spectra: Vec<FoundationSpectrum> = order
            .iter()
            .map(|&i| FoundationSpectrum::from_training_record(selected[i]).unwrap())
            .collect();
        let shuffled_spectrum = spectrum_collator.collate(&spectra, device)?;
        let shuffled = foundation_causal_next_token_loss(
            &model.forward_t(
                &packed.causal.input,
                &shuffled_spectrum,
                &packed.precursor,
                false,
            )?,
            &packed.causal,
        )?;
        matched_sum += f64::from(matched.to_scalar::<f32>()?);
        shuffled_sum += f64::from(shuffled.to_scalar::<f32>()?);
        batches += 1;
    }
    if batches == 0 {
        anyhow::bail!("v0.19 conditioning evaluation produced no batches");
    }
    let matched_nll = matched_sum / batches as f64;
    let shuffled_nll = shuffled_sum / batches as f64;
    let conditioning_gap = shuffled_nll - matched_nll;
    let objective = matched_nll
        + FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190
            * (FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190 - conditioning_gap).max(0.0);
    Ok(ConditioningV0190 {
        matched_nll,
        shuffled_nll,
        conditioning_gap,
        objective,
    })
}

fn print_conditioning_v0190(label: &str, step: usize, metrics: ConditioningV0190) {
    println!(
        "{label}\tstep={step}\tobjective={:.6}\tmatched_spectrum_token_nll={:.6}\tshuffled_spectrum_token_nll={:.6}\tconditioning_gap={:.6}",
        metrics.objective, metrics.matched_nll, metrics.shuffled_nll, metrics.conditioning_gap
    );
}

#[derive(Debug, Serialize)]
struct DirectCheckpointV0190<'a> {
    version: &'a str,
    objective: &'a str,
    architecture: &'a str,
    run_mode: &'a str,
    test_partition_consumed: bool,
    global_step: usize,
    validation: ConditioningV0190,
    unified_parent: String,
    inverse_config: FoundationDiffusionConfig,
}

fn save_v0190_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    config: &FoundationDiffusionConfig,
    mode: &str,
    step: usize,
    validation: ConditioningV0190,
    parent: &Path,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    let metadata = DirectCheckpointV0190 {
        version: "v0.19.0",
        objective: FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0190,
        architecture: "masked_self_attention+full_peak_cross_attention+ff",
        run_mode: mode,
        test_partition_consumed: false,
        global_step: step,
        validation,
        unified_parent: parent.display().to_string(),
        inverse_config: config.clone(),
    };
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(&metadata)?,
    )?;
    Ok(())
}

#[derive(Debug)]
struct GenerationV0190 {
    records: usize,
    literal_top1: usize,
    sequence_top1: usize,
    il_top1: usize,
    topk: Vec<(usize, usize, usize)>,
    mass_valid_beams: usize,
    returned_beams: usize,
    zero_returned_beam_records: usize,
}

impl GenerationV0190 {
    fn mass_valid_fraction(&self) -> f64 {
        if self.returned_beams == 0 {
            0.0
        } else {
            self.mass_valid_beams as f64 / self.returned_beams as f64
        }
    }

    fn mean_returned_beams(&self) -> f64 {
        if self.records == 0 {
            0.0
        } else {
            self.returned_beams as f64 / self.records as f64
        }
    }
}

fn print_generation_v0190(prefix: &str, generation: &GenerationV0190) {
    println!("{prefix}_records\t{}", generation.records);
    println!("{prefix}_literal_top1\t{}", generation.literal_top1);
    println!("{prefix}_sequence_top1\t{}", generation.sequence_top1);
    println!("{prefix}_il_top1\t{}", generation.il_top1);
    for &(k, literal, il) in &generation.topk {
        println!("{prefix}_top{k}_literal\t{literal}");
        println!("{prefix}_top{k}_il\t{il}");
    }
    println!("{prefix}_mass_valid_beams\t{}", generation.mass_valid_beams);
    println!("{prefix}_returned_beams\t{}", generation.returned_beams);
    println!(
        "{prefix}_zero_returned_beam_records\t{}",
        generation.zero_returned_beam_records
    );
    println!(
        "{prefix}_mean_returned_beams\t{:.6}",
        generation.mean_returned_beams()
    );
    println!(
        "{prefix}_mass_valid_beam_fraction\t{:.8}",
        generation.mass_valid_fraction()
    );
}

#[derive(Debug, Default)]
struct DirectDecoderAuditV0190 {
    records: usize,
    target_mass_feasible: usize,
    target_path_valid: usize,
    target_returned_literal: usize,
    target_returned_il: usize,
    zero_returned_beam_records: usize,
    returned_beams: usize,
    target_tokens: usize,
    target_token_rank_sum: usize,
    target_token_top1: usize,
    target_token_top5: usize,
    target_token_top10: usize,
    target_token_top20: usize,
    target_score_beats_best_returned: usize,
    target_score_would_rank_top128_vs_returned: usize,
}

fn print_direct_decoder_audit_v0190(audit: &DirectDecoderAuditV0190) {
    let token_fraction = |count: usize| -> f64 {
        if audit.target_tokens == 0 {
            0.0
        } else {
            count as f64 / audit.target_tokens as f64
        }
    };
    let mean_rank = if audit.target_tokens == 0 {
        0.0
    } else {
        audit.target_token_rank_sum as f64 / audit.target_tokens as f64
    };
    println!("decoder_audit_records\t{}", audit.records);
    println!(
        "decoder_audit_target_mass_feasible\t{}",
        audit.target_mass_feasible
    );
    println!(
        "decoder_audit_target_path_valid\t{}",
        audit.target_path_valid
    );
    println!(
        "decoder_audit_target_returned_literal\t{}",
        audit.target_returned_literal
    );
    println!(
        "decoder_audit_target_returned_il\t{}",
        audit.target_returned_il
    );
    println!(
        "decoder_audit_zero_returned_beam_records\t{}",
        audit.zero_returned_beam_records
    );
    println!("decoder_audit_returned_beams\t{}", audit.returned_beams);
    println!("decoder_audit_target_tokens\t{}", audit.target_tokens);
    println!("decoder_audit_target_token_mean_raw_rank\t{mean_rank:.6}");
    println!(
        "decoder_audit_target_token_top1_fraction\t{:.8}",
        token_fraction(audit.target_token_top1)
    );
    println!(
        "decoder_audit_target_token_top5_fraction\t{:.8}",
        token_fraction(audit.target_token_top5)
    );
    println!(
        "decoder_audit_target_token_top10_fraction\t{:.8}",
        token_fraction(audit.target_token_top10)
    );
    println!(
        "decoder_audit_target_token_top20_fraction\t{:.8}",
        token_fraction(audit.target_token_top20)
    );
    println!(
        "decoder_audit_target_score_beats_best_returned\t{}",
        audit.target_score_beats_best_returned
    );
    println!(
        "decoder_audit_target_score_would_rank_top128_vs_returned\t{}",
        audit.target_score_would_rank_top128_vs_returned
    );
}

fn v0190_target_token_allowed(prefix: &[u32], token: u32) -> bool {
    if token == FOUNDATION_DIFFUSION_PAD
        || token == FOUNDATION_DIFFUSION_MASK
        || token == FOUNDATION_DIFFUSION_EOS
    {
        return false;
    }
    if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
        return prefix.is_empty();
    }
    if foundation_diffusion_token_residue(token).is_some() {
        return true;
    }
    if !matches!(
        token,
        FOUNDATION_DIFFUSION_RESIDUE_ACETYL
            | FOUNDATION_DIFFUSION_CARBAMIDOMETHYL
            | FOUNDATION_DIFFUSION_DEAMIDATED
            | FOUNDATION_DIFFUSION_OXIDATION
    ) {
        return false;
    }
    let Some(previous) = prefix.last().copied() else {
        return false;
    };
    let Some(residue) = foundation_diffusion_token_residue(previous) else {
        return false;
    };
    foundation_diffusion_residue_ptm_valid(token, residue)
}

fn v0190_selected_log_softmax(logits: &[f32], selected: usize) -> Result<f64> {
    if selected >= logits.len() || !logits[selected].is_finite() {
        anyhow::bail!("selected v0.19 audit logit is invalid");
    }
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let denominator: f64 = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| f64::from(value - max).exp())
        .sum();
    if !(denominator > 0.0 && denominator.is_finite()) {
        anyhow::bail!("v0.19 audit softmax denominator is invalid");
    }
    Ok(f64::from(logits[selected] - max) - denominator.ln())
}

fn active_target_tokens_v0190(
    vocabulary: FoundationDiffusionVocabulary,
    peptide: &PeptidoformInput,
    max_tokens: usize,
) -> Result<Vec<u32>> {
    let encoded = vocabulary
        .encode(peptide, max_tokens)
        .map_err(anyhow::Error::msg)?;
    let active = encoded
        .into_iter()
        .take_while(|&token| token != FOUNDATION_DIFFUSION_PAD)
        .collect::<Vec<_>>();
    if active.last().copied() != Some(FOUNDATION_DIFFUSION_EOS) {
        anyhow::bail!("v0.19 audit target token row does not terminate in EOS");
    }
    Ok(active)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_direct_decoder_audit_v0190(
    model: &PeptideSpectrumCausalModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    beam_width: usize,
    device: &Device,
    output_root: &Path,
) -> Result<DirectDecoderAuditV0190> {
    fs::create_dir_all(output_root)?;
    let path = output_root.join("direct_decoder_v0190_audit_records.tsv");
    let mut writer = BufWriter::new(
        fs::File::create(&path).with_context(|| format!("create v0.19 decoder audit {path:?}"))?,
    );
    writeln!(
        writer,
        "record_index\ttarget_sequence\ttarget_mass_error_da\ttarget_mass_feasible\ttarget_path_valid\ttarget_active_tokens\ttarget_mean_raw_token_rank\ttarget_token_top1_fraction\ttarget_token_top5_fraction\ttarget_token_top10_fraction\ttarget_token_top20_fraction\ttarget_total_log_probability\treturned_beams\tbest_returned_log_probability\treturned_literal\treturned_il\ttarget_score_beats_best_returned\ttarget_score_rank_vs_returned"
    )?;

    let vocabulary = FoundationDiffusionVocabulary;
    let mut audit = DirectDecoderAuditV0190::default();
    for &index in indices {
        let record = &records[index];
        let mz = record
            .context
            .precursor_mz
            .ok_or_else(|| anyhow::anyhow!("validation record {index} lacks precursor m/z"))?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("validation record {index} lacks charge"))?;
        let precursor_mass =
            foundation_precursor_neutral_mass(f64::from(mz), charge).map_err(anyhow::Error::msg)?;
        let target_mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        let target_mass_error = target_mass - precursor_mass;
        let target_mass_feasible = target_mass_error.abs() <= config.precursor_mass_tolerance_da;

        let target_tokens =
            active_target_tokens_v0190(vocabulary, &record.peptidoform, config.max_tokens)?;
        let target_nonterminal = &target_tokens[..target_tokens.len() - 1];
        let mut target_path_valid = true;
        let mut running_mass = FOUNDATION_PEPTIDE_WATER_MASS_DA;
        let mut prefix = Vec::<u32>::new();
        for &token in target_nonterminal {
            if !v0190_target_token_allowed(&prefix, token) {
                target_path_valid = false;
                break;
            }
            let Some(token_mass) = foundation_diffusion_token_mass_da(token) else {
                target_path_valid = false;
                break;
            };
            running_mass += token_mass;
            if running_mass > precursor_mass + config.precursor_mass_tolerance_da {
                target_path_valid = false;
                break;
            }
            prefix.push(token);
        }
        if (running_mass - precursor_mass).abs() > config.precursor_mass_tolerance_da {
            target_path_valid = false;
        }

        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("validation record {index} lacks spectrum"))?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum], device)?;
        let precursor = precursor_context(&[record], device)?;
        let context = model.prepare_context(&spectrum_batch, &precursor, false)?;

        let mut target_total_log_probability = 0.0f64;
        let mut token_rank_sum = 0usize;
        let mut token_top1 = 0usize;
        let mut token_top5 = 0usize;
        let mut token_top10 = 0usize;
        let mut token_top20 = 0usize;
        let mut prefix = Vec::<u32>::new();
        for &target_token in &target_tokens {
            let rows = vec![prefix.clone()];
            let input = causal_collator.collate_compact_prefix_rows(&rows, device)?;
            let logits = model
                .forward_next_t_with_context(&input, &context, false)?
                .to_vec2::<f32>()?;
            let row = logits
                .first()
                .ok_or_else(|| anyhow::anyhow!("v0.19 audit produced no next-token logits"))?;
            let selected = target_token as usize;
            let rank = 1 + row
                .iter()
                .enumerate()
                .filter(|(token_index, value)| {
                    *token_index != selected && value.is_finite() && **value > row[selected]
                })
                .count();
            target_total_log_probability += v0190_selected_log_softmax(row, selected)?;
            token_rank_sum += rank;
            token_top1 += usize::from(rank <= 1);
            token_top5 += usize::from(rank <= 5);
            token_top10 += usize::from(rank <= 10);
            token_top20 += usize::from(rank <= 20);
            if target_token != FOUNDATION_DIFFUSION_EOS {
                prefix.push(target_token);
            }
        }

        let candidates = foundation_direct_beam_search(
            precursor_mass,
            DirectDecoderBeamConfig {
                beam_width,
                top_k: 128,
                mass_tolerance_da: config.precursor_mass_tolerance_da,
                max_tokens: config.max_tokens,
            },
            |prefixes| {
                let input = causal_collator
                    .collate_compact_prefix_rows(prefixes, device)
                    .map_err(|error| error.to_string())?;
                model
                    .forward_next_t_with_context(&input, &context, false)
                    .and_then(|tensor| tensor.to_vec2::<f32>())
                    .map_err(|error| error.to_string())
            },
        )
        .map_err(anyhow::Error::msg)?;
        let decoded = candidates
            .iter()
            .map(|candidate| vocabulary.decode(&candidate.tokens))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::msg)?;
        let returned_literal = decoded.iter().any(|peptide| peptide == &record.peptidoform);
        let returned_il = decoded.iter().any(|peptide| {
            il_sequence(&peptide.sequence) == il_sequence(&record.peptidoform.sequence)
        });
        let best_returned = candidates
            .first()
            .map(|candidate| candidate.log_probability);
        let target_score_beats_best = best_returned
            .map(|score| target_total_log_probability > score)
            .unwrap_or(target_mass_feasible && target_path_valid);
        let better_than_target = candidates
            .iter()
            .filter(|candidate| candidate.log_probability > target_total_log_probability)
            .count();
        let target_rank_vs_returned = better_than_target + 1;

        audit.records += 1;
        audit.target_mass_feasible += usize::from(target_mass_feasible);
        audit.target_path_valid += usize::from(target_path_valid);
        audit.target_returned_literal += usize::from(returned_literal);
        audit.target_returned_il += usize::from(returned_il);
        audit.zero_returned_beam_records += usize::from(candidates.is_empty());
        audit.returned_beams += candidates.len();
        audit.target_tokens += target_tokens.len();
        audit.target_token_rank_sum += token_rank_sum;
        audit.target_token_top1 += token_top1;
        audit.target_token_top5 += token_top5;
        audit.target_token_top10 += token_top10;
        audit.target_token_top20 += token_top20;
        audit.target_score_beats_best_returned +=
            usize::from(target_mass_feasible && target_path_valid && target_score_beats_best);
        audit.target_score_would_rank_top128_vs_returned += usize::from(
            target_mass_feasible && target_path_valid && target_rank_vs_returned <= 128,
        );

        let target_token_count = target_tokens.len().max(1) as f64;
        let mean_rank = token_rank_sum as f64 / target_token_count;
        let best_returned_text = best_returned
            .map(|value| format!("{value:.8}"))
            .unwrap_or_default();
        writeln!(
            writer,
            "{index}\t{}\t{target_mass_error:.8}\t{}\t{}\t{}\t{mean_rank:.6}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{target_total_log_probability:.8}\t{}\t{}\t{}\t{}\t{}\t{}",
            record.peptidoform.sequence,
            target_mass_feasible,
            target_path_valid,
            target_tokens.len(),
            token_top1 as f64 / target_token_count,
            token_top5 as f64 / target_token_count,
            token_top10 as f64 / target_token_count,
            token_top20 as f64 / target_token_count,
            candidates.len(),
            best_returned_text,
            returned_literal,
            returned_il,
            target_score_beats_best,
            target_rank_vs_returned,
        )?;
    }
    writer.flush()?;
    println!("decoder_audit_records_tsv\t{}", path.display());
    Ok(audit)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_direct_generation_v0190(
    model: &PeptideSpectrumCausalModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    beam_width: usize,
    device: &Device,
) -> Result<GenerationV0190> {
    let vocabulary = FoundationDiffusionVocabulary;
    let ks = [5usize, 10, 32, 128];
    let mut out = GenerationV0190 {
        records: 0,
        literal_top1: 0,
        sequence_top1: 0,
        il_top1: 0,
        topk: ks.iter().copied().map(|k| (k, 0, 0)).collect(),
        mass_valid_beams: 0,
        returned_beams: 0,
        zero_returned_beam_records: 0,
    };
    for &index in indices {
        let record = &records[index];
        let mz = record
            .context
            .precursor_mz
            .ok_or_else(|| anyhow::anyhow!("validation record {index} lacks precursor m/z"))?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("validation record {index} lacks charge"))?;
        let neutral_mass =
            foundation_precursor_neutral_mass(f64::from(mz), charge).map_err(anyhow::Error::msg)?;
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("validation record {index} lacks spectrum"))?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum], device)?;
        let precursor = precursor_context(&[record], device)?;
        let context = model.prepare_context(&spectrum_batch, &precursor, false)?;
        let candidates = foundation_direct_beam_search(
            neutral_mass,
            DirectDecoderBeamConfig {
                beam_width,
                top_k: 128,
                mass_tolerance_da: config.precursor_mass_tolerance_da,
                max_tokens: config.max_tokens,
            },
            |prefixes| {
                let input = causal_collator
                    .collate_compact_prefix_rows(prefixes, device)
                    .map_err(|e| e.to_string())?;
                model
                    .forward_next_t_with_context(&input, &context, false)
                    .and_then(|t| t.to_vec2::<f32>())
                    .map_err(|e| e.to_string())
            },
        )
        .map_err(anyhow::Error::msg)?;
        out.records += 1;
        out.zero_returned_beam_records += usize::from(candidates.is_empty());
        out.returned_beams += candidates.len();
        out.mass_valid_beams += candidates
            .iter()
            .filter(|c| c.mass_error_da.abs() <= config.precursor_mass_tolerance_da)
            .count();
        let decoded: Vec<PeptidoformInput> = candidates
            .iter()
            .filter_map(|candidate| vocabulary.decode(&candidate.tokens).ok())
            .collect();
        if let Some(first) = decoded.first() {
            out.literal_top1 += usize::from(first == &record.peptidoform);
            out.sequence_top1 += usize::from(first.sequence == record.peptidoform.sequence);
            out.il_top1 += usize::from(
                il_sequence(&first.sequence) == il_sequence(&record.peptidoform.sequence),
            );
        }
        for metric in &mut out.topk {
            let limit = metric.0.min(decoded.len());
            metric.1 += usize::from(decoded[..limit].iter().any(|p| p == &record.peptidoform));
            metric.2 +=
                usize::from(decoded[..limit].iter().any(|p| {
                    il_sequence(&p.sequence) == il_sequence(&record.peptidoform.sequence)
                }));
        }
    }
    Ok(out)
}

fn il_sequence(sequence: &str) -> String {
    sequence
        .chars()
        .map(|aa| if aa == 'I' { 'L' } else { aa })
        .collect()
}

fn reject_test_path_v0190(path: &Path) -> Result<()> {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|part| part == "test")
    {
        anyhow::bail!(
            "v0.19 refuses TEST-labelled input/output path: {}",
            path.display()
        );
    }
    Ok(())
}

struct PackedBatch {
    causal: redeem_properties::foundation::FoundationCausalBatch,
    spectrum: redeem_properties::foundation::FoundationSpectrumBatch,
    precursor: PrecursorContextBatch,
}

fn collate_records(
    records: &[&FoundationTrainingRecord],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<PackedBatch> {
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let spectra: Vec<FoundationSpectrum> = records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record).ok_or_else(|| {
                anyhow::anyhow!("selected causal record unexpectedly lacks observed spectrum")
            })
        })
        .collect::<Result<_>>()?;
    Ok(PackedBatch {
        causal: causal_collator.collate(&peptides, device)?,
        spectrum: spectrum_collator.collate(&spectra, device)?,
        precursor: precursor_context(records, device)?,
    })
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
    let b = records.len();
    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(charge, b, device)?,
        charge_present: Tensor::from_vec(charge_present, b, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, b, device)?,
        precursor_mz_present: Tensor::from_vec(precursor_mz_present, b, device)?,
        nce: Tensor::zeros(b, DType::F32, device)?,
        nce_present: Tensor::zeros(b, DType::F32, device)?,
        instrument_ids: Tensor::zeros(b, DType::U32, device)?,
        instrument_present: Tensor::zeros(b, DType::F32, device)?,
    })
}

fn usable_indices(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    config: &FoundationDiffusionConfig,
    vocabulary: FoundationDiffusionVocabulary,
) -> Vec<usize> {
    benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
        .filter_map(|entry| {
            let record = &records[entry.record_index];
            (FoundationSpectrum::from_training_record(record).is_some()
                && vocabulary
                    .encode(&record.peptidoform, config.max_tokens)
                    .is_ok())
            .then_some(entry.record_index)
        })
        .collect()
}

fn evaluate(
    model: &PeptideSpectrumCausalModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<CausalMetrics> {
    let mut loss_sum = 0.0f64;
    let mut correct_tokens = 0usize;
    let mut active_tokens = 0usize;
    let mut correct_eos = 0usize;
    let mut sequences = 0usize;
    let mut exact_sequences = 0usize;

    for chunk in indices.chunks(batch_size) {
        let selected: Vec<&FoundationTrainingRecord> =
            chunk.iter().map(|&index| &records[index]).collect();
        let packed = collate_records(&selected, causal_collator, spectrum_collator, device)?;
        let output = model.forward_t(
            &packed.causal.input,
            &packed.spectrum,
            &packed.precursor,
            false,
        )?;
        let loss = foundation_causal_next_token_loss(&output, &packed.causal)?;
        let chunk_active = packed.causal.active_indices.dims1()?;
        loss_sum += f64::from(loss.to_scalar::<f32>()?) * chunk_active as f64;

        let logits = output.token_logits.to_vec3::<f32>()?;
        let targets = packed.causal.target_tokens.to_vec2::<u32>()?;
        let mask = packed.causal.input.token_mask.to_vec2::<f32>()?;
        for row in 0..selected.len() {
            let mut exact = true;
            let mut eos_seen = false;
            for position in 0..targets[row].len() {
                if mask[row][position] <= 0.0 {
                    break;
                }
                let predicted = argmax(&logits[row][position]) as u32;
                let target = targets[row][position];
                correct_tokens += usize::from(predicted == target);
                active_tokens += 1;
                if target == FOUNDATION_DIFFUSION_EOS {
                    correct_eos += usize::from(predicted == target);
                    eos_seen = true;
                }
                if predicted != target {
                    exact = false;
                }
            }
            if !eos_seen {
                anyhow::bail!("causal validation target unexpectedly lacks EOS");
            }
            exact_sequences += usize::from(exact);
            sequences += 1;
        }
    }
    if active_tokens == 0 || sequences == 0 {
        anyhow::bail!("causal validation produced no active tokens");
    }
    let mean_loss = loss_sum / active_tokens as f64;
    Ok(CausalMetrics {
        loss: mean_loss,
        perplexity: mean_loss.exp(),
        token_accuracy: correct_tokens as f64 / active_tokens as f64,
        eos_accuracy: correct_eos as f64 / sequences as f64,
        exact_sequence_rate: exact_sequences as f64 / sequences as f64,
        active_tokens,
        sequences,
    })
}

fn println_metrics(label: &str, step: usize, metrics: CausalMetrics) {
    println!(
        "{label}\tstep={step}\tloss={:.6}\tperplexity={:.4}\ttoken_accuracy={:.6}\teos_accuracy={:.6}\texact_sequence_rate={:.6}\tactive_tokens={}\tsequences={}",
        metrics.loss,
        metrics.perplexity,
        metrics.token_accuracy,
        metrics.eos_accuracy,
        metrics.exact_sequence_rate,
        metrics.active_tokens,
        metrics.sequences
    );
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_metadata(
    config: &FoundationDiffusionConfig,
    corpus: &redeem_properties::foundation::FoundationCorpus,
    benchmark: &FoundationBenchmarkManifest,
    train_fingerprint: u64,
    validation_fingerprint: u64,
    usable_train_pairs: usize,
    usable_validation_pairs: usize,
    train_steps: usize,
    batch_size: usize,
    validation_batches: usize,
    seed: u64,
    learning_rate: f64,
    max_gradient_norm: f64,
    warm_start: &Path,
    global_step: usize,
    best_validation_loss: f64,
) -> CausalCheckpointMetadata {
    CausalCheckpointMetadata {
        version: 1,
        objective: "causal_next_token_ce_v1".into(),
        corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
        benchmark_manifest_fingerprint: format!(
            "fnv1a64:{:016x}",
            benchmark.manifest_fingerprint()
        ),
        train_diffusion_fingerprint: format!("fnv1a64:{train_fingerprint:016x}"),
        validation_diffusion_fingerprint: format!("fnv1a64:{validation_fingerprint:016x}"),
        usable_train_pairs,
        usable_validation_pairs,
        train_steps,
        batch_size,
        validation_batches,
        seed,
        learning_rate,
        max_gradient_norm,
        warm_start_diffusion_checkpoint: warm_start.display().to_string(),
        global_step,
        best_validation_loss,
        diffusion: config.clone(),
    }
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    metadata: &CausalCheckpointMetadata,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    Ok(())
}

fn resolve_model_safetensors(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_path_buf()
    }
}

fn deterministic_batch(indices: &[usize], batch_size: usize, seed: u64) -> Vec<usize> {
    let mut rng = CausalRng::new(seed);
    (0..batch_size)
        .map(|_| indices[rng.next_u64() as usize % indices.len()])
        .collect()
}

fn deterministic_subset(indices: &[usize], requested: usize, seed: u64) -> Vec<usize> {
    let mut ranked: Vec<(u64, usize)> = indices
        .iter()
        .copied()
        .map(|index| (mix64(seed ^ index as u64), index))
        .collect();
    ranked.sort_unstable();
    ranked
        .into_iter()
        .take(requested.min(indices.len()))
        .map(|(_, index)| index)
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn parse_or<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    args.get(index)
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|error| anyhow::anyhow!("failed to parse argument {index}: {error}"))
        })
        .unwrap_or(Ok(default))
}

#[derive(Debug, Clone, Copy)]
struct CausalRng {
    state: u64,
}

impl CausalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xa076_1d64_78bd_642f,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

// -----------------------------------------------------------------------------
// v0.21 chemistry-conditioned whole-sequence diffusion / iterative refinement.
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Default)]
struct ChemistryDiffusionMetricsV0210 {
    timestep: usize,
    matched_nll: f64,
    shuffled_nll: f64,
    conditioning_gap: f64,
    token_accuracy: f64,
    exact_sequence_rate: f64,
    active_tokens: usize,
    sequences: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ChemistryDiffusionCheckpointV0210<'a> {
    version: &'a str,
    objective: &'a str,
    architecture: &'a str,
    run_mode: &'a str,
    test_partition_consumed: bool,
    global_step: usize,
    fixed_refinement_steps: usize,
    refinement_timesteps: Vec<usize>,
    fixed_initial_beam_width: usize,
    corruption_schedule: &'a str,
    initialization: &'a str,
    final_mass_policy: &'a str,
    validation_selection_timestep: usize,
    validation: ChemistryDiffusionMetricsV0210,
    v0200_checkpoint: String,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Default)]
struct RefinementMetricsV0210 {
    records: usize,
    initialized_records: usize,
    zero_initial_records: usize,
    initial_literal_top1: usize,
    initial_sequence_top1: usize,
    initial_il_top1: usize,
    final_literal_top1: usize,
    final_sequence_top1: usize,
    final_il_top1: usize,
    initial_mass_error_abs_sum: f64,
    pre_projection_mass_error_abs_sum: f64,
    final_mass_error_abs_sum: f64,
    initial_mass_error_records: usize,
    pre_projection_mass_error_records: usize,
    final_mass_error_records: usize,
    pre_projection_mass_valid: usize,
    final_mass_valid: usize,
    changed_positions: usize,
    compared_positions: usize,
    initially_incorrect_positions: usize,
    incorrect_positions_corrected: usize,
    initially_correct_positions: usize,
    correct_positions_damaged: usize,
    per_step_changed_positions: Vec<usize>,
    per_step_converged_records: Vec<usize>,
    mass_projection_used: usize,
    mass_projection_failed: usize,
}

impl RefinementMetricsV0210 {
    fn print(&self) {
        println!("final_records\t{}", self.records);
        println!("v0200_initialized_records\t{}", self.initialized_records);
        println!("v0200_zero_initial_records\t{}", self.zero_initial_records);
        println!("initial_literal_top1\t{}", self.initial_literal_top1);
        println!("initial_sequence_top1\t{}", self.initial_sequence_top1);
        println!("initial_il_top1\t{}", self.initial_il_top1);
        println!("final_literal_top1\t{}", self.final_literal_top1);
        println!("final_sequence_top1\t{}", self.final_sequence_top1);
        println!("final_il_top1\t{}", self.final_il_top1);
        println!(
            "initial_mean_abs_precursor_mass_error_da\t{:.8}",
            if self.initial_mass_error_records == 0 {
                f64::NAN
            } else {
                self.initial_mass_error_abs_sum / self.initial_mass_error_records as f64
            }
        );
        println!(
            "pre_projection_mean_abs_precursor_mass_error_da\t{:.8}",
            if self.pre_projection_mass_error_records == 0 {
                f64::NAN
            } else {
                self.pre_projection_mass_error_abs_sum
                    / self.pre_projection_mass_error_records as f64
            }
        );
        println!(
            "final_mean_abs_precursor_mass_error_da\t{:.8}",
            if self.final_mass_error_records == 0 {
                f64::NAN
            } else {
                self.final_mass_error_abs_sum / self.final_mass_error_records as f64
            }
        );
        println!(
            "pre_projection_mass_valid_records\t{}",
            self.pre_projection_mass_valid
        );
        println!("final_mass_valid_records\t{}", self.final_mass_valid);
        println!(
            "final_mass_valid_fraction\t{:.8}",
            if self.records == 0 {
                0.0
            } else {
                self.final_mass_valid as f64 / self.records as f64
            }
        );
        println!("positions_compared_to_v0200\t{}", self.compared_positions);
        println!("positions_changed_from_v0200\t{}", self.changed_positions);
        println!(
            "position_change_fraction_from_v0200\t{:.8}",
            if self.compared_positions == 0 {
                0.0
            } else {
                self.changed_positions as f64 / self.compared_positions as f64
            }
        );
        println!(
            "initially_incorrect_positions\t{}",
            self.initially_incorrect_positions
        );
        println!(
            "incorrect_v0200_positions_corrected\t{}",
            self.incorrect_positions_corrected
        );
        println!(
            "incorrect_position_correction_fraction\t{:.8}",
            if self.initially_incorrect_positions == 0 {
                0.0
            } else {
                self.incorrect_positions_corrected as f64
                    / self.initially_incorrect_positions as f64
            }
        );
        println!(
            "initially_correct_positions\t{}",
            self.initially_correct_positions
        );
        println!(
            "correct_v0200_positions_damaged\t{}",
            self.correct_positions_damaged
        );
        println!(
            "correct_position_damage_fraction\t{:.8}",
            if self.initially_correct_positions == 0 {
                0.0
            } else {
                self.correct_positions_damaged as f64 / self.initially_correct_positions as f64
            }
        );
        println!("final_mass_projection_used\t{}", self.mass_projection_used);
        println!(
            "final_mass_projection_failed\t{}",
            self.mass_projection_failed
        );
        for (step, (&changed, &converged)) in self
            .per_step_changed_positions
            .iter()
            .zip(&self.per_step_converged_records)
            .enumerate()
        {
            println!(
                "refinement_step\tstep={}\tpositions_changed={}\tconverged_records={}",
                step + 1,
                changed,
                converged
            );
        }
    }
}

/// v0.21 fixed first experiment.
///
/// The TRAIN objective is categorical x0 reconstruction from one fixed
/// multinomial schedule plus the already-validated matched-vs-shuffled spectrum
/// guard. Frozen validation initialization comes only from the trained v0.20
/// width-128 chemistry decoder. TEST is never opened.
pub(crate) fn v0210_main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_train_chemistry_diffusion_v0210 RUN.yaml V0200_CHECKPOINT VALIDATION_CANDIDATES.tsv OUTPUT_DIR [full|smoke]"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let v0200_checkpoint = PathBuf::from(&args[2]);
    let validation_candidates = PathBuf::from(&args[3]);
    let output_root = PathBuf::from(&args[4]);
    let mode = args.get(5).map(String::as_str).unwrap_or("full");
    let (train_steps, batch_size, validation_limit) = match mode {
        "full" => (4_000usize, 32usize, None),
        // Technical smoke is intentionally tiny. One optimizer update and two
        // validation records exercise matched/shuffled training plumbing; the
        // frozen full scientific protocol remains unchanged.
        "smoke" => (1usize, 2usize, Some(2usize)),
        other => anyhow::bail!("unsupported v0.21 mode {other:?}; expected full or smoke"),
    };
    let runtime_started = Instant::now();
    v0210_stage("start", &runtime_started);
    for path in [
        training_yaml.as_path(),
        v0200_checkpoint.as_path(),
        validation_candidates.as_path(),
        output_root.as_path(),
    ] {
        reject_test_path_v0190(path)?;
    }

    let run = read_foundation_training_run_config(&training_yaml)?;
    v0210_stage("run_config_loaded", &runtime_started);
    let corpus = load_foundation_corpus(&run.corpus)?;
    v0210_stage("corpus_loaded", &runtime_started);
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    v0210_stage("benchmark_validated", &runtime_started);

    let v0200_metadata_path = if v0200_checkpoint.is_dir() {
        v0200_checkpoint.join("metadata.yaml")
    } else {
        v0200_checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    };
    let v0200_metadata: UnifiedV0190Metadata = serde_yaml::from_str(
        &fs::read_to_string(&v0200_metadata_path)
            .with_context(|| format!("read v0.20 metadata {v0200_metadata_path:?}"))?,
    )?;
    let mut config = v0200_metadata.inverse_config.clone();
    // Freeze one schedule and width for the first v0.21 experiment regardless
    // of incidental historical metadata drift.
    config.model_dim = 96;
    config.diffusion_steps = 20;
    config.beta_start = 0.02;
    config.beta_end = 0.35;
    config.validate().map_err(anyhow::Error::msg)?;

    let vocabulary = FoundationDiffusionVocabulary;
    let mut train_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &config,
        vocabulary,
    );
    train_indices.retain(|&index| v0200_precursor_mass(&corpus.records[index]).is_ok());
    let validation_cohort = read_frozen_validation_cohort_v0190(
        &validation_candidates,
        &corpus.records,
        &benchmark,
        &config,
        vocabulary,
    )?;
    let mut validation_indices = validation_cohort.indices.clone();
    if let Some(limit) = validation_limit {
        validation_indices.truncate(limit);
    }
    if train_indices.len() < batch_size || validation_indices.is_empty() {
        anyhow::bail!(
            "insufficient v0.21 records: train={} validation={} batch={batch_size}",
            train_indices.len(),
            validation_indices.len()
        );
    }

    // Runtime partition isolation must be O(records + manifest entries). The
    // first v0.21 implementation accidentally performed a full manifest scan
    // for every TRAIN record (O(N^2)), which made technical smoke CPU-bound for
    // hours before model construction. Materialize one direct lookup table and
    // verify membership without ever materializing the TEST record set.
    let mut partition_by_record = vec![None; corpus.records.len()];
    for entry in &benchmark.entries {
        let slot = partition_by_record
            .get_mut(entry.record_index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "benchmark record index {} outside corpus length {}",
                    entry.record_index,
                    corpus.records.len()
                )
            })?;
        if slot.replace(entry.partition).is_some() {
            anyhow::bail!(
                "benchmark contains duplicate record index {}",
                entry.record_index
            );
        }
    }
    let train_isolated = train_indices.iter().all(|&index| {
        partition_by_record.get(index).copied().flatten() == Some(FoundationPartition::Train)
    });
    let validation_isolated = validation_cohort.indices.iter().all(|&index| {
        partition_by_record.get(index).copied().flatten() == Some(FoundationPartition::Validation)
    });
    if !train_isolated || !validation_isolated {
        anyhow::bail!("v0.21 partition isolation failed before model construction");
    }
    drop(partition_by_record);
    v0210_stage("partitions_and_frozen_cohort_ready", &runtime_started);

    fs::create_dir_all(&output_root)?;
    let device = Device::cuda_if_available(0)?;
    let diffusion_collator = FoundationDiffusionCollator::new(config.clone())?;
    let causal_collator = FoundationCausalCollator::new(config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;
    let chemistry_diffusion =
        ChemistryDiffusionFeaturizer::new(&config).map_err(anyhow::Error::msg)?;

    // v0.20 autoregressive initialization is needed only for the final
    // refinement exercise. Construct its suffix lattice/model lazily after
    // denoising training/telemetry, so technical smoke startup cannot be
    // dominated by unrelated AR preprocessing.
    let v0200_model_path = resolve_model_safetensors(&v0200_checkpoint);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumChemistryDiffusionModel::new(config.clone(), vb)?;
    let warm = load_chemistry_diffusion_from_v0200_checkpoint(&varmap, &v0200_model_path, &device)?;
    if warm.chemistry_variables != 4
        || warm.new_diffusion_variables != 4
        || warm.new_full_state_variables != 4
        || warm.ignored_v0200_variables != 1
    {
        anyhow::bail!(
            "v0.21 warm-start namespace contract drift: chemistry={} new_diffusion={} new_full_state={} ignored_v0200={} (expected 4/4/4/1)",
            warm.chemistry_variables,
            warm.new_diffusion_variables,
            warm.new_full_state_variables,
            warm.ignored_v0200_variables,
        );
    }

    v0210_stage("diffusion_model_warm_started", &runtime_started);

    let refinement_timesteps =
        foundation_chemistry_diffusion_refinement_timesteps(config.diffusion_steps)
            .map_err(anyhow::Error::msg)?;
    let refinement_start_cumulative_corruption = 1.0
        - config
            .alpha_bar(FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210)
            .map_err(anyhow::Error::msg)?;
    println!("version\tv0.21.0");
    println!("architecture\t{FOUNDATION_CHEMISTRY_DIFFUSION_ARCHITECTURE_V0210}");
    println!("objective\t{FOUNDATION_CHEMISTRY_DIFFUSION_OBJECTIVE_V0210}");
    println!("scientific_change\twhole_sequence_bidirectional_revision_of_v0200_hypotheses");
    println!("autoregressive_family\tCLOSED_INITIALIZATION_ONLY");
    println!("initialization\tfrozen_v0200_width128_top1");
    println!("initial_beam_width\t{FOUNDATION_CHEMISTRY_DIFFUSION_INITIAL_BEAM_WIDTH_V0210}");
    println!("diffusion_steps\t{}", config.diffusion_steps);
    println!("beta_start\t{:.6}", config.beta_start);
    println!("beta_end\t{:.6}", config.beta_end);
    println!("refinement_steps\t{FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_STEPS_V0210}");
    println!("refinement_timesteps\t{:?}", refinement_timesteps);
    println!(
        "refinement_start_timestep\t{}",
        FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210
    );
    println!("refinement_start_cumulative_corruption\t{refinement_start_cumulative_corruption:.6}");
    println!("sequence_length_policy\tpreserve_v0200_initial_active_length");
    println!("early_mid_mass_policy\tsoft_conditioning_only");
    println!("final_mass_policy\thard_validity_plus_fixed_hamming_radius2_projection_then_v0200_valid_fallback");
    println!("fragment_peaks_define_state_existence\tfalse");
    println!("fragment_peaks_are_soft_evidence\ttrue");
    println!("model_dim\t{}", config.model_dim);
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("learning_rate\t0.0001");
    println!("weight_decay\t0.0001");
    println!("max_gradient_norm\t5");
    println!("training_seed\t20260921");
    println!("device\t{device:?}");
    println!("warm_shared_variables\t{}", warm.shared_variables);
    println!("warm_chemistry_variables\t{}", warm.chemistry_variables);
    println!("new_diffusion_variables\t{}", warm.new_diffusion_variables);
    println!(
        "new_full_state_variables\t{}",
        warm.new_full_state_variables
    );
    println!("ignored_v0200_variables\t{}", warm.ignored_v0200_variables);
    println!("validation_cohort_policy\tfrozen_v01323_mass_valid_record_ids_only");
    println!("validation_records\t{}", validation_indices.len());
    println!("frozen_v0200_reference\ttop1_literal=7\ttop1_il=12\ttop5_literal=12\ttop5_il=20\ttop10_literal=18\ttop10_il=24\ttop32_literal=24\ttop32_il=28\ttop128_literal=27\ttop128_il=33");
    println!("accepted_v01323_baseline\ttop1_literal=24\ttop1_il=38");
    println!("progress_target\ttop1_literal=28\ttop1_il=42");
    println!("final_candidate_policy\tone_refined_hypothesis_per_initialized_spectrum");
    println!("test_partition_consumed\tfalse");
    println!("mode\t{mode}");

    let validation_timestep = 8usize;
    let initial_metrics = evaluate_chemistry_diffusion_v0210(
        &model,
        &corpus.records,
        &validation_indices,
        batch_size,
        validation_timestep,
        &diffusion_collator,
        &spectrum_collator,
        &chemistry_diffusion,
        &device,
        20_260_921,
    )?;
    print_chemistry_diffusion_v0210("initial_validation", 0, initial_metrics);
    save_v0210_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &config,
        mode,
        0,
        validation_timestep,
        initial_metrics,
        &v0200_model_path,
    )?;
    save_v0210_checkpoint(
        &output_root.join("best"),
        &varmap,
        &config,
        mode,
        0,
        validation_timestep,
        initial_metrics,
        &v0200_model_path,
    )?;
    v0210_stage(
        "initial_validation_and_checkpoint_complete",
        &runtime_started,
    );
    let mut best_step = 0usize;
    let mut best_metrics = initial_metrics;
    let mut best_objective = v0210_validation_objective(initial_metrics);

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: 1.0e-4,
            weight_decay: 1.0e-4,
            ..FoundationAdamWConfig::default()
        },
    )?;

    for step in 1..=train_steps {
        let selected = deterministic_batch(
            &train_indices,
            batch_size,
            20_260_921 ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let records = selected
            .iter()
            .map(|&i| &corpus.records[i])
            .collect::<Vec<_>>();
        let components = v0200_batch_components(&records)?;
        let diffusion = diffusion_collator.collate_random_timesteps(
            &components.peptides,
            20_260_921 ^ step as u64,
            &device,
        )?;
        let spectrum = spectrum_collator.collate(&components.spectra, &device)?;
        let precursor = precursor_context(&records, &device)?;
        let chemistry = chemistry_diffusion.featurize(
            &diffusion.noisy_token_rows,
            &diffusion.active_lengths,
            &components.spectra,
            &components.precursor_masses,
            &components.charges,
            &device,
        )?;
        let matched_output =
            model.forward_t(&diffusion, &spectrum, &precursor, &chemistry, true)?;
        let matched = foundation_diffusion_x0_loss(&matched_output, &diffusion)?;

        let order = foundation_direct_shuffled_order(records.len(), 20_260_921 ^ step as u64)?;
        let shuffled_spectra = order
            .iter()
            .map(|&i| components.spectra[i].clone())
            .collect::<Vec<_>>();
        let shuffled_spectrum = spectrum_collator.collate(&shuffled_spectra, &device)?;
        let shuffled_chemistry = chemistry_diffusion.featurize(
            &diffusion.noisy_token_rows,
            &diffusion.active_lengths,
            &shuffled_spectra,
            &components.precursor_masses,
            &components.charges,
            &device,
        )?;
        let shuffled_output = model.forward_t(
            &diffusion,
            &shuffled_spectrum,
            &precursor,
            &shuffled_chemistry,
            true,
        )?;
        let shuffled = foundation_diffusion_x0_loss(&shuffled_output, &diffusion)?;
        let loss = foundation_direct_conditioning_loss(
            &matched,
            &shuffled,
            FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
            FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
        )?;
        let matched_value = f64::from(matched.to_scalar::<f32>()?);
        let shuffled_value = f64::from(shuffled.to_scalar::<f32>()?);
        let objective_value = f64::from(loss.to_scalar::<f32>()?);
        let update = optimizer.backward_step(&loss, Some(5.0))?;
        if step == 1 || step % 25 == 0 || step == train_steps {
            println!(
                "train\tstep={step}\tobjective={objective_value:.6}\tmatched_nll={matched_value:.6}\tshuffled_nll={shuffled_value:.6}\tconditioning_gap={:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                shuffled_value - matched_value,
                update.gradient_norm,
                update.gradient_scale
            );
        }
        if step % 100 == 0 || step == train_steps {
            let metrics = evaluate_chemistry_diffusion_v0210(
                &model,
                &corpus.records,
                &validation_indices,
                batch_size,
                validation_timestep,
                &diffusion_collator,
                &spectrum_collator,
                &chemistry_diffusion,
                &device,
                20_260_921 ^ step as u64,
            )?;
            print_chemistry_diffusion_v0210("validation", step, metrics);
            save_v0210_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &config,
                mode,
                step,
                validation_timestep,
                metrics,
                &v0200_model_path,
            )?;
            let objective = v0210_validation_objective(metrics);
            if objective < best_objective
                && metrics.conditioning_gap >= FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190
            {
                best_objective = objective;
                best_metrics = metrics;
                best_step = step;
                save_v0210_checkpoint(
                    &output_root.join("best"),
                    &varmap,
                    &config,
                    mode,
                    step,
                    validation_timestep,
                    metrics,
                    &v0200_model_path,
                )?;
            }
        }
    }

    v0210_stage("optimizer_loop_complete", &runtime_started);
    load_varmap_checkpoint_v0210(
        &varmap,
        &output_root.join("best/model.safetensors"),
        &device,
    )?;
    println!("best_step\t{best_step}");
    print_chemistry_diffusion_v0210("final_best_validation", best_step, best_metrics);

    // Full scientific telemetry remains the frozen six-timestep curve. The
    // technical smoke exercises only the actual inference start level t=8.
    let reconstruction_timesteps: &[usize] = if mode == "smoke" {
        &[FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210]
    } else {
        &[1usize, 5, 8, 10, 15, 20]
    };
    for &timestep in reconstruction_timesteps {
        let metrics = evaluate_chemistry_diffusion_v0210(
            &model,
            &corpus.records,
            &validation_indices,
            batch_size,
            timestep,
            &diffusion_collator,
            &spectrum_collator,
            &chemistry_diffusion,
            &device,
            20_260_921 ^ 0x5a5a_0000 ^ timestep as u64,
        )?;
        print_chemistry_diffusion_v0210("reconstruction_by_timestep", best_step, metrics);
    }
    v0210_stage("reconstruction_telemetry_complete", &runtime_started);

    // Exercise the expensive v0.20 initializer on exactly one frozen record in
    // smoke mode. Full mode still evaluates all 125 records with all eight
    // refinement passes.
    let refinement_indices: Vec<usize> = if mode == "smoke" {
        validation_indices.iter().copied().take(1).collect()
    } else {
        validation_indices.clone()
    };
    let refinement_timesteps_eval: Vec<usize> = if mode == "smoke" {
        vec![FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210]
    } else {
        refinement_timesteps.clone()
    };
    v0210_stage("v0200_initializer_setup_begin", &runtime_started);
    let max_refinement_mass = refinement_indices
        .iter()
        .map(|&index| v0200_precursor_mass(&corpus.records[index]))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(0.0f64, f64::max);
    let initial_suffix_lattice =
        ChemistrySuffixMassLattice::new(config.max_tokens, max_refinement_mass + 100.0)
            .map_err(anyhow::Error::msg)?;
    let v0200_featurizer = ChemistryTransitionFeaturizer::new(&config, initial_suffix_lattice)
        .map_err(anyhow::Error::msg)?;
    let mut v0200_varmap = VarMap::new();
    let v0200_vb = VarBuilder::from_varmap(&v0200_varmap, DType::F32, &device);
    let v0200_model = PeptideSpectrumChemistryDecoder::new(config.clone(), v0200_vb)?;
    v0200_varmap.load(&v0200_model_path).with_context(|| {
        format!(
            "load frozen trained v0.20 initialization checkpoint {}",
            v0200_model_path.display()
        )
    })?;
    v0210_stage("v0200_initializer_setup_complete", &runtime_started);

    let refinement = evaluate_refinement_v0210(
        &model,
        &v0200_model,
        &corpus.records,
        &refinement_indices,
        &refinement_timesteps_eval,
        &causal_collator,
        &diffusion_collator,
        &spectrum_collator,
        &v0200_featurizer,
        &chemistry_diffusion,
        &config,
        &device,
    )?;
    v0210_stage("refinement_runtime_exercise_complete", &runtime_started);
    refinement.print();

    if mode == "full" {
        if refinement.initial_literal_top1 != 7
            || refinement.initial_sequence_top1 != 7
            || refinement.initial_il_top1 != 12
            || refinement.initialized_records != 118
            || refinement.zero_initial_records != 7
        {
            anyhow::bail!(
                "v0.21 frozen v0.20 initialization drift: observed literal/sequence/I-L={}/{}/{} initialized/zero={}/{}; expected 7/7/12 and 118/7",
                refinement.initial_literal_top1,
                refinement.initial_sequence_top1,
                refinement.initial_il_top1,
                refinement.initialized_records,
                refinement.zero_initial_records,
            );
        }
        // Predeclare "material" as lifting top-1 to at least the frozen v0.20
        // top-5 recovery in both literal and I/L space (12/20). This avoids
        // granting another architecture iteration for a marginal fluctuation.
        let materially_improved =
            refinement.final_literal_top1 >= 12 && refinement.final_il_top1 >= 20;
        let baseline_recovered =
            refinement.final_literal_top1 >= 24 && refinement.final_il_top1 >= 38;
        let progress = refinement.final_literal_top1 >= 28 && refinement.final_il_top1 >= 42;
        println!(
            "baseline_recovery_gate\t{}",
            if baseline_recovered { "PASS" } else { "FAIL" }
        );
        println!("progress_gate\t{}", if progress { "PASS" } else { "FAIL" });
        println!("material_improvement_threshold\tliteral>=12_and_il>=20");
        println!(
            "material_improvement_over_v0200\t{}",
            if materially_improved { "YES" } else { "NO" }
        );
        let decision = if progress {
            "PROGRESS_TARGET_MET_ADVANCE_FROM_V0210"
        } else if baseline_recovered {
            "BASELINE_RECOVERED_V0210_IS_VIABLE"
        } else if materially_improved {
            "PARTIAL_SUCCESS_ALLOW_AT_MOST_ONE_COHERENT_ARCHITECTURAL_CORRECTION"
        } else {
            "STOP_LOCAL_DIFFUSION_REASSESS_INVERSE_GENERATION_FORMULATION"
        };
        println!("v0210_stop_rule\t{decision}");
    } else {
        println!("v0210_stop_rule\tSMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION");
    }
    println!("test_partition_consumed\tfalse");
    v0210_stage("complete", &runtime_started);
    Ok(())
}

fn v0210_stage(stage: &str, started: &Instant) {
    eprintln!(
        "v0210_stage\tstage={stage}\telapsed_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    let _ = std::io::stderr().flush();
}

fn v0210_validation_objective(metrics: ChemistryDiffusionMetricsV0210) -> f64 {
    metrics.matched_nll
        + FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190
            * (FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190 - metrics.conditioning_gap).max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_chemistry_diffusion_v0210(
    model: &PeptideSpectrumChemistryDiffusionModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    timestep: usize,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    chemistry: &ChemistryDiffusionFeaturizer,
    device: &Device,
    seed: u64,
) -> Result<ChemistryDiffusionMetricsV0210> {
    let mut matched_weighted = 0.0f64;
    let mut shuffled_weighted = 0.0f64;
    let mut active_tokens = 0usize;
    let mut correct_tokens = 0usize;
    let mut exact_sequences = 0usize;
    let mut sequences = 0usize;
    for (chunk_index, chunk) in indices.chunks(batch_size).enumerate() {
        if chunk.len() < 2 {
            continue;
        }
        let selected = chunk.iter().map(|&i| &records[i]).collect::<Vec<_>>();
        let components = v0200_batch_components(&selected)?;
        let timesteps = vec![timestep; selected.len()];
        let diffusion = diffusion_collator.collate(
            &components.peptides,
            &timesteps,
            seed ^ chunk_index as u64,
            device,
        )?;
        let spectrum = spectrum_collator.collate(&components.spectra, device)?;
        let precursor = precursor_context(&selected, device)?;
        let matched_chemistry = chemistry.featurize(
            &diffusion.noisy_token_rows,
            &diffusion.active_lengths,
            &components.spectra,
            &components.precursor_masses,
            &components.charges,
            device,
        )?;
        let matched_output =
            model.forward_t(&diffusion, &spectrum, &precursor, &matched_chemistry, false)?;
        let matched = foundation_diffusion_x0_loss(&matched_output, &diffusion)?;

        let order =
            foundation_direct_shuffled_order(selected.len(), seed ^ 0x7777 ^ chunk_index as u64)?;
        let shuffled_spectra = order
            .iter()
            .map(|&i| components.spectra[i].clone())
            .collect::<Vec<_>>();
        let shuffled_spectrum = spectrum_collator.collate(&shuffled_spectra, device)?;
        let shuffled_chemistry = chemistry.featurize(
            &diffusion.noisy_token_rows,
            &diffusion.active_lengths,
            &shuffled_spectra,
            &components.precursor_masses,
            &components.charges,
            device,
        )?;
        let shuffled_output = model.forward_t(
            &diffusion,
            &shuffled_spectrum,
            &precursor,
            &shuffled_chemistry,
            false,
        )?;
        let shuffled = foundation_diffusion_x0_loss(&shuffled_output, &diffusion)?;
        let chunk_active = diffusion.active_indices.dims1()?;
        matched_weighted += f64::from(matched.to_scalar::<f32>()?) * chunk_active as f64;
        shuffled_weighted += f64::from(shuffled.to_scalar::<f32>()?) * chunk_active as f64;

        let logits = matched_output.token_logits.to_vec3::<f32>()?;
        let targets = diffusion.clean_tokens.to_vec2::<u32>()?;
        for row in 0..selected.len() {
            let mut exact = true;
            for position in 0..diffusion.active_lengths[row] {
                let predicted = argmax(&logits[row][position]) as u32;
                let target = targets[row][position];
                correct_tokens += usize::from(predicted == target);
                active_tokens += 1;
                exact &= predicted == target;
            }
            exact_sequences += usize::from(exact);
            sequences += 1;
        }
    }
    if active_tokens == 0 || sequences == 0 {
        anyhow::bail!("v0.21 denoising validation produced no active tokens");
    }
    let matched_nll = matched_weighted / active_tokens as f64;
    let shuffled_nll = shuffled_weighted / active_tokens as f64;
    Ok(ChemistryDiffusionMetricsV0210 {
        timestep,
        matched_nll,
        shuffled_nll,
        conditioning_gap: shuffled_nll - matched_nll,
        token_accuracy: correct_tokens as f64 / active_tokens as f64,
        exact_sequence_rate: exact_sequences as f64 / sequences as f64,
        active_tokens,
        sequences,
    })
}

fn print_chemistry_diffusion_v0210(
    label: &str,
    step: usize,
    metrics: ChemistryDiffusionMetricsV0210,
) {
    println!(
        "{label}\tstep={step}\ttimestep={}\tmatched_spectrum_token_nll={:.6}\tshuffled_spectrum_token_nll={:.6}\tconditioning_gap={:.6}\ttoken_reconstruction_accuracy={:.8}\tcomplete_sequence_reconstruction_rate={:.8}\tactive_tokens={}\tsequences={}",
        metrics.timestep,
        metrics.matched_nll,
        metrics.shuffled_nll,
        metrics.conditioning_gap,
        metrics.token_accuracy,
        metrics.exact_sequence_rate,
        metrics.active_tokens,
        metrics.sequences,
    );
}

fn save_v0210_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    config: &FoundationDiffusionConfig,
    mode: &str,
    step: usize,
    validation_timestep: usize,
    validation: ChemistryDiffusionMetricsV0210,
    v0200_checkpoint: &Path,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    let metadata = ChemistryDiffusionCheckpointV0210 {
        version: "v0.21.0",
        objective: FOUNDATION_CHEMISTRY_DIFFUSION_OBJECTIVE_V0210,
        architecture: FOUNDATION_CHEMISTRY_DIFFUSION_ARCHITECTURE_V0210,
        run_mode: mode,
        test_partition_consumed: false,
        global_step: step,
        fixed_refinement_steps: FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_STEPS_V0210,
        refinement_timesteps: foundation_chemistry_diffusion_refinement_timesteps(
            config.diffusion_steps,
        )
        .map_err(anyhow::Error::msg)?,
        fixed_initial_beam_width: FOUNDATION_CHEMISTRY_DIFFUSION_INITIAL_BEAM_WIDTH_V0210,
        corruption_schedule: "linear_beta_0.02_to_0.35_over_20_steps",
        initialization: "frozen_v0200_width128_top1_preserve_length",
        final_mass_policy: "hard_exact_precursor_mass_with_fixed_hamming_radius2_projection_then_valid_v0200_fallback",
        validation_selection_timestep: validation_timestep,
        validation,
        v0200_checkpoint: v0200_checkpoint.display().to_string(),
        inverse_config: config.clone(),
    };
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(&metadata)?,
    )?;
    Ok(())
}

fn load_varmap_checkpoint_v0210(varmap: &VarMap, path: &Path, device: &Device) -> Result<()> {
    let checkpoint = candle_core::safetensors::load(path, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.21 VarMap lock poisoned"))?;
    for (name, variable) in data.iter() {
        variable.set(
            checkpoint
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("v0.21 checkpoint missing {name}"))?,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_refinement_v0210(
    model: &PeptideSpectrumChemistryDiffusionModel,
    v0200_model: &PeptideSpectrumChemistryDecoder,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    timesteps: &[usize],
    causal_collator: &FoundationCausalCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    v0200_featurizer: &ChemistryTransitionFeaturizer,
    chemistry: &ChemistryDiffusionFeaturizer,
    config: &FoundationDiffusionConfig,
    device: &Device,
) -> Result<RefinementMetricsV0210> {
    let vocabulary = FoundationDiffusionVocabulary;
    if timesteps.is_empty()
        || timesteps
            .iter()
            .any(|&timestep| timestep == 0 || timestep > config.diffusion_steps)
    {
        anyhow::bail!("v0.21 refinement runtime received invalid timestep schedule");
    }
    let mut metrics = RefinementMetricsV0210 {
        per_step_changed_positions: vec![0; timesteps.len()],
        per_step_converged_records: vec![0; timesteps.len()],
        ..RefinementMetricsV0210::default()
    };

    for (record_ordinal, &index) in indices.iter().enumerate() {
        if record_ordinal == 0 || record_ordinal % 10 == 0 {
            eprintln!(
                "v0210_refinement_progress\trecord={}/{}",
                record_ordinal + 1,
                indices.len()
            );
            let _ = std::io::stderr().flush();
        }
        metrics.records += 1;
        let record = &records[index];
        let precursor_mass = v0200_precursor_mass(record)?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("v0.21 validation record {index} lacks charge"))?;
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("v0.21 validation record {index} lacks spectrum"))?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
        let precursor = precursor_context(&[record], device)?;
        let context = v0200_model.prepare_context(&spectrum_batch, &precursor, false)?;
        let candidates = foundation_direct_beam_search(
            precursor_mass,
            DirectDecoderBeamConfig {
                beam_width: FOUNDATION_CHEMISTRY_DIFFUSION_INITIAL_BEAM_WIDTH_V0210,
                top_k: 1,
                mass_tolerance_da: config.precursor_mass_tolerance_da,
                max_tokens: config.max_tokens,
            },
            |prefixes| {
                let input = causal_collator
                    .collate_compact_prefix_rows(prefixes, device)
                    .map_err(|error| error.to_string())?;
                let candidate_chemistry = v0200_featurizer
                    .next_prefixes(prefixes, &spectrum, precursor_mass, charge, device)
                    .map_err(|error| error.to_string())?;
                let mut rows = v0200_model
                    .forward_next_t_with_context(&input, &context, &candidate_chemistry, false)
                    .and_then(|tensor| tensor.to_vec2::<f32>())
                    .map_err(|error| error.to_string())?;
                v0200_featurizer.mask_infeasible_next_logits(
                    prefixes,
                    &mut rows,
                    precursor_mass,
                )?;
                Ok(rows)
            },
        )
        .map_err(anyhow::Error::msg)?;
        let Some(initial_candidate) = candidates.first() else {
            metrics.zero_initial_records += 1;
            continue;
        };
        metrics.initialized_records += 1;
        let mut initial = vec![FOUNDATION_DIFFUSION_PAD; config.max_tokens];
        if initial_candidate.tokens.len() > config.max_tokens {
            anyhow::bail!("v0.21 v0.20 initialization exceeds configured token width");
        }
        initial[..initial_candidate.tokens.len()].copy_from_slice(&initial_candidate.tokens);
        let active_length = initial_candidate.tokens.len();
        let initial_decoded = vocabulary.decode(&initial).map_err(anyhow::Error::msg)?;
        metrics.initial_literal_top1 += usize::from(initial_decoded == record.peptidoform);
        metrics.initial_sequence_top1 +=
            usize::from(initial_decoded.sequence == record.peptidoform.sequence);
        metrics.initial_il_top1 += usize::from(
            il_sequence(&initial_decoded.sequence) == il_sequence(&record.peptidoform.sequence),
        );
        if let Ok(mass) = foundation_chemistry_diffusion_row_neutral_mass(&initial, active_length) {
            metrics.initial_mass_error_abs_sum += (mass - precursor_mass).abs();
            metrics.initial_mass_error_records += 1;
        }

        let target = vocabulary
            .encode(&record.peptidoform, config.max_tokens)
            .map_err(anyhow::Error::msg)?;
        let target_active = target
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
            .unwrap_or(config.max_tokens);
        let editable_len = active_length.saturating_sub(1);
        let compare_len = editable_len.min(target_active.saturating_sub(1));

        let mut current = initial.clone();
        let mut last_logits = None::<Vec<Vec<f32>>>;
        for (step_index, &timestep) in timesteps.iter().enumerate() {
            let diffusion = diffusion_collator.collate_inference_tokens(
                &[current.clone()],
                &[active_length],
                timestep,
                device,
            )?;
            let candidate_chemistry = chemistry.featurize(
                &diffusion.noisy_token_rows,
                &diffusion.active_lengths,
                &[spectrum.clone()],
                &[precursor_mass],
                &[charge],
                device,
            )?;
            let output = model.forward_t(
                &diffusion,
                &spectrum_batch,
                &precursor,
                &candidate_chemistry,
                false,
            )?;
            let logits = output.token_logits.to_vec3::<f32>()?;
            let (next, changed) = foundation_chemistry_diffusion_argmax_refine(
                &[current.clone()],
                &[active_length],
                &logits,
            )
            .map_err(anyhow::Error::msg)?;
            metrics.per_step_changed_positions[step_index] += changed;
            metrics.per_step_converged_records[step_index] += usize::from(changed == 0);
            current = next.into_iter().next().unwrap();
            last_logits = Some(logits.into_iter().next().unwrap());
        }

        let final_logits =
            last_logits.ok_or_else(|| anyhow::anyhow!("v0.21 no refinement logits"))?;
        if let Ok(mass) = foundation_chemistry_diffusion_row_neutral_mass(&current, active_length) {
            metrics.pre_projection_mass_error_abs_sum += (mass - precursor_mass).abs();
            metrics.pre_projection_mass_error_records += 1;
        }
        metrics.pre_projection_mass_valid +=
            usize::from(foundation_chemistry_diffusion_final_mass_valid(
                &current,
                active_length,
                precursor_mass,
                config.precursor_mass_tolerance_da,
            ));
        let projected = foundation_chemistry_diffusion_project_mass_valid(
            &current,
            active_length,
            &final_logits,
            precursor_mass,
            config.precursor_mass_tolerance_da,
        )
        .map_err(anyhow::Error::msg)?;
        let final_row = if let Some(projected) = projected {
            metrics.mass_projection_used += usize::from(projected != current);
            projected
        } else if foundation_chemistry_diffusion_final_mass_valid(
            &initial,
            active_length,
            precursor_mass,
            config.precursor_mass_tolerance_da,
        ) {
            metrics.mass_projection_failed += 1;
            initial.clone()
        } else {
            metrics.mass_projection_failed += 1;
            current.clone()
        };

        metrics.final_mass_valid += usize::from(foundation_chemistry_diffusion_final_mass_valid(
            &final_row,
            active_length,
            precursor_mass,
            config.precursor_mass_tolerance_da,
        ));
        if let Ok(mass) = foundation_chemistry_diffusion_row_neutral_mass(&final_row, active_length)
        {
            metrics.final_mass_error_abs_sum += (mass - precursor_mass).abs();
            metrics.final_mass_error_records += 1;
        }
        let final_decoded = vocabulary.decode(&final_row).map_err(anyhow::Error::msg)?;
        metrics.final_literal_top1 += usize::from(final_decoded == record.peptidoform);
        metrics.final_sequence_top1 +=
            usize::from(final_decoded.sequence == record.peptidoform.sequence);
        metrics.final_il_top1 += usize::from(
            il_sequence(&final_decoded.sequence) == il_sequence(&record.peptidoform.sequence),
        );

        metrics.compared_positions += editable_len;
        metrics.changed_positions += (0..editable_len)
            .filter(|&position| initial[position] != final_row[position])
            .count();
        for position in 0..compare_len {
            if initial[position] == target[position] {
                metrics.initially_correct_positions += 1;
                metrics.correct_positions_damaged +=
                    usize::from(final_row[position] != target[position]);
            } else {
                metrics.initially_incorrect_positions += 1;
                metrics.incorrect_positions_corrected +=
                    usize::from(final_row[position] == target[position]);
            }
        }
    }
    Ok(metrics)
}

// -----------------------------------------------------------------------------
// v0.22 on-policy structured edit transducer
// -----------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct V0210ParentMetadataV0220 {
    inverse_config: FoundationDiffusionConfig,
    global_step: usize,
}

#[derive(Debug, Clone)]
struct StructuredEditPairV0220 {
    record_index: usize,
    initial_row: Vec<u32>,
    initial_active: usize,
    target_row: Vec<u32>,
    target_active: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct StructuredEditValidationMetricsV0220 {
    token_nll: f64,
    shuffled_token_nll: f64,
    conditioning_gap: f64,
    gate_nll: f64,
    length_nll: f64,
    token_accuracy: f64,
    gate_accuracy: f64,
    length_accuracy: f64,
    raw_exact_rate: f64,
    active_tokens: usize,
    sequences: usize,
}

#[derive(Debug, Serialize)]
struct StructuredEditCheckpointV0220<'a> {
    version: u32,
    architecture: &'a str,
    objective: &'a str,
    mode: &'a str,
    global_step: usize,
    training_seed: u64,
    train_steps: usize,
    batch_size: usize,
    train_initializers: usize,
    validation_initializers: usize,
    validation_zero_initializers: usize,
    v0200_checkpoint: String,
    v0210_checkpoint: String,
    v0210_parent_step: usize,
    validation: StructuredEditValidationMetricsV0220,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Default)]
struct StructuredEditFrozenMetricsV0220 {
    records: usize,
    initialized_records: usize,
    zero_initial_records: usize,
    initial_literal_top1: usize,
    initial_sequence_top1: usize,
    initial_il_top1: usize,
    final_literal_top1: usize,
    final_sequence_top1: usize,
    final_il_top1: usize,
    initial_length_exact: usize,
    final_length_exact: usize,
    length_mismatch_records: usize,
    length_mismatches_corrected: usize,
    positions_compared: usize,
    initially_incorrect_positions: usize,
    incorrect_positions_corrected: usize,
    initially_correct_positions: usize,
    correct_positions_damaged: usize,
    changed_positions: usize,
    mass_projection_used: usize,
    initial_fallback_used: usize,
    final_mass_valid: usize,
    initial_mass_error_abs_sum: f64,
    final_mass_error_abs_sum: f64,
    initial_mass_error_records: usize,
    final_mass_error_records: usize,
}

impl StructuredEditFrozenMetricsV0220 {
    fn print(&self) {
        println!("final_records\t{}", self.records);
        println!("v0200_initialized_records\t{}", self.initialized_records);
        println!("v0200_zero_initial_records\t{}", self.zero_initial_records);
        println!("initial_literal_top1\t{}", self.initial_literal_top1);
        println!("initial_sequence_top1\t{}", self.initial_sequence_top1);
        println!("initial_il_top1\t{}", self.initial_il_top1);
        println!("final_literal_top1\t{}", self.final_literal_top1);
        println!("final_sequence_top1\t{}", self.final_sequence_top1);
        println!("final_il_top1\t{}", self.final_il_top1);
        println!(
            "initial_length_exact_records\t{}",
            self.initial_length_exact
        );
        println!("final_length_exact_records\t{}", self.final_length_exact);
        println!(
            "initial_length_mismatch_records\t{}",
            self.length_mismatch_records
        );
        println!(
            "length_mismatches_corrected\t{}",
            self.length_mismatches_corrected
        );
        println!("positions_compared_to_v0200\t{}", self.positions_compared);
        println!("positions_changed_from_v0200\t{}", self.changed_positions);
        println!(
            "position_change_fraction_from_v0200\t{:.8}",
            if self.positions_compared == 0 {
                0.0
            } else {
                self.changed_positions as f64 / self.positions_compared as f64
            }
        );
        println!(
            "initially_incorrect_positions\t{}",
            self.initially_incorrect_positions
        );
        println!(
            "incorrect_v0200_positions_corrected\t{}",
            self.incorrect_positions_corrected
        );
        println!(
            "incorrect_position_correction_fraction\t{:.8}",
            if self.initially_incorrect_positions == 0 {
                0.0
            } else {
                self.incorrect_positions_corrected as f64
                    / self.initially_incorrect_positions as f64
            }
        );
        println!(
            "initially_correct_positions\t{}",
            self.initially_correct_positions
        );
        println!(
            "correct_v0200_positions_damaged\t{}",
            self.correct_positions_damaged
        );
        println!(
            "correct_position_damage_fraction\t{:.8}",
            if self.initially_correct_positions == 0 {
                0.0
            } else {
                self.correct_positions_damaged as f64 / self.initially_correct_positions as f64
            }
        );
        println!("final_mass_projection_used\t{}", self.mass_projection_used);
        println!(
            "final_initial_fallback_used\t{}",
            self.initial_fallback_used
        );
        println!("final_mass_valid_records\t{}", self.final_mass_valid);
        println!(
            "final_mass_valid_fraction\t{:.8}",
            if self.records == 0 {
                0.0
            } else {
                self.final_mass_valid as f64 / self.records as f64
            }
        );
        println!(
            "initial_mean_abs_precursor_mass_error_da\t{:.8}",
            if self.initial_mass_error_records == 0 {
                f64::NAN
            } else {
                self.initial_mass_error_abs_sum / self.initial_mass_error_records as f64
            }
        );
        println!(
            "final_mean_abs_precursor_mass_error_da\t{:.8}",
            if self.final_mass_error_records == 0 {
                f64::NAN
            } else {
                self.final_mass_error_abs_sum / self.final_mass_error_records as f64
            }
        );
    }
}

fn v0220_stage(stage: &str, started: &Instant) {
    println!(
        "v0220_stage\tstage={stage}\telapsed_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();
}

/// v0.22 first experiment: learn one whole-sequence edit directly from actual
/// frozen v0.20 TRAIN top-1 hypotheses. No categorical corruption and no
/// candidate-ranking objective are used.
pub(crate) fn v0220_main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 || args.len() > 7 {
        anyhow::bail!(
            "usage: foundation_train_structured_edit_v0220 RUN.yaml V0200_CHECKPOINT V0210_CHECKPOINT VALIDATION_CANDIDATES.tsv OUTPUT_DIR [full|smoke]"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let v0200_checkpoint = PathBuf::from(&args[2]);
    let v0210_checkpoint = PathBuf::from(&args[3]);
    let validation_candidates = PathBuf::from(&args[4]);
    let output_root = PathBuf::from(&args[5]);
    let mode = args.get(6).map(String::as_str).unwrap_or("full");
    let (train_steps, batch_size, train_initializer_target, validation_limit) = match mode {
        "full" => (
            4_000usize,
            32usize,
            FOUNDATION_STRUCTURED_EDIT_TRAIN_INITIALIZERS_V0220,
            None,
        ),
        "smoke" => (1usize, 2usize, 8usize, Some(2usize)),
        other => anyhow::bail!("unsupported v0.22 mode {other:?}; expected full or smoke"),
    };
    let seed = 20_260_922u64;
    let runtime_started = Instant::now();
    v0220_stage("start", &runtime_started);
    for path in [
        training_yaml.as_path(),
        v0200_checkpoint.as_path(),
        v0210_checkpoint.as_path(),
        validation_candidates.as_path(),
        output_root.as_path(),
    ] {
        reject_test_path_v0190(path)?;
    }

    let run = read_foundation_training_run_config(&training_yaml)?;
    v0220_stage("run_config_loaded", &runtime_started);
    let corpus = load_foundation_corpus(&run.corpus)?;
    v0220_stage("corpus_loaded", &runtime_started);
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark.validate_against_records(&corpus.records)?;
    v0220_stage("benchmark_validated", &runtime_started);

    let v0210_metadata_path = if v0210_checkpoint.is_dir() {
        v0210_checkpoint.join("metadata.yaml")
    } else {
        v0210_checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    };
    let v0210_metadata: V0210ParentMetadataV0220 = serde_yaml::from_str(
        &fs::read_to_string(&v0210_metadata_path)
            .with_context(|| format!("read v0.21 metadata {v0210_metadata_path:?}"))?,
    )?;
    let config = v0210_metadata.inverse_config.clone();
    config.validate().map_err(anyhow::Error::msg)?;
    let v0200_model_path = resolve_model_safetensors(&v0200_checkpoint);
    let v0210_model_path = resolve_model_safetensors(&v0210_checkpoint);

    let vocabulary = FoundationDiffusionVocabulary;
    let mut train_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &config,
        vocabulary,
    );
    train_indices.retain(|&index| v0200_precursor_mass(&corpus.records[index]).is_ok());
    let validation_cohort = read_frozen_validation_cohort_v0190(
        &validation_candidates,
        &corpus.records,
        &benchmark,
        &config,
        vocabulary,
    )?;
    let mut validation_indices = validation_cohort.indices.clone();
    if let Some(limit) = validation_limit {
        validation_indices.truncate(limit);
    }
    if train_indices.len() < train_initializer_target || validation_indices.is_empty() {
        anyhow::bail!(
            "insufficient v0.22 records: train={} requested_train_initializers={} validation={}",
            train_indices.len(),
            train_initializer_target,
            validation_indices.len()
        );
    }

    let mut partition_by_record = vec![None; corpus.records.len()];
    for entry in &benchmark.entries {
        let slot = partition_by_record
            .get_mut(entry.record_index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "benchmark record index {} outside corpus",
                    entry.record_index
                )
            })?;
        *slot = Some(entry.partition);
    }
    let train_labels = train_indices
        .iter()
        .map(|&index| {
            partition_by_record[index]
                .ok_or_else(|| anyhow::anyhow!("missing TRAIN partition label for {index}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let validation_labels = validation_indices
        .iter()
        .map(|&index| {
            partition_by_record[index]
                .ok_or_else(|| anyhow::anyhow!("missing VALIDATION partition label for {index}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if !foundation_structured_edit_partition_isolated(&train_labels, &validation_labels) {
        anyhow::bail!("v0.22 partition isolation failed");
    }
    drop(partition_by_record);
    println!("test_partition_consumed\tfalse");
    v0220_stage("partitions_and_frozen_cohort_ready", &runtime_started);

    let device = Device::cuda_if_available(0)?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;
    let causal_collator = FoundationCausalCollator::new(config.clone())?;
    let diffusion_collator = FoundationDiffusionCollator::new(config.clone())?;
    let chemistry = ChemistryDiffusionFeaturizer::new(&config).map_err(anyhow::Error::msg)?;

    // Build the frozen v0.20 initializer once. Full mode uses a deterministic
    // TRAIN subset and the exact frozen validation cohort; no TEST record ids are
    // materialized.
    let train_candidate_count =
        (train_initializer_target + train_initializer_target / 4 + 32).min(train_indices.len());
    let train_materialization_indices =
        deterministic_subset(&train_indices, train_candidate_count, seed ^ 0x2200_0001);
    let max_initializer_mass = train_materialization_indices
        .iter()
        .chain(validation_indices.iter())
        .map(|&index| v0200_precursor_mass(&corpus.records[index]))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(0.0f64, f64::max);
    let suffix_lattice =
        ChemistrySuffixMassLattice::new(config.max_tokens, max_initializer_mass + 100.0)
            .map_err(anyhow::Error::msg)?;
    let v0200_featurizer =
        ChemistryTransitionFeaturizer::new(&config, suffix_lattice).map_err(anyhow::Error::msg)?;
    let mut v0200_varmap = VarMap::new();
    let v0200_vb = VarBuilder::from_varmap(&v0200_varmap, DType::F32, &device);
    let v0200_model = PeptideSpectrumChemistryDecoder::new(config.clone(), v0200_vb)?;
    v0200_varmap.load(&v0200_model_path).with_context(|| {
        format!(
            "load frozen v0.20 checkpoint {}",
            v0200_model_path.display()
        )
    })?;
    v0220_stage("v0200_initializer_ready", &runtime_started);

    let mut train_pairs = Vec::with_capacity(train_initializer_target);
    for (ordinal, &index) in train_materialization_indices.iter().enumerate() {
        if ordinal == 0 || ordinal % 256 == 0 {
            eprintln!(
                "v0220_train_initializer_progress\trecord={}/{}\tpairs={}",
                ordinal + 1,
                train_materialization_indices.len(),
                train_pairs.len()
            );
            let _ = std::io::stderr().flush();
        }
        if let Some(pair) = generate_v0200_pair_v0220(
            &v0200_model,
            &corpus.records,
            index,
            &causal_collator,
            &spectrum_collator,
            &v0200_featurizer,
            &config,
            &device,
        )? {
            train_pairs.push(pair);
            if train_pairs.len() >= train_initializer_target {
                break;
            }
        }
    }
    if train_pairs.len() < train_initializer_target {
        anyhow::bail!(
            "v0.22 could materialize only {} of {} requested v0.20 TRAIN initializers",
            train_pairs.len(),
            train_initializer_target
        );
    }

    let mut validation_pairs = Vec::with_capacity(validation_indices.len());
    let mut validation_zero_initializers = 0usize;
    for (ordinal, &index) in validation_indices.iter().enumerate() {
        if ordinal == 0 || ordinal % 25 == 0 {
            eprintln!(
                "v0220_validation_initializer_progress\trecord={}/{}",
                ordinal + 1,
                validation_indices.len()
            );
            let _ = std::io::stderr().flush();
        }
        let pair = generate_v0200_pair_v0220(
            &v0200_model,
            &corpus.records,
            index,
            &causal_collator,
            &spectrum_collator,
            &v0200_featurizer,
            &config,
            &device,
        )?;
        validation_zero_initializers += usize::from(pair.is_none());
        validation_pairs.push(pair);
    }
    v0220_stage("on_policy_initializers_materialized", &runtime_started);

    fs::create_dir_all(&output_root)?;
    write_structured_edit_pair_manifest_v0220(
        &output_root.join("v0200_train_initializers.tsv"),
        &train_pairs,
        &corpus.records,
    )?;
    let validation_present = validation_pairs
        .iter()
        .filter_map(|pair| pair.clone())
        .collect::<Vec<_>>();
    write_structured_edit_pair_manifest_v0220(
        &output_root.join("v0200_validation_initializers.tsv"),
        &validation_present,
        &corpus.records,
    )?;
    print_initializer_summary_v0220("train_initializer", &train_pairs, &corpus.records)?;
    print_initializer_summary_v0220(
        "validation_initializer",
        &validation_present,
        &corpus.records,
    )?;

    // The scientific baseline must remain the frozen v0.20 result before any
    // structured-editor training is accepted.
    if mode == "full" {
        let baseline = baseline_counts_v0220(&validation_pairs, &corpus.records)?;
        println!("frozen_v0200_initializer_literal_top1\t{}", baseline.0);
        println!("frozen_v0200_initializer_sequence_top1\t{}", baseline.1);
        println!("frozen_v0200_initializer_il_top1\t{}", baseline.2);
        println!("frozen_v0200_initializer_records\t{}", baseline.3);
        println!(
            "frozen_v0200_zero_initializer_records\t{}",
            validation_zero_initializers
        );
        if baseline != (7, 7, 12, 118) || validation_zero_initializers != 7 {
            anyhow::bail!(
                "v0.22 frozen v0.20 initializer drift: observed {:?} zero={} expected (7,7,12,118) zero=7",
                baseline,
                validation_zero_initializers
            );
        }
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumStructuredEditor::new(config.clone(), vb)?;
    let warm = load_structured_editor_from_v0210_checkpoint(&varmap, &v0210_model_path, &device)?;
    if warm.new_gate_variables != 2 {
        anyhow::bail!(
            "v0.22 warm-start expected exactly two new gate tensors, observed {}",
            warm.new_gate_variables
        );
    }
    v0220_stage("structured_editor_warm_started", &runtime_started);

    println!("version\tv0.22.0");
    println!("architecture\t{FOUNDATION_STRUCTURED_EDIT_ARCHITECTURE_V0220}");
    println!("objective\t{FOUNDATION_STRUCTURED_EDIT_OBJECTIVE_V0220}");
    println!("scientific_change\ton_policy_full_sequence_editing_of_real_v0200_top1_errors");
    println!("synthetic_corruption\tNONE");
    println!("candidate_reranking\tNONE");
    println!("edit_passes\t{FOUNDATION_STRUCTURED_EDIT_PASSES_V0220}");
    println!("context_timestep_plumbing_only\t{FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220}");
    println!("initial_beam_width\t{FOUNDATION_STRUCTURED_EDIT_INITIAL_BEAM_WIDTH_V0220}");
    println!("train_initializers\t{}", train_pairs.len());
    println!("validation_records\t{}", validation_indices.len());
    println!("validation_initializers\t{}", validation_present.len());
    println!("validation_zero_initializers\t{validation_zero_initializers}");
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("training_seed\t{seed}");
    println!("gate_loss_weight\t{FOUNDATION_STRUCTURED_EDIT_GATE_WEIGHT_V0220}");
    println!("length_loss_weight\t{FOUNDATION_STRUCTURED_EDIT_LENGTH_WEIGHT_V0220}");
    println!("shared_v0210_variables\t{}", warm.shared_v0210_variables);
    println!("new_gate_variables\t{}", warm.new_gate_variables);
    println!("ignored_v0210_variables\t{}", warm.ignored_v0210_variables);
    println!("v0210_parent_global_step\t{}", v0210_metadata.global_step);
    println!("test_partition_consumed\tfalse");

    let initial_metrics = evaluate_structured_edit_v0220(
        &model,
        &validation_present,
        &corpus.records,
        batch_size,
        &diffusion_collator,
        &spectrum_collator,
        &chemistry,
        &device,
        seed ^ 0x2200_1000,
    )?;
    print_structured_edit_metrics_v0220("initial_validation", 0, initial_metrics);
    save_v0220_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &config,
        mode,
        0,
        seed,
        train_steps,
        batch_size,
        train_pairs.len(),
        validation_present.len(),
        validation_zero_initializers,
        initial_metrics,
        &v0200_checkpoint,
        &v0210_checkpoint,
        v0210_metadata.global_step,
    )?;
    save_v0220_checkpoint(
        &output_root.join("best"),
        &varmap,
        &config,
        mode,
        0,
        seed,
        train_steps,
        batch_size,
        train_pairs.len(),
        validation_present.len(),
        validation_zero_initializers,
        initial_metrics,
        &v0200_checkpoint,
        &v0210_checkpoint,
        v0210_metadata.global_step,
    )?;
    let mut best_step = 0usize;
    let mut best_metrics = initial_metrics;
    let mut best_objective = structured_edit_validation_objective_v0220(initial_metrics);
    v0220_stage(
        "initial_validation_and_checkpoint_complete",
        &runtime_started,
    );

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: 1.0e-4,
            weight_decay: 1.0e-4,
            ..FoundationAdamWConfig::default()
        },
    )?;
    let pair_indices = (0..train_pairs.len()).collect::<Vec<_>>();
    for step in 1..=train_steps {
        let selected = deterministic_batch(
            &pair_indices,
            batch_size,
            seed ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let selected_pairs = selected
            .iter()
            .map(|&index| train_pairs[index].clone())
            .collect::<Vec<_>>();
        let train = structured_edit_forward_batch_v0220(
            &model,
            &selected_pairs,
            &corpus.records,
            &diffusion_collator,
            &spectrum_collator,
            &chemistry,
            &device,
            true,
            false,
            seed ^ step as u64,
        )?;
        let token_loss = foundation_diffusion_x0_loss(&train.output.sequence, &train.batch)?;
        let gate_loss = foundation_structured_edit_gate_loss(
            &train.output,
            &train.input_rows,
            &train.target_rows,
            &train.target_active_lengths,
        )?;
        let length_loss = foundation_diffusion_length_loss(&train.output.sequence, &train.batch)?;

        let shuffled = structured_edit_forward_batch_v0220(
            &model,
            &selected_pairs,
            &corpus.records,
            &diffusion_collator,
            &spectrum_collator,
            &chemistry,
            &device,
            true,
            true,
            seed ^ 0x7777_0000 ^ step as u64,
        )?;
        let shuffled_token =
            foundation_diffusion_x0_loss(&shuffled.output.sequence, &shuffled.batch)?;
        let conditioned = foundation_direct_conditioning_loss(
            &token_loss,
            &shuffled_token,
            FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
            FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
        )?;
        let loss = (&conditioned
            + &gate_loss.affine(FOUNDATION_STRUCTURED_EDIT_GATE_WEIGHT_V0220, 0.0)?)?;
        let loss =
            (&loss + &length_loss.affine(FOUNDATION_STRUCTURED_EDIT_LENGTH_WEIGHT_V0220, 0.0)?)?;
        let token_value = f64::from(token_loss.to_scalar::<f32>()?);
        let shuffled_value = f64::from(shuffled_token.to_scalar::<f32>()?);
        let gate_value = f64::from(gate_loss.to_scalar::<f32>()?);
        let length_value = f64::from(length_loss.to_scalar::<f32>()?);
        let objective_value = f64::from(loss.to_scalar::<f32>()?);
        let update = optimizer.backward_step(&loss, Some(5.0))?;
        if step == 1 || step % 25 == 0 || step == train_steps {
            println!(
                "train\tstep={step}\tobjective={objective_value:.6}\ttoken_nll={token_value:.6}\tshuffled_token_nll={shuffled_value:.6}\tconditioning_gap={:.6}\tgate_nll={gate_value:.6}\tlength_nll={length_value:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                shuffled_value - token_value,
                update.gradient_norm,
                update.gradient_scale
            );
        }
        if step % 100 == 0 || step == train_steps {
            let metrics = evaluate_structured_edit_v0220(
                &model,
                &validation_present,
                &corpus.records,
                batch_size,
                &diffusion_collator,
                &spectrum_collator,
                &chemistry,
                &device,
                seed ^ step as u64,
            )?;
            print_structured_edit_metrics_v0220("validation", step, metrics);
            save_v0220_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &config,
                mode,
                step,
                seed,
                train_steps,
                batch_size,
                train_pairs.len(),
                validation_present.len(),
                validation_zero_initializers,
                metrics,
                &v0200_checkpoint,
                &v0210_checkpoint,
                v0210_metadata.global_step,
            )?;
            let objective = structured_edit_validation_objective_v0220(metrics);
            if objective < best_objective
                && metrics.conditioning_gap >= FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190
            {
                best_objective = objective;
                best_metrics = metrics;
                best_step = step;
                save_v0220_checkpoint(
                    &output_root.join("best"),
                    &varmap,
                    &config,
                    mode,
                    step,
                    seed,
                    train_steps,
                    batch_size,
                    train_pairs.len(),
                    validation_present.len(),
                    validation_zero_initializers,
                    metrics,
                    &v0200_checkpoint,
                    &v0210_checkpoint,
                    v0210_metadata.global_step,
                )?;
            }
        }
    }
    v0220_stage("optimizer_loop_complete", &runtime_started);
    println!("best_step\t{best_step}");
    print_structured_edit_metrics_v0220("final_best_validation", best_step, best_metrics);
    load_varmap_checkpoint_v0210(
        &varmap,
        &output_root.join("best/model.safetensors"),
        &device,
    )?;

    if mode == "smoke" {
        let smoke_pair = validation_pairs
            .iter()
            .find(|pair| pair.is_some())
            .cloned()
            .context("v0.22 smoke requires at least one initialized validation peptide")?;
        let smoke_metrics = evaluate_frozen_structured_edit_v0220(
            &model,
            &[smoke_pair],
            &corpus.records,
            &diffusion_collator,
            &spectrum_collator,
            &chemistry,
            &config,
            &device,
        )?;
        smoke_metrics.print();
        println!("v0220_stop_rule\tSMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION");
        println!("test_partition_consumed\tfalse");
        v0220_stage("complete", &runtime_started);
        return Ok(());
    }

    let frozen_metrics = evaluate_frozen_structured_edit_v0220(
        &model,
        &validation_pairs,
        &corpus.records,
        &diffusion_collator,
        &spectrum_collator,
        &chemistry,
        &config,
        &device,
    )?;
    frozen_metrics.print();
    let baseline_recovery =
        frozen_metrics.final_literal_top1 >= 24 && frozen_metrics.final_il_top1 >= 38;
    let progress = frozen_metrics.final_literal_top1 >= 28 && frozen_metrics.final_il_top1 >= 42;
    let material = frozen_metrics.final_literal_top1 >= 12 && frozen_metrics.final_il_top1 >= 20;
    println!(
        "baseline_recovery_gate\t{}",
        if baseline_recovery { "PASS" } else { "FAIL" }
    );
    println!("progress_gate\t{}", if progress { "PASS" } else { "FAIL" });
    println!("material_improvement_threshold\tliteral>=12_and_il>=20");
    println!(
        "material_improvement_over_v0200\t{}",
        if material { "YES" } else { "NO" }
    );
    let stop_rule = if progress {
        "V0220_PROGRESS_TARGET_MET"
    } else if baseline_recovery {
        "V0220_BASELINE_RECOVERED"
    } else if material {
        "V0220_PARTIAL_SUCCESS_ALLOW_AT_MOST_ONE_COHERENT_ARCHITECTURAL_CORRECTION"
    } else {
        "STOP_STRUCTURED_EDIT_REASSESS_PROPOSAL_SPACE"
    };
    println!("v0220_stop_rule\t{stop_rule}");
    println!("test_partition_consumed\tfalse");
    v0220_stage("complete", &runtime_started);
    Ok(())
}

struct StructuredEditForwardBatchV0220 {
    output: FoundationStructuredEditOutput,
    batch: redeem_properties::foundation::FoundationDiffusionBatch,
    input_rows: Vec<Vec<u32>>,
    target_rows: Vec<Vec<u32>>,
    target_active_lengths: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn structured_edit_forward_batch_v0220(
    model: &PeptideSpectrumStructuredEditor,
    pairs: &[StructuredEditPairV0220],
    records: &[FoundationTrainingRecord],
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    chemistry: &ChemistryDiffusionFeaturizer,
    device: &Device,
    train: bool,
    shuffle_spectra: bool,
    seed: u64,
) -> Result<StructuredEditForwardBatchV0220> {
    if pairs.is_empty() {
        anyhow::bail!("v0.22 forward batch is empty");
    }
    let width = model.config().max_tokens;
    let input_rows = pairs
        .iter()
        .map(|pair| {
            foundation_structured_edit_open_row(&pair.initial_row, pair.initial_active, width)
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)?;
    let active_lengths = vec![width; pairs.len()];
    let inference = diffusion_collator.collate_inference_tokens(
        &input_rows,
        &active_lengths,
        FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220,
        device,
    )?;
    let target_rows = pairs
        .iter()
        .map(|pair| pair.target_row.clone())
        .collect::<Vec<_>>();
    let target_active_lengths = pairs
        .iter()
        .map(|pair| pair.target_active)
        .collect::<Vec<_>>();
    let batch =
        foundation_structured_edit_set_targets(inference, &target_rows, &target_active_lengths)?;
    let selected_records = pairs
        .iter()
        .map(|pair| &records[pair.record_index])
        .collect::<Vec<_>>();
    let components = v0200_batch_components(&selected_records)?;
    let spectra = if shuffle_spectra {
        let order = foundation_direct_shuffled_order(components.spectra.len(), seed)?;
        order
            .into_iter()
            .map(|index| components.spectra[index].clone())
            .collect::<Vec<_>>()
    } else {
        components.spectra.clone()
    };
    let spectrum = spectrum_collator.collate(&spectra, device)?;
    let precursor = precursor_context(&selected_records, device)?;
    let chemistry_batch = chemistry.featurize(
        &input_rows,
        &active_lengths,
        &spectra,
        &components.precursor_masses,
        &components.charges,
        device,
    )?;
    let output = model.forward_t(&batch, &spectrum, &precursor, &chemistry_batch, train)?;
    Ok(StructuredEditForwardBatchV0220 {
        output,
        batch,
        input_rows,
        target_rows,
        target_active_lengths,
    })
}

#[allow(clippy::too_many_arguments)]
fn generate_v0200_pair_v0220(
    v0200_model: &PeptideSpectrumChemistryDecoder,
    records: &[FoundationTrainingRecord],
    index: usize,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    v0200_featurizer: &ChemistryTransitionFeaturizer,
    config: &FoundationDiffusionConfig,
    device: &Device,
) -> Result<Option<StructuredEditPairV0220>> {
    let record = records
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("v0.22 record index {index} missing"))?;
    let precursor_mass = v0200_precursor_mass(record)?;
    let charge = record
        .context
        .charge
        .ok_or_else(|| anyhow::anyhow!("v0.22 record {index} lacks charge"))?;
    let spectrum = FoundationSpectrum::from_training_record(record)
        .ok_or_else(|| anyhow::anyhow!("v0.22 record {index} lacks observed spectrum"))?;
    let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
    let precursor = precursor_context(&[record], device)?;
    let context = v0200_model.prepare_context(&spectrum_batch, &precursor, false)?;
    let candidates = foundation_direct_beam_search(
        precursor_mass,
        DirectDecoderBeamConfig {
            beam_width: FOUNDATION_STRUCTURED_EDIT_INITIAL_BEAM_WIDTH_V0220,
            top_k: 1,
            mass_tolerance_da: config.precursor_mass_tolerance_da,
            max_tokens: config.max_tokens,
        },
        |prefixes| {
            let input = causal_collator
                .collate_compact_prefix_rows(prefixes, device)
                .map_err(|error| error.to_string())?;
            let candidate_chemistry = v0200_featurizer
                .next_prefixes(prefixes, &spectrum, precursor_mass, charge, device)
                .map_err(|error| error.to_string())?;
            let mut rows = v0200_model
                .forward_next_t_with_context(&input, &context, &candidate_chemistry, false)
                .and_then(|tensor| tensor.to_vec2::<f32>())
                .map_err(|error| error.to_string())?;
            v0200_featurizer.mask_infeasible_next_logits(prefixes, &mut rows, precursor_mass)?;
            Ok(rows)
        },
    )
    .map_err(anyhow::Error::msg)?;
    let Some(candidate) = candidates.first() else {
        return Ok(None);
    };
    let vocabulary = FoundationDiffusionVocabulary;
    let mut initial_row = vec![FOUNDATION_DIFFUSION_PAD; config.max_tokens];
    if candidate.tokens.len() > config.max_tokens {
        anyhow::bail!("v0.22 v0.20 initializer exceeds max_tokens");
    }
    initial_row[..candidate.tokens.len()].copy_from_slice(&candidate.tokens);
    let target_row = vocabulary
        .encode(&record.peptidoform, config.max_tokens)
        .map_err(anyhow::Error::msg)?;
    let target_active = target_row
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap_or(config.max_tokens);
    Ok(Some(StructuredEditPairV0220 {
        record_index: index,
        initial_row,
        initial_active: candidate.tokens.len(),
        target_row,
        target_active,
    }))
}

fn write_structured_edit_pair_manifest_v0220(
    path: &Path,
    pairs: &[StructuredEditPairV0220],
    records: &[FoundationTrainingRecord],
) -> Result<()> {
    let vocabulary = FoundationDiffusionVocabulary;
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "record_index\tinitial_active\ttarget_active\tinitial_sequence\ttarget_sequence\tliteral_exact\tsequence_exact\til_exact"
    )?;
    for pair in pairs {
        let initial = vocabulary
            .decode(&pair.initial_row)
            .map_err(anyhow::Error::msg)?;
        let target = &records[pair.record_index].peptidoform;
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            pair.record_index,
            pair.initial_active,
            pair.target_active,
            initial.sequence,
            target.sequence,
            initial == *target,
            initial.sequence == target.sequence,
            il_sequence(&initial.sequence) == il_sequence(&target.sequence),
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn print_initializer_summary_v0220(
    label: &str,
    pairs: &[StructuredEditPairV0220],
    records: &[FoundationTrainingRecord],
) -> Result<()> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut literal = 0usize;
    let mut sequence = 0usize;
    let mut il = 0usize;
    let mut length_exact = 0usize;
    let mut edit_sum = 0usize;
    let mut edit_max = 0usize;
    for pair in pairs {
        let initial = vocabulary
            .decode(&pair.initial_row)
            .map_err(anyhow::Error::msg)?;
        let target = &records[pair.record_index].peptidoform;
        literal += usize::from(initial == *target);
        sequence += usize::from(initial.sequence == target.sequence);
        il += usize::from(il_sequence(&initial.sequence) == il_sequence(&target.sequence));
        length_exact += usize::from(pair.initial_active == pair.target_active);
        let initial_content = pair.initial_active.saturating_sub(1);
        let target_content = pair.target_active.saturating_sub(1);
        let compare = initial_content.max(target_content);
        let edits = (0..compare)
            .filter(|&position| {
                let left = if position < initial_content {
                    pair.initial_row[position]
                } else {
                    FOUNDATION_DIFFUSION_PAD
                };
                let right = if position < target_content {
                    pair.target_row[position]
                } else {
                    FOUNDATION_DIFFUSION_PAD
                };
                left != right
            })
            .count();
        edit_sum += edits;
        edit_max = edit_max.max(edits);
    }
    let n = pairs.len().max(1) as f64;
    println!("{label}_records\t{}", pairs.len());
    println!("{label}_literal_exact\t{literal}");
    println!("{label}_sequence_exact\t{sequence}");
    println!("{label}_il_exact\t{il}");
    println!("{label}_length_exact\t{length_exact}");
    println!(
        "{label}_length_mismatch\t{}",
        pairs.len().saturating_sub(length_exact)
    );
    println!(
        "{label}_mean_positional_edit_distance\t{:.6}",
        edit_sum as f64 / n
    );
    println!("{label}_max_positional_edit_distance\t{edit_max}");
    Ok(())
}

fn baseline_counts_v0220(
    pairs: &[Option<StructuredEditPairV0220>],
    records: &[FoundationTrainingRecord],
) -> Result<(usize, usize, usize, usize)> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut literal = 0usize;
    let mut sequence = 0usize;
    let mut il = 0usize;
    let mut initialized = 0usize;
    for pair in pairs.iter().flatten() {
        initialized += 1;
        let initial = vocabulary
            .decode(&pair.initial_row)
            .map_err(anyhow::Error::msg)?;
        let target = &records[pair.record_index].peptidoform;
        literal += usize::from(initial == *target);
        sequence += usize::from(initial.sequence == target.sequence);
        il += usize::from(il_sequence(&initial.sequence) == il_sequence(&target.sequence));
    }
    Ok((literal, sequence, il, initialized))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_structured_edit_v0220(
    model: &PeptideSpectrumStructuredEditor,
    pairs: &[StructuredEditPairV0220],
    records: &[FoundationTrainingRecord],
    batch_size: usize,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    chemistry: &ChemistryDiffusionFeaturizer,
    device: &Device,
    seed: u64,
) -> Result<StructuredEditValidationMetricsV0220> {
    if pairs.is_empty() {
        anyhow::bail!("v0.22 validation has no initialized pairs");
    }
    let mut token_weighted = 0.0f64;
    let mut shuffled_weighted = 0.0f64;
    let mut gate_weighted = 0.0f64;
    let mut length_weighted = 0.0f64;
    let mut active_tokens = 0usize;
    let mut token_correct = 0usize;
    let mut gate_correct = 0usize;
    let mut gate_total = 0usize;
    let mut length_correct = 0usize;
    let mut raw_exact = 0usize;
    for (chunk_index, chunk) in pairs.chunks(batch_size).enumerate() {
        let selected = chunk.to_vec();
        let matched = structured_edit_forward_batch_v0220(
            model,
            &selected,
            records,
            diffusion_collator,
            spectrum_collator,
            chemistry,
            device,
            false,
            false,
            seed ^ chunk_index as u64,
        )?;
        let token_loss = foundation_diffusion_x0_loss(&matched.output.sequence, &matched.batch)?;
        let gate_loss = foundation_structured_edit_gate_loss(
            &matched.output,
            &matched.input_rows,
            &matched.target_rows,
            &matched.target_active_lengths,
        )?;
        let length_loss =
            foundation_diffusion_length_loss(&matched.output.sequence, &matched.batch)?;
        let shuffled = structured_edit_forward_batch_v0220(
            model,
            &selected,
            records,
            diffusion_collator,
            spectrum_collator,
            chemistry,
            device,
            false,
            true,
            seed ^ 0x7777 ^ chunk_index as u64,
        )?;
        let shuffled_loss =
            foundation_diffusion_x0_loss(&shuffled.output.sequence, &shuffled.batch)?;
        let chunk_active: usize = matched.target_active_lengths.iter().sum();
        token_weighted += f64::from(token_loss.to_scalar::<f32>()?) * chunk_active as f64;
        shuffled_weighted += f64::from(shuffled_loss.to_scalar::<f32>()?) * chunk_active as f64;
        gate_weighted += f64::from(gate_loss.to_scalar::<f32>()?) * chunk_active as f64;
        length_weighted += f64::from(length_loss.to_scalar::<f32>()?) * selected.len() as f64;
        active_tokens += chunk_active;

        let token_logits = matched.output.sequence.token_logits.to_vec3::<f32>()?;
        let gate_logits = matched.output.gate_logits.to_vec3::<f32>()?;
        let length_logits = matched.output.sequence.length_logits.to_vec2::<f32>()?;
        for row_index in 0..selected.len() {
            let target_active = matched.target_active_lengths[row_index];
            let mut all_correct = true;
            for position in 0..target_active {
                let predicted = argmax(&token_logits[row_index][position]) as u32;
                let target = matched.target_rows[row_index][position];
                token_correct += usize::from(predicted == target);
                all_correct &= predicted == target;
                let gate_pred = argmax(&gate_logits[row_index][position]) as u32;
                let gate_target = if matched.input_rows[row_index][position]
                    != matched.target_rows[row_index][position]
                {
                    1
                } else {
                    0
                };
                gate_correct += usize::from(gate_pred == gate_target);
                gate_total += 1;
            }
            raw_exact += usize::from(all_correct);
            let predicted_length = argmax(&length_logits[row_index]) + 1;
            length_correct += usize::from(predicted_length == target_active);
        }
    }
    let sequences = pairs.len();
    let token_nll = token_weighted / active_tokens as f64;
    let shuffled_token_nll = shuffled_weighted / active_tokens as f64;
    Ok(StructuredEditValidationMetricsV0220 {
        token_nll,
        shuffled_token_nll,
        conditioning_gap: shuffled_token_nll - token_nll,
        gate_nll: gate_weighted / active_tokens as f64,
        length_nll: length_weighted / sequences as f64,
        token_accuracy: token_correct as f64 / active_tokens as f64,
        gate_accuracy: gate_correct as f64 / gate_total.max(1) as f64,
        length_accuracy: length_correct as f64 / sequences as f64,
        raw_exact_rate: raw_exact as f64 / sequences as f64,
        active_tokens,
        sequences,
    })
}

fn structured_edit_validation_objective_v0220(
    metrics: StructuredEditValidationMetricsV0220,
) -> f64 {
    metrics.token_nll
        + FOUNDATION_STRUCTURED_EDIT_GATE_WEIGHT_V0220 * metrics.gate_nll
        + FOUNDATION_STRUCTURED_EDIT_LENGTH_WEIGHT_V0220 * metrics.length_nll
}

fn print_structured_edit_metrics_v0220(
    label: &str,
    step: usize,
    metrics: StructuredEditValidationMetricsV0220,
) {
    println!(
        "{label}\tstep={step}\ttoken_nll={:.6}\tshuffled_token_nll={:.6}\tconditioning_gap={:.6}\tgate_nll={:.6}\tlength_nll={:.6}\ttoken_accuracy={:.8}\tgate_accuracy={:.8}\tlength_accuracy={:.8}\traw_exact_rate={:.8}\tactive_tokens={}\tsequences={}",
        metrics.token_nll,
        metrics.shuffled_token_nll,
        metrics.conditioning_gap,
        metrics.gate_nll,
        metrics.length_nll,
        metrics.token_accuracy,
        metrics.gate_accuracy,
        metrics.length_accuracy,
        metrics.raw_exact_rate,
        metrics.active_tokens,
        metrics.sequences,
    );
}

#[allow(clippy::too_many_arguments)]
fn save_v0220_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    config: &FoundationDiffusionConfig,
    mode: &str,
    global_step: usize,
    seed: u64,
    train_steps: usize,
    batch_size: usize,
    train_initializers: usize,
    validation_initializers: usize,
    validation_zero_initializers: usize,
    validation: StructuredEditValidationMetricsV0220,
    v0200_checkpoint: &Path,
    v0210_checkpoint: &Path,
    v0210_parent_step: usize,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    let metadata = StructuredEditCheckpointV0220 {
        version: 1,
        architecture: FOUNDATION_STRUCTURED_EDIT_ARCHITECTURE_V0220,
        objective: FOUNDATION_STRUCTURED_EDIT_OBJECTIVE_V0220,
        mode,
        global_step,
        training_seed: seed,
        train_steps,
        batch_size,
        train_initializers,
        validation_initializers,
        validation_zero_initializers,
        v0200_checkpoint: v0200_checkpoint.display().to_string(),
        v0210_checkpoint: v0210_checkpoint.display().to_string(),
        v0210_parent_step,
        validation,
        inverse_config: config.clone(),
    };
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(&metadata)?,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn structured_edit_inference_v0220(
    model: &PeptideSpectrumStructuredEditor,
    pair: &StructuredEditPairV0220,
    record: &FoundationTrainingRecord,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    chemistry: &ChemistryDiffusionFeaturizer,
    device: &Device,
) -> Result<StructuredEditForwardBatchV0220> {
    let width = model.config().max_tokens;
    let input_row =
        foundation_structured_edit_open_row(&pair.initial_row, pair.initial_active, width)
            .map_err(anyhow::Error::msg)?;
    let active_lengths = vec![width];
    // Target-free inference: collate_inference_tokens fills inert clean/target
    // tensors from the input itself. The true peptide is not passed into the
    // model or chemistry featurizer anywhere in this path.
    let batch = diffusion_collator.collate_inference_tokens(
        &[input_row.clone()],
        &active_lengths,
        FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220,
        device,
    )?;
    let components = v0200_batch_components(&[record])?;
    let spectrum = spectrum_collator.collate(&components.spectra, device)?;
    let precursor = precursor_context(&[record], device)?;
    let chemistry_batch = chemistry.featurize(
        &[input_row.clone()],
        &active_lengths,
        &components.spectra,
        &components.precursor_masses,
        &components.charges,
        device,
    )?;
    let output = model.forward_t(&batch, &spectrum, &precursor, &chemistry_batch, false)?;
    Ok(StructuredEditForwardBatchV0220 {
        output,
        batch,
        input_rows: vec![input_row],
        target_rows: Vec::new(),
        target_active_lengths: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_frozen_structured_edit_v0220(
    model: &PeptideSpectrumStructuredEditor,
    pairs: &[Option<StructuredEditPairV0220>],
    records: &[FoundationTrainingRecord],
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    chemistry: &ChemistryDiffusionFeaturizer,
    config: &FoundationDiffusionConfig,
    device: &Device,
) -> Result<StructuredEditFrozenMetricsV0220> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut metrics = StructuredEditFrozenMetricsV0220::default();
    metrics.records = pairs.len();
    for (ordinal, pair) in pairs.iter().enumerate() {
        if ordinal == 0 || ordinal % 10 == 0 {
            eprintln!(
                "v0220_frozen_edit_progress\trecord={}/{}",
                ordinal + 1,
                pairs.len()
            );
            let _ = std::io::stderr().flush();
        }
        let Some(pair) = pair else {
            metrics.zero_initial_records += 1;
            continue;
        };
        metrics.initialized_records += 1;
        let record = &records[pair.record_index];
        let initial = vocabulary
            .decode(&pair.initial_row)
            .map_err(anyhow::Error::msg)?;
        metrics.initial_literal_top1 += usize::from(initial == record.peptidoform);
        metrics.initial_sequence_top1 +=
            usize::from(initial.sequence == record.peptidoform.sequence);
        metrics.initial_il_top1 += usize::from(
            il_sequence(&initial.sequence) == il_sequence(&record.peptidoform.sequence),
        );
        metrics.initial_length_exact += usize::from(pair.initial_active == pair.target_active);
        if pair.initial_active != pair.target_active {
            metrics.length_mismatch_records += 1;
        }
        let precursor_mass = v0200_precursor_mass(record)?;
        if let Ok(mass) =
            foundation_chemistry_diffusion_row_neutral_mass(&pair.initial_row, pair.initial_active)
        {
            metrics.initial_mass_error_abs_sum += (mass - precursor_mass).abs();
            metrics.initial_mass_error_records += 1;
        }

        let forward = structured_edit_inference_v0220(
            model,
            pair,
            record,
            diffusion_collator,
            spectrum_collator,
            chemistry,
            device,
        )?;
        let mut token_logits_rows = forward.output.sequence.token_logits.to_vec3::<f32>()?;
        let token_logits = token_logits_rows.remove(0);
        let mut gate_logits_rows = forward.output.gate_logits.to_vec3::<f32>()?;
        let gate_logits = gate_logits_rows.remove(0);
        let mut length_logits_rows = forward.output.sequence.length_logits.to_vec2::<f32>()?;
        let length_logits = length_logits_rows.remove(0);
        let (draft, draft_active, _) = foundation_structured_edit_argmax(
            &forward.input_rows[0],
            &token_logits,
            &gate_logits,
            &length_logits,
        )
        .map_err(anyhow::Error::msg)?;
        let (final_row, projected, fallback) = foundation_structured_edit_finalize(
            &draft,
            draft_active,
            &token_logits,
            &pair.initial_row,
            pair.initial_active,
            precursor_mass,
            config.precursor_mass_tolerance_da,
        )
        .map_err(anyhow::Error::msg)?;
        metrics.mass_projection_used += usize::from(projected);
        metrics.initial_fallback_used += usize::from(fallback);
        let final_active = final_row
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
            .unwrap_or(config.max_tokens);
        metrics.final_mass_valid += usize::from(foundation_chemistry_diffusion_final_mass_valid(
            &final_row,
            final_active,
            precursor_mass,
            config.precursor_mass_tolerance_da,
        ));
        if let Ok(mass) = foundation_chemistry_diffusion_row_neutral_mass(&final_row, final_active)
        {
            metrics.final_mass_error_abs_sum += (mass - precursor_mass).abs();
            metrics.final_mass_error_records += 1;
        }
        let final_peptide = vocabulary.decode(&final_row).map_err(anyhow::Error::msg)?;
        metrics.final_literal_top1 += usize::from(final_peptide == record.peptidoform);
        metrics.final_sequence_top1 +=
            usize::from(final_peptide.sequence == record.peptidoform.sequence);
        metrics.final_il_top1 += usize::from(
            il_sequence(&final_peptide.sequence) == il_sequence(&record.peptidoform.sequence),
        );
        metrics.final_length_exact += usize::from(final_active == pair.target_active);
        metrics.length_mismatches_corrected += usize::from(
            pair.initial_active != pair.target_active && final_active == pair.target_active,
        );

        let initial_content = pair.initial_active.saturating_sub(1);
        let target_content = pair.target_active.saturating_sub(1);
        let final_content = final_active.saturating_sub(1);
        let compare = initial_content
            .max(target_content)
            .max(final_content)
            .min(config.max_tokens - 1);
        metrics.positions_compared += compare;
        for position in 0..compare {
            let initial_token = if position < initial_content {
                pair.initial_row[position]
            } else {
                FOUNDATION_DIFFUSION_PAD
            };
            let target_token = if position < target_content {
                pair.target_row[position]
            } else {
                FOUNDATION_DIFFUSION_PAD
            };
            let final_token = if position < final_content {
                final_row[position]
            } else {
                FOUNDATION_DIFFUSION_PAD
            };
            metrics.changed_positions += usize::from(initial_token != final_token);
            if initial_token == target_token {
                metrics.initially_correct_positions += 1;
                metrics.correct_positions_damaged += usize::from(final_token != target_token);
            } else {
                metrics.initially_incorrect_positions += 1;
                metrics.incorrect_positions_corrected += usize::from(final_token == target_token);
            }
        }
    }
    Ok(metrics)
}
