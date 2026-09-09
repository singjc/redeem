//! Train the spectrum-conditioned causal next-token reranker.
//!
//! This lane is warm-started from a trained diffusion checkpoint but optimizes
//! only the teacher-forced causal chain-rule objective. Reverse diffusion is not
//! changed by this executable.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_causal_next_token_loss, foundation_diffusion_dataset_fingerprint,
    foundation_direct_beam_search, foundation_direct_conditioning_loss,
    foundation_direct_shuffled_order, foundation_precursor_neutral_mass,
    load_causal_from_diffusion_checkpoint, load_direct_decoder_from_unified_checkpoint,
    load_foundation_corpus, read_foundation_training_run_config, DirectDecoderBeamConfig,
    FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCausalCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationPartition,
    FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideSpectrumCausalModel, PeptidoformInput, PrecursorContextBatch, FOUNDATION_DIFFUSION_EOS,
    FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190, FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
    FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0190,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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
/// remain the single implementation used by both causal generations.  Unlike
/// the historical executable, v0.19 trains on CUDA, adds the spectrum-use
/// objective, evaluates the entire frozen VALIDATION partition, and finishes
/// with direct mass-constrained beam decoding.
pub(crate) fn v0190_main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        anyhow::bail!(
            "usage: foundation_train_direct_decoder_v0190 RUN.yaml UNIFIED_CHECKPOINT OUTPUT_DIR [full|smoke]"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let parent = PathBuf::from(&args[2]);
    let output_root = PathBuf::from(&args[3]);
    let mode = args.get(4).map(String::as_str).unwrap_or("full");
    let (train_steps, batch_size, validation_limit, beam_width) = match mode {
        "full" => (4_000usize, 32usize, None, 128usize),
        "smoke" => (2usize, 2usize, Some(4usize), 8usize),
        other => anyhow::bail!("unsupported v0.19 mode {other:?}; expected full or smoke"),
    };
    reject_test_path_v0190(&training_yaml)?;
    reject_test_path_v0190(&parent)?;
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
    let mut validation_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &metadata.inverse_config,
        vocabulary,
    );
    if let Some(limit) = validation_limit {
        validation_indices.truncate(limit);
    }
    if train_indices.len() < batch_size || validation_indices.is_empty() {
        anyhow::bail!(
            "insufficient v0.19 pairs: train={} validation={} batch={batch_size}",
            train_indices.len(),
            validation_indices.len()
        );
    }
    if mode == "full" && validation_indices.len() != 125 {
        anyhow::bail!(
            "frozen v0.19 VALIDATION contract requires exactly 125 usable records, found {}",
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
    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: 1.0e-4,
            weight_decay: 1.0e-4,
            ..FoundationAdamWConfig::default()
        },
    )?;

    println!("version\tv0.19.0");
    println!("architecture\tmasked_self_attention+full_peak_cross_attention+ff");
    println!("objective\t{FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0190}");
    println!("proposal_generation_required\tfalse");
    println!("test_partition_consumed\tfalse");
    println!("mode\t{mode}");
    println!("device\t{device:?}");
    println!("train_records\t{}", train_indices.len());
    println!("validation_records\t{}", validation_indices.len());
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("beam_width\t{beam_width}");
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
    let gate = if generation.literal_top1 >= 28 && generation.il_top1 >= 42 {
        "PROGRESS_MILESTONE"
    } else if generation.literal_top1 >= 24 && generation.il_top1 >= 38 {
        "BASELINE_RECOVERY"
    } else {
        "BELOW_BASELINE_RECOVERY"
    };
    println!("scientific_gate\t{gate}");
    Ok(())
}

#[derive(Debug, Deserialize)]
struct UnifiedV0190Metadata {
    inverse_config: FoundationDiffusionConfig,
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
    literal_top1: usize,
    sequence_top1: usize,
    il_top1: usize,
    topk: Vec<(usize, usize, usize)>,
    mass_valid_beams: usize,
    returned_beams: usize,
}

impl GenerationV0190 {
    fn mass_valid_fraction(&self) -> f64 {
        if self.returned_beams == 0 {
            0.0
        } else {
            self.mass_valid_beams as f64 / self.returned_beams as f64
        }
    }
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
        literal_top1: 0,
        sequence_top1: 0,
        il_top1: 0,
        topk: ks.iter().copied().map(|k| (k, 0, 0)).collect(),
        mass_valid_beams: 0,
        returned_beams: 0,
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
