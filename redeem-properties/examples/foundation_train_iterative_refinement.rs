//! Train the isolated v0.13.15 iterative masked-refinement proposal branch.
//!
//! The accepted v0.13.10 unified checkpoint is copied into the dedicated
//! `iterative_refinement.*` namespace. Each example masks a fixed quarter of
//! residue positions while preserving the surrounding residue/PTM/EOS context.
//! Only the masked residue positions contribute to x0 cross-entropy. No accepted
//! forward, diffusion, N->C causal, or C->N reverse-causal parameter participates
//! in this optimizer.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_diffusion_dataset_fingerprint, foundation_diffusion_x0_loss,
    foundation_iterative_refinement_collate, load_foundation_corpus,
    load_iterative_refinement_from_unified_checkpoint, read_foundation_training_run_config,
    validate_iterative_refinement_namespace, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationDiffusionBatch, FoundationDiffusionCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationPartition,
    FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator,
    FoundationTrainingRecord, PeptideSpectrumDiffusionModel, PrecursorContextBatch,
    FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_OBJECTIVE_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct UnifiedParentMetadata {
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    completed_steps: usize,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Serialize)]
struct IterativeRefinementCheckpointMetadata {
    version: u32,
    objective: String,
    parameter_namespace: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    train_dataset_fingerprint: String,
    validation_dataset_fingerprint: String,
    usable_train_pairs: usize,
    usable_validation_pairs: usize,
    train_steps: usize,
    batch_size: usize,
    validation_batches: usize,
    seed: u64,
    learning_rate: f64,
    max_gradient_norm: f64,
    mask_fraction: f64,
    refinement_rounds: usize,
    seed_hypotheses: usize,
    replacement_topk: usize,
    replacement_beam_width: usize,
    parent_unified_checkpoint: String,
    parent_unified_completed_steps: usize,
    global_step: usize,
    best_validation_loss: f64,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Copy, Default)]
struct RefinementMetrics {
    loss: f64,
    perplexity: f64,
    masked_token_accuracy: f64,
    exact_masked_set_rate: f64,
    masked_tokens: usize,
    sequences: usize,
}

struct PackedBatch {
    diffusion: FoundationDiffusionBatch,
    spectrum: FoundationSpectrumBatch,
    precursor: PrecursorContextBatch,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 10 {
        anyhow::bail!(
            "usage: foundation_train_iterative_refinement FOUNDATION_TRAINING.yaml OUTPUT_DIR PARENT_UNIFIED_CHECKPOINT [train_steps=1000] [batch_size=8] [validation_batches=32] [seed=20260919] [learning_rate=1e-5] [max_gradient_norm=1.0]"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let train_steps = parse_or(&args, 4, 1_000usize)?;
    let batch_size = parse_or(&args, 5, 8usize)?;
    let validation_batches = parse_or(&args, 6, 32usize)?;
    let seed = parse_or(&args, 7, 20_260_919u64)?;
    let learning_rate = parse_or(&args, 8, 1.0e-5f64)?;
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

    let parent_metadata_path = if parent_checkpoint.is_dir() {
        parent_checkpoint.join("metadata.yaml")
    } else {
        parent_checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    };
    let parent_metadata: UnifiedParentMetadata = serde_yaml::from_str(
        &fs::read_to_string(&parent_metadata_path)
            .with_context(|| format!("failed to read {parent_metadata_path:?}"))?,
    )?;
    let config = parent_metadata.inverse_config.clone();
    config.validate().map_err(anyhow::Error::msg)?;
    let parent_model_path = resolve_model_safetensors(&parent_checkpoint);

    let expected_corpus = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let expected_benchmark = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != expected_corpus {
        anyhow::bail!(
            "parent corpus fingerprint mismatch: parent={}, current={expected_corpus}",
            parent_metadata.corpus_fingerprint
        );
    }
    if parent_metadata.benchmark_manifest_fingerprint != expected_benchmark {
        anyhow::bail!(
            "parent benchmark fingerprint mismatch: parent={}, current={expected_benchmark}",
            parent_metadata.benchmark_manifest_fingerprint
        );
    }

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
            "insufficient iterative-refinement pairs: train={} validation={} batch={batch_size}",
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
        seed ^ 0xc17a_4f92_6d30_b8e1,
    );

    fs::create_dir_all(&output_root)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device)
        .pp(FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315);
    let model = PeptideSpectrumDiffusionModel::new(config.clone(), vb)?;
    validate_iterative_refinement_namespace(&varmap, &config)?;
    let warm_start =
        load_iterative_refinement_from_unified_checkpoint(&varmap, &parent_model_path, &device)
            .with_context(|| format!("failed to warm-start from {parent_model_path:?}"))?;

    let diffusion_collator = FoundationDiffusionCollator::new(config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;
    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            ..FoundationAdamWConfig::default()
        },
    )?;

    println!("objective\t{FOUNDATION_ITERATIVE_REFINEMENT_OBJECTIVE_V01315}");
    println!("parameter_namespace\t{FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315}");
    println!("optimizer_scope\titerative_refinement_only");
    println!("parent_unified_frozen\tYES");
    println!("diffusion_parameters_in_optimizer\tNO");
    println!("n_to_c_causal_parameters_in_optimizer\tNO");
    println!("reverse_causal_parameters_in_optimizer\tNO");
    println!("forward_parameters_in_optimizer\tNO");
    println!("test_partition_consumed\tNO");
    println!("corpus_fingerprint\t{expected_corpus}");
    println!("benchmark_manifest_fingerprint\t{expected_benchmark}");
    println!("train_dataset_fingerprint\tfnv1a64:{train_fingerprint:016x}");
    println!("validation_dataset_fingerprint\tfnv1a64:{validation_fingerprint:016x}");
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
        "mask_fraction\t{}",
        FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315
    );
    println!("training_mask_policy\tquarter_residues_only_preserve_ptm_eos_context_v01315");
    println!(
        "refinement_rounds\t{}",
        FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315
    );
    println!(
        "seed_hypotheses\t{}",
        FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315
    );
    println!(
        "replacement_topk\t{}",
        FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315
    );
    println!(
        "replacement_beam_width\t{}",
        FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315
    );
    println!("parent_unified_checkpoint\t{}", parent_model_path.display());
    println!(
        "parent_unified_completed_steps\t{}",
        parent_metadata.completed_steps
    );
    println!(
        "refinement_warm_start\tloaded_variables={}\tignored_parent_variables={}",
        warm_start.loaded_variables, warm_start.ignored_parent_variables
    );

    let initial_metrics = evaluate(
        &model,
        &corpus.records,
        &validation_selection,
        batch_size,
        &diffusion_collator,
        &spectrum_collator,
        &config,
        seed ^ 0x17b4_5c3e_a908_d261,
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
        &parent_model_path,
        parent_metadata.completed_steps,
        0,
        initial_metrics.loss,
    );
    save_checkpoint(&output_root.join("initial"), &varmap, &initial_metadata)?;
    save_checkpoint(&output_root.join("best"), &varmap, &initial_metadata)?;

    let mut best_validation_loss = initial_metrics.loss;
    let mut best_step = 0usize;
    let eval_every = train_steps.min(100).max(1);
    let mut final_metrics = initial_metrics;

    for step in 1..=train_steps {
        let selected = deterministic_batch(
            &train_indices,
            batch_size,
            seed ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let records = selected
            .iter()
            .map(|&index| &corpus.records[index])
            .collect::<Vec<_>>();
        let packed = collate_records(
            &records,
            &diffusion_collator,
            &spectrum_collator,
            &config,
            seed ^ (step as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
            &device,
        )?;
        let output =
            model.forward_t(&packed.diffusion, &packed.spectrum, &packed.precursor, true)?;
        let loss = foundation_diffusion_x0_loss(&output, &packed.diffusion)?;
        let loss_value = f64::from(loss.to_scalar::<f32>()?);
        let optimizer_step = optimizer.backward_step(&loss, Some(max_gradient_norm))?;
        if step == 1 || step % 10 == 0 || step == train_steps {
            println!(
                "train\tstep={step}\tloss={loss_value:.6}\tperplexity={:.4}\tmasked_positions={}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                loss_value.exp(),
                packed.diffusion.active_indices.dims1()?,
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
                &diffusion_collator,
                &spectrum_collator,
                &config,
                seed ^ 0x17b4_5c3e_a908_d261,
                &device,
            )?;
            final_metrics = metrics;
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
                &parent_model_path,
                parent_metadata.completed_steps,
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
                    &parent_model_path,
                    parent_metadata.completed_steps,
                    step,
                    best_validation_loss,
                );
                save_checkpoint(&output_root.join("best"), &varmap, &best_metadata)?;
            }
        }
    }

    let final_metadata = checkpoint_metadata(
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
        &parent_model_path,
        parent_metadata.completed_steps,
        train_steps,
        best_validation_loss,
    );
    save_checkpoint(&output_root.join("final"), &varmap, &final_metadata)?;
    println!("best_step\t{best_step}");
    println!("best_validation_iterative_refinement_loss\t{best_validation_loss:.8}");
    println!(
        "final_validation_iterative_refinement_loss\t{:.8}",
        final_metrics.loss
    );
    println!("best_checkpoint\t{}", output_root.join("best").display());
    println!("final_checkpoint\t{}", output_root.join("final").display());
    println!("test_partition_consumed\tNO");
    Ok(())
}

fn collate_records(
    records: &[&FoundationTrainingRecord],
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    mask_seed: u64,
    device: &Device,
) -> Result<PackedBatch> {
    let peptides = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect::<Vec<_>>();
    let spectra = records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record).ok_or_else(|| {
                anyhow::anyhow!("iterative-refinement record unexpectedly lacks spectrum")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PackedBatch {
        diffusion: foundation_iterative_refinement_collate(
            diffusion_collator,
            FoundationDiffusionVocabulary,
            config,
            &peptides,
            FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315,
            mask_seed,
            device,
        )?,
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
        .collect::<Vec<_>>();
    let charge_present: Vec<f32> = records
        .iter()
        .map(|record| {
            if record.context.charge.is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let precursor_mz: Vec<f32> = records
        .iter()
        .map(|record| record.context.precursor_mz.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let precursor_mz_present: Vec<f32> = records
        .iter()
        .map(|record| {
            if record.context.precursor_mz.is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
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
            let usable = FoundationSpectrum::from_training_record(record).is_some()
                && vocabulary
                    .encode(&record.peptidoform, config.max_tokens)
                    .ok()
                    .is_some();
            usable.then_some(entry.record_index)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    model: &PeptideSpectrumDiffusionModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    mask_seed: u64,
    device: &Device,
) -> Result<RefinementMetrics> {
    let mut loss_sum = 0.0f64;
    let mut correct = 0usize;
    let mut masked_tokens = 0usize;
    let mut sequences = 0usize;
    let mut exact_sequences = 0usize;

    for (chunk_index, chunk) in indices.chunks(batch_size).enumerate() {
        let selected = chunk
            .iter()
            .map(|&index| &records[index])
            .collect::<Vec<_>>();
        let packed = collate_records(
            &selected,
            diffusion_collator,
            spectrum_collator,
            config,
            mix64(mask_seed ^ chunk_index as u64),
            device,
        )?;
        let output = model.forward_t(
            &packed.diffusion,
            &packed.spectrum,
            &packed.precursor,
            false,
        )?;
        let loss = foundation_diffusion_x0_loss(&output, &packed.diffusion)?;
        let active = packed.diffusion.active_indices.to_vec1::<u32>()?;
        let targets = packed.diffusion.target_classes.to_vec1::<u32>()?;
        if active.len() != targets.len() {
            anyhow::bail!("iterative-refinement validation active/target arity mismatch");
        }
        loss_sum += f64::from(loss.to_scalar::<f32>()?) * active.len() as f64;
        let logits = output.token_logits.to_vec3::<f32>()?;
        let mut row_exact = vec![true; selected.len()];
        let mut row_seen = vec![false; selected.len()];
        for (&flat, &target) in active.iter().zip(&targets) {
            let flat = flat as usize;
            let row = flat / config.max_tokens;
            let position = flat % config.max_tokens;
            let predicted = argmax(&logits[row][position]) as u32;
            let hit = predicted == target;
            correct += usize::from(hit);
            masked_tokens += 1;
            row_exact[row] &= hit;
            row_seen[row] = true;
        }
        for row in 0..selected.len() {
            if !row_seen[row] {
                anyhow::bail!("iterative-refinement validation row contains no masked residues");
            }
            exact_sequences += usize::from(row_exact[row]);
            sequences += 1;
        }
    }
    if masked_tokens == 0 || sequences == 0 {
        anyhow::bail!("iterative-refinement validation produced no masked tokens");
    }
    let mean_loss = loss_sum / masked_tokens as f64;
    Ok(RefinementMetrics {
        loss: mean_loss,
        perplexity: mean_loss.exp(),
        masked_token_accuracy: correct as f64 / masked_tokens as f64,
        exact_masked_set_rate: exact_sequences as f64 / sequences as f64,
        masked_tokens,
        sequences,
    })
}

fn println_metrics(label: &str, step: usize, metrics: RefinementMetrics) {
    println!(
        "{label}\tstep={step}\tloss={:.6}\tperplexity={:.4}\tmasked_token_accuracy={:.6}\texact_masked_set_rate={:.6}\tmasked_tokens={}\tsequences={}",
        metrics.loss,
        metrics.perplexity,
        metrics.masked_token_accuracy,
        metrics.exact_masked_set_rate,
        metrics.masked_tokens,
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
    parent_checkpoint: &Path,
    parent_completed_steps: usize,
    global_step: usize,
    best_validation_loss: f64,
) -> IterativeRefinementCheckpointMetadata {
    IterativeRefinementCheckpointMetadata {
        version: 1,
        objective: FOUNDATION_ITERATIVE_REFINEMENT_OBJECTIVE_V01315.into(),
        parameter_namespace: FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315.into(),
        corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
        benchmark_manifest_fingerprint: format!(
            "fnv1a64:{:016x}",
            benchmark.manifest_fingerprint()
        ),
        train_dataset_fingerprint: format!("fnv1a64:{train_fingerprint:016x}"),
        validation_dataset_fingerprint: format!("fnv1a64:{validation_fingerprint:016x}"),
        usable_train_pairs,
        usable_validation_pairs,
        train_steps,
        batch_size,
        validation_batches,
        seed,
        learning_rate,
        max_gradient_norm,
        mask_fraction: FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315,
        refinement_rounds: FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315,
        seed_hypotheses: FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315,
        replacement_topk: FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315,
        replacement_beam_width: FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315,
        parent_unified_checkpoint: parent_checkpoint.display().to_string(),
        parent_unified_completed_steps: parent_completed_steps,
        global_step,
        best_validation_loss,
        inverse_config: config.clone(),
    }
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    metadata: &IterativeRefinementCheckpointMetadata,
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
    let mut rng = RefinementRng::new(seed);
    (0..batch_size)
        .map(|_| indices[rng.next_u64() as usize % indices.len()])
        .collect()
}

fn deterministic_subset(indices: &[usize], requested: usize, seed: u64) -> Vec<usize> {
    let mut ranked = indices
        .iter()
        .copied()
        .map(|index| (mix64(seed ^ index as u64), index))
        .collect::<Vec<_>>();
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
struct RefinementRng {
    state: u64,
}

impl RefinementRng {
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
