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
    load_causal_from_diffusion_checkpoint, load_foundation_corpus,
    read_foundation_training_run_config, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCausalCollator, FoundationDiffusionConfig,
    FoundationDiffusionVocabulary, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumCollator, FoundationTrainingRecord, PeptideSpectrumCausalModel,
    PeptidoformInput, PrecursorContextBatch, FOUNDATION_DIFFUSION_EOS,
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
