//! Train the first real paired-spectrum peptide diffusion model.
//!
//! This bounded CPU-oriented trainer consumes only leakage-safe benchmark TRAIN
//! records that contain explicit observed/library product m/z values. It never
//! synthesizes peaks from peptide identity and never reads the held-out TEST
//! partition. Validation uses a deterministic fixed subset and reports both
//! random-timestep and maximum-noise x0 denoising metrics.

use anyhow::{Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_diffusion_dataset_fingerprint, foundation_diffusion_length_loss,
    foundation_diffusion_record_fingerprint, foundation_diffusion_x0_loss,
    foundation_spectrum_peptide_alignment_loss, load_foundation_corpus,
    read_foundation_training_run_config, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationDiffusionCollator, FoundationDiffusionConfig,
    FoundationDiffusionVocabulary, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumCollator, FoundationSpectrumConfig, FoundationTrainingRecord,
    PeptideFoundationEncoder, PeptideGraphFeaturizer, PeptideSpectrumDiffusionModel,
    PeptidoformInput, PrecursorContextBatch, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIFFUSION_VOCAB_SIZE,
};
use serde::Serialize;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
struct DiffusionPilotMetadata {
    version: u32,
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
    spectrum_forcing_fraction: f64,
    length_loss_weight: f64,
    alignment_weight: f64,
    alignment_temperature: f64,
    alignment_every: usize,
    initial_diffusion_checkpoint: Option<String>,
    peptide_encoder_checkpoint: Option<String>,
    alignment_target: String,
    alignment_projection_dim: usize,
    best_step: usize,
    best_validation_spectrum_only_loss: f64,
    diffusion: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Copy, Default)]
struct DenoisingMetrics {
    loss: f64,
    length_loss: f64,
    token_accuracy: f64,
    exact_sequence_rate: f64,
    input_match_rate: f64,
    length_accuracy: f64,
    length_mae_tokens: f64,
    active_tokens: usize,
    sequences: usize,
}

#[derive(Debug, Clone, Copy)]
enum CorruptionMode {
    Random,
    MaxNoise,
    SpectrumOnlyMasked,
}

#[derive(Debug, Clone, Copy, Default)]
struct AlignmentMetrics {
    loss: f64,
    retrieval_top1: f64,
    mean_positive_cosine: f64,
    mean_rotated_cosine: f64,
    pairs: usize,
}

#[derive(Debug, Clone)]
struct TeacherEmbeddings {
    pooled: Tensor,
    contrastive: Tensor,
}

struct FrozenPeptideTeacher {
    _varmap: VarMap,
    encoder: PeptideFoundationEncoder,
    contrastive_head: Linear,
    featurizer: PeptideGraphFeaturizer,
    contrastive_dim: usize,
}

impl FrozenPeptideTeacher {
    fn load(
        model_config: redeem_properties::foundation::FoundationConfig,
        checkpoint: &Path,
        device: &Device,
    ) -> Result<Self> {
        let model_path = resolve_model_safetensors(checkpoint);
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let encoder = PeptideFoundationEncoder::new(model_config.clone(), vb.pp("encoder"))?;
        let contrastive_head = linear(
            model_config.model_dim,
            model_config.contrastive_dim,
            vb.pp("heads.contrastive"),
        )?;
        varmap.load(&model_path).with_context(|| {
            format!("failed to load frozen peptide encoder/contrastive head from {model_path:?}")
        })?;
        let contrastive_dim = model_config.contrastive_dim;
        let featurizer = PeptideGraphFeaturizer::new(model_config)?;
        Ok(Self {
            _varmap: varmap,
            encoder,
            contrastive_head,
            featurizer,
            contrastive_dim,
        })
    }

    fn encode_records(
        &self,
        records: &[&FoundationTrainingRecord],
        device: &Device,
    ) -> Result<TeacherEmbeddings> {
        let peptides: Vec<PeptidoformInput> = records
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect();
        let batch = self.featurizer.featurize(&peptides, device)?;
        let pooled = self
            .encoder
            .forward_t(&batch, false)?
            .peptide_embedding
            .detach();
        let contrastive = self.contrastive_head.forward(&pooled)?.detach();
        Ok(TeacherEmbeddings {
            pooled,
            contrastive,
        })
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 14 {
        anyhow::bail!(
            "usage: foundation_train_diffusion FOUNDATION_TRAINING.yaml OUTPUT_DIR [train_steps=1000] [batch_size=16] [validation_batches=32] [seed=20260901] [spectrum_forcing_fraction=0.5] [length_loss_weight=0.1] [initial_diffusion_checkpoint=none] [peptide_encoder_checkpoint=none] [alignment_weight=0.0] [alignment_temperature=0.07] [alignment_every=4]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let train_steps = parse_or(&args, 3, 1_000usize)?;
    let batch_size = parse_or(&args, 4, 16usize)?;
    let validation_batches = parse_or(&args, 5, 32usize)?;
    let seed = parse_or(&args, 6, 20_260_901u64)?;
    let spectrum_forcing_fraction = parse_or(&args, 7, 0.5f64)?;
    let length_loss_weight = parse_or(&args, 8, 0.1f64)?;
    let initial_diffusion_checkpoint = optional_path(&args, 9);
    let peptide_encoder_checkpoint = optional_path(&args, 10);
    let alignment_weight = parse_or(&args, 11, 0.0f64)?;
    let alignment_temperature = parse_or(&args, 12, 0.07f64)?;
    let alignment_every = parse_or(&args, 13, 4usize)?;
    if train_steps == 0 || batch_size == 0 || validation_batches == 0 {
        anyhow::bail!("train_steps, batch_size, and validation_batches must all be positive");
    }
    if !(0.0..=1.0).contains(&spectrum_forcing_fraction) {
        anyhow::bail!("spectrum_forcing_fraction must be in [0, 1]");
    }
    if !(length_loss_weight >= 0.0 && length_loss_weight.is_finite()) {
        anyhow::bail!("length_loss_weight must be finite and non-negative");
    }
    if !(alignment_weight >= 0.0 && alignment_weight.is_finite()) {
        anyhow::bail!("alignment_weight must be finite and non-negative");
    }
    if !(alignment_temperature > 0.0 && alignment_temperature.is_finite()) {
        anyhow::bail!("alignment_temperature must be finite and positive");
    }
    if alignment_every == 0 {
        anyhow::bail!("alignment_every must be positive");
    }
    if alignment_weight > 0.0 && peptide_encoder_checkpoint.is_none() {
        anyhow::bail!("alignment_weight > 0 requires peptide_encoder_checkpoint");
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let config = FoundationDiffusionConfig {
        max_tokens: 80,
        model_dim: 96,
        num_attention_heads: 4,
        feed_forward_dim: 192,
        spectrum_layers: 2,
        decoder_layers: 2,
        dropout: 0.05,
        diffusion_steps: 20,
        beta_start: 0.02,
        beta_end: 0.35,
        spectrum: FoundationSpectrumConfig {
            max_peaks: 64,
            ..FoundationSpectrumConfig::default()
        },
        precursor_mass_tolerance_da: 0.05,
    };
    config.validate().map_err(anyhow::Error::msg)?;

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
    if train_indices.len() < batch_size {
        anyhow::bail!("only {} usable diffusion train pairs", train_indices.len());
    }
    if validation_indices.len() < batch_size {
        anyhow::bail!(
            "only {} usable diffusion validation pairs",
            validation_indices.len()
        );
    }

    let train_fingerprint =
        foundation_diffusion_dataset_fingerprint(&corpus.records, &train_indices)?;
    let validation_fingerprint =
        foundation_diffusion_dataset_fingerprint(&corpus.records, &validation_indices)?;
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
    println!("spectrum_layers\t{}", config.spectrum_layers);
    println!("decoder_layers\t{}", config.decoder_layers);
    println!("spectrum_max_peaks\t{}", config.spectrum.max_peaks);
    println!(
        "spectrum_peak_feature_dim\t{}",
        config.spectrum.peak_feature_dim
    );
    println!("max_tokens\t{}", config.max_tokens);
    println!("diffusion_steps\t{}", config.diffusion_steps);
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("validation_batches\t{validation_batches}");
    println!("seed\t{seed}");
    println!("spectrum_forcing_fraction\t{spectrum_forcing_fraction}");
    println!("length_loss_weight\t{length_loss_weight}");
    println!("alignment_weight\t{alignment_weight}");
    println!("alignment_temperature\t{alignment_temperature}");
    println!("alignment_every\t{alignment_every}");
    println!(
        "initial_diffusion_checkpoint\t{}",
        initial_diffusion_checkpoint
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".into())
    );
    println!(
        "peptide_encoder_checkpoint\t{}",
        peptide_encoder_checkpoint
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".into())
    );
    println!(
        "sampled_training_pairs\t{}",
        train_steps.saturating_mul(batch_size)
    );
    println!(
        "nominal_train_pair_exposure_fraction\t{:.8}",
        train_steps.saturating_mul(batch_size) as f64 / train_indices.len() as f64
    );

    let baseline = token_baselines(
        &corpus.records,
        &train_indices,
        &validation_indices,
        vocabulary,
        config.max_tokens,
    )?;
    println!(
        "baseline\tuniform_clean_classes\tloss={:.6}\tperplexity={:.4}\ttoken_accuracy={:.6}",
        baseline.uniform_loss,
        baseline.uniform_loss.exp(),
        baseline.uniform_accuracy
    );
    println!("baseline\ttrain_unigram\tloss={:.6}\tperplexity={:.4}\ttoken_accuracy={:.6}\tmode_token={}", baseline.unigram_loss, baseline.unigram_loss.exp(), baseline.unigram_accuracy, baseline.mode_token);
    let length_baseline = length_baseline(
        &corpus.records,
        &train_indices,
        &validation_indices,
        vocabulary,
        config.max_tokens,
    )?;
    println!(
        "baseline\ttrain_length_unigram\tloss={:.6}\tperplexity={:.4}\tmode_active_tokens={}\tmode_accuracy={:.6}\tmode_mae_tokens={:.4}",
        length_baseline.unigram_loss,
        length_baseline.unigram_loss.exp(),
        length_baseline.mode_active_tokens,
        length_baseline.mode_accuracy,
        length_baseline.mode_mae_tokens,
    );

    fs::create_dir_all(&output_root)?;
    write_pair_manifest(
        &output_root.join("train_manifest.tsv"),
        "train",
        &train_indices,
        &corpus.records,
        &corpus.provenance,
    )?;
    write_pair_manifest(
        &output_root.join("validation_manifest.tsv"),
        "validation",
        &validation_indices,
        &corpus.records,
        &corpus.provenance,
    )?;
    let initial_dir = output_root.join("initial");
    fs::create_dir_all(&initial_dir)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumDiffusionModel::new(config.clone(), vb)?;

    let peptide_teacher = peptide_encoder_checkpoint
        .as_ref()
        .map(|path| FrozenPeptideTeacher::load(run.model.clone(), path, &device))
        .transpose()?;
    let alignment_projection = if let Some(teacher) = peptide_teacher.as_ref() {
        let alignment_vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Some(linear(
            config.model_dim,
            teacher.contrastive_dim,
            alignment_vb.pp("alignment.spectrum_projection"),
        )?)
    } else {
        None
    };

    if let Some(checkpoint) = initial_diffusion_checkpoint.as_ref() {
        let model_path = resolve_model_safetensors(checkpoint);
        let (loaded, missing_alignment) =
            load_matching_diffusion_checkpoint(&varmap, &model_path, &device)?;
        println!(
            "warm_start	loaded_variables={loaded}	missing_alignment_projection_variables={missing_alignment}"
        );
    }
    varmap.save(initial_dir.join("model.safetensors"))?;

    let learning_rate = 1.0e-4;
    let max_gradient_norm = 1.0;
    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            ..FoundationAdamWConfig::default()
        },
    )?;
    let diffusion_collator = FoundationDiffusionCollator::new(config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;

    let validation_selection = deterministic_subset(
        &validation_indices,
        validation_batches * batch_size,
        seed ^ 0xa7f4_39d1_2e68_5c0b,
    );

    if let Some(teacher) = peptide_teacher.as_ref() {
        let geometry_count = validation_selection.len().min(256);
        let geometry_records: Vec<&FoundationTrainingRecord> = validation_selection
            .iter()
            .take(geometry_count)
            .map(|&index| &corpus.records[index])
            .collect();
        let embeddings = teacher.encode_records(&geometry_records, &device)?;
        let raw_geometry = embedding_geometry(&embeddings.pooled)?;
        let contrastive_geometry = embedding_geometry(&embeddings.contrastive)?;
        println!(
            "teacher_geometry\traw_dim={}\traw_mean_norm={:.6}\traw_mean_rotated_cosine={:.6}\traw_mean_dimension_std={:.6}\tcontrastive_dim={}\tcontrastive_mean_norm={:.6}\tcontrastive_mean_rotated_cosine={:.6}\tcontrastive_mean_dimension_std={:.6}\tpairs={}",
            embeddings.pooled.dims2()?.1,
            raw_geometry.mean_norm,
            raw_geometry.mean_rotated_cosine,
            raw_geometry.mean_dimension_std,
            embeddings.contrastive.dims2()?.1,
            contrastive_geometry.mean_norm,
            contrastive_geometry.mean_rotated_cosine,
            contrastive_geometry.mean_dimension_std,
            geometry_count,
        );
        println!(
            "alignment_target\tforward_contrastive_projection\tprojection_dim={}",
            teacher.contrastive_dim
        );
    }

    let mut best_spectrum_only = f64::INFINITY;
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
        let forcing_draw = mix64(seed ^ (step as u64).wrapping_mul(0xd6e8_feb8_6659_fd93));
        let force_spectrum = (forcing_draw as f64 / u64::MAX as f64) < spectrum_forcing_fraction;
        let corruption = if force_spectrum {
            CorruptionMode::SpectrumOnlyMasked
        } else {
            CorruptionMode::Random
        };
        let packed = collate_records(
            &records,
            &config,
            &diffusion_collator,
            &spectrum_collator,
            corruption,
            false,
            seed ^ step as u64,
            &device,
        )?;
        let output =
            model.forward_t(&packed.diffusion, &packed.spectrum, &packed.precursor, true)?;
        let x0_loss = foundation_diffusion_x0_loss(&output, &packed.diffusion)?;
        let length_loss = foundation_diffusion_length_loss(&output, &packed.diffusion)?;
        let weighted_length = length_loss.affine(length_loss_weight, 0.0)?;
        let mut loss = (&x0_loss + &weighted_length)?;
        let use_alignment = alignment_weight > 0.0 && step % alignment_every == 0;
        let mut alignment_loss_value = None;
        if use_alignment {
            let teacher = peptide_teacher
                .as_ref()
                .expect("alignment configuration validated before training");
            let projection = alignment_projection
                .as_ref()
                .expect("alignment projection constructed with peptide teacher");
            let teacher_embeddings = teacher.encode_records(&records, &device)?;
            let spectrum_projection = projection.forward(&output.spectrum_embedding)?;
            let alignment_loss = foundation_spectrum_peptide_alignment_loss(
                &spectrum_projection,
                &teacher_embeddings.contrastive,
                alignment_temperature,
            )?;
            if step == 1 {
                let gradients = alignment_loss.backward()?;
                let projection_gradient_norm =
                    gradient_norm_for_prefix(&varmap, &gradients, "alignment.spectrum_projection")?;
                let spectrum_encoder_gradient_norm =
                    gradient_norm_for_prefix(&varmap, &gradients, "spectrum_encoder")?;
                println!(
                    "alignment_gradient_probe\tprojection_gradient_norm={projection_gradient_norm:.8}\tspectrum_encoder_gradient_norm={spectrum_encoder_gradient_norm:.8}"
                );
            }
            alignment_loss_value = Some(f64::from(alignment_loss.to_scalar::<f32>()?));
            loss = (&loss + &alignment_loss.affine(alignment_weight, 0.0)?)?;
        }
        let x0_loss_value = f64::from(x0_loss.to_scalar::<f32>()?);
        let length_loss_value = f64::from(length_loss.to_scalar::<f32>()?);
        let total_loss_value = f64::from(loss.to_scalar::<f32>()?);
        let optimizer_step = optimizer.backward_step(&loss, Some(max_gradient_norm))?;

        if step == 1 || step % 10 == 0 || step == train_steps {
            let mode = if force_spectrum {
                "spectrum_only_masked"
            } else {
                "random_t"
            };
            println!(
                "train\tstep={step}\tmode={mode}\tloss={x0_loss_value:.6}\tlength_loss={length_loss_value:.6}\talignment_loss={}\ttotal_loss={total_loss_value:.6}\tperplexity={:.4}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                alignment_loss_value
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "na".into()),
                x0_loss_value.exp(),
                optimizer_step.gradient_norm,
                optimizer_step.gradient_scale,
            );
        }

        if step % eval_every == 0 || step == train_steps {
            let random_metrics = evaluate(
                &model,
                &corpus.records,
                &validation_selection,
                batch_size,
                &config,
                &diffusion_collator,
                &spectrum_collator,
                CorruptionMode::Random,
                false,
                seed ^ 0x48e2_7c61_934d_ab05,
                &device,
            )?;
            let high_noise_metrics = evaluate(
                &model,
                &corpus.records,
                &validation_selection,
                batch_size,
                &config,
                &diffusion_collator,
                &spectrum_collator,
                CorruptionMode::MaxNoise,
                false,
                seed ^ 0xd1b5_4a32_07c9_ef86,
                &device,
            )?;
            let spectrum_only_metrics = evaluate(
                &model,
                &corpus.records,
                &validation_selection,
                batch_size,
                &config,
                &diffusion_collator,
                &spectrum_collator,
                CorruptionMode::SpectrumOnlyMasked,
                false,
                seed ^ 0x278d_e6e4_f126_c7b5,
                &device,
            )?;
            let shuffled_spectrum_metrics = evaluate(
                &model,
                &corpus.records,
                &validation_selection,
                batch_size,
                &config,
                &diffusion_collator,
                &spectrum_collator,
                CorruptionMode::SpectrumOnlyMasked,
                true,
                seed ^ 0x8c34_ef51_70da_2bb1,
                &device,
            )?;
            print_validation("random_t", step, random_metrics);
            print_validation("max_noise_t", step, high_noise_metrics);
            print_validation("spectrum_only_masked", step, spectrum_only_metrics);
            print_validation("spectrum_only_shuffled", step, shuffled_spectrum_metrics);
            println!(
                "validation_ablation\tstep={step}\tspectrum_loss_delta={:.6}\tspectrum_token_accuracy_delta={:.6}\tlength_accuracy_delta={:.6}",
                shuffled_spectrum_metrics.loss - spectrum_only_metrics.loss,
                spectrum_only_metrics.token_accuracy - shuffled_spectrum_metrics.token_accuracy,
                spectrum_only_metrics.length_accuracy - shuffled_spectrum_metrics.length_accuracy,
            );

            if let (Some(teacher), Some(projection)) =
                (peptide_teacher.as_ref(), alignment_projection.as_ref())
            {
                let matched_alignment = evaluate_alignment(
                    &model,
                    projection,
                    teacher,
                    &corpus.records,
                    &validation_selection,
                    batch_size,
                    &config,
                    &diffusion_collator,
                    &spectrum_collator,
                    false,
                    alignment_temperature,
                    &device,
                )?;
                let shuffled_alignment = evaluate_alignment(
                    &model,
                    projection,
                    teacher,
                    &corpus.records,
                    &validation_selection,
                    batch_size,
                    &config,
                    &diffusion_collator,
                    &spectrum_collator,
                    true,
                    alignment_temperature,
                    &device,
                )?;
                print_alignment("matched", step, matched_alignment);
                print_alignment("shuffled", step, shuffled_alignment);
                println!(
                    "validation_alignment_ablation\tstep={step}\tloss_delta={:.6}\tretrieval_top1_delta={:.6}\tpositive_cosine_delta={:.6}",
                    shuffled_alignment.loss - matched_alignment.loss,
                    matched_alignment.retrieval_top1 - shuffled_alignment.retrieval_top1,
                    matched_alignment.mean_positive_cosine
                        - shuffled_alignment.mean_positive_cosine,
                );
            }

            if spectrum_only_metrics.loss < best_spectrum_only {
                best_spectrum_only = spectrum_only_metrics.loss;
                best_step = step;
                save_checkpoint(
                    &output_root.join("best"),
                    &varmap,
                    &optimizer,
                    &DiffusionPilotMetadata {
                        version: 4,
                        corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
                        benchmark_manifest_fingerprint: format!(
                            "fnv1a64:{:016x}",
                            benchmark.manifest_fingerprint()
                        ),
                        train_diffusion_fingerprint: format!("fnv1a64:{train_fingerprint:016x}"),
                        validation_diffusion_fingerprint: format!(
                            "fnv1a64:{validation_fingerprint:016x}"
                        ),
                        usable_train_pairs: train_indices.len(),
                        usable_validation_pairs: validation_indices.len(),
                        train_steps,
                        batch_size,
                        validation_batches,
                        seed,
                        learning_rate,
                        max_gradient_norm,
                        spectrum_forcing_fraction,
                        length_loss_weight,
                        alignment_weight,
                        alignment_temperature,
                        alignment_every,
                        initial_diffusion_checkpoint: initial_diffusion_checkpoint
                            .as_ref()
                            .map(|path| path.display().to_string()),
                        peptide_encoder_checkpoint: peptide_encoder_checkpoint
                            .as_ref()
                            .map(|path| path.display().to_string()),
                        alignment_target: if peptide_teacher.is_some() {
                            "forward_contrastive_projection".into()
                        } else {
                            "none".into()
                        },
                        alignment_projection_dim: peptide_teacher
                            .as_ref()
                            .map(|teacher| teacher.contrastive_dim)
                            .unwrap_or(0),
                        best_step,
                        best_validation_spectrum_only_loss: best_spectrum_only,
                        diffusion: config.clone(),
                    },
                )?;
            }

            save_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &optimizer,
                &DiffusionPilotMetadata {
                    version: 4,
                    corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
                    benchmark_manifest_fingerprint: format!(
                        "fnv1a64:{:016x}",
                        benchmark.manifest_fingerprint()
                    ),
                    train_diffusion_fingerprint: format!("fnv1a64:{train_fingerprint:016x}"),
                    validation_diffusion_fingerprint: format!(
                        "fnv1a64:{validation_fingerprint:016x}"
                    ),
                    usable_train_pairs: train_indices.len(),
                    usable_validation_pairs: validation_indices.len(),
                    train_steps,
                    batch_size,
                    validation_batches,
                    seed,
                    learning_rate,
                    max_gradient_norm,
                    spectrum_forcing_fraction,
                    length_loss_weight,
                    alignment_weight,
                    alignment_temperature,
                    alignment_every,
                    initial_diffusion_checkpoint: initial_diffusion_checkpoint
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    peptide_encoder_checkpoint: peptide_encoder_checkpoint
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    alignment_target: if peptide_teacher.is_some() {
                        "forward_contrastive_projection".into()
                    } else {
                        "none".into()
                    },
                    alignment_projection_dim: peptide_teacher
                        .as_ref()
                        .map(|teacher| teacher.contrastive_dim)
                        .unwrap_or(0),
                    best_step,
                    best_validation_spectrum_only_loss: best_spectrum_only,
                    diffusion: config.clone(),
                },
            )?;
        }
    }

    println!("best_step\t{best_step}");
    println!("best_validation_spectrum_only_loss\t{best_spectrum_only:.8}");
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

fn resolve_model_safetensors(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_path_buf()
    }
}

/// Load every checkpoint tensor that matches the current diffusion VarMap.
///
/// Historical v0.11.3 checkpoints do not contain the v0.11.8 spectrum
/// alignment projection, so those two variables are allowed to remain freshly
/// initialized. Any other missing current-model variable is treated as an
/// incompatible warm start.
fn load_matching_diffusion_checkpoint(
    varmap: &VarMap,
    path: &Path,
    device: &Device,
) -> Result<(usize, usize)> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("failed to read diffusion checkpoint {path:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("diffusion VarMap lock poisoned"))?;
    let mut loaded = 0usize;
    let mut missing_alignment = 0usize;
    let mut missing_required = Vec::<String>::new();
    for (name, variable) in data.iter() {
        let Some(checkpoint_tensor) = tensors.get(name) else {
            if name.starts_with("alignment.spectrum_projection.") {
                missing_alignment += 1;
            } else {
                missing_required.push(name.clone());
            }
            continue;
        };
        if variable.as_tensor().dims() != checkpoint_tensor.dims() {
            anyhow::bail!(
                "diffusion warm-start shape mismatch for '{name}': current {:?}, checkpoint {:?}",
                variable.as_tensor().dims(),
                checkpoint_tensor.dims()
            );
        }
        variable.set(checkpoint_tensor)?;
        loaded += 1;
    }
    drop(data);
    if !missing_required.is_empty() {
        anyhow::bail!(
            "diffusion warm start {path:?} is missing required variables: {}",
            missing_required.join(", ")
        );
    }
    Ok((loaded, missing_alignment))
}

fn gradient_norm_for_prefix(varmap: &VarMap, gradients: &GradStore, prefix: &str) -> Result<f64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("diffusion VarMap lock poisoned"))?;
    let mut squared_norm = 0.0f64;
    let mut matched = 0usize;
    for (name, variable) in data.iter() {
        if !name.starts_with(prefix) {
            continue;
        }
        matched += 1;
        if let Some(gradient) = gradients.get(variable) {
            squared_norm += f64::from(gradient.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }
    if matched == 0 {
        anyhow::bail!("alignment gradient probe matched no variables for prefix '{prefix}'");
    }
    Ok(squared_norm.sqrt())
}

fn optional_path(args: &[String], index: usize) -> Option<PathBuf> {
    args.get(index).and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("none"))
            .then(|| PathBuf::from(trimmed))
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_alignment(
    model: &PeptideSpectrumDiffusionModel,
    projection: &Linear,
    teacher: &FrozenPeptideTeacher,
    all_records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    config: &FoundationDiffusionConfig,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    shuffle_spectra: bool,
    temperature: f64,
    device: &Device,
) -> Result<AlignmentMetrics> {
    let mut loss_sum = 0.0f64;
    let mut batches = 0usize;
    let mut retrieval_correct = 0usize;
    let mut positive_sum = 0.0f64;
    let mut rotated_sum = 0.0f64;
    let mut pairs = 0usize;

    for chunk in indices.chunks(batch_size) {
        if chunk.is_empty() {
            continue;
        }
        let records: Vec<&FoundationTrainingRecord> =
            chunk.iter().map(|&index| &all_records[index]).collect();
        let packed = collate_records(
            &records,
            config,
            diffusion_collator,
            spectrum_collator,
            CorruptionMode::SpectrumOnlyMasked,
            shuffle_spectra,
            0x73d1_f581_2a94_bc06 ^ pairs as u64,
            device,
        )?;
        let output = model.forward_t(
            &packed.diffusion,
            &packed.spectrum,
            &packed.precursor,
            false,
        )?;
        let teacher_embeddings = teacher.encode_records(&records, device)?;
        let spectrum_projection = projection.forward(&output.spectrum_embedding)?;
        let loss = foundation_spectrum_peptide_alignment_loss(
            &spectrum_projection,
            &teacher_embeddings.contrastive,
            temperature,
        )?;
        loss_sum += f64::from(loss.to_scalar::<f32>()?);
        batches += 1;

        let spectrum = spectrum_projection.to_vec2::<f32>()?;
        let peptide = teacher_embeddings.contrastive.to_vec2::<f32>()?;
        for row in 0..spectrum.len() {
            let mut best_index = 0usize;
            let mut best_cosine = f64::NEG_INFINITY;
            for candidate in 0..peptide.len() {
                let cosine = cosine_similarity(&spectrum[row], &peptide[candidate]);
                if cosine > best_cosine {
                    best_cosine = cosine;
                    best_index = candidate;
                }
            }
            retrieval_correct += usize::from(best_index == row);
            positive_sum += cosine_similarity(&spectrum[row], &peptide[row]);
            let rotated = (row + 1) % peptide.len();
            rotated_sum += cosine_similarity(&spectrum[row], &peptide[rotated]);
            pairs += 1;
        }
    }

    if batches == 0 || pairs == 0 {
        anyhow::bail!("alignment validation selected no peptide-spectrum pairs");
    }
    Ok(AlignmentMetrics {
        loss: loss_sum / batches as f64,
        retrieval_top1: retrieval_correct as f64 / pairs as f64,
        mean_positive_cosine: positive_sum / pairs as f64,
        mean_rotated_cosine: rotated_sum / pairs as f64,
        pairs,
    })
}

fn print_alignment(label: &str, step: usize, metrics: AlignmentMetrics) {
    println!(
        "validation_alignment\tmode={label}\tstep={step}\tloss={:.6}\tretrieval_top1={:.6}\tmean_positive_cosine={:.6}\tmean_rotated_cosine={:.6}\tcosine_delta={:.6}\tpairs={}",
        metrics.loss,
        metrics.retrieval_top1,
        metrics.mean_positive_cosine,
        metrics.mean_rotated_cosine,
        metrics.mean_positive_cosine - metrics.mean_rotated_cosine,
        metrics.pairs,
    );
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingGeometry {
    mean_norm: f64,
    mean_rotated_cosine: f64,
    mean_dimension_std: f64,
}

fn embedding_geometry(values: &Tensor) -> Result<EmbeddingGeometry> {
    let rows = values.to_vec2::<f32>()?;
    if rows.is_empty() || rows[0].is_empty() {
        anyhow::bail!("embedding geometry requires a non-empty rank-2 tensor");
    }
    let dim = rows[0].len();
    let mut norm_sum = 0.0f64;
    let mut rotated_sum = 0.0f64;
    let mut means = vec![0.0f64; dim];
    for (row_index, row) in rows.iter().enumerate() {
        let norm = row
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        norm_sum += norm;
        rotated_sum += cosine_similarity(row, &rows[(row_index + 1) % rows.len()]);
        for (index, &value) in row.iter().enumerate() {
            means[index] += f64::from(value);
        }
    }
    for mean in &mut means {
        *mean /= rows.len() as f64;
    }
    let mut variance_sum = vec![0.0f64; dim];
    for row in &rows {
        for (index, &value) in row.iter().enumerate() {
            let delta = f64::from(value) - means[index];
            variance_sum[index] += delta * delta;
        }
    }
    let mean_dimension_std = variance_sum
        .into_iter()
        .map(|sum| (sum / rows.len() as f64).sqrt())
        .sum::<f64>()
        / dim as f64;
    Ok(EmbeddingGeometry {
        mean_norm: norm_sum / rows.len() as f64,
        mean_rotated_cosine: rotated_sum / rows.len() as f64,
        mean_dimension_std,
    })
}

fn cosine_similarity(first: &[f32], second: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut first_norm = 0.0f64;
    let mut second_norm = 0.0f64;
    for (&left, &right) in first.iter().zip(second) {
        let left = left as f64;
        let right = right as f64;
        dot += left * right;
        first_norm += left * left;
        second_norm += right * right;
    }
    let denominator = (first_norm * second_norm).sqrt();
    if denominator > 0.0 {
        dot / denominator
    } else {
        0.0
    }
}

struct PackedBatch {
    diffusion: redeem_properties::foundation::FoundationDiffusionBatch,
    spectrum: redeem_properties::foundation::FoundationSpectrumBatch,
    precursor: PrecursorContextBatch,
}

#[allow(clippy::too_many_arguments)]
fn collate_records(
    records: &[&FoundationTrainingRecord],
    config: &FoundationDiffusionConfig,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    corruption: CorruptionMode,
    shuffle_spectra: bool,
    seed: u64,
    device: &Device,
) -> Result<PackedBatch> {
    let spectrum_records: Vec<&FoundationTrainingRecord> = if shuffle_spectra && records.len() > 1 {
        (0..records.len())
            .map(|index| records[(index + 1) % records.len()])
            .collect()
    } else {
        records.to_vec()
    };
    let spectra: Vec<FoundationSpectrum> = spectrum_records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record).ok_or_else(|| {
                anyhow::anyhow!("selected diffusion record unexpectedly lacks observed spectrum")
            })
        })
        .collect::<Result<_>>()?;
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let diffusion = match corruption {
        CorruptionMode::Random => {
            diffusion_collator.collate_random_timesteps(&peptides, seed, device)?
        }
        CorruptionMode::MaxNoise => {
            let timesteps = vec![config.diffusion_steps; peptides.len()];
            diffusion_collator.collate(&peptides, &timesteps, seed, device)?
        }
        CorruptionMode::SpectrumOnlyMasked => {
            diffusion_collator.collate_all_masked(&peptides, config.diffusion_steps, device)?
        }
    };
    let spectrum = spectrum_collator.collate(&spectra, device)?;
    let precursor = precursor_context(records, config, device)?;
    Ok(PackedBatch {
        diffusion,
        spectrum,
        precursor,
    })
}

fn precursor_context(
    records: &[&FoundationTrainingRecord],
    _config: &FoundationDiffusionConfig,
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

fn deterministic_batch(indices: &[usize], batch_size: usize, seed: u64) -> Vec<usize> {
    let mut rng = PilotRng::new(seed);
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

#[allow(clippy::too_many_arguments)]
fn evaluate(
    model: &PeptideSpectrumDiffusionModel,
    all_records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    config: &FoundationDiffusionConfig,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    corruption: CorruptionMode,
    shuffle_spectra: bool,
    seed: u64,
    device: &Device,
) -> Result<DenoisingMetrics> {
    let mut loss_sum = 0.0f64;
    let mut length_loss_sum = 0.0f64;
    let mut batches = 0usize;
    let mut correct_tokens = 0usize;
    let mut input_matches = 0usize;
    let mut active_tokens = 0usize;
    let mut exact_sequences = 0usize;
    let mut correct_lengths = 0usize;
    let mut absolute_length_error = 0usize;
    let mut sequences = 0usize;

    for (batch_index, chunk) in indices.chunks(batch_size).enumerate() {
        if chunk.is_empty() {
            continue;
        }
        let records: Vec<&FoundationTrainingRecord> =
            chunk.iter().map(|&index| &all_records[index]).collect();
        let packed = collate_records(
            &records,
            config,
            diffusion_collator,
            spectrum_collator,
            corruption,
            shuffle_spectra,
            seed ^ batch_index as u64,
            device,
        )?;
        let output = model.forward_t(
            &packed.diffusion,
            &packed.spectrum,
            &packed.precursor,
            false,
        )?;
        let loss = foundation_diffusion_x0_loss(&output, &packed.diffusion)?;
        let length_loss = foundation_diffusion_length_loss(&output, &packed.diffusion)?;
        loss_sum += f64::from(loss.to_scalar::<f32>()?);
        length_loss_sum += f64::from(length_loss.to_scalar::<f32>()?);
        batches += 1;

        let logits = output.token_logits.to_vec3::<f32>()?;
        let length_logits = output.length_logits.to_vec2::<f32>()?;
        let length_targets = packed.diffusion.length_targets.to_vec1::<u32>()?;
        let clean = packed.diffusion.clean_tokens.to_vec2::<u32>()?;
        let noisy = packed.diffusion.noisy_tokens.to_vec2::<u32>()?;
        let mask = packed.diffusion.token_mask.to_vec2::<f32>()?;
        for batch_row in 0..logits.len() {
            let predicted_length_class = argmax(&length_logits[batch_row]);
            let target_length_class = length_targets[batch_row] as usize;
            if predicted_length_class == target_length_class {
                correct_lengths += 1;
            }
            absolute_length_error += predicted_length_class.abs_diff(target_length_class);

            let mut sequence_exact = true;
            let mut has_active = false;
            for position in 0..logits[batch_row].len() {
                if mask[batch_row][position] <= 0.0 {
                    continue;
                }
                has_active = true;
                active_tokens += 1;
                if noisy[batch_row][position] == clean[batch_row][position] {
                    input_matches += 1;
                }
                let predicted = argmax(&logits[batch_row][position]) as u32;
                if predicted == clean[batch_row][position] {
                    correct_tokens += 1;
                } else {
                    sequence_exact = false;
                }
            }
            if has_active {
                sequences += 1;
                if sequence_exact {
                    exact_sequences += 1;
                }
            }
        }
    }

    if batches == 0 || active_tokens == 0 || sequences == 0 {
        anyhow::bail!("diffusion validation produced no active batches/tokens");
    }
    Ok(DenoisingMetrics {
        loss: loss_sum / batches as f64,
        length_loss: length_loss_sum / batches as f64,
        token_accuracy: correct_tokens as f64 / active_tokens as f64,
        exact_sequence_rate: exact_sequences as f64 / sequences as f64,
        input_match_rate: input_matches as f64 / active_tokens as f64,
        length_accuracy: correct_lengths as f64 / sequences as f64,
        length_mae_tokens: absolute_length_error as f64 / sequences as f64,
        active_tokens,
        sequences,
    })
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn print_validation(label: &str, step: usize, metrics: DenoisingMetrics) {
    println!(
        "validation\tmode={label}\tstep={step}\tloss={:.6}\tperplexity={:.4}\ttoken_accuracy={:.6}\texact_sequence_rate={:.6}\tinput_match_rate={:.6}\tlength_loss={:.6}\tlength_accuracy={:.6}\tlength_mae_tokens={:.4}\tactive_tokens={}\tsequences={}",
        metrics.loss,
        metrics.loss.exp(),
        metrics.token_accuracy,
        metrics.exact_sequence_rate,
        metrics.input_match_rate,
        metrics.length_loss,
        metrics.length_accuracy,
        metrics.length_mae_tokens,
        metrics.active_tokens,
        metrics.sequences,
    );
}

fn save_checkpoint(
    dir: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &DiffusionPilotMetadata,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    varmap.save(dir.join("model.safetensors"))?;
    optimizer.save_safetensors(dir.join("optimizer.safetensors"))?;
    fs::write(dir.join("metadata.yaml"), serde_yaml::to_string(metadata)?)?;
    Ok(())
}

fn write_pair_manifest(
    path: &Path,
    partition: &str,
    indices: &[usize],
    records: &[FoundationTrainingRecord],
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut output = BufWriter::new(file);
    writeln!(
        output,
        "record_index\tdiffusion_record_fingerprint\tpartition\tsource_id\tsequence\tobserved_peaks"
    )?;
    for &index in indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow::anyhow!("diffusion manifest record index {index} is out of bounds")
        })?;
        let source = provenance.get(index).ok_or_else(|| {
            anyhow::anyhow!("diffusion manifest provenance index {index} is out of bounds")
        })?;
        let observed_peaks = FoundationSpectrum::from_training_record(record)
            .map(|spectrum| spectrum.peaks.len())
            .unwrap_or(0);
        writeln!(
            output,
            "{index}\t{:016x}\t{partition}\t{}\t{}\t{observed_peaks}",
            foundation_diffusion_record_fingerprint(record),
            source.source_id,
            record.peptidoform.sequence,
        )?;
    }
    output.flush()?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct TokenBaselineMetrics {
    uniform_loss: f64,
    uniform_accuracy: f64,
    unigram_loss: f64,
    unigram_accuracy: f64,
    mode_token: usize,
}

fn token_baselines(
    records: &[FoundationTrainingRecord],
    train_indices: &[usize],
    validation_indices: &[usize],
    vocabulary: FoundationDiffusionVocabulary,
    max_tokens: usize,
) -> Result<TokenBaselineMetrics> {
    let mut train_counts = vec![0u64; FOUNDATION_DIFFUSION_VOCAB_SIZE];
    let mut validation_counts = vec![0u64; FOUNDATION_DIFFUSION_VOCAB_SIZE];
    accumulate_token_counts(
        records,
        train_indices,
        vocabulary,
        max_tokens,
        &mut train_counts,
    )?;
    accumulate_token_counts(
        records,
        validation_indices,
        vocabulary,
        max_tokens,
        &mut validation_counts,
    )?;

    // Clean x0 targets never contain PAD or MASK.
    let active_classes = FOUNDATION_DIFFUSION_VOCAB_SIZE - 2;
    let uniform_loss = (active_classes as f64).ln();
    let uniform_accuracy = 1.0 / active_classes as f64;

    let train_total: u64 = train_counts.iter().sum();
    let validation_total: u64 = validation_counts.iter().sum();
    if train_total == 0 || validation_total == 0 {
        anyhow::bail!("diffusion token baseline contains no active train/validation tokens");
    }
    let smoothed_total = train_total as f64 + active_classes as f64;
    let mut unigram_loss = 0.0f64;
    for token in 2..FOUNDATION_DIFFUSION_VOCAB_SIZE {
        let probability = (train_counts[token] as f64 + 1.0) / smoothed_total;
        unigram_loss -= validation_counts[token] as f64 * probability.ln();
    }
    unigram_loss /= validation_total as f64;
    let mode_token = (2..FOUNDATION_DIFFUSION_VOCAB_SIZE)
        .max_by_key(|&token| train_counts[token])
        .unwrap_or(1);
    let unigram_accuracy = validation_counts[mode_token] as f64 / validation_total as f64;

    Ok(TokenBaselineMetrics {
        uniform_loss,
        uniform_accuracy,
        unigram_loss,
        unigram_accuracy,
        mode_token,
    })
}

#[derive(Debug, Clone, Copy)]
struct LengthBaselineMetrics {
    unigram_loss: f64,
    mode_active_tokens: usize,
    mode_accuracy: f64,
    mode_mae_tokens: f64,
}

fn length_baseline(
    records: &[FoundationTrainingRecord],
    train_indices: &[usize],
    validation_indices: &[usize],
    vocabulary: FoundationDiffusionVocabulary,
    max_tokens: usize,
) -> Result<LengthBaselineMetrics> {
    let mut train_counts = vec![0u64; max_tokens];
    let mut validation_counts = vec![0u64; max_tokens];
    accumulate_length_counts(
        records,
        train_indices,
        vocabulary,
        max_tokens,
        &mut train_counts,
    )?;
    accumulate_length_counts(
        records,
        validation_indices,
        vocabulary,
        max_tokens,
        &mut validation_counts,
    )?;

    let train_total: u64 = train_counts.iter().sum();
    let validation_total: u64 = validation_counts.iter().sum();
    if train_total == 0 || validation_total == 0 {
        anyhow::bail!("diffusion length baseline contains no train/validation sequences");
    }

    let smoothed_total = train_total as f64 + max_tokens as f64;
    let mut unigram_loss = 0.0f64;
    for class in 0..max_tokens {
        let probability = (train_counts[class] as f64 + 1.0) / smoothed_total;
        unigram_loss -= validation_counts[class] as f64 * probability.ln();
    }
    unigram_loss /= validation_total as f64;

    let mode_class = (0..max_tokens)
        .max_by_key(|&class| train_counts[class])
        .unwrap_or(0);
    let mode_active_tokens = mode_class + 1;
    let mode_accuracy = validation_counts[mode_class] as f64 / validation_total as f64;
    let mode_mae_tokens = validation_counts
        .iter()
        .enumerate()
        .map(|(class, &count)| {
            let active_tokens = class + 1;
            (active_tokens.abs_diff(mode_active_tokens) as f64) * count as f64
        })
        .sum::<f64>()
        / validation_total as f64;

    Ok(LengthBaselineMetrics {
        unigram_loss,
        mode_active_tokens,
        mode_accuracy,
        mode_mae_tokens,
    })
}

fn accumulate_length_counts(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    vocabulary: FoundationDiffusionVocabulary,
    max_tokens: usize,
    counts: &mut [u64],
) -> Result<()> {
    for &index in indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow::anyhow!("diffusion length baseline index {index} out of bounds")
        })?;
        let tokens = vocabulary
            .encode(&record.peptidoform, max_tokens)
            .map_err(anyhow::Error::msg)?;
        let active_tokens = tokens
            .iter()
            .take_while(|&&token| token != FOUNDATION_DIFFUSION_PAD)
            .count();
        if active_tokens == 0 || active_tokens > max_tokens {
            anyhow::bail!(
                "diffusion length baseline observed invalid active-token length {active_tokens}"
            );
        }
        counts[active_tokens - 1] += 1;
    }
    Ok(())
}

fn accumulate_token_counts(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    vocabulary: FoundationDiffusionVocabulary,
    max_tokens: usize,
    counts: &mut [u64],
) -> Result<()> {
    for &index in indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow::anyhow!("diffusion token baseline index {index} out of bounds")
        })?;
        let tokens = vocabulary
            .encode(&record.peptidoform, max_tokens)
            .map_err(anyhow::Error::msg)?;
        for token in tokens {
            if token == FOUNDATION_DIFFUSION_PAD {
                break;
            }
            let slot = counts
                .get_mut(token as usize)
                .ok_or_else(|| anyhow::anyhow!("diffusion token {token} exceeds vocabulary"))?;
            *slot += 1;
        }
    }
    Ok(())
}

fn parse_or<T>(args: &[String], index: usize, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value.parse::<T>().map_err(|error| {
            anyhow::anyhow!("invalid argument {} ('{}'): {}", index, value, error)
        }),
        None => Ok(default),
    }
}

#[derive(Clone, Copy)]
struct PilotRng(u64);

impl PilotRng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0xa076_1d64_78bd_642f)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = mix64(self.0.wrapping_add(0x9e37_79b9_7f4a_7c15));
        self.0
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
