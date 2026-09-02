//! Bounded joint continuation of the peptide-property and spectrum-to-peptide foundation model.
//!
//! This pilot intentionally keeps the validated forward and inverse architectures
//! unchanged. One VarMap owns all parameters, diffusion/causal modes share their
//! historical inverse namespaces, and a trainable spectrum projection aligns the
//! inverse spectrum representation with the forward peptide contrastive space.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    contrastive_info_nce_loss, foundation_causal_next_token_loss, foundation_diffusion_length_loss,
    foundation_diffusion_x0_loss, foundation_spectrum_peptide_alignment_loss,
    load_foundation_corpus, load_unified_foundation_components, multi_task_loss,
    read_foundation_training_run_config, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCausalCollator, FoundationCheckpointMetadata,
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationDiffusionCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationLossWeights, FoundationPartition, FoundationRegressionNormalization,
    FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator,
    FoundationTargetNormalizationConfig, FoundationTrainingRecord, FoundationTrainingViews,
    PeptideFoundationUnifiedModel, PeptidoformInput, PrecursorContextBatch,
    FOUNDATION_DIFFUSION_VOCAB_SIZE,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct InverseCheckpointMetadata {
    diffusion: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Serialize)]
struct UnifiedPilotMetadata {
    version: u32,
    objective: String,
    schedule: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    forward_checkpoint: String,
    diffusion_checkpoint: String,
    causal_checkpoint: String,
    train_steps: usize,
    batch_size: usize,
    validation_batches: usize,
    seed: u64,
    learning_rate: f64,
    max_gradient_norm: f64,
    diffusion_length_weight: f64,
    alignment_weight: f64,
    alignment_temperature: f64,
    completed_steps: usize,
    forward_config: redeem_properties::foundation::FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Copy, Default)]
struct PropertyMetrics {
    rt_mae_native: Option<f64>,
    rt_rmse_native: Option<f64>,
    ccs_mae_native: Option<f64>,
    ccs_rmse_native: Option<f64>,
    ms2_loss: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct InverseMetrics {
    diffusion_loss: f64,
    diffusion_length_loss: f64,
    diffusion_token_accuracy: f64,
    causal_loss: f64,
    causal_perplexity: f64,
    causal_token_accuracy: f64,
    alignment_loss: f64,
    alignment_retrieval_top1: f64,
    shuffled_alignment_loss: f64,
    shuffled_alignment_retrieval_top1: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 || args.len() > 13 {
        anyhow::bail!(
            "usage: foundation_train_unified FOUNDATION_TRAINING.yaml OUTPUT_DIR FORWARD_CHECKPOINT DIFFUSION_CHECKPOINT CAUSAL_CHECKPOINT [train_steps=30] [batch_size=8] [validation_batches=8] [seed=20260908] [learning_rate=2e-5] [alignment_weight=0.05] [alignment_temperature=0.07]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let forward_checkpoint = PathBuf::from(&args[3]);
    let diffusion_checkpoint = PathBuf::from(&args[4]);
    let causal_checkpoint = PathBuf::from(&args[5]);
    let train_steps = parse_or(&args, 6, 30usize)?;
    let batch_size = parse_or(&args, 7, 8usize)?;
    let validation_batches = parse_or(&args, 8, 8usize)?;
    let seed = parse_or(&args, 9, 20_260_908u64)?;
    let learning_rate = parse_or(&args, 10, 2.0e-5f64)?;
    let alignment_weight = parse_or(&args, 11, 0.05f64)?;
    let alignment_temperature = parse_or(&args, 12, 0.07f64)?;
    let max_gradient_norm = 1.0f64;
    let diffusion_length_weight = 0.1f64;

    if train_steps == 0 || batch_size < 2 || validation_batches == 0 {
        anyhow::bail!("train_steps/validation_batches must be positive and batch_size must be >=2");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("learning_rate must be finite and positive");
    }
    if !(alignment_weight >= 0.0 && alignment_weight.is_finite()) {
        anyhow::bail!("alignment_weight must be finite and non-negative");
    }
    if !(alignment_temperature > 0.0 && alignment_temperature.is_finite()) {
        anyhow::bail!("alignment_temperature must be finite and positive");
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let forward_state = resolve_forward_state(&forward_checkpoint);
    let forward_metadata = FoundationCheckpointMetadata::read_yaml(&forward_state)
        .with_context(|| format!("failed to read corrected forward state {forward_state:?}"))?;
    let forward_model_path = resolve_model_safetensors(&forward_checkpoint);

    let diffusion_metadata = read_inverse_metadata(&diffusion_checkpoint)?;
    let causal_metadata = read_inverse_metadata(&causal_checkpoint)?;
    if diffusion_metadata.diffusion != causal_metadata.diffusion {
        anyhow::bail!(
            "diffusion and causal checkpoints do not describe the same inverse architecture"
        );
    }
    let inverse_config = diffusion_metadata.diffusion;
    inverse_config.validate().map_err(anyhow::Error::msg)?;
    let diffusion_model_path = resolve_model_safetensors(&diffusion_checkpoint);
    let causal_model_path = resolve_model_safetensors(&causal_checkpoint);

    let train_forward_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .collect();
    let validation_forward_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Validation)
        .map(|entry| entry.record_index)
        .collect();
    let vocabulary = FoundationDiffusionVocabulary;
    let train_inverse_indices = usable_inverse_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &inverse_config,
        vocabulary,
    );
    let validation_inverse_indices = usable_inverse_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &inverse_config,
        vocabulary,
    );
    if train_forward_indices.len() < batch_size
        || validation_forward_indices.len() < batch_size
        || train_inverse_indices.len() < batch_size
        || validation_inverse_indices.len() < batch_size
    {
        anyhow::bail!(
            "insufficient records for unified pilot: forward train/val={}/{}, inverse train/val={}/{}",
            train_forward_indices.len(),
            validation_forward_indices.len(),
            train_inverse_indices.len(),
            validation_inverse_indices.len()
        );
    }

    let validation_forward = deterministic_subset(
        &validation_forward_indices,
        validation_batches.saturating_mul(batch_size),
        seed ^ 0x3d13_7f24_559c_81e7,
    );
    let validation_inverse = deterministic_subset(
        &validation_inverse_indices,
        validation_batches.saturating_mul(batch_size),
        seed ^ 0x72a4_c11d_0b95_e683,
    );

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationUnifiedModel::new(
        forward_metadata.model_config.clone(),
        inverse_config.clone(),
        vb,
    )?;
    let warm_start = load_unified_foundation_components(
        &varmap,
        &forward_model_path,
        &diffusion_model_path,
        &causal_model_path,
        &device,
    )?;

    let forward_trainer = &forward_metadata.trainer_config;
    let forward_collator = FoundationCollator::new(
        forward_metadata.model_config.clone(),
        forward_trainer.collator.clone(),
    )?;
    let clean_collator = FoundationCollator::new(
        forward_metadata.model_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: forward_trainer.collator.retention_time_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let diffusion_collator = FoundationDiffusionCollator::new(inverse_config.clone())?;
    let causal_collator = FoundationCausalCollator::new(inverse_config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(inverse_config.spectrum.clone())?;

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: forward_trainer.adam_beta1,
            beta2: forward_trainer.adam_beta2,
            epsilon: forward_trainer.adam_epsilon,
            weight_decay: forward_trainer.weight_decay,
        },
    )?;

    fs::create_dir_all(&output_root)?;
    println!("objective\tunified_forward_diffusion_causal_alignment_v1");
    println!("schedule\tone_optimizer_update=(forward_a+forward_b+diffusion+causal)/4");
    println!("component_batches_per_optimizer_update\t4");
    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!(
        "benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        benchmark.manifest_fingerprint()
    );
    println!("forward_checkpoint\t{}", forward_model_path.display());
    println!("diffusion_checkpoint\t{}", diffusion_model_path.display());
    println!("causal_checkpoint\t{}", causal_model_path.display());
    println!(
        "forward_model_dim\t{}",
        forward_metadata.model_config.model_dim
    );
    println!("inverse_model_dim\t{}", inverse_config.model_dim);
    println!(
        "alignment_dim\t{}",
        forward_metadata.model_config.contrastive_dim
    );
    println!("train_steps\t{train_steps}");
    println!("batch_size\t{batch_size}");
    println!("validation_batches\t{validation_batches}");
    println!("learning_rate\t{learning_rate}");
    println!("alignment_weight\t{alignment_weight}");
    println!("alignment_temperature\t{alignment_temperature}");
    println!("max_gradient_norm\t{max_gradient_norm}");
    println!(
        "warm_start\tforward_loaded={}\tdiffusion_loaded={}\tcausal_overlay_loaded={}\tfresh_alignment={}",
        warm_start.forward_loaded_variables,
        warm_start.diffusion_loaded_variables,
        warm_start.causal_overlay_loaded_variables,
        warm_start.fresh_alignment_variables,
    );
    println!("optimizer_variables\t{}", optimizer.variable_count());

    let metadata = |completed_steps| UnifiedPilotMetadata {
        version: 1,
        objective: "unified_forward_diffusion_causal_alignment_v1".into(),
        schedule: "one_optimizer_update=(forward_a+forward_b+diffusion+causal)/4".into(),
        corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
        benchmark_manifest_fingerprint: format!(
            "fnv1a64:{:016x}",
            benchmark.manifest_fingerprint()
        ),
        forward_checkpoint: forward_model_path.display().to_string(),
        diffusion_checkpoint: diffusion_model_path.display().to_string(),
        causal_checkpoint: causal_model_path.display().to_string(),
        train_steps,
        batch_size,
        validation_batches,
        seed,
        learning_rate,
        max_gradient_norm,
        diffusion_length_weight,
        alignment_weight,
        alignment_temperature,
        completed_steps,
        forward_config: forward_metadata.model_config.clone(),
        inverse_config: inverse_config.clone(),
    };

    save_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &optimizer,
        &metadata(0),
    )?;
    print_evaluation(
        "initial",
        0,
        evaluate_all(
            &model,
            &corpus.records,
            &validation_forward,
            &validation_inverse,
            batch_size,
            &clean_collator,
            &diffusion_collator,
            &causal_collator,
            &spectrum_collator,
            &forward_trainer.target_normalization,
            alignment_temperature,
            &device,
        )?,
    );

    let probe_indices = deterministic_batch(
        &train_inverse_indices,
        batch_size,
        seed ^ 0x5c7d_92a1_b460_31ef,
    );
    let probe_records: Vec<&FoundationTrainingRecord> = probe_indices
        .iter()
        .map(|&index| &corpus.records[index])
        .collect();
    alignment_gradient_probe(
        &model,
        &varmap,
        &probe_records,
        &clean_collator,
        &diffusion_collator,
        &spectrum_collator,
        alignment_temperature,
        &device,
    )?;

    let eval_every = train_steps.min(10).max(1);
    for step in 1..=train_steps {
        let forward_a_indices = deterministic_batch(
            &train_forward_indices,
            batch_size,
            seed ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let forward_b_indices = deterministic_batch(
            &train_forward_indices,
            batch_size,
            seed ^ (step as u64).wrapping_mul(0x4cf5_ad43_2745_937f),
        );
        let forward_a_records: Vec<FoundationTrainingRecord> = forward_a_indices
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect();
        let forward_b_records: Vec<FoundationTrainingRecord> = forward_b_indices
            .iter()
            .map(|&index| corpus.records[index].clone())
            .collect();
        let forward_a = forward_loss(
            &model,
            &forward_collator,
            &forward_a_records,
            forward_trainer.loss_weights,
            forward_trainer.contrastive_temperature,
            forward_trainer.shared_gradient_scales.rt_encoder,
            forward_trainer.shared_gradient_scales.ccs_encoder,
            &forward_trainer.target_normalization,
            seed ^ (step as u64).wrapping_mul(0xa24b_1c62_4073_f5d9),
            &device,
        )?;
        let forward_b = forward_loss(
            &model,
            &forward_collator,
            &forward_b_records,
            forward_trainer.loss_weights,
            forward_trainer.contrastive_temperature,
            forward_trainer.shared_gradient_scales.rt_encoder,
            forward_trainer.shared_gradient_scales.ccs_encoder,
            &forward_trainer.target_normalization,
            seed ^ (step as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
            &device,
        )?;

        let diffusion_indices = deterministic_batch(
            &train_inverse_indices,
            batch_size,
            seed ^ (step as u64).wrapping_mul(0x8cb9_2baa_4f31_7e0d),
        );
        let diffusion_records: Vec<&FoundationTrainingRecord> = diffusion_indices
            .iter()
            .map(|&index| &corpus.records[index])
            .collect();
        let force_all_masked = mix64(seed ^ step as u64) & 1 == 0;
        let (diffusion, diffusion_alignment) = diffusion_loss(
            &model,
            &diffusion_records,
            &clean_collator,
            &diffusion_collator,
            &spectrum_collator,
            &forward_trainer.target_normalization,
            diffusion_length_weight,
            alignment_weight,
            alignment_temperature,
            force_all_masked,
            seed ^ step as u64,
            &device,
        )?;

        let causal_indices = deterministic_batch(
            &train_inverse_indices,
            batch_size,
            seed ^ (step as u64).wrapping_mul(0x94d0_49bb_1331_11eb),
        );
        let causal_records: Vec<&FoundationTrainingRecord> = causal_indices
            .iter()
            .map(|&index| &corpus.records[index])
            .collect();
        let (causal, causal_alignment) = causal_loss(
            &model,
            &causal_records,
            &clean_collator,
            &causal_collator,
            &spectrum_collator,
            &forward_trainer.target_normalization,
            alignment_weight,
            alignment_temperature,
            &device,
        )?;

        let forward_a_value = f64::from(forward_a.to_scalar::<f32>()?);
        let forward_b_value = f64::from(forward_b.to_scalar::<f32>()?);
        let diffusion_value = f64::from(diffusion.to_scalar::<f32>()?);
        let causal_value = f64::from(causal.to_scalar::<f32>()?);
        let total = (((forward_a + forward_b)? + diffusion)? + causal)?.affine(0.25, 0.0)?;
        let total_value = f64::from(total.to_scalar::<f32>()?);
        let update = optimizer.backward_step(&total, Some(max_gradient_norm))?;
        println!(
            "train\tstep={step}\tforward_a={forward_a_value:.6}\tforward_b={forward_b_value:.6}\tdiffusion={diffusion_value:.6}\tcausal={causal_value:.6}\tdiffusion_alignment={diffusion_alignment:.6}\tcausal_alignment={causal_alignment:.6}\ttotal={total_value:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
            update.gradient_norm,
            update.gradient_scale,
        );

        if step % eval_every == 0 || step == train_steps {
            let metrics = evaluate_all(
                &model,
                &corpus.records,
                &validation_forward,
                &validation_inverse,
                batch_size,
                &clean_collator,
                &diffusion_collator,
                &causal_collator,
                &spectrum_collator,
                &forward_trainer.target_normalization,
                alignment_temperature,
                &device,
            )?;
            print_evaluation("validation", step, metrics);
            save_checkpoint(
                &output_root.join("latest"),
                &varmap,
                &optimizer,
                &metadata(step),
            )?;
        }
    }

    save_checkpoint(
        &output_root.join("final"),
        &varmap,
        &optimizer,
        &metadata(train_steps),
    )?;
    println!("final_checkpoint\t{}", output_root.join("final").display());
    Ok(())
}

fn forward_loss(
    model: &PeptideFoundationUnifiedModel,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    weights: FoundationLossWeights,
    contrastive_temperature: f64,
    rt_scale: f64,
    ccs_scale: f64,
    normalization: &FoundationTargetNormalizationConfig,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let mut views = collator.collate_views(records, device, seed)?;
    normalize_targets(&mut views, normalization)?;
    let first = model.forward().forward_t_with_shared_gradient_scales(
        &views.first.input,
        &views.first.context,
        true,
        rt_scale,
        ccs_scale,
    )?;
    let second = model
        .forward()
        .forward_t(&views.second.input, &views.second.context, true)?;
    let losses = multi_task_loss(&first, &views.first.targets, weights)?;
    let contrastive = contrastive_info_nce_loss(
        &first.contrastive_projection,
        &second.contrastive_projection,
        contrastive_temperature,
    )?;
    (losses.total + contrastive.affine(weights.contrastive, 0.0)?).map_err(anyhow::Error::from)
}

#[allow(clippy::too_many_arguments)]
fn diffusion_loss(
    model: &PeptideFoundationUnifiedModel,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    _normalization: &FoundationTargetNormalizationConfig,
    length_weight: f64,
    alignment_weight: f64,
    alignment_temperature: f64,
    force_all_masked: bool,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, f64)> {
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let diffusion = if force_all_masked {
        diffusion_collator.collate_all_masked(
            &peptides,
            model.inverse_config().diffusion_steps,
            device,
        )?
    } else {
        diffusion_collator.collate_random_timesteps(&peptides, seed, device)?
    };
    let spectrum = collate_spectra(records, spectrum_collator, false, device)?;
    let precursor = precursor_context(records, device)?;
    let output = model
        .diffusion()
        .forward_t(&diffusion, &spectrum, &precursor, true)?;
    let x0 = foundation_diffusion_x0_loss(&output, &diffusion)?;
    let length = foundation_diffusion_length_loss(&output, &diffusion)?;
    let peptide_projection =
        clean_peptide_projection(model, records, clean_collator, true, device)?;
    let spectrum_projection = model.project_spectrum_embedding(&output.spectrum_embedding)?;
    let alignment = foundation_spectrum_peptide_alignment_loss(
        &spectrum_projection,
        &peptide_projection,
        alignment_temperature,
    )?;
    let alignment_value = f64::from(alignment.to_scalar::<f32>()?);
    let total =
        ((x0 + length.affine(length_weight, 0.0)?)? + alignment.affine(alignment_weight, 0.0)?)?;
    Ok((total, alignment_value))
}

#[allow(clippy::too_many_arguments)]
fn causal_loss(
    model: &PeptideFoundationUnifiedModel,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    _normalization: &FoundationTargetNormalizationConfig,
    alignment_weight: f64,
    alignment_temperature: f64,
    device: &Device,
) -> Result<(Tensor, f64)> {
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let causal = causal_collator.collate(&peptides, device)?;
    let spectrum = collate_spectra(records, spectrum_collator, false, device)?;
    let precursor = precursor_context(records, device)?;
    let output = model
        .causal()
        .forward_t(&causal.input, &spectrum, &precursor, true)?;
    let causal_ce = foundation_causal_next_token_loss(&output, &causal)?;
    let peptide_projection =
        clean_peptide_projection(model, records, clean_collator, true, device)?;
    let spectrum_projection = model.project_spectrum_embedding(&output.spectrum_embedding)?;
    let alignment = foundation_spectrum_peptide_alignment_loss(
        &spectrum_projection,
        &peptide_projection,
        alignment_temperature,
    )?;
    let alignment_value = f64::from(alignment.to_scalar::<f32>()?);
    let total = (causal_ce + alignment.affine(alignment_weight, 0.0)?)?;
    Ok((total, alignment_value))
}

fn normalize_targets(
    views: &mut FoundationTrainingViews,
    normalization: &FoundationTargetNormalizationConfig,
) -> Result<()> {
    normalize_one_targets(&mut views.first.targets, normalization)?;
    normalize_one_targets(&mut views.second.targets, normalization)
}

fn normalize_one_targets(
    targets: &mut redeem_properties::foundation::FoundationTargets,
    normalization: &FoundationTargetNormalizationConfig,
) -> Result<()> {
    if let Some(rt) = targets.rt.take() {
        targets.rt = Some(normalization.rt.normalize_tensor(&rt)?);
    }
    if let Some(ccs) = targets.ccs.take() {
        targets.ccs = Some(normalization.ccs.normalize_tensor(&ccs)?);
    }
    Ok(())
}

fn clean_peptide_projection(
    model: &PeptideFoundationUnifiedModel,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    train: bool,
    device: &Device,
) -> Result<Tensor> {
    let owned: Vec<FoundationTrainingRecord> =
        records.iter().map(|record| (*record).clone()).collect();
    let batch = clean_collator.collate(&owned, device, 0)?;
    Ok(model
        .forward()
        .forward_t(&batch.input, &batch.context, train)?
        .contrastive_projection)
}

fn alignment_gradient_probe(
    model: &PeptideFoundationUnifiedModel,
    varmap: &VarMap,
    records: &[&FoundationTrainingRecord],
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    alignment_temperature: f64,
    device: &Device,
) -> Result<()> {
    let peptides: Vec<PeptidoformInput> = records
        .iter()
        .map(|record| record.peptidoform.clone())
        .collect();
    let diffusion = diffusion_collator.collate_all_masked(
        &peptides,
        model.inverse_config().diffusion_steps,
        device,
    )?;
    let spectrum = collate_spectra(records, spectrum_collator, false, device)?;
    let precursor = precursor_context(records, device)?;
    let inverse = model
        .diffusion()
        .forward_t(&diffusion, &spectrum, &precursor, true)?;
    let peptide_projection =
        clean_peptide_projection(model, records, clean_collator, true, device)?;
    let spectrum_projection = model.project_spectrum_embedding(&inverse.spectrum_embedding)?;
    let alignment = foundation_spectrum_peptide_alignment_loss(
        &spectrum_projection,
        &peptide_projection,
        alignment_temperature,
    )?;
    let gradients = alignment.backward()?;
    println!(
        "alignment_gradient_probe\tloss={:.6}\tpeptide_encoder={:.8}\tpeptide_contrastive_head={:.8}\tspectrum_encoder={:.8}\tspectrum_projection={:.8}",
        f64::from(alignment.to_scalar::<f32>()?),
        gradient_norm_for_prefix(varmap, &gradients, "encoder.")?,
        gradient_norm_for_prefix(varmap, &gradients, "heads.contrastive.")?,
        gradient_norm_for_prefix(varmap, &gradients, "spectrum_encoder.")?,
        gradient_norm_for_prefix(varmap, &gradients, "alignment.spectrum_projection.")?,
    );
    Ok(())
}

fn gradient_norm_for_prefix(
    varmap: &VarMap,
    gradients: &candle_core::backprop::GradStore,
    prefix: &str,
) -> Result<f64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("unified VarMap lock poisoned"))?;
    let mut squared = 0.0f64;
    let mut matched = 0usize;
    for (name, variable) in data.iter() {
        if !name.starts_with(prefix) {
            continue;
        }
        matched += 1;
        if let Some(gradient) = gradients.get(variable) {
            squared += f64::from(gradient.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }
    if matched == 0 {
        anyhow::bail!("gradient probe matched no variables for prefix '{prefix}'");
    }
    Ok(squared.sqrt())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_all(
    model: &PeptideFoundationUnifiedModel,
    records: &[FoundationTrainingRecord],
    forward_indices: &[usize],
    inverse_indices: &[usize],
    batch_size: usize,
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    normalization: &FoundationTargetNormalizationConfig,
    alignment_temperature: f64,
    device: &Device,
) -> Result<(PropertyMetrics, InverseMetrics)> {
    let properties = evaluate_properties(
        model,
        records,
        forward_indices,
        batch_size,
        clean_collator,
        normalization,
        device,
    )?;
    let inverse = evaluate_inverse(
        model,
        records,
        inverse_indices,
        batch_size,
        clean_collator,
        diffusion_collator,
        causal_collator,
        spectrum_collator,
        alignment_temperature,
        device,
    )?;
    Ok((properties, inverse))
}

fn evaluate_properties(
    model: &PeptideFoundationUnifiedModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    clean_collator: &FoundationCollator,
    normalization: &FoundationTargetNormalizationConfig,
    device: &Device,
) -> Result<PropertyMetrics> {
    let mut rt_abs = 0.0f64;
    let mut rt_sq = 0.0f64;
    let mut rt_n = 0usize;
    let mut ccs_abs = 0.0f64;
    let mut ccs_sq = 0.0f64;
    let mut ccs_n = 0usize;
    let mut ms2_weighted = 0.0f64;
    let mut ms2_batches = 0usize;

    for chunk in indices.chunks(batch_size) {
        let owned: Vec<FoundationTrainingRecord> =
            chunk.iter().map(|&i| records[i].clone()).collect();
        let mut batch = clean_collator.collate(&owned, device, 0)?;
        if let Some(rt) = batch.targets.rt.take() {
            batch.targets.rt = Some(normalization.rt.normalize_tensor(&rt)?);
        }
        if let Some(ccs) = batch.targets.ccs.take() {
            batch.targets.ccs = Some(normalization.ccs.normalize_tensor(&ccs)?);
        }
        let output = model
            .forward()
            .forward_t(&batch.input, &batch.context, false)?;
        accumulate_regression(
            &output.rt,
            batch.targets.rt.as_ref(),
            batch.targets.rt_mask.as_ref(),
            &normalization.rt,
            &mut rt_abs,
            &mut rt_sq,
            &mut rt_n,
        )?;
        accumulate_regression(
            &output.ccs,
            batch.targets.ccs.as_ref(),
            batch.targets.ccs_mask.as_ref(),
            &normalization.ccs,
            &mut ccs_abs,
            &mut ccs_sq,
            &mut ccs_n,
        )?;
        let weights = FoundationLossWeights {
            rt: 0.0,
            ccs: 0.0,
            ms2: 1.0,
            masked_residue: 0.0,
            chemistry: 0.0,
            contrastive: 0.0,
        };
        let losses = multi_task_loss(&output, &batch.targets, weights)?;
        if let Some(ms2) = losses.ms2 {
            ms2_weighted += f64::from(ms2.to_scalar::<f32>()?);
            ms2_batches += 1;
        }
    }

    Ok(PropertyMetrics {
        rt_mae_native: (rt_n > 0).then(|| rt_abs / rt_n as f64),
        rt_rmse_native: (rt_n > 0).then(|| (rt_sq / rt_n as f64).sqrt()),
        ccs_mae_native: (ccs_n > 0).then(|| ccs_abs / ccs_n as f64),
        ccs_rmse_native: (ccs_n > 0).then(|| (ccs_sq / ccs_n as f64).sqrt()),
        ms2_loss: (ms2_batches > 0).then(|| ms2_weighted / ms2_batches as f64),
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_inverse(
    model: &PeptideFoundationUnifiedModel,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    clean_collator: &FoundationCollator,
    diffusion_collator: &FoundationDiffusionCollator,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    alignment_temperature: f64,
    device: &Device,
) -> Result<InverseMetrics> {
    let mut metrics = InverseMetrics::default();
    let mut batches = 0usize;
    for chunk in indices.chunks(batch_size) {
        let selected: Vec<&FoundationTrainingRecord> = chunk.iter().map(|&i| &records[i]).collect();
        let peptides: Vec<PeptidoformInput> = selected
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect();
        let diffusion = diffusion_collator.collate_all_masked(
            &peptides,
            model.inverse_config().diffusion_steps,
            device,
        )?;
        let causal = causal_collator.collate(&peptides, device)?;
        let precursor = precursor_context(&selected, device)?;
        let spectrum = collate_spectra(&selected, spectrum_collator, false, device)?;
        let shuffled_spectrum = collate_spectra(&selected, spectrum_collator, true, device)?;

        let diffusion_output = model
            .diffusion()
            .forward_t(&diffusion, &spectrum, &precursor, false)?;
        let diffusion_loss = foundation_diffusion_x0_loss(&diffusion_output, &diffusion)?;
        let length_loss = foundation_diffusion_length_loss(&diffusion_output, &diffusion)?;
        metrics.diffusion_loss += f64::from(diffusion_loss.to_scalar::<f32>()?);
        metrics.diffusion_length_loss += f64::from(length_loss.to_scalar::<f32>()?);
        metrics.diffusion_token_accuracy += token_accuracy(
            &diffusion_output.token_logits,
            &diffusion.active_indices,
            &diffusion.target_classes,
        )?;

        let causal_output =
            model
                .causal()
                .forward_t(&causal.input, &spectrum, &precursor, false)?;
        let causal_loss = foundation_causal_next_token_loss(&causal_output, &causal)?;
        let causal_loss_value = f64::from(causal_loss.to_scalar::<f32>()?);
        metrics.causal_loss += causal_loss_value;
        metrics.causal_perplexity += causal_loss_value.exp();
        metrics.causal_token_accuracy += token_accuracy(
            &causal_output.token_logits,
            &causal.active_indices,
            &causal.target_classes,
        )?;

        let peptide_projection =
            clean_peptide_projection(model, &selected, clean_collator, false, device)?;
        let matched_projection =
            model.project_spectrum_embedding(&diffusion_output.spectrum_embedding)?;
        let alignment_loss = foundation_spectrum_peptide_alignment_loss(
            &matched_projection,
            &peptide_projection,
            alignment_temperature,
        )?;
        metrics.alignment_loss += f64::from(alignment_loss.to_scalar::<f32>()?);
        metrics.alignment_retrieval_top1 +=
            retrieval_top1(&matched_projection, &peptide_projection)?;

        let shuffled_output =
            model
                .diffusion()
                .forward_t(&diffusion, &shuffled_spectrum, &precursor, false)?;
        let shuffled_projection =
            model.project_spectrum_embedding(&shuffled_output.spectrum_embedding)?;
        let shuffled_loss = foundation_spectrum_peptide_alignment_loss(
            &shuffled_projection,
            &peptide_projection,
            alignment_temperature,
        )?;
        metrics.shuffled_alignment_loss += f64::from(shuffled_loss.to_scalar::<f32>()?);
        metrics.shuffled_alignment_retrieval_top1 +=
            retrieval_top1(&shuffled_projection, &peptide_projection)?;
        batches += 1;
    }
    if batches == 0 {
        anyhow::bail!("unified inverse validation produced zero batches");
    }
    let n = batches as f64;
    metrics.diffusion_loss /= n;
    metrics.diffusion_length_loss /= n;
    metrics.diffusion_token_accuracy /= n;
    metrics.causal_loss /= n;
    metrics.causal_perplexity /= n;
    metrics.causal_token_accuracy /= n;
    metrics.alignment_loss /= n;
    metrics.alignment_retrieval_top1 /= n;
    metrics.shuffled_alignment_loss /= n;
    metrics.shuffled_alignment_retrieval_top1 /= n;
    Ok(metrics)
}

fn accumulate_regression(
    prediction: &Tensor,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    normalization: &FoundationRegressionNormalization,
    abs_sum: &mut f64,
    sq_sum: &mut f64,
    count: &mut usize,
) -> Result<()> {
    let (Some(target), Some(mask)) = (target, mask) else {
        return Ok(());
    };
    let prediction = normalization
        .denormalize_tensor(prediction)?
        .to_vec2::<f32>()?;
    let target = normalization.denormalize_tensor(target)?.to_vec2::<f32>()?;
    let mask = mask.to_vec2::<f32>()?;
    for row in 0..prediction.len() {
        if mask[row][0] <= 0.0 {
            continue;
        }
        let error = f64::from(prediction[row][0] - target[row][0]);
        *abs_sum += error.abs();
        *sq_sum += error * error;
        *count += 1;
    }
    Ok(())
}

fn token_accuracy(logits: &Tensor, active_indices: &Tensor, classes: &Tensor) -> Result<f64> {
    let (b, l, vocab) = logits.dims3()?;
    if vocab != FOUNDATION_DIFFUSION_VOCAB_SIZE {
        anyhow::bail!("unexpected unified token vocabulary {vocab}");
    }
    let flat = logits.reshape((b * l, vocab))?;
    let selected = flat.index_select(active_indices, 0)?.to_vec2::<f32>()?;
    let targets = classes.to_vec1::<u32>()?;
    let correct = selected
        .iter()
        .zip(targets.iter())
        .filter(|(row, target)| argmax(row) as u32 == **target)
        .count();
    Ok(correct as f64 / targets.len().max(1) as f64)
}

fn retrieval_top1(spectrum: &Tensor, peptide: &Tensor) -> Result<f64> {
    let spectrum = l2_rows(spectrum)?;
    let peptide = l2_rows(peptide)?;
    let s = spectrum.to_vec2::<f32>()?;
    let p = peptide.to_vec2::<f32>()?;
    if s.len() != p.len() || s.is_empty() {
        anyhow::bail!("alignment retrieval requires equal non-empty batches");
    }
    let mut correct = 0usize;
    for (i, row) in s.iter().enumerate() {
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (j, candidate) in p.iter().enumerate() {
            let score: f64 = row
                .iter()
                .zip(candidate.iter())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            if score > best_score {
                best_score = score;
                best = j;
            }
        }
        correct += usize::from(best == i);
    }
    Ok(correct as f64 / s.len() as f64)
}

fn l2_rows(values: &Tensor) -> Result<Tensor> {
    let denominator = values
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?
        .unsqueeze(1)?;
    Ok(values.broadcast_div(&denominator)?)
}

fn collate_spectra(
    records: &[&FoundationTrainingRecord],
    spectrum_collator: &FoundationSpectrumCollator,
    shuffled: bool,
    device: &Device,
) -> Result<FoundationSpectrumBatch> {
    let mut spectra: Vec<FoundationSpectrum> = records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record)
                .ok_or_else(|| anyhow::anyhow!("selected unified inverse record lacks spectrum"))
        })
        .collect::<Result<_>>()?;
    if shuffled && spectra.len() > 1 {
        spectra.rotate_left(1);
    }
    Ok(spectrum_collator.collate(&spectra, device)?)
}

fn precursor_context(
    records: &[&FoundationTrainingRecord],
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let charge: Vec<f32> = records
        .iter()
        .map(|r| r.context.charge.unwrap_or(0) as f32)
        .collect();
    let charge_present: Vec<f32> = records
        .iter()
        .map(|r| if r.context.charge.is_some() { 1.0 } else { 0.0 })
        .collect();
    let precursor_mz: Vec<f32> = records
        .iter()
        .map(|r| r.context.precursor_mz.unwrap_or(0.0))
        .collect();
    let precursor_mz_present: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.context.precursor_mz.is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let nce: Vec<f32> = records
        .iter()
        .map(|r| r.context.nce.unwrap_or(0.0))
        .collect();
    let nce_present: Vec<f32> = records
        .iter()
        .map(|r| if r.context.nce.is_some() { 1.0 } else { 0.0 })
        .collect();
    let b = records.len();
    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(charge, b, device)?,
        charge_present: Tensor::from_vec(charge_present, b, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, b, device)?,
        precursor_mz_present: Tensor::from_vec(precursor_mz_present, b, device)?,
        nce: Tensor::from_vec(nce, b, device)?,
        nce_present: Tensor::from_vec(nce_present, b, device)?,
        instrument_ids: Tensor::zeros(b, DType::U32, device)?,
        instrument_present: Tensor::zeros(b, DType::F32, device)?,
    })
}

fn usable_inverse_indices(
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

fn read_inverse_metadata(checkpoint: &Path) -> Result<InverseCheckpointMetadata> {
    let metadata_path = if checkpoint.is_dir() {
        checkpoint.join("metadata.yaml")
    } else {
        checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    };
    Ok(serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read inverse metadata {metadata_path:?}"))?,
    )?)
}

fn resolve_forward_state(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.join("state.yaml")
    } else {
        checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("state.yaml")
    }
}

fn resolve_model_safetensors(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.join("model.safetensors")
    } else {
        checkpoint.to_path_buf()
    }
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &UnifiedPilotMetadata,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    optimizer.save_safetensors(directory.join("optimizer.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    Ok(())
}

fn print_evaluation(label: &str, step: usize, metrics: (PropertyMetrics, InverseMetrics)) {
    let (p, i) = metrics;
    println!(
        "{label}_forward\tstep={step}\trt_mae_native={}\trt_rmse_native={}\tccs_mae_native={}\tccs_rmse_native={}\tms2_loss={}",
        fmt_opt(p.rt_mae_native),
        fmt_opt(p.rt_rmse_native),
        fmt_opt(p.ccs_mae_native),
        fmt_opt(p.ccs_rmse_native),
        fmt_opt(p.ms2_loss),
    );
    println!(
        "{label}_inverse\tstep={step}\tdiffusion_loss={:.6}\tdiffusion_length_loss={:.6}\tdiffusion_token_accuracy={:.6}\tcausal_loss={:.6}\tcausal_perplexity={:.4}\tcausal_token_accuracy={:.6}\talignment_loss={:.6}\talignment_retrieval_top1={:.6}\tshuffled_alignment_loss={:.6}\tshuffled_alignment_retrieval_top1={:.6}",
        i.diffusion_loss,
        i.diffusion_length_loss,
        i.diffusion_token_accuracy,
        i.causal_loss,
        i.causal_perplexity,
        i.causal_token_accuracy,
        i.alignment_loss,
        i.alignment_retrieval_top1,
        i.shuffled_alignment_loss,
        i.shuffled_alignment_retrieval_top1,
    );
    println!(
        "{label}_alignment_ablation\tstep={step}\tloss_delta={:.6}\tretrieval_top1_delta={:.6}",
        i.shuffled_alignment_loss - i.alignment_loss,
        i.alignment_retrieval_top1 - i.shuffled_alignment_retrieval_top1,
    );
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "NA".into())
}

fn deterministic_batch(indices: &[usize], batch_size: usize, seed: u64) -> Vec<usize> {
    let mut result = Vec::with_capacity(batch_size);
    let mut state = mix64(seed);
    for _ in 0..batch_size {
        state = mix64(state ^ 0x9e37_79b9_7f4a_7c15);
        result.push(indices[(state as usize) % indices.len()]);
    }
    result
}

fn deterministic_subset(indices: &[usize], count: usize, seed: u64) -> Vec<usize> {
    if indices.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut keyed: Vec<(u64, usize)> = indices
        .iter()
        .copied()
        .map(|index| (mix64(seed ^ index as u64), index))
        .collect();
    keyed.sort_by_key(|entry| entry.0);
    keyed
        .into_iter()
        .take(count.min(indices.len()))
        .map(|entry| entry.1)
        .collect()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
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
