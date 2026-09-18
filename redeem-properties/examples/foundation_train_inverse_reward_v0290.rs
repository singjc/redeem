//! v0.29 physics-aware sequence-reward post-training for spectrum -> peptide.
//!
//! This lane is intentionally independent of the forward-property v0.30 lane.
//! The complete frozen v0.27 architecture is restored twice: one trainable
//! policy and one frozen reference policy. No model parameters are added.
//! Optimization touches only the causal/spectrum path exercised by the losses.
//! Historical VALIDATION and TEST are never consumed by this executable.
//!
//! The first v0.29 experiment is deliberately restricted to unmodified peptide
//! targets because the accepted hard-mass direct beam search uses the legacy
//! discrete vocabulary. Open-PTM inverse support remains present in v0.27 and is
//! not regressed; this post-training pilot asks one clean question first: does a
//! complete-sequence physical reward improve direct peptide generation?

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_causal_next_token_loss, foundation_causal_sequence_mean_nlls,
    foundation_diffusion_token_residue, foundation_direct_beam_search,
    foundation_fragment_relation_features, foundation_group_relative_advantages_v0290,
    foundation_group_relative_policy_loss_v0290, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, foundation_reference_nll_anchor_v0290,
    foundation_sequence_reward_v0290, load_foundation_corpus, read_foundation_training_run_config,
    DirectDecoderBeamConfig, FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest,
    FoundationCausalCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationLearningRateSchedule, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideFoundationInverseRewardV0290Config, PeptideFoundationInverseRewardV0290Model,
    PeptideFoundationMultimodalV0270Config, PrecursorContextBatch, FOUNDATION_DIFFUSION_EOS,
    FOUNDATION_DIFFUSION_VOCAB_SIZE, FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240,
    FOUNDATION_FRAGMENT_RELATION_EXPLAINED_INTENSITY_V0240,
    FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240,
    FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240,
    FOUNDATION_FRAGMENT_RELATION_PEAK_COVERAGE_V0240, FOUNDATION_INVERSE_REWARD_ARCHITECTURE_V0290,
    FOUNDATION_INVERSE_REWARD_BEAM_WIDTH_V0290, FOUNDATION_INVERSE_REWARD_GROUPS_PER_STEP_V0290,
    FOUNDATION_INVERSE_REWARD_POLICY_WEIGHT_V0290,
    FOUNDATION_INVERSE_REWARD_REFERENCE_WEIGHT_V0290,
    FOUNDATION_INVERSE_REWARD_SUPERVISED_WEIGHT_V0290, FOUNDATION_INVERSE_REWARD_TOP_K_V0290,
    FOUNDATION_SEQUENCE_REWARD_OBJECTIVE_V0290,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TRAIN_STEPS_PER_EPOCH_V0290: usize = 256;
const DEV_GENERATION_RECORDS_V0290: usize = 64;
const HOLDOUT_GENERATION_RECORDS_V0290: usize = 64;
const DEV_BEAM_WIDTH_V0290: usize = 32;
const DEV_TOP_K_V0290: usize = 10;
const MAX_GRADIENT_NORM_V0290: f64 = 1.0;

#[derive(Debug, Deserialize)]
struct V0270ParentMetadata {
    version: u32,
    objective: String,
    completed_steps: usize,
    v0270_config: PeptideFoundationMultimodalV0270Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V0290CheckpointMetadata {
    version: u32,
    objective: String,
    architecture: String,
    reward_objective: String,
    parent_checkpoint: String,
    parent_completed_steps: usize,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    completed_steps: usize,
    completed_epoch: usize,
    batch_size: usize,
    seed: u64,
    learning_rate: f64,
    max_epochs: usize,
    patience: usize,
    min_delta: f64,
    v0290_config: PeptideFoundationInverseRewardV0290Config,
    dev_generation_indices: Vec<usize>,
    holdout_generation_indices: Vec<usize>,
    train_reward_eligible_records: usize,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct GenerationMetrics {
    records: usize,
    literal_top1: usize,
    il_top1: usize,
    literal_topk: usize,
    il_topk: usize,
    returned_candidates: usize,
    zero_candidate_records: usize,
}

impl GenerationMetrics {
    fn literal_top1_rate(self) -> f64 {
        ratio(self.literal_top1, self.records)
    }
    fn il_top1_rate(self) -> f64 {
        ratio(self.il_top1, self.records)
    }
    fn literal_topk_rate(self) -> f64 {
        ratio(self.literal_topk, self.records)
    }
    fn il_topk_rate(self) -> f64 {
        ratio(self.il_topk, self.records)
    }
    fn mean_returned(self) -> f64 {
        ratio(self.returned_candidates, self.records)
    }
    fn selection_score(self) -> f64 {
        0.50 * self.literal_top1_rate() + 0.30 * self.il_top1_rate() + 0.20 * self.il_topk_rate()
    }
    fn objective(self) -> f64 {
        1.0 - self.selection_score()
    }
}

#[derive(Debug)]
struct RewardGroupLoss {
    total: Tensor,
    policy: f64,
    reference: f64,
    reward_mean: f64,
    reward_max: f64,
    candidates: usize,
    literal_present: bool,
    il_present: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 11 {
        anyhow::bail!(
            "usage: foundation_train_inverse_reward_v0290 RUN_V0260.yaml OUTPUT_DIR PARENT_V0270_CHECKPOINT [max_epochs=8] [batch_size=16] [patience=3] [min_delta=0.002] [seed=20260929] [learning_rate=1e-5] [mode=train|resume|finalize]"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let max_epochs = parse_or(&args, 4, 8usize)?;
    let batch_size = parse_or(&args, 5, 16usize)?;
    let patience = parse_or(&args, 6, 3usize)?;
    let min_delta = parse_or(&args, 7, 0.002f64)?;
    let seed = parse_or(&args, 8, 20_260_929u64)?;
    let learning_rate = parse_or(&args, 9, 1.0e-5f64)?;
    let mode = args.get(10).map(String::as_str).unwrap_or("train");
    if !matches!(mode, "train" | "resume" | "finalize") {
        anyhow::bail!("v0.29 mode must be train, resume, or finalize");
    }
    if max_epochs == 0 || batch_size < 2 || patience == 0 {
        anyhow::bail!("v0.29 max_epochs/patience must be positive and batch_size >=2");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite())
        || !(min_delta >= 0.0 && min_delta.is_finite())
    {
        anyhow::bail!("v0.29 learning rate/min_delta are invalid");
    }
    if mode == "train" && output_root.exists() {
        anyhow::bail!("v0.29 output directory already exists: {output_root:?}");
    }
    if mode != "train" && !output_root.is_dir() {
        anyhow::bail!("v0.29 {mode} requires existing output directory: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.29 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let parent_metadata = read_parent_metadata(&parent_checkpoint)?;
    if parent_metadata.version != 270
        || parent_metadata.objective != "v0270_rt_specialist_contextual_fragment_tokens_openptm32"
    {
        anyhow::bail!(
            "v0.29 requires frozen v0.27 parent (version=270), observed version={} objective={:?}",
            parent_metadata.version,
            parent_metadata.objective
        );
    }
    let v0290_config =
        PeptideFoundationInverseRewardV0290Config::fixed(parent_metadata.v0270_config.clone())?;
    let inverse_config = v0290_config.inverse().clone();

    let train_indices = reward_eligible_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &inverse_config,
    );
    let dev_indices = reward_eligible_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &inverse_config,
    );
    let holdout_indices = reward_eligible_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Test,
        &inverse_config,
    );
    if train_indices.len() < batch_size
        || dev_indices.len() < DEV_GENERATION_RECORDS_V0290
        || holdout_indices.len() < HOLDOUT_GENERATION_RECORDS_V0290
    {
        anyhow::bail!(
            "insufficient v0.29 reward-eligible records: train={} dev={} holdout={} batch={batch_size}",
            train_indices.len(), dev_indices.len(), holdout_indices.len()
        );
    }
    let dev_generation_indices = deterministic_subset(
        &dev_indices,
        DEV_GENERATION_RECORDS_V0290,
        seed ^ 0x5644_4556_3032_3930,
    );
    let holdout_generation_indices = deterministic_subset(
        &holdout_indices,
        HOLDOUT_GENERATION_RECORDS_V0290,
        seed ^ 0x484f_4c44_3032_3930,
    );

    let causal_collator = FoundationCausalCollator::new_open_ptm(inverse_config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(inverse_config.spectrum.clone())?;

    let mut varmap = VarMap::new();
    let model = PeptideFoundationInverseRewardV0290Model::new(
        v0290_config.clone(),
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
    )?;
    load_exact_v0270(
        &varmap,
        &parent_checkpoint.join("model.safetensors"),
        &device,
    )?;

    let reference_varmap = VarMap::new();
    let reference_model = PeptideFoundationInverseRewardV0290Model::new(
        v0290_config.clone(),
        VarBuilder::from_varmap(&reference_varmap, DType::F32, &device),
    )?;
    load_exact_v0270(
        &reference_varmap,
        &parent_checkpoint.join("model.safetensors"),
        &device,
    )?;

    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: run.trainer.weight_decay,
        },
    )?;
    let max_steps = max_epochs.saturating_mul(TRAIN_STEPS_PER_EPOCH_V0290);
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 250u64.min(max_steps.saturating_sub(1) as u64),
        total_steps: max_steps as u64,
        min_lr_ratio: 0.10,
    };

    fs::create_dir_all(&output_root)?;
    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    let metadata = |completed_steps: usize, completed_epoch: usize| V0290CheckpointMetadata {
        version: 290,
        objective: "v0290_inverse_group_relative_physics_reward".into(),
        architecture: FOUNDATION_INVERSE_REWARD_ARCHITECTURE_V0290.into(),
        reward_objective: FOUNDATION_SEQUENCE_REWARD_OBJECTIVE_V0290.into(),
        parent_checkpoint: parent_checkpoint.display().to_string(),
        parent_completed_steps: parent_metadata.completed_steps,
        corpus_fingerprint: corpus_fingerprint.clone(),
        benchmark_manifest_fingerprint: benchmark_fingerprint.clone(),
        completed_steps,
        completed_epoch,
        batch_size,
        seed,
        learning_rate,
        max_epochs,
        patience,
        min_delta,
        v0290_config: v0290_config.clone(),
        dev_generation_indices: dev_generation_indices.clone(),
        holdout_generation_indices: holdout_generation_indices.clone(),
        train_reward_eligible_records: train_indices.len(),
        historical_validation_consumed: false,
        historical_test_consumed: false,
    };

    print_header(
        &parent_checkpoint,
        &v0290_config,
        train_indices.len(),
        dev_indices.len(),
        holdout_indices.len(),
    );

    if mode == "finalize" {
        let best_dir = output_root.join("best");
        let best_meta = read_v0290_metadata(&best_dir)?;
        if best_meta.completed_steps == 0 {
            anyhow::bail!("v0.29 finalize refused: DEV never improved over frozen v0.27 baseline");
        }
        varmap.load(best_dir.join("model.safetensors"))?;
        let metrics = evaluate_generation(
            &model,
            &corpus.records,
            &holdout_generation_indices,
            &causal_collator,
            &spectrum_collator,
            DEV_BEAM_WIDTH_V0290,
            DEV_TOP_K_V0290,
            &device,
        )?;
        print_generation(
            "train_holdout_once_generation",
            best_meta.completed_steps,
            metrics,
        );
        copy_checkpoint(&best_dir, &output_root.join("final"))?;
        println!("train_holdout_consumed_for_selection\tNO");
        println!("historical_validation_consumed\tNO");
        println!("historical_test_consumed\tNO");
        println!(
            "v0290_finalize_complete\tbest_step={}",
            best_meta.completed_steps
        );
        return Ok(());
    }

    let (
        mut global_step,
        mut best_epoch,
        mut best_step,
        mut best_objective,
        mut stale_epochs,
        start_epoch,
    ) = if mode == "resume" {
        let latest = read_v0290_metadata(&output_root.join("latest"))?;
        let best = read_v0290_metadata(&output_root.join("best"))?;
        validate_resume(
            &latest,
            &metadata(latest.completed_steps, latest.completed_epoch),
        )?;
        varmap.load(output_root.join("latest/model.safetensors"))?;
        optimizer.load_safetensors(output_root.join("latest/optimizer.safetensors"))?;
        optimizer.set_step_count(latest.completed_steps as u64);
        varmap.load(output_root.join("best/model.safetensors"))?;
        let best_metrics = evaluate_generation(
            &model,
            &corpus.records,
            &dev_generation_indices,
            &causal_collator,
            &spectrum_collator,
            DEV_BEAM_WIDTH_V0290,
            DEV_TOP_K_V0290,
            &device,
        )?;
        // Restore latest after evaluating the saved best checkpoint.
        varmap.load(output_root.join("latest/model.safetensors"))?;
        (
            latest.completed_steps,
            best.completed_epoch,
            best.completed_steps,
            best_metrics.objective(),
            latest.completed_epoch.saturating_sub(best.completed_epoch),
            latest.completed_epoch + 1,
        )
    } else {
        save_checkpoint(
            &output_root.join("initial"),
            &varmap,
            &optimizer,
            &metadata(0, 0),
        )?;
        let initial = evaluate_generation(
            &model,
            &corpus.records,
            &dev_generation_indices,
            &causal_collator,
            &spectrum_collator,
            DEV_BEAM_WIDTH_V0290,
            DEV_TOP_K_V0290,
            &device,
        )?;
        print_generation("train_dev_initial_generation", 0, initial);
        println!(
            "train_dev_generation_objective\tepoch=0\tstep=0\tvalue={:.8}\tbest=true",
            initial.objective()
        );
        save_checkpoint(
            &output_root.join("best"),
            &varmap,
            &optimizer,
            &metadata(0, 0),
        )?;
        (0usize, 0usize, 0usize, initial.objective(), 0usize, 1usize)
    };

    let mut stopped_early = stale_epochs >= patience;
    for epoch in start_epoch..=max_epochs {
        if stopped_early {
            break;
        }
        println!("v0290_epoch\tstage=start\tepoch={epoch}\tsteps={TRAIN_STEPS_PER_EPOCH_V0290}");
        for local_step in 0..TRAIN_STEPS_PER_EPOCH_V0290 {
            global_step += 1;
            let lr =
                lr_schedule.learning_rate(learning_rate, global_step.saturating_sub(1) as u64)?;
            optimizer.set_learning_rate(lr)?;
            let batch_indices = deterministic_training_batch(
                &train_indices,
                batch_size,
                seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ local_step as u64,
            );
            let batch_records = batch_indices
                .iter()
                .map(|&i| &corpus.records[i])
                .collect::<Vec<_>>();
            let supervised = supervised_causal_loss(
                &model,
                &batch_records,
                &causal_collator,
                &spectrum_collator,
                &device,
            )?;

            let mut group_losses = Vec::<RewardGroupLoss>::new();
            for &record_index in batch_indices
                .iter()
                .take(FOUNDATION_INVERSE_REWARD_GROUPS_PER_STEP_V0290)
            {
                group_losses.push(reward_group_loss(
                    &model,
                    &reference_model,
                    &corpus.records[record_index],
                    &causal_collator,
                    &spectrum_collator,
                    &device,
                )?);
            }
            let policy_tensors = group_losses.iter().map(|g| &g.total).collect::<Vec<_>>();
            let reward_objective = Tensor::stack(&policy_tensors, 0)?.mean_all()?;
            let total = (supervised
                .affine(FOUNDATION_INVERSE_REWARD_SUPERVISED_WEIGHT_V0290, 0.0)?
                + reward_objective)?;
            let total_value = total.to_scalar::<f32>()?;
            let update = optimizer.backward_step(&total, Some(MAX_GRADIENT_NORM_V0290))?;
            if local_step == 0
                || (local_step + 1) % 32 == 0
                || local_step + 1 == TRAIN_STEPS_PER_EPOCH_V0290
            {
                let groups = group_losses.len().max(1) as f64;
                println!(
                    "v0290_train\tepoch={epoch}\tstep={global_step}\tepoch_step={}\tlr={lr:.8}\ttotal={:.6}\tpolicy={:.6}\treference={:.6}\treward_mean={:.6}\treward_max={:.6}\tcandidates={}\tliteral_group_fraction={:.3}\til_group_fraction={:.3}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                    local_step + 1,
                    total_value,
                    group_losses.iter().map(|g| g.policy).sum::<f64>() / groups,
                    group_losses.iter().map(|g| g.reference).sum::<f64>() / groups,
                    group_losses.iter().map(|g| g.reward_mean).sum::<f64>() / groups,
                    group_losses.iter().map(|g| g.reward_max).sum::<f64>() / groups,
                    group_losses.iter().map(|g| g.candidates).sum::<usize>(),
                    group_losses.iter().filter(|g| g.literal_present).count() as f64 / groups,
                    group_losses.iter().filter(|g| g.il_present).count() as f64 / groups,
                    update.gradient_norm,
                    update.gradient_scale,
                );
            }
        }

        let dev = evaluate_generation(
            &model,
            &corpus.records,
            &dev_generation_indices,
            &causal_collator,
            &spectrum_collator,
            DEV_BEAM_WIDTH_V0290,
            DEV_TOP_K_V0290,
            &device,
        )?;
        print_generation("train_dev_generation", global_step, dev);
        let objective = dev.objective();
        let improved = best_objective - objective > min_delta;
        println!(
            "train_dev_generation_objective\tepoch={epoch}\tstep={global_step}\tvalue={objective:.8}\tprevious_best={best_objective:.8}\timproved={improved}"
        );
        save_checkpoint(
            &output_root.join("latest"),
            &varmap,
            &optimizer,
            &metadata(global_step, epoch),
        )?;
        if improved {
            best_objective = objective;
            best_epoch = epoch;
            best_step = global_step;
            stale_epochs = 0;
            save_checkpoint(
                &output_root.join("best"),
                &varmap,
                &optimizer,
                &metadata(global_step, epoch),
            )?;
            println!("v0290_best_checkpoint\tepoch={best_epoch}\tstep={best_step}\tdev_objective={best_objective:.8}");
        } else {
            stale_epochs += 1;
        }
        println!("v0290_epoch\tstage=complete\tepoch={epoch}\tstep={global_step}\tstale_epochs={stale_epochs}");
        if stale_epochs >= patience {
            stopped_early = true;
            println!("v0290_early_stop\tepoch={epoch}\tstep={global_step}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}");
        }
    }

    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!("v0290_training_complete\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}\tstopped_early={stopped_early}");
    if best_step == 0 {
        println!("v0290_no_dev_improvement_over_parent\tYES");
        println!("v0290_finalize_required\tNO");
    } else {
        println!("v0290_no_dev_improvement_over_parent\tNO");
        println!("v0290_finalize_required\tYES");
    }
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

fn supervised_causal_loss(
    model: &PeptideFoundationInverseRewardV0290Model,
    records: &[&FoundationTrainingRecord],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<Tensor> {
    let peptides = records
        .iter()
        .map(|r| r.peptidoform.clone())
        .collect::<Vec<_>>();
    let causal = causal_collator.collate(&peptides, device)?;
    let spectra = records
        .iter()
        .map(|r| {
            FoundationSpectrum::from_training_record(r)
                .ok_or_else(|| anyhow::anyhow!("reward record lacks spectrum"))
        })
        .collect::<Result<Vec<_>>>()?;
    let spectrum = spectrum_collator.collate(&spectra, device)?;
    let precursor = precursor_context(records, device)?;
    let output = model
        .causal()
        .forward_t(&causal.input, &spectrum, &precursor, true)?;
    Ok(foundation_causal_next_token_loss(&output, &causal)?)
}

fn reward_group_loss(
    model: &PeptideFoundationInverseRewardV0290Model,
    reference: &PeptideFoundationInverseRewardV0290Model,
    record: &FoundationTrainingRecord,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<RewardGroupLoss> {
    let spectrum = FoundationSpectrum::from_training_record(record)
        .ok_or_else(|| anyhow::anyhow!("v0.29 reward record lacks spectrum"))?;
    let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
    let precursor = precursor_context(&[record], device)?;
    let context = model
        .causal()
        .prepare_context(&spectrum_batch, &precursor, false)?;
    let reference_context =
        reference
            .causal()
            .prepare_context(&spectrum_batch, &precursor, false)?;
    let precursor_mass = record_precursor_mass(record)?;
    let config = model.inverse_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let mut candidates = foundation_direct_beam_search(
        precursor_mass,
        DirectDecoderBeamConfig {
            beam_width: FOUNDATION_INVERSE_REWARD_BEAM_WIDTH_V0290,
            top_k: FOUNDATION_INVERSE_REWARD_TOP_K_V0290,
            mass_tolerance_da: config.precursor_mass_tolerance_da,
            max_tokens: config.max_tokens,
        },
        |prefixes| legacy_next_logits(model, causal_collator, &context, prefixes, device),
    )
    .map_err(anyhow::Error::msg)?;

    let target_row = vocabulary
        .encode(&record.peptidoform, config.max_tokens)
        .map_err(anyhow::Error::msg)?;
    let target_active = active_row(&target_row);
    if !candidates
        .iter()
        .any(|candidate| candidate.tokens == target_active)
    {
        let target_mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        candidates.push(redeem_properties::foundation::DirectDecoderBeamCandidate {
            tokens: target_active,
            log_probability: f64::NEG_INFINITY,
            mass_error_da: target_mass - precursor_mass,
        });
    }
    candidates.sort_by(|a, b| b.log_probability.total_cmp(&a.log_probability));
    candidates.truncate(FOUNDATION_INVERSE_REWARD_TOP_K_V0290 + 1);
    if candidates.len() < 2 {
        anyhow::bail!("v0.29 reward group requires at least two candidates");
    }

    let decoded = candidates
        .iter()
        .map(|candidate| {
            vocabulary
                .decode(&candidate.tokens)
                .map_err(anyhow::Error::msg)
        })
        .collect::<Result<Vec<_>>>()?;
    let fragment_evidence = fragment_evidence_by_candidate(
        &decoded,
        &spectrum,
        &candidates
            .iter()
            .map(|candidate| candidate.mass_error_da)
            .collect::<Vec<_>>(),
        model.forward_config().max_sequence_len.saturating_sub(1),
    )?;
    let rewards = decoded
        .iter()
        .zip(candidates.iter())
        .zip(fragment_evidence.iter())
        .map(|((peptide, candidate), &fragment)| {
            foundation_sequence_reward_v0290(
                &record.peptidoform.sequence,
                &peptide.sequence,
                fragment,
                candidate.mass_error_da,
                config.precursor_mass_tolerance_da,
            )
        })
        .collect::<Vec<_>>();
    let reward_values = rewards
        .iter()
        .map(|reward| reward.total)
        .collect::<Vec<_>>();
    let advantages = foundation_group_relative_advantages_v0290(&reward_values);

    let rows = candidates
        .iter()
        .map(|candidate| pad_active_tokens(&candidate.tokens, config.max_tokens))
        .collect::<Result<Vec<_>>>()?;
    let causal_batch = causal_collator.collate_token_rows(&rows, device)?;
    let current_output =
        model
            .causal()
            .forward_t_with_context(&causal_batch.input, &context, true)?;
    let reference_output = reference.causal().forward_t_with_context(
        &causal_batch.input,
        &reference_context,
        false,
    )?;
    let current_nll = foundation_causal_sequence_mean_nlls(&current_output, &causal_batch)?;
    let reference_nll = foundation_causal_sequence_mean_nlls(&reference_output, &causal_batch)?;
    let policy = foundation_group_relative_policy_loss_v0290(&current_nll, &advantages)?;
    let reference_anchor = foundation_reference_nll_anchor_v0290(&current_nll, &reference_nll)?;
    let policy_value = policy.to_scalar::<f32>()? as f64;
    let reference_value = reference_anchor.to_scalar::<f32>()? as f64;
    let total = (policy.affine(FOUNDATION_INVERSE_REWARD_POLICY_WEIGHT_V0290, 0.0)?
        + reference_anchor.affine(FOUNDATION_INVERSE_REWARD_REFERENCE_WEIGHT_V0290, 0.0)?)?;
    Ok(RewardGroupLoss {
        total,
        policy: policy_value,
        reference: reference_value,
        reward_mean: reward_values.iter().sum::<f64>() / reward_values.len() as f64,
        reward_max: reward_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        candidates: candidates.len(),
        literal_present: rewards.iter().any(|reward| reward.literal_match),
        il_present: rewards.iter().any(|reward| reward.il_match),
    })
}

fn legacy_next_logits(
    model: &PeptideFoundationInverseRewardV0290Model,
    causal_collator: &FoundationCausalCollator,
    context: &redeem_properties::foundation::FoundationCausalContext,
    prefixes: &[Vec<u32>],
    device: &Device,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    let input = causal_collator
        .collate_compact_prefix_rows(prefixes, device)
        .map_err(|e| e.to_string())?;
    let mut rows = model
        .causal()
        .forward_next_t_with_context(&input, context, false)
        .and_then(|tensor| tensor.to_vec2::<f32>())
        .map_err(|e| e.to_string())?;
    for row in &mut rows {
        if row.len() < FOUNDATION_DIFFUSION_VOCAB_SIZE {
            return Err(format!("v0.29 causal row has {} classes", row.len()));
        }
        row.truncate(FOUNDATION_DIFFUSION_VOCAB_SIZE);
        // This first reward experiment is explicitly unmodified-only. Mask the
        // legacy discrete PTM tokens so every generated candidate matches the
        // declared evaluation domain and the sequence reward cannot accidentally
        // treat a modified peptidoform as an exact unmodified target.
        for (token, value) in row.iter_mut().enumerate() {
            let token = token as u32;
            if token != FOUNDATION_DIFFUSION_EOS
                && foundation_diffusion_token_residue(token).is_none()
            {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    Ok(rows)
}

fn fragment_evidence_by_candidate(
    peptides: &[redeem_properties::foundation::PeptidoformInput],
    spectrum: &FoundationSpectrum,
    mass_errors: &[f64],
    max_cleavages: usize,
) -> Result<Vec<f64>> {
    let predicted = peptides
        .iter()
        .map(|peptide| {
            vec![
                vec![0.0f32; FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240];
                peptide.sequence.chars().count().saturating_sub(1)
            ]
        })
        .collect::<Vec<_>>();
    let rows = foundation_fragment_relation_features(
        peptides,
        spectrum,
        &predicted,
        mass_errors,
        max_cleavages,
    )
    .map_err(anyhow::Error::msg)?;
    let mut evidence = vec![0.0f64; peptides.len()];
    for candidate in 0..peptides.len() {
        let mut matched = 0.0f64;
        let mut relations = 0.0f64;
        let mut coverage = 0.0f64;
        let mut explained = 0.0f64;
        let mut active = 0usize;
        for cleavage in 0..max_cleavages {
            if rows.mask[candidate * max_cleavages + cleavage] <= 0.5 {
                continue;
            }
            let base = (candidate * max_cleavages + cleavage)
                * FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240;
            coverage +=
                f64::from(rows.features[base + FOUNDATION_FRAGMENT_RELATION_PEAK_COVERAGE_V0240]);
            explained += f64::from(
                rows.features[base + FOUNDATION_FRAGMENT_RELATION_EXPLAINED_INTENSITY_V0240],
            );
            for channel in 0..FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240 {
                matched += f64::from(
                    rows.features
                        [base + FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240 + channel],
                );
                relations += 1.0;
            }
            active += 1;
        }
        if active > 0 {
            let matched_fraction = if relations > 0.0 {
                matched / relations
            } else {
                0.0
            };
            evidence[candidate] = (0.30 * matched_fraction
                + 0.20 * coverage / active as f64
                + 0.50 * explained / active as f64)
                .clamp(0.0, 1.0);
        }
    }
    Ok(evidence)
}

fn evaluate_generation(
    model: &PeptideFoundationInverseRewardV0290Model,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    beam_width: usize,
    top_k: usize,
    device: &Device,
) -> Result<GenerationMetrics> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut metrics = GenerationMetrics::default();
    for &index in indices {
        let record = &records[index];
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("v0.29 DEV reward record lacks spectrum"))?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum], device)?;
        let precursor = precursor_context(&[record], device)?;
        let context = model
            .causal()
            .prepare_context(&spectrum_batch, &precursor, false)?;
        let candidates = foundation_direct_beam_search(
            record_precursor_mass(record)?,
            DirectDecoderBeamConfig {
                beam_width,
                top_k,
                mass_tolerance_da: model.inverse_config().precursor_mass_tolerance_da,
                max_tokens: model.inverse_config().max_tokens,
            },
            |prefixes| legacy_next_logits(model, causal_collator, &context, prefixes, device),
        )
        .map_err(anyhow::Error::msg)?;
        let decoded = candidates
            .iter()
            .filter_map(|candidate| vocabulary.decode(&candidate.tokens).ok())
            .collect::<Vec<_>>();
        metrics.records += 1;
        metrics.returned_candidates += decoded.len();
        metrics.zero_candidate_records += usize::from(decoded.is_empty());
        if let Some(first) = decoded.first() {
            metrics.literal_top1 += usize::from(first.sequence == record.peptidoform.sequence);
            metrics.il_top1 += usize::from(
                il_sequence(&first.sequence) == il_sequence(&record.peptidoform.sequence),
            );
        }
        metrics.literal_topk += usize::from(
            decoded
                .iter()
                .any(|p| p.sequence == record.peptidoform.sequence),
        );
        metrics.il_topk += usize::from(
            decoded
                .iter()
                .any(|p| il_sequence(&p.sequence) == il_sequence(&record.peptidoform.sequence)),
        );
    }
    Ok(metrics)
}

fn print_generation(label: &str, step: usize, m: GenerationMetrics) {
    println!(
        "{label}\tstep={step}\trecords={}\tliteral_top1={:.6}\til_top1={:.6}\tliteral_top{}={:.6}\til_top{}={:.6}\tmean_returned={:.3}\tzero_candidate_records={}\tselection_score={:.6}",
        m.records,
        m.literal_top1_rate(),
        m.il_top1_rate(),
        DEV_TOP_K_V0290,
        m.literal_topk_rate(),
        DEV_TOP_K_V0290,
        m.il_topk_rate(),
        m.mean_returned(),
        m.zero_candidate_records,
        m.selection_score(),
    );
}

fn reward_eligible_indices(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    config: &FoundationDiffusionConfig,
) -> Vec<usize> {
    let vocabulary = FoundationDiffusionVocabulary;
    benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
        .filter_map(|entry| {
            let record = &records[entry.record_index];
            let unmodified = record.peptidoform.modifications.is_empty();
            let representable = vocabulary
                .encode(&record.peptidoform, config.max_tokens)
                .is_ok();
            let spectrum = FoundationSpectrum::from_training_record(record).is_some();
            let physical = record_precursor_mass(record)
                .ok()
                .zip(foundation_peptidoform_neutral_mass(&record.peptidoform).ok())
                .is_some_and(|(observed, peptide)| {
                    (peptide - observed).abs() <= config.precursor_mass_tolerance_da
                });
            (unmodified && representable && spectrum && physical).then_some(entry.record_index)
        })
        .collect()
}

fn record_precursor_mass(record: &FoundationTrainingRecord) -> Result<f64> {
    let mz = record
        .context
        .precursor_mz
        .ok_or_else(|| anyhow::anyhow!("record lacks precursor m/z"))?;
    let charge = record
        .context
        .charge
        .ok_or_else(|| anyhow::anyhow!("record lacks precursor charge"))?;
    foundation_precursor_neutral_mass(f64::from(mz), charge).map_err(anyhow::Error::msg)
}

fn precursor_context(
    records: &[&FoundationTrainingRecord],
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let charge = records
        .iter()
        .map(|r| r.context.charge.unwrap_or(0) as f32)
        .collect::<Vec<_>>();
    let charge_present = records
        .iter()
        .map(|r| if r.context.charge.is_some() { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let precursor_mz = records
        .iter()
        .map(|r| r.context.precursor_mz.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let precursor_mz_present = records
        .iter()
        .map(|r| {
            if r.context.precursor_mz.is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let nce = records
        .iter()
        .map(|r| r.context.nce.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let nce_present = records
        .iter()
        .map(|r| if r.context.nce.is_some() { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
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

fn active_row(row: &[u32]) -> Vec<u32> {
    row.iter()
        .copied()
        .take_while(|&token| token != redeem_properties::foundation::FOUNDATION_DIFFUSION_PAD)
        .collect()
}

fn pad_active_tokens(tokens: &[u32], width: usize) -> Result<Vec<u32>> {
    if tokens.is_empty()
        || tokens.len() > width
        || tokens.last().copied() != Some(redeem_properties::foundation::FOUNDATION_DIFFUSION_EOS)
    {
        anyhow::bail!("invalid v0.29 completed candidate token row");
    }
    let mut row = tokens.to_vec();
    row.resize(
        width,
        redeem_properties::foundation::FOUNDATION_DIFFUSION_PAD,
    );
    Ok(row)
}

fn deterministic_subset(indices: &[usize], n: usize, seed: u64) -> Vec<usize> {
    let mut keyed = indices
        .iter()
        .copied()
        .map(|index| (mix64(seed ^ index as u64), index))
        .collect::<Vec<_>>();
    keyed.sort_unstable();
    keyed.into_iter().take(n).map(|(_, index)| index).collect()
}

fn deterministic_training_batch(indices: &[usize], batch_size: usize, seed: u64) -> Vec<usize> {
    (0..batch_size)
        .map(|slot| {
            let key = mix64(seed ^ (slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            indices[(key as usize) % indices.len()]
        })
        .collect()
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
        .map(|aa| if aa == 'I' { 'L' } else { aa })
        .collect()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn load_exact_v0270(varmap: &VarMap, checkpoint: &Path, device: &Device) -> Result<()> {
    let tensors = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("failed to load frozen v0.27 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.29 VarMap lock poisoned"))?;
    let model_names = data.keys().cloned().collect::<BTreeSet<_>>();
    let parent_names = tensors.keys().cloned().collect::<BTreeSet<_>>();
    if model_names != parent_names {
        let missing = model_names
            .difference(&parent_names)
            .cloned()
            .collect::<Vec<_>>();
        let extra = parent_names
            .difference(&model_names)
            .cloned()
            .collect::<Vec<_>>();
        anyhow::bail!(
            "v0.29 requires exact v0.27 namespace match; missing={missing:?} extra={extra:?}"
        );
    }
    for (name, variable) in data.iter() {
        let tensor = &tensors[name];
        if tensor.dims() != variable.as_tensor().dims() {
            anyhow::bail!("v0.29 shape mismatch for {name}");
        }
        variable.set(tensor)?;
    }
    Ok(())
}

fn read_parent_metadata(checkpoint: &Path) -> Result<V0270ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(&fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?)
        .map_err(anyhow::Error::from)
}

fn read_v0290_metadata(checkpoint: &Path) -> Result<V0290CheckpointMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(&fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?)
        .map_err(anyhow::Error::from)
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V0290CheckpointMetadata,
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

fn copy_checkpoint(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for name in [
        "model.safetensors",
        "optimizer.safetensors",
        "metadata.yaml",
    ] {
        fs::copy(source.join(name), destination.join(name))?;
    }
    Ok(())
}

fn validate_resume(
    actual: &V0290CheckpointMetadata,
    expected: &V0290CheckpointMetadata,
) -> Result<()> {
    if actual.version != 290
        || actual.objective != expected.objective
        || actual.corpus_fingerprint != expected.corpus_fingerprint
        || actual.benchmark_manifest_fingerprint != expected.benchmark_manifest_fingerprint
        || actual.batch_size != expected.batch_size
        || actual.seed != expected.seed
        || actual.learning_rate != expected.learning_rate
        || actual.v0290_config != expected.v0290_config
        || actual.dev_generation_indices != expected.dev_generation_indices
        || actual.holdout_generation_indices != expected.holdout_generation_indices
    {
        anyhow::bail!("v0.29 resume metadata does not match current fixed experiment");
    }
    Ok(())
}

fn print_header(
    parent: &Path,
    config: &PeptideFoundationInverseRewardV0290Config,
    train: usize,
    dev: usize,
    holdout: usize,
) {
    println!("v0290_version\tv0.29-inverse-group-relative-physics-reward");
    println!("objective\tv0290_inverse_group_relative_physics_reward");
    println!(
        "architecture\t{}",
        FOUNDATION_INVERSE_REWARD_ARCHITECTURE_V0290
    );
    println!(
        "reward_objective\t{}",
        FOUNDATION_SEQUENCE_REWARD_OBJECTIVE_V0290
    );
    println!("parent_checkpoint\t{}", parent.display());
    println!("parent_checkpoint_weights_loaded\texact_all_v0270_namespaces");
    println!("new_model_parameters\t0");
    println!("reward_training_subset\tunmodified_mass_feasible_spectrum_records_v1");
    println!("train_reward_eligible_records\t{train}");
    println!("dev_reward_eligible_records\t{dev}");
    println!("holdout_reward_eligible_records\t{holdout}");
    println!("beam_width\t{}", config.beam_width);
    println!("reward_top_k\t{}", config.top_k);
    println!("reward_groups_per_step\t{}", config.groups_per_step);
    println!("supervised_weight\t{}", config.supervised_weight);
    println!("policy_weight\t{}", config.policy_weight);
    println!("reference_anchor_weight\t{}", config.reference_weight);
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
}

fn parse_or<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("parse argument {index}: {e}")),
        None => Ok(default),
    }
}
