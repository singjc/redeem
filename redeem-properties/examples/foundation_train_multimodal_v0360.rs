//! v0.36 protected scalar-property refinement from frozen v0.35.
//!
//! v0.35 is treated as an immutable forward anchor. This executable trains
//! only zero-initialized RT and CCS residual specialists on detached v0.35
//! representations. MS2, inverse, alignment, and relation parameters are never
//! included in the optimizer. DEV is used for checkpoint selection; the
//! TRAIN-HOLDOUT partition is touched only by explicit `finalize` mode.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_ms2_loss, load_foundation_corpus, read_foundation_training_run_config,
    sample_foundation_training_indices, sample_foundation_validation_indices, FoundationAdamW,
    FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCollator,
    FoundationCollatorConfig, FoundationCorruptionConfig, FoundationFragmentContextBatchV0350,
    FoundationLearningRateSchedule, FoundationMs2LossConfig, FoundationPartition,
    FoundationRegressionNormalization, FoundationSamplePlan, FoundationSamplingConfig,
    FoundationScalarPhysicsBatchV0360, FoundationTargetNormalizationConfig,
    FoundationTrainingRecord, PeptideFoundationMultimodalV0350Config,
    PeptideFoundationMultimodalV0360Config, PeptideFoundationMultimodalV0360Model,
    RetentionTimeObjective, FOUNDATION_CCS_STRETCH_TARGET_MAE_V0360,
    FOUNDATION_CCS_TARGET_MAE_V0360, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0360,
    FOUNDATION_RT_STRETCH_TARGET_MAE_V0360, FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V036_VERSION: u32 = 360;
const V036_OBJECTIVE: &str = "v0360_frozen_v0350_protected_rt_ccs_residual_refinement";
const V036_MAX_STEPS_PER_EPOCH: usize = 1_024;
const V036_MAX_DEV_BATCHES: usize = 32;
const V036_MAX_HOLDOUT_BATCHES: usize = 32;
const V036_RT_OBJECTIVE_WEIGHT: f64 = 1.0;
const V036_CCS_OBJECTIVE_WEIGHT: f64 = 1.0;
const V036_ROBUST_AUX_WEIGHT: f64 = 0.35;
const V036_MAX_GRADIENT_NORM: f64 = 1.0;
const V036_MATERIAL_OBJECTIVE: f64 = 0.90;
const V036_MATERIAL_RT_RATIO: f64 = 0.98;
const V036_MATERIAL_CCS_RATIO: f64 = 0.88;
const V036_MS2_INVARIANCE_TOLERANCE: f64 = 5.0e-5;

#[derive(Debug, Clone, Deserialize)]
struct V035ParentMetadata {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    completed_steps: usize,
    v0350_config: PeptideFoundationMultimodalV0350Config,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct PropertyMetrics {
    rt_mae_native: Option<f64>,
    rt_rmse_native: Option<f64>,
    ccs_mae_native: Option<f64>,
    ccs_rmse_native: Option<f64>,
    ms2_loss: Option<f64>,
    ms2_pointwise_mse: Option<f64>,
    ms2_pointwise_mae: Option<f64>,
    ms2_mean_cosine: Option<f64>,
    ms2_mean_spectral_angle: Option<f64>,
    ms2_mean_pearson: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V036Metadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_v0350_checkpoint: String,
    parent_v0350_completed_steps: usize,
    v0360_config: PeptideFoundationMultimodalV0360Config,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    train_steps: usize,
    batch_size: usize,
    dev_batches: usize,
    holdout_batches: usize,
    seed: u64,
    learning_rate: f64,
    completed_steps: usize,
    initial_dev_metrics: PropertyMetrics,
}

#[derive(Debug, Clone, Copy)]
struct WarmStartReport {
    loaded_variables: usize,
    fresh_variables: usize,
    ignored_parent_variables: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 11 {
        anyhow::bail!(
            "usage: foundation_train_multimodal_v0360 RUN_V0260.yaml OUTPUT_DIR PARENT_V0350_CHECKPOINT [max_epochs=6] [batch_size=128] [patience=2] [min_delta=0.005] [seed=20261036] [learning_rate=8e-5] [mode=train|finalize]"
        );
    }

    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let max_epochs = parse_or(&args, 4, 6usize)?;
    let batch_size = parse_or(&args, 5, 128usize)?;
    let patience = parse_or(&args, 6, 2usize)?;
    let min_delta = parse_or(&args, 7, 0.005f64)?;
    let seed = parse_or(&args, 8, 20_261_036u64)?;
    let learning_rate = parse_or(&args, 9, 8.0e-5f64)?;
    let run_mode = args.get(10).map(String::as_str).unwrap_or("train");
    let finalize_only = match run_mode {
        "train" => false,
        "finalize" => true,
        other => anyhow::bail!("unsupported v0.36 run mode {other:?}; expected train or finalize"),
    };

    if max_epochs == 0 || batch_size < 2 || patience == 0 {
        anyhow::bail!("v0.36 requires max_epochs>0, batch_size>=2, and patience>0");
    }
    if !(min_delta > 0.0 && min_delta.is_finite()) {
        anyhow::bail!("v0.36 min_delta must be positive and finite");
    }
    if !(learning_rate > 0.0 && learning_rate.is_finite()) {
        anyhow::bail!("v0.36 learning_rate must be positive and finite");
    }

    if finalize_only {
        for name in ["initial", "best"] {
            let directory = output_root.join(name);
            for file in [
                "model.safetensors",
                "optimizer.safetensors",
                "metadata.yaml",
            ] {
                if !directory.join(file).is_file() {
                    anyhow::bail!("v0.36 finalize is missing {:?}", directory.join(file));
                }
            }
        }
        if output_root.join("final").exists() {
            anyhow::bail!(
                "v0.36 final checkpoint already exists; HOLDOUT must not be consumed twice"
            );
        }
    } else if output_root.exists() {
        anyhow::bail!(
            "v0.36 TRAIN output directory already exists: {:?}",
            output_root
        );
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.36 requires a CUDA device")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let train_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
        .map(|entry| entry.record_index)
        .collect();
    let dev_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Validation)
        .map(|entry| entry.record_index)
        .collect();
    let holdout_indices: Vec<usize> = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .map(|entry| entry.record_index)
        .collect();
    for (label, indices) in [
        ("TRAIN", &train_indices),
        ("DEV", &dev_indices),
        ("HOLDOUT", &holdout_indices),
    ] {
        if indices.len() < batch_size {
            anyhow::bail!(
                "v0.36 {label} has {} records, fewer than batch_size={batch_size}",
                indices.len()
            );
        }
    }

    let parent_metadata = read_v035_metadata(&parent_checkpoint)?;
    if parent_metadata.version != 350 {
        anyhow::bail!(
            "v0.36 requires v0.35 metadata version 350, observed {}",
            parent_metadata.version
        );
    }
    if parent_metadata.objective != "v0350_trainable_forward_representation_context_conditioned_ms2"
    {
        anyhow::bail!(
            "v0.36 requires the accepted v0.35 objective, observed {}",
            parent_metadata.objective
        );
    }
    if parent_metadata.completed_steps == 0 {
        anyhow::bail!("v0.36 refuses an unselected v0.35 baseline checkpoint");
    }
    parent_metadata.v0350_config.validate()?;
    let v0360_config =
        PeptideFoundationMultimodalV0360Config::fixed(parent_metadata.v0350_config.clone())?;
    let forward_config = v0360_config.forward().clone();

    let current_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let current_benchmark_fingerprint =
        format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != current_corpus_fingerprint {
        anyhow::bail!("v0.36 corpus fingerprint differs from frozen v0.35 parent");
    }
    if parent_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint {
        anyhow::bail!("v0.36 benchmark fingerprint differs from frozen v0.35 parent");
    }

    let mut max_prepared_len = 0usize;
    for &index in train_indices
        .iter()
        .chain(&dev_indices)
        .chain(&holdout_indices)
    {
        let length = corpus.records[index].peptidoform.sequence.chars().count();
        max_prepared_len = max_prepared_len.max(length);
        if length > forward_config.max_sequence_len {
            anyhow::bail!(
                "v0.36 record {index} length {length} exceeds max_sequence_len={}",
                forward_config.max_sequence_len
            );
        }
    }

    let steps_per_epoch = (train_indices.len() / batch_size)
        .min(V036_MAX_STEPS_PER_EPOCH)
        .max(1);
    let requested_dev_batches = (dev_indices.len() / batch_size)
        .min(V036_MAX_DEV_BATCHES)
        .max(1);
    let requested_holdout_batches = (holdout_indices.len() / batch_size)
        .min(V036_MAX_HOLDOUT_BATCHES)
        .max(1);
    let max_total_steps = max_epochs.saturating_mul(steps_per_epoch);

    let mut train_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_indices,
        &dev_indices,
    );
    train_sampling.train_steps_per_epoch = Some(steps_per_epoch);
    let dev_batches = feasible_validation_batches(
        "dev",
        &train_sampling,
        &corpus.provenance,
        &dev_indices,
        batch_size,
        requested_dev_batches,
    )?;
    train_sampling.validation_steps = Some(dev_batches);
    let dev_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &dev_indices,
        batch_size,
        seed ^ 0x3600_d3f0_1234_5678,
        &train_sampling,
    )?;

    let mut holdout_sampling = filtered_sampling_config(
        &run.trainer.sampling,
        &corpus.provenance,
        &train_indices,
        &holdout_indices,
    );
    let holdout_batches = feasible_validation_batches(
        "holdout",
        &holdout_sampling,
        &corpus.provenance,
        &holdout_indices,
        batch_size,
        requested_holdout_batches,
    )?;
    holdout_sampling.validation_steps = Some(holdout_batches);
    let holdout_plan = sample_foundation_validation_indices(
        &corpus.records,
        &corpus.provenance,
        &holdout_indices,
        batch_size,
        seed ^ 0x3600_484f_4c44_4f55,
        &holdout_sampling,
    )?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultimodalV0360Model::new(v0360_config.clone(), vb)?;
    let warm_start = load_v0350_parent_variables(
        &varmap,
        &parent_checkpoint.join("model.safetensors"),
        &device,
    )?;

    let clean_collator = FoundationCollator::new(
        forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: parent_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let target_normalization = parent_metadata.target_normalization;
    let ms2_loss = parent_metadata.ms2_loss;

    let optimizer_prefixes = ["rt_specialist_v0360.", "ccs_specialist_v0360."];
    let mut optimizer = FoundationAdamW::new_for_prefixes(
        &varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: run.trainer.weight_decay,
        },
        &optimizer_prefixes,
    )?;
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 250u64.min(max_total_steps.saturating_sub(1) as u64),
        total_steps: max_total_steps as u64,
        min_lr_ratio: 0.10,
    };

    fs::create_dir_all(&output_root)?;
    println!("v0360_version\tv0.36-protected-scalar-specialists");
    println!("objective\t{V036_OBJECTIVE}");
    println!("architecture\t{}", FOUNDATION_MULTIMODAL_ARCHITECTURE_V0360);
    println!("parent_v0350_checkpoint\t{}", parent_checkpoint.display());
    println!(
        "parent_v0350_completed_steps\t{}",
        parent_metadata.completed_steps
    );
    println!(
        "warm_start_loaded_variables\t{}",
        warm_start.loaded_variables
    );
    println!("warm_start_fresh_variables\t{}", warm_start.fresh_variables);
    println!(
        "warm_start_ignored_parent_variables\t{}",
        warm_start.ignored_parent_variables
    );
    println!("device\t{:?}", device);
    println!("corpus_fingerprint\t{current_corpus_fingerprint}");
    println!("benchmark_manifest_fingerprint\t{current_benchmark_fingerprint}");
    println!("prepared_max_sequence_len\t{max_prepared_len}");
    println!("train_records\t{}", train_indices.len());
    println!("dev_records\t{}", dev_indices.len());
    println!("holdout_records\t{}", holdout_indices.len());
    println!("steps_per_epoch\t{steps_per_epoch}");
    println!("dev_batches\t{dev_batches}");
    println!("holdout_batches\t{holdout_batches}");
    println!("max_epochs\t{max_epochs}");
    println!("batch_size\t{batch_size}");
    println!("base_learning_rate\t{learning_rate}");
    println!("lr_schedule\twarmup_cosine_250_to_0.10");
    println!("optimizer_scope\trt_specialist_v0360+ccs_specialist_v0360_only");
    println!("optimizer_variable_count\t{}", optimizer.variable_count());
    println!("v0350_update_policy\tfrozen_detached_anchor");
    println!("ms2_update_policy\texact_frozen_v0350");
    println!("inverse_update_policy\texact_frozen_v0350");
    println!("rt_residual_initialization\texact_zero_step0_identity");
    println!("ccs_residual_initialization\texact_zero_step0_identity");
    println!(
        "rt_stretch_target_native_mae\t{}",
        FOUNDATION_RT_STRETCH_TARGET_MAE_V0360
    );
    println!("ccs_target_native_mae\t{}", FOUNDATION_CCS_TARGET_MAE_V0360);
    println!(
        "ccs_stretch_target_native_mae\t{}",
        FOUNDATION_CCS_STRETCH_TARGET_MAE_V0360
    );
    println!("material_dev_gate\tobjective<=0.90_and_rt_ratio<=0.98_and_ccs_ratio<=0.88_and_ms2_invariant");
    println!("historical_validation_reused_for_v0360_selection\tNO");
    println!("historical_test_consumed\tNO");
    print_sample_plan("dev_forward", &dev_plan);
    print_sample_plan("holdout_forward_reserved", &holdout_plan);
    println!("run_mode\t{run_mode}");

    if finalize_only {
        let best_dir = output_root.join("best");
        let best_metadata = read_v036_metadata(&best_dir)?;
        validate_metadata(
            &best_metadata,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &v0360_config,
            batch_size,
            seed,
            learning_rate,
            max_total_steps,
        )?;
        if best_metadata.completed_steps == 0 {
            anyhow::bail!("v0.36 has no DEV improvement; refusing to consume HOLDOUT");
        }
        varmap.load(best_dir.join("model.safetensors"))?;
        let best_dev = evaluate_properties(
            &model,
            &corpus.records,
            &dev_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        let objective = normalized_dev_objective(best_dev, best_metadata.initial_dev_metrics)?;
        validate_ms2_invariance(best_dev, best_metadata.initial_dev_metrics)?;
        let rt_ratio = finite_ratio(
            best_dev.rt_mae_native,
            best_metadata.initial_dev_metrics.rt_mae_native,
            "RT",
        )?;
        let ccs_ratio = finite_ratio(
            best_dev.ccs_mae_native,
            best_metadata.initial_dev_metrics.ccs_mae_native,
            "CCS",
        )?;
        println!("v0360_finalize_dev_objective\t{objective:.8}");
        println!("v0360_finalize_rt_ratio\t{rt_ratio:.8}");
        println!("v0360_finalize_ccs_ratio\t{ccs_ratio:.8}");
        if objective > V036_MATERIAL_OBJECTIVE
            || rt_ratio > V036_MATERIAL_RT_RATIO
            || ccs_ratio > V036_MATERIAL_CCS_RATIO
        {
            anyhow::bail!(
                "v0.36 DEV result does not satisfy the fixed materiality gate; refusing HOLDOUT"
            );
        }
        println!("v0360_finalize_only\ttrue");
        println!(
            "v0360_finalize_best_step\t{}",
            best_metadata.completed_steps
        );
        let holdout = evaluate_properties(
            &model,
            &corpus.records,
            &holdout_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        print_metrics("train_holdout_once", best_metadata.completed_steps, holdout);
        println!("train_holdout_consumed_for_selection\tNO");
        println!("historical_validation_reused_for_v0360_selection\tNO");
        println!("historical_test_consumed\tNO");
        copy_checkpoint_dir(&best_dir, &output_root.join("final"))?;
        println!(
            "v0360_finalize_complete\tbest_step={}",
            best_metadata.completed_steps
        );
        println!("final_checkpoint\t{}", output_root.join("final").display());
        return Ok(());
    }

    let initial_dev = evaluate_properties(
        &model,
        &corpus.records,
        &dev_plan.indices,
        batch_size,
        &clean_collator,
        &target_normalization,
        ms2_loss,
        &device,
    )?;
    print_metrics("train_dev_initial", 0, initial_dev);
    let initial_objective = normalized_dev_objective(initial_dev, initial_dev)?;
    println!("train_dev_objective\tepoch=0\tstep=0\tvalue={initial_objective:.8}\tbest=true");

    let metadata = |completed_steps| V036Metadata {
        version: V036_VERSION,
        objective: V036_OBJECTIVE.into(),
        architecture: FOUNDATION_MULTIMODAL_ARCHITECTURE_V0360.into(),
        corpus_fingerprint: current_corpus_fingerprint.clone(),
        benchmark_manifest_fingerprint: current_benchmark_fingerprint.clone(),
        parent_v0350_checkpoint: parent_checkpoint.display().to_string(),
        parent_v0350_completed_steps: parent_metadata.completed_steps,
        v0360_config: v0360_config.clone(),
        rt_objective: parent_metadata.rt_objective,
        target_normalization,
        ms2_loss,
        train_steps: max_total_steps,
        batch_size,
        dev_batches,
        holdout_batches,
        seed,
        learning_rate,
        completed_steps,
        initial_dev_metrics: initial_dev,
    };

    save_checkpoint(
        &output_root.join("initial"),
        &varmap,
        &optimizer,
        &metadata(0),
    )?;
    save_checkpoint(&output_root.join("best"), &varmap, &optimizer, &metadata(0))?;

    let mut global_step = 0usize;
    let mut best_epoch = 0usize;
    let mut best_step = 0usize;
    let mut best_objective = initial_objective;
    let mut stale_epochs = 0usize;
    let mut stopped_early = false;

    for epoch in 1..=max_epochs {
        let train_plan = sample_foundation_training_indices(
            &corpus.records,
            &corpus.provenance,
            &train_indices,
            batch_size,
            epoch as u64,
            seed ^ 0x3600_7a11_2233_4455,
            true,
            &train_sampling,
        )?;
        require_plan_records(
            "train",
            &train_plan,
            steps_per_epoch.saturating_mul(batch_size),
        )?;
        println!("v0360_epoch\tstage=start\tepoch={epoch}\tsteps={steps_per_epoch}");

        for local_step in 0..steps_per_epoch {
            global_step += 1;
            let lr =
                lr_schedule.learning_rate(learning_rate, global_step.saturating_sub(1) as u64)?;
            optimizer.set_learning_rate(lr)?;
            let offset = local_step.saturating_mul(batch_size);
            let selected: Vec<FoundationTrainingRecord> = train_plan.indices
                [offset..offset + batch_size]
                .iter()
                .map(|&index| corpus.records[index].clone())
                .collect();
            let loss = scalar_loss(
                &model,
                &clean_collator,
                &selected,
                &target_normalization,
                seed ^ (global_step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                &device,
            )?;
            let loss_value = f64::from(loss.to_scalar::<f32>()?);
            let update = optimizer.backward_step(&loss, Some(V036_MAX_GRADIENT_NORM))?;
            if global_step == 1 || global_step % 100 == 0 || local_step + 1 == steps_per_epoch {
                println!(
                    "v0360_train\tepoch={epoch}\tstep={global_step}\tepoch_step={}\tlr={:.8}\ttotal={loss_value:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                    local_step + 1,
                    update.learning_rate,
                    update.gradient_norm,
                    update.gradient_scale,
                );
            }
        }

        let dev = evaluate_properties(
            &model,
            &corpus.records,
            &dev_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        print_metrics("train_dev", global_step, dev);
        validate_ms2_invariance(dev, initial_dev)?;
        let objective = normalized_dev_objective(dev, initial_dev)?;
        let rt_ratio = finite_ratio(dev.rt_mae_native, initial_dev.rt_mae_native, "RT")?;
        let ccs_ratio = finite_ratio(dev.ccs_mae_native, initial_dev.ccs_mae_native, "CCS")?;
        let both_improved = rt_ratio < 1.0 && ccs_ratio < 1.0;
        let improved = both_improved && best_objective - objective > min_delta;
        println!(
            "train_dev_objective\tepoch={epoch}\tstep={global_step}\tvalue={objective:.8}\trt_ratio={rt_ratio:.8}\tccs_ratio={ccs_ratio:.8}\tprevious_best={best_objective:.8}\timproved={improved}"
        );

        save_checkpoint(
            &output_root.join("latest"),
            &varmap,
            &optimizer,
            &metadata(global_step),
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
                &metadata(global_step),
            )?;
            println!("v0360_best_checkpoint\tepoch={best_epoch}\tstep={best_step}\tdev_objective={best_objective:.8}");
        } else {
            stale_epochs += 1;
        }
        println!("v0360_epoch\tstage=complete\tepoch={epoch}\tstep={global_step}\tstale_epochs={stale_epochs}");
        if stale_epochs >= patience {
            stopped_early = true;
            println!("v0360_early_stop\tepoch={epoch}\tstep={global_step}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}");
            break;
        }
    }

    println!("train_holdout_consumed\tNO");
    println!("historical_validation_reused_for_v0360_selection\tNO");
    println!("historical_test_consumed\tNO");
    println!("v0360_training_complete\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}\tstopped_early={stopped_early}");

    if best_step == 0 {
        println!("v0360_material_dev_gain\tNO");
        println!("v0360_finalize_required\tNO");
        println!("v0360_rethink_required\tYES");
    } else {
        varmap.load(output_root.join("best/model.safetensors"))?;
        let best_dev = evaluate_properties(
            &model,
            &corpus.records,
            &dev_plan.indices,
            batch_size,
            &clean_collator,
            &target_normalization,
            ms2_loss,
            &device,
        )?;
        let rt_ratio = finite_ratio(best_dev.rt_mae_native, initial_dev.rt_mae_native, "RT")?;
        let ccs_ratio = finite_ratio(best_dev.ccs_mae_native, initial_dev.ccs_mae_native, "CCS")?;
        let material = best_objective <= V036_MATERIAL_OBJECTIVE
            && rt_ratio <= V036_MATERIAL_RT_RATIO
            && ccs_ratio <= V036_MATERIAL_CCS_RATIO;
        println!("v0360_best_rt_ratio\t{rt_ratio:.8}");
        println!("v0360_best_ccs_ratio\t{ccs_ratio:.8}");
        println!(
            "v0360_rt_stretch_target_met\t{}",
            yes_no(
                best_dev
                    .rt_mae_native
                    .is_some_and(|v| v <= FOUNDATION_RT_STRETCH_TARGET_MAE_V0360)
            )
        );
        println!(
            "v0360_ccs_target_met\t{}",
            yes_no(
                best_dev
                    .ccs_mae_native
                    .is_some_and(|v| v <= FOUNDATION_CCS_TARGET_MAE_V0360)
            )
        );
        println!(
            "v0360_ccs_stretch_target_met\t{}",
            yes_no(
                best_dev
                    .ccs_mae_native
                    .is_some_and(|v| v <= FOUNDATION_CCS_STRETCH_TARGET_MAE_V0360)
            )
        );
        if material {
            println!("v0360_material_dev_gain\tYES");
            println!("v0360_finalize_required\tYES");
            println!("v0360_rethink_required\tNO");
        } else {
            println!("v0360_material_dev_gain\tNO");
            println!("v0360_finalize_required\tNO");
            println!("v0360_rethink_required\tYES");
        }
    }
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

fn scalar_loss(
    model: &PeptideFoundationMultimodalV0360Model,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    normalization: &FoundationTargetNormalizationConfig,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let mut batch = collator.collate(records, device, seed)?;
    normalize_scalar_targets(&mut batch.targets.rt, &normalization.rt)?;
    normalize_scalar_targets(&mut batch.targets.ccs, &normalization.ccs)?;
    let physics = FoundationScalarPhysicsBatchV0360::from_records(
        records,
        model.forward_config().max_sequence_len,
        device,
    )?;
    let output = model.scalar_v0360_t(&batch.input, &batch.context, &physics, true)?;
    let mut terms = Vec::<Tensor>::new();

    if let (Some(target), Some(mask)) = (batch.targets.rt.as_ref(), batch.targets.rt_mask.as_ref())
    {
        let mse = masked_mse(&output.rt, target, mask)?;
        let robust = masked_pseudo_huber(
            &output.rt,
            target,
            mask,
            FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
        )?;
        terms.push(
            (mse + robust.affine(V036_ROBUST_AUX_WEIGHT, 0.0)?)?
                .affine(V036_RT_OBJECTIVE_WEIGHT, 0.0)?,
        );
    }
    if let (Some(target), Some(mask)) =
        (batch.targets.ccs.as_ref(), batch.targets.ccs_mask.as_ref())
    {
        let mse = masked_mse(&output.ccs, target, mask)?;
        let robust = masked_pseudo_huber(
            &output.ccs,
            target,
            mask,
            FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
        )?;
        terms.push(
            (mse + robust.affine(V036_ROBUST_AUX_WEIGHT, 0.0)?)?
                .affine(V036_CCS_OBJECTIVE_WEIGHT, 0.0)?,
        );
    }
    if terms.is_empty() {
        anyhow::bail!("v0.36 training batch contains no RT or CCS labels");
    }
    let mut total = terms[0].clone();
    for term in terms.iter().skip(1) {
        total = (total + term)?;
    }
    Ok(total.affine(1.0 / terms.len() as f64, 0.0)?)
}

fn normalize_scalar_targets(
    target: &mut Option<Tensor>,
    normalization: &FoundationRegressionNormalization,
) -> Result<()> {
    if let Some(values) = target.take() {
        *target = Some(normalization.normalize_tensor(&values)?);
    }
    Ok(())
}

fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!(
            "v0.36 scalar loss shape mismatch: prediction {:?}, target {:?}",
            prediction.dims(),
            target.dims()
        );
    }
    let mask = mask.broadcast_as(prediction.dims())?;
    let numerator = (prediction - target)?
        .sqr()?
        .broadcast_mul(&mask)?
        .sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    Ok(numerator.broadcast_div(&denominator)?)
}

fn masked_pseudo_huber(
    prediction: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    delta: f64,
) -> Result<Tensor> {
    let mask = mask.broadcast_as(prediction.dims())?;
    let scaled = (prediction - target)?.affine(1.0 / delta, 0.0)?;
    let robust = (scaled.sqr()? + 1.0)?
        .sqrt()?
        .affine(delta * delta, -(delta * delta))?;
    let numerator = robust.broadcast_mul(&mask)?.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    Ok(numerator.broadcast_div(&denominator)?)
}

fn evaluate_properties(
    model: &PeptideFoundationMultimodalV0360Model,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    collator: &FoundationCollator,
    normalization: &FoundationTargetNormalizationConfig,
    ms2_loss: FoundationMs2LossConfig,
    device: &Device,
) -> Result<PropertyMetrics> {
    let mut rt_abs = 0.0f64;
    let mut rt_sq = 0.0f64;
    let mut rt_n = 0usize;
    let mut ccs_abs = 0.0f64;
    let mut ccs_sq = 0.0f64;
    let mut ccs_n = 0usize;
    let mut ms2_objective = 0.0f64;
    let mut ms2_batches = 0usize;
    let mut ms2_shape = Ms2ShapeAccumulator::default();

    for chunk in indices.chunks(batch_size) {
        let owned: Vec<FoundationTrainingRecord> =
            chunk.iter().map(|&index| records[index].clone()).collect();
        let mut batch = collator.collate(&owned, device, 0)?;
        normalize_scalar_targets(&mut batch.targets.rt, &normalization.rt)?;
        normalize_scalar_targets(&mut batch.targets.ccs, &normalization.ccs)?;
        let fragment = FoundationFragmentContextBatchV0350::from_records(
            &owned,
            model.forward_config(),
            device,
        )?;
        let fragment_mask = fragment.channel_mask()?;
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            &owned,
            model.forward_config().max_sequence_len,
            device,
        )?;
        let output = model.forward_v0360_t(&batch.input, &batch.context, &fragment, &physics)?;
        accumulate_regression(
            &output.base.base.rt,
            batch.targets.rt.as_ref(),
            batch.targets.rt_mask.as_ref(),
            &normalization.rt,
            &mut rt_abs,
            &mut rt_sq,
            &mut rt_n,
        )?;
        accumulate_regression(
            &output.base.base.ccs,
            batch.targets.ccs.as_ref(),
            batch.targets.ccs_mask.as_ref(),
            &normalization.ccs,
            &mut ccs_abs,
            &mut ccs_sq,
            &mut ccs_n,
        )?;
        if let (Some(target), Some(mask)) = (&batch.targets.ms2, &batch.targets.ms2_mask) {
            let contextual_mask = mask.broadcast_mul(&fragment_mask)?;
            let components =
                foundation_ms2_loss(&output.base.base.ms2, target, &contextual_mask, ms2_loss)?;
            ms2_objective += f64::from(components.total.to_scalar::<f32>()?);
            ms2_batches += 1;
            ms2_shape.accumulate(&output.base.base.ms2, target, &contextual_mask)?;
        }
    }

    Ok(PropertyMetrics {
        rt_mae_native: (rt_n > 0).then(|| rt_abs / rt_n as f64),
        rt_rmse_native: (rt_n > 0).then(|| (rt_sq / rt_n as f64).sqrt()),
        ccs_mae_native: (ccs_n > 0).then(|| ccs_abs / ccs_n as f64),
        ccs_rmse_native: (ccs_n > 0).then(|| (ccs_sq / ccs_n as f64).sqrt()),
        ms2_loss: (ms2_batches > 0).then(|| ms2_objective / ms2_batches as f64),
        ms2_pointwise_mse: ms2_shape.pointwise_mse(),
        ms2_pointwise_mae: ms2_shape.pointwise_mae(),
        ms2_mean_cosine: ms2_shape.mean_cosine(),
        ms2_mean_spectral_angle: ms2_shape.mean_spectral_angle(),
        ms2_mean_pearson: ms2_shape.mean_pearson(),
    })
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

fn normalized_dev_objective(metrics: PropertyMetrics, baseline: PropertyMetrics) -> Result<f64> {
    let rt = finite_ratio(metrics.rt_mae_native, baseline.rt_mae_native, "RT")?;
    let ccs = finite_ratio(metrics.ccs_mae_native, baseline.ccs_mae_native, "CCS")?;
    validate_ms2_invariance(metrics, baseline)?;
    Ok(0.45 * rt + 0.55 * ccs)
}

fn validate_ms2_invariance(metrics: PropertyMetrics, baseline: PropertyMetrics) -> Result<()> {
    for (label, current, initial) in [
        ("MS2 loss", metrics.ms2_loss, baseline.ms2_loss),
        (
            "MS2 cosine",
            metrics.ms2_mean_cosine,
            baseline.ms2_mean_cosine,
        ),
        (
            "MS2 spectral angle",
            metrics.ms2_mean_spectral_angle,
            baseline.ms2_mean_spectral_angle,
        ),
        (
            "MS2 Pearson",
            metrics.ms2_mean_pearson,
            baseline.ms2_mean_pearson,
        ),
    ] {
        let current = current.ok_or_else(|| anyhow::anyhow!("missing {label} metric"))?;
        let initial = initial.ok_or_else(|| anyhow::anyhow!("missing baseline {label} metric"))?;
        if (current - initial).abs() > V036_MS2_INVARIANCE_TOLERANCE {
            anyhow::bail!(
                "v0.36 changed protected {label}: baseline={initial:.8} current={current:.8}"
            );
        }
    }
    Ok(())
}

fn finite_ratio(value: Option<f64>, baseline: Option<f64>, label: &str) -> Result<f64> {
    let value = value.ok_or_else(|| anyhow::anyhow!("missing {label} metric"))?;
    let baseline = baseline.ok_or_else(|| anyhow::anyhow!("missing baseline {label} metric"))?;
    if !(value.is_finite() && baseline.is_finite() && baseline > 0.0) {
        anyhow::bail!("invalid {label} ratio inputs: value={value}, baseline={baseline}");
    }
    Ok(value / baseline)
}

fn print_metrics(label: &str, step: usize, p: PropertyMetrics) {
    println!(
        "{label}_forward\tstep={step}\trt_mae_native={}\trt_rmse_native={}\tccs_mae_native={}\tccs_rmse_native={}\tms2_loss={}\tms2_pointwise_mse={}\tms2_pointwise_mae={}\tms2_cosine={}\tms2_spectral_angle={}\tms2_pearson={}",
        fmt_opt(p.rt_mae_native),
        fmt_opt(p.rt_rmse_native),
        fmt_opt(p.ccs_mae_native),
        fmt_opt(p.ccs_rmse_native),
        fmt_opt(p.ms2_loss),
        fmt_opt(p.ms2_pointwise_mse),
        fmt_opt(p.ms2_pointwise_mae),
        fmt_opt(p.ms2_mean_cosine),
        fmt_opt(p.ms2_mean_spectral_angle),
        fmt_opt(p.ms2_mean_pearson),
    );
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "NA".into())
}

#[derive(Debug, Default)]
struct Ms2ShapeAccumulator {
    fragment_count: usize,
    squared_error_sum: f64,
    absolute_error_sum: f64,
    spectrum_count: usize,
    cosine_sum: f64,
    spectral_angle_sum: f64,
    pearson_count: usize,
    pearson_sum: f64,
}

impl Ms2ShapeAccumulator {
    fn accumulate(&mut self, prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<()> {
        let predicted = prediction.to_vec3::<f32>()?;
        let targets = target.to_vec3::<f32>()?;
        let masks = mask.to_vec3::<f32>()?;
        for batch_index in 0..predicted.len() {
            let mut pred_values = Vec::new();
            let mut target_values = Vec::new();
            for residue_index in 0..predicted[batch_index].len() {
                for channel_index in 0..predicted[batch_index][residue_index].len() {
                    if masks[batch_index][residue_index][channel_index] <= 0.0 {
                        continue;
                    }
                    let pred = f64::from(predicted[batch_index][residue_index][channel_index]);
                    let truth = f64::from(targets[batch_index][residue_index][channel_index]);
                    let error = pred - truth;
                    self.fragment_count += 1;
                    self.squared_error_sum += error * error;
                    self.absolute_error_sum += error.abs();
                    pred_values.push(pred);
                    target_values.push(truth);
                }
            }
            if pred_values.is_empty() {
                continue;
            }
            let dot = pred_values
                .iter()
                .zip(&target_values)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let pnorm = pred_values.iter().map(|v| v * v).sum::<f64>().sqrt();
            let tnorm = target_values.iter().map(|v| v * v).sum::<f64>().sqrt();
            let cosine = if pnorm > 0.0 && tnorm > 0.0 {
                (dot / (pnorm * tnorm)).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            self.spectrum_count += 1;
            self.cosine_sum += cosine;
            self.spectral_angle_sum += 1.0 - (2.0 / std::f64::consts::PI) * cosine.acos();
            if let Some(pearson) = pearson_correlation(&pred_values, &target_values) {
                self.pearson_count += 1;
                self.pearson_sum += pearson;
            }
        }
        Ok(())
    }

    fn pointwise_mse(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.squared_error_sum / self.fragment_count as f64)
    }
    fn pointwise_mae(&self) -> Option<f64> {
        (self.fragment_count > 0).then(|| self.absolute_error_sum / self.fragment_count as f64)
    }
    fn mean_cosine(&self) -> Option<f64> {
        (self.spectrum_count > 0).then(|| self.cosine_sum / self.spectrum_count as f64)
    }
    fn mean_spectral_angle(&self) -> Option<f64> {
        (self.spectrum_count > 0).then(|| self.spectral_angle_sum / self.spectrum_count as f64)
    }
    fn mean_pearson(&self) -> Option<f64> {
        (self.pearson_count > 0).then(|| self.pearson_sum / self.pearson_count as f64)
    }
}

fn pearson_correlation(first: &[f64], second: &[f64]) -> Option<f64> {
    if first.len() != second.len() || first.len() < 2 {
        return None;
    }
    let n = first.len() as f64;
    let first_mean = first.iter().sum::<f64>() / n;
    let second_mean = second.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut first_squared = 0.0;
    let mut second_squared = 0.0;
    for (&a, &b) in first.iter().zip(second) {
        let da = a - first_mean;
        let db = b - second_mean;
        numerator += da * db;
        first_squared += da * da;
        second_squared += db * db;
    }
    let denominator = (first_squared * second_squared).sqrt();
    (denominator > 1.0e-12).then(|| (numerator / denominator).clamp(-1.0, 1.0))
}

fn load_v0350_parent_variables(
    varmap: &VarMap,
    checkpoint: &Path,
    device: &Device,
) -> Result<WarmStartReport> {
    let tensors = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("failed to load frozen v0.35 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.36 VarMap lock poisoned during warm start"))?;
    let mut loaded = 0usize;
    let mut fresh = 0usize;
    let mut used = BTreeSet::<String>::new();
    let mut missing = Vec::new();
    for (name, variable) in data.iter() {
        if name.starts_with("rt_specialist_v0360.") || name.starts_with("ccs_specialist_v0360.") {
            fresh += 1;
            continue;
        }
        match tensors.get(name) {
            Some(tensor) => {
                if tensor.dims() != variable.as_tensor().dims() {
                    anyhow::bail!(
                        "v0.36 warm-start shape mismatch for {name}: parent {:?}, model {:?}",
                        tensor.dims(),
                        variable.as_tensor().dims()
                    );
                }
                variable.set(tensor)?;
                used.insert(name.clone());
                loaded += 1;
            }
            None => missing.push(name.clone()),
        }
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "v0.35 parent is missing required v0.36 anchor variables: {}",
            missing.join(", ")
        );
    }
    let ignored = tensors.keys().filter(|name| !used.contains(*name)).count();
    drop(data);
    if loaded == 0 || fresh == 0 {
        anyhow::bail!("v0.36 warm start is nonfunctional: loaded={loaded} fresh={fresh}");
    }
    Ok(WarmStartReport {
        loaded_variables: loaded,
        fresh_variables: fresh,
        ignored_parent_variables: ignored,
    })
}

fn read_v035_metadata(checkpoint: &Path) -> Result<V035ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.35 metadata {path:?}"))
}

fn read_v036_metadata(checkpoint: &Path) -> Result<V036Metadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {path:?}"))?,
    )
    .with_context(|| format!("failed to parse v0.36 metadata {path:?}"))
}

#[allow(clippy::too_many_arguments)]
fn validate_metadata(
    metadata: &V036Metadata,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    config: &PeptideFoundationMultimodalV0360Config,
    batch_size: usize,
    seed: u64,
    learning_rate: f64,
    train_steps: usize,
) -> Result<()> {
    if metadata.version != V036_VERSION || metadata.objective != V036_OBJECTIVE {
        anyhow::bail!("v0.36 checkpoint identity mismatch");
    }
    if metadata.corpus_fingerprint != corpus_fingerprint
        || metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
    {
        anyhow::bail!("v0.36 checkpoint data provenance mismatch");
    }
    if metadata.v0360_config != *config {
        anyhow::bail!("v0.36 checkpoint architecture mismatch");
    }
    if metadata.batch_size != batch_size
        || metadata.seed != seed
        || metadata.train_steps != train_steps
    {
        anyhow::bail!("v0.36 checkpoint run-contract mismatch");
    }
    if (metadata.learning_rate - learning_rate).abs()
        > f64::EPSILON * 64.0 * learning_rate.abs().max(1.0)
    {
        anyhow::bail!("v0.36 checkpoint learning-rate mismatch");
    }
    Ok(())
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    optimizer: &FoundationAdamW,
    metadata: &V036Metadata,
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

fn copy_checkpoint_dir(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        anyhow::bail!("refusing to overwrite checkpoint directory {destination:?}");
    }
    fs::create_dir_all(destination)?;
    for file in [
        "model.safetensors",
        "optimizer.safetensors",
        "metadata.yaml",
    ] {
        fs::copy(source.join(file), destination.join(file))?;
    }
    Ok(())
}

fn filtered_sampling_config(
    base: &FoundationSamplingConfig,
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    train_indices: &[usize],
    validation_indices: &[usize],
) -> FoundationSamplingConfig {
    let train_sources: BTreeSet<String> = train_indices
        .iter()
        .filter_map(|&index| provenance.get(index))
        .map(|item| item.source_id.clone())
        .collect();
    let validation_sources: BTreeSet<String> = validation_indices
        .iter()
        .filter_map(|&index| provenance.get(index))
        .map(|item| item.source_id.clone())
        .collect();
    let mut config = base.clone();
    if !config.source_weights.is_empty() {
        config
            .source_weights
            .retain(|source, _| train_sources.contains(source));
    }
    if !config.validation_source_weights.is_empty() {
        config
            .validation_source_weights
            .retain(|source, _| validation_sources.contains(source));
    }
    config
}

fn feasible_validation_batches(
    label: &str,
    config: &FoundationSamplingConfig,
    provenance: &[redeem_properties::foundation::FoundationRecordProvenance],
    validation_indices: &[usize],
    batch_size: usize,
    requested_batches: usize,
) -> Result<usize> {
    let max_batches = requested_batches.min(validation_indices.len() / batch_size);
    if max_batches == 0 {
        anyhow::bail!("v0.36 {label} cannot form one full batch");
    }
    if config.validation_source_weights.is_empty() {
        return Ok(max_batches);
    }
    let mut available = BTreeMap::<String, usize>::new();
    for &index in validation_indices {
        let source = provenance
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("v0.36 {label} provenance index out of bounds"))?;
        *available.entry(source.source_id.clone()).or_default() += 1;
    }
    let total_weight: f64 = config.validation_source_weights.values().copied().sum();
    if !(total_weight > 0.0 && total_weight.is_finite()) {
        anyhow::bail!("v0.36 {label} validation weights are invalid");
    }
    for batches in (1..=max_batches).rev() {
        let quotas = weighted_quotas(
            batches * batch_size,
            &config.validation_source_weights,
            total_weight,
        );
        if quotas
            .iter()
            .all(|(source, desired)| *desired <= available.get(source).copied().unwrap_or(0))
        {
            return Ok(batches);
        }
    }
    anyhow::bail!("v0.36 {label} cannot satisfy source quotas for one full batch")
}

fn weighted_quotas(
    target_records: usize,
    weights: &BTreeMap<String, f64>,
    total_weight: f64,
) -> BTreeMap<String, usize> {
    let mut quotas = BTreeMap::<String, usize>::new();
    let mut fractions = Vec::<(f64, String)>::new();
    let mut assigned = 0usize;
    for (source, weight) in weights {
        let exact = target_records as f64 * *weight / total_weight;
        let base = exact.floor() as usize;
        assigned += base;
        quotas.insert(source.clone(), base);
        fractions.push((exact - base as f64, source.clone()));
    }
    fractions.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    let mut remaining = target_records.saturating_sub(assigned);
    for (_, source) in fractions {
        if remaining == 0 {
            break;
        }
        *quotas.entry(source).or_default() += 1;
        remaining -= 1;
    }
    quotas
}

fn require_plan_records(label: &str, plan: &FoundationSamplePlan, expected: usize) -> Result<()> {
    if plan.indices.len() != expected {
        anyhow::bail!(
            "v0.36 {label} sample plan has {} records but {expected} are required",
            plan.indices.len()
        );
    }
    Ok(())
}

fn print_sample_plan(label: &str, plan: &FoundationSamplePlan) {
    println!(
        "sample_plan\tlane={label}\trecords={}\tunique_records={}",
        plan.indices.len(),
        plan.unique_records
    );
    for (source, count) in &plan.source_records {
        println!("sample_plan_source\tlane={label}\tsource={source}\trecords={count}");
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
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
