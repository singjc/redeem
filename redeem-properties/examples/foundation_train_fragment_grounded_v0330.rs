//! v0.33 hypothesis-specific fragment-grounded spectrum->peptide decoding.
//!
//! Scientific question: v0.32 showed that search breadth can increase candidate
//! coverage while correct-sequence recall stays almost flat. v0.33 therefore
//! freezes the complete v0.27 causal decoder and learns one small nonlinear
//! candidate-specific residual. At every prefix, each possible next residue is
//! scored using explicit precursor-mass/chemistry features plus observed b/y
//! fragment support. Beam width/search legality are held fixed at the v0.32
//! baseline so improvements must come from the model distribution, not search.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_causal_next_token_loss, foundation_diffusion_token_residue,
    foundation_direct_beam_search, foundation_direct_conditioning_loss,
    foundation_direct_shuffled_order, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    ChemistrySuffixMassLattice, ChemistryTransitionFeaturizer, DirectDecoderBeamConfig,
    FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCausalCollator,
    FoundationCausalOutput, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationLearningRateSchedule, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumCollator, FoundationTrainingRecord, FragmentGroundedTransitionHeadV0330,
    PeptideFoundationMultimodalV0270Config, PeptideSpectrumCausalModel, PrecursorContextBatch,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190, FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
    FOUNDATION_FRAGMENT_GROUNDED_ARCHITECTURE_V0330, FOUNDATION_FRAGMENT_GROUNDED_OBJECTIVE_V0330,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TRAIN_STEPS_PER_EPOCH_V0330: usize = 256;
const DEV_RECORDS_V0330: usize = 128;
const HOLDOUT_RECORDS_V0330: usize = 128;
const BEAM_WIDTH_V0330: usize = 32;
const TOP_K_V0330: usize = 16;
const MAX_GRADIENT_NORM_V0330: f64 = 1.0;
const BASELINE_RECORDS_V0330: usize = 128;
const BASELINE_LITERAL_TOP1_V0330: usize = 0;
const BASELINE_IL_TOP1_V0330: usize = 1;
const BASELINE_LITERAL_TOP16_V0330: usize = 1;
const BASELINE_IL_TOP16_V0330: usize = 2;
const BASELINE_ZERO_CANDIDATES_V0330: usize = 69;

#[derive(Debug, Deserialize)]
struct V0270ParentMetadata {
    version: u32,
    objective: String,
    completed_steps: usize,
    v0270_config: PeptideFoundationMultimodalV0270Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct V0330CheckpointMetadata {
    version: u32,
    objective: String,
    architecture: String,
    parent_checkpoint: String,
    parent_completed_steps: usize,
    completed_steps: usize,
    completed_epoch: usize,
    batch_size: usize,
    seed: u64,
    learning_rate: f64,
    max_epochs: usize,
    patience: usize,
    min_delta: f64,
    dev_indices: Vec<usize>,
    holdout_indices: Vec<usize>,
    inverse_config: FoundationDiffusionConfig,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct GenerationMetrics {
    records: usize,
    literal_top1: usize,
    il_top1: usize,
    literal_top5: usize,
    il_top5: usize,
    literal_top10: usize,
    il_top10: usize,
    literal_top16: usize,
    il_top16: usize,
    returned_candidates: usize,
    zero_candidate_records: usize,
    mass_valid_candidates: usize,
    mass_error_abs_sum: f64,
}

impl GenerationMetrics {
    fn rate(self, value: usize) -> f64 {
        if self.records == 0 {
            0.0
        } else {
            value as f64 / self.records as f64
        }
    }
    fn coverage(self) -> f64 {
        1.0 - self.rate(self.zero_candidate_records)
    }
    fn mean_returned(self) -> f64 {
        self.rate(self.returned_candidates)
    }
    fn mass_valid_fraction(self) -> f64 {
        if self.returned_candidates == 0 {
            0.0
        } else {
            self.mass_valid_candidates as f64 / self.returned_candidates as f64
        }
    }
    fn mean_abs_mass_error(self) -> f64 {
        if self.returned_candidates == 0 {
            0.0
        } else {
            self.mass_error_abs_sum / self.returned_candidates as f64
        }
    }
    fn selection_score(self) -> f64 {
        // Correctness dominates. Coverage has only a small tie-breaking role so
        // v0.33 cannot repeat v0.32's "more candidates, fewer correct" outcome.
        0.35 * self.rate(self.il_top16)
            + 0.25 * self.rate(self.literal_top16)
            + 0.25 * self.rate(self.il_top1)
            + 0.10 * self.rate(self.literal_top1)
            + 0.05 * self.coverage()
    }
    fn objective(self) -> f64 {
        1.0 - self.selection_score()
    }
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 4 || args.len() > 11 {
        anyhow::bail!(
            "usage: foundation_train_fragment_grounded_v0330 RUN_V0260.yaml OUTPUT_DIR PARENT_V0270_CHECKPOINT [max_epochs=8] [batch_size=32] [patience=3] [min_delta=0.005] [seed=20261033] [learning_rate=1e-4] [mode=train|finalize]"
        );
    }
    let run_yaml = &args[1];
    let out = PathBuf::from(&args[2]);
    let parent = PathBuf::from(&args[3]);
    let max_epochs = parse_or(&args, 4, 8usize)?;
    let batch_size = parse_or(&args, 5, 32usize)?;
    let patience = parse_or(&args, 6, 3usize)?;
    let min_delta = parse_or(&args, 7, 0.005f64)?;
    let seed = parse_or(&args, 8, 20_261_033u64)?;
    let learning_rate = parse_or(&args, 9, 1.0e-4f64)?;
    let mode = args.get(10).map(String::as_str).unwrap_or("train");
    if !matches!(mode, "train" | "finalize") {
        anyhow::bail!("v0.33 mode must be train or finalize");
    }
    if max_epochs == 0
        || batch_size < 2
        || patience == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        anyhow::bail!("v0.33 training configuration is invalid");
    }
    if mode == "train" && out.exists() {
        anyhow::bail!("v0.33 output directory already exists: {out:?}");
    }
    if mode == "finalize" && !out.is_dir() {
        anyhow::bail!("v0.33 finalize requires existing output directory: {out:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.33 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(run_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;
    let parent_meta = read_parent_metadata(&parent)?;
    if parent_meta.version != 270
        || parent_meta.objective != "v0270_rt_specialist_contextual_fragment_tokens_openptm32"
    {
        anyhow::bail!(
            "v0.33 requires frozen v0.27 parent; observed version={} objective={:?}",
            parent_meta.version,
            parent_meta.objective
        );
    }
    let inverse = parent_meta.v0270_config.inverse.clone();
    inverse.validate().map_err(anyhow::Error::msg)?;

    let train_indices = eligible_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &inverse,
    );
    let dev_eligible = eligible_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &inverse,
    );
    let holdout_eligible = eligible_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Test,
        &inverse,
    );
    if train_indices.len() < batch_size
        || dev_eligible.len() < DEV_RECORDS_V0330
        || holdout_eligible.len() < HOLDOUT_RECORDS_V0330
    {
        anyhow::bail!(
            "insufficient v0.33 eligible records: train={} dev={} holdout={} batch={batch_size}",
            train_indices.len(),
            dev_eligible.len(),
            holdout_eligible.len()
        );
    }
    // Preserve the exact v0.29/v0.32 DEV ordering for scientific continuity.
    let dev_indices = deterministic_subset(
        &dev_eligible,
        DEV_RECORDS_V0330,
        20_260_929u64 ^ 0x5644_4556_3032_3930,
    );
    let holdout_indices = deterministic_subset(
        &holdout_eligible,
        HOLDOUT_RECORDS_V0330,
        seed ^ 0x484f_4c44_3033_3030,
    );

    let max_precursor_mass = train_indices
        .iter()
        .chain(dev_indices.iter())
        .chain(holdout_indices.iter())
        .filter_map(|&i| record_precursor_mass(&corpus.records[i]).ok())
        .fold(0.0f64, f64::max);
    let suffix_lattice =
        ChemistrySuffixMassLattice::new(inverse.max_tokens, max_precursor_mass + 100.0)
            .map_err(anyhow::Error::msg)?;
    let chemistry =
        ChemistryTransitionFeaturizer::new(&inverse, suffix_lattice).map_err(anyhow::Error::msg)?;
    let causal_collator = FoundationCausalCollator::new_open_ptm(inverse.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(inverse.spectrum.clone())?;

    let base_varmap = VarMap::new();
    let base = PeptideSpectrumCausalModel::new_open_ptm(
        inverse.clone(),
        VarBuilder::from_varmap(&base_varmap, DType::F32, &device),
    )?;
    let (base_loaded, parent_ignored) =
        load_frozen_causal_subset(&base_varmap, &parent.join("model.safetensors"), &device)?;

    let mut head_varmap = VarMap::new();
    let head = FragmentGroundedTransitionHeadV0330::new(
        inverse.model_dim,
        VarBuilder::from_varmap(&head_varmap, DType::F32, &device),
    )?;
    let head_variables = head_varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.33 head VarMap lock poisoned"))?
        .len();

    fs::create_dir_all(&out)?;
    print_header(
        &parent,
        parent_meta.completed_steps,
        &inverse,
        train_indices.len(),
        base_loaded,
        parent_ignored,
        head_variables,
        max_precursor_mass,
    );

    if mode == "finalize" {
        let best = out.join("model/best");
        let meta = read_checkpoint_metadata(&best)?;
        if meta.completed_steps == 0 {
            anyhow::bail!("v0.33 finalize refused: trained DEV checkpoint never beat step 0");
        }
        head_varmap.load(best.join("head.safetensors"))?;
        let metrics = evaluate_generation(
            &base,
            &head,
            &chemistry,
            &corpus.records,
            &holdout_indices,
            &causal_collator,
            &spectrum_collator,
            &inverse,
            &device,
        )?;
        print_generation(
            "train_holdout_once_generation",
            meta.completed_steps,
            metrics,
        );
        fs::create_dir_all(out.join("model/final"))?;
        fs::copy(
            best.join("head.safetensors"),
            out.join("model/final/head.safetensors"),
        )?;
        fs::copy(
            best.join("metadata.yaml"),
            out.join("model/final/metadata.yaml"),
        )?;
        println!("train_holdout_consumed_for_selection\tNO");
        println!("historical_validation_consumed\tNO");
        println!("historical_test_consumed\tNO");
        println!(
            "v0330_finalize_complete\tbest_step={}",
            meta.completed_steps
        );
        println!("final_checkpoint\t{}", out.join("model/final").display());
        return Ok(());
    }

    let mut optimizer = FoundationAdamW::new(
        &head_varmap,
        FoundationAdamWConfig {
            learning_rate,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: run.trainer.weight_decay,
        },
    )?;
    let total_steps = max_epochs * TRAIN_STEPS_PER_EPOCH_V0330;
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 128u64.min(total_steps.saturating_sub(1) as u64),
        total_steps: total_steps as u64,
        min_lr_ratio: 0.10,
    };

    let metadata = |step: usize, epoch: usize| V0330CheckpointMetadata {
        version: 330,
        objective: FOUNDATION_FRAGMENT_GROUNDED_OBJECTIVE_V0330.into(),
        architecture: FOUNDATION_FRAGMENT_GROUNDED_ARCHITECTURE_V0330.into(),
        parent_checkpoint: parent.display().to_string(),
        parent_completed_steps: parent_meta.completed_steps,
        completed_steps: step,
        completed_epoch: epoch,
        batch_size,
        seed,
        learning_rate,
        max_epochs,
        patience,
        min_delta,
        dev_indices: dev_indices.clone(),
        holdout_indices: holdout_indices.clone(),
        inverse_config: inverse.clone(),
        historical_validation_consumed: false,
        historical_test_consumed: false,
    };

    let initial = evaluate_generation(
        &base,
        &head,
        &chemistry,
        &corpus.records,
        &dev_indices,
        &causal_collator,
        &spectrum_collator,
        &inverse,
        &device,
    )?;
    print_generation("train_dev_initial_generation", 0, initial);
    validate_step0_contract(initial)?;
    save_checkpoint(&out.join("model/initial"), &head_varmap, &metadata(0, 0))?;
    save_checkpoint(&out.join("model/best"), &head_varmap, &metadata(0, 0))?;
    println!(
        "train_dev_generation_objective\tepoch=0\tstep=0\tvalue={:.8}\tbest=true",
        initial.objective()
    );

    let mut global_step = 0usize;
    let mut best_epoch = 0usize;
    let mut best_step = 0usize;
    let mut best = initial;
    let mut best_objective = initial.objective();
    let mut stale_epochs = 0usize;

    for epoch in 1..=max_epochs {
        println!("v0330_epoch\tstage=start\tepoch={epoch}\tsteps={TRAIN_STEPS_PER_EPOCH_V0330}");
        for local_step in 0..TRAIN_STEPS_PER_EPOCH_V0330 {
            global_step += 1;
            let lr =
                lr_schedule.learning_rate(learning_rate, global_step.saturating_sub(1) as u64)?;
            optimizer.set_learning_rate(lr)?;
            let batch_indices = deterministic_training_batch(
                &train_indices,
                batch_size,
                seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ local_step as u64,
            );
            let selected = batch_indices
                .iter()
                .map(|&i| &corpus.records[i])
                .collect::<Vec<_>>();
            let (loss, matched_nll, shuffled_nll) = training_loss(
                &base,
                &head,
                &chemistry,
                &selected,
                &causal_collator,
                &spectrum_collator,
                &device,
                seed ^ global_step as u64,
            )?;
            let total_value = loss.to_scalar::<f32>()?;
            let update = optimizer.backward_step(&loss, Some(MAX_GRADIENT_NORM_V0330))?;
            if local_step == 0
                || (local_step + 1) % 32 == 0
                || local_step + 1 == TRAIN_STEPS_PER_EPOCH_V0330
            {
                println!(
                    "v0330_train\tepoch={epoch}\tstep={global_step}\tepoch_step={}\tlr={lr:.8}\ttotal={total_value:.6}\tmatched_nll={matched_nll:.6}\tshuffled_nll={shuffled_nll:.6}\tconditioning_gap={:.6}\tgradient_norm={:.6}\tgradient_scale={:.6}",
                    local_step + 1,
                    shuffled_nll - matched_nll,
                    update.gradient_norm,
                    update.gradient_scale,
                );
            }
        }

        let dev = evaluate_generation(
            &base,
            &head,
            &chemistry,
            &corpus.records,
            &dev_indices,
            &causal_collator,
            &spectrum_collator,
            &inverse,
            &device,
        )?;
        print_generation("train_dev_generation", global_step, dev);
        let objective = dev.objective();
        let improved = objective + min_delta < best_objective;
        println!(
            "train_dev_generation_objective\tepoch={epoch}\tstep={global_step}\tvalue={objective:.8}\tprevious_best={best_objective:.8}\timproved={improved}"
        );
        if improved {
            best = dev;
            best_objective = objective;
            best_epoch = epoch;
            best_step = global_step;
            stale_epochs = 0;
            save_checkpoint(
                &out.join("model/best"),
                &head_varmap,
                &metadata(global_step, epoch),
            )?;
            println!("v0330_best_checkpoint\tepoch={epoch}\tstep={global_step}\tdev_objective={objective:.8}");
        } else {
            stale_epochs += 1;
        }
        save_checkpoint(
            &out.join("model/latest"),
            &head_varmap,
            &metadata(global_step, epoch),
        )?;
        println!("v0330_epoch\tstage=complete\tepoch={epoch}\tstep={global_step}\tstale_epochs={stale_epochs}");
        if stale_epochs >= patience {
            println!("v0330_early_stop\tepoch={epoch}\tstep={global_step}\tpatience={patience}\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}");
            break;
        }
    }

    let baseline_coverage =
        1.0 - BASELINE_ZERO_CANDIDATES_V0330 as f64 / BASELINE_RECORDS_V0330 as f64;
    let coverage_guard = best.coverage() + 0.02 >= baseline_coverage;
    let candidate_recall_gain = best.il_top16 >= BASELINE_IL_TOP16_V0330 + 6
        && best.literal_top16 >= BASELINE_LITERAL_TOP16_V0330 + 3;
    let ranking_gain = best.il_top1 >= BASELINE_IL_TOP1_V0330 + 2
        || best.literal_top1 >= BASELINE_LITERAL_TOP1_V0330 + 2;
    let material = best_step > 0 && coverage_guard && candidate_recall_gain && ranking_gain;

    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!("v0330_training_complete\tbest_epoch={best_epoch}\tbest_step={best_step}\tbest_dev_objective={best_objective:.8}");
    println!(
        "v0330_candidate_recall_gain\t{}",
        yes_no(candidate_recall_gain)
    );
    println!("v0330_ranking_gain\t{}", yes_no(ranking_gain));
    println!("v0330_coverage_guard\t{}", yes_no(coverage_guard));
    println!("v0330_material_dev_gain\t{}", yes_no(material));
    println!("v0330_finalize_required\t{}", yes_no(material));
    println!("v0330_rethink_required\t{}", yes_no(!material));
    println!(
        "v0330_next_action\t{}",
        if material {
            "one_time_train_holdout_finalize"
        } else {
            "close_fragment_grounded_residual_lane_and_rethink_inverse_architecture"
        }
    );
    println!("best_checkpoint\t{}", out.join("model/best").display());
    Ok(())
}

fn training_loss(
    base: &PeptideSpectrumCausalModel,
    head: &FragmentGroundedTransitionHeadV0330,
    chemistry: &ChemistryTransitionFeaturizer,
    records: &[&FoundationTrainingRecord],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
    shuffle_seed: u64,
) -> Result<(Tensor, f64, f64)> {
    let peptides = records
        .iter()
        .map(|r| r.peptidoform.clone())
        .collect::<Vec<_>>();
    let spectra = records
        .iter()
        .map(|r| {
            FoundationSpectrum::from_training_record(r)
                .ok_or_else(|| anyhow::anyhow!("v0.33 TRAIN record lacks spectrum"))
        })
        .collect::<Result<Vec<_>>>()?;
    let masses = records
        .iter()
        .map(|r| record_precursor_mass(r))
        .collect::<Result<Vec<_>>>()?;
    let charges = records
        .iter()
        .map(|r| {
            r.context
                .charge
                .ok_or_else(|| anyhow::anyhow!("v0.33 TRAIN record lacks charge"))
        })
        .collect::<Result<Vec<_>>>()?;
    let causal = causal_collator.collate(&peptides, device)?;
    let precursor = precursor_context(records, device)?;

    let matched_spectrum = spectrum_collator.collate(&spectra, device)?;
    let matched_features =
        chemistry.teacher_forced(&peptides, &spectra, &masses, &charges, device)?;
    let matched_base = base.forward_t(&causal.input, &matched_spectrum, &precursor, false)?;
    let matched_out =
        mask_unmodified_output(head.apply_teacher_forced(matched_base, &matched_features)?)?;
    let matched = foundation_causal_next_token_loss(&matched_out, &causal)?;

    let order = foundation_direct_shuffled_order(records.len(), shuffle_seed)?;
    let shuffled_spectra = order
        .iter()
        .map(|&i| spectra[i].clone())
        .collect::<Vec<_>>();
    let shuffled_spectrum = spectrum_collator.collate(&shuffled_spectra, device)?;
    let shuffled_features =
        chemistry.teacher_forced(&peptides, &shuffled_spectra, &masses, &charges, device)?;
    let shuffled_base = base.forward_t(&causal.input, &shuffled_spectrum, &precursor, false)?;
    let shuffled_out =
        mask_unmodified_output(head.apply_teacher_forced(shuffled_base, &shuffled_features)?)?;
    let shuffled = foundation_causal_next_token_loss(&shuffled_out, &causal)?;

    let matched_value = matched.to_scalar::<f32>()? as f64;
    let shuffled_value = shuffled.to_scalar::<f32>()? as f64;
    let total = foundation_direct_conditioning_loss(
        &matched,
        &shuffled,
        FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190,
        FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
    )?;
    Ok((total, matched_value, shuffled_value))
}

fn mask_unmodified_output(mut output: FoundationCausalOutput) -> Result<FoundationCausalOutput> {
    let classes = output.token_logits.dims3()?.2;
    let mut bias = vec![-1.0e9f32; classes];
    for (token, value) in bias.iter_mut().enumerate() {
        let token = token as u32;
        if token == FOUNDATION_DIFFUSION_EOS || foundation_diffusion_token_residue(token).is_some()
        {
            *value = 0.0;
        }
    }
    let bias = Tensor::from_vec(bias, classes, output.token_logits.device())?;
    output.token_logits = output.token_logits.broadcast_add(&bias)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_generation(
    base: &PeptideSpectrumCausalModel,
    head: &FragmentGroundedTransitionHeadV0330,
    chemistry: &ChemistryTransitionFeaturizer,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    device: &Device,
) -> Result<GenerationMetrics> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut m = GenerationMetrics::default();
    for &index in indices {
        let record = &records[index];
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("v0.33 DEV record {index} lacks spectrum"))?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
        let precursor = precursor_context(&[record], device)?;
        let context = base.prepare_context(&spectrum_batch, &precursor, false)?;
        let neutral_mass = record_precursor_mass(record)?;
        let charge = record
            .context
            .charge
            .ok_or_else(|| anyhow::anyhow!("v0.33 DEV record lacks charge"))?;
        let candidates = foundation_direct_beam_search(
            neutral_mass,
            DirectDecoderBeamConfig {
                beam_width: BEAM_WIDTH_V0330,
                top_k: TOP_K_V0330,
                mass_tolerance_da: config.precursor_mass_tolerance_da,
                max_tokens: config.max_tokens,
            },
            |prefixes| {
                next_logits(
                    base,
                    head,
                    chemistry,
                    causal_collator,
                    &context,
                    prefixes,
                    &spectrum,
                    neutral_mass,
                    charge,
                    device,
                )
            },
        )
        .map_err(anyhow::Error::msg)?;

        let decoded = candidates
            .iter()
            .filter_map(|candidate| vocabulary.decode(&candidate.tokens).ok())
            .collect::<Vec<_>>();
        let literal_rank = decoded
            .iter()
            .position(|p| p == &record.peptidoform)
            .map(|i| i + 1);
        let target_il = il_sequence(&record.peptidoform.sequence);
        let il_rank = decoded
            .iter()
            .position(|p| il_sequence(&p.sequence) == target_il)
            .map(|i| i + 1);
        m.records += 1;
        m.returned_candidates += candidates.len();
        m.zero_candidate_records += usize::from(candidates.is_empty());
        m.mass_valid_candidates += candidates
            .iter()
            .filter(|c| c.mass_error_da.abs() <= config.precursor_mass_tolerance_da)
            .count();
        m.mass_error_abs_sum += candidates
            .iter()
            .map(|c| c.mass_error_da.abs())
            .sum::<f64>();
        m.literal_top1 += usize::from(literal_rank == Some(1));
        m.il_top1 += usize::from(il_rank == Some(1));
        m.literal_top5 += usize::from(literal_rank.is_some_and(|r| r <= 5));
        m.il_top5 += usize::from(il_rank.is_some_and(|r| r <= 5));
        m.literal_top10 += usize::from(literal_rank.is_some_and(|r| r <= 10));
        m.il_top10 += usize::from(il_rank.is_some_and(|r| r <= 10));
        m.literal_top16 += usize::from(literal_rank.is_some_and(|r| r <= TOP_K_V0330));
        m.il_top16 += usize::from(il_rank.is_some_and(|r| r <= TOP_K_V0330));
    }
    Ok(m)
}

#[allow(clippy::too_many_arguments)]
fn next_logits(
    base: &PeptideSpectrumCausalModel,
    head: &FragmentGroundedTransitionHeadV0330,
    chemistry: &ChemistryTransitionFeaturizer,
    causal_collator: &FoundationCausalCollator,
    context: &redeem_properties::foundation::FoundationCausalContext,
    prefixes: &[Vec<u32>],
    spectrum: &FoundationSpectrum,
    precursor_mass: f64,
    charge: i32,
    device: &Device,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    let input = causal_collator
        .collate_compact_prefix_rows(prefixes, device)
        .map_err(|e| e.to_string())?;
    let output = base
        .forward_t_with_context(&input, context, false)
        .map_err(|e| e.to_string())?;
    let (_, token_len, _) = output.token_logits.dims3().map_err(|e| e.to_string())?;
    let hidden = output
        .decoder_hidden
        .narrow(1, token_len - 1, 1)
        .and_then(|x| x.squeeze(1))
        .and_then(|x| x.contiguous())
        .map_err(|e| e.to_string())?;
    let base_logits = output
        .token_logits
        .narrow(1, token_len - 1, 1)
        .and_then(|x| x.squeeze(1))
        .and_then(|x| x.contiguous())
        .map_err(|e| e.to_string())?;
    let features = chemistry
        .next_prefixes(prefixes, spectrum, precursor_mass, charge, device)
        .map_err(|e| e.to_string())?;
    let mut rows = head
        .apply_next(&hidden, &base_logits, &features)
        .and_then(|x| x.to_vec2::<f32>())
        .map_err(|e| e.to_string())?;
    for row in &mut rows {
        if row.len() < FOUNDATION_DIFFUSION_VOCAB_SIZE {
            return Err(format!("v0.33 causal row has {} classes", row.len()));
        }
        row.truncate(FOUNDATION_DIFFUSION_VOCAB_SIZE);
        // First architecture test remains on the exact v0.29/v0.32 unmodified
        // domain. Open-PTM search should only follow if the grounded decoder
        // materially improves ordinary sequence recovery.
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

fn eligible_indices(
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
        .map(|r| {
            if r.context.charge.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect::<Vec<_>>();
    let precursor_mz = records
        .iter()
        .map(|r| r.context.precursor_mz.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let precursor_mz_present = records
        .iter()
        .map(|r| {
            if r.context.precursor_mz.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect::<Vec<_>>();
    let nce = records
        .iter()
        .map(|r| r.context.nce.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let nce_present = records
        .iter()
        .map(|r| {
            if r.context.nce.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
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

fn deterministic_training_batch(indices: &[usize], n: usize, seed: u64) -> Vec<usize> {
    let mut keyed = indices
        .iter()
        .copied()
        .map(|i| (mix64(seed ^ i as u64), i))
        .collect::<Vec<_>>();
    keyed.sort_unstable();
    keyed.into_iter().take(n).map(|(_, i)| i).collect()
}

fn deterministic_subset(indices: &[usize], n: usize, seed: u64) -> Vec<usize> {
    deterministic_training_batch(indices, n, seed)
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

fn validate_step0_contract(m: GenerationMetrics) -> Result<()> {
    let exact = m.records == BASELINE_RECORDS_V0330
        && m.literal_top1 == BASELINE_LITERAL_TOP1_V0330
        && m.il_top1 == BASELINE_IL_TOP1_V0330
        && m.literal_top16 == BASELINE_LITERAL_TOP16_V0330
        && m.il_top16 == BASELINE_IL_TOP16_V0330
        && m.zero_candidate_records == BASELINE_ZERO_CANDIDATES_V0330;
    println!("v0330_step0_v0320_baseline_contract\t{}", yes_no(exact));
    if !exact {
        anyhow::bail!(
            "v0.33 step-0 drift from frozen v0.32 baseline: records={} literal_top1={} il_top1={} literal_top16={} il_top16={} zero={}",
            m.records, m.literal_top1, m.il_top1, m.literal_top16, m.il_top16, m.zero_candidate_records
        );
    }
    Ok(())
}

fn print_generation(label: &str, step: usize, m: GenerationMetrics) {
    println!(
        "{label}\tstep={step}\trecords={}\tcandidate_coverage={:.6}\tzero_candidate_records={}\tmean_returned={:.3}\tliteral_top1={:.6}\til_top1={:.6}\tliteral_top5={:.6}\til_top5={:.6}\tliteral_top10={:.6}\til_top10={:.6}\tliteral_top16={:.6}\til_top16={:.6}\tmass_valid_fraction={:.6}\tmean_abs_mass_error_da={:.6}\tselection_score={:.6}",
        m.records, m.coverage(), m.zero_candidate_records, m.mean_returned(),
        m.rate(m.literal_top1), m.rate(m.il_top1), m.rate(m.literal_top5), m.rate(m.il_top5),
        m.rate(m.literal_top10), m.rate(m.il_top10), m.rate(m.literal_top16), m.rate(m.il_top16),
        m.mass_valid_fraction(), m.mean_abs_mass_error(), m.selection_score(),
    );
}

#[allow(clippy::too_many_arguments)]
fn print_header(
    parent: &Path,
    parent_steps: usize,
    inverse: &FoundationDiffusionConfig,
    train_records: usize,
    base_loaded: usize,
    parent_ignored: usize,
    head_variables: usize,
    max_precursor_mass: f64,
) {
    println!("v0330_version\tv0.33-hypothesis-specific-fragment-grounded-decoder");
    println!("objective\t{FOUNDATION_FRAGMENT_GROUNDED_OBJECTIVE_V0330}");
    println!("architecture\t{FOUNDATION_FRAGMENT_GROUNDED_ARCHITECTURE_V0330}");
    println!("parent_checkpoint\t{}", parent.display());
    println!("parent_completed_steps\t{parent_steps}");
    println!("base_causal_update_policy\tfrozen_v0270_stop_gradient_by_optimizer_scope");
    println!("base_causal_loaded_variables\t{base_loaded}");
    println!("parent_variables_ignored_outside_causal\t{parent_ignored}");
    println!("new_fragment_grounded_variables\t{head_variables}");
    println!("fragment_feature_source\tv0200_mass_chemistry_plus_observed_b_y_support_reused_as_physics_featurizer_only");
    println!("fragment_feature_role\tinside_next_token_logit_not_posthoc_reranking");
    println!("transition_residual_initialization\texact_zero_output_step0_identity");
    println!("inverse_domain\tunmodified_mass_feasible_first_architecture_test");
    println!("beam_width\t{BEAM_WIDTH_V0330}");
    println!("return_top_k\t{TOP_K_V0330}");
    println!("search_policy\tfrozen_v0320_baseline_beam_no_mass_strata_no_width_sweep");
    println!("train_records\t{train_records}");
    println!("max_precursor_mass_for_soft_suffix_feature\t{max_precursor_mass:.6}");
    println!("inverse_model_dim\t{}", inverse.model_dim);
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!("train_holdout_consumed\tNO");
}

fn save_checkpoint(
    dir: &Path,
    head_varmap: &VarMap,
    metadata: &V0330CheckpointMetadata,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    head_varmap.save(dir.join("head.safetensors"))?;
    fs::write(dir.join("metadata.yaml"), serde_yaml::to_string(metadata)?)?;
    Ok(())
}

fn read_checkpoint_metadata(dir: &Path) -> Result<V0330CheckpointMetadata> {
    let path = dir.join("metadata.yaml");
    serde_yaml::from_str(&fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?)
        .map_err(anyhow::Error::from)
}

fn load_frozen_causal_subset(
    varmap: &VarMap,
    checkpoint: &Path,
    device: &Device,
) -> Result<(usize, usize)> {
    let parent = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("load frozen v0.27 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.33 base VarMap lock poisoned"))?;
    let mut loaded = 0usize;
    let mut consumed = BTreeSet::new();
    for (name, variable) in data.iter() {
        let tensor = parent
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("v0.33 parent missing causal variable {name}"))?;
        if tensor.dims() != variable.as_tensor().dims() {
            anyhow::bail!("v0.33 parent shape mismatch for {name}");
        }
        variable.set(tensor)?;
        consumed.insert(name.clone());
        loaded += 1;
    }
    Ok((
        loaded,
        parent
            .keys()
            .filter(|name| !consumed.contains(*name))
            .count(),
    ))
}

fn read_parent_metadata(checkpoint: &Path) -> Result<V0270ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(&fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?)
        .map_err(anyhow::Error::from)
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
            .map_err(|e| anyhow::anyhow!("parse argument {index}: {e}")),
        None => Ok(default),
    }
}
