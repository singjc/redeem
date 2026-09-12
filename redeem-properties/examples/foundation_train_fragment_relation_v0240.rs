//! v0.24 same-spectrum hard-negative fragment-to-peak relational energy.
//!
//! The accepted v0.13.23 proposal pool is retained unchanged.  A frozen unified
//! forward model predicts expected b/y intensities for each complete candidate.
//! The trainable v0.24 head sees only explicit cleavage<->peak relations plus
//! local residue/modification context and same-spectrum peak competition.  TRAIN
//! positives compete listwise against the top 31 non-I/L-equivalent proposal
//! survivors from the same precursor.  Validation labels are used only after
//! scores are frozen. TEST is never consumed.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_compatibility_listwise_loss, foundation_fragment_relation_features,
    foundation_fragment_relation_legacy_log_prior, foundation_precursor_mass_error_da,
    load_foundation_corpus, parse_modified_peptide, read_foundation_training_run_config,
    FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCollator,
    FoundationCollatorConfig, FoundationConfig, FoundationCorruptionConfig,
    FoundationDiffusionConfig, FoundationFragmentRelationBatch,
    FoundationFragmentRelationFeatureRows, FoundationPartition, FoundationSpectrum,
    FoundationTrainingRecord, PeptideFoundationUnifiedModel, PeptideSpectrumFragmentRelationEnergy,
    PeptidoformInput, FOUNDATION_FRAGMENT_RELATION_ARCHITECTURE_V0240,
    FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240, FOUNDATION_FRAGMENT_RELATION_OBJECTIVE_V0240,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const VERSION: &str = "v0.24.0";
const PROPOSAL_NEGATIVES_PER_ANCHOR: usize = 31;
const PROPOSAL_LIST_WIDTH: usize = PROPOSAL_NEGATIVES_PER_ANCHOR + 1;
const MIN_FULL_PROPOSAL_GROUPS: usize = 10_000;
const FULL_TRAIN_STEPS: usize = 2_500;
const FULL_ANCHOR_BATCH: usize = 4;
const SMOKE_TRAIN_STEPS: usize = 1;
const SMOKE_ANCHOR_BATCH: usize = 1;
const SMOKE_VALIDATION_GROUPS: usize = 2;
const FORWARD_CANDIDATE_BATCH: usize = 128;
const ACCEPTED_MAX_SEQUENCE_LEN: usize = 64;
const LEARNING_RATE: f64 = 1.0e-4;
const WEIGHT_DECAY: f64 = 1.0e-4;
const MAX_GRADIENT_NORM: f64 = 5.0;
const LOG_EVERY_STEPS: usize = 100;
const SEED: u64 = 20_260_924;

const REQUIRED_RECORDS: usize = 125;
const REQUIRED_ORACLE_LITERAL: usize = 44;
const REQUIRED_ORACLE_IL: usize = 54;
const REQUIRED_LEGACY_LITERAL: usize = 24;
const REQUIRED_LEGACY_IL: usize = 38;
const PROGRESS_LITERAL: usize = 28;
const PROGRESS_IL: usize = 42;
const MATERIAL_LITERAL: usize = 26;
const MATERIAL_IL: usize = 40;

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
            other => anyhow::bail!("v0.24 mode must be full or smoke, got '{other}'"),
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
struct UnifiedMetadata {
    forward_config: FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
}

struct FrozenForwardPredictor {
    _varmap: VarMap,
    model: PeptideFoundationUnifiedModel,
    metadata: UnifiedMetadata,
}

impl FrozenForwardPredictor {
    fn load(checkpoint: &Path, device: &Device) -> Result<Self> {
        let metadata_path = checkpoint.join("metadata.yaml");
        let metadata: UnifiedMetadata = serde_yaml::from_str(
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
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = PeptideFoundationUnifiedModel::new(
            metadata.forward_config.clone(),
            metadata.inverse_config.clone(),
            vb,
        )?;
        let model_path = checkpoint.join("model.safetensors");
        varmap
            .load(&model_path)
            .with_context(|| format!("load frozen unified model {model_path:?}"))?;
        Ok(Self {
            _varmap: varmap,
            model,
            metadata,
        })
    }
}

#[derive(Debug, Clone)]
struct CandidateRow {
    sequence: String,
    modifications: String,
    exact: bool,
    il_exact: bool,
    legacy_rank: usize,
    mass_error_da: f64,
}

#[derive(Debug, Clone)]
struct CandidateGroup {
    record_index: usize,
    rows: Vec<CandidateRow>,
}

#[derive(Debug, Clone)]
struct ProposalHardGroup {
    anchor_index: usize,
    positive: CandidateRow,
    negatives: Vec<CandidateRow>,
}

#[derive(Debug, Clone, Copy)]
struct ProposalMiningStats {
    input_candidate_groups: usize,
    spectrum_anchors: usize,
    positive_exact_present_groups: usize,
    positive_il_present_groups: usize,
    retained_raw_spectra: usize,
    retained_fallback_spectra: usize,
    mean_positive_legacy_rank: f64,
    mean_selected_legacy_rank: f64,
    max_selected_legacy_rank: usize,
}

#[derive(Debug, Clone, Copy)]
struct ValidationContract {
    records: usize,
    oracle_literal: usize,
    oracle_il: usize,
    legacy_literal: usize,
    legacy_il: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct ValidationMetrics {
    records: usize,
    oracle_literal: usize,
    oracle_il: usize,
    legacy_literal: usize,
    legacy_il: usize,
    relation_only_literal: usize,
    relation_only_il: usize,
    global_literal: usize,
    global_il: usize,
    global_literal_top5: usize,
    global_il_top5: usize,
    global_literal_top10: usize,
    global_il_top10: usize,
    true_rank_improved: usize,
    true_rank_worsened: usize,
    true_rank_tied: usize,
    raw_spectra: usize,
    fallback_spectra: usize,
    candidates_scored: usize,
    matched_relations: usize,
    contested_peaks: usize,
}

#[derive(Debug, Serialize)]
struct RelationMetadata {
    version: String,
    run_mode: String,
    architecture: String,
    objective: String,
    proposal_policy: String,
    negative_sampling_policy: String,
    global_energy: String,
    legacy_prior_formula: String,
    relation_output_initialization: String,
    candidate_labels_used_as_model_features: bool,
    legacy_rank_used_as_model_feature: bool,
    test_partition_consumed: bool,
    unified_checkpoint: String,
    training_candidate_tsv: String,
    validation_candidate_tsv: String,
    relation_feature_dim: usize,
    train_steps: usize,
    anchor_batch: usize,
    list_width: usize,
    learning_rate: f64,
    seed: u64,
    train_proposal_groups: usize,
    initialization_fingerprint: String,
    final_validation_records: usize,
    final_literal_top1: usize,
    final_il_top1: usize,
    validation_oracle_literal: usize,
    validation_oracle_il: usize,
    stop_rule: String,
}

#[derive(Debug)]
struct RelationTrainingBatch {
    relations: FoundationFragmentRelationBatch,
    prior_scores: Tensor,
    anchor_record_indices: Vec<usize>,
    matched_relations: usize,
    contested_peaks: usize,
    mean_selected_negative_legacy_rank: f64,
    max_selected_negative_legacy_rank: usize,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 5 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_train_fragment_relation_v0240 RUN.yaml UNIFIED_CHECKPOINT TRAIN_PROPOSALS.tsv VALIDATION_CANDIDATES.tsv OUTPUT_DIR [smoke|full]"
        );
    }
    let run_yaml = PathBuf::from(&args[0]);
    let unified_checkpoint = PathBuf::from(&args[1]);
    let train_tsv = PathBuf::from(&args[2]);
    let validation_tsv = PathBuf::from(&args[3]);
    let output_dir = PathBuf::from(&args[4]);
    let mode = RunMode::parse(args.get(5).map(String::as_str))?;
    reject_test_path(&train_tsv)?;
    reject_test_path(&validation_tsv)?;

    let started = Instant::now();
    stage(started, "start");
    println!("version\t{VERSION}");
    println!("mode\t{}", mode.as_str());
    println!("architecture\t{FOUNDATION_FRAGMENT_RELATION_ARCHITECTURE_V0240}");
    println!("objective\t{FOUNDATION_FRAGMENT_RELATION_OBJECTIVE_V0240}");
    println!("proposal_policy\tv01323_final_two_view_fixed_budget_frozen");
    println!("candidate_labels_used_as_model_features\tfalse");
    println!("legacy_rank_used_as_model_feature\tfalse");
    println!("legacy_score_used_as_model_feature\tfalse");
    println!("global_energy\tfixed_negative_log_legacy_rank_prior_plus_learned_fragment_relation_residual");
    println!("legacy_prior_formula\t-negative_ln_one_based_legacy_rank");
    println!("relation_output_initialization\tzero_residual_exact_legacy_baseline_at_step0");
    println!("forward_ms2_model\tfrozen_unified_v01310_train_derived");
    println!("relation_features\tleft_right_residue+local_modification+predicted_core_intensity+observed_peak_intensity+mass_error+complementary_by+same_spectrum_peak_claim_competition+precursor_error");
    println!("test_partition_consumed\tfalse");

    let run = read_foundation_training_run_config(&run_yaml)
        .with_context(|| format!("read run config {run_yaml:?}"))?;
    stage(started, "run_config_loaded");
    let corpus = load_foundation_corpus(&run.corpus).context("load foundation corpus")?;
    stage(started, "corpus_loaded");
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("read benchmark manifest {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;
    stage(started, "benchmark_validated");

    let train_candidate_groups = read_candidate_groups(&train_tsv, true)
        .with_context(|| format!("read TRAIN proposal TSV {train_tsv:?}"))?;
    verify_partition(
        &train_candidate_groups,
        &benchmark,
        FoundationPartition::Train,
        "TRAIN",
    )?;
    let (proposal_groups, proposal_stats) =
        build_proposal_hard_groups(&train_candidate_groups, &corpus.records)?;
    drop(train_candidate_groups);
    if proposal_groups.len() < mode.anchor_batch() {
        anyhow::bail!(
            "v0.24 retained {} TRAIN hard-negative groups, fewer than anchor batch {}",
            proposal_groups.len(),
            mode.anchor_batch()
        );
    }
    if mode == RunMode::Full && proposal_groups.len() < MIN_FULL_PROPOSAL_GROUPS {
        anyhow::bail!(
            "v0.24 full training requires at least {MIN_FULL_PROPOSAL_GROUPS} frozen v0.13.23 TRAIN hard-negative groups; observed {}",
            proposal_groups.len()
        );
    }
    stage(started, "train_hard_negative_groups_ready");

    let validation_groups = read_candidate_groups(&validation_tsv, false)
        .with_context(|| format!("read validation candidate TSV {validation_tsv:?}"))?;
    verify_partition(
        &validation_groups,
        &benchmark,
        FoundationPartition::Validation,
        "VALIDATION",
    )?;
    let validation_contract = validate_candidate_contract(&validation_groups)?;
    stage(started, "validation_candidates_ready");

    let device = Device::cuda_if_available(0)?;
    println!("device\t{device:?}");
    let predictor = FrozenForwardPredictor::load(&unified_checkpoint, &device)?;
    verify_checkpoint_fingerprints(&predictor.metadata, &corpus, &benchmark)?;
    if predictor.metadata.forward_config.model_dim != 96 {
        anyhow::bail!(
            "v0.24 frozen forward checkpoint must use model_dim=96, observed {}",
            predictor.metadata.forward_config.model_dim
        );
    }
    if predictor.metadata.forward_config.max_sequence_len != ACCEPTED_MAX_SEQUENCE_LEN {
        anyhow::bail!(
            "v0.24 frozen forward checkpoint must use max_sequence_len={ACCEPTED_MAX_SEQUENCE_LEN}, observed {}",
            predictor.metadata.forward_config.max_sequence_len
        );
    }
    if predictor.metadata.forward_config.ms2_fragment_channels < 4 {
        anyhow::bail!("v0.24 requires at least four forward MS2 channels");
    }
    let collator = FoundationCollator::new(
        predictor.metadata.forward_config.clone(),
        FoundationCollatorConfig {
            retention_time_objective: run.trainer.collator.retention_time_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    println!(
        "forward_model_ms2_channels\t{}",
        predictor.metadata.forward_config.ms2_fragment_channels
    );
    println!("forward_candidate_batch\t{FORWARD_CANDIDATE_BATCH}");
    stage(started, "frozen_forward_model_loaded");

    let relation_varmap = VarMap::new();
    let relation_vb = VarBuilder::from_varmap(&relation_varmap, DType::F32, &device);
    let relation_model = PeptideSpectrumFragmentRelationEnergy::new(relation_vb)?;
    initialize_relation_variables(&relation_varmap, SEED, &device)?;
    let initialization_fingerprint = relation_model_fingerprint(&relation_varmap)?;
    let mut optimizer = FoundationAdamW::new(
        &relation_varmap,
        FoundationAdamWConfig {
            learning_rate: LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            weight_decay: WEIGHT_DECAY,
        },
    )?;
    stage(started, "relation_model_initialized");

    println!("relation_feature_dim\t{FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240}");
    println!("proposal_negatives_per_anchor\t{PROPOSAL_NEGATIVES_PER_ANCHOR}");
    println!("training_list_width\t{PROPOSAL_LIST_WIDTH}");
    println!(
        "negative_sampling_policy\ttop_31_frozen_v01323_legacy_rank_non_il_equivalent_deduplicated"
    );
    println!("train_steps\t{}", mode.train_steps());
    println!("anchor_batch\t{}", mode.anchor_batch());
    println!(
        "candidate_pairs_per_optimizer_step\t{}",
        mode.anchor_batch() * PROPOSAL_LIST_WIDTH
    );
    println!("learning_rate\t{LEARNING_RATE}");
    println!("weight_decay\t{WEIGHT_DECAY}");
    println!("max_gradient_norm\t{MAX_GRADIENT_NORM}");
    println!("seed\t{SEED}");
    println!("training_candidate_tsv\t{}", train_tsv.display());
    println!("validation_candidate_tsv\t{}", validation_tsv.display());
    println!(
        "train_candidate_groups_input\t{}",
        proposal_stats.input_candidate_groups
    );
    println!(
        "train_spectrum_anchors\t{}",
        proposal_stats.spectrum_anchors
    );
    println!("train_proposal_groups\t{}", proposal_groups.len());
    println!(
        "train_positive_exact_present_groups\t{}",
        proposal_stats.positive_exact_present_groups
    );
    println!(
        "train_positive_il_present_groups\t{}",
        proposal_stats.positive_il_present_groups
    );
    println!(
        "train_raw_observed_spectrum_records\t{}",
        proposal_stats.retained_raw_spectra
    );
    println!(
        "train_annotated_fragment_spectrum_fallback_records\t{}",
        proposal_stats.retained_fallback_spectra
    );
    println!(
        "proposal_positive_legacy_rank_mean\t{:.3}",
        proposal_stats.mean_positive_legacy_rank
    );
    println!(
        "proposal_negative_legacy_rank_mean\t{:.3}",
        proposal_stats.mean_selected_legacy_rank
    );
    println!(
        "proposal_negative_legacy_rank_max\t{}",
        proposal_stats.max_selected_legacy_rank
    );
    println!("initialization_fingerprint\t{initialization_fingerprint}");
    println!(
        "validation_candidate_contract\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}",
        validation_contract.records,
        validation_contract.oracle_literal,
        validation_contract.oracle_il,
        validation_contract.legacy_literal,
        validation_contract.legacy_il,
    );

    fs::create_dir_all(&output_dir)?;
    let initial_diagnostics_path =
        output_dir.join(format!("validation_relation_initial_{}.tsv", mode.as_str()));
    let initial_metrics = evaluate_validation(
        &validation_groups,
        &corpus.records,
        &predictor,
        &collator,
        &relation_model,
        &device,
        mode.validation_limit(),
        &initial_diagnostics_path,
    )?;
    if initial_metrics.global_literal != initial_metrics.legacy_literal
        || initial_metrics.global_il != initial_metrics.legacy_il
    {
        anyhow::bail!(
            "v0.24 zero-residual initialization must reproduce the frozen legacy baseline: global={}/{} legacy={}/{}",
            initial_metrics.global_literal,
            initial_metrics.global_il,
            initial_metrics.legacy_literal,
            initial_metrics.legacy_il,
        );
    }
    println!(
        "initial_global_relation_literal_top1\t{}",
        initial_metrics.global_literal
    );
    println!(
        "initial_global_relation_il_top1\t{}",
        initial_metrics.global_il
    );
    println!(
        "initial_relation_only_literal_top1\t{}",
        initial_metrics.relation_only_literal
    );
    println!(
        "initial_relation_only_il_top1\t{}",
        initial_metrics.relation_only_il
    );
    stage(started, "initial_validation_complete");

    let mut sampler = AnchorSampler::new(proposal_groups.len(), SEED);
    let mut unique_anchors = HashSet::<usize>::new();
    let mut window_loss = 0.0f64;
    let mut window_accuracy = 0.0f64;
    let mut window_grad_norm = 0.0f64;
    let mut window_grad_scale = 0.0f64;
    let mut window_matched_relations = 0.0f64;
    let mut window_contested_peaks = 0.0f64;
    let mut window_rank_mean = 0.0f64;
    let mut window_rank_max = 0usize;
    let mut window_steps = 0usize;

    for step0 in 0..mode.train_steps() {
        let step = step0 + 1;
        let group_indices = sampler.next_batch(mode.anchor_batch());
        let batch = build_training_batch(
            &group_indices,
            &proposal_groups,
            &corpus.records,
            &predictor,
            &collator,
            &device,
            SEED ^ step as u64,
        )?;
        for &record_index in &batch.anchor_record_indices {
            unique_anchors.insert(record_index);
        }
        let relation_scores =
            relation_model.forward_grouped(&batch.relations, PROPOSAL_LIST_WIDTH)?;
        let scores = (&relation_scores + &batch.prior_scores)?;
        let loss = foundation_compatibility_listwise_loss(&scores)?;
        let loss_value = f64::from(loss.to_scalar::<f32>()?);
        let accuracy = positive_first_accuracy(&scores)?;
        let optimizer_step = optimizer.backward_step(&loss, Some(MAX_GRADIENT_NORM))?;

        window_loss += loss_value;
        window_accuracy += accuracy;
        window_grad_norm += optimizer_step.gradient_norm;
        window_grad_scale += optimizer_step.gradient_scale;
        window_matched_relations += batch.matched_relations as f64;
        window_contested_peaks += batch.contested_peaks as f64;
        window_rank_mean += batch.mean_selected_negative_legacy_rank;
        window_rank_max = window_rank_max.max(batch.max_selected_negative_legacy_rank);
        window_steps += 1;

        if step == 1 || step % LOG_EVERY_STEPS == 0 || step == mode.train_steps() {
            println!(
                "relation_training\tstep={}\tsteps_total={}\tmean_listwise_loss={:.8}\tmean_train_top1_fraction={:.6}\tmean_gradient_norm={:.6}\tmean_gradient_scale={:.6}\tmean_matched_relations={:.1}\tmean_contested_peaks={:.1}\tmean_selected_negative_legacy_rank={:.3}\tmax_selected_negative_legacy_rank={}\tunique_anchor_groups_seen={}",
                step,
                mode.train_steps(),
                window_loss / window_steps as f64,
                window_accuracy / window_steps as f64,
                window_grad_norm / window_steps as f64,
                window_grad_scale / window_steps as f64,
                window_matched_relations / window_steps as f64,
                window_contested_peaks / window_steps as f64,
                window_rank_mean / window_steps as f64,
                window_rank_max,
                unique_anchors.len(),
            );
            window_loss = 0.0;
            window_accuracy = 0.0;
            window_grad_norm = 0.0;
            window_grad_scale = 0.0;
            window_matched_relations = 0.0;
            window_contested_peaks = 0.0;
            window_rank_mean = 0.0;
            window_rank_max = 0;
            window_steps = 0;
        }
    }
    stage(started, "optimizer_loop_complete");

    let model_path = output_dir.join("fragment_relation_model.safetensors");
    let optimizer_path = output_dir.join("optimizer.safetensors");
    relation_varmap.save(&model_path)?;
    optimizer.save_safetensors(&optimizer_path)?;

    let diagnostics_path = output_dir.join(format!("validation_relation_{}.tsv", mode.as_str()));
    let metrics = evaluate_validation(
        &validation_groups,
        &corpus.records,
        &predictor,
        &collator,
        &relation_model,
        &device,
        mode.validation_limit(),
        &diagnostics_path,
    )?;
    stage(started, "validation_complete");

    print_validation_metrics(&metrics);
    let stop_rule = if mode == RunMode::Smoke {
        "SMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION"
    } else {
        enforce_frozen_contract(&metrics)?;
        if metrics.global_literal >= PROGRESS_LITERAL && metrics.global_il >= PROGRESS_IL {
            "V0240_PROGRESS_TARGET_MET_HARD_NEGATIVE_FRAGMENT_RELATION"
        } else if metrics.global_literal >= MATERIAL_LITERAL && metrics.global_il >= MATERIAL_IL {
            "V0240_PARTIAL_SUCCESS_ALLOW_ONE_RAW_SPECTRUM_UNCERTAINTY_UPGRADE"
        } else if metrics.global_literal >= REQUIRED_LEGACY_LITERAL
            && metrics.global_il >= REQUIRED_LEGACY_IL
        {
            "V0240_BASELINE_PRESERVED_RELATIONAL_SIGNAL_INSUFFICIENT_STOP"
        } else {
            "STOP_FRAGMENT_PEAK_RELATION_REASSESS_PROPOSAL_TOPOLOGY"
        }
    };
    if mode == RunMode::Full {
        println!(
            "baseline_recovery_gate\t{}",
            yes_no(
                metrics.global_literal >= REQUIRED_LEGACY_LITERAL
                    && metrics.global_il >= REQUIRED_LEGACY_IL
            )
        );
        println!(
            "progress_gate\t{}",
            yes_no(metrics.global_literal >= PROGRESS_LITERAL && metrics.global_il >= PROGRESS_IL)
        );
        println!(
            "material_improvement_threshold\tliteral>={MATERIAL_LITERAL}_and_il>={MATERIAL_IL}"
        );
    }
    println!("v0240_stop_rule\t{stop_rule}");
    println!("test_partition_consumed\tfalse");

    let metadata = RelationMetadata {
        version: VERSION.into(),
        run_mode: mode.as_str().into(),
        architecture: FOUNDATION_FRAGMENT_RELATION_ARCHITECTURE_V0240.into(),
        objective: FOUNDATION_FRAGMENT_RELATION_OBJECTIVE_V0240.into(),
        proposal_policy: "v01323_final_two_view_fixed_budget_frozen".into(),
        negative_sampling_policy: "top_31_frozen_v01323_legacy_rank_non_il_equivalent_deduplicated"
            .into(),
        global_energy:
            "fixed_negative_log_legacy_rank_prior_plus_learned_fragment_relation_residual".into(),
        legacy_prior_formula: "-ln(one_based_legacy_rank)".into(),
        relation_output_initialization: "zero_residual_exact_legacy_baseline_at_step0".into(),
        candidate_labels_used_as_model_features: false,
        legacy_rank_used_as_model_feature: false,
        test_partition_consumed: false,
        unified_checkpoint: unified_checkpoint.display().to_string(),
        training_candidate_tsv: train_tsv.display().to_string(),
        validation_candidate_tsv: validation_tsv.display().to_string(),
        relation_feature_dim: FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240,
        train_steps: mode.train_steps(),
        anchor_batch: mode.anchor_batch(),
        list_width: PROPOSAL_LIST_WIDTH,
        learning_rate: LEARNING_RATE,
        seed: SEED,
        train_proposal_groups: proposal_groups.len(),
        initialization_fingerprint,
        final_validation_records: metrics.records,
        final_literal_top1: metrics.global_literal,
        final_il_top1: metrics.global_il,
        validation_oracle_literal: metrics.oracle_literal,
        validation_oracle_il: metrics.oracle_il,
        stop_rule: stop_rule.into(),
    };
    serde_yaml::to_writer(
        BufWriter::new(File::create(output_dir.join("metadata.yaml"))?),
        &metadata,
    )?;

    println!("checkpoint\t{}", model_path.display());
    println!("optimizer_checkpoint\t{}", optimizer_path.display());
    println!("validation_diagnostics\t{}", diagnostics_path.display());
    stage(started, "complete");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_training_batch(
    group_indices: &[usize],
    groups: &[ProposalHardGroup],
    records: &[FoundationTrainingRecord],
    predictor: &FrozenForwardPredictor,
    collator: &FoundationCollator,
    device: &Device,
    seed: u64,
) -> Result<RelationTrainingBatch> {
    let mut candidate_records =
        Vec::<FoundationTrainingRecord>::with_capacity(group_indices.len() * PROPOSAL_LIST_WIDTH);
    let mut all_peptides =
        Vec::<PeptidoformInput>::with_capacity(group_indices.len() * PROPOSAL_LIST_WIDTH);
    let mut all_mass_errors = Vec::<f64>::with_capacity(group_indices.len() * PROPOSAL_LIST_WIDTH);
    let mut spectra = Vec::<FoundationSpectrum>::with_capacity(group_indices.len());
    let mut prior_values = Vec::<f32>::with_capacity(group_indices.len() * PROPOSAL_LIST_WIDTH);
    let mut anchor_record_indices = Vec::<usize>::with_capacity(group_indices.len());
    let mut selected_rank_sum = 0.0f64;
    let mut selected_rank_count = 0usize;
    let mut selected_rank_max = 0usize;

    for &group_index in group_indices {
        let group = groups
            .get(group_index)
            .with_context(|| format!("proposal group index {group_index} out of range"))?;
        let anchor = records
            .get(group.anchor_index)
            .with_context(|| format!("TRAIN anchor {} out of range", group.anchor_index))?;
        if group.negatives.len() != PROPOSAL_NEGATIVES_PER_ANCHOR {
            anyhow::bail!("v0.24 internal TRAIN proposal width mismatch");
        }
        spectra.push(
            FoundationSpectrum::from_training_record(anchor).with_context(|| {
                format!(
                    "TRAIN anchor {} lacks observed spectrum",
                    group.anchor_index
                )
            })?,
        );
        anchor_record_indices.push(group.anchor_index);

        let mut peptides = Vec::<PeptidoformInput>::with_capacity(PROPOSAL_LIST_WIDTH);
        let mut mass_errors = Vec::<f64>::with_capacity(PROPOSAL_LIST_WIDTH);
        peptides.push(anchor.peptidoform.clone());
        mass_errors.push(precursor_mass_error(anchor, &anchor.peptidoform));
        prior_values.push(foundation_fragment_relation_legacy_log_prior(
            group.positive.legacy_rank,
        ));
        for negative in &group.negatives {
            peptides.push(exported_peptidoform(negative)?);
            mass_errors.push(negative.mass_error_da);
            prior_values.push(foundation_fragment_relation_legacy_log_prior(
                negative.legacy_rank,
            ));
            selected_rank_sum += negative.legacy_rank as f64;
            selected_rank_count += 1;
            selected_rank_max = selected_rank_max.max(negative.legacy_rank);
        }

        for peptide in &peptides {
            let mut record = anchor.clone();
            record.peptidoform = peptide.clone();
            record.retention_time = Default::default();
            record.ccs = None;
            record.fragments.clear();
            candidate_records.push(record);
        }
        all_peptides.extend(peptides);
        all_mass_errors.extend(mass_errors);
    }

    let batch = collator.collate(&candidate_records, device, seed)?;
    let predicted_ms2 = predictor
        .model
        .forward()
        .forward_ms2_t(&batch.input, &batch.context, false)?
        .detach();
    let predicted = predicted_ms2.to_vec3::<f32>()?;
    if predicted.len() != all_peptides.len() {
        anyhow::bail!("v0.24 frozen forward prediction count mismatch");
    }

    let max_cleavages = predictor.metadata.forward_config.max_sequence_len - 1;
    let mut feature_groups =
        Vec::<FoundationFragmentRelationFeatureRows>::with_capacity(group_indices.len());
    let mut matched_relations = 0usize;
    let mut contested_peaks = 0usize;
    for group_local in 0..group_indices.len() {
        let start = group_local * PROPOSAL_LIST_WIDTH;
        let end = start + PROPOSAL_LIST_WIDTH;
        let rows = foundation_fragment_relation_features(
            &all_peptides[start..end],
            &spectra[group_local],
            &predicted[start..end],
            &all_mass_errors[start..end],
            max_cleavages,
        )
        .map_err(anyhow::Error::msg)?;
        matched_relations += rows.matched_relations;
        contested_peaks += rows.contested_peaks;
        feature_groups.push(rows);
    }
    let relations = FoundationFragmentRelationBatch::cat(&feature_groups, device)?;
    let prior_scores = Tensor::from_vec(
        prior_values,
        (group_indices.len(), PROPOSAL_LIST_WIDTH),
        device,
    )?;

    Ok(RelationTrainingBatch {
        relations,
        prior_scores,
        anchor_record_indices,
        matched_relations,
        contested_peaks,
        mean_selected_negative_legacy_rank: selected_rank_sum / selected_rank_count.max(1) as f64,
        max_selected_negative_legacy_rank: selected_rank_max,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_validation(
    groups: &[CandidateGroup],
    records: &[FoundationTrainingRecord],
    predictor: &FrozenForwardPredictor,
    collator: &FoundationCollator,
    relation_model: &PeptideSpectrumFragmentRelationEnergy,
    device: &Device,
    limit: Option<usize>,
    diagnostics_path: &Path,
) -> Result<ValidationMetrics> {
    let total = limit.unwrap_or(groups.len()).min(groups.len());
    let mut metrics = ValidationMetrics {
        records: total,
        ..ValidationMetrics::default()
    };
    let mut diagnostics = BufWriter::new(File::create(diagnostics_path)?);
    writeln!(
        diagnostics,
        "record_index\toracle_literal\toracle_il\tlegacy_top1_literal\tlegacy_top1_il\trelation_only_top1_literal\trelation_only_top1_il\tglobal_top1_literal\tglobal_top1_il\tbest_literal_global_rank\tbest_il_global_rank\tlegacy_top1_sequence\trelation_only_top1_sequence\tglobal_top1_sequence\trelation_residual_score\tglobal_score\tcandidates\tmatched_relations\tcontested_peaks"
    )?;

    for (group_number, group) in groups.iter().take(total).enumerate() {
        let record = records
            .get(group.record_index)
            .with_context(|| format!("VALIDATION record {} out of range", group.record_index))?;
        let spectrum = FoundationSpectrum::from_training_record(record)
            .with_context(|| format!("VALIDATION record {} lacks spectrum", group.record_index))?;
        if record.observed_spectrum_peaks.is_empty() {
            metrics.fallback_spectra += 1;
        } else {
            metrics.raw_spectra += 1;
        }
        metrics.oracle_literal += usize::from(group.rows.iter().any(|row| row.exact));
        metrics.oracle_il += usize::from(group.rows.iter().any(|row| row.il_exact));
        let legacy_index = group
            .rows
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.legacy_rank)
            .map(|(index, _)| index)
            .context("validation candidate group is empty")?;
        metrics.legacy_literal += usize::from(group.rows[legacy_index].exact);
        metrics.legacy_il += usize::from(group.rows[legacy_index].il_exact);

        let peptides = group
            .rows
            .iter()
            .map(exported_peptidoform)
            .collect::<Result<Vec<_>>>()?;
        let mass_errors = group
            .rows
            .iter()
            .map(|row| row.mass_error_da)
            .collect::<Vec<_>>();
        let mut predicted = Vec::<Vec<Vec<f32>>>::with_capacity(peptides.len());
        for (chunk_index, peptide_chunk) in peptides.chunks(FORWARD_CANDIDATE_BATCH).enumerate() {
            let mut candidate_records =
                Vec::<FoundationTrainingRecord>::with_capacity(peptide_chunk.len());
            for peptide in peptide_chunk {
                let mut candidate = record.clone();
                candidate.peptidoform = peptide.clone();
                candidate.retention_time = Default::default();
                candidate.ccs = None;
                candidate.fragments.clear();
                candidate_records.push(candidate);
            }
            let batch = collator.collate(
                &candidate_records,
                device,
                SEED ^ group.record_index as u64 ^ chunk_index as u64,
            )?;
            let output = predictor
                .model
                .forward()
                .forward_ms2_t(&batch.input, &batch.context, false)?
                .detach()
                .to_vec3::<f32>()?;
            predicted.extend(output);
        }
        let rows = foundation_fragment_relation_features(
            &peptides,
            &spectrum,
            &predicted,
            &mass_errors,
            predictor.metadata.forward_config.max_sequence_len - 1,
        )
        .map_err(anyhow::Error::msg)?;
        metrics.matched_relations += rows.matched_relations;
        metrics.contested_peaks += rows.contested_peaks;
        metrics.candidates_scored += peptides.len();
        let relation_batch = rows.to_batch(device)?;
        let relation_scores = relation_model
            .forward_grouped(&relation_batch, peptides.len())?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let global_scores = relation_scores
            .iter()
            .zip(&group.rows)
            .map(|(&residual, row)| {
                residual + foundation_fragment_relation_legacy_log_prior(row.legacy_rank)
            })
            .collect::<Vec<_>>();

        let mut relation_ranked = (0..relation_scores.len()).collect::<Vec<_>>();
        relation_ranked.sort_by(|&left, &right| {
            relation_scores[right]
                .total_cmp(&relation_scores[left])
                .then_with(|| {
                    group.rows[left]
                        .legacy_rank
                        .cmp(&group.rows[right].legacy_rank)
                })
                .then_with(|| left.cmp(&right))
        });
        let relation_index = relation_ranked[0];
        metrics.relation_only_literal += usize::from(group.rows[relation_index].exact);
        metrics.relation_only_il += usize::from(group.rows[relation_index].il_exact);

        let mut global_ranked = (0..global_scores.len()).collect::<Vec<_>>();
        global_ranked.sort_by(|&left, &right| {
            global_scores[right]
                .total_cmp(&global_scores[left])
                .then_with(|| {
                    group.rows[left]
                        .legacy_rank
                        .cmp(&group.rows[right].legacy_rank)
                })
                .then_with(|| left.cmp(&right))
        });
        let global_index = global_ranked[0];
        metrics.global_literal += usize::from(group.rows[global_index].exact);
        metrics.global_il += usize::from(group.rows[global_index].il_exact);
        metrics.global_literal_top5 += usize::from(
            global_ranked
                .iter()
                .take(5)
                .any(|&index| group.rows[index].exact),
        );
        metrics.global_il_top5 += usize::from(
            global_ranked
                .iter()
                .take(5)
                .any(|&index| group.rows[index].il_exact),
        );
        metrics.global_literal_top10 += usize::from(
            global_ranked
                .iter()
                .take(10)
                .any(|&index| group.rows[index].exact),
        );
        metrics.global_il_top10 += usize::from(
            global_ranked
                .iter()
                .take(10)
                .any(|&index| group.rows[index].il_exact),
        );

        let exact_rank = global_ranked
            .iter()
            .position(|&index| group.rows[index].exact)
            .map(|rank| rank + 1);
        let il_rank = global_ranked
            .iter()
            .position(|&index| group.rows[index].il_exact)
            .map(|rank| rank + 1);
        if let Some(rank) = exact_rank {
            let legacy_rank = group
                .rows
                .iter()
                .filter(|row| row.exact)
                .map(|row| row.legacy_rank)
                .min()
                .unwrap_or(usize::MAX);
            match rank.cmp(&legacy_rank) {
                std::cmp::Ordering::Less => metrics.true_rank_improved += 1,
                std::cmp::Ordering::Greater => metrics.true_rank_worsened += 1,
                std::cmp::Ordering::Equal => metrics.true_rank_tied += 1,
            }
        }

        writeln!(
            diagnostics,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{}\t{}\t{}",
            group.record_index,
            yes_no(group.rows.iter().any(|row| row.exact)),
            yes_no(group.rows.iter().any(|row| row.il_exact)),
            yes_no(group.rows[legacy_index].exact),
            yes_no(group.rows[legacy_index].il_exact),
            yes_no(group.rows[relation_index].exact),
            yes_no(group.rows[relation_index].il_exact),
            yes_no(group.rows[global_index].exact),
            yes_no(group.rows[global_index].il_exact),
            format_rank(exact_rank),
            format_rank(il_rank),
            group.rows[legacy_index].sequence,
            group.rows[relation_index].sequence,
            group.rows[global_index].sequence,
            relation_scores[global_index],
            global_scores[global_index],
            group.rows.len(),
            rows.matched_relations,
            rows.contested_peaks,
        )?;

        let done = group_number + 1;
        if done == 1 || done % 25 == 0 || done == total {
            println!(
                "relation_validation_progress\tgroups_done={}\tgroups_total={}\tglobal_literal_top1={}\tglobal_il_top1={}",
                done,
                total,
                metrics.global_literal,
                metrics.global_il,
            );
        }
    }
    diagnostics.flush()?;
    Ok(metrics)
}

fn print_validation_metrics(metrics: &ValidationMetrics) {
    println!("final_records\t{}", metrics.records);
    println!("candidate_rows_scored\t{}", metrics.candidates_scored);
    println!("proposal_oracle_literal\t{}", metrics.oracle_literal);
    println!("proposal_oracle_il\t{}", metrics.oracle_il);
    println!("legacy_literal_top1\t{}", metrics.legacy_literal);
    println!("legacy_il_top1\t{}", metrics.legacy_il);
    println!(
        "relation_only_literal_top1\t{}",
        metrics.relation_only_literal
    );
    println!("relation_only_il_top1\t{}", metrics.relation_only_il);
    println!("global_relation_literal_top1\t{}", metrics.global_literal);
    println!("global_relation_il_top1\t{}", metrics.global_il);
    println!(
        "global_relation_literal_top5\t{}",
        metrics.global_literal_top5
    );
    println!("global_relation_il_top5\t{}", metrics.global_il_top5);
    println!(
        "global_relation_literal_top10\t{}",
        metrics.global_literal_top10
    );
    println!("global_relation_il_top10\t{}", metrics.global_il_top10);
    println!(
        "true_candidate_global_relation_rank_improved_vs_legacy\t{}",
        metrics.true_rank_improved
    );
    println!(
        "true_candidate_global_relation_rank_worsened_vs_legacy\t{}",
        metrics.true_rank_worsened
    );
    println!(
        "true_candidate_global_relation_rank_tied_vs_legacy\t{}",
        metrics.true_rank_tied
    );
    println!("raw_observed_spectrum_records\t{}", metrics.raw_spectra);
    println!(
        "annotated_fragment_spectrum_fallback_records\t{}",
        metrics.fallback_spectra
    );
    println!("relation_matched_core_ions\t{}", metrics.matched_relations);
    println!(
        "relation_contested_peak_instances\t{}",
        metrics.contested_peaks
    );
}

fn enforce_frozen_contract(metrics: &ValidationMetrics) -> Result<()> {
    if metrics.records != REQUIRED_RECORDS {
        anyhow::bail!(
            "v0.24 frozen validation expected {REQUIRED_RECORDS} records, observed {}",
            metrics.records
        );
    }
    if metrics.oracle_literal != REQUIRED_ORACLE_LITERAL || metrics.oracle_il != REQUIRED_ORACLE_IL
    {
        anyhow::bail!(
            "v0.24 proposal oracle drift: expected {REQUIRED_ORACLE_LITERAL}/{REQUIRED_ORACLE_IL}, observed {}/{}",
            metrics.oracle_literal,
            metrics.oracle_il
        );
    }
    if metrics.legacy_literal != REQUIRED_LEGACY_LITERAL || metrics.legacy_il != REQUIRED_LEGACY_IL
    {
        anyhow::bail!(
            "v0.24 legacy baseline drift: expected {REQUIRED_LEGACY_LITERAL}/{REQUIRED_LEGACY_IL}, observed {}/{}",
            metrics.legacy_literal,
            metrics.legacy_il
        );
    }
    Ok(())
}

fn validate_candidate_contract(groups: &[CandidateGroup]) -> Result<ValidationContract> {
    let mut contract = ValidationContract {
        records: groups.len(),
        oracle_literal: 0,
        oracle_il: 0,
        legacy_literal: 0,
        legacy_il: 0,
    };
    for group in groups {
        contract.oracle_literal += usize::from(group.rows.iter().any(|row| row.exact));
        contract.oracle_il += usize::from(group.rows.iter().any(|row| row.il_exact));
        if let Some(top) = group.rows.iter().min_by_key(|row| row.legacy_rank) {
            contract.legacy_literal += usize::from(top.exact);
            contract.legacy_il += usize::from(top.il_exact);
        }
    }
    if contract.records != REQUIRED_RECORDS
        || contract.oracle_literal != REQUIRED_ORACLE_LITERAL
        || contract.oracle_il != REQUIRED_ORACLE_IL
        || contract.legacy_literal != REQUIRED_LEGACY_LITERAL
        || contract.legacy_il != REQUIRED_LEGACY_IL
    {
        anyhow::bail!(
            "v0.24 frozen validation contract mismatch: records={} oracle={}/{} legacy={}/{} expected records={} oracle={}/{} legacy={}/{}",
            contract.records,
            contract.oracle_literal,
            contract.oracle_il,
            contract.legacy_literal,
            contract.legacy_il,
            REQUIRED_RECORDS,
            REQUIRED_ORACLE_LITERAL,
            REQUIRED_ORACLE_IL,
            REQUIRED_LEGACY_LITERAL,
            REQUIRED_LEGACY_IL,
        );
    }
    Ok(contract)
}

fn build_proposal_hard_groups(
    candidate_groups: &[CandidateGroup],
    records: &[FoundationTrainingRecord],
) -> Result<(Vec<ProposalHardGroup>, ProposalMiningStats)> {
    let mut groups = Vec::<ProposalHardGroup>::new();
    let mut positive_exact_present = 0usize;
    let mut positive_il_present = 0usize;
    let mut retained_raw_spectra = 0usize;
    let mut retained_fallback_spectra = 0usize;
    let mut positive_rank_sum = 0.0f64;
    let mut negative_rank_sum = 0.0f64;
    let mut negative_rank_count = 0usize;
    let mut rank_max = 0usize;

    for candidate_group in candidate_groups {
        let anchor = records.get(candidate_group.record_index).with_context(|| {
            format!(
                "TRAIN candidate record {} out of range",
                candidate_group.record_index
            )
        })?;
        if FoundationSpectrum::from_training_record(anchor).is_none()
            || !peptide_supported(&anchor.peptidoform, ACCEPTED_MAX_SEQUENCE_LEN)
        {
            continue;
        }
        let Some(positive) = candidate_group
            .rows
            .iter()
            .filter(|row| row.exact && row.legacy_rank != usize::MAX)
            .min_by_key(|row| row.legacy_rank)
            .cloned()
        else {
            continue;
        };
        positive_exact_present += 1;
        positive_il_present += usize::from(candidate_group.rows.iter().any(|row| row.il_exact));
        let target_il = il_sequence_key(&anchor.peptidoform.sequence);
        let negative_indices = proposal_negative_indices(&candidate_group.rows, &target_il);
        if negative_indices.len() != PROPOSAL_NEGATIVES_PER_ANCHOR {
            continue;
        }
        let mut negatives = Vec::with_capacity(PROPOSAL_NEGATIVES_PER_ANCHOR);
        let mut supported = true;
        for index in negative_indices {
            let row = candidate_group.rows[index].clone();
            let peptide = exported_peptidoform(&row).with_context(|| {
                format!(
                    "reconstruct TRAIN proposal candidate '{}' for record {}",
                    row.sequence, candidate_group.record_index
                )
            })?;
            if !peptide_supported(&peptide, ACCEPTED_MAX_SEQUENCE_LEN) {
                supported = false;
                break;
            }
            negatives.push(row);
        }
        if !supported || negatives.len() != PROPOSAL_NEGATIVES_PER_ANCHOR {
            continue;
        }
        positive_rank_sum += positive.legacy_rank as f64;
        for negative in &negatives {
            negative_rank_sum += negative.legacy_rank as f64;
            negative_rank_count += 1;
            rank_max = rank_max.max(negative.legacy_rank);
        }
        if anchor.observed_spectrum_peaks.is_empty() {
            retained_fallback_spectra += 1;
        } else {
            retained_raw_spectra += 1;
        }
        groups.push(ProposalHardGroup {
            anchor_index: candidate_group.record_index,
            positive,
            negatives,
        });
    }

    let retained_groups = groups.len();
    Ok((
        groups,
        ProposalMiningStats {
            input_candidate_groups: candidate_groups.len(),
            spectrum_anchors: candidate_groups
                .iter()
                .filter(|group| {
                    records
                        .get(group.record_index)
                        .and_then(|record| FoundationSpectrum::from_training_record(record))
                        .is_some()
                })
                .count(),
            positive_exact_present_groups: positive_exact_present,
            positive_il_present_groups: positive_il_present,
            retained_raw_spectra,
            retained_fallback_spectra,
            mean_positive_legacy_rank: positive_rank_sum / retained_groups.max(1) as f64,
            mean_selected_legacy_rank: negative_rank_sum / negative_rank_count.max(1) as f64,
            max_selected_legacy_rank: rank_max,
        },
    ))
}

fn proposal_negative_indices(rows: &[CandidateRow], target_il: &str) -> Vec<usize> {
    let mut ranked = (0..rows.len()).collect::<Vec<_>>();
    ranked.sort_by_key(|&index| (rows[index].legacy_rank, index));
    let mut seen = HashSet::<(String, String)>::new();
    let mut selected = Vec::with_capacity(PROPOSAL_NEGATIVES_PER_ANCHOR);
    for index in ranked {
        let row = &rows[index];
        if row.legacy_rank == usize::MAX || il_sequence_key(&row.sequence) == target_il {
            continue;
        }
        let identity = (row.sequence.clone(), row.modifications.clone());
        if !seen.insert(identity) {
            continue;
        }
        selected.push(index);
        if selected.len() == PROPOSAL_NEGATIVES_PER_ANCHOR {
            break;
        }
    }
    selected
}

fn read_candidate_groups(
    path: &Path,
    require_proposal_export: bool,
) -> Result<Vec<CandidateGroup>> {
    let file =
        BufReader::new(File::open(path).with_context(|| format!("open candidate TSV {path:?}"))?);
    let mut lines = file.lines();
    let header = lines.next().context("candidate TSV is empty")??;
    let columns = header.split('\t').collect::<Vec<_>>();
    let index = columns
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, index))
        .collect::<HashMap<_, _>>();
    for required in [
        "record_index",
        "candidate_sequence",
        "candidate_modifications",
        "fragment_causal_mass_rank",
        "mass_error_da",
        "mass_valid",
        "peptidoform_exact",
        "il_sequence_exact",
    ] {
        if !index.contains_key(required) {
            anyhow::bail!("candidate TSV missing required column '{required}'");
        }
    }
    if require_proposal_export {
        for required in [
            "from_diffusion",
            "from_causal_beam",
            "from_reverse_causal_beam",
            "from_bidirectional_mitm",
        ] {
            if !index.contains_key(required) {
                anyhow::bail!(
                    "v0.24 TRAIN proposal TSV missing frozen-generator provenance column '{required}'"
                );
            }
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
            let column = *index
                .get(name)
                .context("internal candidate column missing")?;
            fields
                .get(column)
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
            legacy_rank: parse_usize(get("fragment_causal_mass_rank")?, usize::MAX),
            mass_error_da: parse_finite(get("mass_error_da")?, f64::INFINITY),
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
                "v0.24 {label} candidate group {} is not assigned to expected partition",
                group.record_index
            );
        }
    }
    Ok(())
}

fn exported_peptidoform(row: &CandidateRow) -> Result<PeptidoformInput> {
    if row.modifications.trim().is_empty() {
        return Ok(PeptidoformInput::unmodified(row.sequence.clone()));
    }
    let residues = row.sequence.chars().collect::<Vec<_>>();
    let mut encoded = String::new();
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
            .with_context(|| format!("unsupported modification identity '{identity}'"))?
            .parse()?;
        if site == "NTerm" {
            nterm.push(id);
        } else if let Some(inner) = site
            .strip_prefix("Residue(")
            .and_then(|value| value.strip_suffix(')'))
        {
            let residue_index: usize = inner.parse()?;
            if residue_index >= residues.len() {
                anyhow::bail!("exported modification residue index {residue_index} out of range");
            }
            residue_mods.entry(residue_index).or_default().push(id);
        } else {
            anyhow::bail!("unsupported exported modification site '{site}'");
        }
    }
    for id in nterm {
        encoded.push_str(&format!("[UniMod:{id}]-"));
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

fn precursor_mass_error(record: &FoundationTrainingRecord, peptide: &PeptidoformInput) -> f64 {
    match (record.context.precursor_mz, record.context.charge) {
        (Some(mz), Some(charge)) if charge > 0 => {
            foundation_precursor_mass_error_da(peptide, f64::from(mz), charge as i32).unwrap_or(0.0)
        }
        _ => 0.0,
    }
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

#[derive(Clone)]
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
        for _ in 0..batch {
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

fn initialize_relation_variables(varmap: &VarMap, seed: u64, device: &Device) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.24 relation VarMap lock poisoned"))?;
    let mut names = data.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut rng = DeterministicRng::new(seed);
    let mut zeroed_candidate_output = false;
    for name in names {
        let variable = data
            .get(&name)
            .with_context(|| format!("missing v0.24 relation variable {name}"))?;
        if !variable.dtype().is_float() {
            continue;
        }
        let dims = variable.as_tensor().dims().to_vec();
        let tensor = if name.ends_with("candidate_out.weight") {
            zeroed_candidate_output = true;
            Tensor::zeros(variable.shape(), DType::F32, device)?
        } else if name.ends_with(".bias") {
            Tensor::zeros(variable.shape(), DType::F32, device)?
        } else if dims.len() == 2 {
            let out_dim = dims[0];
            let in_dim = dims[1];
            let stdev = (2.0f64 / in_dim.max(1) as f64).sqrt();
            let values = (0..out_dim * in_dim)
                .map(|_| (stdev * rng.standard_normal()) as f32)
                .collect::<Vec<_>>();
            Tensor::from_vec(values, (out_dim, in_dim), device)?
        } else {
            anyhow::bail!("v0.24 relation variable '{name}' has unsupported shape {dims:?}");
        };
        variable.set(&tensor)?;
    }
    if !zeroed_candidate_output {
        anyhow::bail!("v0.24 relation initialization did not find candidate_out.weight");
    }
    Ok(())
}

fn relation_model_fingerprint(varmap: &VarMap) -> Result<String> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.24 relation VarMap lock poisoned"))?;
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

fn verify_checkpoint_fingerprints(
    metadata: &UnifiedMetadata,
    corpus: &redeem_properties::foundation::FoundationCorpus,
    benchmark: &FoundationBenchmarkManifest,
) -> Result<()> {
    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if metadata.corpus_fingerprint != corpus_fingerprint {
        anyhow::bail!(
            "v0.24 unified checkpoint corpus fingerprint {} != loaded {}",
            metadata.corpus_fingerprint,
            corpus_fingerprint
        );
    }
    if metadata.benchmark_manifest_fingerprint != benchmark_fingerprint {
        anyhow::bail!(
            "v0.24 unified checkpoint benchmark fingerprint {} != loaded {}",
            metadata.benchmark_manifest_fingerprint,
            benchmark_fingerprint
        );
    }
    Ok(())
}

fn reject_test_path(path: &Path) -> Result<()> {
    let name = path.to_string_lossy().to_ascii_lowercase();
    if name.contains("/test/")
        || name.contains("\\test\\")
        || name.contains("test_partition")
        || name.ends_with("/test.tsv")
        || name.ends_with("\\test.tsv")
    {
        anyhow::bail!("v0.24 forbids TEST-partition inputs; suspicious path {path:?}");
    }
    Ok(())
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
        .map(|residue| match residue.to_ascii_uppercase() {
            'I' | 'L' => 'J',
            other => other,
        })
        .collect()
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

fn parse_finite(value: &str, fallback: f64) -> f64 {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

fn parse_usize(value: &str, fallback: usize) -> usize {
    value.trim().parse::<usize>().unwrap_or(fallback)
}

fn format_rank(rank: Option<usize>) -> String {
    rank.map(|value| value.to_string())
        .unwrap_or_else(|| "NA".into())
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

fn stage(started: Instant, stage: &str) {
    println!(
        "v0240_stage\tstage={}\telapsed_seconds={:.3}",
        stage,
        started.elapsed().as_secs_f64()
    );
}

fn fnv1a64_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: mix64(seed) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    fn uniform01(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 / ((1u64 << 53) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform01().max(f64::MIN_POSITIVE);
        let u2 = self.uniform01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn deterministic_shuffle<T>(values: &mut [T], seed: u64) {
    let mut rng = DeterministicRng::new(seed);
    for index in (1..values.len()).rev() {
        let selected = (rng.next_u64() as usize) % (index + 1);
        values.swap(index, selected);
    }
}
