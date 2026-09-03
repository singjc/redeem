//! Definitive validation-only audit of causal target-prefix survival.
//!
//! v0.13.11 keeps the frozen v0.13.10 checkpoint and generation policy intact.
//! It compares the production causal beam against two deliberately generous
//! counterfactual searches on exactly the same validation records:
//! 1. 8x wider search with mass-bin/last-token state compression effectively disabled; and
//! 2. the same wide/no-dedup search using autoregressive probability alone.
//!
//! This creates a one-run decision gate between search compression, the prefix
//! fragment heuristic, precursor/token feasibility, finalization, and the causal
//! model score itself. Ground-truth prefixes are observed only for diagnostics;
//! they are never injected into any candidate beam. TEST is never consumed.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, foundation_fragment_causal_rerank_score,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    FoundationBenchmarkManifest, FoundationCausalCollator, FoundationCausalContext,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationPartition,
    FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideSpectrumCausalModel, PrecursorContextBatch, FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct UnifiedCheckpointMetadata {
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone)]
struct CausalBeamState {
    prefix: Vec<u32>,
    neutral_mass: f64,
    ar_total_log_probability: f64,
    fragment_score: f64,
    matched_cleavages: usize,
    residue_count: usize,
    priority: f64,
}

#[derive(Debug, Clone)]
struct CausalBeamCandidate {
    tokens: Vec<u32>,
    ar_total_log_probability: f64,
    fragment_score: f64,
    matched_cleavages: usize,
    fragment_causal_score: f64,
    abs_mass_error_da: f64,
}

struct CausalReranker {
    _varmap: VarMap,
    model: PeptideSpectrumCausalModel,
    collator: FoundationCausalCollator,
}

#[derive(Debug, Clone, Copy, Default)]
struct CleavageEvidence {
    score: f64,
    matched: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuditPriorityMode {
    LockedFragmentCausal,
    ArOnly,
}

impl AuditPriorityMode {
    fn label(self) -> &'static str {
        match self {
            Self::LockedFragmentCausal => "fragment_plus_weighted_ar",
            Self::ArOnly => "ar_total_only",
        }
    }

    fn score(self, fragment_score: f64, ar_total_log_probability: f64, causal_weight: f64) -> f64 {
        match self {
            Self::LockedFragmentCausal => foundation_fragment_causal_rerank_score(
                fragment_score,
                ar_total_log_probability,
                causal_weight,
            ),
            Self::ArOnly => ar_total_log_probability,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AuditSearchVariant {
    name: &'static str,
    beam_width: usize,
    bin_capacity: usize,
    final_candidates: usize,
    priority_mode: AuditPriorityMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AuditLossKind {
    TokenLogitNonfinite,
    TokenRuleRejected,
    TokenMassMissing,
    MassOvershoot,
    MassUnreachable,
    DedupCollision,
    BeamPruned,
    FinalCandidateTruncation,
    TerminationMissing,
}

impl AuditLossKind {
    fn label(&self) -> &'static str {
        match self {
            Self::TokenLogitNonfinite => "token_logit_nonfinite",
            Self::TokenRuleRejected => "token_rule_rejected",
            Self::TokenMassMissing => "token_mass_missing",
            Self::MassOvershoot => "mass_overshoot",
            Self::MassUnreachable => "mass_unreachable",
            Self::DedupCollision => "mass_bin_last_token_dedup_collision",
            Self::BeamPruned => "global_beam_pruned",
            Self::FinalCandidateTruncation => "final_candidate_truncation",
            Self::TerminationMissing => "termination_missing",
        }
    }
}

#[derive(Debug, Clone)]
struct AuditLossEvent {
    kind: AuditLossKind,
    position: usize,
    target_token: u32,
    target_priority: f64,
    target_rank_pretruncate: Option<usize>,
    candidates_pretruncate: usize,
    competitor_priority: Option<f64>,
    competitor_prefix: Option<Vec<u32>>,
}

#[derive(Debug, Clone)]
struct TargetShadowStep {
    position: usize,
    next_token: u32,
    next_token_log_probability: f64,
    raw_logit_rank: usize,
    legal_mass_rank: Option<usize>,
    cumulative_ar_total: f64,
    cumulative_fragment_score: f64,
    locked_priority: f64,
    ar_only_priority: f64,
    token_rule_allowed: bool,
    token_mass_known: bool,
    mass_feasible: bool,
}

#[derive(Debug, Clone)]
struct AuditBeamStep {
    position: usize,
    parent_live: bool,
    child_eligible: bool,
    child_survived_bin: bool,
    child_rank_pretruncate: Option<usize>,
    child_live_after_truncate: bool,
    shadow_rank_pretruncate: Option<usize>,
    candidates_pretruncate: usize,
    beam_size_after_truncate: usize,
    target_priority: f64,
    beam_cutoff_priority: Option<f64>,
    beam_cutoff_prefix: Option<Vec<u32>>,
    target_minus_cutoff: Option<f64>,
    same_bin_competitor_priority: Option<f64>,
    same_bin_competitor_prefix: Option<Vec<u32>>,
}

#[derive(Debug, Clone)]
struct AuditCompletedCandidate {
    candidate: CausalBeamCandidate,
    priority: f64,
}

#[derive(Debug, Clone)]
struct AuditBeamRun {
    variant: AuditSearchVariant,
    final_candidates: Vec<AuditCompletedCandidate>,
    completed_before_final_count: usize,
    target_completed_before_final: bool,
    target_completed_rank: Option<usize>,
    il_completed_before_final: bool,
    il_completed_rank: Option<usize>,
    exact_generated: bool,
    il_sequence_generated: bool,
    first_loss: Option<AuditLossEvent>,
    steps: Vec<AuditBeamStep>,
}

#[derive(Debug, Default)]
struct AuditAggregate {
    records: usize,
    baseline_literal_success: usize,
    baseline_il_success: usize,
    locked_wide_literal_success: usize,
    locked_wide_il_success: usize,
    ar_wide_literal_success: usize,
    ar_wide_il_success: usize,
    classes: HashMap<String, usize>,
    baseline_losses: HashMap<String, usize>,
    locked_wide_losses: HashMap<String, usize>,
    ar_wide_losses: HashMap<String, usize>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 12 {
        anyhow::bail!(
            "usage: foundation_audit_causal_prefix_survival FOUNDATION_TRAINING.yaml UNIFIED_CHECKPOINT OUTPUT_DIR [validation_records=128] [seed=20260912] [mass_tolerance_da=0.05] [fragment_tolerance_ppm=20] [causal_weight=0.1] [baseline_beam_width=32] [baseline_final_candidates=16] [counterfactual_beam_width=256]"
        );
    }

    let training_yaml = &args[1];
    let checkpoint_dir = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);
    let validation_records = parse_or(&args, 4, 128usize)?;
    let seed = parse_or(&args, 5, 20_260_912u64)?;
    let mass_tolerance_da = parse_or(&args, 6, 0.05f64)?;
    let fragment_tolerance_ppm = parse_or(&args, 7, 20.0f64)?;
    let causal_weight = parse_or(&args, 8, FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123)?;
    let baseline_beam_width = parse_or(&args, 9, 32usize)?;
    let baseline_final_candidates = parse_or(&args, 10, 16usize)?;
    let counterfactual_beam_width = parse_or(&args, 11, 256usize)?;
    // Counterfactual searches retain every completed candidate. This removes
    // final-candidate truncation entirely from the wide decision gate.
    let counterfactual_final_candidates = usize::MAX;

    if validation_records == 0
        || baseline_beam_width == 0
        || baseline_final_candidates == 0
        || counterfactual_beam_width == 0
    {
        anyhow::bail!("record, beam, and final-candidate counts must be positive");
    }
    let minimum_counterfactual_width = baseline_beam_width.saturating_mul(8);
    if counterfactual_beam_width < minimum_counterfactual_width {
        anyhow::bail!(
            "definitive decision gate requires counterfactual beam width >= 8x baseline ({minimum_counterfactual_width})"
        );
    }
    if !(mass_tolerance_da > 0.0 && mass_tolerance_da.is_finite()) {
        anyhow::bail!("mass_tolerance_da must be positive and finite");
    }
    if !(fragment_tolerance_ppm > 0.0 && fragment_tolerance_ppm.is_finite()) {
        anyhow::bail!("fragment_tolerance_ppm must be positive and finite");
    }
    if !causal_weight.is_finite() {
        anyhow::bail!("causal_weight must be finite");
    }

    let baseline_variant = AuditSearchVariant {
        name: "baseline_locked",
        beam_width: baseline_beam_width,
        bin_capacity: 1,
        final_candidates: baseline_final_candidates,
        priority_mode: AuditPriorityMode::LockedFragmentCausal,
    };
    let locked_wide_variant = AuditSearchVariant {
        name: "wide_no_dedup_locked",
        beam_width: counterfactual_beam_width,
        // A bin cannot retain more states than the global beam. Making the
        // per-bin capacity equal to the beam width therefore removes the
        // mass-bin/last-token compression as an active pruning mechanism.
        bin_capacity: counterfactual_beam_width,
        final_candidates: counterfactual_final_candidates,
        priority_mode: AuditPriorityMode::LockedFragmentCausal,
    };
    let ar_wide_variant = AuditSearchVariant {
        name: "wide_no_dedup_ar_only",
        beam_width: counterfactual_beam_width,
        bin_capacity: counterfactual_beam_width,
        final_candidates: counterfactual_final_candidates,
        priority_mode: AuditPriorityMode::ArOnly,
    };

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let metadata_path = checkpoint_dir.join("metadata.yaml");
    let checkpoint_metadata: UnifiedCheckpointMetadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read {metadata_path:?}"))?,
    )?;
    let config = checkpoint_metadata.inverse_config;
    config.validate().map_err(anyhow::Error::msg)?;

    let vocabulary = FoundationDiffusionVocabulary;
    let usable_validation = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &config,
        vocabulary,
    );
    let selected = deterministic_subset(&usable_validation, validation_records, seed);
    if selected.is_empty() {
        anyhow::bail!("no usable validation causal pairs were selected");
    }

    let causal_varmap = VarMap::new();
    let causal_vb = VarBuilder::from_varmap(&causal_varmap, DType::F32, &device);
    let causal_model = PeptideSpectrumCausalModel::new(config.clone(), causal_vb)?;
    load_matching_variables(
        &causal_varmap,
        &checkpoint_dir.join("model.safetensors"),
        &device,
    )
    .with_context(|| format!("failed to load unified causal variables {checkpoint_dir:?}"))?;
    let causal = CausalReranker {
        _varmap: causal_varmap,
        model: causal_model,
        collator: FoundationCausalCollator::new(config.clone())?,
    };
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;

    fs::create_dir_all(&output_dir)?;
    let records_path = output_dir.join("causal_prefix_survival_records.tsv");
    let steps_path = output_dir.join("causal_prefix_survival_steps.tsv");
    let summary_path = output_dir.join("causal_prefix_survival_summary.tsv");
    let mut records_output = BufWriter::new(fs::File::create(&records_path)?);
    let mut steps_output = BufWriter::new(fs::File::create(&steps_path)?);

    writeln!(
        records_output,
        "record_index\tsource_id\ttarget_sequence\ttarget_active_tokens\ttarget_residues\tmodified\tprecursor_neutral_mass\tencoded_target_neutral_mass\ttarget_precursor_mass_error_da\tmass_feasible\ttoken_path_valid\tbaseline_exact\tbaseline_il\tbaseline_first_loss\tbaseline_first_loss_position\tbaseline_first_loss_rank\tbaseline_first_loss_target_priority\tbaseline_first_loss_competitor_priority\tbaseline_first_loss_margin\tbaseline_first_loss_competitor_prefix\tlocked_wide_exact\tlocked_wide_il\tlocked_wide_first_loss\tlocked_wide_first_loss_position\tlocked_wide_first_loss_rank\tlocked_wide_first_loss_target_priority\tlocked_wide_first_loss_competitor_priority\tlocked_wide_first_loss_margin\tlocked_wide_first_loss_competitor_prefix\tar_wide_exact\tar_wide_il\tar_wide_first_loss\tar_wide_first_loss_position\tar_wide_first_loss_rank\tar_wide_first_loss_target_priority\tar_wide_first_loss_competitor_priority\tar_wide_first_loss_margin\tar_wide_first_loss_competitor_prefix\tdecision_class\tdecision_action"
    )?;
    writeln!(
        steps_output,
        "record_index\tsource_id\tvariant\tposition\ttarget_token\ttarget_token_log_probability\traw_logit_rank\tlegal_mass_rank\tcumulative_ar_total\tcumulative_fragment_score\tlocked_priority\tar_only_priority\ttoken_rule_allowed\ttoken_mass_known\tmass_feasible\tparent_live\tchild_eligible\tchild_survived_bin\tchild_rank_pretruncate\tchild_live_after_truncate\tshadow_rank_pretruncate\tcandidates_pretruncate\tbeam_size_after_truncate\ttarget_variant_priority\tbeam_cutoff_priority\tbeam_cutoff_prefix\ttarget_minus_cutoff\tsame_bin_competitor_priority\tsame_bin_competitor_prefix"
    )?;

    println!("partition\tValidation");
    println!("test_partition_consumed\tNO");
    println!("diagnostic\tcausal_prefix_survival_decision_gate_v01311");
    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!(
        "benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        benchmark.manifest_fingerprint()
    );
    println!("checkpoint\t{}", checkpoint_dir.display());
    println!("validation_records\t{}", selected.len());
    println!("seed\t{seed}");
    println!("mass_tolerance_da\t{mass_tolerance_da}");
    println!("fragment_tolerance_ppm\t{fragment_tolerance_ppm}");
    println!("causal_weight\t{causal_weight}");
    print_audit_variant(baseline_variant);
    print_audit_variant(locked_wide_variant);
    print_audit_variant(ar_wide_variant);
    println!(
        "decision_gate_policy\tbaseline_vs_8x_wide_no_dedup_locked_vs_8x_wide_no_dedup_ar_only"
    );
    println!("decision_gate_stop_rule\tif_target_remains_absent_under_wide_no_dedup_ar_only_treat_causal_representation_score_as_bottleneck_and_do_not_continue_beam_width_diagnostics");
    println!("il_policy\tI_and_L_are_treated_as_mass_spectrometrically_indistinguishable_for_identifiable_success");

    let mut aggregate = AuditAggregate::default();
    for &record_index in &selected {
        let record = &corpus.records[record_index];
        let source_id = &corpus.provenance[record_index].source_id;
        let target_tokens = vocabulary
            .encode(&record.peptidoform, config.max_tokens)
            .map_err(anyhow::Error::msg)?;
        let target_active_length = active_token_length(&target_tokens, config.max_tokens)?;
        if target_active_length < 2
            || target_tokens[target_active_length - 1] != FOUNDATION_DIFFUSION_EOS
        {
            anyhow::bail!(
                "record {record_index} target row lacks a valid terminal EOS at active length {target_active_length}"
            );
        }
        let target_nonterminal = &target_tokens[..target_active_length - 1];
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("selected validation record lacks observed spectrum"))?;
        let observed_peaks = normalized_observed_peaks(&spectrum);
        let fragment_charge = record
            .context
            .charge
            .unwrap_or(1)
            .unsigned_abs()
            .clamp(1, 2) as usize;
        let precursor_mass = precursor_neutral_mass(record)?;
        let encoded_mass = encoded_target_neutral_mass(target_nonterminal);
        let target_precursor_mass_error = match (precursor_mass, encoded_mass) {
            (Some(left), Some(right)) => Some(right - left),
            _ => None,
        };
        let mass_feasible = target_precursor_mass_error
            .map(|error| error.abs() <= mass_tolerance_da)
            .unwrap_or(false);

        let spectrum_batch = spectrum_collator.collate(std::slice::from_ref(&spectrum), &device)?;
        let precursor = precursor_context(&[record], &device)?;
        let causal_context = causal
            .model
            .prepare_context(&spectrum_batch, &precursor, false)?;
        let shadow = score_target_shadow_path(
            &causal,
            &causal_context,
            &config,
            target_nonterminal,
            precursor_mass,
            mass_tolerance_da,
            &observed_peaks,
            fragment_charge,
            fragment_tolerance_ppm,
            causal_weight,
            &device,
        )?;
        let token_path_valid = shadow
            .iter()
            .take(target_nonterminal.len())
            .all(|step| step.token_rule_allowed && step.token_mass_known && step.mass_feasible);

        let baseline = run_causal_beam_decision_audit(
            &causal,
            &causal_context,
            &config,
            record,
            &target_tokens,
            &record.peptidoform.sequence,
            precursor_mass,
            mass_tolerance_da,
            baseline_variant,
            &shadow,
            &observed_peaks,
            fragment_charge,
            fragment_tolerance_ppm,
            causal_weight,
            vocabulary,
            &device,
        )?;
        let locked_wide = run_causal_beam_decision_audit(
            &causal,
            &causal_context,
            &config,
            record,
            &target_tokens,
            &record.peptidoform.sequence,
            precursor_mass,
            mass_tolerance_da,
            locked_wide_variant,
            &shadow,
            &observed_peaks,
            fragment_charge,
            fragment_tolerance_ppm,
            causal_weight,
            vocabulary,
            &device,
        )?;
        let ar_wide = run_causal_beam_decision_audit(
            &causal,
            &causal_context,
            &config,
            record,
            &target_tokens,
            &record.peptidoform.sequence,
            precursor_mass,
            mass_tolerance_da,
            ar_wide_variant,
            &shadow,
            &observed_peaks,
            fragment_charge,
            fragment_tolerance_ppm,
            causal_weight,
            vocabulary,
            &device,
        )?;

        let (decision_class, decision_action) = classify_prefix_failure(
            &baseline,
            &locked_wide,
            &ar_wide,
            precursor_mass,
            mass_feasible,
            token_path_valid,
        );
        aggregate.records += 1;
        aggregate.baseline_literal_success += usize::from(baseline.exact_generated);
        aggregate.baseline_il_success += usize::from(baseline.il_sequence_generated);
        aggregate.locked_wide_literal_success += usize::from(locked_wide.exact_generated);
        aggregate.locked_wide_il_success += usize::from(locked_wide.il_sequence_generated);
        aggregate.ar_wide_literal_success += usize::from(ar_wide.exact_generated);
        aggregate.ar_wide_il_success += usize::from(ar_wide.il_sequence_generated);
        *aggregate
            .classes
            .entry(decision_class.to_owned())
            .or_default() += 1;
        increment_loss_count(&mut aggregate.baseline_losses, baseline.first_loss.as_ref());
        increment_loss_count(
            &mut aggregate.locked_wide_losses,
            locked_wide.first_loss.as_ref(),
        );
        increment_loss_count(&mut aggregate.ar_wide_losses, ar_wide.first_loss.as_ref());

        write_beam_steps(
            &mut steps_output,
            record_index,
            source_id,
            &baseline,
            &shadow,
        )?;
        write_beam_steps(
            &mut steps_output,
            record_index,
            source_id,
            &locked_wide,
            &shadow,
        )?;
        write_beam_steps(
            &mut steps_output,
            record_index,
            source_id,
            &ar_wide,
            &shadow,
        )?;

        let record_fields = vec![
            record_index.to_string(),
            source_id.to_string(),
            record.peptidoform.sequence.clone(),
            target_active_length.to_string(),
            record.peptidoform.sequence.len().to_string(),
            (!record.peptidoform.modifications.is_empty()).to_string(),
            format_optional_f64(precursor_mass),
            format_optional_f64(encoded_mass),
            format_optional_f64(target_precursor_mass_error),
            mass_feasible.to_string(),
            token_path_valid.to_string(),
            baseline.exact_generated.to_string(),
            baseline.il_sequence_generated.to_string(),
            loss_label(&baseline).to_owned(),
            loss_position(&baseline),
            loss_rank(&baseline),
            loss_target_priority(&baseline),
            loss_competitor_priority(&baseline),
            loss_margin(&baseline),
            loss_competitor_prefix(&baseline),
            locked_wide.exact_generated.to_string(),
            locked_wide.il_sequence_generated.to_string(),
            loss_label(&locked_wide).to_owned(),
            loss_position(&locked_wide),
            loss_rank(&locked_wide),
            loss_target_priority(&locked_wide),
            loss_competitor_priority(&locked_wide),
            loss_margin(&locked_wide),
            loss_competitor_prefix(&locked_wide),
            ar_wide.exact_generated.to_string(),
            ar_wide.il_sequence_generated.to_string(),
            loss_label(&ar_wide).to_owned(),
            loss_position(&ar_wide),
            loss_rank(&ar_wide),
            loss_target_priority(&ar_wide),
            loss_competitor_priority(&ar_wide),
            loss_margin(&ar_wide),
            loss_competitor_prefix(&ar_wide),
            decision_class.to_owned(),
            decision_action.to_owned(),
        ];
        writeln!(records_output, "{}", record_fields.join("\t"))?;

        println!(
            "prefix_survival_record\trecord_index={}\ttarget={}\tmass_error_da={}\tbaseline_exact={}\tbaseline_il={}\tbaseline_loss={}\tlocked_wide_exact={}\tlocked_wide_il={}\tar_wide_exact={}\tar_wide_il={}\tdecision={}",
            record_index,
            record.peptidoform.sequence,
            format_optional_f64(target_precursor_mass_error),
            baseline.exact_generated,
            baseline.il_sequence_generated,
            loss_label(&baseline),
            locked_wide.exact_generated,
            locked_wide.il_sequence_generated,
            ar_wide.exact_generated,
            ar_wide.il_sequence_generated,
            decision_class,
        );
    }

    records_output.flush()?;
    steps_output.flush()?;
    write_audit_summary(&summary_path, &aggregate)?;
    print_audit_summary(&aggregate);
    let frozen_reference_parameters = selected.len() == 128
        && seed == 20_260_912
        && (mass_tolerance_da - 0.05).abs() <= f64::EPSILON
        && (fragment_tolerance_ppm - 20.0).abs() <= f64::EPSILON
        && (causal_weight - FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123).abs() <= f64::EPSILON
        && baseline_beam_width == 32
        && baseline_final_candidates == 16;
    if frozen_reference_parameters {
        let parity =
            aggregate.baseline_literal_success == 21 && aggregate.baseline_il_success == 33;
        println!(
            "frozen_v01310_baseline_parity\texpected_literal=21\texpected_il=33\tobserved_literal={}\tobserved_il={}\tparity={}",
            aggregate.baseline_literal_success,
            aggregate.baseline_il_success,
            if parity { "YES" } else { "NO" },
        );
    }
    println!("records_tsv\t{}", records_path.display());
    println!("steps_tsv\t{}", steps_path.display());
    println!("summary_tsv\t{}", summary_path.display());
    println!("test_partition_consumed\tNO");
    Ok(())
}

fn print_audit_variant(variant: AuditSearchVariant) {
    let final_candidates = if variant.final_candidates == usize::MAX {
        "ALL".to_owned()
    } else {
        variant.final_candidates.to_string()
    };
    println!(
        "search_variant\tname={}\tbeam_width={}\tbin_capacity={}\tfinal_candidates={}\tpriority={}",
        variant.name,
        variant.beam_width,
        variant.bin_capacity,
        final_candidates,
        variant.priority_mode.label(),
    );
}

fn increment_loss_count(counts: &mut HashMap<String, usize>, loss: Option<&AuditLossEvent>) {
    let label = loss
        .map(|event| event.kind.label())
        .unwrap_or("none")
        .to_owned();
    *counts.entry(label).or_default() += 1;
}

fn loss_label(run: &AuditBeamRun) -> &'static str {
    run.first_loss
        .as_ref()
        .map(|loss| loss.kind.label())
        .unwrap_or("none")
}

fn loss_position(run: &AuditBeamRun) -> String {
    run.first_loss
        .as_ref()
        .map(|loss| loss.position.to_string())
        .unwrap_or_default()
}

fn loss_rank(run: &AuditBeamRun) -> String {
    run.first_loss
        .as_ref()
        .and_then(|loss| loss.target_rank_pretruncate)
        .map(|rank| rank.to_string())
        .unwrap_or_default()
}

fn loss_target_priority(run: &AuditBeamRun) -> String {
    run.first_loss
        .as_ref()
        .map(|loss| format!("{:.8}", loss.target_priority))
        .unwrap_or_default()
}

fn loss_competitor_priority(run: &AuditBeamRun) -> String {
    run.first_loss
        .as_ref()
        .and_then(|loss| loss.competitor_priority)
        .map(|value| format!("{value:.8}"))
        .unwrap_or_default()
}

fn loss_margin(run: &AuditBeamRun) -> String {
    run.first_loss
        .as_ref()
        .and_then(|loss| {
            loss.competitor_priority
                .map(|competitor| loss.target_priority - competitor)
        })
        .map(|value| format!("{value:.8}"))
        .unwrap_or_default()
}

fn loss_competitor_prefix(run: &AuditBeamRun) -> String {
    run.first_loss
        .as_ref()
        .and_then(|loss| loss.competitor_prefix.as_ref())
        .map(|tokens| format_token_prefix(tokens))
        .unwrap_or_default()
}

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.8}"))
        .unwrap_or_default()
}

fn encoded_target_neutral_mass(tokens: &[u32]) -> Option<f64> {
    let mut mass = FOUNDATION_PEPTIDE_WATER_MASS_DA;
    for &token in tokens {
        let token_mass = foundation_diffusion_token_mass_da(token)?;
        if !(token_mass.is_finite() && token_mass >= 0.0) {
            return None;
        }
        mass += token_mass;
    }
    mass.is_finite().then_some(mass)
}

#[allow(clippy::too_many_arguments)]
fn score_target_shadow_path(
    causal: &CausalReranker,
    causal_context: &FoundationCausalContext,
    config: &FoundationDiffusionConfig,
    target_nonterminal: &[u32],
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    causal_weight: f64,
    device: &Device,
) -> Result<Vec<TargetShadowStep>> {
    let max_token_mass = maximum_diffusion_token_mass()?;
    let mut cumulative_ar = 0.0f64;
    let mut fragment_score = 0.0f64;
    let mut neutral_mass = FOUNDATION_PEPTIDE_WATER_MASS_DA;
    let mut residue_count = 0usize;
    let mut steps = Vec::with_capacity(target_nonterminal.len() + 1);

    // Compact causal-prefix collation requires equal prefix lengths within a
    // batch. The oracle path contains one prefix at each successive length, so
    // score those prefixes one at a time while reusing the already cached
    // spectrum/precursor context. This is diagnostic-only and never injects a
    // target prefix into a live beam.
    for position in 0..=target_nonterminal.len() {
        let prefix = vec![target_nonterminal[..position].to_vec()];
        let input = causal
            .collator
            .collate_compact_prefix_rows(&prefix, device)?;
        let next_rows = causal
            .model
            .forward_next_t_with_context(&input, causal_context, false)?
            .to_vec2::<f32>()?;
        let row = next_rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("target shadow produced no next-token logits"))?;
        let next_token = if position < target_nonterminal.len() {
            target_nonterminal[position]
        } else {
            FOUNDATION_DIFFUSION_EOS
        };
        let token_index = next_token as usize;
        let next_token_log_probability = selected_log_softmax(row, token_index)?;
        cumulative_ar += next_token_log_probability;
        let raw_logit_rank = 1 + row
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                **value > row[token_index] && *index != token_index && (**value).is_finite()
            })
            .count();

        let mut token_rule_allowed = true;
        let mut token_mass_known = true;
        let mut mass_feasible = true;
        let legal_mass_rank = if next_token == FOUNDATION_DIFFUSION_EOS {
            None
        } else {
            token_rule_allowed = mass_beam_token_allowed(
                &target_nonterminal[..position],
                next_token,
                position,
                config.max_tokens - 1,
            );
            let token_mass = foundation_diffusion_token_mass_da(next_token);
            token_mass_known = token_mass.is_some();
            if let (Some(target), Some(token_mass)) = (target_neutral_mass, token_mass) {
                let proposed_mass = neutral_mass + token_mass;
                let remaining_slots = config.max_tokens - 1 - (position + 1);
                let maximum_reachable_mass =
                    proposed_mass + remaining_slots as f64 * max_token_mass;
                mass_feasible = proposed_mass <= target + mass_tolerance_da
                    && maximum_reachable_mass + mass_tolerance_da >= target;
            } else if target_neutral_mass.is_some() {
                mass_feasible = false;
            }

            if token_rule_allowed && token_mass_known && mass_feasible {
                let mut better = 0usize;
                for candidate_index in
                    FOUNDATION_DIFFUSION_EOS as usize + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE
                {
                    if candidate_index == token_index || !row[candidate_index].is_finite() {
                        continue;
                    }
                    let candidate_token = candidate_index as u32;
                    if !mass_beam_token_allowed(
                        &target_nonterminal[..position],
                        candidate_token,
                        position,
                        config.max_tokens - 1,
                    ) {
                        continue;
                    }
                    let Some(candidate_mass) = foundation_diffusion_token_mass_da(candidate_token)
                    else {
                        continue;
                    };
                    if let Some(target) = target_neutral_mass {
                        let proposed_mass = neutral_mass + candidate_mass;
                        if proposed_mass > target + mass_tolerance_da {
                            continue;
                        }
                        let remaining_slots = config.max_tokens - 1 - (position + 1);
                        if proposed_mass
                            + remaining_slots as f64 * max_token_mass
                            + mass_tolerance_da
                            < target
                        {
                            continue;
                        }
                    }
                    if row[candidate_index] > row[token_index] {
                        better += 1;
                    }
                }
                Some(better + 1)
            } else {
                None
            }
        };

        if next_token != FOUNDATION_DIFFUSION_EOS {
            if let Some(token_mass) = foundation_diffusion_token_mass_da(next_token) {
                let is_residue = foundation_diffusion_token_residue(next_token).is_some();
                if is_residue && residue_count > 0 {
                    if let Some(target) = target_neutral_mass {
                        let prefix_mass_without_water =
                            neutral_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA;
                        let evidence = cleavage_fragment_evidence(
                            prefix_mass_without_water,
                            target,
                            observed_peaks,
                            max_fragment_charge,
                            fragment_tolerance_ppm,
                        );
                        fragment_score += evidence.score;
                    }
                }
                residue_count += usize::from(is_residue);
                neutral_mass += token_mass;
            }
        }

        steps.push(TargetShadowStep {
            position,
            next_token,
            next_token_log_probability,
            raw_logit_rank,
            legal_mass_rank,
            cumulative_ar_total: cumulative_ar,
            cumulative_fragment_score: fragment_score,
            locked_priority: foundation_fragment_causal_rerank_score(
                fragment_score,
                cumulative_ar,
                causal_weight,
            ),
            ar_only_priority: cumulative_ar,
            token_rule_allowed,
            token_mass_known,
            mass_feasible,
        });
    }
    Ok(steps)
}

fn maximum_diffusion_token_mass() -> Result<f64> {
    let max_token_mass = (FOUNDATION_DIFFUSION_EOS + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
        .filter_map(foundation_diffusion_token_mass_da)
        .filter(|mass| mass.is_finite() && *mass > 0.0)
        .fold(0.0f64, f64::max);
    if !(max_token_mass > 0.0 && max_token_mass.is_finite()) {
        anyhow::bail!("could not determine a positive maximum token mass");
    }
    Ok(max_token_mass)
}

#[allow(clippy::too_many_arguments)]
fn run_causal_beam_decision_audit(
    causal: &CausalReranker,
    causal_context: &FoundationCausalContext,
    config: &FoundationDiffusionConfig,
    _record: &FoundationTrainingRecord,
    target_tokens: &[u32],
    target_sequence: &str,
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    variant: AuditSearchVariant,
    shadow: &[TargetShadowStep],
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    causal_weight: f64,
    vocabulary: FoundationDiffusionVocabulary,
    device: &Device,
) -> Result<AuditBeamRun> {
    let Some(target) = target_neutral_mass.filter(|value| value.is_finite()) else {
        return Ok(AuditBeamRun {
            variant,
            final_candidates: Vec::new(),
            completed_before_final_count: 0,
            target_completed_before_final: false,
            target_completed_rank: None,
            il_completed_before_final: false,
            il_completed_rank: None,
            exact_generated: false,
            il_sequence_generated: false,
            first_loss: None,
            steps: Vec::new(),
        });
    };
    let target_active_length = active_token_length(target_tokens, config.max_tokens)?;
    let target_nonterminal = &target_tokens[..target_active_length - 1];
    let target_il = normalize_il(target_sequence);
    let max_token_mass = maximum_diffusion_token_mass()?;
    let mass_bin_width = mass_tolerance_da.max(0.05);
    let mut beam = vec![CausalBeamState {
        prefix: Vec::new(),
        neutral_mass: FOUNDATION_PEPTIDE_WATER_MASS_DA,
        ar_total_log_probability: 0.0,
        fragment_score: 0.0,
        matched_cleavages: 0,
        residue_count: 0,
        priority: 0.0,
    }];
    let mut completed = HashMap::<Vec<u32>, AuditCompletedCandidate>::new();
    let mut first_loss: Option<AuditLossEvent> = None;
    let mut steps = Vec::new();

    for position in 0..config.max_tokens {
        if beam.is_empty() {
            break;
        }
        debug_assert!(beam.iter().all(|state| state.prefix.len() == position));
        let prefixes: Vec<Vec<u32>> = beam.iter().map(|state| state.prefix.clone()).collect();
        let input = causal
            .collator
            .collate_compact_prefix_rows(&prefixes, device)?;
        let logits = causal
            .model
            .forward_next_t_with_context(&input, causal_context, false)?
            .to_vec2::<f32>()?;

        let target_parent_prefix = if position <= target_nonterminal.len() {
            Some(&target_nonterminal[..position])
        } else {
            None
        };
        let target_parent_index = target_parent_prefix.and_then(|prefix| {
            beam.iter()
                .position(|state| state.prefix.as_slice() == prefix)
        });

        let mut single_bins = HashMap::<(i64, u32), CausalBeamState>::new();
        let mut uncompressed_candidates = Vec::<CausalBeamState>::new();
        for (state_index, state) in beam.iter().enumerate() {
            let next_logits = &logits[state_index];
            let abs_mass_error = (state.neutral_mass - target).abs();
            if state.residue_count > 0 && abs_mass_error <= mass_tolerance_da {
                let eos_log_probability =
                    selected_log_softmax(next_logits, FOUNDATION_DIFFUSION_EOS as usize)?;
                let ar_total_log_probability = state.ar_total_log_probability + eos_log_probability;
                let locked_score = foundation_fragment_causal_rerank_score(
                    state.fragment_score,
                    ar_total_log_probability,
                    causal_weight,
                );
                let priority = variant.priority_mode.score(
                    state.fragment_score,
                    ar_total_log_probability,
                    causal_weight,
                );
                let mut row = vec![FOUNDATION_DIFFUSION_PAD; config.max_tokens];
                for (token_position, &token) in state.prefix.iter().enumerate() {
                    row[token_position] = token;
                }
                row[state.prefix.len()] = FOUNDATION_DIFFUSION_EOS;
                let candidate = CausalBeamCandidate {
                    tokens: row.clone(),
                    ar_total_log_probability,
                    fragment_score: state.fragment_score,
                    matched_cleavages: state.matched_cleavages,
                    fragment_causal_score: locked_score,
                    abs_mass_error_da: abs_mass_error,
                };
                let audit_candidate = AuditCompletedCandidate {
                    candidate,
                    priority,
                };
                completed
                    .entry(row)
                    .and_modify(|existing| {
                        if audit_candidate.priority > existing.priority {
                            *existing = audit_candidate.clone();
                        }
                    })
                    .or_insert(audit_candidate);
            }

            if position + 1 >= config.max_tokens {
                continue;
            }

            let mut token_order: Vec<usize> =
                (FOUNDATION_DIFFUSION_EOS as usize + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE).collect();
            token_order.sort_by(|&left, &right| next_logits[right].total_cmp(&next_logits[left]));
            for token_index in token_order {
                if !next_logits[token_index].is_finite() {
                    continue;
                }
                let token = token_index as u32;
                if !mass_beam_token_allowed(&state.prefix, token, position, config.max_tokens - 1) {
                    continue;
                }
                let Some(token_mass) = foundation_diffusion_token_mass_da(token) else {
                    continue;
                };
                let neutral_mass = state.neutral_mass + token_mass;
                if neutral_mass > target + mass_tolerance_da {
                    continue;
                }
                let remaining_slots = config.max_tokens - 1 - (state.prefix.len() + 1);
                let maximum_reachable_mass = neutral_mass + remaining_slots as f64 * max_token_mass;
                if maximum_reachable_mass + mass_tolerance_da < target {
                    continue;
                }

                let token_log_probability = selected_log_softmax(next_logits, token_index)?;
                let ar_total_log_probability =
                    state.ar_total_log_probability + token_log_probability;
                let mut fragment_score = state.fragment_score;
                let mut matched_cleavages = state.matched_cleavages;
                let is_residue = foundation_diffusion_token_residue(token).is_some();
                if is_residue && state.residue_count > 0 {
                    let prefix_mass_without_water =
                        state.neutral_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA;
                    let evidence = cleavage_fragment_evidence(
                        prefix_mass_without_water,
                        target,
                        observed_peaks,
                        max_fragment_charge,
                        fragment_tolerance_ppm,
                    );
                    fragment_score += evidence.score;
                    matched_cleavages += usize::from(evidence.matched);
                }
                let residue_count = state.residue_count + usize::from(is_residue);
                let priority = variant.priority_mode.score(
                    fragment_score,
                    ar_total_log_probability,
                    causal_weight,
                );
                let mut prefix = state.prefix.clone();
                prefix.push(token);
                let candidate = CausalBeamState {
                    prefix,
                    neutral_mass,
                    ar_total_log_probability,
                    fragment_score,
                    matched_cleavages,
                    residue_count,
                    priority,
                };
                let mass_bin = (neutral_mass / mass_bin_width).round() as i64;
                let key = (mass_bin, token);
                if variant.bin_capacity == 1 {
                    match single_bins.get_mut(&key) {
                        Some(existing) if candidate.priority > existing.priority => {
                            *existing = candidate;
                        }
                        None => {
                            single_bins.insert(key, candidate);
                        }
                        _ => {}
                    }
                } else {
                    // Counterfactual variants intentionally disable the
                    // (mass_bin, last_token) state compression. Their
                    // bin_capacity equals the global beam width, so any
                    // per-bin truncation could only remove states that the
                    // global width could otherwise retain. Keep every
                    // expansion here and let the global beam be the sole cap.
                    uncompressed_candidates.push(candidate);
                }
            }
        }

        let mut pretruncate: Vec<CausalBeamState> = if variant.bin_capacity == 1 {
            single_bins.into_values().collect()
        } else {
            uncompressed_candidates
        };
        pretruncate.sort_by(|left, right| right.priority.total_cmp(&left.priority));

        if position < target_nonterminal.len() {
            let target_token = target_nonterminal[position];
            let target_prefix = target_nonterminal[..=position].to_vec();
            let parent_live = target_parent_index.is_some();
            let mut child_eligible = false;
            let mut child_priority = shadow_priority(&shadow[position], variant.priority_mode);
            let mut child_key = None;
            let mut eligibility_failure: Option<AuditLossKind> = None;
            if let Some(state_index) = target_parent_index {
                let state = &beam[state_index];
                let next_logits = &logits[state_index];
                match build_target_child(
                    state,
                    next_logits,
                    target_token,
                    position,
                    config.max_tokens,
                    target,
                    mass_tolerance_da,
                    max_token_mass,
                    observed_peaks,
                    max_fragment_charge,
                    fragment_tolerance_ppm,
                    causal_weight,
                    variant.priority_mode,
                )? {
                    Ok((candidate, key)) => {
                        child_eligible = true;
                        child_priority = candidate.priority;
                        child_key = Some(key);
                    }
                    Err(kind) => eligibility_failure = Some(kind),
                }
            }

            let matching_state = pretruncate
                .iter()
                .position(|state| state.prefix == target_prefix);
            let child_survived_bin = matching_state.is_some();
            let child_rank_pretruncate = matching_state.map(|index| index + 1);
            let child_live_after_truncate = child_rank_pretruncate
                .map(|rank| rank <= variant.beam_width)
                .unwrap_or(false);

            let shadow_key = shadow_target_key(target_nonterminal, position, mass_bin_width);
            let same_bin_competitor = shadow_key.and_then(|key| {
                pretruncate
                    .iter()
                    .filter(|state| {
                        let state_bin = (state.neutral_mass / mass_bin_width).round() as i64;
                        let state_last = state.prefix.last().copied().unwrap_or(0);
                        (state_bin, state_last) == key && state.prefix != target_prefix
                    })
                    .max_by(|left, right| left.priority.total_cmp(&right.priority))
            });
            let shadow_rank_pretruncate = if shadow[position].token_rule_allowed
                && shadow[position].token_mass_known
                && shadow[position].mass_feasible
            {
                Some(
                    1 + pretruncate
                        .iter()
                        .filter(|state| state.priority > child_priority)
                        .count(),
                )
            } else {
                None
            };
            let beam_cutoff_state = if pretruncate.len() > variant.beam_width {
                pretruncate.get(variant.beam_width - 1)
            } else {
                pretruncate.last()
            };
            let beam_cutoff_priority = beam_cutoff_state.map(|state| state.priority);
            let beam_cutoff_prefix = beam_cutoff_state.map(|state| state.prefix.clone());
            let target_minus_cutoff = beam_cutoff_priority.map(|cutoff| child_priority - cutoff);

            if first_loss.is_none() && parent_live && !child_live_after_truncate {
                let (kind, competitor_priority, competitor_prefix) =
                    if let Some(kind) = eligibility_failure {
                        (kind, None, None)
                    } else if child_eligible && !child_survived_bin {
                        let competitor = child_key.and_then(|key| {
                            pretruncate.iter().find(|state| {
                                let bin = (state.neutral_mass / mass_bin_width).round() as i64;
                                let last = state.prefix.last().copied().unwrap_or(0);
                                (bin, last) == key
                            })
                        });
                        (
                            AuditLossKind::DedupCollision,
                            competitor.map(|state| state.priority),
                            competitor.map(|state| state.prefix.clone()),
                        )
                    } else {
                        (
                            AuditLossKind::BeamPruned,
                            beam_cutoff_priority,
                            beam_cutoff_prefix.clone(),
                        )
                    };
                first_loss = Some(AuditLossEvent {
                    kind,
                    position,
                    target_token,
                    target_priority: child_priority,
                    target_rank_pretruncate: child_rank_pretruncate.or(shadow_rank_pretruncate),
                    candidates_pretruncate: pretruncate.len(),
                    competitor_priority,
                    competitor_prefix,
                });
            }

            steps.push(AuditBeamStep {
                position,
                parent_live,
                child_eligible,
                child_survived_bin,
                child_rank_pretruncate,
                child_live_after_truncate,
                shadow_rank_pretruncate,
                candidates_pretruncate: pretruncate.len(),
                beam_size_after_truncate: pretruncate.len().min(variant.beam_width),
                target_priority: child_priority,
                beam_cutoff_priority,
                beam_cutoff_prefix,
                target_minus_cutoff,
                same_bin_competitor_priority: same_bin_competitor.map(|state| state.priority),
                same_bin_competitor_prefix: same_bin_competitor.map(|state| state.prefix.clone()),
            });
        }

        beam = pretruncate;
        beam.truncate(variant.beam_width);
    }

    let mut completed: Vec<AuditCompletedCandidate> = completed.into_values().collect();
    completed.sort_by(|left, right| {
        right
            .priority
            .total_cmp(&left.priority)
            .then_with(|| {
                left.candidate
                    .abs_mass_error_da
                    .total_cmp(&right.candidate.abs_mass_error_da)
            })
            .then_with(|| {
                right
                    .candidate
                    .ar_total_log_probability
                    .total_cmp(&left.candidate.ar_total_log_probability)
            })
    });
    let completed_before_final_count = completed.len();
    let target_completed_rank = completed
        .iter()
        .position(|entry| entry.candidate.tokens == target_tokens)
        .map(|index| index + 1);
    let target_completed_before_final = target_completed_rank.is_some();
    let il_completed_rank = completed
        .iter()
        .position(|entry| {
            vocabulary
                .decode(&entry.candidate.tokens)
                .map(|peptide| normalize_il(&peptide.sequence) == target_il)
                .unwrap_or(false)
        })
        .map(|index| index + 1);
    let il_completed_before_final = il_completed_rank.is_some();
    if first_loss.is_none() {
        if let Some(rank) = target_completed_rank {
            if rank > variant.final_candidates {
                let cutoff = completed.get(variant.final_candidates.saturating_sub(1));
                first_loss = Some(AuditLossEvent {
                    kind: AuditLossKind::FinalCandidateTruncation,
                    position: target_nonterminal.len(),
                    target_token: FOUNDATION_DIFFUSION_EOS,
                    target_priority: shadow_priority(
                        shadow.last().expect("shadow includes EOS"),
                        variant.priority_mode,
                    ),
                    target_rank_pretruncate: Some(rank),
                    candidates_pretruncate: completed_before_final_count,
                    competitor_priority: cutoff.map(|entry| entry.priority),
                    competitor_prefix: cutoff.map(|entry| entry.candidate.tokens.clone()),
                });
            }
        } else if target_nonterminal.len() < config.max_tokens {
            let target_prefix_live = beam
                .iter()
                .any(|state| state.prefix.as_slice() == target_nonterminal);
            if target_prefix_live {
                first_loss = Some(AuditLossEvent {
                    kind: AuditLossKind::TerminationMissing,
                    position: target_nonterminal.len(),
                    target_token: FOUNDATION_DIFFUSION_EOS,
                    target_priority: shadow_priority(
                        shadow.last().expect("shadow includes EOS"),
                        variant.priority_mode,
                    ),
                    target_rank_pretruncate: None,
                    candidates_pretruncate: completed_before_final_count,
                    competitor_priority: None,
                    competitor_prefix: None,
                });
            }
        }
    }

    completed.truncate(variant.final_candidates);
    let exact_generated = target_completed_rank
        .map(|rank| rank <= variant.final_candidates)
        .unwrap_or(false);
    let il_sequence_generated = il_completed_rank
        .map(|rank| rank <= variant.final_candidates)
        .unwrap_or(false);

    Ok(AuditBeamRun {
        variant,
        final_candidates: completed,
        completed_before_final_count,
        target_completed_before_final,
        target_completed_rank,
        il_completed_before_final,
        il_completed_rank,
        exact_generated,
        il_sequence_generated,
        first_loss,
        steps,
    })
}

fn shadow_priority(step: &TargetShadowStep, mode: AuditPriorityMode) -> f64 {
    match mode {
        AuditPriorityMode::LockedFragmentCausal => step.locked_priority,
        AuditPriorityMode::ArOnly => step.ar_only_priority,
    }
}

fn shadow_target_key(
    target_nonterminal: &[u32],
    position: usize,
    mass_bin_width: f64,
) -> Option<(i64, u32)> {
    if position >= target_nonterminal.len() {
        return None;
    }
    let mut mass = FOUNDATION_PEPTIDE_WATER_MASS_DA;
    for &token in &target_nonterminal[..=position] {
        mass += foundation_diffusion_token_mass_da(token)?;
    }
    Some((
        (mass / mass_bin_width).round() as i64,
        target_nonterminal[position],
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_target_child(
    state: &CausalBeamState,
    next_logits: &[f32],
    token: u32,
    position: usize,
    max_tokens: usize,
    target: f64,
    mass_tolerance_da: f64,
    max_token_mass: f64,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    causal_weight: f64,
    priority_mode: AuditPriorityMode,
) -> Result<std::result::Result<(CausalBeamState, (i64, u32)), AuditLossKind>> {
    let token_index = token as usize;
    if token_index >= next_logits.len() || !next_logits[token_index].is_finite() {
        return Ok(Err(AuditLossKind::TokenLogitNonfinite));
    }
    if !mass_beam_token_allowed(&state.prefix, token, position, max_tokens - 1) {
        return Ok(Err(AuditLossKind::TokenRuleRejected));
    }
    let Some(token_mass) = foundation_diffusion_token_mass_da(token) else {
        return Ok(Err(AuditLossKind::TokenMassMissing));
    };
    let neutral_mass = state.neutral_mass + token_mass;
    if neutral_mass > target + mass_tolerance_da {
        return Ok(Err(AuditLossKind::MassOvershoot));
    }
    let remaining_slots = max_tokens - 1 - (state.prefix.len() + 1);
    let maximum_reachable_mass = neutral_mass + remaining_slots as f64 * max_token_mass;
    if maximum_reachable_mass + mass_tolerance_da < target {
        return Ok(Err(AuditLossKind::MassUnreachable));
    }
    let token_log_probability = selected_log_softmax(next_logits, token_index)?;
    let ar_total_log_probability = state.ar_total_log_probability + token_log_probability;
    let mut fragment_score = state.fragment_score;
    let mut matched_cleavages = state.matched_cleavages;
    let is_residue = foundation_diffusion_token_residue(token).is_some();
    if is_residue && state.residue_count > 0 {
        let prefix_mass_without_water = state.neutral_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA;
        let evidence = cleavage_fragment_evidence(
            prefix_mass_without_water,
            target,
            observed_peaks,
            max_fragment_charge,
            fragment_tolerance_ppm,
        );
        fragment_score += evidence.score;
        matched_cleavages += usize::from(evidence.matched);
    }
    let residue_count = state.residue_count + usize::from(is_residue);
    let priority = priority_mode.score(fragment_score, ar_total_log_probability, causal_weight);
    let mut prefix = state.prefix.clone();
    prefix.push(token);
    let candidate = CausalBeamState {
        prefix,
        neutral_mass,
        ar_total_log_probability,
        fragment_score,
        matched_cleavages,
        residue_count,
        priority,
    };
    let mass_bin = (neutral_mass / mass_tolerance_da.max(0.05)).round() as i64;
    Ok(Ok((candidate, (mass_bin, token))))
}

fn classify_prefix_failure(
    baseline: &AuditBeamRun,
    locked_wide: &AuditBeamRun,
    ar_wide: &AuditBeamRun,
    precursor_mass: Option<f64>,
    mass_feasible: bool,
    token_path_valid: bool,
) -> (&'static str, &'static str) {
    if baseline.exact_generated {
        return (
            "baseline_literal_success",
            "none_generator_already_recovers_literal_target",
        );
    }
    if baseline.il_sequence_generated {
        return (
            "baseline_il_only_success",
            "none_treat_I_L_as_mass_spectrometrically_indistinguishable",
        );
    }
    if precursor_mass.is_none() {
        return (
            "precursor_mass_missing",
            "repair_or_define_precursor_mass_context_before_search_changes",
        );
    }
    if !mass_feasible {
        return (
            "precursor_mass_infeasible",
            "investigate_precursor_mass_or_peptidoform_mass_model_not_beam_width",
        );
    }
    if !token_path_valid {
        return (
            "target_token_constraint",
            "repair_token_or_modification_search_validity_not_model_training",
        );
    }
    if locked_wide.exact_generated || locked_wide.il_sequence_generated {
        let class = match baseline.first_loss.as_ref().map(|loss| &loss.kind) {
            Some(AuditLossKind::DedupCollision) => "search_state_dedup_limited",
            Some(AuditLossKind::BeamPruned) => "search_beam_width_limited",
            Some(AuditLossKind::FinalCandidateTruncation) => "search_final_candidate_limited",
            _ => "search_limited_wide_rescue",
        };
        return (
            class,
            "redesign_search_diversity_once_then_freeze_search_policy_no_more_width_sweeps",
        );
    }
    if ar_wide.exact_generated || ar_wide.il_sequence_generated {
        return (
            "fragment_priority_conflict",
            "replace_or_delay_prefix_fragment_heuristic_keep_causal_model_and_mass_constraint",
        );
    }
    if locked_wide.target_completed_before_final
        || locked_wide.il_completed_before_final
        || ar_wide.target_completed_before_final
        || ar_wide.il_completed_before_final
    {
        return (
            "wide_final_candidate_scoring_limited",
            "repair_finalization_or_final_candidate_selection_not_prefix_model",
        );
    }
    if matches!(
        locked_wide.first_loss.as_ref().map(|loss| &loss.kind),
        Some(&AuditLossKind::TerminationMissing)
    ) || matches!(
        ar_wide.first_loss.as_ref().map(|loss| &loss.kind),
        Some(&AuditLossKind::TerminationMissing)
    ) {
        return (
            "termination_logic_failure",
            "repair_EOS_mass_completion_logic_directly",
        );
    }
    (
        "causal_ar_score_limited_after_wide_no_dedup",
        "stop_search_diagnostics_and_improve_spectrum_conditioned_causal_representation_or_training",
    )
}

fn write_beam_steps<W: Write>(
    output: &mut W,
    record_index: usize,
    source_id: &str,
    run: &AuditBeamRun,
    shadow: &[TargetShadowStep],
) -> Result<()> {
    for step in &run.steps {
        let target = &shadow[step.position];
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{}\t{}\t{}\t{}",
            record_index,
            source_id,
            run.variant.name,
            step.position,
            target.next_token,
            target.next_token_log_probability,
            target.raw_logit_rank,
            target.legal_mass_rank.map(|rank| rank.to_string()).unwrap_or_default(),
            target.cumulative_ar_total,
            target.cumulative_fragment_score,
            target.locked_priority,
            target.ar_only_priority,
            target.token_rule_allowed,
            target.token_mass_known,
            target.mass_feasible,
            step.parent_live,
            step.child_eligible,
            step.child_survived_bin,
            step.child_rank_pretruncate.map(|rank| rank.to_string()).unwrap_or_default(),
            step.child_live_after_truncate,
            step.shadow_rank_pretruncate.map(|rank| rank.to_string()).unwrap_or_default(),
            step.candidates_pretruncate,
            step.beam_size_after_truncate,
            step.target_priority,
            step.beam_cutoff_priority.map(|value| format!("{value:.8}")).unwrap_or_default(),
            step.beam_cutoff_prefix.as_ref().map(|tokens| format_token_prefix(tokens)).unwrap_or_default(),
            step.target_minus_cutoff.map(|value| format!("{value:.8}")).unwrap_or_default(),
            step.same_bin_competitor_priority.map(|value| format!("{value:.8}")).unwrap_or_default(),
            step.same_bin_competitor_prefix.as_ref().map(|tokens| format_token_prefix(tokens)).unwrap_or_default(),
        )?;
    }
    if let Some(target) = shadow.last() {
        let parent_live = run
            .steps
            .last()
            .map(|step| step.child_live_after_truncate)
            .unwrap_or(true);
        let final_live = run
            .target_completed_rank
            .map(|rank| rank <= run.variant.final_candidates)
            .unwrap_or(false);
        let cutoff = run
            .final_candidates
            .last()
            .map(|candidate| candidate.priority);
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t\t{}\t\t\t",
            record_index,
            source_id,
            run.variant.name,
            target.position,
            target.next_token,
            target.next_token_log_probability,
            target.raw_logit_rank,
            target.cumulative_ar_total,
            target.cumulative_fragment_score,
            target.locked_priority,
            target.ar_only_priority,
            target.token_rule_allowed,
            target.token_mass_known,
            target.mass_feasible,
            parent_live,
            run.target_completed_before_final,
            run.target_completed_before_final,
            run.target_completed_rank.map(|rank| rank.to_string()).unwrap_or_default(),
            final_live,
            run.target_completed_rank.map(|rank| rank.to_string()).unwrap_or_default(),
            run.completed_before_final_count,
            run.final_candidates.len(),
            shadow_priority(target, run.variant.priority_mode),
            cutoff.map(|value| format!("{value:.8}")).unwrap_or_default(),
            cutoff
                .map(|value| format!("{:.8}", shadow_priority(target, run.variant.priority_mode) - value))
                .unwrap_or_default(),
        )?;
    }
    Ok(())
}

fn format_token_prefix(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn write_audit_summary(path: &std::path::Path, aggregate: &AuditAggregate) -> Result<()> {
    let mut output = BufWriter::new(fs::File::create(path)?);
    writeln!(output, "section\tkey\tcount\tfraction")?;
    let records = aggregate.records.max(1) as f64;
    for (key, value) in [
        (
            "baseline_literal_success",
            aggregate.baseline_literal_success,
        ),
        ("baseline_il_success", aggregate.baseline_il_success),
        (
            "locked_wide_literal_success",
            aggregate.locked_wide_literal_success,
        ),
        ("locked_wide_il_success", aggregate.locked_wide_il_success),
        ("ar_wide_literal_success", aggregate.ar_wide_literal_success),
        ("ar_wide_il_success", aggregate.ar_wide_il_success),
    ] {
        writeln!(
            output,
            "recall\t{}\t{}\t{:.8}",
            key,
            value,
            value as f64 / records
        )?;
    }
    write_count_section(
        &mut output,
        "decision_class",
        &aggregate.classes,
        aggregate.records,
    )?;
    write_count_section(
        &mut output,
        "baseline_first_loss",
        &aggregate.baseline_losses,
        aggregate.records,
    )?;
    write_count_section(
        &mut output,
        "locked_wide_first_loss",
        &aggregate.locked_wide_losses,
        aggregate.records,
    )?;
    write_count_section(
        &mut output,
        "ar_wide_first_loss",
        &aggregate.ar_wide_losses,
        aggregate.records,
    )?;
    output.flush()?;
    Ok(())
}

fn write_count_section<W: Write>(
    output: &mut W,
    section: &str,
    counts: &HashMap<String, usize>,
    denominator: usize,
) -> Result<()> {
    let mut rows: Vec<(&String, &usize)> = counts.iter().collect();
    rows.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));
    let denominator = denominator.max(1) as f64;
    for (key, value) in rows {
        writeln!(
            output,
            "{}\t{}\t{}\t{:.8}",
            section,
            key,
            value,
            *value as f64 / denominator,
        )?;
    }
    Ok(())
}

fn print_audit_summary(aggregate: &AuditAggregate) {
    let records = aggregate.records.max(1) as f64;
    println!("prefix_survival_summary\trecords\t{}", aggregate.records);
    println!(
        "prefix_survival_summary\tbaseline_literal_success\t{}\t{:.6}",
        aggregate.baseline_literal_success,
        aggregate.baseline_literal_success as f64 / records
    );
    println!(
        "prefix_survival_summary\tbaseline_il_success\t{}\t{:.6}",
        aggregate.baseline_il_success,
        aggregate.baseline_il_success as f64 / records
    );
    println!(
        "prefix_survival_summary\tlocked_wide_literal_success\t{}\t{:.6}",
        aggregate.locked_wide_literal_success,
        aggregate.locked_wide_literal_success as f64 / records
    );
    println!(
        "prefix_survival_summary\tlocked_wide_il_success\t{}\t{:.6}",
        aggregate.locked_wide_il_success,
        aggregate.locked_wide_il_success as f64 / records
    );
    println!(
        "prefix_survival_summary\tar_wide_literal_success\t{}\t{:.6}",
        aggregate.ar_wide_literal_success,
        aggregate.ar_wide_literal_success as f64 / records
    );
    println!(
        "prefix_survival_summary\tar_wide_il_success\t{}\t{:.6}",
        aggregate.ar_wide_il_success,
        aggregate.ar_wide_il_success as f64 / records
    );
    let mut classes: Vec<(&String, &usize)> = aggregate.classes.iter().collect();
    classes.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));
    for (class, count) in classes {
        println!(
            "prefix_survival_decision\tclass={}\tcount={}\tfraction={:.6}",
            class,
            count,
            *count as f64 / records,
        );
    }
}

fn mass_beam_token_allowed(
    prefix: &[u32],
    token: u32,
    position: usize,
    nonterminal_positions: usize,
) -> bool {
    if foundation_diffusion_token_residue(token).is_some() {
        return true;
    }
    if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
        return position == 0 && nonterminal_positions >= 2;
    }
    if token < FOUNDATION_DIFFUSION_RESIDUE_ACETYL {
        return false;
    }
    let Some(&previous) = prefix.last() else {
        return false;
    };
    let Some(previous_residue) = foundation_diffusion_token_residue(previous) else {
        return false;
    };
    foundation_diffusion_residue_ptm_valid(token, previous_residue)
}

fn active_token_length(tokens: &[u32], max_tokens: usize) -> Result<usize> {
    if tokens.len() != max_tokens {
        anyhow::bail!(
            "candidate token width {} does not match configured {max_tokens}",
            tokens.len()
        );
    }
    let active_length = tokens
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap_or(max_tokens);
    if active_length == 0 || tokens[active_length - 1] != FOUNDATION_DIFFUSION_EOS {
        anyhow::bail!("candidate token row must end its active prefix with EOS");
    }
    Ok(active_length)
}

fn selected_log_softmax(logits: &[f32], selected: usize) -> Result<f64> {
    if selected >= logits.len() || logits.is_empty() {
        anyhow::bail!("selected neural-reranking class {selected} is outside logits");
    }
    let max = logits
        .iter()
        .copied()
        .map(f64::from)
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        anyhow::bail!("neural-reranking logits contain no finite values");
    }
    let normalizer: f64 = logits
        .iter()
        .copied()
        .map(f64::from)
        .filter(|value| value.is_finite())
        .map(|value| (value - max).exp())
        .sum();
    let selected_value = f64::from(logits[selected]);
    if !selected_value.is_finite() || !(normalizer > 0.0 && normalizer.is_finite()) {
        anyhow::bail!("selected neural-reranking logit is not finite");
    }
    Ok(selected_value - max - normalizer.ln())
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
    let batch = records.len();
    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(charge, batch, device)?,
        charge_present: Tensor::from_vec(charge_present, batch, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, batch, device)?,
        precursor_mz_present: Tensor::from_vec(precursor_mz_present, batch, device)?,
        nce: Tensor::zeros(batch, DType::F32, device)?,
        nce_present: Tensor::zeros(batch, DType::F32, device)?,
        instrument_ids: Tensor::zeros(batch, DType::U32, device)?,
        instrument_present: Tensor::zeros(batch, DType::F32, device)?,
    })
}

fn precursor_neutral_mass(record: &FoundationTrainingRecord) -> Result<Option<f64>> {
    match (record.context.precursor_mz, record.context.charge) {
        (Some(mz), Some(charge)) if charge > 0 => Ok(Some(
            foundation_precursor_neutral_mass(mz as f64, charge as i32)
                .map_err(anyhow::Error::msg)?,
        )),
        _ => Ok(None),
    }
}

fn normalized_observed_peaks(spectrum: &FoundationSpectrum) -> Vec<(f64, f64)> {
    let max_intensity = spectrum
        .peaks
        .iter()
        .filter_map(|peak| {
            (peak.intensity.is_finite() && peak.intensity > 0.0).then_some(peak.intensity as f64)
        })
        .fold(0.0f64, f64::max)
        .max(f64::EPSILON);
    let mut peaks: Vec<(f64, f64)> = spectrum
        .peaks
        .iter()
        .filter_map(|peak| {
            (peak.mz.is_finite()
                && peak.mz > 0.0
                && peak.intensity.is_finite()
                && peak.intensity > 0.0)
                .then_some((
                    peak.mz as f64,
                    (peak.intensity as f64 / max_intensity).clamp(0.0, 1.0),
                ))
        })
        .collect();
    peaks.sort_by(|left, right| left.0.total_cmp(&right.0));
    peaks
}

fn cleavage_fragment_evidence(
    prefix_mass_without_water: f64,
    precursor_neutral_mass: f64,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
) -> CleavageEvidence {
    if !(prefix_mass_without_water > 0.0
        && precursor_neutral_mass > prefix_mass_without_water
        && !observed_peaks.is_empty())
    {
        return CleavageEvidence::default();
    }
    let suffix_with_water = precursor_neutral_mass - prefix_mass_without_water;
    let mut best_b = 0.0f64;
    let mut best_y = 0.0f64;
    for charge in 1..=max_fragment_charge.max(1) {
        let z = charge as f64;
        let b_mz = (prefix_mass_without_water + z * 1.007_276_466_77) / z;
        let y_mz = (suffix_with_water + z * 1.007_276_466_77) / z;
        best_b = best_b.max(theoretical_peak_match_score(
            b_mz,
            observed_peaks,
            fragment_tolerance_ppm,
        ));
        best_y = best_y.max(theoretical_peak_match_score(
            y_mz,
            observed_peaks,
            fragment_tolerance_ppm,
        ));
    }
    let score = best_b + best_y;
    CleavageEvidence {
        score,
        matched: score > 0.0,
    }
}

fn theoretical_peak_match_score(
    theoretical_mz: f64,
    observed_peaks: &[(f64, f64)],
    tolerance_ppm: f64,
) -> f64 {
    if !(theoretical_mz > 0.0 && theoretical_mz.is_finite()) {
        return 0.0;
    }
    let sigma = (theoretical_mz * tolerance_ppm * 1e-6).max(0.0025);
    let cutoff = 3.0 * sigma;
    let mut best = 0.0f64;
    for &(observed_mz, normalized_intensity) in observed_peaks {
        let error = (observed_mz - theoretical_mz).abs();
        if error > cutoff {
            continue;
        }
        let mass_weight = (-0.5 * (error / sigma).powi(2)).exp();
        let intensity_weight = normalized_intensity.sqrt();
        best = best.max(mass_weight * intensity_weight);
    }
    best
}

fn load_matching_variables(
    varmap: &VarMap,
    checkpoint: &std::path::Path,
    device: &Device,
) -> Result<()> {
    let tensors = candle_core::safetensors::load(checkpoint, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("unified inverse VarMap lock poisoned"))?;
    let mut missing = Vec::new();
    for (name, variable) in data.iter() {
        match tensors.get(name) {
            Some(tensor) => {
                if tensor.dims() != variable.as_tensor().dims() {
                    anyhow::bail!(
                        "shape mismatch for '{name}': checkpoint {:?}, model {:?}",
                        tensor.dims(),
                        variable.as_tensor().dims()
                    );
                }
                variable.set(tensor)?;
            }
            None => missing.push(name.clone()),
        }
    }
    drop(data);
    if !missing.is_empty() {
        anyhow::bail!(
            "unified checkpoint is missing inverse variables: {}",
            missing.join(", ")
        );
    }
    Ok(())
}

fn normalize_il(sequence: &str) -> String {
    sequence
        .chars()
        .map(|residue| {
            if residue == 'I' || residue == 'L' {
                'J'
            } else {
                residue
            }
        })
        .collect()
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

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
