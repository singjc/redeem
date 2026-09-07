//! Generate peptide/PTM candidates from observed spectra with a unified foundation checkpoint.
//!
//! This is the first true inverse-generation evaluator: sequence length is predicted from
//! spectrum/precursor context, active tokens start from MASK, and categorical reverse refinement
//! proceeds without access to the clean peptide. The clean validation peptidoform is used only
//! after generation for metrics. Generated candidates are additionally reranked with an
//! all-MASK x0 score from the trained inverse model: candidate identity is never supplied to the
//! decoder input and is used only to read the candidate token probabilities after inference.
//! An optional v0.12.2 causal checkpoint adds true prefix-conditioned sequence likelihood
//! (including EOS) as a parallel candidate ranking. v0.12.4 can additionally enable an
//! opt-in precursor-mass-constrained causal prefix beam that augments, rather than replaces,
//! the frozen diffusion candidate pool.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_build_cleavage_graph, foundation_canonicalize_reverse_causal_token_row,
    foundation_cleavage_graph_structured_decode, foundation_cleavage_graph_true_path_audit,
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_reverse_probabilities,
    foundation_diffusion_token_mass_da, foundation_diffusion_token_residue,
    foundation_fragment_causal_rerank_score, foundation_precursor_mass_error_da,
    foundation_precursor_neutral_mass, foundation_reverse_causal_token_row, load_foundation_corpus,
    read_foundation_training_run_config, FoundationBenchmarkManifest, FoundationCausalCollator,
    FoundationDiffusionCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationPartition, FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideSpectrumCausalModel, PeptideSpectrumCleavageGraphStructuredScorer,
    PeptideSpectrumDiffusionModel, PeptidoformInput, PrecursorContextBatch,
    FOUNDATION_CAUSAL_RERANK_POLICY_V0123, FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123,
    FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_ANCHOR_BRIDGE_EDGES_V01317,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_NAMESPACE_V01318,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_OBJECTIVE_V01318, FOUNDATION_DIFFUSION_EOS,
    FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_OBJECTIVE_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315, FOUNDATION_PEPTIDE_WATER_MASS_DA,
    FOUNDATION_REVERSE_CAUSAL_DIRECTION_V01313, FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313,
};
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct UnifiedCheckpointMetadata {
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    completed_steps: usize,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Deserialize)]
struct ReverseCausalCheckpointMetadata {
    objective: String,
    direction: String,
    parameter_namespace: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_unified_checkpoint: String,
    parent_unified_completed_steps: usize,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Deserialize)]
struct IterativeRefinementCheckpointMetadata {
    objective: String,
    parameter_namespace: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_unified_checkpoint: String,
    mask_fraction: f64,
    refinement_rounds: usize,
    seed_hypotheses: usize,
    replacement_topk: usize,
    replacement_beam_width: usize,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Deserialize)]
struct CleavageGraphCheckpointMetadata {
    objective: String,
    parameter_namespace: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_unified_checkpoint: String,
    reverse_causal_checkpoint: String,
    graph_mass_tolerance_da: f64,
    maximum_outgoing_edges: usize,
    k_best_graph_paths: usize,
    maximum_fragment_charge: usize,
    maximum_anchor_bridge_edges: usize,
    feature_dim: usize,
    hidden_dim: usize,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone)]
struct GeneratedCandidate {
    tokens: Vec<u32>,
    peptide: PeptidoformInput,
    reverse_log_probability: f64,
    fragment_score: f64,
    matched_cleavages: usize,
    neural_all_mask_log_probability: f64,
    neural_length_log_probability: f64,
    hybrid_score: f64,
    ar_total_log_probability: f64,
    ar_mean_log_probability: f64,
    ar_perplexity: f64,
    fragment_causal_score: f64,
    reverse_ar_total_log_probability: f64,
    reverse_ar_mean_log_probability: f64,
    reverse_ar_perplexity: f64,
    bidirectional_ar_total_log_probability: f64,
    fragment_bidirectional_causal_score: f64,
    mass_error_da: Option<f64>,
    mass_valid: bool,
    from_diffusion: bool,
    from_causal_beam: bool,
    from_reverse_causal_beam: bool,
    from_bidirectional_mitm: bool,
    from_iterative_refinement: bool,
    from_cleavage_graph: bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct MitmRankCutoffMetrics {
    top1: usize,
    top8: usize,
    top32: usize,
    top64: usize,
    top128: usize,
    top256: usize,
}

impl MitmRankCutoffMetrics {
    fn observe(&mut self, rank: Option<usize>) {
        let Some(rank) = rank else {
            return;
        };
        self.top1 += usize::from(rank <= 1);
        self.top8 += usize::from(rank <= 8);
        self.top32 += usize::from(rank <= 32);
        self.top64 += usize::from(rank <= 64);
        self.top128 += usize::from(rank <= 128);
        self.top256 += usize::from(rank <= 256);
    }
}

#[derive(Debug, Default)]
struct GenerationMetrics {
    records: usize,
    predicted_length_exact: usize,
    predicted_length_abs_error: usize,
    chains: usize,
    final_states: usize,
    valid_decodes: usize,
    unique_candidates: usize,
    mass_valid_candidates: usize,
    records_with_mass_valid_candidate: usize,
    raw_top1_peptidoform_exact: usize,
    mass_top1_peptidoform_exact: usize,
    mass_topk_peptidoform_exact: usize,
    mass_top1_sequence_exact: usize,
    mass_topk_sequence_exact: usize,
    mass_top1_il_sequence_exact: usize,
    mass_topk_il_sequence_exact: usize,
    fragment_top1_peptidoform_exact: usize,
    fragment_top1_sequence_exact: usize,
    fragment_top1_il_sequence_exact: usize,
    candidate_pool_mass_valid_peptidoform_exact: usize,
    candidate_pool_mass_valid_sequence_exact: usize,
    candidate_pool_mass_valid_il_sequence_exact: usize,
    frozen_parent_pool_mass_valid_peptidoform_exact: usize,
    frozen_parent_pool_mass_valid_sequence_exact: usize,
    frozen_parent_pool_mass_valid_il_sequence_exact: usize,
    diffusion_pool_mass_valid_peptidoform_exact: usize,
    diffusion_pool_mass_valid_sequence_exact: usize,
    diffusion_pool_mass_valid_il_sequence_exact: usize,
    causal_beam_pool_mass_valid_peptidoform_exact: usize,
    causal_beam_pool_mass_valid_sequence_exact: usize,
    causal_beam_pool_mass_valid_il_sequence_exact: usize,
    causal_beam_top1_peptidoform_exact: usize,
    causal_beam_top1_sequence_exact: usize,
    causal_beam_top1_il_sequence_exact: usize,
    causal_beam_final_candidates: usize,
    causal_beam_records_with_candidate: usize,
    reverse_causal_beam_pool_mass_valid_peptidoform_exact: usize,
    reverse_causal_beam_pool_mass_valid_sequence_exact: usize,
    reverse_causal_beam_pool_mass_valid_il_sequence_exact: usize,
    reverse_causal_beam_final_candidates: usize,
    reverse_causal_beam_records_with_candidate: usize,
    mitm_prefix_records_with_states: usize,
    mitm_suffix_records_with_states: usize,
    mitm_records_with_mass_join: usize,
    mitm_unique_mass_joins_before_cap: usize,
    mitm_joined_candidates: usize,
    mitm_selector_scored_candidates: usize,
    mitm_selector_displaced_legacy_candidates: usize,
    mitm_legacy_pool_mass_valid_peptidoform_exact: usize,
    mitm_legacy_pool_mass_valid_sequence_exact: usize,
    mitm_legacy_pool_mass_valid_il_sequence_exact: usize,
    mitm_v01320_shadow_pool_mass_valid_peptidoform_exact: usize,
    mitm_v01320_shadow_pool_mass_valid_sequence_exact: usize,
    mitm_v01320_shadow_pool_mass_valid_il_sequence_exact: usize,
    mitm_v01320_shadow_union_peptidoform_exact: usize,
    mitm_v01320_shadow_union_sequence_exact: usize,
    mitm_v01320_shadow_union_il_sequence_exact: usize,
    mitm_v01323_displaced_v01320_candidates: usize,
    mitm_precap_pool_peptidoform_exact: usize,
    mitm_precap_pool_sequence_exact: usize,
    mitm_precap_pool_il_sequence_exact: usize,
    mitm_precap_union_peptidoform_exact: usize,
    mitm_precap_union_sequence_exact: usize,
    mitm_precap_union_il_sequence_exact: usize,
    mitm_precap_legacy_literal_rank_cutoffs: MitmRankCutoffMetrics,
    mitm_precap_legacy_sequence_rank_cutoffs: MitmRankCutoffMetrics,
    mitm_precap_legacy_il_rank_cutoffs: MitmRankCutoffMetrics,
    mitm_precap_evidence_literal_rank_cutoffs: MitmRankCutoffMetrics,
    mitm_precap_evidence_sequence_rank_cutoffs: MitmRankCutoffMetrics,
    mitm_precap_evidence_il_rank_cutoffs: MitmRankCutoffMetrics,
    mitm_component_audit_incremental_deep_literal_records: usize,
    mitm_component_audit_incremental_deep_il_records: usize,
    mitm_component_audit_deep_literal_top256_any_component: usize,
    mitm_component_audit_deep_il_top256_any_component: usize,
    mitm_component_audit_deep_literal_pareto_le256: usize,
    mitm_component_audit_deep_il_pareto_le256: usize,
    mitm_component_audit_deep_literal_actionable_records: usize,
    mitm_component_audit_deep_il_actionable_records: usize,
    mitm_pool_mass_valid_peptidoform_exact: usize,
    mitm_pool_mass_valid_sequence_exact: usize,
    mitm_pool_mass_valid_il_sequence_exact: usize,
    iterative_refinement_pool_mass_valid_peptidoform_exact: usize,
    iterative_refinement_pool_mass_valid_sequence_exact: usize,
    iterative_refinement_pool_mass_valid_il_sequence_exact: usize,
    iterative_refinement_final_candidates: usize,
    iterative_refinement_records_with_candidate: usize,
    cleavage_graph_pool_mass_valid_peptidoform_exact: usize,
    cleavage_graph_pool_mass_valid_sequence_exact: usize,
    cleavage_graph_pool_mass_valid_il_sequence_exact: usize,
    cleavage_graph_final_candidates: usize,
    cleavage_graph_records_with_candidate: usize,
    cleavage_graph_true_path_structural_present: usize,
    cleavage_graph_structural_records: usize,
    cleavage_graph_true_nodes_present: usize,
    cleavage_graph_true_nodes_total: usize,
    cleavage_graph_true_edges_present: usize,
    cleavage_graph_true_edges_total: usize,
    cleavage_graph_structured_true_path_top1: usize,
    cleavage_graph_structured_true_path_top8: usize,
    cleavage_graph_structured_true_path_top32: usize,
    cleavage_graph_structured_true_path_top64: usize,
    frozen_v01313_pool_mass_valid_peptidoform_exact: usize,
    frozen_v01313_pool_mass_valid_sequence_exact: usize,
    frozen_v01313_pool_mass_valid_il_sequence_exact: usize,
    frozen_v01313_forward_ranking_peptidoform_exact: usize,
    frozen_v01313_forward_ranking_sequence_exact: usize,
    frozen_v01313_forward_ranking_il_sequence_exact: usize,
    neural_top1_peptidoform_exact: usize,
    neural_top1_sequence_exact: usize,
    neural_top1_il_sequence_exact: usize,
    hybrid_top1_peptidoform_exact: usize,
    hybrid_top1_sequence_exact: usize,
    hybrid_top1_il_sequence_exact: usize,
    causal_top1_peptidoform_exact: usize,
    causal_top1_sequence_exact: usize,
    causal_top1_il_sequence_exact: usize,
    fragment_causal_top1_peptidoform_exact: usize,
    fragment_causal_top1_sequence_exact: usize,
    fragment_causal_top1_il_sequence_exact: usize,
    fragment_bidirectional_causal_top1_peptidoform_exact: usize,
    fragment_bidirectional_causal_top1_sequence_exact: usize,
    fragment_bidirectional_causal_top1_il_sequence_exact: usize,
    records_with_candidate: usize,
    best_abs_mass_error_sum: f64,
    best_abs_mass_error_records: usize,
    best_abs_mass_errors_mass_valid: Vec<f64>,
    best_abs_mass_errors_no_mass_valid: Vec<f64>,
    target_fragment_score_sum: f64,
    top1_fragment_score_sum: f64,
    target_matched_cleavages: usize,
    top1_matched_cleavages: usize,
    target_neural_all_mask_log_probability_sum: f64,
    target_neural_length_log_probability_sum: f64,
    fragment_top1_neural_all_mask_log_probability_sum: f64,
    neural_top1_neural_all_mask_log_probability_sum: f64,
    hybrid_top1_neural_all_mask_log_probability_sum: f64,
    target_ar_total_log_probability_sum: f64,
    target_ar_mean_log_probability_sum: f64,
    fragment_top1_ar_total_log_probability_sum: f64,
    causal_top1_ar_total_log_probability_sum: f64,
    fragment_causal_top1_ar_total_log_probability_sum: f64,
    causal_scored_records: usize,
}

#[derive(Debug, Clone, Copy)]
struct AllMaskCandidateScore {
    mean_token_log_probability: f64,
    length_log_probability: f64,
}

#[derive(Debug, Clone, Copy)]
struct CausalCandidateScore {
    total_log_probability: f64,
    mean_log_probability: f64,
    perplexity: f64,
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

const FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01319: &str = "midpoint_residue_mass_join_v01319";
const FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01320: &str =
    "evidence_aware_fixed_budget_midpoint_join_v01320";
const FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01321: &str =
    "diagnostic_precap_join_oracle_audit_v01321";
const FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01322: &str =
    "terminal_deep_candidate_component_rank_audit_v01322";
const FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01323: &str =
    "final_two_view_fixed_budget_midpoint_join_v01323";

#[derive(Debug, Clone, Copy, Default)]
struct MitmOracleRanks {
    peptidoform: Option<usize>,
    sequence: Option<usize>,
    il_sequence: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct MitmPrecapOracleAudit {
    candidates: usize,
    peptidoform_present: bool,
    sequence_present: bool,
    il_sequence_present: bool,
    legacy_ranks: MitmOracleRanks,
    evidence_ranks: MitmOracleRanks,
}

#[derive(Debug, Clone, Copy, Default)]
struct MitmComponentRanks {
    fragment: MitmOracleRanks,
    prefix_ar_mean: MitmOracleRanks,
    suffix_ar_mean: MitmOracleRanks,
    prefix_ar_total: MitmOracleRanks,
    suffix_ar_total: MitmOracleRanks,
    mass_error: MitmOracleRanks,
    seam_fragment: MitmOracleRanks,
}

#[derive(Debug, Clone, Copy, Default)]
struct MitmTargetProvenance {
    found: bool,
    prefix_partial_rank: usize,
    suffix_partial_rank: usize,
    best_prefix_partial_rank: usize,
    best_suffix_partial_rank: usize,
    prefix_token_count: usize,
    suffix_token_count: usize,
    prefix_residue_count: usize,
    suffix_residue_count: usize,
    join_seam_fragment_score: f64,
    fragment_score: f64,
    prefix_ar_mean: f64,
    suffix_ar_mean: f64,
    prefix_ar_total: f64,
    suffix_ar_total: f64,
    abs_mass_error_da: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct MitmComponentAudit {
    ranks: MitmComponentRanks,
    pareto_frontier_size: usize,
    pareto_peptidoform_present: bool,
    pareto_sequence_present: bool,
    pareto_il_present: bool,
    il_provenance: MitmTargetProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MitmDirection {
    NToC,
    CToN,
}

#[derive(Debug, Clone)]
struct MitmPartialState {
    tokens: Vec<u32>,
    assigned_mass_da: f64,
    ar_total_log_probability: f64,
    fragment_score: f64,
    matched_cleavages: usize,
    residue_count: usize,
    priority: f64,
}

#[derive(Debug, Clone, Default)]
struct MitmJoinedCandidate {
    tokens: Vec<u32>,
    proposal_score: f64,
    selector_score: f64,
    selector_fragment_score: f64,
    selector_matched_cleavages: usize,
    join_mass_error_da: f64,
    component_fragment_score: f64,
    component_prefix_ar_mean: f64,
    component_suffix_ar_mean: f64,
    component_prefix_ar_total: f64,
    component_suffix_ar_total: f64,
    component_join_seam_fragment_score: f64,
    best_prefix_partial_rank: usize,
    best_suffix_partial_rank: usize,
    selector_prefix_partial_rank: usize,
    selector_suffix_partial_rank: usize,
    selector_prefix_token_count: usize,
    selector_suffix_token_count: usize,
    selector_prefix_residue_count: usize,
    selector_suffix_residue_count: usize,
    selector_join_seam_fragment_score: f64,
    selector_prefix_ar_mean: f64,
    selector_suffix_ar_mean: f64,
    selector_prefix_ar_total: f64,
    selector_suffix_ar_total: f64,
}

struct CausalReranker {
    _varmap: VarMap,
    model: PeptideSpectrumCausalModel,
    collator: FoundationCausalCollator,
}

struct IterativeRefiner {
    _varmap: VarMap,
    model: PeptideSpectrumDiffusionModel,
    collator: FoundationDiffusionCollator,
}

struct CleavageGraphProposer {
    _varmap: VarMap,
    model: PeptideSpectrumCleavageGraphStructuredScorer,
}

#[derive(Debug, Clone)]
struct RefinementFillState {
    tokens: Vec<u32>,
    assigned_mass_da: f64,
    log_probability: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 23 {
        anyhow::bail!(
            "usage: foundation_generate_unified FOUNDATION_TRAINING.yaml UNIFIED_CHECKPOINT OUTPUT.tsv [validation_records=128] [samples_per_record=16] [seed=20260912] [mass_tolerance_da=0.05] [temperature=1.0] [mass_beam_width=512] [final_candidates_per_chain=4] [fragment_tolerance_ppm=20] [spectral_beam_weight=16] [neural_rerank_weight=1.0] [causal_rerank_weight=0.1] [causal_generation_beam_width=32] [causal_generation_final_candidates=16] [reverse_causal_checkpoint=none] [reverse_causal_generation_beam_width=32] [reverse_causal_generation_final_candidates=16] [iterative_refinement_checkpoint=none] [cleavage_graph_checkpoint=none] [bidirectional_mitm=none|v01319|v01320|v01321|v01322|v01323]"
        );
    }

    let training_yaml = &args[1];
    let checkpoint_dir = PathBuf::from(&args[2]);
    let output_tsv = PathBuf::from(&args[3]);
    let validation_records = parse_or(&args, 4, 128usize)?;
    let samples_per_record = parse_or(&args, 5, 16usize)?;
    let seed = parse_or(&args, 6, 20_260_912u64)?;
    let mass_tolerance_da = parse_or(&args, 7, 0.05f64)?;
    let temperature = parse_or(&args, 8, 1.0f64)?;
    let mass_beam_width = parse_or(&args, 9, 512usize)?;
    let final_candidates_per_chain = parse_or(&args, 10, 4usize)?;
    let fragment_tolerance_ppm = parse_or(&args, 11, 20.0f64)?;
    let spectral_beam_weight = parse_or(&args, 12, 16.0f64)?;
    let neural_rerank_weight = parse_or(&args, 13, 1.0f64)?;
    let causal_rerank_weight = parse_or(&args, 14, FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123)?;
    let causal_generation_beam_width = parse_or(&args, 15, 32usize)?;
    let causal_generation_final_candidates = parse_or(&args, 16, 16usize)?;
    let reverse_causal_checkpoint = args.get(17).and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("none"))
            .then(|| PathBuf::from(trimmed))
    });
    let reverse_causal_generation_beam_width = parse_or(&args, 18, 32usize)?;
    let reverse_causal_generation_final_candidates = parse_or(&args, 19, 16usize)?;
    let iterative_refinement_checkpoint = args.get(20).and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("none"))
            .then(|| PathBuf::from(trimmed))
    });
    let cleavage_graph_checkpoint = args.get(21).and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("none"))
            .then(|| PathBuf::from(trimmed))
    });
    let (
        bidirectional_mitm,
        bidirectional_mitm_evidence_aware,
        bidirectional_mitm_precap_audit,
        bidirectional_mitm_component_audit,
        bidirectional_mitm_final_two_view,
    ) = match args.get(22).map(|value| value.trim()) {
            None | Some("") => (false, false, false, false, false),
            Some(value) if value.eq_ignore_ascii_case("none") => (false, false, false, false, false),
            Some(value)
                if value.eq_ignore_ascii_case("v01319")
                    || value.eq_ignore_ascii_case(FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01319) =>
            {
                (true, false, false, false, false)
            }
            Some(value)
                if value.eq_ignore_ascii_case("v01320")
                    || value.eq_ignore_ascii_case(FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01320) =>
            {
                (true, true, false, false, false)
            }
            Some(value)
                if value.eq_ignore_ascii_case("v01321")
                    || value.eq_ignore_ascii_case(FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01321) =>
            {
                (true, true, true, false, false)
            }
            Some(value)
                if value.eq_ignore_ascii_case("v01322")
                    || value.eq_ignore_ascii_case(FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01322) =>
            {
                (true, true, true, true, false)
            }
            Some(value)
                if value.eq_ignore_ascii_case("v01323")
                    || value.eq_ignore_ascii_case(FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01323) =>
            {
                (true, true, false, false, true)
            }
            Some(value) => anyhow::bail!(
                "unsupported bidirectional_mitm policy '{value}'; expected 'none', 'v01319', 'v01320', 'v01321', 'v01322', or 'v01323'"
            ),
        };
    let generation_partition_train = match env::var("REDEEM_GENERATION_PARTITION") {
        Ok(value) if value.eq_ignore_ascii_case("train") => true,
        Ok(value) if value.eq_ignore_ascii_case("validation") || value.trim().is_empty() => false,
        Ok(value) => anyhow::bail!(
            "unsupported REDEEM_GENERATION_PARTITION='{value}'; only 'train' or 'validation' are allowed, and TEST is intentionally unavailable"
        ),
        Err(_) => false,
    };
    if generation_partition_train && !bidirectional_mitm_final_two_view {
        anyhow::bail!("TRAIN candidate export requires frozen v0.13.23 two-view MITM");
    }
    if validation_records == 0
        || samples_per_record == 0
        || mass_beam_width == 0
        || final_candidates_per_chain == 0
    {
        anyhow::bail!(
            "validation_records, samples_per_record, mass_beam_width and final_candidates_per_chain must be positive"
        );
    }
    if !(mass_tolerance_da > 0.0 && mass_tolerance_da.is_finite()) {
        anyhow::bail!("mass_tolerance_da must be positive and finite");
    }
    if !(temperature > 0.0 && temperature.is_finite()) {
        anyhow::bail!("temperature must be positive and finite");
    }
    if !(fragment_tolerance_ppm > 0.0 && fragment_tolerance_ppm.is_finite()) {
        anyhow::bail!("fragment_tolerance_ppm must be positive and finite");
    }
    if !(spectral_beam_weight >= 0.0 && spectral_beam_weight.is_finite()) {
        anyhow::bail!("spectral_beam_weight must be finite and non-negative");
    }
    if !neural_rerank_weight.is_finite() {
        anyhow::bail!("neural_rerank_weight must be finite");
    }
    if !causal_rerank_weight.is_finite() {
        anyhow::bail!("causal_rerank_weight must be finite");
    }
    if causal_generation_beam_width > 0 && causal_generation_final_candidates == 0 {
        anyhow::bail!(
            "causal_generation_final_candidates must be positive when causal generation is enabled"
        );
    }
    if reverse_causal_checkpoint.is_some()
        && reverse_causal_generation_beam_width > 0
        && reverse_causal_generation_final_candidates == 0
    {
        anyhow::bail!(
            "reverse_causal_generation_final_candidates must be positive when reverse causal generation is enabled"
        );
    }
    if iterative_refinement_checkpoint.is_some() && reverse_causal_checkpoint.is_none() {
        anyhow::bail!(
            "v0.13.15 iterative refinement requires the accepted v0.13.13 reverse-causal checkpoint so frozen three-way parity can be verified"
        );
    }
    if cleavage_graph_checkpoint.is_some() && reverse_causal_checkpoint.is_none() {
        anyhow::bail!(
            "v0.13.18 structured cleavage graph requires the accepted v0.13.13 reverse-causal checkpoint so frozen three-way parity can be verified"
        );
    }
    if cleavage_graph_checkpoint.is_some() && iterative_refinement_checkpoint.is_some() {
        anyhow::bail!(
            "v0.13.18 structured cleavage graph cannot be combined with the rejected v0.13.15 iterative-refinement branch"
        );
    }
    if bidirectional_mitm && reverse_causal_checkpoint.is_none() {
        anyhow::bail!(
            "v0.13.19-v0.13.23 bidirectional MITM requires the accepted v0.13.13 reverse-causal checkpoint"
        );
    }
    if bidirectional_mitm
        && (iterative_refinement_checkpoint.is_some() || cleavage_graph_checkpoint.is_some())
    {
        anyhow::bail!(
            "v0.13.19-v0.13.23 bidirectional MITM must be evaluated as an isolated post-v0.13.13 proposal extension"
        );
    }
    if bidirectional_mitm {
        let fixed_policy = (generation_partition_train || validation_records == 128)
            && samples_per_record == 16
            && seed == 20_260_912
            && (mass_tolerance_da - 0.05).abs() <= 1.0e-12
            && (temperature - 1.0).abs() <= 1.0e-12
            && mass_beam_width == 512
            && final_candidates_per_chain == 4
            && (fragment_tolerance_ppm - 20.0).abs() <= 1.0e-12
            && (spectral_beam_weight - 16.0).abs() <= 1.0e-12
            && (neural_rerank_weight - 1.0).abs() <= 1.0e-12
            && (causal_rerank_weight - FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123).abs() <= 1.0e-12
            && causal_generation_beam_width == 32
            && causal_generation_final_candidates == 16
            && reverse_causal_generation_beam_width == 32
            && reverse_causal_generation_final_candidates == 16;
        if !fixed_policy {
            anyhow::bail!(
                "v0.13.19-v0.13.23 MITM uses frozen proposal/ranking/search parameters; TRAIN export may change record count only"
            );
        }
    }
    if cleavage_graph_checkpoint.is_some() {
        let fixed_policy = validation_records == 128
            && samples_per_record == 16
            && seed == 20_260_912
            && (mass_tolerance_da - 0.05).abs() <= 1.0e-12
            && (temperature - 1.0).abs() <= 1.0e-12
            && mass_beam_width == 512
            && final_candidates_per_chain == 4
            && (fragment_tolerance_ppm - 20.0).abs() <= 1.0e-12
            && (spectral_beam_weight - 16.0).abs() <= 1.0e-12
            && (neural_rerank_weight - 1.0).abs() <= 1.0e-12
            && (causal_rerank_weight - FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123).abs() <= 1.0e-12
            && causal_generation_beam_width == 32
            && causal_generation_final_candidates == 16
            && reverse_causal_generation_beam_width == 32
            && reverse_causal_generation_final_candidates == 16;
        if !fixed_policy {
            anyhow::bail!(
                "v0.13.18 structured cleavage-graph evaluation is a fixed val128 decision run; do not sweep legacy proposal/ranking/search parameters"
            );
        }
    }

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
    let generation_partition = if generation_partition_train {
        FoundationPartition::Train
    } else {
        FoundationPartition::Validation
    };
    let usable_generation = usable_indices(
        &corpus.records,
        &benchmark,
        generation_partition,
        &config,
        vocabulary,
    );
    let selected = deterministic_subset(&usable_generation, validation_records, seed);
    if selected.is_empty() {
        anyhow::bail!(
            "no usable {:?} diffusion pairs were selected",
            generation_partition
        );
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumDiffusionModel::new(config.clone(), vb)?;
    load_matching_variables(&varmap, &checkpoint_dir.join("model.safetensors"), &device)
        .with_context(|| {
            format!("failed to load unified diffusion variables {checkpoint_dir:?}")
        })?;

    let diffusion_collator = FoundationDiffusionCollator::new(config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;

    let causal_varmap = VarMap::new();
    let causal_vb = VarBuilder::from_varmap(&causal_varmap, DType::F32, &device);
    let causal_model = PeptideSpectrumCausalModel::new(config.clone(), causal_vb)?;
    load_matching_variables(
        &causal_varmap,
        &checkpoint_dir.join("model.safetensors"),
        &device,
    )
    .with_context(|| format!("failed to load unified causal variables {checkpoint_dir:?}"))?;
    let causal_reranker = Some(CausalReranker {
        _varmap: causal_varmap,
        model: causal_model,
        collator: FoundationCausalCollator::new(config.clone())?,
    });

    let reverse_causal_reranker = if let Some(reverse_checkpoint) =
        reverse_causal_checkpoint.as_ref()
    {
        let reverse_metadata_path = if reverse_checkpoint.is_dir() {
            reverse_checkpoint.join("metadata.yaml")
        } else {
            reverse_checkpoint
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("metadata.yaml")
        };
        let reverse_metadata: ReverseCausalCheckpointMetadata = serde_yaml::from_str(
            &fs::read_to_string(&reverse_metadata_path)
                .with_context(|| format!("failed to read {reverse_metadata_path:?}"))?,
        )?;
        if reverse_metadata.objective != "reverse_causal_next_token_ce_v01313" {
            anyhow::bail!(
                "reverse causal checkpoint objective {:?} is not v0.13.13",
                reverse_metadata.objective
            );
        }
        if reverse_metadata.direction != FOUNDATION_REVERSE_CAUSAL_DIRECTION_V01313
            || reverse_metadata.parameter_namespace != FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313
        {
            anyhow::bail!(
                "reverse causal checkpoint has incompatible direction/namespace: direction={:?} namespace={:?}",
                reverse_metadata.direction,
                reverse_metadata.parameter_namespace
            );
        }
        if reverse_metadata.inverse_config != config {
            anyhow::bail!(
                "reverse causal checkpoint inverse config does not match unified checkpoint"
            );
        }
        let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
        let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
        if reverse_metadata.corpus_fingerprint != corpus_fingerprint
            || reverse_metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        {
            anyhow::bail!("reverse causal checkpoint corpus/benchmark fingerprint mismatch");
        }
        if reverse_metadata.corpus_fingerprint != checkpoint_metadata.corpus_fingerprint
            || reverse_metadata.benchmark_manifest_fingerprint
                != checkpoint_metadata.benchmark_manifest_fingerprint
            || reverse_metadata.parent_unified_completed_steps
                != checkpoint_metadata.completed_steps
        {
            anyhow::bail!(
                "reverse causal checkpoint parent lineage does not match evaluated unified parent: reverse corpus={} benchmark={} parent_completed_steps={}, unified corpus={} benchmark={} completed_steps={}",
                reverse_metadata.corpus_fingerprint,
                reverse_metadata.benchmark_manifest_fingerprint,
                reverse_metadata.parent_unified_completed_steps,
                checkpoint_metadata.corpus_fingerprint,
                checkpoint_metadata.benchmark_manifest_fingerprint,
                checkpoint_metadata.completed_steps
            );
        }
        let expected_parent = checkpoint_dir.join("model.safetensors");
        let recorded_parent = PathBuf::from(&reverse_metadata.parent_unified_checkpoint);
        if recorded_parent != expected_parent {
            println!(
                "reverse_causal_parent_relocation\trecorded={}\tevaluated={}\tlineage=accepted_by_metadata_identity",
                recorded_parent.display(),
                expected_parent.display()
            );
        }
        let reverse_model_path = if reverse_checkpoint.is_dir() {
            reverse_checkpoint.join("model.safetensors")
        } else {
            reverse_checkpoint.clone()
        };
        let reverse_varmap = VarMap::new();
        let reverse_vb = VarBuilder::from_varmap(&reverse_varmap, DType::F32, &device)
            .pp(FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313);
        let reverse_model = PeptideSpectrumCausalModel::new(config.clone(), reverse_vb)?;
        load_matching_variables(&reverse_varmap, &reverse_model_path, &device).with_context(
            || format!("failed to load reverse causal variables {reverse_model_path:?}"),
        )?;
        Some(CausalReranker {
            _varmap: reverse_varmap,
            model: reverse_model,
            collator: FoundationCausalCollator::new(config.clone())?,
        })
    } else {
        None
    };

    let iterative_refiner = if let Some(refinement_checkpoint) =
        iterative_refinement_checkpoint.as_ref()
    {
        let refinement_metadata_path = if refinement_checkpoint.is_dir() {
            refinement_checkpoint.join("metadata.yaml")
        } else {
            refinement_checkpoint
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("metadata.yaml")
        };
        let refinement_metadata: IterativeRefinementCheckpointMetadata = serde_yaml::from_str(
            &fs::read_to_string(&refinement_metadata_path)
                .with_context(|| format!("failed to read {refinement_metadata_path:?}"))?,
        )?;
        if refinement_metadata.objective != FOUNDATION_ITERATIVE_REFINEMENT_OBJECTIVE_V01315
            || refinement_metadata.parameter_namespace
                != FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315
        {
            anyhow::bail!(
                "iterative-refinement checkpoint has incompatible objective/namespace: objective={:?} namespace={:?}",
                refinement_metadata.objective,
                refinement_metadata.parameter_namespace
            );
        }
        if refinement_metadata.inverse_config != config {
            anyhow::bail!(
                "iterative-refinement checkpoint inverse config does not match unified checkpoint"
            );
        }
        if (refinement_metadata.mask_fraction
            - FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315)
            .abs()
            > f64::EPSILON
            || refinement_metadata.refinement_rounds
                != FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315
            || refinement_metadata.seed_hypotheses
                != FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315
            || refinement_metadata.replacement_topk
                != FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315
            || refinement_metadata.replacement_beam_width
                != FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315
        {
            anyhow::bail!(
                "iterative-refinement checkpoint policy does not match fixed v0.13.15 constants"
            );
        }
        let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
        let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
        if refinement_metadata.corpus_fingerprint != corpus_fingerprint
            || refinement_metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        {
            anyhow::bail!("iterative-refinement checkpoint corpus/benchmark fingerprint mismatch");
        }
        let expected_parent = checkpoint_dir.join("model.safetensors");
        if PathBuf::from(&refinement_metadata.parent_unified_checkpoint) != expected_parent {
            anyhow::bail!(
                "iterative-refinement checkpoint parent {:?} does not match evaluated unified parent {:?}",
                refinement_metadata.parent_unified_checkpoint,
                expected_parent
            );
        }
        let refinement_model_path = if refinement_checkpoint.is_dir() {
            refinement_checkpoint.join("model.safetensors")
        } else {
            refinement_checkpoint.clone()
        };
        let refinement_varmap = VarMap::new();
        let refinement_vb = VarBuilder::from_varmap(&refinement_varmap, DType::F32, &device)
            .pp(FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315);
        let refinement_model = PeptideSpectrumDiffusionModel::new(config.clone(), refinement_vb)?;
        load_matching_variables(&refinement_varmap, &refinement_model_path, &device).with_context(
            || format!("failed to load iterative-refinement variables {refinement_model_path:?}"),
        )?;
        Some(IterativeRefiner {
            _varmap: refinement_varmap,
            model: refinement_model,
            collator: FoundationDiffusionCollator::new(config.clone())?,
        })
    } else {
        None
    };

    let cleavage_graph_proposer = if let Some(graph_checkpoint) = cleavage_graph_checkpoint.as_ref()
    {
        let graph_metadata_path = if graph_checkpoint.is_dir() {
            graph_checkpoint.join("metadata.yaml")
        } else {
            graph_checkpoint
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("metadata.yaml")
        };
        let graph_metadata: CleavageGraphCheckpointMetadata = serde_yaml::from_str(
            &fs::read_to_string(&graph_metadata_path)
                .with_context(|| format!("failed to read {graph_metadata_path:?}"))?,
        )?;
        if graph_metadata.objective != FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_OBJECTIVE_V01318
            || graph_metadata.parameter_namespace
                != FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_NAMESPACE_V01318
        {
            anyhow::bail!(
                "cleavage-graph checkpoint has incompatible objective/namespace: objective={:?} namespace={:?}",
                graph_metadata.objective,
                graph_metadata.parameter_namespace
            );
        }
        if graph_metadata.inverse_config != config {
            anyhow::bail!(
                "cleavage-graph checkpoint inverse config does not match unified checkpoint"
            );
        }
        if (graph_metadata.graph_mass_tolerance_da
            - FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316)
            .abs()
            > f64::EPSILON
            || graph_metadata.maximum_outgoing_edges
                != FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316
            || graph_metadata.k_best_graph_paths != FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316
            || graph_metadata.maximum_fragment_charge
                != FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316
            || graph_metadata.maximum_anchor_bridge_edges
                != FOUNDATION_CLEAVAGE_GRAPH_MAX_ANCHOR_BRIDGE_EDGES_V01317
            || graph_metadata.feature_dim != FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318
            || graph_metadata.hidden_dim != FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318
        {
            anyhow::bail!(
                "cleavage-graph checkpoint policy does not match fixed v0.13.18 constants"
            );
        }
        let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
        let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
        if graph_metadata.corpus_fingerprint != corpus_fingerprint
            || graph_metadata.benchmark_manifest_fingerprint != benchmark_fingerprint
        {
            anyhow::bail!("cleavage-graph checkpoint corpus/benchmark fingerprint mismatch");
        }
        let expected_parent = checkpoint_dir.join("model.safetensors");
        if PathBuf::from(&graph_metadata.parent_unified_checkpoint) != expected_parent {
            anyhow::bail!(
                "cleavage-graph checkpoint parent {:?} does not match evaluated unified parent {:?}",
                graph_metadata.parent_unified_checkpoint,
                expected_parent
            );
        }
        let expected_reverse = reverse_causal_checkpoint
            .as_ref()
            .map(|path| {
                if path.is_dir() {
                    path.join("model.safetensors")
                } else {
                    path.clone()
                }
            })
            .ok_or_else(|| {
                anyhow::anyhow!("cleavage-graph evaluation requires reverse checkpoint")
            })?;
        if PathBuf::from(&graph_metadata.reverse_causal_checkpoint) != expected_reverse {
            anyhow::bail!(
                "cleavage-graph checkpoint reverse parent {:?} does not match evaluated reverse {:?}",
                graph_metadata.reverse_causal_checkpoint,
                expected_reverse
            );
        }
        let graph_model_path = if graph_checkpoint.is_dir() {
            graph_checkpoint.join("model.safetensors")
        } else {
            graph_checkpoint.clone()
        };
        let graph_varmap = VarMap::new();
        let graph_vb = VarBuilder::from_varmap(&graph_varmap, DType::F32, &device);
        let graph_model = PeptideSpectrumCleavageGraphStructuredScorer::new(graph_vb)?;
        load_matching_variables(&graph_varmap, &graph_model_path, &device).with_context(|| {
            format!("failed to load cleavage-graph variables {graph_model_path:?}")
        })?;
        Some(CleavageGraphProposer {
            _varmap: graph_varmap,
            model: graph_model,
        })
    } else {
        None
    };

    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!(
        "benchmark_manifest_fingerprint\tfnv1a64:{:016x}",
        benchmark.manifest_fingerprint()
    );
    println!("checkpoint\t{}", checkpoint_dir.display());
    println!(
        "generation_partition\t{}",
        if generation_partition_train {
            "TRAIN"
        } else {
            "VALIDATION"
        }
    );
    println!("validation_records\t{}", selected.len());
    println!("test_partition_consumed\tNO");
    if generation_partition_train {
        println!("v0140_candidate_export_role\tTRAIN_supervision_candidates");
        println!("v0140_candidate_export_target_usage\tpost_generation_labels_only");
        println!("v0140_candidate_export_backbone\tfrozen_v01323");
    }
    println!("samples_per_record\t{samples_per_record}");
    println!("diffusion_steps\t{}", config.diffusion_steps);
    println!("mass_tolerance_da\t{mass_tolerance_da}");
    println!("temperature\t{temperature}");
    println!("mass_beam_width\t{mass_beam_width}");
    println!("final_candidates_per_chain\t{final_candidates_per_chain}");
    println!("fragment_tolerance_ppm\t{fragment_tolerance_ppm}");
    println!("spectral_beam_weight\t{spectral_beam_weight}");
    println!("neural_rerank_weight\t{neural_rerank_weight}");
    println!("causal_checkpoint\t{}", checkpoint_dir.display());
    println!("causal_rerank_weight\t{causal_rerank_weight}");
    println!("causal_rerank_score_definition\tfragment_score+weight*ar_total_log_probability");
    let causal_rerank_policy =
        if (causal_rerank_weight - FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123).abs() <= f64::EPSILON {
            FOUNDATION_CAUSAL_RERANK_POLICY_V0123
        } else {
            "custom_fragment_plus_weighted_ar_total"
        };
    println!("causal_rerank_policy\t{causal_rerank_policy}");
    if reverse_causal_reranker.is_some()
        && iterative_refiner.is_none()
        && cleavage_graph_proposer.is_none()
        && !bidirectional_mitm
    {
        println!("bidirectional_causal_rerank_policy\tfragment_plus_0.05_n_to_c_ar_total_plus_0.05_c_to_n_ar_total_v01314");
        println!(
            "bidirectional_causal_rerank_direction_weight\t{}",
            0.5 * causal_rerank_weight
        );
        println!(
            "bidirectional_causal_rerank_normalization\tarithmetic_mean_total_log_probability"
        );
    }
    println!("causal_generation_beam_width\t{causal_generation_beam_width}");
    println!("causal_generation_final_candidates\t{causal_generation_final_candidates}");
    println!(
        "reverse_causal_checkpoint\t{}",
        reverse_causal_checkpoint
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "disabled".into())
    );
    println!("reverse_causal_generation_beam_width\t{reverse_causal_generation_beam_width}");
    println!(
        "reverse_causal_generation_final_candidates\t{reverse_causal_generation_final_candidates}"
    );
    println!(
        "bidirectional_mitm_policy\t{}",
        if bidirectional_mitm_final_two_view {
            FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01323
        } else if bidirectional_mitm_component_audit {
            FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01322
        } else if bidirectional_mitm_precap_audit {
            FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01321
        } else if bidirectional_mitm_evidence_aware {
            FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01320
        } else if bidirectional_mitm {
            FOUNDATION_BIDIRECTIONAL_MITM_POLICY_V01319
        } else {
            "disabled"
        }
    );
    if bidirectional_mitm {
        println!("bidirectional_mitm_midpoint_fraction\t0.5");
        println!("bidirectional_mitm_mass_definition\tprecursor_neutral_mass_minus_water");
        println!("bidirectional_mitm_prefix_beam_width\t{causal_generation_beam_width}");
        println!("bidirectional_mitm_suffix_beam_width\t{reverse_causal_generation_beam_width}");
        println!(
            "bidirectional_mitm_join_candidate_cap\t{}",
            causal_generation_final_candidates
                .saturating_mul(reverse_causal_generation_final_candidates)
        );
        println!("bidirectional_mitm_reverse_ar_usage\tproposal_only");
        println!(
            "bidirectional_mitm_join_selector\t{}",
            if bidirectional_mitm_final_two_view {
                "128_v01320_evidence+128_join_seam_fragment_with_v01320_backfill_v01323"
            } else if bidirectional_mitm_evidence_aware {
                "full_peptide_fragment+0.1*mean_bidirectional_partial_ar_v01320"
            } else {
                "sum_partial_fragment_plus_0.1*directional_ar_total_v01319"
            }
        );
        if bidirectional_mitm_evidence_aware {
            println!("bidirectional_mitm_selector_full_fragment_evidence\texact_partial_plus_join_seam_assembly");
            println!("bidirectional_mitm_selector_ar_normalization\tmean_log_probability_per_emitted_partial_token");
            println!(
                "bidirectional_mitm_selector_candidate_budget\t{}",
                causal_generation_final_candidates
                    .saturating_mul(reverse_causal_generation_final_candidates)
            );
            println!("bidirectional_mitm_selector_legacy_shadow\tv01319_top256_same_join_pool");
            if bidirectional_mitm_final_two_view {
                println!("bidirectional_mitm_selector_v01320_shadow\tv01320_top256_same_join_pool");
                println!("bidirectional_mitm_selector_view_quota_v01320_evidence\t128");
                println!("bidirectional_mitm_selector_view_quota_join_seam_fragment\t128");
                println!("bidirectional_mitm_selector_view_dedup\tcanonical_candidate_identity");
                println!("bidirectional_mitm_selector_backfill\tremaining_v01320_evidence_order_until_budget_256");
                println!("bidirectional_mitm_selector_stop_rule\tclose_mitm_selector_lane_after_this_single_fixed_run_regardless_of_outcome");
            }
            if bidirectional_mitm_precap_audit {
                println!("bidirectional_mitm_precap_audit\tfull_mass_compatible_join_pool_target_evaluation_only_v01321");
                println!("bidirectional_mitm_precap_audit_candidate_effect\tNONE");
                println!("bidirectional_mitm_precap_audit_target_identity_usage\tpost_generation_metrics_only");
                if bidirectional_mitm_component_audit {
                    println!("bidirectional_mitm_component_audit\tterminal_existing_signal_rank_and_pareto_audit_v01322");
                    println!("bidirectional_mitm_component_audit_candidate_effect\tNONE");
                    println!("bidirectional_mitm_component_audit_pareto_axes\tfragment,prefix_ar_mean,suffix_ar_mean,abs_mass_error");
                    println!("bidirectional_mitm_component_audit_component_reduction\tbest_observed_value_per_complete_canonical_candidate");
                    println!("bidirectional_mitm_component_audit_provenance\tv01320_selector_representative_join_plus_best_directional_partial_ranks");
                    println!("bidirectional_mitm_component_audit_partial_rank_definition\tdirectional_priority_rank_within_midpoint_frontier");
                    println!("bidirectional_mitm_component_audit_stop_rule\tallow_at_most_one_v01323_if_every_incremental_deep_il_target_is_component_top256_or_on_pareto_frontier_le256_else_close_lane");
                }
            }
        }
        println!(
            "bidirectional_mitm_final_ranking\tfragment_score+0.1*n_to_c_ar_total_log_probability"
        );
    }
    println!(
        "iterative_refinement_checkpoint\t{}",
        iterative_refinement_checkpoint
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "disabled".into())
    );
    println!(
        "iterative_refinement_policy\t{}",
        if iterative_refiner.is_some() {
            "four_round_quarter_residue_joint_mass_refill_v01315"
        } else {
            "disabled"
        }
    );
    if iterative_refiner.is_some() {
        println!(
            "iterative_refinement_rounds\t{}",
            FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315
        );
        println!(
            "iterative_refinement_mask_fraction\t{}",
            FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315
        );
        println!(
            "iterative_refinement_seed_hypotheses\t{}",
            FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315
        );
        println!(
            "iterative_refinement_replacement_topk\t{}",
            FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315
        );
        println!(
            "iterative_refinement_replacement_beam_width\t{}",
            FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315
        );
    }
    println!(
        "cleavage_graph_checkpoint\t{}",
        cleavage_graph_checkpoint
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "disabled".into())
    );
    println!(
        "cleavage_graph_policy\t{}",
        if cleavage_graph_proposer.is_some() {
            "anchor_bridge_contextual_global_path_nll_kbest_v01318"
        } else {
            "disabled"
        }
    );
    if cleavage_graph_proposer.is_some() {
        println!(
            "cleavage_graph_mass_tolerance_da\t{}",
            FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
        );
        println!(
            "cleavage_graph_maximum_outgoing_edges\t{}",
            FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316
        );
        println!(
            "cleavage_graph_k_best_paths\t{}",
            FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316
        );
        println!(
            "cleavage_graph_maximum_fragment_charge\t{}",
            FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316
        );
        println!(
            "cleavage_graph_maximum_anchor_bridge_edges\t{}",
            FOUNDATION_CLEAVAGE_GRAPH_MAX_ANCHOR_BRIDGE_EDGES_V01317
        );
        println!("cleavage_graph_bridge_node_spectrum_support\tzero_direct_support");
        println!(
            "cleavage_graph_structured_feature_dim\t{}",
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318
        );
        println!(
            "cleavage_graph_structured_hidden_dim\t{}",
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318
        );
        println!(
            "cleavage_graph_structured_objective\texact_global_source_to_sink_path_nll_v01318"
        );
        println!(
            "cleavage_graph_true_path_diagnostic\tpre_neural_scoring+structured_rank_per_record"
        );
        println!(
            "cleavage_graph_ranking_policy\tfragment_score+0.1*n_to_c_ar_total_log_probability"
        );
    }
    println!(
        "reverse_causal_generation_policy\t{}",
        if reverse_causal_reranker.is_some() && reverse_causal_generation_beam_width > 0 {
            "c_to_n_prefix_fragment_plus_weighted_ar_total_mass_constrained_v01313"
        } else {
            "disabled"
        }
    );
    println!(
        "causal_generation_policy\t{}",
        if causal_generation_beam_width > 0 {
            "prefix_fragment_plus_weighted_ar_total_mass_constrained_v1"
        } else {
            "disabled"
        }
    );
    println!(
        "causal_generation_context_cache\t{}",
        if causal_generation_beam_width > 0 {
            "spectrum_encoder+precursor_once_per_record_v0125"
        } else {
            "disabled"
        }
    );
    println!(
        "causal_generation_prefix_execution\t{}",
        if causal_generation_beam_width > 0 {
            "compact_active_prefix_last_logits_v0128"
        } else {
            "disabled"
        }
    );
    println!(
        "primary_candidate_ranking\t{}",
        if bidirectional_mitm {
            if bidirectional_mitm_final_two_view {
                "fragment_score+0.1*n_to_c_ar_total_log_probability_v01323_final_selector_same_final_ranking"
            } else if bidirectional_mitm_precap_audit {
                if bidirectional_mitm_component_audit {
                    "fragment_score+0.1*n_to_c_ar_total_log_probability_v01322_terminal_diagnostic_same_as_v01320"
                } else {
                    "fragment_score+0.1*n_to_c_ar_total_log_probability_v01321_diagnostic_same_as_v01320"
                }
            } else if bidirectional_mitm_evidence_aware {
                "fragment_score+0.1*n_to_c_ar_total_log_probability_v01320"
            } else {
                "fragment_score+0.1*n_to_c_ar_total_log_probability_v01319"
            }
        } else if cleavage_graph_proposer.is_some() {
            "fragment_score+0.1*n_to_c_ar_total_log_probability_v01318"
        } else {
            "fragment_mass"
        }
    );
    println!(
        "parallel_candidate_rankings\t{}",
        if bidirectional_mitm {
            "neural_all_mask_mass,hybrid_fragment_neural_mass,causal_ar_mass,hybrid_fragment_causal_mass"
        } else {
            "neural_all_mask_mass,hybrid_fragment_neural_mass,causal_ar_mass,hybrid_fragment_causal_mass,hybrid_fragment_bidirectional_causal_mass"
        }
    );
    println!(
        "candidate_pool_sources\t{}",
        if bidirectional_mitm
            && reverse_causal_reranker.is_some()
            && reverse_causal_generation_beam_width > 0
        {
            if bidirectional_mitm_final_two_view {
                "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313+two_view_bidirectional_mitm_v01323"
            } else if bidirectional_mitm_precap_audit {
                if bidirectional_mitm_component_audit {
                    "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313+evidence_aware_bidirectional_mitm_v01320+terminal_component_audit_v01322"
                } else {
                    "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313+evidence_aware_bidirectional_mitm_v01320+diagnostic_precap_audit_v01321"
                }
            } else if bidirectional_mitm_evidence_aware {
                "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313+evidence_aware_bidirectional_mitm_v01320"
            } else {
                "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313+bidirectional_midpoint_mitm_v01319"
            }
        } else if cleavage_graph_proposer.is_some()
            && reverse_causal_reranker.is_some()
            && reverse_causal_generation_beam_width > 0
        {
            "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313+cleavage_contextual_structured_graph_v01318"
        } else if iterative_refiner.is_some()
            && reverse_causal_reranker.is_some()
            && reverse_causal_generation_beam_width > 0
        {
            "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313+iterative_masked_refinement_v01315"
        } else if reverse_causal_reranker.is_some() && reverse_causal_generation_beam_width > 0 {
            "diffusion_reverse_v0115+n_to_c_causal_v0124+c_to_n_reverse_causal_v01313"
        } else if causal_generation_beam_width > 0 {
            "diffusion_reverse_v0115+causal_prefix_mass_beam_v0124"
        } else {
            "diffusion_reverse_v0115"
        }
    );
    println!("candidate_reranker\tall_mask_x0_v1+causal_next_token_v1");
    println!("accepted_candidate_ranking\tfragment_score+0.1*n_to_c_ar_total_log_probability");
    println!("neural_candidate_input\tspectrum+precursor+length_all_masked");
    println!("causal_candidate_input\tspectrum+precursor+START+shifted_candidate_prefix");
    println!("seed\t{seed}");

    if let Some(parent) = output_tsv.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let file = fs::File::create(&output_tsv)?;
    let mut output = BufWriter::new(file);
    writeln!(
        output,
        "record_index\tsource_id\ttarget_sequence\ttarget_active_tokens\tpredicted_active_tokens\tfragment_mass_rank\tneural_mass_rank\thybrid_mass_rank\tcausal_mass_rank\tfragment_causal_mass_rank\tfragment_bidirectional_causal_mass_rank\tmass_rank\treverse_rank\tcandidate_sequence\tcandidate_modifications\treverse_log_probability\tfragment_score\tmatched_cleavages\tneural_all_mask_log_probability\tneural_all_mask_perplexity\tneural_length_log_probability\thybrid_score\tar_total_log_probability\tar_mean_log_probability\tar_perplexity\tfragment_causal_score\treverse_ar_total_log_probability\treverse_ar_mean_log_probability\treverse_ar_perplexity\tbidirectional_ar_total_log_probability\tfragment_bidirectional_causal_score\tmass_error_da\tmass_valid\tfrom_diffusion\tfrom_causal_beam\tfrom_reverse_causal_beam\tfrom_bidirectional_mitm\tfrom_iterative_refinement\tfrom_cleavage_graph\tpeptidoform_exact\tsequence_exact\til_sequence_exact"
    )?;

    let spectra_path = companion_spectra_path(&output_tsv);
    let mut spectra_output = BufWriter::new(fs::File::create(&spectra_path)?);
    writeln!(
        spectra_output,
        "record_index\tsource_id\tsequence\tpeak_index\tmz\tintensity\tnormalized_intensity"
    )?;

    let mut metrics = GenerationMetrics::default();
    for (selection_index, &record_index) in selected.iter().enumerate() {
        let record = &corpus.records[record_index];
        let target_tokens = vocabulary
            .encode(&record.peptidoform, config.max_tokens)
            .map_err(anyhow::Error::msg)?;
        let target_active_length = target_tokens
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
            .unwrap_or(config.max_tokens);

        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("selected validation record lacks observed spectrum"))?;
        let predicted_length_distribution = predict_length_distribution(
            &model,
            &diffusion_collator,
            &spectrum_collator,
            &config,
            record,
            &spectrum,
            &device,
        )?;
        let predicted_active_length =
            (argmax_f64(&predicted_length_distribution) + 1).clamp(2, config.max_tokens);
        metrics.records += 1;
        if predicted_active_length == target_active_length {
            metrics.predicted_length_exact += 1;
        }
        metrics.predicted_length_abs_error +=
            predicted_active_length.abs_diff(target_active_length);

        let mut rng =
            GenerationRng::new(seed ^ mix64(record_index as u64) ^ selection_index as u64);
        let target_neutral_mass = precursor_neutral_mass(record)?;
        let fragment_charge = record
            .context
            .charge
            .unwrap_or(1)
            .unsigned_abs()
            .clamp(1, 2) as usize;
        let observed_peaks = normalized_observed_peaks(&spectrum);
        let spectrum_max_intensity = spectrum
            .peaks
            .iter()
            .filter_map(|peak| {
                (peak.intensity.is_finite() && peak.intensity > 0.0).then_some(peak.intensity)
            })
            .fold(0.0f32, f32::max)
            .max(f32::EPSILON);
        for (peak_index, raw_peak) in spectrum.peaks.iter().enumerate() {
            if !(raw_peak.mz.is_finite()
                && raw_peak.mz > 0.0
                && raw_peak.intensity.is_finite()
                && raw_peak.intensity > 0.0)
            {
                continue;
            }
            writeln!(
                spectra_output,
                "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}",
                record_index,
                corpus.provenance[record_index].source_id,
                record.peptidoform.sequence,
                peak_index,
                raw_peak.mz,
                raw_peak.intensity,
                raw_peak.intensity / spectrum_max_intensity,
            )?;
        }
        let target_fragment_evidence = peptidoform_fragment_evidence(
            &record.peptidoform,
            target_neutral_mass,
            &observed_peaks,
            fragment_charge,
            fragment_tolerance_ppm,
        );
        metrics.target_fragment_score_sum += target_fragment_evidence.score;
        metrics.target_matched_cleavages += target_fragment_evidence.matched_cleavages;
        let active_lengths = sample_generation_lengths(
            &predicted_length_distribution,
            predicted_active_length,
            samples_per_record,
            config.max_tokens,
            target_neutral_mass,
            &mut rng,
        );
        metrics.chains += active_lengths.len();
        let (rows, reverse_scores, fragment_scores, matched_cleavages) = reverse_generate(
            &model,
            &diffusion_collator,
            &spectrum_collator,
            &config,
            record,
            &spectrum,
            &active_lengths,
            target_neutral_mass,
            mass_tolerance_da,
            mass_beam_width,
            final_candidates_per_chain,
            fragment_tolerance_ppm,
            spectral_beam_weight,
            temperature,
            &mut rng,
            &device,
        )?;
        metrics.final_states += rows.len();

        let mut unique = HashMap::<Vec<u32>, GeneratedCandidate>::new();
        for (((tokens, reverse_log_probability), fragment_score), matched_cleavages) in rows
            .into_iter()
            .zip(reverse_scores)
            .zip(fragment_scores)
            .zip(matched_cleavages)
        {
            let peptide = match vocabulary.decode(&tokens) {
                Ok(peptide) => peptide,
                Err(_) => continue,
            };
            metrics.valid_decodes += 1;
            let mass_error_da = precursor_mass_error(record, &peptide)?;
            let mass_valid = mass_error_da
                .map(|error| error.abs() <= mass_tolerance_da)
                .unwrap_or(false);
            if mass_valid {
                metrics.mass_valid_candidates += 1;
            }
            let candidate = GeneratedCandidate {
                tokens: tokens.clone(),
                peptide,
                reverse_log_probability,
                fragment_score,
                matched_cleavages,
                neural_all_mask_log_probability: f64::NEG_INFINITY,
                neural_length_log_probability: f64::NEG_INFINITY,
                hybrid_score: f64::NEG_INFINITY,
                ar_total_log_probability: f64::NEG_INFINITY,
                ar_mean_log_probability: f64::NEG_INFINITY,
                ar_perplexity: f64::INFINITY,
                fragment_causal_score: f64::NEG_INFINITY,
                reverse_ar_total_log_probability: f64::NEG_INFINITY,
                reverse_ar_mean_log_probability: f64::NEG_INFINITY,
                reverse_ar_perplexity: f64::INFINITY,
                bidirectional_ar_total_log_probability: f64::NEG_INFINITY,
                fragment_bidirectional_causal_score: f64::NEG_INFINITY,
                mass_error_da,
                mass_valid,
                from_diffusion: true,
                from_causal_beam: false,
                from_reverse_causal_beam: false,
                from_bidirectional_mitm: false,
                from_iterative_refinement: false,
                from_cleavage_graph: false,
            };
            unique
                .entry(tokens)
                .and_modify(|existing| {
                    existing.from_diffusion = true;
                    if candidate.reverse_log_probability > existing.reverse_log_probability {
                        let from_causal_beam = existing.from_causal_beam;
                        let from_reverse_causal_beam = existing.from_reverse_causal_beam;
                        let from_bidirectional_mitm = existing.from_bidirectional_mitm;
                        let from_iterative_refinement = existing.from_iterative_refinement;
                        let from_cleavage_graph = existing.from_cleavage_graph;
                        *existing = candidate.clone();
                        existing.from_causal_beam = from_causal_beam;
                        existing.from_reverse_causal_beam = from_reverse_causal_beam;
                        existing.from_bidirectional_mitm = from_bidirectional_mitm;
                        existing.from_iterative_refinement = from_iterative_refinement;
                        existing.from_cleavage_graph = from_cleavage_graph;
                    }
                })
                .or_insert(candidate);
        }

        if causal_generation_beam_width > 0 {
            if let Some(causal) = causal_reranker.as_ref() {
                let causal_generated = causal_prefix_mass_beam(
                    causal,
                    &spectrum_collator,
                    &config,
                    record,
                    &spectrum,
                    target_neutral_mass,
                    mass_tolerance_da,
                    causal_generation_beam_width,
                    causal_generation_final_candidates,
                    &observed_peaks,
                    fragment_charge,
                    fragment_tolerance_ppm,
                    causal_rerank_weight,
                    &device,
                )?;
                metrics.causal_beam_final_candidates += causal_generated.len();
                if !causal_generated.is_empty() {
                    metrics.causal_beam_records_with_candidate += 1;
                }
                for generated in causal_generated {
                    let peptide = match vocabulary.decode(&generated.tokens) {
                        Ok(peptide) => peptide,
                        Err(_) => continue,
                    };
                    let mass_error_da = precursor_mass_error(record, &peptide)?;
                    let mass_valid = mass_error_da
                        .map(|error| error.abs() <= mass_tolerance_da)
                        .unwrap_or(false);
                    let active_length = active_token_length(&generated.tokens, config.max_tokens)?;
                    let ar_mean_log_probability =
                        generated.ar_total_log_probability / active_length as f64;
                    let candidate = GeneratedCandidate {
                        tokens: generated.tokens.clone(),
                        peptide,
                        reverse_log_probability: f64::NEG_INFINITY,
                        fragment_score: generated.fragment_score,
                        matched_cleavages: generated.matched_cleavages,
                        neural_all_mask_log_probability: f64::NEG_INFINITY,
                        neural_length_log_probability: f64::NEG_INFINITY,
                        hybrid_score: f64::NEG_INFINITY,
                        ar_total_log_probability: generated.ar_total_log_probability,
                        ar_mean_log_probability,
                        ar_perplexity: (-ar_mean_log_probability).exp(),
                        fragment_causal_score: generated.fragment_causal_score,
                        reverse_ar_total_log_probability: f64::NEG_INFINITY,
                        reverse_ar_mean_log_probability: f64::NEG_INFINITY,
                        reverse_ar_perplexity: f64::INFINITY,
                        bidirectional_ar_total_log_probability: f64::NEG_INFINITY,
                        fragment_bidirectional_causal_score: f64::NEG_INFINITY,
                        mass_error_da,
                        mass_valid,
                        from_diffusion: false,
                        from_causal_beam: true,
                        from_reverse_causal_beam: false,
                        from_bidirectional_mitm: false,
                        from_iterative_refinement: false,
                        from_cleavage_graph: false,
                    };
                    unique
                        .entry(generated.tokens)
                        .and_modify(|existing| {
                            existing.from_causal_beam = true;
                        })
                        .or_insert(candidate);
                }
            }
        }

        if reverse_causal_generation_beam_width > 0 {
            if let Some(reverse_causal) = reverse_causal_reranker.as_ref() {
                let reverse_generated = reverse_causal_prefix_mass_beam(
                    reverse_causal,
                    &spectrum_collator,
                    &config,
                    record,
                    &spectrum,
                    target_neutral_mass,
                    mass_tolerance_da,
                    reverse_causal_generation_beam_width,
                    reverse_causal_generation_final_candidates,
                    &observed_peaks,
                    fragment_charge,
                    fragment_tolerance_ppm,
                    causal_rerank_weight,
                    &device,
                )?;
                metrics.reverse_causal_beam_final_candidates += reverse_generated.len();
                if !reverse_generated.is_empty() {
                    metrics.reverse_causal_beam_records_with_candidate += 1;
                }
                for generated in reverse_generated {
                    let peptide = match vocabulary.decode(&generated.tokens) {
                        Ok(peptide) => peptide,
                        Err(_) => continue,
                    };
                    let mass_error_da = precursor_mass_error(record, &peptide)?;
                    let mass_valid = mass_error_da
                        .map(|error| error.abs() <= mass_tolerance_da)
                        .unwrap_or(false);
                    let active_length = active_token_length(&generated.tokens, config.max_tokens)?;
                    let ar_mean_log_probability =
                        generated.ar_total_log_probability / active_length as f64;
                    let candidate = GeneratedCandidate {
                        tokens: generated.tokens.clone(),
                        peptide,
                        reverse_log_probability: f64::NEG_INFINITY,
                        fragment_score: generated.fragment_score,
                        matched_cleavages: generated.matched_cleavages,
                        neural_all_mask_log_probability: f64::NEG_INFINITY,
                        neural_length_log_probability: f64::NEG_INFINITY,
                        hybrid_score: f64::NEG_INFINITY,
                        ar_total_log_probability: generated.ar_total_log_probability,
                        ar_mean_log_probability,
                        ar_perplexity: (-ar_mean_log_probability).exp(),
                        fragment_causal_score: generated.fragment_causal_score,
                        reverse_ar_total_log_probability: f64::NEG_INFINITY,
                        reverse_ar_mean_log_probability: f64::NEG_INFINITY,
                        reverse_ar_perplexity: f64::INFINITY,
                        bidirectional_ar_total_log_probability: f64::NEG_INFINITY,
                        fragment_bidirectional_causal_score: f64::NEG_INFINITY,
                        mass_error_da,
                        mass_valid,
                        from_diffusion: false,
                        from_causal_beam: false,
                        from_reverse_causal_beam: true,
                        from_bidirectional_mitm: false,
                        from_iterative_refinement: false,
                        from_cleavage_graph: false,
                    };
                    unique
                        .entry(generated.tokens)
                        .and_modify(|existing| {
                            existing.from_reverse_causal_beam = true;
                        })
                        .or_insert(candidate);
                }
            }
        }

        let mut frozen_extension_candidates = None;
        if bidirectional_mitm || iterative_refiner.is_some() || cleavage_graph_proposer.is_some() {
            // Snapshot and independently rescore the accepted v0.13.13 three-way pool
            // before any extension candidate is inserted. This is the parity boundary
            // for rejected v0.13.15/v0.13.18 branches and the v0.13.19/v0.13.20 MITM extensions.
            let mut frozen_candidates = unique.values().cloned().collect::<Vec<_>>();
            if let Some(causal) = causal_reranker.as_ref() {
                let frozen_scores = score_causal_candidates(
                    &causal.model,
                    &causal.collator,
                    &spectrum_collator,
                    record,
                    &spectrum,
                    &frozen_candidates,
                    &device,
                )?;
                for (candidate, score) in frozen_candidates.iter_mut().zip(frozen_scores) {
                    candidate.ar_total_log_probability = score.total_log_probability;
                    candidate.ar_mean_log_probability = score.mean_log_probability;
                    candidate.ar_perplexity = score.perplexity;
                    candidate.fragment_causal_score = foundation_fragment_causal_rerank_score(
                        candidate.fragment_score,
                        candidate.ar_total_log_probability,
                        causal_rerank_weight,
                    );
                }
            }

            let frozen_target_il = normalize_il(&record.peptidoform.sequence);
            let frozen_mass_valid = frozen_candidates
                .iter()
                .filter(|candidate| candidate.mass_valid)
                .collect::<Vec<_>>();
            metrics.frozen_v01313_pool_mass_valid_peptidoform_exact += usize::from(
                frozen_mass_valid
                    .iter()
                    .any(|candidate| candidate.peptide == record.peptidoform),
            );
            metrics.frozen_v01313_pool_mass_valid_sequence_exact +=
                usize::from(frozen_mass_valid.iter().any(|candidate| {
                    candidate.peptide.sequence.as_str() == record.peptidoform.sequence.as_str()
                }));
            metrics.frozen_v01313_pool_mass_valid_il_sequence_exact +=
                usize::from(frozen_mass_valid.iter().any(|candidate| {
                    normalize_il(&candidate.peptide.sequence) == frozen_target_il
                }));

            let mut frozen_ranked = frozen_candidates.clone();
            frozen_ranked.sort_by(fragment_causal_mass_candidate_order);
            if let Some(top1) = frozen_ranked.first() {
                let exact = ranking_exact_flags(top1, record, &frozen_target_il);
                metrics.frozen_v01313_forward_ranking_peptidoform_exact += exact.0;
                metrics.frozen_v01313_forward_ranking_sequence_exact += exact.1;
                metrics.frozen_v01313_forward_ranking_il_sequence_exact += exact.2;
            }
            frozen_extension_candidates = Some(frozen_candidates);
        }

        if bidirectional_mitm {
            let causal = causal_reranker
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing accepted N->C causal model for MITM"))?;
            let reverse_causal = reverse_causal_reranker
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing accepted C->N causal model for MITM"))?;
            let prefix_states = causal_midpoint_partial_beam(
                causal,
                &spectrum_collator,
                &config,
                record,
                &spectrum,
                target_neutral_mass,
                mass_tolerance_da,
                causal_generation_beam_width,
                &observed_peaks,
                fragment_charge,
                fragment_tolerance_ppm,
                causal_rerank_weight,
                MitmDirection::NToC,
                &device,
            )?;
            let suffix_states = causal_midpoint_partial_beam(
                reverse_causal,
                &spectrum_collator,
                &config,
                record,
                &spectrum,
                target_neutral_mass,
                mass_tolerance_da,
                reverse_causal_generation_beam_width,
                &observed_peaks,
                fragment_charge,
                fragment_tolerance_ppm,
                causal_rerank_weight,
                MitmDirection::CToN,
                &device,
            )?;
            metrics.mitm_prefix_records_with_states += usize::from(!prefix_states.is_empty());
            metrics.mitm_suffix_records_with_states += usize::from(!suffix_states.is_empty());

            let max_joined_candidates = causal_generation_final_candidates
                .saturating_mul(reverse_causal_generation_final_candidates);
            let (
                unique_mass_joins_before_cap,
                joined,
                legacy_shadow,
                v01320_shadow,
                precap_audit_pool,
            ) = bidirectional_mitm_join(
                &prefix_states,
                &suffix_states,
                target_neutral_mass,
                mass_tolerance_da,
                config.max_tokens,
                vocabulary,
                max_joined_candidates,
                &observed_peaks,
                fragment_charge,
                fragment_tolerance_ppm,
                causal_rerank_weight,
                bidirectional_mitm_evidence_aware,
                bidirectional_mitm_final_two_view,
                bidirectional_mitm_precap_audit,
            )?;
            metrics.mitm_unique_mass_joins_before_cap += unique_mass_joins_before_cap;
            metrics.mitm_records_with_mass_join += usize::from(unique_mass_joins_before_cap > 0);
            metrics.mitm_joined_candidates += joined.len();
            if bidirectional_mitm_evidence_aware {
                metrics.mitm_selector_scored_candidates += unique_mass_joins_before_cap;
                let selected_tokens: HashSet<Vec<u32>> = joined
                    .iter()
                    .map(|candidate| candidate.tokens.clone())
                    .collect();
                let legacy_tokens: HashSet<Vec<u32>> = legacy_shadow
                    .iter()
                    .map(|candidate| candidate.tokens.clone())
                    .collect();
                metrics.mitm_selector_displaced_legacy_candidates +=
                    selected_tokens.difference(&legacy_tokens).count();

                let target_il = normalize_il(&record.peptidoform.sequence);
                let legacy_peptides: Vec<PeptidoformInput> = legacy_shadow
                    .iter()
                    .filter_map(|candidate| vocabulary.decode(&candidate.tokens).ok())
                    .collect();
                metrics.mitm_legacy_pool_mass_valid_peptidoform_exact += usize::from(
                    legacy_peptides
                        .iter()
                        .any(|peptide| peptide == &record.peptidoform),
                );
                metrics.mitm_legacy_pool_mass_valid_sequence_exact +=
                    usize::from(legacy_peptides.iter().any(|peptide| {
                        peptide.sequence.as_str() == record.peptidoform.sequence.as_str()
                    }));
                metrics.mitm_legacy_pool_mass_valid_il_sequence_exact += usize::from(
                    legacy_peptides
                        .iter()
                        .any(|peptide| normalize_il(&peptide.sequence) == target_il),
                );

                if bidirectional_mitm_final_two_view {
                    let v01320_tokens: HashSet<Vec<u32>> = v01320_shadow
                        .iter()
                        .map(|candidate| candidate.tokens.clone())
                        .collect();
                    metrics.mitm_v01323_displaced_v01320_candidates +=
                        selected_tokens.difference(&v01320_tokens).count();

                    let v01320_peptides: Vec<PeptidoformInput> = v01320_shadow
                        .iter()
                        .filter_map(|candidate| vocabulary.decode(&candidate.tokens).ok())
                        .collect();
                    let v01320_literal_present = v01320_peptides
                        .iter()
                        .any(|peptide| peptide == &record.peptidoform);
                    let v01320_sequence_present = v01320_peptides.iter().any(|peptide| {
                        peptide.sequence.as_str() == record.peptidoform.sequence.as_str()
                    });
                    let v01320_il_present = v01320_peptides
                        .iter()
                        .any(|peptide| normalize_il(&peptide.sequence) == target_il);
                    metrics.mitm_v01320_shadow_pool_mass_valid_peptidoform_exact +=
                        usize::from(v01320_literal_present);
                    metrics.mitm_v01320_shadow_pool_mass_valid_sequence_exact +=
                        usize::from(v01320_sequence_present);
                    metrics.mitm_v01320_shadow_pool_mass_valid_il_sequence_exact +=
                        usize::from(v01320_il_present);

                    let frozen_for_union = frozen_extension_candidates.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "missing frozen v0.13.13 pool during v0.13.23 final selector shadow parity"
                        )
                    })?;
                    let frozen_literal_present = frozen_for_union.iter().any(|candidate| {
                        candidate.mass_valid && candidate.peptide == record.peptidoform
                    });
                    let frozen_sequence_present = frozen_for_union.iter().any(|candidate| {
                        candidate.mass_valid
                            && candidate.peptide.sequence.as_str()
                                == record.peptidoform.sequence.as_str()
                    });
                    let frozen_il_present = frozen_for_union.iter().any(|candidate| {
                        candidate.mass_valid
                            && normalize_il(&candidate.peptide.sequence) == target_il
                    });
                    metrics.mitm_v01320_shadow_union_peptidoform_exact +=
                        usize::from(frozen_literal_present || v01320_literal_present);
                    metrics.mitm_v01320_shadow_union_sequence_exact +=
                        usize::from(frozen_sequence_present || v01320_sequence_present);
                    metrics.mitm_v01320_shadow_union_il_sequence_exact +=
                        usize::from(frozen_il_present || v01320_il_present);
                }
            }
            if bidirectional_mitm_precap_audit {
                let audit = mitm_precap_oracle_audit(
                    &precap_audit_pool,
                    &record.peptidoform,
                    config.max_tokens,
                    vocabulary,
                )?;
                metrics.mitm_precap_pool_peptidoform_exact +=
                    usize::from(audit.peptidoform_present);
                metrics.mitm_precap_pool_sequence_exact += usize::from(audit.sequence_present);
                metrics.mitm_precap_pool_il_sequence_exact +=
                    usize::from(audit.il_sequence_present);

                let frozen_for_union = frozen_extension_candidates.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("missing frozen v0.13.13 pool during v0.13.21 audit")
                })?;
                let frozen_literal_present = frozen_for_union.iter().any(|candidate| {
                    candidate.mass_valid && candidate.peptide == record.peptidoform
                });
                let frozen_sequence_present = frozen_for_union.iter().any(|candidate| {
                    candidate.mass_valid
                        && candidate.peptide.sequence.as_str()
                            == record.peptidoform.sequence.as_str()
                });
                let target_il = normalize_il(&record.peptidoform.sequence);
                let frozen_il_present = frozen_for_union.iter().any(|candidate| {
                    candidate.mass_valid && normalize_il(&candidate.peptide.sequence) == target_il
                });
                metrics.mitm_precap_union_peptidoform_exact +=
                    usize::from(frozen_literal_present || audit.peptidoform_present);
                metrics.mitm_precap_union_sequence_exact +=
                    usize::from(frozen_sequence_present || audit.sequence_present);
                metrics.mitm_precap_union_il_sequence_exact +=
                    usize::from(frozen_il_present || audit.il_sequence_present);

                metrics
                    .mitm_precap_legacy_literal_rank_cutoffs
                    .observe(audit.legacy_ranks.peptidoform);
                metrics
                    .mitm_precap_legacy_sequence_rank_cutoffs
                    .observe(audit.legacy_ranks.sequence);
                metrics
                    .mitm_precap_legacy_il_rank_cutoffs
                    .observe(audit.legacy_ranks.il_sequence);
                metrics
                    .mitm_precap_evidence_literal_rank_cutoffs
                    .observe(audit.evidence_ranks.peptidoform);
                metrics
                    .mitm_precap_evidence_sequence_rank_cutoffs
                    .observe(audit.evidence_ranks.sequence);
                metrics
                    .mitm_precap_evidence_il_rank_cutoffs
                    .observe(audit.evidence_ranks.il_sequence);
                println!(
                    "generation_bidirectional_mitm_precap_oracle\trecord_index={record_index}\tprecap_candidates={}\tpeptidoform_present={}\tsequence_present={}\til_present={}\tunion_peptidoform_present={}\tunion_sequence_present={}\tunion_il_present={}\tlegacy_peptidoform_rank={}\tlegacy_sequence_rank={}\tlegacy_il_rank={}\tevidence_peptidoform_rank={}\tevidence_sequence_rank={}\tevidence_il_rank={}",
                    audit.candidates,
                    yes_no(audit.peptidoform_present),
                    yes_no(audit.sequence_present),
                    yes_no(audit.il_sequence_present),
                    yes_no(frozen_literal_present || audit.peptidoform_present),
                    yes_no(frozen_sequence_present || audit.sequence_present),
                    yes_no(frozen_il_present || audit.il_sequence_present),
                    format_mitm_rank(audit.legacy_ranks.peptidoform),
                    format_mitm_rank(audit.legacy_ranks.sequence),
                    format_mitm_rank(audit.legacy_ranks.il_sequence),
                    format_mitm_rank(audit.evidence_ranks.peptidoform),
                    format_mitm_rank(audit.evidence_ranks.sequence),
                    format_mitm_rank(audit.evidence_ranks.il_sequence),
                );
                if bidirectional_mitm_component_audit {
                    let deep_literal = !frozen_literal_present
                        && audit.peptidoform_present
                        && audit
                            .legacy_ranks
                            .peptidoform
                            .map(|rank| rank > 256)
                            .unwrap_or(false)
                        && audit
                            .evidence_ranks
                            .peptidoform
                            .map(|rank| rank > 256)
                            .unwrap_or(false);
                    let deep_il = !frozen_il_present
                        && audit.il_sequence_present
                        && audit
                            .legacy_ranks
                            .il_sequence
                            .map(|rank| rank > 256)
                            .unwrap_or(false)
                        && audit
                            .evidence_ranks
                            .il_sequence
                            .map(|rank| rank > 256)
                            .unwrap_or(false);
                    let component = mitm_component_rank_audit(
                        &precap_audit_pool,
                        &record.peptidoform,
                        config.max_tokens,
                        vocabulary,
                        deep_literal || deep_il,
                    )?;
                    let literal_top256_any =
                        mitm_component_rank_any_top256(&component.ranks, false);
                    let il_top256_any = mitm_component_rank_any_top256(&component.ranks, true);
                    let literal_pareto_le256 = component.pareto_frontier_size <= 256
                        && component.pareto_peptidoform_present;
                    let il_pareto_le256 =
                        component.pareto_frontier_size <= 256 && component.pareto_il_present;
                    if deep_literal {
                        metrics.mitm_component_audit_incremental_deep_literal_records += 1;
                        metrics.mitm_component_audit_deep_literal_top256_any_component +=
                            usize::from(literal_top256_any);
                        metrics.mitm_component_audit_deep_literal_pareto_le256 +=
                            usize::from(literal_pareto_le256);
                        metrics.mitm_component_audit_deep_literal_actionable_records +=
                            usize::from(literal_top256_any || literal_pareto_le256);
                    }
                    if deep_il {
                        metrics.mitm_component_audit_incremental_deep_il_records += 1;
                        metrics.mitm_component_audit_deep_il_top256_any_component +=
                            usize::from(il_top256_any);
                        metrics.mitm_component_audit_deep_il_pareto_le256 +=
                            usize::from(il_pareto_le256);
                        metrics.mitm_component_audit_deep_il_actionable_records +=
                            usize::from(il_top256_any || il_pareto_le256);
                    }
                    let provenance = component.il_provenance;
                    println!(
                        "generation_bidirectional_mitm_component_audit\trecord_index={record_index}\tdeep_incremental_literal={}\tdeep_incremental_il={}\tfragment_literal_rank={}\tfragment_sequence_rank={}\tfragment_il_rank={}\tprefix_mean_literal_rank={}\tprefix_mean_sequence_rank={}\tprefix_mean_il_rank={}\tsuffix_mean_literal_rank={}\tsuffix_mean_sequence_rank={}\tsuffix_mean_il_rank={}\tprefix_total_literal_rank={}\tprefix_total_sequence_rank={}\tprefix_total_il_rank={}\tsuffix_total_literal_rank={}\tsuffix_total_sequence_rank={}\tsuffix_total_il_rank={}\tmass_error_literal_rank={}\tmass_error_sequence_rank={}\tmass_error_il_rank={}\tseam_fragment_literal_rank={}\tseam_fragment_sequence_rank={}\tseam_fragment_il_rank={}\tliteral_top256_any_component={}\til_top256_any_component={}\tpareto_frontier_size={}\tpareto_literal_present={}\tpareto_sequence_present={}\tpareto_il_present={}\tliteral_pareto_le256={}\til_pareto_le256={}\til_provenance_found={}\til_prefix_partial_rank={}\til_suffix_partial_rank={}\til_best_prefix_partial_rank={}\til_best_suffix_partial_rank={}\til_prefix_token_count={}\til_suffix_token_count={}\til_prefix_residue_count={}\til_suffix_residue_count={}\til_join_seam_fragment_score={:.6}\til_fragment_score={:.6}\til_prefix_ar_mean={:.6}\til_suffix_ar_mean={:.6}\til_prefix_ar_total={:.6}\til_suffix_ar_total={:.6}\til_abs_mass_error_da={:.6}",
                        yes_no(deep_literal),
                        yes_no(deep_il),
                        format_mitm_rank(component.ranks.fragment.peptidoform),
                        format_mitm_rank(component.ranks.fragment.sequence),
                        format_mitm_rank(component.ranks.fragment.il_sequence),
                        format_mitm_rank(component.ranks.prefix_ar_mean.peptidoform),
                        format_mitm_rank(component.ranks.prefix_ar_mean.sequence),
                        format_mitm_rank(component.ranks.prefix_ar_mean.il_sequence),
                        format_mitm_rank(component.ranks.suffix_ar_mean.peptidoform),
                        format_mitm_rank(component.ranks.suffix_ar_mean.sequence),
                        format_mitm_rank(component.ranks.suffix_ar_mean.il_sequence),
                        format_mitm_rank(component.ranks.prefix_ar_total.peptidoform),
                        format_mitm_rank(component.ranks.prefix_ar_total.sequence),
                        format_mitm_rank(component.ranks.prefix_ar_total.il_sequence),
                        format_mitm_rank(component.ranks.suffix_ar_total.peptidoform),
                        format_mitm_rank(component.ranks.suffix_ar_total.sequence),
                        format_mitm_rank(component.ranks.suffix_ar_total.il_sequence),
                        format_mitm_rank(component.ranks.mass_error.peptidoform),
                        format_mitm_rank(component.ranks.mass_error.sequence),
                        format_mitm_rank(component.ranks.mass_error.il_sequence),
                        format_mitm_rank(component.ranks.seam_fragment.peptidoform),
                        format_mitm_rank(component.ranks.seam_fragment.sequence),
                        format_mitm_rank(component.ranks.seam_fragment.il_sequence),
                        yes_no(literal_top256_any),
                        yes_no(il_top256_any),
                        component.pareto_frontier_size,
                        yes_no(component.pareto_peptidoform_present),
                        yes_no(component.pareto_sequence_present),
                        yes_no(component.pareto_il_present),
                        yes_no(literal_pareto_le256),
                        yes_no(il_pareto_le256),
                        yes_no(provenance.found),
                        provenance.prefix_partial_rank,
                        provenance.suffix_partial_rank,
                        provenance.best_prefix_partial_rank,
                        provenance.best_suffix_partial_rank,
                        provenance.prefix_token_count,
                        provenance.suffix_token_count,
                        provenance.prefix_residue_count,
                        provenance.suffix_residue_count,
                        provenance.join_seam_fragment_score,
                        provenance.fragment_score,
                        provenance.prefix_ar_mean,
                        provenance.suffix_ar_mean,
                        provenance.prefix_ar_total,
                        provenance.suffix_ar_total,
                        provenance.abs_mass_error_da,
                    );
                }
            }

            println!(
                "generation_bidirectional_mitm_search\trecord_index={record_index}\tprefix_states={}\tsuffix_states={}\tunique_mass_joins_before_cap={}\tretained_joined_candidates={}\tselector={}\tlegacy_shadow_candidates={}\tmidpoint_fraction=0.5",
                prefix_states.len(),
                suffix_states.len(),
                unique_mass_joins_before_cap,
                joined.len(),
                if bidirectional_mitm_final_two_view {
                    "v01323_two_view_128_evidence_128_join_seam_with_evidence_backfill"
                } else if bidirectional_mitm_evidence_aware {
                    "full_fragment_plus_bidirectional_partial_ar_mean_v01320"
                } else {
                    "legacy_partial_priority_v01319"
                },
                legacy_shadow.len(),
            );

            for generated in joined {
                let peptide = match vocabulary.decode(&generated.tokens) {
                    Ok(peptide) => peptide,
                    Err(_) => continue,
                };
                let mass_error_da = precursor_mass_error(record, &peptide)?;
                let mass_valid = mass_error_da
                    .map(|error| error.abs() <= mass_tolerance_da)
                    .unwrap_or(false);
                if !mass_valid {
                    continue;
                }
                let evidence = peptidoform_fragment_evidence(
                    &peptide,
                    target_neutral_mass,
                    &observed_peaks,
                    fragment_charge,
                    fragment_tolerance_ppm,
                );
                let candidate = GeneratedCandidate {
                    tokens: generated.tokens.clone(),
                    peptide,
                    reverse_log_probability: f64::NEG_INFINITY,
                    fragment_score: evidence.score,
                    matched_cleavages: evidence.matched_cleavages,
                    neural_all_mask_log_probability: f64::NEG_INFINITY,
                    neural_length_log_probability: f64::NEG_INFINITY,
                    hybrid_score: f64::NEG_INFINITY,
                    ar_total_log_probability: f64::NEG_INFINITY,
                    ar_mean_log_probability: f64::NEG_INFINITY,
                    ar_perplexity: f64::INFINITY,
                    fragment_causal_score: f64::NEG_INFINITY,
                    reverse_ar_total_log_probability: f64::NEG_INFINITY,
                    reverse_ar_mean_log_probability: f64::NEG_INFINITY,
                    reverse_ar_perplexity: f64::INFINITY,
                    bidirectional_ar_total_log_probability: f64::NEG_INFINITY,
                    fragment_bidirectional_causal_score: f64::NEG_INFINITY,
                    mass_error_da,
                    mass_valid,
                    from_diffusion: false,
                    from_causal_beam: false,
                    from_reverse_causal_beam: false,
                    from_bidirectional_mitm: true,
                    from_iterative_refinement: false,
                    from_cleavage_graph: false,
                };
                unique
                    .entry(generated.tokens)
                    .and_modify(|existing| {
                        existing.from_bidirectional_mitm = true;
                    })
                    .or_insert(candidate);
            }
        }

        if let Some(refiner) = iterative_refiner.as_ref() {
            let frozen_candidates = frozen_extension_candidates
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing frozen v0.13.13 refinement snapshot"))?;
            let seeds = select_iterative_refinement_seeds(frozen_candidates);
            let refined_rows = iterative_refine_candidates(
                refiner,
                &spectrum_collator,
                &config,
                record,
                &spectrum,
                &seeds,
                target_neutral_mass,
                mass_tolerance_da,
                &device,
            )?;
            metrics.iterative_refinement_final_candidates += refined_rows.len();
            if !refined_rows.is_empty() {
                metrics.iterative_refinement_records_with_candidate += 1;
            }
            for tokens in refined_rows {
                let peptide = match vocabulary.decode(&tokens) {
                    Ok(peptide) => peptide,
                    Err(_) => continue,
                };
                let mass_error_da = precursor_mass_error(record, &peptide)?;
                let mass_valid = mass_error_da
                    .map(|error| error.abs() <= mass_tolerance_da)
                    .unwrap_or(false);
                if !mass_valid {
                    continue;
                }
                let evidence = peptidoform_fragment_evidence(
                    &peptide,
                    target_neutral_mass,
                    &observed_peaks,
                    fragment_charge,
                    fragment_tolerance_ppm,
                );
                let candidate = GeneratedCandidate {
                    tokens: tokens.clone(),
                    peptide,
                    reverse_log_probability: f64::NEG_INFINITY,
                    fragment_score: evidence.score,
                    matched_cleavages: evidence.matched_cleavages,
                    neural_all_mask_log_probability: f64::NEG_INFINITY,
                    neural_length_log_probability: f64::NEG_INFINITY,
                    hybrid_score: f64::NEG_INFINITY,
                    ar_total_log_probability: f64::NEG_INFINITY,
                    ar_mean_log_probability: f64::NEG_INFINITY,
                    ar_perplexity: f64::INFINITY,
                    fragment_causal_score: f64::NEG_INFINITY,
                    reverse_ar_total_log_probability: f64::NEG_INFINITY,
                    reverse_ar_mean_log_probability: f64::NEG_INFINITY,
                    reverse_ar_perplexity: f64::INFINITY,
                    bidirectional_ar_total_log_probability: f64::NEG_INFINITY,
                    fragment_bidirectional_causal_score: f64::NEG_INFINITY,
                    mass_error_da,
                    mass_valid,
                    from_diffusion: false,
                    from_causal_beam: false,
                    from_reverse_causal_beam: false,
                    from_bidirectional_mitm: false,
                    from_iterative_refinement: true,
                    from_cleavage_graph: false,
                };
                unique
                    .entry(tokens)
                    .and_modify(|existing| {
                        existing.from_iterative_refinement = true;
                    })
                    .or_insert(candidate);
            }
        }

        if let Some(graph_proposer) = cleavage_graph_proposer.as_ref() {
            // Construction and true-path audit are deliberately emitted before any
            // v0.13.18 neural edge-energy scoring.
            metrics.cleavage_graph_structural_records += 1;
            let expected_edges = record.peptidoform.sequence.chars().count();
            metrics.cleavage_graph_true_nodes_total += expected_edges + 1;
            metrics.cleavage_graph_true_edges_total += expected_edges;

            let graph = match foundation_build_cleavage_graph(record, &spectrum) {
                Ok(Some(graph)) => Some(graph),
                Ok(None) => {
                    println!(
                        "cleavage_graph_structural\trecord_index={record_index}\ttrue_path_structurally_present=NO\tpresent_nodes=0\ttotal_nodes={}\tpresent_edges=0\ttotal_edges={}\treason=graph_unavailable",
                        expected_edges + 1,
                        expected_edges
                    );
                    None
                }
                Err(error) => {
                    println!(
                        "cleavage_graph_structural\trecord_index={record_index}\ttrue_path_structurally_present=NO\tpresent_nodes=0\ttotal_nodes={}\tpresent_edges=0\ttotal_edges={}\treason=graph_construction_error:{}",
                        expected_edges + 1,
                        expected_edges,
                        sanitize_diagnostic_text(&error)
                    );
                    None
                }
            };

            if let Some(graph) = graph {
                let audit = match foundation_cleavage_graph_true_path_audit(
                    &graph,
                    &record.peptidoform,
                ) {
                    Ok(audit) => {
                        metrics.cleavage_graph_true_path_structural_present +=
                            usize::from(audit.structurally_present);
                        metrics.cleavage_graph_true_nodes_present += audit.present_nodes;
                        metrics.cleavage_graph_true_edges_present += audit.present_edges;
                        println!(
                            "cleavage_graph_structural\trecord_index={record_index}\ttrue_path_structurally_present={}\tpresent_nodes={}\ttotal_nodes={}\tpresent_edges={}\ttotal_edges={}\tgraph_nodes={}\tgraph_edges={}",
                            if audit.structurally_present { "YES" } else { "NO" },
                            audit.present_nodes,
                            audit.total_nodes,
                            audit.present_edges,
                            audit.total_edges,
                            graph.nodes.len(),
                            graph.edge_count()
                        );
                        Some(audit)
                    }
                    Err(error) => {
                        println!(
                            "cleavage_graph_structural\trecord_index={record_index}\ttrue_path_structurally_present=NO\tpresent_nodes=0\ttotal_nodes={}\tpresent_edges=0\ttotal_edges={}\tgraph_nodes={}\tgraph_edges={}\treason=true_path_audit_error:{}",
                            expected_edges + 1,
                            expected_edges,
                            graph.nodes.len(),
                            graph.edge_count(),
                            sanitize_diagnostic_text(&error)
                        );
                        None
                    }
                };

                // Contextual globally structured scoring starts only after the
                // target-independent structural diagnostic above has been emitted.
                let decode = foundation_cleavage_graph_structured_decode(
                    &graph_proposer.model,
                    &graph,
                    audit.as_ref(),
                    &device,
                )?;
                if let Some(audit) = audit.as_ref().filter(|audit| audit.structurally_present) {
                    let literal_rank = decode
                        .candidates
                        .iter()
                        .position(|candidate| candidate.peptide == record.peptidoform)
                        .map(|position| position + 1);
                    metrics.cleavage_graph_structured_true_path_top1 +=
                        usize::from(literal_rank.is_some_and(|rank| rank <= 1));
                    metrics.cleavage_graph_structured_true_path_top8 +=
                        usize::from(literal_rank.is_some_and(|rank| rank <= 8));
                    metrics.cleavage_graph_structured_true_path_top32 +=
                        usize::from(literal_rank.is_some_and(|rank| rank <= 32));
                    metrics.cleavage_graph_structured_true_path_top64 +=
                        usize::from(literal_rank.is_some_and(|rank| rank <= 64));
                    println!(
                        "cleavage_graph_structured_rank\trecord_index={record_index}\ttrue_path_score={}\tbest_decoded_path_score={}\tlog_partition={:.6}\ttrue_path_rank={}\ttop1={}\ttop8={}\ttop32={}\ttop64={}\ttrue_edges={}",
                        decode
                            .true_path_score
                            .map(|score| format!("{score:.6}"))
                            .unwrap_or_else(|| "NA".into()),
                        decode
                            .candidates
                            .first()
                            .map(|candidate| format!("{:.6}", candidate.path_log_probability))
                            .unwrap_or_else(|| "NA".into()),
                        decode.log_partition,
                        literal_rank
                            .map(|rank| rank.to_string())
                            .unwrap_or_else(|| ">64".into()),
                        if literal_rank.is_some_and(|rank| rank <= 1) { "YES" } else { "NO" },
                        if literal_rank.is_some_and(|rank| rank <= 8) { "YES" } else { "NO" },
                        if literal_rank.is_some_and(|rank| rank <= 32) { "YES" } else { "NO" },
                        if literal_rank.is_some_and(|rank| rank <= 64) { "YES" } else { "NO" },
                        audit.total_edges
                    );
                }

                metrics.cleavage_graph_final_candidates += decode.candidates.len();
                let mut inserted = 0usize;
                for graph_candidate in decode.candidates {
                    let tokens =
                        match vocabulary.encode(&graph_candidate.peptide, config.max_tokens) {
                            Ok(tokens) => tokens,
                            Err(_) => continue,
                        };
                    let mass_error_da = precursor_mass_error(record, &graph_candidate.peptide)?;
                    let mass_valid = mass_error_da
                        .map(|error| error.abs() <= mass_tolerance_da)
                        .unwrap_or(false);
                    if !mass_valid {
                        continue;
                    }
                    let evidence = peptidoform_fragment_evidence(
                        &graph_candidate.peptide,
                        target_neutral_mass,
                        &observed_peaks,
                        fragment_charge,
                        fragment_tolerance_ppm,
                    );
                    let candidate = GeneratedCandidate {
                        tokens: tokens.clone(),
                        peptide: graph_candidate.peptide,
                        reverse_log_probability: f64::NEG_INFINITY,
                        fragment_score: evidence.score,
                        matched_cleavages: evidence.matched_cleavages,
                        neural_all_mask_log_probability: f64::NEG_INFINITY,
                        neural_length_log_probability: f64::NEG_INFINITY,
                        hybrid_score: f64::NEG_INFINITY,
                        ar_total_log_probability: f64::NEG_INFINITY,
                        ar_mean_log_probability: f64::NEG_INFINITY,
                        ar_perplexity: f64::INFINITY,
                        fragment_causal_score: f64::NEG_INFINITY,
                        reverse_ar_total_log_probability: f64::NEG_INFINITY,
                        reverse_ar_mean_log_probability: f64::NEG_INFINITY,
                        reverse_ar_perplexity: f64::INFINITY,
                        bidirectional_ar_total_log_probability: f64::NEG_INFINITY,
                        fragment_bidirectional_causal_score: f64::NEG_INFINITY,
                        mass_error_da,
                        mass_valid,
                        from_diffusion: false,
                        from_causal_beam: false,
                        from_reverse_causal_beam: false,
                        from_bidirectional_mitm: false,
                        from_iterative_refinement: false,
                        from_cleavage_graph: true,
                    };
                    unique
                        .entry(tokens)
                        .and_modify(|existing| {
                            existing.from_cleavage_graph = true;
                        })
                        .or_insert(candidate);
                    inserted += 1;
                }
                if inserted > 0 {
                    metrics.cleavage_graph_records_with_candidate += 1;
                }
            }
        }

        let mut candidates: Vec<GeneratedCandidate> = unique.into_values().collect();
        metrics.unique_candidates += candidates.len();
        if candidates.is_empty() {
            println!(
                "generation_record\trecord_index={record_index}\ttarget={}\ttarget_length={target_active_length}\tpredicted_length={predicted_active_length}\tvalid_candidates=0",
                record.peptidoform.sequence
            );
            continue;
        }
        metrics.records_with_candidate += 1;

        let target_all_mask_score = score_all_mask_token_row(
            &model,
            &diffusion_collator,
            &spectrum_collator,
            &config,
            record,
            &spectrum,
            &target_tokens,
            target_active_length,
            &device,
        )?;
        metrics.target_neural_all_mask_log_probability_sum +=
            target_all_mask_score.mean_token_log_probability;
        metrics.target_neural_length_log_probability_sum +=
            target_all_mask_score.length_log_probability;

        let candidate_scores = score_all_mask_candidates(
            &model,
            &diffusion_collator,
            &spectrum_collator,
            &config,
            record,
            &spectrum,
            &candidates,
            &device,
        )?;
        for (candidate, score) in candidates.iter_mut().zip(candidate_scores) {
            candidate.neural_all_mask_log_probability = score.mean_token_log_probability;
            candidate.neural_length_log_probability = score.length_log_probability;
            candidate.hybrid_score = candidate.fragment_score
                + neural_rerank_weight * candidate.neural_all_mask_log_probability;
        }

        let target_causal_score = if let Some(causal) = causal_reranker.as_ref() {
            let score = score_causal_token_row(
                &causal.model,
                &causal.collator,
                &spectrum_collator,
                record,
                &spectrum,
                &target_tokens,
                &device,
            )?;
            metrics.target_ar_total_log_probability_sum += score.total_log_probability;
            metrics.target_ar_mean_log_probability_sum += score.mean_log_probability;
            metrics.causal_scored_records += 1;
            Some(score)
        } else {
            None
        };
        if let Some(causal) = causal_reranker.as_ref() {
            let causal_scores = score_causal_candidates(
                &causal.model,
                &causal.collator,
                &spectrum_collator,
                record,
                &spectrum,
                &candidates,
                &device,
            )?;
            for (candidate, score) in candidates.iter_mut().zip(causal_scores) {
                candidate.ar_total_log_probability = score.total_log_probability;
                candidate.ar_mean_log_probability = score.mean_log_probability;
                candidate.ar_perplexity = score.perplexity;
                candidate.fragment_causal_score = foundation_fragment_causal_rerank_score(
                    candidate.fragment_score,
                    candidate.ar_total_log_probability,
                    causal_rerank_weight,
                );
            }
        }

        if !bidirectional_mitm && iterative_refiner.is_none() && cleavage_graph_proposer.is_none() {
            if let Some(reverse_causal) = reverse_causal_reranker.as_ref() {
                let reverse_causal_scores = score_reverse_causal_candidates(
                    &reverse_causal.model,
                    &reverse_causal.collator,
                    &spectrum_collator,
                    record,
                    &spectrum,
                    &candidates,
                    &device,
                )?;
                for (candidate, score) in candidates.iter_mut().zip(reverse_causal_scores) {
                    candidate.reverse_ar_total_log_probability = score.total_log_probability;
                    candidate.reverse_ar_mean_log_probability = score.mean_log_probability;
                    candidate.reverse_ar_perplexity = score.perplexity;
                    candidate.bidirectional_ar_total_log_probability = 0.5
                        * (candidate.ar_total_log_probability
                            + candidate.reverse_ar_total_log_probability);
                    candidate.fragment_bidirectional_causal_score =
                        foundation_fragment_causal_rerank_score(
                            candidate.fragment_score,
                            candidate.bidirectional_ar_total_log_probability,
                            causal_rerank_weight,
                        );
                }
            }
        }

        let mut reverse_ranked = candidates.clone();
        reverse_ranked.sort_by(|left, right| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        });
        if reverse_ranked[0].peptide == record.peptidoform {
            metrics.raw_top1_peptidoform_exact += 1;
        }

        let mut mass_ranked = candidates.clone();
        mass_ranked.sort_by(mass_candidate_order);
        let mut neural_ranked = candidates.clone();
        neural_ranked.sort_by(neural_mass_candidate_order);
        let mut hybrid_ranked = candidates.clone();
        hybrid_ranked.sort_by(hybrid_mass_candidate_order);
        let causal_ranked = causal_reranker.as_ref().map(|_| {
            let mut ranked = candidates.clone();
            ranked.sort_by(causal_mass_candidate_order);
            ranked
        });
        let fragment_causal_ranked = causal_reranker.as_ref().map(|_| {
            let mut ranked = candidates.clone();
            ranked.sort_by(fragment_causal_mass_candidate_order);
            ranked
        });
        let fragment_bidirectional_causal_ranked = if !bidirectional_mitm
            && iterative_refiner.is_none()
            && cleavage_graph_proposer.is_none()
        {
            reverse_causal_reranker.as_ref().map(|_| {
                let mut ranked = candidates.clone();
                ranked.sort_by(fragment_bidirectional_causal_mass_candidate_order);
                ranked
            })
        } else {
            None
        };
        candidates.sort_by(fragment_mass_candidate_order);
        if candidates.iter().any(|candidate| candidate.mass_valid) {
            metrics.records_with_mass_valid_candidate += 1;
        }
        if let Some(error) = mass_ranked
            .iter()
            .filter_map(|candidate| candidate.mass_error_da)
            .next()
        {
            metrics.best_abs_mass_error_sum += error.abs();
            metrics.best_abs_mass_error_records += 1;
        }
        if let Some(error) = mass_ranked
            .iter()
            .filter(|candidate| candidate.mass_valid)
            .filter_map(|candidate| candidate.mass_error_da)
            .next()
        {
            metrics.best_abs_mass_errors_mass_valid.push(error.abs());
        } else if let Some(error) = mass_ranked
            .iter()
            .filter_map(|candidate| candidate.mass_error_da)
            .next()
        {
            metrics.best_abs_mass_errors_no_mass_valid.push(error.abs());
        }
        metrics.top1_fragment_score_sum += candidates[0].fragment_score;
        metrics.top1_matched_cleavages += candidates[0].matched_cleavages;
        metrics.fragment_top1_neural_all_mask_log_probability_sum +=
            candidates[0].neural_all_mask_log_probability;
        metrics.neural_top1_neural_all_mask_log_probability_sum +=
            neural_ranked[0].neural_all_mask_log_probability;
        metrics.hybrid_top1_neural_all_mask_log_probability_sum +=
            hybrid_ranked[0].neural_all_mask_log_probability;
        if let (Some(causal_ranked), Some(fragment_causal_ranked)) =
            (causal_ranked.as_ref(), fragment_causal_ranked.as_ref())
        {
            metrics.fragment_top1_ar_total_log_probability_sum +=
                candidates[0].ar_total_log_probability;
            metrics.causal_top1_ar_total_log_probability_sum +=
                causal_ranked[0].ar_total_log_probability;
            metrics.fragment_causal_top1_ar_total_log_probability_sum +=
                fragment_causal_ranked[0].ar_total_log_probability;
        }

        let target_sequence = &record.peptidoform.sequence;
        let target_il = normalize_il(target_sequence);
        let mass_valid_pool: Vec<&GeneratedCandidate> = candidates
            .iter()
            .filter(|candidate| candidate.mass_valid)
            .collect();
        let frozen_parent_mass_valid_pool: Vec<&GeneratedCandidate> = mass_valid_pool
            .iter()
            .copied()
            .filter(|candidate| candidate.from_diffusion || candidate.from_causal_beam)
            .collect();
        let diffusion_mass_valid_pool: Vec<&GeneratedCandidate> = mass_valid_pool
            .iter()
            .copied()
            .filter(|candidate| candidate.from_diffusion)
            .collect();
        let causal_beam_mass_valid_pool: Vec<&GeneratedCandidate> = mass_valid_pool
            .iter()
            .copied()
            .filter(|candidate| candidate.from_causal_beam)
            .collect();
        let reverse_causal_beam_mass_valid_pool: Vec<&GeneratedCandidate> = mass_valid_pool
            .iter()
            .copied()
            .filter(|candidate| candidate.from_reverse_causal_beam)
            .collect();
        let mitm_mass_valid_pool: Vec<&GeneratedCandidate> = mass_valid_pool
            .iter()
            .copied()
            .filter(|candidate| candidate.from_bidirectional_mitm)
            .collect();
        let iterative_refinement_mass_valid_pool: Vec<&GeneratedCandidate> = mass_valid_pool
            .iter()
            .copied()
            .filter(|candidate| candidate.from_iterative_refinement)
            .collect();
        let cleavage_graph_mass_valid_pool: Vec<&GeneratedCandidate> = mass_valid_pool
            .iter()
            .copied()
            .filter(|candidate| candidate.from_cleavage_graph)
            .collect();
        if mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.candidate_pool_mass_valid_peptidoform_exact += 1;
        }
        if mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.candidate_pool_mass_valid_sequence_exact += 1;
        }
        if mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.candidate_pool_mass_valid_il_sequence_exact += 1;
        }
        if frozen_parent_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.frozen_parent_pool_mass_valid_peptidoform_exact += 1;
        }
        if frozen_parent_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.frozen_parent_pool_mass_valid_sequence_exact += 1;
        }
        if frozen_parent_mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.frozen_parent_pool_mass_valid_il_sequence_exact += 1;
        }
        if diffusion_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.diffusion_pool_mass_valid_peptidoform_exact += 1;
        }
        if diffusion_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.diffusion_pool_mass_valid_sequence_exact += 1;
        }
        if diffusion_mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.diffusion_pool_mass_valid_il_sequence_exact += 1;
        }
        if causal_beam_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.causal_beam_pool_mass_valid_peptidoform_exact += 1;
        }
        if causal_beam_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.causal_beam_pool_mass_valid_sequence_exact += 1;
        }
        if causal_beam_mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.causal_beam_pool_mass_valid_il_sequence_exact += 1;
        }
        if reverse_causal_beam_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.reverse_causal_beam_pool_mass_valid_peptidoform_exact += 1;
        }
        if reverse_causal_beam_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.reverse_causal_beam_pool_mass_valid_sequence_exact += 1;
        }
        if reverse_causal_beam_mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.reverse_causal_beam_pool_mass_valid_il_sequence_exact += 1;
        }
        if mitm_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.mitm_pool_mass_valid_peptidoform_exact += 1;
        }
        if mitm_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.mitm_pool_mass_valid_sequence_exact += 1;
        }
        if mitm_mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.mitm_pool_mass_valid_il_sequence_exact += 1;
        }
        if iterative_refinement_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.iterative_refinement_pool_mass_valid_peptidoform_exact += 1;
        }
        if iterative_refinement_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.iterative_refinement_pool_mass_valid_sequence_exact += 1;
        }
        if iterative_refinement_mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.iterative_refinement_pool_mass_valid_il_sequence_exact += 1;
        }
        if cleavage_graph_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.cleavage_graph_pool_mass_valid_peptidoform_exact += 1;
        }
        if cleavage_graph_mass_valid_pool
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.cleavage_graph_pool_mass_valid_sequence_exact += 1;
        }
        if cleavage_graph_mass_valid_pool
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.cleavage_graph_pool_mass_valid_il_sequence_exact += 1;
        }
        if mass_ranked[0].peptide == record.peptidoform {
            metrics.mass_top1_peptidoform_exact += 1;
        }
        if candidates
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.mass_topk_peptidoform_exact += 1;
        }
        if mass_ranked[0].peptide.sequence.as_str() == target_sequence.as_str() {
            metrics.mass_top1_sequence_exact += 1;
        }
        if candidates
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.mass_topk_sequence_exact += 1;
        }
        if normalize_il(&mass_ranked[0].peptide.sequence) == target_il {
            metrics.mass_top1_il_sequence_exact += 1;
        }
        if candidates
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.mass_topk_il_sequence_exact += 1;
        }
        if candidates[0].peptide == record.peptidoform {
            metrics.fragment_top1_peptidoform_exact += 1;
        }
        if candidates[0].peptide.sequence.as_str() == target_sequence.as_str() {
            metrics.fragment_top1_sequence_exact += 1;
        }
        if normalize_il(&candidates[0].peptide.sequence) == target_il {
            metrics.fragment_top1_il_sequence_exact += 1;
        }
        let neural_exact = ranking_exact_flags(&neural_ranked[0], record, &target_il);
        metrics.neural_top1_peptidoform_exact += neural_exact.0;
        metrics.neural_top1_sequence_exact += neural_exact.1;
        metrics.neural_top1_il_sequence_exact += neural_exact.2;
        let hybrid_exact = ranking_exact_flags(&hybrid_ranked[0], record, &target_il);
        metrics.hybrid_top1_peptidoform_exact += hybrid_exact.0;
        metrics.hybrid_top1_sequence_exact += hybrid_exact.1;
        metrics.hybrid_top1_il_sequence_exact += hybrid_exact.2;
        if let Some(causal_ranked) = causal_ranked.as_ref() {
            let exact = ranking_exact_flags(&causal_ranked[0], record, &target_il);
            metrics.causal_top1_peptidoform_exact += exact.0;
            metrics.causal_top1_sequence_exact += exact.1;
            metrics.causal_top1_il_sequence_exact += exact.2;
        }
        if let Some(fragment_causal_ranked) = fragment_causal_ranked.as_ref() {
            let exact = ranking_exact_flags(&fragment_causal_ranked[0], record, &target_il);
            metrics.fragment_causal_top1_peptidoform_exact += exact.0;
            metrics.fragment_causal_top1_sequence_exact += exact.1;
            metrics.fragment_causal_top1_il_sequence_exact += exact.2;

            if let Some(causal_beam_top1) = fragment_causal_ranked
                .iter()
                .find(|candidate| candidate.from_causal_beam)
            {
                let causal_beam_exact = ranking_exact_flags(causal_beam_top1, record, &target_il);
                metrics.causal_beam_top1_peptidoform_exact += causal_beam_exact.0;
                metrics.causal_beam_top1_sequence_exact += causal_beam_exact.1;
                metrics.causal_beam_top1_il_sequence_exact += causal_beam_exact.2;
            }
        }
        if let Some(fragment_bidirectional_causal_ranked) =
            fragment_bidirectional_causal_ranked.as_ref()
        {
            let exact =
                ranking_exact_flags(&fragment_bidirectional_causal_ranked[0], record, &target_il);
            metrics.fragment_bidirectional_causal_top1_peptidoform_exact += exact.0;
            metrics.fragment_bidirectional_causal_top1_sequence_exact += exact.1;
            metrics.fragment_bidirectional_causal_top1_il_sequence_exact += exact.2;
        }

        let reverse_rank_by_tokens: HashMap<Vec<u32>, usize> = reverse_ranked
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
            .collect();
        let mass_rank_by_tokens: HashMap<Vec<u32>, usize> = mass_ranked
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
            .collect();
        let neural_rank_by_tokens: HashMap<Vec<u32>, usize> = neural_ranked
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
            .collect();
        let hybrid_rank_by_tokens: HashMap<Vec<u32>, usize> = hybrid_ranked
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
            .collect();
        let causal_rank_by_tokens: HashMap<Vec<u32>, usize> = causal_ranked
            .as_ref()
            .map(|ranked| {
                ranked
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
                    .collect()
            })
            .unwrap_or_default();
        let fragment_causal_rank_by_tokens: HashMap<Vec<u32>, usize> = fragment_causal_ranked
            .as_ref()
            .map(|ranked| {
                ranked
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
                    .collect()
            })
            .unwrap_or_default();
        let fragment_bidirectional_causal_rank_by_tokens: HashMap<Vec<u32>, usize> =
            fragment_bidirectional_causal_ranked
                .as_ref()
                .map(|ranked| {
                    ranked
                        .iter()
                        .enumerate()
                        .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
                        .collect()
                })
                .unwrap_or_default();
        for (fragment_mass_index, candidate) in candidates.iter().enumerate() {
            let fields = vec![
                record_index.to_string(),
                corpus.provenance[record_index].source_id.clone(),
                record.peptidoform.sequence.clone(),
                target_active_length.to_string(),
                predicted_active_length.to_string(),
                (fragment_mass_index + 1).to_string(),
                neural_rank_by_tokens
                    .get(&candidate.tokens)
                    .copied()
                    .unwrap_or(0)
                    .to_string(),
                hybrid_rank_by_tokens
                    .get(&candidate.tokens)
                    .copied()
                    .unwrap_or(0)
                    .to_string(),
                causal_rank_by_tokens
                    .get(&candidate.tokens)
                    .copied()
                    .unwrap_or(0)
                    .to_string(),
                fragment_causal_rank_by_tokens
                    .get(&candidate.tokens)
                    .copied()
                    .unwrap_or(0)
                    .to_string(),
                fragment_bidirectional_causal_rank_by_tokens
                    .get(&candidate.tokens)
                    .copied()
                    .unwrap_or(0)
                    .to_string(),
                mass_rank_by_tokens
                    .get(&candidate.tokens)
                    .copied()
                    .unwrap_or(0)
                    .to_string(),
                reverse_rank_by_tokens
                    .get(&candidate.tokens)
                    .copied()
                    .unwrap_or(0)
                    .to_string(),
                candidate.peptide.sequence.clone(),
                format_modifications(&candidate.peptide),
                format!("{:.8}", candidate.reverse_log_probability),
                format!("{:.8}", candidate.fragment_score),
                candidate.matched_cleavages.to_string(),
                format!("{:.8}", candidate.neural_all_mask_log_probability),
                format!("{:.8}", (-candidate.neural_all_mask_log_probability).exp()),
                format!("{:.8}", candidate.neural_length_log_probability),
                format!("{:.8}", candidate.hybrid_score),
                format_finite(candidate.ar_total_log_probability),
                format_finite(candidate.ar_mean_log_probability),
                format_finite(candidate.ar_perplexity),
                format_finite(candidate.fragment_causal_score),
                format_finite(candidate.reverse_ar_total_log_probability),
                format_finite(candidate.reverse_ar_mean_log_probability),
                format_finite(candidate.reverse_ar_perplexity),
                format_finite(candidate.bidirectional_ar_total_log_probability),
                format_finite(candidate.fragment_bidirectional_causal_score),
                candidate
                    .mass_error_da
                    .map(|value| format!("{value:.8}"))
                    .unwrap_or_default(),
                candidate.mass_valid.to_string(),
                candidate.from_diffusion.to_string(),
                candidate.from_causal_beam.to_string(),
                candidate.from_reverse_causal_beam.to_string(),
                candidate.from_bidirectional_mitm.to_string(),
                candidate.from_iterative_refinement.to_string(),
                candidate.from_cleavage_graph.to_string(),
                (candidate.peptide == record.peptidoform).to_string(),
                (candidate.peptide.sequence.as_str() == record.peptidoform.sequence.as_str())
                    .to_string(),
                (normalize_il(&candidate.peptide.sequence) == target_il).to_string(),
            ];
            writeln!(output, "{}", fields.join("\t"))?;
        }

        println!(
            "generation_record\trecord_index={record_index}\ttarget={}\ttarget_length={target_active_length}\tpredicted_length={predicted_active_length}\tvalid_candidates={}\tmass_valid_candidates={}\tbest_mass_error_da={}\ttarget_fragment_score={:.4}\ttarget_matched_cleavages={}\ttarget_neural_all_mask_logp={:.4}\tfragment_top1={}\tfragment_top1_score={:.4}\tfragment_top1_neural_logp={:.4}\tneural_top1={}\tneural_top1_logp={:.4}\thybrid_top1={}\thybrid_top1_score={:.4}\ttopk_exact={}\ttopk_il_exact={}",
            record.peptidoform.sequence,
            candidates.len(),
            candidates.iter().filter(|candidate| candidate.mass_valid).count(),
            mass_ranked[0]
                .mass_error_da
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "NA".into()),
            target_fragment_evidence.score,
            target_fragment_evidence.matched_cleavages,
            target_all_mask_score.mean_token_log_probability,
            candidates[0].peptide.sequence,
            candidates[0].fragment_score,
            candidates[0].neural_all_mask_log_probability,
            neural_ranked[0].peptide.sequence,
            neural_ranked[0].neural_all_mask_log_probability,
            hybrid_ranked[0].peptide.sequence,
            hybrid_ranked[0].hybrid_score,
            candidates.iter().any(|candidate| candidate.peptide == record.peptidoform),
            candidates
                .iter()
                .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il),
        );
        if let (Some(target_score), Some(causal_ranked), Some(fragment_causal_ranked)) = (
            target_causal_score,
            causal_ranked.as_ref(),
            fragment_causal_ranked.as_ref(),
        ) {
            println!(
                "generation_causal\trecord_index={record_index}\ttarget_ar_total_logp={:.4}\ttarget_ar_mean_logp={:.4}\tfragment_top1_ar_total_logp={:.4}\tcausal_top1={}\tcausal_top1_ar_total_logp={:.4}\tfragment_causal_top1={}\tfragment_causal_score={:.4}\tfragment_causal_ar_total_logp={:.4}",
                target_score.total_log_probability,
                target_score.mean_log_probability,
                candidates[0].ar_total_log_probability,
                causal_ranked[0].peptide.sequence,
                causal_ranked[0].ar_total_log_probability,
                fragment_causal_ranked[0].peptide.sequence,
                fragment_causal_ranked[0].fragment_causal_score,
                fragment_causal_ranked[0].ar_total_log_probability,
            );
            if causal_generation_beam_width > 0 {
                if let Some(causal_beam_top1) = fragment_causal_ranked
                    .iter()
                    .find(|candidate| candidate.from_causal_beam)
                {
                    println!(
                        "generation_causal_beam\trecord_index={record_index}\tcandidates={}\tmass_valid_candidates={}\ttop1={}\ttop1_score={:.4}\ttop1_ar_total_logp={:.4}\ttopk_exact={}\ttopk_il_exact={}",
                        candidates.iter().filter(|candidate| candidate.from_causal_beam).count(),
                        causal_beam_mass_valid_pool.len(),
                        causal_beam_top1.peptide.sequence,
                        causal_beam_top1.fragment_causal_score,
                        causal_beam_top1.ar_total_log_probability,
                        causal_beam_mass_valid_pool
                            .iter()
                            .any(|candidate| candidate.peptide == record.peptidoform),
                        causal_beam_mass_valid_pool
                            .iter()
                            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il),
                    );
                }
            }
        }
        if let Some(fragment_bidirectional_causal_ranked) =
            fragment_bidirectional_causal_ranked.as_ref()
        {
            let top1 = &fragment_bidirectional_causal_ranked[0];
            println!(
                "generation_bidirectional_causal\trecord_index={record_index}\ttop1={}\ttop1_score={:.4}\ttop1_n_to_c_ar_total_logp={:.4}\ttop1_c_to_n_ar_total_logp={:.4}\ttop1_bidirectional_ar_total_logp={:.4}",
                top1.peptide.sequence,
                top1.fragment_bidirectional_causal_score,
                top1.ar_total_log_probability,
                top1.reverse_ar_total_log_probability,
                top1.bidirectional_ar_total_log_probability,
            );
        }
        if bidirectional_mitm {
            let mitm_top1 = fragment_causal_ranked.as_ref().and_then(|ranked| {
                ranked
                    .iter()
                    .find(|candidate| candidate.from_bidirectional_mitm)
            });
            println!(
                "generation_bidirectional_mitm\trecord_index={record_index}\tcandidates={}\tmass_valid_candidates={}\ttop1={}\ttop1_score={}\ttopk_exact={}\ttopk_il_exact={}",
                candidates
                    .iter()
                    .filter(|candidate| candidate.from_bidirectional_mitm)
                    .count(),
                mitm_mass_valid_pool.len(),
                mitm_top1
                    .map(|candidate| candidate.peptide.sequence.as_str())
                    .unwrap_or("NA"),
                mitm_top1
                    .map(|candidate| format!("{:.4}", candidate.fragment_causal_score))
                    .unwrap_or_else(|| "NA".into()),
                mitm_mass_valid_pool
                    .iter()
                    .any(|candidate| candidate.peptide == record.peptidoform),
                mitm_mass_valid_pool
                    .iter()
                    .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il),
            );
        }
        if iterative_refiner.is_some() {
            let iterative_top1 = fragment_causal_ranked.as_ref().and_then(|ranked| {
                ranked
                    .iter()
                    .find(|candidate| candidate.from_iterative_refinement)
            });
            println!(
                "generation_iterative_refinement\trecord_index={record_index}\tcandidates={}\tmass_valid_candidates={}\ttop1={}\ttop1_score={}\ttopk_exact={}\ttopk_il_exact={}",
                candidates
                    .iter()
                    .filter(|candidate| candidate.from_iterative_refinement)
                    .count(),
                iterative_refinement_mass_valid_pool.len(),
                iterative_top1
                    .map(|candidate| candidate.peptide.sequence.as_str())
                    .unwrap_or("NA"),
                iterative_top1
                    .map(|candidate| format!("{:.4}", candidate.fragment_causal_score))
                    .unwrap_or_else(|| "NA".into()),
                iterative_refinement_mass_valid_pool
                    .iter()
                    .any(|candidate| candidate.peptide == record.peptidoform),
                iterative_refinement_mass_valid_pool
                    .iter()
                    .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il),
            );
        }
        if cleavage_graph_proposer.is_some() {
            let graph_top1 = fragment_causal_ranked.as_ref().and_then(|ranked| {
                ranked
                    .iter()
                    .find(|candidate| candidate.from_cleavage_graph)
            });
            println!(
                "generation_cleavage_graph\trecord_index={record_index}\tcandidates={}\tmass_valid_candidates={}\ttop1={}\ttop1_score={}\ttopk_exact={}\ttopk_il_exact={}",
                candidates
                    .iter()
                    .filter(|candidate| candidate.from_cleavage_graph)
                    .count(),
                cleavage_graph_mass_valid_pool.len(),
                graph_top1
                    .map(|candidate| candidate.peptide.sequence.as_str())
                    .unwrap_or("NA"),
                graph_top1
                    .map(|candidate| format!("{:.4}", candidate.fragment_causal_score))
                    .unwrap_or_else(|| "NA".into()),
                cleavage_graph_mass_valid_pool
                    .iter()
                    .any(|candidate| candidate.peptide == record.peptidoform),
                cleavage_graph_mass_valid_pool
                    .iter()
                    .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il),
            );
        }
    }
    output.flush()?;
    spectra_output.flush()?;
    println!("observed_spectra\t{}", spectra_path.display());

    let records = metrics.records.max(1) as f64;
    println!("generation_summary\trecords\t{}", metrics.records);
    println!(
        "generation_summary\tpredicted_length_accuracy\t{:.6}",
        metrics.predicted_length_exact as f64 / records
    );
    println!(
        "generation_summary\tpredicted_length_mae_tokens\t{:.4}",
        metrics.predicted_length_abs_error as f64 / records
    );
    println!("generation_summary\tchains\t{}", metrics.chains);
    println!(
        "generation_summary\tfinal_mass_beam_states\t{}",
        metrics.final_states
    );
    println!(
        "generation_summary\tvalid_decode_rate\t{:.6}",
        metrics.valid_decodes as f64 / metrics.final_states.max(1) as f64
    );
    println!(
        "generation_summary\tmean_unique_candidates\t{:.4}",
        metrics.unique_candidates as f64 / records
    );
    println!(
        "generation_summary\tmass_valid_candidate_rate\t{:.6}",
        metrics.mass_valid_candidates as f64 / metrics.valid_decodes.max(1) as f64
    );
    println!(
        "generation_summary\trecords_with_mass_valid_candidate_rate\t{:.6}",
        metrics.records_with_mass_valid_candidate as f64 / records
    );
    println!(
        "generation_summary\traw_top1_peptidoform_exact\t{:.6}",
        metrics.raw_top1_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tmass_top1_peptidoform_exact\t{:.6}",
        metrics.mass_top1_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tmass_topk_peptidoform_exact\t{:.6}",
        metrics.mass_topk_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tmass_top1_sequence_exact\t{:.6}",
        metrics.mass_top1_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tmass_topk_sequence_exact\t{:.6}",
        metrics.mass_topk_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tmass_top1_il_sequence_exact\t{:.6}",
        metrics.mass_top1_il_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tmass_topk_il_sequence_exact\t{:.6}",
        metrics.mass_topk_il_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tfragment_top1_peptidoform_exact\t{:.6}",
        metrics.fragment_top1_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tfragment_top1_sequence_exact\t{:.6}",
        metrics.fragment_top1_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tfragment_top1_il_sequence_exact\t{:.6}",
        metrics.fragment_top1_il_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tcandidate_pool_mass_valid_peptidoform_exact\t{:.6}",
        metrics.candidate_pool_mass_valid_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tcandidate_pool_mass_valid_sequence_exact\t{:.6}",
        metrics.candidate_pool_mass_valid_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tcandidate_pool_mass_valid_il_sequence_exact\t{:.6}",
        metrics.candidate_pool_mass_valid_il_sequence_exact as f64 / records
    );
    if reverse_causal_reranker.is_some() && reverse_causal_generation_beam_width > 0 {
        println!(
            "generation_summary\tfrozen_parent_pool_mass_valid_peptidoform_exact\t{:.6}",
            metrics.frozen_parent_pool_mass_valid_peptidoform_exact as f64 / records
        );
        println!(
            "generation_summary\tfrozen_parent_pool_mass_valid_sequence_exact\t{:.6}",
            metrics.frozen_parent_pool_mass_valid_sequence_exact as f64 / records
        );
        println!(
            "generation_summary\tfrozen_parent_pool_mass_valid_il_sequence_exact\t{:.6}",
            metrics.frozen_parent_pool_mass_valid_il_sequence_exact as f64 / records
        );
        let has_post_v01313_extension =
            bidirectional_mitm || iterative_refiner.is_some() || cleavage_graph_proposer.is_some();
        let accepted_threeway_literal = if has_post_v01313_extension {
            metrics.frozen_v01313_pool_mass_valid_peptidoform_exact
        } else {
            metrics.candidate_pool_mass_valid_peptidoform_exact
        };
        let accepted_threeway_il = if has_post_v01313_extension {
            metrics.frozen_v01313_pool_mass_valid_il_sequence_exact
        } else {
            metrics.candidate_pool_mass_valid_il_sequence_exact
        };
        println!(
            "generation_summary\treverse_causal_incremental_literal_records\t{}",
            accepted_threeway_literal
                .saturating_sub(metrics.frozen_parent_pool_mass_valid_peptidoform_exact)
        );
        println!(
            "generation_summary\treverse_causal_incremental_il_records\t{}",
            accepted_threeway_il
                .saturating_sub(metrics.frozen_parent_pool_mass_valid_il_sequence_exact)
        );
        println!(
            "frozen_v01310_parent_parity\texpected_literal=23\texpected_il=39\tobserved_literal={}\tobserved_il={}\tparity={}",
            metrics.frozen_parent_pool_mass_valid_peptidoform_exact,
            metrics.frozen_parent_pool_mass_valid_il_sequence_exact,
            if metrics.frozen_parent_pool_mass_valid_peptidoform_exact == 23
                && metrics.frozen_parent_pool_mass_valid_il_sequence_exact == 39
            {
                "YES"
            } else {
                "NO"
            }
        );
        let parent_parity = metrics.frozen_parent_pool_mass_valid_peptidoform_exact == 23
            && metrics.frozen_parent_pool_mass_valid_il_sequence_exact == 39;
        let branch_parity = metrics.diffusion_pool_mass_valid_peptidoform_exact == 7
            && metrics.diffusion_pool_mass_valid_il_sequence_exact == 30
            && metrics.causal_beam_pool_mass_valid_peptidoform_exact == 21
            && metrics.causal_beam_pool_mass_valid_il_sequence_exact == 33;
        println!(
            "frozen_v01310_branch_parity\tdiffusion_expected_literal=7\tdiffusion_expected_il=30\tdiffusion_observed_literal={}\tdiffusion_observed_il={}\tn_to_c_expected_literal=21\tn_to_c_expected_il=33\tn_to_c_observed_literal={}\tn_to_c_observed_il={}\tparity={}",
            metrics.diffusion_pool_mass_valid_peptidoform_exact,
            metrics.diffusion_pool_mass_valid_il_sequence_exact,
            metrics.causal_beam_pool_mass_valid_peptidoform_exact,
            metrics.causal_beam_pool_mass_valid_il_sequence_exact,
            if branch_parity { "YES" } else { "NO" }
        );
        let reverse_gate = parent_parity
            && branch_parity
            && accepted_threeway_literal >= 25
            && accepted_threeway_il >= 45;
        println!(
            "reverse_causal_acceptance_gate\trequired_literal=25\trequired_il=45\tobserved_literal={}\tobserved_il={}\tparent_parity={}\tbranch_parity={}\tgate={}",
            accepted_threeway_literal,
            accepted_threeway_il,
            if parent_parity { "YES" } else { "NO" },
            if branch_parity { "YES" } else { "NO" },
            if reverse_gate { "PASS" } else { "FAIL" }
        );

        let candidate_pool_parity = accepted_threeway_literal == 28 && accepted_threeway_il == 46;
        let accepted_forward_literal = if has_post_v01313_extension {
            metrics.frozen_v01313_forward_ranking_peptidoform_exact
        } else {
            metrics.fragment_causal_top1_peptidoform_exact
        };
        let accepted_forward_il = if has_post_v01313_extension {
            metrics.frozen_v01313_forward_ranking_il_sequence_exact
        } else {
            metrics.fragment_causal_top1_il_sequence_exact
        };
        let forward_ranking_parity = accepted_forward_literal == 23 && accepted_forward_il == 37;
        println!(
            "frozen_v01313_candidate_pool_parity\texpected_literal=28\texpected_il=46\tobserved_literal={}\tobserved_il={}\tparity={}",
            accepted_threeway_literal,
            accepted_threeway_il,
            if candidate_pool_parity { "YES" } else { "NO" }
        );
        println!(
            "frozen_v01313_forward_ranking_parity\texpected_literal=23\texpected_il=37\tobserved_literal={}\tobserved_il={}\tparity={}",
            accepted_forward_literal,
            accepted_forward_il,
            if forward_ranking_parity { "YES" } else { "NO" }
        );

        if !has_post_v01313_extension {
            let bidirectional_gate = parent_parity
                && branch_parity
                && candidate_pool_parity
                && forward_ranking_parity
                && metrics.fragment_bidirectional_causal_top1_peptidoform_exact >= 25
                && metrics.fragment_bidirectional_causal_top1_il_sequence_exact >= 40;
            println!(
                "bidirectional_causal_acceptance_gate\trequired_literal=25\trequired_il=40\tobserved_literal={}\tobserved_il={}\tcandidate_pool_literal={}\tcandidate_pool_il={}\tcandidate_pool_parity={}\tforward_ranking_parity={}\tparent_parity={}\tbranch_parity={}\tgate={}",
                metrics.fragment_bidirectional_causal_top1_peptidoform_exact,
                metrics.fragment_bidirectional_causal_top1_il_sequence_exact,
                accepted_threeway_literal,
                accepted_threeway_il,
                if candidate_pool_parity { "YES" } else { "NO" },
                if forward_ranking_parity { "YES" } else { "NO" },
                if parent_parity { "YES" } else { "NO" },
                if branch_parity { "YES" } else { "NO" },
                if bidirectional_gate { "PASS" } else { "FAIL" }
            );
        }

        if bidirectional_mitm && !generation_partition_train {
            println!(
                "generation_summary\tfrozen_v01313_pool_mass_valid_peptidoform_exact\t{:.6}",
                metrics.frozen_v01313_pool_mass_valid_peptidoform_exact as f64 / records
            );
            println!(
                "generation_summary\tfrozen_v01313_pool_mass_valid_il_sequence_exact\t{:.6}",
                metrics.frozen_v01313_pool_mass_valid_il_sequence_exact as f64 / records
            );
            println!(
                "generation_summary\tmitm_incremental_literal_records\t{}",
                metrics
                    .candidate_pool_mass_valid_peptidoform_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_peptidoform_exact)
            );
            println!(
                "generation_summary\tmitm_incremental_il_records\t{}",
                metrics
                    .candidate_pool_mass_valid_il_sequence_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_il_sequence_exact)
            );
            let legacy_join_pool_parity = !bidirectional_mitm_evidence_aware
                || (metrics.mitm_records_with_mass_join == 123
                    && metrics.mitm_unique_mass_joins_before_cap == 1_162_210
                    && metrics.mitm_joined_candidates == 31_361);
            let legacy_selector_parity = !bidirectional_mitm_evidence_aware
                || (metrics.mitm_legacy_pool_mass_valid_peptidoform_exact == 42
                    && metrics.mitm_legacy_pool_mass_valid_il_sequence_exact == 51);
            if bidirectional_mitm_evidence_aware {
                println!(
                    "v01319_join_pool_parity\texpected_records_with_join=123\texpected_unique_joins=1162210\texpected_retained=31361\tobserved_records_with_join={}\tobserved_unique_joins={}\tobserved_retained={}\tparity={}",
                    metrics.mitm_records_with_mass_join,
                    metrics.mitm_unique_mass_joins_before_cap,
                    metrics.mitm_joined_candidates,
                    if legacy_join_pool_parity { "YES" } else { "NO" }
                );
                println!(
                    "v01319_selector_shadow_parity\texpected_literal=42\texpected_il=51\tobserved_literal={}\tobserved_il={}\tparity={}",
                    metrics.mitm_legacy_pool_mass_valid_peptidoform_exact,
                    metrics.mitm_legacy_pool_mass_valid_il_sequence_exact,
                    if legacy_selector_parity { "YES" } else { "NO" }
                );
            }
            let mitm_gate = parent_parity
                && branch_parity
                && candidate_pool_parity
                && forward_ranking_parity
                && legacy_join_pool_parity
                && legacy_selector_parity
                && metrics.candidate_pool_mass_valid_peptidoform_exact >= 33
                && metrics.candidate_pool_mass_valid_il_sequence_exact >= 54;
            println!(
                "bidirectional_mitm_acceptance_gate\tpolicy={}\trequired_literal=33\trequired_il=54\tobserved_literal={}\tobserved_il={}\taccepted_top1_literal={}\taccepted_top1_il={}\tfrozen_pool_literal={}\tfrozen_pool_il={}\tlegacy_join_pool_parity={}\tlegacy_selector_parity={}\tcandidate_pool_parity={}\tforward_ranking_parity={}\tparent_parity={}\tbranch_parity={}\tgate={}",
                if bidirectional_mitm_final_two_view {
                    "v01323_final_two_view_fixed_budget"
                } else if bidirectional_mitm_component_audit {
                    "v01322_terminal_diagnostic_same_candidates_as_v01320"
                } else if bidirectional_mitm_precap_audit {
                    "v01321_diagnostic_same_candidates_as_v01320"
                } else if bidirectional_mitm_evidence_aware {
                    "v01320"
                } else {
                    "v01319"
                },
                metrics.candidate_pool_mass_valid_peptidoform_exact,
                metrics.candidate_pool_mass_valid_il_sequence_exact,
                metrics.fragment_causal_top1_peptidoform_exact,
                metrics.fragment_causal_top1_il_sequence_exact,
                accepted_threeway_literal,
                accepted_threeway_il,
                if legacy_join_pool_parity { "YES" } else { "NO" },
                if legacy_selector_parity { "YES" } else { "NO" },
                if candidate_pool_parity { "YES" } else { "NO" },
                if forward_ranking_parity { "YES" } else { "NO" },
                if parent_parity { "YES" } else { "NO" },
                if branch_parity { "YES" } else { "NO" },
                if mitm_gate { "PASS" } else { "FAIL" }
            );
            if bidirectional_mitm_final_two_view {
                let v01320_selector_shadow_parity =
                    metrics.mitm_v01320_shadow_pool_mass_valid_peptidoform_exact == 42
                        && metrics.mitm_v01320_shadow_pool_mass_valid_sequence_exact == 42
                        && metrics.mitm_v01320_shadow_pool_mass_valid_il_sequence_exact == 51;
                let v01320_union_shadow_parity = metrics.mitm_v01320_shadow_union_peptidoform_exact
                    == 43
                    && metrics.mitm_v01320_shadow_union_sequence_exact == 43
                    && metrics.mitm_v01320_shadow_union_il_sequence_exact == 53;
                let final_gate =
                    mitm_gate && v01320_selector_shadow_parity && v01320_union_shadow_parity;
                let lane_decision = if final_gate {
                    "ACCEPT_V01323_AND_CLOSE_MITM_SELECTOR_LANE"
                } else {
                    "REJECT_V01323_AND_CLOSE_MITM_SELECTOR_LANE"
                };
                println!(
                    "v01320_selector_shadow_parity\texpected_literal=42\texpected_sequence=42\texpected_il=51\tobserved_literal={}\tobserved_sequence={}\tobserved_il={}\tparity={}",
                    metrics.mitm_v01320_shadow_pool_mass_valid_peptidoform_exact,
                    metrics.mitm_v01320_shadow_pool_mass_valid_sequence_exact,
                    metrics.mitm_v01320_shadow_pool_mass_valid_il_sequence_exact,
                    yes_no(v01320_selector_shadow_parity),
                );
                println!(
                    "v01320_union_shadow_parity\texpected_literal=43\texpected_sequence=43\texpected_il=53\tobserved_literal={}\tobserved_sequence={}\tobserved_il={}\tparity={}",
                    metrics.mitm_v01320_shadow_union_peptidoform_exact,
                    metrics.mitm_v01320_shadow_union_sequence_exact,
                    metrics.mitm_v01320_shadow_union_il_sequence_exact,
                    yes_no(v01320_union_shadow_parity),
                );
                println!(
                    "v01323_final_two_view_selector\tview_quota_evidence=128\tview_quota_seam=128\tretained_literal={}\tretained_sequence={}\tretained_il={}\tunion_literal={}\tunion_sequence={}\tunion_il={}\tdisplaced_v01320_candidates={}\tv01320_selector_shadow_parity={}\tv01320_union_shadow_parity={}\toriginal_gate={}\tdecision={}",
                    metrics.mitm_pool_mass_valid_peptidoform_exact,
                    metrics.mitm_pool_mass_valid_sequence_exact,
                    metrics.mitm_pool_mass_valid_il_sequence_exact,
                    metrics.candidate_pool_mass_valid_peptidoform_exact,
                    metrics.candidate_pool_mass_valid_sequence_exact,
                    metrics.candidate_pool_mass_valid_il_sequence_exact,
                    metrics.mitm_v01323_displaced_v01320_candidates,
                    yes_no(v01320_selector_shadow_parity),
                    yes_no(v01320_union_shadow_parity),
                    if mitm_gate { "PASS" } else { "FAIL" },
                    lane_decision,
                );
            }
            if bidirectional_mitm_precap_audit {
                let v01320_selector_parity = metrics.mitm_pool_mass_valid_peptidoform_exact == 42
                    && metrics.mitm_pool_mass_valid_il_sequence_exact == 51;
                let v01320_union_parity = metrics.candidate_pool_mass_valid_peptidoform_exact == 43
                    && metrics.candidate_pool_mass_valid_il_sequence_exact == 53;
                let diagnosis = if metrics.mitm_precap_union_il_sequence_exact >= 54 {
                    "SELECTION_OR_BUDGET_LIMITED"
                } else {
                    "CONSTRUCTION_LIMITED_AT_FIXED_MIDPOINT_BEAMS"
                };
                println!(
                    "v01320_selector_parity\texpected_literal=42\texpected_il=51\tobserved_literal={}\tobserved_il={}\tparity={}",
                    metrics.mitm_pool_mass_valid_peptidoform_exact,
                    metrics.mitm_pool_mass_valid_il_sequence_exact,
                    yes_no(v01320_selector_parity),
                );
                println!(
                    "v01320_union_parity\texpected_literal=43\texpected_il=53\tobserved_literal={}\tobserved_il={}\tparity={}",
                    metrics.candidate_pool_mass_valid_peptidoform_exact,
                    metrics.candidate_pool_mass_valid_il_sequence_exact,
                    yes_no(v01320_union_parity),
                );
                println!(
                    "v01321_precap_join_oracle_audit\trequired_literal=33\trequired_il=54\tprecap_literal={}\tprecap_sequence={}\tprecap_il={}\tprecap_union_literal={}\tprecap_union_sequence={}\tprecap_union_il={}\tretained_v01320_literal={}\tretained_v01320_il={}\tfull_join_pool_candidates={}\tv01319_join_pool_parity={}\tv01319_selector_parity={}\tv01320_selector_parity={}\tv01320_union_parity={}\tdiagnosis={}",
                    metrics.mitm_precap_pool_peptidoform_exact,
                    metrics.mitm_precap_pool_sequence_exact,
                    metrics.mitm_precap_pool_il_sequence_exact,
                    metrics.mitm_precap_union_peptidoform_exact,
                    metrics.mitm_precap_union_sequence_exact,
                    metrics.mitm_precap_union_il_sequence_exact,
                    metrics.mitm_pool_mass_valid_peptidoform_exact,
                    metrics.mitm_pool_mass_valid_il_sequence_exact,
                    metrics.mitm_unique_mass_joins_before_cap,
                    yes_no(legacy_join_pool_parity),
                    yes_no(legacy_selector_parity),
                    yes_no(v01320_selector_parity),
                    yes_no(v01320_union_parity),
                    diagnosis,
                );
                if bidirectional_mitm_component_audit {
                    let v01321_precap_parity = metrics.mitm_precap_pool_peptidoform_exact == 43
                        && metrics.mitm_precap_pool_sequence_exact == 43
                        && metrics.mitm_precap_pool_il_sequence_exact == 52
                        && metrics.mitm_precap_union_peptidoform_exact == 44
                        && metrics.mitm_precap_union_sequence_exact == 44
                        && metrics.mitm_precap_union_il_sequence_exact == 54;
                    let deep_il = metrics.mitm_component_audit_incremental_deep_il_records;
                    let actionable_il = metrics.mitm_component_audit_deep_il_actionable_records;
                    let terminal_decision =
                        if v01321_precap_parity && deep_il > 0 && actionable_il == deep_il {
                            "ALLOW_ONE_FINAL_FIXED_SELECTOR_V01323_THEN_CLOSE_LANE"
                        } else {
                            "CLOSE_MITM_SELECTOR_LANE_AND_MOVE_ON"
                        };
                    println!(
                        "v01321_precap_parity\texpected_precap_literal=43\texpected_precap_sequence=43\texpected_precap_il=52\texpected_union_literal=44\texpected_union_sequence=44\texpected_union_il=54\tobserved_precap_literal={}\tobserved_precap_sequence={}\tobserved_precap_il={}\tobserved_union_literal={}\tobserved_union_sequence={}\tobserved_union_il={}\tparity={}",
                        metrics.mitm_precap_pool_peptidoform_exact,
                        metrics.mitm_precap_pool_sequence_exact,
                        metrics.mitm_precap_pool_il_sequence_exact,
                        metrics.mitm_precap_union_peptidoform_exact,
                        metrics.mitm_precap_union_sequence_exact,
                        metrics.mitm_precap_union_il_sequence_exact,
                        yes_no(v01321_precap_parity),
                    );
                    println!(
                        "v01322_terminal_component_rank_audit\tincremental_deep_literal_records={}\tincremental_deep_il_records={}\tdeep_literal_top256_any_component={}\tdeep_il_top256_any_component={}\tdeep_literal_pareto_le256={}\tdeep_il_pareto_le256={}\tdeep_literal_actionable={}\tdeep_il_actionable={}\tv01321_precap_parity={}\tdecision={}",
                        metrics.mitm_component_audit_incremental_deep_literal_records,
                        metrics.mitm_component_audit_incremental_deep_il_records,
                        metrics.mitm_component_audit_deep_literal_top256_any_component,
                        metrics.mitm_component_audit_deep_il_top256_any_component,
                        metrics.mitm_component_audit_deep_literal_pareto_le256,
                        metrics.mitm_component_audit_deep_il_pareto_le256,
                        metrics.mitm_component_audit_deep_literal_actionable_records,
                        metrics.mitm_component_audit_deep_il_actionable_records,
                        yes_no(v01321_precap_parity),
                        terminal_decision,
                    );
                }
            }
        }

        if iterative_refiner.is_some() {
            println!(
                "generation_summary\tfrozen_v01313_pool_mass_valid_peptidoform_exact\t{:.6}",
                metrics.frozen_v01313_pool_mass_valid_peptidoform_exact as f64 / records
            );
            println!(
                "generation_summary\tfrozen_v01313_pool_mass_valid_il_sequence_exact\t{:.6}",
                metrics.frozen_v01313_pool_mass_valid_il_sequence_exact as f64 / records
            );
            println!(
                "generation_summary\titerative_refinement_incremental_literal_records\t{}",
                metrics
                    .candidate_pool_mass_valid_peptidoform_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_peptidoform_exact)
            );
            println!(
                "generation_summary\titerative_refinement_incremental_il_records\t{}",
                metrics
                    .candidate_pool_mass_valid_il_sequence_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_il_sequence_exact)
            );
            let refinement_gate = parent_parity
                && branch_parity
                && candidate_pool_parity
                && forward_ranking_parity
                && metrics.candidate_pool_mass_valid_peptidoform_exact >= 32
                && metrics.candidate_pool_mass_valid_il_sequence_exact >= 52;
            println!(
                "iterative_refinement_acceptance_gate\trequired_literal=32\trequired_il=52\tobserved_literal={}\tobserved_il={}\tfrozen_pool_literal={}\tfrozen_pool_il={}\tcandidate_pool_parity={}\tforward_ranking_parity={}\tparent_parity={}\tbranch_parity={}\tgate={}",
                metrics.candidate_pool_mass_valid_peptidoform_exact,
                metrics.candidate_pool_mass_valid_il_sequence_exact,
                accepted_threeway_literal,
                accepted_threeway_il,
                if candidate_pool_parity { "YES" } else { "NO" },
                if forward_ranking_parity { "YES" } else { "NO" },
                if parent_parity { "YES" } else { "NO" },
                if branch_parity { "YES" } else { "NO" },
                if refinement_gate { "PASS" } else { "FAIL" }
            );
        }
        if cleavage_graph_proposer.is_some() {
            println!(
                "generation_summary\tfrozen_v01313_pool_mass_valid_peptidoform_exact\t{:.6}",
                metrics.frozen_v01313_pool_mass_valid_peptidoform_exact as f64 / records
            );
            println!(
                "generation_summary\tfrozen_v01313_pool_mass_valid_il_sequence_exact\t{:.6}",
                metrics.frozen_v01313_pool_mass_valid_il_sequence_exact as f64 / records
            );
            println!(
                "generation_summary\tcleavage_graph_incremental_literal_records\t{}",
                metrics
                    .candidate_pool_mass_valid_peptidoform_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_peptidoform_exact)
            );
            println!(
                "generation_summary\tcleavage_graph_incremental_il_records\t{}",
                metrics
                    .candidate_pool_mass_valid_il_sequence_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_il_sequence_exact)
            );
            let graph_gate = parent_parity
                && branch_parity
                && candidate_pool_parity
                && forward_ranking_parity
                && metrics.candidate_pool_mass_valid_peptidoform_exact >= 33
                && metrics.candidate_pool_mass_valid_il_sequence_exact >= 54;
            println!(
                "cleavage_graph_acceptance_gate\trequired_literal=33\trequired_il=54\tobserved_literal={}\tobserved_il={}\taccepted_top1_literal={}\taccepted_top1_il={}\tfrozen_pool_literal={}\tfrozen_pool_il={}\tcandidate_pool_parity={}\tforward_ranking_parity={}\tparent_parity={}\tbranch_parity={}\tgate={}",
                metrics.candidate_pool_mass_valid_peptidoform_exact,
                metrics.candidate_pool_mass_valid_il_sequence_exact,
                metrics.fragment_causal_top1_peptidoform_exact,
                metrics.fragment_causal_top1_il_sequence_exact,
                accepted_threeway_literal,
                accepted_threeway_il,
                if candidate_pool_parity { "YES" } else { "NO" },
                if forward_ranking_parity { "YES" } else { "NO" },
                if parent_parity { "YES" } else { "NO" },
                if branch_parity { "YES" } else { "NO" },
                if graph_gate { "PASS" } else { "FAIL" }
            );
        }
    }
    if generation_partition_train {
        println!(
            "v0140_train_candidate_export\trecords={}\tpool_literal={}\tpool_sequence={}\tpool_il={}\tretained_mitm_candidates={}\tpolicy=frozen_v01323_two_view\ttest_partition_consumed=NO",
            metrics.records,
            metrics.candidate_pool_mass_valid_peptidoform_exact,
            metrics.candidate_pool_mass_valid_sequence_exact,
            metrics.candidate_pool_mass_valid_il_sequence_exact,
            metrics.mitm_joined_candidates,
        );
    }

    println!(
        "generation_summary\tdiffusion_pool_mass_valid_peptidoform_exact\t{:.6}",
        metrics.diffusion_pool_mass_valid_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tdiffusion_pool_mass_valid_sequence_exact\t{:.6}",
        metrics.diffusion_pool_mass_valid_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tdiffusion_pool_mass_valid_il_sequence_exact\t{:.6}",
        metrics.diffusion_pool_mass_valid_il_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tcausal_beam_pool_mass_valid_peptidoform_exact\t{:.6}",
        metrics.causal_beam_pool_mass_valid_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tcausal_beam_pool_mass_valid_sequence_exact\t{:.6}",
        metrics.causal_beam_pool_mass_valid_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tcausal_beam_pool_mass_valid_il_sequence_exact\t{:.6}",
        metrics.causal_beam_pool_mass_valid_il_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tcausal_beam_final_candidates\t{}",
        metrics.causal_beam_final_candidates
    );
    println!(
        "generation_summary\tcausal_beam_records_with_candidate_rate\t{:.6}",
        metrics.causal_beam_records_with_candidate as f64 / records
    );
    if reverse_causal_reranker.is_some() && reverse_causal_generation_beam_width > 0 {
        println!(
            "generation_summary\treverse_causal_beam_pool_mass_valid_peptidoform_exact\t{:.6}",
            metrics.reverse_causal_beam_pool_mass_valid_peptidoform_exact as f64 / records
        );
        println!(
            "generation_summary\treverse_causal_beam_pool_mass_valid_sequence_exact\t{:.6}",
            metrics.reverse_causal_beam_pool_mass_valid_sequence_exact as f64 / records
        );
        println!(
            "generation_summary\treverse_causal_beam_pool_mass_valid_il_sequence_exact\t{:.6}",
            metrics.reverse_causal_beam_pool_mass_valid_il_sequence_exact as f64 / records
        );
        println!(
            "generation_summary\treverse_causal_beam_final_candidates\t{}",
            metrics.reverse_causal_beam_final_candidates
        );
        println!(
            "generation_summary\treverse_causal_beam_records_with_candidate_rate\t{:.6}",
            metrics.reverse_causal_beam_records_with_candidate as f64 / records
        );
    }
    if bidirectional_mitm {
        println!(
            "generation_summary\tmitm_prefix_records_with_states\t{}",
            metrics.mitm_prefix_records_with_states
        );
        println!(
            "generation_summary\tmitm_suffix_records_with_states\t{}",
            metrics.mitm_suffix_records_with_states
        );
        println!(
            "generation_summary\tmitm_records_with_mass_join\t{}",
            metrics.mitm_records_with_mass_join
        );
        println!(
            "generation_summary\tmitm_unique_mass_joins_before_cap\t{}",
            metrics.mitm_unique_mass_joins_before_cap
        );
        println!(
            "generation_summary\tmitm_joined_candidates\t{}",
            metrics.mitm_joined_candidates
        );
        if bidirectional_mitm_evidence_aware {
            println!(
                "generation_summary\tmitm_selector_scored_candidates\t{}",
                metrics.mitm_selector_scored_candidates
            );
            println!(
                "generation_summary\tmitm_selector_displaced_legacy_candidates\t{}",
                metrics.mitm_selector_displaced_legacy_candidates
            );
            println!(
                "generation_summary\tmitm_legacy_selector_literal_oracle\t{}",
                metrics.mitm_legacy_pool_mass_valid_peptidoform_exact
            );
            println!(
                "generation_summary\tmitm_legacy_selector_sequence_oracle\t{}",
                metrics.mitm_legacy_pool_mass_valid_sequence_exact
            );
            println!(
                "generation_summary\tmitm_legacy_selector_il_oracle\t{}",
                metrics.mitm_legacy_pool_mass_valid_il_sequence_exact
            );
        }
        if bidirectional_mitm_precap_audit {
            println!(
                "generation_summary\tmitm_precap_literal_oracle\t{}",
                metrics.mitm_precap_pool_peptidoform_exact
            );
            println!(
                "generation_summary\tmitm_precap_sequence_oracle\t{}",
                metrics.mitm_precap_pool_sequence_exact
            );
            println!(
                "generation_summary\tmitm_precap_il_oracle\t{}",
                metrics.mitm_precap_pool_il_sequence_exact
            );
            println!(
                "generation_summary\tmitm_precap_union_literal_oracle\t{}",
                metrics.mitm_precap_union_peptidoform_exact
            );
            println!(
                "generation_summary\tmitm_precap_union_sequence_oracle\t{}",
                metrics.mitm_precap_union_sequence_exact
            );
            println!(
                "generation_summary\tmitm_precap_union_il_oracle\t{}",
                metrics.mitm_precap_union_il_sequence_exact
            );
            println!(
                "generation_summary\tmitm_precap_incremental_literal_records\t{}",
                metrics
                    .mitm_precap_union_peptidoform_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_peptidoform_exact)
            );
            println!(
                "generation_summary\tmitm_precap_incremental_il_records\t{}",
                metrics
                    .mitm_precap_union_il_sequence_exact
                    .saturating_sub(metrics.frozen_v01313_pool_mass_valid_il_sequence_exact)
            );
            print_mitm_rank_cutoffs(
                "mitm_precap_v01319_literal_rank",
                &metrics.mitm_precap_legacy_literal_rank_cutoffs,
            );
            print_mitm_rank_cutoffs(
                "mitm_precap_v01319_sequence_rank",
                &metrics.mitm_precap_legacy_sequence_rank_cutoffs,
            );
            print_mitm_rank_cutoffs(
                "mitm_precap_v01319_il_rank",
                &metrics.mitm_precap_legacy_il_rank_cutoffs,
            );
            print_mitm_rank_cutoffs(
                "mitm_precap_v01320_literal_rank",
                &metrics.mitm_precap_evidence_literal_rank_cutoffs,
            );
            print_mitm_rank_cutoffs(
                "mitm_precap_v01320_sequence_rank",
                &metrics.mitm_precap_evidence_sequence_rank_cutoffs,
            );
            print_mitm_rank_cutoffs(
                "mitm_precap_v01320_il_rank",
                &metrics.mitm_precap_evidence_il_rank_cutoffs,
            );
        }
        println!(
            "generation_summary\tmitm_standalone_literal_oracle\t{}",
            metrics.mitm_pool_mass_valid_peptidoform_exact
        );
        println!(
            "generation_summary\tmitm_standalone_sequence_oracle\t{}",
            metrics.mitm_pool_mass_valid_sequence_exact
        );
        println!(
            "generation_summary\tmitm_standalone_il_oracle\t{}",
            metrics.mitm_pool_mass_valid_il_sequence_exact
        );
    }
    if iterative_refiner.is_some() {
        println!(
            "generation_summary\titerative_refinement_pool_mass_valid_peptidoform_exact\t{:.6}",
            metrics.iterative_refinement_pool_mass_valid_peptidoform_exact as f64 / records
        );
        println!(
            "generation_summary\titerative_refinement_pool_mass_valid_sequence_exact\t{:.6}",
            metrics.iterative_refinement_pool_mass_valid_sequence_exact as f64 / records
        );
        println!(
            "generation_summary\titerative_refinement_pool_mass_valid_il_sequence_exact\t{:.6}",
            metrics.iterative_refinement_pool_mass_valid_il_sequence_exact as f64 / records
        );
        println!(
            "generation_summary\titerative_refinement_final_candidates\t{}",
            metrics.iterative_refinement_final_candidates
        );
        println!(
            "generation_summary\titerative_refinement_records_with_candidate_rate\t{:.6}",
            metrics.iterative_refinement_records_with_candidate as f64 / records
        );
    }
    if cleavage_graph_proposer.is_some() {
        println!(
            "generation_summary\tcleavage_graph_pool_mass_valid_peptidoform_exact\t{:.6}",
            metrics.cleavage_graph_pool_mass_valid_peptidoform_exact as f64 / records
        );
        println!(
            "generation_summary\tcleavage_graph_pool_mass_valid_sequence_exact\t{:.6}",
            metrics.cleavage_graph_pool_mass_valid_sequence_exact as f64 / records
        );
        println!(
            "generation_summary\tcleavage_graph_pool_mass_valid_il_sequence_exact\t{:.6}",
            metrics.cleavage_graph_pool_mass_valid_il_sequence_exact as f64 / records
        );
        println!(
            "generation_summary\tcleavage_graph_final_candidates\t{}",
            metrics.cleavage_graph_final_candidates
        );
        println!(
            "generation_summary\tcleavage_graph_records_with_candidate_rate\t{:.6}",
            metrics.cleavage_graph_records_with_candidate as f64 / records
        );
        println!(
            "generation_summary\ttrue_path_structural_coverage\t{:.6}",
            metrics.cleavage_graph_true_path_structural_present as f64
                / metrics.cleavage_graph_structural_records.max(1) as f64
        );
        println!(
            "generation_summary\ttrue_path_structural_present_records\t{}",
            metrics.cleavage_graph_true_path_structural_present
        );
        println!(
            "generation_summary\ttrue_path_structural_records\t{}",
            metrics.cleavage_graph_structural_records
        );
        println!(
            "generation_summary\ttrue_path_node_coverage\t{:.6}",
            metrics.cleavage_graph_true_nodes_present as f64
                / metrics.cleavage_graph_true_nodes_total.max(1) as f64
        );
        println!(
            "generation_summary\ttrue_path_edge_coverage\t{:.6}",
            metrics.cleavage_graph_true_edges_present as f64
                / metrics.cleavage_graph_true_edges_total.max(1) as f64
        );
        println!(
            "generation_summary\tstructured_true_path_top1_records\t{}",
            metrics.cleavage_graph_structured_true_path_top1
        );
        println!(
            "generation_summary\tstructured_true_path_top8_records\t{}",
            metrics.cleavage_graph_structured_true_path_top8
        );
        println!(
            "generation_summary\tstructured_true_path_top32_records\t{}",
            metrics.cleavage_graph_structured_true_path_top32
        );
        println!(
            "generation_summary\tstructured_true_path_top64_records\t{}",
            metrics.cleavage_graph_structured_true_path_top64
        );
        println!(
            "generation_summary\tstructured_true_path_top1_rate_among_structural\t{:.6}",
            metrics.cleavage_graph_structured_true_path_top1 as f64
                / metrics.cleavage_graph_true_path_structural_present.max(1) as f64
        );
        println!(
            "generation_summary\tstructured_true_path_top8_rate_among_structural\t{:.6}",
            metrics.cleavage_graph_structured_true_path_top8 as f64
                / metrics.cleavage_graph_true_path_structural_present.max(1) as f64
        );
        println!(
            "generation_summary\tstructured_true_path_top32_rate_among_structural\t{:.6}",
            metrics.cleavage_graph_structured_true_path_top32 as f64
                / metrics.cleavage_graph_true_path_structural_present.max(1) as f64
        );
        println!(
            "generation_summary\tstructured_true_path_top64_rate_among_structural\t{:.6}",
            metrics.cleavage_graph_structured_true_path_top64 as f64
                / metrics.cleavage_graph_true_path_structural_present.max(1) as f64
        );
    }
    println!(
        "generation_summary\tneural_top1_peptidoform_exact\t{:.6}",
        metrics.neural_top1_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\tneural_top1_sequence_exact\t{:.6}",
        metrics.neural_top1_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\tneural_top1_il_sequence_exact\t{:.6}",
        metrics.neural_top1_il_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\thybrid_top1_peptidoform_exact\t{:.6}",
        metrics.hybrid_top1_peptidoform_exact as f64 / records
    );
    println!(
        "generation_summary\thybrid_top1_sequence_exact\t{:.6}",
        metrics.hybrid_top1_sequence_exact as f64 / records
    );
    println!(
        "generation_summary\thybrid_top1_il_sequence_exact\t{:.6}",
        metrics.hybrid_top1_il_sequence_exact as f64 / records
    );
    if metrics.causal_scored_records > 0 {
        let causal_records = metrics.causal_scored_records as f64;
        println!(
            "generation_summary\tcausal_top1_peptidoform_exact\t{:.6}",
            metrics.causal_top1_peptidoform_exact as f64 / causal_records
        );
        println!(
            "generation_summary\tcausal_top1_sequence_exact\t{:.6}",
            metrics.causal_top1_sequence_exact as f64 / causal_records
        );
        println!(
            "generation_summary\tcausal_top1_il_sequence_exact\t{:.6}",
            metrics.causal_top1_il_sequence_exact as f64 / causal_records
        );
        println!(
            "generation_summary\tfragment_causal_top1_peptidoform_exact\t{:.6}",
            metrics.fragment_causal_top1_peptidoform_exact as f64 / causal_records
        );
        println!(
            "generation_summary\tfragment_causal_top1_sequence_exact\t{:.6}",
            metrics.fragment_causal_top1_sequence_exact as f64 / causal_records
        );
        println!(
            "generation_summary\tfragment_causal_top1_il_sequence_exact\t{:.6}",
            metrics.fragment_causal_top1_il_sequence_exact as f64 / causal_records
        );
        if reverse_causal_reranker.is_some()
            && iterative_refiner.is_none()
            && cleavage_graph_proposer.is_none()
            && !bidirectional_mitm
        {
            println!(
                "generation_summary\tfragment_bidirectional_causal_top1_peptidoform_exact\t{:.6}",
                metrics.fragment_bidirectional_causal_top1_peptidoform_exact as f64
                    / causal_records
            );
            println!(
                "generation_summary\tfragment_bidirectional_causal_top1_sequence_exact\t{:.6}",
                metrics.fragment_bidirectional_causal_top1_sequence_exact as f64 / causal_records
            );
            println!(
                "generation_summary\tfragment_bidirectional_causal_top1_il_sequence_exact\t{:.6}",
                metrics.fragment_bidirectional_causal_top1_il_sequence_exact as f64
                    / causal_records
            );
        }
        if causal_generation_beam_width > 0 {
            println!(
                "generation_summary\tcausal_beam_top1_peptidoform_exact\t{:.6}",
                metrics.causal_beam_top1_peptidoform_exact as f64 / causal_records
            );
            println!(
                "generation_summary\tcausal_beam_top1_sequence_exact\t{:.6}",
                metrics.causal_beam_top1_sequence_exact as f64 / causal_records
            );
            println!(
                "generation_summary\tcausal_beam_top1_il_sequence_exact\t{:.6}",
                metrics.causal_beam_top1_il_sequence_exact as f64 / causal_records
            );
        }
    }
    if metrics.best_abs_mass_error_records > 0 {
        println!(
            "generation_summary\tmean_best_abs_mass_error_da\t{:.6}",
            metrics.best_abs_mass_error_sum / metrics.best_abs_mass_error_records as f64
        );
    }
    println!(
        "generation_summary\tmass_valid_record_count\t{}",
        metrics.best_abs_mass_errors_mass_valid.len()
    );
    println!(
        "generation_summary\tno_mass_valid_record_count\t{}",
        metrics.best_abs_mass_errors_no_mass_valid.len()
    );
    if !metrics.best_abs_mass_errors_mass_valid.is_empty() {
        println!(
            "generation_summary\tmean_best_abs_mass_error_da_mass_valid_only\t{:.6}",
            mean(&metrics.best_abs_mass_errors_mass_valid)
        );
        println!(
            "generation_summary\tmedian_best_abs_mass_error_da_mass_valid_only\t{:.6}",
            median(&metrics.best_abs_mass_errors_mass_valid)
        );
    }
    if !metrics.best_abs_mass_errors_no_mass_valid.is_empty() {
        println!(
            "generation_summary\tmean_fallback_abs_mass_error_da_no_mass_valid\t{:.6}",
            mean(&metrics.best_abs_mass_errors_no_mass_valid)
        );
    }
    println!(
        "generation_summary\tmean_target_fragment_score\t{:.6}",
        metrics.target_fragment_score_sum / records
    );
    println!(
        "generation_summary\tmean_top1_fragment_score\t{:.6}",
        metrics.top1_fragment_score_sum / records
    );
    println!(
        "generation_summary\tmean_target_matched_cleavages\t{:.4}",
        metrics.target_matched_cleavages as f64 / records
    );
    println!(
        "generation_summary\tmean_top1_matched_cleavages\t{:.4}",
        metrics.top1_matched_cleavages as f64 / records
    );
    println!(
        "generation_summary\tmean_target_neural_all_mask_log_probability\t{:.6}",
        metrics.target_neural_all_mask_log_probability_sum / records
    );
    println!(
        "generation_summary\tmean_target_neural_length_log_probability\t{:.6}",
        metrics.target_neural_length_log_probability_sum / records
    );
    println!(
        "generation_summary\tmean_fragment_top1_neural_all_mask_log_probability\t{:.6}",
        metrics.fragment_top1_neural_all_mask_log_probability_sum / records
    );
    println!(
        "generation_summary\tmean_neural_top1_neural_all_mask_log_probability\t{:.6}",
        metrics.neural_top1_neural_all_mask_log_probability_sum / records
    );
    println!(
        "generation_summary\tmean_hybrid_top1_neural_all_mask_log_probability\t{:.6}",
        metrics.hybrid_top1_neural_all_mask_log_probability_sum / records
    );
    if metrics.causal_scored_records > 0 {
        let causal_records = metrics.causal_scored_records as f64;
        println!(
            "generation_summary\tmean_target_ar_total_log_probability\t{:.6}",
            metrics.target_ar_total_log_probability_sum / causal_records
        );
        println!(
            "generation_summary\tmean_target_ar_mean_log_probability\t{:.6}",
            metrics.target_ar_mean_log_probability_sum / causal_records
        );
        println!(
            "generation_summary\tmean_fragment_top1_ar_total_log_probability\t{:.6}",
            metrics.fragment_top1_ar_total_log_probability_sum / causal_records
        );
        println!(
            "generation_summary\tmean_causal_top1_ar_total_log_probability\t{:.6}",
            metrics.causal_top1_ar_total_log_probability_sum / causal_records
        );
        println!(
            "generation_summary\tmean_fragment_causal_top1_ar_total_log_probability\t{:.6}",
            metrics.fragment_causal_top1_ar_total_log_probability_sum / causal_records
        );
    }
    println!("generation_candidates\t{}", output_tsv.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reverse_generate(
    model: &PeptideSpectrumDiffusionModel,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    active_lengths: &[usize],
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    mass_beam_width: usize,
    final_candidates_per_chain: usize,
    fragment_tolerance_ppm: f64,
    spectral_beam_weight: f64,
    temperature: f64,
    rng: &mut GenerationRng,
    device: &Device,
) -> Result<(Vec<Vec<u32>>, Vec<f64>, Vec<f64>, Vec<usize>)> {
    let batch = active_lengths.len();
    let mut rows = Vec::<Vec<u32>>::with_capacity(batch);
    for &active_length in active_lengths {
        let mut row = vec![FOUNDATION_DIFFUSION_PAD; config.max_tokens];
        for token in row.iter_mut().take(active_length.saturating_sub(1)) {
            *token = FOUNDATION_DIFFUSION_MASK;
        }
        row[active_length - 1] = FOUNDATION_DIFFUSION_EOS;
        rows.push(row);
    }
    let spectra = vec![spectrum.clone(); batch];
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let record_refs = vec![record; batch];
    let precursor = precursor_context(&record_refs, device)?;
    let mut reverse_log_probability = vec![0.0f64; batch];

    // Stochastically traverse t=T..2. The final t=1 posterior equals the model's
    // predicted x0 distribution, so v0.11.5 replaces independent token sampling
    // at that final step with a global precursor-mass-aware beam over the entire
    // peptide/PTM row.
    for timestep in (2..=config.diffusion_steps).rev() {
        let diffusion =
            diffusion_collator.collate_inference_tokens(&rows, active_lengths, timestep, device)?;
        let output = model.forward_t(&diffusion, &spectrum_batch, &precursor, false)?;
        let logits = output.token_logits.to_vec3::<f32>()?;

        for batch_index in 0..batch {
            let active_length = active_lengths[batch_index];
            for position in 0..active_length.saturating_sub(1) {
                let x0_probabilities =
                    clean_x0_probabilities(&logits[batch_index][position], position, temperature);
                let mut reverse = foundation_diffusion_reverse_probabilities(
                    config,
                    rows[batch_index][position],
                    &x0_probabilities,
                    timestep,
                )
                .map_err(anyhow::Error::msg)?;
                apply_nonterminal_constraints(&mut reverse, position, timestep);
                let selected = sample_probability(&reverse, rng);
                let selected_probability = reverse[selected].max(1e-300);
                reverse_log_probability[batch_index] += selected_probability.ln();
                rows[batch_index][position] = selected as u32;
            }
            rows[batch_index][active_length - 1] = FOUNDATION_DIFFUSION_EOS;
        }
    }

    let final_diffusion =
        diffusion_collator.collate_inference_tokens(&rows, active_lengths, 1, device)?;
    let final_output = model.forward_t(&final_diffusion, &spectrum_batch, &precursor, false)?;
    let final_logits = final_output.token_logits.to_vec3::<f32>()?;
    let observed_peaks = normalized_observed_peaks(spectrum);
    let fragment_charge = record
        .context
        .charge
        .unwrap_or(1)
        .unsigned_abs()
        .clamp(1, 2) as usize;
    let mut finalized_rows = Vec::new();
    let mut finalized_scores = Vec::new();
    let mut finalized_fragment_scores = Vec::new();
    let mut finalized_matched_cleavages = Vec::new();

    for batch_index in 0..batch {
        let active_length = active_lengths[batch_index];
        let finalized = mass_guided_final_beam(
            &final_logits[batch_index],
            active_length,
            target_neutral_mass,
            mass_tolerance_da,
            mass_beam_width,
            final_candidates_per_chain,
            &observed_peaks,
            fragment_charge,
            fragment_tolerance_ppm,
            spectral_beam_weight,
            temperature,
            config.max_tokens,
        );
        for (row, final_log_probability, fragment_score, matched_cleavages) in finalized {
            finalized_rows.push(row);
            finalized_scores.push(reverse_log_probability[batch_index] + final_log_probability);
            finalized_fragment_scores.push(fragment_score);
            finalized_matched_cleavages.push(matched_cleavages);
        }
    }
    Ok((
        finalized_rows,
        finalized_scores,
        finalized_fragment_scores,
        finalized_matched_cleavages,
    ))
}

#[allow(clippy::too_many_arguments)]
fn causal_prefix_mass_beam(
    causal: &CausalReranker,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    beam_width: usize,
    final_candidates: usize,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    causal_weight: f64,
    device: &Device,
) -> Result<Vec<CausalBeamCandidate>> {
    let Some(target) = target_neutral_mass.filter(|value| value.is_finite()) else {
        return Ok(Vec::new());
    };
    if beam_width == 0 || final_candidates == 0 {
        return Ok(Vec::new());
    }

    let max_token_mass = (FOUNDATION_DIFFUSION_EOS + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
        .filter_map(foundation_diffusion_token_mass_da)
        .filter(|mass| mass.is_finite() && *mass > 0.0)
        .fold(0.0f64, f64::max);
    if !(max_token_mass > 0.0 && max_token_mass.is_finite()) {
        anyhow::bail!("causal generation could not determine a positive maximum token mass");
    }

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
    let mut completed = HashMap::<Vec<u32>, CausalBeamCandidate>::new();

    // v0.12.5 performance lane: spectrum and precursor context are invariant
    // across every prefix expansion for this record, so encode them once and
    // broadcast the cached memory across each changing beam batch.
    let spectrum_batch = spectrum_collator.collate(std::slice::from_ref(spectrum), device)?;
    let precursor = precursor_context(&[record], device)?;
    let causal_context = causal
        .model
        .prepare_context(&spectrum_batch, &precursor, false)?;

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
            .forward_next_t_with_context(&input, &causal_context, false)?
            .to_vec2::<f32>()?;

        let mut binned = HashMap::<(i64, u32), CausalBeamState>::new();
        for (state_index, state) in beam.iter().enumerate() {
            let next_logits = &logits[state_index];
            let abs_mass_error = (state.neutral_mass - target).abs();
            if state.residue_count > 0 && abs_mass_error <= mass_tolerance_da {
                let eos_log_probability =
                    selected_log_softmax(next_logits, FOUNDATION_DIFFUSION_EOS as usize)?;
                let ar_total_log_probability = state.ar_total_log_probability + eos_log_probability;
                let fragment_causal_score = foundation_fragment_causal_rerank_score(
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
                    fragment_causal_score,
                    abs_mass_error_da: abs_mass_error,
                };
                completed
                    .entry(row)
                    .and_modify(|existing| {
                        if candidate.fragment_causal_score > existing.fragment_causal_score {
                            *existing = candidate.clone();
                        }
                    })
                    .or_insert(candidate);
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
                let priority = foundation_fragment_causal_rerank_score(
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
                match binned.get_mut(&key) {
                    Some(existing) if candidate.priority > existing.priority => {
                        *existing = candidate;
                    }
                    None => {
                        binned.insert(key, candidate);
                    }
                    _ => {}
                }
            }
        }

        beam = binned.into_values().collect();
        beam.sort_by(|left, right| right.priority.total_cmp(&left.priority));
        beam.truncate(beam_width);
    }

    let mut completed: Vec<CausalBeamCandidate> = completed.into_values().collect();
    completed.sort_by(|left, right| {
        right
            .fragment_causal_score
            .total_cmp(&left.fragment_causal_score)
            .then_with(|| left.abs_mass_error_da.total_cmp(&right.abs_mass_error_da))
            .then_with(|| {
                right
                    .ar_total_log_probability
                    .total_cmp(&left.ar_total_log_probability)
            })
    });
    completed.truncate(final_candidates);
    Ok(completed)
}

/// C-terminal-to-N-terminal counterpart of the frozen N->C causal proposal beam.
///
/// Prefix tokens are kept in the reverse-causal residue-unit representation
/// during search. Completed rows are canonicalized back to N->C before entering
/// the shared candidate pool. Fragment evidence scores the same physical
/// cleavage set by converting the emitted suffix mass into its complementary
/// canonical prefix mass.
#[allow(clippy::too_many_arguments)]
fn reverse_causal_prefix_mass_beam(
    causal: &CausalReranker,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    beam_width: usize,
    final_candidates: usize,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    causal_weight: f64,
    device: &Device,
) -> Result<Vec<CausalBeamCandidate>> {
    let Some(target) = target_neutral_mass.filter(|value| value.is_finite()) else {
        return Ok(Vec::new());
    };
    if beam_width == 0 || final_candidates == 0 {
        return Ok(Vec::new());
    }

    let max_token_mass = (FOUNDATION_DIFFUSION_EOS + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
        .filter_map(foundation_diffusion_token_mass_da)
        .filter(|mass| mass.is_finite() && *mass > 0.0)
        .fold(0.0f64, f64::max);
    if !(max_token_mass > 0.0 && max_token_mass.is_finite()) {
        anyhow::bail!(
            "reverse causal generation could not determine a positive maximum token mass"
        );
    }

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
    let mut completed = HashMap::<Vec<u32>, CausalBeamCandidate>::new();

    let spectrum_batch = spectrum_collator.collate(std::slice::from_ref(spectrum), device)?;
    let precursor = precursor_context(&[record], device)?;
    let causal_context = causal
        .model
        .prepare_context(&spectrum_batch, &precursor, false)?;

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
            .forward_next_t_with_context(&input, &causal_context, false)?
            .to_vec2::<f32>()?;

        let mut binned = HashMap::<(i64, u32), CausalBeamState>::new();
        for (state_index, state) in beam.iter().enumerate() {
            let next_logits = &logits[state_index];
            let abs_mass_error = (state.neutral_mass - target).abs();
            if state.residue_count > 0 && abs_mass_error <= mass_tolerance_da {
                let eos_log_probability =
                    selected_log_softmax(next_logits, FOUNDATION_DIFFUSION_EOS as usize)?;
                let ar_total_log_probability = state.ar_total_log_probability + eos_log_probability;
                let fragment_causal_score = foundation_fragment_causal_rerank_score(
                    state.fragment_score,
                    ar_total_log_probability,
                    causal_weight,
                );
                let mut reverse_row = vec![FOUNDATION_DIFFUSION_PAD; config.max_tokens];
                for (token_position, &token) in state.prefix.iter().enumerate() {
                    reverse_row[token_position] = token;
                }
                reverse_row[state.prefix.len()] = FOUNDATION_DIFFUSION_EOS;
                let canonical_row = foundation_canonicalize_reverse_causal_token_row(&reverse_row)
                    .map_err(anyhow::Error::msg)?;
                let candidate = CausalBeamCandidate {
                    tokens: canonical_row.clone(),
                    ar_total_log_probability,
                    fragment_score: state.fragment_score,
                    matched_cleavages: state.matched_cleavages,
                    fragment_causal_score,
                    abs_mass_error_da: abs_mass_error,
                };
                completed
                    .entry(canonical_row)
                    .and_modify(|existing| {
                        if candidate.fragment_causal_score > existing.fragment_causal_score {
                            *existing = candidate.clone();
                        }
                    })
                    .or_insert(candidate);
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
                    let global_nterm_mass = if state.prefix.first().copied()
                        == Some(FOUNDATION_DIFFUSION_NTERM_ACETYL)
                    {
                        foundation_diffusion_token_mass_da(FOUNDATION_DIFFUSION_NTERM_ACETYL)
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let canonical_prefix_mass_without_water =
                        target - state.neutral_mass + global_nterm_mass;
                    let evidence = cleavage_fragment_evidence(
                        canonical_prefix_mass_without_water,
                        target,
                        observed_peaks,
                        max_fragment_charge,
                        fragment_tolerance_ppm,
                    );
                    fragment_score += evidence.score;
                    matched_cleavages += usize::from(evidence.matched);
                }
                let residue_count = state.residue_count + usize::from(is_residue);
                let priority = foundation_fragment_causal_rerank_score(
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
                match binned.get_mut(&key) {
                    Some(existing) if candidate.priority > existing.priority => {
                        *existing = candidate;
                    }
                    None => {
                        binned.insert(key, candidate);
                    }
                    _ => {}
                }
            }
        }

        beam = binned.into_values().collect();
        beam.sort_by(|left, right| right.priority.total_cmp(&left.priority));
        beam.truncate(beam_width);
    }

    let mut completed: Vec<CausalBeamCandidate> = completed.into_values().collect();
    completed.sort_by(|left, right| {
        right
            .fragment_causal_score
            .total_cmp(&left.fragment_causal_score)
            .then_with(|| left.abs_mass_error_da.total_cmp(&right.abs_mass_error_da))
            .then_with(|| {
                right
                    .ar_total_log_probability
                    .total_cmp(&left.ar_total_log_probability)
            })
    });
    completed.truncate(final_candidates);
    Ok(completed)
}

#[allow(clippy::too_many_arguments)]
fn causal_midpoint_partial_beam(
    causal: &CausalReranker,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    beam_width: usize,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    causal_weight: f64,
    direction: MitmDirection,
    device: &Device,
) -> Result<Vec<MitmPartialState>> {
    let Some(target_neutral_mass) = target_neutral_mass.filter(|value| value.is_finite()) else {
        return Ok(Vec::new());
    };
    if beam_width == 0 || target_neutral_mass <= FOUNDATION_PEPTIDE_WATER_MASS_DA {
        return Ok(Vec::new());
    }

    let target_residue_mass = target_neutral_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA;
    let midpoint_mass = 0.5 * target_residue_mass;
    let max_token_mass = (FOUNDATION_DIFFUSION_EOS + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32)
        .filter_map(foundation_diffusion_token_mass_da)
        .filter(|mass| mass.is_finite() && *mass > 0.0)
        .fold(0.0f64, f64::max);
    if !(max_token_mass > 0.0 && max_token_mass.is_finite()) {
        anyhow::bail!("MITM generation could not determine a positive maximum token mass");
    }
    let nterm_acetyl_mass = foundation_diffusion_token_mass_da(FOUNDATION_DIFFUSION_NTERM_ACETYL)
        .unwrap_or(0.0)
        .abs();
    // Fixed v0.13.19 midpoint frontier: retain states within at most one
    // maximum residue/token step plus the possible global N-terminal acetyl
    // marker on either side of the exact 50% residue-mass midpoint.
    let midpoint_window = max_token_mass + nterm_acetyl_mass + mass_tolerance_da;
    let maximum_partial_mass = midpoint_mass + midpoint_window;
    let mass_bin_width = mass_tolerance_da.max(0.05);

    let mut beam = vec![MitmPartialState {
        tokens: Vec::new(),
        assigned_mass_da: 0.0,
        ar_total_log_probability: 0.0,
        fragment_score: 0.0,
        matched_cleavages: 0,
        residue_count: 0,
        priority: 0.0,
    }];
    let mut frontier = HashMap::<Vec<u32>, MitmPartialState>::new();

    let spectrum_batch = spectrum_collator.collate(std::slice::from_ref(spectrum), device)?;
    let precursor = precursor_context(&[record], device)?;
    let causal_context = causal
        .model
        .prepare_context(&spectrum_batch, &precursor, false)?;

    for position in 0..config.max_tokens.saturating_sub(1) {
        if beam.is_empty() {
            break;
        }
        debug_assert!(beam.iter().all(|state| state.tokens.len() == position));
        let prefixes: Vec<Vec<u32>> = beam.iter().map(|state| state.tokens.clone()).collect();
        let input = causal
            .collator
            .collate_compact_prefix_rows(&prefixes, device)?;
        let logits = causal
            .model
            .forward_next_t_with_context(&input, &causal_context, false)?
            .to_vec2::<f32>()?;

        let mut binned = HashMap::<(i64, u32), MitmPartialState>::new();
        for (state_index, state) in beam.iter().enumerate() {
            let next_logits = &logits[state_index];
            let mut token_order: Vec<usize> =
                (FOUNDATION_DIFFUSION_EOS as usize + 1..FOUNDATION_DIFFUSION_VOCAB_SIZE).collect();
            token_order.sort_by(|&left, &right| next_logits[right].total_cmp(&next_logits[left]));

            for token_index in token_order {
                if !next_logits[token_index].is_finite() {
                    continue;
                }
                let token = token_index as u32;
                if !mass_beam_token_allowed(
                    &state.tokens,
                    token,
                    position,
                    config.max_tokens.saturating_sub(1),
                ) {
                    continue;
                }
                let Some(token_mass) = foundation_diffusion_token_mass_da(token) else {
                    continue;
                };
                let assigned_mass_da = state.assigned_mass_da + token_mass;
                if assigned_mass_da > maximum_partial_mass + mass_tolerance_da {
                    continue;
                }

                let token_log_probability = selected_log_softmax(next_logits, token_index)?;
                let ar_total_log_probability =
                    state.ar_total_log_probability + token_log_probability;
                let is_residue = foundation_diffusion_token_residue(token).is_some();
                let mut fragment_score = state.fragment_score;
                let mut matched_cleavages = state.matched_cleavages;
                if is_residue && state.residue_count > 0 {
                    let canonical_prefix_mass_without_water = match direction {
                        MitmDirection::NToC => state.assigned_mass_da,
                        MitmDirection::CToN => {
                            let global_nterm_mass = if state.tokens.first().copied()
                                == Some(FOUNDATION_DIFFUSION_NTERM_ACETYL)
                            {
                                foundation_diffusion_token_mass_da(
                                    FOUNDATION_DIFFUSION_NTERM_ACETYL,
                                )
                                .unwrap_or(0.0)
                            } else {
                                0.0
                            };
                            target_residue_mass - state.assigned_mass_da + global_nterm_mass
                        }
                    };
                    let evidence = cleavage_fragment_evidence(
                        canonical_prefix_mass_without_water,
                        target_neutral_mass,
                        observed_peaks,
                        max_fragment_charge,
                        fragment_tolerance_ppm,
                    );
                    fragment_score += evidence.score;
                    matched_cleavages += usize::from(evidence.matched);
                }
                let residue_count = state.residue_count + usize::from(is_residue);
                let priority = foundation_fragment_causal_rerank_score(
                    fragment_score,
                    ar_total_log_probability,
                    causal_weight,
                );
                let mut tokens = state.tokens.clone();
                tokens.push(token);
                let candidate = MitmPartialState {
                    tokens: tokens.clone(),
                    assigned_mass_da,
                    ar_total_log_probability,
                    fragment_score,
                    matched_cleavages,
                    residue_count,
                    priority,
                };

                if residue_count > 0 && (assigned_mass_da - midpoint_mass).abs() <= midpoint_window
                {
                    frontier
                        .entry(tokens.clone())
                        .and_modify(|existing| {
                            if candidate.priority > existing.priority {
                                *existing = candidate.clone();
                            }
                        })
                        .or_insert_with(|| candidate.clone());
                }

                let mass_bin = (assigned_mass_da / mass_bin_width).round() as i64;
                let key = (mass_bin, token);
                match binned.get_mut(&key) {
                    Some(existing) if candidate.priority > existing.priority => {
                        *existing = candidate;
                    }
                    None => {
                        binned.insert(key, candidate);
                    }
                    _ => {}
                }
            }
        }

        beam = binned.into_values().collect();
        beam.sort_by(|left, right| {
            right
                .priority
                .total_cmp(&left.priority)
                .then_with(|| {
                    (left.assigned_mass_da - midpoint_mass)
                        .abs()
                        .total_cmp(&(right.assigned_mass_da - midpoint_mass).abs())
                })
                .then_with(|| left.tokens.cmp(&right.tokens))
        });
        beam.truncate(beam_width);
    }

    let mut frontier: Vec<MitmPartialState> = frontier.into_values().collect();
    frontier.sort_by(|left, right| {
        (left.assigned_mass_da - midpoint_mass)
            .abs()
            .total_cmp(&(right.assigned_mass_da - midpoint_mass).abs())
            .then_with(|| right.priority.total_cmp(&left.priority))
            .then_with(|| left.tokens.cmp(&right.tokens))
    });
    Ok(frontier)
}

fn mitm_reverse_partial_to_canonical_suffix(
    reverse_tokens: &[u32],
    max_tokens: usize,
) -> Result<(bool, Vec<u32>)> {
    if reverse_tokens.is_empty() || reverse_tokens.len() + 1 > max_tokens {
        anyhow::bail!("invalid reverse MITM partial token width");
    }
    let mut row = vec![FOUNDATION_DIFFUSION_PAD; max_tokens];
    for (position, &token) in reverse_tokens.iter().enumerate() {
        row[position] = token;
    }
    row[reverse_tokens.len()] = FOUNDATION_DIFFUSION_EOS;
    let canonical =
        foundation_canonicalize_reverse_causal_token_row(&row).map_err(anyhow::Error::msg)?;
    let eos = canonical
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
        .ok_or_else(|| anyhow::anyhow!("canonical reverse MITM partial is missing EOS"))?;
    let active = &canonical[..eos];
    let has_nterm_acetyl = active.first().copied() == Some(FOUNDATION_DIFFUSION_NTERM_ACETYL);
    let body = active[usize::from(has_nterm_acetyl)..].to_vec();
    if !body
        .iter()
        .any(|&token| foundation_diffusion_token_residue(token).is_some())
    {
        anyhow::bail!("reverse MITM partial canonicalized without a residue");
    }
    Ok((has_nterm_acetyl, body))
}

fn mitm_legacy_join_order(left: &MitmJoinedCandidate, right: &MitmJoinedCandidate) -> Ordering {
    right
        .proposal_score
        .total_cmp(&left.proposal_score)
        .then_with(|| {
            left.join_mass_error_da
                .abs()
                .total_cmp(&right.join_mass_error_da.abs())
        })
        .then_with(|| left.tokens.cmp(&right.tokens))
}

fn mitm_evidence_join_order(left: &MitmJoinedCandidate, right: &MitmJoinedCandidate) -> Ordering {
    right
        .selector_score
        .total_cmp(&left.selector_score)
        .then_with(|| {
            right
                .selector_matched_cleavages
                .cmp(&left.selector_matched_cleavages)
        })
        .then_with(|| {
            right
                .selector_fragment_score
                .total_cmp(&left.selector_fragment_score)
        })
        .then_with(|| {
            left.join_mass_error_da
                .abs()
                .total_cmp(&right.join_mass_error_da.abs())
        })
        .then_with(|| right.proposal_score.total_cmp(&left.proposal_score))
        .then_with(|| left.tokens.cmp(&right.tokens))
}

fn mitm_partial_priority_ranks(states: &[MitmPartialState]) -> HashMap<Vec<u32>, usize> {
    let mut indices: Vec<usize> = (0..states.len()).collect();
    indices.sort_by(|&left, &right| {
        states[right]
            .priority
            .total_cmp(&states[left].priority)
            .then_with(|| states[left].tokens.cmp(&states[right].tokens))
    });
    let mut ranks = HashMap::with_capacity(indices.len());
    for (zero_based_rank, index) in indices.into_iter().enumerate() {
        ranks.insert(states[index].tokens.clone(), zero_based_rank + 1);
    }
    ranks
}

fn mitm_retain_top_candidates(
    mut candidates: Vec<MitmJoinedCandidate>,
    max_candidates: usize,
    evidence_aware: bool,
) -> Vec<MitmJoinedCandidate> {
    let order: fn(&MitmJoinedCandidate, &MitmJoinedCandidate) -> Ordering = if evidence_aware {
        mitm_evidence_join_order
    } else {
        mitm_legacy_join_order
    };
    if candidates.len() <= max_candidates {
        candidates.sort_by(order);
        return candidates;
    }
    candidates.select_nth_unstable_by(max_candidates, order);
    candidates.truncate(max_candidates);
    candidates.sort_by(order);
    candidates
}

fn mitm_retain_v01323_two_view_candidates(
    candidates: &[MitmJoinedCandidate],
    max_candidates: usize,
) -> Vec<MitmJoinedCandidate> {
    if max_candidates == 0 || candidates.is_empty() {
        return Vec::new();
    }
    if candidates.len() <= max_candidates {
        let mut selected = candidates.to_vec();
        selected.sort_by(mitm_evidence_join_order);
        return selected;
    }

    // Frozen v0.13.23 policy: split the fixed 256-candidate budget evenly
    // between the accepted v0.13.20 evidence-aware view and the complementary
    // join-seam fragment view.  Canonical identities are already unique in the
    // pre-cap pool.  Duplicates across views consume one slot; any resulting
    // deficit is backfilled strictly from the remaining v0.13.20 ordering.
    let evidence_quota = max_candidates / 2;
    let seam_quota = max_candidates.saturating_sub(evidence_quota);

    // Keep the top max_candidates evidence indices because at most seam_quota
    // overlaps can need backfilling, so this contains every possible backfill.
    let mut evidence_indices: Vec<usize> = (0..candidates.len()).collect();
    if evidence_indices.len() > max_candidates {
        evidence_indices.select_nth_unstable_by(max_candidates, |&left, &right| {
            mitm_evidence_join_order(&candidates[left], &candidates[right])
        });
        evidence_indices.truncate(max_candidates);
    }
    evidence_indices
        .sort_by(|&left, &right| mitm_evidence_join_order(&candidates[left], &candidates[right]));

    let mut seam_indices: Vec<usize> = (0..candidates.len()).collect();
    if seam_indices.len() > seam_quota {
        seam_indices.select_nth_unstable_by(seam_quota, |&left, &right| {
            mitm_component_seam_fragment_order(&candidates[left], &candidates[right])
        });
        seam_indices.truncate(seam_quota);
    }
    seam_indices.sort_by(|&left, &right| {
        mitm_component_seam_fragment_order(&candidates[left], &candidates[right])
    });

    let mut selected_indices = HashSet::<usize>::with_capacity(max_candidates);
    for &index in evidence_indices.iter().take(evidence_quota) {
        selected_indices.insert(index);
    }
    for &index in seam_indices.iter().take(seam_quota) {
        selected_indices.insert(index);
    }
    for &index in &evidence_indices {
        if selected_indices.len() >= max_candidates {
            break;
        }
        selected_indices.insert(index);
    }

    let mut selected: Vec<MitmJoinedCandidate> = selected_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect();
    selected.sort_by(mitm_evidence_join_order);
    selected
}

#[allow(clippy::too_many_arguments)]
fn bidirectional_mitm_join(
    prefix_states: &[MitmPartialState],
    suffix_states: &[MitmPartialState],
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    max_tokens: usize,
    vocabulary: FoundationDiffusionVocabulary,
    max_joined_candidates: usize,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    causal_weight: f64,
    evidence_aware: bool,
    final_two_view: bool,
    return_precap_audit_pool: bool,
) -> Result<(
    usize,
    Vec<MitmJoinedCandidate>,
    Vec<MitmJoinedCandidate>,
    Vec<MitmJoinedCandidate>,
    Vec<MitmJoinedCandidate>,
)> {
    let Some(target_neutral_mass) = target_neutral_mass.filter(|value| value.is_finite()) else {
        return Ok((0, Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    };
    if target_neutral_mass <= FOUNDATION_PEPTIDE_WATER_MASS_DA || max_joined_candidates == 0 {
        return Ok((0, Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    let target_residue_mass = target_neutral_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA;
    let nterm_acetyl_mass =
        foundation_diffusion_token_mass_da(FOUNDATION_DIFFUSION_NTERM_ACETYL).unwrap_or(0.0);
    let bin_width = mass_tolerance_da.max(0.05);
    let prefix_priority_ranks = mitm_partial_priority_ranks(prefix_states);
    let suffix_priority_ranks = mitm_partial_priority_ranks(suffix_states);

    #[derive(Debug, Clone)]
    struct SuffixEntry {
        state: MitmPartialState,
        has_nterm_acetyl: bool,
        canonical_body: Vec<u32>,
    }

    let mut suffix_bins = HashMap::<(bool, i64), Vec<SuffixEntry>>::new();
    for state in suffix_states {
        let Ok((has_nterm_acetyl, canonical_body)) =
            mitm_reverse_partial_to_canonical_suffix(&state.tokens, max_tokens)
        else {
            continue;
        };
        let bin = (state.assigned_mass_da / bin_width).floor() as i64;
        suffix_bins
            .entry((has_nterm_acetyl, bin))
            .or_default()
            .push(SuffixEntry {
                state: state.clone(),
                has_nterm_acetyl,
                canonical_body,
            });
    }

    let mut joined = HashMap::<Vec<u32>, MitmJoinedCandidate>::new();
    for prefix in prefix_states {
        let prefix_has_nterm =
            prefix.tokens.first().copied() == Some(FOUNDATION_DIFFUSION_NTERM_ACETYL);
        let prefix_body = &prefix.tokens[usize::from(prefix_has_nterm)..];
        if !prefix_body
            .iter()
            .any(|&token| foundation_diffusion_token_residue(token).is_some())
        {
            continue;
        }

        for suffix_has_nterm in [false, true] {
            let duplicate_global_mass = if prefix_has_nterm && suffix_has_nterm {
                nterm_acetyl_mass
            } else {
                0.0
            };
            let required_suffix_mass =
                target_residue_mass - prefix.assigned_mass_da + duplicate_global_mass;
            if !(required_suffix_mass > 0.0 && required_suffix_mass.is_finite()) {
                continue;
            }
            let min_bin = ((required_suffix_mass - mass_tolerance_da) / bin_width).floor() as i64;
            let max_bin = ((required_suffix_mass + mass_tolerance_da) / bin_width).floor() as i64;
            for bin in min_bin..=max_bin {
                let Some(entries) = suffix_bins.get(&(suffix_has_nterm, bin)) else {
                    continue;
                };
                for suffix in entries {
                    debug_assert_eq!(suffix.has_nterm_acetyl, suffix_has_nterm);
                    let combined_mass = prefix.assigned_mass_da + suffix.state.assigned_mass_da
                        - duplicate_global_mass;
                    let join_mass_error_da = combined_mass - target_residue_mass;
                    if join_mass_error_da.abs() > mass_tolerance_da {
                        continue;
                    }

                    let global_nterm = prefix_has_nterm || suffix_has_nterm;
                    let mut active = Vec::<u32>::with_capacity(
                        usize::from(global_nterm)
                            + prefix_body.len()
                            + suffix.canonical_body.len()
                            + 1,
                    );
                    if global_nterm {
                        active.push(FOUNDATION_DIFFUSION_NTERM_ACETYL);
                    }
                    active.extend_from_slice(prefix_body);
                    active.extend_from_slice(&suffix.canonical_body);
                    if active.len() + 1 > max_tokens {
                        continue;
                    }
                    let mut row = vec![FOUNDATION_DIFFUSION_PAD; max_tokens];
                    for (position, &token) in active.iter().enumerate() {
                        row[position] = token;
                    }
                    row[active.len()] = FOUNDATION_DIFFUSION_EOS;
                    if vocabulary.decode(&row).is_err() {
                        continue;
                    }
                    let mut exact_token_mass = 0.0f64;
                    let mut token_mass_valid = true;
                    for &token in &active {
                        let Some(mass) = foundation_diffusion_token_mass_da(token) else {
                            token_mass_valid = false;
                            break;
                        };
                        exact_token_mass += mass;
                    }
                    if !token_mass_valid {
                        continue;
                    }
                    let exact_join_error = exact_token_mass - target_residue_mass;
                    if exact_join_error.abs() > mass_tolerance_da {
                        continue;
                    }

                    let proposal_score = prefix.priority + suffix.state.priority;
                    let prefix_ar_mean =
                        prefix.ar_total_log_probability / prefix.tokens.len().max(1) as f64;
                    let suffix_ar_mean = suffix.state.ar_total_log_probability
                        / suffix.state.tokens.len().max(1) as f64;
                    let prefix_partial_rank = prefix_priority_ranks
                        .get(&prefix.tokens)
                        .copied()
                        .unwrap_or(usize::MAX);
                    let suffix_partial_rank = suffix_priority_ranks
                        .get(&suffix.state.tokens)
                        .copied()
                        .unwrap_or(usize::MAX);
                    let (
                        selector_fragment_score,
                        selector_matched_cleavages,
                        selector_score,
                        join_seam_fragment_score,
                    ) = if evidence_aware {
                        // The two partial searches have already scored every cleavage
                        // internal to their respective halves. Add only the join-seam
                        // cleavage to reconstruct complete-peptide fragment evidence exactly.
                        let seam_prefix_mass = prefix.assigned_mass_da
                            + if !prefix_has_nterm && suffix_has_nterm {
                                nterm_acetyl_mass
                            } else {
                                0.0
                            };
                        let seam = cleavage_fragment_evidence(
                            seam_prefix_mass,
                            target_neutral_mass,
                            observed_peaks,
                            max_fragment_charge,
                            fragment_tolerance_ppm,
                        );
                        let complete_fragment_score =
                            prefix.fragment_score + suffix.state.fragment_score + seam.score;
                        let complete_matched_cleavages = prefix
                            .matched_cleavages
                            .saturating_add(suffix.state.matched_cleavages)
                            .saturating_add(usize::from(seam.matched));
                        let bidirectional_partial_mean = 0.5 * (prefix_ar_mean + suffix_ar_mean);
                        (
                            complete_fragment_score,
                            complete_matched_cleavages,
                            complete_fragment_score + causal_weight * bidirectional_partial_mean,
                            seam.score,
                        )
                    } else {
                        (0.0, 0, proposal_score, 0.0)
                    };

                    joined
                        .entry(row.clone())
                        .and_modify(|existing| {
                            if proposal_score > existing.proposal_score {
                                existing.proposal_score = proposal_score;
                            }
                            if selector_score > existing.selector_score {
                                existing.selector_score = selector_score;
                                existing.selector_fragment_score = selector_fragment_score;
                                existing.selector_matched_cleavages = selector_matched_cleavages;
                                existing.selector_prefix_partial_rank = prefix_partial_rank;
                                existing.selector_suffix_partial_rank = suffix_partial_rank;
                                existing.selector_prefix_token_count = prefix.tokens.len();
                                existing.selector_suffix_token_count = suffix.state.tokens.len();
                                existing.selector_prefix_residue_count = prefix.residue_count;
                                existing.selector_suffix_residue_count = suffix.state.residue_count;
                                existing.selector_join_seam_fragment_score =
                                    join_seam_fragment_score;
                                existing.selector_prefix_ar_mean = prefix_ar_mean;
                                existing.selector_suffix_ar_mean = suffix_ar_mean;
                                existing.selector_prefix_ar_total = prefix.ar_total_log_probability;
                                existing.selector_suffix_ar_total =
                                    suffix.state.ar_total_log_probability;
                            }
                            existing.component_fragment_score = existing
                                .component_fragment_score
                                .max(selector_fragment_score);
                            existing.component_prefix_ar_mean =
                                existing.component_prefix_ar_mean.max(prefix_ar_mean);
                            existing.component_suffix_ar_mean =
                                existing.component_suffix_ar_mean.max(suffix_ar_mean);
                            existing.component_prefix_ar_total = existing
                                .component_prefix_ar_total
                                .max(prefix.ar_total_log_probability);
                            existing.component_suffix_ar_total = existing
                                .component_suffix_ar_total
                                .max(suffix.state.ar_total_log_probability);
                            existing.component_join_seam_fragment_score = existing
                                .component_join_seam_fragment_score
                                .max(join_seam_fragment_score);
                            existing.best_prefix_partial_rank =
                                existing.best_prefix_partial_rank.min(prefix_partial_rank);
                            existing.best_suffix_partial_rank =
                                existing.best_suffix_partial_rank.min(suffix_partial_rank);
                            if exact_join_error.abs() < existing.join_mass_error_da.abs() {
                                existing.join_mass_error_da = exact_join_error;
                            }
                        })
                        .or_insert(MitmJoinedCandidate {
                            tokens: row,
                            proposal_score,
                            selector_score,
                            selector_fragment_score,
                            selector_matched_cleavages,
                            join_mass_error_da: exact_join_error,
                            component_fragment_score: selector_fragment_score,
                            component_prefix_ar_mean: prefix_ar_mean,
                            component_suffix_ar_mean: suffix_ar_mean,
                            component_prefix_ar_total: prefix.ar_total_log_probability,
                            component_suffix_ar_total: suffix.state.ar_total_log_probability,
                            component_join_seam_fragment_score: join_seam_fragment_score,
                            best_prefix_partial_rank: prefix_partial_rank,
                            best_suffix_partial_rank: suffix_partial_rank,
                            selector_prefix_partial_rank: prefix_partial_rank,
                            selector_suffix_partial_rank: suffix_partial_rank,
                            selector_prefix_token_count: prefix.tokens.len(),
                            selector_suffix_token_count: suffix.state.tokens.len(),
                            selector_prefix_residue_count: prefix.residue_count,
                            selector_suffix_residue_count: suffix.state.residue_count,
                            selector_join_seam_fragment_score: join_seam_fragment_score,
                            selector_prefix_ar_mean: prefix_ar_mean,
                            selector_suffix_ar_mean: suffix_ar_mean,
                            selector_prefix_ar_total: prefix.ar_total_log_probability,
                            selector_suffix_ar_total: suffix.state.ar_total_log_probability,
                        });
                }
            }
        }
    }

    let unique_before_cap = joined.len();
    let all: Vec<MitmJoinedCandidate> = joined.into_values().collect();
    let precap_audit_pool = if return_precap_audit_pool {
        all.clone()
    } else {
        Vec::new()
    };
    if !evidence_aware {
        let selected = mitm_retain_top_candidates(all, max_joined_candidates, false);
        return Ok((
            unique_before_cap,
            selected,
            Vec::new(),
            Vec::new(),
            precap_audit_pool,
        ));
    }

    // Shadow the frozen v0.13.19 selector on the exact same join pool.  Only the
    // ordering changes in v0.13.20; beams, midpoint, chemistry and budget remain fixed.
    let mut legacy_indices: Vec<usize> = (0..all.len()).collect();
    if legacy_indices.len() > max_joined_candidates {
        legacy_indices.select_nth_unstable_by(max_joined_candidates, |&left, &right| {
            mitm_legacy_join_order(&all[left], &all[right])
        });
        legacy_indices.truncate(max_joined_candidates);
    }
    legacy_indices.sort_by(|&left, &right| mitm_legacy_join_order(&all[left], &all[right]));
    let legacy_shadow: Vec<MitmJoinedCandidate> = legacy_indices
        .into_iter()
        .map(|index| all[index].clone())
        .collect();

    if final_two_view {
        let v01320_shadow = mitm_retain_top_candidates(all.clone(), max_joined_candidates, true);
        let selected = mitm_retain_v01323_two_view_candidates(&all, max_joined_candidates);
        Ok((
            unique_before_cap,
            selected,
            legacy_shadow,
            v01320_shadow,
            precap_audit_pool,
        ))
    } else {
        let selected = mitm_retain_top_candidates(all, max_joined_candidates, true);
        Ok((
            unique_before_cap,
            selected,
            legacy_shadow,
            Vec::new(),
            precap_audit_pool,
        ))
    }
}

fn mitm_token_sequence_matches(
    tokens: &[u32],
    target_sequence: &str,
    normalize_isoleucine_leucine: bool,
) -> bool {
    let mut target = target_sequence.chars().map(|residue| {
        if normalize_isoleucine_leucine && (residue == 'I' || residue == 'L') {
            'J'
        } else {
            residue
        }
    });
    let mut observed = tokens
        .iter()
        .filter_map(|&token| foundation_diffusion_token_residue(token))
        .map(|residue| {
            if normalize_isoleucine_leucine && (residue == 'I' || residue == 'L') {
                'J'
            } else {
                residue
            }
        });
    loop {
        match (observed.next(), target.next()) {
            (None, None) => return true,
            (Some(left), Some(right)) if left == right => {}
            _ => return false,
        }
    }
}

fn mitm_oracle_ranks_with_order(
    candidates: &[MitmJoinedCandidate],
    order: fn(&MitmJoinedCandidate, &MitmJoinedCandidate) -> Ordering,
    target_tokens: &[u32],
    target_sequence: &str,
) -> MitmOracleRanks {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&left, &right| order(&candidates[left], &candidates[right]));
    let mut ranks = MitmOracleRanks::default();
    for (zero_based_rank, index) in indices.into_iter().enumerate() {
        let candidate = &candidates[index];
        let rank = zero_based_rank + 1;
        if ranks.peptidoform.is_none() && candidate.tokens.as_slice() == target_tokens {
            ranks.peptidoform = Some(rank);
        }
        if ranks.sequence.is_none()
            && mitm_token_sequence_matches(&candidate.tokens, target_sequence, false)
        {
            ranks.sequence = Some(rank);
        }
        if ranks.il_sequence.is_none()
            && mitm_token_sequence_matches(&candidate.tokens, target_sequence, true)
        {
            ranks.il_sequence = Some(rank);
        }
        if ranks.peptidoform.is_some() && ranks.sequence.is_some() && ranks.il_sequence.is_some() {
            break;
        }
    }
    ranks
}

fn mitm_component_fragment_order(
    left: &MitmJoinedCandidate,
    right: &MitmJoinedCandidate,
) -> Ordering {
    right
        .component_fragment_score
        .total_cmp(&left.component_fragment_score)
        .then_with(|| mitm_evidence_join_order(left, right))
}

fn mitm_component_prefix_ar_mean_order(
    left: &MitmJoinedCandidate,
    right: &MitmJoinedCandidate,
) -> Ordering {
    right
        .component_prefix_ar_mean
        .total_cmp(&left.component_prefix_ar_mean)
        .then_with(|| mitm_evidence_join_order(left, right))
}

fn mitm_component_suffix_ar_mean_order(
    left: &MitmJoinedCandidate,
    right: &MitmJoinedCandidate,
) -> Ordering {
    right
        .component_suffix_ar_mean
        .total_cmp(&left.component_suffix_ar_mean)
        .then_with(|| mitm_evidence_join_order(left, right))
}

fn mitm_component_prefix_ar_total_order(
    left: &MitmJoinedCandidate,
    right: &MitmJoinedCandidate,
) -> Ordering {
    right
        .component_prefix_ar_total
        .total_cmp(&left.component_prefix_ar_total)
        .then_with(|| mitm_evidence_join_order(left, right))
}

fn mitm_component_suffix_ar_total_order(
    left: &MitmJoinedCandidate,
    right: &MitmJoinedCandidate,
) -> Ordering {
    right
        .component_suffix_ar_total
        .total_cmp(&left.component_suffix_ar_total)
        .then_with(|| mitm_evidence_join_order(left, right))
}

fn mitm_component_mass_error_order(
    left: &MitmJoinedCandidate,
    right: &MitmJoinedCandidate,
) -> Ordering {
    left.join_mass_error_da
        .abs()
        .total_cmp(&right.join_mass_error_da.abs())
        .then_with(|| mitm_evidence_join_order(left, right))
}

fn mitm_component_seam_fragment_order(
    left: &MitmJoinedCandidate,
    right: &MitmJoinedCandidate,
) -> Ordering {
    right
        .component_join_seam_fragment_score
        .total_cmp(&left.component_join_seam_fragment_score)
        .then_with(|| mitm_evidence_join_order(left, right))
}

fn mitm_component_dominates(left: &MitmJoinedCandidate, right: &MitmJoinedCandidate) -> bool {
    let left_mass = left.join_mass_error_da.abs();
    let right_mass = right.join_mass_error_da.abs();
    let no_worse = left.component_fragment_score >= right.component_fragment_score
        && left.component_prefix_ar_mean >= right.component_prefix_ar_mean
        && left.component_suffix_ar_mean >= right.component_suffix_ar_mean
        && left_mass <= right_mass;
    let strictly_better = left.component_fragment_score > right.component_fragment_score
        || left.component_prefix_ar_mean > right.component_prefix_ar_mean
        || left.component_suffix_ar_mean > right.component_suffix_ar_mean
        || left_mass < right_mass;
    no_worse && strictly_better
}

fn mitm_component_pareto_frontier(candidates: &[MitmJoinedCandidate]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&left, &right| {
        mitm_component_fragment_order(&candidates[left], &candidates[right])
    });
    let mut frontier = Vec::<usize>::new();
    for index in order {
        if frontier
            .iter()
            .any(|&existing| mitm_component_dominates(&candidates[existing], &candidates[index]))
        {
            continue;
        }
        frontier.retain(|&existing| {
            !mitm_component_dominates(&candidates[index], &candidates[existing])
        });
        frontier.push(index);
    }
    frontier
        .sort_by(|&left, &right| mitm_evidence_join_order(&candidates[left], &candidates[right]));
    frontier
}

fn mitm_component_ranks(
    candidates: &[MitmJoinedCandidate],
    target_tokens: &[u32],
    target_sequence: &str,
) -> MitmComponentRanks {
    MitmComponentRanks {
        fragment: mitm_oracle_ranks_with_order(
            candidates,
            mitm_component_fragment_order,
            target_tokens,
            target_sequence,
        ),
        prefix_ar_mean: mitm_oracle_ranks_with_order(
            candidates,
            mitm_component_prefix_ar_mean_order,
            target_tokens,
            target_sequence,
        ),
        suffix_ar_mean: mitm_oracle_ranks_with_order(
            candidates,
            mitm_component_suffix_ar_mean_order,
            target_tokens,
            target_sequence,
        ),
        prefix_ar_total: mitm_oracle_ranks_with_order(
            candidates,
            mitm_component_prefix_ar_total_order,
            target_tokens,
            target_sequence,
        ),
        suffix_ar_total: mitm_oracle_ranks_with_order(
            candidates,
            mitm_component_suffix_ar_total_order,
            target_tokens,
            target_sequence,
        ),
        mass_error: mitm_oracle_ranks_with_order(
            candidates,
            mitm_component_mass_error_order,
            target_tokens,
            target_sequence,
        ),
        seam_fragment: mitm_oracle_ranks_with_order(
            candidates,
            mitm_component_seam_fragment_order,
            target_tokens,
            target_sequence,
        ),
    }
}

fn mitm_component_rank_any_top256(ranks: &MitmComponentRanks, il: bool) -> bool {
    let select = |ranks: MitmOracleRanks| {
        if il {
            ranks.il_sequence
        } else {
            ranks.peptidoform
        }
    };
    [
        select(ranks.fragment),
        select(ranks.prefix_ar_mean),
        select(ranks.suffix_ar_mean),
        select(ranks.prefix_ar_total),
        select(ranks.suffix_ar_total),
        select(ranks.mass_error),
        select(ranks.seam_fragment),
    ]
    .into_iter()
    .flatten()
    .any(|rank| rank <= 256)
}

fn mitm_target_il_provenance(
    candidates: &[MitmJoinedCandidate],
    target_sequence: &str,
) -> MitmTargetProvenance {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices
        .sort_by(|&left, &right| mitm_evidence_join_order(&candidates[left], &candidates[right]));
    let Some(index) = indices.into_iter().find(|&index| {
        mitm_token_sequence_matches(&candidates[index].tokens, target_sequence, true)
    }) else {
        return MitmTargetProvenance::default();
    };
    let candidate = &candidates[index];
    MitmTargetProvenance {
        found: true,
        prefix_partial_rank: candidate.selector_prefix_partial_rank,
        suffix_partial_rank: candidate.selector_suffix_partial_rank,
        best_prefix_partial_rank: candidate.best_prefix_partial_rank,
        best_suffix_partial_rank: candidate.best_suffix_partial_rank,
        prefix_token_count: candidate.selector_prefix_token_count,
        suffix_token_count: candidate.selector_suffix_token_count,
        prefix_residue_count: candidate.selector_prefix_residue_count,
        suffix_residue_count: candidate.selector_suffix_residue_count,
        join_seam_fragment_score: candidate.selector_join_seam_fragment_score,
        fragment_score: candidate.selector_fragment_score,
        prefix_ar_mean: candidate.selector_prefix_ar_mean,
        suffix_ar_mean: candidate.selector_suffix_ar_mean,
        prefix_ar_total: candidate.selector_prefix_ar_total,
        suffix_ar_total: candidate.selector_suffix_ar_total,
        abs_mass_error_da: candidate.join_mass_error_da.abs(),
    }
}

fn mitm_component_rank_audit(
    candidates: &[MitmJoinedCandidate],
    target: &PeptidoformInput,
    max_tokens: usize,
    vocabulary: FoundationDiffusionVocabulary,
    compute_pareto: bool,
) -> Result<MitmComponentAudit> {
    // Evaluation-only terminal diagnostic. Target identity is never available to join
    // construction, selector scoring, candidate retention, or the accepted final reranker.
    let target_tokens = vocabulary.encode(target, max_tokens).map_err(|error| {
        anyhow::anyhow!("encode v0.13.22 target for component-rank audit: {error}")
    })?;
    let ranks = mitm_component_ranks(candidates, &target_tokens, &target.sequence);
    let frontier = if compute_pareto {
        mitm_component_pareto_frontier(candidates)
    } else {
        Vec::new()
    };
    let mut pareto_peptidoform_present = false;
    let mut pareto_sequence_present = false;
    let mut pareto_il_present = false;
    for index in frontier.iter().copied() {
        let candidate = &candidates[index];
        pareto_peptidoform_present |= candidate.tokens.as_slice() == target_tokens.as_slice();
        pareto_sequence_present |=
            mitm_token_sequence_matches(&candidate.tokens, &target.sequence, false);
        pareto_il_present |= mitm_token_sequence_matches(&candidate.tokens, &target.sequence, true);
    }
    Ok(MitmComponentAudit {
        ranks,
        pareto_frontier_size: frontier.len(),
        pareto_peptidoform_present,
        pareto_sequence_present,
        pareto_il_present,
        il_provenance: mitm_target_il_provenance(candidates, &target.sequence),
    })
}

fn mitm_precap_oracle_audit(
    candidates: &[MitmJoinedCandidate],
    target: &PeptidoformInput,
    max_tokens: usize,
    vocabulary: FoundationDiffusionVocabulary,
) -> Result<MitmPrecapOracleAudit> {
    // IMPORTANT: this is evaluation-only.  The join pool has already been fully generated,
    // deduplicated and scored without target identity before this function is called.
    let target_tokens = vocabulary.encode(target, max_tokens).map_err(|error| {
        anyhow::anyhow!("encode v0.13.21 target for post-generation MITM pre-cap audit: {error}")
    })?;
    let legacy_ranks = mitm_oracle_ranks_with_order(
        candidates,
        mitm_legacy_join_order,
        &target_tokens,
        &target.sequence,
    );
    let evidence_ranks = mitm_oracle_ranks_with_order(
        candidates,
        mitm_evidence_join_order,
        &target_tokens,
        &target.sequence,
    );
    Ok(MitmPrecapOracleAudit {
        candidates: candidates.len(),
        peptidoform_present: legacy_ranks.peptidoform.is_some(),
        sequence_present: legacy_ranks.sequence.is_some(),
        il_sequence_present: legacy_ranks.il_sequence.is_some(),
        legacy_ranks,
        evidence_ranks,
    })
}

fn format_mitm_rank(rank: Option<usize>) -> String {
    rank.map(|value| value.to_string())
        .unwrap_or_else(|| "NA".to_string())
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

fn print_mitm_rank_cutoffs(label: &str, cutoffs: &MitmRankCutoffMetrics) {
    println!(
        "generation_summary\t{label}\ttop1={}\ttop8={}\ttop32={}\ttop64={}\ttop128={}\ttop256={}",
        cutoffs.top1, cutoffs.top8, cutoffs.top32, cutoffs.top64, cutoffs.top128, cutoffs.top256
    );
}

#[derive(Debug, Clone)]
struct MassBeamState {
    prefix: Vec<u32>,
    neutral_mass: f64,
    log_probability: f64,
    fragment_score: f64,
    matched_cleavages: usize,
    residue_count: usize,
    priority: f64,
}

#[allow(clippy::too_many_arguments)]
fn mass_guided_final_beam(
    logits: &[Vec<f32>],
    active_length: usize,
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    beam_width: usize,
    final_candidates_per_chain: usize,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
    spectral_beam_weight: f64,
    temperature: f64,
    max_tokens: usize,
) -> Vec<(Vec<u32>, f64, f64, usize)> {
    let nonterminal_positions = active_length.saturating_sub(1);
    if nonterminal_positions == 0 {
        return Vec::new();
    }
    let target = target_neutral_mass.unwrap_or(f64::NAN);
    let has_mass = target.is_finite() && target > FOUNDATION_PEPTIDE_WATER_MASS_DA;
    let expected_per_position = if has_mass {
        (target - FOUNDATION_PEPTIDE_WATER_MASS_DA) / nonterminal_positions as f64
    } else {
        110.0
    };
    let overshoot_slack = mass_tolerance_da.max(25.0);
    let mass_bin_width = mass_tolerance_da.max(0.05);
    let mut beam = vec![MassBeamState {
        prefix: Vec::with_capacity(nonterminal_positions),
        neutral_mass: FOUNDATION_PEPTIDE_WATER_MASS_DA,
        log_probability: 0.0,
        fragment_score: 0.0,
        matched_cleavages: 0,
        residue_count: 0,
        priority: 0.0,
    }];

    for position in 0..nonterminal_positions {
        let probabilities = clean_x0_probabilities(&logits[position], position, temperature);
        let mut token_order: Vec<usize> = (FOUNDATION_DIFFUSION_EOS as usize
            ..FOUNDATION_DIFFUSION_VOCAB_SIZE)
            .filter(|&token| token != FOUNDATION_DIFFUSION_EOS as usize)
            .collect();
        token_order.sort_by(|&left, &right| probabilities[right].total_cmp(&probabilities[left]));

        let mut binned = HashMap::<(i64, u32), MassBeamState>::new();
        for state in &beam {
            for &token_index in &token_order {
                let probability = probabilities[token_index];
                if probability <= 0.0 || !probability.is_finite() {
                    continue;
                }
                let token = token_index as u32;
                if !mass_beam_token_allowed(&state.prefix, token, position, nonterminal_positions) {
                    continue;
                }
                let Some(token_mass) = foundation_diffusion_token_mass_da(token) else {
                    continue;
                };
                let neutral_mass = state.neutral_mass + token_mass;
                if has_mass && neutral_mass > target + overshoot_slack {
                    continue;
                }
                let log_probability = state.log_probability + probability.max(1e-300).ln();
                let mut fragment_score = state.fragment_score;
                let mut matched_cleavages = state.matched_cleavages;
                let is_residue = foundation_diffusion_token_residue(token).is_some();
                if is_residue && state.residue_count > 0 && has_mass {
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
                let remaining = nonterminal_positions - position - 1;
                let projected_mass = neutral_mass + remaining as f64 * expected_per_position;
                let mass_penalty = if has_mass {
                    0.002 * (projected_mass - target).abs()
                } else {
                    0.0
                };
                let priority =
                    log_probability + spectral_beam_weight * fragment_score - mass_penalty;
                let mut prefix = state.prefix.clone();
                prefix.push(token);
                let candidate = MassBeamState {
                    prefix,
                    neutral_mass,
                    log_probability,
                    fragment_score,
                    matched_cleavages,
                    residue_count,
                    priority,
                };
                let bin = (neutral_mass / mass_bin_width).round() as i64;
                let key = (bin, token);
                match binned.get_mut(&key) {
                    Some(existing) if candidate.priority > existing.priority => {
                        *existing = candidate
                    }
                    None => {
                        binned.insert(key, candidate);
                    }
                    _ => {}
                }
            }
        }
        beam = binned.into_values().collect();
        beam.sort_by(|left, right| right.priority.total_cmp(&left.priority));
        beam.truncate(beam_width);
        if beam.is_empty() {
            return Vec::new();
        }
    }

    beam.sort_by(|left, right| {
        if has_mass {
            let left_error = (left.neutral_mass - target).abs();
            let right_error = (right.neutral_mass - target).abs();
            let left_valid = left_error <= mass_tolerance_da;
            let right_valid = right_error <= mass_tolerance_da;
            right_valid
                .cmp(&left_valid)
                .then_with(|| right.fragment_score.total_cmp(&left.fragment_score))
                .then_with(|| left_error.total_cmp(&right_error))
                .then_with(|| right.log_probability.total_cmp(&left.log_probability))
        } else {
            right.log_probability.total_cmp(&left.log_probability)
        }
    });
    beam.truncate(final_candidates_per_chain);
    beam.into_iter()
        .map(|state| {
            let mut row = vec![FOUNDATION_DIFFUSION_PAD; max_tokens];
            for (position, token) in state.prefix.into_iter().enumerate() {
                row[position] = token;
            }
            row[active_length - 1] = FOUNDATION_DIFFUSION_EOS;
            (
                row,
                state.log_probability,
                state.fragment_score,
                state.matched_cleavages,
            )
        })
        .collect()
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

fn predict_length_distribution(
    model: &PeptideSpectrumDiffusionModel,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    device: &Device,
) -> Result<Vec<f64>> {
    let row = vec![FOUNDATION_DIFFUSION_MASK; config.max_tokens];
    let diffusion = diffusion_collator.collate_inference_tokens(
        &[row],
        &[config.max_tokens],
        config.diffusion_steps,
        device,
    )?;
    let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
    let precursor = precursor_context(&[record], device)?;
    let output = model.forward_t(&diffusion, &spectrum_batch, &precursor, false)?;
    let logits = output.length_logits.to_vec2::<f32>()?;
    Ok(softmax(&logits[0], 1.0))
}

#[allow(clippy::too_many_arguments)]
fn score_all_mask_token_row(
    model: &PeptideSpectrumDiffusionModel,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    candidate_tokens: &[u32],
    active_length: usize,
    device: &Device,
) -> Result<AllMaskCandidateScore> {
    let rows = all_mask_inference_rows(&[active_length], config.max_tokens)?;
    let diffusion = diffusion_collator.collate_inference_tokens(
        &rows,
        &[active_length],
        config.diffusion_steps,
        device,
    )?;
    let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
    let precursor = precursor_context(&[record], device)?;
    let output = model.forward_t(&diffusion, &spectrum_batch, &precursor, false)?;
    let token_logits = output.token_logits.to_vec3::<f32>()?;
    let length_logits = output.length_logits.to_vec2::<f32>()?;
    score_candidate_from_logits(
        &token_logits[0],
        &length_logits[0],
        candidate_tokens,
        active_length,
    )
}

#[allow(clippy::too_many_arguments)]
fn score_all_mask_candidates(
    model: &PeptideSpectrumDiffusionModel,
    diffusion_collator: &FoundationDiffusionCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    candidates: &[GeneratedCandidate],
    device: &Device,
) -> Result<Vec<AllMaskCandidateScore>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let active_lengths: Vec<usize> = candidates
        .iter()
        .map(|candidate| active_token_length(&candidate.tokens, config.max_tokens))
        .collect::<Result<_>>()?;
    let rows = all_mask_inference_rows(&active_lengths, config.max_tokens)?;
    let diffusion = diffusion_collator.collate_inference_tokens(
        &rows,
        &active_lengths,
        config.diffusion_steps,
        device,
    )?;
    let spectra = vec![spectrum.clone(); candidates.len()];
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let record_refs = vec![record; candidates.len()];
    let precursor = precursor_context(&record_refs, device)?;
    let output = model.forward_t(&diffusion, &spectrum_batch, &precursor, false)?;
    let token_logits = output.token_logits.to_vec3::<f32>()?;
    let length_logits = output.length_logits.to_vec2::<f32>()?;

    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            score_candidate_from_logits(
                &token_logits[index],
                &length_logits[index],
                &candidate.tokens,
                active_lengths[index],
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn score_causal_token_row(
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    candidate_tokens: &[u32],
    device: &Device,
) -> Result<CausalCandidateScore> {
    let causal = causal_collator.collate_token_rows(&[candidate_tokens.to_vec()], device)?;
    let spectrum_batch = spectrum_collator.collate(&[spectrum.clone()], device)?;
    let precursor = precursor_context(&[record], device)?;
    let output = model.forward_t(&causal.input, &spectrum_batch, &precursor, false)?;
    let logits = output.token_logits.to_vec3::<f32>()?;
    score_causal_candidate_from_logits(&logits[0], candidate_tokens)
}

#[allow(clippy::too_many_arguments)]
fn score_causal_candidates(
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    candidates: &[GeneratedCandidate],
    device: &Device,
) -> Result<Vec<CausalCandidateScore>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let target_rows: Vec<Vec<u32>> = candidates
        .iter()
        .map(|candidate| candidate.tokens.clone())
        .collect();
    let causal = causal_collator.collate_token_rows(&target_rows, device)?;
    let spectra = vec![spectrum.clone(); candidates.len()];
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let record_refs = vec![record; candidates.len()];
    let precursor = precursor_context(&record_refs, device)?;
    let output = model.forward_t(&causal.input, &spectrum_batch, &precursor, false)?;
    let logits = output.token_logits.to_vec3::<f32>()?;
    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            score_causal_candidate_from_logits(&logits[index], &candidate.tokens)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn score_reverse_causal_candidates(
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    candidates: &[GeneratedCandidate],
    device: &Device,
) -> Result<Vec<CausalCandidateScore>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let reverse_rows: Vec<Vec<u32>> = candidates
        .iter()
        .map(|candidate| {
            foundation_reverse_causal_token_row(&candidate.tokens).map_err(anyhow::Error::msg)
        })
        .collect::<Result<_>>()?;
    let causal = causal_collator.collate_token_rows(&reverse_rows, device)?;
    let spectra = vec![spectrum.clone(); candidates.len()];
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let record_refs = vec![record; candidates.len()];
    let precursor = precursor_context(&record_refs, device)?;
    let output = model.forward_t(&causal.input, &spectrum_batch, &precursor, false)?;
    let logits = output.token_logits.to_vec3::<f32>()?;
    reverse_rows
        .iter()
        .enumerate()
        .map(|(index, row)| score_causal_candidate_from_logits(&logits[index], row))
        .collect()
}

fn score_causal_candidate_from_logits(
    token_logits: &[Vec<f32>],
    candidate_tokens: &[u32],
) -> Result<CausalCandidateScore> {
    let active_length = candidate_tokens
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap_or(candidate_tokens.len());
    if active_length == 0
        || active_length > token_logits.len()
        || candidate_tokens[active_length - 1] != FOUNDATION_DIFFUSION_EOS
    {
        anyhow::bail!("causal candidate must have a non-empty EOS-terminated active prefix");
    }
    let mut total = 0.0f64;
    for position in 0..active_length {
        let token = candidate_tokens[position] as usize;
        if token == FOUNDATION_DIFFUSION_PAD as usize || token == FOUNDATION_DIFFUSION_MASK as usize
        {
            anyhow::bail!("causal clean candidate contains PAD/MASK in its active prefix");
        }
        total += selected_log_softmax(&token_logits[position], token)?;
    }
    let mean = total / active_length as f64;
    Ok(CausalCandidateScore {
        total_log_probability: total,
        mean_log_probability: mean,
        perplexity: (-mean).exp(),
    })
}

fn all_mask_inference_rows(active_lengths: &[usize], max_tokens: usize) -> Result<Vec<Vec<u32>>> {
    active_lengths
        .iter()
        .map(|&active_length| {
            if active_length == 0 || active_length > max_tokens {
                anyhow::bail!(
                    "all-MASK candidate active length {active_length} is outside 1..={max_tokens}"
                );
            }
            let mut row = vec![FOUNDATION_DIFFUSION_PAD; max_tokens];
            // Match the spectrum-only training objective exactly: every active clean
            // token, including EOS, is hidden behind MASK. Candidate identity is not
            // present in the model input; it is used only after inference to index the
            // returned x0 probability distribution.
            for token in row.iter_mut().take(active_length) {
                *token = FOUNDATION_DIFFUSION_MASK;
            }
            Ok(row)
        })
        .collect()
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

fn score_candidate_from_logits(
    token_logits: &[Vec<f32>],
    length_logits: &[f32],
    candidate_tokens: &[u32],
    active_length: usize,
) -> Result<AllMaskCandidateScore> {
    if active_length == 0
        || active_length > token_logits.len()
        || active_length > candidate_tokens.len()
    {
        anyhow::bail!("candidate active length is incompatible with neural reranking logits");
    }
    let mut token_log_probability_sum = 0.0f64;
    for position in 0..active_length {
        let token = candidate_tokens[position] as usize;
        if token == FOUNDATION_DIFFUSION_PAD as usize || token == FOUNDATION_DIFFUSION_MASK as usize
        {
            anyhow::bail!("clean reranking candidate contains PAD/MASK in its active prefix");
        }
        token_log_probability_sum += selected_log_softmax(&token_logits[position], token)?;
    }
    let length_class = active_length - 1;
    let length_log_probability = selected_log_softmax(length_logits, length_class)?;
    Ok(AllMaskCandidateScore {
        mean_token_log_probability: token_log_probability_sum / active_length as f64,
        length_log_probability,
    })
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

fn sample_generation_lengths(
    probabilities: &[f64],
    argmax_length: usize,
    samples: usize,
    max_tokens: usize,
    target_neutral_mass: Option<f64>,
    rng: &mut GenerationRng,
) -> Vec<usize> {
    let mut lengths = Vec::with_capacity(samples);
    push_unique_length(&mut lengths, argmax_length, max_tokens);

    // Precursor mass supplies a target-independent estimate of residue count.
    // Seed nearby lengths before stochastic draws so badly calibrated length
    // logits cannot exclude the physically plausible region entirely.
    if let Some(target) = target_neutral_mass.filter(|value| value.is_finite()) {
        let residue_estimate = ((target - FOUNDATION_PEPTIDE_WATER_MASS_DA) / 111.0)
            .round()
            .max(1.0) as isize;
        for offset in [0isize, 1, -1, 2, -2] {
            let active = residue_estimate + 1 + offset; // + EOS; PTMs may consume extra slots.
            if active >= 2 {
                push_unique_length(&mut lengths, active as usize, max_tokens);
                if lengths.len() >= samples {
                    return lengths;
                }
            }
        }
    }

    let mut ranked: Vec<usize> = (0..probabilities.len()).collect();
    ranked.sort_by(|&left, &right| probabilities[right].total_cmp(&probabilities[left]));
    for index in ranked.into_iter().take(samples) {
        push_unique_length(&mut lengths, index + 1, max_tokens);
        if lengths.len() >= samples {
            return lengths;
        }
    }
    let mut attempts = 0usize;
    while lengths.len() < samples && attempts < samples.saturating_mul(16).max(32) {
        push_unique_length(
            &mut lengths,
            sample_probability(probabilities, rng) + 1,
            max_tokens,
        );
        attempts += 1;
        if lengths.len() >= max_tokens.saturating_sub(1) {
            break;
        }
    }
    lengths.truncate(samples);
    lengths
}

fn push_unique_length(lengths: &mut Vec<usize>, length: usize, max_tokens: usize) {
    let length = length.clamp(2, max_tokens);
    if !lengths.contains(&length) {
        lengths.push(length);
    }
}

fn clean_x0_probabilities(logits: &[f32], position: usize, temperature: f64) -> Vec<f64> {
    let mut adjusted = vec![f64::NEG_INFINITY; FOUNDATION_DIFFUSION_VOCAB_SIZE];
    for token in FOUNDATION_DIFFUSION_EOS as usize..FOUNDATION_DIFFUSION_VOCAB_SIZE {
        if token == FOUNDATION_DIFFUSION_EOS as usize {
            continue;
        }
        if token == FOUNDATION_DIFFUSION_NTERM_ACETYL as usize && position != 0 {
            continue;
        }
        if position == 0 && token >= FOUNDATION_DIFFUSION_RESIDUE_ACETYL as usize {
            continue;
        }
        adjusted[token] = logits[token] as f64 / temperature;
    }
    softmax_log_values(&adjusted)
}

fn apply_nonterminal_constraints(probabilities: &mut [f64], position: usize, timestep: usize) {
    probabilities[FOUNDATION_DIFFUSION_PAD as usize] = 0.0;
    probabilities[FOUNDATION_DIFFUSION_EOS as usize] = 0.0;
    if position != 0 {
        probabilities[FOUNDATION_DIFFUSION_NTERM_ACETYL as usize] = 0.0;
    } else {
        for probability in probabilities
            .iter_mut()
            .skip(FOUNDATION_DIFFUSION_RESIDUE_ACETYL as usize)
        {
            *probability = 0.0;
        }
    }
    if timestep == 1 {
        probabilities[FOUNDATION_DIFFUSION_MASK as usize] = 0.0;
    }
    renormalize(probabilities);
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

fn precursor_mass_error(
    record: &FoundationTrainingRecord,
    peptide: &PeptidoformInput,
) -> Result<Option<f64>> {
    match (record.context.precursor_mz, record.context.charge) {
        (Some(mz), Some(charge)) if charge > 0 => Ok(Some(
            foundation_precursor_mass_error_da(peptide, mz as f64, charge as i32)
                .map_err(anyhow::Error::msg)?,
        )),
        _ => Ok(None),
    }
}

fn mass_candidate_order(left: &GeneratedCandidate, right: &GeneratedCandidate) -> Ordering {
    right
        .mass_valid
        .cmp(&left.mass_valid)
        .then_with(|| {
            let left_error = left.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            let right_error = right.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            left_error.total_cmp(&right_error)
        })
        .then_with(|| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        })
}

#[derive(Debug, Clone, Copy, Default)]
struct FragmentEvidence {
    score: f64,
    matched_cleavages: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct CleavageEvidence {
    score: f64,
    matched: bool,
}

fn fragment_mass_candidate_order(
    left: &GeneratedCandidate,
    right: &GeneratedCandidate,
) -> Ordering {
    right
        .mass_valid
        .cmp(&left.mass_valid)
        .then_with(|| right.fragment_score.total_cmp(&left.fragment_score))
        .then_with(|| {
            let left_error = left.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            let right_error = right.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            left_error.total_cmp(&right_error)
        })
        .then_with(|| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        })
}

fn neural_mass_candidate_order(left: &GeneratedCandidate, right: &GeneratedCandidate) -> Ordering {
    right
        .mass_valid
        .cmp(&left.mass_valid)
        .then_with(|| {
            right
                .neural_all_mask_log_probability
                .total_cmp(&left.neural_all_mask_log_probability)
        })
        .then_with(|| {
            let left_error = left.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            let right_error = right.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            left_error.total_cmp(&right_error)
        })
        .then_with(|| right.fragment_score.total_cmp(&left.fragment_score))
        .then_with(|| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        })
}

fn hybrid_mass_candidate_order(left: &GeneratedCandidate, right: &GeneratedCandidate) -> Ordering {
    right
        .mass_valid
        .cmp(&left.mass_valid)
        .then_with(|| right.hybrid_score.total_cmp(&left.hybrid_score))
        .then_with(|| {
            let left_error = left.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            let right_error = right.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            left_error.total_cmp(&right_error)
        })
        .then_with(|| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        })
}

fn causal_mass_candidate_order(left: &GeneratedCandidate, right: &GeneratedCandidate) -> Ordering {
    right
        .mass_valid
        .cmp(&left.mass_valid)
        .then_with(|| {
            right
                .ar_total_log_probability
                .total_cmp(&left.ar_total_log_probability)
        })
        .then_with(|| {
            let left_error = left.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            let right_error = right.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            left_error.total_cmp(&right_error)
        })
        .then_with(|| right.fragment_score.total_cmp(&left.fragment_score))
        .then_with(|| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        })
}

fn fragment_causal_mass_candidate_order(
    left: &GeneratedCandidate,
    right: &GeneratedCandidate,
) -> Ordering {
    right
        .mass_valid
        .cmp(&left.mass_valid)
        .then_with(|| {
            right
                .fragment_causal_score
                .total_cmp(&left.fragment_causal_score)
        })
        .then_with(|| {
            let left_error = left.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            let right_error = right.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            left_error.total_cmp(&right_error)
        })
        .then_with(|| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        })
}

fn fragment_bidirectional_causal_mass_candidate_order(
    left: &GeneratedCandidate,
    right: &GeneratedCandidate,
) -> Ordering {
    right
        .mass_valid
        .cmp(&left.mass_valid)
        .then_with(|| {
            right
                .fragment_bidirectional_causal_score
                .total_cmp(&left.fragment_bidirectional_causal_score)
        })
        .then_with(|| {
            let left_error = left.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            let right_error = right.mass_error_da.map(f64::abs).unwrap_or(f64::INFINITY);
            left_error.total_cmp(&right_error)
        })
        .then_with(|| right.fragment_score.total_cmp(&left.fragment_score))
        .then_with(|| {
            right
                .reverse_log_probability
                .total_cmp(&left.reverse_log_probability)
        })
}

fn ranking_exact_flags(
    candidate: &GeneratedCandidate,
    record: &FoundationTrainingRecord,
    target_il: &str,
) -> (usize, usize, usize) {
    (
        (candidate.peptide == record.peptidoform) as usize,
        (candidate.peptide.sequence == record.peptidoform.sequence) as usize,
        (normalize_il(&candidate.peptide.sequence) == target_il) as usize,
    )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
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

fn peptidoform_fragment_evidence(
    peptide: &PeptidoformInput,
    precursor_neutral_mass: Option<f64>,
    observed_peaks: &[(f64, f64)],
    max_fragment_charge: usize,
    fragment_tolerance_ppm: f64,
) -> FragmentEvidence {
    let Some(target_mass) = precursor_neutral_mass.filter(|value| value.is_finite()) else {
        return FragmentEvidence::default();
    };
    let vocabulary = FoundationDiffusionVocabulary;
    let Ok(tokens) = vocabulary.encode(peptide, 256) else {
        return FragmentEvidence::default();
    };
    let mut prefix_mass = 0.0f64;
    let mut residue_count = 0usize;
    let mut total = FragmentEvidence::default();
    for token in tokens {
        if token == FOUNDATION_DIFFUSION_PAD || token == FOUNDATION_DIFFUSION_EOS {
            break;
        }
        if foundation_diffusion_token_residue(token).is_some() {
            if residue_count > 0 {
                let evidence = cleavage_fragment_evidence(
                    prefix_mass,
                    target_mass,
                    observed_peaks,
                    max_fragment_charge,
                    fragment_tolerance_ppm,
                );
                total.score += evidence.score;
                total.matched_cleavages += usize::from(evidence.matched);
            }
            residue_count += 1;
        }
        if let Some(mass) = foundation_diffusion_token_mass_da(token) {
            prefix_mass += mass;
        }
    }
    total
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

fn companion_spectra_path(output: &std::path::Path) -> PathBuf {
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("generation");
    output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(format!("{stem}.spectra.tsv"))
}

fn format_modifications(peptide: &PeptidoformInput) -> String {
    peptide
        .modifications
        .iter()
        .map(|modification| format!("{}@{:?}", modification.identity_label(), modification.site))
        .collect::<Vec<_>>()
        .join(";")
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

fn softmax(values: &[f32], temperature: f64) -> Vec<f64> {
    let adjusted: Vec<f64> = values
        .iter()
        .map(|&value| value as f64 / temperature)
        .collect();
    softmax_log_values(&adjusted)
}

fn softmax_log_values(values: &[f64]) -> Vec<f64> {
    let finite_max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = vec![0.0f64; values.len()];
    if !finite_max.is_finite() {
        return probabilities;
    }
    for (index, &value) in values.iter().enumerate() {
        if value.is_finite() {
            probabilities[index] = (value - finite_max).exp();
        }
    }
    renormalize(&mut probabilities);
    probabilities
}

fn renormalize(probabilities: &mut [f64]) {
    let total: f64 = probabilities.iter().sum();
    if total > 0.0 && total.is_finite() {
        for probability in probabilities {
            *probability /= total;
        }
    }
}

fn sample_probability(probabilities: &[f64], rng: &mut GenerationRng) -> usize {
    let mut threshold = rng.next_f64();
    let mut last_nonzero = 0usize;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability <= 0.0 {
            continue;
        }
        last_nonzero = index;
        if threshold <= probability {
            return index;
        }
        threshold -= probability;
    }
    last_nonzero
}

fn argmax_f64(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
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

fn sanitize_diagnostic_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch == '\t' || ch == '\n' || ch == '\r' {
                ' '
            } else {
                ch
            }
        })
        .collect()
}

fn format_finite(value: f64) -> String {
    value
        .is_finite()
        .then(|| format!("{value:.8}"))
        .unwrap_or_default()
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

#[derive(Debug, Clone, Copy)]
struct GenerationRng {
    state: u64,
}

impl GenerationRng {
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

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 / ((1u64 << 53) - 1) as f64
    }
}

fn select_iterative_refinement_seeds(candidates: &[GeneratedCandidate]) -> Vec<GeneratedCandidate> {
    let mut selected = Vec::<GeneratedCandidate>::new();
    let mut seen = HashSet::<Vec<u32>>::new();
    let per_branch = FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315 / 3;

    for source in 0..3 {
        let mut branch = candidates
            .iter()
            .filter(|candidate| candidate.mass_valid)
            .filter(|candidate| match source {
                0 => candidate.from_diffusion,
                1 => candidate.from_causal_beam,
                _ => candidate.from_reverse_causal_beam,
            })
            .cloned()
            .collect::<Vec<_>>();
        branch.sort_by(fragment_causal_mass_candidate_order);
        let mut branch_added = 0usize;
        for candidate in branch {
            if seen.insert(candidate.tokens.clone()) {
                selected.push(candidate);
                branch_added += 1;
                if selected.len() >= FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315
                    || branch_added >= per_branch
                {
                    break;
                }
            }
        }
    }

    if selected.len() < FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315 {
        let mut global = candidates
            .iter()
            .filter(|candidate| candidate.mass_valid)
            .cloned()
            .collect::<Vec<_>>();
        global.sort_by(fragment_causal_mass_candidate_order);
        for candidate in global {
            if seen.insert(candidate.tokens.clone()) {
                selected.push(candidate);
                if selected.len() >= FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315 {
                    break;
                }
            }
        }
    }
    selected.truncate(FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315);
    selected
}

#[allow(clippy::too_many_arguments)]
fn iterative_refine_candidates(
    refiner: &IterativeRefiner,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    seeds: &[GeneratedCandidate],
    target_neutral_mass: Option<f64>,
    mass_tolerance_da: f64,
    device: &Device,
) -> Result<Vec<Vec<u32>>> {
    let Some(target_neutral_mass) = target_neutral_mass.filter(|value| value.is_finite()) else {
        return Ok(Vec::new());
    };
    let mut emitted = Vec::<Vec<u32>>::new();
    let mut emitted_seen = HashSet::<Vec<u32>>::new();

    for (seed_index, seed) in seeds.iter().enumerate() {
        let mut current = seed.tokens.clone();
        let mut trajectory_seen = HashSet::<Vec<u32>>::new();
        trajectory_seen.insert(current.clone());
        for round in 0..FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315 {
            let Some(next) = iterative_refinement_round(
                refiner,
                spectrum_collator,
                config,
                record,
                spectrum,
                &current,
                target_neutral_mass,
                mass_tolerance_da,
                mix64((seed_index as u64) ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                &trajectory_seen,
                device,
            )?
            else {
                break;
            };
            trajectory_seen.insert(next.clone());
            if emitted_seen.insert(next.clone()) {
                emitted.push(next.clone());
            }
            current = next;
        }
    }
    Ok(emitted)
}

#[allow(clippy::too_many_arguments)]
fn iterative_refinement_round(
    refiner: &IterativeRefiner,
    spectrum_collator: &FoundationSpectrumCollator,
    config: &FoundationDiffusionConfig,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    current: &[u32],
    target_neutral_mass: f64,
    mass_tolerance_da: f64,
    seed: u64,
    forbidden: &HashSet<Vec<u32>>,
    device: &Device,
) -> Result<Option<Vec<u32>>> {
    let active_length = active_token_length(current, config.max_tokens)?;
    let residue_positions = current[..active_length]
        .iter()
        .enumerate()
        .filter_map(|(position, &token)| {
            foundation_diffusion_token_residue(token).map(|_| position)
        })
        .collect::<Vec<_>>();
    if residue_positions.is_empty() {
        return Ok(None);
    }

    let confidence_groups = 4usize;
    let mut confidence_rows = Vec::<Vec<u32>>::with_capacity(confidence_groups);
    for group in 0..confidence_groups {
        let mut row = current.to_vec();
        for (residue_index, &position) in residue_positions.iter().enumerate() {
            if residue_index % confidence_groups == group {
                row[position] = FOUNDATION_DIFFUSION_MASK;
            }
        }
        confidence_rows.push(row);
    }
    let confidence_lengths = vec![active_length; confidence_groups];
    let confidence_logits = iterative_refinement_logits(
        refiner,
        spectrum_collator,
        record,
        spectrum,
        &confidence_rows,
        &confidence_lengths,
        device,
    )?;
    let mut confidence = Vec::<(f64, u64, usize)>::with_capacity(residue_positions.len());
    for (residue_index, &position) in residue_positions.iter().enumerate() {
        let group = residue_index % confidence_groups;
        let token = current[position] as usize;
        confidence.push((
            selected_log_softmax(&confidence_logits[group][position], token)?,
            mix64(seed ^ position as u64),
            position,
        ));
    }
    confidence.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let refine_count = ((residue_positions.len() as f64
        * FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315)
        .ceil() as usize)
        .clamp(1, residue_positions.len());
    let mut selected_positions = confidence
        .into_iter()
        .take(refine_count)
        .map(|(_, _, position)| position)
        .collect::<Vec<_>>();
    selected_positions.sort_unstable();

    let mut masked = current.to_vec();
    for &position in &selected_positions {
        masked[position] = FOUNDATION_DIFFUSION_MASK;
    }
    let refill_logits = iterative_refinement_logits(
        refiner,
        spectrum_collator,
        record,
        spectrum,
        &[masked],
        &[active_length],
        device,
    )?;

    let mut options = Vec::<Vec<(u32, f64, f64)>>::with_capacity(selected_positions.len());
    for &position in &selected_positions {
        let mut position_options = Vec::<(u32, f64, f64)>::new();
        for token in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
            if foundation_diffusion_token_residue(token).is_none() {
                continue;
            }
            if !replacement_residue_ptm_compatible(current, position, active_length, token) {
                continue;
            }
            let Some(mass) = foundation_diffusion_token_mass_da(token) else {
                continue;
            };
            let log_probability =
                selected_log_softmax(&refill_logits[0][position], token as usize)?;
            position_options.push((token, log_probability, mass));
        }
        position_options.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let original = current[position];
        let original_option = position_options
            .iter()
            .find(|(token, _, _)| *token == original)
            .copied();
        position_options.truncate(FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315);
        if let Some(original_option) = original_option {
            if !position_options
                .iter()
                .any(|(token, _, _)| *token == original)
            {
                position_options.push(original_option);
            }
        }
        if position_options.is_empty() {
            return Ok(None);
        }
        options.push(position_options);
    }

    let selected_set = selected_positions.iter().copied().collect::<HashSet<_>>();
    let fixed_mass = FOUNDATION_PEPTIDE_WATER_MASS_DA
        + current[..active_length]
            .iter()
            .enumerate()
            .filter(|(position, _)| !selected_set.contains(position))
            .filter_map(|(_, &token)| foundation_diffusion_token_mass_da(token))
            .sum::<f64>();
    let mut future_min = vec![0.0f64; options.len() + 1];
    let mut future_max = vec![0.0f64; options.len() + 1];
    for index in (0..options.len()).rev() {
        let min_mass = options[index]
            .iter()
            .map(|(_, _, mass)| *mass)
            .fold(f64::INFINITY, f64::min);
        let max_mass = options[index]
            .iter()
            .map(|(_, _, mass)| *mass)
            .fold(f64::NEG_INFINITY, f64::max);
        future_min[index] = future_min[index + 1] + min_mass;
        future_max[index] = future_max[index + 1] + max_mass;
    }

    let mut beam = vec![RefinementFillState {
        tokens: current.to_vec(),
        assigned_mass_da: fixed_mass,
        log_probability: 0.0,
    }];
    for (option_index, &position) in selected_positions.iter().enumerate() {
        let mut expanded = Vec::<RefinementFillState>::new();
        for state in &beam {
            for &(token, log_probability, mass) in &options[option_index] {
                let assigned_mass_da = state.assigned_mass_da + mass;
                let min_final = assigned_mass_da + future_min[option_index + 1];
                let max_final = assigned_mass_da + future_max[option_index + 1];
                if target_neutral_mass < min_final - mass_tolerance_da
                    || target_neutral_mass > max_final + mass_tolerance_da
                {
                    continue;
                }
                let mut tokens = state.tokens.clone();
                tokens[position] = token;
                expanded.push(RefinementFillState {
                    tokens,
                    assigned_mass_da,
                    log_probability: state.log_probability + log_probability,
                });
            }
        }
        expanded.sort_by(|left, right| {
            right
                .log_probability
                .total_cmp(&left.log_probability)
                .then_with(|| {
                    (left.assigned_mass_da - target_neutral_mass)
                        .abs()
                        .total_cmp(&(right.assigned_mass_da - target_neutral_mass).abs())
                })
                .then_with(|| left.tokens.cmp(&right.tokens))
        });
        expanded.truncate(FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315);
        if expanded.is_empty() {
            return Ok(None);
        }
        beam = expanded;
    }

    beam.sort_by(|left, right| {
        right
            .log_probability
            .total_cmp(&left.log_probability)
            .then_with(|| {
                (left.assigned_mass_da - target_neutral_mass)
                    .abs()
                    .total_cmp(&(right.assigned_mass_da - target_neutral_mass).abs())
            })
            .then_with(|| left.tokens.cmp(&right.tokens))
    });
    Ok(beam
        .into_iter()
        .find(|state| {
            (state.assigned_mass_da - target_neutral_mass).abs() <= mass_tolerance_da
                && state.tokens != current
                && !forbidden.contains(&state.tokens)
        })
        .map(|state| state.tokens))
}

fn replacement_residue_ptm_compatible(
    tokens: &[u32],
    position: usize,
    active_length: usize,
    residue_token: u32,
) -> bool {
    let Some(residue) = foundation_diffusion_token_residue(residue_token) else {
        return false;
    };
    let mut cursor = position + 1;
    while cursor < active_length {
        let token = tokens[cursor];
        if token == FOUNDATION_DIFFUSION_EOS || foundation_diffusion_token_residue(token).is_some()
        {
            break;
        }
        if token == FOUNDATION_DIFFUSION_NTERM_ACETYL
            || token == FOUNDATION_DIFFUSION_MASK
            || token == FOUNDATION_DIFFUSION_PAD
            || !foundation_diffusion_residue_ptm_valid(token, residue)
        {
            return false;
        }
        cursor += 1;
    }
    true
}

fn iterative_refinement_logits(
    refiner: &IterativeRefiner,
    spectrum_collator: &FoundationSpectrumCollator,
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
    rows: &[Vec<u32>],
    active_lengths: &[usize],
    device: &Device,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let diffusion = refiner.collator.collate_inference_tokens(
        rows,
        active_lengths,
        refiner.model.config().diffusion_steps,
        device,
    )?;
    let spectra = vec![spectrum.clone(); rows.len()];
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let records = vec![record; rows.len()];
    let precursor = precursor_context(&records, device)?;
    Ok(refiner
        .model
        .forward_t(&diffusion, &spectrum_batch, &precursor, false)?
        .token_logits
        .to_vec3::<f32>()?)
}

#[cfg(test)]
mod fragment_evidence_tests {
    use super::*;
    use redeem_properties::foundation::FoundationSpectrumPeak;

    #[test]
    fn all_mask_reranker_input_depends_on_length_not_candidate_identity() {
        let vocabulary = FoundationDiffusionVocabulary;
        let first = vocabulary
            .encode(
                &PeptidoformInput {
                    sequence: "PEPTIDEK".into(),
                    modifications: Vec::new(),
                },
                16,
            )
            .unwrap();
        let second = vocabulary
            .encode(
                &PeptidoformInput {
                    sequence: "KEDITPEP".into(),
                    modifications: Vec::new(),
                },
                16,
            )
            .unwrap();
        assert_ne!(first, second);
        let first_length = active_token_length(&first, 16).unwrap();
        let second_length = active_token_length(&second, 16).unwrap();
        assert_eq!(first_length, second_length);

        let rows = all_mask_inference_rows(&[first_length, second_length], 16).unwrap();
        assert_eq!(rows[0], rows[1]);
        assert!(rows[0][..first_length]
            .iter()
            .all(|&token| token == FOUNDATION_DIFFUSION_MASK));
        assert!(rows[0][first_length..]
            .iter()
            .all(|&token| token == FOUNDATION_DIFFUSION_PAD));
    }

    #[test]
    fn neural_reranker_uses_candidate_only_to_read_post_inference_logits() {
        let vocabulary = FoundationDiffusionVocabulary;
        let supported = vocabulary
            .encode(
                &PeptidoformInput {
                    sequence: "AC".into(),
                    modifications: Vec::new(),
                },
                8,
            )
            .unwrap();
        let alternative = vocabulary
            .encode(
                &PeptidoformInput {
                    sequence: "CA".into(),
                    modifications: Vec::new(),
                },
                8,
            )
            .unwrap();
        let active_length = active_token_length(&supported, 8).unwrap();
        assert_eq!(active_length, active_token_length(&alternative, 8).unwrap());

        let mut token_logits = vec![vec![0.0f32; FOUNDATION_DIFFUSION_VOCAB_SIZE]; 8];
        for position in 0..active_length {
            token_logits[position][supported[position] as usize] = 4.0;
        }
        let mut length_logits = vec![0.0f32; 8];
        length_logits[active_length - 1] = 2.0;
        let supported_score =
            score_candidate_from_logits(&token_logits, &length_logits, &supported, active_length)
                .unwrap();
        let alternative_score =
            score_candidate_from_logits(&token_logits, &length_logits, &alternative, active_length)
                .unwrap();
        assert!(
            supported_score.mean_token_log_probability
                > alternative_score.mean_token_log_probability
        );

        // Both candidates would have produced the exact same all-MASK model input.
        let rows = all_mask_inference_rows(&[active_length, active_length], 8).unwrap();
        assert_eq!(rows[0], rows[1]);
    }

    #[test]
    fn causal_sequence_likelihood_includes_eos_and_reports_perplexity() {
        let vocabulary = FoundationDiffusionVocabulary;
        let candidate = vocabulary
            .encode(
                &PeptidoformInput {
                    sequence: "AC".into(),
                    modifications: Vec::new(),
                },
                8,
            )
            .unwrap();
        let active_length = active_token_length(&candidate, 8).unwrap();
        assert_eq!(candidate[active_length - 1], FOUNDATION_DIFFUSION_EOS);

        let mut logits = vec![vec![0.0f32; FOUNDATION_DIFFUSION_VOCAB_SIZE]; 8];
        for position in 0..active_length {
            logits[position][candidate[position] as usize] = 4.0;
        }
        let supported = score_causal_candidate_from_logits(&logits, &candidate).unwrap();

        let mut bad_eos = logits.clone();
        bad_eos[active_length - 1][FOUNDATION_DIFFUSION_EOS as usize] = -4.0;
        let unsupported_eos = score_causal_candidate_from_logits(&bad_eos, &candidate).unwrap();
        assert!(supported.total_log_probability > unsupported_eos.total_log_probability);
        assert!((supported.perplexity - (-supported.mean_log_probability).exp()).abs() < 1e-12);
    }

    #[test]
    fn target_fragment_ladder_scores_above_mass_scrambled_sequence() {
        let target = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let scrambled = PeptidoformInput {
            sequence: "KEDITPEP".into(),
            modifications: Vec::new(),
        };
        let target_mass =
            redeem_properties::foundation::foundation_peptidoform_neutral_mass(&target).unwrap();
        let vocabulary = FoundationDiffusionVocabulary;
        let tokens = vocabulary.encode(&target, 32).unwrap();
        let mut prefix_mass = 0.0;
        let mut residue_count = 0usize;
        let mut peaks = Vec::new();
        for token in tokens {
            if token == FOUNDATION_DIFFUSION_EOS || token == FOUNDATION_DIFFUSION_PAD {
                break;
            }
            if foundation_diffusion_token_residue(token).is_some() {
                if residue_count > 0 {
                    peaks.push(FoundationSpectrumPeak {
                        mz: (prefix_mass + 1.007_276_466_77) as f32,
                        intensity: 1.0,
                    });
                }
                residue_count += 1;
            }
            prefix_mass += foundation_diffusion_token_mass_da(token).unwrap();
        }
        let spectrum = FoundationSpectrum { peaks };
        let observed = normalized_observed_peaks(&spectrum);
        let target_score =
            peptidoform_fragment_evidence(&target, Some(target_mass), &observed, 1, 20.0);
        let scrambled_score =
            peptidoform_fragment_evidence(&scrambled, Some(target_mass), &observed, 1, 20.0);
        assert!(target_score.score > scrambled_score.score);
        assert!(target_score.matched_cleavages >= 4);
    }

    #[test]
    fn mitm_complete_fragment_decomposition_matches_full_ladder() {
        let peptide = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let target_mass =
            redeem_properties::foundation::foundation_peptidoform_neutral_mass(&peptide).unwrap();
        let vocabulary = FoundationDiffusionVocabulary;
        let tokens = vocabulary.encode(&peptide, 32).unwrap();
        let eos = tokens
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
            .unwrap();
        let active = &tokens[..eos];
        let mut prefix_masses = Vec::new();
        let mut mass = 0.0;
        for &token in active {
            mass += foundation_diffusion_token_mass_da(token).unwrap();
            if foundation_diffusion_token_residue(token).is_some() {
                prefix_masses.push(mass);
            }
        }
        let mut peaks = Vec::new();
        for &prefix_mass in prefix_masses.iter().take(prefix_masses.len() - 1) {
            peaks.push(FoundationSpectrumPeak {
                mz: (prefix_mass + 1.007_276_466_77) as f32,
                intensity: 1.0,
            });
        }
        let observed = normalized_observed_peaks(&FoundationSpectrum { peaks });
        let split = 3usize;
        let score_at = |residues: usize| {
            cleavage_fragment_evidence(prefix_masses[residues - 1], target_mass, &observed, 1, 20.0)
                .score
        };
        let prefix_internal: f64 = (1..split).map(|residues| score_at(residues)).sum();
        let seam = score_at(split);
        let suffix_internal: f64 = ((split + 1)..prefix_masses.len())
            .map(|residues| score_at(residues))
            .sum();
        let full = peptidoform_fragment_evidence(&peptide, Some(target_mass), &observed, 1, 20.0);
        assert!((prefix_internal + seam + suffix_internal - full.score).abs() < 1e-12);
    }

    #[test]
    fn mitm_reverse_partial_canonicalizes_suffix_units() {
        let vocabulary = FoundationDiffusionVocabulary;
        let suffix = vocabulary
            .encode(
                &PeptidoformInput {
                    sequence: "TIDEK".into(),
                    modifications: Vec::new(),
                },
                16,
            )
            .unwrap();
        let eos = suffix
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
            .unwrap();
        let reverse = foundation_reverse_causal_token_row(&suffix).unwrap();
        let reverse_eos = reverse
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
            .unwrap();
        let (has_nterm, canonical_body) =
            mitm_reverse_partial_to_canonical_suffix(&reverse[..reverse_eos], 16).unwrap();
        assert!(!has_nterm);
        assert_eq!(canonical_body, suffix[..eos]);
    }

    #[test]
    fn mitm_join_reconstructs_mass_complementary_peptide() {
        let vocabulary = FoundationDiffusionVocabulary;
        let peptide = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let canonical = vocabulary.encode(&peptide, 32).unwrap();
        let eos = canonical
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
            .unwrap();
        let active = &canonical[..eos];
        let split = 3usize;
        let prefix_tokens = active[..split].to_vec();
        let suffix_tokens = active[split..].to_vec();

        let mut suffix_row = vec![FOUNDATION_DIFFUSION_PAD; 32];
        for (position, &token) in suffix_tokens.iter().enumerate() {
            suffix_row[position] = token;
        }
        suffix_row[suffix_tokens.len()] = FOUNDATION_DIFFUSION_EOS;
        let reverse_suffix_row = foundation_reverse_causal_token_row(&suffix_row).unwrap();
        let reverse_eos = reverse_suffix_row
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
            .unwrap();
        let reverse_suffix_tokens = reverse_suffix_row[..reverse_eos].to_vec();

        let token_mass = |tokens: &[u32]| -> f64 {
            tokens
                .iter()
                .map(|&token| foundation_diffusion_token_mass_da(token).unwrap())
                .sum()
        };
        let prefix_state = MitmPartialState {
            assigned_mass_da: token_mass(&prefix_tokens),
            tokens: prefix_tokens,
            ar_total_log_probability: -1.0,
            fragment_score: 1.0,
            matched_cleavages: 1,
            residue_count: split,
            priority: 0.9,
        };
        let suffix_state = MitmPartialState {
            assigned_mass_da: token_mass(&reverse_suffix_tokens),
            tokens: reverse_suffix_tokens,
            ar_total_log_probability: -1.0,
            fragment_score: 1.0,
            matched_cleavages: 1,
            residue_count: active.len() - split,
            priority: 0.8,
        };
        let target_neutral_mass =
            redeem_properties::foundation::foundation_peptidoform_neutral_mass(&peptide).unwrap();
        let (before_cap, joined, legacy_shadow, v01320_shadow, precap_audit_pool) =
            bidirectional_mitm_join(
                &[prefix_state],
                &[suffix_state],
                Some(target_neutral_mass),
                0.05,
                32,
                vocabulary,
                16,
                &[],
                1,
                20.0,
                FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123,
                false,
                false,
                false,
            )
            .unwrap();
        assert_eq!(before_cap, 1);
        assert_eq!(joined.len(), 1);
        assert!(legacy_shadow.is_empty());
        assert!(v01320_shadow.is_empty());
        assert!(precap_audit_pool.is_empty());
        assert_eq!(joined[0].tokens, canonical);
        assert!(joined[0].join_mass_error_da.abs() <= 0.05);
    }

    #[test]
    fn mitm_v01320_selector_can_displace_legacy_partial_priority() {
        let legacy_favored = MitmJoinedCandidate {
            tokens: vec![1, 2, 3],
            proposal_score: 10.0,
            selector_score: 2.0,
            selector_fragment_score: 2.2,
            selector_matched_cleavages: 2,
            join_mass_error_da: 0.001,
            ..Default::default()
        };
        let evidence_favored = MitmJoinedCandidate {
            tokens: vec![4, 5, 6],
            proposal_score: 9.0,
            selector_score: 5.0,
            selector_fragment_score: 5.2,
            selector_matched_cleavages: 5,
            join_mass_error_da: 0.002,
            ..Default::default()
        };

        let legacy = mitm_retain_top_candidates(
            vec![legacy_favored.clone(), evidence_favored.clone()],
            1,
            false,
        );
        let evidence =
            mitm_retain_top_candidates(vec![legacy_favored, evidence_favored.clone()], 1, true);
        assert_eq!(legacy[0].tokens, vec![1, 2, 3]);
        assert_eq!(evidence[0].tokens, evidence_favored.tokens);
    }

    #[test]
    fn mitm_v01321_precap_audit_reports_exact_and_il_ranks_without_selection() {
        let vocabulary = FoundationDiffusionVocabulary;
        let target = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let target_tokens = vocabulary.encode(&target, 32).unwrap();
        let distractor = PeptidoformInput {
            sequence: "PEPTLDEK".into(),
            modifications: Vec::new(),
        };
        let distractor_tokens = vocabulary.encode(&distractor, 32).unwrap();
        let candidates = vec![
            MitmJoinedCandidate {
                tokens: distractor_tokens,
                proposal_score: 10.0,
                selector_score: 1.0,
                selector_fragment_score: 1.0,
                selector_matched_cleavages: 1,
                join_mass_error_da: 0.001,
                ..Default::default()
            },
            MitmJoinedCandidate {
                tokens: target_tokens,
                proposal_score: 9.0,
                selector_score: 5.0,
                selector_fragment_score: 5.0,
                selector_matched_cleavages: 5,
                join_mass_error_da: 0.002,
                ..Default::default()
            },
        ];
        let audit = mitm_precap_oracle_audit(&candidates, &target, 32, vocabulary).unwrap();
        assert_eq!(audit.candidates, 2);
        assert!(audit.peptidoform_present);
        assert!(audit.sequence_present);
        assert!(audit.il_sequence_present);
        assert_eq!(audit.legacy_ranks.peptidoform, Some(2));
        assert_eq!(audit.legacy_ranks.sequence, Some(2));
        assert_eq!(audit.legacy_ranks.il_sequence, Some(1));
        assert_eq!(audit.evidence_ranks.peptidoform, Some(1));
        assert_eq!(audit.evidence_ranks.sequence, Some(1));
        assert_eq!(audit.evidence_ranks.il_sequence, Some(1));
    }

    #[test]
    fn mitm_v01322_component_audit_exposes_actionable_individual_signal() {
        let vocabulary = FoundationDiffusionVocabulary;
        let target = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let distractor = PeptidoformInput {
            sequence: "AAAAAAAK".into(),
            modifications: Vec::new(),
        };
        let target_tokens = vocabulary.encode(&target, 32).unwrap();
        let distractor_tokens = vocabulary.encode(&distractor, 32).unwrap();
        let candidates = vec![
            MitmJoinedCandidate {
                tokens: distractor_tokens,
                selector_score: 5.0,
                selector_fragment_score: 1.0,
                component_fragment_score: 1.0,
                component_prefix_ar_mean: -1.0,
                component_suffix_ar_mean: -1.0,
                component_prefix_ar_total: -5.0,
                component_suffix_ar_total: -5.0,
                component_join_seam_fragment_score: 0.5,
                join_mass_error_da: 0.001,
                ..Default::default()
            },
            MitmJoinedCandidate {
                tokens: target_tokens,
                selector_score: 1.0,
                selector_fragment_score: 10.0,
                component_fragment_score: 10.0,
                component_prefix_ar_mean: -2.0,
                component_suffix_ar_mean: -2.0,
                component_prefix_ar_total: -10.0,
                component_suffix_ar_total: -10.0,
                component_join_seam_fragment_score: 2.0,
                join_mass_error_da: 0.002,
                selector_prefix_partial_rank: 400,
                selector_suffix_partial_rank: 500,
                selector_prefix_token_count: 4,
                selector_suffix_token_count: 4,
                selector_prefix_residue_count: 4,
                selector_suffix_residue_count: 4,
                selector_join_seam_fragment_score: 2.0,
                ..Default::default()
            },
        ];
        let audit = mitm_component_rank_audit(&candidates, &target, 32, vocabulary, true).unwrap();
        assert_eq!(audit.ranks.fragment.peptidoform, Some(1));
        assert!(mitm_component_rank_any_top256(&audit.ranks, false));
        assert!(audit.pareto_peptidoform_present);
        assert!(audit.pareto_frontier_size <= 256);
        assert!(audit.il_provenance.found);
    }

    #[test]
    fn mitm_v01323_two_view_selector_preserves_evidence_core_and_adds_seam_view() {
        fn candidate(token: u32, evidence: f64, seam: f64) -> MitmJoinedCandidate {
            MitmJoinedCandidate {
                tokens: vec![token],
                selector_score: evidence,
                selector_fragment_score: evidence,
                component_join_seam_fragment_score: seam,
                ..Default::default()
            }
        }

        let candidates = vec![
            candidate(1, 10.0, 10.0),
            candidate(2, 9.0, 9.0),
            candidate(3, 8.0, 100.0),
            candidate(4, 7.0, 8.0),
            candidate(5, 6.0, 7.0),
        ];
        let selected = mitm_retain_v01323_two_view_candidates(&candidates, 4);
        let selected_tokens: HashSet<Vec<u32>> = selected
            .iter()
            .map(|candidate| candidate.tokens.clone())
            .collect();

        assert_eq!(selected.len(), 4);
        assert!(selected_tokens.contains(&vec![1]));
        assert!(selected_tokens.contains(&vec![2]));
        assert!(selected_tokens.contains(&vec![3]));
        assert!(selected_tokens.contains(&vec![4]));
        assert!(!selected_tokens.contains(&vec![5]));
    }
}
