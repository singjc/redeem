//! ReDeeM v0.75 practical charge + neutral-mass + deterministic fragment-geometry identifier.
//!
//! Scientific/engineering contract:
//! - build the source-closed DEV candidate catalog directly from the prepared corpus/benchmark;
//! - reconstruct the frozen v0.70/v0.73 query cohort from fixed historical selection seeds only;
//! - require a positive observed precursor charge and precursor m/z;
//! - retrieve exactly the nearest 256 same-charge theoretical-neutral-mass candidates when available;
//! - never insert or force the true target into the candidate pool;
//! - rank candidates only by the validated v0.74.2 deterministic b1/b2/y1/y2 geometry score;
//! - use the exact v0.74.2 observed-peak normalization/capping and fragment support semantics;
//! - require no v0.70 checkpoint, no v0.52 checkpoint, no optimizer, and no training;
//! - use DEV only for the bounded implementation-reproduction audit;
//! - expose one explicitly confirmed TRAIN-HOLDOUT evaluation mode after the DEV implementation is frozen;
//! - never use TRAIN-HOLDOUT for selection and never consume historical VALIDATION/APD or TEST.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    foundation_peptidoform_neutral_mass, foundation_precursor_neutral_mass, load_foundation_corpus,
    read_foundation_training_run_config, FoundationBenchmarkManifest, FoundationPartition,
    FoundationPracticalIdentifierCandidateV0751, FoundationPracticalIdentifierV0751,
    FoundationSpectrum, FoundationTrainingRecord, PeptidoformInput,
    FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POLICY_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751,
};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const V075_VERSION: u32 = 750;
const V075_OBJECTIVE: &str = "v0750_charge_mass_geometry_identifier";
const V075_ARCHITECTURE: &str = FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751;
const V075_SCORE: &str = FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751;
const V075_CANDIDATE_POLICY: &str = FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POLICY_V0751;

// Frozen cohort-construction values used by v0.70-v0.74.2. These are provenance constants only;
// v0.75 does not load or execute the v0.70 model.
const V070_SELECTION_SEED: u64 = 20_261_070;
const V070_DEV_IDENTITIES: usize = 2048;
const V073_QUERY_SEED: u64 = 20_261_073;
const V075_SMOKE_QUERIES: usize = 32;
const V075_AUDIT_QUERIES: usize = 512;
const V075_HOLDOUT_QUERIES: usize = V070_DEV_IDENTITIES;
const V075_HOLDOUT_QUERY_SEED: u64 = 20_261_075;
const V075_HOLDOUT_RECORD_SEED: u64 = 20_261_075 ^ 0x7500_001d_3a7a_0001;
const V075_HOLDOUT_CONFIRM_TOKEN: &str = "CONFIRM_TRAIN_HOLDOUT_ONCE";
const V075_MAX_SEQUENCE_LEN: usize = 64;
const V075_MASS_POOL: usize = FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751;
const V075_TARGET_FORCING: bool = false;
const V075_EXPECTED_PARENT_DEV_FINGERPRINT: &str = "fnv1a64:2aa9a31055c720a6";
const V075_EXPECTED_CANDIDATE_UNIVERSE: usize = 63_332;
const V075_EXPECTED_CANDIDATE_FINGERPRINT: &str = "fnv1a64:a5699533553116e8";
const V075_EXPECTED_CORPUS_FINGERPRINT: &str = "fnv1a64:a2a6f57d31064ba6";
const V075_EXPECTED_BENCHMARK_FINGERPRINT: &str = "fnv1a64:2133c039625f77df";

// Reuse the already-declared v0.74 geometry promotion floors. These are reproduction guards,
// not a new architecture-selection sweep.
const V075_MIN_IL_COVERAGE: f64 = 1.0;
const V075_MIN_GEOMETRY_IL_TOP1: f64 = 0.85;
const V075_MIN_GEOMETRY_IL_TOP10: f64 = 0.97;
// The one-time protected confirmation reuses the already-declared v0.74/v0.75 floors rather than
// introducing a HOLDOUT-tuned criterion. Candidate coverage may fall below 1.0 on an unseen
// partition, so its confirmatory floor is fixed at 0.99 before looking at HOLDOUT results.
const V075_HOLDOUT_MIN_IL_COVERAGE: f64 = 0.99;

#[derive(Debug, Clone)]
struct AlignmentGroup {
    key: String,
    peptidoform: String,
    sequence: String,
    charge: i32,
    record_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct CandidateIdentity {
    record_index: usize,
    exact_key: String,
    il_key: String,
    charge: i32,
    observed_neutral_mass: f64,
    candidate_neutral_mass: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct RankMetrics {
    exact_top1: f64,
    exact_top10: f64,
    exact_mrr: f64,
    il_top1: f64,
    il_top10: f64,
    il_mrr: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct RankAccumulator {
    queries: usize,
    exact_top1: usize,
    exact_top10: usize,
    exact_rr: f64,
    il_top1: usize,
    il_top10: usize,
    il_rr: f64,
}

impl RankAccumulator {
    fn observe(&mut self, exact_rank: Option<usize>, il_rank: Option<usize>) {
        self.queries += 1;
        if let Some(rank) = exact_rank {
            self.exact_top1 += usize::from(rank == 1);
            self.exact_top10 += usize::from(rank <= 10);
            self.exact_rr += 1.0 / rank as f64;
        }
        if let Some(rank) = il_rank {
            self.il_top1 += usize::from(rank == 1);
            self.il_top10 += usize::from(rank <= 10);
            self.il_rr += 1.0 / rank as f64;
        }
    }

    fn metrics(self) -> RankMetrics {
        if self.queries == 0 {
            return RankMetrics::default();
        }
        let denom = self.queries as f64;
        RankMetrics {
            exact_top1: self.exact_top1 as f64 / denom,
            exact_top10: self.exact_top10 as f64 / denom,
            exact_mrr: self.exact_rr / denom,
            il_top1: self.il_top1 as f64 / denom,
            il_top10: self.il_top10 as f64 / denom,
            il_mrr: self.il_rr / denom,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct V075Metrics {
    queries: usize,
    candidate_universe: usize,
    candidate_pool_target: usize,
    candidate_count_min: usize,
    candidate_count_max: usize,
    candidate_count_mean: f64,
    exact_candidate_coverage: f64,
    il_candidate_coverage: f64,
    geometry: RankMetrics,
    target_beats_best_il_negative_fraction: f64,
    mean_target_score: f64,
    mean_best_il_negative_score: f64,
    mean_target_minus_best_il_negative_margin: f64,
    catalog_build_seconds: f64,
    index_build_seconds: f64,
    geometry_precompute_seconds: f64,
    scoring_seconds: f64,
    learned_candidate_embedding_seconds: f64,
    elapsed_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
struct V075Metadata {
    version: u32,
    objective: String,
    architecture: String,
    score: String,
    candidate_policy: String,
    mode: String,
    evaluation_partition: String,
    protected_evaluation: bool,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_dev_identity_fingerprint: String,
    candidate_universe_fingerprint: String,
    expected_candidate_universe_fingerprint: String,
    query_cohort_fingerprint: String,
    audit_queries: usize,
    candidate_pool: usize,
    max_sequence_len: usize,
    target_forcing: bool,
    v070_checkpoint_required: bool,
    v052_checkpoint_required: bool,
    optimizer_present: bool,
    training_performed: bool,
    threshold_tuning_performed: bool,
    candidate_pool_sweep_performed: bool,
    metrics: V075Metrics,
    train_holdout_consumed: bool,
    train_holdout_consumed_for_selection: bool,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

#[derive(Debug, Clone)]
struct QueryDiagnostic {
    query_slot: usize,
    selected_identity_index: usize,
    full_identity_index: usize,
    exact_key: String,
    charge: i32,
    observed_neutral_mass: f64,
    candidate_count: usize,
    exact_covered: bool,
    il_covered: bool,
    exact_rank: Option<usize>,
    il_rank: Option<usize>,
    target_score: Option<f64>,
    best_il_negative_score: f64,
    margin: Option<f64>,
    top_candidate: String,
    top_score: f64,
    query_scoring_seconds: f64,
}

#[derive(Debug, Clone)]
struct RankedCandidateRow {
    query_slot: usize,
    rank: usize,
    candidate_index: usize,
    candidate_exact_key: String,
    candidate_il_key: String,
    charge: i32,
    theoretical_neutral_mass: f64,
    absolute_neutral_mass_error: f64,
    geometry_score: f64,
    exact_target: bool,
    il_target: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        self_test()?;
        println!("v0750_charge_mass_geometry_identifier_self_test=PASS");
        return Ok(());
    }
    if !(args.len() == 4 || args.len() == 5) {
        anyhow::bail!(
            "usage: foundation_identify_charge_mass_geometry_v0750 RUN_V0260.yaml OUTPUT_DIR mode=smoke|audit|holdout [CONFIRM_TRAIN_HOLDOUT_ONCE]"
        );
    }

    let training_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let mode = args[3].as_str();
    if !matches!(mode, "smoke" | "audit" | "holdout") {
        anyhow::bail!("v0.75 mode must be smoke, audit, or holdout");
    }
    let holdout_mode = mode == "holdout";
    if holdout_mode {
        if args.len() != 5 || args[4] != V075_HOLDOUT_CONFIRM_TOKEN {
            anyhow::bail!(
                "v0.75 HOLDOUT is protected and requires the explicit token {V075_HOLDOUT_CONFIRM_TOKEN}"
            );
        }
    } else if args.len() != 4 {
        anyhow::bail!("v0.75 smoke/audit modes do not accept a HOLDOUT confirmation token");
    }
    if output_root.exists() {
        anyhow::bail!("v0.75 output directory must be fresh: {output_root:?}");
    }

    let evaluation_queries = match mode {
        "smoke" => V075_SMOKE_QUERIES,
        "audit" => V075_AUDIT_QUERIES,
        "holdout" => V075_HOLDOUT_QUERIES,
        _ => unreachable!(),
    };
    let evaluation_partition = if holdout_mode {
        "TRAIN_HOLDOUT_ONCE"
    } else {
        "DEV_IMPLEMENTATION_REPRODUCTION_ONLY"
    };
    let total_started = Instant::now();

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let current_corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let current_benchmark_fingerprint =
        format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if holdout_mode {
        if current_corpus_fingerprint != V075_EXPECTED_CORPUS_FINGERPRINT {
            anyhow::bail!(
                "v0.75 HOLDOUT corpus fingerprint mismatch: current={} expected={}",
                current_corpus_fingerprint,
                V075_EXPECTED_CORPUS_FINGERPRINT
            );
        }
        if current_benchmark_fingerprint != V075_EXPECTED_BENCHMARK_FINGERPRINT {
            anyhow::bail!(
                "v0.75 HOLDOUT benchmark fingerprint mismatch: current={} expected={}",
                current_benchmark_fingerprint,
                V075_EXPECTED_BENCHMARK_FINGERPRINT
            );
        }
    }

    let catalog_started = Instant::now();
    let (
        selected_fingerprint,
        full_identities,
        full_fingerprint,
        selected_query_indices,
        full_query_indices,
        query_cohort_fingerprint,
    ) = if holdout_mode {
        // In the prepared benchmark, FoundationPartition::Test is the reserved TRAIN-HOLDOUT.
        // Historical TEST is a separate external evaluation resource and is not accessed here.
        let holdout_groups = build_alignment_groups(
            &corpus.records,
            &benchmark,
            FoundationPartition::Test,
            V075_MAX_SEQUENCE_LEN,
        )?;
        let holdout_identities = build_all_partition_identities(
            &corpus.records,
            &holdout_groups,
            V075_HOLDOUT_RECORD_SEED,
        )?;
        if holdout_identities.len() < V075_HOLDOUT_QUERIES {
            anyhow::bail!(
                "v0.75 HOLDOUT expected at least {V075_HOLDOUT_QUERIES} eligible identities, observed {}",
                holdout_identities.len()
            );
        }
        let query_indices = deterministic_eval_queries(
            &holdout_identities,
            V075_HOLDOUT_QUERIES,
            V075_HOLDOUT_QUERY_SEED,
        );
        let query_identities = query_indices
            .iter()
            .map(|&index| holdout_identities[index].clone())
            .collect::<Vec<_>>();
        let query_fingerprint = format!("fnv1a64:{:016x}", identity_fingerprint(&query_identities));
        let universe_fingerprint =
            format!("fnv1a64:{:016x}", identity_fingerprint(&holdout_identities));
        (
            "NOT_APPLICABLE_HOLDOUT".to_string(),
            holdout_identities,
            universe_fingerprint,
            query_indices.clone(),
            query_indices,
            query_fingerprint,
        )
    } else {
        let dev_groups = build_alignment_groups(
            &corpus.records,
            &benchmark,
            FoundationPartition::Validation,
            V075_MAX_SEQUENCE_LEN,
        )?;
        let identity_seed = V070_SELECTION_SEED ^ 0x7000_d3f0_a11e_0001;
        let selected_identities = select_dev_identities(
            &corpus.records,
            &dev_groups,
            V070_DEV_IDENTITIES,
            identity_seed,
        )?;
        if selected_identities.len() != V070_DEV_IDENTITIES {
            anyhow::bail!(
                "v0.75 expected {V070_DEV_IDENTITIES} frozen parent DEV identities, observed {}",
                selected_identities.len()
            );
        }
        let parent_fingerprint = format!(
            "fnv1a64:{:016x}",
            identity_fingerprint(&selected_identities)
        );
        if parent_fingerprint != V075_EXPECTED_PARENT_DEV_FINGERPRINT {
            anyhow::bail!(
                "v0.75 frozen query-cohort parent fingerprint mismatch: current={} expected={}",
                parent_fingerprint,
                V075_EXPECTED_PARENT_DEV_FINGERPRINT
            );
        }

        let all_dev_identities = build_all_partition_identities(
            &corpus.records,
            &dev_groups,
            identity_seed.rotate_left(17),
        )?;
        let universe_fingerprint =
            format!("fnv1a64:{:016x}", identity_fingerprint(&all_dev_identities));
        if all_dev_identities.len() != V075_EXPECTED_CANDIDATE_UNIVERSE {
            anyhow::bail!(
                "v0.75 candidate-universe size mismatch: current={} expected={}",
                all_dev_identities.len(),
                V075_EXPECTED_CANDIDATE_UNIVERSE
            );
        }
        if universe_fingerprint != V075_EXPECTED_CANDIDATE_FINGERPRINT {
            anyhow::bail!(
                "v0.75 candidate-universe fingerprint mismatch: current={} expected={}",
                universe_fingerprint,
                V075_EXPECTED_CANDIDATE_FINGERPRINT
            );
        }

        let full_index = full_identity_index(&all_dev_identities)?;
        let query_indices = deterministic_eval_queries(
            &selected_identities,
            evaluation_queries,
            V073_QUERY_SEED ^ 0x7300_d3f0_0000_0001,
        );
        let query_identities = query_indices
            .iter()
            .map(|&index| selected_identities[index].clone())
            .collect::<Vec<_>>();
        let query_fingerprint = format!("fnv1a64:{:016x}", identity_fingerprint(&query_identities));
        let mut mapped_query_indices = Vec::with_capacity(query_indices.len());
        for &selected_index in &query_indices {
            let selected = &selected_identities[selected_index];
            let full_index_value = *full_index.get(&selected.exact_key).with_context(|| {
                format!(
                    "v0.75 frozen query {} absent from direct candidate catalog",
                    selected.exact_key
                )
            })?;
            if all_dev_identities[full_index_value].record_index != selected.record_index {
                anyhow::bail!(
                    "v0.75 frozen query record drift for {}: selected={} full={}",
                    selected.exact_key,
                    selected.record_index,
                    all_dev_identities[full_index_value].record_index
                );
            }
            mapped_query_indices.push(full_index_value);
        }
        (
            parent_fingerprint,
            all_dev_identities,
            universe_fingerprint,
            query_indices,
            mapped_query_indices,
            query_fingerprint,
        )
    };
    let catalog_build_seconds = catalog_started.elapsed().as_secs_f64();

    fs::create_dir_all(&output_root)?;
    write_candidate_catalog(&output_root.join("candidate_catalog.tsv"), &full_identities)?;

    let practical_candidates = full_identities
        .iter()
        .map(|identity| {
            FoundationPracticalIdentifierCandidateV0751::new(
                identity.exact_key.clone(),
                corpus.records[identity.record_index].peptidoform.clone(),
                identity.charge,
            )
        })
        .collect::<Vec<_>>();
    let practical_identifier = FoundationPracticalIdentifierV0751::new(practical_candidates)?;
    if practical_identifier.candidate_count() != full_identities.len() {
        anyhow::bail!("v0.75.1 practical identifier candidate-count drift");
    }
    let practical_build_timings = practical_identifier.build_timings();
    let index_build_seconds = practical_build_timings.index_build_seconds;
    let geometry_precompute_seconds = practical_build_timings.geometry_precompute_seconds;

    println!("v0750_version\tv0.75-charge-mass-geometry-identifier");
    println!("objective\t{V075_OBJECTIVE}");
    println!("architecture\t{V075_ARCHITECTURE}");
    println!("score\t{V075_SCORE}");
    println!("mode\t{mode}");
    println!("evaluation_partition\t{evaluation_partition}");
    println!("protected_evaluation\t{}", yes_no(holdout_mode));
    println!("execution_device\tCPU");
    println!("training_scope\tNONE");
    println!(
        "selection_scope\t{}",
        if holdout_mode {
            "FROZEN_IMPLEMENTATION_CONFIRMATION_ONLY"
        } else {
            "DEV_IMPLEMENTATION_REPRODUCTION_ONLY"
        }
    );
    println!(
        "candidate_universe_policy\t{}",
        if holdout_mode {
            "all_source_closed_train_holdout_eligible_unique_peptidoform_charge_identities"
        } else {
            "all_source_closed_dev_eligible_unique_peptidoform_charge_identities"
        }
    );
    println!("candidate_policy\t{V075_CANDIDATE_POLICY}");
    println!("precursor_charge_policy\tobserved_positive_charge_required");
    println!("target_forcing\tNO");
    println!("v070_checkpoint_required\tNO");
    println!("v052_checkpoint_required\tNO");
    println!("learned_candidate_embeddings\tNO");
    println!("learned_fragment_intensity\tNO");
    println!("optimizer_present\tNO");
    println!("training_performed\tNO");
    println!("threshold_tuning_performed\tNO");
    println!("candidate_pool_sweep_performed\tNO");
    println!("candidate_pool\t{V075_MASS_POOL}");
    println!("parent_dev_identity_fingerprint\t{selected_fingerprint}");
    println!("candidate_universe\t{}", full_identities.len());
    println!("candidate_universe_fingerprint\t{full_fingerprint}");
    println!("query_cohort_fingerprint\t{query_cohort_fingerprint}");
    println!("evaluation_queries\t{evaluation_queries}");
    println!(
        "dev_labels_used_for_analysis\t{}",
        if holdout_mode {
            "NO"
        } else {
            "YES_DEV_REPRODUCTION"
        }
    );
    println!("dev_partition_used_for_selection\tNO");
    println!("train_holdout_consumed\t{}", yes_no(holdout_mode));
    println!("train_holdout_consumed_for_selection\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let scoring_started = Instant::now();
    let (mut metrics, diagnostics, rankings) = identify_queries(
        &corpus.records,
        &full_identities,
        &full_query_indices,
        &selected_query_indices,
        &practical_identifier,
    )?;
    metrics.catalog_build_seconds = catalog_build_seconds;
    metrics.index_build_seconds = index_build_seconds;
    metrics.geometry_precompute_seconds = geometry_precompute_seconds;
    metrics.scoring_seconds = scoring_started.elapsed().as_secs_f64();
    metrics.learned_candidate_embedding_seconds = 0.0;
    metrics.elapsed_seconds = total_started.elapsed().as_secs_f64();

    print_metrics(&metrics);

    let gate_candidate_provenance = if holdout_mode {
        current_corpus_fingerprint == V075_EXPECTED_CORPUS_FINGERPRINT
            && current_benchmark_fingerprint == V075_EXPECTED_BENCHMARK_FINGERPRINT
    } else {
        full_fingerprint == V075_EXPECTED_CANDIDATE_FINGERPRINT
    };
    let gate_target_forcing = !V075_TARGET_FORCING;
    let gate_coverage = if holdout_mode {
        metrics.il_candidate_coverage >= V075_HOLDOUT_MIN_IL_COVERAGE
    } else {
        (metrics.il_candidate_coverage - V075_MIN_IL_COVERAGE).abs() <= 1.0e-12
    };
    let gate_top1 = metrics.geometry.il_top1 >= V075_MIN_GEOMETRY_IL_TOP1;
    let gate_top10 = metrics.geometry.il_top10 >= V075_MIN_GEOMETRY_IL_TOP10;
    let gate_no_learned_dependency = metrics.learned_candidate_embedding_seconds == 0.0;

    println!(
        "v0750_gate_candidate_provenance\t{}",
        pass_fail(gate_candidate_provenance)
    );
    println!(
        "v0750_gate_target_forcing_no\t{}",
        pass_fail(gate_target_forcing)
    );
    println!(
        "v0750_gate_mass256_il_coverage\t{}",
        pass_fail(gate_coverage)
    );
    println!(
        "v0750_gate_geometry_il_top1_ge_frozen_floor\t{}",
        pass_fail(gate_top1)
    );
    println!(
        "v0750_gate_geometry_il_top10_ge_frozen_floor\t{}",
        pass_fail(gate_top10)
    );
    println!(
        "v0750_gate_no_learned_candidate_embedding\t{}",
        pass_fail(gate_no_learned_dependency)
    );

    let all_confirmation_gates = gate_candidate_provenance
        && gate_target_forcing
        && gate_coverage
        && gate_top1
        && gate_top10
        && gate_no_learned_dependency;
    let decision = v075_decision(mode, all_confirmation_gates);
    println!("v0750_audit_decision\t{decision}");
    println!("train_holdout_consumed\t{}", yes_no(holdout_mode));
    println!("train_holdout_consumed_for_selection\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    write_metrics(&output_root.join("identifier_metrics.tsv"), &metrics)?;
    write_summary(
        &output_root.join("summary.tsv"),
        mode,
        evaluation_partition,
        &current_corpus_fingerprint,
        &current_benchmark_fingerprint,
        &selected_fingerprint,
        &full_fingerprint,
        &query_cohort_fingerprint,
        decision,
        holdout_mode,
        &metrics,
    )?;
    write_query_diagnostics(&output_root.join("query_diagnostics.tsv"), &diagnostics)?;
    write_rankings(&output_root.join("ranked_candidates.tsv"), &rankings)?;

    let metadata = V075Metadata {
        version: V075_VERSION,
        objective: V075_OBJECTIVE.to_string(),
        architecture: V075_ARCHITECTURE.to_string(),
        score: V075_SCORE.to_string(),
        candidate_policy: V075_CANDIDATE_POLICY.to_string(),
        mode: mode.to_string(),
        evaluation_partition: evaluation_partition.to_string(),
        protected_evaluation: holdout_mode,
        corpus_fingerprint: current_corpus_fingerprint,
        benchmark_manifest_fingerprint: current_benchmark_fingerprint,
        parent_dev_identity_fingerprint: selected_fingerprint,
        candidate_universe_fingerprint: full_fingerprint,
        expected_candidate_universe_fingerprint: if holdout_mode {
            "NOT_PREINSPECTED_PROTECTED_PARTITION".to_string()
        } else {
            V075_EXPECTED_CANDIDATE_FINGERPRINT.to_string()
        },
        query_cohort_fingerprint,
        audit_queries: evaluation_queries,
        candidate_pool: V075_MASS_POOL,
        max_sequence_len: V075_MAX_SEQUENCE_LEN,
        target_forcing: V075_TARGET_FORCING,
        v070_checkpoint_required: false,
        v052_checkpoint_required: false,
        optimizer_present: false,
        training_performed: false,
        threshold_tuning_performed: false,
        candidate_pool_sweep_performed: false,
        metrics,
        train_holdout_consumed: holdout_mode,
        train_holdout_consumed_for_selection: false,
        historical_validation_consumed: false,
        historical_test_consumed: false,
    };
    fs::write(
        output_root.join("metadata.yaml"),
        serde_yaml::to_string(&metadata)?,
    )?;

    Ok(())
}

fn build_alignment_groups(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    max_sequence_len: usize,
) -> Result<Vec<AlignmentGroup>> {
    let mut groups = BTreeMap::<String, AlignmentGroup>::new();
    for entry in benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
    {
        let record = records
            .get(entry.record_index)
            .context("v0.75 benchmark record index out of range")?;
        if record.peptidoform.sequence.chars().count() > max_sequence_len {
            continue;
        }
        let Some(charge) = record.context.charge else {
            continue;
        };
        if charge <= 0 || record.context.precursor_mz.is_none() {
            continue;
        }
        if FoundationSpectrum::from_training_record(record).is_none() {
            continue;
        }
        let key = format!("{}|z{charge}", entry.peptidoform);
        let group = groups.entry(key.clone()).or_insert_with(|| AlignmentGroup {
            key,
            peptidoform: entry.peptidoform.clone(),
            sequence: entry.sequence.clone(),
            charge,
            record_indices: Vec::new(),
        });
        if group.peptidoform != entry.peptidoform
            || group.sequence != entry.sequence
            || group.charge != charge
        {
            anyhow::bail!("v0.75 identity grouping collision");
        }
        group.record_indices.push(entry.record_index);
    }
    Ok(groups.into_values().collect())
}

fn select_dev_identities(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    count: usize,
    seed: u64,
) -> Result<Vec<CandidateIdentity>> {
    let mut group_order = (0..groups.len()).collect::<Vec<_>>();
    group_order.sort_by_key(|&index| mix64(seed ^ hash64_str(&groups[index].key)));
    let mut selected = Vec::with_capacity(count);
    for group_index in group_order.into_iter().take(count) {
        selected.push(identity_from_group(
            records,
            &groups[group_index],
            seed.rotate_left(17),
        )?);
    }
    validate_unique_identities(&selected)?;
    Ok(selected)
}

fn build_all_partition_identities(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    record_seed: u64,
) -> Result<Vec<CandidateIdentity>> {
    let mut identities = Vec::with_capacity(groups.len());
    for group in groups {
        identities.push(identity_from_group(records, group, record_seed)?);
    }
    validate_unique_identities(&identities)?;
    Ok(identities)
}

fn identity_from_group(
    records: &[FoundationTrainingRecord],
    group: &AlignmentGroup,
    seed: u64,
) -> Result<CandidateIdentity> {
    let mut record_indices = group.record_indices.clone();
    record_indices.sort_by_key(|&index| mix64(seed ^ index as u64));
    let record_index = *record_indices
        .first()
        .context("v0.75 identity group has no records")?;
    let record = &records[record_index];
    let mz = record
        .context
        .precursor_mz
        .context("v0.75 identity record lacks precursor m/z")?;
    if group.charge <= 0 {
        anyhow::bail!("v0.75 observed precursor charge must be positive");
    }
    let observed_neutral_mass = foundation_precursor_neutral_mass(f64::from(mz), group.charge)
        .map_err(anyhow::Error::msg)?;
    let candidate_neutral_mass =
        foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
    Ok(CandidateIdentity {
        record_index,
        exact_key: group.key.clone(),
        il_key: format!("{}|z{}", il_label(&group.peptidoform), group.charge),
        charge: group.charge,
        observed_neutral_mass,
        candidate_neutral_mass,
    })
}

fn validate_unique_identities(identities: &[CandidateIdentity]) -> Result<()> {
    let unique = identities
        .iter()
        .map(|item| item.exact_key.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() != identities.len() {
        anyhow::bail!("v0.75 candidate catalog contains duplicate exact identities");
    }
    Ok(())
}

fn full_identity_index(identities: &[CandidateIdentity]) -> Result<BTreeMap<String, usize>> {
    let mut out = BTreeMap::new();
    for (index, identity) in identities.iter().enumerate() {
        if out.insert(identity.exact_key.clone(), index).is_some() {
            anyhow::bail!("v0.75 duplicate exact identity while indexing candidate catalog");
        }
    }
    Ok(out)
}

fn deterministic_eval_queries(
    identities: &[CandidateIdentity],
    count: usize,
    seed: u64,
) -> Vec<usize> {
    let mut indices = (0..identities.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&index| mix64(seed ^ hash64_str(&identities[index].exact_key)));
    indices.truncate(count.min(indices.len()));
    indices
}

fn identity_fingerprint(identities: &[CandidateIdentity]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for item in identities {
        for byte in item.exact_key.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= item.record_index as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn optional_exact_rank(
    ranked: &[usize],
    identities: &[CandidateIdentity],
    query: &CandidateIdentity,
) -> Option<usize> {
    ranked
        .iter()
        .position(|&index| identities[index].exact_key == query.exact_key)
        .map(|rank| rank + 1)
}

fn optional_il_rank(
    ranked: &[usize],
    identities: &[CandidateIdentity],
    query: &CandidateIdentity,
) -> Option<usize> {
    ranked
        .iter()
        .position(|&index| identities[index].il_key == query.il_key)
        .map(|rank| rank + 1)
}

fn pool_exact_covered(
    pool: &[usize],
    identities: &[CandidateIdentity],
    query: &CandidateIdentity,
) -> bool {
    pool.iter()
        .any(|&index| identities[index].exact_key == query.exact_key)
}

fn pool_il_covered(
    pool: &[usize],
    identities: &[CandidateIdentity],
    query: &CandidateIdentity,
) -> bool {
    pool.iter()
        .any(|&index| identities[index].il_key == query.il_key)
}

fn identify_queries(
    records: &[FoundationTrainingRecord],
    identities: &[CandidateIdentity],
    full_query_indices: &[usize],
    selected_query_indices: &[usize],
    practical_identifier: &FoundationPracticalIdentifierV0751,
) -> Result<(V075Metrics, Vec<QueryDiagnostic>, Vec<RankedCandidateRow>)> {
    if full_query_indices.len() != selected_query_indices.len()
        || practical_identifier.candidate_count() != identities.len()
    {
        anyhow::bail!("v0.75 identifier input shape mismatch");
    }

    let mut rank_acc = RankAccumulator::default();
    let mut diagnostics = Vec::with_capacity(full_query_indices.len());
    let mut rankings = Vec::with_capacity(full_query_indices.len() * V075_MASS_POOL);
    let mut exact_covered_count = 0usize;
    let mut il_covered_count = 0usize;
    let mut candidate_counts = Vec::with_capacity(full_query_indices.len());
    let mut target_scores = Vec::new();
    let mut negative_scores = Vec::new();
    let mut margins = Vec::new();
    let mut target_beats = 0usize;
    let mut margin_queries = 0usize;

    for (query_slot, (&query_index, &selected_identity_index)) in full_query_indices
        .iter()
        .zip(selected_query_indices)
        .enumerate()
    {
        let query_started = Instant::now();
        let query = &identities[query_index];
        if query.charge <= 0 {
            anyhow::bail!("v0.75 observed precursor charge must be positive");
        }
        let query_record = &records[query.record_index];
        let spectrum = FoundationSpectrum::from_training_record(query_record)
            .ok_or_else(|| anyhow::anyhow!("v0.75 query lacks observed spectrum"))?;
        let hits = practical_identifier.identify_neutral_mass(
            query.charge,
            query.observed_neutral_mass,
            &spectrum,
        )?;
        let mass256 = hits
            .iter()
            .map(|hit| hit.candidate_index)
            .collect::<Vec<_>>();
        if mass256.is_empty() {
            anyhow::bail!("v0.75 same-charge mass candidate pool is empty");
        }

        let exact_covered = pool_exact_covered(&mass256, identities, query);
        let il_covered = pool_il_covered(&mass256, identities, query);
        exact_covered_count += usize::from(exact_covered);
        il_covered_count += usize::from(il_covered);
        candidate_counts.push(mass256.len());

        let scored = hits
            .iter()
            .map(|hit| (hit.candidate_index, hit.geometry_score))
            .collect::<Vec<_>>();
        let ranked = scored.iter().map(|row| row.0).collect::<Vec<_>>();
        let exact_rank = optional_exact_rank(&ranked, identities, query);
        let il_rank = optional_il_rank(&ranked, identities, query);
        rank_acc.observe(exact_rank, il_rank);

        let target_score = scored
            .iter()
            .filter(|(index, _)| identities[*index].il_key == query.il_key)
            .map(|(_, score)| *score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let best_il_negative_score = scored
            .iter()
            .filter(|(index, _)| identities[*index].il_key != query.il_key)
            .map(|(_, score)| *score)
            .fold(0.0f64, f64::max);
        let margin = target_score.map(|score| score - best_il_negative_score);
        if let Some(score) = target_score {
            target_scores.push(score);
            negative_scores.push(best_il_negative_score);
            let value = score - best_il_negative_score;
            margins.push(value);
            target_beats += usize::from(value > 0.0);
            margin_queries += 1;
        }

        for (rank0, &(candidate_index, score)) in scored.iter().enumerate() {
            let candidate = &identities[candidate_index];
            rankings.push(RankedCandidateRow {
                query_slot,
                rank: rank0 + 1,
                candidate_index,
                candidate_exact_key: candidate.exact_key.clone(),
                candidate_il_key: candidate.il_key.clone(),
                charge: candidate.charge,
                theoretical_neutral_mass: candidate.candidate_neutral_mass,
                absolute_neutral_mass_error: (candidate.candidate_neutral_mass
                    - query.observed_neutral_mass)
                    .abs(),
                geometry_score: score,
                exact_target: candidate.exact_key == query.exact_key,
                il_target: candidate.il_key == query.il_key,
            });
        }

        diagnostics.push(QueryDiagnostic {
            query_slot,
            selected_identity_index,
            full_identity_index: query_index,
            exact_key: query.exact_key.clone(),
            charge: query.charge,
            observed_neutral_mass: query.observed_neutral_mass,
            candidate_count: mass256.len(),
            exact_covered,
            il_covered,
            exact_rank,
            il_rank,
            target_score,
            best_il_negative_score,
            margin,
            top_candidate: identities[ranked[0]].exact_key.clone(),
            top_score: scored[0].1,
            query_scoring_seconds: query_started.elapsed().as_secs_f64(),
        });
    }

    let candidate_count_min = candidate_counts.iter().copied().min().unwrap_or(0);
    let candidate_count_max = candidate_counts.iter().copied().max().unwrap_or(0);
    let candidate_count_mean = if candidate_counts.is_empty() {
        0.0
    } else {
        candidate_counts.iter().sum::<usize>() as f64 / candidate_counts.len() as f64
    };

    Ok((
        V075Metrics {
            queries: full_query_indices.len(),
            candidate_universe: identities.len(),
            candidate_pool_target: V075_MASS_POOL,
            candidate_count_min,
            candidate_count_max,
            candidate_count_mean,
            exact_candidate_coverage: safe_fraction(exact_covered_count, full_query_indices.len()),
            il_candidate_coverage: safe_fraction(il_covered_count, full_query_indices.len()),
            geometry: rank_acc.metrics(),
            target_beats_best_il_negative_fraction: safe_fraction(target_beats, margin_queries),
            mean_target_score: mean(&target_scores),
            mean_best_il_negative_score: mean(&negative_scores),
            mean_target_minus_best_il_negative_margin: mean(&margins),
            catalog_build_seconds: 0.0,
            index_build_seconds: 0.0,
            geometry_precompute_seconds: 0.0,
            scoring_seconds: 0.0,
            learned_candidate_embedding_seconds: 0.0,
            elapsed_seconds: 0.0,
        },
        diagnostics,
        rankings,
    ))
}

fn print_metrics(metrics: &V075Metrics) {
    println!("v0750_queries\t{}", metrics.queries);
    println!("v0750_candidate_universe\t{}", metrics.candidate_universe);
    println!(
        "v0750_candidate_pool_target\t{}",
        metrics.candidate_pool_target
    );
    println!("v0750_candidate_count_min\t{}", metrics.candidate_count_min);
    println!(
        "v0750_candidate_count_mean\t{:.8}",
        metrics.candidate_count_mean
    );
    println!("v0750_candidate_count_max\t{}", metrics.candidate_count_max);
    println!(
        "v0750_exact_candidate_coverage\t{:.8}",
        metrics.exact_candidate_coverage
    );
    println!(
        "v0750_il_candidate_coverage\t{:.8}",
        metrics.il_candidate_coverage
    );
    println!(
        "v0750_geometry_exact_top1\t{:.8}",
        metrics.geometry.exact_top1
    );
    println!(
        "v0750_geometry_exact_top10\t{:.8}",
        metrics.geometry.exact_top10
    );
    println!(
        "v0750_geometry_exact_mrr\t{:.8}",
        metrics.geometry.exact_mrr
    );
    println!("v0750_geometry_il_top1\t{:.8}", metrics.geometry.il_top1);
    println!("v0750_geometry_il_top10\t{:.8}", metrics.geometry.il_top10);
    println!("v0750_geometry_il_mrr\t{:.8}", metrics.geometry.il_mrr);
    println!(
        "v0750_target_beats_best_il_negative_fraction\t{:.8}",
        metrics.target_beats_best_il_negative_fraction
    );
    println!("v0750_mean_target_score\t{:.8}", metrics.mean_target_score);
    println!(
        "v0750_mean_best_il_negative_score\t{:.8}",
        metrics.mean_best_il_negative_score
    );
    println!(
        "v0750_mean_target_minus_best_il_negative_margin\t{:.8}",
        metrics.mean_target_minus_best_il_negative_margin
    );
    println!(
        "v0750_catalog_build_seconds\t{:.6}",
        metrics.catalog_build_seconds
    );
    println!(
        "v0750_index_build_seconds\t{:.6}",
        metrics.index_build_seconds
    );
    println!(
        "v0750_geometry_precompute_seconds\t{:.6}",
        metrics.geometry_precompute_seconds
    );
    println!("v0750_scoring_seconds\t{:.6}", metrics.scoring_seconds);
    println!(
        "v0750_learned_candidate_embedding_seconds\t{:.6}",
        metrics.learned_candidate_embedding_seconds
    );
    println!("v0750_elapsed_seconds\t{:.6}", metrics.elapsed_seconds);
}

fn write_metrics(path: &Path, metrics: &V075Metrics) -> Result<()> {
    let rows = [
        ("queries", metrics.queries as f64),
        ("candidate_universe", metrics.candidate_universe as f64),
        (
            "candidate_pool_target",
            metrics.candidate_pool_target as f64,
        ),
        ("candidate_count_min", metrics.candidate_count_min as f64),
        ("candidate_count_mean", metrics.candidate_count_mean),
        ("candidate_count_max", metrics.candidate_count_max as f64),
        ("exact_candidate_coverage", metrics.exact_candidate_coverage),
        ("il_candidate_coverage", metrics.il_candidate_coverage),
        ("geometry_exact_top1", metrics.geometry.exact_top1),
        ("geometry_exact_top10", metrics.geometry.exact_top10),
        ("geometry_exact_mrr", metrics.geometry.exact_mrr),
        ("geometry_il_top1", metrics.geometry.il_top1),
        ("geometry_il_top10", metrics.geometry.il_top10),
        ("geometry_il_mrr", metrics.geometry.il_mrr),
        (
            "target_beats_best_il_negative_fraction",
            metrics.target_beats_best_il_negative_fraction,
        ),
        ("mean_target_score", metrics.mean_target_score),
        (
            "mean_best_il_negative_score",
            metrics.mean_best_il_negative_score,
        ),
        (
            "mean_target_minus_best_il_negative_margin",
            metrics.mean_target_minus_best_il_negative_margin,
        ),
        ("catalog_build_seconds", metrics.catalog_build_seconds),
        ("index_build_seconds", metrics.index_build_seconds),
        (
            "geometry_precompute_seconds",
            metrics.geometry_precompute_seconds,
        ),
        ("scoring_seconds", metrics.scoring_seconds),
        (
            "learned_candidate_embedding_seconds",
            metrics.learned_candidate_embedding_seconds,
        ),
        ("elapsed_seconds", metrics.elapsed_seconds),
    ];
    let mut text = String::from("metric\tvalue\n");
    for (name, value) in rows {
        text.push_str(&format!("{name}\t{value:.12}\n"));
    }
    fs::write(path, text)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    path: &Path,
    mode: &str,
    evaluation_partition: &str,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    parent_dev_fingerprint: &str,
    candidate_fingerprint: &str,
    query_cohort_fingerprint: &str,
    decision: &str,
    train_holdout_consumed: bool,
    metrics: &V075Metrics,
) -> Result<()> {
    let mut text = String::from("key\tvalue\n");
    let rows = [
        ("version", "v0.75".to_string()),
        ("objective", V075_OBJECTIVE.to_string()),
        ("architecture", V075_ARCHITECTURE.to_string()),
        ("score", V075_SCORE.to_string()),
        ("mode", mode.to_string()),
        ("evaluation_partition", evaluation_partition.to_string()),
        (
            "protected_evaluation",
            yes_no(train_holdout_consumed).to_string(),
        ),
        ("corpus_fingerprint", corpus_fingerprint.to_string()),
        (
            "benchmark_manifest_fingerprint",
            benchmark_fingerprint.to_string(),
        ),
        (
            "parent_dev_identity_fingerprint",
            parent_dev_fingerprint.to_string(),
        ),
        (
            "candidate_universe_fingerprint",
            candidate_fingerprint.to_string(),
        ),
        (
            "query_cohort_fingerprint",
            query_cohort_fingerprint.to_string(),
        ),
        ("candidate_universe", metrics.candidate_universe.to_string()),
        ("evaluation_queries", metrics.queries.to_string()),
        ("candidate_pool", V075_MASS_POOL.to_string()),
        ("target_forcing", "NO".to_string()),
        ("v070_checkpoint_required", "NO".to_string()),
        ("v052_checkpoint_required", "NO".to_string()),
        ("optimizer_present", "NO".to_string()),
        ("training_performed", "NO".to_string()),
        ("threshold_tuning_performed", "NO".to_string()),
        ("candidate_pool_sweep_performed", "NO".to_string()),
        (
            "il_candidate_coverage",
            format!("{:.12}", metrics.il_candidate_coverage),
        ),
        (
            "geometry_il_top1",
            format!("{:.12}", metrics.geometry.il_top1),
        ),
        (
            "geometry_il_top10",
            format!("{:.12}", metrics.geometry.il_top10),
        ),
        (
            "geometry_il_mrr",
            format!("{:.12}", metrics.geometry.il_mrr),
        ),
        (
            "target_beats_best_il_negative_fraction",
            format!("{:.12}", metrics.target_beats_best_il_negative_fraction),
        ),
        (
            "mean_target_minus_best_il_negative_margin",
            format!("{:.12}", metrics.mean_target_minus_best_il_negative_margin),
        ),
        (
            "catalog_build_seconds",
            format!("{:.6}", metrics.catalog_build_seconds),
        ),
        (
            "index_build_seconds",
            format!("{:.6}", metrics.index_build_seconds),
        ),
        (
            "geometry_precompute_seconds",
            format!("{:.6}", metrics.geometry_precompute_seconds),
        ),
        ("scoring_seconds", format!("{:.6}", metrics.scoring_seconds)),
        (
            "learned_candidate_embedding_seconds",
            format!("{:.6}", metrics.learned_candidate_embedding_seconds),
        ),
        ("elapsed_seconds", format!("{:.6}", metrics.elapsed_seconds)),
        ("audit_decision", decision.to_string()),
        (
            "train_holdout_consumed",
            yes_no(train_holdout_consumed).to_string(),
        ),
        ("train_holdout_consumed_for_selection", "NO".to_string()),
        (
            "dev_labels_used_for_analysis",
            if train_holdout_consumed {
                "NO".to_string()
            } else {
                "YES_DEV_REPRODUCTION".to_string()
            },
        ),
        ("dev_partition_used_for_selection", "NO".to_string()),
        ("historical_validation_consumed", "NO".to_string()),
        ("historical_test_consumed", "NO".to_string()),
    ];
    for (key, value) in rows {
        text.push_str(key);
        text.push('\t');
        text.push_str(&value);
        text.push('\n');
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_candidate_catalog(path: &Path, identities: &[CandidateIdentity]) -> Result<()> {
    let mut text = String::from(
        "candidate_index\trecord_index\texact_key\til_key\tcharge\tobserved_neutral_mass\ttheoretical_neutral_mass\n",
    );
    for (candidate_index, identity) in identities.iter().enumerate() {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\n",
            candidate_index,
            identity.record_index,
            identity.exact_key,
            identity.il_key,
            identity.charge,
            identity.observed_neutral_mass,
            identity.candidate_neutral_mass,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_query_diagnostics(path: &Path, rows: &[QueryDiagnostic]) -> Result<()> {
    let mut text = String::from(
        "query_slot\tselected_identity_index\tfull_identity_index\texact_key\tcharge\tobserved_neutral_mass\tcandidate_count\texact_covered\til_covered\texact_rank\til_rank\ttarget_score\tbest_il_negative_score\tmargin\ttop_candidate\ttop_score\tquery_scoring_seconds\n",
    );
    for row in rows {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{}\t{:.8}\t{:.8}\n",
            row.query_slot,
            row.selected_identity_index,
            row.full_identity_index,
            row.exact_key,
            row.charge,
            row.observed_neutral_mass,
            row.candidate_count,
            yes_no(row.exact_covered),
            yes_no(row.il_covered),
            optional_usize(row.exact_rank),
            optional_usize(row.il_rank),
            optional_f64(row.target_score),
            row.best_il_negative_score,
            optional_f64(row.margin),
            row.top_candidate,
            row.top_score,
            row.query_scoring_seconds,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_rankings(path: &Path, rows: &[RankedCandidateRow]) -> Result<()> {
    let mut text = String::from(
        "query_slot\trank\tcandidate_index\tcandidate_exact_key\tcandidate_il_key\tcharge\ttheoretical_neutral_mass\tabsolute_neutral_mass_error\tgeometry_score\texact_target\til_target\n",
    );
    for row in rows {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\n",
            row.query_slot,
            row.rank,
            row.candidate_index,
            row.candidate_exact_key,
            row.candidate_il_key,
            row.charge,
            row.theoretical_neutral_mass,
            row.absolute_neutral_mass_error,
            row.geometry_score,
            yes_no(row.exact_target),
            yes_no(row.il_target),
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn safe_fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn optional_usize(value: Option<usize>) -> String {
    value
        .map(|item| item.to_string())
        .unwrap_or_else(|| "NA".to_string())
}

fn optional_f64(value: Option<f64>) -> String {
    value
        .map(|item| format!("{item:.8}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

fn pass_fail(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

fn v075_decision(mode: &str, all_confirmation_gates: bool) -> &'static str {
    match mode {
        "smoke" => "SMOKE_MECHANICAL_ONLY",
        "audit" if all_confirmation_gates => "IMPLEMENTATION_REPRODUCTION_PASS",
        "audit" => "IMPLEMENTATION_REPRODUCTION_FAIL",
        "holdout" if all_confirmation_gates => "HOLDOUT_CONFIRMATION_PASS",
        "holdout" => "HOLDOUT_CONFIRMATION_FAIL_DO_NOT_TUNE",
        _ => "INVALID_MODE",
    }
}

fn il_label(label: &str) -> String {
    label
        .chars()
        .map(|residue| if residue == 'I' { 'L' } else { residue })
        .collect()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hash64_str(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn self_test() -> Result<()> {
    let target = PeptidoformInput::unmodified("PEPTIDE");
    let practical_identifier = FoundationPracticalIdentifierV0751::new(vec![
        FoundationPracticalIdentifierCandidateV0751::new("PEPTIDE|z2", target.clone(), 2),
        FoundationPracticalIdentifierCandidateV0751::new(
            "PEPTIDK|z2",
            PeptidoformInput::unmodified("PEPTIDK"),
            2,
        ),
    ])?;
    let target_mass = foundation_peptidoform_neutral_mass(&target).map_err(anyhow::Error::msg)?;
    let spectrum = FoundationSpectrum::from_pairs([(100.0f32, 1.0f32)]);
    let hits = practical_identifier.identify_neutral_mass(2, target_mass, &spectrum)?;
    if hits.len() != 2 || hits.iter().any(|hit| hit.charge != 2) {
        anyhow::bail!("v0.75 self-test reusable practical identifier integration failed");
    }
    if v075_decision("holdout", false) != "HOLDOUT_CONFIRMATION_FAIL_DO_NOT_TUNE"
        || v075_decision("holdout", true) != "HOLDOUT_CONFIRMATION_PASS"
    {
        anyhow::bail!("v0.75 self-test protected HOLDOUT decision is not fail-closed");
    }
    if V075_MASS_POOL != 256
        || V075_EXPECTED_CANDIDATE_UNIVERSE != 63_332
        || V075_HOLDOUT_QUERIES != 2048
        || V075_HOLDOUT_CONFIRM_TOKEN != "CONFIRM_TRAIN_HOLDOUT_ONCE"
        || V075_HOLDOUT_MIN_IL_COVERAGE != 0.99
    {
        anyhow::bail!("v0.75 self-test frozen implementation constants drifted");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(index: usize, key: &str, charge: i32, mass: f64) -> CandidateIdentity {
        CandidateIdentity {
            record_index: index,
            exact_key: key.to_string(),
            il_key: il_label(key),
            charge,
            observed_neutral_mass: mass,
            candidate_neutral_mass: mass,
        }
    }

    #[test]
    fn reusable_practical_identifier_api_is_wired_to_frozen_pool() {
        assert_eq!(
            V075_MASS_POOL,
            FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751
        );
        self_test().expect("v0.75 reusable practical identifier self-test");
    }

    #[test]
    fn missing_target_counts_as_ranking_failure() {
        let mut acc = RankAccumulator::default();
        acc.observe(Some(1), Some(1));
        acc.observe(None, None);
        let metrics = acc.metrics();
        assert_eq!(metrics.exact_top1, 0.5);
        assert_eq!(metrics.il_top1, 0.5);
        assert_eq!(metrics.exact_mrr, 0.5);
        assert_eq!(metrics.il_mrr, 0.5);
    }

    #[test]
    fn holdout_confirmation_is_fail_closed_without_tuning() {
        assert_eq!(
            v075_decision("holdout", false),
            "HOLDOUT_CONFIRMATION_FAIL_DO_NOT_TUNE"
        );
        assert_eq!(v075_decision("holdout", true), "HOLDOUT_CONFIRMATION_PASS");
    }

    #[test]
    fn implementation_constants_are_frozen() {
        assert_eq!(V075_MASS_POOL, 256);
        assert_eq!(V075_MAX_SEQUENCE_LEN, 64);
        assert_eq!(V075_HOLDOUT_QUERIES, 2048);
        assert_eq!(V075_HOLDOUT_QUERY_SEED, 20_261_075);
        assert_eq!(V075_HOLDOUT_CONFIRM_TOKEN, "CONFIRM_TRAIN_HOLDOUT_ONCE");
        assert_eq!(V075_HOLDOUT_MIN_IL_COVERAGE, 0.99);
        assert_eq!(V075_EXPECTED_CORPUS_FINGERPRINT, "fnv1a64:a2a6f57d31064ba6");
        assert_eq!(
            V075_EXPECTED_BENCHMARK_FINGERPRINT,
            "fnv1a64:2133c039625f77df"
        );
        assert_eq!(
            V075_EXPECTED_PARENT_DEV_FINGERPRINT,
            "fnv1a64:2aa9a31055c720a6"
        );
        assert_eq!(
            V075_EXPECTED_CANDIDATE_FINGERPRINT,
            "fnv1a64:a5699533553116e8"
        );
    }
}
