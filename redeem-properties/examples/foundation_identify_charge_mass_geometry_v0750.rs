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
//! - use DEV only for the bounded implementation-reproduction audit; never consume HOLDOUT,
//!   historical VALIDATION/APD, or TEST.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    foundation_fragment_cleavage_geometry, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    FoundationBenchmarkManifest, FoundationPartition, FoundationSpectrum, FoundationTrainingRecord,
    FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230, FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230,
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
const V075_ARCHITECTURE: &str =
    "observed_charge_neutral_mass256_plus_deterministic_open_ptm_fragment_geometry";
const V075_SCORE: &str = "open_ptm_core_b1_b2_y1_y2_uniform_sqrt_intensity_cosine_20ppm_abs0p02Da";
const V075_CANDIDATE_POLICY: &str =
    "observed_charge_compatible_nearest_neutral_mass256_no_target_forcing";

// Frozen cohort-construction values used by v0.70-v0.74.2. These are provenance constants only;
// v0.75 does not load or execute the v0.70 model.
const V070_SELECTION_SEED: u64 = 20_261_070;
const V070_DEV_IDENTITIES: usize = 2048;
const V073_QUERY_SEED: u64 = 20_261_073;
const V075_SMOKE_QUERIES: usize = 32;
const V075_AUDIT_QUERIES: usize = 512;
const V075_MAX_SEQUENCE_LEN: usize = 64;
const V075_MASS_POOL: usize = 256;
const V075_TARGET_FORCING: bool = false;
const V075_EXPECTED_PARENT_DEV_FINGERPRINT: &str = "fnv1a64:2aa9a31055c720a6";
const V075_EXPECTED_CANDIDATE_UNIVERSE: usize = 63_332;
const V075_EXPECTED_CANDIDATE_FINGERPRINT: &str = "fnv1a64:a5699533553116e8";

// Reuse the already-declared v0.74 geometry promotion floors. These are reproduction guards,
// not a new architecture-selection sweep.
const V075_MIN_IL_COVERAGE: f64 = 1.0;
const V075_MIN_GEOMETRY_IL_TOP1: f64 = 0.85;
const V075_MIN_GEOMETRY_IL_TOP10: f64 = 0.97;

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
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_dev_identity_fingerprint: String,
    candidate_universe_fingerprint: String,
    expected_candidate_universe_fingerprint: String,
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
    if args.len() != 4 {
        anyhow::bail!(
            "usage: foundation_identify_charge_mass_geometry_v0750 RUN_V0260.yaml OUTPUT_DIR mode=smoke|audit"
        );
    }

    let training_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let mode = args[3].as_str();
    if !matches!(mode, "smoke" | "audit") {
        anyhow::bail!("v0.75 mode must be smoke or audit");
    }
    if output_root.exists() {
        anyhow::bail!("v0.75 output directory must be fresh: {output_root:?}");
    }

    let audit_queries = if mode == "smoke" {
        V075_SMOKE_QUERIES
    } else {
        V075_AUDIT_QUERIES
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

    let catalog_started = Instant::now();
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
    let selected_fingerprint = format!(
        "fnv1a64:{:016x}",
        identity_fingerprint(&selected_identities)
    );
    if selected_fingerprint != V075_EXPECTED_PARENT_DEV_FINGERPRINT {
        anyhow::bail!(
            "v0.75 frozen query-cohort parent fingerprint mismatch: current={} expected={}",
            selected_fingerprint,
            V075_EXPECTED_PARENT_DEV_FINGERPRINT
        );
    }

    let full_identities =
        build_all_dev_identities(&corpus.records, &dev_groups, identity_seed.rotate_left(17))?;
    let full_fingerprint = format!("fnv1a64:{:016x}", identity_fingerprint(&full_identities));
    if full_identities.len() != V075_EXPECTED_CANDIDATE_UNIVERSE {
        anyhow::bail!(
            "v0.75 candidate-universe size mismatch: current={} expected={}",
            full_identities.len(),
            V075_EXPECTED_CANDIDATE_UNIVERSE
        );
    }
    if full_fingerprint != V075_EXPECTED_CANDIDATE_FINGERPRINT {
        anyhow::bail!(
            "v0.75 candidate-universe fingerprint mismatch: current={} expected={}",
            full_fingerprint,
            V075_EXPECTED_CANDIDATE_FINGERPRINT
        );
    }

    let full_index = full_identity_index(&full_identities)?;
    let selected_query_indices = deterministic_eval_queries(
        &selected_identities,
        audit_queries,
        V073_QUERY_SEED ^ 0x7300_d3f0_0000_0001,
    );
    let mut full_query_indices = Vec::with_capacity(selected_query_indices.len());
    for &selected_index in &selected_query_indices {
        let selected = &selected_identities[selected_index];
        let full_index_value = *full_index.get(&selected.exact_key).with_context(|| {
            format!(
                "v0.75 frozen query {} absent from direct candidate catalog",
                selected.exact_key
            )
        })?;
        if full_identities[full_index_value].record_index != selected.record_index {
            anyhow::bail!(
                "v0.75 frozen query record drift for {}: selected={} full={}",
                selected.exact_key,
                selected.record_index,
                full_identities[full_index_value].record_index
            );
        }
        full_query_indices.push(full_index_value);
    }
    let catalog_build_seconds = catalog_started.elapsed().as_secs_f64();

    fs::create_dir_all(&output_root)?;
    write_candidate_catalog(&output_root.join("candidate_catalog.tsv"), &full_identities)?;

    let index_started = Instant::now();
    let mass_sorted_by_charge = mass_sorted_candidates_by_charge(&full_identities);
    let index_build_seconds = index_started.elapsed().as_secs_f64();

    let geometry_started = Instant::now();
    let candidate_geometry = precompute_candidate_geometry(&corpus.records, &full_identities)?;
    let geometry_precompute_seconds = geometry_started.elapsed().as_secs_f64();

    println!("v0750_version\tv0.75-charge-mass-geometry-identifier");
    println!("objective\t{V075_OBJECTIVE}");
    println!("architecture\t{V075_ARCHITECTURE}");
    println!("score\t{V075_SCORE}");
    println!("mode\t{mode}");
    println!("execution_device\tCPU");
    println!("training_scope\tNONE");
    println!("selection_scope\tDEV_IMPLEMENTATION_REPRODUCTION_ONLY");
    println!("candidate_universe_policy\tall_source_closed_dev_eligible_unique_peptidoform_charge_identities");
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
    println!("full_dev_candidate_universe\t{}", full_identities.len());
    println!("full_dev_identity_fingerprint\t{full_fingerprint}");
    println!("audit_queries\t{audit_queries}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let scoring_started = Instant::now();
    let (mut metrics, diagnostics, rankings) = identify_queries(
        &corpus.records,
        &full_identities,
        &full_query_indices,
        &selected_query_indices,
        &mass_sorted_by_charge,
        &candidate_geometry,
    )?;
    metrics.catalog_build_seconds = catalog_build_seconds;
    metrics.index_build_seconds = index_build_seconds;
    metrics.geometry_precompute_seconds = geometry_precompute_seconds;
    metrics.scoring_seconds = scoring_started.elapsed().as_secs_f64();
    metrics.learned_candidate_embedding_seconds = 0.0;
    metrics.elapsed_seconds = total_started.elapsed().as_secs_f64();

    print_metrics(&metrics);

    let gate_fingerprint = full_fingerprint == V075_EXPECTED_CANDIDATE_FINGERPRINT;
    let gate_target_forcing = !V075_TARGET_FORCING;
    let gate_coverage = (metrics.il_candidate_coverage - V075_MIN_IL_COVERAGE).abs() <= 1.0e-12;
    let gate_top1 = metrics.geometry.il_top1 >= V075_MIN_GEOMETRY_IL_TOP1;
    let gate_top10 = metrics.geometry.il_top10 >= V075_MIN_GEOMETRY_IL_TOP10;
    let gate_no_learned_dependency = metrics.learned_candidate_embedding_seconds == 0.0;

    println!(
        "v0750_gate_candidate_universe_fingerprint_exact\t{}",
        pass_fail(gate_fingerprint)
    );
    println!(
        "v0750_gate_target_forcing_no\t{}",
        pass_fail(gate_target_forcing)
    );
    println!(
        "v0750_gate_mass256_il_coverage_eq_1\t{}",
        pass_fail(gate_coverage)
    );
    println!(
        "v0750_gate_geometry_il_top1_ge_v074_floor\t{}",
        pass_fail(gate_top1)
    );
    println!(
        "v0750_gate_geometry_il_top10_ge_v074_floor\t{}",
        pass_fail(gate_top10)
    );
    println!(
        "v0750_gate_no_learned_candidate_embedding\t{}",
        pass_fail(gate_no_learned_dependency)
    );

    let decision = if mode == "smoke" {
        "SMOKE_MECHANICAL_ONLY"
    } else if gate_fingerprint
        && gate_target_forcing
        && gate_coverage
        && gate_top1
        && gate_top10
        && gate_no_learned_dependency
    {
        "IMPLEMENTATION_REPRODUCTION_PASS"
    } else {
        "IMPLEMENTATION_REPRODUCTION_FAIL"
    };
    println!("v0750_audit_decision\t{decision}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    write_metrics(&output_root.join("identifier_metrics.tsv"), &metrics)?;
    write_summary(
        &output_root.join("summary.tsv"),
        mode,
        &current_corpus_fingerprint,
        &current_benchmark_fingerprint,
        &selected_fingerprint,
        &full_fingerprint,
        decision,
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
        corpus_fingerprint: current_corpus_fingerprint,
        benchmark_manifest_fingerprint: current_benchmark_fingerprint,
        parent_dev_identity_fingerprint: selected_fingerprint,
        candidate_universe_fingerprint: full_fingerprint,
        expected_candidate_universe_fingerprint: V075_EXPECTED_CANDIDATE_FINGERPRINT.to_string(),
        audit_queries,
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
        train_holdout_consumed: false,
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

fn build_all_dev_identities(
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

fn mass_sorted_candidates_by_charge(
    identities: &[CandidateIdentity],
) -> BTreeMap<i32, Vec<(f64, usize)>> {
    let mut out = BTreeMap::<i32, Vec<(f64, usize)>>::new();
    for (index, identity) in identities.iter().enumerate() {
        out.entry(identity.charge)
            .or_default()
            .push((identity.candidate_neutral_mass, index));
    }
    for rows in out.values_mut() {
        rows.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });
    }
    out
}

fn nearest_mass_pool(mass_sorted: &[(f64, usize)], observed_mass: f64, count: usize) -> Vec<usize> {
    if mass_sorted.is_empty() || count == 0 {
        return Vec::new();
    }
    let split = mass_sorted.partition_point(|row| row.0 < observed_mass);
    let mut left = split;
    let mut right = split;
    let mut out = Vec::with_capacity(count.min(mass_sorted.len()));
    while out.len() < count.min(mass_sorted.len()) {
        let left_item = if left > 0 {
            Some(mass_sorted[left - 1])
        } else {
            None
        };
        let right_item = mass_sorted.get(right).copied();
        let take_left = match (left_item, right_item) {
            (Some(l), Some(r)) => {
                let ld = (l.0 - observed_mass).abs();
                let rd = (r.0 - observed_mass).abs();
                ld < rd || (ld == rd && l.1 < r.1)
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_left {
            left -= 1;
            out.push(mass_sorted[left].1);
        } else {
            out.push(mass_sorted[right].1);
            right += 1;
        }
    }
    out
}

fn precompute_candidate_geometry(
    records: &[FoundationTrainingRecord],
    identities: &[CandidateIdentity],
) -> Result<Vec<Vec<f64>>> {
    let mut all = Vec::with_capacity(identities.len());
    for identity in identities {
        let record = &records[identity.record_index];
        let geometry = foundation_fragment_cleavage_geometry(&record.peptidoform)
            .map_err(anyhow::Error::msg)?;
        let mut mz = Vec::with_capacity(geometry.len() * 4);
        for cleavage in geometry {
            mz.extend_from_slice(&cleavage.core_mz);
        }
        all.push(mz);
    }
    Ok(all)
}

fn normalized_retained_peaks(spectrum: &FoundationSpectrum) -> Vec<(f64, f64)> {
    let mut peaks = spectrum
        .peaks
        .iter()
        .copied()
        .filter(|peak| {
            peak.mz.is_finite()
                && peak.mz > 0.0
                && peak.intensity.is_finite()
                && peak.intensity > 0.0
        })
        .collect::<Vec<_>>();
    peaks.sort_by(|left, right| {
        right
            .intensity
            .total_cmp(&left.intensity)
            .then_with(|| left.mz.total_cmp(&right.mz))
    });
    peaks.truncate(FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230);
    let max_intensity = peaks
        .iter()
        .map(|peak| f64::from(peak.intensity))
        .fold(0.0f64, f64::max)
        .max(f64::EPSILON);
    let mut normalized = peaks
        .into_iter()
        .map(|peak| {
            (
                f64::from(peak.mz),
                (f64::from(peak.intensity) / max_intensity).clamp(0.0, 1.0),
            )
        })
        .collect::<Vec<_>>();
    normalized.sort_by(|left, right| left.0.total_cmp(&right.0));
    normalized
}

fn best_peak_support_fast(theoretical_mz: f64, peaks: &[(f64, f64)]) -> f64 {
    if !(theoretical_mz > 0.0 && theoretical_mz.is_finite()) {
        return 0.0;
    }
    let sigma = (theoretical_mz * FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230 * 1e-6)
        .max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230 / 3.0);
    let cutoff = (3.0 * sigma).max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230);
    let low = theoretical_mz - cutoff;
    let high = theoretical_mz + cutoff;
    let start = peaks.partition_point(|row| row.0 < low);
    let mut best = 0.0f64;
    for &(observed_mz, normalized_intensity) in &peaks[start..] {
        if observed_mz > high {
            break;
        }
        let error = (observed_mz - theoretical_mz).abs();
        let mass_weight = (-0.5 * (error / sigma).powi(2)).exp();
        best = best.max(normalized_intensity * mass_weight);
    }
    best
}

fn geometry_uniform_score(theoretical_mz: &[f64], peaks: &[(f64, f64)]) -> f64 {
    if theoretical_mz.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut obs_norm = 0.0f64;
    for &mz in theoretical_mz {
        let observed = best_peak_support_fast(mz, peaks).max(0.0);
        dot += observed.sqrt();
        obs_norm += observed;
    }
    if obs_norm > 0.0 {
        (dot / ((theoretical_mz.len() as f64).sqrt() * obs_norm.sqrt())).clamp(0.0, 1.0)
    } else {
        0.0
    }
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
    mass_sorted_by_charge: &BTreeMap<i32, Vec<(f64, usize)>>,
    candidate_geometry: &[Vec<f64>],
) -> Result<(V075Metrics, Vec<QueryDiagnostic>, Vec<RankedCandidateRow>)> {
    if full_query_indices.len() != selected_query_indices.len()
        || candidate_geometry.len() != identities.len()
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
        let peaks = normalized_retained_peaks(&spectrum);

        let charge_mass_sorted = mass_sorted_by_charge.get(&query.charge).ok_or_else(|| {
            anyhow::anyhow!(
                "v0.75 no candidate universe for observed precursor charge {}",
                query.charge
            )
        })?;
        let mass256 = nearest_mass_pool(
            charge_mass_sorted,
            query.observed_neutral_mass,
            V075_MASS_POOL,
        );
        if mass256.is_empty() {
            anyhow::bail!("v0.75 same-charge mass candidate pool is empty");
        }
        if mass256
            .iter()
            .any(|&index| identities[index].charge != query.charge)
        {
            anyhow::bail!("v0.75 charge-incompatible candidate escaped per-charge mass index");
        }

        let exact_covered = pool_exact_covered(&mass256, identities, query);
        let il_covered = pool_il_covered(&mass256, identities, query);
        exact_covered_count += usize::from(exact_covered);
        il_covered_count += usize::from(il_covered);
        candidate_counts.push(mass256.len());

        let mut scored = Vec::<(usize, f64)>::with_capacity(mass256.len());
        for &candidate_index in &mass256 {
            let score = geometry_uniform_score(&candidate_geometry[candidate_index], &peaks);
            if !score.is_finite() {
                anyhow::bail!("v0.75 geometry score is not finite");
            }
            scored.push((candidate_index, score));
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
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

    let denom = full_query_indices.len() as f64;
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
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    parent_dev_fingerprint: &str,
    candidate_fingerprint: &str,
    decision: &str,
    metrics: &V075Metrics,
) -> Result<()> {
    let mut text = String::from("key\tvalue\n");
    let rows = [
        ("version", "v0.75".to_string()),
        ("objective", V075_OBJECTIVE.to_string()),
        ("architecture", V075_ARCHITECTURE.to_string()),
        ("score", V075_SCORE.to_string()),
        ("mode", mode.to_string()),
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
        ("candidate_universe", metrics.candidate_universe.to_string()),
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
        ("train_holdout_consumed", "NO".to_string()),
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
    let mass_sorted = vec![(99.0, 0), (100.0, 1), (100.3, 2), (101.0, 3), (103.0, 4)];
    let pool = nearest_mass_pool(&mass_sorted, 100.2, 3);
    if pool != vec![2, 1, 3] {
        anyhow::bail!("v0.75 self-test nearest-mass ordering failed: {pool:?}");
    }
    let peaks = vec![(100.0, 1.0), (200.0, 0.64), (300.0, 0.36)];
    let aligned = geometry_uniform_score(&[100.0, 200.0, 300.0], &peaks);
    let shifted = geometry_uniform_score(&[110.0, 210.0, 310.0], &peaks);
    if !(aligned > 0.9 && shifted == 0.0 && aligned > shifted) {
        anyhow::bail!("v0.75 self-test geometry score failed: aligned={aligned} shifted={shifted}");
    }
    if V075_MASS_POOL != 256 || V075_EXPECTED_CANDIDATE_UNIVERSE != 63_332 {
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
    fn nearest_mass_pool_is_distance_ordered_without_target_forcing() {
        let mass_sorted = vec![(99.0, 0), (100.0, 1), (100.3, 2), (101.0, 3), (103.0, 4)];
        let pool = nearest_mass_pool(&mass_sorted, 100.2, 3);
        assert_eq!(pool, vec![2, 1, 3]);
        assert!(!pool.contains(&0));
    }

    #[test]
    fn per_charge_index_excludes_other_precursor_charges() {
        let identities = vec![
            identity(0, "PEPTIDE|z2", 2, 100.0),
            identity(1, "PEPTIDE|z3", 3, 100.0),
            identity(2, "OTHER|z2", 2, 100.2),
        ];
        let by_charge = mass_sorted_candidates_by_charge(&identities);
        let charge2 = by_charge.get(&2).expect("charge 2 candidates");
        let pool = nearest_mass_pool(charge2, 100.0, 8);
        assert_eq!(pool, vec![0, 2]);
        assert!(pool.iter().all(|&index| identities[index].charge == 2));
        assert!(!pool.contains(&1));
    }

    #[test]
    fn geometry_score_rewards_mass_aligned_fragment_support() {
        let peaks = vec![(100.0, 1.0), (200.0, 0.64), (300.0, 0.36)];
        let aligned = geometry_uniform_score(&[100.0, 200.0, 300.0], &peaks);
        let shifted = geometry_uniform_score(&[110.0, 210.0, 310.0], &peaks);
        assert!(aligned > 0.9);
        assert_eq!(shifted, 0.0);
        assert!(aligned > shifted);
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
    fn implementation_constants_are_frozen() {
        assert_eq!(V075_MASS_POOL, 256);
        assert_eq!(V075_MAX_SEQUENCE_LEN, 64);
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
