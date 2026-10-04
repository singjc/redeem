//! v0.74 large-DEV candidate-universe retrieval + deterministic fragment-geometry audit.
//!
//! Scientific contract:
//! - reproduce the selected v0.70 2,048-identity DEV checkpoint exactly before interpretation;
//! - keep the exact v0.73/v0.73.1 query cohort for controlled comparison;
//! - expand candidates to every source-closed DEV-eligible unique peptidoform+charge identity;
//! - never insert or force the true target into any mass pool or retrieval shortlist;
//! - build a fixed nearest-neutral-mass 1,024-candidate pool per query;
//! - compare frozen v0.70 retrieval, geometry-only ranking of all 1,024, and one fixed
//!   two-stage policy (v0.70 top-64 -> deterministic geometry reranking);
//! - report nearest-64/256/1024 target coverage and 20-ppm/0.02-Da candidate density;
//! - use DEV only; never touch TRAIN-HOLDOUT, historical VALIDATION/APD, or TEST.
//!
//! No trainable parameters, learned fusion, target forcing, or hyperparameter sweep exists.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_fragment_cleavage_geometry, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    FoundationBenchmarkManifest, FoundationCollator, FoundationCollatorConfig,
    FoundationCorruptionConfig, FoundationDiffusionConfig, FoundationFragmentContextBatchV0350,
    FoundationPartition, FoundationScalarPhysicsBatchV0360, FoundationSpectrum,
    FoundationSpectrumBatch, FoundationSpectrumCollator, FoundationSpectrumEncoder,
    FoundationTrainingRecord, PeptideFoundationMultimodalV0350Config, PeptideFoundationV0520Config,
    PeptideFoundationV0520Model, RetentionTimeObjective,
    FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230, FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const V073_VERSION: u32 = 731;
const V073_OBJECTIVE: &str = "v0731_frozen_v070_mass64_forward_ms2_intensity_attribution_controls";
const V073_ARCHITECTURE: &str =
    "frozen_v070_mass64_plus_v052_predicted_vs_geometry_vs_shuffled_intensity_attribution";
const V073_SCORE: &str = "open_ptm_core_b1_b2_y1_y2_sqrt_intensity_cosine_20ppm_abs0p02Da";
const V070_OBJECTIVE: &str = "v0700_frozen_v0520_spectrum_peptide_alignment";
const V070_ARCHITECTURE: &str =
    "frozen_v0520_peptide_plus_observed_spectrum_transformer_contrastive_v0700";
const V070_NAMESPACE: &str = "student_v070";
const V070_TEMPERATURE: f64 = 0.07;
const V070_DEV_IDENTITIES: usize = 2048;
const MASS_POOL: usize = 64;
const V073_SEED: u64 = 20_261_073;
const V073_SMOKE_QUERIES: usize = 32;
const V073_AUDIT_QUERIES: usize = 512;
const V073_MIN_CANDIDATE_COVERAGE: f64 = 0.99;
const V0731_MIN_PREDICTED_TOP1_GAIN_OVER_GEOMETRY: f64 = 0.02;
const V0731_MIN_PREDICTED_MRR_GAIN_OVER_GEOMETRY: f64 = 0.01;
const V0731_MIN_PREDICTED_TOP1_GAIN_OVER_SHUFFLED: f64 = 0.02;
const V0731_MIN_TARGET_BEAT_GAIN_OVER_GEOMETRY: f64 = 0.02;
const V073_PROTON_MASS_DA: f64 = 1.007_276_466_77;

#[derive(Debug, Clone, Deserialize)]
struct V052ParentMetadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_v0350_checkpoint: String,
    v0520_config: PeptideFoundationV0520Config,
    rt_objective: RetentionTimeObjective,
    completed_epochs: usize,
    completed_updates: usize,
    dev_objective: f64,
    smoke_mode: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct V035ParentMetadataV073 {
    version: u32,
    objective: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    v0350_config: PeptideFoundationMultimodalV0350Config,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct V070Config {
    spectrum: FoundationDiffusionConfig,
    peptide_input_dim: usize,
    spectrum_hidden_dim: usize,
    alignment_dim: usize,
    precursor_features: usize,
    temperature: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct V070ParentMetadata {
    version: u32,
    objective: String,
    architecture: String,
    parent_v052_checkpoint: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    config: V070Config,
    seed: u64,
    dev_identity_count: usize,
    dev_identity_fingerprint: String,
    completed_epochs: usize,
    completed_updates: usize,
    dev_selection_score: f64,
    smoke_mode: bool,
}

#[derive(Clone)]
struct PeptideSpectrumAlignmentV0700 {
    config: V070Config,
    spectrum_encoder: FoundationSpectrumEncoder,
    precursor_projection: candle_nn::Linear,
    spectrum_hidden: candle_nn::Linear,
    spectrum_projection: candle_nn::Linear,
    peptide_hidden: candle_nn::Linear,
    peptide_projection: candle_nn::Linear,
}
impl PeptideSpectrumAlignmentV0700 {
    fn new(config: V070Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.spectrum.validate().map_err(anyhow::Error::msg)?;
        let ns = vb.pp(V070_NAMESPACE);
        Ok(Self {
            spectrum_encoder: FoundationSpectrumEncoder::new(
                &config.spectrum,
                ns.pp("spectrum_encoder"),
            )?,
            precursor_projection: candle_nn::linear(
                config.precursor_features,
                config.spectrum_hidden_dim,
                ns.pp("precursor_projection"),
            )?,
            spectrum_hidden: candle_nn::linear(
                config.spectrum_hidden_dim,
                config.spectrum_hidden_dim,
                ns.pp("spectrum_hidden"),
            )?,
            spectrum_projection: candle_nn::linear(
                config.spectrum_hidden_dim,
                config.alignment_dim,
                ns.pp("spectrum_projection"),
            )?,
            peptide_hidden: candle_nn::linear(
                config.peptide_input_dim,
                config.spectrum_hidden_dim,
                ns.pp("peptide_hidden"),
            )?,
            peptide_projection: candle_nn::linear(
                config.spectrum_hidden_dim,
                config.alignment_dim,
                ns.pp("peptide_projection"),
            )?,
            config,
        })
    }

    fn encode_spectrum_t(
        &self,
        spectrum: &FoundationSpectrumBatch,
        precursor_features: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let encoded = self.spectrum_encoder.forward_t(spectrum, train)?;
        let precursor = self.precursor_projection.forward(precursor_features)?;
        let fused = (encoded.spectrum_embedding + precursor)?;
        let hidden = self.spectrum_hidden.forward(&fused.contiguous()?)?.relu()?;
        let aligned = normalize_rows(&self.spectrum_projection.forward(&hidden.contiguous()?)?)?;
        Ok((encoded.peak_embeddings.detach(), aligned.detach()))
    }

    fn encode_peptide(&self, frozen_peptide_features: &Tensor) -> Result<Tensor> {
        let hidden = self
            .peptide_hidden
            .forward(&frozen_peptide_features.contiguous()?)?
            .relu()?;
        Ok(normalize_rows(&self.peptide_projection.forward(&hidden.contiguous()?)?)?.detach())
    }
}

#[derive(Debug, Clone)]
struct AlignmentGroup {
    key: String,
    peptidoform: String,
    sequence: String,
    charge: i32,
    record_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct RerankIdentity {
    record_index: usize,
    exact_key: String,
    il_key: String,
    charge: i32,
    observed_neutral_mass: f64,
    candidate_neutral_mass: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct RetrievalMetrics {
    identities: usize,
    exact_top1: f64,
    exact_top5: f64,
    exact_top10: f64,
    exact_mrr: f64,
    il_top1: f64,
    il_top5: f64,
    il_top10: f64,
    il_mrr: f64,
    mean_exact_rank: f64,
    median_exact_rank: f64,
    mass_true_coverage: f64,
    mass_exact_top1: f64,
    mass_exact_top10: f64,
    mass_exact_mrr: f64,
    mass_il_top1: f64,
    mass_il_top10: f64,
    mass_il_mrr: f64,
}

impl RetrievalMetrics {
    fn selection_score(self) -> f64 {
        0.5 * self.il_mrr + 0.5 * self.mass_il_mrr
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct V073RankMetrics {
    exact_top1: f64,
    exact_top10: f64,
    exact_mrr: f64,
    il_top1: f64,
    il_top10: f64,
    il_mrr: f64,
}

impl V073RankMetrics {
    fn selection(self) -> f64 {
        0.7 * self.il_top1 + 0.3 * self.il_mrr
    }
}

#[derive(Debug, Clone, Serialize)]
struct V073AuditMetrics {
    queries: usize,
    candidates_per_query: usize,
    candidate_predictions: usize,
    candidate_coverage: f64,
    baseline: V073RankMetrics,
    predicted_intensity: V073RankMetrics,
    geometry_only: V073RankMetrics,
    shuffled_intensity: V073RankMetrics,
    mean_predicted_target_core_cosine: f64,
    mean_predicted_best_il_negative_core_cosine: f64,
    mean_predicted_target_minus_best_il_negative_margin: f64,
    median_predicted_target_minus_best_il_negative_margin: f64,
    predicted_target_beats_best_il_negative_fraction: f64,
    mean_geometry_target_core_cosine: f64,
    mean_geometry_best_il_negative_core_cosine: f64,
    mean_geometry_target_minus_best_il_negative_margin: f64,
    geometry_target_beats_best_il_negative_fraction: f64,
    mean_shuffled_target_core_cosine: f64,
    mean_shuffled_best_il_negative_core_cosine: f64,
    mean_shuffled_target_minus_best_il_negative_margin: f64,
    shuffled_target_beats_best_il_negative_fraction: f64,
    mean_within_query_v070_predicted_pearson: f64,
    mean_within_query_predicted_geometry_pearson: f64,
    mean_within_query_predicted_shuffled_pearson: f64,
    baseline_il_top1_errors: usize,
    predicted_rescued_baseline_errors: usize,
    predicted_rescue_fraction_of_baseline_errors: f64,
    predicted_harmed_baseline_correct: usize,
    predicted_harm_fraction_of_baseline_correct: f64,
    geometry_rescued_baseline_errors: usize,
    geometry_rescue_fraction_of_baseline_errors: f64,
    geometry_harmed_baseline_correct: usize,
    geometry_harm_fraction_of_baseline_correct: f64,
    shuffled_rescued_baseline_errors: usize,
    shuffled_rescue_fraction_of_baseline_errors: f64,
    shuffled_harmed_baseline_correct: usize,
    shuffled_harm_fraction_of_baseline_correct: f64,
    mean_predicted_core_cosine_all_candidates: f64,
    mean_geometry_core_cosine_all_candidates: f64,
    mean_shuffled_core_cosine_all_candidates: f64,
    nonzero_predicted_core_cosine_fraction: f64,
    elapsed_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
struct V073Metadata {
    version: u32,
    objective: String,
    architecture: String,
    score: String,
    mode: String,
    parent_v070_checkpoint: String,
    parent_v052_checkpoint: String,
    parent_v0350_checkpoint: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    dev_identity_fingerprint: String,
    parent_v070_selection_score: f64,
    reproduction_delta: f64,
    audit_queries: usize,
    mass_pool: usize,
    seed: u64,
    metrics: V073AuditMetrics,
    train_holdout_consumed: bool,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

#[derive(Debug, Clone)]
struct V073QueryDiagnostic {
    query_slot: usize,
    identity_index: usize,
    exact_key: String,
    baseline_il_rank: usize,
    predicted_il_rank: usize,
    geometry_il_rank: usize,
    shuffled_il_rank: usize,
    predicted_target_score: f64,
    geometry_target_score: f64,
    shuffled_target_score: f64,
    predicted_best_il_negative_score: f64,
    geometry_best_il_negative_score: f64,
    shuffled_best_il_negative_score: f64,
    predicted_margin: f64,
    geometry_margin: f64,
    shuffled_margin: f64,
    v070_predicted_pearson: f64,
    predicted_geometry_pearson: f64,
    predicted_shuffled_pearson: f64,
    baseline_top_candidate: String,
    predicted_top_candidate: String,
    geometry_top_candidate: String,
    shuffled_top_candidate: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct V073AttributionScores {
    predicted_core_cosine: f64,
    geometry_core_cosine: f64,
    shuffled_core_cosine: f64,
}

const V074_VERSION: u32 = 740;
const V074_OBJECTIVE: &str =
    "v0740_large_dev_candidate_retrieval_plus_deterministic_fragment_geometry_reranking";
const V074_ARCHITECTURE: &str =
    "frozen_v070_full_dev_candidate_retrieval_plus_mass1024_geometry_reranking";
const V074_SCORE: &str = "open_ptm_core_b1_b2_y1_y2_uniform_sqrt_intensity_cosine_20ppm_abs0p02Da";
const V074_SEED: u64 = 20_261_074;
const V074_SMOKE_QUERIES: usize = 32;
const V074_AUDIT_QUERIES: usize = 512;
const V074_MASS_POOL: usize = 1024;
const V074_MASS64: usize = 64;
const V074_MASS256: usize = 256;
const V074_RETRIEVAL_SHORTLIST: usize = 64;
const V074_CANDIDATE_BATCH: usize = 64;
const V074_QUERY_BATCH: usize = 64;
const V074_PRECURSOR_PPM: f64 = 20.0;
const V074_PRECURSOR_ABS_DA: f64 = 0.02;
const V074_MIN_UNIVERSE_MULTIPLE: usize = 10;
const V074_MIN_MASS1024_IL_COVERAGE: f64 = 0.99;
const V074_MIN_GEOMETRY_IL_TOP1: f64 = 0.85;
const V074_MIN_GEOMETRY_IL_TOP10: f64 = 0.97;
const V074_MIN_SHORTLIST_IL_COVERAGE: f64 = 0.90;
const V074_MIN_TWO_STAGE_IL_TOP1: f64 = 0.85;
const V074_MIN_TWO_STAGE_GAIN_OVER_RETRIEVAL: f64 = 0.10;

#[derive(Debug, Clone, Serialize)]
struct V074Metrics {
    queries: usize,
    candidate_universe: usize,
    candidate_universe_multiple_vs_v070: f64,
    mass_pool: usize,
    retrieval_shortlist: usize,
    mass64_exact_coverage: f64,
    mass64_il_coverage: f64,
    mass256_exact_coverage: f64,
    mass256_il_coverage: f64,
    mass1024_exact_coverage: f64,
    mass1024_il_coverage: f64,
    shortlist_exact_coverage: f64,
    shortlist_il_coverage: f64,
    precursor_tolerance_exact_coverage: f64,
    precursor_tolerance_il_coverage: f64,
    precursor_tolerance_mean_candidates: f64,
    precursor_tolerance_median_candidates: f64,
    precursor_tolerance_max_candidates: usize,
    retrieval_mass1024: V073RankMetrics,
    geometry_mass1024: V073RankMetrics,
    two_stage_top64_geometry: V073RankMetrics,
    geometry_target_beats_best_il_negative_fraction: f64,
    geometry_mean_target_score: f64,
    geometry_mean_best_il_negative_score: f64,
    geometry_mean_target_minus_best_il_negative_margin: f64,
    mean_within_query_retrieval_geometry_pearson: f64,
    candidate_encoding_seconds: f64,
    scoring_seconds: f64,
    elapsed_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
struct V074Metadata {
    version: u32,
    objective: String,
    architecture: String,
    score: String,
    mode: String,
    parent_v070_checkpoint: String,
    parent_v052_checkpoint: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_dev_identity_fingerprint: String,
    full_dev_identity_fingerprint: String,
    parent_v070_selection_score: f64,
    reproduction_delta: f64,
    audit_queries: usize,
    mass_pool: usize,
    retrieval_shortlist: usize,
    target_forcing: bool,
    seed: u64,
    metrics: V074Metrics,
    train_holdout_consumed: bool,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

#[derive(Debug, Clone)]
struct V074QueryDiagnostic {
    query_slot: usize,
    selected_identity_index: usize,
    full_identity_index: usize,
    exact_key: String,
    mass64_exact_covered: bool,
    mass64_il_covered: bool,
    mass256_exact_covered: bool,
    mass256_il_covered: bool,
    mass1024_exact_covered: bool,
    mass1024_il_covered: bool,
    shortlist_exact_covered: bool,
    shortlist_il_covered: bool,
    tolerance_candidates: usize,
    tolerance_exact_covered: bool,
    tolerance_il_covered: bool,
    retrieval_il_rank: Option<usize>,
    geometry_il_rank: Option<usize>,
    two_stage_il_rank: Option<usize>,
    geometry_target_score: Option<f64>,
    geometry_best_il_negative_score: f64,
    geometry_margin: Option<f64>,
    retrieval_geometry_pearson: f64,
    retrieval_top_candidate: String,
    geometry_top_candidate: String,
    two_stage_top_candidate: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct V074RankAccumulator {
    queries: usize,
    exact_top1: usize,
    exact_top10: usize,
    exact_rr: f64,
    il_top1: usize,
    il_top10: usize,
    il_rr: f64,
}

impl V074RankAccumulator {
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

    fn metrics(self) -> V073RankMetrics {
        if self.queries == 0 {
            return V073RankMetrics::default();
        }
        let denom = self.queries as f64;
        V073RankMetrics {
            exact_top1: self.exact_top1 as f64 / denom,
            exact_top10: self.exact_top10 as f64 / denom,
            exact_mrr: self.exact_rr / denom,
            il_top1: self.il_top1 as f64 / denom,
            il_top10: self.il_top10 as f64 / denom,
            il_mrr: self.il_rr / denom,
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        anyhow::bail!(
            "usage: foundation_audit_large_candidate_geometry_v0740 RUN_V0260.yaml OUTPUT_DIR PARENT_V070_BEST mode=smoke|audit"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let parent_v070_checkpoint = PathBuf::from(&args[3]);
    let mode = args[4].as_str();
    if !matches!(mode, "smoke" | "audit") {
        anyhow::bail!("v0.74 mode must be smoke or audit");
    }
    if output_root.exists() {
        anyhow::bail!("v0.74 output directory must be fresh: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.74 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let audit_queries = if mode == "smoke" {
        V074_SMOKE_QUERIES
    } else {
        V074_AUDIT_QUERIES
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

    let parent_v070_metadata = read_v070_metadata(&parent_v070_checkpoint)?;
    validate_v070_parent(&parent_v070_metadata)?;
    if parent_v070_metadata.corpus_fingerprint != current_corpus_fingerprint
        || parent_v070_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint
    {
        anyhow::bail!("v0.74 v0.70 parent provenance differs from current corpus/benchmark");
    }

    let parent_v052_checkpoint = PathBuf::from(&parent_v070_metadata.parent_v052_checkpoint);
    let parent_v052_metadata = read_v052_metadata(&parent_v052_checkpoint)?;
    validate_v052_parent(&parent_v052_metadata)?;
    if parent_v052_metadata.corpus_fingerprint != current_corpus_fingerprint
        || parent_v052_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint
    {
        anyhow::bail!("v0.74 v0.52 parent provenance differs from current corpus/benchmark");
    }

    let max_sequence_len = parent_v052_metadata
        .v0520_config
        .base_v0510
        .base_v0500
        .max_sequence_len;
    let dev_groups = build_alignment_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        max_sequence_len,
    )?;
    if dev_groups.len() < V070_DEV_IDENTITIES * V074_MIN_UNIVERSE_MULTIPLE {
        anyhow::bail!(
            "v0.74 requires >= {} DEV identities for a meaningful universe expansion; observed {}",
            V070_DEV_IDENTITIES * V074_MIN_UNIVERSE_MULTIPLE,
            dev_groups.len()
        );
    }

    let identity_seed = parent_v070_metadata.seed ^ 0x7000_d3f0_a11e_0001;
    let selected_identities = select_dev_identities(
        &corpus.records,
        &dev_groups,
        parent_v070_metadata.dev_identity_count,
        identity_seed,
    )?;
    if selected_identities.len() != V070_DEV_IDENTITIES {
        anyhow::bail!("v0.74 expected {V070_DEV_IDENTITIES} parent identities");
    }
    let selected_fingerprint = format!(
        "fnv1a64:{:016x}",
        identity_fingerprint(&selected_identities)
    );
    if selected_fingerprint != parent_v070_metadata.dev_identity_fingerprint {
        anyhow::bail!(
            "v0.74 selected DEV identity fingerprint mismatch: current={} parent={}",
            selected_fingerprint,
            parent_v070_metadata.dev_identity_fingerprint
        );
    }

    let full_identities =
        build_all_dev_identities_v074(&corpus.records, &dev_groups, identity_seed.rotate_left(17))?;
    let full_fingerprint = format!("fnv1a64:{:016x}", identity_fingerprint(&full_identities));
    let full_index = full_identity_index_v074(&full_identities)?;

    let selected_query_indices = deterministic_eval_queries(
        &selected_identities,
        audit_queries,
        V073_SEED ^ 0x7300_d3f0_0000_0001,
    );
    let mut full_query_indices = Vec::with_capacity(selected_query_indices.len());
    for &selected_index in &selected_query_indices {
        let selected = &selected_identities[selected_index];
        let full_index_value = *full_index.get(&selected.exact_key).with_context(|| {
            format!(
                "v0.74 selected query {} absent from full DEV universe",
                selected.exact_key
            )
        })?;
        if full_identities[full_index_value].record_index != selected.record_index {
            anyhow::bail!(
                "v0.74 query record drift for {}: selected={} full={}",
                selected.exact_key,
                selected.record_index,
                full_identities[full_index_value].record_index
            );
        }
        full_query_indices.push(full_index_value);
    }

    let holdout_reserved = benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Test)
        .count();

    let collator = FoundationCollator::new(
        parent_v052_metadata
            .v0520_config
            .base_v0510
            .base_v0500
            .featurizer_config(),
        FoundationCollatorConfig {
            retention_time_objective: parent_v052_metadata.rt_objective,
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.0,
                chemistry_mask_probability: 0.0,
            },
        },
    )?;
    let spectrum_collator =
        FoundationSpectrumCollator::new(parent_v070_metadata.config.spectrum.spectrum.clone())?;

    let mut v052_varmap = VarMap::new();
    let v052 = PeptideFoundationV0520Model::new(
        parent_v052_metadata.v0520_config.clone(),
        VarBuilder::from_varmap(&v052_varmap, DType::F32, &device),
    )?;
    v052_varmap
        .load(parent_v052_checkpoint.join("model.safetensors"))
        .with_context(|| format!("load frozen v0.52 parent from {parent_v052_checkpoint:?}"))?;

    let mut v070_varmap = VarMap::new();
    let v070 = PeptideSpectrumAlignmentV0700::new(
        parent_v070_metadata.config.clone(),
        VarBuilder::from_varmap(&v070_varmap, DType::F32, &device),
    )?;
    v070_varmap
        .load(parent_v070_checkpoint.join("model.safetensors"))
        .with_context(|| format!("load frozen selected v0.70 from {parent_v070_checkpoint:?}"))?;

    let v052_checksum_initial = varmap_checksum(&v052_varmap)?;
    let v070_checksum_initial = varmap_checksum(&v070_varmap)?;

    println!("v0740_version\tv0.74-large-dev-candidate-geometry-audit");
    println!("objective\t{V074_OBJECTIVE}");
    println!("architecture\t{V074_ARCHITECTURE}");
    println!("score\t{V074_SCORE}");
    println!("device\t{device:?}");
    println!("mode\t{mode}");
    println!(
        "parent_v070_checkpoint\t{}",
        parent_v070_checkpoint.display()
    );
    println!(
        "parent_v052_checkpoint\t{}",
        parent_v052_checkpoint.display()
    );
    println!("parent_update_policy\tfrozen_v070_plus_frozen_v052_no_trainable_variables");
    println!("candidate_universe_policy\tall_source_closed_dev_eligible_unique_peptidoform_charge_identities");
    println!("candidate_policy\tnearest_neutral_mass1024_no_target_forcing");
    println!("two_stage_policy\tv070_top64_within_mass1024_then_geometry");
    println!("target_forcing\tNO");
    println!("forward_ms2_intensity\tNO");
    println!("mass_pool\t{V074_MASS_POOL}");
    println!("candidate_encoding_batch\t{V074_CANDIDATE_BATCH}");
    println!("retrieval_shortlist\t{V074_RETRIEVAL_SHORTLIST}");
    println!(
        "parent_v070_dev_identity_count\t{}",
        selected_identities.len()
    );
    println!("full_dev_candidate_universe\t{}", full_identities.len());
    println!("parent_dev_identity_fingerprint\t{selected_fingerprint}");
    println!("full_dev_identity_fingerprint\t{full_fingerprint}");
    println!("audit_queries\t{audit_queries}");
    println!("holdout_records_reserved_not_read\t{holdout_reserved}");
    println!("rt_conditioning\tNO");
    println!("ccs_conditioning\tNO");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let (parent_baseline, _) = evaluate_parent_baseline(
        &v052,
        &v070,
        &collator,
        &spectrum_collator,
        &corpus.records,
        &selected_identities,
        64,
        &device,
    )?;
    print_parent_retrieval("v0740_parent_v070_reproduction", parent_baseline);
    let reproduction_delta =
        (parent_baseline.selection_score() - parent_v070_metadata.dev_selection_score).abs();
    if reproduction_delta > 1.0e-5 {
        anyhow::bail!(
            "v0.74 failed v0.70 parent reproduction: current={:.8} parent={:.8} delta={:.8}",
            parent_baseline.selection_score(),
            parent_v070_metadata.dev_selection_score,
            reproduction_delta
        );
    }
    println!("v0740_parent_reproduction_gate\tPASS\tdelta={reproduction_delta:.8}");

    fs::create_dir_all(&output_root)?;
    let encode_started = Instant::now();
    let candidate_alignment = encode_full_candidate_alignment_v074(
        &v052,
        &v070,
        &collator,
        &corpus.records,
        &full_identities,
        V074_CANDIDATE_BATCH,
        &device,
    )?;
    let query_alignment = encode_query_alignment_v074(
        &v070,
        &spectrum_collator,
        &corpus.records,
        &full_identities,
        &full_query_indices,
        V074_QUERY_BATCH,
        &device,
    )?;
    let similarities = query_alignment
        .matmul(&candidate_alignment.transpose(0, 1)?.contiguous()?)?
        .to_vec2::<f32>()?;
    let candidate_encoding_seconds = encode_started.elapsed().as_secs_f64();

    let mass_sorted = mass_sorted_candidates_v074(&full_identities);
    let candidate_geometry = precompute_candidate_geometry_v074(&corpus.records, &full_identities)?;

    let scoring_started = Instant::now();
    let (mut metrics, diagnostics) = audit_large_candidate_geometry_v074(
        &corpus.records,
        &full_identities,
        &full_query_indices,
        &selected_query_indices,
        &similarities,
        &mass_sorted,
        &candidate_geometry,
    )?;
    metrics.candidate_encoding_seconds = candidate_encoding_seconds;
    metrics.scoring_seconds = scoring_started.elapsed().as_secs_f64();
    metrics.elapsed_seconds = total_started.elapsed().as_secs_f64();

    let v052_checksum = varmap_checksum(&v052_varmap)?;
    let v070_checksum = varmap_checksum(&v070_varmap)?;
    assert_frozen_checksum_v074("v052_peptide_parent", v052_checksum_initial, v052_checksum)?;
    assert_frozen_checksum_v074(
        "v070_alignment_parent",
        v070_checksum_initial,
        v070_checksum,
    )?;

    print_v074_metrics(&metrics);

    let gate_universe = full_identities.len() >= V070_DEV_IDENTITIES * V074_MIN_UNIVERSE_MULTIPLE;
    let gate_mass_coverage = metrics.mass1024_il_coverage >= V074_MIN_MASS1024_IL_COVERAGE;
    let gate_geometry_top1 = metrics.geometry_mass1024.il_top1 >= V074_MIN_GEOMETRY_IL_TOP1;
    let gate_geometry_top10 = metrics.geometry_mass1024.il_top10 >= V074_MIN_GEOMETRY_IL_TOP10;
    let gate_shortlist_coverage = metrics.shortlist_il_coverage >= V074_MIN_SHORTLIST_IL_COVERAGE;
    let gate_two_stage_top1 =
        metrics.two_stage_top64_geometry.il_top1 >= V074_MIN_TWO_STAGE_IL_TOP1;
    let geometry_gain = metrics.geometry_mass1024.il_top1 - metrics.retrieval_mass1024.il_top1;
    let gate_gain = geometry_gain >= V074_MIN_TWO_STAGE_GAIN_OVER_RETRIEVAL;

    println!("v0740_geometry_top1_gain_over_retrieval\t{geometry_gain:.8}");
    println!(
        "v0740_gate_candidate_universe_ge_10x_parent\t{}",
        pass_fail(gate_universe)
    );
    println!(
        "v0740_gate_mass1024_il_coverage_ge_0_99\t{}",
        pass_fail(gate_mass_coverage)
    );
    println!(
        "v0740_gate_geometry_mass1024_il_top1_ge_0_85\t{}",
        pass_fail(gate_geometry_top1)
    );
    println!(
        "v0740_gate_geometry_mass1024_il_top10_ge_0_97\t{}",
        pass_fail(gate_geometry_top10)
    );
    println!(
        "v0740_gate_shortlist64_il_coverage_ge_0_90\t{}",
        pass_fail(gate_shortlist_coverage)
    );
    println!(
        "v0740_gate_two_stage_il_top1_ge_0_85\t{}",
        pass_fail(gate_two_stage_top1)
    );
    println!(
        "v0740_gate_geometry_top1_gain_over_retrieval_ge_0_10\t{}",
        pass_fail(gate_gain)
    );

    let decision = if mode == "smoke" {
        "SMOKE_MECHANICAL_ONLY"
    } else if metrics.mass1024_il_coverage < 0.95 || metrics.shortlist_il_coverage < 0.85 {
        "CANDIDATE_GENERATION_BOTTLENECK"
    } else if gate_universe
        && gate_mass_coverage
        && gate_geometry_top1
        && gate_geometry_top10
        && gate_shortlist_coverage
        && gate_two_stage_top1
        && gate_gain
    {
        "PROMOTE_RETRIEVAL_GEOMETRY_ARCHITECTURE"
    } else {
        "GEOMETRY_SCALABILITY_NOT_PROVEN"
    };
    println!("v0740_audit_decision\t{decision}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    write_v074_metrics(&output_root.join("large_candidate_metrics.tsv"), &metrics)?;
    write_v074_diagnostics(&output_root.join("query_diagnostics.tsv"), &diagnostics)?;
    let metadata = V074Metadata {
        version: V074_VERSION,
        objective: V074_OBJECTIVE.to_string(),
        architecture: V074_ARCHITECTURE.to_string(),
        score: V074_SCORE.to_string(),
        mode: mode.to_string(),
        parent_v070_checkpoint: parent_v070_checkpoint.display().to_string(),
        parent_v052_checkpoint: parent_v052_checkpoint.display().to_string(),
        corpus_fingerprint: current_corpus_fingerprint,
        benchmark_manifest_fingerprint: current_benchmark_fingerprint,
        parent_dev_identity_fingerprint: selected_fingerprint,
        full_dev_identity_fingerprint: full_fingerprint,
        parent_v070_selection_score: parent_v070_metadata.dev_selection_score,
        reproduction_delta,
        audit_queries,
        mass_pool: V074_MASS_POOL,
        retrieval_shortlist: V074_RETRIEVAL_SHORTLIST,
        target_forcing: false,
        seed: V074_SEED,
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

fn build_all_dev_identities_v074(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    record_seed: u64,
) -> Result<Vec<RerankIdentity>> {
    let mut identities = Vec::with_capacity(groups.len());
    for group in groups {
        identities.push(identity_from_group(records, group, record_seed)?);
    }
    validate_unique_identities(&identities)?;
    Ok(identities)
}

fn full_identity_index_v074(identities: &[RerankIdentity]) -> Result<BTreeMap<String, usize>> {
    let mut out = BTreeMap::new();
    for (index, identity) in identities.iter().enumerate() {
        if out.insert(identity.exact_key.clone(), index).is_some() {
            anyhow::bail!("v0.74 duplicate exact identity while indexing full DEV universe");
        }
    }
    Ok(out)
}

fn encode_full_candidate_alignment_v074(
    v052: &PeptideFoundationV0520Model,
    v070: &PeptideSpectrumAlignmentV0700,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut rows = Vec::<Vec<f32>>::with_capacity(identities.len());
    for chunk in identities.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|item| records[item.record_index].clone())
            .collect::<Vec<_>>();
        let (_, _, aligned) = encode_candidates(v052, v070, collator, &owned, device)?;
        rows.extend(aligned.to_vec2::<f32>()?);
    }
    let d = v070.config.alignment_dim;
    Ok(Tensor::from_vec(
        rows.into_iter().flatten().collect::<Vec<_>>(),
        (identities.len(), d),
        device,
    )?)
}

fn encode_query_alignment_v074(
    v070: &PeptideSpectrumAlignmentV0700,
    spectrum_collator: &FoundationSpectrumCollator,
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
    query_indices: &[usize],
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut rows = Vec::<Vec<f32>>::with_capacity(query_indices.len());
    for chunk in query_indices.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|&index| records[identities[index].record_index].clone())
            .collect::<Vec<_>>();
        let (_, _, aligned) = encode_query_spectrum(v070, spectrum_collator, &owned, device)?;
        rows.extend(aligned.to_vec2::<f32>()?);
    }
    let d = v070.config.alignment_dim;
    Ok(Tensor::from_vec(
        rows.into_iter().flatten().collect::<Vec<_>>(),
        (query_indices.len(), d),
        device,
    )?)
}

fn mass_sorted_candidates_v074(identities: &[RerankIdentity]) -> Vec<(f64, usize)> {
    let mut out = identities
        .iter()
        .enumerate()
        .map(|(index, identity)| (identity.candidate_neutral_mass, index))
        .collect::<Vec<_>>();
    out.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    out
}

fn nearest_mass_pool_v074(
    mass_sorted: &[(f64, usize)],
    observed_mass: f64,
    count: usize,
) -> Vec<usize> {
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

fn tolerance_candidates_v074(mass_sorted: &[(f64, usize)], observed_mass: f64) -> Vec<usize> {
    let tolerance = (observed_mass.abs() * V074_PRECURSOR_PPM * 1.0e-6).max(V074_PRECURSOR_ABS_DA);
    let low = observed_mass - tolerance;
    let high = observed_mass + tolerance;
    let start = mass_sorted.partition_point(|row| row.0 < low);
    let end = mass_sorted.partition_point(|row| row.0 <= high);
    mass_sorted[start..end].iter().map(|row| row.1).collect()
}

fn precompute_candidate_geometry_v074(
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
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

fn best_peak_support_fast_v074(theoretical_mz: f64, peaks: &[(f64, f64)]) -> f64 {
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

fn geometry_uniform_score_v074(theoretical_mz: &[f64], peaks: &[(f64, f64)]) -> f64 {
    if theoretical_mz.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut obs_norm = 0.0f64;
    for &mz in theoretical_mz {
        let observed = best_peak_support_fast_v074(mz, peaks).max(0.0);
        dot += observed.sqrt();
        obs_norm += observed;
    }
    if obs_norm > 0.0 {
        (dot / ((theoretical_mz.len() as f64).sqrt() * obs_norm.sqrt())).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn optional_exact_rank_v074(
    ranked: &[usize],
    identities: &[RerankIdentity],
    query: &RerankIdentity,
) -> Option<usize> {
    ranked
        .iter()
        .position(|&index| identities[index].exact_key == query.exact_key)
        .map(|rank| rank + 1)
}

fn optional_il_rank_v074(
    ranked: &[usize],
    identities: &[RerankIdentity],
    query: &RerankIdentity,
) -> Option<usize> {
    ranked
        .iter()
        .position(|&index| identities[index].il_key == query.il_key)
        .map(|rank| rank + 1)
}

fn pool_exact_covered_v074(
    pool: &[usize],
    identities: &[RerankIdentity],
    query: &RerankIdentity,
) -> bool {
    pool.iter()
        .any(|&index| identities[index].exact_key == query.exact_key)
}

fn pool_il_covered_v074(
    pool: &[usize],
    identities: &[RerankIdentity],
    query: &RerankIdentity,
) -> bool {
    pool.iter()
        .any(|&index| identities[index].il_key == query.il_key)
}

#[allow(clippy::too_many_arguments)]
fn audit_large_candidate_geometry_v074(
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
    full_query_indices: &[usize],
    selected_query_indices: &[usize],
    similarities: &[Vec<f32>],
    mass_sorted: &[(f64, usize)],
    candidate_geometry: &[Vec<f64>],
) -> Result<(V074Metrics, Vec<V074QueryDiagnostic>)> {
    if full_query_indices.len() != selected_query_indices.len()
        || similarities.len() != full_query_indices.len()
        || similarities.iter().any(|row| row.len() != identities.len())
        || candidate_geometry.len() != identities.len()
    {
        anyhow::bail!("v0.74 audit input shape mismatch");
    }

    let mut retrieval_acc = V074RankAccumulator::default();
    let mut geometry_acc = V074RankAccumulator::default();
    let mut two_stage_acc = V074RankAccumulator::default();
    let mut diagnostics = Vec::with_capacity(full_query_indices.len());

    let mut mass64_exact = 0usize;
    let mut mass64_il = 0usize;
    let mut mass256_exact = 0usize;
    let mut mass256_il = 0usize;
    let mut mass1024_exact = 0usize;
    let mut mass1024_il = 0usize;
    let mut shortlist_exact = 0usize;
    let mut shortlist_il = 0usize;
    let mut tolerance_exact = 0usize;
    let mut tolerance_il = 0usize;
    let mut tolerance_counts = Vec::<usize>::with_capacity(full_query_indices.len());
    let mut target_geometry_scores = Vec::<f64>::new();
    let mut negative_geometry_scores = Vec::<f64>::new();
    let mut geometry_margins = Vec::<f64>::new();
    let mut geometry_beats = 0usize;
    let mut geometry_margin_queries = 0usize;
    let mut correlations = Vec::<f64>::new();

    for (query_slot, (&query_index, &selected_identity_index)) in full_query_indices
        .iter()
        .zip(selected_query_indices)
        .enumerate()
    {
        let query = &identities[query_index];
        let query_record = &records[query.record_index];
        let spectrum = FoundationSpectrum::from_training_record(query_record)
            .ok_or_else(|| anyhow::anyhow!("v0.74 query lacks observed spectrum"))?;
        let peaks = normalized_retained_peaks_v073(&spectrum);

        let mass1024 =
            nearest_mass_pool_v074(mass_sorted, query.observed_neutral_mass, V074_MASS_POOL);
        let mass64 = mass1024
            .iter()
            .copied()
            .take(V074_MASS64)
            .collect::<Vec<_>>();
        let mass256 = mass1024
            .iter()
            .copied()
            .take(V074_MASS256)
            .collect::<Vec<_>>();
        let tolerance_pool = tolerance_candidates_v074(mass_sorted, query.observed_neutral_mass);

        let mass64_exact_covered = pool_exact_covered_v074(&mass64, identities, query);
        let mass64_il_covered = pool_il_covered_v074(&mass64, identities, query);
        let mass256_exact_covered = pool_exact_covered_v074(&mass256, identities, query);
        let mass256_il_covered = pool_il_covered_v074(&mass256, identities, query);
        let mass1024_exact_covered = pool_exact_covered_v074(&mass1024, identities, query);
        let mass1024_il_covered = pool_il_covered_v074(&mass1024, identities, query);
        let tolerance_exact_covered = pool_exact_covered_v074(&tolerance_pool, identities, query);
        let tolerance_il_covered = pool_il_covered_v074(&tolerance_pool, identities, query);

        mass64_exact += usize::from(mass64_exact_covered);
        mass64_il += usize::from(mass64_il_covered);
        mass256_exact += usize::from(mass256_exact_covered);
        mass256_il += usize::from(mass256_il_covered);
        mass1024_exact += usize::from(mass1024_exact_covered);
        mass1024_il += usize::from(mass1024_il_covered);
        tolerance_exact += usize::from(tolerance_exact_covered);
        tolerance_il += usize::from(tolerance_il_covered);
        tolerance_counts.push(tolerance_pool.len());

        let mut retrieval_ranked = mass1024.clone();
        retrieval_ranked.sort_by(|&a, &b| {
            similarities[query_slot][b]
                .partial_cmp(&similarities[query_slot][a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let mut geometry_scored = Vec::<(usize, f64)>::with_capacity(mass1024.len());
        for &candidate_index in &mass1024 {
            let score = geometry_uniform_score_v074(&candidate_geometry[candidate_index], &peaks);
            if !score.is_finite() {
                anyhow::bail!("v0.74 geometry score is not finite");
            }
            geometry_scored.push((candidate_index, score));
        }
        geometry_scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let geometry_ranked = geometry_scored.iter().map(|row| row.0).collect::<Vec<_>>();

        let shortlist = retrieval_ranked
            .iter()
            .copied()
            .take(V074_RETRIEVAL_SHORTLIST.min(retrieval_ranked.len()))
            .collect::<Vec<_>>();
        let shortlist_exact_covered = pool_exact_covered_v074(&shortlist, identities, query);
        let shortlist_il_covered = pool_il_covered_v074(&shortlist, identities, query);
        shortlist_exact += usize::from(shortlist_exact_covered);
        shortlist_il += usize::from(shortlist_il_covered);

        let geometry_map = geometry_scored.iter().copied().collect::<BTreeMap<_, _>>();
        let mut two_stage_ranked = shortlist.clone();
        two_stage_ranked.sort_by(|&a, &b| {
            geometry_map[&b]
                .partial_cmp(&geometry_map[&a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let retrieval_exact_rank = optional_exact_rank_v074(&retrieval_ranked, identities, query);
        let retrieval_il_rank = optional_il_rank_v074(&retrieval_ranked, identities, query);
        let geometry_exact_rank = optional_exact_rank_v074(&geometry_ranked, identities, query);
        let geometry_il_rank = optional_il_rank_v074(&geometry_ranked, identities, query);
        let two_stage_exact_rank = optional_exact_rank_v074(&two_stage_ranked, identities, query);
        let two_stage_il_rank = optional_il_rank_v074(&two_stage_ranked, identities, query);

        retrieval_acc.observe(retrieval_exact_rank, retrieval_il_rank);
        geometry_acc.observe(geometry_exact_rank, geometry_il_rank);
        two_stage_acc.observe(two_stage_exact_rank, two_stage_il_rank);

        let geometry_target_score = mass1024
            .iter()
            .filter(|&&index| identities[index].il_key == query.il_key)
            .map(|index| geometry_map[index])
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let best_negative_geometry = mass1024
            .iter()
            .filter(|&&index| identities[index].il_key != query.il_key)
            .map(|index| geometry_map[index])
            .fold(0.0f64, f64::max);
        let geometry_margin = geometry_target_score.map(|score| score - best_negative_geometry);
        if let Some(target_score) = geometry_target_score {
            target_geometry_scores.push(target_score);
            negative_geometry_scores.push(best_negative_geometry);
            let margin = target_score - best_negative_geometry;
            geometry_margins.push(margin);
            geometry_beats += usize::from(margin > 0.0);
            geometry_margin_queries += 1;
        }

        let retrieval_scores = mass1024
            .iter()
            .map(|&index| f64::from(similarities[query_slot][index]))
            .collect::<Vec<_>>();
        let geometry_scores = mass1024
            .iter()
            .map(|index| geometry_map[index])
            .collect::<Vec<_>>();
        let correlation = pearson_v073(&retrieval_scores, &geometry_scores);
        correlations.push(correlation);

        diagnostics.push(V074QueryDiagnostic {
            query_slot,
            selected_identity_index,
            full_identity_index: query_index,
            exact_key: query.exact_key.clone(),
            mass64_exact_covered,
            mass64_il_covered,
            mass256_exact_covered,
            mass256_il_covered,
            mass1024_exact_covered,
            mass1024_il_covered,
            shortlist_exact_covered,
            shortlist_il_covered,
            tolerance_candidates: tolerance_pool.len(),
            tolerance_exact_covered,
            tolerance_il_covered,
            retrieval_il_rank,
            geometry_il_rank,
            two_stage_il_rank,
            geometry_target_score,
            geometry_best_il_negative_score: best_negative_geometry,
            geometry_margin,
            retrieval_geometry_pearson: correlation,
            retrieval_top_candidate: identities[retrieval_ranked[0]].exact_key.clone(),
            geometry_top_candidate: identities[geometry_ranked[0]].exact_key.clone(),
            two_stage_top_candidate: identities[two_stage_ranked[0]].exact_key.clone(),
        });
    }

    let denom = full_query_indices.len() as f64;
    let mut sorted_counts = tolerance_counts.clone();
    sorted_counts.sort_unstable();
    let tolerance_median = if sorted_counts.is_empty() {
        0.0
    } else if sorted_counts.len() % 2 == 0 {
        (sorted_counts[sorted_counts.len() / 2 - 1] as f64
            + sorted_counts[sorted_counts.len() / 2] as f64)
            / 2.0
    } else {
        sorted_counts[sorted_counts.len() / 2] as f64
    };
    let tolerance_mean = if sorted_counts.is_empty() {
        0.0
    } else {
        sorted_counts.iter().sum::<usize>() as f64 / sorted_counts.len() as f64
    };
    let tolerance_max = sorted_counts.last().copied().unwrap_or(0);

    Ok((
        V074Metrics {
            queries: full_query_indices.len(),
            candidate_universe: identities.len(),
            candidate_universe_multiple_vs_v070: identities.len() as f64
                / V070_DEV_IDENTITIES as f64,
            mass_pool: V074_MASS_POOL,
            retrieval_shortlist: V074_RETRIEVAL_SHORTLIST,
            mass64_exact_coverage: mass64_exact as f64 / denom,
            mass64_il_coverage: mass64_il as f64 / denom,
            mass256_exact_coverage: mass256_exact as f64 / denom,
            mass256_il_coverage: mass256_il as f64 / denom,
            mass1024_exact_coverage: mass1024_exact as f64 / denom,
            mass1024_il_coverage: mass1024_il as f64 / denom,
            shortlist_exact_coverage: shortlist_exact as f64 / denom,
            shortlist_il_coverage: shortlist_il as f64 / denom,
            precursor_tolerance_exact_coverage: tolerance_exact as f64 / denom,
            precursor_tolerance_il_coverage: tolerance_il as f64 / denom,
            precursor_tolerance_mean_candidates: tolerance_mean,
            precursor_tolerance_median_candidates: tolerance_median,
            precursor_tolerance_max_candidates: tolerance_max,
            retrieval_mass1024: retrieval_acc.metrics(),
            geometry_mass1024: geometry_acc.metrics(),
            two_stage_top64_geometry: two_stage_acc.metrics(),
            geometry_target_beats_best_il_negative_fraction: safe_fraction_v073(
                geometry_beats,
                geometry_margin_queries,
            ),
            geometry_mean_target_score: mean_v073(&target_geometry_scores),
            geometry_mean_best_il_negative_score: mean_v073(&negative_geometry_scores),
            geometry_mean_target_minus_best_il_negative_margin: mean_v073(&geometry_margins),
            mean_within_query_retrieval_geometry_pearson: mean_v073(&correlations),
            candidate_encoding_seconds: 0.0,
            scoring_seconds: 0.0,
            elapsed_seconds: 0.0,
        },
        diagnostics,
    ))
}

fn assert_frozen_checksum_v074(label: &str, initial: f64, current: f64) -> Result<()> {
    let delta = (current - initial).abs();
    let tolerance = 1.0e-6 * initial.abs().max(1.0);
    if delta > tolerance {
        anyhow::bail!(
            "v0.74 frozen {label} changed: initial={initial:.8} current={current:.8} delta={delta:.8}"
        );
    }
    println!(
        "v0740_freeze_audit\tcomponent={label}\tstatus=PASS\tchecksum={current:.8}\tdelta={delta:.8}"
    );
    Ok(())
}

fn print_v074_metrics(m: &V074Metrics) {
    println!(
        "v0740_audit\tqueries={}\tcandidate_universe={}\tuniverse_multiple={:.3}\tmass64_il_coverage={:.6}\tmass256_il_coverage={:.6}\tmass1024_il_coverage={:.6}\tshortlist64_il_coverage={:.6}\ttolerance20ppm_mean_candidates={:.3}\ttolerance20ppm_median_candidates={:.3}\ttolerance20ppm_il_coverage={:.6}\tretrieval_il_top1={:.6}\tretrieval_il_top10={:.6}\tretrieval_il_mrr={:.6}\tgeometry_il_top1={:.6}\tgeometry_il_top10={:.6}\tgeometry_il_mrr={:.6}\ttwo_stage_il_top1={:.6}\ttwo_stage_il_top10={:.6}\ttwo_stage_il_mrr={:.6}\tgeometry_target_beats_negative={:.6}\tmean_retrieval_geometry_pearson={:.6}\tcandidate_encoding_seconds={:.3}\tscoring_seconds={:.3}\telapsed_seconds={:.3}",
        m.queries,
        m.candidate_universe,
        m.candidate_universe_multiple_vs_v070,
        m.mass64_il_coverage,
        m.mass256_il_coverage,
        m.mass1024_il_coverage,
        m.shortlist_il_coverage,
        m.precursor_tolerance_mean_candidates,
        m.precursor_tolerance_median_candidates,
        m.precursor_tolerance_il_coverage,
        m.retrieval_mass1024.il_top1,
        m.retrieval_mass1024.il_top10,
        m.retrieval_mass1024.il_mrr,
        m.geometry_mass1024.il_top1,
        m.geometry_mass1024.il_top10,
        m.geometry_mass1024.il_mrr,
        m.two_stage_top64_geometry.il_top1,
        m.two_stage_top64_geometry.il_top10,
        m.two_stage_top64_geometry.il_mrr,
        m.geometry_target_beats_best_il_negative_fraction,
        m.mean_within_query_retrieval_geometry_pearson,
        m.candidate_encoding_seconds,
        m.scoring_seconds,
        m.elapsed_seconds,
    );
}

fn write_v074_metrics(path: &Path, m: &V074Metrics) -> Result<()> {
    let mut rows = Vec::<(&str, f64)>::new();
    rows.push(("queries", m.queries as f64));
    rows.push(("candidate_universe", m.candidate_universe as f64));
    rows.push((
        "candidate_universe_multiple_vs_v070",
        m.candidate_universe_multiple_vs_v070,
    ));
    rows.push(("mass64_exact_coverage", m.mass64_exact_coverage));
    rows.push(("mass64_il_coverage", m.mass64_il_coverage));
    rows.push(("mass256_exact_coverage", m.mass256_exact_coverage));
    rows.push(("mass256_il_coverage", m.mass256_il_coverage));
    rows.push(("mass1024_exact_coverage", m.mass1024_exact_coverage));
    rows.push(("mass1024_il_coverage", m.mass1024_il_coverage));
    rows.push(("shortlist_exact_coverage", m.shortlist_exact_coverage));
    rows.push(("shortlist_il_coverage", m.shortlist_il_coverage));
    rows.push((
        "precursor_tolerance_exact_coverage",
        m.precursor_tolerance_exact_coverage,
    ));
    rows.push((
        "precursor_tolerance_il_coverage",
        m.precursor_tolerance_il_coverage,
    ));
    rows.push((
        "precursor_tolerance_mean_candidates",
        m.precursor_tolerance_mean_candidates,
    ));
    rows.push((
        "precursor_tolerance_median_candidates",
        m.precursor_tolerance_median_candidates,
    ));
    rows.push((
        "precursor_tolerance_max_candidates",
        m.precursor_tolerance_max_candidates as f64,
    ));
    rows.push(("retrieval_il_top1", m.retrieval_mass1024.il_top1));
    rows.push(("retrieval_il_top10", m.retrieval_mass1024.il_top10));
    rows.push(("retrieval_il_mrr", m.retrieval_mass1024.il_mrr));
    rows.push(("geometry_il_top1", m.geometry_mass1024.il_top1));
    rows.push(("geometry_il_top10", m.geometry_mass1024.il_top10));
    rows.push(("geometry_il_mrr", m.geometry_mass1024.il_mrr));
    rows.push(("two_stage_il_top1", m.two_stage_top64_geometry.il_top1));
    rows.push(("two_stage_il_top10", m.two_stage_top64_geometry.il_top10));
    rows.push(("two_stage_il_mrr", m.two_stage_top64_geometry.il_mrr));
    rows.push((
        "geometry_target_beats_best_il_negative_fraction",
        m.geometry_target_beats_best_il_negative_fraction,
    ));
    rows.push(("geometry_mean_target_score", m.geometry_mean_target_score));
    rows.push((
        "geometry_mean_best_il_negative_score",
        m.geometry_mean_best_il_negative_score,
    ));
    rows.push((
        "geometry_mean_target_minus_best_il_negative_margin",
        m.geometry_mean_target_minus_best_il_negative_margin,
    ));
    rows.push((
        "mean_within_query_retrieval_geometry_pearson",
        m.mean_within_query_retrieval_geometry_pearson,
    ));
    rows.push(("candidate_encoding_seconds", m.candidate_encoding_seconds));
    rows.push(("scoring_seconds", m.scoring_seconds));
    rows.push(("elapsed_seconds", m.elapsed_seconds));
    let mut text = String::from("metric\tvalue\n");
    for (metric, value) in rows {
        text.push_str(&format!("{metric}\t{value:.12}\n"));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_v074_diagnostics(path: &Path, rows: &[V074QueryDiagnostic]) -> Result<()> {
    let mut text = String::from(
        "query_slot\tselected_identity_index\tfull_identity_index\texact_key\tmass64_exact_covered\tmass64_il_covered\tmass256_exact_covered\tmass256_il_covered\tmass1024_exact_covered\tmass1024_il_covered\tshortlist_exact_covered\tshortlist_il_covered\ttolerance_candidates\ttolerance_exact_covered\ttolerance_il_covered\tretrieval_il_rank\tgeometry_il_rank\ttwo_stage_il_rank\tgeometry_target_score\tgeometry_best_il_negative_score\tgeometry_margin\tretrieval_geometry_pearson\tretrieval_top_candidate\tgeometry_top_candidate\ttwo_stage_top_candidate\n",
    );
    for row in rows {
        let option_usize = |value: Option<usize>| {
            value
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NA".to_string())
        };
        let option_f64 = |value: Option<f64>| {
            value
                .map(|v| format!("{v:.8}"))
                .unwrap_or_else(|| "NA".to_string())
        };
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{:.8}\t{}\t{}\t{}\n",
            row.query_slot,
            row.selected_identity_index,
            row.full_identity_index,
            row.exact_key,
            row.mass64_exact_covered,
            row.mass64_il_covered,
            row.mass256_exact_covered,
            row.mass256_il_covered,
            row.mass1024_exact_covered,
            row.mass1024_il_covered,
            row.shortlist_exact_covered,
            row.shortlist_il_covered,
            row.tolerance_candidates,
            row.tolerance_exact_covered,
            row.tolerance_il_covered,
            option_usize(row.retrieval_il_rank),
            option_usize(row.geometry_il_rank),
            option_usize(row.two_stage_il_rank),
            option_f64(row.geometry_target_score),
            row.geometry_best_il_negative_score,
            option_f64(row.geometry_margin),
            row.retrieval_geometry_pearson,
            row.retrieval_top_candidate,
            row.geometry_top_candidate,
            row.two_stage_top_candidate,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_query_spectrum(
    v070: &PeptideSpectrumAlignmentV0700,
    spectrum_collator: &FoundationSpectrumCollator,
    records: &[FoundationTrainingRecord],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let spectra = records
        .iter()
        .map(|record| {
            FoundationSpectrum::from_training_record(record)
                .ok_or_else(|| anyhow::anyhow!("v0.73.1 query record lacks observed spectrum"))
        })
        .collect::<Result<Vec<_>>>()?;
    let spectrum_batch = spectrum_collator.collate(&spectra, device)?;
    let precursor = precursor_features(records, device)?;
    let peak_mask = spectrum_batch.peak_mask.clone();
    let (memory, aligned) = v070.encode_spectrum_t(&spectrum_batch, &precursor, false)?;
    Ok((memory.detach(), peak_mask.detach(), aligned.detach()))
}

fn encode_candidates(
    v052: &PeptideFoundationV0520Model,
    v070: &PeptideSpectrumAlignmentV0700,
    collator: &FoundationCollator,
    records: &[FoundationTrainingRecord],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch = collator.collate(records, device, 0)?;
    let output = v052.base_v0500_t(&batch.input, &batch.context, false)?;
    let peptide_features = Tensor::cat(
        &[
            &output.representation.global_embedding.detach(),
            &output.representation.ms2_embedding.detach(),
        ],
        1,
    )?;
    let aligned = v070.encode_peptide(&peptide_features)?;
    let residue_hidden = output.representation.residue_embeddings.detach();
    let residue_mask = output.representation.residue_mask.detach();
    if residue_hidden.dim(2)? != v070.config.spectrum.model_dim {
        anyhow::bail!(
            "v0.73.1 frozen peptide residue width {} != v0.70 spectrum width {}",
            residue_hidden.dim(2)?,
            v070.config.spectrum.model_dim
        );
    }
    Ok((residue_hidden, residue_mask, aligned.detach()))
}

fn cosine_against_query(candidate_alignment: &Tensor, query_alignment: &Tensor) -> Result<Tensor> {
    let dims = candidate_alignment.dims2()?;
    let query = query_alignment.broadcast_as(dims)?;
    Ok(candidate_alignment.broadcast_mul(&query)?.sum(1)?)
}

fn evaluate_parent_baseline(
    v052: &PeptideFoundationV0520Model,
    v070: &PeptideSpectrumAlignmentV0700,
    collator: &FoundationCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
    batch_size: usize,
    device: &Device,
) -> Result<(RetrievalMetrics, Vec<Vec<f32>>)> {
    let mut spectrum_rows = Vec::<Vec<f32>>::new();
    let mut peptide_rows = Vec::<Vec<f32>>::new();
    for chunk in identities.chunks(batch_size) {
        let owned = chunk
            .iter()
            .map(|item| records[item.record_index].clone())
            .collect::<Vec<_>>();
        let (_, _, spectrum_alignment) =
            encode_query_spectrum(v070, spectrum_collator, &owned, device)?;
        let (_, _, peptide_alignment) = encode_candidates(v052, v070, collator, &owned, device)?;
        spectrum_rows.extend(spectrum_alignment.to_vec2::<f32>()?);
        peptide_rows.extend(peptide_alignment.to_vec2::<f32>()?);
    }
    let n = identities.len();
    let d = v070.config.alignment_dim;
    let spectrum = Tensor::from_vec(
        spectrum_rows.into_iter().flatten().collect::<Vec<_>>(),
        (n, d),
        device,
    )?;
    let peptide = Tensor::from_vec(
        peptide_rows.into_iter().flatten().collect::<Vec<_>>(),
        (n, d),
        device,
    )?;
    let similarities = spectrum
        .matmul(&peptide.transpose(0, 1)?.contiguous()?)?
        .to_vec2::<f32>()?;
    let metrics = retrieval_metrics(&similarities, identities, MASS_POOL)?;
    Ok((metrics, similarities))
}

fn retrieval_metrics(
    similarities: &[Vec<f32>],
    identities: &[RerankIdentity],
    mass_candidates: usize,
) -> Result<RetrievalMetrics> {
    if similarities.len() != identities.len()
        || similarities.iter().any(|row| row.len() != identities.len())
    {
        anyhow::bail!("v0.73.1 parent retrieval similarity matrix shape mismatch");
    }
    let n = identities.len();
    if n == 0 {
        anyhow::bail!("v0.73.1 parent retrieval cohort is empty");
    }
    let mut exact_top1 = 0usize;
    let mut exact_top5 = 0usize;
    let mut exact_top10 = 0usize;
    let mut exact_rr = 0.0f64;
    let mut il_top1 = 0usize;
    let mut il_top5 = 0usize;
    let mut il_top10 = 0usize;
    let mut il_rr = 0.0f64;
    let mut exact_ranks = Vec::with_capacity(n);
    let mut mass_true_coverage = 0usize;
    let mut mass_exact_top1 = 0usize;
    let mut mass_exact_top10 = 0usize;
    let mut mass_exact_rr = 0.0f64;
    let mut mass_il_top1 = 0usize;
    let mut mass_il_top10 = 0usize;
    let mut mass_il_rr = 0.0f64;

    for query in 0..n {
        let mut ranked = (0..n).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            similarities[query][b]
                .partial_cmp(&similarities[query][a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let exact_rank = ranked
            .iter()
            .position(|&candidate| candidate == query)
            .map(|rank| rank + 1)
            .context("v0.73.1 exact v0.70 candidate vanished")?;
        let il_rank = ranked
            .iter()
            .position(|&candidate| identities[candidate].il_key == identities[query].il_key)
            .map(|rank| rank + 1)
            .context("v0.73.1 I/L v0.70 candidate vanished")?;
        exact_top1 += usize::from(exact_rank <= 1);
        exact_top5 += usize::from(exact_rank <= 5);
        exact_top10 += usize::from(exact_rank <= 10);
        exact_rr += 1.0 / exact_rank as f64;
        il_top1 += usize::from(il_rank <= 1);
        il_top5 += usize::from(il_rank <= 5);
        il_top10 += usize::from(il_rank <= 10);
        il_rr += 1.0 / il_rank as f64;
        exact_ranks.push(exact_rank);

        let mut mass_pool = (0..n).collect::<Vec<_>>();
        mass_pool.sort_by(|&a, &b| {
            (identities[a].candidate_neutral_mass - identities[query].observed_neutral_mass)
                .abs()
                .partial_cmp(
                    &(identities[b].candidate_neutral_mass
                        - identities[query].observed_neutral_mass)
                        .abs(),
                )
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        mass_pool.truncate(mass_candidates.min(n));
        mass_true_coverage += usize::from(mass_pool.contains(&query));
        mass_pool.sort_by(|&a, &b| {
            similarities[query][b]
                .partial_cmp(&similarities[query][a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        if let Some(position) = mass_pool.iter().position(|&candidate| candidate == query) {
            let rank = position + 1;
            mass_exact_top1 += usize::from(rank <= 1);
            mass_exact_top10 += usize::from(rank <= 10);
            mass_exact_rr += 1.0 / rank as f64;
        }
        if let Some(position) = mass_pool
            .iter()
            .position(|&candidate| identities[candidate].il_key == identities[query].il_key)
        {
            let rank = position + 1;
            mass_il_top1 += usize::from(rank <= 1);
            mass_il_top10 += usize::from(rank <= 10);
            mass_il_rr += 1.0 / rank as f64;
        }
    }
    let mut sorted_ranks = exact_ranks.clone();
    sorted_ranks.sort_unstable();
    let median_exact_rank = if n % 2 == 0 {
        (sorted_ranks[n / 2 - 1] as f64 + sorted_ranks[n / 2] as f64) / 2.0
    } else {
        sorted_ranks[n / 2] as f64
    };
    let denom = n as f64;
    Ok(RetrievalMetrics {
        identities: n,
        exact_top1: exact_top1 as f64 / denom,
        exact_top5: exact_top5 as f64 / denom,
        exact_top10: exact_top10 as f64 / denom,
        exact_mrr: exact_rr / denom,
        il_top1: il_top1 as f64 / denom,
        il_top5: il_top5 as f64 / denom,
        il_top10: il_top10 as f64 / denom,
        il_mrr: il_rr / denom,
        mean_exact_rank: exact_ranks.iter().sum::<usize>() as f64 / denom,
        median_exact_rank,
        mass_true_coverage: mass_true_coverage as f64 / denom,
        mass_exact_top1: mass_exact_top1 as f64 / denom,
        mass_exact_top10: mass_exact_top10 as f64 / denom,
        mass_exact_mrr: mass_exact_rr / denom,
        mass_il_top1: mass_il_top1 as f64 / denom,
        mass_il_top10: mass_il_top10 as f64 / denom,
        mass_il_mrr: mass_il_rr / denom,
    })
}

fn print_parent_retrieval(label: &str, m: RetrievalMetrics) {
    println!(
        "{label}\tidentities={}\texact_top1={:.6}\texact_top10={:.6}\texact_mrr={:.6}\til_top1={:.6}\til_top10={:.6}\til_mrr={:.6}\tmass64_true_coverage={:.6}\tmass64_exact_top1={:.6}\tmass64_exact_top10={:.6}\tmass64_exact_mrr={:.6}\tmass64_il_top1={:.6}\tmass64_il_top10={:.6}\tmass64_il_mrr={:.6}\tselection_score={:.8}",
        m.identities,
        m.exact_top1,
        m.exact_top10,
        m.exact_mrr,
        m.il_top1,
        m.il_top10,
        m.il_mrr,
        m.mass_true_coverage,
        m.mass_exact_top1,
        m.mass_exact_top10,
        m.mass_exact_mrr,
        m.mass_il_top1,
        m.mass_il_top10,
        m.mass_il_mrr,
        m.selection_score(),
    );
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
            .context("v0.73.1 benchmark record index out of range")?;
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
            anyhow::bail!("v0.73.1 identity grouping collision");
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
) -> Result<Vec<RerankIdentity>> {
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

fn identity_from_group(
    records: &[FoundationTrainingRecord],
    group: &AlignmentGroup,
    seed: u64,
) -> Result<RerankIdentity> {
    let mut record_indices = group.record_indices.clone();
    record_indices.sort_by_key(|&index| mix64(seed ^ index as u64));
    let record_index = *record_indices
        .first()
        .context("v0.73.1 identity group has no records")?;
    let record = &records[record_index];
    let mz = record
        .context
        .precursor_mz
        .context("v0.73.1 identity record lacks precursor m/z")?;
    let observed_neutral_mass = foundation_precursor_neutral_mass(f64::from(mz), group.charge)
        .map_err(anyhow::Error::msg)?;
    let candidate_neutral_mass =
        foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
    Ok(RerankIdentity {
        record_index,
        exact_key: group.key.clone(),
        il_key: format!("{}|z{}", il_label(&group.peptidoform), group.charge),
        charge: group.charge,
        observed_neutral_mass,
        candidate_neutral_mass,
    })
}

fn validate_unique_identities(identities: &[RerankIdentity]) -> Result<()> {
    let unique = identities
        .iter()
        .map(|item| item.exact_key.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() != identities.len() {
        anyhow::bail!("v0.73.1 identity cohort contains duplicate exact identities");
    }
    Ok(())
}

fn deterministic_eval_queries(
    identities: &[RerankIdentity],
    count: usize,
    seed: u64,
) -> Vec<usize> {
    let mut indices = (0..identities.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&index| mix64(seed ^ hash64_str(&identities[index].exact_key)));
    indices.truncate(count.min(indices.len()));
    indices
}

fn precursor_features(records: &[FoundationTrainingRecord], device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(records.len() * 6);
    for record in records {
        let charge = record
            .context
            .charge
            .map(|value| value as f32)
            .unwrap_or(0.0);
        let mz = record.context.precursor_mz.unwrap_or(0.0);
        let nce = record.context.nce.unwrap_or(0.0);
        values.extend_from_slice(&[
            charge / 6.0,
            if record.context.charge.is_some() {
                1.0
            } else {
                0.0
            },
            mz / 2000.0,
            if record.context.precursor_mz.is_some() {
                1.0
            } else {
                0.0
            },
            nce / 100.0,
            if record.context.nce.is_some() {
                1.0
            } else {
                0.0
            },
        ]);
    }
    Ok(Tensor::from_vec(values, (records.len(), 6), device)?)
}

fn normalize_rows(values: &Tensor) -> Result<Tensor> {
    let dims = values.dims2()?;
    let norm = values
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1.0e-8, f64::INFINITY)?
        .unsqueeze(1)?
        .broadcast_as(dims)?;
    Ok(values.broadcast_div(&norm)?)
}

fn varmap_checksum(varmap: &VarMap) -> Result<f64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.73.1 VarMap lock poisoned"))?;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for variable in data.values() {
        if variable.dtype().is_float() {
            sum += f64::from(variable.as_tensor().sum_all()?.to_scalar::<f32>()?);
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.73.1 checksum saw zero floating variables");
    }
    Ok(sum)
}

fn assert_frozen_checksum(label: &str, initial: f64, current: f64) -> Result<()> {
    let delta = (current - initial).abs();
    let tolerance = 1.0e-6 * initial.abs().max(1.0);
    if delta > tolerance {
        anyhow::bail!(
            "v0.73.1 frozen {label} changed: initial={initial:.8} current={current:.8} delta={delta:.8}"
        );
    }
    println!(
        "v0731_freeze_audit\tcomponent={label}\tstatus=PASS\tchecksum={current:.8}\tdelta={delta:.8}"
    );
    Ok(())
}

fn identity_fingerprint(identities: &[RerankIdentity]) -> u64 {
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

fn read_v070_metadata(checkpoint: &Path) -> Result<V070ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.70 metadata {path:?}"))?,
    )
    .map_err(anyhow::Error::from)
}

fn read_v052_metadata(checkpoint: &Path) -> Result<V052ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read v0.52 metadata {path:?}"))?,
    )
    .map_err(anyhow::Error::from)
}

fn validate_v070_parent(metadata: &V070ParentMetadata) -> Result<()> {
    if metadata.version != 700
        || metadata.objective != V070_OBJECTIVE
        || metadata.architecture != V070_ARCHITECTURE
        || metadata.completed_epochs != 6
        || metadata.completed_updates != 6000
        || metadata.dev_identity_count != V070_DEV_IDENTITIES
        || metadata.smoke_mode
        || !metadata.dev_selection_score.is_finite()
        || (metadata.config.temperature - V070_TEMPERATURE).abs() > 1.0e-12
        || metadata.config.spectrum.model_dim != 320
        || metadata.config.alignment_dim != 192
    {
        anyhow::bail!("v0.73.1 requires the selected completed v0.70 epoch6/update6000 checkpoint");
    }
    Ok(())
}

fn validate_v052_parent(metadata: &V052ParentMetadata) -> Result<()> {
    if metadata.version != 520
        || metadata.objective != "v0520_mobility_aware_pair_representation"
        || metadata.architecture != FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520
        || metadata.completed_epochs == 0
        || metadata.completed_updates == 0
        || !metadata.dev_objective.is_finite()
        || metadata.smoke_mode
    {
        anyhow::bail!("v0.73.1 requires the completed non-smoke v0.52 parent referenced by v0.70");
    }
    metadata.v0520_config.validate()?;
    Ok(())
}

fn pass_fail(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

fn read_v035_metadata_v073(checkpoint: &Path) -> Result<V035ParentMetadataV073> {
    let path = checkpoint.join("metadata.yaml");
    let text = fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?;
    serde_yaml::from_str(&text).with_context(|| format!("parse {path:?}"))
}

fn validate_v035_parent_v073(metadata: &V035ParentMetadataV073) -> Result<()> {
    if metadata.version != 350
        || metadata.objective != "v0350_trainable_forward_representation_context_conditioned_ms2"
    {
        anyhow::bail!(
            "v0.73.1 requires the authoritative v0.35 forward metadata referenced by v0.52"
        );
    }
    metadata.v0350_config.validate()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn audit_forward_ms2_attribution_v0731(
    v052: &PeptideFoundationV0520Model,
    collator: &FoundationCollator,
    forward_config: &redeem_properties::foundation::FoundationConfig,
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
    similarities: &[Vec<f32>],
    query_indices: &[usize],
    device: &Device,
) -> Result<(V073AuditMetrics, Vec<V073QueryDiagnostic>)> {
    if query_indices.is_empty() {
        anyhow::bail!("v0.73.1 attribution audit requires at least one query");
    }
    let mut baseline_ranks = Vec::<(usize, usize)>::with_capacity(query_indices.len());
    let mut predicted_ranks = Vec::<(usize, usize)>::with_capacity(query_indices.len());
    let mut geometry_ranks = Vec::<(usize, usize)>::with_capacity(query_indices.len());
    let mut shuffled_ranks = Vec::<(usize, usize)>::with_capacity(query_indices.len());
    let mut diagnostics = Vec::<V073QueryDiagnostic>::with_capacity(query_indices.len());

    let mut predicted_target_scores = Vec::<f64>::new();
    let mut predicted_negative_scores = Vec::<f64>::new();
    let mut predicted_margins = Vec::<f64>::new();
    let mut geometry_target_scores = Vec::<f64>::new();
    let mut geometry_negative_scores = Vec::<f64>::new();
    let mut geometry_margins = Vec::<f64>::new();
    let mut shuffled_target_scores = Vec::<f64>::new();
    let mut shuffled_negative_scores = Vec::<f64>::new();
    let mut shuffled_margins = Vec::<f64>::new();

    let mut v070_predicted_correlations = Vec::<f64>::new();
    let mut predicted_geometry_correlations = Vec::<f64>::new();
    let mut predicted_shuffled_correlations = Vec::<f64>::new();

    let mut candidate_predictions = 0usize;
    let mut candidate_scored = 0usize;
    let mut predicted_score_sum = 0.0f64;
    let mut geometry_score_sum = 0.0f64;
    let mut shuffled_score_sum = 0.0f64;
    let mut score_count = 0usize;
    let mut predicted_nonzero = 0usize;

    let mut baseline_errors = 0usize;
    let mut baseline_correct = 0usize;
    let mut predicted_rescued = 0usize;
    let mut predicted_harmed = 0usize;
    let mut geometry_rescued = 0usize;
    let mut geometry_harmed = 0usize;
    let mut shuffled_rescued = 0usize;
    let mut shuffled_harmed = 0usize;
    let mut predicted_beats_negative = 0usize;
    let mut geometry_beats_negative = 0usize;
    let mut shuffled_beats_negative = 0usize;

    for (query_slot, &query_index) in query_indices.iter().enumerate() {
        let query = &identities[query_index];
        let query_record = &records[query.record_index];
        let observed_spectrum = FoundationSpectrum::from_training_record(query_record)
            .ok_or_else(|| anyhow::anyhow!("v0.73.1 query lacks an observed spectrum"))?;

        let mut mass_pool = v073_mass64_pool(identities, query_index);
        mass_pool.sort_by(|&a, &b| {
            similarities[query_index][b]
                .partial_cmp(&similarities[query_index][a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let baseline_ranked = mass_pool.clone();
        let baseline_exact_rank = rank_exact(&baseline_ranked, identities, query)?;
        let baseline_il_rank = rank_il(&baseline_ranked, identities, query)?;
        baseline_ranks.push((baseline_exact_rank, baseline_il_rank));

        let synthetic = mass_pool
            .iter()
            .map(|&candidate_index| {
                synthetic_forward_candidate_v073(
                    query_record,
                    &records[identities[candidate_index].record_index],
                    &identities[candidate_index],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        candidate_predictions += synthetic.len();
        let predicted =
            predict_forward_ms2_v073(v052, collator, forward_config, &synthetic, device)?;
        if predicted.len() != mass_pool.len() {
            anyhow::bail!("v0.73.1 forward-MS2 prediction count mismatch");
        }

        let mut predicted_scored = Vec::<(usize, f64)>::with_capacity(mass_pool.len());
        let mut geometry_scored = Vec::<(usize, f64)>::with_capacity(mass_pool.len());
        let mut shuffled_scored = Vec::<(usize, f64)>::with_capacity(mass_pool.len());
        for (slot, &candidate_index) in mass_pool.iter().enumerate() {
            let candidate_record = &records[identities[candidate_index].record_index];
            let scores = forward_ms2_attribution_scores_v0731(
                &candidate_record.peptidoform,
                &observed_spectrum,
                &predicted[slot],
                deterministic_shuffle_seed_v0731(query_slot, candidate_index),
            )?;
            if scores.predicted_core_cosine.is_finite()
                && scores.geometry_core_cosine.is_finite()
                && scores.shuffled_core_cosine.is_finite()
            {
                candidate_scored += 1;
                score_count += 1;
                predicted_score_sum += scores.predicted_core_cosine;
                geometry_score_sum += scores.geometry_core_cosine;
                shuffled_score_sum += scores.shuffled_core_cosine;
                predicted_nonzero += usize::from(scores.predicted_core_cosine > 0.0);
                predicted_scored.push((candidate_index, scores.predicted_core_cosine));
                geometry_scored.push((candidate_index, scores.geometry_core_cosine));
                shuffled_scored.push((candidate_index, scores.shuffled_core_cosine));
            }
        }
        if predicted_scored.len() != mass_pool.len()
            || geometry_scored.len() != mass_pool.len()
            || shuffled_scored.len() != mass_pool.len()
        {
            anyhow::bail!("v0.73.1 requires complete attribution-score coverage per mass64 pool");
        }

        let sort_scores = |rows: &mut Vec<(usize, f64)>| {
            rows.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        };
        sort_scores(&mut predicted_scored);
        sort_scores(&mut geometry_scored);
        sort_scores(&mut shuffled_scored);

        let predicted_ranked = predicted_scored.iter().map(|row| row.0).collect::<Vec<_>>();
        let geometry_ranked = geometry_scored.iter().map(|row| row.0).collect::<Vec<_>>();
        let shuffled_ranked = shuffled_scored.iter().map(|row| row.0).collect::<Vec<_>>();

        let predicted_exact_rank = rank_exact(&predicted_ranked, identities, query)?;
        let predicted_il_rank = rank_il(&predicted_ranked, identities, query)?;
        let geometry_exact_rank = rank_exact(&geometry_ranked, identities, query)?;
        let geometry_il_rank = rank_il(&geometry_ranked, identities, query)?;
        let shuffled_exact_rank = rank_exact(&shuffled_ranked, identities, query)?;
        let shuffled_il_rank = rank_il(&shuffled_ranked, identities, query)?;
        predicted_ranks.push((predicted_exact_rank, predicted_il_rank));
        geometry_ranks.push((geometry_exact_rank, geometry_il_rank));
        shuffled_ranks.push((shuffled_exact_rank, shuffled_il_rank));

        let predicted_map = predicted_scored.iter().copied().collect::<BTreeMap<_, _>>();
        let geometry_map = geometry_scored.iter().copied().collect::<BTreeMap<_, _>>();
        let shuffled_map = shuffled_scored.iter().copied().collect::<BTreeMap<_, _>>();

        let target_predicted = predicted_map[&query_index];
        let target_geometry = geometry_map[&query_index];
        let target_shuffled = shuffled_map[&query_index];
        let best_negative_predicted = mass_pool
            .iter()
            .filter(|&&index| identities[index].il_key != query.il_key)
            .map(|index| predicted_map[index])
            .fold(f64::NEG_INFINITY, f64::max);
        let best_negative_geometry = mass_pool
            .iter()
            .filter(|&&index| identities[index].il_key != query.il_key)
            .map(|index| geometry_map[index])
            .fold(f64::NEG_INFINITY, f64::max);
        let best_negative_shuffled = mass_pool
            .iter()
            .filter(|&&index| identities[index].il_key != query.il_key)
            .map(|index| shuffled_map[index])
            .fold(f64::NEG_INFINITY, f64::max);
        if !target_predicted.is_finite()
            || !target_geometry.is_finite()
            || !target_shuffled.is_finite()
            || !best_negative_predicted.is_finite()
            || !best_negative_geometry.is_finite()
            || !best_negative_shuffled.is_finite()
        {
            anyhow::bail!("v0.73.1 could not form attribution positive/negative margins");
        }

        let predicted_margin = target_predicted - best_negative_predicted;
        let geometry_margin = target_geometry - best_negative_geometry;
        let shuffled_margin = target_shuffled - best_negative_shuffled;
        predicted_target_scores.push(target_predicted);
        predicted_negative_scores.push(best_negative_predicted);
        predicted_margins.push(predicted_margin);
        geometry_target_scores.push(target_geometry);
        geometry_negative_scores.push(best_negative_geometry);
        geometry_margins.push(geometry_margin);
        shuffled_target_scores.push(target_shuffled);
        shuffled_negative_scores.push(best_negative_shuffled);
        shuffled_margins.push(shuffled_margin);
        predicted_beats_negative += usize::from(predicted_margin > 0.0);
        geometry_beats_negative += usize::from(geometry_margin > 0.0);
        shuffled_beats_negative += usize::from(shuffled_margin > 0.0);

        let baseline_scores = mass_pool
            .iter()
            .map(|&index| f64::from(similarities[query_index][index]))
            .collect::<Vec<_>>();
        let predicted_scores = mass_pool
            .iter()
            .map(|index| predicted_map[index])
            .collect::<Vec<_>>();
        let geometry_scores = mass_pool
            .iter()
            .map(|index| geometry_map[index])
            .collect::<Vec<_>>();
        let shuffled_scores = mass_pool
            .iter()
            .map(|index| shuffled_map[index])
            .collect::<Vec<_>>();
        let v070_predicted_corr = pearson_v073(&baseline_scores, &predicted_scores);
        let predicted_geometry_corr = pearson_v073(&predicted_scores, &geometry_scores);
        let predicted_shuffled_corr = pearson_v073(&predicted_scores, &shuffled_scores);
        v070_predicted_correlations.push(v070_predicted_corr);
        predicted_geometry_correlations.push(predicted_geometry_corr);
        predicted_shuffled_correlations.push(predicted_shuffled_corr);

        let baseline_ok = baseline_il_rank == 1;
        let predicted_ok = predicted_il_rank == 1;
        let geometry_ok = geometry_il_rank == 1;
        let shuffled_ok = shuffled_il_rank == 1;
        if baseline_ok {
            baseline_correct += 1;
            predicted_harmed += usize::from(!predicted_ok);
            geometry_harmed += usize::from(!geometry_ok);
            shuffled_harmed += usize::from(!shuffled_ok);
        } else {
            baseline_errors += 1;
            predicted_rescued += usize::from(predicted_ok);
            geometry_rescued += usize::from(geometry_ok);
            shuffled_rescued += usize::from(shuffled_ok);
        }

        diagnostics.push(V073QueryDiagnostic {
            query_slot,
            identity_index: query_index,
            exact_key: query.exact_key.clone(),
            baseline_il_rank,
            predicted_il_rank,
            geometry_il_rank,
            shuffled_il_rank,
            predicted_target_score: target_predicted,
            geometry_target_score: target_geometry,
            shuffled_target_score: target_shuffled,
            predicted_best_il_negative_score: best_negative_predicted,
            geometry_best_il_negative_score: best_negative_geometry,
            shuffled_best_il_negative_score: best_negative_shuffled,
            predicted_margin,
            geometry_margin,
            shuffled_margin,
            v070_predicted_pearson: v070_predicted_corr,
            predicted_geometry_pearson: predicted_geometry_corr,
            predicted_shuffled_pearson: predicted_shuffled_corr,
            baseline_top_candidate: identities[baseline_ranked[0]].exact_key.clone(),
            predicted_top_candidate: identities[predicted_ranked[0]].exact_key.clone(),
            geometry_top_candidate: identities[geometry_ranked[0]].exact_key.clone(),
            shuffled_top_candidate: identities[shuffled_ranked[0]].exact_key.clone(),
        });
    }

    predicted_margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let candidate_coverage = if candidate_predictions == 0 {
        0.0
    } else {
        candidate_scored as f64 / candidate_predictions as f64
    };
    let denom = query_indices.len() as f64;
    Ok((
        V073AuditMetrics {
            queries: query_indices.len(),
            candidates_per_query: MASS_POOL,
            candidate_predictions,
            candidate_coverage,
            baseline: rank_metrics_v073(&baseline_ranks),
            predicted_intensity: rank_metrics_v073(&predicted_ranks),
            geometry_only: rank_metrics_v073(&geometry_ranks),
            shuffled_intensity: rank_metrics_v073(&shuffled_ranks),
            mean_predicted_target_core_cosine: mean_v073(&predicted_target_scores),
            mean_predicted_best_il_negative_core_cosine: mean_v073(&predicted_negative_scores),
            mean_predicted_target_minus_best_il_negative_margin: mean_v073(&predicted_margins),
            median_predicted_target_minus_best_il_negative_margin: median_sorted_v073(
                &predicted_margins,
            ),
            predicted_target_beats_best_il_negative_fraction: predicted_beats_negative as f64
                / denom,
            mean_geometry_target_core_cosine: mean_v073(&geometry_target_scores),
            mean_geometry_best_il_negative_core_cosine: mean_v073(&geometry_negative_scores),
            mean_geometry_target_minus_best_il_negative_margin: mean_v073(&geometry_margins),
            geometry_target_beats_best_il_negative_fraction: geometry_beats_negative as f64 / denom,
            mean_shuffled_target_core_cosine: mean_v073(&shuffled_target_scores),
            mean_shuffled_best_il_negative_core_cosine: mean_v073(&shuffled_negative_scores),
            mean_shuffled_target_minus_best_il_negative_margin: mean_v073(&shuffled_margins),
            shuffled_target_beats_best_il_negative_fraction: shuffled_beats_negative as f64 / denom,
            mean_within_query_v070_predicted_pearson: mean_v073(&v070_predicted_correlations),
            mean_within_query_predicted_geometry_pearson: mean_v073(
                &predicted_geometry_correlations,
            ),
            mean_within_query_predicted_shuffled_pearson: mean_v073(
                &predicted_shuffled_correlations,
            ),
            baseline_il_top1_errors: baseline_errors,
            predicted_rescued_baseline_errors: predicted_rescued,
            predicted_rescue_fraction_of_baseline_errors: safe_fraction_v073(
                predicted_rescued,
                baseline_errors,
            ),
            predicted_harmed_baseline_correct: predicted_harmed,
            predicted_harm_fraction_of_baseline_correct: safe_fraction_v073(
                predicted_harmed,
                baseline_correct,
            ),
            geometry_rescued_baseline_errors: geometry_rescued,
            geometry_rescue_fraction_of_baseline_errors: safe_fraction_v073(
                geometry_rescued,
                baseline_errors,
            ),
            geometry_harmed_baseline_correct: geometry_harmed,
            geometry_harm_fraction_of_baseline_correct: safe_fraction_v073(
                geometry_harmed,
                baseline_correct,
            ),
            shuffled_rescued_baseline_errors: shuffled_rescued,
            shuffled_rescue_fraction_of_baseline_errors: safe_fraction_v073(
                shuffled_rescued,
                baseline_errors,
            ),
            shuffled_harmed_baseline_correct: shuffled_harmed,
            shuffled_harm_fraction_of_baseline_correct: safe_fraction_v073(
                shuffled_harmed,
                baseline_correct,
            ),
            mean_predicted_core_cosine_all_candidates: if score_count == 0 {
                0.0
            } else {
                predicted_score_sum / score_count as f64
            },
            mean_geometry_core_cosine_all_candidates: if score_count == 0 {
                0.0
            } else {
                geometry_score_sum / score_count as f64
            },
            mean_shuffled_core_cosine_all_candidates: if score_count == 0 {
                0.0
            } else {
                shuffled_score_sum / score_count as f64
            },
            nonzero_predicted_core_cosine_fraction: safe_fraction_v073(
                predicted_nonzero,
                score_count,
            ),
            elapsed_seconds: 0.0,
        },
        diagnostics,
    ))
}

fn v073_mass64_pool(identities: &[RerankIdentity], query_index: usize) -> Vec<usize> {
    let observed = identities[query_index].observed_neutral_mass;
    let mut pool = (0..identities.len()).collect::<Vec<_>>();
    pool.sort_by(|&a, &b| {
        (identities[a].candidate_neutral_mass - observed)
            .abs()
            .partial_cmp(&(identities[b].candidate_neutral_mass - observed).abs())
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    pool.truncate(MASS_POOL.min(pool.len()));
    pool
}

fn synthetic_forward_candidate_v073(
    query_record: &FoundationTrainingRecord,
    candidate_record: &FoundationTrainingRecord,
    candidate: &RerankIdentity,
) -> Result<FoundationTrainingRecord> {
    if candidate.charge <= 0 {
        anyhow::bail!("v0.73.1 candidate charge must be positive");
    }
    let mut record = candidate_record.clone();
    record.retention_time.normalized = None;
    record.retention_time.harmonized = None;
    record.retention_time.observed_seconds = None;
    record.ccs = None;
    record.fragments.clear();
    record.observed_spectrum_peaks.clear();
    record.context = query_record.context.clone();
    record.context.charge = Some(candidate.charge);
    record.context.precursor_mz = Some(
        ((candidate.candidate_neutral_mass + candidate.charge as f64 * V073_PROTON_MASS_DA)
            / candidate.charge as f64) as f32,
    );
    record.run_id = query_record.run_id.clone();
    Ok(record)
}

fn predict_forward_ms2_v073(
    v052: &PeptideFoundationV0520Model,
    collator: &FoundationCollator,
    forward_config: &redeem_properties::foundation::FoundationConfig,
    records: &[FoundationTrainingRecord],
    device: &Device,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let batch = collator.collate(records, device, 0)?;
    let physics = FoundationScalarPhysicsBatchV0360::from_records(
        records,
        v052.config().base_v0510.base_v0500.max_sequence_len,
        device,
    )?;
    let fragment =
        FoundationFragmentContextBatchV0350::from_records(records, forward_config, device)?;
    let output =
        v052.property_forward_t(&batch.input, &batch.context, &physics, &fragment, false)?;
    output.ms2.to_vec3::<f32>().map_err(anyhow::Error::from)
}

fn deterministic_shuffle_seed_v0731(query_slot: usize, candidate_index: usize) -> u64 {
    let mut x = (query_slot as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (candidate_index as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn forward_ms2_attribution_scores_v0731(
    peptide: &redeem_properties::foundation::PeptidoformInput,
    spectrum: &FoundationSpectrum,
    predicted_ms2: &[Vec<f32>],
    shuffle_seed: u64,
) -> Result<V073AttributionScores> {
    let geometry = foundation_fragment_cleavage_geometry(peptide).map_err(anyhow::Error::msg)?;
    if predicted_ms2.len() < geometry.len() {
        anyhow::bail!(
            "v0.73.1 predicted MS2 rows {} shorter than candidate cleavage count {}",
            predicted_ms2.len(),
            geometry.len()
        );
    }
    let peaks = normalized_retained_peaks_v073(spectrum);
    let mut predicted_core = Vec::<f64>::with_capacity(geometry.len() * 4);
    let mut observed_core = Vec::<f64>::with_capacity(geometry.len() * 4);
    for cleavage in &geometry {
        let row = &predicted_ms2[cleavage.cleavage_index];
        if row.len() < 4 {
            anyhow::bail!("v0.73.1 forward MS2 row has fewer than four core channels");
        }
        for channel in 0..4 {
            predicted_core.push(f64::from(row[channel]).max(0.0));
            observed_core.push(best_peak_support_v073(cleavage.core_mz[channel], &peaks));
        }
    }

    let geometry_core = vec![1.0f64; predicted_core.len()];
    let mut shuffled_core = predicted_core.clone();
    if shuffled_core.len() > 1 {
        let offset = 1 + (shuffle_seed as usize % (shuffled_core.len() - 1));
        shuffled_core.rotate_left(offset);
    }

    Ok(V073AttributionScores {
        predicted_core_cosine: sqrt_intensity_cosine_v073(&predicted_core, &observed_core),
        geometry_core_cosine: sqrt_intensity_cosine_v073(&geometry_core, &observed_core),
        shuffled_core_cosine: sqrt_intensity_cosine_v073(&shuffled_core, &observed_core),
    })
}

fn normalized_retained_peaks_v073(spectrum: &FoundationSpectrum) -> Vec<(f64, f64)> {
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

fn best_peak_support_v073(theoretical_mz: f64, peaks: &[(f64, f64)]) -> f64 {
    if !(theoretical_mz > 0.0 && theoretical_mz.is_finite()) {
        return 0.0;
    }
    let sigma = (theoretical_mz * FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230 * 1e-6)
        .max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230 / 3.0);
    let cutoff = (3.0 * sigma).max(FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230);
    peaks
        .iter()
        .filter_map(|&(observed_mz, normalized_intensity)| {
            let error = (observed_mz - theoretical_mz).abs();
            (error <= cutoff).then(|| {
                let mass_weight = (-0.5 * (error / sigma).powi(2)).exp();
                normalized_intensity * mass_weight
            })
        })
        .fold(0.0f64, f64::max)
}

fn sqrt_intensity_cosine_v073(predicted: &[f64], observed: &[f64]) -> f64 {
    if predicted.len() != observed.len() || predicted.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut pred_norm = 0.0f64;
    let mut obs_norm = 0.0f64;
    for (&p, &o) in predicted.iter().zip(observed) {
        let p = p.max(0.0).sqrt();
        let o = o.max(0.0).sqrt();
        dot += p * o;
        pred_norm += p * p;
        obs_norm += o * o;
    }
    if pred_norm > 0.0 && obs_norm > 0.0 {
        (dot / (pred_norm.sqrt() * obs_norm.sqrt())).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn rank_exact(
    ranked: &[usize],
    identities: &[RerankIdentity],
    query: &RerankIdentity,
) -> Result<usize> {
    ranked
        .iter()
        .position(|&index| identities[index].exact_key == query.exact_key)
        .map(|rank| rank + 1)
        .context("v0.73.1 exact target vanished from mass64 candidate pool")
}

fn rank_il(
    ranked: &[usize],
    identities: &[RerankIdentity],
    query: &RerankIdentity,
) -> Result<usize> {
    ranked
        .iter()
        .position(|&index| identities[index].il_key == query.il_key)
        .map(|rank| rank + 1)
        .context("v0.73.1 I/L target vanished from mass64 candidate pool")
}

fn rank_metrics_v073(ranks: &[(usize, usize)]) -> V073RankMetrics {
    if ranks.is_empty() {
        return V073RankMetrics::default();
    }
    let denom = ranks.len() as f64;
    V073RankMetrics {
        exact_top1: ranks.iter().filter(|row| row.0 == 1).count() as f64 / denom,
        exact_top10: ranks.iter().filter(|row| row.0 <= 10).count() as f64 / denom,
        exact_mrr: ranks.iter().map(|row| 1.0 / row.0 as f64).sum::<f64>() / denom,
        il_top1: ranks.iter().filter(|row| row.1 == 1).count() as f64 / denom,
        il_top10: ranks.iter().filter(|row| row.1 <= 10).count() as f64 / denom,
        il_mrr: ranks.iter().map(|row| 1.0 / row.1 as f64).sum::<f64>() / denom,
    }
}

fn pearson_v073(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < 2 {
        return 0.0;
    }
    let mean_left = mean_v073(left);
    let mean_right = mean_v073(right);
    let mut dot = 0.0;
    let mut left_sq = 0.0;
    let mut right_sq = 0.0;
    for (&a, &b) in left.iter().zip(right) {
        let da = a - mean_left;
        let db = b - mean_right;
        dot += da * db;
        left_sq += da * da;
        right_sq += db * db;
    }
    if left_sq > 0.0 && right_sq > 0.0 {
        (dot / (left_sq.sqrt() * right_sq.sqrt())).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn mean_v073(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median_sorted_v073(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

fn safe_fraction_v073(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn print_v073_audit(label: &str, m: &V073AuditMetrics) {
    println!(
        "{label}\tqueries={}\tcandidates_per_query={}\tcandidate_coverage={:.6}\tbaseline_il_top1={:.6}\tbaseline_il_top10={:.6}\tbaseline_il_mrr={:.6}\tpredicted_il_top1={:.6}\tpredicted_il_top10={:.6}\tpredicted_il_mrr={:.6}\tgeometry_il_top1={:.6}\tgeometry_il_top10={:.6}\tgeometry_il_mrr={:.6}\tshuffled_il_top1={:.6}\tshuffled_il_top10={:.6}\tshuffled_il_mrr={:.6}\tpredicted_target_beats_negative={:.6}\tgeometry_target_beats_negative={:.6}\tshuffled_target_beats_negative={:.6}\tpredicted_rescue_fraction={:.6}\tgeometry_rescue_fraction={:.6}\tshuffled_rescue_fraction={:.6}\tpredicted_harm_fraction={:.6}\tgeometry_harm_fraction={:.6}\tshuffled_harm_fraction={:.6}\tmean_v070_predicted_pearson={:.6}\tmean_predicted_geometry_pearson={:.6}\tmean_predicted_shuffled_pearson={:.6}\telapsed_seconds={:.3}",
        m.queries,
        m.candidates_per_query,
        m.candidate_coverage,
        m.baseline.il_top1,
        m.baseline.il_top10,
        m.baseline.il_mrr,
        m.predicted_intensity.il_top1,
        m.predicted_intensity.il_top10,
        m.predicted_intensity.il_mrr,
        m.geometry_only.il_top1,
        m.geometry_only.il_top10,
        m.geometry_only.il_mrr,
        m.shuffled_intensity.il_top1,
        m.shuffled_intensity.il_top10,
        m.shuffled_intensity.il_mrr,
        m.predicted_target_beats_best_il_negative_fraction,
        m.geometry_target_beats_best_il_negative_fraction,
        m.shuffled_target_beats_best_il_negative_fraction,
        m.predicted_rescue_fraction_of_baseline_errors,
        m.geometry_rescue_fraction_of_baseline_errors,
        m.shuffled_rescue_fraction_of_baseline_errors,
        m.predicted_harm_fraction_of_baseline_correct,
        m.geometry_harm_fraction_of_baseline_correct,
        m.shuffled_harm_fraction_of_baseline_correct,
        m.mean_within_query_v070_predicted_pearson,
        m.mean_within_query_predicted_geometry_pearson,
        m.mean_within_query_predicted_shuffled_pearson,
        m.elapsed_seconds,
    );
}

fn write_v073_metrics(path: &Path, m: &V073AuditMetrics) -> Result<()> {
    let rows = [
        ("queries", m.queries as f64),
        ("candidate_predictions", m.candidate_predictions as f64),
        ("candidate_coverage", m.candidate_coverage),
        ("baseline_il_top1", m.baseline.il_top1),
        ("baseline_il_top10", m.baseline.il_top10),
        ("baseline_il_mrr", m.baseline.il_mrr),
        ("predicted_il_top1", m.predicted_intensity.il_top1),
        ("predicted_il_top10", m.predicted_intensity.il_top10),
        ("predicted_il_mrr", m.predicted_intensity.il_mrr),
        ("geometry_il_top1", m.geometry_only.il_top1),
        ("geometry_il_top10", m.geometry_only.il_top10),
        ("geometry_il_mrr", m.geometry_only.il_mrr),
        ("shuffled_il_top1", m.shuffled_intensity.il_top1),
        ("shuffled_il_top10", m.shuffled_intensity.il_top10),
        ("shuffled_il_mrr", m.shuffled_intensity.il_mrr),
        (
            "mean_predicted_target_core_cosine",
            m.mean_predicted_target_core_cosine,
        ),
        (
            "mean_predicted_best_il_negative_core_cosine",
            m.mean_predicted_best_il_negative_core_cosine,
        ),
        (
            "mean_predicted_target_minus_best_il_negative_margin",
            m.mean_predicted_target_minus_best_il_negative_margin,
        ),
        (
            "median_predicted_target_minus_best_il_negative_margin",
            m.median_predicted_target_minus_best_il_negative_margin,
        ),
        (
            "predicted_target_beats_best_il_negative_fraction",
            m.predicted_target_beats_best_il_negative_fraction,
        ),
        (
            "mean_geometry_target_core_cosine",
            m.mean_geometry_target_core_cosine,
        ),
        (
            "mean_geometry_best_il_negative_core_cosine",
            m.mean_geometry_best_il_negative_core_cosine,
        ),
        (
            "mean_geometry_target_minus_best_il_negative_margin",
            m.mean_geometry_target_minus_best_il_negative_margin,
        ),
        (
            "geometry_target_beats_best_il_negative_fraction",
            m.geometry_target_beats_best_il_negative_fraction,
        ),
        (
            "mean_shuffled_target_core_cosine",
            m.mean_shuffled_target_core_cosine,
        ),
        (
            "mean_shuffled_best_il_negative_core_cosine",
            m.mean_shuffled_best_il_negative_core_cosine,
        ),
        (
            "mean_shuffled_target_minus_best_il_negative_margin",
            m.mean_shuffled_target_minus_best_il_negative_margin,
        ),
        (
            "shuffled_target_beats_best_il_negative_fraction",
            m.shuffled_target_beats_best_il_negative_fraction,
        ),
        (
            "mean_within_query_v070_predicted_pearson",
            m.mean_within_query_v070_predicted_pearson,
        ),
        (
            "mean_within_query_predicted_geometry_pearson",
            m.mean_within_query_predicted_geometry_pearson,
        ),
        (
            "mean_within_query_predicted_shuffled_pearson",
            m.mean_within_query_predicted_shuffled_pearson,
        ),
        (
            "predicted_rescue_fraction_of_baseline_errors",
            m.predicted_rescue_fraction_of_baseline_errors,
        ),
        (
            "predicted_harm_fraction_of_baseline_correct",
            m.predicted_harm_fraction_of_baseline_correct,
        ),
        (
            "geometry_rescue_fraction_of_baseline_errors",
            m.geometry_rescue_fraction_of_baseline_errors,
        ),
        (
            "geometry_harm_fraction_of_baseline_correct",
            m.geometry_harm_fraction_of_baseline_correct,
        ),
        (
            "shuffled_rescue_fraction_of_baseline_errors",
            m.shuffled_rescue_fraction_of_baseline_errors,
        ),
        (
            "shuffled_harm_fraction_of_baseline_correct",
            m.shuffled_harm_fraction_of_baseline_correct,
        ),
        (
            "mean_predicted_core_cosine_all_candidates",
            m.mean_predicted_core_cosine_all_candidates,
        ),
        (
            "mean_geometry_core_cosine_all_candidates",
            m.mean_geometry_core_cosine_all_candidates,
        ),
        (
            "mean_shuffled_core_cosine_all_candidates",
            m.mean_shuffled_core_cosine_all_candidates,
        ),
        (
            "nonzero_predicted_core_cosine_fraction",
            m.nonzero_predicted_core_cosine_fraction,
        ),
        ("elapsed_seconds", m.elapsed_seconds),
    ];
    let mut text = String::from("metric\tvalue\n");
    for (name, value) in rows {
        text.push_str(&format!("{name}\t{value:.12}\n"));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_v073_diagnostics(path: &Path, rows: &[V073QueryDiagnostic]) -> Result<()> {
    let mut text = String::from("query_slot\tidentity_index\texact_key\tbaseline_il_rank\tpredicted_il_rank\tgeometry_il_rank\tshuffled_il_rank\tpredicted_target_score\tgeometry_target_score\tshuffled_target_score\tpredicted_best_il_negative_score\tgeometry_best_il_negative_score\tshuffled_best_il_negative_score\tpredicted_margin\tgeometry_margin\tshuffled_margin\tv070_predicted_pearson\tpredicted_geometry_pearson\tpredicted_shuffled_pearson\tbaseline_top_candidate\tpredicted_top_candidate\tgeometry_top_candidate\tshuffled_top_candidate\n");
    for row in rows {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\n",
            row.query_slot,
            row.identity_index,
            row.exact_key,
            row.baseline_il_rank,
            row.predicted_il_rank,
            row.geometry_il_rank,
            row.shuffled_il_rank,
            row.predicted_target_score,
            row.geometry_target_score,
            row.shuffled_target_score,
            row.predicted_best_il_negative_score,
            row.geometry_best_il_negative_score,
            row.shuffled_best_il_negative_score,
            row.predicted_margin,
            row.geometry_margin,
            row.shuffled_margin,
            row.v070_predicted_pearson,
            row.predicted_geometry_pearson,
            row.predicted_shuffled_pearson,
            row.baseline_top_candidate,
            row.predicted_top_candidate,
            row.geometry_top_candidate,
            row.shuffled_top_candidate,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod v0731_tests {
    use super::*;
    use redeem_properties::foundation::PeptidoformInput;

    #[test]
    fn v0731_predicted_and_geometry_controls_score_aligned_ions() {
        let peptide = PeptidoformInput::unmodified("AG");
        let spectrum = FoundationSpectrum::from_pairs([(72.0444, 100.0), (76.0393, 80.0)]);
        let predicted = vec![vec![1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let score =
            forward_ms2_attribution_scores_v0731(&peptide, &spectrum, &predicted, 0).unwrap();
        assert!(score.predicted_core_cosine > 0.8, "score={score:?}");
        assert!(score.geometry_core_cosine > 0.0, "score={score:?}");
        assert!(
            score.predicted_core_cosine > score.shuffled_core_cosine,
            "score={score:?}"
        );
    }

    #[test]
    fn v0731_rank_metrics_prioritize_il_equivalence() {
        let ranks = vec![(2, 1), (1, 1), (5, 4)];
        let metrics = rank_metrics_v073(&ranks);
        assert!(metrics.il_top1 > metrics.exact_top1);
        assert!(metrics.il_mrr >= metrics.exact_mrr);
    }

    #[test]
    fn v0731_shuffle_seed_is_deterministic_and_candidate_specific() {
        let a = deterministic_shuffle_seed_v0731(7, 11);
        let b = deterministic_shuffle_seed_v0731(7, 11);
        let c = deterministic_shuffle_seed_v0731(7, 12);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn v0731_pearson_detects_independent_ordering() {
        let same = pearson_v073(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        let reverse = pearson_v073(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
        assert!((same - 1.0).abs() < 1.0e-12);
        assert!((reverse + 1.0).abs() < 1.0e-12);
    }
}

#[cfg(test)]
mod v074_tests {
    use super::*;

    #[test]
    fn v074_nearest_mass_pool_is_distance_ordered_without_target_forcing() {
        let mass_sorted = vec![(99.0, 0), (100.0, 1), (100.3, 2), (101.0, 3), (103.0, 4)];
        let pool = nearest_mass_pool_v074(&mass_sorted, 100.2, 3);
        assert_eq!(pool, vec![2, 1, 3]);
        assert!(!pool.contains(&0));
    }

    #[test]
    fn v074_geometry_score_rewards_mass_aligned_fragment_support() {
        let peaks = vec![(100.0, 1.0), (200.0, 0.64), (300.0, 0.36)];
        let aligned = geometry_uniform_score_v074(&[100.0, 200.0, 300.0], &peaks);
        let shifted = geometry_uniform_score_v074(&[110.0, 210.0, 310.0], &peaks);
        assert!(aligned > 0.9);
        assert_eq!(shifted, 0.0);
        assert!(aligned > shifted);
    }

    #[test]
    fn v074_missing_target_counts_as_ranking_failure() {
        let mut acc = V074RankAccumulator::default();
        acc.observe(Some(1), Some(1));
        acc.observe(None, None);
        let metrics = acc.metrics();
        assert_eq!(metrics.exact_top1, 0.5);
        assert_eq!(metrics.il_top1, 0.5);
        assert_eq!(metrics.exact_mrr, 0.5);
        assert_eq!(metrics.il_mrr, 0.5);
    }

    #[test]
    fn v074_full_candidate_encoding_uses_v070_proven_batch_bound() {
        assert!(V074_CANDIDATE_BATCH <= 64);
        assert!(V074_CANDIDATE_BATCH > 0);
    }
}
