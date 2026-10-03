//! v0.73 frozen forward-MS2 candidate-consistency audit.
//!
//! Scientific contract:
//! - reproduce the selected v0.70 DEV retrieval cohort and mass64 ranking exactly;
//! - freeze both v0.70 and its selected v0.52 forward parent;
//! - predict v0.52 MS2 for every mass64 candidate under the query acquisition context;
//! - align predicted b/y core intensities to the measured spectrum using open-PTM
//!   theoretical fragment geometry and the established v0.23 fixed mass tolerance;
//! - quantify whether forward-MS2 consistency rescues v0.70 ranking errors before
//!   fitting any fusion model;
//! - use DEV only; never touch TRAIN-HOLDOUT, historical VALIDATION/APD, or TEST.
//!
//! The only fusion reported is one predeclared equal-rank-sum diagnostic. No
//! trainable parameters or fitted combination weights exist in this audit.

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

const V073_VERSION: u32 = 730;
const V073_OBJECTIVE: &str =
    "v0730_frozen_v070_mass64_frozen_v052_forward_ms2_candidate_consistency_audit";
const V073_ARCHITECTURE: &str =
    "frozen_v070_mass64_retrieval_plus_frozen_v052_open_ptm_forward_ms2_consistency";
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
const V073_MIN_TARGET_BEATS_BEST_NEGATIVE: f64 = 0.10;
const V073_MIN_MS2_ERROR_RESCUE_FRACTION: f64 = 0.10;
const V073_MIN_FUSION_SELECTION_GAIN: f64 = 0.01;
const V073_MAX_FUSION_TOP10_REGRESSION: f64 = 0.005;
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
    ms2_only: V073RankMetrics,
    fixed_equal_rank_fusion: V073RankMetrics,
    oracle_il_top1: f64,
    mean_exact_target_core_cosine: f64,
    mean_best_il_negative_core_cosine: f64,
    mean_exact_target_minus_best_il_negative_margin: f64,
    median_exact_target_minus_best_il_negative_margin: f64,
    exact_target_beats_best_il_negative_fraction: f64,
    mean_within_query_v070_ms2_pearson: f64,
    baseline_il_top1_errors: usize,
    ms2_rescued_baseline_errors: usize,
    ms2_rescue_fraction_of_baseline_errors: f64,
    baseline_il_top1_correct: usize,
    ms2_harmed_baseline_correct: usize,
    ms2_harm_fraction_of_baseline_correct: f64,
    fusion_rescued_baseline_errors: usize,
    fusion_rescue_fraction_of_baseline_errors: f64,
    fusion_harmed_baseline_correct: usize,
    fusion_harm_fraction_of_baseline_correct: f64,
    mean_core_cosine_all_candidates: f64,
    nonzero_core_cosine_fraction: f64,
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
    baseline_exact_rank: usize,
    baseline_il_rank: usize,
    ms2_exact_rank: usize,
    ms2_il_rank: usize,
    fusion_exact_rank: usize,
    fusion_il_rank: usize,
    exact_target_core_cosine: f64,
    best_il_negative_core_cosine: f64,
    exact_target_minus_best_il_negative_margin: f64,
    within_query_pearson: f64,
    baseline_top_candidate: String,
    ms2_top_candidate: String,
    fusion_top_candidate: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct V073ForwardMs2Score {
    core_cosine: f64,
    cleavage_cosine: f64,
    matched_core_ions: usize,
    core_ions: usize,
    predicted_supported_fraction: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        anyhow::bail!(
            "usage: foundation_audit_forward_ms2_consistency_v0730 RUN_V0260.yaml OUTPUT_DIR PARENT_V070_BEST mode=smoke|audit"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let parent_v070_checkpoint = PathBuf::from(&args[3]);
    let mode = args[4].as_str();
    if !matches!(mode, "smoke" | "audit") {
        anyhow::bail!("v0.73 mode must be smoke or audit");
    }
    if output_root.exists() {
        anyhow::bail!("v0.73 output directory must be fresh: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.73 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let audit_queries = if mode == "smoke" {
        V073_SMOKE_QUERIES
    } else {
        V073_AUDIT_QUERIES
    };

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
        anyhow::bail!("v0.73 v0.70 parent provenance differs from current corpus/benchmark");
    }

    let parent_v052_checkpoint = PathBuf::from(&parent_v070_metadata.parent_v052_checkpoint);
    let parent_v052_metadata = read_v052_metadata(&parent_v052_checkpoint)?;
    validate_v052_parent(&parent_v052_metadata)?;
    if parent_v052_metadata.corpus_fingerprint != current_corpus_fingerprint
        || parent_v052_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint
    {
        anyhow::bail!("v0.73 v0.52 parent provenance differs from current corpus/benchmark");
    }

    let parent_v0350_checkpoint = PathBuf::from(&parent_v052_metadata.parent_v0350_checkpoint);
    let parent_v0350_metadata = read_v035_metadata_v073(&parent_v0350_checkpoint)?;
    validate_v035_parent_v073(&parent_v0350_metadata)?;
    if parent_v0350_metadata.corpus_fingerprint != current_corpus_fingerprint
        || parent_v0350_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint
    {
        anyhow::bail!("v0.73 v0.35 parent provenance differs from current corpus/benchmark");
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
    if dev_groups.len() < V070_DEV_IDENTITIES {
        anyhow::bail!(
            "v0.73 requires the established v0.70 DEV retrieval population; dev={}",
            dev_groups.len()
        );
    }
    let dev_identities = select_dev_identities(
        &corpus.records,
        &dev_groups,
        parent_v070_metadata.dev_identity_count,
        parent_v070_metadata.seed ^ 0x7000_d3f0_a11e_0001,
    )?;
    if dev_identities.len() != V070_DEV_IDENTITIES {
        anyhow::bail!(
            "v0.73 expected {V070_DEV_IDENTITIES} v0.70 DEV identities, observed {}",
            dev_identities.len()
        );
    }
    let dev_identity_fingerprint =
        format!("fnv1a64:{:016x}", identity_fingerprint(&dev_identities));
    if dev_identity_fingerprint != parent_v070_metadata.dev_identity_fingerprint {
        anyhow::bail!(
            "v0.73 DEV identity fingerprint differs from selected v0.70 parent: current={} parent={}",
            dev_identity_fingerprint,
            parent_v070_metadata.dev_identity_fingerprint
        );
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

    println!("v0730_version\tv0.73-forward-ms2-candidate-consistency-audit");
    println!("objective\t{V073_OBJECTIVE}");
    println!("architecture\t{V073_ARCHITECTURE}");
    println!("score\t{V073_SCORE}");
    println!("device\t{device:?}");
    println!("mode\t{mode}");
    println!(
        "parent_v070_checkpoint\t{}",
        parent_v070_checkpoint.display()
    );
    println!(
        "parent_v070_completed_epochs\t{}",
        parent_v070_metadata.completed_epochs
    );
    println!(
        "parent_v070_completed_updates\t{}",
        parent_v070_metadata.completed_updates
    );
    println!(
        "parent_v070_dev_selection_score\t{:.8}",
        parent_v070_metadata.dev_selection_score
    );
    println!(
        "parent_v052_checkpoint\t{}",
        parent_v052_checkpoint.display()
    );
    println!(
        "parent_v0350_checkpoint\t{}",
        parent_v0350_checkpoint.display()
    );
    println!("parent_update_policy\tfrozen_v070_plus_frozen_v052_no_trainable_variables");
    println!("candidate_policy\texact_v070_neutral_mass64");
    println!("candidate_forward_context\tquery_nce_instrument_plus_candidate_charge_theoretical_precursor_mz");
    println!("forward_ms2_primary_score\t{V073_SCORE}");
    println!("fusion_diagnostic\tfixed_equal_rank_sum_v070_plus_forward_ms2_no_fitted_weight");
    println!("mass_pool\t{MASS_POOL}");
    println!("v070_dev_identity_count\t{}", dev_identities.len());
    println!("v070_dev_identity_fingerprint\t{dev_identity_fingerprint}");
    println!("audit_queries\t{audit_queries}");
    println!("holdout_records_reserved_not_read\t{holdout_reserved}");
    println!("rt_conditioning\tNO");
    println!("ccs_conditioning\tNO");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    let (parent_baseline, similarities) = evaluate_parent_baseline(
        &v052,
        &v070,
        &collator,
        &spectrum_collator,
        &corpus.records,
        &dev_identities,
        64,
        &device,
    )?;
    print_parent_retrieval("v0730_parent_v070_reproduction", parent_baseline);
    let reproduction_delta =
        (parent_baseline.selection_score() - parent_v070_metadata.dev_selection_score).abs();
    if reproduction_delta > 1.0e-5 {
        anyhow::bail!(
            "v0.73 failed v0.70 parent reproduction: current={:.8} parent={:.8} delta={:.8}",
            parent_baseline.selection_score(),
            parent_v070_metadata.dev_selection_score,
            reproduction_delta
        );
    }
    println!("v0730_parent_reproduction_gate\tPASS\tdelta={reproduction_delta:.8}");

    let eval_query_indices = deterministic_eval_queries(
        &dev_identities,
        audit_queries,
        V073_SEED ^ 0x7300_d3f0_0000_0001,
    );
    fs::create_dir_all(&output_root)?;
    let audit_started = Instant::now();
    let (mut audit, diagnostics) = audit_forward_ms2_consistency_v073(
        &v052,
        &collator,
        parent_v0350_metadata.v0350_config.forward(),
        &corpus.records,
        &dev_identities,
        &similarities,
        &eval_query_indices,
        &device,
    )?;
    audit.elapsed_seconds = audit_started.elapsed().as_secs_f64();

    let v052_checksum = varmap_checksum(&v052_varmap)?;
    let v070_checksum = varmap_checksum(&v070_varmap)?;
    assert_frozen_checksum("v052_forward_parent", v052_checksum_initial, v052_checksum)?;
    assert_frozen_checksum(
        "v070_alignment_parent",
        v070_checksum_initial,
        v070_checksum,
    )?;

    print_v073_audit("v0730_audit", &audit);
    let fusion_selection_gain =
        audit.fixed_equal_rank_fusion.selection() - audit.baseline.selection();
    let fusion_top10_delta = audit.fixed_equal_rank_fusion.il_top10 - audit.baseline.il_top10;
    let gate_coverage = audit.candidate_coverage >= V073_MIN_CANDIDATE_COVERAGE;
    let gate_target_margin =
        audit.exact_target_beats_best_il_negative_fraction >= V073_MIN_TARGET_BEATS_BEST_NEGATIVE;
    let gate_rescue =
        audit.ms2_rescue_fraction_of_baseline_errors >= V073_MIN_MS2_ERROR_RESCUE_FRACTION;
    let gate_fusion = fusion_selection_gain >= V073_MIN_FUSION_SELECTION_GAIN;
    let gate_top10 = fusion_top10_delta >= -V073_MAX_FUSION_TOP10_REGRESSION;
    println!("v0730_fixed_fusion_selection_gain\t{fusion_selection_gain:.8}");
    println!("v0730_fixed_fusion_il_top10_delta\t{fusion_top10_delta:.8}");
    println!(
        "v0730_gate_candidate_coverage_ge_0_99\t{}",
        pass_fail(gate_coverage)
    );
    println!(
        "v0730_gate_exact_target_beats_best_negative_ge_0_10\t{}",
        pass_fail(gate_target_margin)
    );
    println!(
        "v0730_gate_ms2_error_rescue_fraction_ge_0_10\t{}",
        pass_fail(gate_rescue)
    );
    println!(
        "v0730_gate_fixed_fusion_selection_gain_ge_0_01\t{}",
        pass_fail(gate_fusion)
    );
    println!(
        "v0730_gate_fixed_fusion_top10_no_regression_gt_0_005\t{}",
        pass_fail(gate_top10)
    );
    let decision = if mode == "smoke" {
        "SMOKE_MECHANICAL_ONLY"
    } else if gate_coverage && gate_target_margin && gate_rescue && gate_fusion && gate_top10 {
        "GO_FORWARD_MS2_FUSION"
    } else {
        "CLOSE_FORWARD_MS2_FUSION_NO_MATERIAL_COMPLEMENTARITY"
    };
    println!("v0730_audit_decision\t{decision}");
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");

    write_v073_metrics(&output_root.join("audit_metrics.tsv"), &audit)?;
    write_v073_diagnostics(&output_root.join("query_diagnostics.tsv"), &diagnostics)?;
    let metadata = V073Metadata {
        version: V073_VERSION,
        objective: V073_OBJECTIVE.into(),
        architecture: V073_ARCHITECTURE.into(),
        score: V073_SCORE.into(),
        mode: mode.into(),
        parent_v070_checkpoint: parent_v070_checkpoint.display().to_string(),
        parent_v052_checkpoint: parent_v052_checkpoint.display().to_string(),
        parent_v0350_checkpoint: parent_v0350_checkpoint.display().to_string(),
        corpus_fingerprint: current_corpus_fingerprint,
        benchmark_manifest_fingerprint: current_benchmark_fingerprint,
        dev_identity_fingerprint,
        parent_v070_selection_score: parent_v070_metadata.dev_selection_score,
        reproduction_delta,
        audit_queries,
        mass_pool: MASS_POOL,
        seed: V073_SEED,
        metrics: audit,
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
                .ok_or_else(|| anyhow::anyhow!("v0.73 query record lacks observed spectrum"))
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
            "v0.73 frozen peptide residue width {} != v0.70 spectrum width {}",
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
        anyhow::bail!("v0.73 parent retrieval similarity matrix shape mismatch");
    }
    let n = identities.len();
    if n == 0 {
        anyhow::bail!("v0.73 parent retrieval cohort is empty");
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
            .context("v0.73 exact v0.70 candidate vanished")?;
        let il_rank = ranked
            .iter()
            .position(|&candidate| identities[candidate].il_key == identities[query].il_key)
            .map(|rank| rank + 1)
            .context("v0.73 I/L v0.70 candidate vanished")?;
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
            .context("v0.73 benchmark record index out of range")?;
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
            anyhow::bail!("v0.73 identity grouping collision");
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
        .context("v0.73 identity group has no records")?;
    let record = &records[record_index];
    let mz = record
        .context
        .precursor_mz
        .context("v0.73 identity record lacks precursor m/z")?;
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
        anyhow::bail!("v0.73 identity cohort contains duplicate exact identities");
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
        .map_err(|_| anyhow::anyhow!("v0.73 VarMap lock poisoned"))?;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for variable in data.values() {
        if variable.dtype().is_float() {
            sum += f64::from(variable.as_tensor().sum_all()?.to_scalar::<f32>()?);
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.73 checksum saw zero floating variables");
    }
    Ok(sum)
}

fn assert_frozen_checksum(label: &str, initial: f64, current: f64) -> Result<()> {
    let delta = (current - initial).abs();
    let tolerance = 1.0e-6 * initial.abs().max(1.0);
    if delta > tolerance {
        anyhow::bail!(
            "v0.73 frozen {label} changed: initial={initial:.8} current={current:.8} delta={delta:.8}"
        );
    }
    println!(
        "v0730_freeze_audit\tcomponent={label}\tstatus=PASS\tchecksum={current:.8}\tdelta={delta:.8}"
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
        anyhow::bail!("v0.73 requires the selected completed v0.70 epoch6/update6000 checkpoint");
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
        anyhow::bail!("v0.73 requires the completed non-smoke v0.52 parent referenced by v0.70");
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
            "v0.73 requires the authoritative v0.35 forward metadata referenced by v0.52"
        );
    }
    metadata.v0350_config.validate()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn audit_forward_ms2_consistency_v073(
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
        anyhow::bail!("v0.73 audit requires at least one query");
    }
    let mut baseline_ranks = Vec::<(usize, usize)>::with_capacity(query_indices.len());
    let mut ms2_ranks = Vec::<(usize, usize)>::with_capacity(query_indices.len());
    let mut fusion_ranks = Vec::<(usize, usize)>::with_capacity(query_indices.len());
    let mut diagnostics = Vec::with_capacity(query_indices.len());
    let mut margins = Vec::<f64>::with_capacity(query_indices.len());
    let mut exact_target_scores = Vec::<f64>::with_capacity(query_indices.len());
    let mut best_negative_scores = Vec::<f64>::with_capacity(query_indices.len());
    let mut correlations = Vec::<f64>::with_capacity(query_indices.len());
    let mut all_score_sum = 0.0f64;
    let mut all_score_count = 0usize;
    let mut all_nonzero = 0usize;
    let mut candidate_predictions = 0usize;
    let mut candidate_scored = 0usize;
    let mut baseline_errors = 0usize;
    let mut baseline_correct = 0usize;
    let mut ms2_rescued = 0usize;
    let mut ms2_harmed = 0usize;
    let mut fusion_rescued = 0usize;
    let mut fusion_harmed = 0usize;
    let mut oracle_correct = 0usize;
    let mut exact_target_beats_best_negative = 0usize;

    for (query_slot, &query_index) in query_indices.iter().enumerate() {
        let query = &identities[query_index];
        let query_record = &records[query.record_index];
        let observed_spectrum = FoundationSpectrum::from_training_record(query_record)
            .ok_or_else(|| anyhow::anyhow!("v0.73 query lacks an observed spectrum"))?;
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
            anyhow::bail!("v0.73 forward-MS2 prediction count mismatch");
        }

        let mut scored = Vec::<(usize, f64)>::with_capacity(mass_pool.len());
        for (slot, &candidate_index) in mass_pool.iter().enumerate() {
            let candidate_record = &records[identities[candidate_index].record_index];
            let score = forward_ms2_open_ptm_score_v073(
                &candidate_record.peptidoform,
                &observed_spectrum,
                &predicted[slot],
            )?;
            let core = score.core_cosine;
            if core.is_finite() {
                candidate_scored += 1;
                all_score_sum += core;
                all_score_count += 1;
                all_nonzero += usize::from(core > 0.0);
                scored.push((candidate_index, core));
            }
        }
        if scored.len() != mass_pool.len() {
            anyhow::bail!(
                "v0.73 requires complete forward-MS2 coverage per mass64 pool; scored={} expected={}",
                scored.len(), mass_pool.len()
            );
        }
        let mut ms2_ranked = scored.clone();
        ms2_ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let ms2_ranked_indices = ms2_ranked.iter().map(|row| row.0).collect::<Vec<_>>();
        let ms2_exact_rank = rank_exact(&ms2_ranked_indices, identities, query)?;
        let ms2_il_rank = rank_il(&ms2_ranked_indices, identities, query)?;
        ms2_ranks.push((ms2_exact_rank, ms2_il_rank));

        let baseline_position = baseline_ranked
            .iter()
            .enumerate()
            .map(|(rank, &index)| (index, rank + 1))
            .collect::<BTreeMap<_, _>>();
        let ms2_position = ms2_ranked_indices
            .iter()
            .enumerate()
            .map(|(rank, &index)| (index, rank + 1))
            .collect::<BTreeMap<_, _>>();
        let mut fusion_ranked = mass_pool
            .iter()
            .map(|&index| {
                let rank_sum = baseline_position[&index] + ms2_position[&index];
                (index, rank_sum)
            })
            .collect::<Vec<_>>();
        fusion_ranked.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let fusion_ranked_indices = fusion_ranked.iter().map(|row| row.0).collect::<Vec<_>>();
        let fusion_exact_rank = rank_exact(&fusion_ranked_indices, identities, query)?;
        let fusion_il_rank = rank_il(&fusion_ranked_indices, identities, query)?;
        fusion_ranks.push((fusion_exact_rank, fusion_il_rank));

        let score_map = scored.iter().copied().collect::<BTreeMap<_, _>>();
        let exact_target_core = *score_map
            .get(&query_index)
            .context("v0.73 exact target missing forward-MS2 score")?;
        let best_negative = mass_pool
            .iter()
            .filter(|&&index| identities[index].il_key != query.il_key)
            .map(|index| score_map[index])
            .fold(f64::NEG_INFINITY, f64::max);
        if !exact_target_core.is_finite() || !best_negative.is_finite() {
            anyhow::bail!("v0.73 could not form positive/negative forward-MS2 margin");
        }
        let margin = exact_target_core - best_negative;
        exact_target_scores.push(exact_target_core);
        best_negative_scores.push(best_negative);
        margins.push(margin);
        exact_target_beats_best_negative += usize::from(margin > 0.0);

        let baseline_scores = mass_pool
            .iter()
            .map(|&index| f64::from(similarities[query_index][index]))
            .collect::<Vec<_>>();
        let ms2_scores = mass_pool
            .iter()
            .map(|index| score_map[index])
            .collect::<Vec<_>>();
        let correlation = pearson_v073(&baseline_scores, &ms2_scores);
        correlations.push(correlation);

        let baseline_ok = baseline_il_rank == 1;
        let ms2_ok = ms2_il_rank == 1;
        let fusion_ok = fusion_il_rank == 1;
        if baseline_ok {
            baseline_correct += 1;
            ms2_harmed += usize::from(!ms2_ok);
            fusion_harmed += usize::from(!fusion_ok);
        } else {
            baseline_errors += 1;
            ms2_rescued += usize::from(ms2_ok);
            fusion_rescued += usize::from(fusion_ok);
        }
        oracle_correct += usize::from(baseline_ok || ms2_ok);

        diagnostics.push(V073QueryDiagnostic {
            query_slot,
            identity_index: query_index,
            exact_key: query.exact_key.clone(),
            baseline_exact_rank,
            baseline_il_rank,
            ms2_exact_rank,
            ms2_il_rank,
            fusion_exact_rank,
            fusion_il_rank,
            exact_target_core_cosine: exact_target_core,
            best_il_negative_core_cosine: best_negative,
            exact_target_minus_best_il_negative_margin: margin,
            within_query_pearson: correlation,
            baseline_top_candidate: identities[baseline_ranked[0]].exact_key.clone(),
            ms2_top_candidate: identities[ms2_ranked_indices[0]].exact_key.clone(),
            fusion_top_candidate: identities[fusion_ranked_indices[0]].exact_key.clone(),
        });
    }

    margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
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
            ms2_only: rank_metrics_v073(&ms2_ranks),
            fixed_equal_rank_fusion: rank_metrics_v073(&fusion_ranks),
            oracle_il_top1: oracle_correct as f64 / denom,
            mean_exact_target_core_cosine: mean_v073(&exact_target_scores),
            mean_best_il_negative_core_cosine: mean_v073(&best_negative_scores),
            mean_exact_target_minus_best_il_negative_margin: mean_v073(&margins),
            median_exact_target_minus_best_il_negative_margin: median_sorted_v073(&margins),
            exact_target_beats_best_il_negative_fraction: exact_target_beats_best_negative as f64
                / denom,
            mean_within_query_v070_ms2_pearson: mean_v073(&correlations),
            baseline_il_top1_errors: baseline_errors,
            ms2_rescued_baseline_errors: ms2_rescued,
            ms2_rescue_fraction_of_baseline_errors: safe_fraction_v073(
                ms2_rescued,
                baseline_errors,
            ),
            baseline_il_top1_correct: baseline_correct,
            ms2_harmed_baseline_correct: ms2_harmed,
            ms2_harm_fraction_of_baseline_correct: safe_fraction_v073(ms2_harmed, baseline_correct),
            fusion_rescued_baseline_errors: fusion_rescued,
            fusion_rescue_fraction_of_baseline_errors: safe_fraction_v073(
                fusion_rescued,
                baseline_errors,
            ),
            fusion_harmed_baseline_correct: fusion_harmed,
            fusion_harm_fraction_of_baseline_correct: safe_fraction_v073(
                fusion_harmed,
                baseline_correct,
            ),
            mean_core_cosine_all_candidates: if all_score_count == 0 {
                0.0
            } else {
                all_score_sum / all_score_count as f64
            },
            nonzero_core_cosine_fraction: safe_fraction_v073(all_nonzero, all_score_count),
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
        anyhow::bail!("v0.73 candidate charge must be positive");
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

fn forward_ms2_open_ptm_score_v073(
    peptide: &redeem_properties::foundation::PeptidoformInput,
    spectrum: &FoundationSpectrum,
    predicted_ms2: &[Vec<f32>],
) -> Result<V073ForwardMs2Score> {
    let geometry = foundation_fragment_cleavage_geometry(peptide).map_err(anyhow::Error::msg)?;
    if predicted_ms2.len() < geometry.len() {
        anyhow::bail!(
            "v0.73 predicted MS2 rows {} shorter than candidate cleavage count {}",
            predicted_ms2.len(),
            geometry.len()
        );
    }
    let peaks = normalized_retained_peaks_v073(spectrum);
    let mut predicted_core = Vec::<f64>::with_capacity(geometry.len() * 4);
    let mut observed_core = Vec::<f64>::with_capacity(geometry.len() * 4);
    let mut predicted_cleavage = Vec::<f64>::with_capacity(geometry.len());
    let mut observed_cleavage = Vec::<f64>::with_capacity(geometry.len());
    let mut matched_core_ions = 0usize;
    let mut predicted_supported = 0.0f64;
    let mut predicted_sum = 0.0f64;
    for cleavage in &geometry {
        let row = &predicted_ms2[cleavage.cleavage_index];
        if row.len() < 4 {
            anyhow::bail!("v0.73 forward MS2 row has fewer than four core channels");
        }
        let mut cleavage_pred = 0.0f64;
        let mut cleavage_obs = 0.0f64;
        for channel in 0..4 {
            let pred = f64::from(row[channel]).max(0.0);
            let obs = best_peak_support_v073(cleavage.core_mz[channel], &peaks);
            predicted_core.push(pred);
            observed_core.push(obs);
            cleavage_pred += pred;
            cleavage_obs += obs;
            predicted_sum += pred;
            if obs > 0.0 {
                matched_core_ions += 1;
                predicted_supported += pred;
            }
        }
        predicted_cleavage.push(cleavage_pred);
        observed_cleavage.push(cleavage_obs);
    }
    Ok(V073ForwardMs2Score {
        core_cosine: sqrt_intensity_cosine_v073(&predicted_core, &observed_core),
        cleavage_cosine: sqrt_intensity_cosine_v073(&predicted_cleavage, &observed_cleavage),
        matched_core_ions,
        core_ions: predicted_core.len(),
        predicted_supported_fraction: if predicted_sum > 0.0 {
            (predicted_supported / predicted_sum).clamp(0.0, 1.0)
        } else {
            0.0
        },
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
        .context("v0.73 exact target vanished from mass64 candidate pool")
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
        .context("v0.73 I/L target vanished from mass64 candidate pool")
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
        "{label}\tqueries={}\tcandidates_per_query={}\tcandidate_coverage={:.6}\tbaseline_il_top1={:.6}\tbaseline_il_top10={:.6}\tbaseline_il_mrr={:.6}\tms2_il_top1={:.6}\tms2_il_top10={:.6}\tms2_il_mrr={:.6}\tfusion_il_top1={:.6}\tfusion_il_top10={:.6}\tfusion_il_mrr={:.6}\toracle_il_top1={:.6}\tmean_exact_target_core_cosine={:.6}\tmean_best_il_negative_core_cosine={:.6}\tmean_margin={:.6}\tmedian_margin={:.6}\texact_target_beats_best_negative_fraction={:.6}\tmean_v070_ms2_pearson={:.6}\tms2_error_rescue_fraction={:.6}\tms2_harm_fraction={:.6}\tfusion_error_rescue_fraction={:.6}\tfusion_harm_fraction={:.6}\tmean_core_cosine_all={:.6}\tnonzero_core_cosine_fraction={:.6}\telapsed_seconds={:.3}",
        m.queries,
        m.candidates_per_query,
        m.candidate_coverage,
        m.baseline.il_top1,
        m.baseline.il_top10,
        m.baseline.il_mrr,
        m.ms2_only.il_top1,
        m.ms2_only.il_top10,
        m.ms2_only.il_mrr,
        m.fixed_equal_rank_fusion.il_top1,
        m.fixed_equal_rank_fusion.il_top10,
        m.fixed_equal_rank_fusion.il_mrr,
        m.oracle_il_top1,
        m.mean_exact_target_core_cosine,
        m.mean_best_il_negative_core_cosine,
        m.mean_exact_target_minus_best_il_negative_margin,
        m.median_exact_target_minus_best_il_negative_margin,
        m.exact_target_beats_best_il_negative_fraction,
        m.mean_within_query_v070_ms2_pearson,
        m.ms2_rescue_fraction_of_baseline_errors,
        m.ms2_harm_fraction_of_baseline_correct,
        m.fusion_rescue_fraction_of_baseline_errors,
        m.fusion_harm_fraction_of_baseline_correct,
        m.mean_core_cosine_all_candidates,
        m.nonzero_core_cosine_fraction,
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
        ("ms2_il_top1", m.ms2_only.il_top1),
        ("ms2_il_top10", m.ms2_only.il_top10),
        ("ms2_il_mrr", m.ms2_only.il_mrr),
        ("fusion_il_top1", m.fixed_equal_rank_fusion.il_top1),
        ("fusion_il_top10", m.fixed_equal_rank_fusion.il_top10),
        ("fusion_il_mrr", m.fixed_equal_rank_fusion.il_mrr),
        ("oracle_il_top1", m.oracle_il_top1),
        (
            "mean_exact_target_core_cosine",
            m.mean_exact_target_core_cosine,
        ),
        (
            "mean_best_il_negative_core_cosine",
            m.mean_best_il_negative_core_cosine,
        ),
        (
            "mean_exact_target_minus_best_il_negative_margin",
            m.mean_exact_target_minus_best_il_negative_margin,
        ),
        (
            "median_exact_target_minus_best_il_negative_margin",
            m.median_exact_target_minus_best_il_negative_margin,
        ),
        (
            "exact_target_beats_best_il_negative_fraction",
            m.exact_target_beats_best_il_negative_fraction,
        ),
        (
            "mean_within_query_v070_ms2_pearson",
            m.mean_within_query_v070_ms2_pearson,
        ),
        (
            "ms2_rescue_fraction_of_baseline_errors",
            m.ms2_rescue_fraction_of_baseline_errors,
        ),
        (
            "ms2_harm_fraction_of_baseline_correct",
            m.ms2_harm_fraction_of_baseline_correct,
        ),
        (
            "fusion_rescue_fraction_of_baseline_errors",
            m.fusion_rescue_fraction_of_baseline_errors,
        ),
        (
            "fusion_harm_fraction_of_baseline_correct",
            m.fusion_harm_fraction_of_baseline_correct,
        ),
        (
            "mean_core_cosine_all_candidates",
            m.mean_core_cosine_all_candidates,
        ),
        (
            "nonzero_core_cosine_fraction",
            m.nonzero_core_cosine_fraction,
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
    let mut text = String::from("query_slot\tidentity_index\texact_key\tbaseline_exact_rank\tbaseline_il_rank\tms2_exact_rank\tms2_il_rank\tfusion_exact_rank\tfusion_il_rank\texact_target_core_cosine\tbest_il_negative_core_cosine\texact_target_minus_best_il_negative_margin\twithin_query_pearson\tbaseline_top_candidate\tms2_top_candidate\tfusion_top_candidate\n");
    for row in rows {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\n",
            row.query_slot,
            row.identity_index,
            row.exact_key,
            row.baseline_exact_rank,
            row.baseline_il_rank,
            row.ms2_exact_rank,
            row.ms2_il_rank,
            row.fusion_exact_rank,
            row.fusion_il_rank,
            row.exact_target_core_cosine,
            row.best_il_negative_core_cosine,
            row.exact_target_minus_best_il_negative_margin,
            row.within_query_pearson,
            row.baseline_top_candidate,
            row.ms2_top_candidate,
            row.fusion_top_candidate,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod v073_tests {
    use super::*;
    use redeem_properties::foundation::PeptidoformInput;

    #[test]
    fn v073_open_ptm_core_score_rewards_aligned_ions() {
        let peptide = PeptidoformInput::unmodified("AG");
        let spectrum = FoundationSpectrum::from_pairs([(72.0444, 100.0), (76.0393, 80.0)]);
        let predicted = vec![vec![1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let score = forward_ms2_open_ptm_score_v073(&peptide, &spectrum, &predicted).unwrap();
        assert!(score.core_cosine > 0.8, "score={score:?}");
    }

    #[test]
    fn v073_rank_metrics_prioritize_il_equivalence() {
        let ranks = vec![(2, 1), (1, 1), (5, 4)];
        let metrics = rank_metrics_v073(&ranks);
        assert!(metrics.il_top1 > metrics.exact_top1);
        assert!(metrics.il_mrr >= metrics.exact_mrr);
    }

    #[test]
    fn v073_pearson_detects_independent_ordering() {
        let same = pearson_v073(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        let reverse = pearson_v073(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
        assert!((same - 1.0).abs() < 1.0e-12);
        assert!((reverse + 1.0).abs() < 1.0e-12);
    }
}
