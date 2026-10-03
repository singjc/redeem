//! v0.72 retrieval-conditioned spectrum/peptide reranking.
//!
//! Scientific contract:
//! - freeze the selected v0.70 spectrum<->peptide alignment model;
//! - freeze the v0.52 peptide representation used by v0.70;
//! - reproduce the exact v0.70 DEV retrieval cohort and baseline before training;
//! - construct the same mass-64 candidate pool used by v0.70;
//! - train only a zero-residual candidate/spectrum cross-attention adapter over
//!   the v0.70 top-16 candidates inside that mass-64 pool;
//! - optimize listwise I/L-positive compatibility, with an exact-positive term
//!   when an exact target is present in the interaction window;
//! - select/decide on DEV only; never touch TRAIN-HOLDOUT, historical
//!   VALIDATION/APD, or historical TEST.
//!
//! This experiment tests whether fine-grained candidate/residue <-> peak
//! interaction can improve the already-proven v0.70 retrieval ranking.  It is
//! not an autoregressive decoder and it does not generate new peptide syntax.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{ops, VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_peptidoform_neutral_mass, foundation_precursor_neutral_mass, load_foundation_corpus,
    read_foundation_training_run_config, FoundationAdamW, FoundationAdamWConfig,
    FoundationBenchmarkManifest, FoundationCollator, FoundationCollatorConfig,
    FoundationCorruptionConfig, FoundationDiffusionConfig, FoundationLearningRateSchedule,
    FoundationPartition, FoundationSpectrum, FoundationSpectrumBatch,
    FoundationSpectrumCandidateInteractionAdapter, FoundationSpectrumCollator,
    FoundationSpectrumEncoder, FoundationTrainingRecord, PeptideFoundationV0520Config,
    PeptideFoundationV0520Model, RetentionTimeObjective, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V072_VERSION: u32 = 720;
const V072_OBJECTIVE: &str =
    "v0720_frozen_v070_mass64_top16_candidate_cross_attention_listwise_reranking";
const V072_ARCHITECTURE: &str =
    "frozen_v070_alignment_plus_frozen_v052_residue_states_zero_residual_cross_attention_reranker";
const V070_OBJECTIVE: &str = "v0700_frozen_v0520_spectrum_peptide_alignment";
const V070_ARCHITECTURE: &str =
    "frozen_v0520_peptide_plus_observed_spectrum_transformer_contrastive_v0700";
const V070_NAMESPACE: &str = "student_v070";
const V072_NAMESPACE: &str = "student_v072";
const V070_TEMPERATURE: f64 = 0.07;
const V070_DEV_IDENTITIES: usize = 2048;
const MASS_POOL: usize = 64;
const INTERACTION_WINDOW: usize = 16;
const TRAIN_QUERY_BATCH: usize = 8;
const SMOKE_UPDATES: usize = 8;
const PROBE_UPDATES: usize = 256;
const SMOKE_DEV_QUERIES: usize = 64;
const PROBE_DEV_QUERIES: usize = 512;
const ADAPTER_HEADS: usize = 8;
const ADAPTER_BOTTLENECK: usize = 160;
const LEARNING_RATE: f64 = 1.0e-4;
const WEIGHT_DECAY: f64 = 1.0e-4;
const MAX_GRADIENT_NORM: f64 = 1.0;
const SEED: u64 = 20_261_072;
const MIN_SELECTION_GAIN: f64 = 0.020;
const MIN_IL_TOP1_GAIN: f64 = 0.015;
const MAX_IL_TOP10_REGRESSION: f64 = 0.005;
const MIN_INTERACTION_IL_COVERAGE: f64 = 0.95;

#[derive(Debug, Clone, Deserialize)]
struct V052ParentMetadata {
    version: u32,
    objective: String,
    architecture: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    v0520_config: PeptideFoundationV0520Config,
    rt_objective: RetentionTimeObjective,
    completed_epochs: usize,
    completed_updates: usize,
    dev_objective: f64,
    smoke_mode: bool,
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
    length: usize,
    modified: bool,
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
struct RerankMetrics {
    queries: usize,
    exact_target_in_mass64: f64,
    il_target_in_mass64: f64,
    exact_target_in_interaction16: f64,
    il_target_in_interaction16: f64,
    baseline_exact_top1: f64,
    baseline_exact_top10: f64,
    baseline_exact_mrr: f64,
    baseline_il_top1: f64,
    baseline_il_top10: f64,
    baseline_il_mrr: f64,
    reranked_exact_top1: f64,
    reranked_exact_top10: f64,
    reranked_exact_mrr: f64,
    reranked_il_top1: f64,
    reranked_il_top10: f64,
    reranked_il_mrr: f64,
    mean_abs_residual: f64,
}

impl RerankMetrics {
    fn baseline_selection(self) -> f64 {
        0.7 * self.baseline_il_top1 + 0.3 * self.baseline_il_mrr
    }

    fn reranked_selection(self) -> f64 {
        0.7 * self.reranked_il_top1 + 0.3 * self.reranked_il_mrr
    }
}

#[derive(Debug, Serialize)]
struct V072Metadata {
    version: u32,
    objective: String,
    architecture: String,
    parent_v070_checkpoint: String,
    parent_v070_completed_epochs: usize,
    parent_v070_completed_updates: usize,
    parent_v070_dev_selection_score: f64,
    parent_v052_checkpoint: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    dev_identity_fingerprint: String,
    mode: String,
    updates: usize,
    train_query_batch: usize,
    mass_pool: usize,
    interaction_window: usize,
    adapter_heads: usize,
    adapter_bottleneck: usize,
    learning_rate: f64,
    weight_decay: f64,
    seed: u64,
    baseline_selection_score: f64,
    reranked_selection_score: f64,
    train_holdout_consumed: bool,
    historical_validation_consumed: bool,
    historical_test_consumed: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        anyhow::bail!(
            "usage: foundation_train_retrieval_reranker_v0720 RUN_V0260.yaml OUTPUT_DIR PARENT_V070_BEST mode=smoke|probe"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let parent_v070_checkpoint = PathBuf::from(&args[3]);
    let mode = args[4].as_str();
    if !matches!(mode, "smoke" | "probe") {
        anyhow::bail!("v0.72 mode must be smoke or probe");
    }
    if output_root.exists() {
        anyhow::bail!("v0.72 output directory must be fresh: {output_root:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.72 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let updates = if mode == "smoke" {
        SMOKE_UPDATES
    } else {
        PROBE_UPDATES
    };
    let eval_queries = if mode == "smoke" {
        SMOKE_DEV_QUERIES
    } else {
        PROBE_DEV_QUERIES
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
        anyhow::bail!("v0.72 v0.70 parent provenance differs from current corpus/benchmark");
    }

    let parent_v052_checkpoint = PathBuf::from(&parent_v070_metadata.parent_v052_checkpoint);
    let parent_v052_metadata = read_v052_metadata(&parent_v052_checkpoint)?;
    validate_v052_parent(&parent_v052_metadata)?;
    if parent_v052_metadata.corpus_fingerprint != current_corpus_fingerprint
        || parent_v052_metadata.benchmark_manifest_fingerprint != current_benchmark_fingerprint
    {
        anyhow::bail!("v0.72 v0.52 parent provenance differs from current corpus/benchmark");
    }

    let max_sequence_len = parent_v052_metadata
        .v0520_config
        .base_v0510
        .base_v0500
        .max_sequence_len;
    let train_groups = build_alignment_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        max_sequence_len,
    )?;
    let dev_groups = build_alignment_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        max_sequence_len,
    )?;
    if dev_groups.len() < V070_DEV_IDENTITIES || train_groups.len() < 10_000 {
        anyhow::bail!(
            "v0.72 requires the established large TRAIN/DEV retrieval populations; train={} dev={}",
            train_groups.len(),
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
            "v0.72 expected {V070_DEV_IDENTITIES} v0.70 DEV identities, observed {}",
            dev_identities.len()
        );
    }
    let dev_identity_fingerprint =
        format!("fnv1a64:{:016x}", identity_fingerprint(&dev_identities));
    if dev_identity_fingerprint != parent_v070_metadata.dev_identity_fingerprint {
        anyhow::bail!(
            "v0.72 DEV identity fingerprint differs from selected v0.70 parent: current={} parent={}",
            dev_identity_fingerprint,
            parent_v070_metadata.dev_identity_fingerprint
        );
    }

    let train_identities =
        select_all_identities(&corpus.records, &train_groups, SEED ^ 0x7200_7a11_0000_0001)?;
    let mut train_mass_order = (0..train_identities.len()).collect::<Vec<_>>();
    train_mass_order.sort_by(|&a, &b| {
        train_identities[a]
            .candidate_neutral_mass
            .partial_cmp(&train_identities[b].candidate_neutral_mass)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

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

    let mut reranker_varmap = VarMap::new();
    let reranker = FoundationSpectrumCandidateInteractionAdapter::new(
        parent_v070_metadata.config.spectrum.model_dim,
        ADAPTER_HEADS,
        ADAPTER_BOTTLENECK,
        VarBuilder::from_varmap(&reranker_varmap, DType::F32, &device).pp(V072_NAMESPACE),
    )?;
    let mut optimizer = FoundationAdamW::new(
        &reranker_varmap,
        FoundationAdamWConfig {
            learning_rate: LEARNING_RATE,
            beta1: run.trainer.adam_beta1,
            beta2: run.trainer.adam_beta2,
            epsilon: run.trainer.adam_epsilon,
            weight_decay: WEIGHT_DECAY,
        },
    )?;
    let lr_schedule = FoundationLearningRateSchedule::WarmupCosine {
        warmup_steps: 64u64.min(updates.saturating_sub(1) as u64),
        total_steps: updates as u64,
        min_lr_ratio: 0.10,
    };

    let v052_checksum_initial = varmap_checksum(&v052_varmap)?;
    let v070_checksum_initial = varmap_checksum(&v070_varmap)?;

    println!("v0720_version\tv0.72-retrieval-conditioned-reranker");
    println!("objective\t{V072_OBJECTIVE}");
    println!("architecture\t{V072_ARCHITECTURE}");
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
    println!("parent_update_policy\tfrozen_v070_and_frozen_v052_separate_varmaps");
    println!("reranker_update_policy\tstudent_v072_cross_attention_adapter_only");
    println!(
        "reranker_optimizer_variables\t{}",
        optimizer.variable_count()
    );
    println!("candidate_policy\tv070_mass64_then_v070_top16_interaction");
    println!("base_score\tv070_normalized_cosine_div_temperature_0.07");
    println!("residual_initialization\texact_zero_residual_head");
    println!("train_objective\thierarchical_il_then_exact_listwise_positive_set");
    println!("mass_pool\t{MASS_POOL}");
    println!("interaction_window\t{INTERACTION_WINDOW}");
    println!("train_query_batch\t{TRAIN_QUERY_BATCH}");
    println!("updates\t{updates}");
    println!("learning_rate\t{LEARNING_RATE}");
    println!(
        "train_eligible_unique_identities\t{}",
        train_identities.len()
    );
    println!("dev_eligible_unique_identities\t{}", dev_groups.len());
    println!("v070_dev_identity_count\t{}", dev_identities.len());
    println!("v070_dev_identity_fingerprint\t{dev_identity_fingerprint}");
    println!("eval_queries\t{eval_queries}");
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
    print_parent_retrieval("v0720_parent_v070_reproduction", parent_baseline);
    let reproduction_delta =
        (parent_baseline.selection_score() - parent_v070_metadata.dev_selection_score).abs();
    if reproduction_delta > 1.0e-5 {
        anyhow::bail!(
            "v0.72 failed v0.70 parent reproduction: current={:.8} parent={:.8} delta={:.8}",
            parent_baseline.selection_score(),
            parent_v070_metadata.dev_selection_score,
            reproduction_delta
        );
    }
    println!("v0720_parent_reproduction_gate\tPASS\tdelta={reproduction_delta:.8}");

    let eval_query_indices =
        deterministic_eval_queries(&dev_identities, eval_queries, SEED ^ 0x7200_d3f0_0000_0001);
    let initial_metrics = evaluate_reranker(
        &v052,
        &v070,
        &reranker,
        &collator,
        &spectrum_collator,
        &corpus.records,
        &dev_identities,
        &similarities,
        &eval_query_indices,
        &device,
    )?;
    assert_zero_residual_parity(initial_metrics)?;
    fs::create_dir_all(&output_root)?;
    print_rerank("v0720_dev_initial", 0, initial_metrics);
    save_snapshot(
        &output_root.join("model/initial"),
        &reranker_varmap,
        &metadata(
            mode,
            0,
            &parent_v070_checkpoint,
            &parent_v070_metadata,
            &parent_v052_checkpoint,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &dev_identity_fingerprint,
            initial_metrics,
        ),
        initial_metrics,
    )?;

    let mut supervised_groups_total = 0usize;
    let mut skipped_no_il_top16 = 0usize;
    let mut natural_positive_mass64 = 0usize;
    for update in 1..=updates {
        let lr = lr_schedule.learning_rate(LEARNING_RATE, update.saturating_sub(1) as u64)?;
        optimizer.set_learning_rate(lr)?;
        let mut losses = Vec::<Tensor>::new();
        let mut attempts = 0usize;
        let mut slot = 0usize;
        while losses.len() < TRAIN_QUERY_BATCH && attempts < TRAIN_QUERY_BATCH * 32 {
            let query_index = (mix64(
                SEED ^ (update as u64).rotate_left(17)
                    ^ (slot as u64).rotate_left(37)
                    ^ (attempts as u64),
            ) as usize)
                % train_identities.len();
            attempts += 1;
            slot += 1;
            let (mass_pool, natural_positive) =
                training_mass_pool(&train_identities, &train_mass_order, query_index, MASS_POOL);
            natural_positive_mass64 += usize::from(natural_positive);
            match training_group_losses(
                &v052,
                &v070,
                &reranker,
                &collator,
                &spectrum_collator,
                &corpus.records,
                &train_identities,
                query_index,
                &mass_pool,
                &device,
            )? {
                Some(group_losses) => {
                    supervised_groups_total += 1;
                    losses.extend(group_losses);
                }
                None => skipped_no_il_top16 += 1,
            }
        }
        if losses.is_empty() {
            anyhow::bail!("v0.72 update {update} found no supervised top-16 TRAIN query");
        }
        let batch_loss = Tensor::stack(&losses, 0)?.mean_all()?;
        let loss_value = f64::from(batch_loss.to_scalar::<f32>()?);
        let step = optimizer.backward_step(&batch_loss, Some(MAX_GRADIENT_NORM))?;
        if update <= 4 || update % 50 == 0 || update == updates {
            println!(
                "v0720_train\tupdate={update}\tlr={:.8}\tloss={loss_value:.6}\tobjective_terms={}\tgradient_norm={:.6}\tgradient_scale={:.6}\tsupervised_groups_total={supervised_groups_total}\tskipped_no_il_top16={skipped_no_il_top16}",
                step.learning_rate,
                losses.len(),
                step.gradient_norm,
                step.gradient_scale,
            );
        }
    }

    let v052_checksum = varmap_checksum(&v052_varmap)?;
    let v070_checksum = varmap_checksum(&v070_varmap)?;
    assert_frozen_checksum("v052_peptide_parent", v052_checksum_initial, v052_checksum)?;
    assert_frozen_checksum(
        "v070_alignment_parent",
        v070_checksum_initial,
        v070_checksum,
    )?;

    let final_metrics = evaluate_reranker(
        &v052,
        &v070,
        &reranker,
        &collator,
        &spectrum_collator,
        &corpus.records,
        &dev_identities,
        &similarities,
        &eval_query_indices,
        &device,
    )?;
    print_rerank("v0720_dev_final", updates, final_metrics);
    save_snapshot(
        &output_root.join("model/final"),
        &reranker_varmap,
        &metadata(
            mode,
            updates,
            &parent_v070_checkpoint,
            &parent_v070_metadata,
            &parent_v052_checkpoint,
            &current_corpus_fingerprint,
            &current_benchmark_fingerprint,
            &dev_identity_fingerprint,
            final_metrics,
        ),
        final_metrics,
    )?;

    let selection_gain = final_metrics.reranked_selection() - initial_metrics.baseline_selection();
    let il_top1_gain = final_metrics.reranked_il_top1 - initial_metrics.baseline_il_top1;
    let il_top10_delta = final_metrics.reranked_il_top10 - initial_metrics.baseline_il_top10;
    let selection_pass = selection_gain >= MIN_SELECTION_GAIN;
    let top1_pass = il_top1_gain >= MIN_IL_TOP1_GAIN;
    let top10_pass = il_top10_delta >= -MAX_IL_TOP10_REGRESSION;
    let coverage_pass = final_metrics.il_target_in_interaction16 >= MIN_INTERACTION_IL_COVERAGE;

    println!("v0720_train_supervised_groups_total\t{supervised_groups_total}");
    println!("v0720_train_skipped_no_il_top16\t{skipped_no_il_top16}");
    println!("v0720_train_natural_positive_mass64_events\t{natural_positive_mass64}");
    println!("v0720_selection_gain\t{selection_gain:.8}");
    println!("v0720_il_top1_gain\t{il_top1_gain:.8}");
    println!("v0720_il_top10_delta\t{il_top10_delta:.8}");
    println!(
        "v0720_gate_selection_gain_ge_0_02\t{}",
        pass_fail(selection_pass)
    );
    println!("v0720_gate_il_top1_gain_ge_0_015\t{}", pass_fail(top1_pass));
    println!(
        "v0720_gate_il_top10_no_regression_gt_0_005\t{}",
        pass_fail(top10_pass)
    );
    println!(
        "v0720_gate_interaction_il_coverage_ge_0_95\t{}",
        pass_fail(coverage_pass)
    );
    if mode == "smoke" {
        println!("v0720_probe_decision\tSMOKE_MECHANICAL_ONLY");
    } else if selection_pass && top1_pass && top10_pass && coverage_pass {
        println!("v0720_probe_decision\tPROMOTE_RERANKER_FOR_FULL_DEV_CONFIRMATION");
    } else {
        println!("v0720_probe_decision\tCLOSE_RERANKER_NO_MATERIAL_DEV_GAIN");
    }
    println!("train_holdout_consumed\tNO");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn metadata(
    mode: &str,
    updates: usize,
    parent_v070_checkpoint: &Path,
    parent_v070_metadata: &V070ParentMetadata,
    parent_v052_checkpoint: &Path,
    corpus_fingerprint: &str,
    benchmark_fingerprint: &str,
    dev_identity_fingerprint: &str,
    metrics: RerankMetrics,
) -> V072Metadata {
    V072Metadata {
        version: V072_VERSION,
        objective: V072_OBJECTIVE.into(),
        architecture: V072_ARCHITECTURE.into(),
        parent_v070_checkpoint: parent_v070_checkpoint.display().to_string(),
        parent_v070_completed_epochs: parent_v070_metadata.completed_epochs,
        parent_v070_completed_updates: parent_v070_metadata.completed_updates,
        parent_v070_dev_selection_score: parent_v070_metadata.dev_selection_score,
        parent_v052_checkpoint: parent_v052_checkpoint.display().to_string(),
        corpus_fingerprint: corpus_fingerprint.into(),
        benchmark_manifest_fingerprint: benchmark_fingerprint.into(),
        dev_identity_fingerprint: dev_identity_fingerprint.into(),
        mode: mode.into(),
        updates,
        train_query_batch: TRAIN_QUERY_BATCH,
        mass_pool: MASS_POOL,
        interaction_window: INTERACTION_WINDOW,
        adapter_heads: ADAPTER_HEADS,
        adapter_bottleneck: ADAPTER_BOTTLENECK,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        seed: SEED,
        baseline_selection_score: metrics.baseline_selection(),
        reranked_selection_score: metrics.reranked_selection(),
        train_holdout_consumed: false,
        historical_validation_consumed: false,
        historical_test_consumed: false,
    }
}

fn save_snapshot(
    directory: &Path,
    varmap: &VarMap,
    metadata: &V072Metadata,
    metrics: RerankMetrics,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    let mut text = String::from("metric\tvalue\n");
    macro_rules! row {
        ($name:literal, $value:expr) => {
            text.push_str(&format!("{}\t{:.12}\n", $name, $value));
        };
    }
    text.push_str(&format!("queries\t{}\n", metrics.queries));
    row!("exact_target_in_mass64", metrics.exact_target_in_mass64);
    row!("il_target_in_mass64", metrics.il_target_in_mass64);
    row!(
        "exact_target_in_interaction16",
        metrics.exact_target_in_interaction16
    );
    row!(
        "il_target_in_interaction16",
        metrics.il_target_in_interaction16
    );
    row!("baseline_exact_top1", metrics.baseline_exact_top1);
    row!("baseline_exact_top10", metrics.baseline_exact_top10);
    row!("baseline_exact_mrr", metrics.baseline_exact_mrr);
    row!("baseline_il_top1", metrics.baseline_il_top1);
    row!("baseline_il_top10", metrics.baseline_il_top10);
    row!("baseline_il_mrr", metrics.baseline_il_mrr);
    row!("reranked_exact_top1", metrics.reranked_exact_top1);
    row!("reranked_exact_top10", metrics.reranked_exact_top10);
    row!("reranked_exact_mrr", metrics.reranked_exact_mrr);
    row!("reranked_il_top1", metrics.reranked_il_top1);
    row!("reranked_il_top10", metrics.reranked_il_top10);
    row!("reranked_il_mrr", metrics.reranked_il_mrr);
    row!("mean_abs_residual", metrics.mean_abs_residual);
    row!("baseline_selection_score", metrics.baseline_selection());
    row!("reranked_selection_score", metrics.reranked_selection());
    fs::write(directory.join("dev_metrics.tsv"), text)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn training_group_losses(
    v052: &PeptideFoundationV0520Model,
    v070: &PeptideSpectrumAlignmentV0700,
    reranker: &FoundationSpectrumCandidateInteractionAdapter,
    collator: &FoundationCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
    query_index: usize,
    mass_pool: &[usize],
    device: &Device,
) -> Result<Option<Vec<Tensor>>> {
    let query = &identities[query_index];
    let query_record = records[query.record_index].clone();
    let (peak_memory, peak_mask, query_alignment) =
        encode_query_spectrum(v070, spectrum_collator, &[query_record], device)?;

    let candidate_records = mass_pool
        .iter()
        .map(|&index| records[identities[index].record_index].clone())
        .collect::<Vec<_>>();
    let (residue_hidden, residue_mask, candidate_alignment) =
        encode_candidates(v052, v070, collator, &candidate_records, device)?;
    let baseline = cosine_against_query(&candidate_alignment, &query_alignment)?;
    let baseline_values = baseline.to_vec1::<f32>()?;
    let mut local_order = (0..mass_pool.len()).collect::<Vec<_>>();
    local_order.sort_by(|&a, &b| {
        baseline_values[b]
            .partial_cmp(&baseline_values[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    local_order.truncate(INTERACTION_WINDOW.min(local_order.len()));
    let il_positive = local_order
        .iter()
        .map(|&local| identities[mass_pool[local]].il_key == query.il_key)
        .collect::<Vec<_>>();
    if !il_positive.iter().any(|&value| value) {
        return Ok(None);
    }
    let exact_positive = local_order
        .iter()
        .map(|&local| identities[mass_pool[local]].exact_key == query.exact_key)
        .collect::<Vec<_>>();
    let scores = score_interaction_subset(
        reranker,
        &residue_hidden,
        &residue_mask,
        &peak_memory,
        &peak_mask,
        &baseline,
        &local_order,
        device,
    )?;
    let mut losses = vec![listwise_positive_set_loss(&scores, &il_positive, device)?];
    if exact_positive.iter().any(|&value| value) {
        losses.push(listwise_positive_set_loss(
            &scores,
            &exact_positive,
            device,
        )?);
    }
    Ok(Some(losses))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_reranker(
    v052: &PeptideFoundationV0520Model,
    v070: &PeptideSpectrumAlignmentV0700,
    reranker: &FoundationSpectrumCandidateInteractionAdapter,
    collator: &FoundationCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    records: &[FoundationTrainingRecord],
    identities: &[RerankIdentity],
    similarities: &[Vec<f32>],
    query_indices: &[usize],
    device: &Device,
) -> Result<RerankMetrics> {
    let mut exact_mass = 0usize;
    let mut il_mass = 0usize;
    let mut exact_window = 0usize;
    let mut il_window = 0usize;
    let mut baseline_exact_top1 = 0usize;
    let mut baseline_exact_top10 = 0usize;
    let mut baseline_exact_rr = 0.0f64;
    let mut baseline_il_top1 = 0usize;
    let mut baseline_il_top10 = 0usize;
    let mut baseline_il_rr = 0.0f64;
    let mut reranked_exact_top1 = 0usize;
    let mut reranked_exact_top10 = 0usize;
    let mut reranked_exact_rr = 0.0f64;
    let mut reranked_il_top1 = 0usize;
    let mut reranked_il_top10 = 0usize;
    let mut reranked_il_rr = 0.0f64;
    let mut residual_abs_sum = 0.0f64;
    let mut residual_count = 0usize;

    for &query_index in query_indices {
        let query = &identities[query_index];
        let mut mass_pool = (0..identities.len()).collect::<Vec<_>>();
        mass_pool.sort_by(|&a, &b| {
            (identities[a].candidate_neutral_mass - query.observed_neutral_mass)
                .abs()
                .partial_cmp(
                    &(identities[b].candidate_neutral_mass - query.observed_neutral_mass).abs(),
                )
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        mass_pool.truncate(MASS_POOL.min(mass_pool.len()));
        exact_mass += usize::from(
            mass_pool
                .iter()
                .any(|&index| identities[index].exact_key == query.exact_key),
        );
        il_mass += usize::from(
            mass_pool
                .iter()
                .any(|&index| identities[index].il_key == query.il_key),
        );

        let mut baseline_order = mass_pool.clone();
        baseline_order.sort_by(|&a, &b| {
            similarities[query_index][b]
                .partial_cmp(&similarities[query_index][a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let baseline_exact_rank = baseline_order
            .iter()
            .position(|&index| identities[index].exact_key == query.exact_key)
            .map(|value| value + 1);
        let baseline_il_rank = baseline_order
            .iter()
            .position(|&index| identities[index].il_key == query.il_key)
            .map(|value| value + 1);
        if let Some(rank) = baseline_exact_rank {
            baseline_exact_top1 += usize::from(rank == 1);
            baseline_exact_top10 += usize::from(rank <= 10);
            baseline_exact_rr += 1.0 / rank as f64;
        }
        if let Some(rank) = baseline_il_rank {
            baseline_il_top1 += usize::from(rank == 1);
            baseline_il_top10 += usize::from(rank <= 10);
            baseline_il_rr += 1.0 / rank as f64;
        }

        let interaction = baseline_order
            .iter()
            .take(INTERACTION_WINDOW)
            .copied()
            .collect::<Vec<_>>();
        exact_window += usize::from(
            interaction
                .iter()
                .any(|&index| identities[index].exact_key == query.exact_key),
        );
        il_window += usize::from(
            interaction
                .iter()
                .any(|&index| identities[index].il_key == query.il_key),
        );

        let query_record = records[query.record_index].clone();
        let (peak_memory, peak_mask, _query_alignment) =
            encode_query_spectrum(v070, spectrum_collator, &[query_record], device)?;
        let candidate_records = interaction
            .iter()
            .map(|&index| records[identities[index].record_index].clone())
            .collect::<Vec<_>>();
        let (residue_hidden, residue_mask, _candidate_alignment) =
            encode_candidates(v052, v070, collator, &candidate_records, device)?;
        let residual = reranker.forward(
            &residue_hidden,
            &residue_mask,
            &peak_memory.broadcast_as((
                interaction.len(),
                peak_memory.dim(1)?,
                peak_memory.dim(2)?,
            ))?,
            &peak_mask.broadcast_as((interaction.len(), peak_mask.dim(1)?))?,
        )?;
        let residual_values = residual.to_vec1::<f32>()?;
        residual_abs_sum += residual_values
            .iter()
            .map(|value| f64::from(value.abs()))
            .sum::<f64>();
        residual_count += residual_values.len();

        let mut reranked = mass_pool
            .iter()
            .map(|&candidate| {
                (
                    candidate,
                    f64::from(similarities[query_index][candidate]) / V070_TEMPERATURE,
                )
            })
            .collect::<Vec<_>>();
        for (slot, &candidate) in interaction.iter().enumerate() {
            if let Some(row) = reranked.iter_mut().find(|row| row.0 == candidate) {
                row.1 += f64::from(residual_values[slot]);
            }
        }
        reranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let reranked_exact_rank = reranked
            .iter()
            .position(|(index, _)| identities[*index].exact_key == query.exact_key)
            .map(|value| value + 1);
        let reranked_il_rank = reranked
            .iter()
            .position(|(index, _)| identities[*index].il_key == query.il_key)
            .map(|value| value + 1);
        if let Some(rank) = reranked_exact_rank {
            reranked_exact_top1 += usize::from(rank == 1);
            reranked_exact_top10 += usize::from(rank <= 10);
            reranked_exact_rr += 1.0 / rank as f64;
        }
        if let Some(rank) = reranked_il_rank {
            reranked_il_top1 += usize::from(rank == 1);
            reranked_il_top10 += usize::from(rank <= 10);
            reranked_il_rr += 1.0 / rank as f64;
        }
    }

    let denom = query_indices.len() as f64;
    Ok(RerankMetrics {
        queries: query_indices.len(),
        exact_target_in_mass64: exact_mass as f64 / denom,
        il_target_in_mass64: il_mass as f64 / denom,
        exact_target_in_interaction16: exact_window as f64 / denom,
        il_target_in_interaction16: il_window as f64 / denom,
        baseline_exact_top1: baseline_exact_top1 as f64 / denom,
        baseline_exact_top10: baseline_exact_top10 as f64 / denom,
        baseline_exact_mrr: baseline_exact_rr / denom,
        baseline_il_top1: baseline_il_top1 as f64 / denom,
        baseline_il_top10: baseline_il_top10 as f64 / denom,
        baseline_il_mrr: baseline_il_rr / denom,
        reranked_exact_top1: reranked_exact_top1 as f64 / denom,
        reranked_exact_top10: reranked_exact_top10 as f64 / denom,
        reranked_exact_mrr: reranked_exact_rr / denom,
        reranked_il_top1: reranked_il_top1 as f64 / denom,
        reranked_il_top10: reranked_il_top10 as f64 / denom,
        reranked_il_mrr: reranked_il_rr / denom,
        mean_abs_residual: if residual_count == 0 {
            0.0
        } else {
            residual_abs_sum / residual_count as f64
        },
    })
}

fn assert_zero_residual_parity(metrics: RerankMetrics) -> Result<()> {
    let values = [
        (metrics.baseline_exact_top1, metrics.reranked_exact_top1),
        (metrics.baseline_exact_top10, metrics.reranked_exact_top10),
        (metrics.baseline_exact_mrr, metrics.reranked_exact_mrr),
        (metrics.baseline_il_top1, metrics.reranked_il_top1),
        (metrics.baseline_il_top10, metrics.reranked_il_top10),
        (metrics.baseline_il_mrr, metrics.reranked_il_mrr),
    ];
    if values
        .iter()
        .any(|(left, right)| (left - right).abs() > 1.0e-12)
        || metrics.mean_abs_residual > 1.0e-7
    {
        anyhow::bail!("v0.72 zero-residual initialization failed exact v0.70 ranking parity");
    }
    println!("v0720_zero_residual_parent_parity\tPASS");
    Ok(())
}

fn print_rerank(label: &str, update: usize, m: RerankMetrics) {
    println!(
        "{label}\tupdate={update}\tqueries={}\texact_mass64_coverage={:.6}\til_mass64_coverage={:.6}\texact_top16_coverage={:.6}\til_top16_coverage={:.6}\tbaseline_exact_top1={:.6}\tbaseline_exact_top10={:.6}\tbaseline_exact_mrr={:.6}\tbaseline_il_top1={:.6}\tbaseline_il_top10={:.6}\tbaseline_il_mrr={:.6}\treranked_exact_top1={:.6}\treranked_exact_top10={:.6}\treranked_exact_mrr={:.6}\treranked_il_top1={:.6}\treranked_il_top10={:.6}\treranked_il_mrr={:.6}\tmean_abs_residual={:.6}\tbaseline_selection={:.6}\treranked_selection={:.6}",
        m.queries,
        m.exact_target_in_mass64,
        m.il_target_in_mass64,
        m.exact_target_in_interaction16,
        m.il_target_in_interaction16,
        m.baseline_exact_top1,
        m.baseline_exact_top10,
        m.baseline_exact_mrr,
        m.baseline_il_top1,
        m.baseline_il_top10,
        m.baseline_il_mrr,
        m.reranked_exact_top1,
        m.reranked_exact_top10,
        m.reranked_exact_mrr,
        m.reranked_il_top1,
        m.reranked_il_top10,
        m.reranked_il_mrr,
        m.mean_abs_residual,
        m.baseline_selection(),
        m.reranked_selection(),
    );
}

fn score_interaction_subset(
    reranker: &FoundationSpectrumCandidateInteractionAdapter,
    residue_hidden: &Tensor,
    residue_mask: &Tensor,
    peak_memory: &Tensor,
    peak_mask: &Tensor,
    baseline: &Tensor,
    subset: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let index = Tensor::from_vec(
        subset.iter().map(|&value| value as u32).collect::<Vec<_>>(),
        subset.len(),
        device,
    )?;
    let candidate_hidden = residue_hidden.index_select(&index, 0)?;
    let candidate_mask = residue_mask.index_select(&index, 0)?;
    let base = baseline
        .index_select(&index, 0)?
        .affine(1.0 / V070_TEMPERATURE, 0.0)?;
    let (_, memory_len, memory_dim) = peak_memory.dims3()?;
    let memory = peak_memory.broadcast_as((subset.len(), memory_len, memory_dim))?;
    let (_, mask_len) = peak_mask.dims2()?;
    let memory_mask = peak_mask.broadcast_as((subset.len(), mask_len))?;
    let residual = reranker.forward(&candidate_hidden, &candidate_mask, &memory, &memory_mask)?;
    Ok((base + residual)?)
}

fn listwise_positive_set_loss(
    scores: &Tensor,
    positive: &[bool],
    device: &Device,
) -> Result<Tensor> {
    let n = scores.dims1()?;
    if positive.len() != n || !positive.iter().any(|&value| value) {
        anyhow::bail!("v0.72 listwise loss requires a positive mask aligned to scores");
    }
    let log_prob = ops::log_softmax(scores, 0)?;
    let prob = log_prob.exp()?;
    let mask_values = positive
        .iter()
        .map(|&value| if value { 1.0f32 } else { 0.0f32 })
        .collect::<Vec<_>>();
    let positive_mass = prob
        .broadcast_mul(&Tensor::from_vec(mask_values, n, device)?)?
        .sum_all()?
        .clamp(1.0e-12, 1.0)?;
    Ok(positive_mass.log()?.affine(-1.0, 0.0)?)
}

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
                .ok_or_else(|| anyhow::anyhow!("v0.72 query record lacks observed spectrum"))
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
            "v0.72 frozen peptide residue width {} != v0.70 spectrum width {}",
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

#[allow(clippy::too_many_arguments)]
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
        anyhow::bail!("v0.72 parent retrieval similarity matrix shape mismatch");
    }
    let n = identities.len();
    if n == 0 {
        anyhow::bail!("v0.72 parent retrieval cohort is empty");
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
            .context("v0.72 exact v0.70 candidate vanished")?;
        let il_rank = ranked
            .iter()
            .position(|&candidate| identities[candidate].il_key == identities[query].il_key)
            .map(|rank| rank + 1)
            .context("v0.72 I/L v0.70 candidate vanished")?;
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
            .context("v0.72 benchmark record index out of range")?;
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
            anyhow::bail!("v0.72 identity grouping collision");
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

fn select_all_identities(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    seed: u64,
) -> Result<Vec<RerankIdentity>> {
    let mut selected = Vec::with_capacity(groups.len());
    for group in groups {
        selected.push(identity_from_group(
            records,
            group,
            seed ^ hash64_str(&group.key),
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
        .context("v0.72 identity group has no records")?;
    let record = &records[record_index];
    let mz = record
        .context
        .precursor_mz
        .context("v0.72 identity record lacks precursor m/z")?;
    let observed_neutral_mass = foundation_precursor_neutral_mass(f64::from(mz), group.charge)
        .map_err(anyhow::Error::msg)?;
    let candidate_neutral_mass =
        foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
    Ok(RerankIdentity {
        record_index,
        exact_key: group.key.clone(),
        il_key: format!("{}|z{}", il_label(&group.peptidoform), group.charge),
        charge: group.charge,
        length: group.sequence.chars().count(),
        modified: !record.peptidoform.modifications.is_empty(),
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
        anyhow::bail!("v0.72 identity cohort contains duplicate exact identities");
    }
    Ok(())
}

fn training_mass_pool(
    identities: &[RerankIdentity],
    sorted_by_mass: &[usize],
    query_index: usize,
    k: usize,
) -> (Vec<usize>, bool) {
    let observed = identities[query_index].observed_neutral_mass;
    let insertion = sorted_by_mass
        .partition_point(|&index| identities[index].candidate_neutral_mass < observed);
    let radius = k.saturating_mul(8).max(k);
    let start = insertion.saturating_sub(radius);
    let end = (insertion + radius).min(sorted_by_mass.len());
    let mut candidates = sorted_by_mass[start..end].to_vec();
    candidates.sort_by(|&a, &b| {
        (identities[a].candidate_neutral_mass - observed)
            .abs()
            .partial_cmp(&(identities[b].candidate_neutral_mass - observed).abs())
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    candidates.dedup();
    candidates.truncate(k.min(candidates.len()));
    let natural_positive = candidates.contains(&query_index);
    if !natural_positive {
        if candidates.len() == k && !candidates.is_empty() {
            candidates.pop();
        }
        candidates.push(query_index);
    }
    (candidates, natural_positive)
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
        .map_err(|_| anyhow::anyhow!("v0.72 VarMap lock poisoned"))?;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for variable in data.values() {
        if variable.dtype().is_float() {
            sum += f64::from(variable.as_tensor().sum_all()?.to_scalar::<f32>()?);
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("v0.72 checksum saw zero floating variables");
    }
    Ok(sum)
}

fn assert_frozen_checksum(label: &str, initial: f64, current: f64) -> Result<()> {
    let delta = (current - initial).abs();
    let tolerance = 1.0e-6 * initial.abs().max(1.0);
    if delta > tolerance {
        anyhow::bail!(
            "v0.72 frozen {label} changed: initial={initial:.8} current={current:.8} delta={delta:.8}"
        );
    }
    println!(
        "v0720_freeze_audit\tcomponent={label}\tstatus=PASS\tchecksum={current:.8}\tdelta={delta:.8}"
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
        anyhow::bail!("v0.72 requires the selected completed v0.70 epoch6/update6000 checkpoint");
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
        anyhow::bail!("v0.72 requires the completed non-smoke v0.52 parent referenced by v0.70");
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn v072_zero_residual_adapter_starts_at_exact_zero() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let adapter = FoundationSpectrumCandidateInteractionAdapter::new(
            8,
            2,
            4,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let candidate = Tensor::ones((3, 5, 8), DType::F32, &device)?;
        let candidate_mask = Tensor::ones((3, 5), DType::F32, &device)?;
        let memory = Tensor::ones((3, 7, 8), DType::F32, &device)?;
        let memory_mask = Tensor::ones((3, 7), DType::F32, &device)?;
        let residual = adapter.forward(&candidate, &candidate_mask, &memory, &memory_mask)?;
        assert!(residual
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.abs() <= 1.0e-7));
        Ok(())
    }

    #[test]
    fn v072_selection_prioritizes_il_top1() {
        let a = RerankMetrics {
            baseline_il_top1: 0.5,
            baseline_il_mrr: 0.6,
            reranked_il_top1: 0.6,
            reranked_il_mrr: 0.6,
            ..Default::default()
        };
        let b = RerankMetrics {
            reranked_il_top1: 0.5,
            reranked_il_mrr: 0.8,
            ..Default::default()
        };
        assert!(a.reranked_selection() > b.reranked_selection());
    }

    #[test]
    fn v072_training_pool_forces_true_identity() {
        let identities = (0..100)
            .map(|index| RerankIdentity {
                record_index: index,
                exact_key: format!("P{index}|z2"),
                il_key: format!("P{index}|z2"),
                charge: 2,
                length: 8,
                modified: false,
                observed_neutral_mass: 0.0,
                candidate_neutral_mass: index as f64,
            })
            .collect::<Vec<_>>();
        let sorted = (0..100).collect::<Vec<_>>();
        let (pool, natural) = training_mass_pool(&identities, &sorted, 99, 16);
        assert!(!natural);
        assert!(pool.contains(&99));
        assert_eq!(pool.len(), 16);
    }
}
