//! v0.23 global proposal selection by learned forward fragmentation intensity.
//!
//! This is deliberately not another candidate-label-trained reranker.  The
//! peptide->MS2 model is the frozen TRAIN-derived unified forward branch.  Each
//! complete mass-valid v0.13.23 proposal is converted into its predicted
//! cleavage-channel spectrum and compared with the measured spectrum using one
//! fixed biochemical likelihood score.  Validation target labels are consulted
//! only after scores/ranks have been frozen for metrics.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_fragment_likelihood_score, load_foundation_corpus, parse_modified_peptide,
    read_foundation_training_run_config, FoundationBenchmarkManifest, FoundationCollator,
    FoundationCollatorConfig, FoundationConfig, FoundationCorruptionConfig,
    FoundationDiffusionConfig, FoundationFragmentLikelihoodScore, FoundationPartition,
    FoundationSpectrum, FoundationTrainingRecord, PeptideFoundationUnifiedModel, PeptidoformInput,
    FOUNDATION_FRAGMENT_LIKELIHOOD_ARCHITECTURE_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_PRIMARY_SCORE_V0230,
};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const VERSION: &str = "v0.23.0";
const FULL_EXPECTED_RECORDS: usize = 125;
const ACCEPTED_LEGACY_LITERAL_TOP1: usize = 24;
const ACCEPTED_LEGACY_IL_TOP1: usize = 38;
const ACCEPTED_ORACLE_LITERAL: usize = 44;
const ACCEPTED_ORACLE_IL: usize = 54;
const PROGRESS_LITERAL: usize = 28;
const PROGRESS_IL: usize = 42;
const MATERIAL_LITERAL: usize = 26;
const MATERIAL_IL: usize = 40;
const CANDIDATE_BATCH: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Smoke,
    Full,
}

impl RunMode {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("full") {
            "smoke" => Ok(Self::Smoke),
            "full" => Ok(Self::Full),
            other => anyhow::bail!("v0.23 mode must be smoke or full, got '{other}'"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Full => "full",
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

struct UnifiedPredictor {
    _varmap: VarMap,
    model: PeptideFoundationUnifiedModel,
    metadata: UnifiedMetadata,
}

impl UnifiedPredictor {
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
            .with_context(|| format!("load unified model {model_path:?}"))?;
        Ok(Self {
            _varmap: varmap,
            model,
            metadata,
        })
    }
}

#[derive(Debug, Clone)]
struct CandidateRow {
    row_index: usize,
    sequence: String,
    modifications: String,
    mass_error_da: f64,
    legacy_score: f64,
    legacy_rank: usize,
    peptidoform_exact: bool,
    sequence_exact: bool,
    il_exact: bool,
}

#[derive(Debug, Clone)]
struct CandidateGroup {
    record_index: usize,
    source_id: String,
    rows: Vec<CandidateRow>,
}

#[derive(Debug, Clone)]
struct ScoredCandidate {
    row: CandidateRow,
    peptide: PeptidoformInput,
    predicted_ms2: Vec<Vec<f32>>,
    score: FoundationFragmentLikelihoodScore,
    forward_rank: usize,
}

#[derive(Debug, Default)]
struct Metrics {
    records: usize,
    oracle_literal: usize,
    oracle_sequence: usize,
    oracle_il: usize,
    legacy_literal_top1: usize,
    legacy_sequence_top1: usize,
    legacy_il_top1: usize,
    forward_literal_top1: usize,
    forward_sequence_top1: usize,
    forward_il_top1: usize,
    global_literal_top1: usize,
    global_sequence_top1: usize,
    global_il_top1: usize,
    global_literal_top5: usize,
    global_il_top5: usize,
    global_literal_top10: usize,
    global_il_top10: usize,
    target_core_cosine_sum: f64,
    target_shuffled_core_cosine_sum: f64,
    target_score_pairs: usize,
    true_rank_improved: usize,
    true_rank_worsened: usize,
    true_rank_tied: usize,
    true_primary_ranks: Vec<usize>,
    true_legacy_ranks: Vec<usize>,
    candidates: usize,
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 4 || args.len() > 5 {
        anyhow::bail!(
            "usage: foundation_score_fragment_likelihood_v0230 RUN.yaml UNIFIED_CHECKPOINT VALIDATION_CANDIDATES.tsv OUTPUT_DIR [smoke|full]"
        );
    }
    let run_yaml = PathBuf::from(&args[0]);
    let unified_checkpoint = PathBuf::from(&args[1]);
    let validation_tsv = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);
    let mode = RunMode::parse(args.get(4).map(String::as_str))?;
    reject_test_path(&validation_tsv)?;

    let started = Instant::now();
    stage(started, "start");
    println!("version\t{VERSION}");
    println!("mode\t{}", mode.as_str());
    println!("architecture\t{FOUNDATION_FRAGMENT_LIKELIHOOD_ARCHITECTURE_V0230}");
    println!("primary_score\t{FOUNDATION_FRAGMENT_LIKELIHOOD_PRIMARY_SCORE_V0230}");
    println!("candidate_labels_used_for_scoring\tfalse");
    println!("candidate_reranker_training\tNONE");
    println!("synthetic_corruption\tNONE");
    println!("forward_ms2_context\tfull_peptide_transformer_left_right_cleavage_embeddings_plus_charge_nce_instrument");
    println!("global_energy_fusion\tfixed_equal_rank_sum_no_validation_fitted_weight");
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

    let mut groups = read_candidate_groups(&validation_tsv)?;
    verify_validation_partition(&groups, &benchmark)?;
    if mode == RunMode::Smoke {
        groups.truncate(2);
    }
    if groups.is_empty() {
        anyhow::bail!("v0.23 candidate input contains no mass-valid validation groups");
    }
    stage(started, "validation_candidates_ready");

    let device = Device::cuda_if_available(0)?;
    println!("device\t{device:?}");
    let predictor = UnifiedPredictor::load(&unified_checkpoint, &device)?;
    verify_checkpoint_fingerprints(&predictor.metadata, &corpus, &benchmark)?;
    if predictor.metadata.forward_config.ms2_fragment_channels < 4 {
        anyhow::bail!("v0.23 requires at least four forward MS2 channels");
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
    println!("candidate_batch\t{CANDIDATE_BATCH}");
    stage(started, "forward_model_loaded");

    fs::create_dir_all(&output_dir)?;
    let score_path = output_dir.join(format!("fragment_likelihood_{}.tsv", mode.as_str()));
    let mut score_writer = BufWriter::new(File::create(&score_path)?);
    writeln!(
        score_writer,
        "record_index\tsource_id\tglobal_rank\tforward_rank\tlegacy_rank\trank_sum\tcandidate_sequence\tcandidate_modifications\tcore_cosine\tall_channel_cosine\tcleavage_cosine\tmatched_core_ions\tcore_ions\tpredicted_supported_fraction\tmass_error_da\tpeptidoform_exact\tsequence_exact\til_sequence_exact"
    )?;

    let raw_spectrum_records = groups
        .iter()
        .filter(|group| {
            !corpus.records[group.record_index]
                .observed_spectrum_peaks
                .is_empty()
        })
        .count();
    println!("raw_observed_spectrum_records\t{raw_spectrum_records}");
    println!(
        "annotated_fragment_spectrum_fallback_records\t{}",
        groups.len().saturating_sub(raw_spectrum_records)
    );

    let spectra = groups
        .iter()
        .map(|group| {
            corpus
                .records
                .get(group.record_index)
                .with_context(|| {
                    format!("candidate record index {} out of range", group.record_index)
                })
                .and_then(|record| {
                    FoundationSpectrum::from_training_record(record).with_context(|| {
                        format!(
                            "validation record {} has no observed spectrum",
                            group.record_index
                        )
                    })
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut metrics = Metrics::default();
    for (group_index, group) in groups.iter().enumerate() {
        if group_index % 10 == 0 || group_index + 1 == groups.len() {
            println!(
                "v0230_scoring_progress\trecord={}/{}",
                group_index + 1,
                groups.len()
            );
        }
        let original_record = &corpus.records[group.record_index];
        let spectrum = &spectra[group_index];
        let shuffled_spectrum = &spectra[(group_index + 1) % spectra.len()];
        let mut scored = Vec::<ScoredCandidate>::with_capacity(group.rows.len());

        for chunk in group.rows.chunks(CANDIDATE_BATCH) {
            let mut candidate_records = Vec::<FoundationTrainingRecord>::with_capacity(chunk.len());
            let mut peptides = Vec::<PeptidoformInput>::with_capacity(chunk.len());
            for row in chunk {
                let peptide = exported_peptidoform(row)?;
                let mut record = original_record.clone();
                record.peptidoform = peptide.clone();
                // Do not feed target-specific property labels for a different candidate.
                record.retention_time = Default::default();
                record.ccs = None;
                record.fragments.clear();
                candidate_records.push(record);
                peptides.push(peptide);
            }
            let batch = collator.collate(
                &candidate_records,
                &device,
                20260923 ^ group.record_index as u64,
            )?;
            let output =
                predictor
                    .model
                    .forward()
                    .forward_t(&batch.input, &batch.context, false)?;
            let predicted = output.ms2.to_vec3::<f32>()?;
            for ((row, peptide), prediction) in chunk.iter().zip(peptides).zip(predicted) {
                let score = foundation_fragment_likelihood_score(&peptide, spectrum, &prediction)
                    .map_err(anyhow::Error::msg)?;
                scored.push(ScoredCandidate {
                    row: row.clone(),
                    peptide,
                    predicted_ms2: prediction,
                    score,
                    forward_rank: usize::MAX,
                });
            }
        }

        if scored.is_empty() {
            anyhow::bail!(
                "validation group {} produced no scored candidates",
                group.record_index
            );
        }
        metrics.records += 1;
        metrics.candidates += scored.len();
        metrics.oracle_literal += usize::from(scored.iter().any(|row| row.row.peptidoform_exact));
        metrics.oracle_sequence += usize::from(scored.iter().any(|row| row.row.sequence_exact));
        metrics.oracle_il += usize::from(scored.iter().any(|row| row.row.il_exact));

        let legacy_top = scored
            .iter()
            .min_by(|left, right| {
                left.row
                    .legacy_rank
                    .cmp(&right.row.legacy_rank)
                    .then_with(|| right.row.legacy_score.total_cmp(&left.row.legacy_score))
                    .then_with(|| left.row.row_index.cmp(&right.row.row_index))
            })
            .context("legacy top1 missing")?;
        metrics.legacy_literal_top1 += usize::from(legacy_top.row.peptidoform_exact);
        metrics.legacy_sequence_top1 += usize::from(legacy_top.row.sequence_exact);
        metrics.legacy_il_top1 += usize::from(legacy_top.row.il_exact);

        // Frozen v0.23 biochemical likelihood rank.  This is label-free and
        // uses only the frozen forward peptide->MS2 model plus the measured spectrum.
        scored.sort_by(forward_order);
        for (rank, candidate) in scored.iter_mut().enumerate() {
            candidate.forward_rank = rank + 1;
        }
        let forward_top = &scored[0];
        metrics.forward_literal_top1 += usize::from(forward_top.row.peptidoform_exact);
        metrics.forward_sequence_top1 += usize::from(forward_top.row.sequence_exact);
        metrics.forward_il_top1 += usize::from(forward_top.row.il_exact);

        // Primary global competition is one predeclared Borda/rank-sum product
        // of experts: the accepted v0.13.23 mass-aware rank acts as the proposal
        // prior and the forward fragmentation-intensity rank as biochemical
        // likelihood.  No validation labels or fitted fusion weight are used.
        scored.sort_by(global_order);
        let top = &scored[0];
        metrics.global_literal_top1 += usize::from(top.row.peptidoform_exact);
        metrics.global_sequence_top1 += usize::from(top.row.sequence_exact);
        metrics.global_il_top1 += usize::from(top.row.il_exact);
        metrics.global_literal_top5 +=
            usize::from(scored.iter().take(5).any(|row| row.row.peptidoform_exact));
        metrics.global_il_top5 += usize::from(scored.iter().take(5).any(|row| row.row.il_exact));
        metrics.global_literal_top10 +=
            usize::from(scored.iter().take(10).any(|row| row.row.peptidoform_exact));
        metrics.global_il_top10 += usize::from(scored.iter().take(10).any(|row| row.row.il_exact));

        if let Some((rank, target)) = scored
            .iter()
            .enumerate()
            .find(|(_, row)| row.row.peptidoform_exact)
        {
            let primary_rank = rank + 1;
            metrics.true_primary_ranks.push(primary_rank);
            metrics.true_legacy_ranks.push(target.row.legacy_rank);
            match primary_rank.cmp(&target.row.legacy_rank) {
                std::cmp::Ordering::Less => metrics.true_rank_improved += 1,
                std::cmp::Ordering::Greater => metrics.true_rank_worsened += 1,
                std::cmp::Ordering::Equal => metrics.true_rank_tied += 1,
            }
            metrics.target_core_cosine_sum += target.score.core_cosine;
            let shuffled = foundation_fragment_likelihood_score(
                &target.peptide,
                shuffled_spectrum,
                &target.predicted_ms2,
            )
            .map_err(anyhow::Error::msg)?;
            metrics.target_shuffled_core_cosine_sum += shuffled.core_cosine;
            metrics.target_score_pairs += 1;
        }

        for (rank, candidate) in scored.iter().enumerate() {
            writeln!(
                score_writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{:.8}\t{:.8}\t{}\t{}\t{}",
                group.record_index,
                escape_tsv(&group.source_id),
                rank + 1,
                candidate.forward_rank,
                candidate.row.legacy_rank,
                candidate.forward_rank.saturating_add(candidate.row.legacy_rank),
                escape_tsv(&candidate.row.sequence),
                escape_tsv(&candidate.row.modifications),
                candidate.score.core_cosine,
                candidate.score.all_channel_cosine,
                candidate.score.cleavage_cosine,
                candidate.score.matched_core_ions,
                candidate.score.core_ions,
                candidate.score.predicted_supported_fraction,
                candidate.row.mass_error_da,
                candidate.row.peptidoform_exact,
                candidate.row.sequence_exact,
                candidate.row.il_exact,
            )?;
        }
    }
    score_writer.flush()?;
    stage(started, "candidate_scoring_complete");

    print_metrics(&metrics);
    if mode == RunMode::Full {
        enforce_frozen_contract(&metrics)?;
        let stop = if metrics.global_literal_top1 >= PROGRESS_LITERAL
            && metrics.global_il_top1 >= PROGRESS_IL
        {
            "V0230_PROGRESS_TARGET_MET_GLOBAL_FRAGMENT_ENERGY"
        } else if metrics.global_literal_top1 >= MATERIAL_LITERAL
            && metrics.global_il_top1 >= MATERIAL_IL
        {
            "V0230_PARTIAL_SUCCESS_ALLOW_ONE_PROBABILISTIC_CLEAVAGE_MODEL_UPGRADE"
        } else if metrics.global_literal_top1 >= ACCEPTED_LEGACY_LITERAL_TOP1
            && metrics.global_il_top1 >= ACCEPTED_LEGACY_IL_TOP1
        {
            "V0230_BASELINE_PRESERVED_NO_MATERIAL_GAIN_STOP"
        } else {
            "STOP_FORWARD_FRAGMENT_LIKELIHOOD_REASSESS_GLOBAL_PROPOSAL_ENERGY"
        };
        println!(
            "baseline_recovery_gate\t{}",
            yes_no(
                metrics.global_literal_top1 >= ACCEPTED_LEGACY_LITERAL_TOP1
                    && metrics.global_il_top1 >= ACCEPTED_LEGACY_IL_TOP1
            )
        );
        println!(
            "progress_gate\t{}",
            yes_no(
                metrics.global_literal_top1 >= PROGRESS_LITERAL
                    && metrics.global_il_top1 >= PROGRESS_IL
            )
        );
        println!(
            "material_improvement_threshold\tliteral>={MATERIAL_LITERAL}_and_il>={MATERIAL_IL}"
        );
        println!("v0230_stop_rule\t{stop}");
    } else {
        println!("v0230_stop_rule\tSMOKE_RUNTIME_ONLY_NO_SCIENTIFIC_DECISION");
    }
    println!("test_partition_consumed\tfalse");
    stage(started, "complete");
    Ok(())
}

fn forward_order(left: &ScoredCandidate, right: &ScoredCandidate) -> std::cmp::Ordering {
    right
        .score
        .core_cosine
        .total_cmp(&left.score.core_cosine)
        .then_with(|| {
            right
                .score
                .cleavage_cosine
                .total_cmp(&left.score.cleavage_cosine)
        })
        .then_with(|| {
            right
                .score
                .all_channel_cosine
                .total_cmp(&left.score.all_channel_cosine)
        })
        .then_with(|| {
            left.row
                .mass_error_da
                .abs()
                .total_cmp(&right.row.mass_error_da.abs())
        })
        .then_with(|| left.row.sequence.cmp(&right.row.sequence))
        .then_with(|| left.row.modifications.cmp(&right.row.modifications))
        .then_with(|| left.row.row_index.cmp(&right.row.row_index))
}

fn global_order(left: &ScoredCandidate, right: &ScoredCandidate) -> std::cmp::Ordering {
    let left_sum = left.forward_rank.saturating_add(left.row.legacy_rank);
    let right_sum = right.forward_rank.saturating_add(right.row.legacy_rank);
    left_sum
        .cmp(&right_sum)
        .then_with(|| left.forward_rank.cmp(&right.forward_rank))
        .then_with(|| left.row.legacy_rank.cmp(&right.row.legacy_rank))
        .then_with(|| right.score.core_cosine.total_cmp(&left.score.core_cosine))
        .then_with(|| {
            left.row
                .mass_error_da
                .abs()
                .total_cmp(&right.row.mass_error_da.abs())
        })
        .then_with(|| left.row.sequence.cmp(&right.row.sequence))
        .then_with(|| left.row.modifications.cmp(&right.row.modifications))
        .then_with(|| left.row.row_index.cmp(&right.row.row_index))
}

fn print_metrics(metrics: &Metrics) {
    println!("final_records\t{}", metrics.records);
    println!("candidate_rows_scored\t{}", metrics.candidates);
    println!("proposal_oracle_literal\t{}", metrics.oracle_literal);
    println!("proposal_oracle_sequence\t{}", metrics.oracle_sequence);
    println!("proposal_oracle_il\t{}", metrics.oracle_il);
    println!("legacy_literal_top1\t{}", metrics.legacy_literal_top1);
    println!("legacy_sequence_top1\t{}", metrics.legacy_sequence_top1);
    println!("legacy_il_top1\t{}", metrics.legacy_il_top1);
    println!(
        "forward_intensity_literal_top1\t{}",
        metrics.forward_literal_top1
    );
    println!(
        "forward_intensity_sequence_top1\t{}",
        metrics.forward_sequence_top1
    );
    println!("forward_intensity_il_top1\t{}", metrics.forward_il_top1);
    println!(
        "global_energy_literal_top1\t{}",
        metrics.global_literal_top1
    );
    println!(
        "global_energy_sequence_top1\t{}",
        metrics.global_sequence_top1
    );
    println!("global_energy_il_top1\t{}", metrics.global_il_top1);
    println!(
        "global_energy_literal_top5\t{}",
        metrics.global_literal_top5
    );
    println!("global_energy_il_top5\t{}", metrics.global_il_top5);
    println!(
        "global_energy_literal_top10\t{}",
        metrics.global_literal_top10
    );
    println!("global_energy_il_top10\t{}", metrics.global_il_top10);
    println!(
        "true_candidate_global_rank_improved_vs_legacy\t{}",
        metrics.true_rank_improved
    );
    println!(
        "true_candidate_global_rank_worsened_vs_legacy\t{}",
        metrics.true_rank_worsened
    );
    println!(
        "true_candidate_global_rank_tied_vs_legacy\t{}",
        metrics.true_rank_tied
    );
    println!(
        "true_candidate_global_median_rank\t{}",
        median_usize(&metrics.true_primary_ranks)
    );
    println!(
        "true_candidate_legacy_median_rank\t{}",
        median_usize(&metrics.true_legacy_ranks)
    );
    if metrics.target_score_pairs > 0 {
        let matched = metrics.target_core_cosine_sum / metrics.target_score_pairs as f64;
        let shuffled = metrics.target_shuffled_core_cosine_sum / metrics.target_score_pairs as f64;
        println!("target_forward_core_cosine_matched_mean\t{matched:.8}");
        println!("target_forward_core_cosine_shuffled_mean\t{shuffled:.8}");
        println!(
            "target_forward_core_cosine_conditioning_gap\t{:.8}",
            matched - shuffled
        );
        println!(
            "target_forward_core_cosine_pairs\t{}",
            metrics.target_score_pairs
        );
    }
}

fn enforce_frozen_contract(metrics: &Metrics) -> Result<()> {
    if metrics.records != FULL_EXPECTED_RECORDS {
        anyhow::bail!(
            "v0.23 frozen validation expected {FULL_EXPECTED_RECORDS} records, observed {}",
            metrics.records
        );
    }
    if metrics.legacy_literal_top1 != ACCEPTED_LEGACY_LITERAL_TOP1
        || metrics.legacy_il_top1 != ACCEPTED_LEGACY_IL_TOP1
    {
        anyhow::bail!(
            "v0.23 legacy starting point drift: expected {}/{} literal/I-L, observed {}/{}",
            ACCEPTED_LEGACY_LITERAL_TOP1,
            ACCEPTED_LEGACY_IL_TOP1,
            metrics.legacy_literal_top1,
            metrics.legacy_il_top1
        );
    }
    if metrics.oracle_literal != ACCEPTED_ORACLE_LITERAL || metrics.oracle_il != ACCEPTED_ORACLE_IL
    {
        anyhow::bail!(
            "v0.23 frozen proposal oracle drift: expected {}/{} literal/I-L, observed {}/{}",
            ACCEPTED_ORACLE_LITERAL,
            ACCEPTED_ORACLE_IL,
            metrics.oracle_literal,
            metrics.oracle_il
        );
    }
    Ok(())
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
            "v0.23 unified checkpoint corpus fingerprint {} != loaded {}",
            metadata.corpus_fingerprint,
            corpus_fingerprint
        );
    }
    if metadata.benchmark_manifest_fingerprint != benchmark_fingerprint {
        anyhow::bail!(
            "v0.23 unified checkpoint benchmark fingerprint {} != loaded {}",
            metadata.benchmark_manifest_fingerprint,
            benchmark_fingerprint
        );
    }
    Ok(())
}

fn verify_validation_partition(
    groups: &[CandidateGroup],
    benchmark: &FoundationBenchmarkManifest,
) -> Result<()> {
    let partition = benchmark
        .entries
        .iter()
        .map(|entry| (entry.record_index, entry.partition))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    for group in groups {
        if !seen.insert(group.record_index) {
            anyhow::bail!(
                "v0.23 candidate TSV contains duplicate record group {}",
                group.record_index
            );
        }
        match partition.get(&group.record_index) {
            Some(FoundationPartition::Validation) => {}
            Some(other) => anyhow::bail!(
                "v0.23 candidate record {} is {:?}, not VALIDATION",
                group.record_index,
                other
            ),
            None => anyhow::bail!(
                "v0.23 candidate record {} absent from benchmark",
                group.record_index
            ),
        }
    }
    Ok(())
}

fn read_candidate_groups(path: &Path) -> Result<Vec<CandidateGroup>> {
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
        "source_id",
        "candidate_sequence",
        "candidate_modifications",
        "fragment_causal_mass_rank",
        "fragment_causal_score",
        "mass_error_da",
        "mass_valid",
        "peptidoform_exact",
        "sequence_exact",
        "il_sequence_exact",
    ] {
        if !index.contains_key(required) {
            anyhow::bail!("v0.23 candidate TSV missing required column '{required}'");
        }
    }

    let mut groups = Vec::<CandidateGroup>::new();
    let mut current_id = None::<usize>;
    let mut source_id = String::new();
    let mut rows = Vec::<CandidateRow>::new();
    let mut row_index = 0usize;
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let get = |name: &str| -> Result<&str> {
            let position = *index
                .get(name)
                .context("internal candidate TSV index missing")?;
            fields
                .get(position)
                .copied()
                .with_context(|| format!("candidate row missing column {name}"))
        };
        if !parse_bool(get("mass_valid")?) {
            continue;
        }
        let record_index: usize = get("record_index")?.parse()?;
        if current_id != Some(record_index) {
            if let Some(id) = current_id.take() {
                groups.push(CandidateGroup {
                    record_index: id,
                    source_id: std::mem::take(&mut source_id),
                    rows: std::mem::take(&mut rows),
                });
            }
            current_id = Some(record_index);
            source_id = get("source_id")?.to_string();
        }
        rows.push(CandidateRow {
            row_index,
            sequence: get("candidate_sequence")?.to_string(),
            modifications: get("candidate_modifications")?.to_string(),
            mass_error_da: parse_finite(get("mass_error_da")?, f64::INFINITY),
            legacy_score: parse_finite(get("fragment_causal_score")?, f64::NEG_INFINITY),
            legacy_rank: get("fragment_causal_mass_rank")?
                .parse()
                .unwrap_or(usize::MAX),
            peptidoform_exact: parse_bool(get("peptidoform_exact")?),
            sequence_exact: parse_bool(get("sequence_exact")?),
            il_exact: parse_bool(get("il_sequence_exact")?),
        });
        row_index += 1;
    }
    if let Some(id) = current_id {
        groups.push(CandidateGroup {
            record_index: id,
            source_id,
            rows,
        });
    }
    groups.retain(|group| !group.rows.is_empty());
    Ok(groups)
}

fn exported_peptidoform(row: &CandidateRow) -> Result<PeptidoformInput> {
    if row.modifications.trim().is_empty() {
        return Ok(PeptidoformInput::unmodified(row.sequence.clone()));
    }
    let residues = row.sequence.chars().collect::<Vec<_>>();
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
                anyhow::bail!(
                    "candidate modification site {residue_index} outside sequence '{}'",
                    row.sequence
                );
            }
            residue_mods.entry(residue_index).or_default().push(id);
        } else {
            anyhow::bail!("unsupported exported modification site '{site}'");
        }
    }
    nterm.sort_unstable();
    for mods in residue_mods.values_mut() {
        mods.sort_unstable();
    }
    let mut encoded = String::new();
    for id in nterm {
        encoded.push_str(&format!("[UniMod:{id}]"));
    }
    for (index, residue) in residues.into_iter().enumerate() {
        encoded.push(residue);
        if let Some(mods) = residue_mods.get(&index) {
            for id in mods {
                encoded.push_str(&format!("[UniMod:{id}]"));
            }
        }
    }
    parse_modified_peptide(&encoded)
        .with_context(|| format!("reconstruct v0.23 candidate '{}'", row.sequence))
}

fn reject_test_path(path: &Path) -> Result<()> {
    let name = path.to_string_lossy().to_ascii_lowercase();
    if name.contains("/test/")
        || name.contains("\\test\\")
        || name.contains("test_partition")
        || name.ends_with("/test.tsv")
        || name.ends_with("\\test.tsv")
    {
        anyhow::bail!("v0.23 forbids TEST inputs; suspicious path {path:?}");
    }
    Ok(())
}

fn median_usize(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    values[values.len() / 2]
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

fn parse_finite(value: &str, fallback: f64) -> f64 {
    value
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\n', " ")
        .replace('\r', " ")
}

fn stage(started: Instant, stage: &str) {
    println!(
        "v0230_stage\tstage={stage}\telapsed_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();
}
