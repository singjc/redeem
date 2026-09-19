//! v0.32 spectrum->peptide candidate-coverage search diagnostic.
//!
//! v0.29 showed that sequence-level reward optimization is downstream of a much
//! more basic bottleneck: many DEV spectra return no hard-mass candidate at all.
//! v0.32 therefore performs no training and consumes no holdout. It evaluates
//! the frozen v0.27 causal decoder under three deterministic DEV-only policies:
//!
//! 1. frozen v0.29 DEV-selection beam width 32;
//! 2. ordinary beam width 64 as a one-off capacity control;
//! 3. width-32 mass-stratified beam retaining a global elite plus diverse
//!    remaining-mass trajectories.
//!
//! This is one search-architecture diagnostic, not a beam-width sweep. The
//! capacity control exists only to distinguish algorithmic diversity from raw
//! search budget.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_diffusion_token_residue, foundation_direct_beam_search,
    foundation_mass_stratified_beam_search, foundation_peptidoform_neutral_mass,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    DirectDecoderBeamCandidate, DirectDecoderBeamConfig, FoundationBenchmarkManifest,
    FoundationCausalCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationPartition, FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    MassStratifiedBeamConfig, PeptideFoundationInverseRewardV0290Config,
    PeptideFoundationInverseRewardV0290Model, PeptideFoundationMultimodalV0270Config,
    PrecursorContextBatch, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_VOCAB_SIZE,
};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const DEFAULT_DEV_RECORDS_V0320: usize = 128;
const BASELINE_BEAM_WIDTH_V0320: usize = 32;
const CAPACITY_BEAM_WIDTH_V0320: usize = 64;
const RETURN_TOP_K_V0320: usize = 16;
const MASS_STRATA_V0320: usize = 8;
const GLOBAL_ELITE_V0320: usize = 8;

#[derive(Debug, Deserialize)]
struct V0270ParentMetadata {
    version: u32,
    objective: String,
    completed_steps: usize,
    v0270_config: PeptideFoundationMultimodalV0270Config,
}

#[derive(Debug, Clone, Copy, Default)]
struct SearchMetrics {
    records: usize,
    literal_top1: usize,
    il_top1: usize,
    literal_top5: usize,
    il_top5: usize,
    literal_top10: usize,
    il_top10: usize,
    literal_topk: usize,
    il_topk: usize,
    returned_candidates: usize,
    zero_candidate_records: usize,
    mass_valid_candidates: usize,
    mass_error_abs_sum: f64,
}

impl SearchMetrics {
    fn rate(self, value: usize) -> f64 {
        if self.records == 0 {
            0.0
        } else {
            value as f64 / self.records as f64
        }
    }

    fn mean_returned(self) -> f64 {
        self.rate(self.returned_candidates)
    }

    fn candidate_coverage(self) -> f64 {
        1.0 - self.rate(self.zero_candidate_records)
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
}

#[derive(Debug, Clone, Copy, Default)]
struct PolicyMetrics {
    full: SearchMetrics,
    legacy64: SearchMetrics,
}

#[derive(Debug, Clone, Copy)]
enum SearchPolicy {
    Baseline32,
    Capacity64,
    MassStratified32,
}

impl SearchPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::Baseline32 => "baseline_beam32",
            Self::Capacity64 => "capacity_control_beam64",
            Self::MassStratified32 => "mass_stratified_beam32",
        }
    }
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 4 || args.len() > 6 {
        anyhow::bail!(
            "usage: foundation_search_coverage_v0320 RUN_V0260.yaml OUTPUT_DIR PARENT_V0270_CHECKPOINT [dev_records=128] [seed=20260929]"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let dev_records = parse_or(&args, 4, DEFAULT_DEV_RECORDS_V0320)?;
    let seed = parse_or(&args, 5, 20_260_929u64)?;
    if dev_records == 0 {
        anyhow::bail!("v0.32 dev_records must be positive");
    }
    // The Slurm harness creates the experiment directory before launching so it
    // can place GPU-monitoring/log files there. Fail closed on the scientific
    // result artifact instead of requiring the directory itself not to exist.
    let detail_path = output_root.join("dev_search_detail.tsv");
    if detail_path.exists() {
        anyhow::bail!("v0.32 search result already exists; refusing to overwrite {detail_path:?}");
    }

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("v0.32 requires CUDA")?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let parent_metadata = read_parent_metadata(&parent_checkpoint)?;
    if parent_metadata.version != 270
        || parent_metadata.objective != "v0270_rt_specialist_contextual_fragment_tokens_openptm32"
    {
        anyhow::bail!(
            "v0.32 requires frozen v0.27 parent; observed version={} objective={:?}",
            parent_metadata.version,
            parent_metadata.objective
        );
    }
    let model_config =
        PeptideFoundationInverseRewardV0290Config::fixed(parent_metadata.v0270_config.clone())?;
    let inverse_config = model_config.inverse().clone();
    let eligible = search_eligible_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &inverse_config,
    );
    if eligible.len() < dev_records {
        anyhow::bail!(
            "v0.32 requested {dev_records} DEV records but only {} are search-eligible",
            eligible.len()
        );
    }
    let dev_indices = deterministic_subset(&eligible, dev_records, seed ^ 0x5644_4556_3032_3930);

    let causal_collator = FoundationCausalCollator::new_open_ptm(inverse_config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(inverse_config.spectrum.clone())?;
    let varmap = VarMap::new();
    let model = PeptideFoundationInverseRewardV0290Model::new(
        model_config,
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
    )?;
    load_exact_v0270(
        &varmap,
        &parent_checkpoint.join("model.safetensors"),
        &device,
    )?;

    fs::create_dir_all(&output_root)?;
    println!("v0320_version\tv0.32-mass-stratified-candidate-coverage");
    println!("objective\tv0320_dev_candidate_coverage_search_diagnostic");
    println!("parent_checkpoint\t{}", parent_checkpoint.display());
    println!(
        "parent_completed_steps\t{}",
        parent_metadata.completed_steps
    );
    println!("model_parameters_updated\t0");
    println!("evaluation_partition\tDEV-selection-only");
    println!("dev_records\t{dev_records}");
    println!("dev_subset_policy\tv0290_deterministic_order_extended_to_requested_n");
    println!(
        "v0290_legacy_dev64_is_prefix\t{}",
        if seed == 20_260_929u64 && dev_records >= 64 {
            "YES"
        } else {
            "NO"
        }
    );
    println!("return_top_k\t{RETURN_TOP_K_V0320}");
    println!("baseline_beam_width\t{BASELINE_BEAM_WIDTH_V0320}");
    println!("capacity_control_beam_width\t{CAPACITY_BEAM_WIDTH_V0320}");
    println!("mass_stratified_beam_width\t{BASELINE_BEAM_WIDTH_V0320}");
    println!("mass_strata\t{MASS_STRATA_V0320}");
    println!("mass_stratified_global_elite\t{GLOBAL_ELITE_V0320}");
    println!("historical_validation_consumed\tNO");
    println!("historical_test_consumed\tNO");
    println!("train_holdout_consumed\tNO");

    let mut detail = BufWriter::new(File::create(&detail_path)?);
    writeln!(
        detail,
        "record_index\tsequence\tpolicy\treturned\tliteral_rank\til_rank\tmass_valid_candidates\tmean_abs_mass_error_da"
    )?;

    let mut results = Vec::new();
    for policy in [
        SearchPolicy::Baseline32,
        SearchPolicy::Capacity64,
        SearchPolicy::MassStratified32,
    ] {
        let metrics = evaluate_policy(
            policy,
            &model,
            &corpus.records,
            &dev_indices,
            &causal_collator,
            &spectrum_collator,
            &inverse_config,
            &device,
            &mut detail,
        )?;
        print_metrics("v0320_search", policy.label(), metrics.full);
        if metrics.legacy64.records == 64 {
            print_metrics("v0320_search_legacy64", policy.label(), metrics.legacy64);
        }
        results.push((policy, metrics));
    }
    detail.flush()?;

    let baseline = results[0].1.full;
    let capacity = results[1].1.full;
    let stratified = results[2].1.full;
    let coverage_delta_vs_baseline =
        stratified.candidate_coverage() - baseline.candidate_coverage();
    let coverage_delta_vs_capacity =
        stratified.candidate_coverage() - capacity.candidate_coverage();
    let il_topk_delta_vs_baseline =
        stratified.rate(stratified.il_topk) - baseline.rate(baseline.il_topk);
    let il_topk_delta_vs_capacity =
        stratified.rate(stratified.il_topk) - capacity.rate(capacity.il_topk);
    println!("v0320_coverage_delta_vs_baseline\t{coverage_delta_vs_baseline:.6}");
    println!("v0320_coverage_delta_vs_capacity_control\t{coverage_delta_vs_capacity:.6}");
    println!("v0320_il_top16_delta_vs_baseline\t{il_topk_delta_vs_baseline:.6}");
    println!("v0320_il_top16_delta_vs_capacity_control\t{il_topk_delta_vs_capacity:.6}");

    // A material search win must improve reference coverage, not merely return
    // more wrong candidates. The primary comparison is same-width beam32 versus
    // mass-stratified beam32. Beam64 is a one-off raw-capacity control only.
    let material = coverage_delta_vs_baseline >= 0.05
        && stratified.il_topk >= baseline.il_topk.saturating_add(2);
    let capacity_only = !material
        && capacity.candidate_coverage() > baseline.candidate_coverage()
        && capacity.il_topk > baseline.il_topk;
    println!(
        "v0320_material_search_gain\t{}",
        if material { "YES" } else { "NO" }
    );
    println!(
        "v0320_capacity_only_gain\t{}",
        if capacity_only { "YES" } else { "NO" }
    );
    println!(
        "v0320_rethink_required\t{}",
        if material { "NO" } else { "YES" }
    );
    println!(
        "v0320_next_action\t{}",
        if material {
            "integrate_mass_stratified_candidates_with_fixed_reranking_before_any_reward_retraining"
        } else {
            "close_beam_retention_lane_and_rethink_inverse_generation_architecture"
        }
    );
    println!("detail_tsv\t{}", detail_path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_policy(
    policy: SearchPolicy,
    model: &PeptideFoundationInverseRewardV0290Model,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    inverse_config: &FoundationDiffusionConfig,
    device: &Device,
    detail: &mut BufWriter<File>,
) -> Result<PolicyMetrics> {
    let vocabulary = FoundationDiffusionVocabulary;
    let mut metrics = PolicyMetrics::default();
    for (position, &index) in indices.iter().enumerate() {
        let record = &records[index];
        let spectrum = FoundationSpectrum::from_training_record(record)
            .ok_or_else(|| anyhow::anyhow!("v0.32 DEV search record lacks spectrum"))?;
        let spectrum_batch = spectrum_collator.collate(&[spectrum], device)?;
        let precursor = precursor_context(&[record], device)?;
        let context = model
            .causal()
            .prepare_context(&spectrum_batch, &precursor, false)?;
        let precursor_mass = record_precursor_mass(record)?;
        let candidates = match policy {
            SearchPolicy::Baseline32 => foundation_direct_beam_search(
                precursor_mass,
                DirectDecoderBeamConfig {
                    beam_width: BASELINE_BEAM_WIDTH_V0320,
                    top_k: RETURN_TOP_K_V0320,
                    mass_tolerance_da: inverse_config.precursor_mass_tolerance_da,
                    max_tokens: inverse_config.max_tokens,
                },
                |prefixes| legacy_next_logits(model, causal_collator, &context, prefixes, device),
            ),
            SearchPolicy::Capacity64 => foundation_direct_beam_search(
                precursor_mass,
                DirectDecoderBeamConfig {
                    beam_width: CAPACITY_BEAM_WIDTH_V0320,
                    top_k: RETURN_TOP_K_V0320,
                    mass_tolerance_da: inverse_config.precursor_mass_tolerance_da,
                    max_tokens: inverse_config.max_tokens,
                },
                |prefixes| legacy_next_logits(model, causal_collator, &context, prefixes, device),
            ),
            SearchPolicy::MassStratified32 => foundation_mass_stratified_beam_search(
                precursor_mass,
                MassStratifiedBeamConfig {
                    beam_width: BASELINE_BEAM_WIDTH_V0320,
                    top_k: RETURN_TOP_K_V0320,
                    mass_tolerance_da: inverse_config.precursor_mass_tolerance_da,
                    max_tokens: inverse_config.max_tokens,
                    mass_strata: MASS_STRATA_V0320,
                    global_elite: GLOBAL_ELITE_V0320,
                },
                |prefixes| legacy_next_logits(model, causal_collator, &context, prefixes, device),
            ),
        }
        .map_err(anyhow::Error::msg)?;
        let decoded = decode_candidates(&vocabulary, &candidates);
        let literal_rank = decoded
            .iter()
            .position(|(p, _)| p.sequence == record.peptidoform.sequence)
            .map(|rank| rank + 1);
        let target_il = il_sequence(&record.peptidoform.sequence);
        let il_rank = decoded
            .iter()
            .position(|(p, _)| il_sequence(&p.sequence) == target_il)
            .map(|rank| rank + 1);
        let mass_valid = decoded
            .iter()
            .filter(|(_, mass_error)| {
                mass_error.abs() <= inverse_config.precursor_mass_tolerance_da
            })
            .count();
        let mean_abs_mass_error = if decoded.is_empty() {
            0.0
        } else {
            decoded
                .iter()
                .map(|(_, mass_error)| mass_error.abs())
                .sum::<f64>()
                / decoded.len() as f64
        };

        let mass_error_abs_sum = decoded
            .iter()
            .map(|(_, mass_error)| mass_error.abs())
            .sum::<f64>();
        accumulate_search_metrics(
            &mut metrics.full,
            decoded.len(),
            literal_rank,
            il_rank,
            mass_valid,
            mass_error_abs_sum,
        );
        if position < 64 {
            accumulate_search_metrics(
                &mut metrics.legacy64,
                decoded.len(),
                literal_rank,
                il_rank,
                mass_valid,
                mass_error_abs_sum,
            );
        }

        writeln!(
            detail,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
            index,
            record.peptidoform.sequence,
            policy.label(),
            decoded.len(),
            literal_rank
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NA".into()),
            il_rank
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NA".into()),
            mass_valid,
            mean_abs_mass_error,
        )?;
    }
    Ok(metrics)
}

fn decode_candidates(
    vocabulary: &FoundationDiffusionVocabulary,
    candidates: &[DirectDecoderBeamCandidate],
) -> Vec<(redeem_properties::foundation::PeptidoformInput, f64)> {
    candidates
        .iter()
        .filter_map(|candidate| {
            vocabulary
                .decode(&candidate.tokens)
                .ok()
                .map(|peptide| (peptide, candidate.mass_error_da))
        })
        .collect()
}

fn accumulate_search_metrics(
    metrics: &mut SearchMetrics,
    returned: usize,
    literal_rank: Option<usize>,
    il_rank: Option<usize>,
    mass_valid: usize,
    mass_error_abs_sum: f64,
) {
    metrics.records += 1;
    metrics.returned_candidates += returned;
    metrics.zero_candidate_records += usize::from(returned == 0);
    metrics.mass_valid_candidates += mass_valid;
    metrics.mass_error_abs_sum += mass_error_abs_sum;
    metrics.literal_top1 += usize::from(literal_rank == Some(1));
    metrics.il_top1 += usize::from(il_rank == Some(1));
    metrics.literal_top5 += usize::from(literal_rank.is_some_and(|rank| rank <= 5));
    metrics.il_top5 += usize::from(il_rank.is_some_and(|rank| rank <= 5));
    metrics.literal_top10 += usize::from(literal_rank.is_some_and(|rank| rank <= 10));
    metrics.il_top10 += usize::from(il_rank.is_some_and(|rank| rank <= 10));
    metrics.literal_topk +=
        usize::from(literal_rank.is_some_and(|rank| rank <= RETURN_TOP_K_V0320));
    metrics.il_topk += usize::from(il_rank.is_some_and(|rank| rank <= RETURN_TOP_K_V0320));
}

fn print_metrics(prefix: &str, label: &str, m: SearchMetrics) {
    println!(
        "{prefix}\tpolicy={label}\trecords={}\tcandidate_coverage={:.6}\tzero_candidate_records={}\tmean_returned={:.3}\tliteral_top1={:.6}\til_top1={:.6}\tliteral_top5={:.6}\til_top5={:.6}\tliteral_top10={:.6}\til_top10={:.6}\tliteral_top16={:.6}\til_top16={:.6}\tmass_valid_fraction={:.6}\tmean_abs_mass_error_da={:.6}",
        m.records,
        m.candidate_coverage(),
        m.zero_candidate_records,
        m.mean_returned(),
        m.rate(m.literal_top1),
        m.rate(m.il_top1),
        m.rate(m.literal_top5),
        m.rate(m.il_top5),
        m.rate(m.literal_top10),
        m.rate(m.il_top10),
        m.rate(m.literal_topk),
        m.rate(m.il_topk),
        m.mass_valid_fraction(),
        m.mean_abs_mass_error(),
    );
}

fn legacy_next_logits(
    model: &PeptideFoundationInverseRewardV0290Model,
    causal_collator: &FoundationCausalCollator,
    context: &redeem_properties::foundation::FoundationCausalContext,
    prefixes: &[Vec<u32>],
    device: &Device,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    let input = causal_collator
        .collate_compact_prefix_rows(prefixes, device)
        .map_err(|e| e.to_string())?;
    let mut rows = model
        .causal()
        .forward_next_t_with_context(&input, context, false)
        .and_then(|tensor| tensor.to_vec2::<f32>())
        .map_err(|e| e.to_string())?;
    for row in &mut rows {
        if row.len() < FOUNDATION_DIFFUSION_VOCAB_SIZE {
            return Err(format!("v0.32 causal row has {} classes", row.len()));
        }
        row.truncate(FOUNDATION_DIFFUSION_VOCAB_SIZE);
        // Match the closed v0.29 pilot domain exactly: unmodified targets only.
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

fn search_eligible_indices(
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
    let charge: Vec<f32> = records
        .iter()
        .map(|r| r.context.charge.unwrap_or(0) as f32)
        .collect();
    let charge_present: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.context.charge.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect();
    let precursor_mz: Vec<f32> = records
        .iter()
        .map(|r| r.context.precursor_mz.unwrap_or(0.0))
        .collect();
    let precursor_mz_present: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.context.precursor_mz.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect();
    let nce: Vec<f32> = records
        .iter()
        .map(|r| r.context.nce.unwrap_or(0.0))
        .collect();
    let nce_present: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.context.nce.is_some() {
                1.0f32
            } else {
                0.0f32
            }
        })
        .collect();
    let b = records.len();
    let context = PrecursorContextBatch {
        charge: Tensor::from_vec(charge, b, device)?,
        charge_present: Tensor::from_vec(charge_present, b, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, b, device)?,
        precursor_mz_present: Tensor::from_vec(precursor_mz_present, b, device)?,
        nce: Tensor::from_vec(nce, b, device)?,
        nce_present: Tensor::from_vec(nce_present, b, device)?,
        instrument_ids: Tensor::zeros(b, DType::U32, device)?,
        instrument_present: Tensor::zeros(b, DType::F32, device)?,
    };
    for (name, tensor) in [
        ("charge", &context.charge),
        ("charge_present", &context.charge_present),
        ("precursor_mz", &context.precursor_mz),
        ("precursor_mz_present", &context.precursor_mz_present),
        ("nce", &context.nce),
        ("nce_present", &context.nce_present),
        ("instrument_present", &context.instrument_present),
    ] {
        if tensor.dtype() != DType::F32 {
            anyhow::bail!(
                "v0.32 precursor context tensor {name} must be F32, observed {:?}",
                tensor.dtype()
            );
        }
    }
    Ok(context)
}

fn deterministic_subset(indices: &[usize], n: usize, seed: u64) -> Vec<usize> {
    let mut keyed = indices
        .iter()
        .copied()
        .map(|index| (mix64(seed ^ index as u64), index))
        .collect::<Vec<_>>();
    keyed.sort_unstable();
    keyed.into_iter().take(n).map(|(_, index)| index).collect()
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

fn load_exact_v0270(varmap: &VarMap, checkpoint: &Path, device: &Device) -> Result<()> {
    let tensors = candle_core::safetensors::load(checkpoint, device)
        .with_context(|| format!("failed to load frozen v0.27 checkpoint {checkpoint:?}"))?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.32 VarMap lock poisoned"))?;
    let model_names = data.keys().cloned().collect::<BTreeSet<_>>();
    let parent_names = tensors.keys().cloned().collect::<BTreeSet<_>>();
    if model_names != parent_names {
        let missing = model_names
            .difference(&parent_names)
            .cloned()
            .collect::<Vec<_>>();
        let extra = parent_names
            .difference(&model_names)
            .cloned()
            .collect::<Vec<_>>();
        anyhow::bail!(
            "v0.32 requires exact v0.27 namespace match; missing={missing:?} extra={extra:?}"
        );
    }
    for (name, variable) in data.iter() {
        let tensor = &tensors[name];
        if tensor.dims() != variable.as_tensor().dims() {
            anyhow::bail!("v0.32 shape mismatch for {name}");
        }
        variable.set(tensor)?;
    }
    Ok(())
}

fn read_parent_metadata(checkpoint: &Path) -> Result<V0270ParentMetadata> {
    let path = checkpoint.join("metadata.yaml");
    serde_yaml::from_str(&fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?)
        .map_err(anyhow::Error::from)
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
