//! Generate peptide/PTM candidates from observed spectra with a trained diffusion checkpoint.
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
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_reverse_probabilities,
    foundation_diffusion_token_mass_da, foundation_diffusion_token_residue,
    foundation_fragment_causal_rerank_score, foundation_precursor_mass_error_da,
    foundation_precursor_neutral_mass, load_foundation_corpus, read_foundation_training_run_config,
    FoundationBenchmarkManifest, FoundationCausalCollator, FoundationDiffusionCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationPartition,
    FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideSpectrumCausalModel, PeptideSpectrumDiffusionModel, PeptidoformInput,
    PrecursorContextBatch, FOUNDATION_CAUSAL_RERANK_POLICY_V0123,
    FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct DiffusionCheckpointMetadata {
    diffusion: FoundationDiffusionConfig,
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
    mass_error_da: Option<f64>,
    mass_valid: bool,
    from_diffusion: bool,
    from_causal_beam: bool,
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

struct CausalReranker {
    _varmap: VarMap,
    model: PeptideSpectrumCausalModel,
    collator: FoundationCausalCollator,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 18 {
        anyhow::bail!(
            "usage: foundation_generate_diffusion FOUNDATION_TRAINING.yaml CHECKPOINT_DIR OUTPUT.tsv [validation_records=64] [samples_per_record=16] [seed=20260901] [mass_tolerance_da=0.05] [temperature=1.0] [mass_beam_width=512] [final_candidates_per_chain=4] [fragment_tolerance_ppm=20] [spectral_beam_weight=2.0] [neural_rerank_weight=1.0] [causal_checkpoint=none] [causal_rerank_weight=0.1] [causal_generation_beam_width=0] [causal_generation_final_candidates=16]"
        );
    }

    let training_yaml = &args[1];
    let checkpoint_dir = PathBuf::from(&args[2]);
    let output_tsv = PathBuf::from(&args[3]);
    let validation_records = parse_or(&args, 4, 64usize)?;
    let samples_per_record = parse_or(&args, 5, 16usize)?;
    let seed = parse_or(&args, 6, 20_260_901u64)?;
    let mass_tolerance_da = parse_or(&args, 7, 0.05f64)?;
    let temperature = parse_or(&args, 8, 1.0f64)?;
    let mass_beam_width = parse_or(&args, 9, 512usize)?;
    let final_candidates_per_chain = parse_or(&args, 10, 4usize)?;
    let fragment_tolerance_ppm = parse_or(&args, 11, 20.0f64)?;
    let spectral_beam_weight = parse_or(&args, 12, 2.0f64)?;
    let neural_rerank_weight = parse_or(&args, 13, 1.0f64)?;
    let causal_checkpoint = optional_path(&args, 14);
    let causal_rerank_weight = parse_or(&args, 15, FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123)?;
    let causal_generation_beam_width = parse_or(&args, 16, 0usize)?;
    let causal_generation_final_candidates = parse_or(&args, 17, 16usize)?;
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
    if causal_generation_beam_width > 0 && causal_checkpoint.is_none() {
        anyhow::bail!("causal generation requires a causal checkpoint");
    }

    let device = Device::Cpu;
    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let metadata_path = checkpoint_dir.join("metadata.yaml");
    let checkpoint_metadata: DiffusionCheckpointMetadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read {metadata_path:?}"))?,
    )?;
    let config = checkpoint_metadata.diffusion;
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
        anyhow::bail!("no usable validation diffusion pairs were selected");
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumDiffusionModel::new(config.clone(), vb)?;
    varmap
        .load(checkpoint_dir.join("model.safetensors"))
        .with_context(|| format!("failed to load diffusion checkpoint {checkpoint_dir:?}"))?;

    let diffusion_collator = FoundationDiffusionCollator::new(config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;

    let causal_reranker = causal_checkpoint
        .as_ref()
        .map(|checkpoint| -> Result<CausalReranker> {
            let metadata_path = checkpoint.join("metadata.yaml");
            let metadata: DiffusionCheckpointMetadata =
                serde_yaml::from_str(&fs::read_to_string(&metadata_path).with_context(|| {
                    format!("failed to read causal metadata {metadata_path:?}")
                })?)?;
            if metadata.diffusion != config {
                anyhow::bail!(
                    "causal checkpoint architecture differs from frozen diffusion generator config"
                );
            }
            let mut causal_varmap = VarMap::new();
            let causal_vb = VarBuilder::from_varmap(&causal_varmap, DType::F32, &device);
            let causal_model = PeptideSpectrumCausalModel::new(config.clone(), causal_vb)?;
            causal_varmap
                .load(checkpoint.join("model.safetensors"))
                .with_context(|| format!("failed to load causal checkpoint {checkpoint:?}"))?;
            Ok(CausalReranker {
                _varmap: causal_varmap,
                model: causal_model,
                collator: FoundationCausalCollator::new(config.clone())?,
            })
        })
        .transpose()?;

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
    println!("samples_per_record\t{samples_per_record}");
    println!("diffusion_steps\t{}", config.diffusion_steps);
    println!("mass_tolerance_da\t{mass_tolerance_da}");
    println!("temperature\t{temperature}");
    println!("mass_beam_width\t{mass_beam_width}");
    println!("final_candidates_per_chain\t{final_candidates_per_chain}");
    println!("fragment_tolerance_ppm\t{fragment_tolerance_ppm}");
    println!("spectral_beam_weight\t{spectral_beam_weight}");
    println!("neural_rerank_weight\t{neural_rerank_weight}");
    println!(
        "causal_checkpoint\t{}",
        causal_checkpoint
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".into())
    );
    println!("causal_rerank_weight\t{causal_rerank_weight}");
    println!("causal_rerank_score_definition\tfragment_score+weight*ar_total_log_probability");
    let causal_rerank_policy =
        if (causal_rerank_weight - FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123).abs() <= f64::EPSILON {
            FOUNDATION_CAUSAL_RERANK_POLICY_V0123
        } else {
            "custom_fragment_plus_weighted_ar_total"
        };
    println!("causal_rerank_policy\t{causal_rerank_policy}");
    println!("causal_generation_beam_width\t{causal_generation_beam_width}");
    println!("causal_generation_final_candidates\t{causal_generation_final_candidates}");
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
    println!("primary_candidate_ranking\tfragment_mass");
    println!("parallel_candidate_rankings\tneural_all_mask_mass,hybrid_fragment_neural_mass,causal_ar_mass,hybrid_fragment_causal_mass");
    println!(
        "candidate_pool_sources\t{}",
        if causal_generation_beam_width > 0 {
            "diffusion_reverse_v0115+causal_prefix_mass_beam_v0124"
        } else {
            "diffusion_reverse_v0115"
        }
    );
    println!("candidate_reranker\tall_mask_x0_v1+causal_next_token_v1");
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
        "record_index\ttarget_sequence\ttarget_active_tokens\tpredicted_active_tokens\tfragment_mass_rank\tneural_mass_rank\thybrid_mass_rank\tcausal_mass_rank\tfragment_causal_mass_rank\tmass_rank\treverse_rank\tcandidate_sequence\tcandidate_modifications\treverse_log_probability\tfragment_score\tmatched_cleavages\tneural_all_mask_log_probability\tneural_all_mask_perplexity\tneural_length_log_probability\thybrid_score\tar_total_log_probability\tar_mean_log_probability\tar_perplexity\tfragment_causal_score\tmass_error_da\tmass_valid\tfrom_diffusion\tfrom_causal_beam\tpeptidoform_exact\tsequence_exact\til_sequence_exact"
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
                mass_error_da,
                mass_valid,
                from_diffusion: true,
                from_causal_beam: false,
            };
            unique
                .entry(tokens)
                .and_modify(|existing| {
                    existing.from_diffusion = true;
                    if candidate.reverse_log_probability > existing.reverse_log_probability {
                        let from_causal_beam = existing.from_causal_beam;
                        *existing = candidate.clone();
                        existing.from_causal_beam = from_causal_beam;
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
                        mass_error_da,
                        mass_valid,
                        from_diffusion: false,
                        from_causal_beam: true,
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
        for (fragment_mass_index, candidate) in candidates.iter().enumerate() {
            let fields = vec![
                record_index.to_string(),
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
                candidate
                    .mass_error_da
                    .map(|value| format!("{value:.8}"))
                    .unwrap_or_default(),
                candidate.mass_valid.to_string(),
                candidate.from_diffusion.to_string(),
                candidate.from_causal_beam.to_string(),
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
    }
    output.flush()?;

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

fn optional_path(args: &[String], index: usize) -> Option<PathBuf> {
    args.get(index).and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("none"))
            .then(|| PathBuf::from(trimmed))
    })
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
}
