//! Generate peptide/PTM candidates from observed spectra with a trained diffusion checkpoint.
//!
//! This is the first true inverse-generation evaluator: sequence length is predicted from
//! spectrum/precursor context, active tokens start from MASK, and categorical reverse refinement
//! proceeds without access to the clean peptide. The clean validation peptidoform is used only
//! after generation for metrics.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_reverse_probabilities,
    foundation_diffusion_token_mass_da, foundation_diffusion_token_residue,
    foundation_precursor_mass_error_da, foundation_precursor_neutral_mass, load_foundation_corpus,
    read_foundation_training_run_config, FoundationBenchmarkManifest, FoundationDiffusionCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationPartition,
    FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideSpectrumDiffusionModel, PeptidoformInput, PrecursorContextBatch,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_NTERM_ACETYL,
    FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
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
    mass_error_da: Option<f64>,
    mass_valid: bool,
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
    records_with_candidate: usize,
    best_abs_mass_error_sum: f64,
    best_abs_mass_error_records: usize,
    target_fragment_score_sum: f64,
    top1_fragment_score_sum: f64,
    target_matched_cleavages: usize,
    top1_matched_cleavages: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 13 {
        anyhow::bail!(
            "usage: foundation_generate_diffusion FOUNDATION_TRAINING.yaml CHECKPOINT_DIR OUTPUT.tsv [validation_records=64] [samples_per_record=16] [seed=20260901] [mass_tolerance_da=0.05] [temperature=1.0] [mass_beam_width=512] [final_candidates_per_chain=4] [fragment_tolerance_ppm=20] [spectral_beam_weight=2.0]"
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
    println!("primary_candidate_ranking\tfragment_mass");
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
        "record_index\ttarget_sequence\ttarget_active_tokens\tpredicted_active_tokens\tfragment_mass_rank\tmass_rank\treverse_rank\tcandidate_sequence\tcandidate_modifications\treverse_log_probability\tfragment_score\tmatched_cleavages\tmass_error_da\tmass_valid\tpeptidoform_exact\tsequence_exact\til_sequence_exact"
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
                mass_error_da,
                mass_valid,
            };
            unique
                .entry(tokens)
                .and_modify(|existing| {
                    if candidate.reverse_log_probability > existing.reverse_log_probability {
                        *existing = candidate.clone();
                    }
                })
                .or_insert(candidate);
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
        metrics.top1_fragment_score_sum += candidates[0].fragment_score;
        metrics.top1_matched_cleavages += candidates[0].matched_cleavages;

        let target_sequence = &record.peptidoform.sequence;
        let target_il = normalize_il(target_sequence);
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
        for (fragment_mass_index, candidate) in candidates.iter().enumerate() {
            writeln!(
                output,
                "{record_index}\t{}\t{target_active_length}\t{predicted_active_length}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}",
                record.peptidoform.sequence,
                fragment_mass_index + 1,
                mass_rank_by_tokens.get(&candidate.tokens).copied().unwrap_or(0),
                reverse_rank_by_tokens.get(&candidate.tokens).copied().unwrap_or(0),
                candidate.peptide.sequence,
                format_modifications(&candidate.peptide),
                candidate.reverse_log_probability,
                candidate.fragment_score,
                candidate.matched_cleavages,
                candidate
                    .mass_error_da
                    .map(|value| format!("{value:.8}"))
                    .unwrap_or_default(),
                candidate.mass_valid,
                candidate.peptide == record.peptidoform,
                candidate.peptide.sequence.as_str() == record.peptidoform.sequence.as_str(),
                normalize_il(&candidate.peptide.sequence) == target_il,
            )?;
        }

        println!(
            "generation_record\trecord_index={record_index}\ttarget={}\ttarget_length={target_active_length}\tpredicted_length={predicted_active_length}\tvalid_candidates={}\tmass_valid_candidates={}\tbest_mass_error_da={}\ttarget_fragment_score={:.4}\ttarget_matched_cleavages={}\ttop1={}\ttop1_fragment_score={:.4}\ttop1_matched_cleavages={}\ttop1_exact={}\ttopk_exact={}",
            record.peptidoform.sequence,
            candidates.len(),
            candidates.iter().filter(|candidate| candidate.mass_valid).count(),
            mass_ranked[0]
                .mass_error_da
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "NA".into()),
            target_fragment_evidence.score,
            target_fragment_evidence.matched_cleavages,
            candidates[0].peptide.sequence,
            candidates[0].fragment_score,
            candidates[0].matched_cleavages,
            candidates[0].peptide == record.peptidoform,
            candidates.iter().any(|candidate| candidate.peptide == record.peptidoform),
        );
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
    if metrics.best_abs_mass_error_records > 0 {
        println!(
            "generation_summary\tmean_best_abs_mass_error_da\t{:.6}",
            metrics.best_abs_mass_error_sum / metrics.best_abs_mass_error_records as f64
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
