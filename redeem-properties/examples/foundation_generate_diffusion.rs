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
    foundation_diffusion_reverse_probabilities, foundation_precursor_mass_error_da,
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationDiffusionCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationPartition, FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideSpectrumDiffusionModel, PeptidoformInput, PrecursorContextBatch,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_NTERM_ACETYL,
    FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
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
    mass_error_da: Option<f64>,
    mass_valid: bool,
}

#[derive(Debug, Default)]
struct GenerationMetrics {
    records: usize,
    predicted_length_exact: usize,
    predicted_length_abs_error: usize,
    chains: usize,
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
    records_with_candidate: usize,
    best_abs_mass_error_sum: f64,
    best_abs_mass_error_records: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 9 {
        anyhow::bail!(
            "usage: foundation_generate_diffusion FOUNDATION_TRAINING.yaml CHECKPOINT_DIR OUTPUT.tsv [validation_records=64] [samples_per_record=16] [seed=20260901] [mass_tolerance_da=0.05] [temperature=1.0]"
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
    if validation_records == 0 || samples_per_record == 0 {
        anyhow::bail!("validation_records and samples_per_record must be positive");
    }
    if !(mass_tolerance_da > 0.0 && mass_tolerance_da.is_finite()) {
        anyhow::bail!("mass_tolerance_da must be positive and finite");
    }
    if !(temperature > 0.0 && temperature.is_finite()) {
        anyhow::bail!("temperature must be positive and finite");
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
        "record_index\ttarget_sequence\ttarget_active_tokens\tpredicted_active_tokens\tmass_rank\treverse_rank\tcandidate_sequence\tcandidate_modifications\treverse_log_probability\tmass_error_da\tmass_valid\tpeptidoform_exact\tsequence_exact\til_sequence_exact"
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
        let active_lengths = sample_generation_lengths(
            &predicted_length_distribution,
            predicted_active_length,
            samples_per_record,
            config.max_tokens,
            &mut rng,
        );
        let (rows, reverse_scores) = reverse_generate(
            &model,
            &diffusion_collator,
            &spectrum_collator,
            &config,
            record,
            &spectrum,
            &active_lengths,
            temperature,
            &mut rng,
            &device,
        )?;
        metrics.chains += rows.len();

        let mut unique = HashMap::<Vec<u32>, GeneratedCandidate>::new();
        for (tokens, reverse_log_probability) in rows.into_iter().zip(reverse_scores) {
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

        candidates.sort_by(mass_candidate_order);
        if candidates.iter().any(|candidate| candidate.mass_valid) {
            metrics.records_with_mass_valid_candidate += 1;
        }
        if let Some(error) = candidates
            .iter()
            .filter_map(|candidate| candidate.mass_error_da)
            .next()
        {
            metrics.best_abs_mass_error_sum += error.abs();
            metrics.best_abs_mass_error_records += 1;
        }

        let target_sequence = &record.peptidoform.sequence;
        let target_il = normalize_il(target_sequence);
        if candidates[0].peptide == record.peptidoform {
            metrics.mass_top1_peptidoform_exact += 1;
        }
        if candidates
            .iter()
            .any(|candidate| candidate.peptide == record.peptidoform)
        {
            metrics.mass_topk_peptidoform_exact += 1;
        }
        if candidates[0].peptide.sequence.as_str() == target_sequence.as_str() {
            metrics.mass_top1_sequence_exact += 1;
        }
        if candidates
            .iter()
            .any(|candidate| candidate.peptide.sequence.as_str() == target_sequence.as_str())
        {
            metrics.mass_topk_sequence_exact += 1;
        }
        if normalize_il(&candidates[0].peptide.sequence) == target_il {
            metrics.mass_top1_il_sequence_exact += 1;
        }
        if candidates
            .iter()
            .any(|candidate| normalize_il(&candidate.peptide.sequence) == target_il)
        {
            metrics.mass_topk_il_sequence_exact += 1;
        }

        let reverse_rank_by_tokens: HashMap<Vec<u32>, usize> = reverse_ranked
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.tokens.clone(), index + 1))
            .collect();
        for (mass_index, candidate) in candidates.iter().enumerate() {
            writeln!(
                output,
                "{record_index}\t{}\t{target_active_length}\t{predicted_active_length}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{}\t{}\t{}\t{}",
                record.peptidoform.sequence,
                mass_index + 1,
                reverse_rank_by_tokens.get(&candidate.tokens).copied().unwrap_or(0),
                candidate.peptide.sequence,
                format_modifications(&candidate.peptide),
                candidate.reverse_log_probability,
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
            "generation_record\trecord_index={record_index}\ttarget={}\ttarget_length={target_active_length}\tpredicted_length={predicted_active_length}\tvalid_candidates={}\tmass_valid_candidates={}\tbest_mass_error_da={}\ttop1={}\ttop1_exact={}\ttopk_exact={}",
            record.peptidoform.sequence,
            candidates.len(),
            candidates.iter().filter(|candidate| candidate.mass_valid).count(),
            candidates[0]
                .mass_error_da
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "NA".into()),
            candidates[0].peptide.sequence,
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
        "generation_summary\tvalid_decode_rate\t{:.6}",
        metrics.valid_decodes as f64 / metrics.chains.max(1) as f64
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
    if metrics.best_abs_mass_error_records > 0 {
        println!(
            "generation_summary\tmean_best_abs_mass_error_da\t{:.6}",
            metrics.best_abs_mass_error_sum / metrics.best_abs_mass_error_records as f64
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
    temperature: f64,
    rng: &mut GenerationRng,
    device: &Device,
) -> Result<(Vec<Vec<u32>>, Vec<f64>)> {
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

    for timestep in (1..=config.diffusion_steps).rev() {
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
    Ok((rows, reverse_log_probability))
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
    rng: &mut GenerationRng,
) -> Vec<usize> {
    let mut lengths = Vec::with_capacity(samples);
    lengths.push(argmax_length.clamp(2, max_tokens));
    for _ in 1..samples {
        lengths.push((sample_probability(probabilities, rng) + 1).clamp(2, max_tokens));
    }
    lengths
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
