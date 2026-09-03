//! Assemble multiple real-data sources and materialize one corpus-wide benchmark.
//!
//! The YAML config carries source paths plus optional source-level metadata.
//! `.zst` inputs are streamed through the system `zstd -dc`, avoiding a new
//! crate dependency while preserving one shared loader/instrument vocabulary.

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationCorpusConfig, FoundationSplitConfig, FoundationSplitMode,
};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;

fn main() -> Result<()> {
    const USAGE: &str = "usage: foundation_prepare_corpus_benchmark <corpus.yaml> <benchmark.tsv> <provenance.tsv> [--mode sequence|peptidoform|run|instrument|modification-signature|modification-family] [--modified-only] [--validation FRACTION] [--test FRACTION] [--seed N]";

    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.len() < 3 {
        bail!(
            "foundation_prepare_corpus_benchmark requires corpus, benchmark, and provenance paths\n{USAGE}"
        );
    }
    if arguments[1].starts_with('-') {
        bail!(
            "missing benchmark output path before option '{}'\n{USAGE}",
            arguments[1]
        );
    }
    if arguments[2].starts_with('-') {
        bail!(
            "missing provenance output path before option '{}'\n{USAGE}",
            arguments[2]
        );
    }

    let config_path = arguments[0].clone();
    let benchmark_path = arguments[1].clone();
    let provenance_path = arguments[2].clone();
    let mut args = arguments.into_iter().skip(3);

    let file = File::open(&config_path)
        .with_context(|| format!("failed to open corpus config '{config_path}'"))?;
    let value: serde_yaml::Value = serde_yaml::from_reader(file)
        .with_context(|| format!("failed to parse corpus config '{config_path}'"))?;
    let config_value = value.get("corpus").cloned().unwrap_or(value);
    let config: FoundationCorpusConfig = serde_yaml::from_value(config_value)
        .with_context(|| format!("failed to parse foundation corpus from '{config_path}'"))?;
    let mut split = FoundationSplitConfig::default();
    let mut modified_only = false;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--modified-only" => modified_only = true,
            "--mode" => {
                let value = args.next().context("--mode requires a value")?;
                split.mode = match value.as_str() {
                    "sequence" => FoundationSplitMode::Sequence,
                    "peptidoform" => FoundationSplitMode::Peptidoform,
                    "run" => FoundationSplitMode::Run,
                    "instrument" => FoundationSplitMode::Instrument,
                    "modification-signature" => FoundationSplitMode::ModificationSignature,
                    "modification-family" => FoundationSplitMode::ModificationFamily,
                    other => bail!("unsupported split mode '{other}'"),
                };
            }
            "--validation" => {
                split.validation_fraction = args
                    .next()
                    .context("--validation requires a fraction")?
                    .parse()?;
            }
            "--test" => {
                split.test_fraction = args.next().context("--test requires a fraction")?.parse()?;
            }
            "--seed" => {
                split.seed = args.next().context("--seed requires an integer")?.parse()?;
            }
            other => bail!("unknown argument '{other}'"),
        }
    }

    let corpus = load_foundation_corpus(&config)?;
    let benchmark = corpus.build_benchmark_manifest(split, modified_only)?;
    benchmark.write_tsv(&benchmark_path)?;
    corpus.write_provenance_tsv(&provenance_path)?;

    println!("corpus_config\t{config_path}");
    println!("benchmark\t{benchmark_path}");
    println!("provenance\t{provenance_path}");
    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!("records\t{}", corpus.records.len());
    println!("sources\t{}", corpus.sources.len());
    for source in &corpus.sources {
        println!(
            "source\t{}\tprofile={}\trecords={}\tfingerprint=fnv1a64:{:016x}\tnormalized_rt={}\tnrt_min={:?}\tnrt_mean={:?}\tnrt_max={:?}\tharmonized_rt={}\thrt_min={:?}\thrt_mean={:?}\thrt_max={:?}\trt_calibration={}\tobserved_rt={}\tobserved_rt_min_s={:?}\tobserved_rt_mean_s={:?}\tobserved_rt_max_s={:?}\tccs={}\texplicit_ccs={}\tderived_ccs={}\tccs_min={:?}\tccs_mean={:?}\tccs_max={:?}\tion_mobility={}\tforward_ms2={}\tobserved_spectra={}\traw_observed_peaks={}",
            source.id,
            source.profile,
            source.record_count,
            source.dataset_fingerprint,
            source.stats.normalized_rt_records,
            source.stats.min_normalized_rt,
            source.stats.mean_normalized_rt,
            source.stats.max_normalized_rt,
            source.harmonized_rt_records,
            source.min_harmonized_rt,
            source.mean_harmonized_rt,
            source.max_harmonized_rt,
            source.rt_harmonization_calibration_id.as_deref().unwrap_or("none"),
            source.stats.observed_rt_records,
            source.stats.min_observed_rt_seconds,
            source.stats.mean_observed_rt_seconds,
            source.stats.max_observed_rt_seconds,
            source.stats.ccs_records,
            source.stats.explicit_ccs_records,
            source.stats.derived_ccs_records,
            source.stats.min_ccs,
            source.stats.mean_ccs,
            source.stats.max_ccs,
            source.stats.ion_mobility_records,
            source.stats.ms2_records,
            source.stats.observed_spectrum_records,
            source.stats.raw_observed_peak_rows,
        );
    }
    println!("instrument_vocab\t{}", corpus.instrument_names.join(","));
    print_source_overlap(&corpus, &benchmark);
    println!("train_records\t{}", benchmark.summary.train_records);
    println!(
        "validation_records\t{}",
        benchmark.summary.validation_records
    );
    println!("test_records\t{}", benchmark.summary.test_records);
    println!("total_groups\t{}", benchmark.summary.total_groups);
    Ok(())
}

fn print_source_overlap(
    corpus: &redeem_properties::foundation::FoundationCorpus,
    benchmark: &redeem_properties::foundation::FoundationBenchmarkManifest,
) {
    let peptidoform_by_record: BTreeMap<usize, &str> = benchmark
        .entries
        .iter()
        .map(|entry| (entry.record_index, entry.peptidoform.as_str()))
        .collect();
    let mut sequences = BTreeMap::<String, BTreeSet<String>>::new();
    let mut precursor_identities = BTreeMap::<String, BTreeSet<String>>::new();
    for (record_index, record) in corpus.records.iter().enumerate() {
        let Some(provenance) = corpus.provenance.get(record_index) else {
            continue;
        };
        sequences
            .entry(provenance.source_id.clone())
            .or_default()
            .insert(record.peptidoform.sequence.clone());
        if let Some(peptidoform) = peptidoform_by_record.get(&record_index) {
            let charge = record
                .context
                .charge
                .map(|charge| charge.to_string())
                .unwrap_or_else(|| "NA".to_string());
            precursor_identities
                .entry(provenance.source_id.clone())
                .or_default()
                .insert(format!("{peptidoform}|z{charge}"));
        }
    }

    let source_ids: Vec<&String> = corpus.sources.iter().map(|source| &source.id).collect();
    for (left_index, left) in source_ids.iter().enumerate() {
        for right in source_ids.iter().skip(left_index + 1) {
            let empty = BTreeSet::new();
            let left_sequences = sequences.get(*left).unwrap_or(&empty);
            let right_sequences = sequences.get(*right).unwrap_or(&empty);
            let sequence_overlap = left_sequences.intersection(right_sequences).count();
            let sequence_left_fraction = if left_sequences.is_empty() {
                0.0
            } else {
                sequence_overlap as f64 / left_sequences.len() as f64
            };
            let sequence_right_fraction = if right_sequences.is_empty() {
                0.0
            } else {
                sequence_overlap as f64 / right_sequences.len() as f64
            };

            let left_precursors = precursor_identities.get(*left).unwrap_or(&empty);
            let right_precursors = precursor_identities.get(*right).unwrap_or(&empty);
            let precursor_overlap = left_precursors.intersection(right_precursors).count();
            let precursor_left_fraction = if left_precursors.is_empty() {
                0.0
            } else {
                precursor_overlap as f64 / left_precursors.len() as f64
            };
            let precursor_right_fraction = if right_precursors.is_empty() {
                0.0
            } else {
                precursor_overlap as f64 / right_precursors.len() as f64
            };

            println!(
                "source_overlap\tleft={}\tright={}\tsequence_overlap={}\tleft_sequences={}\tright_sequences={}\tsequence_left_fraction={:.6}\tsequence_right_fraction={:.6}\tpeptidoform_charge_overlap={}\tleft_peptidoform_charge={}\tright_peptidoform_charge={}\tpeptidoform_charge_left_fraction={:.6}\tpeptidoform_charge_right_fraction={:.6}",
                left,
                right,
                sequence_overlap,
                left_sequences.len(),
                right_sequences.len(),
                sequence_left_fraction,
                sequence_right_fraction,
                precursor_overlap,
                left_precursors.len(),
                right_precursors.len(),
                precursor_left_fraction,
                precursor_right_fraction,
            );
        }
    }
}
