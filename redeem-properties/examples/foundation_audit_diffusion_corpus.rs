//! Audit how much of an existing foundation corpus can supervise the inverse
//! observed-spectrum -> peptide diffusion lane without theoretical-m/z leakage.

use anyhow::{Context, Result};
use redeem_properties::foundation::{
    load_foundation_corpus, read_foundation_training_run_config, FoundationBenchmarkManifest,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationPartition,
    FoundationSpectrum,
};
use std::collections::BTreeMap;
use std::env;

#[derive(Default)]
struct Counts {
    records: usize,
    observed_spectra: usize,
    tokenizable: usize,
    usable_pairs: usize,
    observed_peaks: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        anyhow::bail!("usage: foundation_audit_diffusion_corpus /path/to/foundation_training.yaml");
    }
    let run = read_foundation_training_run_config(&args[1])?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let manifest = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    manifest.validate_against_records(&corpus.records)?;

    let diffusion = FoundationDiffusionConfig::default();
    let vocabulary = FoundationDiffusionVocabulary;
    let mut overall = BTreeMap::<String, Counts>::new();
    let mut by_source = BTreeMap::<(String, String), Counts>::new();
    let mut unsupported_examples = Vec::<String>::new();

    for entry in &manifest.entries {
        let partition = match entry.partition {
            FoundationPartition::Train => "train",
            FoundationPartition::Validation => "validation",
            FoundationPartition::Test => "test",
        }
        .to_string();
        let record = &corpus.records[entry.record_index];
        let provenance = &corpus.provenance[entry.record_index];
        let spectrum = FoundationSpectrum::from_training_record(record);
        let tokenizable = vocabulary
            .encode(&record.peptidoform, diffusion.max_tokens)
            .is_ok();
        if !tokenizable && unsupported_examples.len() < 8 {
            if let Err(error) = vocabulary.encode(&record.peptidoform, diffusion.max_tokens) {
                unsupported_examples.push(format!(
                    "{}\t{}\t{}",
                    provenance.source_id, record.peptidoform.sequence, error
                ));
            }
        }
        update(
            overall.entry(partition.clone()).or_default(),
            spectrum.as_ref(),
            tokenizable,
        );
        update(
            by_source
                .entry((partition, provenance.source_id.clone()))
                .or_default(),
            spectrum.as_ref(),
            tokenizable,
        );
    }

    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!("records\t{}", corpus.records.len());
    println!("diffusion_max_tokens\t{}", diffusion.max_tokens);
    println!("diffusion_vocab_size\t{}", vocabulary.size());
    for (partition, counts) in overall {
        print_counts("partition", &partition, &counts);
    }
    for ((partition, source), counts) in by_source {
        print_counts(&format!("partition_source\t{partition}"), &source, &counts);
    }
    for example in unsupported_examples {
        println!("unsupported_token_example\t{example}");
    }
    Ok(())
}

fn update(counts: &mut Counts, spectrum: Option<&FoundationSpectrum>, tokenizable: bool) {
    counts.records += 1;
    if tokenizable {
        counts.tokenizable += 1;
    }
    if let Some(spectrum) = spectrum {
        counts.observed_spectra += 1;
        counts.observed_peaks += spectrum.peaks.len();
        if tokenizable {
            counts.usable_pairs += 1;
        }
    }
}

fn print_counts(prefix: &str, name: &str, counts: &Counts) {
    println!(
        "{prefix}\t{name}\trecords={}\tobserved_spectra={}\ttokenizable={}\tusable_pairs={}\tobserved_peaks={}",
        counts.records,
        counts.observed_spectra,
        counts.tokenizable,
        counts.usable_pairs,
        counts.observed_peaks,
    );
}
