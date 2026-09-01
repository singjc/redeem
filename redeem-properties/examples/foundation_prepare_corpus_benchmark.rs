//! Assemble multiple real-data sources and materialize one corpus-wide benchmark.
//!
//! The YAML config carries source paths plus optional source-level metadata.
//! `.zst` inputs are streamed through the system `zstd -dc`, avoiding a new
//! crate dependency while preserving one shared loader/instrument vocabulary.

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationCorpusConfig, FoundationSplitConfig, FoundationSplitMode,
};
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
    let config: FoundationCorpusConfig = serde_yaml::from_reader(file)
        .with_context(|| format!("failed to parse corpus config '{config_path}'"))?;
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
            "source\t{}\tprofile={}\trecords={}\tfingerprint=fnv1a64:{:016x}\tccs={}\texplicit_ccs={}\tderived_ccs={}\tccs_min={:?}\tccs_mean={:?}\tccs_max={:?}\tion_mobility={}",
            source.id,
            source.profile,
            source.record_count,
            source.dataset_fingerprint,
            source.stats.ccs_records,
            source.stats.explicit_ccs_records,
            source.stats.derived_ccs_records,
            source.stats.min_ccs,
            source.stats.mean_ccs,
            source.stats.max_ccs,
            source.stats.ion_mobility_records,
        );
    }
    println!("instrument_vocab\t{}", corpus.instrument_names.join(","));
    println!("train_records\t{}", benchmark.summary.train_records);
    println!(
        "validation_records\t{}",
        benchmark.summary.validation_records
    );
    println!("test_records\t{}", benchmark.summary.test_records);
    println!("total_groups\t{}", benchmark.summary.total_groups);
    Ok(())
}
