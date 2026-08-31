//! Materialize a deterministic leakage-safe foundation benchmark manifest.
//!
//! Examples:
//!
//! ```text
//! cargo run -p redeem-properties --example foundation_prepare_benchmark -- library.tsv benchmark.sequence.tsv
//! zstd -dc library.tsv.zst | cargo run -p redeem-properties --example foundation_prepare_benchmark -- - benchmark.ptm.tsv --delimiter tab --mode modification-family --modified-only
//! ```

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    build_foundation_benchmark_manifest, FoundationDatasetLoader, FoundationSplitConfig,
    FoundationSplitMode, FoundationTableLoaderConfig,
};
use std::env;
use std::io;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(input) = args.next() else {
        bail!("usage: foundation_prepare_benchmark <PATH|-> <OUTPUT.tsv> [--mode sequence|peptidoform|run|instrument|modification-signature|modification-family] [--modified-only] [--validation FRACTION] [--test FRACTION] [--seed N] [--delimiter tab|comma] [--strict]");
    };
    let Some(output) = args.next() else {
        bail!("foundation_prepare_benchmark requires an output manifest path");
    };

    let mut loader_config = FoundationTableLoaderConfig {
        strict: false,
        ..FoundationTableLoaderConfig::default()
    };
    let mut split_config = FoundationSplitConfig::default();
    let mut modified_only = false;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--strict" => loader_config.strict = true,
            "--modified-only" => modified_only = true,
            "--delimiter" => {
                let value = args.next().context("--delimiter requires tab or comma")?;
                loader_config.delimiter = Some(match value.as_str() {
                    "tab" | "tsv" => b'\t',
                    "comma" | "csv" => b',',
                    other => bail!("unsupported delimiter '{other}'"),
                });
            }
            "--mode" => {
                let value = args.next().context("--mode requires a value")?;
                split_config.mode = match value.as_str() {
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
                split_config.validation_fraction = args
                    .next()
                    .context("--validation requires a fraction")?
                    .parse()?;
            }
            "--test" => {
                split_config.test_fraction =
                    args.next().context("--test requires a fraction")?.parse()?;
            }
            "--seed" => {
                split_config.seed = args.next().context("--seed requires an integer")?.parse()?;
            }
            other => bail!("unknown argument '{other}'"),
        }
    }

    let mut loader = FoundationDatasetLoader::new(64);
    let records = if input == "-" {
        let delimiter = loader_config.delimiter.unwrap_or(b'\t');
        loader.load_reader(io::stdin().lock(), delimiter, &loader_config)?
    } else {
        let path = PathBuf::from(&input);
        if path.extension().and_then(|value| value.to_str()) == Some("zst") {
            bail!(
                "compressed .zst input should be streamed: zstd -dc '{}' | cargo run -p redeem-properties --example foundation_prepare_benchmark -- - '{}' --delimiter tab ...",
                path.display(),
                output
            );
        }
        loader.load_path(&path, &loader_config)?
    };

    let manifest = build_foundation_benchmark_manifest(&records, split_config, modified_only)?;
    manifest.write_tsv(&output)?;
    println!("input\t{input}");
    println!("output\t{output}");
    println!(
        "dataset_fingerprint\tfnv1a64:{:016x}",
        manifest.dataset_fingerprint
    );
    println!("source_records\t{}", manifest.source_records);
    println!("selected_records\t{}", manifest.selected_records);
    println!(
        "excluded_mixed_family_records\t{}",
        manifest.excluded_mixed_family_records
    );
    println!("train_records\t{}", manifest.summary.train_records);
    println!(
        "validation_records\t{}",
        manifest.summary.validation_records
    );
    println!("test_records\t{}", manifest.summary.test_records);
    println!("total_groups\t{}", manifest.summary.total_groups);
    println!("train_groups\t{}", manifest.summary.train_groups);
    println!("validation_groups\t{}", manifest.summary.validation_groups);
    println!("test_groups\t{}", manifest.summary.test_groups);
    Ok(())
}
