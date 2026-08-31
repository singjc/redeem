//! Inspect an unfamiliar transition/spectral-library table before training.
//!
//! Examples:
//!
//! ```text
//! cargo run -p redeem-properties --example foundation_inspect_table -- library.tsv
//! zstd -dc library.tsv.zst | cargo run -p redeem-properties --example foundation_inspect_table -- - --delimiter tab
//! ```

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{FoundationDatasetLoader, FoundationTableLoaderConfig};
use std::env;
use std::io;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(input) = args.next() else {
        bail!("usage: foundation_inspect_table <PATH|-> [--delimiter tab|comma] [--strict]");
    };

    let mut config = FoundationTableLoaderConfig {
        strict: false,
        ..FoundationTableLoaderConfig::default()
    };
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--strict" => config.strict = true,
            "--delimiter" => {
                let value = args
                    .next()
                    .context("--delimiter requires 'tab' or 'comma'")?;
                config.delimiter = Some(match value.as_str() {
                    "tab" | "tsv" => b'\t',
                    "comma" | "csv" => b',',
                    other => bail!("unsupported delimiter '{other}'; use tab or comma"),
                });
            }
            other => bail!("unknown argument '{other}'"),
        }
    }

    let mut loader = FoundationDatasetLoader::new(64);
    let report = if input == "-" {
        let delimiter = config.delimiter.unwrap_or(b'\t');
        loader.load_reader_with_report(io::stdin().lock(), delimiter, &config)?
    } else {
        let path = PathBuf::from(&input);
        if path.extension().and_then(|value| value.to_str()) == Some("zst") {
            bail!(
                "compressed .zst input is not opened directly; stream it with: zstd -dc '{}' | cargo run -p redeem-properties --example foundation_inspect_table -- - --delimiter tab",
                path.display()
            );
        }
        loader.load_path_with_report(&path, &config)?
    };

    println!("input: {input}");
    println!(
        "delimiter: {}",
        if report.delimiter == b'\t' {
            "tab"
        } else if report.delimiter == b',' {
            "comma"
        } else {
            "custom"
        }
    );
    println!("\n# inferred schema");
    print!("{}", serde_yaml::to_string(&report.schema)?);
    println!("\n# load statistics");
    print!("{}", serde_yaml::to_string(&report.stats)?);
    println!("\n# instrument vocabulary");
    for (index, name) in loader.instruments().names().iter().enumerate() {
        println!("{index}\t{name}");
    }
    Ok(())
}
