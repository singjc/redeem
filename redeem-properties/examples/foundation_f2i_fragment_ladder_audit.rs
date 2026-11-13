//! F2-I M1 audit: per-source MS2 fragment-ladder coverage on TRAIN (read-only, CPU).
//!
//! For every TRAIN record with annotated fragments, count annotated cells in channels 0..4
//! (b1+, b2+, y1+, y2+) against the possible cells `(len - 1) x n_channels`, where 2+
//! channels count only when precursor charge >= 2. A source whose spectra list nearly
//! the full ladder supports zero targets for unannotated cells; a source that lists only
//! the top-N fragments does not (missing = "not reported", not "absent").
//!
//! usage: foundation_f2i_fragment_ladder_audit RUN.yaml RUN_SHA256 > audit.tsv
use anyhow::{Context, Result};
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationBenchmarkManifest, FoundationCorpusConfig,
    FoundationPartition,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Run {
    corpus: FoundationCorpusConfig,
    benchmark_manifest: std::path::PathBuf,
}

#[derive(Default)]
struct Acc {
    spectra: usize,
    annotated: usize,
    possible: usize,
    full: usize,
    per_spectrum: Vec<f64>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() == 2, "usage: RUN.yaml RUN_SHA256");
    let out = std::process::Command::new("sha256sum").arg(&args[0]).output()?;
    let observed = String::from_utf8(out.stdout)?;
    anyhow::ensure!(observed.starts_with(&args[1]), "run YAML SHA256 mismatch");
    let run: Run = serde_yaml::from_str(&std::fs::read_to_string(&args[0])?)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let bench = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .context("benchmark manifest")?;
    bench.validate_against_records(&corpus.records)?;
    let mut acc: BTreeMap<String, Acc> = BTreeMap::new();
    for i in bench.partition_indices(FoundationPartition::Train) {
        let r = &corpus.records[i];
        let by = r.fragments.iter().filter(|f| f.channel < 4).count();
        if by == 0 {
            continue;
        }
        let len = r.peptidoform.sequence.chars().count();
        let charge = r.context.charge.unwrap_or(1).max(1) as usize;
        let channels = if charge >= 2 { 4 } else { 2 };
        let possible = len.saturating_sub(1) * channels;
        let a = acc.entry(corpus.provenance[i].source_id.clone()).or_default();
        a.spectra += 1;
        a.annotated += by;
        a.possible += possible;
        let frac = by as f64 / possible.max(1) as f64;
        a.full += usize::from(frac >= 0.9);
        a.per_spectrum.push(frac);
    }
    println!("source\tspectra\tannotated_by_cells\tpossible_by_cells\tcoverage\tmedian_spectrum_coverage\tshare_spectra_ge90pct");
    for (src, mut a) in acc {
        a.per_spectrum.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let median = a.per_spectrum[a.per_spectrum.len() / 2];
        println!(
            "{src}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}",
            a.spectra,
            a.annotated,
            a.possible,
            a.annotated as f64 / a.possible.max(1) as f64,
            median,
            a.full as f64 / a.spectra as f64
        );
    }
    println!("#F2I_LADDER_AUDIT=COMPLETE");
    Ok(())
}
