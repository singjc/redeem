use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    FoundationPracticalIdentifierBatchRequestV0760, FoundationPracticalIdentifierCatalogV0752,
    FoundationPracticalIdentifierPeakV0752, FoundationPracticalIdentifierRequestV0752,
    FoundationPracticalIdentifierSearchSpaceV0760, FoundationPracticalIdentifierServiceV0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const V0770_VERSION: &str = "v0.77.0-practical-identifier-stream";
const V0770_RUNTIME: &str = "grouped_spectra_tsv_stream_to_frozen_v0752_service";
const FIXTURE_SEARCH_SPACE: &str =
    include_str!("../tests/fixtures/foundation_identifier_v0760/search_space.yaml");
const FIXTURE_BATCH: &str =
    include_str!("../tests/fixtures/foundation_identifier_v0760/batch.yaml");

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    BatchToTsv {
        batch: PathBuf,
        output_spectra: PathBuf,
    },
    IdentifyStream {
        catalog: PathBuf,
        spectra: PathBuf,
        output_hits: PathBuf,
        output_summary: PathBuf,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpectrumRow {
    query_id: String,
    observed_precursor_mz: f64,
    observed_charge: i32,
    top_k: usize,
    peak_mz: f32,
    intensity: f32,
}

#[derive(Debug)]
struct PendingRequest {
    query_id: String,
    observed_precursor_mz: f64,
    observed_charge: i32,
    top_k: usize,
    peaks: Vec<FoundationPracticalIdentifierPeakV0752>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamCounts {
    requests: usize,
    peak_rows: usize,
    candidates_scored: usize,
    returned_hits: usize,
}

fn usage() -> &'static str {
    "usage:\n  foundation_identify_stream_v0770 batch-to-tsv --batch BATCH.yaml --output-spectra SPECTRA.tsv\n  foundation_identify_stream_v0770 identify-stream --catalog CATALOG.yaml --spectra SPECTRA.tsv --output-hits HITS.tsv --output-summary SUMMARY.tsv\n  foundation_identify_stream_v0770 --self-test\n  foundation_identify_stream_v0770 --version"
}

fn parse_named_values(args: &[String], allowed: &[&str]) -> Result<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index].as_str();
        if !allowed.contains(&flag) {
            bail!("unknown v0.77.0 CLI argument {flag:?}\n{}", usage());
        }
        if values.contains_key(flag) {
            bail!("duplicate v0.77.0 CLI argument {flag:?}\n{}", usage());
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| anyhow::anyhow!("missing value for {flag:?}\n{}", usage()))?;
        if value.starts_with("--") {
            bail!("missing value for {flag:?}\n{}", usage());
        }
        values.insert(flag.to_string(), value.clone());
        index += 2;
    }
    Ok(values)
}

fn required_path(values: &BTreeMap<String, String>, flag: &str) -> Result<PathBuf> {
    values
        .get(flag)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("missing required {flag}\n{}", usage()))
}

fn parse_cli_args(args: &[String]) -> Result<Command> {
    let subcommand = args.get(1).ok_or_else(|| anyhow::anyhow!(usage()))?;
    match subcommand.as_str() {
        "batch-to-tsv" => {
            let values = parse_named_values(&args[2..], &["--batch", "--output-spectra"])?;
            Ok(Command::BatchToTsv {
                batch: required_path(&values, "--batch")?,
                output_spectra: required_path(&values, "--output-spectra")?,
            })
        }
        "identify-stream" => {
            let values = parse_named_values(
                &args[2..],
                &[
                    "--catalog",
                    "--spectra",
                    "--output-hits",
                    "--output-summary",
                ],
            )?;
            Ok(Command::IdentifyStream {
                catalog: required_path(&values, "--catalog")?,
                spectra: required_path(&values, "--spectra")?,
                output_hits: required_path(&values, "--output-hits")?,
                output_summary: required_path(&values, "--output-summary")?,
            })
        }
        _ => bail!("unknown v0.77.0 subcommand {subcommand:?}\n{}", usage()),
    }
}

fn ensure_output_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    Ok(())
}

fn write_batch_tsv<W: Write>(
    batch: &FoundationPracticalIdentifierBatchRequestV0760,
    writer: W,
) -> Result<()> {
    if batch.requests.is_empty() {
        bail!("v0.77.0 source batch must contain at least one request");
    }
    let mut seen = BTreeSet::new();
    let mut out = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_writer(writer);
    out.write_record([
        "query_id",
        "observed_precursor_mz",
        "observed_charge",
        "top_k",
        "peak_mz",
        "intensity",
    ])?;
    for request in &batch.requests {
        if !seen.insert(request.query_id.as_str()) {
            bail!(
                "v0.77.0 duplicate query_id {:?} in source batch",
                request.query_id
            );
        }
        if request.peaks.is_empty() {
            bail!(
                "v0.77.0 request {:?} has no spectrum peaks",
                request.query_id
            );
        }
        for peak in &request.peaks {
            out.write_record([
                request.query_id.clone(),
                request.observed_precursor_mz.to_string(),
                request.observed_charge.to_string(),
                request.top_k.to_string(),
                peak.mz.to_string(),
                peak.intensity.to_string(),
            ])?;
        }
    }
    out.flush()?;
    Ok(())
}

fn finish_request<W: Write>(
    service: &FoundationPracticalIdentifierServiceV0752,
    hits: &mut csv::Writer<W>,
    request: PendingRequest,
) -> Result<(usize, usize)> {
    let response = service.identify(&FoundationPracticalIdentifierRequestV0752 {
        schema: FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752.to_string(),
        query_id: request.query_id,
        observed_precursor_mz: request.observed_precursor_mz,
        observed_charge: request.observed_charge,
        peaks: request.peaks,
        top_k: request.top_k,
    })?;
    for hit in &response.hits {
        hits.write_record([
            response.query_id.clone(),
            hit.rank.to_string(),
            hit.candidate_index.to_string(),
            hit.key.clone(),
            hit.charge.to_string(),
            format!("{:.12}", hit.theoretical_neutral_mass),
            format!("{:.12}", hit.absolute_neutral_mass_error),
            format!("{:.12}", hit.geometry_score),
        ])?;
    }
    Ok((response.candidates_scored, response.returned_hits))
}

fn identify_stream<R: Read, W: Write>(
    service: &FoundationPracticalIdentifierServiceV0752,
    input: R,
    output: W,
) -> Result<StreamCounts> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(input);
    let headers = reader.headers()?.clone();
    let expected = [
        "query_id",
        "observed_precursor_mz",
        "observed_charge",
        "top_k",
        "peak_mz",
        "intensity",
    ];
    if headers.len() != expected.len()
        || headers
            .iter()
            .zip(expected)
            .any(|(observed, expected)| observed != expected)
    {
        bail!("v0.77.0 spectra TSV header mismatch");
    }

    let mut hits = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_writer(output);
    hits.write_record([
        "query_id",
        "rank",
        "candidate_index",
        "key",
        "charge",
        "theoretical_neutral_mass",
        "absolute_neutral_mass_error",
        "geometry_score",
    ])?;

    // ponytail: rows must be grouped by query_id; sort upstream if a producer cannot guarantee it.
    let mut seen = BTreeSet::<String>::new();
    let mut pending: Option<PendingRequest> = None;
    let mut counts = StreamCounts {
        requests: 0,
        peak_rows: 0,
        candidates_scored: 0,
        returned_hits: 0,
    };

    for row in reader.deserialize::<SpectrumRow>() {
        let row = row.context("failed to parse v0.77.0 spectra TSV row")?;
        counts.peak_rows += 1;
        let same_query = pending
            .as_ref()
            .is_some_and(|current| current.query_id == row.query_id);
        if !same_query {
            if let Some(request) = pending.take() {
                let (scored, returned) = finish_request(service, &mut hits, request)?;
                counts.requests += 1;
                counts.candidates_scored += scored;
                counts.returned_hits += returned;
            }
            if !seen.insert(row.query_id.clone()) {
                bail!(
                    "v0.77.0 query_id {:?} reappeared after another query; spectra TSV must be grouped",
                    row.query_id
                );
            }
            pending = Some(PendingRequest {
                query_id: row.query_id.clone(),
                observed_precursor_mz: row.observed_precursor_mz,
                observed_charge: row.observed_charge,
                top_k: row.top_k,
                peaks: Vec::new(),
            });
        }

        let current = pending
            .as_mut()
            .context("v0.77.0 internal grouped-row state missing")?;
        if current.observed_precursor_mz.to_bits() != row.observed_precursor_mz.to_bits()
            || current.observed_charge != row.observed_charge
            || current.top_k != row.top_k
        {
            bail!(
                "v0.77.0 inconsistent metadata within query_id {:?}",
                row.query_id
            );
        }
        current.peaks.push(FoundationPracticalIdentifierPeakV0752 {
            mz: row.peak_mz,
            intensity: row.intensity,
        });
    }

    if let Some(request) = pending.take() {
        let (scored, returned) = finish_request(service, &mut hits, request)?;
        counts.requests += 1;
        counts.candidates_scored += scored;
        counts.returned_hits += returned;
    }
    if counts.requests == 0 {
        bail!("v0.77.0 spectra TSV contained no requests");
    }
    hits.flush()?;
    Ok(counts)
}

fn run_batch_to_tsv(batch_path: &Path, output_spectra: &Path) -> Result<()> {
    if batch_path == output_spectra {
        bail!("v0.77.0 refuses to overwrite --batch with --output-spectra");
    }
    ensure_output_parent(output_spectra)?;
    let batch_yaml = fs::read_to_string(batch_path)
        .with_context(|| format!("failed to read batch {}", batch_path.display()))?;
    let batch = FoundationPracticalIdentifierBatchRequestV0760::from_yaml_str(&batch_yaml)?;
    let file = File::create(output_spectra)
        .with_context(|| format!("failed to create spectra TSV {}", output_spectra.display()))?;
    write_batch_tsv(&batch, file)?;
    let peak_rows: usize = batch
        .requests
        .iter()
        .map(|request| request.peaks.len())
        .sum();
    println!("v0770_version={V0770_VERSION}");
    println!("runtime={V0770_RUNTIME}");
    println!("request_count={}", batch.requests.len());
    println!("peak_rows={peak_rows}");
    println!("protected_data_required=NO");
    println!("output_spectra={}", output_spectra.display());
    Ok(())
}

fn run_identify_stream(
    catalog_path: &Path,
    spectra_path: &Path,
    output_hits: &Path,
    output_summary: &Path,
) -> Result<()> {
    if output_hits == catalog_path
        || output_hits == spectra_path
        || output_summary == catalog_path
        || output_summary == spectra_path
        || output_summary == output_hits
    {
        bail!("v0.77.0 output paths must be distinct from inputs and from each other");
    }
    ensure_output_parent(output_hits)?;
    ensure_output_parent(output_summary)?;
    let catalog_yaml = fs::read_to_string(catalog_path)
        .with_context(|| format!("failed to read catalog {}", catalog_path.display()))?;
    let catalog = FoundationPracticalIdentifierCatalogV0752::from_yaml_str(&catalog_yaml)?;
    let service = FoundationPracticalIdentifierServiceV0752::new(catalog)?;
    let input = File::open(spectra_path)
        .with_context(|| format!("failed to open spectra TSV {}", spectra_path.display()))?;
    let output = File::create(output_hits)
        .with_context(|| format!("failed to create hits TSV {}", output_hits.display()))?;
    let started = Instant::now();
    let counts = identify_stream(&service, input, output)?;
    let elapsed = started.elapsed().as_secs_f64();
    fs::write(
        output_summary,
        format!(
            concat!(
                "key\tvalue\n",
                "status\tPASS\n",
                "runtime\t{}\n",
                "catalog_fingerprint\t{}\n",
                "catalog_size\t{}\n",
                "request_count\t{}\n",
                "peak_rows\t{}\n",
                "total_candidates_scored\t{}\n",
                "total_returned_hits\t{}\n",
                "elapsed_seconds\t{:.6}\n",
                "candidate_pool\t{}\n",
                "target_forcing\tNO\n",
                "protected_data_required\tNO\n"
            ),
            V0770_RUNTIME,
            service.catalog_fingerprint(),
            service.catalog_size(),
            counts.requests,
            counts.peak_rows,
            counts.candidates_scored,
            counts.returned_hits,
            elapsed,
            FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
        ),
    )?;
    println!("v0770_version={V0770_VERSION}");
    println!("runtime={V0770_RUNTIME}");
    println!("catalog_fingerprint={}", service.catalog_fingerprint());
    println!("catalog_size={}", service.catalog_size());
    println!("request_count={}", counts.requests);
    println!("peak_rows={}", counts.peak_rows);
    println!("total_candidates_scored={}", counts.candidates_scored);
    println!("total_returned_hits={}", counts.returned_hits);
    println!("elapsed_seconds={elapsed:.6}");
    println!("candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
    println!("target_forcing=NO");
    println!("protected_data_required=NO");
    println!("output_hits={}", output_hits.display());
    println!("output_summary={}", output_summary.display());
    Ok(())
}

fn self_test() -> Result<()> {
    let search_space =
        FoundationPracticalIdentifierSearchSpaceV0760::from_yaml_str(FIXTURE_SEARCH_SPACE)?;
    let catalog = search_space.materialize_catalog()?;
    let service = FoundationPracticalIdentifierServiceV0752::new(catalog)?;
    let batch = FoundationPracticalIdentifierBatchRequestV0760::from_yaml_str(FIXTURE_BATCH)?;
    let mut spectra = Vec::new();
    write_batch_tsv(&batch, &mut spectra)?;
    let mut hits = Vec::new();
    let counts = identify_stream(&service, Cursor::new(spectra), &mut hits)?;
    let hits = String::from_utf8(hits)?;
    if counts.requests != 2
        || counts.candidates_scored != 3
        || counts.returned_hits != 3
        || !hits.contains("batch-q2\t1\t")
        || !hits.contains("batch-q3\t1\t")
    {
        bail!("v0.77.0 streaming self-test failed");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 2 && args[1] == "--self-test" {
        self_test()?;
        println!("v0770_practical_identifier_stream_self_test=PASS");
        println!("v0770_candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
        println!("v0770_protected_data_required=NO");
        return Ok(());
    }
    if args.len() == 2 && args[1] == "--version" {
        println!("{V0770_VERSION}");
        return Ok(());
    }
    if args.len() == 2 && args[1] == "--help" {
        println!("{}", usage());
        return Ok(());
    }

    match parse_cli_args(&args)? {
        Command::BatchToTsv {
            batch,
            output_spectra,
        } => run_batch_to_tsv(&batch, &output_spectra),
        Command::IdentifyStream {
            catalog,
            spectra,
            output_hits,
            output_summary,
        } => run_identify_stream(&catalog, &spectra, &output_hits, &output_summary),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_test_passes() {
        self_test().unwrap();
    }

    #[test]
    fn parser_rejects_tuning_flags() {
        let args = vec![
            "foundation_identify_stream_v0770".to_string(),
            "identify-stream".to_string(),
            "--catalog".to_string(),
            "catalog.yaml".to_string(),
            "--spectra".to_string(),
            "spectra.tsv".to_string(),
            "--output-hits".to_string(),
            "hits.tsv".to_string(),
            "--output-summary".to_string(),
            "summary.tsv".to_string(),
            "--candidate-pool".to_string(),
            "1024".to_string(),
        ];
        assert!(parse_cli_args(&args).is_err());
    }
}
