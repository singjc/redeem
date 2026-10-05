use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    FoundationPracticalIdentifierRequestV0752, FoundationPracticalIdentifierResponseV0752,
    FoundationPracticalIdentifierServiceV0752, FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const V0753_VERSION: &str = "v0.75.3-practical-identifier-cli";
const V0753_RUNTIME: &str = "production_yaml_request_to_frozen_v0752_service";
const FIXTURE_CATALOG: &str =
    include_str!("../tests/fixtures/foundation_identifier_v0753/catalog.yaml");
const FIXTURE_REQUEST: &str =
    include_str!("../tests/fixtures/foundation_identifier_v0753/request.yaml");

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    catalog: PathBuf,
    request: PathBuf,
    output_yaml: PathBuf,
    output_tsv: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: foundation_identify_practical_v0753 --catalog CATALOG.yaml --request REQUEST.yaml --output-yaml RESPONSE.yaml [--output-tsv HITS.tsv]\n       foundation_identify_practical_v0753 --self-test\n       foundation_identify_practical_v0753 --version"
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs> {
    let mut values = BTreeMap::<String, String>::new();
    let mut index = 1usize;
    while index < args.len() {
        let flag = args[index].as_str();
        if !matches!(
            flag,
            "--catalog" | "--request" | "--output-yaml" | "--output-tsv"
        ) {
            bail!("unknown v0.75.3 CLI argument {flag:?}\n{}", usage());
        }
        if values.contains_key(flag) {
            bail!("duplicate v0.75.3 CLI argument {flag:?}\n{}", usage());
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

    let required = |flag: &str| -> Result<PathBuf> {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("missing required {flag}\n{}", usage()))
    };

    Ok(CliArgs {
        catalog: required("--catalog")?,
        request: required("--request")?,
        output_yaml: required("--output-yaml")?,
        output_tsv: values.get("--output-tsv").map(PathBuf::from),
    })
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

fn write_response_yaml(
    path: &Path,
    response: &FoundationPracticalIdentifierResponseV0752,
) -> Result<()> {
    ensure_output_parent(path)?;
    let body = response.to_yaml_string()?;
    fs::write(path, body)
        .with_context(|| format!("failed to write response YAML {}", path.display()))
}

fn response_hits_tsv(response: &FoundationPracticalIdentifierResponseV0752) -> Result<Vec<u8>> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_writer(Vec::new());
    writer.write_record([
        "rank",
        "candidate_index",
        "key",
        "charge",
        "theoretical_neutral_mass",
        "absolute_neutral_mass_error",
        "geometry_score",
    ])?;
    for hit in &response.hits {
        writer.write_record([
            hit.rank.to_string(),
            hit.candidate_index.to_string(),
            hit.key.clone(),
            hit.charge.to_string(),
            format!("{:.12}", hit.theoretical_neutral_mass),
            format!("{:.12}", hit.absolute_neutral_mass_error),
            format!("{:.12}", hit.geometry_score),
        ])?;
    }
    writer
        .into_inner()
        .map_err(|error| anyhow::anyhow!(error.into_error()))
}

fn write_response_tsv(
    path: &Path,
    response: &FoundationPracticalIdentifierResponseV0752,
) -> Result<()> {
    ensure_output_parent(path)?;
    fs::write(path, response_hits_tsv(response)?)
        .with_context(|| format!("failed to write response TSV {}", path.display()))
}

fn identify_from_strings(
    catalog_yaml: &str,
    request_yaml: &str,
) -> Result<FoundationPracticalIdentifierResponseV0752> {
    let service = FoundationPracticalIdentifierServiceV0752::from_catalog_yaml_str(catalog_yaml)?;
    let request = FoundationPracticalIdentifierRequestV0752::from_yaml_str(request_yaml)?;
    service.identify(&request)
}

fn run_cli(cli: &CliArgs) -> Result<FoundationPracticalIdentifierResponseV0752> {
    if cli.output_yaml == cli.catalog || cli.output_yaml == cli.request {
        bail!("v0.75.3 refuses to overwrite an input file with --output-yaml");
    }
    if let Some(output_tsv) = &cli.output_tsv {
        if output_tsv == &cli.catalog
            || output_tsv == &cli.request
            || output_tsv == &cli.output_yaml
        {
            bail!("v0.75.3 output paths must be distinct from inputs and from each other");
        }
    }

    let catalog_yaml = fs::read_to_string(&cli.catalog)
        .with_context(|| format!("failed to read catalog {}", cli.catalog.display()))?;
    let request_yaml = fs::read_to_string(&cli.request)
        .with_context(|| format!("failed to read request {}", cli.request.display()))?;
    let response = identify_from_strings(&catalog_yaml, &request_yaml)?;
    write_response_yaml(&cli.output_yaml, &response)?;
    if let Some(path) = &cli.output_tsv {
        write_response_tsv(path, &response)?;
    }
    Ok(response)
}

fn self_test() -> Result<()> {
    let response = identify_from_strings(FIXTURE_CATALOG, FIXTURE_REQUEST)?;
    if response.schema != FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752
        || response.query_id != "codon-smoke-query"
        || response.candidate_pool != FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751
        || response.catalog_size != 1
        || response.candidates_scored != 1
        || response.returned_hits != 1
        || response.hits.len() != 1
        || response.hits[0].rank != 1
        || response.hits[0].candidate_index != 0
        || response.hits[0].key != "PEPTIDE|z2"
        || response.hits[0].charge != 2
    {
        bail!("v0.75.3 embedded production fixture produced an unexpected response");
    }
    let response_yaml = response.to_yaml_string()?;
    if !response_yaml.contains("schema: redeem.foundation.practical_identifier.v0752")
        || !response_yaml.contains("candidate_pool: 256")
        || !response_yaml.contains("query_id: codon-smoke-query")
    {
        bail!("v0.75.3 response YAML contract self-test failed");
    }
    let hits_tsv = String::from_utf8(response_hits_tsv(&response)?)?;
    if !hits_tsv.starts_with("rank\tcandidate_index\tkey\tcharge\t")
        || !hits_tsv.contains("1\t0\tPEPTIDE|z2\t2\t")
    {
        bail!("v0.75.3 hits TSV contract self-test failed");
    }
    println!("v0753_practical_identifier_cli_self_test=PASS");
    println!("v0753_schema={FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752}");
    println!("v0753_candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
    println!("v0753_protected_data_required=NO");
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() == 2 && args[1] == "--self-test" {
        return self_test();
    }
    if args.len() == 2 && args[1] == "--version" {
        println!("{V0753_VERSION}");
        println!("runtime={V0753_RUNTIME}");
        println!("schema={FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752}");
        println!("candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
        return Ok(());
    }
    if args.len() == 2 && matches!(args[1].as_str(), "--help" | "-h") {
        println!("{}", usage());
        return Ok(());
    }

    let cli = parse_cli_args(&args)?;
    let response = run_cli(&cli)?;
    println!("v0753_version={V0753_VERSION}");
    println!("v0753_runtime={V0753_RUNTIME}");
    println!("schema={}", response.schema);
    println!("query_id={}", response.query_id);
    println!("catalog_fingerprint={}", response.catalog_fingerprint);
    println!("catalog_size={}", response.catalog_size);
    println!("candidate_pool={}", response.candidate_pool);
    println!("candidates_scored={}", response.candidates_scored);
    println!("returned_hits={}", response.returned_hits);
    println!("target_forcing=NO");
    println!("v070_checkpoint_required=NO");
    println!("v052_checkpoint_required=NO");
    println!("protected_data_required=NO");
    println!("response_yaml={}", cli.output_yaml.display());
    if let Some(path) = &cli.output_tsv {
        println!("hits_tsv={}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        std::iter::once("foundation_identify_practical_v0753")
            .chain(items.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn embedded_fixture_exercises_production_service_contract() {
        self_test().unwrap();
    }

    #[test]
    fn cli_parser_accepts_named_outputs_and_rejects_unknown_flags() {
        let parsed = parse_cli_args(&args(&[
            "--catalog",
            "catalog.yaml",
            "--request",
            "request.yaml",
            "--output-yaml",
            "response.yaml",
            "--output-tsv",
            "hits.tsv",
        ]))
        .unwrap();
        assert_eq!(parsed.catalog, PathBuf::from("catalog.yaml"));
        assert_eq!(parsed.request, PathBuf::from("request.yaml"));
        assert_eq!(parsed.output_yaml, PathBuf::from("response.yaml"));
        assert_eq!(parsed.output_tsv, Some(PathBuf::from("hits.tsv")));

        assert!(parse_cli_args(&args(&[
            "--catalog",
            "catalog.yaml",
            "--request",
            "request.yaml",
            "--output-yaml",
            "response.yaml",
            "--candidate-pool",
            "64",
        ]))
        .is_err());
    }

    #[test]
    fn tsv_contract_is_deterministic_for_embedded_fixture() {
        let response = identify_from_strings(FIXTURE_CATALOG, FIXTURE_REQUEST).unwrap();
        let first = response_hits_tsv(&response).unwrap();
        let second = response_hits_tsv(&response).unwrap();
        assert_eq!(first, second);
        let text = String::from_utf8(first).unwrap();
        assert!(text.starts_with("rank\tcandidate_index\tkey\tcharge\t"));
        assert!(text.contains("1\t0\tPEPTIDE|z2\t2\t"));
    }

    #[test]
    fn cli_refuses_scientific_candidate_pool_override() {
        let error = parse_cli_args(&args(&[
            "--catalog",
            "catalog.yaml",
            "--request",
            "request.yaml",
            "--output-yaml",
            "response.yaml",
            "--candidate-pool",
            "64",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("unknown v0.75.3 CLI argument"));
    }
}
