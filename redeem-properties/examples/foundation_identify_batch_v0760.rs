use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    FoundationPracticalIdentifierBatchRequestV0760,
    FoundationPracticalIdentifierBatchResponseV0760,
    FoundationPracticalIdentifierBatchServiceV0760, FoundationPracticalIdentifierCatalogV0752,
    FoundationPracticalIdentifierSearchSpaceV0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_RESPONSE_SCHEMA_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const V0760_VERSION: &str = "v0.76.0-practical-identifier-batch";
const V0760_RUNTIME: &str = "search_space_materialization_plus_ordered_batch_identification";
const FIXTURE_SEARCH_SPACE: &str =
    include_str!("../tests/fixtures/foundation_identifier_v0760/search_space.yaml");
const FIXTURE_BATCH: &str =
    include_str!("../tests/fixtures/foundation_identifier_v0760/batch.yaml");

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    MaterializeCatalog {
        search_space: PathBuf,
        output_catalog: PathBuf,
    },
    IdentifyBatch {
        catalog: PathBuf,
        batch: PathBuf,
        output_yaml: PathBuf,
        output_tsv: Option<PathBuf>,
    },
}

fn usage() -> &'static str {
    "usage:\n  foundation_identify_batch_v0760 materialize-catalog --search-space SEARCH_SPACE.yaml --output-catalog CATALOG.yaml\n  foundation_identify_batch_v0760 identify-batch --catalog CATALOG.yaml --batch BATCH.yaml --output-yaml RESPONSE.yaml [--output-tsv HITS.tsv]\n  foundation_identify_batch_v0760 --self-test\n  foundation_identify_batch_v0760 --version"
}

fn parse_named_values(args: &[String], allowed: &[&str]) -> Result<BTreeMap<String, String>> {
    let mut values = BTreeMap::<String, String>::new();
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index].as_str();
        if !allowed.contains(&flag) {
            bail!("unknown v0.76.0 CLI argument {flag:?}\n{}", usage());
        }
        if values.contains_key(flag) {
            bail!("duplicate v0.76.0 CLI argument {flag:?}\n{}", usage());
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
        "materialize-catalog" => {
            let values = parse_named_values(&args[2..], &["--search-space", "--output-catalog"])?;
            Ok(Command::MaterializeCatalog {
                search_space: required_path(&values, "--search-space")?,
                output_catalog: required_path(&values, "--output-catalog")?,
            })
        }
        "identify-batch" => {
            let values = parse_named_values(
                &args[2..],
                &["--catalog", "--batch", "--output-yaml", "--output-tsv"],
            )?;
            Ok(Command::IdentifyBatch {
                catalog: required_path(&values, "--catalog")?,
                batch: required_path(&values, "--batch")?,
                output_yaml: required_path(&values, "--output-yaml")?,
                output_tsv: values.get("--output-tsv").map(PathBuf::from),
            })
        }
        _ => bail!("unknown v0.76.0 subcommand {subcommand:?}\n{}", usage()),
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

fn write_text(path: &Path, body: &str, description: &str) -> Result<()> {
    ensure_output_parent(path)?;
    fs::write(path, body)
        .with_context(|| format!("failed to write {description} {}", path.display()))
}

fn batch_hits_tsv(response: &FoundationPracticalIdentifierBatchResponseV0760) -> Result<Vec<u8>> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_writer(Vec::new());
    writer.write_record([
        "query_id",
        "rank",
        "candidate_index",
        "key",
        "charge",
        "theoretical_neutral_mass",
        "absolute_neutral_mass_error",
        "geometry_score",
    ])?;
    for query in &response.responses {
        for hit in &query.hits {
            writer.write_record([
                query.query_id.clone(),
                hit.rank.to_string(),
                hit.candidate_index.to_string(),
                hit.key.clone(),
                hit.charge.to_string(),
                format!("{:.12}", hit.theoretical_neutral_mass),
                format!("{:.12}", hit.absolute_neutral_mass_error),
                format!("{:.12}", hit.geometry_score),
            ])?;
        }
    }
    writer
        .into_inner()
        .map_err(|error| anyhow::anyhow!(error.into_error()))
}

fn run_materialize(search_space_path: &Path, output_catalog: &Path) -> Result<()> {
    if search_space_path == output_catalog {
        bail!("v0.76.0 refuses to overwrite --search-space with --output-catalog");
    }
    let search_space_yaml = fs::read_to_string(search_space_path).with_context(|| {
        format!(
            "failed to read search space {}",
            search_space_path.display()
        )
    })?;
    let search_space =
        FoundationPracticalIdentifierSearchSpaceV0760::from_yaml_str(&search_space_yaml)?;
    let catalog = search_space.materialize_catalog()?;

    // Construct the frozen service once so invalid chemistry/duplicate candidate keys fail before output.
    let service = FoundationPracticalIdentifierBatchServiceV0760::from_catalog(catalog.clone())?;
    write_text(
        output_catalog,
        &catalog.to_yaml_string()?,
        "materialized catalog",
    )?;

    println!("v0760_version={V0760_VERSION}");
    println!("v0760_runtime={V0760_RUNTIME}");
    println!("search_space_schema={FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760}");
    println!("catalog_size={}", service.catalog_size());
    println!("catalog_fingerprint={}", service.catalog_fingerprint());
    println!("candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
    println!("target_forcing=NO");
    println!("protected_data_required=NO");
    println!("output_catalog={}", output_catalog.display());
    Ok(())
}

fn run_identify_batch(
    catalog_path: &Path,
    batch_path: &Path,
    output_yaml: &Path,
    output_tsv: Option<&Path>,
) -> Result<FoundationPracticalIdentifierBatchResponseV0760> {
    if output_yaml == catalog_path || output_yaml == batch_path {
        bail!("v0.76.0 refuses to overwrite an input file with --output-yaml");
    }
    if let Some(path) = output_tsv {
        if path == catalog_path || path == batch_path || path == output_yaml {
            bail!("v0.76.0 output paths must be distinct from inputs and from each other");
        }
    }

    let catalog_yaml = fs::read_to_string(catalog_path)
        .with_context(|| format!("failed to read catalog {}", catalog_path.display()))?;
    let batch_yaml = fs::read_to_string(batch_path)
        .with_context(|| format!("failed to read batch {}", batch_path.display()))?;
    let catalog = FoundationPracticalIdentifierCatalogV0752::from_yaml_str(&catalog_yaml)?;
    let batch = FoundationPracticalIdentifierBatchRequestV0760::from_yaml_str(&batch_yaml)?;
    let service = FoundationPracticalIdentifierBatchServiceV0760::from_catalog(catalog)?;
    let response = service.identify_batch(&batch)?;

    write_text(
        output_yaml,
        &response.to_yaml_string()?,
        "batch response YAML",
    )?;
    if let Some(path) = output_tsv {
        ensure_output_parent(path)?;
        fs::write(path, batch_hits_tsv(&response)?)
            .with_context(|| format!("failed to write batch hits TSV {}", path.display()))?;
    }

    println!("v0760_version={V0760_VERSION}");
    println!("v0760_runtime={V0760_RUNTIME}");
    println!("batch_schema={FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760}");
    println!("response_schema={}", response.schema);
    println!("catalog_fingerprint={}", response.catalog_fingerprint);
    println!("catalog_size={}", response.catalog_size);
    println!("request_count={}", response.request_count);
    println!(
        "total_candidates_scored={}",
        response.total_candidates_scored
    );
    println!("total_returned_hits={}", response.total_returned_hits);
    println!("candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
    println!("target_forcing=NO");
    println!("protected_data_required=NO");
    println!("output_yaml={}", output_yaml.display());
    if let Some(path) = output_tsv {
        println!("output_tsv={}", path.display());
    }
    Ok(response)
}

fn self_test() -> Result<()> {
    let search_space =
        FoundationPracticalIdentifierSearchSpaceV0760::from_yaml_str(FIXTURE_SEARCH_SPACE)?;
    let catalog = search_space.materialize_catalog()?;
    if catalog.candidates.len() != 3
        || catalog.candidates[0].key != "pep-a|z2"
        || catalog.candidates[1].key != "pep-a|z3"
        || catalog.candidates[2].key != "pep-b|z2"
    {
        bail!("v0.76.0 search-space materialization self-test failed");
    }
    let service = FoundationPracticalIdentifierBatchServiceV0760::from_catalog(catalog)?;
    let batch = FoundationPracticalIdentifierBatchRequestV0760::from_yaml_str(FIXTURE_BATCH)?;
    let response = service.identify_batch(&batch)?;
    if response.schema != FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_RESPONSE_SCHEMA_V0760
        || response.catalog_size != 3
        || response.request_count != 2
        || response.total_candidates_scored != 3
        || response.total_returned_hits != 3
        || response.responses.len() != 2
        || response.responses[0].query_id != "batch-q2"
        || response.responses[1].query_id != "batch-q3"
        || response
            .responses
            .iter()
            .any(|item| item.candidate_pool != FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751)
    {
        bail!("v0.76.0 batch identification self-test failed");
    }
    let tsv = String::from_utf8(batch_hits_tsv(&response)?)?;
    if !tsv.starts_with("query_id\trank\tcandidate_index\tkey\tcharge\t")
        || !tsv.contains("batch-q2\t1\t0\tpep-a|z2\t2\t")
        || !tsv.contains("batch-q3\t1\t1\tpep-a|z3\t3\t")
    {
        bail!("v0.76.0 flattened TSV self-test failed");
    }

    println!("v0760_practical_identifier_batch_self_test=PASS");
    println!(
        "v0760_search_space_schema={FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760}"
    );
    println!("v0760_batch_schema={FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760}");
    println!("v0760_response_schema={FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_RESPONSE_SCHEMA_V0760}");
    println!("v0760_candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
    println!("v0760_protected_data_required=NO");
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() == 2 && args[1] == "--self-test" {
        return self_test();
    }
    if args.len() == 2 && args[1] == "--version" {
        println!("{V0760_VERSION}");
        println!("runtime={V0760_RUNTIME}");
        println!("search_space_schema={FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760}");
        println!("batch_schema={FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760}");
        println!("candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
        return Ok(());
    }
    if args.len() == 2 && matches!(args[1].as_str(), "--help" | "-h") {
        println!("{}", usage());
        return Ok(());
    }

    match parse_cli_args(&args)? {
        Command::MaterializeCatalog {
            search_space,
            output_catalog,
        } => run_materialize(&search_space, &output_catalog),
        Command::IdentifyBatch {
            catalog,
            batch,
            output_yaml,
            output_tsv,
        } => {
            run_identify_batch(&catalog, &batch, &output_yaml, output_tsv.as_deref())?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        std::iter::once("foundation_identify_batch_v0760")
            .chain(items.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn embedded_fixture_exercises_materialization_and_batch_contract() {
        self_test().unwrap();
    }

    #[test]
    fn parser_accepts_both_subcommands_and_rejects_tuning_flags() {
        assert_eq!(
            parse_cli_args(&args(&[
                "materialize-catalog",
                "--search-space",
                "search.yaml",
                "--output-catalog",
                "catalog.yaml",
            ]))
            .unwrap(),
            Command::MaterializeCatalog {
                search_space: PathBuf::from("search.yaml"),
                output_catalog: PathBuf::from("catalog.yaml"),
            }
        );

        assert!(parse_cli_args(&args(&[
            "identify-batch",
            "--catalog",
            "catalog.yaml",
            "--batch",
            "batch.yaml",
            "--output-yaml",
            "response.yaml",
            "--candidate-pool",
            "64",
        ]))
        .is_err());
    }

    #[test]
    fn flattened_tsv_is_deterministic() {
        let search_space =
            FoundationPracticalIdentifierSearchSpaceV0760::from_yaml_str(FIXTURE_SEARCH_SPACE)
                .unwrap();
        let service =
            FoundationPracticalIdentifierBatchServiceV0760::from_search_space(&search_space)
                .unwrap();
        let batch =
            FoundationPracticalIdentifierBatchRequestV0760::from_yaml_str(FIXTURE_BATCH).unwrap();
        let response = service.identify_batch(&batch).unwrap();
        assert_eq!(
            batch_hits_tsv(&response).unwrap(),
            batch_hits_tsv(&response).unwrap()
        );
    }
}
