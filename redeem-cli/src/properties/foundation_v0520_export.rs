//! Opt-in RT and annotated-fragment adapter for the *original*, separately executed
//! v0.52 reference exporter. It never approximates v0.52 weights or equations.
use crate::properties::inference::input::PropertyInferenceConfig;
use anyhow::{Context, Result, ensure};
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use redeem_properties::foundation::{FoundationBenchmarkManifest, FoundationPartition};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const PREFIX: usize = 2048;
const PRECURSOR_HEADER: [&str; 16] = [
    "record_index",
    "partition",
    "source_id",
    "sequence",
    "mods",
    "mod_sites",
    "charge",
    "nce",
    "instrument",
    "selected_rt",
    "selected_ccs",
    "selected_ms2",
    "target_rt",
    "foundation_rt",
    "target_ccs",
    "foundation_ccs",
];
const FRAGMENT_HEADER: [&str; 9] = [
    "record_index",
    "sequence",
    "cleavage_index",
    "channel",
    "charged_frag_type",
    "fragment_number",
    "product_mz",
    "target_intensity",
    "foundation_intensity",
];

#[derive(Debug)]
struct Precursor {
    ordinal: usize,
    original_index: usize,
    sequence: String,
    charge: String,
    rt: String,
    selected_ms2: bool,
}

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    ensure!(
        config.foundation_partition == "validation",
        "v0.52 reference export permits Validation only"
    );
    ensure!(
        !config.foundation_spectra_only,
        "v0.52 reference export does not support spectra_only"
    );
    ensure!(
        matches!(config.foundation_max_records, Some(PREFIX)),
        "v0.52 reference export requires foundation_max_records=2048"
    );
    ensure!(
        !config.inference_data.trim().is_empty(),
        "prepared run YAML must be provided"
    );
    ensure!(
        Path::new(&config.inference_data).is_file(),
        "prepared run YAML not found"
    );
    let checkpoint = Path::new(&config.model_path);
    ensure!(
        checkpoint.join("metadata.yaml").is_file()
            && checkpoint.join("model.safetensors").is_file(),
        "v0.52 reference checkpoint is incomplete"
    );
    let export_dir = config
        .foundation_reference_export_dir
        .as_deref()
        .context("foundation_reference_export_dir is required")?;
    let benchmark_path = config
        .foundation_reference_benchmark
        .as_deref()
        .context("foundation_reference_benchmark is required")?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(benchmark_path)?;
    // A reference export cannot be reconciled to a different TRAIN/Validation split.
    let metadata = fs::read_to_string(checkpoint.join("metadata.yaml"))?;
    ensure!(
        metadata_scalar(&metadata, "version") == Some("520"),
        "checkpoint is not original v0.52"
    );
    ensure!(
        metadata_scalar(&metadata, "objective") == Some("v0520_mobility_aware_pair_representation"),
        "incorrect reference checkpoint objective"
    );
    // The benchmark dataset hash describes records, while the corpus hash also
    // incorporates source provenance. Their values are not interchangeable.
    validate_reference_fingerprints(&metadata, benchmark.manifest_fingerprint())?;

    let prefix = benchmark.partition_indices(FoundationPartition::Validation);
    ensure!(
        prefix.len() >= PREFIX,
        "benchmark has fewer than 2048 Validation records"
    );
    let sequence: HashMap<usize, &str> = benchmark
        .entries
        .iter()
        .map(|entry| (entry.record_index, entry.sequence.as_str()))
        .collect();
    let indices: HashMap<usize, usize> = prefix
        .iter()
        .take(PREFIX)
        .enumerate()
        .map(|(ordinal, &original)| (original, ordinal))
        .collect();
    let original_precursors = Path::new(export_dir).join("foundation_comparison_precursors.tsv");
    let original_fragments = Path::new(export_dir).join("foundation_comparison_fragments.tsv");
    let destination = PathBuf::from(&config.output_file);
    let fragment_destination =
        PathBuf::from(format!("{}.annotated_fragments.tsv", config.output_file));
    ensure!(
        !destination.exists() && !fragment_destination.exists(),
        "reference output already exists; refusing overwrite"
    );
    ensure!(
        destination != original_precursors && fragment_destination != original_fragments,
        "reference input and output paths overlap"
    );

    let (rows, selected) = validate_precursors(&original_precursors, &indices, &sequence)?;
    validate_fragments(&original_fragments, &rows, &selected)?;

    let mut output = WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&destination)
        .with_context(|| format!("create {}", destination.display()))?;
    output.write_record([
        "record_index",
        "sequence",
        "charge",
        "predicted_rt",
        "predicted_ccs",
        "predicted_ms2",
        "inverse_mean_nll",
        "generation_succeeded",
        "generated_sequence",
        "generated_modifications",
        "corpus_record_index",
        "ms2_annotation_scope",
    ])?;
    for row in rows.values() {
        output.write_record([
            row.ordinal.to_string(),
            row.sequence.clone(),
            row.charge.clone(),
            row.rt.clone(),
            String::new(),
            String::new(),
            String::new(),
            "false".into(),
            String::new(),
            String::new(),
            row.original_index.to_string(),
            if row.selected_ms2 {
                "annotated_fragments_only"
            } else {
                "none"
            }
            .into(),
        ])?;
    }
    output.flush()?;
    fs::copy(&original_fragments, &fragment_destination).with_context(|| {
        format!(
            "copy validated reference fragments to {}",
            fragment_destination.display()
        )
    })?;
    println!("foundation_inference=PASS");
    println!("foundation_backend=v0520_reference_export");
    println!("partition=validation");
    println!("records={}", rows.len());
    println!("output={}", destination.display());
    println!("annotated_ms2={}", fragment_destination.display());
    println!("historical_train_holdout_previously_consumed=YES");
    println!("train_holdout_read_by_this_adapter=NO");
    Ok(())
}

fn validate_reference_fingerprints(metadata: &str, manifest_fingerprint: u64) -> Result<()> {
    let corpus = metadata_scalar(metadata, "corpus_fingerprint")
        .context("missing historical reference corpus fingerprint")?;
    let corpus_hash = corpus.strip_prefix("fnv1a64:")
        .context("invalid historical reference corpus fingerprint")?;
    ensure!(corpus_hash.len() == 16 && u64::from_str_radix(corpus_hash, 16).is_ok(),
        "invalid historical reference corpus fingerprint");
    let benchmark_hash = format!("fnv1a64:{manifest_fingerprint:016x}");
    ensure!(metadata_scalar(metadata, "benchmark_manifest_fingerprint") == Some(benchmark_hash.as_str()),
        "reference benchmark fingerprint mismatch");
    Ok(())
}

fn metadata_scalar<'a>(yaml: &'a str, key: &str) -> Option<&'a str> {
    yaml.lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name == key).then(|| value.trim().trim_matches('"').trim_matches('\''))
        })
        .next()
}

fn check_header(actual: &StringRecord, expected: &[&str]) -> Result<()> {
    ensure!(
        actual.iter().eq(expected.iter().copied()),
        "unexpected v0.52 reference export columns: {actual:?}"
    );
    Ok(())
}

fn validate_precursors(
    path: &Path,
    indices: &HashMap<usize, usize>,
    sequences: &HashMap<usize, &str>,
) -> Result<(BTreeMap<usize, Precursor>, HashSet<usize>)> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    check_header(reader.headers()?, &PRECURSOR_HEADER)?;
    let mut rows = BTreeMap::new();
    let mut selected = HashSet::new();
    for record in reader.records() {
        let r = record?;
        let original: usize = r[0].parse().context("invalid precursor original index")?;
        let ordinal = *indices
            .get(&original)
            .context("v0.52 record outside frozen Validation prefix")?;
        ensure!(
            &r[1] == "VALIDATION",
            "v0.52 export contains a non-Validation precursor"
        );
        ensure!(
            sequences.get(&original).copied() == Some(&r[3]),
            "v0.52 sequence does not match benchmark"
        );
        ensure!(
            r[4].is_empty() && r[5].is_empty(),
            "v0.52 export unexpectedly contains modified peptides"
        );
        ensure!(
            matches!(r.get(9), Some("YES" | "NO")),
            "invalid selected_rt flag"
        );
        ensure!(
            matches!(r.get(11), Some("YES" | "NO")),
            "invalid selected_ms2 flag"
        );
        let rt = r[13]
            .parse::<f32>()
            .context("invalid v0.52 RT prediction")?;
        ensure!(rt.is_finite(), "nonfinite v0.52 RT prediction");
        let selected_ms2 = &r[11] == "YES";
        if selected_ms2 {
            selected.insert(original);
        }
        ensure!(
            rows.insert(
                ordinal,
                Precursor {
                    ordinal,
                    original_index: original,
                    sequence: r[3].to_owned(),
                    charge: if &r[6] == "NA" {
                        String::new()
                    } else {
                        r[6].to_owned()
                    },
                    rt: r[13].to_owned(),
                    selected_ms2,
                }
            )
            .is_none(),
            "duplicate v0.52 precursor record"
        );
    }
    ensure!(
        !rows.is_empty() && !selected.is_empty(),
        "v0.52 export has no selected RT/MS2 records"
    );
    Ok((rows, selected))
}

fn validate_fragments(
    path: &Path,
    rows: &BTreeMap<usize, Precursor>,
    selected: &HashSet<usize>,
) -> Result<()> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    check_header(reader.headers()?, &FRAGMENT_HEADER)?;
    let by_original: HashMap<usize, &Precursor> =
        rows.values().map(|row| (row.original_index, row)).collect();
    let mut seen = HashSet::new();
    for record in reader.records() {
        let r = record?;
        let original: usize = r[0].parse().context("invalid fragment original index")?;
        ensure!(
            selected.contains(&original),
            "fragment outside selected MS2 cohort"
        );
        let precursor = by_original
            .get(&original)
            .context("missing selected MS2 precursor")?;
        ensure!(
            &r[1] == precursor.sequence,
            "fragment peptide mismatches precursor"
        );
        let cleavage: usize = r[2].parse().context("invalid fragment cleavage")?;
        let channel: usize = r[3].parse().context("invalid fragment channel")?;
        ensure!(
            cleavage + 1 < precursor.sequence.len() && channel < 4,
            "invalid historical fragment coordinate"
        );
        ensure!(
            r[8].parse::<f32>().is_ok_and(f32::is_finite),
            "nonfinite fragment prediction"
        );
        ensure!(
            seen.insert((original, cleavage, channel)),
            "duplicate annotated historical fragment"
        );
    }
    ensure!(
        !seen.is_empty(),
        "v0.52 reference export has no annotated MS2 predictions"
    );
    for index in selected {
        ensure!(
            seen.iter().any(|(found, _, _)| found == index),
            "selected MS2 precursor has no annotated predictions"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_fingerprints_distinguish_corpus_from_benchmark_dataset() {
        // Historical and production checkpoint corpus hashes matched in job 62823987,
        // but benchmark.dataset_fingerprint hashes a different object.
        let metadata = "corpus_fingerprint: fnv1a64:a2a6f57d31064ba6\n\
benchmark_manifest_fingerprint: fnv1a64:2133c039625f77df\n";
        assert!(validate_reference_fingerprints(metadata, 0x2133c039625f77df).is_ok());
        assert!(validate_reference_fingerprints(metadata, 0).is_err());
        assert!(validate_reference_fingerprints(
            "corpus_fingerprint: corrupted\nbenchmark_manifest_fingerprint: fnv1a64:2133c039625f77df\n",
            0x2133c039625f77df,
        ).is_err());
    }

    #[test]
    fn reference_export_adapter_preserves_rt_and_rejects_mismatched_sequences() {
        let temp = tempfile::tempdir().unwrap();
        let precursors = temp.path().join("precursors.tsv");
        let fragments = temp.path().join("fragments.tsv");
        let mut input = PRECURSOR_HEADER.join("\t") + "\n";
        input.push_str("42\tVALIDATION\tsource\tPEPTIDEK\t\t\t2\t20\tQE\tYES\tNO\tYES\t10\t11.12500000\tNA\tNA\n");
        fs::write(&precursors, &input).unwrap();
        fs::write(
            &fragments,
            format!(
                "{}\n42\tPEPTIDEK\t2\t0\tb_z1\t3\t300\t0.4\t0.8\n",
                FRAGMENT_HEADER.join("\t")
            ),
        )
        .unwrap();
        let indices = HashMap::from([(42, 5)]);
        let sequences = HashMap::from([(42, "PEPTIDEK")]);
        let (rows, selected) = validate_precursors(&precursors, &indices, &sequences).unwrap();
        assert_eq!(rows[&5].rt, "11.12500000");
        validate_fragments(&fragments, &rows, &selected).unwrap();
        let wrong_sequences = HashMap::from([(42, "WRONGPEP")]);
        assert!(validate_precursors(&precursors, &indices, &wrong_sequences).is_err());
        fs::write(
            &fragments,
            format!(
                "{}\n42\tWRONGPEP\t2\t0\tb_z1\t3\t300\t0.4\t0.8\n",
                FRAGMENT_HEADER.join("\t")
            ),
        )
        .unwrap();
        assert!(validate_fragments(&fragments, &rows, &selected).is_err());
    }

    #[test]
    fn metadata_field_parser_is_exact() {
        let metadata = "version: 520\ncorpus_fingerprint: fnv1a64:a2a6f57d31064ba6\n";
        assert_eq!(metadata_scalar(metadata, "version"), Some("520"));
        assert_eq!(
            metadata_scalar(metadata, "corpus_fingerprint"),
            Some("fnv1a64:a2a6f57d31064ba6")
        );
        assert_eq!(metadata_scalar(metadata, "other"), None);
    }
}
