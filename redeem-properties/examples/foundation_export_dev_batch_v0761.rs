//! v0.76.1 actual-DEV adapter for the frozen practical-identifier production path.
//!
//! This executable is engineering regression infrastructure only. It reconstructs the exact
//! source-closed DEV candidate universe and frozen 512-query cohort used by v0.75, then exports
//! those actual records into the v0.76 production search-space and batch-request contracts.
//! It does not rank candidates, tune parameters, or access protected TRAIN-HOLDOUT / historical TEST.

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    foundation_peptidoform_neutral_mass, foundation_precursor_neutral_mass, load_foundation_corpus,
    read_foundation_training_run_config, FoundationBenchmarkManifest, FoundationModification,
    FoundationModificationSite, FoundationPartition,
    FoundationPracticalIdentifierBatchRequestV0760,
    FoundationPracticalIdentifierModificationSiteV0752,
    FoundationPracticalIdentifierModificationV0752, FoundationPracticalIdentifierPeakV0752,
    FoundationPracticalIdentifierRequestV0752, FoundationPracticalIdentifierSearchSpaceEntryV0760,
    FoundationPracticalIdentifierSearchSpaceV0760, FoundationSpectrum, FoundationTrainingRecord,
    FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const V0761_VERSION: u32 = 761;
const V0761_OBJECTIVE: &str = "v0761_actual_dev_production_path_equivalence_export";
const V070_SELECTION_SEED: u64 = 20_261_070;
const V070_DEV_IDENTITIES: usize = 2048;
const V073_QUERY_SEED: u64 = 20_261_073;
const V0761_DEV_QUERIES: usize = 512;
const V0761_MAX_SEQUENCE_LEN: usize = 64;
const V0761_EXPECTED_CORPUS_FINGERPRINT: &str = "fnv1a64:a2a6f57d31064ba6";
const V0761_EXPECTED_BENCHMARK_FINGERPRINT: &str = "fnv1a64:2133c039625f77df";
const V0761_EXPECTED_PARENT_DEV_FINGERPRINT: &str = "fnv1a64:2aa9a31055c720a6";
const V0761_EXPECTED_CANDIDATE_UNIVERSE: usize = 63_332;
const V0761_EXPECTED_CANDIDATE_FINGERPRINT: &str = "fnv1a64:a5699533553116e8";
const V0761_EXPECTED_QUERY_COHORT_FINGERPRINT: &str = "fnv1a64:bc6d4b8b3090af2b";

#[derive(Debug, Clone)]
struct AlignmentGroup {
    key: String,
    peptidoform: String,
    sequence: String,
    charge: i32,
    record_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct CandidateIdentity {
    record_index: usize,
    exact_key: String,
    il_key: String,
    charge: i32,
    observed_neutral_mass: f64,
    candidate_neutral_mass: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ExportMetadata {
    version: u32,
    objective: String,
    evaluation_partition: String,
    protected_evaluation: bool,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_dev_identity_fingerprint: String,
    candidate_universe: usize,
    candidate_universe_fingerprint: String,
    query_count: usize,
    query_cohort_fingerprint: String,
    candidate_pool: usize,
    target_forcing: bool,
    dev_partition_used_for_selection: bool,
    train_holdout_accessed: bool,
    historical_validation_accessed: bool,
    historical_test_accessed: bool,
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 2 && args[1] == "--self-test" {
        self_test()?;
        println!("v0761_actual_dev_export_self_test=PASS");
        println!("v0761_candidate_pool={FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
        println!("v0761_protected_data_required=NO");
        return Ok(());
    }
    if args.len() != 3 {
        bail!("usage: foundation_export_dev_batch_v0761 RUN_V0260.yaml OUTPUT_DIR");
    }

    let training_yaml = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    if output_root.exists() {
        bail!("v0.76.1 export output directory must be fresh: {output_root:?}");
    }

    let run = read_foundation_training_run_config(&training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let corpus_fingerprint = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let benchmark_fingerprint = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    require_eq(
        "corpus fingerprint",
        &corpus_fingerprint,
        V0761_EXPECTED_CORPUS_FINGERPRINT,
    )?;
    require_eq(
        "benchmark fingerprint",
        &benchmark_fingerprint,
        V0761_EXPECTED_BENCHMARK_FINGERPRINT,
    )?;

    let dev_groups = build_alignment_groups(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        V0761_MAX_SEQUENCE_LEN,
    )?;
    let identity_seed = V070_SELECTION_SEED ^ 0x7000_d3f0_a11e_0001;
    let selected_identities = select_dev_identities(
        &corpus.records,
        &dev_groups,
        V070_DEV_IDENTITIES,
        identity_seed,
    )?;
    if selected_identities.len() != V070_DEV_IDENTITIES {
        bail!(
            "v0.76.1 expected {V070_DEV_IDENTITIES} frozen parent DEV identities, observed {}",
            selected_identities.len()
        );
    }
    let parent_fingerprint = format!(
        "fnv1a64:{:016x}",
        identity_fingerprint(&selected_identities)
    );
    require_eq(
        "parent DEV identity fingerprint",
        &parent_fingerprint,
        V0761_EXPECTED_PARENT_DEV_FINGERPRINT,
    )?;

    let all_dev_identities = build_all_partition_identities(
        &corpus.records,
        &dev_groups,
        identity_seed.rotate_left(17),
    )?;
    if all_dev_identities.len() != V0761_EXPECTED_CANDIDATE_UNIVERSE {
        bail!(
            "v0.76.1 candidate-universe size mismatch: current={} expected={}",
            all_dev_identities.len(),
            V0761_EXPECTED_CANDIDATE_UNIVERSE
        );
    }
    let candidate_fingerprint =
        format!("fnv1a64:{:016x}", identity_fingerprint(&all_dev_identities));
    require_eq(
        "candidate-universe fingerprint",
        &candidate_fingerprint,
        V0761_EXPECTED_CANDIDATE_FINGERPRINT,
    )?;

    let full_index = full_identity_index(&all_dev_identities)?;
    let selected_query_indices = deterministic_eval_queries(
        &selected_identities,
        V0761_DEV_QUERIES,
        V073_QUERY_SEED ^ 0x7300_d3f0_0000_0001,
    );
    let query_identities = selected_query_indices
        .iter()
        .map(|&index| selected_identities[index].clone())
        .collect::<Vec<_>>();
    let query_fingerprint = format!("fnv1a64:{:016x}", identity_fingerprint(&query_identities));
    require_eq(
        "query-cohort fingerprint",
        &query_fingerprint,
        V0761_EXPECTED_QUERY_COHORT_FINGERPRINT,
    )?;

    let mut full_query_indices = Vec::with_capacity(selected_query_indices.len());
    for &selected_index in &selected_query_indices {
        let selected = &selected_identities[selected_index];
        let full_index_value = *full_index.get(&selected.exact_key).with_context(|| {
            format!(
                "v0.76.1 frozen query {} absent from actual DEV candidate universe",
                selected.exact_key
            )
        })?;
        if all_dev_identities[full_index_value].record_index != selected.record_index {
            bail!(
                "v0.76.1 frozen query record drift for {}: selected={} full={}",
                selected.exact_key,
                selected.record_index,
                all_dev_identities[full_index_value].record_index
            );
        }
        full_query_indices.push(full_index_value);
    }

    let search_space = build_search_space(&corpus.records, &dev_groups, &all_dev_identities)?;
    let materialized = search_space.materialize_catalog()?;
    validate_materialized_catalog(&materialized.candidates, &all_dev_identities)?;

    let batch = build_batch_requests(&corpus.records, &all_dev_identities, &full_query_indices)?;

    fs::create_dir_all(&output_root)?;
    fs::write(
        output_root.join("search_space.yaml"),
        search_space.to_yaml_string()?,
    )?;
    fs::write(output_root.join("batch.yaml"), batch.to_yaml_string()?)?;
    write_candidate_catalog(
        &output_root.join("candidate_catalog.tsv"),
        &all_dev_identities,
    )?;
    write_query_manifest(
        &output_root.join("query_manifest.tsv"),
        &all_dev_identities,
        &full_query_indices,
        &selected_query_indices,
    )?;

    let metadata = ExportMetadata {
        version: V0761_VERSION,
        objective: V0761_OBJECTIVE.to_string(),
        evaluation_partition: "DEV_ENGINEERING_EQUIVALENCE_ONLY".to_string(),
        protected_evaluation: false,
        corpus_fingerprint: corpus_fingerprint.clone(),
        benchmark_manifest_fingerprint: benchmark_fingerprint.clone(),
        parent_dev_identity_fingerprint: parent_fingerprint.clone(),
        candidate_universe: all_dev_identities.len(),
        candidate_universe_fingerprint: candidate_fingerprint.clone(),
        query_count: full_query_indices.len(),
        query_cohort_fingerprint: query_fingerprint.clone(),
        candidate_pool: FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
        target_forcing: false,
        dev_partition_used_for_selection: false,
        train_holdout_accessed: false,
        historical_validation_accessed: false,
        historical_test_accessed: false,
    };
    fs::write(
        output_root.join("export_metadata.yaml"),
        serde_yaml::to_string(&metadata)?,
    )?;
    fs::write(
        output_root.join("export_summary.tsv"),
        format!(
            concat!(
                "key\tvalue\n",
                "status\tPASS\n",
                "objective\t{}\n",
                "candidate_universe\t{}\n",
                "candidate_universe_fingerprint\t{}\n",
                "query_count\t{}\n",
                "query_cohort_fingerprint\t{}\n",
                "candidate_pool\t{}\n",
                "target_forcing\tNO\n",
                "train_holdout_accessed\tNO\n",
                "historical_test_accessed\tNO\n"
            ),
            V0761_OBJECTIVE,
            all_dev_identities.len(),
            candidate_fingerprint,
            full_query_indices.len(),
            query_fingerprint,
            FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
        ),
    )?;

    println!("v0761_version\tv0.76.1-actual-dev-production-equivalence-export");
    println!("objective\t{V0761_OBJECTIVE}");
    println!("evaluation_partition\tDEV_ENGINEERING_EQUIVALENCE_ONLY");
    println!("candidate_universe\t{}", all_dev_identities.len());
    println!("candidate_universe_fingerprint\t{candidate_fingerprint}");
    println!("query_count\t{}", full_query_indices.len());
    println!("query_cohort_fingerprint\t{query_fingerprint}");
    println!("candidate_pool\t{FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751}");
    println!("target_forcing\tNO");
    println!("train_holdout_accessed\tNO");
    println!("historical_validation_accessed\tNO");
    println!("historical_test_accessed\tNO");
    println!("v0761_actual_dev_export=PASS");
    println!("v0761_out\t{}", output_root.display());

    Ok(())
}

fn require_eq(label: &str, observed: &str, expected: &str) -> Result<()> {
    if observed != expected {
        bail!("v0.76.1 {label} mismatch: current={observed} expected={expected}");
    }
    Ok(())
}

fn build_alignment_groups(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    max_sequence_len: usize,
) -> Result<Vec<AlignmentGroup>> {
    let mut groups = BTreeMap::<String, AlignmentGroup>::new();
    for entry in benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
    {
        let record = records
            .get(entry.record_index)
            .context("v0.76.1 benchmark record index out of range")?;
        if record.peptidoform.sequence.chars().count() > max_sequence_len {
            continue;
        }
        let Some(charge) = record.context.charge else {
            continue;
        };
        if charge <= 0 || record.context.precursor_mz.is_none() {
            continue;
        }
        if FoundationSpectrum::from_training_record(record).is_none() {
            continue;
        }
        let key = format!("{}|z{charge}", entry.peptidoform);
        let group = groups.entry(key.clone()).or_insert_with(|| AlignmentGroup {
            key,
            peptidoform: entry.peptidoform.clone(),
            sequence: entry.sequence.clone(),
            charge,
            record_indices: Vec::new(),
        });
        if group.peptidoform != entry.peptidoform
            || group.sequence != entry.sequence
            || group.charge != charge
        {
            bail!("v0.76.1 identity grouping collision");
        }
        group.record_indices.push(entry.record_index);
    }
    Ok(groups.into_values().collect())
}

fn select_dev_identities(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    count: usize,
    seed: u64,
) -> Result<Vec<CandidateIdentity>> {
    let mut group_order = (0..groups.len()).collect::<Vec<_>>();
    group_order.sort_by_key(|&index| mix64(seed ^ hash64_str(&groups[index].key)));
    let mut selected = Vec::with_capacity(count);
    for group_index in group_order.into_iter().take(count) {
        selected.push(identity_from_group(
            records,
            &groups[group_index],
            seed.rotate_left(17),
        )?);
    }
    validate_unique_identities(&selected)?;
    Ok(selected)
}

fn build_all_partition_identities(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    record_seed: u64,
) -> Result<Vec<CandidateIdentity>> {
    let mut identities = Vec::with_capacity(groups.len());
    for group in groups {
        identities.push(identity_from_group(records, group, record_seed)?);
    }
    validate_unique_identities(&identities)?;
    Ok(identities)
}

fn identity_from_group(
    records: &[FoundationTrainingRecord],
    group: &AlignmentGroup,
    seed: u64,
) -> Result<CandidateIdentity> {
    let mut record_indices = group.record_indices.clone();
    record_indices.sort_by_key(|&index| mix64(seed ^ index as u64));
    let record_index = *record_indices
        .first()
        .context("v0.76.1 identity group has no records")?;
    let record = &records[record_index];
    let mz = record
        .context
        .precursor_mz
        .context("v0.76.1 identity record lacks precursor m/z")?;
    if group.charge <= 0 {
        bail!("v0.76.1 observed precursor charge must be positive");
    }
    let observed_neutral_mass = foundation_precursor_neutral_mass(f64::from(mz), group.charge)
        .map_err(anyhow::Error::msg)?;
    let candidate_neutral_mass =
        foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
    Ok(CandidateIdentity {
        record_index,
        exact_key: group.key.clone(),
        il_key: format!("{}|z{}", il_label(&group.peptidoform), group.charge),
        charge: group.charge,
        observed_neutral_mass,
        candidate_neutral_mass,
    })
}

fn validate_unique_identities(identities: &[CandidateIdentity]) -> Result<()> {
    let unique = identities
        .iter()
        .map(|item| item.exact_key.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() != identities.len() {
        bail!("v0.76.1 candidate catalog contains duplicate exact identities");
    }
    Ok(())
}

fn full_identity_index(identities: &[CandidateIdentity]) -> Result<BTreeMap<String, usize>> {
    let mut out = BTreeMap::new();
    for (index, identity) in identities.iter().enumerate() {
        if out.insert(identity.exact_key.clone(), index).is_some() {
            bail!("v0.76.1 duplicate exact identity while indexing candidate catalog");
        }
    }
    Ok(out)
}

fn deterministic_eval_queries(
    identities: &[CandidateIdentity],
    count: usize,
    seed: u64,
) -> Vec<usize> {
    let mut indices = (0..identities.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&index| mix64(seed ^ hash64_str(&identities[index].exact_key)));
    indices.truncate(count.min(indices.len()));
    indices
}

fn identity_fingerprint(identities: &[CandidateIdentity]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for item in identities {
        for byte in item.exact_key.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= item.record_index as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn build_search_space(
    records: &[FoundationTrainingRecord],
    groups: &[AlignmentGroup],
    identities: &[CandidateIdentity],
) -> Result<FoundationPracticalIdentifierSearchSpaceV0760> {
    if groups.len() != identities.len() {
        bail!("v0.76.1 group/identity length mismatch");
    }

    let mut entries = Vec::<FoundationPracticalIdentifierSearchSpaceEntryV0760>::new();
    let mut positions = BTreeMap::<String, usize>::new();

    for (group, identity) in groups.iter().zip(identities) {
        if group.key != identity.exact_key || group.charge != identity.charge {
            bail!("v0.76.1 group/identity order drift");
        }
        let record = &records[identity.record_index];
        if record.peptidoform.sequence != group.sequence {
            bail!("v0.76.1 representative sequence drift for {}", group.key);
        }
        let modifications = record
            .peptidoform
            .modifications
            .iter()
            .map(api_modification)
            .collect::<Vec<_>>();

        if let Some(&position) = positions.get(&group.peptidoform) {
            let entry = &mut entries[position];
            if entry.sequence != group.sequence || entry.modifications != modifications {
                bail!(
                    "v0.76.1 peptidoform chemistry drift across charge states for {}",
                    group.peptidoform
                );
            }
            if entry
                .charges
                .last()
                .is_some_and(|&charge| group.charge <= charge)
            {
                bail!(
                    "v0.76.1 charge order is not strictly increasing for {}",
                    group.peptidoform
                );
            }
            entry.charges.push(group.charge);
        } else {
            positions.insert(group.peptidoform.clone(), entries.len());
            entries.push(FoundationPracticalIdentifierSearchSpaceEntryV0760 {
                id: group.peptidoform.clone(),
                sequence: group.sequence.clone(),
                modifications,
                charges: vec![group.charge],
            });
        }
    }

    Ok(FoundationPracticalIdentifierSearchSpaceV0760::new(entries))
}

fn api_modification(
    modification: &FoundationModification,
) -> FoundationPracticalIdentifierModificationV0752 {
    let location = match modification.site {
        FoundationModificationSite::Residue(residue_index) => {
            FoundationPracticalIdentifierModificationSiteV0752::Residue { residue_index }
        }
        FoundationModificationSite::NTerm => {
            FoundationPracticalIdentifierModificationSiteV0752::NTerm
        }
        FoundationModificationSite::CTerm => {
            FoundationPracticalIdentifierModificationSiteV0752::CTerm
        }
    };
    FoundationPracticalIdentifierModificationV0752 {
        location,
        mass_delta: modification.mass_delta,
        unimod_id: modification.unimod_id,
    }
}

fn validate_materialized_catalog(
    materialized: &[redeem_properties::foundation::FoundationPracticalIdentifierCatalogCandidateV0752],
    identities: &[CandidateIdentity],
) -> Result<()> {
    if materialized.len() != identities.len() {
        bail!(
            "v0.76.1 materialized candidate-count drift: current={} expected={}",
            materialized.len(),
            identities.len()
        );
    }
    for (index, (candidate, identity)) in materialized.iter().zip(identities).enumerate() {
        if candidate.key != identity.exact_key || candidate.charge != identity.charge {
            bail!(
                "v0.76.1 materialized candidate order/key drift at index {index}: current={} expected={}",
                candidate.key,
                identity.exact_key
            );
        }
    }
    Ok(())
}

fn build_batch_requests(
    records: &[FoundationTrainingRecord],
    identities: &[CandidateIdentity],
    full_query_indices: &[usize],
) -> Result<FoundationPracticalIdentifierBatchRequestV0760> {
    let mut requests = Vec::with_capacity(full_query_indices.len());
    for (query_slot, &query_index) in full_query_indices.iter().enumerate() {
        let identity = identities
            .get(query_index)
            .context("v0.76.1 query identity index out of range")?;
        let record = &records[identity.record_index];
        let precursor_mz = record
            .context
            .precursor_mz
            .context("v0.76.1 actual DEV query lacks precursor m/z")?;
        let spectrum = FoundationSpectrum::from_training_record(record)
            .context("v0.76.1 actual DEV query lacks observed spectrum")?;
        let peaks = spectrum
            .peaks
            .iter()
            .map(|peak| FoundationPracticalIdentifierPeakV0752 {
                mz: peak.mz,
                intensity: peak.intensity,
            })
            .collect::<Vec<_>>();
        requests.push(FoundationPracticalIdentifierRequestV0752 {
            schema: FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752.to_string(),
            query_id: query_slot.to_string(),
            observed_precursor_mz: f64::from(precursor_mz),
            observed_charge: identity.charge,
            peaks,
            top_k: FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
        });
    }
    Ok(FoundationPracticalIdentifierBatchRequestV0760::new(
        requests,
    ))
}

fn write_candidate_catalog(path: &Path, identities: &[CandidateIdentity]) -> Result<()> {
    let mut text = String::from(
        "candidate_index\trecord_index\texact_key\til_key\tcharge\tobserved_neutral_mass\ttheoretical_neutral_mass\n",
    );
    for (candidate_index, identity) in identities.iter().enumerate() {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\n",
            candidate_index,
            identity.record_index,
            identity.exact_key,
            identity.il_key,
            identity.charge,
            identity.observed_neutral_mass,
            identity.candidate_neutral_mass,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_query_manifest(
    path: &Path,
    identities: &[CandidateIdentity],
    full_query_indices: &[usize],
    selected_query_indices: &[usize],
) -> Result<()> {
    if full_query_indices.len() != selected_query_indices.len() {
        bail!("v0.76.1 query manifest shape mismatch");
    }
    let mut text = String::from(
        "query_slot\tselected_identity_index\tfull_identity_index\texact_key\tcharge\tobserved_neutral_mass\n",
    );
    for (query_slot, (&full_index, &selected_index)) in full_query_indices
        .iter()
        .zip(selected_query_indices)
        .enumerate()
    {
        let identity = &identities[full_index];
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.8}\n",
            query_slot,
            selected_index,
            full_index,
            identity.exact_key,
            identity.charge,
            identity.observed_neutral_mass,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn il_label(label: &str) -> String {
    label
        .chars()
        .map(|residue| if residue == 'I' { 'L' } else { residue })
        .collect()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hash64_str(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn self_test() -> Result<()> {
    let modification =
        FoundationModification::unimod(FoundationModificationSite::Residue(1), 1, 35, 15.994_915);
    let api = api_modification(&modification);
    if api.unimod_id != Some(35) || (api.mass_delta - 15.994_915).abs() > 1.0e-6 {
        bail!("v0.76.1 modification conversion self-test failed");
    }
    if !matches!(
        api.location,
        FoundationPracticalIdentifierModificationSiteV0752::Residue { residue_index: 1 }
    ) {
        bail!("v0.76.1 modification-site conversion self-test failed");
    }

    let search_space = FoundationPracticalIdentifierSearchSpaceV0760::new(vec![
        FoundationPracticalIdentifierSearchSpaceEntryV0760 {
            id: "PEPTIDE".to_string(),
            sequence: "PEPTIDE".to_string(),
            modifications: Vec::new(),
            charges: vec![2, 3],
        },
    ]);
    let catalog = search_space.materialize_catalog()?;
    let keys = catalog
        .candidates
        .iter()
        .map(|candidate| candidate.key.as_str())
        .collect::<Vec<_>>();
    if keys != vec!["PEPTIDE|z2", "PEPTIDE|z3"] {
        bail!("v0.76.1 deterministic materialization self-test failed");
    }
    if FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751 != 256 {
        bail!("v0.76.1 frozen candidate pool drift");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_test_passes() {
        self_test().unwrap();
    }

    #[test]
    fn query_seed_and_frozen_contract_are_stable() {
        assert_eq!(V0761_DEV_QUERIES, 512);
        assert_eq!(V070_DEV_IDENTITIES, 2048);
        assert_eq!(V0761_EXPECTED_CANDIDATE_UNIVERSE, 63_332);
        assert_eq!(
            V0761_EXPECTED_CANDIDATE_FINGERPRINT,
            "fnv1a64:a5699533553116e8"
        );
        assert_eq!(
            V0761_EXPECTED_QUERY_COHORT_FINGERPRINT,
            "fnv1a64:bc6d4b8b3090af2b"
        );
    }
}
