use redeem_properties::foundation::{
    build_foundation_benchmark_manifest, split_foundation_records, FoundationBenchmarkManifest,
    FoundationModification, FoundationModificationSite, FoundationSplitConfig, FoundationSplitMode,
    FoundationTrainingRecord, PeptidoformInput, RetentionTimeLabels, TrainingContext,
};

fn record(sequence: &str) -> FoundationTrainingRecord {
    FoundationTrainingRecord {
        peptidoform: PeptidoformInput::unmodified(sequence),
        retention_time: RetentionTimeLabels {
            normalized: Some(sequence.len() as f32),
            observed_seconds: None,
        },
        ccs: None,
        fragments: Vec::new(),
        context: TrainingContext {
            charge: Some(2),
            ..TrainingContext::default()
        },
        run_id: None,
    }
}

fn with_unimod(
    mut record: FoundationTrainingRecord,
    residue: usize,
    id: u32,
    mass: f32,
) -> FoundationTrainingRecord {
    record
        .peptidoform
        .modifications
        .push(FoundationModification::unimod(
            FoundationModificationSite::Residue(residue),
            residue,
            id,
            mass,
        ));
    record
}

fn partition_for(manifest: &FoundationBenchmarkManifest, index: usize) -> &'static str {
    match manifest
        .entries
        .iter()
        .find(|entry| entry.record_index == index)
        .unwrap()
        .partition
    {
        redeem_properties::foundation::FoundationPartition::Train => "train",
        redeem_properties::foundation::FoundationPartition::Validation => "validation",
        redeem_properties::foundation::FoundationPartition::Test => "test",
    }
}

#[test]
fn sequence_manifest_is_materialized_and_round_trips() {
    let records = vec![
        record("PEPTIDEK"),
        record("PEPTIDEK"),
        record("AAAAAAK"),
        record("CCCCCCK"),
        record("DDDDDDK"),
        record("EEEEEEK"),
    ];
    let manifest = build_foundation_benchmark_manifest(
        &records,
        FoundationSplitConfig {
            validation_fraction: 0.2,
            test_fraction: 0.2,
            ..FoundationSplitConfig::default()
        },
        false,
    )
    .unwrap();
    assert_eq!(partition_for(&manifest, 0), partition_for(&manifest, 1));
    assert_eq!(manifest.selected_records, records.len());
    manifest.validate_against_records(&records).unwrap();

    let path = std::env::temp_dir().join(format!(
        "redeem-foundation-benchmark-{}-{}.tsv",
        std::process::id(),
        manifest.dataset_fingerprint
    ));
    manifest.write_tsv(&path).unwrap();
    let loaded = FoundationBenchmarkManifest::read_tsv(&path).unwrap();
    std::fs::remove_file(path).ok();
    assert_eq!(loaded, manifest);
    loaded.validate_against_records(&records).unwrap();
}

#[test]
fn manifest_detects_source_record_changes() {
    let records = vec![record("PEPTIDEK"), record("AAAAAAK"), record("CCCCCCK")];
    let manifest = build_foundation_benchmark_manifest(
        &records,
        FoundationSplitConfig {
            validation_fraction: 0.0,
            test_fraction: 0.34,
            ..FoundationSplitConfig::default()
        },
        false,
    )
    .unwrap();
    let mut changed = records.clone();
    changed[0].retention_time.normalized = Some(999.0);
    let error = manifest.validate_against_records(&changed).unwrap_err();
    assert!(error.to_string().contains("fingerprint changed"));
}

#[test]
fn modification_family_split_uses_connected_components() {
    let mut both = with_unimod(record("CMPEPTIDE"), 0, 4, 57.021_465);
    both = with_unimod(both, 1, 35, 15.994_915);
    let records = vec![
        with_unimod(record("CPEPTIDEK"), 0, 4, 57.021_465),
        with_unimod(record("MPEPTIDEK"), 0, 35, 15.994_915),
        both,
        with_unimod(record("NPEPTIDEK"), 0, 7, 0.984_016),
        with_unimod(record("KPEPTIDEK"), 0, 1, 42.010_565),
    ];
    let split_config = FoundationSplitConfig {
        mode: FoundationSplitMode::ModificationFamily,
        validation_fraction: 0.2,
        test_fraction: 0.2,
        seed: 123,
    };

    // The general splitter uses connected components, so UniMod:4 and
    // UniMod:35 are inseparable when mixed-family records are retained.
    let connected = split_foundation_records(&records, &split_config).unwrap();
    assert_eq!(connected.summary.total_groups, 3);

    // The strict benchmark materializer excludes the mixed-family bridge so
    // individual PTM families can be assigned as genuinely unseen families.
    let manifest = build_foundation_benchmark_manifest(&records, split_config, true).unwrap();
    assert_eq!(manifest.excluded_mixed_family_records, 1);
    assert_eq!(manifest.selected_records, 4);
    assert!(!manifest.entries.iter().any(|entry| entry.record_index == 2));
    assert_eq!(manifest.summary.total_groups, 4);
    manifest.validate_against_records(&records).unwrap();
}

#[test]
fn modification_family_manifest_requires_modified_only_selection() {
    let records = vec![
        record("PEPTIDEK"),
        with_unimod(record("MPEPTIDEK"), 0, 35, 15.994_915),
    ];
    let error = build_foundation_benchmark_manifest(
        &records,
        FoundationSplitConfig {
            mode: FoundationSplitMode::ModificationFamily,
            ..FoundationSplitConfig::default()
        },
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("modified_only=true"));
}
