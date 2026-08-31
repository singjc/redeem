use redeem_properties::foundation::{
    split_foundation_records, FoundationDatasetLoader, FoundationSplitConfig, FoundationSplitMode,
    FoundationTableLoaderConfig,
};

fn inspection_table() -> &'static [u8] {
    concat!(
        "ModifiedPeptide\tPrecursorCharge\tNormalizedRetentionTime\tRetentionTime\tCCS\tRun\tInstrument\tFragmentType\tFragmentSeriesNumber\tProductCharge\tLibraryIntensity\n",
        "PEPTIDEK\t2\t31.5\t1800\t410\trun-a\ttimsTOF\tb\t2\t1\t50\n",
        "PEPTIDEK\t2\t31.5\t1800\t410\trun-a\ttimsTOF\ty\t3\t1\t100\n",
        "AGHCEWQMK\t3\t47.0\t2500\t455\trun-b\tQE\ty\t4\t2\t80\n",
    )
    .as_bytes()
}

#[test]
fn table_report_exposes_schema_and_label_coverage() {
    let mut loader = FoundationDatasetLoader::new(16);
    let report = loader
        .load_reader_with_report(
            inspection_table(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();

    assert_eq!(report.schema.profile, "generic");
    assert_eq!(report.schema.sequence.header, "ModifiedPeptide");
    assert_eq!(
        report.schema.normalized_rt.as_ref().unwrap().header,
        "NormalizedRetentionTime"
    );
    assert_eq!(
        report.schema.observed_rt.as_ref().unwrap().header,
        "RetentionTime"
    );
    assert!(report.schema.collisions.is_empty());
    assert_eq!(report.stats.input_rows, 3);
    assert_eq!(report.stats.parsed_rows, 3);
    assert_eq!(report.stats.precursor_records, 2);
    assert_eq!(report.stats.unique_sequences, 2);
    assert_eq!(report.stats.normalized_rt_records, 2);
    assert_eq!(report.stats.observed_rt_records, 2);
    assert_eq!(report.stats.ccs_records, 2);
    assert_eq!(report.stats.ms2_records, 2);
    assert_eq!(report.stats.unique_runs, 2);
    assert_eq!(report.stats.unique_instruments, 2);
    assert!(report.stats.error_examples.is_empty());
}

#[test]
fn public_run_split_keeps_complete_runs_disjoint() {
    let table = concat!(
        "ModifiedPeptide\tPrecursorCharge\tRun\tInstrument\n",
        "PEPTIDEK\t2\trun-a\tA\n",
        "AGHCEWQMK\t2\trun-a\tA\n",
        "AAAAAAK\t2\trun-b\tB\n",
        "CCCCCCK\t2\trun-b\tB\n",
        "DDDDDDK\t2\trun-c\tC\n",
        "EEEEEEK\t2\trun-d\tD\n",
    );
    let mut loader = FoundationDatasetLoader::new(16);
    let records = loader
        .load_reader(
            table.as_bytes(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
    let split = split_foundation_records(
        &records,
        &FoundationSplitConfig {
            mode: FoundationSplitMode::Run,
            validation_fraction: 0.25,
            test_fraction: 0.25,
            seed: 11,
        },
    )
    .unwrap();

    let partition = |index: usize| {
        if split.train.contains(&index) {
            0
        } else if split.validation.contains(&index) {
            1
        } else {
            2
        }
    };
    assert_eq!(partition(0), partition(1));
    assert_eq!(partition(2), partition(3));
    assert_eq!(split.summary.total_records, records.len());
    assert_eq!(
        split.summary.train_records + split.summary.validation_records + split.summary.test_records,
        records.len()
    );
}

#[test]
fn openswath_profile_avoids_sequence_nce_collision_and_parses_terminal_modification() {
    let table = concat!(
        "sequence\tprecursor_mz\tprecursor_charge\tfragment_type\tfragment_series_number\tproduct_charge\tretention_time\tion_mobility\tintensity\n",
        ".(UniMod:1)AAAAAAGAASGLPGPVAQGLK\t500.2\t2\tb\t3\t1\t1234.5\t1.05\t1000\n",
    );
    let mut loader = FoundationDatasetLoader::new(16);
    let report = loader
        .load_reader_with_report(
            table.as_bytes(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();

    assert_eq!(report.schema.profile, "openswath_finetuning");
    assert_eq!(report.schema.sequence.header, "sequence");
    assert!(report.schema.nce.is_none());
    assert!(report.schema.collisions.is_empty());
    assert_eq!(report.stats.parsed_rows, 1);
    assert_eq!(report.stats.skipped_error_rows, 0);
    assert_eq!(report.stats.modified_records, 1);
    assert_eq!(report.stats.canonical_unimod_records, 1);
    assert_eq!(report.stats.unresolved_modification_records, 0);
    assert_eq!(report.stats.n_terminal_modification_occurrences, 1);
    assert_eq!(
        report.stats.unimod_occurrences.get("UniMod:1 Acetyl"),
        Some(&1)
    );
    assert_eq!(
        report.records[0].retention_time.observed_seconds,
        Some(1234.5)
    );
    assert_eq!(report.records[0].context.nce, None);
}

#[test]
fn ip2_profile_prefers_modified_peptide_sequence() {
    let table = concat!(
        "PrecursorMz\tProductMz\tAnnotation\tProteinId\tGeneName\tPeptideSequence\tModifiedPeptideSequence\tPrecursorCharge\tLibraryIntensity\tNormalizedRetentionTime\tPrecursorIonMobility\tFragmentType\tFragmentCharge\tFragmentSeriesNumber\tFragmentLossType\tDecoyMobility\n",
        "500.2\t300.1\ty3\tP1\tGENE\tACDEFGK\tAC(UniMod:4)DEFGK\t2\t1000\t42.5\t1.02\ty\t1\t3\tnoloss\t0\n",
    );
    let mut loader = FoundationDatasetLoader::new(16);
    let report = loader
        .load_reader_with_report(
            table.as_bytes(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();

    assert_eq!(report.schema.profile, "ip2_bruker_spectral_library");
    assert_eq!(report.schema.sequence.header, "ModifiedPeptideSequence");
    assert!(report.schema.nce.is_none());
    assert!(report.schema.collisions.is_empty());
    assert_eq!(report.stats.modified_records, 1);
    assert_eq!(report.records[0].peptidoform.sequence, "ACDEFGK");
    assert_eq!(report.records[0].peptidoform.modifications.len(), 1);
    assert_eq!(
        report.records[0].peptidoform.modifications[0].unimod_id,
        Some(4)
    );
    assert!((report.records[0].peptidoform.modifications[0].mass_delta - 57.021465).abs() < 1e-5);
    assert_eq!(
        report
            .stats
            .unimod_occurrences
            .get("UniMod:4 Carbamidomethyl"),
        Some(&1)
    );
    assert_eq!(report.records[0].retention_time.normalized, Some(42.5));
}

#[test]
fn modification_family_split_groups_same_unimod_across_sequences() {
    let table = concat!(
        "ModifiedPeptide\tPrecursorCharge\n",
        "PEPM(UniMod:35)IDEK\t2\n",
        "AAAAAM(UniMod:35)K\t2\n",
        "AC(UniMod:4)DEFGK\t2\n",
        "CC(UniMod:4)AAAAK\t2\n",
        "PEPTIDEK\t2\n",
    );
    let mut loader = FoundationDatasetLoader::new(16);
    let records = loader
        .load_reader(
            table.as_bytes(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
    let split = split_foundation_records(
        &records,
        &FoundationSplitConfig {
            mode: FoundationSplitMode::ModificationFamily,
            validation_fraction: 0.2,
            test_fraction: 0.2,
            seed: 19,
        },
    )
    .unwrap();

    let partition = |index: usize| {
        if split.train.contains(&index) {
            0
        } else if split.validation.contains(&index) {
            1
        } else {
            2
        }
    };
    assert_eq!(partition(0), partition(1));
    assert_eq!(partition(2), partition(3));
}

#[test]
fn modification_family_split_rejects_numeric_only_open_modifications() {
    let table = concat!(
        "ModifiedPeptide\tPrecursorCharge\n",
        "PEPM[+15.9949]IDEK\t2\n",
        "AC(UniMod:4)DEFGK\t2\n",
    );
    let mut loader = FoundationDatasetLoader::new(16);
    let records = loader
        .load_reader(
            table.as_bytes(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
    let error = split_foundation_records(
        &records,
        &FoundationSplitConfig {
            mode: FoundationSplitMode::ModificationFamily,
            validation_fraction: 0.2,
            test_fraction: 0.2,
            seed: 19,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("canonical UniMod identity"));
}
