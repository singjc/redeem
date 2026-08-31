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
