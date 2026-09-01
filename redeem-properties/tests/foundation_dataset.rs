use redeem_properties::foundation::{
    apply_source_metadata, split_foundation_records, FoundationCcsDerivationMode,
    FoundationDatasetLoader, FoundationMetadataMergePolicy, FoundationSourceMetadata,
    FoundationSplitConfig, FoundationSplitMode, FoundationTableLoaderConfig,
};
use redeem_properties::utils::peptdeep_utils::ion_mobility_to_ccs_bruker;

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
    assert_eq!(
        report
            .schema
            .fragment_mz
            .as_ref()
            .map(|field| field.header.as_str()),
        Some("ProductMz")
    );
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
    assert_eq!(report.records[0].fragments[0].product_mz, Some(300.1));
    assert_eq!(report.stats.observed_fragment_mz_rows, 1);
    assert_eq!(report.stats.observed_spectrum_records, 1);
}

#[test]
fn auto_bruker_derives_ccs_from_ion_mobility() {
    let table = concat!(
        "sequence\tprecursor_mz\tprecursor_charge\tfragment_type\tfragment_series_number\tproduct_charge\tretention_time\tion_mobility\tintensity\n",
        "PEPTIDEK\t500.2\t2\tb\t3\t1\t1234.5\t1.05\t1000\n",
    );
    let config = FoundationTableLoaderConfig {
        ccs_derivation: FoundationCcsDerivationMode::AutoBruker,
        ..FoundationTableLoaderConfig::default()
    };
    let mut loader = FoundationDatasetLoader::new(16);
    let report = loader
        .load_reader_with_report(table.as_bytes(), b'\t', &config)
        .unwrap();

    let expected = ion_mobility_to_ccs_bruker(1.05, 2, 500.2);
    assert_eq!(report.records.len(), 1);
    assert!((report.records[0].ccs.unwrap() - expected).abs() < 1e-5);
    assert_eq!(report.stats.ccs_records, 1);
    assert_eq!(report.stats.explicit_ccs_records, 0);
    assert_eq!(report.stats.derived_ccs_records, 1);
    assert!((report.stats.min_ccs.unwrap() - f64::from(expected)).abs() < 1e-5);
    assert!((report.stats.mean_ccs.unwrap() - f64::from(expected)).abs() < 1e-5);
    assert!((report.stats.max_ccs.unwrap() - f64::from(expected)).abs() < 1e-5);
}

#[test]
fn explicit_ccs_wins_over_ion_mobility_derivation() {
    let table = concat!(
        "ModifiedPeptide\tPrecursorCharge\tPrecursorMz\tIonMobility\tCCS\n",
        "PEPTIDEK\t2\t500.2\t1.05\t499.5\n",
    );
    let config = FoundationTableLoaderConfig {
        ccs_derivation: FoundationCcsDerivationMode::AutoBruker,
        ..FoundationTableLoaderConfig::default()
    };
    let mut loader = FoundationDatasetLoader::new(16);
    let report = loader
        .load_reader_with_report(table.as_bytes(), b'\t', &config)
        .unwrap();

    assert_eq!(report.records[0].ccs, Some(499.5));
    assert_eq!(report.stats.explicit_ccs_records, 1);
    assert_eq!(report.stats.derived_ccs_records, 0);
}

#[test]
fn ccs_derivation_can_be_disabled() {
    let table = concat!(
        "sequence\tprecursor_mz\tprecursor_charge\tfragment_type\tfragment_series_number\tproduct_charge\tretention_time\tion_mobility\tintensity\n",
        "PEPTIDEK\t500.2\t2\tb\t3\t1\t1234.5\t1.05\t1000\n",
    );
    let mut loader = FoundationDatasetLoader::new(16);
    let report = loader
        .load_reader_with_report(
            table.as_bytes(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();

    assert_eq!(report.records[0].ccs, None);
    assert_eq!(report.stats.ccs_records, 0);
    assert_eq!(report.stats.explicit_ccs_records, 0);
    assert_eq!(report.stats.derived_ccs_records, 0);
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

#[test]
fn exact_graph_templates_cover_the_real_high_frequency_ptm_sites() {
    use redeem_properties::foundation::chemistry::{residue_graph, Element};
    use redeem_properties::foundation::{
        exact_graph_modification_for, parse_modified_peptide, FoundationModification,
        FoundationModificationSite,
    };

    let cases = [
        ("AC(UniMod:4)DEFGK", 'C'),
        ("PEPM(UniMod:35)IDEK", 'M'),
        ("PEPN(UniMod:7)IDEK", 'N'),
        ("PEPQ(UniMod:7)IDEK", 'Q'),
    ];
    for (modified, residue) in cases {
        let peptide = parse_modified_peptide(modified).unwrap();
        let modification = &peptide.modifications[0];
        let kind = exact_graph_modification_for(residue, modification).unwrap();
        let mut graph = residue_graph(residue).unwrap();
        assert!(graph.apply_exact_modification(kind));
        assert!(graph
            .atoms
            .iter()
            .all(|atom| atom.element != Element::Pseudo));
    }

    let n_term = parse_modified_peptide(".(UniMod:1)PEPTIDEK").unwrap();
    assert!(exact_graph_modification_for('P', &n_term.modifications[0]).is_some());

    let lysine_acetyl =
        FoundationModification::unimod(FoundationModificationSite::Residue(0), 0, 1, 42.010565);
    assert!(exact_graph_modification_for('K', &lysine_acetyl).is_some());

    let unsupported_site =
        FoundationModification::unimod(FoundationModificationSite::Residue(0), 0, 35, 15.994915);
    assert!(exact_graph_modification_for('W', &unsupported_site).is_none());
}

#[test]
fn table_report_quantifies_exact_graph_and_fallback_ptms() {
    let table = concat!(
        "ModifiedPeptide\tPrecursorCharge\n",
        "AC(UniMod:4)DEFGK\t2\n",
        "PEPM(UniMod:35)IDEK\t2\n",
        "PEPN(UniMod:7)IDEK\t2\n",
        ".(UniMod:1)PEPTIDEK\t2\n",
        "PEPS(UniMod:21)IDEK\t2\n",
    );
    let mut loader = FoundationDatasetLoader::new(16);
    let report = loader
        .load_reader_with_report(
            table.as_bytes(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();

    assert_eq!(report.stats.modification_occurrences, 5);
    assert_eq!(report.stats.exact_graph_modification_occurrences, 4);
    assert_eq!(
        report.stats.pseudo_graph_fallback_modification_occurrences,
        1
    );
    assert_eq!(
        report
            .stats
            .exact_graph_unimod_occurrences
            .get("UniMod:4 Carbamidomethyl"),
        Some(&1)
    );
    assert_eq!(
        report
            .stats
            .exact_graph_unimod_occurrences
            .get("UniMod:35 Oxidation"),
        Some(&1)
    );
    assert_eq!(
        report
            .stats
            .exact_graph_unimod_occurrences
            .get("UniMod:7 Deamidated"),
        Some(&1)
    );
    assert_eq!(
        report
            .stats
            .exact_graph_unimod_occurrences
            .get("UniMod:1 Acetyl"),
        Some(&1)
    );
}

#[test]
fn source_metadata_is_optional_and_fill_missing_preserves_row_values() {
    let input = b"sequence\tprecursor_charge\tfragment_type\tfragment_series_number\tproduct_charge\tintensity\nPEPTIDEK\t2\ty\t3\t1\t100\n";
    let config = FoundationTableLoaderConfig {
        strict: true,
        ..FoundationTableLoaderConfig::default()
    };
    let mut loader = FoundationDatasetLoader::new(16);
    let records = loader.load_reader(&input[..], b'\t', &config).unwrap();
    let mut dataset = loader.finish(records);

    assert_eq!(dataset.records[0].context.nce, None);
    assert_eq!(dataset.records[0].context.instrument_id, None);
    assert_eq!(dataset.records[0].run_id, None);

    let empty_stats = apply_source_metadata(
        &mut dataset,
        &FoundationSourceMetadata::default(),
        FoundationMetadataMergePolicy::FillMissing,
    );
    assert_eq!(empty_stats.nce_assignments, 0);
    assert_eq!(dataset.records[0].context.nce, None);

    let stats = apply_source_metadata(
        &mut dataset,
        &FoundationSourceMetadata {
            nce: Some(27.0),
            instrument: Some("Orbitrap Astral".to_string()),
            run_id: None,
            gradient_seconds: None,
        },
        FoundationMetadataMergePolicy::FillMissing,
    );
    assert_eq!(stats.nce_assignments, 1);
    assert_eq!(stats.instrument_assignments, 1);
    assert_eq!(stats.run_id_assignments, 0);
    assert_eq!(dataset.records[0].context.nce, Some(27.0));
    assert_eq!(
        dataset.records[0].context.instrument_name.as_deref(),
        Some("Orbitrap Astral")
    );
    assert!(dataset.records[0].context.instrument_id.is_some());
    assert_eq!(dataset.records[0].run_id, None);

    dataset.records[0].context.nce = Some(30.0);
    apply_source_metadata(
        &mut dataset,
        &FoundationSourceMetadata {
            nce: Some(20.0),
            ..FoundationSourceMetadata::default()
        },
        FoundationMetadataMergePolicy::FillMissing,
    );
    assert_eq!(dataset.records[0].context.nce, Some(30.0));
}
