use candle_core::Device;
use redeem_properties::foundation::{
    FoundationCollator, FoundationCollatorConfig, FoundationConfig, FoundationCorruptionConfig,
    FoundationDatasetLoader, FoundationTableLoaderConfig, FoundationTrainer,
    FoundationTrainerConfig,
};

fn tiny_config() -> FoundationConfig {
    FoundationConfig {
        max_sequence_len: 16,
        graph_hidden_dim: 16,
        graph_layers: 1,
        model_dim: 32,
        num_attention_heads: 4,
        transformer_ff_dim: 64,
        transformer_layers: 1,
        contrastive_dim: 16,
        ..FoundationConfig::default()
    }
}

fn example_transition_table() -> &'static [u8] {
    concat!(
        "ModifiedPeptide\tPrecursorCharge\tFragmentType\tFragmentSeriesNumber\tProductCharge\tLibraryIntensity\tNormalizedRetentionTime\tCCS\tCollisionEnergy\tInstrument\n",
        "PEPTIDEK\t2\tb\t2\t1\t50\t31.5\t410\t27\ttimsTOF\n",
        "PEPTIDEK\t2\ty\t3\t1\t100\t31.5\t410\t27\ttimsTOF\n",
        "AGHCEWQMK\t3\tb\t3\t1\t80\t47.0\t455\t30\tQE\n",
        "AGHCEWQMK\t3\ty\t4\t2\t40\t47.0\t455\t30\tQE\n",
    )
    .as_bytes()
}

#[test]
fn foundation_transition_loader_groups_rows_and_keeps_portable_rt() {
    let config = tiny_config();
    let mut loader = FoundationDatasetLoader::new(config.instrument_vocab_size);
    let records = loader
        .load_reader(
            example_transition_table(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].fragments.len(), 2);
    assert!(records
        .iter()
        .all(|record| record.retention_time.normalized.is_some()));
    assert!(records
        .iter()
        .all(|record| record.retention_time.observed_seconds.is_none()));
    assert_eq!(loader.instruments().names().len(), 3);
}

#[test]
fn foundation_collator_emits_two_masked_views_and_sparse_label_masks() {
    let config = tiny_config();
    let mut loader = FoundationDatasetLoader::new(config.instrument_vocab_size);
    let records = loader
        .load_reader(
            example_transition_table(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
    let collator = FoundationCollator::new(
        config,
        FoundationCollatorConfig {
            corruption: FoundationCorruptionConfig {
                residue_mask_probability: 0.25,
                chemistry_mask_probability: 0.25,
            },
            ..FoundationCollatorConfig::default()
        },
    )
    .unwrap();
    let views = collator.collate_views(&records, &Device::Cpu, 7).unwrap();
    assert!(views.first.targets.rt_mask.is_some());
    assert!(views.first.targets.ccs_mask.is_some());
    assert!(views.first.targets.ms2_mask.is_some());
    assert!(views.first.targets.masked_residue_indices.is_some());
    assert!(views.first.targets.chemistry_mask.is_some());
}

#[test]
fn foundation_trainer_performs_a_real_adamw_step() {
    let config = tiny_config();
    let mut loader = FoundationDatasetLoader::new(config.instrument_vocab_size);
    let records = loader
        .load_reader(
            example_transition_table(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap();
    let mut trainer = FoundationTrainer::new(
        config,
        FoundationTrainerConfig {
            batch_size: 2,
            collator: FoundationCollatorConfig {
                corruption: FoundationCorruptionConfig {
                    residue_mask_probability: 0.25,
                    chemistry_mask_probability: 0.25,
                },
                ..FoundationCollatorConfig::default()
            },
            ..FoundationTrainerConfig::default()
        },
        Device::Cpu,
    )
    .unwrap();
    let metrics = trainer.train_step(&records).unwrap();
    assert_eq!(trainer.global_step(), 1);
    assert!(metrics.total_loss.is_finite());
    assert!(metrics.contrastive_loss.unwrap().is_finite());
}
