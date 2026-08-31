use candle_core::Device;
use redeem_properties::foundation::{
    load_foundation_corpus, sample_foundation_training_indices, FoundationCheckpointProvenance,
    FoundationCollatorConfig, FoundationConfig, FoundationCorpusConfig, FoundationCorpusDelimiter,
    FoundationCorpusSourceSpec, FoundationCorruptionConfig, FoundationDatasetLoader,
    FoundationLearningRateSchedule, FoundationPartition, FoundationRecordProvenance,
    FoundationSamplingConfig, FoundationSamplingStrategy, FoundationSplitConfig,
    FoundationTableLoaderConfig, FoundationTrainer, FoundationTrainerConfig,
    FoundationTrainingProgress,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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
        dropout: 0.0,
        ..FoundationConfig::default()
    }
}

fn transition_table() -> &'static [u8] {
    concat!(
        "ModifiedPeptide\tPrecursorCharge\tFragmentType\tFragmentSeriesNumber\tProductCharge\tLibraryIntensity\tNormalizedRetentionTime\tCCS\n",
        "PEPTIDEK\t2\tb\t2\t1\t50\t31.5\t410\n",
        "PEPTIDEK\t2\ty\t3\t1\t100\t31.5\t410\n",
        "AGHCEWQMK\t3\tb\t3\t1\t80\t47.0\t455\n",
        "AGHCEWQMK\t3\ty\t4\t2\t40\t47.0\t455\n",
    )
    .as_bytes()
}

fn records() -> Vec<redeem_properties::foundation::FoundationTrainingRecord> {
    let config = tiny_config();
    let mut loader = FoundationDatasetLoader::new(config.instrument_vocab_size);
    loader
        .load_reader(
            transition_table(),
            b'\t',
            &FoundationTableLoaderConfig::default(),
        )
        .unwrap()
}

fn trainer_config() -> FoundationTrainerConfig {
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
    }
}

fn temp_dir(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "redeem_foundation_{label}_{}_{}",
        std::process::id(),
        unique
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn optimizer_checkpoint_restores_moments_and_step_for_same_continuation() {
    let records = records();
    let mut original =
        FoundationTrainer::new(tiny_config(), trainer_config(), Device::Cpu).unwrap();
    original.train_step(&records).unwrap();

    let root = temp_dir("resume");
    original
        .save_checkpoint(
            &root,
            FoundationTrainingProgress::default(),
            FoundationCheckpointProvenance {
                corpus_fingerprint: Some(123),
                experiment_id: Some("resume-test".to_string()),
                ..FoundationCheckpointProvenance::default()
            },
        )
        .unwrap();
    let (mut resumed, metadata) = FoundationTrainer::from_checkpoint(&root, Device::Cpu).unwrap();
    assert_eq!(metadata.global_step, 1);
    assert_eq!(resumed.global_step(), 1);
    assert_eq!(resumed.optimizer_step(), 1);

    let original_metrics = original.train_step(&records).unwrap();
    let resumed_metrics = resumed.train_step(&records).unwrap();
    assert!((original_metrics.total_loss - resumed_metrics.total_loss).abs() < 1e-5);
    assert_eq!(original.global_step(), 2);
    assert_eq!(resumed.global_step(), 2);

    let peptide = vec![records[0].peptidoform.clone()];
    let original_embedding = original
        .model()
        .embed(&peptide)
        .unwrap()
        .peptide_embedding
        .to_vec2::<f32>()
        .unwrap();
    let resumed_embedding = resumed
        .model()
        .embed(&peptide)
        .unwrap()
        .peptide_embedding
        .to_vec2::<f32>()
        .unwrap();
    for (left, right) in original_embedding[0]
        .iter()
        .zip(resumed_embedding[0].iter())
    {
        assert!((left - right).abs() < 1e-5);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn warmup_cosine_and_gradient_clipping_are_applied_per_step() {
    let records = records();
    let mut trainer = FoundationTrainer::new(
        tiny_config(),
        FoundationTrainerConfig {
            learning_rate: 1e-3,
            max_gradient_norm: Some(1e-8),
            learning_rate_schedule: FoundationLearningRateSchedule::WarmupCosine {
                warmup_steps: 2,
                total_steps: 4,
                min_lr_ratio: 0.1,
            },
            ..trainer_config()
        },
        Device::Cpu,
    )
    .unwrap();

    let first = trainer.train_step(&records).unwrap();
    assert!((first.learning_rate - 5e-4).abs() < 1e-12);
    assert!(first.gradient_norm.is_finite());
    assert!(first.gradient_scale > 0.0 && first.gradient_scale <= 1.0);

    let second = trainer.train_step(&records).unwrap();
    assert!((second.learning_rate - 1e-3).abs() < 1e-12);
}

#[test]
fn corpus_wide_sequence_split_prevents_cross_source_leakage() {
    let root = temp_dir("corpus");
    let source_a = root.join("a.tsv");
    let source_b = root.join("b.tsv");
    fs::write(
        &source_a,
        concat!(
            "ModifiedPeptide\tPrecursorCharge\tNormalizedRetentionTime\n",
            "PEPTIDEK\t2\t31.0\n",
            "AAAAAAK\t2\t20.0\n",
            "CCCCCCK\t2\t25.0\n",
        ),
    )
    .unwrap();
    fs::write(
        &source_b,
        concat!(
            "ModifiedPeptide\tPrecursorCharge\tNormalizedRetentionTime\n",
            "PEPTIDEK\t3\t32.0\n",
            "DDDDDDK\t2\t35.0\n",
            "EEEEEEK\t2\t40.0\n",
        ),
    )
    .unwrap();

    let corpus = load_foundation_corpus(&FoundationCorpusConfig {
        instrument_vocab_size: 16,
        loader: FoundationTableLoaderConfig::default(),
        sources: vec![
            FoundationCorpusSourceSpec {
                id: "source-a".to_string(),
                path: source_a,
                delimiter: FoundationCorpusDelimiter::Tab,
                ..FoundationCorpusSourceSpec::default()
            },
            FoundationCorpusSourceSpec {
                id: "source-b".to_string(),
                path: source_b,
                delimiter: FoundationCorpusDelimiter::Tab,
                ..FoundationCorpusSourceSpec::default()
            },
        ],
    })
    .unwrap();

    assert_eq!(corpus.records.len(), 6);
    assert_eq!(corpus.sources.len(), 2);
    assert_eq!(corpus.provenance.len(), 6);
    let manifest = corpus
        .build_benchmark_manifest(
            FoundationSplitConfig {
                validation_fraction: 0.2,
                test_fraction: 0.2,
                seed: 20260831,
                ..FoundationSplitConfig::default()
            },
            false,
        )
        .unwrap();

    let peptide_partitions: BTreeSet<FoundationPartition> = manifest
        .entries
        .iter()
        .filter(|entry| entry.sequence == "PEPTIDEK")
        .map(|entry| entry.partition)
        .collect();
    assert_eq!(peptide_partitions.len(), 1);
    assert_eq!(
        manifest
            .entries
            .iter()
            .filter(|entry| entry.sequence == "PEPTIDEK")
            .count(),
        2
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn corpus_yaml_accepts_name_as_source_id_alias() {
    let yaml = r#"
instrument_vocab_size: 16
sources:
  - name: openswath_finetuning
    path: /tmp/openswath.tsv
    delimiter: tab
    metadata: {}
  - id: ip2_bruker_human
    path: /tmp/ip2.tsv.zst
    delimiter: tab
    metadata: {}
"#;
    let config: FoundationCorpusConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.sources.len(), 2);
    assert_eq!(config.sources[0].id, "openswath_finetuning");
    assert_eq!(config.sources[1].id, "ip2_bruker_human");
}

#[test]
fn source_weighted_sampling_is_bounded_deterministic_and_auditable() {
    let records = records();
    let provenance = vec![
        FoundationRecordProvenance {
            source_index: 0,
            source_id: "openswath".to_string(),
            source_record_index: 0,
        },
        FoundationRecordProvenance {
            source_index: 1,
            source_id: "ip2".to_string(),
            source_record_index: 0,
        },
    ];
    let weights = BTreeMap::from([("openswath".to_string(), 0.25), ("ip2".to_string(), 0.75)]);
    let config = FoundationSamplingConfig {
        strategy: FoundationSamplingStrategy::SourceWeighted,
        train_steps_per_epoch: Some(4),
        validation_steps: Some(2),
        source_weights: weights,
    };
    let first = sample_foundation_training_indices(
        &records,
        &provenance,
        &[0, 1],
        2,
        0,
        20260831,
        true,
        &config,
    )
    .unwrap();
    let second = sample_foundation_training_indices(
        &records,
        &provenance,
        &[0, 1],
        2,
        0,
        20260831,
        true,
        &config,
    )
    .unwrap();

    assert_eq!(first.indices, second.indices);
    assert_eq!(first.indices.len(), 8);
    assert_eq!(first.unique_records, 2);
    assert_eq!(first.source_records.get("openswath"), Some(&2));
    assert_eq!(first.source_records.get("ip2"), Some(&6));
    assert_eq!(first.coverage.normalized_rt_records, 8);
    assert_eq!(first.coverage.ms2_records, 8);
}
