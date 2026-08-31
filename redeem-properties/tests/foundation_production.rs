use candle_core::Device;
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationCheckpointProvenance, FoundationCollatorConfig,
    FoundationConfig, FoundationCorpusConfig, FoundationCorpusDelimiter, FoundationCorpusSourceSpec,
    FoundationCorruptionConfig, FoundationDatasetLoader, FoundationGradientDiagnosticsConfig,
    FoundationLearningRateSchedule, FoundationPartition, FoundationRecordProvenance, FoundationSamplingConfig,
    FoundationSamplingStrategy, FoundationSplitConfig, FoundationTableLoaderConfig,
    FoundationTargetNormalizationConfig, FoundationRegressionNormalization,
    FoundationRegressionNormalizationStrategy, RetentionTimeObjective,
    FoundationTrainer, FoundationTrainerConfig, FoundationTrainingProgress,
    sample_foundation_training_indices, sample_foundation_validation_indices,
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
    let mut original = FoundationTrainer::new(tiny_config(), trainer_config(), Device::Cpu).unwrap();
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
    let weights = BTreeMap::from([
        ("openswath".to_string(), 0.25),
        ("ip2".to_string(), 0.75),
    ]);
    let config = FoundationSamplingConfig {
        strategy: FoundationSamplingStrategy::SourceWeighted,
        train_steps_per_epoch: Some(4),
        validation_steps: Some(2),
        source_weights: weights,
        ..FoundationSamplingConfig::default()
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


#[test]
fn source_weighted_validation_is_fixed_stratified_and_without_replacement() {
    let base = records();
    let mut expanded = Vec::new();
    let mut provenance = Vec::new();
    for source_index in 0..2usize {
        for record_index in 0..4usize {
            expanded.push(base[record_index % base.len()].clone());
            provenance.push(FoundationRecordProvenance {
                source_index,
                source_id: if source_index == 0 {
                    "openswath".to_string()
                } else {
                    "ip2".to_string()
                },
                source_record_index: record_index,
            });
        }
    }
    let config = FoundationSamplingConfig {
        validation_steps: Some(2),
        validation_source_weights: BTreeMap::from([
            ("openswath".to_string(), 0.25),
            ("ip2".to_string(), 0.75),
        ]),
        ..FoundationSamplingConfig::default()
    };
    let indices = (0..expanded.len()).collect::<Vec<_>>();
    let first = sample_foundation_validation_indices(
        &expanded,
        &provenance,
        &indices,
        2,
        20260831,
        &config,
    )
    .unwrap();
    let second = sample_foundation_validation_indices(
        &expanded,
        &provenance,
        &indices,
        2,
        20260831,
        &config,
    )
    .unwrap();
    assert_eq!(first.indices, second.indices);
    assert_eq!(first.indices.len(), 4);
    assert_eq!(first.unique_records, 4);
    assert_eq!(first.source_records.get("openswath"), Some(&1));
    assert_eq!(first.source_records.get("ip2"), Some(&3));
}

#[test]
fn epoch_metrics_report_gradient_clipping_frequency_and_scale() {
    let records = records();
    let mut trainer = FoundationTrainer::new(
        tiny_config(),
        FoundationTrainerConfig {
            max_gradient_norm: Some(1e-8),
            ..trainer_config()
        },
        Device::Cpu,
    )
    .unwrap();
    let metrics = trainer.train_epoch(&records).unwrap();
    assert_eq!(metrics.steps, 1);
    assert_eq!(metrics.clipped_steps, 1);
    assert_eq!(metrics.clipped_fraction, Some(1.0));
    assert!(metrics
        .mean_gradient_scale
        .is_some_and(|value| value > 0.0 && value < 1.0));
}


#[test]
fn regression_normalization_uses_train_partition_only() {
    let mut records = records();
    let mut validation_only = records[0].clone();
    validation_only.retention_time.normalized = Some(10_000.0);
    validation_only.ccs = Some(9_999.0);
    records.push(validation_only);

    let mut normalization = FoundationTargetNormalizationConfig {
        rt: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
        ccs: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
    };
    normalization
        .resolve_from_training_partition(
            &records,
            &[0, 1],
            RetentionTimeObjective::Normalized,
        )
        .unwrap();

    assert_eq!(normalization.rt.label_count, 2);
    assert_eq!(normalization.ccs.label_count, 2);
    assert!((normalization.rt.mean.unwrap() - 39.25).abs() < 1e-6);
    assert!((normalization.ccs.mean.unwrap() - 432.5).abs() < 1e-6);
    assert!(normalization.rt.standard_deviation.unwrap() > 0.0);
    assert!(normalization.ccs.standard_deviation.unwrap() > 0.0);
}

#[test]
fn normalized_regression_reports_native_unit_errors() {
    let records = records();
    let mut normalization = FoundationTargetNormalizationConfig {
        rt: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
        ccs: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
    };
    normalization
        .resolve_from_training_partition(
            &records,
            &[0, 1],
            RetentionTimeObjective::Normalized,
        )
        .unwrap();
    let mut config = trainer_config();
    config.target_normalization = normalization;
    let mut trainer = FoundationTrainer::new(tiny_config(), config, Device::Cpu).unwrap();
    let metrics = trainer.train_step(&records).unwrap();
    assert!(metrics.rt_loss.is_some_and(f32::is_finite));
    assert!(metrics.rt_mae_native.is_some_and(f32::is_finite));
    assert!(metrics.rt_rmse_native.is_some_and(f32::is_finite));
    assert!(metrics.ccs_mae_native.is_some_and(f32::is_finite));
    assert!(metrics.ccs_rmse_native.is_some_and(f32::is_finite));
}

#[test]
fn task_gradient_diagnostics_report_weighted_objective_norms() {
    let records = records();
    let mut normalization = FoundationTargetNormalizationConfig {
        rt: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
        ccs: FoundationRegressionNormalization {
            strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
            ..FoundationRegressionNormalization::default()
        },
    };
    normalization
        .resolve_from_training_partition(
            &records,
            &[0, 1],
            RetentionTimeObjective::Normalized,
        )
        .unwrap();
    let mut config = trainer_config();
    config.target_normalization = normalization;
    config.gradient_diagnostics = FoundationGradientDiagnosticsConfig {
        enabled: true,
        every_n_steps: 1,
    };
    let mut trainer = FoundationTrainer::new(tiny_config(), config, Device::Cpu).unwrap();
    let metrics = trainer.train_step(&records).unwrap();
    let gradients = metrics
        .task_gradient_norms
        .expect("gradient diagnostics should run on step zero");
    assert!(gradients.rt.is_some_and(|value| value.is_finite() && value >= 0.0));
    assert!(gradients.ccs.is_some_and(|value| value.is_finite() && value >= 0.0));
    assert!(gradients.ms2.is_some_and(|value| value.is_finite() && value >= 0.0));
    assert!(gradients
        .contrastive
        .is_some_and(|value| value.is_finite() && value >= 0.0));
    for cosine in [
        gradients.rt_cosine_to_total,
        gradients.ccs_cosine_to_total,
        gradients.ms2_cosine_to_total,
        gradients.masked_residue_cosine_to_total,
        gradients.chemistry_cosine_to_total,
        gradients.contrastive_cosine_to_total,
    ] {
        assert!(cosine.is_some_and(|value| value.is_finite() && (-1.0..=1.0).contains(&value)));
    }
}

#[test]
fn task_gradient_diagnostics_follow_global_step_interval() {
    let records = records();
    let mut config = trainer_config();
    config.gradient_diagnostics = FoundationGradientDiagnosticsConfig {
        enabled: true,
        every_n_steps: 2,
    };
    let mut trainer = FoundationTrainer::new(tiny_config(), config, Device::Cpu).unwrap();
    assert!(trainer.train_step(&records).unwrap().task_gradient_norms.is_some());
    assert!(trainer.train_step(&records).unwrap().task_gradient_norms.is_none());
    assert!(trainer.train_step(&records).unwrap().task_gradient_norms.is_some());
}

