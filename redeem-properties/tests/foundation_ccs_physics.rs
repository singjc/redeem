use redeem_properties::foundation::{
    evaluate_foundation_ccs_physics_baseline, fit_foundation_ccs_physics_baseline,
    fit_foundation_ccs_physics_baseline_source_weighted, foundation_ccs_physics_features,
    FoundationCcsPhysicsFitConfig, FoundationRecordProvenance, FoundationTrainingRecord,
    PeptidoformInput, RetentionTimeLabels, TrainingContext,
};
use std::collections::BTreeMap;

fn record(
    sequence: &str,
    charge: Option<i32>,
    mz: Option<f32>,
    ccs: f64,
) -> FoundationTrainingRecord {
    FoundationTrainingRecord {
        peptidoform: PeptidoformInput::unmodified(sequence),
        retention_time: RetentionTimeLabels::default(),
        ccs: Some(ccs as f32),
        fragments: Vec::new(),
        context: TrainingContext {
            charge,
            precursor_mz: mz,
            ..TrainingContext::default()
        },
        run_id: None,
    }
}

#[test]
fn ccs_physics_features_match_the_documented_scaling_and_missing_masks() {
    let peptide = PeptidoformInput::unmodified("PEPTIDEK");
    let context = TrainingContext {
        charge: Some(3),
        precursor_mz: Some(500.0),
        ..TrainingContext::default()
    };
    let features = foundation_ccs_physics_features(&peptide, &context);
    assert_eq!(features[0], 1.0);
    assert_eq!(features[1], 0.75);
    assert_eq!(features[2], 9.0 / 16.0);
    assert_eq!(features[3], 0.5);
    assert_eq!(features[4], 0.5);
    assert_eq!(features[5], 8.0 / 30.0);
    assert_eq!(features[6], 1.0);
    assert_eq!(features[7], 1.0);

    let missing = foundation_ccs_physics_features(
        &peptide,
        &TrainingContext {
            charge: None,
            precursor_mz: Some(500.0),
            ..TrainingContext::default()
        },
    );
    assert_eq!(missing[1], 0.0);
    assert_eq!(missing[2], 0.0);
    assert_eq!(missing[4], 0.0);
    assert_eq!(missing[6], 0.0);
    assert_eq!(missing[7], 1.0);
}

#[test]
fn full_train_ccs_physics_fit_recovers_a_synthetic_linear_prior_without_validation_leakage() {
    let coefficients = [103.0, 96.0, 293.0, 112.0, 168.0, 53.0, 7.0, -4.0];
    let sequences = [
        "PEPTIDEK",
        "AGHCEWQMK",
        "MPEPTIDER",
        "AAAAAAAK",
        "VVVVVVVVVK",
        "YGFYTPK",
    ];
    let mut records = Vec::new();
    for index in 0..96usize {
        let charge = if index % 11 == 0 {
            None
        } else {
            Some(2 + (index % 3) as i32)
        };
        let mz = if index % 13 == 0 {
            None
        } else {
            Some(350.0 + (index % 17) as f32 * 31.0)
        };
        let sequence = sequences[index % sequences.len()];
        let probe = record(sequence, charge, mz, 0.0);
        let features = foundation_ccs_physics_features(&probe.peptidoform, &probe.context);
        let target = coefficients
            .iter()
            .zip(features)
            .map(|(coefficient, feature)| *coefficient * feature)
            .sum::<f64>();
        records.push(record(sequence, charge, mz, target));
    }

    // This validation-only outlier must not affect the train fit or train target statistics.
    records.push(record("TESTPEPTIDEK", Some(4), Some(1200.0), 5000.0));
    let train_indices = (0..96usize).collect::<Vec<_>>();
    let validation_indices = vec![96usize];

    let fit = fit_foundation_ccs_physics_baseline(
        &records,
        &train_indices,
        FoundationCcsPhysicsFitConfig { ridge_lambda: 1e-8 },
    )
    .unwrap();
    assert_eq!(fit.train_label_count, 96);
    assert!(fit.baseline.target_mean_native < 1000.0);
    assert!(fit.train_metrics.rmse_native.unwrap() < 1e-3);
    assert!(fit.train_metrics.pearson_r.unwrap() > 0.999999);

    let validation =
        evaluate_foundation_ccs_physics_baseline(&records, &validation_indices, &fit.baseline)
            .unwrap();
    assert_eq!(validation.label_count, 1);
    assert!(validation.rmse_native.unwrap() > 1000.0);
}

#[test]
fn source_weighted_full_train_fit_matches_the_training_source_objective() {
    let coefficients = [120.0, 80.0, 160.0, 70.0, 110.0, 35.0, 0.0, 0.0];
    let sequences = [
        "PEPTIDEK",
        "AGHCEWQMK",
        "MPEPTIDER",
        "VVVVVVVVVK",
        "YGFYTPK",
        "TQDFVQK",
    ];
    let mut records = Vec::new();
    let mut provenance = Vec::new();
    let mut train_indices = Vec::new();

    // Source A contributes ten times more physical records than source B, but the
    // intended trainer objective below is 50/50. Source B has a constant +40 CCS
    // calibration shift that the source-agnostic physical prior can only compromise
    // between. Repeating identical feature patterns isolates the weighting effect.
    for source_index in 0..2usize {
        let source_id = if source_index == 0 {
            "source_a"
        } else {
            "source_b"
        };
        let repeats = if source_index == 0 { 10 } else { 1 };
        let offset = if source_index == 0 { 0.0 } else { 40.0 };
        let mut source_record_index = 0usize;
        for _ in 0..repeats {
            for pattern in 0..24usize {
                let sequence = sequences[pattern % sequences.len()];
                let charge = Some(2 + (pattern % 3) as i32);
                let mz = Some(375.0 + (pattern % 8) as f32 * 55.0);
                let probe = record(sequence, charge, mz, 0.0);
                let features = foundation_ccs_physics_features(&probe.peptidoform, &probe.context);
                let target = coefficients
                    .iter()
                    .zip(features)
                    .map(|(coefficient, feature)| *coefficient * feature)
                    .sum::<f64>()
                    + offset;
                let index = records.len();
                records.push(record(sequence, charge, mz, target));
                provenance.push(FoundationRecordProvenance {
                    source_index,
                    source_id: source_id.to_string(),
                    source_record_index,
                });
                train_indices.push(index);
                source_record_index += 1;
            }
        }
    }

    let uniform = fit_foundation_ccs_physics_baseline(
        &records,
        &train_indices,
        FoundationCcsPhysicsFitConfig { ridge_lambda: 1e-8 },
    )
    .unwrap();
    let source_weights =
        BTreeMap::from([("source_a".to_string(), 0.5), ("source_b".to_string(), 0.5)]);
    let weighted = fit_foundation_ccs_physics_baseline_source_weighted(
        &records,
        &provenance,
        &train_indices,
        &source_weights,
        FoundationCcsPhysicsFitConfig { ridge_lambda: 1e-8 },
    )
    .unwrap();

    assert_eq!(weighted.source_weight_summaries.len(), 2);
    assert!((weighted.effective_weight_sum - train_indices.len() as f64).abs() < 1e-8);
    assert_eq!(
        weighted.baseline.target_mean_native,
        uniform.baseline.target_mean_native
    );
    assert_eq!(
        weighted.baseline.target_std_native,
        uniform.baseline.target_std_native
    );

    // Build a balanced evaluation set from one matched feature panel per source.
    let mut validation_indices = Vec::new();
    for source_index in 0..2usize {
        let offset = if source_index == 0 { 0.0 } else { 40.0 };
        for pattern in 0..24usize {
            let sequence = sequences[pattern % sequences.len()];
            let charge = Some(2 + (pattern % 3) as i32);
            let mz = Some(375.0 + (pattern % 8) as f32 * 55.0);
            let probe = record(sequence, charge, mz, 0.0);
            let features = foundation_ccs_physics_features(&probe.peptidoform, &probe.context);
            let target = coefficients
                .iter()
                .zip(features)
                .map(|(coefficient, feature)| *coefficient * feature)
                .sum::<f64>()
                + offset;
            let index = records.len();
            records.push(record(sequence, charge, mz, target));
            validation_indices.push(index);
        }
    }

    let uniform_validation =
        evaluate_foundation_ccs_physics_baseline(&records, &validation_indices, &uniform.baseline)
            .unwrap();
    let weighted_validation =
        evaluate_foundation_ccs_physics_baseline(&records, &validation_indices, &weighted.baseline)
            .unwrap();
    assert!(
        weighted_validation.rmse_native.unwrap() < uniform_validation.rmse_native.unwrap() - 5.0
    );
}

#[test]
fn trainer_rejects_a_physics_prior_whose_target_space_does_not_match_ccs_normalization() {
    use candle_core::Device;
    use redeem_properties::foundation::{
        FoundationCcsPhysicsBaselineConfig, FoundationConfig, FoundationRegressionNormalization,
        FoundationRegressionNormalizationStrategy, FoundationTargetNormalizationConfig,
        FoundationTrainer, FoundationTrainerConfig,
    };

    let model = FoundationConfig {
        max_sequence_len: 16,
        graph_hidden_dim: 16,
        graph_layers: 1,
        model_dim: 32,
        num_attention_heads: 4,
        transformer_ff_dim: 64,
        transformer_layers: 1,
        contrastive_dim: 16,
        dropout: 0.0,
        ccs_physics_baseline: Some(FoundationCcsPhysicsBaselineConfig {
            coefficients_native: [0.0; 8],
            target_mean_native: 450.0,
            target_std_native: 90.0,
        }),
        ..FoundationConfig::default()
    };
    let trainer = FoundationTrainerConfig {
        target_normalization: FoundationTargetNormalizationConfig {
            ccs: FoundationRegressionNormalization {
                strategy: FoundationRegressionNormalizationStrategy::TrainStandardize,
                mean: Some(451.0),
                standard_deviation: Some(90.0),
                label_count: 10,
                ..FoundationRegressionNormalization::default()
            },
            ..FoundationTargetNormalizationConfig::default()
        },
        ..FoundationTrainerConfig::default()
    };

    let error = match FoundationTrainer::new(model, trainer, Device::Cpu) {
        Ok(_) => panic!("mismatched CCS physics target space should be rejected"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("target-space statistics do not match"));
}
