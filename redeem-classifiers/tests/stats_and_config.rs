//! Integration tests for stats (TDC q-value estimation) and config types.

use redeem_classifiers::config::{ModelConfig, ModelType};
use redeem_classifiers::math::Array1;
use redeem_classifiers::stats::tdc;

// ---------------------------------------------------------------------------
// TDC q-value estimation
// ---------------------------------------------------------------------------

#[test]
fn tdc_basic_all_targets() {
    let scores = Array1::from_vec(vec![10.0, 8.0, 6.0, 4.0]);
    let target = Array1::from_vec(vec![true, true, true, true]);
    let qvals = tdc(&scores, &target, true);
    assert_eq!(qvals.len(), 4);
    // All targets, one decoy term (+1) → FDR = 1/N so q-values should be small
    for v in qvals.iter() {
        assert!(*v > 0.0, "q-value should be positive");
        assert!(*v <= 1.0, "q-value should be <= 1");
    }
}

#[test]
fn tdc_mixed_targets_decoys() {
    let scores = Array1::from_vec(vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0]);
    let target = Array1::from_vec(vec![true, true, false, true, false, true]);
    let qvals = tdc(&scores, &target, true);
    assert_eq!(qvals.len(), 6);

    // q-values should be non-decreasing when traversing by decreasing score
    // (monotonicity is enforced by the backward pass in tdc)
    for v in qvals.iter() {
        assert!(*v >= 0.0);
        assert!(*v <= 1.5); // FDR can exceed 1 before capping
    }
}

#[test]
fn tdc_ascending_order() {
    // desc = false: lower scores are better
    let scores = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let target = Array1::from_vec(vec![true, true, false, false]);
    let qvals = tdc(&scores, &target, false);
    assert_eq!(qvals.len(), 4);
    // Better scores (lower) should have lower q-values
    assert!(
        qvals[0] <= qvals[3],
        "lower score should have lower q-value"
    );
}

#[test]
#[should_panic(expected = "equal lengths")]
fn tdc_mismatched_lengths_panics() {
    let scores = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let target = Array1::from_vec(vec![true, false]);
    let _ = tdc(&scores, &target, true);
}

// ---------------------------------------------------------------------------
// Config / ModelType
// ---------------------------------------------------------------------------

#[test]
fn model_type_default_is_gbdt() {
    let mt = ModelType::default();
    match mt {
        ModelType::GBDT { .. } => {} // expected
        #[allow(unreachable_patterns)]
        _ => panic!("default ModelType should be GBDT"),
    }
}

#[test]
fn model_type_from_str_gbdt() {
    let mt: ModelType = "gbdt".parse().unwrap();
    match mt {
        ModelType::GBDT { max_depth, .. } => assert_eq!(max_depth, 6),
        #[allow(unreachable_patterns)]
        _ => panic!("expected GBDT"),
    }
}

#[test]
fn model_type_from_str_unknown_errors() {
    let result: Result<ModelType, _> = "random_forest".parse();
    assert!(result.is_err());
}

#[test]
fn model_config_default_values() {
    let cfg = ModelConfig::default();
    assert!(cfg.learning_rate > 0.0);
    match cfg.model_type {
        ModelType::GBDT {
            num_boost_round, ..
        } => {
            assert!(num_boost_round > 0);
        }
        #[allow(unreachable_patterns)]
        _ => panic!("default should be GBDT"),
    }
}

#[test]
fn model_config_new() {
    let mt = ModelType::GBDT {
        max_depth: 3,
        num_boost_round: 10,
        debug: false,
        training_optimization_level: 1,
        loss_type: "LogLikelyhood".to_string(),
    };
    let cfg = ModelConfig::new(0.05, mt);
    assert!((cfg.learning_rate - 0.05).abs() < 1e-6);
}

#[test]
fn model_config_serializes_to_json() {
    let cfg = ModelConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("learning_rate"));
    assert!(json.contains("GBDT"));
}

#[test]
fn model_config_round_trips_json() {
    let cfg = ModelConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: ModelConfig = serde_json::from_str(&json).unwrap();
    assert!((cfg.learning_rate - cfg2.learning_rate).abs() < 1e-6);
}
