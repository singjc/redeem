//! Integration tests for CLI config parsing, util helpers, and score config.

use std::io::Write;

use redeem_cli::classifiers::score::score::ScoreConfig;
use redeem_cli::properties::util::validate_tsv_or_csv_file;

// ---------------------------------------------------------------------------
// validate_tsv_or_csv_file
// ---------------------------------------------------------------------------

#[test]
fn validate_tsv_file_exists() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.tsv");
    std::fs::File::create(&path).unwrap();
    assert!(validate_tsv_or_csv_file(path.to_str().unwrap()).is_ok());
}

#[test]
fn validate_csv_file_exists() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.csv");
    std::fs::File::create(&path).unwrap();
    assert!(validate_tsv_or_csv_file(path.to_str().unwrap()).is_ok());
}

#[test]
fn validate_wrong_extension_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.txt");
    std::fs::File::create(&path).unwrap();
    assert!(validate_tsv_or_csv_file(path.to_str().unwrap()).is_err());
}

#[test]
fn validate_nonexistent_file_errors() {
    assert!(validate_tsv_or_csv_file("/nonexistent/path/data.tsv").is_err());
}

// ---------------------------------------------------------------------------
// ScoreConfig defaults & serialization
// ---------------------------------------------------------------------------

#[test]
fn score_config_default_values() {
    let cfg = ScoreConfig::default();
    assert!(cfg.train_fdr > 0.0);
    assert!(cfg.max_iterations > 0);
    assert!(!cfg.deduplicate);
    assert!(!cfg.scale_features);
    assert!(!cfg.normalize_scores);
}

#[test]
fn score_config_serializes_to_json() {
    let cfg = ScoreConfig::default();
    let json = serde_json::to_string_pretty(&cfg).unwrap();
    assert!(json.contains("train_fdr"));
    assert!(json.contains("max_iterations"));
}

#[test]
fn score_config_round_trips_json() {
    let cfg = ScoreConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: ScoreConfig = serde_json::from_str(&json).unwrap();
    assert!((cfg.train_fdr - cfg2.train_fdr).abs() < 1e-6);
    assert_eq!(cfg.max_iterations, cfg2.max_iterations);
}

#[test]
fn score_config_loads_from_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("score_config.json");
    let json = serde_json::to_string_pretty(&ScoreConfig::default()).unwrap();
    std::fs::write(&path, json).unwrap();

    let loaded: ScoreConfig =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert!(loaded.train_fdr > 0.0);
}

// ---------------------------------------------------------------------------
// PropertyTrainConfig defaults & serialization
// ---------------------------------------------------------------------------

#[test]
fn train_config_default_values() {
    use redeem_cli::properties::train::input::PropertyTrainConfig;
    let cfg = PropertyTrainConfig::default();
    assert!(!cfg.train_data.is_empty() || cfg.train_data.is_empty()); // exists
    assert!(cfg.learning_rate > 0.0);
    assert!(cfg.epochs > 0);
    assert!(cfg.batch_size > 0);
    assert_eq!(cfg.model_arch, "rt_cnn_tf");
}

#[test]
fn train_config_serializes() {
    use redeem_cli::properties::train::input::PropertyTrainConfig;
    let cfg = PropertyTrainConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("model_arch"));
    assert!(json.contains("learning_rate"));
}

// ---------------------------------------------------------------------------
// PropertyInferenceConfig defaults & serialization
// ---------------------------------------------------------------------------

#[test]
fn inference_config_default_values() {
    use redeem_cli::properties::inference::input::PropertyInferenceConfig;
    let cfg = PropertyInferenceConfig::default();
    assert_eq!(cfg.model_arch, "rt_cnn_tf");
    assert_eq!(cfg.device, "cpu");
    assert!(cfg.batch_size > 0);
}

#[test]
fn inference_config_serializes() {
    use redeem_cli::properties::inference::input::PropertyInferenceConfig;
    let cfg = PropertyInferenceConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("model_arch"));
    assert!(json.contains("batch_size"));
}
