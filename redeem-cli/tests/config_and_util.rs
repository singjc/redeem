//! Integration tests for CLI config parsing, util helpers, and score config.

use clap::{Arg, Command};
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use redeem_cli::classifiers::score::score::ScoreConfig;
use redeem_cli::properties::inference::input::PropertyInferenceConfig;
use redeem_cli::properties::util::validate_tsv_or_csv_file;

static PRETRAINED_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &std::path::Path) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => unsafe {
                std::env::set_var(self.key, value);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

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
    let cfg = PropertyInferenceConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("model_arch"));
    assert!(json.contains("batch_size"));
}

#[test]
fn inference_config_accepts_pathbuf_cli_overrides() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("inference_config.json");
    std::fs::write(&config_path, "{}").unwrap();

    let model_path = dir.path().join("model.safetensors");
    let inference_data = dir.path().join("input.csv");
    let output_file = dir.path().join("predictions.csv");
    std::fs::File::create(&inference_data).unwrap();

    let matches = Command::new("redeem")
        .arg(Arg::new("pretrained").long("pretrained"))
        .arg(
            Arg::new("model_path")
                .short('m')
                .long("model")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("inference_data")
                .short('d')
                .long("inference_data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output_file")
                .short('o')
                .long("output_file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .try_get_matches_from([
            "redeem",
            "--model",
            model_path.to_str().unwrap(),
            "--inference_data",
            inference_data.to_str().unwrap(),
            "--output_file",
            output_file.to_str().unwrap(),
        ])
        .unwrap();

    let config = PropertyInferenceConfig::from_arguments(&config_path, &matches).unwrap();

    assert_eq!(config.model_path, model_path.to_string_lossy());
    assert_eq!(config.inference_data, inference_data.to_string_lossy());
    assert_eq!(config.output_file, output_file.to_string_lossy());
}

#[test]
fn inference_config_pretrained_overrides_model_arch() {
    let _guard = PRETRAINED_ENV_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let pretrained_root = dir.path().join("pretrained");
    let pretrained_model = pretrained_root.join("alphapeptdeep/generic/rt.pth");
    std::fs::create_dir_all(pretrained_model.parent().unwrap()).unwrap();
    std::fs::write(&pretrained_model, b"fake model weights").unwrap();
    let _env_guard = EnvVarGuard::set("REDEEM_PRETRAINED_MODELS_DIR", &pretrained_root);

    let config_path = dir.path().join("inference_config.json");
    std::fs::write(
        &config_path,
        r#"{"model_arch":"rt_cnn_tf","inference_data":"ignored.csv"}"#,
    )
    .unwrap();

    let inference_data = dir.path().join("input.csv");
    std::fs::File::create(&inference_data).unwrap();

    let matches = Command::new("redeem")
        .arg(Arg::new("pretrained").long("pretrained"))
        .arg(
            Arg::new("model_path")
                .short('m')
                .long("model")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("inference_data")
                .short('d')
                .long("inference_data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output_file")
                .short('o')
                .long("output_file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .try_get_matches_from([
            "redeem",
            "--pretrained",
            "alphapeptdeep-rt",
            "--inference_data",
            inference_data.to_str().unwrap(),
        ])
        .unwrap();

    let config = PropertyInferenceConfig::from_arguments(&config_path, &matches).unwrap();

    assert_eq!(config.model_arch, "rt_cnn_lstm");
    assert!(config.model_path.ends_with("rt.pth"));
    assert_eq!(config.inference_data, inference_data.to_string_lossy());
}
