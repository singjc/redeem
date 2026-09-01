use redeem_properties::foundation::FoundationTrainingRunConfig;
use std::path::PathBuf;

#[test]
fn shared_initialization_is_optional_and_mutually_exclusive_with_resume() {
    let mut config = FoundationTrainingRunConfig::default();
    config.benchmark_manifest = PathBuf::from("benchmark.tsv");
    config.checkpoint_root = PathBuf::from("checkpoints");
    assert!(config.initial_model_safetensors.is_none());
    assert!(config.validate().is_ok());

    config.initial_model_safetensors = Some(PathBuf::from("initial.safetensors"));
    assert!(config.validate().is_ok());

    config.resume = true;
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("mutually exclusive"), "{error}");
}
