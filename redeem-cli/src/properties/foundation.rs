use crate::properties::inference::input::PropertyInferenceConfig;
use crate::properties::train::input::PropertyTrainConfig;
use anyhow::{Context, Result};
use redeem_properties::foundation::{
    FoundationModel, FoundationTrainingConfig, load_foundation_records_from_run,
    train_foundation_model,
};
use redeem_properties::utils::utils::get_device;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn run_training(config: &PropertyTrainConfig) -> Result<()> {
    if config.train_data.trim().is_empty() {
        anyhow::bail!("foundation training requires train_data to point to a prepared run YAML");
    }
    if config.validation_data.is_some() {
        anyhow::bail!(
            "foundation training gets TRAIN/Validation assignments from the prepared benchmark manifest; validation_data must be omitted"
        );
    }

    let training = config
        .foundation
        .clone()
        .unwrap_or_else(|| FoundationTrainingConfig {
            batch_size: config.batch_size,
            learning_rate: f64::from(config.learning_rate),
            epochs: config.epochs,
            early_stopping_patience: config.early_stopping_patience,
            ..FoundationTrainingConfig::default()
        });
    let device = get_device(&config.device)?;
    let summary = train_foundation_model(
        &config.train_data,
        &config.output_file,
        training,
        config.checkpoint_file.as_deref().map(Path::new),
        device,
    )?;

    println!("foundation_training=PASS");
    println!("completed_epochs={}", summary.completed_epochs);
    println!("best_validation_loss={:.8}", summary.best_validation_loss);
    println!("checkpoint={}", summary.checkpoint.display());
    Ok(())
}

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    if config.model_path.trim().is_empty() {
        anyhow::bail!(
            "foundation inference requires model_path to point to a checkpoint directory"
        );
    }
    if config.inference_data.trim().is_empty() {
        anyhow::bail!(
            "foundation inference requires inference_data to point to a prepared run YAML"
        );
    }
    if config.batch_size == 0 {
        anyhow::bail!("foundation inference batch_size must be positive");
    }

    let device = get_device(&config.device)?;
    let model = FoundationModel::load(&config.model_path, device)?;
    let records = load_foundation_records_from_run(&config.inference_data)?;
    let output = File::create(&config.output_file)
        .with_context(|| format!("create foundation inference output {}", config.output_file))?;
    let mut writer = BufWriter::new(output);
    writeln!(
        writer,
        "record_index\tsequence\tcharge\tpredicted_rt\tpredicted_ccs\tpredicted_ms2\tinverse_mean_nll\tgeneration_succeeded\tgenerated_sequence\tgenerated_modifications"
    )?;

    let mut written = 0usize;
    for (chunk_index, chunk) in records.chunks(config.batch_size).enumerate() {
        let predictions = model.infer_records(chunk)?;
        for (row_index, (record, prediction)) in chunk.iter().zip(predictions).enumerate() {
            let record_index = chunk_index * config.batch_size + row_index;
            let ms2 = prediction
                .properties
                .ms2
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?
                .unwrap_or_default();
            let generation_succeeded = prediction.generated_peptidoform.is_some();
            let (generated_sequence, generated_modifications) = prediction
                .generated_peptidoform
                .as_ref()
                .map(|peptide| {
                    (
                        peptide.sequence.clone(),
                        peptide
                            .modifications
                            .iter()
                            .map(|modification| modification.identity_label())
                            .collect::<Vec<_>>()
                            .join(";"),
                    )
                })
                .unwrap_or_default();
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                record_index,
                record.peptidoform.sequence,
                optional_display(record.context.charge),
                optional_float(prediction.properties.rt),
                optional_float(prediction.properties.ccs),
                ms2,
                optional_float(prediction.inverse_mean_nll),
                generation_succeeded,
                generated_sequence,
                generated_modifications,
            )?;
            written += 1;
        }
    }
    writer.flush()?;

    println!("foundation_inference=PASS");
    println!("records={written}");
    println!("output={}", config.output_file);
    Ok(())
}

fn optional_float(value: Option<f32>) -> String {
    value.map(|value| format!("{value:.8}")).unwrap_or_default()
}

fn optional_display<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}
