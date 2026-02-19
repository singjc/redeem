use anyhow::{Context, Result};
use maud::{PreEscaped, html};
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::models::{
    ccs_cnn_lstm_model::CCSCNNLSTMModel, ccs_cnn_tf_model::CCSCNNTFModel,
    rt_cnn_lstm_model::RTCNNLSTMModel, rt_cnn_transformer_model::RTCNNTFModel,
};
use redeem_properties::utils::data_handling::{PeptideData, TargetNormalization};
use redeem_properties::utils::peptdeep_utils::load_modifications;
use redeem_properties::utils::utils::get_device;
use report_builder::{Report, ReportSection, plots::plot_scatter};

use crate::properties::load_data;
use crate::properties::train::plot::{plot_losses, plot_training_metric};
use crate::properties::train::sample_peptides;
use crate::properties::util::write_bytes_to_file;
use input::PropertyTrainConfig;
use load_data::load_peptide_data;

use super::input;

pub fn run_training(config: &PropertyTrainConfig) -> Result<()> {
    log::trace!("Loading modifications map");
    let modifications = load_modifications().context("Failed to load modifications")?;

    // Load training data
    let (train_peptides, norm_factor) = load_peptide_data(
        &config.train_data,
        &config.model_arch,
        Some(config.nce),
        Some(config.instrument.clone()),
        Some(config.normalization.clone().unwrap()),
        None,
        true,
        &modifications,
    )?;
    log::info!("Loaded {} training peptides", train_peptides.len());

    // Load validation data if specified
    let (val_peptides, _val_norm_factor) = if let Some(ref val_path) = config.validation_data {
        let (peptides, norm) = load_peptide_data(
            val_path,
            &config.model_arch,
            Some(config.nce),
            Some(config.instrument.clone()),
            Some(config.normalization.clone().unwrap()),
            Some(norm_factor.clone()),
            true,
            &modifications,
        )
        .context("Failed to load validation data")?;
        (Some(peptides), Some(norm))
    } else {
        (None, None)
    };

    if let Some(ref val_data) = val_peptides {
        log::info!("Loaded {} validation peptides", val_data.len());
    } else {
        log::warn!("No validation data provided.");
    }

    // Dispatch model training based on architecture
    let model_arch = config.model_arch.as_str();
    let device = get_device(&config.device)?;
    log::trace!(
        "Loading model architecture: {} on device: {:?}",
        model_arch,
        device
    );

    let mut model: Box<dyn ModelInterface + Send + Sync> = match &config.checkpoint_file {
        Some(checkpoint_path) => {
            log::info!("Loading model from checkpoint: {}", checkpoint_path);
            match config.model_arch.as_str() {
                "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                "rt_cnn_tf" => Box::new(RTCNNTFModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported model architecture: {}",
                        config.model_arch
                    ));
                }
            }
        }
        None => match config.model_arch.as_str() {
            "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new_untrained(device.clone())?),
            "rt_cnn_tf" => Box::new(RTCNNTFModel::new_untrained_with_options(
                device.clone(),
                &config.head_type,
                config.head_learnable_scaler,
            )?),
            "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new_untrained(device.clone())?),
            "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new_untrained_with_options(
                device.clone(),
                &config.head_type,
                config.head_learnable_scaler,
            )?),
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported model architecture: {}",
                    config.model_arch
                ));
            }
        },
    };

    log::trace!("Model loaded successfully");

    let start_time = std::time::Instant::now();
    log::trace!("Training started");
    let train_step_metrics = model
        .train(
            &train_peptides,
            val_peptides.as_ref(),
            modifications.clone(),
            config.batch_size,
            config.validation_batch_size.unwrap_or(config.batch_size),
            config.learning_rate as f64,
            config.epochs,
            config.early_stopping_patience,
            "training",
            true,
            true,
            norm_factor.clone(),
            config.train_var_prefixes.clone(),
            config.warmup_fraction.map(|v| v as f64),
        )
        .with_context(|| "Training failed: an error occurred during the model training process")?;
    log::info!("Training completed in {:?}", start_time.elapsed());
    model.save(&config.output_file)?;
    log::info!("Model saved to: {}", config.output_file);

    // Generate report
    let mut report = Report::new(
        "ReDeeM",
        &config.version,
        Some("https://github.com/singjc/redeem/blob/master/img/redeem_logo_new.png?raw=true"),
        &format!("ReDeeM {:?} Trainer Report", config.model_arch),
    );

        // Section 1: Overview (training metrics)
        let mut overview_section = ReportSection::new("Overview");

        overview_section.add_content(html! {
            "This report summarizes the training process of the {} model. It includes epoch-level summaries and step-wise dynamics such as learning rate scheduling and accuracy tracking over time. These plots provide insight into model convergence behavior and training stability."
        });

        let epoch_losses = train_step_metrics.summarize_loss_for_plotting();
        let losses_plot = plot_losses(&epoch_losses);
        overview_section.add_plot(losses_plot);

        // Explanation for epoch losses
        overview_section.add_content(html! {
            div {
                p { "Epoch losses: look for a decreasing trend across epochs and eventual stabilization. Sudden spikes indicate instability or sharp LR changes; a flat line early may indicate underfitting." }
            }
        });

        // Step-wise learning rate plot
        let lr_plot = plot_training_metric(
            &train_step_metrics,
            "lr",
            "Learning Rate Over Steps",
            "Step",
            "Learning Rate",
        );
        overview_section.add_plot(lr_plot);

        // Explanation for learning rate
        overview_section.add_content(html! {
            div {
                p { "Learning rate: shows schedule/updates during training. Correlate LR drops or warmups with loss changes; large LR can cause noisy losses." }
            }
        });

        // Step-wise loss plot
        let step_loss_plot = plot_training_metric(
            &train_step_metrics,
            "loss",
            "Loss Over Steps",
            "Step",
            "Loss",
        );
        overview_section.add_plot(step_loss_plot);

        // Explanation for step-wise loss
        overview_section.add_content(html! {
            div {
                p { "Step-wise loss: expected to be noisy but trending downward. Persistent high variance suggests batch instability, poor normalization, or inappropriate optimizer settings." }
            }
        });

        // Step-wise regression quality plots (MAE, RMSE, R²)
        let mae_plot = plot_training_metric(&train_step_metrics, "mae", "MAE Over Steps", "Step", "MAE");
        overview_section.add_plot(mae_plot);

        // Explanation for MAE
        overview_section.add_content(html! {
            div {
                p { "MAE (Mean Absolute Error): reports the average absolute prediction error over steps. Use it alongside loss to assess practical prediction quality; lower is better." }
            }
        });

        let rmse_plot = plot_training_metric(
            &train_step_metrics,
            "rmse",
            "RMSE Over Steps",
            "Step",
            "RMSE",
        );
        overview_section.add_plot(rmse_plot);

        // Explanation for RMSE
        overview_section.add_content(html! {
            div {
                p { "RMSE (Root Mean Squared Error): emphasizes larger errors. A rising RMSE relative to MAE indicates increasing outliers; aim for both to decrease." }
            }
        });

        let r2_plot = plot_training_metric(&train_step_metrics, "r2", "R² Over Steps", "Step", "R²");
        overview_section.add_plot(r2_plot);

        // Explanation for R²
        overview_section.add_content(html! {
            div {
                p { "R²: coefficient of determination (higher is better). Values near 1 indicate strong fit. Large gaps between train and validation R² may indicate overfitting." }
            }
        });

        report.add_section(overview_section);

        // Section: Validation (inference diagnostics)
        {
            let mut validation_section = ReportSection::new("Validation");

            // Inference scatter plot
            let val_peptides: Vec<PeptideData> = sample_peptides(&val_peptides.as_ref().unwrap(), 5000);
            // (suppressed) Previously this wrote sampled peptide sequences to `analysis/`
            // for offline comparison with standalone inference. That diagnostic I/O has
            // been removed to keep runs clean; the sampled sequences are still available
            // in-memory via `val_peptides` during the report generation.

            // Dump model tensors (trainer) to analysis for parity with the standalone inference run
            // (suppressed) Previously this dumped trainer model tensor info into `analysis/`
            // for offline parity checks with standalone inference. That diagnostic I/O is
            // intentionally disabled to avoid generating files during normal training runs.
            // If you need these diagnostics again, re-enable them manually.

            // Ensure model is in evaluation mode for inference (disables dropout / training-specific layers)
            model.set_evaluation_mode();
            let inference_results: Vec<PeptideData> = model.inference(&val_peptides, config.batch_size, modifications, norm_factor)?;

            // (suppressed) Trainer raw predictions are no longer written to `analysis/`.
            // The predictions remain available in `inference_results` for report generation
            // and any downstream processing; re-enable file dumps only when explicitly needed.
            let (true_rt, pred_rt): (Vec<f64>, Vec<f64>) = val_peptides
                .iter()
                .zip(&inference_results)
                .filter_map(|(true_pep, pred_pep)| {
                    // check if model is RT or CCS
                    if config.model_arch == "ccs_cnn_lstm" || config.model_arch == "ccs_cnn_tf" {
                        match (true_pep.ccs, pred_pep.ccs) {
                            (Some(t), Some(p)) => {
                                let t_denorm = match norm_factor {
                                    TargetNormalization::ZScore(mean, std) => {
                                        t as f64 * std as f64 + mean as f64
                                    }
                                    TargetNormalization::MinMax(min, range) => {
                                        t as f64 * range as f64 + min as f64
                                    }
                                    TargetNormalization::None => t as f64,
                                };
                                Some((t_denorm, p as f64))
                            }
                            _ => None,
                        }
                    } else if config.model_arch == "rt_cnn_lstm" || config.model_arch == "rt_cnn_tf" {
                        match (true_pep.retention_time, pred_pep.retention_time) {
                            (Some(t), Some(p)) => {
                                let t_denorm = match norm_factor {
                                    TargetNormalization::ZScore(mean, std) => {
                                        t as f64 * std as f64 + mean as f64
                                    }
                                    TargetNormalization::MinMax(min, range) => {
                                        t as f64 * range as f64 + min as f64
                                    }
                                    TargetNormalization::None => t as f64,
                                };
                                Some((t_denorm, p as f64))
                            }
                            _ => None,
                        }
                    } else {
                        return None;
                    }
                })
                .unzip();

            let scatter_plot = plot_scatter(
                &vec![true_rt.clone()],
                &vec![pred_rt.clone()],
                vec!["Prediction".to_string()],
                "Predicted vs True (Random 1000 Validation Peptides)",
                "Target",
                "Predicted",
            )
            .unwrap();
            validation_section.add_plot(scatter_plot);

            // Explanation for Predicted vs True scatter
            validation_section.add_content(html! {
                div {
                    p { "Predicted vs True scatter: points should lie near the diagonal (1:1). A consistent offset indicates bias; slope differences indicate scale mismatch; wide spread means high variance." }
                }
            });

            // Delta histogram (predicted - true) in original RT units
            let delta: Vec<f64> = pred_rt.iter().zip(&true_rt).map(|(p, t)| p - t).collect();
            let delta_hist = crate::properties::train::plot::plot_delta_histogram(
                &delta,
                "ΔRT Distribution (Pred - True)",
                "ΔRT",
                "Count",
            );
            validation_section.add_plot(delta_hist);

            // Explanation for Δ histogram
            validation_section.add_content(html! {
                div {
                    p { "Δ (Pred - True) histogram: shows the error distribution. Ideally this is centered at zero and symmetric. Long tails or skew show outliers or bias." }
                }
            });

            // Additional diagnostics: error percentiles / CDF, residuals vs features, top-N worst examples
            // Absolute errors and percentiles
            let abs_errors: Vec<f64> = delta.iter().map(|d| d.abs()).collect();
            let mut sorted_abs = abs_errors.clone();
            sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| -> f64 {
                if sorted_abs.is_empty() {
                    return f64::NAN;
                }
                let idx = ((p / 100.0) * (sorted_abs.len() - 1) as f64).round() as usize;
                sorted_abs[idx]
            };
            let median = pct(50.0);
            let p90 = pct(90.0);
            let p95 = pct(95.0);

            // Insert a small summary table for percentiles
            validation_section.add_content(html! {
                div {
                    h4 { "Error percentiles (absolute)" }
                    table {
                        tr { th { "Metric" } th { "Value" } }
                        tr { td { "Median (50%)" } td { (format!("{:.4}", median)) } }
                        tr { td { "90th percentile" } td { (format!("{:.4}", p90)) } }
                        tr { td { "95th percentile" } td { (format!("{:.4}", p95)) } }
                    }
                }
            });

            // CDF plot
            let cdf_plot = crate::properties::train::plot::plot_error_cdf(
                &abs_errors,
                "Absolute error CDF",
                "|Pred - True|",
                "Cumulative fraction",
            );
            validation_section.add_plot(cdf_plot);

            // Explanation for CDF / percentiles
            validation_section.add_content(html! {
                div {
                    p { "Absolute error CDF & percentiles: use the median and 90/95th percentiles to summarize typical and worst-case errors. For example, \"80% within 0.5 units\" means most predictions are close to ground truth." }
                }
            });

            // Residuals vs peptide features: length, charge, number of mods
            let mut lengths = Vec::new();
            let mut charges_vec = Vec::new();
            let mut mod_counts = Vec::new();
            let mut residuals_vec = Vec::new();
            for (true_p, pred_p) in val_peptides.iter().zip(&inference_results) {
                // target value depending on arch
                let maybe_pair = if config.model_arch == "ccs_cnn_lstm" || config.model_arch == "ccs_cnn_tf" {
                    match (true_p.ccs, pred_p.ccs) {
                        (Some(t), Some(p)) => Some((t as f64, p as f64)),
                        _ => None,
                    }
                } else {
                    match (true_p.retention_time, pred_p.retention_time) {
                        (Some(t), Some(p)) => Some((t as f64, p as f64)),
                        _ => None,
                    }
                };
                if let Some((t, p)) = maybe_pair {
                    let residual = p - t;
                    residuals_vec.push(residual);
                    lengths.push(true_p.naked_sequence_str().len() as f64);
                    charges_vec.push(true_p.charge.unwrap_or(0) as f64);
                    let mod_sites = true_p.mod_sites_str();
                    let mc = if mod_sites.trim().is_empty() { 0 } else { mod_sites.split(';').filter(|s| !s.is_empty()).count() };
                    mod_counts.push(mc as f64);
                }
            }

            if !residuals_vec.is_empty() {
                let res_len_plot = crate::properties::train::plot::plot_residuals_vs_feature(
                    &lengths,
                    &residuals_vec,
                    "Residuals vs Peptide Length",
                    "Peptide length",
                    "Residual (Pred - True)",
                );
                validation_section.add_plot(res_len_plot);

                let res_charge_plot = crate::properties::train::plot::plot_residuals_vs_feature(
                    &charges_vec,
                    &residuals_vec,
                    "Residuals vs Precursor Charge",
                    "Charge",
                    "Residual (Pred - True)",
                );
                validation_section.add_plot(res_charge_plot);

                let res_mod_plot = crate::properties::train::plot::plot_residuals_vs_feature(
                    &mod_counts,
                    &residuals_vec,
                    "Residuals vs Number of Mods",
                    "Mod count",
                    "Residual (Pred - True)",
                );
                validation_section.add_plot(res_mod_plot);

                // Explanation for residuals vs features
                validation_section.add_content(html! {
                    div {
                        p { "Residuals vs feature plots: look for non-zero trends or increasing spread across bins. A flat, zero-centered trend indicates the model performs similarly across that feature (length, charge, mod count)." }
                    }
                });
            }

            // Top-N worst examples (by absolute error)
            let mut examples: Vec<(f64, &PeptideData, &PeptideData)> = Vec::new();
            for (true_p, pred_p) in val_peptides.iter().zip(&inference_results) {
                let maybe_pair = if config.model_arch == "ccs_cnn_lstm" || config.model_arch == "ccs_cnn_tf" {
                    match (true_p.ccs, pred_p.ccs) {
                        (Some(t), Some(p)) => Some((t as f64, p as f64)),
                        _ => None,
                    }
                } else {
                    match (true_p.retention_time, pred_p.retention_time) {
                        (Some(t), Some(p)) => Some((t as f64, p as f64)),
                        _ => None,
                    }
                };
                if let Some((t, p)) = maybe_pair {
                    examples.push(((p - t).abs(), true_p, pred_p));
                }
            }
            examples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let top_n = 10usize.min(examples.len());
            if top_n > 0 {
                let mut table_rows = Vec::new();
                for i in 0..top_n {
                    let (err, true_p, pred_p) = examples[i];
                    let (t, p) = if config.model_arch == "ccs_cnn_lstm" || config.model_arch == "ccs_cnn_tf" {
                        (true_p.ccs.unwrap() as f64, pred_p.ccs.unwrap() as f64)
                    } else {
                        (true_p.retention_time.unwrap() as f64, pred_p.retention_time.unwrap() as f64)
                    };
                    table_rows.push(html! {
                        tr {
                            td { (i + 1) }
                            td { (true_p.naked_sequence_str()) }
                            td { (true_p.mods_str()) }
                            td { (format!("{:.4}", t)) }
                            td { (format!("{:.4}", p)) }
                            td { (format!("{:.4}", p - t)) }
                            td { (true_p.charge.unwrap_or(0)) }
                            td { (true_p.instrument_str().unwrap_or("")) }
                        }
                    });
                }
                validation_section.add_content(html! {
                    div {
                        h4 { "Top worst prediction errors" }
                        table {
                            tr { th { "#" } th { "Sequence" } th { "Mods" } th { "True" } th { "Pred" } th { "Err" } th { "Charge" } th { "Instrument" } }
                            (PreEscaped(""))
                            @for row in table_rows.iter() {
                                (row)
                            }
                        }
                    }
                });

                // Short note on inspecting top-N examples
                validation_section.add_content(html! {
                    div {
                        p { "Top-N worst examples: inspect these sequences for recurring problems (specific modifications, unusual instruments, or featurization issues). They are useful for targeted data or feature fixes." }
                    }
                });
            }

            report.add_section(validation_section);
        }

    /* Section 2: Configuration */
    {
        let mut config_section = ReportSection::new("Configuration");
        config_section.add_content(html! {
            style {
                ".code-container {
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-family: monospace;
                    white-space: pre-wrap;
                }"
            }
            div class="code-container" {
                pre {
                    code { (PreEscaped(serde_json::to_string_pretty(&config)?)) }
                }
            }
        });
        report.add_section(config_section);
    }

    // Save the report to HTML file
    let path = "redeem_trainer_report.html";
    report.save_to_file(&path.to_string())?;

    // Save configuration to JSON file
    let path = "redeem_trainer_config.json";
    let json = serde_json::to_string_pretty(&config)?;
    println!("{}", json);
    let bytes = serde_json::to_vec_pretty(&config)?;
    write_bytes_to_file(path, &bytes)?;

    Ok(())
}
