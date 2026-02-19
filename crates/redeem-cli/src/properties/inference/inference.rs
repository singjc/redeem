use crate::properties::train::plot::{
    plot_delta_histogram, plot_error_cdf, plot_residuals_vs_feature,
};
use anyhow::{Context, Result};
use maud::{PreEscaped, html};
use redeem_properties::models::ccs_cnn_lstm_model::CCSCNNLSTMModel;
use redeem_properties::models::ccs_cnn_tf_model::CCSCNNTFModel;
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::models::rt_cnn_lstm_model::RTCNNLSTMModel;
use redeem_properties::models::rt_cnn_transformer_model::RTCNNTFModel;
use redeem_properties::utils::data_handling::{PeptideData, TargetNormalization};
use redeem_properties::utils::peptdeep_utils::{MODIFICATION_MAP, load_modifications};
use redeem_properties::utils::stats::Metrics;
use redeem_properties::utils::utils::get_device;
use report_builder::{Report, ReportSection, plots::plot_scatter};

use crate::properties::inference::input::PropertyInferenceConfig;
use crate::properties::inference::output::write_peptide_data;
use crate::properties::load_data::load_peptide_data;
use crate::properties::train::{sample_indices, sample_peptides};
use crate::properties::util::write_bytes_to_file;

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    let modifications = load_modifications().context("Failed to load modifications")?;

    // If requested, force decoder head selection via env var so the model loader can honor it.
    if let Some(head) = &config.head_type {
        unsafe {
            std::env::set_var("REDEEM_RT_DECODER_HEAD", head);
        }
        log::info!("Forcing decoder head selection to '{}'", head);
    }

    // If provided, use explicit normalization override; otherwise fall back to reference-data stats.
    let norm_override = if let Some(norm_type) = &config.normalization_override_type {
        match norm_type.as_str() {
            "min_max" => {
                if let (Some(min), Some(max)) = (
                    config.normalization_override_min,
                    config.normalization_override_max,
                ) {
                    log::info!(
                        "Using explicit normalization override: MinMax min={} max={}",
                        min,
                        max
                    );
                    Some(TargetNormalization::MinMax(min, max))
                } else {
                    return Err(anyhow::anyhow!(
                        "normalization_override_type=min_max requires normalization_override_min and normalization_override_max"
                    ));
                }
            }
            "z_score" => {
                if let (Some(mean), Some(std)) = (
                    config.normalization_override_mean,
                    config.normalization_override_std,
                ) {
                    log::info!(
                        "Using explicit normalization override: ZScore mean={} std={}",
                        mean,
                        std
                    );
                    Some(TargetNormalization::ZScore(mean, std))
                } else {
                    return Err(anyhow::anyhow!(
                        "normalization_override_type=z_score requires normalization_override_mean and normalization_override_std"
                    ));
                }
            }
            other => {
                return Err(anyhow::anyhow!(
                    "Unsupported normalization_override_type: {} (use \"min_max\" or \"z_score\")",
                    other
                ));
            }
        }
    } else if let Some(ref_path) = &config.normalization_reference_data {
        let (_, norm) = load_peptide_data(
            ref_path,
            &config.model_arch,
            Some(config.nce),
            Some(config.instrument.clone()),
            Some(config.normalization.clone().unwrap()),
            None,
            true,
            &modifications,
        )?;
        log::info!(
            "Using normalization stats from reference data: {}",
            ref_path
        );
        Some(norm)
    } else {
        None
    };

    // Load inference data (optionally applying the reference normalization)
    let (inference_data, norm_factor) = load_peptide_data(
        &config.inference_data,
        &config.model_arch,
        Some(config.nce),
        Some(config.instrument.clone()),
        Some(config.normalization.clone().unwrap()),
        norm_override.clone(),
        true,
        &modifications,
    )?;
    log::info!("Loaded {} peptides", inference_data.len());

    // Dispatch model training based on architecture
    let model_arch = config.model_arch.as_str();
    let device = get_device(&config.device)?;

    let mut model: Box<dyn ModelInterface + Send + Sync> = match model_arch {
        "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "rt_cnn_tf" => Box::new(RTCNNTFModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported RT model architecture: {}",
                model_arch
            ));
        }
    };

    // (suppressed) Normalization parameters are not written to files by default.
    // The `norm_factor` value remains available in-memory and is included in
    // the generated inference report when appropriate.

    // Dump tensors from the model for parity/debugging (match trainer behavior)
    match redeem_properties::models::model_interface::load_tensors_from_model(
        &config.model_path,
        &device,
    ) {
        Ok(tensors) => {
            let mut list_lines: Vec<String> = Vec::new();
            for (name, tensor) in tensors.iter() {
                // Keep special "scale" dump as a separate artifact to aid debugging
                let lname = name.to_lowercase();
                // (suppressed) Per-tensor scale dumps are disabled to avoid producing
                // analysis artifacts during normal CLI inference runs.

                // For parity with the trainer, write a summary line for every tensor
                let mut info_line = format!("{} shape={:?}", name, tensor.shape());
                if let Ok(flat) = tensor.clone().flatten_all() {
                    if let Ok(vals) = flat.to_vec1::<f32>() {
                        let sample = vals
                            .iter()
                            .take(8)
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        info_line = format!("{} sample=[{}]", info_line, sample);
                    }
                }
                list_lines.push(info_line);
            }
            // (suppressed) Model tensor list generation to `analysis/` is disabled.

            // Also write a deterministic small sample for load-time parity checks.
            // Prefer a known decoder tensor name; fall back to first decoder-like tensor.
            let sample_out = "analysis/model_load_sample_inference.txt";
            let mut sample_lines: Vec<String> = Vec::new();
            sample_lines.push(format!("model_path={}", &config.model_path));
            if let Some((name, tensor)) = tensors
                .iter()
                .find(|(n, _)| n.to_lowercase().contains("rt_decoder.nn.0.weight"))
                .cloned()
            {
                if let Ok(flat) = tensor.clone().flatten_all() {
                    if let Ok(vals) = flat.to_vec1::<f32>() {
                        let sample = vals
                            .iter()
                            .take(8)
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        sample_lines.push(format!("{} sample=[{}]", name, sample));
                    }
                }
            } else if let Some((name, tensor)) = tensors
                .iter()
                .find(|(n, _)| n.to_lowercase().contains("decoder"))
                .cloned()
            {
                if let Ok(flat) = tensor.clone().flatten_all() {
                    if let Ok(vals) = flat.to_vec1::<f32>() {
                        let sample = vals
                            .iter()
                            .take(8)
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        sample_lines.push(format!("{} sample=[{}]", name, sample));
                    }
                }
            }
            // (suppressed) Deterministic model-load sample is not written to disk.
        }
        Err(e) => {
            log::debug!(
                "Failed to read tensors from model for scale inspection: {:?}",
                e
            );
        }
    }

    let start_time = std::time::Instant::now();
    model.set_evaluation_mode();
    let inference_results: Vec<PeptideData> = model.inference(
        &inference_data,
        config.batch_size,
        modifications.clone(),
        norm_factor,
    )?;
    log::info!("Inference completed in {:?}", start_time.elapsed());

    log::info!("Predictions saved to: {}", config.output_file);
    let normalize_field = if config.model_arch.contains("ccs") {
        "ccs"
    } else {
        "retention time"
    };
    write_peptide_data(
        &inference_results,
        &inference_data,
        norm_factor,
        normalize_field,
        &config.output_file,
    )?;

    // Generate report
    let mut report = Report::new(
        "ReDeeM",
        &config.version,
        Some("https://github.com/singjc/redeem/blob/master/img/redeem_logo.png?raw=true"),
        &format!("ReDeeM {:?} Inference Report", config.model_arch),
    );

    /* Section 1: Overview */
    {
        let mut overview_section = ReportSection::new("Overview");

        overview_section.add_content(html! {
            "This report summarizes the inference process of the ReDeeM model."
        });

        // Inference scatter plot
        // Index-based sampling: choose random indices from the full dataset,
        // then extract the corresponding inputs and predictions from the
        // already-computed `inference_results`. This guarantees perfect
        // alignment and avoids running inference twice.
        let sample_n = 5000usize;
        let sampled_idxs = sample_indices(inference_data.len(), sample_n);

        let mut true_and_pred: Vec<(f64, f64)> = Vec::with_capacity(sampled_idxs.len());
        for &i in &sampled_idxs {
            if i >= inference_data.len() || i >= inference_results.len() {
                continue;
            }
            let true_pep = &inference_data[i];
            let pred_pep = &inference_results[i];
            match normalize_field {
                "ccs" => match (true_pep.ccs, pred_pep.ccs) {
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
                        true_and_pred.push((t_denorm, p as f64));
                    }
                    _ => {}
                },
                _ => match (true_pep.retention_time, pred_pep.retention_time) {
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
                        true_and_pred.push((t_denorm, p as f64));
                    }
                    _ => {}
                },
            }
        }

        let (true_rt, pred_rt): (Vec<f64>, Vec<f64>) = true_and_pred.into_iter().unzip();

        // Scatter: Predicted vs True
        overview_section.add_content(html! {
            p { "Scatter plot interpretation: Each point shows one peptide with the x-value equal to the ground-truth target and the y-value equal to the model prediction. Points that lie close to the diagonal (y = x) indicate accurate predictions. Systematic offsets from the diagonal indicate bias; increased vertical spread indicates larger error variance." }
        });

        let scatter_plot = plot_scatter(
            &vec![true_rt.clone()],
            &vec![pred_rt.clone()],
            vec!["Prediction".to_string()],
            "Predicted vs True (Random 1000 Validation Peptides)",
            "Target",
            "Predicted",
        )
        .unwrap();
        overview_section.add_plot(scatter_plot);

        // Additional diagnostics: delta histogram, error CDF, residuals vs features
        let mut deltas: Vec<f64> = Vec::with_capacity(true_rt.len());
        let mut abs_errors: Vec<f64> = Vec::with_capacity(true_rt.len());
        for (t, p) in true_rt.iter().zip(pred_rt.iter()) {
            let d = p - t;
            deltas.push(d);
            abs_errors.push(d.abs());
        }

        // Summary metrics (MAE / RMSE / R2) using f32 helpers
        let pred_f32: Vec<f32> = pred_rt.iter().map(|v| *v as f32).collect();
        let true_f32: Vec<f32> = true_rt.iter().map(|v| *v as f32).collect();
        let mae = Metrics::mae(&pred_f32, &true_f32);
        let rmse = Metrics::rmse(&pred_f32, &true_f32);
        let r2 = Metrics::r2(&pred_f32, &true_f32);

        overview_section.add_content(html! {
            p { (format!("MAE={:.4}, RMSE={:.4}, R2={:.4}", mae, rmse, r2)) }
        });

        overview_section.add_content(html! {
            p { "Metric interpretation: MAE (mean absolute error) describes the average magnitude of prediction errors in the same units as the target. RMSE (root-mean-square error) penalizes larger errors more strongly; use it when large deviations are particularly undesirable. R² measures the fraction of variance explained by the model (1.0 is perfect). Typical good values depend on the task: for CCS a low MAE (e.g., < 5 Å^2 depending on units) and R² approaching 1.0 indicate strong performance, while larger MAE or low/negative R² indicate model issues or mismatched normalization." }
        });

        // Delta histogram
        overview_section.add_content(html! {
            p { "Delta histogram explanation: This histogram shows the distribution of prediction errors (Predicted − True). The center should be close to zero for an unbiased model. Narrower histograms indicate smaller typical errors; skew or heavy tails indicate bias or occasional large mistakes. Use the histogram to detect systematic offsets and outliers." }
        });
        let delta_plot = plot_delta_histogram(
            &deltas,
            "Prediction Error Histogram",
            "Predicted - True",
            "Count",
        );
        overview_section.add_plot(delta_plot);

        // Absolute error CDF
        overview_section.add_content(html! {
            p { "Absolute error CDF interpretation: The CDF shows the fraction of peptides whose absolute error is below a given threshold. For example, the value at x = 2.0 tells you the proportion of predictions within ±2 units of the true value. Use percentiles (e.g., 50th, 90th) from this plot to set practical error tolerances for downstream analysis." }
        });
        let cdf_plot = plot_error_cdf(&abs_errors, "Absolute Error CDF", "Absolute Error", "CDF");
        overview_section.add_plot(cdf_plot);

        // Residuals vs peptide numeric features (length and charge)
        let mut lengths: Vec<f64> = Vec::new();
        let mut charges: Vec<f64> = Vec::new();
        let mut residuals: Vec<f64> = Vec::new();
        for &i in &sampled_idxs {
            if i >= inference_data.len() || i >= inference_results.len() {
                continue;
            }
            let true_pep = &inference_data[i];
            let pred_pep = &inference_results[i];
            match normalize_field {
                "ccs" => match (true_pep.ccs, pred_pep.ccs) {
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
                        let res = p as f64 - t_denorm;
                        residuals.push(res);
                        lengths.push(true_pep.naked_sequence_str().len() as f64);
                        charges.push(true_pep.charge.unwrap_or(0) as f64);
                    }
                    _ => {}
                },
                _ => match (true_pep.retention_time, pred_pep.retention_time) {
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
                        let res = p as f64 - t_denorm;
                        residuals.push(res);
                        lengths.push(true_pep.naked_sequence_str().len() as f64);
                        charges.push(true_pep.charge.unwrap_or(0) as f64);
                    }
                    _ => {}
                },
            }
        }

        if !lengths.is_empty() && !residuals.is_empty() {
            overview_section.add_content(html! {
                p { "Residuals vs peptide length: This scatter shows whether prediction errors correlate with peptide sequence length. A flat cloud centered on zero suggests no length bias. A trend (positive/negative slope) indicates the model systematically over- or under-predicts for longer peptides." }
            });
            let res_vs_len = plot_residuals_vs_feature(
                &lengths,
                &residuals,
                "Residuals vs Peptide Length",
                "Peptide Length",
                "Residual (Pred - True)",
            );
            overview_section.add_plot(res_vs_len);
        }
        if !charges.is_empty() && !residuals.is_empty() {
            overview_section.add_content(html! {
                p { "Residuals vs charge: This plot checks for systematic prediction differences across precursor charge states. Ideally residuals should be centered near zero for all charges. Charge-dependent shifts suggest the model could benefit from better charge-aware features or stratified training." }
            });
            let res_vs_charge = plot_residuals_vs_feature(
                &charges,
                &residuals,
                "Residuals vs Charge",
                "Charge",
                "Residual (Pred - True)",
            );
            overview_section.add_plot(res_vs_charge);
        }

        report.add_section(overview_section);
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
    let path = "redeem_inference_report.html";
    report.save_to_file(&path.to_string())?;

    let path = "redeem_inference_config.json";
    let json = serde_json::to_string_pretty(&config)?;
    println!("{}", json);
    let bytes = serde_json::to_vec_pretty(&config)?;
    write_bytes_to_file(path, &bytes)?;

    Ok(())
}
