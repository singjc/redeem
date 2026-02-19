use anyhow::Result;

// Feature-gated heavy imports: keep these inside the xgboost cfg so the
// example compiles even when the `xgboost` feature isn't enabled.
#[cfg(feature = "xgboost")]
use {
    anyhow::anyhow,
    anyhow::Context,
    csv::ReaderBuilder,
    maud::html,
    redeem_classifiers::config::ModelType,
    redeem_classifiers::data_handling::{PsmMetadata, RankGrouping},
    redeem_classifiers::math::{Array1, Array2},
    redeem_classifiers::psm_scorer::SemiSupervisedLearner,
    redeem_classifiers::report::plots::{plot_pp, plot_score_histogram},
    report_builder::{Report, ReportSection},
    std::error::Error,
    std::fs::File,
    std::io::{BufReader, Write},
};

fn main() -> Result<()> {
    #[cfg(feature = "xgboost")]
    {
        return run_xgb_example();
    }

    eprintln!(
        "xgboost feature not enabled for this build; enable with --features xgboost to run this example"
    );
    Ok(())
}

#[cfg(feature = "xgboost")]
// The full example body is implemented below as `run_xgb_example` (re-purposed
// from the original `main`). Keep the heavy imports gated to avoid leaking
// feature-only types into non-xgboost builds.

/// Load a test PSM CSV file into feature matrix, labels, and metadata.
///
/// # Arguments
/// * `path` - Path to the CSV file
///
/// # Returns
/// A tuple of (`x`, `y`, `PsmMetadata`)
#[cfg(feature = "xgboost")]
pub fn load_test_psm_csv(path: &str) -> Result<(Array2<f32>, Array1<i32>, PsmMetadata)> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let headers = reader
        .headers()?
        .iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>();

    // Find indices
    let file_id_idx = headers.iter().position(|h| h == "file_id").unwrap();
    let spec_id_idx = headers.iter().position(|h| h == "spec_id").unwrap();
    let label_idx = headers.iter().position(|h| h == "label").unwrap();

    // Everything else is a feature
    let feature_indices: Vec<usize> = (0..headers.len())
        .filter(|&i| i != file_id_idx && i != spec_id_idx && i != label_idx)
        .collect();

    let feature_names = feature_indices
        .iter()
        .map(|&i| headers[i].clone())
        .collect::<Vec<_>>();

    let mut file_ids = Vec::new();
    let mut spec_ids = Vec::new();
    let mut labels = Vec::new();
    let mut features = Vec::new();

    for result in reader.records() {
        let record = result?;

        file_ids.push(record[file_id_idx].parse::<usize>()?);
        spec_ids.push(record[spec_id_idx].to_string());
        labels.push(record[label_idx].parse::<i32>()?);

        let row = feature_indices
            .iter()
            .map(|&i| record[i].parse::<f32>().unwrap_or(f32::NAN))
            .collect::<Vec<f32>>();

        features.extend(row);
    }

    let n_rows = labels.len();
    let n_cols = feature_indices.len();

    let x = Array2::from_shape_vec((n_rows, n_cols), features)?;
    let y = Array1::from_vec(labels);

    let metadata = PsmMetadata {
        file_id: file_ids,
        spec_id: spec_ids,
        feature_names,
        scan_nr: None,
        exp_mass: None,
    };

    Ok((x, y, metadata))
}

#[cfg(feature = "xgboost")]
fn save_predictions_to_csv(
    predictions: &Array1<f32>,
    file_path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(file_path)?;

    for &pred in predictions.iter() {
        writeln!(file, "{}", pred)?;
    }

    Ok(())
}

#[cfg(feature = "xgboost")]
fn run_psm_scorer(
    x: &Array2<f32>,
    y: &Array1<i32>,
    metadata: &PsmMetadata,
    scale_features: bool,
    normalize_scores: bool,
) -> Result<Array1<f32>> {
    // Create and train your SemiSupervisedLearner
    let xgb_params = ModelType::XGBoost {
        max_depth: 6,
        num_boost_round: 100,
        early_stopping_rounds: 10,
        verbose_eval: false,
    };
    let mut learner = SemiSupervisedLearner::new(
        xgb_params,
        0.01,
        1.0,
        5,
        10,
        Some((1.0, 1.0)),
        scale_features,
        normalize_scores,
        RankGrouping::SpecId,
    );
    // `fit` expects owned values and returns Result<(predictions, ranks)>
    let (predictions, _ranks) = learner.fit(x.clone(), y.clone(), metadata.clone())?;

    Ok(predictions)
}

#[cfg(not(feature = "xgboost"))]
#[allow(dead_code)]
fn run_psm_scorer(
    _x: &redeem_classifiers::math::Array2<f32>,
    _y: &redeem_classifiers::math::Array1<i32>,
    _metadata: &redeem_classifiers::data_handling::PsmMetadata,
) -> Result<redeem_classifiers::math::Array1<f32>> {
    unimplemented!("xgboost is not available in this build. Please enable the xgboost feature.");
}

#[cfg(feature = "xgboost")]
fn run_xgb_example() -> Result<()> {
    env_logger::init();

    // Load the test data from the TSV files
    let (x, y, metadata) = load_test_psm_csv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_with_metadata_for_testing_redeem.csv")?;

    println!("Loaded features shape: {:?}", x.shape());
    println!("Loaded labels shape: {:?}", y.shape());

    // Detect optional flags and pass them to the learner so preprocessing
    // happens consistently inside the SemiSupervisedLearner.
    let scale = std::env::args().any(|a| a == "--scale");
    let normalize_scores = std::env::args().any(|a| a == "--normalize-scores");

    let mut predictions = run_psm_scorer(&x, &y, &metadata, scale, normalize_scores)
        .context("Failed to run PSM scorer")?;

    // Print a concise sample of labels and predictions (avoid huge dumps)
    // Convert labels to Vec and print a short sample
    let y_vec_full = y.to_vec();
    let y_sample = y_vec_full.len().min(10);
    println!(
        "Labels: len={} first {} = {:?}",
        y_vec_full.len(),
        y_sample,
        &y_vec_full[..y_sample]
    );

    // Evaluate the predictions (print concise sample)
    // Convert predictions to Vec and print a short sample
    let preds_vec_full = predictions.to_vec();
    let p_sample = preds_vec_full.len().min(10);
    println!(
        "Predictions: len={} first {} = {:?}",
        preds_vec_full.len(),
        p_sample,
        &preds_vec_full[..p_sample]
    );
    // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();

    // Create a report similar to the GBDT example
    let mut report = Report::new(
        "Sage Report",
        "14",
        Some("/home/singjc/Documents/github/redeem/img/redeem_logo.png"),
        "XGBoost Data Analysis Report",
    );

    let mut intro_section = ReportSection::new("Introduction");
    intro_section.add_content(html! {
        "This report contains XGBoost score distributions and diagnostics."
    });
    report.add_section(intro_section);

    // convert predictions and labels
    let preds_vec = predictions.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let y_vec = y.iter().copied().collect::<Vec<i32>>();

    let plot = plot_score_histogram(&preds_vec, &y_vec, "XGBoost Score", "Score")
        .map_err(|e| anyhow!(e))?;
    let pp_plot = plot_pp(&preds_vec, &y_vec, "XGBoost Score").map_err(|e| anyhow!(e))?;

    let mut plot_section = ReportSection::new("Score Distribution");
    plot_section.add_content(html! { "This plot shows the distribution of the XGBoost scores." });
    plot_section.add_plot(plot);
    plot_section.add_content(html! { "P-P plot comparing ECDF distributions." });
    plot_section.add_plot(pp_plot);
    report.add_section(plot_section);

    report.save_to_file("report_xgb.html")?;
    println!("Report saved to report_xgb.html");

    Ok(())
}
