use anyhow::Result;
use csv::ReaderBuilder;
use maud::html;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;

use redeem_classifiers::config::ModelType;
use redeem_classifiers::data_handling::{PsmMetadata, RankGrouping};
use redeem_classifiers::math::{Array1, Array2};
use redeem_classifiers::psm_scorer::SemiSupervisedLearner;
use redeem_classifiers::report::plots::{plot_pp, plot_score_histogram};
use report_builder::{Report, ReportSection};

/// Load a test PSM CSV file into feature matrix, labels, and metadata.
///
/// # Arguments
/// * `path` - Path to the CSV file
///
/// # Returns
/// A tuple of (`x`, `y`, `PsmMetadata`)
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

#[allow(dead_code)]
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

fn main() -> Result<()> {
    env_logger::init();
    // Set log level to debug
    log::set_max_level(log::LevelFilter::Debug);

    let (x, y, metadata) = load_test_psm_csv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_with_metadata_for_testing_redeem.csv")?;

    println!("Loaded features shape: {:?}", x.shape());
    println!("Loaded labels shape: {:?}", y.shape());

    // Create and train your SemiSupervisedLearner
    let params = ModelType::GBDT {
        max_depth: 3,
        num_boost_round: 10,
        debug: false,
        training_optimization_level: 1,
        loss_type: "LogLikelyhood".to_string(),
    };

    let scale = std::env::args().any(|a| a == "--scale");
    let normalize_scores = std::env::args().any(|a| a == "--normalize-scores");

    let mut learner = SemiSupervisedLearner::new(
        params,
        0.001,
        0.01,
        3,
        10,
        Some((0.15, 1.0)),
        scale,
        normalize_scores,
        RankGrouping::SpecId,
    );
    let (predictions, _ranks) = learner.fit(x, y.clone(), metadata)?;

    // Print concise samples (avoid dumping huge vectors)
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
    let preds_vec: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
    let p_sample = preds_vec.len().min(10);
    println!(
        "Predictions: len={} first {} = {:?}",
        preds_vec.len(),
        p_sample,
        &preds_vec[..p_sample]
    );
    // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();

    // Create a new report
    let mut report = Report::new(
        "Sage Report",
        "14",
        Some("/home/singjc/Documents/github/redeem/img/redeem_logo.png"),
        "My Data Analysis Report",
    );

    // Section 1: Introduction
    let mut intro_section = ReportSection::new("Introduction");
    intro_section.add_content(html! {
        "This report provides an analysis of the dataset. Below are some visualizations."
    });
    report.add_section(intro_section);

    // Score Distribution Section

    // convert the predictions to Array1<f32> to Vec<f64>
    let predictions = predictions.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    // convert the y to Array1<i32> to Vec<i32>
    let y = y.iter().copied().collect::<Vec<i32>>();

    let plot = plot_score_histogram(&predictions, &y, "GBDT Score", "Score").unwrap();
    let pp_plot = plot_pp(&predictions, &y, "GBDT Score").unwrap();

    let mut plot_section = ReportSection::new("Score Distribution");
    plot_section.add_content(html! {
        "This plot shows the distribution of the GBDT scores."
    });
    plot_section.add_plot(plot);

    plot_section.add_content(html! {
        "Now we show the P-P plot, which compares ECDF distributions."
    });
    plot_section.add_plot(pp_plot);
    report.add_section(plot_section);

    // Save the report to an HTML file
    report.save_to_file("report.html")?;

    println!("Report saved to report.html");

    Ok(())
}
