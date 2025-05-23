use anyhow::{Context, Result};
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use std::error::Error;
use std::fs::File;
use std::io::Write;

use redeem_classifiers::psm_scorer::SemiSupervisedLearner;
use redeem_classifiers::models::utils::ModelType;

fn read_features_tsv(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_path(path)?;

    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: Vec<f32> = record
            .iter()
            .map(|field| field.parse::<f32>())
            .collect::<Result<_, _>>()?;
        data.push(row);
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    Array2::from_shape_vec(
        (n_samples, n_features),
        data.into_iter().flatten().collect(),
    )
    .map_err(|e| e.into())
}

fn read_labels_tsv(path: &str) -> Result<Array1<i32>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_path(path)?;

    let labels: Vec<i32> = reader
        .records()
        .map(|r| {
            let record = r?;
            let value = record.get(0).ok_or_else(|| "Empty row".to_string())?;
            value.parse::<i32>().map_err(|e| e.into())
        })
        .collect::<Result<_, Box<dyn Error>>>()?;

    Ok(Array1::from_vec(labels))
}

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
    // Load the test data from the TSV files
    let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
    let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();

    // Select first 10 columns of data
    let x = x.slice(ndarray::s![.., ..10]).to_owned();

    println!("Loaded features shape: {:?}", x.shape());
    println!("Loaded labels shape: {:?}", y.shape());

    // Create and train your SemiSupervisedLearner
    let params = ModelType::SVM  {
        eps: 0.1,
        c: (1.0, 1.0),
        kernel: "linear".to_string(),
        gaussian_kernel_eps: 0.1,
        polynomial_kernel_constant: 1.0,
        polynomial_kernel_degree: 3.0
    };
    let mut learner = SemiSupervisedLearner::new(
        params,
        0.001,
        1.0,
        500,
        Some((0.15, 1.0))
    );
    let predictions = learner.fit(x, y.clone());

    println!("Labels: {:?}", y);

    // Evaluate the predictions
    println!("Predictions: {:?}", predictions);
    // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    Ok(())
}