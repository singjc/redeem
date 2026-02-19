fn main() {
    #[cfg(feature = "xgboost")]
    {
        run_xgb_synthetic();
        return;
    }

    eprintln!("xgboost feature not enabled for this build; enable with --features xgboost to run this example");
}

#[cfg(feature = "xgboost")]
fn run_xgb_synthetic() {
    use redeem_classifiers::config::ModelType;
    use redeem_classifiers::math::{Array1, Array2};
    use redeem_classifiers::models::xgboost::XGBoostClassifier;

    env_logger::init();

    // Tiny synthetic dataset: 6 samples, 2 features
    // Labels alternate 1, -1 so model has signal to learn
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 0.0, // class 1
            0.0, 1.0, // class -1
            1.0, 0.1, // class 1
            0.0, 0.9, // class -1
            1.1, 0.0, // class 1
            0.0, 1.2, // class -1
        ],
    )
    .expect("failed to create feature matrix");

    let y = Array1::from_vec(vec![1i32, -1i32, 1i32, -1i32, 1i32, -1i32]);

    println!("Synthetic X shape: {:?}", x.shape());
    println!("Synthetic y shape: {:?}", y.shape());

    // Model params: small number of boosting rounds for speed
    let params = redeem_classifiers::config::ModelConfig {
        learning_rate: 0.3,
        model_type: ModelType::XGBoost {
            max_depth: 3,
            num_boost_round: 20,
            early_stopping_rounds: 5,
            verbose_eval: false,
        },
    };

    let mut clf = XGBoostClassifier::new(params);

    // Fit on the same data and use it as eval too
    clf.fit(&x, &y.to_vec(), Some(&x), Some(&y.to_vec()));

    // Predict probabilities / raw outputs
    let preds = clf.predict_proba(&x);

    println!(
        "Predictions len={} first 10 (or less) = {:?}",
        preds.len(),
        &preds[..preds.len().min(10)]
    );
}
