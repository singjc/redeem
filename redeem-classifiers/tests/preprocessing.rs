//! Integration tests for the preprocessing module (Scaler, normalize_scores).

use redeem_classifiers::math::Array2;
use redeem_classifiers::preprocessing::{
    fit_scaler, fit_transform, normalize_scores, transform_all,
};

// ---------------------------------------------------------------------------
// Scaler fit / transform
// ---------------------------------------------------------------------------

#[test]
fn fit_scaler_computes_mean_and_std() {
    let x =
        Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]).unwrap();

    let sc = fit_scaler(&x);
    assert_eq!(sc.mean.len(), 2);
    assert!((sc.mean[0] - 2.5).abs() < 1e-5, "mean[0] = {}", sc.mean[0]);
    assert!((sc.mean[1] - 25.0).abs() < 1e-5, "mean[1] = {}", sc.mean[1]);
    assert!(sc.std[0] > 0.0);
    assert!(sc.std[1] > 0.0);
}

#[test]
fn transform_all_centers_data() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let sc = fit_scaler(&x);
    let t = transform_all(&x, &sc);

    // After centering, mean should be ~0
    let col_sum: f32 = (0..4).map(|r| t[(r, 0)]).sum();
    assert!(
        (col_sum / 4.0).abs() < 1e-5,
        "column mean after transform should be ~0, got {}",
        col_sum / 4.0
    );
}

#[test]
fn fit_transform_returns_standardized() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
        .unwrap();

    let t = fit_transform(&x);
    assert_eq!(t.shape(), (4, 2));

    // Each column mean should be ~0
    for c in 0..2 {
        let col_mean: f32 = (0..4).map(|r| t[(r, c)]).sum::<f32>() / 4.0;
        assert!(
            col_mean.abs() < 1e-4,
            "col {} mean after fit_transform = {}",
            c,
            col_mean
        );
    }
}

// ---------------------------------------------------------------------------
// normalize_scores
// ---------------------------------------------------------------------------

#[test]
fn normalize_scores_zero_mean_unit_var() {
    let mut scores = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    normalize_scores(&mut scores);

    let n = scores.len() as f32;
    let mean = scores.iter().sum::<f32>() / n;
    assert!(mean.abs() < 1e-5, "mean after normalization = {}", mean);

    let var: f32 = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
    assert!(
        (var - 1.0).abs() < 1e-4,
        "variance after normalization = {}",
        var
    );
}

#[test]
fn normalize_scores_empty_no_panic() {
    let mut scores: Vec<f32> = vec![];
    normalize_scores(&mut scores); // should not panic
    assert!(scores.is_empty());
}

#[test]
fn normalize_scores_single_element() {
    let mut scores = vec![42.0f32];
    normalize_scores(&mut scores);
    // With one element, mean=42, var=0 → std clamped to 1e-6
    // Result: (42 - 42) / 1e-6 = 0
    assert!(
        scores[0].abs() < 1e-2,
        "single-element should normalize to ~0"
    );
}

#[test]
fn normalize_scores_constant_values() {
    let mut scores = vec![5.0f32; 10];
    normalize_scores(&mut scores);
    // All values are the same → after normalization all should be ~0
    for s in &scores {
        assert!(s.abs() < 1e-2, "constant values should normalize to ~0");
    }
}
