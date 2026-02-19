//! Small preprocessing utilities shared by examples and models.
//!
//! Provides a simple Scaler for mean/std standardization and a score
//! normalization helper. The API operates on the crate math `Array2` and
//! plain slices for scores so it can be reused by different model
//! implementations.

use crate::math::Array2;

/// Simple standard scaler (per-column mean/std).
#[derive(Clone, Debug)]
pub struct Scaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Scaler {
    /// Minimum stddev to avoid division by zero when transforming.
    const MIN_STD: f32 = 1e-6;
}

/// Fit a `Scaler` from an `Array2<f32>` where rows are samples and
/// columns are features.
pub fn fit_scaler(x: &Array2<f32>) -> Scaler {
    let (nrows, ncols) = x.shape();
    assert!(
        nrows > 0 && ncols > 0,
        "fit_scaler requires non-empty matrix"
    );

    let mut mean = vec![0.0f32; ncols];
    for r in 0..nrows {
        for c in 0..ncols {
            mean[c] += x[(r, c)];
        }
    }
    let nrows_f = nrows as f32;
    for v in mean.iter_mut() {
        *v /= nrows_f;
    }

    let mut var = vec![0.0f32; ncols];
    for r in 0..nrows {
        for c in 0..ncols {
            let d = x[(r, c)] - mean[c];
            var[c] += d * d;
        }
    }
    for v in var.iter_mut() {
        *v = (*v / nrows_f).sqrt().max(Scaler::MIN_STD);
    }

    Scaler { mean, std: var }
}

/// Transform all rows using the provided `Scaler` and return a new `Array2<f32>`.
///
/// This will panic if allocation or shape creation fails; callers inside
/// this crate expect the input shapes to be valid.
pub fn transform_all(x: &Array2<f32>, sc: &Scaler) -> Array2<f32> {
    let (nrows, ncols) = x.shape();
    let mut out = Vec::with_capacity(nrows * ncols);

    for r in 0..nrows {
        for c in 0..ncols {
            let v = (x[(r, c)] - sc.mean[c]) / sc.std[c];
            out.push(v);
        }
    }

    Array2::from_shape_vec((nrows, ncols), out).expect("transform_all: shape mismatch")
}

/// Normalize a slice of scores to zero-mean, unit-variance in-place.
pub fn normalize_scores(scores: &mut [f32]) {
    let n = scores.len() as f32;
    if n == 0.0 {
        return;
    }
    let mean = scores.iter().copied().sum::<f32>() / n;
    let mut var = 0f32;
    for &s in scores.iter() {
        let d = s - mean;
        var += d * d;
    }
    let std = (var / n).sqrt().max(1e-6);
    for s in scores.iter_mut() {
        *s = (*s - mean) / std;
    }
}

/// Optional convenience: fit scaler and return transformed matrix in one call.
pub fn fit_transform(x: &Array2<f32>) -> Array2<f32> {
    let sc = fit_scaler(x);
    transform_all(x, &sc)
}
