//! Univariate feature selection methods following scikit-learn's API.
//!
//! See: https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::math::{Array1, Array2};

/// Compute row-wise (squared) Euclidean norms of a 2D array.
///
/// This function calculates the Euclidean norm for each row in the input matrix.
/// If `squared` is true, it returns the squared norms instead of the regular norms.
///
/// # Parameters
///
/// * `x` - A 2D array of shape (n_samples, n_features) representing the input data.
/// * `squared` - A boolean indicating whether to return squared norms.
///
/// # Returns
///
/// An array of shape (n_samples,) containing the row-wise (squared) Euclidean norms.
///
/// # Examples
///
/// ```rust
/// use redeem_classifiers::math::{Array1, Array2};
/// use redeem_classifiers::feature_selection::univariate_selection::row_norms;
///
/// let x = Array2::from_shape_vec((3, 3), vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// ]).unwrap();
/// let norms = row_norms(&x, false);
/// println!("Row norms: {:?}", norms);
/// ```
pub fn row_norms(x: &Array2<f64>, squared: bool) -> Array1<f64> {
    let mut norms = Vec::with_capacity(x.nrows());
    for row in 0..x.nrows() {
        let sum_of_squares: f64 = x.row_slice(row).iter().map(|val| val.powi(2)).sum();
        norms.push(if squared {
            sum_of_squares
        } else {
            sum_of_squares.sqrt()
        });
    }
    Array1::from_vec(norms)
}

/// Compute Pearson's r for each feature and the target.
///
/// Pearson's r is also known as the Pearson correlation coefficient.
/// This function tests the individual effect of each regressor
/// on the target variable. It is a scoring function used in feature
/// selection procedures.
///
/// # Parameters
///
/// * `x` - A 2D array of shape (n_samples, n_features) representing
///   the data matrix (features).
/// * `y` - A 1D array of shape (n_samples,) representing the target vector.
/// * `center` - A boolean indicating whether to center the data.
///   If true, both `X` and `y` will be centered by subtracting their means.
/// * `force_finite` - A boolean indicating whether to force the correlation
///   coefficients to be finite. If true, non-finite values will be replaced
///   with 0.0.
///
/// # Returns
///
/// An array of shape (n_features,) containing the Pearson's r correlation
/// coefficients for each feature.
///
/// # Examples
///
/// ```rust
/// use redeem_classifiers::math::{Array1, Array2};
/// use redeem_classifiers::feature_selection::univariate_selection::r_regression;
///
/// let x = Array2::from_shape_vec((10, 5), (0..50).map(|i| i as f64).collect()).unwrap();
/// let y = Array1::from_vec((0..10).map(|i| i as f64).collect());
/// let correlation_coefficients = r_regression(&x, &y, true, true);
/// println!("Correlation coefficients: {:?}", correlation_coefficients);
/// ```
pub fn r_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
    center: bool,
    force_finite: bool,
) -> Array1<f64> {
    let n_features = x.ncols();
    let mut y_centered = y.to_vec();
    let mut x_means = vec![0.0; n_features];
    let mut x_norms = vec![0.0; n_features];

    if center {
        let y_mean = y_centered.iter().copied().sum::<f64>() / y_centered.len() as f64;
        for value in &mut y_centered {
            *value -= y_mean;
        }

        for col in 0..n_features {
            let mut sum = 0.0;
            for row in 0..x.nrows() {
                sum += x[(row, col)];
            }
            let mean = sum / x.nrows() as f64;
            x_means[col] = mean;
        }

        for col in 0..n_features {
            let mut sum_sq = 0.0;
            for row in 0..x.nrows() {
                let centered = x[(row, col)] - x_means[col];
                sum_sq += centered * centered;
            }
            x_norms[col] = sum_sq.sqrt();
        }
    } else {
        for col in 0..n_features {
            let mut sum_sq = 0.0;
            for row in 0..x.nrows() {
                let val = x[(row, col)];
                sum_sq += val * val;
            }
            x_norms[col] = sum_sq.sqrt();
        }
    }

    let mut correlation = vec![0.0; n_features];
    for col in 0..n_features {
        let mut dot = 0.0;
        for row in 0..x.nrows() {
            let value = if center {
                x[(row, col)] - x_means[col]
            } else {
                x[(row, col)]
            };
            dot += value * y_centered[row];
        }
        correlation[col] = dot;
    }

    let y_norm = y_centered.iter().map(|v| v * v).sum::<f64>().sqrt();
    for i in 0..n_features {
        let denom = x_norms[i] * y_norm;
        correlation[i] = if denom == 0.0 {
            0.0
        } else {
            correlation[i] / denom
        };
        if force_finite && !correlation[i].is_finite() {
            correlation[i] = 0.0;
        }
    }

    Array1::from_vec(correlation)
}

/// Univariate linear regression tests returning F-statistic and p-values.
///
/// This function performs a quick linear model test for assessing
/// the effect of a single regressor on the target variable,
/// sequentially for many regressors.
///
/// # Parameters
///
/// * `x` - A 2D array of shape (n_samples, n_features) representing
///   the data matrix (features).
/// * `y` - A 1D array of shape (n_samples,) representing the target vector.
/// * `center` - A boolean indicating whether to center the data.
/// * `force_finite` - A boolean indicating whether to force F-statistics
///   and associated p-values to be finite.
///
/// # Returns
///
/// A tuple containing:
/// - An array of shape (n_features,) with F-statistics for each feature.
/// - An array of shape (n_features,) with p-values associated with each F-statistic.
///
/// # Examples
///
/// ```rust
/// use redeem_classifiers::math::{Array1, Array2};
/// use redeem_classifiers::feature_selection::univariate_selection::f_regression;
///
/// let x = Array2::from_shape_vec((10, 5), (0..50).map(|i| i as f64).collect()).unwrap();
/// let y = Array1::from_vec((0..10).map(|i| i as f64).collect());
/// let (f_statistic, p_values) = f_regression(&x, &y, true, true);
/// println!("F-statistic: {:?}", f_statistic);
/// println!("p-values: {:?}", p_values);
/// ```
pub fn f_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
    center: bool,
    force_finite: bool,
) -> (Array1<f64>, Array1<f64>) {
    let correlation_coefficient = r_regression(x, y, center, force_finite);
    let deg_of_freedom = y.len() as f64 - if center { 2.0 } else { 1.0 };

    let corr_coef_squared = correlation_coefficient.mapv(|x| x.powi(2));

    let mut f_statistics = Vec::with_capacity(corr_coef_squared.len());
    for &coef in corr_coef_squared.iter() {
        let denom = 1.0 - coef;
        let value = if denom <= 0.0 {
            f64::MAX
        } else {
            (coef / denom) * deg_of_freedom
        };
        f_statistics.push(value);
    }

    let f_dist = FisherSnedecor::new(1.0, deg_of_freedom).unwrap();
    let mut p_values = Vec::with_capacity(f_statistics.len());
    for &f in &f_statistics {
        p_values.push(1.0 - f_dist.cdf(f));
    }

    if force_finite {
        for (f_val, p_val) in f_statistics.iter_mut().zip(p_values.iter_mut()) {
            if !f_val.is_finite() {
                if f_val.is_infinite() {
                    *f_val = f64::MAX;
                    *p_val = 0.0;
                } else {
                    *f_val = 0.0;
                    *p_val = 1.0;
                }
            }
        }
    }

    (Array1::from_vec(f_statistics), Array1::from_vec(p_values))
}

/// A struct for selecting the k best features based on F-scores.
///
/// This struct implements a feature selection method similar to scikit-learn's SelectKBest
/// with f_regression as the scoring function.
pub struct SelectKBest {
    /// The number of top features to select.
    k: usize,
}

impl SelectKBest {
    /// Creates a new SelectKBest instance.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of top features to select.
    ///
    /// # Returns
    ///
    /// A new SelectKBest instance.
    pub fn new(k: usize) -> Self {
        SelectKBest { k }
    }

    /// Fits the SelectKBest model and returns the indices of the k best features.
    ///
    /// # Arguments
    ///
    /// * `x` - The feature matrix (n_samples x n_features).
    /// * `y` - The target vector.
    /// * `center` - A boolean indicating whether to center the data. Default is true.
    /// * `force_finite` - A boolean indicating whether to force F-statistics and associated p-values to be finite. Default is true.
    ///
    /// # Returns
    ///
    /// A vector of indices corresponding to the k best features.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        center: Option<bool>,
        force_finite: Option<bool>,
    ) -> Vec<usize> {
        let center = center.unwrap_or(true);
        let force_finite = force_finite.unwrap_or(true);

        let (f_scores, _) = f_regression(x, y, center, force_finite);

        // Create a vector of indices
        let mut indices: Vec<usize> = (0..f_scores.len()).collect();

        // Sort indices based on scores in ascending order using a stable sort
        indices.sort_by(|&i, &j| {
            f_scores[i]
                .partial_cmp(&f_scores[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top k features by taking the last k elements
        indices.iter().rev().take(self.k).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Array1, Array2};

    #[test]
    fn test_select_k_best() {
        // Create a feature matrix with 5 features and 10 samples
        // Features: [random, collinear with target, constant, collinear with feature 1, noise]
        let x = Array2::from_shape_vec(
            (10, 5),
            vec![
                0.1, 1.0, 5.0, 0.2, -0.3, 0.4, -1.0, 5.0, 0.8, 0.1, 0.6, 1.0, 5.0, 1.2, 0.2, 0.9,
                -1.0, 5.0, 1.8, -0.1, 1.2, 1.0, 5.0, 2.4, 0.3, 1.5, -1.0, 5.0, 3.0, 0.0, 1.8, 1.0,
                5.0, 3.6, -0.2, 2.1, -1.0, 5.0, 4.2, 0.4, 2.4, 1.0, 5.0, 4.8, -0.1, 2.7, -1.0, 5.0,
                5.4, 0.2,
            ],
        )
        .unwrap();

        // Create a target vector perfectly correlated with the second feature
        let y = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        // Create a SelectKBest instance to select the top 3 features
        let selector = SelectKBest::new(3);

        // Fit the selector and get the indices of the best features
        let selected_indices = selector.fit(&x, &y, Some(true), Some(true));

        // Print f-scores for debugging
        println!("Selected feature indices: {:?}", selected_indices);
        let (f_scores, p_values) = f_regression(&x, &y, true, true);
        println!("F-scores: {:?}", f_scores);
        println!("P-values: {:?}", p_values);
        // Print selected features for debugging
        println!("Selected features:");
        for i in selected_indices.iter() {
            println!("Feature {:?}: {:?}", i, x.column(*i));
        }

        // Check that we got 3 indices
        assert_eq!(selected_indices.len(), 3);

        // Check that the indices are within the valid range (0 to 4)
        assert!(selected_indices.iter().all(|&idx| idx < 5));

        // Check that the indices are unique
        assert!(
            selected_indices
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                == 3
        );

        // The second feature (index 1) should definitely be selected as it's perfectly correlated with the target
        assert!(selected_indices.contains(&1));

        // The third feature (index 2) should not be selected as it's constant
        assert!(!selected_indices.contains(&2));
    }
}
