//! SVM classifier adapter (feature-gated).
//!
//! Wraps the `light-svm` crate to provide a `ClassifierModel` implementation
//! with optional Platt calibration for probability estimates.
use light_svm::{
    calibration::PlattCalibrator,
    data::{CsrMatrix, DenseMatrix as LightDenseMatrix},
    svc::{ClassStrategy, DecisionScores, LinearSVC},
};

use crate::config::{ModelConfig, ModelType};
use crate::math::Array2;
use crate::models::classifier_trait::ClassifierModel;

pub struct SVMClassifier {
    params: ModelConfig,
    model: Option<LinearSVC>,
    calibrator: Option<PlattCalibrator>,
    n_features: Option<usize>,
}

impl SVMClassifier {
    pub fn new(params: ModelConfig) -> Self {
        Self {
            params,
            model: None,
            calibrator: None,
            n_features: None,
        }
    }

    fn build_model(&self) -> LinearSVC {
        if let ModelType::SVM { eps, c, kernel, .. } = &self.params.model_type {
            if kernel != "linear" {
                log::warn!(
                    "light-svm currently supports linear kernels; requested `{}` will be treated as linear",
                    kernel
                );
            }
            let mut params = light_svm::SvmParams::builder()
                .solver(light_svm::solver::Solver::Dcd)
                .tol(*eps as f32)
                .max_epochs(200)
                .fit_intercept(true)
                .penalize_intercept(false)
                .projection(false)
                .build();
            params.c_neg = Some(c.0 as f32);
            params.c_pos = Some(c.1 as f32);

            LinearSVC::builder()
                .class_strategy(ClassStrategy::Binary)
                .params(params)
                .build()
        } else {
            panic!("SVMClassifier expected ModelType::SVM configuration");
        }
    }

    fn dense_rows(x: &Array2<f32>) -> Vec<Vec<f32>> {
        (0..x.nrows())
            .map(|row| x.row_slice(row).to_vec())
            .collect()
    }

    fn collect_scores(model: &LinearSVC, csr: &CsrMatrix) -> Vec<f32> {
        match model.decision_function(csr) {
            DecisionScores::Binary { scores, .. } => scores,
            _ => panic!("Binary strategy expected for SVMClassifier"),
        }
    }

    fn decision_scores(&self, x: &Array2<f32>) -> Vec<f32> {
        let model = self.model.as_ref().expect("SVM model not trained");
        let rows = Self::dense_rows(x);
        let csr = CsrMatrix::from_dense(&rows, 0.0);
        Self::collect_scores(model, &csr)
    }
}

impl SVMClassifier {
    pub fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        _x_eval: Option<&Array2<f32>>,
        _y_eval: Option<&[i32]>,
    ) {
        let labels: Vec<i32> = y.iter().map(|&val| if val == 1 { 1 } else { -1 }).collect();
        self.n_features = Some(x.ncols());
        let dense_rows = Self::dense_rows(x);
        let dense_matrix = LightDenseMatrix::from_vec_rows(&dense_rows);

        let mut model = self.build_model();
        model.fit_dense(&dense_matrix, &labels);

        self.model = Some(model);
        self.calibrator = None;
    }

    pub fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        self.decision_scores(x)
            .into_iter()
            .map(|score| if score >= 0.0 { 1.0 } else { 0.0 })
            .collect()
    }

    pub fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32> {
        // Return raw decision scores; semi-supervised ranking should be based
        // on margins rather than calibrated probabilities.
        self.decision_scores(x)
    }
}

impl ClassifierModel for SVMClassifier {
    fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        x_eval: Option<&Array2<f32>>,
        y_eval: Option<&[i32]>,
    ) {
        SVMClassifier::fit(self, x, y, x_eval, y_eval)
    }

    fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        SVMClassifier::predict(self, x)
    }

    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32> {
        SVMClassifier::predict_proba(self, x)
    }

    fn clone_box(&self) -> Box<dyn ClassifierModel> {
        Box::new(SVMClassifier::new(self.params.clone()))
    }

    fn feature_weights(&self) -> Option<Vec<f32>> {
        let model = self.model.as_ref()?;
        let n_features = self.n_features?;
        if n_features == 0 {
            return None;
        }

        let mut basis: Vec<Vec<f32>> = Vec::with_capacity(n_features + 1);
        basis.push(vec![0.0; n_features]);
        for idx in 0..n_features {
            let mut row = vec![0.0; n_features];
            row[idx] = 1.0;
            basis.push(row);
        }

        let csr = CsrMatrix::from_dense(&basis, 0.0);
        let scores = Self::collect_scores(model, &csr);
        if scores.len() != n_features + 1 {
            return None;
        }

        let bias = scores[0];
        let weights = scores[1..]
            .iter()
            .map(|score| score - bias)
            .collect::<Vec<f32>>();
        Some(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Array1, Array2};

    #[test]
    fn test_svm_classifier_linear() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 1.0, 1.1, 2.0, 1.9, 3.0, 3.0, -1.0, -1.1, -2.0, -2.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1, 1, 1, 1, -1, -1]);

        let params = ModelConfig {
            learning_rate: 0.01,
            model_type: ModelType::SVM {
                eps: 1e-4,
                c: (1.0, 1.0),
                kernel: "linear".to_string(),
                gaussian_kernel_eps: 0.0,
                polynomial_kernel_constant: 0.0,
                polynomial_kernel_degree: 0.0,
            },
        };

        let mut classifier = SVMClassifier::new(params);
        classifier.fit(&x, y.as_slice(), None, None);
        let probs = classifier.predict_proba(&x);

        assert_eq!(probs.len(), x.nrows());
        assert!(probs.iter().all(|p| *p >= 0.0 && *p <= 1.0));
    }
}
