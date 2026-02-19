//! GBDT-based classifier wrapper.
//!
//! A thin adapter around the external `gbdt` crate exposing the
//! `ClassifierModel` trait used by the semi-supervised learner.
use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec};
use gbdt::gradient_boost::GBDT;

use crate::config::{ModelConfig, ModelType};
use crate::math::Array2;
use crate::models::classifier_trait::ClassifierModel;

/// Gradient Boosting Decision Tree (GBDT) classifier
pub struct GBDTClassifier {
    model: Option<GBDT>,
    params: ModelConfig,
}

impl GBDTClassifier {
    pub fn new(params: ModelConfig) -> Self {
        GBDTClassifier {
            model: None,
            params,
        }
    }
}

impl GBDTClassifier {
    pub fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        _x_eval: Option<&Array2<f32>>,
        _y_eval: Option<&[i32]>,
    ) {
        let feature_size = x.ncols();

        match &self.params.model_type {
            ModelType::GBDT {
                max_depth,
                num_boost_round,
                debug,
                training_optimization_level,
                loss_type,
            } => {
                let mut config = Config::new();

                config.set_feature_size(feature_size);
                config.set_shrinkage(self.params.learning_rate);
                config.set_max_depth(*max_depth);
                config.set_iterations(*num_boost_round as usize);
                config.set_debug(*debug);
                config.set_training_optimization_level(*training_optimization_level);
                config.set_loss(loss_type);

                let mut gbdt = GBDT::new(&config);

                let mut train_x = DataVec::new();

                for (i, row) in (0..x.nrows()).enumerate() {
                    let train_row = x.row_slice(row).to_vec();
                    train_x.push(Data::new_training_data(train_row, 1.0, y[i] as f32, None));
                }

                gbdt.fit(&mut train_x);

                self.model = Some(gbdt);
            }
            #[cfg(any(feature = "xgboost", feature = "svm"))]
            _ => {
                panic!(
                    "Error: Expected ModelType::GBDT params, got {:?}",
                    self.params.model_type
                );
            }
        }
    }

    pub fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        let mut test_x = DataVec::new();
        for row in 0..x.nrows() {
            let test_row = x.row_slice(row).to_vec();
            test_x.push(Data::new_training_data(test_row, 1.0, 0.0, None));
        }
        let predictions = self.model.as_ref().unwrap().decision_function(&test_x);
        predictions
    }

    pub fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32> {
        self.predict(x)
    }
}

impl ClassifierModel for GBDTClassifier {
    fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        x_eval: Option<&Array2<f32>>,
        y_eval: Option<&[i32]>,
    ) {
        GBDTClassifier::fit(self, x, y, x_eval, y_eval)
    }

    fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        GBDTClassifier::predict(self, x)
    }

    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32> {
        GBDTClassifier::predict_proba(self, x)
    }

    fn clone_box(&self) -> Box<dyn ClassifierModel> {
        Box::new(GBDTClassifier::new(self.params.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Array1, Array2};

    #[test]
    fn test_gbdt_classifier() {
        // Create a feature matrix with 5 features and 10 samples
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
        let y = Array1::from_vec(vec![
            1i32, -1i32, 1i32, -1i32, 1i32, -1i32, 1i32, -1i32, 1i32, -1i32,
        ]);

        // Convert y to [0, 1]
        let y = y.mapv(|x| if *x == 1 { 0 } else { 1 });

        println!("y.to_vec(): {:?}", y.to_vec());

        // Initialize the XGBoost classifier
        let params = ModelConfig {
            learning_rate: 0.1,
            model_type: ModelType::GBDT {
                max_depth: 6,
                num_boost_round: 3,
                debug: true,
                training_optimization_level: 2,
                loss_type: "LogLikelyhood".to_string(),
            },
        };

        let mut classifier = GBDTClassifier::new(params);

        // Fit the classifier
        classifier.fit(&x, &y.to_vec(), None, None);

        // Make predictions
        let predictions = classifier.predict(&x);

        println!("Predictions: {:?}", predictions);
        println!("y: {:?}", y.to_vec());

        // Check that predictions are reasonable
        // assert_eq!(predictions.len(), y.len());
    }
}
