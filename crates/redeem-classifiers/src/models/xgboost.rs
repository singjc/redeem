//! XGBoost classifier adapter (feature-gated).
//!
//! Provides a thin wrapper around the `xgb` crate. The implementation
//! performs an explicit training loop to ensure per-iteration updates are
//! executed (workaround for some upstream crate versions that omit updates).
use log::debug;
use xgb::{
    parameters::{
        learning::{LearningTaskParametersBuilder, Objective},
        tree::{TreeBoosterParametersBuilder, TreeMethod},
        BoosterParametersBuilder, BoosterType,
    },
    Booster, DMatrix,
};

use crate::config::{ModelConfig, ModelType};
use crate::math::Array2;
use crate::models::classifier_trait::ClassifierModel;

fn eval_auc(preds: &[f32], dtrain: &DMatrix) -> f32 {
    let labels = dtrain.get_labels().unwrap();

    // Calculate AUC using the trapezoidal rule
    let mut auc = 0.0;

    // Sort predictions and labels by prediction score (ascending)
    let mut combined: Vec<(f32, f32)> = preds.iter().copied().zip(labels.iter().copied()).collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Count positive and negative examples
    let total_pos = labels.iter().filter(|&&x| x == 1.0).count() as f32;
    let total_neg = labels.len() as f32 - total_pos;

    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5; // Return 0.5 (random) if all labels are the same
    }

    let mut cum_pos = 0.0;
    let mut cum_neg = 0.0;
    let mut prev_pred = f32::NEG_INFINITY;
    let mut prev_pos = 0.0;
    let mut prev_neg = 0.0;

    for (pred, label) in combined.iter() {
        if *pred != prev_pred {
            auc += (cum_pos - prev_pos) * (cum_neg + prev_neg) / 2.0;
            prev_pred = *pred;
            prev_pos = cum_pos;
            prev_neg = cum_neg;
        }

        if *label == 1.0 {
            cum_pos += 1.0;
        } else {
            cum_neg += 1.0;
        }
    }

    // Add the last rectangle
    auc += (total_pos - prev_pos) * (total_neg + prev_neg) / 2.0;

    // Normalize AUC to [0, 1]
    auc / (total_pos * total_neg)
}

pub struct XGBoostClassifier {
    booster: Option<Booster>,
    params: ModelConfig,
}

// Booster handles are local to a single thread in our usage (each fold trains
// its own model), so we can safely mark the wrapper as Send.
unsafe impl Send for XGBoostClassifier {}

impl XGBoostClassifier {
    pub fn new(params: ModelConfig) -> Self {
        XGBoostClassifier {
            booster: None,
            params,
        }
    }
}

impl XGBoostClassifier {
    pub fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        x_eval: Option<&Array2<f32>>,
        y_eval: Option<&[i32]>,
    ) {
        // Convert y to [0, 1] for XGBoost binary regression
        // Note: we set targets (original 1) as 1 and decoys (original -1) as 0, so that the scores are positive for targets and negative for decoys
        // TODO: this maybe should be done outside of the model
        let y = y
            .iter()
            .map(|&l| if l == 1 { 1 } else { 0 })
            .collect::<Vec<i32>>();
        let y_eval = y_eval.map(|vals| {
            vals.iter()
                .map(|&l| if l == 1 { 1 } else { 0 })
                .collect::<Vec<i32>>()
        });

        // Track class balance to set scale_pos_weight (helps when classes are imbalanced).
        let pos_count = y.iter().filter(|&&v| v == 1).count() as f32;
        let neg_count = y.len() as f32 - pos_count;
        let scale_pos_weight = if pos_count > 0.0 {
            (neg_count / pos_count).max(0.0001)
        } else {
            1.0
        };

        // Convert feature matrix into DMatrix
        // Log matrix shape info to help diagnose potential shape mismatches
        debug!(
            "Creating DMatrix from dense data: rows={}, cols={}, len={}",
            x.nrows(),
            x.ncols(),
            x.as_slice().len()
        );
        // `from_dense` expects the number of rows as the second argument.
        let mut dmat = DMatrix::from_dense(x.as_slice(), x.nrows()).unwrap();
        dmat.set_labels(&y.iter().map(|&l| l as f32).collect::<Vec<f32>>())
            .unwrap();

        // println!("TRAIN dmat: {:?}", dmat);

        let mut eval_matrix = None;
        if let (Some(x_e), Some(y_e)) = (x_eval, y_eval.as_ref()) {
            debug!(
                "Creating eval DMatrix: rows={}, cols={}, len={}",
                x_e.nrows(),
                x_e.ncols(),
                x_e.as_slice().len()
            );
            let mut matrix = DMatrix::from_dense(x_e.as_slice(), x_e.nrows()).unwrap();
            matrix
                .set_labels(&y_e.iter().map(|&l| l as f32).collect::<Vec<f32>>())
                .unwrap();
            eval_matrix = Some(matrix);
        }

        if let ModelType::XGBoost {
            max_depth,
            num_boost_round,
            early_stopping_rounds: _early_stopping_rounds,
            verbose_eval,
        } = &self.params.model_type
        {
            // Configure learning objective
            let learning_params = LearningTaskParametersBuilder::default()
                // Train for raw margins so downstream scoring can use
                // decision values instead of squashed probabilities.
                .objective(Objective::BinaryLogisticRaw)
                // .eval_metrics(Metrics::Custom(vec![EvaluationMetric::LogLoss, EvaluationMetric::MAE]))
                // .num_feature(x.ncols())
                .build()
                .unwrap();

            // Configure the tree-based learning model's parameters
            let tree_params = TreeBoosterParametersBuilder::default()
                .tree_method(TreeMethod::Hist)
                .max_depth(*max_depth)
                .eta(self.params.learning_rate)
                .build()
                .unwrap();

            // Overall configuration for Booster
            let booster_params = BoosterParametersBuilder::default()
                .booster_type(BoosterType::Tree(tree_params))
                .learning_params(learning_params)
                .verbose(*verbose_eval)
                .build()
                .unwrap();

            // Create Training Parameters with evaluation sets if needed.
            // NOTE: early stopping and custom evaluation hooks were part of
            // the forked xgboost crate used previously. The upstream `xgb`
            // crate may offer different APIs for early stopping. To keep
            // compatibility with the crates.io `xgb` crate we comment out
            // those evaluation-specific calls here. We intentionally avoid
            // calling the crate's convenience `train` API because some
            // published versions omit the per-iteration `update` call. We
            // will instead construct the Booster directly and perform the
            // update loop below.

            // The code below shows the original approach using the
            // TrainingParametersBuilder and `Booster::train`. It's kept
            // commented so it is easy to re-enable once the upstream
            // `xgb` crate includes the fix for per-iteration updates.
            /*
            let training_params = TrainingParametersBuilder::default()
                .dtrain(&dmat)
                .boost_rounds(*num_boost_round)
                // .early_stopping_rounds(Some(*early_stopping_rounds))
                .booster_params(booster_params)
                // .evaluation_sets(dmat_eval.as_deref())
                // .evaluation_score_direction(Some("high"))
                // .custom_evaluation_fn(Some(eval_auc))
                .build()
                .unwrap();

            // self.booster = Some(Booster::train(&training_params).unwrap());
            */

            // The upstream `xgb` crate's `Booster::train` has been observed to
            // omit the per-iteration `update` call in some published
            // releases. That results in a Booster that only contains the
            // `base_score` and no trained trees (predictions = base_score).
            // To work around this, construct the Booster and perform the
            // update loop manually so training definitely occurs.
            let mut cached_dmats: Vec<&DMatrix> = vec![&dmat];
            if let Some(ref m) = eval_matrix {
                cached_dmats.push(m);
            }

            // Create booster with cached dmats
            let mut bst = Booster::new_with_cached_dmats(&booster_params, &cached_dmats)
                .expect("failed to create Booster");

            // Class weighting to counter imbalance (similar to scale_pos_weight in xgboost CLI).
            let _ = bst.set_param("scale_pos_weight", &format!("{}", scale_pos_weight));

            // Perform explicit training updates for num_boost_round iterations
            for i in 0..*num_boost_round as i32 {
                bst.update(&dmat, i).expect("Booster.update failed");

                // Optionally evaluate on eval sets and print progress when verbose
                if *verbose_eval {
                    if let Some(ref evals) = eval_matrix {
                        // perform a simple evaluation by predicting on the eval dmatrix
                        if let Ok(preds_eval) = bst.predict(evals) {
                            // compute AUC using the helper defined above
                            let auc = eval_auc(&preds_eval, evals);
                            debug!("[{}]\t eval_auc:{}", i, auc);
                        }
                    }
                }
            }

            // store booster and print a small model dump to verify trees exist
            // (if the model only contains base_score, the dump will be tiny)
            let bst_storage = bst;
            if let Ok(buf) = bst_storage.save_buffer(false) {
                debug!("model dump size after training = {} bytes", buf.len());
                // print first 200 bytes as UTF-8 lossily for quick inspection
                if buf.len() > 0 {
                    let snippet = String::from_utf8_lossy(&buf[..buf.len().min(200)]);
                    debug!("model dump snippet: {}", snippet);
                }
            } else {
                debug!("failed to save model buffer after training");
            }
            self.booster = Some(bst_storage);
        } else {
            eprintln!("Error: Expected ModelType::XGBoost but got another type.");
        }
    }

    pub fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        debug!(
            "Creating final DMatrix for prediction: rows={}, cols={}, len={}",
            x.nrows(),
            x.ncols(),
            x.as_slice().len()
        );
        let dmat = DMatrix::from_dense(x.as_slice(), x.nrows()).unwrap();
        // println!("PREDICT: dmat: {:?}", dmat);
        self.booster.as_ref().unwrap().predict(&dmat).unwrap()
    }

    pub fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32> {
        // Ensure booster has a record of the number of features. Some
        // XGBoost builds require `num_feature` to be set on the booster
        // before predicting. Setting it here avoids a runtime failure with
        // the upstream `xgb` crate when the internal learner hasn't been
        // fully configured.
        if let Some(bst) = self.booster.as_mut() {
            let _ = bst.set_param("num_feature", &x.ncols().to_string());
        }

        // Create DMatrix and predict
        debug!(
            "Creating final DMatrix for prediction: rows={}, cols={}, len={}",
            x.nrows(),
            x.ncols(),
            x.as_slice().len()
        );
        let dmat = DMatrix::from_dense(x.as_slice(), x.nrows()).unwrap();
        // Use predict_matrix with an explicit config to ensure the full
        // boosted ensemble is used for prediction. Some builds interpret
        // the older `XGBoosterPredict` ntree_limit=0 as "use zero trees"
        // which returns the base_score; using predict_matrix with a large
        // iteration_end avoids that ambiguity.
        // Determine the number of trees present in the trained model by
        // saving the model to a JSON buffer and parsing the `num_trees`
        // field. Then use that value as iteration_end so the full
        // ensemble is used for prediction.
        let bst_ref = self.booster.as_ref().unwrap();
        let buf = bst_ref.save_buffer(false).unwrap_or_else(|_| vec![]);
        let mut iteration_end = 0i64;
        if !buf.is_empty() {
            let s = String::from_utf8_lossy(&buf);
            if let Some(idx) = s.find("\"num_trees\":\"") {
                let start = idx + "\"num_trees\":\"".len();
                if let Some(rest) = s.get(start..) {
                    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(n) = digits.parse::<i64>() {
                        iteration_end = n;
                    }
                }
            }
        }

        // fallback: if parsing failed, use 0 (old behavior) — but prefer
        // the parsed tree count when available.
        let config = format!(
            "{{\"type\":0,\"training\":false,\"iteration_begin\":0,\"iteration_end\":{},\"strict_shape\":false}}\0",
            iteration_end
        );
        if let Ok(margins) = bst_ref.predict_margin(&dmat) {
            if !margins.is_empty() {
                let sample_end = margins.len().min(10);
                debug!(
                    "[xgb.predict_margin] len = {}, first {} = {:?}",
                    margins.len(),
                    sample_end,
                    &margins[..sample_end]
                );
            }
            return margins;
        }

        debug!("predict_margin failed; falling back to predict_matrix");
        let (preds, _shape) = bst_ref.predict_matrix(&dmat, &config).unwrap();
        preds
    }
}

// Implement the newer classifier trait so XGBoost can be used via the
// `models::factory::build_model` factory and as a boxed `ClassifierModel`.
impl ClassifierModel for XGBoostClassifier {
    fn fit(
        &mut self,
        x: &crate::math::Array2<f32>,
        y: &[i32],
        x_eval: Option<&crate::math::Array2<f32>>,
        y_eval: Option<&[i32]>,
    ) {
        // delegate to the inherent impl
        XGBoostClassifier::fit(self, x, y, x_eval, y_eval)
    }

    fn predict(&self, x: &crate::math::Array2<f32>) -> Vec<f32> {
        XGBoostClassifier::predict(self, x)
    }

    fn predict_proba(&mut self, x: &crate::math::Array2<f32>) -> Vec<f32> {
        XGBoostClassifier::predict_proba(self, x)
    }

    fn name(&self) -> &str {
        "xgboost"
    }

    fn clone_box(&self) -> Box<dyn ClassifierModel> {
        Box::new(XGBoostClassifier::new(self.params.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Array1, Array2};

    #[test]
    fn test_xgboost_classifier() {
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
            learning_rate: 0.3,
            model_type: ModelType::XGBoost {
                max_depth: 6,
                num_boost_round: 100,
                early_stopping_rounds: 10,
                verbose_eval: true,
            },
        };
        let mut classifier = XGBoostClassifier::new(params);

        // Fit the classifier
        classifier.fit(&x, &y.to_vec(), Some(&x), Some(&y.to_vec()));

        // Make predictions
        let predictions = classifier.predict(&x);

        println!("Predictions: {:?}", predictions);
        // println!("y: {:?}", y.to_vec());

        // Check that predictions are reasonable
        // assert_eq!(predictions.len(), y.len());

        // // You can add more specific assertions depending on what you expect from your model
        // for (_i, &pred) in predictions.iter().enumerate() {
        //     assert!(pred >= -1f32 && pred <= 1f32); // Example assertion for binary-like output
        // }

        // // Get attribute names
        // let attr_names = classifier.booster.unwrap().get_attribute("weight").unwrap().unwrap();
        // println!("Attribute names: {:?}", attr_names);

        // Predict Contributions
        debug!(
            "Creating DMatrix for contributions: rows={}, cols={}, len={}",
            x.nrows(),
            x.ncols(),
            x.as_slice().len()
        );
        let mut dmat = DMatrix::from_dense(x.as_slice(), x.nrows()).unwrap();
        dmat.set_labels(&y.iter().map(|&l| l as f32).collect::<Vec<f32>>())
            .unwrap();

        let _contributions = classifier
            .booster
            .as_ref()
            .unwrap()
            .predict_contributions(&dmat)
            .unwrap();

        let _interactions = classifier
            .booster
            .as_ref()
            .unwrap()
            .predict_interactions(&dmat)
            .unwrap();
    }
}
