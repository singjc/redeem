use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::data_handling::Experiment;
use crate::models::xgboost::XGBoostClassifier;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelParams {
    pub learning_rate: f32,
    // XGBoost-specific parameters
    /// Maximum depth of a tree. 
    pub max_depth: u32,
    /// Number of boosting rounds. 
    pub num_boost_round: u32,
}

impl ModelParams {
    pub fn new(learning_rate: f32, max_depth: u32, num_boost_round: u32) -> Self {
        Self {
            learning_rate,
            max_depth,
            num_boost_round,
        }
    }
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_depth: 6,
            num_boost_round: 3,
        }
    }
}


pub enum ModelType {
    XGBoost,
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "xgboost" => Ok(ModelType::XGBoost),
            // Add other model types as needed
            _ => Err(format!("Unknown model type: {}", s)),
        }
    }
}


pub trait SemiSupervisedModel {
    fn fit(&mut self, x: &Array2<f32>, y: &[i32], x_eval: Option<&Array2<f32>>, y_eval: Option<&[i32]>);
    fn predict(&self, x: &Array2<f32>) -> Vec<f32>;
    fn predict_proba(&self, x: &Array2<f32>) -> Vec<f32>;
}


pub struct SemiSupervisedLearner {
    model: Box<dyn SemiSupervisedModel>,
    train_fdr: f32,
    xeval_num_iter: usize,
}

impl SemiSupervisedLearner {
    /// Create a new SemiSupervisedLearner
    /// 
    /// # Arguments
    /// 
    /// * `model_type` - The type of model to use
    /// * `train_fdr` - The FDR threshold to use for training
    /// * `xeval_num_iter` - The number of iterations to use for cross-validation
    /// 
    /// # Returns
    /// 
    /// A new SemiSupervisedLearner
    pub fn new(model_type: ModelType, model_params: Option<ModelParams>, train_fdr: f32, xeval_num_iter: usize) -> Self {

        let model: Box<dyn SemiSupervisedModel> = match model_type {
            ModelType::XGBoost => Box::new(XGBoostClassifier::new(model_params.unwrap_or_default())),
        };

        SemiSupervisedLearner {
            model,
            train_fdr,
            xeval_num_iter,
        }
    }

    /// Initialize the best feature
    /// 
    /// Adapted from MS2Rescore
    /// 
    /// # Arguments
    /// 
    /// * `experiment` - The experiment to use
    /// * `eval_fdr` - The FDR threshold to use for evaluation
    pub fn init_best_feature(&mut self, experiment: &Experiment, eval_fdr: f32) -> (usize, usize, Array1<i32>, bool, Array1<f32>) {
        // Helper function to count targets by feature
        let targets_count_by_feature = |desc: bool| -> Vec<usize> {
            (0..experiment.x.ncols())
                .map(|col| {
                    let scores = experiment.x.column(col).to_owned();
                    let labels = experiment.update_labels(&scores, eval_fdr, desc);
                    labels.iter().filter(|&&x| x == 1).count()
                })
                .collect()
        };

        // Find the best feature
        let mut best_feat = 0;
        let mut best_positives = 0;
        let mut new_labels = Array1::zeros(experiment.x.nrows());
        let mut best_desc = false;

        for desc in &[true, false] {
            let num_passing = targets_count_by_feature(*desc);
            let feat_idx = num_passing.iter()
                .enumerate()
                .max_by_key(|&(_, count)| count)
                .map(|(idx, _)| idx)
                .unwrap();
            let num_passing = num_passing[feat_idx];

            if num_passing > best_positives {
                best_positives = num_passing;
                best_feat = feat_idx;
                let scores = experiment.x.column(feat_idx).to_owned();
                new_labels = experiment.update_labels(&scores, eval_fdr, *desc);
                best_desc = *desc;
            }
        }

        if best_positives == 0 {
            panic!("No PSMs found below the 'eval_fdr' {}", eval_fdr);
        }

        let best_feature_scores = experiment.x.column(best_feat).to_owned();

        (best_feat, best_positives, new_labels, best_desc, best_feature_scores)
    }

    /// Remove unlabeled PSMs
    /// 
    /// This function removes PSMs with a label of 0 from the experiment. These PSMs are not used for training.
    /// 
    /// # Arguments
    /// 
    /// * `experiment` - The experiment to use
    /// 
    /// # Returns
    /// 
    /// The experiment with the unlabeled PSMs removed
    fn remove_unlabeled_psms(&self, experiment: &mut Experiment) {
        let indices_to_remove: Vec<usize> = experiment.y
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == 0 { Some(i) } else { None })
            .collect();

        experiment.remove_psms(&indices_to_remove);
    }

    /// Create folds for cross-validation
    /// 
    /// # Arguments
    /// 
    /// * `experiment` - The experiment to use
    /// * `n_folds` - The number of folds to create
    /// 
    /// # Returns
    /// 
    /// A vector of tuples containing the training and testing experiments for each fold
    fn create_folds(&self, experiment: &Experiment, n_folds: usize) -> Vec<(Experiment, Experiment)> {
        let n_samples = experiment.x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());

        let fold_size = n_samples / n_folds;
        
        (0..n_folds).map(|i| {
            let test_indices: Vec<usize> = indices[i*fold_size..(i+1)*fold_size].to_vec();
            let mut train_mask = Array1::from_elem(n_samples, true);
            for &idx in &test_indices {
                train_mask[idx] = false;
            }
            let test_mask = train_mask.mapv(|x| !x);

            let train_exp = experiment.filter(&train_mask);
            let test_exp = experiment.filter(&test_mask);

            (train_exp, test_exp)
        }).collect()
    }

    /// Fit the SemiSupervisedLearner
    /// 
    /// # Arguments
    /// 
    /// * `x` - The features to use, shape (n_samples, n_features)
    /// * `y` - The labels to use, shape (n_samples,)
    /// 
    /// # Returns
    /// 
    /// The predictions for the input features
    pub fn fit(&mut self, x: Array2<f32>, y: Array1<i32>) -> Array1<f32> {
        let mut experiment = Experiment::new(x.clone(), y.clone());
        experiment.log_input_data_summary();

        // Get initial best feature
        let (best_feat, best_positives, mut new_labels, best_desc, best_feature_scores) = self.init_best_feature(&experiment, self.train_fdr);
        experiment.y = new_labels.clone();

        let folds = self.create_folds(&experiment, self.xeval_num_iter);
        
        for (fold, (mut train_exp, test_exp)) in folds.into_iter().enumerate() {
            println!("Learning on Cross-Validation Fold: {}", fold);
            let n_samples = experiment.x.nrows();
            let mut all_predictions = Array1::zeros(n_samples);

            self.remove_unlabeled_psms(&mut train_exp);

            self.model.fit(&train_exp.x, &train_exp.y.to_vec(), None, None);
            let fold_predictions = Array1::from(self.model.predict_proba(&test_exp.x));

            // Update predictions
            for (i, pred) in fold_predictions.iter().enumerate() {
                all_predictions[test_exp.tg_num_id[i] as usize] = *pred;
            }

            let new_labels = experiment.update_labels(&all_predictions, self.train_fdr, best_desc);
            experiment.y = new_labels;
        }

        // Final prediction on the entire dataset
        println!("Final prediction on the entire dataset");
        let experiment = Experiment::new(x, y);

        self.model.fit(&experiment.x, &experiment.y.to_vec(), None, None);
        Array1::from(self.model.predict_proba(&experiment.x))
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fs::File;
    use std::io::Write;
    use csv::ReaderBuilder;
    use ndarray::{Array2, Array1};


    fn read_features_tsv(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b',')
            .from_path(path)?;
    
        let mut data = Vec::new();
    
        for result in reader.records() {
            let record = result?;
            let row: Vec<f32> = record.iter()
                .map(|field| field.parse::<f32>())
                .collect::<Result<_, _>>()?;
            data.push(row);
        }
    
        let n_samples = data.len();
        let n_features = data[0].len();
    
        Array2::from_shape_vec((n_samples, n_features), data.into_iter().flatten().collect())
            .map_err(|e| e.into())
    }
    
    fn read_labels_tsv(path: &str) -> Result<Array1<i32>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'\t')
            .from_path(path)?;
    
        let labels: Vec<i32> = reader.records()
            .map(|r| {
                let record = r?;
                let value = record.get(0).ok_or_else(|| "Empty row".to_string())?;
                value.parse::<i32>().map_err(|e| e.into())
            })
            .collect::<Result<_, Box<dyn Error>>>()?;
    
        Ok(Array1::from_vec(labels))
    }

    fn save_predictions_to_csv(predictions: &Array1<f32>, file_path: &str) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(file_path)?;
        
        for &pred in predictions.iter() {
            writeln!(file, "{}", pred)?;
        }
    
        Ok(())
    }

    #[test]
    fn test_semi_supervised_learner() {
       // Load the test data from the TSV files
        let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
        let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();


        println!("Loaded features shape: {:?}", x.shape());
        println!("Loaded labels shape: {:?}", y.shape());

        // Create and train your SemiSupervisedLearner
        let mut learner = SemiSupervisedLearner::new(ModelType::XGBoost, Some(ModelParams::new(0.01, 10, 100)), 1.0, 2);
        let predictions = learner.fit(x, y.clone());

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }
}