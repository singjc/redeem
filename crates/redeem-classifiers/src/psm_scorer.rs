//! Semi-supervised learner and orchestration utilities.
//!
//! This module implements `SemiSupervisedLearner`, the core training loop
//! used by the examples to iteratively refine labels and fit models. It
//! delegates to implementations of the `models::ClassifierModel` trait and
//! centralizes preprocessing flags (scaling / score normalization) so the
//! examples can pass flags through the CLI.
use std::collections::HashMap;

use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use crate::data_handling::{Experiment, PsmMetadata, RankGrouping};
use crate::math::{Array1, Array2};
use crate::preprocessing;
use crate::stats::tdc;

use crate::config::{ModelConfig, ModelType};
use crate::models::classifier_trait::ClassifierModel;
use crate::models::factory;

// Legacy `SemiSupervisedModel` behavior is provided by implementations of
// `models::classifier_trait::ClassifierModel`. The learner stores a boxed
// `ClassifierModel` trait object.

pub struct SemiSupervisedLearner {
    model: Box<dyn ClassifierModel>,
    train_fdr: f32,
    xeval_num_iter: usize,
    max_iterations: usize,
    class_pct: Option<(f64, f64)>,
    /// If true, fit a standard scaler on the input features and use the
    /// scaled features for training and evaluation.
    scale_features: bool,
    /// If true, normalize final prediction scores to zero-mean/unit-variance
    /// before producing the final output.
    normalize_scores: bool,
    last_feature_weights: Option<Vec<f32>>,
    rank_grouping: RankGrouping,
}

impl SemiSupervisedLearner {
    /// Create a new SemiSupervisedLearner
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of model to use
    /// * `train_fdr` - The FDR threshold to use for training
    /// * `xeval_num_iter` - The number of cross-validation folds
    /// * `max_iterations` - Maximum semi-supervised iterations (label refinements)
    /// * `class_pct` - (f64, f64) The percentage of targets and decoys to use for training
    ///
    /// # Returns
    ///
    /// A new SemiSupervisedLearner
    pub fn new(
        model_type: ModelType,
        learning_rate: f32,
        train_fdr: f32,
        xeval_num_iter: usize,
        max_iterations: usize,
        class_pct: Option<(f64, f64)>,
        scale_features: bool,
        normalize_scores: bool,
        rank_grouping: RankGrouping,
    ) -> Self {
        // Centralize construction to the models factory which returns a
        // `Box<dyn ClassifierModel>`.
        let cfg = ModelConfig {
            learning_rate,
            model_type,
        };
        let model: Box<dyn ClassifierModel> = factory::build_model(cfg);

        SemiSupervisedLearner {
            model,
            train_fdr,
            xeval_num_iter,
            max_iterations,
            class_pct,
            scale_features,
            normalize_scores,
            last_feature_weights: None,
            rank_grouping,
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
    pub fn init_best_feature(
        &mut self,
        experiment: &Experiment,
        eval_fdr: f32,
    ) -> (usize, usize, Array1<i32>, bool, Array1<f32>) {
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
            let feat_idx = num_passing
                .iter()
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

        log::trace!(
            "Best feature: {} with {} positives",
            best_feat,
            best_positives
        );

        if best_positives == 0 {
            panic!("No PSMs found below the 'eval_fdr' {}", eval_fdr);
        }

        let best_feature_scores = experiment.x.column(best_feat).to_owned();

        (
            best_feat,
            best_positives,
            new_labels,
            best_desc,
            best_feature_scores,
        )
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
    fn remove_unlabeled_psms(experiment: &mut Experiment) {
        let indices_to_remove: Vec<usize> = experiment
            .y
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == 0 { Some(i) } else { None })
            .collect();
        log::trace!("Removing {} unlabeled PSMs", indices_to_remove.len());
        experiment.remove_psms(&indices_to_remove);
    }

    fn mask_from_indices(len: usize, indices: &[usize]) -> Array1<bool> {
        let mut mask = Array1::from_elem(len, false);
        for &idx in indices {
            if idx < len {
                mask[idx] = true;
            }
        }
        mask
    }

    fn build_spectrum_folds(
        &self,
        experiment: &Experiment,
        requested_folds: usize,
    ) -> Vec<Vec<usize>> {
        let n_samples = experiment.x.nrows();
        if n_samples == 0 {
            return vec![Vec::new()];
        }
        let n_folds = requested_folds.max(2).min(n_samples);
        let mut spectra: HashMap<(usize, String), Vec<usize>> = HashMap::new();
        for (idx, (file_id, spec_id)) in experiment
            .psm_metadata
            .file_id
            .iter()
            .zip(experiment.psm_metadata.spec_id.iter())
            .enumerate()
        {
            spectra
                .entry((*file_id, spec_id.clone()))
                .or_default()
                .push(idx);
        }

        let mut grouped: Vec<Vec<usize>> = spectra.into_values().collect();
        grouped.shuffle(&mut thread_rng());

        let mut folds = vec![Vec::new(); n_folds];
        for (idx, group) in grouped.into_iter().enumerate() {
            folds[idx % n_folds].extend(group);
        }
        folds
    }

    fn sample_training_indices(
        labels: &Array1<i32>,
        candidates: &[usize],
        class_pct: Option<(f64, f64)>,
    ) -> Vec<usize> {
        let Some((target_pct, decoy_pct)) = class_pct else {
            return candidates.to_vec();
        };

        let mut target_indices = Vec::new();
        let mut decoy_indices = Vec::new();
        for &idx in candidates {
            match labels[idx] {
                1 => target_indices.push(idx),
                -1 => decoy_indices.push(idx),
                _ => {}
            }
        }

        let mut rng = thread_rng();
        let mut sampled_targets = Self::sample_class_subset(&target_indices, target_pct, &mut rng);
        let mut sampled_decoys = Self::sample_class_subset(&decoy_indices, decoy_pct, &mut rng);

        if sampled_targets.is_empty() || sampled_decoys.is_empty() {
            log::warn!(
                "Class sampling removed all targets or decoys; using all available training examples."
            );
            return candidates.to_vec();
        }

        sampled_targets.append(&mut sampled_decoys);
        sampled_targets
    }

    fn sample_class_subset(indices: &[usize], pct: f64, rng: &mut ThreadRng) -> Vec<usize> {
        if indices.is_empty() || pct >= 0.999 {
            return indices.to_vec();
        }

        let mut take = ((indices.len() as f64) * pct).round() as usize;
        take = take.max(1).min(indices.len());

        let mut shuffled = indices.to_vec();
        shuffled.shuffle(rng);
        shuffled.truncate(take);
        shuffled
    }

    fn score_with_cross_validation(
        &mut self,
        experiment: &Experiment,
        folds: &[Vec<usize>],
        labels: &Array1<i32>,
        fallback_scores: &Array1<f32>,
    ) -> Array1<f32> {
        let n_samples = experiment.x.nrows();
        let mut assigned: Vec<Option<f32>> = vec![None; n_samples];
        let class_pct = self.class_pct;

        let model_clones: Vec<Box<dyn ClassifierModel>> =
            (0..folds.len()).map(|_| self.model.clone_box()).collect();

        let fold_results: Vec<Vec<(usize, f32)>> = folds
            .par_iter()
            .enumerate()
            .zip(model_clones.into_par_iter())
            .map(|((fold_idx, test_indices_unsorted), mut fold_model)| {
                let mut local_assignments = Vec::new();
                if test_indices_unsorted.is_empty() {
                    return local_assignments;
                }

                let mut test_indices = test_indices_unsorted.clone();
                test_indices.sort_unstable();

                let test_mask = Self::mask_from_indices(n_samples, &test_indices);
                let train_indices: Vec<usize> =
                    (0..n_samples).filter(|idx| !test_mask[*idx]).collect();
                if train_indices.is_empty() {
                    log::warn!("Fold {} has no training samples; skipping.", fold_idx);
                    return local_assignments;
                }

                let mut selected_train_indices =
                    Self::sample_training_indices(labels, &train_indices, class_pct);
                if selected_train_indices.is_empty() {
                    log::warn!(
                        "Fold {} has no training samples after sampling; skipping.",
                        fold_idx
                    );
                    return local_assignments;
                }

                selected_train_indices.sort_unstable();

                let train_mask = Self::mask_from_indices(n_samples, &selected_train_indices);
                let mut train_exp = experiment.filter(&train_mask);
                train_exp.y = Array1::from_vec(
                    selected_train_indices
                        .iter()
                        .map(|&idx| labels[idx])
                        .collect(),
                );
                Self::remove_unlabeled_psms(&mut train_exp);

                let has_pos = train_exp.y.iter().any(|&v| v == 1);
                let has_neg = train_exp.y.iter().any(|&v| v == -1);
                if !(has_pos && has_neg) {
                    log::warn!(
                        "Fold {} lacks positive or negative examples after filtering; skipping.",
                        fold_idx
                    );
                    return local_assignments;
                }

                fold_model.fit(&train_exp.x, train_exp.y.as_slice(), None, None);

                let test_exp = experiment.filter(&test_mask);
                let fold_scores = fold_model.predict_proba(&test_exp.x);
                for (local_idx, &global_idx) in test_indices.iter().enumerate() {
                    local_assignments.push((global_idx, fold_scores[local_idx]));
                }

                local_assignments
            })
            .collect();

        for fold_assignment in fold_results {
            for (idx, score) in fold_assignment {
                assigned[idx] = Some(score);
            }
        }

        let mut filled = Vec::with_capacity(n_samples);
        let mut fallback_count = 0usize;
        for (idx, maybe_score) in assigned.into_iter().enumerate() {
            if let Some(score) = maybe_score {
                filled.push(score);
            } else {
                fallback_count += 1;
                filled.push(fallback_scores[idx]);
            }
        }
        if fallback_count > 0 {
            log::warn!(
                "Cross-validation scoring fell back to previous scores for {} of {} PSMs ({:.2}%). Consider adjusting class sampling or folds to cover all PSMs.",
                fallback_count,
                n_samples,
                (fallback_count as f64 / n_samples as f64) * 100.0
            );
        }
        Array1::from_vec(filled)
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
    pub fn fit(
        &mut self,
        x: Array2<f32>,
        y: Array1<i32>,
        psm_metadata: PsmMetadata,
    ) -> anyhow::Result<(Array1<f32>, Array1<u32>)> {
        // Optionally scale features before building the Experiment. We fit the
        // scaler on the supplied feature matrix and transform the full matrix
        // so training/evaluation use standardized features.
        let x = if self.scale_features {
            log::info!("Fitting scaler and transforming input features");
            preprocessing::fit_transform(&x)
        } else {
            x
        };

        let mut experiment = Experiment::new(x, y, psm_metadata);

        experiment.log_input_data_summary();

        // Get initial best feature to seed labels/scores.
        let (_, _, mut labels, _best_desc, mut current_scores) =
            self.init_best_feature(&experiment, self.train_fdr);

        let target_mask = experiment.y.mapv(|&v| v == 1);
        let folds = self.build_spectrum_folds(&experiment, self.xeval_num_iter.max(2));

        if folds.len() < 2 {
            log::warn!(
                "Only {} fold(s) were created; cross-validation may be unstable.",
                folds.len()
            );
        }

        if self.max_iterations == 0 {
            log::info!(
                "max_iterations set to 0; returning initial feature scores without refinement."
            );
        }

        let mut iter = 0usize;
        while iter < self.max_iterations {
            log::info!(
                "Semi-supervised iteration {}/{}",
                iter + 1,
                self.max_iterations
            );

            let fold_scores =
                self.score_with_cross_validation(&experiment, &folds, &labels, &current_scores);

            let updated_labels =
                update_labels_from_scores(&fold_scores, &target_mask, self.train_fdr, true);

            let changed = updated_labels
                .iter()
                .zip(labels.iter())
                .any(|(a, b)| a != b);

            current_scores = fold_scores;
            labels = updated_labels;

            if !changed {
                log::info!("Converged after {} iteration(s).", iter + 1);
                break;
            }

            iter += 1;
        }

        if iter == self.max_iterations && self.max_iterations > 0 {
            log::info!(
                "Reached maximum iterations ({}); using the latest scores.",
                self.max_iterations
            );
        }

        let mut final_predictions = current_scores.clone();

        // Log basic statistics of final predictions before normalization so
        // we can see whether they are already constant (which would lead to
        // zeroed normalized scores).
        {
            let s = final_predictions.as_slice();
            if !s.is_empty() {
                let len = s.len() as f32;
                let mean = s.iter().copied().sum::<f32>() / len;
                let mut var = 0f32;
                for &v in s.iter() {
                    let d = v - mean;
                    var += d * d;
                }
                let std = (var / len).sqrt();
                log::debug!(
                    "Final predictions before normalization: len={}, mean={:.6}, std={:.6}, min={}, max={}",
                    s.len(),
                    mean,
                    std,
                    s.iter().cloned().fold(f32::INFINITY, f32::min),
                    s.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
            } else {
                log::debug!("Final predictions before normalization: empty");
            }
        }

        if self.normalize_scores {
            log::info!("Normalizing final prediction scores (zero-mean, unit-variance)");
            preprocessing::normalize_scores(final_predictions.as_mut_slice());

            // Log stats after normalization
            let s = final_predictions.as_slice();
            if !s.is_empty() {
                let len = s.len() as f32;
                let mean = s.iter().copied().sum::<f32>() / len;
                let mut var = 0f32;
                for &v in s.iter() {
                    let d = v - mean;
                    var += d * d;
                }
                let std = (var / len).sqrt();
                log::debug!(
                    "Final predictions after normalization: len={}, mean={:.6}, std={:.6}, min={}, max={}",
                    s.len(),
                    mean,
                    std,
                    s.iter().cloned().fold(f32::INFINITY, f32::min),
                    s.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
            }
        }

        self.last_feature_weights = self.fit_final_model_for_weights(&experiment, &labels);

        experiment.update_rank_feature(&final_predictions, &experiment.psm_metadata.clone());
        let updated_ranks = match experiment.get_rank_column() {
            Ok(ranks) => ranks,
            Err(err) => {
                log::warn!(
                    "Rank feature missing or invalid ({}); computing ranks from scores.",
                    err
                );
                experiment.compute_rank_from_scores(&final_predictions, self.rank_grouping)?
            }
        };

        Ok((final_predictions, updated_ranks))
    }

    pub fn feature_weights(&self) -> Option<&[f32]> {
        self.last_feature_weights.as_deref()
    }

    fn fit_final_model_for_weights(
        &mut self,
        experiment: &Experiment,
        labels: &Array1<i32>,
    ) -> Option<Vec<f32>> {
        let n_samples = experiment.x.nrows();
        if n_samples == 0 {
            return None;
        }

        let candidates: Vec<usize> = (0..n_samples).collect();
        let mut selected = Self::sample_training_indices(labels, &candidates, self.class_pct);
        if selected.is_empty() {
            return None;
        }
        selected.sort_unstable();

        let train_mask = Self::mask_from_indices(n_samples, &selected);
        let mut train_exp = experiment.filter(&train_mask);
        train_exp.y = Array1::from_vec(selected.iter().map(|&idx| labels[idx]).collect());
        Self::remove_unlabeled_psms(&mut train_exp);

        let has_pos = train_exp.y.iter().any(|&v| v == 1);
        let has_neg = train_exp.y.iter().any(|&v| v == -1);
        if !(has_pos && has_neg) {
            log::warn!(
                "Final model training skipped; requires both target and decoy examples."
            );
            return None;
        }

        self.model.fit(&train_exp.x, train_exp.y.as_slice(), None, None);
        self.model.feature_weights()
    }
}

fn update_labels_from_scores(
    scores: &Array1<f32>,
    targets: &Array1<bool>,
    eval_fdr: f32,
    desc: bool,
) -> Array1<i32> {
    let qvals = tdc(scores, targets, desc);
    let unlabeled = (&qvals.mapv(|&v| v > eval_fdr)) & targets;

    let mut new_labels = Array1::ones(qvals.len());
    for (i, &is_target) in targets.iter().enumerate() {
        if !is_target {
            new_labels[i] = -1;
        } else if unlabeled[i] {
            new_labels[i] = 0;
        }
    }
    new_labels
}

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use std::error::Error;
    use std::fs::File;
    use std::io::Write;

    use crate::math::{Array1, Array2};

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[test]
    #[cfg(feature = "xgboost")]
    #[ignore]
    fn test_xgb_semi_supervised_learner() {
        // Load the test data from the TSV files
        let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
        let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();

        println!("Loaded features shape: {:?}", x.shape());
        println!("Loaded labels shape: {:?}", y.shape());

        // Create and train your SemiSupervisedLearner
        let xgb_params = ModelType::XGBoost {
            max_depth: 8,
            num_boost_round: 100,
            early_stopping_rounds: 10,
            verbose_eval: false,
        };
        let mut learner = SemiSupervisedLearner::new(
            xgb_params,
            0.001,
            1.0,
            2,
            10,
            Some((0.2, 0.5)),
            false,
            false,
            RankGrouping::SpecId,
        );
        let metadata = crate::data_handling::PsmMetadata {
            file_id: vec![0usize; x.nrows()],
            spec_id: vec!["spec".to_string(); x.nrows()],
            feature_names: (0..x.ncols()).map(|i| format!("f{}", i)).collect(),
            scan_nr: None,
            exp_mass: None,
        };

        let (predictions, _ranks) = learner.fit(x, y.clone(), metadata).unwrap();

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }

    #[test]
    #[cfg(feature = "svm")]
    #[ignore]
    fn test_svm_semi_supervised_learner() {
        // Load the test data from the TSV files
        let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
        let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();

        println!("Loaded features shape: {:?}", x.shape());
        println!("Loaded labels shape: {:?}", y.shape());

        // Create and train your SemiSupervisedLearner
        let params = ModelType::SVM {
            eps: 0.1,
            c: (1.0, 1.0),
            kernel: "linear".to_string(),
            gaussian_kernel_eps: 0.1,
            polynomial_kernel_constant: 1.0,
            polynomial_kernel_degree: 3.0,
        };
        let mut learner = SemiSupervisedLearner::new(
            params,
            0.001,
            1.0,
            1000,
            10,
            Some((0.2, 0.5)),
            false,
            false,
            RankGrouping::SpecId,
        );
        let metadata = crate::data_handling::PsmMetadata {
            file_id: vec![0usize; x.nrows()],
            spec_id: vec!["spec".to_string(); x.nrows()],
            feature_names: (0..x.ncols()).map(|i| format!("f{}", i)).collect(),
            scan_nr: None,
            exp_mass: None,
        };

        let (predictions, _ranks) = learner.fit(x, y.clone(), metadata).unwrap();

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }
}
