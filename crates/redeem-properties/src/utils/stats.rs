/// Represents a single phase of training: either Training or Validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrainingPhase {
    Train,
    Validation,
}

/// Stores step-wise metrics for all training/validation iterations in a Struct of Arrays layout.
#[derive(Debug, Clone)]
pub struct TrainingStepMetrics {
    pub epochs: Vec<usize>,
    pub steps: Vec<usize>,
    pub learning_rates: Vec<f64>,
    pub losses: Vec<f32>,
    pub phases: Vec<TrainingPhase>,
    pub precisions: Vec<Option<f32>>,
    pub recalls: Vec<Option<f32>>,
    pub accuracies: Vec<Option<f32>>, // retained for MS2/CCS style metrics
    pub maes: Vec<Option<f32>>,
    pub rmses: Vec<Option<f32>>,
    pub r2s: Vec<Option<f32>>,
}

impl TrainingStepMetrics {
    /// Computes the average and standard deviation of loss values grouped by epoch and training phase.
    ///
    /// # Returns
    /// A `HashMap` where each key is a tuple `(epoch, TrainingPhase)` and each value is a tuple `(avg_loss, std_loss)`.
    /// This can be used for reporting or plotting epoch-level training and validation loss trends.
    pub fn summarize_by_epoch_phase(
        &self,
    ) -> std::collections::HashMap<(usize, TrainingPhase), (f32, f32)> {
        use std::collections::HashMap;

        let mut grouped: HashMap<(usize, TrainingPhase), Vec<f32>> = HashMap::new();

        for i in 0..self.epochs.len() {
            let key = (self.epochs[i], self.phases[i].clone());
            grouped.entry(key).or_default().push(self.losses[i]);
        }

        let mut summary = HashMap::new();
        for (key, values) in grouped {
            let avg = values.iter().copied().sum::<f32>() / values.len() as f32;
            let std = (values.iter().map(|v| (v - avg).powi(2)).sum::<f32>() / values.len() as f32)
                .sqrt();
            summary.insert(key, (avg, std)); // insert avg/std loss for this epoch + phase
        }

        summary
    }

    /// Summarizes average and std loss per epoch for training and validation phases.
    ///
    /// Returns a vector of tuples:
    /// (epoch, avg_train_loss, avg_val_loss, std_train_loss, std_val_loss)
    pub fn summarize_loss_for_plotting(&self) -> Vec<(usize, f32, Option<f32>, f32, Option<f32>)> {
        use std::collections::HashMap;

        let mut train_map: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut val_map: HashMap<usize, Vec<f32>> = HashMap::new();

        for i in 0..self.epochs.len() {
            match self.phases[i] {
                TrainingPhase::Train => train_map
                    .entry(self.epochs[i])
                    .or_default()
                    .push(self.losses[i]),
                TrainingPhase::Validation => val_map
                    .entry(self.epochs[i])
                    .or_default()
                    .push(self.losses[i]),
            }
        }

        let mut epochs: Vec<_> = train_map.keys().chain(val_map.keys()).copied().collect();
        epochs.sort_unstable();
        epochs.dedup();

        epochs
            .into_iter()
            .map(|epoch| {
                let (avg_train, std_train) = train_map
                    .get(&epoch)
                    .map(|v| compute_loss_stats(v))
                    .unwrap_or((f32::NAN, f32::NAN));
                let (avg_val, std_val) = val_map
                    .get(&epoch)
                    .map(|v| compute_loss_stats(v))
                    .map_or((None, None), |(avg, std)| (Some(avg), Some(std)));

                (epoch, avg_train, avg_val, std_train, std_val)
            })
            .collect()
    }

    /// Computes the average and standard deviation of precision, recall, accuracy, and regression metrics grouped by epoch and training phase.
    ///
    /// # Returns
    /// A `HashMap` where each key is a tuple `(epoch, TrainingPhase)` and each value is a tuple of:
    /// `(avg_precision, std_precision, avg_recall, std_recall, avg_accuracy, std_accuracy, avg_mae, std_mae, avg_rmse, std_rmse, avg_r2, std_r2)`.
    pub fn summarize_metrics_by_epoch_phase(
        &self,
    ) -> std::collections::HashMap<
        (usize, TrainingPhase),
        (
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<f32>,
        ),
    > {
        use std::collections::{HashMap, HashSet};

        let mut prec_map: HashMap<(usize, TrainingPhase), Vec<f32>> = HashMap::new();
        let mut rec_map: HashMap<(usize, TrainingPhase), Vec<f32>> = HashMap::new();
        let mut acc_map: HashMap<(usize, TrainingPhase), Vec<f32>> = HashMap::new();
        let mut mae_map: HashMap<(usize, TrainingPhase), Vec<f32>> = HashMap::new();
        let mut rmse_map: HashMap<(usize, TrainingPhase), Vec<f32>> = HashMap::new();
        let mut r2_map: HashMap<(usize, TrainingPhase), Vec<f32>> = HashMap::new();

        for i in 0..self.epochs.len() {
            let key = (self.epochs[i], self.phases[i].clone());
            if let Some(p) = self.precisions[i] {
                prec_map.entry(key.clone()).or_default().push(p);
            }
            if let Some(r) = self.recalls[i] {
                rec_map.entry(key.clone()).or_default().push(r);
            }
            if let Some(a) = self.accuracies[i] {
                acc_map.entry(key.clone()).or_default().push(a);
            }
            if let Some(m) = self.maes[i] {
                mae_map.entry(key.clone()).or_default().push(m);
            }
            if let Some(r) = self.rmses[i] {
                rmse_map.entry(key.clone()).or_default().push(r);
            }
            if let Some(r2) = self.r2s[i] {
                r2_map.entry(key.clone()).or_default().push(r2);
            }
        }

        let mut result = HashMap::new();
        let keys: HashSet<_> = self
            .epochs
            .iter()
            .zip(&self.phases)
            .map(|(e, p)| (*e, p.clone()))
            .collect();

        let summarize = |vals: &Vec<f32>| {
            let avg = vals.iter().copied().sum::<f32>() / vals.len() as f32;
            let std =
                (vals.iter().map(|v| (v - avg).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
            (avg, std)
        };

        for key in keys {
            let (prec_avg, prec_std) = prec_map
                .get(&key)
                .map(summarize)
                .map_or((None, None), |(a, s)| (Some(a), Some(s)));
            let (rec_avg, rec_std) = rec_map
                .get(&key)
                .map(summarize)
                .map_or((None, None), |(a, s)| (Some(a), Some(s)));
            let (acc_avg, acc_std) = acc_map
                .get(&key)
                .map(summarize)
                .map_or((None, None), |(a, s)| (Some(a), Some(s)));
            let (mae_avg, mae_std) = mae_map
                .get(&key)
                .map(summarize)
                .map_or((None, None), |(a, s)| (Some(a), Some(s)));
            let (rmse_avg, rmse_std) = rmse_map
                .get(&key)
                .map(summarize)
                .map_or((None, None), |(a, s)| (Some(a), Some(s)));
            let (r2_avg, r2_std) = r2_map
                .get(&key)
                .map(summarize)
                .map_or((None, None), |(a, s)| (Some(a), Some(s)));

            result.insert(
                key,
                (
                    prec_avg, prec_std, rec_avg, rec_std, acc_avg, acc_std, mae_avg, mae_std,
                    rmse_avg, rmse_std, r2_avg, r2_std,
                ),
            );
        }

        result
    }
}

/// Utility functions for evaluating prediction metrics.
pub struct Metrics;

impl Metrics {
    /// Computes accuracy as the proportion of predictions within a tolerance of the target.
    pub fn accuracy(pred: &[f32], target: &[f32], tolerance: f32) -> f32 {
        let correct = pred
            .iter()
            .zip(target)
            .filter(|(p, t)| (*p - *t).abs() <= tolerance)
            .count();
        correct as f32 / pred.len() as f32
    }

    /// Computes accuracy as the proportion of predictions within a dynamic tolerance of the target.
    pub fn accuracy_dynamic(pred: &[f32], target: &[f32], tolerance: &[f32]) -> f32 {
        pred.iter()
            .zip(target)
            .zip(tolerance)
            .filter(|((p, t), tol)| (*p - *t).abs() <= **tol)
            .count() as f32
            / pred.len() as f32
    }

    /// Computes precision as TP / (TP + FP), based on a binary threshold.
    pub fn precision(pred: &[f32], target: &[f32], threshold: f32) -> Option<f32> {
        let mut tp = 0;
        let mut fp = 0;
        for (&p, &t) in pred.iter().zip(target) {
            if p > threshold {
                if t > threshold {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        if tp + fp > 0 {
            Some(tp as f32 / (tp + fp) as f32)
        } else {
            None
        }
    }

    /// Computes recall as TP / (TP + FN), based on a binary threshold.
    pub fn recall(pred: &[f32], target: &[f32], threshold: f32) -> Option<f32> {
        let mut tp = 0;
        let mut fn_ = 0;
        for (&p, &t) in pred.iter().zip(target) {
            if t > threshold {
                if p > threshold {
                    tp += 1;
                } else {
                    fn_ += 1;
                }
            }
        }
        if tp + fn_ > 0 {
            Some(tp as f32 / (tp + fn_) as f32)
        } else {
            None
        }
    }

    pub fn mae(pred: &[f32], target: &[f32]) -> f32 {
        pred.iter()
            .zip(target)
            .map(|(p, t)| (p - t).abs())
            .sum::<f32>()
            / pred.len() as f32
    }

    pub fn rmse(pred: &[f32], target: &[f32]) -> f32 {
        let mse = pred
            .iter()
            .zip(target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / pred.len() as f32;
        mse.sqrt()
    }

    pub fn r2(pred: &[f32], target: &[f32]) -> f32 {
        let mean_t = target.iter().copied().sum::<f32>() / target.len() as f32;
        let ss_res = pred
            .iter()
            .zip(target)
            .map(|(p, t)| (t - p).powi(2))
            .sum::<f32>();
        let ss_tot = target.iter().map(|t| (t - mean_t).powi(2)).sum::<f32>();
        if ss_tot.abs() < 1e-9 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }
}

/// Compute average and std deviation from a slice of loss values.
pub fn compute_loss_stats(losses: &[f32]) -> (f32, f32) {
    let avg = losses.iter().copied().sum::<f32>() / losses.len() as f32;
    let std = (losses.iter().map(|l| (l - avg).powi(2)).sum::<f32>() / losses.len() as f32).sqrt();
    (avg, std)
}
