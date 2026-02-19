//! Classifier trait shared by model implementations.
//!
//! This trait defines the minimal contract expected by `SemiSupervisedLearner`.
//! Implementations adapt their internal APIs to satisfy `fit`, `predict` and
//! `predict_proba`.
use crate::math::Array2;
/// A small trait abstraction for classifier models used by the semi-supervised
/// learner. This mirrors the existing internal `SemiSupervisedModel` API but
/// centralizes the contract in the `models` module so implementations can live
/// next to model code.
pub trait ClassifierModel: Send {
    /// Fit the model. `y` uses the crate convention (1 for target, -1 for decoy)
    fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        x_eval: Option<&Array2<f32>>,
        y_eval: Option<&[i32]>,
    );

    /// Predict raw scores (may be margins or probabilistic depending on impl)
    fn predict(&self, x: &Array2<f32>) -> Vec<f32>;

    /// Predict probabilities (0..1) when available. Implementations that only
    /// produce margins should convert appropriately.
    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32>;

    /// Create a boxed clone configured identically to `self` but with no
    /// trained state. This is used to spawn per-fold models so each fold can
    /// train independently.
    fn clone_box(&self) -> Box<dyn ClassifierModel>;

    /// Optional human readable name for the model
    fn name(&self) -> &str {
        "classifier"
    }

    /// Optional linear feature weights/importances for the trained model.
    /// Returns `None` when not supported or before fitting.
    fn feature_weights(&self) -> Option<Vec<f32>> {
        None
    }
}
