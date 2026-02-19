use crate::config::ModelConfig;
use crate::models::classifier_trait::ClassifierModel;

/// Build a boxed classifier model from a `ModelConfig`.
/// Currently this is a thin factory implemented as a single function.
pub fn build_model(params: ModelConfig) -> Box<dyn ClassifierModel> {
    match params.model_type {
        #[cfg(feature = "xgboost")]
        crate::config::ModelType::XGBoost { .. } => {
            Box::new(crate::models::xgboost::XGBoostClassifier::new(params))
        }

        crate::config::ModelType::GBDT { .. } => {
            Box::new(crate::models::gbdt::GBDTClassifier::new(params))
        }

        #[cfg(feature = "svm")]
        crate::config::ModelType::SVM { .. } => {
            Box::new(crate::models::svm::SVMClassifier::new(params))
        } // When compiled, `ModelType` only contains the variants enabled by
          // features. The above arms are exhaustive for the compiled enum, so
          // no catch-all arm is necessary.
    }
}
