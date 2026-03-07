//! Cross-run score calibration for TOPAZ bag scores.
//!
//! XRUN treats each precursor as a short sequence over runs where every item is
//! composed of the base bag score plus the base model's winner hidden vector.
//! A lightweight attention calibrator then predicts per-run additive deltas that
//! are applied after base inference.

pub mod calibrator;
pub mod pipeline;
pub mod sequence;
pub mod train;

#[cfg(feature = "io-sqlite")]
pub use pipeline::write_xrun_scores_to_osw;
pub use pipeline::{
    XrunBagData, XrunPredictConfig, apply_xrun_deltas, apply_xrun_deltas_to_rows,
    build_xrun_bag_data_from_rows, build_xrun_bag_data_from_rows_with_cols,
    score_bags_with_hidden_chunked, xrun_predict_deltas_for_bags,
};
pub use sequence::{XrunSeq, build_xrun_sequences_from_bags, split_group_id_run_prec};
pub use train::{
    XrunDataset, XrunPoolMode, XrunTrainConfig, XrunTrainMeta, XrunTrainer, XrunVarWeight,
    split_train_val,
};
