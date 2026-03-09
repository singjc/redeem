//! Training losses, dataset preparation, scheduling, and optimization.

pub mod losses;
pub mod pipeline;
pub mod scheduler;
pub mod trainer;

#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub use pipeline::bags_to_train_batches;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub use pipeline::bags_to_train_batches_with_aux;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub use pipeline::build_train_batches_from_osw_xic;
pub use pipeline::{
    TrainFilter, filter_training_rows, fit_preprocessor_from_rows,
    fit_preprocessor_from_rows_with_cols, split_rows_by_precursor, subsample_train_rows_by_bag,
};
pub use scheduler::CosineWarmupScheduler;
pub use trainer::{TrainBatch, TrainMetrics, Trainer};
