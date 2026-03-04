pub mod losses;
pub mod trainer;
pub mod pipeline;
pub mod scheduler;

pub use trainer::{TrainBatch, TrainMetrics, Trainer};
pub use scheduler::CosineWarmupScheduler;
pub use pipeline::{
    TrainFilter,
    filter_training_rows,
    fit_preprocessor_from_rows,
    split_rows_by_precursor,
    bags_to_train_batches,
};
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub use pipeline::build_train_batches_from_osw_xic;
