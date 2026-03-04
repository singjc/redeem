// redeem-topaz/src/lib.rs

pub mod building_blocks;
pub mod model;
pub mod train;
pub mod infer;
pub mod xrun;
pub mod io;
pub mod preprocess;
pub mod run;

pub mod config;
pub mod checkpoint;
pub mod model_interface;

pub use model::topaz::{TopazBagRanker, TopazConfig};
pub use model_interface::{
    BagRankerInterface, BagRankerWithHiddenInterface, CandidateScorerInterface, ModelInterface,
};
pub use preprocess::Preprocessor;
pub use run::{
    DiagnosticsConfig,
    TrainRunConfig,
    TrainRunOutput,
    InferRunConfig,
    InferRunOutput,
    XrunSweepConfig,
    XrunSweepRow,
    run_training,
    run_inference,
    run_xrun_sweep,
};
pub use xrun::calibrator::{XrunAttentionCalibrator, XrunConfig};
