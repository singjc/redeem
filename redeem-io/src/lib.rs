//! IO helpers shared by ReDeeM models.
//!
//! This crate intentionally keeps file-format concerns separate from the model
//! crates. At the moment it focuses on:
//!
//! - OSW feature-table reading and score-table writeback via SQLite.
//! - OpenMS chromatogram parquet decoding, including MSNumpress payloads.
//! - Simple in-memory XIC domain types reused by `redeem-topaz`.

pub mod osw;
pub mod xic;
pub mod msnumpress;
pub mod xic_parquet;

pub use osw::{FeatureRow, OswFeatureTable, OswReadConfig, OswLevel, ScoreRow};
pub use xic::{PrecursorXic, TransitionTrace, XicPoint, XicSource};
pub use xic_parquet::XicParquetReader;
