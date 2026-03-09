//! Inference, scoring statistics, diagnostics, and trace-to-tensor helpers.
//!
//! The inference layer sits between raw OSW/XIC data and final score tables:
//!
//! - `pipeline` builds model-ready tensors from rows and chromatograms.
//! - `score_rows` and `score_bags` provide chunked scoring helpers.
//! - `stats` computes ranking, p-values, q-values, and PEPs.
//! - `diagnostics` contains lightweight numerical summaries that can later be
//!   consumed by reports and plotting code.

pub mod diagnostics;
pub mod pipeline;
pub mod score_bags;
pub mod score_rows;
pub mod score_table;
pub mod stats;

#[cfg(feature = "io-sqlite")]
pub use diagnostics::write_rank1_disagreement_tsvs;
pub use diagnostics::{
    Rank1DisagreementSummary, TraceSummary, print_trace_summary, probe_ms1_presence, trace_summary,
    warn_if_missing_ms1,
};
#[cfg(feature = "io-parquet")]
pub use pipeline::build_trace_tensors_from_parquet;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_trace_tensors_from_parquet_cached;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_trace_tensors_from_parquet_map;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_trace_tensors_from_parquet_map_cached;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_xim_tensors_from_parquet;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_xim_tensors_from_parquet_cached;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_xim_tensors_from_parquet_map;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_xim_tensors_from_parquet_map_cached;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub use pipeline::infer_score_table_from_osw_xic;
#[cfg(feature = "io-sqlite")]
pub use pipeline::read_osw_features;
pub use pipeline::{
    BagHeadOutput, BagScoreOutput, TraceBuildConfig, XicFetchConfig, XimFetchConfig,
    build_trace_tensors_from_source, rows_to_feature_matrix, rows_to_feature_matrix_preprocessed,
    rows_to_feature_matrix_with_cols, score_bags_from_rows, score_bags_from_rows_with_cols,
    score_bags_with_heads_from_rows, score_bags_with_heads_from_rows_with_cols,
    score_rows_from_rows, score_rows_from_rows_with_cols,
};
#[cfg(feature = "io-parquet")]
pub use pipeline::{SharedXicCache, XicCacheStats, XicDiskCache};
#[cfg(feature = "io-parquet")]
pub use pipeline::{SharedXimCache, XimCacheStats, XimDiskCache};
pub use score_bags::{ScoreBags, score_bags, score_bags_with_aux};
pub use score_rows::{ScoreRows, score_candidates, score_candidates_with_aux};
pub use score_table::{
    ScoreTableRow, build_score_table, build_score_table_from_rows, write_score_tsv,
};
pub use stats::{
    TdcSummary, binned_pep, decoy_tail_pvalues, rank_within_key, tdc_qvalues, tdc_summary,
};
