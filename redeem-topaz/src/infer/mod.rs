pub mod score_rows;
pub mod score_bags;
pub mod stats;
pub mod score_table;
pub mod pipeline;
pub mod diagnostics;

pub use score_rows::{score_candidates, ScoreRows};
pub use score_bags::{score_bags, ScoreBags};
pub use stats::{binned_pep, decoy_tail_pvalues, rank_within_key, tdc_qvalues, tdc_summary, TdcSummary};
pub use score_table::{build_score_table, build_score_table_from_rows, write_score_tsv, ScoreTableRow};
pub use diagnostics::{probe_ms1_presence, print_trace_summary, trace_summary, warn_if_missing_ms1, TraceSummary, Rank1DisagreementSummary};
pub use pipeline::{
    build_trace_tensors_from_source,
    rows_to_feature_matrix,
    rows_to_feature_matrix_preprocessed,
    rows_to_feature_matrix_with_cols,
    score_bags_from_rows,
    score_bags_from_rows_with_cols,
    score_bags_with_heads_from_rows,
    score_bags_with_heads_from_rows_with_cols,
    score_rows_from_rows,
    score_rows_from_rows_with_cols,
    BagScoreOutput,
    BagHeadOutput,
    TraceBuildConfig,
    XicFetchConfig,
};
#[cfg(feature = "io-parquet")]
pub use pipeline::build_trace_tensors_from_parquet;
#[cfg(feature = "io-parquet")]
pub use pipeline::build_trace_tensors_from_parquet_map;
#[cfg(feature = "io-sqlite")]
pub use pipeline::read_osw_features;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub use pipeline::infer_score_table_from_osw_xic;
#[cfg(feature = "io-sqlite")]
pub use diagnostics::write_rank1_disagreement_tsvs;
