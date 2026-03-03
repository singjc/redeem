pub mod score_rows;
pub mod score_bags;
pub mod stats;
pub mod score_table;
pub mod pipeline;

pub use score_rows::{score_candidates, ScoreRows};
pub use score_bags::{score_bags, ScoreBags};
pub use stats::{binned_pep, decoy_tail_pvalues, rank_within_key, tdc_qvalues, tdc_summary, TdcSummary};
pub use score_table::{build_score_table, build_score_table_from_rows, write_score_tsv, ScoreTableRow};
pub use pipeline::{
    build_trace_tensors_from_source,
    rows_to_feature_matrix,
    score_bags_from_rows,
    score_rows_from_rows,
    BagScoreOutput,
    TraceBuildConfig,
    XicFetchConfig,
};
#[cfg(feature = "io-parquet")]
pub use pipeline::build_trace_tensors_from_parquet;
#[cfg(feature = "io-sqlite")]
pub use pipeline::read_osw_features;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub use pipeline::infer_score_table_from_osw_xic;
