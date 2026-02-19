//! redeem-classifiers: machine-learning helpers for PSM rescoring.
//!
//! This crate provides lightweight model wrappers (GBDT, optional XGBoost and SVM),
//! data handling and preprocessing utilities, feature selection helpers, and
//! reporting/plotting helpers used by the examples and higher-level tooling.
//!
//! The design favors small, testable modules with feature flags to avoid
//! requiring native dependencies (e.g., libxgboost) unless explicitly enabled.
pub mod config;
pub mod data_handling;
pub mod feature_selection;
pub mod io;
pub mod math;
pub mod models;
pub mod preprocessing;
pub mod psm_scorer;
pub mod report;
pub mod stats;
