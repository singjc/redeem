//! Model implementations and factory functions.
//!
//! This module exposes concrete model implementations (GBDT by default,
//! optional `xgboost` and `svm` behind feature flags) and a small
//! `ClassifierModel` trait plus a `factory` for runtime construction.
pub mod gbdt;
#[cfg(feature = "svm")]
pub mod svm;
#[cfg(feature = "xgboost")]
pub mod xgboost;

pub mod classifier_trait;
pub mod factory;
