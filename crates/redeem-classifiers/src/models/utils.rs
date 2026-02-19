//! Convenience re-exports used by examples and tests.
//!
//! This module provides a small set of re-exported types (e.g. `ModelConfig`)
//! so callers can import configuration types from `models::utils` when
//! convenient. Prefer importing from `crate::config` in library code.
pub use crate::config::{ModelConfig, ModelType};
