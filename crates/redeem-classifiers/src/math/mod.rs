//! Small ndarray-like types used throughout the crate.
//!
//! Provides `Array2` (2D) and `Array1` (1D) lightweight containers with
//! minimal convenience methods. These types are intentionally small and
//! dependency-free to keep the crate portable and easy to test.
pub mod matrix;
pub mod vector;

pub use matrix::{Array2, ShapeError};
pub use vector::Array1;
