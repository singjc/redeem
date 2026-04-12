use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ExperimentError {
    DimensionMismatch(usize, usize), // (expected, actual)
    SingleClass(bool),               // true if only targets, false if only decoys
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ExperimentError::DimensionMismatch(expected, actual) => {
                write!(f, "Expected dimension {}, but got {}", expected, actual)
            }
            ExperimentError::SingleClass(is_target) => {
                if *is_target {
                    write!(f, "Only target class present in the dataset")
                } else {
                    write!(f, "Only decoy class present in the dataset")
                }
            }
        }
    }
}

impl Error for ExperimentError {}

/// Custom error type for TDC calculation failures
#[derive(Debug)]
pub enum TdcError {
    NaNFound(usize), // Number of NaN values found
    LengthMismatch,
}

impl fmt::Display for TdcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TdcError::NaNFound(count) => write!(f, "Found {} NaN values in scores array", count),
            TdcError::LengthMismatch => {
                write!(f, "Scores and target arrays must have equal length")
            }
        }
    }
}

impl Error for TdcError {}
