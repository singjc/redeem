//! Input transforms applied before convolutional trace encoding.
//!
//! Shape notation used in this module:
//!
//! - `N`: number of flattened candidate rows.
//! - `C`: number of chromatogram channels for each row.
//! - `L`: fixed trace-window length.
//!
//! In other words, a raw trace tensor has shape `(N, C, L)`, meaning "for each
//! candidate row, `C` aligned traces sampled over `L` retention-time points".

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

/// How a raw trace tensor should be presented to the convolutional encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceInputMode {
    /// Use the extracted trace tensor exactly as it was built by the trace
    /// windowing code.
    ///
    /// Input shape: `(N, C, L)`  
    /// Output shape: `(N, C, L)`
    Single,
    /// Concatenate two views of the same traces along the channel axis:
    ///
    /// - max-normalized traces: `x / (amax(x) + eps)`
    /// - log-intensity traces: `log1p(max(x, 0))`
    ///
    /// This mirrors the Python `trace_input_mode="dual"` behavior and lets the
    /// encoder see both relative shape information and compressed absolute
    /// intensity information.
    ///
    /// Input shape: `(N, C, L)`  
    /// Output shape: `(N, 2C, L)`
    Dual,
}

impl TraceInputMode {
    /// Parse a user-facing string into a trace-input mode.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "dual" => Self::Dual,
            _ => Self::Single,
        }
    }
}

/// Prepare a raw trace tensor for the convolutional encoder.
///
/// # Inputs
/// - `x`: raw trace tensor `(N, C, L)`.
/// - `mode`: whether to keep the tensor as-is or expand it into the dual-view
///   representation expected by the Python model.
/// - `eps`: small positive stabilizer used during per-channel max
///   normalization.
///
/// # Output
/// Returns either the original `(N, C, L)` tensor or the dual-view
/// `(N, 2C, L)` tensor, depending on `mode`.
pub fn make_trace_input(x: &Tensor, mode: TraceInputMode, eps: f64) -> Result<Tensor> {
    match mode {
        TraceInputMode::Single => Ok(x.clone()),
        TraceInputMode::Dual => {
            // x0 = x / (amax(x, dim=2, keepdim=true) + eps)
            // x1 = log1p(clamp_min(x,0))
            // cat along C dim -> (N,2C,L)

            let m = x.max_keepdim(2)?;
            let eps_t = Tensor::full(eps as f32, m.dims(), m.device())?.to_dtype(m.dtype())?;
            let m = m.broadcast_add(&eps_t)?;
            let x0 = x.broadcast_div(&m)?;

            let x_cl = x.maximum(0f32)?;
            let ones = Tensor::ones(x_cl.dims(), x_cl.dtype(), x_cl.device())?;
            let x1 = x_cl.broadcast_add(&ones)?.log()?;

            Tensor::cat(&[x0, x1], 1)
        }
    }
}
