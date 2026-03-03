// redeem-topaz/src/building_blocks/trace_input.rs

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceInputMode {
    Single,
    Dual,
}

impl TraceInputMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "dual" => Self::Dual,
            _ => Self::Single,
        }
    }
}

/// x: (N,C,L)
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
