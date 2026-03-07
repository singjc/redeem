//! Reusable low-level components used by TOPAZ and future trace-first models.
//!
//! These modules are intentionally model-agnostic where possible so that new
//! DIA scorers can reuse the same trace extraction, bagging, encoding, and MLP
//! utilities without depending on the concrete `TopazBagRanker`.

pub mod bagging;
pub mod coelution;
pub mod conv_encoder;
pub mod mlp;
pub mod trace_input;
pub mod trace_window;
