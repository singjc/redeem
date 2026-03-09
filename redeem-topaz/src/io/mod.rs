//! Thin re-exports of shared IO types from `redeem-io`.
//!
//! `redeem-topaz` keeps the model-facing types local to avoid forcing callers
//! to depend on two crates for the common `FeatureRow` / `PrecursorXic`
//! vocabulary.

pub use redeem_io::msnumpress;
pub use redeem_io::osw;
pub use redeem_io::xic;
pub use redeem_io::xic_parquet;
pub use redeem_io::xim;
pub use redeem_io::xim_parquet;

pub use redeem_io::osw::FeatureRow;
pub use redeem_io::xic::{PrecursorXic, TransitionTrace, XicPoint, XicSource};
pub use redeem_io::xim::{FeatureXim, MobilogramTrace, XimPoint, XimSource};
