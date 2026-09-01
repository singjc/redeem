//! Create one model-only SafeTensors initialization snapshot for controlled experiments.
//!
//! The snapshot contains model parameters only. Reusing the same file across
//! fresh runs gives paired experiments byte-identical starting parameters while
//! each run still creates a fresh optimizer with zero moments.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use redeem_properties::foundation::{read_foundation_training_run_config, FoundationModelWrapper};
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("usage: foundation_initialize_model <training.yaml> <output.safetensors>");
    }
    let config = read_foundation_training_run_config(&args[0])?;
    let output = PathBuf::from(&args[1]);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create initialization directory {parent:?}"))?;
    }

    let wrapper = FoundationModelWrapper::new(config.model, Device::Cpu)?;
    wrapper
        .save_safetensors(&output)
        .with_context(|| format!("failed to save foundation initialization {output:?}"))?;

    println!("initial_model_safetensors\t{}", output.display());
    Ok(())
}
