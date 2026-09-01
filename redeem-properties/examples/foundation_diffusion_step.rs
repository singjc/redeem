//! Minimal spectrum-conditioned peptide diffusion training step.
//!
//! This example uses tiny synthetic *observed-spectrum-shaped* peak arrays only
//! to validate the tensor/autograd path. It is not a de-novo benchmark and does
//! not reconstruct theoretical peaks from the peptide sequence.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_diffusion_x0_loss, FoundationAdamW, FoundationAdamWConfig,
    FoundationDiffusionCollator, FoundationDiffusionConfig, FoundationSpectrum,
    FoundationSpectrumCollator, FoundationSpectrumConfig, PeptideSpectrumDiffusionModel,
    PeptidoformInput, PrecursorContextBatch,
};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let config = FoundationDiffusionConfig {
        max_tokens: 24,
        model_dim: 32,
        num_attention_heads: 4,
        feed_forward_dim: 64,
        spectrum_layers: 1,
        decoder_layers: 1,
        spectrum: FoundationSpectrumConfig {
            max_peaks: 8,
            ..FoundationSpectrumConfig::default()
        },
        ..FoundationDiffusionConfig::default()
    };

    let spectra = vec![
        FoundationSpectrum::from_pairs([(101.1, 3.0), (247.2, 10.0), (504.3, 6.0)]),
        FoundationSpectrum::from_pairs([(120.2, 5.0), (333.3, 9.0), (701.4, 2.0)]),
    ];
    let spectrum_batch =
        FoundationSpectrumCollator::new(config.spectrum.clone())?.collate(&spectra, &device)?;
    let peptides = vec![
        PeptidoformInput::unmodified("PEPTIDEK"),
        PeptidoformInput::unmodified("MELTQK"),
    ];
    let diffusion_batch = FoundationDiffusionCollator::new(config.clone())?.collate(
        &peptides,
        &[5, 15],
        20260901,
        &device,
    )?;
    let precursor = PrecursorContextBatch {
        charge: Tensor::new(&[2.0f32, 3.0], &device)?,
        charge_present: Tensor::ones(2, DType::F32, &device)?,
        precursor_mz: Tensor::new(&[500.0f32, 600.0], &device)?,
        precursor_mz_present: Tensor::ones(2, DType::F32, &device)?,
        nce: Tensor::zeros(2, DType::F32, &device)?,
        nce_present: Tensor::zeros(2, DType::F32, &device)?,
        instrument_ids: Tensor::zeros(2, DType::U32, &device)?,
        instrument_present: Tensor::zeros(2, DType::F32, &device)?,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumDiffusionModel::new(config, vb)?;
    let output = model.forward_t(&diffusion_batch, &spectrum_batch, &precursor, true)?;
    let loss = foundation_diffusion_x0_loss(&output, &diffusion_batch)?;
    println!("diffusion_loss\t{}", loss.to_scalar::<f32>()?);
    println!("token_logits\t{:?}", output.token_logits.dims());
    println!("spectrum_memory\t{:?}", output.spectrum_memory.dims());

    let mut optimizer = FoundationAdamW::new(&varmap, FoundationAdamWConfig::default())?;
    let step = optimizer.backward_step(&loss, Some(5.0))?;
    println!("optimizer_step\t{}", step.step);
    println!("gradient_norm\t{}", step.gradient_norm);
    println!("gradient_scale\t{}", step.gradient_scale);
    Ok(())
}
