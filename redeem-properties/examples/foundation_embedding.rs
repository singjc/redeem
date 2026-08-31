//! Minimal embedding-only example for the ReDeeM peptide foundation encoder.

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    FoundationConfig, PeptideFoundationEncoder, PeptideGraphFeaturizer, PeptidoformInput,
};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let config = FoundationConfig::default();
    let featurizer = PeptideGraphFeaturizer::new(config.clone())?;
    let batch = featurizer.featurize(
        &[
            PeptidoformInput::unmodified("PEPTIDEK"),
            PeptidoformInput::unmodified("AGHCEWQMKYR"),
        ],
        &device,
    )?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let encoder = PeptideFoundationEncoder::new(config, vb)?;
    let output = encoder.forward_t(&batch, false)?;

    println!("peptide embeddings: {:?}", output.peptide_embedding.dims());
    println!("residue embeddings: {:?}", output.residue_embeddings.dims());
    Ok(())
}
