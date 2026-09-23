//! v0.39 precursor-conditioned conformer-token CCS representation on frozen v0.35.
//!
//! v0.38 established that a task-isolated trainable CCS representation is
//! materially better than a detached scalar residual, but it still missed the
//! fixed DEV promotion gate. v0.38.1 showed that heavy gradient clipping was
//! not the scientific bottleneck. v0.39 therefore changes the representation
//! mechanism rather than optimizer conditioning.
//!
//! The CCS branch is initialized from the accepted v0.35 encoder/property
//! representation. Precursor physics is encoded as an explicit conformer token
//! prepended to the residue sequence, allowing bidirectional self-attention
//! between charge/mass context and residue features before global pooling.
//! A zero-initialized native ion-mobility residual preserves exact v0.35 step-0
//! behavior. RT, MS2, and inverse paths remain immutable.

use super::config::FoundationConfig;
use super::diffusion::FoundationDiffusionConfig;
use super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
use super::model::{FoundationOutput, PrecursorContextBatch};
use super::multimodal_v0270::PeptideFoundationMultimodalForwardV0270;
use super::multimodal_v0350::{
    FoundationFragmentContextBatchV0350, FoundationMultimodalForwardOutputV0350,
    PeptideFoundationMultimodalV0350Config, PeptideFoundationMultimodalV0350Model,
    PropertyResidueRefinementV0350,
};
use super::multimodal_v0360::{
    FoundationScalarPhysicsBatchV0360, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
};
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0390: &str =
    "v0.39-frozen-v0350-precursor-conditioned-conformer-token-ccs";
pub const FOUNDATION_CCS_CONFORMER_LAYERS_V0390: usize = 2;
pub const FOUNDATION_CCS_CONFORMER_HEADS_V0390: usize = 6;
pub const FOUNDATION_CCS_CONFORMER_FF_DIM_V0390: usize = 768;
pub const FOUNDATION_CCS_MOBILITY_HIDDEN_V0390: usize = 384;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationMultimodalV0390Config {
    pub base_v0350: PeptideFoundationMultimodalV0350Config,
    pub ccs_conformer_layers: usize,
    pub ccs_conformer_heads: usize,
    pub ccs_conformer_ff_dim: usize,
    pub ccs_mobility_hidden: usize,
}

impl PeptideFoundationMultimodalV0390Config {
    pub fn fixed(base_v0350: PeptideFoundationMultimodalV0350Config) -> Result<Self> {
        base_v0350.validate()?;
        let config = Self {
            base_v0350,
            ccs_conformer_layers: FOUNDATION_CCS_CONFORMER_LAYERS_V0390,
            ccs_conformer_heads: FOUNDATION_CCS_CONFORMER_HEADS_V0390,
            ccs_conformer_ff_dim: FOUNDATION_CCS_CONFORMER_FF_DIM_V0390,
            ccs_mobility_hidden: FOUNDATION_CCS_MOBILITY_HIDDEN_V0390,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0350.validate()?;
        if self.forward().model_dim != 192 {
            candle_core::bail!("v0.39 requires the accepted 192d v0.35 representation");
        }
        if self.ccs_conformer_layers != FOUNDATION_CCS_CONFORMER_LAYERS_V0390
            || self.ccs_conformer_heads != FOUNDATION_CCS_CONFORMER_HEADS_V0390
            || self.ccs_conformer_ff_dim != FOUNDATION_CCS_CONFORMER_FF_DIM_V0390
            || self.ccs_mobility_hidden != FOUNDATION_CCS_MOBILITY_HIDDEN_V0390
        {
            candle_core::bail!("v0.39 dimensions differ from the fixed architecture");
        }
        if self.forward().model_dim % self.ccs_conformer_heads != 0 {
            candle_core::bail!("v0.39 model width must be divisible by CCS attention heads");
        }
        Ok(())
    }

    pub fn forward(&self) -> &FoundationConfig {
        self.base_v0350.forward()
    }

    pub fn inverse(&self) -> &FoundationDiffusionConfig {
        self.base_v0350.inverse()
    }
}

#[derive(Clone)]
struct ConformerTokenHeadV0390 {
    physics_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
    model_dim: usize,
}

impl ConformerTokenHeadV0390 {
    fn new(config: &PeptideFoundationMultimodalV0390Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        let mut blocks = Vec::with_capacity(config.ccs_conformer_layers);
        for layer in 0..config.ccs_conformer_layers {
            blocks.push(PeptideTransformerBlock::new(
                model_dim,
                config.ccs_conformer_heads,
                config.ccs_conformer_ff_dim,
                config.forward().dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        let head_input = 3 * model_dim + FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            physics_projection: nn::linear(
                FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
                model_dim,
                vb.pp("physics_projection"),
            )?,
            input_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("input_norm"))?,
            blocks,
            hidden: nn::linear(head_input, config.ccs_mobility_hidden, vb.pp("hidden"))?,
            bottleneck: nn::linear(config.ccs_mobility_hidden, 192, vb.pp("bottleneck"))?,
            output: zero_initialized_linear_v0390(192, 1, vb.pp("output"))?,
            model_dim,
        })
    }

    fn forward_t(
        &self,
        foundation: &FoundationOutput,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<Tensor> {
        let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
        if model_dim != self.model_dim {
            candle_core::bail!("v0.39 conformer head received incompatible residue width");
        }

        // The precursor-conditioned token is part of self-attention rather than
        // being added uniformly to every residue. Residues can therefore adapt
        // differently to the same charge/mass context.
        let conformer_token = self
            .physics_projection
            .forward(&physics.ccs_physics)?
            .unsqueeze(1)?;
        let extended = Tensor::cat(&[&conformer_token, &foundation.residue_embeddings], 1)?;
        let mut hidden = self.input_norm.forward(&extended)?;

        let token_mask = foundation.residue_mask.narrow(1, 0, 1)?.affine(0.0, 1.0)?;
        let extended_mask = Tensor::cat(&[&token_mask, &foundation.residue_mask], 1)?;
        let expanded_mask =
            extended_mask
                .unsqueeze(2)?
                .broadcast_as((batch, sequence + 1, model_dim))?;
        hidden = hidden.broadcast_mul(&expanded_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &extended_mask, train)?;
        }

        let conformer = hidden.narrow(1, 0, 1)?.squeeze(1)?;
        let residue_hidden = hidden.narrow(1, 1, sequence)?;
        let pooled = masked_mean_v0390(&residue_hidden, &foundation.residue_mask)?;
        let features = Tensor::cat(
            &[
                &foundation.peptide_embedding,
                &conformer,
                &pooled,
                &physics.ccs_physics,
            ],
            1,
        )?;
        let hidden = self.hidden.forward(&features)?.relu()?;
        let hidden = self.bottleneck.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
pub struct FoundationMobilityOutputV0390 {
    /// Frozen v0.35 CCS output in the parent model's normalized CCS coordinate.
    pub base_ccs_model: Tensor,
    /// Native raw ion-mobility residual. Step 0 is exactly zero.
    pub mobility_residual_native: Tensor,
}

#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0390Model {
    base_v0350: PeptideFoundationMultimodalV0350Model,
    ccs_forward_v0390: PeptideFoundationMultimodalForwardV0270,
    ccs_property_refinement_v0390: PropertyResidueRefinementV0350,
    ccs_conformer_v0390: ConformerTokenHeadV0390,
    config: PeptideFoundationMultimodalV0390Config,
}

impl PeptideFoundationMultimodalV0390Model {
    pub fn new(config: PeptideFoundationMultimodalV0390Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let base_v0350 =
            PeptideFoundationMultimodalV0350Model::new(config.base_v0350.clone(), vb.clone())?;
        let ccs_forward_v0390 = PeptideFoundationMultimodalForwardV0270::new(
            config.base_v0350.base_v0310.base_v0270.clone(),
            vb.pp("ccs_forward_v0390"),
        )?;
        let ccs_property_refinement_v0390 = PropertyResidueRefinementV0350::new(
            &config.base_v0350,
            vb.pp("ccs_property_refinement_v0390"),
        )?;
        let ccs_conformer_v0390 =
            ConformerTokenHeadV0390::new(&config, vb.pp("ccs_conformer_v0390"))?;
        Ok(Self {
            base_v0350,
            ccs_forward_v0390,
            ccs_property_refinement_v0390,
            ccs_conformer_v0390,
            config,
        })
    }

    /// CCS-only trainable path. The accepted v0.35 anchor remains detached.
    pub fn mobility_v0390_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<FoundationMobilityOutputV0390> {
        let (_, _, base_ccs_model) = self
            .base_v0350
            .detached_scalar_anchor_v0350_t(batch, context)?;
        let base = self.ccs_forward_v0390.encode_foundation_t(batch, train)?;
        let foundation = self.ccs_property_refinement_v0390.forward_t(&base, train)?;
        let mobility_residual_native =
            self.ccs_conformer_v0390
                .forward_t(&foundation, physics, train)?;
        Ok(FoundationMobilityOutputV0390 {
            base_ccs_model,
            mobility_residual_native,
        })
    }

    /// Exact protected v0.35 forward path for RT/MS2 invariance checks.
    pub fn protected_forward_v0350_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0350,
    ) -> Result<FoundationMultimodalForwardOutputV0350> {
        self.base_v0350
            .forward_v0350_t(batch, context, fragment, false)
    }

    pub fn forward_config(&self) -> &FoundationConfig {
        self.config.forward()
    }

    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        self.config.inverse()
    }

    pub fn config(&self) -> &PeptideFoundationMultimodalV0390Config {
        &self.config
    }
}

fn masked_mean_v0390(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn zero_initialized_linear_v0390(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder<'_>,
) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", nn::Init::Const(0.0))?;
    let bias = vb.get_with_hints(out_dim, "bias", nn::Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

#[cfg(test)]
mod tests {
    use super::super::multimodal_v0270::PeptideFoundationMultimodalV0270Config;
    use super::super::multimodal_v0310::PeptideFoundationMultimodalV0310Config;
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn config() -> PeptideFoundationMultimodalV0390Config {
        let mut forward = FoundationConfig::default();
        forward.max_sequence_len = 8;
        forward.max_atoms_per_residue = 24;
        forward.graph_hidden_dim = 32;
        forward.graph_layers = 1;
        forward.model_dim = 192;
        forward.num_attention_heads = 6;
        forward.transformer_layers = 1;
        forward.transformer_ff_dim = 768;
        forward.ms2_fragment_channels = 8;
        let mut inverse = FoundationDiffusionConfig::default();
        inverse.max_tokens = 8;
        inverse.model_dim = 192;
        inverse.num_attention_heads = 6;
        inverse.feed_forward_dim = 768;
        inverse.spectrum_layers = 1;
        inverse.decoder_layers = 1;
        let v0270 = PeptideFoundationMultimodalV0270Config::fixed(forward, inverse).unwrap();
        let v0310 = PeptideFoundationMultimodalV0310Config::fixed(v0270).unwrap();
        let v0350 = PeptideFoundationMultimodalV0350Config::fixed(v0310).unwrap();
        PeptideFoundationMultimodalV0390Config::fixed(v0350).unwrap()
    }

    #[test]
    fn v0390_namespaces_separate_conformer_ccs_from_v0350_anchor() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = PeptideFoundationMultimodalV0390Model::new(config(), vb).unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(data.keys().any(|name| name.starts_with("forward_v0350.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("ccs_forward_v0390.encoder.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("ccs_property_refinement_v0390.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("ccs_conformer_v0390.")));
    }

    #[test]
    fn v0390_mobility_residual_is_zero_initialized() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = PeptideFoundationMultimodalV0390Model::new(config(), vb).unwrap();
        let data = varmap.data().lock().unwrap();
        let weight = data
            .get("ccs_conformer_v0390.output.weight")
            .unwrap()
            .as_tensor()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(weight, 0.0);
    }
}
