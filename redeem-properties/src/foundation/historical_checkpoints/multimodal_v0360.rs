//! v0.36 protected scalar-property refinement on top of frozen v0.35.
//!
//! v0.35 established that relearning the forward representation can close a
//! large fraction of the RT/MS2 gap, but its CCS path remained intentionally
//! frozen. v0.36 keeps the complete v0.35 model immutable and trains two small,
//! task-isolated scalar residuals from detached v0.35 representations:
//!
//! - an intrinsic RT residual that may refine RT without acquisition context;
//! - a physics-aware CCS residual that consumes peptide representation plus
//!   precursor charge/mass/length/PTM context.
//!
//! Both output layers are initialized to exact zero, so step 0 reproduces the
//! accepted v0.35 RT/CCS predictions exactly. MS2 and all inverse paths remain
//! bitwise anchored to v0.35 because no v0.35 parameter belongs to the optimizer.

use super::config::FoundationConfig;
use super::data::FoundationTrainingRecord;
use super::diffusion::{foundation_peptidoform_neutral_mass, FoundationDiffusionConfig};
use super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
use super::model::{FoundationOutput, PrecursorContextBatch};
use super::multimodal_v0350::{
    FoundationFragmentContextBatchV0350, FoundationMultimodalForwardOutputV0350,
    PeptideFoundationMultimodalV0350Config, PeptideFoundationMultimodalV0350Model,
};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0360: &str =
    "v0.36-frozen-v0350-protected-rt-ccs-specialists";
pub const FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360: usize = 6;
pub const FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360: usize = 8;
pub const FOUNDATION_CCS_SPECIALIST_LAYERS_V0360: usize = 2;
pub const FOUNDATION_CCS_SPECIALIST_HEADS_V0360: usize = 6;
pub const FOUNDATION_CCS_SPECIALIST_FF_DIM_V0360: usize = 768;
pub const FOUNDATION_CCS_SPECIALIST_HIDDEN_V0360: usize = 384;
pub const FOUNDATION_RT_SPECIALIST_HIDDEN_V0360: usize = 256;
pub const FOUNDATION_SCALAR_ROBUST_DELTA_V0360: f64 = 0.50;
pub const FOUNDATION_RT_STRETCH_TARGET_MAE_V0360: f64 = 4.0;
pub const FOUNDATION_CCS_TARGET_MAE_V0360: f64 = 8.5;
pub const FOUNDATION_CCS_STRETCH_TARGET_MAE_V0360: f64 = 7.5;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationMultimodalV0360Config {
    pub base_v0350: PeptideFoundationMultimodalV0350Config,
    pub ccs_specialist_layers: usize,
    pub ccs_specialist_heads: usize,
    pub ccs_specialist_ff_dim: usize,
    pub ccs_specialist_hidden: usize,
    pub rt_specialist_hidden: usize,
}

impl PeptideFoundationMultimodalV0360Config {
    pub fn fixed(base_v0350: PeptideFoundationMultimodalV0350Config) -> Result<Self> {
        base_v0350.validate()?;
        let config = Self {
            base_v0350,
            ccs_specialist_layers: FOUNDATION_CCS_SPECIALIST_LAYERS_V0360,
            ccs_specialist_heads: FOUNDATION_CCS_SPECIALIST_HEADS_V0360,
            ccs_specialist_ff_dim: FOUNDATION_CCS_SPECIALIST_FF_DIM_V0360,
            ccs_specialist_hidden: FOUNDATION_CCS_SPECIALIST_HIDDEN_V0360,
            rt_specialist_hidden: FOUNDATION_RT_SPECIALIST_HIDDEN_V0360,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0350.validate()?;
        if self.forward().model_dim != 192 {
            candle_core::bail!("v0.36 requires the accepted 192d v0.35 forward representation");
        }
        if self.ccs_specialist_layers != FOUNDATION_CCS_SPECIALIST_LAYERS_V0360
            || self.ccs_specialist_heads != FOUNDATION_CCS_SPECIALIST_HEADS_V0360
            || self.ccs_specialist_ff_dim != FOUNDATION_CCS_SPECIALIST_FF_DIM_V0360
            || self.ccs_specialist_hidden != FOUNDATION_CCS_SPECIALIST_HIDDEN_V0360
            || self.rt_specialist_hidden != FOUNDATION_RT_SPECIALIST_HIDDEN_V0360
        {
            candle_core::bail!("v0.36 dimensions differ from the fixed architecture");
        }
        if self.forward().model_dim % self.ccs_specialist_heads != 0 {
            candle_core::bail!("v0.36 model width must be divisible by CCS attention heads");
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

/// Physics/context features used only by the protected scalar specialists.
///
/// `rt_intrinsic` contains no acquisition variables. `ccs_physics` adds charge
/// and precursor m/z while retaining intrinsic peptide mass/length terms.
#[derive(Debug, Clone)]
pub struct FoundationScalarPhysicsBatchV0360 {
    pub rt_intrinsic: Tensor,
    pub ccs_physics: Tensor,
}

impl FoundationScalarPhysicsBatchV0360 {
    pub fn from_records(
        records: &[FoundationTrainingRecord],
        max_sequence_len: usize,
        device: &Device,
    ) -> Result<Self> {
        if max_sequence_len == 0 {
            candle_core::bail!("v0.36 scalar physics requires positive max_sequence_len");
        }
        let mut rt =
            Vec::<f32>::with_capacity(records.len() * FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360);
        let mut ccs =
            Vec::<f32>::with_capacity(records.len() * FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360);

        for record in records {
            let length = record.peptidoform.sequence.chars().count() as f64;
            let neutral_mass = foundation_peptidoform_neutral_mass(&record.peptidoform)
                .map_err(candle_core::Error::Msg)?;
            let total_mod_mass = record
                .peptidoform
                .modifications
                .iter()
                .map(|modification| f64::from(modification.mass_delta))
                .sum::<f64>();
            let absolute_mod_mass = record
                .peptidoform
                .modifications
                .iter()
                .map(|modification| f64::from(modification.mass_delta).abs())
                .sum::<f64>();
            let modification_count = record.peptidoform.modifications.len() as f64;
            let length_scaled = length / max_sequence_len as f64;
            let mass_scaled = neutral_mass / 3000.0;
            let sqrt_mass_scaled = neutral_mass.max(0.0).sqrt() / 60.0;
            let total_mod_scaled = total_mod_mass / 500.0;
            let abs_mod_scaled = absolute_mod_mass / 500.0;
            let mod_count_scaled = modification_count / 8.0;

            rt.extend_from_slice(&[
                length_scaled as f32,
                mass_scaled as f32,
                sqrt_mass_scaled as f32,
                total_mod_scaled as f32,
                abs_mod_scaled as f32,
                mod_count_scaled as f32,
            ]);

            let charge = f64::from(record.context.charge.unwrap_or(0));
            let charge_present = f64::from(record.context.charge.is_some() as u8);
            let precursor_mz = f64::from(record.context.precursor_mz.unwrap_or(0.0));
            let mz_present = f64::from(record.context.precursor_mz.is_some() as u8);
            ccs.extend_from_slice(&[
                mass_scaled as f32,
                sqrt_mass_scaled as f32,
                length_scaled as f32,
                (charge / 6.0) as f32,
                ((charge * charge) / 36.0) as f32,
                charge_present as f32,
                (precursor_mz / 2000.0) as f32,
                mz_present as f32,
            ]);
        }

        let batch = records.len();
        Ok(Self {
            rt_intrinsic: Tensor::from_vec(
                rt,
                (batch, FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360),
                device,
            )?,
            ccs_physics: Tensor::from_vec(
                ccs,
                (batch, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360),
                device,
            )?,
        })
    }
}

#[derive(Clone)]
struct RtResidualSpecialistV0360 {
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
    model_dim: usize,
}

impl RtResidualSpecialistV0360 {
    fn new(config: &PeptideFoundationMultimodalV0360Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        let input_dim = model_dim + FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            hidden: nn::linear(input_dim, config.rt_specialist_hidden, vb.pp("hidden"))?,
            bottleneck: nn::linear(config.rt_specialist_hidden, 128, vb.pp("bottleneck"))?,
            output: zero_initialized_linear_v0360(128, 1, vb.pp("output"))?,
            model_dim,
        })
    }

    fn forward(
        &self,
        foundation: &FoundationOutput,
        physics: &FoundationScalarPhysicsBatchV0360,
    ) -> Result<Tensor> {
        let (_, model_dim) = foundation.peptide_embedding.dims2()?;
        if model_dim != self.model_dim {
            candle_core::bail!("v0.36 RT specialist received incompatible peptide width");
        }
        let features = Tensor::cat(&[&foundation.peptide_embedding, &physics.rt_intrinsic], 1)?;
        let hidden = self.hidden.forward(&features)?.relu()?;
        let hidden = self.bottleneck.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)
    }
}

#[derive(Clone)]
struct CcsResidualSpecialistV0360 {
    physics_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
    model_dim: usize,
}

impl CcsResidualSpecialistV0360 {
    fn new(config: &PeptideFoundationMultimodalV0360Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        let mut blocks = Vec::with_capacity(config.ccs_specialist_layers);
        for layer in 0..config.ccs_specialist_layers {
            blocks.push(PeptideTransformerBlock::new(
                model_dim,
                config.ccs_specialist_heads,
                config.ccs_specialist_ff_dim,
                config.forward().dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        let head_input = 2 * model_dim + FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            physics_projection: nn::linear(
                FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
                model_dim,
                vb.pp("physics_projection"),
            )?,
            input_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("input_norm"))?,
            blocks,
            hidden: nn::linear(head_input, config.ccs_specialist_hidden, vb.pp("hidden"))?,
            bottleneck: nn::linear(config.ccs_specialist_hidden, 192, vb.pp("bottleneck"))?,
            output: zero_initialized_linear_v0360(192, 1, vb.pp("output"))?,
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
            candle_core::bail!("v0.36 CCS specialist received incompatible residue width");
        }
        let context = self.physics_projection.forward(&physics.ccs_physics)?;
        let context_residue = context
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, model_dim))?;
        let mut hidden = (&foundation.residue_embeddings + &context_residue)?;
        hidden = self.input_norm.forward(&hidden)?;
        let expanded_mask = foundation
            .residue_mask
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, model_dim))?;
        hidden = hidden.broadcast_mul(&expanded_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &foundation.residue_mask, train)?;
        }
        let pooled = masked_mean_v0360(&hidden, &foundation.residue_mask)?;
        let features = Tensor::cat(
            &[&foundation.peptide_embedding, &pooled, &physics.ccs_physics],
            1,
        )?;
        let hidden = self.hidden.forward(&features)?.relu()?;
        let hidden = self.bottleneck.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
pub struct FoundationScalarOutputV0360 {
    pub rt: Tensor,
    pub ccs: Tensor,
    pub rt_residual: Tensor,
    pub ccs_residual: Tensor,
}

#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0360 {
    pub base: FoundationMultimodalForwardOutputV0350,
    pub rt_residual: Tensor,
    pub ccs_residual: Tensor,
}

#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0360Model {
    base_v0350: PeptideFoundationMultimodalV0350Model,
    rt_specialist_v0360: RtResidualSpecialistV0360,
    ccs_specialist_v0360: CcsResidualSpecialistV0360,
    config: PeptideFoundationMultimodalV0360Config,
}

impl PeptideFoundationMultimodalV0360Model {
    pub fn new(config: PeptideFoundationMultimodalV0360Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let base_v0350 =
            PeptideFoundationMultimodalV0350Model::new(config.base_v0350.clone(), vb.clone())?;
        let rt_specialist_v0360 =
            RtResidualSpecialistV0360::new(&config, vb.pp("rt_specialist_v0360"))?;
        let ccs_specialist_v0360 =
            CcsResidualSpecialistV0360::new(&config, vb.pp("ccs_specialist_v0360"))?;
        Ok(Self {
            base_v0350,
            rt_specialist_v0360,
            ccs_specialist_v0360,
            config,
        })
    }

    /// Fast scalar-only path for v0.36 optimization. The v0.35 anchor emits a
    /// detached representation plus frozen RT/CCS predictions; no MS2 decoder
    /// is evaluated during optimizer steps.
    pub fn scalar_v0360_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<FoundationScalarOutputV0360> {
        let (foundation, base_rt, base_ccs) = self
            .base_v0350
            .detached_scalar_anchor_v0350_t(batch, context)?;
        let rt_residual = self.rt_specialist_v0360.forward(&foundation, physics)?;
        let ccs_residual = self
            .ccs_specialist_v0360
            .forward_t(&foundation, physics, train)?;
        let rt = (&base_rt + &rt_residual)?;
        let ccs = (&base_ccs + &ccs_residual)?;
        Ok(FoundationScalarOutputV0360 {
            rt,
            ccs,
            rt_residual,
            ccs_residual,
        })
    }

    /// Full evaluation path. v0.35 RT/MS2 are evaluated exactly as before and
    /// only the two scalar predictions are replaced by the isolated residuals.
    pub fn forward_v0360_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0350,
        physics: &FoundationScalarPhysicsBatchV0360,
    ) -> Result<FoundationMultimodalForwardOutputV0360> {
        let mut base = self
            .base_v0350
            .forward_v0350_t(batch, context, fragment, false)?;
        let detached = detach_foundation_v0360(&base.base.foundation);
        let rt_residual = self.rt_specialist_v0360.forward(&detached, physics)?;
        let ccs_residual = self
            .ccs_specialist_v0360
            .forward_t(&detached, physics, false)?;
        base.base.rt = (base.base.rt.detach() + &rt_residual)?;
        base.base.ccs = (base.base.ccs.detach() + &ccs_residual)?;
        base.base.ms2 = base.base.ms2.detach();
        base.ms2_presence_logits = base.ms2_presence_logits.detach();
        base.ms2_positive_intensity = base.ms2_positive_intensity.detach();
        base.fragment_representation_aux = base.fragment_representation_aux.detach();
        Ok(FoundationMultimodalForwardOutputV0360 {
            base,
            rt_residual,
            ccs_residual,
        })
    }

    pub fn forward_config(&self) -> &FoundationConfig {
        self.config.forward()
    }

    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        self.config.inverse()
    }

    pub fn config(&self) -> &PeptideFoundationMultimodalV0360Config {
        &self.config
    }
}

fn detach_foundation_v0360(base: &FoundationOutput) -> FoundationOutput {
    FoundationOutput {
        residue_embeddings: base.residue_embeddings.detach(),
        peptide_embedding: base.peptide_embedding.detach(),
        residue_mask: base.residue_mask.clone(),
        chemistry_targets: base.chemistry_targets.clone(),
    }
}

fn masked_mean_v0360(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn zero_initialized_linear_v0360(
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

    fn config() -> PeptideFoundationMultimodalV0360Config {
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
        PeptideFoundationMultimodalV0360Config::fixed(v0350).unwrap()
    }

    #[test]
    fn v0360_optimizer_namespaces_are_task_isolated() {
        let varmap = VarMap::new();
        let _model = PeptideFoundationMultimodalV0360Model::new(
            config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(data
            .keys()
            .any(|name| name.starts_with("rt_specialist_v0360.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("ccs_specialist_v0360.")));
        assert!(data.keys().any(|name| name.starts_with("forward_v0350.")));
    }

    #[test]
    fn v0360_residual_outputs_are_zero_initialized() {
        let varmap = VarMap::new();
        let _model = PeptideFoundationMultimodalV0360Model::new(
            config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let data = varmap.data().lock().unwrap();
        for name in [
            "rt_specialist_v0360.output.weight",
            "ccs_specialist_v0360.output.weight",
        ] {
            let values = data
                .get(name)
                .unwrap()
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(values.iter().all(|&value| value == 0.0));
        }
    }
}
