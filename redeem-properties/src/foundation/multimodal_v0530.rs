//! ReDeeM v0.53 v0.38-teacher-distilled mobility adapter on the deep v0.52 parent.
//!
//! v0.52 recovered RT and MS2 but its pair-aware mobility correction plateaued above the
//! dedicated v0.38 CCS reference.  v0.53 therefore freezes the v0.52-derived
//! `student_v050.*` + `student_v051.*` representation and replaces the mobility branch with a
//! new `student_v053.*` adapter.  The adapter projects the deep residue/pair representation into
//! the accepted v0.38 mobility feature space and predicts a native ion-mobility residual from
//! that aligned latent.  A frozen external v0.38 model supplies TRAIN-only representation and
//! prediction distillation targets; it is not part of the student checkpoint optimizer scope.

use super::featurize::FoundationBatch;
use super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
use super::model::PrecursorContextBatch;
use super::multimodal_v0270::FoundationFragmentContextBatchV0270;
use super::multimodal_v0360::{
    FoundationScalarPhysicsBatchV0360, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
};
use super::multimodal_v0500::{
    FoundationMultimodalForwardOutputV0500, FOUNDATION_V0500_TASK_COUNT,
};
use super::multimodal_v0510::{
    FoundationMultimodalForwardOutputV0510, PeptideFoundationV0510Config,
    PeptideFoundationV0510Model,
};
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0530: &str =
    "deep_pair_v038_teacher_distilled_mobility_v0530";
pub const FOUNDATION_V0530_STUDENT_NAMESPACE: &str = "student_v053";
pub const FOUNDATION_V0530_TEACHER_SOURCE: &str = "external_frozen_v0380";
pub const FOUNDATION_V0530_TEACHER_DIM: usize = 192;
pub const FOUNDATION_V0530_MOBILITY_LAYERS: usize = 2;
pub const FOUNDATION_V0530_MOBILITY_HEADS: usize = 6;
pub const FOUNDATION_V0530_MOBILITY_FF_DIM: usize = 768;
pub const FOUNDATION_V0530_MOBILITY_HIDDEN: usize = 384;
pub const FOUNDATION_V0530_MOBILITY_BOTTLENECK: usize = 192;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PeptideFoundationV0530Config {
    pub base_v0510: PeptideFoundationV0510Config,
    pub teacher_dim: usize,
    pub mobility_layers: usize,
    pub mobility_heads: usize,
    pub mobility_ff_dim: usize,
    pub mobility_hidden: usize,
    pub mobility_bottleneck: usize,
}

impl Default for PeptideFoundationV0530Config {
    fn default() -> Self {
        Self {
            base_v0510: PeptideFoundationV0510Config::default(),
            teacher_dim: FOUNDATION_V0530_TEACHER_DIM,
            mobility_layers: FOUNDATION_V0530_MOBILITY_LAYERS,
            mobility_heads: FOUNDATION_V0530_MOBILITY_HEADS,
            mobility_ff_dim: FOUNDATION_V0530_MOBILITY_FF_DIM,
            mobility_hidden: FOUNDATION_V0530_MOBILITY_HIDDEN,
            mobility_bottleneck: FOUNDATION_V0530_MOBILITY_BOTTLENECK,
        }
    }
}

impl PeptideFoundationV0530Config {
    pub fn fixed(base_v0510: PeptideFoundationV0510Config) -> Result<Self> {
        let config = Self {
            base_v0510,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn local_smoke() -> Self {
        Self {
            base_v0510: PeptideFoundationV0510Config::local_smoke(),
            teacher_dim: 32,
            mobility_layers: 1,
            mobility_heads: 4,
            mobility_ff_dim: 128,
            mobility_hidden: 96,
            mobility_bottleneck: 64,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0510.validate()?;
        if self.teacher_dim == 0 || self.mobility_hidden == 0 || self.mobility_bottleneck == 0 {
            candle_core::bail!("v0.53 teacher/mobile dimensions must be non-zero");
        }
        if self.mobility_layers == 0
            || self.mobility_heads == 0
            || self.mobility_ff_dim < self.teacher_dim
            || self.teacher_dim % self.mobility_heads != 0
        {
            candle_core::bail!(
                "v0.53 mobility transformer is incompatible with teacher_dim={} heads={} ff={}",
                self.teacher_dim,
                self.mobility_heads,
                self.mobility_ff_dim
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FoundationMobilityStudentOutputV0530 {
    /// Native raw ion-mobility residual relative to the frozen v0.35 CCS anchor.
    pub mobility_residual_native: Tensor,
    /// Context-conditioned student residue states in the v0.38 teacher width.
    pub residue_teacher_space: Tensor,
    /// Pooled student mobility state in the v0.38 teacher width.
    pub pooled_teacher_space: Tensor,
}

#[derive(Clone)]
struct MobilityTeacherBridgeV0530 {
    residue_projection: Linear,
    pair_projection: Linear,
    context_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    pooled_norm: FoundationLayerNorm,
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
    teacher_dim: usize,
    pair_dim: usize,
    max_sequence_len: usize,
}

impl MobilityTeacherBridgeV0530 {
    fn new(config: &PeptideFoundationV0530Config, vb: VarBuilder<'_>) -> Result<Self> {
        let student_width = config.base_v0510.base_v0500.residue_dim;
        let pair_dim = config.base_v0510.base_v0500.pair_dim;
        let mut blocks = Vec::with_capacity(config.mobility_layers);
        for layer in 0..config.mobility_layers {
            blocks.push(PeptideTransformerBlock::new(
                config.teacher_dim,
                config.mobility_heads,
                config.mobility_ff_dim,
                config.base_v0510.base_v0500.dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        let context_input = 2 * student_width + FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            residue_projection: nn::linear(
                student_width,
                config.teacher_dim,
                vb.pp("residue_projection"),
            )?,
            pair_projection: nn::linear(pair_dim, config.teacher_dim, vb.pp("pair_projection"))?,
            context_projection: nn::linear(
                context_input,
                config.teacher_dim,
                vb.pp("context_projection"),
            )?,
            input_norm: FoundationLayerNorm::new(config.teacher_dim, 1e-5, vb.pp("input_norm"))?,
            blocks,
            pooled_norm: FoundationLayerNorm::new(config.teacher_dim, 1e-5, vb.pp("pooled_norm"))?,
            hidden: nn::linear(
                3 * config.teacher_dim,
                config.mobility_hidden,
                vb.pp("hidden"),
            )?,
            bottleneck: nn::linear(
                config.mobility_hidden,
                config.mobility_bottleneck,
                vb.pp("bottleneck"),
            )?,
            output: zero_initialized_linear_v0530(config.mobility_bottleneck, 1, vb.pp("output"))?,
            teacher_dim: config.teacher_dim,
            pair_dim,
            max_sequence_len: config.base_v0510.base_v0500.max_sequence_len,
        })
    }

    fn forward_t(
        &self,
        base: &FoundationMultimodalForwardOutputV0500,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<FoundationMobilityStudentOutputV0530> {
        // The v0.52-derived parent is deliberately frozen.  Detaching at the adapter boundary
        // makes that contract explicit and keeps backward memory focused on student_v053.*.
        let residue = base
            .representation
            .residue_embeddings
            .detach()
            .contiguous()?;
        let residue_mask = base.representation.residue_mask.clone();
        let mobility_embedding = base
            .representation
            .mobility_embedding
            .detach()
            .contiguous()?;
        let global_embedding = base.representation.global_embedding.detach().contiguous()?;
        let pair_embeddings = base.representation.pair_embeddings.detach();
        let pair_mask = base.representation.pair_mask.clone();

        let (batch, sequence, _) = residue.dims3()?;
        if sequence != self.max_sequence_len {
            candle_core::bail!(
                "v0.53 residue length {sequence} differs from configured {}",
                self.max_sequence_len
            );
        }
        let context_input = Tensor::cat(
            &[&mobility_embedding, &global_embedding, &physics.ccs_physics],
            1,
        )?;
        let context = self.context_projection.forward(&context_input)?.relu()?;
        let context_residue =
            context
                .unsqueeze(1)?
                .broadcast_as((batch, sequence, self.teacher_dim))?;
        let mut hidden = self.residue_projection.forward(&residue)?;
        hidden = (&hidden + &context_residue)?;
        hidden = self.input_norm.forward(&hidden)?;
        let expanded_mask =
            residue_mask
                .unsqueeze(2)?
                .broadcast_as((batch, sequence, self.teacher_dim))?;
        hidden = hidden.broadcast_mul(&expanded_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &residue_mask, train)?;
        }
        let pooled = masked_mean_v0530(&hidden, &residue_mask)?;
        let pooled = self.pooled_norm.forward(&pooled)?;

        let pair_summary = masked_residue_pair_mean_v0530(
            &pair_embeddings,
            &pair_mask,
            self.max_sequence_len,
            self.pair_dim,
        )?;
        let pair_teacher_space = self.pair_projection.forward(&pair_summary)?.relu()?;
        let fused = Tensor::cat(&[&pooled, &pair_teacher_space, &context], 1)?;
        let hidden_head = self.hidden.forward(&fused)?.relu()?;
        let hidden_head = self.bottleneck.forward(&hidden_head)?.relu()?;
        let mobility_residual_native = self.output.forward(&hidden_head)?;
        Ok(FoundationMobilityStudentOutputV0530 {
            mobility_residual_native,
            residue_teacher_space: hidden,
            pooled_teacher_space: pooled,
        })
    }
}

#[derive(Clone)]
pub struct PeptideFoundationV0530Model {
    config: PeptideFoundationV0530Config,
    base_v0510: PeptideFoundationV0510Model,
    mobility_teacher_bridge: MobilityTeacherBridgeV0530,
}

impl PeptideFoundationV0530Model {
    pub fn new(config: PeptideFoundationV0530Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        // Preserve the exact v0.50/v0.51 namespaces.  A selected v0.52 checkpoint can then
        // warm-start all shared RT/MS2/backbone tensors by name while student_v052.* is ignored.
        let base_v0510 = PeptideFoundationV0510Model::new(config.base_v0510.clone(), vb.clone())?;
        let mobility_teacher_bridge = MobilityTeacherBridgeV0530::new(
            &config,
            vb.pp(FOUNDATION_V0530_STUDENT_NAMESPACE)
                .pp("mobility_teacher_bridge"),
        )?;
        Ok(Self {
            config,
            base_v0510,
            mobility_teacher_bridge,
        })
    }

    pub fn config(&self) -> &PeptideFoundationV0530Config {
        &self.config
    }

    /// Frozen property path used to prove RT/MS2 invariance against the selected v0.52 parent.
    pub fn property_forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        fragment: &FoundationFragmentContextBatchV0270,
    ) -> Result<FoundationMultimodalForwardOutputV0510> {
        self.base_v0510
            .property_forward_t(batch, context, physics, fragment, false)
    }

    /// Train/evaluate only the new mobility adapter.  The deep parent representation is always
    /// evaluated with train=false and detached before entering student_v053.*.
    pub fn mobility_student_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<FoundationMobilityStudentOutputV0530> {
        let base = self.base_v0510.base_v0500_t(batch, context, false)?;
        self.mobility_teacher_bridge
            .forward_t(&base, physics, train)
    }
}

fn masked_mean_v0530(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn masked_residue_pair_mean_v0530(
    pair: &Tensor,
    pair_mask: &Tensor,
    sequence: usize,
    pair_dim: usize,
) -> Result<Tensor> {
    let (batch, tokens_i, tokens_j, observed_pair_dim) = pair.dims4()?;
    if observed_pair_dim != pair_dim
        || tokens_i < FOUNDATION_V0500_TASK_COUNT + sequence
        || tokens_j < FOUNDATION_V0500_TASK_COUNT + sequence
    {
        candle_core::bail!("v0.53 pair tensor shape is incompatible with configured dimensions");
    }
    let residue_pair = pair
        .narrow(1, FOUNDATION_V0500_TASK_COUNT, sequence)?
        .narrow(2, FOUNDATION_V0500_TASK_COUNT, sequence)?;
    let residue_pair_mask = pair_mask
        .narrow(1, FOUNDATION_V0500_TASK_COUNT, sequence)?
        .narrow(2, FOUNDATION_V0500_TASK_COUNT, sequence)?;
    let expanded_mask = residue_pair_mask
        .unsqueeze(3)?
        .broadcast_as((batch, sequence, sequence, pair_dim))?;
    let summed = residue_pair.broadcast_mul(&expanded_mask)?.sum(1)?.sum(1)?;
    let denominator = residue_pair_mask
        .sum(1)?
        .sum(1)?
        .clamp(1.0, f64::INFINITY)?
        .unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn zero_initialized_linear_v0530(
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
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn v0530_namespace_preserves_parent_and_adds_only_new_mobility_adapter() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = PeptideFoundationV0530Model::new(PeptideFoundationV0530Config::local_smoke(), vb)
            .unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(data.keys().any(|name| name.starts_with("student_v050.")));
        assert!(data.keys().any(|name| name.starts_with("student_v051.")));
        assert!(data.keys().any(|name| name.starts_with("student_v053.")));
        assert!(!data.keys().any(|name| name.starts_with("student_v052.")));
    }

    #[test]
    fn v0530_mobility_output_is_zero_initialized() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = PeptideFoundationV0530Model::new(PeptideFoundationV0530Config::local_smoke(), vb)
            .unwrap();
        let data = varmap.data().lock().unwrap();
        let weight = data
            .get("student_v053.mobility_teacher_bridge.output.weight")
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
