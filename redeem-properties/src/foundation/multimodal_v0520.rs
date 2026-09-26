//! ReDeeM v0.52 mobility-aware pair representation on top of the completed v0.51 model.
//!
//! Source review before this iteration showed that v0.50 already injects precursor charge,
//! m/z, and a neutral-mass proxy into the mobility task token before all eight deep pair
//! interaction blocks. v0.52 therefore does not duplicate that path. Instead it addresses the
//! remaining representation gap exposed by v0.51: the mobility specialist did not consume the
//! learned residue-pair state directly.
//!
//! v0.52 keeps the selected `student_v050.*` backbone and successful `student_v051.*` RT/MS2
//! specialists, freezes the selected v0.51 mobility residual as a detached step-0 baseline, and
//! adds a `student_v052.*` mobility branch that explicitly integrates:
//!
//! - mobility-task -> residue and residue -> mobility pair states;
//! - residue-residue pair summaries;
//! - charge/mass/mz physics gates over the pair representation;
//! - a small mobility-specific residue Transformer refinement;
//! - TRAIN-only coarse physicochemical/conformation proxy prediction.
//!
//! The new native-mobility correction is zero initialized, so step 0 reproduces the selected
//! v0.51 mobility prediction exactly. Mobility gradients are forced through the v0.52 pair-aware
//! branch rather than through the old v0.51 mobility residual.

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
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{self as nn, ops, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520: &str =
    "deep_pair_mobility_aware_representation_v0520";
pub const FOUNDATION_V0520_STUDENT_NAMESPACE: &str = "student_v052";
pub const FOUNDATION_V0520_MOBILITY_PAIR_LAYERS: usize = 2;
pub const FOUNDATION_V0520_MOBILITY_PAIR_HEADS: usize = 8;
pub const FOUNDATION_V0520_MOBILITY_PAIR_FF_DIM: usize = 1280;
pub const FOUNDATION_V0520_SPECIALIST_HIDDEN: usize = 640;
pub const FOUNDATION_V0520_SPECIALIST_BOTTLENECK: usize = 320;
pub const FOUNDATION_V0520_CONFORMATION_PROXY_DIM: usize = 14;

const TASK_MOBILITY_V0520: usize = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PeptideFoundationV0520Config {
    pub base_v0510: PeptideFoundationV0510Config,
    pub mobility_pair_layers: usize,
    pub mobility_pair_heads: usize,
    pub mobility_pair_ff_dim: usize,
    pub specialist_hidden: usize,
    pub specialist_bottleneck: usize,
    pub conformation_proxy_dim: usize,
}

impl Default for PeptideFoundationV0520Config {
    fn default() -> Self {
        Self {
            base_v0510: PeptideFoundationV0510Config::default(),
            mobility_pair_layers: FOUNDATION_V0520_MOBILITY_PAIR_LAYERS,
            mobility_pair_heads: FOUNDATION_V0520_MOBILITY_PAIR_HEADS,
            mobility_pair_ff_dim: FOUNDATION_V0520_MOBILITY_PAIR_FF_DIM,
            specialist_hidden: FOUNDATION_V0520_SPECIALIST_HIDDEN,
            specialist_bottleneck: FOUNDATION_V0520_SPECIALIST_BOTTLENECK,
            conformation_proxy_dim: FOUNDATION_V0520_CONFORMATION_PROXY_DIM,
        }
    }
}

impl PeptideFoundationV0520Config {
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
            mobility_pair_layers: 1,
            mobility_pair_heads: 4,
            mobility_pair_ff_dim: 128,
            specialist_hidden: 96,
            specialist_bottleneck: 64,
            conformation_proxy_dim: FOUNDATION_V0520_CONFORMATION_PROXY_DIM,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0510.validate()?;
        let width = self.base_v0510.base_v0500.residue_dim;
        if self.mobility_pair_layers == 0
            || self.mobility_pair_heads == 0
            || self.mobility_pair_ff_dim < width
            || width % self.mobility_pair_heads != 0
        {
            candle_core::bail!(
                "v0.52 mobility pair refinement is incompatible with residue width {width}"
            );
        }
        if self.specialist_hidden == 0 || self.specialist_bottleneck == 0 {
            candle_core::bail!("v0.52 specialist dimensions must be non-zero");
        }
        if self.conformation_proxy_dim != FOUNDATION_V0520_CONFORMATION_PROXY_DIM {
            candle_core::bail!(
                "v0.52 conformation proxy dimension must remain {}",
                FOUNDATION_V0520_CONFORMATION_PROXY_DIM
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FoundationMobilityAwareOutputV0520 {
    pub base_v0500: FoundationMultimodalForwardOutputV0500,
    /// Selected v0.51 native mobility residual, detached and therefore immutable to v0.52
    /// mobility gradients.
    pub baseline_v0510_residual_native: Tensor,
    /// Zero-initialized v0.52 pair-aware correction.
    pub mobility_correction_native: Tensor,
    /// Detached v0.51 residual + v0.52 correction.
    pub mobility_residual_native: Tensor,
    /// TRAIN-only physicochemical/conformation proxy prediction `[batch, 14]`.
    pub conformation_proxy: Tensor,
}

#[derive(Clone)]
struct MobilityAwarePairRefinementV0520 {
    physics_projection: Linear,
    physics_pair_gate: Linear,
    mobility_to_residue_projection: Linear,
    residue_to_mobility_projection: Linear,
    residue_pair_projection: Linear,
    pair_global_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
    conformation_proxy: Linear,
    residue_dim: usize,
    pair_dim: usize,
    max_sequence_len: usize,
}

impl MobilityAwarePairRefinementV0520 {
    fn new(config: &PeptideFoundationV0520Config, vb: VarBuilder<'_>) -> Result<Self> {
        let residue_dim = config.base_v0510.base_v0500.residue_dim;
        let pair_dim = config.base_v0510.base_v0500.pair_dim;
        let mut blocks = Vec::with_capacity(config.mobility_pair_layers);
        for layer in 0..config.mobility_pair_layers {
            blocks.push(PeptideTransformerBlock::new(
                residue_dim,
                config.mobility_pair_heads,
                config.mobility_pair_ff_dim,
                config.base_v0510.base_v0500.dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        let head_input = 4 * residue_dim + FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            physics_projection: nn::linear(
                FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
                residue_dim,
                vb.pp("physics_projection"),
            )?,
            physics_pair_gate: nn::linear(
                FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
                pair_dim,
                vb.pp("physics_pair_gate"),
            )?,
            mobility_to_residue_projection: nn::linear(
                pair_dim,
                residue_dim,
                vb.pp("mobility_to_residue_projection"),
            )?,
            residue_to_mobility_projection: nn::linear(
                pair_dim,
                residue_dim,
                vb.pp("residue_to_mobility_projection"),
            )?,
            residue_pair_projection: nn::linear(
                pair_dim,
                residue_dim,
                vb.pp("residue_pair_projection"),
            )?,
            pair_global_projection: nn::linear(
                pair_dim,
                residue_dim,
                vb.pp("pair_global_projection"),
            )?,
            input_norm: FoundationLayerNorm::new(residue_dim, 1e-5, vb.pp("input_norm"))?,
            blocks,
            hidden: nn::linear(head_input, config.specialist_hidden, vb.pp("hidden"))?,
            bottleneck: nn::linear(
                config.specialist_hidden,
                config.specialist_bottleneck,
                vb.pp("bottleneck"),
            )?,
            output: zero_initialized_linear_v0520(
                config.specialist_bottleneck,
                1,
                vb.pp("output"),
            )?,
            conformation_proxy: nn::linear(
                residue_dim,
                config.conformation_proxy_dim,
                vb.pp("conformation_proxy"),
            )?,
            residue_dim,
            pair_dim,
            max_sequence_len: config.base_v0510.base_v0500.max_sequence_len,
        })
    }

    fn forward_t(
        &self,
        base: &FoundationMultimodalForwardOutputV0500,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let residues = &base.representation.residue_embeddings;
        let mask = &base.representation.residue_mask;
        let pair = &base.representation.pair_embeddings;
        let (batch, sequence, residue_dim) = residues.dims3()?;
        let (pair_batch, pair_left, pair_right, pair_dim) = pair.dims4()?;
        let expected_tokens = FOUNDATION_V0500_TASK_COUNT + sequence;
        if sequence != self.max_sequence_len
            || residue_dim != self.residue_dim
            || pair_batch != batch
            || pair_left != expected_tokens
            || pair_right != expected_tokens
            || pair_dim != self.pair_dim
        {
            candle_core::bail!("v0.52 mobility-aware pair representation shape mismatch");
        }

        let mobility_to_residue = pair
            .narrow(1, TASK_MOBILITY_V0520, 1)?
            .squeeze(1)?
            .narrow(1, FOUNDATION_V0500_TASK_COUNT, sequence)?
            .contiguous()?;
        let residue_to_mobility = pair
            .narrow(1, FOUNDATION_V0500_TASK_COUNT, sequence)?
            .narrow(2, TASK_MOBILITY_V0520, 1)?
            .squeeze(2)?
            .contiguous()?;
        let residue_pair = pair
            .narrow(1, FOUNDATION_V0500_TASK_COUNT, sequence)?
            .narrow(2, FOUNDATION_V0500_TASK_COUNT, sequence)?
            .contiguous()?;

        // Average each residue's learned pair environment across valid partner residues.
        let pair_sum = residue_pair.sum(2)?;
        let partner_count = mask
            .sum(1)?
            .clamp(1.0, f64::INFINITY)?
            .unsqueeze(1)?
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, 1))?;
        let pair_summary = pair_sum.broadcast_div(&partner_count)?;
        let pair_summary = pair_summary.broadcast_mul(&mask.unsqueeze(2)?.broadcast_as((
            batch,
            sequence,
            self.pair_dim,
        ))?)?;

        // Charge/mass/mz context gates pair-state channels directly. This is distinct from the
        // existing v0.50 mobility task-token context and gives the mobility branch an explicit
        // physics-conditioned view of residue-pair compatibility.
        let pair_gate = ops::sigmoid(&self.physics_pair_gate.forward(&physics.ccs_physics)?)?;
        let pair_gate = pair_gate
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, self.pair_dim))?;
        let gated_pair_summary = pair_summary.broadcast_mul(&pair_gate)?;
        let gated_mobility_to_residue = mobility_to_residue.broadcast_mul(&pair_gate)?;
        let gated_residue_to_mobility = residue_to_mobility.broadcast_mul(&pair_gate)?;

        let pair_residue = self
            .residue_pair_projection
            .forward(&gated_pair_summary.contiguous()?)?;
        let mobility_out = self
            .mobility_to_residue_projection
            .forward(&gated_mobility_to_residue.contiguous()?)?;
        let mobility_in = self
            .residue_to_mobility_projection
            .forward(&gated_residue_to_mobility.contiguous()?)?;
        let physics_context = self.physics_projection.forward(&physics.ccs_physics)?;
        let physics_residue =
            physics_context
                .unsqueeze(1)?
                .broadcast_as((batch, sequence, self.residue_dim))?;

        let mut hidden = (((residues + &pair_residue)? + &mobility_out)? + &mobility_in)?;
        hidden = (hidden + physics_residue)?;
        hidden = self.input_norm.forward(&hidden)?;
        hidden = hidden.broadcast_mul(&mask.unsqueeze(2)?.broadcast_as((
            batch,
            sequence,
            self.residue_dim,
        ))?)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, mask, train)?;
        }

        let pooled = masked_mean_v0520(&hidden, mask)?;
        let pair_global = masked_mean_v0520(&gated_pair_summary, mask)?;
        let pair_global = self.pair_global_projection.forward(&pair_global)?;
        let features = Tensor::cat(
            &[
                &base.representation.mobility_embedding,
                &base.representation.global_embedding,
                &pooled,
                &pair_global,
                &physics.ccs_physics,
            ],
            1,
        )?;
        let head = self.hidden.forward(&features)?.relu()?;
        let head = self.bottleneck.forward(&head)?.relu()?;
        let correction = self.output.forward(&head)?;
        let conformation_proxy = self.conformation_proxy.forward(&pooled)?;
        Ok((correction, conformation_proxy))
    }
}

#[derive(Clone)]
pub struct PeptideFoundationV0520Model {
    config: PeptideFoundationV0520Config,
    base_v0510: PeptideFoundationV0510Model,
    mobility_pair_refinement: MobilityAwarePairRefinementV0520,
}

impl PeptideFoundationV0520Model {
    pub fn new(config: PeptideFoundationV0520Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let base_v0510 = PeptideFoundationV0510Model::new(config.base_v0510.clone(), vb.clone())?;
        let mobility_pair_refinement = MobilityAwarePairRefinementV0520::new(
            &config,
            vb.pp(FOUNDATION_V0520_STUDENT_NAMESPACE)
                .pp("mobility_pair_refinement"),
        )?;
        Ok(Self {
            config,
            base_v0510,
            mobility_pair_refinement,
        })
    }

    pub fn config(&self) -> &PeptideFoundationV0520Config {
        &self.config
    }

    pub fn property_forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0510> {
        self.base_v0510
            .property_forward_t(batch, context, physics, fragment, train)
    }

    pub fn base_v0500_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0500> {
        self.base_v0510.base_v0500_t(batch, context, train)
    }

    pub fn mobility_aware_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<FoundationMobilityAwareOutputV0520> {
        let (base_v0500, baseline_v0510_residual_native) = self
            .base_v0510
            .mobility_components_t(batch, context, physics, train)?;
        let baseline_v0510_residual_native = baseline_v0510_residual_native.detach();
        let (mobility_correction_native, conformation_proxy) = self
            .mobility_pair_refinement
            .forward_t(&base_v0500, physics, train)?;
        let mobility_residual_native =
            (&baseline_v0510_residual_native + &mobility_correction_native)?;
        Ok(FoundationMobilityAwareOutputV0520 {
            base_v0500,
            baseline_v0510_residual_native,
            mobility_correction_native,
            mobility_residual_native,
            conformation_proxy,
        })
    }
}

fn masked_mean_v0520(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn zero_initialized_linear_v0520(
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
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn v0520_fixed_config_preserves_completed_v0510_contract() {
        let cfg = PeptideFoundationV0520Config::default();
        cfg.validate().unwrap();
        assert_eq!(cfg.base_v0510.base_v0500.residue_dim, 320);
        assert_eq!(cfg.base_v0510.base_v0500.pair_dim, 128);
        assert_eq!(cfg.base_v0510.base_v0500.interaction_blocks, 8);
        assert_eq!(cfg.mobility_pair_layers, 2);
        assert_eq!(cfg.conformation_proxy_dim, 14);
    }

    #[test]
    fn v0520_namespace_adds_zero_initialized_pair_aware_correction() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model =
            PeptideFoundationV0520Model::new(PeptideFoundationV0520Config::local_smoke(), vb)
                .unwrap();
        assert_eq!(model.config().base_v0510.base_v0500.residue_dim, 64);
        let data = varmap.data().lock().unwrap();
        assert!(data.keys().any(|name| name.starts_with("student_v050.")));
        assert!(data.keys().any(|name| name.starts_with("student_v051.")));
        assert!(data.keys().any(|name| name.starts_with("student_v052.")));
        let output = data
            .get("student_v052.mobility_pair_refinement.output.weight")
            .unwrap()
            .as_tensor()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(output, 0.0);
        assert!(data
            .get("student_v052.mobility_pair_refinement.physics_pair_gate.weight")
            .is_some());
        assert!(data
            .get("student_v052.mobility_pair_refinement.conformation_proxy.weight")
            .is_some());
    }
}
