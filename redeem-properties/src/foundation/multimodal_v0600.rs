//! ReDeeM v0.60 mobility-first conformational representation.
//!
//! v0.60 deliberately leaves the v0.37-v0.53 CCS specialist/adapter lane.  Instead of
//! distilling or freezing the accepted v0.38 representation, it trains a dedicated deep
//! chemistry/residue-pair backbone end-to-end from TRAIN mobility supervision.  Precursor charge
//! is represented by latent charge-carrier slots that interact with residue states, and a small
//! latent conformer set then summarizes multiple gas-phase conformational regimes before native
//! ion-mobility prediction.  The v0.38 model is a DEV benchmark only, never a teacher.

use super::featurize::FoundationBatch;
use super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
use super::model::PrecursorContextBatch;
use super::multimodal_v0360::{
    FoundationScalarPhysicsBatchV0360, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
};
use super::multimodal_v0500::{
    FoundationRepresentationV0500, PeptideFoundationV0500Config, PeptideFoundationV0500Model,
    FOUNDATION_V0500_TASK_COUNT,
};
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{self as nn, ops, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0600: &str =
    "mobility_first_charge_conformer_pair_representation_v0600";
pub const FOUNDATION_V0600_STUDENT_NAMESPACE: &str = "student_v060";
pub const FOUNDATION_V0600_CHARGE_SLOTS: usize = 6;
pub const FOUNDATION_V0600_CONFORMER_SLOTS: usize = 4;
pub const FOUNDATION_V0600_CHARGE_LAYERS: usize = 2;
pub const FOUNDATION_V0600_CONFORMER_LAYERS: usize = 2;
pub const FOUNDATION_V0600_HEAD_HIDDEN: usize = 512;
pub const FOUNDATION_V0600_HEAD_BOTTLENECK: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PeptideFoundationV0600Config {
    pub backbone: PeptideFoundationV0500Config,
    pub charge_slots: usize,
    pub conformer_slots: usize,
    pub charge_layers: usize,
    pub conformer_layers: usize,
    pub latent_heads: usize,
    pub latent_ff_dim: usize,
    pub head_hidden: usize,
    pub head_bottleneck: usize,
}

impl Default for PeptideFoundationV0600Config {
    fn default() -> Self {
        let backbone = PeptideFoundationV0500Config::default();
        Self {
            latent_heads: backbone.num_attention_heads,
            latent_ff_dim: backbone.feed_forward_dim,
            backbone,
            charge_slots: FOUNDATION_V0600_CHARGE_SLOTS,
            conformer_slots: FOUNDATION_V0600_CONFORMER_SLOTS,
            charge_layers: FOUNDATION_V0600_CHARGE_LAYERS,
            conformer_layers: FOUNDATION_V0600_CONFORMER_LAYERS,
            head_hidden: FOUNDATION_V0600_HEAD_HIDDEN,
            head_bottleneck: FOUNDATION_V0600_HEAD_BOTTLENECK,
        }
    }
}

impl PeptideFoundationV0600Config {
    pub fn fixed(backbone: PeptideFoundationV0500Config) -> Result<Self> {
        // v0.60 is mobility-only, but the established v0.50 chemistry/pair encoder is reused as a
        // trainable submodule. Keep its production widths/depth while allowing the prepared
        // sequence width to come from the authoritative corpus config.
        backbone.validate()?;
        let config = Self {
            latent_heads: backbone.num_attention_heads,
            latent_ff_dim: backbone.feed_forward_dim,
            backbone,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn local_smoke() -> Self {
        let backbone = PeptideFoundationV0500Config::local_smoke();
        Self {
            latent_heads: backbone.num_attention_heads,
            latent_ff_dim: backbone.feed_forward_dim,
            backbone,
            charge_slots: 4,
            conformer_slots: 3,
            charge_layers: 1,
            conformer_layers: 1,
            head_hidden: 96,
            head_bottleneck: 48,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.backbone.validate()?;
        if self.charge_slots == 0 || self.charge_slots > 8 {
            candle_core::bail!("v0.60 charge_slots must be in 1..=8");
        }
        if self.conformer_slots < 2 || self.conformer_slots > 8 {
            candle_core::bail!("v0.60 conformer_slots must be in 2..=8");
        }
        if self.charge_layers == 0 || self.conformer_layers == 0 {
            candle_core::bail!("v0.60 latent interaction layer counts must be positive");
        }
        if self.latent_heads == 0 || self.backbone.residue_dim % self.latent_heads != 0 {
            candle_core::bail!("v0.60 residue width must be divisible by latent_heads");
        }
        if self.latent_ff_dim < self.backbone.residue_dim {
            candle_core::bail!("v0.60 latent_ff_dim must be at least residue_dim");
        }
        if self.head_hidden == 0 || self.head_bottleneck == 0 {
            candle_core::bail!("v0.60 mobility head widths must be positive");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FoundationMobilityRepresentationV0600 {
    /// Absolute native ion-mobility prediction `[batch, 1]`.
    pub mobility_native: Tensor,
    /// Mobility-specific representation used by cross-source consistency supervision.
    pub mobility_latent: Tensor,
    /// Charge-slot states `[batch, charge_slots, residue_dim]`.
    pub charge_slot_embeddings: Tensor,
    /// Charge-slot validity mask `[batch, charge_slots]`.
    pub charge_slot_mask: Tensor,
    /// Latent conformer states `[batch, conformer_slots, residue_dim]`.
    pub conformer_embeddings: Tensor,
    /// Mixture weights over latent conformers `[batch, conformer_slots]`.
    pub conformer_weights: Tensor,
    /// Pair summary after projection into the residue width `[batch, residue_dim]`.
    pub pair_summary: Tensor,
}

#[derive(Clone)]
pub struct PeptideFoundationV0600Model {
    config: PeptideFoundationV0600Config,
    backbone: PeptideFoundationV0500Model,
    charge_slot_embedding: Embedding,
    charge_context_projection: Linear,
    pair_projection: Linear,
    charge_blocks: Vec<PeptideTransformerBlock>,
    charge_summary_norm: FoundationLayerNorm,
    conformer_embedding: Embedding,
    conformer_blocks: Vec<PeptideTransformerBlock>,
    conformer_norm: FoundationLayerNorm,
    conformer_score: Linear,
    latent_norm: FoundationLayerNorm,
    mobility_hidden: Linear,
    mobility_bottleneck: Linear,
    mobility_output: Linear,
}

impl PeptideFoundationV0600Model {
    pub fn new(config: PeptideFoundationV0600Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let student = vb.pp(FOUNDATION_V0600_STUDENT_NAMESPACE);
        // Passing a nested VarBuilder gives the reused deep-pair module an unambiguous v0.60
        // namespace: student_v060.mobility_backbone.student_v050.*. No historical v0.50 weights
        // are loaded; this backbone is optimized end-to-end by the mobility objective.
        let backbone = PeptideFoundationV0500Model::new(
            config.backbone.clone(),
            student.pp("mobility_backbone"),
        )?;
        let dim = config.backbone.residue_dim;
        let mut charge_blocks = Vec::with_capacity(config.charge_layers);
        for layer in 0..config.charge_layers {
            charge_blocks.push(PeptideTransformerBlock::new(
                dim,
                config.latent_heads,
                config.latent_ff_dim,
                config.backbone.dropout,
                student.pp(format!("charge.blocks.{layer}")),
            )?);
        }
        let mut conformer_blocks = Vec::with_capacity(config.conformer_layers);
        for layer in 0..config.conformer_layers {
            conformer_blocks.push(PeptideTransformerBlock::new(
                dim,
                config.latent_heads,
                config.latent_ff_dim,
                config.backbone.dropout,
                student.pp(format!("conformer.blocks.{layer}")),
            )?);
        }
        let mobility_head_input = 3 * dim + FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            backbone,
            charge_slot_embedding: nn::embedding(
                config.charge_slots,
                dim,
                student.pp("charge.slot_embedding"),
            )?,
            charge_context_projection: nn::linear(
                FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
                dim,
                student.pp("charge.context_projection"),
            )?,
            pair_projection: nn::linear(
                config.backbone.pair_dim,
                dim,
                student.pp("pair.summary_projection"),
            )?,
            charge_blocks,
            charge_summary_norm: FoundationLayerNorm::new(
                dim,
                1e-5,
                student.pp("charge.summary_norm"),
            )?,
            conformer_embedding: nn::embedding(
                config.conformer_slots,
                dim,
                student.pp("conformer.slot_embedding"),
            )?,
            conformer_blocks,
            conformer_norm: FoundationLayerNorm::new(
                dim,
                1e-5,
                student.pp("conformer.output_norm"),
            )?,
            conformer_score: nn::linear(dim, 1, student.pp("conformer.score"))?,
            latent_norm: FoundationLayerNorm::new(dim, 1e-5, student.pp("mobility.latent_norm"))?,
            mobility_hidden: nn::linear(
                mobility_head_input,
                config.head_hidden,
                student.pp("mobility.hidden"),
            )?,
            mobility_bottleneck: nn::linear(
                config.head_hidden,
                config.head_bottleneck,
                student.pp("mobility.bottleneck"),
            )?,
            // Ordinary trainable initialization preserves gradient flow into the representation
            // from the first update. The forward path below scales the raw output around a
            // neutral native-mobility prior of 1.0 without importing any historical teacher.
            mobility_output: nn::linear(config.head_bottleneck, 1, student.pp("mobility.output"))?,
            config,
        })
    }

    pub fn config(&self) -> &PeptideFoundationV0600Config {
        &self.config
    }

    pub fn forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<FoundationMobilityRepresentationV0600> {
        let base = self.backbone.representation_t(batch, context, train)?;
        self.mobility_from_base_t(&base, context, physics, train)
    }

    fn mobility_from_base_t(
        &self,
        base: &FoundationRepresentationV0500,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<FoundationMobilityRepresentationV0600> {
        let residues = &base.residue_embeddings;
        let residue_mask = &base.residue_mask;
        let (batch, sequence, dim) = residues.dims3()?;
        if dim != self.config.backbone.residue_dim {
            candle_core::bail!("v0.60 backbone residue width mismatch");
        }

        let pair_summary_raw = masked_residue_pair_mean_v0600(
            &base.pair_embeddings,
            &base.pair_mask,
            sequence,
            self.config.backbone.pair_dim,
        )?;
        let pair_summary = self.pair_projection.forward(&pair_summary_raw)?.relu()?;
        let physics_context = self
            .charge_context_projection
            .forward(&physics.ccs_physics)?
            .relu()?;

        let charge_ids = Tensor::arange(0u32, self.config.charge_slots as u32, residues.device())?
            .to_dtype(DType::U32)?;
        let charge_seed = self
            .charge_slot_embedding
            .forward(&charge_ids)?
            .unsqueeze(0)?
            .broadcast_as((batch, self.config.charge_slots, dim))?;
        let shared_charge_context = (&physics_context + &pair_summary)?
            .unsqueeze(1)?
            .broadcast_as((batch, self.config.charge_slots, dim))?;
        let charge_slots = (&charge_seed + &shared_charge_context)?;
        let charge_slot_mask = charge_slot_mask_v0600(context, self.config.charge_slots)?;
        let charge_mask_expanded =
            charge_slot_mask
                .unsqueeze(2)?
                .broadcast_as((batch, self.config.charge_slots, dim))?;
        let charge_slots = charge_slots.broadcast_mul(&charge_mask_expanded)?;

        let residue_context = physics_context
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, dim))?;
        let conditioned_residues = (residues + residue_context)?;
        let mut charge_tokens =
            Tensor::cat(&[&charge_slots, &conditioned_residues], 1)?.contiguous()?;
        let charge_token_mask = Tensor::cat(&[&charge_slot_mask, residue_mask], 1)?;
        for block in &self.charge_blocks {
            charge_tokens = block.forward_t(&charge_tokens, &charge_token_mask, train)?;
        }
        let charge_slot_embeddings = charge_tokens
            .narrow(1, 0, self.config.charge_slots)?
            .contiguous()?;
        let charge_conditioned_residues = charge_tokens
            .narrow(1, self.config.charge_slots, sequence)?
            .contiguous()?;
        let charge_summary = masked_mean_v0600(&charge_slot_embeddings, &charge_slot_mask)?;
        let charge_summary = self.charge_summary_norm.forward(&charge_summary)?;

        let conformer_ids =
            Tensor::arange(0u32, self.config.conformer_slots as u32, residues.device())?
                .to_dtype(DType::U32)?;
        let conformer_seed = self
            .conformer_embedding
            .forward(&conformer_ids)?
            .unsqueeze(0)?
            .broadcast_as((batch, self.config.conformer_slots, dim))?;
        let conformer_context = (&charge_summary + &pair_summary)?
            .unsqueeze(1)?
            .broadcast_as((batch, self.config.conformer_slots, dim))?;
        let conformer_slots = (&conformer_seed + &conformer_context)?;
        let conformer_mask = Tensor::ones(
            (batch, self.config.conformer_slots),
            DType::F32,
            residues.device(),
        )?;
        let mut conformer_tokens =
            Tensor::cat(&[&conformer_slots, &charge_conditioned_residues], 1)?.contiguous()?;
        let conformer_token_mask = Tensor::cat(&[&conformer_mask, residue_mask], 1)?;
        for block in &self.conformer_blocks {
            conformer_tokens = block.forward_t(&conformer_tokens, &conformer_token_mask, train)?;
        }
        let conformer_embeddings = conformer_tokens
            .narrow(1, 0, self.config.conformer_slots)?
            .contiguous()?;
        let conformer_embeddings = self.conformer_norm.forward(&conformer_embeddings)?;
        let flat_conformers = conformer_embeddings
            .reshape((batch * self.config.conformer_slots, dim))?
            .contiguous()?;
        let conformer_scores = self
            .conformer_score
            .forward(&flat_conformers)?
            .reshape((batch, self.config.conformer_slots))?;
        let conformer_weights = ops::softmax(&conformer_scores, 1)?;
        let ensemble = conformer_embeddings
            .broadcast_mul(&conformer_weights.unsqueeze(2)?.broadcast_as((
                batch,
                self.config.conformer_slots,
                dim,
            ))?)?
            .sum(1)?;
        let mobility_latent = self.latent_norm.forward(&ensemble)?;
        let head_input = Tensor::cat(
            &[
                &mobility_latent,
                &charge_summary,
                &pair_summary,
                &physics.ccs_physics,
            ],
            1,
        )?
        .contiguous()?;
        let hidden = self.mobility_hidden.forward(&head_input)?.relu()?;
        let bottleneck = self.mobility_bottleneck.forward(&hidden)?.relu()?;
        let mobility_native = self
            .mobility_output
            .forward(&bottleneck)?
            .affine(0.05, 1.0)?;

        Ok(FoundationMobilityRepresentationV0600 {
            mobility_native,
            mobility_latent,
            charge_slot_embeddings,
            charge_slot_mask,
            conformer_embeddings,
            conformer_weights,
            pair_summary,
        })
    }
}

fn charge_slot_mask_v0600(context: &PrecursorContextBatch, slots: usize) -> Result<Tensor> {
    let charge = context.charge.to_vec1::<f32>()?;
    let present = context.charge_present.to_vec1::<f32>()?;
    if charge.len() != present.len() {
        candle_core::bail!("v0.60 charge/presence shape mismatch");
    }
    let mut mask = Vec::with_capacity(charge.len() * slots);
    for (&value, &is_present) in charge.iter().zip(&present) {
        let active = if is_present > 0.0 && value.is_finite() {
            (value.round() as isize).clamp(1, slots as isize) as usize
        } else {
            1usize
        };
        for slot in 0..slots {
            mask.push((slot < active) as u8 as f32);
        }
    }
    Tensor::from_vec(mask, (charge.len(), slots), context.charge.device())
}

fn masked_mean_v0600(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn masked_residue_pair_mean_v0600(
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
        candle_core::bail!("v0.60 pair tensor shape is incompatible with configured dimensions");
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn v0600_namespace_is_dedicated_and_contains_trainable_deep_pair_backbone() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = PeptideFoundationV0600Model::new(PeptideFoundationV0600Config::local_smoke(), vb)
            .unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(
            data.keys()
                .any(|name| name
                    .starts_with("student_v060.mobility_backbone.student_v050.chemistry."))
        );
        assert!(data
            .keys()
            .any(|name| name.starts_with("student_v060.charge.slot_embedding")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("student_v060.conformer.slot_embedding")));
        assert!(data
            .keys()
            .any(|name| name == "student_v060.mobility.output.weight"));
        assert!(!data
            .keys()
            .any(|name| name.starts_with("ccs_context_v0380.")));
        assert!(!data.keys().any(|name| name.starts_with("student_v053.")));
    }

    #[test]
    fn v0600_config_keeps_multiple_charge_and_conformer_latents() {
        let config = PeptideFoundationV0600Config::local_smoke();
        assert!(config.charge_slots >= 2);
        assert!(config.conformer_slots >= 2);
        config.validate().unwrap();
    }
}
