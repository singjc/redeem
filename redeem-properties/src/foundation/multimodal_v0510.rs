//! ReDeeM v0.51 teacher-bridged specialist continuation of the v0.50 deep pair model.
//!
//! v0.50 established that the 320d chemistry/residue-pair backbone is mechanically
//! trainable, but its generic RT/MS2/mobility heads did not reach the promoted v0.35
//! and v0.38 DEV references. v0.51 therefore keeps the learned `student_v050.*`
//! backbone intact and adds task-specific `student_v051.*` modules:
//!
//! - learned 320 -> teacher-space projections for feature-level v0.35 distillation;
//! - a zero-initialized RT residual specialist;
//! - an identity-initialized v0.35-style MS2 context conditioner followed by a
//!   fragment-token Transformer with factorized presence/intensity outputs;
//! - a v0.38-style native ion-mobility residual head driven by peptide physics.
//!
//! The authoritative v0.35 model remains external and frozen. Exact Bruker
//! mobility->CCS conversion and the v0.35 CCS baseline are applied by the trainer,
//! so the mobility residual is zero at initialization and cannot silently redefine
//! the physics contract.

use super::config::FoundationMs2OutputActivation;
use super::featurize::FoundationBatch;
use super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
use super::model::{apply_ms2_output_activation, PrecursorContextBatch};
use super::multimodal_v0270::{
    FoundationFragmentContextBatchV0270, FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270,
};
use super::multimodal_v0350::PeptideFoundationMultimodalV0350Model;
use super::multimodal_v0360::{
    FoundationScalarPhysicsBatchV0360, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
    FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360,
};
use super::multimodal_v0500::{
    FoundationMultimodalForwardOutputV0500, PeptideFoundationV0500Config,
    PeptideFoundationV0500Model,
};
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{self as nn, ops, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0510: &str =
    "deep_pair_teacher_bridged_specialists_v0510";
pub const FOUNDATION_V0510_STUDENT_NAMESPACE: &str = "student_v051";
pub const FOUNDATION_V0510_TEACHER_SOURCE: &str = "external_frozen_v0350";
pub const FOUNDATION_V0510_TEACHER_DIM: usize = 192;
pub const FOUNDATION_V0510_MS2_CONTEXT_LAYERS: usize = 3;
pub const FOUNDATION_V0510_MS2_CONTEXT_HEADS: usize = 8;
pub const FOUNDATION_V0510_MS2_CONTEXT_FF_DIM: usize = 1280;
pub const FOUNDATION_V0510_FRAGMENT_LAYERS: usize = 2;
pub const FOUNDATION_V0510_FRAGMENT_HEADS: usize = 8;
pub const FOUNDATION_V0510_FRAGMENT_FF_DIM: usize = 1280;
pub const FOUNDATION_V0510_MOBILITY_LAYERS: usize = 2;
pub const FOUNDATION_V0510_MOBILITY_HEADS: usize = 8;
pub const FOUNDATION_V0510_MOBILITY_FF_DIM: usize = 1280;
pub const FOUNDATION_V0510_SPECIALIST_HIDDEN: usize = 640;
pub const FOUNDATION_V0510_SPECIALIST_BOTTLENECK: usize = 320;

const MS2_CONTEXT_SCALARS_V0510: usize = 5;
const ION_SERIES_CLASSES_V0510: usize = 6;
const FRAGMENT_CHARGE_CLASSES_V0510: usize = 3;
const PRECURSOR_CHARGE_CLASSES_V0510: usize = 7;
const ION_SERIES_EMBED_DIM_V0510: usize = 24;
const FRAGMENT_CHARGE_EMBED_DIM_V0510: usize = 8;
const PRECURSOR_CHARGE_EMBED_DIM_V0510: usize = 8;
const INSTRUMENT_EMBED_DIM_V0510: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PeptideFoundationV0510Config {
    pub base_v0500: PeptideFoundationV0500Config,
    pub teacher_dim: usize,
    pub ms2_context_layers: usize,
    pub ms2_context_heads: usize,
    pub ms2_context_ff_dim: usize,
    pub fragment_layers: usize,
    pub fragment_heads: usize,
    pub fragment_ff_dim: usize,
    pub mobility_layers: usize,
    pub mobility_heads: usize,
    pub mobility_ff_dim: usize,
    pub specialist_hidden: usize,
    pub specialist_bottleneck: usize,
}

impl Default for PeptideFoundationV0510Config {
    fn default() -> Self {
        Self {
            base_v0500: PeptideFoundationV0500Config::default(),
            teacher_dim: FOUNDATION_V0510_TEACHER_DIM,
            ms2_context_layers: FOUNDATION_V0510_MS2_CONTEXT_LAYERS,
            ms2_context_heads: FOUNDATION_V0510_MS2_CONTEXT_HEADS,
            ms2_context_ff_dim: FOUNDATION_V0510_MS2_CONTEXT_FF_DIM,
            fragment_layers: FOUNDATION_V0510_FRAGMENT_LAYERS,
            fragment_heads: FOUNDATION_V0510_FRAGMENT_HEADS,
            fragment_ff_dim: FOUNDATION_V0510_FRAGMENT_FF_DIM,
            mobility_layers: FOUNDATION_V0510_MOBILITY_LAYERS,
            mobility_heads: FOUNDATION_V0510_MOBILITY_HEADS,
            mobility_ff_dim: FOUNDATION_V0510_MOBILITY_FF_DIM,
            specialist_hidden: FOUNDATION_V0510_SPECIALIST_HIDDEN,
            specialist_bottleneck: FOUNDATION_V0510_SPECIALIST_BOTTLENECK,
        }
    }
}

impl PeptideFoundationV0510Config {
    pub fn fixed(base_v0500: PeptideFoundationV0500Config) -> Result<Self> {
        let config = Self {
            base_v0500,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn local_smoke() -> Self {
        let base = PeptideFoundationV0500Config::local_smoke();
        Self {
            teacher_dim: 32,
            ms2_context_layers: 1,
            ms2_context_heads: 4,
            ms2_context_ff_dim: 128,
            fragment_layers: 1,
            fragment_heads: 4,
            fragment_ff_dim: 128,
            mobility_layers: 1,
            mobility_heads: 4,
            mobility_ff_dim: 128,
            specialist_hidden: 96,
            specialist_bottleneck: 64,
            base_v0500: base,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0500.validate()?;
        let width = self.base_v0500.residue_dim;
        if self.teacher_dim == 0 || self.specialist_hidden == 0 || self.specialist_bottleneck == 0 {
            candle_core::bail!("v0.51 projection/specialist dimensions must be non-zero");
        }
        for (label, layers, heads, ff) in [
            (
                "ms2_context",
                self.ms2_context_layers,
                self.ms2_context_heads,
                self.ms2_context_ff_dim,
            ),
            (
                "fragment",
                self.fragment_layers,
                self.fragment_heads,
                self.fragment_ff_dim,
            ),
            (
                "mobility",
                self.mobility_layers,
                self.mobility_heads,
                self.mobility_ff_dim,
            ),
        ] {
            if layers == 0 || heads == 0 || ff < width || width % heads != 0 {
                candle_core::bail!(
                    "v0.51 {label} configuration is incompatible with residue width {width}"
                );
            }
        }
        if self.base_v0500.ms2_fragment_channels != 8 {
            candle_core::bail!("v0.51 requires the established 8-channel MS2 layout");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FoundationTeacherBridgeOutputV0510 {
    pub peptide_projection: Tensor,
    pub residue_projection: Tensor,
}

#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0510 {
    pub base_v0500: FoundationMultimodalForwardOutputV0500,
    pub rt: Tensor,
    /// Native raw ion-mobility residual. The trainer adds this to mobility implied
    /// by the frozen v0.35 CCS anchor before exact CCS conversion.
    pub mobility_residual_native: Tensor,
    pub ms2: Tensor,
    pub ms2_presence_logits: Tensor,
    pub ms2_positive_intensity: Tensor,
    pub teacher_bridge: FoundationTeacherBridgeOutputV0510,
}

#[derive(Clone)]
struct RtResidualHeadV0510 {
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
}

impl RtResidualHeadV0510 {
    fn new(config: &PeptideFoundationV0510Config, vb: VarBuilder<'_>) -> Result<Self> {
        let width = config.base_v0500.residue_dim;
        let input = 2 * width + FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            hidden: nn::linear(input, config.specialist_hidden, vb.pp("hidden"))?,
            bottleneck: nn::linear(
                config.specialist_hidden,
                config.specialist_bottleneck,
                vb.pp("bottleneck"),
            )?,
            output: zero_initialized_linear_v0510(
                config.specialist_bottleneck,
                1,
                vb.pp("output"),
            )?,
        })
    }

    fn forward(
        &self,
        base: &FoundationMultimodalForwardOutputV0500,
        physics: &FoundationScalarPhysicsBatchV0360,
    ) -> Result<Tensor> {
        let features = Tensor::cat(
            &[
                &base.representation.rt_embedding,
                &base.representation.global_embedding,
                &physics.rt_intrinsic,
            ],
            1,
        )?;
        let hidden = self.hidden.forward(&features)?.relu()?;
        let hidden = self.bottleneck.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)
    }
}

#[derive(Clone)]
struct Ms2ContextConditionerV0510 {
    instrument_embedding: Embedding,
    context_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    delta_output: Linear,
    width: usize,
    instrument_dim: usize,
}

impl Ms2ContextConditionerV0510 {
    fn new(config: &PeptideFoundationV0510Config, vb: VarBuilder<'_>) -> Result<Self> {
        let width = config.base_v0500.residue_dim;
        let instrument_dim = config.base_v0500.instrument_dim;
        let mut blocks = Vec::with_capacity(config.ms2_context_layers);
        for layer in 0..config.ms2_context_layers {
            blocks.push(PeptideTransformerBlock::new(
                width,
                config.ms2_context_heads,
                config.ms2_context_ff_dim,
                config.base_v0500.dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        Ok(Self {
            instrument_embedding: nn::embedding(
                config.base_v0500.instrument_vocab_size,
                instrument_dim,
                vb.pp("instrument_embedding"),
            )?,
            context_projection: nn::linear(
                instrument_dim + MS2_CONTEXT_SCALARS_V0510,
                width,
                vb.pp("context_projection"),
            )?,
            input_norm: FoundationLayerNorm::new(width, 1e-5, vb.pp("input_norm"))?,
            blocks,
            delta_output: zero_initialized_linear_v0510(width, width, vb.pp("delta_output"))?,
            width,
            instrument_dim,
        })
    }

    fn forward_t(
        &self,
        residues: &Tensor,
        residue_mask: &Tensor,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<Tensor> {
        let (batch, sequence, width) = residues.dims3()?;
        if width != self.width {
            candle_core::bail!("v0.51 MS2 context conditioner received incompatible width");
        }
        let instrument = self.instrument_embedding.forward(&context.instrument_ids)?;
        if instrument.dims2()? != (batch, self.instrument_dim) {
            candle_core::bail!("v0.51 MS2 instrument embedding shape mismatch");
        }
        let charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let charge_present = context.charge_present.unsqueeze(1)?;
        let nce = context.nce.affine(1.0 / 100.0, 0.0)?.unsqueeze(1)?;
        let nce_present = context.nce_present.unsqueeze(1)?;
        let instrument_present = context.instrument_present.unsqueeze(1)?;
        let scalars = Tensor::cat(
            &[
                &charge,
                &charge_present,
                &nce,
                &nce_present,
                &instrument_present,
            ],
            1,
        )?;
        let context_features = Tensor::cat(&[&instrument, &scalars], 1)?;
        let context_embedding = self
            .context_projection
            .forward(&context_features)?
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, width))?;
        let input = (residues + &context_embedding)?;
        let mut hidden = self.input_norm.forward(&input)?;
        let expanded_mask = residue_mask
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, width))?;
        hidden = hidden.broadcast_mul(&expanded_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, residue_mask, train)?;
        }
        let delta = self
            .delta_output
            .forward(&hidden)?
            .broadcast_mul(&expanded_mask)?;
        residues + &delta
    }
}

#[derive(Clone)]
struct ContextualFragmentDecoderV0510 {
    ion_series_embedding: Embedding,
    fragment_charge_embedding: Embedding,
    precursor_charge_embedding: Embedding,
    instrument_embedding: Embedding,
    feature_norm: FoundationLayerNorm,
    token_projection: Linear,
    blocks: Vec<PeptideTransformerBlock>,
    output_norm: FoundationLayerNorm,
    presence_head: Linear,
    intensity_head: Linear,
    blend_gate: Linear,
    width: usize,
}

impl ContextualFragmentDecoderV0510 {
    fn new(config: &PeptideFoundationV0510Config, vb: VarBuilder<'_>) -> Result<Self> {
        let width = config.base_v0500.residue_dim;
        let feature_dim = width * 2
            + ION_SERIES_EMBED_DIM_V0510
            + FRAGMENT_CHARGE_EMBED_DIM_V0510
            + PRECURSOR_CHARGE_EMBED_DIM_V0510
            + INSTRUMENT_EMBED_DIM_V0510
            + FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270;
        let mut blocks = Vec::with_capacity(config.fragment_layers);
        for layer in 0..config.fragment_layers {
            blocks.push(PeptideTransformerBlock::new(
                width,
                config.fragment_heads,
                config.fragment_ff_dim,
                config.base_v0500.dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        Ok(Self {
            ion_series_embedding: nn::embedding(
                ION_SERIES_CLASSES_V0510,
                ION_SERIES_EMBED_DIM_V0510,
                vb.pp("ion_series_embedding"),
            )?,
            fragment_charge_embedding: nn::embedding(
                FRAGMENT_CHARGE_CLASSES_V0510,
                FRAGMENT_CHARGE_EMBED_DIM_V0510,
                vb.pp("fragment_charge_embedding"),
            )?,
            precursor_charge_embedding: nn::embedding(
                PRECURSOR_CHARGE_CLASSES_V0510,
                PRECURSOR_CHARGE_EMBED_DIM_V0510,
                vb.pp("precursor_charge_embedding"),
            )?,
            instrument_embedding: nn::embedding(
                config.base_v0500.instrument_vocab_size,
                INSTRUMENT_EMBED_DIM_V0510,
                vb.pp("instrument_embedding"),
            )?,
            feature_norm: FoundationLayerNorm::new(feature_dim, 1e-5, vb.pp("feature_norm"))?,
            token_projection: nn::linear(feature_dim, width, vb.pp("token_projection"))?,
            blocks,
            output_norm: FoundationLayerNorm::new(width, 1e-5, vb.pp("output_norm"))?,
            presence_head: nn::linear(width, 1, vb.pp("presence"))?,
            intensity_head: nn::linear(width, 1, vb.pp("intensity"))?,
            // Zero gate means v0.51 starts exactly at the warm-started v0.50 MS2 surface.
            blend_gate: zero_initialized_linear_v0510(width, 1, vb.pp("blend_gate"))?,
            width,
        })
    }

    fn forward_t(
        &self,
        residues: &Tensor,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        base_ms2: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, sequence, width) = residues.dims3()?;
        if sequence < 2 || width != self.width {
            candle_core::bail!("v0.51 fragment decoder received incompatible residue states");
        }
        if fragment.cleavage_count > sequence - 1 || fragment.output_cleavage_count != sequence - 1
        {
            candle_core::bail!("v0.51 fragment context width is incompatible with residue width");
        }
        let channels = fragment.channels;
        let tokens = fragment.cleavage_count * channels;
        let left = residues
            .narrow(1, 0, fragment.cleavage_count)?
            .unsqueeze(2)?
            .broadcast_as((batch, fragment.cleavage_count, channels, width))?;
        let right = residues
            .narrow(1, 1, fragment.cleavage_count)?
            .unsqueeze(2)?
            .broadcast_as((batch, fragment.cleavage_count, channels, width))?;
        let cleavage = Tensor::cat(&[&left, &right], 3)?.reshape((batch, tokens, width * 2))?;
        let ion_series = self
            .ion_series_embedding
            .forward(&fragment.ion_series_ids)?;
        let fragment_charge = self
            .fragment_charge_embedding
            .forward(&fragment.fragment_charge_ids)?;
        let precursor_charge = self
            .precursor_charge_embedding
            .forward(&fragment.precursor_charge_ids)?;
        let instrument = self
            .instrument_embedding
            .forward(&context.instrument_ids)?
            .unsqueeze(1)?
            .broadcast_as((batch, tokens, INSTRUMENT_EMBED_DIM_V0510))?;
        let features = Tensor::cat(
            &[
                &cleavage,
                &ion_series,
                &fragment_charge,
                &precursor_charge,
                &instrument,
                &fragment.continuous_features,
            ],
            2,
        )?;
        let normalized = self.feature_norm.forward(&features)?;
        let mut hidden = self.token_projection.forward(&normalized)?.relu()?;
        let expanded_mask = fragment
            .token_mask
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, width))?;
        hidden = hidden.broadcast_mul(&expanded_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &fragment.token_mask, train)?;
        }
        let hidden = self.output_norm.forward(&hidden)?;
        let presence_logits = self.presence_head.forward(&hidden)?.squeeze(2)?;
        let intensity_logits = self.intensity_head.forward(&hidden)?.squeeze(2)?;
        let positive_intensity = apply_ms2_output_activation(
            &intensity_logits,
            FoundationMs2OutputActivation::SoftplusV0138,
        )?;
        let probability = ops::sigmoid(&presence_logits)?;
        let factorized = probability.broadcast_mul(&positive_intensity)?;

        // A zero-initialized signed interpolation gate lets the established v0.50
        // spectrum remain the exact step-0 prediction while the contextual decoder
        // becomes responsible for the spectrum as training proceeds.
        let gate_logits = self.blend_gate.forward(&hidden)?.squeeze(2)?;
        let gate = ops::sigmoid(&gate_logits)?.affine(2.0, -1.0)?;

        let presence_logits = presence_logits.broadcast_mul(&fragment.token_mask)?;
        let positive_intensity = positive_intensity.broadcast_mul(&fragment.token_mask)?;
        let factorized = factorized.broadcast_mul(&fragment.token_mask)?;
        let gate = gate.broadcast_mul(&fragment.token_mask)?;
        let active_presence =
            presence_logits.reshape((batch, fragment.cleavage_count, channels))?;
        let active_positive =
            positive_intensity.reshape((batch, fragment.cleavage_count, channels))?;
        let active_factorized = factorized.reshape((batch, fragment.cleavage_count, channels))?;
        let active_gate = gate.reshape((batch, fragment.cleavage_count, channels))?;
        let presence = pad_fragment_canvas_v0510(
            &active_presence,
            batch,
            fragment.cleavage_count,
            fragment.output_cleavage_count,
            channels,
        )?;
        let positive = pad_fragment_canvas_v0510(
            &active_positive,
            batch,
            fragment.cleavage_count,
            fragment.output_cleavage_count,
            channels,
        )?;
        let factorized = pad_fragment_canvas_v0510(
            &active_factorized,
            batch,
            fragment.cleavage_count,
            fragment.output_cleavage_count,
            channels,
        )?;
        let gate = pad_fragment_canvas_v0510(
            &active_gate,
            batch,
            fragment.cleavage_count,
            fragment.output_cleavage_count,
            channels,
        )?;
        if base_ms2.dims() != factorized.dims() {
            candle_core::bail!("v0.51 base/factorized MS2 shapes differ");
        }
        let delta = (&factorized - base_ms2)?;
        // Do not mask the blended surface here: gate=0 must preserve every warm-started
        // v0.50 MS2 value exactly at step 0. The trainer applies the theoretical-token
        // mask to supervised/teacher objectives.
        let blended = (base_ms2 + &gate.broadcast_mul(&delta)?)?.relu()?;
        Ok((presence, positive, blended))
    }
}

#[derive(Clone)]
struct MobilityResidualHeadV0510 {
    physics_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
    width: usize,
}

impl MobilityResidualHeadV0510 {
    fn new(config: &PeptideFoundationV0510Config, vb: VarBuilder<'_>) -> Result<Self> {
        let width = config.base_v0500.residue_dim;
        let mut blocks = Vec::with_capacity(config.mobility_layers);
        for layer in 0..config.mobility_layers {
            blocks.push(PeptideTransformerBlock::new(
                width,
                config.mobility_heads,
                config.mobility_ff_dim,
                config.base_v0500.dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        let head_input = 3 * width + FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360;
        Ok(Self {
            physics_projection: nn::linear(
                FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
                width,
                vb.pp("physics_projection"),
            )?,
            input_norm: FoundationLayerNorm::new(width, 1e-5, vb.pp("input_norm"))?,
            blocks,
            hidden: nn::linear(head_input, config.specialist_hidden, vb.pp("hidden"))?,
            bottleneck: nn::linear(
                config.specialist_hidden,
                config.specialist_bottleneck,
                vb.pp("bottleneck"),
            )?,
            output: zero_initialized_linear_v0510(
                config.specialist_bottleneck,
                1,
                vb.pp("output"),
            )?,
            width,
        })
    }

    fn forward_t(
        &self,
        base: &FoundationMultimodalForwardOutputV0500,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<Tensor> {
        let residues = &base.representation.residue_embeddings;
        let mask = &base.representation.residue_mask;
        let (batch, sequence, width) = residues.dims3()?;
        if width != self.width {
            candle_core::bail!("v0.51 mobility specialist received incompatible residue width");
        }
        let context = self.physics_projection.forward(&physics.ccs_physics)?;
        let context_residue = context
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, width))?;
        let mut hidden = (residues + &context_residue)?;
        hidden = self.input_norm.forward(&hidden)?;
        let expanded_mask = mask.unsqueeze(2)?.broadcast_as((batch, sequence, width))?;
        hidden = hidden.broadcast_mul(&expanded_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, mask, train)?;
        }
        let pooled = masked_mean_v0510(&hidden, mask)?;
        let features = Tensor::cat(
            &[
                &base.representation.mobility_embedding,
                &base.representation.global_embedding,
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

#[derive(Clone)]
pub struct PeptideFoundationV0510Model {
    config: PeptideFoundationV0510Config,
    base_v0500: PeptideFoundationV0500Model,
    teacher_peptide_projection: Linear,
    teacher_residue_projection: Linear,
    rt_refinement: RtResidualHeadV0510,
    ms2_context: Ms2ContextConditionerV0510,
    fragment_decoder: ContextualFragmentDecoderV0510,
    mobility_residual: MobilityResidualHeadV0510,
}

impl PeptideFoundationV0510Model {
    pub fn new(config: PeptideFoundationV0510Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        // Preserve the exact v0.50 namespace so a selected v0.50 checkpoint can
        // warm-start the entire backbone without key remapping.
        let base_v0500 = PeptideFoundationV0500Model::new(config.base_v0500.clone(), vb.clone())?;
        let specialist = vb.pp(FOUNDATION_V0510_STUDENT_NAMESPACE);
        let width = config.base_v0500.residue_dim;
        Ok(Self {
            teacher_peptide_projection: nn::linear(
                width,
                config.teacher_dim,
                specialist.pp("teacher_bridge.peptide"),
            )?,
            teacher_residue_projection: nn::linear(
                width,
                config.teacher_dim,
                specialist.pp("teacher_bridge.residue"),
            )?,
            rt_refinement: RtResidualHeadV0510::new(&config, specialist.pp("rt_refinement"))?,
            ms2_context: Ms2ContextConditionerV0510::new(&config, specialist.pp("ms2_context"))?,
            fragment_decoder: ContextualFragmentDecoderV0510::new(
                &config,
                specialist.pp("fragment_decoder"),
            )?,
            mobility_residual: MobilityResidualHeadV0510::new(
                &config,
                specialist.pp("mobility_residual"),
            )?,
            config,
            base_v0500,
        })
    }

    pub fn config(&self) -> &PeptideFoundationV0510Config {
        &self.config
    }

    /// Forward only through the warm-started v0.50 backbone. This is used by
    /// representation-only updates so they do not pay for v0.51 task specialists.
    pub fn base_v0500_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0500> {
        self.base_v0500.forward_t(batch, context, train)
    }

    /// Compute only the mobility residual specialist on top of the v0.50 backbone.
    pub fn mobility_residual_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        train: bool,
    ) -> Result<Tensor> {
        let base = self.base_v0500.forward_t(batch, context, train)?;
        self.mobility_residual.forward_t(&base, physics, train)
    }

    pub fn property_forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0510> {
        let base = self.base_v0500.forward_t(batch, context, train)?;
        let rt_residual = self.rt_refinement.forward(&base, physics)?;
        let rt = (&base.rt + &rt_residual)?;
        let contextual_residues = self.ms2_context.forward_t(
            &base.representation.residue_embeddings,
            &base.representation.residue_mask,
            context,
            train,
        )?;
        let (ms2_presence_logits, ms2_positive_intensity, ms2) = self.fragment_decoder.forward_t(
            &contextual_residues,
            context,
            fragment,
            &base.ms2,
            train,
        )?;
        let peptide_projection = self
            .teacher_peptide_projection
            .forward(&base.representation.global_embedding)?;
        let residue_projection = self
            .teacher_residue_projection
            .forward(&base.representation.residue_embeddings)?;
        let (batch_size, _) = batch.residue_mask.dims2()?;
        let mobility_residual_native =
            Tensor::zeros((batch_size, 1), DType::F32, batch.residue_mask.device())?;
        Ok(FoundationMultimodalForwardOutputV0510 {
            base_v0500: base,
            rt,
            mobility_residual_native,
            ms2,
            ms2_presence_logits,
            ms2_positive_intensity,
            teacher_bridge: FoundationTeacherBridgeOutputV0510 {
                peptide_projection,
                residue_projection,
            },
        })
    }

    pub fn forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        physics: &FoundationScalarPhysicsBatchV0360,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0510> {
        let mut output = self.property_forward_t(batch, context, physics, fragment, train)?;
        output.mobility_residual_native =
            self.mobility_residual
                .forward_t(&output.base_v0500, physics, train)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct FoundationTeacherAnchorOutputV0510 {
    pub rt: Tensor,
    pub ccs: Tensor,
    pub ms2: Tensor,
    pub peptide_embedding: Tensor,
    pub residue_embeddings: Tensor,
    pub residue_mask: Tensor,
}

#[derive(Clone)]
pub struct FoundationV0350TeacherV0510 {
    model: PeptideFoundationMultimodalV0350Model,
}

impl FoundationV0350TeacherV0510 {
    pub fn new(model: PeptideFoundationMultimodalV0350Model) -> Self {
        Self { model }
    }

    pub fn forward_detached_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
    ) -> Result<FoundationTeacherAnchorOutputV0510> {
        let output = self
            .model
            .forward_v0350_t(batch, context, fragment, false)?;
        Ok(FoundationTeacherAnchorOutputV0510 {
            rt: output.base.rt.detach(),
            ccs: output.base.ccs.detach(),
            ms2: output.base.ms2.detach(),
            peptide_embedding: output.base.foundation.peptide_embedding.detach(),
            residue_embeddings: output.base.foundation.residue_embeddings.detach(),
            residue_mask: output.base.foundation.residue_mask.clone(),
        })
    }

    pub fn protected_ccs_detached_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
    ) -> Result<Tensor> {
        Ok(self.model.protected_ccs_v0350_t(batch, context)?.detach())
    }
}

fn masked_mean_v0510(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn pad_fragment_canvas_v0510(
    active: &Tensor,
    batch: usize,
    active_cleavages: usize,
    output_cleavages: usize,
    channels: usize,
) -> Result<Tensor> {
    if active_cleavages == output_cleavages {
        return Ok(active.clone());
    }
    if active_cleavages > output_cleavages {
        candle_core::bail!("v0.51 active fragment canvas exceeds output width");
    }
    let padding = Tensor::zeros(
        (batch, output_cleavages - active_cleavages, channels),
        active.dtype(),
        active.device(),
    )?;
    Tensor::cat(&[active, &padding], 1)
}

fn zero_initialized_linear_v0510(
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
    fn v0510_fixed_config_preserves_the_v0500_backbone_contract() {
        let cfg = PeptideFoundationV0510Config::default();
        cfg.validate().unwrap();
        assert_eq!(cfg.base_v0500.residue_dim, 320);
        assert_eq!(cfg.base_v0500.pair_dim, 128);
        assert_eq!(cfg.base_v0500.interaction_blocks, 8);
        assert_eq!(cfg.teacher_dim, 192);
        assert_eq!(cfg.ms2_context_layers, 3);
        assert_eq!(cfg.fragment_layers, 2);
        assert_eq!(cfg.mobility_layers, 2);
    }

    #[test]
    fn v0510_namespaces_keep_v0500_warm_start_separate_from_new_specialists() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model =
            PeptideFoundationV0510Model::new(PeptideFoundationV0510Config::local_smoke(), vb)
                .unwrap();
        assert_eq!(model.config().base_v0500.residue_dim, 64);
        let data = varmap.data().lock().unwrap();
        assert!(data.keys().any(|name| name.starts_with("student_v050.")));
        assert!(data.keys().any(|name| name.starts_with("student_v051.")));
        let mobility_output = data
            .get("student_v051.mobility_residual.output.weight")
            .unwrap()
            .as_tensor()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(mobility_output, 0.0);
        let gate = data
            .get("student_v051.fragment_decoder.blend_gate.weight")
            .unwrap()
            .as_tensor()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(gate, 0.0);
    }
}
