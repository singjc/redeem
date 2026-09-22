//! v0.35 trainable forward-representation continuation.
//!
//! v0.34 showed that substantially larger RT/MS2 heads placed on top of a
//! frozen v0.31 representation do not materially improve MS2 shape/correlation.
//! v0.35 therefore moves the adaptation boundary into the peptide representation
//! itself while preserving the accepted v0.31 model as an immutable anchor for
//! CCS, inverse generation, alignment, and relation scoring.
//!
//! The forward branch is an exact trainable clone of the v0.31 peptide-side
//! encoder/refinement/RT/MS2 weights at step 0. During optimization the clone is
//! allowed to move end-to-end under RT, factorized MS2, masked-residue,
//! chemistry-reconstruction, contrastive, and fragment-deep-supervision losses.
//! An identity-initialized acquisition-context conditioner injects precursor
//! charge, NCE, and instrument information *before* additional residue
//! Transformer blocks on the MS2 path. This lets fragmentation context reshape
//! residue states instead of being consumed only by the terminal fragment head.
//!
//! CCS remains the exact frozen v0.31/v0.27 prediction path. The inverse model
//! remains the exact frozen v0.31 path as well.

use super::config::FoundationConfig;
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
};
use super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
use super::model::{
    apply_ms2_output_activation, FoundationMultiTaskOutput, FoundationOutput, PrecursorContextBatch,
};
use super::multimodal_v0270::{
    foundation_multimodal_ms2_loss_v0270, FoundationMultimodalForwardOutputV0270,
    FoundationMultimodalMs2LossesV0270, PeptideFoundationMultimodalForwardV0270,
};
use super::multimodal_v0310::{
    FoundationFragmentContextBatchV0310, PeptideFoundationMultimodalV0310Config,
    PeptideFoundationMultimodalV0310Model,
};
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0350: &str =
    "v0.35-trainable-forward-representation-context-conditioned-ms2-v031-anchor";
pub const FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0350: usize = 1;
pub const FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0350: usize = 6;
pub const FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0350: usize = 768;
pub const FOUNDATION_MS2_CONTEXT_LAYERS_V0350: usize = 3;
pub const FOUNDATION_MS2_CONTEXT_HEADS_V0350: usize = 6;
pub const FOUNDATION_MS2_CONTEXT_FF_DIM_V0350: usize = 768;
pub const FOUNDATION_MS2_CONTEXT_INSTRUMENT_DIM_V0350: usize = 16;
pub const FOUNDATION_FRAGMENT_AUX_HIDDEN_V0350: usize = 192;
pub const FOUNDATION_FRAGMENT_AUX_WEIGHT_V0350: f64 = 0.20;
pub const FOUNDATION_RT_ROBUST_WEIGHT_V0350: f64 = 0.25;
pub const FOUNDATION_RT_ROBUST_DELTA_V0350: f64 = 0.50;
pub const FOUNDATION_MS2_PEARSON_WEIGHT_V0350: f64 = 0.25;
pub const FOUNDATION_REPRESENTATION_MASKED_WEIGHT_V0350: f64 = 0.15;
pub const FOUNDATION_REPRESENTATION_CHEMISTRY_WEIGHT_V0350: f64 = 0.10;
pub const FOUNDATION_REPRESENTATION_CONTRASTIVE_WEIGHT_V0350: f64 = 0.05;
pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0350: usize =
    super::multimodal_v0310::FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0310;
pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0350: f64 =
    super::multimodal_v0310::FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0310;

const MS2_CONTEXT_SCALARS_V0350: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationMultimodalV0350Config {
    pub base_v0310: PeptideFoundationMultimodalV0310Config,
    pub property_refinement_layers: usize,
    pub property_refinement_heads: usize,
    pub property_refinement_ff_dim: usize,
    pub ms2_context_layers: usize,
    pub ms2_context_heads: usize,
    pub ms2_context_ff_dim: usize,
    pub ms2_context_instrument_dim: usize,
    pub fragment_aux_hidden: usize,
}

impl PeptideFoundationMultimodalV0350Config {
    pub fn fixed(base_v0310: PeptideFoundationMultimodalV0310Config) -> Result<Self> {
        base_v0310.validate()?;
        let config = Self {
            base_v0310,
            property_refinement_layers: FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0350,
            property_refinement_heads: FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0350,
            property_refinement_ff_dim: FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0350,
            ms2_context_layers: FOUNDATION_MS2_CONTEXT_LAYERS_V0350,
            ms2_context_heads: FOUNDATION_MS2_CONTEXT_HEADS_V0350,
            ms2_context_ff_dim: FOUNDATION_MS2_CONTEXT_FF_DIM_V0350,
            ms2_context_instrument_dim: FOUNDATION_MS2_CONTEXT_INSTRUMENT_DIM_V0350,
            fragment_aux_hidden: FOUNDATION_FRAGMENT_AUX_HIDDEN_V0350,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0310.validate()?;
        if self.forward().model_dim != 192 {
            candle_core::bail!("v0.35 requires the accepted 192d v0.31 parent configuration");
        }
        if self.property_refinement_layers != FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0350
            || self.property_refinement_heads != FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0350
            || self.property_refinement_ff_dim != FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0350
            || self.ms2_context_layers != FOUNDATION_MS2_CONTEXT_LAYERS_V0350
            || self.ms2_context_heads != FOUNDATION_MS2_CONTEXT_HEADS_V0350
            || self.ms2_context_ff_dim != FOUNDATION_MS2_CONTEXT_FF_DIM_V0350
            || self.ms2_context_instrument_dim != FOUNDATION_MS2_CONTEXT_INSTRUMENT_DIM_V0350
            || self.fragment_aux_hidden != FOUNDATION_FRAGMENT_AUX_HIDDEN_V0350
        {
            candle_core::bail!("v0.35 dimensions differ from the fixed architecture");
        }
        if self.forward().model_dim % self.property_refinement_heads != 0
            || self.forward().model_dim % self.ms2_context_heads != 0
        {
            candle_core::bail!("v0.35 model width must be divisible by attention-head counts");
        }
        Ok(())
    }

    pub fn forward(&self) -> &FoundationConfig {
        self.base_v0310.forward()
    }

    pub fn inverse(&self) -> &FoundationDiffusionConfig {
        self.base_v0310.inverse()
    }
}

#[derive(Clone)]
struct PropertyResidueRefinementV0350 {
    input_norm: FoundationLayerNorm,
    transformer: PeptideTransformerBlock,
    delta_output: Linear,
    model_dim: usize,
}

impl PropertyResidueRefinementV0350 {
    fn new(config: &PeptideFoundationMultimodalV0350Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        Ok(Self {
            input_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("input_norm"))?,
            transformer: PeptideTransformerBlock::new(
                model_dim,
                config.property_refinement_heads,
                config.property_refinement_ff_dim,
                config.forward().dropout,
                vb.pp("transformer.0"),
            )?,
            delta_output: zero_initialized_linear_v0350(
                model_dim,
                model_dim,
                vb.pp("delta_output"),
            )?,
            model_dim,
        })
    }

    fn forward_t(&self, base: &FoundationOutput, train: bool) -> Result<FoundationOutput> {
        let (batch, sequence, model_dim) = base.residue_embeddings.dims3()?;
        if model_dim != self.model_dim {
            candle_core::bail!(
                "v0.35 property refinement expected residue width {}, got {}",
                self.model_dim,
                model_dim
            );
        }
        let normalized = self.input_norm.forward(&base.residue_embeddings)?;
        let hidden = self
            .transformer
            .forward_t(&normalized, &base.residue_mask, train)?;
        let delta = self.delta_output.forward(&hidden)?;
        let expanded_mask = base
            .residue_mask
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, model_dim))?;
        let delta = delta.broadcast_mul(&expanded_mask)?;
        let residue_embeddings = (&base.residue_embeddings + &delta)?;
        let pooled_delta = masked_mean_v0350(&delta, &base.residue_mask)?;
        let peptide_embedding = (&base.peptide_embedding + &pooled_delta)?;
        Ok(FoundationOutput {
            residue_embeddings,
            peptide_embedding,
            residue_mask: base.residue_mask.clone(),
            chemistry_targets: base.chemistry_targets.clone(),
        })
    }
}

#[derive(Clone)]
struct Ms2ContextConditionerV0350 {
    instrument_embedding: Embedding,
    context_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    delta_output: Linear,
    model_dim: usize,
    instrument_dim: usize,
}

impl Ms2ContextConditionerV0350 {
    fn new(config: &PeptideFoundationMultimodalV0350Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        let mut blocks = Vec::with_capacity(config.ms2_context_layers);
        for layer in 0..config.ms2_context_layers {
            blocks.push(PeptideTransformerBlock::new(
                model_dim,
                config.ms2_context_heads,
                config.ms2_context_ff_dim,
                config.forward().dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        let context_dim = config.ms2_context_instrument_dim + MS2_CONTEXT_SCALARS_V0350;
        Ok(Self {
            instrument_embedding: nn::embedding(
                config.forward().instrument_vocab_size,
                config.ms2_context_instrument_dim,
                vb.pp("instrument_embedding"),
            )?,
            context_projection: nn::linear(context_dim, model_dim, vb.pp("context_projection"))?,
            input_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("input_norm"))?,
            blocks,
            delta_output: zero_initialized_linear_v0350(
                model_dim,
                model_dim,
                vb.pp("delta_output"),
            )?,
            model_dim,
            instrument_dim: config.ms2_context_instrument_dim,
        })
    }

    fn forward_t(
        &self,
        foundation: &FoundationOutput,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationOutput> {
        let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
        if model_dim != self.model_dim {
            candle_core::bail!("v0.35 MS2 context conditioner received incompatible residue width");
        }
        let instrument = self.instrument_embedding.forward(&context.instrument_ids)?;
        if instrument.dims2()? != (batch, self.instrument_dim) {
            candle_core::bail!("v0.35 instrument embedding shape mismatch");
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
            .broadcast_as((batch, sequence, model_dim))?;
        let input = (&foundation.residue_embeddings + &context_embedding)?;
        let mut hidden = self.input_norm.forward(&input)?;
        let expanded_mask = foundation
            .residue_mask
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, model_dim))?;
        hidden = hidden.broadcast_mul(&expanded_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &foundation.residue_mask, train)?;
        }
        let delta = self
            .delta_output
            .forward(&hidden)?
            .broadcast_mul(&expanded_mask)?;
        let residue_embeddings = (&foundation.residue_embeddings + &delta)?;
        let pooled_delta = masked_mean_v0350(&delta, &foundation.residue_mask)?;
        let peptide_embedding = (&foundation.peptide_embedding + &pooled_delta)?;
        Ok(FoundationOutput {
            residue_embeddings,
            peptide_embedding,
            residue_mask: foundation.residue_mask.clone(),
            chemistry_targets: foundation.chemistry_targets.clone(),
        })
    }
}

#[derive(Clone)]
struct FragmentRepresentationAuxV0350 {
    hidden: Linear,
    output: Linear,
    model_dim: usize,
    channels: usize,
}

impl FragmentRepresentationAuxV0350 {
    fn new(config: &PeptideFoundationMultimodalV0350Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        let channels = config.forward().ms2_fragment_channels;
        let input_dim = 2 * model_dim + 4;
        Ok(Self {
            hidden: nn::linear(input_dim, config.fragment_aux_hidden, vb.pp("hidden"))?,
            output: nn::linear(config.fragment_aux_hidden, channels, vb.pp("output"))?,
            model_dim,
            channels,
        })
    }

    fn forward(
        &self,
        foundation: &FoundationOutput,
        context: &PrecursorContextBatch,
        activation: super::config::FoundationMs2OutputActivation,
    ) -> Result<Tensor> {
        let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
        if model_dim != self.model_dim || sequence < 2 {
            candle_core::bail!("v0.35 fragment deep-supervision representation shape mismatch");
        }
        let cleavages = sequence - 1;
        let left = foundation.residue_embeddings.narrow(1, 0, cleavages)?;
        let right = foundation.residue_embeddings.narrow(1, 1, cleavages)?;
        let charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let charge_present = context.charge_present.unsqueeze(1)?;
        let nce = context.nce.affine(1.0 / 100.0, 0.0)?.unsqueeze(1)?;
        let nce_present = context.nce_present.unsqueeze(1)?;
        let scalar_context = Tensor::cat(&[&charge, &charge_present, &nce, &nce_present], 1)?
            .unsqueeze(1)?
            .broadcast_as((batch, cleavages, 4))?;
        let features = Tensor::cat(&[&left, &right, &scalar_context], 2)?.contiguous()?;
        let (_, _, feature_dim) = features.dims3()?;
        let hidden = self
            .hidden
            .forward(
                &features
                    .reshape((batch * cleavages, feature_dim))?
                    .contiguous()?,
            )?
            .relu()?;
        let logits = self
            .output
            .forward(&hidden)?
            .reshape((batch, cleavages, self.channels))?;
        let prediction = apply_ms2_output_activation(&logits, activation)?;
        let left_mask = foundation.residue_mask.narrow(1, 0, cleavages)?;
        let right_mask = foundation.residue_mask.narrow(1, 1, cleavages)?;
        let mask = left_mask
            .broadcast_mul(&right_mask)?
            .unsqueeze(2)?
            .broadcast_as((batch, cleavages, self.channels))?;
        prediction.broadcast_mul(&mask)
    }
}

#[derive(Debug, Clone)]
pub struct FoundationRepresentationAuxOutputV0350 {
    pub foundation: FoundationOutput,
    pub residue_logits: Tensor,
    pub chemistry_reconstruction: Tensor,
    pub contrastive_projection: Tensor,
}

#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0350 {
    pub base: FoundationMultiTaskOutput,
    pub ms2_presence_logits: Tensor,
    pub ms2_positive_intensity: Tensor,
    pub fragment_representation_aux: Tensor,
}

#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0350Model {
    base_v0310: PeptideFoundationMultimodalV0310Model,
    forward_v0350: PeptideFoundationMultimodalForwardV0270,
    property_refinement_v0350: PropertyResidueRefinementV0350,
    ms2_context_v0350: Ms2ContextConditionerV0350,
    fragment_aux_v0350: FragmentRepresentationAuxV0350,
    config: PeptideFoundationMultimodalV0350Config,
}

impl PeptideFoundationMultimodalV0350Model {
    pub fn new(config: PeptideFoundationMultimodalV0350Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let base_v0310 =
            PeptideFoundationMultimodalV0310Model::new(config.base_v0310.clone(), vb.clone())?;
        let forward_v0350 = PeptideFoundationMultimodalForwardV0270::new(
            config.base_v0310.base_v0270.clone(),
            vb.pp("forward_v0350"),
        )?;
        let property_refinement_v0350 =
            PropertyResidueRefinementV0350::new(&config, vb.pp("property_refinement_v0350"))?;
        let ms2_context_v0350 =
            Ms2ContextConditionerV0350::new(&config, vb.pp("ms2_context_v0350"))?;
        let fragment_aux_v0350 =
            FragmentRepresentationAuxV0350::new(&config, vb.pp("fragment_aux_v0350"))?;
        Ok(Self {
            base_v0310,
            forward_v0350,
            property_refinement_v0350,
            ms2_context_v0350,
            fragment_aux_v0350,
            config,
        })
    }

    fn trainable_property_foundation_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<FoundationOutput> {
        let encoded = self.forward_v0350.encode_foundation_t(batch, train)?;
        self.property_refinement_v0350.forward_t(&encoded, train)
    }

    /// Frozen scalar anchor used by later protected specialist stages.
    ///
    /// This evaluates only the v0.35 trainable-forward representation, RT head,
    /// and protected v0.31 CCS path. Returned tensors are detached so downstream
    /// specialist optimizers cannot backpropagate into the accepted v0.35 model.
    pub fn detached_scalar_anchor_v0350_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
    ) -> Result<(FoundationOutput, Tensor, Tensor)> {
        let foundation = self.trainable_property_foundation_t(batch, false)?;
        let rt = self
            .forward_v0350
            .rt_from_foundation_t(&foundation, false, 0.0)?
            .detach();
        let ccs = self.base_v0310.protected_ccs_t(batch, context)?.detach();
        Ok((
            FoundationOutput {
                residue_embeddings: foundation.residue_embeddings.detach(),
                peptide_embedding: foundation.peptide_embedding.detach(),
                residue_mask: foundation.residue_mask.clone(),
                chemistry_targets: foundation.chemistry_targets.clone(),
            },
            rt,
            ccs,
        ))
    }

    pub fn forward_representation_aux_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<FoundationRepresentationAuxOutputV0350> {
        let foundation = self.trainable_property_foundation_t(batch, train)?;
        let (residue_logits, chemistry_reconstruction, contrastive_projection) = self
            .forward_v0350
            .auxiliaries_from_foundation(&foundation)?;
        Ok(FoundationRepresentationAuxOutputV0350 {
            foundation,
            residue_logits,
            chemistry_reconstruction,
            contrastive_projection,
        })
    }

    pub fn forward_v0350_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0310,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0350> {
        // Immutable accepted parent supplies only protected CCS here. Inverse,
        // alignment, and relation methods remain delegated to v0.31 below.
        let protected_ccs = self.base_v0310.protected_ccs_t(batch, context)?;

        // Trainable forward representation starts as an exact clone of the
        // parent's encoder + property refinement and can now move end-to-end.
        let property_foundation = self.trainable_property_foundation_t(batch, train)?;
        let rt = self
            .forward_v0350
            .rt_from_foundation_t(&property_foundation, train, 1.0)?;
        let contextual_foundation =
            self.ms2_context_v0350
                .forward_t(&property_foundation, context, train)?;
        let (ms2_presence_logits, ms2_positive_intensity, ms2) = self
            .forward_v0350
            .ms2_from_foundation_t(&contextual_foundation, context, fragment, train)?;
        let (residue_logits, chemistry_reconstruction, contrastive_projection) = self
            .forward_v0350
            .auxiliaries_from_foundation(&property_foundation)?;
        let fragment_representation_aux = self.fragment_aux_v0350.forward(
            &contextual_foundation,
            context,
            self.config.forward().ms2_output_activation,
        )?;

        Ok(FoundationMultimodalForwardOutputV0350 {
            base: FoundationMultiTaskOutput {
                foundation: property_foundation,
                rt,
                ccs: protected_ccs,
                ms2,
                residue_logits,
                chemistry_reconstruction,
                contrastive_projection,
            },
            ms2_presence_logits,
            ms2_positive_intensity,
            fragment_representation_aux,
        })
    }

    /// Frozen v0.31 property representation retained for inverse diagnostics.
    pub fn property_foundation_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<FoundationOutput> {
        self.base_v0310.property_foundation_t(batch, train)
    }

    /// Frozen v0.31 projection retained for inverse/alignment diagnostics.
    pub fn peptide_projection_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<Tensor> {
        self.base_v0310.peptide_projection_t(batch, train)
    }

    pub fn diffusion(&self) -> &PeptideSpectrumDiffusionModel {
        self.base_v0310.diffusion()
    }

    pub fn causal(&self) -> &super::causal::PeptideSpectrumCausalModel {
        self.base_v0310.causal()
    }

    pub fn encode_spectrum_t(
        &self,
        spectrum: &FoundationSpectrumBatch,
        train: bool,
    ) -> Result<FoundationSpectrumEncoding> {
        self.base_v0310.encode_spectrum_t(spectrum, train)
    }

    pub fn project_spectrum_embedding(&self, embedding: &Tensor) -> Result<Tensor> {
        self.base_v0310.project_spectrum_embedding(embedding)
    }

    pub fn relation_score(
        &self,
        peptide: &FoundationOutput,
        spectrum: &FoundationSpectrumEncoding,
    ) -> Result<Tensor> {
        self.base_v0310.relation_score(peptide, spectrum)
    }

    pub fn forward_config(&self) -> &FoundationConfig {
        self.config.forward()
    }

    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        self.config.inverse()
    }

    pub fn config(&self) -> &PeptideFoundationMultimodalV0350Config {
        &self.config
    }
}

pub type FoundationFragmentContextBatchV0350 = FoundationFragmentContextBatchV0310;
pub type FoundationMultimodalMs2LossesV0350 = FoundationMultimodalMs2LossesV0270;

pub fn foundation_multimodal_relation_margin_loss_v0350(
    positive_score: &Tensor,
    negative_score: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    super::multimodal_v0310::foundation_multimodal_relation_margin_loss_v0310(
        positive_score,
        negative_score,
        margin,
    )
}

pub fn foundation_multimodal_ms2_loss_v0350(
    output: &FoundationMultimodalForwardOutputV0350,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    presence_target: Option<&Tensor>,
    presence_mask: Option<&Tensor>,
) -> Result<FoundationMultimodalMs2LossesV0350> {
    let proxy = FoundationMultimodalForwardOutputV0270 {
        base: output.base.clone(),
        ms2_presence_logits: output.ms2_presence_logits.clone(),
        ms2_positive_intensity: output.ms2_positive_intensity.clone(),
    };
    foundation_multimodal_ms2_loss_v0270(&proxy, target, mask, presence_target, presence_mask)
}

pub fn foundation_fragment_representation_aux_loss_v0350(
    output: &FoundationMultimodalForwardOutputV0350,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    config: super::loss::FoundationMs2LossConfig,
) -> Result<Option<super::loss::FoundationMs2Losses>> {
    match (target, mask) {
        (Some(target), Some(mask)) => Ok(Some(super::loss::foundation_ms2_loss(
            &output.fragment_representation_aux,
            target,
            mask,
            config,
        )?)),
        (None, None) => Ok(None),
        _ => candle_core::bail!("v0.35 fragment auxiliary target/mask must be supplied together"),
    }
}

fn masked_mean_v0350(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn zero_initialized_linear_v0350(
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
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn config() -> PeptideFoundationMultimodalV0350Config {
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
        PeptideFoundationMultimodalV0350Config::fixed(v0310).unwrap()
    }

    #[test]
    fn v0350_has_trainable_forward_clone_and_context_conditioner() {
        let varmap = VarMap::new();
        let _model = PeptideFoundationMultimodalV0350Model::new(
            config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(data
            .keys()
            .any(|name| name.starts_with("forward_v0350.encoder.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("property_refinement_v0350.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("ms2_context_v0350.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("fragment_aux_v0350.")));
    }

    #[test]
    fn v0350_ms2_context_delta_is_zero_initialized() {
        let varmap = VarMap::new();
        let _model = PeptideFoundationMultimodalV0350Model::new(
            config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let data = varmap.data().lock().unwrap();
        let weight = data
            .get("ms2_context_v0350.delta_output.weight")
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(weight.iter().all(|&value| value == 0.0));
    }
}
