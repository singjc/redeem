//! v0.26 multimodal peptide/spectrum foundation architecture.
//!
//! This module keeps the chemistry-aware peptide encoder and the successful
//! spectrum-conditioned inverse backbone, but separates property-specific
//! adapters from the shared peptide representation and adds an explicit
//! peptide<->peak relation head. MS2 prediction is factorized into fragment
//! presence and positive intensity so absence is no longer represented only as
//! a continuous-regression target.

use super::causal::PeptideSpectrumCausalModel;
use super::ccs_physics::FOUNDATION_CCS_PHYSICS_FEATURE_COUNT;
use super::config::{
    FoundationCcsContextMode, FoundationCcsPhysicsBaselineConfig, FoundationConfig,
    FoundationMs2OutputActivation,
};
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoder, FoundationSpectrumEncoding,
    PeptideSpectrumDiffusionModel,
};
use super::layers::{FoundationLayerNorm, MultiHeadCrossAttention};
use super::loss::{foundation_ms2_loss, FoundationMs2LossConfig};
use super::model::{
    apply_ms2_output_activation, gradient_scaled_identity, FoundationMultiTaskOutput,
    FoundationOutput, PeptideFoundationEncoder, PrecursorContextBatch,
};
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{self as nn, ops, Embedding, Linear, VarBuilder};

/// Stable architecture label used in metadata/logs.
pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0260: &str =
    "v0.26.0-192d-task-adapters-ms2-presence-intensity-crossmodal-relation-phospho29";
/// Hidden width of the scalar-property adapters.
pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0260: usize = 384;
/// Fixed margin used by the same-spectrum positive-vs-negative relation loss.
pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0260: f64 = 0.25;
/// Fixed presence component weight in the factorized MS2 objective.
pub const FOUNDATION_MULTIMODAL_MS2_PRESENCE_WEIGHT_V0260: f64 = 0.25;
/// Fixed positive-intensity component weight in the factorized MS2 objective.
pub const FOUNDATION_MULTIMODAL_MS2_INTENSITY_WEIGHT_V0260: f64 = 1.0;
/// Fixed expected-spectrum cosine component weight in the factorized MS2 objective.
pub const FOUNDATION_MULTIMODAL_MS2_COSINE_WEIGHT_V0260: f64 = 0.25;

#[derive(Clone)]
struct PropertyAdapter {
    norm: FoundationLayerNorm,
    hidden_in: Linear,
    hidden_out: Linear,
    output: Linear,
}

impl PropertyAdapter {
    fn new(input_dim: usize, hidden_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            norm: FoundationLayerNorm::new(input_dim, 1e-5, vb.pp("norm"))?,
            hidden_in: nn::linear(input_dim, hidden_dim, vb.pp("hidden_in"))?,
            hidden_out: nn::linear(hidden_dim, hidden_dim, vb.pp("hidden_out"))?,
            output: zero_initialized_linear(hidden_dim, 1, vb.pp("output"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let normalized = self.norm.forward(input)?;
        let hidden = self.hidden_in.forward(&normalized)?.relu()?;
        let hidden = self.hidden_out.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)
    }
}

#[derive(Clone)]
struct MultimodalMs2Head {
    instrument_embedding: Embedding,
    feature_norm: FoundationLayerNorm,
    hidden_in: Linear,
    hidden_out: Linear,
    presence_head: Linear,
    intensity_head: Linear,
    channels: usize,
}

impl MultimodalMs2Head {
    fn new(config: &FoundationConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let feature_dim = config.model_dim * 2 + 21;
        Ok(Self {
            instrument_embedding: nn::embedding(
                config.instrument_vocab_size,
                16,
                vb.pp("instrument_embedding"),
            )?,
            feature_norm: FoundationLayerNorm::new(feature_dim, 1e-5, vb.pp("feature_norm"))?,
            hidden_in: nn::linear(feature_dim, config.model_dim, vb.pp("hidden_in"))?,
            hidden_out: nn::linear(config.model_dim, config.model_dim, vb.pp("hidden_out"))?,
            presence_head: nn::linear(
                config.model_dim,
                config.ms2_fragment_channels,
                vb.pp("presence"),
            )?,
            intensity_head: nn::linear(
                config.model_dim,
                config.ms2_fragment_channels,
                vb.pp("intensity"),
            )?,
            channels: config.ms2_fragment_channels,
        })
    }

    fn forward(
        &self,
        foundation: &FoundationOutput,
        context: &PrecursorContextBatch,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, sequence_len, model_dim) = foundation.residue_embeddings.dims3()?;
        if sequence_len < 2 {
            candle_core::bail!("v0.26 MS2 head requires at least two sequence positions");
        }
        let cleavage_count = sequence_len - 1;
        let left = foundation.residue_embeddings.narrow(1, 0, cleavage_count)?;
        let right = foundation.residue_embeddings.narrow(1, 1, cleavage_count)?;
        let instrument = self.instrument_embedding.forward(&context.instrument_ids)?;
        let scaled_charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let scaled_nce = context.nce.affine(1.0 / 100.0, 0.0)?.unsqueeze(1)?;
        let charge_present = context.charge_present.unsqueeze(1)?;
        let nce_present = context.nce_present.unsqueeze(1)?;
        let instrument_present = context.instrument_present.unsqueeze(1)?;
        let scalar_context = Tensor::cat(
            &[
                &scaled_charge,
                &scaled_nce,
                &charge_present,
                &nce_present,
                &instrument_present,
            ],
            1,
        )?;
        let context_features = Tensor::cat(&[&instrument, &scalar_context], 1)?
            .unsqueeze(1)?
            .broadcast_as((batch_size, cleavage_count, 21))?;
        let features = Tensor::cat(&[&left, &right, &context_features], 2)?.contiguous()?;
        let feature_dim = model_dim * 2 + 21;
        let flat = features
            .reshape((batch_size * cleavage_count, feature_dim))?
            .contiguous()?;
        let normalized = self.feature_norm.forward(&flat)?;
        let hidden = self.hidden_in.forward(&normalized)?.relu()?;
        let hidden = self.hidden_out.forward(&hidden)?.relu()?;
        let presence_logits = self.presence_head.forward(&hidden)?.reshape((
            batch_size,
            cleavage_count,
            self.channels,
        ))?;
        let intensity_logits = self.intensity_head.forward(&hidden)?.reshape((
            batch_size,
            cleavage_count,
            self.channels,
        ))?;
        let positive_intensity = apply_ms2_output_activation(
            &intensity_logits,
            FoundationMs2OutputActivation::SoftplusV0138,
        )?;
        let presence_probability = ops::sigmoid(&presence_logits)?;
        let expected_intensity = presence_probability.broadcast_mul(&positive_intensity)?;

        let left_mask = foundation.residue_mask.narrow(1, 0, cleavage_count)?;
        let right_mask = foundation.residue_mask.narrow(1, 1, cleavage_count)?;
        let cleavage_mask = left_mask
            .broadcast_mul(&right_mask)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, cleavage_count, self.channels))?;

        Ok((
            presence_logits.broadcast_mul(&cleavage_mask)?,
            positive_intensity.broadcast_mul(&cleavage_mask)?,
            expected_intensity.broadcast_mul(&cleavage_mask)?,
        ))
    }
}

/// v0.26 forward output exposing the factorized MS2 branches as well as the
/// historical multi-task view used by downstream evaluation code.
#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0260 {
    /// Backward-compatible forward/property output. `ms2` is the expected
    /// intensity `P(fragment present) * E[intensity | present]`.
    pub base: FoundationMultiTaskOutput,
    /// Fragment-presence logits matching the MS2 target tensor shape.
    pub ms2_presence_logits: Tensor,
    /// Positive conditional intensity prediction matching the MS2 target shape.
    pub ms2_positive_intensity: Tensor,
}

/// Peptide-side v0.26 model with task-specific property adapters and factorized
/// MS2 prediction.
#[derive(Clone)]
pub struct PeptideFoundationMultimodalForwardV0260 {
    encoder: PeptideFoundationEncoder,
    rt_adapter: PropertyAdapter,
    ccs_adapter: PropertyAdapter,
    ms2_head: MultimodalMs2Head,
    residue_head: Linear,
    chemistry_head: Linear,
    contrastive_head: Linear,
    config: FoundationConfig,
}

impl PeptideFoundationMultimodalForwardV0260 {
    /// Construct the v0.26 peptide/property branch.
    pub fn new(config: FoundationConfig, hidden_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let encoder = PeptideFoundationEncoder::new(config.clone(), vb.pp("encoder"))?;
        Ok(Self {
            rt_adapter: PropertyAdapter::new(config.model_dim, hidden_dim, vb.pp("adapters.rt"))?,
            ccs_adapter: PropertyAdapter::new(
                config.model_dim + 2,
                hidden_dim,
                vb.pp("adapters.ccs"),
            )?,
            ms2_head: MultimodalMs2Head::new(&config, vb.pp("adapters.ms2"))?,
            residue_head: nn::linear(config.model_dim, 21, vb.pp("heads.masked_residue"))?,
            chemistry_head: nn::linear(
                config.model_dim,
                config.atom_feature_dim,
                vb.pp("heads.chemistry"),
            )?,
            contrastive_head: nn::linear(
                config.model_dim,
                config.contrastive_dim,
                vb.pp("heads.contrastive"),
            )?,
            encoder,
            config,
        })
    }

    /// Historical forward-output surface used by evaluation/export code.
    pub fn forward_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationMultiTaskOutput> {
        Ok(self
            .forward_v0260_t_with_shared_gradient_scales(batch, context, train, 1.0, 1.0)?
            .base)
    }

    /// Forward with independent RT/CCS gradient scaling into the shared peptide encoder.
    pub fn forward_t_with_shared_gradient_scales(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
        rt_encoder_gradient_scale: f64,
        ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultiTaskOutput> {
        Ok(self
            .forward_v0260_t_with_shared_gradient_scales(
                batch,
                context,
                train,
                rt_encoder_gradient_scale,
                ccs_encoder_gradient_scale,
            )?
            .base)
    }

    /// Full v0.26 forward pass exposing the factorized MS2 branches.
    pub fn forward_v0260_t_with_shared_gradient_scales(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
        rt_encoder_gradient_scale: f64,
        ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultimodalForwardOutputV0260> {
        validate_gradient_scale("RT", rt_encoder_gradient_scale)?;
        validate_gradient_scale("CCS", ccs_encoder_gradient_scale)?;
        let foundation = self.encoder.forward_t(batch, train)?;

        let rt_embedding =
            gradient_scaled_identity(&foundation.peptide_embedding, rt_encoder_gradient_scale)?;
        let rt = self.rt_adapter.forward(&rt_embedding)?;

        let ccs_embedding =
            gradient_scaled_identity(&foundation.peptide_embedding, ccs_encoder_gradient_scale)?;
        let scaled_charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let ccs_scalar_context = match self.config.ccs_context_mode {
            FoundationCcsContextMode::ChargePresence => {
                let present = context.charge_present.unsqueeze(1)?;
                Tensor::cat(&[&scaled_charge, &present], 1)?
            }
            FoundationCcsContextMode::NeutralMassCharge => {
                let neutral_mass = context.precursor_mz.broadcast_mul(&context.charge)?;
                let physical_present = context
                    .precursor_mz_present
                    .broadcast_mul(&context.charge_present)?;
                let scaled_mass = neutral_mass
                    .broadcast_mul(&physical_present)?
                    .affine(1.0 / 3000.0, 0.0)?
                    .unsqueeze(1)?;
                Tensor::cat(&[&scaled_mass, &scaled_charge], 1)?
            }
        };
        let ccs_features = Tensor::cat(&[&ccs_embedding, &ccs_scalar_context], 1)?;
        let ccs_residual = self.ccs_adapter.forward(&ccs_features)?;
        let ccs = if let Some(baseline) = &self.config.ccs_physics_baseline {
            let baseline = standardized_ccs_physics_baseline_v0260(&foundation, context, baseline)?;
            (&baseline + &ccs_residual)?
        } else {
            ccs_residual
        };

        let (ms2_presence_logits, ms2_positive_intensity, ms2) =
            self.ms2_head.forward(&foundation, context)?;
        let residue_logits = self.residue_head.forward(&foundation.residue_embeddings)?;
        let chemistry_reconstruction = self
            .chemistry_head
            .forward(&foundation.residue_embeddings)?;
        let contrastive_projection = self
            .contrastive_head
            .forward(&foundation.peptide_embedding)?;

        Ok(FoundationMultimodalForwardOutputV0260 {
            base: FoundationMultiTaskOutput {
                foundation,
                rt,
                ccs,
                ms2,
                residue_logits,
                chemistry_reconstruction,
                contrastive_projection,
            },
            ms2_presence_logits,
            ms2_positive_intensity,
        })
    }

    /// Shared peptide encoder.
    pub fn encoder(&self) -> &PeptideFoundationEncoder {
        &self.encoder
    }

    /// Forward architecture config.
    pub fn config(&self) -> &FoundationConfig {
        &self.config
    }
}

/// Full v0.26 multimodal model. Diffusion and causal auxiliaries share the same
/// spectrum encoder parameters used by the explicit relation head.
#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0260Model {
    forward: PeptideFoundationMultimodalForwardV0260,
    diffusion: PeptideSpectrumDiffusionModel,
    causal: PeptideSpectrumCausalModel,
    spectrum_encoder: FoundationSpectrumEncoder,
    spectrum_projection: Linear,
    relation_query_norm: FoundationLayerNorm,
    relation_cross_attention: MultiHeadCrossAttention,
    relation_hidden: Linear,
    relation_output: Linear,
    forward_config: FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
}

impl PeptideFoundationMultimodalV0260Model {
    /// Construct the full random-initialized multimodal architecture.
    pub fn new(
        forward_config: FoundationConfig,
        inverse_config: FoundationDiffusionConfig,
        property_hidden_dim: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        forward_config.validate().map_err(candle_core::Error::Msg)?;
        inverse_config.validate().map_err(candle_core::Error::Msg)?;
        if forward_config.model_dim != inverse_config.model_dim {
            candle_core::bail!(
                "v0.26 requires equal peptide/spectrum model_dim, got {} vs {}",
                forward_config.model_dim,
                inverse_config.model_dim
            );
        }
        let diffusion = PeptideSpectrumDiffusionModel::new(inverse_config.clone(), vb.clone())?;
        let causal = PeptideSpectrumCausalModel::new(inverse_config.clone(), vb.clone())?;
        // This resolves to the same `spectrum_encoder.*` variables instantiated
        // by the inverse auxiliaries, so relation/alignment training updates one
        // shared observed-spectrum representation.
        let spectrum_encoder =
            FoundationSpectrumEncoder::new(&inverse_config, vb.pp("spectrum_encoder"))?;
        let forward = PeptideFoundationMultimodalForwardV0260::new(
            forward_config.clone(),
            property_hidden_dim,
            vb.clone(),
        )?;
        let spectrum_projection = nn::linear(
            inverse_config.model_dim,
            forward_config.contrastive_dim,
            vb.pp("alignment.spectrum_projection"),
        )?;
        let relation_query_norm = FoundationLayerNorm::new(
            forward_config.model_dim,
            1e-5,
            vb.pp("multimodal_relation.query_norm"),
        )?;
        let relation_cross_attention = MultiHeadCrossAttention::new(
            forward_config.model_dim,
            forward_config.num_attention_heads,
            vb.pp("multimodal_relation.cross_attention"),
        )?;
        let relation_hidden = nn::linear(
            forward_config.model_dim * 5,
            forward_config.model_dim,
            vb.pp("multimodal_relation.hidden"),
        )?;
        let relation_output = nn::linear(
            forward_config.model_dim,
            1,
            vb.pp("multimodal_relation.output"),
        )?;
        Ok(Self {
            forward,
            diffusion,
            causal,
            spectrum_encoder,
            spectrum_projection,
            relation_query_norm,
            relation_cross_attention,
            relation_hidden,
            relation_output,
            forward_config,
            inverse_config,
        })
    }

    /// Peptide/property branch.
    pub fn forward(&self) -> &PeptideFoundationMultimodalForwardV0260 {
        &self.forward
    }

    /// Spectrum-conditioned diffusion auxiliary.
    pub fn diffusion(&self) -> &PeptideSpectrumDiffusionModel {
        &self.diffusion
    }

    /// Spectrum-conditioned causal auxiliary.
    pub fn causal(&self) -> &PeptideSpectrumCausalModel {
        &self.causal
    }

    /// Encode measured/library product-ion peaks with the shared spectrum encoder.
    pub fn encode_spectrum_t(
        &self,
        spectrum: &FoundationSpectrumBatch,
        train: bool,
    ) -> Result<FoundationSpectrumEncoding> {
        self.spectrum_encoder.forward_t(spectrum, train)
    }

    /// Project pooled spectrum embeddings into the peptide contrastive space.
    pub fn project_spectrum_embedding(&self, embedding: &Tensor) -> Result<Tensor> {
        self.spectrum_projection.forward(embedding)
    }

    /// Score one peptide representation against one observed-spectrum encoding.
    /// Rows are paired by batch index; this is used for matched and same-spectrum
    /// negative peptide candidates.
    pub fn relation_score(
        &self,
        peptide: &FoundationOutput,
        spectrum: &FoundationSpectrumEncoding,
    ) -> Result<Tensor> {
        let normalized = self
            .relation_query_norm
            .forward(&peptide.residue_embeddings)?;
        let attended = self.relation_cross_attention.forward(
            &normalized,
            &spectrum.peak_embeddings,
            &spectrum_peak_mask_from_embeddings(spectrum)?,
        )?;
        let fused = (&peptide.residue_embeddings + &attended)?;
        let pooled_relation = masked_mean(&fused, &peptide.residue_mask)?;
        let peptide_pooled = &peptide.peptide_embedding;
        let spectrum_pooled = &spectrum.spectrum_embedding;
        let difference = (peptide_pooled - spectrum_pooled)?.abs()?;
        let product = peptide_pooled.broadcast_mul(spectrum_pooled)?;
        let features = Tensor::cat(
            &[
                peptide_pooled,
                spectrum_pooled,
                &difference,
                &product,
                &pooled_relation,
            ],
            1,
        )?;
        let hidden = self.relation_hidden.forward(&features)?.relu()?;
        self.relation_output.forward(&hidden)
    }

    /// Forward architecture.
    pub fn forward_config(&self) -> &FoundationConfig {
        &self.forward_config
    }

    /// Inverse/spectrum architecture.
    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        &self.inverse_config
    }
}

/// Factorized v0.26 MS2 loss components.
#[derive(Debug, Clone)]
pub struct FoundationMultimodalMs2LossesV0260 {
    /// Weighted total used for optimization.
    pub total: Tensor,
    /// Binary fragment-presence loss over annotated channels.
    pub presence: Tensor,
    /// Positive-intensity MSE restricted to observed non-zero fragments.
    pub positive_intensity: Tensor,
    /// Expected-spectrum cosine loss.
    pub cosine: Tensor,
}

/// Factorized MS2 objective: presence BCE + positive intensity MSE + expected
/// spectrum cosine. The target/mask tensor shapes must match the model outputs.
pub fn foundation_multimodal_ms2_loss_v0260(
    output: &FoundationMultimodalForwardOutputV0260,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    presence_target: Option<&Tensor>,
    presence_mask: Option<&Tensor>,
) -> Result<FoundationMultimodalMs2LossesV0260> {
    let zero = output.ms2_presence_logits.affine(0.0, 0.0)?.sum_all()?;

    let (positive_intensity, cosine) = match (target, mask) {
        (Some(target), Some(mask)) => {
            if output.base.ms2.dims() != target.dims()
                || output.ms2_presence_logits.dims() != target.dims()
                || output.ms2_positive_intensity.dims() != target.dims()
            {
                candle_core::bail!(
                    "v0.26 MS2 shape mismatch: expected {:?}, target {:?}, presence {:?}, positive {:?}",
                    output.base.ms2.dims(),
                    target.dims(),
                    output.ms2_presence_logits.dims(),
                    output.ms2_positive_intensity.dims()
                );
            }
            let mask = mask.broadcast_as(target.dims())?;
            let sparse_positive = target.gt(1.0e-6)?.to_dtype(DType::F32)?;
            let positive_mask = mask.broadcast_mul(&sparse_positive)?;
            let positive_intensity =
                masked_mse(&output.ms2_positive_intensity, target, &positive_mask)?;
            let shape = foundation_ms2_loss(
                &output.base.ms2,
                target,
                &mask,
                FoundationMs2LossConfig {
                    pointwise_weight: 0.0,
                    cosine_weight: 1.0,
                    cosine_epsilon: 1.0e-8,
                },
            )?;
            (positive_intensity, shape.cosine)
        }
        (None, None) => (zero.clone(), zero.clone()),
        _ => candle_core::bail!(
            "v0.26 intensity target and mask must either both be present or both be absent"
        ),
    };

    let presence = match (presence_target, presence_mask) {
        (Some(target), Some(mask)) => {
            if output.ms2_presence_logits.dims() != target.dims() {
                candle_core::bail!(
                    "v0.26 presence shape mismatch: logits {:?}, target {:?}",
                    output.ms2_presence_logits.dims(),
                    target.dims()
                );
            }
            masked_bce_with_logits(&output.ms2_presence_logits, target, mask)?
        }
        (None, None) => zero,
        _ => candle_core::bail!(
            "v0.26 presence target and mask must either both be present or both be absent"
        ),
    };

    let total = ((presence.affine(FOUNDATION_MULTIMODAL_MS2_PRESENCE_WEIGHT_V0260, 0.0)?
        + positive_intensity.affine(FOUNDATION_MULTIMODAL_MS2_INTENSITY_WEIGHT_V0260, 0.0)?)?
        + cosine.affine(FOUNDATION_MULTIMODAL_MS2_COSINE_WEIGHT_V0260, 0.0)?)?;
    Ok(FoundationMultimodalMs2LossesV0260 {
        total,
        presence,
        positive_intensity,
        cosine,
    })
}

/// Pairwise same-spectrum relation margin. Lower loss means the matched peptide
/// receives a larger score than the negative peptide by at least `margin`.
pub fn foundation_multimodal_relation_margin_loss_v0260(
    positive_score: &Tensor,
    negative_score: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    if positive_score.dims() != negative_score.dims() {
        candle_core::bail!(
            "v0.26 relation score mismatch: positive {:?}, negative {:?}",
            positive_score.dims(),
            negative_score.dims()
        );
    }
    ((negative_score - positive_score)? + margin)?
        .relu()?
        .mean_all()
}

fn spectrum_peak_mask_from_embeddings(encoding: &FoundationSpectrumEncoding) -> Result<Tensor> {
    // The encoding intentionally does not retain its input mask. Peak embeddings
    // are exactly zero at padded positions after every spectrum-encoder stage,
    // so a non-zero L1 norm is a stable reconstruction of the mask.
    encoding
        .peak_embeddings
        .abs()?
        .sum(2)?
        .gt(1.0e-12)?
        .to_dtype(DType::F32)
}

fn masked_mean(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mask = mask.broadcast_as(prediction.dims())?;
    let squared = (prediction - target)?.sqr()?.broadcast_mul(&mask)?;
    let numerator = squared.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator)
}

fn masked_bce_with_logits(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // Stable BCE(logit, y) = max(logit,0) - logit*y + log(1+exp(-abs(logit))).
    let positive = logits.relu()?;
    let linear = logits.broadcast_mul(target)?;
    let tail = logits.abs()?.affine(-1.0, 0.0)?.exp()?;
    let tail = (tail + 1.0)?.log()?;
    let loss = ((positive - linear)? + tail)?;
    let mask = mask.broadcast_as(logits.dims())?;
    let numerator = loss.broadcast_mul(&mask)?.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator)
}

fn standardized_ccs_physics_baseline_v0260(
    foundation: &FoundationOutput,
    context: &PrecursorContextBatch,
    baseline: &FoundationCcsPhysicsBaselineConfig,
) -> Result<Tensor> {
    let batch_size = context.charge.dims1()?;
    let device = context.charge.device();
    let charge_present = &context.charge_present;
    let mz_present = &context.precursor_mz_present;
    let physical_present = charge_present.broadcast_mul(mz_present)?;
    let charge = context.charge.broadcast_mul(charge_present)?;
    let precursor_mz = context.precursor_mz.broadcast_mul(mz_present)?;
    let charge_squared = charge.sqr()?;
    let neutral_mass_proxy = charge
        .broadcast_mul(&precursor_mz)?
        .broadcast_mul(&physical_present)?;
    let sequence_len = foundation.residue_mask.sum(1)?;
    let ones = Tensor::ones(batch_size, DType::F32, device)?;
    let features = Tensor::cat(
        &[
            &ones.unsqueeze(1)?,
            &charge.affine(1.0 / 4.0, 0.0)?.unsqueeze(1)?,
            &charge_squared.affine(1.0 / 16.0, 0.0)?.unsqueeze(1)?,
            &precursor_mz.affine(1.0 / 1000.0, 0.0)?.unsqueeze(1)?,
            &neutral_mass_proxy.affine(1.0 / 3000.0, 0.0)?.unsqueeze(1)?,
            &sequence_len.affine(1.0 / 30.0, 0.0)?.unsqueeze(1)?,
            &charge_present.unsqueeze(1)?,
            &mz_present.unsqueeze(1)?,
        ],
        1,
    )?;
    let coefficients = Tensor::from_vec(
        baseline
            .coefficients_native
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        (FOUNDATION_CCS_PHYSICS_FEATURE_COUNT, 1),
        device,
    )?;
    let native = features.matmul(&coefficients)?;
    native.affine(
        1.0 / baseline.target_std_native,
        -baseline.target_mean_native / baseline.target_std_native,
    )
}

fn validate_gradient_scale(label: &str, scale: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&scale) || !scale.is_finite() {
        candle_core::bail!(
            "v0.26 {label} encoder gradient scale must be finite and within [0,1], got {scale}"
        );
    }
    Ok(())
}

fn zero_initialized_linear(in_dim: usize, out_dim: usize, vb: VarBuilder<'_>) -> Result<Linear> {
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
    fn v0260_model_has_compatible_peptide_and_spectrum_widths() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut forward = FoundationConfig::default();
        forward.model_dim = 48;
        forward.graph_hidden_dim = 32;
        forward.transformer_layers = 2;
        forward.transformer_ff_dim = 192;
        forward.num_attention_heads = 4;
        forward.contrastive_dim = 32;
        let mut inverse = FoundationDiffusionConfig::default();
        inverse.model_dim = 48;
        inverse.feed_forward_dim = 192;
        inverse.num_attention_heads = 4;
        inverse.spectrum_layers = 2;
        inverse.decoder_layers = 2;
        let model = PeptideFoundationMultimodalV0260Model::new(forward, inverse, 96, vb)
            .expect("construct v0.26 model");
        assert_eq!(model.forward_config().model_dim, 48);
        assert_eq!(model.inverse_config().model_dim, 48);
    }
}
