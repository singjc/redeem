//! v0.34 deep task-specialist forward continuation.
//!
//! v0.31 proved that RT/MS2 can improve without sacrificing CCS, but the paired
//! historical VALIDATION comparison closed only a minority of the gap to the
//! external AlphaPeptDeep reference. v0.34 therefore stops extending the shared
//! residue-refinement family and introduces two independent, substantially more
//! expressive forward specialists on top of the frozen v0.31 representation:
//!
//! * RT: multi-scale 3/5/7 local motif convolutions -> 256d sequence states ->
//!   three Transformer blocks -> learned-query pooling -> residual RT correction.
//! * MS2: independent 256d four-layer residue Transformer -> cleavage/channel
//!   decoder conditioned on exact fragment geometry, charge, NCE/instrument
//!   context -> residual presence/log-intensity correction.
//!
//! The complete v0.31 model is immutable in this lane. Both new outputs are
//! identity-initialized, so step 0 reproduces the accepted v0.31 forward model
//! exactly. CCS remains the exact protected v0.27/v0.31 path.

use super::config::FoundationConfig;
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
};
use super::layers::{FoundationLayerNorm, MultiHeadCrossAttention, PeptideTransformerBlock};
use super::model::{FoundationMultiTaskOutput, FoundationOutput, PrecursorContextBatch};
use super::multimodal_v0270::{
    foundation_multimodal_ms2_loss_v0270, FoundationMultimodalForwardOutputV0270,
    FoundationMultimodalMs2LossesV0270,
};
use super::multimodal_v0310::{
    FoundationFragmentContextBatchV0310, PeptideFoundationMultimodalV0310Config,
    PeptideFoundationMultimodalV0310Model,
};
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{self as nn, ops, Conv1d, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0340: &str =
    "v0.34-v031-frozen-deep-rt-ms2-specialists-256d";
pub const FOUNDATION_SPECIALIST_DIM_V0340: usize = 256;
pub const FOUNDATION_RT_LOCAL_CHANNELS_V0340: usize = 64;
pub const FOUNDATION_RT_TRANSFORMER_LAYERS_V0340: usize = 3;
pub const FOUNDATION_RT_ATTENTION_HEADS_V0340: usize = 8;
pub const FOUNDATION_RT_FF_DIM_V0340: usize = 1024;
pub const FOUNDATION_MS2_TRANSFORMER_LAYERS_V0340: usize = 4;
pub const FOUNDATION_MS2_ATTENTION_HEADS_V0340: usize = 8;
pub const FOUNDATION_MS2_FF_DIM_V0340: usize = 1024;
pub const FOUNDATION_MS2_DECODER_HIDDEN_V0340: usize = 256;
pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0340: usize =
    super::multimodal_v0310::FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0310;
pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0340: f64 =
    super::multimodal_v0310::FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0310;

const ION_SERIES_CLASSES_V0340: usize = 6;
const FRAGMENT_CHARGE_CLASSES_V0340: usize = 3;
const PRECURSOR_CHARGE_CLASSES_V0340: usize = 7;
const ION_SERIES_EMBED_DIM_V0340: usize = 16;
const FRAGMENT_CHARGE_EMBED_DIM_V0340: usize = 8;
const PRECURSOR_CHARGE_EMBED_DIM_V0340: usize = 8;
const INSTRUMENT_EMBED_DIM_V0340: usize = 24;
const FRAGMENT_CONTINUOUS_FEATURES_V0340: usize = 19;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationMultimodalV0340Config {
    pub base_v0310: PeptideFoundationMultimodalV0310Config,
    pub specialist_dim: usize,
    pub rt_local_channels: usize,
    pub rt_transformer_layers: usize,
    pub rt_attention_heads: usize,
    pub rt_ff_dim: usize,
    pub ms2_transformer_layers: usize,
    pub ms2_attention_heads: usize,
    pub ms2_ff_dim: usize,
    pub ms2_decoder_hidden: usize,
}

impl PeptideFoundationMultimodalV0340Config {
    pub fn fixed(base_v0310: PeptideFoundationMultimodalV0310Config) -> Result<Self> {
        base_v0310.validate()?;
        let config = Self {
            base_v0310,
            specialist_dim: FOUNDATION_SPECIALIST_DIM_V0340,
            rt_local_channels: FOUNDATION_RT_LOCAL_CHANNELS_V0340,
            rt_transformer_layers: FOUNDATION_RT_TRANSFORMER_LAYERS_V0340,
            rt_attention_heads: FOUNDATION_RT_ATTENTION_HEADS_V0340,
            rt_ff_dim: FOUNDATION_RT_FF_DIM_V0340,
            ms2_transformer_layers: FOUNDATION_MS2_TRANSFORMER_LAYERS_V0340,
            ms2_attention_heads: FOUNDATION_MS2_ATTENTION_HEADS_V0340,
            ms2_ff_dim: FOUNDATION_MS2_FF_DIM_V0340,
            ms2_decoder_hidden: FOUNDATION_MS2_DECODER_HIDDEN_V0340,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0310.validate()?;
        if self.forward().model_dim != 192 {
            candle_core::bail!("v0.34 requires the frozen 192d v0.31 base representation");
        }
        if self.specialist_dim != FOUNDATION_SPECIALIST_DIM_V0340
            || self.rt_local_channels != FOUNDATION_RT_LOCAL_CHANNELS_V0340
            || self.rt_transformer_layers != FOUNDATION_RT_TRANSFORMER_LAYERS_V0340
            || self.rt_attention_heads != FOUNDATION_RT_ATTENTION_HEADS_V0340
            || self.rt_ff_dim != FOUNDATION_RT_FF_DIM_V0340
            || self.ms2_transformer_layers != FOUNDATION_MS2_TRANSFORMER_LAYERS_V0340
            || self.ms2_attention_heads != FOUNDATION_MS2_ATTENTION_HEADS_V0340
            || self.ms2_ff_dim != FOUNDATION_MS2_FF_DIM_V0340
            || self.ms2_decoder_hidden != FOUNDATION_MS2_DECODER_HIDDEN_V0340
        {
            candle_core::bail!("v0.34 specialist dimensions differ from the fixed design");
        }
        if self.specialist_dim % self.rt_attention_heads != 0
            || self.specialist_dim % self.ms2_attention_heads != 0
        {
            candle_core::bail!("v0.34 specialist width must be divisible by attention heads");
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
struct MultiScaleRtStemV0340 {
    identity: Linear,
    conv3: Conv1d,
    conv5: Conv1d,
    conv7: Conv1d,
    norm: FoundationLayerNorm,
    local_channels: usize,
    output_dim: usize,
}

impl MultiScaleRtStemV0340 {
    fn new(config: &PeptideFoundationMultimodalV0340Config, vb: VarBuilder<'_>) -> Result<Self> {
        let input_dim = config.forward().model_dim;
        let local = config.rt_local_channels;
        let output_dim = local * 4;
        if output_dim != config.specialist_dim {
            candle_core::bail!("v0.34 RT local channels must concatenate to specialist_dim");
        }
        Ok(Self {
            identity: nn::linear(input_dim, local, vb.pp("identity"))?,
            conv3: nn::conv1d(
                input_dim,
                local,
                3,
                nn::Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv3"),
            )?,
            conv5: nn::conv1d(
                input_dim,
                local,
                5,
                nn::Conv1dConfig {
                    padding: 2,
                    ..Default::default()
                },
                vb.pp("conv5"),
            )?,
            conv7: nn::conv1d(
                input_dim,
                local,
                7,
                nn::Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
                vb.pp("conv7"),
            )?,
            norm: FoundationLayerNorm::new(output_dim, 1e-5, vb.pp("norm"))?,
            local_channels: local,
            output_dim,
        })
    }

    fn forward(&self, residues: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, sequence, _) = residues.dims3()?;
        let identity = self.identity.forward(residues)?.relu()?;
        let channels_first = residues.transpose(1, 2)?.contiguous()?;
        let conv3 = self
            .conv3
            .forward(&channels_first)?
            .transpose(1, 2)?
            .contiguous()?
            .relu()?;
        let conv5 = self
            .conv5
            .forward(&channels_first)?
            .transpose(1, 2)?
            .contiguous()?
            .relu()?;
        let conv7 = self
            .conv7
            .forward(&channels_first)?
            .transpose(1, 2)?
            .contiguous()?
            .relu()?;
        for tensor in [&identity, &conv3, &conv5, &conv7] {
            if tensor.dims3()? != (batch, sequence, self.local_channels) {
                candle_core::bail!("v0.34 RT multi-scale convolution shape mismatch");
            }
        }
        let hidden = Tensor::cat(&[&identity, &conv3, &conv5, &conv7], 2)?;
        let hidden = self.norm.forward(&hidden)?;
        let expanded = mask
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, self.output_dim))?;
        hidden.broadcast_mul(&expanded)
    }
}

#[derive(Clone)]
struct RtDeepSpecialistV0340 {
    stem: MultiScaleRtStemV0340,
    blocks: Vec<PeptideTransformerBlock>,
    learned_query: Embedding,
    query_norm: FoundationLayerNorm,
    pooling: MultiHeadCrossAttention,
    base_peptide_projection: Linear,
    head_norm: FoundationLayerNorm,
    hidden_512: Linear,
    hidden_256: Linear,
    delta_output: Linear,
    specialist_dim: usize,
}

impl RtDeepSpecialistV0340 {
    fn new(config: &PeptideFoundationMultimodalV0340Config, vb: VarBuilder<'_>) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.rt_transformer_layers);
        for layer in 0..config.rt_transformer_layers {
            blocks.push(PeptideTransformerBlock::new(
                config.specialist_dim,
                config.rt_attention_heads,
                config.rt_ff_dim,
                config.forward().dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        Ok(Self {
            stem: MultiScaleRtStemV0340::new(config, vb.pp("stem"))?,
            blocks,
            learned_query: nn::embedding(1, config.specialist_dim, vb.pp("learned_query"))?,
            query_norm: FoundationLayerNorm::new(config.specialist_dim, 1e-5, vb.pp("query_norm"))?,
            pooling: MultiHeadCrossAttention::new(
                config.specialist_dim,
                config.rt_attention_heads,
                vb.pp("attention_pool"),
            )?,
            base_peptide_projection: nn::linear(
                config.forward().model_dim,
                config.specialist_dim,
                vb.pp("base_peptide_projection"),
            )?,
            head_norm: FoundationLayerNorm::new(
                config.specialist_dim * 2,
                1e-5,
                vb.pp("head.norm"),
            )?,
            hidden_512: nn::linear(config.specialist_dim * 2, 512, vb.pp("head.hidden_512"))?,
            hidden_256: nn::linear(512, 256, vb.pp("head.hidden_256"))?,
            delta_output: zero_initialized_linear_v0340(256, 1, vb.pp("head.delta_output"))?,
            specialist_dim: config.specialist_dim,
        })
    }

    fn forward_t(&self, foundation: &FoundationOutput, train: bool) -> Result<Tensor> {
        let residues = foundation.residue_embeddings.detach();
        let mut hidden = self.stem.forward(&residues, &foundation.residue_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &foundation.residue_mask, train)?;
        }
        let (batch, _, width) = hidden.dims3()?;
        if width != self.specialist_dim {
            candle_core::bail!("v0.34 RT specialist width mismatch");
        }
        let query_ids = Tensor::zeros((batch, 1), DType::U32, hidden.device())?;
        let query = self
            .query_norm
            .forward(&self.learned_query.forward(&query_ids)?)?;
        let pooled = self
            .pooling
            .forward(&query, &hidden, &foundation.residue_mask)?
            .squeeze(1)?;
        let base_peptide = self
            .base_peptide_projection
            .forward(&foundation.peptide_embedding.detach())?
            .relu()?;
        let features = Tensor::cat(&[&pooled, &base_peptide], 1)?;
        let features = self.head_norm.forward(&features)?;
        let hidden = self.hidden_512.forward(&features)?.relu()?;
        let hidden = self.hidden_256.forward(&hidden)?.relu()?;
        self.delta_output.forward(&hidden)
    }
}

#[derive(Clone)]
struct Ms2DeepSpecialistV0340 {
    input_norm: FoundationLayerNorm,
    input_projection: Linear,
    blocks: Vec<PeptideTransformerBlock>,
    ion_series_embedding: Embedding,
    fragment_charge_embedding: Embedding,
    precursor_charge_embedding: Embedding,
    instrument_embedding: Embedding,
    token_norm: FoundationLayerNorm,
    token_projection: Linear,
    token_hidden: Linear,
    presence_delta: Linear,
    log_intensity_scale: Linear,
    specialist_dim: usize,
    channels: usize,
}

impl Ms2DeepSpecialistV0340 {
    fn new(config: &PeptideFoundationMultimodalV0340Config, vb: VarBuilder<'_>) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.ms2_transformer_layers);
        for layer in 0..config.ms2_transformer_layers {
            blocks.push(PeptideTransformerBlock::new(
                config.specialist_dim,
                config.ms2_attention_heads,
                config.ms2_ff_dim,
                config.forward().dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        let token_feature_dim = config.specialist_dim * 2
            + ION_SERIES_EMBED_DIM_V0340
            + FRAGMENT_CHARGE_EMBED_DIM_V0340
            + PRECURSOR_CHARGE_EMBED_DIM_V0340
            + INSTRUMENT_EMBED_DIM_V0340
            + FRAGMENT_CONTINUOUS_FEATURES_V0340;
        Ok(Self {
            input_norm: FoundationLayerNorm::new(
                config.forward().model_dim,
                1e-5,
                vb.pp("input_norm"),
            )?,
            input_projection: nn::linear(
                config.forward().model_dim,
                config.specialist_dim,
                vb.pp("input_projection"),
            )?,
            blocks,
            ion_series_embedding: nn::embedding(
                ION_SERIES_CLASSES_V0340,
                ION_SERIES_EMBED_DIM_V0340,
                vb.pp("ion_series_embedding"),
            )?,
            fragment_charge_embedding: nn::embedding(
                FRAGMENT_CHARGE_CLASSES_V0340,
                FRAGMENT_CHARGE_EMBED_DIM_V0340,
                vb.pp("fragment_charge_embedding"),
            )?,
            precursor_charge_embedding: nn::embedding(
                PRECURSOR_CHARGE_CLASSES_V0340,
                PRECURSOR_CHARGE_EMBED_DIM_V0340,
                vb.pp("precursor_charge_embedding"),
            )?,
            instrument_embedding: nn::embedding(
                config.forward().instrument_vocab_size,
                INSTRUMENT_EMBED_DIM_V0340,
                vb.pp("instrument_embedding"),
            )?,
            token_norm: FoundationLayerNorm::new(token_feature_dim, 1e-5, vb.pp("token_norm"))?,
            token_projection: nn::linear(
                token_feature_dim,
                config.ms2_decoder_hidden,
                vb.pp("token_projection"),
            )?,
            token_hidden: nn::linear(
                config.ms2_decoder_hidden,
                config.ms2_decoder_hidden,
                vb.pp("token_hidden"),
            )?,
            presence_delta: zero_initialized_linear_v0340(
                config.ms2_decoder_hidden,
                1,
                vb.pp("presence_delta"),
            )?,
            log_intensity_scale: zero_initialized_linear_v0340(
                config.ms2_decoder_hidden,
                1,
                vb.pp("log_intensity_scale"),
            )?,
            specialist_dim: config.specialist_dim,
            channels: config.forward().ms2_fragment_channels,
        })
    }

    fn forward_t(
        &self,
        foundation: &FoundationOutput,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0310,
        base_presence_logits: &Tensor,
        base_positive_intensity: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, sequence, _) = foundation.residue_embeddings.dims3()?;
        if sequence < 2 || fragment.channels != self.channels {
            candle_core::bail!("v0.34 MS2 specialist received incompatible fragment canvas");
        }
        let normalized = self
            .input_norm
            .forward(&foundation.residue_embeddings.detach())?;
        let mut hidden = self.input_projection.forward(&normalized)?.relu()?;
        let expanded_residue_mask = foundation.residue_mask.unsqueeze(2)?.broadcast_as((
            batch,
            sequence,
            self.specialist_dim,
        ))?;
        hidden = hidden.broadcast_mul(&expanded_residue_mask)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &foundation.residue_mask, train)?;
        }

        let channels = fragment.channels;
        let tokens = fragment.cleavage_count * channels;
        let left = hidden
            .narrow(1, 0, fragment.cleavage_count)?
            .unsqueeze(2)?
            .broadcast_as((
                batch,
                fragment.cleavage_count,
                channels,
                self.specialist_dim,
            ))?;
        let right = hidden
            .narrow(1, 1, fragment.cleavage_count)?
            .unsqueeze(2)?
            .broadcast_as((
                batch,
                fragment.cleavage_count,
                channels,
                self.specialist_dim,
            ))?;
        let cleavage =
            Tensor::cat(&[&left, &right], 3)?.reshape((batch, tokens, self.specialist_dim * 2))?;
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
            .broadcast_as((batch, tokens, INSTRUMENT_EMBED_DIM_V0340))?;
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
        let features = self.token_norm.forward(&features)?;
        let token_hidden = self.token_projection.forward(&features)?.relu()?;
        let token_hidden = self.token_hidden.forward(&token_hidden)?.relu()?;
        let (_, _, token_hidden_dim) = token_hidden.dims3()?;
        let token_mask =
            fragment
                .token_mask
                .unsqueeze(2)?
                .broadcast_as((batch, tokens, token_hidden_dim))?;
        let token_hidden = token_hidden.broadcast_mul(&token_mask)?;

        let delta_presence = self
            .presence_delta
            .forward(&token_hidden)?
            .squeeze(2)?
            .broadcast_mul(&fragment.token_mask)?
            .reshape((batch, fragment.cleavage_count, channels))?;
        let log_scale = self
            .log_intensity_scale
            .forward(&token_hidden)?
            .squeeze(2)?
            .broadcast_mul(&fragment.token_mask)?
            .clamp(-2.0, 2.0)?
            .reshape((batch, fragment.cleavage_count, channels))?;
        let delta_presence = pad_fragment_canvas_v0340(
            &delta_presence,
            batch,
            fragment.cleavage_count,
            fragment.output_cleavage_count,
            channels,
        )?;
        let log_scale = pad_fragment_canvas_v0340(
            &log_scale,
            batch,
            fragment.cleavage_count,
            fragment.output_cleavage_count,
            channels,
        )?;

        let presence_logits = (base_presence_logits.detach() + delta_presence)?;
        let intensity_scale = log_scale.exp()?;
        let positive_intensity = base_positive_intensity
            .detach()
            .broadcast_mul(&intensity_scale)?;
        let expected = ops::sigmoid(&presence_logits)?.broadcast_mul(&positive_intensity)?;
        Ok((presence_logits, positive_intensity, expected))
    }
}

#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0340 {
    pub base: FoundationMultiTaskOutput,
    pub ms2_presence_logits: Tensor,
    pub ms2_positive_intensity: Tensor,
}

#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0340Model {
    base_v0310: PeptideFoundationMultimodalV0310Model,
    rt_specialist_v0340: RtDeepSpecialistV0340,
    ms2_specialist_v0340: Ms2DeepSpecialistV0340,
    config: PeptideFoundationMultimodalV0340Config,
}

impl PeptideFoundationMultimodalV0340Model {
    pub fn new(config: PeptideFoundationMultimodalV0340Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let base_v0310 =
            PeptideFoundationMultimodalV0310Model::new(config.base_v0310.clone(), vb.clone())?;
        let rt_specialist_v0340 =
            RtDeepSpecialistV0340::new(&config, vb.pp("rt_specialist_v0340"))?;
        let ms2_specialist_v0340 =
            Ms2DeepSpecialistV0340::new(&config, vb.pp("ms2_specialist_v0340"))?;
        Ok(Self {
            base_v0310,
            rt_specialist_v0340,
            ms2_specialist_v0340,
            config,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_v0340_t_with_shared_gradient_scales(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0310,
        train: bool,
        _rt_encoder_gradient_scale: f64,
        _ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultimodalForwardOutputV0340> {
        self.forward_v0340_t(batch, context, fragment, train)
    }

    pub fn forward_v0340_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0310,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0340> {
        // The complete v0.31 parent is an immutable inference-mode teacher/base.
        let base = self
            .base_v0310
            .forward_v0310_t(batch, context, fragment, false)?;
        let foundation = detach_foundation_v0340(&base.base.foundation);
        let rt_delta = self.rt_specialist_v0340.forward_t(&foundation, train)?;
        let rt = (base.base.rt.detach() + rt_delta)?;
        let ccs = base.base.ccs.detach();
        let (ms2_presence_logits, ms2_positive_intensity, ms2) =
            self.ms2_specialist_v0340.forward_t(
                &foundation,
                context,
                fragment,
                &base.ms2_presence_logits,
                &base.ms2_positive_intensity,
                train,
            )?;
        Ok(FoundationMultimodalForwardOutputV0340 {
            base: FoundationMultiTaskOutput {
                foundation,
                rt,
                ccs,
                ms2,
                residue_logits: base.base.residue_logits.detach(),
                chemistry_reconstruction: base.base.chemistry_reconstruction.detach(),
                contrastive_projection: base.base.contrastive_projection.detach(),
            },
            ms2_presence_logits,
            ms2_positive_intensity,
        })
    }

    pub fn property_foundation_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<FoundationOutput> {
        self.base_v0310.property_foundation_t(batch, train)
    }

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

    pub fn config(&self) -> &PeptideFoundationMultimodalV0340Config {
        &self.config
    }
}

pub type FoundationFragmentContextBatchV0340 = FoundationFragmentContextBatchV0310;
pub type FoundationMultimodalMs2LossesV0340 = FoundationMultimodalMs2LossesV0270;

pub fn foundation_multimodal_relation_margin_loss_v0340(
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

pub fn foundation_multimodal_ms2_loss_v0340(
    output: &FoundationMultimodalForwardOutputV0340,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    presence_target: Option<&Tensor>,
    presence_mask: Option<&Tensor>,
) -> Result<FoundationMultimodalMs2LossesV0340> {
    let proxy = FoundationMultimodalForwardOutputV0270 {
        base: output.base.clone(),
        ms2_presence_logits: output.ms2_presence_logits.clone(),
        ms2_positive_intensity: output.ms2_positive_intensity.clone(),
    };
    foundation_multimodal_ms2_loss_v0270(&proxy, target, mask, presence_target, presence_mask)
}

fn detach_foundation_v0340(base: &FoundationOutput) -> FoundationOutput {
    FoundationOutput {
        residue_embeddings: base.residue_embeddings.detach(),
        peptide_embedding: base.peptide_embedding.detach(),
        residue_mask: base.residue_mask.clone(),
        chemistry_targets: base.chemistry_targets.clone(),
    }
}

fn pad_fragment_canvas_v0340(
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
        candle_core::bail!("v0.34 active fragment canvas exceeds output width");
    }
    let pad = Tensor::zeros(
        (batch, output_cleavages - active_cleavages, channels),
        active.dtype(),
        active.device(),
    )?;
    Tensor::cat(&[active, &pad], 1)
}

fn zero_initialized_linear_v0340(
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

    fn config() -> PeptideFoundationMultimodalV0340Config {
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
        PeptideFoundationMultimodalV0340Config::fixed(v0310).unwrap()
    }

    #[test]
    fn v0340_config_is_fixed_deep_specialist_design() {
        let config = config();
        assert_eq!(config.specialist_dim, 256);
        assert_eq!(config.rt_transformer_layers, 3);
        assert_eq!(config.ms2_transformer_layers, 4);
        assert_eq!(config.rt_attention_heads, 8);
        assert_eq!(config.ms2_attention_heads, 8);
    }

    #[test]
    fn v0340_identity_correction_heads_are_zero_initialized() {
        let varmap = VarMap::new();
        let _model = PeptideFoundationMultimodalV0340Model::new(
            config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let data = varmap.data().lock().unwrap();
        for name in [
            "rt_specialist_v0340.head.delta_output.weight",
            "ms2_specialist_v0340.presence_delta.weight",
            "ms2_specialist_v0340.log_intensity_scale.weight",
        ] {
            let values = data
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(values.iter().all(|&value| value == 0.0), "{name}");
        }
    }
}
