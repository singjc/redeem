//! v0.27 multimodal peptide/spectrum foundation architecture.
//!
//! v0.27 preserves the accepted v0.26.1 shared peptide encoder, CCS physical
//! residual path, observed-spectrum encoder, inverse diffusion/causal models,
//! and same-spectrum relation objective. It changes only the two forward paths
//! identified by the frozen historical VALIDATION comparison:
//!
//! * RT receives two residue-level specialist Transformer blocks plus learned-
//!   query attention pooling before regression.
//! * MS2 is decoded from one token per theoretical cleavage/channel, with exact
//!   open-PTM fragment geometry and two contextual fragment Transformer blocks.
//!   Presence and positive conditional intensity remain factorized exactly as in
//!   v0.26.1.

use super::causal::PeptideSpectrumCausalModel;
use super::ccs_physics::FOUNDATION_CCS_PHYSICS_FEATURE_COUNT;
use super::config::{
    FoundationCcsContextMode, FoundationCcsPhysicsBaselineConfig, FoundationConfig,
    FoundationMs2OutputActivation,
};
use super::data::FoundationTrainingRecord;
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoder, FoundationSpectrumEncoding,
    PeptideSpectrumDiffusionModel, FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use super::featurize::FoundationModificationSite;
use super::fragment_relation::foundation_fragment_cleavage_geometry;
use super::layers::{FoundationLayerNorm, MultiHeadCrossAttention, PeptideTransformerBlock};
use super::loss::{foundation_ms2_loss, FoundationMs2LossConfig};
use super::model::{
    apply_ms2_output_activation, gradient_scaled_identity, FoundationMultiTaskOutput,
    FoundationOutput, PeptideFoundationEncoder, PrecursorContextBatch,
};
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, ops, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

/// Stable v0.27 architecture identifier.
pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0270: &str =
    "v0.27-192d-rt-specialist-contextual-fragment-transformer-openptm32";
/// Frozen shared/property hidden width inherited from v0.26.1.
pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270: usize = 384;
/// Fixed v0.27 RT specialist depth.
pub const FOUNDATION_RT_SPECIALIST_LAYERS_V0270: usize = 2;
/// Fixed v0.27 RT specialist attention heads.
pub const FOUNDATION_RT_SPECIALIST_HEADS_V0270: usize = 6;
/// Fixed v0.27 RT specialist FFN width.
pub const FOUNDATION_RT_SPECIALIST_FF_DIM_V0270: usize = 768;
/// Fixed v0.27 fragment contextualizer depth.
pub const FOUNDATION_FRAGMENT_TRANSFORMER_LAYERS_V0270: usize = 2;
/// Fixed v0.27 fragment contextualizer attention heads.
pub const FOUNDATION_FRAGMENT_TRANSFORMER_HEADS_V0270: usize = 6;
/// Fixed v0.27 fragment contextualizer FFN width.
pub const FOUNDATION_FRAGMENT_TRANSFORMER_FF_DIM_V0270: usize = 768;
/// Current forward-MS2 channel contract inherited from v0.26.1.
pub const FOUNDATION_FRAGMENT_CHANNELS_V0270: usize = 8;
/// Number of deterministic continuous features attached to every fragment token.
pub const FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270: usize = 19;
/// Same-spectrum relation margin inherited unchanged from v0.26.1.
pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270: f64 = 0.25;
/// Presence loss weight inherited unchanged from v0.26.1.
pub const FOUNDATION_MULTIMODAL_MS2_PRESENCE_WEIGHT_V0270: f64 = 0.25;
/// Positive-intensity loss weight inherited unchanged from v0.26.1.
pub const FOUNDATION_MULTIMODAL_MS2_INTENSITY_WEIGHT_V0270: f64 = 1.0;
/// Expected-spectrum cosine loss weight inherited unchanged from v0.26.1.
pub const FOUNDATION_MULTIMODAL_MS2_COSINE_WEIGHT_V0270: f64 = 0.25;

const PROTON_MASS_DA: f64 = 1.007_276_466_77;
const AMMONIA_MASS_DA: f64 = 17.026_549_101;
const ION_SERIES_CLASSES_V0270: usize = 6;
const FRAGMENT_CHARGE_CLASSES_V0270: usize = 3;
const PRECURSOR_CHARGE_CLASSES_V0270: usize = 7;
const ION_SERIES_EMBED_DIM_V0270: usize = 24;
const FRAGMENT_CHARGE_EMBED_DIM_V0270: usize = 8;
const PRECURSOR_CHARGE_EMBED_DIM_V0270: usize = 8;
const INSTRUMENT_EMBED_DIM_V0270: usize = 32;

/// Complete fixed v0.27 architecture configuration.
///
/// The explicit specialist fields are serialized into checkpoint metadata so a
/// future reader does not need to infer v0.27 semantics from the base configs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationMultimodalV0270Config {
    /// Shared peptide/property configuration inherited from v0.26.1.
    pub forward: FoundationConfig,
    /// Observed-spectrum/inverse configuration inherited from v0.26.1.
    pub inverse: FoundationDiffusionConfig,
    /// RT specialist Transformer depth (fixed to 2).
    pub rt_transformer_layers: usize,
    /// RT specialist attention heads (fixed to 6).
    pub rt_attention_heads: usize,
    /// RT specialist FFN width (fixed to 768).
    pub rt_ff_dim: usize,
    /// Fragment contextualizer depth (fixed to 2).
    pub fragment_transformer_layers: usize,
    /// Fragment contextualizer attention heads (fixed to 6).
    pub fragment_attention_heads: usize,
    /// Fragment contextualizer FFN width (fixed to 768).
    pub fragment_ff_dim: usize,
    /// Scalar-property hidden width (fixed to 384).
    pub property_hidden_dim: usize,
}

impl PeptideFoundationMultimodalV0270Config {
    /// Construct the single predeclared v0.27 architecture from accepted base configs.
    pub fn fixed(forward: FoundationConfig, inverse: FoundationDiffusionConfig) -> Result<Self> {
        let config = Self {
            forward,
            inverse,
            rt_transformer_layers: FOUNDATION_RT_SPECIALIST_LAYERS_V0270,
            rt_attention_heads: FOUNDATION_RT_SPECIALIST_HEADS_V0270,
            rt_ff_dim: FOUNDATION_RT_SPECIALIST_FF_DIM_V0270,
            fragment_transformer_layers: FOUNDATION_FRAGMENT_TRANSFORMER_LAYERS_V0270,
            fragment_attention_heads: FOUNDATION_FRAGMENT_TRANSFORMER_HEADS_V0270,
            fragment_ff_dim: FOUNDATION_FRAGMENT_TRANSFORMER_FF_DIM_V0270,
            property_hidden_dim: FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate that no silent architecture sweep has changed the frozen design.
    pub fn validate(&self) -> Result<()> {
        self.forward.validate().map_err(candle_core::Error::Msg)?;
        self.inverse.validate().map_err(candle_core::Error::Msg)?;
        if self.forward.model_dim != 192 || self.inverse.model_dim != 192 {
            candle_core::bail!(
                "v0.27 is fixed at model_dim=192, got forward={} inverse={}",
                self.forward.model_dim,
                self.inverse.model_dim
            );
        }
        if self.forward.model_dim != self.inverse.model_dim {
            candle_core::bail!("v0.27 requires equal peptide/spectrum model widths");
        }
        if self.forward.ms2_fragment_channels != FOUNDATION_FRAGMENT_CHANNELS_V0270 {
            candle_core::bail!(
                "v0.27 requires exactly {} forward-MS2 channels, got {}",
                FOUNDATION_FRAGMENT_CHANNELS_V0270,
                self.forward.ms2_fragment_channels
            );
        }
        if self.rt_transformer_layers != FOUNDATION_RT_SPECIALIST_LAYERS_V0270
            || self.rt_attention_heads != FOUNDATION_RT_SPECIALIST_HEADS_V0270
            || self.rt_ff_dim != FOUNDATION_RT_SPECIALIST_FF_DIM_V0270
            || self.fragment_transformer_layers != FOUNDATION_FRAGMENT_TRANSFORMER_LAYERS_V0270
            || self.fragment_attention_heads != FOUNDATION_FRAGMENT_TRANSFORMER_HEADS_V0270
            || self.fragment_ff_dim != FOUNDATION_FRAGMENT_TRANSFORMER_FF_DIM_V0270
            || self.property_hidden_dim != FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270
        {
            candle_core::bail!("v0.27 specialist dimensions differ from the frozen architecture");
        }
        if self.forward.model_dim % self.rt_attention_heads != 0
            || self.forward.model_dim % self.fragment_attention_heads != 0
        {
            candle_core::bail!("v0.27 model width must be divisible by six specialist heads");
        }
        Ok(())
    }
}

/// Deterministic runtime fragment metadata for one forward batch.
///
/// Nothing here is persisted in the prepared manifest: it is reconstructed from
/// the existing peptidoform/context records, keeping the v0.26 split reusable.
#[derive(Debug, Clone)]
pub struct FoundationFragmentContextBatchV0270 {
    /// Ion-series/loss ids `[batch, fragment_tokens]`.
    pub ion_series_ids: Tensor,
    /// Fragment-charge ids `[batch, fragment_tokens]`; zero means legacy loss channel with unspecified charge.
    pub fragment_charge_ids: Tensor,
    /// Precursor-charge ids `[batch, fragment_tokens]`; zero means unavailable/out of range.
    pub precursor_charge_ids: Tensor,
    /// Continuous geometry/context features `[batch, fragment_tokens, 19]`.
    pub continuous_features: Tensor,
    /// Valid theoretical-token mask `[batch, fragment_tokens]`.
    pub token_mask: Tensor,
    /// Number of active cleavage slots contextualized in this batch.
    pub cleavage_count: usize,
    /// Historical fixed output cleavage width (`max_sequence_len - 1`).
    pub output_cleavage_count: usize,
    /// Number of channels per cleavage.
    pub channels: usize,
}

impl FoundationFragmentContextBatchV0270 {
    /// Build contextual fragment metadata deterministically from existing records.
    pub fn from_records(
        records: &[FoundationTrainingRecord],
        config: &FoundationConfig,
        device: &Device,
    ) -> Result<Self> {
        if config.max_sequence_len < 2 {
            candle_core::bail!("v0.27 fragment context requires max_sequence_len >= 2");
        }
        if config.ms2_fragment_channels != FOUNDATION_FRAGMENT_CHANNELS_V0270 {
            candle_core::bail!("v0.27 fragment context requires the frozen 8-channel layout");
        }
        let batch = records.len();
        if batch == 0 {
            candle_core::bail!("v0.27 fragment context cannot be built for an empty batch");
        }
        let output_cleavage_count = config.max_sequence_len - 1;
        // Attention is cropped to the longest real peptide in this batch. Invalid
        // padded tokens never participate, but outputs are padded back to the
        // historical fixed target shape after contextualization.
        let cleavage_count = records
            .iter()
            .map(|record| {
                record
                    .peptidoform
                    .sequence
                    .chars()
                    .count()
                    .saturating_sub(1)
            })
            .max()
            .unwrap_or(1)
            .clamp(1, output_cleavage_count);
        let channels = config.ms2_fragment_channels;
        let tokens = cleavage_count * channels;
        let mut ion_series_ids = vec![0u32; batch * tokens];
        let mut fragment_charge_ids = vec![0u32; batch * tokens];
        let mut precursor_charge_ids = vec![0u32; batch * tokens];
        let mut continuous =
            vec![0.0f32; batch * tokens * FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270];
        let mut mask = vec![0.0f32; batch * tokens];

        for (batch_index, record) in records.iter().enumerate() {
            let residues = record.peptidoform.sequence.chars().collect::<Vec<_>>();
            if residues.len() < 2 || residues.len() > config.max_sequence_len {
                continue;
            }
            let geometry = foundation_fragment_cleavage_geometry(&record.peptidoform)
                .map_err(candle_core::Error::Msg)?;
            let (local_mod_mass, local_mod_flag) =
                residue_modification_state(record, residues.len())?;
            let precursor_charge = record.context.charge.filter(|z| *z > 0);
            let precursor_charge_id = precursor_charge
                .and_then(|z| usize::try_from(z).ok())
                .filter(|z| *z < PRECURSOR_CHARGE_CLASSES_V0270)
                .unwrap_or(0) as u32;
            let precursor_charge_scaled = precursor_charge
                .map(|z| (z as f32 / 6.0).clamp(0.0, 2.0))
                .unwrap_or(0.0);
            let precursor_charge_present = precursor_charge.is_some() as u8 as f32;
            let nce = record.context.nce.filter(|value| value.is_finite());
            let nce_scaled = nce.map(|value| value / 100.0).unwrap_or(0.0);
            let nce_present = nce.is_some() as u8 as f32;
            let instrument_present = record
                .context
                .instrument_id
                .is_some_and(|instrument| instrument > 0) as u8
                as f32;
            let peptide_len = residues.len() as f32;
            let total_mass = geometry
                .first()
                .map(|first| first.prefix_mass_da + first.suffix_with_water_mass_da)
                .unwrap_or(1.0)
                .max(1.0);

            for cleavage in geometry {
                let left_index = cleavage.cleavage_index;
                let right_index = left_index + 1;
                for channel in 0..channels {
                    let spec = fragment_channel_spec(channel, &cleavage)?;
                    let token_index = cleavage.cleavage_index * channels + channel;
                    let flat_index = batch_index * tokens + token_index;
                    let charge_valid = match (spec.fragment_charge, precursor_charge) {
                        (Some(2), Some(1)) => false,
                        _ => true,
                    };
                    let mass_valid = spec.theoretical_mz.is_finite()
                        && spec.theoretical_mz > 0.0
                        && spec.fragment_neutral_mass.is_finite()
                        && spec.fragment_neutral_mass > 0.0;
                    if !(charge_valid && mass_valid) {
                        continue;
                    }
                    mask[flat_index] = 1.0;
                    ion_series_ids[flat_index] = spec.ion_series_id;
                    fragment_charge_ids[flat_index] = spec.fragment_charge.unwrap_or(0) as u32;
                    precursor_charge_ids[flat_index] = precursor_charge_id;

                    let fragment_fraction = (spec.fragment_neutral_mass / total_mass) as f32;
                    let complement_fraction = (spec.complement_neutral_mass / total_mass) as f32;
                    let feature_values = [
                        (cleavage.cleavage_index as f32 + 1.0) / peptide_len,
                        (residues.len() - cleavage.cleavage_index - 1) as f32 / peptide_len,
                        peptide_len / config.max_sequence_len as f32,
                        precursor_charge_scaled,
                        precursor_charge_present,
                        nce_scaled,
                        nce_present,
                        (spec.theoretical_mz / 2000.0) as f32,
                        (spec.complement_mz / 2000.0) as f32,
                        (spec.fragment_neutral_mass / 3000.0) as f32,
                        (spec.complement_neutral_mass / 3000.0) as f32,
                        fragment_fraction,
                        complement_fraction,
                        (local_mod_mass[left_index] / 300.0).clamp(-2.0, 2.0),
                        (local_mod_mass[right_index] / 300.0).clamp(-2.0, 2.0),
                        local_mod_flag[left_index],
                        local_mod_flag[right_index],
                        spec.fragment_charge.is_some() as u8 as f32,
                        instrument_present,
                    ];
                    let base = flat_index * FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270;
                    continuous[base..base + FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270]
                        .copy_from_slice(&feature_values);
                }
            }
        }

        Ok(Self {
            ion_series_ids: Tensor::from_vec(ion_series_ids, (batch, tokens), device)?
                .to_dtype(DType::U32)?,
            fragment_charge_ids: Tensor::from_vec(fragment_charge_ids, (batch, tokens), device)?
                .to_dtype(DType::U32)?,
            precursor_charge_ids: Tensor::from_vec(precursor_charge_ids, (batch, tokens), device)?
                .to_dtype(DType::U32)?,
            continuous_features: Tensor::from_vec(
                continuous,
                (batch, tokens, FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270),
                device,
            )?,
            token_mask: Tensor::from_vec(mask, (batch, tokens), device)?,
            cleavage_count,
            output_cleavage_count,
            channels,
        })
    }

    /// Return the valid-token mask reshaped to the historical MS2 tensor surface.
    pub fn channel_mask(&self) -> Result<Tensor> {
        let (batch, _) = self.token_mask.dims2()?;
        let active = self
            .token_mask
            .reshape((batch, self.cleavage_count, self.channels))?;
        pad_fragment_canvas(
            &active,
            batch,
            self.cleavage_count,
            self.output_cleavage_count,
            self.channels,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct FragmentChannelSpecV0270 {
    ion_series_id: u32,
    fragment_charge: Option<usize>,
    theoretical_mz: f64,
    complement_mz: f64,
    fragment_neutral_mass: f64,
    complement_neutral_mass: f64,
}

fn fragment_channel_spec(
    channel: usize,
    geometry: &super::fragment_relation::FoundationFragmentCleavageGeometry,
) -> Result<FragmentChannelSpecV0270> {
    let prefix = geometry.prefix_mass_da;
    let suffix = geometry.suffix_with_water_mass_da;
    let spec = match channel {
        0 => FragmentChannelSpecV0270 {
            ion_series_id: 0,
            fragment_charge: Some(1),
            theoretical_mz: geometry.core_mz[0],
            complement_mz: geometry.core_mz[2],
            fragment_neutral_mass: prefix,
            complement_neutral_mass: suffix,
        },
        1 => FragmentChannelSpecV0270 {
            ion_series_id: 0,
            fragment_charge: Some(2),
            theoretical_mz: geometry.core_mz[1],
            complement_mz: geometry.core_mz[3],
            fragment_neutral_mass: prefix,
            complement_neutral_mass: suffix,
        },
        2 => FragmentChannelSpecV0270 {
            ion_series_id: 1,
            fragment_charge: Some(1),
            theoretical_mz: geometry.core_mz[2],
            complement_mz: geometry.core_mz[0],
            fragment_neutral_mass: suffix,
            complement_neutral_mass: prefix,
        },
        3 => FragmentChannelSpecV0270 {
            ion_series_id: 1,
            fragment_charge: Some(2),
            theoretical_mz: geometry.core_mz[3],
            complement_mz: geometry.core_mz[1],
            fragment_neutral_mass: suffix,
            complement_neutral_mass: prefix,
        },
        // Historical channels 4-7 aggregate neutral-loss observations without
        // retaining product charge. Their charge embedding is therefore the
        // explicit "unspecified" class (0), while mass features use the exact
        // neutral-loss mass and a singly protonated representative m/z.
        4 => loss_channel_spec(
            2,
            prefix - FOUNDATION_PEPTIDE_WATER_MASS_DA,
            geometry.core_mz[2],
            suffix,
        ),
        5 => loss_channel_spec(
            3,
            suffix - FOUNDATION_PEPTIDE_WATER_MASS_DA,
            geometry.core_mz[0],
            prefix,
        ),
        6 => loss_channel_spec(4, prefix - AMMONIA_MASS_DA, geometry.core_mz[2], suffix),
        7 => loss_channel_spec(5, suffix - AMMONIA_MASS_DA, geometry.core_mz[0], prefix),
        _ => candle_core::bail!("unsupported v0.27 fragment channel {channel}"),
    };
    Ok(spec)
}

fn loss_channel_spec(
    ion_series_id: u32,
    neutral_mass: f64,
    complement_mz: f64,
    complement_neutral_mass: f64,
) -> FragmentChannelSpecV0270 {
    FragmentChannelSpecV0270 {
        ion_series_id,
        fragment_charge: None,
        theoretical_mz: neutral_mass + PROTON_MASS_DA,
        complement_mz,
        fragment_neutral_mass: neutral_mass,
        complement_neutral_mass,
    }
}

fn residue_modification_state(
    record: &FoundationTrainingRecord,
    residue_count: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut masses = vec![0.0f32; residue_count];
    let mut flags = vec![0.0f32; residue_count];
    for modification in &record.peptidoform.modifications {
        if !modification.mass_delta.is_finite() {
            candle_core::bail!("v0.27 encountered non-finite PTM mass");
        }
        let index = match modification.site {
            FoundationModificationSite::Residue(index) => index,
            FoundationModificationSite::NTerm => 0,
            FoundationModificationSite::CTerm => residue_count.saturating_sub(1),
        };
        if index >= residue_count {
            candle_core::bail!("v0.27 PTM index {index} exceeds peptide length {residue_count}");
        }
        masses[index] += modification.mass_delta;
        flags[index] = 1.0;
    }
    Ok((masses, flags))
}

#[derive(Clone)]
struct CcsPropertyAdapterV0270 {
    norm: FoundationLayerNorm,
    hidden_in: Linear,
    hidden_out: Linear,
    output: Linear,
}

impl CcsPropertyAdapterV0270 {
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
struct RtSpecialistV0270 {
    blocks: Vec<PeptideTransformerBlock>,
    learned_query: Embedding,
    query_norm: FoundationLayerNorm,
    pooling: MultiHeadCrossAttention,
    head_norm: FoundationLayerNorm,
    hidden_384: Linear,
    hidden_192: Linear,
    output: Linear,
}

impl RtSpecialistV0270 {
    fn new(config: &PeptideFoundationMultimodalV0270Config, vb: VarBuilder<'_>) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.rt_transformer_layers);
        for layer in 0..config.rt_transformer_layers {
            blocks.push(PeptideTransformerBlock::new(
                config.forward.model_dim,
                config.rt_attention_heads,
                config.rt_ff_dim,
                config.forward.dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        Ok(Self {
            blocks,
            learned_query: nn::embedding(1, config.forward.model_dim, vb.pp("learned_query"))?,
            query_norm: FoundationLayerNorm::new(
                config.forward.model_dim,
                1e-5,
                vb.pp("query_norm"),
            )?,
            pooling: MultiHeadCrossAttention::new(
                config.forward.model_dim,
                config.rt_attention_heads,
                vb.pp("attention_pool"),
            )?,
            head_norm: FoundationLayerNorm::new(
                config.forward.model_dim * 2,
                1e-5,
                vb.pp("head.norm"),
            )?,
            hidden_384: nn::linear(config.forward.model_dim * 2, 384, vb.pp("head.hidden_384"))?,
            hidden_192: nn::linear(384, 192, vb.pp("head.hidden_192"))?,
            output: zero_initialized_linear(192, 1, vb.pp("head.output"))?,
        })
    }

    fn forward_t(
        &self,
        foundation: &FoundationOutput,
        train: bool,
        shared_gradient_scale: f64,
    ) -> Result<Tensor> {
        validate_gradient_scale("RT", shared_gradient_scale)?;
        let mut hidden =
            gradient_scaled_identity(&foundation.residue_embeddings, shared_gradient_scale)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, &foundation.residue_mask, train)?;
        }
        let (batch, _, _) = hidden.dims3()?;
        let query_ids = Tensor::zeros((batch, 1), DType::U32, hidden.device())?;
        let query = self.learned_query.forward(&query_ids)?;
        let query = self.query_norm.forward(&query)?;
        let pooled = self
            .pooling
            .forward(&query, &hidden, &foundation.residue_mask)?
            .squeeze(1)?;
        let shared =
            gradient_scaled_identity(&foundation.peptide_embedding, shared_gradient_scale)?;
        let features = Tensor::cat(&[&pooled, &shared], 1)?;
        let normalized = self.head_norm.forward(&features)?;
        let hidden = self.hidden_384.forward(&normalized)?.relu()?;
        let hidden = self.hidden_192.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)
    }
}

#[derive(Clone)]
struct ContextualFragmentDecoderV0270 {
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
    model_dim: usize,
}

impl ContextualFragmentDecoderV0270 {
    fn new(config: &PeptideFoundationMultimodalV0270Config, vb: VarBuilder<'_>) -> Result<Self> {
        let feature_dim = config.forward.model_dim * 2
            + ION_SERIES_EMBED_DIM_V0270
            + FRAGMENT_CHARGE_EMBED_DIM_V0270
            + PRECURSOR_CHARGE_EMBED_DIM_V0270
            + INSTRUMENT_EMBED_DIM_V0270
            + FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270;
        let mut blocks = Vec::with_capacity(config.fragment_transformer_layers);
        for layer in 0..config.fragment_transformer_layers {
            blocks.push(PeptideTransformerBlock::new(
                config.forward.model_dim,
                config.fragment_attention_heads,
                config.fragment_ff_dim,
                config.forward.dropout,
                vb.pp(format!("transformer.{layer}")),
            )?);
        }
        Ok(Self {
            ion_series_embedding: nn::embedding(
                ION_SERIES_CLASSES_V0270,
                ION_SERIES_EMBED_DIM_V0270,
                vb.pp("ion_series_embedding"),
            )?,
            fragment_charge_embedding: nn::embedding(
                FRAGMENT_CHARGE_CLASSES_V0270,
                FRAGMENT_CHARGE_EMBED_DIM_V0270,
                vb.pp("fragment_charge_embedding"),
            )?,
            precursor_charge_embedding: nn::embedding(
                PRECURSOR_CHARGE_CLASSES_V0270,
                PRECURSOR_CHARGE_EMBED_DIM_V0270,
                vb.pp("precursor_charge_embedding"),
            )?,
            instrument_embedding: nn::embedding(
                config.forward.instrument_vocab_size,
                INSTRUMENT_EMBED_DIM_V0270,
                vb.pp("instrument_embedding"),
            )?,
            feature_norm: FoundationLayerNorm::new(feature_dim, 1e-5, vb.pp("feature_norm"))?,
            token_projection: nn::linear(
                feature_dim,
                config.forward.model_dim,
                vb.pp("token_projection"),
            )?,
            blocks,
            output_norm: FoundationLayerNorm::new(
                config.forward.model_dim,
                1e-5,
                vb.pp("output_norm"),
            )?,
            presence_head: nn::linear(config.forward.model_dim, 1, vb.pp("presence"))?,
            intensity_head: nn::linear(config.forward.model_dim, 1, vb.pp("intensity"))?,
            model_dim: config.forward.model_dim,
        })
    }

    fn forward_t(
        &self,
        foundation: &FoundationOutput,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
        if sequence < 2 || model_dim != self.model_dim {
            candle_core::bail!("v0.27 fragment decoder received incompatible peptide states");
        }
        if fragment.cleavage_count > sequence - 1 || fragment.output_cleavage_count != sequence - 1
        {
            candle_core::bail!(
                "v0.27 fragment context widths active={} output={} are incompatible with encoder width {}",
                fragment.cleavage_count,
                fragment.output_cleavage_count,
                sequence - 1
            );
        }
        let channels = fragment.channels;
        let tokens = fragment.cleavage_count * channels;
        if fragment.token_mask.dims2()? != (batch, tokens) {
            candle_core::bail!("v0.27 fragment token mask shape mismatch");
        }

        let left = foundation
            .residue_embeddings
            .narrow(1, 0, fragment.cleavage_count)?
            .unsqueeze(2)?
            .broadcast_as((batch, fragment.cleavage_count, channels, model_dim))?;
        let right = foundation
            .residue_embeddings
            .narrow(1, 1, fragment.cleavage_count)?
            .unsqueeze(2)?
            .broadcast_as((batch, fragment.cleavage_count, channels, model_dim))?;
        let cleavage = Tensor::cat(&[&left, &right], 3)?.reshape((batch, tokens, model_dim * 2))?;
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
            .broadcast_as((batch, tokens, INSTRUMENT_EMBED_DIM_V0270))?;
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
            .broadcast_as((batch, tokens, model_dim))?;
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
        let expected = probability.broadcast_mul(&positive_intensity)?;
        let presence_logits = presence_logits.broadcast_mul(&fragment.token_mask)?;
        let positive_intensity = positive_intensity.broadcast_mul(&fragment.token_mask)?;
        let expected = expected.broadcast_mul(&fragment.token_mask)?;
        let presence_logits =
            presence_logits.reshape((batch, fragment.cleavage_count, channels))?;
        let positive_intensity =
            positive_intensity.reshape((batch, fragment.cleavage_count, channels))?;
        let expected = expected.reshape((batch, fragment.cleavage_count, channels))?;
        Ok((
            pad_fragment_canvas(
                &presence_logits,
                batch,
                fragment.cleavage_count,
                fragment.output_cleavage_count,
                channels,
            )?,
            pad_fragment_canvas(
                &positive_intensity,
                batch,
                fragment.cleavage_count,
                fragment.output_cleavage_count,
                channels,
            )?,
            pad_fragment_canvas(
                &expected,
                batch,
                fragment.cleavage_count,
                fragment.output_cleavage_count,
                channels,
            )?,
        ))
    }
}

fn pad_fragment_canvas(
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
        candle_core::bail!(
            "v0.27 active fragment canvas {active_cleavages} exceeds output width {output_cleavages}"
        );
    }
    let padding = Tensor::zeros(
        (batch, output_cleavages - active_cleavages, channels),
        active.dtype(),
        active.device(),
    )?;
    Tensor::cat(&[active, &padding], 1)
}

/// v0.27 forward output retaining the historical multi-task surface.
#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0270 {
    /// Backward-compatible property output; `ms2` is expected intensity.
    pub base: FoundationMultiTaskOutput,
    /// Fragment-presence logits.
    pub ms2_presence_logits: Tensor,
    /// Non-negative conditional intensity predictions.
    pub ms2_positive_intensity: Tensor,
}

/// Peptide-side v0.27 forward model.
#[derive(Clone)]
pub struct PeptideFoundationMultimodalForwardV0270 {
    encoder: PeptideFoundationEncoder,
    rt_specialist: RtSpecialistV0270,
    ccs_adapter: CcsPropertyAdapterV0270,
    fragment_decoder: ContextualFragmentDecoderV0270,
    residue_head: Linear,
    chemistry_head: Linear,
    contrastive_head: Linear,
    config: PeptideFoundationMultimodalV0270Config,
}

impl PeptideFoundationMultimodalForwardV0270 {
    /// Construct the fixed v0.27 peptide/property branch.
    pub fn new(config: PeptideFoundationMultimodalV0270Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let encoder = PeptideFoundationEncoder::new(config.forward.clone(), vb.pp("encoder"))?;
        Ok(Self {
            rt_specialist: RtSpecialistV0270::new(&config, vb.pp("rt_specialist"))?,
            // Keep the exact v0.26 CCS variable namespace/shapes so the frozen
            // checkpoint can warm-start this unchanged path.
            ccs_adapter: CcsPropertyAdapterV0270::new(
                config.forward.model_dim + 2,
                config.property_hidden_dim,
                vb.pp("adapters.ccs"),
            )?,
            fragment_decoder: ContextualFragmentDecoderV0270::new(
                &config,
                vb.pp("fragment_decoder"),
            )?,
            residue_head: nn::linear(config.forward.model_dim, 21, vb.pp("heads.masked_residue"))?,
            chemistry_head: nn::linear(
                config.forward.model_dim,
                config.forward.atom_feature_dim,
                vb.pp("heads.chemistry"),
            )?,
            contrastive_head: nn::linear(
                config.forward.model_dim,
                config.forward.contrastive_dim,
                vb.pp("heads.contrastive"),
            )?,
            encoder,
            config,
        })
    }

    /// Full v0.27 forward pass.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_v0270_t_with_shared_gradient_scales(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
        rt_encoder_gradient_scale: f64,
        ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultimodalForwardOutputV0270> {
        validate_gradient_scale("CCS", ccs_encoder_gradient_scale)?;
        let foundation = self.encoder.forward_t(batch, train)?;
        let rt = self
            .rt_specialist
            .forward_t(&foundation, train, rt_encoder_gradient_scale)?;

        let ccs_embedding =
            gradient_scaled_identity(&foundation.peptide_embedding, ccs_encoder_gradient_scale)?;
        let scaled_charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let ccs_scalar_context = match self.config.forward.ccs_context_mode {
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
        let ccs = if let Some(baseline) = &self.config.forward.ccs_physics_baseline {
            let baseline = standardized_ccs_physics_baseline_v0270(&foundation, context, baseline)?;
            (&baseline + &ccs_residual)?
        } else {
            ccs_residual
        };

        let (ms2_presence_logits, ms2_positive_intensity, ms2) =
            self.fragment_decoder
                .forward_t(&foundation, context, fragment, train)?;
        let residue_logits = self.residue_head.forward(&foundation.residue_embeddings)?;
        let chemistry_reconstruction = self
            .chemistry_head
            .forward(&foundation.residue_embeddings)?;
        let contrastive_projection = self
            .contrastive_head
            .forward(&foundation.peptide_embedding)?;

        Ok(FoundationMultimodalForwardOutputV0270 {
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

    /// Full v0.27 forward pass with ordinary shared-encoder gradients.
    pub fn forward_v0270_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0270> {
        self.forward_v0270_t_with_shared_gradient_scales(batch, context, fragment, train, 1.0, 1.0)
    }

    /// Encode a clean peptide and project only the shared/global representation.
    /// This avoids running the expensive fragment contextualizer for contrastive
    /// and inverse alignment paths that do not consume forward-MS2 predictions.
    pub fn peptide_projection_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<Tensor> {
        let foundation = self.encoder.forward_t(batch, train)?;
        self.contrastive_head.forward(&foundation.peptide_embedding)
    }

    /// Shared peptide encoder used unchanged by relation/inverse auxiliaries.
    pub fn encoder(&self) -> &PeptideFoundationEncoder {
        &self.encoder
    }

    /// Complete fixed v0.27 configuration.
    pub fn config(&self) -> &PeptideFoundationMultimodalV0270Config {
        &self.config
    }
}

/// Full v0.27 multimodal model.
#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0270Model {
    forward: PeptideFoundationMultimodalForwardV0270,
    diffusion: PeptideSpectrumDiffusionModel,
    causal: PeptideSpectrumCausalModel,
    spectrum_encoder: FoundationSpectrumEncoder,
    spectrum_projection: Linear,
    relation_query_norm: FoundationLayerNorm,
    relation_cross_attention: MultiHeadCrossAttention,
    relation_hidden: Linear,
    relation_output: Linear,
    config: PeptideFoundationMultimodalV0270Config,
}

impl PeptideFoundationMultimodalV0270Model {
    /// Construct the fixed v0.27 architecture.
    pub fn new(config: PeptideFoundationMultimodalV0270Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let diffusion =
            PeptideSpectrumDiffusionModel::new_open_ptm(config.inverse.clone(), vb.clone())?;
        let causal = PeptideSpectrumCausalModel::new_open_ptm(config.inverse.clone(), vb.clone())?;
        let spectrum_encoder =
            FoundationSpectrumEncoder::new(&config.inverse, vb.pp("spectrum_encoder"))?;
        let forward = PeptideFoundationMultimodalForwardV0270::new(config.clone(), vb.clone())?;
        let spectrum_projection = nn::linear(
            config.inverse.model_dim,
            config.forward.contrastive_dim,
            vb.pp("alignment.spectrum_projection"),
        )?;
        let relation_query_norm = FoundationLayerNorm::new(
            config.forward.model_dim,
            1e-5,
            vb.pp("multimodal_relation.query_norm"),
        )?;
        let relation_cross_attention = MultiHeadCrossAttention::new(
            config.forward.model_dim,
            config.forward.num_attention_heads,
            vb.pp("multimodal_relation.cross_attention"),
        )?;
        let relation_hidden = nn::linear(
            config.forward.model_dim * 5,
            config.forward.model_dim,
            vb.pp("multimodal_relation.hidden"),
        )?;
        let relation_output = nn::linear(
            config.forward.model_dim,
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
            config,
        })
    }

    /// Peptide/property branch.
    pub fn forward(&self) -> &PeptideFoundationMultimodalForwardV0270 {
        &self.forward
    }

    /// Spectrum-conditioned diffusion auxiliary, unchanged from v0.26.1.
    pub fn diffusion(&self) -> &PeptideSpectrumDiffusionModel {
        &self.diffusion
    }

    /// Spectrum-conditioned causal auxiliary, unchanged from v0.26.1.
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

    /// Same v0.26 relation score; the relation objective is intentionally not redesigned.
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

    /// Shared forward config.
    pub fn forward_config(&self) -> &FoundationConfig {
        &self.config.forward
    }

    /// Shared inverse config.
    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        &self.config.inverse
    }

    /// Complete fixed v0.27 config.
    pub fn config(&self) -> &PeptideFoundationMultimodalV0270Config {
        &self.config
    }
}

/// Factorized v0.27 MS2 loss components.
#[derive(Debug, Clone)]
pub struct FoundationMultimodalMs2LossesV0270 {
    /// Weighted optimization total.
    pub total: Tensor,
    /// Binary fragment-presence loss.
    pub presence: Tensor,
    /// Positive-intensity MSE on observed non-zero fragments.
    pub positive_intensity: Tensor,
    /// Expected-spectrum cosine loss.
    pub cosine: Tensor,
}

/// v0.26.1 factorized MS2 objective applied to contextual v0.27 fragment outputs.
pub fn foundation_multimodal_ms2_loss_v0270(
    output: &FoundationMultimodalForwardOutputV0270,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    presence_target: Option<&Tensor>,
    presence_mask: Option<&Tensor>,
) -> Result<FoundationMultimodalMs2LossesV0270> {
    let zero = output.ms2_presence_logits.affine(0.0, 0.0)?.sum_all()?;
    let (positive_intensity, cosine) = match (target, mask) {
        (Some(target), Some(mask)) => {
            if output.base.ms2.dims() != target.dims()
                || output.ms2_presence_logits.dims() != target.dims()
                || output.ms2_positive_intensity.dims() != target.dims()
            {
                candle_core::bail!("v0.27 MS2 output/target shape mismatch");
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
        _ => candle_core::bail!("v0.27 intensity target and mask must be paired"),
    };
    let presence = match (presence_target, presence_mask) {
        (Some(target), Some(mask)) => {
            if output.ms2_presence_logits.dims() != target.dims() {
                candle_core::bail!("v0.27 presence output/target shape mismatch");
            }
            masked_bce_with_logits(&output.ms2_presence_logits, target, mask)?
        }
        (None, None) => zero,
        _ => candle_core::bail!("v0.27 presence target and mask must be paired"),
    };
    let total = ((presence.affine(FOUNDATION_MULTIMODAL_MS2_PRESENCE_WEIGHT_V0270, 0.0)?
        + positive_intensity.affine(FOUNDATION_MULTIMODAL_MS2_INTENSITY_WEIGHT_V0270, 0.0)?)?
        + cosine.affine(FOUNDATION_MULTIMODAL_MS2_COSINE_WEIGHT_V0270, 0.0)?)?;
    Ok(FoundationMultimodalMs2LossesV0270 {
        total,
        presence,
        positive_intensity,
        cosine,
    })
}

/// Pairwise same-spectrum relation margin, unchanged from v0.26.1.
pub fn foundation_multimodal_relation_margin_loss_v0270(
    positive_score: &Tensor,
    negative_score: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    if positive_score.dims() != negative_score.dims() {
        candle_core::bail!("v0.27 relation score shape mismatch");
    }
    ((negative_score - positive_score)? + margin)?
        .relu()?
        .mean_all()
}

fn spectrum_peak_mask_from_embeddings(encoding: &FoundationSpectrumEncoding) -> Result<Tensor> {
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

fn standardized_ccs_physics_baseline_v0270(
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
            "v0.27 {label} encoder gradient scale must be finite and within [0,1], got {scale}"
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
    use super::super::data::TrainingContext;
    use super::super::featurize::{FoundationModification, PeptidoformInput};
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    fn config() -> PeptideFoundationMultimodalV0270Config {
        let mut forward = FoundationConfig::default();
        forward.max_sequence_len = 8;
        forward.max_atoms_per_residue = 24;
        forward.graph_hidden_dim = 32;
        forward.graph_layers = 1;
        forward.model_dim = 192;
        forward.num_attention_heads = 6;
        forward.transformer_ff_dim = 768;
        forward.transformer_layers = 1;
        forward.contrastive_dim = 32;
        let mut inverse = FoundationDiffusionConfig::default();
        inverse.model_dim = 192;
        inverse.feed_forward_dim = 768;
        inverse.num_attention_heads = 6;
        inverse.spectrum_layers = 1;
        inverse.decoder_layers = 1;
        PeptideFoundationMultimodalV0270Config::fixed(forward, inverse).unwrap()
    }

    #[test]
    fn v0270_fragment_context_uses_open_ptm_geometry_and_masks_charge2_for_z1() -> Result<()> {
        let device = Device::Cpu;
        let mut peptide = PeptidoformInput::unmodified("ACDE");
        peptide
            .modifications
            .push(FoundationModification::mass_delta(1, 304.1772));
        let record = FoundationTrainingRecord {
            peptidoform: peptide,
            retention_time: Default::default(),
            ccs: None,
            fragments: Vec::new(),
            observed_spectrum_peaks: Vec::new(),
            context: TrainingContext {
                charge: Some(1),
                nce: Some(30.0),
                instrument_id: Some(1),
                ..TrainingContext::default()
            },
            run_id: None,
        };
        let cfg = config();
        let context =
            FoundationFragmentContextBatchV0270::from_records(&[record], &cfg.forward, &device)?;
        let mask = context.channel_mask()?.to_vec3::<f32>()?;
        assert_eq!(mask[0][0][0], 1.0);
        assert_eq!(mask[0][0][1], 0.0);
        assert_eq!(mask[0][0][2], 1.0);
        assert_eq!(mask[0][0][3], 0.0);
        let features = context.continuous_features.to_vec3::<f32>()?;
        let modified_cleavage_token = 1 * FOUNDATION_FRAGMENT_CHANNELS_V0270;
        assert!(features[0][modified_cleavage_token][13].abs() > 0.1);
        Ok(())
    }

    #[test]
    fn v0270_model_constructs_fixed_specialists_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = config();
        let model = PeptideFoundationMultimodalV0270Model::new(cfg, vb)?;
        assert_eq!(model.forward_config().model_dim, 192);
        assert_eq!(model.config().rt_attention_heads, 6);
        assert_eq!(model.config().fragment_transformer_layers, 2);
        Ok(())
    }
}
