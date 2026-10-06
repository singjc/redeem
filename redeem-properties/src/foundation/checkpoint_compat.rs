//! Private compatibility implementation for the accepted foundation checkpoints.
//!
//! This file intentionally quarantines the historical architecture lineage required
//! to load already-trained checkpoints. Production callers use `FoundationPredictor`;
//! none of these versioned implementation names are part of the public API.
#![allow(dead_code, unused_imports, unused_variables)]

mod base_forward {
    // v0.27 multimodal peptide/spectrum foundation architecture.
    //
    // v0.27 preserves the accepted v0.26.1 shared peptide encoder, CCS physical
    // residual path, observed-spectrum encoder, inverse diffusion/causal models,
    // and same-spectrum relation objective. It changes only the two forward paths
    // identified by the frozen historical VALIDATION comparison:
    //
    // * RT receives two residue-level specialist Transformer blocks plus learned-
    //   query attention pooling before regression.
    // * MS2 is decoded from one token per theoretical cleavage/channel, with exact
    //   open-PTM fragment geometry and two contextual fragment Transformer blocks.
    //   Presence and positive conditional intensity remain factorized exactly as in
    //   v0.26.1.

    use super::super::causal::PeptideSpectrumCausalModel;
    use super::super::ccs_physics::FOUNDATION_CCS_PHYSICS_FEATURE_COUNT;
    use super::super::config::{
        FoundationCcsContextMode, FoundationCcsPhysicsBaselineConfig, FoundationConfig,
        FoundationMs2OutputActivation,
    };
    use super::super::data::FoundationTrainingRecord;
    use super::super::diffusion::{
        FoundationDiffusionConfig, FoundationSpectrumEncoder, FoundationSpectrumEncoding,
        PeptideSpectrumDiffusionModel, FOUNDATION_PEPTIDE_WATER_MASS_DA,
    };
    use super::super::featurize::FoundationModificationSite;
    use super::super::fragment_relation::foundation_fragment_cleavage_geometry;
    use super::super::layers::{
        FoundationLayerNorm, MultiHeadCrossAttention, PeptideTransformerBlock,
    };
    use super::super::loss::{foundation_ms2_loss, FoundationMs2LossConfig};
    use super::super::model::{
        apply_ms2_output_activation, gradient_scaled_identity, FoundationMultiTaskOutput,
        FoundationOutput, PeptideFoundationEncoder, PrecursorContextBatch,
    };
    use super::super::spectrum::FoundationSpectrumBatch;
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
        pub fn fixed(
            forward: FoundationConfig,
            inverse: FoundationDiffusionConfig,
        ) -> Result<Self> {
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
                candle_core::bail!(
                    "v0.27 specialist dimensions differ from the frozen architecture"
                );
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
                    .is_some_and(|instrument| instrument > 0)
                    as u8 as f32;
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
                        let complement_fraction =
                            (spec.complement_neutral_mass / total_mass) as f32;
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
                fragment_charge_ids: Tensor::from_vec(
                    fragment_charge_ids,
                    (batch, tokens),
                    device,
                )?
                .to_dtype(DType::U32)?,
                precursor_charge_ids: Tensor::from_vec(
                    precursor_charge_ids,
                    (batch, tokens),
                    device,
                )?
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
        geometry: &super::super::fragment_relation::FoundationFragmentCleavageGeometry,
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
                candle_core::bail!(
                    "v0.27 PTM index {index} exceeds peptide length {residue_count}"
                );
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
        fn new(
            config: &PeptideFoundationMultimodalV0270Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
                hidden_384: nn::linear(
                    config.forward.model_dim * 2,
                    384,
                    vb.pp("head.hidden_384"),
                )?,
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
        fn new(
            config: &PeptideFoundationMultimodalV0270Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
            if fragment.cleavage_count > sequence - 1
                || fragment.output_cleavage_count != sequence - 1
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
            let cleavage =
                Tensor::cat(&[&left, &right], 3)?.reshape((batch, tokens, model_dim * 2))?;
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
        pub fn new(
            config: PeptideFoundationMultimodalV0270Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
                residue_head: nn::linear(
                    config.forward.model_dim,
                    21,
                    vb.pp("heads.masked_residue"),
                )?,
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

        /// Encode one peptide batch with the frozen shared v0.27 encoder.
        ///
        /// This crate-visible hook exists so later architectures can refine the
        /// shared representation without duplicating the validated v0.27 heads.
        pub(crate) fn encode_foundation_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            train: bool,
        ) -> Result<FoundationOutput> {
            self.encoder.forward_t(batch, train)
        }

        /// Evaluate the validated v0.27 RT specialist from a supplied representation.
        pub(crate) fn rt_from_foundation_t(
            &self,
            foundation: &FoundationOutput,
            train: bool,
            shared_gradient_scale: f64,
        ) -> Result<Tensor> {
            self.rt_specialist
                .forward_t(foundation, train, shared_gradient_scale)
        }

        /// Evaluate the validated v0.27 CCS path from a supplied representation.
        pub(crate) fn ccs_from_foundation(
            &self,
            foundation: &FoundationOutput,
            context: &PrecursorContextBatch,
            shared_gradient_scale: f64,
        ) -> Result<Tensor> {
            validate_gradient_scale("CCS", shared_gradient_scale)?;
            let ccs_embedding =
                gradient_scaled_identity(&foundation.peptide_embedding, shared_gradient_scale)?;
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
            if let Some(baseline) = &self.config.forward.ccs_physics_baseline {
                let baseline =
                    standardized_ccs_physics_baseline_v0270(foundation, context, baseline)?;
                Ok((&baseline + &ccs_residual)?)
            } else {
                Ok(ccs_residual)
            }
        }

        /// Evaluate the validated contextual fragment decoder from a supplied representation.
        pub(crate) fn ms2_from_foundation_t(
            &self,
            foundation: &FoundationOutput,
            context: &PrecursorContextBatch,
            fragment: &FoundationFragmentContextBatchV0270,
            train: bool,
        ) -> Result<(Tensor, Tensor, Tensor)> {
            self.fragment_decoder
                .forward_t(foundation, context, fragment, train)
        }

        /// Evaluate the unchanged shared auxiliary heads from a supplied representation.
        pub(crate) fn auxiliaries_from_foundation(
            &self,
            foundation: &FoundationOutput,
        ) -> Result<(Tensor, Tensor, Tensor)> {
            let residue_logits = self.residue_head.forward(&foundation.residue_embeddings)?;
            let chemistry_reconstruction = self
                .chemistry_head
                .forward(&foundation.residue_embeddings)?;
            let contrastive_projection = self
                .contrastive_head
                .forward(&foundation.peptide_embedding)?;
            Ok((
                residue_logits,
                chemistry_reconstruction,
                contrastive_projection,
            ))
        }

        /// Full v0.27 forward pass.
        #[allow(clippy::too_many_arguments)]
        pub fn forward_v0270_t_with_shared_gradient_scales(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
            fragment: &FoundationFragmentContextBatchV0270,
            train: bool,
            rt_encoder_gradient_scale: f64,
            ccs_encoder_gradient_scale: f64,
        ) -> Result<FoundationMultimodalForwardOutputV0270> {
            let foundation = self.encode_foundation_t(batch, train)?;
            let rt = self.rt_from_foundation_t(&foundation, train, rt_encoder_gradient_scale)?;
            let ccs = self.ccs_from_foundation(&foundation, context, ccs_encoder_gradient_scale)?;
            let (ms2_presence_logits, ms2_positive_intensity, ms2) =
                self.ms2_from_foundation_t(&foundation, context, fragment, train)?;
            let (residue_logits, chemistry_reconstruction, contrastive_projection) =
                self.auxiliaries_from_foundation(&foundation)?;

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
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
            fragment: &FoundationFragmentContextBatchV0270,
            train: bool,
        ) -> Result<FoundationMultimodalForwardOutputV0270> {
            self.forward_v0270_t_with_shared_gradient_scales(
                batch, context, fragment, train, 1.0, 1.0,
            )
        }

        /// Encode a clean peptide and project only the shared/global representation.
        /// This avoids running the expensive fragment contextualizer for contrastive
        /// and inverse alignment paths that do not consume forward-MS2 predictions.
        pub fn peptide_projection_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
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
        pub fn new(
            config: PeptideFoundationMultimodalV0270Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
            config.validate()?;
            let diffusion =
                PeptideSpectrumDiffusionModel::new_open_ptm(config.inverse.clone(), vb.clone())?;
            let causal =
                PeptideSpectrumCausalModel::new_open_ptm(config.inverse.clone(), vb.clone())?;
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
            + positive_intensity
                .affine(FOUNDATION_MULTIMODAL_MS2_INTENSITY_WEIGHT_V0270, 0.0)?)?
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

    fn zero_initialized_linear(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Linear> {
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", nn::Init::Const(0.0))?;
        let bias = vb.get_with_hints(out_dim, "bias", nn::Init::Const(0.0))?;
        Ok(Linear::new(weight, Some(bias)))
    }
}

mod property_refinement {
    // v0.31 protected-CCS fragment-aware property-view continuation.
    //
    // v0.30 demonstrated that fragment-aware residue supervision can improve RT and
    // MS2 on DEV and TRAIN-HOLDOUT, but the same shared-encoder update materially
    // regressed CCS. v0.31 therefore keeps the complete frozen v0.27 encoder + CCS
    // path as an immutable base view and learns a separate residue-level property
    // refinement used by RT/MS2/self-supervision. The refinement is identity at
    // initialization through a zero-initialized residual projection, so step 0
    // reproduces v0.27 exactly for all accepted property heads.
    //
    // This is deliberately not the closed v0.28 task-conditioned low-rank adapter:
    // there are no task embeddings or post-pooled task adapters. The new branch is
    // one residue Transformer refinement shared by RT/MS2 and trained with the same
    // cleavage-local fragment auxiliary that produced the useful v0.30 signal.

    use super::super::causal::PeptideSpectrumCausalModel;
    use super::super::config::FoundationConfig;
    use super::super::diffusion::{
        FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
    };
    use super::super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
    use super::super::loss::{foundation_ms2_loss, FoundationMs2LossConfig, FoundationMs2Losses};
    use super::super::model::{
        apply_ms2_output_activation, FoundationMultiTaskOutput, FoundationOutput,
        PrecursorContextBatch,
    };
    use super::super::spectrum::FoundationSpectrumBatch;
    use super::base_forward::{
        foundation_multimodal_ms2_loss_v0270, foundation_multimodal_relation_margin_loss_v0270,
        FoundationFragmentContextBatchV0270, FoundationMultimodalForwardOutputV0270,
        FoundationMultimodalMs2LossesV0270, PeptideFoundationMultimodalV0270Config,
        PeptideFoundationMultimodalV0270Model, FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270,
        FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270,
    };
    use candle_core::{Module, Result, Tensor};
    use candle_nn::{self as nn, Linear, VarBuilder};
    use serde::{Deserialize, Serialize};

    pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0310: &str =
        "v0.31-192d-protected-ccs-fragment-aware-property-view-v027-heads-openptm32";
    pub const FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0310: usize = 1;
    pub const FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0310: usize = 6;
    pub const FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0310: usize = 768;
    pub const FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0310: usize = 192;
    pub const FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0310: usize = 4;
    pub const FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0310: f64 = 0.05;
    pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0310: usize =
        FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270;
    pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0310: f64 =
        FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct PeptideFoundationMultimodalV0310Config {
        pub base_v0270: PeptideFoundationMultimodalV0270Config,
        pub property_refinement_layers: usize,
        pub property_refinement_heads: usize,
        pub property_refinement_ff_dim: usize,
        pub fragment_aux_hidden: usize,
        pub fragment_aux_weight: f64,
    }

    impl PeptideFoundationMultimodalV0310Config {
        pub fn fixed(base_v0270: PeptideFoundationMultimodalV0270Config) -> Result<Self> {
            base_v0270.validate()?;
            let config = Self {
                base_v0270,
                property_refinement_layers: FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0310,
                property_refinement_heads: FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0310,
                property_refinement_ff_dim: FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0310,
                fragment_aux_hidden: FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0310,
                fragment_aux_weight: FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0310,
            };
            config.validate()?;
            Ok(config)
        }

        pub fn validate(&self) -> Result<()> {
            self.base_v0270.validate()?;
            if self.base_v0270.forward.model_dim != 192 {
                candle_core::bail!("v0.31 is fixed at model_dim=192");
            }
            if self.property_refinement_layers != FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0310
                || self.property_refinement_heads != FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0310
                || self.property_refinement_ff_dim != FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0310
                || self.fragment_aux_hidden != FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0310
                || (self.fragment_aux_weight - FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0310)
                    .abs()
                    > f64::EPSILON
            {
                candle_core::bail!(
                    "v0.31 dimensions/auxiliary weight differ from the fixed design"
                );
            }
            Ok(())
        }

        pub fn forward(&self) -> &FoundationConfig {
            &self.base_v0270.forward
        }

        pub fn inverse(&self) -> &FoundationDiffusionConfig {
            &self.base_v0270.inverse
        }
    }

    #[derive(Clone)]
    struct PropertyResidueRefinementV0310 {
        input_norm: FoundationLayerNorm,
        transformer: PeptideTransformerBlock,
        delta_output: Linear,
        model_dim: usize,
    }

    impl PropertyResidueRefinementV0310 {
        fn new(
            config: &PeptideFoundationMultimodalV0310Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
                delta_output: zero_initialized_linear_v0310(
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
                    "v0.31 property refinement expected residue width {}, got {}",
                    self.model_dim,
                    model_dim
                );
            }

            // Scientific invariant: the accepted v0.27 peptide representation is
            // immutable in this lane. All property-view gradients stop here.
            let base_residues = base.residue_embeddings.detach();
            let base_peptide = base.peptide_embedding.detach();
            let normalized = self.input_norm.forward(&base_residues)?;
            let hidden = self
                .transformer
                .forward_t(&normalized, &base.residue_mask, train)?;
            let delta = self.delta_output.forward(&hidden)?;
            let expanded_mask = base
                .residue_mask
                .unsqueeze(2)?
                .broadcast_as((batch, sequence, model_dim))?;
            let delta = delta.broadcast_mul(&expanded_mask)?;
            let residue_embeddings = (&base_residues + &delta)?;
            let pooled_delta = masked_mean_v0310(&delta, &base.residue_mask)?;
            let peptide_embedding = (&base_peptide + &pooled_delta)?;
            Ok(FoundationOutput {
                residue_embeddings,
                peptide_embedding,
                residue_mask: base.residue_mask.clone(),
                chemistry_targets: base.chemistry_targets.clone(),
            })
        }
    }

    #[derive(Clone)]
    struct FragmentRepresentationAuxV0310 {
        hidden: Linear,
        output: Linear,
        model_dim: usize,
        channels: usize,
    }

    impl FragmentRepresentationAuxV0310 {
        fn new(
            config: &PeptideFoundationMultimodalV0310Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
            let model_dim = config.forward().model_dim;
            let channels = config.forward().ms2_fragment_channels;
            let input_dim = 2 * model_dim + FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0310;
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
            activation: super::super::config::FoundationMs2OutputActivation,
        ) -> Result<Tensor> {
            let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
            if model_dim != self.model_dim || sequence < 2 {
                candle_core::bail!(
                    "v0.31 fragment auxiliary expected residue width {} and sequence>=2, got {:?}",
                    self.model_dim,
                    foundation.residue_embeddings.dims()
                );
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
                .broadcast_as((
                    batch,
                    cleavages,
                    FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0310,
                ))?;
            let features = Tensor::cat(&[&left, &right, &scalar_context], 2)?.contiguous()?;
            let (_, _, feature_dim) = features.dims3()?;
            let hidden = self
                .hidden
                .forward(
                    &features
                        .reshape((batch * cleavages, feature_dim))?
                        .contiguous()?,
                )?
                .relu()?
                .contiguous()?;
            let logits =
                self.output
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
    pub struct FoundationMultimodalForwardOutputV0310 {
        pub base: FoundationMultiTaskOutput,
        pub ms2_presence_logits: Tensor,
        pub ms2_positive_intensity: Tensor,
        pub fragment_representation_aux: Tensor,
    }

    #[derive(Clone)]
    pub struct PeptideFoundationMultimodalV0310Model {
        base_v0270: PeptideFoundationMultimodalV0270Model,
        property_refinement: PropertyResidueRefinementV0310,
        fragment_aux: FragmentRepresentationAuxV0310,
        config: PeptideFoundationMultimodalV0310Config,
    }

    impl PeptideFoundationMultimodalV0310Model {
        pub fn new(
            config: PeptideFoundationMultimodalV0310Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
            config.validate()?;
            let base_v0270 =
                PeptideFoundationMultimodalV0270Model::new(config.base_v0270.clone(), vb.clone())?;
            let property_refinement =
                PropertyResidueRefinementV0310::new(&config, vb.pp("property_refinement"))?;
            let fragment_aux =
                FragmentRepresentationAuxV0310::new(&config, vb.pp("fragment_representation_aux"))?;
            Ok(Self {
                base_v0270,
                property_refinement,
                fragment_aux,
                config,
            })
        }

        pub fn property_foundation_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            train: bool,
        ) -> Result<FoundationOutput> {
            // The accepted base representation is an inference-mode frozen view.
            // Keeping base dropout disabled makes this branch an exact functional
            // copy of v0.27 rather than a stochastic teacher during v0.31 training.
            let base = self
                .base_v0270
                .forward()
                .encode_foundation_t(batch, false)?;
            self.property_refinement.forward_t(&base, train)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward_v0310_t_with_shared_gradient_scales(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
            fragment: &FoundationFragmentContextBatchV0270,
            train: bool,
            rt_encoder_gradient_scale: f64,
            _ccs_encoder_gradient_scale: f64,
        ) -> Result<FoundationMultimodalForwardOutputV0310> {
            let base_foundation = self
                .base_v0270
                .forward()
                .encode_foundation_t(batch, false)?;
            let property_foundation = self
                .property_refinement
                .forward_t(&base_foundation, train)?;

            let rt = self.base_v0270.forward().rt_from_foundation_t(
                &property_foundation,
                train,
                rt_encoder_gradient_scale,
            )?;

            // CCS is a protected v0.27 path. Detaching the scalar output freezes both
            // the base encoder and CCS adapter even though the optimizer owns the full VarMap.
            let ccs = self
                .base_v0270
                .forward()
                .ccs_from_foundation(&base_foundation, context, 0.0)?
                .detach();

            let (ms2_presence_logits, ms2_positive_intensity, ms2) = self
                .base_v0270
                .forward()
                .ms2_from_foundation_t(&property_foundation, context, fragment, train)?;
            let (residue_logits, chemistry_reconstruction, contrastive_projection) = self
                .base_v0270
                .forward()
                .auxiliaries_from_foundation(&property_foundation)?;
            let fragment_representation_aux = self.fragment_aux.forward(
                &property_foundation,
                context,
                self.config.forward().ms2_output_activation,
            )?;

            Ok(FoundationMultimodalForwardOutputV0310 {
                base: FoundationMultiTaskOutput {
                    foundation: property_foundation,
                    rt,
                    ccs,
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

        pub fn forward_v0310_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
            fragment: &FoundationFragmentContextBatchV0270,
            train: bool,
        ) -> Result<FoundationMultimodalForwardOutputV0310> {
            self.forward_v0310_t_with_shared_gradient_scales(
                batch, context, fragment, train, 1.0, 0.0,
            )
        }

        /// Evaluate only the immutable protected CCS path. Later forward-only
        /// architecture experiments use this hook to preserve exact v0.31/v0.27
        /// CCS without paying for the frozen RT/MS2 decoders on every train step.
        pub fn protected_ccs_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
        ) -> Result<Tensor> {
            let base_foundation = self
                .base_v0270
                .forward()
                .encode_foundation_t(batch, false)?;
            Ok(self
                .base_v0270
                .forward()
                .ccs_from_foundation(&base_foundation, context, 0.0)?
                .detach())
        }

        pub fn peptide_projection_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            train: bool,
        ) -> Result<Tensor> {
            let foundation = self.property_foundation_t(batch, train)?;
            let (_, _, projection) = self
                .base_v0270
                .forward()
                .auxiliaries_from_foundation(&foundation)?;
            Ok(projection)
        }

        pub fn diffusion(&self) -> &PeptideSpectrumDiffusionModel {
            self.base_v0270.diffusion()
        }

        pub fn causal(&self) -> &PeptideSpectrumCausalModel {
            self.base_v0270.causal()
        }

        pub fn encode_spectrum_t(
            &self,
            spectrum: &FoundationSpectrumBatch,
            train: bool,
        ) -> Result<FoundationSpectrumEncoding> {
            self.base_v0270.encode_spectrum_t(spectrum, train)
        }

        pub fn project_spectrum_embedding(&self, embedding: &Tensor) -> Result<Tensor> {
            self.base_v0270.project_spectrum_embedding(embedding)
        }

        pub fn relation_score(
            &self,
            peptide: &FoundationOutput,
            spectrum: &FoundationSpectrumEncoding,
        ) -> Result<Tensor> {
            self.base_v0270.relation_score(peptide, spectrum)
        }

        pub fn forward_config(&self) -> &FoundationConfig {
            self.config.forward()
        }

        pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
            self.config.inverse()
        }

        pub fn config(&self) -> &PeptideFoundationMultimodalV0310Config {
            &self.config
        }
    }

    pub type FoundationFragmentContextBatchV0310 = FoundationFragmentContextBatchV0270;
    pub type FoundationMultimodalMs2LossesV0310 = FoundationMultimodalMs2LossesV0270;

    pub fn foundation_multimodal_ms2_loss_v0310(
        output: &FoundationMultimodalForwardOutputV0310,
        target: Option<&Tensor>,
        mask: Option<&Tensor>,
        presence_target: Option<&Tensor>,
        presence_mask: Option<&Tensor>,
    ) -> Result<FoundationMultimodalMs2LossesV0310> {
        let proxy = FoundationMultimodalForwardOutputV0270 {
            base: output.base.clone(),
            ms2_presence_logits: output.ms2_presence_logits.clone(),
            ms2_positive_intensity: output.ms2_positive_intensity.clone(),
        };
        foundation_multimodal_ms2_loss_v0270(&proxy, target, mask, presence_target, presence_mask)
    }

    pub fn foundation_fragment_representation_aux_loss_v0310(
        output: &FoundationMultimodalForwardOutputV0310,
        target: Option<&Tensor>,
        mask: Option<&Tensor>,
        config: FoundationMs2LossConfig,
    ) -> Result<Option<FoundationMs2Losses>> {
        match (target, mask) {
            (Some(target), Some(mask)) => Ok(Some(foundation_ms2_loss(
                &output.fragment_representation_aux,
                target,
                mask,
                config,
            )?)),
            (None, None) => Ok(None),
            _ => {
                candle_core::bail!("v0.31 fragment auxiliary target/mask must be supplied together")
            }
        }
    }

    pub fn foundation_multimodal_relation_margin_loss_v0310(
        positive_score: &Tensor,
        negative_score: &Tensor,
        margin: f64,
    ) -> Result<Tensor> {
        foundation_multimodal_relation_margin_loss_v0270(positive_score, negative_score, margin)
    }

    fn masked_mean_v0310(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, length, dim) = values.dims3()?;
        let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
        let summed = values.broadcast_mul(&expanded)?.sum(1)?;
        let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
        summed.broadcast_div(&denominator)
    }

    fn zero_initialized_linear_v0310(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Linear> {
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", nn::Init::Const(0.0))?;
        let bias = vb.get_with_hints(out_dim, "bias", nn::Init::Const(0.0))?;
        Ok(Linear::new(weight, Some(bias)))
    }
}

mod forward_specialists {
    // v0.35 trainable forward-representation continuation.
    //
    // v0.34 showed that substantially larger RT/MS2 heads placed on top of a
    // frozen v0.31 representation do not materially improve MS2 shape/correlation.
    // v0.35 therefore moves the adaptation boundary into the peptide representation
    // itself while preserving the accepted v0.31 model as an immutable anchor for
    // CCS, inverse generation, alignment, and relation scoring.
    //
    // The forward branch is an exact trainable clone of the v0.31 peptide-side
    // encoder/refinement/RT/MS2 weights at step 0. During optimization the clone is
    // allowed to move end-to-end under RT, factorized MS2, masked-residue,
    // chemistry-reconstruction, contrastive, and fragment-deep-supervision losses.
    // An identity-initialized acquisition-context conditioner injects precursor
    // charge, NCE, and instrument information *before* additional residue
    // Transformer blocks on the MS2 path. This lets fragmentation context reshape
    // residue states instead of being consumed only by the terminal fragment head.
    //
    // CCS remains the exact frozen v0.31/v0.27 prediction path. The inverse model
    // remains the exact frozen v0.31 path as well.

    use super::super::config::FoundationConfig;
    use super::super::diffusion::{
        FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
    };
    use super::super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
    use super::super::model::{
        apply_ms2_output_activation, FoundationMultiTaskOutput, FoundationOutput,
        PrecursorContextBatch,
    };
    use super::super::spectrum::FoundationSpectrumBatch;
    use super::base_forward::{
        foundation_multimodal_ms2_loss_v0270, FoundationMultimodalForwardOutputV0270,
        FoundationMultimodalMs2LossesV0270, PeptideFoundationMultimodalForwardV0270,
    };
    use super::property_refinement::{
        FoundationFragmentContextBatchV0310, PeptideFoundationMultimodalV0310Config,
        PeptideFoundationMultimodalV0310Model,
    };
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
        super::property_refinement::FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0310;
    pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0350: f64 =
        super::property_refinement::FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0310;

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
    pub(crate) struct PropertyResidueRefinementV0350 {
        input_norm: FoundationLayerNorm,
        transformer: PeptideTransformerBlock,
        delta_output: Linear,
        model_dim: usize,
    }

    impl PropertyResidueRefinementV0350 {
        pub(crate) fn new(
            config: &PeptideFoundationMultimodalV0350Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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

        pub(crate) fn forward_t(
            &self,
            base: &FoundationOutput,
            train: bool,
        ) -> Result<FoundationOutput> {
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
        fn new(
            config: &PeptideFoundationMultimodalV0350Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
                context_projection: nn::linear(
                    context_dim,
                    model_dim,
                    vb.pp("context_projection"),
                )?,
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
                candle_core::bail!(
                    "v0.35 MS2 context conditioner received incompatible residue width"
                );
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
        fn new(
            config: &PeptideFoundationMultimodalV0350Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
            activation: super::super::config::FoundationMs2OutputActivation,
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
            let logits =
                self.output
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
        pub fn new(
            config: PeptideFoundationMultimodalV0350Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
            batch: &super::super::featurize::FoundationBatch,
            train: bool,
        ) -> Result<FoundationOutput> {
            let encoded = self.forward_v0350.encode_foundation_t(batch, train)?;
            self.property_refinement_v0350.forward_t(&encoded, train)
        }

        /// Fast protected CCS-only path for diagnostics/calibration stages.
        ///
        /// This delegates directly to the accepted v0.31 CCS path embedded in
        /// v0.35 and does not evaluate the trainable v0.35 RT/MS2 representation.
        pub fn protected_ccs_v0350_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
        ) -> Result<Tensor> {
            self.base_v0310.protected_ccs_t(batch, context)
        }

        /// Frozen scalar anchor used by later protected specialist stages.
        ///
        /// This evaluates only the v0.35 trainable-forward representation, RT head,
        /// and protected v0.31 CCS path. Returned tensors are detached so downstream
        /// specialist optimizers cannot backpropagate into the accepted v0.35 model.
        pub fn detached_scalar_anchor_v0350_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
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
            batch: &super::super::featurize::FoundationBatch,
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
            batch: &super::super::featurize::FoundationBatch,
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
            batch: &super::super::featurize::FoundationBatch,
            train: bool,
        ) -> Result<FoundationOutput> {
            self.base_v0310.property_foundation_t(batch, train)
        }

        /// Frozen v0.31 projection retained for inverse/alignment diagnostics.
        pub fn peptide_projection_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            train: bool,
        ) -> Result<Tensor> {
            self.base_v0310.peptide_projection_t(batch, train)
        }

        pub fn diffusion(&self) -> &PeptideSpectrumDiffusionModel {
            self.base_v0310.diffusion()
        }

        pub fn causal(&self) -> &super::super::causal::PeptideSpectrumCausalModel {
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
        super::property_refinement::foundation_multimodal_relation_margin_loss_v0310(
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
        config: super::super::loss::FoundationMs2LossConfig,
    ) -> Result<Option<super::super::loss::FoundationMs2Losses>> {
        match (target, mask) {
            (Some(target), Some(mask)) => Ok(Some(super::super::loss::foundation_ms2_loss(
                &output.fragment_representation_aux,
                target,
                mask,
                config,
            )?)),
            (None, None) => Ok(None),
            _ => {
                candle_core::bail!("v0.35 fragment auxiliary target/mask must be supplied together")
            }
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
}

mod scalar_physics {
    // v0.36 protected scalar-property refinement on top of frozen v0.35.
    //
    // v0.35 established that relearning the forward representation can close a
    // large fraction of the RT/MS2 gap, but its CCS path remained intentionally
    // frozen. v0.36 keeps the complete v0.35 model immutable and trains two small,
    // task-isolated scalar residuals from detached v0.35 representations:
    //
    // - an intrinsic RT residual that may refine RT without acquisition context;
    // - a physics-aware CCS residual that consumes peptide representation plus
    //   precursor charge/mass/length/PTM context.
    //
    // Both output layers are initialized to exact zero, so step 0 reproduces the
    // accepted v0.35 RT/CCS predictions exactly. MS2 and all inverse paths remain
    // bitwise anchored to v0.35 because no v0.35 parameter belongs to the optimizer.

    use super::super::config::FoundationConfig;
    use super::super::data::FoundationTrainingRecord;
    use super::super::diffusion::{foundation_peptidoform_neutral_mass, FoundationDiffusionConfig};
    use super::super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
    use super::super::model::{FoundationOutput, PrecursorContextBatch};
    use super::forward_specialists::{
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
        fn new(
            config: &PeptideFoundationMultimodalV0360Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
        fn new(
            config: &PeptideFoundationMultimodalV0360Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
        pub fn new(
            config: PeptideFoundationMultimodalV0360Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
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
            batch: &super::super::featurize::FoundationBatch,
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
            batch: &super::super::featurize::FoundationBatch,
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
}

mod ccs_checkpoint {
    // v0.38 mobility-native CCS representation refinement on top of frozen v0.35.
    //
    // v0.37 showed that TRAIN-only consensus/reliability weighting improves CCS,
    // but a residual head on a detached v0.35 representation plateaus before the
    // material DEV gate. v0.38 therefore gives CCS its own trainable copy of the
    // accepted v0.35 peptide representation while leaving RT, MS2, and inverse
    // paths immutable.
    //
    // The CCS child branch is initialized from `forward_v0350` plus
    // `property_refinement_v0350`. A zero-initialized native ion-mobility residual
    // is predicted from the trainable CCS representation and precursor physics.
    // The trainer adds that residual to the mobility implied by frozen v0.35 CCS
    // and applies the exact Bruker mobility->CCS conversion outside this module.
    // Therefore step 0 is exactly the accepted v0.35 CCS prediction.

    use super::super::config::FoundationConfig;
    use super::super::diffusion::FoundationDiffusionConfig;
    use super::super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
    use super::super::model::{FoundationOutput, PrecursorContextBatch};
    use super::base_forward::PeptideFoundationMultimodalForwardV0270;
    use super::forward_specialists::{
        FoundationFragmentContextBatchV0350, FoundationMultimodalForwardOutputV0350,
        PeptideFoundationMultimodalV0350Config, PeptideFoundationMultimodalV0350Model,
        PropertyResidueRefinementV0350,
    };
    use super::scalar_physics::{
        FoundationScalarPhysicsBatchV0360, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
    };
    use candle_core::{Module, Result, Tensor};
    use candle_nn::{self as nn, Linear, VarBuilder};
    use serde::{Deserialize, Serialize};

    pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0380: &str =
        "v0.38-frozen-v0350-mobility-native-trainable-ccs-representation";
    pub const FOUNDATION_CCS_CONTEXT_LAYERS_V0380: usize = 2;
    pub const FOUNDATION_CCS_CONTEXT_HEADS_V0380: usize = 6;
    pub const FOUNDATION_CCS_CONTEXT_FF_DIM_V0380: usize = 768;
    pub const FOUNDATION_CCS_MOBILITY_HIDDEN_V0380: usize = 384;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct PeptideFoundationMultimodalV0380Config {
        pub base_v0350: PeptideFoundationMultimodalV0350Config,
        pub ccs_context_layers: usize,
        pub ccs_context_heads: usize,
        pub ccs_context_ff_dim: usize,
        pub ccs_mobility_hidden: usize,
    }

    impl PeptideFoundationMultimodalV0380Config {
        pub fn fixed(base_v0350: PeptideFoundationMultimodalV0350Config) -> Result<Self> {
            base_v0350.validate()?;
            let config = Self {
                base_v0350,
                ccs_context_layers: FOUNDATION_CCS_CONTEXT_LAYERS_V0380,
                ccs_context_heads: FOUNDATION_CCS_CONTEXT_HEADS_V0380,
                ccs_context_ff_dim: FOUNDATION_CCS_CONTEXT_FF_DIM_V0380,
                ccs_mobility_hidden: FOUNDATION_CCS_MOBILITY_HIDDEN_V0380,
            };
            config.validate()?;
            Ok(config)
        }

        pub fn validate(&self) -> Result<()> {
            self.base_v0350.validate()?;
            if self.forward().model_dim != 192 {
                candle_core::bail!("v0.38 requires the accepted 192d v0.35 representation");
            }
            if self.ccs_context_layers != FOUNDATION_CCS_CONTEXT_LAYERS_V0380
                || self.ccs_context_heads != FOUNDATION_CCS_CONTEXT_HEADS_V0380
                || self.ccs_context_ff_dim != FOUNDATION_CCS_CONTEXT_FF_DIM_V0380
                || self.ccs_mobility_hidden != FOUNDATION_CCS_MOBILITY_HIDDEN_V0380
            {
                candle_core::bail!("v0.38 dimensions differ from the fixed architecture");
            }
            if self.forward().model_dim % self.ccs_context_heads != 0 {
                candle_core::bail!("v0.38 model width must be divisible by CCS attention heads");
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
    struct MobilityContextHeadV0380 {
        physics_projection: Linear,
        input_norm: FoundationLayerNorm,
        blocks: Vec<PeptideTransformerBlock>,
        hidden: Linear,
        bottleneck: Linear,
        output: Linear,
        model_dim: usize,
    }

    impl MobilityContextHeadV0380 {
        fn new(
            config: &PeptideFoundationMultimodalV0380Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
            let model_dim = config.forward().model_dim;
            let mut blocks = Vec::with_capacity(config.ccs_context_layers);
            for layer in 0..config.ccs_context_layers {
                blocks.push(PeptideTransformerBlock::new(
                    model_dim,
                    config.ccs_context_heads,
                    config.ccs_context_ff_dim,
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
                hidden: nn::linear(head_input, config.ccs_mobility_hidden, vb.pp("hidden"))?,
                bottleneck: nn::linear(config.ccs_mobility_hidden, 192, vb.pp("bottleneck"))?,
                output: zero_initialized_linear_v0380(192, 1, vb.pp("output"))?,
                model_dim,
            })
        }

        fn contextual_features_t(
            &self,
            foundation: &FoundationOutput,
            physics: &FoundationScalarPhysicsBatchV0360,
            train: bool,
        ) -> Result<(Tensor, Tensor)> {
            let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
            if model_dim != self.model_dim {
                candle_core::bail!("v0.38 mobility head received incompatible residue width");
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
            let pooled = masked_mean_v0380(&hidden, &foundation.residue_mask)?;
            Ok((hidden, pooled))
        }

        fn residual_from_pooled(
            &self,
            foundation: &FoundationOutput,
            physics: &FoundationScalarPhysicsBatchV0360,
            pooled: &Tensor,
        ) -> Result<Tensor> {
            let features = Tensor::cat(
                &[&foundation.peptide_embedding, pooled, &physics.ccs_physics],
                1,
            )?;
            let hidden = self.hidden.forward(&features)?.relu()?;
            let hidden = self.bottleneck.forward(&hidden)?.relu()?;
            self.output.forward(&hidden)
        }

        fn forward_t(
            &self,
            foundation: &FoundationOutput,
            physics: &FoundationScalarPhysicsBatchV0360,
            train: bool,
        ) -> Result<Tensor> {
            let (_, pooled) = self.contextual_features_t(foundation, physics, train)?;
            self.residual_from_pooled(foundation, physics, &pooled)
        }
    }

    #[derive(Debug, Clone)]
    pub struct FoundationMobilityOutputV0380 {
        /// Frozen v0.35 CCS output in the parent model's normalized CCS coordinate.
        pub base_ccs_model: Tensor,
        /// Native raw ion-mobility residual. Step 0 is exactly zero.
        pub mobility_residual_native: Tensor,
    }

    /// Frozen v0.38 mobility features exposed for later representation distillation.
    ///
    /// This does not alter the v0.38 training contract.  It only exposes the representation
    /// that already drives the accepted v0.38 mobility head so later students can learn the
    /// same mobility geometry without using DEV/HOLDOUT labels as representation targets.
    #[derive(Debug, Clone)]
    pub struct FoundationMobilityTeacherFeaturesV0380 {
        pub base_ccs_model: Tensor,
        pub mobility_residual_native: Tensor,
        pub peptide_embedding: Tensor,
        pub residue_embeddings: Tensor,
        pub residue_mask: Tensor,
        pub context_residue_embeddings: Tensor,
        pub context_pooled_embedding: Tensor,
    }

    #[derive(Clone)]
    pub struct PeptideFoundationMultimodalV0380Model {
        base_v0350: PeptideFoundationMultimodalV0350Model,
        ccs_forward_v0380: PeptideFoundationMultimodalForwardV0270,
        ccs_property_refinement_v0380: PropertyResidueRefinementV0350,
        ccs_context_v0380: MobilityContextHeadV0380,
        config: PeptideFoundationMultimodalV0380Config,
    }

    impl PeptideFoundationMultimodalV0380Model {
        pub fn new(
            config: PeptideFoundationMultimodalV0380Config,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
            config.validate()?;
            let base_v0350 =
                PeptideFoundationMultimodalV0350Model::new(config.base_v0350.clone(), vb.clone())?;
            let ccs_forward_v0380 = PeptideFoundationMultimodalForwardV0270::new(
                config.base_v0350.base_v0310.base_v0270.clone(),
                vb.pp("ccs_forward_v0380"),
            )?;
            let ccs_property_refinement_v0380 = PropertyResidueRefinementV0350::new(
                &config.base_v0350,
                vb.pp("ccs_property_refinement_v0380"),
            )?;
            let ccs_context_v0380 =
                MobilityContextHeadV0380::new(&config, vb.pp("ccs_context_v0380"))?;
            Ok(Self {
                base_v0350,
                ccs_forward_v0380,
                ccs_property_refinement_v0380,
                ccs_context_v0380,
                config,
            })
        }

        /// CCS-only trainable path. The accepted v0.35 anchor remains detached.
        pub fn mobility_v0380_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
            physics: &FoundationScalarPhysicsBatchV0360,
            train: bool,
        ) -> Result<FoundationMobilityOutputV0380> {
            let output = self.mobility_teacher_features_v0380_t(batch, context, physics, train)?;
            Ok(FoundationMobilityOutputV0380 {
                base_ccs_model: output.base_ccs_model,
                mobility_residual_native: output.mobility_residual_native,
            })
        }

        /// Expose the accepted v0.38 mobility representation for frozen teacher distillation.
        /// The returned tensors are not detached here so the method remains generally useful;
        /// callers that use v0.38 as a frozen teacher must detach them explicitly.
        pub fn mobility_teacher_features_v0380_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
            context: &PrecursorContextBatch,
            physics: &FoundationScalarPhysicsBatchV0360,
            train: bool,
        ) -> Result<FoundationMobilityTeacherFeaturesV0380> {
            let (_, _, base_ccs_model) = self
                .base_v0350
                .detached_scalar_anchor_v0350_t(batch, context)?;
            let base = self.ccs_forward_v0380.encode_foundation_t(batch, train)?;
            let foundation = self.ccs_property_refinement_v0380.forward_t(&base, train)?;
            let (context_residue_embeddings, context_pooled_embedding) = self
                .ccs_context_v0380
                .contextual_features_t(&foundation, physics, train)?;
            let mobility_residual_native = self.ccs_context_v0380.residual_from_pooled(
                &foundation,
                physics,
                &context_pooled_embedding,
            )?;
            Ok(FoundationMobilityTeacherFeaturesV0380 {
                base_ccs_model,
                mobility_residual_native,
                peptide_embedding: foundation.peptide_embedding,
                residue_embeddings: foundation.residue_embeddings,
                residue_mask: foundation.residue_mask,
                context_residue_embeddings,
                context_pooled_embedding,
            })
        }

        /// Exact protected v0.35 forward path for RT/MS2 invariance checks.
        pub fn protected_forward_v0350_t(
            &self,
            batch: &super::super::featurize::FoundationBatch,
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

        pub fn config(&self) -> &PeptideFoundationMultimodalV0380Config {
            &self.config
        }
    }

    fn masked_mean_v0380(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, length, dim) = values.dims3()?;
        let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
        let summed = values.broadcast_mul(&expanded)?.sum(1)?;
        let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
        summed.broadcast_div(&denominator)
    }

    fn zero_initialized_linear_v0380(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Linear> {
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", nn::Init::Const(0.0))?;
        let bias = vb.get_with_hints(out_dim, "bias", nn::Init::Const(0.0))?;
        Ok(Linear::new(weight, Some(bias)))
    }
}

mod deep_backbone {
    // ReDeeM v0.50 deep chemistry / residue-pair foundation encoder.
    //
    // v0.50 is intentionally a new student representation rather than a shape-compatible
    // continuation of v0.35.  The student combines a deeper residue-local atom graph with an
    // explicit O(L^2) residue-pair state and learned task tokens that participate in the same
    // interaction stack.  The accepted v0.35 model remains external to the student checkpoint and
    // is exposed only through a detached teacher adapter.

    use super::super::chemistry::ATOM_FEATURE_DIM;
    use super::super::config::FoundationConfig;
    use super::super::featurize::FoundationBatch;
    use super::super::layers::{FoundationLayerNorm, GraphMessageLayer};
    use super::super::model::PrecursorContextBatch;
    use super::forward_specialists::{
        FoundationFragmentContextBatchV0350, PeptideFoundationMultimodalV0350Model,
    };
    use candle_core::{DType, Module, ModuleT, Result, Tensor, D};
    use candle_nn::{self as nn, ops, Dropout, Embedding, Linear, VarBuilder};
    use serde::{Deserialize, Serialize};

    pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500: &str =
        "deep_chemistry_residue_pair_foundation_v0500";
    pub const FOUNDATION_V0500_STUDENT_NAMESPACE: &str = "student_v050";
    pub const FOUNDATION_V0500_TEACHER_SOURCE: &str = "external_frozen_v0350";
    pub const FOUNDATION_V0500_TASK_COUNT: usize = 4;
    pub const FOUNDATION_V0500_PAIR_CLASS_COUNT: usize = 6;
    pub const FOUNDATION_V0500_RELATIVE_FEATURE_DIM: usize = 6;
    pub const FOUNDATION_V0500_MOBILITY_CONTEXT_DIM: usize = 6;
    pub const FOUNDATION_V0500_MS2_CONTEXT_SCALAR_DIM: usize = 5;
    pub const FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM: usize = 8;

    const TASK_RT: usize = 0;
    const TASK_MOBILITY: usize = 1;
    const TASK_MS2: usize = 2;
    const TASK_GLOBAL: usize = 3;

    /// Configuration for the v0.50 student encoder.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(default)]
    pub struct PeptideFoundationV0500Config {
        pub max_sequence_len: usize,
        pub max_atoms_per_residue: usize,
        pub graph_hidden_dim: usize,
        pub graph_layers: usize,
        pub residue_dim: usize,
        pub pair_dim: usize,
        pub interaction_blocks: usize,
        pub num_attention_heads: usize,
        pub feed_forward_dim: usize,
        pub pair_feed_forward_dim: usize,
        pub dropout: f32,
        pub instrument_vocab_size: usize,
        pub instrument_dim: usize,
        pub ms2_fragment_channels: usize,
        pub contrastive_dim: usize,
    }

    impl Default for PeptideFoundationV0500Config {
        fn default() -> Self {
            Self {
                max_sequence_len: 64,
                max_atoms_per_residue: 24,
                graph_hidden_dim: 128,
                graph_layers: 5,
                residue_dim: 320,
                pair_dim: 128,
                interaction_blocks: 8,
                num_attention_heads: 8,
                feed_forward_dim: 1280,
                pair_feed_forward_dim: 512,
                dropout: 0.05,
                instrument_vocab_size: 16,
                instrument_dim: 32,
                ms2_fragment_channels: 8,
                contrastive_dim: 128,
            }
        }
    }

    impl PeptideFoundationV0500Config {
        /// Tiny CPU configuration used only for correctness tests and local smoke runs.
        pub fn local_smoke() -> Self {
            Self {
                max_sequence_len: 8,
                max_atoms_per_residue: 24,
                graph_hidden_dim: 32,
                graph_layers: 1,
                residue_dim: 64,
                pair_dim: 24,
                interaction_blocks: 1,
                num_attention_heads: 4,
                feed_forward_dim: 128,
                pair_feed_forward_dim: 64,
                dropout: 0.0,
                instrument_vocab_size: 16,
                instrument_dim: 8,
                ms2_fragment_channels: 8,
                contrastive_dim: 32,
            }
        }

        /// A100 architecture smoke: production widths/graph depth, but only two pair blocks.
        pub fn a100_smoke() -> Self {
            let mut config = Self::default();
            config.interaction_blocks = 2;
            config
        }

        pub fn validate(&self) -> Result<()> {
            if self.max_sequence_len < 2 {
                candle_core::bail!("v0.50 max_sequence_len must be at least 2");
            }
            if self.max_atoms_per_residue < 4 {
                candle_core::bail!("v0.50 max_atoms_per_residue must be at least 4");
            }
            if self.graph_hidden_dim == 0 || self.graph_layers == 0 {
                candle_core::bail!("v0.50 graph width/layers must be non-zero");
            }
            if self.residue_dim == 0 || self.pair_dim == 0 || self.interaction_blocks == 0 {
                candle_core::bail!(
                    "v0.50 residue/pair dimensions and block count must be non-zero"
                );
            }
            if self.num_attention_heads == 0 || self.residue_dim % self.num_attention_heads != 0 {
                candle_core::bail!("v0.50 residue_dim must be divisible by num_attention_heads");
            }
            if self.feed_forward_dim < self.residue_dim {
                candle_core::bail!("v0.50 feed_forward_dim must be at least residue_dim");
            }
            if self.pair_feed_forward_dim < self.pair_dim {
                candle_core::bail!("v0.50 pair_feed_forward_dim must be at least pair_dim");
            }
            if !(0.0..1.0).contains(&self.dropout) {
                candle_core::bail!("v0.50 dropout must be in [0, 1)");
            }
            if self.instrument_vocab_size == 0 || self.instrument_dim == 0 {
                candle_core::bail!("v0.50 instrument vocabulary/dimension must be non-zero");
            }
            if self.ms2_fragment_channels == 0 || self.contrastive_dim == 0 {
                candle_core::bail!("v0.50 output dimensions must be non-zero");
            }
            Ok(())
        }

        /// Compatibility config for the established chemistry featurizer only.
        ///
        /// The historical peptide Transformer dimensions are irrelevant here because v0.50 consumes
        /// the featurized atom/residue tensors directly and owns its own deep interaction stack.
        pub fn featurizer_config(&self) -> FoundationConfig {
            let mut config = FoundationConfig::default();
            config.max_sequence_len = self.max_sequence_len;
            config.max_atoms_per_residue = self.max_atoms_per_residue;
            config.graph_hidden_dim = self.graph_hidden_dim;
            config.graph_layers = self.graph_layers;
            config.instrument_vocab_size = self.instrument_vocab_size;
            config.ms2_fragment_channels = self.ms2_fragment_channels;
            config
        }

        pub fn total_token_count(&self) -> usize {
            FOUNDATION_V0500_TASK_COUNT + self.max_sequence_len
        }
    }

    /// Deep v0.50 representation exposed to property/self-supervision heads.
    #[derive(Debug, Clone)]
    pub struct FoundationRepresentationV0500 {
        /// Residue-only states `[batch, residues, residue_dim]`.
        pub residue_embeddings: Tensor,
        /// Explicit task+residue pair state `[batch, tokens, tokens, pair_dim]`.
        pub pair_embeddings: Tensor,
        /// Residue validity mask `[batch, residues]`.
        pub residue_mask: Tensor,
        /// Task+residue validity mask `[batch, tokens]`.
        pub token_mask: Tensor,
        /// Pair validity mask `[batch, tokens, tokens]`.
        pub pair_mask: Tensor,
        /// Context-independent reusable peptide representation `[batch, residue_dim]`.
        pub global_embedding: Tensor,
        /// RT task token `[batch, residue_dim]`.
        pub rt_embedding: Tensor,
        /// Mobility/CCS task token `[batch, residue_dim]`.
        pub mobility_embedding: Tensor,
        /// MS2 task token `[batch, residue_dim]`.
        pub ms2_embedding: Tensor,
        /// Mean raw atom descriptors per residue, preserved for chemistry reconstruction targets.
        pub chemistry_targets: Tensor,
    }

    #[derive(Debug, Clone)]
    pub struct FoundationMultimodalForwardOutputV0500 {
        pub representation: FoundationRepresentationV0500,
        /// RT prediction `[batch, 1]`.
        pub rt: Tensor,
        /// Native mobility prediction `[batch, 1]`. Conversion/consensus policy belongs in training.
        pub mobility_native: Tensor,
        /// Fragment intensities `[batch, max_sequence_len - 1, channels]`.
        pub ms2: Tensor,
        /// Masked-residue reconstruction logits `[batch, residues, 21]`.
        pub residue_logits: Tensor,
        /// Chemistry reconstruction `[batch, residues, ATOM_FEATURE_DIM]`.
        pub chemistry_reconstruction: Tensor,
        /// Reusable projected global representation `[batch, contrastive_dim]`.
        pub contrastive_projection: Tensor,
        /// Pair-class auxiliary logits over residue-residue pairs.
        pub pair_interaction_logits: Tensor,
        /// Deterministic peptide-chemistry summary prediction `[batch, 8]`.
        pub chemistry_summary: Tensor,
    }

    #[derive(Clone)]
    struct PairConditionedSelfAttentionV0500 {
        query: Linear,
        key: Linear,
        value: Linear,
        output: Linear,
        pair_bias_heads: Vec<Linear>,
        num_heads: usize,
        head_dim: usize,
    }

    impl PairConditionedSelfAttentionV0500 {
        fn new(
            residue_dim: usize,
            pair_dim: usize,
            num_heads: usize,
            vb: VarBuilder<'_>,
        ) -> Result<Self> {
            let head_dim = residue_dim / num_heads;
            let pair_bias_heads = (0..num_heads)
                .map(|head| nn::linear_no_bias(pair_dim, 1, vb.pp(format!("pair_bias.{head}"))))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                query: nn::linear_no_bias(residue_dim, residue_dim, vb.pp("query"))?,
                key: nn::linear_no_bias(residue_dim, residue_dim, vb.pp("key"))?,
                value: nn::linear_no_bias(residue_dim, residue_dim, vb.pp("value"))?,
                output: nn::linear(residue_dim, residue_dim, vb.pp("output"))?,
                pair_bias_heads,
                num_heads,
                head_dim,
            })
        }

        fn forward(&self, hidden: &Tensor, pair: &Tensor, token_mask: &Tensor) -> Result<Tensor> {
            let (batch, tokens, residue_dim) = hidden.dims3()?;
            let (pair_batch, pair_i, pair_j, pair_dim) = pair.dims4()?;
            if pair_batch != batch || pair_i != tokens || pair_j != tokens {
                candle_core::bail!("v0.50 pair-conditioned attention shape mismatch");
            }

            // Candle's CUDA batched Linear/matmul path requires contiguous inputs.
            // Task-token concatenation, narrow views, and layer-norm elementwise ops can preserve
            // strided layouts that the CPU backend accepts but CUDA rejects. Materialize once at
            // every projection boundary that can receive such a view.
            let hidden = hidden.contiguous()?;
            let q = self
                .query
                .forward(&hidden)?
                .reshape((batch, tokens, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = self
                .key
                .forward(&hidden)?
                .reshape((batch, tokens, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = self
                .value
                .forward(&hidden)?
                .reshape((batch, tokens, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let mut scores = q
                .matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;

            let pair_flat = pair
                .reshape((batch * tokens * tokens, pair_dim))?
                .contiguous()?;
            let mut per_head = Vec::with_capacity(self.num_heads);
            for projection in &self.pair_bias_heads {
                per_head.push(
                    projection
                        .forward(&pair_flat)?
                        .reshape((batch, tokens, tokens))?
                        .unsqueeze(1)?,
                );
            }
            let refs = per_head.iter().collect::<Vec<_>>();
            let pair_bias = Tensor::cat(&refs, 1)?;
            scores = (scores + pair_bias)?;

            let key_mask = token_mask
                .affine(-1.0, 1.0)?
                .affine(-10_000.0, 0.0)?
                .unsqueeze(1)?
                .unsqueeze(1)?
                .broadcast_as((batch, self.num_heads, tokens, tokens))?;
            let probabilities = ops::softmax(&(scores + key_mask)?, D::Minus1)?;
            let context = probabilities
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, tokens, residue_dim))?
                .contiguous()?;
            let output = self.output.forward(&context)?;
            let query_mask = token_mask
                .unsqueeze(2)?
                .broadcast_as((batch, tokens, residue_dim))?;
            output.broadcast_mul(&query_mask)
        }
    }

    #[derive(Clone)]
    struct ResiduePairInteractionBlockV0500 {
        residue_attention_norm: FoundationLayerNorm,
        pair_attention_norm: FoundationLayerNorm,
        attention: PairConditionedSelfAttentionV0500,
        residue_to_pair_norm: FoundationLayerNorm,
        pair_update_left: Linear,
        pair_update_right: Linear,
        pair_update_out: Linear,
        pair_transition_norm: FoundationLayerNorm,
        pair_transition_in: Linear,
        pair_transition_out: Linear,
        residue_transition_norm: FoundationLayerNorm,
        residue_transition_in: Linear,
        residue_transition_out: Linear,
        dropout: Dropout,
        residue_dim: usize,
        pair_dim: usize,
    }

    impl ResiduePairInteractionBlockV0500 {
        fn new(config: &PeptideFoundationV0500Config, vb: VarBuilder<'_>) -> Result<Self> {
            Ok(Self {
                residue_attention_norm: FoundationLayerNorm::new(
                    config.residue_dim,
                    1e-5,
                    vb.pp("residue_attention_norm"),
                )?,
                pair_attention_norm: FoundationLayerNorm::new(
                    config.pair_dim,
                    1e-5,
                    vb.pp("pair_attention_norm"),
                )?,
                attention: PairConditionedSelfAttentionV0500::new(
                    config.residue_dim,
                    config.pair_dim,
                    config.num_attention_heads,
                    vb.pp("attention"),
                )?,
                residue_to_pair_norm: FoundationLayerNorm::new(
                    config.residue_dim,
                    1e-5,
                    vb.pp("residue_to_pair_norm"),
                )?,
                pair_update_left: nn::linear(
                    config.residue_dim,
                    config.pair_dim,
                    vb.pp("pair_update_left"),
                )?,
                pair_update_right: nn::linear(
                    config.residue_dim,
                    config.pair_dim,
                    vb.pp("pair_update_right"),
                )?,
                pair_update_out: nn::linear(
                    config.pair_dim,
                    config.pair_dim,
                    vb.pp("pair_update_out"),
                )?,
                pair_transition_norm: FoundationLayerNorm::new(
                    config.pair_dim,
                    1e-5,
                    vb.pp("pair_transition_norm"),
                )?,
                pair_transition_in: nn::linear(
                    config.pair_dim,
                    config.pair_feed_forward_dim,
                    vb.pp("pair_transition_in"),
                )?,
                pair_transition_out: nn::linear(
                    config.pair_feed_forward_dim,
                    config.pair_dim,
                    vb.pp("pair_transition_out"),
                )?,
                residue_transition_norm: FoundationLayerNorm::new(
                    config.residue_dim,
                    1e-5,
                    vb.pp("residue_transition_norm"),
                )?,
                residue_transition_in: nn::linear(
                    config.residue_dim,
                    config.feed_forward_dim,
                    vb.pp("residue_transition_in"),
                )?,
                residue_transition_out: nn::linear(
                    config.feed_forward_dim,
                    config.residue_dim,
                    vb.pp("residue_transition_out"),
                )?,
                dropout: Dropout::new(config.dropout),
                residue_dim: config.residue_dim,
                pair_dim: config.pair_dim,
            })
        }

        fn forward_t(
            &self,
            hidden: &Tensor,
            pair: &Tensor,
            token_mask: &Tensor,
            pair_mask: &Tensor,
            train: bool,
        ) -> Result<(Tensor, Tensor)> {
            let (batch, tokens, residue_dim) = hidden.dims3()?;
            if residue_dim != self.residue_dim {
                candle_core::bail!("v0.50 interaction block residue width mismatch");
            }
            let (_, pair_i, pair_j, pair_dim) = pair.dims4()?;
            if pair_i != tokens || pair_j != tokens || pair_dim != self.pair_dim {
                candle_core::bail!("v0.50 interaction block pair shape mismatch");
            }

            // 1. Pair-conditioned residue self-attention.
            let normalized_hidden = self.residue_attention_norm.forward(hidden)?;
            let normalized_pair = self.pair_attention_norm.forward(pair)?;
            let attention =
                self.attention
                    .forward(&normalized_hidden, &normalized_pair, token_mask)?;
            let mut hidden = (hidden + self.dropout.forward_t(&attention, train)?)?;
            hidden = mask_token_state(&hidden, token_mask)?;

            // 2. Residue-to-pair update.  Left/right projections plus a multiplicative
            // interaction give the pair state an explicit place to encode compatibility.
            let normalized_hidden = self.residue_to_pair_norm.forward(&hidden)?.contiguous()?;
            let left = self
                .pair_update_left
                .forward(&normalized_hidden)?
                .unsqueeze(2)?
                .broadcast_as((batch, tokens, tokens, self.pair_dim))?;
            let right = self
                .pair_update_right
                .forward(&normalized_hidden)?
                .unsqueeze(1)?
                .broadcast_as((batch, tokens, tokens, self.pair_dim))?;
            let product = left.broadcast_mul(&right)?;
            let residue_pair = ((&left + &right)? + product)?.relu()?;
            let pair_update = self
                .pair_update_out
                .forward(
                    &residue_pair
                        .reshape((batch * tokens * tokens, self.pair_dim))?
                        .contiguous()?,
                )?
                .reshape((batch, tokens, tokens, self.pair_dim))?;
            let mut pair = (pair + self.dropout.forward_t(&pair_update, train)?)?;
            pair = mask_pair_state(&pair, pair_mask)?;

            // 3. Pair transition / MLP.
            let normalized_pair = self.pair_transition_norm.forward(&pair)?;
            let flat = normalized_pair
                .reshape((batch * tokens * tokens, self.pair_dim))?
                .contiguous()?;
            let pair_transition = self.pair_transition_in.forward(&flat)?.relu()?;
            let pair_transition = self
                .pair_transition_out
                .forward(&pair_transition)?
                .reshape((batch, tokens, tokens, self.pair_dim))?;
            pair = (&pair + self.dropout.forward_t(&pair_transition, train)?)?;
            pair = mask_pair_state(&pair, pair_mask)?;

            // 4. Residue transition / MLP.
            let normalized_hidden = self
                .residue_transition_norm
                .forward(&hidden)?
                .contiguous()?;
            let residue_transition = self
                .residue_transition_in
                .forward(&normalized_hidden)?
                .relu()?;
            let residue_transition = self.residue_transition_out.forward(&residue_transition)?;
            hidden = (&hidden + self.dropout.forward_t(&residue_transition, train)?)?;
            hidden = mask_token_state(&hidden, token_mask)?;

            Ok((hidden, pair))
        }
    }

    /// v0.50 student model.  All trainable parameters live under `student_v050.*`.
    #[derive(Clone)]
    pub struct PeptideFoundationV0500Model {
        config: PeptideFoundationV0500Config,
        atom_input: Linear,
        graph_layers: Vec<GraphMessageLayer>,
        graph_to_residue: Linear,
        chemistry_to_residue: Linear,
        residue_embedding: Embedding,
        position_embedding: Embedding,
        residue_input_norm: FoundationLayerNorm,
        task_embedding: Embedding,
        mobility_context_projection: Linear,
        instrument_embedding: Embedding,
        ms2_context_projection: Linear,
        pair_left: Linear,
        pair_right: Linear,
        pair_chemistry_left: Linear,
        pair_chemistry_right: Linear,
        pair_relative_projection: Linear,
        pair_input_norm: FoundationLayerNorm,
        interaction_blocks: Vec<ResiduePairInteractionBlockV0500>,
        residue_output_norm: FoundationLayerNorm,
        pair_output_norm: FoundationLayerNorm,
        rt_head_hidden: Linear,
        rt_head_output: Linear,
        mobility_head_hidden: Linear,
        mobility_head_output: Linear,
        ms2_head_hidden: Linear,
        ms2_head_output: Linear,
        residue_head: Linear,
        chemistry_head: Linear,
        contrastive_head: Linear,
        pair_interaction_head: Linear,
        chemistry_summary_head: Linear,
    }

    impl PeptideFoundationV0500Model {
        pub fn new(config: PeptideFoundationV0500Config, vb: VarBuilder<'_>) -> Result<Self> {
            config.validate()?;
            let vb = vb.pp(FOUNDATION_V0500_STUDENT_NAMESPACE);
            let atom_input = nn::linear(
                ATOM_FEATURE_DIM,
                config.graph_hidden_dim,
                vb.pp("chemistry.atom_input"),
            )?;
            let graph_layers = (0..config.graph_layers)
                .map(|layer| {
                    GraphMessageLayer::new(
                        config.graph_hidden_dim,
                        vb.pp(format!("chemistry.graph.{layer}")),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let graph_to_residue = nn::linear(
                config.graph_hidden_dim,
                config.residue_dim,
                vb.pp("chemistry.graph_to_residue"),
            )?;
            let chemistry_to_residue = nn::linear(
                ATOM_FEATURE_DIM,
                config.residue_dim,
                vb.pp("chemistry.raw_to_residue"),
            )?;
            let residue_embedding =
                nn::embedding(21, config.residue_dim, vb.pp("sequence.residue_embedding"))?;
            let position_embedding = nn::embedding(
                config.max_sequence_len,
                config.residue_dim,
                vb.pp("sequence.position_embedding"),
            )?;
            let residue_input_norm =
                FoundationLayerNorm::new(config.residue_dim, 1e-5, vb.pp("sequence.input_norm"))?;
            let task_embedding = nn::embedding(
                FOUNDATION_V0500_TASK_COUNT,
                config.residue_dim,
                vb.pp("task.embedding"),
            )?;
            let mobility_context_projection = nn::linear(
                FOUNDATION_V0500_MOBILITY_CONTEXT_DIM,
                config.residue_dim,
                vb.pp("task.mobility_context"),
            )?;
            let instrument_embedding = nn::embedding(
                config.instrument_vocab_size,
                config.instrument_dim,
                vb.pp("task.ms2_instrument_embedding"),
            )?;
            let ms2_context_projection = nn::linear(
                config.instrument_dim + FOUNDATION_V0500_MS2_CONTEXT_SCALAR_DIM,
                config.residue_dim,
                vb.pp("task.ms2_context"),
            )?;
            let pair_left =
                nn::linear(config.residue_dim, config.pair_dim, vb.pp("pair.init_left"))?;
            let pair_right = nn::linear(
                config.residue_dim,
                config.pair_dim,
                vb.pp("pair.init_right"),
            )?;
            let pair_chemistry_left = nn::linear(
                ATOM_FEATURE_DIM,
                config.pair_dim,
                vb.pp("pair.chemistry_left"),
            )?;
            let pair_chemistry_right = nn::linear(
                ATOM_FEATURE_DIM,
                config.pair_dim,
                vb.pp("pair.chemistry_right"),
            )?;
            let pair_relative_projection = nn::linear(
                FOUNDATION_V0500_RELATIVE_FEATURE_DIM,
                config.pair_dim,
                vb.pp("pair.relative_projection"),
            )?;
            let pair_input_norm =
                FoundationLayerNorm::new(config.pair_dim, 1e-5, vb.pp("pair.input_norm"))?;
            let interaction_blocks = (0..config.interaction_blocks)
                .map(|block| {
                    ResiduePairInteractionBlockV0500::new(
                        &config,
                        vb.pp(format!("interaction.{block}")),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let residue_output_norm =
                FoundationLayerNorm::new(config.residue_dim, 1e-5, vb.pp("output.residue_norm"))?;
            let pair_output_norm =
                FoundationLayerNorm::new(config.pair_dim, 1e-5, vb.pp("output.pair_norm"))?;

            Ok(Self {
                rt_head_hidden: nn::linear(
                    config.residue_dim,
                    config.residue_dim,
                    vb.pp("heads.rt.hidden"),
                )?,
                rt_head_output: nn::linear(config.residue_dim, 1, vb.pp("heads.rt.output"))?,
                mobility_head_hidden: nn::linear(
                    config.residue_dim,
                    config.residue_dim,
                    vb.pp("heads.mobility.hidden"),
                )?,
                mobility_head_output: nn::linear(
                    config.residue_dim,
                    1,
                    vb.pp("heads.mobility.output"),
                )?,
                ms2_head_hidden: nn::linear(
                    3 * config.residue_dim,
                    config.feed_forward_dim,
                    vb.pp("heads.ms2.hidden"),
                )?,
                ms2_head_output: nn::linear(
                    config.feed_forward_dim,
                    config.ms2_fragment_channels,
                    vb.pp("heads.ms2.output"),
                )?,
                residue_head: nn::linear(config.residue_dim, 21, vb.pp("heads.residue"))?,
                chemistry_head: nn::linear(
                    config.residue_dim,
                    ATOM_FEATURE_DIM,
                    vb.pp("heads.chemistry"),
                )?,
                contrastive_head: nn::linear(
                    config.residue_dim,
                    config.contrastive_dim,
                    vb.pp("heads.contrastive"),
                )?,
                pair_interaction_head: nn::linear(
                    config.pair_dim,
                    FOUNDATION_V0500_PAIR_CLASS_COUNT,
                    vb.pp("heads.pair_interaction"),
                )?,
                chemistry_summary_head: nn::linear(
                    config.residue_dim,
                    FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM,
                    vb.pp("heads.chemistry_summary"),
                )?,
                config,
                atom_input,
                graph_layers,
                graph_to_residue,
                chemistry_to_residue,
                residue_embedding,
                position_embedding,
                residue_input_norm,
                task_embedding,
                mobility_context_projection,
                instrument_embedding,
                ms2_context_projection,
                pair_left,
                pair_right,
                pair_chemistry_left,
                pair_chemistry_right,
                pair_relative_projection,
                pair_input_norm,
                interaction_blocks,
                residue_output_norm,
                pair_output_norm,
            })
        }

        pub fn config(&self) -> &PeptideFoundationV0500Config {
            &self.config
        }

        /// Encode only the reusable deep chemistry/residue-pair representation.
        ///
        /// This is an additive entry point for downstream representation-learning experiments that
        /// do not need the historical v0.50 RT/MS2/reconstruction heads. `forward_t` remains
        /// unchanged, preserving all historical v0.50/v0.51/v0.52 behavior.
        pub fn representation_t(
            &self,
            batch: &FoundationBatch,
            context: &PrecursorContextBatch,
            train: bool,
        ) -> Result<FoundationRepresentationV0500> {
            let (batch_size, sequence_len, _atom_count, feature_dim) =
                batch.atom_features.dims4()?;
            if sequence_len != self.config.max_sequence_len {
                candle_core::bail!(
                    "v0.50 expected sequence width {}, got {}",
                    self.config.max_sequence_len,
                    sequence_len
                );
            }
            if feature_dim != ATOM_FEATURE_DIM {
                candle_core::bail!(
                    "v0.50 atom feature mismatch: expected {}, got {}",
                    ATOM_FEATURE_DIM,
                    feature_dim
                );
            }

            let chemistry_targets = mean_raw_chemistry(batch)?;
            let mut residues = self.encode_residues(batch, &chemistry_targets)?;
            residues = mask_token_state(&residues, &batch.residue_mask)?;
            let task_tokens = self.contextual_task_tokens(batch_size, context)?;
            let mut hidden = Tensor::cat(&[&task_tokens, &residues], 1)?.contiguous()?;
            // Representation-only consumers do not need RT/MS2 acquisition task tokens. Keep only
            // the mobility and global tokens active so unrelated RT/NCE/instrument context cannot
            // leak into a mobility-specific latent through self-attention. Historical `forward_t`
            // below still uses all four task tokens exactly as before.
            let mut task_mask_values = Vec::with_capacity(batch_size * FOUNDATION_V0500_TASK_COUNT);
            for _ in 0..batch_size {
                task_mask_values.extend_from_slice(&[0.0f32, 1.0, 0.0, 1.0]);
            }
            let task_mask = Tensor::from_vec(
                task_mask_values,
                (batch_size, FOUNDATION_V0500_TASK_COUNT),
                batch.residue_mask.device(),
            )?;
            let token_mask = Tensor::cat(&[&task_mask, &batch.residue_mask], 1)?;
            let pair_mask = token_mask
                .unsqueeze(2)?
                .broadcast_mul(&token_mask.unsqueeze(1)?)?;
            let mut pair = self.initialize_pair_state(&hidden, &chemistry_targets, &pair_mask)?;
            for block in &self.interaction_blocks {
                (hidden, pair) = block.forward_t(&hidden, &pair, &token_mask, &pair_mask, train)?;
            }
            hidden = self.residue_output_norm.forward(&hidden)?;
            hidden = mask_token_state(&hidden, &token_mask)?;
            pair = self.pair_output_norm.forward(&pair)?;
            pair = mask_pair_state(&pair, &pair_mask)?;

            let rt_embedding = hidden.narrow(1, TASK_RT, 1)?.squeeze(1)?.contiguous()?;
            let mobility_embedding = hidden
                .narrow(1, TASK_MOBILITY, 1)?
                .squeeze(1)?
                .contiguous()?;
            let ms2_embedding = hidden.narrow(1, TASK_MS2, 1)?.squeeze(1)?.contiguous()?;
            let global_embedding = hidden.narrow(1, TASK_GLOBAL, 1)?.squeeze(1)?.contiguous()?;
            let residue_embeddings = hidden
                .narrow(1, FOUNDATION_V0500_TASK_COUNT, self.config.max_sequence_len)?
                .contiguous()?;

            Ok(FoundationRepresentationV0500 {
                residue_embeddings,
                pair_embeddings: pair,
                residue_mask: batch.residue_mask.clone(),
                token_mask,
                pair_mask,
                global_embedding,
                rt_embedding,
                mobility_embedding,
                ms2_embedding,
                chemistry_targets,
            })
        }

        pub fn forward_t(
            &self,
            batch: &FoundationBatch,
            context: &PrecursorContextBatch,
            train: bool,
        ) -> Result<FoundationMultimodalForwardOutputV0500> {
            let (batch_size, sequence_len, atom_count, feature_dim) =
                batch.atom_features.dims4()?;
            if sequence_len != self.config.max_sequence_len {
                candle_core::bail!(
                    "v0.50 expected sequence width {}, got {}",
                    self.config.max_sequence_len,
                    sequence_len
                );
            }
            if feature_dim != ATOM_FEATURE_DIM {
                candle_core::bail!(
                    "v0.50 atom feature mismatch: expected {}, got {}",
                    ATOM_FEATURE_DIM,
                    feature_dim
                );
            }

            let chemistry_targets = mean_raw_chemistry(batch)?;
            let mut residues = self.encode_residues(batch, &chemistry_targets)?;
            residues = mask_token_state(&residues, &batch.residue_mask)?;

            let task_tokens = self.contextual_task_tokens(batch_size, context)?;
            // `Tensor::cat` may retain a strided layout when one input originates from a broadcast.
            // CUDA Linear requires the combined task+residue state to be contiguous.
            let mut hidden = Tensor::cat(&[&task_tokens, &residues], 1)?.contiguous()?;
            let task_mask = Tensor::ones(
                (batch_size, FOUNDATION_V0500_TASK_COUNT),
                DType::F32,
                batch.residue_mask.device(),
            )?;
            let token_mask = Tensor::cat(&[&task_mask, &batch.residue_mask], 1)?;
            let pair_mask = token_mask
                .unsqueeze(2)?
                .broadcast_mul(&token_mask.unsqueeze(1)?)?;

            let mut pair = self.initialize_pair_state(&hidden, &chemistry_targets, &pair_mask)?;
            for block in &self.interaction_blocks {
                (hidden, pair) = block.forward_t(&hidden, &pair, &token_mask, &pair_mask, train)?;
            }
            hidden = self.residue_output_norm.forward(&hidden)?;
            hidden = mask_token_state(&hidden, &token_mask)?;
            pair = self.pair_output_norm.forward(&pair)?;
            pair = mask_pair_state(&pair, &pair_mask)?;

            // Narrow/squeeze produces views. Materialize task and residue slices before the
            // property/reconstruction heads so GPU Linear never receives a strided matrix.
            let rt_embedding = hidden.narrow(1, TASK_RT, 1)?.squeeze(1)?.contiguous()?;
            let mobility_embedding = hidden
                .narrow(1, TASK_MOBILITY, 1)?
                .squeeze(1)?
                .contiguous()?;
            let ms2_embedding = hidden.narrow(1, TASK_MS2, 1)?.squeeze(1)?.contiguous()?;
            let global_embedding = hidden.narrow(1, TASK_GLOBAL, 1)?.squeeze(1)?.contiguous()?;
            let residue_embeddings = hidden
                .narrow(1, FOUNDATION_V0500_TASK_COUNT, self.config.max_sequence_len)?
                .contiguous()?;

            let rt = self
                .rt_head_output
                .forward(&self.rt_head_hidden.forward(&rt_embedding)?.relu()?)?;
            let mobility_native = self.mobility_head_output.forward(
                &self
                    .mobility_head_hidden
                    .forward(&mobility_embedding)?
                    .relu()?,
            )?;
            let ms2 = self.ms2_from_representation(&residue_embeddings, &ms2_embedding, batch)?;
            let residue_logits = self.residue_head.forward(&residue_embeddings)?;
            let chemistry_reconstruction = self.chemistry_head.forward(&residue_embeddings)?;
            let contrastive_projection = self.contrastive_head.forward(&global_embedding)?;
            let chemistry_summary = self.chemistry_summary_head.forward(&global_embedding)?;

            let residue_pair =
                pair.narrow(1, FOUNDATION_V0500_TASK_COUNT, self.config.max_sequence_len)?;
            let residue_pair = residue_pair.narrow(
                2,
                FOUNDATION_V0500_TASK_COUNT,
                self.config.max_sequence_len,
            )?;
            let pair_flat = residue_pair
                .reshape((
                    batch_size * self.config.max_sequence_len * self.config.max_sequence_len,
                    self.config.pair_dim,
                ))?
                .contiguous()?;
            let pair_interaction_logits =
                self.pair_interaction_head.forward(&pair_flat)?.reshape((
                    batch_size,
                    self.config.max_sequence_len,
                    self.config.max_sequence_len,
                    FOUNDATION_V0500_PAIR_CLASS_COUNT,
                ))?;
            let residue_pair_mask = batch
                .residue_mask
                .unsqueeze(2)?
                .broadcast_mul(&batch.residue_mask.unsqueeze(1)?)?
                .unsqueeze(3)?
                .broadcast_as((
                    batch_size,
                    self.config.max_sequence_len,
                    self.config.max_sequence_len,
                    FOUNDATION_V0500_PAIR_CLASS_COUNT,
                ))?;
            let pair_interaction_logits =
                pair_interaction_logits.broadcast_mul(&residue_pair_mask)?;

            Ok(FoundationMultimodalForwardOutputV0500 {
                representation: FoundationRepresentationV0500 {
                    residue_embeddings,
                    pair_embeddings: pair,
                    residue_mask: batch.residue_mask.clone(),
                    token_mask,
                    pair_mask,
                    global_embedding,
                    rt_embedding,
                    mobility_embedding,
                    ms2_embedding,
                    chemistry_targets,
                },
                rt,
                mobility_native,
                ms2,
                residue_logits,
                chemistry_reconstruction,
                contrastive_projection,
                pair_interaction_logits,
                chemistry_summary,
            })
        }

        fn encode_residues(
            &self,
            batch: &FoundationBatch,
            chemistry_targets: &Tensor,
        ) -> Result<Tensor> {
            let (batch_size, sequence_len, atom_count, feature_dim) =
                batch.atom_features.dims4()?;
            let graph_count = batch_size * sequence_len;
            let atoms = batch
                .atom_features
                .reshape((graph_count, atom_count, feature_dim))?;
            let adjacency = batch
                .adjacency
                .reshape((graph_count, atom_count, atom_count))?;
            let atom_mask = batch.atom_mask.reshape((graph_count, atom_count))?;
            let mut hidden = self.atom_input.forward(&atoms)?;
            hidden = hidden.broadcast_mul(&atom_mask.unsqueeze(2)?.broadcast_as((
                graph_count,
                atom_count,
                self.config.graph_hidden_dim,
            ))?)?;
            for layer in &self.graph_layers {
                hidden = layer.forward(&hidden, &adjacency, &atom_mask)?;
            }
            let atom_sum = hidden.sum(1)?;
            let atom_denominator = atom_mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
            let graph_residue = self
                .graph_to_residue
                .forward(&atom_sum.broadcast_div(&atom_denominator)?)?
                .reshape((batch_size, sequence_len, self.config.residue_dim))?;
            let raw_chemistry = self.chemistry_to_residue.forward(chemistry_targets)?;
            let identity = self.residue_embedding.forward(&batch.residue_ids)?;
            let positions: Vec<u32> = (0..sequence_len as u32).collect();
            let position_ids =
                Tensor::from_vec(positions, sequence_len, batch.residue_ids.device())?
                    .to_dtype(DType::U32)?;
            let position = self
                .position_embedding
                .forward(&position_ids)?
                .unsqueeze(0)?
                .broadcast_as((batch_size, sequence_len, self.config.residue_dim))?;
            let residue = ((graph_residue + raw_chemistry)? + identity)?;
            self.residue_input_norm.forward(&(residue + position)?)
        }

        fn contextual_task_tokens(
            &self,
            batch_size: usize,
            context: &PrecursorContextBatch,
        ) -> Result<Tensor> {
            let device = context.charge.device();
            let ids = Tensor::from_vec(
                vec![
                    TASK_RT as u32,
                    TASK_MOBILITY as u32,
                    TASK_MS2 as u32,
                    TASK_GLOBAL as u32,
                ],
                FOUNDATION_V0500_TASK_COUNT,
                device,
            )?
            .to_dtype(DType::U32)?;
            let tasks = self
                .task_embedding
                .forward(&ids)?
                .unsqueeze(0)?
                .broadcast_as((
                    batch_size,
                    FOUNDATION_V0500_TASK_COUNT,
                    self.config.residue_dim,
                ))?;

            let known_charge = context.charge.broadcast_mul(&context.charge_present)?;
            let known_mz = context
                .precursor_mz
                .broadcast_mul(&context.precursor_mz_present)?;
            let physical_present = context
                .charge_present
                .broadcast_mul(&context.precursor_mz_present)?;
            let neutral_mass_proxy = known_charge
                .broadcast_mul(&known_mz)?
                .broadcast_mul(&physical_present)?;
            let mobility_context = Tensor::cat(
                &[
                    &known_charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?,
                    &context.charge_present.unsqueeze(1)?,
                    &known_mz.affine(1.0 / 1000.0, 0.0)?.unsqueeze(1)?,
                    &context.precursor_mz_present.unsqueeze(1)?,
                    &neutral_mass_proxy.affine(1.0 / 3000.0, 0.0)?.unsqueeze(1)?,
                    &physical_present.unsqueeze(1)?,
                ],
                1,
            )?;
            let mobility_context = self
                .mobility_context_projection
                .forward(&mobility_context)?
                .unsqueeze(1)?;

            let instrument = self.instrument_embedding.forward(&context.instrument_ids)?;
            let ms2_scalars = Tensor::cat(
                &[
                    &known_charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?,
                    &context.charge_present.unsqueeze(1)?,
                    &context.nce.affine(1.0 / 100.0, 0.0)?.unsqueeze(1)?,
                    &context.nce_present.unsqueeze(1)?,
                    &context.instrument_present.unsqueeze(1)?,
                ],
                1,
            )?;
            let ms2_context = self
                .ms2_context_projection
                .forward(&Tensor::cat(&[&instrument, &ms2_scalars], 1)?)?
                .unsqueeze(1)?;

            let rt = tasks.narrow(1, TASK_RT, 1)?;
            let mobility = (tasks.narrow(1, TASK_MOBILITY, 1)? + mobility_context)?;
            let ms2 = (tasks.narrow(1, TASK_MS2, 1)? + ms2_context)?;
            let global = tasks.narrow(1, TASK_GLOBAL, 1)?;
            Tensor::cat(&[&rt, &mobility, &ms2, &global], 1)
        }

        fn initialize_pair_state(
            &self,
            hidden: &Tensor,
            chemistry_targets: &Tensor,
            pair_mask: &Tensor,
        ) -> Result<Tensor> {
            let (batch, tokens, _) = hidden.dims3()?;
            let sequence = self.config.max_sequence_len;
            let pair_dim = self.config.pair_dim;
            let hidden = hidden.contiguous()?;

            let left = self
                .pair_left
                .forward(&hidden)?
                .unsqueeze(2)?
                .broadcast_as((batch, tokens, tokens, pair_dim))?;
            let right = self
                .pair_right
                .forward(&hidden)?
                .unsqueeze(1)?
                .broadcast_as((batch, tokens, tokens, pair_dim))?;

            // PTM mass/composition and terminal chemistry are already present in the raw atom
            // descriptors. Project them explicitly into pair initialization instead of relying only
            // on residue identity attention.
            let chemistry_left = self.pair_chemistry_left.forward(chemistry_targets)?;
            let chemistry_right = self.pair_chemistry_right.forward(chemistry_targets)?;
            let task_zeros = Tensor::zeros(
                (batch, FOUNDATION_V0500_TASK_COUNT, pair_dim),
                DType::F32,
                hidden.device(),
            )?;
            let chemistry_left = Tensor::cat(&[&task_zeros, &chemistry_left], 1)?
                .unsqueeze(2)?
                .broadcast_as((batch, tokens, tokens, pair_dim))?;
            let chemistry_right = Tensor::cat(&[&task_zeros, &chemistry_right], 1)?
                .unsqueeze(1)?
                .broadcast_as((batch, tokens, tokens, pair_dim))?;

            let relative = relative_pair_features_v0500(sequence, hidden.device())?;
            let relative = self
                .pair_relative_projection
                .forward(
                    &relative
                        .reshape((tokens * tokens, FOUNDATION_V0500_RELATIVE_FEATURE_DIM))?
                        .contiguous()?,
                )?
                .reshape((1, tokens, tokens, pair_dim))?
                .broadcast_as((batch, tokens, tokens, pair_dim))?;

            let pair = (((left + right)? + chemistry_left)? + chemistry_right)?;
            let pair = self.pair_input_norm.forward(&(pair + relative)?)?;
            mask_pair_state(&pair, pair_mask)
        }

        fn ms2_from_representation(
            &self,
            residues: &Tensor,
            ms2_task: &Tensor,
            batch: &FoundationBatch,
        ) -> Result<Tensor> {
            let (batch_size, sequence_len, residue_dim) = residues.dims3()?;
            let cleavages = sequence_len - 1;
            let left = residues.narrow(1, 0, cleavages)?;
            let right = residues.narrow(1, 1, cleavages)?;
            let task = ms2_task
                .unsqueeze(1)?
                .broadcast_as((batch_size, cleavages, residue_dim))?;
            let features = Tensor::cat(&[&left, &right, &task], 2)?.contiguous()?;
            let hidden = self
                .ms2_head_hidden
                .forward(
                    &features
                        .reshape((batch_size * cleavages, 3 * residue_dim))?
                        .contiguous()?,
                )?
                .relu()?;
            let prediction = self.ms2_head_output.forward(&hidden)?.relu()?.reshape((
                batch_size,
                cleavages,
                self.config.ms2_fragment_channels,
            ))?;
            let cleavage_mask = batch
                .residue_mask
                .narrow(1, 0, cleavages)?
                .broadcast_mul(&batch.residue_mask.narrow(1, 1, cleavages)?)?
                .unsqueeze(2)?
                .broadcast_as((batch_size, cleavages, self.config.ms2_fragment_channels))?;
            prediction.broadcast_mul(&cleavage_mask)
        }
    }

    /// Detached outputs from the external, authoritative v0.35 teacher.
    #[derive(Debug, Clone)]
    pub struct FoundationTeacherAnchorOutputV0500 {
        pub rt: Tensor,
        pub ccs: Tensor,
        pub ms2: Tensor,
        pub peptide_embedding: Tensor,
    }

    /// Frozen-teacher adapter used by v0.50 training.
    ///
    /// The adapter owns a separately constructed v0.35 model.  It never shares the student VarMap,
    /// and every returned teacher tensor is detached.  This makes teacher checkpoint loading and
    /// student checkpoint serialization independent and prevents accidental optimization of v0.35.
    #[derive(Clone)]
    pub struct FoundationV0350TeacherV0500 {
        model: PeptideFoundationMultimodalV0350Model,
    }

    impl FoundationV0350TeacherV0500 {
        pub fn new(model: PeptideFoundationMultimodalV0350Model) -> Self {
            Self { model }
        }

        pub fn forward_detached_t(
            &self,
            batch: &FoundationBatch,
            context: &PrecursorContextBatch,
            fragment: &FoundationFragmentContextBatchV0350,
        ) -> Result<FoundationTeacherAnchorOutputV0500> {
            let output = self
                .model
                .forward_v0350_t(batch, context, fragment, false)?;
            Ok(FoundationTeacherAnchorOutputV0500 {
                rt: output.base.rt.detach(),
                ccs: output.base.ccs.detach(),
                ms2: output.base.ms2.detach(),
                peptide_embedding: output.base.foundation.peptide_embedding.detach(),
            })
        }
    }

    fn mean_raw_chemistry(batch: &FoundationBatch) -> Result<Tensor> {
        let raw_sum = batch.atom_features.sum(2)?;
        let denominator = batch
            .atom_mask
            .sum(2)?
            .clamp(1.0, f64::INFINITY)?
            .unsqueeze(2)?;
        raw_sum.broadcast_div(&denominator)
    }

    fn mask_token_state(hidden: &Tensor, token_mask: &Tensor) -> Result<Tensor> {
        let (batch, tokens, dim) = hidden.dims3()?;
        hidden.broadcast_mul(
            &token_mask
                .unsqueeze(2)?
                .broadcast_as((batch, tokens, dim))?,
        )
    }

    fn mask_pair_state(pair: &Tensor, pair_mask: &Tensor) -> Result<Tensor> {
        let (batch, left, right, dim) = pair.dims4()?;
        pair.broadcast_mul(
            &pair_mask
                .unsqueeze(3)?
                .broadcast_as((batch, left, right, dim))?,
        )
    }

    /// Fixed relative descriptors for task+residue pair initialization.
    ///
    /// Residue-residue entries encode signed and absolute sequence separation explicitly.  Task
    /// entries receive the same normalized coordinate system plus task/residue indicator features;
    /// the learned task embeddings distinguish RT, mobility, MS2, and global roles.
    fn relative_pair_features_v0500(
        sequence_len: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let tokens = FOUNDATION_V0500_TASK_COUNT + sequence_len;
        let denom = sequence_len.max(1) as f32;
        let mut values =
            Vec::with_capacity(tokens * tokens * FOUNDATION_V0500_RELATIVE_FEATURE_DIM);
        for i in 0..tokens {
            for j in 0..tokens {
                let i_residue = i >= FOUNDATION_V0500_TASK_COUNT;
                let j_residue = j >= FOUNDATION_V0500_TASK_COUNT;
                let i_pos = if i_residue {
                    (i - FOUNDATION_V0500_TASK_COUNT) as f32
                } else {
                    0.0
                };
                let j_pos = if j_residue {
                    (j - FOUNDATION_V0500_TASK_COUNT) as f32
                } else {
                    0.0
                };
                let signed = if i_residue && j_residue {
                    (j_pos - i_pos) / denom
                } else {
                    0.0
                };
                let absolute = signed.abs();
                values.extend_from_slice(&[
                    signed,
                    absolute,
                    i_pos / denom,
                    j_pos / denom,
                    if i == j { 1.0 } else { 0.0 },
                    if i_residue && j_residue { 1.0 } else { 0.0 },
                ]);
            }
        }
        Tensor::from_vec(
            values,
            (1, tokens, tokens, FOUNDATION_V0500_RELATIVE_FEATURE_DIM),
            device,
        )
    }
}

mod rt_ms2_specialists {
    // ReDeeM v0.51 teacher-bridged specialist continuation of the v0.50 deep pair model.
    //
    // v0.50 established that the 320d chemistry/residue-pair backbone is mechanically
    // trainable, but its generic RT/MS2/mobility heads did not reach the promoted v0.35
    // and v0.38 DEV references. v0.51 therefore keeps the learned `student_v050.*`
    // backbone intact and adds task-specific `student_v051.*` modules:
    //
    // - learned 320 -> teacher-space projections for feature-level v0.35 distillation;
    // - a zero-initialized RT residual specialist;
    // - an identity-initialized v0.35-style MS2 context conditioner followed by a
    //   fragment-token Transformer with factorized presence/intensity outputs;
    // - a v0.38-style native ion-mobility residual head driven by peptide physics.
    //
    // The authoritative v0.35 model remains external and frozen. Exact Bruker
    // mobility->CCS conversion and the v0.35 CCS baseline are applied by the trainer,
    // so the mobility residual is zero at initialization and cannot silently redefine
    // the physics contract.

    use super::super::config::FoundationMs2OutputActivation;
    use super::super::featurize::FoundationBatch;
    use super::super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
    use super::super::model::{apply_ms2_output_activation, PrecursorContextBatch};
    use super::base_forward::{
        FoundationFragmentContextBatchV0270, FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270,
    };
    use super::deep_backbone::{
        FoundationMultimodalForwardOutputV0500, PeptideFoundationV0500Config,
        PeptideFoundationV0500Model,
    };
    use super::forward_specialists::PeptideFoundationMultimodalV0350Model;
    use super::scalar_physics::{
        FoundationScalarPhysicsBatchV0360, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
        FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360,
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
            if self.teacher_dim == 0
                || self.specialist_hidden == 0
                || self.specialist_bottleneck == 0
            {
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
            if fragment.cleavage_count > sequence - 1
                || fragment.output_cleavage_count != sequence - 1
            {
                candle_core::bail!(
                    "v0.51 fragment context width is incompatible with residue width"
                );
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
            let active_factorized =
                factorized.reshape((batch, fragment.cleavage_count, channels))?;
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
            let base_v0500 =
                PeptideFoundationV0500Model::new(config.base_v0500.clone(), vb.clone())?;
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
                ms2_context: Ms2ContextConditionerV0510::new(
                    &config,
                    specialist.pp("ms2_context"),
                )?,
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

        /// Compute the warm-started v0.50 mobility representation and the v0.51 residual
        /// in one pass. v0.52 uses this hook to add a pair-aware mobility refinement
        /// without duplicating the expensive deep backbone forward.
        pub fn mobility_components_t(
            &self,
            batch: &FoundationBatch,
            context: &PrecursorContextBatch,
            physics: &FoundationScalarPhysicsBatchV0360,
            train: bool,
        ) -> Result<(FoundationMultimodalForwardOutputV0500, Tensor)> {
            let base = self.base_v0500.forward_t(batch, context, train)?;
            let residual = self.mobility_residual.forward_t(&base, physics, train)?;
            Ok((base, residual))
        }

        /// Compute only the v0.51 mobility residual specialist on top of the v0.50 backbone.
        pub fn mobility_residual_t(
            &self,
            batch: &FoundationBatch,
            context: &PrecursorContextBatch,
            physics: &FoundationScalarPhysicsBatchV0360,
            train: bool,
        ) -> Result<Tensor> {
            let (_, residual) = self.mobility_components_t(batch, context, physics, train)?;
            Ok(residual)
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
            let (ms2_presence_logits, ms2_positive_intensity, ms2) = self
                .fragment_decoder
                .forward_t(&contextual_residues, context, fragment, &base.ms2, train)?;
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
}

mod final_checkpoint {
    // ReDeeM v0.52 mobility-aware pair representation on top of the completed v0.51 model.
    //
    // Source review before this iteration showed that v0.50 already injects precursor charge,
    // m/z, and a neutral-mass proxy into the mobility task token before all eight deep pair
    // interaction blocks. v0.52 therefore does not duplicate that path. Instead it addresses the
    // remaining representation gap exposed by v0.51: the mobility specialist did not consume the
    // learned residue-pair state directly.
    //
    // v0.52 keeps the selected `student_v050.*` backbone and successful `student_v051.*` RT/MS2
    // specialists, freezes the selected v0.51 mobility residual as a detached step-0 baseline, and
    // adds a `student_v052.*` mobility branch that explicitly integrates:
    //
    // - mobility-task -> residue and residue -> mobility pair states;
    // - residue-residue pair summaries;
    // - charge/mass/mz physics gates over the pair representation;
    // - a small mobility-specific residue Transformer refinement;
    // - TRAIN-only coarse physicochemical/conformation proxy prediction.
    //
    // The new native-mobility correction is zero initialized, so step 0 reproduces the selected
    // v0.51 mobility prediction exactly. Mobility gradients are forced through the v0.52 pair-aware
    // branch rather than through the old v0.51 mobility residual.

    use super::super::featurize::FoundationBatch;
    use super::super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
    use super::super::model::PrecursorContextBatch;
    use super::base_forward::FoundationFragmentContextBatchV0270;
    use super::deep_backbone::{
        FoundationMultimodalForwardOutputV0500, FOUNDATION_V0500_TASK_COUNT,
    };
    use super::rt_ms2_specialists::{
        FoundationMultimodalForwardOutputV0510, PeptideFoundationV0510Config,
        PeptideFoundationV0510Model,
    };
    use super::scalar_physics::{
        FoundationScalarPhysicsBatchV0360, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
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
            let pair_summary = pair_summary.broadcast_mul(
                &mask
                    .unsqueeze(2)?
                    .broadcast_as((batch, sequence, self.pair_dim))?,
            )?;

            // Charge/mass/mz context gates pair-state channels directly. This is distinct from the
            // existing v0.50 mobility task-token context and gives the mobility branch an explicit
            // physics-conditioned view of residue-pair compatibility.
            let pair_gate = ops::sigmoid(&self.physics_pair_gate.forward(&physics.ccs_physics)?)?;
            let pair_gate =
                pair_gate
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
            let base_v0510 =
                PeptideFoundationV0510Model::new(config.base_v0510.clone(), vb.clone())?;
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
}

// Narrow crate-private surface consumed by the stable production predictor.
// Historical implementation names stay quarantined below this line.
pub(crate) use ccs_checkpoint::{
    PeptideFoundationMultimodalV0380Config as CcsCheckpointConfig,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0380 as CCS_CHECKPOINT_ARCHITECTURE,
};
pub(crate) use final_checkpoint::{
    PeptideFoundationV0520Config as RtMs2CheckpointConfig,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520 as RT_MS2_CHECKPOINT_ARCHITECTURE,
};
pub(crate) use forward_specialists::FoundationFragmentContextBatchV0350 as FragmentContextBatch;
pub(crate) use scalar_physics::FoundationScalarPhysicsBatchV0360 as ScalarPhysicsBatch;

#[derive(Debug, Clone)]
pub(crate) struct RtMs2Prediction {
    pub rt: candle_core::Tensor,
    pub ms2: candle_core::Tensor,
}

pub(crate) struct RtMs2Model(final_checkpoint::PeptideFoundationV0520Model);

impl RtMs2Model {
    pub fn new(
        config: RtMs2CheckpointConfig,
        vb: candle_nn::VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        Ok(Self(final_checkpoint::PeptideFoundationV0520Model::new(
            config, vb,
        )?))
    }

    pub fn featurizer_config(&self) -> super::config::FoundationConfig {
        self.0.config().base_v0510.base_v0500.featurizer_config()
    }

    pub fn max_sequence_len(&self) -> usize {
        self.0.config().base_v0510.base_v0500.max_sequence_len
    }

    pub fn predict(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &super::model::PrecursorContextBatch,
        physics: &ScalarPhysicsBatch,
        fragment: &FragmentContextBatch,
    ) -> candle_core::Result<RtMs2Prediction> {
        let output = self
            .0
            .property_forward_t(batch, context, physics, fragment, false)?;
        Ok(RtMs2Prediction {
            rt: output.rt,
            ms2: output.ms2,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CcsPrediction {
    pub base_ccs_model: candle_core::Tensor,
    pub mobility_residual_native: candle_core::Tensor,
}

pub(crate) struct CcsModel(ccs_checkpoint::PeptideFoundationMultimodalV0380Model);

impl CcsModel {
    pub fn new(
        config: CcsCheckpointConfig,
        vb: candle_nn::VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        Ok(Self(
            ccs_checkpoint::PeptideFoundationMultimodalV0380Model::new(config, vb)?,
        ))
    }

    pub fn featurizer_config(&self) -> super::config::FoundationConfig {
        self.0.forward_config().clone()
    }

    pub fn max_sequence_len(&self) -> usize {
        self.0.forward_config().max_sequence_len
    }

    pub fn predict(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &super::model::PrecursorContextBatch,
        physics: &ScalarPhysicsBatch,
    ) -> candle_core::Result<CcsPrediction> {
        let output = self.0.mobility_v0380_t(batch, context, physics, false)?;
        Ok(CcsPrediction {
            base_ccs_model: output.base_ccs_model,
            mobility_residual_native: output.mobility_residual_native,
        })
    }
}
