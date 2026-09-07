//! End-to-end spectrum-peptide compatibility representation pretraining.
//!
//! This module is the representation-level successor to the rejected v0.15/v0.16
//! downstream interaction-head lanes. It deliberately trains an isolated copy of
//! the chemistry-aware bidirectional peptide encoder and observed-spectrum encoder
//! together with residue- and cleavage-level cross-modal interaction blocks.
//!
//! The accepted production forward/inverse branches are not mutated. A v0.17
//! experiment can warm-start the isolated compatibility encoders from a unified
//! checkpoint and then optimize every compatibility parameter end to end.

use super::config::FoundationConfig;
use super::diffusion::{FoundationDiffusionConfig, FoundationSpectrumEncoder};
use super::featurize::FoundationBatch;
use super::layers::{FoundationLayerNorm, MultiHeadCrossAttention, MultiHeadSelfAttention};
use super::model::{PeptideFoundationEncoder, PrecursorContextBatch};
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{DType, Device, Module, ModuleT, Result, Tensor};
use candle_nn::{self as nn, ops, Dropout, Embedding, Linear, VarBuilder, VarMap};
use std::path::Path;

/// Stable namespace used by the v0.17 compatibility representation.
pub const FOUNDATION_COMPATIBILITY_NAMESPACE_V0170: &str = "compatibility";

/// Warm-start accounting for an isolated compatibility model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoundationCompatibilityWarmStartReport {
    /// Chemistry-aware peptide-encoder tensors copied from `encoder.*`.
    pub peptide_encoder_loaded_variables: usize,
    /// Observed-spectrum encoder tensors copied from `spectrum_encoder.*`.
    pub spectrum_encoder_loaded_variables: usize,
    /// Fresh compatibility-specific variables intentionally not loaded.
    pub fresh_compatibility_variables: usize,
}

/// Encoded observed-spectrum context that can be reused for many candidate peptides.
#[derive(Debug, Clone)]
pub struct FoundationCompatibilitySpectrumContext {
    /// Peak/precursor memory `[spectra, memory_len, model_dim]`.
    pub memory: Tensor,
    /// Valid-memory mask `[spectra, memory_len]`.
    pub memory_mask: Tensor,
    /// Pooled spectrum/context representation `[spectra, model_dim]`.
    pub spectrum_embedding: Tensor,
}

/// Output from grouped spectrum-peptide compatibility scoring.
#[derive(Debug, Clone)]
pub struct FoundationCompatibilityOutput {
    /// Compatibility logits `[spectra, candidates_per_spectrum]`.
    pub scores: Tensor,
}

#[derive(Clone)]
struct CompatibilityInteractionBlock {
    cross_norm: FoundationLayerNorm,
    cross_attention: MultiHeadCrossAttention,
    self_norm: FoundationLayerNorm,
    self_attention: MultiHeadSelfAttention,
    ff_norm: FoundationLayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    dropout: Dropout,
}

impl CompatibilityInteractionBlock {
    fn new(
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout: f32,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            cross_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("cross_norm"))?,
            cross_attention: MultiHeadCrossAttention::new(
                model_dim,
                num_heads,
                vb.pp("cross_attention"),
            )?,
            self_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("self_norm"))?,
            self_attention: MultiHeadSelfAttention::new(
                model_dim,
                num_heads,
                vb.pp("self_attention"),
            )?,
            ff_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("ff_norm"))?,
            ff_in: nn::linear(model_dim, ff_dim, vb.pp("ff_in"))?,
            ff_out: nn::linear(ff_dim, model_dim, vb.pp("ff_out"))?,
            dropout: Dropout::new(dropout),
        })
    }

    fn forward_t(
        &self,
        hidden: &Tensor,
        token_mask: &Tensor,
        memory: &Tensor,
        memory_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let normalized = self.cross_norm.forward(hidden)?;
        let attended = self
            .cross_attention
            .forward(&normalized, memory, memory_mask)?;
        let hidden = (hidden + self.dropout.forward_t(&attended, train)?)?;

        let normalized = self.self_norm.forward(&hidden)?;
        let self_attended = self.self_attention.forward(&normalized, token_mask)?;
        let hidden = (hidden + self.dropout.forward_t(&self_attended, train)?)?;

        let normalized = self.ff_norm.forward(&hidden)?;
        let ff = self.ff_in.forward(&normalized)?.relu()?;
        let ff = self.ff_out.forward(&ff)?;
        let hidden = (hidden + self.dropout.forward_t(&ff, train)?)?;

        let (batch, tokens, model_dim) = hidden.dims3()?;
        hidden.broadcast_mul(
            &token_mask
                .unsqueeze(2)?
                .broadcast_as((batch, tokens, model_dim))?,
        )
    }
}

/// Trainable whole-peptidoform/observed-spectrum compatibility representation.
///
/// The model has an isolated copy of the accepted chemistry-aware peptide encoder
/// and observed-spectrum encoder. Both copies are trainable. Full bidirectional
/// residue states query observed peak memory, then adjacent residue states are
/// converted into cleavage representations that independently query the same
/// spectrum memory. A small scalar head reads the resulting cross-modal
/// representation, but the discriminative representation itself is trained end
/// to end rather than being frozen behind that head.
#[derive(Clone)]
pub struct FoundationSpectrumPeptideCompatibilityModel {
    peptide_encoder: PeptideFoundationEncoder,
    spectrum_encoder: FoundationSpectrumEncoder,
    context_instrument_embedding: Embedding,
    context_projection: Linear,
    context_norm: FoundationLayerNorm,
    spectrum_pool_projection: Linear,
    spectrum_pool_norm: FoundationLayerNorm,
    residue_interaction: Vec<CompatibilityInteractionBlock>,
    cleavage_projection: Linear,
    cleavage_interaction: Vec<CompatibilityInteractionBlock>,
    residue_output_norm: FoundationLayerNorm,
    cleavage_output_norm: FoundationLayerNorm,
    pair_norm: FoundationLayerNorm,
    pair_hidden: Linear,
    pair_dropout: Dropout,
    score_head: Linear,
    model_dim: usize,
}

impl FoundationSpectrumPeptideCompatibilityModel {
    /// Construct one isolated end-to-end compatibility model.
    ///
    /// The forward and inverse model widths must match because residue/cleavage
    /// queries attend directly into the observed-spectrum memory.
    pub fn new(
        forward_config: FoundationConfig,
        inverse_config: FoundationDiffusionConfig,
        residue_interaction_layers: usize,
        cleavage_interaction_layers: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        forward_config.validate().map_err(candle_core::Error::Msg)?;
        inverse_config.validate().map_err(candle_core::Error::Msg)?;
        if forward_config.model_dim != inverse_config.model_dim {
            candle_core::bail!(
                "compatibility model requires matching peptide/spectrum widths; forward={} inverse={}",
                forward_config.model_dim,
                inverse_config.model_dim
            );
        }
        if forward_config.num_attention_heads == 0
            || forward_config.model_dim % forward_config.num_attention_heads != 0
        {
            candle_core::bail!("compatibility model requires valid peptide attention heads");
        }
        if residue_interaction_layers == 0 || cleavage_interaction_layers == 0 {
            candle_core::bail!(
                "compatibility model requires at least one residue and one cleavage interaction layer"
            );
        }

        let model_dim = forward_config.model_dim;
        let root = vb.pp(FOUNDATION_COMPATIBILITY_NAMESPACE_V0170);
        let peptide_encoder =
            PeptideFoundationEncoder::new(forward_config.clone(), root.pp("peptide_encoder"))?;
        let spectrum_encoder =
            FoundationSpectrumEncoder::new(&inverse_config, root.pp("spectrum_encoder"))?;

        let context_embedding_dim = (model_dim / 4).max(8);
        let context_instrument_embedding = nn::embedding(
            forward_config.instrument_vocab_size,
            context_embedding_dim,
            root.pp("context.instrument_embedding"),
        )?;
        let context_projection = nn::linear(
            context_embedding_dim + 7,
            model_dim,
            root.pp("context.projection"),
        )?;
        let context_norm = FoundationLayerNorm::new(model_dim, 1e-5, root.pp("context.norm"))?;
        let spectrum_pool_projection = nn::linear(
            model_dim * 2,
            model_dim,
            root.pp("spectrum_pool.projection"),
        )?;
        let spectrum_pool_norm =
            FoundationLayerNorm::new(model_dim, 1e-5, root.pp("spectrum_pool.norm"))?;

        let mut residue_interaction = Vec::with_capacity(residue_interaction_layers);
        for index in 0..residue_interaction_layers {
            residue_interaction.push(CompatibilityInteractionBlock::new(
                model_dim,
                forward_config.num_attention_heads,
                forward_config.transformer_ff_dim,
                forward_config.dropout,
                root.pp(format!("residue_interaction.{index}")),
            )?);
        }

        let cleavage_projection =
            nn::linear(model_dim * 2, model_dim, root.pp("cleavage.projection"))?;
        let mut cleavage_interaction = Vec::with_capacity(cleavage_interaction_layers);
        for index in 0..cleavage_interaction_layers {
            cleavage_interaction.push(CompatibilityInteractionBlock::new(
                model_dim,
                forward_config.num_attention_heads,
                forward_config.transformer_ff_dim,
                forward_config.dropout,
                root.pp(format!("cleavage.interaction.{index}")),
            )?);
        }

        Ok(Self {
            peptide_encoder,
            spectrum_encoder,
            context_instrument_embedding,
            context_projection,
            context_norm,
            spectrum_pool_projection,
            spectrum_pool_norm,
            residue_interaction,
            cleavage_projection,
            cleavage_interaction,
            residue_output_norm: FoundationLayerNorm::new(
                model_dim,
                1e-5,
                root.pp("output.residue_norm"),
            )?,
            cleavage_output_norm: FoundationLayerNorm::new(
                model_dim,
                1e-5,
                root.pp("output.cleavage_norm"),
            )?,
            pair_norm: FoundationLayerNorm::new(model_dim * 5, 1e-5, root.pp("output.pair_norm"))?,
            pair_hidden: nn::linear(model_dim * 5, model_dim * 2, root.pp("output.pair_hidden"))?,
            pair_dropout: Dropout::new(forward_config.dropout),
            score_head: nn::linear(model_dim * 2, 1, root.pp("output.score"))?,
            model_dim,
        })
    }

    /// Encode observed spectra plus precursor/acquisition context once.
    pub fn encode_spectrum_t(
        &self,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationCompatibilitySpectrumContext> {
        let (batch, _, _) = spectrum.peak_features.dims3()?;
        if precursor.charge.dims1()? != batch
            || precursor.precursor_mz.dims1()? != batch
            || precursor.nce.dims1()? != batch
            || precursor.instrument_ids.dims1()? != batch
        {
            candle_core::bail!("compatibility spectrum/precursor batch mismatch");
        }

        let encoded = self.spectrum_encoder.forward_t(spectrum, train)?;
        let instrument = self
            .context_instrument_embedding
            .forward(&precursor.instrument_ids)?;
        let instrument_dim = instrument.dim(1)?;
        let instrument_mask = precursor
            .instrument_present
            .unsqueeze(1)?
            .broadcast_as((batch, instrument_dim))?;
        let instrument = instrument.broadcast_mul(&instrument_mask)?;

        let charge_present = &precursor.charge_present;
        let mz_present = &precursor.precursor_mz_present;
        let nce_present = &precursor.nce_present;
        let physical_present = charge_present.broadcast_mul(mz_present)?;
        let scaled_charge = precursor
            .charge
            .broadcast_mul(charge_present)?
            .affine(1.0 / 6.0, 0.0)?
            .unsqueeze(1)?;
        let scaled_mz = precursor
            .precursor_mz
            .broadcast_mul(mz_present)?
            .affine(1.0 / 2_000.0, 0.0)?
            .unsqueeze(1)?;
        let neutral_mass_proxy = precursor
            .precursor_mz
            .broadcast_mul(&precursor.charge)?
            .broadcast_mul(&physical_present)?
            .affine(1.0 / 3_000.0, 0.0)?
            .unsqueeze(1)?;
        let scaled_nce = precursor
            .nce
            .broadcast_mul(nce_present)?
            .affine(1.0 / 100.0, 0.0)?
            .unsqueeze(1)?;
        let scalar_context = Tensor::cat(
            &[
                &scaled_charge,
                &scaled_mz,
                &neutral_mass_proxy,
                &scaled_nce,
                &charge_present.unsqueeze(1)?,
                &mz_present.unsqueeze(1)?,
                &nce_present.unsqueeze(1)?,
            ],
            1,
        )?;
        let context_features = Tensor::cat(&[&instrument, &scalar_context], 1)?;
        let precursor_token = self
            .context_norm
            .forward(&self.context_projection.forward(&context_features)?)?;

        let memory = Tensor::cat(
            &[&precursor_token.unsqueeze(1)?, &encoded.peak_embeddings],
            1,
        )?;
        let memory_mask = Tensor::cat(
            &[
                &Tensor::ones((batch, 1), DType::F32, spectrum.peak_mask.device())?,
                &spectrum.peak_mask,
            ],
            1,
        )?;
        let spectrum_embedding =
            self.spectrum_pool_norm
                .forward(&self.spectrum_pool_projection.forward(&Tensor::cat(
                    &[&encoded.spectrum_embedding, &precursor_token],
                    1,
                )?)?)?;

        Ok(FoundationCompatibilitySpectrumContext {
            memory,
            memory_mask,
            spectrum_embedding,
        })
    }

    /// Score grouped candidates against reusable spectrum contexts.
    ///
    /// `peptides` contains `spectra * candidates_per_spectrum` rows ordered in
    /// contiguous spectrum-major groups.
    pub fn score_candidates_t(
        &self,
        peptides: &FoundationBatch,
        spectrum: &FoundationCompatibilitySpectrumContext,
        candidates_per_spectrum: usize,
        train: bool,
    ) -> Result<FoundationCompatibilityOutput> {
        if candidates_per_spectrum == 0 {
            candle_core::bail!(
                "compatibility scoring requires at least one candidate per spectrum"
            );
        }
        let (spectrum_batch, memory_len, memory_dim) = spectrum.memory.dims3()?;
        if memory_dim != self.model_dim {
            candle_core::bail!(
                "compatibility spectrum memory dim {memory_dim} != configured {}",
                self.model_dim
            );
        }
        let pair_batch = peptides.residue_mask.dim(0)?;
        let expected_pairs = spectrum_batch * candidates_per_spectrum;
        if pair_batch != expected_pairs {
            candle_core::bail!(
                "compatibility grouped batch mismatch: peptide_pairs={pair_batch} expected={expected_pairs}"
            );
        }

        let memory = spectrum
            .memory
            .unsqueeze(1)?
            .broadcast_as((
                spectrum_batch,
                candidates_per_spectrum,
                memory_len,
                self.model_dim,
            ))?
            .contiguous()?
            .reshape((pair_batch, memory_len, self.model_dim))?;
        let memory_mask = spectrum
            .memory_mask
            .unsqueeze(1)?
            .broadcast_as((spectrum_batch, candidates_per_spectrum, memory_len))?
            .contiguous()?
            .reshape((pair_batch, memory_len))?;
        let spectrum_embedding = spectrum
            .spectrum_embedding
            .unsqueeze(1)?
            .broadcast_as((spectrum_batch, candidates_per_spectrum, self.model_dim))?
            .contiguous()?
            .reshape((pair_batch, self.model_dim))?;

        let peptide = self.peptide_encoder.forward_t(peptides, train)?;
        let mut residues = peptide.residue_embeddings;
        for block in &self.residue_interaction {
            residues = block.forward_t(
                &residues,
                &peptide.residue_mask,
                &memory,
                &memory_mask,
                train,
            )?;
        }
        residues = self.residue_output_norm.forward(&residues)?;
        let residue_pooled = masked_mean(&residues, &peptide.residue_mask)?;

        let sequence_len = peptide.residue_mask.dim(1)?;
        if sequence_len < 2 {
            candle_core::bail!("compatibility peptide sequence width must be at least two");
        }
        let left = residues.narrow(1, 0, sequence_len - 1)?;
        let right = residues.narrow(1, 1, sequence_len - 1)?;
        let cleavage_mask = peptide
            .residue_mask
            .narrow(1, 0, sequence_len - 1)?
            .broadcast_mul(&peptide.residue_mask.narrow(1, 1, sequence_len - 1)?)?;
        // `left` and `right` are narrow views into the residue tensor. On CUDA,
        // Candle may preserve a strided layout through concatenation along the
        // feature axis, while `Linear`/matmul requires contiguous storage for
        // this 3-D batched projection. Materialize the exact same concatenated
        // cleavage features before the 2*model_dim -> model_dim projection.
        let cleavage_input = Tensor::cat(&[&left, &right], 2)?.contiguous()?;
        let mut cleavage = self.cleavage_projection.forward(&cleavage_input)?;
        cleavage = cleavage.broadcast_mul(&cleavage_mask.unsqueeze(2)?.broadcast_as((
            pair_batch,
            sequence_len - 1,
            self.model_dim,
        ))?)?;
        for block in &self.cleavage_interaction {
            cleavage = block.forward_t(&cleavage, &cleavage_mask, &memory, &memory_mask, train)?;
        }
        cleavage = self.cleavage_output_norm.forward(&cleavage)?;
        let cleavage_pooled = masked_mean(&cleavage, &cleavage_mask)?;

        let residue_spectrum_product = residue_pooled.broadcast_mul(&spectrum_embedding)?;
        let cleavage_spectrum_product = cleavage_pooled.broadcast_mul(&spectrum_embedding)?;
        let pair_features = Tensor::cat(
            &[
                &residue_pooled,
                &cleavage_pooled,
                &spectrum_embedding,
                &residue_spectrum_product,
                &cleavage_spectrum_product,
            ],
            1,
        )?;
        let pair_features = self.pair_norm.forward(&pair_features)?;
        let hidden = self.pair_hidden.forward(&pair_features)?.relu()?;
        let hidden = self.pair_dropout.forward_t(&hidden, train)?;
        let scores = self
            .score_head
            .forward(&hidden)?
            .squeeze(1)?
            .reshape((spectrum_batch, candidates_per_spectrum))?;

        Ok(FoundationCompatibilityOutput { scores })
    }

    /// Encode spectra and score aligned spectrum-major candidate groups.
    pub fn forward_grouped_t(
        &self,
        peptides: &FoundationBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        candidates_per_spectrum: usize,
        train: bool,
    ) -> Result<FoundationCompatibilityOutput> {
        let context = self.encode_spectrum_t(spectrum, precursor, train)?;
        self.score_candidates_t(peptides, &context, candidates_per_spectrum, train)
    }

    /// Shared compatibility width.
    pub fn model_dim(&self) -> usize {
        self.model_dim
    }
}

fn masked_mean(hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, tokens, model_dim) = hidden.dims3()?;
    let (mask_batch, mask_tokens) = mask.dims2()?;
    if batch != mask_batch || tokens != mask_tokens {
        candle_core::bail!(
            "compatibility masked mean mismatch: hidden [{batch},{tokens},{model_dim}] mask [{mask_batch},{mask_tokens}]"
        );
    }
    let masked = hidden.broadcast_mul(
        &mask
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, model_dim))?,
    )?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    masked.sum(1)?.broadcast_div(&denominator)
}

/// Positive-first grouped listwise compatibility loss.
///
/// Every row in `scores` represents one observed spectrum. Column zero is the
/// matched TRAIN peptidoform; remaining columns are precursor-compatible hard
/// negatives. The loss is the mean negative log probability assigned to the
/// matched candidate under a row-wise softmax.
pub fn foundation_compatibility_listwise_loss(scores: &Tensor) -> Result<Tensor> {
    let (groups, candidates) = scores.dims2()?;
    if groups == 0 || candidates < 2 {
        candle_core::bail!(
            "compatibility listwise loss requires non-empty groups with at least one negative"
        );
    }
    let log_probabilities = ops::log_softmax(scores, 1)?;
    log_probabilities
        .narrow(1, 0, 1)?
        .mean_all()?
        .affine(-1.0, 0.0)
}

/// Warm-start isolated compatibility encoders from an accepted unified checkpoint.
///
/// Only two parent namespaces are copied:
///
/// - `encoder.*` -> `compatibility.peptide_encoder.*`
/// - `spectrum_encoder.*` -> `compatibility.spectrum_encoder.*`
///
/// Every cross-modal/context/output parameter remains fresh. All variables stay
/// trainable after loading; this function performs no freezing.
pub fn load_compatibility_from_unified_checkpoint(
    varmap: &VarMap,
    checkpoint: &Path,
    device: &Device,
) -> Result<FoundationCompatibilityWarmStartReport> {
    let tensors = candle_core::safetensors::load(checkpoint, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("compatibility VarMap lock poisoned".into()))?;

    let mut peptide_loaded = 0usize;
    let mut spectrum_loaded = 0usize;
    let mut fresh = 0usize;
    let mut missing = Vec::<String>::new();

    for (name, variable) in data.iter() {
        let parent_name = if let Some(suffix) = name.strip_prefix("compatibility.peptide_encoder.")
        {
            peptide_loaded += 1;
            Some(format!("encoder.{suffix}"))
        } else if let Some(suffix) = name.strip_prefix("compatibility.spectrum_encoder.") {
            spectrum_loaded += 1;
            Some(format!("spectrum_encoder.{suffix}"))
        } else {
            fresh += 1;
            None
        };

        let Some(parent_name) = parent_name else {
            continue;
        };
        let Some(value) = tensors.get(&parent_name) else {
            missing.push(format!("{name}<-{parent_name}"));
            continue;
        };
        if variable.as_tensor().dims() != value.dims() {
            candle_core::bail!(
                "compatibility warm-start shape mismatch for '{name}' from '{parent_name}': current {:?}, checkpoint {:?}",
                variable.as_tensor().dims(),
                value.dims()
            );
        }
        variable.set(value)?;
    }
    drop(data);

    if !missing.is_empty() {
        candle_core::bail!(
            "compatibility warm start is missing required parent variables: {}",
            missing.join(", ")
        );
    }
    if peptide_loaded == 0 || spectrum_loaded == 0 {
        candle_core::bail!(
            "compatibility warm start loaded no peptide or spectrum encoder variables"
        );
    }

    Ok(FoundationCompatibilityWarmStartReport {
        peptide_encoder_loaded_variables: peptide_loaded,
        spectrum_encoder_loaded_variables: spectrum_loaded,
        fresh_compatibility_variables: fresh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationSpectrum, FoundationSpectrumCollator, FoundationSpectrumConfig,
        PeptideGraphFeaturizer, PeptidoformInput,
    };
    use candle_core::{DType, Device};

    fn forward_config() -> FoundationConfig {
        FoundationConfig {
            max_sequence_len: 6,
            max_atoms_per_residue: 20,
            graph_hidden_dim: 8,
            graph_layers: 1,
            model_dim: 8,
            num_attention_heads: 2,
            transformer_ff_dim: 16,
            transformer_layers: 1,
            dropout: 0.0,
            contrastive_dim: 8,
            instrument_vocab_size: 4,
            ..FoundationConfig::default()
        }
    }

    fn inverse_config() -> FoundationDiffusionConfig {
        FoundationDiffusionConfig {
            max_tokens: 8,
            model_dim: 8,
            num_attention_heads: 2,
            feed_forward_dim: 16,
            spectrum_layers: 1,
            decoder_layers: 1,
            dropout: 0.0,
            spectrum: FoundationSpectrumConfig {
                max_peaks: 4,
                ..FoundationSpectrumConfig::default()
            },
            ..FoundationDiffusionConfig::default()
        }
    }

    #[test]
    fn listwise_loss_rewards_positive_column() -> Result<()> {
        let device = Device::Cpu;
        let good = Tensor::new(&[[4.0f32, 0.0, -1.0]], &device)?;
        let bad = Tensor::new(&[[-1.0f32, 0.0, 4.0]], &device)?;
        let good_loss = foundation_compatibility_listwise_loss(&good)?.to_scalar::<f32>()?;
        let bad_loss = foundation_compatibility_listwise_loss(&bad)?.to_scalar::<f32>()?;
        assert!(good_loss < bad_loss);
        Ok(())
    }

    #[test]
    fn grouped_compatibility_backprop_reaches_both_encoders() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = FoundationSpectrumPeptideCompatibilityModel::new(
            forward_config(),
            inverse_config(),
            1,
            1,
            vb,
        )?;
        let featurizer = PeptideGraphFeaturizer::new(forward_config())?;
        let peptides = featurizer.featurize(
            &[
                PeptidoformInput::unmodified("ACDE"),
                PeptidoformInput::unmodified("ACDF"),
            ],
            &device,
        )?;
        let spectrum_collator = FoundationSpectrumCollator::new(inverse_config().spectrum)?;
        let spectra = spectrum_collator.collate(
            &[FoundationSpectrum::from_pairs([
                (150.0, 10.0),
                (250.0, 8.0),
                (350.0, 4.0),
            ])],
            &device,
        )?;
        let precursor = PrecursorContextBatch::unknown(1, &device)?;
        let output = model.forward_grouped_t(&peptides, &spectra, &precursor, 2, true)?;
        assert_eq!(output.scores.dims2()?, (1, 2));
        let loss = foundation_compatibility_listwise_loss(&output.scores)?;
        let gradients = loss.backward()?;

        let data = varmap.data().lock().unwrap();
        let peptide = data
            .get("compatibility.peptide_encoder.atom_input.weight")
            .expect("peptide encoder weight");
        let spectrum = data
            .get("compatibility.spectrum_encoder.input_projection.weight")
            .expect("spectrum encoder weight");
        let peptide_norm = gradients
            .get(peptide)
            .expect("peptide gradient")
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?;
        let spectrum_norm = gradients
            .get(spectrum)
            .expect("spectrum gradient")
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(peptide_norm > 0.0);
        assert!(spectrum_norm > 0.0);
        Ok(())
    }
}
