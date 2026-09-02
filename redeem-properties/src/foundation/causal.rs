//! Spectrum-conditioned causal next-token peptide scoring/training.
//!
//! This lane deliberately reuses the shape-compatible v0.12 diffusion spectrum
//! encoder and decoder parameters while changing the training distribution to a
//! true teacher-forced left-to-right objective. START is represented by a
//! dedicated learned embedding and is not inserted into the residue/PTM
//! vocabulary.

use super::diffusion::{
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FoundationSpectrumEncoder,
    SpectrumConditionedDiffusionBlock, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_VOCAB_SIZE,
};
use super::featurize::PeptidoformInput;
use super::layers::FoundationLayerNorm;
use super::model::PrecursorContextBatch;
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, loss, Embedding, Linear, VarBuilder, VarMap};
use std::collections::HashSet;
use std::path::Path;

/// Validated v0.12.3 coefficient for combining fragment evidence with causal
/// sequence likelihood. The coefficient was frozen before the 512-record
/// confirmation slice and recovered every literal candidate-pool oracle there.
pub const FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123: f64 = 0.1;

/// Stable identifier for the validated v0.12.3 hybrid ranking policy.
pub const FOUNDATION_CAUSAL_RERANK_POLICY_V0123: &str = "fragment_plus_0.1_ar_total_v1";

/// Combine fragment evidence with causal total log-likelihood.
///
/// The total autoregressive log-likelihood is deliberately retained rather
/// than length-normalized here. At the small validated coefficient it acts as
/// a sequence-plausibility regularizer without overwhelming fragment evidence.
pub fn foundation_fragment_causal_rerank_score(
    fragment_score: f64,
    ar_total_log_probability: f64,
    weight: f64,
) -> f64 {
    fragment_score + weight * ar_total_log_probability
}

/// Model inputs for causal decoding.
///
/// Targets are intentionally absent from this structure. Candidate/target tokens
/// can therefore never be consumed by the model except through the explicitly
/// shifted prefix token tensor.
#[derive(Debug, Clone)]
pub struct FoundationCausalInputBatch {
    /// Shifted teacher-forcing token ids `[batch, max_tokens]`.
    /// Position zero is a PAD placeholder that is replaced by the learned START
    /// embedding inside the model. Position `i>0` contains target token `i-1`.
    pub input_tokens: Tensor,
    /// Active prediction-position mask `[batch, max_tokens]`.
    pub token_mask: Tensor,
}

/// Teacher-forced batch for causal next-token training.
#[derive(Debug, Clone)]
pub struct FoundationCausalBatch {
    /// Model-visible shifted prefix inputs.
    pub input: FoundationCausalInputBatch,
    /// Clean target rows `[batch, max_tokens]`, ending with EOS before padding.
    pub target_tokens: Tensor,
    /// Flattened active prediction positions.
    pub active_indices: Tensor,
    /// Target classes aligned with `active_indices`.
    pub target_classes: Tensor,
}

/// Deterministic causal teacher-forcing collator.
#[derive(Debug, Clone)]
pub struct FoundationCausalCollator {
    config: FoundationDiffusionConfig,
    vocabulary: FoundationDiffusionVocabulary,
}

impl FoundationCausalCollator {
    /// Construct a collator using the same token/config contract as diffusion.
    pub fn new(config: FoundationDiffusionConfig) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        Ok(Self {
            config,
            vocabulary: FoundationDiffusionVocabulary,
        })
    }

    /// Encode peptidoforms and create `START,c1,... -> c1,c2,...,EOS` batches.
    pub fn collate(
        &self,
        peptides: &[PeptidoformInput],
        device: &Device,
    ) -> Result<FoundationCausalBatch> {
        if peptides.is_empty() {
            candle_core::bail!("causal collation requires at least one peptide");
        }
        let rows: Vec<Vec<u32>> = peptides
            .iter()
            .map(|peptide| {
                self.vocabulary
                    .encode(peptide, self.config.max_tokens)
                    .map_err(candle_core::Error::Msg)
            })
            .collect::<Result<_>>()?;
        self.collate_token_rows(&rows, device)
    }

    /// Create a causal batch from already encoded clean candidate rows.
    ///
    /// This is the candidate-scoring path. It verifies that every active row is
    /// a clean sequence ending in EOS, shifts the candidate by one position, and
    /// keeps the unshifted targets outside the model input structure.
    pub fn collate_token_rows(
        &self,
        target_rows: &[Vec<u32>],
        device: &Device,
    ) -> Result<FoundationCausalBatch> {
        if target_rows.is_empty() {
            candle_core::bail!("causal token-row collation requires at least one row");
        }
        let batch = target_rows.len();
        let width = self.config.max_tokens;
        let mut shifted = vec![FOUNDATION_DIFFUSION_PAD; batch * width];
        let mut targets = vec![FOUNDATION_DIFFUSION_PAD; batch * width];
        let mut mask = vec![0.0f32; batch * width];
        let mut active_indices = Vec::<u32>::new();
        let mut target_classes = Vec::<u32>::new();

        for (row_index, row) in target_rows.iter().enumerate() {
            if row.len() != width {
                candle_core::bail!(
                    "causal target row width {} does not match configured {width}",
                    row.len()
                );
            }
            let active_length = row
                .iter()
                .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
                .unwrap_or(width);
            if active_length == 0 {
                candle_core::bail!("causal target row has no active tokens");
            }
            if row[active_length - 1] != FOUNDATION_DIFFUSION_EOS {
                candle_core::bail!("causal target active prefix must terminate with EOS");
            }
            for (position, &token) in row.iter().enumerate() {
                if position < active_length {
                    if token == FOUNDATION_DIFFUSION_PAD || token == FOUNDATION_DIFFUSION_MASK {
                        candle_core::bail!(
                            "causal clean target position {position} contains PAD/MASK"
                        );
                    }
                    if token as usize >= FOUNDATION_DIFFUSION_VOCAB_SIZE {
                        candle_core::bail!("causal clean target token {token} exceeds vocabulary");
                    }
                    if token == FOUNDATION_DIFFUSION_EOS && position + 1 != active_length {
                        candle_core::bail!(
                            "causal clean target may contain EOS only at the final active position"
                        );
                    }
                    let flat = row_index * width + position;
                    targets[flat] = token;
                    mask[flat] = 1.0;
                    active_indices.push(flat as u32);
                    target_classes.push(token);
                    if position > 0 {
                        shifted[flat] = row[position - 1];
                    }
                } else if token != FOUNDATION_DIFFUSION_PAD {
                    candle_core::bail!("causal padded target suffix must contain only PAD");
                }
            }
        }

        let active_count = active_indices.len();
        Ok(FoundationCausalBatch {
            input: FoundationCausalInputBatch {
                input_tokens: Tensor::from_vec(shifted, (batch, width), device)?
                    .to_dtype(DType::U32)?,
                token_mask: Tensor::from_vec(mask, (batch, width), device)?,
            },
            target_tokens: Tensor::from_vec(targets, (batch, width), device)?
                .to_dtype(DType::U32)?,
            active_indices: Tensor::from_vec(active_indices, active_count, device)?
                .to_dtype(DType::U32)?,
            target_classes: Tensor::from_vec(target_classes, active_count, device)?
                .to_dtype(DType::U32)?,
        })
    }
}

/// Output from a teacher-forced causal next-token prediction.
#[derive(Debug, Clone)]
pub struct FoundationCausalOutput {
    /// Next-token logits `[batch, max_tokens, vocabulary]`.
    pub token_logits: Tensor,
    /// Cross-attention memory, including the prepended precursor token.
    pub spectrum_memory: Tensor,
    /// Mask-aware pooled observed-spectrum embedding.
    pub spectrum_embedding: Tensor,
}

/// Spectrum-conditioned causal peptide decoder.
///
/// Shape-compatible parameters deliberately use the exact historical diffusion
/// names so a v0.12 diffusion checkpoint can warm-start them. The only new
/// trainable parameter is `decoder.causal_start_embedding.weight`.
#[derive(Clone)]
pub struct PeptideSpectrumCausalModel {
    config: FoundationDiffusionConfig,
    spectrum_encoder: FoundationSpectrumEncoder,
    token_embedding: Embedding,
    causal_start_embedding: Embedding,
    position_embedding: Embedding,
    precursor_projection: Linear,
    layers: Vec<SpectrumConditionedDiffusionBlock>,
    output_norm: FoundationLayerNorm,
    token_head: Linear,
}

impl PeptideSpectrumCausalModel {
    /// Construct the causal model with checkpoint-compatible shared namespaces.
    pub fn new(config: FoundationDiffusionConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let spectrum_encoder = FoundationSpectrumEncoder::new(&config, vb.pp("spectrum_encoder"))?;
        let token_embedding = nn::embedding(
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            config.model_dim,
            vb.pp("decoder.token_embedding"),
        )?;
        let causal_start_weight = vb.pp("decoder.causal_start_embedding").get_with_hints(
            (1, config.model_dim),
            "weight",
            nn::Init::Const(0.0),
        )?;
        let causal_start_embedding = Embedding::new(causal_start_weight, config.model_dim);
        let position_embedding = nn::embedding(
            config.max_tokens,
            config.model_dim,
            vb.pp("decoder.position_embedding"),
        )?;
        let precursor_projection = nn::linear(6, config.model_dim, vb.pp("decoder.precursor"))?;
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for index in 0..config.decoder_layers {
            layers.push(SpectrumConditionedDiffusionBlock::new(
                &config,
                vb.pp(format!("decoder.layers.{index}")),
            )?);
        }
        let output_norm =
            FoundationLayerNorm::new(config.model_dim, 1e-5, vb.pp("decoder.output_norm"))?;
        let token_head = nn::linear(
            config.model_dim,
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            vb.pp("decoder.token_head"),
        )?;
        Ok(Self {
            config,
            spectrum_encoder,
            token_embedding,
            causal_start_embedding,
            position_embedding,
            precursor_projection,
            layers,
            output_norm,
            token_head,
        })
    }

    /// Predict every next token in one shifted, teacher-forced causal pass.
    pub fn forward_t(
        &self,
        input: &FoundationCausalInputBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationCausalOutput> {
        let (batch, token_len) = input.input_tokens.dims2()?;
        let (spectrum_batch, _, _) = spectrum.peak_features.dims3()?;
        if batch != spectrum_batch {
            candle_core::bail!(
                "causal/spectrum batch mismatch: causal {batch}, spectrum {spectrum_batch}"
            );
        }
        if token_len != self.config.max_tokens {
            candle_core::bail!(
                "causal token width {token_len} does not match configured {}",
                self.config.max_tokens
            );
        }

        let spectrum_encoding = self.spectrum_encoder.forward_t(spectrum, train)?;

        // START is not a vocabulary id. The first position uses the dedicated
        // embedding directly; positions 1.. use shifted clean prefix tokens.
        let start_ids = Tensor::zeros(batch, DType::U32, input.input_tokens.device())?;
        let start_embedding = self
            .causal_start_embedding
            .forward(&start_ids)?
            .unsqueeze(1)?;
        let shifted_suffix = input.input_tokens.narrow(1, 1, token_len - 1)?;
        let shifted_embedding = self.token_embedding.forward(&shifted_suffix)?;
        let token_embedding = Tensor::cat(&[&start_embedding, &shifted_embedding], 1)?;

        let positions: Vec<u32> = (0..token_len as u32).collect();
        let position_ids = Tensor::from_vec(positions, token_len, input.input_tokens.device())?
            .to_dtype(DType::U32)?;
        let position_embedding = self
            .position_embedding
            .forward(&position_ids)?
            .unsqueeze(0)?
            .broadcast_as((batch, token_len, self.config.model_dim))?;

        let scaled_precursor_mz = precursor
            .precursor_mz
            .affine(1.0 / 2_000.0, 0.0)?
            .unsqueeze(1)?;
        let scaled_charge = precursor.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let charge_squared = precursor
            .charge
            .broadcast_mul(&precursor.charge)?
            .affine(1.0 / 36.0, 0.0)?
            .unsqueeze(1)?;
        let physical_present = precursor
            .precursor_mz_present
            .broadcast_mul(&precursor.charge_present)?;
        let neutral_mass_proxy = precursor
            .precursor_mz
            .broadcast_mul(&precursor.charge)?
            .broadcast_mul(&physical_present)?
            .affine(1.0 / 6_000.0, 0.0)?
            .unsqueeze(1)?;
        let precursor_mz_present = precursor.precursor_mz_present.unsqueeze(1)?;
        let charge_present = precursor.charge_present.unsqueeze(1)?;
        let precursor_features = Tensor::cat(
            &[
                &scaled_precursor_mz,
                &scaled_charge,
                &neutral_mass_proxy,
                &charge_squared,
                &precursor_mz_present,
                &charge_present,
            ],
            1,
        )?;
        let precursor_summary = self.precursor_projection.forward(&precursor_features)?;

        let precursor_memory = precursor_summary.unsqueeze(1)?;
        let spectrum_memory =
            Tensor::cat(&[&precursor_memory, &spectrum_encoding.peak_embeddings], 1)?;
        let precursor_memory_mask =
            Tensor::ones((batch, 1), DType::F32, input.input_tokens.device())?;
        let spectrum_memory_mask = Tensor::cat(&[&precursor_memory_mask, &spectrum.peak_mask], 1)?;

        let precursor_embedding = precursor_summary.unsqueeze(1)?.broadcast_as((
            batch,
            token_len,
            self.config.model_dim,
        ))?;
        let mut hidden = ((token_embedding + position_embedding)? + precursor_embedding)?;
        let token_mask = input.token_mask.unsqueeze(2)?.broadcast_as((
            batch,
            token_len,
            self.config.model_dim,
        ))?;
        hidden = hidden.broadcast_mul(&token_mask)?;
        for layer in &self.layers {
            hidden = layer.forward_t_causal(
                &hidden,
                &input.token_mask,
                &spectrum_memory,
                &spectrum_memory_mask,
                train,
            )?;
        }
        hidden = self
            .output_norm
            .forward(&hidden)?
            .broadcast_mul(&token_mask)?;
        let token_logits = self.token_head.forward(&hidden)?;
        Ok(FoundationCausalOutput {
            token_logits,
            spectrum_memory,
            spectrum_embedding: spectrum_encoding.spectrum_embedding,
        })
    }

    /// Shared architecture configuration inherited from the diffusion checkpoint.
    pub fn config(&self) -> &FoundationDiffusionConfig {
        &self.config
    }
}

/// Cross-entropy next-token objective over all active targets, including EOS.
pub fn foundation_causal_next_token_loss(
    output: &FoundationCausalOutput,
    batch: &FoundationCausalBatch,
) -> Result<Tensor> {
    let (b, l, classes) = output.token_logits.dims3()?;
    if classes != FOUNDATION_DIFFUSION_VOCAB_SIZE {
        candle_core::bail!(
            "causal logits expose {classes} classes, expected {}",
            FOUNDATION_DIFFUSION_VOCAB_SIZE
        );
    }
    let flat_logits = output.token_logits.reshape((b * l, classes))?;
    let selected_logits = flat_logits.index_select(&batch.active_indices, 0)?;
    loss::cross_entropy(&selected_logits, &batch.target_classes)
}

/// Warm-start report for loading an old diffusion checkpoint into the causal lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoundationCausalWarmStartReport {
    /// Number of shared current causal variables loaded from the checkpoint.
    pub loaded_variables: usize,
    /// Number of new causal-only variables left at fresh initialization.
    pub causal_only_variables: usize,
    /// Number of checkpoint tensors intentionally ignored because causal mode
    /// does not instantiate the diffusion timestep/length/alignment parameters.
    pub ignored_checkpoint_variables: usize,
}

/// Load every shape-compatible causal variable from a historical diffusion model.
///
/// All causal variables except `decoder.causal_start_embedding.*` are required to
/// be present. Extra diffusion-only checkpoint tensors are ignored. This is the
/// compatibility boundary that lets v0.12.2 reuse v0.12.0 without modifying the
/// historical diffusion model or its checkpoint loader.
pub fn load_causal_from_diffusion_checkpoint(
    varmap: &VarMap,
    path: &Path,
    device: &Device,
) -> Result<FoundationCausalWarmStartReport> {
    let checkpoint = candle_core::safetensors::load(path, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("causal VarMap lock poisoned".into()))?;
    let current_names: HashSet<String> = data.keys().cloned().collect();
    let mut loaded_variables = 0usize;
    let mut causal_only_variables = 0usize;
    let mut required_missing = Vec::<String>::new();

    for (name, variable) in data.iter() {
        let Some(tensor) = checkpoint.get(name) else {
            if name.starts_with("decoder.causal_start_embedding.") {
                causal_only_variables += 1;
                continue;
            }
            required_missing.push(name.clone());
            continue;
        };
        if variable.as_tensor().dims() != tensor.dims() {
            candle_core::bail!(
                "causal warm-start shape mismatch for '{name}': current {:?}, checkpoint {:?}",
                variable.as_tensor().dims(),
                tensor.dims()
            );
        }
        variable.set(tensor)?;
        loaded_variables += 1;
    }
    drop(data);

    if !required_missing.is_empty() {
        candle_core::bail!(
            "diffusion checkpoint is missing required causal warm-start variables: {}",
            required_missing.join(", ")
        );
    }
    let ignored_checkpoint_variables = checkpoint
        .keys()
        .filter(|name| !current_names.contains(*name))
        .count();

    Ok(FoundationCausalWarmStartReport {
        loaded_variables,
        causal_only_variables,
        ignored_checkpoint_variables,
    })
}
