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

    /// Create model-visible causal inputs for arbitrary clean prefixes.
    ///
    /// Each prefix contains only already-emitted non-EOS tokens. Position zero
    /// is supplied by the learned START embedding inside the model and the
    /// final active position predicts the next token after the supplied prefix.
    /// This is the inference contract used by the v0.12.4 left-to-right beam
    /// generator; unlike teacher-forcing collation it does not require a
    /// completed candidate or an EOS target.
    pub fn collate_prefix_rows(
        &self,
        prefixes: &[Vec<u32>],
        device: &Device,
    ) -> Result<FoundationCausalInputBatch> {
        if prefixes.is_empty() {
            candle_core::bail!("causal prefix collation requires at least one prefix");
        }
        let batch = prefixes.len();
        let width = self.config.max_tokens;
        let mut shifted = vec![FOUNDATION_DIFFUSION_PAD; batch * width];
        let mut mask = vec![0.0f32; batch * width];

        for (row_index, prefix) in prefixes.iter().enumerate() {
            if prefix.len() >= width {
                candle_core::bail!(
                    "causal prefix length {} leaves no position for next-token prediction in width {width}",
                    prefix.len()
                );
            }
            for (position, &token) in prefix.iter().enumerate() {
                if token == FOUNDATION_DIFFUSION_PAD
                    || token == FOUNDATION_DIFFUSION_MASK
                    || token == FOUNDATION_DIFFUSION_EOS
                {
                    candle_core::bail!("causal prefix position {position} contains PAD/MASK/EOS");
                }
                if token as usize >= FOUNDATION_DIFFUSION_VOCAB_SIZE {
                    candle_core::bail!("causal prefix token {token} exceeds vocabulary");
                }
                shifted[row_index * width + position + 1] = token;
            }
            for position in 0..=prefix.len() {
                mask[row_index * width + position] = 1.0;
            }
        }

        Ok(FoundationCausalInputBatch {
            input_tokens: Tensor::from_vec(shifted, (batch, width), device)?
                .to_dtype(DType::U32)?,
            token_mask: Tensor::from_vec(mask, (batch, width), device)?,
        })
    }

    /// Create a compact causal input containing only active prefix positions.
    ///
    /// All prefixes in one batch must have the same length, as they do at one
    /// beam-search depth. The returned width is `prefix.len() + 1` rather than
    /// `config.max_tokens`, eliminating inactive future positions during
    /// inference while preserving START + shifted-prefix semantics.
    pub fn collate_compact_prefix_rows(
        &self,
        prefixes: &[Vec<u32>],
        device: &Device,
    ) -> Result<FoundationCausalInputBatch> {
        if prefixes.is_empty() {
            candle_core::bail!("compact causal prefix collation requires at least one prefix");
        }
        let prefix_len = prefixes[0].len();
        if prefix_len >= self.config.max_tokens {
            candle_core::bail!(
                "causal prefix length {prefix_len} leaves no position for next-token prediction in configured width {}",
                self.config.max_tokens
            );
        }
        if prefixes.iter().any(|prefix| prefix.len() != prefix_len) {
            candle_core::bail!("compact causal prefix batches require equal prefix lengths");
        }

        let batch = prefixes.len();
        let width = prefix_len + 1;
        let mut shifted = vec![FOUNDATION_DIFFUSION_PAD; batch * width];
        let mask = vec![1.0f32; batch * width];
        for (row_index, prefix) in prefixes.iter().enumerate() {
            for (position, &token) in prefix.iter().enumerate() {
                if token == FOUNDATION_DIFFUSION_PAD
                    || token == FOUNDATION_DIFFUSION_MASK
                    || token == FOUNDATION_DIFFUSION_EOS
                {
                    candle_core::bail!("causal prefix position {position} contains PAD/MASK/EOS");
                }
                if token as usize >= FOUNDATION_DIFFUSION_VOCAB_SIZE {
                    candle_core::bail!("causal prefix token {token} exceeds vocabulary");
                }
                shifted[row_index * width + position + 1] = token;
            }
        }

        Ok(FoundationCausalInputBatch {
            input_tokens: Tensor::from_vec(shifted, (batch, width), device)?
                .to_dtype(DType::U32)?,
            token_mask: Tensor::from_vec(mask, (batch, width), device)?,
        })
    }
}

/// Output from a teacher-forced causal next-token prediction.
#[derive(Debug, Clone)]
pub struct FoundationCausalOutput {
    /// Next-token logits `[batch, max_tokens, vocabulary]`.
    pub token_logits: Tensor,
    /// Final spectrum-conditioned decoder hidden states `[batch, max_tokens, model_dim]`.
    ///
    /// This representation is exposed for frozen-backbone downstream interaction
    /// heads. It contains only model-visible shifted candidate prefixes plus
    /// spectrum/precursor cross-attention; clean next-token targets remain outside
    /// the model input contract.
    pub decoder_hidden: Tensor,
    /// Cross-attention memory, including the prepended precursor token.
    pub spectrum_memory: Tensor,
    /// Key-validity mask for `spectrum_memory` `[batch, memory_len]`.
    pub spectrum_memory_mask: Tensor,
    /// Mask-aware pooled observed-spectrum embedding.
    pub spectrum_embedding: Tensor,
}

/// Spectrum/precursor context cached independently of autoregressive prefixes.
///
/// v0.12.5 uses this immutable context during causal beam generation so the
/// observed-spectrum encoder and precursor projection run once per record rather
/// than once per beam position. Prefix tokens and decoder states are not cached,
/// so this changes execution cost only and preserves the v0.12.4 scoring model.
#[derive(Debug, Clone)]
pub struct FoundationCausalContext {
    spectrum_memory: Tensor,
    spectrum_memory_mask: Tensor,
    precursor_summary: Tensor,
    spectrum_embedding: Tensor,
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

    /// Encode the observed spectrum and precursor context once.
    ///
    /// The returned context contains no peptide tokens and can therefore be
    /// safely reused across arbitrary causal prefixes for the same record.
    pub fn prepare_context(
        &self,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationCausalContext> {
        let (spectrum_batch, _, _) = spectrum.peak_features.dims3()?;
        let precursor_batch = precursor.charge.dims1()?;
        if spectrum_batch != precursor_batch {
            candle_core::bail!(
                "causal spectrum/precursor batch mismatch: spectrum {spectrum_batch}, precursor {precursor_batch}"
            );
        }

        let spectrum_encoding = self.spectrum_encoder.forward_t(spectrum, train)?;

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
        let precursor_memory_mask = Tensor::ones(
            (spectrum_batch, 1),
            DType::F32,
            spectrum.peak_features.device(),
        )?;
        let spectrum_memory_mask = Tensor::cat(&[&precursor_memory_mask, &spectrum.peak_mask], 1)?;

        Ok(FoundationCausalContext {
            spectrum_memory,
            spectrum_memory_mask,
            precursor_summary,
            spectrum_embedding: spectrum_encoding.spectrum_embedding,
        })
    }

    /// Predict every next token using a precomputed spectrum/precursor context.
    ///
    /// A single-record context is broadcast across the prefix batch used by
    /// beam search. A context already matching the prefix batch is also valid.
    pub fn forward_t_with_context(
        &self,
        input: &FoundationCausalInputBatch,
        context: &FoundationCausalContext,
        train: bool,
    ) -> Result<FoundationCausalOutput> {
        let (batch, token_len) = input.input_tokens.dims2()?;
        if token_len == 0 || token_len > self.config.max_tokens {
            candle_core::bail!(
                "causal token width {token_len} must be within 1..={} ",
                self.config.max_tokens
            );
        }

        let (context_batch, memory_len, memory_dim) = context.spectrum_memory.dims3()?;
        if context_batch != 1 && context_batch != batch {
            candle_core::bail!(
                "causal/context batch mismatch: causal {batch}, context {context_batch}"
            );
        }
        let spectrum_memory = if context_batch == batch {
            context.spectrum_memory.clone()
        } else {
            context
                .spectrum_memory
                .broadcast_as((batch, memory_len, memory_dim))?
        };
        let (_, memory_mask_len) = context.spectrum_memory_mask.dims2()?;
        let spectrum_memory_mask = if context_batch == batch {
            context.spectrum_memory_mask.clone()
        } else {
            context
                .spectrum_memory_mask
                .broadcast_as((batch, memory_mask_len))?
        };
        let (_, precursor_dim) = context.precursor_summary.dims2()?;
        let precursor_summary = if context_batch == batch {
            context.precursor_summary.clone()
        } else {
            context
                .precursor_summary
                .broadcast_as((batch, precursor_dim))?
        };
        let (_, spectrum_embedding_dim) = context.spectrum_embedding.dims2()?;
        let spectrum_embedding = if context_batch == batch {
            context.spectrum_embedding.clone()
        } else {
            context
                .spectrum_embedding
                .broadcast_as((batch, spectrum_embedding_dim))?
        };

        // START is not a vocabulary id. The first position uses the dedicated
        // embedding directly; positions 1.. use shifted clean prefix tokens.
        let start_ids = Tensor::zeros(batch, DType::U32, input.input_tokens.device())?;
        let start_embedding = self
            .causal_start_embedding
            .forward(&start_ids)?
            .unsqueeze(1)?;
        let token_embedding = if token_len == 1 {
            start_embedding
        } else {
            let shifted_suffix = input.input_tokens.narrow(1, 1, token_len - 1)?;
            let shifted_embedding = self.token_embedding.forward(&shifted_suffix)?;
            Tensor::cat(&[&start_embedding, &shifted_embedding], 1)?
        };

        let positions: Vec<u32> = (0..token_len as u32).collect();
        let position_ids = Tensor::from_vec(positions, token_len, input.input_tokens.device())?
            .to_dtype(DType::U32)?;
        let position_embedding = self
            .position_embedding
            .forward(&position_ids)?
            .unsqueeze(0)?
            .broadcast_as((batch, token_len, self.config.model_dim))?;

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
            decoder_hidden: hidden,
            spectrum_memory,
            spectrum_memory_mask,
            spectrum_embedding,
        })
    }

    /// Predict only the next-token logits at the final active compact-prefix position.
    ///
    /// This inference-only helper preserves the historical causal decoder but
    /// avoids transferring or consuming logits for earlier prefix positions.
    pub fn forward_next_t_with_context(
        &self,
        input: &FoundationCausalInputBatch,
        context: &FoundationCausalContext,
        train: bool,
    ) -> Result<Tensor> {
        let (_, token_len) = input.input_tokens.dims2()?;
        if token_len == 0 {
            candle_core::bail!("causal next-token inference requires at least START position");
        }
        let output = self.forward_t_with_context(input, context, train)?;
        output.token_logits.narrow(1, token_len - 1, 1)?.squeeze(1)
    }

    /// Predict every next token in one shifted, teacher-forced causal pass.
    pub fn forward_t(
        &self,
        input: &FoundationCausalInputBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationCausalOutput> {
        let (batch, _) = input.input_tokens.dims2()?;
        let (spectrum_batch, _, _) = spectrum.peak_features.dims3()?;
        if batch != spectrum_batch {
            candle_core::bail!(
                "causal/spectrum batch mismatch: causal {batch}, spectrum {spectrum_batch}"
            );
        }
        let context = self.prepare_context(spectrum, precursor, train)?;
        self.forward_t_with_context(input, &context, train)
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

/// Hinge penalty that makes causal next-token likelihood spectrum-discriminative.
///
/// `matched_loss` and `shuffled_loss` are the same teacher-forced target loss
/// evaluated with the correct spectrum and a deliberately mismatched spectrum,
/// respectively. The penalty is
/// `max(0, margin + matched_loss - shuffled_loss)`, so minimizing it requires
/// the matched spectrum to beat the shuffled spectrum by at least `margin` nats.
/// Prefix tokens and precursor context are intentionally held fixed by callers.
pub fn foundation_causal_conditioning_margin_loss(
    matched_loss: &Tensor,
    shuffled_loss: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    if !(margin >= 0.0 && margin.is_finite()) {
        candle_core::bail!("causal conditioning margin must be finite and non-negative");
    }
    (matched_loss - shuffled_loss)?.affine(1.0, margin)?.relu()
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

#[cfg(test)]
mod conditioning_margin_tests {
    use super::foundation_causal_conditioning_margin_loss;
    use candle_core::{Device, Tensor};

    #[test]
    fn conditioning_margin_is_zero_after_required_gap() {
        let device = Device::Cpu;
        let matched = Tensor::new(&[1.0f32], &device).unwrap();
        let shuffled = Tensor::new(&[1.5f32], &device).unwrap();
        let loss = foundation_causal_conditioning_margin_loss(&matched, &shuffled, 0.25)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(loss.abs() < 1.0e-7);
    }

    #[test]
    fn conditioning_margin_penalizes_insufficient_gap() {
        let device = Device::Cpu;
        let matched = Tensor::new(&[1.0f32], &device).unwrap();
        let shuffled = Tensor::new(&[1.1f32], &device).unwrap();
        let loss = foundation_causal_conditioning_margin_loss(&matched, &shuffled, 0.25)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((loss - 0.15).abs() < 1.0e-6);
    }
}
