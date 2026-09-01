//! Spectrum-conditioned discrete diffusion for inverse peptide generation.
//!
//! The first implementation operates in residue/PTM token space and keeps the
//! existing chemistry-aware peptide graph encoder unchanged. The inverse model
//! consumes *observed* centroided MS/MS peaks, precursor m/z/charge, a noised
//! peptide-token sequence, and a diffusion timestep. It predicts the clean token
//! identity at every active sequence position.
//!
//! This is intentionally an x0-parameterized multinomial denoising scaffold.
//! Candidate initialization, exact reverse-posterior sampling, top-k generation,
//! cycle consistency, and forward-model rescoring are subsequent milestones.

use super::chemistry::common_unimod_definition;
use super::featurize::{
    exact_graph_modification_for, FoundationModification, FoundationModificationSite,
    PeptidoformInput,
};
use super::layers::{MultiHeadCrossAttention, MultiHeadSelfAttention, PeptideTransformerBlock};
use super::loss::contrastive_info_nce_loss;
use super::model::PrecursorContextBatch;
use super::spectrum::{FoundationSpectrumBatch, FoundationSpectrumConfig};
use candle_core::{DType, Device, Module, ModuleT, Result, Tensor};
use candle_nn::{self as nn, loss, Dropout, Embedding, LayerNorm, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

/// Padding token; never diffused or scored.
pub const FOUNDATION_DIFFUSION_PAD: u32 = 0;
/// Explicit unknown/noise token available to the multinomial corruption process.
pub const FOUNDATION_DIFFUSION_MASK: u32 = 1;
/// End-of-peptidoform token.
pub const FOUNDATION_DIFFUSION_EOS: u32 = 2;
/// First amino-acid token id.
pub const FOUNDATION_DIFFUSION_FIRST_RESIDUE: u32 = 3;
/// Peptide N-terminal UniMod:1 acetyl marker.
pub const FOUNDATION_DIFFUSION_NTERM_ACETYL: u32 = 23;
/// Residue-local UniMod:1 acetyl marker.
pub const FOUNDATION_DIFFUSION_RESIDUE_ACETYL: u32 = 24;
/// Residue-local UniMod:4 carbamidomethyl marker.
pub const FOUNDATION_DIFFUSION_CARBAMIDOMETHYL: u32 = 25;
/// Residue-local UniMod:7 deamidated marker.
pub const FOUNDATION_DIFFUSION_DEAMIDATED: u32 = 26;
/// Residue-local UniMod:35 oxidation marker.
pub const FOUNDATION_DIFFUSION_OXIDATION: u32 = 27;
/// Size of the initial residue/PTM vocabulary.
pub const FOUNDATION_DIFFUSION_VOCAB_SIZE: usize = 28;

const PROTON_MASS_DA: f64 = 1.007_276_466_77;
const WATER_MASS_DA: f64 = 18.010_564_684;

/// Configuration for the first spectrum-conditioned peptide diffusion model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FoundationDiffusionConfig {
    /// Maximum number of residue/PTM/EOS tokens.
    pub max_tokens: usize,
    /// Shared spectrum/decoder hidden width.
    pub model_dim: usize,
    /// Number of self/cross-attention heads.
    pub num_attention_heads: usize,
    /// Feed-forward hidden width.
    pub feed_forward_dim: usize,
    /// Spectrum-encoder Transformer layers.
    pub spectrum_layers: usize,
    /// Spectrum-conditioned denoising layers.
    pub decoder_layers: usize,
    /// Dropout probability.
    pub dropout: f32,
    /// Number of discrete diffusion/refinement steps.
    pub diffusion_steps: usize,
    /// First-step categorical replacement probability.
    pub beta_start: f64,
    /// Final-step categorical replacement probability.
    pub beta_end: f64,
    /// Observed-spectrum packing configuration.
    pub spectrum: FoundationSpectrumConfig,
    /// Default hard precursor neutral-mass tolerance for candidate filtering.
    pub precursor_mass_tolerance_da: f64,
}

impl Default for FoundationDiffusionConfig {
    fn default() -> Self {
        Self {
            max_tokens: 80,
            model_dim: 192,
            num_attention_heads: 4,
            feed_forward_dim: 768,
            spectrum_layers: 4,
            decoder_layers: 4,
            dropout: 0.05,
            diffusion_steps: 20,
            beta_start: 0.02,
            beta_end: 0.35,
            spectrum: FoundationSpectrumConfig::default(),
            precursor_mass_tolerance_da: 0.05,
        }
    }
}

impl FoundationDiffusionConfig {
    /// Validate model/schedule dimensions.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.max_tokens < 3 {
            return Err("foundation diffusion max_tokens must be at least 3".into());
        }
        if self.num_attention_heads == 0 {
            return Err("foundation diffusion num_attention_heads must be positive".into());
        }
        if self.model_dim == 0 || self.model_dim % self.num_attention_heads != 0 {
            return Err(
                "foundation diffusion model_dim must be positive and divisible by num_attention_heads"
                    .into(),
            );
        }
        if self.feed_forward_dim == 0 || self.spectrum_layers == 0 || self.decoder_layers == 0 {
            return Err("foundation diffusion layer dimensions/counts must be positive".into());
        }
        if !(0.0 <= self.dropout && self.dropout < 1.0) {
            return Err("foundation diffusion dropout must be in [0, 1)".into());
        }
        if self.diffusion_steps == 0 {
            return Err("foundation diffusion diffusion_steps must be positive".into());
        }
        if !(self.beta_start > 0.0
            && self.beta_start < 1.0
            && self.beta_end > 0.0
            && self.beta_end < 1.0
            && self.beta_start <= self.beta_end)
        {
            return Err(
                "foundation diffusion beta schedule must satisfy 0 < beta_start <= beta_end < 1"
                    .into(),
            );
        }
        if !(self.precursor_mass_tolerance_da > 0.0 && self.precursor_mass_tolerance_da.is_finite())
        {
            return Err(
                "foundation diffusion precursor mass tolerance must be positive and finite".into(),
            );
        }
        self.spectrum.validate()?;
        Ok(())
    }

    /// Linear per-step categorical replacement probability, using one-based timesteps.
    pub fn beta(&self, timestep: usize) -> std::result::Result<f64, String> {
        if timestep == 0 || timestep > self.diffusion_steps {
            return Err(format!(
                "foundation diffusion timestep {timestep} is outside 1..={}",
                self.diffusion_steps
            ));
        }
        if self.diffusion_steps == 1 {
            return Ok(self.beta_end);
        }
        let fraction = (timestep - 1) as f64 / (self.diffusion_steps - 1) as f64;
        Ok(self.beta_start + fraction * (self.beta_end - self.beta_start))
    }

    /// Cumulative clean-token retention probability `alpha_bar_t`.
    pub fn alpha_bar(&self, timestep: usize) -> std::result::Result<f64, String> {
        if timestep == 0 || timestep > self.diffusion_steps {
            return Err(format!(
                "foundation diffusion timestep {timestep} is outside 1..={}",
                self.diffusion_steps
            ));
        }
        let mut value = 1.0;
        for step in 1..=timestep {
            value *= 1.0 - self.beta(step)?;
        }
        Ok(value)
    }
}

/// Initial extensible residue/PTM vocabulary.
///
/// PTM markers are separate sequence tokens rather than atom-graph states. A
/// residue-local PTM marker follows the residue it modifies. N-terminal acetyl
/// precedes the first residue. The current vocabulary exactly covers the four
/// canonical modification families observed in the validated OpenSWATH/IP2
/// corpus; unsupported PTMs fail explicitly rather than being silently dropped.
#[derive(Debug, Clone, Copy, Default)]
pub struct FoundationDiffusionVocabulary;

impl FoundationDiffusionVocabulary {
    /// Vocabulary size.
    pub const fn size(self) -> usize {
        FOUNDATION_DIFFUSION_VOCAB_SIZE
    }

    /// Encode one peptidoform and append EOS/padding to `max_tokens`.
    pub fn encode(
        self,
        peptide: &PeptidoformInput,
        max_tokens: usize,
    ) -> std::result::Result<Vec<u32>, String> {
        let residues: Vec<char> = peptide.sequence.chars().collect();
        if residues.is_empty() {
            return Err("diffusion vocabulary cannot encode an empty peptide".into());
        }

        let mut tokens = Vec::with_capacity(max_tokens);
        let mut nterm_mods: Vec<&FoundationModification> = peptide
            .modifications
            .iter()
            .filter(|modification| modification.site == FoundationModificationSite::NTerm)
            .collect();
        nterm_mods.sort_by_key(|modification| modification.unimod_id.unwrap_or(u32::MAX));
        for modification in nterm_mods {
            match modification.unimod_id {
                Some(1) => tokens.push(FOUNDATION_DIFFUSION_NTERM_ACETYL),
                _ => {
                    return Err(format!(
                        "diffusion vocabulary does not yet support N-terminal {}",
                        modification.identity_label()
                    ))
                }
            }
        }

        for (residue_index, residue) in residues.iter().copied().enumerate() {
            tokens.push(residue_diffusion_token(residue).ok_or_else(|| {
                format!("diffusion vocabulary does not support residue '{residue}'")
            })?);
            let mut modifications: Vec<&FoundationModification> = peptide
                .modifications
                .iter()
                .filter(|modification| {
                    modification.site == FoundationModificationSite::Residue(residue_index)
                })
                .collect();
            modifications.sort_by_key(|modification| modification.unimod_id.unwrap_or(u32::MAX));
            for modification in modifications {
                if exact_graph_modification_for(residue, modification).is_none() {
                    return Err(format!(
                        "diffusion vocabulary cannot encode {} on residue {}",
                        modification.identity_label(),
                        residue
                    ));
                }
                tokens.push(residue_modification_token(modification)?);
            }
        }

        if let Some(modification) = peptide
            .modifications
            .iter()
            .find(|modification| modification.site == FoundationModificationSite::CTerm)
        {
            return Err(format!(
                "diffusion vocabulary does not yet support C-terminal {}",
                modification.identity_label()
            ));
        }

        tokens.push(FOUNDATION_DIFFUSION_EOS);
        if tokens.len() > max_tokens {
            return Err(format!(
                "diffusion token length {} exceeds configured maximum {} for {}",
                tokens.len(),
                max_tokens,
                peptide.sequence
            ));
        }
        tokens.resize(max_tokens, FOUNDATION_DIFFUSION_PAD);
        Ok(tokens)
    }

    /// Decode one token row into a chemistry-aware peptidoform.
    pub fn decode(self, tokens: &[u32]) -> std::result::Result<PeptidoformInput, String> {
        let mut sequence = String::new();
        let mut modifications = Vec::new();
        let mut saw_eos = false;

        for &token in tokens {
            match token {
                FOUNDATION_DIFFUSION_PAD => break,
                FOUNDATION_DIFFUSION_MASK => {
                    return Err("cannot decode a peptide containing MASK tokens".into())
                }
                FOUNDATION_DIFFUSION_EOS => {
                    saw_eos = true;
                    break;
                }
                FOUNDATION_DIFFUSION_NTERM_ACETYL => {
                    if !sequence.is_empty() {
                        return Err("N-terminal acetyl token must precede residues".into());
                    }
                    let definition = common_unimod_definition(1)
                        .ok_or_else(|| "UniMod:1 definition is unavailable".to_string())?;
                    modifications.push(FoundationModification::unimod(
                        FoundationModificationSite::NTerm,
                        0,
                        1,
                        definition.mass_delta,
                    ));
                }
                FOUNDATION_DIFFUSION_RESIDUE_ACETYL
                | FOUNDATION_DIFFUSION_CARBAMIDOMETHYL
                | FOUNDATION_DIFFUSION_DEAMIDATED
                | FOUNDATION_DIFFUSION_OXIDATION => {
                    let residue_index =
                        sequence.chars().count().checked_sub(1).ok_or_else(|| {
                            "residue PTM token appeared before any residue".to_string()
                        })?;
                    let unimod_id = match token {
                        FOUNDATION_DIFFUSION_RESIDUE_ACETYL => 1,
                        FOUNDATION_DIFFUSION_CARBAMIDOMETHYL => 4,
                        FOUNDATION_DIFFUSION_DEAMIDATED => 7,
                        FOUNDATION_DIFFUSION_OXIDATION => 35,
                        _ => unreachable!(),
                    };
                    let definition = common_unimod_definition(unimod_id)
                        .ok_or_else(|| format!("UniMod:{unimod_id} definition is unavailable"))?;
                    let residue = sequence
                        .chars()
                        .nth(residue_index)
                        .ok_or_else(|| "diffusion PTM residue index is invalid".to_string())?;
                    let modification = FoundationModification::unimod(
                        FoundationModificationSite::Residue(residue_index),
                        residue_index,
                        unimod_id,
                        definition.mass_delta,
                    );
                    if exact_graph_modification_for(residue, &modification).is_none() {
                        return Err(format!(
                            "diffusion token {} is not valid on residue {}",
                            self.token_label(token),
                            residue
                        ));
                    }
                    modifications.push(modification);
                }
                _ => {
                    let residue = diffusion_token_residue(token)
                        .ok_or_else(|| format!("unknown diffusion token id {token}"))?;
                    sequence.push(residue);
                }
            }
        }

        if !saw_eos {
            return Err("diffusion peptide token sequence did not contain EOS".into());
        }
        if sequence.is_empty() {
            return Err("diffusion token sequence decoded to an empty peptide".into());
        }
        Ok(PeptidoformInput {
            sequence,
            modifications,
        })
    }

    /// Human-readable token label for diagnostics.
    pub fn token_label(self, token: u32) -> &'static str {
        match token {
            FOUNDATION_DIFFUSION_PAD => "PAD",
            FOUNDATION_DIFFUSION_MASK => "MASK",
            FOUNDATION_DIFFUSION_EOS => "EOS",
            FOUNDATION_DIFFUSION_NTERM_ACETYL => "[N-term UniMod:1]",
            FOUNDATION_DIFFUSION_RESIDUE_ACETYL => "[UniMod:1]",
            FOUNDATION_DIFFUSION_CARBAMIDOMETHYL => "[UniMod:4]",
            FOUNDATION_DIFFUSION_DEAMIDATED => "[UniMod:7]",
            FOUNDATION_DIFFUSION_OXIDATION => "[UniMod:35]",
            3 => "A",
            4 => "C",
            5 => "D",
            6 => "E",
            7 => "F",
            8 => "G",
            9 => "H",
            10 => "I",
            11 => "K",
            12 => "L",
            13 => "M",
            14 => "N",
            15 => "P",
            16 => "Q",
            17 => "R",
            18 => "S",
            19 => "T",
            20 => "V",
            21 => "W",
            22 => "Y",
            _ => "UNKNOWN",
        }
    }
}

fn residue_diffusion_token(residue: char) -> Option<u32> {
    let offset = match residue {
        'A' => 0,
        'C' => 1,
        'D' => 2,
        'E' => 3,
        'F' => 4,
        'G' => 5,
        'H' => 6,
        'I' => 7,
        'K' => 8,
        'L' => 9,
        'M' => 10,
        'N' => 11,
        'P' => 12,
        'Q' => 13,
        'R' => 14,
        'S' => 15,
        'T' => 16,
        'V' => 17,
        'W' => 18,
        'Y' => 19,
        _ => return None,
    };
    Some(FOUNDATION_DIFFUSION_FIRST_RESIDUE + offset)
}

fn diffusion_token_residue(token: u32) -> Option<char> {
    const RESIDUES: [char; 20] = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
        'W', 'Y',
    ];
    let offset = token.checked_sub(FOUNDATION_DIFFUSION_FIRST_RESIDUE)? as usize;
    RESIDUES.get(offset).copied()
}

fn residue_modification_token(
    modification: &FoundationModification,
) -> std::result::Result<u32, String> {
    match modification.unimod_id {
        Some(1) => Ok(FOUNDATION_DIFFUSION_RESIDUE_ACETYL),
        Some(4) => Ok(FOUNDATION_DIFFUSION_CARBAMIDOMETHYL),
        Some(7) => Ok(FOUNDATION_DIFFUSION_DEAMIDATED),
        Some(35) => Ok(FOUNDATION_DIFFUSION_OXIDATION),
        _ => Err(format!(
            "diffusion vocabulary does not yet support residue {}",
            modification.identity_label()
        )),
    }
}

/// Tensorized noised/clean token batch for x0 denoising training.
#[derive(Debug, Clone)]
pub struct FoundationDiffusionBatch {
    /// Noised token ids `[batch, max_tokens]`.
    pub noisy_tokens: Tensor,
    /// Clean token ids `[batch, max_tokens]`.
    pub clean_tokens: Tensor,
    /// Valid clean sequence mask including EOS `[batch, max_tokens]`.
    pub token_mask: Tensor,
    /// One-based integer diffusion timestep `[batch]`.
    pub timesteps: Tensor,
    /// Bounded continuous timestep features `[batch, 4]`.
    pub timestep_features: Tensor,
    /// Flattened valid positions used by cross-entropy.
    pub active_indices: Tensor,
    /// Clean classes aligned with `active_indices`.
    pub target_classes: Tensor,
    /// Zero-based active-token length classes `[batch]`, where class `k` means `k + 1` active tokens.
    pub length_targets: Tensor,
}

/// Deterministic multinomial forward-process collator.
#[derive(Debug, Clone)]
pub struct FoundationDiffusionCollator {
    config: FoundationDiffusionConfig,
    vocabulary: FoundationDiffusionVocabulary,
}

impl FoundationDiffusionCollator {
    /// Construct the diffusion collator.
    pub fn new(config: FoundationDiffusionConfig) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        Ok(Self {
            config,
            vocabulary: FoundationDiffusionVocabulary,
        })
    }

    /// Collate using deterministic random timesteps in `1..=diffusion_steps`.
    pub fn collate_random_timesteps(
        &self,
        peptides: &[PeptidoformInput],
        seed: u64,
        device: &Device,
    ) -> Result<FoundationDiffusionBatch> {
        let mut rng = DiffusionRng::new(seed);
        let timesteps: Vec<usize> = peptides
            .iter()
            .map(|_| 1 + (rng.next_u64() as usize % self.config.diffusion_steps))
            .collect();
        self.collate(peptides, &timesteps, seed ^ 0x517c_c1b7_2722_0a95, device)
    }

    /// Collate with explicit timesteps, useful for deterministic tests/audits.
    pub fn collate(
        &self,
        peptides: &[PeptidoformInput],
        timesteps: &[usize],
        seed: u64,
        device: &Device,
    ) -> Result<FoundationDiffusionBatch> {
        self.collate_impl(peptides, timesteps, seed, false, device)
    }

    /// Collate a spectrum-forcing batch whose active peptide tokens are all replaced by MASK.
    ///
    /// The clean active-token mask is retained for this auxiliary training objective, so this is
    /// not yet a de-novo generation interface. It is intentionally useful for verifying that the
    /// spectrum/precursor path carries sequence information when x_t itself contains none.
    pub fn collate_all_masked(
        &self,
        peptides: &[PeptidoformInput],
        timestep: usize,
        device: &Device,
    ) -> Result<FoundationDiffusionBatch> {
        let timesteps = vec![timestep; peptides.len()];
        self.collate_impl(peptides, &timesteps, 0, true, device)
    }

    fn collate_impl(
        &self,
        peptides: &[PeptidoformInput],
        timesteps: &[usize],
        seed: u64,
        force_all_masked: bool,
        device: &Device,
    ) -> Result<FoundationDiffusionBatch> {
        if peptides.is_empty() || peptides.len() != timesteps.len() {
            candle_core::bail!(
                "diffusion collation requires non-empty peptide/timestep arrays of equal length"
            );
        }
        let b = peptides.len();
        let l = self.config.max_tokens;
        let mut clean = vec![FOUNDATION_DIFFUSION_PAD; b * l];
        let mut noisy = vec![FOUNDATION_DIFFUSION_PAD; b * l];
        let mut mask = vec![0.0f32; b * l];
        let mut active_indices = Vec::<u32>::new();
        let mut target_classes = Vec::<u32>::new();
        let mut length_targets = Vec::<u32>::with_capacity(b);
        let mut timestep_ids = Vec::<u32>::with_capacity(b);
        let mut timestep_features = Vec::<f32>::with_capacity(b * 4);
        let mut rng = DiffusionRng::new(seed);

        for (batch_idx, (peptide, &timestep)) in peptides.iter().zip(timesteps).enumerate() {
            let tokens = self
                .vocabulary
                .encode(peptide, l)
                .map_err(candle_core::Error::Msg)?;
            let alpha_bar = self
                .config
                .alpha_bar(timestep)
                .map_err(candle_core::Error::Msg)?;
            timestep_ids.push(timestep as u32);
            timestep_features.extend_from_slice(&diffusion_timestep_features(
                timestep,
                self.config.diffusion_steps,
                alpha_bar,
            ));

            let mut active_length = 0usize;
            for (position, &token) in tokens.iter().enumerate() {
                let flat = batch_idx * l + position;
                clean[flat] = token;
                if token == FOUNDATION_DIFFUSION_PAD {
                    noisy[flat] = FOUNDATION_DIFFUSION_PAD;
                    continue;
                }
                active_length += 1;
                mask[flat] = 1.0;
                active_indices.push(flat as u32);
                target_classes.push(token);
                noisy[flat] = if force_all_masked {
                    FOUNDATION_DIFFUSION_MASK
                } else if rng.next_f64() < alpha_bar {
                    token
                } else {
                    // Multinomial replacement over all non-padding categories,
                    // including MASK and EOS. Padding is structural and remains fixed.
                    1 + (rng.next_u64() % (FOUNDATION_DIFFUSION_VOCAB_SIZE as u64 - 1)) as u32
                };
            }
            debug_assert!(active_length > 0 && active_length <= l);
            length_targets.push((active_length - 1) as u32);
        }

        let active_count = target_classes.len();
        debug_assert_eq!(active_indices.len(), active_count);

        Ok(FoundationDiffusionBatch {
            noisy_tokens: Tensor::from_vec(noisy, (b, l), device)?.to_dtype(DType::U32)?,
            clean_tokens: Tensor::from_vec(clean, (b, l), device)?.to_dtype(DType::U32)?,
            token_mask: Tensor::from_vec(mask, (b, l), device)?,
            timesteps: Tensor::from_vec(timestep_ids, b, device)?.to_dtype(DType::U32)?,
            timestep_features: Tensor::from_vec(timestep_features, (b, 4), device)?,
            active_indices: Tensor::from_vec(active_indices, active_count, device)?
                .to_dtype(DType::U32)?,
            target_classes: Tensor::from_vec(target_classes, active_count, device)?
                .to_dtype(DType::U32)?,
            length_targets: Tensor::from_vec(length_targets, b, device)?.to_dtype(DType::U32)?,
        })
    }
}

fn diffusion_timestep_features(timestep: usize, total_steps: usize, alpha_bar: f64) -> [f32; 4] {
    let fraction = timestep as f64 / total_steps as f64;
    [
        fraction as f32,
        (std::f64::consts::PI * fraction).sin() as f32,
        (std::f64::consts::PI * fraction).cos() as f32,
        alpha_bar as f32,
    ]
}

/// Contextualized spectrum representation.
#[derive(Debug, Clone)]
pub struct FoundationSpectrumEncoding {
    /// Per-peak contextual embeddings `[batch, peaks, model_dim]`.
    pub peak_embeddings: Tensor,
    /// Mask-aware pooled spectrum embedding `[batch, model_dim]`.
    pub spectrum_embedding: Tensor,
}

/// Transformer spectrum encoder for observed peak features.
#[derive(Clone)]
pub struct FoundationSpectrumEncoder {
    input_projection: Linear,
    layers: Vec<PeptideTransformerBlock>,
    output_norm: LayerNorm,
}

impl FoundationSpectrumEncoder {
    /// Build the spectrum encoder.
    pub fn new(config: &FoundationDiffusionConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.spectrum_layers);
        for index in 0..config.spectrum_layers {
            layers.push(PeptideTransformerBlock::new(
                config.model_dim,
                config.num_attention_heads,
                config.feed_forward_dim,
                config.dropout,
                vb.pp(format!("layers.{index}")),
            )?);
        }
        Ok(Self {
            input_projection: nn::linear(
                config.spectrum.peak_feature_dim,
                config.model_dim,
                vb.pp("input_projection"),
            )?,
            layers,
            output_norm: nn::layer_norm(config.model_dim, 1e-5, vb.pp("output_norm"))?,
        })
    }

    /// Encode observed product-ion peaks and produce a pooled spectrum embedding.
    pub fn forward_t(
        &self,
        batch: &FoundationSpectrumBatch,
        train: bool,
    ) -> Result<FoundationSpectrumEncoding> {
        let mut hidden = self.input_projection.forward(&batch.peak_features)?;
        let (b, p, d) = hidden.dims3()?;
        let mask = batch.peak_mask.unsqueeze(2)?.broadcast_as((b, p, d))?;
        hidden = hidden.broadcast_mul(&mask)?;
        for layer in &self.layers {
            hidden = layer.forward_t(&hidden, &batch.peak_mask, train)?;
        }
        let peak_embeddings = self.output_norm.forward(&hidden)?.broadcast_mul(&mask)?;
        let summed = peak_embeddings.sum(1)?;
        let denominator = batch
            .peak_mask
            .sum(1)?
            .clamp(1.0, f64::INFINITY)?
            .unsqueeze(1)?;
        let spectrum_embedding = summed.broadcast_div(&denominator)?;
        Ok(FoundationSpectrumEncoding {
            peak_embeddings,
            spectrum_embedding,
        })
    }
}

/// One pre-norm spectrum-conditioned denoising block.
#[derive(Clone)]
struct SpectrumConditionedDiffusionBlock {
    self_norm: LayerNorm,
    self_attention: MultiHeadSelfAttention,
    cross_norm: LayerNorm,
    cross_attention: MultiHeadCrossAttention,
    ff_norm: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    dropout: Dropout,
}

impl SpectrumConditionedDiffusionBlock {
    fn new(config: &FoundationDiffusionConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            self_norm: nn::layer_norm(config.model_dim, 1e-5, vb.pp("self_norm"))?,
            self_attention: MultiHeadSelfAttention::new(
                config.model_dim,
                config.num_attention_heads,
                vb.pp("self_attention"),
            )?,
            cross_norm: nn::layer_norm(config.model_dim, 1e-5, vb.pp("cross_norm"))?,
            cross_attention: MultiHeadCrossAttention::new(
                config.model_dim,
                config.num_attention_heads,
                vb.pp("cross_attention"),
            )?,
            ff_norm: nn::layer_norm(config.model_dim, 1e-5, vb.pp("ff_norm"))?,
            ff_in: nn::linear(config.model_dim, config.feed_forward_dim, vb.pp("ff_in"))?,
            ff_out: nn::linear(config.feed_forward_dim, config.model_dim, vb.pp("ff_out"))?,
            dropout: Dropout::new(config.dropout),
        })
    }

    fn forward_t(
        &self,
        hidden: &Tensor,
        token_mask: &Tensor,
        spectrum_memory: &Tensor,
        spectrum_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let normalized = self.self_norm.forward(hidden)?;
        let self_attention = self.self_attention.forward(&normalized, token_mask)?;
        let hidden = (hidden + self.dropout.forward_t(&self_attention, train)?)?;

        let normalized = self.cross_norm.forward(&hidden)?;
        let cross_attention =
            self.cross_attention
                .forward(&normalized, spectrum_memory, spectrum_mask)?;
        let hidden = (hidden + self.dropout.forward_t(&cross_attention, train)?)?;

        let normalized = self.ff_norm.forward(&hidden)?;
        let ff = self.ff_in.forward(&normalized)?.relu()?;
        let ff = self.ff_out.forward(&ff)?;
        let hidden = (hidden + self.dropout.forward_t(&ff, train)?)?;

        let (b, l, d) = hidden.dims3()?;
        hidden.broadcast_mul(&token_mask.unsqueeze(2)?.broadcast_as((b, l, d))?)
    }
}

/// Output from one spectrum-conditioned x0 denoising prediction.
#[derive(Debug, Clone)]
pub struct FoundationDiffusionOutput {
    /// Clean-token logits `[batch, max_tokens, vocabulary]`.
    pub token_logits: Tensor,
    /// Contextualized observed-spectrum peaks `[batch, peaks, model_dim]`.
    pub spectrum_memory: Tensor,
    /// Mask-aware pooled observed-spectrum embedding `[batch, model_dim]`.
    pub spectrum_embedding: Tensor,
    /// Active-token length logits `[batch, max_tokens]`; class `k` means `k + 1` active tokens.
    pub length_logits: Tensor,
}

/// Bidirectional foundation-model inverse scaffold.
#[derive(Clone)]
pub struct PeptideSpectrumDiffusionModel {
    config: FoundationDiffusionConfig,
    spectrum_encoder: FoundationSpectrumEncoder,
    token_embedding: Embedding,
    position_embedding: Embedding,
    precursor_projection: Linear,
    timestep_projection: Linear,
    layers: Vec<SpectrumConditionedDiffusionBlock>,
    output_norm: LayerNorm,
    token_head: Linear,
    length_head: Linear,
}

impl PeptideSpectrumDiffusionModel {
    /// Construct the observed-spectrum encoder and peptide diffusion decoder.
    pub fn new(config: FoundationDiffusionConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let spectrum_encoder = FoundationSpectrumEncoder::new(&config, vb.pp("spectrum_encoder"))?;
        let token_embedding = nn::embedding(
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            config.model_dim,
            vb.pp("decoder.token_embedding"),
        )?;
        let position_embedding = nn::embedding(
            config.max_tokens,
            config.model_dim,
            vb.pp("decoder.position_embedding"),
        )?;
        let precursor_projection = nn::linear(4, config.model_dim, vb.pp("decoder.precursor"))?;
        let timestep_projection = nn::linear(4, config.model_dim, vb.pp("decoder.timestep"))?;
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for index in 0..config.decoder_layers {
            layers.push(SpectrumConditionedDiffusionBlock::new(
                &config,
                vb.pp(format!("decoder.layers.{index}")),
            )?);
        }
        let output_norm = nn::layer_norm(config.model_dim, 1e-5, vb.pp("decoder.output_norm"))?;
        let token_head = nn::linear(
            config.model_dim,
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            vb.pp("decoder.token_head"),
        )?;
        let length_head = nn::linear(
            config.model_dim,
            config.max_tokens,
            vb.pp("decoder.length_head"),
        )?;
        Ok(Self {
            config,
            spectrum_encoder,
            token_embedding,
            position_embedding,
            precursor_projection,
            timestep_projection,
            layers,
            output_norm,
            token_head,
            length_head,
        })
    }

    /// Predict the clean peptide/PTM token sequence from a noised sequence and observed spectrum.
    pub fn forward_t(
        &self,
        diffusion: &FoundationDiffusionBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationDiffusionOutput> {
        let (batch, token_len) = diffusion.noisy_tokens.dims2()?;
        let (spectrum_batch, _, _) = spectrum.peak_features.dims3()?;
        if batch != spectrum_batch {
            candle_core::bail!(
                "diffusion/spectrum batch mismatch: diffusion {batch}, spectrum {spectrum_batch}"
            );
        }
        if token_len != self.config.max_tokens {
            candle_core::bail!(
                "diffusion token width {token_len} does not match configured {}",
                self.config.max_tokens
            );
        }

        let spectrum_encoding = self.spectrum_encoder.forward_t(spectrum, train)?;
        let spectrum_memory = spectrum_encoding.peak_embeddings.clone();
        let token_embedding = self.token_embedding.forward(&diffusion.noisy_tokens)?;
        let positions: Vec<u32> = (0..token_len as u32).collect();
        let position_ids = Tensor::from_vec(positions, token_len, diffusion.noisy_tokens.device())?
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
        let precursor_mz_present = precursor.precursor_mz_present.unsqueeze(1)?;
        let charge_present = precursor.charge_present.unsqueeze(1)?;
        let precursor_features = Tensor::cat(
            &[
                &scaled_precursor_mz,
                &scaled_charge,
                &precursor_mz_present,
                &charge_present,
            ],
            1,
        )?;
        let precursor_summary = self.precursor_projection.forward(&precursor_features)?;
        let length_hidden = (&spectrum_encoding.spectrum_embedding + &precursor_summary)?;
        let length_logits = self.length_head.forward(&length_hidden)?;
        let precursor_embedding = precursor_summary.unsqueeze(1)?.broadcast_as((
            batch,
            token_len,
            self.config.model_dim,
        ))?;
        let timestep_embedding = self
            .timestep_projection
            .forward(&diffusion.timestep_features)?
            .unsqueeze(1)?
            .broadcast_as((batch, token_len, self.config.model_dim))?;

        let mut hidden = (((token_embedding + position_embedding)? + precursor_embedding)?
            + timestep_embedding)?;
        let token_mask = diffusion.token_mask.unsqueeze(2)?.broadcast_as((
            batch,
            token_len,
            self.config.model_dim,
        ))?;
        hidden = hidden.broadcast_mul(&token_mask)?;
        for layer in &self.layers {
            hidden = layer.forward_t(
                &hidden,
                &diffusion.token_mask,
                &spectrum_memory,
                &spectrum.peak_mask,
                train,
            )?;
        }
        hidden = self
            .output_norm
            .forward(&hidden)?
            .broadcast_mul(&token_mask)?;
        let token_logits = self.token_head.forward(&hidden)?;
        Ok(FoundationDiffusionOutput {
            token_logits,
            spectrum_memory,
            spectrum_embedding: spectrum_encoding.spectrum_embedding,
            length_logits,
        })
    }

    /// Return the model configuration.
    pub fn config(&self) -> &FoundationDiffusionConfig {
        &self.config
    }
}

/// Cross-entropy x0 reconstruction objective over all non-padding clean tokens.
pub fn foundation_diffusion_x0_loss(
    output: &FoundationDiffusionOutput,
    batch: &FoundationDiffusionBatch,
) -> Result<Tensor> {
    let (b, l, classes) = output.token_logits.dims3()?;
    if classes != FOUNDATION_DIFFUSION_VOCAB_SIZE {
        candle_core::bail!(
            "diffusion logits expose {classes} classes, expected {}",
            FOUNDATION_DIFFUSION_VOCAB_SIZE
        );
    }
    let flat_logits = output.token_logits.reshape((b * l, classes))?;
    let selected_logits = flat_logits.index_select(&batch.active_indices, 0)?;
    loss::cross_entropy(&selected_logits, &batch.target_classes)
}

/// Cross-entropy sequence-length objective used to initialize true de-novo generation.
///
/// Class `k` corresponds to `k + 1` active residue/PTM/EOS tokens. The target is
/// derived from the clean sequence during training but the prediction itself depends
/// only on the observed spectrum and precursor context.
pub fn foundation_diffusion_length_loss(
    output: &FoundationDiffusionOutput,
    batch: &FoundationDiffusionBatch,
) -> Result<Tensor> {
    let (_, classes) = output.length_logits.dims2()?;
    if classes == 0 {
        candle_core::bail!("diffusion length head exposes zero classes");
    }
    loss::cross_entropy(&output.length_logits, &batch.length_targets)
}

/// Symmetric contrastive alignment between observed-spectrum and chemistry-aware
/// peptide embeddings.
///
/// This is the first explicit bridge by which the inverse spectrum lane can
/// regularize the existing foundation representation. It is intentionally an
/// optional objective rather than being baked into the denoising loss.
pub fn foundation_spectrum_peptide_alignment_loss(
    spectrum_embedding: &Tensor,
    peptide_embedding: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    contrastive_info_nce_loss(spectrum_embedding, peptide_embedding, temperature)
}

/// Neutral peptide mass including water and all modification deltas.
pub fn foundation_peptidoform_neutral_mass(
    peptide: &PeptidoformInput,
) -> std::result::Result<f64, String> {
    let mut mass = WATER_MASS_DA;
    for residue in peptide.sequence.chars() {
        mass += residue_mass(residue)
            .ok_or_else(|| format!("unsupported residue '{residue}' for peptide mass"))?;
    }
    for modification in &peptide.modifications {
        if !modification.mass_delta.is_finite() {
            return Err(format!(
                "non-finite modification mass for {}",
                modification.identity_label()
            ));
        }
        mass += modification.mass_delta as f64;
    }
    Ok(mass)
}

/// Convert measured precursor m/z and positive charge into neutral mass.
pub fn foundation_precursor_neutral_mass(
    precursor_mz: f64,
    charge: i32,
) -> std::result::Result<f64, String> {
    if !(precursor_mz > 0.0 && precursor_mz.is_finite()) || charge <= 0 {
        return Err("precursor m/z must be finite/positive and charge must be positive".into());
    }
    Ok(precursor_mz * charge as f64 - PROTON_MASS_DA * charge as f64)
}

/// Signed candidate-minus-observed neutral precursor mass error in Da.
pub fn foundation_precursor_mass_error_da(
    peptide: &PeptidoformInput,
    precursor_mz: f64,
    charge: i32,
) -> std::result::Result<f64, String> {
    Ok(foundation_peptidoform_neutral_mass(peptide)?
        - foundation_precursor_neutral_mass(precursor_mz, charge)?)
}

/// Whether a candidate satisfies a hard precursor neutral-mass tolerance.
pub fn foundation_precursor_mass_consistent(
    peptide: &PeptidoformInput,
    precursor_mz: f64,
    charge: i32,
    tolerance_da: f64,
) -> std::result::Result<bool, String> {
    if !(tolerance_da > 0.0 && tolerance_da.is_finite()) {
        return Err("precursor mass tolerance must be positive and finite".into());
    }
    Ok(foundation_precursor_mass_error_da(peptide, precursor_mz, charge)?.abs() <= tolerance_da)
}

fn residue_mass(residue: char) -> Option<f64> {
    // Monoisotopic amino-acid residue masses (free amino acid minus H2O).
    Some(match residue {
        'A' => 71.037_113_805,
        'C' => 103.009_184_505,
        'D' => 115.026_943_065,
        'E' => 129.042_593_135,
        'F' => 147.068_413_945,
        'G' => 57.021_463_735,
        'H' => 137.058_911_875,
        'I' | 'L' => 113.084_063_975,
        'K' => 128.094_963_015,
        'M' => 131.040_484_645,
        'N' => 114.042_927_470,
        'P' => 97.052_763_875,
        'Q' => 128.058_577_540,
        'R' => 156.101_111_050,
        'S' => 87.032_028_435,
        'T' => 101.047_678_505,
        'V' => 99.068_413_945,
        'W' => 186.079_312_980,
        'Y' => 163.063_328_575,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy)]
struct DiffusionRng {
    state: u64,
}

impl DiffusionRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xa076_1d64_78bd_642f,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 / ((1u64 << 53) - 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::FoundationSpectrum;
    use candle_nn::{VarBuilder, VarMap};

    fn modified_peptide() -> PeptidoformInput {
        let oxidation = common_unimod_definition(35).unwrap();
        let acetyl = common_unimod_definition(1).unwrap();
        PeptidoformInput {
            sequence: "PEPMIDEK".into(),
            modifications: vec![
                FoundationModification::unimod(
                    FoundationModificationSite::NTerm,
                    0,
                    1,
                    acetyl.mass_delta,
                ),
                FoundationModification::unimod(
                    FoundationModificationSite::Residue(3),
                    3,
                    35,
                    oxidation.mass_delta,
                ),
            ],
        }
    }

    #[test]
    fn residue_ptm_vocabulary_round_trips_current_corpus_chemistry() {
        let vocabulary = FoundationDiffusionVocabulary;
        let peptide = modified_peptide();
        let tokens = vocabulary.encode(&peptide, 32).unwrap();
        assert_eq!(tokens[0], FOUNDATION_DIFFUSION_NTERM_ACETYL);
        assert!(tokens.contains(&FOUNDATION_DIFFUSION_OXIDATION));
        let decoded = vocabulary.decode(&tokens).unwrap();
        assert_eq!(decoded, peptide);
    }

    #[test]
    fn multinomial_corruption_is_deterministic_and_schedule_adds_noise() {
        let device = Device::Cpu;
        let config = FoundationDiffusionConfig {
            max_tokens: 24,
            ..FoundationDiffusionConfig::default()
        };
        assert!(config.alpha_bar(1).unwrap() > config.alpha_bar(20).unwrap());
        let collator = FoundationDiffusionCollator::new(config).unwrap();
        let peptide = PeptidoformInput::unmodified("PEPTIDEK");
        let early = collator
            .collate(&[peptide.clone()], &[1], 7, &device)
            .unwrap();
        let early_again = collator.collate(&[peptide], &[1], 7, &device).unwrap();
        assert_eq!(
            early.noisy_tokens.to_vec2::<u32>().unwrap(),
            early_again.noisy_tokens.to_vec2::<u32>().unwrap()
        );
    }

    #[test]
    fn precursor_mass_constraint_round_trips_theoretical_mz() {
        let peptide = modified_peptide();
        let charge = 2;
        let neutral = foundation_peptidoform_neutral_mass(&peptide).unwrap();
        let mz = (neutral + charge as f64 * PROTON_MASS_DA) / charge as f64;
        let error = foundation_precursor_mass_error_da(&peptide, mz, charge).unwrap();
        assert!(error.abs() < 1e-8);
        assert!(foundation_precursor_mass_consistent(&peptide, mz, charge, 0.01).unwrap());
    }

    #[test]
    fn spectrum_conditioned_diffusion_forward_and_loss_have_expected_shapes() {
        let device = Device::Cpu;
        let config = FoundationDiffusionConfig {
            max_tokens: 24,
            model_dim: 32,
            num_attention_heads: 4,
            feed_forward_dim: 64,
            spectrum_layers: 1,
            decoder_layers: 1,
            spectrum: FoundationSpectrumConfig {
                max_peaks: 8,
                ..FoundationSpectrumConfig::default()
            },
            ..FoundationDiffusionConfig::default()
        };
        let spectra = vec![
            FoundationSpectrum::from_pairs([(100.0, 10.0), (250.0, 30.0), (500.0, 20.0)]),
            FoundationSpectrum::from_pairs([(120.0, 15.0), (330.0, 25.0), (700.0, 5.0)]),
        ];
        let spectrum_collator =
            crate::foundation::FoundationSpectrumCollator::new(config.spectrum.clone()).unwrap();
        let spectrum_batch = spectrum_collator.collate(&spectra, &device).unwrap();
        let peptides = vec![
            PeptidoformInput::unmodified("PEPTIDEK"),
            PeptidoformInput::unmodified("MELTQK"),
        ];
        let diffusion_batch = FoundationDiffusionCollator::new(config.clone())
            .unwrap()
            .collate(&peptides, &[4, 17], 19, &device)
            .unwrap();
        let precursor = PrecursorContextBatch {
            charge: Tensor::new(&[2.0f32, 3.0], &device).unwrap(),
            charge_present: Tensor::ones(2, DType::F32, &device).unwrap(),
            precursor_mz: Tensor::new(&[500.0f32, 600.0], &device).unwrap(),
            precursor_mz_present: Tensor::ones(2, DType::F32, &device).unwrap(),
            nce: Tensor::zeros(2, DType::F32, &device).unwrap(),
            nce_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
            instrument_ids: Tensor::zeros(2, DType::U32, &device).unwrap(),
            instrument_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideSpectrumDiffusionModel::new(config.clone(), vb).unwrap();
        let output = model
            .forward_t(&diffusion_batch, &spectrum_batch, &precursor, true)
            .unwrap();
        assert_eq!(
            output.token_logits.dims(),
            &[2, config.max_tokens, FOUNDATION_DIFFUSION_VOCAB_SIZE]
        );
        assert_eq!(
            output.spectrum_memory.dims(),
            &[2, config.spectrum.max_peaks, config.model_dim]
        );
        assert_eq!(output.spectrum_embedding.dims(), &[2, config.model_dim]);
        let value = foundation_diffusion_x0_loss(&output, &diffusion_batch)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(value.is_finite());
        assert!(value >= 0.0);
    }
}
