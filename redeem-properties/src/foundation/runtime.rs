//! Trainable end-to-end foundation model and inference helpers.
//!
//! The model deliberately reuses the production peptide multi-task encoder and
//! spectrum-conditioned causal decoder already present in ReDeeM. A small
//! learned projection aligns both modalities in one contrastive space.

use super::causal::{
    foundation_causal_sequence_mean_nlls, FoundationCausalBatch, FoundationCausalInputBatch,
    FoundationCausalOutput, PeptideSpectrumCausalModel,
};
use super::collate::{FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig};
use super::config::{FoundationCcsContextMode, FoundationConfig, FoundationMs2OutputActivation};
use super::data::{FoundationTrainingRecord, RetentionTimeObjective};
use super::diffusion::{
    foundation_diffusion_token_residue, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_OPEN_CTERM_MOD,
    FOUNDATION_DIFFUSION_OPEN_NTERM_MOD, FOUNDATION_DIFFUSION_OPEN_RESIDUE_MOD,
    FOUNDATION_DIFFUSION_PAD, FOUNDATION_OPEN_PTM_MASS_SCALE_DA, FOUNDATION_OPEN_PTM_VOCAB_SIZE,
};
use super::featurize::PeptidoformInput;
use super::model::{
    FoundationMultiTaskOutput, PeptideFoundationMultiTaskModel, PrecursorContextBatch,
};
use super::normalization::FoundationTargetNormalizationConfig;
use super::pair_encoder::{PairTaskAuxiliaryPredictions, PairTaskPeptideModel};
use super::predictor::prediction_record;
use super::spectrum::{FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator};
use crate::models::model_interface::{PredictionInput, PredictionModel, PredictionOutput};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, Var};
use candle_nn::{self as nn, Linear, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const CHECKPOINT_SCHEMA: &str = "redeem.foundation.model";

/// Production peptide representation used by the end-to-end foundation model.
///
/// Missing values in checkpoints created before the pair-backbone restoration
/// deserialize as `ResidueTransformer`, preserving those checkpoints exactly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoundationPeptideBackbone {
    #[default]
    ResidueTransformer,
    ResiduePairTaskTokens,
}

fn legacy_peptide_backbone() -> FoundationPeptideBackbone {
    FoundationPeptideBackbone::ResidueTransformer
}

/// Native specialist heads distilled from the successful v0.51/v0.52 research
/// architecture.  These are ordinary trainable components of the production
/// model; they do not depend on historical teacher or parent checkpoints.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct FoundationSpecialistConfig {
    pub enabled: bool,
    pub rt: bool,
    pub ms2: bool,
    pub mobility_ccs: bool,
}

impl Default for FoundationSpecialistConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rt: true,
            ms2: true,
            mobility_ccs: true,
        }
    }
}

fn legacy_foundation_specialists() -> FoundationSpecialistConfig {
    FoundationSpecialistConfig {
        enabled: false,
        rt: false,
        ms2: false,
        mobility_ccs: false,
    }
}

/// One stable architecture config shared by training and inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FoundationModelConfig {
    /// Peptide encoder plus RT/CCS/MS2 and self-supervision heads.
    pub peptide: FoundationConfig,
    /// Peptide representation used by the production property model.
    #[serde(default = "legacy_peptide_backbone")]
    pub peptide_backbone: FoundationPeptideBackbone,
    /// Native best-of-research specialist heads. Historical checkpoints that
    /// predate these heads deserialize with them disabled.
    #[serde(default = "legacy_foundation_specialists")]
    pub specialists: FoundationSpecialistConfig,
    /// Adds supervised pair, chemistry-summary and conformation heads for new
    /// from-scratch models. Missing in older checkpoint metadata => disabled.
    #[serde(default)]
    pub auxiliary_supervision_heads: bool,
    /// Spectrum encoder plus autoregressive peptide decoder.
    pub inverse: FoundationDiffusionConfig,
}

impl Default for FoundationModelConfig {
    fn default() -> Self {
        // Preserve the deep widths/depth that survived the architecture search,
        // without preserving experiment-version types in the production API.
        let mut peptide = FoundationConfig::default();
        peptide.max_sequence_len = 64;
        peptide.max_atoms_per_residue = 24;
        peptide.graph_hidden_dim = 128;
        peptide.graph_layers = 5;
        peptide.model_dim = 320;
        peptide.num_attention_heads = 8;
        peptide.transformer_ff_dim = 1280;
        peptide.transformer_layers = 8;
        peptide.dropout = 0.05;
        peptide.contrastive_dim = 128;
        peptide.ms2_output_activation = FoundationMs2OutputActivation::SoftplusV0138;
        peptide.ccs_context_mode = FoundationCcsContextMode::NeutralMassCharge;
        peptide.ccs_physics_baseline = None;

        let mut inverse = FoundationDiffusionConfig::default();
        inverse.max_tokens = 80;
        inverse.model_dim = 320;
        inverse.num_attention_heads = 8;
        inverse.feed_forward_dim = 1280;
        inverse.spectrum_layers = 6;
        inverse.decoder_layers = 6;
        inverse.dropout = 0.05;

        Self {
            peptide,
            peptide_backbone: FoundationPeptideBackbone::ResiduePairTaskTokens,
            specialists: FoundationSpecialistConfig::default(),
            auxiliary_supervision_heads: true,
            inverse,
        }
    }
}

impl FoundationModelConfig {
    pub fn validate(&self) -> Result<()> {
        self.peptide.validate().map_err(anyhow::Error::msg)?;
        self.inverse.validate().map_err(anyhow::Error::msg)?;
        if self.peptide.instrument_vocab_size == 0 {
            anyhow::bail!("foundation instrument vocabulary must be non-zero");
        }
        Ok(())
    }
}

/// Serializable checkpoint metadata required for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundationCheckpointMetadata {
    pub schema: String,
    pub model: FoundationModelConfig,
    pub target_normalization: FoundationTargetNormalizationConfig,
    pub completed_epochs: usize,
    pub best_validation_loss: Option<f64>,
    pub corpus_fingerprint: Option<String>,
    pub benchmark_fingerprint: Option<String>,
    #[serde(default)]
    pub instrument_names: Vec<String>,
}

impl FoundationCheckpointMetadata {
    pub fn new(
        model: FoundationModelConfig,
        target_normalization: FoundationTargetNormalizationConfig,
        instrument_names: Vec<String>,
    ) -> Self {
        Self {
            schema: CHECKPOINT_SCHEMA.to_string(),
            model,
            target_normalization,
            completed_epochs: 0,
            best_validation_loss: None,
            corpus_fingerprint: None,
            benchmark_fingerprint: None,
            instrument_names,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.schema != CHECKPOINT_SCHEMA {
            anyhow::bail!(
                "unsupported foundation checkpoint schema {:?}; expected {:?}",
                self.schema,
                CHECKPOINT_SCHEMA
            );
        }
        self.model.validate()?;
        self.target_normalization.validate()?;
        if self.instrument_names.len() > self.model.peptide.instrument_vocab_size {
            anyhow::bail!(
                "foundation checkpoint has {} instrument names but vocabulary size is {}",
                self.instrument_names.len(),
                self.model.peptide.instrument_vocab_size
            );
        }
        Ok(())
    }
}

/// Per-record forward and inverse inference result.
#[derive(Debug, Clone)]
pub struct FoundationRecordPrediction {
    pub properties: PredictionOutput,
    /// Teacher-forced mean spectrum->peptide next-token NLL when a spectrum exists.
    pub inverse_mean_nll: Option<f32>,
    /// Greedy free generation from the observed spectrum when decoding succeeds.
    pub generated_peptidoform: Option<PeptidoformInput>,
}

/// One trainable end-to-end model.
///
/// The peptide and spectrum branches do not share raw weights; they share a
/// learned representation space through the alignment projection/loss. This is
/// intentional: the two modalities need different input encoders while still
/// being forced to describe the same peptidoform identity.
enum FoundationPeptideModel {
    ResidueTransformer(PeptideFoundationMultiTaskModel),
    ResiduePairTaskTokens(PairTaskPeptideModel),
}

impl FoundationPeptideModel {
    fn forward_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> candle_core::Result<FoundationMultiTaskOutput> {
        match self {
            Self::ResidueTransformer(model) => model.forward_t(batch, context, train),
            Self::ResiduePairTaskTokens(model) => model.forward_t(batch, context, train),
        }
    }

    fn forward_with_auxiliaries_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> candle_core::Result<(
        FoundationMultiTaskOutput,
        Option<PairTaskAuxiliaryPredictions>,
    )> {
        match self {
            Self::ResidueTransformer(model) => Ok((model.forward_t(batch, context, train)?, None)),
            Self::ResiduePairTaskTokens(model) => {
                model.forward_with_auxiliaries_t(batch, context, train)
            }
        }
    }
}

pub struct FoundationModel {
    config: FoundationModelConfig,
    device: Device,
    variables: VarMap,
    peptide: FoundationPeptideModel,
    inverse: PeptideSpectrumCausalModel,
    spectrum_alignment: Linear,
    peptide_collator: FoundationCollator,
    causal_collator: super::causal::FoundationCausalCollator,
    spectrum_collator: FoundationSpectrumCollator,
    normalization: FoundationTargetNormalizationConfig,
    instrument_names: Vec<String>,
}

impl FoundationModel {
    /// Randomly initialize the complete trainable model.
    pub fn new(
        config: FoundationModelConfig,
        normalization: FoundationTargetNormalizationConfig,
        instrument_names: Vec<String>,
        device: Device,
    ) -> Result<Self> {
        config.validate()?;
        normalization.validate()?;
        if instrument_names.len() > config.peptide.instrument_vocab_size {
            anyhow::bail!(
                "foundation instrument vocabulary has {} names but model capacity is {}",
                instrument_names.len(),
                config.peptide.instrument_vocab_size
            );
        }
        let variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let peptide = match config.peptide_backbone {
            FoundationPeptideBackbone::ResidueTransformer => {
                FoundationPeptideModel::ResidueTransformer(PeptideFoundationMultiTaskModel::new(
                    config.peptide.clone(),
                    vb.pp("peptide"),
                )?)
            }
            FoundationPeptideBackbone::ResiduePairTaskTokens => {
                FoundationPeptideModel::ResiduePairTaskTokens(PairTaskPeptideModel::new(
                    config.peptide.clone(),
                    config.specialists,
                    config.auxiliary_supervision_heads,
                    vb.pp("peptide"),
                )?)
            }
        };
        let inverse =
            PeptideSpectrumCausalModel::new_open_ptm(config.inverse.clone(), vb.pp("inverse"))?;
        let spectrum_alignment = nn::linear(
            config.inverse.model_dim,
            config.peptide.contrastive_dim,
            vb.pp("alignment.spectrum"),
        )?;
        let peptide_collator = FoundationCollator::new(
            config.peptide.clone(),
            FoundationCollatorConfig {
                retention_time_objective: RetentionTimeObjective::IntrinsicAndObserved,
                corruption: no_corruption(),
            },
        )?;
        let causal_collator =
            super::causal::FoundationCausalCollator::new_open_ptm(config.inverse.clone())?;
        let spectrum_collator = FoundationSpectrumCollator::new(config.inverse.spectrum.clone())?;
        Ok(Self {
            config,
            device,
            variables,
            peptide,
            inverse,
            spectrum_alignment,
            peptide_collator,
            causal_collator,
            spectrum_collator,
            normalization,
            instrument_names,
        })
    }

    /// Load one end-to-end checkpoint directory.
    pub fn load(checkpoint: impl AsRef<Path>, device: Device) -> Result<Self> {
        let checkpoint = checkpoint.as_ref();
        let metadata = read_metadata(checkpoint)?;
        let mut model = Self::new(
            metadata.model.clone(),
            metadata.target_normalization,
            metadata.instrument_names,
            device,
        )?;
        model
            .variables
            .load(checkpoint.join("model.safetensors"))
            .with_context(|| format!("load foundation weights from {}", checkpoint.display()))?;
        Ok(model)
    }

    pub fn config(&self) -> &FoundationModelConfig {
        &self.config
    }

    pub fn target_normalization(&self) -> FoundationTargetNormalizationConfig {
        self.normalization
    }

    pub fn instrument_names(&self) -> &[String] {
        &self.instrument_names
    }

    pub(crate) fn varmap(&self) -> &VarMap {
        &self.variables
    }

    pub(crate) fn trainable_variables(&self) -> Vec<Var> {
        self.variables
            .data()
            .lock()
            .expect("foundation VarMap lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Select trainable variables by stable production namespace prefixes.
    /// This supports staged curriculum updates without creating separate model
    /// instances or historical version-specific optimizers.
    pub(crate) fn trainable_variables_with_prefixes(&self, prefixes: &[&str]) -> Vec<Var> {
        self.variables
            .data()
            .lock()
            .expect("foundation VarMap lock poisoned")
            .iter()
            .filter(|(name, _)| prefixes.iter().any(|prefix| name.starts_with(prefix)))
            .map(|(_, var)| var.clone())
            .collect()
    }

    pub(crate) fn load_weights(&mut self, checkpoint: impl AsRef<Path>) -> Result<()> {
        let checkpoint = checkpoint.as_ref();
        if !checkpoint.is_dir() {
            anyhow::bail!(
                "foundation warm start requires a checkpoint directory containing metadata.yaml and model.safetensors: {}",
                checkpoint.display()
            );
        }
        let metadata = read_metadata(checkpoint)?;
        if metadata.model != self.config {
            anyhow::bail!("foundation warm-start architecture does not match the current model");
        }
        if metadata.instrument_names != self.instrument_names {
            anyhow::bail!(
                "foundation warm-start instrument vocabulary does not match the current corpus"
            );
        }
        let model_path = checkpoint.join("model.safetensors");
        self.variables
            .load(&model_path)
            .with_context(|| format!("load foundation initialization {}", model_path.display()))?;
        Ok(())
    }

    pub(crate) fn save_checkpoint(
        &self,
        checkpoint: impl AsRef<Path>,
        metadata: &FoundationCheckpointMetadata,
    ) -> Result<()> {
        let checkpoint = checkpoint.as_ref();
        fs::create_dir_all(checkpoint)
            .with_context(|| format!("create foundation checkpoint {}", checkpoint.display()))?;
        self.variables
            .save(checkpoint.join("model.safetensors"))
            .with_context(|| format!("save foundation weights to {}", checkpoint.display()))?;
        fs::write(
            checkpoint.join("metadata.yaml"),
            serde_yaml::to_string(metadata)?,
        )
        .with_context(|| format!("save foundation metadata to {}", checkpoint.display()))?;
        Ok(())
    }

    pub(crate) fn forward_properties_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> candle_core::Result<FoundationMultiTaskOutput> {
        self.peptide.forward_t(batch, context, train)
    }

    pub(crate) fn forward_properties_with_auxiliaries_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> candle_core::Result<(
        FoundationMultiTaskOutput,
        Option<PairTaskAuxiliaryPredictions>,
    )> {
        self.peptide
            .forward_with_auxiliaries_t(batch, context, train)
    }

    pub(crate) fn inverse_forward_t(
        &self,
        batch: &FoundationCausalBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        train: bool,
    ) -> candle_core::Result<FoundationCausalOutput> {
        self.inverse
            .forward_t(&batch.input, spectrum, precursor, train)
    }

    pub(crate) fn inverse_mass_loss(
        &self,
        output: &FoundationCausalOutput,
        batch: &FoundationCausalBatch,
    ) -> candle_core::Result<Tensor> {
        self.inverse.open_ptm_mass_loss(output, batch)
    }

    pub(crate) fn project_spectrum(&self, embedding: &Tensor) -> candle_core::Result<Tensor> {
        self.spectrum_alignment.forward(embedding)
    }

    pub(crate) fn causal_collator(&self) -> &super::causal::FoundationCausalCollator {
        &self.causal_collator
    }

    pub(crate) fn spectrum_collator(&self) -> &FoundationSpectrumCollator {
        &self.spectrum_collator
    }

    pub(crate) fn peptide_collator(&self) -> &FoundationCollator {
        &self.peptide_collator
    }

    /// Forward prediction for materialized foundation records.
    pub fn predict_records(
        &self,
        records: &[FoundationTrainingRecord],
    ) -> Result<Vec<PredictionOutput>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let batch = self.peptide_collator.collate(records, &self.device, 0)?;
        let output = self.forward_properties_t(&batch.input, &batch.context, false)?;
        outputs_to_predictions(&output, self.normalization, records.len())
    }

    /// Forward + inverse inference for materialized records.
    pub fn infer_records(
        &self,
        records: &[FoundationTrainingRecord],
    ) -> Result<Vec<FoundationRecordPrediction>> {
        let properties = self.predict_records(records)?;
        let mut results = Vec::with_capacity(records.len());
        for (record, properties) in records.iter().zip(properties) {
            let inverse_mean_nll = self.inverse_mean_nll(record)?;
            let generated_peptidoform =
                if FoundationSpectrum::from_training_record(record).is_some() {
                    self.generate_peptide(record).ok()
                } else {
                    None
                };
            results.push(FoundationRecordPrediction {
                properties,
                inverse_mean_nll,
                generated_peptidoform,
            });
        }
        Ok(results)
    }

    /// Teacher-forced inverse score for one known spectrum/peptidoform pair.
    pub fn inverse_mean_nll(&self, record: &FoundationTrainingRecord) -> Result<Option<f32>> {
        let Some(spectrum) = FoundationSpectrum::from_training_record(record) else {
            return Ok(None);
        };
        let causal = self
            .causal_collator
            .collate(&[record.peptidoform.clone()], &self.device)?;
        let spectrum = self.spectrum_collator.collate(&[spectrum], &self.device)?;
        let peptide =
            self.peptide_collator
                .collate(std::slice::from_ref(record), &self.device, 0)?;
        let output = self.inverse_forward_t(&causal, &spectrum, &peptide.context, false)?;
        let losses = foundation_causal_sequence_mean_nlls(&output, &causal)?;
        Ok(Some(losses.to_vec1::<f32>()?[0]))
    }

    /// Greedy free generation from an observed spectrum.
    ///
    /// This is intentionally the minimal production decoder. Search/reranking
    /// callers normally use teacher-forced candidate NLL; beam search should only
    /// be added if free-generation evaluation demonstrates that greedy decoding is
    /// the actual bottleneck.
    pub fn generate_peptide(&self, record: &FoundationTrainingRecord) -> Result<PeptidoformInput> {
        let spectrum = FoundationSpectrum::from_training_record(record).ok_or_else(|| {
            anyhow::anyhow!("foundation inverse generation requires an observed spectrum")
        })?;
        let spectrum = self.spectrum_collator.collate(&[spectrum], &self.device)?;
        let peptide =
            self.peptide_collator
                .collate(std::slice::from_ref(record), &self.device, 0)?;
        let context = self
            .inverse
            .prepare_context(&spectrum, &peptide.context, false)?;
        let vocabulary = FoundationDiffusionVocabulary;
        let mut tokens = Vec::<u32>::new();
        let mut masses = Vec::<f32>::new();
        let mut saw_residue = false;
        let mut saw_cterm = false;

        for _ in 0..self.config.inverse.max_tokens {
            let input = compact_open_prefix(&tokens, &masses, &self.device)?;
            let output = self
                .inverse
                .forward_t_with_context(&input, &context, false)?;
            let (_, width, classes) = output.token_logits.dims3()?;
            let logits = output
                .token_logits
                .narrow(1, width - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?
                .to_vec1::<f32>()?;
            if classes != FOUNDATION_OPEN_PTM_VOCAB_SIZE || logits.len() != classes {
                anyhow::bail!("foundation inverse decoder vocabulary shape mismatch");
            }
            let next = best_valid_token(&logits, saw_residue, saw_cterm).ok_or_else(|| {
                anyhow::anyhow!("foundation inverse decoder produced no valid next token")
            })?;
            if next == FOUNDATION_DIFFUSION_EOS {
                tokens.push(next);
                masses.push(0.0);
                return vocabulary
                    .decode_open_ptm(&tokens, &masses, FOUNDATION_OPEN_PTM_MASS_SCALE_DA)
                    .map_err(anyhow::Error::msg);
            }

            let mass = if matches!(
                next,
                FOUNDATION_DIFFUSION_OPEN_NTERM_MOD
                    | FOUNDATION_DIFFUSION_OPEN_RESIDUE_MOD
                    | FOUNDATION_DIFFUSION_OPEN_CTERM_MOD
            ) {
                self.inverse
                    .open_ptm_mass_prediction(&output)?
                    .ok_or_else(|| anyhow::anyhow!("open-PTM decoder is missing its mass head"))?
                    .narrow(1, width - 1, 1)?
                    .squeeze(1)?
                    .squeeze(0)?
                    .to_scalar::<f32>()?
            } else {
                0.0
            };

            if foundation_diffusion_token_residue(next).is_some() {
                saw_residue = true;
            }
            if next == FOUNDATION_DIFFUSION_OPEN_CTERM_MOD {
                saw_cterm = true;
            }
            tokens.push(next);
            masses.push(mass);
        }
        anyhow::bail!("foundation inverse generation reached max_tokens without EOS")
    }
}

impl PredictionModel for FoundationModel {
    fn predict_batch(&self, inputs: &[PredictionInput]) -> Result<Vec<PredictionOutput>> {
        let records = inputs
            .iter()
            .map(|input| {
                let mut resolved = input.clone();
                if resolved.instrument_id.is_none() {
                    if let Some(name) = resolved.instrument_name.as_deref() {
                        resolved.instrument_id = self
                            .instrument_names
                            .iter()
                            .position(|known| known.eq_ignore_ascii_case(name))
                            .map(|index| index as u32);
                    }
                }
                prediction_record(&resolved)
            })
            .collect::<Result<Vec<_>>>()?;
        self.predict_records(&records)
    }
}

pub fn read_foundation_checkpoint_metadata(
    checkpoint: impl AsRef<Path>,
) -> Result<FoundationCheckpointMetadata> {
    read_metadata(checkpoint.as_ref())
}

fn read_metadata(checkpoint: &Path) -> Result<FoundationCheckpointMetadata> {
    let path = checkpoint.join("metadata.yaml");
    let metadata: FoundationCheckpointMetadata = serde_yaml::from_str(
        &fs::read_to_string(&path)
            .with_context(|| format!("read foundation metadata {}", path.display()))?,
    )
    .with_context(|| format!("parse foundation metadata {}", path.display()))?;
    metadata.validate()?;
    Ok(metadata)
}

fn outputs_to_predictions(
    output: &FoundationMultiTaskOutput,
    normalization: FoundationTargetNormalizationConfig,
    expected: usize,
) -> Result<Vec<PredictionOutput>> {
    let rt = normalization
        .rt
        .denormalize_tensor(&output.rt)?
        .squeeze(1)?
        .to_vec1::<f32>()?;
    let ccs = normalization
        .ccs
        .denormalize_tensor(&output.ccs)?
        .squeeze(1)?
        .to_vec1::<f32>()?;
    let ms2 = output.ms2.to_vec3::<f32>()?;
    if rt.len() != expected || ccs.len() != expected || ms2.len() != expected {
        anyhow::bail!("foundation prediction batch length mismatch");
    }
    Ok((0..expected)
        .map(|index| PredictionOutput {
            rt: Some(rt[index]),
            ccs: Some(ccs[index]),
            ms2: Some(ms2[index].clone()),
        })
        .collect())
}

fn compact_open_prefix(
    tokens: &[u32],
    masses: &[f32],
    device: &Device,
) -> Result<FoundationCausalInputBatch> {
    if tokens.len() != masses.len() {
        anyhow::bail!("foundation inverse prefix token/mass lengths differ");
    }
    let width = tokens.len() + 1;
    let mut shifted = vec![FOUNDATION_DIFFUSION_PAD; width];
    let mut modification_features = vec![0.0f32; width * 2];
    for (position, (&token, &mass)) in tokens.iter().zip(masses).enumerate() {
        shifted[position + 1] = token;
        modification_features[(position + 1) * 2] = mass;
        modification_features[(position + 1) * 2 + 1] = if matches!(
            token,
            FOUNDATION_DIFFUSION_OPEN_NTERM_MOD
                | FOUNDATION_DIFFUSION_OPEN_RESIDUE_MOD
                | FOUNDATION_DIFFUSION_OPEN_CTERM_MOD
        ) {
            1.0
        } else {
            0.0
        };
    }
    Ok(FoundationCausalInputBatch {
        input_tokens: Tensor::from_vec(shifted, (1, width), device)?.to_dtype(DType::U32)?,
        token_mask: Tensor::ones((1, width), DType::F32, device)?,
        modification_features: Tensor::from_vec(modification_features, (1, width, 2), device)?,
    })
}

fn best_valid_token(logits: &[f32], saw_residue: bool, saw_cterm: bool) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .filter(|(token, value)| {
            value.is_finite() && token_allowed(*token as u32, saw_residue, saw_cterm)
        })
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(token, _)| token as u32)
}

fn token_allowed(token: u32, saw_residue: bool, saw_cterm: bool) -> bool {
    if token == FOUNDATION_DIFFUSION_PAD || token == FOUNDATION_DIFFUSION_MASK {
        return false;
    }
    if token == FOUNDATION_DIFFUSION_EOS {
        return saw_residue;
    }
    if token == FOUNDATION_DIFFUSION_OPEN_NTERM_MOD {
        return !saw_residue && !saw_cterm;
    }
    if token == FOUNDATION_DIFFUSION_OPEN_CTERM_MOD {
        return saw_residue;
    }
    if token == FOUNDATION_DIFFUSION_OPEN_RESIDUE_MOD {
        return saw_residue && !saw_cterm;
    }
    foundation_diffusion_token_residue(token).is_some() && !saw_cterm
}

fn no_corruption() -> FoundationCorruptionConfig {
    FoundationCorruptionConfig {
        residue_mask_probability: 0.0,
        chemistry_mask_probability: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_deep_and_cross_modal_dimensions_are_valid() {
        let config = FoundationModelConfig::default();
        config.validate().unwrap();
        assert_eq!(config.peptide.model_dim, 320);
        assert_eq!(config.peptide.transformer_layers, 8);
        assert_eq!(
            config.peptide_backbone,
            FoundationPeptideBackbone::ResiduePairTaskTokens
        );
        assert_eq!(config.inverse.model_dim, 320);
        assert_eq!(config.inverse.decoder_layers, 6);
    }

    #[test]
    fn historical_model_config_without_backbone_defaults_to_residue_transformer() {
        let mut value = serde_yaml::to_value(FoundationModelConfig::default()).unwrap();
        value
            .as_mapping_mut()
            .unwrap()
            .remove(&serde_yaml::Value::String("peptide_backbone".into()));
        let restored: FoundationModelConfig = serde_yaml::from_value(value).unwrap();
        assert_eq!(
            restored.peptide_backbone,
            FoundationPeptideBackbone::ResidueTransformer
        );
    }

    #[test]
    fn historical_model_config_without_specialists_keeps_old_parameter_tree() {
        let mut value = serde_yaml::to_value(FoundationModelConfig::default()).unwrap();
        value
            .as_mapping_mut()
            .unwrap()
            .remove(&serde_yaml::Value::String("specialists".into()));
        let restored: FoundationModelConfig = serde_yaml::from_value(value).unwrap();
        assert!(!restored.specialists.enabled);
        assert!(!restored.specialists.rt);
        assert!(!restored.specialists.ms2);
        assert!(!restored.specialists.mobility_ccs);
    }

    #[test]
    fn missing_auxiliary_heads_preserves_older_checkpoint_parameter_tree() {
        let defaults = FoundationModelConfig::default();
        assert!(defaults.auxiliary_supervision_heads);
        let mut value = serde_yaml::to_value(defaults).unwrap();
        value
            .as_mapping_mut()
            .unwrap()
            .remove(&serde_yaml::Value::String(
                "auxiliary_supervision_heads".into(),
            ));
        let legacy: FoundationModelConfig = serde_yaml::from_value(value).unwrap();
        assert!(!legacy.auxiliary_supervision_heads);
    }

    #[test]
    fn greedy_token_filter_enforces_basic_peptide_grammar() {
        assert!(!token_allowed(FOUNDATION_DIFFUSION_EOS, false, false));
        assert!(token_allowed(
            FOUNDATION_DIFFUSION_OPEN_NTERM_MOD,
            false,
            false
        ));
        assert!(!token_allowed(
            FOUNDATION_DIFFUSION_OPEN_NTERM_MOD,
            true,
            false
        ));
        assert!(token_allowed(FOUNDATION_DIFFUSION_EOS, true, false));
        assert!(!token_allowed(3, true, true));
    }
}
