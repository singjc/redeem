//! High-level loading, embedding, and prediction API for foundation models.
//!
//! The existing ReDeeM property models expose wrapper types that own model
//! loading and prediction.  [`FoundationModelWrapper`] follows the same
//! ergonomic pattern while preserving the richer graph input required by the
//! foundation encoder.  Checkpoints use Candle [`candle_nn::VarMap`] names and
//! SafeTensors, matching the rest of ReDeeM's native model infrastructure.

use super::config::FoundationConfig;
use super::data::TrainingContext;
use super::featurize::{PeptideGraphFeaturizer, PeptidoformInput};
use super::model::{
    FoundationMultiTaskOutput, FoundationOutput, PeptideFoundationMultiTaskModel,
    PrecursorContextBatch,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;

/// High-level foundation model with checkpoint and inference helpers.
pub struct FoundationModelWrapper {
    varmap: VarMap,
    model: PeptideFoundationMultiTaskModel,
    featurizer: PeptideGraphFeaturizer,
    config: FoundationConfig,
    device: Device,
}

impl FoundationModelWrapper {
    /// Create a randomly initialized foundation model.
    pub fn new(config: FoundationConfig, device: Device) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideFoundationMultiTaskModel::new(config.clone(), vb)?;
        let featurizer = PeptideGraphFeaturizer::new(config.clone())?;
        Ok(Self {
            varmap,
            model,
            featurizer,
            config,
            device,
        })
    }

    /// Construct a model and load SafeTensors weights into its `VarMap`.
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        config: FoundationConfig,
        device: Device,
    ) -> Result<Self> {
        let mut wrapper = Self::new(config, device)?;
        wrapper.varmap.load(path)?;
        Ok(wrapper)
    }

    /// Save all named model variables in SafeTensors format.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path)
    }

    /// Generate intrinsic peptide/residue embeddings without acquisition heads.
    pub fn embed(&self, peptides: &[PeptidoformInput]) -> Result<FoundationOutput> {
        let batch = self.featurizer.featurize(peptides, &self.device)?;
        self.model.encoder().forward_t(&batch, false)
    }

    /// Predict RT/CCS/MS2 and self-supervised head outputs for peptides with
    /// explicit precursor/acquisition context.
    pub fn predict(
        &self,
        peptides: &[PeptidoformInput],
        context: &[TrainingContext],
    ) -> Result<FoundationMultiTaskOutput> {
        if peptides.len() != context.len() {
            candle_core::bail!(
                "foundation prediction requires one context row per peptide: {} peptides, {} contexts",
                peptides.len(),
                context.len()
            );
        }
        let batch = self.featurizer.featurize(peptides, &self.device)?;
        let context = context_batch(context, self.config.instrument_vocab_size, &self.device)?;
        self.model.forward_t(&batch, &context, false)
    }

    /// Foundation model configuration associated with this wrapper.
    pub fn config(&self) -> &FoundationConfig {
        &self.config
    }

    /// Candle device used by this wrapper.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Access the underlying multi-task model.
    pub fn model(&self) -> &PeptideFoundationMultiTaskModel {
        &self.model
    }

    /// Access the checkpoint variable map for optimizer construction.
    pub(crate) fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Mutable variable-map access used for checkpoint resume.
    pub(crate) fn varmap_mut(&mut self) -> &mut VarMap {
        &mut self.varmap
    }
}

pub(crate) fn context_batch(
    contexts: &[TrainingContext],
    instrument_vocab_size: usize,
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let charge: Vec<f32> = contexts
        .iter()
        .map(|context| context.charge.unwrap_or(0) as f32)
        .collect();
    let nce: Vec<f32> = contexts
        .iter()
        .map(|context| context.nce.unwrap_or(0.0))
        .collect();
    let instrument_ids: Vec<u32> = contexts
        .iter()
        .map(|context| {
            context
                .instrument_id
                .unwrap_or(0)
                .min(instrument_vocab_size.saturating_sub(1) as u32)
        })
        .collect();
    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(charge, contexts.len(), device)?,
        nce: Tensor::from_vec(nce, contexts.len(), device)?,
        instrument_ids: Tensor::from_vec(instrument_ids, contexts.len(), device)?
            .to_dtype(DType::U32)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_embeds_and_predicts_without_property_specific_models() {
        let config = FoundationConfig {
            max_sequence_len: 12,
            transformer_layers: 1,
            ..FoundationConfig::default()
        };
        let wrapper = FoundationModelWrapper::new(config.clone(), Device::Cpu).unwrap();
        let peptides = vec![PeptidoformInput::unmodified("PEPTIDEK")];
        let embedding = wrapper.embed(&peptides).unwrap();
        assert_eq!(embedding.peptide_embedding.dims(), &[1, config.model_dim]);
        let prediction = wrapper
            .predict(
                &peptides,
                &[TrainingContext {
                    charge: Some(2),
                    nce: Some(27.0),
                    ..TrainingContext::default()
                }],
            )
            .unwrap();
        assert_eq!(prediction.rt.dims(), &[1, 1]);
        assert_eq!(prediction.ccs.dims(), &[1, 1]);
    }
}
