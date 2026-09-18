//! v0.30 fragment-aware shared-representation continuation.
//!
//! v0.30 keeps the accepted v0.27 RT specialist, CCS physics/residual path,
//! contextual fragment decoder, inverse models, spectrum encoder, alignment,
//! and same-spectrum relation objective unchanged. The single architectural
//! addition is a low-capacity cleavage-local auxiliary head that predicts MS2
//! intensities directly from adjacent **shared encoder residue states** plus
//! minimal acquisition context. The auxiliary is used only during training.
//!
//! This creates an explicit gradient path from observed fragment evidence into
//! the shared residue representation without replacing or widening the proven
//! v0.27 property heads.

use super::causal::PeptideSpectrumCausalModel;
use super::config::FoundationConfig;
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
};
use super::loss::{foundation_ms2_loss, FoundationMs2LossConfig, FoundationMs2Losses};
use super::model::{
    apply_ms2_output_activation, FoundationMultiTaskOutput, FoundationOutput, PrecursorContextBatch,
};
use super::multimodal_v0270::{
    foundation_multimodal_ms2_loss_v0270, foundation_multimodal_relation_margin_loss_v0270,
    FoundationFragmentContextBatchV0270, FoundationMultimodalForwardOutputV0270,
    FoundationMultimodalMs2LossesV0270, PeptideFoundationMultimodalV0270Config,
    PeptideFoundationMultimodalV0270Model, FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270,
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270,
};
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

/// Stable v0.30 architecture identifier.
pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0300: &str =
    "v0.30-192d-fragment-aware-shared-residue-aux-v027-heads-openptm32";
/// Hidden width of the low-capacity cleavage-local auxiliary.
pub const FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0300: usize = 192;
/// Scalar acquisition features appended to every cleavage: charge, charge-present,
/// NCE, NCE-present.
pub const FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0300: usize = 4;
/// Fixed training-only auxiliary weight. This is intentionally small because the
/// accepted v0.27 contextual decoder remains the authoritative MS2 output path.
pub const FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0300: f64 = 0.05;
/// v0.30 preserves the accepted v0.27 property hidden width.
pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0300: usize =
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270;
/// v0.30 preserves the accepted same-spectrum relation margin.
pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0300: f64 =
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270;

/// Complete fixed v0.30 architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationMultimodalV0300Config {
    /// Entire accepted v0.27 architecture.
    pub base_v0270: PeptideFoundationMultimodalV0270Config,
    /// Cleavage-local auxiliary hidden width (fixed to 192).
    pub fragment_aux_hidden: usize,
    /// Fixed auxiliary loss weight.
    pub fragment_aux_weight: f64,
}

impl PeptideFoundationMultimodalV0300Config {
    /// Construct the single predeclared v0.30 architecture.
    pub fn fixed(base_v0270: PeptideFoundationMultimodalV0270Config) -> Result<Self> {
        base_v0270.validate()?;
        let config = Self {
            base_v0270,
            fragment_aux_hidden: FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0300,
            fragment_aux_weight: FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0300,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate that v0.30 has not silently become a width/loss sweep.
    pub fn validate(&self) -> Result<()> {
        self.base_v0270.validate()?;
        if self.base_v0270.forward.model_dim != 192 {
            candle_core::bail!("v0.30 is fixed at model_dim=192");
        }
        if self.fragment_aux_hidden != FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0300 {
            candle_core::bail!("v0.30 fragment auxiliary width differs from the frozen design");
        }
        if (self.fragment_aux_weight - FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0300).abs()
            > f64::EPSILON
        {
            candle_core::bail!("v0.30 fragment auxiliary weight differs from the frozen design");
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
struct FragmentRepresentationAuxV0300 {
    hidden: Linear,
    output: Linear,
    model_dim: usize,
    channels: usize,
}

impl FragmentRepresentationAuxV0300 {
    fn new(config: &PeptideFoundationMultimodalV0300Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        let channels = config.forward().ms2_fragment_channels;
        let input_dim = 2 * model_dim + FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0300;
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
        activation: super::config::FoundationMs2OutputActivation,
    ) -> Result<Tensor> {
        let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
        if model_dim != self.model_dim || sequence < 2 {
            candle_core::bail!(
                "v0.30 fragment auxiliary expected residue width {} and sequence>=2, got {:?}",
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
                FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0300,
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
        let logits = self
            .output
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

/// v0.30 output: accepted v0.27 predictions plus a training-only local fragment auxiliary.
#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0300 {
    /// Backward-compatible v0.27 multi-task surface.
    pub base: FoundationMultiTaskOutput,
    /// Accepted v0.27 fragment-presence logits.
    pub ms2_presence_logits: Tensor,
    /// Accepted v0.27 positive conditional fragment intensities.
    pub ms2_positive_intensity: Tensor,
    /// Training-only direct cleavage-local prediction from shared residue states.
    pub fragment_representation_aux: Tensor,
}

/// Complete v0.30 model.
#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0300Model {
    base_v0270: PeptideFoundationMultimodalV0270Model,
    fragment_aux: FragmentRepresentationAuxV0300,
    config: PeptideFoundationMultimodalV0300Config,
}

impl PeptideFoundationMultimodalV0300Model {
    pub fn new(config: PeptideFoundationMultimodalV0300Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let base_v0270 =
            PeptideFoundationMultimodalV0270Model::new(config.base_v0270.clone(), vb.clone())?;
        let fragment_aux =
            FragmentRepresentationAuxV0300::new(&config, vb.pp("fragment_representation_aux"))?;
        Ok(Self {
            base_v0270,
            fragment_aux,
            config,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_v0300_t_with_shared_gradient_scales(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
        rt_encoder_gradient_scale: f64,
        ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultimodalForwardOutputV0300> {
        let base_v0270 = self
            .base_v0270
            .forward()
            .forward_v0270_t_with_shared_gradient_scales(
                batch,
                context,
                fragment,
                train,
                rt_encoder_gradient_scale,
                ccs_encoder_gradient_scale,
            )?;
        let fragment_representation_aux = self.fragment_aux.forward(
            &base_v0270.base.foundation,
            context,
            self.config.forward().ms2_output_activation,
        )?;
        Ok(FoundationMultimodalForwardOutputV0300 {
            base: base_v0270.base,
            ms2_presence_logits: base_v0270.ms2_presence_logits,
            ms2_positive_intensity: base_v0270.ms2_positive_intensity,
            fragment_representation_aux,
        })
    }

    pub fn forward_v0300_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0300> {
        self.forward_v0300_t_with_shared_gradient_scales(batch, context, fragment, train, 1.0, 1.0)
    }

    pub fn peptide_projection_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<Tensor> {
        self.base_v0270.forward().peptide_projection_t(batch, train)
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

    pub fn encoder(&self) -> &super::model::PeptideFoundationEncoder {
        self.base_v0270.forward().encoder()
    }

    pub fn forward_config(&self) -> &FoundationConfig {
        self.config.forward()
    }

    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        self.config.inverse()
    }

    pub fn config(&self) -> &PeptideFoundationMultimodalV0300Config {
        &self.config
    }
}

pub type FoundationFragmentContextBatchV0300 = FoundationFragmentContextBatchV0270;
pub type FoundationMultimodalMs2LossesV0300 = FoundationMultimodalMs2LossesV0270;

pub fn foundation_multimodal_ms2_loss_v0300(
    output: &FoundationMultimodalForwardOutputV0300,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    presence_target: Option<&Tensor>,
    presence_mask: Option<&Tensor>,
) -> Result<FoundationMultimodalMs2LossesV0300> {
    let proxy = FoundationMultimodalForwardOutputV0270 {
        base: output.base.clone(),
        ms2_presence_logits: output.ms2_presence_logits.clone(),
        ms2_positive_intensity: output.ms2_positive_intensity.clone(),
    };
    foundation_multimodal_ms2_loss_v0270(&proxy, target, mask, presence_target, presence_mask)
}

pub fn foundation_fragment_representation_aux_loss_v0300(
    output: &FoundationMultimodalForwardOutputV0300,
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
        _ => candle_core::bail!("v0.30 fragment auxiliary target/mask must be supplied together"),
    }
}

pub fn foundation_multimodal_relation_margin_loss_v0300(
    positive_score: &Tensor,
    negative_score: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    foundation_multimodal_relation_margin_loss_v0270(positive_score, negative_score, margin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn v0300_config_is_fixed_and_keeps_v0270_base() {
        let mut forward = FoundationConfig::default();
        forward.model_dim = 192;
        forward.num_attention_heads = 6;
        forward.transformer_ff_dim = 768;
        let mut inverse = FoundationDiffusionConfig::default();
        inverse.model_dim = 192;
        inverse.num_attention_heads = 6;
        inverse.feed_forward_dim = 768;
        let base = PeptideFoundationMultimodalV0270Config::fixed(forward, inverse).unwrap();
        let config = PeptideFoundationMultimodalV0300Config::fixed(base.clone()).unwrap();
        assert_eq!(config.base_v0270, base);
        assert_eq!(config.fragment_aux_hidden, 192);
        assert_eq!(config.fragment_aux_weight, 0.05);
    }

    #[test]
    fn v0300_model_constructs_auxiliary_without_replacing_v0270_namespaces() {
        let mut forward = FoundationConfig::default();
        forward.model_dim = 192;
        forward.num_attention_heads = 6;
        forward.transformer_ff_dim = 768;
        let mut inverse = FoundationDiffusionConfig::default();
        inverse.model_dim = 192;
        inverse.num_attention_heads = 6;
        inverse.feed_forward_dim = 768;
        let base = PeptideFoundationMultimodalV0270Config::fixed(forward, inverse).unwrap();
        let config = PeptideFoundationMultimodalV0300Config::fixed(base).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let _ = PeptideFoundationMultimodalV0300Model::new(config, vb).unwrap();
        let names = varmap
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert!(names.iter().any(|name| name.starts_with("encoder.")));
        assert!(names
            .iter()
            .any(|name| name.starts_with("fragment_decoder.")));
        assert!(names
            .iter()
            .any(|name| name.starts_with("fragment_representation_aux.")));
    }
}
