//! v0.29 sequence-level reward post-training wrapper.
//!
//! v0.29 adds no model parameters. It preserves the complete frozen v0.27
//! architecture and changes only the inverse causal optimization objective:
//! same-spectrum mass-constrained candidates receive deterministic sequence +
//! fragment + mass rewards, optimized with group-relative advantages while a
//! frozen v0.27 reference policy anchors sequence likelihood.

use super::causal::PeptideSpectrumCausalModel;
use super::config::FoundationConfig;
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
};
use super::model::{FoundationOutput, PeptideFoundationEncoder, PrecursorContextBatch};
use super::multimodal_v0270::{
    foundation_multimodal_ms2_loss_v0270, foundation_multimodal_relation_margin_loss_v0270,
    FoundationFragmentContextBatchV0270, FoundationMultimodalForwardOutputV0270,
    FoundationMultimodalMs2LossesV0270, PeptideFoundationMultimodalV0270Config,
    PeptideFoundationMultimodalV0270Model, FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270,
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270,
};
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

pub const FOUNDATION_INVERSE_REWARD_ARCHITECTURE_V0290: &str =
    "v0.29-v0270-causal-group-relative-physics-reward-posttraining";
pub const FOUNDATION_INVERSE_REWARD_BEAM_WIDTH_V0290: usize = 16;
pub const FOUNDATION_INVERSE_REWARD_TOP_K_V0290: usize = 6;
pub const FOUNDATION_INVERSE_REWARD_GROUPS_PER_STEP_V0290: usize = 2;
pub const FOUNDATION_INVERSE_REWARD_SUPERVISED_WEIGHT_V0290: f64 = 0.50;
pub const FOUNDATION_INVERSE_REWARD_POLICY_WEIGHT_V0290: f64 = 1.00;
pub const FOUNDATION_INVERSE_REWARD_REFERENCE_WEIGHT_V0290: f64 = 0.05;
pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0290: usize =
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270;
pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0290: f64 =
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationInverseRewardV0290Config {
    pub base_v0270: PeptideFoundationMultimodalV0270Config,
    pub beam_width: usize,
    pub top_k: usize,
    pub groups_per_step: usize,
    pub supervised_weight: f64,
    pub policy_weight: f64,
    pub reference_weight: f64,
}

impl PeptideFoundationInverseRewardV0290Config {
    pub fn fixed(base_v0270: PeptideFoundationMultimodalV0270Config) -> Result<Self> {
        base_v0270.validate()?;
        let config = Self {
            base_v0270,
            beam_width: FOUNDATION_INVERSE_REWARD_BEAM_WIDTH_V0290,
            top_k: FOUNDATION_INVERSE_REWARD_TOP_K_V0290,
            groups_per_step: FOUNDATION_INVERSE_REWARD_GROUPS_PER_STEP_V0290,
            supervised_weight: FOUNDATION_INVERSE_REWARD_SUPERVISED_WEIGHT_V0290,
            policy_weight: FOUNDATION_INVERSE_REWARD_POLICY_WEIGHT_V0290,
            reference_weight: FOUNDATION_INVERSE_REWARD_REFERENCE_WEIGHT_V0290,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.base_v0270.validate()?;
        if self.beam_width != FOUNDATION_INVERSE_REWARD_BEAM_WIDTH_V0290
            || self.top_k != FOUNDATION_INVERSE_REWARD_TOP_K_V0290
            || self.groups_per_step != FOUNDATION_INVERSE_REWARD_GROUPS_PER_STEP_V0290
        {
            candle_core::bail!("v0.29 search/group dimensions differ from the fixed experiment");
        }
        if (self.supervised_weight - FOUNDATION_INVERSE_REWARD_SUPERVISED_WEIGHT_V0290).abs()
            > f64::EPSILON
            || (self.policy_weight - FOUNDATION_INVERSE_REWARD_POLICY_WEIGHT_V0290).abs()
                > f64::EPSILON
            || (self.reference_weight - FOUNDATION_INVERSE_REWARD_REFERENCE_WEIGHT_V0290).abs()
                > f64::EPSILON
        {
            candle_core::bail!("v0.29 reward loss weights differ from the fixed experiment");
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
pub struct PeptideFoundationInverseRewardV0290Model {
    base_v0270: PeptideFoundationMultimodalV0270Model,
    config: PeptideFoundationInverseRewardV0290Config,
}

impl PeptideFoundationInverseRewardV0290Model {
    pub fn new(
        config: PeptideFoundationInverseRewardV0290Config,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        config.validate()?;
        let base_v0270 = PeptideFoundationMultimodalV0270Model::new(config.base_v0270.clone(), vb)?;
        Ok(Self { base_v0270, config })
    }

    pub fn forward_v0290_t_with_shared_gradient_scales(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
        rt_encoder_gradient_scale: f64,
        ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultimodalForwardOutputV0270> {
        self.base_v0270
            .forward()
            .forward_v0270_t_with_shared_gradient_scales(
                batch,
                context,
                fragment,
                train,
                rt_encoder_gradient_scale,
                ccs_encoder_gradient_scale,
            )
    }

    pub fn forward_v0290_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0270> {
        self.base_v0270
            .forward()
            .forward_v0270_t(batch, context, fragment, train)
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

    pub fn encoder(&self) -> &PeptideFoundationEncoder {
        self.base_v0270.forward().encoder()
    }

    pub fn forward_config(&self) -> &FoundationConfig {
        self.config.forward()
    }

    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        self.config.inverse()
    }

    pub fn config(&self) -> &PeptideFoundationInverseRewardV0290Config {
        &self.config
    }
}

pub type FoundationFragmentContextBatchV0290 = FoundationFragmentContextBatchV0270;
pub type FoundationMultimodalMs2LossesV0290 = FoundationMultimodalMs2LossesV0270;

pub fn foundation_multimodal_ms2_loss_v0290(
    output: &FoundationMultimodalForwardOutputV0270,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    presence_target: Option<&Tensor>,
    presence_mask: Option<&Tensor>,
) -> Result<FoundationMultimodalMs2LossesV0290> {
    foundation_multimodal_ms2_loss_v0270(output, target, mask, presence_target, presence_mask)
}

pub fn foundation_multimodal_relation_margin_loss_v0290(
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
    use std::collections::BTreeSet;

    fn base_config() -> PeptideFoundationMultimodalV0270Config {
        let mut forward = FoundationConfig::default();
        forward.model_dim = 192;
        forward.num_attention_heads = 6;
        forward.transformer_ff_dim = 768;
        let mut inverse = FoundationDiffusionConfig::default();
        inverse.model_dim = 192;
        inverse.num_attention_heads = 6;
        inverse.feed_forward_dim = 768;
        PeptideFoundationMultimodalV0270Config::fixed(forward, inverse).unwrap()
    }

    #[test]
    fn v0290_configuration_is_fixed() {
        let config = PeptideFoundationInverseRewardV0290Config::fixed(base_config()).unwrap();
        assert_eq!(config.beam_width, 16);
        assert_eq!(config.top_k, 6);
        assert_eq!(config.groups_per_step, 2);
        assert_eq!(config.supervised_weight, 0.50);
        assert_eq!(config.policy_weight, 1.00);
        assert_eq!(config.reference_weight, 0.05);
    }

    #[test]
    fn v0290_adds_no_model_parameters_to_v0270() {
        let base = base_config();
        let base_varmap = VarMap::new();
        let _base_model = PeptideFoundationMultimodalV0270Model::new(
            base.clone(),
            VarBuilder::from_varmap(&base_varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let reward_varmap = VarMap::new();
        let _reward_model = PeptideFoundationInverseRewardV0290Model::new(
            PeptideFoundationInverseRewardV0290Config::fixed(base).unwrap(),
            VarBuilder::from_varmap(&reward_varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let base_names = base_varmap
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let reward_names = reward_varmap
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        assert_eq!(reward_names, base_names);
    }
}
