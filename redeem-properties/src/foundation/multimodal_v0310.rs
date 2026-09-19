//! v0.31 protected-CCS fragment-aware property-view continuation.
//!
//! v0.30 demonstrated that fragment-aware residue supervision can improve RT and
//! MS2 on DEV and TRAIN-HOLDOUT, but the same shared-encoder update materially
//! regressed CCS. v0.31 therefore keeps the complete frozen v0.27 encoder + CCS
//! path as an immutable base view and learns a separate residue-level property
//! refinement used by RT/MS2/self-supervision. The refinement is identity at
//! initialization through a zero-initialized residual projection, so step 0
//! reproduces v0.27 exactly for all accepted property heads.
//!
//! This is deliberately not the closed v0.28 task-conditioned low-rank adapter:
//! there are no task embeddings or post-pooled task adapters. The new branch is
//! one residue Transformer refinement shared by RT/MS2 and trained with the same
//! cleavage-local fragment auxiliary that produced the useful v0.30 signal.

use super::causal::PeptideSpectrumCausalModel;
use super::config::FoundationConfig;
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
};
use super::layers::{FoundationLayerNorm, PeptideTransformerBlock};
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
            candle_core::bail!("v0.31 dimensions/auxiliary weight differ from the fixed design");
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
    fn new(config: &PeptideFoundationMultimodalV0310Config, vb: VarBuilder<'_>) -> Result<Self> {
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
    fn new(config: &PeptideFoundationMultimodalV0310Config, vb: VarBuilder<'_>) -> Result<Self> {
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
        activation: super::config::FoundationMs2OutputActivation,
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
    pub fn new(config: PeptideFoundationMultimodalV0310Config, vb: VarBuilder<'_>) -> Result<Self> {
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
        batch: &super::featurize::FoundationBatch,
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
        batch: &super::featurize::FoundationBatch,
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
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0310> {
        self.forward_v0310_t_with_shared_gradient_scales(batch, context, fragment, train, 1.0, 0.0)
    }

    pub fn peptide_projection_t(
        &self,
        batch: &super::featurize::FoundationBatch,
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
        _ => candle_core::bail!("v0.31 fragment auxiliary target/mask must be supplied together"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn config() -> PeptideFoundationMultimodalV0310Config {
        let mut forward = FoundationConfig::default();
        forward.max_sequence_len = 8;
        forward.max_atoms_per_residue = 24;
        forward.graph_hidden_dim = 32;
        forward.graph_layers = 1;
        forward.model_dim = 192;
        forward.num_attention_heads = 6;
        forward.transformer_layers = 1;
        forward.transformer_ff_dim = 768;
        forward.ms2_fragment_channels = 8;
        let mut inverse = FoundationDiffusionConfig::default();
        inverse.max_tokens = 8;
        inverse.model_dim = 192;
        inverse.num_attention_heads = 6;
        inverse.feed_forward_dim = 768;
        inverse.spectrum_layers = 1;
        inverse.decoder_layers = 1;
        PeptideFoundationMultimodalV0310Config::fixed(
            PeptideFoundationMultimodalV0270Config::fixed(forward, inverse).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn v0310_config_is_fixed_and_protects_ccs_by_design() {
        let config = config();
        assert_eq!(config.property_refinement_layers, 1);
        assert_eq!(config.property_refinement_heads, 6);
        assert_eq!(config.property_refinement_ff_dim, 768);
        assert_eq!(config.fragment_aux_weight, 0.05);
    }

    #[test]
    fn v0310_adds_residue_refinement_and_fragment_aux_namespaces() {
        let varmap = VarMap::new();
        let _model = PeptideFoundationMultimodalV0310Model::new(
            config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
        )
        .unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(data
            .keys()
            .any(|name| name.starts_with("property_refinement.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("fragment_representation_aux.")));
        let delta_weight = data
            .get("property_refinement.delta_output.weight")
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(delta_weight.iter().all(|&value| value == 0.0));
    }
}
