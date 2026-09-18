//! v0.28 property-aware shared representation refinement.
//!
//! v0.28 keeps the frozen v0.27 RT specialist, contextual fragment decoder,
//! CCS physics/residual path, inverse models, spectrum encoder, alignment, and
//! relation objective unchanged. The single architectural change is a small
//! **shared task-conditioned residual adapter** applied after the common peptide
//! encoder and before the three forward-property heads.
//!
//! One adapter is reused for RT, CCS, and MS2. A learned task embedding selects
//! a view of the shared residue representation, while a zero-initialized output
//! projection makes the complete v0.28 forward path exactly reproduce v0.27 at
//! initialization. Auxiliary representation objectives continue to consume the
//! unconditioned shared encoder output so the new lane cannot silently replace
//! the accepted foundation representation with three independent encoders.

use super::causal::PeptideSpectrumCausalModel;
use super::config::FoundationConfig;
use super::diffusion::{
    FoundationDiffusionConfig, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
};
use super::layers::FoundationLayerNorm;
use super::model::{
    FoundationMultiTaskOutput, FoundationOutput, PeptideFoundationEncoder, PrecursorContextBatch,
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
use candle_nn::{self as nn, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

/// Stable v0.28 architecture identifier.
pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0280: &str =
    "v0.28-192d-shared-task-conditioned-residual-v027-heads-openptm32";
/// Number of forward property views. RT, CCS, and MS2 are the only conditioned tasks.
pub const FOUNDATION_TASK_CONDITION_COUNT_V0280: usize = 3;
/// Width of the learned task embedding injected into the shared adapter.
pub const FOUNDATION_TASK_CONDITION_EMBED_DIM_V0280: usize = 32;
/// Fixed low-rank bottleneck width of the shared task adapter.
pub const FOUNDATION_TASK_CONDITION_BOTTLENECK_V0280: usize = 64;
/// v0.28 intentionally preserves the accepted v0.27 scalar-property width.
pub const FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0280: usize =
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270;
/// v0.28 intentionally preserves the accepted same-spectrum relation margin.
pub const FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0280: f64 =
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270;

/// Complete fixed v0.28 architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeptideFoundationMultimodalV0280Config {
    /// Frozen v0.27 forward/inverse/head architecture.
    pub base_v0270: PeptideFoundationMultimodalV0270Config,
    /// Shared task-embedding width (fixed to 32).
    pub task_embedding_dim: usize,
    /// Shared residual-adapter bottleneck width (fixed to 64).
    pub task_bottleneck_dim: usize,
    /// Number of conditioned forward-property tasks (fixed to 3).
    pub task_count: usize,
}

impl PeptideFoundationMultimodalV0280Config {
    /// Construct the single predeclared v0.28 architecture from a frozen v0.27 config.
    pub fn fixed(base_v0270: PeptideFoundationMultimodalV0270Config) -> Result<Self> {
        base_v0270.validate()?;
        let config = Self {
            base_v0270,
            task_embedding_dim: FOUNDATION_TASK_CONDITION_EMBED_DIM_V0280,
            task_bottleneck_dim: FOUNDATION_TASK_CONDITION_BOTTLENECK_V0280,
            task_count: FOUNDATION_TASK_CONDITION_COUNT_V0280,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate that no silent width/depth sweep has changed the v0.28 design.
    pub fn validate(&self) -> Result<()> {
        self.base_v0270.validate()?;
        if self.base_v0270.forward.model_dim != 192 {
            candle_core::bail!(
                "v0.28 is fixed at model_dim=192, got {}",
                self.base_v0270.forward.model_dim
            );
        }
        if self.task_embedding_dim != FOUNDATION_TASK_CONDITION_EMBED_DIM_V0280
            || self.task_bottleneck_dim != FOUNDATION_TASK_CONDITION_BOTTLENECK_V0280
            || self.task_count != FOUNDATION_TASK_CONDITION_COUNT_V0280
        {
            candle_core::bail!(
                "v0.28 task-conditioning dimensions differ from the frozen architecture"
            );
        }
        Ok(())
    }

    /// Shared forward configuration inherited unchanged from v0.27.
    pub fn forward(&self) -> &FoundationConfig {
        &self.base_v0270.forward
    }

    /// Shared inverse configuration inherited unchanged from v0.27.
    pub fn inverse(&self) -> &FoundationDiffusionConfig {
        &self.base_v0270.inverse
    }
}

#[derive(Debug, Clone, Copy)]
enum FoundationForwardTaskV0280 {
    Rt = 0,
    Ccs = 1,
    Ms2 = 2,
}

#[derive(Clone)]
struct SharedTaskConditionerV0280 {
    input_norm: FoundationLayerNorm,
    task_embedding: Embedding,
    down: Linear,
    up: Linear,
    model_dim: usize,
    task_embedding_dim: usize,
}

impl SharedTaskConditionerV0280 {
    fn new(config: &PeptideFoundationMultimodalV0280Config, vb: VarBuilder<'_>) -> Result<Self> {
        let model_dim = config.forward().model_dim;
        Ok(Self {
            input_norm: FoundationLayerNorm::new(model_dim, 1e-5, vb.pp("input_norm"))?,
            task_embedding: nn::embedding(
                config.task_count,
                config.task_embedding_dim,
                vb.pp("task_embedding"),
            )?,
            down: nn::linear(
                model_dim + config.task_embedding_dim,
                config.task_bottleneck_dim,
                vb.pp("down"),
            )?,
            // Zero output is a scientific invariant: before optimization, every
            // v0.28 property prediction must be exactly the frozen v0.27 result.
            up: zero_initialized_linear_v0280(config.task_bottleneck_dim, model_dim, vb.pp("up"))?,
            model_dim,
            task_embedding_dim: config.task_embedding_dim,
        })
    }

    fn condition(
        &self,
        foundation: &FoundationOutput,
        task: FoundationForwardTaskV0280,
    ) -> Result<FoundationOutput> {
        let (batch, sequence, model_dim) = foundation.residue_embeddings.dims3()?;
        if model_dim != self.model_dim {
            candle_core::bail!(
                "v0.28 task conditioner expected residue width {}, got {}",
                self.model_dim,
                model_dim
            );
        }
        let normalized = self.input_norm.forward(&foundation.residue_embeddings)?;
        let task_ids = Tensor::from_vec(vec![task as u32; batch], batch, normalized.device())?;
        let task_embedding = self
            .task_embedding
            .forward(&task_ids)?
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, self.task_embedding_dim))?;
        // The shared encoder can return a logically [B, S, D] tensor with a
        // non-contiguous CUDA layout. Concatenating the broadcast task embedding
        // preserves that layout, while Candle's CUDA Linear/matmul requires a
        // contiguous left operand. Materialize the two Linear inputs explicitly;
        // this is layout-only and does not change v0.28 numerics or gradients.
        let features = Tensor::cat(&[&normalized, &task_embedding], 2)?.contiguous()?;
        let hidden = self.down.forward(&features)?.relu()?.contiguous()?;
        let residual = self.up.forward(&hidden)?;
        let expanded_mask = foundation
            .residue_mask
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, model_dim))?;
        let residual = residual.broadcast_mul(&expanded_mask)?;
        let residue_embeddings = (&foundation.residue_embeddings + &residual)?;
        let pooled_residual = masked_mean_v0280(&residual, &foundation.residue_mask)?;
        let peptide_embedding = (&foundation.peptide_embedding + &pooled_residual)?;
        Ok(FoundationOutput {
            residue_embeddings,
            peptide_embedding,
            residue_mask: foundation.residue_mask.clone(),
            chemistry_targets: foundation.chemistry_targets.clone(),
        })
    }
}

/// v0.28 forward output retaining the historical multi-task surface.
#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0280 {
    /// Shared/unconditioned foundation representation plus conditioned property outputs.
    pub base: FoundationMultiTaskOutput,
    /// Fragment-presence logits from the conditioned v0.27 decoder.
    pub ms2_presence_logits: Tensor,
    /// Non-negative conditional fragment intensities.
    pub ms2_positive_intensity: Tensor,
}

/// Complete v0.28 model: frozen v0.27 architecture plus one shared task adapter.
#[derive(Clone)]
pub struct PeptideFoundationMultimodalV0280Model {
    base_v0270: PeptideFoundationMultimodalV0270Model,
    task_conditioner: SharedTaskConditionerV0280,
    config: PeptideFoundationMultimodalV0280Config,
}

impl PeptideFoundationMultimodalV0280Model {
    /// Construct the fixed v0.28 model while preserving all v0.27 variable namespaces.
    pub fn new(config: PeptideFoundationMultimodalV0280Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let base_v0270 =
            PeptideFoundationMultimodalV0270Model::new(config.base_v0270.clone(), vb.clone())?;
        let task_conditioner = SharedTaskConditionerV0280::new(&config, vb.pp("task_conditioner"))?;
        Ok(Self {
            base_v0270,
            task_conditioner,
            config,
        })
    }

    /// Forward-property pass with v0.28 task-conditioned views.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_v0280_t_with_shared_gradient_scales(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
        rt_encoder_gradient_scale: f64,
        ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultimodalForwardOutputV0280> {
        let foundation = self
            .base_v0270
            .forward()
            .encode_foundation_t(batch, train)?;

        let rt_foundation = self
            .task_conditioner
            .condition(&foundation, FoundationForwardTaskV0280::Rt)?;
        let ccs_foundation = self
            .task_conditioner
            .condition(&foundation, FoundationForwardTaskV0280::Ccs)?;
        let ms2_foundation = self
            .task_conditioner
            .condition(&foundation, FoundationForwardTaskV0280::Ms2)?;

        let rt = self.base_v0270.forward().rt_from_foundation_t(
            &rt_foundation,
            train,
            rt_encoder_gradient_scale,
        )?;
        let ccs = self.base_v0270.forward().ccs_from_foundation(
            &ccs_foundation,
            context,
            ccs_encoder_gradient_scale,
        )?;
        let (ms2_presence_logits, ms2_positive_intensity, ms2) = self
            .base_v0270
            .forward()
            .ms2_from_foundation_t(&ms2_foundation, context, fragment, train)?;

        // Representation/inverse auxiliaries remain anchored to the unconditioned
        // shared encoder. This is what makes v0.28 a task-view refinement rather
        // than three hidden task-specific encoders.
        let (residue_logits, chemistry_reconstruction, contrastive_projection) = self
            .base_v0270
            .forward()
            .auxiliaries_from_foundation(&foundation)?;

        Ok(FoundationMultimodalForwardOutputV0280 {
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

    /// Forward-property pass with ordinary shared-encoder gradients.
    pub fn forward_v0280_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0270,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0280> {
        self.forward_v0280_t_with_shared_gradient_scales(batch, context, fragment, train, 1.0, 1.0)
    }

    /// Shared peptide encoder, unchanged from v0.27.
    pub fn encoder(&self) -> &PeptideFoundationEncoder {
        self.base_v0270.forward().encoder()
    }

    /// Shared peptide projection used by alignment/contrastive objectives.
    pub fn peptide_projection_t(
        &self,
        batch: &super::featurize::FoundationBatch,
        train: bool,
    ) -> Result<Tensor> {
        self.base_v0270.forward().peptide_projection_t(batch, train)
    }

    /// Spectrum-conditioned diffusion auxiliary, unchanged from v0.27.
    pub fn diffusion(&self) -> &PeptideSpectrumDiffusionModel {
        self.base_v0270.diffusion()
    }

    /// Spectrum-conditioned causal auxiliary, unchanged from v0.27.
    pub fn causal(&self) -> &PeptideSpectrumCausalModel {
        self.base_v0270.causal()
    }

    /// Encode measured/library peaks with the unchanged v0.27 spectrum encoder.
    pub fn encode_spectrum_t(
        &self,
        spectrum: &FoundationSpectrumBatch,
        train: bool,
    ) -> Result<FoundationSpectrumEncoding> {
        self.base_v0270.encode_spectrum_t(spectrum, train)
    }

    /// Project pooled spectrum embeddings into the unchanged contrastive space.
    pub fn project_spectrum_embedding(&self, embedding: &Tensor) -> Result<Tensor> {
        self.base_v0270.project_spectrum_embedding(embedding)
    }

    /// Same-spectrum relation score remains anchored to the unconditioned shared representation.
    pub fn relation_score(
        &self,
        peptide: &FoundationOutput,
        spectrum: &FoundationSpectrumEncoding,
    ) -> Result<Tensor> {
        self.base_v0270.relation_score(peptide, spectrum)
    }

    /// Shared forward config inherited unchanged from v0.27.
    pub fn forward_config(&self) -> &FoundationConfig {
        self.config.forward()
    }

    /// Shared inverse config inherited unchanged from v0.27.
    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        self.config.inverse()
    }

    /// Complete fixed v0.28 configuration.
    pub fn config(&self) -> &PeptideFoundationMultimodalV0280Config {
        &self.config
    }
}

/// v0.28 reuses the exact v0.27 fragment-context geometry contract.
pub type FoundationFragmentContextBatchV0280 = FoundationFragmentContextBatchV0270;

/// v0.28 uses the exact validated v0.27 factorized MS2 objective.
pub type FoundationMultimodalMs2LossesV0280 = FoundationMultimodalMs2LossesV0270;

/// Apply the unchanged v0.27 factorized MS2 loss to v0.28 outputs.
pub fn foundation_multimodal_ms2_loss_v0280(
    output: &FoundationMultimodalForwardOutputV0280,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
    presence_target: Option<&Tensor>,
    presence_mask: Option<&Tensor>,
) -> Result<FoundationMultimodalMs2LossesV0280> {
    let proxy = FoundationMultimodalForwardOutputV0270 {
        base: output.base.clone(),
        ms2_presence_logits: output.ms2_presence_logits.clone(),
        ms2_positive_intensity: output.ms2_positive_intensity.clone(),
    };
    foundation_multimodal_ms2_loss_v0270(&proxy, target, mask, presence_target, presence_mask)
}

/// Same-spectrum relation objective remains unchanged from v0.27.
pub fn foundation_multimodal_relation_margin_loss_v0280(
    positive_score: &Tensor,
    negative_score: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    foundation_multimodal_relation_margin_loss_v0270(positive_score, negative_score, margin)
}

fn masked_mean_v0280(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, length, dim) = values.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, length, dim))?;
    let summed = values.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    summed.broadcast_div(&denominator)
}

fn zero_initialized_linear_v0280(
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

    fn config() -> PeptideFoundationMultimodalV0280Config {
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
        PeptideFoundationMultimodalV0280Config::fixed(
            PeptideFoundationMultimodalV0270Config::fixed(forward, inverse).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn v0280_task_conditioner_zero_initialization_is_exact_identity() -> Result<()> {
        let device = Device::Cpu;
        let config = config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let conditioner = SharedTaskConditionerV0280::new(&config, vb)?;
        let residue_embeddings = Tensor::arange(0f32, 4f32 * 192f32, &device)?
            .reshape((1, 4, 192))?
            .affine(1.0 / 1000.0, 0.0)?;
        let foundation = FoundationOutput {
            peptide_embedding: residue_embeddings.mean(1)?,
            residue_embeddings,
            residue_mask: Tensor::ones((1, 4), DType::F32, &device)?,
            chemistry_targets: Tensor::zeros((1, 4, 3), DType::F32, &device)?,
        };
        for task in [
            FoundationForwardTaskV0280::Rt,
            FoundationForwardTaskV0280::Ccs,
            FoundationForwardTaskV0280::Ms2,
        ] {
            let conditioned = conditioner.condition(&foundation, task)?;
            assert_eq!(
                conditioned.residue_embeddings.to_vec3::<f32>()?,
                foundation.residue_embeddings.to_vec3::<f32>()?
            );
            assert_eq!(
                conditioned.peptide_embedding.to_vec2::<f32>()?,
                foundation.peptide_embedding.to_vec2::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn v0280_model_constructs_one_shared_task_conditioner_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideFoundationMultimodalV0280Model::new(config(), vb)?;
        assert_eq!(model.config().task_count, 3);
        assert_eq!(model.config().task_embedding_dim, 32);
        assert_eq!(model.config().task_bottleneck_dim, 64);
        Ok(())
    }
}
