//! v0.33 hypothesis-specific fragment-grounded inverse decoding.
//!
//! v0.32 showed that broader or mass-stratified beam retention can increase the
//! number of hard-mass candidates without increasing correct-sequence recall.
//! v0.33 therefore changes the *model score* rather than the beam policy.
//! The complete v0.27 causal decoder is kept frozen. At every autoregressive
//! prefix, every legal next-token candidate receives explicit mass/chemistry and
//! observed b/y fragment-support features. A small trainable nonlinear residual
//! converts those candidate-specific features plus the frozen causal hidden state
//! into one additive next-token logit. The residual output is zero initialized so
//! step 0 reproduces the frozen v0.27 decoder exactly.

use super::causal::FoundationCausalOutput;
use super::chemistry_decoder::{
    ChemistryTransitionBatch, FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
};
use super::diffusion::{FOUNDATION_DIFFUSION_VOCAB_SIZE, FOUNDATION_OPEN_PTM_VOCAB_SIZE};
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};

pub const FOUNDATION_FRAGMENT_GROUNDED_ARCHITECTURE_V0330: &str =
    "frozen_v0270_causal_plus_hypothesis_specific_fragment_grounded_transition_residual_v1";
pub const FOUNDATION_FRAGMENT_GROUNDED_OBJECTIVE_V0330: &str =
    "teacher_forced_next_token_ce_plus_matched_shuffled_spectrum_dependence_v1";
pub const FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330: usize = 128;
pub const FOUNDATION_FRAGMENT_GROUNDED_JOINT_V0330: usize = 96;
pub const FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330: usize =
    FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200;

/// Trainable v0.33 candidate-specific residual head.
///
/// The caller owns the frozen v0.27 causal model and supplies its decoder hidden
/// states and base token logits. Only this head is optimized in v0.33.
#[derive(Clone)]
pub struct FragmentGroundedTransitionHeadV0330 {
    feature_in: Linear,
    hidden_in: Linear,
    joint_hidden: Linear,
    output: Linear,
    model_dim: usize,
}

impl FragmentGroundedTransitionHeadV0330 {
    pub fn new(model_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        if model_dim == 0 {
            candle_core::bail!("v0.33 model_dim must be positive");
        }
        let root = vb.pp("fragment_grounded_v0330");
        let feature_in = nn::linear(
            FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330,
            FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
            root.pp("feature_in"),
        )?;
        let hidden_in = nn::linear(
            model_dim,
            FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
            root.pp("hidden_in"),
        )?;
        let joint_hidden = nn::linear(
            FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
            FOUNDATION_FRAGMENT_GROUNDED_JOINT_V0330,
            root.pp("joint_hidden"),
        )?;

        // Exact identity at step 0. Earlier layers may use ordinary random
        // initialization because the zero output projection makes the complete
        // residual exactly zero until optimization begins.
        let output_vb = root.pp("output");
        let output_weight = output_vb.get_with_hints(
            (1, FOUNDATION_FRAGMENT_GROUNDED_JOINT_V0330),
            "weight",
            nn::Init::Const(0.0),
        )?;
        let output_bias = output_vb.get_with_hints(1, "bias", nn::Init::Const(0.0))?;
        let output = Linear::new(output_weight, Some(output_bias));

        Ok(Self {
            feature_in,
            hidden_in,
            joint_hidden,
            output,
            model_dim,
        })
    }

    pub fn apply_teacher_forced(
        &self,
        base: FoundationCausalOutput,
        chemistry: &ChemistryTransitionBatch,
    ) -> Result<FoundationCausalOutput> {
        let (batch, positions, classes) = base.token_logits.dims3()?;
        if classes != FOUNDATION_OPEN_PTM_VOCAB_SIZE {
            candle_core::bail!(
                "v0.33 expected open-PTM causal vocabulary {}, observed {classes}",
                FOUNDATION_OPEN_PTM_VOCAB_SIZE
            );
        }
        let (hidden_batch, hidden_positions, hidden_dim) = base.decoder_hidden.dims3()?;
        if (hidden_batch, hidden_positions, hidden_dim) != (batch, positions, self.model_dim) {
            candle_core::bail!("v0.33 causal hidden shape mismatch");
        }
        let features = &chemistry.features;
        if features.dims4()?
            != (
                batch,
                positions,
                FOUNDATION_DIFFUSION_VOCAB_SIZE,
                FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330,
            )
        {
            candle_core::bail!("v0.33 teacher-forced fragment feature shape mismatch");
        }

        let residual = self.score_full(&base.decoder_hidden, features)?;
        let legacy_logits = base
            .token_logits
            .narrow(2, 0, FOUNDATION_DIFFUSION_VOCAB_SIZE)?;
        let adjusted_legacy = (legacy_logits + residual)?;
        let open_tail = base.token_logits.narrow(
            2,
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_OPEN_PTM_VOCAB_SIZE - FOUNDATION_DIFFUSION_VOCAB_SIZE,
        )?;
        let token_logits = Tensor::cat(&[&adjusted_legacy, &open_tail], 2)?;
        Ok(FoundationCausalOutput {
            token_logits,
            decoder_hidden: base.decoder_hidden,
            spectrum_memory: base.spectrum_memory,
            spectrum_memory_mask: base.spectrum_memory_mask,
            spectrum_embedding: base.spectrum_embedding,
        })
    }

    /// Add v0.33 candidate-specific residuals to one next-token logit matrix.
    pub fn apply_next(
        &self,
        hidden: &Tensor,
        base_logits: &Tensor,
        features: &Tensor,
    ) -> Result<Tensor> {
        let (batch, classes) = base_logits.dims2()?;
        if classes != FOUNDATION_OPEN_PTM_VOCAB_SIZE {
            candle_core::bail!("v0.33 next-token base vocabulary mismatch");
        }
        if hidden.dims2()? != (batch, self.model_dim) {
            candle_core::bail!("v0.33 next-token hidden shape mismatch");
        }
        if features.dims3()?
            != (
                batch,
                FOUNDATION_DIFFUSION_VOCAB_SIZE,
                FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330,
            )
        {
            candle_core::bail!("v0.33 next-token fragment feature shape mismatch");
        }
        let residual = self.score_next(hidden, features)?;
        let legacy_logits = base_logits.narrow(1, 0, FOUNDATION_DIFFUSION_VOCAB_SIZE)?;
        let adjusted_legacy = (legacy_logits + residual)?;
        let open_tail = base_logits.narrow(
            1,
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_OPEN_PTM_VOCAB_SIZE - FOUNDATION_DIFFUSION_VOCAB_SIZE,
        )?;
        Tensor::cat(&[&adjusted_legacy, &open_tail], 1)
    }

    fn score_full(&self, hidden: &Tensor, features: &Tensor) -> Result<Tensor> {
        let (batch, positions, _, _) = features.dims4()?;
        let flat_features = features.reshape((
            batch * positions * FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330,
        ))?;
        let feature_state = self.feature_in.forward(&flat_features)?.relu()?.reshape((
            batch,
            positions,
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
        ))?;
        let hidden_state = self
            .hidden_in
            .forward(hidden)?
            .relu()?
            .unsqueeze(2)?
            .broadcast_as((
                batch,
                positions,
                FOUNDATION_DIFFUSION_VOCAB_SIZE,
                FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
            ))?;
        let joint = (feature_state + hidden_state)?.relu()?.reshape((
            batch * positions * FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
        ))?;
        self.output
            .forward(&self.joint_hidden.forward(&joint)?.relu()?)?
            .reshape((batch, positions, FOUNDATION_DIFFUSION_VOCAB_SIZE))
    }

    fn score_next(&self, hidden: &Tensor, features: &Tensor) -> Result<Tensor> {
        let (batch, _, _) = features.dims3()?;
        let flat_features = features.reshape((
            batch * FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330,
        ))?;
        let feature_state = self.feature_in.forward(&flat_features)?.relu()?.reshape((
            batch,
            FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
        ))?;
        let hidden_state = self
            .hidden_in
            .forward(hidden)?
            .relu()?
            .unsqueeze(1)?
            .broadcast_as((
                batch,
                FOUNDATION_DIFFUSION_VOCAB_SIZE,
                FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
            ))?;
        let joint = (feature_state + hidden_state)?.relu()?.reshape((
            batch * FOUNDATION_DIFFUSION_VOCAB_SIZE,
            FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
        ))?;
        self.output
            .forward(&self.joint_hidden.forward(&joint)?.relu()?)?
            .reshape((batch, FOUNDATION_DIFFUSION_VOCAB_SIZE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn v0330_zero_output_initialization_is_exact_identity() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let head = FragmentGroundedTransitionHeadV0330::new(
            16,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )
        .unwrap();
        let hidden = Tensor::ones((2, 16), DType::F32, &device).unwrap();
        let base = Tensor::ones((2, FOUNDATION_OPEN_PTM_VOCAB_SIZE), DType::F32, &device).unwrap();
        let features = Tensor::ones(
            (
                2,
                FOUNDATION_DIFFUSION_VOCAB_SIZE,
                FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330,
            ),
            DType::F32,
            &device,
        )
        .unwrap();
        let adjusted = head.apply_next(&hidden, &base, &features).unwrap();
        assert_eq!(
            adjusted.to_vec2::<f32>().unwrap(),
            base.to_vec2::<f32>().unwrap()
        );
    }
}
