//! Hierarchical chemistry-aware peptide encoder and multi-task prediction heads.

use super::ccs_physics::FOUNDATION_CCS_PHYSICS_FEATURE_COUNT;
use super::chemistry::ATOM_FEATURE_DIM;
use super::config::{
    FoundationCcsContextMode, FoundationCcsPhysicsBaselineConfig, FoundationConfig,
    FoundationMs2OutputActivation, FOUNDATION_MS2_SOFTPLUS_BETA_V0138,
};
use super::featurize::FoundationBatch;
use super::layers::{FoundationLayerNorm, GraphMessageLayer, PeptideTransformerBlock};
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{self as nn, Embedding, Linear, VarBuilder};

/// Foundation representation shared by all downstream peptide-property heads.
#[derive(Debug, Clone)]
pub struct FoundationOutput {
    /// Contextual residue representations `[batch, residues, model_dim]`.
    pub residue_embeddings: Tensor,
    /// Mask-aware pooled peptide representation `[batch, model_dim]`.
    pub peptide_embedding: Tensor,
    /// Residue-validity mask `[batch, residues]`.
    pub residue_mask: Tensor,
    /// Mean raw atom descriptor per residue, used as a reconstruction target.
    pub chemistry_targets: Tensor,
}

/// Chemistry-aware atom/residue GNN followed by a peptide Transformer.
#[derive(Clone)]
pub struct PeptideFoundationEncoder {
    config: FoundationConfig,
    atom_input: Linear,
    graph_layers: Vec<GraphMessageLayer>,
    graph_to_model: Linear,
    residue_embedding: Embedding,
    position_embedding: Embedding,
    input_norm: FoundationLayerNorm,
    transformer_layers: Vec<PeptideTransformerBlock>,
    output_norm: FoundationLayerNorm,
}

impl PeptideFoundationEncoder {
    /// Construct a randomly initialized encoder from a Candle variable builder.
    pub fn new(config: FoundationConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let atom_input = nn::linear(
            config.atom_feature_dim,
            config.graph_hidden_dim,
            vb.pp("atom_input"),
        )?;
        let graph_layers = (0..config.graph_layers)
            .map(|index| {
                GraphMessageLayer::new(config.graph_hidden_dim, vb.pp(format!("graph.{index}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let graph_to_model = nn::linear(
            config.graph_hidden_dim,
            config.model_dim,
            vb.pp("graph_to_model"),
        )?;
        let residue_embedding = nn::embedding(21, config.model_dim, vb.pp("residue_embedding"))?;
        let position_embedding = nn::embedding(
            config.max_sequence_len,
            config.model_dim,
            vb.pp("position_embedding"),
        )?;
        let input_norm = FoundationLayerNorm::new(config.model_dim, 1e-5, vb.pp("input_norm"))?;
        let transformer_layers = (0..config.transformer_layers)
            .map(|index| {
                PeptideTransformerBlock::new(
                    config.model_dim,
                    config.num_attention_heads,
                    config.transformer_ff_dim,
                    config.dropout,
                    vb.pp(format!("transformer.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let output_norm = FoundationLayerNorm::new(config.model_dim, 1e-5, vb.pp("output_norm"))?;

        Ok(Self {
            config,
            atom_input,
            graph_layers,
            graph_to_model,
            residue_embedding,
            position_embedding,
            input_norm,
            transformer_layers,
            output_norm,
        })
    }

    /// Encode atom graphs and peptide sequence into reusable representations.
    pub fn forward_t(&self, batch: &FoundationBatch, train: bool) -> Result<FoundationOutput> {
        let (batch_size, sequence_len, atom_count, feature_dim) = batch.atom_features.dims4()?;
        if feature_dim != ATOM_FEATURE_DIM {
            candle_core::bail!(
                "foundation atom feature mismatch: expected {}, got {}",
                ATOM_FEATURE_DIM,
                feature_dim
            );
        }
        let graph_count = batch_size * sequence_len;
        let atoms = batch
            .atom_features
            .reshape((graph_count, atom_count, feature_dim))?;
        let adjacency = batch
            .adjacency
            .reshape((graph_count, atom_count, atom_count))?;
        let atom_mask = batch.atom_mask.reshape((graph_count, atom_count))?;

        let mut hidden = self.atom_input.forward(&atoms)?;
        let expanded_atom_mask = atom_mask.unsqueeze(2)?.broadcast_as((
            graph_count,
            atom_count,
            self.config.graph_hidden_dim,
        ))?;
        hidden = hidden.broadcast_mul(&expanded_atom_mask)?;
        for layer in &self.graph_layers {
            hidden = layer.forward(&hidden, &adjacency, &atom_mask)?;
        }

        let atom_sum = hidden.sum(1)?;
        let atom_denominator = atom_mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
        let residue_chemistry = atom_sum.broadcast_div(&atom_denominator)?;
        let residue_chemistry = self.graph_to_model.forward(&residue_chemistry)?.reshape((
            batch_size,
            sequence_len,
            self.config.model_dim,
        ))?;

        let sequence_embedding = self.residue_embedding.forward(&batch.residue_ids)?;
        let positions: Vec<u32> = (0..sequence_len as u32).collect();
        let position_ids = Tensor::from_vec(positions, sequence_len, batch.residue_ids.device())?
            .to_dtype(DType::U32)?;
        let position_embedding = self
            .position_embedding
            .forward(&position_ids)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, sequence_len, self.config.model_dim))?;

        let mut residues = self
            .input_norm
            .forward(&((residue_chemistry + sequence_embedding)? + position_embedding)?)?;
        let residue_mask = batch.residue_mask.unsqueeze(2)?.broadcast_as((
            batch_size,
            sequence_len,
            self.config.model_dim,
        ))?;
        residues = residues.broadcast_mul(&residue_mask)?;
        for block in &self.transformer_layers {
            residues = block.forward_t(&residues, &batch.residue_mask, train)?;
        }
        residues = self.output_norm.forward(&residues)?;
        residues = residues.broadcast_mul(&residue_mask)?;

        let residue_sum = residues.sum(1)?;
        let residue_denominator = batch
            .residue_mask
            .sum(1)?
            .clamp(1.0, f64::INFINITY)?
            .unsqueeze(1)?;
        let peptide_embedding = residue_sum.broadcast_div(&residue_denominator)?;

        let raw_chemistry_sum = batch.atom_features.sum(2)?;
        let raw_chemistry_denominator = batch
            .atom_mask
            .sum(2)?
            .clamp(1.0, f64::INFINITY)?
            .unsqueeze(2)?;
        let chemistry_targets = raw_chemistry_sum.broadcast_div(&raw_chemistry_denominator)?;

        Ok(FoundationOutput {
            residue_embeddings: residues,
            peptide_embedding,
            residue_mask: batch.residue_mask.clone(),
            chemistry_targets,
        })
    }

    /// Return the model configuration used by this encoder.
    pub fn config(&self) -> &FoundationConfig {
        &self.config
    }
}

/// Experimental context deliberately kept separate from the intrinsic peptide embedding.
#[derive(Debug, Clone)]
pub struct PrecursorContextBatch {
    /// Precursor charge as `[batch]` floating-point values. Missing values are zero-filled.
    pub charge: Tensor,
    /// One where precursor charge is known, zero where it is unavailable.
    pub charge_present: Tensor,
    /// Precursor m/z as `[batch]` floating-point values. Missing values are zero-filled.
    pub precursor_mz: Tensor,
    /// One where precursor m/z is known, zero where it is unavailable.
    pub precursor_mz_present: Tensor,
    /// Normalized collision energy as `[batch]` floating-point values. Missing values are zero-filled.
    pub nce: Tensor,
    /// One where NCE is known, zero where it is unavailable.
    pub nce_present: Tensor,
    /// Integer instrument ids as `[batch]`; id zero is the learned unknown category.
    pub instrument_ids: Tensor,
    /// One where an instrument identity is known, zero for the unknown category.
    pub instrument_present: Tensor,
}

impl PrecursorContextBatch {
    /// Construct a fully unknown acquisition-context batch.
    ///
    /// This is the canonical inference path when only peptide chemistry is
    /// available. Unknown context is represented explicitly rather than being
    /// confused with a measured zero-valued charge/NCE.
    pub fn unknown(batch_size: usize, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            charge: Tensor::zeros(batch_size, DType::F32, device)?,
            charge_present: Tensor::zeros(batch_size, DType::F32, device)?,
            precursor_mz: Tensor::zeros(batch_size, DType::F32, device)?,
            precursor_mz_present: Tensor::zeros(batch_size, DType::F32, device)?,
            nce: Tensor::zeros(batch_size, DType::F32, device)?,
            nce_present: Tensor::zeros(batch_size, DType::F32, device)?,
            instrument_ids: Tensor::zeros(batch_size, DType::U32, device)?,
            instrument_present: Tensor::zeros(batch_size, DType::F32, device)?,
        })
    }
}

/// Outputs produced by the shared encoder and all first-generation heads.
#[derive(Debug, Clone)]
pub struct FoundationMultiTaskOutput {
    /// Shared foundation representation.
    pub foundation: FoundationOutput,
    /// RT/iRT prediction `[batch, 1]`.
    pub rt: Tensor,
    /// CCS prediction `[batch, 1]`.
    pub ccs: Tensor,
    /// Fragment intensities `[batch, max_sequence_len - 1, channels]`.
    pub ms2: Tensor,
    /// Residue-token reconstruction logits `[batch, residues, 21]`.
    pub residue_logits: Tensor,
    /// Chemistry reconstruction `[batch, residues, atom_feature_dim]`.
    pub chemistry_reconstruction: Tensor,
    /// Contrastive projection `[batch, contrastive_dim]`.
    pub contrastive_projection: Tensor,
}

/// Multi-task model used to pretrain the foundation encoder.
#[derive(Clone)]
pub struct PeptideFoundationMultiTaskModel {
    encoder: PeptideFoundationEncoder,
    rt_head: Linear,
    ccs_head: Linear,
    instrument_embedding: Embedding,
    ms2_head: Linear,
    residue_head: Linear,
    chemistry_head: Linear,
    contrastive_head: Linear,
    config: FoundationConfig,
}

impl PeptideFoundationMultiTaskModel {
    /// Construct the shared encoder and prediction/self-supervision heads.
    pub fn new(config: FoundationConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let encoder = PeptideFoundationEncoder::new(config.clone(), vb.pp("encoder"))?;
        Ok(Self {
            rt_head: zero_initialized_regression_linear(config.model_dim, vb.pp("heads.rt"))?,
            ccs_head: zero_initialized_regression_linear(config.model_dim + 2, vb.pp("heads.ccs"))?,
            instrument_embedding: nn::embedding(
                config.instrument_vocab_size,
                16,
                vb.pp("context.instrument"),
            )?,
            ms2_head: nn::linear(
                config.model_dim * 2 + 21,
                config.ms2_fragment_channels,
                vb.pp("heads.ms2"),
            )?,
            residue_head: nn::linear(config.model_dim, 21, vb.pp("heads.masked_residue"))?,
            chemistry_head: nn::linear(
                config.model_dim,
                config.atom_feature_dim,
                vb.pp("heads.chemistry"),
            )?,
            contrastive_head: nn::linear(
                config.model_dim,
                config.contrastive_dim,
                vb.pp("heads.contrastive"),
            )?,
            encoder,
            config,
        })
    }

    /// Forward pass for joint supervised and self-supervised pretraining.
    pub fn forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationMultiTaskOutput> {
        self.forward_t_with_shared_gradient_scales(batch, context, train, 1.0, 1.0)
    }

    /// Forward pass with independent control over the RT gradient entering the shared encoder.
    ///
    /// The forward value supplied to the RT head is unchanged. During backpropagation,
    /// `rt_encoder_gradient_scale` multiplies only the gradient flowing from the RT head
    /// into the shared peptide embedding. Gradients for the RT head parameters themselves
    /// remain unscaled. A value of `1` reproduces the ordinary forward pass; `0` trains the
    /// RT head on a detached foundation embedding.
    pub fn forward_t_with_rt_encoder_gradient_scale(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
        rt_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultiTaskOutput> {
        self.forward_t_with_shared_gradient_scales(
            batch,
            context,
            train,
            rt_encoder_gradient_scale,
            1.0,
        )
    }

    /// Forward pass with independent RT and CCS gradient control at the shared encoder.
    ///
    /// Forward predictions and head-parameter gradients are unchanged. The supplied
    /// scales only multiply the gradients that flow from the RT/CCS heads back into
    /// the shared peptide embedding. A value of `1` is ordinary training; `0` trains
    /// the corresponding property head on a detached foundation representation.
    pub fn forward_t_with_shared_gradient_scales(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
        rt_encoder_gradient_scale: f64,
        ccs_encoder_gradient_scale: f64,
    ) -> Result<FoundationMultiTaskOutput> {
        validate_encoder_gradient_scale("RT", rt_encoder_gradient_scale)?;
        validate_encoder_gradient_scale("CCS", ccs_encoder_gradient_scale)?;

        let foundation = self.encoder.forward_t(batch, train)?;
        let rt_features =
            gradient_scaled_identity(&foundation.peptide_embedding, rt_encoder_gradient_scale)?;
        let rt = self.rt_head.forward(&rt_features)?;

        let ccs_embedding =
            gradient_scaled_identity(&foundation.peptide_embedding, ccs_encoder_gradient_scale)?;
        let scaled_charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let ccs_scalar_context = match self.config.ccs_context_mode {
            FoundationCcsContextMode::ChargePresence => {
                let charge_present = context.charge_present.unsqueeze(1)?;
                Tensor::cat(&[&scaled_charge, &charge_present], 1)?
            }
            FoundationCcsContextMode::NeutralMassCharge => {
                // `m/z * z` is a stable neutral-mass proxy; the proton correction is tiny
                // relative to peptide mass and can be learned as part of the scalar affine
                // head. Missing m/z is zero-filled, so unknown context remains a valid path.
                let neutral_mass = context.precursor_mz.broadcast_mul(&context.charge)?;
                let physical_present = context
                    .precursor_mz_present
                    .broadcast_mul(&context.charge_present)?;
                let scaled_neutral_mass = neutral_mass
                    .broadcast_mul(&physical_present)?
                    .affine(1.0 / 3000.0, 0.0)?
                    .unsqueeze(1)?;
                Tensor::cat(&[&scaled_neutral_mass, &scaled_charge], 1)?
            }
        };
        let ccs_features = Tensor::cat(&[&ccs_embedding, &ccs_scalar_context], 1)?;
        let ccs_residual = self.ccs_head.forward(&ccs_features)?;
        let ccs = if let Some(physics_baseline) = &self.config.ccs_physics_baseline {
            let baseline =
                standardized_ccs_physics_baseline(&foundation, context, physics_baseline)?;
            (&baseline + &ccs_residual)?
        } else {
            ccs_residual
        };

        let (batch_size, sequence_len, _) = foundation.residue_embeddings.dims3()?;
        let left = foundation
            .residue_embeddings
            .narrow(1, 0, sequence_len - 1)?;
        let right = foundation
            .residue_embeddings
            .narrow(1, 1, sequence_len - 1)?;
        let instrument = self.instrument_embedding.forward(&context.instrument_ids)?;
        let scaled_charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let scaled_nce = context.nce.affine(1.0 / 100.0, 0.0)?.unsqueeze(1)?;
        let charge_present = context.charge_present.unsqueeze(1)?;
        let nce_present = context.nce_present.unsqueeze(1)?;
        let instrument_present = context.instrument_present.unsqueeze(1)?;
        let scalar_context = Tensor::cat(
            &[
                &scaled_charge,
                &scaled_nce,
                &charge_present,
                &nce_present,
                &instrument_present,
            ],
            1,
        )?;
        let context_features = Tensor::cat(&[&instrument, &scalar_context], 1)?
            .unsqueeze(1)?
            .broadcast_as((batch_size, sequence_len - 1, 21))?;
        let cleavage_features = Tensor::cat(&[&left, &right, &context_features], 2)?;
        let ms2_logits = self.ms2_head.forward(&cleavage_features)?;
        let ms2 = apply_ms2_output_activation(&ms2_logits, self.config.ms2_output_activation)?;
        let left_mask = foundation.residue_mask.narrow(1, 0, sequence_len - 1)?;
        let right_mask = foundation.residue_mask.narrow(1, 1, sequence_len - 1)?;
        let cleavage_mask = left_mask
            .broadcast_mul(&right_mask)?
            .unsqueeze(2)?
            .broadcast_as((
                batch_size,
                sequence_len - 1,
                self.config.ms2_fragment_channels,
            ))?;
        let ms2 = ms2.broadcast_mul(&cleavage_mask)?;

        let residue_logits = self.residue_head.forward(&foundation.residue_embeddings)?;
        let chemistry_reconstruction = self
            .chemistry_head
            .forward(&foundation.residue_embeddings)?;
        let contrastive_projection = self
            .contrastive_head
            .forward(&foundation.peptide_embedding)?;

        Ok(FoundationMultiTaskOutput {
            foundation,
            rt,
            ccs,
            ms2,
            residue_logits,
            chemistry_reconstruction,
            contrastive_projection,
        })
    }

    /// Return the shared encoder for embedding-only inference or transfer learning.
    pub fn encoder(&self) -> &PeptideFoundationEncoder {
        &self.encoder
    }

    /// Return the model configuration.
    pub fn config(&self) -> &FoundationConfig {
        &self.config
    }
}

/// Evaluate a frozen train-derived physical CCS prior and map it into the
/// standardized target space used by the learned residual head.
///
/// The prior is intentionally parameter-free. This lets a model start from a
/// strong precursor-physics prediction while reserving all learned CCS
/// capacity for peptide-specific deviations from that baseline.
fn standardized_ccs_physics_baseline(
    foundation: &FoundationOutput,
    context: &PrecursorContextBatch,
    baseline: &FoundationCcsPhysicsBaselineConfig,
) -> Result<Tensor> {
    let batch_size = context.charge.dims1()?;
    let device = context.charge.device();

    let charge_present = &context.charge_present;
    let mz_present = &context.precursor_mz_present;
    let physical_present = charge_present.broadcast_mul(mz_present)?;

    let charge = context.charge.broadcast_mul(charge_present)?;
    let precursor_mz = context.precursor_mz.broadcast_mul(mz_present)?;
    let charge_squared = charge.sqr()?;
    let neutral_mass_proxy = charge
        .broadcast_mul(&precursor_mz)?
        .broadcast_mul(&physical_present)?;
    let sequence_len = foundation.residue_mask.sum(1)?;

    let ones = Tensor::ones(batch_size, DType::F32, device)?;
    let features = Tensor::cat(
        &[
            &ones.unsqueeze(1)?,
            &charge.affine(1.0 / 4.0, 0.0)?.unsqueeze(1)?,
            &charge_squared.affine(1.0 / 16.0, 0.0)?.unsqueeze(1)?,
            &precursor_mz.affine(1.0 / 1000.0, 0.0)?.unsqueeze(1)?,
            &neutral_mass_proxy.affine(1.0 / 3000.0, 0.0)?.unsqueeze(1)?,
            &sequence_len.affine(1.0 / 30.0, 0.0)?.unsqueeze(1)?,
            &charge_present.unsqueeze(1)?,
            &mz_present.unsqueeze(1)?,
        ],
        1,
    )?;

    let coefficients = Tensor::from_vec(
        baseline
            .coefficients_native
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        (FOUNDATION_CCS_PHYSICS_FEATURE_COUNT, 1),
        device,
    )?;
    let native = features.matmul(&coefficients)?;
    native.affine(
        1.0 / baseline.target_std_native,
        -baseline.target_mean_native / baseline.target_std_native,
    )
}

/// Create a scalar regression head with a deterministic zero prediction prior.
///
/// RT and CCS training targets are normally standardized on the train partition, so zero is
/// the neutral target-space prior. Zero-initializing the final scalar projection prevents a
/// random fresh head from starting several standard deviations away from the label mean and
/// dominating global gradient clipping. Checkpoint/safetensors loading is unaffected because
/// a file-backed `VarBuilder` supplies the stored tensors instead of applying these hints.
fn zero_initialized_regression_linear(in_dim: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get_with_hints((1, in_dim), "weight", nn::Init::Const(0.0))?;
    let bias = vb.get_with_hints(1, "bias", nn::Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

/// Apply the configured non-negative MS2 intensity activation.
///
/// The v0.13.8 Softplus uses a numerically stable decomposition:
/// `max(x, 0) + log(1 + exp(-beta * abs(x))) / beta`.
/// With beta=5 its value at zero is ~0.1386, keeping the activation close to
/// ReLU while restoring non-zero gradients through negative pre-activations.
#[doc(hidden)]
pub fn apply_ms2_output_activation(
    logits: &Tensor,
    activation: FoundationMs2OutputActivation,
) -> Result<Tensor> {
    match activation {
        FoundationMs2OutputActivation::Relu => logits.relu(),
        FoundationMs2OutputActivation::SoftplusV0138 => {
            let positive = logits.relu()?;
            let tail = logits
                .abs()?
                .affine(-FOUNDATION_MS2_SOFTPLUS_BETA_V0138, 0.0)?
                .exp()?;
            let tail = (tail + 1.0)?
                .log()?
                .affine(1.0 / FOUNDATION_MS2_SOFTPLUS_BETA_V0138, 0.0)?;
            (&positive + &tail)
        }
    }
}

/// Identity in the forward pass with a configurable gradient multiplier.
///
/// `detached + scale * (input - detached)` is numerically equal to `input`, while
/// only the second term participates in backpropagation to `input`.
fn validate_encoder_gradient_scale(label: &str, scale: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&scale) || !scale.is_finite() {
        candle_core::bail!(
            "{label} encoder gradient scale must be finite and within [0, 1], got {scale}"
        );
    }
    Ok(())
}

#[doc(hidden)]
pub fn gradient_scaled_identity(input: &Tensor, scale: f64) -> Result<Tensor> {
    if scale == 1.0 {
        return Ok(input.clone());
    }
    let detached = input.detach();
    let residual = (input - &detached)?;
    let scaled_residual = residual.affine(scale, 0.0)?;
    (&detached + &scaled_residual)
}
