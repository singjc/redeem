//! Production residue-pair/task-token peptide representation.
//!
//! This is the version-free production form of the representation mechanism that
//! proved useful during the foundation-model research: local chemistry is encoded
//! per residue, four learned task tokens join the residue sequence, and an explicit
//! O(L^2) pair state both biases attention and is updated from residue interactions.
//! Historical checkpoint compatibility remains quarantined in `checkpoint_compat`;
//! this module owns only the current production implementation.

use super::chemistry::ATOM_FEATURE_DIM;
use super::config::{FoundationCcsContextMode, FoundationConfig};
use super::featurize::FoundationBatch;
use super::layers::{FoundationLayerNorm, GraphMessageLayer, PeptideTransformerBlock};
use super::model::{
    apply_ms2_output_activation, standardized_ccs_physics_baseline, FoundationMultiTaskOutput,
    FoundationOutput, PrecursorContextBatch,
};
use super::runtime::FoundationSpecialistConfig;
use candle_core::{DType, Module, ModuleT, Result, Tensor, D};
use candle_nn::{self as nn, ops, Dropout, Embedding, Linear, VarBuilder};

const TASK_COUNT: usize = 4;
const TASK_RT: usize = 0;
const TASK_CCS: usize = 1;
const TASK_MS2: usize = 2;
const TASK_GLOBAL: usize = 3;
const PAIR_DIM: usize = 128;
const PAIR_FF_DIM: usize = 512;
const RELATIVE_FEATURE_DIM: usize = 6;
const CCS_TASK_CONTEXT_DIM: usize = 6;
const MS2_TASK_INSTRUMENT_DIM: usize = 32;
const MS2_TASK_CONTEXT_SCALAR_DIM: usize = 5;

#[derive(Debug, Clone)]
struct PairRepresentation {
    foundation: FoundationOutput,
    rt_embedding: Tensor,
    ccs_embedding: Tensor,
    ms2_embedding: Tensor,
    pair_embeddings: Tensor,
}

/// Auxiliary predictions are training-only; public RT/CCS/MS2 outputs remain unchanged.
#[derive(Debug, Clone)]
pub(crate) struct PairTaskAuxiliaryPredictions {
    pub(crate) pair_logits: Tensor,
    pub(crate) chemistry_summary: Tensor,
    pub(crate) conformation_proxy: Option<Tensor>,
}

#[derive(Clone)]
struct PairConditionedSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    pair_bias_heads: Vec<Linear>,
    num_heads: usize,
    head_dim: usize,
}

impl PairConditionedSelfAttention {
    fn new(model_dim: usize, num_heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let head_dim = model_dim / num_heads;
        let pair_bias_heads = (0..num_heads)
            .map(|head| nn::linear_no_bias(PAIR_DIM, 1, vb.pp(format!("pair_bias.{head}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            query: nn::linear_no_bias(model_dim, model_dim, vb.pp("query"))?,
            key: nn::linear_no_bias(model_dim, model_dim, vb.pp("key"))?,
            value: nn::linear_no_bias(model_dim, model_dim, vb.pp("value"))?,
            output: nn::linear(model_dim, model_dim, vb.pp("output"))?,
            pair_bias_heads,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden: &Tensor, pair: &Tensor, token_mask: &Tensor) -> Result<Tensor> {
        let (batch, tokens, model_dim) = hidden.dims3()?;
        let (pair_batch, pair_i, pair_j, pair_dim) = pair.dims4()?;
        if pair_batch != batch || pair_i != tokens || pair_j != tokens || pair_dim != PAIR_DIM {
            candle_core::bail!("foundation pair-conditioned attention shape mismatch");
        }

        let hidden = hidden.contiguous()?;
        let q = self
            .query
            .forward(&hidden)?
            .reshape((batch, tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .key
            .forward(&hidden)?
            .reshape((batch, tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .value
            .forward(&hidden)?
            .reshape((batch, tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;

        let pair_flat = pair
            .reshape((batch * tokens * tokens, PAIR_DIM))?
            .contiguous()?;
        let mut per_head = Vec::with_capacity(self.num_heads);
        for projection in &self.pair_bias_heads {
            per_head.push(
                projection
                    .forward(&pair_flat)?
                    .reshape((batch, tokens, tokens))?
                    .unsqueeze(1)?,
            );
        }
        let refs = per_head.iter().collect::<Vec<_>>();
        let pair_bias = Tensor::cat(&refs, 1)?;
        scores = (scores + pair_bias)?;

        let key_mask = token_mask
            .affine(-1.0, 1.0)?
            .affine(-10_000.0, 0.0)?
            .unsqueeze(1)?
            .unsqueeze(1)?
            .broadcast_as((batch, self.num_heads, tokens, tokens))?;
        let probabilities = ops::softmax(&(scores + key_mask)?, D::Minus1)?;
        let context = probabilities
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, tokens, model_dim))?
            .contiguous()?;
        let output = self.output.forward(&context)?;
        let query_mask = token_mask
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, model_dim))?;
        output.broadcast_mul(&query_mask)
    }
}

#[derive(Clone)]
struct ResiduePairInteractionBlock {
    residue_attention_norm: FoundationLayerNorm,
    pair_attention_norm: FoundationLayerNorm,
    attention: PairConditionedSelfAttention,
    residue_to_pair_norm: FoundationLayerNorm,
    pair_update_left: Linear,
    pair_update_right: Linear,
    pair_update_out: Linear,
    pair_transition_norm: FoundationLayerNorm,
    pair_transition_in: Linear,
    pair_transition_out: Linear,
    residue_transition_norm: FoundationLayerNorm,
    residue_transition_in: Linear,
    residue_transition_out: Linear,
    dropout: Dropout,
    model_dim: usize,
}

impl ResiduePairInteractionBlock {
    fn new(config: &FoundationConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            residue_attention_norm: FoundationLayerNorm::new(
                config.model_dim,
                1e-5,
                vb.pp("residue_attention_norm"),
            )?,
            pair_attention_norm: FoundationLayerNorm::new(
                PAIR_DIM,
                1e-5,
                vb.pp("pair_attention_norm"),
            )?,
            attention: PairConditionedSelfAttention::new(
                config.model_dim,
                config.num_attention_heads,
                vb.pp("attention"),
            )?,
            residue_to_pair_norm: FoundationLayerNorm::new(
                config.model_dim,
                1e-5,
                vb.pp("residue_to_pair_norm"),
            )?,
            pair_update_left: nn::linear(config.model_dim, PAIR_DIM, vb.pp("pair_update_left"))?,
            pair_update_right: nn::linear(config.model_dim, PAIR_DIM, vb.pp("pair_update_right"))?,
            pair_update_out: nn::linear(PAIR_DIM, PAIR_DIM, vb.pp("pair_update_out"))?,
            pair_transition_norm: FoundationLayerNorm::new(
                PAIR_DIM,
                1e-5,
                vb.pp("pair_transition_norm"),
            )?,
            pair_transition_in: nn::linear(PAIR_DIM, PAIR_FF_DIM, vb.pp("pair_transition_in"))?,
            pair_transition_out: nn::linear(PAIR_FF_DIM, PAIR_DIM, vb.pp("pair_transition_out"))?,
            residue_transition_norm: FoundationLayerNorm::new(
                config.model_dim,
                1e-5,
                vb.pp("residue_transition_norm"),
            )?,
            residue_transition_in: nn::linear(
                config.model_dim,
                config.transformer_ff_dim,
                vb.pp("residue_transition_in"),
            )?,
            residue_transition_out: nn::linear(
                config.transformer_ff_dim,
                config.model_dim,
                vb.pp("residue_transition_out"),
            )?,
            dropout: Dropout::new(config.dropout),
            model_dim: config.model_dim,
        })
    }

    fn forward_t(
        &self,
        hidden: &Tensor,
        pair: &Tensor,
        token_mask: &Tensor,
        pair_mask: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, tokens, model_dim) = hidden.dims3()?;
        if model_dim != self.model_dim {
            candle_core::bail!("foundation residue-pair block residue width mismatch");
        }
        let (_, pair_i, pair_j, pair_dim) = pair.dims4()?;
        if pair_i != tokens || pair_j != tokens || pair_dim != PAIR_DIM {
            candle_core::bail!("foundation residue-pair block pair shape mismatch");
        }

        let normalized_hidden = self.residue_attention_norm.forward(hidden)?;
        let normalized_pair = self.pair_attention_norm.forward(pair)?;
        let attention = self
            .attention
            .forward(&normalized_hidden, &normalized_pair, token_mask)?;
        let mut hidden = (hidden + self.dropout.forward_t(&attention, train)?)?;
        hidden = mask_token_state(&hidden, token_mask)?;

        let normalized_hidden = self.residue_to_pair_norm.forward(&hidden)?.contiguous()?;
        let left = self
            .pair_update_left
            .forward(&normalized_hidden)?
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, tokens, PAIR_DIM))?;
        let right = self
            .pair_update_right
            .forward(&normalized_hidden)?
            .unsqueeze(1)?
            .broadcast_as((batch, tokens, tokens, PAIR_DIM))?;
        let product = left.broadcast_mul(&right)?;
        let residue_pair = ((&left + &right)? + product)?.relu()?;
        let pair_update = self
            .pair_update_out
            .forward(
                &residue_pair
                    .reshape((batch * tokens * tokens, PAIR_DIM))?
                    .contiguous()?,
            )?
            .reshape((batch, tokens, tokens, PAIR_DIM))?;
        let mut pair = (pair + self.dropout.forward_t(&pair_update, train)?)?;
        pair = mask_pair_state(&pair, pair_mask)?;

        let normalized_pair = self.pair_transition_norm.forward(&pair)?;
        let flat = normalized_pair
            .reshape((batch * tokens * tokens, PAIR_DIM))?
            .contiguous()?;
        let pair_transition = self.pair_transition_in.forward(&flat)?.relu()?;
        let pair_transition = self
            .pair_transition_out
            .forward(&pair_transition)?
            .reshape((batch, tokens, tokens, PAIR_DIM))?;
        pair = (&pair + self.dropout.forward_t(&pair_transition, train)?)?;
        pair = mask_pair_state(&pair, pair_mask)?;

        let normalized_hidden = self
            .residue_transition_norm
            .forward(&hidden)?
            .contiguous()?;
        let residue_transition = self
            .residue_transition_in
            .forward(&normalized_hidden)?
            .relu()?;
        let residue_transition = self.residue_transition_out.forward(&residue_transition)?;
        hidden = (&hidden + self.dropout.forward_t(&residue_transition, train)?)?;
        hidden = mask_token_state(&hidden, token_mask)?;

        Ok((hidden, pair))
    }
}

#[derive(Clone)]
struct ProductionPairEncoder {
    config: FoundationConfig,
    atom_input: Linear,
    graph_layers: Vec<GraphMessageLayer>,
    graph_to_residue: Linear,
    chemistry_to_residue: Linear,
    residue_embedding: Embedding,
    position_embedding: Embedding,
    residue_input_norm: FoundationLayerNorm,
    task_embedding: Embedding,
    ccs_context_projection: Linear,
    task_instrument_embedding: Embedding,
    ms2_context_projection: Linear,
    pair_left: Linear,
    pair_right: Linear,
    pair_chemistry_left: Linear,
    pair_chemistry_right: Linear,
    pair_relative_projection: Linear,
    pair_input_norm: FoundationLayerNorm,
    interaction_blocks: Vec<ResiduePairInteractionBlock>,
    residue_output_norm: FoundationLayerNorm,
}

impl ProductionPairEncoder {
    fn new(config: FoundationConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let atom_input = nn::linear(
            ATOM_FEATURE_DIM,
            config.graph_hidden_dim,
            vb.pp("chemistry.atom_input"),
        )?;
        let graph_layers = (0..config.graph_layers)
            .map(|layer| {
                GraphMessageLayer::new(
                    config.graph_hidden_dim,
                    vb.pp(format!("chemistry.graph.{layer}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let graph_to_residue = nn::linear(
            config.graph_hidden_dim,
            config.model_dim,
            vb.pp("chemistry.graph_to_residue"),
        )?;
        let chemistry_to_residue = nn::linear(
            ATOM_FEATURE_DIM,
            config.model_dim,
            vb.pp("chemistry.raw_to_residue"),
        )?;
        let residue_embedding =
            nn::embedding(21, config.model_dim, vb.pp("sequence.residue_embedding"))?;
        let position_embedding = nn::embedding(
            config.max_sequence_len,
            config.model_dim,
            vb.pp("sequence.position_embedding"),
        )?;
        let residue_input_norm =
            FoundationLayerNorm::new(config.model_dim, 1e-5, vb.pp("sequence.input_norm"))?;
        let task_embedding = nn::embedding(TASK_COUNT, config.model_dim, vb.pp("task.embedding"))?;
        let ccs_context_projection = nn::linear(
            CCS_TASK_CONTEXT_DIM,
            config.model_dim,
            vb.pp("task.ccs_context"),
        )?;
        let task_instrument_embedding = nn::embedding(
            config.instrument_vocab_size,
            MS2_TASK_INSTRUMENT_DIM,
            vb.pp("task.ms2_instrument_embedding"),
        )?;
        let ms2_context_projection = nn::linear(
            MS2_TASK_INSTRUMENT_DIM + MS2_TASK_CONTEXT_SCALAR_DIM,
            config.model_dim,
            vb.pp("task.ms2_context"),
        )?;
        let pair_left = nn::linear(config.model_dim, PAIR_DIM, vb.pp("pair.init_left"))?;
        let pair_right = nn::linear(config.model_dim, PAIR_DIM, vb.pp("pair.init_right"))?;
        let pair_chemistry_left =
            nn::linear(ATOM_FEATURE_DIM, PAIR_DIM, vb.pp("pair.chemistry_left"))?;
        let pair_chemistry_right =
            nn::linear(ATOM_FEATURE_DIM, PAIR_DIM, vb.pp("pair.chemistry_right"))?;
        let pair_relative_projection = nn::linear(
            RELATIVE_FEATURE_DIM,
            PAIR_DIM,
            vb.pp("pair.relative_projection"),
        )?;
        let pair_input_norm = FoundationLayerNorm::new(PAIR_DIM, 1e-5, vb.pp("pair.input_norm"))?;
        let interaction_blocks = (0..config.transformer_layers)
            .map(|block| {
                ResiduePairInteractionBlock::new(&config, vb.pp(format!("interaction.{block}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let residue_output_norm =
            FoundationLayerNorm::new(config.model_dim, 1e-5, vb.pp("output.residue_norm"))?;

        Ok(Self {
            config,
            atom_input,
            graph_layers,
            graph_to_residue,
            chemistry_to_residue,
            residue_embedding,
            position_embedding,
            residue_input_norm,
            task_embedding,
            ccs_context_projection,
            task_instrument_embedding,
            ms2_context_projection,
            pair_left,
            pair_right,
            pair_chemistry_left,
            pair_chemistry_right,
            pair_relative_projection,
            pair_input_norm,
            interaction_blocks,
            residue_output_norm,
        })
    }

    fn forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<PairRepresentation> {
        let (batch_size, sequence_len, _, feature_dim) = batch.atom_features.dims4()?;
        if sequence_len != self.config.max_sequence_len {
            candle_core::bail!(
                "foundation pair encoder expected sequence width {}, got {}",
                self.config.max_sequence_len,
                sequence_len
            );
        }
        if feature_dim != ATOM_FEATURE_DIM {
            candle_core::bail!(
                "foundation pair encoder atom feature mismatch: expected {}, got {}",
                ATOM_FEATURE_DIM,
                feature_dim
            );
        }

        let chemistry_targets = mean_raw_chemistry(batch)?;
        let mut residues = self.encode_residues(batch, &chemistry_targets)?;
        residues = mask_token_state(&residues, &batch.residue_mask)?;
        let task_tokens = self.contextual_task_tokens(batch_size, context)?;
        let mut hidden = Tensor::cat(&[&task_tokens, &residues], 1)?.contiguous()?;
        let task_mask = Tensor::ones((batch_size, TASK_COUNT), DType::F32, hidden.device())?;
        let token_mask = Tensor::cat(&[&task_mask, &batch.residue_mask], 1)?;
        let pair_mask = token_mask
            .unsqueeze(2)?
            .broadcast_mul(&token_mask.unsqueeze(1)?)?;
        let mut pair = self.initialize_pair_state(&hidden, &chemistry_targets, &pair_mask)?;
        for block in &self.interaction_blocks {
            (hidden, pair) = block.forward_t(&hidden, &pair, &token_mask, &pair_mask, train)?;
        }
        hidden = self.residue_output_norm.forward(&hidden)?;
        hidden = mask_token_state(&hidden, &token_mask)?;

        let rt_embedding = hidden.narrow(1, TASK_RT, 1)?.squeeze(1)?.contiguous()?;
        let ccs_embedding = hidden.narrow(1, TASK_CCS, 1)?.squeeze(1)?.contiguous()?;
        let ms2_embedding = hidden.narrow(1, TASK_MS2, 1)?.squeeze(1)?.contiguous()?;
        let global_embedding = hidden.narrow(1, TASK_GLOBAL, 1)?.squeeze(1)?.contiguous()?;
        let residue_embeddings = hidden
            .narrow(1, TASK_COUNT, self.config.max_sequence_len)?
            .contiguous()?;

        Ok(PairRepresentation {
            foundation: FoundationOutput {
                residue_embeddings,
                peptide_embedding: global_embedding,
                residue_mask: batch.residue_mask.clone(),
                chemistry_targets,
            },
            rt_embedding,
            ccs_embedding,
            ms2_embedding,
            pair_embeddings: pair,
        })
    }

    fn encode_residues(
        &self,
        batch: &FoundationBatch,
        chemistry_targets: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, atom_count, feature_dim) = batch.atom_features.dims4()?;
        let graph_count = batch_size * sequence_len;
        let atoms = batch
            .atom_features
            .reshape((graph_count, atom_count, feature_dim))?;
        let adjacency = batch
            .adjacency
            .reshape((graph_count, atom_count, atom_count))?;
        let atom_mask = batch.atom_mask.reshape((graph_count, atom_count))?;
        let mut hidden = self.atom_input.forward(&atoms)?;
        hidden = hidden.broadcast_mul(&atom_mask.unsqueeze(2)?.broadcast_as((
            graph_count,
            atom_count,
            self.config.graph_hidden_dim,
        ))?)?;
        for layer in &self.graph_layers {
            hidden = layer.forward(&hidden, &adjacency, &atom_mask)?;
        }
        let atom_sum = hidden.sum(1)?;
        let atom_denominator = atom_mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
        let graph_residue = self
            .graph_to_residue
            .forward(&atom_sum.broadcast_div(&atom_denominator)?)?
            .reshape((batch_size, sequence_len, self.config.model_dim))?;
        let raw_chemistry = self.chemistry_to_residue.forward(chemistry_targets)?;
        let identity = self.residue_embedding.forward(&batch.residue_ids)?;
        let positions: Vec<u32> = (0..sequence_len as u32).collect();
        let position_ids = Tensor::from_vec(positions, sequence_len, batch.residue_ids.device())?
            .to_dtype(DType::U32)?;
        let position = self
            .position_embedding
            .forward(&position_ids)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, sequence_len, self.config.model_dim))?;
        let residue = ((graph_residue + raw_chemistry)? + identity)?;
        self.residue_input_norm.forward(&(residue + position)?)
    }

    fn contextual_task_tokens(
        &self,
        batch_size: usize,
        context: &PrecursorContextBatch,
    ) -> Result<Tensor> {
        let device = context.charge.device();
        let ids = Tensor::from_vec(
            vec![
                TASK_RT as u32,
                TASK_CCS as u32,
                TASK_MS2 as u32,
                TASK_GLOBAL as u32,
            ],
            TASK_COUNT,
            device,
        )?
        .to_dtype(DType::U32)?;
        let tasks = self
            .task_embedding
            .forward(&ids)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, TASK_COUNT, self.config.model_dim))?;

        let known_charge = context.charge.broadcast_mul(&context.charge_present)?;
        let known_mz = context
            .precursor_mz
            .broadcast_mul(&context.precursor_mz_present)?;
        let physical_present = context
            .charge_present
            .broadcast_mul(&context.precursor_mz_present)?;
        let neutral_mass_proxy = known_charge
            .broadcast_mul(&known_mz)?
            .broadcast_mul(&physical_present)?;
        let ccs_context = Tensor::cat(
            &[
                &known_charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?,
                &context.charge_present.unsqueeze(1)?,
                &known_mz.affine(1.0 / 1000.0, 0.0)?.unsqueeze(1)?,
                &context.precursor_mz_present.unsqueeze(1)?,
                &neutral_mass_proxy.affine(1.0 / 3000.0, 0.0)?.unsqueeze(1)?,
                &physical_present.unsqueeze(1)?,
            ],
            1,
        )?;
        let ccs_context = self
            .ccs_context_projection
            .forward(&ccs_context)?
            .unsqueeze(1)?;

        let instrument = self
            .task_instrument_embedding
            .forward(&context.instrument_ids)?;
        let ms2_scalars = Tensor::cat(
            &[
                &known_charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?,
                &context.charge_present.unsqueeze(1)?,
                &context.nce.affine(1.0 / 100.0, 0.0)?.unsqueeze(1)?,
                &context.nce_present.unsqueeze(1)?,
                &context.instrument_present.unsqueeze(1)?,
            ],
            1,
        )?;
        let ms2_context = self
            .ms2_context_projection
            .forward(&Tensor::cat(&[&instrument, &ms2_scalars], 1)?)?
            .unsqueeze(1)?;

        let rt = tasks.narrow(1, TASK_RT, 1)?;
        let ccs = (tasks.narrow(1, TASK_CCS, 1)? + ccs_context)?;
        let ms2 = (tasks.narrow(1, TASK_MS2, 1)? + ms2_context)?;
        let global = tasks.narrow(1, TASK_GLOBAL, 1)?;
        Tensor::cat(&[&rt, &ccs, &ms2, &global], 1)
    }

    fn initialize_pair_state(
        &self,
        hidden: &Tensor,
        chemistry_targets: &Tensor,
        pair_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch, tokens, _) = hidden.dims3()?;
        let sequence = self.config.max_sequence_len;
        let hidden = hidden.contiguous()?;
        let left = self
            .pair_left
            .forward(&hidden)?
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, tokens, PAIR_DIM))?;
        let right = self
            .pair_right
            .forward(&hidden)?
            .unsqueeze(1)?
            .broadcast_as((batch, tokens, tokens, PAIR_DIM))?;

        let chemistry_left = self.pair_chemistry_left.forward(chemistry_targets)?;
        let chemistry_right = self.pair_chemistry_right.forward(chemistry_targets)?;
        let task_zeros = Tensor::zeros((batch, TASK_COUNT, PAIR_DIM), DType::F32, hidden.device())?;
        let chemistry_left = Tensor::cat(&[&task_zeros, &chemistry_left], 1)?
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, tokens, PAIR_DIM))?;
        let chemistry_right = Tensor::cat(&[&task_zeros, &chemistry_right], 1)?
            .unsqueeze(1)?
            .broadcast_as((batch, tokens, tokens, PAIR_DIM))?;

        let relative = relative_pair_features(sequence, hidden.device())?;
        let relative = self
            .pair_relative_projection
            .forward(
                &relative
                    .reshape((tokens * tokens, RELATIVE_FEATURE_DIM))?
                    .contiguous()?,
            )?
            .reshape((1, tokens, tokens, PAIR_DIM))?
            .broadcast_as((batch, tokens, tokens, PAIR_DIM))?;
        let pair = (((left + right)? + chemistry_left)? + chemistry_right)?;
        let pair = self.pair_input_norm.forward(&(pair + relative)?)?;
        mask_pair_state(&pair, pair_mask)
    }
}

#[derive(Clone)]
struct NativeRtSpecialist {
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
}

impl NativeRtSpecialist {
    fn new(model_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            hidden: nn::linear(2 * model_dim + 6, 640, vb.pp("hidden"))?,
            bottleneck: nn::linear(640, 320, vb.pp("bottleneck"))?,
            output: nn::linear(320, 1, vb.pp("output"))?,
        })
    }

    fn forward(
        &self,
        representation: &PairRepresentation,
        context: &PrecursorContextBatch,
    ) -> Result<Tensor> {
        let scalar = specialist_scalar_context(context)?;
        let features = Tensor::cat(
            &[
                &representation.rt_embedding,
                &representation.foundation.peptide_embedding,
                &scalar,
            ],
            1,
        )?;
        let hidden = self.hidden.forward(&features)?.relu()?;
        let hidden = self.bottleneck.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)
    }
}

#[derive(Clone)]
struct NativeMs2Specialist {
    instrument_embedding: Embedding,
    context_projection: Linear,
    input_norm: FoundationLayerNorm,
    context_blocks: Vec<PeptideTransformerBlock>,
    fragment_blocks: Vec<PeptideTransformerBlock>,
    fragment_projection: Linear,
    presence_head: Linear,
    intensity_head: Linear,
    model_dim: usize,
    channels: usize,
}

impl NativeMs2Specialist {
    fn new(config: &FoundationConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let context_blocks = (0..3)
            .map(|index| {
                PeptideTransformerBlock::new(
                    config.model_dim,
                    config.num_attention_heads,
                    config.transformer_ff_dim,
                    config.dropout,
                    vb.pp(format!("context_transformer.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let fragment_blocks = (0..2)
            .map(|index| {
                PeptideTransformerBlock::new(
                    config.model_dim,
                    config.num_attention_heads,
                    config.transformer_ff_dim,
                    config.dropout,
                    vb.pp(format!("fragment_transformer.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            instrument_embedding: nn::embedding(
                config.instrument_vocab_size,
                32,
                vb.pp("instrument_embedding"),
            )?,
            context_projection: nn::linear(38, config.model_dim, vb.pp("context_projection"))?,
            input_norm: FoundationLayerNorm::new(config.model_dim, 1e-5, vb.pp("input_norm"))?,
            context_blocks,
            fragment_blocks,
            fragment_projection: nn::linear(
                3 * config.model_dim,
                config.model_dim,
                vb.pp("fragment_projection"),
            )?,
            presence_head: nn::linear(
                config.model_dim,
                config.ms2_fragment_channels,
                vb.pp("presence"),
            )?,
            intensity_head: nn::linear(
                config.model_dim,
                config.ms2_fragment_channels,
                vb.pp("intensity"),
            )?,
            model_dim: config.model_dim,
            channels: config.ms2_fragment_channels,
        })
    }

    fn forward_t(
        &self,
        representation: &PairRepresentation,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        activation: super::config::FoundationMs2OutputActivation,
        train: bool,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, model_dim) =
            representation.foundation.residue_embeddings.dims3()?;
        let instrument = self.instrument_embedding.forward(&context.instrument_ids)?;
        let scalar = specialist_scalar_context(context)?;
        let context_features = Tensor::cat(&[&instrument, &scalar], 1)?;
        let context_hidden = self.context_projection.forward(&context_features)?;
        let context_hidden =
            context_hidden
                .unsqueeze(1)?
                .broadcast_as((batch_size, sequence_len, model_dim))?;
        let mut residues = (&representation.foundation.residue_embeddings + &context_hidden)?;
        residues = self.input_norm.forward(&residues)?;
        residues = residues.broadcast_mul(&batch.residue_mask.unsqueeze(2)?.broadcast_as((
            batch_size,
            sequence_len,
            model_dim,
        ))?)?;
        for block in &self.context_blocks {
            residues = block.forward_t(&residues, &batch.residue_mask, train)?;
        }
        if sequence_len < 2 {
            candle_core::bail!("foundation contextual MS2 requires at least two residues");
        }
        let cleavages = sequence_len - 1;
        let left = residues.narrow(1, 0, cleavages)?;
        let right = residues.narrow(1, 1, cleavages)?;
        let task = representation.ms2_embedding.unsqueeze(1)?.broadcast_as((
            batch_size,
            cleavages,
            self.model_dim,
        ))?;
        let token_features = Tensor::cat(&[&left, &right, &task], 2)?.contiguous()?;
        let mut tokens = self
            .fragment_projection
            .forward(
                &token_features
                    .reshape((batch_size * cleavages, 3 * self.model_dim))?
                    .contiguous()?,
            )?
            .reshape((batch_size, cleavages, self.model_dim))?;
        let cleavage_mask = batch
            .residue_mask
            .narrow(1, 0, cleavages)?
            .broadcast_mul(&batch.residue_mask.narrow(1, 1, cleavages)?)?;
        for block in &self.fragment_blocks {
            tokens = block.forward_t(&tokens, &cleavage_mask, train)?;
        }
        let presence_logits = self.presence_head.forward(&tokens)?;
        let intensity_logits = self.intensity_head.forward(&tokens)?;
        let presence = ops::sigmoid(&presence_logits)?;
        let positive = apply_ms2_output_activation(&intensity_logits, activation)?;
        let prediction = presence.broadcast_mul(&positive)?;
        prediction.broadcast_mul(&cleavage_mask.unsqueeze(2)?.broadcast_as((
            batch_size,
            cleavages,
            self.channels,
        ))?)
    }
}

#[derive(Clone)]
struct PairAwareMobilitySpecialist {
    physics_projection: Linear,
    physics_pair_gate: Linear,
    mobility_to_residue_projection: Linear,
    residue_to_mobility_projection: Linear,
    residue_pair_projection: Linear,
    pair_global_projection: Linear,
    input_norm: FoundationLayerNorm,
    blocks: Vec<PeptideTransformerBlock>,
    hidden: Linear,
    bottleneck: Linear,
    output: Linear,
    conformation_proxy: Option<Linear>,
    model_dim: usize,
}

impl PairAwareMobilitySpecialist {
    fn new(config: &FoundationConfig, auxiliary_heads: bool, vb: VarBuilder<'_>) -> Result<Self> {
        let blocks = (0..2)
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
        Ok(Self {
            physics_projection: nn::linear(6, config.model_dim, vb.pp("physics_projection"))?,
            physics_pair_gate: nn::linear(6, PAIR_DIM, vb.pp("physics_pair_gate"))?,
            mobility_to_residue_projection: nn::linear(
                PAIR_DIM,
                config.model_dim,
                vb.pp("mobility_to_residue"),
            )?,
            residue_to_mobility_projection: nn::linear(
                PAIR_DIM,
                config.model_dim,
                vb.pp("residue_to_mobility"),
            )?,
            residue_pair_projection: nn::linear(PAIR_DIM, config.model_dim, vb.pp("residue_pair"))?,
            pair_global_projection: nn::linear(PAIR_DIM, config.model_dim, vb.pp("pair_global"))?,
            input_norm: FoundationLayerNorm::new(config.model_dim, 1e-5, vb.pp("input_norm"))?,
            blocks,
            hidden: nn::linear(4 * config.model_dim + 6, 640, vb.pp("hidden"))?,
            bottleneck: nn::linear(640, 320, vb.pp("bottleneck"))?,
            output: nn::linear(320, 1, vb.pp("output"))?,
            conformation_proxy: auxiliary_heads
                .then(|| nn::linear(320, 14, vb.pp("conformation_proxy")))
                .transpose()?,
            model_dim: config.model_dim,
        })
    }

    fn forward_with_proxy_t(
        &self,
        representation: &PairRepresentation,
        context: &PrecursorContextBatch,
        train: bool,
        need_auxiliary: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let residues = &representation.foundation.residue_embeddings;
        let mask = &representation.foundation.residue_mask;
        let pair = &representation.pair_embeddings;
        let (batch, sequence, model_dim) = residues.dims3()?;
        let expected_tokens = TASK_COUNT + sequence;
        let (_, pair_left, pair_right, pair_dim) = pair.dims4()?;
        if model_dim != self.model_dim
            || pair_left != expected_tokens
            || pair_right != expected_tokens
            || pair_dim != PAIR_DIM
        {
            candle_core::bail!("foundation pair-aware mobility specialist shape mismatch");
        }
        let mobility_to_residue = pair
            .narrow(1, TASK_CCS, 1)?
            .squeeze(1)?
            .narrow(1, TASK_COUNT, sequence)?
            .contiguous()?;
        let residue_to_mobility = pair
            .narrow(1, TASK_COUNT, sequence)?
            .narrow(2, TASK_CCS, 1)?
            .squeeze(2)?
            .contiguous()?;
        let residue_pair = pair
            .narrow(1, TASK_COUNT, sequence)?
            .narrow(2, TASK_COUNT, sequence)?
            .contiguous()?;
        let partner_count = mask
            .sum(1)?
            .clamp(1.0, f64::INFINITY)?
            .unsqueeze(1)?
            .unsqueeze(2)?;
        let pair_summary = residue_pair
            .sum(2)?
            .broadcast_div(&partner_count.broadcast_as((batch, sequence, 1))?)?
            .broadcast_mul(
                &mask
                    .unsqueeze(2)?
                    .broadcast_as((batch, sequence, PAIR_DIM))?,
            )?;
        let physics = specialist_scalar_context(context)?;
        let gate = ops::sigmoid(&self.physics_pair_gate.forward(&physics)?)?
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, PAIR_DIM))?;
        let pair_summary = pair_summary.broadcast_mul(&gate)?;
        let mobility_out = self
            .mobility_to_residue_projection
            .forward(&mobility_to_residue.broadcast_mul(&gate)?.contiguous()?)?;
        let mobility_in = self
            .residue_to_mobility_projection
            .forward(&residue_to_mobility.broadcast_mul(&gate)?.contiguous()?)?;
        let pair_residue = self
            .residue_pair_projection
            .forward(&pair_summary.contiguous()?)?;
        let physics_residue = self
            .physics_projection
            .forward(&physics)?
            .unsqueeze(1)?
            .broadcast_as((batch, sequence, self.model_dim))?;
        let mut hidden = (((residues + &pair_residue)? + &mobility_out)? + &mobility_in)?;
        hidden = (hidden + physics_residue)?;
        hidden = self.input_norm.forward(&hidden)?;
        hidden = hidden.broadcast_mul(&mask.unsqueeze(2)?.broadcast_as((
            batch,
            sequence,
            self.model_dim,
        ))?)?;
        for block in &self.blocks {
            hidden = block.forward_t(&hidden, mask, train)?;
        }
        let pooled = masked_mean_specialist(&hidden, mask)?;
        let pair_global = self
            .pair_global_projection
            .forward(&masked_mean_specialist(&pair_summary, mask)?)?;
        let features = Tensor::cat(
            &[
                &representation.ccs_embedding,
                &representation.foundation.peptide_embedding,
                &pooled,
                &pair_global,
                &physics,
            ],
            1,
        )?;
        let hidden = self.hidden.forward(&features)?.relu()?;
        let hidden = self.bottleneck.forward(&hidden)?.relu()?;
        let ccs = self.output.forward(&hidden)?;
        let proxy = if need_auxiliary {
            self.conformation_proxy
                .as_ref()
                .map(|head| head.forward(&hidden))
                .transpose()?
        } else {
            None
        };
        Ok((ccs, proxy))
    }
}

fn specialist_scalar_context(context: &PrecursorContextBatch) -> Result<Tensor> {
    let scaled_charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
    let scaled_mz = context
        .precursor_mz
        .affine(1.0 / 2000.0, 0.0)?
        .unsqueeze(1)?;
    let scaled_nce = context.nce.affine(1.0 / 50.0, 0.0)?.unsqueeze(1)?;
    Tensor::cat(
        &[
            &scaled_charge,
            &context.charge_present.unsqueeze(1)?,
            &scaled_mz,
            &context.precursor_mz_present.unsqueeze(1)?,
            &scaled_nce,
            &context.nce_present.unsqueeze(1)?,
        ],
        1,
    )
}

fn masked_mean_specialist(hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, sequence, dim) = hidden.dims3()?;
    let expanded = mask.unsqueeze(2)?.broadcast_as((batch, sequence, dim))?;
    let numerator = hidden.broadcast_mul(&expanded)?.sum(1)?;
    let denominator = mask.sum(1)?.clamp(1.0, f64::INFINITY)?.unsqueeze(1)?;
    numerator.broadcast_div(&denominator)
}

/// Version-free production peptide-property model using the residue-pair/task-token backbone.
#[derive(Clone)]
pub(crate) struct PairTaskPeptideModel {
    encoder: ProductionPairEncoder,
    rt_head_hidden: Linear,
    rt_head_output: Linear,
    ccs_head_hidden: Linear,
    ccs_head_output: Linear,
    ms2_head_hidden: Linear,
    ms2_head_output: Linear,
    residue_head: Linear,
    chemistry_head: Linear,
    contrastive_head: Linear,
    pair_class_head: Option<Linear>,
    chemistry_summary_head: Option<Linear>,
    rt_specialist: Option<NativeRtSpecialist>,
    ms2_specialist: Option<NativeMs2Specialist>,
    mobility_specialist: Option<PairAwareMobilitySpecialist>,
    specialist_config: FoundationSpecialistConfig,
    config: FoundationConfig,
}

impl PairTaskPeptideModel {
    pub(crate) fn new(
        config: FoundationConfig,
        specialist_config: FoundationSpecialistConfig,
        auxiliary_heads: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        let encoder = ProductionPairEncoder::new(config.clone(), vb.pp("encoder"))?;
        let ccs_context_dim = 2;
        Ok(Self {
            rt_head_hidden: nn::linear(
                config.model_dim,
                config.model_dim,
                vb.pp("heads.rt.hidden"),
            )?,
            rt_head_output: nn::linear(config.model_dim, 1, vb.pp("heads.rt.output"))?,
            ccs_head_hidden: nn::linear(
                config.model_dim + ccs_context_dim,
                config.model_dim,
                vb.pp("heads.ccs.hidden"),
            )?,
            ccs_head_output: nn::linear(config.model_dim, 1, vb.pp("heads.ccs.output"))?,
            ms2_head_hidden: nn::linear(
                3 * config.model_dim,
                config.transformer_ff_dim,
                vb.pp("heads.ms2.hidden"),
            )?,
            ms2_head_output: nn::linear(
                config.transformer_ff_dim,
                config.ms2_fragment_channels,
                vb.pp("heads.ms2.output"),
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
            pair_class_head: auxiliary_heads
                .then(|| nn::linear(PAIR_DIM, 6, vb.pp("heads.pair_classes")))
                .transpose()?,
            chemistry_summary_head: auxiliary_heads
                .then(|| nn::linear(config.model_dim, 8, vb.pp("heads.chemistry_summary")))
                .transpose()?,
            rt_specialist: (specialist_config.enabled && specialist_config.rt)
                .then(|| NativeRtSpecialist::new(config.model_dim, vb.pp("specialists.rt")))
                .transpose()?,
            ms2_specialist: (specialist_config.enabled && specialist_config.ms2)
                .then(|| NativeMs2Specialist::new(&config, vb.pp("specialists.ms2")))
                .transpose()?,
            mobility_specialist: (specialist_config.enabled && specialist_config.mobility_ccs)
                .then(|| {
                    PairAwareMobilitySpecialist::new(
                        &config,
                        auxiliary_heads,
                        vb.pp("specialists.mobility_ccs"),
                    )
                })
                .transpose()?,
            specialist_config,
            encoder,
            config,
        })
    }

    pub(crate) fn forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationMultiTaskOutput> {
        self.forward_impl(batch, context, train, false)
            .map(|(main, _)| main)
    }

    pub(crate) fn forward_with_auxiliaries_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<(
        FoundationMultiTaskOutput,
        Option<PairTaskAuxiliaryPredictions>,
    )> {
        self.forward_impl(batch, context, train, true)
    }

    fn forward_impl(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
        need_auxiliary: bool,
    ) -> Result<(
        FoundationMultiTaskOutput,
        Option<PairTaskAuxiliaryPredictions>,
    )> {
        let representation = self.encoder.forward_t(batch, context, train)?;
        let rt = if let Some(specialist) = &self.rt_specialist {
            specialist.forward(&representation, context)?
        } else {
            self.rt_head_output.forward(
                &self
                    .rt_head_hidden
                    .forward(&representation.rt_embedding)?
                    .relu()?,
            )?
        };

        let scaled_charge = context.charge.affine(1.0 / 6.0, 0.0)?.unsqueeze(1)?;
        let ccs_scalar_context = match self.config.ccs_context_mode {
            FoundationCcsContextMode::ChargePresence => {
                Tensor::cat(&[&scaled_charge, &context.charge_present.unsqueeze(1)?], 1)?
            }
            FoundationCcsContextMode::NeutralMassCharge => {
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
        let ccs_features = Tensor::cat(&[&representation.ccs_embedding, &ccs_scalar_context], 1)?;
        let ccs_residual = self
            .ccs_head_output
            .forward(&self.ccs_head_hidden.forward(&ccs_features)?.relu()?)?;
        let legacy_ccs = if let Some(physics_baseline) = &self.config.ccs_physics_baseline {
            let baseline = standardized_ccs_physics_baseline(
                &representation.foundation,
                context,
                physics_baseline,
            )?;
            (&baseline + &ccs_residual)?
        } else {
            ccs_residual
        };
        let (ccs, conformation_proxy) = if let Some(specialist) = &self.mobility_specialist {
            specialist.forward_with_proxy_t(&representation, context, train, need_auxiliary)?
        } else {
            (legacy_ccs, None)
        };

        let ms2 = if let Some(specialist) = &self.ms2_specialist {
            specialist.forward_t(
                &representation,
                batch,
                context,
                self.config.ms2_output_activation,
                train,
            )?
        } else {
            self.forward_ms2(&representation, batch)?
        };
        let residue_logits = self
            .residue_head
            .forward(&representation.foundation.residue_embeddings)?;
        let chemistry_reconstruction = self
            .chemistry_head
            .forward(&representation.foundation.residue_embeddings)?;
        let contrastive_projection = self
            .contrastive_head
            .forward(&representation.foundation.peptide_embedding)?;

        // Construct auxiliary tensors before moving the shared representation
        // into the public prediction struct. Old checkpoints have no such heads.
        let auxiliaries = if need_auxiliary {
            if let (Some(pair_head), Some(chemistry_head)) =
                (&self.pair_class_head, &self.chemistry_summary_head)
            {
                let (batch_size, seq_len, _) =
                    representation.foundation.residue_embeddings.dims3()?;
                let pair = representation
                    .pair_embeddings
                    .narrow(1, TASK_COUNT, seq_len)?
                    .narrow(2, TASK_COUNT, seq_len)?
                    .contiguous()?;
                let pair_logits = pair_head
                    .forward(&pair.reshape((batch_size * seq_len * seq_len, PAIR_DIM))?)?
                    .reshape((batch_size, seq_len, seq_len, 6))?;
                let chemistry_summary =
                    chemistry_head.forward(&representation.foundation.peptide_embedding)?;
                Some(PairTaskAuxiliaryPredictions {
                    pair_logits,
                    chemistry_summary,
                    conformation_proxy,
                })
            } else {
                None
            }
        } else {
            None
        };
        Ok((
            FoundationMultiTaskOutput {
                foundation: representation.foundation,
                rt,
                ccs,
                ms2,
                residue_logits,
                chemistry_reconstruction,
                contrastive_projection,
            },
            auxiliaries,
        ))
    }

    fn forward_ms2(
        &self,
        representation: &PairRepresentation,
        batch: &FoundationBatch,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, model_dim) =
            representation.foundation.residue_embeddings.dims3()?;
        if sequence_len < 2 {
            candle_core::bail!(
                "foundation pair MS2 forward requires at least two sequence positions"
            );
        }
        let cleavages = sequence_len - 1;
        let left = representation
            .foundation
            .residue_embeddings
            .narrow(1, 0, cleavages)?;
        let right = representation
            .foundation
            .residue_embeddings
            .narrow(1, 1, cleavages)?;
        let task = representation
            .ms2_embedding
            .unsqueeze(1)?
            .broadcast_as((batch_size, cleavages, model_dim))?;
        let features = Tensor::cat(&[&left, &right, &task], 2)?.contiguous()?;
        let hidden = self
            .ms2_head_hidden
            .forward(
                &features
                    .reshape((batch_size * cleavages, 3 * model_dim))?
                    .contiguous()?,
            )?
            .relu()?;
        let logits = self.ms2_head_output.forward(&hidden)?.reshape((
            batch_size,
            cleavages,
            self.config.ms2_fragment_channels,
        ))?;
        let prediction = apply_ms2_output_activation(&logits, self.config.ms2_output_activation)?;
        let cleavage_mask = batch
            .residue_mask
            .narrow(1, 0, cleavages)?
            .broadcast_mul(&batch.residue_mask.narrow(1, 1, cleavages)?)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, cleavages, self.config.ms2_fragment_channels))?;
        prediction.broadcast_mul(&cleavage_mask)
    }
}

fn mean_raw_chemistry(batch: &FoundationBatch) -> Result<Tensor> {
    let raw_sum = batch.atom_features.sum(2)?;
    let denominator = batch
        .atom_mask
        .sum(2)?
        .clamp(1.0, f64::INFINITY)?
        .unsqueeze(2)?;
    raw_sum.broadcast_div(&denominator)
}

fn mask_token_state(hidden: &Tensor, token_mask: &Tensor) -> Result<Tensor> {
    let (batch, tokens, dim) = hidden.dims3()?;
    hidden.broadcast_mul(
        &token_mask
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, dim))?,
    )
}

fn mask_pair_state(pair: &Tensor, pair_mask: &Tensor) -> Result<Tensor> {
    let (batch, left, right, dim) = pair.dims4()?;
    pair.broadcast_mul(
        &pair_mask
            .unsqueeze(3)?
            .broadcast_as((batch, left, right, dim))?,
    )
}

fn relative_pair_features(sequence_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let tokens = TASK_COUNT + sequence_len;
    let denom = sequence_len.max(1) as f32;
    let mut values = Vec::with_capacity(tokens * tokens * RELATIVE_FEATURE_DIM);
    for i in 0..tokens {
        for j in 0..tokens {
            let i_residue = i >= TASK_COUNT;
            let j_residue = j >= TASK_COUNT;
            let i_pos = if i_residue {
                (i - TASK_COUNT) as f32
            } else {
                0.0
            };
            let j_pos = if j_residue {
                (j - TASK_COUNT) as f32
            } else {
                0.0
            };
            let signed = if i_residue && j_residue {
                (j_pos - i_pos) / denom
            } else {
                0.0
            };
            values.extend_from_slice(&[
                signed,
                signed.abs(),
                i_pos / denom,
                j_pos / denom,
                if i == j { 1.0 } else { 0.0 },
                if i_residue && j_residue { 1.0 } else { 0.0 },
            ]);
        }
    }
    Tensor::from_vec(values, (1, tokens, tokens, RELATIVE_FEATURE_DIM), device)
}

#[cfg(test)]
mod tests {
    use super::super::featurize::{
        FoundationModification, PeptideGraphFeaturizer, PeptidoformInput,
    };
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    fn smoke_config() -> FoundationConfig {
        let mut config = FoundationConfig::default();
        config.max_sequence_len = 12;
        config.graph_hidden_dim = 32;
        config.graph_layers = 1;
        config.model_dim = 64;
        config.num_attention_heads = 4;
        config.transformer_ff_dim = 128;
        config.transformer_layers = 1;
        config.dropout = 0.0;
        config.contrastive_dim = 32;
        config
    }

    #[test]
    fn relative_pair_features_cover_tasks_and_residues() {
        let features = relative_pair_features(8, &Device::Cpu).unwrap();
        assert_eq!(
            features.dims4().unwrap(),
            (1, TASK_COUNT + 8, TASK_COUNT + 8, RELATIVE_FEATURE_DIM)
        );
    }

    #[test]
    fn pair_task_model_forward_shapes_are_consistent() -> Result<()> {
        let device = Device::Cpu;
        let config = smoke_config();
        let featurizer = PeptideGraphFeaturizer::new(config.clone())?;
        let mut modified = PeptidoformInput::unmodified("PEPTIDE");
        modified
            .modifications
            .push(FoundationModification::mass_delta(2, 15.9949));
        let batch =
            featurizer.featurize(&[modified, PeptidoformInput::unmodified("ACDK")], &device)?;
        let context = PrecursorContextBatch::unknown(2, &device)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PairTaskPeptideModel::new(
            config.clone(),
            FoundationSpecialistConfig::default(),
            true,
            vb,
        )?;
        let (output, auxiliaries) = model.forward_with_auxiliaries_t(&batch, &context, false)?;
        let auxiliaries = auxiliaries.expect("enabled pair/chemistry heads");
        assert_eq!(
            auxiliaries.pair_logits.dims4()?,
            (2, config.max_sequence_len, config.max_sequence_len, 6)
        );
        assert_eq!(auxiliaries.chemistry_summary.dims2()?, (2, 8));
        assert_eq!(
            auxiliaries.conformation_proxy.as_ref().unwrap().dims2()?,
            (2, 14)
        );

        assert_eq!(output.rt.dims2()?, (2, 1));
        assert_eq!(output.ccs.dims2()?, (2, 1));
        assert_eq!(
            output.ms2.dims3()?,
            (2, config.max_sequence_len - 1, config.ms2_fragment_channels)
        );
        assert_eq!(
            output.foundation.residue_embeddings.dims3()?,
            (2, config.max_sequence_len, config.model_dim)
        );
        assert_eq!(
            output.contrastive_projection.dims2()?,
            (2, config.contrastive_dim)
        );
        Ok(())
    }
}
