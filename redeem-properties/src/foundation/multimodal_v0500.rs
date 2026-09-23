//! ReDeeM v0.50 deep chemistry / residue-pair foundation encoder.
//!
//! v0.50 is intentionally a new student representation rather than a shape-compatible
//! continuation of v0.35.  The student combines a deeper residue-local atom graph with an
//! explicit O(L^2) residue-pair state and learned task tokens that participate in the same
//! interaction stack.  The accepted v0.35 model remains external to the student checkpoint and
//! is exposed only through a detached teacher adapter.

use super::chemistry::ATOM_FEATURE_DIM;
use super::config::FoundationConfig;
use super::featurize::FoundationBatch;
use super::layers::{FoundationLayerNorm, GraphMessageLayer};
use super::model::PrecursorContextBatch;
use super::multimodal_v0350::{
    FoundationFragmentContextBatchV0350, PeptideFoundationMultimodalV0350Model,
};
use candle_core::{DType, Module, ModuleT, Result, Tensor, D};
use candle_nn::{self as nn, ops, Dropout, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

pub const FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500: &str =
    "deep_chemistry_residue_pair_foundation_v0500";
pub const FOUNDATION_V0500_STUDENT_NAMESPACE: &str = "student_v050";
pub const FOUNDATION_V0500_TEACHER_SOURCE: &str = "external_frozen_v0350";
pub const FOUNDATION_V0500_TASK_COUNT: usize = 4;
pub const FOUNDATION_V0500_PAIR_CLASS_COUNT: usize = 6;
pub const FOUNDATION_V0500_RELATIVE_FEATURE_DIM: usize = 6;
pub const FOUNDATION_V0500_MOBILITY_CONTEXT_DIM: usize = 6;
pub const FOUNDATION_V0500_MS2_CONTEXT_SCALAR_DIM: usize = 5;

const TASK_RT: usize = 0;
const TASK_MOBILITY: usize = 1;
const TASK_MS2: usize = 2;
const TASK_GLOBAL: usize = 3;

/// Configuration for the v0.50 student encoder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PeptideFoundationV0500Config {
    pub max_sequence_len: usize,
    pub max_atoms_per_residue: usize,
    pub graph_hidden_dim: usize,
    pub graph_layers: usize,
    pub residue_dim: usize,
    pub pair_dim: usize,
    pub interaction_blocks: usize,
    pub num_attention_heads: usize,
    pub feed_forward_dim: usize,
    pub pair_feed_forward_dim: usize,
    pub dropout: f32,
    pub instrument_vocab_size: usize,
    pub instrument_dim: usize,
    pub ms2_fragment_channels: usize,
    pub contrastive_dim: usize,
}

impl Default for PeptideFoundationV0500Config {
    fn default() -> Self {
        Self {
            max_sequence_len: 64,
            max_atoms_per_residue: 24,
            graph_hidden_dim: 128,
            graph_layers: 5,
            residue_dim: 320,
            pair_dim: 128,
            interaction_blocks: 8,
            num_attention_heads: 8,
            feed_forward_dim: 1280,
            pair_feed_forward_dim: 512,
            dropout: 0.05,
            instrument_vocab_size: 16,
            instrument_dim: 32,
            ms2_fragment_channels: 8,
            contrastive_dim: 128,
        }
    }
}

impl PeptideFoundationV0500Config {
    /// Tiny CPU configuration used only for correctness tests and local smoke runs.
    pub fn local_smoke() -> Self {
        Self {
            max_sequence_len: 8,
            max_atoms_per_residue: 24,
            graph_hidden_dim: 32,
            graph_layers: 1,
            residue_dim: 64,
            pair_dim: 24,
            interaction_blocks: 1,
            num_attention_heads: 4,
            feed_forward_dim: 128,
            pair_feed_forward_dim: 64,
            dropout: 0.0,
            instrument_vocab_size: 16,
            instrument_dim: 8,
            ms2_fragment_channels: 8,
            contrastive_dim: 32,
        }
    }

    /// A100 architecture smoke: production widths/graph depth, but only two pair blocks.
    pub fn a100_smoke() -> Self {
        let mut config = Self::default();
        config.interaction_blocks = 2;
        config
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_sequence_len < 2 {
            candle_core::bail!("v0.50 max_sequence_len must be at least 2");
        }
        if self.max_atoms_per_residue < 4 {
            candle_core::bail!("v0.50 max_atoms_per_residue must be at least 4");
        }
        if self.graph_hidden_dim == 0 || self.graph_layers == 0 {
            candle_core::bail!("v0.50 graph width/layers must be non-zero");
        }
        if self.residue_dim == 0 || self.pair_dim == 0 || self.interaction_blocks == 0 {
            candle_core::bail!("v0.50 residue/pair dimensions and block count must be non-zero");
        }
        if self.num_attention_heads == 0 || self.residue_dim % self.num_attention_heads != 0 {
            candle_core::bail!("v0.50 residue_dim must be divisible by num_attention_heads");
        }
        if self.feed_forward_dim < self.residue_dim {
            candle_core::bail!("v0.50 feed_forward_dim must be at least residue_dim");
        }
        if self.pair_feed_forward_dim < self.pair_dim {
            candle_core::bail!("v0.50 pair_feed_forward_dim must be at least pair_dim");
        }
        if !(0.0..1.0).contains(&self.dropout) {
            candle_core::bail!("v0.50 dropout must be in [0, 1)");
        }
        if self.instrument_vocab_size == 0 || self.instrument_dim == 0 {
            candle_core::bail!("v0.50 instrument vocabulary/dimension must be non-zero");
        }
        if self.ms2_fragment_channels == 0 || self.contrastive_dim == 0 {
            candle_core::bail!("v0.50 output dimensions must be non-zero");
        }
        Ok(())
    }

    /// Compatibility config for the established chemistry featurizer only.
    ///
    /// The historical peptide Transformer dimensions are irrelevant here because v0.50 consumes
    /// the featurized atom/residue tensors directly and owns its own deep interaction stack.
    pub fn featurizer_config(&self) -> FoundationConfig {
        let mut config = FoundationConfig::default();
        config.max_sequence_len = self.max_sequence_len;
        config.max_atoms_per_residue = self.max_atoms_per_residue;
        config.graph_hidden_dim = self.graph_hidden_dim;
        config.graph_layers = self.graph_layers;
        config.instrument_vocab_size = self.instrument_vocab_size;
        config.ms2_fragment_channels = self.ms2_fragment_channels;
        config
    }

    pub fn total_token_count(&self) -> usize {
        FOUNDATION_V0500_TASK_COUNT + self.max_sequence_len
    }
}

/// Deep v0.50 representation exposed to property/self-supervision heads.
#[derive(Debug, Clone)]
pub struct FoundationRepresentationV0500 {
    /// Residue-only states `[batch, residues, residue_dim]`.
    pub residue_embeddings: Tensor,
    /// Explicit task+residue pair state `[batch, tokens, tokens, pair_dim]`.
    pub pair_embeddings: Tensor,
    /// Residue validity mask `[batch, residues]`.
    pub residue_mask: Tensor,
    /// Task+residue validity mask `[batch, tokens]`.
    pub token_mask: Tensor,
    /// Pair validity mask `[batch, tokens, tokens]`.
    pub pair_mask: Tensor,
    /// Context-independent reusable peptide representation `[batch, residue_dim]`.
    pub global_embedding: Tensor,
    /// RT task token `[batch, residue_dim]`.
    pub rt_embedding: Tensor,
    /// Mobility/CCS task token `[batch, residue_dim]`.
    pub mobility_embedding: Tensor,
    /// MS2 task token `[batch, residue_dim]`.
    pub ms2_embedding: Tensor,
    /// Mean raw atom descriptors per residue, preserved for chemistry reconstruction targets.
    pub chemistry_targets: Tensor,
}

#[derive(Debug, Clone)]
pub struct FoundationMultimodalForwardOutputV0500 {
    pub representation: FoundationRepresentationV0500,
    /// RT prediction `[batch, 1]`.
    pub rt: Tensor,
    /// Native mobility prediction `[batch, 1]`. Conversion/consensus policy belongs in training.
    pub mobility_native: Tensor,
    /// Fragment intensities `[batch, max_sequence_len - 1, channels]`.
    pub ms2: Tensor,
    /// Masked-residue reconstruction logits `[batch, residues, 21]`.
    pub residue_logits: Tensor,
    /// Chemistry reconstruction `[batch, residues, ATOM_FEATURE_DIM]`.
    pub chemistry_reconstruction: Tensor,
    /// Reusable projected global representation `[batch, contrastive_dim]`.
    pub contrastive_projection: Tensor,
    /// Pair-class auxiliary logits over residue-residue pairs.
    pub pair_interaction_logits: Tensor,
}

#[derive(Clone)]
struct PairConditionedSelfAttentionV0500 {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    pair_bias_heads: Vec<Linear>,
    num_heads: usize,
    head_dim: usize,
}

impl PairConditionedSelfAttentionV0500 {
    fn new(
        residue_dim: usize,
        pair_dim: usize,
        num_heads: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let head_dim = residue_dim / num_heads;
        let pair_bias_heads = (0..num_heads)
            .map(|head| nn::linear_no_bias(pair_dim, 1, vb.pp(format!("pair_bias.{head}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            query: nn::linear_no_bias(residue_dim, residue_dim, vb.pp("query"))?,
            key: nn::linear_no_bias(residue_dim, residue_dim, vb.pp("key"))?,
            value: nn::linear_no_bias(residue_dim, residue_dim, vb.pp("value"))?,
            output: nn::linear(residue_dim, residue_dim, vb.pp("output"))?,
            pair_bias_heads,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden: &Tensor, pair: &Tensor, token_mask: &Tensor) -> Result<Tensor> {
        let (batch, tokens, residue_dim) = hidden.dims3()?;
        let (pair_batch, pair_i, pair_j, pair_dim) = pair.dims4()?;
        if pair_batch != batch || pair_i != tokens || pair_j != tokens {
            candle_core::bail!("v0.50 pair-conditioned attention shape mismatch");
        }

        let q = self
            .query
            .forward(hidden)?
            .reshape((batch, tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .key
            .forward(hidden)?
            .reshape((batch, tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .value
            .forward(hidden)?
            .reshape((batch, tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;

        let pair_flat = pair
            .reshape((batch * tokens * tokens, pair_dim))?
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
        let context =
            probabilities
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, tokens, residue_dim))?;
        let output = self.output.forward(&context)?;
        let query_mask = token_mask
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, residue_dim))?;
        output.broadcast_mul(&query_mask)
    }
}

#[derive(Clone)]
struct ResiduePairInteractionBlockV0500 {
    residue_attention_norm: FoundationLayerNorm,
    pair_attention_norm: FoundationLayerNorm,
    attention: PairConditionedSelfAttentionV0500,
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
    residue_dim: usize,
    pair_dim: usize,
}

impl ResiduePairInteractionBlockV0500 {
    fn new(config: &PeptideFoundationV0500Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            residue_attention_norm: FoundationLayerNorm::new(
                config.residue_dim,
                1e-5,
                vb.pp("residue_attention_norm"),
            )?,
            pair_attention_norm: FoundationLayerNorm::new(
                config.pair_dim,
                1e-5,
                vb.pp("pair_attention_norm"),
            )?,
            attention: PairConditionedSelfAttentionV0500::new(
                config.residue_dim,
                config.pair_dim,
                config.num_attention_heads,
                vb.pp("attention"),
            )?,
            residue_to_pair_norm: FoundationLayerNorm::new(
                config.residue_dim,
                1e-5,
                vb.pp("residue_to_pair_norm"),
            )?,
            pair_update_left: nn::linear(
                config.residue_dim,
                config.pair_dim,
                vb.pp("pair_update_left"),
            )?,
            pair_update_right: nn::linear(
                config.residue_dim,
                config.pair_dim,
                vb.pp("pair_update_right"),
            )?,
            pair_update_out: nn::linear(
                config.pair_dim,
                config.pair_dim,
                vb.pp("pair_update_out"),
            )?,
            pair_transition_norm: FoundationLayerNorm::new(
                config.pair_dim,
                1e-5,
                vb.pp("pair_transition_norm"),
            )?,
            pair_transition_in: nn::linear(
                config.pair_dim,
                config.pair_feed_forward_dim,
                vb.pp("pair_transition_in"),
            )?,
            pair_transition_out: nn::linear(
                config.pair_feed_forward_dim,
                config.pair_dim,
                vb.pp("pair_transition_out"),
            )?,
            residue_transition_norm: FoundationLayerNorm::new(
                config.residue_dim,
                1e-5,
                vb.pp("residue_transition_norm"),
            )?,
            residue_transition_in: nn::linear(
                config.residue_dim,
                config.feed_forward_dim,
                vb.pp("residue_transition_in"),
            )?,
            residue_transition_out: nn::linear(
                config.feed_forward_dim,
                config.residue_dim,
                vb.pp("residue_transition_out"),
            )?,
            dropout: Dropout::new(config.dropout),
            residue_dim: config.residue_dim,
            pair_dim: config.pair_dim,
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
        let (batch, tokens, residue_dim) = hidden.dims3()?;
        if residue_dim != self.residue_dim {
            candle_core::bail!("v0.50 interaction block residue width mismatch");
        }
        let (_, pair_i, pair_j, pair_dim) = pair.dims4()?;
        if pair_i != tokens || pair_j != tokens || pair_dim != self.pair_dim {
            candle_core::bail!("v0.50 interaction block pair shape mismatch");
        }

        // 1. Pair-conditioned residue self-attention.
        let normalized_hidden = self.residue_attention_norm.forward(hidden)?;
        let normalized_pair = self.pair_attention_norm.forward(pair)?;
        let attention = self
            .attention
            .forward(&normalized_hidden, &normalized_pair, token_mask)?;
        let mut hidden = (hidden + self.dropout.forward_t(&attention, train)?)?;
        hidden = mask_token_state(&hidden, token_mask)?;

        // 2. Residue-to-pair update.  Left/right projections plus a multiplicative
        // interaction give the pair state an explicit place to encode compatibility.
        let normalized_hidden = self.residue_to_pair_norm.forward(&hidden)?;
        let left = self
            .pair_update_left
            .forward(&normalized_hidden)?
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, tokens, self.pair_dim))?;
        let right = self
            .pair_update_right
            .forward(&normalized_hidden)?
            .unsqueeze(1)?
            .broadcast_as((batch, tokens, tokens, self.pair_dim))?;
        let product = left.broadcast_mul(&right)?;
        let residue_pair = ((&left + &right)? + product)?.relu()?;
        let pair_update = self
            .pair_update_out
            .forward(
                &residue_pair
                    .reshape((batch * tokens * tokens, self.pair_dim))?
                    .contiguous()?,
            )?
            .reshape((batch, tokens, tokens, self.pair_dim))?;
        let mut pair = (pair + self.dropout.forward_t(&pair_update, train)?)?;
        pair = mask_pair_state(&pair, pair_mask)?;

        // 3. Pair transition / MLP.
        let normalized_pair = self.pair_transition_norm.forward(&pair)?;
        let flat = normalized_pair
            .reshape((batch * tokens * tokens, self.pair_dim))?
            .contiguous()?;
        let pair_transition = self.pair_transition_in.forward(&flat)?.relu()?;
        let pair_transition = self
            .pair_transition_out
            .forward(&pair_transition)?
            .reshape((batch, tokens, tokens, self.pair_dim))?;
        pair = (&pair + self.dropout.forward_t(&pair_transition, train)?)?;
        pair = mask_pair_state(&pair, pair_mask)?;

        // 4. Residue transition / MLP.
        let normalized_hidden = self.residue_transition_norm.forward(&hidden)?;
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

/// v0.50 student model.  All trainable parameters live under `student_v050.*`.
#[derive(Clone)]
pub struct PeptideFoundationV0500Model {
    config: PeptideFoundationV0500Config,
    atom_input: Linear,
    graph_layers: Vec<GraphMessageLayer>,
    graph_to_residue: Linear,
    chemistry_to_residue: Linear,
    residue_embedding: Embedding,
    position_embedding: Embedding,
    residue_input_norm: FoundationLayerNorm,
    task_embedding: Embedding,
    mobility_context_projection: Linear,
    instrument_embedding: Embedding,
    ms2_context_projection: Linear,
    pair_left: Linear,
    pair_right: Linear,
    pair_chemistry_left: Linear,
    pair_chemistry_right: Linear,
    pair_relative_projection: Linear,
    pair_input_norm: FoundationLayerNorm,
    interaction_blocks: Vec<ResiduePairInteractionBlockV0500>,
    residue_output_norm: FoundationLayerNorm,
    pair_output_norm: FoundationLayerNorm,
    rt_head_hidden: Linear,
    rt_head_output: Linear,
    mobility_head_hidden: Linear,
    mobility_head_output: Linear,
    ms2_head_hidden: Linear,
    ms2_head_output: Linear,
    residue_head: Linear,
    chemistry_head: Linear,
    contrastive_head: Linear,
    pair_interaction_head: Linear,
}

impl PeptideFoundationV0500Model {
    pub fn new(config: PeptideFoundationV0500Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let vb = vb.pp(FOUNDATION_V0500_STUDENT_NAMESPACE);
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
            config.residue_dim,
            vb.pp("chemistry.graph_to_residue"),
        )?;
        let chemistry_to_residue = nn::linear(
            ATOM_FEATURE_DIM,
            config.residue_dim,
            vb.pp("chemistry.raw_to_residue"),
        )?;
        let residue_embedding =
            nn::embedding(21, config.residue_dim, vb.pp("sequence.residue_embedding"))?;
        let position_embedding = nn::embedding(
            config.max_sequence_len,
            config.residue_dim,
            vb.pp("sequence.position_embedding"),
        )?;
        let residue_input_norm =
            FoundationLayerNorm::new(config.residue_dim, 1e-5, vb.pp("sequence.input_norm"))?;
        let task_embedding = nn::embedding(
            FOUNDATION_V0500_TASK_COUNT,
            config.residue_dim,
            vb.pp("task.embedding"),
        )?;
        let mobility_context_projection = nn::linear(
            FOUNDATION_V0500_MOBILITY_CONTEXT_DIM,
            config.residue_dim,
            vb.pp("task.mobility_context"),
        )?;
        let instrument_embedding = nn::embedding(
            config.instrument_vocab_size,
            config.instrument_dim,
            vb.pp("task.ms2_instrument_embedding"),
        )?;
        let ms2_context_projection = nn::linear(
            config.instrument_dim + FOUNDATION_V0500_MS2_CONTEXT_SCALAR_DIM,
            config.residue_dim,
            vb.pp("task.ms2_context"),
        )?;
        let pair_left = nn::linear(config.residue_dim, config.pair_dim, vb.pp("pair.init_left"))?;
        let pair_right = nn::linear(
            config.residue_dim,
            config.pair_dim,
            vb.pp("pair.init_right"),
        )?;
        let pair_chemistry_left = nn::linear(
            ATOM_FEATURE_DIM,
            config.pair_dim,
            vb.pp("pair.chemistry_left"),
        )?;
        let pair_chemistry_right = nn::linear(
            ATOM_FEATURE_DIM,
            config.pair_dim,
            vb.pp("pair.chemistry_right"),
        )?;
        let pair_relative_projection = nn::linear(
            FOUNDATION_V0500_RELATIVE_FEATURE_DIM,
            config.pair_dim,
            vb.pp("pair.relative_projection"),
        )?;
        let pair_input_norm =
            FoundationLayerNorm::new(config.pair_dim, 1e-5, vb.pp("pair.input_norm"))?;
        let interaction_blocks = (0..config.interaction_blocks)
            .map(|block| {
                ResiduePairInteractionBlockV0500::new(
                    &config,
                    vb.pp(format!("interaction.{block}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let residue_output_norm =
            FoundationLayerNorm::new(config.residue_dim, 1e-5, vb.pp("output.residue_norm"))?;
        let pair_output_norm =
            FoundationLayerNorm::new(config.pair_dim, 1e-5, vb.pp("output.pair_norm"))?;

        Ok(Self {
            rt_head_hidden: nn::linear(
                config.residue_dim,
                config.residue_dim,
                vb.pp("heads.rt.hidden"),
            )?,
            rt_head_output: nn::linear(config.residue_dim, 1, vb.pp("heads.rt.output"))?,
            mobility_head_hidden: nn::linear(
                config.residue_dim,
                config.residue_dim,
                vb.pp("heads.mobility.hidden"),
            )?,
            mobility_head_output: nn::linear(
                config.residue_dim,
                1,
                vb.pp("heads.mobility.output"),
            )?,
            ms2_head_hidden: nn::linear(
                3 * config.residue_dim,
                config.feed_forward_dim,
                vb.pp("heads.ms2.hidden"),
            )?,
            ms2_head_output: nn::linear(
                config.feed_forward_dim,
                config.ms2_fragment_channels,
                vb.pp("heads.ms2.output"),
            )?,
            residue_head: nn::linear(config.residue_dim, 21, vb.pp("heads.residue"))?,
            chemistry_head: nn::linear(
                config.residue_dim,
                ATOM_FEATURE_DIM,
                vb.pp("heads.chemistry"),
            )?,
            contrastive_head: nn::linear(
                config.residue_dim,
                config.contrastive_dim,
                vb.pp("heads.contrastive"),
            )?,
            pair_interaction_head: nn::linear(
                config.pair_dim,
                FOUNDATION_V0500_PAIR_CLASS_COUNT,
                vb.pp("heads.pair_interaction"),
            )?,
            config,
            atom_input,
            graph_layers,
            graph_to_residue,
            chemistry_to_residue,
            residue_embedding,
            position_embedding,
            residue_input_norm,
            task_embedding,
            mobility_context_projection,
            instrument_embedding,
            ms2_context_projection,
            pair_left,
            pair_right,
            pair_chemistry_left,
            pair_chemistry_right,
            pair_relative_projection,
            pair_input_norm,
            interaction_blocks,
            residue_output_norm,
            pair_output_norm,
        })
    }

    pub fn config(&self) -> &PeptideFoundationV0500Config {
        &self.config
    }

    pub fn forward_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        train: bool,
    ) -> Result<FoundationMultimodalForwardOutputV0500> {
        let (batch_size, sequence_len, atom_count, feature_dim) = batch.atom_features.dims4()?;
        if sequence_len != self.config.max_sequence_len {
            candle_core::bail!(
                "v0.50 expected sequence width {}, got {}",
                self.config.max_sequence_len,
                sequence_len
            );
        }
        if feature_dim != ATOM_FEATURE_DIM {
            candle_core::bail!(
                "v0.50 atom feature mismatch: expected {}, got {}",
                ATOM_FEATURE_DIM,
                feature_dim
            );
        }

        let chemistry_targets = mean_raw_chemistry(batch)?;
        let mut residues = self.encode_residues(batch, &chemistry_targets)?;
        residues = mask_token_state(&residues, &batch.residue_mask)?;

        let task_tokens = self.contextual_task_tokens(batch_size, context)?;
        let mut hidden = Tensor::cat(&[&task_tokens, &residues], 1)?;
        let task_mask = Tensor::ones(
            (batch_size, FOUNDATION_V0500_TASK_COUNT),
            DType::F32,
            batch.residue_mask.device(),
        )?;
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
        pair = self.pair_output_norm.forward(&pair)?;
        pair = mask_pair_state(&pair, &pair_mask)?;

        let rt_embedding = hidden.narrow(1, TASK_RT, 1)?.squeeze(1)?;
        let mobility_embedding = hidden.narrow(1, TASK_MOBILITY, 1)?.squeeze(1)?;
        let ms2_embedding = hidden.narrow(1, TASK_MS2, 1)?.squeeze(1)?;
        let global_embedding = hidden.narrow(1, TASK_GLOBAL, 1)?.squeeze(1)?;
        let residue_embeddings =
            hidden.narrow(1, FOUNDATION_V0500_TASK_COUNT, self.config.max_sequence_len)?;

        let rt = self
            .rt_head_output
            .forward(&self.rt_head_hidden.forward(&rt_embedding)?.relu()?)?;
        let mobility_native = self.mobility_head_output.forward(
            &self
                .mobility_head_hidden
                .forward(&mobility_embedding)?
                .relu()?,
        )?;
        let ms2 = self.ms2_from_representation(&residue_embeddings, &ms2_embedding, batch)?;
        let residue_logits = self.residue_head.forward(&residue_embeddings)?;
        let chemistry_reconstruction = self.chemistry_head.forward(&residue_embeddings)?;
        let contrastive_projection = self.contrastive_head.forward(&global_embedding)?;

        let residue_pair =
            pair.narrow(1, FOUNDATION_V0500_TASK_COUNT, self.config.max_sequence_len)?;
        let residue_pair =
            residue_pair.narrow(2, FOUNDATION_V0500_TASK_COUNT, self.config.max_sequence_len)?;
        let pair_flat = residue_pair
            .reshape((
                batch_size * self.config.max_sequence_len * self.config.max_sequence_len,
                self.config.pair_dim,
            ))?
            .contiguous()?;
        let pair_interaction_logits = self.pair_interaction_head.forward(&pair_flat)?.reshape((
            batch_size,
            self.config.max_sequence_len,
            self.config.max_sequence_len,
            FOUNDATION_V0500_PAIR_CLASS_COUNT,
        ))?;
        let residue_pair_mask = batch
            .residue_mask
            .unsqueeze(2)?
            .broadcast_mul(&batch.residue_mask.unsqueeze(1)?)?
            .unsqueeze(3)?
            .broadcast_as((
                batch_size,
                self.config.max_sequence_len,
                self.config.max_sequence_len,
                FOUNDATION_V0500_PAIR_CLASS_COUNT,
            ))?;
        let pair_interaction_logits = pair_interaction_logits.broadcast_mul(&residue_pair_mask)?;

        Ok(FoundationMultimodalForwardOutputV0500 {
            representation: FoundationRepresentationV0500 {
                residue_embeddings,
                pair_embeddings: pair,
                residue_mask: batch.residue_mask.clone(),
                token_mask,
                pair_mask,
                global_embedding,
                rt_embedding,
                mobility_embedding,
                ms2_embedding,
                chemistry_targets,
            },
            rt,
            mobility_native,
            ms2,
            residue_logits,
            chemistry_reconstruction,
            contrastive_projection,
            pair_interaction_logits,
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
            .reshape((batch_size, sequence_len, self.config.residue_dim))?;
        let raw_chemistry = self.chemistry_to_residue.forward(chemistry_targets)?;
        let identity = self.residue_embedding.forward(&batch.residue_ids)?;
        let positions: Vec<u32> = (0..sequence_len as u32).collect();
        let position_ids = Tensor::from_vec(positions, sequence_len, batch.residue_ids.device())?
            .to_dtype(DType::U32)?;
        let position = self
            .position_embedding
            .forward(&position_ids)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, sequence_len, self.config.residue_dim))?;
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
                TASK_MOBILITY as u32,
                TASK_MS2 as u32,
                TASK_GLOBAL as u32,
            ],
            FOUNDATION_V0500_TASK_COUNT,
            device,
        )?
        .to_dtype(DType::U32)?;
        let tasks = self
            .task_embedding
            .forward(&ids)?
            .unsqueeze(0)?
            .broadcast_as((
                batch_size,
                FOUNDATION_V0500_TASK_COUNT,
                self.config.residue_dim,
            ))?;

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
        let mobility_context = Tensor::cat(
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
        let mobility_context = self
            .mobility_context_projection
            .forward(&mobility_context)?
            .unsqueeze(1)?;

        let instrument = self.instrument_embedding.forward(&context.instrument_ids)?;
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
        let mobility = (tasks.narrow(1, TASK_MOBILITY, 1)? + mobility_context)?;
        let ms2 = (tasks.narrow(1, TASK_MS2, 1)? + ms2_context)?;
        let global = tasks.narrow(1, TASK_GLOBAL, 1)?;
        Tensor::cat(&[&rt, &mobility, &ms2, &global], 1)
    }

    fn initialize_pair_state(
        &self,
        hidden: &Tensor,
        chemistry_targets: &Tensor,
        pair_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch, tokens, _) = hidden.dims3()?;
        let sequence = self.config.max_sequence_len;
        let pair_dim = self.config.pair_dim;

        let left = self
            .pair_left
            .forward(hidden)?
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, tokens, pair_dim))?;
        let right = self
            .pair_right
            .forward(hidden)?
            .unsqueeze(1)?
            .broadcast_as((batch, tokens, tokens, pair_dim))?;

        // PTM mass/composition and terminal chemistry are already present in the raw atom
        // descriptors. Project them explicitly into pair initialization instead of relying only
        // on residue identity attention.
        let chemistry_left = self.pair_chemistry_left.forward(chemistry_targets)?;
        let chemistry_right = self.pair_chemistry_right.forward(chemistry_targets)?;
        let task_zeros = Tensor::zeros(
            (batch, FOUNDATION_V0500_TASK_COUNT, pair_dim),
            DType::F32,
            hidden.device(),
        )?;
        let chemistry_left = Tensor::cat(&[&task_zeros, &chemistry_left], 1)?
            .unsqueeze(2)?
            .broadcast_as((batch, tokens, tokens, pair_dim))?;
        let chemistry_right = Tensor::cat(&[&task_zeros, &chemistry_right], 1)?
            .unsqueeze(1)?
            .broadcast_as((batch, tokens, tokens, pair_dim))?;

        let relative = relative_pair_features_v0500(sequence, hidden.device())?;
        let relative = self
            .pair_relative_projection
            .forward(
                &relative
                    .reshape((tokens * tokens, FOUNDATION_V0500_RELATIVE_FEATURE_DIM))?
                    .contiguous()?,
            )?
            .reshape((1, tokens, tokens, pair_dim))?
            .broadcast_as((batch, tokens, tokens, pair_dim))?;

        let pair = (((left + right)? + chemistry_left)? + chemistry_right)?;
        let pair = self.pair_input_norm.forward(&(pair + relative)?)?;
        mask_pair_state(&pair, pair_mask)
    }

    fn ms2_from_representation(
        &self,
        residues: &Tensor,
        ms2_task: &Tensor,
        batch: &FoundationBatch,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, residue_dim) = residues.dims3()?;
        let cleavages = sequence_len - 1;
        let left = residues.narrow(1, 0, cleavages)?;
        let right = residues.narrow(1, 1, cleavages)?;
        let task = ms2_task
            .unsqueeze(1)?
            .broadcast_as((batch_size, cleavages, residue_dim))?;
        let features = Tensor::cat(&[&left, &right, &task], 2)?.contiguous()?;
        let hidden = self
            .ms2_head_hidden
            .forward(&features.reshape((batch_size * cleavages, 3 * residue_dim))?)?
            .relu()?;
        let prediction = self.ms2_head_output.forward(&hidden)?.relu()?.reshape((
            batch_size,
            cleavages,
            self.config.ms2_fragment_channels,
        ))?;
        let cleavage_mask = batch
            .residue_mask
            .narrow(1, 0, cleavages)?
            .broadcast_mul(&batch.residue_mask.narrow(1, 1, cleavages)?)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, cleavages, self.config.ms2_fragment_channels))?;
        prediction.broadcast_mul(&cleavage_mask)
    }
}

/// Detached outputs from the external, authoritative v0.35 teacher.
#[derive(Debug, Clone)]
pub struct FoundationTeacherAnchorOutputV0500 {
    pub rt: Tensor,
    pub ccs: Tensor,
    pub ms2: Tensor,
    pub peptide_embedding: Tensor,
}

/// Frozen-teacher adapter used by v0.50 training.
///
/// The adapter owns a separately constructed v0.35 model.  It never shares the student VarMap,
/// and every returned teacher tensor is detached.  This makes teacher checkpoint loading and
/// student checkpoint serialization independent and prevents accidental optimization of v0.35.
#[derive(Clone)]
pub struct FoundationV0350TeacherV0500 {
    model: PeptideFoundationMultimodalV0350Model,
}

impl FoundationV0350TeacherV0500 {
    pub fn new(model: PeptideFoundationMultimodalV0350Model) -> Self {
        Self { model }
    }

    pub fn forward_detached_t(
        &self,
        batch: &FoundationBatch,
        context: &PrecursorContextBatch,
        fragment: &FoundationFragmentContextBatchV0350,
    ) -> Result<FoundationTeacherAnchorOutputV0500> {
        let output = self
            .model
            .forward_v0350_t(batch, context, fragment, false)?;
        Ok(FoundationTeacherAnchorOutputV0500 {
            rt: output.base.rt.detach(),
            ccs: output.base.ccs.detach(),
            ms2: output.base.ms2.detach(),
            peptide_embedding: output.base.foundation.peptide_embedding.detach(),
        })
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

/// Fixed relative descriptors for task+residue pair initialization.
///
/// Residue-residue entries encode signed and absolute sequence separation explicitly.  Task
/// entries receive the same normalized coordinate system plus task/residue indicator features;
/// the learned task embeddings distinguish RT, mobility, MS2, and global roles.
fn relative_pair_features_v0500(
    sequence_len: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let tokens = FOUNDATION_V0500_TASK_COUNT + sequence_len;
    let denom = sequence_len.max(1) as f32;
    let mut values = Vec::with_capacity(tokens * tokens * FOUNDATION_V0500_RELATIVE_FEATURE_DIM);
    for i in 0..tokens {
        for j in 0..tokens {
            let i_residue = i >= FOUNDATION_V0500_TASK_COUNT;
            let j_residue = j >= FOUNDATION_V0500_TASK_COUNT;
            let i_pos = if i_residue {
                (i - FOUNDATION_V0500_TASK_COUNT) as f32
            } else {
                0.0
            };
            let j_pos = if j_residue {
                (j - FOUNDATION_V0500_TASK_COUNT) as f32
            } else {
                0.0
            };
            let signed = if i_residue && j_residue {
                (j_pos - i_pos) / denom
            } else {
                0.0
            };
            let absolute = signed.abs();
            values.extend_from_slice(&[
                signed,
                absolute,
                i_pos / denom,
                j_pos / denom,
                if i == j { 1.0 } else { 0.0 },
                if i_residue && j_residue { 1.0 } else { 0.0 },
            ]);
        }
    }
    Tensor::from_vec(
        values,
        (1, tokens, tokens, FOUNDATION_V0500_RELATIVE_FEATURE_DIM),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::super::featurize::{
        FoundationModification, PeptideGraphFeaturizer, PeptidoformInput,
    };
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    fn smoke_batch(
        config: &PeptideFoundationV0500Config,
        device: &Device,
    ) -> Result<FoundationBatch> {
        let featurizer = PeptideGraphFeaturizer::new(config.featurizer_config())?;
        let mut modified = PeptidoformInput::unmodified("PEPTIDE");
        modified
            .modifications
            .push(FoundationModification::mass_delta(2, 15.9949));
        featurizer.featurize(&[modified, PeptidoformInput::unmodified("ACDK")], device)
    }

    #[test]
    fn v0500_default_matches_intended_first_full_architecture() {
        let config = PeptideFoundationV0500Config::default();
        assert_eq!(config.graph_hidden_dim, 128);
        assert_eq!(config.graph_layers, 5);
        assert_eq!(config.residue_dim, 320);
        assert_eq!(config.pair_dim, 128);
        assert_eq!(config.interaction_blocks, 8);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.feed_forward_dim, 1280);
        assert!((config.dropout - 0.05).abs() < f32::EPSILON);
        config.validate().unwrap();
    }

    #[test]
    fn v0500_student_parameters_have_an_isolated_checkpoint_namespace() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _model =
            PeptideFoundationV0500Model::new(PeptideFoundationV0500Config::local_smoke(), vb)?;
        let data = varmap.data().lock().unwrap();
        assert!(!data.is_empty());
        assert!(data.keys().all(|name| name.starts_with("student_v050.")));
        assert!(data.keys().all(|name| !name.contains("v0350")));
        Ok(())
    }

    #[test]
    fn v0500_forward_shapes_and_masks_are_consistent() -> Result<()> {
        let device = Device::Cpu;
        let config = PeptideFoundationV0500Config::local_smoke();
        let batch = smoke_batch(&config, &device)?;
        let context = PrecursorContextBatch::unknown(2, &device)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideFoundationV0500Model::new(config.clone(), vb)?;
        let output = model.forward_t(&batch, &context, false)?;
        let tokens = config.total_token_count();
        assert_eq!(output.rt.dims2()?, (2, 1));
        assert_eq!(output.mobility_native.dims2()?, (2, 1));
        assert_eq!(
            output.ms2.dims3()?,
            (2, config.max_sequence_len - 1, config.ms2_fragment_channels)
        );
        assert_eq!(
            output.representation.residue_embeddings.dims3()?,
            (2, config.max_sequence_len, config.residue_dim)
        );
        assert_eq!(
            output.representation.pair_embeddings.dims4()?,
            (2, tokens, tokens, config.pair_dim)
        );
        assert_eq!(
            output.representation.pair_mask.dims3()?,
            (2, tokens, tokens)
        );
        assert_eq!(
            output.pair_interaction_logits.dims4()?,
            (
                2,
                config.max_sequence_len,
                config.max_sequence_len,
                FOUNDATION_V0500_PAIR_CLASS_COUNT,
            )
        );

        // Peptide 2 has length four: its first padded residue is index 4, therefore token 8.
        let padded_token = FOUNDATION_V0500_TASK_COUNT + 4;
        let padded_row = output
            .representation
            .pair_embeddings
            .narrow(0, 1, 1)?
            .narrow(1, padded_token, 1)?;
        let padded_column = output
            .representation
            .pair_embeddings
            .narrow(0, 1, 1)?
            .narrow(2, padded_token, 1)?;
        assert_eq!(padded_row.abs()?.sum_all()?.to_scalar::<f32>()?, 0.0);
        assert_eq!(padded_column.abs()?.sum_all()?.to_scalar::<f32>()?, 0.0);
        Ok(())
    }

    #[test]
    fn v0500_joint_loss_reaches_graph_pair_task_and_property_parameters() -> Result<()> {
        let device = Device::Cpu;
        let config = PeptideFoundationV0500Config::local_smoke();
        let batch = smoke_batch(&config, &device)?;
        let context = PrecursorContextBatch::unknown(2, &device)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideFoundationV0500Model::new(config, vb)?;
        let output = model.forward_t(&batch, &context, true)?;
        let loss = (((output.rt.sum_all()? + output.mobility_native.sum_all()?)?
            + output.ms2.sum_all()?)?
            + output.pair_interaction_logits.sum_all()?)?;
        let gradients = loss.backward()?;
        let data = varmap.data().lock().unwrap();
        for name in [
            "student_v050.chemistry.atom_input.weight",
            "student_v050.interaction.0.attention.query.weight",
            "student_v050.interaction.0.pair_update_left.weight",
            "student_v050.task.embedding.weight",
            "student_v050.heads.rt.output.weight",
            "student_v050.heads.mobility.output.weight",
            "student_v050.heads.ms2.output.weight",
        ] {
            let variable = data.get(name).unwrap_or_else(|| panic!("missing {name}"));
            let gradient = gradients
                .get(variable)
                .unwrap_or_else(|| panic!("missing gradient for {name}"));
            assert!(gradient.sqr()?.sum_all()?.to_scalar::<f32>()?.is_finite());
        }
        Ok(())
    }
}
