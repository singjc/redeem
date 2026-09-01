//! Configuration for the chemistry-aware peptide foundation encoder.

use serde::{Deserialize, Serialize};

/// Scalar context supplied to the CCS prediction head.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FoundationCcsContextMode {
    /// Historical CCS context: scaled charge plus an explicit charge-presence mask.
    #[default]
    ChargePresence,
    /// Physics-conditioned CCS context: scaled neutral-mass proxy (`m/z * charge`) plus charge.
    ///
    /// This keeps the intrinsic peptide embedding acquisition-independent while giving the
    /// CCS head direct access to precursor size and charge without changing parameter shapes.
    NeutralMassCharge,
}

/// Hyperparameters for [`crate::foundation::PeptideFoundationEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FoundationConfig {
    /// Maximum peptide length accepted by the encoder.
    pub max_sequence_len: usize,
    /// Maximum number of heavy/pseudo atoms retained per residue graph.
    pub max_atoms_per_residue: usize,
    /// Width of the raw atom feature vector.
    pub atom_feature_dim: usize,
    /// Hidden width used by atom-level graph message passing.
    pub graph_hidden_dim: usize,
    /// Number of atom-level message-passing layers.
    pub graph_layers: usize,
    /// Shared residue/peptide embedding width.
    pub model_dim: usize,
    /// Number of self-attention heads in each peptide Transformer block.
    pub num_attention_heads: usize,
    /// Feed-forward width in each Transformer block.
    pub transformer_ff_dim: usize,
    /// Number of peptide Transformer blocks.
    pub transformer_layers: usize,
    /// Dropout probability used in Transformer blocks.
    pub dropout: f32,
    /// Dimension of the contrastive projection used for self-supervision.
    pub contrastive_dim: usize,
    /// Number of supported instrument-condition categories.
    pub instrument_vocab_size: usize,
    /// Number of MS2 fragment-intensity channels emitted per cleavage.
    pub ms2_fragment_channels: usize,
    /// Scalar precursor context supplied only to the CCS head.
    pub ccs_context_mode: FoundationCcsContextMode,
}

impl Default for FoundationConfig {
    fn default() -> Self {
        Self {
            max_sequence_len: 64,
            max_atoms_per_residue: 20,
            atom_feature_dim: 12,
            graph_hidden_dim: 64,
            graph_layers: 3,
            model_dim: 192,
            num_attention_heads: 4,
            transformer_ff_dim: 768,
            transformer_layers: 4,
            dropout: 0.05,
            contrastive_dim: 128,
            instrument_vocab_size: 16,
            ms2_fragment_channels: 8,
            ccs_context_mode: FoundationCcsContextMode::ChargePresence,
        }
    }
}

impl FoundationConfig {
    /// Validate shape relationships that are required by the encoder.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_sequence_len < 2 {
            return Err("max_sequence_len must be at least 2".into());
        }
        if self.max_atoms_per_residue < 4 {
            return Err("max_atoms_per_residue must be at least 4".into());
        }
        if self.atom_feature_dim != 12 {
            return Err(
                "the initial chemistry featurizer currently emits exactly 12 atom features".into(),
            );
        }
        if self.graph_layers == 0 || self.transformer_layers == 0 {
            return Err("graph_layers and transformer_layers must be non-zero".into());
        }
        if self.num_attention_heads == 0 || self.model_dim % self.num_attention_heads != 0 {
            return Err("model_dim must be divisible by num_attention_heads".into());
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err("dropout must be in [0, 1)".into());
        }
        Ok(())
    }
}
