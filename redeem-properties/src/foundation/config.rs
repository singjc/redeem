//! Configuration for the chemistry-aware peptide foundation encoder.

use serde::{Deserialize, Serialize};

/// Non-negative output activation used by the forward MS2 intensity head.
///
/// Historical checkpoints/configs default to [`Self::Relu`]. The v0.13.8
/// candidate uses a fixed high-beta Softplus so negative pre-activations retain
/// a learning signal without introducing a new hyperparameter sweep.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FoundationMs2OutputActivation {
    /// Historical hard rectifier. Negative pre-activations have zero gradient.
    #[default]
    Relu,
    /// v0.13.8 smooth positive rescue: Softplus with fixed beta=5.
    SoftplusV0138,
}

impl FoundationMs2OutputActivation {
    /// Stable CLI/config spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::SoftplusV0138 => "softplus-v0138",
        }
    }
}

impl std::str::FromStr for FoundationMs2OutputActivation {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "relu" => Ok(Self::Relu),
            "softplus-v0138" | "softplus" => Ok(Self::SoftplusV0138),
            other => Err(format!(
                "unsupported foundation MS2 output activation {other:?}; expected relu or softplus-v0138"
            )),
        }
    }
}

/// Fixed Softplus beta used by the v0.13.8 controlled activation rescue.
pub const FOUNDATION_MS2_SOFTPLUS_BETA_V0138: f64 = 5.0;

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

/// Frozen train-derived physical CCS prior used by the residual CCS head.
///
/// Coefficients operate on native CCS units using the feature order:
///
/// 0. intercept,
/// 1. charge / 4,
/// 2. charge^2 / 16,
/// 3. precursor m/z / 1000,
/// 4. (precursor m/z * charge) / 3000,
/// 5. peptide sequence length / 30,
/// 6. charge-present mask,
/// 7. precursor-m/z-present mask.
///
/// The native baseline is converted into the standardized CCS space used by
/// training before the learned CCS residual is added. Fitting these values on
/// a train-only partition keeps the shared peptide representation independent
/// of dataset-specific calibration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FoundationCcsPhysicsBaselineConfig {
    /// Native-unit linear coefficients in the feature order documented above.
    pub coefficients_native: [f64; 8],
    /// Train-partition native CCS mean used by target standardization.
    pub target_mean_native: f64,
    /// Train-partition native CCS standard deviation used by target standardization.
    pub target_std_native: f64,
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
    /// Non-negative activation applied to raw MS2 head logits.
    pub ms2_output_activation: FoundationMs2OutputActivation,
    /// Scalar precursor context supplied only to the CCS head.
    pub ccs_context_mode: FoundationCcsContextMode,
    /// Optional frozen train-derived physical CCS prior. When present, the
    /// trainable CCS head learns an additive residual in standardized target
    /// space rather than the entire CCS value from scratch.
    pub ccs_physics_baseline: Option<FoundationCcsPhysicsBaselineConfig>,
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
            ms2_output_activation: FoundationMs2OutputActivation::Relu,
            ccs_context_mode: FoundationCcsContextMode::ChargePresence,
            ccs_physics_baseline: None,
        }
    }
}

impl FoundationConfig {
    /// Whether two configs instantiate the same trainable parameter shapes/names.
    ///
    /// MS2 output activation changes forward/autograd semantics but does not add, remove,
    /// or resize parameters, so a historical ReLU checkpoint can be loaded into the
    /// v0.13.8 Softplus candidate for a controlled zero-step comparison.
    pub fn parameter_compatible_with(&self, other: &Self) -> bool {
        let mut left = self.clone();
        let mut right = other.clone();
        left.ms2_output_activation = FoundationMs2OutputActivation::Relu;
        right.ms2_output_activation = FoundationMs2OutputActivation::Relu;
        left == right
    }

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
        if let Some(baseline) = &self.ccs_physics_baseline {
            if !baseline.target_mean_native.is_finite() {
                return Err("ccs_physics_baseline target_mean_native must be finite".into());
            }
            if !baseline.target_std_native.is_finite() || baseline.target_std_native <= 0.0 {
                return Err(
                    "ccs_physics_baseline target_std_native must be finite and positive".into(),
                );
            }
            if baseline
                .coefficients_native
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err("ccs_physics_baseline coefficients_native must all be finite".into());
            }
        }
        Ok(())
    }
}
