//! Conversion of peptide sequences into batched hierarchical molecular graphs.

use super::chemistry::{
    exact_graph_modification, residue_graph, ExactGraphModification, ModificationAttachmentSite,
    ATOM_FEATURE_DIM,
};
use super::config::FoundationConfig;
use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

/// Chemical scope of a peptide modification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FoundationModificationSite {
    /// Modification is attached to a specific zero-based residue index.
    Residue(usize),
    /// Modification is attached to the peptide N terminus.
    NTerm,
    /// Modification is attached to the peptide C terminus.
    CTerm,
}

/// A site-specific modification supplied to the chemistry featurizer.
///
/// `unimod_id` preserves canonical modification identity when the source table
/// provides it.  `mass_delta` is retained for numerical compatibility and as a
/// fallback for open/unknown mass-shift annotations.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationModification {
    /// Zero-based residue index used by the current residue-local graph.
    ///
    /// Terminal modifications are provisionally anchored to residue 0 or the
    /// final residue respectively until explicit terminal graph nodes are added.
    pub residue_index: usize,
    /// Monoisotopic mass delta.
    pub mass_delta: f32,
    /// Canonical UniMod accession when known.
    pub unimod_id: Option<u32>,
    /// Chemical site/scope of the modification.
    pub site: FoundationModificationSite,
}

impl FoundationModification {
    /// Construct an unresolved/open mass-shift modification on a residue.
    pub fn mass_delta(residue_index: usize, mass_delta: f32) -> Self {
        Self::mass_delta_at_site(
            FoundationModificationSite::Residue(residue_index),
            residue_index,
            mass_delta,
        )
    }

    /// Construct an unresolved/open mass-shift modification at an explicit
    /// residue or terminal site.
    pub fn mass_delta_at_site(
        site: FoundationModificationSite,
        residue_index: usize,
        mass_delta: f32,
    ) -> Self {
        Self {
            residue_index,
            mass_delta,
            unimod_id: None,
            site,
        }
    }

    /// Construct a canonical UniMod modification.
    pub fn unimod(
        site: FoundationModificationSite,
        residue_index: usize,
        unimod_id: u32,
        mass_delta: f32,
    ) -> Self {
        Self {
            residue_index,
            mass_delta,
            unimod_id: Some(unimod_id),
            site,
        }
    }

    /// Stable human-readable identity used by reports and split keys.
    pub fn identity_label(&self) -> String {
        match self.unimod_id {
            Some(id) => format!("UniMod:{id}"),
            None => format!("Mass:{:+.4}", self.mass_delta),
        }
    }
}

/// Resolve the exact local graph transformation for one canonical modification.
///
/// Returning `None` is intentional: unsupported UniMod/site combinations and
/// open mass shifts retain the generic pseudo-mass representation.
pub fn exact_graph_modification_for(
    residue: char,
    modification: &FoundationModification,
) -> Option<ExactGraphModification> {
    let unimod_id = modification.unimod_id?;
    let site = match modification.site {
        FoundationModificationSite::Residue(_) => ModificationAttachmentSite::Residue,
        FoundationModificationSite::NTerm => ModificationAttachmentSite::NTerm,
        FoundationModificationSite::CTerm => ModificationAttachmentSite::CTerm,
    };
    exact_graph_modification(unimod_id, residue, site)
}

/// Input peptidoform consumed by the foundation featurizer.
#[derive(Debug, Clone, PartialEq)]
pub struct PeptidoformInput {
    /// Unmodified amino-acid sequence.
    pub sequence: String,
    /// Site-specific modifications.
    pub modifications: Vec<FoundationModification>,
}

impl PeptidoformInput {
    /// Construct an unmodified peptidoform.
    pub fn unmodified(sequence: impl Into<String>) -> Self {
        Self {
            sequence: sequence.into(),
            modifications: Vec::new(),
        }
    }
}

/// Tensorized peptide batch used by [`crate::foundation::PeptideFoundationEncoder`].
#[derive(Debug, Clone)]
pub struct FoundationBatch {
    /// Atom descriptors with shape `[batch, residues, atoms, atom_features]`.
    pub atom_features: Tensor,
    /// Bond-order adjacency with shape `[batch, residues, atoms, atoms]`.
    pub adjacency: Tensor,
    /// Atom mask with shape `[batch, residues, atoms]`.
    pub atom_mask: Tensor,
    /// Residue token ids with shape `[batch, residues]`.
    pub residue_ids: Tensor,
    /// Residue mask with shape `[batch, residues]`.
    pub residue_mask: Tensor,
}

/// CPU-side featurizer for canonical residue chemistry.
#[derive(Debug, Clone)]
pub struct PeptideGraphFeaturizer {
    config: FoundationConfig,
}

impl PeptideGraphFeaturizer {
    /// Create a featurizer and validate the model configuration.
    pub fn new(config: FoundationConfig) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        Ok(Self { config })
    }

    /// Convert a peptide batch into dense graph tensors on `device`.
    pub fn featurize(
        &self,
        peptides: &[PeptidoformInput],
        device: &Device,
    ) -> Result<FoundationBatch> {
        let b = peptides.len();
        let l = self.config.max_sequence_len;
        let a = self.config.max_atoms_per_residue;
        let f = self.config.atom_feature_dim;

        let mut atom_features = vec![0.0f32; b * l * a * f];
        let mut adjacency = vec![0.0f32; b * l * a * a];
        let mut atom_mask = vec![0.0f32; b * l * a];
        let mut residue_ids = vec![0u32; b * l];
        let mut residue_mask = vec![0.0f32; b * l];

        for (batch_idx, peptide) in peptides.iter().enumerate() {
            let residues: Vec<char> = peptide.sequence.chars().collect();
            if residues.len() > l {
                candle_core::bail!(
                    "peptide length {} exceeds configured maximum {} for {}",
                    residues.len(),
                    l,
                    peptide.sequence
                );
            }

            for (residue_idx, residue) in residues.iter().copied().enumerate() {
                let mut graph = residue_graph(residue).ok_or_else(|| {
                    candle_core::Error::Msg(format!("unsupported amino acid '{residue}'"))
                })?;
                for modification in peptide
                    .modifications
                    .iter()
                    .filter(|m| m.residue_index == residue_idx)
                {
                    let applied_exactly = exact_graph_modification_for(residue, modification)
                        .is_some_and(|kind| graph.apply_exact_modification(kind));
                    if !applied_exactly {
                        graph.add_mass_delta_modification(modification.mass_delta);
                    }
                }
                if graph.atoms.len() > a {
                    candle_core::bail!(
                        "residue {} at index {} needs {} atom slots, configured maximum is {}",
                        residue,
                        residue_idx,
                        graph.atoms.len(),
                        a
                    );
                }

                residue_ids[batch_idx * l + residue_idx] = residue_token_id(residue) as u32;
                residue_mask[batch_idx * l + residue_idx] = 1.0;
                let is_n_terminal = residue_idx == 0;
                let is_c_terminal = residue_idx + 1 == residues.len();

                for (atom_idx, atom) in graph.atoms.iter().enumerate() {
                    atom_mask[(batch_idx * l + residue_idx) * a + atom_idx] = 1.0;
                    let features = atom.features(is_n_terminal, is_c_terminal);
                    let base = ((batch_idx * l + residue_idx) * a + atom_idx) * f;
                    atom_features[base..base + ATOM_FEATURE_DIM].copy_from_slice(&features);
                }

                // Add self-loops plus symmetric bond-order weights.  The graph
                // encoder normalizes rows before message passing.
                for atom_idx in 0..graph.atoms.len() {
                    let index = ((batch_idx * l + residue_idx) * a + atom_idx) * a + atom_idx;
                    adjacency[index] = 1.0;
                }
                for edge in &graph.bonds {
                    let base = (batch_idx * l + residue_idx) * a * a;
                    adjacency[base + edge.source * a + edge.target] = edge.order;
                    adjacency[base + edge.target * a + edge.source] = edge.order;
                }
            }
        }

        Ok(FoundationBatch {
            atom_features: Tensor::from_vec(atom_features, (b, l, a, f), device)?,
            adjacency: Tensor::from_vec(adjacency, (b, l, a, a), device)?,
            atom_mask: Tensor::from_vec(atom_mask, (b, l, a), device)?,
            residue_ids: Tensor::from_vec(residue_ids, (b, l), device)?.to_dtype(DType::U32)?,
            residue_mask: Tensor::from_vec(residue_mask, (b, l), device)?,
        })
    }
}

/// Stable residue vocabulary used by the learned sequence embedding.
pub fn residue_token_id(residue: char) -> usize {
    match residue {
        'A' => 1,
        'C' => 2,
        'D' => 3,
        'E' => 4,
        'F' => 5,
        'G' => 6,
        'H' => 7,
        'I' => 8,
        'K' => 9,
        'L' => 10,
        'M' => 11,
        'N' => 12,
        'P' => 13,
        'Q' => 14,
        'R' => 15,
        'S' => 16,
        'T' => 17,
        'V' => 18,
        'W' => 19,
        'Y' => 20,
        _ => 0,
    }
}
