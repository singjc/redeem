//! Chemistry-aware peptide foundation model.
//!
//! This module implements a hierarchical representation in three stages:
//! atom-level residue graphs, residue-level fusion, and peptide-level
//! Transformer attention.  The intrinsic peptidoform representation is kept
//! separate from experimental precursor context so the learned embedding can
//! be reused for RT, CCS, MS2, rescoring, detectability, or future adapters.
//!
//! The first implementation uses canonical heavy-atom amino-acid graphs and
//! can represent unresolved PTMs as mass-delta pseudo-atoms.  The public data
//! structures are intentionally designed so exact modification chemistry can
//! replace those pseudo-atoms later without changing the neural interface.

pub mod chemistry;
pub mod config;
pub mod data;
pub mod featurize;
pub mod layers;
pub mod loss;
pub mod model;

pub use config::FoundationConfig;
pub use data::{
    FoundationTrainingRecord, FragmentTarget, RetentionTimeLabels, RetentionTimeObjective,
    TrainingContext,
};
pub use featurize::{
    FoundationBatch, FoundationModification, PeptideGraphFeaturizer, PeptidoformInput,
};
pub use loss::{multi_task_loss, FoundationLossWeights, FoundationLosses, FoundationTargets};
pub use model::{
    FoundationMultiTaskOutput, FoundationOutput, PeptideFoundationEncoder,
    PeptideFoundationMultiTaskModel, PrecursorContextBatch,
};
