//! Chemistry-aware peptide foundation model.
//!
//! This module implements a hierarchical representation in three stages:
//! atom-level residue graphs, residue-level fusion, and peptide-level
//! Transformer attention. The intrinsic peptidoform representation is kept
//! separate from experimental precursor context so the learned embedding can
//! be reused for RT, CCS, MS2, rescoring, detectability, or future adapters.
//!
//! The training stack supports heterogeneous public proteomics corpora: long
//! transition tables are grouped into precursor records, missing property labels
//! are represented with masks, and two independently corrupted graph/sequence
//! views provide masked reconstruction plus contrastive self-supervision.

pub mod chemistry;
pub mod collate;
pub mod config;
pub mod data;
pub mod dataset;
pub mod featurize;
pub mod layers;
pub mod loss;
pub mod metadata;
pub mod model;
pub mod split;
pub mod trainer;
pub mod wrapper;

pub use chemistry::{
    common_unimod_definition, exact_graph_modification, ElementalComposition,
    ExactGraphModification, FoundationModificationDefinition, ModificationAttachmentSite,
};
pub use collate::{
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationTrainingBatch, FoundationTrainingViews,
};
pub use config::FoundationConfig;
pub use data::{
    FoundationTrainingRecord, FragmentTarget, RetentionTimeLabels, RetentionTimeObjective,
    TrainingContext,
};
pub use dataset::{
    parse_modified_peptide, FoundationDataset, FoundationDatasetLoader, FoundationSchemaCollision,
    FoundationSchemaField, FoundationTableLoadReport, FoundationTableLoadStats,
    FoundationTableLoaderConfig, FoundationTableSchemaReport, FragmentIntensityNormalization,
    InstrumentVocabulary,
};
pub use featurize::{
    exact_graph_modification_for, FoundationBatch, FoundationModification,
    FoundationModificationSite, PeptideGraphFeaturizer, PeptidoformInput,
};
pub use loss::{
    contrastive_info_nce_loss, multi_task_loss, FoundationLossWeights, FoundationLosses,
    FoundationTargets,
};
pub use metadata::{
    apply_source_metadata, FoundationMetadataApplicationStats, FoundationMetadataMergePolicy,
    FoundationSourceMetadata,
};
pub use model::{
    FoundationMultiTaskOutput, FoundationOutput, PeptideFoundationEncoder,
    PeptideFoundationMultiTaskModel, PrecursorContextBatch,
};
pub use split::{
    split_foundation_records, FoundationSplitConfig, FoundationSplitIndices, FoundationSplitMode,
    FoundationSplitSummary,
};
pub use trainer::{
    FoundationEpochMetrics, FoundationStepMetrics, FoundationTrainer, FoundationTrainerConfig,
};
pub use wrapper::FoundationModelWrapper;
