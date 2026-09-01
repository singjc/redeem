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

pub mod checkpoint;
pub mod chemistry;
pub mod collate;
pub mod config;
pub mod control;
pub mod corpus;
pub mod data;
pub mod dataset;
pub mod experiment;
pub mod featurize;
pub mod layers;
pub mod loss;
pub mod metadata;
pub mod model;
pub mod normalization;
pub mod optimizer;
pub mod run;
pub mod sampling;
pub mod split;
pub mod trainer;
pub mod wrapper;

pub use checkpoint::{
    foundation_checkpoint_paths, FoundationCheckpointMetadata, FoundationCheckpointProvenance,
    FoundationTrainingProgress, FOUNDATION_CHECKPOINT_VERSION, FOUNDATION_MODEL_FILE,
    FOUNDATION_OPTIMIZER_FILE, FOUNDATION_STATE_FILE,
};
pub use chemistry::{
    common_unimod_definition, exact_graph_modification, ElementalComposition,
    ExactGraphModification, FoundationModificationDefinition, ModificationAttachmentSite,
};
pub use collate::{
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationTrainingBatch, FoundationTrainingViews,
};
pub use config::{FoundationCcsContextMode, FoundationCcsPhysicsBaselineConfig, FoundationConfig};
pub use control::{FoundationFitConfig, FoundationLearningRateSchedule};
pub use corpus::{
    load_foundation_corpus, FoundationCorpus, FoundationCorpusConfig, FoundationCorpusDelimiter,
    FoundationCorpusSourceSpec, FoundationCorpusSourceSummary, FoundationRecordProvenance,
};
pub use data::{
    FoundationTrainingRecord, FragmentTarget, RetentionTimeLabels, RetentionTimeObjective,
    TrainingContext,
};
pub use dataset::{
    parse_modified_peptide, FoundationCcsDerivationMode, FoundationDataset,
    FoundationDatasetLoader, FoundationSchemaCollision, FoundationSchemaField,
    FoundationTableLoadReport, FoundationTableLoadStats, FoundationTableLoaderConfig,
    FoundationTableSchemaReport, FragmentIntensityNormalization, InstrumentVocabulary,
};
pub use experiment::{
    build_foundation_benchmark_manifest, foundation_dataset_fingerprint,
    foundation_record_fingerprint, FoundationBenchmarkEntry, FoundationBenchmarkManifest,
    FoundationPartition, FOUNDATION_BENCHMARK_MANIFEST_VERSION,
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
pub use normalization::{
    FoundationRegressionNormalization, FoundationRegressionNormalizationStrategy,
    FoundationTargetNormalizationConfig,
};
pub use optimizer::{FoundationAdamW, FoundationAdamWConfig, FoundationOptimizerStep};
pub use run::{
    evaluate_foundation_checkpoint, read_foundation_training_run_config,
    run_foundation_pretraining, FoundationCheckpointEvaluationSummary, FoundationTrainingRunConfig,
    FoundationTrainingRunSummary,
};
pub use sampling::{
    sample_foundation_training_indices, sample_foundation_validation_indices,
    FoundationSampleCoverage, FoundationSamplePlan, FoundationSamplingConfig,
    FoundationSamplingStrategy,
};
pub use split::{
    foundation_split_group_key, split_foundation_record_indices, split_foundation_records,
    FoundationSplitConfig, FoundationSplitIndices, FoundationSplitMode, FoundationSplitSummary,
};
pub use trainer::{
    FoundationEpochMetrics, FoundationEvaluationConfig, FoundationFitEpochMetrics,
    FoundationFitSummary, FoundationGradientDiagnosticsConfig, FoundationPropertyEvaluationMetrics,
    FoundationRegressionEvaluationMetrics, FoundationSharedGradientScalesConfig,
    FoundationStepMetrics, FoundationTaskGradientNorms, FoundationTrainer, FoundationTrainerConfig,
};
pub use wrapper::FoundationModelWrapper;
