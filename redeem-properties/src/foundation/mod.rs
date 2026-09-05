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

pub mod causal;
pub mod ccs_physics;
pub mod checkpoint;
pub mod chemistry;
pub mod cleavage_graph;
pub mod collate;
pub mod config;
pub mod control;
pub mod corpus;
pub mod data;
pub mod dataset;
pub mod diffusion;
pub mod experiment;
pub mod featurize;
pub mod interaction;
pub mod iterative_refinement;
pub mod layers;
pub mod loss;
pub mod metadata;
pub mod model;
pub mod msp;
pub mod normalization;
pub mod optimizer;
pub mod reverse_causal;
pub mod rt_harmonization;
pub mod run;
pub mod sampling;
pub mod spectrum;
pub mod split;
pub mod trainer;
pub mod unified;
pub mod wrapper;

pub use causal::{
    foundation_causal_conditioning_margin_loss, foundation_causal_next_token_loss,
    foundation_fragment_causal_rerank_score, load_causal_from_diffusion_checkpoint,
    FoundationCausalBatch, FoundationCausalCollator, FoundationCausalContext,
    FoundationCausalInputBatch, FoundationCausalOutput, FoundationCausalWarmStartReport,
    PeptideSpectrumCausalModel, FOUNDATION_CAUSAL_RERANK_POLICY_V0123,
    FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123,
};
pub use ccs_physics::{
    evaluate_foundation_ccs_physics_baseline, fit_foundation_ccs_physics_baseline,
    fit_foundation_ccs_physics_baseline_source_weighted, foundation_ccs_physics_features,
    foundation_ccs_physics_features_from_values, predict_foundation_ccs_physics_native,
    FoundationCcsPhysicsFeatureSummary, FoundationCcsPhysicsFitConfig,
    FoundationCcsPhysicsFitResult, FoundationCcsPhysicsMetrics,
    FoundationCcsPhysicsSourceWeightSummary, FOUNDATION_CCS_PHYSICS_FEATURE_COUNT,
    FOUNDATION_CCS_PHYSICS_FEATURE_NAMES,
};
pub use checkpoint::{
    foundation_checkpoint_paths, FoundationCheckpointMetadata, FoundationCheckpointProvenance,
    FoundationTrainingProgress, FOUNDATION_CHECKPOINT_VERSION, FOUNDATION_MODEL_FILE,
    FOUNDATION_OPTIMIZER_FILE, FOUNDATION_STATE_FILE,
};
pub use chemistry::{
    common_unimod_definition, exact_graph_modification, ElementalComposition,
    ExactGraphModification, FoundationModificationDefinition, ModificationAttachmentSite,
};
pub use cleavage_graph::{
    foundation_build_cleavage_graph, foundation_cleavage_graph_edge_features,
    foundation_cleavage_graph_k_best_candidates, foundation_cleavage_graph_k_best_from_edge_scores,
    foundation_cleavage_graph_outgoing_edge_loss, foundation_cleavage_graph_structured_decode,
    foundation_cleavage_graph_structured_edge_features,
    foundation_cleavage_graph_structured_k_best_candidates,
    foundation_cleavage_graph_structured_k_best_from_edge_scores,
    foundation_cleavage_graph_structured_loss, foundation_cleavage_graph_training_batch,
    foundation_cleavage_graph_true_path_audit, validate_cleavage_graph_namespace,
    validate_cleavage_graph_structured_namespace, FoundationCleavageGraph,
    FoundationCleavageGraphBatch, FoundationCleavageGraphCandidate, FoundationCleavageGraphEdge,
    FoundationCleavageGraphNode, FoundationCleavageGraphStructuredDecode,
    FoundationCleavageGraphStructuredLossStats, FoundationCleavageGraphTrainingGroup,
    FoundationCleavageGraphTruePathAudit, FoundationCleavageGraphUnit,
    PeptideSpectrumCleavageGraphScorer, PeptideSpectrumCleavageGraphStructuredScorer,
    FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316, FOUNDATION_CLEAVAGE_GRAPH_HIDDEN_DIM_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_ANCHOR_BRIDGE_EDGES_V01317,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316, FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01317,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_NAMESPACE_V01318,
    FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_OBJECTIVE_V01318,
};
pub use collate::{
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationTrainingBatch, FoundationTrainingViews,
};
pub use config::{
    FoundationCcsContextMode, FoundationCcsPhysicsBaselineConfig, FoundationConfig,
    FoundationMs2OutputActivation, FOUNDATION_MS2_SOFTPLUS_BETA_V0138,
};
pub use control::{FoundationFitConfig, FoundationLearningRateSchedule};
pub use corpus::{
    load_foundation_corpus, FoundationCorpus, FoundationCorpusConfig, FoundationCorpusDelimiter,
    FoundationCorpusSourceFormat, FoundationCorpusSourceSpec, FoundationCorpusSourceSummary,
    FoundationRecordProvenance,
};
pub use data::{
    FoundationTrainingRecord, FragmentTarget, ObservedSpectrumPeak, RetentionTimeLabels,
    RetentionTimeObjective, TrainingContext,
};
pub use dataset::{
    parse_modified_peptide, FoundationCcsDerivationMode, FoundationDataset,
    FoundationDatasetLoader, FoundationSchemaCollision, FoundationSchemaField,
    FoundationTableLoadReport, FoundationTableLoadStats, FoundationTableLoaderConfig,
    FoundationTableSchemaReport, FragmentIntensityNormalization, InstrumentVocabulary,
};
pub use diffusion::{
    foundation_diffusion_length_loss, foundation_diffusion_residue_ptm_valid,
    foundation_diffusion_reverse_probabilities, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, foundation_diffusion_x0_loss,
    foundation_peptidoform_neutral_mass, foundation_precursor_mass_consistent,
    foundation_precursor_mass_error_da, foundation_precursor_neutral_mass,
    foundation_spectrum_peptide_alignment_loss, FoundationDiffusionBatch,
    FoundationDiffusionCollator, FoundationDiffusionConfig, FoundationDiffusionOutput,
    FoundationDiffusionVocabulary, FoundationSpectrumEncoder, FoundationSpectrumEncoding,
    PeptideSpectrumDiffusionModel, FOUNDATION_DIFFUSION_CARBAMIDOMETHYL,
    FOUNDATION_DIFFUSION_DEAMIDATED, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_FIRST_RESIDUE,
    FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_OXIDATION,
    FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
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
pub use interaction::FoundationSpectrumCandidateInteractionAdapter;
pub use iterative_refinement::{
    foundation_iterative_refinement_collate, foundation_iterative_refinement_mask_positions,
    load_iterative_refinement_from_unified_checkpoint, validate_iterative_refinement_namespace,
    FoundationIterativeRefinementWarmStartReport,
    FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_OBJECTIVE_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315,
    FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315,
};
pub use loss::{
    contrastive_info_nce_loss, foundation_ms2_loss, multi_task_loss,
    multi_task_loss_with_ms2_config, FoundationLossWeights, FoundationLosses,
    FoundationMs2LossConfig, FoundationMs2Losses, FoundationTargets,
};
pub use metadata::{
    apply_source_metadata, FoundationMetadataApplicationStats, FoundationMetadataMergePolicy,
    FoundationSourceMetadata,
};
pub use model::{
    FoundationMultiTaskOutput, FoundationOutput, PeptideFoundationEncoder,
    PeptideFoundationMultiTaskModel, PrecursorContextBatch,
};
pub use msp::{load_foundation_msp_reader, FoundationMspLoadReport};
pub use normalization::{
    FoundationRegressionNormalization, FoundationRegressionNormalizationStrategy,
    FoundationTargetNormalizationConfig,
};
pub use optimizer::{FoundationAdamW, FoundationAdamWConfig, FoundationOptimizerStep};
pub use reverse_causal::{
    foundation_canonicalize_reverse_causal_token_row, foundation_reverse_causal_token_row,
    load_reverse_causal_from_unified_checkpoint, validate_reverse_causal_namespace,
    FoundationReverseCausalWarmStartReport, FOUNDATION_REVERSE_CAUSAL_DIRECTION_V01313,
    FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313,
};
pub use rt_harmonization::{
    apply_foundation_rt_harmonization, fit_foundation_rt_harmonization,
    FoundationRtCrossSourceConsistency, FoundationRtHarmonizationFitConfig,
    FoundationRtHarmonizationFitResult, FoundationRtHarmonizationTransform,
    FoundationRtSourceCalibrationSummary,
};
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
pub use spectrum::{
    foundation_diffusion_dataset_fingerprint, foundation_diffusion_record_fingerprint,
    FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator,
    FoundationSpectrumConfig, FoundationSpectrumPeak,
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
pub use unified::{
    load_unified_foundation_components, FoundationUnifiedWarmStartReport,
    PeptideFoundationUnifiedModel,
};
pub use wrapper::FoundationModelWrapper;
