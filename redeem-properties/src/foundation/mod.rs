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
pub mod chemistry_decoder;
pub mod chemistry_diffusion;
pub mod cleavage_graph;
pub mod collate;
pub mod compatibility;
pub mod config;
pub mod control;
pub mod corpus;
pub mod data;
pub mod dataset;
pub mod diffusion;
pub mod direct_decoder;
pub mod experiment;
pub mod featurize;
mod fragment_grounded_v0330;
pub mod fragment_likelihood;
pub mod fragment_relation;
pub mod interaction;
pub mod inverse_reward_v0290;
pub mod iterative_refinement;
pub mod layers;
pub mod loss;
pub mod metadata;
pub mod model;
pub mod msp;
pub mod multimodal_v0260;
pub mod multimodal_v0270;
pub mod multimodal_v0280;
pub mod multimodal_v0300;
pub mod multimodal_v0310;
pub mod multimodal_v0340;
pub mod multimodal_v0350;
pub mod multimodal_v0360;
pub mod multimodal_v0380;
pub mod multimodal_v0390;
pub mod normalization;
pub mod optimizer;
pub mod reverse_causal;
pub mod rt_harmonization;
pub mod run;
pub mod sampling;
pub mod sequence_reward;
pub mod spectrum;
pub mod split;
pub mod structured_edit;
pub mod trainer;
pub mod unified;
pub mod wrapper;

pub use causal::{
    foundation_causal_conditioning_margin_loss, foundation_causal_next_token_loss,
    foundation_causal_sequence_mean_nlls, foundation_fragment_causal_rerank_score,
    load_causal_from_diffusion_checkpoint, FoundationCausalBatch, FoundationCausalCollator,
    FoundationCausalContext, FoundationCausalInputBatch, FoundationCausalOutput,
    FoundationCausalWarmStartReport, PeptideSpectrumCausalModel,
    FOUNDATION_CAUSAL_RERANK_POLICY_V0123, FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123,
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
pub use chemistry_decoder::{
    load_chemistry_decoder_from_unified_checkpoint, ChemistryDecoderWarmStartReport,
    ChemistrySuffixMassLattice, ChemistryTransitionBatch, ChemistryTransitionFeaturizer,
    PeptideSpectrumChemistryDecoder, FOUNDATION_CHEMISTRY_DECODER_ARCHITECTURE_V0200,
    FOUNDATION_CHEMISTRY_DECODER_OBJECTIVE_V0200,
    FOUNDATION_CHEMISTRY_FRAGMENT_ABS_TOLERANCE_DA_V0200, FOUNDATION_CHEMISTRY_FRAGMENT_PPM_V0200,
    FOUNDATION_CHEMISTRY_SUFFIX_BIN_DA_V0200, FOUNDATION_CHEMISTRY_TRANSITION_FEATURE_DIM_V0200,
};
pub use chemistry_diffusion::{
    foundation_chemistry_diffusion_argmax_refine, foundation_chemistry_diffusion_final_mass_valid,
    foundation_chemistry_diffusion_partition_isolated,
    foundation_chemistry_diffusion_project_mass_valid,
    foundation_chemistry_diffusion_refinement_timesteps,
    foundation_chemistry_diffusion_row_neutral_mass,
    load_chemistry_diffusion_from_v0200_checkpoint, ChemistryDiffusionFeatureBatch,
    ChemistryDiffusionFeaturizer, ChemistryDiffusionWarmStartReport,
    PeptideSpectrumChemistryDiffusionModel, FOUNDATION_CHEMISTRY_DIFFUSION_ARCHITECTURE_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_FULL_STATE_FEATURE_DIM_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_INITIAL_BEAM_WIDTH_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_OBJECTIVE_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_START_TIMESTEP_V0210,
    FOUNDATION_CHEMISTRY_DIFFUSION_REFINEMENT_STEPS_V0210,
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
pub use compatibility::{
    foundation_compatibility_listwise_loss, load_compatibility_from_unified_checkpoint,
    FoundationCompatibilityOutput, FoundationCompatibilitySpectrumContext,
    FoundationCompatibilityWarmStartReport, FoundationSpectrumPeptideCompatibilityModel,
    FOUNDATION_COMPATIBILITY_NAMESPACE_V0170,
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
    foundation_diffusion_is_open_modification_token, foundation_diffusion_length_loss,
    foundation_diffusion_open_ptm_mass_loss, foundation_diffusion_open_ptm_mass_mae_da,
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_reverse_probabilities,
    foundation_diffusion_token_mass_da, foundation_diffusion_token_residue,
    foundation_diffusion_x0_loss, foundation_peptidoform_neutral_mass,
    foundation_precursor_mass_consistent, foundation_precursor_mass_error_da,
    foundation_precursor_neutral_mass, foundation_spectrum_peptide_alignment_loss,
    FoundationDiffusionBatch, FoundationDiffusionCollator, FoundationDiffusionConfig,
    FoundationDiffusionOutput, FoundationDiffusionVocabulary, FoundationOpenPtmTokenRow,
    FoundationSpectrumEncoder, FoundationSpectrumEncoding, PeptideSpectrumDiffusionModel,
    FOUNDATION_DIFFUSION_CARBAMIDOMETHYL, FOUNDATION_DIFFUSION_DEAMIDATED,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_FIRST_RESIDUE, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_OPEN_CTERM_MOD,
    FOUNDATION_DIFFUSION_OPEN_NTERM_MOD, FOUNDATION_DIFFUSION_OPEN_RESIDUE_MOD,
    FOUNDATION_DIFFUSION_OXIDATION, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_PHOSPHO,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_OPEN_PTM_MASS_SCALE_DA, FOUNDATION_OPEN_PTM_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
pub use direct_decoder::{
    foundation_direct_beam_search, foundation_direct_conditioning_loss,
    foundation_direct_prefix_competitive_loss, foundation_direct_shuffled_order,
    foundation_mass_stratified_beam_search, load_direct_decoder_from_unified_checkpoint,
    DirectDecoderBeamCandidate, DirectDecoderBeamConfig, DirectDecoderWarmStartReport,
    DirectPrefixCompetition, MassStratifiedBeamConfig, PeptideSpectrumDirectDecoder,
    FOUNDATION_DIRECT_CONDITIONING_MARGIN_V0190, FOUNDATION_DIRECT_CONDITIONING_WEIGHT_V0190,
    FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0190, FOUNDATION_DIRECT_DECODER_OBJECTIVE_V0191,
    FOUNDATION_DIRECT_PREFIX_MARGIN_V0191,
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

pub use fragment_grounded_v0330::{
    FragmentGroundedTransitionHeadV0330, FOUNDATION_FRAGMENT_GROUNDED_ARCHITECTURE_V0330,
    FOUNDATION_FRAGMENT_GROUNDED_FEATURE_DIM_V0330, FOUNDATION_FRAGMENT_GROUNDED_HIDDEN_V0330,
    FOUNDATION_FRAGMENT_GROUNDED_JOINT_V0330, FOUNDATION_FRAGMENT_GROUNDED_OBJECTIVE_V0330,
};
pub use fragment_likelihood::{
    foundation_fragment_likelihood_score, FoundationFragmentLikelihoodScore,
    FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_ARCHITECTURE_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230, FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_PRIMARY_SCORE_V0230,
};
pub use fragment_relation::{
    foundation_fragment_cleavage_geometry, foundation_fragment_relation_features,
    foundation_fragment_relation_legacy_log_prior,
    foundation_fragment_relation_validate_mass_geometry, FoundationFragmentCleavageGeometry,
    FoundationFragmentRelationBatch, FoundationFragmentRelationFeatureRows,
    PeptideSpectrumFragmentRelationEnergy, FOUNDATION_FRAGMENT_RELATION_ABS_TOLERANCE_DA_V0240,
    FOUNDATION_FRAGMENT_RELATION_ARCHITECTURE_V0240,
    FOUNDATION_FRAGMENT_RELATION_CANDIDATE_HIDDEN_V0240,
    FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240,
    FOUNDATION_FRAGMENT_RELATION_EXPLAINED_INTENSITY_V0240,
    FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240, FOUNDATION_FRAGMENT_RELATION_HIDDEN_V0240,
    FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240,
    FOUNDATION_FRAGMENT_RELATION_MAX_PEAKS_V0240, FOUNDATION_FRAGMENT_RELATION_OBJECTIVE_V0240,
    FOUNDATION_FRAGMENT_RELATION_PEAK_COVERAGE_V0240, FOUNDATION_FRAGMENT_RELATION_POOLED_V0240,
    FOUNDATION_FRAGMENT_RELATION_PPM_V0240,
};
pub use interaction::FoundationSpectrumCandidateInteractionAdapter;
pub use inverse_reward_v0290::{
    foundation_multimodal_ms2_loss_v0290, foundation_multimodal_relation_margin_loss_v0290,
    FoundationFragmentContextBatchV0290, FoundationMultimodalMs2LossesV0290,
    PeptideFoundationInverseRewardV0290Config, PeptideFoundationInverseRewardV0290Model,
    FOUNDATION_INVERSE_REWARD_ARCHITECTURE_V0290, FOUNDATION_INVERSE_REWARD_BEAM_WIDTH_V0290,
    FOUNDATION_INVERSE_REWARD_GROUPS_PER_STEP_V0290, FOUNDATION_INVERSE_REWARD_POLICY_WEIGHT_V0290,
    FOUNDATION_INVERSE_REWARD_REFERENCE_WEIGHT_V0290,
    FOUNDATION_INVERSE_REWARD_SUPERVISED_WEIGHT_V0290, FOUNDATION_INVERSE_REWARD_TOP_K_V0290,
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0290, FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0290,
};
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
pub use multimodal_v0260::{
    foundation_multimodal_ms2_loss_v0260, foundation_multimodal_relation_margin_loss_v0260,
    FoundationMultimodalForwardOutputV0260, FoundationMultimodalMs2LossesV0260,
    PeptideFoundationMultimodalForwardV0260, PeptideFoundationMultimodalV0260Model,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0260, FOUNDATION_MULTIMODAL_MS2_COSINE_WEIGHT_V0260,
    FOUNDATION_MULTIMODAL_MS2_INTENSITY_WEIGHT_V0260,
    FOUNDATION_MULTIMODAL_MS2_PRESENCE_WEIGHT_V0260, FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0260,
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0260,
};
pub use multimodal_v0270::{
    foundation_multimodal_ms2_loss_v0270, foundation_multimodal_relation_margin_loss_v0270,
    FoundationFragmentContextBatchV0270, FoundationMultimodalForwardOutputV0270,
    FoundationMultimodalMs2LossesV0270, PeptideFoundationMultimodalForwardV0270,
    PeptideFoundationMultimodalV0270Config, PeptideFoundationMultimodalV0270Model,
    FOUNDATION_FRAGMENT_CHANNELS_V0270, FOUNDATION_FRAGMENT_CONTINUOUS_FEATURES_V0270,
    FOUNDATION_FRAGMENT_TRANSFORMER_FF_DIM_V0270, FOUNDATION_FRAGMENT_TRANSFORMER_HEADS_V0270,
    FOUNDATION_FRAGMENT_TRANSFORMER_LAYERS_V0270, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0270,
    FOUNDATION_MULTIMODAL_MS2_COSINE_WEIGHT_V0270,
    FOUNDATION_MULTIMODAL_MS2_INTENSITY_WEIGHT_V0270,
    FOUNDATION_MULTIMODAL_MS2_PRESENCE_WEIGHT_V0270, FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0270,
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0270, FOUNDATION_RT_SPECIALIST_FF_DIM_V0270,
    FOUNDATION_RT_SPECIALIST_HEADS_V0270, FOUNDATION_RT_SPECIALIST_LAYERS_V0270,
};

pub use multimodal_v0280::{
    foundation_multimodal_ms2_loss_v0280, foundation_multimodal_relation_margin_loss_v0280,
    FoundationFragmentContextBatchV0280, FoundationMultimodalForwardOutputV0280,
    FoundationMultimodalMs2LossesV0280, PeptideFoundationMultimodalV0280Config,
    PeptideFoundationMultimodalV0280Model, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0280,
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0280, FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0280,
    FOUNDATION_TASK_CONDITION_BOTTLENECK_V0280, FOUNDATION_TASK_CONDITION_COUNT_V0280,
    FOUNDATION_TASK_CONDITION_EMBED_DIM_V0280,
};

pub use multimodal_v0300::{
    foundation_fragment_representation_aux_loss_v0300, foundation_multimodal_ms2_loss_v0300,
    foundation_multimodal_relation_margin_loss_v0300, FoundationFragmentContextBatchV0300,
    FoundationMultimodalForwardOutputV0300, FoundationMultimodalMs2LossesV0300,
    PeptideFoundationMultimodalV0300Config, PeptideFoundationMultimodalV0300Model,
    FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0300,
    FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0300,
    FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0300, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0300,
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0300, FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0300,
};

pub use multimodal_v0310::{
    foundation_fragment_representation_aux_loss_v0310, foundation_multimodal_ms2_loss_v0310,
    foundation_multimodal_relation_margin_loss_v0310, FoundationFragmentContextBatchV0310,
    FoundationMultimodalForwardOutputV0310, FoundationMultimodalMs2LossesV0310,
    PeptideFoundationMultimodalV0310Config, PeptideFoundationMultimodalV0310Model,
    FOUNDATION_FRAGMENT_REPRESENTATION_AUX_CONTEXT_V0310,
    FOUNDATION_FRAGMENT_REPRESENTATION_AUX_HIDDEN_V0310,
    FOUNDATION_FRAGMENT_REPRESENTATION_AUX_WEIGHT_V0310, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0310,
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0310, FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0310,
    FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0310, FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0310,
    FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0310,
};

pub use multimodal_v0340::{
    foundation_multimodal_ms2_loss_v0340, foundation_multimodal_relation_margin_loss_v0340,
    FoundationFragmentContextBatchV0340, FoundationMultimodalForwardOutputV0340,
    FoundationMultimodalMs2LossesV0340, PeptideFoundationMultimodalV0340Config,
    PeptideFoundationMultimodalV0340Model, FOUNDATION_MS2_ATTENTION_HEADS_V0340,
    FOUNDATION_MS2_DECODER_HIDDEN_V0340, FOUNDATION_MS2_FF_DIM_V0340,
    FOUNDATION_MS2_TRANSFORMER_LAYERS_V0340, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0340,
    FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0340, FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0340,
    FOUNDATION_RT_ATTENTION_HEADS_V0340, FOUNDATION_RT_FF_DIM_V0340,
    FOUNDATION_RT_LOCAL_CHANNELS_V0340, FOUNDATION_RT_TRANSFORMER_LAYERS_V0340,
    FOUNDATION_SPECIALIST_DIM_V0340,
};

pub use multimodal_v0350::{
    foundation_fragment_representation_aux_loss_v0350, foundation_multimodal_ms2_loss_v0350,
    foundation_multimodal_relation_margin_loss_v0350, FoundationFragmentContextBatchV0350,
    FoundationMultimodalForwardOutputV0350, FoundationMultimodalMs2LossesV0350,
    FoundationRepresentationAuxOutputV0350, PeptideFoundationMultimodalV0350Config,
    PeptideFoundationMultimodalV0350Model, FOUNDATION_FRAGMENT_AUX_HIDDEN_V0350,
    FOUNDATION_FRAGMENT_AUX_WEIGHT_V0350, FOUNDATION_MS2_CONTEXT_FF_DIM_V0350,
    FOUNDATION_MS2_CONTEXT_HEADS_V0350, FOUNDATION_MS2_CONTEXT_INSTRUMENT_DIM_V0350,
    FOUNDATION_MS2_CONTEXT_LAYERS_V0350, FOUNDATION_MS2_PEARSON_WEIGHT_V0350,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0350, FOUNDATION_MULTIMODAL_PROPERTY_HIDDEN_V0350,
    FOUNDATION_MULTIMODAL_RELATION_MARGIN_V0350, FOUNDATION_PROPERTY_REFINEMENT_FF_DIM_V0350,
    FOUNDATION_PROPERTY_REFINEMENT_HEADS_V0350, FOUNDATION_PROPERTY_REFINEMENT_LAYERS_V0350,
    FOUNDATION_REPRESENTATION_CHEMISTRY_WEIGHT_V0350,
    FOUNDATION_REPRESENTATION_CONTRASTIVE_WEIGHT_V0350,
    FOUNDATION_REPRESENTATION_MASKED_WEIGHT_V0350, FOUNDATION_RT_ROBUST_DELTA_V0350,
    FOUNDATION_RT_ROBUST_WEIGHT_V0350,
};

pub use multimodal_v0360::{
    FoundationMultimodalForwardOutputV0360, FoundationScalarOutputV0360,
    FoundationScalarPhysicsBatchV0360, PeptideFoundationMultimodalV0360Config,
    PeptideFoundationMultimodalV0360Model, FOUNDATION_CCS_SCALAR_CONTEXT_DIM_V0360,
    FOUNDATION_CCS_SPECIALIST_FF_DIM_V0360, FOUNDATION_CCS_SPECIALIST_HEADS_V0360,
    FOUNDATION_CCS_SPECIALIST_HIDDEN_V0360, FOUNDATION_CCS_SPECIALIST_LAYERS_V0360,
    FOUNDATION_CCS_STRETCH_TARGET_MAE_V0360, FOUNDATION_CCS_TARGET_MAE_V0360,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0360, FOUNDATION_RT_SCALAR_CONTEXT_DIM_V0360,
    FOUNDATION_RT_SPECIALIST_HIDDEN_V0360, FOUNDATION_RT_STRETCH_TARGET_MAE_V0360,
    FOUNDATION_SCALAR_ROBUST_DELTA_V0360,
};

pub use multimodal_v0380::{
    FoundationMobilityOutputV0380, PeptideFoundationMultimodalV0380Config,
    PeptideFoundationMultimodalV0380Model, FOUNDATION_CCS_CONTEXT_FF_DIM_V0380,
    FOUNDATION_CCS_CONTEXT_HEADS_V0380, FOUNDATION_CCS_CONTEXT_LAYERS_V0380,
    FOUNDATION_CCS_MOBILITY_HIDDEN_V0380, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0380,
};

pub use multimodal_v0390::{
    FoundationMobilityOutputV0390, PeptideFoundationMultimodalV0390Config,
    PeptideFoundationMultimodalV0390Model, FOUNDATION_CCS_CONFORMER_FF_DIM_V0390,
    FOUNDATION_CCS_CONFORMER_HEADS_V0390, FOUNDATION_CCS_CONFORMER_LAYERS_V0390,
    FOUNDATION_CCS_MOBILITY_HIDDEN_V0390, FOUNDATION_MULTIMODAL_ARCHITECTURE_V0390,
};

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
pub use sequence_reward::{
    foundation_group_relative_advantages_v0290, foundation_group_relative_policy_loss_v0290,
    foundation_reference_nll_anchor_v0290, foundation_sequence_reward_v0290,
    FoundationSequenceRewardV0290, FOUNDATION_SEQUENCE_REWARD_EPS_V0290,
    FOUNDATION_SEQUENCE_REWARD_FRAGMENT_WEIGHT_V0290, FOUNDATION_SEQUENCE_REWARD_MASS_WEIGHT_V0290,
    FOUNDATION_SEQUENCE_REWARD_OBJECTIVE_V0290, FOUNDATION_SEQUENCE_REWARD_SEQUENCE_WEIGHT_V0290,
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
pub use structured_edit::{
    foundation_structured_edit_argmax, foundation_structured_edit_finalize,
    foundation_structured_edit_gate_loss, foundation_structured_edit_open_row,
    foundation_structured_edit_partition_isolated, foundation_structured_edit_set_targets,
    load_structured_editor_from_v0210_checkpoint, FoundationStructuredEditOutput,
    PeptideSpectrumStructuredEditor, StructuredEditWarmStartReport,
    FOUNDATION_STRUCTURED_EDIT_ARCHITECTURE_V0220,
    FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220,
    FOUNDATION_STRUCTURED_EDIT_GATE_WEIGHT_V0220,
    FOUNDATION_STRUCTURED_EDIT_INITIAL_BEAM_WIDTH_V0220,
    FOUNDATION_STRUCTURED_EDIT_LENGTH_WEIGHT_V0220, FOUNDATION_STRUCTURED_EDIT_OBJECTIVE_V0220,
    FOUNDATION_STRUCTURED_EDIT_PASSES_V0220, FOUNDATION_STRUCTURED_EDIT_TRAIN_INITIALIZERS_V0220,
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
