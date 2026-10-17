//! Production foundation-model surface.
//!
//! [`FoundationModel`] is the stable train-from-scratch and inference path for the
//! joint forward/inverse model. Historical accepted RT/MS2 and CCS checkpoints
//! remain available through [`FoundationPredictor`] for compatibility and parity.

mod auxiliary_supervision;
mod causal;
pub mod ccs_physics;
mod checkpoint_compat;
pub mod chemistry;
pub mod collate;
pub mod config;
pub mod corpus;
pub mod data;
pub mod dataset;
pub mod diffusion;
pub mod experiment;
pub mod featurize;
pub mod fragment_likelihood;
pub mod fragment_relation;
pub mod inverse_identifier_api_v0752;
pub mod inverse_identifier_batch_v0760;
pub mod inverse_identifier_v0751;
pub mod layers;
pub mod loss;
pub mod metadata;
mod mobility_consensus;
pub mod model;
pub mod msp;
// Frozen research checkpoint chain; not the trainable production architecture.
#[path = "historical_checkpoints/multimodal_v0270.rs"]
mod multimodal_v0270;
#[path = "historical_checkpoints/multimodal_v0310.rs"]
mod multimodal_v0310;
#[path = "historical_checkpoints/multimodal_v0350.rs"]
mod multimodal_v0350;
#[path = "historical_checkpoints/multimodal_v0360.rs"]
mod multimodal_v0360;
#[path = "historical_checkpoints/multimodal_v0500.rs"]
mod multimodal_v0500;
#[path = "historical_checkpoints/multimodal_v0510.rs"]
mod multimodal_v0510;
#[path = "historical_checkpoints/multimodal_v0520.rs"]
mod multimodal_v0520;
pub mod normalization;
pub mod optimizer;
mod pair_encoder;
pub mod predictor;
mod research_v0520;
pub mod rt_harmonization;
pub mod runtime;
pub mod spectrum;
pub mod split;
pub mod training;

pub use ccs_physics::{
    FOUNDATION_CCS_PHYSICS_FEATURE_COUNT, FOUNDATION_CCS_PHYSICS_FEATURE_NAMES,
    FoundationCcsPhysicsFeatureSummary, FoundationCcsPhysicsFitConfig,
    FoundationCcsPhysicsFitResult, FoundationCcsPhysicsMetrics,
    FoundationCcsPhysicsSourceWeightSummary, evaluate_foundation_ccs_physics_baseline,
    fit_foundation_ccs_physics_baseline, fit_foundation_ccs_physics_baseline_source_weighted,
    foundation_ccs_physics_features, foundation_ccs_physics_features_from_values,
    predict_foundation_ccs_physics_native,
};

pub use chemistry::{
    ElementalComposition, ExactGraphModification, FoundationModificationDefinition,
    ModificationAttachmentSite, common_unimod_definition, exact_graph_modification,
};

pub use config::{
    FOUNDATION_MS2_SOFTPLUS_BETA_V0138, FoundationCcsContextMode,
    FoundationCcsPhysicsBaselineConfig, FoundationConfig, FoundationMs2OutputActivation,
};

pub use collate::{
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationTrainingBatch, FoundationTrainingViews,
};

pub use corpus::{
    FoundationCorpus, FoundationCorpusConfig, FoundationCorpusDelimiter,
    FoundationCorpusSourceFormat, FoundationCorpusSourceSpec, FoundationCorpusSourceSummary,
    FoundationRecordProvenance, load_foundation_corpus,
};

pub use data::{
    FoundationTrainingRecord, FragmentTarget, ObservedSpectrumPeak, RetentionTimeLabels,
    RetentionTimeObjective, TrainingContext,
};

pub use dataset::{
    FoundationCcsDerivationMode, FoundationDataset, FoundationDatasetLoader,
    FoundationSchemaCollision, FoundationSchemaField, FoundationTableLoadReport,
    FoundationTableLoadStats, FoundationTableLoaderConfig, FoundationTableSchemaReport,
    FragmentIntensityNormalization, InstrumentVocabulary, parse_modified_peptide,
};

pub use diffusion::{
    FOUNDATION_DIFFUSION_CARBAMIDOMETHYL, FOUNDATION_DIFFUSION_DEAMIDATED,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_FIRST_RESIDUE, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_OPEN_CTERM_MOD,
    FOUNDATION_DIFFUSION_OPEN_NTERM_MOD, FOUNDATION_DIFFUSION_OPEN_RESIDUE_MOD,
    FOUNDATION_DIFFUSION_OXIDATION, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_PHOSPHO,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_OPEN_PTM_MASS_SCALE_DA, FOUNDATION_OPEN_PTM_VOCAB_SIZE,
    FOUNDATION_PEPTIDE_WATER_MASS_DA, FoundationDiffusionBatch, FoundationDiffusionCollator,
    FoundationDiffusionConfig, FoundationDiffusionOutput, FoundationDiffusionVocabulary,
    FoundationOpenPtmTokenRow, FoundationSpectrumEncoder, FoundationSpectrumEncoding,
    PeptideSpectrumDiffusionModel, foundation_diffusion_is_open_modification_token,
    foundation_diffusion_length_loss, foundation_diffusion_open_ptm_mass_loss,
    foundation_diffusion_open_ptm_mass_mae_da, foundation_diffusion_residue_ptm_valid,
    foundation_diffusion_reverse_probabilities, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, foundation_diffusion_x0_loss,
    foundation_peptidoform_neutral_mass, foundation_precursor_mass_consistent,
    foundation_precursor_mass_error_da, foundation_precursor_neutral_mass,
    foundation_spectrum_peptide_alignment_loss,
};

pub use experiment::{
    FOUNDATION_BENCHMARK_MANIFEST_VERSION, FoundationBenchmarkEntry, FoundationBenchmarkManifest,
    FoundationPartition, build_foundation_benchmark_manifest, foundation_dataset_fingerprint,
    foundation_record_fingerprint,
};

pub use featurize::{
    FoundationBatch, FoundationModification, FoundationModificationSite, PeptideGraphFeaturizer,
    PeptidoformInput, exact_graph_modification_for,
};

pub use fragment_likelihood::{
    FOUNDATION_FRAGMENT_LIKELIHOOD_ABS_TOLERANCE_DA_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_ARCHITECTURE_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_MAX_PEAKS_V0230, FOUNDATION_FRAGMENT_LIKELIHOOD_PPM_V0230,
    FOUNDATION_FRAGMENT_LIKELIHOOD_PRIMARY_SCORE_V0230, FoundationFragmentLikelihoodScore,
    foundation_fragment_likelihood_score,
};

pub use fragment_relation::{
    FOUNDATION_FRAGMENT_RELATION_ABS_TOLERANCE_DA_V0240,
    FOUNDATION_FRAGMENT_RELATION_ARCHITECTURE_V0240,
    FOUNDATION_FRAGMENT_RELATION_CANDIDATE_HIDDEN_V0240,
    FOUNDATION_FRAGMENT_RELATION_CORE_CHANNELS_V0240,
    FOUNDATION_FRAGMENT_RELATION_EXPLAINED_INTENSITY_V0240,
    FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240, FOUNDATION_FRAGMENT_RELATION_HIDDEN_V0240,
    FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240,
    FOUNDATION_FRAGMENT_RELATION_MAX_PEAKS_V0240, FOUNDATION_FRAGMENT_RELATION_OBJECTIVE_V0240,
    FOUNDATION_FRAGMENT_RELATION_PEAK_COVERAGE_V0240, FOUNDATION_FRAGMENT_RELATION_POOLED_V0240,
    FOUNDATION_FRAGMENT_RELATION_PPM_V0240, FoundationFragmentCleavageGeometry,
    FoundationFragmentRelationBatch, FoundationFragmentRelationFeatureRows,
    PeptideSpectrumFragmentRelationEnergy, foundation_fragment_cleavage_geometry,
    foundation_fragment_relation_features, foundation_fragment_relation_legacy_log_prior,
    foundation_fragment_relation_validate_mass_geometry,
};

pub use inverse_identifier_api_v0752::{
    FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_API_VERSION_V0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_DEFAULT_TOP_K_V0752,
    FoundationPracticalIdentifierCatalogCandidateV0752, FoundationPracticalIdentifierCatalogV0752,
    FoundationPracticalIdentifierModificationSiteV0752,
    FoundationPracticalIdentifierModificationV0752, FoundationPracticalIdentifierPeakV0752,
    FoundationPracticalIdentifierRequestV0752, FoundationPracticalIdentifierResponseV0752,
    FoundationPracticalIdentifierResultHitV0752, FoundationPracticalIdentifierServiceV0752,
};

pub use inverse_identifier_batch_v0760::{
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_RESPONSE_SCHEMA_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_VERSION_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760,
    FoundationPracticalIdentifierBatchRequestV0760,
    FoundationPracticalIdentifierBatchResponseV0760,
    FoundationPracticalIdentifierBatchServiceV0760,
    FoundationPracticalIdentifierSearchSpaceEntryV0760,
    FoundationPracticalIdentifierSearchSpaceV0760, materialize_catalog_v0760,
};

pub use inverse_identifier_v0751::{
    FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POLICY_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751, FOUNDATION_PRACTICAL_IDENTIFIER_VERSION_V0751,
    FoundationPracticalIdentifierBuildTimingsV0751, FoundationPracticalIdentifierCandidateV0751,
    FoundationPracticalIdentifierHitV0751, FoundationPracticalIdentifierV0751,
};

pub use loss::{
    FoundationLossWeights, FoundationLosses, FoundationMs2LossConfig, FoundationMs2Losses,
    FoundationTargets, contrastive_info_nce_loss, foundation_ms2_loss, multi_task_loss,
    multi_task_loss_with_ms2_config,
};

pub use metadata::{
    FoundationMetadataApplicationStats, FoundationMetadataMergePolicy, FoundationSourceMetadata,
    apply_source_metadata,
};

pub use model::{
    FoundationMultiTaskOutput, FoundationOutput, PeptideFoundationEncoder,
    PeptideFoundationMultiTaskModel, PrecursorContextBatch,
};

pub use msp::{FoundationMspLoadReport, load_foundation_msp_reader};

pub use normalization::{
    FoundationRegressionNormalization, FoundationRegressionNormalizationStrategy,
    FoundationTargetNormalizationConfig,
};

pub use predictor::{FoundationPredictor, FoundationPredictorConfig};

pub use research_v0520::{
    ResearchForwardPrediction, ResearchMobilityPrediction, ResearchV0520Forward,
    ResearchV0520Mobility,
};

pub use runtime::{
    FoundationCheckpointMetadata, FoundationModel, FoundationModelConfig,
    FoundationPeptideBackbone, FoundationRecordPrediction, FoundationSpecialistConfig,
    read_foundation_checkpoint_metadata,
};

pub use training::{
    FoundationEpochLosses, FoundationEvaluationMetrics, FoundationTrainingConfig,
    FoundationTrainingStrategy, FoundationTrainingSummary, load_foundation_records_from_run,
    load_foundation_records_from_run_partition, train_foundation_model,
};

pub use rt_harmonization::{
    FoundationRtCrossSourceConsistency, FoundationRtHarmonizationFitConfig,
    FoundationRtHarmonizationFitResult, FoundationRtHarmonizationTransform,
    FoundationRtSourceCalibrationSummary, apply_foundation_rt_harmonization,
    fit_foundation_rt_harmonization,
};

pub use spectrum::{
    FoundationSpectrum, FoundationSpectrumBatch, FoundationSpectrumCollator,
    FoundationSpectrumConfig, FoundationSpectrumPeak, foundation_diffusion_dataset_fingerprint,
    foundation_diffusion_record_fingerprint,
};

pub use split::{
    FoundationSplitConfig, FoundationSplitIndices, FoundationSplitMode, FoundationSplitSummary,
    foundation_split_group_key, split_foundation_record_indices, split_foundation_records,
};
