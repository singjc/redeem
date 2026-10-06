//! Production foundation-model surface.
//!
//! [`FoundationModel`] is the stable train-from-scratch and inference path for the
//! joint forward/inverse model. Historical accepted RT/MS2 and CCS checkpoints
//! remain available through [`FoundationPredictor`] for compatibility and parity.

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
pub mod model;
pub mod msp;
pub mod normalization;
pub mod predictor;
pub mod rt_harmonization;
pub mod runtime;
pub mod spectrum;
pub mod split;
pub mod training;

pub use ccs_physics::{
    evaluate_foundation_ccs_physics_baseline, fit_foundation_ccs_physics_baseline,
    fit_foundation_ccs_physics_baseline_source_weighted, foundation_ccs_physics_features,
    foundation_ccs_physics_features_from_values, predict_foundation_ccs_physics_native,
    FoundationCcsPhysicsFeatureSummary, FoundationCcsPhysicsFitConfig,
    FoundationCcsPhysicsFitResult, FoundationCcsPhysicsMetrics,
    FoundationCcsPhysicsSourceWeightSummary, FOUNDATION_CCS_PHYSICS_FEATURE_COUNT,
    FOUNDATION_CCS_PHYSICS_FEATURE_NAMES,
};

pub use chemistry::{
    common_unimod_definition, exact_graph_modification, ElementalComposition,
    ExactGraphModification, FoundationModificationDefinition, ModificationAttachmentSite,
};

pub use config::{
    FoundationCcsContextMode, FoundationCcsPhysicsBaselineConfig, FoundationConfig,
    FoundationMs2OutputActivation, FOUNDATION_MS2_SOFTPLUS_BETA_V0138,
};

pub use collate::{
    FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig,
    FoundationTrainingBatch, FoundationTrainingViews,
};

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

pub use experiment::{
    build_foundation_benchmark_manifest, foundation_dataset_fingerprint,
    foundation_record_fingerprint, FoundationBenchmarkEntry, FoundationBenchmarkManifest,
    FoundationPartition, FOUNDATION_BENCHMARK_MANIFEST_VERSION,
};

pub use featurize::{
    exact_graph_modification_for, FoundationBatch, FoundationModification,
    FoundationModificationSite, PeptideGraphFeaturizer, PeptidoformInput,
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

pub use inverse_identifier_api_v0752::{
    FoundationPracticalIdentifierCatalogCandidateV0752, FoundationPracticalIdentifierCatalogV0752,
    FoundationPracticalIdentifierModificationSiteV0752,
    FoundationPracticalIdentifierModificationV0752, FoundationPracticalIdentifierPeakV0752,
    FoundationPracticalIdentifierRequestV0752, FoundationPracticalIdentifierResponseV0752,
    FoundationPracticalIdentifierResultHitV0752, FoundationPracticalIdentifierServiceV0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_API_SCHEMA_V0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_API_VERSION_V0752,
    FOUNDATION_PRACTICAL_IDENTIFIER_DEFAULT_TOP_K_V0752,
};

pub use inverse_identifier_batch_v0760::{
    materialize_catalog_v0760, FoundationPracticalIdentifierBatchRequestV0760,
    FoundationPracticalIdentifierBatchResponseV0760,
    FoundationPracticalIdentifierBatchServiceV0760,
    FoundationPracticalIdentifierSearchSpaceEntryV0760,
    FoundationPracticalIdentifierSearchSpaceV0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_RESPONSE_SCHEMA_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_SCHEMA_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_BATCH_VERSION_V0760,
    FOUNDATION_PRACTICAL_IDENTIFIER_SEARCH_SPACE_SCHEMA_V0760,
};

pub use inverse_identifier_v0751::{
    FoundationPracticalIdentifierBuildTimingsV0751, FoundationPracticalIdentifierCandidateV0751,
    FoundationPracticalIdentifierHitV0751, FoundationPracticalIdentifierV0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_ARCHITECTURE_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POLICY_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_CANDIDATE_POOL_V0751,
    FOUNDATION_PRACTICAL_IDENTIFIER_SCORE_V0751, FOUNDATION_PRACTICAL_IDENTIFIER_VERSION_V0751,
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

pub use predictor::{FoundationPredictor, FoundationPredictorConfig};

pub use runtime::{
    read_foundation_checkpoint_metadata, FoundationCheckpointMetadata, FoundationModel,
    FoundationModelConfig, FoundationRecordPrediction,
};

pub use training::{
    load_foundation_records_from_run, train_foundation_model, FoundationEpochLosses,
    FoundationTrainingConfig, FoundationTrainingSummary,
};

pub use rt_harmonization::{
    apply_foundation_rt_harmonization, fit_foundation_rt_harmonization,
    FoundationRtCrossSourceConsistency, FoundationRtHarmonizationFitConfig,
    FoundationRtHarmonizationFitResult, FoundationRtHarmonizationTransform,
    FoundationRtSourceCalibrationSummary,
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
