//! Native inference of the original trained v0.52 research forward stack.
//!
//! The historical v0.50/v0.51/v0.52 model modules are imported verbatim.
//! This is an opt-in checkpoint reader; it does not alter unified training.
use super::collate::{FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig};
use super::data::{FoundationTrainingRecord, RetentionTimeObjective};
use super::multimodal_v0350::FoundationFragmentContextBatchV0350;
use super::multimodal_v0360::FoundationScalarPhysicsBatchV0360;
use super::multimodal_v0520::{
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520, PeptideFoundationV0520Config,
    PeptideFoundationV0520Model,
};
use super::normalization::FoundationTargetNormalizationConfig;
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct HistoricalMetadata {
    version: u32,
    objective: String,
    architecture: String,
    v0520_config: PeptideFoundationV0520Config,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    completed_updates: usize,
    smoke_mode: bool,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
}

/// One forward prediction. MS2 is indexed by zero-based cleavage and channel.
#[derive(Debug, Clone)]
pub struct ResearchForwardPrediction {
    pub rt: f32,
    pub ms2: Vec<Vec<f32>>,
}

pub struct ResearchV0520Forward {
    model: PeptideFoundationV0520Model,
    collator: FoundationCollator,
    normalization: FoundationTargetNormalizationConfig,
    device: Device,
    max_sequence_len: usize,
    corpus_fingerprint: String,
    benchmark_fingerprint: String,
}

impl ResearchV0520Forward {
    pub fn load(checkpoint: &Path, device: Device) -> Result<Self> {
        let metadata_path = checkpoint.join("metadata.yaml");
        let metadata: HistoricalMetadata = serde_yaml::from_str(
            &fs::read_to_string(&metadata_path)
                .with_context(|| format!("read {}", metadata_path.display()))?,
        )?;
        ensure!(
            metadata.version == 520
                && metadata.objective == "v0520_mobility_aware_pair_representation"
                && metadata.architecture == FOUNDATION_MULTIMODAL_ARCHITECTURE_V0520
                && !metadata.smoke_mode
                && metadata.completed_updates > 0,
            "requires original trained non-smoke v0.52 checkpoint"
        );
        metadata.v0520_config.validate()?;
        metadata.target_normalization.validate()?;
        let max_sequence_len = metadata.v0520_config.base_v0510.base_v0500.max_sequence_len;
        let featurizer = metadata
            .v0520_config
            .base_v0510
            .base_v0500
            .featurizer_config();
        let collator = FoundationCollator::new(
            featurizer,
            FoundationCollatorConfig {
                retention_time_objective: metadata.rt_objective,
                corruption: FoundationCorruptionConfig {
                    residue_mask_probability: 0.0,
                    chemistry_mask_probability: 0.0,
                },
            },
        )?;
        // Instantiate *all* historical namespaces, including student_v052,
        // so checkpoint shape/parameter coverage matches the original exporter.
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideFoundationV0520Model::new(metadata.v0520_config, vb)?;
        let weights = checkpoint.join("model.safetensors");
        varmap
            .load(&weights)
            .with_context(|| format!("load original v0.52 weights {}", weights.display()))?;
        Ok(Self {
            model,
            collator,
            normalization: metadata.target_normalization,
            device,
            max_sequence_len,
            corpus_fingerprint: metadata.corpus_fingerprint,
            benchmark_fingerprint: metadata.benchmark_manifest_fingerprint,
        })
    }

    pub fn verify_fingerprints(&self, corpus: u64, benchmark: u64) -> Result<()> {
        ensure!(
            self.corpus_fingerprint == format!("fnv1a64:{corpus:016x}"),
            "historical checkpoint corpus fingerprint mismatch"
        );
        ensure!(
            self.benchmark_fingerprint == format!("fnv1a64:{benchmark:016x}"),
            "historical checkpoint benchmark fingerprint mismatch"
        );
        Ok(())
    }

    pub fn infer(
        &self,
        records: &[FoundationTrainingRecord],
    ) -> Result<Vec<ResearchForwardPrediction>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let batch = self.collator.collate(records, &self.device, 0)?;
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            records,
            self.max_sequence_len,
            &self.device,
        )?;
        let fragments = FoundationFragmentContextBatchV0350::from_records(
            records,
            &self
                .model
                .config()
                .base_v0510
                .base_v0500
                .featurizer_config(),
            &self.device,
        )?;
        let forward = self.model.property_forward_t(
            &batch.input,
            &batch.context,
            &physics,
            &fragments,
            false,
        )?;
        let rt = self
            .normalization
            .rt
            .denormalize_tensor(&forward.rt)?
            .squeeze(1)?
            .to_vec1::<f32>()?;
        let ms2 = forward.ms2.to_vec3::<f32>()?;
        ensure!(
            rt.len() == records.len() && ms2.len() == records.len(),
            "v0.52 forward batch length mismatch"
        );
        Ok(rt
            .into_iter()
            .zip(ms2)
            .map(|(rt, ms2)| ResearchForwardPrediction { rt, ms2 })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_research_modules_retain_all_student_namespaces() {
        let map = VarMap::new();
        let vb = VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu);
        let _model =
            PeptideFoundationV0520Model::new(PeptideFoundationV0520Config::local_smoke(), vb)
                .unwrap();
        let vars = map.data().lock().unwrap();
        for name in ["student_v050.", "student_v051.", "student_v052."] {
            assert!(
                vars.keys().any(|key| key.starts_with(name)),
                "missing namespace {name}"
            );
        }
    }
}
