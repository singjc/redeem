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

/// Native mobility and CCS outputs from the frozen v0.35 teacher + v0.52 student.
#[derive(Debug, Clone)]
pub struct ResearchMobilityPrediction {
    pub native_mobility: f32,
    pub ccs: f32,
    pub factor: f32,
}

/// Separate wrapper: the proven native RT/MS2 path remains untouched.
pub struct ResearchV0520Mobility {
    student: ResearchV0520Forward,
    teacher: super::multimodal_v0510::FoundationV0350TeacherV0510,
}

#[derive(Deserialize)]
struct HistoricalTeacherMetadata {
    version: u32,
    objective: String,
    completed_steps: usize,
    v0350_config: super::multimodal_v0350::PeptideFoundationMultimodalV0350Config,
    target_normalization: FoundationTargetNormalizationConfig,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
}

impl ResearchV0520Mobility {
    pub fn load(
        student_checkpoint: &Path,
        teacher_checkpoint: &Path,
        device: Device,
    ) -> Result<Self> {
        let student = ResearchV0520Forward::load(student_checkpoint, device)?;
        let path = teacher_checkpoint.join("metadata.yaml");
        let meta: HistoricalTeacherMetadata = serde_yaml::from_str(
            &fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?,
        )?;
        ensure!(
            meta.version == 350
                && meta.objective
                    == "v0350_trainable_forward_representation_context_conditioned_ms2"
                && meta.completed_steps > 0,
            "requires original trained v0.35 teacher"
        );
        meta.v0350_config.validate()?;
        ensure!(
            meta.corpus_fingerprint == student.corpus_fingerprint
                && meta.benchmark_manifest_fingerprint == student.benchmark_fingerprint,
            "v0.35 teacher/v0.52 student fingerprints disagree"
        );
        // v0.52 inherited the v0.35 TRAIN-fitted normalization; do not silently replace it.
        ensure!(
            serde_yaml::to_value(&meta.target_normalization)?
                == serde_yaml::to_value(&student.normalization)?,
            "v0.35 teacher/v0.52 student normalization mismatch"
        );
        let mut variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &student.device);
        let model = super::multimodal_v0350::PeptideFoundationMultimodalV0350Model::new(
            meta.v0350_config,
            vb,
        )?;
        let weights = teacher_checkpoint.join("model.safetensors");
        variables
            .load(&weights)
            .with_context(|| format!("load v0.35 teacher {}", weights.display()))?;
        Ok(Self {
            student,
            teacher: super::multimodal_v0510::FoundationV0350TeacherV0510::new(model),
        })
    }

    pub fn verify_fingerprints(&self, corpus: u64, benchmark: u64) -> Result<()> {
        self.student.verify_fingerprints(corpus, benchmark)
    }

    pub fn infer(
        &self,
        records: &[FoundationTrainingRecord],
    ) -> Result<Vec<ResearchMobilityPrediction>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let batch = self
            .student
            .collator
            .collate(records, &self.student.device, 0)?;
        let physics = FoundationScalarPhysicsBatchV0360::from_records(
            records,
            self.student.max_sequence_len,
            &self.student.device,
        )?;
        let mobility =
            self.student
                .model
                .mobility_aware_t(&batch.input, &batch.context, &physics, false)?;
        let base_ccs_model = self
            .teacher
            .protected_ccs_detached_t(&batch.input, &batch.context)?;
        let base_ccs_native = self
            .student
            .normalization
            .ccs
            .denormalize_tensor(&base_ccs_model)?;
        let factor = original_bruker_ccs_factor(&batch.context)?;
        let base_mobility = base_ccs_native.broadcast_div(&factor)?;
        let predicted_mobility = (&base_mobility + &mobility.mobility_residual_native)?;
        let predicted_ccs = predicted_mobility.broadcast_mul(&factor)?;
        let native = predicted_mobility.to_vec2::<f32>()?;
        let ccs = predicted_ccs.to_vec2::<f32>()?;
        let factors = factor.to_vec2::<f32>()?;
        ensure!(
            native.len() == records.len()
                && ccs.len() == records.len()
                && factors.len() == records.len(),
            "mobility inference output count mismatch"
        );
        Ok((0..records.len())
            .map(|i| ResearchMobilityPrediction {
                native_mobility: native[i][0],
                ccs: ccs[i][0],
                factor: factors[i][0],
            })
            .collect())
    }
}

// Exact tensor operations and constant from historical
// foundation_train_multimodal_v0520.rs::bruker_ccs_factor.
fn original_bruker_ccs_factor(
    context: &super::model::PrecursorContextBatch,
) -> Result<candle_core::Tensor> {
    let charge = context.charge.unsqueeze(1)?;
    let mz = context.precursor_mz.unsqueeze(1)?;
    let neutral_mass = charge.broadcast_mul(&mz)?;
    let reduced_mass = neutral_mass
        .affine(28.0, 0.0)?
        .broadcast_div(&neutral_mass.affine(1.0, 28.0)?)?;
    let denominator = reduced_mass.sqrt()?.clamp(1.0e-6, f64::INFINITY)?;
    Ok(charge
        .affine(1059.62245, 0.0)?
        .broadcast_div(&denominator)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_bruker_factor_matches_analytical_conversion() {
        let device = Device::Cpu;
        let mut context = super::super::model::PrecursorContextBatch::unknown(1, &device).unwrap();
        context.charge = candle_core::Tensor::new(&[2.0f32], &device).unwrap();
        context.precursor_mz = candle_core::Tensor::new(&[400.0f32], &device).unwrap();
        let actual = original_bruker_ccs_factor(&context)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        let neutral_mass = 800.0f64;
        let reduced_mass = 28.0 * neutral_mass / (neutral_mass + 28.0);
        let expected = 2.0 * 1059.62245 / reduced_mass.sqrt();
        assert!((f64::from(actual) - expected).abs() < 1.0e-4);
    }

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
