//! Production inference wrapper for the selected foundation-property checkpoints.
//!
//! The public surface is deliberately stable and version-free. Historical model
//! module names remain private checkpoint-compatibility details.

use super::checkpoint_compat::{
    CcsCheckpointConfig, CcsModel, FragmentContextBatch, RtMs2CheckpointConfig, RtMs2Model,
    ScalarPhysicsBatch, CCS_CHECKPOINT_ARCHITECTURE, RT_MS2_CHECKPOINT_ARCHITECTURE,
};
use super::collate::{FoundationCollator, FoundationCollatorConfig, FoundationCorruptionConfig};
use super::data::{
    FoundationTrainingRecord, RetentionTimeLabels, RetentionTimeObjective, TrainingContext,
};
use super::diffusion::foundation_peptidoform_neutral_mass;
use super::featurize::{FoundationModification, FoundationModificationSite, PeptidoformInput};
use super::normalization::FoundationTargetNormalizationConfig;
use crate::models::model_interface::{
    PredictionInput, PredictionModel, PredictionModificationSite, PredictionOutput,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

const PROTON_MASS_DA: f64 = 1.007_276_466_77;

#[derive(Debug, Clone, Deserialize)]
struct RtMs2CheckpointMetadata {
    architecture: String,
    #[serde(rename = "v0520_config")]
    config: RtMs2CheckpointConfig,
    rt_objective: RetentionTimeObjective,
    target_normalization: FoundationTargetNormalizationConfig,
    #[serde(default)]
    smoke_mode: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct CcsCheckpointMetadata {
    architecture: String,
    #[serde(rename = "v0380_config")]
    config: CcsCheckpointConfig,
    target_normalization: FoundationTargetNormalizationConfig,
    #[serde(default)]
    smoke_mode: bool,
}

/// Paths for the accepted foundation-property checkpoints.
#[derive(Debug, Clone)]
pub struct FoundationPredictorConfig {
    /// Checkpoint directory containing the accepted RT/MS2 model.
    pub rt_ms2_checkpoint: PathBuf,
    /// Checkpoint directory containing the accepted CCS model.
    pub ccs_checkpoint: PathBuf,
}

impl FoundationPredictorConfig {
    pub fn new(rt_ms2_checkpoint: impl Into<PathBuf>, ccs_checkpoint: impl Into<PathBuf>) -> Self {
        Self {
            rt_ms2_checkpoint: rt_ms2_checkpoint.into(),
            ccs_checkpoint: ccs_checkpoint.into(),
        }
    }
}

/// Production multi-property predictor backed by the selected foundation checkpoints.
///
/// RT/MS2 and CCS intentionally load separate accepted checkpoints because that
/// is the validated production authority. The wrapper presents them as one
/// stable prediction API without pretending they were trained as one checkpoint.
pub struct FoundationPredictor {
    device: Device,
    rt_ms2_model: RtMs2Model,
    ccs_model: CcsModel,
    rt_ms2_collator: FoundationCollator,
    ccs_collator: FoundationCollator,
    rt_normalization: FoundationTargetNormalizationConfig,
    ccs_normalization: FoundationTargetNormalizationConfig,
    // Keep the variable stores alive with the models they initialized.
    _rt_ms2_vars: VarMap,
    _ccs_vars: VarMap,
}

impl FoundationPredictor {
    pub fn load(config: FoundationPredictorConfig, device: Device) -> Result<Self> {
        let rt_ms2_metadata: RtMs2CheckpointMetadata = read_metadata(&config.rt_ms2_checkpoint)
            .with_context(|| {
                format!(
                    "load foundation RT/MS2 metadata from {}",
                    config.rt_ms2_checkpoint.display()
                )
            })?;
        if rt_ms2_metadata.architecture != RT_MS2_CHECKPOINT_ARCHITECTURE
            || rt_ms2_metadata.smoke_mode
        {
            anyhow::bail!(
                "RT/MS2 checkpoint is not the accepted non-smoke foundation architecture: {}",
                rt_ms2_metadata.architecture
            );
        }
        rt_ms2_metadata.config.validate()?;
        rt_ms2_metadata.target_normalization.validate()?;

        let ccs_metadata: CcsCheckpointMetadata = read_metadata(&config.ccs_checkpoint)
            .with_context(|| {
                format!(
                    "load foundation CCS metadata from {}",
                    config.ccs_checkpoint.display()
                )
            })?;
        if ccs_metadata.architecture != CCS_CHECKPOINT_ARCHITECTURE || ccs_metadata.smoke_mode {
            anyhow::bail!(
                "CCS checkpoint is not the accepted non-smoke foundation architecture: {}",
                ccs_metadata.architecture
            );
        }
        ccs_metadata.config.validate()?;
        ccs_metadata.target_normalization.validate()?;

        let mut rt_ms2_vars = VarMap::new();
        let rt_ms2_vb = VarBuilder::from_varmap(&rt_ms2_vars, DType::F32, &device);
        let rt_ms2_model = RtMs2Model::new(rt_ms2_metadata.config.clone(), rt_ms2_vb)?;
        load_weights(&mut rt_ms2_vars, &config.rt_ms2_checkpoint)?;

        let mut ccs_vars = VarMap::new();
        let ccs_vb = VarBuilder::from_varmap(&ccs_vars, DType::F32, &device);
        let ccs_model = CcsModel::new(ccs_metadata.config.clone(), ccs_vb)?;
        load_weights(&mut ccs_vars, &config.ccs_checkpoint)?;

        let rt_ms2_collator = FoundationCollator::new(
            rt_ms2_model.featurizer_config(),
            FoundationCollatorConfig {
                retention_time_objective: rt_ms2_metadata.rt_objective,
                corruption: no_corruption(),
            },
        )?;
        let ccs_collator = FoundationCollator::new(
            ccs_model.featurizer_config(),
            FoundationCollatorConfig {
                retention_time_objective: RetentionTimeObjective::IntrinsicAndObserved,
                corruption: no_corruption(),
            },
        )?;

        Ok(Self {
            device,
            rt_ms2_model,
            ccs_model,
            rt_ms2_collator,
            ccs_collator,
            rt_normalization: rt_ms2_metadata.target_normalization,
            ccs_normalization: ccs_metadata.target_normalization,
            _rt_ms2_vars: rt_ms2_vars,
            _ccs_vars: ccs_vars,
        })
    }

    pub fn predict_batch(&self, inputs: &[PredictionInput]) -> Result<Vec<PredictionOutput>> {
        let records = inputs
            .iter()
            .map(prediction_record)
            .collect::<Result<Vec<_>>>()?;
        self.predict_records(&records)
    }

    fn predict_records(
        &self,
        records: &[FoundationTrainingRecord],
    ) -> Result<Vec<PredictionOutput>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let rt_ms2_batch = self.rt_ms2_collator.collate(records, &self.device, 0)?;
        let rt_ms2_physics = ScalarPhysicsBatch::from_records(
            records,
            self.rt_ms2_model.max_sequence_len(),
            &self.device,
        )?;
        let fragment = FragmentContextBatch::from_records(
            records,
            &self.rt_ms2_model.featurizer_config(),
            &self.device,
        )?;
        let rt_ms2 = self.rt_ms2_model.predict(
            &rt_ms2_batch.input,
            &rt_ms2_batch.context,
            &rt_ms2_physics,
            &fragment,
        )?;
        let rt = self
            .rt_normalization
            .rt
            .denormalize_tensor(&rt_ms2.rt)?
            .squeeze(1)?
            .to_vec1::<f32>()?;
        let ms2 = rt_ms2.ms2.to_vec3::<f32>()?;

        let ccs_batch = self.ccs_collator.collate(records, &self.device, 0)?;
        let ccs_physics = ScalarPhysicsBatch::from_records(
            records,
            self.ccs_model.max_sequence_len(),
            &self.device,
        )?;
        let ccs_output =
            self.ccs_model
                .predict(&ccs_batch.input, &ccs_batch.context, &ccs_physics)?;
        let base_ccs = self
            .ccs_normalization
            .ccs
            .denormalize_tensor(&ccs_output.base_ccs_model)?;
        let factor = bruker_ccs_factor(&ccs_batch.context)?;
        let base_mobility = base_ccs.broadcast_div(&factor)?;
        let ccs = (&base_mobility + &ccs_output.mobility_residual_native)?
            .broadcast_mul(&factor)?
            .squeeze(1)?
            .to_vec1::<f32>()?;

        if rt.len() != records.len() || ms2.len() != records.len() || ccs.len() != records.len() {
            anyhow::bail!("foundation predictor returned inconsistent batch lengths");
        }

        Ok((0..records.len())
            .map(|i| PredictionOutput {
                rt: Some(rt[i]),
                ccs: Some(ccs[i]),
                ms2: Some(ms2[i].clone()),
            })
            .collect())
    }
}

impl PredictionModel for FoundationPredictor {
    fn predict_batch(&self, inputs: &[PredictionInput]) -> Result<Vec<PredictionOutput>> {
        FoundationPredictor::predict_batch(self, inputs)
    }
}

fn read_metadata<T: for<'de> Deserialize<'de>>(checkpoint: &Path) -> Result<T> {
    let path = checkpoint.join("metadata.yaml");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("read foundation checkpoint metadata {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("parse foundation checkpoint metadata {}", path.display()))
}

fn load_weights(varmap: &mut VarMap, checkpoint: &Path) -> Result<()> {
    let path = checkpoint.join("model.safetensors");
    varmap
        .load(&path)
        .with_context(|| format!("load foundation checkpoint weights {}", path.display()))
}

fn no_corruption() -> FoundationCorruptionConfig {
    FoundationCorruptionConfig {
        residue_mask_probability: 0.0,
        chemistry_mask_probability: 0.0,
    }
}

fn prediction_record(input: &PredictionInput) -> Result<FoundationTrainingRecord> {
    if input.sequence.is_empty() {
        anyhow::bail!("foundation prediction sequence cannot be empty");
    }
    let charge = input.charge.filter(|charge| *charge > 0).ok_or_else(|| {
        anyhow::anyhow!("foundation RT/CCS/MS2 prediction requires positive charge")
    })?;
    let peptide_len = input.sequence.chars().count();
    let modifications = input
        .modifications
        .iter()
        .map(|modification| {
            let (site, residue_index) = match modification.site {
                PredictionModificationSite::Residue(index) => {
                    if index >= peptide_len {
                        anyhow::bail!(
                            "foundation prediction modification residue index {index} is outside peptide length {peptide_len}"
                        );
                    }
                    (FoundationModificationSite::Residue(index), index)
                }
                PredictionModificationSite::NTerm => (FoundationModificationSite::NTerm, 0),
                PredictionModificationSite::CTerm => {
                    (FoundationModificationSite::CTerm, peptide_len.saturating_sub(1))
                }
            };
            Ok(match modification.unimod_id {
                Some(unimod_id) => FoundationModification::unimod(
                    site,
                    residue_index,
                    unimod_id,
                    modification.mass_delta,
                ),
                None => FoundationModification::mass_delta_at_site(
                    site,
                    residue_index,
                    modification.mass_delta,
                ),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let peptidoform = PeptidoformInput {
        sequence: input.sequence.clone(),
        modifications,
    };

    let precursor_mz = match input.precursor_mz {
        Some(mz) if mz.is_finite() && mz > 0.0 => mz,
        Some(mz) => anyhow::bail!(
            "foundation prediction precursor m/z must be positive and finite, got {mz}"
        ),
        None => {
            let neutral_mass =
                foundation_peptidoform_neutral_mass(&peptidoform).map_err(anyhow::Error::msg)?;
            ((neutral_mass + f64::from(charge) * PROTON_MASS_DA) / f64::from(charge)) as f32
        }
    };

    Ok(FoundationTrainingRecord {
        peptidoform,
        retention_time: RetentionTimeLabels::default(),
        ccs: None,
        fragments: Vec::new(),
        observed_spectrum_peaks: Vec::new(),
        context: TrainingContext {
            charge: Some(charge),
            precursor_mz: Some(precursor_mz),
            nce: input.nce.filter(|value| value.is_finite()),
            instrument_id: input.instrument_id,
            instrument_name: input.instrument_name.clone(),
            ion_mobility: None,
            gradient_seconds: None,
        },
        run_id: None,
    })
}

fn bruker_ccs_factor(context: &super::model::PrecursorContextBatch) -> Result<Tensor> {
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
    use crate::models::model_interface::{PredictionInput, PredictionModification};

    #[test]
    fn production_input_maps_terminal_and_residue_modifications() {
        let mut input = PredictionInput::unmodified("PEPTIDE");
        input.charge = Some(2);
        input.modifications = vec![
            PredictionModification {
                site: PredictionModificationSite::Residue(2),
                mass_delta: 15.994_915,
                unimod_id: Some(35),
            },
            PredictionModification {
                site: PredictionModificationSite::NTerm,
                mass_delta: 42.010_567,
                unimod_id: Some(1),
            },
        ];
        let record = prediction_record(&input).unwrap();
        assert_eq!(record.peptidoform.modifications.len(), 2);
        assert!(record.context.precursor_mz.unwrap() > 0.0);
    }
}
