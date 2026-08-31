//! Heterogeneous batching and self-supervised corruption for foundation training.
//!
//! Public proteomics corpora rarely provide RT, CCS, MS2, and acquisition
//! metadata for every observation.  [`FoundationCollator`] therefore emits
//! dense tensors together with per-task masks so a mixed batch can contribute
//! only the labels it actually contains.  The same collator creates masked
//! sequence/chemistry views for self-supervised pretraining.

use super::config::FoundationConfig;
use super::data::{FoundationTrainingRecord, RetentionTimeObjective};
use super::featurize::{residue_token_id, FoundationBatch, PeptideGraphFeaturizer};
use super::loss::FoundationTargets;
use super::model::PrecursorContextBatch;
use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

/// Corruption probabilities used for masked sequence/chemistry pretraining.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationCorruptionConfig {
    /// Fraction of valid residue identities replaced by the mask/unknown token.
    pub residue_mask_probability: f32,
    /// Fraction of valid residue-local atom features zeroed for reconstruction.
    pub chemistry_mask_probability: f32,
}

impl Default for FoundationCorruptionConfig {
    fn default() -> Self {
        Self {
            residue_mask_probability: 0.15,
            chemistry_mask_probability: 0.15,
        }
    }
}

impl FoundationCorruptionConfig {
    fn validate(self) -> Result<Self> {
        if !(0.0..=1.0).contains(&self.residue_mask_probability)
            || !(0.0..=1.0).contains(&self.chemistry_mask_probability)
        {
            candle_core::bail!("foundation corruption probabilities must be in [0, 1]");
        }
        Ok(self)
    }
}

/// Options controlling label selection and self-supervised views.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationCollatorConfig {
    /// RT target selection policy.
    pub retention_time_objective: RetentionTimeObjective,
    /// Corruption probabilities for the first training view.
    pub corruption: FoundationCorruptionConfig,
}

impl Default for FoundationCollatorConfig {
    fn default() -> Self {
        Self {
            retention_time_objective: RetentionTimeObjective::Normalized,
            corruption: FoundationCorruptionConfig::default(),
        }
    }
}

/// One tensorized training view plus all supervised/self-supervised labels.
#[derive(Debug, Clone)]
pub struct FoundationTrainingBatch {
    /// Potentially corrupted model input.
    pub input: FoundationBatch,
    /// Acquisition/precursor context.
    pub context: PrecursorContextBatch,
    /// Dense labels and masks.
    pub targets: FoundationTargets,
}

/// Pair of independently corrupted views used by the contrastive objective.
#[derive(Debug, Clone)]
pub struct FoundationTrainingViews {
    /// First view; supervised losses are normally evaluated on this view.
    pub first: FoundationTrainingBatch,
    /// Second independently corrupted view used for contrastive alignment.
    pub second: FoundationTrainingBatch,
}

/// Converts heterogeneous CPU records into Candle tensors.
#[derive(Debug, Clone)]
pub struct FoundationCollator {
    model_config: FoundationConfig,
    config: FoundationCollatorConfig,
    featurizer: PeptideGraphFeaturizer,
}

impl FoundationCollator {
    /// Construct a collator using the same shape configuration as the model.
    pub fn new(model_config: FoundationConfig, config: FoundationCollatorConfig) -> Result<Self> {
        config.corruption.validate()?;
        let featurizer = PeptideGraphFeaturizer::new(model_config.clone())?;
        Ok(Self {
            model_config,
            config,
            featurizer,
        })
    }

    /// Collate one independently corrupted training view.
    pub fn collate(
        &self,
        records: &[FoundationTrainingRecord],
        device: &Device,
        seed: u64,
    ) -> Result<FoundationTrainingBatch> {
        if records.is_empty() {
            candle_core::bail!("cannot collate an empty foundation batch");
        }
        let peptidoforms: Vec<_> = records
            .iter()
            .map(|record| record.peptidoform.clone())
            .collect();
        let clean = self.featurizer.featurize(&peptidoforms, device)?;
        let chemistry_targets = mean_atom_features(&clean)?;
        let (input, masked_indices, masked_classes, chemistry_mask) =
            self.corrupt(clean, records, device, seed)?;
        // Token id zero acts as a mask token only where `residue_mask` remains
        // one, so padding and masked residues stay distinguishable.
        let context = self.context_tensors(records, device)?;
        let targets = self.target_tensors(
            records,
            chemistry_targets,
            chemistry_mask,
            masked_indices,
            masked_classes,
            device,
        )?;
        Ok(FoundationTrainingBatch {
            input,
            context,
            targets,
        })
    }

    /// Collate two independent corruption views for symmetric contrastive loss.
    pub fn collate_views(
        &self,
        records: &[FoundationTrainingRecord],
        device: &Device,
        seed: u64,
    ) -> Result<FoundationTrainingViews> {
        Ok(FoundationTrainingViews {
            first: self.collate(records, device, seed)?,
            second: self.collate(records, device, seed ^ 0x9e37_79b9_7f4a_7c15)?,
        })
    }

    fn corrupt(
        &self,
        clean: FoundationBatch,
        records: &[FoundationTrainingRecord],
        device: &Device,
        seed: u64,
    ) -> Result<(
        FoundationBatch,
        Option<Tensor>,
        Option<Tensor>,
        Option<Tensor>,
    )> {
        let b = records.len();
        let l = self.model_config.max_sequence_len;
        let a = self.model_config.max_atoms_per_residue;
        let f = self.model_config.atom_feature_dim;
        let mut rng = DeterministicRng::new(seed);

        let mut residue_ids = vec![0u32; b * l];
        let mut chemistry_keep = vec![1.0f32; b * l];
        let mut chemistry_mask = vec![0.0f32; b * l];
        let mut masked_indices = Vec::<u32>::new();
        let mut masked_classes = Vec::<u32>::new();
        let mut valid_positions = Vec::<usize>::new();

        for (batch_index, record) in records.iter().enumerate() {
            for (residue_index, residue) in record.peptidoform.sequence.chars().enumerate() {
                if residue_index >= l {
                    break;
                }
                let flat_index = batch_index * l + residue_index;
                valid_positions.push(flat_index);
                let token = residue_token_id(residue) as u32;
                residue_ids[flat_index] = token;
                if rng.next_f32() < self.config.corruption.residue_mask_probability {
                    masked_indices.push(flat_index as u32);
                    masked_classes.push(token);
                    residue_ids[flat_index] = 0;
                }
                if rng.next_f32() < self.config.corruption.chemistry_mask_probability {
                    chemistry_keep[flat_index] = 0.0;
                    chemistry_mask[flat_index] = 1.0;
                }
            }
        }

        if self.config.corruption.residue_mask_probability > 0.0 && masked_indices.is_empty() {
            if let Some(&flat_index) = valid_positions.first() {
                masked_indices.push(flat_index as u32);
                masked_classes.push(residue_ids[flat_index]);
                residue_ids[flat_index] = 0;
            }
        }
        if self.config.corruption.chemistry_mask_probability > 0.0
            && !chemistry_mask.iter().any(|value| *value > 0.0)
        {
            if let Some(&flat_index) = valid_positions.first() {
                chemistry_keep[flat_index] = 0.0;
                chemistry_mask[flat_index] = 1.0;
            }
        }

        let chemistry_keep =
            Tensor::from_vec(chemistry_keep, (b, l, 1, 1), device)?.broadcast_as((b, l, a, f))?;
        let atom_features = clean.atom_features.broadcast_mul(&chemistry_keep)?;
        let residue_ids = Tensor::from_vec(residue_ids, (b, l), device)?.to_dtype(DType::U32)?;
        let chemistry_mask = if chemistry_mask.iter().any(|value| *value > 0.0) {
            Some(Tensor::from_vec(chemistry_mask, (b, l, 1), device)?)
        } else {
            None
        };
        let masked_indices = if masked_indices.is_empty() {
            None
        } else {
            Some(
                Tensor::from_vec(masked_indices.clone(), masked_indices.len(), device)?
                    .to_dtype(DType::U32)?,
            )
        };
        let masked_classes = if masked_classes.is_empty() {
            None
        } else {
            Some(
                Tensor::from_vec(masked_classes.clone(), masked_classes.len(), device)?
                    .to_dtype(DType::U32)?,
            )
        };

        Ok((
            FoundationBatch {
                atom_features,
                adjacency: clean.adjacency,
                atom_mask: clean.atom_mask,
                residue_ids,
                residue_mask: clean.residue_mask,
            },
            masked_indices,
            masked_classes,
            chemistry_mask,
        ))
    }

    fn context_tensors(
        &self,
        records: &[FoundationTrainingRecord],
        device: &Device,
    ) -> Result<PrecursorContextBatch> {
        let charge: Vec<f32> = records
            .iter()
            .map(|record| record.context.charge.unwrap_or(0) as f32)
            .collect();
        let nce: Vec<f32> = records
            .iter()
            .map(|record| record.context.nce.unwrap_or(0.0))
            .collect();
        let instrument_ids: Vec<u32> = records
            .iter()
            .map(|record| {
                record
                    .context
                    .instrument_id
                    .unwrap_or(0)
                    .min(self.model_config.instrument_vocab_size.saturating_sub(1) as u32)
            })
            .collect();
        Ok(PrecursorContextBatch {
            charge: Tensor::from_vec(charge, records.len(), device)?,
            nce: Tensor::from_vec(nce, records.len(), device)?,
            instrument_ids: Tensor::from_vec(instrument_ids, records.len(), device)?
                .to_dtype(DType::U32)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn target_tensors(
        &self,
        records: &[FoundationTrainingRecord],
        chemistry_targets: Tensor,
        chemistry_mask: Option<Tensor>,
        masked_indices: Option<Tensor>,
        masked_classes: Option<Tensor>,
        device: &Device,
    ) -> Result<FoundationTargets> {
        let b = records.len();
        let l = self.model_config.max_sequence_len;
        let channels = self.model_config.ms2_fragment_channels;

        let mut rt_values = vec![0.0f32; b];
        let mut rt_mask = vec![0.0f32; b];
        let mut ccs_values = vec![0.0f32; b];
        let mut ccs_mask = vec![0.0f32; b];
        let mut ms2_values = vec![0.0f32; b * (l - 1) * channels];
        let mut ms2_mask = vec![0.0f32; b * (l - 1) * channels];

        for (batch_index, record) in records.iter().enumerate() {
            let rt = match self.config.retention_time_objective {
                RetentionTimeObjective::Normalized => record.retention_time.normalized,
                RetentionTimeObjective::Observed => record.retention_time.observed_seconds,
                // The current foundation RT head is intrinsic.  In combined
                // mode we therefore train it only on normalized RT and retain
                // observed RT in the record for a future LC-context head.
                RetentionTimeObjective::IntrinsicAndObserved => record.retention_time.normalized,
            };
            if let Some(rt) = rt.filter(|value| value.is_finite()) {
                rt_values[batch_index] = rt;
                rt_mask[batch_index] = 1.0;
            }
            if let Some(ccs) = record.ccs.filter(|value| value.is_finite()) {
                ccs_values[batch_index] = ccs;
                ccs_mask[batch_index] = 1.0;
            }
            for fragment in &record.fragments {
                if fragment.cleavage_index >= l - 1 || fragment.channel >= channels {
                    continue;
                }
                let index =
                    (batch_index * (l - 1) + fragment.cleavage_index) * channels + fragment.channel;
                ms2_values[index] = fragment.intensity;
                ms2_mask[index] = 1.0;
            }
        }

        Ok(FoundationTargets {
            rt: mask_present(&rt_mask)
                .then(|| Tensor::from_vec(rt_values, (b, 1), device))
                .transpose()?,
            rt_mask: mask_present(&rt_mask)
                .then(|| Tensor::from_vec(rt_mask, (b, 1), device))
                .transpose()?,
            ccs: mask_present(&ccs_mask)
                .then(|| Tensor::from_vec(ccs_values, (b, 1), device))
                .transpose()?,
            ccs_mask: mask_present(&ccs_mask)
                .then(|| Tensor::from_vec(ccs_mask, (b, 1), device))
                .transpose()?,
            ms2: mask_present(&ms2_mask)
                .then(|| Tensor::from_vec(ms2_values, (b, l - 1, channels), device))
                .transpose()?,
            ms2_mask: mask_present(&ms2_mask)
                .then(|| Tensor::from_vec(ms2_mask, (b, l - 1, channels), device))
                .transpose()?,
            masked_residue_classes: masked_classes,
            masked_residue_indices: masked_indices,
            chemistry: chemistry_mask.as_ref().map(|_| chemistry_targets),
            chemistry_mask,
        })
    }
}

fn mask_present(mask: &[f32]) -> bool {
    mask.iter().any(|value| *value > 0.0)
}

fn mean_atom_features(batch: &FoundationBatch) -> Result<Tensor> {
    let sums = batch.atom_features.sum(2)?;
    let denominator = batch
        .atom_mask
        .sum(2)?
        .clamp(1.0, f64::INFINITY)?
        .unsqueeze(2)?;
    sums.broadcast_div(&denominator)
}

/// Tiny deterministic PRNG used to make corruption reproducible without adding
/// another crate dependency to `redeem-properties`.
#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xa076_1d64_78bd_642f,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_f32(&mut self) -> f32 {
        let value = (self.next_u64() >> 40) as u32;
        value as f32 / ((1u32 << 24) - 1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationModification, FragmentTarget, PeptidoformInput, RetentionTimeLabels,
        TrainingContext,
    };

    #[test]
    fn heterogeneous_labels_create_masks_instead_of_dropping_batch() {
        let config = FoundationConfig {
            max_sequence_len: 12,
            transformer_layers: 1,
            ..FoundationConfig::default()
        };
        let collator = FoundationCollator::new(
            config.clone(),
            FoundationCollatorConfig {
                corruption: FoundationCorruptionConfig {
                    residue_mask_probability: 1.0,
                    chemistry_mask_probability: 1.0,
                },
                ..FoundationCollatorConfig::default()
            },
        )
        .unwrap();
        let records = vec![
            FoundationTrainingRecord {
                peptidoform: PeptidoformInput::unmodified("PEPTIDEK"),
                retention_time: RetentionTimeLabels {
                    normalized: Some(25.0),
                    observed_seconds: None,
                },
                ccs: None,
                fragments: vec![FragmentTarget {
                    cleavage_index: 1,
                    channel: 0,
                    intensity: 1.0,
                }],
                context: TrainingContext {
                    charge: Some(2),
                    nce: Some(27.0),
                    ..TrainingContext::default()
                },
                run_id: None,
            },
            FoundationTrainingRecord {
                peptidoform: PeptidoformInput {
                    sequence: "ACDMK".to_string(),
                    modifications: vec![FoundationModification {
                        residue_index: 3,
                        mass_delta: 15.994915,
                    }],
                },
                retention_time: RetentionTimeLabels::default(),
                ccs: Some(430.0),
                fragments: Vec::new(),
                context: TrainingContext {
                    charge: Some(3),
                    ..TrainingContext::default()
                },
                run_id: None,
            },
        ];
        let batch = collator.collate(&records, &Device::Cpu, 1234).unwrap();
        assert_eq!(batch.targets.rt.as_ref().unwrap().dims(), &[2, 1]);
        assert_eq!(batch.targets.ccs.as_ref().unwrap().dims(), &[2, 1]);
        assert_eq!(batch.targets.ms2.as_ref().unwrap().dims(), &[2, 11, 8]);
        assert!(batch.targets.masked_residue_indices.is_some());
        assert!(batch.targets.chemistry_mask.is_some());
    }
}
