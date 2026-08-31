//! Multi-task and contrastive losses for peptide foundation-model pretraining.
//!
//! A foundation batch can mix observations that have different subsets of RT,
//! CCS, and MS2 labels.  Dense target tensors are therefore paired with masks
//! and reduced only over observed values.  Self-supervision is provided by
//! masked-residue classification, masked chemistry reconstruction, and a
//! symmetric InfoNCE objective between independently corrupted peptide views.

use super::model::FoundationMultiTaskOutput;
use candle_core::{DType, Result, Tensor};
use candle_nn::loss;
use serde::{Deserialize, Serialize};

/// Optional labels supplied by heterogeneous proteomics training corpora.
///
/// Any unavailable task remains `None`; partially observed tasks use a dense
/// value tensor plus a binary mask of the same or broadcastable shape.
#[derive(Debug, Clone, Default)]
pub struct FoundationTargets {
    /// RT or normalized RT target `[batch, 1]`.
    pub rt: Option<Tensor>,
    /// Binary RT observation mask `[batch, 1]`.
    pub rt_mask: Option<Tensor>,
    /// CCS target `[batch, 1]`.
    pub ccs: Option<Tensor>,
    /// Binary CCS observation mask `[batch, 1]`.
    pub ccs_mask: Option<Tensor>,
    /// MS2 fragment-intensity target matching the model MS2 tensor.
    pub ms2: Option<Tensor>,
    /// Binary MS2 observation mask matching the model MS2 tensor.
    pub ms2_mask: Option<Tensor>,
    /// Flattened masked-residue class labels `[num_masked_positions]`.
    pub masked_residue_classes: Option<Tensor>,
    /// Flattened indices into `[batch * sequence]` selecting corrupted positions.
    pub masked_residue_indices: Option<Tensor>,
    /// Chemistry-reconstruction target `[batch, residues, atom_feature_dim]`.
    pub chemistry: Option<Tensor>,
    /// Binary chemistry mask `[batch, residues, 1]` or full target shape.
    pub chemistry_mask: Option<Tensor>,
}

/// Relative contributions of supervised and self-supervised objectives.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationLossWeights {
    /// Weight for RT/iRT regression.
    pub rt: f64,
    /// Weight for CCS regression.
    pub ccs: f64,
    /// Weight for MS2 intensity regression.
    pub ms2: f64,
    /// Weight for masked-residue reconstruction.
    pub masked_residue: f64,
    /// Weight for residue chemistry reconstruction.
    pub chemistry: f64,
    /// Weight for contrastive view alignment.
    pub contrastive: f64,
}

impl Default for FoundationLossWeights {
    fn default() -> Self {
        Self {
            rt: 1.0,
            ccs: 1.0,
            ms2: 1.0,
            masked_residue: 0.25,
            chemistry: 0.25,
            contrastive: 0.10,
        }
    }
}

/// Individual loss terms and their weighted supervised/self-supervised total.
#[derive(Debug, Clone)]
pub struct FoundationLosses {
    /// Weighted sum excluding the optional contrastive term, which requires a
    /// second model view and is composed by the trainer.
    pub total: Tensor,
    /// RT loss when RT labels were supplied.
    pub rt: Option<Tensor>,
    /// CCS loss when CCS labels were supplied.
    pub ccs: Option<Tensor>,
    /// MS2 loss when fragment labels were supplied.
    pub ms2: Option<Tensor>,
    /// Masked-residue loss when corruption labels were supplied.
    pub masked_residue: Option<Tensor>,
    /// Chemistry reconstruction loss when targets were supplied.
    pub chemistry: Option<Tensor>,
}

/// Compose all single-view losses while tolerating partially missing labels.
pub fn multi_task_loss(
    output: &FoundationMultiTaskOutput,
    targets: &FoundationTargets,
    weights: FoundationLossWeights,
) -> Result<FoundationLosses> {
    let rt = paired_masked_mse(&output.rt, targets.rt.as_ref(), targets.rt_mask.as_ref())?;
    let ccs = paired_masked_mse(&output.ccs, targets.ccs.as_ref(), targets.ccs_mask.as_ref())?;
    let ms2 = paired_masked_mse(&output.ms2, targets.ms2.as_ref(), targets.ms2_mask.as_ref())?;
    let chemistry = paired_masked_mse(
        &output.chemistry_reconstruction,
        targets.chemistry.as_ref(),
        targets.chemistry_mask.as_ref(),
    )?;
    let masked_residue = match (
        targets.masked_residue_indices.as_ref(),
        targets.masked_residue_classes.as_ref(),
    ) {
        (Some(indices), Some(classes)) => {
            let (batch, sequence, classes_count) = output.residue_logits.dims3()?;
            let flat_logits = output
                .residue_logits
                .reshape((batch * sequence, classes_count))?;
            let selected_logits = flat_logits.index_select(indices, 0)?;
            Some(loss::cross_entropy(&selected_logits, classes)?)
        }
        (None, None) => None,
        _ => candle_core::bail!(
            "masked_residue_indices and masked_residue_classes must either both be present or both be absent"
        ),
    };

    let mut weighted = Vec::new();
    if let Some(value) = &rt {
        weighted.push(value.affine(weights.rt, 0.0)?);
    }
    if let Some(value) = &ccs {
        weighted.push(value.affine(weights.ccs, 0.0)?);
    }
    if let Some(value) = &ms2 {
        weighted.push(value.affine(weights.ms2, 0.0)?);
    }
    if let Some(value) = &masked_residue {
        weighted.push(value.affine(weights.masked_residue, 0.0)?);
    }
    if let Some(value) = &chemistry {
        weighted.push(value.affine(weights.chemistry, 0.0)?);
    }
    if weighted.is_empty() {
        candle_core::bail!("multi_task_loss requires at least one available target");
    }
    let mut total = weighted[0].clone();
    for term in weighted.iter().skip(1) {
        total = (total + term)?;
    }

    Ok(FoundationLosses {
        total,
        rt,
        ccs,
        ms2,
        masked_residue,
        chemistry,
    })
}

/// Symmetric InfoNCE loss between two independently corrupted peptide views.
///
/// Each row is treated as the positive pair for the corresponding row in the
/// other view; other samples in the batch provide in-batch negatives.  Batch
/// size should therefore be greater than one for a useful training signal.
pub fn contrastive_info_nce_loss(
    first_projection: &Tensor,
    second_projection: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    if !(temperature > 0.0 && temperature.is_finite()) {
        candle_core::bail!("contrastive temperature must be positive and finite");
    }
    let (batch, dim) = first_projection.dims2()?;
    let (second_batch, second_dim) = second_projection.dims2()?;
    if batch != second_batch || dim != second_dim {
        candle_core::bail!(
            "contrastive projection mismatch: first [{batch}, {dim}], second [{second_batch}, {second_dim}]"
        );
    }
    if batch == 0 {
        candle_core::bail!("contrastive loss requires a non-empty batch");
    }

    let first = l2_normalize(first_projection)?;
    let second = l2_normalize(second_projection)?;
    let second_t = second.transpose(0, 1)?.contiguous()?;
    let first_t = first.transpose(0, 1)?.contiguous()?;
    let logits_ab = first.matmul(&second_t)?.affine(1.0 / temperature, 0.0)?;
    let logits_ba = second.matmul(&first_t)?.affine(1.0 / temperature, 0.0)?;
    let labels =
        Tensor::arange(0u32, batch as u32, first_projection.device())?.to_dtype(DType::U32)?;
    let loss_ab = loss::cross_entropy(&logits_ab, &labels)?;
    let loss_ba = loss::cross_entropy(&logits_ba, &labels)?;
    (loss_ab + loss_ba)?.affine(0.5, 0.0)
}

fn paired_masked_mse(
    prediction: &Tensor,
    target: Option<&Tensor>,
    mask: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    match (target, mask) {
        (Some(target), Some(mask)) => Ok(Some(masked_mse(prediction, target, mask)?)),
        (None, None) => Ok(None),
        _ => candle_core::bail!("foundation target and observation mask must be supplied together"),
    }
}

fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        candle_core::bail!(
            "masked MSE shape mismatch: prediction {:?}, target {:?}",
            prediction.dims(),
            target.dims()
        );
    }
    let mask = mask.broadcast_as(prediction.dims())?;
    let squared = (prediction - target)?.sqr()?.broadcast_mul(&mask)?;
    let numerator = squared.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator)
}

fn l2_normalize(values: &Tensor) -> Result<Tensor> {
    let denominator = values
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?
        .unsqueeze(1)?;
    values.broadcast_div(&denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn contrastive_loss_is_finite_for_matching_views() {
        let device = Device::Cpu;
        let first = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
        let second = Tensor::new(&[[0.9f32, 0.1], [0.1, 0.9]], &device).unwrap();
        let value = contrastive_info_nce_loss(&first, &second, 0.1)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(value.is_finite());
        assert!(value >= 0.0);
    }
}
