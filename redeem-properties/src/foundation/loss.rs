//! Multi-task loss composition for peptide foundation-model pretraining.

use super::model::FoundationMultiTaskOutput;
use candle_core::{Result, Tensor};
use candle_nn::loss;

/// Optional labels supplied by heterogeneous proteomics training corpora.
///
/// Any unavailable label remains `None`; the corresponding loss is skipped.
#[derive(Debug, Clone, Default)]
pub struct FoundationTargets {
    /// RT or normalized RT target `[batch, 1]`.
    pub rt: Option<Tensor>,
    /// CCS target `[batch, 1]`.
    pub ccs: Option<Tensor>,
    /// MS2 fragment-intensity target matching the model MS2 tensor.
    pub ms2: Option<Tensor>,
    /// Flattened masked-residue class labels `[num_masked_positions]`.
    pub masked_residue_classes: Option<Tensor>,
    /// Flattened indices into `[batch * sequence]` selecting corrupted positions.
    pub masked_residue_indices: Option<Tensor>,
    /// Chemistry-reconstruction target matching the model chemistry tensor.
    pub chemistry: Option<Tensor>,
}

/// Relative contributions of supervised and self-supervised objectives.
#[derive(Debug, Clone, Copy)]
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
}

impl Default for FoundationLossWeights {
    fn default() -> Self {
        Self {
            rt: 1.0,
            ccs: 1.0,
            ms2: 1.0,
            masked_residue: 0.25,
            chemistry: 0.25,
        }
    }
}

/// Individual loss terms and their weighted total.
#[derive(Debug, Clone)]
pub struct FoundationLosses {
    /// Weighted sum used for backpropagation.
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

/// Compose losses while tolerating missing labels in heterogeneous datasets.
pub fn multi_task_loss(
    output: &FoundationMultiTaskOutput,
    targets: &FoundationTargets,
    weights: FoundationLossWeights,
) -> Result<FoundationLosses> {
    let rt = targets
        .rt
        .as_ref()
        .map(|target| loss::mse(&output.rt, target))
        .transpose()?;
    let ccs = targets
        .ccs
        .as_ref()
        .map(|target| loss::mse(&output.ccs, target))
        .transpose()?;
    let ms2 = targets
        .ms2
        .as_ref()
        .map(|target| loss::mse(&output.ms2, target))
        .transpose()?;
    let chemistry = targets
        .chemistry
        .as_ref()
        .map(|target| loss::mse(&output.chemistry_reconstruction, target))
        .transpose()?;
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
