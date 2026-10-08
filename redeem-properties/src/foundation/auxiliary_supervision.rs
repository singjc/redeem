//! Research-source-derived peptide auxiliary supervision for the unified model.
//!
//! Targets use TRAIN records only at the call site; they are composition/charge
//! descriptors, NOT computed conformers or experimental peptide contacts.

use super::data::FoundationTrainingRecord;
use super::diffusion::foundation_peptidoform_neutral_mass;
use super::featurize::FoundationModificationSite;
use anyhow::Result;
use candle_core::{Device, Tensor};

const FOUNDATION_V0500_PAIR_CLASS_COUNT: usize = 6;
const FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM: usize = 8;
const FOUNDATION_V0520_CONFORMATION_PROXY_DIM: usize = 14;

pub(crate) fn chemistry_summary_targets(
    records: &[FoundationTrainingRecord],
    max_sequence_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(records.len() * FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM);
    for record in records {
        let sequence = &record.peptidoform.sequence;
        let length = sequence.chars().count().max(1);
        let mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        let acidic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'D' | 'E'))
            .count();
        let basic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'K' | 'R' | 'H'))
            .count();
        let aromatic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'F' | 'W' | 'Y' | 'H'))
            .count();
        let hydrophobic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y'))
            .count();
        let modifications = record.peptidoform.modifications.len();
        let terminal_modifications = record
            .peptidoform
            .modifications
            .iter()
            .filter(|modification| {
                matches!(
                    modification.site,
                    FoundationModificationSite::NTerm | FoundationModificationSite::CTerm
                )
            })
            .count();
        let denom = length as f32;
        values.extend_from_slice(&[
            (mass / 3000.0) as f32,
            length as f32 / max_sequence_len.max(1) as f32,
            acidic as f32 / denom,
            basic as f32 / denom,
            aromatic as f32 / denom,
            hydrophobic as f32 / denom,
            modifications as f32 / 8.0,
            terminal_modifications as f32 / 2.0,
        ]);
    }
    Tensor::from_vec(
        values,
        (records.len(), FOUNDATION_V0500_CHEMISTRY_SUMMARY_DIM),
        device,
    )
    .map_err(Into::into)
}

pub(crate) fn pair_interaction_targets(
    records: &[FoundationTrainingRecord],
    max_sequence_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let classes = FOUNDATION_V0500_PAIR_CLASS_COUNT;
    let mut target = vec![0.0f32; records.len() * max_sequence_len * max_sequence_len * classes];
    let mut mask = vec![0.0f32; target.len()];
    for (batch_index, record) in records.iter().enumerate() {
        let residues = record.peptidoform.sequence.chars().collect::<Vec<_>>();
        if residues.len() > max_sequence_len {
            anyhow::bail!("pair-target sequence length exceeds model maximum");
        }
        for i in 0..residues.len() {
            for j in 0..residues.len() {
                if i == j {
                    continue;
                }
                let left = residues[i];
                let right = residues[j];
                let left_acid = matches!(left, 'D' | 'E');
                let right_acid = matches!(right, 'D' | 'E');
                let left_basic = matches!(left, 'K' | 'R' | 'H');
                let right_basic = matches!(right, 'K' | 'R' | 'H');
                let left_donor =
                    matches!(left, 'K' | 'R' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let right_donor =
                    matches!(right, 'K' | 'R' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let left_acceptor =
                    matches!(left, 'D' | 'E' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let right_acceptor =
                    matches!(right, 'D' | 'E' | 'H' | 'N' | 'Q' | 'S' | 'T' | 'Y' | 'C');
                let left_aromatic = matches!(left, 'F' | 'W' | 'Y' | 'H');
                let right_aromatic = matches!(right, 'F' | 'W' | 'Y' | 'H');
                let left_hydrophobic =
                    matches!(left, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y');
                let right_hydrophobic =
                    matches!(right, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y');
                let terminal =
                    i == 0 || j == 0 || i + 1 == residues.len() || j + 1 == residues.len();
                let labels = [
                    (left_acid && right_basic) || (left_basic && right_acid),
                    (left_donor && right_acceptor) || (right_donor && left_acceptor),
                    left_aromatic && right_aromatic,
                    left_hydrophobic && right_hydrophobic,
                    left_basic && right_basic,
                    terminal,
                ];
                let base = (((batch_index * max_sequence_len + i) * max_sequence_len + j) * classes)
                    as usize;
                for class_index in 0..classes {
                    target[base + class_index] = if labels[class_index] { 1.0 } else { 0.0 };
                    mask[base + class_index] = 1.0;
                }
            }
        }
    }
    let shape = (records.len(), max_sequence_len, max_sequence_len, classes);
    Ok((
        Tensor::from_vec(target, shape, device)?,
        Tensor::from_vec(mask, shape, device)?,
    ))
}

pub(crate) fn conformation_proxy_targets_v0520(
    records: &[FoundationTrainingRecord],
    max_sequence_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(records.len() * FOUNDATION_V0520_CONFORMATION_PROXY_DIM);
    for record in records {
        let sequence = &record.peptidoform.sequence;
        let length = sequence.chars().count().max(1);
        let length_f = length as f64;
        let mass =
            foundation_peptidoform_neutral_mass(&record.peptidoform).map_err(anyhow::Error::msg)?;
        let charge = f64::from(record.context.charge.unwrap_or(0));
        let precursor_mz = f64::from(record.context.precursor_mz.unwrap_or(0.0));
        let acidic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'D' | 'E'))
            .count();
        let basic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'K' | 'R' | 'H'))
            .count();
        let aromatic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'F' | 'W' | 'Y' | 'H'))
            .count();
        let hydrophobic = sequence
            .chars()
            .filter(|aa| matches!(aa, 'A' | 'V' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y'))
            .count();
        let modifications = record.peptidoform.modifications.len();
        let terminal_modifications = record
            .peptidoform
            .modifications
            .iter()
            .filter(|modification| {
                matches!(
                    modification.site,
                    FoundationModificationSite::NTerm | FoundationModificationSite::CTerm
                )
            })
            .count();
        let denom = length as f32;
        values.extend_from_slice(&[
            (mass / 3000.0) as f32,
            (mass.max(0.0).sqrt() / 60.0) as f32,
            length as f32 / max_sequence_len.max(1) as f32,
            (charge / 6.0) as f32,
            ((charge * charge) / 36.0) as f32,
            (precursor_mz / 2000.0) as f32,
            (charge / length_f) as f32,
            ((mass / length_f) / 200.0) as f32,
            acidic as f32 / denom,
            basic as f32 / denom,
            aromatic as f32 / denom,
            hydrophobic as f32 / denom,
            modifications as f32 / 8.0,
            terminal_modifications as f32 / 2.0,
        ]);
    }
    Tensor::from_vec(
        values,
        (records.len(), FOUNDATION_V0520_CONFORMATION_PROXY_DIM),
        device,
    )
    .map_err(Into::into)
}

pub(crate) fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if prediction.dims() != target.dims() {
        anyhow::bail!(
            "v0.50 masked MSE shape mismatch: prediction {:?}, target {:?}",
            prediction.dims(),
            target.dims()
        );
    }
    let mask = mask.broadcast_as(prediction.dims())?;
    let numerator = (prediction - target)?
        .sqr()?
        .broadcast_mul(&mask)?
        .sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

pub(crate) fn masked_bce_with_logits(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    if logits.dims() != target.dims() {
        anyhow::bail!(
            "v0.50 BCE shape mismatch: logits {:?}, target {:?}",
            logits.dims(),
            target.dims()
        );
    }
    let positive = logits.relu()?;
    let linear = logits.broadcast_mul(target)?;
    let tail = logits.abs()?.affine(-1.0, 0.0)?.exp()?;
    let tail = (tail + 1.0)?.log()?;
    let loss = ((positive - linear)? + tail)?;
    let mask = mask.broadcast_as(logits.dims())?;
    let numerator = loss.broadcast_mul(&mask)?.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::super::data::{RetentionTimeLabels, TrainingContext};
    use super::super::featurize::PeptidoformInput;
    use super::*;

    fn sample(sequence: &str) -> FoundationTrainingRecord {
        FoundationTrainingRecord {
            peptidoform: PeptidoformInput::unmodified(sequence),
            retention_time: RetentionTimeLabels::default(),
            ccs: None,
            fragments: Vec::new(),
            observed_spectrum_peaks: Vec::new(),
            context: TrainingContext {
                charge: Some(2),
                precursor_mz: Some(500.0),
                ..Default::default()
            },
            run_id: None,
        }
    }

    #[test]
    fn historical_pair_classes_exclude_diagonal_and_padding() -> Result<()> {
        let (targets, mask) = pair_interaction_targets(&[sample("DE")], 4, &Device::Cpu)?;
        assert_eq!(targets.dims(), &[1, 4, 4, 6]);
        let mask_values = mask.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(mask_values.iter().filter(|&&v| v > 0.0).count(), 12);
        let target_values = targets.flatten_all()?.to_vec1::<f32>()?;
        assert!(target_values
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0));
        Ok(())
    }

    #[test]
    fn chemistry_and_conformation_proxies_match_research_dimensions() -> Result<()> {
        let samples = [sample("PEPTIDE"), sample("ACDE")];
        let chemistry = chemistry_summary_targets(&samples, 64, &Device::Cpu)?;
        let conformation = conformation_proxy_targets_v0520(&samples, 64, &Device::Cpu)?;
        assert_eq!(chemistry.dims(), &[2, 8]);
        assert_eq!(conformation.dims(), &[2, 14]);
        assert!(chemistry
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        assert!(conformation
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn historical_masked_losses_are_finite_and_differentiable() -> Result<()> {
        let logits = Tensor::zeros((1, 2, 2, 6), candle_core::DType::F32, &Device::Cpu)?;
        let labels = Tensor::ones((1, 2, 2, 6), candle_core::DType::F32, &Device::Cpu)?;
        let mask = Tensor::ones((1, 2, 2, 6), candle_core::DType::F32, &Device::Cpu)?;
        let bce = masked_bce_with_logits(&logits, &labels, &mask)?;
        assert!(bce.to_scalar::<f32>()?.is_finite());
        let mse = masked_mse(
            &Tensor::zeros((1, 8), candle_core::DType::F32, &Device::Cpu)?,
            &Tensor::ones((1, 8), candle_core::DType::F32, &Device::Cpu)?,
            &Tensor::ones((1, 8), candle_core::DType::F32, &Device::Cpu)?,
        )?;
        assert!((mse.to_scalar::<f32>()? - 1.0).abs() < 1e-5);
        Ok(())
    }
}
