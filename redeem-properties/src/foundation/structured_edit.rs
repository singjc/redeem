//! v0.22 spectrum-conditioned structured peptide edit transducer.
//!
//! v0.21 established that a bidirectional spectrum/chemistry model can denoise
//! synthetic categorical corruption but does not repair the structured errors
//! made by the v0.20 inverse decoder. v0.22 therefore removes synthetic
//! corruption from the scientific objective. Training inputs are frozen v0.20
//! top-1 hypotheses from TRAIN, and the model learns the target length,
//! KEEP-vs-CHANGE decisions, and replacement identities directly from those
//! on-policy near misses.
//!
//! This is not a candidate reranker and it is not a diffusion schedule. The
//! complete current hypothesis is encoded bidirectionally once, arbitrary
//! positions may change jointly, and a final bounded physical projection only
//! restores exact precursor-mass/grammar validity. Observed fragment peaks
//! remain soft evidence throughout.

use super::chemistry_diffusion::{
    foundation_chemistry_diffusion_final_mass_valid,
    foundation_chemistry_diffusion_project_mass_valid, ChemistryDiffusionFeatureBatch,
    PeptideSpectrumChemistryDiffusionModel,
};
use super::diffusion::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_residue,
    FoundationDiffusionBatch, FoundationDiffusionConfig, FoundationDiffusionOutput,
    FOUNDATION_DIFFUSION_CARBAMIDOMETHYL, FOUNDATION_DIFFUSION_DEAMIDATED,
    FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_NTERM_ACETYL,
    FOUNDATION_DIFFUSION_OXIDATION, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_PHOSPHO,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_DIFFUSION_VOCAB_SIZE,
};
use super::experiment::FoundationPartition;
use super::model::PrecursorContextBatch;
use super::spectrum::FoundationSpectrumBatch;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder, VarMap};
use std::collections::HashSet;
use std::path::Path;

/// Stable v0.22 architecture identifier.
pub const FOUNDATION_STRUCTURED_EDIT_ARCHITECTURE_V0220: &str =
    "v0210_bidirectional_spectrum_chemistry_backbone_plus_on_policy_keep_change_edit_transducer";
/// Stable v0.22 objective identifier.
pub const FOUNDATION_STRUCTURED_EDIT_OBJECTIVE_V0220: &str =
    "real_v0200_top1_target_token_ce_plus_keep_change_ce_plus_length_ce_plus_conditioning_guard";
/// Frozen number of actual v0.20 TRAIN initializers used by the full experiment.
pub const FOUNDATION_STRUCTURED_EDIT_TRAIN_INITIALIZERS_V0220: usize = 4096;
/// Frozen v0.20 beam width used only to materialize one top-1 on-policy state.
pub const FOUNDATION_STRUCTURED_EDIT_INITIAL_BEAM_WIDTH_V0220: usize = 128;
/// Exactly one learned whole-sequence edit pass in the first v0.22 experiment.
pub const FOUNDATION_STRUCTURED_EDIT_PASSES_V0220: usize = 1;
/// Weight of explicit KEEP-vs-CHANGE supervision.
pub const FOUNDATION_STRUCTURED_EDIT_GATE_WEIGHT_V0220: f64 = 0.5;
/// Weight of spectrum/precursor target-length supervision.
pub const FOUNDATION_STRUCTURED_EDIT_LENGTH_WEIGHT_V0220: f64 = 0.25;
/// Fixed pseudo-timestep used only to reuse the historical bidirectional input
/// plumbing. No diffusion/noising process is used by v0.22.
pub const FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220: usize = 1;

/// Output of the structured editor.
#[derive(Debug, Clone)]
pub struct FoundationStructuredEditOutput {
    /// Spectrum/chemistry-conditioned target-token logits and length logits.
    pub sequence: FoundationDiffusionOutput,
    /// KEEP/CHANGE logits `[batch, max_tokens, 2]`.
    pub gate_logits: Tensor,
}

/// v0.22 model. The complete v0.21 representation is retained as a warm-start
/// backbone, while the explicit gate is new and zero-initialized.
#[derive(Clone)]
pub struct PeptideSpectrumStructuredEditor {
    base: PeptideSpectrumChemistryDiffusionModel,
    gate_head: Linear,
}

impl PeptideSpectrumStructuredEditor {
    pub fn new(config: FoundationDiffusionConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let base = PeptideSpectrumChemistryDiffusionModel::new(config.clone(), vb.clone())?;
        let gate_vb = vb.pp("structured_edit").pp("gate");
        let gate_weight =
            gate_vb.get_with_hints((2, config.model_dim), "weight", nn::Init::Const(0.0))?;
        let gate_bias = gate_vb.get_with_hints(2, "bias", nn::Init::Const(0.0))?;
        Ok(Self {
            base,
            gate_head: Linear::new(gate_weight, Some(gate_bias)),
        })
    }

    pub fn forward_t(
        &self,
        input: &FoundationDiffusionBatch,
        spectrum: &FoundationSpectrumBatch,
        precursor: &PrecursorContextBatch,
        chemistry: &ChemistryDiffusionFeatureBatch,
        train: bool,
    ) -> Result<FoundationStructuredEditOutput> {
        let sequence = self
            .base
            .forward_t(input, spectrum, precursor, chemistry, train)?;
        let gate_logits = self.gate_head.forward(&sequence.decoder_hidden)?;
        Ok(FoundationStructuredEditOutput {
            sequence,
            gate_logits,
        })
    }

    pub fn config(&self) -> &FoundationDiffusionConfig {
        self.base.config()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredEditWarmStartReport {
    pub shared_v0210_variables: usize,
    pub new_gate_variables: usize,
    pub ignored_v0210_variables: usize,
}

/// Warm-start every shared tensor from the selected v0.21 checkpoint. v0.21 is
/// scientifically closed as a denoising lane, but its learned bidirectional
/// spectrum/chemistry representation remains useful initialization.
pub fn load_structured_editor_from_v0210_checkpoint(
    varmap: &VarMap,
    path: &Path,
    device: &Device,
) -> Result<StructuredEditWarmStartReport> {
    let checkpoint = candle_core::safetensors::load(path, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("v0.22 VarMap lock poisoned".into()))?;
    let current_names: HashSet<String> = data.keys().cloned().collect();
    let mut shared = 0usize;
    let mut new_gate = 0usize;
    let mut missing = Vec::new();
    for (name, variable) in data.iter() {
        if let Some(source) = checkpoint.get(name) {
            if variable.as_tensor().dims() != source.dims() {
                candle_core::bail!(
                    "v0.22 warm-start shape mismatch for '{name}': current {:?}, v0.21 {:?}",
                    variable.as_tensor().dims(),
                    source.dims()
                );
            }
            variable.set(source)?;
            shared += 1;
        } else if name.starts_with("structured_edit.gate.") {
            new_gate += 1;
        } else {
            missing.push(name.clone());
        }
    }
    drop(data);
    if !missing.is_empty() {
        candle_core::bail!(
            "v0.21 checkpoint missing required v0.22 tensors: {}",
            missing.join(", ")
        );
    }
    let ignored = checkpoint
        .keys()
        .filter(|name| !current_names.contains(*name))
        .count();
    Ok(StructuredEditWarmStartReport {
        shared_v0210_variables: shared,
        new_gate_variables: new_gate,
        ignored_v0210_variables: ignored,
    })
}

/// Runtime TRAIN/VALIDATION isolation contract for the on-policy editor. TEST
/// is never an accepted label in either materialization path.
pub fn foundation_structured_edit_partition_isolated(
    train_labels: &[FoundationPartition],
    validation_labels: &[FoundationPartition],
) -> bool {
    !train_labels.is_empty()
        && !validation_labels.is_empty()
        && train_labels
            .iter()
            .all(|&label| label == FoundationPartition::Train)
        && validation_labels
            .iter()
            .all(|&label| label == FoundationPartition::Validation)
}

/// Convert a frozen v0.20 hypothesis into the full-width v0.22 edit state.
/// The existing EOS is replaced by MASK and all later positions are MASK so
/// target length may move in either direction. Existing peptide content before
/// EOS is preserved exactly.
pub fn foundation_structured_edit_open_row(
    initial: &[u32],
    initial_active: usize,
    max_tokens: usize,
) -> std::result::Result<Vec<u32>, String> {
    if max_tokens == 0
        || initial.len() != max_tokens
        || initial_active == 0
        || initial_active > max_tokens
    {
        return Err("v0.22 initial row/active length is invalid".into());
    }
    let eos = initial[..initial_active]
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
        .ok_or_else(|| "v0.22 v0.20 initializer does not contain EOS".to_string())?;
    let mut row = vec![FOUNDATION_DIFFUSION_MASK; max_tokens];
    row[..eos].copy_from_slice(&initial[..eos]);
    Ok(row)
}

/// Prepare the exact target tensors used with a full-width open edit state.
/// Attention remains enabled over the complete edit canvas, while token/gate
/// supervision stops at the true EOS. This allows target length to change
/// without teaching PAD as a peptide category.
pub fn foundation_structured_edit_set_targets(
    mut batch: FoundationDiffusionBatch,
    target_rows: &[Vec<u32>],
    target_active_lengths: &[usize],
) -> Result<FoundationDiffusionBatch> {
    let (b, width) = batch.noisy_tokens.dims2()?;
    if target_rows.len() != b || target_active_lengths.len() != b {
        candle_core::bail!("v0.22 target batch lengths differ");
    }
    let device = batch.noisy_tokens.device();
    let mut clean = vec![FOUNDATION_DIFFUSION_PAD; b * width];
    let mut indices = Vec::<u32>::new();
    let mut classes = Vec::<u32>::new();
    let mut lengths = Vec::<u32>::with_capacity(b);
    for row_index in 0..b {
        if target_rows[row_index].len() != width {
            candle_core::bail!("v0.22 target row width mismatch");
        }
        let active = target_active_lengths[row_index];
        if active == 0
            || active > width
            || target_rows[row_index][active - 1] != FOUNDATION_DIFFUSION_EOS
        {
            candle_core::bail!("v0.22 target row has invalid active length/EOS");
        }
        for position in 0..width {
            let token = target_rows[row_index][position];
            clean[row_index * width + position] = token;
            if position < active {
                if token == FOUNDATION_DIFFUSION_PAD || token == FOUNDATION_DIFFUSION_MASK {
                    candle_core::bail!("v0.22 active target contains PAD/MASK");
                }
                indices.push((row_index * width + position) as u32);
                classes.push(token);
            }
        }
        lengths.push((active - 1) as u32);
    }
    batch.clean_tokens = Tensor::from_vec(clean, (b, width), device)?.to_dtype(DType::U32)?;
    batch.active_indices =
        Tensor::from_vec(indices.clone(), indices.len(), device)?.to_dtype(DType::U32)?;
    batch.target_classes =
        Tensor::from_vec(classes.clone(), classes.len(), device)?.to_dtype(DType::U32)?;
    batch.length_targets = Tensor::from_vec(lengths, b, device)?.to_dtype(DType::U32)?;
    Ok(batch)
}

/// KEEP/CHANGE cross entropy over positions through the true EOS. Positions
/// after target EOS are governed by the target-length head and are intentionally
/// excluded from gate supervision.
pub fn foundation_structured_edit_gate_loss(
    output: &FoundationStructuredEditOutput,
    input_rows: &[Vec<u32>],
    target_rows: &[Vec<u32>],
    target_active_lengths: &[usize],
) -> Result<Tensor> {
    let (b, width, classes) = output.gate_logits.dims3()?;
    if classes != 2
        || input_rows.len() != b
        || target_rows.len() != b
        || target_active_lengths.len() != b
    {
        candle_core::bail!("v0.22 gate-loss shapes differ");
    }
    let device = output.gate_logits.device();
    let mut indices = Vec::<u32>::new();
    let mut targets = Vec::<u32>::new();
    for row_index in 0..b {
        if input_rows[row_index].len() != width || target_rows[row_index].len() != width {
            candle_core::bail!("v0.22 gate-loss row width mismatch");
        }
        let active = target_active_lengths[row_index];
        if active == 0 || active > width {
            candle_core::bail!("v0.22 gate-loss active length invalid");
        }
        for position in 0..active {
            indices.push((row_index * width + position) as u32);
            targets.push(
                if input_rows[row_index][position] != target_rows[row_index][position] {
                    1
                } else {
                    0
                },
            );
        }
    }
    let flat = output.gate_logits.reshape((b * width, 2))?;
    let index_tensor =
        Tensor::from_vec(indices.clone(), indices.len(), device)?.to_dtype(DType::U32)?;
    let target_tensor =
        Tensor::from_vec(targets.clone(), targets.len(), device)?.to_dtype(DType::U32)?;
    let selected = flat.index_select(&index_tensor, 0)?;
    candle_nn::loss::cross_entropy(&selected, &target_tensor)
}

/// Produce one complete structured edit proposal. The length head selects the
/// new EOS location. For each pre-EOS position, the gate either preserves a
/// valid current token or accepts the highest-scoring grammar-valid replacement.
pub fn foundation_structured_edit_argmax(
    input_row: &[u32],
    token_logits: &[Vec<f32>],
    gate_logits: &[Vec<f32>],
    length_logits: &[f32],
) -> std::result::Result<(Vec<u32>, usize, usize), String> {
    let width = input_row.len();
    if width < 2
        || token_logits.len() != width
        || gate_logits.len() != width
        || length_logits.len() != width
    {
        return Err("v0.22 argmax dimensions differ".into());
    }
    let predicted_active = 1 + argmax_finite(length_logits)?;
    let active = predicted_active.clamp(2, width);
    let mut row = vec![FOUNDATION_DIFFUSION_PAD; width];
    let mut changed = 0usize;
    for position in 0..active - 1 {
        if token_logits[position].len() != FOUNDATION_DIFFUSION_VOCAB_SIZE
            || gate_logits[position].len() != 2
        {
            return Err("v0.22 argmax class width mismatch".into());
        }
        let current = input_row[position];
        let keep = gate_logits[position][0];
        let change = gate_logits[position][1];
        let keep_current = keep >= change
            && structured_token_valid_after_prefix(current, &row[..position], position);
        let token = if keep_current {
            current
        } else {
            best_structured_token(&token_logits[position], &row[..position], position)?
        };
        changed += usize::from(token != current);
        row[position] = token;
    }
    row[active - 1] = FOUNDATION_DIFFUSION_EOS;
    changed += usize::from(input_row[active - 1] != FOUNDATION_DIFFUSION_EOS);
    Ok((row, active, changed))
}

/// Apply the final v0.22 physical contract. The learned edit itself is free to
/// be mass-inconsistent; exact precursor mass and grammar are enforced only at
/// acceptance. The same fixed radius-two projection used by v0.21 is retained
/// as a physical projection, not a trainable/search hyperparameter.
pub fn foundation_structured_edit_finalize(
    draft: &[u32],
    active: usize,
    token_logits: &[Vec<f32>],
    initial: &[u32],
    initial_active: usize,
    precursor_mass: f64,
    tolerance_da: f64,
) -> std::result::Result<(Vec<u32>, bool, bool), String> {
    if foundation_chemistry_diffusion_final_mass_valid(draft, active, precursor_mass, tolerance_da)
    {
        return Ok((draft.to_vec(), false, false));
    }
    if let Some(projected) = foundation_chemistry_diffusion_project_mass_valid(
        draft,
        active,
        token_logits,
        precursor_mass,
        tolerance_da,
    )? {
        return Ok((projected, true, false));
    }
    if foundation_chemistry_diffusion_final_mass_valid(
        initial,
        initial_active,
        precursor_mass,
        tolerance_da,
    ) {
        return Ok((initial.to_vec(), false, true));
    }
    Ok((draft.to_vec(), false, true))
}

fn best_structured_token(
    logits: &[f32],
    prefix: &[u32],
    position: usize,
) -> std::result::Result<u32, String> {
    let mut best = None::<(f32, u32)>;
    for token in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE as u32 {
        if !structured_token_valid_after_prefix(token, prefix, position) {
            continue;
        }
        let score = logits[token as usize];
        if !score.is_finite() {
            continue;
        }
        if best.map(|(old, _)| score > old).unwrap_or(true) {
            best = Some((score, token));
        }
    }
    best.map(|(_, token)| token)
        .ok_or_else(|| "v0.22 found no grammar-valid replacement token".to_string())
}

fn structured_token_valid_after_prefix(token: u32, prefix: &[u32], position: usize) -> bool {
    if foundation_diffusion_token_residue(token).is_some() {
        return true;
    }
    if token == FOUNDATION_DIFFUSION_NTERM_ACETYL {
        return position == 0;
    }
    if matches!(
        token,
        FOUNDATION_DIFFUSION_RESIDUE_ACETYL
            | FOUNDATION_DIFFUSION_CARBAMIDOMETHYL
            | FOUNDATION_DIFFUSION_DEAMIDATED
            | FOUNDATION_DIFFUSION_OXIDATION
            | FOUNDATION_DIFFUSION_PHOSPHO
    ) {
        let Some(&previous) = prefix.last() else {
            return false;
        };
        let Some(residue) = foundation_diffusion_token_residue(previous) else {
            return false;
        };
        return foundation_diffusion_residue_ptm_valid(token, residue);
    }
    false
}

fn argmax_finite(values: &[f32]) -> std::result::Result<usize, String> {
    let mut best = None::<(f32, usize)>;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if best.map(|(old, _)| value > old).unwrap_or(true) {
            best = Some((value, index));
        }
    }
    best.map(|(_, index)| index)
        .ok_or_else(|| "v0.22 argmax received no finite values".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::diffusion::{
        FOUNDATION_DIFFUSION_FIRST_RESIDUE, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    };

    #[test]
    fn open_row_preserves_content_and_removes_irreversible_eos() {
        let a = FOUNDATION_DIFFUSION_FIRST_RESIDUE;
        let g = FOUNDATION_DIFFUSION_FIRST_RESIDUE + 5;
        let initial = vec![
            a,
            g,
            FOUNDATION_DIFFUSION_EOS,
            FOUNDATION_DIFFUSION_PAD,
            FOUNDATION_DIFFUSION_PAD,
        ];
        let open = foundation_structured_edit_open_row(&initial, 3, 5).unwrap();
        assert_eq!(
            open,
            vec![
                a,
                g,
                FOUNDATION_DIFFUSION_MASK,
                FOUNDATION_DIFFUSION_MASK,
                FOUNDATION_DIFFUSION_MASK
            ]
        );
    }

    #[test]
    fn argmax_can_move_eos_and_preserve_kept_tokens() {
        let a = FOUNDATION_DIFFUSION_FIRST_RESIDUE;
        let g = FOUNDATION_DIFFUSION_FIRST_RESIDUE + 5;
        let input = vec![
            a,
            FOUNDATION_DIFFUSION_MASK,
            FOUNDATION_DIFFUSION_MASK,
            FOUNDATION_DIFFUSION_MASK,
        ];
        let mut token_logits = vec![vec![0.0; FOUNDATION_DIFFUSION_VOCAB_SIZE]; 4];
        token_logits[1][g as usize] = 5.0;
        token_logits[2][g as usize] = 4.0;
        let gate_logits = vec![
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![0.0, 2.0],
            vec![0.0, 2.0],
        ];
        let length_logits = vec![0.0, 0.0, 5.0, 0.0]; // active length 3
        let (row, active, _) =
            foundation_structured_edit_argmax(&input, &token_logits, &gate_logits, &length_logits)
                .unwrap();
        assert_eq!(active, 3);
        assert_eq!(row[0], a);
        assert_eq!(row[1], g);
        assert_eq!(row[2], FOUNDATION_DIFFUSION_EOS);
        assert_eq!(row[3], FOUNDATION_DIFFUSION_PAD);
    }

    #[test]
    fn partition_contract_rejects_test_or_mixed_labels() {
        assert!(foundation_structured_edit_partition_isolated(
            &[FoundationPartition::Train],
            &[FoundationPartition::Validation]
        ));
        assert!(!foundation_structured_edit_partition_isolated(
            &[FoundationPartition::Train, FoundationPartition::Test],
            &[FoundationPartition::Validation]
        ));
    }

    #[test]
    fn structural_token_filter_never_emits_pad_mask_or_internal_eos() {
        let a = FOUNDATION_DIFFUSION_FIRST_RESIDUE;
        assert!(structured_token_valid_after_prefix(a, &[], 0));
        assert!(!structured_token_valid_after_prefix(
            FOUNDATION_DIFFUSION_PAD,
            &[],
            0
        ));
        assert!(!structured_token_valid_after_prefix(
            FOUNDATION_DIFFUSION_MASK,
            &[],
            0
        ));
        assert!(!structured_token_valid_after_prefix(
            FOUNDATION_DIFFUSION_EOS,
            &[],
            0
        ));
    }
}
