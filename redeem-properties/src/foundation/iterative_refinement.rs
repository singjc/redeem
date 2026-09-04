//! Isolated spectrum-conditioned iterative masked refinement support.
//!
//! v0.13.15 reuses the bidirectional diffusion decoder architecture in a dedicated
//! `iterative_refinement.*` namespace. Training masks a fixed quarter of residue
//! positions while retaining the other residues, EOS, and PTM marker structure.
//! Only masked residue positions contribute to the reconstruction loss. This makes
//! the branch suitable for revising internal sequence errors without committing to
//! either N->C or C->N autoregressive order.

use super::diffusion::{
    foundation_diffusion_token_residue, FoundationDiffusionBatch, FoundationDiffusionCollator,
    FoundationDiffusionConfig, FoundationDiffusionVocabulary, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_PAD,
};
use super::featurize::PeptidoformInput;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarMap;
use std::collections::HashSet;
use std::path::Path;

/// Parameter namespace used by the isolated v0.13.15 refinement branch.
pub const FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315: &str = "iterative_refinement";
/// Stable training objective identifier.
pub const FOUNDATION_ITERATIVE_REFINEMENT_OBJECTIVE_V01315: &str =
    "iterative_masked_residue_refinement_ce_v01315";
/// Fraction of residue positions masked per refinement-training example.
pub const FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315: f64 = 0.25;
/// Fixed number of inference refinement rounds in the first architectural pilot.
pub const FOUNDATION_ITERATIVE_REFINEMENT_ROUNDS_V01315: usize = 4;
/// Fixed number of branch-balanced starting hypotheses per validation spectrum.
pub const FOUNDATION_ITERATIVE_REFINEMENT_SEED_HYPOTHESES_V01315: usize = 12;
/// Per-position residue alternatives retained during joint mass-constrained refill.
pub const FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_TOPK_V01315: usize = 8;
/// Beam width used only inside the joint refill of one masked residue subset.
pub const FOUNDATION_ITERATIVE_REFINEMENT_REPLACEMENT_BEAM_V01315: usize = 64;

/// Warm-start accounting for an isolated refinement model copied from the
/// accepted unified diffusion/masked-token checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoundationIterativeRefinementWarmStartReport {
    /// Refinement variables initialized from matching parent diffusion names.
    pub loaded_variables: usize,
    /// Parent tensors not represented by the isolated refinement model.
    pub ignored_parent_variables: usize,
}

/// Select the deterministic residue positions to mask for one token row.
///
/// PTM markers, EOS, N-terminal markers, MASK and PAD are never selected. The
/// number of masked residues is `ceil(residue_count * mask_fraction)`, with at
/// least one residue selected when the row contains any residues.
pub fn foundation_iterative_refinement_mask_positions(
    tokens: &[u32],
    mask_fraction: f64,
    seed: u64,
) -> std::result::Result<Vec<usize>, String> {
    if !(mask_fraction > 0.0 && mask_fraction <= 1.0 && mask_fraction.is_finite()) {
        return Err("iterative-refinement mask fraction must be in (0, 1]".into());
    }
    let mut ranked = tokens
        .iter()
        .enumerate()
        .filter_map(|(position, &token)| {
            foundation_diffusion_token_residue(token).map(|_| {
                let key = mix64(
                    seed ^ (position as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ (token as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9),
                );
                (key, position)
            })
        })
        .collect::<Vec<_>>();
    if ranked.is_empty() {
        return Err("iterative-refinement token row contains no residue positions".into());
    }
    ranked.sort_unstable();
    let count = ((ranked.len() as f64 * mask_fraction).ceil() as usize).clamp(1, ranked.len());
    let mut selected = ranked
        .into_iter()
        .take(count)
        .map(|(_, position)| position)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    Ok(selected)
}

/// Build one partial-mask reconstruction batch.
///
/// The input retains all unselected clean tokens. Selected residue positions are
/// replaced with MASK and are the only positions contributing to x0 cross-entropy.
pub fn foundation_iterative_refinement_collate(
    collator: &FoundationDiffusionCollator,
    vocabulary: FoundationDiffusionVocabulary,
    config: &FoundationDiffusionConfig,
    peptides: &[PeptidoformInput],
    mask_fraction: f64,
    seed: u64,
    device: &Device,
) -> Result<FoundationDiffusionBatch> {
    if peptides.is_empty() {
        candle_core::bail!("iterative-refinement collation requires at least one peptide");
    }
    let width = config.max_tokens;
    let mut clean_rows = Vec::<Vec<u32>>::with_capacity(peptides.len());
    let mut noisy_rows = Vec::<Vec<u32>>::with_capacity(peptides.len());
    let mut active_lengths = Vec::<usize>::with_capacity(peptides.len());
    let mut selected_flat_indices = Vec::<u32>::new();
    let mut selected_targets = Vec::<u32>::new();

    for (row_index, peptide) in peptides.iter().enumerate() {
        let clean = vocabulary
            .encode(peptide, width)
            .map_err(candle_core::Error::Msg)?;
        let active_length = clean
            .iter()
            .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
            .unwrap_or(width);
        let positions = foundation_iterative_refinement_mask_positions(
            &clean[..active_length],
            mask_fraction,
            mix64(seed ^ row_index as u64),
        )
        .map_err(candle_core::Error::Msg)?;
        let mut noisy = clean.clone();
        for &position in &positions {
            selected_flat_indices.push((row_index * width + position) as u32);
            selected_targets.push(clean[position]);
            noisy[position] = FOUNDATION_DIFFUSION_MASK;
        }
        clean_rows.push(clean);
        noisy_rows.push(noisy);
        active_lengths.push(active_length);
    }

    let mut batch = collator.collate_inference_tokens(
        &noisy_rows,
        &active_lengths,
        config.diffusion_steps,
        device,
    )?;
    let flat_clean = clean_rows.into_iter().flatten().collect::<Vec<_>>();
    batch.clean_tokens =
        Tensor::from_vec(flat_clean, (peptides.len(), width), device)?.to_dtype(DType::U32)?;
    batch.active_indices = Tensor::from_vec(
        selected_flat_indices.clone(),
        selected_flat_indices.len(),
        device,
    )?
    .to_dtype(DType::U32)?;
    batch.target_classes =
        Tensor::from_vec(selected_targets.clone(), selected_targets.len(), device)?
            .to_dtype(DType::U32)?;
    Ok(batch)
}

/// Warm-start a model instantiated under `iterative_refinement.*` from matching
/// diffusion variables in the accepted unified checkpoint.
///
/// The parent checkpoint is read-only and the isolated model owns a separate
/// [`VarMap`], so refinement training cannot perturb accepted branches.
pub fn load_iterative_refinement_from_unified_checkpoint(
    refinement_varmap: &VarMap,
    parent_checkpoint: &Path,
    device: &Device,
) -> Result<FoundationIterativeRefinementWarmStartReport> {
    let parent = candle_core::safetensors::load(parent_checkpoint, device)?;
    let data = refinement_varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("iterative-refinement VarMap lock poisoned".into()))?;
    let prefix = format!("{FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315}.");
    let mut loaded_variables = 0usize;
    let mut required_missing = Vec::<String>::new();
    let mut consumed_parent = HashSet::<String>::new();

    for (refinement_name, variable) in data.iter() {
        let parent_name = refinement_name.strip_prefix(&prefix).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "iterative-refinement model contains variable outside namespace: {refinement_name}"
            ))
        })?;
        let Some(tensor) = parent.get(parent_name) else {
            required_missing.push(parent_name.to_string());
            continue;
        };
        if variable.as_tensor().dims() != tensor.dims() {
            candle_core::bail!(
                "iterative-refinement warm-start shape mismatch for '{refinement_name}' from parent '{parent_name}': refinement {:?}, parent {:?}",
                variable.as_tensor().dims(),
                tensor.dims()
            );
        }
        variable.set(tensor)?;
        loaded_variables += 1;
        consumed_parent.insert(parent_name.to_string());
    }
    drop(data);

    if !required_missing.is_empty() {
        candle_core::bail!(
            "unified parent is missing required iterative-refinement warm-start variables: {}",
            required_missing.join(", ")
        );
    }

    let ignored_parent_variables = parent
        .keys()
        .filter(|name| !consumed_parent.contains(*name))
        .count();
    Ok(FoundationIterativeRefinementWarmStartReport {
        loaded_variables,
        ignored_parent_variables,
    })
}

/// Confirm that a refinement checkpoint uses only the expected isolated namespace.
pub fn validate_iterative_refinement_namespace(
    varmap: &VarMap,
    config: &FoundationDiffusionConfig,
) -> Result<()> {
    config.validate().map_err(candle_core::Error::Msg)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("iterative-refinement VarMap lock poisoned".into()))?;
    let prefix = format!("{FOUNDATION_ITERATIVE_REFINEMENT_NAMESPACE_V01315}.");
    if data.is_empty() {
        candle_core::bail!("iterative-refinement VarMap contains no variables");
    }
    if let Some(name) = data.keys().find(|name| !name.starts_with(&prefix)) {
        candle_core::bail!(
            "iterative-refinement VarMap variable '{name}' is outside expected namespace '{prefix}'"
        );
    }
    Ok(())
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationModification, FoundationModificationSite, FOUNDATION_DIFFUSION_EOS,
        FOUNDATION_DIFFUSION_OXIDATION,
    };

    #[test]
    fn mask_selection_only_targets_residues() {
        let vocabulary = FoundationDiffusionVocabulary;
        let peptide = PeptidoformInput {
            sequence: "AMKQ".into(),
            modifications: vec![FoundationModification::unimod(
                FoundationModificationSite::Residue(1),
                1,
                35,
                15.994_915,
            )],
        };
        let tokens = vocabulary.encode(&peptide, 16).unwrap();
        let positions = foundation_iterative_refinement_mask_positions(
            &tokens,
            FOUNDATION_ITERATIVE_REFINEMENT_MASK_FRACTION_V01315,
            17,
        )
        .unwrap();
        assert_eq!(positions.len(), 1);
        for position in positions {
            assert!(foundation_diffusion_token_residue(tokens[position]).is_some());
            assert_ne!(tokens[position], FOUNDATION_DIFFUSION_OXIDATION);
            assert_ne!(tokens[position], FOUNDATION_DIFFUSION_EOS);
        }
    }

    #[test]
    fn mask_selection_is_deterministic() {
        let vocabulary = FoundationDiffusionVocabulary;
        let peptide = PeptidoformInput {
            sequence: "PEPTIDER".into(),
            modifications: vec![],
        };
        let tokens = vocabulary.encode(&peptide, 16).unwrap();
        let left = foundation_iterative_refinement_mask_positions(&tokens, 0.25, 99).unwrap();
        let right = foundation_iterative_refinement_mask_positions(&tokens, 0.25, 99).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.len(), 2);
    }
}
