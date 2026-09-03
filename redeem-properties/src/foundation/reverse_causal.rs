//! Isolated C-terminal-to-N-terminal causal proposal support.
//!
//! Reverse-causal targets reverse residue units rather than raw tokens so a
//! residue-local PTM marker remains attached to the residue it modifies. The
//! global N-terminal acetyl marker stays in a dedicated leading slot. Applying
//! the transformation twice returns the original canonical N->C token row.

use super::diffusion::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_residue,
    FoundationDiffusionConfig, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_MASK,
    FOUNDATION_DIFFUSION_NTERM_ACETYL, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_VOCAB_SIZE,
};
use candle_core::{Device, Result};
use candle_nn::VarMap;
use std::collections::HashSet;
use std::path::Path;

/// Parameter namespace used by the isolated v0.13.13 reverse-causal branch.
pub const FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313: &str = "reverse_causal";

/// Stable identifier for the C->N residue-unit target representation.
pub const FOUNDATION_REVERSE_CAUSAL_DIRECTION_V01313: &str = "c_to_n_residue_units_v01313";

/// Warm-start accounting for an isolated reverse-causal model copied from an
/// accepted unified N->C causal checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoundationReverseCausalWarmStartReport {
    /// Reverse-causal variables initialized from matching parent causal names.
    pub loaded_variables: usize,
    /// Parent tensors not represented by the isolated reverse-causal model.
    pub ignored_parent_variables: usize,
}

/// Reverse one clean peptide token row by residue units while retaining EOS and
/// padding in their canonical positions.
///
/// A residue-local PTM token follows its residue in both directions. For
/// example, canonical `[A, M, Ox, K, EOS]` becomes
/// `[K, M, Ox, A, EOS]`, not `[K, Ox, M, A, EOS]`.
/// N-terminal acetyl remains the first global marker:
/// `[N-acetyl, A, M, EOS] -> [N-acetyl, M, A, EOS]`.
///
/// This transformation is an involution and is therefore also used to convert
/// reverse-generated rows back to canonical N->C form.
pub fn foundation_reverse_causal_token_row(
    tokens: &[u32],
) -> std::result::Result<Vec<u32>, String> {
    if tokens.is_empty() {
        return Err("reverse-causal token row cannot be empty".into());
    }
    let width = tokens.len();
    let eos_position = tokens
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_EOS)
        .ok_or_else(|| "reverse-causal token row does not contain EOS".to_string())?;
    if eos_position == 0 {
        return Err("reverse-causal token row contains no peptide tokens before EOS".into());
    }
    if tokens[..eos_position]
        .iter()
        .any(|&token| token == FOUNDATION_DIFFUSION_PAD || token == FOUNDATION_DIFFUSION_MASK)
    {
        return Err("reverse-causal active token prefix contains PAD/MASK".into());
    }
    if tokens[eos_position + 1..]
        .iter()
        .any(|&token| token != FOUNDATION_DIFFUSION_PAD)
    {
        return Err("reverse-causal token suffix after EOS must contain only PAD".into());
    }

    let active = &tokens[..eos_position];
    let has_nterm_acetyl = active.first().copied() == Some(FOUNDATION_DIFFUSION_NTERM_ACETYL);
    let mut position = usize::from(has_nterm_acetyl);
    let mut units = Vec::<Vec<u32>>::new();

    while position < active.len() {
        let residue_token = active[position];
        let residue = foundation_diffusion_token_residue(residue_token).ok_or_else(|| {
            format!(
                "reverse-causal residue unit starts with non-residue token {residue_token} at position {position}"
            )
        })?;
        let mut unit = vec![residue_token];
        position += 1;
        while position < active.len()
            && foundation_diffusion_token_residue(active[position]).is_none()
        {
            let modification = active[position];
            if modification == FOUNDATION_DIFFUSION_NTERM_ACETYL
                || modification == FOUNDATION_DIFFUSION_EOS
                || modification == FOUNDATION_DIFFUSION_PAD
                || modification == FOUNDATION_DIFFUSION_MASK
                || modification as usize >= FOUNDATION_DIFFUSION_VOCAB_SIZE
            {
                return Err(format!(
                    "reverse-causal residue unit contains invalid token {modification} at position {position}"
                ));
            }
            if !foundation_diffusion_residue_ptm_valid(modification, residue) {
                return Err(format!(
                    "reverse-causal PTM token {modification} is invalid on residue {residue}"
                ));
            }
            unit.push(modification);
            position += 1;
        }
        units.push(unit);
    }

    if units.is_empty() {
        return Err("reverse-causal token row contains no residue units".into());
    }

    let mut reversed = Vec::<u32>::with_capacity(width);
    if has_nterm_acetyl {
        reversed.push(FOUNDATION_DIFFUSION_NTERM_ACETYL);
    }
    for unit in units.iter().rev() {
        reversed.extend_from_slice(unit);
    }
    reversed.push(FOUNDATION_DIFFUSION_EOS);
    reversed.resize(width, FOUNDATION_DIFFUSION_PAD);
    Ok(reversed)
}

/// Convert a reverse C->N token row back to canonical N->C order.
///
/// The residue-unit reversal is an involution, so canonicalization is exactly
/// the same validated transformation used to construct training targets.
pub fn foundation_canonicalize_reverse_causal_token_row(
    tokens: &[u32],
) -> std::result::Result<Vec<u32>, String> {
    foundation_reverse_causal_token_row(tokens)
}

/// Warm-start a model instantiated under `reverse_causal.*` from the matching
/// causal variables in an accepted unified checkpoint.
///
/// No parent variable is mutated. The isolated reverse model lives in its own
/// [`VarMap`], preventing reverse training from changing diffusion or N->C
/// causal proposal parameters.
pub fn load_reverse_causal_from_unified_checkpoint(
    reverse_varmap: &VarMap,
    parent_checkpoint: &Path,
    device: &Device,
) -> Result<FoundationReverseCausalWarmStartReport> {
    let parent = candle_core::safetensors::load(parent_checkpoint, device)?;
    let data = reverse_varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("reverse-causal VarMap lock poisoned".into()))?;
    let prefix = format!("{FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313}.");
    let mut loaded_variables = 0usize;
    let mut required_missing = Vec::<String>::new();
    let mut consumed_parent = HashSet::<String>::new();

    for (reverse_name, variable) in data.iter() {
        let parent_name = reverse_name.strip_prefix(&prefix).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "reverse-causal model contains variable outside namespace: {reverse_name}"
            ))
        })?;
        let Some(tensor) = parent.get(parent_name) else {
            required_missing.push(parent_name.to_string());
            continue;
        };
        if variable.as_tensor().dims() != tensor.dims() {
            candle_core::bail!(
                "reverse-causal warm-start shape mismatch for '{reverse_name}' from parent '{parent_name}': reverse {:?}, parent {:?}",
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
            "unified parent is missing required reverse-causal warm-start variables: {}",
            required_missing.join(", ")
        );
    }

    let ignored_parent_variables = parent
        .keys()
        .filter(|name| !consumed_parent.contains(*name))
        .count();
    Ok(FoundationReverseCausalWarmStartReport {
        loaded_variables,
        ignored_parent_variables,
    })
}

/// Confirm that a reverse-causal checkpoint uses the expected isolated
/// namespace and model dimensionality.
pub fn validate_reverse_causal_namespace(
    varmap: &VarMap,
    config: &FoundationDiffusionConfig,
) -> Result<()> {
    config.validate().map_err(candle_core::Error::Msg)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("reverse-causal VarMap lock poisoned".into()))?;
    let prefix = format!("{FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313}.");
    if data.is_empty() {
        candle_core::bail!("reverse-causal VarMap contains no variables");
    }
    if let Some(name) = data.keys().find(|name| !name.starts_with(&prefix)) {
        candle_core::bail!(
            "reverse-causal VarMap variable '{name}' is outside expected namespace '{prefix}'"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        FoundationDiffusionVocabulary, FoundationModification, FoundationModificationSite,
        PeptidoformInput, FOUNDATION_DIFFUSION_OXIDATION,
    };

    #[test]
    fn reverse_residue_units_is_an_involution() {
        let vocabulary = FoundationDiffusionVocabulary;
        let peptide = PeptidoformInput {
            sequence: "AMK".into(),
            modifications: vec![FoundationModification::unimod(
                FoundationModificationSite::Residue(1),
                1,
                35,
                15.994_915,
            )],
        };
        let canonical = vocabulary.encode(&peptide, 16).unwrap();
        let reverse = foundation_reverse_causal_token_row(&canonical).unwrap();
        assert_eq!(reverse[2], FOUNDATION_DIFFUSION_OXIDATION);
        let restored = foundation_canonicalize_reverse_causal_token_row(&reverse).unwrap();
        assert_eq!(restored, canonical);
        assert_eq!(vocabulary.decode(&restored).unwrap(), peptide);
    }

    #[test]
    fn reverse_keeps_nterm_marker_global() {
        let vocabulary = FoundationDiffusionVocabulary;
        let peptide = PeptidoformInput {
            sequence: "AK".into(),
            modifications: vec![FoundationModification::unimod(
                FoundationModificationSite::NTerm,
                0,
                1,
                42.010_565,
            )],
        };
        let canonical = vocabulary.encode(&peptide, 12).unwrap();
        let reverse = foundation_reverse_causal_token_row(&canonical).unwrap();
        assert_eq!(reverse[0], FOUNDATION_DIFFUSION_NTERM_ACETYL);
        let restored = foundation_canonicalize_reverse_causal_token_row(&reverse).unwrap();
        assert_eq!(restored, canonical);
    }
}
