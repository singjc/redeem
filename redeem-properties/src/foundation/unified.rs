//! Unified forward-property and spectrum-to-peptide foundation model container.
//!
//! The forward peptide encoder/heads and the inverse spectrum decoder intentionally
//! retain their historical parameter namespaces. Diffusion and causal modes are
//! instantiated against the same [`VarMap`], so their shared inverse tensors are
//! represented by one set of variables while mode-specific parameters remain
//! separate.

use super::causal::PeptideSpectrumCausalModel;
use super::config::FoundationConfig;
use super::diffusion::{FoundationDiffusionConfig, PeptideSpectrumDiffusionModel};
use super::model::PeptideFoundationMultiTaskModel;
use candle_core::{Device, Result, Tensor};
use candle_nn::{self as nn, Linear, Module, VarBuilder, VarMap};
use std::collections::HashSet;
use std::path::Path;

/// One jointly checkpointable peptide foundation model.
#[derive(Clone)]
pub struct PeptideFoundationUnifiedModel {
    forward: PeptideFoundationMultiTaskModel,
    diffusion: PeptideSpectrumDiffusionModel,
    causal: PeptideSpectrumCausalModel,
    spectrum_projection: Linear,
    forward_config: FoundationConfig,
    inverse_config: FoundationDiffusionConfig,
}

impl PeptideFoundationUnifiedModel {
    /// Construct all forward and inverse branches in one variable namespace.
    ///
    /// Diffusion is instantiated before causal mode. Their checkpoint-compatible
    /// shared names therefore resolve to the same [`VarMap`] variables. The only
    /// additional causal parameter is the learned START embedding.
    pub fn new(
        forward_config: FoundationConfig,
        inverse_config: FoundationDiffusionConfig,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        forward_config.validate().map_err(candle_core::Error::Msg)?;
        inverse_config.validate().map_err(candle_core::Error::Msg)?;
        let forward = PeptideFoundationMultiTaskModel::new(forward_config.clone(), vb.clone())?;
        let diffusion = PeptideSpectrumDiffusionModel::new(inverse_config.clone(), vb.clone())?;
        let causal = PeptideSpectrumCausalModel::new(inverse_config.clone(), vb.clone())?;
        let spectrum_projection = nn::linear(
            inverse_config.model_dim,
            forward_config.contrastive_dim,
            vb.pp("alignment.spectrum_projection"),
        )?;
        Ok(Self {
            forward,
            diffusion,
            causal,
            spectrum_projection,
            forward_config,
            inverse_config,
        })
    }

    /// Forward peptide/property branch.
    pub fn forward(&self) -> &PeptideFoundationMultiTaskModel {
        &self.forward
    }

    /// Spectrum-conditioned diffusion branch.
    pub fn diffusion(&self) -> &PeptideSpectrumDiffusionModel {
        &self.diffusion
    }

    /// Spectrum-conditioned causal next-token branch.
    pub fn causal(&self) -> &PeptideSpectrumCausalModel {
        &self.causal
    }

    /// Project a pooled inverse spectrum embedding into the peptide contrastive space.
    pub fn project_spectrum_embedding(&self, embedding: &Tensor) -> Result<Tensor> {
        self.spectrum_projection.forward(embedding)
    }

    /// Forward peptide architecture.
    pub fn forward_config(&self) -> &FoundationConfig {
        &self.forward_config
    }

    /// Shared inverse architecture.
    pub fn inverse_config(&self) -> &FoundationDiffusionConfig {
        &self.inverse_config
    }
}

/// Warm-start accounting for a consolidated forward + inverse checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoundationUnifiedWarmStartReport {
    /// Forward encoder/head/context tensors loaded from the corrected forward checkpoint.
    pub forward_loaded_variables: usize,
    /// Shared + diffusion-only inverse tensors loaded from the diffusion checkpoint.
    pub diffusion_loaded_variables: usize,
    /// Shared inverse + causal START tensors overlaid from the causal checkpoint.
    pub causal_overlay_loaded_variables: usize,
    /// Fresh alignment-projection tensors intentionally left at initialization.
    pub fresh_alignment_variables: usize,
}

/// Merge the validated component checkpoints into one unified [`VarMap`].
///
/// Loading order is deliberate:
/// 1. corrected forward checkpoint -> `encoder.*`, `heads.*`, `context.*`;
/// 2. diffusion checkpoint -> shared inverse backbone + diffusion timestep/length head;
/// 3. causal checkpoint -> shared inverse backbone (overwriting diffusion shared values)
///    + causal START embedding;
/// 4. `alignment.spectrum_projection.*` remains freshly initialized.
///
/// This preserves the causal-trained shared inverse backbone while retaining the
/// diffusion-only parameters required to continue the denoising objective.
pub fn load_unified_foundation_components(
    varmap: &VarMap,
    forward_checkpoint: &Path,
    diffusion_checkpoint: &Path,
    causal_checkpoint: &Path,
    device: &Device,
) -> Result<FoundationUnifiedWarmStartReport> {
    validate_unified_namespace(varmap)?;

    let forward = candle_core::safetensors::load(forward_checkpoint, device)?;
    let diffusion = candle_core::safetensors::load(diffusion_checkpoint, device)?;
    let causal = candle_core::safetensors::load(causal_checkpoint, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("unified foundation VarMap lock poisoned".into()))?;

    let mut forward_loaded = 0usize;
    let mut diffusion_loaded = 0usize;
    let mut causal_loaded = 0usize;
    let mut fresh_alignment = 0usize;
    let mut missing = Vec::<String>::new();

    for (name, variable) in data.iter() {
        if is_forward_name(name) {
            if let Some(tensor) = forward.get(name) {
                set_checked(variable, tensor, name, "forward")?;
                forward_loaded += 1;
            } else {
                missing.push(format!("forward:{name}"));
            }
            continue;
        }

        if is_inverse_common_name(name) || is_diffusion_only_name(name) {
            if let Some(tensor) = diffusion.get(name) {
                set_checked(variable, tensor, name, "diffusion")?;
                diffusion_loaded += 1;
            } else {
                missing.push(format!("diffusion:{name}"));
                continue;
            }
        }

        if is_inverse_common_name(name) || is_causal_only_name(name) {
            if let Some(tensor) = causal.get(name) {
                set_checked(variable, tensor, name, "causal")?;
                causal_loaded += 1;
            } else {
                missing.push(format!("causal:{name}"));
            }
            continue;
        }

        if is_alignment_name(name) {
            fresh_alignment += 1;
        }
    }
    drop(data);

    if !missing.is_empty() {
        candle_core::bail!(
            "unified foundation warm start is missing required component variables: {}",
            missing.join(", ")
        );
    }

    Ok(FoundationUnifiedWarmStartReport {
        forward_loaded_variables: forward_loaded,
        diffusion_loaded_variables: diffusion_loaded,
        causal_overlay_loaded_variables: causal_loaded,
        fresh_alignment_variables: fresh_alignment,
    })
}

fn set_checked(
    variable: &candle_core::Var,
    tensor: &Tensor,
    name: &str,
    source: &str,
) -> Result<()> {
    if variable.as_tensor().dims() != tensor.dims() {
        candle_core::bail!(
            "unified {source} warm-start shape mismatch for '{name}': current {:?}, checkpoint {:?}",
            variable.as_tensor().dims(),
            tensor.dims()
        );
    }
    variable.set(tensor)
}

fn validate_unified_namespace(varmap: &VarMap) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("unified foundation VarMap lock poisoned".into()))?;
    let mut unexpected = Vec::<String>::new();
    let mut names = HashSet::<String>::new();
    for name in data.keys() {
        if !names.insert(name.clone()) {
            candle_core::bail!("duplicate unified foundation variable name '{name}'");
        }
        if !(is_forward_name(name)
            || is_inverse_common_name(name)
            || is_diffusion_only_name(name)
            || is_causal_only_name(name)
            || is_alignment_name(name))
        {
            unexpected.push(name.clone());
        }
    }
    drop(data);
    if !unexpected.is_empty() {
        candle_core::bail!(
            "unified foundation model instantiated unexpected variable namespaces: {}",
            unexpected.join(", ")
        );
    }
    Ok(())
}

fn is_forward_name(name: &str) -> bool {
    name.starts_with("encoder.") || name.starts_with("heads.") || name.starts_with("context.")
}

fn is_inverse_common_name(name: &str) -> bool {
    name.starts_with("spectrum_encoder.")
        || name.starts_with("decoder.token_embedding.")
        || name.starts_with("decoder.position_embedding.")
        || name.starts_with("decoder.precursor.")
        || name.starts_with("decoder.layers.")
        || name.starts_with("decoder.output_norm.")
        || name.starts_with("decoder.token_head.")
}

fn is_diffusion_only_name(name: &str) -> bool {
    name.starts_with("decoder.timestep.") || name.starts_with("decoder.length_head.")
}

fn is_causal_only_name(name: &str) -> bool {
    name.starts_with("decoder.causal_start_embedding.")
}

fn is_alignment_name(name: &str) -> bool {
    name.starts_with("alignment.spectrum_projection.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::spectrum::FoundationSpectrumConfig;
    use candle_core::{DType, Device};

    fn forward_config() -> FoundationConfig {
        FoundationConfig {
            model_dim: 32,
            graph_hidden_dim: 16,
            graph_layers: 1,
            transformer_layers: 1,
            transformer_ff_dim: 64,
            num_attention_heads: 4,
            contrastive_dim: 16,
            ..FoundationConfig::default()
        }
    }

    fn inverse_config() -> FoundationDiffusionConfig {
        FoundationDiffusionConfig {
            max_tokens: 16,
            model_dim: 32,
            num_attention_heads: 4,
            feed_forward_dim: 64,
            spectrum_layers: 1,
            decoder_layers: 1,
            spectrum: FoundationSpectrumConfig {
                max_peaks: 8,
                ..FoundationSpectrumConfig::default()
            },
            ..FoundationDiffusionConfig::default()
        }
    }

    #[test]
    fn unified_model_uses_one_clean_parameter_namespace() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = PeptideFoundationUnifiedModel::new(forward_config(), inverse_config(), vb)
            .expect("construct unified model");
        validate_unified_namespace(&varmap).expect("valid unified namespace");
        assert_eq!(model.forward_config().model_dim, 32);
        assert_eq!(model.inverse_config().model_dim, 32);

        let data = varmap.data().lock().unwrap();
        assert!(data.keys().any(|name| name.starts_with("encoder.")));
        assert!(data.keys().any(|name| name.starts_with("heads.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("spectrum_encoder.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("decoder.timestep.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("decoder.causal_start_embedding.")));
        assert!(data
            .keys()
            .any(|name| name.starts_with("alignment.spectrum_projection.")));
    }
}
