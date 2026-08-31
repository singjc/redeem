//! Serializable AdamW optimizer for foundation-model training.
//!
//! Candle's built-in `AdamW` intentionally keeps its moment tensors and step
//! counter private. That is fine for ordinary training, but it prevents an
//! exact optimizer-state resume. This module mirrors Candle's AdamW equations
//! while retaining stable model-variable names so first/second moments can be
//! checkpointed in SafeTensors alongside human-readable trainer metadata.

use candle_core::{backprop::GradStore, safetensors, Result, Tensor, Var};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// AdamW hyperparameters used by [`FoundationAdamW`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoundationAdamWConfig {
    /// Learning rate. A scheduler may update this value between steps.
    pub learning_rate: f64,
    /// Exponential decay for the first moment.
    pub beta1: f64,
    /// Exponential decay for the second moment.
    pub beta2: f64,
    /// Numerical stabilizer added after the square root.
    pub epsilon: f64,
    /// Decoupled AdamW weight decay.
    pub weight_decay: f64,
}

impl Default for FoundationAdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl FoundationAdamWConfig {
    pub(crate) fn validate(self) -> Result<Self> {
        if !(self.learning_rate > 0.0 && self.learning_rate.is_finite()) {
            candle_core::bail!("foundation AdamW learning_rate must be positive and finite");
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            candle_core::bail!("foundation AdamW beta1/beta2 must be in [0, 1)");
        }
        if !(self.epsilon > 0.0 && self.epsilon.is_finite()) {
            candle_core::bail!("foundation AdamW epsilon must be positive and finite");
        }
        if !(self.weight_decay >= 0.0 && self.weight_decay.is_finite()) {
            candle_core::bail!("foundation AdamW weight_decay must be non-negative and finite");
        }
        Ok(self)
    }
}

#[derive(Debug)]
struct FoundationAdamWVariable {
    name: String,
    variable: Var,
    first_moment: Var,
    second_moment: Var,
}

/// Diagnostics returned by one optimizer step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoundationOptimizerStep {
    /// Global L2 gradient norm before clipping.
    pub gradient_norm: f64,
    /// Multiplicative gradient scale applied by clipping (`1` means no clip).
    pub gradient_scale: f64,
    /// Learning rate used for this update.
    pub learning_rate: f64,
    /// 1-based Adam update index after this step.
    pub step: u64,
}

/// AdamW optimizer with named, serializable moment state.
#[derive(Debug)]
pub struct FoundationAdamW {
    variables: Vec<FoundationAdamWVariable>,
    step: u64,
    config: FoundationAdamWConfig,
}

impl FoundationAdamW {
    /// Create zero-initialized AdamW moment state for all floating-point model variables.
    pub fn new(varmap: &VarMap, config: FoundationAdamWConfig) -> Result<Self> {
        let config = config.validate()?;
        let data = varmap
            .data()
            .lock()
            .map_err(|_| candle_core::Error::Msg("foundation VarMap lock poisoned".to_string()))?;
        let mut named: Vec<(String, Var)> = data
            .iter()
            .filter(|(_, variable)| variable.dtype().is_float())
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect();
        drop(data);
        named.sort_by(|left, right| left.0.cmp(&right.0));

        let mut variables = Vec::with_capacity(named.len());
        for (name, variable) in named {
            let first_moment = Var::zeros(variable.shape(), variable.dtype(), variable.device())?;
            let second_moment = Var::zeros(variable.shape(), variable.dtype(), variable.device())?;
            variables.push(FoundationAdamWVariable {
                name,
                variable,
                first_moment,
                second_moment,
            });
        }
        Ok(Self {
            variables,
            step: 0,
            config,
        })
    }

    /// Backpropagate `loss` and execute one AdamW update.
    pub fn backward_step(
        &mut self,
        loss: &Tensor,
        max_gradient_norm: Option<f64>,
    ) -> Result<FoundationOptimizerStep> {
        let gradients = loss.backward()?;
        self.step(&gradients, max_gradient_norm)
    }

    /// Execute one update from a precomputed Candle gradient store.
    pub fn step(
        &mut self,
        gradients: &GradStore,
        max_gradient_norm: Option<f64>,
    ) -> Result<FoundationOptimizerStep> {
        if let Some(max_norm) = max_gradient_norm {
            if !(max_norm > 0.0 && max_norm.is_finite()) {
                candle_core::bail!("foundation max_gradient_norm must be positive and finite");
            }
        }

        let gradient_norm = self.gradient_norm(gradients)?;
        let gradient_scale = match max_gradient_norm {
            Some(max_norm) if gradient_norm > max_norm && gradient_norm > 0.0 => {
                max_norm / gradient_norm
            }
            _ => 1.0,
        };

        self.step = self.step.wrapping_add(1);
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let learning_rate = self.config.learning_rate;
        let decay = learning_rate * self.config.weight_decay;
        let step_i32 = self.step.min(i32::MAX as u64) as i32;
        let bias_m = 1.0 / (1.0 - beta1.powi(step_i32));
        let bias_v = 1.0 / (1.0 - beta2.powi(step_i32));

        for state in &self.variables {
            let Some(raw_gradient) = gradients.get(&state.variable) else {
                continue;
            };
            let gradient = if gradient_scale < 1.0 {
                (raw_gradient * gradient_scale)?
            } else {
                raw_gradient.clone()
            };

            let next_first =
                ((state.first_moment.as_tensor() * beta1)? + (&gradient * (1.0 - beta1))?)?;
            let next_second =
                ((state.second_moment.as_tensor() * beta2)? + (gradient.sqr()? * (1.0 - beta2))?)?;
            let corrected_first = (&next_first * bias_m)?;
            let corrected_second = (&next_second * bias_v)?;
            let decayed_parameter = (state.variable.as_tensor() * (1.0 - decay))?;
            let denominator = (corrected_second.sqrt()? + self.config.epsilon)?;
            let adjusted_gradient = (corrected_first / denominator)?;
            let next_parameter = (decayed_parameter - (adjusted_gradient * learning_rate)?)?;

            state.first_moment.set(&next_first)?;
            state.second_moment.set(&next_second)?;
            state.variable.set(&next_parameter)?;
        }

        Ok(FoundationOptimizerStep {
            gradient_norm,
            gradient_scale,
            learning_rate,
            step: self.step,
        })
    }

    /// Current global L2 norm of available gradients.
    pub fn gradient_norm(&self, gradients: &GradStore) -> Result<f64> {
        let mut squared_norm = 0.0f64;
        for state in &self.variables {
            if let Some(gradient) = gradients.get(&state.variable) {
                let squared_sum = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?;
                squared_norm += f64::from(squared_sum);
            }
        }
        Ok(squared_norm.sqrt())
    }

    /// Dot product between two gradient stores over the optimizer's model variables.
    ///
    /// Variables missing from either store do not contribute. This is useful for
    /// diagnosing whether individual task gradients align with the combined
    /// multi-task update direction without flattening the full model into one
    /// additional tensor.
    pub fn gradient_dot(&self, left: &GradStore, right: &GradStore) -> Result<f64> {
        let mut dot = 0.0f64;
        for state in &self.variables {
            let (Some(left_gradient), Some(right_gradient)) =
                (left.get(&state.variable), right.get(&state.variable))
            else {
                continue;
            };
            let contribution = (left_gradient * right_gradient)?
                .sum_all()?
                .to_scalar::<f32>()?;
            dot += f64::from(contribution);
        }
        Ok(dot)
    }

    /// Cosine similarity between two gradient stores.
    ///
    /// Returns `None` when either gradient store has zero global norm. Values are
    /// clamped to `[-1, 1]` to absorb tiny floating-point excursions.
    pub fn gradient_cosine(&self, left: &GradStore, right: &GradStore) -> Result<Option<f64>> {
        let left_norm = self.gradient_norm(left)?;
        let right_norm = self.gradient_norm(right)?;
        if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
            return Ok(None);
        }
        let cosine = self.gradient_dot(left, right)? / (left_norm * right_norm);
        Ok(Some(cosine.clamp(-1.0, 1.0)))
    }

    /// Save first/second moments in one SafeTensors file.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut tensors = HashMap::<String, Tensor>::with_capacity(self.variables.len() * 2);
        for state in &self.variables {
            tensors.insert(
                format!("first::{}", state.name),
                state.first_moment.as_tensor().clone(),
            );
            tensors.insert(
                format!("second::{}", state.name),
                state.second_moment.as_tensor().clone(),
            );
        }
        safetensors::save(&tensors, path)
    }

    /// Restore first/second moments from a SafeTensors file.
    pub fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let Some(first_state) = self.variables.first() else {
            return Ok(());
        };
        let tensors = safetensors::load(path, first_state.variable.device())?;
        for state in &self.variables {
            let first_name = format!("first::{}", state.name);
            let second_name = format!("second::{}", state.name);
            let first = tensors.get(&first_name).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "optimizer checkpoint is missing first moment '{first_name}'"
                ))
            })?;
            let second = tensors.get(&second_name).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "optimizer checkpoint is missing second moment '{second_name}'"
                ))
            })?;
            state.first_moment.set(first)?;
            state.second_moment.set(second)?;
        }
        Ok(())
    }

    /// AdamW update count used by bias correction.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Restore the AdamW update count after moment loading.
    pub fn set_step_count(&mut self, step: u64) {
        self.step = step;
    }

    /// Current optimizer hyperparameters.
    pub fn config(&self) -> FoundationAdamWConfig {
        self.config
    }

    /// Update the learning rate without disturbing optimizer moments.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        if !(learning_rate > 0.0 && learning_rate.is_finite()) {
            candle_core::bail!("foundation learning rate must be positive and finite");
        }
        self.config.learning_rate = learning_rate;
        Ok(())
    }

    /// Number of floating-point model variables tracked by the optimizer.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }
}
