use anyhow::{anyhow, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Tensor};
use std::f64::consts::PI;

// Learning rate scheduler trait
/// Trait representing a learning rate scheduler that can be updated each step
/// and queried for the current learning rate.
pub trait LRScheduler {
    fn step(&mut self);
    fn get_last_lr(&self) -> f64;
}

// Cosine decay with warmup
/// Cosine learning rate scheduler with linear warmup phase.
///
/// This scheduler increases the learning rate linearly from 0 to `initial_lr`
/// over `num_warmup_steps`, then decays it using cosine annealing over the
/// remaining training steps, optionally over multiple cycles.
///
/// # Fields
/// * `initial_lr` - The peak learning rate after warmup.
/// * `current_step` - Internal counter of the current step.
/// * `num_warmup_steps` - Number of steps to warm up the learning rate.
/// * `num_training_steps` - Total number of training steps.
/// * `num_cycles` - Number of cosine cycles in the annealing phase.
pub struct CosineWithWarmup {
    initial_lr: f64,
    current_step: usize,
    num_warmup_steps: usize,
    num_training_steps: usize,
    num_cycles: f64,
}

impl CosineWithWarmup {
    /// Create a new `CosineWithWarmup` scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Maximum learning rate after warmup.
    /// * `num_warmup_steps` - Number of steps to linearly increase the learning rate.
    /// * `num_training_steps` - Total number of training steps.
    /// * `num_cycles` - Number of cosine cycles during decay.
    pub fn new(
        initial_lr: f64,
        num_warmup_steps: usize,
        num_training_steps: usize,
        num_cycles: f64,
    ) -> Self {
        Self {
            initial_lr,
            current_step: 0,
            num_warmup_steps,
            num_training_steps,
            num_cycles,
        }
    }

    /// Computes the learning rate for the current step based on warmup and cosine decay.
    fn get_lr(&self) -> f64 {
        if self.current_step < self.num_warmup_steps {
            return self.initial_lr * (self.current_step as f64) / (self.num_warmup_steps as f64);
        }

        let progress = (self.current_step - self.num_warmup_steps) as f64
            / (self.num_training_steps - self.num_warmup_steps).max(1) as f64;

        let cosine_decay =
            0.5 * (1.0 + (std::f64::consts::PI * self.num_cycles * 2.0 * progress).cos());
        self.initial_lr * cosine_decay.max(1e-10)
    }
}

impl LRScheduler for CosineWithWarmup {
    fn step(&mut self) {
        self.current_step += 1;
    }

    fn get_last_lr(&self) -> f64 {
        self.get_lr()
    }
}

/// Converts a device string to a Candle Device.
///
/// # Supported Device Strings
///
/// - `"cpu"`: Returns the CPU device
/// - `"cuda"`: Returns the default CUDA device (index 0)
/// - `"cuda:N"`: Returns the CUDA device with the specified index
///
/// # Arguments
///
/// * `device_str` - A string specifying the desired device
///
/// # Returns
///
/// A `Result` containing the Candle `Device` or an error if the device is not available
///
/// # Examples
///
/// ```
/// // Get the CPU device
/// let cpu_device = get_device("cpu")?;
///
/// // Get the default CUDA device (index 0)
/// let default_cuda = get_device("cuda")?;
///
/// // Get a specific CUDA device
/// let cuda_1 = get_device("cuda:1")?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The CUDA device is not available
/// - An unsupported device type is specified
pub fn get_device(device_str: &str) -> Result<Device> {
    if device_str.starts_with("cuda") {
        let cuda_index = if device_str == "cuda" {
            0 // Default to the first CUDA device
        } else {
            device_str
                .split(':')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        };

        let device = Device::cuda_if_available(cuda_index)?;
        if !device.is_cuda() {
            return Err(anyhow!("CUDA device {} is not available", cuda_index));
        }
        Ok(device)
    } else {
        match device_str {
            "cpu" => Ok(Device::Cpu),
            _ => Err(anyhow!("Unsupported device type: {}", device_str)),
        }
    }
}

/// Returns the best available device based on the specified flags.
///
/// # Arguments
///
/// * `cpu` - A flag indicating whether to use the CPU device
///
/// # Returns
///
/// A `Result` containing the Candle `Device` or an error if the device is not available
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn get_tensor_stats(x: &Tensor) -> Result<(f32, f32, f32), candle_core::Error> {
    // let flat: Vec<f32> = match x.rank() {
    //     0 => vec![x.to_scalar::<f32>()?],
    //     1 => x.to_vec1::<f32>()?,
    //     2 => x.to_vec2::<f32>()?.into_iter().flatten().collect(),
    //     3 => x.to_vec3::<f32>()?.into_iter().flatten().flatten().collect(),
    //     _ => return Err(candle_core::Error::Msg(format!("Unsupported tensor rank: {}", x.rank()))),
    // };
    let flat = x.flatten_all()?.to_vec1::<f32>()?;

    if flat.is_empty() {
        return Err(candle_core::Error::Msg(
            "Tensor has no elements to compute stats.".to_string(),
        ));
    }

    let mean = flat.iter().copied().sum::<f32>() / flat.len() as f32;
    let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    Ok((mean, min, max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::utils::{cuda_is_available, metal_is_available};
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn test_device() {
        let device = get_device("cpu").unwrap();
        println!("Device: {:?}", device);

        if cuda_is_available() {
            let device = get_device("cuda").unwrap();
            println!("Device: {:?}", device);
        } else {
            println!("CUDA is not available");
        }

        if metal_is_available() {
            let device = get_device("metal").unwrap();
            println!("Device: {:?}", device);
        }
    }
}
