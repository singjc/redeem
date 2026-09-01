//! Observed-spectrum inputs for the inverse peptide foundation-model lane.
//!
//! This module intentionally accepts measured `(m/z, intensity)` peaks rather
//! than reconstructing theoretical fragment m/z values from a known peptide.
//! The latter would leak the answer into the inverse task and is therefore not
//! used as a training adapter.

use super::data::FoundationTrainingRecord;
use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

/// One observed centroided MS/MS peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoundationSpectrumPeak {
    /// Peak mass-to-charge ratio.
    pub mz: f32,
    /// Observed peak intensity in arbitrary units.
    pub intensity: f32,
}

/// One observed MS/MS spectrum supplied to the inverse model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FoundationSpectrum {
    /// Centroided product-ion peaks.
    pub peaks: Vec<FoundationSpectrumPeak>,
}

impl FoundationSpectrum {
    /// Construct a spectrum from `(m/z, intensity)` pairs.
    pub fn from_pairs(peaks: impl IntoIterator<Item = (f32, f32)>) -> Self {
        Self {
            peaks: peaks
                .into_iter()
                .map(|(mz, intensity)| FoundationSpectrumPeak { mz, intensity })
                .collect(),
        }
    }

    /// Build an inverse-model spectrum from fragment m/z values explicitly
    /// present in a loaded spectral-library/training record.
    ///
    /// Returns `None` when the source did not provide product m/z values. This
    /// function never calculates m/z from the known peptide sequence.
    pub fn from_training_record(record: &FoundationTrainingRecord) -> Option<Self> {
        let peaks: Vec<_> = record
            .fragments
            .iter()
            .filter_map(|fragment| {
                fragment
                    .product_mz
                    .filter(|mz| mz.is_finite() && *mz > 0.0)
                    .filter(|_| fragment.intensity.is_finite() && fragment.intensity > 0.0)
                    .map(|mz| FoundationSpectrumPeak {
                        mz,
                        intensity: fragment.intensity,
                    })
            })
            .collect();
        (!peaks.is_empty()).then_some(Self { peaks })
    }
}

/// Spectrum preprocessing/packing configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FoundationSpectrumConfig {
    /// Maximum number of peaks retained per spectrum.
    pub max_peaks: usize,
    /// Upper m/z scale used by the bounded continuous peak features.
    pub mz_scale: f32,
    /// Width of the CPU-computed continuous peak feature vector.
    pub peak_feature_dim: usize,
}

impl Default for FoundationSpectrumConfig {
    fn default() -> Self {
        Self {
            max_peaks: 256,
            mz_scale: 2_000.0,
            peak_feature_dim: 8,
        }
    }
}

impl FoundationSpectrumConfig {
    /// Validate preprocessing dimensions.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.max_peaks == 0 {
            return Err("foundation spectrum max_peaks must be greater than zero".into());
        }
        if !(self.mz_scale > 0.0 && self.mz_scale.is_finite()) {
            return Err("foundation spectrum mz_scale must be positive and finite".into());
        }
        if self.peak_feature_dim != 8 {
            return Err("foundation spectrum peak_feature_dim is currently fixed at 8".into());
        }
        Ok(())
    }
}

/// Tensorized observed spectra.
#[derive(Debug, Clone)]
pub struct FoundationSpectrumBatch {
    /// Continuous peak features `[batch, peaks, 8]`.
    pub peak_features: Tensor,
    /// One for retained observed peaks, zero for padding `[batch, peaks]`.
    pub peak_mask: Tensor,
    /// Original retained m/z values `[batch, peaks]` for diagnostics.
    pub mz: Tensor,
    /// Original normalized intensities `[batch, peaks]` for diagnostics.
    pub intensity: Tensor,
}

/// Deterministic observed-spectrum collator.
#[derive(Debug, Clone)]
pub struct FoundationSpectrumCollator {
    config: FoundationSpectrumConfig,
}

impl FoundationSpectrumCollator {
    /// Construct a spectrum collator.
    pub fn new(config: FoundationSpectrumConfig) -> Result<Self> {
        config.validate().map_err(candle_core::Error::Msg)?;
        Ok(Self { config })
    }

    /// Return the active spectrum configuration.
    pub fn config(&self) -> &FoundationSpectrumConfig {
        &self.config
    }

    /// Pack measured spectra into dense Candle tensors.
    ///
    /// Invalid/non-positive m/z values and non-finite/non-positive intensities
    /// are discarded. If a spectrum has more than `max_peaks`, the most intense
    /// peaks are retained and then ordered by m/z. Intensities are max-normalized
    /// per spectrum. The eight continuous features intentionally provide a small
    /// stable first implementation; later iterations can replace them with the
    /// multi-scale sinusoidal peak embedding used by modern de-novo models.
    pub fn collate(
        &self,
        spectra: &[FoundationSpectrum],
        device: &Device,
    ) -> Result<FoundationSpectrumBatch> {
        if spectra.is_empty() {
            candle_core::bail!("foundation spectrum collation requires a non-empty batch");
        }
        let b = spectra.len();
        let p = self.config.max_peaks;
        let f = self.config.peak_feature_dim;

        let mut features = vec![0.0f32; b * p * f];
        let mut mask = vec![0.0f32; b * p];
        let mut mz_values = vec![0.0f32; b * p];
        let mut intensities = vec![0.0f32; b * p];

        for (batch_idx, spectrum) in spectra.iter().enumerate() {
            let mut peaks: Vec<FoundationSpectrumPeak> = spectrum
                .peaks
                .iter()
                .copied()
                .filter(|peak| {
                    peak.mz.is_finite()
                        && peak.mz > 0.0
                        && peak.intensity.is_finite()
                        && peak.intensity > 0.0
                })
                .collect();

            peaks.sort_by(|left, right| {
                right
                    .intensity
                    .total_cmp(&left.intensity)
                    .then_with(|| left.mz.total_cmp(&right.mz))
            });
            peaks.truncate(p);
            peaks.sort_by(|left, right| left.mz.total_cmp(&right.mz));

            if peaks.is_empty() {
                candle_core::bail!(
                    "foundation spectrum batch item {batch_idx} contains no finite positive observed peaks"
                );
            }

            let max_intensity = peaks
                .iter()
                .map(|peak| peak.intensity)
                .fold(0.0f32, f32::max)
                .max(f32::EPSILON);

            for (peak_idx, peak) in peaks.iter().enumerate() {
                let normalized_intensity = (peak.intensity / max_intensity).clamp(0.0, 1.0);
                let scaled_mz = peak.mz / self.config.mz_scale;
                let feature = [
                    scaled_mz,
                    scaled_mz * scaled_mz,
                    (peak.mz / 10.0).sin(),
                    (peak.mz / 10.0).cos(),
                    (peak.mz / 100.0).sin(),
                    (peak.mz / 100.0).cos(),
                    normalized_intensity.sqrt(),
                    (1.0 + 9.0 * normalized_intensity).ln() / 10.0f32.ln(),
                ];
                let feature_base = (batch_idx * p + peak_idx) * f;
                features[feature_base..feature_base + f].copy_from_slice(&feature);
                mask[batch_idx * p + peak_idx] = 1.0;
                mz_values[batch_idx * p + peak_idx] = peak.mz;
                intensities[batch_idx * p + peak_idx] = normalized_intensity;
            }
        }

        Ok(FoundationSpectrumBatch {
            peak_features: Tensor::from_vec(features, (b, p, f), device)?,
            peak_mask: Tensor::from_vec(mask, (b, p), device)?,
            mz: Tensor::from_vec(mz_values, (b, p), device)?,
            intensity: Tensor::from_vec(intensities, (b, p), device)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_record_spectrum_uses_only_explicit_product_mz() {
        use crate::foundation::{
            FoundationTrainingRecord, FragmentTarget, PeptidoformInput, RetentionTimeLabels,
            TrainingContext,
        };
        let record = FoundationTrainingRecord {
            peptidoform: PeptidoformInput::unmodified("PEPTIDEK"),
            retention_time: RetentionTimeLabels::default(),
            ccs: None,
            fragments: vec![
                FragmentTarget {
                    cleavage_index: 1,
                    channel: 0,
                    intensity: 0.8,
                    product_mz: Some(250.2),
                },
                FragmentTarget {
                    cleavage_index: 2,
                    channel: 2,
                    intensity: 0.5,
                    product_mz: None,
                },
            ],
            context: TrainingContext::default(),
            run_id: None,
        };
        let spectrum = FoundationSpectrum::from_training_record(&record).unwrap();
        assert_eq!(spectrum.peaks.len(), 1);
        assert_eq!(spectrum.peaks[0].mz, 250.2);
    }

    #[test]
    fn spectrum_collator_filters_truncates_and_normalizes() {
        let device = Device::Cpu;
        let collator = FoundationSpectrumCollator::new(FoundationSpectrumConfig {
            max_peaks: 2,
            ..FoundationSpectrumConfig::default()
        })
        .unwrap();
        let spectrum = FoundationSpectrum::from_pairs([
            (500.0, 10.0),
            (100.0, 30.0),
            (300.0, 20.0),
            (-1.0, 99.0),
        ]);
        let batch = collator.collate(&[spectrum], &device).unwrap();
        assert_eq!(batch.peak_features.dims(), &[1, 2, 8]);
        assert_eq!(
            batch.peak_mask.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 1.0]]
        );
        assert_eq!(batch.mz.to_vec2::<f32>().unwrap(), vec![vec![100.0, 300.0]]);
        let intensity = batch.intensity.to_vec2::<f32>().unwrap();
        assert!((intensity[0][0] - 1.0).abs() < 1e-6);
        assert!((intensity[0][1] - (2.0 / 3.0)).abs() < 1e-6);
    }
}
