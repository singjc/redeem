//! Observed-spectrum inputs for the inverse peptide foundation-model lane.
//!
//! This module intentionally accepts measured `(m/z, intensity)` peaks rather
//! than reconstructing theoretical fragment m/z values from a known peptide.
//! The latter would leak the answer into the inverse task and is therefore not
//! used as a training adapter.

use super::data::FoundationTrainingRecord;
use candle_core::{Device, Result, Tensor};
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
        let raw_peaks: Vec<_> = record
            .observed_spectrum_peaks
            .iter()
            .filter(|peak| {
                peak.mz.is_finite()
                    && peak.mz > 0.0
                    && peak.intensity.is_finite()
                    && peak.intensity > 0.0
            })
            .map(|peak| FoundationSpectrumPeak {
                mz: peak.mz,
                intensity: peak.intensity,
            })
            .collect();
        if !raw_peaks.is_empty() {
            return Some(Self { peaks: raw_peaks });
        }

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
            peak_feature_dim: 32,
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
        if self.peak_feature_dim != 32 {
            return Err("foundation spectrum peak_feature_dim is currently fixed at 32".into());
        }
        Ok(())
    }
}

/// Tensorized observed spectra.
#[derive(Debug, Clone)]
pub struct FoundationSpectrumBatch {
    /// Continuous peak features `[batch, peaks, 32]`.
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
    /// per spectrum. The current 32-dimensional representation
    /// includes a multi-scale Fourier m/z embedding so self-attention can resolve
    /// chemically meaningful peak-to-peak mass differences across sparse spectra.
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
                let mut feature = [0.0f32; 32];
                feature[0] = scaled_mz;
                feature[1] = scaled_mz * scaled_mz;
                feature[2] = normalized_intensity.sqrt();
                feature[3] = (1.0 + 9.0 * normalized_intensity).ln() / 10.0f32.ln();

                // Multi-scale Fourier m/z features. Dot products between these
                // encodings expose peak-to-peak mass differences at resolutions
                // ranging from sub-Da to whole-spectrum scale, which is a much
                // stronger inductive bias for sparse fragment ladders than the
                // original two arbitrary sinusoidal frequencies.
                const WAVELENGTHS_DA: [f32; 14] = [
                    0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1_024.0,
                    2_048.0, 4_096.0,
                ];
                for (scale_index, wavelength) in WAVELENGTHS_DA.iter().enumerate() {
                    let angle = std::f32::consts::TAU * peak.mz / *wavelength;
                    feature[4 + 2 * scale_index] = angle.sin();
                    feature[5 + 2 * scale_index] = angle.cos();
                }

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

/// Stable spectrum-aware fingerprint for one inverse-training record.
///
/// This extends the historical forward record fingerprint with the explicit
/// observed/library product-m/z content retained for the inverse lane. Fragment
/// annotations are not used as inverse-model features, but m/z and intensity are
/// part of the data-integrity contract because changing either changes the
/// spectrum observed by the diffusion model.
pub fn foundation_diffusion_record_fingerprint(record: &FoundationTrainingRecord) -> u64 {
    let mut hash = SpectrumFnv64::new();
    hash.u64(super::experiment::foundation_record_fingerprint(record));
    let mut peaks: Vec<(u32, u32)> = if !record.observed_spectrum_peaks.is_empty() {
        record
            .observed_spectrum_peaks
            .iter()
            .map(|peak| (peak.mz.to_bits(), peak.intensity.to_bits()))
            .collect()
    } else {
        record
            .fragments
            .iter()
            .filter_map(|fragment| {
                fragment
                    .product_mz
                    .map(|mz| (mz.to_bits(), fragment.intensity.to_bits()))
            })
            .collect()
    };
    peaks.sort_unstable();
    hash.usize(peaks.len());
    for (mz, intensity) in peaks {
        hash.u32(mz);
        hash.u32(intensity);
    }
    hash.finish()
}

/// Stable order-independent fingerprint over a selected inverse dataset.
///
/// Unlike [`super::experiment::foundation_dataset_fingerprint`], this hash
/// changes when explicitly observed/library product m/z values change.
pub fn foundation_diffusion_dataset_fingerprint(
    records: &[FoundationTrainingRecord],
    selected_indices: &[usize],
) -> anyhow::Result<u64> {
    let mut record_hashes = Vec::with_capacity(selected_indices.len());
    for &index in selected_indices {
        let record = records.get(index).ok_or_else(|| {
            anyhow::anyhow!("foundation diffusion fingerprint index {index} is out of bounds")
        })?;
        record_hashes.push(foundation_diffusion_record_fingerprint(record));
    }
    record_hashes.sort_unstable();
    let mut hash = SpectrumFnv64::new();
    hash.usize(record_hashes.len());
    for value in record_hashes {
        hash.u64(value);
    }
    Ok(hash.finish())
}

struct SpectrumFnv64(u64);

impl SpectrumFnv64 {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn finish(self) -> u64 {
        self.0
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
            observed_spectrum_peaks: Vec::new(),
            context: TrainingContext::default(),
            run_id: None,
        };
        let spectrum = FoundationSpectrum::from_training_record(&record).unwrap();
        assert_eq!(spectrum.peaks.len(), 1);
        assert_eq!(spectrum.peaks[0].mz, 250.2);
    }

    #[test]
    fn training_record_spectrum_prefers_raw_observed_msp_peaks() {
        use crate::foundation::{
            FoundationTrainingRecord, FragmentTarget, ObservedSpectrumPeak, PeptidoformInput,
            RetentionTimeLabels, TrainingContext,
        };
        let record = FoundationTrainingRecord {
            peptidoform: PeptidoformInput::unmodified("PEPTIDEK"),
            retention_time: RetentionTimeLabels::default(),
            ccs: None,
            fragments: vec![FragmentTarget {
                cleavage_index: 1,
                channel: 0,
                intensity: 0.8,
                product_mz: Some(250.2),
            }],
            observed_spectrum_peaks: vec![
                ObservedSpectrumPeak {
                    mz: 111.1,
                    intensity: 10.0,
                },
                ObservedSpectrumPeak {
                    mz: 222.2,
                    intensity: 20.0,
                },
            ],
            context: TrainingContext::default(),
            run_id: None,
        };
        let spectrum = FoundationSpectrum::from_training_record(&record).unwrap();
        assert_eq!(spectrum.peaks.len(), 2);
        assert_eq!(spectrum.peaks[0].mz, 111.1);
        assert_eq!(spectrum.peaks[1].mz, 222.2);
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
