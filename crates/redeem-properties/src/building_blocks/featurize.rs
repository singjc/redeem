use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;
use std::sync::atomic::Ordering;
use std::{collections::HashMap, sync::Arc};

use crate::building_blocks::building_blocks::AA_EMBEDDING_SIZE;

const VALID_AA: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Precomputes amino acid index map from characters A-Z
fn aa_index_map() -> HashMap<char, i64> {
    VALID_AA
        .chars()
        .enumerate()
        .map(|(i, c)| (c, i as i64 + 1))
        .collect()
}

/// Convert peptide sequences into AA ID array.
///
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L88
pub fn aa_indices_tensor(seq: &str, device: &Device) -> Result<Tensor> {
    let map = aa_index_map();
    let filtered: Vec<i64> = seq
        .chars()
        .map(|c| {
            map.get(&c)
                .copied()
                .ok_or_else(|| anyhow!("Unknown amino acid character: '{}'", c))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut indices = vec![0i64]; // padding start
    indices.extend(filtered);
    indices.push(0); // padding end

    Ok(Tensor::from_slice(&indices, (1, indices.len()), device)?
        .to_dtype(DType::F32)?
        .unsqueeze(2)?)
}

/// Convert peptide sequences into AA ID array using Arc<[u8]>.
/// This avoids converting the whole sequence to a String or &str unless necessary.
pub fn aa_indices_tensor_from_arc(seq: &Arc<[u8]>, device: &Device) -> Result<Tensor> {
    let map = aa_index_map();
    let filtered: Vec<i64> = seq
        .iter()
        .map(|&b| {
            let c = b as char;
            map.get(&c)
                .copied()
                .ok_or_else(|| anyhow!("Unknown amino acid character: '{}'", c))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut indices = vec![0i64]; // padding start
    indices.extend(filtered);
    indices.push(0); // padding end

    Ok(Tensor::from_slice(&indices, (1, indices.len()), device)?
        .to_dtype(DType::F32)?
        .unsqueeze(2)?)
}

/// One-hot encode amino acid indices and concatenate additional tensors.
pub fn aa_one_hot(aa_indices: &Tensor, cat_others: &[&Tensor]) -> Result<Tensor> {
    let (batch_size, seq_len) = aa_indices.shape().dims2()?;
    log::trace!(
        "[aa_one_hot] batch_size: {}, seq_len: {}",
        batch_size,
        seq_len
    );
    let num_classes = AA_EMBEDDING_SIZE;

    let indices = aa_indices.to_vec2::<f32>()?;

    for (i, row) in indices.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if !val.is_finite() || *val < 0.0 || *val > (AA_EMBEDDING_SIZE as f32) {
                log::error!(
                    "[aa_one_hot] Invalid index at batch {}, position {}: {}",
                    i,
                    j,
                    val
                );
            }
        }
    }

    let mut one_hot_data = vec![0.0f32; batch_size * seq_len * num_classes];

    one_hot_data
        .par_chunks_mut(seq_len * num_classes)
        .zip(indices.par_iter())
        .enumerate()
        .try_for_each(|(batch_idx, (chunk, row))| -> Result<()> {
            for (seq_idx, &fidx) in row.iter().enumerate() {
                if !fidx.is_finite() {
                    return Err(anyhow!(
                        "Invalid AA index: found NaN or Inf at batch {}, position {}: {}",
                        batch_idx, seq_idx, fidx
                    ));
                }

                if fidx < 0.0 {
                    return Err(anyhow!(
                        "Invalid AA index: negative value at batch {}, position {}: {}",
                        batch_idx, seq_idx, fidx
                    ));
                }

                let class_idx = fidx.round() as usize;
                if class_idx >= num_classes {
                    return Err(anyhow!(
                        "AA index out of bounds: got {}, but num_classes = {} (batch {}, position {})",
                        class_idx, num_classes, batch_idx, seq_idx
                    ));
                }

                let index = seq_idx * num_classes + class_idx;
                chunk[index] = 1.0;
            }
            Ok(())
        })?;

    let one_hot_tensor = Tensor::from_slice(
        &one_hot_data,
        (batch_size, seq_len, num_classes),
        aa_indices.device(),
    )
    .map_err(|e| anyhow!("Failed to create one-hot tensor: {}", e))?;

    if cat_others.is_empty() {
        Ok(one_hot_tensor)
    } else {
        let mut features = vec![one_hot_tensor];
        features.extend(cat_others.iter().cloned().cloned());
        // Build concatenated tensor so we can both return it and optionally
        // write a post-concatenation diagnostic CSV (per-sample rows) to help
        // align with other encoder dumps.
        let concatenated = Tensor::cat(&features, 2)?;

        Ok(concatenated)
    }
}

/// Get the modification features for a given set of modifications and modification sites.
///
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L47
pub fn get_mod_features_from_parsed(
    mod_names: &[&str],
    mod_sites: &[usize],
    seq_len: usize,
    mod_feature_size: usize,
    mod_to_feature: &HashMap<String, Vec<f32>>,
    device: &Device,
) -> Result<Tensor> {
    // Build mod features deterministically.
    // Previously this used a Vec<AtomicU32> with bitwise fetch_add which
    // effectively added u32 bit patterns (incorrect for floats) and is
    // unsafe for concurrent accumulation. Switch to a simple, correct
    // sequential accumulation here to ensure deterministic, correct
    // behavior for diagnostic runs.
    let mut mod_x = vec![0.0f32; seq_len * mod_feature_size];

    for (&mod_name, &site) in mod_names.iter().zip(mod_sites.iter()) {
        if site >= seq_len {
            log::warn!(
                "Skipping mod {} at invalid site {} (seq_len {})",
                mod_name,
                site,
                seq_len
            );
            continue;
        }
        if let Some(feat) = mod_to_feature.get(mod_name) {
            for (i, &val) in feat.iter().enumerate() {
                let idx = site * mod_feature_size + i;
                mod_x[idx] += val;
            }
        } else {
            log::warn!("Unknown modification feature: {}", mod_name);
        }
    }

    let t = Tensor::from_slice(&mod_x, (1, seq_len, mod_feature_size), device)
        .map_err(|e| anyhow!("Failed to create tensor: {}", e))?;
    Ok(t)
}

pub fn get_mod_features_from_parsed_arc(
    mod_names: &[Arc<[u8]>],
    mod_sites: &[usize],
    seq_len: usize,
    mod_feature_size: usize,
    mod_to_feature: &HashMap<Arc<[u8]>, Vec<f32>>,
    device: &Device,
) -> Result<Tensor> {
    // Deterministic sequential accumulation for Arc<[u8]> mod names.
    let mut mod_x = vec![0.0f32; seq_len * mod_feature_size];

    for (mod_name, &site) in mod_names.iter().zip(mod_sites.iter()) {
        if site >= seq_len {
            log::warn!(
                "Skipping mod {:?} at invalid site {} (seq_len {})",
                std::str::from_utf8(mod_name).unwrap_or("<invalid>"),
                site,
                seq_len
            );
            continue;
        }
        if let Some(feat) = mod_to_feature.get(mod_name) {
            for (i, &val) in feat.iter().enumerate() {
                let idx = site * mod_feature_size + i;
                mod_x[idx] += val;
            }
        } else {
            log::warn!(
                "Unknown modification feature: {:?}",
                std::str::from_utf8(mod_name).unwrap_or("<invalid>")
            );
        }
    }

    Tensor::from_slice(&mod_x, (1, seq_len, mod_feature_size), device)
        .map_err(|e| anyhow!("Failed to create tensor: {}", e))
}

#[cfg(test)]
mod tests {

    use crate::utils::peptdeep_utils::parse_model_constants;
    use crate::utils::peptdeep_utils::ModelConstants;
    use crate::utils::peptdeep_utils::{load_mod_to_feature, load_mod_to_feature_arc};

    use super::*;
    use candle_core::Device;
    use candle_core::Tensor;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn test_aa_indices_tensor() {
        let device = Device::Cpu;
        let seq = "AGHCEWQMKYR";
        let start_time = std::time::Instant::now();
        let result = aa_indices_tensor(seq, &device).unwrap();
        println!("aa_indices_tensor Time taken: {:?}", start_time.elapsed());
        // expected result is [[0, 1, 7, 8, 3, 5, 23, 17, 13, 11, 25, 18, 0]]
        let expect_out = Tensor::from_vec(
            vec![
                0.0f32, 1.0f32, 7.0f32, 8.0f32, 3.0f32, 5.0f32, 23.0f32, 17.0f32, 13.0f32, 11.0f32,
                25.0f32, 18.0f32, 0.0f32,
            ],
            (1, 13),
            &device,
        )
        .unwrap();
        println!(
            "{:?} - aa_indices_tensor: {:?}",
            seq,
            result.to_vec3::<f32>().unwrap()
        );
        println!("result shape: {:?}", result.shape());
        assert_eq!(result.shape().dims(), &[1, 13, 1]);
        // assert_eq!(result.to_vec3::<f32>().unwrap(), expect_out.to_vec3::<f32>().unwrap());

        let seq_bytes = Arc::from(seq.as_bytes().to_vec().into_boxed_slice());
        let start_time = std::time::Instant::now();
        let result = aa_indices_tensor_from_arc(&seq_bytes, &device).unwrap();
        println!(
            "aa_indices_tensor_from_arc Time taken: {:?}",
            start_time.elapsed()
        );
        println!(
            "{:?} - aa_indices_tensor_from_arc: {:?}",
            seq,
            result.to_vec3::<f32>().unwrap()
        );
        assert_eq!(result.shape().dims(), &[1, 13, 1]);
        // assert_eq!(result.to_vec3::<f32>().unwrap(), expect_out.to_vec3::<f32>().unwrap());
    }

    #[test]
    fn test_get_mod_features_from_parsed() {
        let mods_str = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let sites_str = "0;4;8";

        // Manually parse and split
        let mod_names: Vec<&str> = mods_str.split(';').filter(|s| !s.is_empty()).collect();
        let mod_sites: Vec<usize> = sites_str
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().unwrap())
            .collect();
        let seq_len = 11 + 2;
        let mod_feature_size = 109;

        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        let constants: ModelConstants =
            parse_model_constants(constants_path.to_str().unwrap()).unwrap();
        let mod_to_feature: HashMap<String, Vec<f32>> = load_mod_to_feature(&constants).unwrap();

        let device = Device::Cpu;
        let start_time = std::time::Instant::now();
        let tensor = get_mod_features_from_parsed(
            &mod_names,
            &mod_sites,
            seq_len,
            mod_feature_size,
            &mod_to_feature,
            &device,
        )
        .unwrap();
        println!(
            "get_mod_features_from_parsed Time taken: {:?}",
            start_time.elapsed()
        );

        println!("tensor shape: {:?}", tensor.shape());

        assert_eq!(tensor.shape().dims(), &[1, seq_len, mod_feature_size]);

        let mod_to_feature = load_mod_to_feature_arc(&constants).unwrap();
        let mod_names_arc: Vec<Arc<[u8]>> = mod_names
            .iter()
            .map(|&s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();
        let mod_sites_arc: Vec<usize> = mod_sites.iter().map(|&s| s).collect();
        let start_time = std::time::Instant::now();
        let tensor_arc = get_mod_features_from_parsed_arc(
            &mod_names_arc,
            &mod_sites_arc,
            seq_len,
            mod_feature_size,
            &mod_to_feature,
            &device,
        )
        .unwrap();
        println!(
            "get_mod_features_from_parsed_arc Time taken: {:?}",
            start_time.elapsed()
        );
        println!("tensor_arc shape: {:?}", tensor_arc.shape());
        assert_eq!(tensor_arc.shape().dims(), &[1, seq_len, mod_feature_size]);
        assert_eq!(tensor.shape(), tensor_arc.shape());
    }
}
