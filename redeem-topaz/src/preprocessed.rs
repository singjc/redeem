//! Chunked on-disk storage for preprocessed TOPAZ inputs.
//!
//! Large TOPAZ runs spend substantial wall time on host-side work before the
//! GPU can score anything:
//!
//! - OSW row loading
//! - XIC/XIM parquet reads
//! - MSNumpress decode
//! - fixed-width crop/pad/mask logic
//! - trace tensor assembly
//!
//! This module materializes the expensive host-side preprocessing step into a
//! reusable archive so later `train`, `infer`, or `xrun-train` commands can
//! operate on row-aligned tensors directly.
//!
//! The archive format is intentionally simple:
//!
//! - one uncompressed zip file
//! - one JSON manifest describing compatibility-critical metadata
//! - one set of binary tensor/column entries per chunk
//!
//! Chunk payloads are stored in little-endian raw arrays so loading them back is
//! fast and does not require re-running parquet decode.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use zip::CompressionMethod;
use zip::ZipArchive;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

use crate::infer::TraceBuildConfig;
use crate::io::osw::{FeatureRow, OswReadConfig};

const MANIFEST_ENTRY: &str = "manifest.json";
const FORMAT_VERSION: u32 = 1;

/// Provenance metadata stored in the preprocessing manifest.
///
/// These fields are informational and help operators diagnose whether a bundle
/// was built from the expected OSW/XIC/XIM inputs. Compatibility checks rely on
/// the structured trace/config fields in [`PreprocessedManifest`], not on these
/// file paths.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PreprocessedProvenance {
    /// Source OSW path used when the bundle was created.
    pub osw_path: Option<PathBuf>,
    /// XIC parquet paths consulted during preprocessing.
    pub xic_paths: Vec<PathBuf>,
    /// Optional explicit run-to-XIC map path.
    pub xic_map_path: Option<PathBuf>,
    /// XIM parquet paths consulted during preprocessing.
    pub xim_paths: Vec<PathBuf>,
    /// Optional explicit run-to-XIM map path.
    pub xim_map_path: Option<PathBuf>,
}

/// One chunk entry recorded in the preprocessing manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessedChunkMeta {
    /// Zero-based chunk index.
    pub index: usize,
    /// Number of rows stored in this chunk.
    pub rows: usize,
}

/// Bundle-wide metadata used to validate compatibility before scoring or
/// training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessedManifest {
    /// Bundle format version.
    pub version: u32,
    /// Unix timestamp (seconds) when the archive was created.
    pub created_unix_secs: u64,
    /// Total number of candidate rows stored across all chunks.
    pub row_count: usize,
    /// Nominal chunk row count requested during preprocessing.
    pub chunk_row_count: usize,
    /// Scalar feature column order stored in every chunk.
    pub feature_cols: Vec<String>,
    /// Fixed-width XIC tensor layout used for `x_trace`.
    pub trace: TraceBuildConfig,
    /// Optional fixed-width XIM tensor layout used for `x_xim`.
    pub xim_trace: Option<TraceBuildConfig>,
    /// OSW reader configuration used to populate the stored rows.
    pub osw: OswReadConfig,
    /// Informational input provenance.
    pub provenance: PreprocessedProvenance,
    /// Chunk manifest entries in archive order.
    pub chunks: Vec<PreprocessedChunkMeta>,
}

impl PreprocessedManifest {
    /// Create a new manifest before any chunks have been written.
    pub fn new(
        chunk_row_count: usize,
        feature_cols: Vec<String>,
        trace: TraceBuildConfig,
        xim_trace: Option<TraceBuildConfig>,
        osw: OswReadConfig,
        provenance: PreprocessedProvenance,
    ) -> Self {
        let created_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            version: FORMAT_VERSION,
            created_unix_secs,
            row_count: 0,
            chunk_row_count: chunk_row_count.max(1),
            feature_cols,
            trace,
            xim_trace,
            osw,
            provenance,
            chunks: Vec::new(),
        }
    }

    /// Return the scalar feature width `D` stored in the bundle.
    pub fn feat_dim(&self) -> usize {
        self.feature_cols.len()
    }
}

/// One decoded preprocessing chunk.
#[derive(Debug, Clone)]
pub struct PreprocessedChunk {
    /// Candidate rows in the same order as the tensor payloads.
    pub rows: Vec<FeatureRow>,
    /// Flattened `(N, C, L)` XIC tensor for this chunk.
    pub x_trace: Vec<f32>,
    /// Optional flattened `(N, C_xim, L_xim)` XIM tensor for this chunk.
    pub x_xim: Option<Vec<f32>>,
}

/// Fully materialized preprocessing bundle.
///
/// This is convenient for training and XRUN calibration, which typically want
/// the whole filtered dataset in memory before bagging/splitting.
#[derive(Debug, Clone)]
pub struct PreprocessedDataset {
    /// Bundle manifest.
    pub manifest: PreprocessedManifest,
    /// Concatenated candidate rows from all chunks.
    pub rows: Vec<FeatureRow>,
    /// Concatenated XIC tensor buffer.
    pub x_trace: Vec<f32>,
    /// Concatenated XIM tensor buffer when present.
    pub x_xim: Option<Vec<f32>>,
}

/// Sequential reader for a `topaz_inputs.topazdata` archive.
///
/// The reader keeps only the manifest in memory and opens the zip archive on
/// demand when individual chunks are requested.
#[derive(Debug, Clone)]
pub struct PreprocessedBundleReader {
    path: PathBuf,
    manifest: PreprocessedManifest,
}

impl PreprocessedBundleReader {
    /// Open an existing preprocessing bundle and load its manifest.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let manifest = read_manifest(&path)?;
        Ok(Self { path, manifest })
    }

    /// Access the parsed bundle manifest.
    pub fn manifest(&self) -> &PreprocessedManifest {
        &self.manifest
    }

    /// Load one chunk by its zero-based index.
    pub fn load_chunk(&self, index: usize) -> Result<PreprocessedChunk> {
        let Some(chunk_meta) = self.manifest.chunks.get(index) else {
            bail!(
                "preprocessed chunk index {} out of range (n_chunks={})",
                index,
                self.manifest.chunks.len()
            );
        };
        let file = File::open(&self.path)
            .with_context(|| format!("failed to open preprocessed archive {:?}", self.path))?;
        let mut archive = ZipArchive::new(file)?;
        read_chunk_from_archive(&mut archive, &self.manifest, chunk_meta)
    }

    /// Load every chunk and concatenate them into a single in-memory dataset.
    pub fn load_all(&self) -> Result<PreprocessedDataset> {
        let mut rows = Vec::with_capacity(self.manifest.row_count);
        let trace_span = self.manifest.trace.total_c() * self.manifest.trace.l;
        let mut x_trace = Vec::with_capacity(self.manifest.row_count * trace_span);
        let xim_span = self
            .manifest
            .xim_trace
            .as_ref()
            .map(|cfg| cfg.total_c() * cfg.l);
        let mut x_xim = xim_span.map(|span| Vec::with_capacity(self.manifest.row_count * span));

        for idx in 0..self.manifest.chunks.len() {
            let chunk = self.load_chunk(idx)?;
            rows.extend(chunk.rows);
            x_trace.extend(chunk.x_trace);
            if let (Some(dst), Some(src)) = (x_xim.as_mut(), chunk.x_xim) {
                dst.extend(src);
            }
        }

        Ok(PreprocessedDataset {
            manifest: self.manifest.clone(),
            rows,
            x_trace,
            x_xim,
        })
    }
}

/// Streaming writer used by `topaz preprocess`.
#[derive(Debug)]
pub struct PreprocessedBundleWriter {
    path: PathBuf,
    writer: Option<ZipWriter<File>>,
    manifest: PreprocessedManifest,
}

impl PreprocessedBundleWriter {
    /// Create a new archive writer.
    pub fn new(path: impl AsRef<Path>, manifest: PreprocessedManifest) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(&path)
            .with_context(|| format!("failed to create preprocessed archive {:?}", path))?;
        let writer = ZipWriter::new(file);
        Ok(Self {
            path,
            writer: Some(writer),
            manifest,
        })
    }

    /// Append one row-aligned chunk to the archive.
    ///
    /// `x_trace` must contain `rows.len() * trace.total_c() * trace.l` floats.
    /// When present, `x_xim` must contain `rows.len() * xim_trace.total_c() * xim_trace.l`
    /// floats.
    pub fn write_chunk(
        &mut self,
        rows: &[FeatureRow],
        x_trace: &[f32],
        x_xim: Option<&[f32]>,
    ) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .context("cannot write chunk after bundle has been finished")?;
        let rows_len = rows.len();
        let trace_expected = rows_len * self.manifest.trace.total_c() * self.manifest.trace.l;
        if x_trace.len() != trace_expected {
            bail!(
                "preprocessed XIC chunk shape mismatch: rows={} expected {} floats but got {}",
                rows_len,
                trace_expected,
                x_trace.len()
            );
        }
        match (self.manifest.xim_trace.as_ref(), x_xim) {
            (Some(xim_cfg), Some(buf)) => {
                let expected = rows_len * xim_cfg.total_c() * xim_cfg.l;
                if buf.len() != expected {
                    bail!(
                        "preprocessed XIM chunk shape mismatch: rows={} expected {} floats but got {}",
                        rows_len,
                        expected,
                        buf.len()
                    );
                }
            }
            (Some(_), None) => bail!("bundle manifest expects XIM data but chunk omitted it"),
            (None, Some(_)) => bail!("bundle manifest disables XIM but chunk provided XIM data"),
            (None, None) => {}
        }

        let chunk_idx = self.manifest.chunks.len();
        let prefix = chunk_prefix(chunk_idx);
        let feat_dim = self.manifest.feature_cols.len();
        let opts = zip_options();

        write_entry(
            writer,
            &format!("{prefix}/feature_id.u64le"),
            opts,
            &encode_u64_slice(&rows.iter().map(|r| r.feature_id).collect::<Vec<_>>()),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/precursor_id.u64le"),
            opts,
            &encode_u64_slice(&rows.iter().map(|r| r.precursor_id).collect::<Vec<_>>()),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/run_id.u64le"),
            opts,
            &encode_u64_slice(&rows.iter().map(|r| r.run_id).collect::<Vec<_>>()),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/exp_rt.f32le"),
            opts,
            &encode_f32_slice(&rows.iter().map(|r| r.exp_rt).collect::<Vec<_>>()),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/rt_left_width.f32le"),
            opts,
            &encode_f32_slice(
                &rows
                    .iter()
                    .map(|r| r.rt_left_width.unwrap_or(f32::NAN))
                    .collect::<Vec<_>>(),
            ),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/rt_right_width.f32le"),
            opts,
            &encode_f32_slice(
                &rows
                    .iter()
                    .map(|r| r.rt_right_width.unwrap_or(f32::NAN))
                    .collect::<Vec<_>>(),
            ),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/exp_im.f32le"),
            opts,
            &encode_f32_slice(
                &rows
                    .iter()
                    .map(|r| r.exp_im.unwrap_or(f32::NAN))
                    .collect::<Vec<_>>(),
            ),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/exp_im_left_width.f32le"),
            opts,
            &encode_f32_slice(
                &rows
                    .iter()
                    .map(|r| r.exp_im_left_width.unwrap_or(f32::NAN))
                    .collect::<Vec<_>>(),
            ),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/exp_im_right_width.f32le"),
            opts,
            &encode_f32_slice(
                &rows
                    .iter()
                    .map(|r| r.exp_im_right_width.unwrap_or(f32::NAN))
                    .collect::<Vec<_>>(),
            ),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/is_decoy.u8"),
            opts,
            &rows
                .iter()
                .map(|r| if r.is_decoy { 1u8 } else { 0u8 })
                .collect::<Vec<_>>(),
        )?;

        let mut features = Vec::with_capacity(rows_len * feat_dim);
        for row in rows {
            if row.features.len() != feat_dim {
                bail!(
                    "row feature width mismatch in preprocessed chunk: expected {} cols but feature_id {} has {}",
                    feat_dim,
                    row.feature_id,
                    row.features.len()
                );
            }
            features.extend_from_slice(&row.features);
        }
        write_entry(
            writer,
            &format!("{prefix}/features.f32le"),
            opts,
            &encode_f32_slice(&features),
        )?;
        write_entry(
            writer,
            &format!("{prefix}/x_trace.f32le"),
            opts,
            &encode_f32_slice(x_trace),
        )?;
        if let Some(buf) = x_xim {
            write_entry(
                writer,
                &format!("{prefix}/x_xim.f32le"),
                opts,
                &encode_f32_slice(buf),
            )?;
        }

        self.manifest.row_count += rows_len;
        self.manifest.chunks.push(PreprocessedChunkMeta {
            index: chunk_idx,
            rows: rows_len,
        });
        Ok(())
    }

    /// Finish the archive and write `manifest.json`.
    pub fn finish(mut self) -> Result<PreprocessedManifest> {
        let writer = self.writer.take().context("cannot finish bundle twice")?;
        let mut writer = writer;
        let manifest_bytes = serde_json::to_vec_pretty(&self.manifest)?;
        let opts = zip_options();
        write_entry(&mut writer, MANIFEST_ENTRY, opts, &manifest_bytes)?;
        writer.finish()?;
        Ok(self.manifest)
    }

    /// Path of the archive being written.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn zip_options() -> SimpleFileOptions {
    SimpleFileOptions::default().compression_method(CompressionMethod::Stored)
}

fn chunk_prefix(index: usize) -> String {
    format!("chunks/{index:06}")
}

fn write_entry(
    writer: &mut ZipWriter<File>,
    name: &str,
    opts: SimpleFileOptions,
    bytes: &[u8],
) -> Result<()> {
    writer.start_file(name, opts)?;
    writer.write_all(bytes)?;
    Ok(())
}

fn read_manifest(path: &Path) -> Result<PreprocessedManifest> {
    let file = File::open(path)
        .with_context(|| format!("failed to open preprocessed archive {:?}", path))?;
    let mut archive = ZipArchive::new(file)?;
    let mut entry = archive.by_name(MANIFEST_ENTRY).with_context(|| {
        format!(
            "preprocessed archive {:?} is missing {MANIFEST_ENTRY}",
            path
        )
    })?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    let manifest: PreprocessedManifest = serde_json::from_slice(&bytes)?;
    if manifest.version != FORMAT_VERSION {
        bail!(
            "unsupported preprocessed bundle version {} (expected {})",
            manifest.version,
            FORMAT_VERSION
        );
    }
    Ok(manifest)
}

fn read_chunk_from_archive(
    archive: &mut ZipArchive<File>,
    manifest: &PreprocessedManifest,
    chunk_meta: &PreprocessedChunkMeta,
) -> Result<PreprocessedChunk> {
    let prefix = chunk_prefix(chunk_meta.index);
    let rows_len = chunk_meta.rows;
    let feat_dim = manifest.feature_cols.len();

    let feature_id = decode_u64_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/feature_id.u64le"),
    )?)?;
    let precursor_id = decode_u64_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/precursor_id.u64le"),
    )?)?;
    let run_id = decode_u64_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/run_id.u64le"),
    )?)?;
    let exp_rt = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/exp_rt.f32le"),
    )?)?;
    let rt_left_width = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/rt_left_width.f32le"),
    )?)?;
    let rt_right_width = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/rt_right_width.f32le"),
    )?)?;
    let exp_im = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/exp_im.f32le"),
    )?)?;
    let exp_im_left_width = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/exp_im_left_width.f32le"),
    )?)?;
    let exp_im_right_width = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/exp_im_right_width.f32le"),
    )?)?;
    let is_decoy = read_entry_bytes(archive, &format!("{prefix}/is_decoy.u8"))?;
    let features = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/features.f32le"),
    )?)?;
    let x_trace = decode_f32_vec(read_entry_bytes(
        archive,
        &format!("{prefix}/x_trace.f32le"),
    )?)?;
    let x_xim = if manifest.xim_trace.is_some() {
        Some(decode_f32_vec(read_entry_bytes(
            archive,
            &format!("{prefix}/x_xim.f32le"),
        )?)?)
    } else {
        None
    };

    for (name, len) in [
        ("feature_id", feature_id.len()),
        ("precursor_id", precursor_id.len()),
        ("run_id", run_id.len()),
        ("exp_rt", exp_rt.len()),
        ("rt_left_width", rt_left_width.len()),
        ("rt_right_width", rt_right_width.len()),
        ("exp_im", exp_im.len()),
        ("exp_im_left_width", exp_im_left_width.len()),
        ("exp_im_right_width", exp_im_right_width.len()),
        ("is_decoy", is_decoy.len()),
    ] {
        if len != rows_len {
            bail!(
                "preprocessed chunk {} field {} has {} rows but manifest expected {}",
                chunk_meta.index,
                name,
                len,
                rows_len
            );
        }
    }
    if features.len() != rows_len * feat_dim {
        bail!(
            "preprocessed chunk {} feature payload has {} floats but expected {}",
            chunk_meta.index,
            features.len(),
            rows_len * feat_dim
        );
    }
    let trace_expected = rows_len * manifest.trace.total_c() * manifest.trace.l;
    if x_trace.len() != trace_expected {
        bail!(
            "preprocessed chunk {} XIC payload has {} floats but expected {}",
            chunk_meta.index,
            x_trace.len(),
            trace_expected
        );
    }
    if let (Some(xim_cfg), Some(buf)) = (manifest.xim_trace.as_ref(), x_xim.as_ref()) {
        let expected = rows_len * xim_cfg.total_c() * xim_cfg.l;
        if buf.len() != expected {
            bail!(
                "preprocessed chunk {} XIM payload has {} floats but expected {}",
                chunk_meta.index,
                buf.len(),
                expected
            );
        }
    }

    let mut rows = Vec::with_capacity(rows_len);
    for i in 0..rows_len {
        let f_start = i * feat_dim;
        let f_end = f_start + feat_dim;
        rows.push(FeatureRow {
            feature_id: feature_id[i],
            precursor_id: precursor_id[i],
            run_id: run_id[i],
            group_id: format!("{}_{}", run_id[i], precursor_id[i]),
            exp_rt: exp_rt[i],
            rt_left_width: nan_to_none(rt_left_width[i]),
            rt_right_width: nan_to_none(rt_right_width[i]),
            exp_im: nan_to_none(exp_im[i]),
            exp_im_left_width: nan_to_none(exp_im_left_width[i]),
            exp_im_right_width: nan_to_none(exp_im_right_width[i]),
            is_decoy: is_decoy[i] != 0,
            features: features[f_start..f_end].to_vec(),
        });
    }

    Ok(PreprocessedChunk {
        rows,
        x_trace,
        x_xim,
    })
}

fn read_entry_bytes(archive: &mut ZipArchive<File>, name: &str) -> Result<Vec<u8>> {
    let mut entry = archive
        .by_name(name)
        .with_context(|| format!("preprocessed archive missing entry {name}"))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn nan_to_none(v: f32) -> Option<f32> {
    if v.is_nan() { None } else { Some(v) }
}

fn encode_u64_slice(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_f32_slice(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn decode_u64_vec(bytes: Vec<u8>) -> Result<Vec<u64>> {
    if bytes.len() % 8 != 0 {
        bail!("invalid u64 payload length {}", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        out.push(u64::from_le_bytes(buf));
    }
    Ok(out)
}

fn decode_f32_vec(bytes: Vec<u8>) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("invalid f32 payload length {}", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(buf));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rows() -> Vec<FeatureRow> {
        vec![
            FeatureRow {
                feature_id: 1,
                precursor_id: 10,
                run_id: 100,
                group_id: "100_10".to_string(),
                exp_rt: 42.0,
                rt_left_width: Some(41.0),
                rt_right_width: Some(43.0),
                exp_im: Some(0.91),
                exp_im_left_width: Some(0.88),
                exp_im_right_width: Some(0.94),
                is_decoy: false,
                features: vec![1.0, 2.0],
            },
            FeatureRow {
                feature_id: 2,
                precursor_id: 11,
                run_id: 100,
                group_id: "100_11".to_string(),
                exp_rt: 55.0,
                rt_left_width: None,
                rt_right_width: None,
                exp_im: None,
                exp_im_left_width: None,
                exp_im_right_width: None,
                is_decoy: true,
                features: vec![3.0, 4.0],
            },
        ]
    }

    #[test]
    fn test_preprocessed_bundle_roundtrip() -> Result<()> {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "redeem_topaz_preprocessed_test_{}.topazdata",
            std::process::id()
        ));
        let manifest = PreprocessedManifest::new(
            2,
            vec!["a".to_string(), "b".to_string()],
            TraceBuildConfig {
                l: 2,
                ms1_cmax: 0,
                ms2_cmax: 2,
                normalize_max: false,
            },
            Some(TraceBuildConfig {
                l: 3,
                ms1_cmax: 1,
                ms2_cmax: 2,
                normalize_max: true,
            }),
            OswReadConfig::default(),
            PreprocessedProvenance::default(),
        );
        let rows = sample_rows();
        let x_trace = vec![0.0; rows.len() * 4];
        let x_xim = vec![1.0; rows.len() * 9];
        let mut writer = PreprocessedBundleWriter::new(&path, manifest)?;
        writer.write_chunk(&rows, &x_trace, Some(&x_xim))?;
        writer.finish()?;

        let reader = PreprocessedBundleReader::open(&path)?;
        assert_eq!(reader.manifest().row_count, 2);
        assert_eq!(reader.manifest().chunks.len(), 1);
        let chunk = reader.load_chunk(0)?;
        assert_eq!(chunk.rows.len(), 2);
        assert_eq!(chunk.rows[0].feature_id, 1);
        assert_eq!(chunk.rows[1].is_decoy, true);
        assert_eq!(chunk.x_trace.len(), x_trace.len());
        assert_eq!(chunk.x_xim.as_ref().map(|v| v.len()), Some(x_xim.len()));

        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
