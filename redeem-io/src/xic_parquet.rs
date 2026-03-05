// redeem-io/src/xic_parquet.rs

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

use crate::xic::PrecursorXic;

#[cfg(feature = "parquet")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "parquet")]
use std::fs::File;
#[cfg(feature = "parquet")]
use std::io::Read;

#[cfg(feature = "parquet")]
use crate::msnumpress;
#[cfg(feature = "parquet")]
use crate::xic::{TransitionTrace, XicPoint, XicSource};
#[cfg(feature = "parquet")]
use parquet::file::reader::{FileReader, SerializedFileReader};
#[cfg(feature = "parquet")]
use parquet::record::{Row, RowAccessor};
#[cfg(feature = "parquet")]
use flate2::read::ZlibDecoder;

#[derive(Debug, Clone)]
pub struct XicParquetReader {
    path: PathBuf,
    #[cfg(feature = "parquet")]
    filters: XicParquetFilters,
}

impl XicParquetReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            #[cfg(feature = "parquet")]
            filters: XicParquetFilters::default(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn read_precursors(&self, _run_id: u64, _precursor_ids: &[u64]) -> Result<Vec<PrecursorXic>> {
        bail!("XIC parquet reader not implemented yet (enable feature `parquet` for full support)")
    }
}

#[cfg(feature = "parquet")]
pub fn list_run_ids(path: &Path) -> Result<Vec<u64>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let col_idx = XicParquetReader::build_index(&reader)?;
    let idx_run = XicParquetReader::ensure_idx(&col_idx, "RUN_ID")?;
    let mut set: HashSet<u64> = HashSet::new();
    let mut iter = reader.get_row_iter(None)?;
    while let Some(row) = iter.next() {
        let row = row?;
        let run = XicParquetReader::get_i64(&row, idx_run, "RUN_ID")? as u64;
        set.insert(run);
    }
    let mut out: Vec<u64> = set.into_iter().collect();
    out.sort_unstable();
    Ok(out)
}

#[cfg(not(feature = "parquet"))]
pub fn list_run_ids(_path: &Path) -> Result<Vec<u64>> {
    bail!("XIC parquet reader not available (enable feature `parquet`)")
}

#[cfg(feature = "parquet")]
#[derive(Debug, Clone, Default)]
struct XicParquetFilters {
    run_id: Option<u64>,
    precursor_ids: Option<HashSet<u64>>,
    ms_levels: Option<HashSet<i64>>,
    detecting_transition: Option<i64>,
    decoy: Option<i64>,
}

#[cfg(feature = "parquet")]
impl XicParquetReader {
    pub fn filter_run_id(&mut self, run_id: u64) -> &mut Self {
        self.filters.run_id = Some(run_id);
        self
    }

    pub fn filter_precursor_id<I, T>(&mut self, precursor_ids: I) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Into<u64>,
    {
        let set = precursor_ids.into_iter().map(|v| v.into()).collect();
        self.filters.precursor_ids = Some(set);
        self
    }

    pub fn filter_ms_level<I, T>(&mut self, ms_levels: I) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Into<i64>,
    {
        let set = ms_levels.into_iter().map(|v| v.into()).collect();
        self.filters.ms_levels = Some(set);
        self
    }

    pub fn filter_detecting_transition(&mut self, flag: impl Into<i64>) -> &mut Self {
        self.filters.detecting_transition = Some(flag.into());
        self
    }

    pub fn filter_decoy(&mut self, flag: impl Into<i64>) -> &mut Self {
        self.filters.decoy = Some(flag.into());
        self
    }

    pub fn clear_filters(&mut self) -> &mut Self {
        self.filters = XicParquetFilters::default();
        self
    }

    pub fn fetch(&mut self) -> Result<Vec<PrecursorXic>> {
        if self.filters.run_id.is_none() && self.filters.precursor_ids.is_none() {
            bail!("missing filters: set run_id or precursor_ids before fetch() to avoid full scan");
        }
        self.fetch_internal(self.filters.run_id, self.filters.precursor_ids.clone())
    }

    fn decode_raw_doubles(data: &[u8]) -> Result<Vec<f64>> {
        if data.len() % 8 != 0 {
            bail!("raw double buffer length not divisible by 8");
        }
        let mut out = Vec::with_capacity(data.len() / 8);
        for chunk in data.chunks_exact(8) {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            out.push(f64::from_le_bytes(buf));
        }
        Ok(out)
    }

    fn decode_zlib_doubles(data: &[u8]) -> Result<Vec<f64>> {
        let mut decoder = ZlibDecoder::new(data);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;
        Self::decode_raw_doubles(&buf)
    }

    fn decode_zlib_bytes(data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = ZlibDecoder::new(data);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn looks_like_zlib(data: &[u8]) -> bool {
        if data.len() < 2 {
            return false;
        }
        if data[0] != 0x78 {
            return false;
        }
        matches!(data[1], 0x01 | 0x5e | 0x9c | 0xda)
    }

    fn decode_array(data: &[u8], comp: i64) -> Result<Vec<f64>> {
        match comp {
            0 => Self::decode_raw_doubles(data),
            1 => Self::decode_zlib_doubles(data),
            5 => {
                if Self::looks_like_zlib(data) {
                    if let Ok(buf) = Self::decode_zlib_bytes(data) {
                        if let Ok(v) = msnumpress::decode_linear(&buf) {
                            return Ok(v);
                        }
                    }
                }
                msnumpress::decode_linear(data)
            }
            6 => {
                if Self::looks_like_zlib(data) {
                    if let Ok(buf) = Self::decode_zlib_bytes(data) {
                        if let Ok(v) = msnumpress::decode_slof(&buf) {
                            return Ok(v);
                        }
                    }
                }
                msnumpress::decode_slof(data)
            }
            _ => bail!("unsupported compression id {comp}"),
        }
    }

    fn build_index(reader: &SerializedFileReader<File>) -> Result<HashMap<String, usize>> {
        let schema = reader.metadata().file_metadata().schema_descr().root_schema();
        let mut map = HashMap::new();
        for (i, field) in schema.get_fields().iter().enumerate() {
            map.insert(field.name().to_string(), i);
        }
        Ok(map)
    }

    fn get_i64(row: &Row, idx: usize, name: &str) -> Result<i64> {
        row.get_long(idx).map_err(|e| anyhow::anyhow!("missing {name}: {e}"))
    }

    fn get_i64_opt(row: &Row, idx: Option<usize>) -> Option<i64> {
        idx.and_then(|i| row.get_long(i).ok())
    }

    fn get_string_opt(row: &Row, idx: Option<usize>) -> Option<String> {
        idx.and_then(|i| row.get_string(i).ok()).map(|s| s.to_string())
    }

    fn get_bytes(row: &Row, idx: usize, name: &str) -> Result<Vec<u8>> {
        let b = row.get_bytes(idx).map_err(|e| anyhow::anyhow!("missing {name}: {e}"))?;
        Ok(b.data().to_vec())
    }

    fn ensure_idx(map: &HashMap<String, usize>, name: &str) -> Result<usize> {
        map.get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing column {name}"))
    }

    fn sort_traces(traces: &mut [TransitionTrace]) {
        traces.sort_by(|a, b| {
            let ml_a = a.ms_level.unwrap_or(255);
            let ml_b = b.ms_level.unwrap_or(255);
            ml_a
                .cmp(&ml_b)
                .then_with(|| a.ordinal.cmp(&b.ordinal))
                .then_with(|| a.annotation.cmp(&b.annotation))
        });
    }

    fn intersect_precursors(
        a: Option<HashSet<u64>>,
        b: Option<HashSet<u64>>,
    ) -> Option<HashSet<u64>> {
        match (a, b) {
            (Some(left), Some(right)) => Some(
                left.intersection(&right)
                    .copied()
                    .collect::<HashSet<u64>>(),
            ),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        }
    }

    fn fetch_internal(
        &self,
        run_id: Option<u64>,
        precursor_ids: Option<HashSet<u64>>,
    ) -> Result<Vec<PrecursorXic>> {
        let file = File::open(&self.path)?;
        let reader = SerializedFileReader::new(file)?;
        let col_idx = Self::build_index(&reader)?;

        let idx_run = Self::ensure_idx(&col_idx, "RUN_ID")?;
        let idx_ms = Self::ensure_idx(&col_idx, "MS_LEVEL")?;
        let idx_prec = Self::ensure_idx(&col_idx, "PRECURSOR_ID")?;
        let idx_trans = col_idx.get("TRANSITION_ID").copied();
        let idx_ord = col_idx.get("TRANSITION_ORDINAL").copied();
        let idx_ann = col_idx.get("ANNOTATION").copied();
        let idx_rt = Self::ensure_idx(&col_idx, "RT_DATA")?;
        let idx_int = Self::ensure_idx(&col_idx, "INTENSITY_DATA")?;
        let idx_rt_c = Self::ensure_idx(&col_idx, "RT_COMPRESSION")?;
        let idx_int_c = Self::ensure_idx(&col_idx, "INTENSITY_COMPRESSION")?;

        let idx_detect = col_idx.get("DETECTING_TRANSITION").copied();
        let idx_prec_decoy = col_idx.get("PRECURSOR_DECOY").copied();
        let idx_prod_decoy = col_idx.get("PRODUCT_DECOY").copied();

        if self.filters.detecting_transition.is_some() && idx_detect.is_none() {
            bail!("DETECTING_TRANSITION filter requested but column missing");
        }
        if self.filters.decoy.is_some() && idx_prec_decoy.is_none() && idx_prod_decoy.is_none() {
            bail!("decoy filter requested but PRECURSOR_DECOY/PRODUCT_DECOY columns missing");
        }

        let mut map: HashMap<u64, Vec<TransitionTrace>> = HashMap::new();

        let mut iter = reader.get_row_iter(None)?;
        while let Some(row) = iter.next() {
            let row = row?;
            if let Some(run) = run_id {
                let row_run = Self::get_i64(&row, idx_run, "RUN_ID")? as u64;
                if row_run != run {
                    continue;
                }
            }

            let prec = Self::get_i64(&row, idx_prec, "PRECURSOR_ID")? as u64;
            if let Some(want) = &precursor_ids {
                if !want.contains(&prec) {
                    continue;
                }
            }

            let ms_level_i64 = Self::get_i64(&row, idx_ms, "MS_LEVEL")?;
            if let Some(levels) = &self.filters.ms_levels {
                if !levels.contains(&ms_level_i64) {
                    continue;
                }
            }

            if let Some(flag) = self.filters.detecting_transition {
                let detect = Self::get_i64_opt(&row, idx_detect);
                if detect != Some(flag) {
                    continue;
                }
            }

            if let Some(flag) = self.filters.decoy {
                let decoy = if ms_level_i64 == 1 {
                    Self::get_i64_opt(&row, idx_prec_decoy)
                } else {
                    Self::get_i64_opt(&row, idx_prod_decoy)
                        .or_else(|| Self::get_i64_opt(&row, idx_prec_decoy))
                };
                if decoy != Some(flag) {
                    continue;
                }
            }

            let ms_level = ms_level_i64 as u8;
            let transition_id = Self::get_i64_opt(&row, idx_trans);
            let ordinal = Self::get_i64_opt(&row, idx_ord).unwrap_or(0) as i32;
            let annotation = Self::get_string_opt(&row, idx_ann).unwrap_or_else(|| {
                if ms_level == 1 {
                    format!("Precursor_i{ordinal}")
                } else if let Some(tid) = transition_id {
                    format!("transition_{tid}")
                } else {
                    "trace".to_string()
                }
            });

            let rt_bytes = Self::get_bytes(&row, idx_rt, "RT_DATA")?;
            let int_bytes = Self::get_bytes(&row, idx_int, "INTENSITY_DATA")?;
            let rt_comp = Self::get_i64(&row, idx_rt_c, "RT_COMPRESSION")?;
            let int_comp = Self::get_i64(&row, idx_int_c, "INTENSITY_COMPRESSION")?;

            let rts = Self::decode_array(&rt_bytes, rt_comp)?;
            let ints = Self::decode_array(&int_bytes, int_comp)?;
            let n = rts.len().min(ints.len());

            let mut points = Vec::with_capacity(n);
            for i in 0..n {
                points.push(XicPoint { rt: rts[i] as f32, intensity: ints[i] as f32 });
            }

            let trace = TransitionTrace {
                annotation,
                ordinal,
                ms_level: Some(ms_level),
                points,
            };
            map.entry(prec).or_default().push(trace);
        }

        let mut out = Vec::new();
        if let Some(want) = &precursor_ids {
            for &pid in want.iter() {
                if let Some(mut traces) = map.remove(&pid) {
                    Self::sort_traces(&mut traces);
                    out.push(PrecursorXic { precursor_id: pid, transitions: traces });
                }
            }
        } else {
            for (pid, mut traces) in map {
                Self::sort_traces(&mut traces);
                out.push(PrecursorXic { precursor_id: pid, transitions: traces });
            }
            out.sort_by_key(|p| p.precursor_id);
        }

        Ok(out)
    }
}

#[cfg(feature = "parquet")]
impl XicSource for XicParquetReader {
    fn fetch_precursors(&mut self, run_id: u64, precursor_ids: &[u64]) -> Result<Vec<PrecursorXic>> {
        if precursor_ids.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(run_filter) = self.filters.run_id {
            if run_filter != run_id {
                return Ok(Vec::new());
            }
        }

        let requested: HashSet<u64> = precursor_ids.iter().copied().collect();
        let combined = Self::intersect_precursors(self.filters.precursor_ids.clone(), Some(requested));
        self.fetch_internal(Some(run_id), combined)
    }
}

#[cfg(all(test, feature = "parquet"))]
mod tests {
    use super::XicParquetReader;
    use anyhow::Result;
    use std::path::Path;

    #[test]
    fn xic_parquet_filter_chain_smoke() -> Result<()> {
        let path = Path::new(
            "/home/singjc/Documents/github/PASS01508_DIAlignR_Spyo/2026026_for_ptsc_model/hroest_K120808_Strep0%PlasmaBiolRepl1_R01_SW.xic",
        );
        if !path.exists() {
            eprintln!("skipping xic_parquet_filter_chain_smoke: test file not found");
            return Ok(());
        }

        let mut reader = XicParquetReader::new(path);
        let prec_id = 23549_u64;
        let precursors = reader
            .filter_precursor_id([prec_id])
            .filter_ms_level([2])
            .filter_detecting_transition(1)
            .filter_decoy(1)
            .fetch()?;

        assert!(!precursors.is_empty());
        let traces = &precursors[0].transitions;
        assert!(!traces.is_empty());
        assert!(traces.iter().all(|t| t.ms_level == Some(2)));

        Ok(())
    }
}
