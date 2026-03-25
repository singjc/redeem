//! Reader for OpenMS ion-mobilogram parquet files (`*.xim`).
//!
//! The parquet file contains one row per mobilogram trace. This reader groups
//! those rows back into one [`crate::xim::FeatureXim`] per `FEATURE_ID`.

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

use crate::xim::FeatureXim;

#[cfg(feature = "parquet")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "parquet")]
use std::fs::File;
#[cfg(feature = "parquet")]
use std::io::Read;
#[cfg(feature = "parquet")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "parquet")]
use crate::msnumpress;
#[cfg(feature = "parquet")]
use crate::xim::{MobilogramTrace, XimPoint, XimSource};
#[cfg(feature = "parquet")]
use flate2::read::ZlibDecoder;
#[cfg(feature = "parquet")]
use parquet::file::reader::{FileReader, SerializedFileReader};
#[cfg(feature = "parquet")]
use parquet::record::{Row, RowAccessor};

const MAX_XIM_DECODE_ISSUE_SAMPLES: usize = 8192;

/// Diagnostic record emitted when one mobilogram row cannot be decoded.
///
/// Each item corresponds to one parquet row that was skipped during loading.
/// The caller can aggregate these records after a train or inference run and
/// write them to a TSV for troubleshooting malformed or truncated XIM payloads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct XimDecodeIssue {
    /// XIM parquet file that contained the malformed row.
    pub xim_path: PathBuf,
    /// Run identifier stored in the parquet row.
    pub run_id: u64,
    /// OSW `FEATURE.ID` for the mobilogram row.
    pub feature_id: u64,
    /// Human-readable trace annotation, typically a transition label.
    pub annotation: String,
    /// Payload field that failed to decode, e.g. `MOBILITY_DATA`.
    pub field: String,
    /// Compression identifier stored alongside the payload.
    pub compression: i64,
    /// Decoder error message.
    pub error: String,
}

#[cfg(feature = "parquet")]
#[derive(Debug, Default)]
struct XimDecodeIssueAccumulator {
    total: usize,
    unique: HashSet<XimDecodeIssue>,
    omitted_rows: usize,
}

/// Counted summary of XIM decode issues seen so far.
///
/// `unique` retains only a bounded sample of representative issue rows to keep
/// diagnostics from growing without bound on heavily malformed inputs.
#[derive(Debug, Clone, Default)]
pub struct XimDecodeIssueSummary {
    pub total: usize,
    pub unique: std::collections::HashSet<XimDecodeIssue>,
    pub omitted_rows: usize,
}

impl XimDecodeIssueSummary {
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    pub fn merge(&mut self, other: Self) {
        self.total += other.total;
        self.omitted_rows += other.omitted_rows;
        for issue in other.unique {
            if self.unique.contains(&issue) {
                continue;
            }
            if self.unique.len() < MAX_XIM_DECODE_ISSUE_SAMPLES {
                self.unique.insert(issue);
            } else {
                self.omitted_rows += 1;
            }
        }
    }
}

#[cfg(feature = "parquet")]
static XIM_DECODE_ISSUES: OnceLock<Mutex<XimDecodeIssueAccumulator>> = OnceLock::new();

#[cfg(feature = "parquet")]
fn decode_issue_store() -> &'static Mutex<XimDecodeIssueAccumulator> {
    XIM_DECODE_ISSUES.get_or_init(|| Mutex::new(XimDecodeIssueAccumulator::default()))
}

#[cfg(feature = "parquet")]
fn record_decode_issue(issue: XimDecodeIssue) {
    if let Ok(mut issues) = decode_issue_store().lock() {
        issues.total += 1;
        if issues.unique.contains(&issue) {
            return;
        }
        if issues.unique.len() < MAX_XIM_DECODE_ISSUE_SAMPLES {
            issues.unique.insert(issue);
        } else {
            issues.omitted_rows += 1;
        }
    }
}

/// Clear the process-local buffer of skipped XIM decode records.
///
/// `redeem-topaz` calls this at the start of a train or inference run so that
/// the later summary and TSV only describe the current execution.
#[cfg(feature = "parquet")]
pub fn clear_decode_issues() {
    if let Ok(mut issues) = decode_issue_store().lock() {
        *issues = XimDecodeIssueAccumulator::default();
    }
}

/// Drain and return all skipped XIM decode records collected so far.
///
/// The returned summary preserves the total number of skipped traces while
/// storing only a bounded sample of representative diagnostic rows.
#[cfg(feature = "parquet")]
pub fn take_decode_issue_summary() -> XimDecodeIssueSummary {
    if let Ok(mut issues) = decode_issue_store().lock() {
        XimDecodeIssueSummary {
            total: std::mem::take(&mut issues.total),
            unique: std::mem::take(&mut issues.unique),
            omitted_rows: std::mem::take(&mut issues.omitted_rows),
        }
    } else {
        XimDecodeIssueSummary::default()
    }
}

/// Backward-compatible helper that returns only the sampled issue rows.
#[cfg(feature = "parquet")]
pub fn take_decode_issues() -> Vec<XimDecodeIssue> {
    take_decode_issue_summary().unique.into_iter().collect()
}

/// Stub used when `redeem-io` is built without parquet support.
#[cfg(not(feature = "parquet"))]
pub fn clear_decode_issues() {}

/// Stub used when `redeem-io` is built without parquet support.
#[cfg(not(feature = "parquet"))]
pub fn take_decode_issue_summary() -> XimDecodeIssueSummary {
    XimDecodeIssueSummary::default()
}

/// Stub used when `redeem-io` is built without parquet support.
#[cfg(not(feature = "parquet"))]
pub fn take_decode_issues() -> Vec<XimDecodeIssue> {
    Vec::new()
}

/// Builder-style reader for parquet-backed XIM data.
#[derive(Debug, Clone)]
pub struct XimParquetReader {
    path: PathBuf,
    #[cfg(feature = "parquet")]
    filters: XimParquetFilters,
}

impl XimParquetReader {
    /// Create a reader for one parquet file.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            #[cfg(feature = "parquet")]
            filters: XimParquetFilters::default(),
        }
    }

    /// Return the underlying parquet path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Convenience adapter for the [`crate::xim::XimSource`] API.
    pub fn read_features(&self, run_id: u64, feature_ids: &[u64]) -> Result<Vec<FeatureXim>> {
        #[cfg(feature = "parquet")]
        {
            if feature_ids.is_empty() {
                return Ok(Vec::new());
            }
            let requested: HashSet<u64> = feature_ids.iter().copied().collect();
            return self.fetch_internal(Some(run_id), Some(requested));
        }
        #[cfg(not(feature = "parquet"))]
        {
            let _ = (run_id, feature_ids);
            bail!(
                "XIM parquet reader not implemented yet (enable feature `parquet` for full support)"
            )
        }
    }
}

#[cfg(feature = "parquet")]
/// List all run IDs present in a parquet XIM file.
pub fn list_run_ids(path: &Path) -> Result<Vec<u64>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let col_idx = XimParquetReader::build_index(&reader)?;
    let idx_run = XimParquetReader::ensure_idx(&col_idx, "RUN_ID")?;
    let mut set: HashSet<u64> = HashSet::new();
    let mut iter = reader.get_row_iter(None)?;
    while let Some(row) = iter.next() {
        let row = row?;
        let run = XimParquetReader::get_i64(&row, idx_run, "RUN_ID")? as u64;
        set.insert(run);
    }
    let mut out: Vec<u64> = set.into_iter().collect();
    out.sort_unstable();
    Ok(out)
}

#[cfg(feature = "parquet")]
/// Read the first parquet row and return its `RUN_ID`.
///
/// This helper is meant for fast diagnostics when each `*.xim` file is
/// expected to contain mobilograms for exactly one run. It avoids scanning the
/// entire parquet file the way [`list_run_ids`] does.
pub fn first_run_id(path: &Path) -> Result<Option<u64>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let col_idx = XimParquetReader::build_index(&reader)?;
    let idx_run = XimParquetReader::ensure_idx(&col_idx, "RUN_ID")?;
    let mut iter = reader.get_row_iter(None)?;
    if let Some(row) = iter.next() {
        let row = row?;
        return Ok(Some(
            XimParquetReader::get_i64(&row, idx_run, "RUN_ID")? as u64
        ));
    }
    Ok(None)
}

#[cfg(not(feature = "parquet"))]
/// Stub used when `redeem-io` is built without parquet support.
pub fn list_run_ids(_path: &Path) -> Result<Vec<u64>> {
    bail!("XIM parquet reader not available (enable feature `parquet`)")
}

#[cfg(not(feature = "parquet"))]
/// Stub used when `redeem-io` is built without parquet support.
pub fn first_run_id(_path: &Path) -> Result<Option<u64>> {
    bail!("XIM parquet reader not available (enable feature `parquet`)")
}

#[cfg(feature = "parquet")]
#[derive(Debug, Clone, Default)]
struct XimParquetFilters {
    run_id: Option<u64>,
    feature_ids: Option<HashSet<u64>>,
    ms_levels: Option<HashSet<i64>>,
    mobilogram_types: Option<HashSet<String>>,
    detecting_transition: Option<i64>,
    decoy: Option<i64>,
}

#[cfg(feature = "parquet")]
impl XimParquetReader {
    /// Restrict subsequent fetches to a single run.
    pub fn filter_run_id(&mut self, run_id: u64) -> &mut Self {
        self.filters.run_id = Some(run_id);
        self
    }

    /// Restrict subsequent fetches to a set of feature IDs.
    pub fn filter_feature_id<I, T>(&mut self, feature_ids: I) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Into<u64>,
    {
        let set = feature_ids.into_iter().map(|v| v.into()).collect();
        self.filters.feature_ids = Some(set);
        self
    }

    /// Restrict subsequent fetches to one or more MS levels.
    pub fn filter_ms_level<I, T>(&mut self, ms_levels: I) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Into<i64>,
    {
        let set = ms_levels.into_iter().map(|v| v.into()).collect();
        self.filters.ms_levels = Some(set);
        self
    }

    /// Restrict subsequent fetches to one or more mobilogram types.
    pub fn filter_mobilogram_type<I, S>(&mut self, types: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let set = types
            .into_iter()
            .map(|s| s.as_ref().to_lowercase())
            .collect();
        self.filters.mobilogram_types = Some(set);
        self
    }

    /// Restrict subsequent fetches to rows with a specific
    /// `DETECTING_TRANSITION` flag.
    pub fn filter_detecting_transition(&mut self, flag: impl Into<i64>) -> &mut Self {
        self.filters.detecting_transition = Some(flag.into());
        self
    }

    /// Restrict subsequent fetches to a specific decoy flag.
    pub fn filter_decoy(&mut self, flag: impl Into<i64>) -> &mut Self {
        self.filters.decoy = Some(flag.into());
        self
    }

    /// Clear all accumulated filters.
    pub fn clear_filters(&mut self) -> &mut Self {
        self.filters = XimParquetFilters::default();
        self
    }

    /// Execute the filtered parquet scan and return grouped feature mobilograms.
    pub fn fetch(&mut self) -> Result<Vec<FeatureXim>> {
        if self.filters.run_id.is_none() && self.filters.feature_ids.is_none() {
            bail!("missing filters: set run_id or feature_ids before fetch() to avoid full scan");
        }
        self.fetch_internal(self.filters.run_id, self.filters.feature_ids.clone())
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
        let schema = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .root_schema();
        let mut map = HashMap::new();
        for (i, field) in schema.get_fields().iter().enumerate() {
            map.insert(field.name().to_string(), i);
        }
        Ok(map)
    }

    fn get_i64(row: &Row, idx: usize, name: &str) -> Result<i64> {
        row.get_long(idx)
            .map_err(|e| anyhow::anyhow!("missing {name}: {e}"))
    }

    fn get_i64_opt(row: &Row, idx: Option<usize>) -> Option<i64> {
        idx.and_then(|i| row.get_long(i).ok())
    }

    fn get_f64(row: &Row, idx: usize, name: &str) -> Result<f64> {
        row.get_double(idx)
            .map_err(|e| anyhow::anyhow!("missing {name}: {e}"))
    }

    fn get_string_opt(row: &Row, idx: Option<usize>) -> Option<String> {
        idx.and_then(|i| row.get_string(i).ok())
            .map(|s| s.to_string())
    }

    fn get_bytes(row: &Row, idx: usize, name: &str) -> Result<Vec<u8>> {
        let b = row
            .get_bytes(idx)
            .map_err(|e| anyhow::anyhow!("missing {name}: {e}"))?;
        Ok(b.data().to_vec())
    }

    fn ensure_idx(map: &HashMap<String, usize>, name: &str) -> Result<usize> {
        map.get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing column {name}"))
    }

    fn sort_traces(traces: &mut [MobilogramTrace]) {
        traces.sort_by(|a, b| {
            let ty_a = a.mobilogram_type.as_deref().unwrap_or("zz");
            let ty_b = b.mobilogram_type.as_deref().unwrap_or("zz");
            let ml_a = a.ms_level.unwrap_or(255);
            let ml_b = b.ms_level.unwrap_or(255);
            ty_a.cmp(ty_b)
                .then_with(|| ml_a.cmp(&ml_b))
                .then_with(|| a.ordinal.cmp(&b.ordinal))
                .then_with(|| a.annotation.cmp(&b.annotation))
        });
    }

    fn intersect_features(
        a: Option<HashSet<u64>>,
        b: Option<HashSet<u64>>,
    ) -> Option<HashSet<u64>> {
        match (a, b) {
            (Some(left), Some(right)) => {
                Some(left.intersection(&right).copied().collect::<HashSet<u64>>())
            }
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        }
    }

    fn fetch_internal(
        &self,
        run_id: Option<u64>,
        feature_ids: Option<HashSet<u64>>,
    ) -> Result<Vec<FeatureXim>> {
        let file = File::open(&self.path)?;
        let reader = SerializedFileReader::new(file)?;
        let col_idx = Self::build_index(&reader)?;

        let idx_run = Self::ensure_idx(&col_idx, "RUN_ID")?;
        let idx_ms = Self::ensure_idx(&col_idx, "MS_LEVEL")?;
        let idx_type = col_idx.get("MOBILOGRAM_TYPE").copied();
        let idx_prec = Self::ensure_idx(&col_idx, "PRECURSOR_ID")?;
        let idx_feat = Self::ensure_idx(&col_idx, "FEATURE_ID")?;
        let idx_feat_rt = Self::ensure_idx(&col_idx, "FEATURE_RT")?;
        let idx_trans = col_idx.get("TRANSITION_ID").copied();
        let idx_ord = col_idx.get("TRANSITION_ORDINAL").copied();
        let idx_ann = col_idx.get("ANNOTATION").copied();
        let idx_mob = Self::ensure_idx(&col_idx, "MOBILITY_DATA")?;
        let idx_int = Self::ensure_idx(&col_idx, "INTENSITY_DATA")?;
        let idx_mob_c = Self::ensure_idx(&col_idx, "MOBILITY_COMPRESSION")?;
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

        // OpenMS/XIM exports occasionally leave PRECURSOR_ID null even though
        // FEATURE_ID and FEATURE_RT are present and sufficient for TOPAZ's
        // `(run_id, feature_id)` lookup path. Keep loading those rows and use
        // the first non-null precursor id seen for the feature, falling back to
        // `0` only if the entire feature group is missing that metadata.
        let mut map: HashMap<u64, (Option<u64>, f32, Vec<MobilogramTrace>)> = HashMap::new();

        let mut iter = reader.get_row_iter(None)?;
        while let Some(row) = iter.next() {
            let row = row?;
            let row_run = Self::get_i64(&row, idx_run, "RUN_ID")? as u64;
            if let Some(run) = run_id {
                if row_run != run {
                    continue;
                }
            }

            let feature_id = Self::get_i64(&row, idx_feat, "FEATURE_ID")? as u64;
            if let Some(want) = &feature_ids {
                if !want.contains(&feature_id) {
                    continue;
                }
            }

            let ms_level_i64 = Self::get_i64(&row, idx_ms, "MS_LEVEL")?;
            if let Some(levels) = &self.filters.ms_levels {
                if !levels.contains(&ms_level_i64) {
                    continue;
                }
            }

            let mobilogram_type = Self::get_string_opt(&row, idx_type);
            if let Some(types) = &self.filters.mobilogram_types {
                let ty = mobilogram_type
                    .as_deref()
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                if !types.contains(&ty) {
                    continue;
                }
            }

            if let Some(flag) = self.filters.detecting_transition {
                if ms_level_i64 != 1 {
                    let detect = Self::get_i64_opt(&row, idx_detect);
                    if detect != Some(flag) {
                        continue;
                    }
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
            let precursor_id = Self::get_i64_opt(&row, Some(idx_prec)).map(|v| v as u64);
            let feature_rt = Self::get_f64(&row, idx_feat_rt, "FEATURE_RT")? as f32;
            let transition_id = Self::get_i64_opt(&row, idx_trans);
            let ordinal = Self::get_i64_opt(&row, idx_ord).unwrap_or(0) as i32;
            let annotation = Self::get_string_opt(&row, idx_ann).unwrap_or_else(|| {
                if ms_level == 1 {
                    format!("xim_ms1_{ordinal}")
                } else if let Some(tid) = transition_id {
                    format!("xim_transition_{tid}")
                } else {
                    "xim_trace".to_string()
                }
            });
            let mob_bytes = Self::get_bytes(&row, idx_mob, "MOBILITY_DATA")?;
            let int_bytes = Self::get_bytes(&row, idx_int, "INTENSITY_DATA")?;
            let mob_comp = Self::get_i64(&row, idx_mob_c, "MOBILITY_COMPRESSION")?;
            let int_comp = Self::get_i64(&row, idx_int_c, "INTENSITY_COMPRESSION")?;

            let mobs = match Self::decode_array(&mob_bytes, mob_comp) {
                Ok(v) => v,
                Err(err) => {
                    let issue = XimDecodeIssue {
                        xim_path: self.path.clone(),
                        run_id: row_run,
                        feature_id,
                        annotation: annotation.clone(),
                        field: "MOBILITY_DATA".to_string(),
                        compression: mob_comp,
                        error: err.to_string(),
                    };
                    record_decode_issue(issue.clone());
                    log::debug!(
                        "skipping malformed XIM mobility trace: path={:?} run_id={} feature_id={} annotation={} compression={} error={}",
                        self.path,
                        row_run,
                        feature_id,
                        annotation,
                        mob_comp,
                        issue.error
                    );
                    continue;
                }
            };
            let ints = match Self::decode_array(&int_bytes, int_comp) {
                Ok(v) => v,
                Err(err) => {
                    let issue = XimDecodeIssue {
                        xim_path: self.path.clone(),
                        run_id: row_run,
                        feature_id,
                        annotation: annotation.clone(),
                        field: "INTENSITY_DATA".to_string(),
                        compression: int_comp,
                        error: err.to_string(),
                    };
                    record_decode_issue(issue.clone());
                    log::debug!(
                        "skipping malformed XIM intensity trace: path={:?} run_id={} feature_id={} annotation={} compression={} error={}",
                        self.path,
                        row_run,
                        feature_id,
                        annotation,
                        int_comp,
                        issue.error
                    );
                    continue;
                }
            };
            let n = mobs.len().min(ints.len());

            let mut points = Vec::with_capacity(n);
            for i in 0..n {
                points.push(XimPoint {
                    mobility: mobs[i] as f32,
                    intensity: ints[i] as f32,
                });
            }

            let trace = MobilogramTrace {
                annotation,
                ordinal,
                ms_level: Some(ms_level),
                mobilogram_type,
                points,
            };
            let entry = map
                .entry(feature_id)
                .or_insert_with(|| (precursor_id, feature_rt, Vec::new()));
            if entry.0.is_none() && precursor_id.is_some() {
                entry.0 = precursor_id;
            }
            entry.2.push(trace);
        }

        let mut out = Vec::new();
        if let Some(want) = &feature_ids {
            for &fid in want.iter() {
                if let Some((precursor_id, feature_rt, mut traces)) = map.remove(&fid) {
                    Self::sort_traces(&mut traces);
                    out.push(FeatureXim {
                        feature_id: fid,
                        precursor_id: precursor_id.unwrap_or(0),
                        feature_rt,
                        traces,
                    });
                }
            }
        } else {
            for (feature_id, (precursor_id, feature_rt, mut traces)) in map {
                Self::sort_traces(&mut traces);
                out.push(FeatureXim {
                    feature_id,
                    precursor_id: precursor_id.unwrap_or(0),
                    feature_rt,
                    traces,
                });
            }
            out.sort_by_key(|f| f.feature_id);
        }

        Ok(out)
    }
}

#[cfg(feature = "parquet")]
impl XimSource for XimParquetReader {
    fn fetch_features(&mut self, run_id: u64, feature_ids: &[u64]) -> Result<Vec<FeatureXim>> {
        if feature_ids.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(run_filter) = self.filters.run_id {
            if run_filter != run_id {
                return Ok(Vec::new());
            }
        }

        let requested: HashSet<u64> = feature_ids.iter().copied().collect();
        let combined = Self::intersect_features(self.filters.feature_ids.clone(), Some(requested));
        self.fetch_internal(Some(run_id), combined)
    }
}
