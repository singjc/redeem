use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(any(feature = "io-sqlite", feature = "io-parquet"))]
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "io-parquet")]
use std::hash::{Hash, Hasher};
#[cfg(feature = "io-parquet")]
use std::io::Write;
#[cfg(feature = "io-parquet")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "io-parquet")]
use filetime::{FileTime, set_file_mtime};

use candle_core::{Device, Tensor};

use crate::building_blocks::bagging::make_bags_with_traces;
use crate::building_blocks::trace_window::extract_trace_tensor_centered;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::infer::{build_score_table_from_rows, ScoreTableRow};
use crate::infer::score_candidates;
use crate::io::osw::FeatureRow;
#[cfg(feature = "io-sqlite")]
use crate::io::osw::OswFeatureTable;
use crate::io::xic::{PrecursorXic, TransitionTrace, XicSource};
use crate::model_interface::{BagRankerWithHiddenInterface, CandidateScorerInterface};
use crate::preprocess::Preprocessor;
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
use crate::model::topaz::TopazConfig;

static WARNED_MISSING_MS1: AtomicBool = AtomicBool::new(false);

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceBuildConfig {
    pub l: usize,
    pub ms1_cmax: usize,
    pub ms2_cmax: usize,
    pub normalize_max: bool,
}

#[derive(Debug, Clone)]
pub struct BagScoreOutput {
    pub bag_score: Vec<f32>,
    pub bag_y: Vec<f32>,
    pub is_decoy: Vec<bool>,
    pub bag_pid: Vec<String>,
    pub winner_hidden: Vec<f32>,
    pub hidden_dim: usize,
}

#[derive(Debug, Clone)]
pub struct BagHeadOutput {
    pub bag_score: Vec<f32>,
    pub bag_y: Vec<f32>,
    pub is_decoy: Vec<bool>,
    pub bag_pid: Vec<String>,
    pub winner_hidden: Vec<f32>,
    pub hidden_dim: usize,
    pub emb_ms2: Vec<f32>,
    pub emb_ms2_dim: usize,
    pub emb_ms1: Vec<f32>,
    pub emb_ms1_dim: usize,
    pub emb_all: Vec<f32>,
    pub emb_all_dim: usize,
    pub coe_ms2: Vec<f32>,
    pub coe_ms2_dim: usize,
    pub coe_ms1: Vec<f32>,
    pub coe_ms1_dim: usize,
    pub coe_ms12: Vec<f32>,
    pub coe_ms12_dim: usize,
    pub coe_all: Vec<f32>,
    pub coe_all_dim: usize,
}

impl TraceBuildConfig {
    pub fn total_c(&self) -> usize {
        self.ms1_cmax + self.ms2_cmax
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XicFetchConfig {
    pub ms_levels: Option<Vec<i64>>,
    pub detecting_transition: Option<i64>,
    pub decoy: Option<i64>,
}

impl Default for XicFetchConfig {
    fn default() -> Self {
        Self {
            ms_levels: None,
            detecting_transition: Some(1),
            decoy: None,
        }
    }
}

#[cfg(feature = "io-parquet")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    run_id: u64,
    precursor_id: u64,
}

#[cfg(feature = "io-parquet")]
#[derive(Debug, Default)]
pub struct XicCache {
    capacity: usize,
    size: usize,
    data: HashMap<std::path::PathBuf, HashMap<CacheKey, PrecursorXic>>,
    order: VecDeque<(std::path::PathBuf, CacheKey)>,
}

#[cfg(feature = "io-parquet")]
impl XicCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            size: 0,
            data: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    fn get(&mut self, path: &Path, key: CacheKey) -> Option<PrecursorXic> {
        let map = self.data.get(path)?;
        let hit = map.get(&key)?.clone();
        self.order.push_back((path.to_path_buf(), key));
        Some(hit)
    }

    fn insert(&mut self, path: &Path, key: CacheKey, xic: PrecursorXic) {
        if !self.is_enabled() {
            return;
        }
        let entry = self.data.entry(path.to_path_buf()).or_default();
        let existed = entry.insert(key, xic).is_some();
        if !existed {
            self.size += 1;
        }
        self.order.push_back((path.to_path_buf(), key));
        self.evict();
    }

    fn evict(&mut self) {
        while self.size > self.capacity {
            let Some((path, key)) = self.order.pop_front() else { break; };
            if let Some(map) = self.data.get_mut(&path) {
                if map.remove(&key).is_some() {
                    self.size = self.size.saturating_sub(1);
                }
                if map.is_empty() {
                    self.data.remove(&path);
                }
            }
        }
    }
}

#[cfg(feature = "io-parquet")]
#[derive(Debug, Clone, Default)]
pub struct XicCacheStats(Arc<CacheStatsInner>);

#[cfg(feature = "io-parquet")]
#[derive(Debug, Default)]
struct CacheStatsInner {
    mem_hits: std::sync::atomic::AtomicU64,
    disk_hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    stores: std::sync::atomic::AtomicU64,
    evictions: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "io-parquet")]
impl XicCacheStats {
    pub fn new() -> Self {
        Self(Arc::new(CacheStatsInner::default()))
    }

    fn inc_mem_hit(&self, n: u64) {
        self.0.mem_hits.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn inc_disk_hit(&self, n: u64) {
        self.0.disk_hits.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn inc_miss(&self, n: u64) {
        self.0.misses.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn inc_store(&self, n: u64) {
        self.0.stores.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn inc_eviction(&self, n: u64) {
        self.0.evictions.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.0.mem_hits.load(std::sync::atomic::Ordering::Relaxed),
            self.0.disk_hits.load(std::sync::atomic::Ordering::Relaxed),
            self.0.misses.load(std::sync::atomic::Ordering::Relaxed),
            self.0.stores.load(std::sync::atomic::Ordering::Relaxed),
            self.0.evictions.load(std::sync::atomic::Ordering::Relaxed),
        )
    }
}

#[cfg(feature = "io-parquet")]
#[derive(Clone)]
pub struct SharedXicCache {
    cache: Arc<Mutex<XicCache>>,
    stats: XicCacheStats,
}

#[cfg(feature = "io-parquet")]
impl Default for SharedXicCache {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(feature = "io-parquet")]
impl SharedXicCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(XicCache::new(capacity))),
            stats: XicCacheStats::new(),
        }
    }

    pub fn stats(&self) -> XicCacheStats {
        self.stats.clone()
    }

    pub fn is_enabled(&self) -> bool {
        self.cache.lock().map(|c| c.is_enabled()).unwrap_or(false)
    }

    pub(crate) fn get(&self, path: &Path, key: CacheKey) -> Option<PrecursorXic> {
        let mut guard = self.cache.lock().ok()?;
        guard.get(path, key)
    }

    pub(crate) fn insert(&self, path: &Path, key: CacheKey, xic: PrecursorXic) {
        if let Ok(mut guard) = self.cache.lock() {
            guard.insert(path, key, xic);
        }
    }
}

#[cfg(feature = "io-parquet")]
#[derive(Debug, Clone)]
pub struct XicDiskCache {
    root: std::path::PathBuf,
    max_bytes: Option<u64>,
    stats: XicCacheStats,
}

#[cfg(feature = "io-parquet")]
impl XicDiskCache {
    pub fn new(root: std::path::PathBuf, max_bytes: Option<u64>, stats: XicCacheStats) -> Result<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self { root, max_bytes, stats })
    }

    fn hash_path(path: &Path) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        path.to_string_lossy().hash(&mut hasher);
        hasher.finish()
    }

    fn dir_for(&self, xic_path: &Path) -> std::path::PathBuf {
        let h = Self::hash_path(xic_path);
        self.root.join(format!("{:016x}", h))
    }

    fn file_for(&self, xic_path: &Path, run_id: u64, precursor_id: u64) -> std::path::PathBuf {
        let dir = self.dir_for(xic_path);
        dir.join(format!("run{}_prec{}.bin", run_id, precursor_id))
    }

    pub fn load(
        &self,
        xic_path: &Path,
        run_id: u64,
        precursor_id: u64,
    ) -> Result<Option<PrecursorXic>> {
        let path = self.file_for(xic_path, run_id, precursor_id);
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read(&path)?;
        let (got_run, xic) = decode_precursor_xic(&data)?;
        if got_run != run_id || xic.precursor_id != precursor_id {
            return Ok(None);
        }
        let _ = set_file_mtime(&path, FileTime::now());
        Ok(Some(xic))
    }

    pub fn store(
        &self,
        xic_path: &Path,
        run_id: u64,
        xic: &PrecursorXic,
    ) -> Result<()> {
        let path = self.file_for(xic_path, run_id, xic.precursor_id);
        if path.exists() {
            return Ok(());
        }
        let dir = path.parent().unwrap_or(&self.root);
        std::fs::create_dir_all(dir)?;
        let bytes = encode_precursor_xic(run_id, xic);
        let tmp = path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        if std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        let _ = set_file_mtime(&path, FileTime::now());
        self.stats.inc_store(1);
        self.enforce_capacity()?;
        Ok(())
    }

    fn enforce_capacity(&self) -> Result<()> {
        let Some(max_bytes) = self.max_bytes else {
            return Ok(());
        };
        if max_bytes == 0 {
            return Ok(());
        }
        let mut files = Vec::new();
        let mut total = 0u64;
        self.collect_files(&self.root, &mut files, &mut total)?;
        if total <= max_bytes {
            return Ok(());
        }
        files.sort_by_key(|e| e.modified);
        for entry in files {
            if total <= max_bytes {
                break;
            }
            if std::fs::remove_file(&entry.path).is_ok() {
                total = total.saturating_sub(entry.size);
                self.stats.inc_eviction(1);
            }
        }
        Ok(())
    }

    fn collect_files(
        &self,
        dir: &Path,
        files: &mut Vec<DiskEntry>,
        total: &mut u64,
    ) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let meta = entry.metadata()?;
            if meta.is_dir() {
                self.collect_files(&path, files, total)?;
            } else if meta.is_file() {
                let size = meta.len();
                let modified = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                *total += size;
                files.push(DiskEntry { path, size, modified });
            }
        }
        Ok(())
    }
}

#[cfg(feature = "io-parquet")]
#[derive(Debug)]
struct DiskEntry {
    path: std::path::PathBuf,
    size: u64,
    modified: u64,
}

#[cfg(feature = "io-parquet")]
fn encode_precursor_xic(run_id: u64, xic: &PrecursorXic) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RDXC");
    buf.push(1u8);
    buf.extend_from_slice(&run_id.to_le_bytes());
    buf.extend_from_slice(&xic.precursor_id.to_le_bytes());
    let n_tr = xic.transitions.len() as u32;
    buf.extend_from_slice(&n_tr.to_le_bytes());
    for t in &xic.transitions {
        let ann = t.annotation.as_bytes();
        buf.extend_from_slice(&(ann.len() as u32).to_le_bytes());
        buf.extend_from_slice(ann);
        buf.extend_from_slice(&t.ordinal.to_le_bytes());
        buf.push(t.ms_level.unwrap_or(255));
        let n_pt = t.points.len() as u32;
        buf.extend_from_slice(&n_pt.to_le_bytes());
        for p in &t.points {
            buf.extend_from_slice(&p.rt.to_le_bytes());
            buf.extend_from_slice(&p.intensity.to_le_bytes());
        }
    }
    buf
}

#[cfg(feature = "io-parquet")]
fn decode_precursor_xic(data: &[u8]) -> Result<(u64, PrecursorXic)> {
    let mut i = 0usize;
    if data.len() < 5 || &data[..4] != b"RDXC" {
        anyhow::bail!("invalid xic cache header");
    }
    i += 4;
    let _ver = data[i];
    i += 1;
    let run_id = read_u64(data, &mut i)?;
    let precursor_id = read_u64(data, &mut i)?;
    let n_tr = read_u32(data, &mut i)? as usize;
    let mut transitions = Vec::with_capacity(n_tr);
    for _ in 0..n_tr {
        let ann_len = read_u32(data, &mut i)? as usize;
        if i + ann_len > data.len() {
            anyhow::bail!("invalid xic cache (annotation)");
        }
        let ann = std::str::from_utf8(&data[i..i + ann_len])?.to_string();
        i += ann_len;
        let ordinal = read_i32(data, &mut i)?;
        let ms = data.get(i).copied().unwrap_or(255);
        i += 1;
        let n_pt = read_u32(data, &mut i)? as usize;
        let mut points = Vec::with_capacity(n_pt);
        for _ in 0..n_pt {
            let rt = read_f32(data, &mut i)?;
            let intensity = read_f32(data, &mut i)?;
            points.push(crate::io::xic::XicPoint { rt, intensity });
        }
        transitions.push(TransitionTrace {
            annotation: ann,
            ordinal,
            ms_level: if ms == 255 { None } else { Some(ms) },
            points,
        });
    }
    Ok((run_id, PrecursorXic { precursor_id, transitions }))
}

#[cfg(feature = "io-parquet")]
fn read_u32(data: &[u8], i: &mut usize) -> Result<u32> {
    if *i + 4 > data.len() {
        anyhow::bail!("invalid xic cache (u32)");
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[*i..*i + 4]);
    *i += 4;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(feature = "io-parquet")]
fn read_i32(data: &[u8], i: &mut usize) -> Result<i32> {
    if *i + 4 > data.len() {
        anyhow::bail!("invalid xic cache (i32)");
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[*i..*i + 4]);
    *i += 4;
    Ok(i32::from_le_bytes(buf))
}

#[cfg(feature = "io-parquet")]
fn read_u64(data: &[u8], i: &mut usize) -> Result<u64> {
    if *i + 8 > data.len() {
        anyhow::bail!("invalid xic cache (u64)");
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[*i..*i + 8]);
    *i += 8;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(feature = "io-parquet")]
fn read_f32(data: &[u8], i: &mut usize) -> Result<f32> {
    if *i + 4 > data.len() {
        anyhow::bail!("invalid xic cache (f32)");
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[*i..*i + 4]);
    *i += 4;
    Ok(f32::from_le_bytes(buf))
}

/// Build a dense (N, D) feature matrix from OSW feature rows.
pub fn rows_to_feature_matrix(rows: &[FeatureRow], feat_dim: usize) -> Vec<f32> {
    let n = rows.len();
    let mut out = vec![0f32; n * feat_dim];
    if feat_dim == 0 {
        return out;
    }
    for (i, row) in rows.iter().enumerate() {
        let take = row.features.len().min(feat_dim);
        let dst = i * feat_dim;
        if take > 0 {
            out[dst..dst + take].copy_from_slice(&row.features[..take]);
        }
    }
    out
}

/// Build a dense (N, D) feature matrix with optional preprocessing.
pub fn rows_to_feature_matrix_preprocessed(
    rows: &[FeatureRow],
    feat_dim: usize,
    pre: Option<&Preprocessor>,
) -> Vec<f32> {
    let mut x = rows_to_feature_matrix(rows, feat_dim);
    if let Some(p) = pre {
        p.transform_in_place(&mut x, rows.len(), feat_dim);
    }
    x
}

/// Build feature matrix aligned to a target column order.
/// Missing columns are filled with NaN (to be imputed by the preprocessor).
pub fn rows_to_feature_matrix_with_cols(
    rows: &[FeatureRow],
    osw_cols: &[String],
    target_cols: &[String],
    pre: Option<&Preprocessor>,
) -> Vec<f32> {
    let n = rows.len();
    let d = target_cols.len();
    let mut map: Vec<Option<usize>> = Vec::with_capacity(d);
    let mut lookup = std::collections::HashMap::new();
    for (i, name) in osw_cols.iter().enumerate() {
        lookup.insert(name.as_str(), i);
    }
    for name in target_cols {
        map.push(lookup.get(name.as_str()).copied());
    }

    let mut out = vec![f32::NAN; n * d];
    if n == 0 || d == 0 {
        return out;
    }
    for (i, row) in rows.iter().enumerate() {
        let dst = i * d;
        for (j, idx) in map.iter().enumerate() {
            if let Some(k) = idx {
                if *k < row.features.len() {
                    out[dst + j] = row.features[*k];
                }
            }
        }
    }
    if let Some(p) = pre {
        p.transform_in_place(&mut out, n, d);
    }
    out
}

fn sort_series(series: &mut [TransitionTrace]) {
    series.sort_by(|a, b| {
        a.ordinal
            .cmp(&b.ordinal)
            .then_with(|| a.annotation.cmp(&b.annotation))
    });
}

fn split_ms1_ms2(xic: &PrecursorXic) -> (Vec<TransitionTrace>, Vec<TransitionTrace>) {
    let mut ms1 = Vec::new();
    let mut ms2 = Vec::new();
    for t in &xic.transitions {
        match t.ms_level.unwrap_or(2) {
            1 => ms1.push(t.clone()),
            _ => ms2.push(t.clone()),
        }
    }
    sort_series(&mut ms1);
    sort_series(&mut ms2);
    (ms1, ms2)
}

#[cfg(feature = "io-parquet")]
fn fetch_precursors_cached(
    path: &Path,
    run_id: u64,
    prec_set: &HashSet<u64>,
    cache: &SharedXicCache,
    disk: Option<&XicDiskCache>,
    fetch_cfg: &XicFetchConfig,
) -> Result<HashMap<u64, PrecursorXic>> {
    let mut out: HashMap<u64, PrecursorXic> = HashMap::new();
    let mut missing: Vec<u64> = Vec::new();
    let stats = cache.stats();

    for &pid in prec_set {
        let key = CacheKey { run_id, precursor_id: pid };
        if let Some(xic) = cache.get(path, key) {
            stats.inc_mem_hit(1);
            out.insert(pid, xic);
            continue;
        }
        if let Some(disk_cache) = disk {
            if let Ok(Some(xic)) = disk_cache.load(path, run_id, pid) {
                cache.insert(path, key, xic.clone());
                stats.inc_disk_hit(1);
                out.insert(pid, xic);
                continue;
            }
        }
        missing.push(pid);
    }

    if !missing.is_empty() {
        stats.inc_miss(missing.len() as u64);
        let mut reader = crate::io::xic_parquet::XicParquetReader::new(path);
        reader.filter_run_id(run_id);
        if let Some(levels) = &fetch_cfg.ms_levels {
            reader.filter_ms_level(levels.clone());
        }
        if let Some(flag) = fetch_cfg.detecting_transition {
            reader.filter_detecting_transition(flag);
        }
        if let Some(flag) = fetch_cfg.decoy {
            reader.filter_decoy(flag);
        }
        reader.filter_precursor_id(missing.iter().copied());
        let fetched = reader.fetch()?;
        let requested = missing.len();
        let fetched_count = fetched.len();
        if fetched_count == 0 {
            log::warn!(
                "XIC cache path {:?}: fetched 0 of {} precursors (run_id={})",
                path,
                requested,
                run_id
            );
        } else {
            log::info!(
                "XIC cache path {:?}: fetched {} of {} precursors (run_id={})",
                path,
                fetched_count,
                requested,
                run_id
            );
        }
        for xic in fetched {
            let pid = xic.precursor_id;
            cache.insert(path, CacheKey { run_id, precursor_id: pid }, xic.clone());
            if let Some(disk_cache) = disk {
                let _ = disk_cache.store(path, run_id, &xic);
            }
            out.insert(pid, xic);
        }
    }

    Ok(out)
}

#[cfg(feature = "io-parquet")]
fn fetch_precursors_cached_with_fallback(
    path: &Path,
    run_id: u64,
    prec_set: &HashSet<u64>,
    cache: &SharedXicCache,
    disk: Option<&XicDiskCache>,
    fetch_cfg: &XicFetchConfig,
) -> Result<HashMap<u64, PrecursorXic>> {
    let out = fetch_precursors_cached(path, run_id, prec_set, cache, disk, fetch_cfg)?;
    if !out.is_empty() || prec_set.is_empty() {
        return Ok(out);
    }

    log::warn!(
        "XIC map path {:?}: fetched 0 of {} precursors (run_id={}); retrying without RUN_ID filter",
        path,
        prec_set.len(),
        run_id
    );

    let mut reader = crate::io::xic_parquet::XicParquetReader::new(path);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    reader.filter_precursor_id(prec_set.iter().copied());

    let fetched = reader.fetch()?;
    let fetched_count = fetched.len();
    if fetched_count == 0 {
        log::warn!(
            "XIC map path {:?}: fallback fetched 0 of {} precursors (ignoring RUN_ID)",
            path,
            prec_set.len()
        );
        return Ok(out);
    }
    log::info!(
        "XIC map path {:?}: fallback fetched {} of {} precursors (ignoring RUN_ID)",
        path,
        fetched_count,
        prec_set.len()
    );

    let mut map = HashMap::new();
    for xic in fetched {
        let pid = xic.precursor_id;
        cache.insert(path, CacheKey { run_id, precursor_id: pid }, xic.clone());
        if let Some(disk_cache) = disk {
            let _ = disk_cache.store(path, run_id, &xic);
        }
        map.insert(pid, xic);
    }
    Ok(map)
}

/// Build trace tensors for rows using an arbitrary XIC source.
///
/// Output layout: (N, C_total, L) flattened row-major.
pub fn build_trace_tensors_from_source(
    rows: &[FeatureRow],
    xic_source: &mut impl XicSource,
    cfg: &TraceBuildConfig,
) -> Result<Vec<f32>> {
    let n = rows.len();
    let c_total = cfg.total_c();
    let mut out = vec![0f32; n * c_total * cfg.l];
    if n == 0 || c_total == 0 || cfg.l == 0 {
        return Ok(out);
    }

    let mut by_run: HashMap<u64, HashSet<u64>> = HashMap::new();
    for row in rows {
        by_run
            .entry(row.run_id)
            .or_default()
            .insert(row.precursor_id);
    }

    let mut xic_by_run: HashMap<u64, HashMap<u64, PrecursorXic>> = HashMap::new();
    for (run_id, prec_set) in by_run {
        let precs: Vec<u64> = prec_set.into_iter().collect();
        if precs.is_empty() {
            continue;
        }
        let fetched = xic_source.fetch_precursors(run_id, &precs)?;
        let mut map: HashMap<u64, PrecursorXic> = HashMap::new();
        for xic in fetched {
            map.insert(xic.precursor_id, xic);
        }
        xic_by_run.insert(run_id, map);
    }

    let row_len = c_total * cfg.l;
    let fill_row = |row: &FeatureRow, dst: &mut [f32]| {
        if let Some(run_map) = xic_by_run.get(&row.run_id) {
            if let Some(xic) = run_map.get(&row.precursor_id) {
                let (ms1_series, ms2_series) = split_ms1_ms2(xic);

                let mut offset = 0usize;
                if cfg.ms1_cmax > 0 {
                    if ms1_series.is_empty()
                        && !WARNED_MISSING_MS1.swap(true, Ordering::Relaxed)
                    {
                        log::warn!(
                            "missing MS1 traces for at least one precursor; padding zeros"
                        );
                    }
                    let t_ms1 = extract_trace_tensor_centered(
                        &ms1_series,
                        row.exp_rt,
                        cfg.l,
                        cfg.ms1_cmax,
                        cfg.normalize_max,
                    );
                    dst[offset..offset + cfg.ms1_cmax * cfg.l].copy_from_slice(&t_ms1);
                    offset += cfg.ms1_cmax * cfg.l;
                }
                let t_ms2 = extract_trace_tensor_centered(
                    &ms2_series,
                    row.exp_rt,
                    cfg.l,
                    cfg.ms2_cmax,
                    cfg.normalize_max,
                );
                dst[offset..offset + cfg.ms2_cmax * cfg.l].copy_from_slice(&t_ms2);
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        out.par_chunks_mut(row_len)
            .zip(rows.par_iter())
            .for_each(|(dst, row)| fill_row(row, dst));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, row) in rows.iter().enumerate() {
            let dst = &mut out[i * row_len..(i + 1) * row_len];
            fill_row(row, dst);
        }
    }

    Ok(out)
}

/// Score candidates directly from rows + traces.
pub fn score_rows_from_rows(
    model: &impl CandidateScorerInterface,
    rows: &[FeatureRow],
    x_trace: &[f32],
    feat_dim: usize,
    c_total: usize,
    l: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<Vec<f32>> {
    let n = rows.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let x_feat = rows_to_feature_matrix_preprocessed(rows, feat_dim, pre);
    let x_feat_t = Tensor::from_vec(x_feat, (n, feat_dim), device)?;
    let x_trace_t = Tensor::from_vec(x_trace.to_vec(), (n, c_total, l), device)?;
    let scores_t = score_candidates(model, &x_feat_t, &x_trace_t, batch_size.max(1))?;
    Ok(scores_t.to_vec1::<f32>()?)
}

/// Score candidates directly from rows + traces with explicit feature columns.
pub fn score_rows_from_rows_with_cols(
    model: &impl CandidateScorerInterface,
    rows: &[FeatureRow],
    x_trace: &[f32],
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<Vec<f32>> {
    let n = rows.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let x_feat = rows_to_feature_matrix_with_cols(rows, osw_cols, target_cols, pre);
    let x_feat_t = Tensor::from_vec(x_feat, (n, target_cols.len()), device)?;
    let x_trace_t = Tensor::from_vec(x_trace.to_vec(), (n, c_total, l), device)?;
    let scores_t = score_candidates(model, &x_feat_t, &x_trace_t, batch_size.max(1))?;
    Ok(scores_t.to_vec1::<f32>()?)
}

/// Score bags from rows + traces, returning bag-level diagnostics.
pub fn score_bags_from_rows(
    model: &impl BagRankerWithHiddenInterface,
    rows: &[FeatureRow],
    x_trace: &[f32],
    feat_dim: usize,
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<BagScoreOutput> {
    let n = rows.len();
    if n == 0 {
        return Ok(BagScoreOutput {
            bag_score: Vec::new(),
            bag_y: Vec::new(),
            is_decoy: Vec::new(),
            bag_pid: Vec::new(),
            winner_hidden: Vec::new(),
            hidden_dim: 0,
        });
    }

    let x_feat = rows_to_feature_matrix_preprocessed(rows, feat_dim, pre);
    let y_rows: Vec<u8> = rows.iter().map(|r| if r.is_decoy { 1 } else { 0 }).collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat,
        n,
        feat_dim,
        x_trace,
        c_total,
        l,
        &y_rows,
        &pid_rows,
        bag_k,
    );

    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;

    let b = bags.b;
    let mut bag_scores = Vec::with_capacity(b);
    let mut hidden: Vec<f32> = Vec::new();
    let mut hidden_dim = 0usize;
    let bs = batch_size.max(1);

    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = xb.narrow(0, i, take)?;
        let tb_i = tb.narrow(0, i, take)?;
        let m_i = mask.narrow(0, i, take)?;

        let (_cand, bag, win) = model.forward_bags_with_hidden(&xb_i, &tb_i, &m_i)?;
        let bag_vec = bag.to_vec1::<f32>()?;
        bag_scores.extend(bag_vec);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.get(0).map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }
        i += take;
    }

    let is_decoy: Vec<bool> = bags.y_bag.iter().map(|&y| y < 0.5).collect();
    Ok(BagScoreOutput {
        bag_score: bag_scores,
        bag_y: bags.y_bag,
        is_decoy,
        bag_pid: bags.bag_pid,
        winner_hidden: hidden,
        hidden_dim,
    })
}

/// Score bags from rows + traces with explicit feature columns.
pub fn score_bags_from_rows_with_cols(
    model: &impl BagRankerWithHiddenInterface,
    rows: &[FeatureRow],
    x_trace: &[f32],
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<BagScoreOutput> {
    let n = rows.len();
    if n == 0 {
        return Ok(BagScoreOutput {
            bag_score: Vec::new(),
            bag_y: Vec::new(),
            is_decoy: Vec::new(),
            bag_pid: Vec::new(),
            winner_hidden: Vec::new(),
            hidden_dim: 0,
        });
    }

    let d = target_cols.len();
    let x_feat = rows_to_feature_matrix_with_cols(rows, osw_cols, target_cols, pre);
    let y_rows: Vec<u8> = rows.iter().map(|r| if r.is_decoy { 1 } else { 0 }).collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat,
        n,
        d,
        x_trace,
        c_total,
        l,
        &y_rows,
        &pid_rows,
        bag_k,
    );

    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;

    let b = bags.b;
    let mut bag_scores = Vec::with_capacity(b);
    let mut hidden: Vec<f32> = Vec::new();
    let mut hidden_dim = 0usize;
    let bs = batch_size.max(1);

    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = xb.narrow(0, i, take)?;
        let tb_i = tb.narrow(0, i, take)?;
        let m_i = mask.narrow(0, i, take)?;

        let (_cand, bag, win) = model.forward_bags_with_hidden(&xb_i, &tb_i, &m_i)?;
        let bag_vec = bag.to_vec1::<f32>()?;
        bag_scores.extend(bag_vec);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.get(0).map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }
        i += take;
    }

    let is_decoy: Vec<bool> = bags.y_bag.iter().map(|&y| y < 0.5).collect();
    Ok(BagScoreOutput {
        bag_score: bag_scores,
        bag_y: bags.y_bag,
        is_decoy,
        bag_pid: bags.bag_pid,
        winner_hidden: hidden,
        hidden_dim,
    })
}

/// Score bags and extract winner head components for diagnostics.
pub fn score_bags_with_heads_from_rows(
    model: &crate::model::topaz::TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    feat_dim: usize,
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<BagHeadOutput> {
    let n = rows.len();
    if n == 0 {
        return Ok(BagHeadOutput {
            bag_score: Vec::new(),
            bag_y: Vec::new(),
            is_decoy: Vec::new(),
            bag_pid: Vec::new(),
            winner_hidden: Vec::new(),
            hidden_dim: 0,
            emb_ms2: Vec::new(),
            emb_ms2_dim: 0,
            emb_ms1: Vec::new(),
            emb_ms1_dim: 0,
            emb_all: Vec::new(),
            emb_all_dim: 0,
            coe_ms2: Vec::new(),
            coe_ms2_dim: 0,
            coe_ms1: Vec::new(),
            coe_ms1_dim: 0,
            coe_ms12: Vec::new(),
            coe_ms12_dim: 0,
            coe_all: Vec::new(),
            coe_all_dim: 0,
        });
    }

    let x_feat = rows_to_feature_matrix_preprocessed(rows, feat_dim, pre);
    let y_rows: Vec<u8> = rows.iter().map(|r| if r.is_decoy { 1 } else { 0 }).collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat,
        n,
        feat_dim,
        x_trace,
        c_total,
        l,
        &y_rows,
        &pid_rows,
        bag_k,
    );

    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;

    let b = bags.b;
    let mut bag_scores = Vec::with_capacity(b);
    let mut hidden: Vec<f32> = Vec::new();
    let mut hidden_dim = 0usize;

    let mut emb_ms2 = Vec::new();
    let mut emb_ms1 = Vec::new();
    let mut emb_all = Vec::new();
    let mut coe_ms2 = Vec::new();
    let mut coe_ms1 = Vec::new();
    let mut coe_ms12 = Vec::new();
    let mut coe_all = Vec::new();
    let mut emb_ms2_dim = 0usize;
    let mut emb_ms1_dim = 0usize;
    let mut emb_all_dim = 0usize;
    let mut coe_ms2_dim = 0usize;
    let mut coe_ms1_dim = 0usize;
    let mut coe_ms12_dim = 0usize;
    let mut coe_all_dim = 0usize;

    let bs = batch_size.max(1);
    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = xb.narrow(0, i, take)?;
        let tb_i = tb.narrow(0, i, take)?;
        let m_i = mask.narrow(0, i, take)?;

        let (_cand, bag, win, comps) = model.forward_bags_with_heads(&xb_i, &tb_i, &m_i)?;
        let bag_vec = bag.to_vec1::<f32>()?;
        bag_scores.extend(bag_vec);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.get(0).map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }

        let emb2 = comps.emb_ms2.to_vec2::<f32>()?;
        let emb1 = comps.emb_ms1.to_vec2::<f32>()?;
        let emba = comps.emb_all.to_vec2::<f32>()?;
        let coe2 = comps.coe_ms2.to_vec2::<f32>()?;
        let coe1 = comps.coe_ms1.to_vec2::<f32>()?;
        let coe12 = comps.coe_ms12.to_vec2::<f32>()?;
        let coea = comps.coe_all.to_vec2::<f32>()?;

        if emb_ms2_dim == 0 { emb_ms2_dim = emb2.get(0).map(|v| v.len()).unwrap_or(0); }
        if emb_ms1_dim == 0 { emb_ms1_dim = emb1.get(0).map(|v| v.len()).unwrap_or(0); }
        if emb_all_dim == 0 { emb_all_dim = emba.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_ms2_dim == 0 { coe_ms2_dim = coe2.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_ms1_dim == 0 { coe_ms1_dim = coe1.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_ms12_dim == 0 { coe_ms12_dim = coe12.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_all_dim == 0 { coe_all_dim = coea.get(0).map(|v| v.len()).unwrap_or(0); }

        for row in emb2 { emb_ms2.extend(row); }
        for row in emb1 { emb_ms1.extend(row); }
        for row in emba { emb_all.extend(row); }
        for row in coe2 { coe_ms2.extend(row); }
        for row in coe1 { coe_ms1.extend(row); }
        for row in coe12 { coe_ms12.extend(row); }
        for row in coea { coe_all.extend(row); }

        i += take;
    }

    let is_decoy: Vec<bool> = bags.y_bag.iter().map(|&y| y < 0.5).collect();
    Ok(BagHeadOutput {
        bag_score: bag_scores,
        bag_y: bags.y_bag,
        is_decoy,
        bag_pid: bags.bag_pid,
        winner_hidden: hidden,
        hidden_dim,
        emb_ms2,
        emb_ms2_dim,
        emb_ms1,
        emb_ms1_dim,
        emb_all,
        emb_all_dim,
        coe_ms2,
        coe_ms2_dim,
        coe_ms1,
        coe_ms1_dim,
        coe_ms12,
        coe_ms12_dim,
        coe_all,
        coe_all_dim,
    })
}

/// Score bags + heads with explicit feature columns.
pub fn score_bags_with_heads_from_rows_with_cols(
    model: &crate::model::topaz::TopazBagRanker,
    rows: &[FeatureRow],
    x_trace: &[f32],
    osw_cols: &[String],
    target_cols: &[String],
    c_total: usize,
    l: usize,
    bag_k: usize,
    device: &Device,
    batch_size: usize,
    pre: Option<&Preprocessor>,
) -> Result<BagHeadOutput> {
    let n = rows.len();
    if n == 0 {
        return Ok(BagHeadOutput {
            bag_score: Vec::new(),
            bag_y: Vec::new(),
            is_decoy: Vec::new(),
            bag_pid: Vec::new(),
            winner_hidden: Vec::new(),
            hidden_dim: 0,
            emb_ms2: Vec::new(),
            emb_ms2_dim: 0,
            emb_ms1: Vec::new(),
            emb_ms1_dim: 0,
            emb_all: Vec::new(),
            emb_all_dim: 0,
            coe_ms2: Vec::new(),
            coe_ms2_dim: 0,
            coe_ms1: Vec::new(),
            coe_ms1_dim: 0,
            coe_ms12: Vec::new(),
            coe_ms12_dim: 0,
            coe_all: Vec::new(),
            coe_all_dim: 0,
        });
    }

    let d = target_cols.len();
    let x_feat = rows_to_feature_matrix_with_cols(rows, osw_cols, target_cols, pre);
    let y_rows: Vec<u8> = rows.iter().map(|r| if r.is_decoy { 1 } else { 0 }).collect();
    let pid_rows: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();

    let bags = make_bags_with_traces(
        &x_feat,
        n,
        d,
        x_trace,
        c_total,
        l,
        &y_rows,
        &pid_rows,
        bag_k,
    );

    let xb = Tensor::from_vec(bags.x_bag, (bags.b, bags.k, bags.d), device)?;
    let tb = Tensor::from_vec(bags.t_bag, (bags.b, bags.k, bags.c, bags.l), device)?;
    let mask_u8: Vec<u8> = bags.mask.iter().map(|&v| if v { 1 } else { 0 }).collect();
    let mask = Tensor::from_vec(mask_u8, (bags.b, bags.k), device)?;

    let b = bags.b;
    let mut bag_scores = Vec::with_capacity(b);
    let mut hidden: Vec<f32> = Vec::new();
    let mut hidden_dim = 0usize;

    let mut emb_ms2 = Vec::new();
    let mut emb_ms1 = Vec::new();
    let mut emb_all = Vec::new();
    let mut coe_ms2 = Vec::new();
    let mut coe_ms1 = Vec::new();
    let mut coe_ms12 = Vec::new();
    let mut coe_all = Vec::new();
    let mut emb_ms2_dim = 0usize;
    let mut emb_ms1_dim = 0usize;
    let mut emb_all_dim = 0usize;
    let mut coe_ms2_dim = 0usize;
    let mut coe_ms1_dim = 0usize;
    let mut coe_ms12_dim = 0usize;
    let mut coe_all_dim = 0usize;

    let bs = batch_size.max(1);
    let mut i = 0usize;
    while i < b {
        let take = (b - i).min(bs);
        let xb_i = xb.narrow(0, i, take)?;
        let tb_i = tb.narrow(0, i, take)?;
        let m_i = mask.narrow(0, i, take)?;

        let (_cand, bag, win, comps) = model.forward_bags_with_heads(&xb_i, &tb_i, &m_i)?;
        let bag_vec = bag.to_vec1::<f32>()?;
        bag_scores.extend(bag_vec);

        let win_vec = win.to_vec2::<f32>()?;
        if hidden_dim == 0 {
            hidden_dim = win_vec.get(0).map(|v| v.len()).unwrap_or(0);
        }
        for row in win_vec {
            hidden.extend(row);
        }

        let emb2 = comps.emb_ms2.to_vec2::<f32>()?;
        let emb1 = comps.emb_ms1.to_vec2::<f32>()?;
        let emba = comps.emb_all.to_vec2::<f32>()?;
        let coe2 = comps.coe_ms2.to_vec2::<f32>()?;
        let coe1 = comps.coe_ms1.to_vec2::<f32>()?;
        let coe12 = comps.coe_ms12.to_vec2::<f32>()?;
        let coea = comps.coe_all.to_vec2::<f32>()?;

        if emb_ms2_dim == 0 { emb_ms2_dim = emb2.get(0).map(|v| v.len()).unwrap_or(0); }
        if emb_ms1_dim == 0 { emb_ms1_dim = emb1.get(0).map(|v| v.len()).unwrap_or(0); }
        if emb_all_dim == 0 { emb_all_dim = emba.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_ms2_dim == 0 { coe_ms2_dim = coe2.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_ms1_dim == 0 { coe_ms1_dim = coe1.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_ms12_dim == 0 { coe_ms12_dim = coe12.get(0).map(|v| v.len()).unwrap_or(0); }
        if coe_all_dim == 0 { coe_all_dim = coea.get(0).map(|v| v.len()).unwrap_or(0); }

        for row in emb2 { emb_ms2.extend(row); }
        for row in emb1 { emb_ms1.extend(row); }
        for row in emba { emb_all.extend(row); }
        for row in coe2 { coe_ms2.extend(row); }
        for row in coe1 { coe_ms1.extend(row); }
        for row in coe12 { coe_ms12.extend(row); }
        for row in coea { coe_all.extend(row); }

        i += take;
    }

    let is_decoy: Vec<bool> = bags.y_bag.iter().map(|&y| y < 0.5).collect();
    Ok(BagHeadOutput {
        bag_score: bag_scores,
        bag_y: bags.y_bag,
        is_decoy,
        bag_pid: bags.bag_pid,
        winner_hidden: hidden,
        hidden_dim,
        emb_ms2,
        emb_ms2_dim,
        emb_ms1,
        emb_ms1_dim,
        emb_all,
        emb_all_dim,
        coe_ms2,
        coe_ms2_dim,
        coe_ms1,
        coe_ms1_dim,
        coe_ms12,
        coe_ms12_dim,
        coe_all,
        coe_all_dim,
    })
}

#[cfg(feature = "io-parquet")]
pub fn build_trace_tensors_from_parquet(
    rows: &[FeatureRow],
    xic_path: &Path,
    cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
) -> Result<Vec<f32>> {
    let mut reader = crate::io::xic_parquet::XicParquetReader::new(xic_path);
    if let Some(levels) = &fetch_cfg.ms_levels {
        reader.filter_ms_level(levels.clone());
    }
    if let Some(flag) = fetch_cfg.detecting_transition {
        reader.filter_detecting_transition(flag);
    }
    if let Some(flag) = fetch_cfg.decoy {
        reader.filter_decoy(flag);
    }
    build_trace_tensors_from_source(rows, &mut reader, cfg)
}

#[cfg(feature = "io-parquet")]
pub fn build_trace_tensors_from_parquet_cached(
    rows: &[FeatureRow],
    xic_path: &Path,
    cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
    cache: &SharedXicCache,
    disk: Option<&XicDiskCache>,
) -> Result<Vec<f32>> {
    let n = rows.len();
    let c_total = cfg.total_c();
    let mut out = vec![0f32; n * c_total * cfg.l];
    if n == 0 || c_total == 0 || cfg.l == 0 {
        return Ok(out);
    }
    if !cache.is_enabled() && disk.is_none() {
        return build_trace_tensors_from_parquet(rows, xic_path, cfg, fetch_cfg);
    }

    let mut by_run: HashMap<u64, HashSet<u64>> = HashMap::new();
    for row in rows {
        by_run
            .entry(row.run_id)
            .or_default()
            .insert(row.precursor_id);
    }

    let mut xic_by_run: HashMap<u64, HashMap<u64, PrecursorXic>> = HashMap::new();
    for (run_id, prec_set) in by_run {
        let fetched = fetch_precursors_cached(xic_path, run_id, &prec_set, cache, disk, fetch_cfg)?;
        xic_by_run.insert(run_id, fetched);
    }

    let row_len = c_total * cfg.l;
    let fill_row = |row: &FeatureRow, dst: &mut [f32]| {
        if let Some(run_map) = xic_by_run.get(&row.run_id) {
            if let Some(xic) = run_map.get(&row.precursor_id) {
                let (ms1_series, ms2_series) = split_ms1_ms2(xic);

                let mut offset = 0usize;
                if cfg.ms1_cmax > 0 {
                    if ms1_series.is_empty()
                        && !WARNED_MISSING_MS1.swap(true, Ordering::Relaxed)
                    {
                        log::warn!(
                            "missing MS1 traces for at least one precursor; padding zeros"
                        );
                    }
                    let t_ms1 = extract_trace_tensor_centered(
                        &ms1_series,
                        row.exp_rt,
                        cfg.l,
                        cfg.ms1_cmax,
                        cfg.normalize_max,
                    );
                    dst[offset..offset + cfg.ms1_cmax * cfg.l].copy_from_slice(&t_ms1);
                    offset += cfg.ms1_cmax * cfg.l;
                }
                let t_ms2 = extract_trace_tensor_centered(
                    &ms2_series,
                    row.exp_rt,
                    cfg.l,
                    cfg.ms2_cmax,
                    cfg.normalize_max,
                );
                dst[offset..offset + cfg.ms2_cmax * cfg.l].copy_from_slice(&t_ms2);
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        out.par_chunks_mut(row_len)
            .zip(rows.par_iter())
            .for_each(|(dst, row)| fill_row(row, dst));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, row) in rows.iter().enumerate() {
            let dst = &mut out[i * row_len..(i + 1) * row_len];
            fill_row(row, dst);
        }
    }

    Ok(out)
}

#[cfg(feature = "io-parquet")]
pub fn build_trace_tensors_from_parquet_map(
    rows: &[FeatureRow],
    xic_map: &HashMap<u64, std::path::PathBuf>,
    cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
) -> Result<Vec<f32>> {
    let cache = SharedXicCache::new(0);
    build_trace_tensors_from_parquet_map_cached(rows, xic_map, cfg, fetch_cfg, &cache, None)
}

#[cfg(feature = "io-parquet")]
pub fn build_trace_tensors_from_parquet_map_cached(
    rows: &[FeatureRow],
    xic_map: &HashMap<u64, std::path::PathBuf>,
    cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
    cache: &SharedXicCache,
    disk: Option<&XicDiskCache>,
) -> Result<Vec<f32>> {
    let n = rows.len();
    let c_total = cfg.total_c();
    let mut out = vec![0f32; n * c_total * cfg.l];
    if n == 0 || c_total == 0 || cfg.l == 0 {
        return Ok(out);
    }

    let mut by_run: HashMap<u64, HashSet<u64>> = HashMap::new();
    for row in rows {
        by_run
            .entry(row.run_id)
            .or_default()
            .insert(row.precursor_id);
    }

    let mut run_to_path: HashMap<u64, std::path::PathBuf> = HashMap::new();
    let mut filtered_by_run: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut missing_runs = Vec::new();
    for (run_id, precs) in by_run {
        if let Some(path) = xic_map.get(&run_id) {
            run_to_path.insert(run_id, path.clone());
            filtered_by_run.insert(run_id, precs);
        } else {
            missing_runs.push(run_id);
        }
    }
    if !missing_runs.is_empty() {
        missing_runs.sort_unstable();
        missing_runs.dedup();
        log::warn!(
            "XIC map is missing {} run_ids (will leave traces zeroed): {:?}",
            missing_runs.len(),
            missing_runs
        );
    }

    let items: Vec<(u64, HashSet<u64>, std::path::PathBuf)> = filtered_by_run
        .into_iter()
        .filter_map(|(run_id, prec_set)| {
            run_to_path
                .get(&run_id)
                .cloned()
                .map(|path| (run_id, prec_set, path))
        })
        .collect();

    let fetch_one = |run_id: u64,
                     prec_set: HashSet<u64>,
                     path: std::path::PathBuf,
                     fetch_cfg: &XicFetchConfig|
     -> Result<(u64, HashMap<u64, PrecursorXic>)> {
        if prec_set.is_empty() {
            return Ok((run_id, HashMap::new()));
        }
        let mut reader = crate::io::xic_parquet::XicParquetReader::new(&path);
        reader.filter_run_id(run_id);
        if let Some(levels) = &fetch_cfg.ms_levels {
            reader.filter_ms_level(levels.clone());
        }
        if let Some(flag) = fetch_cfg.detecting_transition {
            reader.filter_detecting_transition(flag);
        }
        if let Some(flag) = fetch_cfg.decoy {
            reader.filter_decoy(flag);
        }
        reader.filter_precursor_id(prec_set.iter().copied());
        let fetched = reader.fetch()?;
        let requested = prec_set.len();
        let fetched_count = fetched.len();
        if fetched_count == 0 {
            log::warn!(
                "XIC map path {:?}: fetched 0 of {} precursors (run_id={})",
                path,
                requested,
                run_id
            );
            let mut fallback = crate::io::xic_parquet::XicParquetReader::new(&path);
            if let Some(levels) = &fetch_cfg.ms_levels {
                fallback.filter_ms_level(levels.clone());
            }
            if let Some(flag) = fetch_cfg.detecting_transition {
                fallback.filter_detecting_transition(flag);
            }
            if let Some(flag) = fetch_cfg.decoy {
                fallback.filter_decoy(flag);
            }
            fallback.filter_precursor_id(prec_set.iter().copied());
            let fetched_fb = fallback.fetch()?;
            let fetched_fb_count = fetched_fb.len();
            if fetched_fb_count == 0 {
                log::warn!(
                    "XIC map path {:?}: fallback fetched 0 of {} precursors (ignoring RUN_ID)",
                    path,
                    requested
                );
            } else {
                log::info!(
                    "XIC map path {:?}: fallback fetched {} of {} precursors (ignoring RUN_ID)",
                    path,
                    fetched_fb_count,
                    requested
                );
            }
            let mut map: HashMap<u64, PrecursorXic> = HashMap::new();
            for xic in fetched_fb {
                map.insert(xic.precursor_id, xic);
            }
            return Ok((run_id, map));
        } else {
            log::info!(
                "XIC map path {:?}: fetched {} of {} precursors (run_id={})",
                path,
                fetched_count,
                requested,
                run_id
            );
        }
        let mut map: HashMap<u64, PrecursorXic> = HashMap::new();
        for xic in fetched {
            map.insert(xic.precursor_id, xic);
        }
        Ok((run_id, map))
    };

    let use_cache = cache.is_enabled() || disk.is_some();
    let disk_owned = disk.cloned();
    let cache_owned = cache.clone();

    #[cfg(feature = "rayon")]
    let fetched_all: Vec<(u64, HashMap<u64, PrecursorXic>)> = items
        .into_par_iter()
        .map(|(run_id, prec_set, path)| {
            if use_cache {
                let cache = cache_owned.clone();
                let disk = disk_owned.clone();
                fetch_precursors_cached_with_fallback(
                    &path,
                    run_id,
                    &prec_set,
                    &cache,
                    disk.as_ref(),
                    fetch_cfg,
                )
                .map(|map| (run_id, map))
            } else {
                fetch_one(run_id, prec_set, path, fetch_cfg)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    #[cfg(not(feature = "rayon"))]
    let fetched_all: Vec<(u64, HashMap<u64, PrecursorXic>)> = items
        .into_iter()
        .map(|(run_id, prec_set, path)| {
            if use_cache {
                fetch_precursors_cached_with_fallback(path, run_id, &prec_set, cache, disk, fetch_cfg)
                    .map(|map| (run_id, map))
            } else {
                fetch_one(run_id, prec_set, path, fetch_cfg)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let mut xic_by_run: HashMap<u64, HashMap<u64, PrecursorXic>> = HashMap::new();
    for (run_id, map) in fetched_all {
        xic_by_run.insert(run_id, map);
    }

    if log::log_enabled!(log::Level::Info) {
        let mut logged_runs: HashSet<u64> = HashSet::new();
        for row in rows.iter() {
            if logged_runs.contains(&row.run_id) {
                continue;
            }
            if let Some(run_map) = xic_by_run.get(&row.run_id) {
                if let Some(xic) = run_map.get(&row.precursor_id) {
                    let (ms1_series, ms2_series) = split_ms1_ms2(xic);
                    let (probe_series, label) = if !ms2_series.is_empty() {
                        (&ms2_series, "ms2")
                    } else {
                        (&ms1_series, "ms1")
                    };
                    if let Some(first) = probe_series.get(0) {
                        let mut min_rt = f32::INFINITY;
                        let mut max_rt = f32::NEG_INFINITY;
                        let mut max_int = 0f32;
                        for p in &first.points {
                            if p.rt < min_rt {
                                min_rt = p.rt;
                            }
                            if p.rt > max_rt {
                                max_rt = p.rt;
                            }
                            if p.intensity.abs() > max_int {
                                max_int = p.intensity.abs();
                            }
                        }
                        log::info!(
                            "XIC probe run_id={} ({label}): exp_rt={} rt_range=[{}, {}] max_intensity={}",
                            row.run_id,
                            row.exp_rt,
                            min_rt,
                            max_rt,
                            max_int
                        );
                    } else {
                        log::warn!(
                            "XIC probe run_id={} has no {} transitions for precursor_id={}",
                            row.run_id,
                            label,
                            row.precursor_id
                        );
                    }
                }
            }
            logged_runs.insert(row.run_id);
        }
    }

    let row_len = c_total * cfg.l;
    let fill_row = |row: &FeatureRow, dst: &mut [f32]| {
        if let Some(run_map) = xic_by_run.get(&row.run_id) {
            if let Some(xic) = run_map.get(&row.precursor_id) {
                let (ms1_series, ms2_series) = split_ms1_ms2(xic);

                let mut offset = 0usize;
                if cfg.ms1_cmax > 0 {
                    if ms1_series.is_empty()
                        && !WARNED_MISSING_MS1.swap(true, Ordering::Relaxed)
                    {
                        log::warn!(
                            "missing MS1 traces for at least one precursor; padding zeros"
                        );
                    }
                    let t_ms1 = extract_trace_tensor_centered(
                        &ms1_series,
                        row.exp_rt,
                        cfg.l,
                        cfg.ms1_cmax,
                        cfg.normalize_max,
                    );
                    dst[offset..offset + cfg.ms1_cmax * cfg.l].copy_from_slice(&t_ms1);
                    offset += cfg.ms1_cmax * cfg.l;
                }
                let t_ms2 = extract_trace_tensor_centered(
                    &ms2_series,
                    row.exp_rt,
                    cfg.l,
                    cfg.ms2_cmax,
                    cfg.normalize_max,
                );
                dst[offset..offset + cfg.ms2_cmax * cfg.l].copy_from_slice(&t_ms2);
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        out.par_chunks_mut(row_len)
            .zip(rows.par_iter())
            .for_each(|(dst, row)| fill_row(row, dst));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, row) in rows.iter().enumerate() {
            let dst = &mut out[i * row_len..(i + 1) * row_len];
            fill_row(row, dst);
        }
    }

    Ok(out)
}

#[cfg(feature = "io-sqlite")]
pub fn read_osw_features(
    path: &Path,
    cfg: &crate::io::osw::OswReadConfig,
) -> Result<OswFeatureTable> {
    crate::io::osw::read_feature_rows(path, cfg)
}

/// End-to-end inference: OSW + XIC -> candidate scores -> SCORE table.
#[cfg(all(feature = "io-sqlite", feature = "io-parquet"))]
pub fn infer_score_table_from_osw_xic(
    model: &impl CandidateScorerInterface,
    device: &Device,
    model_cfg: &TopazConfig,
    osw_path: &Path,
    xic_path: &Path,
    osw_cfg: &crate::io::osw::OswReadConfig,
    trace_cfg: &TraceBuildConfig,
    fetch_cfg: &XicFetchConfig,
    batch_size: usize,
    pep_bins: usize,
    pre: Option<&Preprocessor>,
) -> Result<Vec<ScoreTableRow>> {
    let table = read_osw_features(osw_path, osw_cfg)?;
    let rows = table.rows;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let feat_dim = table.feature_cols.len();
    if model_cfg.use_heuristic_features && model_cfg.feat_dim != feat_dim {
        anyhow::bail!(
            "feature dim mismatch: model expects {}, OSW has {}",
            model_cfg.feat_dim,
            feat_dim
        );
    }

    let x_feat = rows_to_feature_matrix_preprocessed(&rows, model_cfg.feat_dim, pre);
    let x_trace = build_trace_tensors_from_parquet(&rows, xic_path, trace_cfg, fetch_cfg)?;

    let n = rows.len();
    let c_total = trace_cfg.total_c();
    let x_feat_t = Tensor::from_vec(x_feat, (n, model_cfg.feat_dim), device)?;
    let x_trace_t = Tensor::from_vec(x_trace, (n, c_total, trace_cfg.l), device)?;

    let scores_t = score_candidates(model, &x_feat_t, &x_trace_t, batch_size.max(1))?;
    let scores = scores_t.to_vec1::<f32>()?;

    Ok(build_score_table_from_rows(&rows, &scores, pep_bins))
}

#[cfg(all(test, feature = "io-sqlite", feature = "io-parquet"))]
mod tests {
    use super::*;
    use crate::infer::{build_score_table_from_rows, score_candidates, write_score_tsv};
    use crate::io::osw::{OswLevel, OswReadConfig};
    use crate::model::topaz::{TopazBagRanker, TopazConfig};
    use crate::building_blocks::trace_input::TraceInputMode;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::fs;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("redeem_topaz_{name}_{stamp}.tsv"));
        p
    }

    #[test]
    fn test_osw_xic_end_to_end_tsv() -> Result<()> {
        let osw_path = Path::new(
            "/home/singjc/Documents/github/PASS01508_DIAlignR_Spyo/2026026_for_ptsc_model/gold_standard_spyo.osw",
        );
        let xic_path = Path::new(
            "/home/singjc/Documents/github/PASS01508_DIAlignR_Spyo/2026026_for_ptsc_model/hroest_K120808_Strep0%PlasmaBiolRepl1_R01_SW.xic",
        );
        if !osw_path.exists() || !xic_path.exists() {
            log::info!("skipping test_osw_xic_end_to_end_tsv: sample files not found");
            return Ok(());
        }

        let osw_cfg = OswReadConfig {
            level: OswLevel::Ms2,
            ..Default::default()
        };
        let table = match read_osw_features(osw_path, &osw_cfg) {
            Ok(table) => table,
            Err(err) => {
                log::info!(
                    "skipping test_osw_xic_end_to_end_tsv: unable to open sample OSW/XIC ({err:#})"
                );
                return Ok(());
            }
        };
        if table.rows.is_empty() {
            log::info!("skipping test_osw_xic_end_to_end_tsv: OSW has no rows");
            return Ok(());
        }

        let run_id = table.rows[0].run_id;
        let rows: Vec<FeatureRow> = table
            .rows
            .into_iter()
            .filter(|r| r.run_id == run_id)
            .take(64)
            .collect();
        let feat_dim = table.feature_cols.len();

        let cfg = TopazConfig {
            feat_dim,
            ms2_cmax: 6,
            ms1_cmax: 0,
            l: 64,
            trace_emb_dim: 8,
            mlp_hidden: vec![16],
            dropout: 0.0,
            trace_input_mode: TraceInputMode::Single,
            use_heuristic_features: true,
            use_coelution_head: false,
            ..Default::default()
        };

        let trace_cfg = TraceBuildConfig {
            l: cfg.l,
            ms1_cmax: cfg.ms1_cmax,
            ms2_cmax: cfg.ms2_cmax,
            normalize_max: false,
        };
        let fetch_cfg = XicFetchConfig {
            ms_levels: Some(vec![2]),
            detecting_transition: Some(1),
            decoy: None,
        };

        let x_feat = rows_to_feature_matrix(&rows, cfg.feat_dim);
        let x_trace = build_trace_tensors_from_parquet(&rows, xic_path, &trace_cfg, &fetch_cfg)?;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = TopazBagRanker::new(vb.pp("topaz"), &cfg)?;

        let n = rows.len();
        let x_feat_t = Tensor::from_vec(x_feat, (n, cfg.feat_dim), &device)?;
        let x_trace_t = Tensor::from_vec(x_trace, (n, trace_cfg.total_c(), trace_cfg.l), &device)?;
        let scores_t = score_candidates(&model, &x_feat_t, &x_trace_t, 128)?;
        let scores = scores_t.to_vec1::<f32>()?;

        let score_rows = build_score_table_from_rows(&rows, &scores, 10);
        let path = tmp_path("osw_xic_score");
        write_score_tsv(&path, &score_rows)?;

        let text = fs::read_to_string(&path)?;
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.len() >= 2);
        assert_eq!(lines[0], "FEATURE_ID\tSCORE\tRANK\tPVALUE\tQVALUE\tPEP");

        let _ = fs::remove_file(&path);
        Ok(())
    }
}
