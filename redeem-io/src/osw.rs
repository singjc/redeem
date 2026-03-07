//! SQLite-backed OSW feature-table reading and score-table writeback.
//!
//! This module provides the narrow OSW operations needed by TOPAZ:
//! - read candidate-level feature tables from one of the OpenSWATH feature
//!   tables (`FEATURE_MS2`, `FEATURE_MS1`, `FEATURE_TRANSITION`),
//! - read lightweight metadata for reports and diagnostics,
//! - write TOPAZ score tables back into the OSW SQLite file.
//!
//! The returned row type is intentionally normalized so that the rest of the
//! code does not need to care which SQL table the features came from.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One candidate feature row extracted from an OSW file.
///
/// Each row corresponds to one scored candidate peak group (or transition-level
/// candidate when `OswLevel::Transition` is used).
#[derive(Debug, Clone)]
pub struct FeatureRow {
    /// Stable OSW `FEATURE.ID`.
    pub feature_id: u64,
    /// OSW `PRECURSOR_ID` for the row.
    pub precursor_id: u64,
    /// OSW `RUN_ID` for the row.
    pub run_id: u64,
    /// Bagging key used by TOPAZ. Typically `RUN_ID_PRECURSOR_ID`.
    pub group_id: String,
    /// Experimental apex retention time reported by OpenSWATH.
    pub exp_rt: f32,
    /// `true` for decoy rows, `false` for targets.
    pub is_decoy: bool,
    /// Selected scalar features in the same order as `OswFeatureTable.feature_cols`.
    pub features: Vec<f32>,
}

/// Peptide/precursor metadata used by reports and diagnostics.
#[derive(Debug, Clone)]
pub struct PrecursorMeta {
    /// OSW `PRECURSOR.ID`.
    pub precursor_id: u64,
    /// Modified peptide sequence associated with the precursor.
    pub modified_sequence: String,
    /// Precursor charge state.
    pub charge: i32,
}

/// Minimal feature metadata keyed by `FEATURE_ID`.
#[derive(Debug, Clone)]
pub struct FeatureMeta {
    /// OSW `FEATURE.ID`.
    pub feature_id: u64,
    /// OSW `RUN_ID`.
    pub run_id: u64,
    /// OSW `PRECURSOR_ID`.
    pub precursor_id: u64,
    /// `true` for decoy rows, `false` for targets.
    pub is_decoy: bool,
}

/// Compact score-table view used when reading an existing OSW score table.
#[derive(Debug, Clone)]
pub struct ScoreTableEntry {
    /// OSW `FEATURE_ID`.
    pub feature_id: u64,
    /// Primary score column.
    pub score: f32,
    /// Optional rank column if the table contains `RANK`.
    pub rank: Option<i32>,
    /// Optional q-value column if the table contains `QVALUE`.
    pub qvalue: Option<f32>,
}

/// Full OSW score-table row written by TOPAZ.
#[derive(Debug, Clone)]
pub struct ScoreRow {
    /// OSW `FEATURE_ID`.
    pub feature_id: u64,
    /// TOPAZ score.
    pub score: f32,
    /// Rank within the corresponding run/precursor group.
    pub rank: i32,
    /// Decoy-tail p-value.
    pub pvalue: f32,
    /// Target-decoy q-value.
    pub qvalue: f32,
    /// Posterior error probability estimate.
    pub pep: f32,
}

/// Feature granularity to read from an OSW file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OswLevel {
    /// Read `FEATURE_MS2`.
    Ms2,
    /// Read `FEATURE_MS1`.
    Ms1,
    /// Read `FEATURE_MS2` and append matching `FEATURE_MS1` columns.
    Ms1Ms2,
    /// Read `FEATURE_TRANSITION` after applying IPF-related restrictions.
    Transition,
    /// Placeholder for alignment-level reading. Not implemented.
    Alignment,
}

/// OSW reader configuration mirroring the PyProphet/OpenSWATH feature-table
/// choices used by TOPAZ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OswReadConfig {
    /// Which feature table(s) to read.
    pub level: OswLevel,
    /// Maximum `SCORE_MS2.RANK` kept for transition-level/IPF reads.
    pub ipf_max_rank: i32,
    /// Maximum `SCORE_MS2.PEP` kept for transition-level/IPF reads.
    pub ipf_max_pep: f32,
    /// Maximum isotope-overlap score kept for transition-level/IPF reads.
    pub ipf_max_transition_isotope_overlap: f32,
    /// Minimum transition signal-to-noise score kept for transition-level/IPF reads.
    pub ipf_min_transition_sn: f32,
}

impl Default for OswReadConfig {
    fn default() -> Self {
        Self {
            level: OswLevel::Ms2,
            ipf_max_rank: 9999,
            ipf_max_pep: 1.0,
            ipf_max_transition_isotope_overlap: 1.0,
            ipf_min_transition_sn: 0.0,
        }
    }
}

/// Result of reading an OSW feature table.
///
/// `feature_cols` gives the exact order of the scalar values stored in
/// `FeatureRow.features`.
#[derive(Debug, Clone)]
pub struct OswFeatureTable {
    pub rows: Vec<FeatureRow>,
    pub feature_cols: Vec<String>,
}

/// Read feature rows from an OSW SQLite file.
///
/// The selected SQL feature table depends on `cfg.level`. The resulting rows
/// are normalized into a single [`FeatureRow`] representation that can be fed
/// directly into TOPAZ preprocessing and bagging.
#[cfg(feature = "sqlite")]
pub fn read_feature_rows(path: &std::path::Path, cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    use rusqlite::Connection;

    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    match cfg.level {
        OswLevel::Ms2 | OswLevel::Ms1Ms2 => read_ms2_features(&conn, cfg),
        OswLevel::Ms1 => read_ms1_features(&conn, cfg),
        OswLevel::Transition => read_transition_features(&conn, cfg),
        OswLevel::Alignment => bail!("alignment-level OSW read not implemented yet"),
    }
}

/// Read precursor metadata keyed by `PRECURSOR_ID`.
///
/// This is used mainly by reports so that points in embedding plots can be
/// annotated with peptide sequence and charge state.
#[cfg(feature = "sqlite")]
pub fn read_precursor_meta(path: &std::path::Path) -> Result<HashMap<u64, PrecursorMeta>> {
    use rusqlite::Connection;

    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let sql = r#"
        SELECT ppm.PRECURSOR_ID, pep.MODIFIED_SEQUENCE, prec.CHARGE
        FROM PRECURSOR_PEPTIDE_MAPPING ppm
        JOIN PEPTIDE pep ON ppm.PEPTIDE_ID = pep.ID
        JOIN PRECURSOR prec ON prec.ID = ppm.PRECURSOR_ID
    "#;
    let mut stmt = conn.prepare(sql)?;
    let mut out: HashMap<u64, PrecursorMeta> = HashMap::new();
    let rows = stmt.query_map([], |row| {
        let prec_id: i64 = row.get(0)?;
        let seq: String = row.get(1)?;
        let charge: i32 = row.get(2)?;
        Ok((prec_id as u64, seq, charge))
    })?;
    for r in rows {
        let (prec_id, seq, charge) = r?;
        out.entry(prec_id).or_insert(PrecursorMeta {
            precursor_id: prec_id,
            modified_sequence: seq,
            charge,
        });
    }
    Ok(out)
}

#[cfg(feature = "sqlite")]
fn table_exists(conn: &rusqlite::Connection, table: &str) -> Result<bool> {
    let mut stmt =
        conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?1 LIMIT 1")?;
    let mut rows = stmt.query([table])?;
    Ok(rows.next()?.is_some())
}

#[cfg(feature = "sqlite")]
fn list_columns(conn: &rusqlite::Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = stmt.query_map([], |row| {
        let name: String = row.get(1)?;
        Ok(name)
    })?;
    let mut cols = Vec::new();
    for r in rows {
        cols.push(r?);
    }
    Ok(cols)
}

/// Read feature metadata keyed by `FEATURE_ID`.
///
/// This is the lightest-weight way to recover run/precursor/decoy context for
/// an already-scored feature table.
#[cfg(feature = "sqlite")]
pub fn read_feature_meta(path: &std::path::Path) -> Result<HashMap<u64, FeatureMeta>> {
    use rusqlite::Connection;

    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let sql = r#"
        SELECT f.ID, f.RUN_ID, f.PRECURSOR_ID, p.DECOY
        FROM FEATURE f
        JOIN PRECURSOR p ON p.ID = f.PRECURSOR_ID
    "#;
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        let feature_id: i64 = row.get(0)?;
        let run_id: i64 = row.get(1)?;
        let precursor_id: i64 = row.get(2)?;
        let decoy: i64 = row.get(3)?;
        Ok(FeatureMeta {
            feature_id: feature_id as u64,
            run_id: run_id as u64,
            precursor_id: precursor_id as u64,
            is_decoy: decoy == 1,
        })
    })?;
    let mut out = HashMap::new();
    for r in rows {
        let meta = r?;
        out.insert(meta.feature_id, meta);
    }
    Ok(out)
}

/// Read an existing OSW score table by name.
///
/// Missing optional columns such as `RANK` or `QVALUE` are tolerated; they are
/// returned as `None` in the resulting [`ScoreTableEntry`] values.
#[cfg(feature = "sqlite")]
pub fn read_score_table(path: &std::path::Path, table: &str) -> Result<Vec<ScoreTableEntry>> {
    use rusqlite::Connection;

    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    if !table_exists(&conn, table)? {
        return Ok(Vec::new());
    }
    let cols = list_columns(&conn, table)?;
    let cols_upper: Vec<String> = cols.iter().map(|c| c.to_uppercase()).collect();
    let has_score = cols_upper.iter().any(|c| c == "SCORE");
    if !has_score {
        return Ok(Vec::new());
    }
    let has_rank = cols_upper.iter().any(|c| c == "RANK");
    let has_q = cols_upper.iter().any(|c| c == "QVALUE");

    let rank_expr = if has_rank { "RANK" } else { "NULL" };
    let q_expr = if has_q { "QVALUE" } else { "NULL" };
    let sql =
        format!("SELECT FEATURE_ID, SCORE, {rank_expr} AS RANK, {q_expr} AS QVALUE FROM {table}");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let feature_id: i64 = row.get(0)?;
        let score: Option<f32> = row.get(1)?;
        let rank: Option<i64> = row.get(2)?;
        let qvalue: Option<f32> = row.get(3)?;
        Ok(ScoreTableEntry {
            feature_id: feature_id as u64,
            score: score.unwrap_or(0.0),
            rank: rank.map(|r| r as i32),
            qvalue,
        })
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

#[cfg(not(feature = "sqlite"))]
/// Stub used when `redeem-io` is built without SQLite support.
pub fn read_precursor_meta(_path: &std::path::Path) -> Result<HashMap<u64, PrecursorMeta>> {
    bail!("redeem-io compiled without feature `sqlite`")
}

#[cfg(not(feature = "sqlite"))]
/// Stub used when `redeem-io` is built without SQLite support.
pub fn read_feature_meta(_path: &std::path::Path) -> Result<HashMap<u64, FeatureMeta>> {
    bail!("redeem-io compiled without feature `sqlite`")
}

#[cfg(not(feature = "sqlite"))]
/// Stub used when `redeem-io` is built without SQLite support.
pub fn read_score_table(_path: &std::path::Path, _table: &str) -> Result<Vec<ScoreTableEntry>> {
    bail!("redeem-io compiled without feature `sqlite`")
}

#[cfg(not(feature = "sqlite"))]
/// Stub used when `redeem-io` is built without SQLite support.
pub fn read_feature_rows(_path: &std::path::Path, _cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    bail!("redeem-io compiled without feature `sqlite`")
}

#[cfg(feature = "sqlite")]
/// Write or update a TOPAZ-compatible score table in an OSW file.
///
/// The target table is created if missing. Existing rows with the same
/// `FEATURE_ID` are replaced.
pub fn write_score_table(path: &std::path::Path, table: &str, rows: &[ScoreRow]) -> Result<()> {
    use rusqlite::{Connection, params};

    let mut conn = Connection::open(path)?;
    let ddl = format!(
        "CREATE TABLE IF NOT EXISTS {table} (
            FEATURE_ID INTEGER PRIMARY KEY,
            SCORE REAL,
            RANK INTEGER,
            PVALUE REAL,
            QVALUE REAL,
            PEP REAL
        )"
    );
    conn.execute(&ddl, [])?;
    let tx = conn.transaction()?;
    let sql = format!(
        "INSERT OR REPLACE INTO {table} (FEATURE_ID, SCORE, RANK, PVALUE, QVALUE, PEP)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
    );
    {
        let mut stmt = tx.prepare(&sql)?;
        for r in rows {
            stmt.execute(params![
                r.feature_id,
                r.score,
                r.rank,
                r.pvalue,
                r.qvalue,
                r.pep
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

#[cfg(feature = "sqlite")]
fn list_var_columns(conn: &rusqlite::Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = stmt.query_map([], |row| {
        let name: String = row.get(1)?;
        Ok(name)
    })?;
    let mut cols = Vec::new();
    for r in rows {
        let name = r?;
        if name.to_uppercase().starts_with("VAR_") {
            cols.push(name);
        }
    }
    Ok(cols)
}

#[cfg(feature = "sqlite")]
fn get_f32_or_nan(row: &rusqlite::Row, idx: usize) -> rusqlite::Result<f32> {
    let v: Option<f32> = row.get(idx)?;
    Ok(v.unwrap_or(f32::NAN))
}

#[cfg(feature = "sqlite")]
fn filter_all_null_columns(
    conn: &rusqlite::Connection,
    table: &str,
    cols: Vec<String>,
) -> Result<(Vec<String>, Vec<String>)> {
    let mut keep = Vec::new();
    let mut dropped = Vec::new();
    for c in cols {
        let sql = format!("SELECT 1 FROM {table} WHERE {c} IS NOT NULL LIMIT 1");
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        if rows.next()?.is_some() {
            keep.push(c);
        } else {
            dropped.push(c);
        }
    }
    Ok((keep, dropped))
}

#[cfg(feature = "sqlite")]
fn read_ms2_features(conn: &rusqlite::Connection, cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    use rusqlite::Row;

    let mut feature_cols = list_var_columns(conn, "FEATURE_MS2")?;
    feature_cols.sort();
    let (feature_cols, dropped_ms2) = filter_all_null_columns(conn, "FEATURE_MS2", feature_cols)?;
    if !dropped_ms2.is_empty() {
        eprintln!(
            "warning: dropping all-NULL FEATURE_MS2 columns: {}",
            dropped_ms2.join(", ")
        );
    }

    let mut ms1_cols: Vec<String> = Vec::new();
    if cfg.level == OswLevel::Ms1Ms2 {
        ms1_cols = list_var_columns(conn, "FEATURE_MS1")?;
        ms1_cols.sort();
        let (kept, dropped) = filter_all_null_columns(conn, "FEATURE_MS1", ms1_cols)?;
        if !dropped.is_empty() {
            eprintln!(
                "warning: dropping all-NULL FEATURE_MS1 columns: {}",
                dropped.join(", ")
            );
        }
        ms1_cols = kept;
    }

    let mut select_cols: Vec<String> = Vec::new();
    select_cols.push("fm.FEATURE_ID".to_string());
    select_cols.push("f.RUN_ID".to_string());
    select_cols.push("f.PRECURSOR_ID".to_string());
    select_cols.push("f.EXP_RT".to_string());
    select_cols.push("p.DECOY".to_string());

    for c in &feature_cols {
        select_cols.push(format!("fm.{c}"));
    }
    if cfg.level == OswLevel::Ms1Ms2 {
        for c in &ms1_cols {
            let suffix = c.trim_start_matches("VAR_");
            select_cols.push(format!("ms1.{c} AS VAR_MS1_{suffix}"));
        }
    }
    let select = select_cols.join(", ");

    let mut joins = String::from(
        "FROM FEATURE_MS2 fm
         INNER JOIN FEATURE f ON fm.FEATURE_ID = f.ID
         INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID",
    );
    if cfg.level == OswLevel::Ms1Ms2 {
        joins.push_str(" LEFT JOIN FEATURE_MS1 ms1 ON ms1.FEATURE_ID = fm.FEATURE_ID");
    }
    let sql = format!(
        "SELECT {select} {joins}
         ORDER BY f.RUN_ID, p.ID, f.EXP_RT"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row: &Row| {
        let feature_id: i64 = row.get(0)?;
        let run_id: i64 = row.get(1)?;
        let precursor_id: i64 = row.get(2)?;
        let exp_rt: f32 = row.get(3)?;
        let decoy: i32 = row.get(4)?;

        let mut idx = 5usize;
        let mut feats = Vec::new();

        for _ in 0..feature_cols.len() {
            let v = get_f32_or_nan(row, idx)?;
            feats.push(v);
            idx += 1;
        }
        if cfg.level == OswLevel::Ms1Ms2 {
            for _ in 0..ms1_cols.len() {
                let v = get_f32_or_nan(row, idx)?;
                feats.push(v);
                idx += 1;
            }
        }
        Ok(FeatureRow {
            feature_id: feature_id as u64,
            precursor_id: precursor_id as u64,
            run_id: run_id as u64,
            group_id: format!("{run_id}_{precursor_id}"),
            exp_rt,
            is_decoy: decoy != 0,
            features: feats,
        })
    })?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }

    let mut feature_cols_out: Vec<String> = Vec::new();
    feature_cols_out.extend(feature_cols.iter().map(|c| c.to_lowercase()));
    if cfg.level == OswLevel::Ms1Ms2 {
        feature_cols_out.extend(
            ms1_cols
                .iter()
                .map(|c| format!("var_ms1_{}", c.trim_start_matches("VAR_").to_lowercase())),
        );
    }

    Ok(OswFeatureTable {
        rows: out,
        feature_cols: feature_cols_out,
    })
}

#[cfg(feature = "sqlite")]
fn read_ms1_features(conn: &rusqlite::Connection, _cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    use rusqlite::Row;

    let mut feature_cols = list_var_columns(conn, "FEATURE_MS1")?;
    feature_cols.sort();
    let (feature_cols, dropped) = filter_all_null_columns(conn, "FEATURE_MS1", feature_cols)?;
    if !dropped.is_empty() {
        eprintln!(
            "warning: dropping all-NULL FEATURE_MS1 columns: {}",
            dropped.join(", ")
        );
    }

    let mut select_cols: Vec<String> = Vec::new();
    select_cols.push("fm.FEATURE_ID".to_string());
    select_cols.push("f.RUN_ID".to_string());
    select_cols.push("f.PRECURSOR_ID".to_string());
    select_cols.push("f.EXP_RT".to_string());
    select_cols.push("p.DECOY".to_string());
    for c in &feature_cols {
        select_cols.push(format!("fm.{c}"));
    }
    let select = select_cols.join(", ");

    let sql = format!(
        "SELECT {select}
         FROM FEATURE_MS1 fm
         INNER JOIN FEATURE f ON fm.FEATURE_ID = f.ID
         INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
         ORDER BY f.RUN_ID, p.ID, f.EXP_RT"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row: &Row| {
        let feature_id: i64 = row.get(0)?;
        let run_id: i64 = row.get(1)?;
        let precursor_id: i64 = row.get(2)?;
        let exp_rt: f32 = row.get(3)?;
        let decoy: i32 = row.get(4)?;

        let mut idx = 5usize;
        let mut feats = Vec::new();
        for _ in 0..feature_cols.len() {
            let v = get_f32_or_nan(row, idx)?;
            feats.push(v);
            idx += 1;
        }

        Ok(FeatureRow {
            feature_id: feature_id as u64,
            precursor_id: precursor_id as u64,
            run_id: run_id as u64,
            group_id: format!("{run_id}_{precursor_id}"),
            exp_rt,
            is_decoy: decoy != 0,
            features: feats,
        })
    })?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }

    let feature_cols_out = feature_cols.iter().map(|c| c.to_lowercase()).collect();

    Ok(OswFeatureTable {
        rows: out,
        feature_cols: feature_cols_out,
    })
}

#[cfg(feature = "sqlite")]
fn read_transition_features(
    conn: &rusqlite::Connection,
    cfg: &OswReadConfig,
) -> Result<OswFeatureTable> {
    use rusqlite::Row;

    let mut feature_cols = list_var_columns(conn, "FEATURE_TRANSITION")?;
    feature_cols.sort();
    let (feature_cols, dropped) =
        filter_all_null_columns(conn, "FEATURE_TRANSITION", feature_cols)?;
    if !dropped.is_empty() {
        eprintln!(
            "warning: dropping all-NULL FEATURE_TRANSITION columns: {}",
            dropped.join(", ")
        );
    }

    let mut select_cols: Vec<String> = Vec::new();
    select_cols.push("ft.FEATURE_ID".to_string());
    select_cols.push("ft.TRANSITION_ID".to_string());
    select_cols.push("f.RUN_ID".to_string());
    select_cols.push("f.PRECURSOR_ID".to_string());
    select_cols.push("f.EXP_RT".to_string());
    select_cols.push("t.DECOY".to_string());
    for c in &feature_cols {
        select_cols.push(format!("ft.{c}"));
    }
    let select = select_cols.join(", ");

    let sql = format!(
        "SELECT {select}
         FROM FEATURE_TRANSITION ft
         INNER JOIN FEATURE f ON ft.FEATURE_ID = f.ID
         INNER JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID
         INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
         INNER JOIN TRANSITION t ON ft.TRANSITION_ID = t.ID
         WHERE s.RANK <= {rank}
           AND s.PEP <= {pep}
           AND ft.VAR_ISOTOPE_OVERLAP_SCORE <= {iso}
           AND ft.VAR_LOG_SN_SCORE > {sn}
           AND p.DECOY = 0
         ORDER BY f.RUN_ID, f.PRECURSOR_ID, f.EXP_RT, ft.TRANSITION_ID",
        rank = cfg.ipf_max_rank,
        pep = cfg.ipf_max_pep,
        iso = cfg.ipf_max_transition_isotope_overlap,
        sn = cfg.ipf_min_transition_sn,
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row: &Row| {
        let feature_id: i64 = row.get(0)?;
        let transition_id: i64 = row.get(1)?;
        let run_id: i64 = row.get(2)?;
        let precursor_id: i64 = row.get(3)?;
        let exp_rt: f32 = row.get(4)?;
        let decoy: i32 = row.get(5)?;

        let mut idx = 6usize;
        let mut feats = Vec::new();
        for _ in 0..feature_cols.len() {
            let v = get_f32_or_nan(row, idx)?;
            feats.push(v);
            idx += 1;
        }

        Ok(FeatureRow {
            feature_id: feature_id as u64,
            precursor_id: precursor_id as u64,
            run_id: run_id as u64,
            group_id: format!("{run_id}_{feature_id}_{precursor_id}_{transition_id}"),
            exp_rt,
            is_decoy: decoy != 0,
            features: feats,
        })
    })?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }

    let feature_cols_out = feature_cols.iter().map(|c| c.to_lowercase()).collect();

    Ok(OswFeatureTable {
        rows: out,
        feature_cols: feature_cols_out,
    })
}

#[cfg(not(feature = "sqlite"))]
/// Stub used when `redeem-io` is built without SQLite support.
pub fn write_score_table(_path: &std::path::Path, _table: &str, _rows: &[ScoreRow]) -> Result<()> {
    bail!("redeem-io compiled without feature `sqlite`")
}
