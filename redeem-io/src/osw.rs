// redeem-io/src/osw.rs

use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub feature_id: u64,
    pub precursor_id: u64,
    pub run_id: u64,
    pub group_id: String,
    pub exp_rt: f32,
    pub is_decoy: bool,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ScoreRow {
    pub feature_id: u64,
    pub score: f32,
    pub rank: i32,
    pub pvalue: f32,
    pub qvalue: f32,
    pub pep: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OswLevel {
    Ms2,
    Ms1,
    Ms1Ms2,
    Transition,
    Alignment,
}

#[derive(Debug, Clone)]
pub struct OswReadConfig {
    pub level: OswLevel,
    pub ipf_max_rank: i32,
    pub ipf_max_pep: f32,
    pub ipf_max_transition_isotope_overlap: f32,
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

#[derive(Debug, Clone)]
pub struct OswFeatureTable {
    pub rows: Vec<FeatureRow>,
    pub feature_cols: Vec<String>,
}

#[cfg(feature = "sqlite")]
pub fn read_feature_rows(path: &std::path::Path, cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    use rusqlite::{Connection, Row};

    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    match cfg.level {
        OswLevel::Ms2 | OswLevel::Ms1Ms2 => read_ms2_features(&conn, cfg),
        OswLevel::Ms1 => read_ms1_features(&conn, cfg),
        OswLevel::Transition => read_transition_features(&conn, cfg),
        OswLevel::Alignment => bail!("alignment-level OSW read not implemented yet"),
    }
}

#[cfg(not(feature = "sqlite"))]
pub fn read_feature_rows(_path: &std::path::Path, _cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    bail!("redeem-io compiled without feature `sqlite`")
}

#[cfg(feature = "sqlite")]
pub fn write_score_table(
    path: &std::path::Path,
    table: &str,
    rows: &[ScoreRow],
) -> Result<()> {
    use rusqlite::{params, Connection};

    let conn = Connection::open(path)?;
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
    let mut stmt = tx.prepare(&sql)?;
    for r in rows {
        stmt.execute(params![r.feature_id, r.score, r.rank, r.pvalue, r.qvalue, r.pep])?;
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
fn read_ms2_features(conn: &rusqlite::Connection, cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    use rusqlite::Row;

    let mut feature_cols = list_var_columns(conn, "FEATURE_MS2")?;
    feature_cols.sort();

    let mut ms1_cols: Vec<String> = Vec::new();
    if cfg.level == OswLevel::Ms1Ms2 {
        ms1_cols = list_var_columns(conn, "FEATURE_MS1")?;
        ms1_cols.sort();
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
            let v: f32 = row.get(idx)?;
            feats.push(v);
            idx += 1;
        }
        if cfg.level == OswLevel::Ms1Ms2 {
            for _ in 0..ms1_cols.len() {
                let v: f32 = row.get(idx)?;
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

    Ok(OswFeatureTable { rows: out, feature_cols: feature_cols_out })
}

#[cfg(feature = "sqlite")]
fn read_ms1_features(conn: &rusqlite::Connection, cfg: &OswReadConfig) -> Result<OswFeatureTable> {
    use rusqlite::Row;

    let mut feature_cols = list_var_columns(conn, "FEATURE_MS1")?;
    feature_cols.sort();

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
            let v: f32 = row.get(idx)?;
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

    Ok(OswFeatureTable { rows: out, feature_cols: feature_cols_out })
}

#[cfg(feature = "sqlite")]
fn read_transition_features(
    conn: &rusqlite::Connection,
    cfg: &OswReadConfig,
) -> Result<OswFeatureTable> {
    use rusqlite::Row;

    let mut feature_cols = list_var_columns(conn, "FEATURE_TRANSITION")?;
    feature_cols.sort();

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
            let v: f32 = row.get(idx)?;
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

    Ok(OswFeatureTable { rows: out, feature_cols: feature_cols_out })
}

#[cfg(not(feature = "sqlite"))]
pub fn write_score_table(
    _path: &std::path::Path,
    _table: &str,
    _rows: &[ScoreRow],
) -> Result<()> {
    bail!("redeem-io compiled without feature `sqlite`")
}
