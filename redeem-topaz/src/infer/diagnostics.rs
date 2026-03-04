use anyhow::Result;
use std::collections::HashMap;

use crate::io::osw::FeatureRow;
use crate::io::xic::{XicSource, TransitionTrace};

#[derive(Debug, Clone)]
pub struct TraceSummary {
    pub n: usize,
    pub l: usize,
    pub ms1_cmax: usize,
    pub ms2_cmax: usize,
    pub ms1_nonzero_rows: usize,
    pub ms2_nonzero_rows: usize,
}

/// Summarize trace tensor content for debugging.
///
/// x is flattened (N, C_total, L) in row-major order.
pub fn trace_summary(
    x: &[f32],
    n: usize,
    c_total: usize,
    l: usize,
    ms1_cmax: usize,
    ms2_cmax: usize,
) -> TraceSummary {
    let mut ms1_nonzero = 0usize;
    let mut ms2_nonzero = 0usize;

    if n == 0 || c_total == 0 || l == 0 {
        return TraceSummary {
            n,
            l,
            ms1_cmax,
            ms2_cmax,
            ms1_nonzero_rows: 0,
            ms2_nonzero_rows: 0,
        };
    }

    for i in 0..n {
        let row_off = i * c_total * l;
        if ms1_cmax > 0 {
            let mut max1 = 0f32;
            for c in 0..ms1_cmax.min(c_total) {
                let off = row_off + c * l;
                for t in 0..l {
                    max1 = max1.max(x[off + t].abs());
                }
            }
            if max1 > 0.0 {
                ms1_nonzero += 1;
            }
        }
        let mut max2 = 0f32;
        let ms2_start = ms1_cmax.min(c_total);
        let ms2_end = ms2_start + ms2_cmax.min(c_total.saturating_sub(ms2_start));
        for c in ms2_start..ms2_end {
            let off = row_off + c * l;
            for t in 0..l {
                max2 = max2.max(x[off + t].abs());
            }
        }
        if max2 > 0.0 {
            ms2_nonzero += 1;
        }
    }

    TraceSummary {
        n,
        l,
        ms1_cmax,
        ms2_cmax,
        ms1_nonzero_rows: ms1_nonzero,
        ms2_nonzero_rows: ms2_nonzero,
    }
}

pub fn print_trace_summary(summary: &TraceSummary, label: &str) {
    if summary.n == 0 || summary.l == 0 {
        log::info!("Trace summary ({label}): empty");
        return;
    }
    if summary.ms1_cmax > 0 {
        log::info!(
            "Trace summary ({label}): N={} L={} | MS1 C={} nonzero_rows={} | MS2 C={} nonzero_rows={}",
            summary.n,
            summary.l,
            summary.ms1_cmax,
            summary.ms1_nonzero_rows,
            summary.ms2_cmax,
            summary.ms2_nonzero_rows
        );
    } else {
        log::info!(
            "Trace summary ({label}): N={} L={} | C={} nonzero_rows={}",
            summary.n,
            summary.l,
            summary.ms2_cmax,
            summary.ms2_nonzero_rows
        );
    }
}

pub fn warn_if_missing_ms1(summary: &TraceSummary, context: &str) {
    if summary.ms1_cmax == 0 {
        return;
    }
    if summary.ms1_nonzero_rows == 0 {
        log::warn!(
            "ms1_cmax={} but no MS1 traces found in {}; using zero-padded MS1 channels.",
            summary.ms1_cmax, context
        );
    }
}

fn count_ms1_rows(transitions: &[TransitionTrace]) -> (usize, usize) {
    let mut n_ms1 = 0usize;
    let mut n_all = 0usize;
    for t in transitions {
        n_all += 1;
        if let Some(level) = t.ms_level {
            if level == 1 {
                n_ms1 += 1;
            }
        } else {
            let a = t.annotation.to_ascii_lowercase();
            if a.starts_with("precursor") || a.starts_with("ms1") {
                n_ms1 += 1;
            }
        }
    }
    (n_ms1, n_all)
}

/// Probe XIC source to see if MS1 traces appear to be present.
pub fn probe_ms1_presence(
    xic: &mut impl XicSource,
    rows: &[FeatureRow],
    n_precursors: usize,
) -> Result<(bool, usize, usize)> {
    let mut by_run: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut seen = HashMap::<(u64, u64), bool>::new();
    for r in rows {
        if seen.len() >= n_precursors.max(1) {
            break;
        }
        let key = (r.run_id, r.precursor_id);
        if seen.contains_key(&key) {
            continue;
        }
        seen.insert(key, true);
        by_run.entry(r.run_id).or_default().push(r.precursor_id);
    }

    let mut n_ms1 = 0usize;
    let mut n_all = 0usize;
    for (run_id, precs) in by_run {
        if precs.is_empty() {
            continue;
        }
        let fetched = xic.fetch_precursors(run_id, &precs)?;
        for x in fetched {
            let (m, a) = count_ms1_rows(&x.transitions);
            n_ms1 += m;
            n_all += a;
        }
    }
    Ok((n_ms1 > 0, n_ms1, n_all))
}

#[cfg(feature = "io-sqlite")]
use rusqlite::Connection;
#[cfg(feature = "io-sqlite")]
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Rank1DisagreementSummary {
    pub rows: usize,
    pub pstc_cutoff: Option<f32>,
    pub ms2_cutoff: Option<f32>,
    pub pstc_targets: usize,
    pub pstc_decoys: usize,
    pub ms2_targets: usize,
    pub ms2_decoys: usize,
    pub quad_both: (usize, usize, usize),
    pub quad_pstc_only: (usize, usize, usize),
    pub quad_ms2_only: (usize, usize, usize),
    pub quad_neither: (usize, usize, usize),
}

#[cfg(feature = "io-sqlite")]
fn cutoff_from_table(conn: &Connection, table: &str, q: f32) -> Result<Option<f32>> {
    let sql = format!(
        "SELECT MIN(SCORE) AS CUTOFF FROM {table} WHERE RANK=1 AND QVALUE <= ?"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([q])?;
    if let Some(row) = rows.next()? {
        let v: Option<f32> = row.get(0)?;
        Ok(v)
    } else {
        Ok(None)
    }
}

#[cfg(feature = "io-sqlite")]
fn counts_for_table(conn: &Connection, table: &str, q: f32) -> Result<(usize, usize)> {
    let sql = format!(
        "SELECT pr.DECOY AS DECOY
         FROM {table} s
         JOIN FEATURE f    ON f.ID = s.FEATURE_ID
         JOIN PRECURSOR pr ON pr.ID = f.PRECURSOR_ID
         WHERE s.RANK=1 AND s.QVALUE <= ?"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([q])?;
    let mut n_t = 0usize;
    let mut n_d = 0usize;
    while let Some(row) = rows.next()? {
        let decoy: i64 = row.get(0)?;
        if decoy == 1 {
            n_d += 1;
        } else {
            n_t += 1;
        }
    }
    Ok((n_t, n_d))
}

#[cfg(feature = "io-sqlite")]
#[derive(Debug)]
struct Rank1Row {
    run_id: u64,
    precursor_id: u64,
    pstc_score: f32,
    pstc_q: f32,
    ms2_score: f32,
    ms2_q: f32,
    decoy: bool,
}

#[cfg(feature = "io-sqlite")]
fn load_rank1_join(conn: &Connection, pstc_table: &str) -> Result<Vec<Rank1Row>> {
    let sql = format!(
        r#"
    WITH pstc AS (
      SELECT
        sp.FEATURE_ID AS PSTC_FEATURE_ID,
        sp.SCORE      AS PSTC_SCORE,
        sp.QVALUE     AS PSTC_QVALUE,
        f.RUN_ID      AS RUN_ID,
        f.PRECURSOR_ID AS PRECURSOR_ID,
        pr.DECOY      AS DECOY
      FROM {pstc_table} sp
      JOIN FEATURE f    ON f.ID = sp.FEATURE_ID
      JOIN PRECURSOR pr ON pr.ID = f.PRECURSOR_ID
      WHERE sp.RANK = 1
    ),
    ms2 AS (
      SELECT
        sm.FEATURE_ID AS MS2_FEATURE_ID,
        sm.SCORE      AS MS2_SCORE,
        sm.QVALUE     AS MS2_QVALUE,
        f.RUN_ID      AS RUN_ID,
        f.PRECURSOR_ID AS PRECURSOR_ID,
        pr.DECOY      AS DECOY
      FROM SCORE_MS2 sm
      JOIN FEATURE f    ON f.ID = sm.FEATURE_ID
      JOIN PRECURSOR pr ON pr.ID = f.PRECURSOR_ID
      WHERE sm.RANK = 1
    )
    SELECT
      pstc.RUN_ID,
      pstc.PRECURSOR_ID,
      pstc.PSTC_SCORE,
      pstc.PSTC_QVALUE,
      ms2.MS2_SCORE,
      ms2.MS2_QVALUE,
      pstc.DECOY
    FROM pstc
    JOIN ms2
      ON ms2.RUN_ID = pstc.RUN_ID
     AND ms2.PRECURSOR_ID = pstc.PRECURSOR_ID
    "#
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let run_id: i64 = r.get(0)?;
        let precursor_id: i64 = r.get(1)?;
        let pstc_score: f32 = r.get(2)?;
        let pstc_q: f32 = r.get(3)?;
        let ms2_score: f32 = r.get(4)?;
        let ms2_q: f32 = r.get(5)?;
        let decoy: i64 = r.get(6)?;
        out.push(Rank1Row {
            run_id: run_id as u64,
            precursor_id: precursor_id as u64,
            pstc_score,
            pstc_q,
            ms2_score,
            ms2_q,
            decoy: decoy == 1,
        });
    }
    Ok(out)
}

#[cfg(feature = "io-sqlite")]
fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

#[cfg(feature = "io-sqlite")]
/// Write rank-1 disagreement TSVs (no plots).
pub fn write_rank1_disagreement_tsvs(
    osw_path: &Path,
    pstc_table: &str,
    q: f32,
    outdir: &Path,
) -> Result<Rank1DisagreementSummary> {
    std::fs::create_dir_all(outdir)?;

    let conn = Connection::open(osw_path)?;
    let rows = load_rank1_join(&conn, pstc_table)?;
    if rows.is_empty() {
        log::warn!("rank1 disagreement analysis found no joined rows");
        return Ok(Rank1DisagreementSummary {
            rows: 0,
            pstc_cutoff: None,
            ms2_cutoff: None,
            pstc_targets: 0,
            pstc_decoys: 0,
            ms2_targets: 0,
            ms2_decoys: 0,
            quad_both: (0, 0, 0),
            quad_pstc_only: (0, 0, 0),
            quad_ms2_only: (0, 0, 0),
            quad_neither: (0, 0, 0),
        });
    }

    let pstc_cut = cutoff_from_table(&conn, pstc_table, q)?;
    let ms2_cut = cutoff_from_table(&conn, "SCORE_MS2", q)?;

    let (pstc_nt, pstc_nd) = counts_for_table(&conn, pstc_table, q)?;
    let (ms2_nt, ms2_nd) = counts_for_table(&conn, "SCORE_MS2", q)?;

    let mut by_run: HashMap<u64, [usize; 4]> = HashMap::new(); // both, pstc_only, ms2_only, neither
    let mut margins: HashMap<&'static str, Vec<(f32, f32)>> = HashMap::new();
    margins.insert("both", Vec::new());
    margins.insert("pstc_only", Vec::new());
    margins.insert("ms2_only", Vec::new());
    margins.insert("neither", Vec::new());

    let mut quad = |cat: &str, decoy: bool, run_id: u64, pstc_margin: f32, ms2_margin: f32| {
        if !decoy {
            let e = by_run.entry(run_id).or_insert([0, 0, 0, 0]);
            match cat {
                "both" => e[0] += 1,
                "pstc_only" => e[1] += 1,
                "ms2_only" => e[2] += 1,
                "neither" => e[3] += 1,
                _ => {}
            }
        }
        if !decoy {
            if let Some(v) = margins.get_mut(cat) {
                v.push((pstc_margin, ms2_margin));
            }
        }
    };

    let mut qb = (0usize, 0usize, 0usize);
    let mut qp = (0usize, 0usize, 0usize);
    let mut qm = (0usize, 0usize, 0usize);
    let mut qn = (0usize, 0usize, 0usize);

    for r in &rows {
        let pass_pstc = r.pstc_q <= q;
        let pass_ms2 = r.ms2_q <= q;
        let cat = if pass_pstc && pass_ms2 {
            "both"
        } else if pass_pstc && !pass_ms2 {
            "pstc_only"
        } else if !pass_pstc && pass_ms2 {
            "ms2_only"
        } else {
            "neither"
        };
        let pstc_margin = if let Some(c) = pstc_cut { r.pstc_score - c } else { f32::NAN };
        let ms2_margin = if let Some(c) = ms2_cut { r.ms2_score - c } else { f32::NAN };
        quad(cat, r.decoy, r.run_id, pstc_margin, ms2_margin);

        let tgt = if r.decoy { 0usize } else { 1usize };
        match cat {
            "both" => qb = (qb.0 + 1, qb.1 + tgt, qb.2 + if r.decoy { 1 } else { 0 }),
            "pstc_only" => qp = (qp.0 + 1, qp.1 + tgt, qp.2 + if r.decoy { 1 } else { 0 }),
            "ms2_only" => qm = (qm.0 + 1, qm.1 + tgt, qm.2 + if r.decoy { 1 } else { 0 }),
            _ => qn = (qn.0 + 1, qn.1 + tgt, qn.2 + if r.decoy { 1 } else { 0 }),
        }
    }

    // write by-run TSV
    let mut by_run_rows: Vec<(u64, [usize; 4], usize)> = by_run
        .iter()
        .map(|(&run, counts)| {
            let total = counts[0] + counts[1] + counts[2] + counts[3];
            (run, *counts, total)
        })
        .collect();
    by_run_rows.sort_by(|a, b| b.2.cmp(&a.2));

    let by_run_path = outdir.join("rank1_disagreement_by_run.tsv");
    let mut out = String::new();
    out.push_str("RUN_ID\tboth\tpstc_only\tms2_only\tneither\ttotal_targets\n");
    for (run, counts, total) in &by_run_rows {
        out.push_str(&format!(
            "{run}\t{}\t{}\t{}\t{}\t{total}\n",
            counts[0], counts[1], counts[2], counts[3]
        ));
    }
    std::fs::write(&by_run_path, out)?;

    // margin summary TSV
    let margin_path = outdir.join("rank1_disagreement_margin_summary.tsv");
    let mut out = String::new();
    out.push_str("cat\tn_targets\tpstc_margin_median\tms2_margin_median\tpstc_margin_mean\tms2_margin_mean\n");

    for cat in ["both", "pstc_only", "ms2_only", "neither"] {
        let mut v = margins.remove(cat).unwrap_or_default();
        let n_targets = v.len();
        let (mut pstc_vals, mut ms2_vals): (Vec<f32>, Vec<f32>) =
            v.drain(..).unzip();
        let pstc_med = median(&mut pstc_vals);
        let ms2_med = median(&mut ms2_vals);
        let pstc_mean = if pstc_vals.is_empty() {
            f32::NAN
        } else {
            pstc_vals.iter().copied().sum::<f32>() / pstc_vals.len() as f32
        };
        let ms2_mean = if ms2_vals.is_empty() {
            f32::NAN
        } else {
            ms2_vals.iter().copied().sum::<f32>() / ms2_vals.len() as f32
        };
        out.push_str(&format!(
            "{cat}\t{n_targets}\t{pstc_med}\t{ms2_med}\t{pstc_mean}\t{ms2_mean}\n"
        ));
    }
    std::fs::write(&margin_path, out)?;

    if !by_run_rows.is_empty() {
        let cap = by_run_rows.len().min(10);
        log::info!(
            "Rank-1 disagreement targets by run (top {cap} by total_targets):"
        );
        for (run, counts, _total) in by_run_rows.iter().take(cap) {
            log::info!(
                "  run {run}: both={} pstc_only={} ms2_only={} neither={}",
                counts[0], counts[1], counts[2], counts[3]
            );
        }
    }

    Ok(Rank1DisagreementSummary {
        rows: rows.len(),
        pstc_cutoff: pstc_cut,
        ms2_cutoff: ms2_cut,
        pstc_targets: pstc_nt,
        pstc_decoys: pstc_nd,
        ms2_targets: ms2_nt,
        ms2_decoys: ms2_nd,
        quad_both: qb,
        quad_pstc_only: qp,
        quad_ms2_only: qm,
        quad_neither: qn,
    })
}
