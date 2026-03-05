use anyhow::{bail, Result};
use plotly::common::{
    DashType, HoverInfo, Line, Marker, Mode, Orientation, Pattern, PatternShape,
};
use plotly::layout::{Axis, AxisType, BarMode};
use plotly::{Bar, Histogram, Layout, Plot, Scatter};
use rand::prelude::*;
use report_builder::{Report, ReportSection};
use redeem_topaz::infer::stats::tdc_summary;
use std::path::Path;

const DEFAULT_PCA_MAX_ROWS: usize = 50_000;
const POWER_ITERS: usize = 50;

pub fn write_topaz_report(
    head_embeddings_path: &Path,
    report_path: &Path,
    seed: u64,
    osw_path: Option<&Path>,
    score_tsv_path: Option<&Path>,
) -> Result<()> {
    let emb = load_head_embeddings_tsv(head_embeddings_path)?;
    if emb.hidden_dim == 0 || emb.n == 0 {
        bail!("head embeddings are empty: {:?}", head_embeddings_path);
    }

    let precursor_meta = if let Some(path) = osw_path {
        redeem_topaz::io::osw::read_precursor_meta(path).ok()
    } else {
        None
    };
    let feature_meta = if let Some(path) = osw_path {
        redeem_topaz::io::osw::read_feature_meta(path).ok()
    } else {
        None
    };

    let topaz_scores = score_tsv_path
        .and_then(|p| load_score_tsv(p).ok())
        .filter(|rows| !rows.is_empty());
    let ms2_scores = osw_path
        .and_then(|p| redeem_topaz::io::osw::read_score_table(p, "SCORE_MS2").ok())
        .filter(|rows| !rows.is_empty())
        .map(|rows| rows.into_iter().map(ScoreLite::from_ms2).collect::<Vec<_>>());

    let pca = pca2(&emb.hidden, emb.n, emb.hidden_dim, DEFAULT_PCA_MAX_ROWS, seed);
    let bag_scores_f32: Vec<f32> = emb.bag_score.iter().map(|&v| v as f32).collect();
    let tdc = tdc_summary(&bag_scores_f32, &emb.is_decoy, 0.01);
    let cutoff = if tdc.cutoff.is_finite() { Some(tdc.cutoff as f64) } else { None };

    let mut report = Report::new(
        "ReDeeM TOPAZ Report",
        "1",
        None,
        "TOPAZ training diagnostics",
    );

    let mut section = ReportSection::new("Embeddings");
    section.add_plot(plot_embedding_with_marginal_hist(
        &pca,
        &emb.bag_score,
        &emb.is_decoy,
        &emb.bag_pid,
        precursor_meta.as_ref(),
        cutoff,
    ));
    report.add_section(section);

    if let (Some(scores), Some(meta)) = (topaz_scores.as_ref(), feature_meta.as_ref()) {
        if let Some(topaz_ids) = compute_id_counts(scores, meta, 0.01) {
            let ms2_ids = ms2_scores
                .as_ref()
                .and_then(|rows| compute_id_counts(rows, meta, 0.01));
            let mut id_section = ReportSection::new("Identifications");
            id_section.add_plot(plot_id_bars(&topaz_ids, ms2_ids.as_ref()));
            report.add_section(id_section);
        }
    }

    if let (Some(scores), Some(meta), Some(ms2)) =
        (topaz_scores.as_ref(), feature_meta.as_ref(), ms2_scores.as_ref())
    {
        let pairs = build_score_pairs(scores, ms2, meta, precursor_meta.as_ref());
        if !pairs.is_empty() {
            let topaz_cutoff = cutoff_from_score_rows(scores, 0.01);
            let ms2_cutoff = cutoff_from_score_rows(ms2, 0.01);
            let mut sec = ReportSection::new("TOPAZ vs SCORE_MS2");
            sec.add_plot(plot_score_scatter_with_marginals(
                &pairs,
                topaz_cutoff,
                ms2_cutoff,
            ));
            report.add_section(sec);
        }
    }

    report.save_to_file(&report_path.to_string_lossy().to_string())?;
    Ok(())
}

struct HeadEmbeddings {
    n: usize,
    hidden_dim: usize,
    hidden: Vec<f64>,
    bag_score: Vec<f64>,
    is_decoy: Vec<bool>,
    bag_pid: Vec<String>,
}

#[derive(Debug, Clone)]
struct ScoreLite {
    feature_id: u64,
    score: f64,
    rank: Option<i32>,
    qvalue: Option<f64>,
}

impl ScoreLite {
    fn from_ms2(entry: redeem_topaz::io::osw::ScoreTableEntry) -> Self {
        Self {
            feature_id: entry.feature_id,
            score: entry.score as f64,
            rank: entry.rank,
            qvalue: entry.qvalue.map(|v| v as f64),
        }
    }
}

#[derive(Debug, Clone)]
struct IdCounts {
    per_run: std::collections::HashMap<u64, usize>,
    union: usize,
}

fn load_head_embeddings_tsv(path: &Path) -> Result<HeadEmbeddings> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .flexible(true)
        .from_path(path)?;
    let headers = rdr.headers()?.clone();

    let idx_score = headers
        .iter()
        .position(|h| h == "bag_score")
        .ok_or_else(|| anyhow::anyhow!("missing bag_score column in {:?}", path))?;
    let idx_decoy = headers
        .iter()
        .position(|h| h == "is_decoy")
        .ok_or_else(|| anyhow::anyhow!("missing is_decoy column in {:?}", path))?;
    let idx_pid = headers
        .iter()
        .position(|h| h == "bag_pid")
        .ok_or_else(|| anyhow::anyhow!("missing bag_pid column in {:?}", path))?;
    let mut win_idxs = Vec::new();
    for (i, h) in headers.iter().enumerate() {
        if h.starts_with("win_hidden_") {
            win_idxs.push(i);
        }
    }
    if win_idxs.is_empty() {
        bail!("no win_hidden_* columns found in {:?}", path);
    }

    let hidden_dim = win_idxs.len();
    let mut hidden = Vec::new();
    let mut bag_score = Vec::new();
    let mut is_decoy = Vec::new();
    let mut bag_pid = Vec::new();

    for rec in rdr.records() {
        let rec = rec?;
        let score = rec.get(idx_score).unwrap_or("0").parse::<f64>().unwrap_or(0.0);
        let decoy = rec.get(idx_decoy).unwrap_or("0") == "1";
        let pid = rec.get(idx_pid).unwrap_or("").to_string();
        bag_score.push(score);
        is_decoy.push(decoy);
        bag_pid.push(pid);
        for &idx in &win_idxs {
            let v = rec.get(idx).unwrap_or("0").parse::<f64>().unwrap_or(0.0);
            hidden.push(v);
        }
    }

    let n = bag_score.len();
    Ok(HeadEmbeddings {
        n,
        hidden_dim,
        hidden,
        bag_score,
        is_decoy,
        bag_pid,
    })
}

fn load_score_tsv(path: &Path) -> Result<Vec<ScoreLite>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .flexible(true)
        .from_path(path)?;
    let headers = rdr.headers()?.clone();
    let idx_feat = headers
        .iter()
        .position(|h| h == "FEATURE_ID")
        .ok_or_else(|| anyhow::anyhow!("missing FEATURE_ID in {:?}", path))?;
    let idx_score = headers
        .iter()
        .position(|h| h == "SCORE")
        .ok_or_else(|| anyhow::anyhow!("missing SCORE in {:?}", path))?;
    let idx_rank = headers.iter().position(|h| h == "RANK");
    let idx_q = headers.iter().position(|h| h == "QVALUE");

    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let fid = rec
            .get(idx_feat)
            .unwrap_or("0")
            .parse::<u64>()
            .unwrap_or(0);
        let score = rec
            .get(idx_score)
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);
        let rank = idx_rank
            .and_then(|i| rec.get(i))
            .and_then(|v| v.parse::<i32>().ok());
        let qvalue = idx_q
            .and_then(|i| rec.get(i))
            .and_then(|v| v.parse::<f64>().ok());
        out.push(ScoreLite {
            feature_id: fid,
            score,
            rank,
            qvalue,
        });
    }
    Ok(out)
}

fn compute_id_counts(
    rows: &[ScoreLite],
    meta: &std::collections::HashMap<u64, redeem_topaz::io::osw::FeatureMeta>,
    q_cut: f64,
) -> Option<IdCounts> {
    if rows.iter().all(|r| r.qvalue.is_none()) {
        return None;
    }
    let mut per_run: std::collections::HashMap<u64, std::collections::HashSet<u64>> =
        std::collections::HashMap::new();
    let mut union: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for row in rows {
        if let Some(q) = row.qvalue {
            if q > q_cut {
                continue;
            }
        } else {
            continue;
        }
        if let Some(rank) = row.rank {
            if rank != 1 {
                continue;
            }
        }
        let m = match meta.get(&row.feature_id) {
            Some(m) => m,
            None => continue,
        };
        if m.is_decoy {
            continue;
        }
        per_run
            .entry(m.run_id)
            .or_insert_with(std::collections::HashSet::new)
            .insert(m.precursor_id);
        union.insert(m.precursor_id);
    }
    let mut per_run_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for (run, set) in per_run {
        per_run_counts.insert(run, set.len());
    }
    Some(IdCounts {
        per_run: per_run_counts,
        union: union.len(),
    })
}

fn cutoff_from_score_rows(rows: &[ScoreLite], q_cut: f64) -> Option<f64> {
    rows.iter()
        .filter(|row| row.rank.unwrap_or(1) == 1)
        .filter_map(|row| match row.qvalue {
            Some(q) if q <= q_cut => Some(row.score),
            _ => None,
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn pca2(hidden: &[f64], n: usize, d: usize, max_rows: usize, seed: u64) -> Vec<(f64, f64)> {
    let m = n.min(max_rows).max(1);
    let mut idx: Vec<usize> = (0..n).collect();
    if n > m {
        let mut rng = StdRng::seed_from_u64(seed);
        idx = rand::seq::index::sample(&mut rng, n, m).into_vec();
    }

    let mut mean = vec![0.0f64; d];
    for &i in &idx {
        let row = &hidden[i * d..(i + 1) * d];
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    let inv_m = 1.0 / (idx.len() as f64);
    for j in 0..d {
        mean[j] *= inv_m;
    }

    let mut cov = vec![0.0f64; d * d];
    for &i in &idx {
        let row = &hidden[i * d..(i + 1) * d];
        for a in 0..d {
            let xa = row[a] - mean[a];
            for b in 0..d {
                cov[a * d + b] += xa * (row[b] - mean[b]);
            }
        }
    }
    let denom = (idx.len().saturating_sub(1)).max(1) as f64;
    for v in &mut cov {
        *v /= denom;
    }

    let v1 = power_iteration(&cov, d, POWER_ITERS, seed.wrapping_add(1));
    let lambda1 = dot(&v1, &mat_vec(&cov, &v1, d));
    let mut cov2 = cov.clone();
    for i in 0..d {
        for j in 0..d {
            cov2[i * d + j] -= lambda1 * v1[i] * v1[j];
        }
    }
    let v2 = power_iteration(&cov2, d, POWER_ITERS, seed.wrapping_add(2));

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = &hidden[i * d..(i + 1) * d];
        let mut pc1 = 0.0;
        let mut pc2 = 0.0;
        for j in 0..d {
            let xc = row[j] - mean[j];
            pc1 += xc * v1[j];
            pc2 += xc * v2[j];
        }
        out.push((pc1, pc2));
    }
    out
}

fn power_iteration(cov: &[f64], d: usize, iters: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v = vec![0.0f64; d];
    for i in 0..d {
        v[i] = rng.gen_range(-0.5..0.5);
    }
    normalize(&mut v);
    for _ in 0..iters {
        let w = mat_vec(cov, &v, d);
        let mut v_next = w;
        normalize(&mut v_next);
        v = v_next;
    }
    v
}

fn mat_vec(mat: &[f64], v: &[f64], d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; d];
    for i in 0..d {
        let mut acc = 0.0;
        let row = &mat[i * d..(i + 1) * d];
        for j in 0..d {
            acc += row[j] * v[j];
        }
        out[i] = acc;
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn plot_embedding_with_marginal_hist(
    pca: &[(f64, f64)],
    bag_score: &[f64],
    is_decoy: &[bool],
    bag_pid: &[String],
    precursor_meta: Option<&std::collections::HashMap<u64, redeem_topaz::io::osw::PrecursorMeta>>,
    cutoff: Option<f64>,
) -> Plot {
    let mut t_x = Vec::new();
    let mut t_y = Vec::new();
    let mut t_hover = Vec::new();
    let mut d_x = Vec::new();
    let mut d_y = Vec::new();
    let mut d_hover = Vec::new();
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (i, ((_, y), decoy)) in pca.iter().zip(is_decoy.iter()).enumerate() {
        let score = bag_score[i];
        let pid = bag_pid.get(i).cloned().unwrap_or_default();
        let (run_id, prec_id) = parse_bag_pid(&pid);
        let pep = precursor_meta
            .and_then(|m| prec_id.and_then(|id| m.get(&id)))
            .map(|m| format!("{} / z={}", m.modified_sequence, m.charge))
            .unwrap_or_else(|| {
                if let Some(id) = prec_id {
                    format!("precursor_id={id}")
                } else {
                    pid.clone()
                }
            });
        let decoy_label = if *decoy { "decoy" } else { "target" };
        let hover = if let Some(r) = run_id {
            format!("{pep}<br>run_id={r}<br>{decoy_label}<br>score={score:.3}")
        } else {
            format!("{pep}<br>{decoy_label}<br>score={score:.3}")
        };
        if *decoy {
            d_x.push(score);
            d_y.push(*y);
            d_hover.push(hover);
        } else {
            t_x.push(score);
            t_y.push(*y);
            t_hover.push(hover);
        }
        if *y < y_min {
            y_min = *y;
        }
        if *y > y_max {
            y_max = *y;
        }
    }
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(t_x, t_y)
            .web_gl_mode(true)
            .name("Target")
            .mode(Mode::Markers)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.7)").size(4))
            .hover_info(HoverInfo::Text)
            .hover_text_array(t_hover),
    );
    plot.add_trace(
        Scatter::new(d_x, d_y)
            .web_gl_mode(true)
            .name("Decoy")
            .mode(Mode::Markers)
            .marker(Marker::new().color("rgba(214, 39, 40, 0.7)").size(4))
            .hover_info(HoverInfo::Text)
            .hover_text_array(d_hover),
    );
    if let Some(cut) = cutoff {
        let y0 = if y_min.is_finite() { y_min } else { 0.0 };
        let y1 = if y_max.is_finite() { y_max } else { 1.0 };
        plot.add_trace(
            Scatter::new(vec![cut, cut], vec![y0, y1])
                .name("Cutoff (TDC q=0.01)")
                .mode(Mode::Lines)
                .line(Line::new().color("rgba(0,0,0,0.8)").dash(DashType::Dash)),
        );
    }

    // Marginal histogram on top.
    let mut t_hist = Vec::new();
    let mut d_hist = Vec::new();
    for (s, decoy) in bag_score.iter().zip(is_decoy.iter()) {
        if *decoy {
            d_hist.push(*s);
        } else {
            t_hist.push(*s);
        }
    }
    plot.add_trace(
        Histogram::new(t_hist)
            .name("Target")
            .opacity(0.6)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.6)"))
            .y_axis("y2"),
    );
    plot.add_trace(
        Histogram::new(d_hist)
            .name("Decoy")
            .opacity(0.6)
            .marker(Marker::new().color("rgba(214, 39, 40, 0.6)"))
            .y_axis("y2"),
    );
    plot.set_layout(
        Layout::new()
            .title("Winner Hidden PCA2")
            .x_axis(
                Axis::new()
                    .title("PSTC bag score (max candidate logit)")
                    .domain(&[0.0, 1.0]),
            )
            .y_axis(
                Axis::new()
                    .title("Embedding PC2 (winner hidden)")
                    .domain(&[0.0, 0.78]),
            )
            .y_axis2(Axis::new().title("Count").domain(&[0.82, 1.0]).show_tick_labels(false))
            .bar_mode(BarMode::Overlay),
    );
    plot
}

#[derive(Debug, Clone)]
struct ScorePair {
    x: f64,
    y: f64,
    decoy: bool,
    hover: String,
}

fn build_score_pairs(
    topaz: &[ScoreLite],
    ms2: &[ScoreLite],
    meta: &std::collections::HashMap<u64, redeem_topaz::io::osw::FeatureMeta>,
    precursor_meta: Option<&std::collections::HashMap<u64, redeem_topaz::io::osw::PrecursorMeta>>,
) -> Vec<ScorePair> {
    let mut ms2_map: std::collections::HashMap<u64, &ScoreLite> = std::collections::HashMap::new();
    for r in ms2 {
        ms2_map.insert(r.feature_id, r);
    }
    let mut out = Vec::new();
    for r in topaz {
        let ms2_row = match ms2_map.get(&r.feature_id) {
            Some(v) => *v,
            None => continue,
        };
        if let Some(rank) = r.rank {
            if rank != 1 {
                continue;
            }
        }
        if let Some(rank) = ms2_row.rank {
            if rank != 1 {
                continue;
            }
        }
        let m = match meta.get(&r.feature_id) {
            Some(v) => v,
            None => continue,
        };
        let pep = precursor_meta
            .and_then(|pm| pm.get(&m.precursor_id))
            .map(|p| format!("{} / z={}", p.modified_sequence, p.charge))
            .unwrap_or_else(|| format!("precursor_id={}", m.precursor_id));
        let decoy_label = if m.is_decoy { "decoy" } else { "target" };
        let hover = format!(
            "{pep}<br>run_id={}<br>{decoy_label}<br>topaz={:.3}<br>ms2={:.3}",
            m.run_id, r.score, ms2_row.score
        );
        out.push(ScorePair {
            x: r.score,
            y: ms2_row.score,
            decoy: m.is_decoy,
            hover,
        });
    }
    out
}

fn plot_score_scatter_with_marginals(
    pairs: &[ScorePair],
    topaz_cutoff: Option<f64>,
    ms2_cutoff: Option<f64>,
) -> Plot {
    let mut t_x = Vec::new();
    let mut t_y = Vec::new();
    let mut t_hover = Vec::new();
    let mut d_x = Vec::new();
    let mut d_y = Vec::new();
    let mut d_hover = Vec::new();
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for p in pairs {
        if p.decoy {
            d_x.push(p.x);
            d_y.push(p.y);
            d_hover.push(p.hover.clone());
        } else {
            t_x.push(p.x);
            t_y.push(p.y);
            t_hover.push(p.hover.clone());
        }
        x_min = x_min.min(p.x);
        x_max = x_max.max(p.x);
        y_min = y_min.min(p.y);
        y_max = y_max.max(p.y);
    }

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(t_x.clone(), t_y.clone())
            .web_gl_mode(true)
            .name("Target")
            .mode(Mode::Markers)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.7)").size(4))
            .hover_info(HoverInfo::Text)
            .hover_text_array(t_hover),
    );
    plot.add_trace(
        Scatter::new(d_x.clone(), d_y.clone())
            .web_gl_mode(true)
            .name("Decoy")
            .mode(Mode::Markers)
            .marker(Marker::new().color("rgba(214, 39, 40, 0.7)").size(4))
            .hover_info(HoverInfo::Text)
            .hover_text_array(d_hover),
    );

    if let Some(cut) = topaz_cutoff {
        plot.add_trace(
            Scatter::new(
                vec![cut, cut],
                vec![
                    if y_min.is_finite() { y_min } else { 0.0 },
                    if y_max.is_finite() { y_max } else { 1.0 },
                ],
            )
            .name("TOPAZ cutoff (1% FDR)")
            .mode(Mode::Lines)
            .line(
                Line::new()
                    .color("rgba(31, 119, 180, 0.95)")
                    .dash(DashType::Dash),
            ),
        );
    }
    if let Some(cut) = ms2_cutoff {
        plot.add_trace(
            Scatter::new(
                vec![
                    if x_min.is_finite() { x_min } else { 0.0 },
                    if x_max.is_finite() { x_max } else { 1.0 },
                ],
                vec![cut, cut],
            )
            .name("SCORE_MS2 cutoff (1% FDR)")
            .mode(Mode::Lines)
            .line(
                Line::new()
                    .color("rgba(255, 127, 14, 0.95)")
                    .dash(DashType::Dash),
            ),
        );
    }

    // Marginal histograms.
    plot.add_trace(
        Histogram::new(t_x)
            .name("Target (TOPAZ)")
            .opacity(0.6)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.6)"))
            .y_axis("y2"),
    );
    plot.add_trace(
        Histogram::new(d_x)
            .name("Decoy (TOPAZ)")
            .opacity(0.6)
            .marker(Marker::new().color("rgba(214, 39, 40, 0.6)"))
            .y_axis("y2"),
    );
    plot.add_trace(
        Histogram::new_vertical(t_y)
            .name("Target (MS2)")
            .opacity(0.6)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.6)"))
            .orientation(Orientation::Horizontal)
            .x_axis("x2"),
    );
    plot.add_trace(
        Histogram::new_vertical(d_y)
            .name("Decoy (MS2)")
            .opacity(0.6)
            .marker(Marker::new().color("rgba(214, 39, 40, 0.6)"))
            .orientation(Orientation::Horizontal)
            .x_axis("x2"),
    );

    plot.set_layout(
        Layout::new()
            .title("TOPAZ vs SCORE_MS2")
            .x_axis(
                Axis::new()
                    .title("TOPAZ score (max candidate logit)")
                    .domain(&[0.0, 0.78]),
            )
            .y_axis(
                Axis::new()
                    .title("SCORE_MS2")
                    .domain(&[0.0, 0.78]),
            )
            .x_axis2(
                Axis::new()
                    .title("Count")
                    .domain(&[0.82, 1.0])
                    .show_tick_labels(false),
            )
            .y_axis2(
                Axis::new()
                    .title("Count")
                    .domain(&[0.82, 1.0])
                    .show_tick_labels(false),
            )
            .bar_mode(BarMode::Overlay),
    );

    plot
}

fn plot_id_bars(topaz: &IdCounts, ms2: Option<&IdCounts>) -> Plot {
    let mut run_set: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for r in topaz.per_run.keys() {
        run_set.insert(*r);
    }
    if let Some(ms2) = ms2 {
        for r in ms2.per_run.keys() {
            run_set.insert(*r);
        }
    }
    let runs: Vec<u64> = run_set.into_iter().collect();
    let run_labels: Vec<String> = runs.iter().map(|r| r.to_string()).collect();

    let topaz_vals: Vec<usize> = runs
        .iter()
        .map(|r| *topaz.per_run.get(r).unwrap_or(&0))
        .collect();
    let topaz_gap: Vec<usize> = topaz_vals
        .iter()
        .map(|v| topaz.union.saturating_sub(*v))
        .collect();

    let hover_topaz: Vec<String> = runs
        .iter()
        .zip(topaz_vals.iter())
        .map(|(r, v)| format!("run_id={r}<br>topaz_ids={v}<br>union_ids={}", topaz.union))
        .collect();
    let hover_topaz_gap: Vec<String> = runs
        .iter()
        .zip(topaz_gap.iter())
        .map(|(r, v)| {
            format!(
                "run_id={r}<br>topaz_missing_vs_union={v}<br>union_ids={}",
                topaz.union
            )
        })
        .collect();

    let mut plot = Plot::new();
    plot.add_trace(
        Bar::new(run_labels.clone(), topaz_vals)
            .name("TOPAZ per-run")
            .marker(Marker::new().color("rgba(31, 119, 180, 0.8)"))
            .hover_info(HoverInfo::Text)
            .hover_text_array(hover_topaz)
            .alignment_group("ids")
            .offset_group("topaz")
            .legend_group("TOPAZ"),
    );
    plot.add_trace(
        Bar::new(run_labels.clone(), topaz_gap)
            .name("TOPAZ union (1% FDR)")
            .marker(
                Marker::new()
                    .color("rgba(31, 119, 180, 0.15)")
                    .line(
                        Line::new()
                            .color("rgba(31, 119, 180, 0.9)")
                            .width(1.5)
                            .dash(DashType::Dash),
                    )
                    .pattern(Pattern::new().shape(PatternShape::RightDiagonalLine)),
            )
            .hover_info(HoverInfo::Text)
            .hover_text_array(hover_topaz_gap)
            .alignment_group("ids")
            .offset_group("topaz")
            .legend_group("TOPAZ"),
    );

    if let Some(ms2) = ms2 {
        let ms2_vals: Vec<usize> = runs
            .iter()
            .map(|r| *ms2.per_run.get(r).unwrap_or(&0))
            .collect();
        let ms2_gap: Vec<usize> = ms2_vals
            .iter()
            .map(|v| ms2.union.saturating_sub(*v))
            .collect();
        let hover_ms2: Vec<String> = runs
            .iter()
            .zip(ms2_vals.iter())
            .map(|(r, v)| format!("run_id={r}<br>ms2_ids={v}<br>union_ids={}", ms2.union))
            .collect();
        let hover_ms2_gap: Vec<String> = runs
            .iter()
            .zip(ms2_gap.iter())
            .map(|(r, v)| {
                format!(
                    "run_id={r}<br>ms2_missing_vs_union={v}<br>union_ids={}",
                    ms2.union
                )
            })
            .collect();

        plot.add_trace(
            Bar::new(run_labels.clone(), ms2_vals)
                .name("SCORE_MS2 per-run")
                .marker(Marker::new().color("rgba(255, 127, 14, 0.8)"))
                .hover_info(HoverInfo::Text)
                .hover_text_array(hover_ms2)
                .alignment_group("ids")
                .offset_group("ms2")
                .legend_group("SCORE_MS2"),
        );
        plot.add_trace(
            Bar::new(run_labels.clone(), ms2_gap)
                .name("SCORE_MS2 union (1% FDR)")
                .marker(
                    Marker::new()
                        .color("rgba(255, 127, 14, 0.15)")
                        .line(
                            Line::new()
                                .color("rgba(255, 127, 14, 0.9)")
                                .width(1.5)
                                .dash(DashType::Dash),
                        )
                        .pattern(Pattern::new().shape(PatternShape::DiagonalCross)),
                )
                .hover_info(HoverInfo::Text)
                .hover_text_array(hover_ms2_gap)
                .alignment_group("ids")
                .offset_group("ms2")
                .legend_group("SCORE_MS2"),
        );
    }

    plot.set_layout(
        Layout::new()
            .title("Identifications @1% FDR (union vs per-run)")
            .x_axis(Axis::new().title("Run ID").type_(AxisType::Category))
            .y_axis(Axis::new().title("Unique precursor IDs"))
            .bar_mode(BarMode::Stack),
    );
    plot
}

fn parse_bag_pid(pid: &str) -> (Option<u64>, Option<u64>) {
    let mut it = pid.split('_');
    let run = it.next().and_then(|v| v.parse::<u64>().ok());
    let prec = it.next().and_then(|v| v.parse::<u64>().ok());
    if it.next().is_some() {
        return (None, None);
    }
    (run, prec)
}
