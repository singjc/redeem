use anyhow::{Context, Result, bail};
use plotly::common::{
    DashType, HoverInfo, Line, Marker, Mode, Orientation, Pattern, PatternShape, Position,
};
use plotly::layout::{Axis, BarMode};
use plotly::{Bar, Histogram, Layout, Plot, Scatter};
use rand::prelude::*;
use redeem_topaz::building_blocks::trace_window::nearest_index_sorted;
use redeem_topaz::infer::stats::tdc_summary;
use redeem_topaz::infer::{XicFetchConfig, XimFetchConfig};
use redeem_topaz::inspect::{
    fetch_xic_for_row, fetch_xim_for_row, read_run_path_map, resolve_run_path, valid_im_bounds,
    valid_rt_bounds,
};
use redeem_topaz::io::osw::FeatureRow;
use redeem_topaz::io::xic::PrecursorXic;
use redeem_topaz::io::xim::FeatureXim;
use report_builder::{Report, ReportSection};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const DEFAULT_PCA_MAX_ROWS: usize = 50_000;
const EXAMPLE_Q_CUT: f64 = 0.01;
const POWER_ITERS: usize = 50;

#[derive(Debug, Clone, Copy)]
enum ExampleCategory {
    HighScoringTopaz,
    LowQualityTopaz,
    TopazOnly,
    Ms2Only,
}

impl ExampleCategory {
    fn title(self) -> &'static str {
        match self {
            Self::HighScoringTopaz => "Raw Traces: High-Scoring TOPAZ Targets",
            Self::LowQualityTopaz => "Raw Traces: Low-Quality TOPAZ Targets",
            Self::TopazOnly => "Raw Traces: TOPAZ-Only Targets",
            Self::Ms2Only => "Raw Traces: SCORE_MS2-Only Targets",
        }
    }
}

pub struct TopazReportInputs<'a> {
    pub head_embeddings_path: &'a Path,
    pub report_path: &'a Path,
    pub seed: u64,
    pub osw_path: Option<&'a Path>,
    pub score_tsv_path: Option<&'a Path>,
    pub topaz_table_name: Option<&'a str>,
    pub topaz_label: Option<&'a str>,
    pub topaz_base_table_name: Option<&'a str>,
    pub topaz_base_label: Option<&'a str>,
    pub xic_path: Option<&'a Path>,
    pub xic_paths: Option<&'a [PathBuf]>,
    pub xic_map_path: Option<&'a Path>,
    pub xim_path: Option<&'a Path>,
    pub xim_paths: Option<&'a [PathBuf]>,
    pub xim_map_path: Option<&'a Path>,
    pub xic_fetch: &'a XicFetchConfig,
    pub xim_fetch: &'a XimFetchConfig,
    pub example_bags: usize,
}

pub fn write_topaz_report(inputs: &TopazReportInputs<'_>) -> Result<()> {
    let emb = load_head_embeddings_tsv(inputs.head_embeddings_path)?;
    if emb.hidden_dim == 0 || emb.n == 0 {
        bail!(
            "head embeddings are empty: {:?}",
            inputs.head_embeddings_path
        );
    }

    let precursor_meta = if let Some(path) = inputs.osw_path {
        redeem_topaz::io::osw::read_precursor_meta(path).ok()
    } else {
        None
    };
    let feature_meta = if let Some(path) = inputs.osw_path {
        redeem_topaz::io::osw::read_feature_meta(path).ok()
    } else {
        None
    };

    let topaz_label = inputs.topaz_label.unwrap_or("TOPAZ");
    let topaz_scores = load_score_source(
        inputs.score_tsv_path,
        inputs.osw_path,
        inputs.topaz_table_name.unwrap_or("SCORE_TOPAZ"),
        topaz_label,
    );
    let topaz_base_label = inputs.topaz_base_label.unwrap_or("TOPAZ Base");
    let topaz_base_scores = inputs.topaz_base_table_name.and_then(|table_name| {
        load_score_source(None, inputs.osw_path, table_name, topaz_base_label)
    });
    let ms2_scores = inputs
        .osw_path
        .and_then(|p| redeem_topaz::io::osw::read_score_table(p, "SCORE_MS2").ok())
        .filter(|rows| !rows.is_empty())
        .map(|rows| {
            rows.into_iter()
                .map(ScoreLite::from_ms2)
                .collect::<Vec<_>>()
        });

    let pca = pca2(
        &emb.hidden,
        emb.n,
        emb.hidden_dim,
        DEFAULT_PCA_MAX_ROWS,
        inputs.seed,
    );
    let bag_scores_f32: Vec<f32> = emb.bag_score.iter().map(|&v| v as f32).collect();
    let tdc = tdc_summary(&bag_scores_f32, &emb.is_decoy, 0.01);
    let cutoff = if tdc.cutoff.is_finite() {
        Some(tdc.cutoff as f64)
    } else {
        None
    };

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
            let mut methods = Vec::new();
            methods.push(NamedIdCounts {
                label: topaz_label.to_string(),
                counts: topaz_ids,
                color: "rgba(31, 119, 180, 0.8)",
                union_color: "rgba(31, 119, 180, 0.15)",
                union_pattern: PatternShape::RightDiagonalLine,
            });
            if let Some(base_scores) = topaz_base_scores.as_ref() {
                if let Some(base_ids) = compute_id_counts(base_scores, meta, 0.01) {
                    methods.push(NamedIdCounts {
                        label: topaz_base_label.to_string(),
                        counts: base_ids,
                        color: "rgba(44, 160, 44, 0.8)",
                        union_color: "rgba(44, 160, 44, 0.15)",
                        union_pattern: PatternShape::VerticalLine,
                    });
                }
            }
            let ms2_ids = ms2_scores
                .as_ref()
                .and_then(|rows| compute_id_counts(rows, meta, 0.01));
            if let Some(ms2_ids) = ms2_ids {
                methods.push(NamedIdCounts {
                    label: "SCORE_MS2".to_string(),
                    counts: ms2_ids,
                    color: "rgba(255, 127, 14, 0.8)",
                    union_color: "rgba(255, 127, 14, 0.15)",
                    union_pattern: PatternShape::DiagonalCross,
                });
            }
            let mut id_section = ReportSection::new("Identifications");
            id_section.add_plot(plot_id_bars(&methods));
            report.add_section(id_section);
        }
    }

    if let (Some(meta), Some(ms2)) = (feature_meta.as_ref(), ms2_scores.as_ref()) {
        let mut sec = ReportSection::new("TOPAZ vs SCORE_MS2");
        let mut added_plot = false;
        if let Some(base_scores) = topaz_base_scores.as_ref() {
            let pairs = build_score_pairs(base_scores, ms2, meta, precursor_meta.as_ref());
            if !pairs.is_empty() {
                sec.add_plot(plot_score_scatter_with_marginals(
                    &pairs,
                    cutoff_from_score_rows(base_scores, 0.01),
                    cutoff_from_score_rows(ms2, 0.01),
                    topaz_base_label,
                    "SCORE_MS2",
                    &format!("{topaz_base_label} vs SCORE_MS2"),
                ));
                added_plot = true;
            }
        }
        if let Some(scores) = topaz_scores.as_ref() {
            let pairs = build_score_pairs(scores, ms2, meta, precursor_meta.as_ref());
            if !pairs.is_empty() {
                sec.add_plot(plot_score_scatter_with_marginals(
                    &pairs,
                    cutoff_from_score_rows(scores, 0.01),
                    cutoff_from_score_rows(ms2, 0.01),
                    topaz_label,
                    "SCORE_MS2",
                    &format!("{topaz_label} vs SCORE_MS2"),
                ));
                added_plot = true;
            }
        }
        if added_plot {
            report.add_section(sec);
        }
    }

    if let (Some(osw_path), Some(scores), Some(feature_meta)) = (
        inputs.osw_path,
        topaz_scores.as_ref(),
        feature_meta.as_ref(),
    ) {
        match build_example_sections(
            osw_path,
            &emb,
            scores,
            ms2_scores.as_ref(),
            inputs,
            precursor_meta.as_ref(),
            feature_meta,
        ) {
            Ok(sections) if !sections.is_empty() => {
                for section in sections {
                    report.add_section(section);
                }
            }
            Ok(_) => {}
            Err(err) => {
                log::warn!("Skipping raw trace report section: {err:#}");
            }
        }
    }

    report.save_to_file(&inputs.report_path.to_string_lossy().to_string())?;
    Ok(())
}

/// Load score rows for one method from either a TSV or an OSW score table.
///
/// The report prefers a TSV when available because it mirrors the exact row set
/// emitted by inference. When the TSV is missing, the function falls back to
/// reading the named score table from the OSW, which is the common case for
/// report-only reruns on remote systems.
fn load_score_source(
    score_tsv_path: Option<&Path>,
    osw_path: Option<&Path>,
    table_name: &str,
    label: &str,
) -> Option<Vec<ScoreLite>> {
    if let Some(path) = score_tsv_path {
        match load_score_tsv(path) {
            Ok(rows) if !rows.is_empty() => {
                log::info!("Loaded {label} scores from TSV {:?}", path);
                return Some(rows);
            }
            Ok(_) => {
                log::warn!(
                    "{label} score TSV {:?} is empty; trying OSW table fallback",
                    path
                );
            }
            Err(err) => {
                log::warn!(
                    "Failed to load {label} score TSV {:?}: {err:#}; trying OSW table fallback",
                    path
                );
            }
        }
    }

    let Some(osw_path) = osw_path else {
        return None;
    };
    match redeem_topaz::io::osw::read_score_table(osw_path, table_name) {
        Ok(rows) if !rows.is_empty() => {
            log::info!(
                "Loaded {label} scores from OSW table {:?} in {:?}",
                table_name,
                osw_path
            );
            Some(rows.into_iter().map(ScoreLite::from_ms2).collect())
        }
        Ok(_) => {
            log::warn!(
                "{label} OSW table {:?} in {:?} is empty; report will skip this comparison",
                table_name,
                osw_path
            );
            None
        }
        Err(err) => {
            log::warn!(
                "Failed to load {label} OSW table {:?} from {:?}: {err:#}",
                table_name,
                osw_path
            );
            None
        }
    }
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

#[derive(Debug, Clone)]
struct NamedIdCounts {
    label: String,
    counts: IdCounts,
    color: &'static str,
    union_color: &'static str,
    union_pattern: PatternShape,
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
        let score = rec
            .get(idx_score)
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);
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
        let fid = rec.get(idx_feat).unwrap_or("0").parse::<u64>().unwrap_or(0);
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
    let mut per_run_counts: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();
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
    x_cutoff: Option<f64>,
    y_cutoff: Option<f64>,
    x_label: &str,
    y_label: &str,
    title: &str,
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

    if let Some(cut) = x_cutoff {
        plot.add_trace(
            Scatter::new(
                vec![cut, cut],
                vec![
                    if y_min.is_finite() { y_min } else { 0.0 },
                    if y_max.is_finite() { y_max } else { 1.0 },
                ],
            )
            .name(format!("{x_label} cutoff (1% FDR)"))
            .mode(Mode::Lines)
            .line(
                Line::new()
                    .color("rgba(31, 119, 180, 0.95)")
                    .dash(DashType::Dash),
            ),
        );
    }
    if let Some(cut) = y_cutoff {
        plot.add_trace(
            Scatter::new(
                vec![
                    if x_min.is_finite() { x_min } else { 0.0 },
                    if x_max.is_finite() { x_max } else { 1.0 },
                ],
                vec![cut, cut],
            )
            .name(format!("{y_label} cutoff (1% FDR)"))
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
            .name(format!("Target ({x_label})"))
            .opacity(0.6)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.6)"))
            .y_axis("y2"),
    );
    plot.add_trace(
        Histogram::new(d_x)
            .name(format!("Decoy ({x_label})"))
            .opacity(0.6)
            .marker(Marker::new().color("rgba(214, 39, 40, 0.6)"))
            .y_axis("y2"),
    );
    plot.add_trace(
        Histogram::new_vertical(t_y)
            .name(format!("Target ({y_label})"))
            .opacity(0.6)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.6)"))
            .orientation(Orientation::Horizontal)
            .x_axis("x2"),
    );
    plot.add_trace(
        Histogram::new_vertical(d_y)
            .name(format!("Decoy ({y_label})"))
            .opacity(0.6)
            .marker(Marker::new().color("rgba(214, 39, 40, 0.6)"))
            .orientation(Orientation::Horizontal)
            .x_axis("x2"),
    );

    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(
                Axis::new()
                    .title(format!("{x_label} score"))
                    .domain(&[0.0, 0.78]),
            )
            .y_axis(Axis::new().title(y_label).domain(&[0.0, 0.78]))
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

fn plot_id_bars(methods: &[NamedIdCounts]) -> Plot {
    if methods.is_empty() {
        return Plot::new();
    }
    let mut run_set: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for method in methods {
        for r in method.counts.per_run.keys() {
            run_set.insert(*r);
        }
    }

    let runs: Vec<u64> = run_set.into_iter().collect();
    let run_labels: Vec<String> = runs.iter().map(|r| r.to_string()).collect();

    let centers: Vec<f64> = (0..runs.len()).map(|i| i as f64).collect();
    let mut plot = Plot::new();
    let n_methods = methods.len() as f64;
    let bar_width = (0.8 / n_methods).min(0.35);

    for (idx, method) in methods.iter().enumerate() {
        let offset = (idx as f64 - (n_methods - 1.0) / 2.0) * bar_width;
        let method_x: Vec<f64> = centers.iter().map(|x| x + offset).collect();
        let method_vals: Vec<usize> = runs
            .iter()
            .map(|r| *method.counts.per_run.get(r).unwrap_or(&0))
            .collect();
        let method_gap: Vec<usize> = method_vals
            .iter()
            .map(|v| method.counts.union.saturating_sub(*v))
            .collect();
        let hover_per_run: Vec<String> = runs
            .iter()
            .zip(method_vals.iter())
            .map(|(r, v)| {
                format!(
                    "run_id={r}<br>{} per-run IDs={v}<br>union_ids={}",
                    method.label, method.counts.union
                )
            })
            .collect();
        let hover_union: Vec<String> = runs
            .iter()
            .zip(method_gap.iter())
            .map(|(r, v)| {
                format!(
                    "run_id={r}<br>{} missing_vs_union={v}<br>union_ids={}",
                    method.label, method.counts.union
                )
            })
            .collect();

        plot.add_trace(
            Bar::new(method_x.clone(), method_vals)
                .name(format!("{} per-run", method.label))
                .width(bar_width)
                .marker(Marker::new().color(method.color))
                .hover_info(HoverInfo::Text)
                .hover_text_array(hover_per_run)
                .legend_group(method.label.clone()),
        );
        plot.add_trace(
            Bar::new(method_x, method_gap)
                .name(format!("{} union (1% FDR)", method.label))
                .width(bar_width)
                .marker(
                    Marker::new()
                        .color(method.union_color)
                        .line(
                            Line::new()
                                .color(method.color)
                                .width(1.5)
                                .dash(DashType::Dash),
                        )
                        .pattern(Pattern::new().shape(method.union_pattern.clone())),
                )
                .hover_info(HoverInfo::Text)
                .hover_text_array(hover_union)
                .legend_group(method.label.clone()),
        );
    }

    plot.set_layout(
        Layout::new()
            .title("Identifications @1% FDR (union vs per-run)")
            .x_axis(
                Axis::new()
                    .title("Run ID")
                    .tick_values(centers.clone())
                    .tick_text(run_labels),
            )
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

#[derive(Debug, Clone)]
struct ExampleCandidate {
    row: FeatureRow,
    topaz: ScoreLite,
    ms2: Option<ScoreLite>,
}

#[derive(Debug, Clone)]
struct ExampleBag {
    bag_pid: String,
    title: String,
    candidates: Vec<ExampleCandidate>,
    raw_xic: Option<PrecursorXic>,
    raw_xim: Option<FeatureXim>,
}

#[derive(Debug, Clone)]
struct BagExampleSummary {
    emb_idx: usize,
    run_id: u64,
    precursor_id: u64,
    bag_score: f64,
    is_decoy: bool,
    topaz_rank1: Option<ScoreLite>,
    ms2_rank1: Option<ScoreLite>,
}

fn infer_run_map_from_paths(
    paths: &[PathBuf],
    list_run_ids: fn(&Path) -> Result<Vec<u64>>,
) -> Result<HashMap<u64, PathBuf>> {
    let mut out = HashMap::new();
    for path in paths {
        for run_id in list_run_ids(path)? {
            out.insert(run_id, path.clone());
        }
    }
    Ok(out)
}

fn resolve_xic_run_map(inputs: &TopazReportInputs<'_>) -> Result<Option<HashMap<u64, PathBuf>>> {
    if let Some(map_path) = inputs.xic_map_path {
        return read_run_path_map(map_path).map(Some);
    }
    if let Some(paths) = inputs.xic_paths {
        return infer_run_map_from_paths(paths, redeem_topaz::io::xic_parquet::list_run_ids)
            .map(Some);
    }
    Ok(None)
}

fn resolve_xim_run_map(inputs: &TopazReportInputs<'_>) -> Result<Option<HashMap<u64, PathBuf>>> {
    if let Some(map_path) = inputs.xim_map_path {
        return read_run_path_map(map_path).map(Some);
    }
    if let Some(paths) = inputs.xim_paths {
        return infer_run_map_from_paths(paths, redeem_topaz::io::xim_parquet::list_run_ids)
            .map(Some);
    }
    Ok(None)
}

fn score_is_rank1(score: &ScoreLite) -> bool {
    score.rank.unwrap_or(1) == 1
}

fn score_passes_q(score: Option<&ScoreLite>, q_cut: f64) -> bool {
    match score {
        Some(score) if score_is_rank1(score) => score.qvalue.is_some_and(|q| q <= q_cut),
        _ => false,
    }
}

fn rank1_scores_by_bag(
    rows: &[ScoreLite],
    meta: &HashMap<u64, redeem_topaz::io::osw::FeatureMeta>,
) -> HashMap<String, ScoreLite> {
    let mut out = HashMap::new();
    for row in rows {
        if !score_is_rank1(row) {
            continue;
        }
        let Some(feature) = meta.get(&row.feature_id) else {
            continue;
        };
        let bag_pid = format!("{}_{}", feature.run_id, feature.precursor_id);
        let replace = out
            .get(&bag_pid)
            .map(|prev: &ScoreLite| row.score > prev.score)
            .unwrap_or(true);
        if replace {
            out.insert(bag_pid, row.clone());
        }
    }
    out
}

fn build_bag_summaries(
    emb: &HeadEmbeddings,
    topaz_scores: &[ScoreLite],
    ms2_scores: Option<&Vec<ScoreLite>>,
    feature_meta: &HashMap<u64, redeem_topaz::io::osw::FeatureMeta>,
) -> Vec<BagExampleSummary> {
    let topaz_by_bag = rank1_scores_by_bag(topaz_scores, feature_meta);
    let ms2_by_bag = ms2_scores
        .map(|rows| rank1_scores_by_bag(rows, feature_meta))
        .unwrap_or_default();

    let mut out = Vec::new();
    for emb_idx in 0..emb.n {
        let bag_pid = emb.bag_pid[emb_idx].clone();
        let (run_id, precursor_id) = parse_bag_pid(&bag_pid);
        let (Some(run_id), Some(precursor_id)) = (run_id, precursor_id) else {
            continue;
        };
        out.push(BagExampleSummary {
            emb_idx,
            run_id,
            precursor_id,
            bag_score: emb.bag_score[emb_idx],
            is_decoy: emb.is_decoy[emb_idx],
            topaz_rank1: topaz_by_bag.get(&bag_pid).cloned(),
            ms2_rank1: ms2_by_bag.get(&bag_pid).cloned(),
        });
    }
    out
}

fn matches_example_category(summary: &BagExampleSummary, category: ExampleCategory) -> bool {
    if summary.is_decoy {
        return false;
    }
    let topaz_pass = score_passes_q(summary.topaz_rank1.as_ref(), EXAMPLE_Q_CUT);
    let ms2_pass = score_passes_q(summary.ms2_rank1.as_ref(), EXAMPLE_Q_CUT);
    match category {
        ExampleCategory::HighScoringTopaz => topaz_pass,
        ExampleCategory::LowQualityTopaz => summary.topaz_rank1.is_some() && !topaz_pass,
        ExampleCategory::TopazOnly => topaz_pass && !ms2_pass,
        ExampleCategory::Ms2Only => ms2_pass && !topaz_pass,
    }
}

fn category_primary_score(summary: &BagExampleSummary, category: ExampleCategory) -> f64 {
    match category {
        ExampleCategory::Ms2Only => summary
            .ms2_rank1
            .as_ref()
            .map(|score| score.score)
            .unwrap_or(summary.bag_score),
        ExampleCategory::TopazOnly | ExampleCategory::LowQualityTopaz => summary
            .topaz_rank1
            .as_ref()
            .map(|score| score.score)
            .unwrap_or(summary.bag_score),
        ExampleCategory::HighScoringTopaz => summary.bag_score,
    }
}

fn select_example_indices(
    summaries: &[BagExampleSummary],
    category: ExampleCategory,
    limit: usize,
) -> Vec<usize> {
    if limit == 0 {
        return Vec::new();
    }
    let mut selected: Vec<&BagExampleSummary> = summaries
        .iter()
        .filter(|summary| matches_example_category(summary, category))
        .collect();
    selected.sort_by(|a, b| {
        category_primary_score(b, category)
            .partial_cmp(&category_primary_score(a, category))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.bag_score
                    .partial_cmp(&a.bag_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.run_id.cmp(&b.run_id))
            .then_with(|| a.precursor_id.cmp(&b.precursor_id))
    });
    selected
        .into_iter()
        .take(limit)
        .map(|summary| summary.emb_idx)
        .collect()
}

fn score_rank_label(score: &ScoreLite) -> String {
    match score.rank {
        Some(1) => "top".to_string(),
        Some(rank) => format!("r{rank}"),
        None => "cand".to_string(),
    }
}

fn score_hover_html(candidate: &ExampleCandidate) -> String {
    let mut out = format!(
        "feature_id={}<br>topaz_score={:.3}<br>topaz_rank={:?}<br>topaz_q={:?}",
        candidate.row.feature_id,
        candidate.topaz.score,
        candidate.topaz.rank,
        candidate.topaz.qvalue
    );
    if let Some(ms2) = candidate.ms2.as_ref() {
        out.push_str(&format!(
            "<br>ms2_score={:.3}<br>ms2_rank={:?}<br>ms2_q={:?}",
            ms2.score, ms2.rank, ms2.qvalue
        ));
    }
    out
}

fn summed_xic_trace(xic: &PrecursorXic) -> Option<(Vec<f64>, Vec<f64>)> {
    let first = xic.transitions.iter().find(|t| !t.points.is_empty())?;
    let coords: Vec<f64> = first.points.iter().map(|p| p.rt as f64).collect();
    let mut sum = vec![0.0f64; coords.len()];
    for trace in &xic.transitions {
        for (i, point) in trace.points.iter().take(sum.len()).enumerate() {
            sum[i] += point.intensity as f64;
        }
    }
    Some((coords, sum))
}

fn summed_xim_trace(xim: &FeatureXim) -> Option<(Vec<f64>, Vec<f64>)> {
    let first = xim.traces.iter().find(|t| !t.points.is_empty())?;
    let coords: Vec<f64> = first.points.iter().map(|p| p.mobility as f64).collect();
    let mut sum = vec![0.0f64; coords.len()];
    for trace in &xim.traces {
        for (i, point) in trace.points.iter().take(sum.len()).enumerate() {
            sum[i] += point.intensity as f64;
        }
    }
    Some((coords, sum))
}

fn nearest_signal(coords: &[f64], signal: &[f64], x: f64) -> f64 {
    if coords.is_empty() || signal.is_empty() {
        return 0.0;
    }
    let coords32: Vec<f32> = coords.iter().map(|&v| v as f32).collect();
    let idx = nearest_index_sorted(&coords32, x as f32).min(signal.len().saturating_sub(1));
    signal[idx]
}

fn palette_color(idx: usize) -> &'static str {
    const COLORS: &[&str] = &[
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
    ];
    COLORS[idx % COLORS.len()]
}

fn build_example_sections(
    osw_path: &Path,
    emb: &HeadEmbeddings,
    topaz_scores: &[ScoreLite],
    ms2_scores: Option<&Vec<ScoreLite>>,
    inputs: &TopazReportInputs<'_>,
    precursor_meta: Option<&HashMap<u64, redeem_topaz::io::osw::PrecursorMeta>>,
    feature_meta: &HashMap<u64, redeem_topaz::io::osw::FeatureMeta>,
) -> Result<Vec<ReportSection>> {
    let summaries = build_bag_summaries(emb, topaz_scores, ms2_scores, feature_meta);
    if summaries.is_empty() {
        return Ok(Vec::new());
    }

    let mut sections = Vec::new();
    for category in [
        ExampleCategory::HighScoringTopaz,
        ExampleCategory::LowQualityTopaz,
        ExampleCategory::TopazOnly,
        ExampleCategory::Ms2Only,
    ] {
        let selected_indices = select_example_indices(&summaries, category, inputs.example_bags);
        if selected_indices.is_empty() {
            continue;
        }
        let plots = build_example_plots_for_indices(
            osw_path,
            emb,
            topaz_scores,
            ms2_scores,
            inputs,
            precursor_meta,
            &selected_indices,
        )?;
        if plots.is_empty() {
            continue;
        }
        let mut section = ReportSection::new(category.title());
        for plot in plots {
            section.add_plot(plot);
        }
        sections.push(section);
    }

    Ok(sections)
}

fn build_example_plots_for_indices(
    osw_path: &Path,
    emb: &HeadEmbeddings,
    topaz_scores: &[ScoreLite],
    ms2_scores: Option<&Vec<ScoreLite>>,
    inputs: &TopazReportInputs<'_>,
    precursor_meta: Option<&HashMap<u64, redeem_topaz::io::osw::PrecursorMeta>>,
    selected_indices: &[usize],
) -> Result<Vec<Plot>> {
    if selected_indices.is_empty() {
        return Ok(Vec::new());
    }
    if inputs.xic_path.is_none() && inputs.xic_paths.is_none() && inputs.xic_map_path.is_none() {
        return Ok(Vec::new());
    }

    let selected_bags: Vec<(u64, u64)> = selected_indices
        .iter()
        .filter_map(|&idx| {
            let (run_id, precursor_id) = parse_bag_pid(&emb.bag_pid[idx]);
            Some((run_id?, precursor_id?))
        })
        .collect();
    if selected_bags.is_empty() {
        return Ok(Vec::new());
    }

    let feature_rows = redeem_topaz::io::osw::read_feature_rows_for_bags(osw_path, &selected_bags)
        .with_context(|| format!("reading OSW feature rows from {:?}", osw_path))?;

    let mut rows_by_pid: HashMap<String, Vec<FeatureRow>> = HashMap::new();
    for row in feature_rows {
        rows_by_pid
            .entry(row.group_id.clone())
            .or_default()
            .push(row);
    }

    let topaz_by_feature: HashMap<u64, ScoreLite> = topaz_scores
        .iter()
        .cloned()
        .map(|row| (row.feature_id, row))
        .collect();
    let ms2_by_feature: HashMap<u64, ScoreLite> = ms2_scores
        .map(|rows| {
            rows.iter()
                .cloned()
                .map(|row| (row.feature_id, row))
                .collect()
        })
        .unwrap_or_default();

    let xic_run_map =
        resolve_xic_run_map(inputs).context("resolving XIC path inputs for raw report traces")?;
    let xim_run_map =
        resolve_xim_run_map(inputs).context("resolving XIM path inputs for raw report traces")?;

    let mut plots = Vec::new();
    for idx in selected_indices {
        let idx = *idx;
        let bag_pid = emb.bag_pid[idx].clone();
        let bag_rows = match rows_by_pid.get(&bag_pid) {
            Some(rows) => rows.clone(),
            None => continue,
        };
        let mut candidates = Vec::new();
        for row in bag_rows {
            let Some(topaz) = topaz_by_feature.get(&row.feature_id).cloned() else {
                continue;
            };
            candidates.push(ExampleCandidate {
                ms2: ms2_by_feature.get(&row.feature_id).cloned(),
                row,
                topaz,
            });
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|a, b| {
            a.topaz
                .rank
                .unwrap_or(i32::MAX)
                .cmp(&b.topaz.rank.unwrap_or(i32::MAX))
                .then_with(|| {
                    b.topaz
                        .score
                        .partial_cmp(&a.topaz.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let first = &candidates[0].row;
        let run_id = first.run_id;
        let precursor_id = first.precursor_id;
        let peptide = precursor_meta
            .and_then(|m| m.get(&precursor_id))
            .map(|m| format!("{} / z={}", m.modified_sequence, m.charge))
            .unwrap_or_else(|| format!("precursor_id={precursor_id}"));

        let raw_xic = match resolve_run_path(run_id, inputs.xic_path, xic_run_map.as_ref()) {
            Some(path) => match fetch_xic_for_row(first, &path, inputs.xic_fetch) {
                Ok(xic) => xic,
                Err(err) => {
                    log::warn!(
                        "Skipping raw XIC example fetch for feature_id={} from {:?}: {err:#}",
                        first.feature_id,
                        path
                    );
                    None
                }
            },
            None => None,
        };

        let raw_xim = if let Some(path) =
            resolve_run_path(run_id, inputs.xim_path, xim_run_map.as_ref())
        {
            match fetch_xim_for_row(first, &path, inputs.xim_fetch) {
                Ok(xim) => xim,
                Err(err) => {
                    log::warn!(
                        "Skipping raw XIM example fetch for bag_pid={} feature_id={} from {:?}: {err:#}",
                        bag_pid,
                        first.feature_id,
                        path
                    );
                    None
                }
            }
        } else {
            None
        };

        let example = ExampleBag {
            bag_pid,
            title: format!(
                "{} | run_id={} | bag_score={:.3} | {}",
                peptide,
                run_id,
                emb.bag_score[idx],
                if emb.is_decoy[idx] { "decoy" } else { "target" }
            ),
            candidates,
            raw_xic,
            raw_xim,
        };
        plots.push(plot_example_bag(&example));
    }
    Ok(plots)
}

fn plot_example_bag(example: &ExampleBag) -> Plot {
    let mut plot = Plot::new();
    let mut xic_y_max = 1.0f64;
    let mut xim_y_max = 1.0f64;
    let top_candidate = example.candidates.first();

    if let Some(raw_xic) = example.raw_xic.as_ref() {
        for trace in &raw_xic.transitions {
            let x: Vec<f64> = trace.points.iter().map(|p| p.rt as f64).collect();
            let y: Vec<f64> = trace.points.iter().map(|p| p.intensity as f64).collect();
            xic_y_max = xic_y_max.max(y.iter().copied().fold(0.0, f64::max));
            plot.add_trace(
                Scatter::new(x, y)
                    .name(format!("XIC {}", trace.annotation))
                    .mode(Mode::Lines)
                    .line(Line::new().color("rgba(31,119,180,0.28)").width(1.0)),
            );
        }
        if let Some((coords, signal)) = summed_xic_trace(raw_xic) {
            xic_y_max = xic_y_max.max(signal.iter().copied().fold(0.0, f64::max));
            plot.add_trace(
                Scatter::new(coords.clone(), signal.clone())
                    .name("XIC sum")
                    .mode(Mode::Lines)
                    .line(Line::new().color("#111111").width(2.5)),
            );
            for (i, candidate) in example.candidates.iter().enumerate() {
                let color = palette_color(i);
                let apex_x = candidate.row.exp_rt as f64;
                let apex_y = nearest_signal(&coords, &signal, apex_x).max(0.0);
                plot.add_trace(
                    Scatter::new(vec![apex_x], vec![apex_y])
                        .name(format!(
                            "{} {}",
                            score_rank_label(&candidate.topaz),
                            candidate.row.feature_id
                        ))
                        .mode(Mode::MarkersText)
                        .text_array(vec![score_rank_label(&candidate.topaz)])
                        .text_position(Position::TopCenter)
                        .marker(Marker::new().color(color).size(9))
                        .show_legend(false)
                        .hover_info(HoverInfo::Text)
                        .hover_text_array(vec![score_hover_html(candidate)]),
                );
            }
            if let Some(candidate) = top_candidate {
                if let Some((left, right)) = valid_rt_bounds(&candidate.row) {
                    let color = palette_color(0);
                    plot.add_trace(
                        Scatter::new(vec![left as f64, left as f64], vec![0.0, xic_y_max])
                            .name(format!("top rt_left {}", candidate.row.feature_id))
                            .mode(Mode::Lines)
                            .show_legend(false)
                            .line(Line::new().color(color).dash(DashType::Dash).width(1.8)),
                    );
                    plot.add_trace(
                        Scatter::new(vec![right as f64, right as f64], vec![0.0, xic_y_max])
                            .name(format!("top rt_right {}", candidate.row.feature_id))
                            .mode(Mode::Lines)
                            .show_legend(false)
                            .line(Line::new().color(color).dash(DashType::Dash).width(1.8)),
                    );
                }
            }
        }
    }

    if let (Some(candidate), Some(raw_xim)) = (top_candidate, example.raw_xim.as_ref()) {
        let Some((coords, signal)) = summed_xim_trace(raw_xim) else {
            plot.set_layout(
                Layout::new()
                    .title(format!("{} | {}", example.title, example.bag_pid))
                    .x_axis(Axis::new().title("Retention time").domain(&[0.0, 0.46]))
                    .y_axis(Axis::new().title("XIC intensity").domain(&[0.0, 1.0]))
                    .x_axis2(Axis::new().title("Ion mobility").domain(&[0.54, 1.0]))
                    .y_axis2(
                        Axis::new()
                            .title("XIM intensity")
                            .domain(&[0.0, 1.0])
                            .anchor("x2"),
                    )
                    .bar_mode(BarMode::Overlay),
            );
            return plot;
        };
        xim_y_max = xim_y_max.max(signal.iter().copied().fold(0.0, f64::max));
        let color = palette_color(0);
        plot.add_trace(
            Scatter::new(coords.clone(), signal.clone())
                .name(format!(
                    "XIM {} {}",
                    score_rank_label(&candidate.topaz),
                    candidate.row.feature_id
                ))
                .mode(Mode::Lines)
                .x_axis("x2")
                .y_axis("y2")
                .line(Line::new().color(color).width(2.5)),
        );
        let apex_x = candidate.row.exp_im.unwrap_or(0.0) as f64;
        let apex_y = nearest_signal(&coords, &signal, apex_x).max(0.0);
        plot.add_trace(
            Scatter::new(vec![apex_x], vec![apex_y])
                .name(format!("xim {}", candidate.row.feature_id))
                .mode(Mode::MarkersText)
                .x_axis("x2")
                .y_axis("y2")
                .text_array(vec![score_rank_label(&candidate.topaz)])
                .text_position(Position::TopCenter)
                .marker(Marker::new().color(color).size(9))
                .show_legend(false)
                .hover_info(HoverInfo::Text)
                .hover_text_array(vec![score_hover_html(candidate)]),
        );
        if let Some((left, right)) = valid_im_bounds(&candidate.row) {
            plot.add_trace(
                Scatter::new(vec![left as f64, left as f64], vec![0.0, xim_y_max])
                    .name(format!("top im_left {}", candidate.row.feature_id))
                    .mode(Mode::Lines)
                    .x_axis("x2")
                    .y_axis("y2")
                    .show_legend(false)
                    .line(Line::new().color(color).dash(DashType::Dash).width(1.8)),
            );
            plot.add_trace(
                Scatter::new(vec![right as f64, right as f64], vec![0.0, xim_y_max])
                    .name(format!("top im_right {}", candidate.row.feature_id))
                    .mode(Mode::Lines)
                    .x_axis("x2")
                    .y_axis("y2")
                    .show_legend(false)
                    .line(Line::new().color(color).dash(DashType::Dash).width(1.8)),
            );
        }
    }

    plot.set_layout(
        Layout::new()
            .title(format!("{} | {}", example.title, example.bag_pid))
            .x_axis(Axis::new().title("Retention time").domain(&[0.0, 0.46]))
            .y_axis(Axis::new().title("XIC intensity").domain(&[0.0, 1.0]))
            .x_axis2(Axis::new().title("Ion mobility").domain(&[0.54, 1.0]))
            .y_axis2(
                Axis::new()
                    .title("XIM intensity")
                    .domain(&[0.0, 1.0])
                    .anchor("x2"),
            )
            .bar_mode(BarMode::Overlay),
    );
    plot
}
