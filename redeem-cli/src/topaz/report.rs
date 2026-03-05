use anyhow::{bail, Result};
use plotly::common::{HoverInfo, Line, Marker, Mode};
use plotly::layout::{Axis, BarMode};
use plotly::{Histogram, Layout, Plot, Scatter};
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
            .name("Target")
            .mode(Mode::Markers)
            .marker(Marker::new().color("rgba(31, 119, 180, 0.7)").size(4))
            .hover_info(HoverInfo::Text)
            .hover_text_array(t_hover),
    );
    plot.add_trace(
        Scatter::new(d_x, d_y)
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
                .line(Line::new().color("rgba(0,0,0,0.8)")),
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

fn parse_bag_pid(pid: &str) -> (Option<u64>, Option<u64>) {
    let mut it = pid.split('_');
    let run = it.next().and_then(|v| v.parse::<u64>().ok());
    let prec = it.next().and_then(|v| v.parse::<u64>().ok());
    if it.next().is_some() {
        return (None, None);
    }
    (run, prec)
}
