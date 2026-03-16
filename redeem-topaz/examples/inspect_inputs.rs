//! Small troubleshooting utility that renders the fixed-width XIC/XIM tensors
//! seen by TOPAZ for a handful of OSW feature rows.
//!
//! Example:
//!
//! ```text
//! cargo run -p redeem-topaz --example inspect_inputs --features io-sqlite,io-parquet -- \
//!   --osw data.osw \
//!   --xic-map xic_map.tsv \
//!   --xim-map xim_map.tsv \
//!   --feature-ids 101,205,309 \
//!   --out inspect.html
//! ```

use anyhow::{Result, bail};
use plotly::common::{DashType, Line, Mode, Title};
use plotly::layout::Layout;
use plotly::{Plot, Scatter};
use redeem_topaz::infer::{
    TraceBuildConfig, XicFetchConfig, XimFetchConfig, build_trace_tensors_from_parquet,
    build_trace_tensors_from_parquet_map, build_xim_tensors_from_parquet,
    build_xim_tensors_from_parquet_map,
};
use redeem_topaz::inspect::{
    boundary_indices, fetch_xic_for_row, fetch_xim_for_row, read_run_path_map,
    representative_xic_coords, representative_xim_coords, resolve_run_path, valid_im_bounds,
    valid_rt_bounds,
};
use redeem_topaz::io::osw::{OswReadConfig, read_feature_rows};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct Args {
    osw: PathBuf,
    xic: Option<PathBuf>,
    xic_map: Option<PathBuf>,
    xim: Option<PathBuf>,
    xim_map: Option<PathBuf>,
    feature_ids: Option<Vec<u64>>,
    limit: usize,
    out: PathBuf,
    xic_trace: TraceBuildConfig,
    xim_trace: TraceBuildConfig,
}

fn parse_args() -> Result<Args> {
    let mut osw = None;
    let mut xic = None;
    let mut xic_map = None;
    let mut xim = None;
    let mut xim_map = None;
    let mut feature_ids = None;
    let mut limit = 4usize;
    let mut out = PathBuf::from("topaz_inputs.html");
    let mut xic_trace = TraceBuildConfig {
        l: 64,
        ms1_cmax: 0,
        ms2_cmax: 6,
        normalize_max: true,
        mask_rt_peak_bounds: false,
    };
    let mut xim_trace = TraceBuildConfig {
        l: 258,
        ms1_cmax: 0,
        ms2_cmax: 6,
        normalize_max: true,
        mask_rt_peak_bounds: false,
    };

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        let key = &args[i];
        let next = |i: &mut usize| -> Result<String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for {}", args[*i - 1]))
        };
        match key.as_str() {
            "--osw" => osw = Some(PathBuf::from(next(&mut i)?)),
            "--xic" => xic = Some(PathBuf::from(next(&mut i)?)),
            "--xic-map" => xic_map = Some(PathBuf::from(next(&mut i)?)),
            "--xim" => xim = Some(PathBuf::from(next(&mut i)?)),
            "--xim-map" => xim_map = Some(PathBuf::from(next(&mut i)?)),
            "--feature-ids" => {
                let v = next(&mut i)?;
                let ids = v
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse::<u64>())
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                feature_ids = Some(ids);
            }
            "--limit" => limit = next(&mut i)?.parse()?,
            "--out" => out = PathBuf::from(next(&mut i)?),
            "--xic-l" => xic_trace.l = next(&mut i)?.parse()?,
            "--xic-ms1-cmax" => xic_trace.ms1_cmax = next(&mut i)?.parse()?,
            "--xic-ms2-cmax" => xic_trace.ms2_cmax = next(&mut i)?.parse()?,
            "--xim-l" => xim_trace.l = next(&mut i)?.parse()?,
            "--xim-ms1-cmax" => xim_trace.ms1_cmax = next(&mut i)?.parse()?,
            "--xim-ms2-cmax" => xim_trace.ms2_cmax = next(&mut i)?.parse()?,
            "--no-normalize" => {
                xic_trace.normalize_max = false;
                xim_trace.normalize_max = false;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: inspect_inputs --osw FILE [--xic FILE|--xic-map FILE] [--xim FILE|--xim-map FILE] [--feature-ids 1,2,3] [--limit N] [--out FILE]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    let osw = osw.ok_or_else(|| anyhow::anyhow!("--osw is required"))?;
    if xic.is_none() && xic_map.is_none() {
        bail!("provide either --xic or --xic-map");
    }

    Ok(Args {
        osw,
        xic,
        xic_map,
        xim,
        xim_map,
        feature_ids,
        limit,
        out,
        xic_trace,
        xim_trace,
    })
}

fn render_tensor_plot(
    title: &str,
    prefix: &str,
    tensor: &[f32],
    c_total: usize,
    l: usize,
    ms1_cmax: usize,
    boundary: Option<(usize, usize)>,
) -> String {
    let mut plot = Plot::new();
    let x: Vec<usize> = (0..l).collect();
    let mut max_y = 0.0f32;
    for c in 0..c_total {
        let start = c * l;
        let end = start + l;
        let y = tensor[start..end].to_vec();
        for &v in &y {
            if v > max_y {
                max_y = v;
            }
        }
        let label = if c < ms1_cmax {
            format!("{prefix} MS1 {}", c)
        } else {
            format!("{prefix} MS2 {}", c - ms1_cmax)
        };
        plot.add_trace(Scatter::new(x.clone(), y).name(label).mode(Mode::Lines));
    }
    let ymax = if max_y > 0.0 { max_y as f64 } else { 1.0 };
    if let Some((left, right)) = boundary {
        let left_trace = Scatter::new(vec![left, left], vec![0.0, ymax])
            .name(format!("{prefix} left boundary"))
            .mode(Mode::Lines)
            .line(Line::new().dash(DashType::Dash).color("#d62728").width(2.0));
        plot.add_trace(left_trace);
        let right_trace = Scatter::new(vec![right, right], vec![0.0, ymax])
            .name(format!("{prefix} right boundary"))
            .mode(Mode::Lines)
            .line(Line::new().dash(DashType::Dash).color("#2ca02c").width(2.0));
        plot.add_trace(right_trace);
    }
    plot.set_layout(
        Layout::new()
            .title(Title::with_text(title))
            .height(420)
            .width(760),
    );
    plot.to_inline_html(None)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let table = read_feature_rows(&args.osw, &OswReadConfig::default())?;
    let mut rows = if let Some(feature_ids) = &args.feature_ids {
        table
            .rows
            .into_iter()
            .filter(|r| feature_ids.contains(&r.feature_id))
            .collect::<Vec<_>>()
    } else {
        table.rows.into_iter().take(args.limit).collect::<Vec<_>>()
    };
    if rows.is_empty() {
        bail!("no rows selected");
    }
    rows.sort_by_key(|r| (r.run_id, r.precursor_id, r.feature_id));
    let xic_map = if let Some(map_path) = &args.xic_map {
        Some(read_run_path_map(map_path)?)
    } else {
        None
    };
    let xim_map = if let Some(map_path) = &args.xim_map {
        Some(read_run_path_map(map_path)?)
    } else {
        None
    };

    let xic_fetch = XicFetchConfig::default();
    let xim_fetch = XimFetchConfig::default();

    let xic = if let Some(map) = &xic_map {
        build_trace_tensors_from_parquet_map(&rows, map, &args.xic_trace, &xic_fetch)?
    } else {
        build_trace_tensors_from_parquet(
            &rows,
            args.xic.as_ref().expect("validated xic path"),
            &args.xic_trace,
            &xic_fetch,
        )?
    };

    let xim = if let Some(map) = &xim_map {
        Some(build_xim_tensors_from_parquet_map(
            &rows,
            map,
            &args.xim_trace,
            &xim_fetch,
        )?)
    } else if let Some(path) = &args.xim {
        Some(build_xim_tensors_from_parquet(
            &rows,
            path,
            &args.xim_trace,
            &xim_fetch,
        )?)
    } else {
        None
    };

    let mut html = String::from(
        "<html><head><meta charset=\"utf-8\"><title>TOPAZ Input Inspection</title>\
         <script src=\"https://cdn.plot.ly/plotly-2.12.1.min.js\"></script>\
         <style>\
         body{font-family:system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;margin:24px;}\
         h2{margin-top:40px;}\
         .feature-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;align-items:start;}\
         .plot-block{margin:12px 0 24px 0;}\
         .meta{color:#444;margin-bottom:8px;}\
         .warn{color:#a33;font-weight:600;}\
         </style></head><body>",
    );
    html.push_str("<h1>TOPAZ Input Inspection</h1>");
    html.push_str("<p>These plots show the fixed-width tensors fed into TOPAZ after cropping, padding, zero-masking, and optional normalization. The x-axis is the model-input sample index inside the fixed window. When valid peak boundaries are available, dashed lines mark their projected positions inside that window.</p>");

    let xic_row_span = args.xic_trace.total_c() * args.xic_trace.l;
    let xim_row_span = args.xim_trace.total_c() * args.xim_trace.l;
    for (i, row) in rows.iter().enumerate() {
        html.push_str(&format!(
            "<h2>FEATURE_ID={} | RUN_ID={} | PRECURSOR_ID={} | EXP_RT={:.4} | EXP_IM={:?}</h2>",
            row.feature_id, row.run_id, row.precursor_id, row.exp_rt, row.exp_im
        ));
        html.push_str(&format!(
            "<div class=\"meta\">XIC bounds={:?} | XIM bounds={:?}</div>",
            valid_rt_bounds(row),
            valid_im_bounds(row)
        ));
        html.push_str("<div class=\"feature-grid\">");
        let xic_row = &xic[i * xic_row_span..(i + 1) * xic_row_span];
        let xic_all_zero = xic_row.iter().all(|v| v.abs() <= f32::EPSILON);
        let xic_path = resolve_run_path(row.run_id, args.xic.as_deref(), xic_map.as_ref());
        let xic_boundary =
            if let (Some(path), Some((left, right))) = (xic_path.as_ref(), valid_rt_bounds(row)) {
                if let Some(raw) = fetch_xic_for_row(row, path, &xic_fetch)? {
                    let coords = representative_xic_coords(&raw, args.xic_trace.ms1_cmax, false)
                        .or_else(|| representative_xic_coords(&raw, args.xic_trace.ms1_cmax, true));
                    coords.and_then(|coords| {
                        boundary_indices(&coords, row.exp_rt, left, right, args.xic_trace.l)
                    })
                } else {
                    None
                }
            } else {
                None
            };
        html.push_str("<div class=\"plot-block\">");
        if xic_all_zero {
            html.push_str("<div class=\"warn\">XIC tensor is all zeros for this feature.</div>");
        }
        html.push_str(&render_tensor_plot(
            &format!("XIC tensor for FEATURE_ID {}", row.feature_id),
            "XIC",
            xic_row,
            args.xic_trace.total_c(),
            args.xic_trace.l,
            args.xic_trace.ms1_cmax,
            xic_boundary,
        ));
        html.push_str("</div>");
        if let Some(xim_buf) = xim.as_ref() {
            let xim_row = &xim_buf[i * xim_row_span..(i + 1) * xim_row_span];
            let xim_path = resolve_run_path(row.run_id, args.xim.as_deref(), xim_map.as_ref());
            let xim_boundary = if let (Some(path), Some(center), Some((left, right))) =
                (xim_path.as_ref(), row.exp_im, valid_im_bounds(row))
            {
                if let Some(raw) = fetch_xim_for_row(row, path, &xim_fetch)? {
                    let coords = representative_xim_coords(&raw, args.xim_trace.ms1_cmax, false)
                        .or_else(|| representative_xim_coords(&raw, args.xim_trace.ms1_cmax, true));
                    coords.and_then(|coords| {
                        boundary_indices(&coords, center, left, right, args.xim_trace.l)
                    })
                } else {
                    None
                }
            } else {
                None
            };
            html.push_str("<div class=\"plot-block\">");
            html.push_str(&render_tensor_plot(
                &format!("XIM tensor for FEATURE_ID {}", row.feature_id),
                "XIM",
                xim_row,
                args.xim_trace.total_c(),
                args.xim_trace.l,
                args.xim_trace.ms1_cmax,
                xim_boundary,
            ));
            html.push_str("</div>");
        }
        html.push_str("</div>");
    }
    html.push_str("</body></html>");
    std::fs::write(&args.out, html)?;
    eprintln!("Wrote {:?}", args.out);
    Ok(())
}
