use anyhow::{Result, bail};
use std::path::PathBuf;

use crate::topaz::infer::input::TopazInferConfig;
use crate::topaz::report::{TopazReportInputs, write_topaz_report};

pub fn run(
    cfg: &TopazInferConfig,
    head_embeddings_path: Option<PathBuf>,
    report_path: Option<PathBuf>,
    example_bags: usize,
    seed: u64,
) -> Result<()> {
    let report_osw = cfg
        .inner
        .output_osw
        .as_deref()
        .unwrap_or(&cfg.inner.osw_path);
    let topaz_table_name = if cfg.inner.xrun.enabled {
        cfg.inner
            .output_table_xrun
            .as_deref()
            .unwrap_or(cfg.inner.output_table.as_str())
    } else {
        cfg.inner.output_table.as_str()
    };
    let outdir = cfg
        .inner
        .diagnostics
        .head_embeddings_outdir
        .clone()
        .unwrap_or_else(|| PathBuf::from("head_embeddings"));
    let head_path = head_embeddings_path.unwrap_or_else(|| outdir.join("head_embeddings.tsv"));
    if !head_path.exists() {
        bail!(
            "head embeddings not found at {:?}; run inference with diagnostics.save_head_embeddings=true or pass --head-embeddings",
            head_path
        );
    }
    let report_path = report_path.unwrap_or_else(|| outdir.join("topaz_report.html"));
    write_topaz_report(&TopazReportInputs {
        head_embeddings_path: &head_path,
        report_path: &report_path,
        seed,
        osw_path: Some(report_osw),
        score_tsv_path: Some(&cfg.inner.output_tsv),
        topaz_table_name: Some(topaz_table_name),
        xic_path: Some(&cfg.inner.xic_path),
        xic_paths: cfg.inner.xic_paths.as_deref(),
        xic_map_path: cfg.inner.xic_map_path.as_deref(),
        xim_path: cfg.inner.xim_path.as_deref(),
        xim_paths: cfg.inner.xim_paths.as_deref(),
        xim_map_path: cfg.inner.xim_map_path.as_deref(),
        xic_fetch: &cfg.inner.fetch,
        xim_fetch: &cfg.inner.xim_fetch,
        example_bags,
    })?;
    log::info!("Wrote TOPAZ report to {:?}", report_path);
    Ok(())
}
