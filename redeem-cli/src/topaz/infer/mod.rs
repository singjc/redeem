use anyhow::Result;

use redeem_topaz::run_inference;

use self::input::TopazInferConfig;
use crate::topaz::report::{TopazReportInputs, write_topaz_report};

pub mod input;

pub fn run(cfg: &TopazInferConfig) -> Result<()> {
    let out = run_inference(&cfg.inner)?;
    eprintln!(
        "[ReDeeM::Topaz] Inference complete. Scored {} rows.",
        out.n_rows
    );
    if cfg.inner.fast_inference {
        if cfg.inner.diagnostics.save_head_embeddings || cfg.inner.xrun.enabled {
            log::info!(
                "fast_inference=true: skipping automatic TOPAZ report generation during main inference run"
            );
        }
    } else if !cfg.inner.auto_report {
        if cfg.inner.diagnostics.save_head_embeddings {
            log::info!(
                "auto_report=false: skipping automatic TOPAZ report generation; run `topaz report` later if needed"
            );
        }
    } else if cfg.inner.diagnostics.save_head_embeddings {
        let report_osw = cfg
            .inner
            .output_osw
            .as_deref()
            .unwrap_or(&cfg.inner.osw_path);
        let topaz_label = if cfg.inner.xrun.enabled {
            "TOPAZ XRUN"
        } else {
            "TOPAZ"
        };
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
            .unwrap_or_else(|| std::path::PathBuf::from("head_embeddings"));
        let head_path = outdir.join("head_embeddings.tsv");
        if head_path.exists() {
            let report_path = outdir.join("topaz_report.html");
            if let Err(e) = write_topaz_report(&TopazReportInputs {
                head_embeddings_path: &head_path,
                report_path: &report_path,
                seed: 0,
                osw_path: Some(report_osw),
                score_tsv_path: Some(&cfg.inner.output_tsv),
                topaz_table_name: Some(topaz_table_name),
                topaz_label: Some(topaz_label),
                topaz_base_table_name: cfg.inner.output_table_base.as_deref(),
                topaz_base_label: Some("TOPAZ Base"),
                xic_path: Some(&cfg.inner.xic_path),
                xic_paths: cfg.inner.xic_paths.as_deref(),
                xic_map_path: cfg.inner.xic_map_path.as_deref(),
                xim_path: cfg.inner.xim_path.as_deref(),
                xim_paths: cfg.inner.xim_paths.as_deref(),
                xim_map_path: cfg.inner.xim_map_path.as_deref(),
                xic_fetch: &cfg.inner.fetch,
                xim_fetch: &cfg.inner.xim_fetch,
                example_bags: 4,
            }) {
                log::warn!("Failed to write TOPAZ report: {e:#}");
            } else {
                log::info!("Wrote TOPAZ report to {:?}", report_path);
            }
        } else {
            log::warn!(
                "Head embeddings not found at {:?}; skipping report generation",
                head_path
            );
        }
    }
    Ok(())
}
