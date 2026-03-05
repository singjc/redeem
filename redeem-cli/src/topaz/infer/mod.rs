use anyhow::Result;

use redeem_topaz::run_inference;

use self::input::TopazInferConfig;
use crate::topaz::report::write_topaz_report;

pub mod input;

pub fn run(cfg: &TopazInferConfig) -> Result<()> {
    let out = run_inference(&cfg.inner)?;
    eprintln!(
        "[ReDeeM::Topaz] Inference complete. Scored {} rows.",
        out.n_rows
    );
    if cfg.inner.diagnostics.save_head_embeddings {
        let outdir = cfg
            .inner
            .diagnostics
            .head_embeddings_outdir
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("head_embeddings"));
        let head_path = outdir.join("head_embeddings.tsv");
        if head_path.exists() {
            let report_path = outdir.join("topaz_report.html");
            if let Err(e) = write_topaz_report(
                &head_path,
                &report_path,
                0,
                Some(&cfg.inner.osw_path),
                Some(&cfg.inner.output_tsv),
            ) {
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
