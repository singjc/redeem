use anyhow::Result;
use std::path::PathBuf;

use redeem_topaz::run_training;

use self::input::TopazTrainConfig;
use crate::topaz::report::write_topaz_report;

pub mod input;
pub fn run(cfg: &TopazTrainConfig) -> Result<()> {
    let out = run_training(&cfg.inner)?;
    log::info!(
        "[ReDeeM::Topaz] Training complete. Checkpoint: {:?}",
        out.checkpoint_prefix
    );
    if cfg.inner.diagnostics.save_head_embeddings {
        let outdir = cfg
            .inner
            .diagnostics
            .head_embeddings_outdir
            .clone()
            .unwrap_or_else(|| PathBuf::from("head_embeddings"));
        let head_path = outdir.join("head_embeddings.tsv");
        if head_path.exists() {
            let report_path = outdir.join("topaz_report.html");
            if let Err(e) = write_topaz_report(
                &head_path,
                &report_path,
                cfg.inner.seed,
                Some(&cfg.inner.osw_path),
                None,
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
