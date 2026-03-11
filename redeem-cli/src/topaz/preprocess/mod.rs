use anyhow::Result;

use redeem_topaz::run_preprocess;

use self::input::TopazPreprocessConfig;

pub mod input;

pub fn run(cfg: &TopazPreprocessConfig) -> Result<()> {
    let out = run_preprocess(&cfg.inner)?;
    eprintln!(
        "[ReDeeM::Topaz] Preprocessing complete. Bundle: {:?} (rows={}).",
        out.output_path, out.n_rows
    );
    Ok(())
}
