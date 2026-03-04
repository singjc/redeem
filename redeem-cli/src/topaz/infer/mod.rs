use anyhow::Result;

use redeem_topaz::run_inference;

use self::input::TopazInferConfig;

pub mod input;

pub fn run(cfg: &TopazInferConfig) -> Result<()> {
    let out = run_inference(&cfg.inner)?;
    eprintln!(
        "[ReDeeM::Topaz] Inference complete. Scored {} rows.",
        out.n_rows
    );
    Ok(())
}
