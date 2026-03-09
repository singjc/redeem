use anyhow::Result;

use redeem_topaz::run_xrun_training;

use self::input::TopazXrunTrainConfig;

pub mod input;

pub fn run(cfg: &TopazXrunTrainConfig) -> Result<()> {
    let out = run_xrun_training(&cfg.inner)?;
    eprintln!(
        "[ReDeeM::Topaz] XRUN training complete. Sidecar saved alongside {:?} (best_val={:.4}).",
        out.checkpoint_prefix,
        out.best_val
    );
    Ok(())
}
