use anyhow::Result;

use redeem_topaz::run_training;

use self::input::TopazTrainConfig;

pub mod input;

pub fn run(cfg: &TopazTrainConfig) -> Result<()> {
    let out = run_training(&cfg.inner)?;
    eprintln!("[ReDeeM::Topaz] Training complete. Checkpoint: {:?}", out.checkpoint_prefix);
    Ok(())
}
