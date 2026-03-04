use anyhow::Result;

use redeem_topaz::run_xrun_sweep;

use self::input::TopazXrunSweepConfig;

pub mod input;

pub fn run(cfg: &TopazXrunSweepConfig) -> Result<()> {
    let rows = run_xrun_sweep(&cfg.inner)?;
    eprintln!(
        "[ReDeeM::Topaz] XRUN sweep complete. {} configs evaluated.",
        rows.len()
    );
    Ok(())
}
