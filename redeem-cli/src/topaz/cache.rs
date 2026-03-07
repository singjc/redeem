use anyhow::{Context, Result};
use std::path::Path;

pub fn clear_xic_cache(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(dir)
        .with_context(|| format!("failed to remove xic cache dir {:?}", dir))?;
    Ok(())
}
