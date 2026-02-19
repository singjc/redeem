use anyhow::Result;
use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

pub fn validate_tsv_or_csv_file(path: &str) -> Result<()> {
    let pb = PathBuf::from(path);

    let ext = pb
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());
    match ext.as_deref() {
        Some("tsv") | Some("csv") => {}
        _ => anyhow::bail!("File must have a .tsv or .csv extension: {}", path),
    }

    if !pb.exists() {
        anyhow::bail!("File does not exist: {}", path);
    }

    Ok(())
}

pub fn write_bytes_to_file(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let path = Path::new(path);
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    Ok(())
}
