//! Shared utility functions

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Resolve a relative file path within a project directory, rejecting path traversal.
/// Returns None if the resolved path escapes the project directory or doesn't exist.
pub(crate) fn safe_join(project_path: &str, relative: &str) -> Option<PathBuf> {
    let base = Path::new(project_path).canonicalize().ok()?;
    let joined = base.join(relative);
    let resolved = joined.canonicalize().ok()?;
    if resolved.starts_with(&base) {
        Some(resolved)
    } else {
        None
    }
}

/// Write data to a file atomically: write to a temp file, then rename.
/// Prevents data loss if the process crashes mid-write.
pub(crate) fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    let mut file = std::fs::File::create(&tmp_path)
        .context("Failed to create temp file")?;
    file.write_all(data)
        .context("Failed to write temp file")?;
    file.sync_all()
        .context("Failed to sync temp file")?;
    std::fs::rename(&tmp_path, path)
        .context("Failed to rename temp file to target")?;
    Ok(())
}
