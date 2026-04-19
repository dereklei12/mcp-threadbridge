//! File Manifest: VCS-independent file hash snapshots for change detection.
//!
//! Each save_thread computes a manifest (file path → content hash) from ProjectScan.
//! On load_thread, a new scan is diffed against the stored manifest to detect changes.
//! This replaces git as the primary CUSUM data source, working on all projects
//! regardless of VCS (git, hg, svn, none).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A snapshot of project file content hashes.
/// Computed from ProjectScan's files_raw at zero additional I/O cost.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileManifest {
    /// file relative path → content hash (MD5, first 16 hex chars)
    #[serde(default)]
    pub files: HashMap<String, String>,
    /// When this snapshot was created
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
}

impl FileManifest {
    /// Compute manifest from ProjectScan's raw file contents.
    /// This is the primary constructor — called during save_thread after ProjectScan.
    pub fn from_scan(files_raw: &[(String, String)]) -> Self {
        let files: HashMap<String, String> = files_raw
            .iter()
            .map(|(path, content)| {
                let digest = md5::compute(content.as_bytes());
                let hash = format!("{:x}", digest);
                (path.clone(), hash[..16].to_string())
            })
            .collect();
        Self {
            files,
            created_at: Utc::now(),
        }
    }

    /// Diff this manifest against a previous one.
    /// Returns which files were changed, added, or removed.
    pub fn diff(&self, previous: &FileManifest) -> ManifestDiff {
        let mut changed = HashSet::new();
        let mut added = HashSet::new();
        let mut removed = HashSet::new();

        for (path, hash) in &self.files {
            match previous.files.get(path) {
                Some(old_hash) if old_hash != hash => {
                    changed.insert(path.clone());
                }
                None => {
                    added.insert(path.clone());
                }
                _ => {}
            }
        }

        for path in previous.files.keys() {
            if !self.files.contains_key(path) {
                removed.insert(path.clone());
            }
        }

        ManifestDiff {
            changed,
            added,
            removed,
        }
    }

    /// Number of files in the manifest
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

/// Result of comparing two manifests
pub struct ManifestDiff {
    /// Files whose content hash changed
    pub changed: HashSet<String>,
    /// Files present in current but not in previous
    pub added: HashSet<String>,
    /// Files present in previous but not in current
    pub removed: HashSet<String>,
}

impl ManifestDiff {
    /// All affected files (union of changed + added + removed)
    pub fn all_affected(&self) -> HashSet<String> {
        let mut all = self.changed.clone();
        all.extend(self.added.iter().cloned());
        all.extend(self.removed.iter().cloned());
        all
    }

    /// Whether any files changed
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.added.is_empty() && self.removed.is_empty()
    }

    /// Total number of affected files
    pub fn total_changes(&self) -> usize {
        self.changed.len() + self.added.len() + self.removed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_scan() {
        let files = vec![
            ("src/main.rs".to_string(), "fn main() {}".to_string()),
            ("Cargo.toml".to_string(), "[package]".to_string()),
        ];
        let manifest = FileManifest::from_scan(&files);
        assert_eq!(manifest.file_count(), 2);
        assert!(manifest.files.contains_key("src/main.rs"));
        assert!(manifest.files.contains_key("Cargo.toml"));
        // Hashes should be 16 chars
        for hash in manifest.files.values() {
            assert_eq!(hash.len(), 16);
        }
    }

    #[test]
    fn test_diff_no_changes() {
        let files = vec![("a.rs".to_string(), "hello".to_string())];
        let m1 = FileManifest::from_scan(&files);
        let m2 = FileManifest::from_scan(&files);
        let diff = m2.diff(&m1);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_changed_file() {
        let m1 = FileManifest::from_scan(&[("a.rs".to_string(), "v1".to_string())]);
        let m2 = FileManifest::from_scan(&[("a.rs".to_string(), "v2".to_string())]);
        let diff = m2.diff(&m1);
        assert_eq!(diff.changed.len(), 1);
        assert!(diff.changed.contains("a.rs"));
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_diff_added_file() {
        let m1 = FileManifest::from_scan(&[("a.rs".to_string(), "hello".to_string())]);
        let m2 = FileManifest::from_scan(&[
            ("a.rs".to_string(), "hello".to_string()),
            ("b.rs".to_string(), "world".to_string()),
        ]);
        let diff = m2.diff(&m1);
        assert!(diff.changed.is_empty());
        assert_eq!(diff.added.len(), 1);
        assert!(diff.added.contains("b.rs"));
    }

    #[test]
    fn test_diff_removed_file() {
        let m1 = FileManifest::from_scan(&[
            ("a.rs".to_string(), "hello".to_string()),
            ("b.rs".to_string(), "world".to_string()),
        ]);
        let m2 = FileManifest::from_scan(&[("a.rs".to_string(), "hello".to_string())]);
        let diff = m2.diff(&m1);
        assert!(diff.changed.is_empty());
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 1);
        assert!(diff.removed.contains("b.rs"));
    }

    #[test]
    fn test_diff_all_affected() {
        let m1 = FileManifest::from_scan(&[
            ("a.rs".to_string(), "v1".to_string()),
            ("b.rs".to_string(), "old".to_string()),
        ]);
        let m2 = FileManifest::from_scan(&[
            ("a.rs".to_string(), "v2".to_string()),
            ("c.rs".to_string(), "new".to_string()),
        ]);
        let diff = m2.diff(&m1);
        let all = diff.all_affected();
        assert_eq!(all.len(), 3); // a.rs changed, b.rs removed, c.rs added
        assert!(all.contains("a.rs"));
        assert!(all.contains("b.rs"));
        assert!(all.contains("c.rs"));
    }

    #[test]
    fn test_same_content_same_hash() {
        let m1 = FileManifest::from_scan(&[("a.rs".to_string(), "content".to_string())]);
        let m2 = FileManifest::from_scan(&[("a.rs".to_string(), "content".to_string())]);
        assert_eq!(m1.files["a.rs"], m2.files["a.rs"]);
    }
}
