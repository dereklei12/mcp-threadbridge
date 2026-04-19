//! Storage Manager for persisting threads to disk
//!
//! Supports both local storage (in project's .threadbridge/) and global storage.
//!
//! ## Storage Structure
//!
//! ```text
//! .threadbridge/
//! ├── meta.json        # Project metadata
//! └── thread.json      # Conversation context, facts, and sessions
//! ```

use crate::types::{ProjectMeta, ProjectStatus, Registry, Thread, ThreadInfo};
use crate::util::atomic_write;
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Where to store thread data.
///
/// Resolved from `THREADBRIDGE_STORAGE` env var:
/// - `"true"` (default): local storage at `<project_path>/.threadbridge/`
/// - `"false"`: global storage at `~/.threadbridge/projects/<hash>/`
/// - absolute path: custom storage at `<path>/.threadbridge/`
#[derive(Debug, Clone)]
pub enum StorageMode {
    Local,
    Global,
    Custom(PathBuf),
}

impl StorageMode {
    /// Resolve storage mode from the `THREADBRIDGE_STORAGE` environment variable.
    /// Defaults to `Local` if not set.
    pub fn from_env() -> Self {
        match std::env::var("THREADBRIDGE_STORAGE") {
            Ok(val) => Self::parse(&val),
            Err(_) => Self::Local,
        }
    }

    /// Parse a string value into a StorageMode.
    pub fn parse(val: &str) -> Self {
        match val.to_lowercase().as_str() {
            "true" => Self::Local,
            "false" => Self::Global,
            path => {
                let p = Path::new(path);
                if p.is_absolute() {
                    Self::Custom(p.to_path_buf())
                } else {
                    tracing::warn!("THREADBRIDGE_STORAGE value '{}' is not an absolute path, defaulting to local", val);
                    Self::Local
                }
            }
        }
    }
}

pub struct StorageManager {
    base_dir: PathBuf,
}

impl StorageManager {
    pub fn new() -> Result<Self> {
        let base_dir = dirs::home_dir()
            .context("Could not find home directory")?
            .join(".threadbridge");
        Self::with_base_dir(base_dir)
    }

    pub fn with_base_dir(base_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&base_dir)
            .context("Failed to create storage directory")?;
        info!("Storage initialized at: {:?}", base_dir);
        Ok(Self { base_dir })
    }

    fn local_threadbridge_dir(project_path: &str) -> PathBuf {
        Path::new(project_path).join(".threadbridge")
    }

    fn local_thread_path(project_path: &str) -> PathBuf {
        Self::local_threadbridge_dir(project_path).join("thread.json")
    }

    fn local_meta_path(project_path: &str) -> PathBuf {
        Self::local_threadbridge_dir(project_path).join("meta.json")
    }

    fn registry_path(&self) -> PathBuf {
        self.base_dir.join("registry.json")
    }

    fn global_thread_path(&self, project_path: &str) -> PathBuf {
        let hash = format!("{:x}", md5::compute(project_path));
        self.base_dir.join("projects").join(&hash).join("thread.json")
    }

    fn global_project_dir(&self, project_path: &str) -> PathBuf {
        let hash = format!("{:x}", md5::compute(project_path));
        self.base_dir.join("projects").join(&hash)
    }

    pub fn load_registry(&self) -> Result<Registry> {
        let path = self.registry_path();
        if !path.exists() {
            return Ok(Registry::new());
        }
        let content = fs::read_to_string(&path)
            .context("Failed to read registry file")?;
        let registry: Registry = serde_json::from_str(&content)
            .context("Failed to parse registry file")?;
        Ok(registry)
    }

    pub fn save_registry(&self, registry: &Registry) -> Result<()> {
        let path = self.registry_path();
        let content = serde_json::to_string_pretty(registry)
            .context("Failed to serialize registry")?;
        atomic_write(&path, content.as_bytes())
            .context("Failed to write registry file")?;
        debug!("Saved registry with {} projects", registry.projects.len());
        Ok(())
    }

    pub fn load_meta(project_path: &str) -> Result<Option<ProjectMeta>> {
        let path = Self::local_meta_path(project_path);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&path)
            .context("Failed to read meta file")?;
        let meta: ProjectMeta = serde_json::from_str(&content)
            .context("Failed to parse meta file")?;
        Ok(Some(meta))
    }

    pub fn save_meta(project_path: &str, meta: &ProjectMeta) -> Result<()> {
        let dir = Self::local_threadbridge_dir(project_path);
        fs::create_dir_all(&dir)
            .context("Failed to create local .threadbridge directory")?;
        let path = Self::local_meta_path(project_path);
        let content = serde_json::to_string_pretty(meta)
            .context("Failed to serialize meta")?;
        atomic_write(&path, content.as_bytes())
            .context("Failed to write meta file")?;
        debug!("Saved meta for project: {}", project_path);
        Ok(())
    }

    /// Return existing meta or create + persist a new one.
    /// Only writes to disk when meta.json does not yet exist.
    fn ensure_meta(project_path: &str) -> Result<ProjectMeta> {
        if let Some(meta) = Self::load_meta(project_path)? {
            return Ok(meta);
        }
        let name = Path::new(project_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let meta = ProjectMeta::new(name);
        Self::save_meta(project_path, &meta)?;
        Ok(meta)
    }

    /// Ensure meta.json exists, then update the global registry.
    /// Called after thread.json is already persisted.
    /// Collects secondary write failures into a single error message.
    fn ensure_meta_and_registry(&self, project_path: &str) -> Result<()> {
        let mut errors: Vec<String> = Vec::new();

        let meta = match Self::ensure_meta(project_path) {
            Ok(m) => Some(m),
            Err(e) => {
                errors.push(format!("meta.json: {}", e));
                None
            }
        };

        if let Some(meta) = meta {
            let registry_result = self.load_registry()
                .and_then(|mut registry| {
                    registry.register(&meta.project_id, &meta.project_name, project_path);
                    self.save_registry(&registry)
                });
            if let Err(e) = registry_result {
                errors.push(format!("registry.json: {}", e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(
                "thread.json saved, but secondary writes failed: {}",
                errors.join("; ")
            )
        }
    }

    fn path_is_valid(path: &str) -> bool {
        Path::new(path).is_dir()
    }

    /// Resolve the .threadbridge directory for a custom storage path.
    fn custom_threadbridge_dir(custom_path: &Path) -> PathBuf {
        custom_path.join(".threadbridge")
    }

    fn custom_thread_path(custom_path: &Path) -> PathBuf {
        Self::custom_threadbridge_dir(custom_path).join("thread.json")
    }

    pub fn load_thread(&self, project_path: &str) -> Result<Option<Thread>> {
        self.load_thread_with_mode(project_path, &StorageMode::from_env())
    }

    pub fn load_thread_with_mode(&self, project_path: &str, mode: &StorageMode) -> Result<Option<Thread>> {
        // For Custom mode, check custom path first
        if let StorageMode::Custom(ref custom_path) = mode {
            let custom_thread = Self::custom_thread_path(custom_path);
            if custom_thread.exists() {
                let content = fs::read_to_string(&custom_thread)
                    .context("Failed to read custom thread file")?;
                let thread: Thread = serde_json::from_str(&content)
                    .context("Failed to parse custom thread file")?;
                debug!("Loaded thread from custom storage: {:?} ({} facts)", custom_path, thread.facts.len());
                return Ok(Some(thread));
            }
        }

        // Try local
        let local_path = Self::local_thread_path(project_path);
        if local_path.exists() {
            let content = fs::read_to_string(&local_path)
                .context("Failed to read local thread file")?;
            let thread: Thread = serde_json::from_str(&content)
                .context("Failed to parse local thread file")?;
            debug!("Loaded thread from local storage: {} ({} facts)", project_path, thread.facts.len());
            return Ok(Some(thread));
        }

        // Fall back to global
        let global_path = self.global_thread_path(project_path);
        if global_path.exists() {
            let content = fs::read_to_string(&global_path)
                .context("Failed to read global thread file")?;
            let thread: Thread = serde_json::from_str(&content)
                .context("Failed to parse global thread file")?;
            debug!("Loaded thread from global storage: {} ({} facts)", project_path, thread.facts.len());
            return Ok(Some(thread));
        }

        debug!("No thread found for project: {}", project_path);
        Ok(None)
    }

    pub fn save_thread(&self, thread: &Thread, mode: &StorageMode) -> Result<()> {
        let project_path = &thread.project_path;

        match mode {
            StorageMode::Local => {
                let local_dir = Self::local_threadbridge_dir(project_path);
                self.save_thread_to_dir(&local_dir, thread)?;
                info!("Saved thread locally: {} ({} facts, {} sessions)",
                    project_path, thread.facts.len(), thread.sessions.len());
            }
            StorageMode::Global => {
                let project_dir = self.global_project_dir(project_path);
                fs::create_dir_all(&project_dir)
                    .context("Failed to create global project directory")?;

                let path = self.global_thread_path(project_path);
                let content = serde_json::to_string_pretty(thread)
                    .context("Failed to serialize thread")?;
                atomic_write(&path, content.as_bytes())
                    .context("Failed to write global thread file")?;

                self.ensure_meta_and_registry(project_path)?;

                info!("Saved thread globally: {} ({} facts)", project_path, thread.facts.len());
            }
            StorageMode::Custom(ref custom_path) => {
                let custom_dir = Self::custom_threadbridge_dir(custom_path);
                self.save_thread_to_dir(&custom_dir, thread)?;
                info!("Saved thread to custom path: {:?} ({} facts, {} sessions)",
                    custom_path, thread.facts.len(), thread.sessions.len());
            }
        }

        Ok(())
    }

    /// Save thread to a .threadbridge directory (shared by Local and Custom modes).
    fn save_thread_to_dir(&self, dir: &Path, thread: &Thread) -> Result<()> {
        let project_path = &thread.project_path;
        fs::create_dir_all(dir)
            .context("Failed to create .threadbridge directory")?;

        // Critical write: thread.json is the source of truth
        let path = dir.join("thread.json");
        let content = serde_json::to_string_pretty(thread)
            .context("Failed to serialize thread")?;
        atomic_write(&path, content.as_bytes())
            .context("Failed to write thread file")?;

        // Secondary writes: meta.json (ensure exists) + registry.json (update index)
        self.ensure_meta_and_registry(project_path)?;

        Ok(())
    }

    pub fn load_thread_with_path_fix(&self, project_path: &str, mode: &StorageMode) -> Result<Option<Thread>> {
        let mut old_path: Option<String> = None;

        if let Some(meta) = Self::load_meta(project_path)? {
            let mut registry = self.load_registry()?;
            if let Some(entry) = registry.projects.get(&meta.project_id) {
                if entry.last_known_path != project_path {
                    info!("Project moved: {} -> {}", entry.last_known_path, project_path);
                    old_path = Some(entry.last_known_path.clone());
                    registry.update_path(&meta.project_id, project_path);
                    self.save_registry(&registry)?;
                }
            } else {
                registry.register(&meta.project_id, &meta.project_name, project_path);
                self.save_registry(&registry)?;
            }
        }

        // Global mode: data lives at md5(old_path), so load from there
        let load_path = match (&old_path, mode) {
            (Some(old), StorageMode::Global) => old.as_str(),
            _ => project_path,
        };

        let thread = self.load_thread_with_mode(load_path, mode)?;
        Ok(thread.map(|mut t| {
            if t.project_path != project_path {
                info!("Fixing thread project_path: {} -> {}", t.project_path, project_path);
                t.project_path = project_path.to_string();
                t.project_hash = format!("{:x}", md5::compute(project_path));
            }
            t
        }))
    }

    pub fn list_projects_with_status(&self) -> Result<Vec<ThreadInfo>> {
        let mut threads_info = Vec::new();
        let registry = self.load_registry()?;

        for (project_id, entry) in &registry.projects {
            let status = if Self::path_is_valid(&entry.last_known_path) {
                ProjectStatus::Valid
            } else {
                ProjectStatus::Invalid
            };

            let (facts_count, created_at, updated_at, has_architecture) =
                if status == ProjectStatus::Valid {
                    if let Ok(Some(thread)) = self.load_thread(&entry.last_known_path) {
                        (thread.facts.len(), thread.created_at, thread.updated_at, thread.architecture.is_some())
                    } else {
                        (0, chrono::Utc::now(), chrono::Utc::now(), false)
                    }
                } else {
                    (0, entry.last_seen, entry.last_seen, false)
                };

            threads_info.push(ThreadInfo {
                project_id: project_id.clone(),
                project_name: entry.name.clone(),
                project_path: entry.last_known_path.clone(),
                status,
                facts_count,
                created_at,
                updated_at,
                has_architecture,
            });
        }

        // Check global storage for projects not in registry
        let projects_dir = self.base_dir.join("projects");
        if projects_dir.exists() {
            let read_dir = match fs::read_dir(&projects_dir) {
                Ok(rd) => rd,
                Err(e) => {
                    tracing::warn!("Failed to read projects directory {:?}: {}", projects_dir, e);
                    return Ok(threads_info);
                }
            };
            for entry in read_dir {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!("Failed to read directory entry in {:?}: {}", projects_dir, e);
                        continue;
                    }
                };
                let thread_path = entry.path().join("thread.json");
                if thread_path.exists() {
                    if let Ok(content) = fs::read_to_string(&thread_path) {
                        if let Ok(thread) = serde_json::from_str::<Thread>(&content) {
                            let already_listed = threads_info.iter()
                                .any(|t| t.project_path == thread.project_path);
                            if !already_listed {
                                let status = if Self::path_is_valid(&thread.project_path) {
                                    ProjectStatus::Valid
                                } else {
                                    ProjectStatus::Invalid
                                };
                                threads_info.push(ThreadInfo {
                                    project_id: thread.project_hash.clone(),
                                    project_name: Path::new(&thread.project_path)
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("unknown")
                                        .to_string(),
                                    project_path: thread.project_path,
                                    status,
                                    facts_count: thread.facts.len(),
                                    created_at: thread.created_at,
                                    updated_at: thread.updated_at,
                                    has_architecture: thread.architecture.is_some(),
                                });
                            }
                        }
                    }
                }
            }
        }

        threads_info.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(threads_info)
    }

    pub fn get_searchable_projects(&self) -> Result<(Vec<String>, Vec<String>)> {
        let threads = self.list_projects_with_status()?;
        let valid: Vec<String> = threads.iter()
            .filter(|t| t.status == ProjectStatus::Valid)
            .map(|t| t.project_path.clone())
            .collect();
        let invalid: Vec<String> = threads.iter()
            .filter(|t| t.status == ProjectStatus::Invalid)
            .map(|t| t.project_path.clone())
            .collect();
        Ok((valid, invalid))
    }

}


impl Clone for StorageManager {
    fn clone(&self) -> Self {
        Self {
            base_dir: self.base_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load_thread_global() {
        let temp_dir = tempdir().unwrap();
        let project_dir = temp_dir.path().join("my_project");
        fs::create_dir_all(&project_dir).unwrap();
        let project_path = project_dir.to_str().unwrap();

        let storage = StorageManager::with_base_dir(temp_dir.path().join(".threadbridge")).unwrap();

        let thread = Thread::new(project_path.to_string());
        storage.save_thread(&thread, &StorageMode::Global).unwrap();

        let loaded = storage.load_thread_with_mode(project_path, &StorageMode::Global).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().project_path, project_path);
    }

    #[test]
    fn test_save_and_load_thread_local() {
        let temp_dir = tempdir().unwrap();
        let project_dir = temp_dir.path().join("my_project");
        fs::create_dir_all(&project_dir).unwrap();

        let storage = StorageManager::with_base_dir(temp_dir.path().join(".threadbridge")).unwrap();
        let project_path = project_dir.to_str().unwrap();
        let thread = Thread::new(project_path.to_string());
        storage.save_thread(&thread, &StorageMode::Local).unwrap();

        let local_thread_path = project_dir.join(".threadbridge").join("thread.json");
        assert!(local_thread_path.exists());

        let local_meta_path = project_dir.join(".threadbridge").join("meta.json");
        assert!(local_meta_path.exists());

        let registry = storage.load_registry().unwrap();
        assert_eq!(registry.projects.len(), 1);
    }

    #[test]
    fn test_registry() {
        let temp_dir = tempdir().unwrap();
        let storage = StorageManager::with_base_dir(temp_dir.path().to_path_buf()).unwrap();

        let mut registry = Registry::new();
        registry.register("id1", "Project 1", "/path/to/project1");
        registry.register("id2", "Project 2", "/path/to/project2");

        storage.save_registry(&registry).unwrap();

        let loaded = storage.load_registry().unwrap();
        assert_eq!(loaded.projects.len(), 2);
        assert_eq!(loaded.projects.get("id1").unwrap().name, "Project 1");
    }
}
