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
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

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
        fs::write(&path, content)
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
        fs::write(&path, content)
            .context("Failed to write meta file")?;
        debug!("Saved meta for project: {}", project_path);
        Ok(())
    }

    fn path_is_valid(path: &str) -> bool {
        Path::new(path).is_dir()
    }

    pub fn load_thread(&self, project_path: &str) -> Result<Option<Thread>> {
        // Try local first
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

    pub fn save_thread(&self, thread: &Thread, use_local: bool) -> Result<()> {
        let project_path = &thread.project_path;
        let local_dir = Self::local_threadbridge_dir(project_path);
        let should_use_local = use_local || local_dir.exists();

        if should_use_local {
            fs::create_dir_all(&local_dir)
                .context("Failed to create local .threadbridge directory")?;

            let path = Self::local_thread_path(project_path);
            let content = serde_json::to_string_pretty(thread)
                .context("Failed to serialize thread")?;
            fs::write(&path, content)
                .context("Failed to write local thread file")?;

            // Ensure meta.json exists
            let meta = Self::load_meta(project_path)?
                .unwrap_or_else(|| {
                    let name = Path::new(project_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    ProjectMeta::new(name)
                });
            Self::save_meta(project_path, &meta)?;

            // Update registry
            let mut registry = self.load_registry()?;
            registry.register(&meta.project_id, &meta.project_name, project_path);
            self.save_registry(&registry)?;

            info!("Saved thread locally: {} ({} facts, {} sessions)",
                project_path, thread.facts.len(), thread.sessions.len());
        } else {
            let project_dir = self.global_project_dir(project_path);
            fs::create_dir_all(&project_dir)
                .context("Failed to create global project directory")?;

            let path = self.global_thread_path(project_path);
            let content = serde_json::to_string_pretty(thread)
                .context("Failed to serialize thread")?;
            fs::write(&path, content)
                .context("Failed to write global thread file")?;

            info!("Saved thread globally: {} ({} facts)", project_path, thread.facts.len());
        }

        Ok(())
    }

    pub fn load_thread_with_path_fix(&self, project_path: &str) -> Result<Option<Thread>> {
        if let Some(meta) = Self::load_meta(project_path)? {
            let mut registry = self.load_registry()?;
            if let Some(entry) = registry.projects.get(&meta.project_id) {
                if entry.last_known_path != project_path {
                    info!("Project moved: {} -> {}", entry.last_known_path, project_path);
                    registry.update_path(&meta.project_id, project_path);
                    self.save_registry(&registry)?;
                }
            } else {
                registry.register(&meta.project_id, &meta.project_name, project_path);
                self.save_registry(&registry)?;
            }
        }
        self.load_thread(project_path)
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
            for entry in fs::read_dir(&projects_dir)? {
                let entry = entry?;
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

impl Default for StorageManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default storage manager")
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
        let storage = StorageManager::with_base_dir(temp_dir.path().to_path_buf()).unwrap();

        let thread = Thread::new("/test/project".to_string());
        storage.save_thread(&thread, false).unwrap();

        let loaded = storage.load_thread("/test/project").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().project_path, "/test/project");
    }

    #[test]
    fn test_save_and_load_thread_local() {
        let temp_dir = tempdir().unwrap();
        let project_dir = temp_dir.path().join("my_project");
        fs::create_dir_all(&project_dir).unwrap();

        let storage = StorageManager::with_base_dir(temp_dir.path().join(".threadbridge")).unwrap();
        let project_path = project_dir.to_str().unwrap();
        let thread = Thread::new(project_path.to_string());
        storage.save_thread(&thread, true).unwrap();

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
