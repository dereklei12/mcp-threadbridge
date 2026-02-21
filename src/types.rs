//! Core types for mcp-threadbridge
//!
//! Fact + Session Context Window model:
//! - Each Fact stores surrounding context for richer retrieval
//! - Sessions group facts from a single save_thread call with a summary

use crate::belief::Belief;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single fact extracted from conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub content: String,
    pub category: FactCategory,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    pub utility: Belief,
    pub access_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<DateTime<Utc>>,
    /// Surrounding facts' content for context-augmented retrieval
    pub context_window: Vec<String>,
    /// Links to the Session this fact was created in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Category of facts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FactCategory {
    Architecture,
    Decision,
    General,
}

/// A session groups facts from a single save_thread call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub summary: String,
    pub fact_ids: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Current state of the project
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectState {
    #[serde(default)]
    pub completed: Vec<String>,
    #[serde(default)]
    pub in_progress: Vec<String>,
    #[serde(default)]
    pub pending: Vec<String>,
}

/// A thread representing conversation context for a project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thread {
    pub project_path: String,
    pub project_hash: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub architecture: Option<String>,
    pub state: ProjectState,
    pub facts: Vec<Fact>,
    pub sessions: Vec<Session>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Thread {
    pub fn new(project_path: String) -> Self {
        let project_hash = format!("{:x}", md5::compute(&project_path));
        let now = Utc::now();
        Self {
            project_path,
            project_hash,
            created_at: now,
            updated_at: now,
            architecture: None,
            state: ProjectState::default(),
            facts: Vec::new(),
            sessions: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Search result with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub fact: Fact,
    pub score: f32,
}

/// Project metadata stored locally in .threadbridge/meta.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMeta {
    pub project_id: String,
    pub project_name: String,
    pub created_at: DateTime<Utc>,
}

impl ProjectMeta {
    pub fn new(project_name: String) -> Self {
        Self {
            project_id: uuid::Uuid::new_v4().to_string(),
            project_name,
            created_at: Utc::now(),
        }
    }
}

/// Registry entry for a project in the global registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub last_known_path: String,
    pub last_seen: DateTime<Utc>,
}

/// Global registry mapping project_id -> RegistryEntry
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Registry {
    #[serde(default = "default_registry_version")]
    pub version: u32,
    #[serde(default)]
    pub projects: HashMap<String, RegistryEntry>,
}

fn default_registry_version() -> u32 {
    1
}

impl Registry {
    pub fn new() -> Self {
        Self {
            version: 1,
            projects: HashMap::new(),
        }
    }

    pub fn register(&mut self, project_id: &str, name: &str, path: &str) {
        self.projects.insert(
            project_id.to_string(),
            RegistryEntry {
                name: name.to_string(),
                last_known_path: path.to_string(),
                last_seen: Utc::now(),
            },
        );
    }

    pub fn update_path(&mut self, project_id: &str, new_path: &str) {
        if let Some(entry) = self.projects.get_mut(project_id) {
            entry.last_known_path = new_path.to_string();
            entry.last_seen = Utc::now();
        }
    }
}

/// Project status in list_threads result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProjectStatus {
    Valid,
    Invalid,
}

/// Thread info with path status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInfo {
    pub project_id: String,
    pub project_name: String,
    pub project_path: String,
    pub status: ProjectStatus,
    pub facts_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub has_architecture: bool,
}
