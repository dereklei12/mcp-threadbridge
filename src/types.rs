//! Core types for mcp-threadbridge
//!
//! Fact + Session Context Window model:
//! - Each Fact stores surrounding context for richer retrieval
//! - Sessions group facts from a single save_thread call with a summary

use crate::anchor::Anchor;
use crate::belief::Belief;
use crate::cusum::ChangeTracker;
use crate::file_manifest::FileManifest;
use crate::provenance::Provenance;
use crate::revision::RevisionGraph;
use crate::sheaf::CohomologyResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ================================================================
// Verification system enums
// ================================================================

/// How one fact relates to another in the revision graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RevisionRelation {
    /// This fact supersedes (replaces) the target fact
    Supersedes,
    /// This fact supports / is consistent with the target
    Supports,
    /// This fact contradicts the target
    Contradicts,
}

/// Status of a fact in the revision chain
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RevisionStatus {
    /// Current, active fact
    Active,
    /// Superseded by a newer fact (soft-deleted)
    Superseded,
    /// Retracted explicitly by the user or system
    Retracted,
}

impl Default for RevisionStatus {
    fn default() -> Self {
        RevisionStatus::Active
    }
}

/// Type of dependency a fact has on a code location
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DependencyType {
    /// Fact references a specific function/struct/type definition
    DefinesSemantic,
    /// Fact describes behavior that depends on this code
    DependsOnBehavior,
    /// Fact references a file path directly
    ReferencesPath,
    /// Fact mentions a configuration key/value
    ReferencesConfig,
    /// Weak association (keyword co-occurrence, lowest confidence)
    KeywordAssociation,
}

/// Result of a single provenance dependency check
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DependencyStatus {
    /// Code location still exists and content hash matches
    Intact,
    /// Code location exists but content changed
    Modified,
    /// Code location (file/function) no longer exists
    Missing,
    /// Not yet verified
    Unknown,
}

impl Default for DependencyStatus {
    fn default() -> Self {
        DependencyStatus::Unknown
    }
}

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
    /// Code anchors for fact verification (LEGACY, kept for backward compat)
    #[serde(default)]
    pub anchors: Vec<Anchor>,

    // === Verification v2 fields ===

    /// Structured provenance replacing anchors
    #[serde(default)]
    pub provenance: Provenance,
    /// Revision status: Active, Superseded, or Retracted
    #[serde(default)]
    pub revision_status: RevisionStatus,
    /// IDs of facts that this fact supersedes
    #[serde(default)]
    pub supersedes: Vec<String>,
    /// IDs of facts that superseded this one (multiple new facts can jointly replace one old fact)
    #[serde(default, deserialize_with = "deserialize_superseded_by")]
    pub superseded_by: Vec<String>,
}

/// Deserialize superseded_by from either a single string (legacy Option<String>),
/// null, or a Vec<String> (current format).
fn deserialize_superseded_by<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: serde_json::Value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(Vec::new()),
        serde_json::Value::String(s) => Ok(vec![s]),
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|v| match v {
                serde_json::Value::String(s) => Ok(s),
                _ => Err(serde::de::Error::custom("expected string in array")),
            })
            .collect(),
        _ => Err(serde::de::Error::custom("expected null, string, or array")),
    }
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

/// A state item with optional code anchors for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateItem {
    pub content: String,
    #[serde(default)]
    pub anchors: Vec<Anchor>,
}

impl StateItem {
    pub fn new(content: String) -> Self {
        Self { content, anchors: Vec::new() }
    }
}

/// Deserialize Vec<StateItem> from either Vec<String> (legacy) or Vec<StateItem>
fn deserialize_state_items<'de, D>(deserializer: D) -> Result<Vec<StateItem>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values: Vec<serde_json::Value> = match Option::<Vec<serde_json::Value>>::deserialize(deserializer)? {
        Some(v) => v,
        None => return Ok(Vec::new()),
    };
    values
        .into_iter()
        .map(|v| match v {
            serde_json::Value::String(s) => Ok(StateItem::new(s)),
            serde_json::Value::Object(_) => {
                serde_json::from_value(v).map_err(serde::de::Error::custom)
            }
            _ => Err(serde::de::Error::custom("expected string or object")),
        })
        .collect()
}

/// Current state of the project
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectState {
    #[serde(default, deserialize_with = "deserialize_state_items")]
    pub completed: Vec<StateItem>,
    #[serde(default, deserialize_with = "deserialize_state_items")]
    pub in_progress: Vec<StateItem>,
    #[serde(default, deserialize_with = "deserialize_state_items")]
    pub pending: Vec<StateItem>,
}

/// A thread representing conversation context for a project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thread {
    pub project_path: String,
    pub project_hash: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub architecture: Option<String>,
    #[serde(default)]
    pub architecture_anchors: Vec<Anchor>,
    pub state: ProjectState,
    pub facts: Vec<Fact>,
    pub sessions: Vec<Session>,
    pub metadata: HashMap<String, serde_json::Value>,

    // === Verification v2 fields ===

    /// Revision graph: edges between facts
    #[serde(default)]
    pub revision_graph: RevisionGraph,
    /// Change tracker for file modification detection (VCS-independent)
    #[serde(default, alias = "cusum_tracker")]
    pub change_tracker: ChangeTracker,
    /// Latest sheaf cohomology result
    #[serde(default)]
    pub cohomology: CohomologyResult,
    /// File hash snapshot (primary change detection, VCS-independent)
    #[serde(default)]
    pub file_manifest: Option<FileManifest>,
    /// Schema version: 1 = legacy, 2 = verification v2
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

fn default_schema_version() -> u32 {
    1
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
            architecture_anchors: Vec::new(),
            state: ProjectState::default(),
            facts: Vec::new(),
            sessions: Vec::new(),
            metadata: HashMap::new(),
            revision_graph: RevisionGraph::default(),
            change_tracker: ChangeTracker::default(),
            cohomology: CohomologyResult::default(),
            file_manifest: None,
            schema_version: 2,
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
