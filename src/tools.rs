//! MCP Tool definitions and handlers
//!
//! Three tools: save_thread, load_thread, search_memory
//! Core pipeline: Arctic Embed M + Dense/BM25 Hybrid + RRF + BLL v2

use crate::anchor::{self, ProjectScan};
use crate::bll::BayesianLastLayer;
use crate::config::Config;
use crate::embedding::{self, EmbeddingService};
use crate::mcp::Tool;
use crate::file_manifest::FileManifest;
use crate::provenance;
use crate::revision;
use crate::storage::{StorageManager, StorageMode};
use crate::types::{Fact, FactCategory, ProjectState, RevisionStatus, Session, StateItem, Thread};
use crate::vector_store::VectorStore;
use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info, warn};

// ================================================================
// Global BLL singleton
// ================================================================

static BLL_INSTANCE: OnceLock<Mutex<Option<BayesianLastLayer>>> = OnceLock::new();

fn get_bll() -> &'static Mutex<Option<BayesianLastLayer>> {
    BLL_INSTANCE.get_or_init(|| {
        let config = Config::global();
        if !config.search.bll_enabled {
            debug!("BLL disabled by config");
            return Mutex::new(None);
        }

        let weights_path = Path::new(&config.search.bll_weights_path);
        match BayesianLastLayer::try_load(weights_path) {
            Some(mut bll) => {
                let posterior_path = dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".threadbridge")
                    .join("bll_posterior.bin");
                if let Err(e) = bll.load_posterior(&posterior_path) {
                    debug!("No saved BLL posterior: {}", e);
                }
                info!("BLL reranker loaded ({} prior updates)", bll.update_count());
                Mutex::new(Some(bll))
            }
            None => {
                debug!("BLL weights not available, reranking disabled");
                Mutex::new(None)
            }
        }
    })
}

// ================================================================
// Search buffer for implicit BLL feedback
// ================================================================

#[derive(Debug, Clone)]
struct SearchRecord {
    query_embedding: Vec<f32>,
    returned_fact_ids: Vec<String>,
}

static SEARCH_BUFFER: OnceLock<Mutex<Vec<SearchRecord>>> = OnceLock::new();

fn search_buffer() -> &'static Mutex<Vec<SearchRecord>> {
    SEARCH_BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

// ================================================================
// Background task queue
// ================================================================

enum BackgroundTask {
    /// After save_thread: implicit feedback + CUSUM + sheaf
    PostSave {
        project_path: String,
        storage_mode: StorageMode,
    },
    /// After load_thread: full provenance verification
    VerifyProvenance {
        project_path: String,
        storage_mode: StorageMode,
    },
}

// ================================================================
// ToolHandler
// ================================================================

#[derive(Clone)]
pub struct ToolHandler {
    storage: Arc<Mutex<StorageManager>>,
    bg_sender: mpsc::Sender<BackgroundTask>,
}

impl ToolHandler {
    pub fn new(storage: StorageManager) -> Self {
        let storage = Arc::new(Mutex::new(storage));
        let (tx, rx) = mpsc::channel::<BackgroundTask>();

        // Single background worker — all bg writes are serialized, no lost updates
        let worker_storage = Arc::clone(&storage);
        std::thread::Builder::new()
            .name("tb-background".into())
            .spawn(move || {
                Self::background_worker(rx, &worker_storage);
            })
            .expect("Failed to spawn background worker thread");

        Self {
            storage,
            bg_sender: tx,
        }
    }

    /// Single worker loop: receives tasks and executes them serially.
    fn background_worker(
        rx: mpsc::Receiver<BackgroundTask>,
        storage: &Arc<Mutex<StorageManager>>,
    ) {
        while let Ok(task) = rx.recv() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match task {
                    BackgroundTask::PostSave { ref project_path, ref storage_mode } => {
                        Self::bg_post_save(storage, project_path, storage_mode);
                    }
                    BackgroundTask::VerifyProvenance { ref project_path, ref storage_mode } => {
                        Self::background_verify_all_provenance(storage, project_path, storage_mode);
                    }
                }
            }));
            if let Err(e) = result {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                warn!("Background task panicked: {}", msg);
            }
        }
        debug!("Background worker shutting down");
    }

    /// Post-save background work: implicit feedback + CUSUM + sheaf.
    /// Loads the latest thread from disk (no stale snapshot).
    fn bg_post_save(
        storage: &Arc<Mutex<StorageManager>>,
        project_path: &str,
        storage_mode: &StorageMode,
    ) {
        // 1. Implicit feedback (BLL weights + per-fact utility)
        Self::process_implicit_feedback(storage, project_path, storage_mode);

        // 2. CUSUM + sheaf
        let config = Config::global();
        if config.verification.cusum_enabled || config.verification.sheaf_enabled {
            if let Ok(storage_guard) = storage.lock() {
                if let Ok(Some(mut thread)) = storage_guard.load_thread_with_path_fix(project_path, storage_mode) {
                    let mut bg_changed = false;

                    if config.verification.cusum_enabled {
                        let scan = ProjectScan::new(project_path);
                        let new_manifest = FileManifest::from_scan(&scan.files_raw);
                        if let Some(ref prev) = thread.file_manifest {
                            let diff = new_manifest.diff(prev);
                            thread.change_tracker.update_from_diff(
                                &diff,
                                config.verification.consecutive_threshold,
                                config.verification.ewma_alpha,
                                config.verification.ewma_threshold,
                            );
                        }
                        thread.file_manifest = Some(new_manifest);
                        bg_changed = true;
                    }

                    if config.verification.sheaf_enabled {
                        thread.cohomology = crate::sheaf::compute_and_store_cohomology(
                            &thread.facts,
                            config.verification.sheaf_edge_threshold,
                            config.verification.sheaf_min_facts,
                        );
                        bg_changed = true;
                    }

                    if bg_changed {
                        let _ = storage_guard.save_thread(&thread, storage_mode);
                    }
                }
            }
        }
    }

    pub fn get_tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "save_thread".to_string(),
                description: r#"Save or update the conversation context for the current project.

When calling this tool, you (the AI) should:
1. Review the entire conversation and extract core architecture and design principles
2. List all key decisions and their reasoning
3. Summarize completed, in-progress, and pending items
4. Provide a concise summary of this session's progress

Each fact should be a single, clear statement that future AI can understand."#.to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path (defaults to current working directory)"
                        },
                        "architecture": {
                            "type": "string",
                            "description": "Core architecture and design philosophy of the project"
                        },
                        "decisions": {
                            "type": "array",
                            "description": "Key decisions made, each with reasoning",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "decision": { "type": "string" },
                                    "reason": { "type": "string" }
                                },
                                "required": ["decision"]
                            }
                        },
                        "facts": {
                            "type": "array",
                            "description": "Important facts to remember (one clear statement each)",
                            "items": { "type": "string" }
                        },
                        "state": {
                            "type": "object",
                            "description": "Current project state",
                            "properties": {
                                "completed": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                },
                                "in_progress": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                },
                                "pending": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                }
                            }
                        },
                        "summary": {
                            "type": "string",
                            "description": "Summary of this session's progress"
                        }
                    },
                    "required": ["summary"]
                }),
            },
            Tool {
                name: "load_thread".to_string(),
                description: r#"Load the saved conversation context for a project.

Returns the project's architecture, key decisions, current state, and recent session summaries.
Use this at the start of a conversation to restore context from previous sessions."#.to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path (defaults to current working directory)"
                        }
                    },
                    "required": []
                }),
            },
            Tool {
                name: "search_memory".to_string(),
                description: r#"Search for relevant facts in the project's memory using semantic search.

Use this when you need to find specific information from past conversations.
Can search across all projects if project_path is not specified.
Results include surrounding context for richer understanding."#.to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project directory path. If not specified, searches across all projects."
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }),
            },
        ]
    }

    // ================================================================
    // save_thread
    // ================================================================

    pub fn handle_save_thread(&self, args: Value) -> Result<Value> {
        let project_path = args["project_path"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| std::env::var("THREADBRIDGE_PROJECT_PATH").ok())
            .unwrap_or_else(|| std::env::current_dir().unwrap().to_string_lossy().to_string());

        let storage_mode = StorageMode::from_env();
        let storage = self.storage.lock().unwrap_or_else(|e| {
            tracing::warn!("Storage mutex was poisoned, recovering");
            e.into_inner()
        });

        // Load existing thread or create new one
        let mut thread = storage
            .load_thread_with_path_fix(&project_path, &storage_mode)?
            .unwrap_or_else(|| Thread::new(project_path.clone()));

        // Update architecture if provided
        if let Some(arch) = args["architecture"].as_str() {
            thread.architecture = Some(arch.to_string());
        }

        // Update state if provided (anchors generated later after ProjectScan)
        if let Some(state) = args.get("state") {
            let parse_items = |key: &str| -> Vec<StateItem> {
                state[key]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(|s| StateItem::new(s.to_string())))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            thread.state = ProjectState {
                completed: parse_items("completed"),
                in_progress: parse_items("in_progress"),
                pending: parse_items("pending"),
            };
        }

        let summary_text = args["summary"].as_str().unwrap_or("").to_string();

        // ================================================================
        // Collect all texts, dedup decisions
        // ================================================================

        let mut seen_decisions: HashSet<String> = HashSet::new();

        struct DecisionEntry {
            fact_content: String,
        }
        let mut decision_entries: Vec<DecisionEntry> = Vec::new();

        if let Some(decisions) = args["decisions"].as_array() {
            for decision in decisions {
                let decision_text = decision["decision"].as_str().unwrap_or("").trim().to_string();
                if decision_text.is_empty() {
                    continue;
                }
                let key = decision_text.to_lowercase();
                if !seen_decisions.insert(key) {
                    continue;
                }
                let reason = decision["reason"].as_str()
                    .map(|r| r.trim().to_string())
                    .filter(|r| !r.is_empty());
                let fact_content = match reason.as_deref() {
                    Some(r) => format!("{} (Reason: {})", decision_text, r),
                    None => decision_text,
                };
                decision_entries.push(DecisionEntry { fact_content });
            }
        }

        let general_facts: Vec<String> = args["facts"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();

        // ================================================================
        // Batch embed ALL texts in one call
        // ================================================================
        // Layout: [decision_contents..., general_facts...]
        let mut all_texts: Vec<String> = Vec::new();
        let decision_offset = 0;
        for entry in &decision_entries {
            all_texts.push(entry.fact_content.clone());
        }
        let general_offset = all_texts.len();
        for content in &general_facts {
            all_texts.push(content.clone());
        }

        let all_embeddings = if !all_texts.is_empty() {
            let refs: Vec<&str> = all_texts.iter().map(|s| s.as_str()).collect();
            match EmbeddingService::embed_batch(&refs) {
                Ok(embs) => embs,
                Err(e) => {
                    warn!("Batch embedding failed: {}, facts will have no embeddings", e);
                    vec![Vec::new(); all_texts.len()]
                }
            }
        } else {
            Vec::new()
        };

        let get_emb = |idx: usize| -> Option<Vec<f32>> {
            all_embeddings.get(idx).and_then(|e| if e.is_empty() { None } else { Some(e.clone()) })
        };

        // ================================================================
        // Create facts with embeddings
        // ================================================================
        let now = Utc::now();
        let session_id = uuid::Uuid::new_v4().to_string();
        let mut session_fact_ids: Vec<String> = Vec::new();
        let session_facts_start_idx = thread.facts.len();

        // Decision facts
        for (i, entry) in decision_entries.iter().enumerate() {
            let fact_id = uuid::Uuid::new_v4().to_string();
            session_fact_ids.push(fact_id.clone());
            let fact = Fact {
                id: fact_id,
                content: entry.fact_content.clone(),
                category: FactCategory::Decision,
                confidence: 0.95,
                created_at: now,
                updated_at: now,
                embedding: get_emb(decision_offset + i),
                utility: crate::belief::Belief::uninformed(),
                access_count: 0,
                last_used_at: None,
                context_window: Vec::new(), // populated below
                session_id: Some(session_id.clone()),
                anchors: Vec::new(), // generated after ProjectScan
                provenance: Default::default(),
                revision_status: Default::default(),
                supersedes: Vec::new(),
                superseded_by: Vec::new(),
            };
            thread.facts.push(fact);
        }

        // General facts
        for (i, content) in general_facts.iter().enumerate() {
            let fact_id = uuid::Uuid::new_v4().to_string();
            session_fact_ids.push(fact_id.clone());
            let fact = Fact {
                id: fact_id,
                content: content.clone(),
                category: FactCategory::General,
                confidence: 0.8,
                created_at: now,
                updated_at: now,
                embedding: get_emb(general_offset + i),
                utility: crate::belief::Belief::uninformed(),
                access_count: 0,
                last_used_at: None,
                context_window: Vec::new(), // populated below
                session_id: Some(session_id.clone()),
                anchors: Vec::new(), // generated after ProjectScan
                provenance: Default::default(),
                revision_status: Default::default(),
                supersedes: Vec::new(),
                superseded_by: Vec::new(),
            };
            thread.facts.push(fact);
        }

        // ================================================================
        // Confidence confirmation: boost existing facts confirmed by new facts
        // ================================================================
        let config = Config::global();
        if session_facts_start_idx > 0 {
            let (existing_facts, new_facts) = thread.facts.split_at_mut(session_facts_start_idx);
            for existing in existing_facts.iter_mut() {
                // Skip superseded facts
                if existing.revision_status != RevisionStatus::Active {
                    continue;
                }
                let existing_emb = match existing.embedding.as_ref() {
                    Some(e) => e,
                    None => continue,
                };
                let max_sim = new_facts.iter()
                    .filter_map(|nf| nf.embedding.as_ref())
                    .map(|ne| embedding::cosine_similarity(existing_emb, ne))
                    .fold(0.0f32, f32::max);

                if max_sim > 0.8 {
                    existing.confidence = (existing.confidence + 0.1).min(1.0);
                    existing.updated_at = now;
                }
            }
        }

        // ================================================================
        // AGM Revision detection: identify supersession pairs
        // ================================================================
        if config.verification.revision_enabled && session_facts_start_idx > 0 {
            let pairs = {
                let (existing, new_part) = thread.facts.split_at(session_facts_start_idx);
                revision::detect_supersession(
                    existing,
                    new_part,
                    config.verification.supersession_threshold,
                )
            };
            for (new_id, old_id) in &pairs {
                revision::apply_agm_revision(
                    &mut thread.facts,
                    &mut thread.revision_graph,
                    new_id,
                    old_id,
                    now,
                );
            }
            if !pairs.is_empty() {
                debug!("AGM revision: {} supersession pairs detected", pairs.len());
            }
        }

        // ================================================================
        // Populate context windows for this session's facts
        // ================================================================
        let n = Config::global().search.context_window_size;
        let session_facts_end_idx = thread.facts.len();
        let session_len = session_facts_end_idx - session_facts_start_idx;

        if session_len > 0 {
            // Collect content strings for the session's facts
            let session_contents: Vec<String> = thread.facts[session_facts_start_idx..session_facts_end_idx]
                .iter()
                .map(|f| f.content.clone())
                .collect();

            for i in 0..session_len {
                let mut window = Vec::new();
                let start = i.saturating_sub(n);
                let end = (i + n).min(session_len - 1);
                for j in start..=end {
                    if j != i {
                        window.push(session_contents[j].clone());
                    }
                }
                thread.facts[session_facts_start_idx + i].context_window = window;
            }
        }

        // ================================================================
        // Generate provenance + legacy anchors (scan project once, reuse for all)
        // ================================================================
        let scan = ProjectScan::new(&project_path);

        // New facts: provenance + legacy anchors
        for fact in &mut thread.facts[session_facts_start_idx..] {
            fact.anchors = scan.generate_anchors(&fact.content);
            if config.verification.provenance_enabled {
                fact.provenance = provenance::generate_provenance(&fact.content, &scan);
            }
        }

        // Architecture anchors (legacy)
        if let Some(ref arch) = thread.architecture {
            thread.architecture_anchors = scan.generate_anchors(arch);
        }

        // State item anchors (legacy)
        for item in &mut thread.state.completed {
            item.anchors = scan.generate_anchors(&item.content);
        }
        for item in &mut thread.state.in_progress {
            item.anchors = scan.generate_anchors(&item.content);
        }
        for item in &mut thread.state.pending {
            item.anchors = scan.generate_anchors(&item.content);
        }

        // CUSUM: ensure all provenance files are tracked
        if config.verification.cusum_enabled {
            for fact in &thread.facts[session_facts_start_idx..] {
                for dep in &fact.provenance.dependencies {
                    thread.change_tracker.ensure_file_tracked(&dep.file);
                }
            }
        }

        // ================================================================
        // Create Session
        // ================================================================
        let session = Session {
            id: session_id,
            summary: summary_text.clone(),
            fact_ids: session_fact_ids,
            created_at: now,
        };
        thread.sessions.push(session);

        thread.schema_version = 2;
        thread.updated_at = now;

        // Save thread
        storage.save_thread(&thread, &storage_mode)?;

        let storage_location = match &storage_mode {
            StorageMode::Local => "local (.threadbridge/)",
            StorageMode::Global => "global (~/.threadbridge/)",
            StorageMode::Custom(p) => {
                info!("Custom storage path: {:?}", p);
                "custom"
            }
        };
        let superseded_count = thread.facts.iter()
            .filter(|f| f.revision_status == RevisionStatus::Superseded)
            .count();
        info!("Saved thread: {} ({} facts, {} superseded, {} sessions) to {}",
            project_path, thread.facts.len(), superseded_count, thread.sessions.len(), storage_location);

        // Build response before spawning background work
        let response = json!({
            "success": true,
            "project_path": project_path,
            "storage_location": storage_location,
            "facts_saved": session_len,
            "total_facts": thread.facts.len(),
            "total_sessions": thread.sessions.len(),
            "superseded_count": superseded_count
        });

        // Queue background work (serialized by worker — no lost updates)
        let _ = self.bg_sender.send(BackgroundTask::PostSave {
            project_path: project_path.clone(),
            storage_mode,
        });

        Ok(response)
    }

    // ================================================================
    // load_thread
    // ================================================================

    pub fn handle_load_thread(&self, args: Value) -> Result<Value> {
        let project_path = args["project_path"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| std::env::var("THREADBRIDGE_PROJECT_PATH").ok())
            .unwrap_or_else(|| std::env::current_dir().unwrap().to_string_lossy().to_string());

        let storage_mode = StorageMode::from_env();
        let storage = self.storage.lock().unwrap_or_else(|e| {
            tracing::warn!("Storage mutex was poisoned, recovering");
            e.into_inner()
        });

        match storage.load_thread_with_path_fix(&project_path, &storage_mode)? {
            Some(mut thread) => {
                let config = Config::global();
                let stale_threshold = config.search.anchor_stale_threshold;

                // CUSUM update via file manifest diff (VCS-independent)
                if config.verification.cusum_enabled {
                    let scan = ProjectScan::new(&project_path);
                    let new_manifest = FileManifest::from_scan(&scan.files_raw);
                    if let Some(ref prev) = thread.file_manifest {
                        let diff = new_manifest.diff(prev);
                        // Always update — even empty diffs reset consecutive counters
                        thread.change_tracker.update_from_diff(
                            &diff,
                            config.verification.consecutive_threshold,
                            config.verification.ewma_alpha,
                            config.verification.ewma_threshold,
                        );
                    }
                    // Don't update stored manifest on load — only on save.
                    // This ensures diff reflects changes since last save.
                }

                // Recent session summaries (last 5)
                let recent_sessions: Vec<Value> = thread.sessions.iter()
                    .rev()
                    .take(5)
                    .map(|s| json!({
                        "summary": s.summary,
                        "facts_count": s.fact_ids.len(),
                        "created_at": s.created_at
                    }))
                    .collect();

                // Recent decisions (last 10, active only) — quick-verify with provenance or anchors
                let decisions: Vec<String> = thread.facts.iter()
                    .filter(|f| f.category == FactCategory::Decision && f.revision_status == RevisionStatus::Active)
                    .rev()
                    .take(10)
                    .map(|f| {
                        // Prefer provenance verification, fall back to anchors
                        if config.verification.provenance_enabled && !f.provenance.dependencies.is_empty() {
                            let mut prov = f.provenance.clone();
                            provenance::verify_provenance(&project_path, &mut prov);
                            if prov.compute_score() < stale_threshold {
                                let broken = prov.broken_patterns();
                                format!("[STALE: {} not found] {}", broken.join(", "), f.content)
                            } else if prov.has_broken_deps() {
                                let broken = prov.broken_patterns();
                                format!("[STALE: {} changed] {}", broken.join(", "), f.content)
                            } else {
                                f.content.clone()
                            }
                        } else if !f.anchors.is_empty() {
                            let verified = anchor::verify_anchors(&project_path, &f.anchors);
                            if anchor::grounding_score(&verified) < stale_threshold {
                                format!("[STALE] {}", f.content)
                            } else if anchor::any_broken(&verified) {
                                let broken = anchor::broken_patterns(&verified);
                                format!("[STALE: {} not found] {}", broken.join(", "), f.content)
                            } else {
                                f.content.clone()
                            }
                        } else {
                            f.content.clone()
                        }
                    })
                    .collect();

                // Verify architecture anchors
                let architecture_stale: Vec<String> = if !thread.architecture_anchors.is_empty() {
                    let verified = anchor::verify_anchors(&project_path, &thread.architecture_anchors);
                    anchor::broken_patterns(&verified)
                        .into_iter()
                        .map(String::from)
                        .collect()
                } else {
                    Vec::new()
                };

                // Verify state items — any broken anchor = stale
                let verify_state_items = |items: &[StateItem]| -> Vec<String> {
                    items.iter().map(|item| {
                        if item.anchors.is_empty() {
                            return item.content.clone();
                        }
                        let verified = anchor::verify_anchors(&project_path, &item.anchors);
                        if anchor::any_broken(&verified) {
                            let broken = anchor::broken_patterns(&verified);
                            format!("[STALE: {} not found] {}", broken.join(", "), item.content)
                        } else {
                            item.content.clone()
                        }
                    }).collect()
                };

                let state_completed = verify_state_items(&thread.state.completed);
                let state_in_progress = verify_state_items(&thread.state.in_progress);
                let state_pending = verify_state_items(&thread.state.pending);

                // Counts
                let active_count = thread.facts.iter()
                    .filter(|f| f.revision_status == RevisionStatus::Active)
                    .count();
                let superseded_count = thread.facts.iter()
                    .filter(|f| f.revision_status == RevisionStatus::Superseded)
                    .count();

                info!("Loaded thread: {} ({} facts, {} active, {} superseded, {} sessions)",
                    project_path, thread.facts.len(), active_count, superseded_count, thread.sessions.len());

                let mut thread_json = json!({
                    "architecture": thread.architecture,
                    "state": {
                        "completed": state_completed,
                        "in_progress": state_in_progress,
                        "pending": state_pending
                    },
                    "recent_sessions": recent_sessions,
                    "recent_decisions": decisions,
                    "total_facts": thread.facts.len(),
                    "total_sessions": thread.sessions.len(),
                    "created_at": thread.created_at,
                    "updated_at": thread.updated_at,
                    "active_facts": active_count,
                    "superseded_facts": superseded_count,
                    "global_consistency": thread.cohomology.global_consistency,
                    "alarm_files": thread.change_tracker.files_in_alarm().collect::<Vec<_>>(),
                    "schema_version": thread.schema_version
                });

                if !architecture_stale.is_empty() {
                    thread_json["architecture_stale_keywords"] = json!(architecture_stale);
                }

                // Queue background provenance verification (serialized by worker)
                let _ = self.bg_sender.send(BackgroundTask::VerifyProvenance {
                    project_path: project_path.to_string(),
                    storage_mode,
                });

                Ok(json!({
                    "success": true,
                    "project_path": project_path,
                    "thread": thread_json
                }))
            }
            None => {
                debug!("No thread found for project: {}", project_path);
                Ok(json!({
                    "success": false,
                    "project_path": project_path
                }))
            }
        }
    }

    /// Background: migrate anchors → provenance, backfill, verify, update half-life, recompute sheaf.
    fn background_verify_all_provenance(
        storage: &Arc<Mutex<StorageManager>>,
        project_path: &str,
        storage_mode: &StorageMode,
    ) {
        debug!("Background provenance verification started");
        let config = Config::global();
        let stale_threshold = config.search.anchor_stale_threshold;

        // Load thread then DROP the lock — don't hold it during verification.
        // Re-acquire only for the final save.
        let mut thread = {
            let storage_guard = match storage.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            match storage_guard.load_thread_with_path_fix(project_path, storage_mode) {
                Ok(Some(t)) => t,
                _ => return,
            }
        }; // lock released here

        let now = Utc::now();
        let mut changed = false;

        // Step 1: Migrate old anchors → provenance (lazy)
        let mut migrated = 0;
        for fact in &mut thread.facts {
            if fact.provenance.dependencies.is_empty() && !fact.anchors.is_empty() {
                fact.provenance.dependencies =
                    provenance::migrate_anchors_to_provenance(&fact.anchors);
                fact.provenance.recompute_score();
                migrated += 1;
            }
        }
        if migrated > 0 {
            debug!("Migrated anchors to provenance for {} facts", migrated);
            changed = true;
        }

        // Step 2: Backfill provenance for facts with neither
        let scan = ProjectScan::new(project_path);
        let mut backfilled = 0;
        for fact in &mut thread.facts {
            if fact.provenance.dependencies.is_empty() {
                fact.provenance = provenance::generate_provenance(&fact.content, &scan);
                fact.anchors = scan.generate_anchors(&fact.content);
                backfilled += 1;
            }
        }
        if backfilled > 0 {
            debug!("Backfilled provenance for {} facts", backfilled);
            changed = true;
        }

        // Step 3: Verify provenance (budget-limited if enabled)
        let budget = if config.verification.budget_enabled {
            Some(crate::verification_budget::VerificationBudget::allocate(
                &thread.facts,
                &thread.change_tracker,
                config.verification.consecutive_threshold,
                config.verification.ewma_threshold,
                config.verification.verification_budget,
            ))
        } else {
            None
        };

        let mut verified_count = 0u32;
        let mut stale_count = 0u32;

        for fact in &mut thread.facts {
            if fact.provenance.dependencies.is_empty() {
                continue;
            }
            if let Some(ref budget) = budget {
                if !budget.should_verify(&fact.id) {
                    continue;
                }
            }

            let old_score = fact.provenance.score;
            provenance::verify_provenance(
                project_path,
                &mut fact.provenance,
            );
            verified_count += 1;

            let new_score = fact.provenance.score;

            if new_score != old_score {
                if new_score < stale_threshold {
                    fact.confidence *= 0.5;
                    stale_count += 1;
                } else if fact.provenance.has_broken_deps() {
                    fact.confidence *= new_score;
                    stale_count += 1;
                }
                fact.confidence = fact.confidence.max(0.05);
                fact.updated_at = now;
                changed = true;
            }
        }

        // Step 4: Sheaf cohomology recomputation
        if config.verification.sheaf_enabled {
            thread.cohomology = crate::sheaf::compute_and_store_cohomology(
                &thread.facts,
                config.verification.sheaf_edge_threshold,
                config.verification.sheaf_min_facts,
            );
            changed = true;
        }

        if changed || verified_count > 0 {
            thread.updated_at = now;
            // Re-acquire lock only for save
            if let Ok(storage_guard) = storage.lock() {
                if let Err(e) = storage_guard.save_thread(&thread, storage_mode) {
                    debug!("Failed to save background provenance verification: {}", e);
                }
            }
            debug!(
                "Provenance verification done: {} verified, {} stale, {} migrated, {} backfilled",
                verified_count, stale_count, migrated, backfilled
            );
        } else {
            debug!(
                "Provenance verification done: no changes needed"
            );
        }
    }

    // ================================================================
    // search_memory
    // ================================================================

    pub fn handle_search_memory(&self, args: Value) -> Result<Value> {
        let project_path = args["project_path"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| std::env::var("THREADBRIDGE_PROJECT_PATH").ok());
        let query = args["query"]
            .as_str()
            .context("query is required")?;
        let limit = args["limit"].as_u64().unwrap_or(5) as usize;

        let storage_mode = StorageMode::from_env();
        let storage = self.storage.lock().unwrap_or_else(|e| {
            tracing::warn!("Storage mutex was poisoned, recovering");
            e.into_inner()
        });
        let config = Config::global();

        let query_embedding = EmbeddingService::embed_query(query)
            .context("Failed to generate query embedding")?;

        if let Some(ref path) = project_path {
            match storage.load_thread_with_mode(path, &storage_mode)? {
                Some(mut thread) => {
                    // Filter out superseded/retracted facts
                    let active_facts: Vec<Fact> = thread.facts.iter()
                        .filter(|f| f.revision_status == RevisionStatus::Active)
                        .cloned()
                        .collect();
                    let vector_store = VectorStore::from_facts(active_facts);

                    // Use BLL reranking if available
                    let bll_guard = get_bll().lock().unwrap_or_else(|e| {
                    tracing::warn!("BLL mutex was poisoned, recovering");
                    e.into_inner()
                });
                    let results = vector_store.search_hybrid_bll(
                        &query_embedding, query, limit, config.search.min_similarity,
                        bll_guard.as_ref(),
                    );
                    drop(bll_guard);

                    // Update access_count and last_used_at
                    if !results.is_empty() {
                        let now = Utc::now();
                        let returned_ids: HashSet<&str> = results.iter()
                            .map(|r| r.fact.id.as_str())
                            .collect();
                        for fact in &mut thread.facts {
                            if returned_ids.contains(fact.id.as_str()) {
                                fact.access_count += 1;
                                fact.last_used_at = Some(now);
                            }
                        }
                        thread.updated_at = now;
                        if let Err(e) = storage.save_thread(&thread, &storage_mode) {
                            debug!("Failed to save access tracking: {}", e);
                        }
                    }

                    // Record search for implicit BLL feedback
                    if let Ok(mut buf) = search_buffer().lock() {
                        buf.push(SearchRecord {
                            query_embedding: query_embedding.clone(),
                            returned_fact_ids: results.iter().map(|r| r.fact.id.clone()).collect(),
                        });
                    }

                    // Build session summary lookup
                    let session_map: std::collections::HashMap<&str, &str> = thread.sessions.iter()
                        .map(|s| (s.id.as_str(), s.summary.as_str()))
                        .collect();

                    info!("Search found {} results for query: {}", results.len(), query);

                    let stale_threshold = config.search.anchor_stale_threshold;

                    Ok(json!({
                        "success": true,
                        "query": query,
                        "project_path": path,
                        "results": results.iter().map(|r| {
                            let session_summary = r.fact.session_id.as_deref()
                                .and_then(|sid| session_map.get(sid).copied());

                            // Enhanced verification markers
                            let (content, confidence, fact_warnings) = {
                                let mut markers: Vec<&str> = Vec::new();
                                let mut conf = r.fact.confidence;
                                let mut fact_warnings: Vec<String> = Vec::new();

                                // Provenance verification (preferred)
                                if config.verification.provenance_enabled
                                    && !r.fact.provenance.dependencies.is_empty()
                                {
                                    let mut prov = r.fact.provenance.clone();
                                    provenance::verify_provenance(path, &mut prov);
                                    let score = prov.compute_score();
                                    if score < stale_threshold {
                                        markers.push("[STALE]");
                                        conf *= score.max(0.1);
                                    } else if prov.has_broken_deps() {
                                        markers.push("[STALE]");
                                        conf *= score;
                                    }
                                }

                                // Contradiction candidate warnings
                                if config.verification.sheaf_enabled {
                                    let candidates = thread.cohomology.candidates_involving(&r.fact.id);
                                    for candidate in &candidates {
                                        let (other_excerpt, other_date) = if candidate.fact_id_a == r.fact.id {
                                            (&candidate.excerpt_b, candidate.created_at_b)
                                        } else {
                                            (&candidate.excerpt_a, candidate.created_at_a)
                                        };
                                        fact_warnings.push(format!(
                                            "Potential contradiction: this fact vs '{}' ({}). These facts discuss the same topic but differ — verify which is current.",
                                            other_excerpt,
                                            other_date.format("%Y-%m")
                                        ));
                                    }
                                    if !candidates.is_empty() {
                                        markers.push("[INCONSISTENT]");
                                    }
                                }

                                // CUSUM alarm
                                if config.verification.cusum_enabled {
                                    let alarm = thread.change_tracker.fact_alarm_score(
                                        &r.fact,
                                        config.verification.consecutive_threshold,
                                        config.verification.ewma_threshold,
                                    );
                                    if alarm > 0.5 {
                                        markers.push("[ACTIVELY CHANGING]");
                                    }
                                }

                                let content = if markers.is_empty() {
                                    r.fact.content.clone()
                                } else {
                                    format!("{} {}", markers.join(" "), r.fact.content)
                                };
                                (content, conf, fact_warnings)
                            };

                            json!({
                                "fact_id": r.fact.id,
                                "content": content,
                                "category": r.fact.category,
                                "score": r.score,
                                "confidence": confidence,
                                "context": r.fact.context_window,
                                "session_summary": session_summary,
                                "utility_mean": r.fact.utility.mean(),
                                "access_count": r.fact.access_count,
                                "created_at": r.fact.created_at,
                                "warnings": fact_warnings
                            })
                        }).collect::<Vec<_>>()
                    }))
                }
                None => {
                    Ok(json!({
                        "success": false,
                        "message": format!("No saved thread found for project: {}", path)
                    }))
                }
            }
        } else {
            // Cross-project search
            let (valid_projects, invalid_projects) = storage.get_searchable_projects()?;

            let mut all_results = Vec::new();

            for project in &valid_projects {
                if let Some(thread) = storage.load_thread(project)? {
                    let active_facts: Vec<Fact> = thread.facts.iter()
                        .filter(|f| f.revision_status == RevisionStatus::Active)
                        .cloned()
                        .collect();
                    let vector_store = VectorStore::from_facts(active_facts);

                    let bll_guard = get_bll().lock().unwrap_or_else(|e| {
                    tracing::warn!("BLL mutex was poisoned, recovering");
                    e.into_inner()
                });
                    let results = vector_store.search_hybrid_bll(
                        &query_embedding, query, limit, config.search.min_similarity,
                        bll_guard.as_ref(),
                    );
                    drop(bll_guard);

                    let session_map: std::collections::HashMap<&str, &str> = thread.sessions.iter()
                        .map(|s| (s.id.as_str(), s.summary.as_str()))
                        .collect();

                    for r in results {
                        let session_summary = r.fact.session_id.as_deref()
                            .and_then(|sid| session_map.get(sid).copied());
                        all_results.push(json!({
                            "content": r.fact.content,
                            "category": r.fact.category,
                            "score": r.score,
                            "context": r.fact.context_window,
                            "session_summary": session_summary,
                            "project_path": project,
                            "created_at": r.fact.created_at
                        }));
                    }
                }
            }

            all_results.sort_by(|a, b| {
                let score_a = a["score"].as_f64().unwrap_or(0.0);
                let score_b = b["score"].as_f64().unwrap_or(0.0);
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            all_results.truncate(limit);

            info!("Cross-project search found {} results for query: {}", all_results.len(), query);

            let mut response = json!({
                "success": true,
                "query": query,
                "searched_projects": valid_projects.len(),
                "results": all_results
            });

            if !invalid_projects.is_empty() {
                warn!("Some projects have invalid paths: {:?}", invalid_projects);
                response["invalid_projects"] = json!(invalid_projects);
            }

            Ok(response)
        }
    }

    // ================================================================
    // Implicit feedback: BLL weight updates + per-fact utility observations
    // ================================================================

    fn process_implicit_feedback(
        storage: &Arc<Mutex<StorageManager>>,
        project_path: &str,
        storage_mode: &StorageMode,
    ) {
        let records = {
            let mut buf = match search_buffer().lock() {
                Ok(buf) => buf,
                Err(_) => return,
            };
            std::mem::take(&mut *buf)
        };

        if records.is_empty() {
            return;
        }

        // Load the latest thread from disk (not a stale snapshot)
        let thread = match storage.lock() {
            Ok(guard) => match guard.load_thread_with_path_fix(project_path, storage_mode) {
                Ok(Some(t)) => t,
                _ => return,
            },
            Err(_) => return,
        };

        let fact_map: std::collections::HashMap<&str, &[f32]> = thread.facts.iter()
            .filter_map(|f| f.embedding.as_ref().map(|e| (f.id.as_str(), e.as_slice())))
            .collect();

        let recent_facts: Vec<&Fact> = thread.facts.iter()
            .rev()
            .take(50)
            .collect();

        // Compute rewards for all (fact_id, record) pairs
        struct FeedbackEntry {
            fact_id: String,
            query_embedding: Vec<f32>,
            fact_embedding: Vec<f32>,
            reward: f32,
        }
        let mut entries: Vec<FeedbackEntry> = Vec::new();

        for record in &records {
            for fact_id in &record.returned_fact_ids {
                let fact_emb = match fact_map.get(fact_id.as_str()) {
                    Some(emb) => *emb,
                    None => continue,
                };

                let mut max_sim = 0.0f32;
                for recent in &recent_facts {
                    if let Some(ref recent_emb) = recent.embedding {
                        let sim = embedding::cosine_similarity(fact_emb, recent_emb);
                        if sim > max_sim {
                            max_sim = sim;
                        }
                    }
                }

                let reward = if max_sim > 0.55 {
                    max_sim
                } else if max_sim < 0.3 {
                    0.0
                } else {
                    continue; // ambiguous — skip
                };

                entries.push(FeedbackEntry {
                    fact_id: fact_id.clone(),
                    query_embedding: record.query_embedding.clone(),
                    fact_embedding: fact_emb.to_vec(),
                    reward,
                });
            }
        }

        if entries.is_empty() {
            return;
        }

        // BLL weight updates (if available)
        let mut bll_updates = 0u32;
        if let Ok(mut bll_guard) = get_bll().lock() {
            if let Some(ref mut bll) = *bll_guard {
                for entry in &entries {
                    if entry.query_embedding.len() != entry.fact_embedding.len() {
                        continue;
                    }
                    let features = bll.extract_features(&entry.query_embedding, &entry.fact_embedding);
                    bll.update(&features, entry.reward);
                    bll_updates += 1;
                }

                if bll_updates > 0 {
                    let posterior_path = dirs::home_dir()
                        .unwrap_or_else(|| PathBuf::from("."))
                        .join(".threadbridge")
                        .join("bll_posterior.bin");
                    if let Err(e) = bll.save_posterior(&posterior_path) {
                        debug!("Failed to save BLL posterior: {}", e);
                    }
                    debug!("BLL implicit feedback: {} updates from {} search records",
                           bll_updates, records.len());
                }
            }
        }

        // Per-fact utility updates (Beta-Bernoulli observations)
        if let Ok(storage_guard) = storage.lock() {
            if let Ok(Some(mut stored_thread)) = storage_guard.load_thread_with_path_fix(project_path, storage_mode) {
                let mut obs_map: std::collections::HashMap<&str, Vec<f32>> =
                    std::collections::HashMap::new();
                for entry in &entries {
                    obs_map.entry(entry.fact_id.as_str()).or_default().push(entry.reward);
                }
                let mut utility_updated = 0u32;
                for fact in &mut stored_thread.facts {
                    if let Some(rewards) = obs_map.get(fact.id.as_str()) {
                        for &r in rewards {
                            fact.utility.observe(r);
                        }
                        utility_updated += 1;
                    }
                }
                if utility_updated > 0 {
                    stored_thread.updated_at = Utc::now();
                    if let Err(e) = storage_guard.save_thread(&stored_thread, storage_mode) {
                        debug!("Failed to save utility updates: {}", e);
                    }
                    debug!("Utility feedback: {} facts updated", utility_updated);
                }
            }
        }
    }

    /// Route a tool call to the appropriate handler
    pub fn handle(&self, tool_name: &str, args: Value) -> Result<Value> {
        match tool_name {
            "save_thread" => self.handle_save_thread(args),
            "load_thread" => self.handle_load_thread(args),
            "search_memory" => self.handle_search_memory(args),
            _ => anyhow::bail!("Unknown tool: {}", tool_name),
        }
    }
}
