//! MCP Tool definitions and handlers
//!
//! Three tools: save_thread, load_thread, search_memory
//! Core pipeline: Arctic Embed M + Dense/BM25 Hybrid + RRF + BLL v2

use crate::bll::BayesianLastLayer;
use crate::config::Config;
use crate::embedding::{self, EmbeddingService};
use crate::mcp::Tool;
use crate::storage::StorageManager;
use crate::types::{Fact, FactCategory, ProjectState, Session, Thread};
use crate::vector_store::VectorStore;
use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
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
                    .unwrap_or_default()
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
// ToolHandler
// ================================================================

#[derive(Clone)]
pub struct ToolHandler {
    storage: Arc<Mutex<StorageManager>>,
}

impl ToolHandler {
    pub fn new(storage: StorageManager) -> Self {
        Self {
            storage: Arc::new(Mutex::new(storage)),
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
                        "use_local_storage": {
                            "type": "boolean",
                            "description": "If true, store in project's .threadbridge/ directory.",
                            "default": true
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
                    "required": ["project_path", "summary"]
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
                    "required": ["project_path"]
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
            .context("project_path is required")?
            .to_string();

        let use_local = args["use_local_storage"].as_bool().unwrap_or(true);
        let storage = self.storage.lock().unwrap_or_else(|e| e.into_inner());

        // Load existing thread or create new one
        let mut thread = storage
            .load_thread(&project_path)?
            .unwrap_or_else(|| Thread::new(project_path.clone()));

        // Update architecture if provided
        if let Some(arch) = args["architecture"].as_str() {
            thread.architecture = Some(arch.to_string());
        }

        // Update state if provided
        if let Some(state) = args.get("state") {
            thread.state = ProjectState {
                completed: state["completed"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
                in_progress: state["in_progress"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
                pending: state["pending"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
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
                confidence: 1.0,
                created_at: now,
                updated_at: now,
                embedding: get_emb(decision_offset + i),
                utility: crate::belief::Belief::uninformed(),
                access_count: 0,
                last_used_at: None,
                context_window: Vec::new(), // populated below
                session_id: Some(session_id.clone()),
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
                confidence: 1.0,
                created_at: now,
                updated_at: now,
                embedding: get_emb(general_offset + i),
                utility: crate::belief::Belief::uninformed(),
                access_count: 0,
                last_used_at: None,
                context_window: Vec::new(), // populated below
                session_id: Some(session_id.clone()),
            };
            thread.facts.push(fact);
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
        // Create Session
        // ================================================================
        let session = Session {
            id: session_id,
            summary: summary_text.clone(),
            fact_ids: session_fact_ids,
            created_at: now,
        };
        thread.sessions.push(session);

        thread.updated_at = now;

        // Save thread
        storage.save_thread(&thread, use_local)?;

        let storage_location = if use_local { "local (.threadbridge/)" } else { "global (~/.threadbridge/)" };
        info!("Saved thread: {} ({} facts, {} sessions) to {}",
            project_path, thread.facts.len(), thread.sessions.len(), storage_location);

        // Build response before spawning background work
        let response = json!({
            "success": true,
            "project_path": project_path,
            "storage_location": storage_location,
            "facts_saved": session_len,
            "total_facts": thread.facts.len(),
            "total_sessions": thread.sessions.len()
        });

        // Background: BLL implicit feedback
        drop(storage);
        let thread_clone = thread.clone();
        std::thread::spawn(move || {
            Self::process_bll_implicit_feedback(&thread_clone);
        });

        Ok(response)
    }

    // ================================================================
    // load_thread
    // ================================================================

    pub fn handle_load_thread(&self, args: Value) -> Result<Value> {
        let project_path = args["project_path"]
            .as_str()
            .context("project_path is required")?;

        let storage = self.storage.lock().unwrap_or_else(|e| e.into_inner());

        match storage.load_thread_with_path_fix(project_path)? {
            Some(thread) => {
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

                // Recent decisions (last 10)
                let decisions: Vec<String> = thread.facts.iter()
                    .filter(|f| f.category == FactCategory::Decision)
                    .rev()
                    .take(10)
                    .map(|f| f.content.clone())
                    .collect();

                info!("Loaded thread: {} ({} facts, {} sessions)",
                    project_path, thread.facts.len(), thread.sessions.len());

                Ok(json!({
                    "success": true,
                    "project_path": project_path,
                    "thread": {
                        "architecture": thread.architecture,
                        "state": {
                            "completed": thread.state.completed,
                            "in_progress": thread.state.in_progress,
                            "pending": thread.state.pending
                        },
                        "recent_sessions": recent_sessions,
                        "recent_decisions": decisions,
                        "total_facts": thread.facts.len(),
                        "total_sessions": thread.sessions.len(),
                        "created_at": thread.created_at,
                        "updated_at": thread.updated_at
                    }
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

    // ================================================================
    // search_memory
    // ================================================================

    pub fn handle_search_memory(&self, args: Value) -> Result<Value> {
        let project_path = args["project_path"].as_str();
        let query = args["query"]
            .as_str()
            .context("query is required")?;
        let limit = args["limit"].as_u64().unwrap_or(5) as usize;

        let storage = self.storage.lock().unwrap_or_else(|e| e.into_inner());
        let config = Config::global();

        let query_embedding = EmbeddingService::embed_query(query)
            .context("Failed to generate query embedding")?;

        if let Some(path) = project_path {
            match storage.load_thread(path)? {
                Some(mut thread) => {
                    let vector_store = VectorStore::from_facts(thread.facts.clone());

                    // Use BLL reranking if available
                    let bll_guard = get_bll().lock().unwrap_or_else(|e| e.into_inner());
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
                        if let Err(e) = storage.save_thread(&thread, true) {
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

                    Ok(json!({
                        "success": true,
                        "query": query,
                        "project_path": path,
                        "results": results.iter().map(|r| {
                            let session_summary = r.fact.session_id.as_deref()
                                .and_then(|sid| session_map.get(sid).copied());
                            json!({
                                "fact_id": r.fact.id,
                                "content": r.fact.content,
                                "category": r.fact.category,
                                "score": r.score,
                                "context": r.fact.context_window,
                                "session_summary": session_summary,
                                "utility_mean": r.fact.utility.mean(),
                                "access_count": r.fact.access_count,
                                "created_at": r.fact.created_at
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
                    let vector_store = VectorStore::from_facts(thread.facts.clone());

                    let bll_guard = get_bll().lock().unwrap_or_else(|e| e.into_inner());
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
    // BLL implicit feedback
    // ================================================================

    fn process_bll_implicit_feedback(thread: &Thread) {
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

        let mut bll_guard = match get_bll().lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let bll = match bll_guard.as_mut() {
            Some(bll) => bll,
            None => return,
        };

        let fact_map: std::collections::HashMap<&str, &[f32]> = thread.facts.iter()
            .filter_map(|f| f.embedding.as_ref().map(|e| (f.id.as_str(), e.as_slice())))
            .collect();

        let recent_facts: Vec<&Fact> = thread.facts.iter()
            .rev()
            .take(50)
            .collect();

        let mut updates = 0u32;

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
                    continue; // ambiguous
                };

                if record.query_embedding.len() != fact_emb.len() {
                    continue;
                }

                let features = bll.extract_features(&record.query_embedding, fact_emb);
                bll.update(&features, reward);
                updates += 1;
            }
        }

        if updates > 0 {
            let posterior_path = dirs::home_dir()
                .unwrap_or_default()
                .join(".threadbridge")
                .join("bll_posterior.bin");
            if let Err(e) = bll.save_posterior(&posterior_path) {
                debug!("Failed to save BLL posterior: {}", e);
            }
            debug!("BLL implicit feedback: {} updates from {} search records",
                   updates, records.len());
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
