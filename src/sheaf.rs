//! Contradiction candidate detection via simplicial complex analysis.
//!
//! Individual fact verification misses inter-fact contradictions. We organize
//! facts as a simplicial complex (related facts = simplex edges based on
//! embedding similarity) and detect pairs of Active facts that discuss the
//! same topic but say different things — contradiction candidates.
//!
//! These candidates are surfaced as natural language warnings in search_memory
//! responses, letting the host LLM judge whether they are true contradictions.
//!
//! Key papers:
//! - Huntsman et al. 2024: sheaf + LLM for inconsistency detection
//! - SuperLocalMemory V3 (Bhardwaj 2026): cellular sheaf on agent memory, +12.7pp on LoCoMo

use crate::embedding::cosine_similarity;
use crate::types::{Fact, RevisionStatus};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::debug;

/// A simplex in the simplicial complex (a pair of related facts = 1-simplex / edge)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Simplex {
    /// Fact IDs forming this simplex (always 2 for edges)
    pub fact_ids: Vec<String>,
    /// Cosine similarity between the facts
    pub similarity: f32,
}

/// Consistency check result for an edge (legacy, kept for backward compat deserialization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConsistency {
    pub fact_id_a: String,
    pub fact_id_b: String,
    pub consistency: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A pair of Active facts that may contradict each other
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionCandidate {
    pub fact_id_a: String,
    pub fact_id_b: String,
    /// Cosine similarity between the two facts' embeddings
    pub similarity: f32,
    /// Content excerpt from fact A
    pub excerpt_a: String,
    /// Content excerpt from fact B
    pub excerpt_b: String,
    /// Timestamp of fact A
    pub created_at_a: DateTime<Utc>,
    /// Timestamp of fact B
    pub created_at_b: DateTime<Utc>,
}

/// Result of consistency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohomologyResult {
    /// Edges in the simplicial complex
    #[serde(default)]
    pub edges: Vec<(String, String)>,
    /// Legacy field, kept for backward compat with old thread.json
    #[serde(default)]
    pub edge_consistencies: Vec<EdgeConsistency>,
    /// Legacy field, kept for backward compat with old thread.json
    #[serde(default)]
    pub h1_cycles: Vec<Vec<String>>,
    /// Detected contradiction candidates
    #[serde(default)]
    pub contradiction_candidates: Vec<ContradictionCandidate>,
    /// Overall consistency score: 1.0 if no candidates, decreasing with more
    #[serde(default = "default_global_consistency")]
    pub global_consistency: f32,
    /// When this was last computed
    #[serde(default = "default_cohomology_time")]
    pub computed_at: DateTime<Utc>,
}

fn default_global_consistency() -> f32 {
    1.0
}

fn default_cohomology_time() -> DateTime<Utc> {
    Utc::now()
}

impl Default for CohomologyResult {
    fn default() -> Self {
        Self {
            edges: Vec::new(),
            edge_consistencies: Vec::new(),
            h1_cycles: Vec::new(),
            contradiction_candidates: Vec::new(),
            global_consistency: 1.0,
            computed_at: Utc::now(),
        }
    }
}

impl CohomologyResult {
    /// Legacy: check if a fact is part of any H¹ inconsistency cycle.
    /// Always returns false for new results (h1_cycles is empty).
    pub fn is_in_h1_cycle(&self, fact_id: &str) -> bool {
        self.h1_cycles
            .iter()
            .any(|cycle| cycle.iter().any(|id| id == fact_id))
    }

    /// Return contradiction candidates involving a specific fact
    pub fn candidates_involving(&self, fact_id: &str) -> Vec<&ContradictionCandidate> {
        self.contradiction_candidates
            .iter()
            .filter(|c| c.fact_id_a == fact_id || c.fact_id_b == fact_id)
            .collect()
    }
}

// ================================================================
// Simplicial complex construction
// ================================================================

/// Build the 1-skeleton of the simplicial complex from facts based on embedding similarity.
/// Edges connect facts whose cosine similarity exceeds the threshold.
pub fn build_simplicial_complex(facts: &[Fact], edge_threshold: f32) -> Vec<Simplex> {
    let active_facts: Vec<&Fact> = facts
        .iter()
        .filter(|f| f.revision_status == RevisionStatus::Active && f.embedding.is_some())
        .collect();

    let mut edges = Vec::new();

    for i in 0..active_facts.len() {
        for j in (i + 1)..active_facts.len() {
            let emb_i = active_facts[i].embedding.as_ref().unwrap();
            let emb_j = active_facts[j].embedding.as_ref().unwrap();
            let sim = cosine_similarity(emb_i, emb_j);

            if sim >= edge_threshold {
                edges.push(Simplex {
                    fact_ids: vec![active_facts[i].id.clone(), active_facts[j].id.clone()],
                    similarity: sim,
                });
            }
        }
    }

    debug!(
        "Sheaf: built simplicial complex with {} edges from {} active facts",
        edges.len(),
        active_facts.len()
    );
    edges
}

// ================================================================
// Contradiction candidate detection
// ================================================================

/// Truncate a string to at most `max_len` characters, appending "..." if truncated.
fn excerpt(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    match s.char_indices().nth(max_len) {
        Some((byte_idx, _)) => format!("{}...", &s[..byte_idx]),
        None => s.to_string(),
    }
}

/// Detect pairs of Active facts that may contradict each other.
///
/// A contradiction candidate is a pair where:
/// - Both facts are Active
/// - Embedding cosine similarity > edge_threshold (same topic — already guaranteed by edges)
/// - Content differs (not exact duplicates)
/// - Not linked by supersession (neither supersedes the other)
pub fn detect_contradiction_candidates(
    facts: &[Fact],
    edges: &[Simplex],
) -> Vec<ContradictionCandidate> {
    // Build set of supersession-linked pairs (unordered)
    let mut supersession_pairs: HashSet<(String, String)> = HashSet::new();
    for fact in facts {
        for sid in &fact.supersedes {
            let pair = ordered_pair(&fact.id, sid);
            supersession_pairs.insert(pair);
        }
        for sid in &fact.superseded_by {
            let pair = ordered_pair(&fact.id, sid);
            supersession_pairs.insert(pair);
        }
    }

    // Build fact lookup
    let fact_map: std::collections::HashMap<&str, &Fact> =
        facts.iter().map(|f| (f.id.as_str(), f)).collect();

    let mut candidates: Vec<ContradictionCandidate> = Vec::new();

    for edge in edges {
        let id_a = &edge.fact_ids[0];
        let id_b = &edge.fact_ids[1];

        let fa = match fact_map.get(id_a.as_str()) {
            Some(f) => f,
            None => continue,
        };
        let fb = match fact_map.get(id_b.as_str()) {
            Some(f) => f,
            None => continue,
        };

        // Both must be Active (build_simplicial_complex filters for this, but double-check)
        if fa.revision_status != RevisionStatus::Active
            || fb.revision_status != RevisionStatus::Active
        {
            continue;
        }

        // Skip exact duplicates
        if fa.content == fb.content {
            continue;
        }

        // Skip supersession-linked pairs
        let pair = ordered_pair(id_a, id_b);
        if supersession_pairs.contains(&pair) {
            continue;
        }

        candidates.push(ContradictionCandidate {
            fact_id_a: id_a.clone(),
            fact_id_b: id_b.clone(),
            similarity: edge.similarity,
            excerpt_a: excerpt(&fa.content, 80),
            excerpt_b: excerpt(&fb.content, 80),
            created_at_a: fa.created_at,
            created_at_b: fb.created_at,
        });
    }

    // Sort by similarity descending (most likely contradictions first)
    candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(20);

    debug!(
        "Sheaf: detected {} contradiction candidates from {} edges",
        candidates.len(),
        edges.len()
    );

    candidates
}

/// Create an ordered pair for deduplication (smaller ID first)
fn ordered_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// ================================================================
// Entry point
// ================================================================

/// Compute contradiction candidates and store in CohomologyResult.
pub fn compute_and_store_cohomology(
    facts: &[Fact],
    edge_threshold: f32,
    min_facts: usize,
) -> CohomologyResult {
    let active_count = facts
        .iter()
        .filter(|f| f.revision_status == RevisionStatus::Active && f.embedding.is_some())
        .count();

    if active_count < min_facts {
        debug!(
            "Sheaf: skipping analysis ({} active facts < {} minimum)",
            active_count, min_facts
        );
        return CohomologyResult::default();
    }

    let edges = build_simplicial_complex(facts, edge_threshold);
    if edges.is_empty() {
        return CohomologyResult::default();
    }

    let candidates = detect_contradiction_candidates(facts, &edges);

    let global_consistency = if candidates.is_empty() {
        1.0
    } else {
        (1.0 - candidates.len() as f32 / edges.len() as f32).clamp(0.0, 1.0)
    };

    CohomologyResult {
        edges: edges
            .iter()
            .map(|e| (e.fact_ids[0].clone(), e.fact_ids[1].clone()))
            .collect(),
        edge_consistencies: Vec::new(),
        h1_cycles: Vec::new(),
        contradiction_candidates: candidates,
        global_consistency,
        computed_at: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::Belief;
    use crate::provenance::Provenance;
    use crate::types::FactCategory;

    fn make_fact_with_content(
        id: &str,
        content: &str,
        emb: Vec<f32>,
        supersedes: Vec<String>,
        superseded_by: Vec<String>,
    ) -> Fact {
        Fact {
            id: id.to_string(),
            content: content.to_string(),
            category: FactCategory::General,
            confidence: 1.0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            embedding: Some(emb),
            utility: Belief::uninformed(),
            access_count: 0,
            last_used_at: None,
            context_window: Vec::new(),
            session_id: None,
            anchors: Vec::new(),
            provenance: Provenance {
                dependencies: Vec::new(),
                score: 1.0,
            },
            revision_status: RevisionStatus::Active,
            supersedes,
            superseded_by,
        }
    }

    fn make_fact(id: &str, emb: Vec<f32>) -> Fact {
        make_fact_with_content(id, &format!("fact {}", id), emb, Vec::new(), Vec::new())
    }

    #[test]
    fn test_simplicial_complex_construction() {
        let facts = vec![
            make_fact("a", vec![1.0, 0.0, 0.0]),
            make_fact("b", vec![0.95, 0.05, 0.0]), // similar to a
            make_fact("c", vec![0.0, 0.0, 1.0]),   // dissimilar
        ];

        let edges = build_simplicial_complex(&facts, 0.9);
        // a-b should be connected (sim ~0.999), c should not connect to either
        assert_eq!(edges.len(), 1);
        assert!(edges[0].fact_ids.contains(&"a".to_string()));
        assert!(edges[0].fact_ids.contains(&"b".to_string()));
    }

    #[test]
    fn test_contradiction_detected() {
        let facts = vec![
            make_fact_with_content("a", "We use SQLite for storage", vec![1.0, 0.0, 0.0], vec![], vec![]),
            make_fact_with_content("b", "Migrated to PostgreSQL for storage", vec![0.95, 0.05, 0.0], vec![], vec![]),
        ];

        let result = compute_and_store_cohomology(&facts, 0.5, 1);
        assert_eq!(result.contradiction_candidates.len(), 1);
        assert_eq!(result.contradiction_candidates[0].fact_id_a, "a");
        assert_eq!(result.contradiction_candidates[0].fact_id_b, "b");
        assert!(result.global_consistency < 1.0);
    }

    #[test]
    fn test_no_contradiction_when_superseded() {
        let facts = vec![
            make_fact_with_content(
                "a", "We use SQLite for storage",
                vec![1.0, 0.0, 0.0],
                vec![], vec!["b".to_string()],
            ),
            make_fact_with_content(
                "b", "Migrated to PostgreSQL for storage",
                vec![0.95, 0.05, 0.0],
                vec!["a".to_string()], vec![],
            ),
        ];

        let result = compute_and_store_cohomology(&facts, 0.5, 1);
        assert!(result.contradiction_candidates.is_empty());
        assert_eq!(result.global_consistency, 1.0);
    }

    #[test]
    fn test_no_contradiction_for_identical_content() {
        let facts = vec![
            make_fact_with_content("a", "same content", vec![1.0, 0.0, 0.0], vec![], vec![]),
            make_fact_with_content("b", "same content", vec![0.95, 0.05, 0.0], vec![], vec![]),
        ];

        let result = compute_and_store_cohomology(&facts, 0.5, 1);
        assert!(result.contradiction_candidates.is_empty());
    }

    #[test]
    fn test_no_contradiction_below_threshold() {
        let facts = vec![
            make_fact_with_content("a", "We use SQLite", vec![1.0, 0.0, 0.0], vec![], vec![]),
            make_fact_with_content("b", "Testing framework is Jest", vec![0.0, 0.0, 1.0], vec![], vec![]),
        ];

        // High threshold means no edge between dissimilar facts
        let result = compute_and_store_cohomology(&facts, 0.9, 1);
        assert!(result.contradiction_candidates.is_empty());
    }

    #[test]
    fn test_candidates_involving() {
        let facts = vec![
            make_fact_with_content("a", "We use SQLite", vec![1.0, 0.0, 0.0], vec![], vec![]),
            make_fact_with_content("b", "Migrated to PostgreSQL", vec![0.95, 0.05, 0.0], vec![], vec![]),
            make_fact_with_content("c", "Unrelated fact", vec![0.0, 0.0, 1.0], vec![], vec![]),
        ];

        let result = compute_and_store_cohomology(&facts, 0.5, 1);
        assert_eq!(result.candidates_involving("a").len(), 1);
        assert_eq!(result.candidates_involving("b").len(), 1);
        assert_eq!(result.candidates_involving("c").len(), 0);
    }

    #[test]
    fn test_empty_facts() {
        let result = compute_and_store_cohomology(&[], 0.6, 10);
        assert_eq!(result.global_consistency, 1.0);
        assert!(result.contradiction_candidates.is_empty());
    }

    #[test]
    fn test_below_min_facts() {
        let facts = vec![make_fact("a", vec![1.0, 0.0])];
        let result = compute_and_store_cohomology(&facts, 0.6, 5);
        assert_eq!(result.global_consistency, 1.0);
        assert!(result.contradiction_candidates.is_empty());
    }

    #[test]
    fn test_excerpt_truncation() {
        assert_eq!(excerpt("short", 80), "short");
        let long = "a".repeat(100);
        let ex = excerpt(&long, 80);
        assert!(ex.ends_with("..."));
        assert!(ex.len() < 100);
    }

    #[test]
    fn test_backward_compat_legacy_fields() {
        // Ensure legacy h1_cycles/edge_consistencies still work
        let result = CohomologyResult::default();
        assert!(!result.is_in_h1_cycle("any_id"));
        assert!(result.edge_consistencies.is_empty());
        assert!(result.h1_cycles.is_empty());
    }
}
