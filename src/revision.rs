//! AGM Belief Revision: versioned fact chains with Supersedes edges.
//!
//! Instead of deleting stale facts, we create revision chains. New facts
//! `Supersede` old facts, preserving history. This implements a simplified
//! AGM revision operator on the fact knowledge base.
//!
//! Key paper: Kumiho (Park 2026) — B(τ) = ∪ φ(τ(t)), revision creates
//! new node + Supersedes edge, old node becomes inactive.

use crate::embedding::cosine_similarity;
use crate::types::{Fact, RevisionRelation, RevisionStatus};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// A revision edge linking two facts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevisionEdge {
    /// The source fact ID (the newer fact)
    pub from_id: String,
    /// The target fact ID (the older fact)
    pub to_id: String,
    /// Relationship type
    pub relation: RevisionRelation,
    /// When this edge was created
    pub created_at: DateTime<Utc>,
}

/// The full revision graph for a thread
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RevisionGraph {
    #[serde(default)]
    pub edges: Vec<RevisionEdge>,
}

impl RevisionGraph {
    /// Add an edge to the graph
    pub fn add_edge(
        &mut self,
        from_id: String,
        to_id: String,
        relation: RevisionRelation,
        created_at: DateTime<Utc>,
    ) {
        self.edges.push(RevisionEdge {
            from_id,
            to_id,
            relation,
            created_at,
        });
    }

    /// Record that new_fact supersedes old_fact
    pub fn supersede(&mut self, new_fact_id: &str, old_fact_id: &str, now: DateTime<Utc>) {
        self.add_edge(
            new_fact_id.to_string(),
            old_fact_id.to_string(),
            RevisionRelation::Supersedes,
            now,
        );
    }

    /// Find the fact that superseded a given fact (if any)
    pub fn get_superseded_by(&self, fact_id: &str) -> Option<&str> {
        for edge in &self.edges {
            if edge.to_id == fact_id && edge.relation == RevisionRelation::Supersedes {
                return Some(&edge.from_id);
            }
        }
        None
    }

    /// Walk the revision chain backward from a fact through Supersedes edges.
    /// Returns [current, predecessor, predecessor_of_predecessor, ...]
    pub fn get_chain(&self, fact_id: &str) -> Vec<String> {
        let mut chain = vec![fact_id.to_string()];
        let mut visited = std::collections::HashSet::new();
        visited.insert(fact_id.to_string());
        let mut current = fact_id.to_string();

        loop {
            let mut found = false;
            for edge in &self.edges {
                if edge.from_id == current && edge.relation == RevisionRelation::Supersedes {
                    if !visited.insert(edge.to_id.clone()) {
                        return chain;
                    }
                    chain.push(edge.to_id.clone());
                    current = edge.to_id.clone();
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        chain
    }

    /// Count total edges
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Detect which new facts supersede existing active facts.
///
/// Uses embedding cosine similarity > threshold AND different content to identify
/// supersession pairs. High semantic similarity + different content = the new fact
/// is saying something updated about the same topic.
///
/// Returns Vec of (new_fact_id, old_fact_id) pairs.
pub fn detect_supersession(
    existing_facts: &[Fact],
    new_facts: &[Fact],
    threshold: f32,
) -> Vec<(String, String)> {
    let mut pairs = Vec::new();

    for new_fact in new_facts {
        let new_emb = match new_fact.embedding.as_ref() {
            Some(e) => e,
            None => continue,
        };

        let mut best_sim = 0.0f32;
        let mut best_old_id: Option<&str> = None;

        for old_fact in existing_facts {
            if old_fact.revision_status != RevisionStatus::Active {
                continue;
            }
            let old_emb = match old_fact.embedding.as_ref() {
                Some(e) => e,
                None => continue,
            };

            let sim = cosine_similarity(new_emb, old_emb);
            if sim > threshold && new_fact.content != old_fact.content && sim > best_sim {
                best_sim = sim;
                best_old_id = Some(&old_fact.id);
            }
        }

        if let Some(old_id) = best_old_id {
            debug!(
                "Supersession detected: new={} supersedes old={} (sim={:.3})",
                &new_fact.id[..8.min(new_fact.id.len())],
                &old_id[..8.min(old_id.len())],
                best_sim
            );
            pairs.push((new_fact.id.clone(), old_id.to_string()));
        }
    }

    pairs
}

/// Apply AGM revision: mark old fact as Superseded, link via revision graph.
pub fn apply_agm_revision(
    facts: &mut [Fact],
    revision_graph: &mut RevisionGraph,
    new_fact_id: &str,
    old_fact_id: &str,
    now: DateTime<Utc>,
) {
    for fact in facts.iter_mut() {
        if fact.id == old_fact_id {
            fact.revision_status = RevisionStatus::Superseded;
            if !fact.superseded_by.contains(&new_fact_id.to_string()) {
                fact.superseded_by.push(new_fact_id.to_string());
            }
            fact.updated_at = now;
        }
        if fact.id == new_fact_id {
            if !fact.supersedes.contains(&old_fact_id.to_string()) {
                fact.supersedes.push(old_fact_id.to_string());
            }
        }
    }
    revision_graph.supersede(new_fact_id, old_fact_id, now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revision_graph_basic() {
        let mut graph = RevisionGraph::default();
        let now = Utc::now();

        graph.supersede("new1", "old1", now);
        assert_eq!(graph.len(), 1);
        assert_eq!(graph.get_superseded_by("old1"), Some("new1"));
        assert_eq!(graph.get_superseded_by("new1"), None);
    }

    #[test]
    fn test_revision_chain() {
        let mut graph = RevisionGraph::default();
        let now = Utc::now();

        // v3 supersedes v2, v2 supersedes v1
        graph.supersede("v2", "v1", now);
        graph.supersede("v3", "v2", now);

        let chain = graph.get_chain("v3");
        assert_eq!(chain, vec!["v3", "v2", "v1"]);
    }

    #[test]
    fn test_revision_chain_cycle() {
        let mut graph = RevisionGraph::default();
        let now = Utc::now();

        // A → B → A forms a cycle
        graph.supersede("a", "b", now);
        graph.supersede("b", "a", now);

        let chain = graph.get_chain("a");
        assert_eq!(chain, vec!["a", "b"]);
    }

    #[test]
    fn test_detect_supersession() {
        use crate::belief::Belief;
        use crate::types::FactCategory;

        let make_fact = |id: &str, content: &str, emb: Vec<f32>| Fact {
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
            provenance: Default::default(),
            revision_status: RevisionStatus::Active,
            supersedes: Vec::new(),
            superseded_by: Vec::new(),
        };

        let existing = vec![make_fact("old1", "uses Redis for caching", vec![1.0, 0.0, 0.0])];
        let new_facts = vec![
            make_fact(
                "new1",
                "switched from Redis to Memcached for caching",
                vec![0.95, 0.05, 0.0],
            ),
            make_fact("new2", "unrelated fact about testing", vec![0.0, 0.0, 1.0]),
        ];

        let pairs = detect_supersession(&existing, &new_facts, 0.85);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], ("new1".to_string(), "old1".to_string()));
    }

    #[test]
    fn test_multiple_new_facts_supersede_same_old() {
        use crate::belief::Belief;
        use crate::types::FactCategory;

        let make_fact = |id: &str, content: &str, emb: Vec<f32>| Fact {
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
            provenance: Default::default(),
            revision_status: RevisionStatus::Active,
            supersedes: Vec::new(),
            superseded_by: Vec::new(),
        };

        // old fact covers two concerns; two new facts each supersede it
        let existing = vec![make_fact(
            "old1",
            "uses Redis for caching and session storage",
            vec![1.0, 0.0, 0.0],
        )];
        let new_facts = vec![
            make_fact("new1", "switched to Memcached for caching", vec![0.95, 0.05, 0.0]),
            make_fact("new2", "session storage moved to JWT", vec![0.90, 0.10, 0.0]),
        ];

        let pairs = detect_supersession(&existing, &new_facts, 0.85);
        // Both new facts should match old1
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&("new1".to_string(), "old1".to_string())));
        assert!(pairs.contains(&("new2".to_string(), "old1".to_string())));

        // Apply both revisions — superseded_by should contain both
        let mut all_facts = existing;
        all_facts.extend(new_facts);
        let mut graph = RevisionGraph::default();
        let now = Utc::now();

        for (new_id, old_id) in &pairs {
            apply_agm_revision(&mut all_facts, &mut graph, new_id, old_id, now);
        }

        let old_fact = all_facts.iter().find(|f| f.id == "old1").unwrap();
        assert_eq!(old_fact.revision_status, RevisionStatus::Superseded);
        assert_eq!(old_fact.superseded_by.len(), 2);
        assert!(old_fact.superseded_by.contains(&"new1".to_string()));
        assert!(old_fact.superseded_by.contains(&"new2".to_string()));

        // Graph should have two edges
        assert_eq!(graph.len(), 2);
    }
}
