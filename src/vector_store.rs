//! Vector Store for semantic search
//!
//! Hybrid search: dense (cosine) + sparse (BM25), fused via Reciprocal Rank Fusion.
//! BM25 indexes fact content + context window for richer lexical matching.

use crate::bll::BayesianLastLayer;
use crate::config::Config;
use crate::embedding::cosine_similarity;
use crate::types::{Fact, SearchResult};
use bm25::{Language, SearchEngine, SearchEngineBuilder};
use std::collections::HashMap;
use tracing::debug;

/// RRF constant (standard value from the original RRF paper)
const RRF_K: f32 = 60.0;

pub struct VectorStore {
    facts: Vec<Fact>,
    bm25_engine: Option<SearchEngine<u32>>,
}

impl VectorStore {
    /// Build BM25 engine from facts with context-augmented text.
    /// Each fact's BM25 document = content + context_window joined.
    fn build_bm25(facts: &[Fact]) -> Option<SearchEngine<u32>> {
        if facts.is_empty() {
            return None;
        }
        let corpus: Vec<String> = facts.iter().map(|f| {
            if f.context_window.is_empty() {
                f.content.clone()
            } else {
                format!("{} {}", f.content, f.context_window.join(" "))
            }
        }).collect();
        Some(SearchEngineBuilder::<u32>::with_corpus(Language::English, corpus).build())
    }

    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            bm25_engine: None,
        }
    }

    pub fn from_facts(facts: Vec<Fact>) -> Self {
        let bm25_engine = Self::build_bm25(&facts);
        Self { facts, bm25_engine }
    }

    /// Hybrid search: Dense + BM25 + RRF + optional BLL reranking.
    pub fn search_hybrid_bll(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        limit: usize,
        threshold: f32,
        bll: Option<&BayesianLastLayer>,
    ) -> Vec<SearchResult> {
        let lambda = Config::global().search.utility_lambda;

        // === Dense retrieval ===
        let mut dense_ranked: Vec<(usize, f32)> = self.facts
            .iter()
            .enumerate()
            .filter_map(|(idx, fact)| {
                let fact_emb = fact.embedding.as_ref()?;
                let cosine = cosine_similarity(query_embedding, fact_emb);

                if cosine >= threshold {
                    let score = if lambda > 0.0 && !fact.utility.is_uninformed() {
                        let utility_sample = fact.utility.sample();
                        (1.0 - lambda) * cosine + lambda * utility_sample
                    } else {
                        cosine
                    };
                    Some((idx, score))
                } else {
                    None
                }
            })
            .collect();
        dense_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // === BM25 retrieval ===
        let bm25_ranked = if !query_text.is_empty() && !self.facts.is_empty() {
            self.bm25_search_facts(query_text, limit * 3)
        } else {
            Vec::new()
        };

        // === RRF fusion ===
        let mut rrf_scores: HashMap<usize, f32> = HashMap::new();
        for (rank, &(idx, _)) in dense_ranked.iter().enumerate() {
            *rrf_scores.entry(idx).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, &(idx, _)) in bm25_ranked.iter().enumerate() {
            *rrf_scores.entry(idx).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
        }

        let mut fused: Vec<(usize, f32)> = rrf_scores.into_iter().collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // === BLL reranking (optional) ===
        if let Some(bll) = bll {
            let bll_weight = 0.3;
            let rrf_max = fused.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
            let rrf_min = fused.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min);
            let rrf_range = (rrf_max - rrf_min).max(1e-8);

            let mut blended: Vec<(usize, f32)> = fused
                .iter()
                .filter_map(|&(idx, rrf_score)| {
                    let fact_emb = self.facts[idx].embedding.as_ref()?;
                    let rrf_norm = (rrf_score - rrf_min) / rrf_range;
                    if fact_emb.len() != query_embedding.len() {
                        return Some((idx, rrf_norm));
                    }
                    let (bll_mu, _bll_var) = bll.predict(query_embedding, fact_emb);
                    let bll_prob = 1.0 / (1.0 + (-bll_mu).exp());
                    let combined = (1.0 - bll_weight) * rrf_norm + bll_weight * bll_prob;
                    Some((idx, combined))
                })
                .collect();

            blended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            blended.truncate(limit);

            let results: Vec<SearchResult> = blended
                .into_iter()
                .map(|(idx, score)| SearchResult {
                    fact: self.facts[idx].clone(),
                    score,
                })
                .collect();

            debug!("Hybrid+BLL search: {} facts, {} results", self.facts.len(), results.len());
            return results;
        }

        // No BLL
        fused.truncate(limit);
        let results: Vec<SearchResult> = fused
            .into_iter()
            .map(|(idx, score)| SearchResult {
                fact: self.facts[idx].clone(),
                score,
            })
            .collect();

        debug!("Hybrid search: {} facts, {} results", self.facts.len(), results.len());
        results
    }

    fn bm25_search_facts(&self, query: &str, limit: usize) -> Vec<(usize, f32)> {
        match &self.bm25_engine {
            Some(engine) => {
                engine.search(query, limit)
                    .into_iter()
                    .map(|r| (r.document.id as usize, r.score))
                    .collect()
            }
            None => Vec::new(),
        }
    }

}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::Belief;
    use crate::types::FactCategory;
    use chrono::Utc;

    fn create_test_fact(id: &str, content: &str, embedding: Vec<f32>) -> Fact {
        Fact {
            id: id.to_string(),
            content: content.to_string(),
            category: FactCategory::General,
            confidence: 1.0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            embedding: Some(embedding),
            utility: Belief::uninformed(),
            access_count: 0,
            last_used_at: None,
            context_window: Vec::new(),
            session_id: None,
        }
    }

    #[test]
    fn test_from_facts_and_search() {
        let facts = vec![
            create_test_fact("1", "rust programming", vec![1.0, 0.0, 0.0]),
            create_test_fact("2", "python programming", vec![0.9, 0.1, 0.0]),
            create_test_fact("3", "cooking recipes", vec![0.0, 0.0, 1.0]),
        ];

        let store = VectorStore::from_facts(facts);
        let results = store.search_hybrid_bll(&[1.0, 0.0, 0.0], "", 2, 0.5, None);

        assert_eq!(results.len(), 2);
        let ids: std::collections::HashSet<&str> = results.iter().map(|r| r.fact.id.as_str()).collect();
        assert!(ids.contains("1"));
        assert!(ids.contains("2"));
    }

    #[test]
    fn test_empty_store() {
        let store = VectorStore::new();
        let results = store.search_hybrid_bll(&[1.0, 0.0, 0.0], "", 10, 0.0, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_context_augmented_bm25() {
        let mut fact = create_test_fact("1", "rust programming", vec![1.0, 0.0, 0.0]);
        fact.context_window = vec!["cargo build system".to_string(), "memory safety".to_string()];

        let store = VectorStore::from_facts(vec![fact]);
        // BM25 should find "cargo" even though it's only in context_window
        let bm25_results = store.bm25_search_facts("cargo", 10);
        assert!(!bm25_results.is_empty());
    }
}
