//! Thompson sampling verification budget allocator.
//!
//! Verification is expensive (requires file I/O, hash comparison, potentially LLM calls).
//! This module allocates a finite verification budget optimally using Thompson sampling
//! from the existing Beta utility distributions, combined with information gain from
//! provenance uncertainty and CUSUM alarm signals.
//!
//! Priority = Thompson_sample(utility) * (entropy(provenance_score) + CUSUM_alarm_score)
//!
//! Key papers:
//! - Christ et al. 2025: verification query complexity O(K/(mε²)) < learning O(K/ε²)
//! - Rad et al. 2026: Youden index J = TPR - FPR phase transition

use crate::cusum::ChangeTracker;
use crate::types::{Fact, RevisionStatus};

/// A verification task: which fact to verify, and its Thompson-sampled priority
#[derive(Debug, Clone)]
pub struct VerificationTask {
    pub fact_id: String,
    /// Thompson-sampled priority (higher = verify first)
    pub priority: f32,
    /// Expected information gain from verification
    pub info_gain: f32,
}

/// Budget allocator (not serialized; recomputed each session)
pub struct VerificationBudget {
    /// Maximum number of verifications per load_thread call
    pub max_verifications: usize,
    /// Tasks sorted by priority (descending)
    pub tasks: Vec<VerificationTask>,
}

impl VerificationBudget {
    /// Allocate verification budget across facts.
    ///
    /// Scoring:
    /// 1. Thompson sample from each fact's utility Beta distribution (exploration)
    /// 2. Compute information entropy of the provenance score (higher uncertainty = more value)
    /// 3. Boost by CUSUM alarm score (files actively changing = urgent)
    /// 4. Take top-K by combined priority
    pub fn allocate(
        facts: &[Fact],
        tracker: &ChangeTracker,
        consecutive_threshold: u32,
        ewma_threshold: f64,
        max_budget: usize,
    ) -> Self {
        let mut tasks = Vec::new();

        for fact in facts {
            // Skip non-active or provenance-less facts
            if fact.revision_status != RevisionStatus::Active {
                continue;
            }
            if fact.provenance.dependencies.is_empty() {
                continue;
            }

            // Thompson sample from the fact's utility Beta distribution
            let utility_sample = fact.utility.sample();

            // Information gain: entropy of Bernoulli(provenance_score)
            // Higher entropy = more uncertain about validity = more value in verifying
            let p = (fact.provenance.score as f64).clamp(0.01, 0.99);
            let entropy = -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
            let max_entropy = std::f64::consts::LN_2; // entropy of Bernoulli(0.5)
            let normalized_entropy = (entropy / max_entropy) as f32; // [0, 1]

            // Change alarm boost: facts touching actively-changing files get priority
            let alarm = tracker.fact_alarm_score(fact, consecutive_threshold, ewma_threshold) as f32;

            // Combined priority: utility * (uncertainty + urgency)
            // Floor of 0.1 prevents high-certainty facts from being permanently
            // excluded — we need to occasionally re-verify "looks fine" facts
            // to discover silent corruption (the whole point of verification).
            let priority = utility_sample * (normalized_entropy + alarm).max(0.1);

            // Information gain approximation via Beta variance
            let var = (fact.utility.alpha * fact.utility.beta)
                / ((fact.utility.alpha + fact.utility.beta).powi(2)
                    * (fact.utility.alpha + fact.utility.beta + 1.0));
            let info_gain = var.sqrt() * (1.0 + alarm);

            tasks.push(VerificationTask {
                fact_id: fact.id.clone(),
                priority,
                info_gain,
            });
        }

        // Sort by priority descending, take top max_budget
        tasks.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tasks.truncate(max_budget);

        Self {
            max_verifications: max_budget,
            tasks,
        }
    }

    /// Check if a fact should be verified in this budget round
    pub fn should_verify(&self, fact_id: &str) -> bool {
        self.tasks.iter().any(|t| t.fact_id == fact_id)
    }

    /// Number of facts selected for verification
    pub fn selected_count(&self) -> usize {
        self.tasks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::Belief;
    use crate::provenance::{Provenance, ProvenanceDep};
    use crate::types::{DependencyStatus, DependencyType, FactCategory, RevisionStatus};
    use chrono::Utc;

    fn make_fact(id: &str, prov_score: f32, has_deps: bool) -> Fact {
        let deps = if has_deps {
            vec![ProvenanceDep {
                file: "src/main.rs".into(),
                symbol: Some("main".into()),
                line_range: None,
                dep_type: DependencyType::DefinesSemantic,
                status: if prov_score >= 1.0 {
                    DependencyStatus::Intact
                } else {
                    DependencyStatus::Modified
                },
                content_hash: None,
                last_verified_at: None,
            }]
        } else {
            Vec::new()
        };

        Fact {
            id: id.to_string(),
            content: format!("fact {}", id),
            category: FactCategory::General,
            confidence: 1.0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            embedding: None,
            utility: Belief::uninformed(),
            access_count: 0,
            last_used_at: None,
            context_window: Vec::new(),
            session_id: None,
            anchors: Vec::new(),
            provenance: Provenance {
                dependencies: deps,
                score: prov_score,
            },
            revision_status: RevisionStatus::Active,
            supersedes: Vec::new(),
            superseded_by: Vec::new(),
        }
    }

    #[test]
    fn test_budget_allocation_basic() {
        let facts = vec![
            make_fact("a", 0.5, true),
            make_fact("b", 1.0, true),
            make_fact("c", 0.3, true),
        ];
        let tracker = ChangeTracker::default();

        let budget = VerificationBudget::allocate(&facts, &tracker, 3, 2.5, 2);
        assert_eq!(budget.selected_count(), 2);
    }

    #[test]
    fn test_budget_skips_no_deps() {
        let facts = vec![make_fact("a", 1.0, false)]; // no deps
        let tracker = ChangeTracker::default();

        let budget = VerificationBudget::allocate(&facts, &tracker, 3, 2.5, 10);
        assert_eq!(budget.selected_count(), 0);
    }

    #[test]
    fn test_budget_skips_superseded() {
        let mut fact = make_fact("a", 0.5, true);
        fact.revision_status = RevisionStatus::Superseded;
        let tracker = ChangeTracker::default();

        let budget = VerificationBudget::allocate(&[fact], &tracker, 3, 2.5, 10);
        assert_eq!(budget.selected_count(), 0);
    }
}
