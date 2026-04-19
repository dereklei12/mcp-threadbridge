//! Change detection for tracked project files.
//!
//! Uses a dual-detector approach designed for session-level granularity:
//!
//! **Detector A: Consecutive Change Counter** (primary)
//!   counter_t = (counter_{t-1} + 1) if file_changed else 0
//!   alarm if counter_t >= threshold
//!   - P(false alarm) = p^threshold (e.g., p=0.3, threshold=3 → 2.7%)
//!   - No parameter estimation needed
//!
//! **Detector B: EWMA** (secondary, catches "4 out of 5" patterns)
//!   score_t = α * score_{t-1} + (1 if changed else 0)
//!   alarm if score_t > threshold
//!
//! **Combined alarm_score** = max(counter/c_threshold, ewma/e_threshold)
//!
//! This replaces CUSUM, which retains only ~4% statistical power at
//! session-level granularity (1 observation per session gap vs N per N commits).

use crate::file_manifest::ManifestDiff;
use crate::types::Fact;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Default detector parameters
const DEFAULT_CONSECUTIVE_THRESHOLD: u32 = 3;
const DEFAULT_EWMA_ALPHA: f64 = 0.7;
const DEFAULT_EWMA_THRESHOLD: f64 = 2.5;

/// Change detection state for a single tracked file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeDetectorState {
    /// Relative file path
    pub file: String,
    /// Consecutive sessions where this file changed
    pub consecutive_changes: u32,
    /// EWMA score (exponentially weighted change frequency)
    pub ewma_score: f64,
    /// Total sessions observed
    pub n_obs: u64,
    /// Whether alarm is currently active
    pub alarm: bool,
    /// Timestamp of last observation
    pub last_observed: DateTime<Utc>,
}

/// Change tracker for all files in a project.
/// Replaces CusumTracker — designed for session-level (not commit-level) granularity.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChangeTracker {
    #[serde(default)]
    pub states: Vec<ChangeDetectorState>,
}

impl ChangeTracker {
    /// Update detector states from a FileManifest diff.
    /// Each call represents one session-level observation per tracked file.
    pub fn update_from_diff(
        &mut self,
        diff: &ManifestDiff,
        consecutive_threshold: u32,
        ewma_alpha: f64,
        ewma_threshold: f64,
    ) {
        // NOTE: we process even empty diffs because tracked files need their
        // consecutive_changes counter reset and EWMA decayed when nothing changes.
        let now = Utc::now();
        let all_affected = diff.all_affected();

        if !diff.is_empty() {
            debug!(
                "ChangeTracker: {} changed, {} added, {} removed",
                diff.changed.len(),
                diff.added.len(),
                diff.removed.len()
            );
        }

        for state in &mut self.states {
            let changed = all_affected.contains(&state.file);

            // Detector A: consecutive change counter
            if changed {
                state.consecutive_changes += 1;
            } else {
                state.consecutive_changes = 0;
            }

            // Detector B: EWMA
            let x = if changed { 1.0 } else { 0.0 };
            state.ewma_score = ewma_alpha * state.ewma_score + x;

            state.n_obs += 1;
            state.last_observed = now;

            // Combined alarm: either detector can trigger
            state.alarm = state.consecutive_changes >= consecutive_threshold
                || state.ewma_score > ewma_threshold;
        }
    }

    /// Ensure a file is being tracked. No-op if already tracked.
    pub fn ensure_file_tracked(&mut self, file: &str) {
        if !self.states.iter().any(|s| s.file == file) {
            self.states.push(ChangeDetectorState {
                file: file.to_string(),
                consecutive_changes: 0,
                ewma_score: 0.0,
                n_obs: 0,
                alarm: false,
                last_observed: Utc::now(),
            });
        }
    }

    /// Iterator over files currently in alarm
    pub fn files_in_alarm(&self) -> impl Iterator<Item = &str> {
        self.states
            .iter()
            .filter(|s| s.alarm)
            .map(|s| s.file.as_str())
    }

    /// Alarm score for a single file: max(counter_ratio, ewma_ratio), in [0, 1].
    pub fn alarm_score(
        &self,
        file: &str,
        consecutive_threshold: u32,
        ewma_threshold: f64,
    ) -> f64 {
        for state in &self.states {
            if state.file == file {
                let counter_ratio =
                    state.consecutive_changes as f64 / consecutive_threshold.max(1) as f64;
                let ewma_ratio = state.ewma_score / ewma_threshold.max(0.01);
                return counter_ratio.max(ewma_ratio).min(1.0);
            }
        }
        0.0
    }

    /// Maximum alarm score across all provenance dependencies of a fact
    pub fn fact_alarm_score(
        &self,
        fact: &Fact,
        consecutive_threshold: u32,
        ewma_threshold: f64,
    ) -> f64 {
        let mut max_alarm = 0.0;
        for dep in &fact.provenance.dependencies {
            let a = self.alarm_score(&dep.file, consecutive_threshold, ewma_threshold);
            if a > max_alarm {
                max_alarm = a;
            }
        }
        max_alarm
    }

    /// Number of tracked files
    pub fn tracked_files(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_manifest::FileManifest;

    fn make_diff(changed: &[&str], added: &[&str], removed: &[&str]) -> ManifestDiff {
        ManifestDiff {
            changed: changed.iter().map(|s| s.to_string()).collect(),
            added: added.iter().map(|s| s.to_string()).collect(),
            removed: removed.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_consecutive_alarm() {
        let mut tracker = ChangeTracker::default();
        tracker.ensure_file_tracked("src/main.rs");
        tracker.ensure_file_tracked("src/lib.rs");

        let threshold = 3u32;
        let alpha = DEFAULT_EWMA_ALPHA;
        let ewma_th = DEFAULT_EWMA_THRESHOLD;

        // 3 consecutive sessions where main.rs changes → alarm
        for _ in 0..3 {
            let diff = make_diff(&["src/main.rs"], &[], &[]);
            tracker.update_from_diff(&diff, threshold, alpha, ewma_th);
        }

        assert!(
            tracker.states.iter().find(|s| s.file == "src/main.rs").unwrap().alarm,
            "main.rs should be in alarm after 3 consecutive changes"
        );
        assert!(
            !tracker.states.iter().find(|s| s.file == "src/lib.rs").unwrap().alarm,
            "lib.rs should NOT be in alarm (never changed)"
        );
    }

    #[test]
    fn test_consecutive_reset() {
        let mut tracker = ChangeTracker::default();
        tracker.ensure_file_tracked("a.rs");

        let threshold = 3u32;

        // 2 changes, then a gap, then 2 more → no alarm (consecutive resets)
        for _ in 0..2 {
            tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), threshold, 0.7, 99.0);
        }
        // Gap
        tracker.update_from_diff(&make_diff(&[], &[], &[]), threshold, 0.7, 99.0);
        // 2 more changes
        for _ in 0..2 {
            tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), threshold, 0.7, 99.0);
        }

        let state = tracker.states.iter().find(|s| s.file == "a.rs").unwrap();
        assert_eq!(state.consecutive_changes, 2);
        assert!(!state.alarm, "Should not alarm — consecutive counter was reset by gap");
    }

    #[test]
    fn test_ewma_alarm() {
        let mut tracker = ChangeTracker::default();
        tracker.ensure_file_tracked("a.rs");

        // EWMA with alpha=0.7, threshold=2.5
        // Recurrence: score_t = 0.7 * score_{t-1} + x_t
        //
        // Session 1 (change):  0*0.7 + 1 = 1.000
        // Session 2 (gap):     1.0*0.7 + 0 = 0.700
        // Session 3 (change):  0.7*0.7 + 1 = 1.490
        // Session 4 (gap):     1.49*0.7 + 0 = 1.043
        // Session 5 (change):  1.043*0.7 + 1 = 1.730
        // Session 6 (change):  1.730*0.7 + 1 = 2.211
        // Session 7 (change):  2.211*0.7 + 1 = 2.548 > 2.5 → alarm!

        let ewma_threshold = 2.5;
        let consecutive_threshold = 99; // disable consecutive for this test

        // Pattern: change, gap, change, gap, change, change
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), consecutive_threshold, 0.7, ewma_threshold);
        tracker.update_from_diff(&make_diff(&[], &[], &[]), consecutive_threshold, 0.7, ewma_threshold);
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), consecutive_threshold, 0.7, ewma_threshold);
        tracker.update_from_diff(&make_diff(&[], &[], &[]), consecutive_threshold, 0.7, ewma_threshold);
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), consecutive_threshold, 0.7, ewma_threshold);
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), consecutive_threshold, 0.7, ewma_threshold);

        let state = tracker.states.iter().find(|s| s.file == "a.rs").unwrap();
        assert!(!state.alarm, "Should not alarm yet, ewma={:.3}", state.ewma_score);

        // Session 7: change → crosses threshold
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), consecutive_threshold, 0.7, ewma_threshold);
        let state = tracker.states.iter().find(|s| s.file == "a.rs").unwrap();
        assert!(state.alarm, "Should alarm now via EWMA, ewma={:.3}", state.ewma_score);
    }

    #[test]
    fn test_no_alarm_stable() {
        let mut tracker = ChangeTracker::default();
        tracker.ensure_file_tracked("a.rs");

        // 10 sessions, file changes only once
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), 3, 0.7, 2.5);
        for _ in 0..9 {
            tracker.update_from_diff(&make_diff(&[], &[], &[]), 3, 0.7, 2.5);
        }

        let state = tracker.states.iter().find(|s| s.file == "a.rs").unwrap();
        assert!(!state.alarm);
        assert_eq!(state.consecutive_changes, 0);
    }

    #[test]
    fn test_ensure_file_tracked() {
        let mut tracker = ChangeTracker::default();
        tracker.ensure_file_tracked("src/main.rs");
        tracker.ensure_file_tracked("src/main.rs"); // duplicate
        tracker.ensure_file_tracked("src/lib.rs");
        assert_eq!(tracker.tracked_files(), 2);
    }

    #[test]
    fn test_alarm_score() {
        let mut tracker = ChangeTracker::default();
        tracker.ensure_file_tracked("a.rs");

        // 2 consecutive changes out of threshold 3
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), 3, 0.7, 2.5);
        tracker.update_from_diff(&make_diff(&["a.rs"], &[], &[]), 3, 0.7, 2.5);

        let score = tracker.alarm_score("a.rs", 3, 2.5);
        assert!(score > 0.0 && score < 1.0, "Score should be partial: {}", score);

        let score_unknown = tracker.alarm_score("unknown.rs", 3, 2.5);
        assert_eq!(score_unknown, 0.0);
    }
}
