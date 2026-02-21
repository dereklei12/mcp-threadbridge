//! Belief - Bayesian Confidence with Beta Distribution
//!
//! A confidence of 0.7 is meaningless without knowing the uncertainty.
//! Beta(7, 3) and Beta(70, 30) both have mean 0.7, but vastly different certainty.

use rand::thread_rng;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};

/// A belief represented as a Beta distribution
///
/// Beta(alpha, beta) where:
/// - alpha = 1 + positive_evidence
/// - beta = 1 + negative_evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub alpha: f32,
    pub beta: f32,
    #[serde(default)]
    pub decay_rate: f32,
}

impl Default for Belief {
    fn default() -> Self {
        Self::uninformed()
    }
}

impl PartialEq<f32> for Belief {
    fn eq(&self, other: &f32) -> bool {
        (self.mean() - other).abs() < 0.0001
    }
}

impl PartialOrd<f32> for Belief {
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.mean().partial_cmp(other)
    }
}

impl Belief {
    /// Beta(1, 1) = uniform distribution = "I have no idea"
    pub fn uninformed() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            decay_rate: 0.0,
        }
    }

    /// Returns true for the default Beta(1,1) prior (no observations)
    pub fn is_uninformed(&self) -> bool {
        (self.alpha - 1.0).abs() < f32::EPSILON && (self.beta - 1.0).abs() < f32::EPSILON
    }

    /// Point estimate of the probability
    pub fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Thompson sampling: sample from the belief distribution
    pub fn sample(&self) -> f32 {
        let dist = Beta::new(self.alpha as f64, self.beta as f64)
            .unwrap_or_else(|_| Beta::new(1.0, 1.0).unwrap());
        dist.sample(&mut thread_rng()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uninformed_belief() {
        let b = Belief::uninformed();
        assert!((b.mean() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_is_uninformed() {
        let b = Belief::uninformed();
        assert!(b.is_uninformed());

        let b2 = Belief { alpha: 2.0, beta: 1.0, decay_rate: 0.0 };
        assert!(!b2.is_uninformed());
    }

    #[test]
    fn test_sample_in_range() {
        let b = Belief { alpha: 5.0, beta: 2.0, decay_rate: 0.0 };
        for _ in 0..100 {
            let s = b.sample();
            assert!(s >= 0.0 && s <= 1.0);
        }
    }
}
