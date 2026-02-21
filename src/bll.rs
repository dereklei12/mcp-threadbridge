//! Bayesian Last Layer (BLL) — learned reranker with online Bayesian adaptation.
//!
//! ## Architecture
//!
//! BLL v1 (legacy):
//!   input = concat(query, candidate) → 1536-dim
//!   features = ReLU(W1 @ input + b1) → 256-dim
//!
//! BLL v2 (current):
//!   input = concat(query, candidate, query*candidate, |query-candidate|) → 3072-dim
//!   features = ReLU(W1 @ input + b1) → 256-dim
//!
//! Bayesian Linear Regression on features:
//!   mu = mu_w · features + bias
//!   var = features^T @ Sigma_w @ features + sigma_sq
//!
//! ## Online Update (closed-form)
//!
//! Given (features, reward), update posterior via Woodbury rank-1:
//!   Sigma_new = Sigma_old - (Sigma_old @ phi @ phi^T @ Sigma_old) / (sigma_sq + phi^T @ Sigma_old @ phi)
//!   mu_new += gain * error
//!
//! ## Binary Formats
//!
//! BLL1 (v1):
//!   [4B] magic "BLL1", embed_dim, hidden_dim, sigma_sq
//!   W1(hidden*embed*2), b1, mu_0, mu_0_bias, prior_cov_diag
//!
//! BLL2 (v2, interaction features):
//!   [4B] magic "BLL2", embed_dim, hidden_dim, sigma_sq
//!   W1(hidden*embed*4), b1
//!   mu_0, mu_0_bias, prior_cov_diag

use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2};
use std::fs;
use std::io::Read;
use std::path::Path;
use tracing::{debug, info, warn};

const POSTERIOR_MAGIC: &[u8; 4] = b"BPO1";

/// Feature extractor architecture variant
enum FeatureExtractor {
    /// v1: single layer, concat(q, c) input
    V1 {
        w1: Array2<f32>,     // (hidden_dim, embed_dim * 2)
        b1: Array1<f32>,     // (hidden_dim,)
    },
    /// v2: single layer, concat(q, c, q*c, |q-c|) input
    V2 {
        w1: Array2<f32>,     // (hidden_dim, embed_dim * 4)
        b1: Array1<f32>,     // (hidden_dim,)
    },
}

/// Bayesian Last Layer model
pub struct BayesianLastLayer {
    embed_dim: usize,
    hidden_dim: usize,

    // Frozen feature extractor
    extractor: FeatureExtractor,

    // BLL posterior (online-updatable)
    mu_w: Array1<f32>,       // (hidden_dim,) — posterior mean
    sigma_w: Array2<f32>,    // (hidden_dim, hidden_dim) — posterior covariance
    bias: f32,               // output bias
    sigma_sq: f32,           // noise variance

    // Update counter
    update_count: u64,
}

impl BayesianLastLayer {
    /// Load model from binary weights file (auto-detects BLL1 or BLL2).
    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path)
            .with_context(|| format!("Failed to read BLL weights: {}", path.display()))?;
        let mut cursor = &data[..];

        // Magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;

        match &magic {
            b"BLL1" => Self::load_v1(&mut cursor, &data),
            b"BLL2" => Self::load_v2(&mut cursor, &data),
            _ => bail!("Invalid BLL magic: expected BLL1 or BLL2, got {:?}", magic),
        }
    }

    fn load_v1(cursor: &mut &[u8], data: &[u8]) -> Result<Self> {
        let embed_dim = read_u32(cursor)? as usize;
        let hidden_dim = read_u32(cursor)? as usize;
        let sigma_sq = read_f32(cursor)?;

        let w1_data = read_f32_vec(cursor, hidden_dim * embed_dim * 2)?;
        let w1 = Array2::from_shape_vec((hidden_dim, embed_dim * 2), w1_data)
            .context("W1 shape mismatch")?;
        let b1 = Array1::from_vec(read_f32_vec(cursor, hidden_dim)?);

        let mu_0 = Array1::from_vec(read_f32_vec(cursor, hidden_dim)?);
        let mu_0_bias = read_f32(cursor)?;
        let sigma_prior_diag = Array1::from_vec(read_f32_vec(cursor, hidden_dim)?);

        let mu_w = mu_0;
        let sigma_w = Array2::from_diag(&sigma_prior_diag);

        info!(
            "Loaded BLL v1: embed_dim={}, hidden_dim={}, sigma_sq={:.4}, weights={}KB",
            embed_dim, hidden_dim, sigma_sq, data.len() / 1024
        );

        Ok(Self {
            embed_dim,
            hidden_dim,
            extractor: FeatureExtractor::V1 { w1, b1 },
            mu_w,
            sigma_w,
            bias: mu_0_bias,
            sigma_sq,
            update_count: 0,
        })
    }

    fn load_v2(cursor: &mut &[u8], data: &[u8]) -> Result<Self> {
        let embed_dim = read_u32(cursor)? as usize;
        let hidden_dim = read_u32(cursor)? as usize;
        let sigma_sq = read_f32(cursor)?;

        // Single layer: (hidden_dim, embed_dim * 4)
        let w1_data = read_f32_vec(cursor, hidden_dim * embed_dim * 4)?;
        let w1 = Array2::from_shape_vec((hidden_dim, embed_dim * 4), w1_data)
            .context("W1 shape mismatch")?;
        let b1 = Array1::from_vec(read_f32_vec(cursor, hidden_dim)?);

        // BLL prior
        let mu_0 = Array1::from_vec(read_f32_vec(cursor, hidden_dim)?);
        let mu_0_bias = read_f32(cursor)?;
        let sigma_prior_diag = Array1::from_vec(read_f32_vec(cursor, hidden_dim)?);

        let mu_w = mu_0;
        let sigma_w = Array2::from_diag(&sigma_prior_diag);

        info!(
            "Loaded BLL v2: embed_dim={}, hidden_dim={}, sigma_sq={:.4}, weights={}KB",
            embed_dim, hidden_dim, sigma_sq, data.len() / 1024
        );

        Ok(Self {
            embed_dim,
            hidden_dim,
            extractor: FeatureExtractor::V2 { w1, b1 },
            mu_w,
            sigma_w,
            bias: mu_0_bias,
            sigma_sq,
            update_count: 0,
        })
    }

    /// Try to load from a path, returning None if file doesn't exist.
    pub fn try_load(path: &Path) -> Option<Self> {
        if !path.exists() {
            debug!("BLL weights not found at {}", path.display());
            return None;
        }
        match Self::load(path) {
            Ok(bll) => Some(bll),
            Err(e) => {
                warn!("Failed to load BLL weights: {}", e);
                None
            }
        }
    }

    /// Extract 256-dim features from (query_emb, candidate_emb).
    pub fn extract_features(&self, query_emb: &[f32], candidate_emb: &[f32]) -> Array1<f32> {
        assert_eq!(query_emb.len(), self.embed_dim);
        assert_eq!(candidate_emb.len(), self.embed_dim);

        match &self.extractor {
            FeatureExtractor::V1 { w1, b1 } => {
                // v1: concat(q, c) → W1 → ReLU
                let mut input = Vec::with_capacity(self.embed_dim * 2);
                input.extend_from_slice(query_emb);
                input.extend_from_slice(candidate_emb);
                let input = Array1::from_vec(input);

                let hidden = w1.dot(&input) + b1;
                hidden.mapv(|x| x.max(0.0))
            }
            FeatureExtractor::V2 { w1, b1 } => {
                // v2: concat(q, c, q*c, |q-c|) → W1 → ReLU
                let mut input = Vec::with_capacity(self.embed_dim * 4);
                input.extend_from_slice(query_emb);
                input.extend_from_slice(candidate_emb);
                for i in 0..self.embed_dim {
                    input.push(query_emb[i] * candidate_emb[i]);
                }
                for i in 0..self.embed_dim {
                    input.push((query_emb[i] - candidate_emb[i]).abs());
                }
                let input = Array1::from_vec(input);

                let hidden = w1.dot(&input) + b1;
                hidden.mapv(|x| x.max(0.0))
            }
        }
    }

    /// Predict relevance score with uncertainty.
    ///
    /// Returns (mean, variance) where:
    /// - mean: expected relevance (higher = more relevant)
    /// - variance: epistemic uncertainty (higher = less certain)
    pub fn predict(&self, query_emb: &[f32], candidate_emb: &[f32]) -> (f32, f32) {
        let features = self.extract_features(query_emb, candidate_emb);
        self.predict_from_features(&features)
    }

    /// Predict from pre-extracted features.
    fn predict_from_features(&self, features: &Array1<f32>) -> (f32, f32) {
        // mu = mu_w · features + bias
        let mu = self.mu_w.dot(features) + self.bias;

        // var = features^T @ Sigma_w @ features + sigma_sq
        let sigma_features = self.sigma_w.dot(features);
        let var = features.dot(&sigma_features) + self.sigma_sq;

        (mu, var.max(1e-8))
    }

    /// Online Bayesian Linear Regression update (closed-form).
    ///
    /// Uses rank-1 update (Woodbury identity) for efficiency: O(d^2) not O(d^3).
    pub fn update(&mut self, features: &Array1<f32>, reward: f32) {
        let d = self.hidden_dim;
        assert_eq!(features.len(), d);

        // Rank-1 update via Woodbury identity:
        // Sigma_new = Sigma_old - (Sigma_old @ phi @ phi^T @ Sigma_old) / (sigma_sq + phi^T @ Sigma_old @ phi)
        let sigma_phi = self.sigma_w.dot(features);
        let phi_sigma_phi = features.dot(&sigma_phi);
        let denom = self.sigma_sq + phi_sigma_phi;

        if denom.abs() < 1e-12 {
            warn!("BLL update: denominator too small, skipping");
            return;
        }

        // Update covariance
        for i in 0..d {
            for j in 0..d {
                self.sigma_w[[i, j]] -= sigma_phi[i] * sigma_phi[j] / denom;
            }
        }

        // Update mean
        let prediction = self.mu_w.dot(features) + self.bias;
        let error = reward - prediction;
        let gain = &sigma_phi / denom;

        for i in 0..d {
            self.mu_w[i] += gain[i] * error;
        }

        self.update_count += 1;
        debug!("BLL update #{}: error={:.4}, reward={:.2}", self.update_count, error, reward);
    }

    /// Save posterior state to disk (for persistence across restarts).
    pub fn save_posterior(&self, path: &Path) -> Result<()> {
        let d = self.hidden_dim;
        let mut buf = Vec::new();

        // Magic + metadata
        buf.extend_from_slice(POSTERIOR_MAGIC);
        write_u32(&mut buf, d as u32);
        write_u64(&mut buf, self.update_count);

        // mu_w
        for &v in self.mu_w.iter() {
            write_f32_val(&mut buf, v);
        }

        // sigma_w (full matrix, row-major)
        for row in self.sigma_w.rows() {
            for &v in row.iter() {
                write_f32_val(&mut buf, v);
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &buf)
            .with_context(|| format!("Failed to save BLL posterior: {}", path.display()))?;

        debug!("Saved BLL posterior ({} updates, {} bytes)", self.update_count, buf.len());
        Ok(())
    }

    /// Load posterior state from disk.
    pub fn load_posterior(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            debug!("No saved posterior at {}", path.display());
            return Ok(());
        }

        let data = fs::read(path)?;
        let mut cursor = &data[..];

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != POSTERIOR_MAGIC {
            bail!("Invalid posterior magic: expected BPO1, got {:?}", magic);
        }

        let d = read_u32(&mut cursor)? as usize;
        if d != self.hidden_dim {
            bail!("Posterior dim mismatch: file={}, model={}", d, self.hidden_dim);
        }

        self.update_count = read_u64(&mut cursor)?;

        let mu_data = read_f32_vec(&mut cursor, d)?;
        self.mu_w = Array1::from_vec(mu_data);

        let sigma_data = read_f32_vec(&mut cursor, d * d)?;
        self.sigma_w = Array2::from_shape_vec((d, d), sigma_data)
            .context("Sigma_w shape mismatch")?;

        info!("Loaded BLL posterior ({} updates)", self.update_count);
        Ok(())
    }

    pub fn update_count(&self) -> u64 {
        self.update_count
    }
}

// ==========================================================================
// Binary I/O helpers
// ==========================================================================

fn read_u32(cursor: &mut &[u8]) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(cursor: &mut &[u8]) -> Result<u64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(cursor: &mut &[u8]) -> Result<f32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f32_vec(cursor: &mut &[u8], n: usize) -> Result<Vec<f32>> {
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        result.push(read_f32(cursor)?);
    }
    Ok(result)
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_f32_val(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use tempfile::TempDir;

    /// Create a minimal BLL v1 for testing.
    fn make_test_bll_v1() -> BayesianLastLayer {
        let embed_dim = 4;
        let hidden_dim = 2;

        let mut w1 = Array2::zeros((hidden_dim, embed_dim * 2));
        w1[[0, 0]] = 1.0;
        w1[[1, 1]] = 1.0;
        let b1 = Array1::zeros(hidden_dim);

        let mu_w = Array1::from_vec(vec![0.5, 0.5]);
        let sigma_w = Array2::from_diag(&Array1::from_vec(vec![1.0, 1.0]));

        BayesianLastLayer {
            embed_dim,
            hidden_dim,
            extractor: FeatureExtractor::V1 { w1, b1 },
            mu_w,
            sigma_w,
            bias: 0.0,
            sigma_sq: 0.1,
            update_count: 0,
        }
    }

    /// Create a minimal BLL v2 for testing.
    fn make_test_bll_v2() -> BayesianLastLayer {
        let embed_dim = 4;
        let hidden_dim = 2;

        let mut w1 = Array2::zeros((hidden_dim, embed_dim * 4));
        w1[[0, 0]] = 1.0;
        w1[[1, 1]] = 1.0;
        let b1 = Array1::zeros(hidden_dim);

        let mu_w = Array1::from_vec(vec![0.5, 0.5]);
        let sigma_w = Array2::from_diag(&Array1::from_vec(vec![1.0, 1.0]));

        BayesianLastLayer {
            embed_dim,
            hidden_dim,
            extractor: FeatureExtractor::V2 { w1, b1 },
            mu_w,
            sigma_w,
            bias: 0.0,
            sigma_sq: 0.1,
            update_count: 0,
        }
    }

    #[test]
    fn test_v1_predict_output_range() {
        let bll = make_test_bll_v1();
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0, 0.0];
        let (mu, var) = bll.predict(&q, &c);
        assert!(mu.is_finite());
        assert!(var > 0.0);
    }

    #[test]
    fn test_v2_predict_output_range() {
        let bll = make_test_bll_v2();
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0, 0.0];
        let (mu, var) = bll.predict(&q, &c);
        assert!(mu.is_finite());
        assert!(var > 0.0);
    }

    #[test]
    fn test_v2_interaction_features() {
        let bll = make_test_bll_v2();
        let q = vec![1.0, 0.5, 0.0, 0.0];
        let c = vec![0.5, 1.0, 0.0, 0.0];
        let features = bll.extract_features(&q, &c);
        assert_eq!(features.len(), 2); // hidden_dim
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_update_changes_posterior() {
        let mut bll = make_test_bll_v2();
        let features = Array1::from_vec(vec![1.0, 0.5]);
        let mu_before = bll.mu_w.clone();

        bll.update(&features, 1.0);

        assert_ne!(bll.mu_w, mu_before);
        assert_eq!(bll.update_count, 1);
    }

    #[test]
    fn test_update_reduces_variance() {
        let mut bll = make_test_bll_v2();
        let features = Array1::from_vec(vec![1.0, 0.0]);

        let var_before = bll.sigma_w[[0, 0]];
        bll.update(&features, 1.0);
        let var_after = bll.sigma_w[[0, 0]];

        assert!(var_after < var_before);
    }

    #[test]
    fn test_posterior_save_load_roundtrip() {
        let mut bll = make_test_bll_v2();
        let features = Array1::from_vec(vec![1.0, 0.5]);
        bll.update(&features, 0.8);
        bll.update(&features, 0.9);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("posterior.bin");

        bll.save_posterior(&path).unwrap();

        let mut bll2 = make_test_bll_v2();
        bll2.load_posterior(&path).unwrap();

        assert_eq!(bll2.update_count, 2);
        assert!((bll.mu_w[0] - bll2.mu_w[0]).abs() < 1e-6);
        assert!((bll.sigma_w[[0, 0]] - bll2.sigma_w[[0, 0]]).abs() < 1e-6);
    }

}
