//! Configuration for mcp-threadbridge
//!
//! TOML config at ~/.threadbridge/config.toml
//! All fields have sensible defaults matching Config I (benchmark optimal).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;
use tracing::{debug, info, warn};

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub verification: VerificationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Enable provenance tracking (subsystem E)
    #[serde(default = "default_true")]
    pub provenance_enabled: bool,
    /// Enable AGM revision tracking (subsystem A)
    #[serde(default = "default_true")]
    pub revision_enabled: bool,
    /// Enable CUSUM change-point detection (subsystem C)
    #[serde(default = "default_true")]
    pub cusum_enabled: bool,
    /// Enable Thompson sampling verification budget (subsystem D)
    #[serde(default = "default_true")]
    pub budget_enabled: bool,
    /// Enable sheaf cohomology consistency checking (subsystem F)
    #[serde(default = "default_true")]
    pub sheaf_enabled: bool,
    /// Maximum expensive verifications per load_thread (subsystem D)
    #[serde(default = "default_verification_budget")]
    pub verification_budget: usize,
    /// Similarity threshold for AGM supersession detection
    #[serde(default = "default_supersession_threshold")]
    pub supersession_threshold: f32,
    /// Minimum facts before running sheaf cohomology
    #[serde(default = "default_sheaf_min_facts")]
    pub sheaf_min_facts: usize,
    /// Edge similarity threshold for sheaf simplicial complex
    #[serde(default = "default_sheaf_edge_threshold")]
    pub sheaf_edge_threshold: f32,
    /// Consecutive change threshold for alarm (detector A)
    #[serde(default = "default_consecutive_threshold")]
    pub consecutive_threshold: u32,
    /// EWMA smoothing factor α (detector B)
    #[serde(default = "default_ewma_alpha")]
    pub ewma_alpha: f64,
    /// EWMA alarm threshold (detector B)
    #[serde(default = "default_ewma_threshold")]
    pub ewma_threshold: f64,
}

fn default_true() -> bool {
    true
}
fn default_verification_budget() -> usize {
    20
}
fn default_supersession_threshold() -> f32 {
    0.85
}
fn default_sheaf_min_facts() -> usize {
    10
}
fn default_sheaf_edge_threshold() -> f32 {
    0.6
}
fn default_consecutive_threshold() -> u32 {
    3
}
fn default_ewma_alpha() -> f64 {
    0.7
}
fn default_ewma_threshold() -> f64 {
    2.5
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            provenance_enabled: true,
            revision_enabled: true,
            cusum_enabled: true,
            budget_enabled: true,
            sheaf_enabled: true,
            verification_budget: default_verification_budget(),
            supersession_threshold: default_supersession_threshold(),
            sheaf_min_facts: default_sheaf_min_facts(),
            sheaf_edge_threshold: default_sheaf_edge_threshold(),
            consecutive_threshold: default_consecutive_threshold(),
            ewma_alpha: default_ewma_alpha(),
            ewma_threshold: default_ewma_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_dimension")]
    pub dimension: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            dimension: default_dimension(),
        }
    }
}

fn default_model() -> String {
    "snowflake-arctic-embed-m".to_string()
}

fn default_dimension() -> usize {
    768
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_limit")]
    pub default_limit: usize,
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
    #[serde(default = "default_utility_lambda")]
    pub utility_lambda: f32,
    #[serde(default = "default_bll_enabled")]
    pub bll_enabled: bool,
    #[serde(default = "default_bll_weights_path")]
    pub bll_weights_path: String,
    #[serde(default = "default_context_window_size")]
    pub context_window_size: usize,
    /// Grounding score below which a fact is considered stale
    #[serde(default = "default_anchor_stale_threshold")]
    pub anchor_stale_threshold: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: default_limit(),
            min_similarity: default_min_similarity(),
            utility_lambda: default_utility_lambda(),
            bll_enabled: default_bll_enabled(),
            bll_weights_path: default_bll_weights_path(),
            context_window_size: default_context_window_size(),
            anchor_stale_threshold: default_anchor_stale_threshold(),
        }
    }
}

fn default_limit() -> usize {
    20
}

fn default_min_similarity() -> f32 {
    0.15
}

fn default_utility_lambda() -> f32 {
    0.2
}

fn default_bll_enabled() -> bool {
    true
}

fn default_bll_weights_path() -> String {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".threadbridge")
        .join("weights")
        .join("bll_v2.bin")
        .to_string_lossy()
        .to_string()
}

fn default_context_window_size() -> usize {
    2
}

fn default_anchor_stale_threshold() -> f32 {
    0.3 // grounding score below 30% = stale
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding: EmbeddingConfig::default(),
            search: SearchConfig::default(),
            verification: VerificationConfig::default(),
        }
    }
}

impl Config {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".threadbridge")
            .join("config.toml")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        if !path.exists() {
            return Self::default();
        }

        match std::fs::read_to_string(&path) {
            Ok(content) if content.trim().is_empty() => {
                warn!("Config file is empty: {:?}, regenerating defaults", path);
                let _ = std::fs::remove_file(&path);
                return Self::default();
            }
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => {
                    debug!("Loaded config from {:?}", path);
                    return config;
                }
                Err(e) => {
                    warn!("Failed to parse config {:?}: {}. Using defaults. Fix the file or delete it to regenerate.", path, e);
                }
            },
            Err(e) => {
                warn!("Failed to read config {:?}: {}. Using defaults.", path, e);
            }
        }
        Self::default()
    }

    pub fn global() -> &'static Config {
        CONFIG.get_or_init(|| {
            let config = Self::load();
            info!("Config: BLL={}, threshold={}, lambda={}, context_window=±{}, verification=[prov={},rev={},cusum={},budget={},sheaf={}]",
                config.search.bll_enabled,
                config.search.min_similarity,
                config.search.utility_lambda,
                config.search.context_window_size,
                config.verification.provenance_enabled,
                config.verification.revision_enabled,
                config.verification.cusum_enabled,
                config.verification.budget_enabled,
                config.verification.sheaf_enabled,
            );
            config
        })
    }

    pub fn create_default_if_missing() -> bool {
        let path = Self::config_path();
        if path.exists() {
            return false;
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let config = Self::default();
        let content = match toml::to_string_pretty(&config) {
            Ok(c) => c,
            Err(_) => return false,
        };
        match crate::util::atomic_write(&path, content.as_bytes()) {
            Ok(()) => {
                info!("Created default config at {:?}", path);
                true
            }
            Err(_) => false,
        }
    }
}
