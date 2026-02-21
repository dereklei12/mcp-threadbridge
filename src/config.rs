//! Configuration for mcp-threadbridge
//!
//! TOML config at ~/.threadbridge/config.toml
//! All fields have sensible defaults matching Config I (benchmark optimal).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;
use tracing::{debug, info};

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub search: SearchConfig,
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
        .unwrap_or_default()
        .join(".threadbridge")
        .join("weights")
        .join("bll_v2.bin")
        .to_string_lossy()
        .to_string()
}

fn default_context_window_size() -> usize {
    2
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding: EmbeddingConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl Config {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_default()
            .join(".threadbridge")
            .join("config.toml")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(config) => {
                        debug!("Loaded config from {:?}", path);
                        return config;
                    }
                    Err(e) => {
                        debug!("Failed to parse config: {}, using defaults", e);
                    }
                },
                Err(e) => {
                    debug!("Failed to read config: {}, using defaults", e);
                }
            }
        }
        Self::default()
    }

    pub fn global() -> &'static Config {
        CONFIG.get_or_init(|| {
            let config = Self::load();
            info!("Config: BLL={}, threshold={}, lambda={}, context_window=±{}",
                config.search.bll_enabled,
                config.search.min_similarity,
                config.search.utility_lambda,
                config.search.context_window_size,
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
        if let Ok(content) = toml::to_string_pretty(&config) {
            if std::fs::write(&path, content).is_ok() {
                info!("Created default config at {:?}", path);
                return true;
            }
        }
        false
    }
}
