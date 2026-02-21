//! Local Embedding using fastembed
//!
//! This module provides vector embeddings using the fastembed library,
//! which runs entirely locally without any API calls.
//!
//! ## Model: Snowflake Arctic Embed M (768-dim)
//!
//! Upgraded from AllMiniLM-L6-v2 (384-dim) for significantly better retrieval quality.
//! Arctic Embed is purpose-built for retrieval tasks and ranks near the top of MTEB.
//!
//! ## Embedding Cache
//!
//! Two-tier cache for deterministic embeddings:
//! - **L1 (memory)**: HashMap<MD5, Vec<f32>> for instant lookup
//! - **L2 (disk)**: Append-only binary file at ~/.threadbridge/embedding_cache.bin
//!   Format per entry: [md5:16B][f32×768:3072B] = 3088B
//!
//! On startup the disk cache is loaded into memory. New embeddings are appended
//! to the file and inserted into the HashMap. This makes repeated benchmark runs
//! skip the model entirely after the first run.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::collections::HashMap;
use std::io::{Write as IoWrite, BufWriter};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tracing::{debug, info, warn};

/// Current embedding dimension (Arctic Embed M = 768)
pub const EMBEDDING_DIM: usize = 768;

/// Bytes per embedding entry on disk: 16 (MD5) + EMBEDDING_DIM * 4 (f32)
const ENTRY_BYTES: usize = 16 + EMBEDDING_DIM * 4;

/// Global embedding model instance (lazy loaded)
static EMBEDDING_MODEL: OnceLock<Mutex<Option<TextEmbedding>>> = OnceLock::new();

/// Global embedding cache: MD5(text) -> embedding vector
/// Populated from disk on first access, updated in-memory + appended to disk on misses.
static EMBEDDING_CACHE: OnceLock<Mutex<EmbeddingCache>> = OnceLock::new();

struct EmbeddingCache {
    map: HashMap<[u8; 16], Vec<f32>>,
    /// Open file handle for appending new entries (None if disk cache failed to open)
    writer: Option<BufWriter<std::fs::File>>,
}

fn disk_cache_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".threadbridge")
        .join("embedding_cache.bin")
}

/// Load all entries from the disk cache file into a HashMap.
fn load_disk_cache(path: &PathBuf) -> HashMap<[u8; 16], Vec<f32>> {
    let mut map = HashMap::new();

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return map, // File doesn't exist yet — normal on first run
    };

    if data.len() < ENTRY_BYTES {
        if !data.is_empty() {
            warn!("Embedding cache file too small ({} bytes), ignoring", data.len());
        }
        return map;
    }

    let entry_count = data.len() / ENTRY_BYTES;
    let usable = entry_count * ENTRY_BYTES;
    if usable < data.len() {
        warn!(
            "Embedding cache has {} trailing bytes (truncated write?), ignoring them",
            data.len() - usable
        );
    }

    map.reserve(entry_count);
    for i in 0..entry_count {
        let offset = i * ENTRY_BYTES;
        let mut key = [0u8; 16];
        key.copy_from_slice(&data[offset..offset + 16]);

        let float_bytes = &data[offset + 16..offset + ENTRY_BYTES];
        let emb: Vec<f32> = float_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        map.insert(key, emb);
    }

    info!(
        "Loaded {} cached embeddings from disk ({:.1} MB)",
        map.len(),
        data.len() as f64 / (1024.0 * 1024.0)
    );
    map
}

fn get_cache() -> &'static Mutex<EmbeddingCache> {
    EMBEDDING_CACHE.get_or_init(|| {
        let path = disk_cache_path();
        let map = load_disk_cache(&path);

        // Open file for appending new entries
        let writer = std::fs::create_dir_all(path.parent().unwrap())
            .ok()
            .and_then(|_| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .ok()
            })
            .map(BufWriter::new);

        if writer.is_none() {
            warn!("Could not open embedding cache file for writing: {:?}", path);
        }

        Mutex::new(EmbeddingCache { map, writer })
    })
}

/// Compute MD5 hash of text, used as cache key.
#[inline]
fn text_key(text: &str) -> [u8; 16] {
    md5::compute(text.as_bytes()).0
}

/// Append a single entry to disk. Caller must hold the cache lock.
fn append_to_disk(cache: &mut EmbeddingCache, key: &[u8; 16], emb: &[f32]) {
    if let Some(ref mut writer) = cache.writer {
        let mut buf = Vec::with_capacity(ENTRY_BYTES);
        buf.extend_from_slice(key);
        for &v in emb {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if writer.write_all(&buf).is_err() {
            warn!("Failed to append to embedding cache file");
        }
        // Don't flush on every write — BufWriter handles batching.
        // We flush explicitly after batch operations.
    }
}

/// Get global cache directory for embedding models
/// Uses ~/.threadbridge/fastembed_cache so all projects share the same model files
fn get_global_cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".threadbridge")
        .join("fastembed_cache")
}

/// Get or initialize the embedding model
fn get_model() -> Result<&'static Mutex<Option<TextEmbedding>>> {
    let mutex = EMBEDDING_MODEL.get_or_init(|| Mutex::new(None));

    // Check if model is initialized
    {
        let guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_some() {
            return Ok(mutex);
        }
    }

    // Initialize the model with global cache directory
    let cache_dir = get_global_cache_dir();
    info!("Initializing Snowflake Arctic Embed M (768-dim, cache: {:?})...", cache_dir);

    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::SnowflakeArcticEmbedM)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(true)
    ).context("Failed to initialize embedding model")?;

    info!("Embedding model initialized successfully (Arctic Embed M, 768-dim)");

    {
        let mut guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(model);
    }

    Ok(mutex)
}

/// Embedding service for generating vector representations of text
pub struct EmbeddingService;

/// Query prefix for asymmetric retrieval models (Arctic Embed)
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

impl EmbeddingService {
    /// Generate embedding for a query (adds retrieval prefix for asymmetric models)
    pub fn embed_query(text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{}", QUERY_PREFIX, text);
        Self::embed_raw(&prefixed)
    }

    /// Generate embeddings for multiple passages (no prefix)
    pub fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let cache_mutex = get_cache();
        let cache = cache_mutex.lock().unwrap_or_else(|e| e.into_inner());

        // Partition into cached hits and misses (preserving original order)
        let mut results = vec![None; texts.len()];
        let mut miss_indices = Vec::new();
        let mut miss_texts = Vec::new();
        let mut miss_keys = Vec::new();

        for (i, &text) in texts.iter().enumerate() {
            let key = text_key(text);
            if let Some(emb) = cache.map.get(&key) {
                results[i] = Some(emb.clone());
            } else {
                miss_indices.push(i);
                miss_texts.push(text);
                miss_keys.push(key);
            }
        }

        let hits = texts.len() - miss_texts.len();
        if hits > 0 {
            debug!("Embedding cache: {}/{} hits ({} misses)", hits, texts.len(), miss_texts.len());
        }

        // Compute only the misses
        if !miss_texts.is_empty() {
            // Drop cache lock before acquiring model lock to avoid potential deadlock
            drop(cache);

            let mutex = get_model()?;
            let mut model_guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
            let model = model_guard.as_mut().context("Model not initialized")?;

            let new_embeddings = model.embed(miss_texts.clone(), None)
                .context("Failed to generate embeddings")?;

            // Re-acquire cache lock to store new results
            let mut cache = cache_mutex.lock().unwrap_or_else(|e| e.into_inner());
            for (j, emb) in new_embeddings.into_iter().enumerate() {
                let orig_idx = miss_indices[j];
                let key = &miss_keys[j];

                // Write to disk
                append_to_disk(&mut cache, key, &emb);

                // Insert into memory cache
                cache.map.insert(*key, emb.clone());
                results[orig_idx] = Some(emb);
            }

            // Flush disk writes after batch
            if let Some(ref mut writer) = cache.writer {
                let _ = writer.flush();
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Raw embed without any prefix
    fn embed_raw(text: &str) -> Result<Vec<f32>> {
        let key = text_key(text);

        // Check cache first
        {
            let cache = get_cache();
            let cache_guard = cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(emb) = cache_guard.map.get(&key) {
                return Ok(emb.clone());
            }
        }

        let mutex = get_model()?;
        let mut guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
        let model = guard.as_mut().context("Model not initialized")?;

        let embeddings = model.embed(vec![text], None)
            .context("Failed to generate embedding")?;

        let emb = embeddings.into_iter().next()
            .context("No embedding generated")?;

        // Store in memory + disk cache
        {
            let cache_mutex = get_cache();
            let mut cache = cache_mutex.lock().unwrap_or_else(|e| e.into_inner());
            append_to_disk(&mut cache, &key, &emb);
            if let Some(ref mut writer) = cache.writer {
                let _ = writer.flush();
            }
            cache.map.insert(key, emb.clone());
        }

        Ok(emb)
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_text_key_deterministic() {
        let k1 = text_key("hello world");
        let k2 = text_key("hello world");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_text_key_different() {
        let k1 = text_key("hello");
        let k2 = text_key("world");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_entry_bytes_size() {
        assert_eq!(ENTRY_BYTES, 16 + 768 * 4);
    }

    #[test]
    fn test_disk_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_cache.bin");

        // Write a few entries
        let key1 = text_key("test text one");
        let emb1: Vec<f32> = (0..EMBEDDING_DIM).map(|i| i as f32 * 0.001).collect();
        let key2 = text_key("test text two");
        let emb2: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (i as f32 + 100.0) * 0.001).collect();

        {
            let mut f = std::fs::File::create(&path).unwrap();
            let mut buf = Vec::with_capacity(ENTRY_BYTES);
            buf.extend_from_slice(&key1);
            for &v in &emb1 { buf.extend_from_slice(&v.to_le_bytes()); }
            f.write_all(&buf).unwrap();

            buf.clear();
            buf.extend_from_slice(&key2);
            for &v in &emb2 { buf.extend_from_slice(&v.to_le_bytes()); }
            f.write_all(&buf).unwrap();
        }

        let loaded = load_disk_cache(&path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get(&key1).unwrap(), &emb1);
        assert_eq!(loaded.get(&key2).unwrap(), &emb2);
    }
}
