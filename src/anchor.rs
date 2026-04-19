//! Fact Anchoring: ground facts against the project codebase.
//!
//! Each fact carries anchors — references to project files where related
//! keywords were found. At save time, scan the project to generate anchors.
//! At load/search time, re-verify to detect staleness.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tracing::debug;

const SKIP_DIRS: &[&str] = &[
    ".git", ".hg", ".svn", ".threadbridge",
    "node_modules", "target", ".venv", "__pycache__",
    ".tox", "dist", "build", ".next", ".nuxt",
    "vendor", ".cargo", ".gradle", ".idea",
];

const SKIP_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "ico",
    "wasm", "bin", "exe", "dll", "so", "dylib",
    "zip", "tar", "gz", "bz2", "xz", "7z",
    "pdf", "doc", "docx", "xls", "xlsx",
    "mp3", "mp4", "avi", "mov", "wav",
    "ttf", "otf", "woff", "woff2", "eot",
    "pyc", "pyo", "class", "o", "a", "lib",
];

const MAX_FILE_SIZE: u64 = 100_000; // 100KB
const MAX_FILES: usize = 500;
const MAX_DEPTH: usize = 5;
const MAX_KEYWORDS: usize = 10;

pub const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "not", "only", "own", "same", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "that", "this", "these", "those", "its",
    "we", "they", "them", "their", "our", "your",
    "what", "which", "who", "whom",
    // Common verbs not useful as code keywords
    "use", "used", "using", "make", "made", "get", "set",
    "add", "added", "adding", "create", "created", "creating",
    "implement", "implemented", "implementing",
    "change", "changed", "changing", "update", "updated", "updating",
    "move", "moved", "moving", "remove", "removed", "removing",
    "new", "old", "also", "now", "still",
    // Fact/decision filler
    "reason", "because", "decided", "decision", "chose", "choose",
    "based", "done", "complete", "completed", "progress", "pending",
    "feature", "currently", "already",
];

/// A code anchor: a keyword found (or expected) in a specific project file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anchor {
    /// Relative path from project root. Empty if pattern was never found.
    #[serde(default)]
    pub file: String,
    /// The keyword being tracked (lowercase).
    pub pattern: String,
    /// Whether the pattern was found at last verification.
    pub found: bool,
}

// ================================================================
// Keyword extraction
// ================================================================

/// Extract code-relevant keywords from natural language text.
/// Splits on non-ASCII-alphanumeric chars (handles mixed CJK/Latin text),
/// filters stopwords, returns lowercase keywords.
pub fn extract_keywords(text: &str) -> Vec<String> {
    let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();

    let words: Vec<&str> = text
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.'))
        .filter(|w| !w.is_empty())
        .collect();

    let mut keywords = Vec::new();
    let mut seen = HashSet::new();

    for word in words {
        let lower = word.to_lowercase();
        if lower.len() < 3 {
            continue;
        }
        if stopwords.contains(lower.as_str()) {
            continue;
        }
        if lower.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if seen.insert(lower.clone()) {
            keywords.push(lower);
        }
    }

    keywords.truncate(MAX_KEYWORDS);
    keywords
}

// ================================================================
// Project scanning
// ================================================================

fn is_skip_extension(name: &str) -> bool {
    if let Some(ext) = name.rsplit('.').next() {
        SKIP_EXTENSIONS.contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}

fn scan_dir(root: &Path, dir: &Path, depth: usize, files: &mut Vec<(String, String)>) {
    if depth > MAX_DEPTH || files.len() >= MAX_FILES {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries {
        if files.len() >= MAX_FILES {
            break;
        }
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if SKIP_DIRS.contains(&name_str.as_ref()) || name_str.starts_with('.') {
                continue;
            }
            scan_dir(root, &path, depth + 1, files);
        } else {
            if is_skip_extension(&name_str) {
                continue;
            }
            if let Ok(meta) = entry.metadata() {
                if meta.len() > MAX_FILE_SIZE {
                    continue;
                }
            }
            if let Ok(content) = std::fs::read_to_string(&path) {
                let relative = path
                    .strip_prefix(root)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| name_str.to_string());
                files.push((relative, content));
            }
        }
    }
}

/// Pre-scanned project files for efficient batch anchor generation.
/// Create once per save_thread call, reuse for all facts/state/architecture.
pub struct ProjectScan {
    /// (relative_path, lowercase_content) for anchor keyword matching
    pub(crate) files: Vec<(String, String)>,
    /// (relative_path, raw_content) for provenance hashing and symbol lookup
    pub(crate) files_raw: Vec<(String, String)>,
}

impl ProjectScan {
    /// Scan project directory, collecting text file contents.
    pub fn new(project_path: &str) -> Self {
        let root = Path::new(project_path);
        let mut raw_files = Vec::new();
        if root.is_dir() {
            scan_dir(root, root, 0, &mut raw_files);
        }
        let files_raw = raw_files;
        let files: Vec<(String, String)> = files_raw
            .iter()
            .map(|(path, content)| (path.clone(), content.to_lowercase()))
            .collect();
        debug!("ProjectScan: scanned project, {} files indexed", files.len());
        Self { files, files_raw }
    }

    /// Generate anchors for a text using the pre-scanned files.
    pub fn generate_anchors(&self, text: &str) -> Vec<Anchor> {
        let keywords = extract_keywords(text);
        if keywords.is_empty() {
            return Vec::new();
        }

        let mut keyword_file: HashMap<&str, String> = HashMap::new();

        for (path, content_lower) in &self.files {
            for kw in &keywords {
                if keyword_file.contains_key(kw.as_str()) {
                    continue;
                }
                if content_lower.contains(kw.as_str()) {
                    keyword_file.insert(kw.as_str(), path.clone());
                }
            }
            if keyword_file.len() == keywords.len() {
                break;
            }
        }

        let found_count = keyword_file.len();
        let anchors = keywords
            .iter()
            .map(|kw| {
                if let Some(file) = keyword_file.get(kw.as_str()) {
                    Anchor {
                        file: file.clone(),
                        pattern: kw.clone(),
                        found: true,
                    }
                } else {
                    Anchor {
                        file: String::new(),
                        pattern: kw.clone(),
                        found: false,
                    }
                }
            })
            .collect();

        debug!(
            "Anchors: {} keywords, {} grounded",
            keywords.len(),
            found_count
        );
        anchors
    }
}

// ================================================================
// Anchor verification
// ================================================================

/// Re-verify existing anchors by checking if patterns still exist in referenced files.
pub fn verify_anchors(project_path: &str, anchors: &[Anchor]) -> Vec<Anchor> {
    anchors
        .iter()
        .map(|anchor| {
            if anchor.file.is_empty() {
                return Anchor {
                    found: false,
                    ..anchor.clone()
                };
            }
            let file_path = match crate::util::safe_join(project_path, &anchor.file) {
                Some(p) => p,
                None => {
                    return Anchor { found: false, ..anchor.clone() };
                }
            };
            let found = match std::fs::read_to_string(&file_path) {
                Ok(content) => content.to_lowercase().contains(&anchor.pattern),
                Err(_) => false,
            };
            Anchor {
                found,
                ..anchor.clone()
            }
        })
        .collect()
}

/// Calculate grounding score: fraction of anchors that are valid.
/// Returns 1.0 if no anchors (ungroundable, assume valid).
pub fn grounding_score(anchors: &[Anchor]) -> f32 {
    if anchors.is_empty() {
        return 1.0;
    }
    let valid = anchors.iter().filter(|a| a.found).count();
    valid as f32 / anchors.len() as f32
}

/// Check if any anchor is broken.
pub fn any_broken(anchors: &[Anchor]) -> bool {
    !anchors.is_empty() && anchors.iter().any(|a| !a.found)
}

/// Get list of broken keyword patterns.
pub fn broken_patterns(anchors: &[Anchor]) -> Vec<&str> {
    anchors
        .iter()
        .filter(|a| !a.found)
        .map(|a| a.pattern.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords_english() {
        let text = "We chose Diesel ORM for PostgreSQL database access";
        let kw = extract_keywords(text);
        assert!(kw.contains(&"diesel".to_string()));
        assert!(kw.contains(&"orm".to_string()));
        assert!(kw.contains(&"postgresql".to_string()));
        assert!(kw.contains(&"database".to_string()));
        assert!(kw.contains(&"access".to_string()));
        assert!(!kw.contains(&"for".to_string()));
        assert!(!kw.contains(&"chose".to_string()));
    }

    #[test]
    fn test_extract_keywords_mixed_cjk() {
        let text = "项目使用 Redis 做缓存层 PostgreSQL 做数据库";
        let kw = extract_keywords(text);
        assert!(kw.contains(&"redis".to_string()));
        assert!(kw.contains(&"postgresql".to_string()));
    }

    #[test]
    fn test_extract_keywords_technical() {
        let text = "Cargo.toml has fastembed dependency in src/embedding.rs";
        let kw = extract_keywords(text);
        assert!(kw.contains(&"cargo.toml".to_string()));
        assert!(kw.contains(&"fastembed".to_string()));
    }

    #[test]
    fn test_extract_keywords_max_limit() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar";
        let kw = extract_keywords(text);
        assert!(kw.len() <= MAX_KEYWORDS);
    }

    #[test]
    fn test_grounding_score() {
        assert_eq!(grounding_score(&[]), 1.0);
        assert_eq!(
            grounding_score(&[
                Anchor { file: "f.rs".into(), pattern: "foo".into(), found: true },
                Anchor { file: "f.rs".into(), pattern: "bar".into(), found: true },
            ]),
            1.0
        );
        assert_eq!(
            grounding_score(&[
                Anchor { file: "f.rs".into(), pattern: "foo".into(), found: true },
                Anchor { file: String::new(), pattern: "bar".into(), found: false },
            ]),
            0.5
        );
    }

    #[test]
    fn test_any_broken() {
        assert!(!any_broken(&[]));
        assert!(!any_broken(&[Anchor {
            file: "f.rs".into(), pattern: "foo".into(), found: true,
        }]));
        assert!(any_broken(&[
            Anchor { file: "f.rs".into(), pattern: "foo".into(), found: true },
            Anchor { file: String::new(), pattern: "bar".into(), found: false },
        ]));
    }
}
