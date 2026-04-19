//! Provenance tracking: structured code dependencies replacing flat anchors.
//!
//! Each fact carries provenance — typed links to code locations (files, symbols,
//! line ranges) with content hashes for fast change detection. This replaces the
//! old keyword-grep anchor system with semantically richer, hash-verified dependencies.

use crate::anchor::{ProjectScan, Anchor, STOPWORDS};
use chrono::{DateTime, Utc};
use crate::types::{DependencyStatus, DependencyType};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use tracing::debug;

/// A single provenance dependency: a structured link from a fact to a code location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceDep {
    /// Relative path from project root
    pub file: String,
    /// Optional: function name, struct name, or other symbol
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// Optional: line range (start, end) at time of creation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_range: Option<(u32, u32)>,
    /// Why this fact depends on this code
    pub dep_type: DependencyType,
    /// Current verification status
    #[serde(default)]
    pub status: DependencyStatus,
    /// Content hash (MD5 hex, first 16 chars) of the relevant code region at creation time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// When this dependency was last verified
    #[serde(skip_serializing_if = "Option::is_none", alias = "last_verified_commit")]
    pub last_verified_at: Option<DateTime<Utc>>,
}

/// Complete provenance record for a fact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(default)]
    pub dependencies: Vec<ProvenanceDep>,
    /// Overall provenance score: fraction of dependencies that are Intact
    #[serde(default = "default_provenance_score")]
    pub score: f32,
}

fn default_provenance_score() -> f32 {
    1.0
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            dependencies: Vec::new(),
            score: 1.0,
        }
    }
}

impl Provenance {
    /// Recompute score from dependency statuses
    pub fn recompute_score(&mut self) {
        self.score = self.compute_score();
    }

    /// Compute the score without mutating
    pub fn compute_score(&self) -> f32 {
        if self.dependencies.is_empty() {
            return 1.0;
        }
        let intact = self
            .dependencies
            .iter()
            .filter(|d| d.status == DependencyStatus::Intact)
            .count();
        intact as f32 / self.dependencies.len() as f32
    }

    /// Check if any dependency is broken (Missing or Modified)
    pub fn has_broken_deps(&self) -> bool {
        self.dependencies.iter().any(|d| {
            matches!(
                d.status,
                DependencyStatus::Missing | DependencyStatus::Modified
            )
        })
    }

    /// Get list of files with broken dependencies
    pub fn broken_files(&self) -> Vec<&str> {
        self.dependencies
            .iter()
            .filter(|d| {
                matches!(
                    d.status,
                    DependencyStatus::Missing | DependencyStatus::Modified
                )
            })
            .map(|d| d.file.as_str())
            .collect()
    }

    /// Get list of broken symbol patterns (for display, like old broken_patterns)
    pub fn broken_patterns(&self) -> Vec<String> {
        self.dependencies
            .iter()
            .filter(|d| {
                matches!(
                    d.status,
                    DependencyStatus::Missing | DependencyStatus::Modified
                )
            })
            .filter_map(|d| {
                d.symbol
                    .clone()
                    .or_else(|| Some(d.file.clone()))
            })
            .collect()
    }
}

// ================================================================
// Content hashing
// ================================================================

fn compute_file_hash(content: &str) -> String {
    let digest = md5::compute(content.as_bytes());
    format!("{:x}", digest)[..16].to_string()
}

fn compute_region_hash(content: &str, start: u32, end: u32) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let start = start as usize;
    let end = (end as usize).min(lines.len().saturating_sub(1));
    if start > end || start >= lines.len() {
        return compute_file_hash(content);
    }
    let region: String = lines[start..=end].join("\n");
    let digest = md5::compute(region.as_bytes());
    format!("{:x}", digest)[..16].to_string()
}

// ================================================================
// Symbol finding
// ================================================================

/// Re-locate a symbol definition in file content by searching for its definition pattern.
/// Returns (start_line, end_line) if found. Used during verification to handle line shifts.
fn relocate_symbol_in_content(content: &str, symbol: &str) -> Option<(u32, u32)> {
    let escaped = regex::escape(symbol);
    let vis = r#"(?:pub(?:\s*\([^)]*\))?\s+)?"#;
    let qualifiers = r#"(?:(?:async|unsafe|const|extern(?:\s+"[^"]*")?)\s+)*"#;
    let keywords = r"(?:fn|struct|enum|trait|type|mod|const|static)";
    let pattern_strs = [
        format!(r"{}{}{}\s+{}", vis, qualifiers, keywords, escaped),
        format!(r"impl(?:<[^>]*>)?\s+{}", escaped),
    ];
    let patterns: Vec<Regex> = pattern_strs
        .iter()
        .filter_map(|p| Regex::new(p).ok())
        .collect();

    for (line_no, line) in content.lines().enumerate() {
        for pattern in &patterns {
            if pattern.is_match(line) {
                let end = find_block_end(content, line_no);
                return Some((line_no as u32, end as u32));
            }
        }
    }
    None
}

/// Find the end of a code block starting at `start_line`.
/// Simple heuristic: track brace depth, end when depth returns to 0,
/// or at next blank line if no braces found within 3 lines.
fn find_block_end(content: &str, start_line: usize) -> usize {
    let lines: Vec<&str> = content.lines().collect();
    let mut depth: i32 = 0;
    let mut found_brace = false;

    for i in start_line..lines.len() {
        for ch in lines[i].chars() {
            if ch == '{' {
                depth += 1;
                found_brace = true;
            } else if ch == '}' {
                depth -= 1;
            }
        }
        if found_brace && depth <= 0 {
            return i;
        }
        // If no brace found within 3 lines, use a smaller block
        if !found_brace && i > start_line + 3 {
            return i;
        }
        // Limit block size
        if i > start_line + 100 {
            return i;
        }
    }
    (lines.len() - 1).max(start_line)
}

fn is_provenance_stopword(s: &str) -> bool {
    let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();
    stopwords.contains(s)
}

// ================================================================
// Enhanced ProjectScan methods
// ================================================================

impl ProjectScan {
    /// Check if a file exists in the scan
    pub fn file_exists(&self, relative_path: &str) -> bool {
        self.files_raw.iter().any(|(p, _)| p == relative_path)
    }

    /// Get content hash of an entire file (uses raw content for accurate hashing)
    pub fn content_hash(&self, relative_path: &str) -> Option<String> {
        for (p, content) in &self.files_raw {
            if p == relative_path {
                return Some(compute_file_hash(content));
            }
        }
        None
    }

    /// Get content hash of a specific line range in a file
    pub fn region_hash(&self, relative_path: &str, start: u32, end: u32) -> Option<String> {
        for (p, content) in &self.files_raw {
            if p == relative_path {
                return Some(compute_region_hash(content, start, end));
            }
        }
        None
    }

    /// Find where a symbol is defined (Rust: fn, struct, enum, trait, type, mod, const, static, impl)
    /// Returns (file_path, (start_line, end_line)) or None.
    /// Uses raw (non-lowercased) content for accurate line matching.
    pub fn find_symbol_definition(&self, symbol: &str) -> Option<(String, (u32, u32))> {
        let escaped = regex::escape(symbol);
        // Handles: pub, pub(crate), pub(super), pub(in path), async, unsafe, const, extern "C"
        let vis = r#"(?:pub(?:\s*\([^)]*\))?\s+)?"#;
        let qualifiers = r#"(?:(?:async|unsafe|const|extern(?:\s+"[^"]*")?)\s+)*"#;
        let keywords = r"(?:fn|struct|enum|trait|type|mod|const|static)";
        let pattern_strs = [
            format!(r"{}{}{}\s+{}", vis, qualifiers, keywords, escaped),
            format!(r"impl(?:<[^>]*>)?\s+{}", escaped),
        ];
        let patterns: Vec<Regex> = pattern_strs
            .iter()
            .filter_map(|p| Regex::new(p).ok())
            .collect();

        for (path, content) in &self.files_raw {
            for (line_no, line) in content.lines().enumerate() {
                for pattern in &patterns {
                    if pattern.is_match(line) {
                        let end = find_block_end(content, line_no);
                        return Some((path.clone(), (line_no as u32, end as u32)));
                    }
                }
            }
        }
        None
    }

    /// Find the first file containing a keyword (case-insensitive, uses lowercase index)
    pub fn find_keyword_in_file(&self, keyword: &str) -> Option<String> {
        let kw_lower = keyword.to_lowercase();
        for (path, content) in &self.files {
            if content.contains(&kw_lower) {
                return Some(path.clone());
            }
        }
        None
    }
}

// ================================================================
// Provenance generation
// ================================================================

/// Generate provenance for a fact by analyzing its content against project files.
/// This replaces the old `generate_anchors()` with structured, typed dependencies.
pub fn generate_provenance(text: &str, scan: &ProjectScan) -> Provenance {
    let mut deps = Vec::new();

    // Strategy 1: Extract explicit file path references (e.g., "src/main.rs")
    let path_re = Regex::new(r"(?:^|\s)((?:[\w.-]+/)+[\w.-]+\.[\w]+)").unwrap();
    for cap in path_re.captures_iter(text) {
        let path = cap[1].to_string();
        if scan.file_exists(&path) {
            let hash = scan.content_hash(&path);
            deps.push(ProvenanceDep {
                file: path,
                symbol: None,
                line_range: None,
                dep_type: DependencyType::ReferencesPath,
                status: DependencyStatus::Intact,
                content_hash: hash,
                last_verified_at: None,
            });
        }
    }

    // Strategy 2: Extract symbol references (CamelCase types, snake_case functions)
    let symbol_re = Regex::new(r"\b([A-Z][a-zA-Z0-9]{2,}|[a-z_][a-z0-9_]{2,})\b").unwrap();
    let mut seen_symbols = HashSet::new();
    let symbols: Vec<String> = symbol_re
        .captures_iter(text)
        .map(|c| c[1].to_string())
        .filter(|s| {
            s.len() >= 3
                && !is_provenance_stopword(&s.to_lowercase())
                && seen_symbols.insert(s.clone())
        })
        .take(15)
        .collect();

    for symbol in &symbols {
        if let Some((file, line_range)) = scan.find_symbol_definition(symbol) {
            let region_hash = scan.region_hash(&file, line_range.0, line_range.1);
            deps.push(ProvenanceDep {
                file,
                symbol: Some(symbol.clone()),
                line_range: Some(line_range),
                dep_type: DependencyType::DefinesSemantic,
                status: DependencyStatus::Intact,
                content_hash: region_hash,
                last_verified_at: None,
            });
        } else if let Some(file) = scan.find_keyword_in_file(symbol) {
            deps.push(ProvenanceDep {
                file,
                symbol: Some(symbol.clone()),
                line_range: None,
                dep_type: DependencyType::KeywordAssociation,
                status: DependencyStatus::Intact,
                content_hash: None,
                last_verified_at: None,
            });
        }
    }

    // Deduplicate: prefer higher-confidence dep_types for same (file, symbol)
    // Sort by (file, symbol, dep_type) so identical (file, symbol) pairs are adjacent,
    // then dedup keeps the first (lowest dep_type = highest confidence).
    deps.sort_by(|a, b| {
        (&a.file, &a.symbol, &a.dep_type).cmp(&(&b.file, &b.symbol, &b.dep_type))
    });
    deps.dedup_by(|a, b| a.file == b.file && a.symbol == b.symbol);

    let score = if deps.is_empty() {
        1.0
    } else {
        deps.iter()
            .filter(|d| d.status == DependencyStatus::Intact)
            .count() as f32
            / deps.len() as f32
    };

    debug!(
        "Provenance: {} deps generated ({} path refs, {} symbols)",
        deps.len(),
        deps.iter()
            .filter(|d| d.dep_type == DependencyType::ReferencesPath)
            .count(),
        deps.iter()
            .filter(|d| d.dep_type != DependencyType::ReferencesPath)
            .count(),
    );

    Provenance {
        dependencies: deps,
        score,
    }
}

// ================================================================
// Provenance verification
// ================================================================

/// Verify provenance dependencies against current project state.
/// Uses content hashing for fast checks, falls back to keyword search.
pub fn verify_provenance(
    project_path: &str,
    provenance: &mut Provenance,
) {
    let now = Utc::now();
    for dep in &mut provenance.dependencies {
        let file_path = match crate::util::safe_join(project_path, &dep.file) {
            Some(p) => p,
            None => {
                dep.status = DependencyStatus::Missing;
                continue;
            }
        };

        let content = match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(_) => {
                dep.status = DependencyStatus::Missing;
                continue;
            }
        };

        if let Some(ref expected_hash) = dep.content_hash {
            // Fast path: check content hash of the relevant region
            let current_hash = if let Some((start, end)) = dep.line_range {
                compute_region_hash(&content, start, end)
            } else {
                compute_file_hash(&content)
            };

            if current_hash == *expected_hash {
                dep.status = DependencyStatus::Intact;
            } else {
                // Hash differs — try to re-locate the symbol and update line_range + hash.
                // This handles the common case where lines were inserted/deleted above
                // the tracked region, shifting line numbers without changing the symbol.
                if let Some(ref symbol) = dep.symbol {
                    if let Some((new_start, new_end)) =
                        relocate_symbol_in_content(&content, symbol)
                    {
                        let new_hash = compute_region_hash(&content, new_start, new_end);
                        if new_hash == *expected_hash {
                            // Symbol moved but content unchanged — update location, mark intact
                            dep.line_range = Some((new_start, new_end));
                            dep.status = DependencyStatus::Intact;
                        } else {
                            // Symbol found at new location but content actually changed
                            dep.line_range = Some((new_start, new_end));
                            dep.content_hash = Some(new_hash);
                            dep.status = DependencyStatus::Modified;
                        }
                    } else if content.to_lowercase().contains(&symbol.to_lowercase()) {
                        dep.status = DependencyStatus::Modified; // exists but can't resolve definition
                    } else {
                        dep.status = DependencyStatus::Missing; // symbol gone
                    }
                } else {
                    dep.status = DependencyStatus::Modified;
                }
            }
        } else {
            // No hash stored: fall back to keyword check (legacy behavior)
            if let Some(ref symbol) = dep.symbol {
                dep.status = if content.to_lowercase().contains(&symbol.to_lowercase()) {
                    DependencyStatus::Intact
                } else {
                    DependencyStatus::Missing
                };
            } else {
                dep.status = DependencyStatus::Intact; // path-only reference, file exists
            }
        }

        dep.last_verified_at = Some(now);
    }

    provenance.recompute_score();
}

// ================================================================
// Migration from old anchors
// ================================================================

/// Migrate old Anchor structs to ProvenanceDep entries.
/// Called lazily during background verification for facts with anchors but no provenance.
pub fn migrate_anchors_to_provenance(anchors: &[Anchor]) -> Vec<ProvenanceDep> {
    anchors
        .iter()
        .filter(|a| !a.file.is_empty())
        .map(|anchor| ProvenanceDep {
            file: anchor.file.clone(),
            symbol: Some(anchor.pattern.clone()),
            line_range: None,
            dep_type: DependencyType::KeywordAssociation,
            status: if anchor.found {
                DependencyStatus::Intact
            } else {
                DependencyStatus::Missing
            },
            content_hash: None,
            last_verified_at: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_file_hash() {
        let hash = compute_file_hash("hello world");
        assert_eq!(hash.len(), 16);
        // Same content = same hash
        assert_eq!(hash, compute_file_hash("hello world"));
        // Different content = different hash
        assert_ne!(hash, compute_file_hash("hello world!"));
    }

    #[test]
    fn test_compute_region_hash() {
        let content = "line0\nline1\nline2\nline3\nline4";
        let hash1 = compute_region_hash(content, 1, 3);
        let hash2 = compute_region_hash(content, 1, 3);
        assert_eq!(hash1, hash2);
        let hash3 = compute_region_hash(content, 0, 2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_find_block_end() {
        let content = "fn foo() {\n    let x = 1;\n    let y = 2;\n}\n\nfn bar() {}";
        assert_eq!(find_block_end(content, 0), 3); // fn foo() ends at line 3 (closing brace)
    }

    #[test]
    fn test_provenance_score() {
        let mut prov = Provenance::default();
        assert_eq!(prov.compute_score(), 1.0);

        prov.dependencies.push(ProvenanceDep {
            file: "a.rs".into(),
            symbol: Some("foo".into()),
            line_range: None,
            dep_type: DependencyType::KeywordAssociation,
            status: DependencyStatus::Intact,
            content_hash: None,
            last_verified_at: None,
        });
        prov.dependencies.push(ProvenanceDep {
            file: "b.rs".into(),
            symbol: Some("bar".into()),
            line_range: None,
            dep_type: DependencyType::KeywordAssociation,
            status: DependencyStatus::Missing,
            content_hash: None,
            last_verified_at: None,
        });
        assert_eq!(prov.compute_score(), 0.5);
        assert!(prov.has_broken_deps());
    }

    #[test]
    fn test_migrate_anchors() {
        let anchors = vec![
            Anchor {
                file: "src/main.rs".into(),
                pattern: "foo".into(),
                found: true,
            },
            Anchor {
                file: String::new(),
                pattern: "bar".into(),
                found: false,
            },
        ];
        let deps = migrate_anchors_to_provenance(&anchors);
        assert_eq!(deps.len(), 1); // empty file filtered out
        assert_eq!(deps[0].file, "src/main.rs");
        assert_eq!(deps[0].status, DependencyStatus::Intact);
        assert_eq!(deps[0].dep_type, DependencyType::KeywordAssociation);
    }
}
