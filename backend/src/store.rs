//! On-disk persistence: one JSON file per policy under `$RYU_DIR/reasoning`.
//!
//! JSON rather than a database because a policy is a document an author reviews,
//! diffs, and occasionally copies between nodes — the same reasons a skill or an
//! agent preset is a file. There is no query surface to speak of (list, get, save,
//! delete) and policies number in the dozens, so an index would cost more than it
//! saves.
//!
//! Writes are atomic: content goes to a temporary file in the same directory and is
//! then renamed over the target, so a crash mid-write leaves the previous version
//! intact rather than a half-written policy that no longer parses.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::policy::Policy;

/// Resolve the data directory, honouring the `RYU_DIR` Core injects at spawn so the
/// sidecar writes under the same node directory Core uses.
pub fn data_dir() -> PathBuf {
    let root = std::env::var("RYU_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".ryu")
        });
    root.join("reasoning")
}

#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn open(root: impl Into<PathBuf>) -> Result<Store> {
        let root = root.into();
        fs::create_dir_all(&root)
            .with_context(|| format!("creating {}", root.display()))?;
        Ok(Store { root })
    }

    /// Policy ids come in over HTTP and become path segments, so they are restricted
    /// to a charset with no separators and no dots — `..` cannot be spelled, and
    /// neither can an absolute path.
    fn path_for(&self, id: &str) -> Result<PathBuf> {
        if id.is_empty() || id.len() > 64 {
            return Err(anyhow!("policy id must be 1–64 characters"));
        }
        if !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(anyhow!(
                "policy id may only contain letters, digits, '-' and '_'"
            ));
        }
        Ok(self.root.join(format!("{id}.json")))
    }

    pub fn list(&self) -> Result<Vec<Policy>> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.root) else {
            return Ok(out);
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match read_policy(&path) {
                Ok(policy) => out.push(policy),
                // One corrupt file must not hide every other policy.
                Err(e) => tracing::warn!(path = %path.display(), error = %e, "skipping policy"),
            }
        }
        out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        Ok(out)
    }

    pub fn get(&self, id: &str) -> Result<Option<Policy>> {
        let path = self.path_for(id)?;
        if !path.exists() {
            return Ok(None);
        }
        read_policy(&path).map(Some)
    }

    /// Write the policy, bumping `version` and stamping `updated_at`.
    pub fn save(&self, mut policy: Policy) -> Result<Policy> {
        let path = self.path_for(&policy.id)?;
        let now = chrono::Utc::now().to_rfc3339();
        if let Ok(Some(existing)) = self.get(&policy.id) {
            policy.version = existing.version.saturating_add(1);
            policy.created_at = existing.created_at;
        } else {
            policy.version = policy.version.max(1);
            policy.created_at = now.clone();
        }
        policy.updated_at = now;

        let body = serde_json::to_string_pretty(&policy)?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, body).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(policy)
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let path = self.path_for(id)?;
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(&path)?;
        Ok(true)
    }
}

fn read_policy(path: &Path) -> Result<Policy> {
    let raw = fs::read_to_string(path)?;
    serde_json::from_str(&raw).with_context(|| format!("{} is not a valid policy", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Policy;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ryu-reasoning-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn policy(id: &str) -> Policy {
        Policy {
            id: id.into(),
            name: format!("Policy {id}"),
            description: String::new(),
            version: 1,
            variables: Vec::new(),
            rules: Vec::new(),
            tests: Vec::new(),
            source_document: None,
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn save_then_get_round_trips_and_bumps_the_version() {
        let store = Store::open(scratch("roundtrip")).unwrap();
        let saved = store.save(policy("hr")).unwrap();
        assert_eq!(saved.version, 1);
        assert!(!saved.created_at.is_empty());

        let again = store.save(policy("hr")).unwrap();
        assert_eq!(again.version, 2, "a save must bump the version");
        assert_eq!(
            again.created_at, saved.created_at,
            "the creation stamp must survive an edit"
        );

        assert_eq!(store.get("hr").unwrap().unwrap().version, 2);
    }

    #[test]
    fn traversal_ids_are_refused() {
        let store = Store::open(scratch("traversal")).unwrap();
        for bad in ["../escape", "a/b", "..", "with.dot", ""] {
            assert!(
                store.get(bad).is_err(),
                "id '{bad}' must be rejected before it reaches the filesystem"
            );
        }
    }

    #[test]
    fn listing_skips_a_corrupt_file_instead_of_failing() {
        let root = scratch("corrupt");
        let store = Store::open(&root).unwrap();
        store.save(policy("good")).unwrap();
        fs::write(root.join("broken.json"), "{ not json").unwrap();
        let all = store.list().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "good");
    }

    #[test]
    fn delete_reports_whether_anything_was_removed() {
        let store = Store::open(scratch("delete")).unwrap();
        store.save(policy("x")).unwrap();
        assert!(store.delete("x").unwrap());
        assert!(!store.delete("x").unwrap());
    }
}
