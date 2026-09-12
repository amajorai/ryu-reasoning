//! The sidecar's one line back into Core.
//!
//! Formalizing prose is a language task, so the two edges of the pipeline need a
//! model. The sidecar does not hold provider keys and must not egress on its own;
//! instead it calls Core's generic sidecar callback:
//!
//! ```text
//! POST http://127.0.0.1:$RYU_CORE_PORT/api/host/model/complete
//!   authorization: Bearer $RYU_EXT_TOKEN
//!   x-ryu-plugin-id: $RYU_EXT_PLUGIN_ID
//!   { "prompt": …, "system": …, "model_pref_key": … }
//! ```
//!
//! Core authenticates the minted per-plugin token, intersects the manifest's
//! declared `host_api.grants` with the Gateway-*approved* grants, and only then runs
//! the completion through the same `host.sideModel` capability the turn-hook sandbox
//! uses. So the app inherits the node's provider routing, its budget, and its egress
//! policy, and holds no credential of its own.
//!
//! Both halves of the grant matter: `hook:side-model` must appear in the sidecar's
//! `host_api.grants` **and** be approved for the plugin. Missing either side is a
//! 403, which surfaces here as a plain error rather than a silent empty answer.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

/// Env keys Core injects into every manifest sidecar at spawn.
const ENV_TOKEN: &str = "RYU_EXT_TOKEN";
const ENV_PLUGIN_ID: &str = "RYU_EXT_PLUGIN_ID";
const ENV_CORE_PORT: &str = "RYU_CORE_PORT";

/// Completions can involve a long document; give them room but never forever.
const TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone)]
pub struct Host {
    base: String,
    plugin_id: String,
    token: String,
    http: reqwest::Client,
}

impl Host {
    /// Build from the injected environment. `None` when the process was not spawned
    /// by Core — the solver-only routes still work, the model-backed ones report
    /// that they are unavailable rather than pretending.
    pub fn from_env() -> Option<Host> {
        let token = std::env::var(ENV_TOKEN).ok().filter(|s| !s.is_empty())?;
        let plugin_id = std::env::var(ENV_PLUGIN_ID)
            .ok()
            .filter(|s| !s.is_empty())?;
        let port = std::env::var(ENV_CORE_PORT)
            .ok()
            .and_then(|p| p.parse::<u16>().ok())?;
        Some(Host {
            base: format!("http://127.0.0.1:{port}"),
            plugin_id,
            token,
            http: reqwest::Client::builder().timeout(TIMEOUT).build().ok()?,
        })
    }

    /// One completion. `model_pref_key` names a settings key the user can point at a
    /// specific model (the `@ryu/double-check` pattern), so policy extraction can run
    /// on a stronger model than the chat itself.
    pub async fn complete(
        &self,
        system: &str,
        prompt: &str,
        model_pref_key: Option<&str>,
    ) -> Result<String> {
        let mut args = json!({ "system": system, "prompt": prompt });
        if let Some(key) = model_pref_key {
            args["model_pref_key"] = Value::String(key.to_owned());
        }
        let resp = self
            .http
            .post(format!("{}/api/host/model/complete", self.base))
            .bearer_auth(&self.token)
            .header("x-ryu-plugin-id", &self.plugin_id)
            .json(&args)
            .send()
            .await
            .context("calling the host model callback")?;

        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .context("host model callback returned a non-JSON body")?;
        if !status.is_success() {
            let msg = body
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("host model callback failed");
            // 403 here almost always means the grant is declared but not approved.
            return Err(anyhow!("{msg} (HTTP {status})"));
        }
        let text = body.get("result").map(render_result).unwrap_or_default();
        if text.trim().is_empty() {
            return Err(anyhow!("the model returned an empty completion"));
        }
        Ok(text)
    }
}

/// The bridge returns either a bare string or a `{ text }`-shaped object depending on
/// the provider; accept both rather than depending on one provider's shape.
fn render_result(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(map) => map
            .get("text")
            .or_else(|| map.get("content"))
            .or_else(|| map.get("output"))
            .map(render_result)
            .unwrap_or_default(),
        Value::Array(items) => items.iter().map(render_result).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// Pull the first JSON object or array out of a completion.
///
/// Models wrap JSON in prose or a ```json fence often enough that retrying on a
/// strict parse would waste a call per request. Scanning for the first balanced
/// brace — string- and escape-aware, so a `{` inside a quoted rule statement does not
/// throw off the depth count — recovers the payload without a second round trip.
pub fn extract_json(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    for (start, open) in bytes.iter().enumerate() {
        let close = match open {
            b'{' => b'}',
            b'[' => b']',
            _ => continue,
        };
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (idx, byte) in bytes.iter().enumerate().skip(start) {
            if in_string {
                if escaped {
                    escaped = false;
                } else if *byte == b'\\' {
                    escaped = true;
                } else if *byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match *byte {
                b'"' => in_string = true,
                b if b == *open => depth += 1,
                b if b == close => {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(parsed) = serde_json::from_str::<Value>(&text[start..=idx]) {
                            return Some(parsed);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_recovered_from_a_fenced_block() {
        let text = "Sure!\n```json\n{\"variables\": []}\n```\nHope that helps.";
        assert_eq!(extract_json(text), Some(json!({ "variables": [] })));
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_the_scan() {
        let text = r#"prefix {"statement": "use {curly} braces", "ok": true} suffix"#;
        let parsed = extract_json(text).expect("parsed");
        assert_eq!(parsed["statement"], "use {curly} braces");
    }

    #[test]
    fn prose_without_json_yields_nothing() {
        assert_eq!(extract_json("I could not do that."), None);
    }

    #[test]
    fn result_shapes_all_render_to_text() {
        assert_eq!(render_result(&json!("hi")), "hi");
        assert_eq!(render_result(&json!({ "text": "hi" })), "hi");
        assert_eq!(
            render_result(&json!([{ "text": "a" }, { "text": "b" }])),
            "ab"
        );
    }
}
