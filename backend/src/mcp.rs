//! `ryu-reasoning mcp` — the same engine, spoken as an MCP stdio server.
//!
//! This is the seam that makes the app usable from an **agent** and from a
//! **workflow** without Core knowing anything about it. The manifest declares the
//! server under `mcp_servers`, Core spawns it like any other MCP process, and the
//! tools appear as `reasoning__solve`, `reasoning__check`, … — which is exactly the
//! `<server>__<tool>` id a workflow's `mcp` node takes. No route, no node kind, and
//! no Core edit is added for this app.
//!
//! Two of the four tools are **deterministic and offline** (`solve`, `analyze`):
//! they take formulas already written in the policy language, so a workflow can gate
//! a branch on a proof without spending a model call. `check` accepts prose and
//! therefore needs the host model callback; when this process was not spawned with
//! one it says so rather than returning an empty result that reads like a pass.
//!
//! Framing is newline-delimited JSON-RPC 2.0 over stdin/stdout, protocol
//! `2024-11-05` — what Core's MCP client speaks.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::api::{evaluate, Ctx};
use crate::policy::analyze;
use crate::translate::{self, Extracted, Extraction};

const PROTOCOL_VERSION: &str = "2024-11-05";

/// Tool descriptors, in the shape `tools/list` returns.
fn tool_list() -> Value {
    json!([
        {
            "name": "solve",
            "description":
                "Decide claims against a saved reasoning policy WITHOUT calling a model. Claims \
                 and premises are formulas in the policy language (for example \
                 `is_manager and tenure_months >= 12 -> vacation_days <= 30`). Returns a verdict \
                 per claim — valid (the policy proves it), invalid (the policy contradicts it, \
                 with a counterexample), satisfiable (the policy does not settle it), impossible \
                 (the policy and premises conflict), or too_complex — plus the minimal set of \
                 rules responsible. Deterministic: same inputs, same answer.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "policy_id": { "type": "string", "description": "Which saved policy to check against." },
                    "premises": {
                        "type": "array", "items": { "type": "string" },
                        "description": "Facts taken as given, as policy-language formulas."
                    },
                    "claims": {
                        "type": "array", "items": { "type": "string" },
                        "description": "The assertions to decide, as policy-language formulas."
                    }
                },
                "required": ["policy_id", "claims"]
            }
        },
        {
            "name": "check",
            "description":
                "Check a natural-language answer against a saved reasoning policy. Translates the \
                 question into premises and the answer into claims, then decides each claim with \
                 the solver. Use this when you have prose; use `solve` when you already have \
                 formulas. Requires a model callback on this node.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "policy_id": { "type": "string" },
                    "question": { "type": "string", "description": "What was asked. Supplies the premises." },
                    "answer": { "type": "string", "description": "The answer to check." }
                },
                "required": ["policy_id", "answer"]
            }
        },
        {
            "name": "policies",
            "description":
                "List the saved reasoning policies with their declared variables and rules. Call \
                 this first to learn a policy's vocabulary before writing formulas for `solve`.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "analyze",
            "description":
                "Interrogate a policy itself: are its rules mutually consistent, is any rule \
                 already implied by the others, and are any declared variables unused. Run this \
                 after editing rules — an inconsistent policy makes every check `impossible`.",
            "inputSchema": {
                "type": "object",
                "properties": { "policy_id": { "type": "string" } },
                "required": ["policy_id"]
            }
        }
    ])
}

/// Serve MCP on stdin/stdout until the stream closes.
pub async fn serve(ctx: Arc<Ctx>) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(frame) = serde_json::from_str::<Value>(line) else {
            continue; // A frame we cannot parse has no id to answer on.
        };
        let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
        let id = frame.get("id").cloned();
        // Notifications carry no id and take no response.
        let Some(id) = id else { continue };

        let response = match method {
            "initialize" => json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "ryu-reasoning", "version": env!("CARGO_PKG_VERSION") }
            }),
            "ping" => json!({}),
            "tools/list" => json!({ "tools": tool_list() }),
            "tools/call" => {
                let params = frame.get("params").cloned().unwrap_or_else(|| json!({}));
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match call_tool(&ctx, name, args).await {
                    Ok(value) => tool_result(&value, false),
                    Err(e) => tool_result(&json!({ "error": e.to_string() }), true),
                }
            }
            other => {
                write_frame(
                    &mut stdout,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("unknown method '{other}'") }
                    }),
                )
                .await?;
                continue;
            }
        };
        write_frame(
            &mut stdout,
            &json!({ "jsonrpc": "2.0", "id": id, "result": response }),
        )
        .await?;
    }
    Ok(())
}

/// MCP returns tool output as content blocks; JSON goes in a text block so a client
/// that only renders text still shows something readable.
fn tool_result(value: &Value, is_error: bool) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        }],
        "isError": is_error
    })
}

async fn call_tool(ctx: &Arc<Ctx>, name: &str, args: Value) -> Result<Value> {
    let policy_id = || -> Result<String> {
        args.get("policy_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| anyhow::anyhow!("policy_id is required"))
    };
    let load = |id: &str| -> Result<crate::policy::Policy> {
        ctx.store
            .get(id)?
            .ok_or_else(|| anyhow::anyhow!("no policy with the id '{id}' — call `policies` to list them"))
    };

    match name {
        "policies" => {
            let policies = ctx.store.list()?;
            Ok(json!({ "policies": policies }))
        }
        "analyze" => {
            let policy = load(&policy_id()?)?;
            Ok(serde_json::to_value(analyze(&policy, &ctx.budget))?)
        }
        "solve" => {
            let policy = load(&policy_id()?)?;
            let strings = |key: &str| -> Vec<Extracted> {
                args.get(key)
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| Extracted {
                        statement: s.to_owned(),
                        expression: s.to_owned(),
                        alternatives: Vec::new(),
                    })
                    .collect()
            };
            let extraction = Extraction {
                premises: strings("premises"),
                claims: strings("claims"),
                notes: Vec::new(),
            };
            if extraction.claims.is_empty() {
                anyhow::bail!("`claims` must contain at least one formula");
            }
            Ok(serde_json::to_value(evaluate(
                &policy,
                &extraction,
                &ctx.budget,
            ))?)
        }
        "check" => {
            let host = ctx.host.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "this process has no model callback, so prose cannot be translated — use \
                     `solve` with formulas in the policy language instead"
                )
            })?;
            let policy = load(&policy_id()?)?;
            let question = args.get("question").and_then(Value::as_str).unwrap_or("");
            let answer = args
                .get("answer")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("answer is required"))?;
            let extraction = translate::extract(host, &policy, question, answer).await?;
            Ok(serde_json::to_value(evaluate(
                &policy,
                &extraction,
                &ctx.budget,
            ))?)
        }
        other => anyhow::bail!("unknown tool '{other}'"),
    }
}

async fn write_frame(out: &mut tokio::io::Stdout, frame: &Value) -> Result<()> {
    let mut line = serde_json::to_string(frame)?;
    line.push('\n');
    out.write_all(line.as_bytes()).await?;
    out.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_declares_a_schema_and_a_description() {
        let tools = tool_list();
        let tools = tools.as_array().expect("array");
        assert_eq!(tools.len(), 4);
        for tool in tools {
            let name = tool["name"].as_str().expect("name");
            assert!(
                tool["description"].as_str().is_some_and(|d| d.len() > 40),
                "{name} needs a description that says when to use it"
            );
            assert_eq!(
                tool["inputSchema"]["type"], "object",
                "{name} needs an object input schema"
            );
        }
    }

    #[test]
    fn the_offline_tools_are_reachable_without_a_model() {
        // `solve` and `analyze` must not require the host callback: a workflow that
        // gates on a proof should not have to spend a model call.
        let tools = tool_list();
        let names: Vec<&str> = tools
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(names.contains(&"solve"));
        assert!(names.contains(&"analyze"));
    }

    #[test]
    fn a_tool_error_is_marked_as_one() {
        let result = tool_result(&json!({ "error": "boom" }), true);
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .expect("text")
            .contains("boom"));
    }
}
