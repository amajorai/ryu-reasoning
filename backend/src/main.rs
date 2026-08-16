//! `ryu-reasoning` — the out-of-process automated-reasoning sidecar.
//!
//! Core spawns it (`kind: local`, sibling on `PATH` or `RYU_REASONING_BIN`),
//! health-checks it, and proxies `/api/reasoning/*` to it on loopback, exactly like
//! `ryu-monitors` / `ryu-simulator`. The engine, the policy store, and the handlers
//! live in the crate lib; this binary is only the process shell around them.
//!
//! SECURITY: loopback-only bind (127.0.0.1) plus a shared-secret bearer gate
//! (`RYU_EXT_TOKEN`, injected by Core at spawn and re-stamped on every proxied hop).
//! Every `/api/reasoning/*` route is protected and the gate is FAIL-CLOSED: with no
//! token configured, every protected route rejects with 401. `/health` is the one
//! un-gated route so Core's pre-auth probe succeeds; it returns no policy data.
//!
//! Port: `RYU_REASONING_PORT`, default 8006. Data: `$RYU_DIR/reasoning`, so policies
//! land under the same node directory Core uses.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::extract::Request;
use axum::http::{header::AUTHORIZATION, StatusCode};
use axum::middleware::{from_fn, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;

use ryu_reasoning::api::{capability_descriptor, routes, Ctx};
use ryu_reasoning::host::Host;
use ryu_reasoning::solver::Budget;
use ryu_reasoning::store::{data_dir, Store};

/// Default loopback port, kept identical to `apps-store/reasoning/manifest.json`.
const DEFAULT_PORT: u16 = 8006;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // stderr, never stdout: in `mcp` mode stdout carries the JSON-RPC stream, and a
    // log line written into it desynchronizes the framing on the client side.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("RYU_REASONING_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let token = std::env::var("RYU_EXT_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    if token.is_none() {
        tracing::warn!(
            "ryu-reasoning: no RYU_EXT_TOKEN set; every /api/reasoning/* route is FAIL-CLOSED \
             (401) until Core spawns this sidecar with one"
        );
    }

    let host = Host::from_env();
    if host.is_none() {
        tracing::warn!(
            "ryu-reasoning: no host model callback in the environment; policy drafting and \
             prose checking will report 503, while /solve and policy analysis still work"
        );
    }

    let store = Store::open(data_dir())?;
    let ctx = Arc::new(Ctx {
        store,
        host,
        budget: Budget::default(),
    });

    // `ryu-reasoning mcp` speaks MCP on stdio instead of serving HTTP — the same
    // engine and the same policy store, reached the way an agent or a workflow node
    // reaches any other tool server. Core spawns this form from the manifest's
    // `mcp_servers` block; nothing binds a port in this mode.
    if std::env::args().nth(1).as_deref() == Some("mcp") {
        return ryu_reasoning::mcp::serve(ctx).await;
    }

    // `/openapi.json` rides INSIDE the bearer gate but OUTSIDE the `/api/reasoning`
    // nest, at the server ROOT: Core fetches `http://127.0.0.1:<port>/openapi.json` on
    // this sidecar's first Healthy edge and derives one LLM tool per operation, and
    // root is the only address it tries. Gated rather than sitting next to `/health`
    // because the document enumerates every route and body field this app accepts.
    let protected = Router::new()
        .nest("/api/reasoning", routes(ctx))
        .route(
            "/openapi.json",
            get(|| async { Json(ryu_reasoning::api::openapi()) }),
        )
        .layer(from_fn(move |req: Request, next: Next| {
            let expected = token.clone();
            async move { bearer_gate(expected.as_deref(), req, next).await }
        }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/capability", get(capability))
        .merge(protected);

    // LOOPBACK ONLY (belt) + shared-secret bearer (suspenders): Core is the auth
    // front and re-stamps the bearer on the proxied hop.
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "ryu-reasoning listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "ok": true, "service": "ryu-reasoning" }))
}

async fn capability() -> Json<serde_json::Value> {
    Json(capability_descriptor())
}

/// Shared-secret bearer gate. Core stamps `authorization: Bearer <RYU_EXT_TOKEN>` on
/// the loopback hop, so a request that did NOT come through Core has no way to
/// present it. Fail-closed when no token is configured.
async fn bearer_gate(expected: Option<&str>, req: Request, next: Next) -> Response {
    let provided = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if bearer_ok(provided, expected) {
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "unauthorized" })),
    )
        .into_response()
}

/// Pure bearer check, factored out so the auth decision is unit-testable without a
/// server. Constant-time comparison: the token is a secret, and a length- or
/// prefix-sensitive compare leaks it a byte at a time.
fn bearer_ok(provided: Option<&str>, expected: Option<&str>) -> bool {
    let (Some(provided), Some(expected)) = (provided, expected) else {
        return false;
    };
    if provided.len() != expected.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in provided.bytes().zip(expected.bytes()) {
        diff |= a ^ b;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gate_is_fail_closed_without_a_configured_token() {
        assert!(!bearer_ok(Some("anything"), None));
        assert!(!bearer_ok(None, None));
    }

    #[test]
    fn only_the_exact_token_passes() {
        assert!(bearer_ok(Some("s3cret"), Some("s3cret")));
        assert!(!bearer_ok(Some("s3cre"), Some("s3cret")));
        assert!(!bearer_ok(Some("s3cretx"), Some("s3cret")));
        assert!(!bearer_ok(None, Some("s3cret")));
    }
}
