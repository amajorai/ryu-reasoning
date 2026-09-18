//! The HTTP surface Core proxies as `/api/reasoning/*`.
//!
//! Two families of route, split by whether a model is involved:
//!
//! * **Solver-only** — `POST /solve`, `POST /policies/:id/analyze`, and all policy
//!   CRUD. Deterministic, offline, and available even when the node has no model
//!   configured. `/solve` takes premises and claims already written in the DSL, which
//!   is what a workflow node or an agent tool should call when it has structured data
//!   rather than prose.
//! * **Model-backed** — `POST /policies/draft`, `POST /check`, and
//!   `POST /policies/:id/tests/run`. These call back into Core for a completion to
//!   translate prose, then hand the result to the same solver.
//!
//! The split is deliberate: everything that decides a verdict is in the first family,
//! so a check can always be reproduced from its findings without re-running a model.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::host::Host;
use crate::logic::Expr;
// `Variable`/`VarType`/`Rule` are imported for the OpenAPI `components(schemas(...))`
// list alone — they are the transitive graph under `Policy`, and utoipa needs the name
// in scope to register it.
use crate::logic::{VarType, Variable};
use crate::parser::parse;
use crate::policy::{analyze, CompiledPolicy, Policy, PolicyAnalysis, Rule, TestCase};
use crate::solver::{Budget, Env};
use crate::store::Store;
use crate::translate::{self, Extracted, Extraction};
use crate::verdict::{check_claim, Constraint, Finding, Origin, Verdict};

pub struct Ctx {
    pub store: Store,
    /// `None` when the process was not spawned by Core: the solver routes still work.
    pub host: Option<Host>,
    pub budget: Budget,
}

pub fn routes(ctx: Arc<Ctx>) -> Router {
    Router::new()
        .route("/policies", get(list_policies).post(create_policy))
        .route("/policies/draft", post(draft_policy))
        .route("/policies/:id", get(get_policy))
        .route("/policies/:id", put(update_policy))
        .route("/policies/:id", delete(delete_policy))
        .route("/policies/:id/analyze", post(analyze_policy))
        .route("/policies/:id/tests/run", post(run_tests))
        .route("/check", post(check))
        .route("/solve", post(solve_route))
        .with_state(ctx)
}

/// The OpenAPI sub-document Core fetches from `GET /openapi.json` and lowers into one
/// LLM tool per operation.
///
/// This app ALSO ships a stdio MCP server (`manifest.mcp_servers.reasoning`) exposing
/// `solve` / `check` / `policies` / `analyze`, so four of these seven routes can reach
/// an agent by two roads. They are annotated anyway: the MCP server is a separate
/// process a node may not be running, and dropping the annotations would leave the
/// HTTP surface toolless whenever it is not. See the note on [`solve_route`].
pub fn openapi() -> utoipa::openapi::OpenApi {
    <ReasoningApiDoc as utoipa::OpenApi>::openapi()
}

/// The document itself.
///
/// `components(schemas(...))` is what makes each `request_body = T` resolve to a real
/// `#/components/schemas/T`. Without the entry the operation still carries a `$ref`
/// whose target is missing, and Core derives a write tool with ZERO visible arguments.
///
/// The rows below `SolveRequest` are the TRANSITIVE graph reachable from `Policy`
/// (`Variable` → `VarType`, `Rule`, `TestCase` → `Verdict`). They are what makes
/// create/update usable: a model handed an opaque `Policy` cannot author one, and
/// every one of these types needed its own `ToSchema` derive for the build to pass.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(
        analyze_policy,
        check,
        create_policy,
        delete_policy,
        draft_policy,
        get_policy,
        list_policies,
        run_tests,
        solve_route,
        update_policy,
    ),
    components(schemas(
        CheckRequest,
        DraftRequest,
        Policy,
        Rule,
        SolveRequest,
        TestCase,
        Variable,
        VarType,
        Verdict,
    ))
)]
struct ReasoningApiDoc;

// ── errors ───────────────────────────────────────────────────────────────────

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl ApiError {
    fn bad(msg: impl Into<String>) -> ApiError {
        ApiError(StatusCode::BAD_REQUEST, msg.into())
    }
    fn missing() -> ApiError {
        ApiError(StatusCode::NOT_FOUND, "no such policy".into())
    }
    fn internal(e: impl std::fmt::Display) -> ApiError {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
    /// Model-backed routes with no host: a clear 503 beats an empty result that
    /// reads like "nothing to report".
    fn no_host() -> ApiError {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "this node has no model callback configured, so prose cannot be translated — use \
             /solve with expressions written in the policy language instead"
                .into(),
        )
    }
}

type ApiResult<T> = Result<T, ApiError>;

// ── policy CRUD ──────────────────────────────────────────────────────────────

/// `GET /policies` — every saved policy.
#[utoipa::path(
    get,
    path = "/api/reasoning/policies",
    tag = "Reasoning",
    summary = "List the saved reasoning policies with their rules and declared variables. Start here to find the policy id and the vocabulary a check or solve call must be written in.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_policies(State(ctx): State<Arc<Ctx>>) -> ApiResult<Json<serde_json::Value>> {
    let policies = ctx.store.list().map_err(ApiError::internal)?;
    Ok(Json(json!({ "policies": policies })))
}

/// `GET /policies/:id` — one policy in full.
#[utoipa::path(
    get,
    path = "/api/reasoning/policies/{id}",
    tag = "Reasoning",
    summary = "Read one policy in full: its declared variables, its rules, and its saved test cases. Read-only.",
    params(("id" = String, Path, description = "Policy id, from GET /api/reasoning/policies")),
    responses((status = 200, description = "OK", body = Policy))
)]
async fn get_policy(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Policy>> {
    ctx.store
        .get(&id)
        .map_err(|e| ApiError::bad(e.to_string()))?
        .map(Json)
        .ok_or_else(ApiError::missing)
}

/// `POST /policies` — save a new policy.
#[utoipa::path(
    post,
    path = "/api/reasoning/policies",
    tag = "Reasoning",
    summary = "Create a new reasoning policy from declared variables and rules. Rejected if a policy with the same id already exists; leave `id` empty to have one derived from the name.",
    request_body = Policy,
    responses((status = 200, description = "Created", body = Policy))
)]
async fn create_policy(
    State(ctx): State<Arc<Ctx>>,
    Json(mut policy): Json<Policy>,
) -> ApiResult<Json<Policy>> {
    if policy.id.trim().is_empty() {
        policy.id = slug(&policy.name);
    }
    if ctx
        .store
        .get(&policy.id)
        .map_err(|e| ApiError::bad(e.to_string()))?
        .is_some()
    {
        return Err(ApiError::bad(format!(
            "a policy with the id '{}' already exists",
            policy.id
        )));
    }
    ctx.store
        .save(policy)
        .map(Json)
        .map_err(|e| ApiError::bad(e.to_string()))
}

/// `PUT /policies/:id` — replace a policy.
#[utoipa::path(
    put,
    path = "/api/reasoning/policies/{id}",
    tag = "Reasoning",
    summary = "Replace a policy wholesale with the body given. This is a full overwrite, not a merge: variables, rules, and tests omitted from the body are DELETED. Read the policy first and send it back modified.",
    params(("id" = String, Path, description = "Policy id; it overrides whatever `id` the body carries")),
    request_body = Policy,
    responses((status = 200, description = "Saved", body = Policy))
)]
async fn update_policy(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<String>,
    Json(mut policy): Json<Policy>,
) -> ApiResult<Json<Policy>> {
    policy.id = id;
    ctx.store
        .save(policy)
        .map(Json)
        .map_err(|e| ApiError::bad(e.to_string()))
}

/// `DELETE /policies/:id` — remove a policy.
#[utoipa::path(
    delete,
    path = "/api/reasoning/policies/{id}",
    tag = "Reasoning",
    summary = "PERMANENTLY delete a policy along with its rules and test cases. This cannot be undone.",
    params(("id" = String, Path, description = "Policy id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn delete_policy(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let removed = ctx
        .store
        .delete(&id)
        .map_err(|e| ApiError::bad(e.to_string()))?;
    Ok(Json(json!({ "deleted": removed })))
}

/// `POST /policies/:id/analyze` — is the policy consistent with itself?
// No `request_body`: the handler takes only `State` + `Path`. Declaring one would
// invent an argument a model then tries to fill.
#[utoipa::path(
    post,
    path = "/api/reasoning/policies/{id}/analyze",
    tag = "Reasoning",
    summary = "Check a policy against ITSELF and report contradictory, redundant, or unreachable rules. Deterministic and offline — no model is called. Use it before trusting a verdict the policy produced.",
    params(("id" = String, Path, description = "Policy id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn analyze_policy(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<String>,
) -> ApiResult<Json<PolicyAnalysis>> {
    let policy = ctx
        .store
        .get(&id)
        .map_err(|e| ApiError::bad(e.to_string()))?
        .ok_or_else(ApiError::missing)?;
    Ok(Json(analyze(&policy, &ctx.budget)))
}

/// A URL- and filename-safe id derived from the policy name.
fn slug(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && out.len() < 48 {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        format!("policy-{}", uuid::Uuid::new_v4().simple())
    } else {
        trimmed
    }
}

// ── drafting ─────────────────────────────────────────────────────────────────

/// The prose a policy is drafted from.
#[derive(Deserialize, utoipa::ToSchema)]
struct DraftRequest {
    /// The full text of the rules to formalize — a policy document, a contract
    /// clause, a set of eligibility criteria. Send the prose as written; the model
    /// proposes variables and rules from it.
    document: String,
}

/// `POST /policies/draft` — turn prose into a proposed policy.
#[utoipa::path(
    post,
    path = "/api/reasoning/policies/draft",
    tag = "Reasoning",
    summary = "Turn a prose document into a PROPOSED set of variables and rules, with parse errors already attached. Nothing is saved — pass the result to POST /api/reasoning/policies to keep it. Calls a model, so it needs one configured.",
    request_body = DraftRequest,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn draft_policy(
    State(ctx): State<Arc<Ctx>>,
    Json(req): Json<DraftRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let host = ctx.host.as_ref().ok_or_else(ApiError::no_host)?;
    let draft = translate::draft_policy(host, &req.document)
        .await
        .map_err(|e| ApiError::bad(e.to_string()))?;

    // Compile the draft immediately so the editor opens with the parse errors
    // already attached, rather than the author discovering them one save later.
    let probe = Policy {
        id: "draft".into(),
        name: "draft".into(),
        description: String::new(),
        version: 1,
        variables: draft.variables.clone(),
        rules: draft.rules.clone(),
        tests: Vec::new(),
        source_document: None,
        created_at: String::new(),
        updated_at: String::new(),
    };
    let compiled = CompiledPolicy::compile(&probe);
    Ok(Json(json!({
        "variables": draft.variables,
        "rules": draft.rules,
        "notes": draft.notes,
        "errors": compiled.errors,
    })))
}

// ── checking ─────────────────────────────────────────────────────────────────

/// A prose answer to be judged against a policy.
#[derive(Deserialize, utoipa::ToSchema)]
struct CheckRequest {
    /// Id of the policy to judge against, from `GET /api/reasoning/policies`.
    policy_id: String,
    /// The question that was asked, in prose. Optional, but it is where the facts of
    /// the situation usually live — without it the check has no premises to reason
    /// from and will more often answer `satisfiable`.
    #[serde(default)]
    question: String,
    /// The answer to judge, in prose. A model translates it into the policy language
    /// before the solver decides it.
    answer: String,
}

/// The result of checking one answer.
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckReport {
    pub policy_id: String,
    pub policy_version: u32,
    /// The aggregate over every finding. See [`aggregate`].
    pub result: Verdict,
    pub findings: Vec<Finding>,
    /// The facts taken from the question, echoed so a reader can see what the check
    /// assumed.
    pub premises: Vec<PremiseView>,
    /// Sentences skipped, rules that did not compile, and other gaps.
    pub notes: Vec<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremiseView {
    pub statement: String,
    pub expression: String,
}

/// `POST /check` — judge a prose answer against a policy.
#[utoipa::path(
    post,
    path = "/api/reasoning/check",
    tag = "Reasoning",
    summary = "Judge a prose answer against a policy and return a verdict (valid / invalid / satisfiable / impossible) with the findings behind it. A model translates the prose first, so the answer is not fully deterministic — use /api/reasoning/solve when you already have structured facts.",
    request_body = CheckRequest,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn check(
    State(ctx): State<Arc<Ctx>>,
    Json(req): Json<CheckRequest>,
) -> ApiResult<Json<CheckReport>> {
    let host = ctx.host.as_ref().ok_or_else(ApiError::no_host)?;
    let policy = ctx
        .store
        .get(&req.policy_id)
        .map_err(|e| ApiError::bad(e.to_string()))?
        .ok_or_else(ApiError::missing)?;
    let extraction = translate::extract(host, &policy, &req.question, &req.answer)
        .await
        .map_err(|e| ApiError::bad(e.to_string()))?;
    Ok(Json(evaluate(&policy, &extraction, &ctx.budget)))
}

/// Facts and claims already written in the policy language.
#[derive(Deserialize, utoipa::ToSchema)]
struct SolveRequest {
    /// Id of the policy to decide against, from `GET /api/reasoning/policies`.
    policy_id: String,
    /// The facts of the situation, each an expression in the policy language over the
    /// variables that policy declares (read them from the policy first). These are
    /// taken as true.
    #[serde(default)]
    premises: Vec<String>,
    /// The claims to decide, each an expression in the same language. Each one gets
    /// its own verdict.
    claims: Vec<String>,
}

/// The model-free entry point: premises and claims arrive as DSL, so nothing is
/// translated and the result depends only on the policy.
//
// This operation is ALSO exposed as the `solve` tool of the stdio MCP server this app
// ships. Keeping both is deliberate rather than redundant: the MCP server is a
// separate process a node may not be running, and without this annotation the HTTP
// route would then be unreachable by any agent. The duplication is a followup to
// resolve at the app level, not something to fix by deleting an annotation.
#[utoipa::path(
    post,
    path = "/api/reasoning/solve",
    tag = "Reasoning",
    summary = "Decide claims against a policy with NO model call at all — fully deterministic, so the same inputs always give the same verdict. Premises and claims must already be written in the policy language; use /api/reasoning/check when all you have is prose.",
    request_body = SolveRequest,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn solve_route(
    State(ctx): State<Arc<Ctx>>,
    Json(req): Json<SolveRequest>,
) -> ApiResult<Json<CheckReport>> {
    let policy = ctx
        .store
        .get(&req.policy_id)
        .map_err(|e| ApiError::bad(e.to_string()))?
        .ok_or_else(ApiError::missing)?;
    let extraction = Extraction {
        premises: req
            .premises
            .into_iter()
            .map(|expression| Extracted {
                statement: expression.clone(),
                expression,
                alternatives: Vec::new(),
            })
            .collect(),
        claims: req
            .claims
            .into_iter()
            .map(|expression| Extracted {
                statement: expression.clone(),
                expression,
                alternatives: Vec::new(),
            })
            .collect(),
        notes: Vec::new(),
    };
    Ok(Json(evaluate(&policy, &extraction, &ctx.budget)))
}

/// Decide every claim in an extraction against a policy.
///
/// Pure: the same policy and extraction always produce the same report, which is what
/// makes a finding reproducible and the test suite meaningful.
pub fn evaluate(policy: &Policy, extraction: &Extraction, budget: &Budget) -> CheckReport {
    let started = Instant::now();
    let compiled = CompiledPolicy::compile(policy);
    let env = compiled.env.clone();
    let mut notes: Vec<String> = extraction.notes.clone();

    for err in &compiled.errors {
        notes.push(format!(
            "Rule '{}' was skipped because it does not compile: {}",
            err.rule_id, err.message
        ));
    }

    // Premises constrain the check but are never checked themselves.
    let mut constraints = compiled.constraints.clone();
    let mut premise_views = Vec::new();
    for (idx, premise) in extraction.premises.iter().enumerate() {
        match parse(&premise.expression, &env) {
            Ok(expr) => {
                constraints.push(Constraint {
                    id: format!("fact-{}", idx + 1),
                    statement: premise.statement.clone(),
                    expression: premise.expression.clone(),
                    expr,
                    origin: Origin::Premise,
                });
                premise_views.push(PremiseView {
                    statement: premise.statement.clone(),
                    expression: premise.expression.clone(),
                });
            }
            Err(e) => notes.push(format!(
                "Skipped the fact \"{}\": {}",
                premise.statement, e.message
            )),
        }
    }

    let mut findings = Vec::new();
    for claim in &extraction.claims {
        findings.push(decide_claim(&env, &constraints, claim, budget));
    }

    CheckReport {
        policy_id: policy.id.clone(),
        policy_version: policy.version,
        result: aggregate(&findings),
        findings,
        premises: premise_views,
        notes,
        elapsed_ms: started.elapsed().as_millis() as u64,
    }
}

/// Decide one claim, taking its alternative readings into account.
///
/// When the extraction offered other plausible formalizations and they do not all
/// reach the same verdict, the claim is reported as
/// [`Verdict::TranslationAmbiguous`] rather than resolved. Picking the primary
/// reading would be a coin flip presented as a proof; saying "this sentence has two
/// readings and they disagree" is actionable — it usually means the answer, or a
/// variable description, is genuinely vague.
fn decide_claim(
    env: &Env,
    constraints: &[Constraint],
    claim: &Extracted,
    budget: &Budget,
) -> Finding {
    let primary = match parse(&claim.expression, env) {
        Ok(expr) => expr,
        Err(e) => {
            return Finding::untranslatable(
                claim.expression.clone(),
                claim.statement.clone(),
                format!(
                    "This sentence could not be expressed in the policy's vocabulary: {}",
                    e.message
                ),
            )
        }
    };
    let mut finding = check_claim(
        env,
        constraints,
        &primary,
        &claim.expression,
        &claim.statement,
        budget,
    );

    let alternates: Vec<(String, Expr)> = claim
        .alternatives
        .iter()
        .filter(|alt| alt.trim() != claim.expression.trim())
        .filter_map(|alt| parse(alt, env).ok().map(|e| (alt.clone(), e)))
        .collect();

    for (src, expr) in alternates {
        let other = check_claim(env, constraints, &expr, &src, &claim.statement, budget);
        if other.verdict != finding.verdict {
            let detail = format!(
                "This sentence has more than one reasonable reading and they disagree: as `{}` it \
                 is {}, but as `{}` it is {}. Sharpen the answer, or add synonyms to the variable \
                 descriptions so the reading is unambiguous.",
                finding.claim,
                finding.verdict.as_str(),
                src,
                other.verdict.as_str()
            );
            finding.verdict = Verdict::TranslationAmbiguous;
            finding.detail = detail;
            finding.counterexample = None;
            finding.supporting_example = None;
            break;
        }
    }
    finding
}

/// Fold the findings into one headline verdict.
///
/// Ordered by how much a reader needs to know about it: a single contradiction
/// outranks any number of proofs, and "the solver gave up" outranks a clean pass so a
/// budget exhaustion is never read as approval.
pub fn aggregate(findings: &[Finding]) -> Verdict {
    if findings.is_empty() {
        return Verdict::NoTranslations;
    }
    for level in [
        Verdict::Invalid,
        Verdict::Impossible,
        Verdict::TooComplex,
        Verdict::TranslationAmbiguous,
    ] {
        if findings.iter().any(|f| f.verdict == level) {
            return level;
        }
    }
    if findings.iter().any(|f| f.verdict == Verdict::Satisfiable) {
        return Verdict::Satisfiable;
    }
    if findings
        .iter()
        .all(|f| f.verdict == Verdict::NoTranslations)
    {
        return Verdict::NoTranslations;
    }
    Verdict::Valid
}

// ── test suite ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct TestRun {
    passed: usize,
    failed: usize,
    results: Vec<TestResult>,
}

#[derive(Serialize)]
struct TestResult {
    id: String,
    passed: bool,
    expected: Verdict,
    actual: Verdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    report: Option<CheckReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// `POST /policies/:id/tests/run` — the policy's own regression suite.
#[utoipa::path(
    post,
    path = "/api/reasoning/policies/{id}/tests/run",
    tag = "Reasoning",
    summary = "Run every saved test case of a policy and report which verdicts still match what the author expected. Calls a model once per case to translate the prose, so it is slower than analyze and needs a model configured.",
    params(("id" = String, Path, description = "Policy id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn run_tests(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<String>,
) -> ApiResult<Json<TestRun>> {
    let host = ctx.host.as_ref().ok_or_else(ApiError::no_host)?;
    let policy = ctx
        .store
        .get(&id)
        .map_err(|e| ApiError::bad(e.to_string()))?
        .ok_or_else(ApiError::missing)?;

    let mut results = Vec::new();
    for case in &policy.tests {
        results.push(run_one_test(host, &policy, case, &ctx.budget).await);
    }
    Ok(Json(TestRun {
        passed: results.iter().filter(|r| r.passed).count(),
        failed: results.iter().filter(|r| !r.passed).count(),
        results,
    }))
}

async fn run_one_test(
    host: &Host,
    policy: &Policy,
    case: &TestCase,
    budget: &Budget,
) -> TestResult {
    match translate::extract(host, policy, &case.question, &case.answer).await {
        Ok(extraction) => {
            let report = evaluate(policy, &extraction, budget);
            TestResult {
                id: case.id.clone(),
                passed: report.result == case.expected,
                expected: case.expected,
                actual: report.result,
                report: Some(report),
                error: None,
            }
        }
        Err(e) => TestResult {
            id: case.id.clone(),
            passed: false,
            expected: case.expected,
            actual: Verdict::NoTranslations,
            report: None,
            error: Some(e.to_string()),
        },
    }
}

/// A machine-readable description of the check surface, served at `/capability` for
/// the capability broker.
pub fn capability_descriptor() -> serde_json::Value {
    json!({
        "capability": "reasoning.check",
        "version": "1.0.0",
        "verdicts": [
            Verdict::Valid, Verdict::Invalid, Verdict::Satisfiable, Verdict::Impossible,
            Verdict::NoTranslations, Verdict::TranslationAmbiguous, Verdict::TooComplex,
        ],
        "routes": {
            "check": { "method": "POST", "path": "/check",
                       "body": { "policy_id": "string", "question": "string", "answer": "string" } },
            "solve": { "method": "POST", "path": "/solve",
                       "body": { "policy_id": "string", "premises": ["expression"],
                                 "claims": ["expression"] } }
        }
    })
}

/// Variable values mentioned by a report, flattened for callers that want a summary
/// rather than the full findings tree.
pub fn assignments_of(report: &CheckReport) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for finding in &report.findings {
        for suggestion in &finding.suggestions {
            for (name, value) in &suggestion.assignments {
                out.insert(name.clone(), value.render());
            }
        }
    }
    out
}

#[cfg(test)]
mod openapi_tests {

    #[test]
    fn multi_method_paths_keep_every_operation() {
        // utoipa keys `paths` by path STRING, so handlers annotated separately on the
        // same path must MERGE into one PathItem. If one overwrote another, the path key
        // would still exist and the write body would still resolve — the read tool would
        // silently never exist, which is exactly the failure this document prevents. The
        // route-coverage test above cannot see that, because it only checks the key.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for (path, methods) in [
            ("/api/reasoning/policies", &["get", "post"][..]),
            (
                "/api/reasoning/policies/{id}",
                &["get", "put", "delete"][..],
            ),
        ] {
            let item = wire
                .pointer(&format!("/paths/{}", path.replace('/', "~1")))
                .unwrap_or_else(|| panic!("{path} has no PathItem"));
            for method in methods {
                assert!(
                    item.get(method).is_some(),
                    "{path} lost its {method} operation"
                );
            }
        }
    }
    /// This app's own manifest, read at compile time. The route contract lives there,
    /// so the invariants below compare the document against the real declaration rather
    /// than against a second list that could drift from it.
    fn manifest() -> serde_json::Value {
        serde_json::from_str(include_str!("../../manifest.json")).expect("valid JSON")
    }

    /// The manifest sidecar whose HTTP surface this router serves: the one declaring an
    /// `http.mount`. Selected BY mount rather than by index so a later mountless sidecar
    /// cannot silently redirect these assertions at the wrong process.
    fn mounted_sidecar() -> serde_json::Value {
        manifest()["sidecars"]
            .as_array()
            .expect("sidecars must be an array")
            .iter()
            .find(|s| s["http"]["mount"].is_string())
            .expect("one sidecar must declare an http.mount")
            .clone()
    }

    /// A manifest route (relative to the mount, `:param` form) rewritten into the form
    /// the OpenAPI document uses (absolute, `{param}` form). The two differ ON PURPOSE:
    /// the router registers relative paths because Core nests it, while the annotations
    /// carry the absolute EXTERNAL path. Normalise here; do not "align" either side.
    fn doc_path_for(mount: &str, route: &str) -> String {
        let joined = if route == "/" {
            mount.to_owned()
        } else {
            format!("{mount}{route}")
        };
        joined
            .split('/')
            .map(|seg| match seg.strip_prefix(':') {
                Some(name) => format!("{{{name}}}"),
                None => seg.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    #[test]
    fn openapi_doc_covers_the_served_routes() {
        assert!(!super::openapi().paths.paths.is_empty());
    }

    #[test]
    fn every_declared_route_appears_in_the_openapi_doc() {
        // The direction that decides tool yield: Core keeps only the document operations
        // the manifest ALSO declares, so a declared route with no `#[utoipa::path]` is a
        // tool that silently never exists — nothing errors, the agent simply cannot call
        // it.
        let sidecar = mounted_sidecar();
        let mount = sidecar["http"]["mount"].as_str().expect("an http.mount");
        let doc = super::openapi();
        for route in sidecar["http"]["routes"]
            .as_array()
            .expect("routes must be an array")
        {
            let path = route["path"].as_str().expect("a route path");
            let expected = doc_path_for(mount, path);
            assert!(
                doc.paths.paths.contains_key(&expected),
                "'{path}' is declared in manifest.json but the OpenAPI document has no \
                 '{expected}' operation — Core derives no tool for it"
            );
        }
    }

    #[test]
    fn write_operations_carry_a_typed_request_body() {
        // An untyped body still yields an operation, so the tool is DISCOVERABLE with
        // zero visible arguments. Assert the `$ref` resolves the way Core's
        // `resolve_ref` will.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for (path, method) in [
            ("/api/reasoning/policies", "post"),
            ("/api/reasoning/policies/{id}", "put"),
            ("/api/reasoning/policies/draft", "post"),
            ("/api/reasoning/check", "post"),
            ("/api/reasoning/solve", "post"),
        ] {
            let schema = wire
                .pointer(&format!(
                    "/paths/{}/{method}/requestBody/content/application~1json/schema/$ref",
                    path.replace('/', "~1")
                ))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("{method} {path} must declare a typed request body"));
            let name = schema
                .rsplit('/')
                .next()
                .expect("a $ref always has a last segment");
            assert!(
                wire.pointer(&format!("/components/schemas/{name}"))
                    .is_some(),
                "{method} {path} references {schema}, which is missing from components.schemas"
            );
        }
    }

    #[test]
    fn the_policy_schema_exposes_its_nested_shape() {
        // `Policy` is the create/update body, and it is only authorable if the graph
        // under it resolved too. A missing `ToSchema` on `Variable`/`Rule`/`TestCase`
        // would leave the model with a body it cannot fill.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for name in [
            "Policy", "Variable", "VarType", "Rule", "TestCase", "Verdict",
        ] {
            assert!(
                wire.pointer(&format!("/components/schemas/{name}"))
                    .is_some(),
                "{name} is reachable from the Policy body but missing from components.schemas"
            );
        }
    }

    #[test]
    fn id_only_operations_declare_no_body() {
        // Analyze and test-run take only `State` + `Path`. A `request_body` here would
        // invent an argument the handler never reads.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for path in [
            "/api/reasoning/policies/{id}/analyze",
            "/api/reasoning/policies/{id}/tests/run",
        ] {
            let op = wire
                .pointer(&format!("/paths/{}/post", path.replace('/', "~1")))
                .unwrap_or_else(|| panic!("{path} must have a POST operation"));
            assert!(
                op.get("requestBody").is_none(),
                "{path} takes no body but the document declares one"
            );
            assert!(
                op.get("parameters").is_some(),
                "{path} must still document its path id"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{VarType, Variable};
    use crate::policy::Rule;

    fn policy() -> Policy {
        Policy {
            id: "hr".into(),
            name: "HR".into(),
            description: String::new(),
            version: 3,
            variables: vec![
                Variable {
                    name: "is_manager".into(),
                    ty: VarType::Bool,
                    description: String::new(),
                },
                Variable {
                    name: "vacation_days".into(),
                    ty: VarType::Int,
                    description: String::new(),
                },
            ],
            rules: vec![Rule {
                id: "cap".into(),
                statement: "Managers get at most 30 vacation days.".into(),
                expression: "is_manager -> vacation_days <= 30".into(),
                enabled: true,
            }],
            tests: Vec::new(),
            source_document: None,
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    fn extraction(premises: &[&str], claims: &[(&str, Vec<&str>)]) -> Extraction {
        Extraction {
            premises: premises
                .iter()
                .map(|e| Extracted {
                    statement: (*e).into(),
                    expression: (*e).into(),
                    alternatives: Vec::new(),
                })
                .collect(),
            claims: claims
                .iter()
                .map(|(e, alts)| Extracted {
                    statement: (*e).into(),
                    expression: (*e).into(),
                    alternatives: alts.iter().map(|a| (*a).into()).collect(),
                })
                .collect(),
            notes: Vec::new(),
        }
    }

    #[test]
    fn a_contradicted_answer_is_invalid_and_carries_the_policy_version() {
        let report = evaluate(
            &policy(),
            &extraction(&["is_manager"], &[("vacation_days >= 45", vec![])]),
            &Budget::default(),
        );
        assert_eq!(report.result, Verdict::Invalid);
        assert_eq!(report.policy_version, 3);
        assert_eq!(report.premises.len(), 1);
    }

    #[test]
    fn an_unparseable_claim_becomes_no_translations_not_a_pass() {
        let report = evaluate(
            &policy(),
            &extraction(&[], &[("bonus_amount >= 10", vec![])]),
            &Budget::default(),
        );
        assert_eq!(report.result, Verdict::NoTranslations);
        assert!(report.findings[0].detail.contains("bonus_amount"));
    }

    #[test]
    fn disagreeing_readings_are_reported_as_ambiguous() {
        let report = evaluate(
            &policy(),
            &extraction(
                &["is_manager"],
                &[("vacation_days <= 30", vec!["vacation_days >= 45"])],
            ),
            &Budget::default(),
        );
        assert_eq!(report.result, Verdict::TranslationAmbiguous);
        assert!(report.findings[0]
            .detail
            .contains("more than one reasonable reading"));
    }

    #[test]
    fn agreeing_readings_do_not_trigger_ambiguity() {
        let report = evaluate(
            &policy(),
            &extraction(
                &["is_manager"],
                &[("vacation_days >= 45", vec!["vacation_days >= 50"])],
            ),
            &Budget::default(),
        );
        assert_eq!(report.result, Verdict::Invalid);
    }

    #[test]
    fn a_broken_rule_is_reported_as_a_note_rather_than_silently_ignored() {
        let mut p = policy();
        p.rules.push(Rule {
            id: "typo".into(),
            statement: "…".into(),
            expression: "vacation_dayz <= 5".into(),
            enabled: true,
        });
        let report = evaluate(
            &p,
            &extraction(&[], &[("vacation_days <= 100", vec![])]),
            &Budget::default(),
        );
        assert!(
            report.notes.iter().any(|n| n.contains("typo")),
            "notes: {:?}",
            report.notes
        );
    }

    #[test]
    fn aggregation_puts_a_single_contradiction_above_any_number_of_proofs() {
        let report = evaluate(
            &policy(),
            &extraction(
                &["is_manager"],
                &[
                    ("vacation_days <= 100", vec![]),
                    ("vacation_days >= 45", vec![]),
                ],
            ),
            &Budget::default(),
        );
        assert_eq!(report.result, Verdict::Invalid);
    }

    #[test]
    fn no_claims_means_no_translations() {
        let report = evaluate(&policy(), &extraction(&[], &[]), &Budget::default());
        assert_eq!(report.result, Verdict::NoTranslations);
    }

    #[test]
    fn slugs_are_filename_safe() {
        assert_eq!(slug("Refund Policy 2026"), "refund-policy-2026");
        assert_eq!(slug("../../etc/passwd"), "etc-passwd");
        assert!(slug("!!!").starts_with("policy-"));
    }
}
