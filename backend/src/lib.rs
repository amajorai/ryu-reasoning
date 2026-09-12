//! Automated reasoning checks: decide whether a model's answer is *provably*
//! consistent with a written policy, instead of asking a second model to have an
//! opinion about it.
//!
//! ```text
//!   policy document ──(model)──▶ variables + rules ──(author edits)──▶ policy
//!                                                                        │
//!   question + answer ──(model)──▶ premises + claims ─────────┐           │
//!                                                            ▼           ▼
//!                                                        solver: is the claim
//!                                                        entailed / contradicted
//!                                                        / merely consistent?
//! ```
//!
//! A model is used at both edges, because turning prose into logic is a language
//! problem. It is used **nowhere in the middle**: the verdict itself comes from a
//! decision procedure ([`solver`]), and so do the counterexamples, the rules a
//! finding blames, and the corrections it suggests ([`verdict`]). That split is the
//! whole design — a mistranslation shows up as a wrong *question* being asked, which
//! an author can see and fix in the policy editor, rather than as a wrong answer
//! delivered with confidence.
//!
//! # Layout
//!
//! | module | role |
//! |--------|------|
//! | [`num`] | exact rational arithmetic (no floats anywhere) |
//! | [`logic`] | the formula language and its normal form |
//! | [`parser`] | the authoring DSL, type-checked against the declared variables |
//! | [`solver`] | the decision procedure, with budgets |
//! | [`verdict`] | entailment ordering, unsat cores, solver-derived suggestions |
//! | [`policy`] | the policy document model and its test suite |
//! | [`store`] | on-disk persistence |
//! | [`host`] | the sidecar's authenticated model callback into Core |
//! | [`translate`] | prose → variables/rules, and question/answer → premises/claims |
//! | [`api`] | the HTTP surface Core proxies as `/api/reasoning/*` |
//! | [`mcp`] | the same engine as an MCP stdio server, for agents and workflows |

pub mod api;
pub mod host;
pub mod logic;
pub mod mcp;
pub mod num;
pub mod parser;
pub mod paths;
pub mod policy;
pub mod solver;
pub mod store;
pub mod translate;
pub mod verdict;
