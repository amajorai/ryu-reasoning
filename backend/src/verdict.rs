//! Turning three solver calls into an answer a person can act on.
//!
//! Given a policy `P`, the premises `Q` extracted from the question, and one claim
//! `C` extracted from the answer, the verdict is decided in a **fixed order**:
//!
//! | # | question asked of the solver | answer | verdict |
//! |---|------------------------------|--------|---------|
//! | 1 | is `P ∧ Q` satisfiable?      | no     | [`Verdict::Impossible`] |
//! | 2 | is `P ∧ Q ∧ ¬C` satisfiable? | no     | [`Verdict::Valid`] |
//! | 3 | is `P ∧ Q ∧ C` satisfiable?  | no     | [`Verdict::Invalid`] |
//! | – | both 2 and 3 satisfiable     |        | [`Verdict::Satisfiable`] |
//!
//! **Step 1 is not optional.** If the policy contradicts itself — or contradicts the
//! situation described in the question — then `P ∧ Q ∧ ¬C` is unsatisfiable for
//! *every* `C`, and skipping straight to step 2 would stamp every claim, true or
//! false, as proven. A policy with two conflicting rules would silently pass
//! everything, which is worse than having no checker at all. So the contradiction is
//! reported as its own outcome, with the conflicting rules named.
//!
//! # Where the suggestions come from
//!
//! Not from a model. A second model reviewing the answer is a different product (the
//! `@ryu/double-check` plugin already is that one) and it can be as wrong as the
//! first. Everything this module reports is read back out of the solver:
//!
//! * **which rules mattered** — a minimal unsatisfiable core, computed by deleting
//!   one constraint at a time and re-asking. What survives is a set with no
//!   removable member, i.e. the rules that actually did the work.
//! * **unstated assumptions** (for a valid claim) — the members of that core that
//!   came from the *premises* rather than the policy. Those are the facts the answer
//!   quietly relies on.
//! * **the counterexample** (for an invalid claim) — a concrete assignment
//!   satisfying the policy and the premises in which the claim is false. It is a
//!   witness, so it cannot be a hallucination.
//! * **what would have to change** — a model of `P ∧ C`, diffed against the
//!   counterexample. The differing variables are exactly the conditions under which
//!   the answer would have been right.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::logic::Expr;
use crate::solver::{decide, Budget, Env, Model, Outcome, Value};

/// The outcome vocabulary. Wider than a pass/fail flag on purpose: "I could not
/// formalize this" and "this could be either way" are different situations from
/// "this is false", and collapsing them is how a checker becomes untrustworthy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The policy and premises **entail** the claim. It cannot be false here.
    Valid,
    /// The policy and premises **contradict** the claim. It cannot be true here.
    Invalid,
    /// Consistent with the policy but not implied by it — the answer might be right,
    /// and the policy does not say. Usually means a missing fact, not a missing rule.
    Satisfiable,
    /// The policy and premises contradict *each other*, so nothing can be concluded.
    /// Either the policy has conflicting rules or the question describes an
    /// impossible situation.
    Impossible,
    /// The answer said nothing this policy can express, or nothing that parsed.
    NoTranslations,
    /// The answer had more than one reasonable formalization, and they disagree.
    /// Reported rather than resolved: picking one at random would be a guess.
    TranslationAmbiguous,
    /// A solver budget ran out. Explicitly not a verdict.
    TooComplex,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Valid => "valid",
            Verdict::Invalid => "invalid",
            Verdict::Satisfiable => "satisfiable",
            Verdict::Impossible => "impossible",
            Verdict::NoTranslations => "no_translations",
            Verdict::TranslationAmbiguous => "translation_ambiguous",
            Verdict::TooComplex => "too_complex",
        }
    }
}

/// Where a constraint came from, which is what separates "the policy says" from
/// "your question implied".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Rule,
    Premise,
}

/// One labelled formula the verdict layer reasons with.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub id: String,
    /// The natural-language sentence, shown in findings.
    pub statement: String,
    /// The DSL source it was compiled from.
    pub expression: String,
    pub expr: Expr,
    pub origin: Origin,
}

/// A named constraint as it appears in a finding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstraintRef {
    pub id: String,
    pub statement: String,
    pub expression: String,
    pub origin: Origin,
}

impl From<&Constraint> for ConstraintRef {
    fn from(c: &Constraint) -> ConstraintRef {
        ConstraintRef {
            id: c.id.clone(),
            statement: c.statement.clone(),
            expression: c.expression.clone(),
            origin: c.origin,
        }
    }
}

/// A solver-derived suggestion. `assignments` is the machine-readable half; `text` is
/// the same content rendered for a human.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Suggestion {
    pub kind: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub assignments: BTreeMap<String, Value>,
}

/// The result for one claim.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub verdict: Verdict,
    /// The claim as the extraction produced it, in DSL form.
    pub claim: String,
    /// The claim in natural language, as the extraction described it.
    pub claim_statement: String,
    /// Rules and premises that actually decided this — a minimal core, not the whole
    /// policy.
    pub responsible: Vec<ConstraintRef>,
    /// Premises inside that core: the facts the answer silently relies on.
    pub unstated_assumptions: Vec<ConstraintRef>,
    /// A world satisfying the policy in which the claim is FALSE.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counterexample: Option<Model>,
    /// A world satisfying the policy in which the claim is TRUE.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supporting_example: Option<Model>,
    pub suggestions: Vec<Suggestion>,
    /// One sentence explaining the verdict.
    pub detail: String,
}

impl Finding {
    pub fn untranslatable(claim: String, statement: String, detail: String) -> Finding {
        Finding {
            verdict: Verdict::NoTranslations,
            claim,
            claim_statement: statement,
            responsible: Vec::new(),
            unstated_assumptions: Vec::new(),
            counterexample: None,
            supporting_example: None,
            suggestions: Vec::new(),
            detail,
        }
    }
}

/// Decide one claim against the policy and premises. See the module docs for the
/// ordering and why step 1 comes first.
pub fn check_claim(
    env: &Env,
    constraints: &[Constraint],
    claim_expr: &Expr,
    claim_src: &str,
    claim_statement: &str,
    budget: &Budget,
) -> Finding {
    let base: Vec<Expr> = constraints.iter().map(|c| c.expr.clone()).collect();

    let mut finding = Finding {
        verdict: Verdict::TooComplex,
        claim: claim_src.to_owned(),
        claim_statement: claim_statement.to_owned(),
        responsible: Vec::new(),
        unstated_assumptions: Vec::new(),
        counterexample: None,
        supporting_example: None,
        suggestions: Vec::new(),
        detail: String::new(),
    };

    // ── 1. Does the world described even exist? ───────────────────────────────
    let base_outcome = decide(env, &base, budget);
    match &base_outcome {
        Outcome::Unknown(why) => {
            finding.detail = format!("The solver could not decide this in budget: {why}.");
            return finding;
        }
        Outcome::Unsat => {
            finding.verdict = Verdict::Impossible;
            let core = minimal_core(env, constraints, &[], budget);
            finding.detail = if core.iter().all(|c| c.origin == Origin::Rule) {
                "The policy contradicts itself: the rules below cannot all hold at once, so \
                 nothing can be proved or disproved against it."
                    .to_owned()
            } else {
                "The question describes a situation the policy rules out, so no claim about it \
                 can be checked."
                    .to_owned()
            };
            finding.responsible = core.iter().map(|c| ConstraintRef::from(*c)).collect();
            return finding;
        }
        Outcome::Sat(_) => {}
    }

    // ── 2. Can the claim fail? ────────────────────────────────────────────────
    let mut with_negated = base.clone();
    with_negated.push(claim_expr.clone().not());
    let negated_outcome = decide(env, &with_negated, budget);
    if let Outcome::Unknown(why) = &negated_outcome {
        finding.detail = format!("The solver could not decide this in budget: {why}.");
        return finding;
    }

    // ── 3. Can the claim hold? ────────────────────────────────────────────────
    let mut with_claim = base.clone();
    with_claim.push(claim_expr.clone());
    let claim_outcome = decide(env, &with_claim, budget);
    if let Outcome::Unknown(why) = &claim_outcome {
        finding.detail = format!("The solver could not decide this in budget: {why}.");
        return finding;
    }

    match (negated_outcome.is_unsat(), claim_outcome.is_unsat()) {
        // The claim cannot be false: it is entailed.
        (true, _) => {
            finding.verdict = Verdict::Valid;
            let core = minimal_core(env, constraints, &[claim_expr.clone().not()], budget);
            finding.unstated_assumptions = core
                .iter()
                .filter(|c| c.origin == Origin::Premise)
                .map(|c| ConstraintRef::from(*c))
                .collect();
            finding.responsible = core.iter().map(|c| ConstraintRef::from(*c)).collect();
            finding.supporting_example = claim_outcome.model().cloned();
            finding.detail = if finding.responsible.is_empty() {
                "This claim is true no matter what the policy says.".to_owned()
            } else {
                // A count label, not "{n} constraint(s)": the sentence reads
                // the same for one constraint or many, in every language.
                format!(
                    "The policy proves this claim. Constraints needed: {needed}.",
                    needed = finding.responsible.len()
                )
            };
            if !finding.unstated_assumptions.is_empty() {
                let names: Vec<String> = finding
                    .unstated_assumptions
                    .iter()
                    .map(|c| c.statement.clone())
                    .collect();
                finding.suggestions.push(Suggestion {
                    kind: "unstated_assumption".into(),
                    text: format!(
                        "The answer is only correct because of unstated assumptions: {}.",
                        names.join("; ")
                    ),
                    assignments: BTreeMap::new(),
                });
            }
        }
        // The claim cannot be true: it is contradicted.
        (false, true) => {
            finding.verdict = Verdict::Invalid;
            let core = minimal_core(env, constraints, &[claim_expr.clone()], budget);
            finding.responsible = core.iter().map(|c| ConstraintRef::from(*c)).collect();
            let counterexample = base_outcome.model().cloned();
            finding.detail = format!(
                "The policy contradicts this claim. Constraints that rule it out: {count}.",
                count = finding.responsible.len()
            );
            // Only the variables this finding is ABOUT. Two models can differ on any
            // variable the constraints leave free, and folding those into the advice
            // produces sentences like "…only if is_manager = false" addressed to
            // someone who just said they are a manager — the one incidental clause
            // undermining an otherwise correct finding.
            let relevant = relevant_vars(claim_expr, &core);
            finding.suggestions.extend(repair_suggestions(
                env,
                constraints,
                claim_expr,
                counterexample.as_ref(),
                &relevant,
                budget,
            ));
            finding.counterexample = counterexample;
        }
        // Neither entailed nor contradicted.
        (false, false) => {
            finding.verdict = Verdict::Satisfiable;
            finding.supporting_example = claim_outcome.model().cloned();
            finding.counterexample = negated_outcome.model().cloned();
            finding.detail = "The policy neither proves nor disproves this claim — it is \
                              consistent either way. Usually a fact is missing rather than a rule."
                .to_owned();
            if let (Some(yes), Some(no)) = (&finding.supporting_example, &finding.counterexample) {
                let mut differing = diff_models(yes, no);
                let relevant = relevant_vars(claim_expr, &[]);
                differing.retain(|name, _| relevant.contains(name));
                if !differing.is_empty() {
                    finding.suggestions.push(Suggestion {
                        kind: "undetermined_variable".into(),
                        text: format!(
                            "Pin down {} to settle this claim.",
                            differing.keys().cloned().collect::<Vec<_>>().join(", ")
                        ),
                        assignments: differing,
                    });
                }
            }
        }
    }

    finding
}

/// Deletion-based minimal unsatisfiable core over the labelled constraints.
///
/// `extra` holds the unlabelled formulas that must stay (the claim, or its negation).
/// The caller guarantees `constraints ∪ extra` is unsatisfiable.
///
/// A constraint is dropped only when the remainder is *provably* still unsatisfiable.
/// If a probe returns `Unknown`, the constraint is kept: the core may then be larger
/// than minimal, but it is never wrong — the opposite trade would let a budget
/// timeout silently delete the rule that actually explained the finding.
fn minimal_core<'a>(
    env: &Env,
    constraints: &'a [Constraint],
    extra: &[Expr],
    budget: &Budget,
) -> Vec<&'a Constraint> {
    let mut keep: Vec<&Constraint> = constraints.iter().collect();
    let mut idx = 0;
    while idx < keep.len() {
        let mut trial: Vec<Expr> = keep
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, c)| c.expr.clone())
            .collect();
        trial.extend_from_slice(extra);
        if decide(env, &trial, budget).is_unsat() {
            keep.remove(idx);
        } else {
            idx += 1;
        }
    }
    keep
}

/// For an invalid claim: what would have had to be different?
///
/// Solve the policy *rules* together with the claim, dropping the premises taken from
/// the question. If that is satisfiable, the claim is not nonsense — it just does not
/// fit this case, and the difference between that world and the counterexample is
/// exactly the set of conditions under which the answer would have been right.
fn repair_suggestions(
    env: &Env,
    constraints: &[Constraint],
    claim: &Expr,
    counterexample: Option<&Model>,
    relevant: &BTreeSet<String>,
    budget: &Budget,
) -> Vec<Suggestion> {
    let rules_only: Vec<Expr> = constraints
        .iter()
        .filter(|c| c.origin == Origin::Rule)
        .map(|c| c.expr.clone())
        .collect();
    let mut probe = rules_only;
    probe.push(claim.clone());

    match decide(env, &probe, budget) {
        Outcome::Sat(model) => {
            let mut assignments = match counterexample {
                Some(base) => diff_models(&model, base),
                None => model.values.clone(),
            };
            assignments.retain(|name, _| relevant.contains(name));
            if assignments.is_empty() {
                return Vec::new();
            }
            let rendered: Vec<String> = assignments
                .iter()
                .map(|(name, value)| format!("{name} = {}", value.render()))
                .collect();
            vec![Suggestion {
                kind: "would_be_valid_if".into(),
                text: format!(
                    "The answer would be correct only if {}.",
                    rendered.join(" and ")
                ),
                assignments,
            }]
        }
        Outcome::Unsat => vec![Suggestion {
            kind: "never_valid".into(),
            text: "This claim contradicts the policy itself, not just the details of this \
                   question — no set of facts would make it correct."
                .into(),
            assignments: BTreeMap::new(),
        }],
        Outcome::Unknown(_) => Vec::new(),
    }
}

/// The variables a finding is actually about: those named in the claim, plus those
/// named by the constraints that decided it.
///
/// A model assigns EVERY variable the constraints mention, and two models can differ
/// on any of them that the constraints leave free. Reporting those differences
/// wholesale is how a correct finding acquires a wrong-sounding sentence, so the
/// suggestions are narrowed to this set.
fn relevant_vars(claim: &Expr, core: &[&Constraint]) -> BTreeSet<String> {
    let mut names = Vec::new();
    claim.vars(&mut names);
    for c in core {
        c.expr.vars(&mut names);
    }
    names.into_iter().collect()
}

/// Variables the two models disagree about.
fn diff_models(a: &Model, b: &Model) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for (name, value) in &a.values {
        if b.values.get(name) != Some(value) {
            out.insert(name.clone(), value.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests;
