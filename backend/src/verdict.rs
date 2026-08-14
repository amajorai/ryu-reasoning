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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
                format!(
                    "The policy proves this claim. {} constraint(s) were needed.",
                    finding.responsible.len()
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
                "The policy contradicts this claim: {} constraint(s) rule it out.",
                finding.responsible.len()
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
            if let (Some(yes), Some(no)) =
                (&finding.supporting_example, &finding.counterexample)
            {
                let mut differing = diff_models(yes, no);
                let relevant = relevant_vars(claim_expr, &[]);
                differing.retain(|name, _| relevant.contains(name));
                if !differing.is_empty() {
                    finding.suggestions.push(Suggestion {
                        kind: "undetermined_variable".into(),
                        text: format!(
                            "Pin down {} to settle this claim.",
                            differing
                                .keys()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
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
mod tests {
    use super::*;
    use crate::logic::{VarType, Variable};
    use crate::parser::parse;
    use std::time::Duration;

    fn vars() -> Vec<Variable> {
        vec![
            Variable {
                name: "is_manager".into(),
                ty: VarType::Bool,
                description: String::new(),
            },
            Variable {
                name: "tenure_months".into(),
                ty: VarType::Int,
                description: String::new(),
            },
            Variable {
                name: "vacation_days".into(),
                ty: VarType::Int,
                description: String::new(),
            },
        ]
    }

    fn constraint(id: &str, src: &str, origin: Origin, env: &Env) -> Constraint {
        Constraint {
            id: id.into(),
            statement: src.into(),
            expression: src.into(),
            expr: parse(src, env).unwrap_or_else(|e| panic!("parse {src}: {e}")),
            origin,
        }
    }

    fn check(rules: &[&str], premises: &[&str], claim: &str) -> Finding {
        let env = Env::from_variables(&vars());
        let mut cs: Vec<Constraint> = rules
            .iter()
            .enumerate()
            .map(|(i, s)| constraint(&format!("rule-{i}"), s, Origin::Rule, &env))
            .collect();
        cs.extend(
            premises
                .iter()
                .enumerate()
                .map(|(i, s)| constraint(&format!("fact-{i}"), s, Origin::Premise, &env)),
        );
        let claim_expr = parse(claim, &env).unwrap_or_else(|e| panic!("parse {claim}: {e}"));
        check_claim(&env, &cs, &claim_expr, claim, claim, &Budget::default())
    }

    #[test]
    fn entailed_claim_is_valid() {
        let f = check(
            &["is_manager -> vacation_days <= 30"],
            &["is_manager"],
            "vacation_days <= 40",
        );
        assert_eq!(f.verdict, Verdict::Valid, "{}", f.detail);
    }

    #[test]
    fn contradicted_claim_is_invalid_with_a_counterexample() {
        let f = check(
            &["is_manager -> vacation_days <= 30"],
            &["is_manager"],
            "vacation_days >= 40",
        );
        assert_eq!(f.verdict, Verdict::Invalid, "{}", f.detail);
        assert!(f.counterexample.is_some(), "invalid needs a witness");
        assert!(!f.responsible.is_empty(), "invalid must name the rule");
    }

    #[test]
    fn undetermined_claim_is_satisfiable_not_valid() {
        let f = check(&["is_manager -> vacation_days <= 30"], &[], "is_manager");
        assert_eq!(f.verdict, Verdict::Satisfiable, "{}", f.detail);
    }

    /// The load-bearing guard. A policy whose rules conflict makes `P ∧ Q ∧ ¬C`
    /// unsatisfiable for EVERY claim; without the step-1 check the checker would
    /// stamp anything at all as proven.
    #[test]
    fn a_self_contradictory_policy_reports_impossible_not_valid() {
        let rules = ["tenure_months >= 12", "tenure_months <= 5"];
        for claim in [
            "vacation_days <= 30",
            "vacation_days >= 9000",
            "is_manager",
            "not is_manager",
        ] {
            let f = check(&rules, &[], claim);
            assert_eq!(
                f.verdict,
                Verdict::Impossible,
                "claim `{claim}` under a contradictory policy must be impossible, got {:?} — {}",
                f.verdict,
                f.detail
            );
            assert!(
                !f.responsible.is_empty(),
                "the conflicting rules must be named"
            );
        }
    }

    #[test]
    fn impossible_is_also_reported_when_the_question_contradicts_the_policy() {
        let f = check(
            &["is_manager -> tenure_months >= 12"],
            &["is_manager", "tenure_months <= 3"],
            "vacation_days <= 30",
        );
        assert_eq!(f.verdict, Verdict::Impossible, "{}", f.detail);
    }

    #[test]
    fn the_core_is_minimal_and_names_only_what_mattered() {
        let f = check(
            &[
                "is_manager -> vacation_days <= 30",
                "tenure_months >= 0",
                "vacation_days >= 0",
            ],
            &["is_manager"],
            "vacation_days >= 40",
        );
        assert_eq!(f.verdict, Verdict::Invalid);
        let ids: Vec<&str> = f.responsible.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"rule-0"), "the deciding rule must be listed");
        assert!(
            !ids.contains(&"rule-1"),
            "an irrelevant rule must not be listed: {ids:?}"
        );
    }

    #[test]
    fn a_valid_claim_reports_the_premises_it_leaned_on() {
        let f = check(
            &["is_manager -> vacation_days <= 30"],
            &["is_manager"],
            "vacation_days <= 30",
        );
        assert_eq!(f.verdict, Verdict::Valid, "{}", f.detail);
        assert!(
            f.unstated_assumptions.iter().any(|c| c.id == "fact-0"),
            "the answer relies on `is_manager`, which the policy did not state"
        );
    }

    #[test]
    fn an_invalid_claim_says_what_would_have_to_change() {
        let f = check(
            &["is_manager -> vacation_days <= 30"],
            &["is_manager", "vacation_days == 30"],
            "vacation_days >= 40",
        );
        assert_eq!(f.verdict, Verdict::Invalid);
        assert!(
            f.suggestions.iter().any(|s| s.kind == "would_be_valid_if"
                || s.kind == "never_valid"),
            "an invalid finding must carry a repair: {:?}",
            f.suggestions
        );
    }

    /// A repair must talk about the finding, not about whatever else the solver
    /// happened to pick. Two models differ on every variable the constraints leave
    /// free, so an unfiltered diff told a manager "this would be right if you were not
    /// a manager" — one nonsense clause discrediting a correct finding.
    #[test]
    fn a_repair_names_only_the_variables_the_finding_is_about() {
        let mut all = vars();
        all.push(Variable {
            name: "status".into(),
            ty: VarType::Enum {
                values: vec!["active".into(), "terminated".into()],
            },
            description: String::new(),
        });
        let env = Env::from_variables(&all);
        let cs = vec![
            constraint(
                "probation",
                "tenure_months < 12 -> vacation_days <= 15",
                Origin::Rule,
                &env,
            ),
            constraint(
                "terminated",
                "status == \"terminated\" -> vacation_days <= 0",
                Origin::Rule,
                &env,
            ),
            constraint("fact-1", "is_manager", Origin::Premise, &env),
            constraint("fact-2", "tenure_months == 8", Origin::Premise, &env),
        ];
        let claim = parse("vacation_days == 35", &env).unwrap();
        let f = check_claim(
            &env,
            &cs,
            &claim,
            "vacation_days == 35",
            "",
            &Budget::default(),
        );
        assert_eq!(f.verdict, Verdict::Invalid, "{}", f.detail);
        let repair = f
            .suggestions
            .iter()
            .find(|s| s.kind == "would_be_valid_if")
            .expect("an invalid finding carries a repair");
        assert!(
            repair.assignments.contains_key("tenure_months"),
            "the rule that decided it is about tenure: {:?}",
            repair.assignments
        );
        for noise in ["status", "is_manager"] {
            assert!(
                !repair.assignments.contains_key(noise),
                "'{noise}' had nothing to do with this finding, but the repair mentions \
                 it: {:?}",
                repair.assignments
            );
        }
    }

    /// A budget that cannot finish must degrade to `too_complex`, never to a verdict,
    /// and must not hang.
    #[test]
    fn an_exhausted_budget_degrades_to_too_complex() {
        let env = Env::from_variables(&vars());
        let cs: Vec<Constraint> = (0..12)
            .map(|i| {
                let src = format!("tenure_months >= {i} or vacation_days <= {i}");
                constraint(&format!("rule-{i}"), &src, Origin::Rule, &env)
            })
            .collect();
        let claim = parse("vacation_days <= 5", &env).unwrap();
        let budget = Budget {
            max_branch_nodes: 3,
            time_limit: Duration::from_millis(50),
            ..Budget::default()
        };
        let started = std::time::Instant::now();
        let f = check_claim(&env, &cs, &claim, "vacation_days <= 5", "", &budget);
        assert_eq!(f.verdict, Verdict::TooComplex, "{}", f.detail);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the budget must stop the search, not merely be documented"
        );
    }
}
