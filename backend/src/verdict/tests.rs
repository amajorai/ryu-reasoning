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
        f.suggestions
            .iter()
            .any(|s| s.kind == "would_be_valid_if" || s.kind == "never_valid"),
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
