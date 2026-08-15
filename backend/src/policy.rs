//! The policy document: what an author edits, and what it compiles to.
//!
//! A policy is a set of **declared variables** and a set of **rules**, each rule
//! carrying both a natural-language `statement` (what the author reads) and a DSL
//! `expression` (what the solver reads). Keeping both is deliberate: the sentence is
//! what a finding quotes back to a user, and the expression is what makes the finding
//! true. Editing one without the other is the failure mode, so
//! [`CompiledPolicy::compile`] reports a rule whose expression does not parse instead
//! of quietly dropping it.
//!
//! Beyond compiling, this module can interrogate a policy on its own terms —
//! [`analyze`] answers two questions no test suite catches:
//!
//! * **Is it consistent?** If the rules contradict each other, every check against
//!   the policy degrades to `impossible`. The analysis names the smallest conflicting
//!   set rather than the whole file.
//! * **Is any rule redundant?** A rule already implied by the others adds nothing.
//!   Harmless, but it usually means the author wrote the same constraint twice in
//!   different words and will later "fix" only one of them.

use serde::{Deserialize, Serialize};

use crate::logic::Variable;
use crate::parser::parse;
use crate::solver::{decide, Budget, Env, Outcome};
use crate::verdict::{Constraint, Origin, Verdict};

/// One rule of a policy.
#[derive(Clone, Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Rule {
    pub id: String,
    /// The rule as prose. This is what findings quote.
    pub statement: String,
    /// The rule in the DSL ([`crate::parser`]).
    pub expression: String,
    #[serde(default = "yes")]
    pub enabled: bool,
}

fn yes() -> bool {
    true
}

/// A saved question/answer pair with the verdict the author expects — the regression
/// test for a policy. Editing a rule to fix one case routinely breaks another; this
/// is what catches that.
#[derive(Clone, Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TestCase {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub expected: Verdict,
    #[serde(default)]
    pub note: String,
}

/// A stored policy.
#[derive(Clone, Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Policy {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Bumped on every save, so a check can record which revision judged it.
    #[serde(default = "one")]
    pub version: u32,
    #[serde(default)]
    pub variables: Vec<Variable>,
    #[serde(default)]
    pub rules: Vec<Rule>,
    #[serde(default)]
    pub tests: Vec<TestCase>,
    /// The document the policy was drafted from, kept so an author can re-read the
    /// source of a rule they no longer recognise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_document: Option<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
}

fn one() -> u32 {
    1
}

impl Policy {
    pub fn env(&self) -> Env {
        Env::from_variables(&self.variables)
    }
}

/// A rule that did not compile, reported with the parser's message so the author can
/// fix it in place.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleError {
    pub rule_id: String,
    pub expression: String,
    pub message: String,
    pub position: usize,
}

/// A policy turned into something the solver can use.
pub struct CompiledPolicy {
    pub env: Env,
    pub constraints: Vec<Constraint>,
    pub errors: Vec<RuleError>,
}

impl CompiledPolicy {
    pub fn compile(policy: &Policy) -> CompiledPolicy {
        let env = policy.env();
        let mut constraints = Vec::new();
        let mut errors = Vec::new();
        for rule in policy.rules.iter().filter(|r| r.enabled) {
            match parse(&rule.expression, &env) {
                Ok(expr) => constraints.push(Constraint {
                    id: rule.id.clone(),
                    statement: if rule.statement.trim().is_empty() {
                        rule.expression.clone()
                    } else {
                        rule.statement.clone()
                    },
                    expression: rule.expression.clone(),
                    expr,
                    origin: Origin::Rule,
                }),
                Err(e) => errors.push(RuleError {
                    rule_id: rule.id.clone(),
                    expression: rule.expression.clone(),
                    message: e.message,
                    position: e.position,
                }),
            }
        }
        CompiledPolicy {
            env,
            constraints,
            errors,
        }
    }
}

/// What [`analyze`] found.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PolicyAnalysis {
    /// `true` when the rules can all hold at once.
    pub consistent: bool,
    /// When inconsistent: the smallest set of rules that still conflicts.
    #[serde(default)]
    pub conflicting_rules: Vec<String>,
    /// Rules already implied by the others.
    #[serde(default)]
    pub redundant_rules: Vec<String>,
    /// Rules that did not compile.
    #[serde(default)]
    pub errors: Vec<RuleError>,
    /// Declared but never used in any rule — usually a typo, or a variable the author
    /// meant to constrain and did not.
    #[serde(default)]
    pub unused_variables: Vec<String>,
    /// Set when a budget ran out, so the report is not read as "all clear".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete: Option<String>,
}

/// Interrogate a policy on its own: consistency, redundancy, unused variables.
pub fn analyze(policy: &Policy, budget: &Budget) -> PolicyAnalysis {
    let compiled = CompiledPolicy::compile(policy);
    let mut out = PolicyAnalysis {
        consistent: true,
        errors: compiled.errors.clone(),
        ..Default::default()
    };

    let mut used: Vec<String> = Vec::new();
    for c in &compiled.constraints {
        c.expr.vars(&mut used);
    }
    out.unused_variables = policy
        .variables
        .iter()
        .filter(|v| !used.contains(&v.name))
        .map(|v| v.name.clone())
        .collect();

    let all: Vec<crate::logic::Expr> = compiled.constraints.iter().map(|c| c.expr.clone()).collect();
    match decide(&compiled.env, &all, budget) {
        Outcome::Unsat => {
            out.consistent = false;
            out.conflicting_rules = smallest_conflict(&compiled, budget);
            return out;
        }
        Outcome::Unknown(why) => {
            out.incomplete = Some(why);
            return out;
        }
        Outcome::Sat(_) => {}
    }

    // Redundancy: rule R is redundant when (others ∧ ¬R) has no model.
    for (idx, rule) in compiled.constraints.iter().enumerate() {
        let mut probe: Vec<crate::logic::Expr> = compiled
            .constraints
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, c)| c.expr.clone())
            .collect();
        probe.push(rule.expr.clone().not());
        match decide(&compiled.env, &probe, budget) {
            Outcome::Unsat => out.redundant_rules.push(rule.id.clone()),
            Outcome::Unknown(why) => {
                out.incomplete.get_or_insert(why);
            }
            Outcome::Sat(_) => {}
        }
    }

    out
}

/// Shrink an inconsistent rule set to a minimal conflicting subset.
fn smallest_conflict(compiled: &CompiledPolicy, budget: &Budget) -> Vec<String> {
    let mut keep: Vec<&Constraint> = compiled.constraints.iter().collect();
    let mut idx = 0;
    while idx < keep.len() {
        let trial: Vec<crate::logic::Expr> = keep
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, c)| c.expr.clone())
            .collect();
        if decide(&compiled.env, &trial, budget).is_unsat() {
            keep.remove(idx);
        } else {
            idx += 1;
        }
    }
    keep.into_iter().map(|c| c.id.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::VarType;

    fn policy(rules: &[(&str, &str)]) -> Policy {
        Policy {
            id: "p1".into(),
            name: "Test".into(),
            description: String::new(),
            version: 1,
            variables: vec![
                Variable {
                    name: "is_manager".into(),
                    ty: VarType::Bool,
                    description: String::new(),
                },
                Variable {
                    name: "days".into(),
                    ty: VarType::Int,
                    description: String::new(),
                },
                Variable {
                    name: "unused".into(),
                    ty: VarType::Bool,
                    description: String::new(),
                },
            ],
            rules: rules
                .iter()
                .map(|(id, expr)| Rule {
                    id: (*id).into(),
                    statement: String::new(),
                    expression: (*expr).into(),
                    enabled: true,
                })
                .collect(),
            tests: Vec::new(),
            source_document: None,
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn a_broken_expression_is_reported_not_dropped() {
        let compiled = CompiledPolicy::compile(&policy(&[("r1", "no_such_var >= 1")]));
        assert!(compiled.constraints.is_empty());
        assert_eq!(compiled.errors.len(), 1);
        assert!(compiled.errors[0].message.contains("no_such_var"));
    }

    #[test]
    fn conflicting_rules_are_named() {
        let out = analyze(
            &policy(&[("r1", "days >= 10"), ("r2", "days <= 2"), ("r3", "is_manager")]),
            &Budget::default(),
        );
        assert!(!out.consistent);
        assert_eq!(out.conflicting_rules.len(), 2);
        assert!(out.conflicting_rules.contains(&"r1".to_string()));
        assert!(out.conflicting_rules.contains(&"r2".to_string()));
    }

    #[test]
    fn a_rule_implied_by_another_is_flagged_redundant() {
        let out = analyze(
            &policy(&[("r1", "days >= 10"), ("r2", "days >= 5")]),
            &Budget::default(),
        );
        assert!(out.consistent);
        assert_eq!(out.redundant_rules, vec!["r2".to_string()]);
    }

    #[test]
    fn declared_but_unconstrained_variables_are_listed() {
        let out = analyze(&policy(&[("r1", "days >= 10")]), &Budget::default());
        assert!(out.unused_variables.contains(&"unused".to_string()));
        assert!(out.unused_variables.contains(&"is_manager".to_string()));
    }
}
