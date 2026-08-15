//! The formal language a policy is compiled into.
//!
//! Deliberately small, because the size of this language is exactly the set of
//! claims the solver can decide **completely**. Everything here is quantifier-free
//! over three sorts:
//!
//! * `bool` — a propositional variable;
//! * `enum` — a variable ranging over a finite, declared set of string values;
//! * `int` / `real` — a numeric variable constrained by *linear* comparisons.
//!
//! There are no functions, no quantifiers, and no arithmetic between two variables
//! (`salary * bonus` is rejected at parse time). That restriction is what makes
//! [`crate::solver`] a decision procedure rather than a heuristic: every formula in
//! this language is decidable, so a `valid`/`invalid` answer is a proof and not an
//! opinion.
//!
//! # Normal form
//!
//! Parsing normalizes away every derived connective and comparison, leaving four
//! atom shapes: a bool variable, an `enum == "value"` test, `linear <= 0`, and
//! `linear < 0`. In particular:
//!
//! * `a -> b` becomes `!a | b`, `a <-> b` becomes `(!a|b) & (!b|a)`;
//! * `x >= k` becomes `k - x <= 0`, `x > k` becomes `k - x < 0`;
//! * `x == k` becomes `(x - k <= 0) & (k - x <= 0)`;
//! * `x != k` becomes `(x - k < 0) | (k - x < 0)`.
//!
//! Negating an atom stays inside the language — `!(l <= 0)` is `-l < 0` — which is
//! what lets the solver case-split on atoms without ever leaving the fragment.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::num::Rat;

/// The sort of a policy variable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VarType {
    Bool,
    Int,
    Real,
    Enum { values: Vec<String> },
}

impl VarType {
    pub fn is_numeric(&self) -> bool {
        matches!(self, VarType::Int | VarType::Real)
    }

    pub fn label(&self) -> &'static str {
        match self {
            VarType::Bool => "bool",
            VarType::Int => "int",
            VarType::Real => "real",
            VarType::Enum { .. } => "enum",
        }
    }
}

/// A declared policy variable. `description` is not decoration: it is the text the
/// extraction model reads when deciding which variable a sentence is talking about,
/// so synonyms and examples belong in it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Variable {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: VarType,
    #[serde(default)]
    pub description: String,
}

/// A linear form `k + Σ cᵢ·xᵢ` over numeric variables.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Linear {
    pub constant: Rat,
    /// Sorted by variable name so two structurally equal forms share a key.
    pub terms: BTreeMap<String, Rat>,
}

impl Linear {
    pub fn constant(k: Rat) -> Linear {
        Linear {
            constant: k,
            terms: BTreeMap::new(),
        }
    }

    pub fn var(name: impl Into<String>) -> Linear {
        let mut terms = BTreeMap::new();
        terms.insert(name.into(), Rat::ONE);
        Linear {
            constant: Rat::ZERO,
            terms,
        }
    }

    pub fn is_constant(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn add(&self, other: &Linear) -> Option<Linear> {
        let mut out = self.clone();
        out.constant = out.constant.add(other.constant)?;
        for (name, coeff) in &other.terms {
            let entry = out.terms.entry(name.clone()).or_insert(Rat::ZERO);
            *entry = entry.add(*coeff)?;
            if entry.is_zero() {
                out.terms.remove(name);
            }
        }
        Some(out)
    }

    pub fn neg(&self) -> Option<Linear> {
        let mut out = Linear {
            constant: self.constant.neg()?,
            terms: BTreeMap::new(),
        };
        for (name, coeff) in &self.terms {
            out.terms.insert(name.clone(), coeff.neg()?);
        }
        Some(out)
    }

    pub fn sub(&self, other: &Linear) -> Option<Linear> {
        self.add(&other.neg()?)
    }

    pub fn scale(&self, factor: Rat) -> Option<Linear> {
        if factor.is_zero() {
            return Some(Linear::constant(Rat::ZERO));
        }
        let mut out = Linear {
            constant: self.constant.mul(factor)?,
            terms: BTreeMap::new(),
        };
        for (name, coeff) in &self.terms {
            out.terms.insert(name.clone(), coeff.mul(factor)?);
        }
        Some(out)
    }

    pub fn coeff(&self, name: &str) -> Rat {
        self.terms.get(name).copied().unwrap_or(Rat::ZERO)
    }

    /// A stable textual key, so `x - 1 <= 0` produced twice is one atom.
    pub fn key(&self) -> String {
        let mut out = String::new();
        for (name, coeff) in &self.terms {
            out.push_str(&format!("{coeff}*{name}+"));
        }
        out.push_str(&self.constant.to_display());
        out
    }

    /// Human-readable rendering, used in findings so a counterexample reads like a
    /// sentence rather than a matrix row.
    pub fn render(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (name, coeff) in &self.terms {
            if *coeff == Rat::ONE {
                parts.push(name.clone());
            } else if coeff.neg() == Some(Rat::ONE) {
                parts.push(format!("-{name}"));
            } else {
                parts.push(format!("{coeff}*{name}"));
            }
        }
        if !self.constant.is_zero() || parts.is_empty() {
            parts.push(self.constant.to_display());
        }
        parts.join(" + ").replace("+ -", "- ")
    }
}

/// A formula in the fragment. See the module docs for the normal form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Const(bool),
    /// A boolean variable.
    Bool(String),
    /// `var == "value"` for an enum-sorted variable.
    EnumEq(String, String),
    /// `linear <= 0`.
    Le(Linear),
    /// `linear < 0`.
    Lt(Linear),
    Not(Box<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}

impl Expr {
    pub fn not(self) -> Expr {
        match self {
            Expr::Const(b) => Expr::Const(!b),
            Expr::Not(inner) => *inner,
            other => Expr::Not(Box::new(other)),
        }
    }

    pub fn and(items: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for item in items {
            match item {
                Expr::Const(true) => {}
                Expr::Const(false) => return Expr::Const(false),
                Expr::And(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => Expr::Const(true),
            1 => flat.pop().expect("length checked"),
            _ => Expr::And(flat),
        }
    }

    pub fn or(items: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for item in items {
            match item {
                Expr::Const(false) => {}
                Expr::Const(true) => return Expr::Const(true),
                Expr::Or(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => Expr::Const(false),
            1 => flat.pop().expect("length checked"),
            _ => Expr::Or(flat),
        }
    }

    pub fn implies(self, other: Expr) -> Expr {
        Expr::or(vec![self.not(), other])
    }

    pub fn iff(self, other: Expr) -> Expr {
        Expr::and(vec![
            self.clone().implies(other.clone()),
            other.implies(self),
        ])
    }

    /// Every variable mentioned anywhere in the formula.
    pub fn vars(&self, out: &mut Vec<String>) {
        match self {
            Expr::Const(_) => {}
            Expr::Bool(name) | Expr::EnumEq(name, _) => out.push(name.clone()),
            Expr::Le(l) | Expr::Lt(l) => out.extend(l.terms.keys().cloned()),
            Expr::Not(inner) => inner.vars(out),
            Expr::And(items) | Expr::Or(items) => {
                for item in items {
                    item.vars(out);
                }
            }
        }
    }

    /// Collect the atoms (the leaves the solver case-splits on).
    pub fn atoms(&self, out: &mut Vec<Atom>) {
        match self {
            Expr::Const(_) => {}
            Expr::Bool(name) => out.push(Atom::Bool(name.clone())),
            Expr::EnumEq(name, value) => out.push(Atom::EnumEq(name.clone(), value.clone())),
            Expr::Le(l) => out.push(Atom::Le(l.clone())),
            Expr::Lt(l) => out.push(Atom::Lt(l.clone())),
            Expr::Not(inner) => inner.atoms(out),
            Expr::And(items) | Expr::Or(items) => {
                for item in items {
                    item.atoms(out);
                }
            }
        }
    }
}

/// An indivisible test. The solver assigns each atom true or false and then asks the
/// theory whether that assignment is consistent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Atom {
    Bool(String),
    EnumEq(String, String),
    Le(Linear),
    Lt(Linear),
}

impl Atom {
    /// Identity used for dedup and for the assignment map.
    pub fn key(&self) -> String {
        match self {
            Atom::Bool(name) => format!("b:{name}"),
            Atom::EnumEq(name, value) => format!("e:{name}={value}"),
            Atom::Le(l) => format!("le:{}", l.key()),
            Atom::Lt(l) => format!("lt:{}", l.key()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implication_normalizes_to_a_disjunction() {
        let a = Expr::Bool("a".into());
        let b = Expr::Bool("b".into());
        assert_eq!(
            a.implies(b),
            Expr::Or(vec![
                Expr::Not(Box::new(Expr::Bool("a".into()))),
                Expr::Bool("b".into())
            ])
        );
    }

    #[test]
    fn double_negation_collapses() {
        let a = Expr::Bool("a".into());
        assert_eq!(a.clone().not().not(), a);
    }

    #[test]
    fn conjunction_flattens_and_absorbs_constants() {
        let e = Expr::and(vec![
            Expr::Const(true),
            Expr::and(vec![Expr::Bool("a".into()), Expr::Bool("b".into())]),
        ]);
        assert_eq!(
            e,
            Expr::And(vec![Expr::Bool("a".into()), Expr::Bool("b".into())])
        );
        assert_eq!(
            Expr::and(vec![Expr::Bool("a".into()), Expr::Const(false)]),
            Expr::Const(false)
        );
    }

    #[test]
    fn linear_forms_share_a_key_when_structurally_equal() {
        let a = Linear::var("x").add(&Linear::constant(Rat::int(-3))).unwrap();
        let b = Linear::constant(Rat::int(-3)).add(&Linear::var("x")).unwrap();
        assert_eq!(a.key(), b.key());
    }

    #[test]
    fn scaling_by_zero_drops_the_variables() {
        let l = Linear::var("x").scale(Rat::ZERO).unwrap();
        assert!(l.is_constant());
    }
}
