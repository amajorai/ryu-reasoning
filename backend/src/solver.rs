//! The decision procedure.
//!
//! Two layers, in the shape of a small SMT solver:
//!
//! 1. **Boolean search.** The formula's atoms ([`Atom`]) are case-split one at a
//!    time. Every partial assignment is checked against the theory before the search
//!    goes deeper, so contradictory branches die early.
//! 2. **Theory.** Enum atoms are checked against the declared domain (a variable
//!    takes exactly one value). Numeric atoms become a conjunction of linear
//!    inequalities and are decided by **Fourier–Motzkin elimination** over the
//!    rationals, with **branch and bound** on top for `int`-sorted variables.
//!
//! Both layers are complete for the fragment in [`crate::logic`], so `Unsat` really
//! means "no model exists" — which is what makes an entailment verdict a proof.
//!
//! # Why there is a third answer
//!
//! Completeness is not the same as termination in useful time. Fourier–Motzkin is
//! doubly exponential in the number of eliminated variables and the boolean search
//! is exponential in the number of atoms. Every loop that can grow therefore
//! decrements a [`Budget`], and exhausting one yields [`Outcome::Unknown`] — never a
//! guess. The verdict layer surfaces that as `too_complex`. A solver that answered
//! `Unsat` because it gave up would report a hallucination as proven fact, which is
//! the exact failure this app exists to prevent.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::logic::{Atom, Expr, Linear, VarType, Variable};
use crate::num::Rat;

/// Resource ceilings. Every one of them is checked *inside* the loop it bounds.
#[derive(Clone, Debug)]
pub struct Budget {
    /// Distinct atoms the formula may contain before the search is refused outright.
    pub max_atoms: usize,
    /// Nodes the boolean case-split may visit.
    pub max_branch_nodes: usize,
    /// Constraints Fourier–Motzkin may hold at any elimination level.
    pub max_fm_constraints: usize,
    /// Integer branch-and-bound nodes.
    pub max_int_branches: usize,
    /// Wall-clock ceiling for one `solve` call.
    pub time_limit: Duration,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            max_atoms: 64,
            max_branch_nodes: 20_000,
            max_fm_constraints: 4_000,
            max_int_branches: 256,
            time_limit: Duration::from_secs(5),
        }
    }
}

/// What the solver decided.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// A model exists; here is one.
    Sat(Model),
    /// No model exists. This is a proof, not a timeout.
    Unsat,
    /// A budget ran out. Says nothing about satisfiability.
    Unknown(String),
}

impl Outcome {
    pub fn is_sat(&self) -> bool {
        matches!(self, Outcome::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, Outcome::Unsat)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, Outcome::Unknown(_))
    }

    pub fn model(&self) -> Option<&Model> {
        match self {
            Outcome::Sat(m) => Some(m),
            _ => None,
        }
    }
}

/// A concrete assignment to every variable the formula mentions — the witness a
/// finding shows the user as "here is a world where your answer is wrong".
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Model {
    pub values: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    Text(String),
    /// Rendered exactly (`"0.25"`, `"1/3"`), never as a lossy float.
    Number(String),
}

impl Value {
    pub fn render(&self) -> String {
        match self {
            Value::Bool(b) => b.to_string(),
            Value::Text(s) => s.clone(),
            Value::Number(n) => n.clone(),
        }
    }
}

/// The declared variables a formula is interpreted against.
#[derive(Clone, Debug, Default)]
pub struct Env {
    pub types: BTreeMap<String, VarType>,
}

impl Env {
    pub fn from_variables(vars: &[Variable]) -> Env {
        Env {
            types: vars
                .iter()
                .map(|v| (v.name.clone(), v.ty.clone()))
                .collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&VarType> {
        self.types.get(name)
    }

    pub fn is_int(&self, name: &str) -> bool {
        matches!(self.types.get(name), Some(VarType::Int))
    }
}

/// A linear inequality: `lin < 0` when `strict`, otherwise `lin <= 0`.
#[derive(Clone, Debug)]
struct NumConstraint {
    lin: Linear,
    strict: bool,
}

impl NumConstraint {
    /// For a constraint with no variables left: does the constant satisfy it?
    fn constant_holds(&self) -> Option<bool> {
        if !self.lin.is_constant() {
            return None;
        }
        Some(if self.strict {
            self.lin.constant.is_negative()
        } else {
            !self.lin.constant.is_positive()
        })
    }
}

struct Ctx {
    budget: Budget,
    started: Instant,
    nodes: usize,
    int_branches: usize,
    exhausted: Option<String>,
}

impl Ctx {
    fn new(budget: Budget) -> Ctx {
        Ctx {
            budget,
            started: Instant::now(),
            nodes: 0,
            int_branches: 0,
            exhausted: None,
        }
    }

    fn out_of_time(&mut self) -> bool {
        if self.exhausted.is_some() {
            return true;
        }
        if self.started.elapsed() > self.budget.time_limit {
            self.exhausted = Some(format!(
                "search exceeded its {} ms time limit",
                self.budget.time_limit.as_millis()
            ));
            return true;
        }
        false
    }

    fn tick_node(&mut self) -> bool {
        if self.out_of_time() {
            return false;
        }
        self.nodes += 1;
        if self.nodes > self.budget.max_branch_nodes {
            self.exhausted = Some(format!(
                "boolean search exceeded {} nodes",
                self.budget.max_branch_nodes
            ));
            return false;
        }
        true
    }

    fn tick_int_branch(&mut self) -> bool {
        if self.out_of_time() {
            return false;
        }
        self.int_branches += 1;
        if self.int_branches > self.budget.max_int_branches {
            self.exhausted = Some(format!(
                "integer branch-and-bound exceeded {} branches",
                self.budget.max_int_branches
            ));
            return false;
        }
        true
    }
}

/// Decide a conjunction of formulas.
///
/// The formulas are implicitly ANDed: this is the single primitive the verdict layer
/// calls with `policy ∧ premises ∧ ¬claim` and friends.
pub fn solve(env: &Env, constraints: &[Expr], budget: &Budget) -> Outcome {
    let mut atoms: Vec<Atom> = Vec::new();
    for c in constraints {
        c.atoms(&mut atoms);
    }
    let mut seen = HashSet::new();
    atoms.retain(|a| seen.insert(a.key()));

    if atoms.len() > budget.max_atoms {
        return Outcome::Unknown(format!(
            "formula has {} distinct atoms, over the limit of {}",
            atoms.len(),
            budget.max_atoms
        ));
    }

    let mut ctx = Ctx::new(budget.clone());
    let mut assignment: HashMap<String, bool> = HashMap::new();
    let found = search(env, constraints, &atoms, &mut assignment, &mut ctx);
    match found {
        Some(model) => Outcome::Sat(model),
        None => match ctx.exhausted {
            Some(reason) => Outcome::Unknown(reason),
            None => Outcome::Unsat,
        },
    }
}

/// Convenience: is this set of formulas satisfiable?
pub fn is_sat(env: &Env, constraints: &[Expr], budget: &Budget) -> Outcome {
    solve(env, constraints, budget)
}

fn search(
    env: &Env,
    constraints: &[Expr],
    atoms: &[Atom],
    assignment: &mut HashMap<String, bool>,
    ctx: &mut Ctx,
) -> Option<Model> {
    if !ctx.tick_node() {
        return None;
    }

    // Propositional pruning: a constraint already false under the partial assignment
    // kills the branch without touching the theory.
    let mut all_true = true;
    for c in constraints {
        match eval(c, assignment) {
            Some(false) => return None,
            Some(true) => {}
            None => all_true = false,
        }
    }

    // Theory consistency of what has been assigned so far.
    let theory = check_theory(env, assignment, ctx);
    match theory {
        TheoryResult::Infeasible => return None,
        TheoryResult::Unknown => return None, // ctx.exhausted is already set
        TheoryResult::Feasible(numeric) => {
            if all_true {
                return Some(build_model(env, assignment, &numeric));
            }
        }
    }

    // Split on an atom that is still undetermined.
    let next = atoms.iter().find(|a| !assignment.contains_key(&a.key()))?;
    let key = next.key();
    for value in [true, false] {
        assignment.insert(key.clone(), value);
        if let Some(model) = search(env, constraints, atoms, assignment, ctx) {
            assignment.remove(&key);
            return Some(model);
        }
        assignment.remove(&key);
        if ctx.exhausted.is_some() {
            return None;
        }
    }
    None
}

/// Three-valued evaluation of a parsed formula under a partial assignment. `None`
/// means "not yet determined by what has been assigned".
///
/// This walks an in-memory [`Expr`] tree. It executes nothing: there is no host
/// language here, no interpreter, and no path from policy text to code — the parser
/// only ever produces the four atom shapes in [`crate::logic`].
fn eval(e: &Expr, assignment: &HashMap<String, bool>) -> Option<bool> {
    match e {
        Expr::Const(b) => Some(*b),
        Expr::Bool(_) | Expr::EnumEq(..) | Expr::Le(_) | Expr::Lt(_) => {
            let key = atom_key_of(e);
            assignment.get(&key).copied()
        }
        Expr::Not(inner) => eval(inner, assignment).map(|v| !v),
        Expr::And(items) => {
            let mut unknown = false;
            for item in items {
                match eval(item, assignment) {
                    Some(false) => return Some(false),
                    Some(true) => {}
                    None => unknown = true,
                }
            }
            if unknown {
                None
            } else {
                Some(true)
            }
        }
        Expr::Or(items) => {
            let mut unknown = false;
            for item in items {
                match eval(item, assignment) {
                    Some(true) => return Some(true),
                    Some(false) => {}
                    None => unknown = true,
                }
            }
            if unknown {
                None
            } else {
                Some(false)
            }
        }
    }
}

fn atom_key_of(e: &Expr) -> String {
    match e {
        Expr::Bool(name) => Atom::Bool(name.clone()).key(),
        Expr::EnumEq(name, value) => Atom::EnumEq(name.clone(), value.clone()).key(),
        Expr::Le(l) => Atom::Le(l.clone()).key(),
        Expr::Lt(l) => Atom::Lt(l.clone()).key(),
        _ => unreachable!("atom_key_of is only called on atoms"),
    }
}

enum TheoryResult {
    Feasible(BTreeMap<String, Rat>),
    Infeasible,
    Unknown,
}

/// Check the assigned atoms against the enum domains and the linear arithmetic.
fn check_theory(
    env: &Env,
    assignment: &HashMap<String, bool>,
    ctx: &mut Ctx,
) -> TheoryResult {
    // ── Enums: exactly one declared value per variable ────────────────────────
    let mut trues: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut falses: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut numeric: Vec<NumConstraint> = Vec::new();

    for (key, value) in assignment {
        if let Some(rest) = key.strip_prefix("e:") {
            let Some((var, val)) = rest.split_once('=') else {
                continue;
            };
            if *value {
                trues.entry(var).or_default().push(val);
            } else {
                falses.entry(var).or_default().insert(val);
            }
        }
    }
    for (var, vals) in &trues {
        if vals.len() > 1 {
            // `status == "active"` and `status == "closed"` cannot both hold.
            return TheoryResult::Infeasible;
        }
        if let Some(VarType::Enum { values }) = env.get(var) {
            if !values.iter().any(|v| v == vals[0]) {
                return TheoryResult::Infeasible;
            }
            if falses.get(var).is_some_and(|f| f.contains(vals[0])) {
                return TheoryResult::Infeasible;
            }
        }
    }
    for (var, excluded) in &falses {
        if trues.contains_key(var) {
            continue;
        }
        if let Some(VarType::Enum { values }) = env.get(var) {
            if values.iter().all(|v| excluded.contains(v.as_str())) {
                // Every declared value has been ruled out.
                return TheoryResult::Infeasible;
            }
        }
    }

    // ── Linear arithmetic ─────────────────────────────────────────────────────
    // Rebuilt from the atom table rather than by re-parsing the key strings: the
    // atoms carry the structured [`Linear`] forms, the keys are only identities.
    numeric.extend(numeric_constraints(assignment));

    if numeric.is_empty() {
        return TheoryResult::Feasible(BTreeMap::new());
    }
    match feasible_linear(env, &numeric, ctx) {
        LinearResult::Feasible(m) => TheoryResult::Feasible(m),
        LinearResult::Infeasible => TheoryResult::Infeasible,
        LinearResult::Unknown => TheoryResult::Unknown,
    }
}

thread_local! {
    /// Atoms indexed by key for the current `solve` call. The assignment map is keyed
    /// by string (cheap to clone through the search) while the theory needs the
    /// structured [`Linear`] behind each numeric key; this table is the bridge.
    static ATOM_TABLE: std::cell::RefCell<HashMap<String, Atom>> =
        std::cell::RefCell::new(HashMap::new());
}

fn numeric_constraints(assignment: &HashMap<String, bool>) -> Vec<NumConstraint> {
    ATOM_TABLE.with(|table| {
        let table = table.borrow();
        let mut out = Vec::new();
        for (key, value) in assignment {
            let Some(atom) = table.get(key) else { continue };
            match atom {
                // `l <= 0` asserted true stays `l <= 0`; asserted false becomes
                // `l > 0`, i.e. `-l < 0`. Negation never leaves the fragment.
                Atom::Le(l) => {
                    if *value {
                        out.push(NumConstraint {
                            lin: l.clone(),
                            strict: false,
                        });
                    } else if let Some(neg) = l.neg() {
                        out.push(NumConstraint {
                            lin: neg,
                            strict: true,
                        });
                    }
                }
                Atom::Lt(l) => {
                    if *value {
                        out.push(NumConstraint {
                            lin: l.clone(),
                            strict: true,
                        });
                    } else if let Some(neg) = l.neg() {
                        out.push(NumConstraint {
                            lin: neg,
                            strict: false,
                        });
                    }
                }
                _ => {}
            }
        }
        out
    })
}

enum LinearResult {
    Feasible(BTreeMap<String, Rat>),
    Infeasible,
    Unknown,
}

/// Rational feasibility by Fourier–Motzkin, then integrality by branch and bound.
fn feasible_linear(env: &Env, constraints: &[NumConstraint], ctx: &mut Ctx) -> LinearResult {
    let mut work: Vec<NumConstraint> = constraints.to_vec();
    let mut depth = 0usize;
    loop {
        match fourier_motzkin(&work, ctx) {
            LinearResult::Infeasible => return LinearResult::Infeasible,
            LinearResult::Unknown => return LinearResult::Unknown,
            LinearResult::Feasible(model) => {
                // Integrality: find an `int` variable that came out fractional.
                let fractional = model
                    .iter()
                    .find(|(name, value)| env.is_int(name) && !value.is_integer());
                let Some((name, value)) = fractional else {
                    return LinearResult::Feasible(model);
                };
                if !ctx.tick_int_branch() {
                    return LinearResult::Unknown;
                }
                depth += 1;
                if depth > ctx.budget.max_int_branches {
                    ctx.exhausted = Some("integer branch-and-bound did not converge".into());
                    return LinearResult::Unknown;
                }
                let (name, value) = (name.clone(), *value);
                let (Some(floor), Some(ceil)) = (value.floor(), value.ceil()) else {
                    ctx.exhausted = Some("integer bound overflowed".into());
                    return LinearResult::Unknown;
                };
                let (Some(minus_floor), Some(plus_ceil)) =
                    (floor.checked_neg().and_then(|v| Rat::new(v, 1)), Rat::new(ceil, 1))
                else {
                    ctx.exhausted = Some("integer bound overflowed".into());
                    return LinearResult::Unknown;
                };
                // Branch 1: x <= floor(v)  ⇒  x - floor <= 0
                let mut lower_branch = work.clone();
                let Some(lin) = Linear::var(&name).add(&Linear::constant(minus_floor)) else {
                    return LinearResult::Unknown;
                };
                lower_branch.push(NumConstraint { lin, strict: false });
                if let LinearResult::Feasible(m) = feasible_linear(env, &lower_branch, ctx) {
                    return LinearResult::Feasible(m);
                }
                if ctx.exhausted.is_some() {
                    return LinearResult::Unknown;
                }
                // Branch 2: x >= ceil(v)  ⇒  ceil - x <= 0
                let Some(neg_var) = Linear::var(&name).neg() else {
                    return LinearResult::Unknown;
                };
                let Some(lin) = neg_var.add(&Linear::constant(plus_ceil)) else {
                    return LinearResult::Unknown;
                };
                work.push(NumConstraint { lin, strict: false });
                // Loop continues with the tightened upper branch.
            }
        }
    }
}

/// Fourier–Motzkin elimination with model reconstruction.
fn fourier_motzkin(constraints: &[NumConstraint], ctx: &mut Ctx) -> LinearResult {
    let mut vars: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for c in constraints {
        for name in c.lin.terms.keys() {
            if seen.insert(name.clone()) {
                vars.push(name.clone());
            }
        }
    }
    vars.sort();

    // `levels[i]` is the constraint set *before* `vars[i]` was eliminated, which is
    // what the back-substitution pass needs to bound that variable.
    let mut levels: Vec<Vec<NumConstraint>> = Vec::new();
    let mut current: Vec<NumConstraint> = constraints.to_vec();

    for var in &vars {
        if ctx.out_of_time() {
            return LinearResult::Unknown;
        }
        levels.push(current.clone());
        let mut lower: Vec<&NumConstraint> = Vec::new();
        let mut upper: Vec<&NumConstraint> = Vec::new();
        let mut rest: Vec<NumConstraint> = Vec::new();
        for c in &current {
            let a = c.lin.coeff(var);
            if a.is_zero() {
                rest.push(c.clone());
            } else if a.is_positive() {
                upper.push(c);
            } else {
                lower.push(c);
            }
        }
        let mut next = rest;
        for lo in &lower {
            for up in &upper {
                let a_lo = lo.lin.coeff(var);
                let a_up = up.lin.coeff(var);
                // Scale `up` by -a_lo (> 0) and `lo` by a_up (> 0) so the two `var`
                // terms cancel. Both multipliers are positive, so the inequality
                // directions survive.
                let Some(neg_a_lo) = a_lo.neg() else {
                    return LinearResult::Unknown;
                };
                let (Some(su), Some(sl)) = (up.lin.scale(neg_a_lo), lo.lin.scale(a_up)) else {
                    return LinearResult::Unknown;
                };
                let Some(mut combined) = su.add(&sl) else {
                    return LinearResult::Unknown;
                };
                combined.terms.remove(var);
                next.push(NumConstraint {
                    lin: combined,
                    strict: lo.strict || up.strict,
                });
                if next.len() > ctx.budget.max_fm_constraints {
                    ctx.exhausted = Some(format!(
                        "Fourier–Motzkin elimination exceeded {} constraints",
                        ctx.budget.max_fm_constraints
                    ));
                    return LinearResult::Unknown;
                }
            }
        }
        current = next;
    }

    // Every variable is gone: what remains must hold on the constants alone.
    for c in &current {
        match c.constant_holds() {
            Some(true) => {}
            Some(false) => return LinearResult::Infeasible,
            None => return LinearResult::Unknown,
        }
    }

    // Back-substitute in reverse elimination order.
    let mut model: BTreeMap<String, Rat> = BTreeMap::new();
    for (idx, var) in vars.iter().enumerate().rev() {
        let level = &levels[idx];
        let mut lower: Option<(Rat, bool)> = None; // (bound, strict)
        let mut upper: Option<(Rat, bool)> = None;
        for c in level {
            let a = c.lin.coeff(var);
            if a.is_zero() {
                continue;
            }
            // Evaluate the rest of the form with the already-chosen values.
            let Some(rest) = eval_linear_without(&c.lin, var, &model) else {
                return LinearResult::Unknown;
            };
            let Some(bound) = rest.neg().and_then(|r| r.div(a)) else {
                return LinearResult::Unknown;
            };
            if a.is_positive() {
                // a*var + rest ⋈ 0  ⇒  var ⋈ -rest/a  (upper bound)
                upper = Some(match upper {
                    None => (bound, c.strict),
                    Some((b, s)) => {
                        if bound < b {
                            (bound, c.strict)
                        } else if bound == b {
                            (b, s || c.strict)
                        } else {
                            (b, s)
                        }
                    }
                });
            } else {
                // Dividing by a negative flips the direction: var ⋈ -rest/a (lower).
                lower = Some(match lower {
                    None => (bound, c.strict),
                    Some((b, s)) => {
                        if bound > b {
                            (bound, c.strict)
                        } else if bound == b {
                            (b, s || c.strict)
                        } else {
                            (b, s)
                        }
                    }
                });
            }
        }
        let Some(value) = pick_value(lower, upper) else {
            return LinearResult::Unknown;
        };
        model.insert(var.clone(), value);
    }

    LinearResult::Feasible(model)
}

/// Sum the linear form's constant and every term except `var`, using `model`.
fn eval_linear_without(lin: &Linear, var: &str, model: &BTreeMap<String, Rat>) -> Option<Rat> {
    let mut total = lin.constant;
    for (name, coeff) in &lin.terms {
        if name == var {
            continue;
        }
        let value = model.get(name)?;
        total = total.add(coeff.mul(*value)?)?;
    }
    Some(total)
}

/// Choose a witness value inside `(lower, upper)`, preferring a whole number so the
/// integer branch-and-bound above almost never has to run.
fn pick_value(lower: Option<(Rat, bool)>, upper: Option<(Rat, bool)>) -> Option<Rat> {
    let integer_in_range = |lo: Option<(Rat, bool)>, hi: Option<(Rat, bool)>| -> Option<Rat> {
        let lo_int = match lo {
            None => None,
            Some((b, strict)) => Some(if strict {
                b.floor()?.checked_add(1)?
            } else {
                b.ceil()?
            }),
        };
        let hi_int = match hi {
            None => None,
            Some((b, strict)) => Some(if strict {
                b.ceil()?.checked_sub(1)?
            } else {
                b.floor()?
            }),
        };
        match (lo_int, hi_int) {
            (Some(l), Some(h)) if l <= h => Rat::new(l, 1),
            (Some(l), None) => Rat::new(l, 1),
            (None, Some(h)) => Rat::new(h, 1),
            (None, None) => Some(Rat::ZERO),
            _ => None,
        }
    };
    if let Some(v) = integer_in_range(lower, upper) {
        return Some(v);
    }
    match (lower, upper) {
        (Some((lo, _)), Some((hi, _))) => {
            if lo < hi {
                lo.midpoint(hi)
            } else if lo == hi {
                Some(lo)
            } else {
                None
            }
        }
        (Some((lo, strict)), None) => {
            if strict {
                lo.add(Rat::ONE)
            } else {
                Some(lo)
            }
        }
        (None, Some((hi, strict))) => {
            if strict {
                hi.sub(Rat::ONE)
            } else {
                Some(hi)
            }
        }
        (None, None) => Some(Rat::ZERO),
    }
}

fn build_model(
    env: &Env,
    assignment: &HashMap<String, bool>,
    numeric: &BTreeMap<String, Rat>,
) -> Model {
    let mut values: BTreeMap<String, Value> = BTreeMap::new();

    for (key, value) in assignment {
        if let Some(name) = key.strip_prefix("b:") {
            values.insert(name.to_owned(), Value::Bool(*value));
        }
    }
    // Enum variables: the value asserted true, else any declared value not excluded.
    let mut excluded: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut chosen: HashMap<&str, &str> = HashMap::new();
    for (key, value) in assignment {
        if let Some(rest) = key.strip_prefix("e:") {
            if let Some((var, val)) = rest.split_once('=') {
                if *value {
                    chosen.insert(var, val);
                } else {
                    excluded.entry(var).or_default().insert(val);
                }
            }
        }
    }
    for (var, val) in &chosen {
        values.insert((*var).to_owned(), Value::Text((*val).to_owned()));
    }
    for (var, excl) in &excluded {
        if chosen.contains_key(var) {
            continue;
        }
        if let Some(VarType::Enum { values: domain }) = env.get(var) {
            if let Some(pick) = domain.iter().find(|v| !excl.contains(v.as_str())) {
                values.insert((*var).to_owned(), Value::Text(pick.clone()));
            }
        }
    }
    for (name, value) in numeric {
        values.insert(name.clone(), Value::Number(value.to_display()));
    }
    Model { values }
}

/// Populate the atom table for one solve call and run `f`.
///
/// The table is thread-local rather than threaded through every frame because the
/// search is a plain recursive function; a solve call never yields, so there is no
/// interleaving to race with.
pub fn with_atom_table<T>(constraints: &[Expr], f: impl FnOnce() -> T) -> T {
    let mut atoms = Vec::new();
    for c in constraints {
        c.atoms(&mut atoms);
    }
    ATOM_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        table.clear();
        for atom in atoms {
            table.insert(atom.key(), atom);
        }
    });
    let out = f();
    ATOM_TABLE.with(|table| table.borrow_mut().clear());
    out
}

/// The entry point callers should use: prepares the atom table, then solves.
pub fn decide(env: &Env, constraints: &[Expr], budget: &Budget) -> Outcome {
    with_atom_table(constraints, || solve(env, constraints, budget))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn env() -> Env {
        Env::from_variables(&[
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
            Variable {
                name: "discount".into(),
                ty: VarType::Real,
                description: String::new(),
            },
            Variable {
                name: "status".into(),
                ty: VarType::Enum {
                    values: vec!["active".into(), "leave".into(), "terminated".into()],
                },
                description: String::new(),
            },
        ])
    }

    fn p(src: &str) -> Expr {
        parse(src, &env()).unwrap_or_else(|e| panic!("parse {src}: {e}"))
    }

    #[test]
    fn simple_conjunction_is_satisfiable() {
        let out = decide(&env(), &[p("is_manager"), p("tenure_months >= 12")], &Budget::default());
        let model = out.model().expect("sat");
        assert_eq!(model.values.get("is_manager"), Some(&Value::Bool(true)));
    }

    #[test]
    fn contradictory_bounds_are_unsat() {
        let out = decide(
            &env(),
            &[p("tenure_months >= 12"), p("tenure_months <= 5")],
            &Budget::default(),
        );
        assert!(out.is_unsat(), "expected unsat, got {out:?}");
    }

    #[test]
    fn enum_cannot_take_two_values_at_once() {
        let out = decide(
            &env(),
            &[p("status == \"active\""), p("status == \"leave\"")],
            &Budget::default(),
        );
        assert!(out.is_unsat(), "expected unsat, got {out:?}");
    }

    #[test]
    fn ruling_out_every_enum_value_is_unsat() {
        let out = decide(
            &env(),
            &[
                p("status != \"active\""),
                p("status != \"leave\""),
                p("status != \"terminated\""),
            ],
            &Budget::default(),
        );
        assert!(out.is_unsat(), "expected unsat, got {out:?}");
    }

    #[test]
    fn integer_gap_is_detected_by_branch_and_bound() {
        // 0 < tenure_months < 1 has rational solutions but no integer one.
        let out = decide(
            &env(),
            &[p("tenure_months > 0"), p("tenure_months < 1")],
            &Budget::default(),
        );
        assert!(out.is_unsat(), "expected unsat, got {out:?}");
    }

    #[test]
    fn the_same_gap_is_satisfiable_for_a_real() {
        let out = decide(&env(), &[p("discount > 0"), p("discount < 1")], &Budget::default());
        assert!(out.is_sat(), "expected sat, got {out:?}");
    }

    #[test]
    fn exact_decimals_do_not_drift() {
        // With f64 this is unsatisfiable by rounding; with rationals it is not.
        let out = decide(
            &env(),
            &[p("discount == 0.1 + 0.2"), p("discount <= 0.3")],
            &Budget::default(),
        );
        assert!(out.is_sat(), "expected sat, got {out:?}");
    }

    #[test]
    fn disjunction_explores_both_arms() {
        let out = decide(
            &env(),
            &[p("tenure_months <= 5 or tenure_months >= 100"), p("tenure_months >= 50")],
            &Budget::default(),
        );
        let model = out.model().expect("sat");
        assert_eq!(
            model.values.get("tenure_months"),
            Some(&Value::Number("100".into()))
        );
    }

    #[test]
    fn a_starved_budget_returns_unknown_not_a_verdict() {
        let budget = Budget {
            max_branch_nodes: 1,
            ..Budget::default()
        };
        let out = decide(
            &env(),
            &[p("is_manager or tenure_months >= 12"), p("not is_manager"), p("tenure_months <= 5")],
            &budget,
        );
        assert!(out.is_unknown(), "a starved search must not claim unsat: {out:?}");
    }

    #[test]
    fn too_many_atoms_is_refused_up_front() {
        let budget = Budget {
            max_atoms: 1,
            ..Budget::default()
        };
        let out = decide(&env(), &[p("is_manager"), p("tenure_months >= 1")], &budget);
        assert!(out.is_unknown(), "{out:?}");
    }
}
