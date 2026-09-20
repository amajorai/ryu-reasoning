//! The policy DSL: a small infix language that compiles to [`crate::logic::Expr`].
//!
//! Rules and claims are written the way a person would write them —
//!
//! ```text
//! is_manager and tenure_months >= 12 -> vacation_days <= 30
//! status == "terminated" -> not eligible_for_bonus
//! discount <= 0.15
//! ```
//!
//! — because two very different authors have to produce this text: a human editing a
//! rule in the policy editor, and the extraction model turning an answer into claims.
//! A JSON syntax tree is fine for the first and terrible for the second; an infix
//! line is good for both and is what the UI round-trips.
//!
//! # Typed, and strict about it
//!
//! Parsing happens against the declared variables ([`Env`]), so the parser knows that
//! `status` is an enum and `tenure_months` is an integer. That is what lets it reject
//! — at authoring time, with a message — the three mistakes an extraction model
//! actually makes:
//!
//! * naming a variable that does not exist (`tenur_months`), rather than silently
//!   inventing one and proving something about nothing;
//! * comparing an enum to a number, or a bool to a string;
//! * multiplying two variables (`salary * rate`), which would leave the linear
//!   fragment the solver can decide.
//!
//! An error here is a *good* outcome: it becomes a `no_translations` finding, which
//! says "I could not formalize this", instead of a confident wrong verdict.

use std::fmt;

use crate::logic::{Expr, Linear, VarType};
use crate::num::Rat;
use crate::solver::Env;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at character {})", self.message, self.position)
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Number(Rat),
    Str(String),
    // punctuation / operators
    LParen,
    RParen,
    Plus,
    Minus,
    Star,
    Slash,
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
    Arrow,
    And,
    Or,
    Not,
    Iff,
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn err(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            position: self.pos,
        }
    }

    fn tokens(mut self) -> Result<Vec<(Tok, usize)>, ParseError> {
        let mut out = Vec::new();
        loop {
            while self.pos < self.src.len() && (self.src[self.pos] as char).is_whitespace() {
                self.pos += 1;
            }
            if self.pos >= self.src.len() {
                return Ok(out);
            }
            let start = self.pos;
            let c = self.src[self.pos] as char;
            let tok = match c {
                '(' => {
                    self.pos += 1;
                    Tok::LParen
                }
                ')' => {
                    self.pos += 1;
                    Tok::RParen
                }
                '+' => {
                    self.pos += 1;
                    Tok::Plus
                }
                '*' => {
                    self.pos += 1;
                    Tok::Star
                }
                '/' => {
                    self.pos += 1;
                    Tok::Slash
                }
                '-' => {
                    if self.peek(1) == Some('>') {
                        self.pos += 2;
                        Tok::Arrow
                    } else {
                        self.pos += 1;
                        Tok::Minus
                    }
                }
                '<' => {
                    if self.peek(1) == Some('=') {
                        self.pos += 2;
                        Tok::Le
                    } else if self.peek(1) == Some('-') && self.peek(2) == Some('>') {
                        self.pos += 3;
                        Tok::Iff
                    } else {
                        self.pos += 1;
                        Tok::Lt
                    }
                }
                '>' => {
                    if self.peek(1) == Some('=') {
                        self.pos += 2;
                        Tok::Ge
                    } else {
                        self.pos += 1;
                        Tok::Gt
                    }
                }
                '=' => {
                    if self.peek(1) == Some('=') {
                        self.pos += 2;
                    } else if self.peek(1) == Some('>') {
                        self.pos += 2;
                        out.push((Tok::Arrow, start));
                        continue;
                    } else {
                        self.pos += 1;
                    }
                    Tok::Eq
                }
                '!' => {
                    if self.peek(1) == Some('=') {
                        self.pos += 2;
                        Tok::Ne
                    } else {
                        self.pos += 1;
                        Tok::Not
                    }
                }
                '&' => {
                    self.pos += if self.peek(1) == Some('&') { 2 } else { 1 };
                    Tok::And
                }
                '|' => {
                    self.pos += if self.peek(1) == Some('|') { 2 } else { 1 };
                    Tok::Or
                }
                '"' | '\'' => self.lex_string(c)?,
                _ if c.is_ascii_digit() => self.lex_number()?,
                _ if c.is_ascii_alphabetic() || c == '_' => self.lex_word(),
                _ => return Err(self.err(format!("unexpected character '{c}'"))),
            };
            out.push((tok, start));
        }
    }

    fn peek(&self, ahead: usize) -> Option<char> {
        self.src.get(self.pos + ahead).map(|b| *b as char)
    }

    fn lex_string(&mut self, quote: char) -> Result<Tok, ParseError> {
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.src.len() && self.src[self.pos] as char != quote {
            self.pos += 1;
        }
        if self.pos >= self.src.len() {
            return Err(self.err("unterminated string literal"));
        }
        let text = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
        self.pos += 1;
        Ok(Tok::Str(text))
    }

    fn lex_number(&mut self) -> Result<Tok, ParseError> {
        let start = self.pos;
        while self.peek(0).is_some_and(|c| c.is_ascii_digit() || c == '_') {
            self.pos += 1;
        }
        let mut scale = 0u32;
        if self.peek(0) == Some('.') && self.peek(1).is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
            while self.peek(0).is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
                scale += 1;
            }
        }
        let raw: String = String::from_utf8_lossy(&self.src[start..self.pos])
            .chars()
            .filter(|c| *c != '_' && *c != '.')
            .collect();
        let digits: i128 = raw
            .parse()
            .map_err(|_| self.err("number is too large to represent exactly"))?;
        // Exact: `0.15` becomes 15/100, never a binary float.
        let den = 10i128
            .checked_pow(scale)
            .ok_or_else(|| self.err("number has too many decimal places"))?;
        let value = Rat::new(digits, den).ok_or_else(|| self.err("number is not representable"))?;
        Ok(Tok::Number(value))
    }

    fn lex_word(&mut self) -> Tok {
        let start = self.pos;
        while self
            .peek(0)
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        {
            self.pos += 1;
        }
        let word = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
        match word.to_ascii_lowercase().as_str() {
            "and" => Tok::And,
            "or" => Tok::Or,
            "not" => Tok::Not,
            "implies" => Tok::Arrow,
            "iff" => Tok::Iff,
            _ => Tok::Ident(word),
        }
    }
}

/// A partially-parsed operand. The parser tracks what *sort* it has so a comparison
/// can be resolved against the declared variable types.
#[derive(Debug, Clone)]
enum Node {
    Formula(Expr),
    Term(Linear),
    Str(String),
    EnumRef(String),
}

struct Parser<'a> {
    toks: Vec<(Tok, usize)>,
    idx: usize,
    env: &'a Env,
}

impl<'a> Parser<'a> {
    fn err(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            position: self.toks.get(self.idx).map(|(_, p)| *p).unwrap_or(0),
        }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.idx).map(|(t, _)| t)
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if self.peek() == Some(want) {
            self.idx += 1;
            true
        } else {
            false
        }
    }

    // implies := or ( "->" implies )?   — right associative, lowest precedence
    fn parse_implies(&mut self) -> Result<Node, ParseError> {
        let left = self.parse_or()?;
        if self.eat(&Tok::Arrow) {
            let right = self.parse_implies()?;
            return Ok(Node::Formula(
                self.as_formula(left)?.implies(self.as_formula(right)?),
            ));
        }
        if self.eat(&Tok::Iff) {
            let right = self.parse_implies()?;
            return Ok(Node::Formula(
                self.as_formula(left)?.iff(self.as_formula(right)?),
            ));
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<Node, ParseError> {
        let mut items = vec![self.parse_and()?];
        while self.eat(&Tok::Or) {
            items.push(self.parse_and()?);
        }
        if items.len() == 1 {
            return Ok(items.pop().expect("length checked"));
        }
        let formulas = items
            .into_iter()
            .map(|n| self.as_formula(n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Node::Formula(Expr::or(formulas)))
    }

    fn parse_and(&mut self) -> Result<Node, ParseError> {
        let mut items = vec![self.parse_not()?];
        while self.eat(&Tok::And) {
            items.push(self.parse_not()?);
        }
        if items.len() == 1 {
            return Ok(items.pop().expect("length checked"));
        }
        let formulas = items
            .into_iter()
            .map(|n| self.as_formula(n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Node::Formula(Expr::and(formulas)))
    }

    fn parse_not(&mut self) -> Result<Node, ParseError> {
        if self.eat(&Tok::Not) {
            let inner = self.parse_not()?;
            return Ok(Node::Formula(self.as_formula(inner)?.not()));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Node, ParseError> {
        let left = self.parse_sum()?;
        let op = match self.peek() {
            Some(Tok::Le) => CmpOp::Le,
            Some(Tok::Lt) => CmpOp::Lt,
            Some(Tok::Ge) => CmpOp::Ge,
            Some(Tok::Gt) => CmpOp::Gt,
            Some(Tok::Eq) => CmpOp::Eq,
            Some(Tok::Ne) => CmpOp::Ne,
            _ => return Ok(left),
        };
        self.idx += 1;
        let right = self.parse_sum()?;
        self.build_comparison(left, op, right)
    }

    fn build_comparison(&self, left: Node, op: CmpOp, right: Node) -> Result<Node, ParseError> {
        match (&left, &right) {
            // enum == "value"
            (Node::EnumRef(var), Node::Str(value)) | (Node::Str(value), Node::EnumRef(var)) => {
                self.check_enum_value(var, value)?;
                let atom = Expr::EnumEq(var.clone(), value.clone());
                match op {
                    CmpOp::Eq => Ok(Node::Formula(atom)),
                    CmpOp::Ne => Ok(Node::Formula(atom.not())),
                    _ => Err(self.err(format!(
                        "'{var}' is an enum, so it can only be compared with == or !="
                    ))),
                }
            }
            (Node::Term(a), Node::Term(b)) => {
                let diff = a
                    .sub(b)
                    .ok_or_else(|| self.err("arithmetic overflowed while normalizing"))?;
                let neg = diff
                    .neg()
                    .ok_or_else(|| self.err("arithmetic overflowed while normalizing"))?;
                let expr = match op {
                    CmpOp::Le => Expr::Le(diff),
                    CmpOp::Lt => Expr::Lt(diff),
                    CmpOp::Ge => Expr::Le(neg),
                    CmpOp::Gt => Expr::Lt(neg),
                    // a == b  ⇔  a - b <= 0  ∧  b - a <= 0
                    CmpOp::Eq => Expr::and(vec![Expr::Le(diff), Expr::Le(neg)]),
                    // a != b  ⇔  a - b < 0  ∨  b - a < 0
                    CmpOp::Ne => Expr::or(vec![Expr::Lt(diff), Expr::Lt(neg)]),
                };
                Ok(Node::Formula(expr))
            }
            (Node::Formula(a), Node::Formula(b)) => match op {
                CmpOp::Eq => Ok(Node::Formula(a.clone().iff(b.clone()))),
                CmpOp::Ne => Ok(Node::Formula(a.clone().iff(b.clone()).not())),
                _ => Err(self.err("booleans can only be compared with == or !=")),
            },
            (Node::EnumRef(var), _) | (_, Node::EnumRef(var)) => Err(self.err(format!(
                "'{var}' is an enum and must be compared with one of its declared values, \
                 written in quotes"
            ))),
            (Node::Str(s), _) | (_, Node::Str(s)) => Err(self.err(format!(
                "the text \"{s}\" can only be compared with an enum variable"
            ))),
            _ => Err(self.err("these two operands have different types")),
        }
    }

    fn check_enum_value(&self, var: &str, value: &str) -> Result<(), ParseError> {
        if let Some(VarType::Enum { values }) = self.env.get(var) {
            if !values.iter().any(|v| v == value) {
                return Err(self.err(format!(
                    "\"{value}\" is not a declared value of '{var}' (declared: {})",
                    values.join(", ")
                )));
            }
        }
        Ok(())
    }

    fn parse_sum(&mut self) -> Result<Node, ParseError> {
        let mut left = self.parse_product()?;
        loop {
            let negate = if self.eat(&Tok::Plus) {
                false
            } else if self.eat(&Tok::Minus) {
                true
            } else {
                return Ok(left);
            };
            let right = self.parse_product()?;
            let (a, b) = (self.as_term(left)?, self.as_term(right)?);
            let combined = if negate { a.sub(&b) } else { a.add(&b) };
            left = Node::Term(combined.ok_or_else(|| self.err("arithmetic overflowed"))?);
        }
    }

    fn parse_product(&mut self) -> Result<Node, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let divide = if self.eat(&Tok::Star) {
                false
            } else if self.eat(&Tok::Slash) {
                true
            } else {
                return Ok(left);
            };
            let right = self.parse_unary()?;
            let (a, b) = (self.as_term(left)?, self.as_term(right)?);
            // Linear only: at least one side must be a constant, or the result leaves
            // the fragment the solver can decide.
            let combined = if divide {
                if !b.is_constant() {
                    return Err(self
                        .err("division by a variable is not supported — rules must stay linear"));
                }
                a.scale(
                    Rat::ONE
                        .div(b.constant)
                        .ok_or_else(|| self.err("division by zero"))?,
                )
            } else if a.is_constant() {
                b.scale(a.constant)
            } else if b.is_constant() {
                a.scale(b.constant)
            } else {
                return Err(self.err(
                    "two variables cannot be multiplied — rules must stay linear (multiply by a \
                     number instead)",
                ));
            };
            left = Node::Term(combined.ok_or_else(|| self.err("arithmetic overflowed"))?);
        }
    }

    fn parse_unary(&mut self) -> Result<Node, ParseError> {
        if self.eat(&Tok::Minus) {
            let inner = self.parse_unary()?;
            let term = self.as_term(inner)?;
            return Ok(Node::Term(
                term.neg()
                    .ok_or_else(|| self.err("arithmetic overflowed"))?,
            ));
        }
        if self.eat(&Tok::Plus) {
            return self.parse_unary();
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Node, ParseError> {
        let Some(tok) = self.peek().cloned() else {
            return Err(self.err("expression ended early"));
        };
        match tok {
            Tok::LParen => {
                self.idx += 1;
                let inner = self.parse_implies()?;
                if !self.eat(&Tok::RParen) {
                    return Err(self.err("missing ')'"));
                }
                Ok(inner)
            }
            Tok::Number(v) => {
                self.idx += 1;
                Ok(Node::Term(Linear::constant(v)))
            }
            Tok::Str(s) => {
                self.idx += 1;
                Ok(Node::Str(s))
            }
            Tok::Ident(name) => {
                self.idx += 1;
                match name.to_ascii_lowercase().as_str() {
                    "true" => return Ok(Node::Formula(Expr::Const(true))),
                    "false" => return Ok(Node::Formula(Expr::Const(false))),
                    _ => {}
                }
                match self.env.get(&name) {
                    Some(VarType::Bool) => Ok(Node::Formula(Expr::Bool(name))),
                    Some(VarType::Int | VarType::Real) => Ok(Node::Term(Linear::var(name))),
                    Some(VarType::Enum { .. }) => Ok(Node::EnumRef(name)),
                    None => Err(self.err(format!(
                        "'{name}' is not a declared variable of this policy"
                    ))),
                }
            }
            other => Err(self.err(format!("unexpected token {other:?}"))),
        }
    }

    fn as_formula(&self, node: Node) -> Result<Expr, ParseError> {
        match node {
            Node::Formula(e) => Ok(e),
            Node::Term(l) => Err(self.err(format!(
                "'{}' is a number, not a condition — compare it with something",
                l.render()
            ))),
            Node::Str(s) => {
                Err(self.err(format!("the text \"{s}\" is not a condition on its own")))
            }
            Node::EnumRef(v) => Err(self.err(format!(
                "'{v}' is an enum — write something like {v} == \"...\""
            ))),
        }
    }

    fn as_term(&self, node: Node) -> Result<Linear, ParseError> {
        match node {
            Node::Term(l) => Ok(l),
            Node::Formula(_) => Err(self.err("a condition cannot be used inside arithmetic")),
            Node::Str(s) => Err(self.err(format!("the text \"{s}\" cannot be used in arithmetic"))),
            Node::EnumRef(v) => {
                Err(self.err(format!("'{v}' is an enum and cannot be used in arithmetic")))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
}

/// Parse one DSL line into a formula, checked against the declared variables.
pub fn parse(src: &str, env: &Env) -> Result<Expr, ParseError> {
    let toks = Lexer::new(src).tokens()?;
    if toks.is_empty() {
        return Err(ParseError {
            message: "the expression is empty".into(),
            position: 0,
        });
    }
    let mut parser = Parser { toks, idx: 0, env };
    let node = parser.parse_implies()?;
    if parser.idx < parser.toks.len() {
        return Err(parser.err("unexpected trailing input"));
    }
    parser.as_formula(node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::Variable;
    use crate::solver::{decide, Budget, Env};

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
                name: "rate".into(),
                ty: VarType::Real,
                description: String::new(),
            },
            Variable {
                name: "status".into(),
                ty: VarType::Enum {
                    values: vec!["active".into(), "leave".into()],
                },
                description: String::new(),
            },
        ])
    }

    #[test]
    fn implication_and_conjunction_parse() {
        let e = parse("is_manager and tenure_months >= 12 -> rate <= 0.5", &env()).unwrap();
        // Well-formed and decidable is the contract; the exact shape is normalized.
        assert!(decide(&env(), &[e], &Budget::default()).is_sat());
    }

    #[test]
    fn unknown_variable_is_rejected_by_name() {
        let err = parse("tenur_months >= 12", &env()).unwrap_err();
        assert!(err.message.contains("tenur_months"), "{}", err.message);
    }

    #[test]
    fn undeclared_enum_value_is_rejected() {
        let err = parse("status == \"retired\"", &env()).unwrap_err();
        assert!(err.message.contains("retired"), "{}", err.message);
    }

    #[test]
    fn nonlinear_multiplication_is_rejected() {
        let err = parse("tenure_months * rate <= 10", &env()).unwrap_err();
        assert!(err.message.contains("linear"), "{}", err.message);
    }

    #[test]
    fn enum_compared_to_a_number_is_rejected() {
        let err = parse("status >= 3", &env()).unwrap_err();
        assert!(!err.message.is_empty());
    }

    #[test]
    fn decimals_are_exact() {
        let e = parse("rate == 0.1 + 0.2 and rate <= 0.3", &env()).unwrap();
        assert!(
            decide(&env(), &[e], &Budget::default()).is_sat(),
            "0.1 + 0.2 must be exactly 0.3"
        );
    }

    #[test]
    fn precedence_puts_implication_last() {
        // Parsed as `(a and b) -> c`, so a model with a=false satisfies it.
        let e = parse(
            "is_manager and tenure_months >= 12 -> tenure_months >= 24",
            &env(),
        )
        .unwrap();
        let extra = parse("not is_manager", &env()).unwrap();
        assert!(decide(&env(), &[e, extra], &Budget::default()).is_sat());
    }

    #[test]
    fn a_bare_number_is_not_a_condition() {
        let err = parse("tenure_months", &env()).unwrap_err();
        assert!(err.message.contains("not a condition"), "{}", err.message);
    }

    #[test]
    fn alternative_spellings_are_accepted() {
        for src in [
            "is_manager && tenure_months > 1",
            "is_manager AND tenure_months > 1",
            "is_manager implies tenure_months > 1",
            "!is_manager || tenure_months > 1",
        ] {
            assert!(parse(src, &env()).is_ok(), "failed to parse: {src}");
        }
    }
}
