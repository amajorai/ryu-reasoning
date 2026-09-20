use std::collections::BTreeSet;

use ryu_reasoning::api::evaluate;
use ryu_reasoning::logic::{VarType, Variable};
use ryu_reasoning::policy::{Policy, Rule};
use ryu_reasoning::solver::Budget;
use ryu_reasoning::translate::{Extracted, Extraction};
use ryu_reasoning::verdict::Verdict;
use serde::Deserialize;

const FIXTURE: &str = include_str!("../fixtures/conformance.json");

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    id: String,
    atoms: Vec<String>,
    rules: Vec<Formula>,
    premises: Vec<Formula>,
    claim: Formula,
    expected: Verdict,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
enum Formula {
    #[serde(rename = "atom")]
    Atom { name: String },
    #[serde(rename = "not")]
    Not { formula: Box<Formula> },
    #[serde(rename = "and")]
    And {
        left: Box<Formula>,
        right: Box<Formula>,
    },
    #[serde(rename = "or")]
    Or {
        left: Box<Formula>,
        right: Box<Formula>,
    },
    #[serde(rename = "implies")]
    Implies {
        left: Box<Formula>,
        right: Box<Formula>,
    },
}

impl Formula {
    fn render(&self) -> String {
        match self {
            Self::Atom { name } => name.clone(),
            Self::Not { formula } => format!("not ({})", formula.render()),
            Self::And { left, right } => format!("({} and {})", left.render(), right.render()),
            Self::Or { left, right } => format!("({} or {})", left.render(), right.render()),
            Self::Implies { left, right } => format!("({} -> {})", left.render(), right.render()),
        }
    }

    fn atoms(&self, output: &mut BTreeSet<String>) {
        match self {
            Self::Atom { name } => {
                output.insert(name.clone());
            }
            Self::Not { formula } => formula.atoms(output),
            Self::And { left, right }
            | Self::Or { left, right }
            | Self::Implies { left, right } => {
                left.atoms(output);
                right.atoms(output);
            }
        }
    }
}

#[test]
fn production_reasoning_matches_the_shared_conformance_fixture() {
    let fixture: FixtureFile = serde_json::from_str(FIXTURE).expect("valid conformance fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.cases.len(), 4);

    let mut ids = BTreeSet::new();
    for case in fixture.cases {
        assert!(
            ids.insert(case.id.clone()),
            "duplicate fixture id: {}",
            case.id
        );
        let declared_atoms: BTreeSet<_> = case.atoms.iter().cloned().collect();
        assert_eq!(
            declared_atoms.len(),
            case.atoms.len(),
            "duplicate atoms: {}",
            case.id
        );

        let mut used_atoms = BTreeSet::new();
        for formula in &case.rules {
            formula.atoms(&mut used_atoms);
        }
        for formula in &case.premises {
            formula.atoms(&mut used_atoms);
        }
        case.claim.atoms(&mut used_atoms);
        assert!(
            used_atoms.is_subset(&declared_atoms),
            "{} uses an undeclared atom: {:?}",
            case.id,
            used_atoms.difference(&declared_atoms).collect::<Vec<_>>()
        );

        let variables = case
            .atoms
            .iter()
            .map(|name| Variable {
                name: name.clone(),
                ty: VarType::Bool,
                description: String::new(),
            })
            .collect();
        let rules = case
            .rules
            .iter()
            .enumerate()
            .map(|(index, formula)| {
                let expression = formula.render();
                Rule {
                    id: format!("rule-{}", index + 1),
                    statement: expression.clone(),
                    expression,
                    enabled: true,
                }
            })
            .collect();
        let policy = Policy {
            id: case.id.clone(),
            name: case.id.clone(),
            description: "Shared Lean conformance fixture".to_owned(),
            version: 1,
            variables,
            rules,
            tests: Vec::new(),
            source_document: None,
            created_at: String::new(),
            updated_at: String::new(),
        };
        let extraction = Extraction {
            premises: case
                .premises
                .iter()
                .map(|formula| {
                    let expression = formula.render();
                    Extracted {
                        statement: expression.clone(),
                        expression,
                        alternatives: Vec::new(),
                    }
                })
                .collect(),
            claims: vec![{
                let expression = case.claim.render();
                Extracted {
                    statement: expression.clone(),
                    expression,
                    alternatives: Vec::new(),
                }
            }],
            notes: Vec::new(),
        };

        let report = evaluate(&policy, &extraction, &Budget::default());
        assert_eq!(
            report.result, case.expected,
            "{}: expected {:?}, got {:?}; notes: {:?}",
            case.id, case.expected, report.result, report.notes
        );
    }
}
