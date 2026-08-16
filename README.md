<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./icon-dark.png" />
    <img src="./icon-light.png" alt="Automated Reasoning" width="144" />
  </picture>
</p>

<div align="center">

# Automated Reasoning

</div>

Decide whether an answer FOLLOWS from a written policy using a decision procedure written from scratch (exact rational arithmetic, finite-domain enumeration, Fourier–Motzkin, branch-and-bound), so a verdict carries the minimal rule set that decided it and a concrete counterexample.

> **The public home of `ryu-reasoning`.** Source, builds, and releases live here —
> binaries for every platform are attached to each release.
>
> This tree is generated from the Ryu monorepo, so commits pushed here
> directly are replaced on the next sync. **Pull requests are welcome** —
> open them here and they are ported into the monorepo, then flow back out.
> Ryu as a whole: https://github.com/amajorai/ryu

## Install

**App:** [Install](ryu://apps/@ryu/reasoning) (opens the Ryu desktop app and asks you to confirm)

**CLI:**

```bash
ryu apps add @ryu/reasoning
```

**Crate:**

```bash
cargo install ryu-reasoning
```

Prebuilt binaries for every platform are attached to [each release](https://github.com/amajorai/ryu/releases).

## License

Apache-2.0 — see [LICENSE](./LICENSE).

## What a verdict means

| Verdict | Meaning |
|---|---|
| `valid` | The policy **proves** the claim. It cannot be false here. |
| `invalid` | The policy **contradicts** the claim, and here is a counterexample. |
| `satisfiable` | Consistent with the policy, but not implied by it. Usually a missing *fact*, not a missing rule. |
| `impossible` | The policy and the situation contradict **each other**, so nothing follows from them. |
| `no_translations` | The answer said nothing this policy can express. |
| `translation_ambiguous` | The sentence has more than one reading and the readings disagree. |
| `too_complex` | A solver budget ran out. Explicitly **not** a verdict. |

Three of those exist because collapsing them is how a checker becomes untrustworthy.

**`impossible` is the load-bearing one.** If a policy's rules conflict, then
`policy ∧ premises ∧ ¬claim` is unsatisfiable for *every* claim — so a checker that
goes straight to "can the claim fail?" stamps anything at all as proven. A policy with
two contradictory rules would silently pass everything, which is worse than having no
checker. The contradiction is reported as its own outcome, with the smallest
conflicting rule set named. `POST /policies/:id/analyze` finds it before a check ever
runs.

**`too_complex` is never a verdict.** The fragment below is decidable, but decidable
is not the same as fast: Fourier–Motzkin is doubly exponential in eliminated variables
and the boolean search is exponential in atoms. Every loop that can grow decrements a
budget, and exhausting one yields `too_complex`. A solver that answered "unsatisfiable"
because it gave up would report a hallucination as fact.

## Where the explanations come from

Not from a model — `@ryu/double-check` is already the "have a second model look at it"
product, and it can be as wrong as the first. Everything a finding carries is read
back out of the solver:

- **which rules mattered** — a minimal unsatisfiable core, computed by deleting one
  constraint at a time and re-asking. What survives has no removable member.
- **unstated assumptions** — the premise members of that core: the facts the answer
  quietly relies on but never states.
- **the counterexample** — a concrete assignment satisfying the policy in which the
  claim is false. It is a witness, so it cannot be a hallucination.
- **what would have to change** — a model of `policy ∧ claim`, diffed against the
  counterexample. The differing variables are the conditions under which the answer
  would have been right.

## The policy language

Rules and claims are one line of infix text, type-checked against the declared
variables:

```
is_manager and tenure_months >= 12 -> vacation_days <= 30
status == "terminated" -> not eligible_for_bonus
refund_amount <= order_total * 0.5
```

Variables are `bool`, `int`, `real`, or an `enum` over a declared set of strings.
Arithmetic is linear only — `salary * rate` is refused at authoring time, because it
would leave the fragment the solver can decide completely. A rule naming an undeclared
variable is refused too; that error is a *good* outcome, because it becomes
"I could not formalize this" rather than a confident wrong verdict.

Numbers are exact rationals end to end. `0.1 + 0.2 <= 0.3` is true here, which it is
not in floating point — a policy about a 30% cap should not reject a legal 30%.

A variable's `description` is not decoration: it is the text the extraction step reads
when deciding which variable a sentence is about. Synonyms and example phrasings
belong in it, and it is the single most influential field in a policy.

## Surfaces

**Companion** (`ui/`) — author policies, interrogate them, run the playground. Drives
the sidecar through one generic `reasoning.request` bridge forwarder.

**HTTP** — Core proxies `/api/reasoning/*` to the sidecar:

| Route | Model? | |
|---|---|---|
| `GET/POST /policies`, `GET/PUT/DELETE /policies/:id` | no | CRUD |
| `POST /policies/:id/analyze` | no | consistency, redundancy, unused variables |
| `POST /solve` | **no** | decide claims already written as formulas |
| `POST /check` | yes | decide claims taken from prose |
| `POST /policies/draft` | yes | propose variables and rules from a document |
| `POST /policies/:id/tests/run` | yes | run the saved question/answer suite |

**Agents and workflows** — the same engine is an MCP server (`ryu-reasoning mcp`),
declared in the manifest's own `mcp_servers`. `reasoning.solve` is the id a workflow
`mcp` node takes, and it is deterministic and offline, so a branch gated on a proof
always takes the same path for the same inputs. `reasoning.policies` lists a policy's
vocabulary so an agent can write formulas for it.

**Per-turn** — the composer's *Check against policy* toggle runs
`contributes.turn_hooks` after each answer and reports anything the policy
contradicts. Set the policy id in Settings → Automated Reasoning.

The hook reaches the solver through `host.runAgent` with the conversation's own
`agent_id`, because the plugin sandbox has no HTTP by design. That path is
conditional: only a named agent **and** a live agent runner get the real chat path,
where `reasoning.check` exists at all; without a runner, delegation falls back to a
single tool-less completion. So the sub-agent is required to end with a
`SOLVER: <verdict>` line copied out of the tool's own output, and a reply without one
is reported as *"did not run"* rather than as a finding. A model with no tool cannot
produce that marker honestly, and the failure it does produce is legible. This is the
one surface whose behaviour depends on how the node is configured; `/solve` and the
MCP tools do not.

## Layout

```
backend/          the crate (also the MCP server; ZERO dependency on apps/core)
  src/num.rs      exact rational arithmetic
  src/logic.rs    the formula language and its normal form
  src/parser.rs   the authoring DSL, type-checked against the variables
  src/solver.rs   the decision procedure, with budgets
  src/verdict.rs  entailment ordering, unsat cores, solver-derived suggestions
  src/policy.rs   the policy document and its self-analysis
  src/store.rs    one JSON file per policy under $RYU_DIR/reasoning
  src/host.rs     the authenticated model callback into Core
  src/translate.rs prose ↔ logic, the only two model calls
  src/api.rs      the HTTP surface
  src/mcp.rs      the same engine over MCP stdio
ui/               the companion (vite + react, built to one self-contained HTML)
hooks/check.js    the post-answer turn hook
```

Rebuild the companion bundle Core ships with:

```
bun run --cwd apps-store/reasoning/ui build
cp apps-store/reasoning/ui/dist/index.html \
   apps/core/src/plugin_manifest/fixtures/reasoning.ui.html
```

## Prompt injection

The answer being checked is untrusted text, and it is fed to the extraction model.
Three properties contain that:

1. **The model cannot state a verdict.** Its entire output is premises and claims; the
   verdict is computed afterwards by the solver, which never sees the prose.
2. **The output is type-checked before use.** Every line goes through the parser
   against the declared variables, so an injected claim can only be a formula over
   this policy's own vocabulary.
3. **A trivially-true claim proves nothing.** `1 <= 2` parses, but comes back valid
   with *no responsible rules* — which reads as "this claim says nothing", not "the
   policy approved it".

The residual risk is a *mis*translation. That is why every finding shows both the
sentence it came from and the formula it decided: a wrong check is visible to the
author, which is the whole reason the middle of the pipeline is a solver.

## Prior art

The product shape (draft a policy from a document, edit it in natural language, check
answers against it) follows AWS's Automated Reasoning checks for Bedrock Guardrails.
Nothing here calls that service or any other: the solver, the policy language, the
verdict vocabulary, and the explanation machinery are implemented in `backend/`. The
richer result set (`satisfiable`, `impossible`, `translation_ambiguous`,
`too_complex`) is this app's own, not a mirror of theirs.
