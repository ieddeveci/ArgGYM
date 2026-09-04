# Defeasible Reasoning — a getting-started primer

Goal: enough understanding of defeasible reasoning, ASPIC+, and grounded semantics to work on the ArgGYM benchmark. Each concept is tied to where it lives in this repo.

## 1. Deductive vs defeasible reasoning

**Deductive** reasoning is what classical logic does: if the premises are true, the conclusion is *guaranteed*. "All squares have four sides; this is a square; therefore it has four sides." Adding new information can never break a valid deduction. This property is called **monotonicity**.

**Defeasible** reasoning is everyday, presumption-based reasoning: conclusions are drawn *by default* and can be **retracted** when new information arrives. The classic example (which appears in this repo's templates, `aspic_content.py`):

- Tweety is a bird → *presumably* Tweety flies.
- Now you learn Tweety is a penguin → conclusion retracted: Tweety does not fly.

That's **non-monotonic**: more information can *remove* conclusions. Most real reasoning (law, medicine, common sense) works this way — you act on defaults and revise. Classical logic can't model this: adding "penguin" to a deductive theory containing "birds fly" just makes it inconsistent.

The research question behind ArgGYM: LLMs are heavily trained on math/code-style *deductive* chains — how good are they at reasoning where conclusions must survive attacks and be revised?

## 2. Formal argumentation: the three-layer cake

The standard way to formalize defeasible reasoning (Dung 1995 + successors) has three layers:

1. **Build arguments** from a knowledge base (premises + rules). An argument is a tree: premises at the leaves, rule applications up to a conclusion.
2. **Find attacks** between arguments (they contradict each other somewhere).
3. **Evaluate**: given the whole attack graph, decide which arguments are acceptable. This step is called applying a **semantics**.

ASPIC+ is the framework used here for layers 1–2; **grounded semantics** is used for layer 3.

## 3. ASPIC+ building blocks

An ASPIC+ **theory** consists of (see `ASPICFramework` in `aspic_engine.py`, and the DSL in `aspic_dsl.py`):

| Element | DSL | Meaning |
|---|---|---|
| **Axiom** | `[axiom: x]` | A certain fact; cannot be attacked. |
| **Ordinary premise** | `[premise: x]` | An assumed fact; *can* be attacked. |
| **Strict rule** | `[strict s1: a AND b -> c]` | If a, b hold, c holds *necessarily* (deductive). |
| **Defeasible rule** | `[defeasible d1: a AND b => c]` | If a, b hold, c holds *by default* (can be defeated). |
| **Rule preference** | `[prefer_rule: d1 > d2]` | d1 is stronger than d2. |
| **Premise preference** | `[prefer_premise: p > q]` | p is stronger than q. |

Negation: `-x` is the contrary of `x` (handled in `contrary()` in `aspic_engine.py`).

Tweety in the DSL:

```
[premise: bird]
[defeasible d1: bird => fly]
[premise: penguin]
[defeasible d2: penguin => -fly]
[prefer_rule: d2 > d1]
```

An **argument** chains these: {premise bird, apply d1} is an argument for `fly`. An argument is **firm** if it uses only axioms and strict rules — nothing defeasible anywhere — so it can't lose to a defeasible opponent:

```
[axiom: penguin]                  [premise: bird]
[strict s1: penguin -> -fly]      [defeasible d1: bird => fly]
```
Engine verdict: `-fly: JUSTIFIED`, `fly: OVERRULED` — the firm argument wins with no preference needed.

*(All verdicts in this primer are real `aspic_engine` outputs, not hand-derived.)*

## 4. The three kinds of attack

Arguments attack each other in exactly three ways (this trichotomy shows up in the `attack` and `attackers_of` tasks):

**Rebut** — attack the *conclusion*: an argument for `x` and an argument for `-x` rebut each other. (Only conclusions of *defeasible* rules can be rebutted.)

```
[premise: clouds]    [defeasible d1: clouds => rain]
[premise: dry_wind]  [defeasible d2: dry_wind => -rain]
```
Verdict: `rain: UNDECIDED`, `-rain: UNDECIDED` — a symmetric clash with no preference resolves nothing (the premises `clouds`, `dry_wind` stay JUSTIFIED).

**Undermine** — attack a *premise*: an argument for `-p` attacks any argument using ordinary premise `p`. (Axioms can't be undermined — that's the point of axioms.)

```
[premise: witness_reliable]   [defeasible d1: witness_reliable => guilty]
[premise: -witness_reliable]
```
Verdict: everything UNDECIDED. Now add `[prefer_premise: -witness_reliable > witness_reliable]`: verdict becomes `guilty: OVERRULED`, `witness_reliable: OVERRULED` — knocking out a premise kills every argument built on it.

**Undercut** — attack the *rule itself*: derive `-d1` ("rule d1 doesn't apply here") to switch off d1 without denying its conclusion. "Birds fly" undercut by "this bird is covered in oil" — you're not claiming it can't fly for other reasons; you're saying this rule gives no support now.

```
[premise: bird]            [defeasible d1: bird => fly]
[premise: covered_in_oil]  [defeasible d2: covered_in_oil => -d1]
```
Verdict: `fly: OVERRULED`, `-d1: JUSTIFIED` — note the asymmetry: the undercut defeats without any preference, and d1's argument can't fight back.

> Audit note: an underminer whose `-p` is *derived by a rule* (not asserted as a premise) loses to the premise `p` under the default last-link ordering even when a premise preference favors it — different from the bare-premise case above. Surprising, and worth keeping in mind while the engine's ordering behaviour is still under audit.

## 5. From attack to defeat: preferences and orderings

An attack only **defeats** if the attacker is not weaker than its target. That's where preferences come in — and *how* an argument's overall strength is computed from its parts is the **ordering** (see `ORDERINGS` in `aspic_engine.py`):

- **Last-link**: an argument's strength is decided by its *final* defeasible rule. A preference on the last rule wins the conflict.
- **Weakest-link**: an argument is only as strong as its *weakest* element across the whole chain (rules and premises).
- The **elitist / democratic** variants differ in how *sets* of rules/premises are compared (weakest member vs. all members). The gym default is `last_link_elitist`.

The same theory can give different answers under different orderings — some tasks exploit exactly this:

```
[premise: p]  [defeasible d1: p => q]   [defeasible d2: q => r]
[premise: a]  [defeasible d3: a => -r]
[prefer_rule: d2 > d3]  [prefer_rule: d3 > d1]
```
- `last_link_elitist`: the r-argument's *last* rule is d2, and d2 > d3 → `r: JUSTIFIED`, `-r: OVERRULED`.
- `weakest_link_elitist`: the r-argument's chain contains d1, and d1 < d3 — the strong last rule no longer saves it → `r: UNDECIDED`, `-r: UNDECIDED`.

This is exactly what the `ordering_sensitivity` task probes.

## 6. Grounded semantics: who survives?

Once you have the attack/defeat graph, grounded semantics computes the *most skeptical* verdict. Intuition — an iterative fixpoint:

1. Accept every argument with no defeaters (label **IN**).
2. Reject everything defeated by an accepted argument (**OUT**).
3. Newly rejected attackers may free other arguments → accept them; repeat until nothing changes.
4. Whatever remains — typically symmetric conflicts with no preference to break the tie — stays **UNDEC** (undecided).

The set of IN arguments is the **grounded extension**. It always exists and is unique, which is what makes engine-verified gold answers possible. (Other semantics — preferred, stable — allow multiple "reasonable" outcomes; ArgGYM avoids that ambiguity.) Implementation: `argument_labels()` in `aspic_engine.py`, using PyArg's `get_grounded_extension`.

A key phenomenon to know: **reinstatement**. If A attacks B and C attacks A, then C "saves" B — B is back IN. Chains of attacks flip statuses alternately; several tasks (robustness, preference_construction) revolve around this.

```
[premise: bird]                [defeasible d1: bird => fly]
[premise: looks_like_penguin]  [defeasible d2: looks_like_penguin => -fly]
[prefer_rule: d2 > d1]
```
Verdict: `fly: OVERRULED`. Now add an undercutter of d2:
```
[premise: halloween_costume]   [defeasible d3: halloween_costume => -d2]
```
Verdict flips: `fly: JUSTIFIED`, `-fly: OVERRULED` — d3 knocks out d2's argument, which reinstates the original `fly` argument. Adding information changed a conclusion twice; that's non-monotonicity in action.

## 7. Claim statuses

Task answers are usually about *claims* (literals), not arguments. A claim's status aggregates over all arguments concluding it (`status_map()` in `aspic_engine.py`):

- **JUSTIFIED** — some argument for it is IN (survives every attack).
- **OVERRULED** — arguments exist, but all are OUT (defeated by stronger surviving arguments).
- **UNDECIDED** — best arguments are stuck in unresolved conflict.
- **UNSATISFIABLE** — no argument for it can be built at all.

These four labels are the answer vocabulary of `status_query`, the most central task. One theory showing all four:

```
[axiom: contract_signed]        [premise: terms_breached]
[defeasible d1: contract_signed AND terms_breached => liable]
[premise: signed_under_duress]  [defeasible d2: signed_under_duress => -liable]
[prefer_rule: d1 > d2]
[premise: ceo_resigned]         [defeasible d3: ceo_resigned => stock_drops]
[premise: record_profits]       [defeasible d4: record_profits => -stock_drops]
[defeasible d5: settlement_paid => case_closed]
```
Engine verdicts:
- `liable: JUSTIFIED` — attacked by d2's argument, but d1 > d2 so it survives (`-liable: OVERRULED`).
- `stock_drops: UNDECIDED` and `-stock_drops: UNDECIDED` — symmetric rebuttal, no preference.
- `case_closed: UNSATISFIABLE` — `settlement_paid` is never given, so no argument for it can even be built.
- (`contract_signed: JUSTIFIED` — axioms are trivially justified.)

## 8. The tools

- **PyArg** (`python-argumentation` on PyPI) — Python library implementing ASPIC+, abstract argumentation frameworks, and semantics. It is the *ground-truth engine*: `aspic_engine.py` is a thin wrapper that builds a PyArg `ArgumentationTheory` and reads off the grounded extension. If PyArg is wrong, gold answers are wrong, which is why auditing it is a standing task.
- **The bracket DSL** (`aspic_dsl.py`, spec shown to models in `prompting.py`) — the interchange format: theories are rendered into it, and constructive-task answers from models are parsed back out of it, applied to the theory, and re-verified by the engine.
- **Local LLM server (vLLM or Ollama)** — used by `build_kb.py`/`generate_argumentations.py` to author natural-language argument material (the "content" KB, `kb.json`) from seed claims in `claims.json`. LLMs author *surface text* only; correctness always comes from the engine. The chat client auto-detects which API the `--host` speaks.

## 9. How ArgGYM puts it together

- **Procedural generation**: `aspic_gym.py` samples random ASPIC+ theories from `TheoryConfig`, so items are unlimited and contamination-free.
- **Engine-verified**: every gold answer is computed by PyArg, and model answers to constructive tasks (build an attack, add a preference…) are *executed* in the engine and checked semantically — not string-matched.
- **Two surface modes**: *symbolic* (abstract atoms like `p`, `q`) vs *content* (plain-language statements drawn from `kb.json` / templates in `aspic_content.py`). Comparing the two isolates "can it do the logic" from "does the wording help or mislead".
- **Difficulty levels 1–15** (`levels.py`): knobs (theory size, conflicts, undercut probability…) grow with level; features unlock at set levels (`FEATURES`: conflict@2, axioms@3, strict rules@4, undercuts@5, …).
- **13 tasks** (`tasks/registry.py`), roughly grouped:
  - *Read the theory*: `status_query` (classify claim statuses), `claim_identification`, `attackers_of`.
  - *Build arguments*: `attack` (mount a defeating attack), `counter_argumentation`, `preference_construction` (add preferences to settle an undecided claim), `evidence_construction`, `enthymeme` (fill in missing element of an incomplete argument).
  - *Translate*: `formalization` (NL → DSL), `syntax` (format drill — likely excluded from the benchmark, kept for training).
  - *Revise beliefs*: `perturbation_prediction` (predict status changes after the theory is edited), `robustness`, `ordering_sensitivity` (same theory, different ordering).

## 10. End-to-end worked example (do this by hand once)

A realistic case, formalized from scratch and pushed all the way to a verdict.

### 10.1 The case in English

> A 34-year-old presents with a sore throat and fever. The throat culture comes
> back positive for *Streptococcus pyogenes*. Standard practice is to treat
> strep throat with amoxicillin. However, the patient's chart lists a penicillin
> allergy, and amoxicillin is a penicillin-class drug — so it should not be
> given. Where protocol and a safety contraindication conflict, the
> contraindication takes precedence.

Should the patient be prescribed amoxicillin?

### 10.2 Formalizing it: which sentence becomes what

The whole skill of using ASPIC+ is deciding, for each statement, *which of the
six element types it is*. Two questions settle almost every case:

- **Is it a fact or an inference step?** Facts become axioms/premises; "if …
  then …" steps become rules.
- **Could it turn out to be wrong?** If not, it's an axiom or a strict rule; if
  yes, it's an ordinary premise or a defeasible rule.

| Sentence | Element | Why |
|---|---|---|
| The culture is positive for strep. | `[axiom: strep_positive]` | A lab result taken as settled. Axioms cannot be undermined — that's the modelling claim we are making about it. |
| Strep throat is treated with amoxicillin. | `[defeasible d1: strep_positive => prescribe_amox]` | An inference step, and a *defeasible* one: it's the standard protocol, not a law of nature. Exceptions exist — which is the whole point. |
| The chart lists a penicillin allergy. | `[premise: penicillin_allergy]` | A fact, but a *disputable* one: chart entries are often self-reported and unverified. Ordinary premise, so it can be attacked later. |
| A penicillin allergy contraindicates amoxicillin. | `[defeasible d2: penicillin_allergy => -prescribe_amox]` | Another inference step. `-prescribe_amox` is the contrary of `prescribe_amox` — this is what sets up the conflict. |
| Contraindications outrank protocol. | `[prefer_rule: d2 > d1]` | Not a fact and not a rule — a statement about *which rule wins* when two collide. This is the ordering ≼, supplied as input. |

Note what is **not** in the theory: nothing states that amoxicillin is a
penicillin. The engine never needs it — that piece of world knowledge was used
by *us*, while writing d2, and then discarded. The engine sees opaque tokens: it
would behave identically if every atom were renamed. The same point holds for
Tweety in §3 — nothing in that theory connects `penguin` to `bird` either, and
the specificity intuition lives entirely in the asserted preference.

### 10.3 The theory

```
[axiom: strep_positive]
[defeasible d1: strep_positive => prescribe_amox]
[premise: penicillin_allergy]
[defeasible d2: penicillin_allergy => -prescribe_amox]
[prefer_rule: d2 > d1]
```

### 10.4 Build the arguments (layer 1)

Four arguments; premises are arguments too (a bare premise is a one-node tree):

| | Argument | Concludes |
|---|---|---|
| A1 | `strep_positive` | `strep_positive` |
| A2 | A1 + d1 | `prescribe_amox` |
| A3 | `penicillin_allergy` | `penicillin_allergy` |
| A4 | A3 + d2 | `-prescribe_amox` |

### 10.5 Find the attacks (layer 2)

A2 and A4 conclude contraries, so they **rebut** each other — and both
conclusions come from defeasible rules, so both directions of the rebuttal are
allowed.

Nothing attacks A1: `strep_positive` is an axiom, so it is immune to
undermining. Nothing attacks A3 either — no argument for `-penicillin_allergy`
exists in this theory.

### 10.6 Rebut → defeat (the ordering)

A mutual rebuttal decides nothing on its own. Under the default
`last_link_elitist` ordering, each argument's strength comes from its final
defeasible rule: A2's is d1, A4's is d2. We asserted d2 > d1, so:

- A4 defeats A2 ✔
- A2 does **not** defeat A4 ✘ (it is strictly weaker)

The symmetric attack has become an asymmetric defeat. That is the only thing
preferences ever do.

### 10.7 Grounded labelling (layer 3)

1. Arguments with no defeaters go **IN**: A1, A3, A4.
2. Anything defeated by an IN argument goes **OUT**: A2 (defeated by A4).
3. Nothing left to flip — done. No argument is UNDEC.

Statuses, aggregating arguments per claim — engine output:

```
prescribe_amox      OVERRULED     (its only argument, A2, is OUT)
-prescribe_amox     JUSTIFIED     (A4 is IN)
strep_positive      JUSTIFIED
penicillin_allergy  JUSTIFIED
```

**Do not give amoxicillin.** Note that `strep_positive` stays JUSTIFIED
throughout: the conflict was about what to *do*, not about the culture.

### 10.8 New information arrives

The allergy note is followed up: it was recorded from the patient's own report
in childhood and never confirmed by testing. Clinically this matters — a large
share of charted penicillin allergies do not hold up when tested.

How should this be modelled? Not as an attack on the *claim* `penicillin_allergy`
— we are not asserting the patient has no allergy. And not as an attack on the
conclusion `-prescribe_amox` either. What is in doubt is whether the
contraindication rule should fire on evidence this weak. That is exactly an
**undercut** — attack the rule itself:

```
[premise: allergy_unverified]
[defeasible d3: allergy_unverified => -d2]
```

Re-running: A5 = `allergy_unverified`, A6 = A5 + d3 concluding `-d2`. A6
undercuts A4. Undercuts need no preference — a rule that has been switched off
gives no support, and A4 cannot fight back.

Relabelling: A6 is undefeated → **IN**; A4 → **OUT**; and now A2's only defeater
is OUT, so A2 comes back **IN**. That is **reinstatement**. Engine output:

```
prescribe_amox      JUSTIFIED     (was OVERRULED)
-prescribe_amox     OVERRULED     (was JUSTIFIED)
-d2                 JUSTIFIED
```

The recommendation flipped twice — protocol says yes, allergy says no, "the
allergy is unverified" says yes again — with every step monotone-looking
locally and the *conclusion set* shrinking and growing. Adding facts removed a
conclusion. That is non-monotonicity, and it is what the benchmark is testing.

### 10.9 Why `strep_positive` was an axiom

Modelling choices have teeth. Suppose someone adds `[premise: -strep_positive]`
("a second lab disputes the culture"). Because `strep_positive` is an **axiom**,
it cannot be undermined, and the challenge simply loses:

```
strep_positive   JUSTIFIED       -strep_positive   OVERRULED
```

Demote it to `[premise: strep_positive]` and the same addition lands:

```
strep_positive   UNDECIDED       -strep_positive   UNDECIDED
```

Same sentences, different status, purely from the axiom/premise choice. When
you write a theory you are choosing what is open to dispute.

### 10.10 The same case under weakest-link

Everything above used the default `last_link_elitist`. Switch to either
weakest-link ordering and §10.7 changes:

```
prescribe_amox   UNDECIDED       -prescribe_amox   UNDECIDED
```

No recommendation at all. Why: weakest-link compares arguments on *both* their
rules and their ordinary premises, and A4 has to win — or at least tie — on both
to defeat A2. It wins on rules (d2 > d1) but loses on premises. A2 rests on an
axiom, so its set of ordinary premises is *empty*, which nothing can be weaker
than; A4 rests on the disputable `penicillin_allergy`. So A4 no longer defeats,
the rebuttal stays symmetric, and grounded semantics leaves both UNDEC.

Read as clinical advice that is arguably the more cautious reading: "your
contraindication rule is stronger, but it is built on a shakier fact, so this
does not settle the matter." The undercut stage (§10.8) is ordering-insensitive
— undercuts never consult preferences — and so is the axiom contrast in §10.9
apart from the same knock-on to `prescribe_amox`.

This is what the `ordering_sensitivity` task probes, and it is the strongest
argument for pinning the ordering in the prompt: the *same theory* has two
defensible answers, and only the declared ordering picks one.

*(All verdicts above are real `aspic_engine` outputs.)*

### 10.11 How this looks as a benchmark item

Given this theory, ArgGYM's `status_query` task asks for the status of
`prescribe_amox`; `attack` asks the model to *construct* the undercut in §10.8;
`perturbation_prediction` gives it the undercut and asks what changes;
`formalization` gives the English case from §10.1 and asks for the DSL of §10.3.
In *content* mode the atoms are replaced by glossed sentences ("the throat
culture is positive for strep") and the rules are numbered "Rule 1, Rule 2, …";
in *symbolic* mode they appear as bare atoms. The gold answer is whatever the
engine just printed — in both modes.

One honest difference from this example: here the preference *came from
somewhere* (a clinical guideline hierarchy). ArgGYM's generated items choose the
direction at random, deliberately, so that a model cannot skip the theory and
answer from world knowledge alone. See `preference-semantics-issues.md`.

The fastest way to build intuition: run the Flask playground (`python app.py`),
generate level-1 `status_query` items, and solve them by hand before checking
the engine's answer.

## 11. Reading list (optional, in order)

1. Modgil & Prakken, *"A general account of argumentation with preferences"* / their **ASPIC+ tutorial** ("The ASPIC+ framework for structured argumentation: a tutorial", Argument & Computation, 2014) — the canonical accessible intro.
2. Dung 1995, *"On the acceptability of arguments..."* — origin of abstract argumentation and grounded semantics (skim §1–2 for the intuition).
3. PyArg docs/paper (Odekerken et al., COMMA 2022) — what the engine actually implements.
