# ArgGYM notation reference

This document explains the DSL, the semantics conventions, and the answer formats used by every
ArgGYM task.

---

## 1. The directive language

A theory is a list of directives, one per line, each in square brackets.

```
[premise: p]                        an ordinary premise -- assumed, but attackable
[axiom: p]                          an axiom -- assumed and NOT attackable
[defeasible r1: p => q]             a defeasible rule named r1
[strict s1: p -> q]                 a strict rule named s1
[defeasible r2: p AND q => s]       several antecedents, all required
[prefer_rule: r1 > r2]              r1 is preferred to r2
[prefer_premise: p > q]             premise p is preferred to premise q
```

**Literals.** An atom such as `p`, or its negation `-p`. Negation is written with a leading hyphen and
is never doubled: the contrary of `-p` is `p`.

**Rule names** are distinct from literals. No atom in a theory shares a name with a rule.

**Multiple antecedents** are joined by `AND` and are jointly required: the rule fires only when every
antecedent holds, so defeating any one of them defeats the rule.

---

## 2. The three attack forms

**UNDERMINING** attacks an ordinary premise, by asserting its contrary.

```
[premise: p]        with        [premise: -p]
```
Two contrary premises with no preference between them produce a TIE, not a defeat. A
`prefer_premise` directive decides it. An **axiom cannot be undermined**.

**REBUTTING** attacks a conclusion, by deriving its contrary.

```
[defeasible r1: a => q]     attacked by     [defeasible r2: b => -q]
```
Again a tie unless a `prefer_rule` decides it. A rebut can only be aimed at a **defeasible**
inference: the conclusion of a strict rule cannot be rebutted.

**UNDERCUTTING** attacks a RULE rather than a conclusion. It is written as a rule whose consequent is
the NAME of another rule, negated:

```
[defeasible r1: a => q]     attacked by     [defeasible r3: b => -r1]
```

This says *the inference r1 does not apply here*. It does not claim `q` is false and it does not deny
`a`. Three consequences follow, and they are the usual source of error:

* an undercut that holds **defeats outright** -- it is not a tie and needs no preference
* an undercut is **preference-immune**: no `prefer_rule` can save the target
* an undercut **cannot be aimed at a strict rule**. `[defeasible r3: b => -s1]` where `s1` is strict
  is INERT: it changes nothing.

---

## 3. Preferences

`prefer_rule` and `prefer_premise` resolve conflicts that would otherwise tie.

**A preference does not replace an earlier one.** If a theory contains `[prefer_rule: a > b]` and
`[prefer_rule: b > a]` is added, both are present and the conflict is a TIE -- the result is UNDECIDED,
not a reversal. Adding the reverse preference is how a settled conflict is made unsettled.

**A preference cannot create support.** A claim with no argument for it stays unjustified whatever the
weighting.

**Strict rules cannot be out-preferred.**

---

## 4. Semantics and orderings

These are two independent axes and an item names both.

**Semantics** decides how the attack graph is evaluated.

* `grounded` -- unique, maximally sceptical. The default for most tasks.
* `sceptical preferred` -- in EVERY preferred extension.
* `credulous preferred` -- in AT LEAST ONE preferred extension.
* `stable` -- in the stable extensions; a framework may have NONE, in which case say
  `no stable extension`.
* `eager` -- unique, and less sceptical than grounded.

**Strength orderings** decide which attacks succeed.

* `last-link` -- an argument's strength is set by its FINAL rule.
* `weakest-link` -- by its WEAKEST element, premises included.

---

## 5. Statuses

A claim's status is read off the arguments FOR it, not off its contrary.

* **justified** -- some argument for it is accepted
* **overruled** -- every argument for it is defeated
* **undecided** -- no argument for it is accepted, but at least one is neither accepted
  nor defeated; the conflict exists and nothing settles it

A justified undercut defeats every argument for a claim without putting the contrary
anywhere, so the claim is overruled while neither it nor its contrary is in the
extension. Reading overruled as "the contrary is in the extension" gets that case wrong
(#29): the contrary being accepted is one way for every argument to be defeated, not the
definition.

---

## 6. Answer formats

The shapes below are what an answer says. Where it goes is the harness's choice: the question
states what a legal answer must contain and stops there, so a fence, a JSON schema, a tool call or
a solver that returns the answer directly all work (`docs/dataset-contract.md` section 1).
Whitespace, blank lines, bullets and numbering are ignored; content is what is parsed. A solver
that produces the answer as a value rather than as text can submit it with `score_value` and skip
this section entirely.

**Status queries** -- one line per claim.
```
ab1: justified
cd2: undecided
```

**Semantics queries** -- one line per claim-and-semantics pair.
```
ab1 under grounded: undecided
ab1 under sceptical preferred: justified
```

**Directive answers** (attack, defence, counter-argument, preference construction) -- one directive per
line, in the DSL of section 1.
```
[premise: -ab1]
[prefer_premise: -ab1 > ab1]
[defeasible z1: cd2 => -r4]
```

Constraints on directive answers:
* every rule needs a name no rule in the theory, no earlier answer line, and no atom of
  the theory or of the answer has used (see "Rule names are distinct from literals" above:
  an atom is a name without its leading -, so `-ko1` uses the atom `ko1`)
* rule antecedents must be literals ALREADY present in the theory
* a new premise is permitted only as the negation of an ordinary premise already present
* no new axioms; no strict rules, except in the tasks that say otherwise
* the target claim may not simply be asserted
* the answer must be minimal: one using more than twice the fewest directives that work scores
  zero, and one directive that cannot be read at all scores the whole answer zero

**Extraction answers** (claim chain) -- the directives forming the line, copied exactly, in order from
premise to claim.

**Diagnosis answers** (defeat diagnosis) -- a status line, then one line per failure point.
```
status: overruled
defeated_at: <target>; defeater: <defeater>; kind: undermine|undercut|rebut
defeated_at: <target>; defeater: <defeater>; kind: undercut; survives_because: <rule>
```
`defeated_at` is the CLAIM attacked for undermine and rebut, and the RULE switched off for undercut.
`survives_because` names the rule that attacks that defeater and is itself defeated; a question asks
for it only where the item has one.

**Formalization answers** -- the full theory in the DSL of section 1. Rule names are your own choice
and are not scored; structure and directive types are.
