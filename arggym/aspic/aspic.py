from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

__all__ = [
    "Theory", "Rule", "Fact", "Preference", "Contrariness",
    "parse", "render", "examples",
    "AXIOM", "PREMISE", "STRICT", "DEFEASIBLE",
    "LAST_LINK_ELITIST", "LAST_LINK_DEMOCRATIC",
    "WEAKEST_LINK_ELITIST", "WEAKEST_LINK_DEMOCRATIC", "ORDERINGS",
    "GROUNDED", "PREFERRED", "STABLE", "SEMISTABLE", "IDEAL", "EAGER",
    "COMPLETE", "ADMISSIBLE", "NAIVE", "CONFLICT_FREE", "SEMANTICS",
    "JUSTIFIED", "OVERRULED", "UNDECIDED", "NO_EXTENSION", "negate",
    "to_apx", "to_tgf", "extension_names",
    "Labelling", "IN", "OUT", "UNDEC",
]


AXIOM, PREMISE = "axiom", "premise"
STRICT, DEFEASIBLE = "strict", "defeasible"

LAST_LINK_ELITIST = "last_link_elitist"
LAST_LINK_DEMOCRATIC = "last_link_democratic"
WEAKEST_LINK_ELITIST = "weakest_link_elitist"
WEAKEST_LINK_DEMOCRATIC = "weakest_link_democratic"
ORDERINGS = (LAST_LINK_ELITIST, LAST_LINK_DEMOCRATIC,
             WEAKEST_LINK_ELITIST, WEAKEST_LINK_DEMOCRATIC)

GROUNDED, PREFERRED, STABLE = "grounded", "preferred", "stable"
SEMISTABLE, IDEAL, EAGER = "semistable", "ideal", "eager"
COMPLETE, ADMISSIBLE, NAIVE, CONFLICT_FREE = "complete", "admissible", "naive", "conflict_free"
SEMANTICS = (GROUNDED, PREFERRED, STABLE, SEMISTABLE, IDEAL, EAGER,
             COMPLETE, ADMISSIBLE, NAIVE, CONFLICT_FREE)

UNIQUE_SEMANTICS = (GROUNDED, IDEAL, EAGER)

JUSTIFIED, OVERRULED, UNDECIDED = "JUSTIFIED", "OVERRULED", "UNDECIDED"
NO_EXTENSION = "NO_EXTENSION"


def negate(literal: str) -> str:
    literal = literal.strip()
    return literal[1:] if literal.startswith("-") else "-" + literal


def atoms_of(rule_names: Set[str], literals: Iterable[str] = (),
             consequents: Iterable[str] = ()) -> Set[str]:
    """The atoms filling a set of literal slots, each stripped of a leading ``-``.

    A consequent ``-<name>`` with ``<name>`` a rule is that rule's undercut target
    rather than an atom (docs/notation.md, undercutting), and a consequent is the only
    slot where that reading applies; every other slot contributes its atom whether
    or not the name is also a rule's. So ``rule_names & atoms_of(rule_names, ...)``
    is the set of names that mean two things at once, which is exactly when
    ``-<name>`` is ambiguous. Which slots a theory has is the caller's business:
    a ``Theory`` has facts and contraries, an ``Operation`` list has premise
    contents and ``prefer_premise`` operands, and both share the rule underneath.
    """
    out = {str(x).lstrip("-") for x in literals if x}
    for c in consequents:
        c = str(c or "")
        if c and not (c.startswith("-") and c[1:] in rule_names):
            out.add(c.lstrip("-"))
    return out


@dataclass(frozen=True)
class Fact:
    literal: str
    kind: str = PREMISE

    def __post_init__(self):
        if self.kind not in (AXIOM, PREMISE):
            raise ValueError(f"unknown fact kind {self.kind!r}")
        if not str(self.literal).strip():
            raise ValueError("a fact's literal cannot be empty")


@dataclass(frozen=True)
class Rule:
    name: str
    antecedents: Tuple[str, ...]
    consequent: str
    kind: str = DEFEASIBLE

    def __post_init__(self):
        if self.kind not in (STRICT, DEFEASIBLE):
            raise ValueError(f"unknown rule kind {self.kind!r}")
        if not str(self.consequent).strip():
            raise ValueError(f"rule {self.name!r} has an empty consequent")
        if any(not str(a).strip() for a in self.antecedents):
            raise ValueError(f"rule {self.name!r} has an empty antecedent")
        if not self.name:
            raise ValueError("rules must be named; undercutting refers to the name")

    @property
    def arrow(self) -> str:
        return "->" if self.kind == STRICT else "=>"


@dataclass(frozen=True)
class Preference:
    stronger: str
    weaker: str
    over: str = "premise"


@dataclass(frozen=True)
class Contrariness:
    source: str
    target: str
    symmetric: bool = False


_DIRECTIVE = re.compile(r"\[\s*([a-z_]+)\s*(?:([A-Za-z0-9_]+)\s*)?:\s*([^\]]*)\]")
_ARROW = re.compile(r"(=>|->)")


def parse(text: str) -> "Theory":
    th = Theory()
    src = text or ""

    cursor = 0
    for m in _DIRECTIVE.finditer(src):
        for line in (l.strip() for l in src[cursor:m.start()].splitlines()):
            if line and not line.startswith("#"):
                th.dropped.append((line, "not a directive"))
        cursor = m.end()
    for line in (l.strip() for l in src[cursor:].splitlines()):
        if line and not line.startswith("#"):
            th.dropped.append((line, "not a directive"))

    for m in _DIRECTIVE.finditer(src):
        kw, label, body = m.group(1).lower(), m.group(2), m.group(3).strip()
        raw = m.group(0)
        try:
            if kw in (AXIOM, PREMISE):
                th.add_fact(body, kw)
            elif kw in (STRICT, DEFEASIBLE):
                want = "->" if kw == STRICT else "=>"
                idx = body.find(want)
                am = None
                if idx >= 0:
                    class _M:
                        def __init__(self, i, w):
                            self._i, self._w = i, w
                        def start(self):
                            return self._i
                        def end(self):
                            return self._i + len(self._w)
                        def group(self, _n):
                            return self._w
                    am = _M(idx, want)
                if am is None:
                    raise ValueError(f"{kw} rule must use {want}")
                lhs, rhs = body[:am.start()], body[am.end():]
                ants = tuple(a.strip() for a in re.split(r"[,&]|\bAND\b", lhs) if a.strip())
                if not ants and lhs.strip() != lhs:
                    raise ValueError("blank antecedent list before the arrow")
                kind = STRICT if am.group(1) == "->" else DEFEASIBLE
                th.add_rule(label or f"r{len(th.rules) + 1}", ants, rhs.strip(), kind)
            elif kw == "prefer_premise":
                a, b = [x.strip() for x in body.split(">", 1)]
                th.prefer_premise(a, b)
            elif kw == "prefer_rule":
                a, b = [x.strip() for x in body.split(">", 1)]
                th.prefer_rule(a, b)
            elif kw == "contrary":
                a, b = [x.strip() for x in re.split(r"\^", body, 1)]
                th.add_contrary(a, b)
            elif kw == "contradictory":
                a, b = [x.strip() for x in re.split(r"(?<!^)-(?!\s*>)", body, 1)]
                th.add_contradictory(a.strip(), b.strip())
            else:
                th.dropped.append((raw, f"unknown directive {kw!r}"))
        except Exception as e:
            th.dropped.append((raw, f"{type(e).__name__}: {e}"))
    return th


def render(th: "Theory") -> str:
    lines: List[str] = []
    for f in th.facts:
        lines.append(f"[{f.kind}: {f.literal}]")
    for r in th.rules:
        lines.append(f"[{r.kind} {r.name}: {', '.join(r.antecedents)} {r.arrow} {r.consequent}]")
    for p in th.preferences:
        kw = "prefer_rule" if p.over == "rule" else "prefer_premise"
        lines.append(f"[{kw}: {p.stronger} > {p.weaker}]")
    for c in th.contraries:
        if c.symmetric:
            lines.append(f"[contradictory: {c.source} - {c.target}]")
        else:
            lines.append(f"[contrary: {c.source} ^ {c.target}]")
    return "\n".join(lines)


class Theory:

    def __init__(self, ordering: str = LAST_LINK_ELITIST):
        if ordering not in ORDERINGS:
            raise ValueError(f"unknown ordering {ordering!r}; expected one of {ORDERINGS}")
        self.facts: List[Fact] = []
        self.rules: List[Rule] = []
        self.preferences: List[Preference] = []
        self.contraries: List[Contrariness] = []
        self.ordering = ordering
        self.dropped: List[Tuple[str, str]] = []
        self._cache: Dict = {}


    def _dirty(self, preferences_only: bool = False) -> None:
        self._cache.clear()

    def add_fact(self, literal: str, kind: str = PREMISE) -> "Theory":
        self.facts.append(Fact(literal.strip(), kind))
        self._dirty()
        return self

    def axiom(self, literal: str) -> "Theory":
        return self.add_fact(literal, AXIOM)

    def premise(self, literal: str) -> "Theory":
        return self.add_fact(literal, PREMISE)


    def add_rule(self, name: str, antecedents: Iterable[str], consequent: str,
                 kind: str = DEFEASIBLE) -> "Theory":
        self.rules.append(Rule(name.strip(), tuple(a.strip() for a in antecedents),
                               consequent.strip(), kind))
        self._dirty()
        return self

    def defeasible(self, name: str, antecedents: Iterable[str], consequent: str) -> "Theory":
        return self.add_rule(name, antecedents, consequent, DEFEASIBLE)

    def strict(self, name: str, antecedents: Iterable[str], consequent: str) -> "Theory":
        return self.add_rule(name, antecedents, consequent, STRICT)

    def prefer_premise(self, stronger: str, weaker: str) -> "Theory":
        self.preferences.append(Preference(stronger.strip(), weaker.strip(), "premise"))
        self._dirty()
        return self

    def prefer_rule(self, stronger: str, weaker: str) -> "Theory":
        self.preferences.append(Preference(stronger.strip(), weaker.strip(), "rule"))
        self._dirty()
        return self

    def add_contrary(self, source: str, target: str) -> "Theory":
        self.contraries.append(Contrariness(source.strip(), target.strip(), False))
        self._dirty()
        return self

    def add_contradictory(self, a: str, b: str) -> "Theory":
        self.contraries.append(Contrariness(a.strip(), b.strip(), True))
        self._dirty()
        return self


    @property
    def rule_names(self) -> Set[str]:
        return {r.name for r in self.rules}

    @property
    def literals(self) -> Set[str]:
        out: Set[str] = set()
        for f in self.facts:
            out.add(f.literal)
        for r in self.rules:
            out.update(r.antecedents)
            out.add(r.consequent)
        return out

    def facts_of(self, kind: str) -> List[str]:
        return [f.literal for f in self.facts if f.kind == kind]

    def name_collisions(self) -> Set[str]:
        names = self.rule_names
        literals: List[str] = [f.literal for f in self.facts]
        for r in self.rules:
            literals.extend(r.antecedents)
        for cx in self.contraries:
            literals.extend((cx.source, cx.target))
        return names & atoms_of(names, literals, [r.consequent for r in self.rules])

    def validate(self) -> List[str]:
        problems: List[str] = []
        axioms = set(self.facts_of(AXIOM))
        strict_names = {r.name for r in self.rules if r.kind == STRICT}

        for p in self.preferences:
            if p.over == "premise":
                for side in (p.stronger, p.weaker):
                    if side in axioms or side.lstrip("-") in axioms:
                        problems.append(f"prefer_premise names an axiom: {side}")
            else:
                for side in (p.stronger, p.weaker):
                    if side in strict_names:
                        problems.append(f"prefer_rule names a strict rule: {side}")

        for a in sorted(axioms):
            if not a.startswith("-") and negate(a) in axioms:
                problems.append(f"axioms are inconsistent: {a} and {negate(a)}")

        strict_consequents = {r.consequent for r in self.rules if r.kind == STRICT}
        for c in self.contraries:
            if c.symmetric:
                continue
            if c.target in axioms:
                problems.append(f"{c.target} is an axiom and cannot have a contrary "
                                f"({c.source} ^ {c.target})")
            if c.target in strict_consequents:
                problems.append(f"{c.target} is a strict consequent and cannot have a contrary "
                                f"({c.source} ^ {c.target})")

        ok_wd, why_wd = self.is_well_defined()
        if not ok_wd:
            problems.extend(why_wd)

        rule_names_present = {r.name for r in self.rules}
        literals_present = set(self.literals)
        for pref in self.preferences:
            if pref.over == "rule":
                for side in (pref.stronger, pref.weaker):
                    if side not in rule_names_present:
                        problems.append(f"prefer_rule mentions unknown rule {side!r}")
            else:
                for side in (pref.stronger, pref.weaker):
                    if side not in literals_present:
                        problems.append(f"prefer_premise mentions unknown literal {side!r}")

        for n in sorted(self.name_collisions()):
            problems.append(f"rule name {n!r} is also a literal, so -{n} is ambiguous")

        seen: Set[str] = set()
        for r in self.rules:
            if r.name in seen:
                problems.append(f"duplicate rule name: {r.name}")
            seen.add(r.name)

        for d, why in self.dropped:
            problems.append(f"unparsed: {d} ({why})")
        return problems

    def is_well_defined(self) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        has_strict = any(r.kind == STRICT for r in self.rules)
        has_axioms = bool(self.facts_of(AXIOM))
        if not has_strict and not has_axioms:
            return True, ["no strict rules and no axioms: postulates hold unconditionally"]
        axioms = set(self.facts_of(AXIOM))
        for a in axioms:
            if not a.startswith("-") and negate(a) in axioms:
                reasons.append(f"axioms inconsistent: {a}")
        missing = self.missing_transpositions()
        if missing:
            reasons.append(f"{len(missing)} strict rule(s) not closed under transposition")
        return (not reasons), reasons


    def transpositions_of(self, rule: Rule) -> List[Rule]:
        if rule.kind != STRICT or not rule.antecedents or not rule.consequent:
            return []
        out = []
        ants = list(rule.antecedents)
        for i, a in enumerate(ants):
            rest = [x for j, x in enumerate(ants) if j != i]
            out.append(Rule(f"{rule.name}_tp{i}",
                            tuple([negate(rule.consequent)] + rest),
                            negate(a), STRICT))
        return out

    def missing_transpositions(self) -> List[Rule]:
        have = {(tuple(r.antecedents), r.consequent) for r in self.rules if r.kind == STRICT}
        out = []
        for r in self.rules:
            if r.kind != STRICT:
                continue
            for t in self.transpositions_of(r):
                if (tuple(t.antecedents), t.consequent) not in have:
                    out.append(t)
        return out

    def close_under_transposition(self) -> "Theory":
        for t in self.missing_transpositions():
            self.rules.append(t)
        self._dirty()
        return self


    def _build(self):
        if "theory" in self._cache:
            return self._cache["theory"]

        from py_arg.aspic_classes.argumentation_system import ArgumentationSystem
        from py_arg.aspic_classes.argumentation_theory import ArgumentationTheory
        from py_arg.aspic_classes.defeasible_rule import DefeasibleRule
        from py_arg.aspic_classes.literal import Literal
        from py_arg.aspic_classes.strict_rule import StrictRule

        names: Set[str] = set()
        for l in self.literals:
            names.add(l)
            names.add(negate(l))
        for f in self.facts:
            names.add(f.literal)
            names.add(negate(f.literal))
        for c in self.contraries:
            names.add(c.source)
            names.add(c.target)
        lang = {n: Literal(n) for n in sorted(names)}

        contra: Dict[str, List] = {n: [] for n in lang}
        declared: Set[Tuple[str, str]] = set()
        for c in self.contraries:
            if c.target in lang and c.source in lang:
                contra[c.target].append(lang[c.source])
                declared.add((c.target, c.source))
                if c.symmetric:
                    contra[c.source].append(lang[c.target])
                    declared.add((c.source, c.target))
        mentioned = {c.source for c in self.contraries} | {c.target for c in self.contraries}
        for n in lang:
            neg = negate(n)
            if n in mentioned or neg in mentioned:
                continue
            if neg in lang and (n, neg) not in declared and not contra[n]:
                contra[n].append(lang[neg])

        strict_rules, defeasible_rules = [], []
        for r in self.rules:
            ants = {lang[a] for a in r.antecedents if a in lang}
            if r.consequent not in lang:
                lang[r.consequent] = Literal(r.consequent)
                contra.setdefault(r.consequent, [])
            if r.kind == STRICT:
                strict_rules.append(StrictRule(r.name, ants, lang[r.consequent]))
            else:
                defeasible_rules.append(DefeasibleRule(r.name, ants, lang[r.consequent]))

        for d in defeasible_rules:
            lit = Literal.from_defeasible_rule(d)
            neg = Literal.from_defeasible_rule_negation(d)
            lang[str(lit)] = lit
            lang[str(neg)] = neg
            contra[str(lit)] = [neg]
            contra[str(neg)] = [lit]

        by_name = {d.id: d for d in defeasible_rules}
        for i, r in enumerate(self.rules):
            if r.kind != DEFEASIBLE:
                continue
            c = r.consequent
            if c.startswith("-") and c[1:] in by_name:
                target = Literal.from_defeasible_rule_negation(by_name[c[1:]])
                ants = {lang[a] for a in r.antecedents if a in lang}
                for j, d in enumerate(defeasible_rules):
                    if d.id == r.name:
                        defeasible_rules[j] = DefeasibleRule(r.name, ants, target)
                        break

        system = ArgumentationSystem(lang, contra, strict_rules, defeasible_rules)

        rule_by_name = {d.id: d for d in defeasible_rules}
        def _closed(pairs):
            out = set(pairs)
            changed = True
            while changed:
                changed = False
                for a, b in list(out):
                    for c, d in list(out):
                        if b == c and a != d and (a, d) not in out:
                            out.add((a, d))
                            changed = True
            return out

        declared_rule = [(p.stronger, p.weaker) for p in self.preferences if p.over == "rule"]
        declared_prem = [(p.stronger, p.weaker) for p in self.preferences if p.over != "rule"]
        closed_rule = _closed(declared_rule)
        closed_prem = _closed(declared_prem)

        rule_prefs = [(d, d) for d in defeasible_rules]
        for stronger, weaker in sorted(closed_rule):
            a, b = rule_by_name.get(stronger), rule_by_name.get(weaker)
            if a is not None and b is not None:
                rule_prefs.append((b, a))
        self._apply_preferences(system, rule_prefs)

        axioms = [lang[l] for l in self.facts_of(AXIOM) if l in lang]
        ordinary = [lang[l] for l in self.facts_of(PREMISE) if l in lang]
        prem_prefs = [(p, p) for p in ordinary]
        for stronger, weaker in sorted(closed_prem):
            a, b = lang.get(stronger), lang.get(weaker)
            if a is not None and b is not None:
                prem_prefs.append((b, a))

        theory = ArgumentationTheory(system, axioms, ordinary)
        self._apply_premise_preferences(theory, prem_prefs)

        self._cache["theory"] = theory
        self._cache["lang"] = lang
        self._cache["contra_map"] = {k: [str(x) for x in v] for k, v in contra.items()}
        return theory

    @staticmethod
    def _apply_preferences(system, pairs) -> None:
        if not pairs:
            return
        try:
            from py_arg.aspic_classes.orderings.preference_preorder import PreferencePreorder
            system.rule_preferences = PreferencePreorder(list(pairs))
        except Exception as exc:
            raise RuntimeError(
                f"could not install {len(pairs)} rule preference(s); refusing to evaluate a theory "
                f"whose preferences were dropped") from exc

    @staticmethod
    def _apply_premise_preferences(theory, pairs) -> None:
        if not pairs:
            return
        try:
            from py_arg.aspic_classes.orderings.preference_preorder import PreferencePreorder
            theory.ordinary_premise_preferences = PreferencePreorder(list(pairs))
        except Exception as exc:
            raise RuntimeError(
                f"could not install {len(pairs)} premise preference(s); refusing to evaluate a "
                f"theory whose preferences were dropped") from exc

    def _ordering_object(self, theory):
        from py_arg.aspic_classes.orderings.argument_orderings.last_link_ordering import (
            LastLinkDemocraticOrdering,
            LastLinkElitistOrdering,
        )
        from py_arg.aspic_classes.orderings.argument_orderings.weakest_link_ordering import (
            WeakestLinkDemocraticOrdering,
            WeakestLinkElitistOrdering,
        )
        cls = {LAST_LINK_ELITIST: LastLinkElitistOrdering,
               LAST_LINK_DEMOCRATIC: LastLinkDemocraticOrdering,
               WEAKEST_LINK_ELITIST: WeakestLinkElitistOrdering,
               WEAKEST_LINK_DEMOCRATIC: WeakestLinkDemocraticOrdering}[self.ordering]
        rule_prefs = getattr(theory.argumentation_system, "rule_preferences", None)
        prem_prefs = getattr(theory, "ordinary_premise_preferences", None)
        from py_arg.aspic_classes.orderings.preference_preorder import PreferencePreorder
        if rule_prefs is None:
            rule_prefs = PreferencePreorder([])
        if prem_prefs is None:
            prem_prefs = PreferencePreorder([])
        return cls(rule_prefs, prem_prefs)

    def _candidate_pairs(self, th, args):
        by_conclusion: Dict[str, List] = {}
        by_premise: Dict[str, List] = {}
        by_rule: Dict[str, List] = {}
        for b in args:
            for sub in (getattr(b, "sub_arguments", None) or [b]):
                by_conclusion.setdefault(str(getattr(sub, "conclusion", sub)), []).append(b)
            for p in (getattr(b, "premises", None) or ()):
                by_premise.setdefault(str(p), []).append(b)
            for r in (getattr(b, "defeasible_rules", None) or ()):
                by_rule.setdefault(str(getattr(r, "id", r)), []).append(b)

        contra = self._cache.get("contra_map", {})
        if "contra_inv" not in self._cache:
            inv: Dict[str, Set[str]] = {}
            for target, sources in contra.items():
                for src in sources:
                    inv.setdefault(src, set()).add(target)
            self._cache["contra_inv"] = inv
        inv = self._cache["contra_inv"]

        pairs = set()
        for a in args:
            c = str(getattr(a, "conclusion", a))
            targets = set(inv.get(c, ())) | {negate(c)}
            for tgt in targets:
                for b in by_conclusion.get(tgt, ()):
                    pairs.add((a, b))
                for b in by_premise.get(tgt, ()):
                    pairs.add((a, b))
            if c.startswith("-"):
                for b in by_rule.get(c[1:], ()):
                    pairs.add((a, b))
        return pairs

    def framework(self):
        if "af" not in self._cache:
            th = self._build()
            try:
                ordering = self._ordering_object(th)
            except Exception as exc:
                raise RuntimeError(
                    f"could not construct the {self.ordering!r} ordering; refusing to evaluate "
                    f"with a different one than was asked for") from exc
            try:
                from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import (
                    AbstractArgumentationFramework,
                )
                from py_arg.abstract_argumentation_classes.defeat import Defeat
                args = list(th.all_arguments)
                defeats = []
                for a, b in self._candidate_pairs(th, args):
                    if ordering is None:
                        if th.attacks(a, b):
                            defeats.append(Defeat(a, b))
                    elif th.defeats(a, b, ordering):
                        defeats.append(Defeat(a, b))
                af = AbstractArgumentationFramework("af", args, defeats)
            except Exception:
                af = (th.create_abstract_argumentation_framework("af", ordering)
                      if ordering is not None
                      else th.create_abstract_argumentation_framework("af"))
            self._cache["af"] = af
        return self._cache["af"]

    def arguments(self) -> List:
        return list(self._build().all_arguments)

    def _defeaters_of(self, arg) -> List:
        if "defeaters" not in self._cache:
            af = self.framework()
            idx: Dict = {a: [] for a in af.arguments}
            for a in af.arguments:
                for b in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
                    if b in idx:
                        idx[b].append(a)
            self._cache["defeaters"] = idx
        return self._cache["defeaters"].get(arg, [])

    def extensions_as_arguments(self, semantics: str = GROUNDED) -> List[Set]:
        key = f"argext:{semantics}"
        if key in self._cache:
            return self._cache[key]
        if semantics not in SEMANTICS:
            raise ValueError(f"unknown semantics {semantics!r}")
        import importlib
        af = self.framework()
        modmap = {GROUNDED: ("get_grounded_extension", "get_grounded_extension"),
                  PREFERRED: ("get_preferred_extensions", "get_preferred_extensions"),
                  STABLE: ("get_stable_extensions", "get_stable_extensions"),
                  SEMISTABLE: ("get_semistable_extensions", "get_semistable_extensions"),
                  IDEAL: ("get_ideal_extension", "get_ideal_extension"),
                  EAGER: ("get_eager_extension", "get_eager_extension"),
                  COMPLETE: ("get_complete_extensions", "get_complete_extensions"),
                  ADMISSIBLE: ("get_admissible_sets", "get_admissible_sets"),
                  NAIVE: ("get_naive_extensions", "apply"),
                  CONFLICT_FREE: ("get_conflict_free_extensions", "apply")}
        mod_name, fn_name = modmap[semantics]
        mod = importlib.import_module(f"py_arg.algorithms.semantics.{mod_name}")
        raw = getattr(mod, fn_name)(af)
        if semantics in UNIQUE_SEMANTICS:
            ext = raw if isinstance(raw, (set, frozenset, list, tuple)) else [raw]
            flat = []
            for e in ext:
                if isinstance(e, (set, frozenset, list, tuple)):
                    flat.extend(e)
                else:
                    flat.append(e)
            groups = [flat]
        elif raw is None:
            groups = []
        elif isinstance(raw, (list, set, frozenset, tuple)):
            groups = [e if isinstance(e, (set, frozenset, list, tuple)) else [e] for e in raw]
        else:
            groups = [[raw]]
        out = [set(g) for g in groups]
        self._cache[key] = out
        return out

    def labellings(self, semantics: str = GROUNDED) -> "List[Labelling]":
        af = self.framework()
        args = list(af.arguments)
        out: List[Labelling] = []
        for in_args in self.extensions_as_arguments(semantics):
            in_args = set(in_args)
            attacked = set()
            for a in in_args:
                attacked |= set(getattr(a, "get_outgoing_defeat_arguments", None) or [])
            mapping = {}
            for a in args:
                if a in in_args:
                    mapping[a] = IN
                elif a in attacked:
                    mapping[a] = OUT
                else:
                    mapping[a] = UNDEC
            out.append(Labelling(self, mapping))
        return out

    def labelling(self, semantics: str = GROUNDED) -> "Optional[Labelling]":
        ls = self.labellings(semantics)
        return ls[0] if ls else None

    def justification_status(self, claim: str, semantics: str = PREFERRED) -> Set[str]:
        labs = self.labellings(semantics)
        if not labs:
            return set()
        return {lab.status_of(claim).replace(JUSTIFIED, IN).replace(OVERRULED, OUT)
                .replace(UNDECIDED, UNDEC) for lab in labs}

    def attacks(self) -> List:
        return list(self._build().all_attacks)


    def _undecided_region(self):
        from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension
        af = self.framework()
        args = list(af.arguments)
        grounded = set(get_grounded_extension(af))
        attacked = set()
        for a in grounded:
            for d in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
                attacked.add(d)
        undecided = [a for a in args if a not in grounded and a not in attacked]
        return grounded, attacked, undecided

    DIRECTIONAL = (COMPLETE, PREFERRED, GROUNDED, IDEAL)

    def strongly_connected_components(self, args=None):
        af = self.framework()
        nodes = list(af.arguments) if args is None else list(args)
        nodeset = set(nodes)
        succ = {a: [b for b in (getattr(a, "get_outgoing_defeat_arguments", None) or [])
                    if b in nodeset] for a in nodes}

        index = {}
        low = {}
        on_stack = {}
        stack = []
        result = []
        counter = [0]

        for root in nodes:
            if root in index:
                continue
            work = [(root, iter(succ[root]))]
            index[root] = low[root] = counter[0]
            counter[0] += 1
            stack.append(root)
            on_stack[root] = True
            while work:
                node, it = work[-1]
                advanced = False
                for nxt in it:
                    if nxt not in index:
                        index[nxt] = low[nxt] = counter[0]
                        counter[0] += 1
                        stack.append(nxt)
                        on_stack[nxt] = True
                        work.append((nxt, iter(succ[nxt])))
                        advanced = True
                        break
                    if on_stack.get(nxt):
                        low[node] = min(low[node], index[nxt])
                if advanced:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp.append(w)
                        if w is node:
                            break
                    result.append(comp)
        result.reverse()
        return result

    @staticmethod
    def _sub_framework(name, args, defeats):
        from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import (
            AbstractArgumentationFramework,
        )
        touched = set(args)
        for d in defeats:
            touched.add(d.from_argument)
            touched.add(d.to_argument)
        saved = {}
        for a in touched:
            saved[a] = (list(getattr(a, "_outgoing_defeat_arguments", []) or []),
                        list(getattr(a, "_ingoing_defeat_arguments", []) or []),
                        list(getattr(a, "_outgoing_defeat_call_arguments", []) or []))
        try:
            return AbstractArgumentationFramework(name, list(args), list(defeats))
        finally:
            for a, (out, ing, calls) in saved.items():
                if hasattr(a, "_outgoing_defeat_arguments"):
                    a._outgoing_defeat_arguments = out
                if hasattr(a, "_ingoing_defeat_arguments"):
                    a._ingoing_defeat_arguments = ing
                if hasattr(a, "_outgoing_defeat_call_arguments"):
                    a._outgoing_defeat_call_arguments = calls

    def _components(self, args):
        idx = {a: i for i, a in enumerate(args)}
        parent = list(range(len(args)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        aset = set(args)
        for a in args:
            for b in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
                if b in aset:
                    union(idx[a], idx[b])
            for b in (getattr(a, "get_ingoing_defeat_arguments", None) or []):
                if b in aset:
                    union(idx[a], idx[b])
        groups: Dict[int, List] = {}
        for a in args:
            groups.setdefault(find(idx[a]), []).append(a)
        return list(groups.values())

    def _extensions_by_scc(self, undecided, semantics, base):
        if semantics != PREFERRED:
            return None
        import importlib

        from py_arg.abstract_argumentation_classes.defeat import Defeat
        mod_name = {PREFERRED: "get_preferred_extensions", COMPLETE: "get_complete_extensions",
                    GROUNDED: "get_grounded_extension",
                    IDEAL: "get_ideal_extension"}.get(semantics)
        if mod_name is None:
            return None
        sccs = self.strongly_connected_components(undecided)
        if len(sccs) < 2:
            return None
        try:
            mod = importlib.import_module(f"py_arg.algorithms.semantics.{mod_name}")
            fn = getattr(mod, mod_name)
            partials: List[Set] = [set()]
            for comp in sccs:
                cset = set(comp)
                nxt: List[Set] = []
                for chosen in partials:
                    killed = set()
                    for a in chosen:
                        killed |= {b for b in
                                   (getattr(a, "get_outgoing_defeat_arguments", None) or [])
                                   if b in cset}
                    live = [a for a in comp if a not in killed]
                    if not live:
                        nxt.append(set(chosen))
                        continue
                    lset = set(live)
                    defeats = []
                    for a in live:
                        for b in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
                            if b in lset:
                                defeats.append(Defeat(a, b))
                    sub = self._sub_framework("scc", live, defeats)
                    raw = fn(sub)
                    if semantics in UNIQUE_SEMANTICS:
                        local = [set(raw) if isinstance(raw, (set, frozenset, list, tuple))
                                 else {raw}]
                    else:
                        local = [set(e) for e in (raw or [])] or [set()]
                    for e in local:
                        nxt.append(set(chosen) | e)
                partials = nxt
                if len(partials) > 512:
                    return None
            out: List[Set[str]] = []
            for p_ in partials:
                merged = set(base) | {str(getattr(a, "conclusion", a)) for a in p_}
                if merged not in out:
                    out.append(merged)
            return out
        except Exception:
            return None

    def _extensions_by_component(self, comps, semantics, base):
        import importlib
        from itertools import product

        from py_arg.abstract_argumentation_classes.defeat import Defeat
        mod_name = {PREFERRED: "get_preferred_extensions", STABLE: "get_stable_extensions",
                    SEMISTABLE: "get_semistable_extensions",
                    COMPLETE: "get_complete_extensions"}[semantics]
        try:
            mod = importlib.import_module(f"py_arg.algorithms.semantics.{mod_name}")
            fn = getattr(mod, mod_name)
            per_comp = []
            for ci, comp in enumerate(comps):
                cset = set(comp)
                defeats = []
                for a in comp:
                    for b in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
                        if b in cset:
                            defeats.append(Defeat(a, b))
                sub = self._sub_framework(f"c{ci}", comp, defeats)
                exts = fn(sub) or []
                if not exts:
                    if semantics in (STABLE, SEMISTABLE):
                        return []
                    per_comp.append([set()])
                else:
                    per_comp.append([{str(getattr(a, "conclusion", a)) for a in e} for e in exts])
            out = []
            for combo in product(*per_comp):
                merged = set(base)
                for e in combo:
                    merged |= e
                if merged not in out:
                    out.append(merged)
            return out
        except Exception:
            return None

    def _eager_from_semistable(self):
        try:
            grounded, attacked, undecided = self._undecided_region()
        except Exception:
            return None
        if not undecided:
            return [{str(getattr(a, "conclusion", a)) for a in grounded}]

        import importlib

        from py_arg.abstract_argumentation_classes.defeat import Defeat
        uset = set(undecided)
        defeats_sub = []
        for a in undecided:
            for b in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
                if b in uset:
                    defeats_sub.append(Defeat(a, b))
        try:
            sub = self._sub_framework("cut", undecided, defeats_sub)
            mod = importlib.import_module(
                "py_arg.algorithms.semantics.get_semistable_extensions")
            raw = mod.get_semistable_extensions(sub) or []
            ss = [set(e) | set(grounded) for e in raw]
        except Exception:
            return None
        if not ss:
            return None
        common = set.intersection(*[set(x) for x in ss]) if len(ss) > 1 else set(ss[0])
        if not common:
            return [set()]

        defeats = {}
        for a in self.framework().arguments:
            defeats[a] = set(getattr(a, "get_outgoing_defeat_arguments", None) or [])
        attackers = {a: [b for b in defeats if a in defeats[b]] for a in defeats}

        cur = set(common)
        changed = True
        while changed:
            changed = False
            for a in list(cur):
                if not all(any(atk in defeats.get(d, ()) for d in cur) for atk in attackers[a]):
                    cur.discard(a)
                    changed = True
            if any(b in defeats.get(a, ()) for a in cur for b in cur):
                for a in list(cur):
                    if any(b in defeats.get(a, ()) for b in cur):
                        cur.discard(a)
                        changed = True
        return [{str(getattr(a, "conclusion", a)) for a in cur}]

    def _sat_unique(self, semantics: str):
        try:
            from satcheck import eager_sat, ideal_sat
        except Exception:
            return None
        try:
            args = ideal_sat(self) if semantics == IDEAL else eager_sat(self)
            return [{str(getattr(a, "conclusion", a)) for a in args}]
        except Exception:
            return None

    def extensions(self, semantics: str = GROUNDED) -> List[Set[str]]:
        if semantics not in SEMANTICS:
            raise ValueError(f"unknown semantics {semantics!r}; expected one of {SEMANTICS}")
        key = f"ext:{semantics}"
        if key in self._cache:
            return self._cache[key]
        import importlib
        af = self.framework()
        modmap = {GROUNDED: ("get_grounded_extension", "get_grounded_extension"),
                  PREFERRED: ("get_preferred_extensions", "get_preferred_extensions"),
                  STABLE: ("get_stable_extensions", "get_stable_extensions"),
                  SEMISTABLE: ("get_semistable_extensions", "get_semistable_extensions"),
                  IDEAL: ("get_ideal_extension", "get_ideal_extension"),
                  EAGER: ("get_eager_extension", "get_eager_extension"),
                  COMPLETE: ("get_complete_extensions", "get_complete_extensions"),
                  ADMISSIBLE: ("get_admissible_sets", "get_admissible_sets"),
                  NAIVE: ("get_naive_extensions", "apply"),
                  CONFLICT_FREE: ("get_conflict_free_extensions", "apply")}
        if semantics == IDEAL:
            sat_result = self._sat_unique(semantics)
            if sat_result is not None:
                self._cache[key] = sat_result
                return sat_result

        if semantics == EAGER:
            eager_result = self._eager_from_semistable()
            if eager_result is not None:
                self._cache[key] = eager_result
                return eager_result

        if semantics in (PREFERRED, STABLE, SEMISTABLE, COMPLETE):
            grounded, _attacked, undecided = self._undecided_region()
            base = {str(getattr(a, "conclusion", a)) for a in grounded}
            if not undecided:
                out = [base]
                self._cache[key] = out
                return out
            scc_result = self._extensions_by_scc(undecided, semantics, base)
            if scc_result is not None:
                self._cache[key] = scc_result
                return scc_result

            comps = self._components(undecided)
            if comps and len(undecided) < len(self.framework().arguments):
                combined = self._extensions_by_component(comps, semantics, base)
                if combined is not None:
                    self._cache[key] = combined
                    return combined

        mod_name, fn_name = modmap[semantics]
        mod = importlib.import_module(f"py_arg.algorithms.semantics.{mod_name}")
        raw = getattr(mod, fn_name)(af)

        if semantics in UNIQUE_SEMANTICS:
            ext = raw if isinstance(raw, (set, frozenset, list, tuple)) else [raw]
            flat = []
            for e in ext:
                if isinstance(e, (set, frozenset, list, tuple)):
                    flat.extend(e)
                else:
                    flat.append(e)
            groups = [flat]
        elif raw is None:
            groups = []
        elif isinstance(raw, (list, set, frozenset, tuple)):
            groups = [e if isinstance(e, (set, frozenset, list, tuple)) else [e] for e in raw]
        else:
            groups = [[raw]]

        out: List[Set[str]] = []
        for ext in groups:
            out.append({str(getattr(a, "conclusion", a)) for a in ext})
        if semantics in UNIQUE_SEMANTICS and not out:
            out = [set()]
        self._cache[key] = out
        return out

    def status(self, claim: str, semantics: str = GROUNDED,
               sceptical: bool = True) -> str:
        exts = self.extensions(semantics)
        if not exts:
            return NO_EXTENSION

        af = self.framework()
        bearers = [a for a in af.arguments
                   if str(getattr(a, "conclusion", a)) == claim]

        def per_extension(ext: Set[str]) -> str:
            if claim in ext:
                return JUSTIFIED
            if not bearers:
                return UNDECIDED
            for b in bearers:
                ds = self._defeaters_of(b)
                if not any(str(getattr(d, "conclusion", d)) in ext for d in ds):
                    return UNDECIDED
            return OVERRULED

        per = [per_extension(e) for e in exts]
        if sceptical:
            if all(v == JUSTIFIED for v in per):
                return JUSTIFIED
            if all(v == OVERRULED for v in per):
                return OVERRULED
            return UNDECIDED
        if any(v == JUSTIFIED for v in per):
            return JUSTIFIED
        if any(v == OVERRULED for v in per):
            return OVERRULED
        return UNDECIDED

    def status_map(self, semantics: str = GROUNDED, sceptical: bool = True) -> Dict[str, str]:
        out = {}
        for l in sorted(self.literals | {negate(x) for x in self.literals}):
            out[l] = self.status(l, semantics, sceptical)
        return out

    def justified(self, semantics: str = GROUNDED) -> Set[str]:
        exts = self.extensions(semantics)
        if not exts:
            return set()
        return set.intersection(*exts) if len(exts) > 1 else set(exts[0])

    def is_consistent(self, semantics: str = GROUNDED) -> bool:
        for e in self.extensions(semantics):
            for a in e:
                if negate(a) in e:
                    return False
                for c in self.contraries:
                    if c.source == a and c.target in e:
                        return False
        return True

    def semantics_report(self, claim: str) -> Dict[str, str]:
        out = {}
        for s in (GROUNDED, PREFERRED, STABLE, SEMISTABLE, IDEAL, EAGER):
            try:
                if s in UNIQUE_SEMANTICS:
                    out[s] = self.status(claim, s)
                else:
                    out[f"{s}_sceptical"] = self.status(claim, s, True)
                    out[f"{s}_credulous"] = self.status(claim, s, False)
            except Exception as e:
                out[s] = f"error: {type(e).__name__}"
        return out


    def with_ordering(self, ordering: str) -> "Theory":
        t = Theory(ordering)
        t.facts = list(self.facts)
        t.rules = list(self.rules)
        t.preferences = list(self.preferences)
        t.contraries = list(self.contraries)
        t.dropped = list(self.dropped)
        return t

    def copy(self) -> "Theory":
        return self.with_ordering(self.ordering)

    def extend(self, text_or_theory: Union[str, "Theory"]) -> "Theory":
        other = parse(text_or_theory) if isinstance(text_or_theory, str) else text_or_theory
        self.facts += other.facts
        self.rules += other.rules
        self.preferences += other.preferences
        self.contraries += other.contraries
        self.dropped += other.dropped
        only_prefs = (not other.facts and not other.rules and not other.contraries
                      and bool(other.preferences))
        self._dirty(preferences_only=only_prefs)
        return self

    def snapshot(self) -> Dict:
        return {"facts": len(self.facts), "rules": len(self.rules),
                "preferences": len(self.preferences), "contraries": len(self.contraries),
                "dropped": len(self.dropped)}

    def restore(self, snap: Dict) -> "Theory":
        self.facts = self.facts[:snap["facts"]]
        self.rules = self.rules[:snap["rules"]]
        self.preferences = self.preferences[:snap["preferences"]]
        self.contraries = self.contraries[:snap["contraries"]]
        self.dropped = self.dropped[:snap["dropped"]]
        self._dirty()
        return self

    def try_extend(self, text_or_theory: Union[str, "Theory"]):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            snap = self.snapshot()
            self.extend(text_or_theory)
            try:
                yield self
            finally:
                self.restore(snap)
        return _ctx()

    def __str__(self) -> str:
        return render(self)

    def __repr__(self) -> str:
        return (f"<Theory {len(self.facts)} facts, {len(self.rules)} rules, "
                f"{len(self.preferences)} preferences, {self.ordering}>")


def examples() -> Dict[str, str]:
    return {
        "basic":
            "[premise: a]\n"
            "[defeasible d1: a => x]\n",

        "rebut_ties":
            "# two arguments for contradictory conclusions and nothing to settle them\n"
            "[premise: a]\n[premise: b]\n"
            "[defeasible d1: a => x]\n[defeasible d2: b => -x]\n",

        "rebut_resolved":
            "[premise: a]\n[premise: b]\n"
            "[defeasible d1: a => x]\n[defeasible d2: b => -x]\n"
            "[prefer_rule: d1 > d2]\n",

        "undercut":
            "# an undercut defeats outright and no preference can save the target\n"
            "[premise: a]\n[premise: b]\n"
            "[defeasible d1: a => x]\n[defeasible d2: b => -d1]\n",

        "undercut_on_strict_is_inert":
            "[premise: a]\n[premise: b]\n"
            "[strict s1: a -> x]\n[defeasible d2: b => -s1]\n",

        "axiom_cannot_be_undermined":
            "[axiom: a]\n[premise: -a]\n"
            "[defeasible d1: a => x]\n",


        "asymmetric_contrary":
            "# p attacks q; q does NOT attack p\n"
            "[premise: p]\n[premise: q]\n"
            "[contrary: p ^ q]\n",

        "married_john":
            "# Caminada & Amgoud: without transposition BOTH conclusions are justified and the\n"
            "# extension is inconsistent. Call close_under_transposition() to repair it.\n"
            "[axiom: rr]\n[axiom: nn]\n"
            "[defeasible d1: rr => mm]\n[defeasible d2: nn => bb]\n"
            "[strict s1: mm -> hs]\n[strict s2: bb -> -hs]\n",

        "floating_conclusion":
            "# f follows either way, so grounded says UNDECIDED and sceptical preferred says JUSTIFIED\n"
            "[premise: a]\n[premise: b]\n"
            "[defeasible d1: a => p]\n[defeasible d2: b => -p]\n"
            "[defeasible d3: p => f]\n[defeasible d4: -p => f]\n",

        "junction":
            "# a rule with several antecedents: cutting EITHER branch kills the conclusion\n"
            "[premise: a]\n[premise: b]\n"
            "[defeasible d1: a, b => x]\n",

        "reinstatement":
            "# an even attack chain restores the original conclusion\n"
            "[premise: a]\n[premise: b]\n[premise: c]\n"
            "[defeasible d1: a => x]\n[defeasible d2: b => -d1]\n[defeasible d3: c => -d2]\n",
    }


def to_apx(t: "Theory") -> str:
    af = t.framework()
    args = list(af.arguments)
    names = {a: f"a{i}" for i, a in enumerate(args)}
    lines = [f"arg({names[a]})." for a in args]
    for a in args:
        for b in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
            if b in names:
                lines.append(f"att({names[a]},{names[b]}).")
    return "\n".join(lines) + "\n"


def to_tgf(t: "Theory") -> str:
    af = t.framework()
    args = list(af.arguments)
    idx = {a: i + 1 for i, a in enumerate(args)}
    lines = [f"{idx[a]}" for a in args] + ["#"]
    for a in args:
        for b in (getattr(a, "get_outgoing_defeat_arguments", None) or []):
            if b in idx:
                lines.append(f"{idx[a]} {idx[b]}")
    return "\n".join(lines) + "\n"


def extension_names(t: "Theory", semantics: str = GROUNDED) -> "List[List[str]]":
    af = t.framework()
    args = list(af.arguments)
    names = {a: f"a{i}" for i, a in enumerate(args)}
    return [sorted(names[a] for a in ext if a in names)
            for ext in t.extensions_as_arguments(semantics)]


IN, OUT, UNDEC = "in", "out", "undec"


class Labelling:

    def __init__(self, theory: "Theory", mapping: Dict):
        self.theory = theory
        self._map = dict(mapping)

    def __getitem__(self, arg):
        return self._map.get(arg, UNDEC)

    def label_of(self, arg) -> str:
        return self._map.get(arg, UNDEC)

    def by_label(self, label: str) -> List:
        return [a for a, v in self._map.items() if v == label]

    @property
    def in_args(self) -> List:
        return self.by_label(IN)

    @property
    def out_args(self) -> List:
        return self.by_label(OUT)

    @property
    def undec_args(self) -> List:
        return self.by_label(UNDEC)

    def conclusions(self, label: str = IN) -> Set[str]:
        return {str(getattr(a, "conclusion", a)) for a in self.by_label(label)}

    def status_of(self, claim: str) -> str:
        labels = [self._map.get(a, UNDEC) for a in self.theory.framework().arguments
                  if str(getattr(a, "conclusion", a)) == claim]
        if not labels:
            return UNDECIDED
        if IN in labels:
            return JUSTIFIED
        if all(v == OUT for v in labels):
            return OVERRULED
        return UNDECIDED

    def is_legally_in(self, arg) -> bool:
        return all(self._map.get(d, UNDEC) == OUT for d in self.theory._defeaters_of(arg))

    def is_legally_out(self, arg) -> bool:
        return any(self._map.get(d, UNDEC) == IN for d in self.theory._defeaters_of(arg))

    def is_legally_undec(self, arg) -> bool:
        ds = self.theory._defeaters_of(arg)
        return (not all(self._map.get(d, UNDEC) == OUT for d in ds)) and \
               (not any(self._map.get(d, UNDEC) == IN for d in ds))

    def is_admissible(self) -> bool:
        return (all(self.is_legally_in(a) for a in self.in_args)
                and all(self.is_legally_out(a) for a in self.out_args))

    def is_complete(self) -> bool:
        return self.is_admissible() and all(self.is_legally_undec(a) for a in self.undec_args)

    def illegal(self) -> Dict[str, List]:
        return {"illegally_in": [a for a in self.in_args if not self.is_legally_in(a)],
                "illegally_out": [a for a in self.out_args if not self.is_legally_out(a)],
                "illegally_undec": [a for a in self.undec_args if not self.is_legally_undec(a)]}

    def to_dict(self) -> Dict[str, str]:
        rank = {IN: 2, OUT: 1, UNDEC: 0}
        out: Dict[str, str] = {}
        for a, v in self._map.items():
            c = str(getattr(a, "conclusion", a))
            if c not in out or rank[v] > rank[out[c]]:
                out[c] = v
        return out

    def __repr__(self):
        return (f"<Labelling in={len(self.in_args)} out={len(self.out_args)} "
                f"undec={len(self.undec_args)}>")

    def __str__(self):
        d = self.to_dict()
        parts = [f"{k}: {v}" for k, v in sorted(d.items())]
        return "\n".join(parts)
