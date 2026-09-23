"""What a taskset is built from.

Today the grid is a function default the CLI cannot override, so a paper's
taskset is defined by "whatever the defaults were at that commit". A spec file
makes it an input that can be checked in, cited and diffed.

The seed policy is the part worth reading. A cell asks for a number of items and
gives the export a bound on how far to look; the export either produces what was
asked for or fails naming the cell. The alternative -- a fixed seed range, take
what builds -- lets a cell that rejects 12 of 20 seeds ship 8 items with nothing
saying so, and the frozen v2 taskset has exactly such a cell.

Version fields are constraints rather than records. Present, they are checked
before anything is generated and the build refuses on a mismatch; absent, the
export records what it used. That is what makes "reproducible from the spec
alone" mean something.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

ALL_ORDERINGS = ("last_link_elitist", "last_link_democratic",
                 "weakest_link_elitist", "weakest_link_democratic")
# The evaluated grid: every level the curriculum defines. A generator whose curriculum
# disagrees with these levels has a feature nothing exercises, which is how #30 stayed
# hidden, so read it from here rather than restating it. Exporting all fifteen is what
# closed that gap -- a level-indexed switch can no longer fire only at levels the grid
# skips, because the grid skips none.
LEVELS = tuple(range(1, 16))
SEEDS = (0, 1)


@dataclass(frozen=True)
class SeedPolicy:
    """How many items a cell needs, and how far the export may look."""

    start: int = 0
    take: int = 2
    scan_limit: int = 40

    def __post_init__(self) -> None:
        if self.take < 1:
            raise ValueError("seeds.take must be at least 1")
        if self.scan_limit < self.take:
            raise ValueError(
                f"seeds.scan_limit ({self.scan_limit}) is below seeds.take "
                f"({self.take}); the cell could never be filled")


@dataclass(frozen=True)
class TasksetSpec:
    tasks: Tuple[str, ...]
    levels: Tuple[int, ...] = LEVELS
    orderings: Tuple[str, ...] = ALL_ORDERINGS
    seeds: SeedPolicy = field(default_factory=SeedPolicy)
    profile: str = "FULL"
    #: Refuse a cell that keeps fewer than this share of the seeds it scanned. A
    #: degraded cell should be a decision, not a silent property of the file.
    #: Where `min_build_acceptance` is on, it refuses a smaller degradation than
    #: this guard on a generator that fails outright, and this guard bounds the
    #: seeds the freeze skips for `minimality_unproven`, which cost one
    #: candidate each. The default serves a spec that names neither guard;
    #: `tasksets/standard.yaml` sets 0.3.
    min_acceptance: float = 0.5
    #: Refuse a cell that keeps fewer than this share of the candidates `build`
    #: produced. Seed acceptance cannot see the candidates the retry loop
    #: discards on a seed that builds: `perturbation` at level 6 reaches its
    #: items from every seed it scans by discarding 34 candidates of 44 (#114).
    #: Off by default, so a spec that does not name it keeps the seed guard
    #: alone; `tasksets/standard.yaml` names it.
    min_build_acceptance: float = 0.0
    #: Optional constraints, checked before generating. See the module docstring.
    arggym: Optional[str] = None
    pyarg: Optional[str] = None
    prompt_version: Optional[int] = None
    theory_schema: Optional[int] = None
    scoring_version: Optional[int] = None

    def __post_init__(self) -> None:
        from arggym.core import registry

        if not self.tasks:
            raise ValueError("a spec must name at least one task")
        unknown = [t for t in self.tasks if t not in registry.REGISTRY]
        if unknown:
            raise ValueError(f"unknown task(s) {', '.join(unknown)}; "
                             f"known: {', '.join(registry.task_names())}")
        bad = [o for o in self.orderings if o not in ALL_ORDERINGS]
        if bad:
            raise ValueError(f"unknown ordering(s) {', '.join(bad)}")
        if not self.levels:
            raise ValueError("a spec must name at least one level")
        # A float level is accepted everywhere and means nothing anywhere:
        # claim_chain raises TypeError on the slice, status_query builds an item.
        # `type(...) is int` rather than isinstance, or `levels: [true]` is level 1.
        bad_levels = [lv for lv in self.levels if type(lv) is not int]
        if bad_levels:
            raise ValueError(f"levels must be integers; got {bad_levels}")
        if not 0.0 < self.min_acceptance <= 1.0:
            raise ValueError("min_acceptance must be in (0, 1]")
        if not 0.0 <= self.min_build_acceptance <= 1.0:
            raise ValueError("min_build_acceptance must be in [0, 1]")
        # The profile axis is real -- nine tasks branch on it -- but two tasks
        # cannot take one at all and only one mixes it into its seed, so a
        # non-FULL taskset would not be reproducible from its coordinates.
        # See docs/dataset-contract.md section 7.
        if self.profile != "FULL":
            raise ValueError(
                f"profile {self.profile!r} is not exportable yet: "
                "counter_argument and semantics_query take no profile, and only "
                "status_query mixes it into its seed, so the seed would not name "
                "the item. Track this on #51.")

    @property
    def cells(self) -> Tuple[Tuple[str, int, str], ...]:
        return tuple((t, lv, o) for t in self.tasks
                     for lv in self.levels for o in self.orderings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tup(v, name):
    if v is None:
        return None
    if isinstance(v, (str, bytes)):
        raise ValueError(f"{name} must be a list, not a bare string")
    return tuple(v)


def from_dict(d: Dict[str, Any]) -> TasksetSpec:
    d = dict(d)
    seeds = d.pop("seeds", None) or {}
    if not isinstance(seeds, dict):
        raise ValueError("seeds must be a mapping of start/take/scan_limit")
    known = {f for f in TasksetSpec.__dataclass_fields__}
    stray = sorted(set(d) - known)
    if stray:
        # A typo in a spec file would otherwise change nothing and be blamed on
        # the generator.
        raise ValueError(f"unknown spec field(s): {', '.join(stray)}")
    for k, n in (("tasks", "tasks"), ("levels", "levels"), ("orderings", "orderings")):
        if k in d:
            d[k] = _tup(d[k], n)
    return TasksetSpec(seeds=SeedPolicy(**seeds), **d)


def load(path: str) -> TasksetSpec:
    """Read a spec from YAML or JSON, decided by the file's suffix."""
    import json

    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:  # pragma: no cover - depends on the install
            raise SystemExit("reading a YAML spec needs pyyaml; "
                             "install it or write the spec as .json") from None
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a spec mapping")
    return from_dict(data)


def check_versions(spec: TasksetSpec) -> None:
    """Refuse to build a spec this tree cannot satisfy.

    Checked before generating rather than recorded after, so a mismatch costs
    seconds instead of a full export.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    import arggym
    from arggym.core.serialize import THEORY_SCHEMA

    def _installed(name: str) -> Optional[str]:
        try:
            return _pkg_version(name)
        except PackageNotFoundError:  # pragma: no cover - depends on the install
            return None

    from arggym.core.freeze import PROMPT_VERSION, SCORING_VERSION

    # Every constraint the spec can express is checked. Accepting a field and
    # ignoring it is worse than not offering it: a spec pinning prompt_version
    # would read as reproducible while guaranteeing nothing.
    for got, want, what in (
        (arggym.__version__, spec.arggym, "arggym"),
        (_installed("python-argumentation"), spec.pyarg, "python-argumentation"),
        (THEORY_SCHEMA, spec.theory_schema, "theory_schema"),
        (PROMPT_VERSION, spec.prompt_version, "prompt_version"),
        (SCORING_VERSION, spec.scoring_version, "scoring_version"),
    ):
        if want is not None and got is not None and str(got) != str(want):
            raise SystemExit(
                f"spec asks for {what} {want}, this tree has {got}; "
                "the taskset would not be the one the spec names")
