"""Items on demand: one dataset object per difficulty, indexed by seed.

`ds[k]` is the item built from seed `start + k`. Difficulty lives in the config,
so every item in one dataset is the same level and ordering, and the grid is a
concatenation of these.

The index is the seed, unconditionally. The tempting alternative -- skip a seed
that fails to build so the index stays dense -- costs more than it looks:

- `ds[5]` could not be computed without knowing whether seeds 0-4 built, so
  random access becomes linear and sharding breaks.
- A change that flips one low-index build shifts every item above it while the
  config, the seed and `len(ds)` stay identical: a total change presenting as a
  no-op.
- It hides the thing worth seeing. The frozen v2 taskset records twelve
  rejections in one `status_query` cell of twenty. Densifying turns that into
  "twenty items, looks fine", which is the failure #72 describes.

So a failed build raises, naming the cell. Densifying is the export's job, where
it is recorded (`core/spec.py`).
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

from arggym.core import registry
from arggym.core.answers import AnswerTemplate, ScoreResult
from arggym.core.rows import encode_fields
from arggym.core.rows import score as score_row
from arggym.core.rows import score_value as score_row_value
from arggym.core.serialize import THEORY_SCHEMA, ops_to_json


def _pyarg_version() -> Optional[str]:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        return _v("python-argumentation")
    except PackageNotFoundError:  # pragma: no cover - depends on the install
        return None


class BuildFailed(RuntimeError):
    """A cell produced no item at this seed."""


class TaskDataset:
    """Items of one task at one difficulty, addressed by seed."""

    def __init__(self, task: str, level: int, ordering: str, size: int = 100,
                 seed: int = 0, profile: str = "FULL",
                 template: Optional[AnswerTemplate] = None) -> None:
        self.spec = registry.get(task)
        self.task = task
        self.level = level
        self.ordering = ordering
        self.size = size
        self.seed = seed
        # Refused rather than recorded-and-ignored. Two tasks take no profile at
        # all and only status_query mixes it into its seed, so a non-FULL
        # dataset would report coordinates that do not name its items. The
        # freeze path is guarded by TasksetSpec; create() is public, so it needs
        # its own guard. See docs/dataset-contract.md section 7.
        if profile != "FULL":
            raise ValueError(
                f"profile {profile!r} is not buildable yet: counter_argument and "
                f"semantics_query take no profile, and only status_query mixes it "
                f"into its seed, so (level, ordering, seed) would not name the "
                f"item. Track this on #51.")
        self.profile = profile
        # Delivery belongs to the harness: it composes the prompt from the question,
        # calls its solver, and extracts the answer before handing it back. So the
        # question says nothing about a fence unless a caller asks for one here
        # (`docs/dataset-contract.md` section 1).
        self.template = template

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(self.size):
            yield self[i]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not 0 <= idx < self.size:
            raise IndexError(idx)
        seed = self.seed + idx
        # `profile` is deliberately not passed: counter_argument and
        # semantics_query take none at all. The constructor refuses anything but
        # FULL, which is every generator's own default, so the recorded value is
        # what built the item rather than what the caller hoped for.
        item = self.spec.make_item(self.level, seed, self.ordering,
                                   template=self.template)
        if item is None:
            raise BuildFailed(
                f"{self.task} L{self.level} {self.ordering} seed {seed} built no item "
                f"after its retry budget. The index is the seed, so this seed is not "
                f"silently replaced; export with a spec to skip it and record why.")
        return self.entry(item, seed, idx)

    def entry(self, item: Any, seed: int, index: int) -> Dict[str, Any]:
        """One item as the row a harness reads.

        Everything a scorer reads is written here, so `score` never needs the
        item back. Which fields those are is the registry's table, and the split
        between `state` and `gold` is what makes "do not show the model
        `metadata.gold`" one rule rather than twelve.
        """
        meta: Dict[str, Any] = {
            # The registered task name, not the package: a composite dispatches
            # scoring on this, so it has to be a registry key.
            "source_dataset": self.task,
            "source_index": index,
            "seed": seed,
            "level": self.level,
            "ordering": self.ordering,
            "profile": self.profile,
            # Null unless the caller asked for a delivery sentence, so a frozen
            # taskset says what its questions promised instead of leaving a reader
            # to infer it from the prompt text.
            "answer_template": self.template.name if self.template else None,
            "theory_schema": THEORY_SCHEMA,
            # Scoring the engine-checked tasks runs PyArg, so a row is
            # re-scorable against the pinned engine and no other.
            "pyarg_version": _pyarg_version(),
            "checker": self.spec.checker,
            "answer_shape": self.spec.answer_shape,
        }
        theory = [getattr(item, f) for f in self.spec.theory_fields]
        for f, ops in zip(self.spec.theory_fields, theory):
            meta[f] = ops_to_json(ops)
        base = theory[0] if theory else []
        meta["state"] = encode_fields(self.spec.state_fields, item, base)
        meta["gold"] = encode_fields(self.spec.gold_fields, item, base)

        ref = item.reference
        return {
            "id": f"{self.task}/L{self.level}/{self.ordering}/s{seed}",
            "task": self.task,
            "question": item.prompt,
            "reference_answer": ref() if callable(ref) else ref,
            "metadata": meta,
        }

    def score(self, answer: str, entry: Dict[str, Any]) -> ScoreResult:
        """Score an answer against a row, not against this dataset.

        The row says which task it belongs to, so a line read back from a
        frozen JSONL scores the same whichever dataset object is holding the
        method, and a concatenation scores each of its parts correctly.
        """
        return score_row(answer, entry)

    def score_value(self, value: Any, entry: Dict[str, Any]) -> ScoreResult:
        """Score an answer a solver produced as a value rather than as text."""
        return score_row_value(value, entry)

    def score_answer(self, answer: str, entry: Dict[str, Any]) -> float:
        return self.score(answer, entry).score


class ConcatDataset:
    """Several datasets read end to end, in the order given.

    Not reasoning-gym's `CompositeDataset`, which seeds an RNG from the global
    index, picks a sub-dataset by weight and passes the same index through
    unchanged. That yields roughly N/K items per sub-dataset, always at the same
    index values -- a training mixture. An evaluation grid wants every cell, once.
    """

    def __init__(self, parts: Sequence[TaskDataset]) -> None:
        self.parts: List[TaskDataset] = list(parts)
        self._starts: List[int] = []
        n = 0
        for p in self.parts:
            self._starts.append(n)
            n += len(p)
        self.size = n

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for p in self.parts:
            yield from p

    def _locate(self, idx: int) -> Any:
        import bisect

        if not 0 <= idx < self.size:
            raise IndexError(idx)
        i = bisect.bisect_right(self._starts, idx) - 1
        return self.parts[i], idx - self._starts[i]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        part, local = self._locate(idx)
        return part[local]


def create(task: str, level: int, ordering: str = "last_link_elitist",
           size: int = 100, seed: int = 0, profile: str = "FULL",
           template: Optional[AnswerTemplate] = None) -> TaskDataset:
    return TaskDataset(task, level, ordering, size=size, seed=seed, profile=profile,
                       template=template)


def from_spec(spec: Any, size: Optional[int] = None,
              template: Optional[AnswerTemplate] = None) -> ConcatDataset:
    """Every cell of a taskset spec, in a stable order."""
    n = size if size is not None else spec.seeds.take
    return ConcatDataset([
        TaskDataset(task, level, ordering, size=n, seed=spec.seeds.start,
                    profile=spec.profile, template=template)
        for task, level, ordering in spec.cells])
