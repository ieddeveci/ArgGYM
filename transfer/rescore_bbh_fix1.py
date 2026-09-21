from pathlib import Path
import json

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import exact_match_hf_evaluate
from lm_eval.tasks import TaskManager, get_task_dict


ROOT = Path("/arf/scratch/futan/ArgGYM")

TASKS = ROOT / "transfer/lm_eval_tasks/pinned_20260920_qwen3_bbhfix1"

OLD = ROOT / "outputs/transfer/pre_rl_qwen3_14b_v3/lm_eval/bbh"

model_dirs = list(
    OLD.glob(
        "__arf__scratch__futan__arggym_hf_cache__hub__models--"
        "Qwen--Qwen3-14B__snapshots__*"
    )
)

if len(model_dirs) != 1:
    raise SystemExit(
        f"Expected exactly one BBH model output dir; found {len(model_dirs)}"
    )

SAMPLES = model_dirs[0]

sample_files = sorted(
    SAMPLES.glob("samples_arggym_bbh_pinned_*_*.jsonl")
)

if len(sample_files) != 27:
    raise SystemExit(
        f"Expected 27 BBH sample files; found {len(sample_files)}"
    )

# Load exactly the repaired ArgGYM task bundle.
tm = TaskManager(include_path=str(TASKS))

# These are copied directly from the pinned upstream cot_zeroshot
# exact-match configuration used by bbhfix1.
REGEXES_TO_IGNORE = [
    r"\.$",
    ",",
    r"\\",
    "\n",
    '"',
]

all_scores = []
rows_out = []
by_task = {}

for sf in sample_files:
    task_name = sf.name[len("samples_"):].split("_2026-", 1)[0]

    loaded = get_task_dict(
        [task_name],
        task_manager=tm,
    )

    if task_name not in loaded:
        raise RuntimeError(
            f"Could not load {task_name}; keys={list(loaded)}"
        )

    task = loaded[task_name]

    rows = [
        json.loads(line)
        for line in sf.read_text().splitlines()
        if line.strip()
    ]

    instances = []

    for r in rows:
        inst = Instance(
            request_type="generate_until",
            doc=r["doc"],
            arguments=(),
            idx=0,
            metadata=(task_name, int(r["doc_id"]), 1),
        )

        raw = r["resps"]

        # Logged lm-eval shape is normally [["completion"]].
        if (
            isinstance(raw, list)
            and len(raw) == 1
            and isinstance(raw[0], list)
        ):
            inst.resps = raw[0]
        elif isinstance(raw, list):
            inst.resps = raw
        else:
            raise RuntimeError(
                f"Unexpected resps shape: task={task_name} "
                f"doc_id={r['doc_id']} type={type(raw)}"
            )

        instances.append(inst)

    # Run the repaired task-specific lm-eval filter pipeline
    # on the already-generated completions.
    task._instances = instances
    task.apply_filters()

    scores = []
    invalid = 0
    changed = 0
    recovered = 0

    for r, inst in zip(rows, instances, strict=True):
        if "flexible-extract" not in inst.filtered_resps:
            raise RuntimeError(
                f"{task_name}: flexible-extract absent; "
                f"available={list(inst.filtered_resps)}"
            )

        pred = inst.filtered_resps["flexible-extract"]
        target = r["target"]

        if pred == "[invalid]":
            invalid += 1

        old_pred = (
            r.get("filtered_resps", [""])[0]
            if r.get("filtered_resps")
            else ""
        )

        if pred != old_pred:
            changed += 1

        metric = exact_match_hf_evaluate(
            predictions=[str(pred)],
            references=[str(target)],
            regexes_to_ignore=REGEXES_TO_IGNORE,
            ignore_case=True,
            ignore_punctuation=False,
        )

        score = float(metric["exact_match"])

        old_score = float(r.get("exact_match", 0.0) or 0.0)

        if old_score == 0.0 and score == 1.0:
            recovered += 1

        scores.append(score)
        all_scores.append(score)

        rows_out.append(
            {
                "task": task_name,
                "doc_id": r["doc_id"],
                "target": target,
                "old_filtered": old_pred,
                "new_filtered": pred,
                "old_exact_match": old_score,
                "new_exact_match": score,
            }
        )

    by_task[task_name] = {
        "n": len(scores),
        "correct": int(sum(scores)),
        "accuracy": sum(scores) / len(scores),
        "invalid_after_flexible_extract": invalid,
        "invalid_rate": invalid / len(scores),
        "changed_by_new_filter": changed,
        "recovered_correct": recovered,
    }

total = len(all_scores)
correct = int(sum(all_scores))
accuracy = correct / total

remaining_invalid = sum(
    x["invalid_after_flexible_extract"]
    for x in by_task.values()
)

recovered_total = sum(
    x["recovered_correct"]
    for x in by_task.values()
)

summary = {
    "status": "DIAGNOSTIC_RESCORE_OF_LEGACY_GENERATIONS",
    "warning": (
        "This rescoring repairs parsing only. It cannot recover text "
        "that was never generated because the legacy BBH run stopped "
        "at the old newline-newline termination sequence."
    ),
    "n": total,
    "correct": correct,
    "accuracy": accuracy,
    "invalid_after_flexible_extract": remaining_invalid,
    "invalid_rate": remaining_invalid / total,
    "recovered_correct_vs_old_scoring": recovered_total,
    "by_task": by_task,
}

outdir = (
    ROOT
    / "outputs/transfer/protocol_freeze/"
    "bbh_legacy_generations_rescored_fix1"
)

outdir.mkdir(parents=True, exist_ok=True)

(outdir / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

with (outdir / "samples.jsonl").open("w") as f:
    for row in rows_out:
        f.write(json.dumps(row) + "\n")


print("=" * 78)
print("BBH LEGACY COMPLETIONS — REPAIRED PARSER DIAGNOSTIC")
print("=" * 78)
print(f"n:                         {total}")
print(f"correct:                   {correct}")
print(f"rescored accuracy:         {accuracy:.6f} ({100*accuracy:.2f}%)")
print(
    f"remaining parser-invalid:  {remaining_invalid} "
    f"({100*remaining_invalid/total:.2f}%)"
)
print(f"newly recovered correct:   {recovered_total}")

print("\nPer task:")
for task_name, x in sorted(by_task.items()):
    short = task_name.replace("arggym_bbh_pinned_", "")
    print(
        f"{short:48s} "
        f"{100*x['accuracy']:6.2f}%  "
        f"invalid={x['invalid_after_flexible_extract']:3d}/{x['n']:3d}  "
        f"recovered={x['recovered_correct']:3d}"
    )

print("\nArtifacts:")
print(outdir / "summary.json")
print(outdir / "samples.jsonl")
