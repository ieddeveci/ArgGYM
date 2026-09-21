"""Record the checkpoint selected by generated-dev reward only.

`GRPOConfig(load_best_model_at_end=True, metric_for_best_model="eval_reward")`
causes Transformers to retain and reload this checkpoint. This helper writes an
explicit audit record for the article. It never reads `data/taskset.jsonl`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--metric", default="eval_reward")
    a = p.parse_args(argv)
    run = Path(a.run_dir)
    state_path = run / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    best_path = state.get("best_model_checkpoint")
    best_metric = state.get("best_metric")
    metric_name = state.get("best_metric_key") or a.metric
    if best_path and best_metric is not None:
        ckpt = Path(best_path)
        if not ckpt.is_absolute():
            # HF normally writes the path relative to the process cwd/output
            # configuration. Accept a basename relative to run for portability.
            candidate = run / ckpt.name
            if candidate.exists():
                ckpt = candidate
        if not ckpt.exists():
            raise SystemExit(
                f"Trainer selected {best_path}, but that checkpoint is absent; "
                "do not publish a different surviving checkpoint as the selected model"
            )
        try:
            step = int(ckpt.name.rsplit("-", 1)[1])
        except Exception:
            step = None
        payload = {
            "selection_source": "RL_DEV",
            "metric": metric_name,
            "value": float(best_metric),
            "step": step,
            "checkpoint": str(ckpt),
        }
    else:
        # Defensive fallback for a smoke run made with load_best_model_at_end
        # disabled. Only saved checkpoints carrying the requested eval metric
        # are eligible.
        candidates = []
        for rec in state.get("log_history", []):
            if a.metric not in rec or "step" not in rec:
                continue
            step = int(rec["step"])
            ckpt = run / f"checkpoint-{step}"
            if ckpt.exists():
                candidates.append((float(rec[a.metric]), step, ckpt))
        if not candidates:
            keys = sorted(
                {k for r in state.get("log_history", []) for k in r if k.startswith("eval_")}
            )
            raise SystemExit(
                f"no saved checkpoint has metric {a.metric!r}; available eval keys: {keys}"
            )
        reward, step, ckpt = max(candidates, key=lambda x: (x[0], x[1]))
        payload = {
            "selection_source": "RL_DEV",
            "metric": a.metric,
            "value": reward,
            "step": step,
            "checkpoint": str(ckpt),
        }

    (run / "selected_checkpoint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
