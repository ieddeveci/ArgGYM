from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import hashlib
import inspect
import json

from transformers import AutoTokenizer

from transfer.registry import ADAPTERS


ROOT = Path("/arf/scratch/futan/ArgGYM")

MODEL = Path(
    "/arf/scratch/futan/arggym_hf_cache/hub/"
    "models--Qwen--Qwen3-14B/snapshots/"
    "40c069824f4251a91eefaf281ebe4c544efd3e18"
)

PIN = (
    ROOT
    / "transfer/lm_eval_tasks/pinned_20260920_qwen3"
)

OUT = (
    ROOT
    / "outputs/transfer/protocol_freeze/"
    "prompt_budget_audit_v3_20260920.json"
)

MAX_MODEL_LEN = 40960
# 8192 is the largest completion cap in the final transfer protocol.
# Benchmarks with smaller caps therefore receive an even larger safety margin.
MAX_COMPLETION = 8192
MAX_PROMPT = MAX_MODEL_LEN - MAX_COMPLETION

if OUT.exists():
    raise RuntimeError(
        f"Refusing to overwrite existing audit: {OUT}"
    )

assert MAX_PROMPT == 32768

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
)


def sha_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_message(m):
    if isinstance(m, dict):
        role = m["role"]
        content = m["content"]
    else:
        role = getattr(m, "role")
        content = getattr(m, "content")

        if hasattr(role, "value"):
            role = role.value

    assert role in {"system", "user", "assistant"}
    assert isinstance(content, str)

    return {
        "role": str(role),
        "content": content,
    }


def render_chat(
    messages,
    add_generation_prompt=True,
    **kwargs,
):
    normalized = [
        normalize_message(m)
        for m in messages
    ]

    return tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )


def count_rendered(text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


records = []
global_max = None


def record(
    source: str,
    item_id: str,
    prompt_tokens: int,
    rendered_sha256: str,
):
    global global_max

    r = {
        "source": source,
        "item_id": item_id,
        "prompt_tokens": prompt_tokens,
        "rendered_sha256": rendered_sha256,
    }

    records.append(r)

    if (
        global_max is None
        or prompt_tokens
        > global_max["prompt_tokens"]
    ):
        global_max = dict(r)

    if prompt_tokens > MAX_PROMPT:
        raise RuntimeError(
            f"PROMPT BUDGET EXCEEDED: "
            f"{source}::{item_id}: "
            f"{prompt_tokens} > {MAX_PROMPT}"
        )


# ============================================================
# 1. CUSTOM BENCHMARKS — all 7,944 frozen prompts
# ============================================================

custom_counts = {}
custom_max = {}

for benchmark, cls in ADAPTERS.items():
    adapter = cls()

    n = 0
    max_tokens = -1
    max_id = None

    for row in adapter.rows():
        messages = adapter.messages(row)

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        ntok = count_rendered(rendered)

        record(
            f"custom:{benchmark}",
            row.row_id,
            ntok,
            sha_text(rendered),
        )

        n += 1

        if ntok > max_tokens:
            max_tokens = ntok
            max_id = row.row_id

    custom_counts[benchmark] = n
    custom_max[benchmark] = {
        "tokens": max_tokens,
        "row_id": max_id,
    }

    print(
        "CUSTOM",
        benchmark,
        "count=",
        n,
        "max_prompt_tokens=",
        max_tokens,
        "max_row=",
        max_id,
        flush=True,
    )

assert sum(custom_counts.values()) == 7944


# ============================================================
# 2. LM-EVAL — actual request construction
# ============================================================

from lm_eval.tasks import TaskManager
from lm_eval.models.vllm_causallms import VLLM


# We will use these exact options during inference.
sig = inspect.signature(VLLM.__init__)

for required in (
    "max_model_len",
    "enable_thinking",
    "think_end_token",
):
    if required not in sig.parameters:
        raise RuntimeError(
            f"Pinned lm-eval VLLM backend lacks "
            f"{required}"
        )

tm = TaskManager(
    include_path=str(PIN),
)

names = set(tm.task_index)

lm_names = (
    ["arggym_gsm8k_cot_pinned"]
    + sorted(
        x
        for x in names
        if x.startswith(
            "arggym_mmlu_pro_pinned_"
        )
    )
    + sorted(
        x
        for x in names
        if x.startswith(
            "arggym_bbh_pinned_"
        )
    )
)

assert len(lm_names) == 42, len(lm_names)

lm_counts = {}
lm_max = {}


def walk_tasks(obj):
    if (
        hasattr(obj, "build_all_requests")
        and hasattr(obj, "instances")
    ):
        yield obj
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk_tasks(value)
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from walk_tasks(value)


for task_name in lm_names:
    # Modern pinned lm-eval API:
    # TaskManager.load() returns a flat mapping of leaf tasks
    # under loaded["tasks"] plus separate group metadata.
    loaded = tm.load(
        [task_name]
    )

    task_objects = list(
        loaded["tasks"].values()
    )

    if len(task_objects) != 1:
        raise RuntimeError(
            f"{task_name}: expected one leaf task "
            f"object, got {len(task_objects)}; "
            f"task_keys={list(loaded['tasks'])}"
        )

    task = task_objects[0]

    if hasattr(task, "set_fewshot_seed"):
        task.set_fewshot_seed(
            seed=1234
        )

    task.build_all_requests(
        limit=None,
        samples=None,
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
        system_instruction=None,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        chat_template=render_chat,
        tokenizer_name=str(MODEL),
    )

    n = 0
    max_tokens = -1
    max_item = None

    for j, instance in enumerate(
        task.instances
    ):
        if not instance.args:
            raise RuntimeError(
                f"{task_name}: empty request args"
            )

        ctx = instance.args[0]

        if isinstance(ctx, str):
            contexts = [ctx]
        elif (
            isinstance(ctx, list)
            and all(
                isinstance(x, str)
                for x in ctx
            )
        ):
            contexts = ctx
        else:
            raise RuntimeError(
                f"{task_name}: unexpected "
                f"context type {type(ctx)}"
            )

        for k, rendered in enumerate(
            contexts
        ):
            ntok = count_rendered(
                rendered
            )

            item_id = f"{j}:{k}"

            record(
                f"lm_eval:{task_name}",
                item_id,
                ntok,
                sha_text(rendered),
            )

            n += 1

            if ntok > max_tokens:
                max_tokens = ntok
                max_item = item_id

    lm_counts[task_name] = n
    lm_max[task_name] = {
        "tokens": max_tokens,
        "item_id": max_item,
    }

    print(
        "LM_EVAL",
        task_name,
        "count=",
        n,
        "max_prompt_tokens=",
        max_tokens,
        flush=True,
    )


summary = {
    "status":
        "FINAL_PRE_RL_PROMPT_BUDGET_PASS",
    "model":
        "Qwen/Qwen3-14B",
    "model_revision":
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "thinking_enabled":
        True,
    "max_model_len":
        MAX_MODEL_LEN,
    "reserved_completion_tokens":
        MAX_COMPLETION,
    "max_allowed_prompt_tokens":
        MAX_PROMPT,
    "custom_total":
        sum(custom_counts.values()),
    "lm_eval_task_configs":
        len(lm_names),
    "lm_eval_request_total":
        sum(lm_counts.values()),
    "custom_counts":
        custom_counts,
    "custom_max":
        custom_max,
    "lm_eval_counts":
        lm_counts,
    "lm_eval_max":
        lm_max,
    "global_max":
        global_max,
}

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print()
print(
    "GLOBAL MAX PROMPT TOKENS:",
    global_max["prompt_tokens"],
)
print(
    "GLOBAL MAX ITEM:",
    global_max["source"],
    global_max["item_id"],
)
print(
    "RESERVED COMPLETION:",
    MAX_COMPLETION,
)
print(
    "MAX MODEL LEN:",
    MAX_MODEL_LEN,
)
print(
    "FINAL PRE-RL PROMPT BUDGET AUDIT: PASS"
)
