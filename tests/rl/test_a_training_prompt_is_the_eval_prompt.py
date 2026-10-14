"""A training prompt is the prompt the eval harness sends for the same row.

Three levels, each closer to the wire. The messages: the eval side is the
harness's own solver, built from the Hydra config the eval lane runs
(`model=hf-qwen3-4b`, default template and elicitation), and the request body it
would put on the socket. The token ids, offline: TRL's own prompt tokenizer
against the chat template rendered the way vLLM's chat endpoint renders it.
The token ids, online: vLLM's `/tokenize` on a live server, which runs only when
`ARGGYM_VLLM_URL` names one (the smoke run sets it).
"""
from __future__ import annotations

import json
import os
import types
import urllib.request
from pathlib import Path

import pytest

from evals import taskset
from rl import config as rl_config
from rl import pool

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ("rl/configs/qwen3-4b.yaml", "rl/configs/smoke.yaml")


def sample_rows():
    """Two rows a task from the lite taskset: every answer shape, two levels."""
    _, rows = taskset.load(str(ROOT / "data/taskset-lite.jsonl"))
    out, seen = [], {}
    for r in rows:
        if seen.get(r["task"], 0) < 2 and r["metadata"]["level"] in (1, 8):
            seen[r["task"]] = seen.get(r["task"], 0) + 1
            out.append(r)
    assert len(out) == 24
    return out


def eval_body(row, model="hf-qwen3-4b"):
    from hydra import compose, initialize_config_dir

    from evals.run import build_solver

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "evals/conf")):
        conf = compose(config_name="config", overrides=[f"model={model}"])
    solver = build_solver(conf)
    return solver.client.body(*solver.prompt_of(row))


@pytest.mark.parametrize("path", CONFIGS)
def test_the_messages_are_the_eval_request_messages(path):
    cfg = rl_config.load(str(ROOT / path))
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    for row in sample_rows():
        body = eval_body(row)
        assert pool.messages(row, cfg["prompt"]["template"], elicit) == body["messages"]
        assert body["model"] == cfg["model"]["name"]
        assert body["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": cfg["prompt"]["enable_thinking"]}


def test_a_training_row_carries_the_row_it_is_scored_against():
    cfg = rl_config.load(str(ROOT / CONFIGS[0]))
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    rows = sample_rows()
    for row, trl in zip(rows, pool.trl_rows(rows, cfg["prompt"]["template"], elicit)):
        assert json.loads(trl["row_json"]) == row
        assert trl["prompt"][-1]["content"].startswith(row["question"])


def _tokenizer(cfg):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            cfg["model"]["name"], revision=cfg["model"]["revision"])
    except OSError as e:  # no network and no cached copy
        pytest.skip(f"tokenizer unavailable: {e}")


def trl_prompt_ids(tokenizer, messages, chat_template_kwargs):
    """The ids `GRPOTrainer` sends to vLLM for these messages, from TRL's own method."""
    trl = pytest.importorskip("trl")
    stub = types.SimpleNamespace(
        _is_vlm=False, environment_factories=None, processing_class=tokenizer, tools=None,
        chat_template=None, chat_template_kwargs=chat_template_kwargs)
    ids, _, _ = trl.GRPOTrainer._tokenize_prompts(stub, [messages])
    return list(ids[0])


def test_the_token_ids_are_the_ids_the_eval_server_renders():
    cfg = rl_config.load(str(ROOT / CONFIGS[0]))
    tok = _tokenizer(cfg)
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    kwargs = {"enable_thinking": cfg["prompt"]["enable_thinking"]}
    for row in sample_rows():
        body = eval_body(row)
        # vLLM's chat endpoint renders the template to text, then tokenizes the
        # text without adding special tokens.
        text = tok.apply_chat_template(body["messages"], tokenize=False,
                                       add_generation_prompt=True,
                                       **body["extra_body"]["chat_template_kwargs"])
        want = tok(text, add_special_tokens=False)["input_ids"]
        got = trl_prompt_ids(tok, pool.messages(row, cfg["prompt"]["template"], elicit), kwargs)
        assert got == want, row["id"]


@pytest.mark.skipif(not os.environ.get("ARGGYM_VLLM_URL"),
                    reason="needs a live vLLM server: set ARGGYM_VLLM_URL=http://host:port")
def test_the_token_ids_are_the_ids_a_live_vllm_server_renders():
    url = os.environ["ARGGYM_VLLM_URL"].rstrip("/")
    cfg = rl_config.load(str(ROOT / CONFIGS[0]))
    tok = _tokenizer(cfg)
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    kwargs = {"enable_thinking": cfg["prompt"]["enable_thinking"]}
    for row in sample_rows():
        body = eval_body(row)
        req = {"model": body["model"], "messages": body["messages"],
               "add_generation_prompt": True, **body["extra_body"]}
        r = urllib.request.Request(f"{url}/tokenize", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=60) as resp:
            served = json.loads(resp.read())["tokens"]
        got = trl_prompt_ids(tok, pool.messages(row, cfg["prompt"]["template"], elicit), kwargs)
        assert got == served, row["id"]
