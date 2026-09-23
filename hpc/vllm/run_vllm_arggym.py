#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from pairing import CONFIGS, PROFILES, ROOT, configs_of, profile_of

OUT = ROOT / "outputs" / "runs"


def load_run(config: str) -> tuple[str, dict, str]:
    """The profile that serves an eval config, its settings, and the checkpoint.

    The checkpoint id lives in the eval config's `model` and nowhere else; the
    profile says only how to serve it. Every config a profile serves must name
    the same checkpoint, since one server answers all of them.
    """
    path = CONFIGS / f"{config}.yaml"
    if not path.is_file():
        near = sorted(p.stem for p in CONFIGS.glob(f"{config}-*.yaml"))
        hint = f" Pick one of: {', '.join(near)}." if near else ""
        raise SystemExit(f"No eval config evals/conf/model/{config}.yaml.{hint}")
    profile = profile_of(config)
    if profile is None:
        raise SystemExit(f"No serving profile under hpc/vllm/models/ serves {config}.")
    cfg = yaml.safe_load((PROFILES / f"{profile}.yaml").read_text(encoding="utf-8"))
    if cfg["dtype"] != "bfloat16":
        raise SystemExit("This benchmark lane requires BF16 weights.")
    models = {yaml.safe_load((CONFIGS / f"{c}.yaml").read_text(encoding="utf-8"))["model"]
              for c in configs_of(profile)}
    if len(models) != 1:
        raise SystemExit(f"The configs {profile} serves name different models: "
                         f"{sorted(models)}")
    return profile, cfg, models.pop()


def resolve_revision(repo: str) -> str:
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo, safe="/")
    headers = {"Accept": "application/json", "User-Agent": "arggym-vllm/2"}
    if os.getenv("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        data = json.load(response)
    sha = data.get("sha")
    if not sha:
        raise SystemExit(f"Hugging Face revision unavailable for {repo}")
    return sha


def selected_tp(cfg: dict) -> int:
    """TP_SIZE if set (submit_truba.sh always sets it), else the profile's count
    for GPU_TYPE."""
    gpu = os.getenv("GPU_TYPE", "H100").lower()
    return int(os.getenv("TP_SIZE", cfg[f"tensor_parallel_size_{gpu}"]))


def vllm_command(cfg: dict, model: str, revision: str, port: int) -> list[str]:
    cmd = [
        os.getenv("VLLM_BIN", "vllm"),
        "serve",
        model,
        "--revision", revision,
        "--served-model-name", model,
        "--dtype", cfg["dtype"],
        "--tensor-parallel-size", str(selected_tp(cfg)),
        "--max-model-len", str(cfg["max_model_len"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        # Experimental control: do not silently inherit generation_config.json
        # defaults from the model repository.
        "--generation-config", "vllm",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if cfg.get("reasoning_parser"):
        cmd += ["--reasoning-parser", cfg["reasoning_parser"]]
    if cfg.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]
    cmd += [str(arg) for arg in cfg.get("extra_args", [])]
    return cmd


def wait_until_ready(base_url: str, proc: subprocess.Popen, expected_model: str,
                     timeout_s: int = 7200) -> None:
    deadline = time.monotonic() + timeout_s
    endpoint = base_url.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"vLLM exited during startup with code {proc.returncode}")
        try:
            req = urllib.request.Request(
                endpoint, headers={"Authorization": "Bearer none"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status < 300:
                    payload = json.load(response)
                    model_ids = {
                        item.get("id")
                        for item in payload.get("data", [])
                        if isinstance(item, dict)
                    }
                    if expected_model in model_ids:
                        return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit(
        f"vLLM startup timeout: {expected_model} was not visible at {endpoint}"
    )


def stop_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def get_vllm_version(vllm_bin: str) -> str | None:
    try:
        out = subprocess.check_output(
            [vllm_bin, "--version"], text=True, stderr=subprocess.STDOUT, timeout=30
        )
        return out.strip()
    except Exception:
        return None


def prepare_run_directory(
    run_dir: Path,
    profile: str,
    cfg: dict,
    model: str,
    revision: str,
    command: list[str],
    resume: bool,
    config: str,
) -> dict:
    manifest_path = run_dir / "hf_runtime_start.json"
    existing_payload = None

    if run_dir.exists() and any(run_dir.iterdir()):
        if manifest_path.exists():
            existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_payload.get("hf_revision") != revision:
                raise SystemExit(
                    "Refusing to mix Hugging Face revisions in one ArgGYM run directory: "
                    f"{existing_payload.get('hf_revision')} != {revision}"
                )
            if existing_payload.get("hf_model") != model:
                raise SystemExit("Refusing to mix model identities in one run directory.")
        elif not resume:
            raise SystemExit(
                f"Run directory already exists and has no HF provenance manifest: {run_dir}\n"
                "Move/archive it before a clean run."
            )
        elif resume:
            raise SystemExit(
                f"Cannot safely resume legacy run without hf_runtime_start.json: {run_dir}"
            )

        # Any benchmark artifacts mean this is an intentional resume.
        benchmark_files = {
            "generations.jsonl", "prompts.jsonl", "run.json",
            "samples.jsonl", "metrics.json"
        }
        if any((run_dir / name).exists() for name in benchmark_files) and not resume:
            raise SystemExit(
                f"Run directory already contains benchmark artifacts: {run_dir}\n"
                "Use RESUME=1 only if you intentionally want to resume the exact same run."
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "profile": profile,
        "eval_config": config,
        "hf_model": model,
        "hf_revision": revision,
        "dtype": cfg["dtype"],
        "tensor_parallel_size": selected_tp(cfg),
        "max_model_len": cfg["max_model_len"],
        "reasoning_parser": cfg.get("reasoning_parser"),
        "vllm_command": command,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "gpu_type": os.getenv("GPU_TYPE"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "vllm_version": get_vllm_version(command[0]),
        "python_version": sys.version.split()[0],
        "container_image": os.getenv("ARGGYM_RUNTIME_IMAGE"),
        "container_image_sha256": os.getenv("ARGGYM_RUNTIME_SHA256"),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="eval config under evals/conf/model/, "
                                         "e.g. hf-qwen3-8b or hf-qwen3.8-27b-medium")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the vLLM command and run directory, then stop")
    parser.add_argument("--template", default="xml_tags")
    parser.add_argument("--elicitation", default="cot")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    profile, cfg, model = load_run(args.config)
    run_dir = OUT / f"{args.config}__{args.template}__{args.elicitation}"
    if args.dry_run:
        revision = str(cfg.get("revision") or "<resolved at start>")
        print("Profile:", profile)
        print("Model:", model)
        print("vLLM command:", " ".join(vllm_command(cfg, model, revision, args.port)))
        print("Run directory:", run_dir)
        return
    revision = str(cfg.get("revision") or resolve_revision(model))
    base_url = f"http://127.0.0.1:{args.port}/v1"
    command = vllm_command(cfg, model, revision, args.port)

    runtime = prepare_run_directory(
        run_dir, profile, cfg, model, revision, command, args.resume, args.config
    )

    logs = ROOT / "outputs" / "vllm_logs"
    logs.mkdir(parents=True, exist_ok=True)
    job_id = os.getenv("SLURM_JOB_ID", "local")
    log_path = logs / f"{args.config}-{job_id}.log"

    print("HF revision:", revision, flush=True)
    print("vLLM command:", " ".join(command), flush=True)
    print("vLLM log:", log_path, flush=True)

    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_until_ready(base_url, proc, model)

            env = os.environ.copy()
            env["VLLM_BASE_URL"] = base_url
            env.setdefault("VLLM_API_KEY", "none")
            env.setdefault("PYTHONHASHSEED", "0")

            subprocess.run(
                [
                    sys.executable, "-m", "evals.run",
                    f"model={args.config}",
                    f"template={args.template}",
                    f"elicitation={args.elicitation}",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )

            run_json = run_dir / "run.json"
            if not run_json.is_file():
                raise SystemExit(f"Expected ArgGYM run metadata not found: {run_json}")

            run_meta = json.loads(run_json.read_text(encoding="utf-8"))
            run_meta["hf_runtime"] = runtime
            run_json.write_text(
                json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            if not args.skip_score:
                subprocess.run(
                    [sys.executable, "-m", "evals.score", str(run_dir)],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
        finally:
            stop_process_group(proc)


if __name__ == "__main__":
    main()
