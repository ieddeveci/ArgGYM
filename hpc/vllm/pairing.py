#!/usr/bin/env python3
"""Which serving profile serves which eval config: by file stem, nothing else.

`hpc/vllm/models/<profile>.yaml` says how to serve a checkpoint;
`evals/conf/model/<config>.yaml` says what each request carries, including the
checkpoint id. A config is served by the profile of the same stem; for a
model with reasoning-effort levels, by the profile its `<profile>-<level>` stem
names; and for a model whose thinking is a switch, the thinking-off config
`<profile>-nothink` by the profile that also serves the thinking-on config
`<profile>`. Neither file names the other.

    python3 hpc/vllm/pairing.py hf-qwen3.8-27b-medium   # prints hf-qwen3.8-27b
    python3 hpc/vllm/pairing.py hf-qwen3.5-9b-nothink   # prints hf-qwen3.5-9b

Standard library only, and Python 3.6 syntax: `submit_truba.sh` runs this on the
login node, whose system python3 may be older than the image's.
"""
import sys
from pathlib import Path
from typing import List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "hpc" / "vllm" / "models"
CONFIGS = ROOT / "evals" / "conf" / "model"

#: Every level any config names. Which of them a model accepts is in its configs.
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh")

#: The suffix of a config that sends `chat_template_kwargs.enable_thinking:
#: false`. A switch is not a level: the thinking-on config carries no suffix.
NO_THINKING = "nothink"


def profile_names() -> Set[str]:
    return {p.stem for p in PROFILES.glob("*.yaml")}


def level_of(config: str, profiles: Optional[Set[str]] = None) -> Optional[str]:
    """The effort level a config's stem carries, if the rest of it is a profile."""
    profiles = profile_names() if profiles is None else profiles
    for level in EFFORT_LEVELS:
        if config.endswith(f"-{level}") and config[: -len(level) - 1] in profiles:
            return level
    return None


def profile_of(config: str, profiles: Optional[Set[str]] = None) -> Optional[str]:
    profiles = profile_names() if profiles is None else profiles
    if config in profiles:
        return config
    level = level_of(config, profiles)
    if level:
        return config[: -len(level) - 1]
    base = config[: -len(NO_THINKING) - 1]
    if config.endswith(f"-{NO_THINKING}") and base in profiles:
        return base
    return None


def configs_of(profile: str) -> List[str]:
    """Every eval config the profile serves."""
    profiles = profile_names()
    return sorted(p.stem for p in CONFIGS.glob(f"{profile}*.yaml")
                  if profile_of(p.stem, profiles) == profile)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} EVAL_CONFIG")
    config = sys.argv[1]
    if not (CONFIGS / f"{config}.yaml").is_file():
        near = sorted(p.stem for p in CONFIGS.glob(f"{config}-*.yaml"))
        hint = f" Pick one of: {', '.join(near)}." if near else ""
        raise SystemExit(f"No eval config evals/conf/model/{config}.yaml.{hint}")
    profile = profile_of(config)
    if profile is None:
        raise SystemExit(f"No serving profile under hpc/vllm/models/ serves {config}.")
    print(profile)


if __name__ == "__main__":
    main()
