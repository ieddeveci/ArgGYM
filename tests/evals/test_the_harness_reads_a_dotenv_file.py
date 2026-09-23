"""`python -m evals.run` loads `.env` before Hydra resolves `${oc.env:...}`.

In a subprocess, because the load happens once at import and this suite has
already imported the module under its own environment.
"""
import os
import subprocess
import sys


def _probe(tmp_path, env_extra):
    (tmp_path / ".env").write_text("ARGGYM_DOTENV_PROBE=from-file\n")
    env = {k: v for k, v in os.environ.items() if k != "ARGGYM_DOTENV_PROBE"}
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c",
         "import os, evals.run; print(os.environ.get('ARGGYM_DOTENV_PROBE'))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_a_variable_in_dotenv_reaches_the_environment(tmp_path):
    assert _probe(tmp_path, {}) == "from-file"


def test_the_shell_wins_over_the_file(tmp_path):
    assert _probe(tmp_path, {"ARGGYM_DOTENV_PROBE": "from-shell"}) == "from-shell"
