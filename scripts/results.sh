#!/usr/bin/env bash
# Pack finished eval runs into results/ and unpack them back into run directories.
#
#   scripts/results.sh add    <run_id>...   outputs/runs/<id>  -> results/<id>
#   scripts/results.sh unpack [<run_id>...] results/<id>       -> outputs/runs/<id>  (no id: all)
#
# A packed run is run.json and metrics.json as plain files and generations.jsonl
# as generations.jsonl.zst, which .gitattributes sends to Git LFS. Everything
# else in a run directory is rebuilt (prompts.jsonl, samples.jsonl) or local
# (run.lock, run.log, .hydra/).
#
# RUNS_DIR (default outputs/runs), RESULTS_DIR (default results) and FORCE=1
# come from the environment; the Makefile passes them through. A relative
# RUNS_DIR or RESULTS_DIR is taken from the directory the script is called in;
# the defaults are taken from the repo root. The script itself runs from the
# root of the git checkout it is called in, so it works from any subdirectory.
set -euo pipefail

LFS_POINTER='version https://git-lfs.github.com/spec/v1'

die() { echo "results: $*" >&2; exit 1; }

command -v zstd >/dev/null ||
  die "zstd is not installed. Install it (apt install zstd, brew install zstd) and retry."
command -v python >/dev/null || die "python is not installed."

absolute() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
[ -n "${RUNS_DIR:-}" ] && RUNS_DIR=$(absolute "$RUNS_DIR")
[ -n "${RESULTS_DIR:-}" ] && RESULTS_DIR=$(absolute "$RESULTS_DIR")
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "run this inside the ArgGYM git checkout."
cd "$ROOT"
RUNS_DIR=${RUNS_DIR:-outputs/runs}
RESULTS_DIR=${RESULTS_DIR:-results}

case ${FORCE:-} in 1|true|yes) FORCE=1 ;; *) FORCE= ;; esac

# The JSON edits go through python3, so the files keep the exact layout
# `evals.artifacts.write_json` gives them (indent 2) and only the paths change.
json() { python - "$@"; }

# Checks the taskset a run was generated against and prints data/<name>.jsonl.
# The recorded taskset is a path on the machine that ran the eval. Only a
# taskset directly under a checkout's data/ can be shared, because only there
# does every checkout have a copy, and that copy must carry the run's hash.
taskset_of() {
  json "$1" <<'PY'
import json, re, sys
run = json.load(open(sys.argv[1]))
path, want = run.get("taskset", ""), run.get("taskset_hash")
m = re.search(r"(?:^|/)(data/[^/]+\.jsonl)$", path)
if not m:
    sys.exit(f"the run's taskset {path} is not directly under a data/ folder, so no "
             "checkout has a copy of it; only a run against a taskset in data/ can be shared.")
rel = m.group(1)
try:
    with open(rel) as f:
        got = json.loads(f.readline()).get("__manifest__", {}).get("taskset_hash")
except FileNotFoundError:
    sys.exit(f"the run's taskset {path} would become {rel}, and this repo has no {rel}.")
if got != want:
    sys.exit(f"{rel} has taskset_hash {got}, and the run was generated against "
             f"taskset_hash {want}; refusing to point the run at a different taskset.")
print(rel)
PY
}

# Writes run.json and metrics.json with the taskset as data/<name>.jsonl and the
# run directory as outputs/runs/<id>, so no absolute path of the packing machine
# is committed.
rewrite() {  # src dst id taskset
  json "$@" <<'PY'
import json, os, sys
src, dst, run_id, taskset = sys.argv[1:]
def edit(name, fn):
    obj = json.load(open(os.path.join(src, name)))
    fn(obj)
    with open(os.path.join(dst, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)
def run(o):
    o["taskset"] = taskset
def metrics(o):
    meta = o["_meta"]
    if meta.get("taskset_hash") != json.load(open(os.path.join(src, "run.json")))["taskset_hash"]:
        sys.exit(f"{src}/metrics.json was scored against a different taskset_hash than "
                 "run.json records; rescore the run before packing it.")
    meta["run_dir"], meta["taskset"] = f"outputs/runs/{run_id}", taskset
edit("run.json", run)
edit("metrics.json", metrics)
PY
}

check_add() {
  local src=$RUNS_DIR/$1 dst=$RESULTS_DIR/$1
  [ -d "$src" ] || die "$src does not exist."
  for f in run.json metrics.json generations.jsonl; do
    [ -f "$src/$f" ] || die "$src has no $f; only a finished, scored run can be packed."
  done
  grep -Eq '"status": *"completed"' "$src/run.json" ||
    die "$src/run.json does not say \"status\": \"completed\"; refusing to pack it."
  if [ -e "$dst" ] && [ -z "$FORCE" ]; then
    die "$dst already exists. Pass FORCE=1 to replace it."
  fi
  taskset_of "$src/run.json" >/dev/null || die "cannot pack $1."
}

add() {
  local id=$1 src=$RUNS_DIR/$1 dst=$RESULTS_DIR/$1 taskset
  taskset=$(taskset_of "$src/run.json")
  rm -rf "$dst" && mkdir -p "$dst"
  rewrite "$src" "$dst" "$id" "$taskset" || die "cannot pack $id."
  zstd -19 -T0 -q -f "$src/generations.jsonl" -o "$dst/generations.jsonl.zst"
  zstd -dc "$dst/generations.jsonl.zst" | cmp -s - "$src/generations.jsonl" ||
    die "$dst/generations.jsonl.zst does not decompress to $src/generations.jsonl."
  echo "packed $id: $(du -h "$dst/generations.jsonl.zst" | cut -f1) from $(du -h "$src/generations.jsonl" | cut -f1)"
}

is_pointer() { head -c ${#LFS_POINTER} "$1" | grep -qF "$LFS_POINTER"; }

unpack() {
  local id=$1 src=$RESULTS_DIR/$1 dst=$RUNS_DIR/$1 zst rel tmp
  zst=$src/generations.jsonl.zst
  [ -f "$zst" ] || die "$src has no generations.jsonl.zst."
  if is_pointer "$zst"; then
    # `git lfs pull --include` matches paths relative to the repo root, so an
    # absolute or ./ RESULTS_DIR has to be made relative first.
    rel=$(cd "$src" && pwd -P)/generations.jsonl.zst
    case $rel in "$ROOT"/*) rel=${rel#"$ROOT"/} ;; *) die "$zst is an LFS pointer outside this repo, so git lfs cannot fetch it." ;; esac
    command -v git-lfs >/dev/null ||
      die "$zst is an LFS pointer and git-lfs is not installed. Install it, run 'git lfs install' once, then 'git lfs pull --include=\"$rel\"'."
    echo "fetching $rel from LFS"
    git lfs pull --include="$rel"
    ! is_pointer "$zst" ||
      die "$zst is still an LFS pointer after 'git lfs pull --include=\"$rel\"'."
  fi
  # A live run writing to the same directory would be overwritten.
  if [ -e "$dst/generations.jsonl" ] && [ -z "$FORCE" ]; then
    die "$dst/generations.jsonl already exists. Pass FORCE=1 to overwrite it."
  fi
  # Decompress to a temporary file first, so a corrupt .zst restores nothing.
  mkdir -p "$RUNS_DIR"
  tmp=$RUNS_DIR/.$id.generations.jsonl.$$.tmp
  zstd -dq -f "$zst" -o "$tmp" || { rm -f "$tmp"; die "$zst does not decompress; nothing was restored."; }
  mkdir -p "$dst"
  mv -f "$tmp" "$dst/generations.jsonl"
  cp "$src/run.json" "$src/metrics.json" "$dst/"
  echo "unpacked $id -> $dst"
}

cmd=${1:-}; shift || true
case $cmd in
  add)
    [ $# -gt 0 ] || die "usage: make results-add RUN='<run_id> [<run_id>...]'"
    # A git that does not know the lfs filter ignores `filter=lfs` in
    # .gitattributes and would commit the .zst as a plain blob.
    git config --get filter.lfs.clean >/dev/null ||
      die "Git LFS is not set up in this checkout, so the .zst would be committed as a plain git file. Install git-lfs and run 'git lfs install' once, then retry."
    for id in "$@"; do check_add "$id"; done
    for id in "$@"; do add "$id"; done ;;
  unpack)
    if [ $# -eq 0 ]; then
      for d in "$RESULTS_DIR"/*/; do [ -f "$d/run.json" ] && set -- "$@" "$(basename "$d")"; done
      [ $# -gt 0 ] || die "$RESULTS_DIR holds no packed runs."
    fi
    for id in "$@"; do [ -f "$RESULTS_DIR/$id/run.json" ] || die "$RESULTS_DIR/$id is not a packed run."; done
    for id in "$@"; do unpack "$id"; done ;;
  *) die "usage: $0 add <run_id>... | unpack [<run_id>...]" ;;
esac
