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
# come from the environment; the Makefile passes them through.
set -euo pipefail

RUNS_DIR=${RUNS_DIR:-outputs/runs}
RESULTS_DIR=${RESULTS_DIR:-results}
FORCE=${FORCE:-}
LFS_POINTER='version https://git-lfs.github.com/spec/v1'

die() { echo "results: $*" >&2; exit 1; }

command -v zstd >/dev/null ||
  die "zstd is not installed. Install it (apt install zstd, brew install zstd) and retry."

add() {
  local id=$1 src=$RUNS_DIR/$1 dst=$RESULTS_DIR/$1
  [ -d "$src" ] || die "$src does not exist."
  for f in run.json metrics.json generations.jsonl; do
    [ -f "$src/$f" ] || die "$src has no $f; only a finished, scored run can be packed."
  done
  grep -Eq '"status": *"completed"' "$src/run.json" ||
    die "$src/run.json does not say \"status\": \"completed\"; refusing to pack it."
  if [ -e "$dst" ] && [ -z "$FORCE" ]; then
    die "$dst already exists. Pass FORCE=1 to replace it."
  fi
  rm -rf "$dst" && mkdir -p "$dst"
  # The recorded taskset is an absolute path on the machine that ran the eval.
  # A taskset under a checkout's data/ is rewritten to data/<name>.jsonl, so
  # `evals.score` finds the committed copy from any checkout's root; the hash
  # beside it still guards against scoring against the wrong file.
  sed -E 's#^(  "taskset": ")[^"]*/(data/[^"/]+\.jsonl)",$#\1\2",#' "$src/run.json" > "$dst/run.json"
  cp "$src/metrics.json" "$dst/metrics.json"
  zstd -19 -T0 -q -f "$src/generations.jsonl" -o "$dst/generations.jsonl.zst"
  zstd -dc "$dst/generations.jsonl.zst" | cmp -s - "$src/generations.jsonl" ||
    die "$dst/generations.jsonl.zst does not decompress to $src/generations.jsonl."
  echo "packed $id: $(du -h "$dst/generations.jsonl.zst" | cut -f1) from $(du -h "$src/generations.jsonl" | cut -f1)"
}

unpack() {
  local id=$1 src=$RESULTS_DIR/$1 dst=$RUNS_DIR/$1
  [ -f "$src/generations.jsonl.zst" ] || die "$src has no generations.jsonl.zst."
  if head -c ${#LFS_POINTER} "$src/generations.jsonl.zst" | grep -qF "$LFS_POINTER"; then
    command -v git-lfs >/dev/null ||
      die "$src/generations.jsonl.zst is an LFS pointer and git-lfs is not installed. Install it, run 'git lfs install' once, then 'git lfs pull'."
    echo "fetching $src/generations.jsonl.zst from LFS"
    git lfs pull --include="$src/generations.jsonl.zst"
    ! head -c ${#LFS_POINTER} "$src/generations.jsonl.zst" | grep -qF "$LFS_POINTER" ||
      die "$src/generations.jsonl.zst is still an LFS pointer after 'git lfs pull'."
  fi
  # A live run writing to the same directory would be overwritten.
  if [ -e "$dst/generations.jsonl" ] && [ -z "$FORCE" ]; then
    die "$dst/generations.jsonl already exists. Pass FORCE=1 to overwrite it."
  fi
  mkdir -p "$dst"
  cp "$src/run.json" "$src/metrics.json" "$dst/"
  zstd -dq -f "$src/generations.jsonl.zst" -o "$dst/generations.jsonl"
  echo "unpacked $id -> $dst"
}

cmd=${1:-}; shift || true
case $cmd in
  add)
    [ $# -gt 0 ] || die "usage: make results-add RUN='<run_id> [<run_id>...]'"
    for id in "$@"; do add "$id"; done ;;
  unpack)
    if [ $# -eq 0 ]; then
      for d in "$RESULTS_DIR"/*/; do [ -f "$d/run.json" ] && set -- "$@" "$(basename "$d")"; done
      [ $# -gt 0 ] || die "$RESULTS_DIR holds no packed runs."
    fi
    for id in "$@"; do unpack "$id"; done ;;
  *) die "usage: $0 add <run_id>... | unpack [<run_id>...]" ;;
esac
