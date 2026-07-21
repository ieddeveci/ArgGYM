#!/usr/bin/env bash
# Serve one model on GPU 2 and gate it behind a health check and a smoke test.
#
# Usage:  serve.sh start <model-config-name>
#         serve.sh stop
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
CONTAINER=arggym-vllm
PORT=8900
READY_TIMEOUT=${READY_TIMEOUT:-2400}

stop_server() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  # vLLM needs a moment to release GPU memory before the next model loads.
  sleep 10
}

case "${1:-}" in
  stop)
    stop_server
    echo "stopped"
    exit 0
    ;;
  start) ;;
  *)
    echo "usage: $0 {start <model>|stop}" >&2
    exit 2
    ;;
esac

MODEL_CFG="${2:?model config name required}"
eval "$("$PY" -m evals.serve_args "$MODEL_CFG")" || { echo "bad model config"; exit 3; }

echo "=== serving $MODEL_NAME ($IMAGE) on GPUs $GPUS"
echo "    args: $VLLM_ARGS"
stop_server

# shellcheck disable=SC2086
docker run -d --rm \
  --runtime nvidia --gpus "\"device=$GPUS\"" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p "$PORT:$PORT" \
  --ipc=host \
  --name "$CONTAINER" \
  "$IMAGE" $VLLM_ARGS >/dev/null || { echo "FAIL: docker run"; exit 4; }

echo -n "    waiting for /health "
deadline=$(( SECONDS + READY_TIMEOUT ))
while (( SECONDS < deadline )); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo " ready after ${SECONDS}s"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^$CONTAINER$"; then
    echo
    echo "FAIL: container exited during load. Last 40 lines:"
    docker logs "$CONTAINER" 2>&1 | tail -40
    exit 5
  fi
  echo -n "."
  sleep 10
done

if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo
  echo "FAIL: not ready within ${READY_TIMEOUT}s"
  docker logs "$CONTAINER" 2>&1 | tail -40
  stop_server
  exit 6
fi

# Smoke test: one real completion. Fails the model here rather than after an
# hour of wasted GPU time, and records whether reasoning_content is populated.
echo "    smoke test..."
"$PY" - "$MODEL_NAME" <<'EOF'
import json, sys, urllib.request
name = sys.argv[1]
body = {"model": name,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 512, "temperature": 0.0}
req = urllib.request.Request("http://localhost:8900/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.load(r)
msg = (out["choices"][0].get("message") or {})
content = (msg.get("content") or "").strip()
reasoning = (msg.get("reasoning_content") or "").strip()
print(f"      content={content[:60]!r}")
print(f"      reasoning_content={'populated' if reasoning else 'EMPTY'} "
      f"({len(reasoning)} chars)")
if not content and not reasoning:
    print("      FAIL: empty response")
    sys.exit(7)
print("      smoke OK")
EOF
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAIL: smoke test (rc=$rc)"
  docker logs "$CONTAINER" 2>&1 | tail -30
  stop_server
  exit 7
fi

echo "=== $MODEL_NAME ready"
