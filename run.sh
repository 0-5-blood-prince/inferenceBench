#!/usr/bin/env bash
# Engine lifecycle and run orchestration (SPEC section 8).
#
# Runs ON the pod. Engines are never co-resident - two ~60 GB weight sets do not
# fit one 80 GB card - so every command brings one engine up, uses it, and takes
# it down, waiting for the driver to actually release VRAM before the next.
#
#   ./run.sh bootstrap          clocks, disk check, one venv per engine, freezes
#   ./run.sh up vllm            start an engine, block until healthy
#   ./run.sh up cold vllm       start with prefix caching DISABLED (Cold block)
#   ./run.sh down               stop whatever is running, wait for VRAM release
#   ./run.sh p0                 probe image tokens, then build + verify
#   ./run.sh p1                 the full P1 gate sequence, both engines
#   ./run.sh cell <workload> <engine> <rate> <run_id>
#
# Pod images cannot run nested Docker, so P0's "pinned container digests" control
# is replaced by the next best thing: one virtualenv per engine, exact pinned
# versions, and a committed pip freeze per engine. Separate venvs matter because
# vLLM and SGLang pull different torch builds; sharing one environment means
# whichever installed last silently defines the other's numerics.
set -euo pipefail

MODEL="${MODEL:?set MODEL to the locked model id}"
WORKSPACE="${WORKSPACE:-/workspace}"          # the resized volume
HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"

# Pinned at P0, unchanged all day. Record these in the P2 commit.
VLLM_VERSION="${VLLM_VERSION:?set VLLM_VERSION, e.g. 0.11.0}"
SGLANG_VERSION="${SGLANG_VERSION:?set SGLANG_VERSION, e.g. 0.4.6}"

GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.90}"          # pinned equal across engines (P1)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LOCK_CLOCK="${LOCK_CLOCK:-1200}"              # MHz; recorded in the P0 log
MIN_FREE_GB="${MIN_FREE_GB:-130}"             # ~60 GB weights + cache + headroom

VLLM_PORT=8000
SGLANG_PORT=30000
VENV_DIR="$WORKSPACE/venvs"
PIDFILE="$WORKSPACE/.engine.pid"

log() { printf '\n=== %s\n' "$*" >&2; }
port_for() { [ "$1" = vllm ] && echo "$VLLM_PORT" || echo "$SGLANG_PORT"; }
base_url() { echo "http://localhost:$(port_for "$1")"; }
py_for() { echo "$VENV_DIR/$1/bin/python"; }

bootstrap() {
  log "GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  local mem
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  # 40 GB forces int4, which SPEC section 4 puts out of scope. Fail loudly rather
  # than silently turning this into a different study.
  if [ "$mem" -lt 70000 ]; then
    echo "FATAL: ${mem} MiB VRAM. Need 80GB, or drop to a ~7B VLM per P0." >&2
    exit 1
  fi

  log "disk"
  mkdir -p "$HF_HOME" "$VENV_DIR"
  local free_gb
  free_gb=$(df -BG --output=avail "$WORKSPACE" | tail -1 | tr -dc '0-9')
  echo "$WORKSPACE has ${free_gb} GB free (need >= ${MIN_FREE_GB})"
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "FATAL: volume too small for a 30B model in bf16. Resize it, or take" >&2
    echo "       P0's fallback and drop to a ~7B VLM with the design unchanged." >&2
    exit 1
  fi

  log "locking clocks (kills thermal drift across the 3-hour matrix)"
  nvidia-smi -pm 1 || true
  nvidia-smi -lgc "$LOCK_CLOCK" || echo "WARNING: clocks not locked; record this" >&2

  # One venv per engine. Installed once, up front - never mid-matrix.
  mkdir -p env
  for engine in vllm sglang; do
    log "building venv: $engine"
    python3 -m venv "$VENV_DIR/$engine"
    local pip="$VENV_DIR/$engine/bin/pip"
    "$pip" install -q --upgrade pip
    if [ "$engine" = vllm ]; then
      "$pip" install -q "vllm==$VLLM_VERSION"
    else
      "$pip" install -q "sglang[all]==$SGLANG_VERSION"
    fi
    # The freeze is the audit trail that replaces the image digest.
    "$pip" freeze > "env/${engine}-freeze.txt"
    "$VENV_DIR/$engine/bin/python" -c \
      'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' \
      | tee "env/${engine}-torch.txt"
  done

  log "client deps"
  pip install -q -r requirements.txt

  {
    echo "model=$MODEL"
    echo "vllm=$VLLM_VERSION"
    echo "sglang=$SGLANG_VERSION"
    echo "gpu_mem_frac=$GPU_MEM_FRAC"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "locked_clock_mhz=$LOCK_CLOCK"
  } | tee env/pinned.txt
}

wait_vram_free() {
  # Engines are never co-resident. Starting the second before the first has
  # released 60 GB gives an OOM that looks like an engine bug.
  local deadline=$((SECONDS + 180)) used
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break
    if [ $SECONDS -gt $deadline ]; then
      echo "WARNING: ${used} MiB still resident after 180s" >&2
      break
    fi
    sleep 3
  done
}

down() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    sleep 5
    kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  pkill -f "sglang.launch_server" 2>/dev/null || true
  wait_vram_free
}

wait_healthy() {
  local url="$1" logfile="$2" deadline=$((SECONDS + 1800))
  until curl -sf "$url/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "FATAL: server exited during startup" >&2; tail -40 "$logfile" >&2; exit 1
    fi
    if [ $SECONDS -gt $deadline ]; then
      echo "FATAL: $url never came up" >&2; tail -40 "$logfile" >&2; exit 1
    fi
    sleep 5
  done
  log "$url is serving"
}

up() {
  local mode=warm engine
  if [ "$1" = cold ]; then mode=cold; shift; fi
  engine="$1"
  down

  mkdir -p logs
  local logfile="logs/${engine}-${mode}.log"
  local py; py=$(py_for "$engine")

  if [ "$engine" = vllm ]; then
    local flags=(--model "$MODEL" --port "$VLLM_PORT"
                 --gpu-memory-utilization "$GPU_MEM_FRAC"
                 --max-model-len "$MAX_MODEL_LEN")
    # Cold disables the cache at the flag level - belt and suspenders on top of
    # the workload already sharing nothing (SPEC section 5). VERIFY this flag
    # against --help on the pinned version; if it has been renamed, the Cold
    # block silently runs WITH caching and H2 becomes meaningless.
    [ "$mode" = cold ] && flags+=(--no-enable-prefix-caching)
    HF_TOKEN="$HF_TOKEN" nohup "$py" -m vllm.entrypoints.openai.api_server \
      "${flags[@]}" > "$logfile" 2>&1 &
  else
    local flags=(--model-path "$MODEL" --host 0.0.0.0 --port "$SGLANG_PORT"
                 --mem-fraction-static "$GPU_MEM_FRAC"
                 --context-length "$MAX_MODEL_LEN")
    [ "$mode" = cold ] && flags+=(--disable-radix-cache)
    HF_TOKEN="$HF_TOKEN" nohup "$py" -m sglang.launch_server \
      "${flags[@]}" > "$logfile" 2>&1 &
  fi

  echo $! > "$PIDFILE"
  wait_healthy "$(base_url "$engine")" "$logfile"
}

p0() {
  # The per-image token count is not knowable offline and everything downstream
  # depends on it, so it is probed against a live server before anything is built.
  up vllm
  log "probing per-image token count"
  python3 scripts/probe_image_tokens.py --base-url "$(base_url vllm)" --model "$MODEL" \
    | tee p0_image_token_probe.txt
  down

  cat <<EOF

Pick a resolution from the table above, then build:
  python3 workloads/build.py --image-tokens <N> --resolution <WxH> \\
      --tokenizer $MODEL --max-rate <highest matrix rate>
  python3 scripts/verify_workloads.py
  python3 scripts/render_readme.py --all
EOF
}

p1() {
  [ -f workloads/manifest.json ] || { echo "FATAL: run p0 and build first" >&2; exit 1; }
  for engine in vllm sglang; do
    log "P1 gates: $engine"
    local url; url=$(base_url "$engine")

    up "$engine"
    python3 scripts/p1_gates.py config-dump --engine "$engine" --base-url "$url"
    python3 scripts/p1_gates.py gate-b --engine "$engine" --base-url "$url" --model "$MODEL"
    python3 scripts/p1_gates.py gate-c --engine "$engine" --base-url "$url" --model "$MODEL"
    down

    # Gate A needs three cache states: one send with the cache off, then two on a
    # freshly started server (cold, then warm) with no restart in between.
    up cold "$engine"
    python3 scripts/p1_gates.py capture --label cache-off --engine "$engine" \
      --base-url "$url" --model "$MODEL"
    down

    up "$engine"
    python3 scripts/p1_gates.py capture --label cold --engine "$engine" \
      --base-url "$url" --model "$MODEL"
    python3 scripts/p1_gates.py capture --label warm --engine "$engine" \
      --base-url "$url" --model "$MODEL"
    down

    python3 scripts/p1_gates.py gate-a --engine "$engine"
  done
  log "P1 complete - gates/ and env/ are ready for the P2 commit"
}

cell() {
  local workload="$1" engine="$2" rate="$3" run_id="$4"
  local dir="workloads/$workload" url; url=$(base_url "$engine")
  local extra=()
  [ "$workload" = single-stream ] && extra=(--concurrency 1)

  curl -s "$url/metrics" > "$dir/metrics/${run_id}_pre.txt"
  python3 scripts/client.py --jsonl "$dir/requests.jsonl" --base-url "$url" \
    --model "$MODEL" --run-id "$run_id" --engine "$engine" --workload "$workload" \
    --rate "$rate" --out "$dir/results" "${extra[@]}"
  curl -s "$url/metrics" > "$dir/metrics/${run_id}_post.txt"

  # Re-render immediately: a broken render found at 6:00 is a lost afternoon.
  python3 scripts/render_readme.py "$workload"
}

case "${1:-}" in
  bootstrap) bootstrap ;;
  up)        shift; up "$@" ;;
  down)      down ;;
  p0)        p0 ;;
  p1)        p1 ;;
  cell)      shift; cell "$@" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
