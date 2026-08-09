#!/usr/bin/env bash
# Text-only control matrix (P4 "Text-only control" - promoted from P5 #1).
# 2 workloads x 2 engines x n=3 replicates at each workload's own 0.5k rate,
# matched 1:1 against the full-reuse/partial-reuse multimodal twins so the
# gap (or its absence) isolates the multimodal prefill path.
set -euo pipefail
cd /workspace/inferenceBench
set -a; . ./.env; set +a
export MODEL=google/gemma-4-31B-it
export WORKSPACE=/workspace
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_HOME=/workspace/hf-cache
CLIENT_PY=/root/venvs/client/bin/python

# Duplicated (not sourced) from run.sh: sourcing run.sh would also execute
# its trailing dispatch `case` on EEOF, which calls `exit 1` for an
# unrecognized $1 and would kill this script's shell too. Kept in lockstep
# with run.sh's up()/down()/wait_healthy() by inspection, not by import.
VLLM_PORT=8000
SGLANG_PORT=30000
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PIDFILE="$WORKSPACE/.engine.pid"
VENV_DIR=/root/venvs
CUDA_COMPAT_DIR=/usr/local/cuda-13.0/compat

port_for() { [ "$1" = vllm ] && echo "$VLLM_PORT" || echo "$SGLANG_PORT"; }
base_url() { echo "http://localhost:$(port_for "$1")"; }
py_for() { echo "$VENV_DIR/$1/bin/python"; }

wait_vram_free() {
  for _ in $(seq 1 30); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 5
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
  echo "=== $url is serving" >&2
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
    HF_TOKEN="$HF_TOKEN" nohup "$py" -m vllm.entrypoints.openai.api_server \
      "${flags[@]}" > "$logfile" 2>&1 &
  else
    local flags=(--model-path "$MODEL" --host 0.0.0.0 --port "$SGLANG_PORT"
                 --mem-fraction-static "$GPU_MEM_FRAC"
                 --context-length "$MAX_MODEL_LEN"
                 --enable-metrics
                 --enable-cache-report)
    HF_TOKEN="$HF_TOKEN" nohup "$py" -m sglang.launch_server \
      "${flags[@]}" > "$logfile" 2>&1 &
  fi

  echo $! > "$PIDFILE"
  echo "$logfile" > "$WORKSPACE/.engine_log"
  wait_healthy "$(base_url "$engine")" "$logfile"
}

trap down EXIT

run_cell() {
  local workload="$1" engine="$2" rate="$3" run_id="$4"
  local dir="workloads/$workload" url; url=$(base_url "$engine")
  mkdir -p "$dir/results" "$dir/metrics"

  if ! curl -sf --max-time 5 "$url/v1/models" >/dev/null 2>&1; then
    echo "WARNING: $engine not responding before cell $run_id - SKIPPING" >&2
    return 1
  fi

  local gpu_query="clocks.sm,clocks.mem,temperature.gpu,power.draw"
  nvidia-smi --query-gpu="$gpu_query" --format=csv > "$dir/metrics/${run_id}_gpu_pre.csv"

  local logfile="" log_start=0
  [ -f "$WORKSPACE/.engine_log" ] && logfile=$(cat "$WORKSPACE/.engine_log")
  [ -f "$logfile" ] && log_start=$(wc -l < "$logfile")

  curl -s "$url/metrics" > "$dir/metrics/${run_id}_pre.txt"
  "$CLIENT_PY" scripts/client.py --jsonl "$dir/requests.jsonl" --base-url "$url" \
    --model "$MODEL" --run-id "$run_id" --engine "$engine" --workload "$workload" \
    --rate "$rate" --out "$dir/results" --image-tokens 0 \
    || echo "WARNING: cell $run_id FAILED" >&2
  curl -s "$url/metrics" > "$dir/metrics/${run_id}_post.txt"
  nvidia-smi --query-gpu="$gpu_query" --format=csv > "$dir/metrics/${run_id}_gpu_post.csv"

  if [ -f "$logfile" ]; then
    "$CLIENT_PY" scripts/check_jit_contamination.py --log "$logfile" \
      --since-line "$log_start" --result "$dir/results/${run_id}.json" || true
  fi
}

log() { printf '\n=== %s\n' "$*" >&2; }

# full-reuse-text @ 2.041 req/s (full-reuse's 0.5k point); partial-reuse-text
# @ 1.134 req/s (partial-reuse's 0.5k point) - each workload keeps its own
# multimodal twin's rate so the comparison is matched, not re-piloted.
declare -A RATE=( [full-reuse-text]=2.041 [partial-reuse-text]=1.134 )

for workload in full-reuse-text partial-reuse-text; do
  for engine in vllm sglang; do
    log "text-only: $workload / $engine"
    up "$engine"
    for rep in 1 2 3; do
      run_cell "$workload" "$engine" "${RATE[$workload]}" "${workload}_${engine}_0.5k_rep${rep}"
    done
  done
done

log "text-only control matrix complete (12 runs)"
