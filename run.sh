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
#   ./run.sh p3                 the full 30-run core matrix (P3-core-matrix.md)
#   ./run.sh rate <workload> <0.5k|0.8k|1.2k>   look up the P2 pilot rate grid
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
MIN_FREE_GB="${MIN_FREE_GB:-130}"          # volume: ~60 GB weights + cache + headroom
MIN_VENV_GB="${MIN_VENV_GB:-30}"           # local disk: two engine venvs, ~10 GB each

VLLM_PORT=8000
SGLANG_PORT=30000
# Venvs go on LOCAL disk, not the volume. Measured on an A100 pod whose volume
# is network-backed: 800 small file writes took 184 ms locally and 15.6 s on the
# volume - 85x. pip writes tens of thousands of small files, so installing to a
# network volume turns a 10-minute bootstrap into an hour, and every engine
# start pays the same tax again on imports. Weights still live on the volume:
# they are too large for local disk and are large sequential reads, where the
# same measurement showed only a 4.7x gap.
VENV_DIR="${VENV_DIR:-/root/venvs}"
PIDFILE="$WORKSPACE/.engine.pid"

# The client gets its own venv too. It must not share one with an engine (that
# would let client deps perturb the system under test), and it cannot use the
# system interpreter (Debian blocks that under PEP 668). Its tokenizer library
# is pinned to the engines' version at bootstrap, because every workload
# constant - T, C, E_min, E_max - is a token count, and a client that tokenizes
# differently from the servers would silently build the wrong workload.
CLIENT_PY="$VENV_DIR/client/bin/python"
CLIENT_PIP="$VENV_DIR/client/bin/pip"

# CUDA forward compatibility. Both engines pull a torch built for CUDA 13, but
# Runpod had no A100 host with a CUDA 13 driver (>=580) - the best available was
# 570 / CUDA 12.8, and requiring 13.0 returned "no instances available". Rather
# than downgrade the engines (which would risk losing support for the model's
# architecture, the one thing already verified), install NVIDIA's cuda-compat
# package and put its libcuda ahead of the system one. This is the supported
# path on datacenter GPUs, not a hack:
#     apt-get install -y cuda-compat-13-0
# If the directory is absent the variable stays empty and the system driver is
# used, so this is a no-op on a host whose driver is already new enough.
CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/usr/local/cuda-13.0/compat}"
if [ -d "$CUDA_COMPAT_DIR" ]; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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

  # The venv disk is local and small, unlike the volume - guard it separately.
  mkdir -p "$VENV_DIR"
  local venv_free
  venv_free=$(df -BG --output=avail "$VENV_DIR" | tail -1 | tr -dc '0-9')
  echo "$VENV_DIR has ${venv_free} GB free (need >= ${MIN_VENV_GB})"
  if [ "$venv_free" -lt "$MIN_VENV_GB" ]; then
    echo "FATAL: not enough local disk for two engine venvs." >&2
    echo "       Set VENV_DIR to a larger local path - not the network volume," >&2
    echo "       which is ~85x slower on the small writes pip does." >&2
    exit 1
  fi

  # Forward-compat libcuda, if the host driver predates the engines' CUDA build.
  # See the CUDA_COMPAT_DIR note above for why this is needed rather than
  # downgrading the engines.
  if [ ! -d "$CUDA_COMPAT_DIR" ] && command -v apt-get >/dev/null; then
    log "installing cuda-compat-13-0 (host driver is older than the engines' CUDA)"
    apt-get update -qq || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 || \
      echo "WARNING: cuda-compat unavailable; engines may fail on an old driver" >&2
    [ -d "$CUDA_COMPAT_DIR" ] && \
      export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
    # torch.compile / inductor shells out to ninja at engine start. Neither
    # engine declares it, so without this the server dies after loading 60 GB
    # of weights with a bare FileNotFoundError.
    "$pip" install -q ninja
    # The freeze is the audit trail that replaces the image digest.
    "$pip" freeze > "env/${engine}-freeze.txt"
    "$VENV_DIR/$engine/bin/python" -c \
      'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' \
      | tee "env/${engine}-torch.txt"
  done

  log "client venv"
  python3 -m venv "$VENV_DIR/client"
  "$CLIENT_PIP" install -q --upgrade pip
  "$CLIENT_PIP" install -q -r requirements.txt
  # Pin the client's tokenizer to whatever the engines resolved. If they
  # disagree with each other, that is itself a finding worth stopping for.
  local tok
  tok=$(grep -i '^tokenizers==' env/vllm-freeze.txt | head -1)
  if [ -n "$tok" ]; then
    if ! grep -qiF "$tok" env/sglang-freeze.txt; then
      echo "WARNING: engines resolved DIFFERENT tokenizers versions." >&2
      echo "         vllm: $tok" >&2
      grep -i '^tokenizers==' env/sglang-freeze.txt >&2 || true
      echo "         Token counts may differ between engines - investigate." >&2
    fi
    "$CLIENT_PIP" install -q "$tok"
  fi
  "$CLIENT_PY" -c 'import tokenizers,transformers,aiohttp,PIL,numpy;
print("client tokenizers", tokenizers.__version__, "transformers", transformers.__version__)' \
    | tee env/client-versions.txt

  {
    echo "model=$MODEL"
    echo "vllm=$VLLM_VERSION"
    echo "sglang=$SGLANG_VERSION"
    echo "gpu_mem_frac=$GPU_MEM_FRAC"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "locked_clock_mhz=$LOCK_CLOCK"
    echo "venv_dir=$VENV_DIR"
    echo "hf_home=$HF_HOME"
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
    # Cold disables caching at the flag level - belt and suspenders on top of
    # the workload already sharing nothing (SPEC section 5). VERIFY these flags
    # against --help on the pinned version; if renamed, Cold silently runs WITH
    # caching and H2 becomes meaningless.
    #  --no-enable-prefix-caching  kills the KV prefix cache.
    #  --mm-processor-cache-gb 0   kills the multimodal PROCESSOR cache, which
    #    --no-enable-prefix-caching does NOT touch. Verified from the P3 scrapes:
    #    Cold's "unique" images are the same bytes replayed at each rate cell, so
    #    the processor cache (default 4 GiB, keyed by image content hash) stayed
    #    warm across cells and Cold ran at 0%/100%/71% mm-hit at 0.5k/0.8k/1.2k -
    #    "cold" only above the lowest rate by accident of eviction. The two
    #    excluded-as-saturated cells hid it; disabling the cache makes the flag's
    #    name true. See learnings/measurement/fairness-audit.md (D3).
    if [ "$mode" = cold ]; then
      flags+=(--no-enable-prefix-caching --mm-processor-cache-gb 0)
    fi
    # Optional ad-hoc flags for one-off experiments (e.g. the D2 graph A/B).
    # Empty in the normal matrix path, so P3 is unaffected.
    [ -n "${VLLM_EXTRA_FLAGS:-}" ] && flags+=(${VLLM_EXTRA_FLAGS})
    HF_TOKEN="$HF_TOKEN" nohup "$py" -m vllm.entrypoints.openai.api_server \
      "${flags[@]}" > "$logfile" 2>&1 &
  else
    # --enable-metrics is REQUIRED: SGLang serves no Prometheus endpoint without
    # it. Gate B recorded zero counters on the first pass because of this, and
    # SPEC section 6's hit-rate validity gate plus H4a both depend on engine
    # cached-token counters. Without it the whole matrix is unmeasurable.
    local flags=(--model-path "$MODEL" --host 0.0.0.0 --port "$SGLANG_PORT"
                 --mem-fraction-static "$GPU_MEM_FRAC"
                 --context-length "$MAX_MODEL_LEN"
                 --enable-metrics
                 # per-request cached_tokens in the non-streaming usage object;
                 # sglang:cache_hit_rate is a gauge and useless post-run, so this
                 # is the only workable per-request cache signal on this engine.
                 --enable-cache-report)
    # --disable-radix-cache kills the KV prefix cache. SGLANG_VLM_CACHE_SIZE_MB=0
    # kills the multimodal embedding cache (default 100 MB, hash-keyed, ViT-skip
    # on hit) - the SGLang analogue of vLLM's processor cache and the same D3
    # exposure, smaller because the cache is smaller. Set as env, not a flag:
    # v0.5.16 reads the VLM cache size from the environment, not server_args.
    local sglang_env=(HF_TOKEN="$HF_TOKEN")
    if [ "$mode" = cold ]; then
      flags+=(--disable-radix-cache)
      sglang_env+=(SGLANG_VLM_CACHE_SIZE_MB=0)
    fi
    # Optional ad-hoc flags: the D2 graph A/B passes
    # --cuda-graph-backend-prefill tc_piecewise here to force prefill graph
    # capture that v0.5.16 auto-disables for this multimodal model. Empty in the
    # normal matrix path.
    [ -n "${SGLANG_EXTRA_FLAGS:-}" ] && flags+=(${SGLANG_EXTRA_FLAGS})
    env "${sglang_env[@]}" nohup "$py" -m sglang.launch_server \
      "${flags[@]}" > "$logfile" 2>&1 &
  fi

  echo $! > "$PIDFILE"
  # Record which log this running server actually writes to. cell()'s JIT
  # check must read this rather than guess a mode - it does not otherwise
  # know whether the currently-running engine was started warm or cold, and
  # guessing wrong (an earlier version hardcoded "-warm.log") means the check
  # silently finds no file and never runs for the entire Cold workload block,
  # exactly the workload with no preemption counter to cross-check against.
  echo "$logfile" > "$WORKSPACE/.engine_log"
  wait_healthy "$(base_url "$engine")" "$logfile"
}

p0() {
  # The per-image token count is not knowable offline and everything downstream
  # depends on it, so it is probed against a live server before anything is built.
  up vllm
  log "probing per-image token count"
  "$CLIENT_PY" scripts/probe_image_tokens.py --base-url "$(base_url vllm)" --model "$MODEL" \
    | tee p0_image_token_probe.txt
  down

  cat <<EOF

Pick a resolution from the table above, then build:
  "$CLIENT_PY" workloads/build.py --image-tokens <N> --resolution <WxH> \\
      --tokenizer $MODEL --max-rate <highest matrix rate>
  "$CLIENT_PY" scripts/verify_workloads.py
  "$CLIENT_PY" scripts/render_readme.py --all
EOF
}

p1() {
  # Gate A (strict output equivalence) was dropped as a blocker after P1 first
  # ran: both engines fail it at a real generation length, it tests a property
  # no latency measurement can see, and it manufactured a false engine
  # asymmetry from window length alone (LEARNINGS.md sections 1-4). Gate R
  # replaces it: reuse-by-counters is Gate B below; budget pinning is enforced
  # per-request by the client and checked per-run (tags budget_unpinned);
  # request-shape matching is structural (both engines read the same
  # requests.jsonl). The old capture/gate-a sequence is replaced by one
  # gate_a_extended.py call per engine - informational, not blocking, recording
  # the divergence index as a graded diagnostic rather than a boolean.
  [ -f workloads/manifest.json ] || { echo "FATAL: run p0 and build first" >&2; exit 1; }
  mkdir -p gates
  for engine in vllm sglang; do
    log "P1 gates: $engine"
    local url; url=$(base_url "$engine")

    up "$engine"
    "$CLIENT_PY" scripts/p1_gates.py config-dump --engine "$engine" --base-url "$url"
    # No endpoint exposes vLLM's CUDA graph capture plan; SGLang's is on
    # /get_server_info (already in config_dump.json). Grep the startup log for
    # both so engine-vs-engine capture behaviour is in the P2 record either
    # way - a decode comparison across mismatched graph strategies measures
    # the launch strategy, not the engine (LEARNINGS.md section 7).
    mkdir -p "gates/$engine"
    grep -iE "Capturing CUDA graph|Graph capturing finished|cudagraph_capture_sizes|max_cudagraph_capture_size|cudagraph_mode|disabling prefill CUDA graph|Capture target.*CUDA graph" \
      "logs/${engine}-warm.log" > "gates/$engine/cuda_graph_info.txt" 2>/dev/null || true
    "$CLIENT_PY" scripts/p1_gates.py gate-b --engine "$engine" --base-url "$url" --model "$MODEL"
    "$CLIENT_PY" scripts/p1_gates.py gate-c --engine "$engine" --base-url "$url" --model "$MODEL"
    "$CLIENT_PY" scripts/gate_a_extended.py --base-url "$url" --engine "$engine" \
      --model "$MODEL" --max-tokens 512 --concurrency 8 --tokenizer "$MODEL" \
      --out "gates/$engine/gate_a_extended.json" || true
    down
  done
  log "P1 complete - gates/ and env/ are ready for the P2 commit"
}

# P2 rate grid (SPEC section 2 / P2-freeze.md: "rate grid written into run.sh").
# kappa found by scripts/pilot.py on vLLM (the recorded pilot engine), fixed
# TTFT-blowup-aware version - see learnings/measurement/pilot-results.md for the
# saturation mechanism behind each number (preemption thrashing for full-reuse
# and partial-reuse; raw throughput limit, zero preemptions, for cold).
# Single stream needs no grid - concurrency 1, no rate sweep (SPEC section 5).
#   workload        kappa    0.5k    0.8k    1.2k
#   full-reuse       4.082   2.041   3.266   4.899
#   cold             1.260   0.630   1.008   1.512
#   partial-reuse    2.268   1.134   1.814   2.722
rate_for() {
  local workload="$1" point="$2"  # point: 0.5k | 0.8k | 1.2k
  case "$workload-$point" in
    full-reuse-0.5k)    echo 2.041 ;;
    full-reuse-0.8k)    echo 3.266 ;;
    full-reuse-1.2k)    echo 4.899 ;;
    cold-0.5k)          echo 0.630 ;;
    cold-0.8k)          echo 1.008 ;;
    cold-1.2k)          echo 1.512 ;;
    partial-reuse-0.5k) echo 1.134 ;;
    partial-reuse-0.8k) echo 1.814 ;;
    partial-reuse-1.2k) echo 2.722 ;;
    *) echo "FATAL: no rate grid entry for $workload/$point" >&2; exit 1 ;;
  esac
}

cell() {
  local workload="$1" engine="$2" rate="$3" run_id="$4"
  local dir="workloads/$workload" url; url=$(base_url "$engine")
  local extra=()
  [ "$workload" = single-stream ] && extra=(--concurrency 1)
  # Optional sample-size overrides (default to client.py's own defaults when
  # unset, so the P3 matrix path is unchanged). The clean re-run (D4c) sets
  # these to lift Full/Partial reuse past ~1000 completions for a powered p99.
  [ -n "${NUM_REQUESTS:-}" ] && extra+=(--num-requests "$NUM_REQUESTS")
  [ -n "${MIN_SECONDS:-}" ]  && extra+=(--min-seconds "$MIN_SECONDS")

  # Read lazily, not at script-parse time: manifest.json does not exist yet
  # during bootstrap/p0, and this constant is only needed once workloads are
  # built (SPEC section 5: fixed resolution -> fixed image-token count, used
  # for the image/text split - see learnings/measurement/metrics-instrumentation.md).
  local image_tokens
  image_tokens=$("$CLIENT_PY" -c \
    'import json;print(json.load(open("workloads/manifest.json"))["geometry"]["image_tokens"])')

  # GPU clocks cannot be locked from inside this container (see
  # learnings/infrastructure/environment.md), so thermal/clock drift across a
  # ~110+ minute matrix is a live, uncontrolled confound. Cheap insurance:
  # record clock + temperature alongside every metrics scrape so post-hoc
  # analysis can check whether drift correlates with rate or session time
  # rather than only guessing from engine-order counterbalancing.
  local gpu_query="clocks.sm,clocks.mem,temperature.gpu,power.draw"
  nvidia-smi --query-gpu="$gpu_query" --format=csv > "$dir/metrics/${run_id}_gpu_pre.csv"

  # JIT contamination check (see scripts/check_jit_contamination.py): capture
  # the engine log's line count NOW, before this cell's traffic starts, so the
  # check below only scans lines this specific run produced. Read the actual
  # log path from what up() just recorded - do not guess warm-vs-cold here.
  local logfile="" log_start=0
  [ -f "$WORKSPACE/.engine_log" ] && logfile=$(cat "$WORKSPACE/.engine_log")
  if [ -z "$logfile" ] || [ ! -f "$logfile" ]; then
    echo "WARNING: no engine log found (.engine_log missing or stale) - " \
         "$run_id will get ZERO JIT-contamination coverage" >&2
  else
    log_start=$(wc -l < "$logfile")
  fi

  curl -s "$url/metrics" > "$dir/metrics/${run_id}_pre.txt"
  "$CLIENT_PY" scripts/client.py --jsonl "$dir/requests.jsonl" --base-url "$url" \
    --model "$MODEL" --run-id "$run_id" --engine "$engine" --workload "$workload" \
    --rate "$rate" --out "$dir/results" --image-tokens "$image_tokens" "${extra[@]}"
  curl -s "$url/metrics" > "$dir/metrics/${run_id}_post.txt"
  nvidia-smi --query-gpu="$gpu_query" --format=csv > "$dir/metrics/${run_id}_gpu_post.csv"

  if [ -f "$logfile" ]; then
    "$CLIENT_PY" scripts/check_jit_contamination.py --log "$logfile" \
      --since-line "$log_start" --result "$dir/results/${run_id}.json" || true
  fi

  # Re-render immediately: a broken render found at 6:00 is a lost afternoon.
  "$CLIENT_PY" scripts/render_readme.py "$workload"
}

# P3 core matrix (P3-core-matrix.md): 30 runs. Rate order within every block is
# fixed ascending (0.5k -> 0.8k -> 1.2k, SPEC section 6 amendment); engine order
# per block matches the execution-order table there exactly.
# A single cell's failure (client.py error, transient network blip) must not
# take down the other ~29 runs in an unattended ~3h sequence - cell() itself
# calls no exit internally, so a plain || is enough to catch and log it here
# without set -e aborting the whole script. Health-checked first: if the
# engine has died between cells (rare - never observed in this session's 15+
# real starts, but this run is unattended for ~3h), fail fast on a 5s probe
# instead of letting client.py spend a full ~240s timing out against a dead
# server for every remaining point in this leg.
run_cell() {
  local engine="$2" url; url=$(base_url "$engine")
  if ! curl -sf --max-time 5 "$url/v1/models" >/dev/null 2>&1; then
    echo "WARNING: $engine not responding before cell $* - SKIPPING (engine likely died)" >&2
    return 1
  fi
  cell "$@" || echo "WARNING: cell $* FAILED - continuing to the next run" >&2
}

p3_block() {
  local workload="$1" engine point rate dead; shift
  local cold_flag=()
  [ "$workload" = cold ] && cold_flag=(cold)
  for engine in "$@"; do
    up "${cold_flag[@]}" "$engine"
    dead=0
    if [ "$workload" = single-stream ]; then
      run_cell single-stream "$engine" 1 "single-stream_${engine}"
    else
      for point in 0.5k 0.8k 1.2k; do
        [ "$dead" = 1 ] && { echo "WARNING: skipping $workload/$engine/$point - engine dead" >&2; continue; }
        rate=$(rate_for "$workload" "$point")
        run_cell "$workload" "$engine" "$rate" "${workload}_${engine}_${point}" || dead=1
      done
    fi
  done
}

p3() {
  [ -f workloads/manifest.json ] || { echo "FATAL: run p0 and build first" >&2; exit 1; }
  # Engine startup (wait_healthy) calls exit on a genuine fatal failure rather
  # than returning an error code - by design, that case has never fired once
  # in this session's 15+ real invocations, and it is not being loosened right
  # before an unattended ~3h run. What the trap guarantees instead: WHATEVER
  # path this function exits by - success, a cell failure, or that rare exit -
  # down() still runs, so VRAM is always released and the pod is never left
  # holding a dead server. Idempotent (down() is already safe to call when
  # nothing is running), so it firing on the normal successful path too is
  # harmless.
  trap down EXIT

  log "P3: Full reuse"
  p3_block full-reuse vllm sglang
  log "P3: Cold"
  p3_block cold sglang vllm
  log "P3: Partial reuse"
  p3_block partial-reuse vllm sglang
  log "P3: Single stream"
  p3_block single-stream sglang vllm

  # Late-pass variance block (SPEC section 6 amendment): Full reuse @
  # {0.8k,1.2k}, both engines, run AFTER everything above - not back-to-back
  # with the originals - so it captures session drift, not just short-term
  # repeatability. 2 replicates per cell here join the original from the
  # block above for n=3 total per cell.
  log "P3: late-pass variance block"
  local engine point rate rep dead
  for engine in vllm sglang; do
    up "$engine"
    dead=0
    for point in 0.8k 1.2k; do
      rate=$(rate_for full-reuse "$point")
      for rep in 1 2; do
        [ "$dead" = 1 ] && { echo "WARNING: skipping late-pass $engine/$point/rep$rep - engine dead" >&2; continue; }
        run_cell full-reuse "$engine" "$rate" "full-reuse_${engine}_${point}_late${rep}" || dead=1
      done
    done
  done
  log "P3 complete - 30 runs done. Review tags (ok/saturated/jit_contaminated/rows_exhausted/void) before P4."
}

# Only dispatch when executed directly. When sourced (the re-run scripts pull in
# up()/down()/cell()/rate_for() so there is one definition of engine lifecycle,
# not a second copy that drifts - it drifted once already), skip the case so
# sourcing does not fall through to the exit-1 default and kill the caller's
# shell. BASH_SOURCE[0] != $0 means "we were sourced".
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  case "${1:-}" in
    bootstrap) bootstrap ;;
    up)        shift; up "$@" ;;
    down)      down ;;
    p0)        p0 ;;
    p1)        p1 ;;
    p3)        p3 ;;
    cell)      shift; cell "$@" ;;
    rate)      shift; rate_for "$@" ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
  esac
fi
