#!/usr/bin/env bash
# D2 extension - does forcing SGLang's prefill CUDA graph ON move its saturation
# KNEE, not just its sub-knee TTFT? The stock re-pilot (scripts/repilot.sh) runs
# SGLang with prefill graphs auto-disabled (the default for this multimodal
# model) and found SGLang's knee ~45-55% of vLLM's on every workload. If forcing
# prefill graphs raises SGLang's knee toward vLLM's, the graph asymmetry is the
# dominant cause of SGLang's saturation disadvantage; if the knee barely moves,
# it is scheduling/KV, not graphs.
#
# Forced prefill-graph capture is UNVALIDATED for this architecture - the reason
# SGLang auto-disables it - so capture may fault. This script preserves the
# startup log either way: a fault is itself the finding (the asymmetry is
# structural and unavoidable for this model), and pilot.py is only invoked if
# the server actually comes up healthy.
#
# Run AFTER scripts/repilot.sh (stock knees) so the two are directly comparable.
set -euo pipefail
cd /workspace/inferenceBench
set -a; . ./.env; set +a
export MODEL=google/gemma-4-31B-it
export WORKSPACE=/workspace
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_HOME=/workspace/hf-cache
CLIENT_PY=/root/venvs/client/bin/python

# shellcheck disable=SC1091
source ./run.sh          # up()/down()/base_url(), no dispatch
trap down EXIT
log() { printf '\n=== %s\n' "$*" >&2; }

mkdir -p gates/sglang-graph-ab
declare -A START=( [full-reuse]=1.0 [partial-reuse]=0.6 )
GRAPH_FLAG="--cuda-graph-backend-prefill tc_piecewise"

for workload in full-reuse partial-reuse; do
  log "graphs-ON knee pilot: $workload / sglang ($GRAPH_FLAG)"
  SGLANG_EXTRA_FLAGS="$GRAPH_FLAG" up sglang
  # Record how the capture went - the deliverable if it faults.
  startlog="gates/sglang-graph-ab/graphson_${workload}_startup.log"
  cp "$(cat "$WORKSPACE/.engine_log")" "$startlog" 2>/dev/null || true
  grep -iE "cuda graph|prefill|multimodal|incompatible|capture|breakable|disabl" \
    "$startlog" > "gates/sglang-graph-ab/graphson_${workload}_graph_lines.txt" 2>/dev/null || true

  if ! curl -sf --max-time 5 "$(base_url sglang)/v1/models" >/dev/null 2>&1; then
    echo "WARNING: sglang did NOT come up healthy with forced prefill graphs on " \
         "$workload - likely a capture fault. Startup log saved; skipping pilot." >&2
    continue
  fi
  # Confirm the graph actually captured (not silently re-disabled) before trusting the knee.
  if grep -qiE "disabling prefill CUDA graph|incompatible with multimodal" "$startlog"; then
    echo "NOTE: server came up but log still shows prefill graph DISABLED for " \
         "$workload - the force flag did not take; knee below is effectively stock." >&2
  fi
  "$CLIENT_PY" scripts/pilot.py \
    --workload "$workload" --engine sglang \
    --base-url "$(base_url sglang)" --model "$MODEL" \
    --start-rate "${START[$workload]}" --duration 120 --warmup 10 \
    --out "workloads/$workload/pilot_sglang_graphson.json"
done

down
log "graphs-ON knee pilot complete. Compare against stock pilot_sglang.json:"
for workload in full-reuse partial-reuse; do
  stock=$("$CLIENT_PY" -c "import json;print(json.load(open('workloads/$workload/pilot_sglang.json'))['kappa'])" 2>/dev/null || echo "?")
  gon=$("$CLIENT_PY" -c "import json;print(json.load(open('workloads/$workload/pilot_sglang_graphson.json'))['kappa'])" 2>/dev/null || echo "faulted/none")
  echo "  $workload: stock kappa=$stock  graphs-on kappa=$gon"
done
