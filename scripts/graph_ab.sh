#!/usr/bin/env bash
# D2 - de-conflate engine architecture from a config default. SGLang v0.5.16
# auto-disables prefill CUDA-graph capture for this multimodal model ("Breakable
# CUDA graph is incompatible with multimodal model"), while vLLM graphs prefill.
# That confound is directly testable: the auto-disable is SKIPPED when the
# prefill backend is set explicitly, so run SGLang both ways and measure the
# delta.
#
#   arm A (stock)  : prefill graphs auto-disabled (what the matrix ran)
#   arm B (forced) : --cuda-graph-backend-prefill tc_piecewise
#
# Full reuse + Partial reuse, at the clean rate (fill CLEAN_RATE from the
# re-pilot), n=3 each, restart between arms. Two outcomes, both useful:
#  - arm B runs: delta = the graph contribution; the mechanism de-conflates and
#    the text-only split verdict gets a second, independent cross-check.
#  - arm B faults on capture/replay (the path is unvalidated for this arch -
#    that is WHY the auto-disable exists): document the fault. The asymmetry
#    then reclassifies from "confound the study failed to control" to "engine
#    property a user of this model cannot avoid", which the comparison is
#    allowed to absorb. Capture the server log either way.
set -euo pipefail
cd /workspace/inferenceBench
set -a; . ./.env; set +a
export MODEL=google/gemma-4-31B-it
export WORKSPACE=/workspace
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_HOME=/workspace/hf-cache
CLIENT_PY=/root/venvs/client/bin/python

# shellcheck disable=SC1091
source ./run.sh          # up()/down()/cell()/base_url(), no dispatch
trap down EXIT
log() { printf '\n=== %s\n' "$*" >&2; }

# Fill from repilot.sh: each workload's clean low rate (below BOTH kappas).
declare -A CLEAN_RATE=( [full-reuse]=1.50 [partial-reuse]=0.90 )
REPS=3
mkdir -p gates/sglang-graph-ab

for w in full-reuse partial-reuse; do
  [ "${CLEAN_RATE[$w]}" = "-1" ] && { echo "FATAL: set CLEAN_RATE[$w] from repilot.sh" >&2; exit 1; }
done

arm_done() {
  # An arm's cell for a workload counts as done when all REPS results exist.
  local arm="$1" w="$2" rep
  for rep in $(seq 1 "$REPS"); do
    local r="workloads/$w/results/${w}_sglang_${arm}_rep${rep}.json"
    [ -f "$r" ] && "$CLIENT_PY" -c "import json,sys; sys.exit(0 if json.load(open('$r')).get('summary',{}).get('completed',0)>0 else 1)" 2>/dev/null || return 1
  done
  return 0
}

run_arm() {
  local arm="$1" extra="$2"
  for w in full-reuse partial-reuse; do
    if [ "${RERUN_FORCE:-0}" != 1 ] && arm_done "$arm" "$w"; then
      log "skip (already done): graph A/B [$arm] $w"
      continue
    fi
    log "graph A/B [$arm]: $w / sglang"
    SGLANG_EXTRA_FLAGS="$extra" up sglang
    # Preserve the startup log so the capture outcome (success or fault) is on
    # record - this is the deliverable when arm B faults.
    cp "$(cat "$WORKSPACE/.engine_log")" "gates/sglang-graph-ab/${arm}_${w}_startup.log" 2>/dev/null || true
    grep -iE "cuda graph|prefill|multimodal|incompatible|capture" \
      "gates/sglang-graph-ab/${arm}_${w}_startup.log" \
      > "gates/sglang-graph-ab/${arm}_${w}_graph_lines.txt" 2>/dev/null || true
    if ! curl -sf --max-time 5 "$(base_url sglang)/v1/models" >/dev/null 2>&1; then
      echo "WARNING: sglang [$arm] not healthy on $w - likely a capture fault; " \
           "startup log saved, skipping runs for this cell" >&2
      continue
    fi
    for rep in $(seq 1 "$REPS"); do
      cell "$w" sglang "${CLEAN_RATE[$w]}" "${w}_sglang_${arm}_rep${rep}"
    done
  done
}

run_arm stock ""
run_arm forced "--cuda-graph-backend-prefill tc_piecewise"

down
log "graph A/B complete. Compare ${w}_sglang_stock_* vs _forced_* p50 TTFT;"
log "read gates/sglang-graph-ab/*_graph_lines.txt for the capture outcome."
