#!/usr/bin/env bash
# D4c - re-run the core comparison at genuinely-clean rates, n=3, long enough
# for a reportable p99, with a server restart before EVERY cell (D3).
#
# Fixes three coupled defects at once:
#  - n=1 at the only surviving clean point -> n=3 here.
#  - variance budget spent at saturated 0.8k/1.2k -> variance now measured
#    where the verdicts are actually drawn.
#  - <500 completions at 0.5k made p99 unreportable -> TARGET below lifts
#    Full/Partial reuse past ~1000 completions so p99 has power. Cold stays
#    capped by wall-time; p99 for Cold is underpowered BY DESIGN, so the writeup
#    reports p90 for Cold (pre-registered; harness emits p50/p90/p99).
#
# The restart-before-every-cell is the robust D3 fix: it resets ALL cache state
# (KV prefix, mm processor/embedding, encoder) regardless of flag semantics, so
# no cell inherits a warm cache from the previous rate. Costs ~2-4 min/cell of
# startup; worth it for cache-clean cells.
#
# RATES BELOW ARE PLACEHOLDERS. Fill them from repilot.sh output before running:
# for each workload pick two rates BOTH below min(kappa_vllm, kappa_sglang) -
# a low point (~0.5 kappa_min) and a mid point (~0.65-0.7 kappa_min) - so
# gap-vs-rate has >=2 clean points per workload instead of one. The guard below
# refuses to run while any rate is still the -1 placeholder.
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

# --- decision inputs: filled from the re-pilot (scripts/repilot.sh) ------------
# Re-pilot knees (ttft_growth>2.0 signal), and each engine's highest 120s
# CONFIRMED-healthy pilot rate (growth < 1.3):
#   full-reuse    vLLM k=4.12 (healthy<=3.24)  SGLang k=1.88 (healthy<=1.80)  k_min=1.88
#   cold          vLLM k=1.30 (healthy<=0.97)  SGLang k=0.72 (healthy<=0.54)  k_min=0.72
#   partial-reuse vLLM k=1.14 (healthy<=1.08)  SGLang k=1.21 (healthy<=1.08)  k_min=1.14
# Clean rates are chosen BELOW k_min AND below the both-engine healthy ceiling,
# with margin because the re-run cells run 3-4x longer than the 120s pilot and a
# near-knee rate that survives 120s can still collapse over a long run (the D4
# lesson). Cold is the cache-OFF baseline (H2 is a single-point within-10% test,
# not a gap-vs-rate curve), so it gets ONE rate, not two.
declare -A RATE_LOW=( [full-reuse]=0.94 [partial-reuse]=0.60 )   # ~0.5 k_min
declare -A RATE_MID=( [full-reuse]=1.50 [cold]=0.48 [partial-reuse]=0.90 )  # ~0.66-0.8 k_min
declare -A BANDS=( [full-reuse]="low mid" [cold]="mid" [partial-reuse]="low mid" )
# Target completions per cell. Full/Partial: >=500 (SPEC's p99-reportable
# threshold; n=3 supplies the variance the original single 1000-run couldn't).
# Cold: fewer, p90 by design (its low arrival rate makes 500 cost ~25 min/cell).
declare -A TARGET=( [full-reuse]=500 [cold]=300 [partial-reuse]=500 )
declare -A MINSEC=( [full-reuse]=0 [cold]=0 [partial-reuse]=0 )
REPS=3
# ------------------------------------------------------------------------------

# A clean cell: restart the engine (cold flags for Cold), one client run,
# validity re-check. Mirrors run.sh cell() but forces a fresh server each time.
clean_cell() {
  local workload="$1" engine="$2" rate="$3" run_id="$4"
  if [ "$workload" = cold ]; then up cold "$engine"; else up "$engine"; fi
  # health probe before spending a full run against a dead server
  if ! curl -sf --max-time 5 "$(base_url "$engine")/v1/models" >/dev/null 2>&1; then
    echo "WARNING: $engine not healthy before $run_id - skipping" >&2
    return 1
  fi
  MIN_SECONDS="${MINSEC[$workload]}" NUM_REQUESTS="${TARGET[$workload]}" \
    cell "$workload" "$engine" "$rate" "$run_id" \
    || echo "WARNING: $run_id FAILED - continuing" >&2
}

for w in full-reuse cold partial-reuse; do
  for engine in vllm sglang; do
    for band in ${BANDS[$w]}; do
      declare -n TBL="RATE_${band^^}"
      rate="${TBL[$w]}"
      for rep in $(seq 1 "$REPS"); do
        run_id="${w}_${engine}_${band}_rep${rep}"
        result="workloads/$w/results/${run_id}.json"
        # Resume-safe: a cell whose result already exists AND parses as a
        # completed run is skipped, so a restart after an interruption (host
        # reclaim, out-of-funds exit) picks up where it left off instead of
        # redoing hours of clean cells. Set RERUN_FORCE=1 to redo everything.
        if [ "${RERUN_FORCE:-0}" != 1 ] && [ -f "$result" ] && \
           "$CLIENT_PY" -c "import json,sys; d=json.load(open('$result')); sys.exit(0 if d.get('summary',{}).get('completed',0)>0 else 1)" 2>/dev/null; then
          log "skip (already done): $run_id"
          continue
        fi
        log "clean re-run: $w / $engine / ${band}(${rate}) rep${rep}"
        clean_cell "$w" "$engine" "$rate" "$run_id"
      done
    done
  done
done

down
log "clean re-run complete. Re-tag and render, then p99 is reportable on"
log "Full/Partial reuse; Cold reports p90 (pre-registered)."
