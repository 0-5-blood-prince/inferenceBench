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
#    reports p95 for Cold (pre-registered here, not chosen after seeing data).
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

# --- decision inputs: fill from repilot.sh, then remove the guard trip --------
declare -A RATE_LOW=( [full-reuse]=-1 [cold]=-1 [partial-reuse]=-1 )
declare -A RATE_MID=( [full-reuse]=-1 [cold]=-1 [partial-reuse]=-1 )
# Target completions per cell. Full/Partial: ~1000 for a powered p99.
# Cold: capped (p95 only, by design).
declare -A TARGET=( [full-reuse]=1000 [cold]=380 [partial-reuse]=1000 )
declare -A MINSEC=( [full-reuse]=0 [cold]=600 [partial-reuse]=0 )
REPS=3
# ------------------------------------------------------------------------------

for w in full-reuse cold partial-reuse; do
  for tbl in "${RATE_LOW[$w]}" "${RATE_MID[$w]}"; do
    if [ "$tbl" = "-1" ]; then
      echo "FATAL: rate grid still has placeholders. Fill RATE_LOW/RATE_MID" >&2
      echo "       from repilot.sh output (both must be < kappa_min)." >&2
      exit 1
    fi
  done
done

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
    for band in low mid; do
      declare -n TBL="RATE_${band^^}"
      rate="${TBL[$w]}"
      for rep in $(seq 1 "$REPS"); do
        log "clean re-run: $w / $engine / ${band}(${rate}) rep${rep}"
        clean_cell "$w" "$engine" "$rate" "${w}_${engine}_${band}_rep${rep}"
      done
    done
  done
done

down
log "clean re-run complete. Re-tag and render, then p99 is reportable on"
log "Full/Partial reuse; Cold reports p95 (pre-registered)."
