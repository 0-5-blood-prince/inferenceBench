#!/usr/bin/env bash
# D4b - re-pilot both engines on all three core workloads with the corrected
# knee signal (ttft_growth_ratio > 2.0 primary; scripts/pilot.py). The first
# pilot used completion-ratio + an 8x blowup backstop, which overestimated
# kappa and put 0.8k past-knee for SGLang (Full/Partial reuse) and both engines
# (Cold) - the reason only 0.5k survived the matrix. This finds the TRUE knee
# per engine, judged by the same rule the matrix cells are.
#
# Output: workloads/<workload>/pilot_<engine>.json (overwrites the P2 pilots -
# the old ones are preserved in git history at the P2 freeze commit). Read the
# kappa per engine, take kappa_min = min(vllm, sglang) per workload, and pick
# the re-run rate grid from there (scripts/rerun_clean.sh).
set -euo pipefail
cd /workspace/inferenceBench
set -a; . ./.env; set +a
export MODEL=google/gemma-4-31B-it
export WORKSPACE=/workspace
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_HOME=/workspace/hf-cache
CLIENT_PY=/root/venvs/client/bin/python

# shellcheck disable=SC1091
source ./run.sh          # up()/down()/wait_healthy()/base_url(), no dispatch
trap down EXIT

log() { printf '\n=== %s\n' "$*" >&2; }

# Cold's knee sits ~3-4x lower than Full reuse (10x the prefill work), so give
# it a lower start rate; the pilot geometric-sweeps up from there either way.
declare -A START=( [full-reuse]=1.0 [cold]=0.3 [partial-reuse]=0.6 )

for workload in full-reuse cold partial-reuse; do
  for engine in vllm sglang; do
    log "re-pilot: $workload / $engine"
    # Cold must pilot with the cache genuinely disabled (D3), same as the matrix.
    if [ "$workload" = cold ]; then up cold "$engine"; else up "$engine"; fi
    "$CLIENT_PY" scripts/pilot.py \
      --workload "$workload" --engine "$engine" \
      --base-url "$(base_url "$engine")" --model "$MODEL" \
      --start-rate "${START[$workload]}" --duration 120 --warmup 10 \
      --out "workloads/$workload/pilot_${engine}.json"
  done
done

down
log "re-pilot complete. kappa per engine:"
for workload in full-reuse cold partial-reuse; do
  for engine in vllm sglang; do
    k=$("$CLIENT_PY" -c "import json;print(json.load(open('workloads/$workload/pilot_${engine}.json'))['kappa'])" 2>/dev/null || echo "?")
    echo "  $workload / $engine: kappa=$k"
  done
done
echo
echo "Next: pick kappa_min per workload, set the grid in scripts/rerun_clean.sh,"
echo "confirm every chosen rate is below BOTH engines' kappa, then fire it."
