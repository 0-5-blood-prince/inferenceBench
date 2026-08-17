#!/usr/bin/env bash
# THE untested lever: does Inductor-COMPILED prefill close SGLang's gap?
#
# Prior pfgraph/pfgraphc tests used --cuda-graph-backend-prefill tc_piecewise
# with the DEFAULT tc_compiler=eager -> they captured a CUDA graph of the EAGER
# prefill, NOT an Inductor-compiled one. That is graph-capture-of-eager, not the
# vLLM-equivalent (Inductor fusion + graph). SGLang's actual analog is:
#     --cuda-graph-backend-prefill tc_piecewise --cuda-graph-tc-compiler inductor
# which drives FX/Inductor through every prefill shape (tc_piecewise_cuda_graph_
# backend.py: install_torch_compiled with compiler='inductor').
#
# Baselines (conc=1 p50, full-reuse): vLLM 132ms, SGLang stock 244ms,
#   pfgraph (graph-of-eager) 256ms, pfgraphc 253ms. If Inductor-compiled prefill
#   lands near vLLM, the gap WAS a fusion/config difference. If it stays ~244,
#   the eager-vs-fused framing is wrong and it is the Triton kernel impl itself.
set -euo pipefail
cd /workspace/inferenceBench
set -a; . ./.env; set +a
export MODEL=google/gemma-4-31B-it
export WORKSPACE=/workspace
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_HOME=/workspace/hf-cache
CLIENT_PY=/root/venvs/client/bin/python
# shellcheck disable=SC1091
source ./run.sh
trap down EXIT
log() { printf '\n=== %s\n' "$*" >&2; }
mkdir -p decomp
img=$("$CLIENT_PY" -c 'import json;print(json.load(open("workloads/manifest.json"))["geometry"]["image_tokens"])')
REPS=2

run_variant() {
  local tag="$1" flags="$2" workload="$3"
  log "inductor-prefill [$tag]: $workload / sglang ($flags)"
  if [ "$workload" = cold ]; then SGLANG_EXTRA_FLAGS="$flags" up cold sglang
  else SGLANG_EXTRA_FLAGS="$flags" up sglang; fi
  local url; url=$(base_url sglang)
  curl -sf --max-time 5 "$url/v1/models" >/dev/null 2>&1 || { echo "WARN [$tag] $workload not healthy" >&2; return 0; }
  # Confirm Inductor actually engaged (not silently eager / fell back)
  grep -iE "inductor|tc_compiler|piecewise|compile pass|Capture target prefill" \
    "$(cat "$WORKSPACE/.engine_log")" 2>/dev/null | tail -6 > "decomp/ind_${tag}_${workload}_startup.txt" || true
  for rep in $(seq 1 "$REPS"); do
    "$CLIENT_PY" scripts/client.py --jsonl "workloads/$workload/requests.jsonl" \
      --base-url "$url" --model "$MODEL" --run-id "${workload}_sglang_${tag}_conc1_rep${rep}" \
      --engine sglang --workload "$workload" --rate 0 --concurrency 1 \
      --num-requests 60 --min-seconds 0 --warmup 10 --image-tokens "$img" --out decomp \
      || echo "WARN ${tag} ${workload} rep${rep} failed" >&2
  done
}

# The real Inductor-compiled prefill (the untested config)
run_variant indprefill "--cuda-graph-backend-prefill tc_piecewise --cuda-graph-tc-compiler inductor" full-reuse
run_variant indprefill "--cuda-graph-backend-prefill tc_piecewise --cuda-graph-tc-compiler inductor" cold

down
log "inductor-prefill test complete."
"$CLIENT_PY" - <<'PY'
import json, glob, statistics
def p50(pat): return [json.load(open(f))["summary"]["ttft_s"]["p50"]*1000 for f in sorted(glob.glob(pat))]
base={"full-reuse":(132,244),"cold":(485,724)}
for w in ("full-reuse","cold"):
    v,s=base[w]
    x=p50(f"decomp/{w}_sglang_indprefill_conc1_rep*.json")
    if x:
        m=round(statistics.mean(x))
        print(f"{w}: vLLM {v}  SGLang stock {s}  Inductor-prefill {m} {[round(y) for y in x]}  (vs stock {round(100*(m/s-1)):+d}%, vs vLLM {round(100*(m/v-1)):+d}%)")
    else:
        print(f"{w}: no inductor-prefill data (may have failed/OOM)")
PY
