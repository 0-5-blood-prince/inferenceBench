#!/usr/bin/env bash
# Does forcing PREFILL graph capture close SGLang's concurrency=1 prefill gap?
#
# torch.compile alone only compiled SGLang's DECODE path (prefill graph is
# auto-disabled for this multimodal model), so it barely moved TTFT (-4%).
# --cuda-graph-backend-prefill tc_piecewise ("tc" = torch-compile piecewise)
# CAPTURES a piecewise CUDA graph of the PREFILL path. NOTE: this alone uses the
# DEFAULT tc_compiler=eager, so it graph-captures the EAGER prefill - it is NOT
# the Inductor-fused analog of vLLM's FULL_AND_PIECEWISE. The true analog needs
# --cuda-graph-tc-compiler inductor (see scripts/inductor_prefill_test.sh). The
# graph A/B measured the eager-graph variant at LOADED rate (-1.5%); this measures
# it at concurrency=1 (pure prefill, no queue), alone and combined with
# --enable-torch-compile.
#
# Baselines (conc=1 p50): vLLM 132ms (full-reuse) / 485ms (cold);
#                         SGLang stock 244 / 724.
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
  log "prefill-graph [$tag]: $workload / sglang ($flags)"
  if [ "$workload" = cold ]; then SGLANG_EXTRA_FLAGS="$flags" up cold sglang
  else SGLANG_EXTRA_FLAGS="$flags" up sglang; fi
  local url; url=$(base_url sglang)
  curl -sf --max-time 5 "$url/v1/models" >/dev/null 2>&1 || { echo "WARN [$tag] $workload not healthy" >&2; return 0; }
  # confirm prefill graph actually captured (not silently re-disabled)
  grep -iE "prefill CUDA graph|Capture target prefill|disabling prefill" "$(cat "$WORKSPACE/.engine_log")" 2>/dev/null | tail -2 > "decomp/pfg_${tag}_${workload}_startup.txt" || true
  for rep in $(seq 1 "$REPS"); do
    "$CLIENT_PY" scripts/client.py --jsonl "workloads/$workload/requests.jsonl" \
      --base-url "$url" --model "$MODEL" --run-id "${workload}_sglang_${tag}_conc1_rep${rep}" \
      --engine sglang --workload "$workload" --rate 0 --concurrency 1 \
      --num-requests 60 --min-seconds 0 --warmup 10 --image-tokens "$img" --out decomp \
      || echo "WARN ${tag} ${workload} rep${rep} failed" >&2
  done
}

for workload in full-reuse cold; do
  run_variant pfgraph  "--cuda-graph-backend-prefill tc_piecewise"                        "$workload"
  run_variant pfgraphc "--enable-torch-compile --cuda-graph-backend-prefill tc_piecewise" "$workload"
done

down
log "prefill-graph test complete."
"$CLIENT_PY" - <<'PY'
import json, glob, statistics
def p50(pat): return [json.load(open(f))["summary"]["ttft_s"]["p50"]*1000 for f in sorted(glob.glob(pat))]
for w in ("full-reuse","cold"):
    v=p50(f"decomp/{w}_vllm_conc1_rep*.json"); s=p50(f"decomp/{w}_sglang_conc1_rep*.json")
    print(f"\n{w}: vLLM={round(statistics.mean(v))}ms  SGLang stock={round(statistics.mean(s))}ms")
    for tag in ("compile","pfgraph","pfgraphc"):
        x=p50(f"decomp/{w}_sglang_{tag}_conc1_rep*.json")
        if x: print(f"  SGLang [{tag}] = {round(statistics.mean(x))}ms  {[round(v) for v in x]}")
PY
