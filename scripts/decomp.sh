#!/usr/bin/env bash
# Prefill-vs-queue decomposition of SGLang's TTFT disadvantage.
#
# TTFT = queue_wait + prefill_compute (+ first-token). At concurrency=1 there is
# no queue (one request in flight at a time), so TTFT(conc=1) ~= prefill compute.
# The loaded clean-rate TTFT (already collected, scripts/rerun_clean.sh) is
# queue + prefill. So:
#     prefill_compute ~= TTFT(conc=1)
#     queue_wait      ~= TTFT(loaded) - TTFT(conc=1)
# Comparing the two components between engines says whether SGLang's gap is in
# the prefill path itself (compute) or in getting admitted (scheduling/queue).
#
# Cold at conc=1 is the purest prefill-compute probe: no queue, no cache, every
# request does a full ~991-token multimodal prefill alone. Full-reuse at conc=1
# is the cached-prefill path (request 1 fills the cache, 2..N hit it).
#
# Also scrapes server-side /metrics (vLLM exposes request_queue_time_seconds and
# request_prefill_time_seconds histograms; SGLang's TTFT histogram) as an
# independent cross-check on the client-side conc=1 decomposition.
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

REPS=3
mkdir -p decomp

for workload in full-reuse cold; do
  img=$("$CLIENT_PY" -c 'import json;print(json.load(open("workloads/manifest.json"))["geometry"]["image_tokens"])')
  for engine in vllm sglang; do
    log "decomp conc=1: $workload / $engine"
    if [ "$workload" = cold ]; then up cold "$engine"; else up "$engine"; fi
    url=$(base_url "$engine")
    curl -sf --max-time 5 "$url/v1/models" >/dev/null 2>&1 || { echo "WARN: $engine not healthy" >&2; continue; }
    for rep in $(seq 1 "$REPS"); do
      run_id="${workload}_${engine}_conc1_rep${rep}"
      curl -s "$url/metrics" > "decomp/${run_id}_metrics_pre.txt"
      "$CLIENT_PY" scripts/client.py --jsonl "workloads/$workload/requests.jsonl" \
        --base-url "$url" --model "$MODEL" --run-id "$run_id" \
        --engine "$engine" --workload "$workload" --rate 0 --concurrency 1 \
        --num-requests 60 --min-seconds 0 --warmup 10 \
        --image-tokens "$img" --out "decomp" \
        || echo "WARN: $run_id failed" >&2
      curl -s "$url/metrics" > "decomp/${run_id}_metrics_post.txt"
    done
  done
done

down
log "decomp complete. conc=1 TTFT ~= prefill compute; compare to loaded clean-rate TTFT."
"$CLIENT_PY" - <<'PY'
import json, glob, statistics
def p50(pat):
    v=[]
    for f in sorted(glob.glob(pat)):
        s=json.load(open(f))["summary"]
        v.append(s["ttft_s"]["p50"]*1000)
    return v
print("\n== concurrency=1 TTFT (no queue ~= prefill compute), p50 ms ==")
for w in ("full-reuse","cold"):
    for e in ("vllm","sglang"):
        v=p50(f"decomp/{w}_{e}_conc1_rep*.json")
        if v: print(f"  {w:12s} {e:6s}  {round(statistics.mean(v))}ms  reps={[round(x) for x in v]}")
PY
