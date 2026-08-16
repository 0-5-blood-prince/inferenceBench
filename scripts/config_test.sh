#!/usr/bin/env bash
# Is SGLang's ~2x prefill-TTFT gap a genuine engine difference or a default-
# config one? vLLM defaults to torch.compile/inductor (enforce_eager=False,
# VLLM_COMPILE); SGLang defaults to eager (enable_torch_compile=False) with
# TF32 matmul off. Test whether flipping SGLang's compute-relevant defaults
# closes the concurrency=1 (no-queue = pure prefill) gap.
#
# Baselines already measured (concurrency=1, p50 TTFT):
#   full-reuse: vLLM 132 ms,  SGLang stock 249 ms  (+88%)
# This script re-measures SGLang at conc=1 under each config variant on the
# SAME full-reuse and cold workloads, so the delta is attributable to the flag.
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

# variant tag -> extra sglang flags
run_variant() {
  local tag="$1" flags="$2" workload="$3"
  log "config-test [$tag]: $workload / sglang ($flags)"
  if [ "$workload" = cold ]; then SGLANG_EXTRA_FLAGS="$flags" up cold sglang
  else SGLANG_EXTRA_FLAGS="$flags" up sglang; fi
  local url; url=$(base_url sglang)
  curl -sf --max-time 5 "$url/v1/models" >/dev/null 2>&1 || { echo "WARN [$tag] $workload not healthy - flag may be unsupported" >&2; return 0; }
  # record whether torch.compile actually engaged
  grep -iE "torch.compile|compiling|inductor|Capture" "$(cat "$WORKSPACE/.engine_log")" 2>/dev/null | tail -2 > "decomp/cfg_${tag}_${workload}_startup.txt" || true
  for rep in $(seq 1 "$REPS"); do
    "$CLIENT_PY" scripts/client.py --jsonl "workloads/$workload/requests.jsonl" \
      --base-url "$url" --model "$MODEL" --run-id "${workload}_sglang_${tag}_conc1_rep${rep}" \
      --engine sglang --workload "$workload" --rate 0 --concurrency 1 \
      --num-requests 60 --min-seconds 0 --warmup 10 --image-tokens "$img" --out decomp \
      || echo "WARN ${tag} ${workload} rep${rep} failed" >&2
  done
}

# Sweep SGLang's compute-relevant levers vs vLLM's compiled default (132ms) and
# SGLang stock (244ms) at conc=1. TF32 dropped (red herring: bf16 model).
# Variants:
#   compile     torch.compile/Inductor (vLLM's on-by-default lever; SGLang eager)
#   flashinfer  swap Triton -> FlashInfer attention (SGLang docs favor it for speed;
#               NB Gemma-4 multimodal image attention may need Triton for
#               correctness - a wrong-but-timed run still shows the kernel's speed)
#   fa3         FlashAttention-3 backend (may be unsupported on A100/SM80)
#   compilefi   compile + flashinfer together
# full-reuse gets the full sweep (primary comparison); cold gets compile only
# (purest uncached prefill, to see if compile helps the compute-heavy path).
for v in "compile:--enable-torch-compile" \
         "flashinfer:--attention-backend flashinfer" \
         "fa3:--attention-backend fa3" \
         "compilefi:--enable-torch-compile --attention-backend flashinfer"; do
  run_variant "${v%%:*}" "${v#*:}" full-reuse
done
run_variant compile "--enable-torch-compile" cold

down
log "config-test complete. Compare against stock conc=1 (full-reuse sglang 249ms):"
"$CLIENT_PY" - <<'PY'
import json, glob, statistics
def p50(pat):
    return [json.load(open(f))["summary"]["ttft_s"]["p50"]*1000 for f in sorted(glob.glob(pat))]
for w in ("full-reuse","cold"):
    stock=p50(f"decomp/{w}_sglang_conc1_rep*.json")
    vllm=p50(f"decomp/{w}_vllm_conc1_rep*.json")
    print(f"\n{w}: vLLM={round(statistics.mean(vllm)) if vllm else '?'}ms  SGLang stock={round(statistics.mean(stock)) if stock else '?'}ms")
    for tag in ("compile","flashinfer","fa3","compilefi"):
        v=p50(f"decomp/{w}_sglang_{tag}_conc1_rep*.json")
        if v: print(f"  SGLang [{tag}] = {round(statistics.mean(v))}ms  {[round(x) for x in v]}")
PY
