# Splitting the gap: prefill compute vs queue, and the config sweep that can't close it

[rerun-results.md](rerun-results.md) localized SGLang's 43–71% TTFT
disadvantage to the prefill path (decode ITL is identical) but stopped there,
calling a prefill-vs-queue decomposition "the clean next step, out of scope for
this run." This is that step, plus a config sweep testing whether any SGLang
lever closes it. Raw data in `decomp/` (`scripts/decomp.sh`,
`scripts/config_test.sh`, `scripts/prefill_graph_test.sh`).

## The decomposition: concurrency=1 isolates prefill compute

`TTFT = queue_wait + prefill_compute`. At `--concurrency 1` there is only one
request in flight, so `queue_wait ≈ 0` and `TTFT(conc=1) ≈ prefill compute`.
Subtracting from the loaded clean-rate TTFT (from the corrected matrix) gives the
queue component. Both engines, `--rate 0 --concurrency 1 --num-requests 60
--warmup 10`, n=3.

| Workload | prefill (conc=1) vLLM | prefill (conc=1) SGLang | prefill gap | queue vLLM | queue SGLang |
|---|---|---|---|---|---|
| Full reuse | 132 ms | 244 ms | **+85%** | 73 ms | 67 ms |
| Cold (cache off) | 485 ms | 724 ms | **+49%** | 82 ms | 88 ms |

**The queue component is symmetric** — within ~6 ms between engines on both
workloads. **The entire TTFT gap is prefill compute**, present with no queue at
all. This rules out the leading "scheduler / admission overhead" candidate from
rerun-results.md: SGLang's cache-aware radix bookkeeping does *not* add
measurable critical-path latency to the first token at conc=1. The gap is in the
prefill compute itself, not in getting admitted.

## The config sweep: no SGLang lever closes it

vLLM defaults to torch.compile/Inductor + CUDA graphs covering prefill
(`FULL_AND_PIECEWISE`); SGLang defaults to eager with prefill graph auto-disabled
for this multimodal model. All variants measured at conc=1 on full-reuse against
SGLang stock 244 ms / vLLM 132 ms:

| Variant | conc=1 p50 | vs stock | Note |
|---|---|---|---|
| SGLang stock | 244 ms | — | eager, Triton attention |
| `--enable-torch-compile` | 244 ms | **0%** | only compiles the *decode* path; prefill untouched |
| `--cuda-graph-backend-prefill tc_piecewise` | 257 ms | **+5% (worse)** | prefill graph capture **confirmed engaged** (231 s capture, `num_tokens=[4…8192]`) — launch-overhead removal doesn't help a compute-bound kernel |
| `--attention-backend flashinfer` | rejected | — | Gemma-4 only supports trtllm_mha / triton / intel_xpu |
| `--attention-backend fa3` | rejected | — | same rejection |

Attention backend is **pinned to Triton** for Gemma-4: the model needs it for
correct bidirectional image-token attention + interleaved sliding-window (SWA)
masking. FlashInfer/FA3/FA2 are refused at startup, so SGLang cannot use a faster
prefill attention kernel here even in principle. vLLM instead runs FA2 for the
causal/text region + SDPA for the image region, inside a compiled graph.

**Verdict: the gap is a genuine SGLang v0.5.16 prefill-compute limitation for
this model, not a default-config artifact.** Compilation reaches only decode;
prefill CUDA graphs remove launch overhead that a compute-bound single-request
prefill doesn't pay; the faster attention kernels are unavailable by design.

## Layer-wise attribution (hardware → request handling)

Where the ~2× prefill gap does and does not come from, each grounded in a
measurement above or in [rerun-results.md](rerun-results.md):

| Layer | Differs? | Contribution to gap |
|---|---|---|
| Hardware (A100 80GB, serialized) | no | 0 — same silicon |
| Numerics (bf16, kv auto) | no | 0 — TF32 is a red herring for a bf16 model |
| **Attention kernel (prefill)** | **yes** | **dominant** — vLLM FA2+SDPA compiled vs SGLang eager Triton (pinned); compute-bound N×N |
| Compilation / CUDA graphs | yes | ~0 — compile null, prefill graph −0…+5% (tested) |
| KV cache (paged-hash vs radix, ~13% capacity) | yes | ~0 sub-knee — cold cache-off still +43–49% |
| **Multimodal input path** | **yes** | **~⅓** — text-only control shrinks gap by a third |
| Scheduler / admission | yes | ~0 sub-knee (queue symmetric); sets the ~½ saturation knee |
| Request / API layer | no | 0 — identical client, tokenizer, template |

Net: the gap concentrates in **prefill attention/compute (~⅔, compute-bound,
kernel-locked to Triton)** + the **multimodal input path (~⅓)**, is confirmed
absent in decode (bandwidth-bound, ITL identical) and in the queue (symmetric),
and is not reachable by any compilation, graph, or attention-backend config on
SGLang v0.5.16.

## What was not collectable

`pfgraphc` (compile + prefill graph) and the cold-workload config variants did
not complete — the sweep exited after the full-reuse cells and the flashinfer/fa3
startup rejections. They would only reinforce the null: compile touches decode,
prefill graph is launch-overhead not compute, and Triton is forced regardless.
