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
| `--cuda-graph-backend-prefill tc_piecewise` | 256 ms | **+5% (worse)** | graph capture confirmed (231 s, `num_tokens=[4…8192]`) — but **default `tc_compiler=eager`**: this captures a graph of the *eager* prefill, **not** Inductor-compiled (see below) |
| `--enable-torch-compile --cuda-graph-backend-prefill tc_piecewise` | 253 ms | **+4% (worse)** | still `tc_compiler=eager`; `--enable-torch-compile` only touches decode, so this is again graph-of-eager |
| `--cuda-graph-backend-prefill tc_piecewise --cuda-graph-tc-compiler inductor` | *measuring* | — | the **real vLLM analog** (Inductor-fused + graph); see next section |
| `--attention-backend flashinfer` | rejected | — | Gemma-4 only supports trtllm_mha / triton / intel_xpu |
| `--attention-backend fa3` | rejected | — | same rejection |

The eager-compiler variants on cold (uncached, purest prefill compute) tell the
identical story against SGLang stock 724 ms: prefill graph 721 ms (**−0.4%**),
compile + prefill graph 730 ms (**+1%**). Every *graph-of-eager* lever is null
within noise (±5%) on both workloads; n=2 each.

## The real question: can the eager prefill be Inductor-compiled? (the vLLM analog)

The tests above have a subtle hole, surfaced by asking directly "can eager prefill
be converted into an Inductor-fused + graph-captured prefill like vLLM's?". The
`tc_piecewise` prefill backend takes a `tc_compiler` that is
`Literal["eager", "inductor"]` (`_VALID_COMPILERS`,
[tc_piecewise_cuda_graph_backend.py]) — and its **default is `eager`**. The
`--enable-torch-compile` flag does *not* change it (that flag is decode-only, via
`torch_compile_max_bs`). So every prior "prefill graph" run captured a CUDA graph
of the **eager** prefill; **none of them applied Inductor fusion.** They answer
"does graph-capturing the eager prefill help?" (no), **not** "does an
Inductor-compiled prefill help?".

SGLang *does* expose the real analog: `--cuda-graph-tc-compiler inductor` with the
`tc_piecewise` prefill backend calls `install_torch_compiled(..., compiler=
"inductor")`, driving FX/Inductor through every prefill shape — the direct
equivalent of vLLM's Inductor + `FULL_AND_PIECEWISE`. On CUDA/A100 this is *not*
force-disabled (the force-to-`eager` path is NPU/Ascend-only,
`_handle_npu_backends`). This is the config that was never run; it is being
measured now (`scripts/inductor_prefill_test.sh`) and the result decides the
verdict below.

> **Result (Inductor-compiled prefill, conc=1):** _pending — measuring on the
> pod; will report full-reuse & cold vs vLLM 132/485 and SGLang stock 244/724._

Attention backend is **pinned to Triton for Gemma-4 on *both* engines** — a
correction to an earlier assumption that vLLM used FlashAttention here. Verified
from the actual engine startup logs: vLLM logs
`Using AttentionBackendEnum.TRITON_ATTN backend` (both warm and cold starts), and
SGLang uses its `TritonAttnBackend`. The pin is architectural: Gemma-4 has
`head_dim=256`, hybrid interleaved sliding-window (SWA) with differing SWA head
dims, and bidirectional image-token attention — vLLM's FLASH_ATTN candidate is
skipped (silent fallthrough in the `[FLASH_ATTN, TRITON_ATTN, …]` priority list)
and SGLang hard-asserts the backend into `{trtllm_mha, triton, intel_xpu}` for
the whole Gemma-4 family, **including the text-only `Gemma4ForCausalLM`**
([server_args.py](../../) gate). So neither engine can use a flash-family prefill
kernel on this model, and the difference is *not* a different attention
algorithm.

**Verdict (partial, pending the Inductor result above): the gap lives *within*
the Triton attention family, not FA-vs-Triton.** Both engines run Triton
attention; vLLM splits it out of the graph and fuses everything around it
(RMSNorm, RoPE, Gemma-4's q/k/v-norms, MLP) via Inductor under a
`FULL_AND_PIECEWISE` CUDA graph, while SGLang stock runs prefill **eager**. What
is now firmly established: neither a different attention algorithm (both Triton),
nor caching, nor decode, nor the queue explains it; and *graph-capturing the
eager prefill* does not help. What remains open until the Inductor-prefill run
lands: whether SGLang's eager prefill can be Inductor-compiled to close the gap
(making it a config difference), or whether it stays ~244 ms even Inductor-fused
(making it the Triton **kernel implementation** itself, vLLM's TRITON_ATTN vs
SGLang's, plus fusion that Inductor can't recover). The earlier flat claim that
"forcing SGLang's compile/graph on is null → genuine limitation, not config" was
**overstated**: it only covered the eager-compiler path.

## Layer-wise attribution (hardware → request handling)

Where the ~2× prefill gap does and does not come from, each grounded in a
measurement above or in [rerun-results.md](rerun-results.md):

| Layer | Differs? | Contribution to gap |
|---|---|---|
| Hardware (A100 80GB, serialized) | no | 0 — same silicon |
| Numerics (bf16, kv auto) | no | 0 — TF32 is a red herring for a bf16 model |
| **Prefill stack (attention + fusion)** | **yes** | **dominant** — *both* run Triton attention (verified: vLLM TRITON_ATTN, SGLang TritonAttnBackend); vLLM's is Inductor-fused + graph-captured, SGLang's prefill is eager; compute-bound N×N |
| Compilation / CUDA graphs | yes | ~0 — compile null, prefill graph −0…+5% (tested) |
| KV cache (paged-hash vs radix, ~13% capacity) | yes | ~0 sub-knee — cold cache-off still +43–49% |
| **Multimodal input path** | **yes** | **~⅓** — text-only control shrinks gap by a third |
| Scheduler / admission | yes | ~0 sub-knee (queue symmetric); sets the ~½ saturation knee |
| Request / API layer | no | 0 — identical client, tokenizer, template |

Net: the gap concentrates in the **prefill compute stack (~⅔, compute-bound;
both engines on Triton attention, vLLM fused+graphed vs SGLang eager)** + the
**multimodal input path (~⅓)**, is confirmed absent in decode (bandwidth-bound,
ITL identical) and in the queue (symmetric), and is not reachable by any
compilation, graph, or attention-backend config on SGLang v0.5.16.

## Coverage note

The one variant genuinely uncollectable is a *faster attention kernel*:
FlashInfer/FA3 are refused at startup for Gemma-4, so there is no way to measure
SGLang with a non-Triton prefill kernel on this model — which is exactly the
lever the decomposition implicates. Everything that *can* be flipped
(compilation, prefill CUDA graph, and their combination, on both full-reuse and
cold) was measured and is null within ±5%.
