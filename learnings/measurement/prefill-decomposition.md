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
| `--cuda-graph-backend-prefill tc_piecewise --cuda-graph-tc-compiler inductor` | 252 ms | **+3% (null)** | the **real vLLM analog** (Inductor genuinely engaged — 414 s compile vs 231 s eager); still null. See next section |
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

> **Result (Inductor-compiled prefill, conc=1):** **null on both workloads.**
> full-reuse 252 ms [257, 246] (vs stock 244 = **+3%**, vs vLLM 132 = +91%);
> cold 747 ms (vs stock 724 = **+3%**, vs vLLM 485 = +54%). Inductor was genuinely
> engaged (startup capture: `tc_compiler='inductor'`, 414 s compile vs 231 s for
> eager), and the ~991-token shape does not fall back to eager (`can_run` returns
> `True` unconditionally; `replay` always calls the compiled `_compiled_fn`; no
> fallback/recompile logged in the measured window). So this is a valid measurement
> of the real Inductor-compiled prefill — and it does **not** close the gap.

### Why Inductor can't help: the attention kernel is walled off by design

The reason is structural, confirmed at the source level. Both engines keep the
attention **kernel** outside compilation:
- SGLang: `self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)`
  (`triton_backend.py:137`) — explicitly excluded from torch.compile/Inductor.
- vLLM: `unified_attention` is in `splitting_ops`, split out of the Inductor graph.

Inductor only ever fuses the *surrounding* ops (RMSNorm, RoPE, q/k/v-norms, MLP);
the attention kernel is untouched in both. So no compilation lever can change it.
And "both use Triton" does **not** mean the same kernel — vLLM's `unified_attention`
(`triton_unified_attention.py`) and SGLang's `extend_attention_fwd`
(`extend_attention.py`) are two independent Triton implementations (copied to
[../kernels/](../kernels/)). The only config that could swap the kernel is the
attention *backend* (flash/flashinfer/fa3/trtllm_mha), and Gemma-4 rejects all
flash-family backends on A100 (trtllm_mha needs Hopper+). A direct kernel-level
benchmark ([../kernels/bench/](../kernels/bench/)) pins down *which* kernel and
*why* — see below.

### Kernel-level confirmation: it is the sliding-window kernel specifically

Microbenchmark of the two kernels head-to-head on the A100 (cold self-attention,
prefix_len=0, bf16, batch=1, median of 60; ratio = SGLang/vLLM latency):

| S | full-causal (SGLang/vLLM) | sliding-window 1024 (SGLang/vLLM) |
|---:|---:|---:|
| 512 | **0.76×** (SGLang faster) | **7.4×** (SGLang slower) |
| 991 | **0.76×** | **8.3×** |
| 2048 | **0.72×** | **8.1×** |

The surprise: SGLang's **full-causal** kernel is ~1.3× *faster* than vLLM's. Its
**sliding-window** kernel is ~8× *slower*. Gemma-4 is **50 sliding + 10 full
attention layers (5:1)**, so the SWA weakness dominates. Weighting the S=991
per-layer medians by the real 50:10 mix gives attention-only totals of vLLM
36.7 ms vs SGLang 272.9 ms — a **236 ms predicted delta vs the 239 ms observed**
cold-prefill delta (724−485). This looked near-exact — **but a subsequent
end-to-end test refuted it (see the correction box below); the match was a
coincidence.**

> ## ⚠️ CORRECTION, then RE-SCOPED: the SWA patch is workload-dependent
> **First correction (multimodal, still stands):** installing the patched kernel
> into the **live SGLang server** (marker-verified in the engine log that the server
> imported it; `__pycache__`/Triton caches purged) and re-running conc=1 on the
> **multimodal** workload changed end-to-end TTFT by **~0%**: cold 721 ms [715,727]
> vs stock 724; full-reuse 242 vs 244 (stock-recheck reproduced ~244, harness sound).
> So on the study's core **multimodal** comparison the SWA patch does not help, and
> the "721≈724" layer-weighting above was coincidental — do **not** cite the SWA
> kernel as the multimodal end-to-end cause; that root cause remains **OPEN**.
>
> **Re-scope (pure text — the blanket refutation was too broad):** the 0% result is
> now explained mechanistically, and the SWA kernel *is* the driver on pure text.
> The patch's dominant fix (dropping the `SKIP_TILE` reduction+branch) is gated
> `if USE_CUSTOM_MASK:` — it only engages when `USE_CUSTOM_MASK` is **False**.
> Gemma-4 multimodal sets `USE_CUSTOM_MASK=True` (bidirectional image-token masking),
> so the fix **never engaged** on the multimodal test — hence 0%. On **pure text**
> (`custom_mask=None`) it fully engages: a torch-profiler trace shows attention is
> ~100% of the pure-text prefill gap (not a small fraction), and installing the patch
> end-to-end against the pure-text workload closes the gap **88% at ilen=991** (the
> study's real prefill length; SGLang 612→374 ms vs vLLM 343) and **98% at 2048**.
> Full breakdown, tables, and the patch-diff gating:
> [../kernels/profiling-resolves-mechanism.md](../kernels/profiling-resolves-mechanism.md).
> The `~8×`-in-isolation, bitwise-identical kernel finding is unchanged.

Root cause, from the source: vLLM truncates its tile loop to the window
(`compute_tile_loop_bounds`), so sliding-window is *cheaper* than full. SGLang's
extend kernel never truncates the stage-2 loop range — it iterates the full causal
extent and only *masks*, plus a per-tile cross-warp `SKIP_TILE` reduction. Smoking
gun: at S=991 ≤ window 1024 the window masks nothing, yet SGLang is **17× slower
than its own full-causal path** (5.39 vs 0.32 ms) — pure loop/masking overhead,
not extra compute. So this is not "Triton is slow on A100" (SGLang's causal Triton
beats vLLM's) and not the `custom_mask` path (this ran `custom_mask=None`): it is a
specific missing loop-bound optimization in SGLang's sliding-window Triton kernel —
a fixable inefficiency, not a hardware or algorithm limit. (Published data that
FA2-CUDA > Triton-FA on A100 is a red herring here; the split is kernel-specific.)

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

**Verdict (final): the gap is the Triton attention *kernel implementation*
itself, not compilation, fusion, caching, decode, or the queue.** Ruled out with
data: different attention *algorithm* (both run Triton), caching (cold cache-off
+43–49%), decode (ITL identical), queue (symmetric ≤6 ms), and — now including the
genuine Inductor-compiled prefill — *every* SGLang compilation lever (compile 0%,
graph-of-eager +3–5%, Inductor-compiled +3%). The attention kernel is walled off
from compilation by design in both engines (`torch.compiler.disable` / `splitting_ops`),
so no compilation config can touch it, and the flash-family backends that would
swap the kernel are architecturally rejected for Gemma-4 on A100. The kernel
benchmark localizes it precisely: **SGLang's sliding-window Triton kernel is ~8×
slower than vLLM's because it does not truncate its tile loop to the window**,
while its full-causal kernel is actually faster; since 50/60 Gemma-4 layers are
sliding-window, that one kernel drives essentially the entire gap (layer-weighted
prediction 721 ms vs 724 ms observed). This IS a genuine SGLang v0.5.16
limitation for this model on A100 — but note it is *narrow and fixable* (a missing
loop-bound optimization in one kernel), not a broad "SGLang/Triton is slow"
result, and not reachable by any SGLang flag short of a backend the model forbids
on this GPU. Earlier framings — "compilation/config", then "the whole Triton
kernel" — were both too broad; it is specifically the SWA kernel's loop bound.

> **Scope update (post-profiling — this verdict was written before the
> multimodal refutation and its re-scoping; read it together with the correction
> box above).** The verdict holds **for pure text**: a torch-profiler trace
> confirms attention is ~100% of the S=991 text-prefill gap and the patch closes it
> **88–98% end-to-end** ([../kernels/profiling-resolves-mechanism.md](../kernels/profiling-resolves-mechanism.md)).
> It does **not** hold for the study's **multimodal** workload — there the same
> patch is null because `USE_CUSTOM_MASK=True` gates the dominant fix off, and the
> multimodal end-to-end driver remains **OPEN**. The "721 vs 724" layer-weighting
> was a multimodal measurement and coincidental; do not read it as the multimodal
> mechanism.

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
