# Triton prefill-attention kernel benchmark: vLLM vs SGLang @ Gemma-4-31B-it shapes

Head-to-head microbenchmark of the two Triton prefill-attention kernels each
engine actually runs on A100 for `google/gemma-4-31B-it`:

- **vLLM 0.26.0** — `vllm.v1.attention.ops.triton_unified_attention.unified_attention`
- **SGLang 0.5.16** — `sglang.kernels.ops.attention.extend_attention.extend_attention_fwd`

Hardware: **NVIDIA A100 80GB PCIe**. dtype **bfloat16**. batch = 1 (concurrency=1).
Each kernel run in its own venv (deps conflict). Scripts: `bench_vllm.py`,
`bench_sglang.py`.

## Model shape (from the pod's config.json `text_config`)

| field | value |
|---|---|
| head_dim | 256 |
| num_attention_heads (query) | 32 |
| **num_key_value_heads** | **16** (GQA group = 2) |
| sliding_window | 1024 |
| num_hidden_layers | 60 |
| **layer mix** | **50 `sliding_attention` + 10 `full_attention` (5:1)** |

The 5:1 SWA:full layer ratio is the crux of the result: **5/6 of Gemma-4's
attention layers are sliding-window**, so the sliding-window kernel dominates
end-to-end prefill.

## What is benchmarked

A **cold, single-request self-attention prefill**: S tokens attending causally
to those same S tokens, **prefix_len = 0**. This is the cleanest apples-to-apples
unit and matches the "cold full prefill" end-to-end case.

- **vLLM**: `unified_attention` reads K/V from a **paged KV cache**
  (`block_size=16`, layout `[num_blocks, block_size, num_kv_heads, head_dim]`)
  via a block table — faithful to how vLLM runs prefill (K/V written to cache,
  then attention gathers from it). `causal=True`; full = `window_size=(-1,-1)`,
  sliding = `window_size=(1023,0)` → in-kernel `SLIDING_WINDOW=1024`.
- **SGLang**: `extend_attention_fwd` with `qo_indptr=[0,S]`, `kv_indptr=[0,0]`,
  empty `kv_indices` (no prefix → stage-1 prefix loop runs 0 iters, all work is
  the stage-2 causal self-attention), `custom_mask=None`, `is_causal=True`;
  full = `sliding_window_size=-1`, sliding = `sliding_window_size=1024`.

Timing: ≥15 warmup iters, then **median of 60** iters timed with
`torch.cuda.Event`. Only the kernel wrapper call is timed (inputs prebuilt).

## FLOP formula (identical for both engines → ratio is layout-independent)

```
flops = 2 * 2 * batch * num_query_heads * S^2 * head_dim
      =  4 * 1 * 32 * S^2 * 256
```
Two 2's: factor 2 for multiply-add, factor 2 for the two matmuls (QK^T and P@V).
This counts the **full S×S** score matrix (does not discount the causal
triangle), so TFLOP/s is a consistent *dense-equivalent* throughput proxy —
comparable across engines, **not** an MFU claim (real causal work ≈ half).

## Results — full-causal layer

| S | vLLM ms | vLLM TFLOP/s | SGLang ms | SGLang TFLOP/s | ratio (SGLang/vLLM) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.079 | 6.8 | 0.063 | 8.5 | **0.80** |
| 512 | 0.142 | 60.7 | 0.107 | 80.1 | **0.76** |
| 991 | 0.426 | 75.5 | 0.322 | 99.8 | **0.76** |
| 2048 | 1.569 | 87.6 | 1.136 | 121.0 | **0.72** |

→ On plain full-causal attention, **SGLang's extend kernel is actually ~1.3× FASTER
than vLLM's paged `unified_attention`** (ratio < 1). The full-attention kernel does
**not** explain the end-to-end SGLang slowdown. (S=128 is launch-overhead-bound.)

## Results — sliding-window(1024) layer

| S | vLLM ms | vLLM TFLOP/s | SGLang ms | SGLang TFLOP/s | ratio (SGLang/vLLM) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.083 | 6.5 | 0.180 | 3.0 | **2.17** |
| 512 | 0.208 | 41.4 | 1.537 | 5.6 | **7.41** |
| 991 | 0.649 | 49.6 | 5.394 | 6.0 | **8.31** |
| 2048 | 1.873 | 73.4 | 15.226 | 9.0 | **8.13** |

→ On sliding-window attention, **SGLang's extend kernel is ~8× SLOWER than vLLM's**,
and its throughput *collapses* (~6 TFLOP/s vs vLLM ~50).

### Why: the kernels handle the window fundamentally differently
- **vLLM** computes tight tile-loop bounds from the window
  (`compute_tile_loop_bounds` uses `SLIDING_WINDOW` to set `loop_lo/loop_hi`), so
  it **iterates only the tiles inside the window** — SW is *faster* than full.
- **SGLang** does **not** truncate the stage-2 loop range for the window. It
  iterates the full causal extent and merely **masks**, plus runs a per-tile
  `SKIP_TILE = tl.max(tl.max(final_mask,...))` cross-warp reduction and extra
  window-mask arithmetic every iteration. This adds huge overhead with no
  compensating work reduction.
- Smoking gun: at **S=991 ≤ window=1024 the window masks *nothing*** (every causal
  key is in-window), yet SGLang is **17× slower than its own full-causal path**
  (5.39 ms vs 0.32 ms). The slowdown is pure kernel overhead of the windowed code
  path, not extra compute. (In production the `window_kv_indices` truncation only
  shortens the *prefix*/stage-1 loop, which is empty in a cold prefill — confirmed
  at `triton_backend.py:1307-1390`: an SWA layer passes `sliding_window_size=1024`
  with `custom_mask=None` into exactly this path.)

## Tie-back to the end-to-end conc=1 gap

Weighting the S=991 kernel medians by Gemma-4's actual 50 SWA + 10 full layers
(attention-only time for one 991-token cold prefill):

| | vLLM | SGLang |
|---|---:|---:|
| 50 × SWA layer | 32.5 ms | 269.7 ms |
| 10 × full layer | 4.3 ms | 3.2 ms |
| **attention total** | **36.7 ms** | **272.9 ms** |

Predicted attention delta = **236 ms**. Observed end-to-end cold-prefill delta
(724 − 485) = **239 ms**. Assuming the non-attention work (MLP/GEMM/norms) is
~equal (≈448 ms, implied from vLLM), SGLang's predicted total is
448 + 273 = **721 ms vs 724 ms observed** — a near-exact match.

## Verdict

**Yes — the kernel-level ratio matches (indeed fully accounts for) the ~1.85×
end-to-end conc=1 prefill gap, but the cause is specifically SGLang's
sliding-window extend kernel (~8× slower), not its full-attention kernel (which
is ~1.3× faster). Because 5/6 of Gemma-4's layers are sliding-window, that one
kernel weakness drives the entire ~1.85× gap.**

## Caveats on input-layout fidelity
1. **Cold prefill only (prefix_len = 0).** Both kernels run their pure
   self-attention path — the fairest comparison and a faithful match to the cold
   991-tok case. The "full-reuse cached-tail" case (extend≈138, prefix≈853) is a
   different regime (SGLang truncates the prefix to the window via
   `window_kv_indices`) and is not directly measured here; the cold-prefill
   decomposition above is the airtight one.
2. **Different-but-faithful K/V source.** vLLM gathers K/V from a paged cache
   (block_size=16); SGLang reads the new tokens from contiguous `k_extend/v_extend`.
   This asymmetry is intentional — it reflects what each engine actually executes
   in prefill — not a harness artifact. It is a minor factor next to the 8× SWA
   gap (and, if anything, disadvantages vLLM, yet vLLM still wins on SWA).
3. `softcap`/`logit_cap = 0` and `sm_scale = 1/sqrt(256)` for both; these affect
   numerics, not the timing comparison. FLOP counting is dense (causal triangle
   not discounted), identical for both engines.
4. S=128 results are launch-overhead-bound (~0.06–0.18 ms) and should not be
   over-interpreted.
