# Why SGLang's sliding-window Triton kernel is ~8× slower

The kernel benchmark ([bench/RESULTS.md](bench/RESULTS.md)) showed SGLang's
`extend_attention_fwd` is ~1.3× *faster* than vLLM's `unified_attention` on a
full-causal layer but **~8× slower on a sliding-window(1024) layer**, and since
Gemma-4 is 50 sliding + 10 full layers, that one kernel drives essentially the
whole prefill gap. This note is the source-level *why*, read from
[sglang/extend_attention.py](sglang/extend_attention.py) (SGLang 0.5.16).

## What the kernel does for sliding-window

The extend (self-attention) stage loops over key tiles:

```python
# extend_attention.py, stage-2 self-attention loop
extend_end = (cur_seq_len_extend if not causal
              else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M))  # ~L511
for start_n in range(0, extend_end, BLOCK_N):                                    # ~L516
    final_mask = mask_m[:, None] & mask_n[None, :]
    ... causal mask ...
    if SLIDING_WINDOW_SIZE > 0:                                                  # ~L546
        window_mask = (cur_block_m*BLOCK_M + offs_m[:,None]) <= \
                      (start_n + offs_n[None,:] + SLIDING_WINDOW_SIZE)
        final_mask &= window_mask                                               # apply as MASK
    if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:                              # ~L554
        SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
    if not SKIP_TILE:
        ... load K/V, qk = tl.dot(q,k), softmax, tl.dot(p,v) ...
```

Two design choices make this slow:

1. **The window is a mask, not a loop bound.** The loop upper bound `extend_end`
   is truncated for *causality* — but the sliding window's *lower* edge is never
   used to move the loop start. Keys older than `query − window` are visited tile
   by tile and then masked/skipped, instead of never being iterated. For
   sequences longer than the window this is wasted iteration.

2. **A per-tile data-dependent `SKIP_TILE` branch (the dominant cost).** Whenever
   `SLIDING_WINDOW_SIZE > 0`, *every* tile computes the window mask and a
   cross-tile reduction `tl.max(tl.max(final_mask))` to a runtime scalar, then
   branches on it. That data-dependent `if not SKIP_TILE:` breaks Triton's
   software-pipelining of the K/V loads against the matmuls — the exact trick that
   makes flash-style attention fast — forcing each iteration to serialize
   load → reduce → branch → dot instead of overlapping across iterations.

## The smoking gun

At **S = 991 ≤ window 1024 the window masks nothing** (every key is within 1024 of
every query), so no tile is ever skipped — yet SGLang's SWA path is **17× slower
than its own full-causal path** (5.39 ms vs 0.32 ms). With nothing masked, the
wasted-iteration effect (#1) is zero, so the slowdown is almost entirely #2: the
per-tile mask + `tl.max` reduction + data-dependent branch that the full-causal
path skips at compile time (there, `SKIP_TILE` stays a `False` literal, the inner
loop is straight-line and fully pipelined).

## How vLLM avoids it

vLLM's `unified_attention` computes window-aware loop bounds up front
(`compute_tile_loop_bounds`), so out-of-window tiles are never visited and the
inner loop stays straight-line and pipelined — which is why its sliding-window
path is actually *cheaper* than its full path.

## Is it known / fixable?

Yes — and it's narrow, not fundamental:
- SWA support for the Triton backend was added recently as a **"good first issue"**
  ([sgl #6161](https://github.com/sgl-project/sglang/issues/6161), May 2025:
  "we need sliding window feature to run gemma 3 model, but triton backend doesn't
  support it") — i.e. a correctness-first implementation, not perf-tuned.
- There is active Gemma-4 Triton `extend_attention` tuning work and a
  [Roadmap] Gemma4 tracking issue ([sgl #26596](https://github.com/sgl-project/sglang/issues/26596)),
  so the SGLang team is aware the Gemma-4 Triton prefill needs optimization.

**Bottom line:** it is not "Triton is slow on A100" (SGLang's causal Triton beats
vLLM's) and not the `custom_mask` path (the bench ran `custom_mask=None`). It is a
specific, fixable inefficiency: SGLang's SWA handling adds a per-tile masking +
`SKIP_TILE` reduction + data-dependent branch that defeats kernel pipelining, and
never narrows the loop to the window.

## Confirmed by patch — the diagnosis is exactly right

A minimal patch making *only* the two edits above (`bench/patch_sglang_swa.py`,
`bench/extend_attention_swa.patch`, both gated on `SLIDING_WINDOW_SIZE>0`) —
(a) narrow the stage-2 loop lower bound to the window, (b) drop the per-tile
`SKIP_TILE` reduction+branch — recovers essentially all of it:

| S (SWA 1024) | stock ms | patched ms | speedup |
|---:|---:|---:|---:|
| 512 | 1.55 | 0.17 | 9.2× |
| 991 | 5.39 | 0.50 | **10.9×** |
| 2048 | 15.26 | 1.24 | 12.3× |

**Output is bitwise identical to stock** (max-abs-diff = 0.000000; the patch only
skips tiles that contribute exactly zero, so it reorders nothing) and matches the
fp32-SDPA oracle to 0.0086 (< 1e-2). The full-causal path is untouched (1.00×).
Patched SGLang SWA (0.50 ms @ S=991) is now **faster than vLLM (0.65) and bf16
SDPA (0.52)** — from ~33× off the FlashInfer floor down to ~3×.

### Reference floor (SWA, S=991, TFLOP/s)
FlashInfer **197** (SGLang's own recommended A100 backend) > SDPA 62 > vLLM 50 >
SGLang stock **6**. SGLang stock is ~33× off the floor; vLLM ~4×; the patch closes
the SGLang gap to ~3×. Confirms both that the deficit is a fixable implementation
detail and that Triton-on-A100 (the pinned backend) already sacrifices ~4× vs
FlashInfer even for vLLM. (FlashAttention-2 unavailable — only a broken FA4 beta is
installed; fp32-SDPA is the correctness oracle.)
