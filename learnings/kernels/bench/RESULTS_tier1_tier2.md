# Reference kernels + patched SGLang SWA kernel @ Gemma-4-31B-it shapes

Extends the vLLM-vs-SGLang head-to-head ([RESULTS.md](RESULTS.md)) with (Tier 1)
reference / gold-standard kernels and (Tier 2) a minimal patch to SGLang's
sliding-window extend kernel that removes its ~8x SWA deficit.

Hardware: **NVIDIA A100 80GB PCIe** (driver 550.127.05). dtype **bfloat16**,
batch = 1. Shapes: head_dim=256, 32 query heads, 16 KV heads (GQA=2),
sliding_window=1024, **cold self-attention prefill (prefix_len=0)**. S in
{512, 991, 2048}.

FLOP formula (identical to RESULTS.md, so TFLOP/s is comparable across all rows):
```
flops = 2 * 2 * batch * num_query_heads * S^2 * head_dim   =  4 * 32 * S^2 * 256
```
Two 2's: multiply-add, and two matmuls (QK^T, P@V). Counts the full S×S score
matrix (causal triangle not discounted) -> a consistent dense-equivalent
throughput proxy, not an MFU claim.

Timing: >=15 warmup, median of 60 iters, `torch.cuda.Event`, only the kernel call
timed (inputs prebuilt). All numbers below (Tier 1, Tier 2, correctness) were
measured in a single session on the free A100. The vLLM / SGLang-stock columns in
the Tier-1 comparison are the published RESULTS.md medians; SGLang-stock was also
re-measured this session in Tier 2 and reproduced (S=991 SWA 5.39 ms).

## Library versions
| lib | version |
|---|---|
| torch | 2.11.0+cu130 (CUDA 13.0) |
| triton | 3.6.0 |
| vllm | 0.26.0 |
| sglang / sglang-kernel | 0.5.16 / 0.4.5 |
| flashinfer-python | 0.6.14 |
| flash-attn | 4.0.0b19 — **skipped**, see below |

### Skipped kernels
- **FlashAttention (flash_attn)** — the only build installed is **4.0.0b19**, the
  CuTe-DSL FA4 beta (`flash_attn.cute.flash_attn_func`). It fails to compile on
  this A100 for **every** configuration tried — including a plain `causal=True`
  call at head_dim 128 and 256 with no window — raising
  `MLIRError('Operation creation failed')` (and a separate DSL type error on the
  `window_size_left` argument). This is a broken beta/toolchain mismatch, not a
  usage error; per guidance we note it and skip rather than spend time
  recompiling. **FlashInfer serves as the CUDA-library performance floor** (it is
  also the backend SGLang recommends on A100). No classic FA2 is installed in
  either venv.
- **FlexAttention** — not pursued; SDPA-with-mask + FlashInfer already bracket the
  baseline and the floor.

---

## TIER 1 — reference kernels vs vLLM & SGLang

### Full-causal layer (median ms / TFLOP/s)
| S | FlashInfer | SDPA (bf16) | vLLM | SGLang stock |
|---:|---:|---:|---:|---:|
| 512  | 0.071 / 121.5 | 0.067 / 128.5 | 0.142 / 60.7 | 0.107 / 80.1 |
| 991  | 0.160 / 200.6 | 0.164 / 196.0 | 0.426 / 75.5 | 0.322 / 99.8 |
| 2048 | 0.524 / 262.4 | 0.485 / 283.3 | 1.569 / 87.6 | 1.136 / 121.0 |

### Sliding-window(1024) layer (median ms / TFLOP/s)
| S | FlashInfer (floor) | SDPA (bf16) | vLLM | SGLang stock | SGLang **patched** |
|---:|---:|---:|---:|---:|---:|
| 512  | 0.120 / 71.9  | 0.160 / 53.7 | 0.208 / 41.4 | 1.537 / 5.6 | **0.169 / 50.9** |
| 991  | 0.163 / 197.3 | 0.517 / 62.2 | 0.649 / 49.6 | 5.394 / 6.0 | **0.496 / 65.0** |
| 2048 | 0.529 / 260.0 | 1.927 / 71.3 | 1.873 / 73.4 | 15.23 / 9.0 | **1.238 / 111.0** |

Notes on window convention: FlashInfer (`window_left=1023`), vLLM
(`window_size=(1023,0)`) and SDPA all use a **1024-key** window; SGLang
(`sliding_window_size=1024`, mask `q<=kv+W`) uses **1025 keys**. The 1-key
difference is irrelevant to timing and, at S<=1024, the window masks nothing at
all (every causal key is in-window).

**How far are vLLM/SGLang from the floor on SWA?** FlashInfer is the floor
(~0.16 ms at S=991). On the SWA shape:
- **vLLM** is ~**4x** off the floor (0.649 vs 0.163 ms @ S=991) — window handled
  as tight loop bounds, so it degrades gracefully.
- **SGLang stock** is ~**33x** off the floor (5.39 vs 0.163 ms @ S=991), and its
  throughput collapses to ~6 TFLOP/s. This is the ~8x-vs-vLLM SWA deficit from
  RESULTS.md, seen here against the absolute floor.

---

## TIER 2 — patching SGLang's SWA kernel

### The patch (two edits, stage-2 loop of `_fwd_kernel`, SWA path only)
Source copied from the installed
`sglang/kernels/ops/attention/extend_attention.py` to
[`extend_attention_patched.py`](extend_attention_patched.py); full diff in
[`extend_attention_swa.patch`](extend_attention_swa.patch); applied
programmatically (auditable string edits) by
[`patch_sglang_swa.py`](patch_sglang_swa.py).

- **(a) Window-aware loop LOWER bound.** Keys older than
  `query - SLIDING_WINDOW_SIZE` are out of window for every row of the M-tile
  (smallest query index = `cur_block_m*BLOCK_M`), so the stage-2 loop now starts
  at `floor_to_BLOCK_N(max(0, cur_block_m*BLOCK_M - SLIDING_WINDOW_SIZE))` instead
  of 0 — the out-of-window tiles are never iterated (analogous to vLLM's
  `compute_tile_loop_bounds`).
- **(b) Remove the per-tile `SKIP_TILE` reduction + data-dependent branch** for
  the pure sliding-window path (keep it only for `USE_CUSTOM_MASK`). The
  `SKIP_TILE = tl.max(tl.max(final_mask))` cross-warp reduction + `if not
  SKIP_TILE:` branch is what broke Triton's software-pipelining. The window is
  still applied correctly via `final_mask` (`tl.where -> -inf`); with the narrowed
  range from (a) no in-range tile is ever fully masked, so removing the skip
  changes no output.

Both edits are gated on `SLIDING_WINDOW_SIZE > 0`, so the **full-causal path is
byte-identical** to stock (confirmed: 1.00x, same TFLOP/s, below).

### Correctness of the patch (S=991, SWA(1024))
| comparison | max abs diff | verdict |
|---|---:|---|
| patched vs **stock** SGLang | **0.000000** | bitwise identical output |
| patched vs fp32-SDPA oracle | 0.008647 | within bf16 tol (< 1e-2) |
| stock vs fp32-SDPA oracle | 0.008647 | (baseline) |

The patch only skips tiles/iterations that contribute exactly zero (fully
out-of-window, masked to `-inf`), so it reorders nothing — output is *identical*
to stock, not merely close.

### Stock vs patched (median ms / TFLOP/s), same session
| mode | S | stock | patched | speedup |
|---|---:|---:|---:|---:|
| SWA(1024)   | 512  | 1.547 / 5.6  | **0.169 / 50.9** | **9.2x** |
| SWA(1024)   | 991  | 5.392 / 6.0  | **0.496 / 65.0** | **10.9x** |
| SWA(1024)   | 2048 | 15.26 / 9.0  | **1.238 / 111.0**| **12.3x** |
| full-causal | 512  | 0.107 / 80.5 | 0.107 / 80.4 | 1.00x |
| full-causal | 991  | 0.322 / 99.8 | 0.322 / 99.8 | 1.00x |
| full-causal | 2048 | 1.137 / 120.9| 1.138 / 120.8| 1.00x |

Patched SGLang SWA @ S=991 (0.496 ms) is now **faster than vLLM (0.649 ms)** and
**faster than bf16 SDPA (0.517 ms)**, sitting ~3x off the FlashInfer floor
(0.163 ms) — down from ~33x for stock.

---

## Correctness validation (all kernels)

Closes the gap that RESULTS.md only *timed* kernels. **Oracle: PyTorch SDPA run
in fp32 with an explicit boolean mask** (causal, or causal+sliding-window).
FlashAttention-2 is unavailable (see skipped kernels), so fp32 SDPA is the oracle;
it is distinct from the bf16 kernels under test. One base Q/K/V set (seed 0);
every kernel derives its layout from the same tensors. Each kernel is compared to
an oracle using **its own window convention** (SGLang 1025 keys; vLLM / FlashInfer
/ SDPA 1024 keys), so the 1-key convention difference is not counted as error.
Pass criterion: max-abs-diff <= 2e-2 (bf16). Run per-venv:
`bench_correctness.py` in both the sglang and vllm venvs.

| kernel | mode | S | max abs | mean abs | pass |
|---|---|---:|---:|---:|:--:|
| vLLM unified      | full | 991  | 0.00865 | 0.000156 | PASS |
| vLLM unified      | SWA  | 991  | 0.00865 | 0.000156 | PASS |
| vLLM unified      | full | 2048 | 0.00819 | 0.000114 | PASS |
| vLLM unified      | SWA  | 2048 | 0.00819 | 0.000121 | PASS |
| SGLang stock      | full | 991  | 0.00865 | 0.000157 | PASS |
| SGLang stock      | SWA  | 991  | 0.00865 | 0.000157 | PASS |
| SGLang stock      | full | 2048 | 0.00819 | 0.000114 | PASS |
| SGLang stock      | SWA  | 2048 | 0.00819 | 0.000121 | PASS |
| SGLang **patched**| full | 991  | 0.00865 | 0.000157 | PASS |
| SGLang **patched**| SWA  | 991  | 0.00865 | 0.000157 | PASS |
| SGLang **patched**| full | 2048 | 0.00819 | 0.000114 | PASS |
| SGLang **patched**| SWA  | 2048 | 0.00819 | 0.000121 | PASS |
| FlashInfer        | full | 991  | 0.00805 | 0.000156 | PASS |
| FlashInfer        | SWA  | 991  | 0.00805 | 0.000156 | PASS |
| FlashInfer        | full | 2048 | 0.00859 | 0.000114 | PASS |
| FlashInfer        | SWA  | 2048 | 0.00859 | 0.000121 | PASS |
| SDPA (bf16)       | full | 991  | 0.00865 | 0.000157 | PASS |
| SDPA (bf16)       | SWA  | 991  | 0.00865 | 0.000158 | PASS |
| SDPA (bf16)       | full | 2048 | 0.00819 | 0.000114 | PASS |
| SDPA (bf16)       | SWA  | 2048 | 0.00819 | 0.000122 | PASS |

**Every kernel passes** (max-abs ~0.008, well under the 0.02 bf16 tolerance;
consistent with fp32 online-softmax reduction). This retroactively confirms the
original RESULTS.md vLLM and SGLang-stock timings were of *correct* computations,
and that the patched kernel is correct. No kernel's timing is invalidated.

At S=991 the sliding-window and full-causal outputs are identical for every
kernel (window inactive at S<=1024); at S=2048 the window is active and the SWA
rows validate the window masking itself.

---

## Verdict

**(a) Distance to the floor on SWA.** FlashInfer is the reference floor
(~0.16 ms, ~197 TFLOP/s @ S=991). vLLM's sliding-window path is ~4x off the floor
(it uses window-aware loop bounds and degrades gracefully). SGLang's **stock** SWA
kernel is ~**33x** off the floor (5.39 ms, ~6 TFLOP/s) — the concrete form of its
~8x-vs-vLLM deficit, and the reason a 50-SWA/10-full Gemma-4 prefill is dominated
by this one kernel.

**(b) Does the minimal patch recover the deficit?** Yes — essentially completely.
The two-edit patch (window loop bound + removing the pipelining-killing
`SKIP_TILE` branch) makes SGLang's SWA kernel **9-12x faster** (10.9x @ S=991)
with **bitwise-identical output**, leaves the full-causal path untouched, and
lifts SGLang SWA from ~33x off the floor to ~3x — now **faster than vLLM** and
faster than bf16 SDPA on the same shape. The 8x SWA deficit is not fundamental to
Triton or to the A100; it is a fixable correctness-first implementation detail,
and this minimal, output-preserving change recovers effectively all of it.
