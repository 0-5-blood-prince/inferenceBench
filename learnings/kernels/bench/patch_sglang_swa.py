"""
TIER 2 -- patch SGLang's extend_attention sliding-window kernel and prove the fix.

Root cause (see learnings/kernels/why-sglang-swa-slow.md): in the stage-2 self-
attention loop of `_fwd_kernel`, SGLang (0.5.16) handles the sliding window as a
per-tile MASK over the FULL causal extent -- it never narrows the loop to the
window -- and, whenever SLIDING_WINDOW_SIZE > 0, computes a per-tile cross-warp
reduction `SKIP_TILE = tl.max(tl.max(final_mask))` plus a data-dependent
`if not SKIP_TILE:` branch. That branch defeats Triton's software-pipelining of
K/V loads against the matmuls, which is why the SWA path is ~8x slower than
vLLM's and ~17x slower than SGLang's own full-causal path even at S<=window
(where nothing is masked).

This script applies a MINIMAL, correctness-preserving patch to a COPY of the
installed kernel (the installed package is left untouched), then:
  1. verifies the patched output matches (a) the stock kernel and (b) an SDPA
     reference within a bf16 tolerance at S=991 SWA(1024) -- a faster-but-wrong
     kernel is worthless;
  2. benchmarks stock vs patched at the SWA shapes and reports the speedup.

The patch (two edits, SWA path only; full-causal path is byte-identical because
both edits are gated on SLIDING_WINDOW_SIZE > 0 / are no-ops when it's <= 0):

  (a) Narrow the stage-2 loop LOWER bound to the window. Keys older than
      (query - SLIDING_WINDOW_SIZE) are out of window for every row of the M-tile
      (smallest query index = cur_block_m*BLOCK_M), so they are never iterated
      (analogous to vLLM's compute_tile_loop_bounds). Aligned down to BLOCK_N so
      the boundary tile is still visited and masked.
  (b) Drop the per-tile SKIP_TILE reduction + data-dependent branch for the pure
      sliding-window path (keep it only for USE_CUSTOM_MASK). The window is still
      applied correctly via `final_mask` (tl.where -> -inf); with the narrowed
      range from (a) no in-range tile is ever fully masked, so removing the skip
      changes no result -- it only restores straight-line, pipelineable code.

Run in the sglang venv on the A100 pod:
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
    /root/venvs/sglang/bin/python patch_sglang_swa.py
"""

import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F

import sglang.kernels.ops.attention.extend_attention as stock_mod
from sglang.kernels.ops.attention.extend_attention import (
    extend_attention_fwd as extend_attention_fwd_stock,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 256
N_QHEADS = 32
N_KVHEADS = 16
SLIDING_WINDOW = 1024
SEQLENS = [512, 991, 2048]
WARMUP = 15
ITERS = 60

PATCHED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "extend_attention_patched.py")

# ---------------------------------------------------------------------------
# Build the patched kernel module from the installed source via two textual
# edits. The (old, new) pairs below ARE the diff.
# ---------------------------------------------------------------------------
PATCH_A_OLD = """    extend_end = 0 if SKIP_EXTEND else cur_block_m_end
    for start_n in range(0, extend_end, BLOCK_N):"""

PATCH_A_NEW = """    extend_end = 0 if SKIP_EXTEND else cur_block_m_end
    # [PATCH a] Sliding-window loop LOWER bound. Keys older than
    # (query - SLIDING_WINDOW_SIZE) are out of window for every row in this
    # M-tile (smallest query index = cur_block_m*BLOCK_M), so never iterate them.
    # Aligned down to BLOCK_N so the boundary tile is still visited & masked.
    extend_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        window_lo = cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE
        window_lo = tl.maximum(window_lo, 0)
        extend_start = (window_lo // BLOCK_N) * BLOCK_N
    for start_n in range(extend_start, extend_end, BLOCK_N):"""

# Target the STAGE-2 SKIP_TILE only (the stage-1/prefix one at ~L396 is a
# different, uniquely-indented block and is left untouched -- prefix loop is
# empty in a cold prefill).
PATCH_B_OLD = """        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # load k in transposed way"""

PATCH_B_NEW = """        SKIP_TILE = False
        # [PATCH b] Do NOT run the per-tile cross-warp reduction + data-dependent
        # branch for the pure sliding-window path -- it defeats Triton's
        # software-pipelining of K/V loads vs the matmuls. The window is still
        # applied via `final_mask` (tl.where below); with the narrowed loop range
        # from [PATCH a] no in-range tile is ever fully masked anyway.
        if USE_CUSTOM_MASK:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # load k in transposed way"""


def build_patched_module():
    src = open(stock_mod.__file__).read()
    for old, new in ((PATCH_A_OLD, PATCH_A_NEW), (PATCH_B_OLD, PATCH_B_NEW)):
        if src.count(old) != 1:
            raise RuntimeError(
                f"patch anchor not found exactly once (found {src.count(old)}):\n{old[:80]}..."
            )
        src = src.replace(old, new)
    with open(PATCHED_PATH, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("extend_attention_patched",
                                                  PATCHED_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extend_attention_patched"] = mod
    spec.loader.exec_module(mod)
    return mod.extend_attention_fwd


# ---------------------------------------------------------------------------
# Inputs (same layout as bench_sglang.py: cold self-attention, prefix_len=0)
# ---------------------------------------------------------------------------
def build_inputs(S, sliding):
    torch.manual_seed(0)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    q_extend = torch.randn(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k_extend = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_extend = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    o_extend = torch.empty(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k_buffer = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_buffer = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    qo_indptr = torch.tensor([0, S], device=DEVICE, dtype=torch.int32)
    kv_indptr = torch.tensor([0, 0], device=DEVICE, dtype=torch.int32)
    kv_indices = torch.empty(0, device=DEVICE, dtype=torch.int64)
    mask_indptr = torch.zeros(2, device=DEVICE, dtype=torch.int64)
    return dict(
        q_extend=q_extend, k_extend=k_extend, v_extend=v_extend,
        o_extend=o_extend, k_buffer=k_buffer, v_buffer=v_buffer,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
        custom_mask=None, is_causal=True, mask_indptr=mask_indptr,
        max_len_extend=S, k_scale=1.0, v_scale=1.0, sm_scale=scale,
        sliding_window_size=(SLIDING_WINDOW if sliding else -1),
    )


def call(fn, a):
    fn(a["q_extend"], a["k_extend"], a["v_extend"], a["o_extend"],
       a["k_buffer"], a["v_buffer"], a["qo_indptr"], a["kv_indptr"],
       a["kv_indices"], a["custom_mask"], a["is_causal"], a["mask_indptr"],
       a["max_len_extend"], a["k_scale"], a["v_scale"],
       sm_scale=a["sm_scale"], sliding_window_size=a["sliding_window_size"])


def sdpa_reference(a, S):
    # SGLang window semantics: q attends kv where q <= kv + W  ->  q-W <= kv <= q
    # i.e. W+1 = 1025 keys. Build the matching boolean mask.
    scale = a["sm_scale"]
    q = a["q_extend"].transpose(0, 1).unsqueeze(0).float()   # [1,H,S,D]
    k = a["k_extend"].transpose(0, 1).unsqueeze(0).float()
    v = a["v_extend"].transpose(0, 1).unsqueeze(0).float()
    rep = N_QHEADS // N_KVHEADS
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    idx = torch.arange(S, device=DEVICE)
    allowed = (idx[None, :] <= idx[:, None]) & (
        idx[:, None] - idx[None, :] <= SLIDING_WINDOW
    )
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scale)
    return out[0].transpose(0, 1).contiguous()  # [S,H,D]


def time_call(fn, a):
    for _ in range(WARMUP):
        call(fn, a)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        call(fn, a)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(ITERS))
    return times[len(times) // 2]


def tflops(S, ms):
    return (2 * 2 * 1 * N_QHEADS * S * S * HEAD_DIM) / (ms * 1e-3) / 1e12


def main():
    patched_fwd = build_patched_module()
    print(json.dumps({"tier": 2, "device": torch.cuda.get_device_name(0),
                      "patched_file": PATCHED_PATH}))

    # ---- correctness at S=991 SWA(1024) ----
    S = 991
    a_stock = build_inputs(S, sliding=True)
    a_patch = build_inputs(S, sliding=True)   # identical seed -> identical inputs
    call(extend_attention_fwd_stock, a_stock)
    call(patched_fwd, a_patch)
    torch.cuda.synchronize()
    ref = sdpa_reference(a_stock, S)
    d_patch_vs_stock = (a_patch["o_extend"].float() - a_stock["o_extend"].float()).abs().max().item()
    d_patch_vs_sdpa = (a_patch["o_extend"].float() - ref.float()).abs().max().item()
    d_stock_vs_sdpa = (a_stock["o_extend"].float() - ref.float()).abs().max().item()
    print(json.dumps({"correctness_S": S, "window": SLIDING_WINDOW,
                      "max_abs_patched_vs_stock": round(d_patch_vs_stock, 6),
                      "max_abs_patched_vs_sdpa": round(d_patch_vs_sdpa, 6),
                      "max_abs_stock_vs_sdpa": round(d_stock_vs_sdpa, 6),
                      "pass": bool(d_patch_vs_stock < 1e-2 and d_patch_vs_sdpa < 1e-2)}))

    # ---- benchmark stock vs patched ----
    for sliding in (True, False):
        mode = "sliding1024" if sliding else "full_causal"
        for S in SEQLENS:
            a_s = build_inputs(S, sliding)
            a_p = build_inputs(S, sliding)
            ms_s = time_call(extend_attention_fwd_stock, a_s)
            ms_p = time_call(patched_fwd, a_p)
            print(json.dumps({
                "mode": mode, "seqlen": S,
                "stock_ms": round(ms_s, 4), "stock_tflops": round(tflops(S, ms_s), 2),
                "patched_ms": round(ms_p, 4), "patched_tflops": round(tflops(S, ms_p), 2),
                "speedup_stock_over_patched": round(ms_s / ms_p, 2),
            }))


if __name__ == "__main__":
    main()
