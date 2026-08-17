"""
Kernel-level microbenchmark: SGLang 0.5.16 extend_attention_fwd (Triton) at
Gemma-4-31B-it prefill shapes.

Run in the SGLang venv on the A100 pod:
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
    /root/venvs/sglang/bin/python bench_sglang.py

Model shapes (google/gemma-4-31B-it, text_config):
    head_dim=256, num_attention_heads=32, num_key_value_heads=16,
    sliding_window=1024, dtype=bfloat16.

We benchmark a COLD, self-attention prefill: prefix_len=0, extend_len=S, so the
kernel's stage-1 (prefix) loop runs 0 iterations and all work is stage-2, the
causal self-attention over S new tokens. This is the same computation vLLM's
unified_attention performs for a cold S-token prefill -> apples-to-apples.

extend_attention_fwd layout:
    q_extend/k_extend/v_extend: [S, heads, head_dim] (new tokens, contiguous)
    o_extend:                   [S, q_heads, head_dim]
    k_buffer/v_buffer:          paged/flat KV buffer (page_size=1); unused here
                                because prefix_len=0 (kv_indices empty).
    qo_indptr=[0,S], kv_indptr=[0,0] (no prefix), kv_indices=empty.
    is_causal=True (no custom_mask -> the fast causal path, matching a real
    Gemma-4 full-attention prefill layer; NOT the general custom_mask path).

FLOP formula (documented, identical for both engines so the ratio is unaffected):
    flops = 2 * 2 * batch * num_query_heads * S^2 * head_dim
"""

import json
import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 256
N_QHEADS = 32
N_KVHEADS = 16
SLIDING_WINDOW = 1024
SEQLENS = [128, 512, 991, 2048]
WARMUP = 15
ITERS = 60


def build_inputs(S, sliding):
    torch.manual_seed(0)
    scale = 1.0 / (HEAD_DIM ** 0.5)

    q_extend = torch.randn(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k_extend = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_extend = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    o_extend = torch.empty(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    # KV buffer (page_size=1). prefix_len=0 so this is never read, but must be a
    # valid [n_tokens, kv_heads, head_dim] tensor for stride extraction.
    k_buffer = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_buffer = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    qo_indptr = torch.tensor([0, S], device=DEVICE, dtype=torch.int32)
    kv_indptr = torch.tensor([0, 0], device=DEVICE, dtype=torch.int32)  # no prefix
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


def run_once(a):
    extend_attention_fwd(
        a["q_extend"], a["k_extend"], a["v_extend"], a["o_extend"],
        a["k_buffer"], a["v_buffer"], a["qo_indptr"], a["kv_indptr"],
        a["kv_indices"], a["custom_mask"], a["is_causal"], a["mask_indptr"],
        a["max_len_extend"], a["k_scale"], a["v_scale"],
        sm_scale=a["sm_scale"], sliding_window_size=a["sliding_window_size"],
    )


def bench(S, sliding):
    a = build_inputs(S, sliding)
    for _ in range(WARMUP):
        run_once(a)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        run_once(a)
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(ITERS))
    median_ms = times[len(times) // 2]
    flops = 2 * 2 * 1 * N_QHEADS * S * S * HEAD_DIM
    tflops = flops / (median_ms * 1e-3) / 1e12
    return median_ms, tflops


def main():
    print(json.dumps({"engine": "sglang", "device": torch.cuda.get_device_name(0),
                      "kv_heads": N_KVHEADS, "q_heads": N_QHEADS,
                      "head_dim": HEAD_DIM}))
    for sliding in (False, True):
        mode = "sliding1024" if sliding else "full_causal"
        for S in SEQLENS:
            try:
                ms, tf = bench(S, sliding)
                print(json.dumps({"engine": "sglang", "mode": mode, "seqlen": S,
                                  "median_ms": round(ms, 4), "tflops": round(tf, 2)}))
            except Exception as e:
                print(json.dumps({"engine": "sglang", "mode": mode, "seqlen": S,
                                  "error": repr(e)}))


if __name__ == "__main__":
    main()
