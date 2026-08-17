"""
Kernel-level microbenchmark: vLLM 0.26.0 unified_attention (Triton) at
Gemma-4-31B-it prefill shapes.

Run in the vLLM venv on the A100 pod:
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
    /root/venvs/vllm/bin/python bench_vllm.py

Model shapes (google/gemma-4-31B-it, text_config):
    head_dim=256, num_attention_heads=32, num_key_value_heads=16,
    sliding_window=1024, dtype=bfloat16.

We benchmark a COLD, self-attention prefill: a single request of S tokens
attending causally to those same S tokens (no cached prefix). This matches the
"cold full prefill" end-to-end case and is the cleanest apples-to-apples unit
vs. SGLang's extend kernel with prefix_len=0.

unified_attention uses a PAGED KV cache. We build a block table that lays the S
tokens contiguously into blocks (slot order == token order) and write the new
K/V into those cache slots, exactly as vLLM does after its reshape_and_cache
step. cu_seqlens_q=[0,S], seqused_k=[S], max_seqlen_q=max_seqlen_k=S.

FLOP formula (documented, identical for both engines so the ratio is unaffected):
    flops = 2 * 2 * batch * num_query_heads * S^2 * head_dim
The two 2's: factor 2 for multiply-add, factor 2 for the two matmuls (QK^T and
P@V). This counts the FULL S*S score matrix (does not discount the causal
triangle), so reported TFLOP/s is a consistent "dense-equivalent" throughput
proxy, comparable across engines but not an MFU claim.
"""

import json
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 256
N_QHEADS = 32
N_KVHEADS = 16
SLIDING_WINDOW = 1024
BLOCK_SIZE = 16
SEQLENS = [128, 512, 991, 2048]
WARMUP = 15
ITERS = 60


def build_inputs(S, sliding):
    torch.manual_seed(0)
    scale = 1.0 / (HEAD_DIM ** 0.5)

    q = torch.randn(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    out = torch.empty_like(q)

    n_blocks_needed = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = n_blocks_needed + 4
    # Paged KV cache: [num_blocks, block_size, num_kv_heads, head_size]
    key_cache = torch.randn(num_blocks, BLOCK_SIZE, N_KVHEADS, HEAD_DIM,
                            device=DEVICE, dtype=DTYPE)
    value_cache = torch.randn(num_blocks, BLOCK_SIZE, N_KVHEADS, HEAD_DIM,
                              device=DEVICE, dtype=DTYPE)

    # Block table: token i -> block (i//BLOCK_SIZE), slot (i%BLOCK_SIZE).
    block_table = torch.arange(num_blocks, device=DEVICE, dtype=torch.int32).view(1, -1)

    cu_seqlens_q = torch.tensor([0, S], device=DEVICE, dtype=torch.int32)
    seqused_k = torch.tensor([S], device=DEVICE, dtype=torch.int32)

    # window_size[0] = w-1 gives SLIDING_WINDOW constexpr = 1+ (w-1) = w in kernel.
    window_size = (SLIDING_WINDOW - 1, 0) if sliding else (-1, -1)

    return dict(
        q=q, k=key_cache, v=value_cache, out=out,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=S,
        seqused_k=seqused_k, max_seqlen_k=S,
        softmax_scale=scale, causal=True, window_size=window_size,
        block_table=block_table, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
    )


def run_once(args):
    unified_attention(
        q=args["q"], k=args["k"], v=args["v"], out=args["out"],
        cu_seqlens_q=args["cu_seqlens_q"], max_seqlen_q=args["max_seqlen_q"],
        seqused_k=args["seqused_k"], max_seqlen_k=args["max_seqlen_k"],
        softmax_scale=args["softmax_scale"], causal=args["causal"],
        window_size=args["window_size"], block_table=args["block_table"],
        softcap=args["softcap"], q_descale=args["q_descale"],
        k_descale=args["k_descale"], v_descale=args["v_descale"],
    )


def bench(S, sliding):
    args = build_inputs(S, sliding)
    for _ in range(WARMUP):
        run_once(args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        run_once(args)
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(ITERS))
    median_ms = times[len(times) // 2]
    flops = 2 * 2 * 1 * N_QHEADS * S * S * HEAD_DIM
    tflops = flops / (median_ms * 1e-3) / 1e12
    return median_ms, tflops


def main():
    print(json.dumps({"engine": "vllm", "device": torch.cuda.get_device_name(0),
                      "kv_heads": N_KVHEADS, "q_heads": N_QHEADS,
                      "head_dim": HEAD_DIM}))
    for sliding in (False, True):
        mode = "sliding1024" if sliding else "full_causal"
        for S in SEQLENS:
            try:
                ms, tf = bench(S, sliding)
                print(json.dumps({"engine": "vllm", "mode": mode, "seqlen": S,
                                  "median_ms": round(ms, 4), "tflops": round(tf, 2)}))
            except Exception as e:
                print(json.dumps({"engine": "vllm", "mode": mode, "seqlen": S,
                                  "error": repr(e)}))


if __name__ == "__main__":
    main()
