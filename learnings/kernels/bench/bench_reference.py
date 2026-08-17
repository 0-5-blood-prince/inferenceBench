"""
TIER 1 reference-kernel microbenchmark at Gemma-4-31B-it prefill shapes.

Benchmarks the reference / gold-standard attention kernels at the SAME shapes the
existing vLLM vs SGLang head-to-head used (see bench/RESULTS.md), so vLLM and
SGLang can be placed against the CUDA / library performance floor on the
sliding-window shape:

    head_dim=256, num_query_heads=32, num_kv_heads=16 (GQA group=2),
    sliding_window=1024, dtype=bfloat16, batch=1,
    COLD self-attention prefill (prefix_len=0): S tokens attend causally to
    those same S tokens.  S in {512, 991, 2048}.

Kernels (whichever import successfully in the sglang venv):
  1. FlashAttention (flash_attn.cute.flash_attn_func) -- the installed build is
     flash-attn 4.0.0b19 (CuTe-DSL FA4), which exposes the classic
     flash_attn_func API with an Ampere/sm80 forward. causal=True; full uses
     window_size=(None,None), SWA uses window_size=(1023,0) -> a 1024-key window.
     This is the CUDA gold standard / performance floor.
  2. FlashInfer prefill (flashinfer.prefill.single_prefill_with_kv_cache), ragged
     layout NHD, causal=True; full uses window_left=-1, SWA uses window_left=1023
     -> a 1024-key window. Backend SGLang recommends on A100.
  3. PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention): full uses
     is_causal=True, SWA uses a materialized boolean sliding-window mask
     (same 1024-key window). Baseline.

All SWA references use an identical 1024-key window (q attends kv where
q-1023 <= kv <= q), matching vLLM's window_size=(1023,0) in the existing bench.
(SGLang's stock bench used sliding_window_size=1024 -> 1025 keys; a 1-key
difference that is irrelevant to timing.)

Timing: >=15 warmup, median of >=60 iters timed with torch.cuda.Event; only the
kernel call is timed (inputs prebuilt).

FLOP formula (identical to the existing bench so TFLOP/s is comparable):
    flops = 2 * 2 * batch * num_query_heads * S^2 * head_dim
"""

import json
import torch
import torch.nn.functional as F

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 256
N_QHEADS = 32
N_KVHEADS = 16
SLIDING_WINDOW = 1024          # number of keys in the window
WINDOW_LEFT = SLIDING_WINDOW - 1  # 1023 -> keys [q-1023, q] inclusive = 1024 keys
SEQLENS = [512, 991, 2048]
WARMUP = 15
ITERS = 60


def flops_tflops(S, median_ms):
    flops = 2 * 2 * 1 * N_QHEADS * S * S * HEAD_DIM
    return flops / (median_ms * 1e-3) / 1e12


def time_kernel(run_once):
    for _ in range(WARMUP):
        run_once()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        run_once()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(ITERS))
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# FlashAttention (flash-attn 4 CuTe DSL)
# ---------------------------------------------------------------------------
def make_fa(S, sliding):
    from flash_attn.cute import flash_attn_func
    torch.manual_seed(0)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    # FA layout: [batch, seqlen, nheads, head_dim]
    q = torch.randn(1, S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    window = (WINDOW_LEFT, 0) if sliding else (None, None)

    def run():
        flash_attn_func(q, k, v, softmax_scale=scale, causal=True,
                        window_size=window)
    return run


# ---------------------------------------------------------------------------
# FlashInfer ragged single prefill
# ---------------------------------------------------------------------------
def make_flashinfer(S, sliding):
    from flashinfer.prefill import single_prefill_with_kv_cache
    torch.manual_seed(0)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    # NHD layout: [seqlen, nheads, head_dim]
    q = torch.randn(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    window_left = WINDOW_LEFT if sliding else -1

    def run():
        single_prefill_with_kv_cache(
            q, k, v, causal=True, kv_layout="NHD", sm_scale=scale,
            window_left=window_left,
        )
    return run


# ---------------------------------------------------------------------------
# PyTorch SDPA
# ---------------------------------------------------------------------------
def make_sdpa(S, sliding):
    torch.manual_seed(0)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    # SDPA layout: [batch, nheads, seqlen, head_dim]; expand KV heads for GQA.
    q = torch.randn(1, N_QHEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, N_KVHEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, N_KVHEADS, S, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    rep = N_QHEADS // N_KVHEADS
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)

    attn_mask = None
    is_causal = True
    if sliding:
        idx = torch.arange(S, device=DEVICE)
        # allowed: j <= i (causal) AND i - j <= WINDOW_LEFT (window) -> 1024 keys
        allowed = (idx[None, :] <= idx[:, None]) & (
            idx[:, None] - idx[None, :] <= WINDOW_LEFT
        )
        attn_mask = allowed  # bool mask, True = keep
        is_causal = False

    def run():
        F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale,
        )
    return run


KERNELS = {
    "flash_attn": make_fa,
    "flashinfer": make_flashinfer,
    "sdpa": make_sdpa,
}


def bench(name, maker, S, sliding):
    run = maker(S, sliding)
    ms = time_kernel(run)
    return ms, flops_tflops(S, ms)


def main():
    print(json.dumps({"tier": 1, "device": torch.cuda.get_device_name(0),
                      "kv_heads": N_KVHEADS, "q_heads": N_QHEADS,
                      "head_dim": HEAD_DIM, "window_keys": SLIDING_WINDOW}))
    for name, maker in KERNELS.items():
        for sliding in (False, True):
            mode = "sliding1024" if sliding else "full_causal"
            for S in SEQLENS:
                try:
                    ms, tf = bench(name, maker, S, sliding)
                    print(json.dumps({"kernel": name, "mode": mode, "seqlen": S,
                                      "median_ms": round(ms, 4),
                                      "tflops": round(tf, 2)}))
                except Exception as e:
                    print(json.dumps({"kernel": name, "mode": mode, "seqlen": S,
                                      "error": repr(e)[:300]}))


if __name__ == "__main__":
    main()
