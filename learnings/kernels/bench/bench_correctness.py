"""
CORRECTNESS validation for every benchmarked attention kernel.

The existing RESULTS.md only TIMED the kernels; it never checked they compute the
right attention output. This closes that gap: for each kernel we compare its
attention output against a trusted oracle and report max / mean absolute error.

ORACLE: PyTorch SDPA run in **fp32** with an explicit boolean mask
(causal, or causal+sliding-window). FlashAttention-2 is NOT available on this pod
(the only flash-attn build is 4.0.0b19, a CuTe-DSL beta that fails to compile on
this A100), so fp32-SDPA-with-explicit-mask is the oracle. It is distinct from the
kernels under test (which run in bf16) so the comparison is meaningful.

Inputs: one base set of Q/K/V (seed 0), head_dim=256, 32 q heads / 16 kv heads
(GQA=2), batch=1, bf16. Every kernel derives its required layout from the SAME
base tensors, so all kernels see identical numbers.

Shapes: S in {991, 2048}, mode in {full_causal, sliding1024}.
  - At S=991 <= window=1024 the sliding window masks nothing (every causal key is
    in-window), so the SWA output is numerically identical to the full-causal
    output -- validating the SWA code path still produces correct attention.
  - At S=2048 the window is active and genuinely exercises window masking. Each
    kernel is compared against an oracle using THAT kernel's own window
    convention (SGLang: q<=kv+W -> W+1=1025 keys; vLLM/FlashInfer/SDPA: W=1024
    keys), so a 1-key convention difference is not counted as an error.

Pass criterion: max-abs-diff <= 2e-2 (bf16 accumulation). Kernels using fp32
reduction are expected to be tighter. A kernel that FAILS the match has an
invalid timing and must not be counted.

This script validates whatever is importable in the CURRENT venv, so run it in
BOTH:
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
    /root/venvs/sglang/bin/python bench_correctness.py   # sglang, patched, flashinfer, sdpa
    /root/venvs/vllm/bin/python   bench_correctness.py   # vllm, sdpa
"""

import json
import torch
import torch.nn.functional as F

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 256
N_QHEADS = 32
N_KVHEADS = 16
SLIDING_WINDOW = 1024
BLOCK_SIZE = 16
SEQLENS = [991, 2048]
TOL = 2e-2


def base_inputs(S):
    torch.manual_seed(0)
    q = torch.randn(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(S, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, v


def oracle(q, k, v, S, window_keys):
    """fp32 SDPA reference. window_keys=None -> pure causal; else keep keys with
    i - j <= window_keys - 1 (and j <= i)."""
    scale = 1.0 / (HEAD_DIM ** 0.5)
    qf = q.float().transpose(0, 1).unsqueeze(0)            # [1,H,S,D]
    kf = k.float().transpose(0, 1).unsqueeze(0).repeat_interleave(N_QHEADS // N_KVHEADS, dim=1)
    vf = v.float().transpose(0, 1).unsqueeze(0).repeat_interleave(N_QHEADS // N_KVHEADS, dim=1)
    idx = torch.arange(S, device=DEVICE)
    allowed = idx[None, :] <= idx[:, None]
    if window_keys is not None:
        allowed = allowed & (idx[:, None] - idx[None, :] <= window_keys - 1)
    out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=allowed, scale=scale)
    return out[0].transpose(0, 1).contiguous()            # [S,H,D] fp32


def diffs(out, ref):
    d = (out.float() - ref.float()).abs()
    return d.max().item(), d.mean().item()


# ---------------------------------------------------------------------------
# Per-kernel runners: return output [S, N_QHEADS, HEAD_DIM]. window_keys is the
# oracle convention for that kernel's SWA mode.
# ---------------------------------------------------------------------------
def run_sglang(fwd, q, k, v, S, sliding):
    o = torch.empty(S, N_QHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    qo_indptr = torch.tensor([0, S], device=DEVICE, dtype=torch.int32)
    kv_indptr = torch.tensor([0, 0], device=DEVICE, dtype=torch.int32)
    kv_indices = torch.empty(0, device=DEVICE, dtype=torch.int64)
    mask_indptr = torch.zeros(2, device=DEVICE, dtype=torch.int64)
    fwd(q, k, v, o, k.clone(), v.clone(), qo_indptr, kv_indptr, kv_indices,
        None, True, mask_indptr, S, 1.0, 1.0, sm_scale=1.0 / (HEAD_DIM ** 0.5),
        sliding_window_size=(SLIDING_WINDOW if sliding else -1))
    return o, (SLIDING_WINDOW + 1 if sliding else None)   # SGLang: q<=kv+W -> 1025 keys


def run_flashinfer(q, k, v, S, sliding):
    from flashinfer.prefill import single_prefill_with_kv_cache
    o = single_prefill_with_kv_cache(
        q, k, v, causal=True, kv_layout="NHD", sm_scale=1.0 / (HEAD_DIM ** 0.5),
        window_left=(SLIDING_WINDOW - 1 if sliding else -1))
    return o, (SLIDING_WINDOW if sliding else None)       # window_left=1023 -> 1024 keys


def run_sdpa(q, k, v, S, sliding):
    scale = 1.0 / (HEAD_DIM ** 0.5)
    qb = q.transpose(0, 1).unsqueeze(0)
    kb = k.transpose(0, 1).unsqueeze(0).repeat_interleave(N_QHEADS // N_KVHEADS, dim=1)
    vb = v.transpose(0, 1).unsqueeze(0).repeat_interleave(N_QHEADS // N_KVHEADS, dim=1)
    if sliding:
        idx = torch.arange(S, device=DEVICE)
        allowed = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] <= SLIDING_WINDOW - 1)
        o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=allowed, scale=scale)
    else:
        o = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True, scale=scale)
    return o[0].transpose(0, 1).contiguous(), (SLIDING_WINDOW if sliding else None)


def run_vllm(q, k, v, S, sliding):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    out = torch.empty_like(q)
    n_blocks = (S + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    key_cache = torch.zeros(n_blocks, BLOCK_SIZE, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    value_cache = torch.zeros(n_blocks, BLOCK_SIZE, N_KVHEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    # block_table is identity -> flat cache slot == token index.
    key_cache.view(-1, N_KVHEADS, HEAD_DIM)[:S] = k
    value_cache.view(-1, N_KVHEADS, HEAD_DIM)[:S] = v
    block_table = torch.arange(n_blocks, device=DEVICE, dtype=torch.int32).view(1, -1)
    cu_seqlens_q = torch.tensor([0, S], device=DEVICE, dtype=torch.int32)
    seqused_k = torch.tensor([S], device=DEVICE, dtype=torch.int32)
    window_size = (SLIDING_WINDOW - 1, 0) if sliding else (-1, -1)
    unified_attention(
        q=q, k=key_cache, v=value_cache, out=out, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=S, seqused_k=seqused_k, max_seqlen_k=S,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5), causal=True, window_size=window_size,
        block_table=block_table, softcap=0.0, q_descale=None, k_descale=None,
        v_descale=None)
    return out, (SLIDING_WINDOW if sliding else None)      # window_size=(1023,0) -> 1024 keys


def collect_runners():
    runners = {}
    # SGLang stock + patched (sglang venv only)
    try:
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd as sg_stock
        runners["sglang_stock"] = lambda q, k, v, S, sl: run_sglang(sg_stock, q, k, v, S, sl)
        try:
            from patch_sglang_swa import build_patched_module
            sg_patched = build_patched_module()
            runners["sglang_patched"] = lambda q, k, v, S, sl: run_sglang(sg_patched, q, k, v, S, sl)
        except Exception as e:
            print(json.dumps({"note": "patched build skipped", "err": repr(e)[:200]}))
    except Exception:
        pass
    # FlashInfer
    try:
        import flashinfer  # noqa
        runners["flashinfer"] = run_flashinfer
    except Exception:
        pass
    # vLLM
    try:
        import vllm  # noqa
        runners["vllm_unified"] = run_vllm
    except Exception:
        pass
    # SDPA always available
    runners["sdpa"] = run_sdpa
    return runners


def main():
    print(json.dumps({"correctness": True, "device": torch.cuda.get_device_name(0),
                      "oracle": "fp32 SDPA + explicit mask", "tol_max_abs": TOL}))
    runners = collect_runners()
    print(json.dumps({"kernels_in_this_venv": list(runners.keys())}))
    for S in SEQLENS:
        q, k, v = base_inputs(S)
        for sliding in (False, True):
            mode = "sliding1024" if sliding else "full_causal"
            for name, fn in runners.items():
                try:
                    out, wk = fn(q, k, v, S, sliding)
                    ref = oracle(q, k, v, S, wk)
                    mx, mn = diffs(out, ref)
                    print(json.dumps({"kernel": name, "mode": mode, "seqlen": S,
                                      "max_abs": round(mx, 5), "mean_abs": round(mn, 6),
                                      "window_keys": wk, "pass": bool(mx <= TOL)}))
                except Exception as e:
                    print(json.dumps({"kernel": name, "mode": mode, "seqlen": S,
                                      "error": repr(e)[:200]}))


if __name__ == "__main__":
    main()
