# Tier 3 — H100 (Hopper) results: trtllm_mha, and the SWA pathology is Ampere-specific

Ran on a rented **H100 NVL (94 GB, Hopper SM90)**, SGLang 0.5.16, same Gemma-4-31B-it
shapes. Two questions: (1) does SGLang's `trtllm_mha` backend close the gap by
config on Hopper (it's in Gemma-4's accepted list, unlike on A100)? (2) how do the
Triton kernels behave on Hopper vs the A100?

## Finding 1 — `trtllm_mha` needs Blackwell, not just Hopper

`--attention-backend trtllm_mha` **failed to start** on the H100:

```
ValueError: TRTLLM MHA backend for prefill is only supported on Blackwell GPUs
(SM100). Please use a different prefill backend.
```

It passes Gemma-4's accepted-backend assert (`trtllm_mha/triton/ascend/intel_xpu`)
but a *separate* compatibility check (`server_args.py:_handle_attention_backend_
compatibility`) rejects trtllm_mha **prefill** below SM100. So the config-available
fused-kernel fix requires **Blackwell (B200/GB200)** — it is unavailable on both
A100 (SM80) and H100 (SM90). On all pre-Blackwell NVIDIA GPUs, SGLang's Gemma-4
prefill is stuck on the Triton backend. (End-to-end trtllm_mha therefore has NO
DATA; triton is the only usable backend.)

## Finding 2 — the A100 SWA pathology is largely Ampere-specific

Kernel microbench on the H100 (SWA(1024), S=991, median ms / TFLOP·s⁻¹):

| kernel | ms | TFLOP/s |
|---|---:|---:|
| flash_attn (FA4) | 0.066 | 487 |
| flashinfer | 0.078 | 415 |
| SDPA | 0.245 | 131 |
| **SGLang extend** | **0.261** | **123** |

The decisive comparison — SGLang's **SWA vs its own full-causal** kernel:

| S | SGLang causal ms | SGLang SWA ms | SWA/causal |
|---:|---:|---:|---:|
| 512 | 0.066 | 0.084 | 1.27× |
| 991 | 0.199 | 0.261 | **1.31×** |
| 2048 | 0.704 | 0.722 | 1.03× |

On the **A100** this ratio was **~17×** (SWA 5.39 ms vs causal 0.32 ms). On the
**H100 it is ~1.3×.** The catastrophic sliding-window penalty — the per-tile
`SKIP_TILE` reduction + data-dependent branch defeating software-pipelining — is an
**Ampere phenomenon**; Hopper's async pipeline hides it. So the specific 8× SWA
deficit that drove the A100 end-to-end gap does **not** reproduce on Hopper.

What *does* persist on Hopper: SGLang's Triton kernels (both causal and SWA) run at
~120–195 TFLOP/s while FlashInfer/FA4 reach ~410–800 — SGLang Triton is a **uniform
~3–4× off the fused kernels** on H100, for causal and SWA alike. That is the general
"Triton vs hand-tuned fused kernel" gap, not the SWA-specific bug.

## Finding 3 — end-to-end conc=1 on H100 (triton only)

| workload | SGLang triton (H100) | SGLang triton (A100) | vLLM (A100) |
|---|---:|---:|---:|
| full-reuse | 110 ms [117,103] | 244 ms | 132 ms |
| cold | 328 ms [329,327] | 724 ms | 485 ms |

H100 is ~2.2× faster than A100 for SGLang cold prefill — more than raw hardware,
consistent with the SWA penalty being milder on Hopper. (No vLLM/trtllm_mha on H100:
trtllm_mha rejected; vLLM not bootstrapped to save time.)

## Verdict
1. **The config fix (`trtllm_mha`) is Blackwell-only** — unavailable on A100 and
   H100. On pre-Blackwell GPUs the only remedies are the source patch
   ([why-sglang-swa-slow.md](../why-sglang-swa-slow.md)) or SGLang optimizing the
   Triton SWA kernel.
2. **The dramatic ~8–17× SWA penalty is Ampere-specific** (A100), driven by the
   pipelining-killing branch; on Hopper it shrinks to ~1.3×. The A100 headline gap
   is real but hardware-conditioned — worth stating precisely rather than as a
   blanket "SGLang's SWA kernel is 8× slow everywhere."
3. **Uniformly, Triton on NVIDIA is ~3–4× off the fused kernels** (FlashInfer/FA)
   for this model — true on both A100 and H100 — but Gemma-4 forbids those backends
   below Blackwell, so Triton is what runs.

Raw data: [tier3_out/](tier3_out/) (bench_reference.jsonl, bench_sglang.jsonl,
e2e_summary.txt, backend_evidence.txt, *_trtllm_mha_failtail.log).
