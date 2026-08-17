# The two Triton prefill attention kernels

On A100, both engines run Gemma-4 prefill through a **Triton** attention kernel
(FlashAttention/FlashInfer are architecturally rejected for this model — see
[../measurement/prefill-decomposition.md](../measurement/prefill-decomposition.md)).
"Triton" is the *language*, not a shared kernel: these are two independent
implementations, and the whole conc=1 prefill gap (vLLM 132 ms vs SGLang 244 ms
full-reuse; 485 vs 724 cold) lives here. These are the exact source files from the
pinned engine versions (vLLM 0.26.0, SGLang 0.5.16), copied verbatim for
side-by-side reading and kernel-level benchmarking.

## vLLM — `vllm/`
- `triton_unified_attention.py` — `unified_attention`, the backend actually
  selected on the pod (`AttentionBackendEnum.TRITON_ATTN`). Single kernel handles
  prefill + decode; applies causal + sliding-window masking **in-kernel** (no
  materialized mask tensor). This is the code split *out* of vLLM's Inductor graph
  via `splitting_ops`.
- `triton_prefill_attention.py` — the dedicated prefill helper.

## SGLang — `sglang/`
- `extend_attention.py` — `extend_attention_fwd` / `extend_attention_fwd_unified`,
  the extend (prefill) kernel from `sglang.kernels.ops.attention.extend_attention`.
  Takes a **`custom_mask` tensor** — a general masked-attention path, not a
  specialized causal/windowed one.
- `triton_backend.py` — `TritonAttnBackend`, which wires the kernel in. Note
  line ~137: `self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)`
  — the kernel is explicitly excluded from torch.compile/Inductor (same posture as
  vLLM's `splitting_ops`), which is why no compilation lever moves the number.

## Why this comparison
Published data already points the same way: FA2-CUDA is ~1.3–1.5× faster than
Triton-FA on A100 forward pass, and SGLang's own docs recommend FlashInfer over
Triton on A100 (Ampere). The kernel benchmark in `bench/` (added by the
comparison run) isolates these two kernels head-to-head at Gemma-4 prefill shapes.
