#!/usr/bin/env python3
"""Aggregate a torch-profiler chrome trace's GPU kernel time by category and by
name. Pure stdlib - runs in any venv. Usage: parse_trace.py trace.json [trace2.json ...]
"""
import json
import re
import sys
import gzip
from collections import defaultdict

# Order matters: checked top-to-bottom, first match wins. Most-specific /
# least-ambiguous categories first, broad catch-alls (gemm/matmul) last, so a
# kernel that is genuinely a norm/attention op isn't stolen by an incidental
# substring (e.g. cutlass-templated norm kernels also contain "cutlass").
#
# Verified against the actual traces (not guessed): vLLM's real attention
# kernel is literally named "kernel_unified_attention" (no "attn" substring);
# SGLang's are the Triton internal names "_fwd_kernel" (extend/prefill),
# "_fwd_kernel_stage2" / "_fwd_grouped_kernel_stage1" (decode flash-decoding
# split-KV). A bare "flash" keyword false-matched vLLM's KV-cache-write kernel
# "reshape_and_cache_kernel_flash" (not attention compute) - dropped in favor
# of exact attention-kernel names. A bare "gemm" keyword false-matched every
# Gemma-model kernel name ("_gemma_..." contains "gemm" as a substring of
# "gemma") - fixed with a regex negative-lookahead so "gemm" doesn't match
# when immediately followed by "a".
GEMM_RE = re.compile(r"gemm(?!a)|cutlass|cublas|sgemm|hgemm|matmul|wgrad|igemm|splitk")

BUCKETS = [
    ("attention", ("kernel_unified_attention", "_fwd_kernel", "fwd_grouped_kernel",
                    "paged_attention", "extend_attention", "decode_attention",
                    "fmha", "sdpa", "flash_attn", "flashattention", "reduce_segments")),
    ("moe", ("moe", "fused_moe", "grouped_gemm", "expert_", "routing")),
    ("norm", ("rms_norm", "rmsnorm", "layer_norm", "layernorm")),
    ("rope", ("rope", "rotary")),
    ("elementwise/act", ("silu", "gelu", "act_and_mul", "vectorized_elementwise",
                           "elementwise_kernel", "add_kernel", "mul_kernel",
                           "cast_kernel", "copy_kernel", "fused_add")),
    ("kv_cache_write", ("reshape_and_cache", "store_kvcache", "cache_kernel")),
    ("memcpy/memset", ("memcpy", "memset")),
    ("reduce/comm", ("all_reduce", "nccl", "reduce_kernel")),
    ("indexing/cat", ("index_kernel", "cat_kernel", "gather", "scatter", "embedding")),
    ("sampling", ("sampl", "topk_", "topp_", "multinomial", "argmax")),
]


def bucket_of(name: str) -> str:
    n = name.lower()
    for label, kws in BUCKETS:
        if any(kw in n for kw in kws):
            return label
    if GEMM_RE.search(n):
        return "gemm/matmul"
    return "other"


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d) if isinstance(d, dict) else d


def is_gpu_kernel_event(e, gpu_tids):
    if e.get("ph") != "X":
        return False
    # Verified against the actual traces: real GPU kernel launches have
    # cat=="kernel" (or gpu_memcpy/gpu_memset) EXACTLY. "user_annotation" and
    # "gpu_user_annotation" are NVTX-style wrapper/range markers (e.g. vLLM's
    # "execute_context_..." or SGLang's "scheduler.run_batch", "step[...]")
    # that get projected onto the same GPU stream track/thread as real kernels
    # - counting them double-counts and dwarfs the real per-kernel time.
    return e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")


def analyze(path):
    events = load(path)
    gpu_tids = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            nm = (e.get("args") or {}).get("name", "")
            if "stream" in nm.lower() or "gpu" in nm.lower():
                gpu_tids.add((e.get("pid"), e.get("tid")))

    per_name = defaultdict(lambda: [0.0, 0])  # name -> [total_us, count]
    per_bucket = defaultdict(float)
    total_us = 0.0
    for e in events:
        if not is_gpu_kernel_event(e, gpu_tids):
            continue
        dur = e.get("dur", 0) or 0
        name = e.get("name", "?")
        per_name[name][0] += dur
        per_name[name][1] += 1
        per_bucket[bucket_of(name)] += dur
        total_us += dur

    return total_us, per_bucket, per_name


def main():
    for path in sys.argv[1:]:
        total_us, per_bucket, per_name = analyze(path)
        print(f"\n=== {path} ===")
        print(f"total GPU kernel time: {total_us/1000:.3f} ms  ({sum(c for _,c in per_name.values())} kernel launches)")
        print("\n-- by bucket --")
        for label, us in sorted(per_bucket.items(), key=lambda x: -x[1]):
            pct = 100 * us / total_us if total_us else 0
            print(f"  {label:16s} {us/1000:10.3f} ms  {pct:5.1f}%")
        print("\n-- top 15 kernels --")
        top = sorted(per_name.items(), key=lambda x: -x[1][0])[:15]
        for name, (us, cnt) in top:
            print(f"  {us/1000:9.3f} ms  x{cnt:4d}  {name[:90]}")


if __name__ == "__main__":
    main()
