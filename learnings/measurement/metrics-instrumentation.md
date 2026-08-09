# Metrics instrumentation, aligned to NVIDIA's LLM benchmarking fundamentals

Reference: [NVIDIA — LLM Benchmarking: Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/).
That article names the standard metric set this field measures by; this module
records what we already had against it, what was missing, and what was added
to `scripts/client.py`, `scripts/pilot.py`, and `scripts/p1_gates.py` as a
result.

## The standard set, and where we stood

| Metric | Definition (NVIDIA) | Had it? |
|---|---|---|
| TTFT | time to first token | yes — p50/p90/p99/mean |
| ITL / TPOT | mean gap between consecutive decode tokens, TTFT excluded | yes — pooled p50/p99/mean |
| e2e latency | TTFT + generation time | yes |
| TPS (system) | `Total_output_tokens / (Ty − Tx)` | **approximated only** — via streamed SSE chunk *count*, not actual token count |
| TPS per user | per-request `output_tokens / e2e_latency`, averaged | missing |
| RPS | completed requests / wall time | missing (present implicitly, never surfaced) |
| ISL | input sequence length | captured per-request, never summarized |
| OSL | output sequence length | captured per-request, never summarized |
| KV cache utilization | fraction of KV capacity in use | missing entirely |

The chunk-count TPS approximation was the most important gap: an SSE `delta`
event is *usually* one token, but that's an implementation detail of the
streaming protocol, not a guarantee — summing the engine's own
`usage.completion_tokens` is what the blog's formula actually asks for, and is
authoritative rather than inferred.

## What was added

**`scripts/client.py`** (per-request, then aggregated in the run summary):

- `tokens_per_second` — `sum(completion_tokens) / wall`, replacing the chunk-count
  proxy as the primary throughput number (chunk-based figure kept alongside as
  `output_chunk_throughput`, a cheap sanity check, not the metric of record).
- `tokens_per_second_per_user` — mean of each request's own
  `completion_tokens / e2e_s`. The blog notes this "asymptotically approaches
  1/ITL as OSL increases" — a useful cross-check between the two numbers.
- `requests_per_second` — `completed / wall`, request-level throughput
  independent of token counts.
- `isl_tokens` / `osl_tokens` — p50/p90/p99/mean summaries of `prompt_tokens`
  and `completion_tokens` across the run, not just buried in per-request rows.
- `image_tokens` / `text_tokens` — per request and summarized. See below for
  why this needed a workaround rather than a direct read.

**Image vs. text token split — an engine asymmetry, worked around with a
workload constant.** SGLang exposes this natively:
`usage.prompt_tokens_details.image_tokens`, when started with
`--enable-cache-report` (confirmed: `256` on every probe). **vLLM never
populates `prompt_tokens_details` at all** — it reads `None` on every request,
every probe, no flag changes that. Reading the split from the API response
is therefore not viable uniformly across engines.

The fix: image token count is a **workload constant**, not a per-request fact
— one fixed image resolution means one fixed ViT output size (SPEC §5;
measured **258** at every resolution tried during the P0 probe, regardless of
input size, since this model's vision tower isn't dynamic-resolution). That
constant is already recorded once, in `workloads/manifest.json`'s
`geometry.image_tokens`. `client.py` now accepts `--image-tokens` (defaulting,
in `pilot.py`, to a read of that manifest field) and computes
`text_tokens = prompt_tokens - image_tokens` uniformly for both engines,
preferring the engine's own reported value when present (SGLang) and falling
back to the constant when absent (vLLM). This makes the split available and
*comparable* across engines despite the underlying API asymmetry, rather than
silently vLLM-only.

**`scripts/p1_gates.py`** — `COUNTER_CANDIDATES` split into true counters
(diff pre/post, monotonic) and a new `GAUGE_CANDIDATES` list (point-in-time,
diffing is meaningless — read post-only). This is the same distinction
[cache-metrics.md](cache-metrics.md) already drew for cache-hit reporting,
now generalized to KV utilization and preemptions. Names confirmed present on
the pinned versions via live probes on this pod:

- **vLLM:** `vllm:gpu_cache_usage_perc` (gauge — fraction of KV blocks in use),
  `vllm:num_preemptions_total` (counter — the classic vLLM saturation tell: a
  request evicted mid-generation to free KV capacity for another).
- **SGLang:** `sglang:token_usage`, `sglang:num_used_tokens`,
  `sglang:kv_available_tokens`, `sglang:kv_evictable_tokens`,
  `sglang:max_total_num_tokens` (all gauges), `sglang:num_preemptions_total`
  (counter).

`scrape()` (in `p1_gates.py`, imported by `pilot.py` rather than duplicated —
one source of truth for metric names, shared by both P1's gate dumps and P2's
pilot points) now returns both families in one snapshot.

**`scripts/pilot.py`** scrapes pre- and post-point, and surfaces both the
counter deltas (how much did *this specific point* add — e.g. how many
preemptions it caused) and the gauge readings (KV state right now) in its
printed output and saved JSON. The point of adding this here rather than only
at the matrix stage: it turns "this point saturated" into "this point
saturated *because* — full KV, thrashing preemptions, or plain queueing with
capacity to spare" — a causal trail, not just a symptom, and it's available
immediately during the pilot sweep that first found the knee rather than only
retrospectively at P3.
