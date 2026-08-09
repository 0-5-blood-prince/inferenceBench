# Cache metrics: per-request values, never the gauge

`sglang:cache_hit_rate` is a **gauge**. Post-run it reads `0.0` even when reuse
demonstrably happened — observed, and why SPEC §6's validity gate could not be
evaluated. The spec pointed at the wrong metric.

Working per-request signals, both verified on this pod:

**SGLang**, with `--enable-cache-report`, non-streaming:

```
send1 cold   prompt_tokens=1009  details={'cached_tokens': 2,    'image_tokens': 256}
send2 warm   prompt_tokens=1009  details={'cached_tokens': 1008, 'image_tokens': 256}
```

**vLLM**, from `/metrics`, diffed around each request in a serial harness:

```
vllm:prefix_cache_queries_total{engine="0",model_name="..."} 4040.0
vllm:prefix_cache_hits_total{engine="0",model_name="..."}    2976.0
```

Note the `_total` suffix and the labels — grep the live endpoint rather than
assuming names. vLLM does not populate `usage.prompt_tokens_details` at all
(`None` in every probe), so the counter-diff is the only route there.

Normalise both to a **per-request cached-token fraction**. H4a survives this way.

Caveats: `cached_tokens` has a reported bug under parallel sampling where it
reads nonzero against an empty cache — validate against a flushed cache first.
Also note SGLang reports `image_tokens: 256` where our differencing probe
measured 258; the 2-token gap is presumably image delimiter tokens. Our `T` is
verified independently at 991 by client-side tokenization, so this does not
affect the workload geometry.

**Extended in [metrics-instrumentation.md](metrics-instrumentation.md):** the
same distinction — counters get diffed, gauges get read post-only — is now
generalized to KV-cache utilization and preemption counts too, not just the
cache-hit signal.
