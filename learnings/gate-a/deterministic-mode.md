# Deterministic mode is not available on the pinned versions

SGLang ships `--enable-deterministic-inference` (batch-invariant operators,
compatible with chunked prefill, CUDA graphs, radix cache, on FlashInfer / FA3 /
Triton backends), and an in-tree equivalent of Gate A:
`test_deterministic --test-mode radix_cache`, expecting `Unique samples: 1`.

**But `sglang 0.5.16` does not have the flag** — it is absent from `--help`,
which lists only the attention-backend options. Using it would mean changing the
pinned version, which is itself a controlled variable.

Recorded for whoever revisits this, three caveats that would hit this setup:

- **Endpoint.** Deterministic/seeded inference has been reported not to engage on
  the OpenAI-compatible endpoint while native `/generate` works. Our path is the
  chat endpoint — verify, or measure through `/generate`.
- **Block-boundary corruption.** Known KV-cache corruption when a request's
  `prefix_len` exactly equals the KV block size (64 tokens). Our shared head is
  `S + I = 39 + 258 = 297` and Partial reuse sweeps `E` over `0..495`, so
  requests *will* land on 64-token boundaries.
- **Cost.** ~1.6x slower unoptimized, ~34% overhead with CUDA graphs. Enabling
  it on one engine only would make TTFT non-comparable. Both or neither.

Related: [why Gate A doesn't belong in a latency comparison, and what replaces it »](vs-latency.md)
