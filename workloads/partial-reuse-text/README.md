# Partial reuse — text-only control

Text-only twin of [`partial-reuse`](../partial-reuse/), built the same way as
[`full-reuse-text`](../full-reuse-text/README.md) — see that README and
[P4's "Text-only control" section](../../spec/phases/P4-writeup.md) for the
construction and rationale.

Rate: `1.134 req/s` — partial-reuse's own `0.5κ` point. n=3 replicates per
engine.

## Results (p50 TTFT, ms)

| Engine | rep1 | rep2 | rep3 | median | multimodal twin (0.5κ, n=1) | gap shrink |
|---|---|---|---|---|---|---|
| vLLM | 316.6 | 317.1 | 314.7 | **316.6** | 328.7 | — |
| SGLang | 504.3 | 508.1 | 507.3 | **507.3** | 561.0 | — |

- Engine gap, multimodal: `(561.0 − 328.7) / 328.7 = 70.7%`
- Engine gap, text-only: `(507.3 − 316.6) / 316.6 = 60.2%`

Same direction and similar magnitude of shrink as
[`full-reuse-text`](../full-reuse-text/README.md) (50.5%→32.9% there,
70.7%→60.2% here — both drop by roughly a third, neither drops to noise).
Two workloads shrinking the same way is stronger evidence than either alone:
**the gap is partly, not wholly, multimodal-prefill-specific.** Write this
into the P4 mechanism bullet as a split finding, not a single winner between
the cache-effect and CUDA-graph-capture-asymmetry hypotheses — both
contribute, and the text-only control is what makes that separable.

Raw results: [`results/`](results/). Both engines completed 208/208 requests
every rep (`completion_ratio: 1.0`).
