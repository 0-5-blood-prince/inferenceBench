# Full reuse — text-only control

Text-only twin of [`full-reuse`](../full-reuse/), built by
[`scripts/build_text_only.py`](../../scripts/build_text_only.py): the shared
image is stripped and replaced 1:1 with an equal-token-count text pad inside
the shared head, so total request length `T` and the cacheable-fraction
structure match the multimodal workload exactly — only the presence of an
image (and therefore the vision-encoder / multimodal-prefill path) differs.
See [P4's "Text-only control" section](../../spec/phases/P4-writeup.md) for
why this control exists and [P5](../../spec/phases/P5-stretch.md) for why it
moved out of stretch and into a first-class run.

Rate: `2.041 req/s` — full-reuse's own `0.5κ` point, matched not re-piloted.
n=3 replicates per engine.

## Results (p50 TTFT, ms)

| Engine | rep1 | rep2 | rep3 | median | multimodal twin (0.5κ, n=1) | gap shrink |
|---|---|---|---|---|---|---|
| vLLM | 218.8 | 216.3 | 219.9 | **218.8** | 226.8 | — |
| SGLang | 292.8 | 287.4 | 290.7 | **290.7** | 341.3 | — |

- Engine gap, multimodal: `(341.3 − 226.8) / 226.8 = 50.5%`
- Engine gap, text-only: `(290.7 − 218.8) / 218.8 = 32.9%`

The gap **shrinks by about a third but does not vanish**. That rules out
"the gap is purely a multimodal-prefill artifact" (it would have gone
to ~noise) but is consistent with the multimodal path contributing part
of it on top of a general scheduling difference that persists on pure text.
Read together with [`partial-reuse-text`](../partial-reuse-text/README.md)
before drawing a conclusion in the P4 mechanism bullet — one workload's
shrink could be noise at n=3, two in the same direction is a real signal.

Raw results: [`results/`](results/). Both engines completed 422/422 requests
every rep (`completion_ratio: 1.0`), well clear of `0.5κ`'s saturation
margin — every rep's `ttft_growth_ratio` sits in 0.97–1.01, nowhere near the
2.0 saturation threshold. vLLM's rep1 is tagged `jit_contaminated` (the
known first-traffic Triton JIT warmup, same as P3's first-cell pattern —
see [`learnings/measurement`](../../learnings/measurement/)); its p50
(218.8ms) is indistinguishable from the clean reps (216.3/219.9ms), so the
contamination is confined to the discarded warmup window. No SGLang rep
tripped the JIT check.
