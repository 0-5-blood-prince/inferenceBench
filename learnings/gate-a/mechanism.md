# Why cache-hit and fresh-prefill outputs diverge: reduction order, one ULP, a forked sentence

Start with the irreducible fact — three numbers in bf16, nothing else involved:

```
x = 0.00390625            (half a ULP of 1.0)
(1 + x) + x  = 1.00000000      bits ...10000000
1 + (x + x)  = 1.00781250      bits ...10000001
```

Same three values, two groupings, results differ in the last mantissa bit — one
full ULP. Group the two small terms last and each is individually too small to
survive the round back to 1.0; group them first and they combine into something
that does. That is the entire root cause. Everything below is this fact at
scale.

Now put it inside an attention reduction. Same query, keys, values and softmax
weights every trial; the only thing varied is whether the value-reduction
sweeps all 16 keys in one pass (fresh prefill) or runs `[12 cached] + [4 new]`
as two partial sums that then merge (cache hit). Across 8000 independent
next-token positions, greedy argmax disagreed between the two paths on 28 —
**0.35%**. One flip in full:

```
attn out dim 3:  fresh -0.511719   cache -0.507812     (differ in the last bit)
fresh logits:    tok 31 = 1.28121  >  tok 14 = 1.28097   margin +0.00024 -> 31
cache logits:    tok 14 = 1.27952  >  tok 31 = 1.27944   margin +0.00007 -> 14
```

The attention output shifted by a ULP in a few dimensions, and that pushed the
token-31-vs-14 logit gap from +0.00024 to −0.00007 — across zero. Fresh emits
31, cache emits 14, and from that position a cold and a warm stream read as
different sentences. Every later token inherits a different context, so they
never realign.

**The flip hazard is memoryless, so the gate's verdict is a function of length.**
At a per-token rate p, P(at least one flip in n tokens) = 1 − (1 − p)ⁿ. At the
toy's p = 0.0035 that is ~20% over 64 tokens and ~36% over 128. This is why a
boolean PASS/FAIL is the wrong output: **record the divergence index**, because
"matches through 64" and "matches through 512" are different grades, not the
same PASS.

Two caveats before porting that number. The toy accumulated in bf16 to make the
effect legible in a handful of terms; real kernels accumulate in fp32 through
mixed-precision reduction trees, so per-step error is smaller — but sequences
are far longer and it compounds, which is why on a real model first divergence
lands around token 100 rather than immediately. And the rate is model-,
precision- and distribution-specific: greedy is the worst case, and sharply
peaked distributions flip far less often. Measure the real rate per engine.

## Our measurements, and why the first read was wrong

First Gate A run, 36-token generations:

| | vLLM | SGLang |
|---|---|---|
| cold vs warm | identical | diverges at token 21 |
| verdict | PASS | FAIL |

That asymmetry did not survive. Re-run at `max_tokens=512`:

| | vLLM | SGLang |
|---|---|---|
| cold vs warm | **diverges at token 55** (of 461/464) | **diverges at token 21** (of 512/512) |
| warm vs warm2 | identical | identical |
| 8 identical concurrent requests | **2 unique outputs** | 1 unique output |
| batched vs serial | identical | diverges at token 21 |
| verdict | **FAIL** | **FAIL** |

Neither engine uses batch-invariant kernels by default. vLLM's PASS at 36
tokens was simply "we did not generate far enough to reach its first flip".
Both engines are self-consistent *within* a mode (`warm == warm2`) and disagree
*across* modes — the reduction-order signature, not noise. Confirmed from the
other side too: with SGLang's radix cache disabled, three sends were byte-identical.

What survives as a real difference is the **divergence index** (55 vs 21) and
vLLM's batch-composition sensitivity (2 unique outputs across 8 identical
concurrent requests, where SGLang gave 1).

Also checked and rejected as a fix: `--disable-chunked-prefix-cache` on SGLang
does not restore equivalence — still diverges at the same character.

## The divergence event is not reproducible run-to-run

A second, independent `gate_a_extended.py` pass (this time through `run.sh
p1()` proper, not a hand-rolled probe) gave a **different** result for SGLang's
serial cold/warm comparison than the first pass above:

| | first pass (standalone probe) | second pass (`run.sh p1`) |
|---|---|---|
| SGLang cold vs warm, serial | diverges at token ~21 | **identical**, 512/512 |
| SGLang batched (8x) vs serial warm | not measured | **diverges at token 21** |
| SGLang unique outputs among 8 concurrent | not measured | 1 (self-consistent) |
| vLLM cold vs warm, serial | diverges at token 55 | **identical** |
| vLLM batched (8x) vs serial warm | not measured | **identical** |
| vLLM unique outputs among 8 concurrent | not measured | 2 |

Both engines' *serial* result flipped from FAIL to PASS between two otherwise
identical runs. This is exactly what the mechanism above predicts and it is the
strongest evidence yet that Gate A was the wrong kind of check: the divergence
is a knife-edge numerical tie, and which side of the tie a given run lands on
depends on incidental conditions this harness does not (and arguably should not
try to) control — request timing, whatever else the allocator or scheduler is
doing, exact batch composition at the moment of the fork. **A gate whose
verdict is not reproducible across two runs of the same command cannot be a
pre-registration blocker.**

The one thing that *did* reproduce: SGLang's 8-way concurrent batch diverging
from the serial warm run at the same token index (21) in both passes. Batch
composition looks like the more stable trigger here than cold-vs-warm alone,
consistent with batch-invariance research naming batch composition as its own
axis separate from cache-hit-vs-fresh.

One instrumentation note for whoever extends `gate_a_extended.py`: its
`VERDICT` field is computed **only** from the serial `cold_vs_warm` check, not
from the concurrency result. In one run that produced `VERDICT: PASS` for both
engines despite vLLM showing 2 unique outputs among 8 identical concurrent
requests and SGLang's batch diverging from its own serial run. Read the full
JSON (`gates/<engine>/gate_a_extended.json`), not just the printed verdict line.

Next: [why this mechanism means Gate A doesn't belong in a latency comparison at all »](vs-latency.md)
