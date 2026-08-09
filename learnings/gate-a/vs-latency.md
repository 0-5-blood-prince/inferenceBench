# Why Gate A does not belong in a latency comparison

Prerequisite: [the reduction-order mechanism and why the divergence isn't reproducible »](mechanism.md)

Strict Gate A is a **within-engine** invariant: "this engine's cache is
output-transparent." Correct for a within-engine claim. A cross-engine
comparison does not need it, for a blunt reason: **the two engines never
produce identical tokens anyway**, even both cold — different attention
backends, reduction orders and fp accumulation. Strict Gate A never bought
cross-engine output-comparability; it only guaranteed each engine's warm path
matched its own cold path.

And nothing in a latency claim reads token values:

- **TTFT** is fixed by the time the first token is produced — position 0, before
  any fork can occur, since a fork needs prior context to diverge on. The timing
  would not change even if the first token value differed.
- **Cached fraction / H4a** is pure accounting; content is irrelevant.
- **Throughput / per-token cost** needs matched token *counts* and matched work,
  not matched content.

So a strict output-equivalence gate tests a property the dependent variable
structurally cannot see — while a short window turns a continuous quantity (the
divergence index) into a binary that reports an artifact of window length as an
engine difference.

## What replaces it: Gate R

Three latency-relevant conditions, applied symmetrically:

1. **The warm path is a genuine cache hit** — verified by mechanism, not output:
   cached-token counters increment *and* latency drops. A run that is fast for
   another reason (e.g. it landed in a smaller batch) is a real number
   mislabelled as cache savings. This is a positive check, not an equivalence
   check.
2. **Token count is matched** across warm/cold and across engines. This is the
   one place output divergence *can* bite a latency number sideways: if a
   divergent token happens to be EOS, the run stops early and you compare a
   40-token run against a 128-token one and call the delta "caching". Pin the
   budget and disable early stopping. Once length is pinned, content can wander
   freely at no cost. See
   [../measurement/output-budget.md](../measurement/output-budget.md) for how
   this is actually enforced and a measurement error found along the way.
3. **The work being timed is the same shape** — same prompt, same image token
   geometry, same batch context. Always required; unrelated to Gate A.

Plus, non-blocking: **record the divergence index per engine**. Free to collect,
and it says whether the engines are running comparably deterministic paths.

**Where this may not be loosened:** any claim that depends on output content —
"the cache is transparent", outputs unchanged, full-generation quality. There
per-engine determinism is load-bearing, and the fix is not a looser gate but
deterministic mode on *both* engines — see
[why that mode isn't available on the pinned versions »](deterministic-mode.md).
