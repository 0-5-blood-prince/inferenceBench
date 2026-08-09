# Learnings

Operational knowledge gathered while running this study. Not part of the frozen
pre-registration in `spec/` — this is what we found out by doing it, including
where the runbook itself was wrong.

The headline: **Gate A was dropped as a blocker.** It tested a property that a
latency comparison structurally cannot see, and its short-window form
manufactured a false engine asymmetry. Sections 1–4 are why.

---

## 1. The mechanism: reduction order, one ULP, and a forked sentence

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

## 2. Our measurements, and why the first read was wrong

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

## 3. Why Gate A does not belong in a latency comparison

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

### What replaces it

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
   freely at no cost.
3. **The work being timed is the same shape** — same prompt, same image token
   geometry, same batch context. Always required; unrelated to Gate A.

Plus, non-blocking: **record the divergence index per engine**. Free to collect,
and it says whether the engines are running comparably deterministic paths.

**Where this may not be loosened:** any claim that depends on output content —
"the cache is transparent", outputs unchanged, full-generation quality. There
per-engine determinism is load-bearing, and the fix is not a looser gate but
deterministic mode on *both* engines. Which we cannot do here (§5).

## 4. Pinning the output budget — and a measurement error worth recording

**Corrected.** An earlier version of this document claimed vLLM dishonoured
`ignore_eos`, citing 461 and 464 tokens against a 512 budget. That was wrong,
and the error was in our instrument, not the engine.

`gate_a_extended.py` reported generated length by re-tokenizing the *decoded
text* client-side with `AutoTokenizer`. Detokenize → retokenize does not
round-trip to the same token count, so the number never matched what the server
actually produced. SGLang's apparent 512/512 was coincidence, not compliance.

Measured properly, from the server's own `usage.completion_tokens`:

| request (vLLM, 512 budget) | completion_tokens | finish_reason |
|---|---|---|
| `ignore_eos` only | **512** | length |
| `ignore_eos` + `min_tokens` + `stop_token_ids: []` | **512** | length |

and at a 64 budget:

| request | completion_tokens | finish_reason |
|---|---|---|
| baseline, no flag | 27 | stop |
| `ignore_eos` | 64 | length |
| `min_tokens` | 64 | length |
| both | 64 | length |

vLLM honours the budget. **Lesson: read generated length from the server's usage
block, never from client-side re-tokenization.** Any harness that measures
"how much work did the engine do" from decoded text is measuring its own
tokenizer.

The belt-and-suspenders form is still what every timed request sends, as
defence in depth rather than a fix for a known bug:

```json
{
  "max_tokens": 512,
  "min_tokens": 512,
  "ignore_eos": true,
  "stop_token_ids": []
}
```

`stop_token_ids: []` clears any stop tokens a chat template introduces beyond
the tokenizer's `eos_token_id`, and `min_tokens` puts a floor under generation.
Neither changed the outcome here, but the failure mode they guard is real and
silent: an early stop shortens a run, and a shorter run is a faster run, so an
unpinned budget turns a stopping difference into a latency difference and
attributes it to caching. Verify `completion_tokens == budget` per run instead
of trusting the flags — the client now tags a run `budget_unpinned` if not.

**Also affected:** the divergence indices in §2 (token 55 / token 21) were
computed on the same client-retokenized text. They locate divergence
approximately, which is enough for a diagnostic, but they are not exact server
token positions.

## 5. Deterministic mode is not available on the pinned versions

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

## 6. Cache metrics: per-request values, never the gauge

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

## 7. CUDA graphs and allocator settings are a benchmark confound

**Why `cudaFree` blocks.** `cudaMalloc` reserves a virtual address range and
installs GPU page-table mappings. `cudaFree` wants to tear that down, but GPU
execution is asynchronous and the driver does not track per-kernel address
dependencies at fine grain. Its only safe move is to block the calling thread
and drain all in-flight work on the **entire device**, then unmap. That
device-wide drain is why frameworks build userspace caching allocators that
never return blocks to the driver on the hot path.

**Stream-ordered allocation** (`cudaMallocAsync`/`cudaFreeAsync`, CUDA 11.2+)
productises that: frees become stream-ordered rather than host-blocking, memory
returns to a `cudaMemPool_t`, and cross-stream reuse inserts a
`cudaStreamWaitEvent` instead of a host stall. Pools carry a release threshold.
PyTorch exposes the backend via `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`,
though for serving the more useful knob is `expandable_segments:True`, letting a
segment grow in place instead of fragmenting — which matters when KV
allocations vary in size.

**Launch overhead.** Each `cudaLaunchKernel` costs the host a few microseconds.
Irrelevant for a long convolution; fatal for **decode**, where one token is a
full forward pass of hundreds of tiny kernels each running microseconds. Launch
cost becomes comparable to kernel runtime, the CPU cannot refill the queue, and
the GPU idles between kernels — **CPU-launch-bound**, not compute-bound.

**CUDA graphs** collapse hundreds of launches into one `cudaGraphLaunch`. The
catch: **replay bakes in captured pointers and shapes**. Hence bucketed batch
sizes with padding up to the nearest captured bucket; hence a private memory
pool per graph so the caching allocator cannot recycle those addresses; hence
captured graphs consuming memory that competes directly with the KV budget.
Prefill is long, compute-bound and variable, so it is usually left eager.
Current direction is **piecewise capture** of dense static regions with
attention outside, often with `torch.compile`.

**Implication — a confound not yet controlled here:**

> A vLLM-vs-SGLang decode number is meaningless unless graph capture and
> `torch.compile` settings are held identical across both engines.

If one runs graphed decode and the other eager, or they bucket batch sizes
differently, the benchmark measures launch strategy, not the engine. Control and
record on both sides: cuda-graph enable / `enforce_eager`, the captured
batch-size set, memory fraction (already pinned equal), compile mode, and
**whether each engine captured graphs at all**.

Observed for vLLM at startup: `Capturing CUDA graphs (PIECEWISE)` 0/51 and
`(FULL)` 0/35, finishing in 14 s and costing 0.94 GiB. SGLang's capture
behaviour must be recorded the same way. Note also that our vision tower injects
dynamic control flow that fights naive capture — exactly where the engines'
piecewise strategies diverge.

## 8. Environment findings from this pod

- **Runpod A100 volumes are network storage (MooseFS).** Measured: 800 small
  file writes took 184 ms on local NVMe and 15.6 s on the volume — **85x**;
  sequential only 4.7x. Venvs on local disk, weights on the volume. Turned a
  45-minute bootstrap into ~4 minutes.
- **No A100 with a CUDA 13 driver was available** (best 570 / CUDA 12.8;
  requiring 13.0 returned "no instances available"), but both engines pull torch
  cu130. Fixed with `cuda-compat-13-0` (580.178.04) ahead of the system libcuda
  — the supported forward-compat path on datacenter GPUs.
- **GPU clocks cannot be locked from inside the container**, so SPEC's
  thermal-drift control is unavailable. Mitigation: P3's engine-order
  alternation and the variance-duplicate cell.
- **`ninja` is required but undeclared** by both engines — torch.compile shells
  out to it, so without it the server dies *after* loading 60 GB of weights.
- **SGLang serves no Prometheus endpoint without `--enable-metrics`**, and no
  per-request cache fields without `--enable-cache-report`.
- **Gemma 4 uses a fixed 258 image tokens at every resolution** (448², 896²,
  1344²) — not a dynamic-resolution ViT, so SPEC §5's resolution-fixing
  rationale does not apply. Resolution was chosen instead to minimise
  client-side base64 payload.
- **SGLang reports `max_total_num_tokens = 16391`** — only ~16 requests of KV at
  our T=991, which will shape the saturation knee and make eviction bite early.
