# I benchmarked two LLM serving engines. Then I spent two weeks proving myself wrong.

*How a simple question — "does prefix caching change which inference engine
wins?" — turned into a root-cause investigation where the leading hypothesis got
falsified twice before the profiler settled it.*

---

## The question

If you serve a large language model in production, you pick an inference engine.
Two of the best open-source ones are **vLLM** and **SGLang**, and they take
genuinely different approaches to one thing that matters a lot: **prefix
caching** — reusing the computation for prompt prefixes that show up again and
again (system prompts, few-shot examples, a shared image in a multimodal chat).

vLLM matches cached prefixes in 16-token blocks. SGLang's RadixAttention matches
at single-token granularity with a radix tree. The folklore says SGLang's finer
matching should win when reuse is high. So I asked a clean question:

> **Does the amount of prefix reuse change which engine has lower latency?**

One model — Google's `gemma-4-31B-it`, a 31-billion-parameter multimodal model —
on a single A100 80GB. I wrote the whole experiment design down *first*: the
hypotheses, the workloads, the pass/fail thresholds, frozen in a spec before I
measured anything. (This turned out to matter more than I expected.)

## The answer (and the first surprise)

**Prefix reuse doesn't change the winner.** vLLM had lower time-to-first-token at
every clean comparison point — by **43% to 71%** — no matter how much of the
prompt was cached. My pre-registered guess had been the *opposite*: I expected
SGLang's finer cache matching to pull ahead under heavy reuse. It never did.

What reuse *does* change is how big vLLM's lead is — it shrinks as more of the
prompt is cached, but it never closes. And one number was suspiciously clean:
**inter-token latency — the decode speed once generation starts — was identical
between the engines, to within a millisecond.** Same model, same decode kernels,
same per-token cost.

So the entire gap was in the *first* token. The prefill. Which raised the real
question: **why?**

## Ruling things out

Before chasing exotic explanations, I killed the obvious ones with a controlled
setup. If you send requests one at a time (concurrency 1), there's no queue —
time-to-first-token is basically just prefill compute. Measured that way:

- SGLang's prefill was **85% slower** on the cached workload, **49% slower** cold.
- But the *queue* component was symmetric — within 6 ms between engines.

That kills "SGLang's fancy cache-aware scheduler adds overhead." It's not
scheduling. And with caching turned fully **off** on both engines, SGLang was
*still* 43–49% slower — so it's not caching either. Decode was identical, so it's
not decode. The gap was pure prefill compute, sitting in the forward pass itself.

Fine. So it's a slower kernel somewhere. Easy, right?

## Wrong turn #1: "vLLM uses FlashAttention, SGLang uses Triton"

The obvious story: vLLM runs the fast hand-tuned FlashAttention kernel, SGLang
falls back to a slower Triton one. I was about to write that down when I checked
the actual engine logs.

**Both engines run a Triton attention backend on the A100.** FlashAttention gets
rejected for this model — its head dimension (256) plus the bidirectional
attention that multimodal image tokens need. And the one fast fused alternative,
NVIDIA's `trtllm_mha`, requires a *Blackwell* GPU — it's rejected on both the
A100 *and* an H100 I spun up to check. So it's Triton versus Triton. Not a
different algorithm. My clean explanation was just wrong.

## Wrong turn #2: the patch that fixed everything and changed nothing

If both use Triton, maybe SGLang's Triton kernel is just worse. I benchmarked the
two attention kernels head-to-head in isolation, and there it was: SGLang's
**sliding-window** attention kernel was **~8× slower** than vLLM's. Reading the
source showed exactly why — SGLang does a per-tile masking check and a
data-dependent branch that breaks the GPU's ability to pipeline memory loads
against math, and it never shrinks its loop to the actual window.

I wrote a minimal patch. Two small changes. The kernel got **10.9× faster**, with
**bitwise-identical output**. I was ready to call it.

Then I installed the patched kernel into the live server and re-ran the actual
benchmark. **It changed the end-to-end latency by zero percent.** 721 milliseconds
before, 721 after. I marker-verified that the server had actually loaded my
patched code. It had. A 10.9× faster kernel moved the real number *not at all.*

That's the moment the investigation got interesting, because it meant my
kernel-microbenchmark story — the one I was most confident about — was wrong at
the level that matters. A faster kernel that doesn't make serving faster isn't
the bottleneck.

## The profiler settles it

I stopped theorizing and captured an op-level GPU profile of each engine's
prefill forward pass — every CUDA kernel, timed. To make it a fair fight I ran
*both* engines in eager mode (no compilation tricks), after a separate experiment
confirmed compilation wasn't the difference either (vLLM stays ~1.9× faster even
with all its compilation turned off).

The profile was blunt:

| | vLLM | SGLang |
|---|---|---|
| Matrix multiplies (the MLP bulk) | 253 ms | 240 ms — *a wash* |
| **Attention** | **49 ms** | **298 ms** |

The matrix multiplies — the biggest chunk of compute — were a *tie* (SGLang's
were even marginally faster). The entire ~240 ms gap was attention. On text,
attention wasn't a *contributor* to the gap. It **was** the gap.

Which brought back the contradiction: if attention is the whole gap, why didn't
my attention patch help? I went back to the patch diff and found the answer in a
single line. My fix was gated to only run when custom masking was *off*. Gemma-4's
multimodal requests turn custom masking *on* (that's how it does bidirectional
attention over image tokens). **So on the multimodal workload, my fix never
actually ran.** On pure *text*, where masking is off, it runs fully — and a
confirmatory re-run there closed **88% of the gap at realistic prompt lengths,
98% at longer ones.** Exactly what the profiler predicted.

## Where it honestly lands

- **On pure text:** the vLLM-vs-SGLang first-token gap is the sliding-window
  attention kernel. I can prove it, and a small fix closes almost all of it.
- **On the actual multimodal workload I set out to measure:** that same kernel is
  *not* the end-to-end cause — my fix is gated off there and moves the number
  zero. The real multimodal root cause is **still open.**

I could have shipped the tidy version — "found an 8× kernel slowdown, patched it,
huge win." It would have been wrong for the workload that motivated the whole
study. The honest version is less clean and more true, and I'd rather have that.

## What I actually take away

The benchmark result is useful: for this model, vLLM leads first-token latency
43–71% and prefix caching doesn't change that. I cross-checked it with SGLang's
*own* benchmarking tool to make sure it wasn't an artifact of my harness (+78%,
same direction, same identical-decode signature).

But the part I'd point to is the process. Pre-registering the design meant I
couldn't quietly move the goalposts when results surprised me. Auditing my own
finished data turned up four defects I'd otherwise have shipped. And the discipline
that mattered most was being willing to run the experiment that could embarrass me
— installing the patch and watching it do nothing — instead of stopping at the
result I wanted. Two of my most confident conclusions got falsified by my own
follow-up tests. That's not the investigation going wrong. That's the only reason
I trust where it ended up.

---

*Full methodology, corrected results, and the mechanism trail are in the repo:
[README](README.md), [technical writeup](WRITEUP.md), and the topical deep-dives
in [LEARNINGS.md](LEARNINGS.md).*
